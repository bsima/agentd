//! GC zoom (t-1609, docs/GC_ZOOM.md): exact recall of any message GC
//! evicted from the model's prompt.
//!
//! agentd's GC is non-destructive: the machine env's `history` var holds
//! the complete append-only log, and GC only shapes the per-Infer copy of
//! it (docs/GC_ZOOM.md "agentd today"). Zoom is therefore a pure read over
//! `history`: the agent loop's `zoom` arm evaluates [`crate::ir::Expr::Zoom`]
//! — no effect, no replay record, deterministic under replay.
//!
//! **Addresses** are positions in the append-only `history`, written `#N`
//! (`#0` is the first message). A position names exactly one message of any
//! kind — tool result, assistant turn, user turn — and never changes,
//! because history only grows. A range `#A-#B` reads consecutive messages.
//! A tool-call id is accepted as an alias for the result it names; ids come
//! verbatim from providers and are not unique, so a duplicate resolves to
//! the most recent match and the result says so, and the empty id is never
//! a handle (the OpenAI streaming path can leave ids empty).
//!
//! The argument schema is `{address: string, page?: integer}`. `address`
//! is a string so that a future summary strategy (t-1612) can add its own
//! address forms (tree nodes) without breaking existing calls.
//!
//! **Markers.** With zoom on, GC eviction markers carry the `#N` address of
//! every message they stand for. Strategies never learn that zoom exists:
//! the interpreter records an [`ZoomBook`] (message id to address) on
//! `GcState` for Infer calls that offer the `zoom` tool, and the shared
//! marker renderers in `gc.rs` read it. Without a book the marker text is
//! byte-identical to the zoom-free runtime.

use crate::op::ChatMessage;
use crate::provider::{ToolFunctionSpec, ToolSpec};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// The zoom tool's name. Reserved for the built-in arm
/// ([`crate::tool::RESERVED_TOOL_NAMES`]).
pub const ZOOM_TOOL: &str = "zoom";

/// Smallest page a zoom returns, in chars. Below this a page header and
/// continuation line would dominate the page.
pub const MIN_PAGE_CHARS: usize = 1024;

/// Largest page a zoom returns, in chars: 16 KiB, the `recall` tool's
/// retrieval cap (`max_bytes: 16 * 1024` in the agent loop), about 5.5k
/// tokens under the GC estimator (chars / 3).
pub const MAX_PAGE_CHARS: usize = 16 * 1024;

/// One page may cost at most this fraction of the context budget. A zoomed
/// page re-enters the window and is write-barrier hot; an eighth of the
/// budget leaves room for the task, the rest of the working set, and the
/// 15% headroom above the default 0.85 GC trigger, so one zoom cannot by
/// itself force a deep collection.
pub const PAGE_BUDGET_DIVISOR: usize = 8;

/// Program-shaping zoom configuration (part of the agent-loop program when
/// zoom is on, so it is covered by the program hash).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoomOptions {
    /// Page size in chars.
    pub page_chars: usize,
}

impl ZoomOptions {
    /// Page size for a context budget (tokens): an eighth of the budget,
    /// converted at the GC estimator's 3 chars per token, clamped to
    /// [`MIN_PAGE_CHARS`]..=[`MAX_PAGE_CHARS`].
    pub fn for_context_budget(context_budget_tokens: usize) -> Self {
        let page_chars = (context_budget_tokens / PAGE_BUDGET_DIVISOR)
            .saturating_mul(3)
            .clamp(MIN_PAGE_CHARS, MAX_PAGE_CHARS);
        Self { page_chars }
    }
}

/// Provider-facing spec for `zoom`. The description is shipped guidance
/// text ([`crate::guidance::ZOOM_TOOL_DESCRIPTION`]).
pub fn zoom_tool_spec() -> ToolSpec {
    ToolSpec {
        kind: "function".into(),
        function: ToolFunctionSpec {
            name: ZOOM_TOOL.into(),
            description: crate::guidance::ZOOM_TOOL_DESCRIPTION.into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "a marker's address: `#N` for one message, `#A-#B` for a range, or a tool-call id"
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "page number for long messages (default 1)"
                    }
                },
                "required": ["address"]
            }),
        },
    }
}

/// Render a history position as an address.
pub fn address(position: usize) -> String {
    format!("#{position}")
}

/// Render the smallest address span covering `positions`: `#N` for one
/// position, `#A-#B` otherwise. `None` for an empty set.
pub fn address_span(positions: impl IntoIterator<Item = usize>) -> Option<String> {
    let mut min = None::<usize>;
    let mut max = None::<usize>;
    for position in positions {
        min = Some(min.map_or(position, |m| m.min(position)));
        max = Some(max.map_or(position, |m| m.max(position)));
    }
    match (min, max) {
        (Some(a), Some(b)) if a == b => Some(address(a)),
        (Some(a), Some(b)) => Some(format!("{}-{}", address(a), address(b))),
        _ => None,
    }
}

/// Every `#N` address written in `text` (marker lines from an earlier
/// collection carry them; a later collection that absorbs such a marker
/// keeps its addresses).
pub fn addresses_in(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'#' {
            let start = index + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                if let Ok(position) = text[start..end].parse() {
                    out.push(position);
                }
            }
            index = end.max(index + 1);
        } else {
            index += 1;
        }
    }
    out
}

/// The addresses a zoom-on GC marker tells the model to zoom: every
/// `zoom #N` / `zoom #A-#B` target in `text`. A later collection that
/// drops such a marker keeps these in its own span. Only the zoom targets
/// count, so a `#` inside a quoted recall query never widens a span.
pub fn zoom_targets_in(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(index) = rest.find("zoom #") {
        let target = &rest[index + "zoom ".len()..];
        let end = target
            .find(|c: char| c.is_whitespace() || c == ';' || c == ')' || c == ']' || c == ',')
            .unwrap_or(target.len());
        out.extend(addresses_in(&target[..end]));
        rest = &target[end..];
    }
    out
}

/// The address book GC markers read (t-1609): each prompt message's
/// position in the append-only history, keyed by the decoded message id,
/// plus each tool-call id's most recent result position. Built by the
/// interpreter for Infer calls that offer the `zoom` tool and stored on
/// `GcState`; absent otherwise, which is what keeps zoom-off markers
/// byte-identical. Runtime-only, rebuilt on every such Infer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ZoomBook {
    positions: HashMap<Uuid, usize>,
    results: HashMap<String, usize>,
}

impl ZoomBook {
    /// Book for a prompt decoded from history: message `i` is `#i`. Empty
    /// tool-call ids are never recorded as handles.
    pub fn from_history(history: &[ChatMessage]) -> Self {
        let mut book = Self::default();
        for (position, message) in history.iter().enumerate() {
            book.positions.entry(message.id).or_insert(position);
            if message.role == "tool" {
                if let Some(id) = message.tool_call_id.as_deref().filter(|id| !id.is_empty()) {
                    book.results.insert(id.to_string(), position);
                }
            }
        }
        book
    }

    /// The history position of a prompt message, when it came from history
    /// (hydrated or GC-synthesized messages have none).
    pub fn position(&self, id: &Uuid) -> Option<usize> {
        self.positions.get(id).copied()
    }

    /// The most recent result position for a non-empty tool-call id.
    pub fn result_position(&self, call_id: &str) -> Option<usize> {
        if call_id.is_empty() {
            return None;
        }
        self.results.get(call_id).copied()
    }
}

/// A parsed `address` argument.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Range(usize, usize),
    CallId(String),
}

fn parse_position(text: &str) -> Option<usize> {
    let digits = text.trim().strip_prefix('#').unwrap_or(text.trim());
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

/// Parse an `address`: `#N`, `#A-#B` (the second `#` optional), a JSON
/// integer, or a tool-call id. A bare digit string is a position unless it
/// is also a tool-call id in history (checked by the caller).
fn parse_target(value: &Value) -> Result<Target, String> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .map(|n| Target::Range(n as usize, n as usize))
            .ok_or_else(|| format!("address must be a non-negative position, got {number}")),
        Value::String(text) => {
            let text = text.trim();
            if text.is_empty() {
                return Err("address is empty: pass a marker's `#N` address".into());
            }
            if let Some(rest) = text.strip_prefix('#') {
                if let Some((left, right)) = rest.split_once('-') {
                    return match (parse_position(left), parse_position(right)) {
                        (Some(a), Some(b)) if a <= b => Ok(Target::Range(a, b)),
                        (Some(_), Some(_)) => Err(format!("address range {text} runs backwards")),
                        _ => Err(format!("cannot parse address {text}: use `#N` or `#A-#B`")),
                    };
                }
                return parse_position(rest)
                    .map(|n| Target::Range(n, n))
                    .ok_or_else(|| format!("cannot parse address {text}: use `#N` or `#A-#B`"));
            }
            Ok(Target::CallId(text.to_string()))
        }
        Value::Null => Err("zoom requires `address`: a marker's `#N` address".into()),
        other => Err(format!("address must be a string like `#12`, got {other}")),
    }
}

fn parse_page(value: Option<&Value>) -> Result<usize, String> {
    let page = match value {
        None | Some(Value::Null) => return Ok(1),
        Some(Value::Number(number)) => number.as_u64(),
        Some(Value::String(text)) => text.trim().parse().ok(),
        Some(_) => None,
    };
    match page {
        Some(page) if page >= 1 => Ok(page as usize),
        _ => Err("page must be a positive integer (pages start at 1)".into()),
    }
}

fn error(code: &str, message: String) -> Value {
    Value::String(serde_json::json!({ "ok": false, "error": code, "message": message }).to_string())
}

fn role_of(message: &Value) -> &str {
    message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("message")
}

fn text_of(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        None | Some(Value::Null) => String::new(),
        Some(other) => other.to_string(),
    }
}

/// The name of the call a tool result answers: the nearest earlier
/// assistant message carrying that call id.
fn call_name(history: &[Value], result_position: usize, call_id: &str) -> Option<String> {
    history[..result_position].iter().rev().find_map(|message| {
        message
            .get("tool_calls")
            .and_then(Value::as_array)?
            .iter()
            .find(|call| call.get("id").and_then(Value::as_str) == Some(call_id))
            .and_then(|call| call.get("name").and_then(Value::as_str))
            .map(str::to_string)
    })
}

/// One message rendered for zoom: a one-line header naming its address and
/// kind, then its exact content. An assistant turn's tool calls follow as
/// `tool call <id>: <name> <arguments JSON>` lines.
fn render_message(history: &[Value], position: usize) -> String {
    let message = &history[position];
    let role = role_of(message);
    let call_id = message
        .get("tool_call_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    let header = match role {
        "tool" => {
            let name = call_name(history, position, call_id);
            match (name, call_id.is_empty()) {
                (Some(name), false) => {
                    format!("{} tool result of {name} {call_id}", address(position))
                }
                (None, false) => format!("{} tool result of {call_id}", address(position)),
                (Some(name), true) => format!("{} tool result of {name}", address(position)),
                (None, true) => format!("{} tool result", address(position)),
            }
        }
        other => format!("{} {other} message", address(position)),
    };
    let mut body = text_of(message);
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            let name = call.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = call
                .get("arguments")
                .map(Value::to_string)
                .unwrap_or_else(|| "{}".into());
            if !body.is_empty() {
                body.push('\n');
            }
            if id.is_empty() {
                body.push_str(&format!("tool call: {name} {arguments}"));
            } else {
                body.push_str(&format!("tool call {id}: {name} {arguments}"));
            }
        }
    }
    let images = message
        .get("images")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if images > 0 {
        if !body.is_empty() {
            body.push('\n');
        }
        body.push_str(&format!(
            "[{images} image{} not shown]",
            if images == 1 { "" } else { "s" }
        ));
    }
    format!("{header}:\n{body}")
}

/// Evaluate a zoom call against history (the `Expr::Zoom` semantics). Total:
/// malformed arguments and unknown addresses become readable error results
/// (errors-as-values, t-1222), never interpreter errors. Returns a JSON
/// string value — the tool result text.
pub fn zoom_history(history: &[Value], arguments: &Value, page_chars: usize) -> Value {
    let page_chars = page_chars.max(1);
    let address_value = arguments.get("address").unwrap_or(&Value::Null);
    let target = match parse_target(address_value) {
        Ok(target) => target,
        Err(message) => return error("invalid_address", message),
    };
    let page = match parse_page(arguments.get("page")) {
        Ok(page) => page,
        Err(message) => return error("invalid_page", message),
    };
    let len = history.len();
    let mut note = None::<String>;
    let (start, end) = match target {
        Target::Range(a, b) => (a, b),
        Target::CallId(id) => {
            let matches: Vec<usize> = history
                .iter()
                .enumerate()
                .filter(|(_, message)| {
                    role_of(message) == "tool"
                        && message.get("tool_call_id").and_then(Value::as_str) == Some(id.as_str())
                })
                .map(|(position, _)| position)
                .collect();
            match matches.as_slice() {
                [] => match parse_position(&id) {
                    // A bare number that is not a call id is a position.
                    Some(position) => (position, position),
                    None => {
                        return error(
                            "address_not_found",
                            format!(
                                "no tool result has id {id:?}; pass the `#N` address a GC \
                                 marker names (history holds #0-#{})",
                                len.saturating_sub(1)
                            ),
                        )
                    }
                },
                [only] => (*only, *only),
                [.., last] => {
                    let others = matches[..matches.len() - 1]
                        .iter()
                        .map(|position| address(*position))
                        .collect::<Vec<_>>()
                        .join(", ");
                    note = Some(format!(
                        "note: id {id:?} names {} results; showing the most recent, {} (earlier: {others})",
                        matches.len(),
                        address(*last)
                    ));
                    (*last, *last)
                }
            }
        }
    };
    if len == 0 || end >= len {
        return error(
            "address_out_of_range",
            format!(
                "history holds {len} message{} ({}); {} is out of range",
                if len == 1 { "" } else { "s" },
                if len == 0 {
                    "none".to_string()
                } else {
                    format!("#0-#{}", len - 1)
                },
                if start == end {
                    address(end)
                } else {
                    format!("{}-{}", address(start), address(end))
                }
            ),
        );
    }
    let body = (start..=end)
        .map(|position| render_message(history, position))
        .collect::<Vec<_>>()
        .join("\n\n");
    let chars: Vec<char> = body.chars().collect();
    let pages = chars.len().div_ceil(page_chars).max(1);
    if page > pages {
        return error(
            "page_out_of_range",
            format!(
                "{} has {pages} page{}",
                address_label(start, end),
                plural(pages)
            ),
        );
    }
    let mut out = String::new();
    if let Some(note) = note {
        out.push_str(&note);
        out.push('\n');
    }
    if pages == 1 {
        out.push_str(&body);
        return Value::String(out);
    }
    let from = (page - 1) * page_chars;
    let to = (from + page_chars).min(chars.len());
    out.push_str(&format!(
        "[zoom {} page {page}/{pages}]\n",
        address_label(start, end)
    ));
    out.extend(&chars[from..to]);
    if page < pages {
        out.push_str(&format!(
            "\n[zoom {} page {page}/{pages} ends; next: zoom {{\"address\": \"{}\", \"page\": {}}}]",
            address_label(start, end),
            address_label(start, end),
            page + 1
        ));
    } else {
        out.push_str(&format!(
            "\n[zoom {} page {page}/{pages}: end]",
            address_label(start, end)
        ));
    }
    Value::String(out)
}

fn address_label(start: usize, end: usize) -> String {
    if start == end {
        address(start)
    } else {
        format!("{}-{}", address(start), address(end))
    }
}

fn plural(count: usize) -> &'static str {
    if count == 1 {
        ""
    } else {
        "s"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn history() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "be helpful"}),
            json!({"role": "user", "content": "read the counter"}),
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "call-1", "name": "shell", "arguments": {"command": "cat counter"}}
            ]}),
            json!({"role": "tool", "tool_call_id": "call-1", "content": "{\"stdout\":\"7\\n\"}"}),
            json!({"role": "assistant", "content": "the counter is 7"}),
            // A provider that reuses an id, and one that leaves it empty.
            json!({"role": "assistant", "content": "again", "tool_calls": [
                {"id": "call-1", "name": "shell", "arguments": {"command": "cat counter"}},
                {"id": "", "name": "shell", "arguments": {"command": "date"}}
            ]}),
            json!({"role": "tool", "tool_call_id": "call-1", "content": "{\"stdout\":\"8\\n\"}"}),
            json!({"role": "tool", "tool_call_id": "", "content": "{\"stdout\":\"Tue\\n\"}"}),
        ]
    }

    fn zoom(arguments: Value) -> String {
        zoom_history(&history(), &arguments, MAX_PAGE_CHARS)
            .as_str()
            .unwrap()
            .to_string()
    }

    fn error_code(text: &str) -> String {
        serde_json::from_str::<Value>(text).unwrap()["error"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn positions_address_every_message_kind_exactly() {
        assert_eq!(
            zoom(json!({"address": "#0"})),
            "#0 system message:\nbe helpful"
        );
        assert_eq!(
            zoom(json!({"address": "#1"})),
            "#1 user message:\nread the counter"
        );
        assert_eq!(
            zoom(json!({"address": "#2"})),
            "#2 assistant message:\ntool call call-1: shell {\"command\":\"cat counter\"}"
        );
        assert_eq!(
            zoom(json!({"address": "#3"})),
            "#3 tool result of shell call-1:\n{\"stdout\":\"7\\n\"}"
        );
        assert_eq!(
            zoom(json!({"address": "#4"})),
            "#4 assistant message:\nthe counter is 7"
        );
        // A bare integer and a bare digit string are positions too.
        assert_eq!(zoom(json!({"address": 4})), zoom(json!({"address": "#4"})));
        assert_eq!(
            zoom(json!({"address": "4"})),
            zoom(json!({"address": "#4"}))
        );
    }

    #[test]
    fn a_tool_call_id_aliases_its_most_recent_result_and_says_so() {
        let text = zoom(json!({"address": "call-1"}));
        assert!(
            text.starts_with(
                "note: id \"call-1\" names 2 results; showing the most recent, #6 (earlier: #3)\n"
            ),
            "{text}"
        );
        assert!(
            text.ends_with("#6 tool result of shell call-1:\n{\"stdout\":\"8\\n\"}"),
            "{text}"
        );
    }

    #[test]
    fn the_empty_id_is_never_a_handle() {
        assert_eq!(error_code(&zoom(json!({"address": ""}))), "invalid_address");
        // The id-less result is still reachable by position, and renders
        // without inventing an id.
        assert_eq!(
            zoom(json!({"address": "#7"})),
            "#7 tool result of shell:\n{\"stdout\":\"Tue\\n\"}"
        );
        assert!(zoom(json!({"address": "#5"})).contains("tool call: shell {\"command\":\"date\"}"));
    }

    #[test]
    fn bad_addresses_and_pages_are_readable_errors() {
        assert_eq!(
            error_code(&zoom(json!({"address": "#8"}))),
            "address_out_of_range"
        );
        assert!(zoom(json!({"address": "#99"})).contains("history holds 8 messages (#0-#7)"));
        assert_eq!(
            error_code(&zoom(json!({"address": "call-404"}))),
            "address_not_found"
        );
        assert_eq!(error_code(&zoom(json!({}))), "invalid_address");
        assert_eq!(
            error_code(&zoom(json!({"address": "#x"}))),
            "invalid_address"
        );
        assert_eq!(
            error_code(&zoom(json!({"address": "#5-#2"}))),
            "invalid_address"
        );
        assert_eq!(error_code(&zoom(json!({"address": -1}))), "invalid_address");
        assert_eq!(
            error_code(&zoom(json!({"address": "#1", "page": 0}))),
            "invalid_page"
        );
        assert_eq!(
            error_code(&zoom(json!({"address": "#1", "page": 2}))),
            "page_out_of_range"
        );
    }

    #[test]
    fn ranges_read_consecutive_messages() {
        let text = zoom(json!({"address": "#3-#4"}));
        assert_eq!(
            text,
            "#3 tool result of shell call-1:\n{\"stdout\":\"7\\n\"}\n\n#4 assistant message:\nthe counter is 7"
        );
        assert_eq!(
            error_code(&zoom(json!({"address": "#6-#9"}))),
            "address_out_of_range"
        );
    }

    #[test]
    fn long_bodies_page_and_pages_reassemble_exactly() {
        let body: String = (0..5000)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let history = vec![json!({"role": "user", "content": body})];
        let page_chars = 1024;
        let full = format!("#0 user message:\n{body}");
        let pages = full.chars().count().div_ceil(page_chars);
        let mut reassembled = String::new();
        for page in 1..=pages {
            let text = zoom_history(
                &history,
                &json!({"address": "#0", "page": page}),
                page_chars,
            );
            let text = text.as_str().unwrap();
            let mut lines = text.splitn(2, '\n');
            assert_eq!(
                lines.next().unwrap(),
                format!("[zoom #0 page {page}/{pages}]")
            );
            let rest = lines.next().unwrap();
            let (slice, footer) = rest.rsplit_once('\n').unwrap();
            if page < pages {
                assert!(
                    footer.contains(&format!("\"page\": {}", page + 1)),
                    "{footer}"
                );
            } else {
                assert!(footer.ends_with(": end]"), "{footer}");
            }
            assert!(slice.chars().count() <= page_chars);
            reassembled.push_str(slice);
        }
        assert_eq!(reassembled, full);
    }

    #[test]
    fn page_size_is_an_eighth_of_the_budget_clamped() {
        assert_eq!(
            ZoomOptions::for_context_budget(200_000).page_chars,
            MAX_PAGE_CHARS
        );
        assert_eq!(ZoomOptions::for_context_budget(16_000).page_chars, 6_000);
        assert_eq!(
            ZoomOptions::for_context_budget(1_000).page_chars,
            MIN_PAGE_CHARS
        );
    }

    #[test]
    fn spans_and_marker_addresses_round_trip() {
        assert_eq!(address_span([7]), Some("#7".into()));
        assert_eq!(address_span([9, 3, 5]), Some("#3-#9".into()));
        assert_eq!(address_span([]), None);
        assert_eq!(
            addresses_in("[gc: 2 evicted — shell call-3 #7, user #1; recover: zoom #1-#7 (exact)]"),
            vec![7, 1, 1, 7]
        );
        assert_eq!(
            zoom_targets_in(
                "[gc: 2 evicted — recall 'issue #12' #4, shell call-3 #7; recover: zoom #4-#7 (exact)]"
            ),
            vec![4, 7]
        );
        assert_eq!(
            zoom_targets_in("[gc: 'call-7' evicted 3 times — ...: zoom #9 for only the part]"),
            vec![9]
        );
    }

    #[test]
    fn the_book_maps_ids_to_positions_and_skips_empty_call_ids() {
        let messages: Vec<ChatMessage> = serde_json::from_value(Value::Array(history())).unwrap();
        let book = ZoomBook::from_history(&messages);
        for (position, message) in messages.iter().enumerate() {
            assert_eq!(book.position(&message.id), Some(position));
        }
        assert_eq!(book.result_position("call-1"), Some(6));
        assert_eq!(book.result_position(""), None);
    }
}
