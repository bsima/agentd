use crate::op::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use uuid::Uuid;

pub trait ContextGc: Send + Sync {
    fn collect(
        &self,
        messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage>;

    fn name(&self) -> &'static str;

    fn cache_preserving(&self) -> bool {
        false
    }
}

pub type MsgId = Uuid;
pub type FrameId = String;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    #[default]
    Active,
    Complete,
    Evictable,
    Pinned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameStatus {
    Open,
    Complete,
    Popped,
}

#[derive(Debug, Default)]
pub struct GcState {
    /// Mark-sweep lifecycle tags, keyed by stable ChatMessage UUID.
    pub lifecycle: HashMap<MsgId, LifecycleState>,
    /// Stack-frame status, keyed by provider tool-call id.
    pub frames: HashMap<FrameId, FrameStatus>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RingGc;

#[derive(Debug, Clone, Copy, Default)]
pub struct MarkSweepGc;

#[derive(Debug, Clone, Copy)]
pub enum GcMode {
    None,
    Ring(RingGc),
    MarkSweep(MarkSweepGc),
}

impl GcMode {
    pub fn collect(
        &self,
        messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage> {
        match self {
            Self::None => messages,
            Self::Ring(gc) => gc.collect(messages, budget, state),
            Self::MarkSweep(gc) => gc.collect(messages, budget, state),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ring(gc) => gc.name(),
            Self::MarkSweep(gc) => gc.name(),
        }
    }

    pub fn cache_preserving(&self) -> bool {
        match self {
            Self::None => true,
            Self::Ring(gc) => gc.cache_preserving(),
            Self::MarkSweep(gc) => gc.cache_preserving(),
        }
    }

    pub fn enabled(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn is_mark_sweep(&self) -> bool {
        matches!(self, Self::MarkSweep(_))
    }
}

impl ContextGc for MarkSweepGc {
    fn collect(
        &self,
        mut messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage> {
        tag_lifecycles(&messages, state);
        annotate_evictable_tool_results(&mut messages, state);

        let mut keep = vec![true; messages.len()];
        sweep_by_lifecycle(
            &messages,
            &mut keep,
            state,
            budget,
            LifecycleState::Evictable,
        );
        sweep_by_lifecycle(
            &messages,
            &mut keep,
            state,
            budget,
            LifecycleState::Complete,
        );

        messages
            .into_iter()
            .zip(keep)
            .filter(|(_, keep)| *keep)
            .map(|(message, _)| message)
            .collect()
    }

    fn name(&self) -> &'static str {
        "mark-sweep"
    }

    fn cache_preserving(&self) -> bool {
        true
    }
}

fn tag_lifecycles(messages: &[ChatMessage], state: &mut GcState) {
    let mut tool_results_by_id = HashMap::new();
    for (idx, message) in messages.iter().enumerate() {
        if message.role == "tool" {
            if let Some(id) = &message.tool_call_id {
                tool_results_by_id.insert(id.as_str(), idx);
            }
        }
    }

    for message in messages {
        if message.role == "system" {
            state.lifecycle.insert(message.id, LifecycleState::Pinned);
        } else {
            state
                .lifecycle
                .entry(message.id)
                .or_insert(LifecycleState::Active);
        }
    }

    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for call in tool_calls {
            let Some(result_idx) = tool_results_by_id.get(call.id.as_str()).copied() else {
                continue;
            };
            let incorporated = messages
                .iter()
                .enumerate()
                .skip(result_idx + 1)
                .any(|(_, later)| later.role == "assistant");
            if incorporated {
                let result = &messages[result_idx];
                let result_state = if is_large_tool_result(result) {
                    LifecycleState::Evictable
                } else {
                    LifecycleState::Complete
                };
                state.lifecycle.insert(message.id, LifecycleState::Complete);
                state.lifecycle.insert(result.id, result_state);
            }
        }
    }
}

fn is_large_tool_result(message: &ChatMessage) -> bool {
    message.role == "tool"
        && message
            .content
            .as_deref()
            .is_some_and(|content| content.len() > 512)
}

fn annotate_evictable_tool_results(messages: &mut [ChatMessage], state: &GcState) {
    let call_summaries = tool_call_summaries(messages);
    for message in messages {
        if message.role != "tool" {
            continue;
        }
        if state.lifecycle.get(&message.id) != Some(&LifecycleState::Evictable) {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_deref() else {
            continue;
        };
        let summary = call_summaries
            .get(tool_call_id)
            .cloned()
            .unwrap_or_else(|| tool_call_id.to_string());
        message.content = Some(format!("[tool: {summary} -- result incorporated]"));
    }
}

fn tool_call_summaries(messages: &[ChatMessage]) -> HashMap<String, String> {
    let mut summaries = HashMap::new();
    for message in messages {
        for call in message.tool_calls.as_deref().unwrap_or_default() {
            let args = call.arguments();
            let arg_summary = args
                .get("path")
                .or_else(|| args.get("file"))
                .or_else(|| args.get("command"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or_default();
            let summary = if arg_summary.is_empty() {
                call.name().to_string()
            } else {
                format!("{} {}", call.name(), preview_chars(&arg_summary, 80))
            };
            summaries.insert(call.id.clone(), summary);
        }
    }
    summaries
}

fn preview_chars(input: &str, max_chars: usize) -> String {
    let mut out = input.chars().take(max_chars).collect::<String>();
    if input.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn sweep_by_lifecycle(
    messages: &[ChatMessage],
    keep: &mut [bool],
    state: &GcState,
    budget: usize,
    target: LifecycleState,
) {
    while estimate_tokens(&kept_messages(messages, keep)) > budget {
        let Some(index) = messages.iter().enumerate().position(|(idx, message)| {
            keep[idx]
                && message.role != "system"
                && state.lifecycle.get(&message.id).copied() == Some(target)
        }) else {
            break;
        };
        drop_atomic_group(messages, keep, index);
    }
}

impl ContextGc for RingGc {
    fn collect(
        &self,
        messages: Vec<ChatMessage>,
        budget: usize,
        _state: &mut GcState,
    ) -> Vec<ChatMessage> {
        let mut keep = vec![true; messages.len()];
        while estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            let Some(index) = oldest_droppable_index(&messages, &keep) else {
                break;
            };
            drop_atomic_group(&messages, &mut keep, index);
        }
        messages
            .into_iter()
            .zip(keep)
            .filter(|(_, keep)| *keep)
            .map(|(message, _)| message)
            .collect()
    }

    fn name(&self) -> &'static str {
        "ring"
    }
}

fn oldest_droppable_index(messages: &[ChatMessage], keep: &[bool]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .find(|(idx, message)| keep[*idx] && message.role != "system")
        .map(|(idx, _)| idx)
}

fn kept_messages(messages: &[ChatMessage], keep: &[bool]) -> Vec<ChatMessage> {
    messages
        .iter()
        .zip(keep.iter())
        .filter(|(_, keep)| **keep)
        .map(|(message, _)| message.clone())
        .collect()
}

fn drop_atomic_group(messages: &[ChatMessage], keep: &mut [bool], index: usize) {
    let mut ids = BTreeSet::new();
    collect_pair_ids(&messages[index], &mut ids);
    keep[index] = false;

    let mut changed = true;
    while changed {
        changed = false;
        for (idx, message) in messages.iter().enumerate() {
            if !keep[idx] {
                continue;
            }
            if message_mentions_any_id(message, &ids) {
                keep[idx] = false;
                collect_pair_ids(message, &mut ids);
                changed = true;
            }
        }
    }
}

fn collect_pair_ids(message: &ChatMessage, ids: &mut BTreeSet<String>) {
    if let Some(tool_calls) = &message.tool_calls {
        ids.extend(tool_calls.iter().map(|call| call.id.clone()));
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        ids.insert(tool_call_id.clone());
    }
}

fn message_mentions_any_id(message: &ChatMessage, ids: &BTreeSet<String>) -> bool {
    message
        .tool_calls
        .as_ref()
        .is_some_and(|calls| calls.iter().any(|call| ids.contains(&call.id)))
        || message
            .tool_call_id
            .as_ref()
            .is_some_and(|id| ids.contains(id))
}

pub fn truncate_oversized_message(messages: &mut Vec<ChatMessage>, budget: usize) {
    const MARKER: &str = "\n...[truncated for context budget]";
    if budget == 0 {
        for message in messages {
            message.content = Some(MARKER.to_string());
        }
        return;
    }
    let marker_tokens = estimate_text_tokens(MARKER);
    let max_content_tokens = budget
        .saturating_sub(estimate_message_overhead_tokens())
        .max(1);
    let target_tokens = max_content_tokens.saturating_sub(marker_tokens).max(1);
    let target_chars = target_tokens.saturating_mul(3);

    for message in messages {
        if estimate_tokens(std::slice::from_ref(message)) <= budget {
            continue;
        }
        if let Some(content) = &mut message.content {
            let mut truncated: String = content.chars().take(target_chars).collect();
            truncated.push_str(MARKER);
            *content = truncated;
        }
    }
}

pub fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|message| {
            estimate_message_overhead_tokens()
                .saturating_add(estimate_text_tokens(&message.role))
                .saturating_add(message.content.as_deref().map_or(0, estimate_text_tokens))
                .saturating_add(
                    message
                        .tool_call_id
                        .as_deref()
                        .map_or(0, estimate_text_tokens),
                )
                .saturating_add(message.tool_calls.as_ref().map_or(0, |calls| {
                    calls
                        .iter()
                        .map(|call| {
                            estimate_text_tokens(&call.id)
                                .saturating_add(estimate_text_tokens(&call.kind))
                                .saturating_add(estimate_text_tokens(&call.function.name))
                                .saturating_add(estimate_text_tokens(&call.function.arguments))
                        })
                        .sum()
                }))
        })
        .sum()
}

fn estimate_message_overhead_tokens() -> usize {
    8
}

fn estimate_text_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(3).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::ResponseToolCall;

    fn tool_call(id: &str) -> ResponseToolCall {
        ResponseToolCall::new(id, "shell", serde_json::json!({}))
    }

    fn read_file_call(id: &str, path: &str) -> ResponseToolCall {
        ResponseToolCall::new(id, "read_file", serde_json::json!({ "path": path }))
    }

    #[test]
    fn mark_sweep_annotates_large_incorporated_tool_results() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(None, vec![read_file_call("call-1", "/tmp/large.txt")]),
            ChatMessage::tool("call-1", "x".repeat(2000)),
            ChatMessage::assistant(Some("I incorporated that result".into()), vec![]),
        ];
        let mut state = GcState::default();
        let collected = MarkSweepGc.collect(messages, 120, &mut state);

        let tool = collected
            .iter()
            .find(|message| message.role == "tool")
            .unwrap();
        assert_eq!(
            tool.content.as_deref(),
            Some("[tool: read_file /tmp/large.txt -- result incorporated]")
        );
        assert_eq!(state.lifecycle[&tool.id], LifecycleState::Evictable);
    }

    #[test]
    fn mark_sweep_evicts_completed_pairs_under_pressure() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(None, vec![tool_call("call-1")]),
            ChatMessage::tool("call-1", "small result"),
            ChatMessage::assistant(Some("done with result".into()), vec![]),
            ChatMessage::user("recent user message that should stay"),
        ];
        let mut state = GcState::default();
        let collected = MarkSweepGc.collect(messages, 40, &mut state);

        assert!(collected.iter().any(|message| message.role == "system"));
        assert!(collected.iter().any(|message| {
            message.role == "user"
                && message.content.as_deref() == Some("recent user message that should stay")
        }));
        assert!(!collected.iter().any(|message| {
            message.tool_call_id.as_deref() == Some("call-1")
                || message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| calls.iter().any(|call| call.id == "call-1"))
        }));
    }

    #[test]
    fn mark_sweep_is_deterministic() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(None, vec![read_file_call("call-1", "/tmp/a")]),
            ChatMessage::tool("call-1", "a".repeat(2000)),
            ChatMessage::assistant(Some("incorporated a".into()), vec![]),
            ChatMessage::assistant(None, vec![tool_call("call-2")]),
            ChatMessage::tool("call-2", "small result"),
            ChatMessage::assistant(Some("incorporated b".into()), vec![]),
            ChatMessage::user("latest"),
        ];
        let mut state_a = GcState::default();
        let mut state_b = GcState::default();

        let a = MarkSweepGc.collect(messages.clone(), 55, &mut state_a);
        let b = MarkSweepGc.collect(messages, 55, &mut state_b);

        assert_eq!(a, b);
    }

    #[test]
    fn ring_gc_drops_tool_call_and_result_atomically() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("old user message with enough text to be dropped"),
            ChatMessage::assistant(None, vec![tool_call("call-1")]),
            ChatMessage::tool("call-1", "tool result with enough text to be paired"),
            ChatMessage::user("recent user message that should remain"),
        ];
        let mut state = GcState::default();
        let collected = RingGc.collect(messages, 45, &mut state);

        let live_call_ids: BTreeSet<_> = collected
            .iter()
            .flat_map(|message| {
                message
                    .tool_calls
                    .iter()
                    .flatten()
                    .map(|call| call.id.as_str())
            })
            .collect();
        for message in &collected {
            if let Some(tool_call_id) = message.tool_call_id.as_deref() {
                assert!(
                    live_call_ids.contains(tool_call_id),
                    "orphaned tool result remained: {tool_call_id}; collected={collected:?}"
                );
            }
        }
        assert!(!collected.iter().any(|message| {
            message.tool_call_id.as_deref() == Some("call-1")
                || message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| calls.iter().any(|call| call.id == "call-1"))
        }));
    }
}
