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
    /// Whether the most recent collect() changed bytes inside the cached
    /// prefix region (provider prompt caches key on a stable prefix).
    /// Set by every strategy on every collection; read for gc_collect
    /// trace events.
    pub prefix_invalidated: bool,
    /// Infer calls seen by this loop run; drives the every-N timing strategy.
    pub infer_calls: u64,
    /// Token budget a catch-overflow retry actually succeeded under. Once
    /// the provider has rejected a prompt, its real window — not our
    /// estimate — is the ceiling; later calls in the same loop collect to
    /// this proactively instead of paying a failed request per turn.
    pub discovered_budget: Option<usize>,
}

/// When GC runs, independent of which strategy reclaims tokens (t-1151).
/// Token estimates diverge from provider tokenizers, so a purely
/// estimate-driven threshold can sit idle while the provider hard-rejects;
/// catch-overflow makes the provider the source of truth instead.
///
/// Every timing is additionally composed with the collect-on-overflow
/// backstop (t-1343, `interpreter::maybe_collect_prompt`): if the assembled
/// prompt would exceed the full context budget at Infer time and the timing
/// policy did not fire, a collection runs before dispatch anyway (emitted
/// as `gc_collect{reason: "backstop"}`). Periodic timings like `every:N`
/// therefore cannot leave the window over budget between collections.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GcTiming {
    /// Collect when the estimated prompt size crosses
    /// `context_budget * gc_threshold` (the historical default).
    #[default]
    Threshold,
    /// No estimate-based trigger: on a provider context-overflow error,
    /// collect to a shrinking budget and retry the same turn.
    CatchOverflow,
    /// Collect before every infer call.
    Eager,
    /// Collect on every Nth infer call (N >= 1).
    EveryN(u64),
}

impl GcTiming {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Threshold => "threshold",
            Self::CatchOverflow => "catch-overflow",
            Self::Eager => "eager",
            Self::EveryN(_) => "every-n",
        }
    }
}

impl std::str::FromStr for GcTiming {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "threshold" => Ok(Self::Threshold),
            "catch-overflow" => Ok(Self::CatchOverflow),
            "eager" => Ok(Self::Eager),
            other => {
                if let Some(n) = other.strip_prefix("every:") {
                    let n: u64 = n
                        .parse()
                        .map_err(|_| format!("invalid every:N turn count {n:?}"))?;
                    if n == 0 {
                        return Err("every:N requires N >= 1".into());
                    }
                    return Ok(Self::EveryN(n));
                }
                Err(format!(
                    "unknown gc timing {other:?}; expected threshold, catch-overflow, eager, or every:N"
                ))
            }
        }
    }
}

/// Fraction of the budget pinned as the stable cache-prefix region under
/// preserve mode: the system prompt plus the oldest messages up to this
/// share of the budget never change, so provider prefix caches keep hitting.
const CACHE_PREFIX_BUDGET_RATIO: f32 = 0.25;

/// Index of the first message *outside* the pinned cache prefix. System
/// messages are always pinned regardless of position; the oldest non-system
/// messages are pinned until the prefix allowance is spent. The boundary
/// never splits a tool-call pair: if a pinned assistant message issued a
/// call, its result is pinned too.
fn cache_prefix_boundary(messages: &[ChatMessage], budget: usize) -> usize {
    let allowance = ((budget as f32) * CACHE_PREFIX_BUDGET_RATIO) as usize;
    let mut spent = 0usize;
    let mut boundary = 0usize;
    let mut pinned_call_ids = BTreeSet::new();
    for (index, message) in messages.iter().enumerate() {
        let tokens = estimate_tokens(std::slice::from_ref(message));
        let completes_pinned_pair = message
            .tool_call_id
            .as_ref()
            .is_some_and(|id| pinned_call_ids.contains(id));
        if message.role != "system" && spent + tokens > allowance && !completes_pinned_pair {
            break;
        }
        spent = spent.saturating_add(tokens);
        collect_pair_ids(message, &mut pinned_call_ids);
        boundary = index + 1;
    }
    boundary
}

#[derive(Debug, Clone, Copy)]
pub struct RingGc {
    /// Preserve the cached prefix: evict oldest-first from the *interior*
    /// (after the pinned prefix region) instead of the front, falling back
    /// to front-drop (and reporting the invalidation) only when preserving
    /// cannot reach the budget.
    pub preserve_prefix: bool,
}

impl Default for RingGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MarkSweepGc {
    /// Preserve the cached prefix: only annotate/evict messages after the
    /// pinned prefix region.
    pub preserve_prefix: bool,
}

impl Default for MarkSweepGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
        }
    }
}

/// Strategy 3 (docs/GC.md): model each tool invocation+result as an
/// activation frame. When over budget, pop completed frames oldest-first:
/// the assistant tool-call message is rewritten in place to a one-line
/// `[frame: tool(args) -> result]` annotation (keeping its stable id) and
/// the tool result messages are dropped. The semantic record survives at
/// ~1% of the tokens, which is why this is the space-efficient choice for
/// tool-heavy agents. Summaries are pure heuristics — no LLM calls (the
/// `stack-smart` variant is gated on the eval harness).
#[derive(Debug, Clone, Copy)]
pub struct StackFrameGc {
    /// Preserve the cached prefix: only pop frames living entirely after
    /// the pinned prefix region.
    pub preserve_prefix: bool,
}

impl Default for StackFrameGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GcMode {
    None,
    Ring(RingGc),
    MarkSweep(MarkSweepGc),
    Stack(StackFrameGc),
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
            Self::Stack(gc) => gc.collect(messages, budget, state),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ring(gc) => gc.name(),
            Self::MarkSweep(gc) => gc.name(),
            Self::Stack(gc) => gc.name(),
        }
    }

    pub fn cache_preserving(&self) -> bool {
        match self {
            Self::None => true,
            Self::Ring(gc) => gc.cache_preserving(),
            Self::MarkSweep(gc) => gc.cache_preserving(),
            Self::Stack(gc) => gc.cache_preserving(),
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
        let boundary = cache_prefix_boundary(&messages, budget);
        let prefix_snapshot = messages[..boundary].to_vec();
        // Under preserve, annotation and eviction are restricted to the
        // interior; in ignore mode the whole window is fair game.
        let restrict = if self.preserve_prefix { boundary } else { 0 };

        tag_lifecycles(&messages, state);
        annotate_evictable_tool_results(&mut messages, state, restrict);

        let mut keep = vec![true; messages.len()];
        sweep_by_lifecycle(
            &messages,
            &mut keep,
            state,
            budget,
            restrict,
            LifecycleState::Evictable,
        );
        sweep_by_lifecycle(
            &messages,
            &mut keep,
            state,
            budget,
            restrict,
            LifecycleState::Complete,
        );

        let collected: Vec<ChatMessage> = messages
            .into_iter()
            .zip(keep)
            .filter(|(_, keep)| *keep)
            .map(|(message, _)| message)
            .collect();
        state.prefix_invalidated = prefix_changed(&prefix_snapshot, &collected);
        collected
    }

    fn name(&self) -> &'static str {
        "mark-sweep"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

/// Did the collected window change anything inside the pinned prefix region?
/// Provider prompt caches key on a byte-stable prefix, so any drop or
/// mutation among the leading messages invalidates them.
fn prefix_changed(prefix_snapshot: &[ChatMessage], collected: &[ChatMessage]) -> bool {
    if collected.len() < prefix_snapshot.len() {
        return true;
    }
    prefix_snapshot != &collected[..prefix_snapshot.len()]
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

fn annotate_evictable_tool_results(messages: &mut [ChatMessage], state: &GcState, boundary: usize) {
    let call_summaries = tool_call_summaries(messages);
    for message in messages.iter_mut().skip(boundary) {
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
            let arg_summary = call
                .arguments
                .get("path")
                .or_else(|| call.arguments.get("file"))
                .or_else(|| call.arguments.get("command"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or_default();
            let summary = if arg_summary.is_empty() {
                call.name.clone()
            } else {
                format!("{} {}", call.name, preview_chars(&arg_summary, 80))
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
    boundary: usize,
    target: LifecycleState,
) {
    while estimate_tokens(&kept_messages(messages, keep)) > budget {
        let Some(index) = messages.iter().enumerate().position(|(idx, message)| {
            idx >= boundary
                && keep[idx]
                && message.role != "system"
                && state.lifecycle.get(&message.id).copied() == Some(target)
                && atomic_group_stays_past(messages, keep, idx, boundary)
        }) else {
            break;
        };
        drop_atomic_group(messages, keep, index);
    }
}

/// Would dropping `index`'s atomic group (tool-call pairs travel together)
/// touch anything before `boundary`? Used to keep preserve-mode sweeps from
/// pulling pinned-prefix messages out via pair atomicity.
fn atomic_group_stays_past(
    messages: &[ChatMessage],
    keep: &[bool],
    index: usize,
    boundary: usize,
) -> bool {
    if boundary == 0 {
        return true;
    }
    let mut scratch = keep.to_vec();
    drop_atomic_group(messages, &mut scratch, index);
    keep.iter()
        .zip(&scratch)
        .take(boundary)
        .all(|(before, after)| before == after)
}

impl ContextGc for RingGc {
    fn collect(
        &self,
        messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage> {
        let boundary = if self.preserve_prefix {
            cache_prefix_boundary(&messages, budget)
        } else {
            0
        };
        let prefix_snapshot =
            messages[..cache_prefix_boundary(&messages, budget).min(messages.len())].to_vec();

        let mut keep = vec![true; messages.len()];
        // Phase 1: drop oldest-first from the interior (boundary 0 in ignore
        // mode makes this the classic front-drop).
        sweep_ring(&messages, &mut keep, budget, boundary);
        // Phase 2 (preserve fallback): the pinned prefix plus the live tail
        // alone exceed the budget. Overflowing the model is worse than a
        // cache miss, so degrade to front-drop; the gc_collect event reports
        // the invalidation via state.prefix_invalidated.
        if boundary > 0 && estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            sweep_ring(&messages, &mut keep, budget, 0);
        }

        let collected: Vec<ChatMessage> = messages
            .into_iter()
            .zip(keep)
            .filter(|(_, keep)| *keep)
            .map(|(message, _)| message)
            .collect();
        state.prefix_invalidated = prefix_changed(&prefix_snapshot, &collected);
        collected
    }

    fn name(&self) -> &'static str {
        "ring"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

impl ContextGc for StackFrameGc {
    fn collect(
        &self,
        mut messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage> {
        let full_boundary = cache_prefix_boundary(&messages, budget).min(messages.len());
        let prefix_snapshot = messages[..full_boundary].to_vec();
        let boundary = if self.preserve_prefix {
            full_boundary
        } else {
            0
        };

        record_frame_statuses(&messages, state);

        let mut keep = vec![true; messages.len()];
        // Phase 1: pop completed frames oldest-first until under budget.
        while estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            let Some(frame) = oldest_completed_frame(&messages, &keep, boundary) else {
                break;
            };
            pop_frame(&mut messages, &mut keep, &frame, state);
        }
        // Phase 2: frames alone were not enough (open frames, chat-heavy
        // windows); drop oldest-first from the interior like ring.
        if estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            sweep_ring(&messages, &mut keep, budget, boundary);
        }
        // Phase 3 (preserve fallback): the pinned prefix plus the live tail
        // alone exceed the budget. Overflowing the model is worse than a
        // cache miss, so degrade to front-drop; the gc_collect event reports
        // the invalidation via state.prefix_invalidated.
        if boundary > 0 && estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            sweep_ring(&messages, &mut keep, budget, 0);
        }

        let collected: Vec<ChatMessage> = messages
            .into_iter()
            .zip(keep)
            .filter(|(_, keep)| *keep)
            .map(|(message, _)| message)
            .collect();
        state.prefix_invalidated = prefix_changed(&prefix_snapshot, &collected);
        collected
    }

    fn name(&self) -> &'static str {
        "stack"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

/// A completed activation frame: the assistant message that issued the
/// tool calls plus every tool result answering them, all inside the window.
struct Frame {
    assistant: usize,
    results: Vec<usize>,
}

/// Update `GcState.frames` from what this window shows: a call with a
/// result in the window is Complete, one still awaiting its result is
/// Open. Popped is terminal — set by `pop_frame`, never downgraded here.
fn record_frame_statuses(messages: &[ChatMessage], state: &mut GcState) {
    let results: BTreeSet<&str> = messages
        .iter()
        .filter_map(|message| message.tool_call_id.as_deref())
        .collect();
    for message in messages {
        for call in message.tool_calls.as_deref().unwrap_or_default() {
            let status = if results.contains(call.id.as_str()) {
                FrameStatus::Complete
            } else {
                FrameStatus::Open
            };
            let entry = state.frames.entry(call.id.clone()).or_insert(status);
            if *entry != FrameStatus::Popped {
                *entry = status;
            }
        }
    }
}

/// The oldest kept frame whose every member sits at or past `boundary`.
/// Frames with any unanswered call are open — never popped, never split.
/// A frame is only poppable once a later assistant message exists past its
/// last result: until the model has spoken again, the result is the live
/// working set, not history (same incorporation rule as mark-sweep).
fn oldest_completed_frame(
    messages: &[ChatMessage],
    keep: &[bool],
    boundary: usize,
) -> Option<Frame> {
    for (index, message) in messages.iter().enumerate().skip(boundary) {
        if !keep[index] || message.role != "assistant" {
            continue;
        }
        let calls = message.tool_calls.as_deref().unwrap_or_default();
        if calls.is_empty() {
            continue;
        }
        let results: Vec<usize> = calls
            .iter()
            .filter_map(|call| {
                messages.iter().enumerate().position(|(idx, candidate)| {
                    keep[idx] && candidate.tool_call_id.as_deref() == Some(call.id.as_str())
                })
            })
            .collect();
        let incorporated = results.iter().max().is_some_and(|last| {
            messages
                .iter()
                .enumerate()
                .skip(last + 1)
                .any(|(idx, later)| keep[idx] && later.role == "assistant")
        });
        if results.len() == calls.len()
            && results.iter().all(|idx| *idx >= boundary)
            && incorporated
        {
            return Some(Frame {
                assistant: index,
                results,
            });
        }
    }
    None
}

/// Pop a completed frame: rewrite the assistant message in place to the
/// summary annotation (stable id preserved — no fresh UUIDs, so repeated
/// collections stay deterministic) and drop the result messages.
fn pop_frame(messages: &mut [ChatMessage], keep: &mut [bool], frame: &Frame, state: &mut GcState) {
    let summary = frame_summary(messages, frame);
    for call in messages[frame.assistant]
        .tool_calls
        .as_deref()
        .unwrap_or_default()
    {
        state.frames.insert(call.id.clone(), FrameStatus::Popped);
    }
    let assistant = &mut messages[frame.assistant];
    assistant.content = Some(summary);
    assistant.tool_calls = None;
    for index in &frame.results {
        keep[*index] = false;
    }
}

/// `[frame: tool(args) -> result]`, one line per call, prefixed with a
/// preview of the assistant's own narration when it had any. Pure
/// heuristics by design — an LLM summarization call here would spend
/// tokens to save tokens (docs/GC.md reserves that for `stack-smart`).
fn frame_summary(messages: &[ChatMessage], frame: &Frame) -> String {
    let assistant = &messages[frame.assistant];
    let mut lines = Vec::new();
    if let Some(content) = assistant.content.as_deref() {
        if !content.trim().is_empty() {
            lines.push(preview_chars(content.trim(), 120));
        }
    }
    for call in assistant.tool_calls.as_deref().unwrap_or_default() {
        let args = call
            .arguments
            .get("path")
            .or_else(|| call.arguments.get("file"))
            .or_else(|| call.arguments.get("command"))
            .or_else(|| call.arguments.get("prompt"))
            .and_then(serde_json::Value::as_str)
            .map(|value| preview_chars(value, 80))
            .unwrap_or_default();
        let result = frame
            .results
            .iter()
            .map(|idx| &messages[*idx])
            .find(|message| message.tool_call_id.as_deref() == Some(call.id.as_str()))
            .and_then(|message| message.content.as_deref())
            .map(|content| preview_chars(content.trim(), 120))
            .unwrap_or_default();
        lines.push(format!("[frame: {}({args}) -> {result}]", call.name));
    }
    lines.join("\n")
}

fn sweep_ring(messages: &[ChatMessage], keep: &mut [bool], budget: usize, boundary: usize) {
    while estimate_tokens(&kept_messages(messages, keep)) > budget {
        let Some(index) = oldest_droppable_index(messages, keep, boundary) else {
            break;
        };
        drop_atomic_group(messages, keep, index);
    }
}

fn oldest_droppable_index(
    messages: &[ChatMessage],
    keep: &[bool],
    boundary: usize,
) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .find(|(idx, message)| {
            *idx >= boundary
                && keep[*idx]
                && message.role != "system"
                && atomic_group_stays_past(messages, keep, *idx, boundary)
        })
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

/// Returns how many messages were shrunk so the caller can emit a distinct
/// gc_truncate trace event: single-message token-budget pressure is a
/// different overflow condition than whole-window gc_collect eviction
/// (t-1133 overflow taxonomy).
pub fn truncate_oversized_message(messages: &mut Vec<ChatMessage>, budget: usize) -> usize {
    const MARKER: &str = "\n...[truncated for context budget]";
    if budget == 0 {
        let count = messages.len();
        for message in messages {
            message.content = Some(MARKER.to_string());
            truncate_tool_call_arguments(message, 1);
        }
        return count;
    }
    let marker_tokens = estimate_text_tokens(MARKER);
    let max_content_tokens = budget
        .saturating_sub(estimate_message_overhead_tokens())
        .max(1);
    let target_tokens = max_content_tokens.saturating_sub(marker_tokens).max(1);

    let mut truncated_count = 0;
    for message in messages {
        // A single over-budget message defeats every strategy: nothing dropped
        // *around* it can help, so the GC loop would bail and ship an
        // over-budget prompt anyway. Shrink content AND tool_call arguments,
        // halving the cap until the message fits (or we hit the floor).
        if estimate_tokens(std::slice::from_ref(message)) > budget {
            truncated_count += 1;
        }
        let mut cap_tokens = target_tokens;
        loop {
            if estimate_tokens(std::slice::from_ref(message)) <= budget {
                break;
            }
            let cap_chars = cap_tokens.saturating_mul(3).max(1);
            if let Some(content) = &mut message.content {
                if content.chars().count() > cap_chars {
                    let mut truncated: String = content.chars().take(cap_chars).collect();
                    truncated.push_str(MARKER);
                    *content = truncated;
                }
            }
            truncate_tool_call_arguments(message, cap_chars);
            if cap_tokens == 1 {
                break;
            }
            cap_tokens = (cap_tokens / 2).max(1);
        }
    }
    truncated_count
}

/// Replace oversized tool-call argument values with a marked preview. The
/// call id and name stay intact so pair-atomicity and provider echo keep
/// working; only the argument payload shrinks.
fn truncate_tool_call_arguments(message: &mut ChatMessage, cap_chars: usize) {
    let Some(calls) = &mut message.tool_calls else {
        return;
    };
    for call in calls.iter_mut() {
        let serialized = call.arguments.to_string();
        if serialized.chars().count() > cap_chars {
            call.arguments = serde_json::json!({
                "truncated": preview_chars(&serialized, cap_chars),
            });
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
                                .saturating_add(estimate_text_tokens(&call.name))
                                .saturating_add(estimate_text_tokens(&call.arguments.to_string()))
                        })
                        .sum()
                }))
        })
        .sum()
}

fn estimate_message_overhead_tokens() -> usize {
    8
}

/// The one token estimator for budget decisions (GC trigger/stop conditions
/// and PromptIR section budgets). Per docs/GC.md this must be a conservative
/// *upper bound*: chars/3 over-counts on prose, which errs toward GC firing
/// early rather than overflowing the provider context.
pub(crate) fn estimate_text_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(3).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::ToolCall;

    fn tool_call(id: &str) -> ToolCall {
        ToolCall::new(id, "shell", serde_json::json!({}))
    }

    /// system + user + two completed shell frames with fat results + a
    /// closing assistant turn. The frames are where the tokens live.
    fn stack_fixture() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user("please run the tests"),
            ChatMessage::assistant(
                Some("Running the suite first.".into()),
                vec![ToolCall::new(
                    "call-1",
                    "shell",
                    serde_json::json!({ "command": "cargo test" }),
                )],
            ),
            ChatMessage::tool("call-1", format!("test output {}", "x".repeat(1200))),
            ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    "call-2",
                    "shell",
                    serde_json::json!({ "command": "cargo clippy" }),
                )],
            ),
            ChatMessage::tool("call-2", format!("clippy output {}", "y".repeat(1200))),
            ChatMessage::assistant(Some("All checks pass.".into()), vec![]),
        ]
    }

    #[test]
    fn stack_pops_completed_frames_to_summary_annotations() {
        let messages = stack_fixture();
        let mut state = GcState::default();
        let budget = 200;

        let collected = StackFrameGc::default().collect(messages, budget, &mut state);

        assert!(
            estimate_tokens(&collected) <= budget,
            "must converge: {} tokens",
            estimate_tokens(&collected)
        );
        assert!(collected.iter().any(|message| message.role == "system"));
        assert!(
            collected.iter().all(|message| message.role != "tool"),
            "popped frames drop their tool results: {collected:?}"
        );
        let summary = collected
            .iter()
            .find(|message| {
                message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("[frame: shell(cargo test)"))
            })
            .expect("popped frame leaves a summary annotation");
        assert!(
            summary.tool_calls.is_none(),
            "summary annotations carry no tool calls"
        );
        assert_eq!(state.frames.get("call-1"), Some(&FrameStatus::Popped));
        assert_eq!(state.frames.get("call-2"), Some(&FrameStatus::Popped));
        // The closing narration survives — frames pop, conversation stays.
        assert!(collected
            .iter()
            .any(|message| message.content.as_deref() == Some("All checks pass.")));
    }

    #[test]
    fn stack_never_pops_or_splits_open_frames() {
        let mut messages = stack_fixture();
        // A pending call with no result yet: the model is mid-tool-turn.
        messages.push(ChatMessage::assistant(
            None,
            vec![ToolCall::new(
                "call-3",
                "shell",
                serde_json::json!({ "command": "cargo build" }),
            )],
        ));
        let mut state = GcState::default();

        let collected = StackFrameGc::default().collect(messages, 200, &mut state);

        let pending = collected
            .iter()
            .find(|message| {
                message
                    .tool_calls
                    .as_deref()
                    .is_some_and(|calls| calls.iter().any(|call| call.id == "call-3"))
            })
            .expect("the open frame's call survives intact");
        assert!(pending.content.is_none(), "open frames are not rewritten");
        assert_eq!(state.frames.get("call-3"), Some(&FrameStatus::Open));
    }

    #[test]
    fn stack_keeps_unincorporated_trailing_frame() {
        // The model just got the clippy result back and has not spoken yet:
        // that result is the live working set, not history. Strip the
        // closing narration from the fixture so frame 2 is unincorporated,
        // and fatten the user turn so it exhausts the pinned-prefix
        // allowance (otherwise pair-pinning absorbs frame 1 into the
        // prefix, where preserve mode correctly refuses to pop).
        let mut messages = stack_fixture();
        messages.pop();
        messages[1] = ChatMessage::user("context ".repeat(80));
        let mut state = GcState::default();

        // Roomy enough to hold the live frame once the older one pops;
        // too tight for both fat results to stay verbatim.
        let collected = StackFrameGc::default().collect(messages, 600, &mut state);

        assert_eq!(state.frames.get("call-1"), Some(&FrameStatus::Popped));
        assert!(
            collected.iter().any(|message| {
                message.tool_call_id.as_deref() == Some("call-2")
                    && message
                        .content
                        .as_deref()
                        .is_some_and(|content| content.contains("clippy output"))
            }),
            "the un-incorporated result must survive intact: {collected:?}"
        );
        assert_eq!(state.frames.get("call-2"), Some(&FrameStatus::Complete));
    }

    #[test]
    fn stack_falls_back_to_ring_drop_when_no_frames_exist() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("a".repeat(400)),
            ChatMessage::user("b".repeat(400)),
            ChatMessage::user("c".repeat(400)),
        ];
        let mut state = GcState::default();
        // Room for the system prompt plus roughly one 400-char message:
        // the oldest must go, the newest must survive.
        let budget = 300;

        let collected = StackFrameGc::default().collect(messages, budget, &mut state);

        assert!(estimate_tokens(&collected) <= budget);
        assert!(collected.iter().any(|message| message.role == "system"));
        // Oldest user content goes first, newest survives.
        assert!(collected.iter().any(|message| message
            .content
            .as_deref()
            .is_some_and(|c| c.starts_with('c'))));
        assert!(!collected.iter().any(|message| message
            .content
            .as_deref()
            .is_some_and(|c| c.starts_with('a'))));
    }

    #[test]
    fn stack_collect_is_idempotent_and_deterministic() {
        let messages = stack_fixture();
        let budget = 200;

        let mut state_a = GcState::default();
        let first = StackFrameGc::default().collect(messages.clone(), budget, &mut state_a);
        let again = StackFrameGc::default().collect(first.clone(), budget, &mut state_a);
        assert_eq!(first, again, "collecting collected output is a no-op");

        let mut state_b = GcState::default();
        let replayed = StackFrameGc::default().collect(messages, budget, &mut state_b);
        assert_eq!(first, replayed, "same inputs, same output, every run");
    }

    #[test]
    fn stack_under_budget_is_untouched_and_cache_preserving() {
        let messages = stack_fixture();
        let mut state = GcState::default();

        let collected = StackFrameGc::default().collect(messages.clone(), 100_000, &mut state);

        assert_eq!(collected, messages);
        assert!(!state.prefix_invalidated);
        // Statuses are still recorded even when nothing pops.
        assert_eq!(state.frames.get("call-1"), Some(&FrameStatus::Complete));
    }

    #[test]
    fn gc_timing_parses_all_forms() {
        assert_eq!("threshold".parse(), Ok(GcTiming::Threshold));
        assert_eq!("catch-overflow".parse(), Ok(GcTiming::CatchOverflow));
        assert_eq!("eager".parse(), Ok(GcTiming::Eager));
        assert_eq!("every:5".parse(), Ok(GcTiming::EveryN(5)));
        assert!("every:0".parse::<GcTiming>().is_err());
        assert!("every:x".parse::<GcTiming>().is_err());
        assert!("sometimes".parse::<GcTiming>().is_err());
    }

    fn read_file_call(id: &str, path: &str) -> ToolCall {
        ToolCall::new(id, "read_file", serde_json::json!({ "path": path }))
    }

    #[test]
    fn truncate_oversized_message_shrinks_giant_tool_call_arguments() {
        let budget = 200;
        let mut messages = vec![ChatMessage::assistant(
            None,
            vec![ToolCall::new(
                "call-1",
                "shell",
                serde_json::json!({ "command": "x".repeat(10_000) }),
            )],
        )];

        truncate_oversized_message(&mut messages, budget);

        assert!(
            estimate_tokens(&messages) <= budget,
            "pre-pass must converge: {} tokens",
            estimate_tokens(&messages)
        );
        let call = &messages[0].tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.id, "call-1");
        assert_eq!(call.name, "shell");
        assert!(
            call.arguments.get("truncated").is_some(),
            "arguments must carry the truncation marker: {:?}",
            call.arguments
        );
    }

    #[test]
    fn truncate_oversized_message_shrinks_content_and_arguments_together() {
        let budget = 300;
        let mut messages = vec![ChatMessage::assistant(
            Some("y".repeat(20_000)),
            vec![ToolCall::new(
                "call-1",
                "shell",
                serde_json::json!({ "command": "x".repeat(20_000) }),
            )],
        )];

        truncate_oversized_message(&mut messages, budget);

        assert!(
            estimate_tokens(&messages) <= budget,
            "pre-pass must converge: {} tokens",
            estimate_tokens(&messages)
        );
        assert!(messages[0]
            .content
            .as_deref()
            .unwrap()
            .contains("[truncated for context budget]"));
    }

    #[test]
    fn truncate_oversized_message_keeps_content_only_behavior() {
        let budget = 100;
        let mut messages = vec![
            ChatMessage::system("small"),
            ChatMessage::user("z".repeat(5_000)),
        ];

        truncate_oversized_message(&mut messages, budget);

        assert!(estimate_tokens(&messages) <= 2 * budget);
        assert_eq!(messages[0].content.as_deref(), Some("small"));
        assert!(messages[1]
            .content
            .as_deref()
            .unwrap()
            .ends_with("[truncated for context budget]"));
        assert!(estimate_tokens(std::slice::from_ref(&messages[1])) <= budget);
    }

    #[test]
    fn mark_sweep_cache_preserving_matches_retained_prompt_bytes() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(None, vec![read_file_call("call-1", "/tmp/large.txt")]),
            ChatMessage::tool("call-1", "x".repeat(2000)),
            ChatMessage::assistant(Some("I incorporated that result".into()), vec![]),
        ];
        let mut state = GcState::default();
        let collected = MarkSweepGc::default().collect(messages.clone(), 10_000, &mut state);

        if MarkSweepGc::default().cache_preserving() {
            assert_eq!(collected, messages);
        }
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
        let collected = MarkSweepGc::default().collect(messages, 120, &mut state);

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
        let collected = MarkSweepGc::default().collect(messages, 40, &mut state);

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

        let a = MarkSweepGc::default().collect(messages.clone(), 55, &mut state_a);
        let b = MarkSweepGc::default().collect(messages, 55, &mut state_b);

        assert_eq!(a, b);
    }

    #[test]
    fn ring_preserve_evicts_interior_and_keeps_the_cached_prefix() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("a".repeat(100)),
            ChatMessage::user("b".repeat(200)),
            ChatMessage::user("c".repeat(300)),
        ];
        // Budget such that the 25% prefix allowance covers system + the
        // oldest message, and something must still drop.
        let prefix_tokens = estimate_tokens(&messages[..2]);
        let budget = prefix_tokens * 4;
        assert!(
            estimate_tokens(&messages) > budget,
            "test setup: collection must be under pressure"
        );

        let mut state = GcState::default();
        let collected = RingGc {
            preserve_prefix: true,
        }
        .collect(messages.clone(), budget, &mut state);

        assert!(estimate_tokens(&collected) <= budget);
        assert_eq!(
            &collected[..2],
            &messages[..2],
            "the cached prefix must stay byte-identical"
        );
        assert!(
            collected
                .iter()
                .any(|message| message.content.as_deref() == Some(&"a".repeat(100))),
            "pinned oldest message must survive"
        );
        assert!(
            !collected
                .iter()
                .any(|message| message.content.as_deref() == Some(&"b".repeat(200))),
            "interior message should be evicted"
        );
        assert!(
            collected
                .iter()
                .any(|message| message.content.as_deref() == Some(&"c".repeat(300))),
            "live tail should survive"
        );
        assert!(!state.prefix_invalidated);
    }

    #[test]
    fn ring_ignore_mode_reports_prefix_invalidation() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("oldest"),
            ChatMessage::user("d".repeat(400)),
        ];
        let prefix_tokens = estimate_tokens(&messages[..2]);
        let budget = prefix_tokens * 4;
        assert!(estimate_tokens(&messages) > budget);

        let mut state = GcState::default();
        let collected = RingGc {
            preserve_prefix: false,
        }
        .collect(messages, budget, &mut state);

        assert!(
            state.prefix_invalidated,
            "front-drop changed the prefix region: {collected:?}"
        );
    }

    #[test]
    fn mark_sweep_preserve_does_not_touch_the_pinned_prefix() {
        let pinned_result = "x".repeat(2000);
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(None, vec![read_file_call("call-1", "/tmp/pinned.txt")]),
            // Completes the pinned pair, so it is pinned despite its size.
            ChatMessage::tool("call-1", pinned_result.clone()),
            ChatMessage::assistant(Some("incorporated pinned".into()), vec![]),
            ChatMessage::assistant(None, vec![tool_call("call-2")]),
            ChatMessage::tool("call-2", "small interior result"),
            ChatMessage::assistant(Some("incorporated interior".into()), vec![]),
            ChatMessage::user("latest"),
        ];
        // Allowance covers system + the call message; the giant result rides
        // along via pair pinning.
        let prefix_tokens = estimate_tokens(&messages[..2]);
        let budget = prefix_tokens * 4;
        assert!(estimate_tokens(&messages) > budget);

        let mut state = GcState::default();
        let collected = MarkSweepGc {
            preserve_prefix: true,
        }
        .collect(messages, budget, &mut state);

        let pinned_tool = collected
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("pinned tool result must survive");
        assert_eq!(
            pinned_tool.content.as_deref(),
            Some(pinned_result.as_str()),
            "preserve mode must not annotate inside the pinned prefix"
        );
        assert!(
            !collected
                .iter()
                .any(|message| message.tool_call_id.as_deref() == Some("call-2")),
            "interior completed pair should be evicted under pressure"
        );
        assert!(!state.prefix_invalidated);
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
        let collected = RingGc {
            preserve_prefix: false,
        }
        .collect(messages, 45, &mut state);

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
