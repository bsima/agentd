use crate::embedding::{content_hash, cosine, Embedder};
use crate::op::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
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
    /// Semantic-GC embedding cache (t-1350), keyed by message content hash
    /// ([`semantic_cache_key`]). Written ONLY by the interpreter's async
    /// pre-pass ([`SemanticGc::prime_cache`], called from
    /// `maybe_collect_prompt`); [`SemanticGc::collect`] consumes it
    /// read-only, so collect() stays a pure function of
    /// (messages, budget, state). Runtime-only: `GcState` never serializes
    /// into checkpoints, so vectors are re-embedded (or the recency
    /// heuristic applies) after a resume, and the pre-pass prunes entries
    /// whose content left the window, bounding the cache to the live
    /// window.
    pub embeddings: HashMap<String, Vec<f32>>,
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

/// Default recent-window size for [`SemanticGc`]: the last 8 messages are
/// roughly the last 3-4 user/assistant exchanges — enough to define the
/// conversation's current topic, small enough that a tangent abandoned a
/// couple of turns ago has already left the window (and so scores as
/// distant instead of anchoring the centroid to itself).
pub const DEFAULT_SEMANTIC_RECENT_WINDOW: usize = 8;

/// Default similarity floor for [`SemanticGc`]: below ~0.25 cosine on
/// typical text-embedding models two passages share almost no topic, so
/// dropping them first is safe; messages at or above the floor are only
/// dropped in the second pass, once dropping the clearly-unrelated ones
/// did not reach the budget.
pub const DEFAULT_SEMANTIC_SIMILARITY_FLOOR: f32 = 0.25;

/// Strategy 4 (docs/GC.md, t-1350): drop messages semantically distant from
/// the recent conversation — conversational dead ends and abandoned
/// tangents — lowest-similarity first, until under budget.
///
/// Each candidate is scored by cosine similarity between its cached
/// embedding and the *centroid* of the last [`Self::recent_window`]
/// messages' embeddings. Centroid over max-similarity-to-any-recent
/// because it scores against the topic of the recent thread as a whole
/// instead of rewarding lexical echo of any single recent message.
///
/// The GC invariant holds: `collect()` is stateless, deterministic, and
/// LLM-free. Embeddings are computed by an async pre-pass in the
/// interpreter ([`Self::prime_cache`], called from `maybe_collect_prompt`
/// — the layer with async + config access, the same shape the t-1166
/// design note sanctioned for memory retrieval); `collect()` consumes only
/// `GcState.embeddings`. A message with no cached vector — embedder not
/// configured, endpoint failed, resumed session — falls back to a
/// deterministic recency score, never an error and never a provider call.
#[derive(Clone)]
pub struct SemanticGc {
    /// Preserve the cached prefix: candidates must live entirely after the
    /// pinned prefix region (same boundary semantics as the other
    /// strategies).
    pub preserve_prefix: bool,
    /// The last N messages define "recent": they form the centroid AND are
    /// immune from eviction (the recency floor).
    pub recent_window: usize,
    /// Messages scoring at or above this cosine similarity are only
    /// dropped once every below-floor candidate is gone.
    pub similarity_floor: f32,
    /// Carried for the interpreter's async pre-pass ONLY
    /// ([`Self::prime_cache`]); `collect()` never touches it, which is
    /// what keeps collect() synchronous, deterministic, and offline.
    pub embedder: Option<Arc<dyn Embedder>>,
}

impl Default for SemanticGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
            recent_window: DEFAULT_SEMANTIC_RECENT_WINDOW,
            similarity_floor: DEFAULT_SEMANTIC_SIMILARITY_FLOOR,
            embedder: None,
        }
    }
}

impl std::fmt::Debug for SemanticGc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticGc")
            .field("preserve_prefix", &self.preserve_prefix)
            .field("recent_window", &self.recent_window)
            .field("similarity_floor", &self.similarity_floor)
            .field(
                "embedder",
                &self.embedder.as_ref().map(|embedder| embedder.model_id()),
            )
            .finish()
    }
}

/// The text a message is embedded as: role, content, and tool-call
/// name/arguments. Excludes the message UUID and tool-call ids so identical
/// content shares one cache entry and the key survives replay/rebuild.
pub fn message_embedding_text(message: &ChatMessage) -> String {
    use std::fmt::Write as _;
    let mut text = format!(
        "{}: {}",
        message.role,
        message.content.as_deref().unwrap_or("")
    );
    for call in message.tool_calls.as_deref().unwrap_or_default() {
        let _ = write!(text, "\n[call {} {}]", call.name, call.arguments);
    }
    text
}

/// Key into `GcState.embeddings`: content hash of the embedded text.
pub fn semantic_cache_key(message: &ChatMessage) -> String {
    content_hash(&message_embedding_text(message))
}

/// What one pre-pass did, for the optional `gc_semantic_embed` trace event.
#[derive(Debug, Clone, Copy, Default)]
pub struct SemanticPrimeReport {
    /// Vectors newly embedded this pass.
    pub embedded: usize,
    /// Cache entries already present (after pruning to the live window).
    pub cached: usize,
    /// The embed call failed; the cache is unchanged and collect() will
    /// score uncached messages with the recency heuristic.
    pub failed: bool,
}

impl SemanticGc {
    /// The async pre-pass (t-1350): embed every window message whose
    /// content hash is not yet cached, pruning entries whose content left
    /// the window (bounding the cache to the live window). Best-effort by
    /// contract — any failure leaves the cache as-is and is reported, never
    /// returned as an error, so an embedding outage can never fail a turn.
    ///
    /// Called from `interpreter::collect_prompt` after the truncate
    /// pre-pass (truncation rewrites content, and the cache keys on
    /// content). Never called by `collect()`.
    pub async fn prime_cache(
        &self,
        messages: &[ChatMessage],
        state: &mut GcState,
    ) -> SemanticPrimeReport {
        let live: BTreeSet<String> = messages.iter().map(semantic_cache_key).collect();
        state.embeddings.retain(|key, _| live.contains(key));

        let mut report = SemanticPrimeReport {
            cached: state.embeddings.len(),
            ..Default::default()
        };
        let Some(embedder) = &self.embedder else {
            return report;
        };
        let mut missing: Vec<(String, String)> = Vec::new();
        let mut seen = BTreeSet::new();
        for message in messages {
            let text = message_embedding_text(message);
            let key = content_hash(&text);
            if state.embeddings.contains_key(&key) || !seen.insert(key.clone()) {
                continue;
            }
            missing.push((key, text));
        }
        if missing.is_empty() {
            return report;
        }
        let texts: Vec<String> = missing.iter().map(|(_, text)| text.clone()).collect();
        match embedder.embed(&texts).await {
            Ok(vectors) if vectors.len() == missing.len() => {
                for ((key, _), vector) in missing.into_iter().zip(vectors) {
                    state.embeddings.insert(key, vector);
                    report.embedded += 1;
                }
            }
            // Failure = heuristic path: cache unchanged, no error.
            _ => report.failed = true,
        }
        report
    }

    /// Centroid of the cached vectors of the last `recent_window` messages.
    /// `None` when none of them have a vector (heuristic-only mode).
    /// Vectors with a mismatched dimension (an embedding-model switch
    /// mid-run) are skipped rather than mixed.
    fn recent_centroid(&self, messages: &[ChatMessage], state: &GcState) -> Option<Vec<f32>> {
        let start = messages.len().saturating_sub(self.recent_window.max(1));
        let mut sum: Option<Vec<f32>> = None;
        let mut count = 0usize;
        for message in &messages[start..] {
            let Some(vector) = state.embeddings.get(&semantic_cache_key(message)) else {
                continue;
            };
            match &mut sum {
                None => {
                    sum = Some(vector.clone());
                    count = 1;
                }
                Some(acc) if acc.len() == vector.len() => {
                    for (slot, value) in acc.iter_mut().zip(vector) {
                        *slot += value;
                    }
                    count += 1;
                }
                Some(_) => {}
            }
        }
        sum.map(|mut acc| {
            for slot in &mut acc {
                *slot /= count as f32;
            }
            acc
        })
    }

    /// Score every message: cosine to the recent centroid when both the
    /// centroid and the message's vector are cached; otherwise a
    /// deterministic recency score mapped into cosine's [-1, 1] range
    /// (older = lower), so an unvouched-for message degrades to
    /// oldest-first — ring's ordering — instead of erroring.
    fn scores(&self, messages: &[ChatMessage], state: &GcState) -> Vec<f32> {
        let centroid = self.recent_centroid(messages, state);
        let len = messages.len();
        messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                if let Some(centroid) = &centroid {
                    if let Some(vector) = state.embeddings.get(&semantic_cache_key(message)) {
                        return cosine(vector, centroid);
                    }
                }
                if len <= 1 {
                    1.0
                } else {
                    -1.0 + 2.0 * (index as f32) / ((len - 1) as f32)
                }
            })
            .collect()
    }

    /// The hard guards plus the recency floor: system messages, the last
    /// user message, everything inside the pinned prefix (preserve mode),
    /// and the last `recent_window` messages are immune from eviction.
    fn protected_mask(&self, messages: &[ChatMessage], boundary: usize) -> Vec<bool> {
        let len = messages.len();
        let floor_start = len.saturating_sub(self.recent_window);
        let last_user = messages.iter().rposition(|message| message.role == "user");
        messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                message.role == "system"
                    || index < boundary
                    || index >= floor_start
                    || Some(index) == last_user
            })
            .collect()
    }
}

/// The guards that survive even the degrade pass: the system message and
/// the last user message (the statement of the current task) are never
/// dropped, no matter the pressure.
fn semantic_hard_protected_mask(messages: &[ChatMessage]) -> Vec<bool> {
    let last_user = messages.iter().rposition(|message| message.role == "user");
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| message.role == "system" || Some(index) == last_user)
        .collect()
}

/// Drop unprotected candidates most-distant-first until under budget.
/// `floor = Some(f)` restricts the pass to candidates scoring below `f`.
/// Tool-call pairs travel atomically; a group that would pull a protected
/// message out is skipped entirely.
fn sweep_semantic(
    messages: &[ChatMessage],
    keep: &mut [bool],
    budget: usize,
    protected: &[bool],
    scores: &[f32],
    floor: Option<f32>,
) {
    if estimate_tokens(&kept_messages(messages, keep)) <= budget {
        return;
    }
    let mut order: Vec<usize> = (0..messages.len())
        .filter(|&index| !protected[index] && messages[index].role != "system")
        .collect();
    // Ascending score (most distant first); index ascending (older first)
    // breaks ties deterministically. total_cmp: no NaN surprises.
    order.sort_by(|a, b| scores[*a].total_cmp(&scores[*b]).then(a.cmp(b)));
    for index in order {
        if estimate_tokens(&kept_messages(messages, keep)) <= budget {
            break;
        }
        if !keep[index] {
            continue;
        }
        if floor.is_some_and(|floor| scores[index] >= floor) {
            continue;
        }
        if !atomic_group_avoids_protected(messages, keep, index, protected) {
            continue;
        }
        drop_atomic_group(messages, keep, index);
    }
}

/// Would dropping `index`'s atomic group (tool-call pairs travel together)
/// remove any protected message? Generalizes [`atomic_group_stays_past`]
/// from a prefix boundary to an arbitrary protection mask.
fn atomic_group_avoids_protected(
    messages: &[ChatMessage],
    keep: &[bool],
    index: usize,
    protected: &[bool],
) -> bool {
    let mut scratch = keep.to_vec();
    drop_atomic_group(messages, &mut scratch, index);
    keep.iter()
        .zip(&scratch)
        .enumerate()
        .all(|(position, (before, after))| !protected[position] || before == after)
}

impl ContextGc for SemanticGc {
    fn collect(
        &self,
        messages: Vec<ChatMessage>,
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

        let mut keep = vec![true; messages.len()];
        if estimate_tokens(&messages) > budget {
            let protected = self.protected_mask(&messages, boundary);
            let scores = self.scores(&messages, state);
            // Phase 1: clearly-unrelated candidates (below the floor),
            // most distant first.
            sweep_semantic(
                &messages,
                &mut keep,
                budget,
                &protected,
                &scores,
                Some(self.similarity_floor),
            );
            // Phase 2: any candidate, most distant first.
            sweep_semantic(&messages, &mut keep, budget, &protected, &scores, None);
            // Phase 3 (degrade): the protected set alone exceeds the
            // budget. Relax the recency floor and the prefix pin —
            // overflowing the model is worse than a cache miss (same
            // stance as ring's front-drop fallback; the invalidation is
            // reported via prefix_invalidated) — but the system message
            // and the last user message stay hard-protected.
            if estimate_tokens(&kept_messages(&messages, &keep)) > budget {
                let hard = semantic_hard_protected_mask(&messages);
                sweep_semantic(&messages, &mut keep, budget, &hard, &scores, None);
            }
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
        "semantic"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

#[derive(Debug, Clone)]
pub enum GcMode {
    None,
    Ring(RingGc),
    MarkSweep(MarkSweepGc),
    Stack(StackFrameGc),
    Semantic(SemanticGc),
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
            Self::Semantic(gc) => gc.collect(messages, budget, state),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ring(gc) => gc.name(),
            Self::MarkSweep(gc) => gc.name(),
            Self::Stack(gc) => gc.name(),
            Self::Semantic(gc) => gc.name(),
        }
    }

    pub fn cache_preserving(&self) -> bool {
        match self {
            Self::None => true,
            Self::Ring(gc) => gc.cache_preserving(),
            Self::MarkSweep(gc) => gc.cache_preserving(),
            Self::Stack(gc) => gc.cache_preserving(),
            Self::Semantic(gc) => gc.cache_preserving(),
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

    // ---- SemanticGc (t-1350) ------------------------------------------------

    /// Cache a unit-basis vector for `message`: `axis` 0/1/2 give three
    /// mutually-orthogonal topics, so cosine to a recent centroid on axis 0
    /// is 1.0 for on-topic messages and 0.0 for tangents.
    fn cache_axis(state: &mut GcState, message: &ChatMessage, axis: usize) {
        let mut vector = vec![0.0f32; 3];
        vector[axis] = 1.0;
        state.embeddings.insert(semantic_cache_key(message), vector);
    }

    /// system + task + old on-topic exchange + tangent exchange + recent
    /// on-topic tail. The tangent is NEWER than the on-topic history: a
    /// recency-only strategy drops the wrong messages, semantic must not.
    fn semantic_fixture(state: &mut GcState) -> Vec<ChatMessage> {
        let messages = vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user(format!("fix the query planner {}", "p".repeat(120))),
            ChatMessage::assistant(Some(format!("old on-topic {}", "q".repeat(200))), vec![]),
            ChatMessage::assistant(
                Some(format!("tangent cache idea {}", "r".repeat(200))),
                vec![],
            ),
            ChatMessage::assistant(Some(format!("more tangent {}", "s".repeat(200))), vec![]),
            ChatMessage::user("back to the planner"),
            ChatMessage::assistant(Some("statistics refreshed".into()), vec![]),
        ];
        for (index, message) in messages.iter().enumerate() {
            let axis = usize::from(matches!(index, 3 | 4));
            cache_axis(state, message, axis);
        }
        messages
    }

    fn semantic_gc() -> SemanticGc {
        SemanticGc {
            preserve_prefix: false,
            recent_window: 2,
            similarity_floor: 0.25,
            embedder: None,
        }
    }

    #[test]
    fn semantic_drops_the_distant_tangent_before_older_on_topic_history() {
        let mut state = GcState::default();
        let messages = semantic_fixture(&mut state);
        let tangent_ids = [messages[3].id, messages[4].id];
        let on_topic_id = messages[2].id;
        let budget = estimate_tokens(&messages) - 100;

        let collected = semantic_gc().collect(messages, budget, &mut state);

        assert!(estimate_tokens(&collected) <= budget);
        assert!(
            !collected
                .iter()
                .any(|message| tangent_ids.contains(&message.id)),
            "the semantically distant tangent goes first: {collected:?}"
        );
        assert!(
            collected.iter().any(|message| message.id == on_topic_id),
            "older but on-topic history outlives the newer tangent"
        );
    }

    #[test]
    fn semantic_never_drops_system_last_user_or_recency_floor() {
        let mut state = GcState::default();
        let messages = semantic_fixture(&mut state);
        let last_user = messages[5].id;
        let tail = messages[6].id;
        // Heavy pressure: everything unprotected must go.
        let budget = estimate_tokens(&messages[..1]) + 60;

        let collected = semantic_gc().collect(messages, budget, &mut state);

        assert!(collected.iter().any(|message| message.role == "system"));
        assert!(
            collected.iter().any(|message| message.id == last_user),
            "the last user message is hard-protected: {collected:?}"
        );
        assert!(
            collected.iter().any(|message| message.id == tail),
            "the recency floor keeps the live tail: {collected:?}"
        );
    }

    #[test]
    fn semantic_without_cached_vectors_degrades_to_oldest_first() {
        // No embeddings cached at all (no embedder configured): the
        // deterministic recency heuristic drops oldest-first, like ring.
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("a".repeat(400)),
            ChatMessage::user("b".repeat(400)),
            ChatMessage::user("c".repeat(400)),
        ];
        let mut state = GcState::default();
        let budget = 300;

        let collected = semantic_gc().collect(messages, budget, &mut state);

        assert!(estimate_tokens(&collected) <= budget);
        assert!(collected.iter().any(|message| message
            .content
            .as_deref()
            .is_some_and(|content| content.starts_with('c'))));
        assert!(!collected.iter().any(|message| message
            .content
            .as_deref()
            .is_some_and(|content| content.starts_with('a'))));
    }

    #[test]
    fn semantic_keeps_tool_call_pairs_atomic() {
        let mut state = GcState::default();
        let call = ChatMessage::assistant(
            Some("tangent tool step".into()),
            vec![tool_call("call-tangent")],
        );
        let result = ChatMessage::tool("call-tangent", "tangent output ".repeat(40));
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("fix the planner"),
            call,
            result,
            ChatMessage::user("back to the planner"),
            ChatMessage::assistant(Some("on it".into()), vec![]),
        ];
        for (index, message) in messages.iter().enumerate() {
            cache_axis(&mut state, message, usize::from(matches!(index, 2 | 3)));
        }
        let budget = estimate_tokens(&messages) - 50;

        let collected = semantic_gc().collect(messages, budget, &mut state);

        assert!(estimate_tokens(&collected) <= budget);
        assert!(
            !collected.iter().any(|message| {
                message.tool_call_id.as_deref() == Some("call-tangent")
                    || message
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| calls.iter().any(|call| call.id == "call-tangent"))
            }),
            "the tangent frame drops call and result together: {collected:?}"
        );
    }

    #[test]
    fn semantic_is_deterministic_with_full_partial_and_empty_caches() {
        let build = |fill: fn(usize) -> bool| {
            let mut state = GcState::default();
            let messages = semantic_fixture(&mut GcState::default());
            for (index, message) in messages.iter().enumerate() {
                if fill(index) {
                    cache_axis(&mut state, message, usize::from(matches!(index, 3 | 4)));
                }
            }
            (messages, state)
        };
        for fill in [
            (|_| true) as fn(usize) -> bool,
            |index| index % 2 == 0,
            |_| false,
        ] {
            let (messages, mut state_a) = build(fill);
            let budget = estimate_tokens(&messages) - 100;
            let first = semantic_gc().collect(messages.clone(), budget, &mut state_a);
            let mut state_b = GcState {
                embeddings: state_a.embeddings.clone(),
                ..Default::default()
            };
            let replayed = semantic_gc().collect(messages, budget, &mut state_b);
            assert_eq!(
                first, replayed,
                "same window + same cached vectors = identical collection"
            );
            let again = semantic_gc().collect(first.clone(), budget, &mut state_a);
            assert_eq!(first, again, "collecting collected output is a no-op");
        }
    }

    #[test]
    fn semantic_under_budget_is_untouched() {
        let mut state = GcState::default();
        let messages = semantic_fixture(&mut state);

        let collected = semantic_gc().collect(messages.clone(), 100_000, &mut state);

        assert_eq!(collected, messages);
        assert!(!state.prefix_invalidated);
    }

    #[test]
    fn semantic_preserve_mode_pins_the_prefix() {
        let mut state = GcState::default();
        let messages = semantic_fixture(&mut state);
        let budget = estimate_tokens(&messages) - 100;

        let gc = SemanticGc {
            preserve_prefix: true,
            ..semantic_gc()
        };
        let collected = gc.collect(messages.clone(), budget, &mut state);

        assert!(estimate_tokens(&collected) <= budget);
        let boundary = cache_prefix_boundary(&messages, budget);
        assert_eq!(
            &collected[..boundary],
            &messages[..boundary],
            "the cached prefix must stay byte-identical"
        );
        assert!(!state.prefix_invalidated);
    }

    /// Deterministic mock embedder for the pre-pass tests: axis by keyword,
    /// flippable to failure.
    struct AxisEmbedder {
        fail: std::sync::atomic::AtomicBool,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl AxisEmbedder {
        fn new() -> Self {
            Self {
                fail: std::sync::atomic::AtomicBool::new(false),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl Embedder for AxisEmbedder {
        async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.fail.load(std::sync::atomic::Ordering::SeqCst) {
                anyhow::bail!("embedding endpoint down");
            }
            Ok(texts
                .iter()
                .map(|text| {
                    if text.contains("tangent") {
                        vec![0.0, 1.0, 0.0]
                    } else {
                        vec![1.0, 0.0, 0.0]
                    }
                })
                .collect())
        }

        fn model_id(&self) -> &str {
            "axis-embedder"
        }
    }

    #[tokio::test]
    async fn semantic_prime_cache_embeds_missing_prunes_stale_and_reuses() {
        let embedder = Arc::new(AxisEmbedder::new());
        let gc = SemanticGc {
            embedder: Some(embedder.clone()),
            ..semantic_gc()
        };
        let mut state = GcState::default();
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("fix the planner"),
        ];

        let report = gc.prime_cache(&messages, &mut state).await;
        assert_eq!(report.embedded, 2);
        assert!(!report.failed);
        assert_eq!(state.embeddings.len(), 2);

        // Second pass: everything cached, no embed call.
        let calls_before = embedder.calls.load(std::sync::atomic::Ordering::SeqCst);
        let report = gc.prime_cache(&messages, &mut state).await;
        assert_eq!(report.embedded, 0);
        assert_eq!(report.cached, 2);
        assert_eq!(
            embedder.calls.load(std::sync::atomic::Ordering::SeqCst),
            calls_before,
            "fully-cached windows must not call the embedder"
        );

        // A message that left the window is pruned from the cache.
        let report = gc.prime_cache(&messages[..1], &mut state).await;
        assert_eq!(state.embeddings.len(), 1);
        assert_eq!(report.cached, 1);
    }

    #[tokio::test]
    async fn semantic_prime_cache_failure_is_reported_not_raised() {
        let embedder = Arc::new(AxisEmbedder::new());
        embedder
            .fail
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let gc = SemanticGc {
            embedder: Some(embedder),
            ..semantic_gc()
        };
        let mut state = GcState::default();
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("a".repeat(400)),
            ChatMessage::user("b".repeat(400)),
        ];

        let report = gc.prime_cache(&messages, &mut state).await;
        assert!(report.failed);
        assert!(state.embeddings.is_empty(), "failed embed caches nothing");

        // collect() still works — heuristic path, no provider call possible.
        let collected = gc.collect(messages, 250, &mut state);
        assert!(estimate_tokens(&collected) <= 250);
    }

    #[tokio::test]
    async fn semantic_prime_cache_without_embedder_is_heuristic_only() {
        let gc = semantic_gc();
        let mut state = GcState::default();
        let messages = vec![ChatMessage::user("hello")];
        let report = gc.prime_cache(&messages, &mut state).await;
        assert_eq!(report.embedded, 0);
        assert!(!report.failed);
        assert!(state.embeddings.is_empty());
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
