use crate::embedding::{content_hash, cosine, Embedder};
use crate::op::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

#[derive(Debug, Default, Clone)]
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
    /// Re-injection write-barrier signal (t-1351, chunk-normalized in
    /// t-1362): normalized chunk keys ([`reinjection_chunk_keys`]) marked
    /// HOT because a tool result re-injected content previously collected
    /// from the window (a re-run call returning the payload an evicted
    /// call returned, or a `recall` hit re-rendering evicted material —
    /// or, for recalls, content still in the window). Written by
    /// [`record_reinjection_overlaps`] in the interpreter pre-pass;
    /// consumed by every strategy's `hot-keep` sweep guard (t-1362) and
    /// observable via `recall_overlap_events`/`recall_hot`/`hot_kept` on
    /// the gc_collect event. Runtime-only like `embeddings`: never
    /// serialized into checkpoints.
    pub recall_hot: BTreeSet<String>,
    /// Normalized chunk keys of content removed by earlier collections
    /// this run — dropped messages and in-place rewrites (mark-sweep
    /// elision, stack frame pops) alike. Written by the eviction-marker
    /// wrapper inside `collect()` itself (t-1362; every caller that
    /// threads a GcState gets the corpus for free), so a later result that
    /// re-injects *collected* content registers as a write-barrier event —
    /// the "evict, re-fetch, re-evict" loop is exactly what the hot signal
    /// exists to expose. Bounded by the run's own drop history;
    /// runtime-only, never serialized.
    pub collected_hashes: BTreeSet<String>,
    /// What the most recent collect() did for the hot set (t-1362). Set by
    /// the eviction-marker wrapper on every collection, like
    /// `marker_summary`; read for gc_collect trace events.
    pub hot_report: HotKeepReport,
    /// Per-content eviction counts (t-1370), keyed by
    /// [`content_fingerprint`] — how many times THIS content (under any
    /// call id or envelope) has been evicted this run. Written by the
    /// eviction-marker wrapper alongside `collected_hashes`; read by the
    /// marker builder to escalate after
    /// [`EVICTION_ESCALATION_AFTER`] evictions. Deterministic (pure
    /// function of the collection sequence), runtime-only, never
    /// serialized.
    pub eviction_counts: BTreeMap<String, u32>,
    /// What the most recent collect() left behind as eviction markers
    /// (t-1360). Set by every strategy on every collection, like
    /// `prefix_invalidated`; read for gc_collect trace events.
    pub marker_summary: EvictionMarkerSummary,
    /// Any collection this run has actually removed or rewritten window
    /// content (t-1373). Gates the progress ledger: until something has
    /// been evicted, the window itself is the complete work record and a
    /// digest would be noise — an under-budget collect() stays a no-op
    /// (the t-1371 lesson generalized: GC that reclaims nothing must stay
    /// invisible). Set by the bookkeeping wrapper's finish pass.
    pub evictions_seen: bool,
    /// The progress-ledger journal (t-1373): every completed tool call this
    /// run has ever observed at a collection, in first-completed order —
    /// call id, tool, args preview, outcome preview (normalized-payload
    /// first meaningful line, the write-barrier machinery), and content
    /// fingerprint. Written by the bookkeeping wrapper inside `collect()`
    /// before the core runs (so a result elided or dropped by the same
    /// collection is journaled from its raw content); read by the ledger
    /// renderer. Deterministic (pure function of the collection sequence),
    /// runtime-only, never serialized — like `eviction_counts`.
    pub ledger: Vec<LedgerEntry>,
    /// What the most recent collect() did for the progress ledger
    /// (t-1373). Set by the bookkeeping wrapper on every collection, like
    /// `marker_summary`; read for gc_collect trace events.
    pub ledger_summary: LedgerSummary,
    /// What the most recent GenerationalGc collect() decided (t-1167):
    /// per-tier assignment counts, per-tier elide/evict counts, and which
    /// degrade rungs fired. Set by [`GenerationalGc`] on every collection
    /// (left at default by every other strategy); read for the gc_collect
    /// trace event's `tiers` object.
    pub tier_report: GenerationalReport,
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
    /// cannot reach the budget. Under either policy the hard guards hold:
    /// the system message and the last user message survive every phase
    /// (t-1367, docs/GC.md invariants).
    pub preserve_prefix: bool,
    /// Hot-keep (t-1362): messages carrying write-barrier-hot content
    /// ([`hot_mask`]) join the protected set during the normal sweep
    /// phase — a value the model re-fetched or recalled after eviction
    /// stops being evictable. Heuristic guard with cited-keep strength:
    /// relaxes in the degrade phases, below the preserve-prefix billing
    /// contract and the hard guards.
    pub hot_keep: bool,
}

impl Default for RingGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
            hot_keep: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MarkSweepGc {
    /// Preserve the cached prefix: only annotate/evict messages after the
    /// pinned prefix region.
    pub preserve_prefix: bool,
    /// Hot-keep (t-1362): hot messages are neither elided in place nor
    /// swept during the normal lifecycle passes; relaxes when the passes
    /// alone cannot reach the budget (see [`RingGc::hot_keep`]).
    pub hot_keep: bool,
}

impl Default for MarkSweepGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
            hot_keep: true,
        }
    }
}

/// Strategy 3 (docs/GC.md): model each tool invocation+result as an
/// activation frame. When over budget, pop completed frames oldest-first:
/// the assistant tool-call message is rewritten in place to a one-line
/// `[frame call-id: tool(args) -> result]` annotation (keeping its stable
/// id; a truncated preview carries an explicit "evicted; re-run to
/// recover" clause, t-1360) and the tool result messages are dropped. The semantic record survives at
/// ~1% of the tokens, which is why this is the space-efficient choice for
/// tool-heavy agents. Summaries are pure heuristics — no LLM calls (the
/// `stack-smart` variant is gated on the eval harness). The ring-fallback
/// phases carry the hard guards: the system message and the last user
/// message survive every phase (t-1367, docs/GC.md invariants).
#[derive(Debug, Clone, Copy)]
pub struct StackFrameGc {
    /// Preserve the cached prefix: only pop frames living entirely after
    /// the pinned prefix region.
    pub preserve_prefix: bool,
    /// Hot-keep (t-1362): frames whose members carry write-barrier-hot
    /// content resist popping, and hot messages join the interior sweep's
    /// protected set; relaxes in the degrade phases (see
    /// [`RingGc::hot_keep`]). Popping destroys the result body — exactly
    /// the loss a re-fetched value must not suffer twice.
    pub hot_keep: bool,
}

impl Default for StackFrameGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
            hot_keep: true,
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
    /// The `cited-keep` modifier (t-1351, docs/GC.md "Citation signals"):
    /// messages cited by later ones ([`cited_mask`]) join the protected set
    /// during the normal sweep phases, so a cited-but-semantically-distant
    /// result — the 2x2's gap cell — survives. Citation is a heuristic
    /// guard with the same strength as the recency floor: it relaxes in the
    /// degrade phases, below the preserve-prefix billing contract and the
    /// system/last-user hard guards. Extraction is pure text analysis and
    /// runs inside collect() (unlike embeddings — no cache, no pre-pass).
    pub cited_keep: bool,
    /// Hot-keep (t-1362): hot messages join the protected set alongside
    /// cited ones during the normal sweep phases, with the same strength
    /// (relaxed in the degrade phases; see [`RingGc::hot_keep`]).
    pub hot_keep: bool,
}

impl Default for SemanticGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
            recent_window: DEFAULT_SEMANTIC_RECENT_WINDOW,
            similarity_floor: DEFAULT_SEMANTIC_SIMILARITY_FLOOR,
            embedder: None,
            cited_keep: true,
            hot_keep: true,
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
            .field("cited_keep", &self.cited_keep)
            .field("hot_keep", &self.hot_keep)
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

/// The async embedding pre-pass shared by the embedding-consuming
/// strategies (t-1350 semantic; t-1167 generational's warm-by-similarity
/// signal): embed every window message whose content hash is not yet
/// cached, pruning entries whose content left the window (bounding the
/// cache to the live window). Best-effort by contract — any failure leaves
/// the cache as-is and is reported, never returned as an error, so an
/// embedding outage can never fail a turn.
///
/// Called from `interpreter::collect_prompt` after the truncate pre-pass
/// (truncation rewrites content, and the cache keys on content). Never
/// called by `collect()`.
pub async fn prime_embedding_cache(
    embedder: Option<&Arc<dyn Embedder>>,
    messages: &[ChatMessage],
    state: &mut GcState,
) -> SemanticPrimeReport {
    let live: BTreeSet<String> = messages.iter().map(semantic_cache_key).collect();
    state.embeddings.retain(|key, _| live.contains(key));

    let mut report = SemanticPrimeReport {
        cached: state.embeddings.len(),
        ..Default::default()
    };
    let Some(embedder) = embedder else {
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

/// Centroid of the cached vectors of the last `window` messages. `None`
/// when none of them have a vector (heuristic-only mode). Vectors with a
/// mismatched dimension (an embedding-model switch mid-run) are skipped
/// rather than mixed.
fn cached_recent_centroid(
    messages: &[ChatMessage],
    state: &GcState,
    window: usize,
) -> Option<Vec<f32>> {
    let start = messages.len().saturating_sub(window.max(1));
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

impl SemanticGc {
    /// The async pre-pass (t-1350): see [`prime_embedding_cache`].
    pub async fn prime_cache(
        &self,
        messages: &[ChatMessage],
        state: &mut GcState,
    ) -> SemanticPrimeReport {
        prime_embedding_cache(self.embedder.as_ref(), messages, state).await
    }

    /// Centroid of the cached vectors of the last `recent_window` messages.
    fn recent_centroid(&self, messages: &[ChatMessage], state: &GcState) -> Option<Vec<f32>> {
        cached_recent_centroid(messages, state, self.recent_window)
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

/// The guards that survive even the degrade passes of EVERY strategy
/// (docs/GC.md invariants, t-1367): the system message and the last user
/// message (the statement of the current task) are never dropped, no
/// matter the pressure. t-1364 proved the failure mode this bans: ring's
/// and stack's front-drop degrade path evicted the live task, the model
/// answered "I'm ready to help!", and the loop accepted that as final.
/// If even this protected set exceeds the budget, collection returns an
/// over-budget window and the overflow paths (t-1343 backstop,
/// catch-overflow) own the outcome — the same terminal case semantic has
/// always had.
fn hard_protected_mask(messages: &[ChatMessage]) -> Vec<bool> {
    let last_user = messages.iter().rposition(|message| message.role == "user");
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| message.role == "system" || Some(index) == last_user)
        .collect()
}

/// The hard guards plus the pinned cache prefix: what preserve-mode sweep
/// phases must not touch. `boundary` 0 (ignore mode) leaves the hard
/// guards alone.
fn protected_with_prefix(messages: &[ChatMessage], boundary: usize) -> Vec<bool> {
    let mut mask = hard_protected_mask(messages);
    for slot in mask.iter_mut().take(boundary) {
        *slot = true;
    }
    mask
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
        with_window_bookkeeping(messages, budget, state, |messages, budget, state| {
            self.collect_inner(messages, budget, state)
        })
    }

    fn name(&self) -> &'static str {
        "semantic"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

impl SemanticGc {
    fn collect_inner(
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
            let mut protected = self.protected_mask(&messages, boundary);
            // cited-keep (t-1351): cited messages are protected through the
            // normal sweep phases. They are NOT in the phase-3/4 masks, so
            // like the recency floor they relax under degrade pressure —
            // heuristic guard, not a hard one. Pure text analysis, so it
            // runs right here inside collect().
            if self.cited_keep {
                for (slot, cited) in protected.iter_mut().zip(cited_mask(&messages)) {
                    *slot = *slot || cited;
                }
            }
            // hot-keep (t-1362): write-barrier-hot messages join the
            // protected set with cited-keep's strength — protected through
            // the normal sweep phases, relaxed in phases 3/4.
            if self.hot_keep {
                for (slot, hot) in protected.iter_mut().zip(hot_mask(&messages, state)) {
                    *slot = *slot || hot;
                }
            }
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
            // Phase 3 (degrade, floor first): the protected set alone
            // exceeds the budget. Relax the recency floor but keep the
            // prefix pin — the floor is a heuristic guard while preserve
            // mode's stable prefix is a billing contract (the
            // gc_cache_preserve gate holds every strategy to it) — with
            // system + last user still hard-protected.
            if estimate_tokens(&kept_messages(&messages, &keep)) > budget {
                let floor_relaxed = protected_with_prefix(&messages, boundary);
                sweep_semantic(&messages, &mut keep, budget, &floor_relaxed, &scores, None);
            }
            // Phase 4 (degrade, prefix last): even the pinned prefix plus
            // system + last user exceed the budget. Overflowing the model
            // is worse than a cache miss (ring's front-drop stance; the
            // invalidation is reported via prefix_invalidated); system and
            // the last user message are never dropped.
            if estimate_tokens(&kept_messages(&messages, &keep)) > budget {
                let hard = hard_protected_mask(&messages);
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
}

// --- Generational GC (t-1167) -----------------------------------------------
//
// The synthesis strategy, designed against six generations of behavioral
// evidence (docs/GC.md "Strategy 5: Generational" carries the full
// failure-mode -> tier mapping; evals/gc/README.md is the corpus). Every
// window message is tiered from live signals at collect() time — nursery
// (recency floor), hot (recall write-barrier ∪ hard guards ∪ escalated
// content), warm (citation in-degree ∪ centroid-near when embeddings are
// cached), cold (the unvouched remainder) — and the reclaim follows
// mark-sweep's behaviorally-validated annotate-don't-drop shape: cold
// bodies ELIDE in place to one-line annotations before anything is
// whole-evicted, and true deletion (markers/ledger accounted, via the
// shared bookkeeping wrapper) is the last resort. Unlike mark-sweep, the
// phases continue into the established degrade ladder, so convergence is
// asserted (not best-effort) — which also leaves the slack that funds the
// progress ledger (the t-1362 honest-ceiling note).

/// Default nursery size for [`GenerationalGc`]: the recency floor. Same
/// value and rationale as [`DEFAULT_SEMANTIC_RECENT_WINDOW`] — the last 8
/// messages are the live working set, including the model's own
/// step-completion narration (evicting that narration is the restart-loop
/// trigger, t-1349 finding 2).
pub const DEFAULT_NURSERY_WINDOW: usize = 8;

/// Default warm-by-similarity floor for [`GenerationalGc`], active only
/// when embeddings are cached. Same value and rationale as
/// [`DEFAULT_SEMANTIC_SIMILARITY_FLOOR`]: below ~0.25 cosine two passages
/// share almost no topic.
pub const DEFAULT_WARM_SIMILARITY_FLOOR: f32 = 0.25;

/// One message's generation (t-1167), assigned deterministically inside
/// collect() from live signals — no cross-turn tier state to corrupt:
/// promotion (cold -> warm on first citation, any -> hot on
/// re-acquisition) and demotion happen implicitly on recomputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GcTier {
    /// The recency floor: the last `nursery_window` messages. Never
    /// collected in any normal phase; falls only on the terminal degrade
    /// rungs every strategy shares.
    Nursery,
    /// The hard guards (system, last user), write-barrier-hot content
    /// (`GcState.recall_hot` — re-acquired after eviction, t-1362), and
    /// in-window results whose content already escalated
    /// ([`EVICTION_ESCALATION_AFTER`] evictions, t-1370): protected
    /// through every normal phase, relaxed only per the hot-keep ladder.
    Hot,
    /// Cited by a later window message (CitationGraph in-degree, t-1351)
    /// or semantically near the nursery centroid (>= `warm_floor`, when
    /// embeddings are cached). Bodies may elide; structure survives the
    /// normal phases.
    Warm,
    /// The unvouched remainder: uncited, distant or unscoreable, not hot,
    /// not recent. First to elide, first to evict.
    Cold,
}

/// What one GenerationalGc collection decided (t-1167), reported on the
/// gc_collect trace event as the `tiers` object: assignment counts,
/// per-tier elide/evict counts, and which degrade rungs fired. The
/// evicted-per-tier counts are computed from the final keep decisions
/// against the tier assignment — a mechanism-vs-accounting cross-check
/// the eval harness asserts on (e.g. `evicted_nursery` must be 0 unless
/// `floor_relaxed`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationalReport {
    /// Tier assignment counts for the pre-collection window.
    pub nursery: usize,
    pub hot: usize,
    pub warm: usize,
    pub cold: usize,
    /// Tool-result bodies elided in place, per phase.
    pub cold_elided: usize,
    pub warm_elided: usize,
    pub hot_elided: usize,
    /// Whole messages evicted, attributed to their assigned tier.
    pub evicted_cold: usize,
    pub evicted_warm: usize,
    pub evicted_hot: usize,
    pub evicted_nursery: usize,
    /// Degrade rungs, in ladder order: warm eviction, hot relax, nursery
    /// floor relax, prefix relax (the billing contract falls last; the
    /// hard guards never do).
    pub warm_relaxed: bool,
    pub hot_relaxed: bool,
    pub floor_relaxed: bool,
    pub prefix_relaxed: bool,
}

/// Strategy 5 (docs/GC.md, t-1167): hot/warm/cold tiered collection with a
/// nursery recency floor, consuming the citation graph (t-1351), the
/// re-injection write-barrier (t-1362), and the escalation counts
/// (t-1370) as tier membership rather than bolt-on masks. Collection
/// policy: cold elides, then cold evicts, then warm elides; the degrade
/// ladder (warm evict -> hot relax -> floor relax -> prefix relax) runs
/// only when the vouched-for window alone exceeds the budget. The GC
/// invariant holds: collect() is stateless, deterministic, and LLM-free —
/// embeddings arrive via the shared async pre-pass
/// ([`prime_embedding_cache`]) and are consumed read-only; without them
/// the warm tier is citation-only (strategy-honest, documented).
#[derive(Clone)]
pub struct GenerationalGc {
    /// Preserve the cached prefix: same boundary semantics as every other
    /// strategy — elision and eviction are restricted to the interior
    /// until the prefix-relax degrade rung.
    pub preserve_prefix: bool,
    /// The recency floor: the last N messages are the nursery.
    pub nursery_window: usize,
    /// Warm-by-similarity floor (cosine against the nursery centroid),
    /// active only for messages with cached embeddings.
    pub warm_floor: f32,
    /// Carried for the interpreter's async pre-pass ONLY
    /// ([`Self::prime_cache`]); `collect()` never touches it. None =
    /// citation-only warm tier.
    pub embedder: Option<Arc<dyn Embedder>>,
}

impl Default for GenerationalGc {
    fn default() -> Self {
        Self {
            preserve_prefix: true,
            nursery_window: DEFAULT_NURSERY_WINDOW,
            warm_floor: DEFAULT_WARM_SIMILARITY_FLOOR,
            embedder: None,
        }
    }
}

impl std::fmt::Debug for GenerationalGc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationalGc")
            .field("preserve_prefix", &self.preserve_prefix)
            .field("nursery_window", &self.nursery_window)
            .field("warm_floor", &self.warm_floor)
            .field(
                "embedder",
                &self.embedder.as_ref().map(|embedder| embedder.model_id()),
            )
            .finish()
    }
}

impl GenerationalGc {
    /// The async pre-pass: see [`prime_embedding_cache`]. Without an
    /// embedder this still prunes the cache to the live window.
    pub async fn prime_cache(
        &self,
        messages: &[ChatMessage],
        state: &mut GcState,
    ) -> SemanticPrimeReport {
        prime_embedding_cache(self.embedder.as_ref(), messages, state).await
    }

    /// Warm-by-similarity scores: cosine against the nursery centroid for
    /// messages with cached vectors, `None` otherwise (never a provider
    /// call — an unscoreable message simply cannot be warm by similarity).
    fn warm_scores(&self, messages: &[ChatMessage], state: &GcState) -> Vec<Option<f32>> {
        let Some(centroid) = cached_recent_centroid(messages, state, self.nursery_window) else {
            return vec![None; messages.len()];
        };
        messages
            .iter()
            .map(|message| {
                state
                    .embeddings
                    .get(&semantic_cache_key(message))
                    .map(|vector| cosine(vector, &centroid))
            })
            .collect()
    }

    /// Assign every window message a tier (t-1167), in precedence order:
    /// nursery (recency floor) > hot (hard guards ∪ write-barrier ∪
    /// escalated content) > warm (cited ∪ centroid-near) > cold. Pure and
    /// deterministic — exactly the inputs collect() already has.
    pub fn tiers(&self, messages: &[ChatMessage], state: &GcState) -> Vec<GcTier> {
        let len = messages.len();
        let nursery_start = len.saturating_sub(self.nursery_window.max(1));
        let last_user = messages.iter().rposition(|message| message.role == "user");
        let hot = hot_mask(messages, state);
        let cited = cited_mask(messages);
        let scores = self.warm_scores(messages, state);
        messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                if index >= nursery_start {
                    return GcTier::Nursery;
                }
                if message.role == "system" || Some(index) == last_user || hot[index] {
                    return GcTier::Hot;
                }
                // Escalated-content protection (t-1370 cost accounting): a
                // result whose content already escalated and is BACK in
                // the window was re-fetched against the honest-exit
                // marker; a further eviction is pure thrash.
                let escalated = message.role == "tool"
                    && message.content.as_deref().is_some_and(|content| {
                        !content.trim_start().starts_with(EVICTION_MARKER_PREFIX)
                            && state
                                .eviction_counts
                                .get(&content_fingerprint(content))
                                .copied()
                                .unwrap_or(0)
                                >= EVICTION_ESCALATION_AFTER
                    });
                if escalated {
                    return GcTier::Hot;
                }
                if cited[index] {
                    return GcTier::Warm;
                }
                if scores[index].is_some_and(|score| score >= self.warm_floor) {
                    return GcTier::Warm;
                }
                GcTier::Cold
            })
            .collect()
    }

    fn collect_inner(
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

        let tiers = self.tiers(&messages, state);
        let mut report = GenerationalReport::default();
        for tier in &tiers {
            match tier {
                GcTier::Nursery => report.nursery += 1,
                GcTier::Hot => report.hot += 1,
                GcTier::Warm => report.warm += 1,
                GcTier::Cold => report.cold += 1,
            }
        }
        let tier_mask =
            |wanted: GcTier| -> Vec<bool> { tiers.iter().map(|tier| *tier == wanted).collect() };

        let mut keep = vec![true; messages.len()];
        let over = |messages: &[ChatMessage], keep: &[bool]| {
            estimate_tokens(&kept_messages(messages, keep)) > budget
        };
        if over(&messages, &keep) {
            // Phase 1: cold bodies elide in place (annotate, don't drop —
            // the behavioral north star; structure and handles survive).
            report.cold_elided = elide_tool_results(
                &mut messages,
                &keep,
                budget,
                &tier_mask(GcTier::Cold),
                boundary,
                &state.eviction_counts,
            );
            // Phase 2: whole cold groups evict, oldest first (deletion is
            // the cold tier's last resort, and it feeds markers + ledger
            // via the bookkeeping wrapper).
            if over(&messages, &keep) {
                let mut protected = protected_with_prefix(&messages, boundary);
                for (slot, tier) in protected.iter_mut().zip(&tiers) {
                    *slot = *slot || *tier != GcTier::Cold;
                }
                sweep_ring(&messages, &mut keep, budget, &protected);
            }
            // Phase 3: warm bodies elide; warm structure stays (the
            // citation handle survives in the annotation).
            if over(&messages, &keep) {
                report.warm_elided = elide_tool_results(
                    &mut messages,
                    &keep,
                    budget,
                    &tier_mask(GcTier::Warm),
                    boundary,
                    &state.eviction_counts,
                );
            }
            // Degrade rung a: warm evicts (cited-keep strength relaxes).
            if over(&messages, &keep) {
                report.warm_relaxed = true;
                let mut protected = protected_with_prefix(&messages, boundary);
                for (slot, tier) in protected.iter_mut().zip(&tiers) {
                    *slot = *slot || matches!(tier, GcTier::Nursery | GcTier::Hot);
                }
                sweep_ring(&messages, &mut keep, budget, &protected);
            }
            // Degrade rung b: hot relaxes (hot-keep strength) — elide
            // first, then evict; the nursery is still untouched.
            if over(&messages, &keep) {
                report.hot_relaxed = true;
                report.hot_elided = elide_tool_results(
                    &mut messages,
                    &keep,
                    budget,
                    &tier_mask(GcTier::Hot),
                    boundary,
                    &state.eviction_counts,
                );
                if over(&messages, &keep) {
                    let mut protected = protected_with_prefix(&messages, boundary);
                    for (slot, tier) in protected.iter_mut().zip(&tiers) {
                        *slot = *slot || *tier == GcTier::Nursery;
                    }
                    sweep_ring(&messages, &mut keep, budget, &protected);
                }
            }
            // Degrade rung c: the nursery floor relaxes (semantic's
            // phase-3 precedent); the prefix pin — a billing contract —
            // still holds.
            if over(&messages, &keep) {
                report.floor_relaxed = true;
                sweep_ring(
                    &messages,
                    &mut keep,
                    budget,
                    &protected_with_prefix(&messages, boundary),
                );
            }
            // Degrade rung d: the pinned prefix falls last (overflowing
            // the model is worse than a cache miss; the invalidation is
            // reported); system + last user never drop.
            if boundary > 0 && over(&messages, &keep) {
                report.prefix_relaxed = true;
                sweep_ring(
                    &messages,
                    &mut keep,
                    budget,
                    &hard_protected_mask(&messages),
                );
            }
        }

        // Evicted-per-tier accounting from the final keep decisions — the
        // mechanism-vs-accounting cross-check the eval harness asserts on.
        for (index, kept) in keep.iter().enumerate() {
            if !kept {
                match tiers[index] {
                    GcTier::Nursery => report.evicted_nursery += 1,
                    GcTier::Hot => report.evicted_hot += 1,
                    GcTier::Warm => report.evicted_warm += 1,
                    GcTier::Cold => report.evicted_cold += 1,
                }
            }
        }

        let collected: Vec<ChatMessage> = messages
            .into_iter()
            .zip(keep)
            .filter(|(_, keep)| *keep)
            .map(|(message, _)| message)
            .collect();
        state.prefix_invalidated = prefix_changed(&prefix_snapshot, &collected);
        state.tier_report = report;
        collected
    }
}

/// Elide eligible kept tool-result bodies in place, oldest first, until
/// the kept window fits the budget (t-1167 generational phases; the
/// annotate-don't-drop shape). A result is only elided when it has been
/// incorporated (a later kept assistant message exists — never destroy
/// the live working set), is not already an annotation, and the
/// annotation actually shrinks it. Returns how many bodies were elided.
fn elide_tool_results(
    messages: &mut [ChatMessage],
    keep: &[bool],
    budget: usize,
    eligible: &[bool],
    boundary: usize,
    counts: &BTreeMap<String, u32>,
) -> usize {
    let summaries = tool_call_summaries(messages);
    let mut elided = 0usize;
    for index in boundary..messages.len() {
        if estimate_tokens(&kept_messages(messages, keep)) <= budget {
            break;
        }
        if !keep[index] || !eligible[index] || messages[index].role != "tool" {
            continue;
        }
        let Some(call_id) = messages[index].tool_call_id.clone() else {
            continue;
        };
        let Some(content) = messages[index].content.clone() else {
            continue;
        };
        if content.trim_start().starts_with(EVICTION_MARKER_PREFIX) {
            continue;
        }
        let incorporated = messages
            .iter()
            .enumerate()
            .skip(index + 1)
            .any(|(idx, later)| keep[idx] && later.role == "assistant");
        if !incorporated {
            continue;
        }
        let summary = summaries
            .get(call_id.as_str())
            .cloned()
            .unwrap_or_else(|| call_id.clone());
        let annotation = elision_annotation(&content, &call_id, &summary, counts);
        if annotation.chars().count() >= content.chars().count() {
            continue;
        }
        messages[index].content = Some(annotation);
        elided += 1;
    }
    elided
}

impl ContextGc for GenerationalGc {
    fn collect(
        &self,
        messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage> {
        with_window_bookkeeping(messages, budget, state, |messages, budget, state| {
            self.collect_inner(messages, budget, state)
        })
    }

    fn name(&self) -> &'static str {
        "generational"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

// --- Citation signals (t-1351) ----------------------------------------------
//
// docs/GC.md "Citation signals": position and topic are proxies for whether a
// message still matters; citation is direct evidence. Extraction is pure text
// analysis over exactly the inputs collect() already has — stateless,
// deterministic, synchronous, LLM-free — so unlike embeddings it runs INSIDE
// collect(), with no cache, no pre-pass, and no degrade mode.

/// The citation graph of one window: which messages are *cited* — referenced
/// by a later message's text (tool-call-id mention) or pulled by reference
/// into a sub-infer child (the `infer` tool's `context_refs`, t-1344).
///
/// Edges point citing message -> cited message. The cited target is the
/// tool-*result* message when it exists in the window (that is where the
/// tokens live), else the dispatching call message; protecting the result
/// implicitly protects its call because pair atomicity refuses to split
/// them. The structural pair members — the assistant message that issued a
/// call and the tool message answering it — carry the id by construction
/// and never count as citing it.
///
/// Content-similarity citation (paraphrase without the id) is deliberately
/// NOT an edge kind: similarity is [`SemanticGc`]'s mechanism. Similarity
/// says "on-topic", citation says "load-bearing" — keeping the extractor
/// exact keeps the two signals orthogonal (docs/GC.md, the 2x2).
#[derive(Debug, Clone, Default)]
pub struct CitationGraph {
    /// Cited message id -> the ids of the messages citing it.
    cited_by: HashMap<MsgId, BTreeSet<MsgId>>,
}

impl CitationGraph {
    /// Build the graph for one window. Pure and total: malformed
    /// `context_refs` entries and ids that resolve to nothing are ignored,
    /// never an error.
    pub fn extract(messages: &[ChatMessage]) -> Self {
        // Every tool-call id minted in the window, with its dispatching
        // call index and (when present) its result index. First occurrence
        // wins, matching the interpreter's resolution order.
        let mut sites: Vec<(String, usize, Option<usize>)> = Vec::new();
        let mut site_index: HashMap<String, usize> = HashMap::new();
        for (index, message) in messages.iter().enumerate() {
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                if !site_index.contains_key(&call.id) {
                    site_index.insert(call.id.clone(), sites.len());
                    sites.push((call.id.clone(), index, None));
                }
            }
            if let Some(result_of) = &message.tool_call_id {
                if let Some(&site) = site_index.get(result_of) {
                    let (_, call_index, result_index) = &mut sites[site];
                    if *call_index < index && result_index.is_none() {
                        *result_index = Some(index);
                    }
                }
            }
        }

        let mut graph = Self::default();
        for (index, message) in messages.iter().enumerate() {
            // context_refs edges: the model asked for these results to be
            // re-materialized into a child's context — explicit citations
            // by construction (t-1344), no text scanning needed.
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                let refs = call
                    .arguments
                    .get("context_refs")
                    .and_then(serde_json::Value::as_array);
                for id in refs
                    .into_iter()
                    .flatten()
                    .filter_map(serde_json::Value::as_str)
                {
                    graph.add_edge(messages, &sites, &site_index, id, index);
                }
            }
            // id-mention edges: the message's own text names the call
            // ("per the output of call-X, proceed with..."). Token-boundary
            // match so `call-1` never fires inside `call-10`.
            let Some(text) = message.content.as_deref() else {
                continue;
            };
            for (id, call_index, _) in &sites {
                // The id exists only from its dispatching call onward.
                if index > *call_index && mentions_id(text, id) {
                    graph.add_edge(messages, &sites, &site_index, id, index);
                }
            }
        }
        graph
    }

    fn add_edge(
        &mut self,
        messages: &[ChatMessage],
        sites: &[(String, usize, Option<usize>)],
        site_index: &HashMap<String, usize>,
        id: &str,
        citing: usize,
    ) {
        let Some(&site) = site_index.get(id) else {
            return;
        };
        let (_, call_index, result_index) = &sites[site];
        let target = result_index.unwrap_or(*call_index);
        // The structural pair members never cite themselves.
        if citing == target || citing == *call_index {
            return;
        }
        self.cited_by
            .entry(messages[target].id)
            .or_default()
            .insert(messages[citing].id);
    }

    /// Is this message cited by at least one other message?
    pub fn is_cited(&self, id: &MsgId) -> bool {
        self.cited_by.contains_key(id)
    }

    /// The messages citing `id` (empty when uncited). In-degree is the
    /// warm-promotion signal generational GC (t-1167) consumes.
    pub fn citers(&self, id: &MsgId) -> impl Iterator<Item = &MsgId> {
        self.cited_by.get(id).into_iter().flatten()
    }

    /// Total number of citation edges in the window.
    pub fn edge_count(&self) -> usize {
        self.cited_by.values().map(BTreeSet::len).sum()
    }
}

/// Per-index cited mask over a window: the `cited-keep` modifier any
/// strategy can OR into its protected set (docs/GC.md). [`SemanticGc`] is
/// the first consumer.
pub fn cited_mask(messages: &[ChatMessage]) -> Vec<bool> {
    let graph = CitationGraph::extract(messages);
    messages
        .iter()
        .map(|message| graph.is_cited(&message.id))
        .collect()
}

/// Does `text` contain `id` as a standalone token? Boundaries are
/// non-identifier characters, so `call-1` does not match inside `call-10`
/// or `recall-1x`.
fn mentions_id(text: &str, id: &str) -> bool {
    if id.is_empty() {
        return false;
    }
    let is_id_char = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-';
    let bytes = text.as_bytes();
    let mut start = 0;
    while let Some(found) = text[start..].find(id) {
        let begin = start + found;
        let end = begin + id.len();
        let open = begin == 0 || !is_id_char(bytes[begin - 1]);
        let close = end == bytes.len() || !is_id_char(bytes[end]);
        if open && close {
            return true;
        }
        start = begin + 1;
    }
    false
}

/// What one re-injection pre-pass observed, reported on the gc_collect
/// event (`recall_overlap_events` / `recall_hot`) so the behavioral eval
/// (t-1349) can watch the write-barrier fire.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecallOverlapReport {
    /// Overlapping re-injection chunks found in this window (a chunk counts
    /// every pass it stays in the window: re-observation is the signal).
    pub overlap_events: usize,
    /// Cumulative size of `GcState.recall_hot` after this pass.
    pub hot_total: usize,
}

/// What the most recent collect() did for the hot set (t-1362), reported on
/// the gc_collect event as `hot_kept`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HotKeepReport {
    /// Messages in the returned window carrying hot content — what
    /// hot-keep is currently protecting.
    pub hot_kept: usize,
    /// Evicted tool results whose content had already been evicted
    /// before (t-1370): the re-eviction count this collection — the loop
    /// signal hot-keep exists to drive to zero.
    pub reevictions: usize,
}

/// Tool names whose results re-inject memory content — the write-barrier
/// sources. Today just the agent loop's `recall` tool (ir_agent.rs).
const RECALL_TOOL_NAMES: [&str; 1] = ["recall"];

// --- Re-injection chunk matching (t-1362) -----------------------------------
//
// t-1351's write-barrier matched by exact content hash and fired 0/15 in
// three generations of the behavioral eval (evals/gc/README.md): a recall
// hit is a memory RENDER and a re-run tool call is a fresh JSON ENVELOPE
// (differing in `duration_ms` alone), so neither ever hash-equals the
// window/collected content it re-injects. The replacement matches
// normalized content CHUNKS instead:
//
// 1. Extract payload strings: JSON content (tool-result envelopes, recall
//    hit arrays) contributes its string leaves — which drops the volatile
//    non-string envelope fields (`duration_ms`, `status`, `ok`,
//    `*_truncated`) by construction, the exact t-1369 re-fetch delta
//    (precedent: a5da89d normalizes wall-clock as metering noise).
//    Non-JSON content is one opaque payload.
// 2. Split payloads into lines, whitespace-fold each line, and key lines
//    of >= MIN_CHUNK_CHARS chars by content hash.
//
// Deterministic (pure text analysis), cheap (one linear pass per message,
// keys stored — never a full-window scan), and it catches both observed
// directions: (a) a recall result re-injecting content whose earlier
// render was collected, and (b) a re-run tool call returning the same
// payload an earlier evicted call returned.

/// Minimum chars a whitespace-folded line needs to become a chunk key.
/// Shorter lines ("ok", "STATUS: OK", bare counts) carry too little
/// entropy: exact-matching them would mark unrelated content hot.
const MIN_CHUNK_CHARS: usize = 12;

/// Is this line GC's own annotation output (`[gc: ...]` markers,
/// `[frame ...]` summaries, `[gc-ledger]` headers)? Annotation lines never
/// become chunk keys: a marker *describing* an eviction must not vouch for
/// evicted content.
fn is_gc_annotation_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with(EVICTION_MARKER_PREFIX)
        || trimmed.starts_with("[frame ")
        || trimmed.starts_with(GC_LEDGER_PREFIX)
}

fn collect_string_leaves(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(text) => {
            if !text.trim().is_empty() {
                out.push(text.clone());
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_string_leaves(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values() {
                collect_string_leaves(item, out);
            }
        }
        _ => {}
    }
}

/// The payload strings inside one message content: JSON string leaves when
/// the content parses as JSON (numbers/booleans — the volatile envelope
/// fields — vanish by construction), else the raw text as one payload.
fn payload_strings(content: &str) -> Vec<String> {
    match serde_json::from_str::<serde_json::Value>(content) {
        Ok(value @ (serde_json::Value::Object(_) | serde_json::Value::Array(_))) => {
            let mut out = Vec::new();
            collect_string_leaves(&value, &mut out);
            out
        }
        _ => vec![content.to_string()],
    }
}

/// Normalized chunk keys for write-barrier matching (t-1362): payload
/// strings split into lines, each line whitespace-folded (interior runs
/// collapse to one space), lines of >= [`MIN_CHUNK_CHARS`] chars hashed.
/// The same stdout re-fetched under a different `duration_ms`, or
/// re-injected as a memory render, yields overlapping keys.
pub fn reinjection_chunk_keys(content: &str) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    // A progress-ledger message (t-1373) is GC's own bookkeeping: its entry
    // lines carry outcome PREVIEWS of real content, and a digest describing
    // content must never vouch for it (the same rule as marker lines).
    if content.trim_start().starts_with(GC_LEDGER_PREFIX) {
        return keys;
    }
    for payload in payload_strings(content) {
        for line in payload.lines() {
            if is_gc_annotation_line(line) {
                continue;
            }
            let folded = line.split_whitespace().collect::<Vec<_>>().join(" ");
            if folded.chars().count() >= MIN_CHUNK_CHARS {
                keys.insert(content_hash(&folded));
            }
        }
    }
    keys
}

/// Stable identity for "the same content" across re-fetches (t-1370):
/// the hash of a message's chunk-key set, so the same payload under a
/// different call id or a different envelope (duration_ms) fingerprints
/// identically. Chunk-less content (too short to key) falls back to the
/// whitespace-folded text hash.
pub fn content_fingerprint(content: &str) -> String {
    let keys = reinjection_chunk_keys(content);
    if keys.is_empty() {
        content_hash(&content.split_whitespace().collect::<Vec<_>>().join(" "))
    } else {
        content_hash(&keys.into_iter().collect::<Vec<_>>().join(","))
    }
}

/// Per-index hot mask over a window: a message is hot when any of its
/// chunk keys is in `GcState.recall_hot` — content the model demonstrably
/// re-acquired after eviction. This is what the `hot-keep` guard (t-1362)
/// unions into a strategy's protected set: a value the model went and got
/// AGAIN stops being evictable under normal pressure. Same strength as
/// cited-keep — weaker than the preserve-prefix billing contract and the
/// system/last-user hard guards, relaxed in the degrade phases.
pub fn hot_mask(messages: &[ChatMessage], state: &GcState) -> Vec<bool> {
    if state.recall_hot.is_empty() {
        return vec![false; messages.len()];
    }
    messages
        .iter()
        .map(|message| {
            message.content.as_deref().is_some_and(|content| {
                reinjection_chunk_keys(content)
                    .iter()
                    .any(|key| state.recall_hot.contains(key))
            })
        })
        .collect()
}

fn union_masks(base: &[bool], extra: &[bool]) -> Vec<bool> {
    base.iter()
        .zip(extra)
        .map(|(left, right)| *left || *right)
        .collect()
}

/// The re-injection write-barrier pre-pass (t-1351/t-1362, docs/GC.md):
/// for each tool result in the window, chunk its content
/// ([`reinjection_chunk_keys`]) and match against content previously
/// collected from the window (`GcState.collected_hashes`); `recall`
/// results additionally match against every other window message (a
/// memory hit re-injecting live content is a re-reference even before
/// anything was evicted). Matches mark the chunks HOT in
/// `GcState.recall_hot` — the model pulled back something GC took away
/// (or already had), so dropping it again would thrash.
///
/// Pure, synchronous, and total, but it lives in the pre-pass rather than
/// inside collect() so the hot set is fresh before the strategy consults
/// it, and because "previously collected" is cross-collection state.
/// Consumed by every strategy's hot-keep guard (t-1362); it remains the
/// promotion signal generational GC (t-1167) is specified against.
pub fn record_reinjection_overlaps(
    messages: &[ChatMessage],
    state: &mut GcState,
) -> RecallOverlapReport {
    let mut call_names: HashMap<&str, &str> = HashMap::new();
    for message in messages {
        for call in message.tool_calls.as_deref().unwrap_or_default() {
            call_names.entry(call.id.as_str()).or_insert(&call.name);
        }
    }
    let is_tool_result = |message: &ChatMessage| message.role == "tool";
    let is_recall_result = |message: &ChatMessage| {
        is_tool_result(message)
            && message.tool_call_id.as_deref().is_some_and(|id| {
                call_names
                    .get(id)
                    .is_some_and(|name| RECALL_TOOL_NAMES.contains(name))
            })
    };

    // What a recall could be re-injecting from the LIVE window: every
    // other (non-recall) message's chunks. Recall results are excluded so
    // two identical recalls do not vouch for each other; built only when
    // the window has recall traffic.
    let window_keys: BTreeSet<String> = if messages.iter().any(&is_recall_result) {
        messages
            .iter()
            .filter(|message| !is_recall_result(message))
            .filter_map(|message| message.content.as_deref())
            .flat_map(reinjection_chunk_keys)
            .collect()
    } else {
        BTreeSet::new()
    };

    let mut report = RecallOverlapReport::default();
    for message in messages.iter().filter(|message| is_tool_result(message)) {
        let Some(content) = message.content.as_deref() else {
            continue;
        };
        let recall = is_recall_result(message);
        for key in reinjection_chunk_keys(content) {
            let collected = state.collected_hashes.contains(&key);
            let live = recall && window_keys.contains(&key);
            if collected || live {
                state.recall_hot.insert(key);
                report.overlap_events += 1;
            }
        }
    }
    report.hot_total = state.recall_hot.len();
    report
}

// --- Eviction markers (t-1360) ----------------------------------------------
//
// The mechanism-level fix for the confabulation failure mode (t-1349/t-1364/
// t-1367 behavioral evidence: models fabricate evicted content — access
// codes, categories — instead of recovering or admitting loss, despite
// do-not-guess guidance). When a collection drops messages, it leaves a
// compact, deterministic marker line in the window saying WHAT was evicted
// (kind + identifying handle: tool-call id, recall query, or turn ordinal)
// and HOW to recover it (re-run the call, recall the memory, ask the user).
//
// Marker economics, in order:
//
// - CHEAP: a dropped 2000-token result becomes a one-line ~30-token marker,
//   and N consecutive drops aggregate into ONE marker line, not N.
// - BUDGET-HONEST: markers count toward the window budget. They are funded
//   by the sweep's natural overshoot (dropping whole messages lands under
//   budget with slack); if the per-run markers do not fit, they degrade to
//   a single "earlier context compacted" line, and if even that does not
//   fit they are suppressed — a collection never ships over budget because
//   of its own markers, so every convergence property is unchanged.
// - DROPPABLE: markers are ordinary assistant messages, never
//   hard-protected. A later collection may drop one; its eviction count is
//   absorbed into the replacing marker rather than vanishing silently.
// - DETERMINISTIC: markers are part of collect()'s pure output. Content
//   derives only from the dropped messages; ids are UUIDv5 over the dropped
//   ids, so the same collection produces byte- and id-identical markers.
//
// Integration is strategy-uniform via [`with_window_bookkeeping`], with two
// strategy-honest carve-outs: a tool result dropped by StackFrameGc's frame
// pop is NOT double-marked (the surviving `[frame ...]` annotation, which
// now names the call id and the recovery affordance, IS its marker), and
// MarkSweepGc's in-place elision annotation is the same `[gc: ...]` family.

/// Every eviction-marker line starts with this. The §2.4 guidance block
/// describes the format to the model; evals count in-window markers by it.
pub const EVICTION_MARKER_PREFIX: &str = "[gc:";

/// At most this many evicted items are named per marker line; the rest are
/// folded into a `+N more` suffix so markers stay cheap under mass drops.
const MAX_MARKER_ITEMS: usize = 4;

/// Marker escalation threshold (t-1370): when the SAME content (by
/// [`content_fingerprint`]) is evicted this many times in one run, its
/// marker escalates from a recovery affordance to an honest exit — stop
/// re-fetching, summarize into memory or ask the user. Three because the
/// first eviction is normal pressure, a second tolerates one legitimate
/// re-fetch losing a race with the next collection, and the third is the
/// t-1369 loop signature (ring re-fetched the code 2-3x and then
/// guessed). With hot-keep (t-1362) protecting re-acquired content, a
/// third eviction can only mean degrade pressure — content that
/// genuinely cannot stay in this window.
pub const EVICTION_ESCALATION_AFTER: u32 = 3;

/// The escalated marker line (t-1370): names the latest handle, the
/// repeat count, and the honest exit. Recognized by
/// [`is_escalation_marker`] via the "cannot stay in context" phrase.
fn escalation_marker_content(handle: &str, evictions: u32) -> String {
    format!(
        "[gc: '{handle}' evicted {evictions} times — this content cannot \
stay in context: summarize what you need into memory (remember) or ask \
the user — do not re-fetch again]"
    )
}

/// Is this an escalated eviction marker (t-1370)? Escalation markers are
/// never fused with neighbors — fusion keeps the later text, which would
/// silently delete the honest-exit instruction.
pub fn is_escalation_marker(message: &ChatMessage) -> bool {
    is_eviction_marker(message)
        && message
            .content
            .as_deref()
            .is_some_and(|content| content.contains("cannot stay in context"))
}

/// What one collection left behind as markers (t-1360). Reported on the
/// gc_collect trace event so behavioral evals can tell marker-driven
/// recovery from re-derivation and fabrication.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvictionMarkerSummary {
    /// Eviction-marker messages present in the collected window after this
    /// collection (post-aggregation and fusion; includes markers surviving
    /// from earlier collections).
    pub markers: usize,
    /// Logical tool results evicted by this collection (frame-covered drops
    /// excluded — the frame annotation is their marker).
    pub evicted_tool_results: usize,
    /// Recall (memory) results evicted by this collection.
    pub evicted_recalls: usize,
    /// User turns evicted by this collection.
    pub evicted_user_turns: usize,
    /// Plain assistant turns evicted by this collection.
    pub evicted_assistant_turns: usize,
    /// Degrade: the per-run markers did not fit the budget; a single
    /// "earlier context compacted" line stands in for all of them.
    pub coalesced: bool,
    /// Terminal degrade: not even the coalesced line fit. This collection
    /// wrote no markers — the trace event still records what was dropped.
    pub suppressed: bool,
    /// Escalated markers present in the collected window (t-1370):
    /// `[gc: '<handle>' evicted N times — ... cannot stay in context ...]`
    /// lines, from repeated-eviction drops and mark-sweep's escalated
    /// elisions alike.
    pub escalated: usize,
}

/// Is this message an eviction-marker line written by a previous collection?
/// Markers are plain assistant messages whose content starts with
/// [`EVICTION_MARKER_PREFIX`] — recognizable so later collections can absorb
/// (or coalesce) them instead of letting them vanish silently.
pub fn is_eviction_marker(message: &ChatMessage) -> bool {
    message.role == "assistant"
        && message.tool_call_id.is_none()
        && message.tool_calls.is_none()
        && message
            .content
            .as_deref()
            .is_some_and(|content| content.starts_with(EVICTION_MARKER_PREFIX))
}

/// How many evictions a marker line stands for: its first integer
/// OUTSIDE single quotes (an escalated marker's quoted handle may carry
/// digits — `'call-7'` must not read as 7; its count is the "evicted N
/// times" N). Defaults to 1 — a marker always represents at least one
/// eviction.
fn marker_evicted_count(message: &ChatMessage) -> usize {
    let content = message.content.as_deref().unwrap_or("");
    let mut digits = String::new();
    let mut in_quote = false;
    for c in content.chars() {
        if c == '\'' {
            if !digits.is_empty() {
                break;
            }
            in_quote = !in_quote;
            continue;
        }
        if !in_quote && c.is_ascii_digit() {
            digits.push(c);
        } else if !digits.is_empty() {
            break;
        }
    }
    digits.parse().unwrap_or(1).max(1)
}

/// Replace the first integer in a marker line (its eviction count) with
/// `count` — used when fusing adjacent markers.
fn replace_marker_count(content: &str, count: usize) -> String {
    let start = match content.find(|c: char| c.is_ascii_digit()) {
        Some(start) => start,
        None => return content.to_string(),
    };
    let end = content[start..]
        .find(|c: char| !c.is_ascii_digit())
        .map_or(content.len(), |offset| start + offset);
    format!("{}{count}{}", &content[..start], &content[end..])
}

/// Deterministic marker identity: UUIDv5 over the ids of the messages the
/// marker stands for. No fresh UUIDs, so the same collection produces the
/// same marker ids every run (the id space is disjoint from the dropped
/// messages' own ids, so retention metrics never mistake a marker for a
/// survivor).
fn marker_id(ids: &[MsgId]) -> MsgId {
    let seed = ids
        .iter()
        .map(MsgId::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let digest = content_hash(&format!("gc-eviction-marker:{seed}"));
    let mut bytes = [0u8; 16];
    for (index, slot) in bytes.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&digest[index * 2..index * 2 + 2], 16)
            .expect("content_hash emits lowercase hex");
    }
    Uuid::from_bytes(bytes)
}

fn marker_message(id: MsgId, content: String) -> ChatMessage {
    ChatMessage {
        id,
        role: "assistant".into(),
        content: Some(content),
        tool_call_id: None,
        tool_calls: None,
    }
}

/// One maximal run of consecutively dropped messages, in original order.
#[derive(Default)]
struct MarkerRun {
    start: Option<usize>,
    /// Messages dropped in this run (markers absorbed from earlier
    /// collections count via `absorbed`, not here).
    dropped: usize,
    /// Eviction counts absorbed from dropped older marker lines.
    absorbed: usize,
    /// Named items, e.g. "shell call-3", "recall 'deploy window'",
    /// "user turn 2".
    items: Vec<String>,
    ids: Vec<MsgId>,
    has_tool: bool,
    has_recall: bool,
    has_user: bool,
}

struct MarkerBuild {
    /// (original index of the run's first dropped message, marker message).
    markers: Vec<(usize, ChatMessage)>,
    summary: EvictionMarkerSummary,
    /// Total evictions represented (dropped messages plus counts absorbed
    /// from dropped older markers) — the coalesced line's N.
    total_evicted: usize,
    /// Original index of the first dropped message (coalesced placement).
    first_drop: Option<usize>,
}

/// Build the marker lines for one collection, from the pre-collection
/// window, the collection's survivors, and the run's per-content
/// eviction counts (t-1370: prior counts decide escalation — this
/// collection's own increments happen afterwards, in the wrapper's
/// finish). Pure text analysis over exactly the inputs collect() already
/// has — stateless, deterministic, LLM-free.
fn build_eviction_markers(
    original: &[ChatMessage],
    collected: &[ChatMessage],
    counts: &BTreeMap<String, u32>,
) -> MarkerBuild {
    let kept: BTreeSet<MsgId> = collected.iter().map(|message| message.id).collect();
    // Every call id minted in the pre-collection window: tool name,
    // arguments, and the issuing assistant message (frame-covered check).
    let mut call_info: HashMap<&str, (&str, &serde_json::Value, MsgId)> = HashMap::new();
    for message in original {
        for call in message.tool_calls.as_deref().unwrap_or_default() {
            call_info.entry(call.id.as_str()).or_insert((
                call.name.as_str(),
                &call.arguments,
                message.id,
            ));
        }
    }
    // Cited-keep interplay: a marker for an evicted-but-cited message
    // carries the citing handle (the first citing message's turn ordinal).
    let citations = CitationGraph::extract(original);
    let index_of: HashMap<MsgId, usize> = original
        .iter()
        .enumerate()
        .map(|(index, message)| (message.id, index))
        .collect();
    let cited_suffix = |message: &ChatMessage| -> String {
        citations
            .citers(&message.id)
            .filter_map(|id| index_of.get(id).copied())
            .min()
            .map(|turn| format!(" (cited by turn {})", turn + 1))
            .unwrap_or_default()
    };

    let mut build = MarkerBuild {
        markers: Vec::new(),
        summary: EvictionMarkerSummary::default(),
        total_evicted: 0,
        first_drop: None,
    };
    // Call ids already itemized (a dropped pair yields one item, and a call
    // split across two runs by a kept message is still itemized once).
    let mut itemized_calls: BTreeSet<String> = BTreeSet::new();
    let mut run = MarkerRun::default();

    let flush = |run: &mut MarkerRun, build: &mut MarkerBuild| {
        let taken = std::mem::take(run);
        let total = taken.dropped + taken.absorbed;
        if total == 0 {
            return;
        }
        let Some(start) = taken.start else {
            return;
        };
        build.total_evicted += total;
        let listing = if taken.items.is_empty() {
            "earlier context".to_string()
        } else {
            let shown = taken
                .items
                .iter()
                .take(MAX_MARKER_ITEMS)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            let extra = taken.items.len().saturating_sub(MAX_MARKER_ITEMS);
            if extra > 0 {
                format!("{shown} +{extra} more")
            } else {
                shown
            }
        };
        let mut affordances: Vec<&str> = Vec::new();
        if taken.has_tool {
            affordances.push("re-run the call");
        }
        if taken.has_recall {
            affordances.push("recall the memory");
        }
        if taken.has_user {
            affordances.push("ask the user again");
        }
        let tail = if affordances.is_empty() {
            "unrecoverable — do not guess".to_string()
        } else {
            format!("recover: {} — do not guess", affordances.join(", "))
        };
        let content = format!("[gc: {total} evicted — {listing}; {tail}]");
        build
            .markers
            .push((start, marker_message(marker_id(&taken.ids), content)));
    };

    for (index, message) in original.iter().enumerate() {
        if kept.contains(&message.id) {
            flush(&mut run, &mut build);
            continue;
        }
        if build.first_drop.is_none() {
            build.first_drop = Some(index);
        }
        run.start.get_or_insert(index);
        run.ids.push(message.id);
        if is_eviction_marker(message) {
            // A dropped older marker: absorb its count instead of letting
            // it vanish silently.
            run.absorbed += marker_evicted_count(message);
            continue;
        }
        run.dropped += 1;
        match message.role.as_str() {
            "tool" => {
                let Some(call_id) = message.tool_call_id.as_deref() else {
                    build.summary.evicted_assistant_turns += 1;
                    run.items.push(format!("turn {}", index + 1));
                    continue;
                };
                let Some((name, arguments, issuer)) = call_info.get(call_id) else {
                    build.summary.evicted_tool_results += 1;
                    run.has_tool = true;
                    run.items
                        .push(format!("tool {call_id}{}", cited_suffix(message)));
                    continue;
                };
                // Escalation (t-1370): this content's Nth eviction stops
                // getting a recovery affordance — re-fetching demonstrably
                // does not stick — and gets the honest exit instead.
                let escalated = message.content.as_deref().and_then(|content| {
                    if content.starts_with(EVICTION_MARKER_PREFIX) {
                        return None;
                    }
                    let evictions = counts
                        .get(&content_fingerprint(content))
                        .copied()
                        .unwrap_or(0)
                        + 1;
                    (evictions >= EVICTION_ESCALATION_AFTER).then_some(evictions)
                });
                if kept.contains(issuer) {
                    // Frame-covered (StackFrameGc pop): the surviving
                    // assistant message was rewritten to a `[frame ...]`
                    // annotation naming this call and its recovery
                    // affordance — that annotation IS the marker. Break
                    // the run so no duplicate `[gc: ...]` line appears.
                    // Escalation outranks the dedup: a repeatedly-lost
                    // result gets the honest-exit line even here (t-1369
                    // finding 4 — stack's recovery loop needs a
                    // termination affordance).
                    run.dropped -= 1;
                    run.ids.pop();
                    flush(&mut run, &mut build);
                    if let Some(evictions) = escalated {
                        if itemized_calls.insert(call_id.to_string()) {
                            build.summary.escalated += 1;
                            build.markers.push((
                                index,
                                marker_message(
                                    marker_id(&[message.id]),
                                    escalation_marker_content(call_id, evictions),
                                ),
                            ));
                        }
                    }
                    continue;
                }
                if !itemized_calls.insert(call_id.to_string()) {
                    continue;
                }
                if let Some(evictions) = escalated {
                    // Stand the escalation marker alone at this position:
                    // pull the message out of the aggregate run (it still
                    // counts as evicted) so the honest-exit line is not
                    // buried in a listing.
                    run.dropped -= 1;
                    run.ids.pop();
                    flush(&mut run, &mut build);
                    build.total_evicted += 1;
                    build.summary.evicted_tool_results += 1;
                    build.summary.escalated += 1;
                    build.markers.push((
                        index,
                        marker_message(
                            marker_id(&[message.id]),
                            escalation_marker_content(call_id, evictions),
                        ),
                    ));
                    continue;
                }
                if RECALL_TOOL_NAMES.contains(name) {
                    let query = arguments
                        .get("query")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    build.summary.evicted_recalls += 1;
                    run.has_recall = true;
                    run.items.push(format!(
                        "recall '{}'{}",
                        preview_chars(query, 40),
                        cited_suffix(message)
                    ));
                } else {
                    build.summary.evicted_tool_results += 1;
                    run.has_tool = true;
                    run.items
                        .push(format!("{name} {call_id}{}", cited_suffix(message)));
                }
            }
            "assistant" => {
                let calls = message.tool_calls.as_deref().unwrap_or_default();
                if calls.is_empty() {
                    build.summary.evicted_assistant_turns += 1;
                    run.items.push(format!(
                        "assistant turn {}{}",
                        index + 1,
                        cited_suffix(message)
                    ));
                } else {
                    // A dropped call message: pair atomicity means its
                    // results are dropped too and carry the items; only a
                    // call whose result never existed (open frame) is
                    // itemized here so the id stays recoverable.
                    for call in calls {
                        let has_result = original
                            .iter()
                            .any(|other| other.tool_call_id.as_deref() == Some(call.id.as_str()));
                        if !has_result && itemized_calls.insert(call.id.clone()) {
                            build.summary.evicted_tool_results += 1;
                            run.has_tool = true;
                            run.items.push(format!("{} {}", call.name, call.id));
                        }
                    }
                }
            }
            "user" => {
                build.summary.evicted_user_turns += 1;
                run.has_user = true;
                run.items.push(format!("user turn {}", index + 1));
            }
            // System messages are hard-protected and never reach here;
            // anything unexpected still counts toward the run total.
            _ => {
                run.items.push(format!("turn {}", index + 1));
            }
        }
    }
    flush(&mut run, &mut build);
    build
}

/// Splice marker lines into a collected window at their runs' positions.
/// A marker is never inserted immediately before a tool-result message
/// (results must stay adjacent to their call), and adjacent marker lines
/// fuse into one (counts summed) so markers never accumulate.
fn merge_eviction_markers(
    original: &[ChatMessage],
    collected: &[ChatMessage],
    markers: &[(usize, ChatMessage)],
) -> Vec<ChatMessage> {
    let index_of: HashMap<MsgId, usize> = original
        .iter()
        .enumerate()
        .map(|(index, message)| (message.id, index))
        .collect();
    let mut out: Vec<ChatMessage> = Vec::with_capacity(collected.len() + markers.len());
    let mut pending = markers.iter().peekable();
    for message in collected {
        let position = index_of.get(&message.id).copied().unwrap_or(usize::MAX);
        while pending
            .peek()
            .is_some_and(|(run_start, _)| *run_start < position)
        {
            if message.role == "tool" {
                // Defer past the tool run: a marker between a call and its
                // results would break provider pair adjacency.
                break;
            }
            out.push(pending.next().expect("peeked").1.clone());
        }
        out.push(message.clone());
    }
    for (_, marker) in pending {
        out.push(marker.clone());
    }
    fuse_adjacent_markers(out)
}

/// Fuse directly adjacent marker lines into one: counts sum, the later
/// (most recent) marker's text wins. Keeps repeated collections from
/// stacking marker upon marker.
fn fuse_adjacent_markers(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    for message in messages {
        if is_eviction_marker(&message) && !is_escalation_marker(&message) {
            if let Some(last) = out.last_mut() {
                if is_eviction_marker(last) && !is_escalation_marker(last) {
                    let combined = marker_evicted_count(last) + marker_evicted_count(&message);
                    let content = replace_marker_count(
                        message.content.as_deref().unwrap_or(EVICTION_MARKER_PREFIX),
                        combined,
                    );
                    let fused_id = marker_id(&[last.id, message.id]);
                    *last = marker_message(fused_id, content);
                    continue;
                }
            }
        }
        out.push(message);
    }
    out
}

/// The degrade form: one line standing in for every marker in the window.
fn coalesced_marker_content(total: usize) -> String {
    format!(
        "[gc: earlier context compacted — {total} messages evicted; \
         recover: re-run tool calls, recall memories, or ask the user — do not guess]"
    )
}

/// Extra tokens reserved beyond the rendered markers' own estimate when
/// re-collecting to make room: a tighter budget can extend the dropped
/// runs, which grows the marker text by a few tokens.
const MARKER_RESERVE_PAD: usize = 16;

/// Coalesce a collection's markers into one line: strip every marker still
/// in the window, absorb their counts, and stand a single compacted line
/// at the earliest gap. Returns the window with the line merged in.
fn coalesce_markers(
    original: &[ChatMessage],
    collected: &[ChatMessage],
    build: &MarkerBuild,
) -> Vec<ChatMessage> {
    let mut total = build.total_evicted;
    let mut first_position = build.first_drop.unwrap_or(0);
    let mut contributing: Vec<MsgId> = Vec::new();
    let mut stripped: Vec<ChatMessage> = Vec::with_capacity(collected.len());
    for message in collected {
        if is_eviction_marker(message) {
            total += marker_evicted_count(message);
            contributing.push(message.id);
            if let Some(position) = original.iter().position(|m| m.id == message.id) {
                first_position = first_position.min(position);
            }
            continue;
        }
        stripped.push(message.clone());
    }
    contributing.extend(build.markers.iter().map(|(_, marker)| marker.id));
    let coalesced = marker_message(marker_id(&contributing), coalesced_marker_content(total));
    merge_eviction_markers(original, &stripped, &[(first_position, coalesced)])
}

// --- Progress ledger (t-1373) -------------------------------------------------
//
// The mechanism-level fix for the restart loop — the dominant failure of
// every behavioral eval generation (evals/gc/README.md: t-1349 finding 2,
// t-1371's refutation where the tuned curator re-ran `cat
// plans/approach-a.txt` ten times). Post-collection, the model loses its
// narrative position: not the facts (markers + hot-keep fixed fact
// recovery) but the PLAN STATE — what it already did, what worked, where
// it was. Markers are per-drop notices scattered through the window;
// nothing gave a consolidated "you are here". The ledger is that
// consolidation: ONE synthetic assistant message, a deterministic digest
// of the session's completed tool calls and their eviction state,
// rebuilt (replaced, never appended) by every collection. No collections
// = no ledger — unfired GC stays invisible (the t-1371 control regime).
//
// Coordination with markers, not duplication: markers say WHERE content
// was removed (positional, per-drop, with recovery affordances); the
// ledger says WHAT the session has already done (global, per-call, with
// each call's current state). When degrade pressure coalesces markers to
// a single handle-less "earlier context compacted" line, the ledger still
// carries the per-call handles and the escalation state ("evicted 3x — do
// not re-fetch"), so recovery affordances survive marker coalescing.

/// Every progress-ledger message starts with this. Distinct from
/// [`EVICTION_MARKER_PREFIX`] (`[gc:`) so ledger and markers never
/// mistake each other; the §2.4 guidance block describes the format to
/// the model.
pub const GC_LEDGER_PREFIX: &str = "[gc-ledger]";

/// At most this many journal entries are itemized per ledger (newest
/// first-class; older entries coalesce into one "older work" line).
pub const MAX_LEDGER_ENTRIES: usize = 10;

/// The compact rung of the ledger ladder: the newest few entries, the
/// rest coalesced. Also the reserve target for re-collections — reserving
/// the FULL ledger would trip the rung-2 gate on exactly the windows that
/// need it most, so the reserve guarantees a compact ledger and the full
/// one appears only when the sweep's natural overshoot funds it.
const LEDGER_COMPACT_ENTRIES: usize = 4;

/// The coalesce-harder ladder for the ledger's own budget honesty: full
/// itemization, the compact tail, the last two calls, then header +
/// coalesced line only.
const LEDGER_SHOWN_LADDER: [usize; 4] = [MAX_LEDGER_ENTRIES, LEDGER_COMPACT_ENTRIES, 2, 0];

/// Cap for a ledger entry's outcome preview (chars of the first
/// meaningful normalized-payload line).
const LEDGER_OUTCOME_CHARS: usize = 48;

/// One journaled completed tool call (t-1373). Journal identity is the
/// call id; the fingerprint ties the entry to its content across
/// re-fetches and envelope changes (the t-1370 machinery).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerEntry {
    pub call_id: String,
    pub tool: String,
    pub args_preview: String,
    /// First meaningful line of the result's normalized payload
    /// ([`payload_strings`] — the write-barrier machinery, so volatile
    /// envelope fields never enter the preview), capped at
    /// [`LEDGER_OUTCOME_CHARS`]. Empty when the journal only ever saw an
    /// already-elided annotation (a resumed window).
    pub outcome_preview: String,
    /// [`content_fingerprint`] of the raw result content — the key into
    /// `GcState.eviction_counts` for the entry's escalation state.
    pub fingerprint: String,
}

/// What the most recent collect() left as the progress ledger (t-1373),
/// reported on the gc_collect trace event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LedgerSummary {
    /// A ledger message is in the returned window.
    pub present: bool,
    /// Itemized entry lines in the rendered ledger (older work coalesces).
    pub entries: usize,
    /// Terminal degrade: the session has tool history but not even the
    /// coalesced ledger fit the budget. Never silent.
    pub suppressed: bool,
    /// Call ids itemized in the rendered ledger — the restart-loop needle:
    /// a repeated command whose earlier call id is listed here was
    /// re-run AGAINST the ledger's own record.
    pub calls: Vec<String>,
}

/// Is this message the progress ledger written by a previous collection?
pub fn is_gc_ledger(message: &ChatMessage) -> bool {
    message.role == "assistant"
        && message.tool_call_id.is_none()
        && message.tool_calls.is_none()
        && message
            .content
            .as_deref()
            .is_some_and(|content| content.starts_with(GC_LEDGER_PREFIX))
}

/// A ledger entry's args preview: the same argument keys the marker/frame
/// summaries use, plus `query` (recall) — first match wins.
fn ledger_args_preview(arguments: &serde_json::Value) -> String {
    ["path", "file", "command", "query", "prompt"]
        .iter()
        .find_map(|key| arguments.get(key).and_then(serde_json::Value::as_str))
        .map(|value| preview_chars(value, 60))
        .unwrap_or_default()
}

/// A ledger entry's outcome preview: the first meaningful line of the
/// normalized payload — the first whitespace-folded non-annotation line
/// of >= [`MIN_CHUNK_CHARS`] chars (falling back to the first non-empty
/// line), capped. The same normalization the write-barrier chunks by, so
/// `duration_ms` and friends never enter the preview.
fn ledger_outcome_preview(content: &str) -> String {
    let mut fallback: Option<String> = None;
    for payload in payload_strings(content) {
        for line in payload.lines() {
            if is_gc_annotation_line(line) {
                continue;
            }
            let folded = line.split_whitespace().collect::<Vec<_>>().join(" ");
            if folded.is_empty() {
                continue;
            }
            if folded.chars().count() >= MIN_CHUNK_CHARS {
                return preview_chars(&folded, LEDGER_OUTCOME_CHARS);
            }
            fallback.get_or_insert(folded);
        }
    }
    fallback
        .map(|line| preview_chars(&line, LEDGER_OUTCOME_CHARS))
        .unwrap_or_default()
}

/// Journal every completed tool call the pre-collection window shows
/// (t-1373): a tool result answering a known call, first occurrence wins,
/// window order preserved. Runs BEFORE the core so a result the same
/// collection elides or drops is journaled from its raw content; an
/// already-annotated result (resumed window) is journaled with an empty
/// outcome preview rather than a preview of GC's own annotation.
fn update_ledger_journal(window: &[ChatMessage], state: &mut GcState) {
    let mut call_info: HashMap<&str, (&str, &serde_json::Value)> = HashMap::new();
    for message in window {
        for call in message.tool_calls.as_deref().unwrap_or_default() {
            call_info
                .entry(call.id.as_str())
                .or_insert((call.name.as_str(), &call.arguments));
        }
    }
    for message in window {
        if message.role != "tool" {
            continue;
        }
        let Some(call_id) = message.tool_call_id.as_deref() else {
            continue;
        };
        let Some((tool, arguments)) = call_info.get(call_id) else {
            continue;
        };
        if state.ledger.iter().any(|entry| entry.call_id == call_id) {
            continue;
        }
        let content = message.content.as_deref().unwrap_or("");
        let annotated = content.trim_start().starts_with(EVICTION_MARKER_PREFIX)
            || content.trim_start().starts_with(GC_LEDGER_PREFIX);
        state.ledger.push(LedgerEntry {
            call_id: call_id.to_string(),
            tool: (*tool).to_string(),
            args_preview: ledger_args_preview(arguments),
            outcome_preview: if annotated {
                String::new()
            } else {
                ledger_outcome_preview(content)
            },
            fingerprint: if annotated {
                String::new()
            } else {
                content_fingerprint(content)
            },
        });
    }
}

/// One entry's current state against a collected window: the full result
/// body is still present (`in-window`), present and write-barrier hot
/// (`hot`), or gone/elided (`evicted`, escalating to the honest exit at
/// [`EVICTION_ESCALATION_AFTER`] evictions — the t-1370 ladder).
fn ledger_entry_state(entry: &LedgerEntry, window: &[ChatMessage], state: &GcState) -> String {
    let live = window.iter().find_map(|message| {
        (message.tool_call_id.as_deref() == Some(entry.call_id.as_str()))
            .then(|| message.content.as_deref().unwrap_or(""))
    });
    if let Some(content) = live {
        if !content.trim_start().starts_with(EVICTION_MARKER_PREFIX)
            && !entry.fingerprint.is_empty()
            && content_fingerprint(content) == entry.fingerprint
        {
            let hot = reinjection_chunk_keys(content)
                .iter()
                .any(|key| state.recall_hot.contains(key));
            return if hot {
                "hot".into()
            } else {
                "in-window".into()
            };
        }
    }
    let evictions = state
        .eviction_counts
        .get(&entry.fingerprint)
        .copied()
        .unwrap_or(0);
    if evictions >= EVICTION_ESCALATION_AFTER {
        format!("evicted {evictions}x — do not re-fetch")
    } else {
        "evicted".into()
    }
}

struct LedgerRender {
    content: String,
    entries: usize,
    calls: Vec<String>,
}

/// Render the ledger for one collected window: header, an optional
/// coalesced older-work line, the newest `shown` entries as
/// `call-id: tool(args) -> outcome [state]` one-liners, and one recovery
/// footer when anything is evicted (the affordance once, not per line —
/// markers already carry the per-drop affordances). Pure text analysis:
/// stateless, deterministic, LLM-free.
fn render_ledger(state: &GcState, window: &[ChatMessage], shown: usize) -> LedgerRender {
    use std::fmt::Write as _;
    let total = state.ledger.len();
    let start = total.saturating_sub(shown);
    let mut content = format!(
        "{GC_LEDGER_PREFIX} your progress record, auto-updated — {total} tool \
call{} done this session; consult it before re-running work, these steps are DONE:",
        if total == 1 { "" } else { "s" }
    );
    let mut entries = 0usize;
    let mut calls = Vec::new();
    let mut older_evicted = 0usize;
    let mut any_evicted = false;
    for (index, entry) in state.ledger.iter().enumerate() {
        let entry_state = ledger_entry_state(entry, window, state);
        let evicted = entry_state.starts_with("evicted");
        any_evicted |= evicted;
        if index < start {
            older_evicted += usize::from(evicted);
            continue;
        }
        if entry.outcome_preview.is_empty() {
            let _ = write!(
                content,
                "\n{}: {}({}) [{entry_state}]",
                entry.call_id, entry.tool, entry.args_preview
            );
        } else {
            let _ = write!(
                content,
                "\n{}: {}({}) -> {} [{entry_state}]",
                entry.call_id, entry.tool, entry.args_preview, entry.outcome_preview
            );
        }
        entries += 1;
        calls.push(entry.call_id.clone());
    }
    if start > 0 {
        // Splice the coalesced line in right after the header.
        let line = format!(
            "\nolder work: {start} earlier call{} completed ({older_evicted} evicted) — already done, do not redo",
            if start == 1 { "" } else { "s" }
        );
        let header_end = content.find('\n').unwrap_or(content.len());
        content.insert_str(header_end, &line);
    }
    if any_evicted {
        content.push_str(
            "\nevicted results are recoverable: re-run the call or recall the memory — do not guess.",
        );
    }
    LedgerRender {
        content,
        entries,
        calls,
    }
}

/// Deterministic ledger identity: derived from the rendered content, so
/// the same collection produces the same message id every run (no fresh
/// UUIDs — the marker-id rule).
fn ledger_id(content: &str) -> MsgId {
    let digest = content_hash(&format!("gc-ledger:{content}"));
    let mut bytes = [0u8; 16];
    for (index, slot) in bytes.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&digest[index * 2..index * 2 + 2], 16)
            .expect("content_hash emits lowercase hex");
    }
    Uuid::from_bytes(bytes)
}

/// Where the ledger goes: the TAIL of the window — the churn region, with
/// maximal recency salience — stepping back over a trailing user run (the
/// ledger sits immediately before the latest user turn when the window
/// ends with one) and over a trailing open tool-call turn (a result
/// arriving later must stay adjacent to its call). Never before `clamp`
/// (the surviving byte-stable pinned prefix): the ledger must not cause a
/// cache invalidation the core did not already commit.
fn ledger_insert_position(window: &[ChatMessage], clamp: usize) -> usize {
    let mut position = window.len();
    while position > clamp {
        let previous = &window[position - 1];
        let open_call_turn = previous.role == "assistant"
            && previous
                .tool_calls
                .as_deref()
                .unwrap_or_default()
                .iter()
                .any(|call| {
                    !window
                        .iter()
                        .any(|message| message.tool_call_id.as_deref() == Some(call.id.as_str()))
                });
        if previous.role == "user" || open_call_turn {
            position -= 1;
        } else {
            break;
        }
    }
    position.max(clamp)
}

/// Wrap a strategy's core collection with eviction-marker emission
/// (t-1360) and progress-ledger maintenance (t-1373). Budget honesty:
/// markers and the ledger count toward the window budget, so
/// the ladder makes room for them instead of shipping over budget —
///
/// 1. per-run markers into the core's collection, when they fit;
/// 2. re-collect with the marker + ledger cost reserved, then per-run
///    markers;
/// 3. one coalesced "earlier context compacted" line, into the core's
///    collection or a re-collection;
/// 4. nothing — recorded as `suppressed` on the summary, never silent.
///
/// The ledger rides `finish`: after the marker ladder settles a window,
/// the ledger is attached at the tail if it fits, coalesced harder
/// (fewer itemized entries) if not, and suppressed-with-record
/// (`ledger_suppressed` on the gc_collect event) under terminal
/// pressure. The previous collection's ledger instance is stripped from
/// the input up front — replaced as GC bookkeeping, never counted or
/// marked as an eviction.
///
/// Re-collections run on a scratch GcState that is committed only when
/// their window is the one returned, so the strategy's cross-collection
/// metadata (frame statuses, lifecycle tags, prefix_invalidated) always
/// describes the returned window. The wrapper never pushes the window over
/// budget: every strategy's convergence contract is exactly its core's.
fn with_window_bookkeeping<F>(
    messages: Vec<ChatMessage>,
    budget: usize,
    state: &mut GcState,
    core: F,
) -> Vec<ChatMessage>
where
    F: Fn(Vec<ChatMessage>, usize, &mut GcState) -> Vec<ChatMessage>,
{
    // Replace, never append: the old ledger instance is bookkeeping. It
    // leaves before the core sees the window, so no strategy can evict it
    // (which would mark bookkeeping as an eviction) and its tokens are
    // freed for real content.
    let messages: Vec<ChatMessage> = messages
        .into_iter()
        .filter(|message| !is_gc_ledger(message))
        .collect();
    // Journal completed calls from the raw pre-collection window (t-1373)
    // before the core elides or drops anything.
    update_ledger_journal(&messages, state);
    let original = messages.clone();
    let full_boundary = cache_prefix_boundary(&original, budget).min(original.len());
    let prefix_snapshot: Vec<ChatMessage> = original[..full_boundary].to_vec();
    let collected = core(messages, budget, state);
    let build = build_eviction_markers(&original, &collected, &state.eviction_counts);
    let count_markers =
        |window: &[ChatMessage]| window.iter().filter(|m| is_eviction_marker(m)).count();
    let attach_ledger = |mut window: Vec<ChatMessage>, state: &mut GcState| -> Vec<ChatMessage> {
        state.ledger_summary = LedgerSummary::default();
        // No evictions yet = no ledger: the window is still the complete
        // work record, and an under-budget collect() must stay a no-op.
        if state.ledger.is_empty() || !state.evictions_seen {
            return window;
        }
        // The surviving byte-stable prefix: insertion below this index
        // would be a cache invalidation the core did not commit.
        let clamp = window
            .iter()
            .zip(&prefix_snapshot)
            .take_while(|(after, before)| after == before)
            .count();
        for shown in LEDGER_SHOWN_LADDER {
            let render = render_ledger(state, &window, shown);
            let message = marker_message(ledger_id(&render.content), render.content.clone());
            if estimate_tokens(&window)
                .saturating_add(estimate_tokens(std::slice::from_ref(&message)))
                > budget
            {
                continue;
            }
            let position = ledger_insert_position(&window, clamp);
            window.insert(position, message);
            state.ledger_summary = LedgerSummary {
                present: true,
                entries: render.entries,
                suppressed: false,
                calls: render.calls,
            };
            return window;
        }
        // Terminal: the session has tool history but no room for even the
        // coalesced ledger — recorded, never silent.
        state.ledger_summary = LedgerSummary {
            suppressed: true,
            ..LedgerSummary::default()
        };
        window
    };
    let finish = |window: Vec<ChatMessage>,
                  mut summary: EvictionMarkerSummary,
                  state: &mut GcState|
     -> Vec<ChatMessage> {
        summary.markers = count_markers(&window);
        // Escalated lines standing in the window (t-1370): standalone
        // escalation markers plus mark-sweep's escalated elisions.
        summary.escalated = window
            .iter()
            .filter(|message| {
                message.content.as_deref().is_some_and(|content| {
                    content.starts_with(EVICTION_MARKER_PREFIX)
                        && content.contains("cannot stay in context")
                })
            })
            .count();
        // Write-barrier corpus (t-1362): the normalized chunks of whatever
        // this collection removed — dropped messages AND in-place rewrites
        // (mark-sweep elision, stack frame pops) — join collected_hashes,
        // so the pre-pass can recognize this content when a re-run call or
        // a recall injects it back. GC annotation lines contribute nothing
        // (reinjection_chunk_keys filters them), so dropped markers never
        // vouch for content. Tool-result removals also feed the per-content
        // eviction counts (t-1370) — a fingerprint seen before is a
        // re-eviction, the loop signal.
        let survivors: HashMap<MsgId, Option<&str>> = window
            .iter()
            .map(|message| (message.id, message.content.as_deref()))
            .collect();
        let mut reevictions = 0usize;
        for message in &original {
            let Some(content) = message.content.as_deref() else {
                continue;
            };
            let removed = match survivors.get(&message.id) {
                None => true,
                Some(now) => *now != Some(content),
            };
            if removed {
                state.evictions_seen = true;
                state
                    .collected_hashes
                    .extend(reinjection_chunk_keys(content));
                if message.role == "tool" && !content.starts_with(EVICTION_MARKER_PREFIX) {
                    let count = state
                        .eviction_counts
                        .entry(content_fingerprint(content))
                        .or_insert(0);
                    if *count >= 1 {
                        reevictions += 1;
                    }
                    *count += 1;
                }
            }
        }
        state.marker_summary = summary;
        state.hot_report = HotKeepReport {
            hot_kept: hot_mask(&window, state)
                .into_iter()
                .filter(|hot| *hot)
                .count(),
            reevictions,
        };
        // The progress ledger (t-1373) attaches last, so entry states see
        // this collection's eviction counts and hot set.
        attach_ledger(window, state)
    };
    if build.markers.is_empty() {
        return finish(collected, build.summary, state);
    }

    // Rung 1: per-run markers fit the core's own collection (the sweep's
    // natural overshoot funds them). The ledger is deliberately NOT part
    // of this gate: bookkeeping must never force a re-collection that
    // perturbs the core's content decisions (cited-keep, hot-keep, the
    // preserve boundary) — the ledger coalesces down to the slack the
    // sweep left, and records its own suppression when there is none.
    let merged = merge_eviction_markers(&original, &collected, &build.markers);
    if estimate_tokens(&merged) <= budget {
        return finish(merged, build.summary, state);
    }

    // Re-collections run at a reduced budget, which also shrinks the
    // pinned-prefix allowance — so a re-collection may only be accepted if
    // it keeps the ORIGINAL budget's prefix byte-stable wherever the first
    // collection did (the preserve-mode billing contract; markers must
    // never cause a cache invalidation the core did not already commit).
    let first_invalidated = prefix_changed(&prefix_snapshot, &collected);
    let recollect =
        |reserve: usize, scratch: &mut GcState| -> Option<(Vec<ChatMessage>, MarkerBuild)> {
            let recollected = core(
                original.clone(),
                budget.saturating_sub(reserve).max(1),
                scratch,
            );
            if prefix_changed(&prefix_snapshot, &recollected) && !first_invalidated {
                return None;
            }
            scratch.prefix_invalidated = prefix_changed(&prefix_snapshot, &recollected);
            let rebuild = build_eviction_markers(&original, &recollected, &scratch.eviction_counts);
            Some((recollected, rebuild))
        };

    // Rung 2: reserve the markers' + ledger's cost and re-collect, so the
    // bookkeeping is paid for by evicting more content — never by
    // overflowing the budget. The ledger reserve is a provisional render
    // against the core's collection (entry states shift the text by a few
    // chars; MARKER_RESERVE_PAD absorbs that).
    let marker_tokens: usize = build
        .markers
        .iter()
        .map(|(_, marker)| estimate_tokens(std::slice::from_ref(marker)))
        .sum();
    // The ledger is deliberately NOT reserved for here: a ledger-sized
    // reserve shrinks the re-collection budget enough to push the core
    // past its heuristic guards (observed: the cited-keep and
    // preserve-prefix promotion gates fail when the ledger rides this
    // reserve), and content decisions outrank bookkeeping. Markers stay
    // reserved — silent eviction is a correctness failure; a coalesced
    // ledger is not. The ledger's own ladder (coalesce harder, then
    // suppress-with-record) runs in finish, funded by the sweep's slack.
    let reserve = marker_tokens.saturating_add(MARKER_RESERVE_PAD);
    if reserve <= budget / 4 {
        let mut scratch = state.clone();
        if let Some((recollected, rebuild)) = recollect(reserve, &mut scratch) {
            let remerged = merge_eviction_markers(&original, &recollected, &rebuild.markers);
            if estimate_tokens(&remerged) <= budget {
                *state = scratch;
                return finish(remerged, rebuild.summary, state);
            }
        }
    }

    // Rung 3: one coalesced line, into the core's collection or (when even
    // that does not fit) a re-collection reserving its cost.
    let with_line = coalesce_markers(&original, &collected, &build);
    if estimate_tokens(&with_line) <= budget {
        let mut summary = build.summary;
        summary.coalesced = true;
        return finish(with_line, summary, state);
    }
    let line_reserve = estimate_text_tokens(&coalesced_marker_content(build.total_evicted))
        .saturating_add(estimate_message_overhead_tokens() + MARKER_RESERVE_PAD);
    if line_reserve <= budget / 2 {
        let mut scratch = state.clone();
        if let Some((recollected, rebuild)) = recollect(line_reserve, &mut scratch) {
            let with_line = coalesce_markers(&original, &recollected, &rebuild);
            if estimate_tokens(&with_line) <= budget {
                *state = scratch;
                let mut summary = rebuild.summary;
                summary.coalesced = true;
                return finish(with_line, summary, state);
            }
        }
    }

    // Rung 4 (terminal): no room for even one line. Ship the core's
    // collection unchanged (markers surviving from earlier collections
    // stay — the core chose to keep them) and record the suppression.
    let mut summary = build.summary;
    summary.suppressed = true;
    finish(collected, summary, state)
}

#[derive(Debug, Clone)]
pub enum GcMode {
    None,
    Ring(RingGc),
    MarkSweep(MarkSweepGc),
    Stack(StackFrameGc),
    Semantic(SemanticGc),
    Generational(GenerationalGc),
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
            Self::Generational(gc) => gc.collect(messages, budget, state),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ring(gc) => gc.name(),
            Self::MarkSweep(gc) => gc.name(),
            Self::Stack(gc) => gc.name(),
            Self::Semantic(gc) => gc.name(),
            Self::Generational(gc) => gc.name(),
        }
    }

    pub fn cache_preserving(&self) -> bool {
        match self {
            Self::None => true,
            Self::Ring(gc) => gc.cache_preserving(),
            Self::MarkSweep(gc) => gc.cache_preserving(),
            Self::Stack(gc) => gc.cache_preserving(),
            Self::Semantic(gc) => gc.cache_preserving(),
            Self::Generational(gc) => gc.cache_preserving(),
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
        messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage> {
        with_window_bookkeeping(messages, budget, state, |messages, budget, state| {
            self.collect_inner(messages, budget, state)
        })
    }

    fn name(&self) -> &'static str {
        "mark-sweep"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

impl MarkSweepGc {
    fn collect_inner(
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

        let hot = if self.hot_keep {
            hot_mask(&messages, state)
        } else {
            vec![false; messages.len()]
        };
        let hot_any = hot.iter().any(|hot| *hot);

        tag_lifecycles(&messages, state);
        // Hot results keep their full body (hot-keep, t-1362): elision is
        // in-place content destruction, exactly the loss a re-fetched
        // value must not suffer twice.
        annotate_evictable_tool_results(&mut messages, state, restrict, &hot);

        let mut keep = vec![true; messages.len()];
        sweep_by_lifecycle(
            &messages,
            &mut keep,
            state,
            budget,
            restrict,
            LifecycleState::Evictable,
            &hot,
        );
        sweep_by_lifecycle(
            &messages,
            &mut keep,
            state,
            budget,
            restrict,
            LifecycleState::Complete,
            &hot,
        );
        // Hot relax: when the lifecycle passes cannot reach the budget
        // with hot messages protected, run them again without hot-keep so
        // mark-sweep reclaims exactly what it could before t-1362 (its
        // convergence remains best-effort either way).
        if hot_any && estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            let unrestricted = vec![false; messages.len()];
            sweep_by_lifecycle(
                &messages,
                &mut keep,
                state,
                budget,
                restrict,
                LifecycleState::Evictable,
                &unrestricted,
            );
            sweep_by_lifecycle(
                &messages,
                &mut keep,
                state,
                budget,
                restrict,
                LifecycleState::Complete,
                &unrestricted,
            );
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

/// The in-place elision annotation for one tool result: the `[gc: ...]`
/// marker family (t-1360) — the old "result incorporated" wording reported
/// the call happened while silently withholding its body, the
/// confabulation-inviting shape; naming the eviction and the recovery
/// affordance invites recovery instead. Escalation (t-1370): content
/// elided/evicted N times already gets the honest exit instead of another
/// recovery affordance — elision joins the same escalation ladder as
/// whole-message drops. Used by generational's elide phases (mark-sweep's
/// annotate pass keeps its own historical logic byte-for-byte — recorded
/// sessions replay its collections strictly).
fn elision_annotation(
    content: &str,
    tool_call_id: &str,
    summary: &str,
    counts: &BTreeMap<String, u32>,
) -> String {
    let evictions = counts
        .get(&content_fingerprint(content))
        .copied()
        .unwrap_or(0)
        + 1;
    if evictions >= EVICTION_ESCALATION_AFTER {
        escalation_marker_content(tool_call_id, evictions)
    } else {
        format!(
            "[gc: result elided — {summary} ({tool_call_id}); recover: re-run the call — do not guess]"
        )
    }
}

fn annotate_evictable_tool_results(
    messages: &mut [ChatMessage],
    state: &GcState,
    boundary: usize,
    avoid: &[bool],
) {
    let call_summaries = tool_call_summaries(messages);
    for (index, message) in messages.iter_mut().enumerate().skip(boundary) {
        if message.role != "tool" || avoid[index] {
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
        // Escalation (t-1370): content elided/evicted N times already
        // gets the honest exit instead of another recovery affordance —
        // mark-sweep's elision joins the same escalation ladder as
        // whole-message drops.
        let escalated = message.content.as_deref().and_then(|content| {
            if content.starts_with(EVICTION_MARKER_PREFIX) {
                return None;
            }
            let evictions = state
                .eviction_counts
                .get(&content_fingerprint(content))
                .copied()
                .unwrap_or(0)
                + 1;
            (evictions >= EVICTION_ESCALATION_AFTER).then_some(evictions)
        });
        if let Some(evictions) = escalated {
            message.content = Some(escalation_marker_content(tool_call_id, evictions));
            continue;
        }
        // The `[gc: ...]` marker family (t-1360): the old "result
        // incorporated" wording reported the call happened while silently
        // withholding its body — the confabulation-inviting shape. Name
        // the eviction and the recovery affordance instead.
        message.content = Some(format!(
            "[gc: result elided — {summary} ({tool_call_id}); recover: re-run the call — do not guess]"
        ));
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
    avoid: &[bool],
) {
    while estimate_tokens(&kept_messages(messages, keep)) > budget {
        let Some(index) = messages.iter().enumerate().position(|(idx, message)| {
            idx >= boundary
                && keep[idx]
                && !avoid[idx]
                && message.role != "system"
                && state.lifecycle.get(&message.id).copied() == Some(target)
                && atomic_group_stays_past(messages, keep, idx, boundary)
                && atomic_group_avoids_protected(messages, keep, idx, avoid)
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
        with_window_bookkeeping(messages, budget, state, |messages, budget, state| {
            self.collect_inner(messages, budget, state)
        })
    }

    fn name(&self) -> &'static str {
        "ring"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

impl RingGc {
    fn collect_inner(
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
        let protected = protected_with_prefix(&messages, boundary);
        let hot = if self.hot_keep {
            hot_mask(&messages, state)
        } else {
            vec![false; messages.len()]
        };
        let hot_any = hot.iter().any(|hot| *hot);
        // Phase 1: drop oldest-first from the interior (boundary 0 in ignore
        // mode makes this the classic front-drop, minus the hard guards),
        // skipping write-barrier-hot messages (hot-keep, t-1362).
        sweep_ring(&messages, &mut keep, budget, &union_masks(&protected, &hot));
        // Phase 1b (hot relax): hot-keep is a heuristic guard with
        // cited-keep strength — it relaxes before the prefix pin (a
        // billing contract) and the hard guards do.
        if hot_any && estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            sweep_ring(&messages, &mut keep, budget, &protected);
        }
        // Phase 2 (preserve fallback): the pinned prefix plus the live tail
        // alone exceed the budget. Overflowing the model is worse than a
        // cache miss, so degrade to front-drop — with system + last user
        // still hard-protected (t-1367) — and the gc_collect event reports
        // the invalidation via state.prefix_invalidated.
        if boundary > 0 && estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            sweep_ring(
                &messages,
                &mut keep,
                budget,
                &hard_protected_mask(&messages),
            );
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
}

impl ContextGc for StackFrameGc {
    fn collect(
        &self,
        messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage> {
        with_window_bookkeeping(messages, budget, state, |messages, budget, state| {
            self.collect_inner(messages, budget, state)
        })
    }

    fn name(&self) -> &'static str {
        "stack"
    }

    fn cache_preserving(&self) -> bool {
        self.preserve_prefix
    }
}

impl StackFrameGc {
    fn collect_inner(
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
        let hot = if self.hot_keep {
            hot_mask(&messages, state)
        } else {
            vec![false; messages.len()]
        };
        let hot_any = hot.iter().any(|hot| *hot);
        let protected = protected_with_prefix(&messages, boundary);
        // Phase 1: pop completed frames oldest-first until under budget,
        // skipping frames with write-barrier-hot members (hot-keep,
        // t-1362: popping destroys the result body — the loss a
        // re-fetched value must not suffer twice).
        while estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            let Some(frame) = oldest_completed_frame(&messages, &keep, boundary, &hot) else {
                break;
            };
            pop_frame(&mut messages, &mut keep, &frame, state);
        }
        // Phase 2: frames alone were not enough (open frames, chat-heavy
        // windows); drop oldest-first from the interior like ring, still
        // skipping hot messages.
        if estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            sweep_ring(&messages, &mut keep, budget, &union_masks(&protected, &hot));
        }
        // Phase 2b (hot relax): hot-keep is a heuristic guard with
        // cited-keep strength — pop and sweep again without it before the
        // prefix pin or the hard guards are touched.
        if hot_any && estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            let unrestricted = vec![false; messages.len()];
            while estimate_tokens(&kept_messages(&messages, &keep)) > budget {
                let Some(frame) = oldest_completed_frame(&messages, &keep, boundary, &unrestricted)
                else {
                    break;
                };
                pop_frame(&mut messages, &mut keep, &frame, state);
            }
            if estimate_tokens(&kept_messages(&messages, &keep)) > budget {
                sweep_ring(&messages, &mut keep, budget, &protected);
            }
        }
        // Phase 3 (preserve fallback): the pinned prefix plus the live tail
        // alone exceed the budget. Overflowing the model is worse than a
        // cache miss, so degrade to front-drop — with system + last user
        // still hard-protected (t-1367) — and the gc_collect event reports
        // the invalidation via state.prefix_invalidated.
        if boundary > 0 && estimate_tokens(&kept_messages(&messages, &keep)) > budget {
            sweep_ring(
                &messages,
                &mut keep,
                budget,
                &hard_protected_mask(&messages),
            );
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

/// The oldest kept frame whose every member sits at or past `boundary`
/// and none of whose members is marked in `avoid` (the hot-keep guard —
/// pass an all-false mask to lift the restriction). Frames with any
/// unanswered call are open — never popped, never split. A frame is only
/// poppable once a later assistant message exists past its last result:
/// until the model has spoken again, the result is the live working set,
/// not history (same incorporation rule as mark-sweep).
fn oldest_completed_frame(
    messages: &[ChatMessage],
    keep: &[bool],
    boundary: usize,
    avoid: &[bool],
) -> Option<Frame> {
    for (index, message) in messages.iter().enumerate().skip(boundary) {
        if !keep[index] || message.role != "assistant" || avoid[index] {
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
            && results.iter().all(|idx| *idx >= boundary && !avoid[*idx])
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

/// `[frame call-id: tool(args) -> result]`, one line per call, prefixed
/// with a preview of the assistant's own narration when it had any. The
/// call id is the recovery handle, and a truncated result preview says so
/// explicitly — t-1349 finding 3: an annotation that reports the call
/// happened while silently withholding its result invites confabulation;
/// one that names the eviction and the affordance invites recovery. Pure
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
        let full_result = frame
            .results
            .iter()
            .map(|idx| &messages[*idx])
            .find(|message| message.tool_call_id.as_deref() == Some(call.id.as_str()))
            .and_then(|message| message.content.as_deref())
            .map(str::trim)
            .unwrap_or_default();
        if full_result.chars().count() > 120 {
            // Truncated: spend the line on the eviction notice rather than
            // a longer sliver of stale preview — the bare preview invites
            // confabulation, the notice invites recovery.
            let result = preview_chars(full_result, 80);
            lines.push(format!(
                "[frame {}: {}({args}) -> {result} — evicted; re-run to recover]",
                call.id, call.name
            ));
        } else {
            lines.push(format!(
                "[frame {}: {}({args}) -> {full_result}]",
                call.id, call.name
            ));
        }
    }
    lines.join("\n")
}

/// Drop unprotected messages oldest-first until under budget. `protected`
/// combines the hard guards (system + last user, [`hard_protected_mask`])
/// with the pinned prefix in preserve mode; tool-call pairs travel
/// atomically, and a group that would pull a protected message out is
/// skipped entirely (the same mechanism the semantic sweep uses). When
/// only protected messages remain, the sweep stops and the window ships
/// over budget — the overflow paths own that terminal case.
fn sweep_ring(messages: &[ChatMessage], keep: &mut [bool], budget: usize, protected: &[bool]) {
    while estimate_tokens(&kept_messages(messages, keep)) > budget {
        let Some(index) = messages.iter().enumerate().position(|(idx, message)| {
            !protected[idx]
                && keep[idx]
                && message.role != "system"
                && atomic_group_avoids_protected(messages, keep, idx, protected)
        }) else {
            break;
        };
        drop_atomic_group(messages, keep, index);
    }
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
                    .is_some_and(|content| content.contains("[frame call-1: shell(cargo test)"))
            })
            .expect("popped frame leaves a summary annotation");
        // The fat result was truncated in the preview, so the annotation
        // names the eviction and the recovery affordance (t-1360).
        assert!(
            summary
                .content
                .as_deref()
                .is_some_and(|content| content.contains("evicted; re-run to recover")),
            "truncated frame previews must carry the recovery affordance: {summary:?}"
        );
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
        // and insert fat assistant narration after the user turn so it
        // exhausts the pinned-prefix allowance (otherwise pair-pinning
        // absorbs frame 1 into the prefix, where preserve mode correctly
        // refuses to pop). The ballast is deliberately NOT the user turn:
        // the last user message is hard-protected (t-1367), so making it
        // the droppable filler would force phase 2 to spend the live frame
        // instead.
        let mut messages = stack_fixture();
        messages.pop();
        messages.insert(
            2,
            ChatMessage::assistant(Some("context ".repeat(80)), vec![]),
        );
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
            Some(
                "[gc: result elided — read_file /tmp/large.txt (call-1); \
                 recover: re-run the call — do not guess]"
            )
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
            hot_keep: true,
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
            hot_keep: true,
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
            hot_keep: true,
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
            hot_keep: true,
            preserve_prefix: false,
            recent_window: 2,
            similarity_floor: 0.25,
            embedder: None,
            cited_keep: true,
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
            hot_keep: true,
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
            hot_keep: true,
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
            hot_keep: true,
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

    // ---- Citation signals (t-1351) ------------------------------------------

    /// system + user + one completed read_file frame + closing narration.
    /// No third-party message mentions the call id: zero citations.
    fn uncited_frame_fixture() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("system"),
            ChatMessage::user("audit the dependency tree"),
            ChatMessage::assistant(None, vec![read_file_call("call-audit", "/tmp/deps.txt")]),
            ChatMessage::tool("call-audit", "vulnerable: libfoo 1.2 ".repeat(30)),
            ChatMessage::assistant(Some("Reviewed the audit output.".into()), vec![]),
        ]
    }

    #[test]
    fn citation_graph_structural_pair_members_are_not_citations() {
        // The dispatching call carries the id in tool_calls and the result
        // in tool_call_id — by construction, not as citations.
        let messages = uncited_frame_fixture();
        let graph = CitationGraph::extract(&messages);
        assert_eq!(graph.edge_count(), 0, "{graph:?}");
        assert!(messages.iter().all(|message| !graph.is_cited(&message.id)));
    }

    #[test]
    fn citation_graph_extracts_id_mentions_and_context_refs() {
        let mut messages = uncited_frame_fixture();
        let result_id = messages[3].id;
        // id-mention: a later message names the call in its text.
        messages.push(ChatMessage::assistant(
            Some("Per the output of call-audit, libfoo is the vulnerable one.".into()),
            vec![],
        ));
        let mention_id = messages.last().unwrap().id;
        // context_refs: an infer dispatch pulls the result by reference
        // (t-1344) — an explicit citation by construction.
        messages.push(ChatMessage::assistant(
            None,
            vec![ToolCall::new(
                "call-child",
                "infer",
                serde_json::json!({
                    "prompt": "summarize the finding",
                    "context_refs": ["call-audit", "no-such-id", 7],
                }),
            )],
        ));
        let refs_id = messages.last().unwrap().id;

        let graph = CitationGraph::extract(&messages);
        assert!(
            graph.is_cited(&result_id),
            "the tool result is the citation target: {graph:?}"
        );
        let citers: BTreeSet<_> = graph.citers(&result_id).copied().collect();
        assert_eq!(citers, BTreeSet::from([mention_id, refs_id]));
        // Unknown/malformed context_refs entries are ignored, never errors.
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn citation_id_mentions_require_token_boundaries() {
        let mut messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(None, vec![tool_call("call-1")]),
            ChatMessage::tool("call-1", "one"),
            ChatMessage::assistant(None, vec![tool_call("call-10")]),
            ChatMessage::tool("call-10", "ten"),
        ];
        let one = messages[2].id;
        let ten = messages[4].id;
        messages.push(ChatMessage::assistant(
            Some("proceed per call-10 (and my recall-1x note)".into()),
            vec![],
        ));

        let graph = CitationGraph::extract(&messages);
        assert!(graph.is_cited(&ten));
        assert!(
            !graph.is_cited(&one),
            "call-1 must not match inside call-10: {graph:?}"
        );
    }

    #[test]
    fn citation_of_an_open_call_targets_the_call_message() {
        let mut messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(None, vec![tool_call("call-open")]),
        ];
        let call_msg = messages[1].id;
        messages.push(ChatMessage::user("while call-open runs, check the logs"));
        let graph = CitationGraph::extract(&messages);
        assert!(graph.is_cited(&call_msg));
    }

    /// system + planner task + a distant-but-later-cited frame + a distant
    /// uncited noise message + recent on-topic tail whose last message cites
    /// the frame by id. Axis 1 = distant, axis 0 = the recent topic.
    fn cited_distant_fixture(state: &mut GcState) -> Vec<ChatMessage> {
        let messages = vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user(format!("fix the query planner {}", "p".repeat(120))),
            ChatMessage::assistant(
                Some("Running the dependency audit first.".into()),
                vec![read_file_call("call-audit", "/tmp/deps.txt")],
            ),
            ChatMessage::tool("call-audit", format!("audit output {}", "a".repeat(300))),
            ChatMessage::assistant(Some(format!("noise sidebar {}", "n".repeat(300))), vec![]),
            ChatMessage::user("back to the planner"),
            ChatMessage::assistant(
                Some(
                    "Per the output of call-audit, pinning libfoo and applying the planner fix."
                        .into(),
                ),
                vec![],
            ),
        ];
        for (index, message) in messages.iter().enumerate() {
            let axis = usize::from(matches!(index, 2..=4));
            cache_axis(state, message, axis);
        }
        messages
    }

    #[test]
    fn semantic_cited_keep_retains_the_cited_distant_frame() {
        let mut state = GcState::default();
        let messages = cited_distant_fixture(&mut state);
        let cited_result = messages[3].id;
        let noise = messages[4].id;
        let budget = estimate_tokens(&messages) - 60;

        // Baseline deficiency (cited_keep off): the cited frame is the
        // oldest most-distant candidate, so pure similarity drops it first.
        let baseline = SemanticGc {
            hot_keep: true,
            cited_keep: false,
            ..semantic_gc()
        };
        let mut baseline_state = GcState {
            embeddings: state.embeddings.clone(),
            ..Default::default()
        };
        let collected = baseline.collect(messages.clone(), budget, &mut baseline_state);
        assert!(estimate_tokens(&collected) <= budget);
        assert!(
            !collected.iter().any(|message| message.id == cited_result),
            "without citations the cited-but-distant frame dies: {collected:?}"
        );

        // cited-keep: the citation protects the frame through the normal
        // phases; the uncited noise pays instead.
        let collected = semantic_gc().collect(messages, budget, &mut state);
        assert!(estimate_tokens(&collected) <= budget);
        assert!(
            collected.iter().any(|message| message.id == cited_result),
            "cited-keep must retain the cited frame: {collected:?}"
        );
        assert!(
            !collected.iter().any(|message| message.id == noise),
            "the uncited distant noise drops instead: {collected:?}"
        );
    }

    #[test]
    fn semantic_cited_keep_relaxes_under_degrade_pressure() {
        // Citation is a heuristic guard, not a hard one: when even the
        // protected set exceeds the budget, cited messages relax with the
        // recency floor while system + last user stay hard-protected.
        let mut state = GcState::default();
        let messages = cited_distant_fixture(&mut state);
        let cited_result = messages[3].id;
        let last_user = messages[5].id;
        let budget = estimate_tokens(&messages[..1]) + 40;

        let collected = semantic_gc().collect(messages, budget, &mut state);

        assert!(
            !collected.iter().any(|message| message.id == cited_result),
            "degrade pressure overrides cited-keep: {collected:?}"
        );
        assert!(collected.iter().any(|message| message.role == "system"));
        assert!(collected.iter().any(|message| message.id == last_user));
    }

    // ---- re-injection write-barrier (t-1351, chunk-normalized t-1362) --------

    fn recall_frame(id: &str, hits: serde_json::Value) -> [ChatMessage; 2] {
        [
            ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    id,
                    "recall",
                    serde_json::json!({ "query": "planner fix" }),
                )],
            ),
            ChatMessage::tool(id, hits.to_string()),
        ]
    }

    /// The tool-result envelope shape ir_agent's shell tool renders — the
    /// content real windows carry. `duration_ms` is the volatile field the
    /// t-1369 re-fetch loop differed by.
    fn shell_envelope(stdout: &str, duration_ms: u64) -> String {
        serde_json::json!({
            "duration_ms": duration_ms,
            "ok": true,
            "status": 0,
            "stderr": "",
            "stderr_truncated": false,
            "stdout": stdout,
            "stdout_truncated": false,
            "timed_out": false,
        })
        .to_string()
    }

    #[test]
    fn chunk_keys_normalize_the_envelope_and_volatile_fields() {
        // Direction (b)'s prerequisite (t-1369 re-fetch loop): the same
        // stdout under a different duration_ms must key identically, and
        // the bare payload must overlap the enveloped one.
        let first = reinjection_chunk_keys(&shell_envelope("MX-7749-KESTREL\n", 9));
        let second = reinjection_chunk_keys(&shell_envelope("MX-7749-KESTREL\n", 4711));
        assert_eq!(first, second, "wall-clock noise must normalize away");
        assert!(!first.is_empty());
        let bare = reinjection_chunk_keys("MX-7749-KESTREL");
        assert_eq!(
            first, bare,
            "envelope and bare payload must share chunk keys"
        );
    }

    #[test]
    fn chunk_keys_fold_whitespace_and_skip_low_entropy_lines() {
        assert_eq!(
            reinjection_chunk_keys("the  planner fix\tis here"),
            reinjection_chunk_keys("the planner fix is here"),
        );
        // Short lines ("ok", counts, STATUS: OK) carry too little entropy
        // to key; GC's own annotation lines never key at all.
        assert!(reinjection_chunk_keys("ok\n42\nSTATUS: OK").is_empty());
        assert!(reinjection_chunk_keys(
            "[gc: 3 evicted — shell call-2; recover: re-run the call — do not guess]"
        )
        .is_empty());
        assert!(reinjection_chunk_keys(
            "[frame call-1: shell(cat a.txt) -> stuff — evicted; re-run to recover]"
        )
        .is_empty());
    }

    #[test]
    fn chunk_keys_are_deterministic() {
        let envelope = shell_envelope("line one of the payload\nline two of the payload\n", 3);
        assert_eq!(
            reinjection_chunk_keys(&envelope),
            reinjection_chunk_keys(&envelope)
        );
    }

    #[test]
    fn recall_overlap_marks_reinjected_window_content_hot() {
        let note = "the planner fix is raising the statistics target";
        let mut messages = vec![ChatMessage::system("system"), ChatMessage::user(note)];
        messages.extend(recall_frame(
            "call-recall",
            serde_json::json!([
                { "source": "memory", "kind": "Semantic", "content": note },
                { "source": "memory", "kind": "Semantic", "content": "an unrelated note body" },
            ]),
        ));
        let mut state = GcState::default();

        let report = record_reinjection_overlaps(&messages, &mut state);

        assert_eq!(report.overlap_events, 1, "only the matching hit fires");
        assert_eq!(report.hot_total, 1);
        assert!(!reinjection_chunk_keys(note).is_disjoint(&state.recall_hot));
    }

    /// Direction (a) as observed in vivo (t-1369 finding 5): the collected
    /// content was a shell-result JSON envelope, the recall hit is a memory
    /// RENDER of the same value — exact-hash matching never fired on this;
    /// chunk matching must.
    #[test]
    fn recall_overlap_matches_previously_collected_content_across_renders() {
        let token = "TOKEN-9QX-RAVEN-7734";
        let mut state = GcState::default();
        state
            .collected_hashes
            .extend(reinjection_chunk_keys(&shell_envelope(
                &format!("{token}\n"),
                9,
            )));
        let mut messages = vec![ChatMessage::system("system")];
        messages.extend(recall_frame(
            "call-recall",
            serde_json::json!([{
                "source": "memory",
                "kind": "Semantic",
                "content": format!("### deploy-token\n{token}"),
            }]),
        ));

        let report = record_reinjection_overlaps(&messages, &mut state);

        assert_eq!(report.overlap_events, 1);
        assert!(!reinjection_chunk_keys(token).is_disjoint(&state.recall_hot));
    }

    /// Direction (b), the t-1369 ring re-fetch loop: a re-run tool call
    /// returns the same payload an earlier evicted call returned (envelope
    /// differing only in duration_ms). The re-fetched result must go hot.
    #[test]
    fn refetched_tool_result_matching_collected_content_goes_hot() {
        let stdout = "MX-7749-KESTREL\n";
        let mut state = GcState::default();
        state
            .collected_hashes
            .extend(reinjection_chunk_keys(&shell_envelope(stdout, 9)));
        let refetched = shell_envelope(stdout, 231);
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(None, vec![tool_call("call-9")]),
            ChatMessage::tool("call-9", refetched.clone()),
            ChatMessage::user("finish the task"),
        ];

        let report = record_reinjection_overlaps(&messages, &mut state);

        assert_eq!(report.overlap_events, 1);
        assert!(!reinjection_chunk_keys(&refetched).is_disjoint(&state.recall_hot));
        // And the hot mask marks exactly the re-fetched result (hot-keep's
        // read side).
        let hot = hot_mask(&messages, &state);
        assert_eq!(hot, vec![false, false, true, false]);
    }

    #[test]
    fn recall_overlap_unparseable_result_falls_back_to_whole_content() {
        // A recall result that is not JSON is treated as one opaque
        // payload; window membership still fires.
        let note = "plain text recall payload";
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user(format!("  {note}  ")),
            ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    "call-recall",
                    "recall",
                    serde_json::json!({ "query": "payload" }),
                )],
            ),
            ChatMessage::tool("call-recall", note),
        ];
        let mut state = GcState::default();
        let report = record_reinjection_overlaps(&messages, &mut state);
        assert_eq!(report.overlap_events, 1);
    }

    #[test]
    fn reinjection_overlap_ignores_live_echo_and_non_overlapping_hits() {
        let note = "content that also appears in a shell result";
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user(note),
            // A shell result echoing LIVE window content is not a
            // re-injection — nothing was evicted (tool results match only
            // the collected corpus; the live-window check is recall-only).
            ChatMessage::assistant(None, vec![tool_call("call-shell")]),
            ChatMessage::tool("call-shell", note),
            // A recall whose hits overlap nothing stays cold.
            ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    "call-recall",
                    "recall",
                    serde_json::json!({ "query": "other" }),
                )],
            ),
            ChatMessage::tool(
                "call-recall",
                serde_json::json!([{ "content": "a note nobody has seen before" }]).to_string(),
            ),
        ];
        let mut state = GcState::default();
        let report = record_reinjection_overlaps(&messages, &mut state);
        assert_eq!(report.overlap_events, 0);
        assert!(state.recall_hot.is_empty());
    }

    // ---- hot-keep consumer (t-1362) -------------------------------------------

    /// A window under pressure with one hot tool pair among cold ones: the
    /// normal sweep must evict around the hot pair, whatever the strategy.
    fn hot_keep_fixture(state: &mut GcState) -> Vec<ChatMessage> {
        let hot_payload = shell_envelope("MX-7749-KESTREL\n", 17);
        state
            .collected_hashes
            .extend(reinjection_chunk_keys(&shell_envelope(
                "MX-7749-KESTREL\n",
                9,
            )));
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("count the OK lines and report the access code"),
            ChatMessage::assistant(None, vec![tool_call("call-hot")]),
            ChatMessage::tool("call-hot", hot_payload),
            ChatMessage::assistant(Some("noted the code".into()), vec![]),
            ChatMessage::assistant(None, vec![tool_call("call-cold-1")]),
            ChatMessage::tool("call-cold-1", format!("cold {}", "x".repeat(400))),
            ChatMessage::assistant(Some("step done".into()), vec![]),
            ChatMessage::assistant(None, vec![tool_call("call-cold-2")]),
            ChatMessage::tool("call-cold-2", format!("cold {}", "y".repeat(400))),
            ChatMessage::assistant(Some("another step done".into()), vec![]),
            ChatMessage::user("now finish"),
        ];
        record_reinjection_overlaps(&messages, state);
        assert!(!state.recall_hot.is_empty(), "fixture must prime a hot set");
        messages
    }

    /// Every strategy protects the hot pair through its normal phases: at
    /// a budget reachable by evicting cold content, the hot result — an
    /// OLDER message than its cold peers — survives with its body intact,
    /// and hot_kept reports the protection.
    #[test]
    fn hot_keep_protects_reacquired_content_in_every_strategy() {
        let strategies: Vec<(&str, Box<dyn ContextGc>)> = vec![
            ("ring", Box::new(RingGc::default())),
            ("mark-sweep", Box::new(MarkSweepGc::default())),
            ("stack", Box::new(StackFrameGc::default())),
            ("semantic", Box::new(semantic_gc())),
        ];
        for (name, gc) in strategies {
            let mut state = GcState::default();
            let messages = hot_keep_fixture(&mut state);
            let hot_id = messages[3].id;
            let budget = estimate_tokens(&messages) - 150;

            let collected = gc.collect(messages, budget, &mut state);

            assert!(
                collected.iter().any(|message| message.id == hot_id
                    && message
                        .content
                        .as_deref()
                        .is_some_and(|content| content.contains("MX-7749-KESTREL"))),
                "{name}: the hot result (body intact) must survive the normal sweep"
            );
            assert!(
                state.hot_report.hot_kept > 0,
                "{name}: hot_kept must report the protection"
            );
        }
    }

    /// Hot-keep is a heuristic guard, not a hard one: under degrade
    /// pressure the hot pair drops while system + last user survive, and
    /// the window still reaches the budget (convergence unchanged).
    #[test]
    fn hot_keep_relaxes_under_degrade_pressure() {
        let strategies: Vec<(&str, Box<dyn ContextGc>)> = vec![
            ("ring", Box::new(RingGc::default())),
            ("stack", Box::new(StackFrameGc::default())),
            ("semantic", Box::new(semantic_gc())),
        ];
        for (name, gc) in strategies {
            let mut state = GcState::default();
            let messages = hot_keep_fixture(&mut state);
            let hot_id = messages[3].id;
            let last_user = messages[messages.len() - 1].id;
            let budget = estimate_tokens(&messages[..1]) + 60;

            let collected = gc.collect(messages, budget, &mut state);

            assert!(
                !collected.iter().any(|message| {
                    message.id == hot_id
                        && message
                            .content
                            .as_deref()
                            .is_some_and(|content| content.contains("MX-7749-KESTREL"))
                }),
                "{name}: degrade pressure overrides hot-keep"
            );
            assert!(collected.iter().any(|message| message.role == "system"));
            assert!(
                collected.iter().any(|message| message.id == last_user),
                "{name}: hard guards outrank hot-keep"
            );
            assert!(
                estimate_tokens(&collected) <= budget,
                "{name}: hot-keep must not break convergence"
            );
        }
    }

    /// The corpus side (t-1362): whatever collect() removes — dropped
    /// messages and in-place rewrites alike — joins collected_hashes via
    /// the marker wrapper, so every GcState-threading caller feeds the
    /// write-barrier without interpreter help.
    #[test]
    fn collect_feeds_the_written_corpus_including_rewrites() {
        // Ring: whole-message drops.
        let doomed = format!("doomed payload {}", "z".repeat(200));
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user(doomed.clone()),
            ChatMessage::user("live tail"),
        ];
        let mut state = GcState::default();
        let budget = estimate_tokens(&messages) - 40;
        let collected = RingGc {
            preserve_prefix: false,
            hot_keep: true,
        }
        .collect(messages, budget, &mut state);
        assert!(collected
            .iter()
            .all(|m| m.content.as_deref() != Some(&*doomed)));
        assert!(
            reinjection_chunk_keys(&doomed).is_subset(&state.collected_hashes),
            "dropped content chunks must join the corpus"
        );

        // Stack: a frame pop rewrites the assistant message and drops the
        // result — the result body must still reach the corpus.
        let fat_result = format!("frame result body {}", "w".repeat(300));
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("task"),
            ChatMessage::assistant(None, vec![tool_call("call-1")]),
            ChatMessage::tool("call-1", fat_result.clone()),
            ChatMessage::assistant(Some("done".into()), vec![]),
            ChatMessage::user("go on"),
        ];
        let mut state = GcState::default();
        let budget = estimate_tokens(&messages) - 60;
        let _ = StackFrameGc::default().collect(messages, budget, &mut state);
        assert!(
            reinjection_chunk_keys(&fat_result).is_subset(&state.collected_hashes),
            "popped frame result chunks must join the corpus"
        );
    }

    // ---- marker escalation (t-1370) --------------------------------------------

    /// One evict/re-fetch cycle: a window whose interior carries the
    /// payload under a fresh call id (and fresh envelope wall-clock),
    /// collected at a budget that forces the pair out. Returns the window
    /// collect() produced.
    fn escalation_cycle(
        gc: &dyn ContextGc,
        state: &mut GcState,
        call_id: &str,
        duration_ms: u64,
    ) -> Vec<ChatMessage> {
        let payload = shell_envelope("the batch access code is MX-7749-KESTREL\n", duration_ms);
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("count OK lines, then report the access code"),
            ChatMessage::assistant(None, vec![tool_call(call_id)]),
            ChatMessage::tool(call_id, payload),
            ChatMessage::assistant(Some("noted; continuing with the batch logs".into()), vec![]),
            ChatMessage::assistant(None, vec![tool_call(&format!("{call_id}-work"))]),
            ChatMessage::tool(
                format!("{call_id}-work"),
                // Distinct per cycle: only the PAYLOAD's fingerprint may
                // repeat across cycles.
                format!("work {call_id} {}", "x".repeat(2000)),
            ),
            ChatMessage::assistant(Some("processed".into()), vec![]),
            ChatMessage::user("keep going"),
        ];
        // Enough pressure that the payload pair (the oldest interior
        // group) is evicted, at a window large enough that the marker
        // ladder's re-collect rung can fund per-run markers — the
        // escalation line must not be coalesced away.
        let budget = estimate_tokens(&messages) - 100;
        gc.collect(messages, budget, state)
    }

    /// The same content evicted for the third time escalates: the marker
    /// stops offering a recovery affordance and names the honest exit,
    /// carrying the LATEST handle and the count.
    #[test]
    fn escalation_marker_fires_on_the_third_eviction_of_the_same_content() {
        let gc = RingGc {
            preserve_prefix: false,
            hot_keep: true,
        };
        let mut state = GcState::default();

        let first = escalation_cycle(&gc, &mut state, "call-1", 9);
        let second = escalation_cycle(&gc, &mut state, "call-2", 40);
        assert!(
            ![&first, &second]
                .iter()
                .any(|window| window.iter().any(|m| {
                    m.content
                        .as_deref()
                        .is_some_and(|c| c.contains("cannot stay in context"))
                })),
            "evictions one and two must not escalate"
        );

        let third = escalation_cycle(&gc, &mut state, "call-3", 77);
        let escalated: Vec<&ChatMessage> =
            third.iter().filter(|m| is_escalation_marker(m)).collect();
        assert_eq!(escalated.len(), 1, "third eviction escalates: {third:?}");
        let text = escalated[0].content.as_deref().unwrap();
        assert!(
            text.contains("'call-3' evicted 3 times"),
            "latest handle + count: {text}"
        );
        assert!(text.contains("do not re-fetch again"), "{text}");
        assert!(
            state.marker_summary.escalated >= 1,
            "summary must report the in-window escalation"
        );
    }

    /// Eviction counts — and the escalated marker bytes and ids — are a
    /// pure function of the collection sequence: two identical runs
    /// produce identical state and windows (replay stability).
    #[test]
    fn escalation_counts_and_markers_are_deterministic() {
        let run = || {
            let gc = RingGc {
                preserve_prefix: false,
                hot_keep: true,
            };
            let mut state = GcState::default();
            // Stable ids so the windows are byte-comparable across runs.
            let mut windows = Vec::new();
            for (call, duration) in [("call-1", 9u64), ("call-2", 40), ("call-3", 77)] {
                let payload =
                    shell_envelope("the batch access code is MX-7749-KESTREL\n", duration);
                let mut messages = vec![
                    ChatMessage::system("system"),
                    ChatMessage::user("count OK lines, then report the access code"),
                    ChatMessage::assistant(None, vec![tool_call(call)]),
                    ChatMessage::tool(call, payload),
                    ChatMessage::assistant(Some("noted".into()), vec![]),
                    ChatMessage::user("keep going"),
                ];
                for (index, message) in messages.iter_mut().enumerate() {
                    message.id = Uuid::from_u128(0x5eed_0000 + index as u128);
                }
                let budget = estimate_tokens(&messages[..2]) + 70;
                windows.push(gc.collect(messages, budget, &mut state));
            }
            (state.eviction_counts.clone(), windows)
        };
        let (counts_a, windows_a) = run();
        let (counts_b, windows_b) = run();
        assert_eq!(counts_a, counts_b, "counts must be deterministic");
        assert_eq!(
            windows_a, windows_b,
            "windows (ids included) must be deterministic"
        );
        // The payload fingerprint reached 3 despite three different call
        // ids and three different duration_ms values.
        assert!(counts_a.values().any(|count| *count == 3), "{counts_a:?}");
    }

    /// The quoted handle's digits never read as the marker's count: an
    /// escalated marker for 'call-7' counts its "evicted 3 times", not 7.
    #[test]
    fn escalated_marker_count_ignores_quoted_handle_digits() {
        let escalated = marker_message(Uuid::from_u128(1), escalation_marker_content("call-7", 3));
        assert_eq!(marker_evicted_count(&escalated), 3);
        // Normal markers still read their leading count.
        let normal = marker_message(
            Uuid::from_u128(2),
            "[gc: 5 evicted — shell call-2; recover: re-run the call — do not guess]".into(),
        );
        assert_eq!(marker_evicted_count(&normal), 5);
    }

    /// Escalation markers never fuse: fusion keeps the later text, which
    /// would silently delete the honest-exit instruction.
    #[test]
    fn escalation_markers_do_not_fuse_with_neighbors() {
        let escalated = marker_message(Uuid::from_u128(1), escalation_marker_content("call-7", 3));
        let normal = marker_message(
            Uuid::from_u128(2),
            "[gc: 2 evicted — shell call-9; recover: re-run the call — do not guess]".into(),
        );
        let fused = fuse_adjacent_markers(vec![escalated.clone(), normal.clone()]);
        assert_eq!(fused.len(), 2, "escalation marker must survive: {fused:?}");
        let fused = fuse_adjacent_markers(vec![normal, escalated]);
        assert_eq!(fused.len(), 2);
    }

    /// Mark-sweep's in-place elision joins the escalation ladder: content
    /// already evicted twice gets the honest-exit annotation instead of
    /// another "re-run the call".
    #[test]
    fn mark_sweep_elision_escalates_after_repeated_evictions() {
        let payload = format!("bulky incorporated result {}", "q".repeat(600));
        let mut state = GcState::default();
        state
            .eviction_counts
            .insert(content_fingerprint(&payload), 2);
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("task"),
            ChatMessage::assistant(None, vec![tool_call("call-el")]),
            ChatMessage::tool("call-el", payload),
            ChatMessage::assistant(Some("incorporated".into()), vec![]),
            ChatMessage::user("go on"),
        ];
        let budget = estimate_tokens(&messages) + 50; // under budget: elision only
        let collected = MarkSweepGc {
            preserve_prefix: false,
            hot_keep: true,
        }
        .collect(messages, budget, &mut state);
        let elided = collected
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call-el"))
            .expect("elided result message survives in place");
        let text = elided.content.as_deref().unwrap();
        assert!(
            text.contains("'call-el' evicted 3 times") && text.contains("cannot stay in context"),
            "elision must escalate: {text}"
        );
        assert!(
            state.marker_summary.escalated >= 1,
            "escalated elision must be reported on the summary"
        );
    }

    // ---- hard protection: system + last user survive any pressure (t-1367) ----
    //
    // t-1364's failure mode, pinned per strategy: a fat system message (the
    // runtime-guidance fragment) plus a budget small enough that the degrade
    // paths fire used to evict the LAST USER MESSAGE — the statement of the
    // live task — from ring and stack. The guards that only semantic carried
    // are now an invariant of every strategy.

    /// Fat system message + task + completed tool frames, at a budget the
    /// protected set barely fits: every degrade phase must fire, and the
    /// task statement must still be standing.
    fn adversarial_fixture() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system(format!("operations manual {}", "g".repeat(900))),
            ChatMessage::user("the task: count the WARN lines in build.log"),
            ChatMessage::assistant(None, vec![tool_call("call-1")]),
            ChatMessage::tool("call-1", format!("output {}", "x".repeat(600))),
            ChatMessage::assistant(Some("step one done".into()), vec![]),
            ChatMessage::assistant(None, vec![tool_call("call-2")]),
            ChatMessage::tool("call-2", format!("output {}", "y".repeat(600))),
        ]
    }

    fn assert_system_and_last_user_survive(strategy: &dyn ContextGc, budget: usize) {
        let messages = adversarial_fixture();
        let last_user = messages[1].id;
        let mut state = GcState::default();
        let collected = strategy.collect(messages.clone(), budget, &mut state);
        assert!(
            collected.iter().any(|message| message.role == "system"),
            "{} at budget {budget} dropped the system message: {collected:?}",
            strategy.name()
        );
        assert!(
            collected.iter().any(|message| message.id == last_user),
            "{} at budget {budget} evicted the live task (last user message): {collected:?}",
            strategy.name()
        );
        // Pair atomicity holds through the same degrade phases.
        let live_call_ids: BTreeSet<_> = collected
            .iter()
            .flat_map(|message| {
                message
                    .tool_calls
                    .iter()
                    .flatten()
                    .map(|call| call.id.clone())
            })
            .collect();
        for message in &collected {
            if let Some(id) = message.tool_call_id.as_deref() {
                assert!(
                    live_call_ids.contains(id),
                    "{} at budget {budget} orphaned tool result {id}",
                    strategy.name()
                );
            }
        }
    }

    #[test]
    fn ring_never_drops_system_or_last_user_under_any_pressure() {
        // The protected set (fat system + task) is ~330 tokens; sweep from
        // barely-fits down to absurd. Below the protected floor the window
        // legitimately ships over budget (the overflow paths own it) — but
        // the guards still hold.
        for budget in [800, 400, 340, 200, 50, 1] {
            for preserve in [true, false] {
                assert_system_and_last_user_survive(
                    &RingGc {
                        hot_keep: true,
                        preserve_prefix: preserve,
                    },
                    budget,
                );
            }
        }
    }

    #[test]
    fn stack_never_drops_system_or_last_user_under_any_pressure() {
        for budget in [800, 400, 340, 200, 50, 1] {
            for preserve in [true, false] {
                assert_system_and_last_user_survive(
                    &StackFrameGc {
                        hot_keep: true,
                        preserve_prefix: preserve,
                    },
                    budget,
                );
            }
        }
    }

    #[test]
    fn mark_sweep_never_drops_system_or_last_user_under_any_pressure() {
        // Mark-sweep only ever evicts complete/evictable lifecycles and user
        // messages stay Active, so the guarantee is structural — pinned here
        // so a future lifecycle change cannot silently lose it.
        for budget in [800, 400, 340, 200, 50, 1] {
            for preserve in [true, false] {
                assert_system_and_last_user_survive(
                    &MarkSweepGc {
                        hot_keep: true,
                        preserve_prefix: preserve,
                    },
                    budget,
                );
            }
        }
    }

    #[test]
    fn semantic_never_drops_system_or_last_user_under_any_pressure() {
        // Semantic pioneered the guards (t-1350); same adversarial sweep,
        // heuristic scoring path (no cached vectors).
        for budget in [800, 400, 340, 200, 50, 1] {
            for preserve in [true, false] {
                assert_system_and_last_user_survive(
                    &SemanticGc {
                        hot_keep: true,
                        preserve_prefix: preserve,
                        ..semantic_gc()
                    },
                    budget,
                );
            }
        }
    }

    #[test]
    fn ring_still_converges_when_protecting_the_task() {
        // The guards must not cost convergence where convergence is
        // possible: everything unprotected drops, the window lands under
        // budget, and the protected set is exactly what remains.
        let messages = adversarial_fixture();
        let protected_tokens = estimate_tokens(&[messages[0].clone(), messages[1].clone()]);
        let budget = protected_tokens + 20;
        let mut state = GcState::default();
        let collected = RingGc {
            hot_keep: true,
            preserve_prefix: true,
        }
        .collect(messages, budget, &mut state);
        assert!(
            estimate_tokens(&collected) <= budget,
            "ring must converge once only the protected set remains: {} > {budget}",
            estimate_tokens(&collected)
        );
        assert_eq!(collected.len(), 2, "{collected:?}");
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
            hot_keep: true,
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

    // ---- Eviction markers (t-1360) ------------------------------------------

    /// Ballast + a droppable pair + protected tail, roomy enough that the
    /// marker fits: the dropped tool result must leave a marker naming the
    /// call id and the re-run affordance.
    #[test]
    fn ring_drop_leaves_a_marker_with_call_id_and_affordance() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    "call-1",
                    "shell",
                    serde_json::json!({ "command": "cat config/access-code.txt" }),
                )],
            ),
            ChatMessage::tool("call-1", "x".repeat(900)),
            ChatMessage::assistant(Some("noted".into()), vec![]),
            ChatMessage::user("now finish the task"),
        ];
        let mut state = GcState::default();
        let budget = 220;
        let collected = RingGc {
            hot_keep: true,
            preserve_prefix: false,
        }
        .collect(messages, budget, &mut state);

        assert!(estimate_tokens(&collected) <= budget, "{collected:?}");
        let marker = collected
            .iter()
            .find(|message| is_eviction_marker(message))
            .expect("dropped pair must leave an eviction marker");
        let content = marker.content.as_deref().unwrap();
        assert!(
            content.contains("shell call-1"),
            "handle missing: {content}"
        );
        assert!(
            content.contains("re-run the call") && content.contains("do not guess"),
            "affordance missing: {content}"
        );
        assert_eq!(state.marker_summary.evicted_tool_results, 1);
        assert_eq!(state.marker_summary.markers, 1);
        assert!(!state.marker_summary.coalesced);
        assert!(!state.marker_summary.suppressed);
    }

    /// N consecutive drops aggregate into ONE marker line whose count is N,
    /// not N marker lines.
    #[test]
    fn consecutive_drops_aggregate_into_one_marker() {
        let mut messages = vec![ChatMessage::system("system")];
        for index in 0..4 {
            messages.push(ChatMessage::assistant(
                Some(format!("step {index} narration {}", "z".repeat(300))),
                vec![],
            ));
        }
        messages.push(ChatMessage::user("final question"));
        let mut state = GcState::default();
        let collected = RingGc {
            hot_keep: true,
            preserve_prefix: false,
        }
        .collect(messages, 250, &mut state);

        let markers: Vec<_> = collected
            .iter()
            .filter(|message| is_eviction_marker(message))
            .collect();
        assert_eq!(markers.len(), 1, "one run = one marker: {collected:?}");
        assert!(
            markers[0]
                .content
                .as_deref()
                .unwrap()
                .starts_with("[gc: 3 evicted"),
            "{markers:?}"
        );
    }

    /// A dropped recall result is identified by its memory query, with the
    /// recall affordance; a dropped user turn gets the ask-again affordance.
    #[test]
    fn markers_name_recall_queries_and_user_turns() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user(format!("old request {}", "q".repeat(400))),
            ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    "call-9",
                    "recall",
                    serde_json::json!({ "query": "deploy window" }),
                )],
            ),
            ChatMessage::tool("call-9", "### deploy-window\nTuesday 9am".repeat(20)),
            ChatMessage::assistant(Some("noted".into()), vec![]),
            ChatMessage::user("now the final step"),
        ];
        let mut state = GcState::default();
        let collected = RingGc {
            hot_keep: true,
            preserve_prefix: false,
        }
        .collect(messages, 200, &mut state);

        let text: String = collected
            .iter()
            .filter(|message| is_eviction_marker(message))
            .filter_map(|message| message.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("recall 'deploy window'"), "{text}");
        assert!(text.contains("recall the memory"), "{text}");
        assert!(text.contains("user turn 2"), "{text}");
        assert!(text.contains("ask the user again"), "{text}");
        assert_eq!(state.marker_summary.evicted_recalls, 1);
        assert_eq!(state.marker_summary.evicted_user_turns, 1);
    }

    /// Markers are part of collect()'s pure output: two runs from equal
    /// inputs produce byte-identical windows AND identical marker ids (no
    /// fresh UUIDs).
    #[test]
    fn markers_are_deterministic_including_ids() {
        let messages = stack_fixture();
        let mut state_a = GcState::default();
        let mut state_b = GcState::default();
        let a = RingGc {
            hot_keep: true,
            preserve_prefix: false,
        }
        .collect(messages.clone(), 200, &mut state_a);
        let b = RingGc {
            hot_keep: true,
            preserve_prefix: false,
        }
        .collect(messages, 200, &mut state_b);
        assert_eq!(a, b);
        assert_eq!(
            a.iter().map(|message| message.id).collect::<Vec<_>>(),
            b.iter().map(|message| message.id).collect::<Vec<_>>(),
            "marker ids must be deterministic"
        );
        assert_eq!(state_a.marker_summary, state_b.marker_summary);
    }

    /// Marker ids are never the dropped messages' own ids, so retention
    /// metrics (id survival) cannot mistake a marker for its content.
    #[test]
    fn markers_do_not_reuse_dropped_ids() {
        let messages = stack_fixture();
        let original_ids: BTreeSet<_> = messages.iter().map(|message| message.id).collect();
        let mut state = GcState::default();
        let collected = RingGc {
            hot_keep: true,
            preserve_prefix: false,
        }
        .collect(messages, 200, &mut state);
        for marker in collected.iter().filter(|m| is_eviction_marker(m)) {
            assert!(
                !original_ids.contains(&marker.id),
                "marker must mint a derived id, not reuse a dropped one"
            );
        }
    }

    /// Under further pressure a marker is itself droppable: a second
    /// collection at a tighter budget absorbs the old marker's count into
    /// its replacement instead of letting it vanish silently — and never
    /// ships the window over budget.
    #[test]
    fn markers_are_droppable_and_counts_are_absorbed() {
        let mut messages = vec![ChatMessage::system("system")];
        for index in 0..3 {
            messages.push(ChatMessage::assistant(
                Some(format!("early step {index} {}", "y".repeat(250))),
                vec![],
            ));
        }
        for index in 0..3 {
            messages.push(ChatMessage::assistant(
                Some(format!("late step {index} {}", "w".repeat(250))),
                vec![],
            ));
        }
        messages.push(ChatMessage::user("wrap up"));
        let ring = RingGc {
            hot_keep: true,
            preserve_prefix: false,
        };
        let mut state = GcState::default();
        let first = ring.collect(messages, 500, &mut state);
        let first_marker_count: usize = first
            .iter()
            .filter(|m| is_eviction_marker(m))
            .map(marker_evicted_count)
            .sum();
        assert!(first_marker_count > 0, "{first:?}");

        let second = ring.collect(first, 200, &mut state);
        assert!(estimate_tokens(&second) <= 200, "{second:?}");
        let second_marker_count: usize = second
            .iter()
            .filter(|m| is_eviction_marker(m))
            .map(marker_evicted_count)
            .sum();
        if second.iter().any(is_eviction_marker) {
            assert!(
                second_marker_count > first_marker_count,
                "the replacement marker must absorb the dropped marker's count: \
                 {second_marker_count} <= {first_marker_count}; {second:?}"
            );
        } else {
            assert!(
                state.marker_summary.suppressed,
                "markers may only disappear via the recorded suppression path"
            );
        }
    }

    /// The terminal degrade: when not even one line fits, markers are
    /// suppressed — recorded on the summary — and the window still
    /// converges exactly as the core strategy left it.
    #[test]
    fn markers_never_break_convergence_and_record_suppression() {
        let messages = adversarial_fixture();
        let protected_tokens = estimate_tokens(&[messages[0].clone(), messages[1].clone()]);
        let budget = protected_tokens + 5;
        let mut state = GcState::default();
        let collected = RingGc {
            hot_keep: true,
            preserve_prefix: false,
        }
        .collect(messages, budget, &mut state);
        assert!(estimate_tokens(&collected) <= budget);
        assert!(
            !collected.iter().any(is_eviction_marker),
            "no room for markers at this budget: {collected:?}"
        );
        assert!(state.marker_summary.suppressed);
    }

    /// Stack integration: results dropped by a frame pop are covered by the
    /// surviving `[frame ...]` annotation — no duplicate `[gc: ...]` line.
    /// (Budget roomy enough that the ring-fallback phase never fires; if an
    /// annotation is itself dropped later, a `[gc: ...]` marker rightly
    /// stands in for it.)
    #[test]
    fn stack_frame_pops_are_not_double_marked() {
        let messages = stack_fixture();
        let mut state = GcState::default();
        let collected = StackFrameGc::default().collect(messages, 200, &mut state);
        assert!(
            collected.iter().any(|message| message
                .content
                .as_deref()
                .is_some_and(|content| content.contains("[frame call-1:"))),
            "{collected:?}"
        );
        assert!(
            !collected.iter().any(is_eviction_marker),
            "frame-covered drops must not grow a duplicate marker: {collected:?}"
        );
        assert_eq!(state.marker_summary.evicted_tool_results, 0);
    }

    /// Cited-keep interplay (semantic): a marker for an evicted-but-cited
    /// message carries the citing handle.
    #[test]
    fn marker_for_cited_evicted_message_names_the_citer() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    "call-7",
                    "shell",
                    serde_json::json!({ "command": "audit" }),
                )],
            ),
            ChatMessage::tool("call-7", "audit output ".repeat(60)),
            ChatMessage::assistant(Some("per the output of call-7, proceed".into()), vec![]),
            ChatMessage::user("finish"),
        ];
        let build = build_eviction_markers(
            &messages,
            &[messages[0].clone(), messages[4].clone()],
            &BTreeMap::new(),
        );
        let text: String = build
            .markers
            .iter()
            .filter_map(|(_, marker)| marker.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("shell call-7 (cited by turn 4)"),
            "cited evicted results must carry the citing handle: {text}"
        );
    }

    // --- progress ledger (t-1373) --------------------------------------------

    fn find_ledger(window: &[ChatMessage]) -> Option<&ChatMessage> {
        window.iter().find(|message| is_gc_ledger(message))
    }

    #[test]
    fn ledger_appears_after_an_evicting_collection_and_names_completed_calls() {
        let messages = stack_fixture();
        let mut state = GcState::default();
        let collected = RingGc::default().collect(messages, 300, &mut state);

        let ledgers = collected
            .iter()
            .filter(|message| is_gc_ledger(message))
            .count();
        assert_eq!(ledgers, 1, "exactly one ledger instance: {collected:#?}");
        let content = find_ledger(&collected)
            .and_then(|message| message.content.as_deref())
            .unwrap();
        assert!(content.starts_with(GC_LEDGER_PREFIX));
        assert!(
            content.contains("call-1: shell(cargo test)"),
            "entries are call-id: tool(args-preview) one-liners: {content}"
        );
        assert!(
            content.contains("-> test output") || content.contains("[evicted]"),
            "entries carry outcome previews and eviction state: {content}"
        );
        assert!(
            content.contains("[evicted]") || content.contains("[in-window]"),
            "entries carry window state tags: {content}"
        );
        assert!(state.ledger_summary.present);
        assert_eq!(state.ledger_summary.entries, 2);
        assert_eq!(
            state.ledger_summary.calls,
            vec!["call-1".to_string(), "call-2".to_string()]
        );
        // Budget honesty: the window including the ledger converges.
        assert!(estimate_tokens(&collected) <= 300);
    }

    #[test]
    fn ledger_absent_until_something_is_evicted() {
        // A fired-but-eviction-free collection must stay a no-op: the
        // window is still the complete work record.
        let messages = stack_fixture();
        let budget = estimate_tokens(&messages) + 100;
        let mut state = GcState::default();
        let collected = RingGc::default().collect(messages.clone(), budget, &mut state);
        assert_eq!(collected, messages, "under budget = untouched");
        assert!(find_ledger(&collected).is_none());
        assert_eq!(state.ledger_summary, LedgerSummary::default());
        // The journal still learned the session's completed calls.
        assert_eq!(state.ledger.len(), 2);
    }

    #[test]
    fn ledger_is_replaced_not_appended_and_never_marked_as_an_eviction() {
        let messages = stack_fixture();
        let mut state = GcState::default();
        let first = RingGc::default().collect(messages, 300, &mut state);
        assert_eq!(
            first.iter().filter(|m| is_gc_ledger(m)).count(),
            1,
            "first collection writes the ledger"
        );
        let markers_before = state.marker_summary;

        // Re-collect the already-collected window: idempotent — the old
        // ledger is replaced (same content, same derived id), nothing is
        // dropped, and no marker counts the replacement as an eviction.
        let second = RingGc::default().collect(first.clone(), 300, &mut state);
        assert_eq!(
            second, first,
            "re-collecting a collected window is a no-op incl. the ledger"
        );
        assert_eq!(
            second.iter().filter(|m| is_gc_ledger(m)).count(),
            1,
            "replaced, never appended"
        );
        let markers_after = state.marker_summary;
        assert_eq!(
            markers_after.evicted_assistant_turns, 0,
            "the replaced ledger instance is bookkeeping, not an eviction"
        );
        assert_eq!(markers_after.markers, markers_before.markers);
    }

    #[test]
    fn ledger_is_deterministic_including_its_id() {
        let messages = stack_fixture();
        let run = |messages: Vec<ChatMessage>| {
            let mut state = GcState::default();
            RingGc::default().collect(messages, 300, &mut state)
        };
        let first = run(messages.clone());
        let second = run(messages);
        assert_eq!(first, second);
        assert_eq!(
            first.iter().map(|m| m.id).collect::<Vec<_>>(),
            second.iter().map(|m| m.id).collect::<Vec<_>>(),
            "ledger ids are derived from content, never minted"
        );
    }

    #[test]
    fn ledger_respects_the_budget_ladder_and_records_suppression() {
        let messages = stack_fixture();
        let mut state = GcState::default();
        // A budget with room for the hard guards and little else: the
        // ledger must degrade to suppression rather than overflow.
        let collected = RingGc::default().collect(messages, 60, &mut state);
        assert!(find_ledger(&collected).is_none());
        assert!(
            state.ledger_summary.suppressed,
            "no room for the ledger must be recorded, never silent"
        );
        assert!(!state.ledger_summary.present);
    }

    #[test]
    fn ledger_never_touches_the_pinned_prefix() {
        // Enough early ballast that the preserve-mode prefix allowance
        // pins real messages, then heavy pressure.
        let mut messages = vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user("please run the tests"),
        ];
        for index in 0..8 {
            let call_id = format!("call-{index}");
            messages.push(ChatMessage::assistant(
                Some(format!("step {index}")),
                vec![ToolCall::new(
                    &call_id,
                    "shell",
                    serde_json::json!({ "command": format!("make step-{index}") }),
                )],
            ));
            messages.push(ChatMessage::tool(
                call_id,
                format!("output {index} {}", "z".repeat(600)),
            ));
        }
        messages.push(ChatMessage::user("now finish up"));
        let budget = estimate_tokens(&messages) / 2;
        let boundary = cache_prefix_boundary(&messages, budget);
        let prefix = messages[..boundary].to_vec();

        let mut state = GcState::default();
        let collected = RingGc::default().collect(messages, budget, &mut state);
        assert!(
            !state.prefix_invalidated,
            "preserve mode with a ledger must keep the prefix byte-stable"
        );
        assert_eq!(
            &collected[..prefix.len()],
            &prefix[..],
            "the pinned prefix is byte-identical with the ledger in-window"
        );
        let ledger_index = collected
            .iter()
            .position(is_gc_ledger)
            .expect("heavy pressure with tool history writes a ledger");
        assert!(
            ledger_index >= prefix.len(),
            "the ledger lives in the tail region, never the pinned prefix"
        );
        // Tail placement: immediately before the trailing user turn.
        assert_eq!(
            collected[ledger_index + 1].role,
            "user",
            "the ledger sits immediately before the latest user turn"
        );
        assert_eq!(ledger_index + 2, collected.len());
    }

    #[test]
    fn ledger_caps_entries_and_coalesces_older_work() {
        let mut messages = vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user("do the work"),
        ];
        for index in 0..(MAX_LEDGER_ENTRIES + 3) {
            let call_id = format!("call-{index}");
            messages.push(ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    &call_id,
                    "shell",
                    serde_json::json!({ "command": format!("make step-{index}") }),
                )],
            ));
            messages.push(ChatMessage::tool(
                call_id,
                format!("output {index} {}", "w".repeat(3000)),
            ));
        }
        messages.push(ChatMessage::assistant(Some("done so far".into()), vec![]));
        let budget = estimate_tokens(&messages) / 2;
        let mut state = GcState::default();
        let collected = RingGc::default().collect(messages, budget, &mut state);
        let content = find_ledger(&collected)
            .and_then(|m| m.content.as_deref())
            .expect("ledger present");
        assert!(
            (1..=MAX_LEDGER_ENTRIES).contains(&state.ledger_summary.entries),
            "itemized entries are capped and non-empty: {}",
            state.ledger_summary.entries
        );
        assert!(
            content.contains("older work:") && content.contains("earlier calls completed"),
            "older entries coalesce into one line: {content}"
        );
        let newest = format!("call-{}:", MAX_LEDGER_ENTRIES + 2);
        assert!(
            content.contains(&newest) && !content.contains("call-0:"),
            "newest entries are itemized, oldest coalesced: {content}"
        );
    }

    #[test]
    fn ledger_carries_escalation_state_for_repeatedly_evicted_content() {
        let messages = stack_fixture();
        let mut state = GcState::default();
        // Two prior evictions of call-1's content: this collection makes
        // the third, which must surface the honest exit in the ledger.
        let fingerprint = content_fingerprint(
            messages
                .iter()
                .find(|m| m.tool_call_id.as_deref() == Some("call-1"))
                .and_then(|m| m.content.as_deref())
                .unwrap(),
        );
        state.eviction_counts.insert(fingerprint, 2);
        let collected = RingGc::default().collect(messages, 420, &mut state);
        let content = find_ledger(&collected)
            .and_then(|m| m.content.as_deref())
            .expect("ledger present");
        assert!(
            content.contains("call-1: shell(cargo test) -> ")
                && content.contains("[evicted 3x — do not re-fetch]"),
            "the ledger names the re-eviction loop and the exit: {content}"
        );
    }

    #[test]
    fn ledger_lines_never_become_write_barrier_chunks() {
        let messages = stack_fixture();
        let mut state = GcState::default();
        let collected = RingGc::default().collect(messages, 300, &mut state);
        let ledger = find_ledger(&collected).expect("ledger present");
        assert!(
            reinjection_chunk_keys(ledger.content.as_deref().unwrap()).is_empty(),
            "a digest describing content must never vouch for it"
        );
        assert!(
            !hot_mask(&collected, &state)[collected.iter().position(is_gc_ledger).unwrap()],
            "the ledger can never be write-barrier hot"
        );
    }

    #[test]
    fn ledger_absent_when_the_session_has_no_tool_history() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user(format!("first question {}", "a".repeat(900))),
            ChatMessage::assistant(Some(format!("first answer {}", "b".repeat(900))), vec![]),
            ChatMessage::user("second question"),
        ];
        let mut state = GcState::default();
        let collected = RingGc::default().collect(messages, 250, &mut state);
        assert!(
            find_ledger(&collected).is_none(),
            "no completed tool calls = nothing to digest"
        );
        assert_eq!(state.ledger_summary, LedgerSummary::default());
    }

    #[test]
    fn ledger_steps_back_over_a_trailing_open_tool_call() {
        let mut messages = stack_fixture();
        messages.push(ChatMessage::assistant(
            None,
            vec![ToolCall::new(
                "call-open",
                "shell",
                serde_json::json!({ "command": "cargo build" }),
            )],
        ));
        let mut state = GcState::default();
        let collected = RingGc::default().collect(messages, 700, &mut state);
        let ledger_index = collected
            .iter()
            .position(is_gc_ledger)
            .expect("ledger present");
        let open_index = collected
            .iter()
            .position(|m| {
                m.tool_calls
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .any(|call| call.id == "call-open")
            })
            .expect("open frame survives");
        assert!(
            ledger_index < open_index,
            "the ledger must not strand an open call from its future result"
        );
    }

    // --- Generational GC (t-1167) -------------------------------------------

    /// A window long enough that the nursery (last 8) is distinct from the
    /// interior: system + task + three fat cold frames + a fat cited frame
    /// + a fat hot frame, then a chatty 8-message tail.
    fn generational_fixture() -> Vec<ChatMessage> {
        let mut messages = vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user("audit the ingest pipeline and report the final total"),
        ];
        for step in 0..3 {
            messages.push(ChatMessage::assistant(
                Some(format!("Reading batch log {step}.")),
                vec![ToolCall::new(
                    format!("call-cold-{step}"),
                    "shell",
                    serde_json::json!({ "command": format!("cat logs/batch-{step}.log") }),
                )],
            ));
            messages.push(ChatMessage::tool(
                format!("call-cold-{step}"),
                format!("batch {step} noise {}", "x".repeat(900)),
            ));
        }
        messages.push(ChatMessage::assistant(
            None,
            vec![ToolCall::new(
                "call-cited",
                "shell",
                serde_json::json!({ "command": "cat audit/summary.txt" }),
            )],
        ));
        messages.push(ChatMessage::tool(
            "call-cited",
            format!("audit summary {}", "y".repeat(900)),
        ));
        messages.push(ChatMessage::assistant(
            None,
            vec![ToolCall::new(
                "call-hot",
                "shell",
                serde_json::json!({ "command": "cat config/access-code.txt" }),
            )],
        ));
        messages.push(ChatMessage::tool(
            "call-hot",
            format!("ACCESS CODE MX-7749-KESTREL {}", "z".repeat(400)),
        ));
        // The citation: a later message builds on call-cited by id.
        messages.push(ChatMessage::assistant(
            Some("Per the output of call-cited, the audit total carries.".into()),
            vec![],
        ));
        for step in 0..3 {
            messages.push(ChatMessage::user(format!("continue step {step}")));
            messages.push(ChatMessage::assistant(
                Some(format!("Working on step {step}.")),
                vec![],
            ));
        }
        messages.push(ChatMessage::user("what is the final total?"));
        messages
    }

    /// Mark the hot frame's content write-barrier hot, as the interpreter
    /// pre-pass would after a re-fetch of evicted content.
    fn mark_hot(messages: &[ChatMessage], state: &mut GcState) {
        let hot = messages
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-hot"))
            .expect("hot frame present");
        state
            .recall_hot
            .extend(reinjection_chunk_keys(hot.content.as_deref().unwrap()));
    }

    #[test]
    fn generational_tiers_assign_by_precedence() {
        let messages = generational_fixture();
        let mut state = GcState::default();
        mark_hot(&messages, &mut state);
        let gc = GenerationalGc::default();
        let tiers = gc.tiers(&messages, &state);
        assert_eq!(
            tiers,
            gc.tiers(&messages, &state),
            "tiering is deterministic"
        );

        let index_of = |call_id: &str| {
            messages
                .iter()
                .position(|m| m.tool_call_id.as_deref() == Some(call_id))
                .unwrap()
        };
        // The tail is the nursery.
        let nursery_start = messages.len() - DEFAULT_NURSERY_WINDOW;
        for (index, tier) in tiers.iter().enumerate() {
            if index >= nursery_start {
                assert_eq!(*tier, GcTier::Nursery, "tail message {index} is nursery");
            }
        }
        assert_eq!(tiers[0], GcTier::Hot, "system is a hard guard: hot");
        assert_eq!(
            tiers[index_of("call-hot")],
            GcTier::Hot,
            "write-barrier-hot content is hot"
        );
        assert_eq!(
            tiers[index_of("call-cited")],
            GcTier::Warm,
            "cited content is warm"
        );
        assert_eq!(
            tiers[index_of("call-cold-0")],
            GcTier::Cold,
            "uncited, unscored interior content is cold"
        );
    }

    #[test]
    fn generational_escalated_content_is_hot() {
        let messages = generational_fixture();
        let mut state = GcState::default();
        let cold0 = messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call-cold-0"))
            .unwrap();
        state.eviction_counts.insert(
            content_fingerprint(cold0.content.as_deref().unwrap()),
            EVICTION_ESCALATION_AFTER,
        );
        let gc = GenerationalGc::default();
        let tiers = gc.tiers(&messages, &state);
        let index = messages
            .iter()
            .position(|m| m.tool_call_id.as_deref() == Some("call-cold-0"))
            .unwrap();
        assert_eq!(
            tiers[index],
            GcTier::Hot,
            "content at the escalation threshold joins hot (t-1370 cost accounting)"
        );
    }

    #[test]
    fn generational_elides_cold_before_evicting_and_converges() {
        let messages = generational_fixture();
        let gc = GenerationalGc::default();
        // Mild pressure: eliding the cold bodies alone reaches the budget.
        let budget = (estimate_tokens(&messages) as f64 * 0.72) as usize;

        let mut state = GcState::default();
        mark_hot(&messages, &mut state);
        let mut state_again = state.clone();
        let collected = gc.collect(messages.clone(), budget, &mut state);
        let again = gc.collect(messages.clone(), budget, &mut state_again);
        assert_eq!(collected, again, "generational must be deterministic");
        assert_eq!(
            collected.iter().map(|m| m.id).collect::<Vec<_>>(),
            again.iter().map(|m| m.id).collect::<Vec<_>>(),
            "ids too"
        );

        assert!(
            estimate_tokens(&collected) <= budget,
            "must converge: {} > {budget}",
            estimate_tokens(&collected)
        );
        let report = state.tier_report;
        assert!(
            report.cold_elided > 0,
            "cold bodies elide first: {report:?}"
        );
        assert_eq!(
            report.evicted_cold + report.evicted_warm + report.evicted_hot + report.evicted_nursery,
            0,
            "elision alone reaches this budget — nothing whole-evicted: {report:?}"
        );
        // The elided results are STILL PRESENT as annotations (structure
        // survives; the mark-sweep behavioral shape). call-cold-0 sits in
        // the pinned prefix (pair-pinned) and keeps its body; the interior
        // cold frames elide.
        let elided = collected
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call-cold-2"))
            .expect("elided cold result keeps its message");
        assert!(
            elided
                .content
                .as_deref()
                .unwrap()
                .starts_with(EVICTION_MARKER_PREFIX),
            "cold body became an annotation: {:?}",
            elided.content
        );
        // Hot and nursery bodies are untouched.
        let hot = collected
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call-hot"))
            .expect("hot frame survives");
        assert!(
            hot.content.as_deref().unwrap().contains("MX-7749-KESTREL"),
            "the hot needle keeps its body"
        );
        let nursery_start = messages.len() - DEFAULT_NURSERY_WINDOW;
        for message in &messages[nursery_start..] {
            let survivor = collected
                .iter()
                .find(|m| m.id == message.id)
                .expect("nursery message survives");
            assert_eq!(
                survivor.content, message.content,
                "nursery bodies are untouched"
            );
        }

        // Idempotence: collecting the collected window is a no-op.
        let mut state_two = state.clone();
        let recollected = gc.collect(collected.clone(), budget, &mut state_two);
        assert_eq!(
            recollected.len(),
            collected.len(),
            "already-under-budget windows are not shrunk further"
        );
    }

    #[test]
    fn generational_evicts_cold_whole_only_after_elision_and_keeps_hot_and_cited() {
        let messages = generational_fixture();
        let gc = GenerationalGc {
            // Ignore mode: the prefix pin would otherwise shelter the cold
            // frames sitting at the front of this small window.
            preserve_prefix: false,
            ..Default::default()
        };
        // Heavy pressure: elision alone cannot reach this budget.
        let budget = (estimate_tokens(&messages) as f64 * 0.35) as usize;

        let mut state = GcState::default();
        mark_hot(&messages, &mut state);
        let collected = gc.collect(messages.clone(), budget, &mut state);
        assert!(
            estimate_tokens(&collected) <= budget,
            "must converge: {} > {budget}",
            estimate_tokens(&collected)
        );
        let report = state.tier_report;
        assert!(
            report.evicted_cold > 0,
            "cold groups evict under heavy pressure: {report:?}"
        );
        assert_eq!(
            report.evicted_nursery, 0,
            "the nursery is untouched while cheaper tiers remain: {report:?}"
        );
        if !report.hot_relaxed {
            assert_eq!(
                report.evicted_hot, 0,
                "hot survives every normal phase: {report:?}"
            );
            let hot = collected
                .iter()
                .find(|m| m.tool_call_id.as_deref() == Some("call-hot"))
                .expect("hot frame survives normal phases");
            assert!(
                hot.content.as_deref().unwrap().contains("MX-7749-KESTREL"),
                "the hot needle keeps its body through normal phases"
            );
        }
        if !report.warm_relaxed {
            assert!(
                collected
                    .iter()
                    .any(|m| m.tool_call_id.as_deref() == Some("call-cited")),
                "cited (warm) structure survives the normal phases"
            );
        }
    }

    #[test]
    fn generational_hard_guards_survive_terminal_pressure() {
        let messages = generational_fixture();
        let gc = GenerationalGc {
            preserve_prefix: false,
            ..Default::default()
        };
        let mut state = GcState::default();
        let collected = gc.collect(messages, 60, &mut state);
        assert!(
            collected.iter().any(|m| m.role == "system"),
            "system survives terminal pressure"
        );
        assert!(
            collected.iter().rfind(|m| m.role == "user").is_some(),
            "a user message survives terminal pressure"
        );
        assert!(
            state.tier_report.floor_relaxed,
            "terminal pressure relaxes the floor"
        );
    }

    #[test]
    fn generational_warm_by_similarity_requires_cached_vectors() {
        let messages = generational_fixture();
        let gc = GenerationalGc::default();
        let state = GcState::default();
        // No embeddings cached: no message is warm-by-similarity; the only
        // warm member is the cited one (citation-only mode, docs/GC.md).
        let tiers = gc.tiers(&messages, &state);
        let warm: Vec<usize> = tiers
            .iter()
            .enumerate()
            .filter(|(_, tier)| **tier == GcTier::Warm)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(warm.len(), 1, "citation-only warm tier without vectors");
    }
}
