# agentd Context GC Design

Status: **`ring`, `mark-sweep`, `stack`, and `semantic` are implemented**
(with `--gc`, `--gc-threshold`, `--gc-log`, `--gc-timing`, the
`truncate_oversized_message` pre-pass, pair atomicity, and the eval
harness). **`stack` is the default strategy** (t-1348, promoted on the
t-1339 strategy-matrix data). **`--gc-cache preserve|ignore` is
implemented** and `preserve` is the default: the system prompt plus the
oldest ~25% of the budget are pinned as the stable cache prefix, eviction
happens in the interior, and ring falls back to front-drop (reported via
`cache_invalidated` on the `gc_collect` event, which is observed per
collection rather than static per strategy) only when preserving cannot
reach the budget. `tests/gc_evals.rs::gc_cache_preserve_keeps_prefix_stable`
gates the preserve behavior. Every timing composes with the
collect-on-overflow backstop (t-1343, see Trigger Policy). The `gc_collect`
event reports `dropped_count`, not `frames_popped`.

## Defaults and quick reference

Current defaults: `--gc stack --gc-cache preserve --gc-timing threshold
--gc-threshold 0.85`. (The SDK's in-process `Runner` runs no GC; SDK
`Session`s spawn the `agent` CLI and inherit these defaults.)

| Knob | What it does | When it wins | Flag |
|---|---|---|---|
| `stack` (default) | Pops completed tool frames to one-line `[frame ...]` annotations; ring fallback otherwise | Tool-heavy sessions; best replay-completion (never dropped the task statement) | `--gc stack` |
| `ring` | Drops oldest messages first | Chat-only windows; simplest, most predictable | `--gc ring` |
| `mark-sweep` | Evicts dead call/result lifecycles; annotates incorporated results | Tool-heavy *batch* runs with `--gc-cache ignore` (best raw reduction) | `--gc mark-sweep` |
| `semantic` | Drops messages semantically distant from the recent thread (abandoned tangents first) | Long meandering sessions with dead ends; needs a registry `embeddings` entry | `--gc semantic` |
| `cited-keep` (default on, semantic only) | Messages cited by later ones (id mentions, `context_refs`) join the protected set | A recent turn builds on an old, off-topic tool result | `--gc-cited-keep false` to disable |
| `none` | No GC; overflow is a hard error | Deterministic evals | `--gc none` |
| cache `preserve` (default) | Pins system prompt + oldest ~25% of budget; evicts interior | Cached providers: zero prefix invalidations | `--gc-cache preserve` |
| cache `ignore` | Front-drop allowed; maximal reclaim | No prompt caching in use | `--gc-cache ignore` |
| timing `threshold` (default) | Collect when the estimate crosses budget x 0.85 | Almost always; cheapest proactive timing | `--gc-timing threshold` |
| timing `catch-overflow` | No estimate trigger; collect + retry on provider overflow | Untrustworthy token estimates | `--gc-timing catch-overflow` |
| timing `eager` / `every:N` | Collect every (Nth) infer call | Rarely — eager is pure waste; every:N buys nothing over threshold | `--gc-timing eager\|every:N` |

## Choosing a strategy

Grounded in the t-1339 matrix (360 cells: 5 cases x 3 pressures x 4
timings x 3 strategies x 2 cache policies):

- **`stack` (default).** The best retention/robustness trade: never dropped
  the current task statement in any of 120 cells (ring+ignore lost it in
  24/60 — the worst replay-completion failure observed), retains 63–70% of
  messages vs ring's 51–53% at the same token cost, and keeps a semantic
  record of popped frames at ~1% of their tokens. On chat-heavy windows it
  degrades exactly to ring, so it has no pathological case. Cost: old tool
  result *bodies* go first — anything needed verbatim from an old result is
  gone.
- **`mark-sweep`.** Best raw reduction on tool-heavy batch shapes (83.9%
  while keeping every message on the long tool-heavy case). But it does
  *nothing* on pure chat (0.0% reduction, all 12 chat-heavy cells hard
  overflow) and its convergence is best-effort: it failed to reach budget
  in 30/60 preserve cells. Use it for tool-heavy batch workloads with
  `--gc-cache ignore`; never pair it with `preserve`.
- **`ring`.** Simplest and most predictable; right for pure chat.
  (Historical: before the t-1367 hard guard, ring with `--gc-cache ignore`
  front-dropped the last user message under pressure — 24/60 t-1339 cells,
  and 24/60 again on the current fixture set. The last user message is now
  hard-protected in every strategy; expected drops are zero, asserted by
  the matrix.)
- **`semantic`** (t-1350). Scores each message by cosine similarity between
  its embedding and the centroid of the last N messages, and drops the most
  distant first — conversational dead ends and abandoned tangents go before
  older but on-topic history, which no position-based strategy can do. On
  the tangent-abandoned fixture at gate pressure it drops 88% of the
  tangent while retaining 100% of the relevant thread (stack: 25%/89%
  preserve, 12%/78% ignore), without regressing stack's replay-completion
  on any existing fixture class. When it wins: long meandering sessions
  that explore and abandon approaches. Costs: one embeddings API call per
  collection pre-pass for uncached messages (visible as
  `gc_semantic_embed{embedded,cached,failed}` under `--gc-log` — embedding
  tokens are billed by the provider but are orders of magnitude cheaper
  than chat tokens). Requires a model-registry `embeddings` entry (the same
  one memory retrieval uses, t-1340); without one — or when the endpoint
  fails, or under replay — scoring degrades to a deterministic recency
  heuristic (ring's oldest-first ordering) rather than erroring. On top of
  the hard guards every strategy now carries (system + last user + pair
  atomicity — see Invariants; semantic is where they originated), the last
  `--gc-semantic-window` messages (default 8) are immune;
  `--gc-semantic-floor` (default 0.25 cosine) keeps plausibly-related
  messages for a second pass. With `--gc-cited-keep`
  (default on, t-1351) messages explicitly cited by later ones —
  tool-call-id mentions in text, `infer` `context_refs` — join the
  protected set during the normal sweep phases, so a semantically distant
  result a recent message builds on ("per the output of call-X…") survives;
  see the Citation signals section.
- **Cache `preserve`.** Delivered exactly what it promises: 0 prefix
  invalidations across all 180 preserve cells vs 733 across ignore cells.
  Every invalidation is a full-window re-read at provider prices, so on
  cached providers `preserve` is strictly cheaper; only switch to `ignore`
  if you have confirmed you don't use prompt caching.
- **Timing.** Keep `threshold`. `eager` pays 2–4x the collections for
  identical output. `every:N` used to leave windows over budget between
  collections (7/15 ring and stack cells); the collect-on-overflow
  backstop (t-1343) now collects before any over-budget dispatch
  regardless of timing, so this is a redundancy question, not a safety
  one.

## Overview

The agent context window is a fixed-size buffer, not an infinite log.
We should treat it as such: apply garbage collection to keep it under budget
rather than crashing with `context_overflow` when it fills up.

This doc specifies the `--gc` flag, a `ContextGc` trait, and the first two
implementations to build: `MarkSweep` and `StackFrame`. The name is deliberate
— this is GC, not "context compaction." Calling it what it is makes the design
constraints and tradeoffs legible.

---

## CLI Flags

```
agent --gc <strategy>          # default: stack
agent --gc-threshold <0.0-1.0> # trigger GC at this fraction of budget (default: 0.85)
agent --gc none                # disable GC entirely (hard overflow = error)
agent --gc-log                 # emit gc_collect trace events (for debugging)
agent --gc-cache <mode>        # prefix-cache policy: preserve | ignore (default: preserve)
agent --gc-timing <when>       # threshold | catch-overflow | eager | every:N (default: threshold)
agent --gc-semantic-window <N> # semantic: recent-window size (centroid + recency floor; default: 8)
agent --gc-semantic-floor <f>  # semantic: similarity floor (default: 0.25)
agent --gc-cited-keep <bool>   # semantic: protect messages cited by later ones (default: true)
```

`--gc none` restores the current behavior and is important for deterministic
evals that must not have GC noise in their context.

`--gc-threshold` is tunable because different workloads have different optimal
trigger points: long tool chains benefit from earlier GC, short chat sessions
don't need it at all. Make it tunable from day one.

`--gc-log` emits structured trace events on every collection:
```json
{"type": "gc_collect", "strategy": "ring", "tokens_before": 90000, "tokens_after": 60000, "cache_invalidated": true, "dropped_count": 12}
```
This is essential for debugging task failures — you want to know which messages
were dropped or summarized.

`--gc-cache` selects the prompt-caching policy (see the Prompt Caching section
for the full guide):
- `preserve` (default): GC never invalidates the cached prefix. Old messages
  are dropped/summarized from the *middle* of the window, leaving the cached
  prefix breakpoint stable. Best for users who cache aggressively.
- `ignore`: GC optimizes purely for token reduction and may drop from the
  front, invalidating the cache. Best for users who don't use prompt caching
  and want maximal context savings per turn.

Both modes are first-class and supported indefinitely — neither is "the right
one." Performance-optimizing users and cache-optimizing users have genuinely
opposed needs, so the toggle is permanent, not a migration step.

`--gc-timing` decouples *when* GC runs from *what* it reclaims (t-1151).
Token estimates diverge from provider tokenizers, so an estimate-driven
threshold can sit idle while the provider hard-rejects the prompt (smith died
exactly this way on t-1145: 261k real tokens against a configured 400k budget
whose 0.85 trigger never fired). The timings:

- `threshold` (default): collect when the estimate crosses
  `context_budget * gc_threshold` — the historical behavior.
- `catch-overflow`: no estimate-based trigger at all; the provider is the
  source of truth. On an infer error classified as a context overflow, GC
  collects to `estimate/2` and retries the same turn, halving again per cycle
  (up to 3 cycles) before failing cleanly. The retry stays inside the one
  InferCall/InferResult trace pair; each cycle is visible as a
  `gc_collect{trigger: "context_overflow", cycle: k}` event. A target that
  the provider accepted is remembered for the rest of the session turn, so
  later calls collect proactively instead of paying a failed request each.
  Requires a GC strategy (`--gc ring` or `mark-sweep`).
- `eager`: collect to the threshold target before every infer call.
- `every:N`: collect to the threshold target on every Nth infer call.

Every timing is composed with the **collect-on-overflow backstop**
(t-1343): if the assembled prompt would exceed the full context budget at
Infer time and the timing policy did not fire, a collection runs before
dispatch anyway. This closes the gap where `every:N` (and a threshold
configured above 1.0) could dispatch over-budget windows between scheduled
collections.

The gc_collect event carries `timing`, `target_budget`, and `reason`
(`scheduled` | `backstop` | `overflow`) fields so `--gc-log` makes all
four timings — and the backstop — observable.

Strategies (planned, not all implemented at once):

| Name           | Description                                      | Status      |
|----------------|--------------------------------------------------|-------------|
| `none`         | No GC; overflow = hard error                     | implemented |
| `ring`         | Drop oldest messages when buffer fills           | implemented |
| `mark-sweep`   | Evict "dead" sections by type annotation         | implemented |
| `stack`        | Pop completed tool-call frames to summaries      | implemented (default) |
| `semantic`     | Drop messages semantically distant from the recent thread | implemented |
| `generational` | Hot/warm/cold compaction (JVM-style)             | future (t-1167; consumes the citation + recall signals below) |
| `refcount`     | Dependency-graph reachability eviction           | future (the citation graph below is its reachability structure) |

Cross-strategy modifier:

| Name         | Description                                                    | Status      |
|--------------|----------------------------------------------------------------|-------------|
| `cited-keep` | Messages cited by later ones join the protected set (t-1351)   | implemented for `semantic` (`--gc-cited-keep`, default on) |

---

## Trait Design

**All GC strategies MUST be stateless.** A strategy is a pure function of
`(messages, budget, gc_state)`. It owns no lifecycle state of its own; any
state that must persist across turns (mark-sweep lifecycle tags, stack-frame
status) lives in a `GcState` value owned by the `Runtime`
and threaded in by `&mut`. This keeps strategies trivially testable, swappable,
and deterministic — two strategies can be benchmarked on the same trace with
no hidden internal state to reset between runs.

```rust
/// Per-runtime GC state. Owned by the Runtime, threaded into every collect().
/// Strategies that need no state simply ignore it.
#[derive(Default)]
pub struct GcState {
    /// mark-sweep lifecycle tags, keyed by stable ChatMessage UUID
    lifecycle: HashMap<MsgId, LifecycleState>,
    /// stack-frame status, keyed by tool-call id
    frames: HashMap<FrameId, FrameStatus>,
}

/// A GC strategy operates on the agent's message history.
/// It is called when token usage exceeds a threshold.
///
/// Strategies are STATELESS: pure functions of (messages, budget, state).
/// No wall-clock reads, no HashMap iteration order in output — collect()
/// must be deterministic so the strategies themselves can be eval'd.
pub trait ContextGc: Send + Sync {
    /// Compact `messages` so total estimated tokens <= `budget`.
    /// `state` carries any cross-turn metadata (owned by Runtime).
    /// Returns the compacted message list.
    fn collect(
        &self,
        messages: Vec<ChatMessage>,
        budget: usize,
        state: &mut GcState,
    ) -> Vec<ChatMessage>;

    fn name(&self) -> &'static str;

    /// Whether this strategy preserves the cached prefix (see Prompt Caching).
    fn cache_preserving(&self) -> bool { false }
}

pub enum GcMode {
    None,
    Ring(RingGc),
    MarkSweep(MarkSweepGc),
    StackFrame(StackFrameGc),
}
```

Wired into `Runtime` as `gc: GcMode` plus `gc_state: GcState`. Called in the
turn loop before each `provider.chat()` call when
`estimate_tokens(messages) > budget * threshold` (threshold ~0.85 to give
headroom for the response). Because strategies are stateless, `gc` itself is
`&self` everywhere; only `gc_state` is `&mut`.

---

## Strategy 1: Ring Buffer (default)

**Policy:** When over budget, drop oldest `assistant`/`user`/`tool` messages
until under budget. Never drop the system prompt or the last user message
(the t-1367 hard guard — see Invariants).

**Rationale:** Simplest. Correct assumption that recent context is more
valuable. Stateless — no metadata required.

**Implementation sketch:**
```rust
pub struct RingGc;

impl ContextGc for RingGc {
    // Ring needs no cross-turn state, so it ignores `_state` entirely.
    fn collect(&self, messages: Vec<ChatMessage>, budget: usize, _state: &mut GcState) -> Vec<ChatMessage> {
        // partition: system messages are pinned
        let (system, mut rest) = messages.into_iter().partition(|m| m.role == Role::System);
        while estimate_tokens(&system) + estimate_tokens(&rest) > budget {
            if rest.is_empty() { break; }
            rest.remove(0);  // drop oldest
        }
        system.into_iter().chain(rest).collect()
    }
}
```

**Pair-atomicity (CRITICAL — implement and test first):** Tool result messages
must be dropped together with their paired tool call. Orphaned tool results
cause hard API errors. Drop pairs atomically. Write a failing test for this
before writing `RingGc` itself — it's the most likely source of subtle bugs.

---

## Strategy 2: Mark-Sweep

**Policy:** Tag each `ChatMessage` (or PromptIR `Section`) with a lifecycle
annotation. A sweep pass evicts messages in "dead" states.

**Lifecycle states:**
```
active    — in-progress tool call or pending reasoning
complete  — tool call + result pair, result already incorporated into a summary
evictable — explicitly marked dead (e.g., large blob only needed once)
pinned    — never evict (system prompt, key decisions)
```

**How tagging works:**
- Tool call + result pairs are marked `complete` once the *next* assistant
  message references or incorporates the result.
- Large `tool_result` messages (e.g., `read_file` output) are tagged
  `evictable` after being incorporated — replaced with a one-line annotation
  in the eviction-marker family (t-1360):
  `[gc: result elided — read_file /path (call-1); recover: re-run the call — do not guess]`
- System prompt and PromptIR `pinned` sections are always `pinned`.

**Implementation:**
```rust
// Stateless: lifecycle tags live in the threaded-in `GcState.lifecycle`,
// NOT in the strategy struct.
pub struct MarkSweepGc;

impl ContextGc for MarkSweepGc {
    fn collect(&self, messages: Vec<ChatMessage>, budget: usize, state: &mut GcState) -> Vec<ChatMessage> {
        // read/update state.lifecycle keyed by msg.id (stable UUID)
        // ...
    }
}
```

**Stable message IDs:** `ChatMessage` gets a stable `id: Uuid`, assigned at
construction. We break the schema rather than carry a surrogate-key workaround:
there's no legacy persistence to preserve here, and a real ID is the correct
foundation for both mark-sweep keying and any future refcount/DAG GcState.

```rust
pub struct ChatMessage {
    pub id: Uuid,        // NEW: stable, assigned on construction
    pub role: Role,
    // ...existing fields...
}
```

The earlier `(index, hash)` surrogate is dropped — index-based keys are
fragile under reordering/compaction and the hash collides on duplicate
content. The UUID is the canonical key into `GcState.lifecycle` and the
natural anchor if we later content-address shared subtrees.

**Integration with PromptIR:** PromptIR sections already have `SectionId` and
`Priority`. `MarkSweepGc` can operate directly on PromptIR sections when in
IR mode, using existing `priority` and `composition` fields to guide eviction.

---

## Strategy 3: Stack Frames

**Policy:** Model each tool invocation + result as an activation frame.
When a frame is "complete" (result received, no pending deps), replace the
full call+result pair with a one-line summary annotation.

**Frame lifecycle:**
```
open     — tool call sent, awaiting result
complete — result received
popped   — replaced with summary annotation
```

**Summary format:** `[frame <call_id>: <tool_name>(<summary_of_args>) → <summary_of_result>]`
— plus, when the result preview truncates, an explicit `— evicted; re-run
to recover` clause (t-1360: an annotation that reports the call happened
while silently withholding its body invites confabulation; one that names
the eviction and the affordance invites recovery). The call id is the
recovery handle.

**Summarization:** Use pure heuristics for v1 — truncate tool results to first
N chars + "…[truncated]". Do NOT use an LLM summarization call for v1: it adds
latency and tokens (ironic). Reserve the LLM-assisted `stack-smart` variant
for after the eval harness exists to justify the tradeoff.

**Key insight:** This is the most space-efficient strategy for tool-heavy
agents. A deep chain of `bash` + `read_file` turns can compress from 20k tokens
to ~200 tokens of annotations without losing the semantic record.

---

## Citation signals (t-1351): RC-via-citation and the recall write-barrier

Position (`ring`, `stack`) and topic (`semantic`) are *proxies* for whether a
message still matters. Citation is direct evidence: a fat tool result that no
later message references is dead weight; one that a later turn quotes by id or
re-pulls through `context_refs` is load-bearing regardless of its age or its
topic. This section defines the citation graph, the `cited-keep` modifier
built on it, and the recall-overlap write-barrier signal — the inputs the
future `generational` (t-1167) and `refcount` strategies are designed against.

### The citation graph

Nodes are the window's messages (keyed by stable `ChatMessage` UUID); the
interesting targets are tool-result messages. Edges point **citing message →
cited message** and come in three kinds, ordered by precision:

| Edge kind | Source | Semantics |
|---|---|---|
| `context_refs` | An `infer` tool call whose arguments carry `context_refs: [ids]` (t-1344) | Explicit citation **by construction**: the model asked for that tool result to be re-materialized into a child's context. Highest precision. |
| id-mention | A message's *text content* contains a tool-call id minted earlier in the window, at token boundaries (`call-1` does not match inside `call-10`) | The model referred to the call/result by name ("per the output of call-X, proceed with…"). The structural pair members — the assistant message that *issued* the call and the tool message that *answers* it — carry the id by construction and are excluded; only third-party mentions are citations. |
| recall-overlap | A `recall` tool result re-injects content whose hash matches content already in (or previously collected from) the window | Not a static edge but a temporal **re-reference event** — see the write-barrier section below. It lives in `GcState`, not in the per-window graph, because it needs cross-collection memory. |

The citation target is the tool-*result* message when it exists in the window
(that is where the tokens are), else the dispatching call message. Protecting
the result implicitly protects its dispatching call: pair atomicity means no
sweep can drop a group that would pull a protected member out.

`parent_op_id` lineage (t-1347) links sub-infer children to their dispatching
calls **in the trace**. It corroborates `context_refs` edges for offline
analysis (the t-1349 behavioral eval can join both), but it is not visible in
the message window, so it is deliberately *not* a `collect()` input — the
graph a strategy sees is computable from the window alone.

**Out of scope, on purpose: content-similarity citation.** A later message
that paraphrases a tool result without naming it is *similar* to it — and
similarity is `SemanticGc`'s mechanism, already scored by embeddings. Keeping
the mechanisms orthogonal is the design: **similarity says "on-topic",
citation says "load-bearing"**, and a message can be either, both, or
neither. Blurring citation into fuzzy content matching would just rebuild a
worse SemanticGc inside the citation extractor.

### The orthogonality 2x2

| | **cited** | **uncited** |
|---|---|---|
| **on-topic (similar to recent thread)** | Keep with high confidence — both signals agree. | Keep under normal pressure — semantic score already protects it; drops only in later phases. |
| **distant (dissimilar)** | **The gap this task closes.** Semantic-only GC drops it first (below the floor); `cited-keep` retains it. Canonical case: an old lookup on another topic that a recent message builds on ("per the output of call-X…"). | Drop first — the abandoned tangent. Both signals agree it is dead. |

### Determinism: extraction lives INSIDE collect()

Citation extraction is pure text analysis over exactly the inputs `collect()`
already has: string scans for ids and `context_refs` arrays over the window.
It is stateless, deterministic, synchronous, and LLM-free — so it runs
**inside `collect()`** (`CitationGraph::extract`, `cited_mask`). This is the
opposite placement from embeddings, and the invariant explains why: the GC
invariant bans *provider calls and nondeterminism* inside `collect()`, not
computation. Embeddings need an async provider → pre-pass + `GcState` cache
(t-1350). Citations need neither, so no cache, no pre-pass, no degrade mode —
`collect()` stays a pure function of `(messages, budget, state)` with the
graph derived on the fly.

The recall-overlap signal is the exception that proves the rule: deciding
"this recall re-injected something we *previously collected*" requires
memory of past collections, which no pure function of the current window
has. So it lives in `GcState` (`recall_hot`, `collected_hashes`), written by
the interpreter pre-pass — the same home and the same write-side/read-side
split as the t-1350 embedding cache.

### `cited-keep`: the modifier, and what each strategy can do with the graph

`cited-keep` adds cited messages to a strategy's *protected set* during the
normal sweep phases. It is a heuristic guard with the same strength as
semantic's recency floor — weaker than the preserve-prefix billing contract
and the system/last-user hard guards — so in the degrade phases it relaxes
together with the floor: under enough pressure a cited message still drops
before the window overflows the model.

- **`semantic` + `cited-keep` (implemented, `--gc-cited-keep`, default on).**
  The exact integration point the t-1350 report named: cited-but-
  semantically-distant messages survive phases 1–2. Gated by the
  cited-distant eval fixture (below).
- **`ring`/`stack`/`mark-sweep` + `cited-keep` (future).** Same shape: skip
  cited atomic groups in the primary sweep, take them only in the degrade
  pass. For `stack`, a cited frame should also resist *popping* (the
  citation refers to the result body, which popping destroys).
- **`refcount` (future).** The graph is the strategy: roots = system + last
  user + recency floor; keep what is reachable over citation edges, evict
  unreachable-first in topological age order.

### What generational (t-1167) inherits

Generational GC needs promotion signals; this task builds both, so t-1167
starts from observed re-reference behavior instead of speculation:

- **Citation in-degree → warm.** A result cited once is demonstrably
  load-bearing: promote out of the nursery ("young, evict cheaply") into
  warm ("referenced; compact, don't drop").
- **Recall-overlap → hot.** A recall hit that re-injects content already
  seen — especially content *previously collected* — is a write barrier
  firing in the JVM sense: a mutation (the recall) just created a reference
  from the live working set to old-generation data. That content is hot;
  re-evicting or re-dropping it thrashes (evict → recall → re-inject →
  evict). `GcState.recall_hot` is exactly the hot-set membership,
  `collected_hashes` the old-generation extent.
- **Neither signal + semantically distant → cold.** The 2x2's bottom-right
  cell, already handled by semantic today.

### The recall-overlap write-barrier (v1 mechanics)

Recorded by `gc::record_recall_overlaps`, called from the interpreter
pre-pass (`interpreter::collect_prompt`) before the strategy runs:

- A window message is a *recall result* when its `tool_call_id` resolves to
  a tool call named `recall` (the agent loop's memory tool).
- Its hit contents (the JSON array of `SourceResult`s; the whole text as one
  chunk when it does not parse) are content-hashed (trimmed) and
  membership-checked against (a) the hashes of every other window message's
  content and (b) `GcState.collected_hashes` — contents of messages dropped
  by earlier collections this run. v1 is exact-hash membership; fuzzy
  overlap is future work and must stay out of `collect()` regardless.
- Matches land in `GcState.recall_hot` and are reported on the `gc_collect`
  event as `recall_overlap_events` (this collection) and `recall_hot`
  (cumulative set size), so the t-1349 behavioral eval can observe the
  signal. Both sets are runtime-only — `GcState` never serializes into
  checkpoints — and are bounded by the run's own history.
- **No strategy consumes it yet.** It is deliberately signal-only: t-1167's
  generational design is its first customer.

---

## Eviction markers (t-1360): the model must know WHAT was dropped

Three behavioral eval rounds (t-1349, t-1364, the t-1367 re-run —
evals/gc/README.md) found the same failure: when a collection silently
removes an early tool result the task needs later, models **fabricate**
its content (confidently wrong access codes) instead of recovering it or
admitting loss — and do-not-guess guidance alone did not stop it. Silent
removal is the confabulation machine; eviction visibility is the
mechanism-level fix.

**The mechanism.** When any strategy's `collect()` drops messages, it
leaves a compact marker line (an ordinary assistant message) at the gap:

```
[gc: 2 evicted — shell call-3; recover: re-run the call — do not guess]
[gc: 1 evicted — recall 'deploy window'; recover: recall the memory — do not guess]
[gc: 3 evicted — user turn 2, assistant turn 4; recover: ask the user again — do not guess]
[gc: earlier context compacted — 41 messages evicted; recover: re-run tool calls, recall memories, or ask the user — do not guess]
```

Kind, identifying handle (tool-call id — the re-run-by-id and citation
handle; recall query; turn ordinal), recovery affordance. An
evicted-but-cited message carries its citing handle (`cited by turn N` —
the cited-keep interplay). Strategy integration is honest, not
duplicated: a stack frame pop's surviving `[frame call-id: ...]`
annotation IS the marker for its results (never double-marked), and
mark-sweep's in-place elision annotation is the same `[gc: ...]` family.

**Marker economics (all enforced by the offline matrix):**

- **Cheap + aggregatable:** a dropped 2000-token result becomes one
  ~30-token line; N consecutive drops share one line (item list capped,
  `+K more`).
- **Budget-honest:** markers count toward the window budget. The ladder
  in `with_eviction_markers`: per-run markers when the sweep's overshoot
  funds them → re-collect with the marker cost reserved (rejected if it
  would invalidate the original budget's pinned prefix when the core
  didn't — markers never break the preserve billing contract) → one
  coalesced line → recorded suppression. A collection never ships over
  budget because of its own markers; convergence contracts are exactly
  the core strategies'.
- **Droppable, never silently:** markers are unprotected assistant
  messages; a later collection that drops one absorbs its count into the
  replacing marker, adjacent markers fuse, and total suppression is
  recorded (`markers_suppressed`) on the gc_collect event.
- **Deterministic:** marker content is a pure function of the dropped
  set; ids are sha256-derived from the dropped ids (never minted), so the
  same collection yields byte- and id-identical output — `collect()`
  stays pure.

**Observability:** gc_collect events carry `markers` (in-window count),
`marker_kinds` (tool_result/recall/user/assistant), `markers_coalesced`,
`markers_suppressed`. The behavioral harness surfaces the high-water
count as the `mkr` column so marker-driven recovery, re-derivation, and
fabrication are distinguishable. Guidance §2.4 (docs/GUIDANCE.md)
describes the marker format to the model — mechanism first, text second.
Online behavioral validation of the markers is t-1369.

---

## Invariants (apply to every strategy)

These hold for `ring`, `mark-sweep`, `stack`, and all future strategies. They
are the reason GC is correct rather than ad-hoc truncation.

**Hard guards: the system message and the last user message always survive
(t-1367).** Under any pressure, in every strategy and every degrade phase,
the system message and the *last user message* — the statement of the live
task — are never evicted, and tool-call pairs stay atomic through the same
mechanism (a pair group that would pull a protected message out is skipped
entirely). Semantic carried these guards from birth (t-1350); ring and stack
gained them after t-1364 proved their degrade paths could evict the live
task — the model answered "I'm ready to help!" and the loop accepted the
non-answer as final. Terminal case: if even the protected set exceeds the
budget, `collect()` returns an over-budget window and the overflow paths
(the t-1343 backstop, catch-overflow) own the outcome — the same stance
semantic always had. The offline matrix asserts zero last-user drops per
cell (`gc_evals.rs::evaluate`).

**Stateless + deterministic.** `collect(messages, budget, state)` is a pure
function. No wall-clock reads, no `HashMap` iteration order leaking into the
output ordering, no RNG. The same inputs produce the same output every run —
required so the strategies themselves can be eval'd reproducibly.

**Idempotent and convergent.** Running `collect()` on already-collected output
must not keep shrinking it below the protected floor, and a single pass must
provably get under budget. Define a hard floor: `pinned + last_n` messages must
fit. If they don't — i.e. a *single* message is itself larger than the budget —
no amount of dropping *around* it helps. So there is a mandatory pre-pass:

```rust
/// Runs before any strategy. Truncates the content of any single message that
/// alone exceeds the budget, so the strategy can then converge by dropping.
fn truncate_oversized_message(msgs: &mut [ChatMessage], budget: usize);
```

Without this, `ring` loops until `rest.is_empty()`, returns an over-budget
list, and you get `context_overflow` anyway.

**Conservative token estimation.** Every trigger and stop condition depends on
`estimate_tokens()`. A cheap `chars/4` heuristic is wrong by 20–30% on code and
JSON. The requirement is that the estimate be a *conservative upper bound*
(over-count). The worst case must be GC firing slightly early (mildly lossy),
never an overflow we failed to prevent. Cheap + conservative beats accurate +
optimistic.

**The cache-consuming pattern (async inputs without breaking statelessness).**
A strategy that wants expensive/async inputs — embeddings today, anything
similar tomorrow — must not compute them inside `collect()` (which is
synchronous, deterministic, and LLM-free by the first invariant). The
sanctioned shape, following the t-1166 design-note precedent, is a split:

- an **async pre-pass** in `interpreter::collect_prompt` (the layer with
  async + config access, where the t-1343 backstop also lives) computes the
  inputs and writes them into a `GcState` cache keyed by *message content
  hash* — it runs after `truncate_oversized_message` because truncation
  rewrites content, and it covers the scheduled, backstop, and overflow
  collection paths uniformly;
- `collect()` **consumes the cache read-only**. A missing entry — the
  pre-pass never ran (no config, replay), failed (endpoint outage), or the
  session resumed (GcState never serializes into checkpoints) — falls back
  to a deterministic heuristic, never an error and never a provider call.

Within a session the cache makes re-collections stable; across runs, the
same window plus the same cached values produce an identical collection
(asserted by the eval harness, which mirrors the pre-pass with a
deterministic mock). The pre-pass must prune entries whose content left
the window so the cache stays bounded by the live window.
`SemanticGc`/`GcState.embeddings` is the reference implementation.

---

## Prompt Caching (first-class, with a toggle)

GC and prompt caching interact, and the interaction can invert the cost
argument for the whole feature — so it gets its own section and an explicit
toggle (`--gc-cache preserve|ignore`).

**The problem.** Anthropic and OpenAI prefix caching keys on a *stable prefix*.
A naive ring buffer drops from the *front* (oldest messages), which changes the
prefix on every collection. That invalidates the cache, so you pay full input
cost on the entire window every GC turn — which can cost *more* than the
`context_overflow` you were avoiding.

**Two genuinely opposed user profiles — both supported permanently:**

| Profile | Wants | Use `--gc-cache` | Behavior |
|---|---|---|---|
| Performance-optimizer | Max context savings per turn; ignores caching | `ignore` | GC drops/summarizes wherever it's cheapest, including the front |
| Cache-optimizer | Stable cached prefix; minimize $ on cache hits | `preserve` (default) | GC keeps the cached prefix breakpoint stable; evicts from the *middle* |

This is not a temporary migration — neither profile is "correct." Some users
batch-process with no caching and care only about tokens; others run
long-lived sessions where cache hits dominate cost. We support both for good.

**`preserve` mechanics.** Keep the system prompt + cached prefix region fixed.
GC operates on the *interior* of the window (old-but-not-prefix messages),
preserving the cache breakpoint. Strategies advertise whether they can do this
via `ContextGc::cache_preserving()`. `ring` is *not* cache-preserving by
default (it drops from the front); under `--gc-cache preserve` it switches to a
middle-drop variant. `mark-sweep` and `stack` are naturally cache-preserving
because they evict by lifecycle/frame state, not strictly by position — but they
must still respect the prefix breakpoint and only invalidate at generational
boundaries if at all.

**Caching guide (ship this in user docs):**
- Default (`preserve`) is safe for everyone; it never makes caching worse.
- Switch to `ignore` only if you've confirmed you don't use prompt caching and
  you want every last token of context savings.
- `--gc-log` reports `cache_invalidated: bool` per collection so you can verify
  your chosen policy is actually doing what you expect.

---

## Trigger Policy

GC runs *before* each `provider.chat()` call, not on overflow:

```rust
const GC_THRESHOLD: f32 = 0.85;  // default; overridden by --gc-threshold

if estimate_tokens(&history) > (budget as f32 * gc_threshold) as usize {
    let before = history.clone();
    // mandatory pre-pass: nothing can be larger than budget on its own
    truncate_oversized_message(&mut history, budget);
    // stateless strategy; cross-turn metadata threaded via gc_state (&mut)
    history = gc.collect(history, budget, &mut gc_state);
    if gc_log {
        emit_gc_collect_event(&before, &history, gc.name(), gc.cache_preserving());
    }
}
```

This avoids the hard overflow and gives the response enough headroom — when
the estimate is honest. `--gc-timing catch-overflow` inverts the policy for
the case where it isn't: skip the proactive check, let the provider reject,
then collect and retry inside the same turn (see CLI Flags above).

Independent of the timing policy, `maybe_collect_prompt` applies the
collect-on-overflow backstop (t-1343): a prompt whose estimate exceeds the
full context budget is collected before dispatch no matter what the timing
would have decided, emitted as `gc_collect{reason: "backstop"}`. If
collection still cannot reach the budget, the existing overflow behavior
(catch-overflow retries, provider error) applies unchanged.

---

## Prior Art: Urbit Loom

The Urbit loom is worth recording as a design influence, with one caveat up
front: the loom is *not* a branching-graph model. It is strictly *linear and
nested* — "roads" are a stack of heap/stack scopes growing toward each other in
a flat 2GB arena. Urbit's graph-like character lives one layer up, in *nouns*
(immutable binary trees with structural sharing). Conflating the two layers is
a category error. With that separated, two ideas transfer:

**Road discipline → sharpen StackFrame's promotion polarity (v1-relevant).**
A road allocates on entry and is reclaimed wholesale on exit (an O(1) pointer
bump); anything you want to keep must be *explicitly copied out* to the outer
road. Applied to our StackFrame strategy, this flips the default: instead of
"keep everything in the frame, summarize under pressure," it becomes "discard
everything in the frame on exit unless explicitly promoted." That promotion
rule is a real design stance worth baking into v1's StackFrame — and it needs
none of the loom's arena or graph machinery, just the polarity.

**Nouns + structural sharing → DAG-backed GcState (deferred).** If the context
buffer and the long-term memory DB shared immutable subtrees (a tool result
referenced by three later messages is *one* node, not three copies), then
dropping a message is a refcount decrement, GC becomes reachability over a DAG
(the deferred `refcount` strategy), and dedup is free. The catch: **the LLM API
consumes a flat token sequence, not a graph.** Any sharing must be flattened
before every call, so it buys cheaper storage and cleaner `GcState`
bookkeeping but does *not* reduce the token budget — the thing GC exists to
manage. So this is a storage/bookkeeping win, not a new strategy.

**The integrated endgame (v3+, not a v1 commitment).** If `GcState` eventually
becomes a persistent content-addressed store shared between the live context
buffer and the memory DB, then "evicting from context" and "writing to memory"
become the *same operation*: you move the reachability root from the context
window to the memory index and the node stays put. GC and memory integrate.
Ben's read is that this convergence will happen — but not for a while; it's an
architectural bet to revisit after the v1 strategies and eval harness exist.

---

## Eval Harness

Every new strategy must be benchmarked before promotion to default. The harness
runs against a collection of real long-running task traces:

- `evals/gc/`: collection of `.jsonl` trace files (real agent sessions)
- Metrics: task completion rate, token reduction %, semantic coherence score
- Matrix axes: strategy x timing x cache policy x budget pressure. The
  timing axis mirrors `--gc-timing`: `final` (one collection on the full
  recorded window, what the first catch-overflow cycle sees) plus
  incremental `threshold`/`eager`/`every:4`, which replay the session
  growing message-by-message and fire at infer points with one `GcState`
  threaded across collections
- Run with: `cargo test --test gc_evals -- --nocapture`
- Recording new fixtures requires `--trace-full-payloads`: the harness reads
  full `InferCall` prompts, which are preview-only in traces by default
- The semantic-coherence score is an LLM-judge column (t-1168):
  online-gated behind `RUN_AGENT_ONLINE_EVAL=1`, with judge responses
  recorded to `evals/gc/judge/recorded.jsonl` and replayed by default so
  offline reruns stay deterministic and comparable (see
  `evals/gc/README.md`)

Gate any strategy change on showing improvement over `ring` on the existing
eval set. This also lets us tune `--gc-threshold` empirically.

---

## Implementation Order

1. `RingGc` + `--gc` flag + `--gc-threshold` + `--gc-log` wiring + **pair-atomic drop test first**
2. `MarkSweepGc` keyed on `ChatMessage.id` (UUID) + PromptIR section integration
3. `StackFrameGc` with heuristic summarization (no LLM in v1)
4. Eval harness before any new strategy is added after v1
5. `GenerationalGc` (future, after eval data exists)
6. `stack-smart` variant with LLM summarization (future, gated on eval improvement)

---

## Files to Touch

- `crates/agent-core/src/gc.rs` — new; trait + all strategy impls
- `crates/agent-core/src/lib.rs` — pub mod gc
- `crates/agent-core/src/interpreter.rs` — wire `ContextGc` into turn loop, emit `gc_collect` trace event
- `crates/agent/src/main.rs` — `--gc`, `--gc-threshold`, `--gc-log` CLI flags; construct `GcMode`; pass to runtime
- `crates/agent-core/src/prompt_ir.rs` — add `LifecycleState` to `Section`, and add a stable `id: Uuid` to `ChatMessage` (assigned on construction; breaks the schema, which is fine — no legacy persistence to preserve). `GcState.lifecycle` is keyed on this UUID.
- `crates/agent-core/src/gc.rs` — also defines `GcState` (owned by Runtime, threaded into `collect`) and `truncate_oversized_message` pre-pass.
- Runtime holds `gc: GcMode` + `gc_state: GcState`; strategies stay `&self`, only `gc_state` is `&mut`.
