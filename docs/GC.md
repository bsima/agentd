# agentd Context GC Design

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
agent --gc <strategy>          # default: ring
agent --gc-threshold <0.0-1.0> # trigger GC at this fraction of budget (default: 0.85)
agent --gc none                # disable GC entirely (hard overflow = error)
agent --gc-log                 # emit gc_collect trace events (for debugging)
agent --gc-cache <mode>        # prefix-cache policy: preserve | ignore (default: preserve)
```

`--gc none` restores the current behavior and is important for deterministic
evals that must not have GC noise in their context.

`--gc-threshold` is tunable because different workloads have different optimal
trigger points: long tool chains benefit from earlier GC, short chat sessions
don't need it at all. Make it tunable from day one.

`--gc-log` emits structured trace events on every collection:
```json
{"type": "gc_collect", "strategy": "ring", "tokens_before": 90000, "tokens_after": 60000, "frames_popped": 12}
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

Strategies (planned, not all implemented at once):

| Name           | Description                                      | Status     |
|----------------|--------------------------------------------------|------------|
| `none`         | No GC; overflow = hard error (current behavior) | existing   |
| `ring`         | Drop oldest messages when buffer fills           | v1 default |
| `mark-sweep`   | Evict "dead" sections by type annotation         | v1         |
| `stack`        | Pop completed tool-call frames to summaries      | v1         |
| `generational` | Hot/warm/cold compaction (JVM-style)             | future     |
| `refcount`     | Dependency-graph reachability eviction           | future     |

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
until under budget. Never drop the system prompt.

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
  `evictable` after being incorporated — replaced with a one-line annotation:
  `[tool: read_file /path — result incorporated]`
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

**Summary format:** `[frame: <tool_name>(<summary_of_args>) → <summary_of_result>]`

**Summarization:** Use pure heuristics for v1 — truncate tool results to first
N chars + "…[truncated]". Do NOT use an LLM summarization call for v1: it adds
latency and tokens (ironic). Reserve the LLM-assisted `stack-smart` variant
for after the eval harness exists to justify the tradeoff.

**Key insight:** This is the most space-efficient strategy for tool-heavy
agents. A deep chain of `bash` + `read_file` turns can compress from 20k tokens
to ~200 tokens of annotations without losing the semantic record.

---

## Invariants (apply to every strategy)

These hold for `ring`, `mark-sweep`, `stack`, and all future strategies. They
are the reason GC is correct rather than ad-hoc truncation.

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

This avoids the hard overflow and gives the response enough headroom.

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
- Run with: `cargo test --test gc_evals -- --nocapture`

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
