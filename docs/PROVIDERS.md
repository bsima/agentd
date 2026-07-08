# Authoring a hydration provider

How to write, register, and test a custom memory/retrieval backend. The
design rationale lives in `docs/MEMORY.md` (t-1165, pinned); this is the
implementer's contract. A complete, runnable out-of-tree provider lives at
`crates/agent-sdk/examples/custom_source.rs`, and
`crates/agent-core/tests/provider_registry.rs` pins that the in-tree
providers register through exactly the API documented here.

## The surface at a glance

A provider is one or two small traits from `agent_core::hydration`
(re-exported at the crate root):

- **`HydrationSource`** — the read half: answer retrieval requests.
- **`HydrationSink`** — the write half: accept `store`/`update`/`delete`.

One type may implement both (std::io `Read`/`Write` precedent);
writability is a compile-time fact, not a capability probe. The in-tree
providers, all registered through the same public API available to
out-of-tree code:

| Provider | name | kind | halves | capabilities | write policy |
|---|---|---|---|---|---|
| `MemorySource` (markdown memory dir + index + optional embeddings) | `memory` | Semantic | source + sink | QUERY | Free |
| `ChatHistory` (checkpoints as sink, recency window as source) | `chat-history` | Temporal | source + sink | SESSION_CONTEXT, QUERY | Free |
| `TemporalSource` (archived checkpoint dirs) | `temporal-checkpoints` | Temporal | source only | SESSION_CONTEXT, QUERY | — |

(The `agent` CLI additionally defines a private `local-files` source for
`--hydration-dir` — itself proof the trait is implementable outside
`agent-core`.)

## The source contract

Verbatim from `crates/agent-core/src/hydration.rs`:

```rust
#[async_trait]
pub trait HydrationSource: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind;
    fn capabilities(&self) -> SourceCapability;
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult>;
}

pub struct SourceParams {
    pub query: Option<String>,
    pub max_bytes: Option<usize>,
}

pub struct SourceResult {
    pub source: String,   // your name(), for provenance
    pub kind: SourceKind, // your kind()
    pub content: String,  // model-facing text
    pub metadata: Value,  // provider-defined
}
```

What each piece means:

- **`name`** — a stable, human-readable id. It lands in
  `SourceResult.source`, trace events, and PromptIR provenance, and it is
  how `Store` selects sinks. The registry does not enforce uniqueness;
  sink lookup returns the *first* match, so pick a name that does not
  collide with the table above (in particular, `memory` is the name the
  built-in `remember` tool targets).
- **`kind`** — where you sit in the retrieval taxonomy. `Semantic` is
  what the built-in `recall` tool queries (`Retrieve { kind: Semantic }`);
  `Temporal` is cross-session history; `Knowledge` is reference material.
  A `Retrieve` with `kind: None` consults every QUERY-capable source
  regardless of kind.
- **`capabilities`** — *retrieval dispatch only* (never writability):
  - `QUERY`: consulted by the `Retrieve` effect (and therefore by the
    model's `recall` tool). `params.query` is `Some`.
  - `SESSION_CONTEXT`: consulted at passive prompt-assembly hydration.
    `params.query` is `None` — return your standing context.
  - `WORKSPACE`: declared in the flag set but **no dispatch path consults
    it today**; treat it as reserved.
- **`retrieve`** — return exactly **one** `SourceResult` per call,
  aggregating your matches into `content` (the memory backend renders a
  ranked list; the example renders `file:line: text` hits). Respect
  `params.max_bytes` when present — the runtime does not truncate for
  you. An *empty* result set is a normal outcome (return a short "no
  matches" content or an empty string), not an error.

What the runtime guarantees a source author:

- **Determinism is not required.** `Retrieve` is a recorded effect: the
  results you return are written to the trace (`RetrieveCall` /
  `RetrieveResult` events), and a replayed run serves the recording — your
  `retrieve()` is never called under replay. Network-backed and
  time-varying sources are fine. (Backend-internal calls — e.g. the memory
  backend's embedding HTTP requests — are not effects, consume no effect
  ids, and are never replayed; only your *results* are recorded.)
- **Provenance is attached for you** on the passive path: each result
  becomes a PromptIR section with `SectionOrigin::Retrieval { backend:
  your name, mode, query, score }`. If your `metadata` carries a numeric
  `score` field, it is surfaced there.

What a source author must guarantee the runtime:

- **Errors abort the whole dispatch**, not just your source: the registry
  fails the `retrieve_*` call on the first source error. On the active
  path the `recall` tool binds that error as a value the model can read
  (errors-as-values, t-1222) and the run continues; a program-sited
  `Retrieve` with the default policy aborts the turn; on the passive path
  it fails prompt assembly. So degrade instead of erroring wherever
  possible — the memory backend's convention is that an embedding outage
  falls back to keyword ranking and *never* fails a retrieve. Reserve
  `Err` for genuine faults (unreadable root, corrupt store).
- Reasonable latency: sources run sequentially in registration order.

## The sink contract

```rust
#[async_trait]
pub trait HydrationSink: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind; // sinks share the source kind taxonomy
    fn write_policy(&self) -> SinkWritePolicy {
        SinkWritePolicy::Free
    }
    async fn store(&self, item: SinkItem) -> Result<SinkId>;
    async fn update(&self, id: &SinkId, item: SinkItem) -> Result<()>;
    async fn delete(&self, id: &SinkId) -> Result<()>;
}

pub struct SinkItem {
    pub payload: Value,         // sink-defined schema — validate it yourself
    pub provenance: Provenance, // runtime-attached: run_id, effect_id, timestamp
}

pub enum SinkWritePolicy {
    Free,            // writes execute immediately; the trace is the audit
    RequireApproval, // writes pause at the approval gate
}
```

- **The payload schema is yours.** The trait and the IR carry opaque
  JSON; *you* validate it and reject what you do not understand (the
  memory sink's schema is `{ name?, description?, type?, body }`; the
  chat-history sink takes a checkpoint record). A validation failure is a
  legitimate `Err` — the `remember` tool binds it as a readable tool
  result (errors-as-values), and a program-sited `Store` follows its
  policy.
- **`SinkId`** is a stable identifier you assign at `store` and interpret
  at `update`/`delete` (the memory sink uses the file slug). It is opaque
  to the runtime and is what the `Store` effect returns to the program.
- **Provenance is runtime-attached, never program-supplied.** Every write
  arrives wrapped with the run id, effect id, and timestamp. Persist it if
  your schema allows (the memory sink writes it into frontmatter); the
  chat-history sink deliberately drops it to keep a fixed checkpoint
  schema — both are valid.
- **`write_policy` is the per-sink prompt-injection defense** ("the model
  was talked into storing something" is the attack; the policy hook sits
  in effect dispatch, not in the model's prompt). `Free` writes execute
  immediately and are audited by `StoreCall`/`StoreResult` trace events.
  `RequireApproval` makes every `Store` targeting your sink pause at the
  approval gate exactly like a gated shell command: in the SDK the
  agent's `on_approval` hook decides at the effect site (no hook = fail
  closed, the run errors with `ApprovalRequired`); in the CLI the run
  pauses durably and resolves via `agent approvals`. A denial binds a
  typed denial value the model can react to; approvals and denials are
  traced (`approval.requested`/`approval.resolved`) and reproduced as
  data on replay — replay never consults the live policy.
- **Replay never writes.** A replayed `Store` returns the recorded
  `SinkId` without calling `store`/`update`/`delete`; the recorded event
  carries a payload content-hash so divergence is detected. Consequences
  for a sink author: never rely on being called during replay, and keep
  writes idempotent-ish at the granularity of your ids, because a *fresh*
  (non-replay) re-run of the same program will call you again with the
  same inputs. Passive lifecycle writes (turn-completion persistence) are
  suppressed entirely under replay and their failures log-and-continue —
  a failing sink must not fail the turn.

## Registration

`SourceRegistry` (the `hydration` field of `SeqConfig`) is the single
registration point; all methods are builder-style:

```rust
let registry = SourceRegistry::new()
    .register(source)              // read-only source
    .register_arc(arc_source)      // same, from an existing Arc
    .register_sink(sink)           // write-only sink
    .register_backend(backend);    // T: HydrationSource + HydrationSink —
                                   // ONE object, one Arc, coerced into
                                   // both lists (both halves share state)
```

Use `register_backend` for a persisting backend rather than registering
two separate objects, so the source half reads what the sink half wrote
through the same state (index caches, locks).

## How results reach the model

**Active path** (model-initiated): the model calls `recall { query }`;
the loop's tool-dispatch arm compiles it onto `Retrieve { kind: Semantic,
max_bytes: 16KiB }`; the registry fans out to every QUERY-capable
Semantic source; the JSON array of `SourceResult`s is recorded in the
`RetrieveResult` trace event and rendered into the tool message the model
reads next turn. Program-sited `Retrieve` instructions are the same
effect with author-chosen kind/query/policy.

**Passive path** (runtime-initiated, prompt assembly): when
`PassiveHydrationConfig` includes `SessionContext`, the runtime calls
every SESSION_CONTEXT-capable source with `query: None` before each model
call; each result is traced as a `HydrationSection` event and becomes a
PromptIR context section carrying retrieval provenance (backend name,
mode, optional `metadata.score`, `RetrievalTiming::Passive`) — this is
what GC and cache planning see. The CLI enables this; in-process SDK runs
do not (see wiring).

**Writes**: the model's `remember { content, name?, type? }` compiles
onto `Store { sink: "memory", op: Create }` — the tool schema *is* the
memory sink's payload schema, and the sink selection is the loop's, not
the model's. Program-sited `Store` instructions select any registered
sink by name and carry Create/Update/Delete.

## Decision table

| I want... | Implement / do |
|---|---|
| Read-only retrieval the model can `recall` | `HydrationSource`, kind `Semantic`, capability `QUERY` |
| Standing context injected into every prompt | `HydrationSource` with `SESSION_CONTEXT` (reachable in the CLI's passive hydration; inert in in-process SDK runs) |
| Cross-session recency ("what did we decide about X") | `HydrationSource`, kind `Temporal`, `QUERY` (+ `SESSION_CONTEXT` for passive injection) |
| Persistence the model writes via `remember` | `HydrationSink` named `memory` (the built-in loop targets that name), registered with `register_backend` so reads see the writes |
| A store only my program writes (program-sited `Store`) | `HydrationSink` under your own name; drive it with `Store { sink: "your-name", ... }` |
| Writes gated behind a human | `write_policy() -> SinkWritePolicy::RequireApproval` |
| Read *and* write over shared state | One type implementing both traits + `register_backend` |
| Semantic ranking instead of substring/keyword | Backend-internal: embed with the shipped `Embedder`/`EmbeddingIndex`/`cosine` infra and blend scores like `MemorySource` (t-1340) — the trait surface does not change |

## Wiring it in

**agent-core** (embedding the loop directly): build the registry and set
`SeqConfig.hydration`; set `AgentLoopOptions.memory_tools` if the
`remember`/`recall` tools should be advertised.

**agent-sdk**: `AgentBuilder::hydration_source(MySource::new(...))`
registers a custom read-only source alongside the built-ins. A
`Semantic + QUERY` source automatically exposes the `recall` tool, the
same rule `memory_dir` follows (MEMORY.md settled question 6: an
unreachable backend is a trap). Custom *sinks* are deliberately not
registrable through the SDK builder: the built-in loop's only write path
is `remember`, which targets the sink named `memory` — a custom sink
would be unreachable dead weight. Full backends require driving
`agent_core::run_agent_loop` yourself. The example round-trip:

```sh
cargo run -p agent-sdk --example custom_source
```

**agent CLI**: **custom providers are not reachable from the CLI without
code changes.** The CLI registers exactly its built-ins, keyed on flags:
`--hydration-dir` (the private local-files source), `--memory-dir` (the
memory backend + `remember`/`recall`), `--temporal-dir`
(`TemporalSource`). There is no plugin/config mechanism for loading an
out-of-tree `HydrationSource` into the `agent` binary, and none is
planned until a concrete need appears — Rust has no stable ABI for
dynamically loading trait objects, so the honest extension points are (a)
embed the loop via agent-sdk/agent-core as above, or (b) add a
registration site next to the existing ones in `crates/agent/src/main.rs`
(both `run` and `resume_run` build the registry) and ship it as a new
built-in flag.

## Testing a provider

The example's pattern, credential-free and offline:

1. Unit-test `retrieve` directly against a fixture directory/store.
2. Round-trip through the loop with a `ScriptedProvider` that issues the
   `recall` tool call, then assert the `RetrieveResult` trace event (or
   the public `retrieve.completed` event) carries your content — that is
   the exact value the model was shown. See
   `crates/agent-sdk/tests/custom_source_example.rs`, which runs the
   example's own code in CI.
3. If you implement a sink, cover the store → retrieve → update →
   retrieve → delete cycle (the memory backend's eval does this) and, for
   `RequireApproval`, both the approve and deny arms.
