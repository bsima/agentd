# Memory, retrieval, and the end of Get/Put (v2 design)

Status: **design pinned, approved for implementation** (t-1165). All
decisions below — including the formerly-open questions at the bottom —
were settled with Ben on 2026-06-12.

## Pinned decisions

1. **Memory is an implementation of hydration, not its own subsystem.**
   Reads already work this way: `MemorySource` (t-1160) implements
   `HydrationSource` and registers in the same `SourceRegistry` as every
   other source. Writes follow the same philosophy via a new trait, not a
   capability bit (below).
2. **Writable backends implement `HydrationSink`,** a separate trait — not
   `HydrationSource` with a `WRITE` capability. A "source you write to" is
   an oxymoron; the std::io precedent (`Read`/`Write` as small role-named
   traits, one type implementing both) is the right shape. This also keeps
   `SourceCapability` meaning exactly one thing — *how retrieval
   dispatches* (SESSION_CONTEXT / QUERY / WORKSPACE) — and makes
   writability a compile-time fact instead of a runtime probe against a
   default-erroring method.
3. **The IR effects are sink/source-generic, not memory-specific** (Ben,
   2026-06-12): `Retrieve` reads from any source, `Store` writes to any
   sink, both routed through the registry and recorded/replayed exactly
   like Infer/Eval. There is no `MemWrite` — memory has no special status
   in the IR vocabulary; it is one backend among several (chat history,
   self-prompt, ...).
4. **Get/Put is deleted as an agent-facing operation.** The null
   hypothesis won: every namespace either moves to the source/sink system
   or is absorbed into the IR machine. No KV verb survives.
5. **Reads and writes each have two channels, split by INITIATOR**
   (Ben's definition, 2026-06-12): **active = model-initiated through an
   LLM tool** (`remember`/`recall` compiled onto `Store`/`Retrieve` by
   the loop's tool-dispatch arm); **passive = runtime-initiated** — both
   program-sited IR effects (a `Store` the program author placed) and
   lifecycle hooks (sources at prompt assembly, shipped; sinks at turn
   completion, new). The 2x2 is the whole model; chat history and
   checkpointing stop being bespoke mechanisms and become a
   passively-written, actively-or-passively-read backend. Within
   passive, one mechanism distinction survives: program-sited effects
   run through the recorded-effect stream (effect ids, replayed), while
   lifecycle hooks sit outside it (no effect ids, suppressed under
   replay).

## The problem being solved

Get/Put crams three contracts into two verbs:

- `semantic:<query>` is a **search** masquerading as a key lookup — the
  "key" is a query, the result is ranked and time-varying, and there is no
  way to express k, filters, or get stable result ids back.
- `session:state` is **durable runtime state** — checkpointing, which the
  interpreter already owns; an agent program poking at its own checkpoint
  JSON is a layering violation.
- A future memory write is a **mutation** needing stable ids, metadata,
  provenance, and update-vs-create semantics — none of which a blind
  `Put(key, value)` can carry.

The magic-prefix convention also invites unbounded namespace creep (the
string key is an open-ended escape hatch around the effect system).

## Traits

```rust
/// Read side — unchanged from today.
#[async_trait]
pub trait HydrationSource: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind;
    fn capabilities(&self) -> SourceCapability; // retrieval dispatch only
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult>;
}

/// Write side — new. Backends that persist implement both traits.
/// The payload is sink-defined: `item` is a JSON value the sink validates
/// against its own schema, wrapped by the RUNTIME in an envelope carrying
/// provenance (run id, effect id, timestamp) — provenance is universal
/// and is not the program's job to supply.
#[async_trait]
pub trait HydrationSink: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind; // sinks share the source kind taxonomy
    async fn store(&self, item: SinkItem) -> Result<SinkId>;
    async fn update(&self, id: &SinkId, item: SinkItem) -> Result<()>;
    async fn delete(&self, id: &SinkId) -> Result<()>;
}

pub struct SinkItem {
    pub payload: serde_json::Value, // sink-validated schema
    pub provenance: Provenance,     // runtime-attached: run_id, effect_id, timestamp
}
```

`SinkId` is a stable identifier assigned by the sink (for the memory file
backend: the slug). `SourceRegistry` grows a `sinks` list alongside
`sources`; `register_backend` accepts a `T: HydrationSource +
HydrationSink` and pushes the same `Arc` into both lists (two unsized
coercions), so one backend object serves both halves.

The memory-specific shape (name / description / type / body) is the
**memory backend's payload schema**, not part of the trait or the IR: the
t-1160 file backend validates it, writes `<name>.md` with frontmatter
(provenance included), `update()` rewrites, `delete()` removes.
Last-writer-wins at file granularity is acceptable for v1; memory
directories are expected to live in git anyway. Other sinks define other
schemas — a chat-history sink takes a turn record, a self-prompt sink
takes a fenced text block.

## The four channels

|  | Source (read) | Sink (write) |
|---|---|---|
| **Passive** (runtime-initiated: program-sited effects + lifecycle hooks) | Prompt-assembly hydration (shipped, `PassiveHydrationConfig`) and program-sited `Retrieve` | Turn-completion persistence (new, `PassivePersistenceConfig`; absorbs checkpointing) and program-sited `Store` |
| **Active** (model-initiated via LLM tool) | `recall` tool -> `Retrieve` | `remember` tool -> `Store` |

Worked example — **chat history as a backend** (the motivating case): one
`ChatHistory` backend implements both traits. Its sink side is registered
passively: the runtime stores each completed turn, which is what
checkpointing is once it stops being bespoke. Its source side is readable
both passively (a recency window into every prompt — what t-1164's
`TemporalSource` already does over checkpoint files, i.e. that source is
this backend's read half avant la lettre) and actively
(`Retrieve { kind: Temporal, query: "what did we decide about X" }`). No
memory-specific anything is involved, which is the point: memory, chat
history, and the self-prompt experiment below are all just backends with
different payload schemas and different channel registrations.

Passive sink writes are runtime effects like trace emission — they are
not part of the program's effect stream, do not consume effect ids, and
are suppressed under replay (replay must never write; see open questions
for the exact semantics).

### The active channel: model-initiated tools (Ben, 2026-06-12)

The model decides mid-conversation — "make a note of this for later" —
which in this architecture means a **tool call**. The agent loop grows
`remember`/`recall` tools alongside `shell` and `infer`, and the loop's
existing tool-dispatch arm compiles them onto the same `Store`/`Retrieve`
effects, exactly as `shell` dispatches to `Eval` (the t-1108 machinery,
unchanged). Tool calls already get stable effect ids via the dispatch
site's dynamic path, so recording and replay work without anything new.

Two consequences:

1. Tools stay sink-specific and small — `remember { content, name?,
   type? }` targeting the memory sink — rather than exposing a generic
   `store { sink, op, ... }` surface to the model. The tool schema is the
   sink's payload schema; the loop supplies the sink selection.
2. Model-initiated writes are where the per-sink write policy earns its
   keep: "the model was talked into storing something" is the
   prompt-injection vector, and the policy hook sits in the dispatch arm,
   not in the model's prompt.

## IR effects

Two new instructions replace the Get/Put pair:

```text
Retrieve { out, query: Expr, kind: Option<SourceKind>, k, max_bytes }
    -> [{ id, name, score, source, content }]   (ranked)

Store { out, sink: Expr, op: Create | Update | Delete,
        id: Option<Expr>, item: Expr }
    -> id
```

`sink` selects the target by registered name (or kind, when unambiguous);
`item` is the sink-schema JSON payload. Both are effects in the full
t-1127 sense: stable effect ids (program-hash x site x dynamic path),
`RetrieveCall`/`RetrieveResult` and `StoreCall`/`StoreResult` trace events
(or the generic effect-event shape if we prefer fewer variants),
previews-not-payloads by default, and deterministic replay:

- A replayed `Retrieve` returns the recorded hits without touching any
  backend.
- A replayed `Store` returns the recorded id **without mutating the
  sink** — replay never writes. The recorded event carries a content hash
  so divergence (same site, different item) is detectable.

Writes are trace-visible by construction, which is the audit story:
`agent` traces show exactly what a session committed to which sink and
why (provenance carries the run id).

## What absorbs each Get/Put namespace

| Today | v2 home |
|---|---|
| `semantic:<query>` Get | `Retrieve` effect (kind = Semantic) |
| `session:state` Get/Put | A passive sink write at turn completion (the ChatHistory/session backend) — checkpointing stops being bespoke. The passive-hydration read of `SESSION_STATE_KEY` becomes that backend's passive source half. |
| `temporal:*` Get/Put | Machine env (the IR loop already keeps history in `history`); cross-session temporal recall is `Retrieve` (kind = Temporal) against the t-1164 backend |
| IR session-local KV (unknown keys) | Machine env vars — already per-session, already checkpointed with the machine |
| Op-runtime Get/Put ops | Deleted in the same change; the Op layer is a parity builder/test API and follows the IR |

Migration mechanics: `docs/STATE_KEYS.md` becomes the migration map and is
retired once the conformance test moves to effect-level assertions. There
are no external agent programs to break; the cut can be clean rather than
staged.

## Experiment: the system prompt as a backend (Ben, 2026-06-12)

If memory backends are just `HydrationSource + HydrationSink` pairs, the
system prompt can be one too: a `SelfPromptBackend` over the agent's spec
file, letting an agent rewrite its own standing instructions via the same
`Store` effect. Nothing new is needed mechanically — it is a backend
registered as both source and sink plus a prompt-assembly section — and it
composes cleanly with two existing designs:

- **t-1105 (spec file is the single source of truth):** "editing agent.md
  by hand is equally valid — same file, same effect" extends naturally to
  the agent itself being one more editor of the same file, with git
  history as the audit trail. No second writer problem, because there is
  still exactly one canonical artifact.
- **GC pinning:** system messages are never evicted and anchor the cache
  prefix, so a self-written instruction gets exactly the semantics
  "always remember this, in front of everything" — which is what
  distinguishes it from ordinary memory.

Guardrails the experiment needs from day one:

1. **A delimited agent-owned section, not free edit of the whole prompt.**
   The operator-authored core stays agent-immutable; the backend exposes
   only a fenced `<!-- self -->` block. Self-modification becomes additive
   self-notes, not constitution rewrites.
2. **Opt-in registration** (a flag / spec-file field), never default. The
   main risk is prompt-injection persistence: a hostile tool output that
   talks the model into writing itself a standing instruction survives
   into every future session, with system-level authority. This is the
   memory-poisoning attack with maximum blast radius, and it is why the
   open question on MemWrite approval policy below is load-bearing — the
   self sink is the case where requiring approval makes obvious sense.
3. **A size cap.** Every byte recurs in every future turn, and every edit
   invalidates the provider prompt cache (the system prompt IS the cache
   prefix) — unbounded growth is a cost bug as much as a safety one.

Not part of the v2 implementation; recorded here so the trait and effect
shapes do not preclude it (they do not — that is rather the point).

## Settled questions (all accepted by Ben, 2026-06-12)

1. **Write policy:** per-sink, not global. Free-but-trace-visible for the
   memory sink; approval-gated for the self-prompt sink.
   Supervisor-level review of sink diffs is an M2+ concern.
2. **Delete semantics:** hard delete; memory dirs live in git, history is
   the tombstone.
3. **Hit payloads:** `Retrieve` returns full bodies under `max_bytes` — a
   second round-trip costs a model turn.
4. **Event shape:** dedicated `RetrieveCall`/`RetrieveResult` and
   `StoreCall`/`StoreResult` trace variants, matching Infer/Eval.
5. **Passive sink semantics:** turn completion is the only lifecycle
   write point for v1; suppressed entirely under replay; passive write
   failures log-and-continue like trace-emission failures (a failing
   sink must not fail the turn).
6. **Tool exposure:** `remember`/`recall` appear in the tool list
   automatically whenever a memory backend is registered (an unreachable
   sink is a trap), and ship in the same change as the effects so the
   round-trip eval lands with the machinery.

## Acceptance (when implementation is approved)

- `Retrieve`/`Store` execute in the IR interpreter with stable effect
  ids, trace events, and recorded-replay parity (an eval fixture proves a
  replayed session never touches any sink).
- A `ChatHistory` backend registered as a passive sink + source replaces
  bespoke checkpoint writing, with t-1164's `TemporalSource` folded in as
  its read half.
- The loop's `remember` tool round-trips: model issues the tool call, the
  dispatch arm executes `Store`, a later `recall`/`Retrieve` finds it —
  recorded and replayed through the standard effect machinery.
- The file backend implements `HydrationSink`; `MemorySource` registered
  via `register_backend` serves both halves.
- Get/Put instructions, `dispatch_get`/`dispatch_put`, and the
  `state_keys.rs` conformance test are removed in the same stack, with
  STATE_KEYS.md replaced by this document's migration table.
- `evals/` gains a memory round-trip eval: create -> retrieve -> update ->
  retrieve -> delete, deterministic offline.
