# Public Trace Event Schema

`schema_version: 1`

This document is the contract for consumers of agentd trace data: the SDK
trace adapter, dashboard ingest, and any external tooling. It defines a
**versioned public view** over the runtime trace, implemented by
`agent_core::public_trace::public_event(&Event) -> Option<PublicEvent>`
(crates/agent-core/src/public_trace.rs).

## Two layers, one file — plus a separate stdout channel

There are three event surfaces around an agent run. Do not conflate them:

1. **Runtime trace events** (`Event` in crates/agent-core/src/trace.rs) —
   the JSONL trace file (`--trace`). Serialized with PascalCase `"event"`
   tags (`InferCall`, `EvalResult`, ...). These are the *replay identity*:
   effect-id replay, divergence detection, and every existing trace file
   depend on their exact shape. **They are NOT public API.** Their names and
   fields may change with the runtime (guarded by replay compatibility, not
   by this document).
2. **Public trace events** (`PublicEvent`, this document) — a pure,
   versioned projection of runtime events into dotted lifecycle names
   (`infer.started`, `run.completed`, ...). This is what observability
   consumers program against. Not every runtime event has a public
   projection, and some public names are reserved before any runtime event
   emits them.
3. **Machine events** (stdout, supervisor-facing) — when the `agent` binary
   runs with `--json` (alias of `--debug`), it writes lines of the form
   `{"type":"custom","custom_type":"agent_start"|"agent_complete"|"agent_error",
   "data":{...},"timestamp":...}` to stdout (interleaved with a mirror of the
   trace JSONL). These are the supervisor's control channel: `data.turn_id`
   (t-1308.2) correlates a session send to its completion, and
   `agent_complete` echoes the envelope's opaque `metadata`. Machine events
   are **not** trace events, are not written to the trace file, and are not
   covered by this schema. Do not merge the two streams: trace events are
   observability-facing, machine events are supervision-facing (see
   docs/SUPERVISOR.md).

### Why a view instead of renaming the runtime events

The runtime's PascalCase variant names are load-bearing for replay: renaming
them would orphan every recorded trace and break `--replay`. The public
vocabulary wants dotted lifecycle names. The resolution is a mapping layer:
the runtime keeps its serialization untouched, and `public_event` converts
each runtime event to its public shape (or `None` for private events). The
projection is a pure function of a single event, so any consumer can apply
it to a trace file, a live tail, or an ingest stream and get identical
results.

## Compatibility policy

- The schema is versioned. Every public event line carries
  `"schema_version": 1`.
- **Additive** changes — new event names, new optional fields, new `attrs`
  keys — bump the schema's *minor* version (documented here; the wire
  `schema_version` integer is the major and does not change).
- **Renames or removals** of event names or fields bump the *major* version
  (the wire `schema_version` integer changes).
- Reserved names/fields (listed below) may start being emitted at any time
  without a major bump — consumers must tolerate unknown event names and
  unknown fields.
- Runtime PascalCase variant names, and any field of the runtime `Event`
  enum, are **not public API**. Consumers must not parse the runtime trace
  directly except to apply `public_event` (or to feed replay tooling, which
  owns that format).

## Envelope fields

Every public event is one JSON object per line (JSONL), with fields in this
order (the golden conformance test pins the ordering):

| field | type | presence | meaning |
|---|---|---|---|
| `schema_version` | integer | always | `1` |
| `event` | string | always | dotted lifecycle name (catalog below) |
| `ts` | string (RFC 3339 UTC) | always | runtime event timestamp |
| `run_id` | string | always | the run this event belongs to |
| `session_id` | string | always | currently **equal to `run_id`**; the runtime does not yet distinguish sessions from runs. Separate field so ingest schemas won't migrate when it does. |
| `turn_id` | string | **reserved** (never emitted in v1) | turn correlation id. Today turn ids exist only on stdout machine events (t-1308.2); once the runtime threads them into trace events this field is populated. |
| `op_id` | integer | when the runtime event has one | runtime operation id; pairs `*.started` with its `*.completed`/`*.failed`, and addresses the full payload in the runtime trace |
| `parent_op_id` | integer | optional | runtime op lineage (e.g. work inside a Par branch) |
| `status` | string | always | `started` \| `completed` \| `failed` |
| `effect` | object | optional | stable IR effect identity, see below |
| `parent_effect_id` | string | **reserved** (never emitted in v1) | effect-level lineage. The runtime tracks op-level lineage (`parent_op_id`), not effect-level; emitted once it does. |
| `error` | string | iff `status == "failed"` | terminal error message |
| `payload_preview` | string | optional | truncated human-oriented excerpt (see payload rules) |
| `payload_ref` | string | **reserved** (never emitted in v1) | opaque reference to an out-of-band full payload, once a payload store exists |
| `attrs` | object | omitted when empty | event-specific attributes (per-event tables below); keys are additive-only within a major version |

### Effect identity (`effect`)

Present on `*.started` events of IR-interpreted runs (absent for op-layer
traces, and absent on `completed`/`failed` events — correlate those via
`op_id`). Projection of the runtime `EffectLocation` (t-1057/t-1058):

```json
"effect": {
  "effect_id": "sha256:...",          // stable id: hash of program_hash + site + dynamic_path
  "program_hash": "sha256:...",       // hash of the canonical IR program
  "site": { "block": 0, "instruction_index": 1 },
  "dynamic_path": {
    "path": "",                        // rolling control-flow digest ("" = entry block)
    "transitions": 0,                  // transitions folded into path
    "visit": 0                         // per-site execution ordinal (0-based)
  }
}
```

`effect_id` is the stable cross-run identity of "this effect at this site
along this control path" — the key for diffing runs of the same program.
See docs/AGENT_IR.md "Effect identity".

### Status semantics

- `started` — the effect was dispatched (`*Call` runtime events).
- `completed` — the effect ran to completion (`*Result`). Note: an Eval
  whose command exited nonzero is still `completed` — the process ran; its
  outcome is in `attrs.ok` / `attrs.exit_code`. This mirrors runtime
  semantics (an `EvalResult` with `ok: false`).
- `failed` — terminal effect failure (`*Error`: provider error after
  retries, spawn failure, sink/policy error, replay divergence). `error`
  carries the message.

### Payload rules: previews vs full payloads

Public events carry **previews only** (`payload_preview`, truncated to
~1024 chars with a `...` suffix). Full payloads follow the runtime trace's
conventions and are addressed by (`run_id`, `op_id`) in the runtime trace
file:

- **Infer**: prompt/response previews always; full prompt only when the
  runtime records it (`trace_full_payloads`, off by default — full prompts
  make traces O(n²) in session length). Full responses are recorded in the
  runtime trace (replay identity).
- **Eval**: `payload_preview` on `eval.completed` is a preview of stdout.
  Full stdout/stderr (capped by the eval byte limits, with `truncated_*`
  flags) live in the runtime `EvalResult.result`. For argv Evals the exact
  argv is the replay identity and is carried in full in both layers
  (`attrs.argv`).
- **Retrieve**: results are recorded **in full** in the runtime trace
  (replay returns them verbatim); the public event carries only the
  preview plus `bytes`/`source_count`.
- **Store**: the runtime records a **preview + content hash** of the item
  (never the full payload); the public event mirrors that
  (`payload_preview`, `attrs.content_hash`). The sink-assigned id is on
  `store.completed` (`attrs.sink_id`).

`payload_ref` is reserved for a future out-of-band payload store; until
then, consumers needing full payloads read the runtime trace line with the
same `run_id`/`op_id`.

## Event catalog (schema 1)

### Emitted events

| public event | mapped from runtime variant | status | `payload_preview` | `attrs` keys |
|---|---|---|---|---|
| `infer.started` | `InferCall` | `started` | prompt preview | `model` |
| `infer.completed` | `InferResult` | `completed` | response preview | `duration_ms`, `input_tokens`, `output_tokens`, `total_tokens` |
| `infer.failed` | `InferError` | `failed` | — | `duration_ms` |
| `eval.started` | `EvalCall` | `started` | display command | `argv`? (string[], direct-exec only), `cwd`?, `env_policy`, `timeout_ms` |
| `eval.completed` | `EvalResult` | `completed` | stdout preview | `command`, `duration_ms`, `ok`?, `exit_code`?, `timed_out`?, `truncated_stdout`, `truncated_stderr` |
| `eval.failed` | `EvalError` | `failed` | — | `command`, `duration_ms` |
| `retrieve.started` | `RetrieveCall` | `started` | query | `kind`?, `max_bytes`? |
| `retrieve.completed` | `RetrieveResult` | `completed` | result preview | `bytes`, `duration_ms`, `source_count` |
| `retrieve.failed` | `RetrieveError` | `failed` | — | `duration_ms` |
| `store.started` | `StoreCall` | `started` | item preview | `content_hash`, `sink`, `store_id`?, `store_op` |
| `store.completed` | `StoreResult` | `completed` | — | `duration_ms`, `sink`, `sink_id` |
| `store.failed` | `StoreError` | `failed` | — | `duration_ms`, `sink` |
| `run.completed` | `AgentDone` | `completed` | — | — |

(`?` = present only when the runtime recorded a value.)

### Reserved event names (documented, not yet emitted)

Consumers must accept these without a schema bump when they start flowing:

- `run.started` — no runtime trace event marks run start today (the
  stdout `agent_start` machine event is supervision-facing, not trace).
- `turn.started`, `turn.completed` — turns exist in the session protocol
  (machine events carry `turn_id`), but the runtime trace does not yet
  record turn boundaries.
- `tool.requested`, `tool.completed`, `tool.failed` — model-initiated tool
  dispatch as a first-class public event (today tools surface as the
  `eval.*` / `retrieve.*` / `store.*` effects they compile to).
- `approval.requested`, `approval.resolved` — human-in-the-loop approval
  gates.
- `output.validation_failed` — structured-output validation failures.

### Private runtime events (no public projection)

These project to `None` — they are runtime internals, reachable only via
the runtime trace, and may change without notice:

| runtime variant / custom name | why private |
|---|---|
| `HydrationStart` / `HydrationSection` / `HydrationEnd` | context-assembly internals; candidate for a future additive projection |
| `ParStart` / `ParEnd` | execution-structure internals (lineage already public via `parent_op_id`) |
| `Checkpoint` | persistence internals |
| `TurnBudgetExhausted` | budget taxonomy internals (t-1133); candidate for a future additive projection |
| `Custom` (all names: `gc_collect`, `gc_truncate`, `context_overflow`, `prompt_ir`, domain tags, ...) | extension/diagnostics channel, unbounded vocabulary — never public |

## Correlation semantics

- **run_id** — one agent process run. Every event carries it.
- **session_id** — reserved distinction; **equals `run_id`** in v1. A
  session (persisted conversation identity across process restarts, see
  docs/MEMORY.md) may later span multiple runs; when the runtime records
  that, `session_id` diverges from `run_id` (additive change).
- **turn_id** — one send/response exchange within a session. Today it is
  minted (or echoed from the caller's envelope) per turn and carried only on
  stdout machine events (`agent_start`/`agent_complete`/`agent_error`,
  t-1308.2). The public `turn_id` field is reserved until the runtime
  threads it into trace events.
- **op_id / parent_op_id** — pair `started` with `completed`/`failed`
  (same `op_id`, unique within a run) and express structural lineage.
- **effect.effect_id** — stable *cross-run* identity of an IR effect site
  + control path; use it to align the same logical step across runs or
  between a recording and a replay. `started`-only; join to the close via
  `op_id`.

## Conformance

- Golden test:
  `agent_core::public_trace::tests::golden_public_projection_of_an_ir_run`
  drives an IR program (Infer + Eval) with a scripted provider, projects the
  trace, and compares byte-for-byte (including field order) against
  `crates/agent-core/testdata/public_trace_golden.jsonl`. Timestamps and
  measured durations are the only normalized fields.
- Decision test:
  `every_variant_projects_or_is_explicitly_private` lists every runtime
  variant with its projection decision; the `public_event` match is
  exhaustive, so adding a runtime variant does not compile until a
  public/private decision is made in both places.

## Relation to the OTel sink

The OTel sink (docs/OTEL.md) is a separate consumer of runtime events with
its own naming (span-per-effect, `gen_ai.*` semconv attributes). It predates
this schema and is unaffected by it; the public event schema is the JSON
contract, OTel is the tracing-backend mapping. They may converge later, but
neither constrains the other today.
