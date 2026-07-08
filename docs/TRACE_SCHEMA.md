# Public Trace Event Schema

`schema_version: 1` (wire major; this document tracks minor revisions — current: **1.4**, see the version history at the end)

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
- **Tool** (native tools, since 1.2): the runtime records the model-supplied
  arguments and the handler result **in full** (they are the replay
  identity: replay checks the arguments and returns the result verbatim
  without invoking the handler); the public events carry previews only
  (arguments preview on `tool.requested`, result preview on
  `tool.completed`).

`payload_ref` is reserved for a future out-of-band payload store; until
then, consumers needing full payloads read the runtime trace line with the
same `run_id`/`op_id`.

## Event catalog (schema 1)

### Emitted events

| public event | mapped from runtime variant | status | `payload_preview` | `attrs` keys |
|---|---|---|---|---|
| `infer.started` | `InferCall` | `started` | prompt preview | `model` |
| `infer.completed` | `InferResult` | `completed` | response preview | `duration_ms`, `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`?, `cost_micro_usd`?, `pricing`? |
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
| `tool.requested` | `ToolCall` | `started` | arguments preview | `name` |
| `tool.completed` | `ToolResult` | `completed` | result preview | `duration_ms`, `name` |
| `tool.failed` | `ToolError` | `failed` | — | `duration_ms`, `name` |
| `approval.requested` | `ApprovalRequested` | `started` | gated request payload | `kind`, `pending_id` |
| `approval.resolved` | `ApprovalResolved` | `completed` | — | `decision`, `effect_id`, `kind`, `pending_id`, `reason`?, `resolved_by`? |
| `run.completed` | `AgentDone` | `completed` | — | `infer_calls`?, `input_tokens`?, `output_tokens`?, `total_tokens`?, `cached_input_tokens`?, `cost_micro_usd`?, `uncosted_infer_calls`?, `failed_infer_calls`? |
| `output.validation_failed` | `Custom { name: "output_validation_failed" }` | `failed` | invalid final-output excerpt | `attempt` (1-based), `errors` (string[], capped at 8) |

(`?` = present only when the runtime recorded a value.)

`tool.requested` / `tool.completed` / `tool.failed` (since 1.2, t-1308.7):
model-initiated dispatch of a **registered native tool** — an in-process
async handler registered with the runtime (`agent_core::tool::ToolRegistry`,
surfaced by the SDK). These are the previously-reserved names, now emitted
by the runtime `ToolCall`/`ToolResult`/`ToolError` events. The built-in
tools are unchanged: shell/infer/remember/recall still surface as the
`eval.*` / `infer.*` / `store.*` / `retrieve.*` effects they compile to.
`tool.requested` carries the stable IR effect identity (`effect`, kind
`Tool`); `attrs.name` on all three is the registered tool name. Handler
failures ride the loop's errors-as-values convention, so a `tool.failed` is
usually followed by the model reacting to the error, not by run failure.

`approval.requested` / `approval.resolved` (since 1.4, t-1308.10):
human-in-the-loop approval gates over effects (DR-7 — Store to a
`RequireApproval` sink, or an Eval whose `EvalPolicy.require_approval` is
set). `approval.requested` fires when a gated effect reaches the gate with
no decision yet: the run either pauses durably (a pending record + mid-turn
machine checkpoint under `~/.local/share/agent/approvals`, resolved by
`agent approvals --approve/--deny`) or asks an in-process hook (the SDK's
`on_approval`). `approval.resolved` fires when the decision is consumed at
the effect site — possibly in a later process, appended to the same trace.
Neither carries an `op_id`: the gate sits ahead of effect dispatch (a
paused or denied effect never becomes an operation). Correlate the pair via
`attrs.pending_id`, and the resolution to its effect via `attrs.effect_id`
(the `requested` side carries the full `effect` identity). A denial is
`completed`, not `failed`: it resolves to a typed denial value the program
and model react to (errors-as-values); the run continues. Approval outcomes
are recorded results — replay reproduces the pause or the decision as data
(re-emitting both events) without pausing or prompting, and diverges when
the recorded request payload does not match the observed effect.

Cost accounting attrs (since 1.3, t-1334): all cost arithmetic is exact
integer math — see `agent_core::cost`.

- `infer.completed`: `cached_input_tokens` is the provider-reported cached
  prompt-token count (OpenAI-compatible `prompt_tokens_details.cached_tokens`,
  Anthropic `cache_read_input_tokens`, codex Responses
  `input_tokens_details.cached_tokens`), present only when the provider
  reported it. `cost_micro_usd` is this call's cost in **integer micro-USD**
  (`input_tokens` and `output_tokens` at the model's registry rates, rounded
  half-up once per call); `pricing` is the rate snapshot used, an object
  `{ "input_micro_usd_per_mtok": u64, "output_micro_usd_per_mtok": u64 }`
  (micro-USD per million tokens, so `$3.00/Mtok` is `3000000`). Both are
  present exactly when the model had pricing configured in models.yaml at
  record time — **absent means unknown pricing, never zero** — and are part
  of the recorded payload: replayed runs re-emit the recorded values
  verbatim rather than repricing.
- `run.completed`: rollup of every InferResult recorded in the run —
  integer sums of the per-event values (floats are never accumulated).
  `uncosted_infer_calls` counts infers recorded without cost, so a partial
  `cost_micro_usd` total is visibly partial. `failed_infer_calls` (since
  1.5, t-1347) counts Infer attempts that ended in `InferError` — a count
  only, present only when nonzero: the provider error path returns no
  Response, so a failed attempt's token usage is structurally unavailable
  and contributes nothing to the sums. `cached_input_tokens` /
  `cost_micro_usd` are absent when no event in the run carried them; the
  whole attr group is absent for infer-less runs and traces recorded before
  1.3. `agent cost --trace <file.jsonl> [--json]` prints the same rollup
  (plus per-model breakdowns) from a trace file.

`output.validation_failed` (since 1.1, t-1308.4): the run has an output
contract (`--output-schema`) and a natural final response failed JSON
Schema validation. One event per failed attempt; `error` carries the first
validation error. Unlike the effect `*.failed` events this is not
necessarily terminal — the loop appends a bounded repair turn after each
failure, so a run may emit these and still complete successfully. It has no
`op_id` (it is loop-level, not an effect). Repairs exhausted means the turn
ends in a typed contract-violation error, visible on the supervisor
channel, not in the trace.

### Reserved event names (documented, not yet emitted)

Consumers must accept these without a schema bump when they start flowing:

- `run.started` — no runtime trace event marks run start today (the
  stdout `agent_start` machine event is supervision-facing, not trace).
- `turn.started`, `turn.completed` — turns exist in the session protocol
  (machine events carry `turn_id`), but the runtime trace does not yet
  record turn boundaries.

### Private runtime events (no public projection)

These project to `None` — they are runtime internals, reachable only via
the runtime trace, and may change without notice:

| runtime variant / custom name | why private |
|---|---|
| `HydrationStart` / `HydrationSection` / `HydrationEnd` | context-assembly internals; candidate for a future additive projection |
| `ParStart` / `ParEnd` | execution-structure internals (lineage already public via `parent_op_id`) |
| `Checkpoint` | persistence internals |
| `TurnBudgetExhausted` | budget taxonomy internals (t-1133); candidate for a future additive projection |
| `Custom` (all names except `output_validation_failed`: `gc_collect`, `gc_truncate`, `context_overflow`, `prompt_ir`, `output_contract`, domain tags, ...) | extension/diagnostics channel, unbounded vocabulary — private by default. `output_validation_failed` is the one projected name (see the emitted table); `output_contract` (the run's output-schema hash, replay identity for `--output-schema`) stays private |

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

## Version history

- **1.5** (t-1347) — additive: sub-infer attribution. `run.completed`
  gains `failed_infer_calls` (present only when nonzero) counting Infer
  attempts that ended in `InferError`; failed attempts carry no usage or
  cost (the provider error path returns no Response), so the token/cost
  sums are unchanged. Nested `infer.*` events dispatched by the agent
  loop's `infer` tool now populate the existing `parent_op_id` field with
  the dispatching turn Infer's `op_id` (no new field; the lineage slot was
  always in the envelope). Wire `schema_version` stays `1` per the
  compatibility policy.
- **1.4** (t-1308.10) — additive: the reserved `approval.requested` /
  `approval.resolved` are now emitted (projected from the new runtime
  `ApprovalRequested`/`ApprovalResolved` events) for approval-gated effects
  (DR-7), with `pending_id`/`kind` attrs (plus
  `decision`/`effect_id`/`resolved_by`?/`reason`? on the resolution) and
  full effect identity on `approval.requested`. Wire `schema_version` stays
  `1` per the compatibility policy.
- **1.3** (t-1334) — additive: cost accounting. `infer.completed` gains
  `cached_input_tokens`, `cost_micro_usd`, and `pricing` attrs (present when
  recorded); `run.completed` gains the usage/cost rollup attrs
  (`infer_calls`, `input_tokens`, `output_tokens`, `total_tokens`,
  `cached_input_tokens`?, `cost_micro_usd`?, `uncosted_infer_calls`). Wire
  `schema_version` stays `1` per the compatibility policy.
- **1.2** (t-1308.7) — additive: the reserved `tool.requested` /
  `tool.completed` / `tool.failed` are now emitted (projected from the new
  runtime `ToolCall`/`ToolResult`/`ToolError` events) for registered native
  tools, with `attrs.name` and effect identity on `tool.requested`. Wire
  `schema_version` stays `1` per the compatibility policy.
- **1.1** (t-1308.4) — additive: the reserved `output.validation_failed`
  is now emitted (projected from the runtime `Custom` event
  `output_validation_failed`), with `attempt`/`errors` attrs. Wire
  `schema_version` stays `1` per the compatibility policy.
- **1.0** (t-1308.3) — initial schema: the `infer.*`/`eval.*`/`retrieve.*`/
  `store.*` lifecycle events and `run.completed`.
