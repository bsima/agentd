# OpenTelemetry

`agent` can export run telemetry to an OTLP collector while still writing the JSONL trace used by replay and eval tooling.

Enable it with either:

```sh
agent --otel-endpoint http://localhost:4318 ...
```

or:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 agent ...
```

OTel is off by default. When enabled, `TraceLogger::emit(&Event)` fans out to the normal JSONL sink and an OTel sink.

The OTel sink maps typed trace events to spans:

- `InferCall` / `InferResult` -> `Infer` span
- `EvalCall` / `EvalResult` -> `Eval` span
- `GetCall` / `GetResult` -> `Get` span
- `PutCall` / `PutResult` -> `Put` span
- `HydrationStart` / `HydrationEnd` -> `Hydration` span
- `ParStart` / `ParEnd` -> `Par` span
- custom and lifecycle events become instant spans

Open parent spans are used as parents for child spans, so work inside a `Par` span nests under it. The active OTel span context is also exposed to child `Eval` processes through W3C `TRACEPARENT` / `TRACESTATE` environment variables.

Diagnostic logs are exported through `tracing-subscriber` and OTLP logs when OTel is enabled.

## Span attributes

Spans are the skeleton; attributes are the query surface. A span says *what
happened and how long*; an attribute says *which one, whose, why*. Steering an
agent in Grafana means slicing the span tree by these dimensions, so the
attribute schema is a first-class deliverable, not an afterthought.

The boundary rule: **the core emits structure, the environment supplies
semantics.** `agent` only knows agent-intrinsic concepts — things true of any
agent regardless of what it is working on (model, tokens, tool, exit code,
retries, durations, run/op lineage). It must not bake in any deployment- or
domain-specific keys.

Litmus test for any new attribute: *would this make sense for an agent
refactoring a Django app, or writing a novel?* If yes (`retry.count`, `model`,
`tool.name`) it is core. If no (`task.id`, `kernel.name`) it is
environment-injected.

### Core attributes (shipped, hardcoded, general-purpose)

Resource (set once per process, attached to every span):

- `service.name` = `agentd`
- `agent.name` — ava, kernel-coder, gc-coder, etc.
- `agent.run_id` — promoted from the per-event `run_id`
- `agent.parent_run_id` — who dispatched me; lets the backend build the
  service graph and separate one operator's runs from another's

Span-level, mapped straight off the existing `Event` fields:

- Infer: `gen_ai.request.model` (`model`), `gen_ai.usage.output_tokens`
  (`tokens`), `duration_ms`. Use the `gen_ai.*` semconv names so
  off-the-shelf panels light up. Add `gen_ai.usage.input_tokens` once the
  provider reports it.
- Eval: `tool.name` (sanitized, low-cardinality, e.g. `cargo build`),
  `command` (full string, detail only — not a group-by key), `exit_code`/`ok`,
  `attempt` / `retry.count`, `cwd`, `timeout_ms`, `truncated_*`.
- Get / Put: `key`, `source_count`.
- Par: `branch_count`.

Span status: an `Eval` that exits nonzero sets span status `ERROR`. That makes
"show me failing tool calls" a one-click filter instead of a text search.

The two attributes that turn a pretty tree into a control surface:

- `retry.count` / `attempt` on repeated Evals — the literal "is this agent
  stuck" signal (e.g. retried the same failing `cargo build` 6 times). Agent
  intrinsic, so it stays in core.
- ERROR status propagation — see above.

### Environment attributes (injected, agent is agnostic to the keys)

Deployment- and domain-specific dimensions (`task.id`, `kernel.name`,
`kernel.config`, `deployment.team`, ...) are **opaque key-value pairs the agent
does not interpret.** Two standard mechanisms, both native to OTel:

- **Static, per-run:** `OTEL_RESOURCE_ATTRIBUTES`. The deployment sets
  e.g. `OTEL_RESOURCE_ATTRIBUTES="task.id=t-1069,deployment.team=parasail"` and
  the resource is initialized from that env var; the keys are stamped onto every
  span. `agent` writes no code for specific keys — it just initializes the
  resource from the standard variable.
- **Dynamic, per-span:** the `Custom` event is the supported way to attach a
  domain tag to the current span mid-run (e.g. `kernel.name` per Eval). `agent`
  defines the *mechanism* (attach kv to the current span), never the *keys*.

So a deployment can `group by kernel.name` in its own Grafana via env +
`Custom` events, while the open-source agent stays domain-free. Same query, but
the dimension is injected by the environment, not shipped by the core.

### Cardinality discipline

Attributes you filter/group by must be low-cardinality: `agent.name`,
`tool.name`, `model`, `exit_code`, and any low-card environment keys. Freeform,
unique-per-span values (`command` string, prompt text, file paths) are
high-cardinality — fine as span detail, but indexing/grouping by them blows up
the trace store. Rule: *enums and ids you'd put in a WHERE clause are
attributes; freeform blobs are span detail, not group-by keys.*

### Span identity

`run_id` and `op_id` are span identity, not filterable attributes: `run_id` ->
resource (`agent.run_id`), `op_id` -> span_id / parent linkage. Derive
`parent_span_id` from the existing `op_id` lineage, not from a wall-clock
"currently open span" stack heuristic — that is what makes Par-nesting and
cross-agent nesting correct rather than fragile.

### Naming convention

Adopt `gen_ai.*` semconv names where they exist (model, tokens). Namespace
everything agentd-specific under `agent.*`. Do not invent names that collide
with the GenAI semconv (experimental, but where off-the-shelf LLM-observability
panels look). One namespace = one grep.
