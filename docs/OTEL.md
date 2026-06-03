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
