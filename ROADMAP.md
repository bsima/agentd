# Roadmap

This repo is the Rust open-core runtime for `agentd`. It started as a port of
a working Haskell system; the Rust implementation — runtime, CLI, and
supervisor — is now the reference. Milestones below are marked with their
actual status; where something is future, it says future.

## M1: Rust runtime and CLI foundation — done

What exists:

- free monad core in `crates/agent-core/src/op.rs`: `OpF`, `Op`, `and_then`, `map`, and effect constructors (now a library builder/test API)
- sequential interpreter in `crates/agent-core/src/interpreter.rs`
- `Infer` through OpenAI-compatible and Anthropic chat providers via `ChatProvider`
- `Eval` through a configured shell with timeout, output caps, cwd, and env policy
- passive hydration before `Infer`
- `agent` CLI for one-shot prompts, NUL-framed stdin sessions, FIFO sessions, checkpointing, traces, config files, and model registry loading
- trace readback and replay for recorded `Infer`/`Eval` results
- tests for monad laws, op constructors, hydration dispatch, checkpoint state, provider loop behavior, eval policy, replay, and OAuth token serialization helpers

(The key-based `Get`/`Put` state effects that shipped with M1 were later
deleted in favor of the typed `Retrieve`/`Store` hydration effects — see
[docs/MEMORY.md](./docs/MEMORY.md).)

## AgentIR track: serializable runtime core — done

The CLI is IR-only (the `--runtime` flag is removed; the closure-based Op
layer remains a library builder/test API). Everything the track set out to
build exists:

- serializable `Program`/`Block`/`Instr`/`Terminator`/`Expr` types, validated before any effect runs
- an explicit machine with mid-turn checkpoints and a step limit
- stable effect ids derived from program hash plus dynamic path; replay keys on them, and divergence errors name the block, instruction, and control path
- a normalization pass: programs normalize to strict SSA and the program hash is taken over the canonical form, so alpha-equivalent programs share identity
- `Par`: dynamic-width map-Par with concurrent branches, join-all in declaration order, errors-as-values per branch, and order-independent replay
- failure semantics: error events with stable ids; effects can abort or bind their errors as values

The status table in [docs/AGENT_IR.md](./docs/AGENT_IR.md) tracks every
design item; all rows are implemented (the in-memory STM store was removed
by design rather than built — machine env and the hydration effects absorbed
its uses).

## Memory and hydration track — done

`Retrieve`/`Store` effects over a registry of hydration sources and sinks;
the file-backed memory backend (`--memory-dir`) with optional
embedding-based semantic retrieval; model-facing `remember`/`recall` tools;
checkpointing through the `ChatHistory` sink. Design:
[docs/MEMORY.md](./docs/MEMORY.md); the provider-author contract:
[docs/PROVIDERS.md](./docs/PROVIDERS.md).

## Context GC track — done, evals ongoing

Five strategies (`stack` default, plus `ring`, `mark-sweep`, `semantic`,
`generational`), hard guards (system prompt and last user message always
survive), cache-prefix preservation, eviction markers with escalation, the
progress ledger, and a collect-on-overflow backstop. Strategy promotion is
gated on an offline matrix plus recorded behavioral evals
([evals/gc/](./evals/gc/README.md)); behavioral validation of the newest
strategy (`generational`) is the open item. Design and user guide:
[docs/GC.md](./docs/GC.md).

## M2: Rust `agentd` supervisor — done

Implemented as `crates/agentd` (binary: `agentd`), a thin CLI over a
conventional directory layout rather than a daemon:

- `start`/`stop`/`resume`/`status`/`logs`/`send`/`attach` against the shipped `agent` binary
- turn delivery over the session FIFO with a v1 turn envelope; responses correlated by turn id, `send --timeout` leaves the turn running and `attach` re-attaches
- the on-disk spec (`<name>/agent.md`) is the canonical session config; `set-model`/`set-provider`/`set-system-prompt`/`set-max-turns` edit it in place, and hand-edits are equally valid
- `gen-systemd` emits a systemd user unit (restart-on-failure through `agentd resume`)
- covered by credential-free integration tests and an offline end-to-end eval

Design and decisions: [docs/SUPERVISOR.md](./docs/SUPERVISOR.md).

## SDK track — done (v1 surface)

`crates/agent-sdk`: embed the agent loop in-process (`Agent`/`Tool`/`Runner`)
with typed native tools (executed as recorded/replayable IR effects, never
via shell), output contracts, approval hooks, streaming public events, and
replay; or drive a persistent `Session` that spawns the `agent` binary and
correlates turns by id. Native tools and injected providers do not cross the
process boundary into spawned sessions yet.

## Guidance track — done, A/Bs ongoing

The runtime ships operations guidance to its models: per-tool descriptions
plus a capability-keyed, budget-aware prompt fragment delivered as a
separate Developer section (so a user `--system-prompt` composes instead of
destroying it; `--no-runtime-guidance` opts out). The delegation block is
validated on recorded behavioral runs; other blocks shipped default-on with
their A/Bs tracked in [docs/GUIDANCE.md](./docs/GUIDANCE.md) — one block
(memory discipline) already failed its A/B and was demoted back to draft.

## Observability and accounting — done

- structured JSONL traces with stable effect identity; a versioned public event schema ([docs/TRACE_SCHEMA.md](./docs/TRACE_SCHEMA.md)) for consumers
- OpenTelemetry OTLP export ([docs/OTEL.md](./docs/OTEL.md))
- cost accounting: integer micro-USD pricing from the model registry, per-`Infer` usage/cost recorded in traces, `agent cost` for rollups
- human-in-the-loop approvals: durable pending-effect records resolved by `agent approvals`, traced and replayed as data

## Release stabilization track — ongoing discipline

v0.1.0 and v0.2.0 are released (static musl `agent` binaries plus
`models.yaml.example`). Every release is gated on:

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test`
- `./evals/smoke.sh` for an offline CLI/replay smoke test
- `./evals/release.sh` (runs all of the above plus the offline shell evals)

Eval fixtures stay small, deterministic, and runnable without API keys by
default; the online evals (`./evals/release-online.sh` and the `RUN_*`-gated
scripts) require provider keys and are optional.

## M3: Sandboxing and richer hydration — partial

Shipped:

- credential stripping in the default `Eval` env policy (the model cannot read the key the agent runs on)
- direct-exec argv `Eval` (no shell, no re-parsing) as the step toward verifiable payloads
- approval gates over shell commands and sink writes
- hydration sources with provenance through PromptIR and traces

Still future:

- hermetic PATH construction for `Eval`
- first-class sandbox runner integration: `bwrap`, containers, VMs, or remote workers (today the honest story is the documented container pattern in the README — you sandbox the process, the runtime does not do it for you)

## PromptIR track: optimizable context primitive — structure shipped, optimization future

PromptIR v1 is implemented as a traceable representation: hydration builds
labeled, sourced, budgeted sections; they compile deterministically to
provider messages; and every `Infer` records the PromptIR hash and section
summaries, so context provenance is auditable per call. The runtime guidance
fragment rides this machinery as a Developer/Constraint section.

The optimization passes (compression, relevance-weighted budgeting,
DSPy-style or rate-distortion-style rewriting) remain future work — the
representation exists precisely so they have something stable to operate on.
Design: [docs/PROMPT_IR.md](./docs/PROMPT_IR.md).

## Intent track: verifiable Eval payload — future

Intent is the planned structured/verifiable payload for `Eval`. Shell remains
the compatibility backend; argv `Eval` (shipped) is the intermediate step —
still an opaque external program, but invoked without a shell. The Intent
compiler/runtime boundary is not built.

## M4: Parallel interpreter — core shipped

`Par` executes concurrently in the IR runtime (see the AgentIR track above).
Remaining work:

- concurrent dispatch of the model's own multi-tool turn batch (the highest-volume fan-out site; today the tool loop executes a turn's tool calls sequentially)
- cancellation semantics and mid-`Par` checkpoints (deliberately excluded from v1)

## M5: Distributed interpreter — future

The long-term target is the same program shape running across machines:
route `Eval` to workers or sandboxes, route `Infer` to model clusters or
provider pools, support multi-VM campaigns without changing agent programs.
Not started; the supervisor's directory layout deliberately avoids baking in
single-host assumptions.

## Current frontier

The runtime substrate is built. The open work is mostly evidence and
boundaries, not features:

- **Behavioral evals**: online validation of the `generational` GC strategy; guidance A/Bs at realistic context budgets (the shipped blocks were only observed under extreme pressure); approval-awareness and chaining arms
- **Delegation mechanics**: a delegate-model catalog in the `infer` tool schema, budget caps on sub-inference, child-process usage lineage in parent traces
- **Model-visible runtime state**: surfacing running usage/cost and GC pressure to the model, so economy guidance can be quantitative
- **Sandbox profiles**: first-class `bwrap`/container integration for `Eval`
- **PromptIR optimization**: the passes the representation was built for

## Public release checklist

Before describing this as stable:

- README examples match shipped binaries
- architecture docs match public traits
- OAuth flows are tested against real providers
- new users have a safe default run recipe
