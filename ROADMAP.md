# Roadmap

This repo is the Rust open-core runtime for `agentd`.

The original Haskell `agentd` supervisor already exists and remains the reference for the daemon/scheduler layer. The Rust port is being built in milestones.

## M1: Rust runtime and CLI foundation

Status: implemented.

What exists now:

- free monad core in `crates/agent-core/src/op.rs`: `OpF`, `Op`, `and_then`, `map`, and effect constructors
- sequential interpreter in `crates/agent-core/src/interpreter.rs`
- `Infer` through OpenAI-compatible chat providers via `ChatProvider`
- `Eval` through a configured shell with timeout, output caps, cwd, and env policy
- `Get` and `Put` for temporal history, semantic hydration, and session state
- passive hydration before `Infer`
- `agent` CLI for one-shot prompts, NUL-framed stdin sessions, FIFO sessions, checkpointing, traces, config files, and model registry loading
- trace readback and replay for recorded `Infer`/`Eval` results
- tests for monad laws, op constructors, hydration dispatch, checkpoint state, provider loop behavior, eval policy, replay, and OAuth token serialization helpers

## Release stabilization track

Status: active for the current minimal release.

The current release should stay small and prove existing workflows before larger IR work starts.

Required evals/checks:

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test`
- `./evals/smoke.sh` for an offline CLI/replay smoke test
- `RUN_RUST_AGENT_INTEGRATION=1 ~/omni/live/Omni/Agentd/Test/rust-agent-integration.sh` for persistent `agentd` compatibility

Planned work:

- keep eval fixtures small, deterministic, and runnable without API keys by default
- add one optional online eval for model + shell-backed `Eval`
- gate release claims on these evals passing
- document any Haskell `agentd` compatibility assumptions in this repo

## AgentIR track: serializable runtime core

Status: required for the next stable release.

The current `Op` free monad uses Rust closure continuations. That is useful for an M1 runtime, but it is not the final interpreter representation. The next stable release should introduce a serializable AgentIR and make the runtime IR-first. The concrete AST/CFG design is captured in [docs/AGENT_IR.md](./docs/AGENT_IR.md).

Planned work:

- add serializable `Program`, `Block`, `Instr`, `Terminator`, `Expr`, and `Var` types
- model effects as AgentIR instructions: `Infer`, `Eval`, `Get`, `Put`, `Emit`, and later `Par`
- implement an explicit machine state with block, program counter, environment, state, budgets, trace, and continuation stack
- make checkpoints serialize machine snapshots, not only chat history between turns
- port the current `agent_loop` to AgentIR while preserving existing CLI behavior
- derive stable effect IDs from program hash plus dynamic path, then use them for trace replay and divergence detection
- keep the closure-based `Op` API only as an ergonomic builder or compatibility layer once AgentIR is ready

Acceptance:

- the existing smoke eval and Haskell `agentd` integration pass with the AgentIR interpreter
- replay does not depend on sequential incidental op numbers
- a mid-turn checkpoint can resume without replaying completed effects

## PromptIR track: optimizable context primitive

Status: separate major initiative.

PromptIR is the structured payload for `Infer`. It should represent context as labeled, sourced, budgeted sections that can later be optimized with DSPy-style or rate-distortion-style passes. The concrete design is captured in [docs/PROMPT_IR.md](./docs/PROMPT_IR.md).

Planned work:

- port the core shape from `~/omni/live/Omni/Agent/Prompt/IR.hs`
- add Rust `PromptIR`, `Section`, `SectionSource`, `CompositionMode`, `Priority`, `TokenBudget`, `ContextStrategy`, and `ContextRequest`
- change hydration from direct prompt string concatenation to `SourceResult -> Section -> PromptIR -> ChatMessage[]`
- trace PromptIR hashes and section metadata for every `Infer`
- keep optimization passes out of the minimal release; start with faithful structure and provenance

Acceptance:

- flat provider prompts are byte-for-byte or semantically equivalent to current prompts before optimization is enabled
- hydration provenance survives in traces as section IDs, sources, priorities, and hashes

## M3: Sandboxing and richer hydration

The bigger issue after PromptIR is control and context quality. The runtime can already execute effects. Next it needs better boundaries around those effects and better sources for model context.

Planned work:

- hermetic PATH construction for `Eval`
- first-class sandbox runner integration: `bwrap`, containers, VMs, or remote workers
- safer default `Eval` policy for cwd, environment, filesystem, and network access
- more `HydrationSource` implementations for workspace context, semantic recall, and temporal search
- active `Get("semantic:...")` and passive PromptIR sections backed by the same source registry
- better trace provenance for passive context injection, including section IDs, sources, priorities, and hashes
- richer runtime budgets for `Infer`, recursively emitted `Infer` calls, and `Eval`

Acceptance:

- shell-backed `Eval` can run inside a documented sandbox profile
- hydration sources preserve provenance through PromptIR and traces
- existing workflows keep working with explicit opt-outs for stricter sandbox defaults where needed

## M2: Rust `agentd` supervisor

Status: future Rust port. Working in the Haskell system today (`~/omni/live/Omni/Agentd`).

The supervisor will manage named long-running sessions around the existing `agent` process model:

```sh
agentd start myagent
agentd send myagent "go build the thing"
agentd logs myagent
agentd stop myagent
```

Planned work:

- session registry and process lifecycle management
- FIFO creation and turn delivery
- systemd integration or equivalent process supervision
- log and checkpoint discovery by session name
- restart/resume from latest checkpoint

## Intent track: verifiable Eval payload

Status: separate major initiative and potential commercial add-on.

Intent is the structured/verifiable payload for `Eval`. Shell remains the compatibility backend. Intent is the high-assurance backend for deterministic, inspectable, and eventually commercial agentic workloads.

Planned work:

- generalize `Eval` from a command string to an `EvalRequest`
- support `EvalRequest::Shell` for current behavior
- add `EvalRequest::Intent` once the Intent compiler/runtime boundary is ready
- record eval request hashes, verification status, structured errors, counterexamples, and result hashes in traces
- use Intent structured failures as training/eval signal for PromptIR optimization and agent retry loops

Acceptance:

- shell-backed `Eval` behavior remains compatible with existing workflows
- Intent-backed `Eval` can typecheck/verify/compile/run without changing AgentIR control flow
- failed Intent verification returns structured observations that can be fed back into `Infer`

## M4: Parallel interpreter

`Par` exists in the Op language today, but M1 interprets it sequentially. M4 gives it real scheduling semantics.

Planned work:

- concurrent execution for `Par`
- deterministic trace structure where possible
- resource limits and cancellation propagation

## M5: Distributed interpreter

The long-term target is the same program shape running across machines.

Planned work:

- route `Eval` to workers or sandboxes
- route `Infer` to model clusters or provider pools
- support multi-VM campaigns without changing agent programs

## Public release checklist

Before describing this as stable:

- README examples match shipped binaries
- architecture docs match public traits
- OAuth flows are tested against real providers
- new users have a safe default run recipe
