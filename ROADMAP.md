# Roadmap

This repo is the Rust open-core runtime for `agentd`.

The original Haskell `agentd` supervisor already exists and remains the reference for the daemon/scheduler layer. The Rust port is being built in milestones.

## M1: Rust runtime and CLI foundation

Status: implemented.

What exists now:

- free monad core in `crates/agent-core/src/op.rs`: `OpF`, `Op`, `and_then`, `map`, and effect constructors
- sequential interpreter in `crates/agent-core/src/interpreter.rs`
- `Infer` through OpenAI-compatible chat providers via `ChatProvider`
- `Eval` through `$SHELL -c`
- `Get` and `Put` for temporal history, semantic hydration, and session state
- passive hydration before `Infer`
- `agent` CLI for one-shot prompts, NUL-framed stdin sessions, FIFO sessions, checkpointing, traces, config files, and model registry loading
- tests for monad laws, op constructors, hydration dispatch, checkpoint state, provider loop behavior, and OAuth token serialization helpers

## M2: Rust `agentd` supervisor

Status: future Rust port. Working in the Haskell system today.

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

## M3: Safer execution and richer hydration

The bigger issue after M2 is control. The runtime can already execute effects. Next it needs better boundaries around those effects.

Planned work:

- hermetic PATH construction for `Eval`
- first-class sandbox runner integration: `bwrap`, containers, VMs, or remote workers
- more `HydrationSource` implementations for workspace context, semantic recall, and temporal search
- better trace provenance for passive context injection
- configurable budgets for `Infer`, `Eval`, and recursively emitted `Infer` calls

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
