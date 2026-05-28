# Roadmap

This repository is the Rust open-core runtime for agentd. The original Haskell `agentd` supervisor exists and is the reference for the daemon/scheduler layer. The Rust port is being built in milestones.

## M1: Rust runtime and CLI foundation

Status: implemented.

- Free monad core in `crates/agent-core/src/op.rs`: `OpF`, `Op`, `and_then`, `map`, and effect constructors.
- Sequential interpreter in `crates/agent-core/src/interpreter.rs`.
- `Infer` calls OpenAI-compatible chat providers through `ChatProvider`.
- `Eval` runs command strings through `$SHELL -c`.
- `Get` and `Put` address temporal history, semantic hydration, and session state by key.
- Passive hydration can inject temporal history and session context before `Infer`.
- `agent` CLI supports one-shot prompts, NUL-framed stdin sessions, FIFO sessions, checkpointing, traces, config files, and model registry loading.
- Tests cover monad laws, op constructors, hydration source dispatch, checkpoint state, provider loop behavior, and OAuth token serialization helpers.

## M2: Rust `agentd` supervisor

Status: future Rust port; working in the original Haskell system.

The supervisor will manage named long-running sessions around the existing `agent` process model:

```sh
agentd start myagent
agentd send myagent "go build the thing"
agentd logs myagent
agentd stop myagent
```

Planned pieces:

- Session registry and process lifecycle management.
- FIFO creation and turn delivery.
- systemd integration or equivalent process supervision.
- Log and checkpoint discovery by session name.
- Restart/resume from latest checkpoint.

## M3: Safer execution and richer hydration

- Hermetic PATH construction for `Eval`.
- First-class sandbox runner integration, e.g. bwrap, containers, VMs, or remote workers.
- More `HydrationSource` implementations for workspace context, semantic recall, and temporal search.
- Better trace provenance for passive context injection.
- Configurable budgets for `Infer`, `Eval`, and recursively emitted `Infer` calls.

## M4: Parallel interpreter

- Implement real concurrent semantics for `Par`.
- Preserve deterministic trace structure where possible.
- Add resource limits and cancellation propagation.

## M5: Distributed interpreter

- Route `Eval` to workers or sandboxes.
- Route `Infer` to model clusters or provider pools.
- Support multi-VM campaigns without changing agent programs.

## Public release checklist

- Keep README examples aligned with shipped binaries.
- Keep architecture docs aligned with the actual public traits.
- Keep OAuth flows tested against real providers before describing them as stable.
- Provide a safe default run recipe for new users.
