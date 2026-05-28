# agentd

A Rust runtime for long-running AI agents.

The thesis is simple: **Linux is the harness.**

## Why this exists

Most agent frameworks wrap the model in scaffolding: tool registries, prompt templates, orchestration DSLs, guardrail layers, plugin systems. That makes sense today. Models still need help navigating their environment.

I think that is temporary. As models improve, a lot of framework machinery becomes dead weight. The durable layer is not the harness. It is the substrate: Linux.

The model should get a real Unix environment. Files. Pipes. Processes. The shell. The compiler. The network, if the sandbox allows it. These are not JSON-schema tools. They are the computing environment.

`agentd` is built around that model.

This repo currently provides:

- a free monad runtime where `Infer` and `Eval` are first-class operations
- an `agent` CLI for one-shot prompts and NUL-framed session loops
- FIFO-compatible turn delivery
- a unified `Get`/`Put` interface for context and session state
- traces and checkpoints for replay, debugging, and resumability

The full `agentd` process supervisor exists today in the original Haskell system under `Omni/Agentd`. Porting that daemon layer to Rust is future work. This repo is the Rust runtime and CLI foundation.

It does not provide a dashboard, plugin marketplace, YAML pipeline language, built-in sandbox, or curated tool universe. Bring those yourself if you want them. The point here is the runtime substrate.

## Core ideas

### The agent is the loop

Traditional computing is:

```text
eval(structured_data)
```

Modern ML is:

```text
infer(unstructured_data)
```

An agent alternates between the two. It infers from context, evaluates effects against the environment, reads the result, then infers again.

`agentd` makes that structure explicit with a free monad over `OpF`:

```rust
pub enum OpF<S, A> {
    Infer { model, prompt, next },  // LLM call: infer(unstructured)
    Eval  { command, next },        // process call: eval(structured), currently $SHELL -c
    Get   { key, next },            // state/context read
    Put   { key, value, next },     // state/context write
    Emit  { event, next },          // trace
    Par   { ops, next },            // parallel effects
    Pure(A),
}
```

The agent is a program. The runtime is an interpreter. That boundary is load-bearing. It makes the system testable, replayable, and replaceable.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the longer version.

### Infer can call Infer

All `OpF` variants are available to agent programs, including `Infer`.

So a multi-agent system is not a special framework layer. It is just an agent program that emits multiple `Infer` calls, maybe with different models, prompts, budgets, or context windows. The outer agent is the orchestrator.

This is the SICP meta-circular idea applied to agents. `eval` calling `eval` collapses the interpreter/object-language boundary. `Infer` calling `Infer` does the same thing for agents.

### Effects are data

Programs built with `Op` constructors do not perform IO immediately. They are data. You can inspect them, transform them, replay them, or run them against a mock interpreter.

That is the bigger lever. Instead of hiding work inside arbitrary async functions, the runtime keeps model calls, process calls, state reads, state writes, trace events, and parallel branches visible as data.

Adding a capability means adding an `OpF` variant and teaching interpreters what it means.

### Get/Put is the hydration interface

Most systems treat context as an append-only prompt log. `agentd` models context as keyed reads and writes.

```text
Get("temporal-passive")   -> recent chat history
Get("semantic:topic")     -> vector search or other recall
Get("session:state")      -> current checkpoint
Put("session:state", v)   -> write checkpoint
```

The interpreter decides what each key means.

That gives one interface for passive context injection, active recall, workspace context, and session state.

|                | Passive, interpreter-owned | Active, agent-emitted |
|----------------|----------------------------|------------------------|
| Temporal       | recent events/history       | `Get("temporal:query")` |
| Semantic       | RAG/static workspace        | `Get("semantic:topic")` |

Passive sources run before the model sees a turn. Active sources are available when the agent decides it needs them.

### Sessions are Unix processes

The Rust CLI runs an agent loop as a normal process. It can take one prompt, read NUL-terminated turns from stdin, or read NUL-terminated turns from a FIFO path. After each turn it can write checkpoints. The protocol is pipes and files.

```sh
# One-shot prompt
agent --model openai/gpt-4o-mini "inspect this repo and summarize it"

# NUL-framed stdin session
printf 'go build the thing\0\0' | agent --session --checkpoint-dir .agent-checkpoints

# FIFO-driven session
mkfifo .agent.fifo
agent --fifo .agent.fifo --checkpoint-dir .agent-checkpoints &
printf 'run cargo test\0' > .agent.fifo
```

The future Rust supervisor will wrap this with commands like:

```sh
agentd start myagent
agentd send myagent "go build the thing"
agentd logs myagent
agentd stop myagent
```

Those commands work in the Haskell implementation today. They are not in this Rust repo yet.

No broker is required. No gRPC protocol is required. If you can write NUL-terminated bytes to stdin or a FIFO, you can steer the agent.

## Design philosophy

**The model is not the agent. The loop is the agent.** The LLM is one effect inside the loop. It is the important effect, but structurally it is still an effect.

**Guardrails are scaffolding, not architecture.** Framework-level guardrails are useful now. They should not define the system boundary.

**Use the Unix answer first.** Scheduling, isolation, IPC, persistence, logging, process supervision: operating systems already have models for these.

**Bring your own sandbox.** `agentd` does not enforce isolation. Run it in the container, VM, jail, or remote worker you trust. Inside that boundary, the agent gets Linux.

**Security warning:** `Eval` currently runs model-requested commands with `$SHELL -c`. Do not point it at a workspace, home directory, network, or credential environment you are not willing to hand to the model. Trace logs record commands and output, which may contain secrets.

## Quickstart

Build and test:

```sh
cargo test
cargo build --release
```

Configure a model registry:

```sh
mkdir -p ~/.config/agent
cp examples/models.yaml ~/.config/agent/models.yaml
```

Set the provider key:

```sh
export OPENROUTER_API_KEY=...
```

Run a one-shot prompt:

```sh
cargo run -- --model openrouter/auto "say hello"
```

You can also skip the registry and pass a raw model id. Then the CLI uses `OPENROUTER_BASE_URL` or `https://openrouter.ai/api/v1`, and `AGENT_API_KEY` or `OPENROUTER_API_KEY`.

## Running safely

The default interpreter gives the model direct shell execution. The sane default is a disposable workspace with only the files and credentials needed for the task.

A minimal container pattern:

```sh
cargo build --release
mkdir -p .agent-home/.config/agent .agent-work
cp -R ./your-project .agent-work/project

podman run --rm -it \
  -e SHELL=/bin/sh \
  -e OPENROUTER_API_KEY \
  -v "$PWD/target/release/agent:/usr/local/bin/agent:ro" \
  -v "$PWD/.agent-home:/home/agent" \
  -v "$PWD/examples/models.yaml:/home/agent/.config/agent/models.yaml:ro" \
  -v "$PWD/.agent-work:/work" \
  -w /work/project \
  docker.io/library/rust:1 \
  agent --model openrouter/auto "inspect this project"
```

For real use, prefer a purpose-built image with `agent`, the allowed toolchain, and no ambient secrets. Add network only when the task needs it. Mount source read-only unless the agent is supposed to edit it. Keep traces and checkpoints outside your main home directory if command output may contain secrets.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the free monad design, hydration model, session model, and interpreter story. See [ROADMAP.md](./ROADMAP.md) for the Rust port plan.

## Crate structure

```text
crates/
  agent-core/   -- Op, OpF, interpreter, hydration, provider traits
  agent/        -- CLI binary, one-shot prompts, NUL/FIFO sessions
  agent-oauth/  -- experimental OAuth flows for claude-code / openai-codex providers
```

`agent-core` defines the Op language, provider traits, hydration model, trace logger, and sequential interpreter. `agent` is the CLI shell around it. This boundary is intentional.

## Status

M1 is implemented: single-agent CLI, sequential interpreter, shell-backed `Eval`, model-backed `Infer`, NUL/FIFO session input, traces, checkpoints, hydration registry, and model registry loading.

Active development:

- M2: Rust `agentd` supervisor/daemon port from the working Haskell implementation
- M3: hermetic PATH, stronger sandbox integration, richer `HydrationSource` implementations
- M4: parallel interpreter with real `Par` execution
- M5: distributed interpreter, multi-VM campaigns

## Prior art

The design comes from `Omni/Agent/Op.hs`, a Haskell prototype that proved the free monad Op abstraction in production use. This Rust port is a translation, not a rewrite. The Haskell codebase remains the reference for Op semantics.

The meta-circular `Infer`-emitting-`Infer` pattern has direct precedent in the SICP meta-circular evaluator.

## License

MIT
