# agentd

A Rust runtime for long-running AI agents. Built on a simple thesis: **Linux is the harness.**

## The Problem with Agent Frameworks

The current landscape — Claude Code, OpenCode, Hermes, Pi, and their kin — wraps models in elaborate scaffolding: tool registries, guardrail layers, prompt templating, orchestration DSLs. The implicit assumption is that the model needs help knowing what to do.

This is a temporary condition. Models are improving fast. The guardrails and affordances these harnesses provide are necessities today, but they're technical debt on the trajectory toward more capable models. A framework that's useful now because the model can't figure out its environment becomes dead weight once it can.

The durable layer is not the harness. It's the substrate: **Linux**.

Sandboxes have always been necessary for secure computing — that predates AI by decades. What changes is everything built on top of the sandbox. The model should wield the full power of a Unix environment directly: files, pipes, processes, the shell, the compiler, the network. These are not tools to be wrapped in JSON schemas. They are the computing environment. The agent lives in it.

agentd is built on this premise. The open-core Rust port currently provides:
- A **free monad runtime** that treats Infer and Eval as first-class operations
- An `agent` CLI for one-shot prompts and NUL-framed session loops
- A **FIFO-compatible turn protocol** — as Unix-native as it gets
- A **unified Get/Put interface** for hydration and session state
- A **checkpoint/trace system** for resumability and debugging

The full `agentd` process supervisor exists today in the original Haskell system under `Omni/Agentd`. Porting that daemon/scheduler layer to Rust is future work. This repository is the Rust runtime and CLI foundation it will sit on.

It does not provide: a web dashboard, a plugin marketplace, a YAML pipeline language, built-in sandboxing, or opinions about which Unix tools your agent should have. Those are your problem. agentd is the runtime substrate; the agent wields Linux.

## Core Ideas

### Infer/Eval as the Agentic Loop

Traditional computing can be reduced to a single operation: `eval(structured_data)`. Modern AI is `infer(unstructured_data)`. The agentic runtime is how both of these become integrated: a loop where model inference and environment evaluation alternate indefinitely.

agentd makes this structural with a free monad over `OpF`:

```rust
pub enum OpF<S, A> {
    Infer { model, prompt, next },  // LLM call: infer(unstructured)
    Eval  { command, next },        // process call: eval(structured) — currently $SHELL -c
    Get   { key, next },            // state/context read
    Put   { key, value, next },     // state/context write
    Emit  { event, next },          // trace
    Par   { ops, next },            // parallel effects
    Pure(A),
}
```

The agent is a *program*. The runtime is an *interpreter*. This boundary is what makes the system testable, replaceable, and compositional. See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full reasoning.

### Meta-Circular: Infer Calling Infer

All `OpF` variants are available to agent programs — including `Infer`. An agent that can emit `Infer` ops can construct sub-agents with different models, different prompts, different context windows. Multi-agent orchestration is not a special layer built on top: it is just an agent program that contains multiple `Infer` calls. The outer agent *is* the orchestrator.

This is the SICP meta-circular insight applied to agents: `eval` calling `eval` collapses the interpreter/object-language boundary. The same structure that enables agent programs to compose is the structure that enables multi-agent systems — no additional abstraction needed.

### Free Monad: Effects as Data

Programs built with `Op` constructors are pure data — they can be inspected, composed, replayed, or run against a mock interpreter before any IO happens. Every new capability is a new `OpF` variant with a new interpreter branch. No async spaghetti.

The free monad implementation was first proved out as a Haskell program in a private repo (`Omni/Agent/Op.hs`) and used extensively at production scale. This Rust port is a direct translation, with some cleanup, for public release.

### Get/Put: The Unified Hydration Interface

Most systems treat context as an append-only log. agentd models all context injection — temporal history, semantic recall, workspace files, session state — as `Get`/`Put` operations over named keys. The interpreter decides what each key means:

```
Get("temporal-passive")   →  inject recent chat history
Get("semantic:topic")     →  vector similarity search
Get("session:state")      →  read current checkpoint
Put("session:state", v)   →  write checkpoint
```

This creates a clean 2×2 taxonomy:

|                | **Passive** (interpreter auto-emits before turn) | **Active** (agent-emitted)              |
|----------------|---------------------------------------------------|-----------------------------------------|
| **Temporal**   | Recent chat history, recent events                | `Get("temporal:search query")`          |
| **Semantic**   | Similarity-based RAG, static workspace            | `Get("semantic:topic")`                 |

Passive sources fire automatically — the runtime assembles the context window before the model ever sees the request. Active sources are available to the agent program on demand, via the same `Get` op. One interface, all sources.

### Linux-Native Session Model

The Rust CLI runs an agent loop as a normal process. It can handle one prompt, read NUL-terminated turns from stdin, or read NUL-terminated turns from a FIFO path. After each turn, it can write checkpoints. The entire protocol is pipes and files.

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

The future Rust `agentd` supervisor will wrap this process model with commands like:

```sh
agentd start myagent
agentd send myagent "go build the thing"
agentd logs myagent
agentd stop myagent
```

Those supervisor commands work in the Haskell implementation today. They are not part of this Rust repo yet. No message broker, no gRPC, no special protocol is required. If you can write NUL-terminated bytes to stdin or a FIFO, you can steer your agent.

## Design Philosophy

**The model is not the agent. The loop is the agent.** An agent is a process that persists, reads its environment, acts, and recurs. The LLM is one effect inside that loop — the most important one, but structurally no different from a shell command.

**Guardrails are scaffolding, not architecture.** As models improve, the value of framework-level guardrails approaches zero. The durable value is in the runtime substrate: process model, IO semantics, context hygiene, trace/replay.

**Follow the Unix design principle.** Every hard problem in agent infrastructure has an analogue in operating systems: scheduling, isolation, IPC, persistence, logging. Before reaching for a new abstraction, check if the Unix answer already works.

**Bring your own sandbox.** agentd does not enforce isolation — that's the job of the container or VM you run it in. Sandboxing is a deployment concern, not a framework concern. The agent inside the sandbox gets the full Linux environment.

**Security warning:** `Eval` currently runs model-requested commands with `$SHELL -c`. Do not run this against a workspace, home directory, network, or credential environment you are not willing to give to the model. Run it in a container, VM, or other sandbox. Trace logs also record commands and command output, which may contain secrets.

## Quickstart

Build and test:

```sh
cargo test
cargo build --release
```

Configure a model registry at `~/.config/agent/models.yaml`, or copy the example:

```sh
mkdir -p ~/.config/agent
cp examples/models.yaml ~/.config/agent/models.yaml
```

Set an API key for the provider you configured:

```sh
export OPENROUTER_API_KEY=...
```

Run a one-shot prompt:

```sh
cargo run -- --model openrouter/auto "say hello"
```

You can also skip the registry and pass a raw model id. In that case the CLI uses `OPENROUTER_BASE_URL` or `https://openrouter.ai/api/v1`, and `AGENT_API_KEY` or `OPENROUTER_API_KEY`.

## Running Safely

The default interpreter gives the model direct shell execution. The safest default is to run it inside a disposable workspace with only the files and credentials needed for that task.

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

For real use, prefer a purpose-built image that contains `agent`, your allowed toolchain, and no ambient secrets. Add network access only when the task requires it; model providers usually require it. Mount source code read-only unless the agent is supposed to edit it. Keep trace and checkpoint directories outside your main home directory if command output may contain secrets.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for a full walkthrough of the free monad design, hydration sources, session model, and interpreter extensibility. See [ROADMAP.md](./ROADMAP.md) for the Rust port plan, including the future `agentd` supervisor.

## Crate Structure

```
crates/
  agent-core/   — Op, OpF, interpreter, hydration, provider traits
  agent/        — CLI binary, one-shot prompts, NUL/FIFO sessions
  agent-oauth/  — experimental OAuth flows for claude-code / openai-codex providers
```

`agent-core` defines the Op language, provider traits, hydration model, trace logger, and sequential interpreter. The current sequential interpreter performs IO for `Infer`, `Eval`, trace writes, and checkpoint reads/writes. The `agent` binary is the CLI shell around it. This boundary is intentional and load-bearing.

## Status

M1 is implemented: single-agent CLI, sequential interpreter, shell-backed `Eval`, model-backed `Infer`, NUL/FIFO session input, traces, checkpoints, hydration registry, and model registry loading.

Active development:
- M2: Rust `agentd` supervisor/daemon port from the working Haskell implementation
- M3: hermetic PATH, stronger sandbox integration, richer HydrationSource implementations
- M4: parallel interpreter with real `Par` execution
- M5: distributed interpreter, multi-VM campaigns

## Prior Art

The design originates in `Omni/Agent/Op.hs`, a Haskell prototype that first demonstrated the free monad Op abstraction at production scale. This Rust port is a faithful translation, not a rewrite. The Haskell codebase remains the reference for Op semantics. The meta-circular Infer-emitting-Infer pattern has direct precedent in the SICP meta-circular evaluator.

## License

MIT
