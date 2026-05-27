# agentd

A Rust runtime for long-running AI agents. Built on a simple thesis: **Linux is the harness.**

## The Problem with Agent Frameworks

The current landscape — Claude Code, OpenCode, Hermes, Pi, and their kin — wraps models in elaborate scaffolding: tool registries, guardrail layers, prompt templating, orchestration DSLs. The implicit assumption is that the model needs help knowing what to do.

This is a temporary condition. Models are improving fast. The guardrails and affordances these harnesses provide are necessities today, but they're technical debt on the trajectory toward more capable models. A framework that's useful now because the model can't figure out its environment becomes dead weight once it can.

The durable layer is not the harness. It's the substrate: **Linux**.

Sandboxes have always been necessary for secure computing — that predates AI by decades. What changes is everything built on top of the sandbox. The model should wield the full power of a Unix environment directly: files, pipes, processes, the shell, the compiler, the network. These are not tools to be wrapped in JSON schemas. They are the computing environment. The agent lives in it.

agentd is built on this premise. It provides:
- A **process supervisor** for long-lived agent sessions
- A **free monad runtime** that treats Infer and Eval as first-class operations
- A **FIFO-based turn protocol** — as Unix-native as it gets
- A **hydration source registry** for typed context injection
- A **checkpoint/trace system** for resumability and replay

It does not provide: a web dashboard, a plugin marketplace, a YAML pipeline language, or opinions about which tools your agent should have. Those are your problem. agentd is the runtime substrate; the agent wields Linux.

## Core Ideas

### Infer/Eval as the Agentic Loop

Traditional computing can be reduced to a single operation: `eval(structured_data)`. Modern AI is `infer(unstructured_data)`. The agentic runtime is how both of these become integrated: a loop where model inference and environment evaluation alternate indefinitely.

agentd makes this structural with a free monad over `OpF`:

```rust
pub enum OpF<S, A> {
    Infer  { model, prompt, next },  // LLM call
    Tool   { name, args, next },     // environment eval
    Get    { next },                 // state read
    Put    { state, next },          // state write
    Emit   { event, next },          // trace
    Par    { ops, next },            // parallel effects
    Pure(A),
}
```

The agent is a *program*. The runtime is an *interpreter*. This boundary is what makes the system testable, replaceable, and compositional. See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full reasoning.

### Free Monad: Effects as Data

Programs built with `Op` constructors are pure data — they can be inspected, composed, replayed, or run against a mock interpreter before any IO happens. Every new capability is a new `OpF` variant with a new interpreter branch. No async spaghetti.

The free monad implementation was first proved out as a Haskell program in a private repo (`~/omni/live/Omni/Agent/Op.hs`) and used extensively in anger. This port to Rust is basically a directy copy, with some cleanup and simplifications, for public release.

### Hydration Sources: Typed Context Injection

Most systems treat context as an append-only log: strings are appended to a prompt in order to build up a useful context for the agent. RAG extends this with an on-demand filter and injection, but the content is still untyped and appended.

agentd models context injection as a typed registry that gets dynamically rehydrated on every agentic request:

```
Temporal  — recent history, time-indexed events
Semantic  — embedding-retrieved memory, RAG
Knowledge — static workspace files, configs
```

On each agentic turn, the context is built up into a datastructure that the LLM can parse. If anything is missing from the context, runtime tools are available to the agent so that it can search for more information in the sources. This is the right abstraction for a runtime that needs to scale context management as models get larger windows and smarter retrieval.

### Linux-Native Session Model

Each agent session is a process, managed by systemd or whatever other supervisor you want. Turns arrive via a FIFO (named pipe). Message are NUL-terminated. After each turn, a checkpoints is written. The entire protocol is pipes and files.

```sh
# TODO: replace this with an example of `agent`, not `agentd`
agentd start myagent
agentd send myagent "go build the thing"
agentd logs myagent
agentd stop myagent
```

No message broker, no gRPC, no special protocol. If you can write to a file, you can steer your agent.

## Design Philosophy

**The model is not the agent. The loop is the agent.** An agent is a process that persists, reads its environment, acts, and recurs. The LLM is one effect inside that loop — the most important one, but structurally no different from a shell command.

**Guardrails are scaffolding, not architecture.** As models improve, the value of framework-level guardrails approaches zero. The durable value is in the runtime substrate: process model, IO semantics, context hygiene, trace/replay.

**Follow the Unix design principle.** Every hard problem in agent infrastructure has an analogue in operating systems: scheduling, isolation, IPC, persistence, logging. Before reaching for a new abstraction, check if the Unix answer already works.

**Bring your own sandbox.** agentd does not enforce isolation — that's the job of the container or VM you run it in. Sandboxing is a deployment concern, not a framework concern. The agent inside the sandbox gets the full Linux environment.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for a full walkthrough of the free monad design, hydration sources, session model, and interpreter extensibility.

## Crate Structure

```
crates/
  agent-core/   — Op, OpF, interpreter, hydration, provider traits
  agent/        — CLI binary, session loop, standard tools
  agent-oauth/  — OAuth flows for claude-code / openai-codex providers
```

`agent-core` has no IO of its own except through the `ChatProvider` and `Tool` traits. It is the pure kernel. The `agent` binary is the shell around it. This boundary is intentional and load-bearing.

## Status

M1 (single-agent, sequential interpreter) is implemented. Active development:
- M2: session history + FIFO protocol
- M3: hermetic PATH, models.yaml, HydrationSource implementations
- M4: parallel interpreter (real `Par` execution)
- M5: distributed interpreter, multi-VM campaigns

## Prior Art

The design originates in `Omni/Agent/Op.hs`, a Haskell prototype that first demonstrated the free monad Op abstraction at production scale. This Rust port is a faithful translation, not a rewrite. The Haskell codebase remains the reference for Op semantics.

## License

MIT
