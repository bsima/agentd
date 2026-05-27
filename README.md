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
- A **unified Get/Put interface** for all hydration and state sources
- A **checkpoint/trace system** for resumability and replay

It does not provide: a web dashboard, a plugin marketplace, a YAML pipeline language, or opinions about which tools your agent should have. Those are your problem. agentd is the runtime substrate; the agent wields Linux.

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

Each agent session is a process, managed by systemd or whatever other supervisor you want. Turns arrive via a FIFO (named pipe). Messages are NUL-terminated. After each turn, a checkpoint is written. The entire protocol is pipes and files.

```sh
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

`agent-core` has no IO of its own except through the `ChatProvider` trait and the `Eval` op interpreter. It is the pure kernel. The `agent` binary is the shell around it. This boundary is intentional and load-bearing.

## Status

M1 (single-agent, sequential interpreter) is implemented. Active development:
- M2: session history + FIFO protocol
- M3: hermetic PATH, models.yaml, HydrationSource implementations
- M4: parallel interpreter (real `Par` execution)
- M5: distributed interpreter, multi-VM campaigns

## Prior Art

The design originates in `Omni/Agent/Op.hs`, a Haskell prototype that first demonstrated the free monad Op abstraction at production scale. This Rust port is a faithful translation, not a rewrite. The Haskell codebase remains the reference for Op semantics. The meta-circular Infer-emitting-Infer pattern has direct precedent in the SICP meta-circular evaluator.

## License

MIT
