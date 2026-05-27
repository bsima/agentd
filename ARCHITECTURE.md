# agentd Architecture

agentd is a Rust runtime for long-running AI agents. It implements several design ideas that are non-obvious in the current landscape of agent frameworks, so this document explains the reasoning behind the key choices.

## Core Idea: The Agentic Loop as Infer/Eval

Traditional computing is `eval(structured_data)`. A program transforms well-typed inputs to outputs deterministically. Modern ML is `infer(unstructured_data)`. A model produces plausible outputs from freeform inputs probabilistically.

The agentic runtime is the marriage of both: a loop where the model infers over its environment (unstructured text, eval results, history), then evaluates effects against that environment (programs, state writes, process execution), then infers again. Neither half is sufficient alone. The loop is the agent.

agentd makes this structural with a free monad over `OpF`. `Infer` and `Eval` are not different kinds of things — they are both `OpF` variants interpreted by the same machinery. The model call and the process execution are both effects in the same program. This boundary is what makes the system testable, composable, and interpreter-portable.

## Free Monad: Programs as Data

The central architectural constraint is the free monad over `OpF`.

```rust
pub enum OpF<S, A> {
    Infer { model, prompt, next },  // LLM call: infer(unstructured)
    Eval  { command, next },        // process call: eval(structured) — currently $SHELL -c
    Get   { key, next },            // state/context read (passive or active)
    Put   { key, value, next },     // state/context write (passive or active)
    Emit  { event, next },          // trace
    Par   { ops, next },            // parallel effects
    Pure(A),
}
```

An `Op<S, A>` is a program that carries state `S` and produces a value `A`. It does not execute anything. It is data that can be:

- **Interpreted** — run by any interpreter (sequential, parallel, replay, dry-run)
- **Inspected** — eval calls, model calls, and state transitions are all visible as enum variants
- **Tested** — mock interpreters can validate program semantics without network calls
- **Transformed** — programs can be composed with `and_then`/`map` before any IO happens

Every new capability starts by adding a variant to `OpF` and teaching interpreters how to handle it. This keeps effects explicit and prevents the "async spaghetti" failure mode of most agent codebases.

### All Ops Are Available to the Agent

This is the meta-circular insight: the agent program is not a black box that emits tool calls — it is a program in the same `Op` language that the interpreter runs. All variants of `OpF` are available to an agent program, including `Infer`.

An agent that can emit `Infer` ops can construct sub-agents: different models, different prompts, different context windows. Multi-agent orchestration is not a special layer built on top — it is just an `Op` program that contains multiple `Infer` calls. The outer agent *is* the orchestrator. This is the same insight as the SICP meta-circular evaluator: `eval` calling `eval` collapses the interpreter/agent boundary.

### Why Eval, not Tool?

Most agent frameworks have a `Tool` abstraction: a named function with a JSON schema, registered in a registry, called by name. This is an unnecessary indirection. Every tool is ultimately a process invocation. Every process invocation is `eval(structured_data)`. Therefore `Eval` _is_ the Op.

This collapses `read_file`, `write_file`, `web_search`, `recall`, and every other "tool" into `Eval { command: "cat file.txt" }`, `Eval { command: "rg ..." }`, etc. The agent uses Unix programs directly. There is no tool registry to maintain, no JSON schema to write, and no special cases in the interpreter.

The current implementation of `Eval` forks `$SHELL -c <command>`, making the shell itself parameterizable. `SHELL=bash`, `SHELL=nushell`, or eventually `SHELL=intent` — the agent program is unchanged. Sandboxing is also free: every `Eval` op goes through the same interpreter branch, which can inject `bwrap`, `nix develop`, or a remote executor before forking. You don't sandbox "tools" — you sandbox the evaluator. That's how Unix has always worked.

### Why not just async functions?

Because `async fn` hides effects. Once you have `async fn run_agent(...)`, you cannot inspect what it will do without running it. You cannot substitute a mock provider without thread-local trickery. You cannot replay a run from a trace without re-executing.

The free monad preserves effect structure across the monad boundary. Interpretation is deferred.

### Reference: Haskell origin

This architecture ports `Omni/Agent/Op.hs` from the Haskell prototype. The Rust implementation preserves the same boundary: programs built with `Op` constructors, interpreted by `run_sequential` (or future interpreters). The Haskell version was the first proof that this structure worked at production scale.

## Get/Put: The Unified Hydration Interface

`Get` and `Put` are not just state threading plumbing — they are the unified interface for all hydration sources.

Every context source (temporal history, semantic recall, workspace files, session checkpoints) is accessed through `Get { key }`. The interpreter decides what a given key means: `Get("temporal-passive")` might inject recent chat history; `Get("semantic:cooking recipes")` might trigger a vector search; `Get("session:checkpoint")` reads the last checkpoint. Same op, different interpreter.

```
Get("temporal-passive")   →  inject recent chat history
Get("semantic:topic")     →  vector similarity search
Get("session:state")      →  read current session checkpoint
Put("session:state", v)   →  write checkpoint
Put("trace:event", e)     →  append to event log
```

This collapses the hydration source registry into the same `Get`/`Put` ops that thread state through the interpreter. The 2×2 passive/active distinction maps directly:

|            | **Passive** (interpreter auto-emits) | **Active** (agent-emitted)              |
|------------|---------------------------------------|-----------------------------------------|
| **Get**    | runtime injects context before turn   | agent emits `Get` to query any source   |
| **Put**    | interpreter writes checkpoints        | agent emits `Put` to mutate state       |

**Passive mode**: the interpreter emits `Get` automatically before each `Infer` — assembling recent history, semantic matches, workspace files — without the agent program ever seeing these ops. This is traditional RAG and context injection.

**Active mode**: the agent itself emits `Get` when the passive window isn't enough. An agent investigating a complex problem can emit `Get("temporal:3 weeks ago")` or `Get("semantic:prior architecture decisions")` as part of its program. This is the same mechanism, just agent-driven.

The hydration source taxonomy is thus not a separate registry — it's a naming convention over `Get`/`Put` keys:

|                | **Passive**                        | **Active**                             |
|----------------|------------------------------------|----------------------------------------|
| **Temporal**   | recent chat history, recent events | `Get("temporal:search query")`         |
| **Semantic**   | similarity RAG, static workspace   | `Get("semantic:topic")`                |

Sources are still registered at startup as interpreters for specific key prefixes. But the agent's interface to them is uniform: `Get` and `Put`.

## Hydration Sources: Implementation

The interpreter maps `Get`/`Put` keys to `HydrationSource` implementations:

```rust
pub trait HydrationSource: Send + Sync {
    fn name(&self) -> &str;
    fn key_prefix(&self) -> &str;     // e.g. "temporal", "semantic", "session"
    fn mode(&self) -> SourceMode;     // Passive | Active
    async fn get(&self, params: SourceParams) -> Result<SourceResult>;
    async fn put(&self, key: &str, value: Value) -> Result<()>;
}
```

Before the first `Infer` each turn, `SourceRegistry::run_passive` assembles context by emitting `Get` ops for all passive sources. Active sources are available to the agent program on demand.

**Why this matters:** Most agent systems have no principled model of _what goes into the context window and why_. The `Get`/`Put` model makes the provenance of every context chunk traceable: every chunk came from a `Get` op with a specific key and a specific source implementation. The interpreter that handled it is logged. The result is auditable.

## Session Model: FIFO + Checkpoints

agentd agents run as long-lived processes. Turn delivery is via a FIFO (named pipe). Each turn is NUL-terminated on the pipe. The agent reads a turn, runs the `agent_loop`, writes a JSONL event, and loops. The session is the process; the FIFO is the IPC mechanism.

Checkpoints are written after each turn via a passive `Put("session:state", ...)`. A crashed agent restarts from the last checkpoint with full history intact. No broker, no coordinator, no special protocol.

### Why FIFO?

- A FIFO is a file. If you can write to a file, you can drive an agent.
- NUL-terminated messages compose with standard Unix tools (`printf '\0'`, `xargs -0`).
- The protocol is implementation-independent: any process can send turns regardless of language or runtime.
- Backpressure is free: the kernel blocks the writer if the agent is busy.

## Interpreter Extensibility

The interpreter is the piece that decides what each `OpF` variant _means_. Changing the interpreter changes the execution model without touching any agent programs:

| Interpreter     | `Eval` behavior              | `Infer` behavior            | `Par` behavior        |
|-----------------|------------------------------|-----------------------------|-----------------------|
| Sequential      | fork `$SHELL -c`             | HTTP to provider            | serial execution      |
| Sandboxed       | `bwrap`-wrapped fork         | HTTP to provider            | serial execution      |
| Parallel        | fork `$SHELL -c`             | HTTP to provider            | concurrent futures    |
| Replay          | return from trace log        | return from trace log       | serial execution      |
| Distributed     | RPC to worker node           | RPC to inference cluster    | distributed dispatch  |
| Dry-run         | log intent, no-op            | return mock response        | serial execution      |

This is the free monad payoff: the agent program is written once. The interpreter upgrades independently. An agent written for the sequential interpreter runs unmodified on the distributed interpreter — the only change is which interpreter is instantiated.

### Resource Governance

When `Infer` is emittable by agent programs (for sub-agent construction), the interpreter is the natural governance boundary. Before forking a sub-agent `Infer`, the interpreter can check a budget, enforce a depth limit, or route to a cheaper model. The same hook that governs `Eval` sandboxing governs `Infer` sub-agent spawning. One mechanism, both ops.

## Trace / Event Log

Every op execution appends to a JSONL event log:

```json
{"ts": "...", "turn": 3, "op": "EvalCall",   "command": "rg TODO src/"}
{"ts": "...", "turn": 3, "op": "EvalResult",  "exit": 0, "stdout": "..."}
{"ts": "...", "turn": 3, "op": "InferCall",   "model": "...", "tokens": 1200}
{"ts": "...", "turn": 3, "op": "InferResult", "tokens": 340}
{"ts": "...", "turn": 3, "op": "GetCall",     "key": "semantic:prior decisions"}
{"ts": "...", "turn": 3, "op": "GetResult",   "chunks": 3}
{"ts": "...", "turn": 3, "op": "PutCall",     "key": "session:state"}
```

The log is the source of truth for replay interpreters and for debugging. A run that misbehaved can be replayed by feeding its log back through a replay interpreter, with `Eval` and `Infer` returning logged results instead of executing.

## Design Non-Goals

- **Not a framework.** agentd provides a runtime substrate, not a scaffolding system.
- **Not LLM-native orchestration.** Orchestration is `Op` programs, not YAML pipelines.
- **Not Python.** The Rust implementation is the reference.
- **Tools are programs.** There is no `Tool` abstraction. The agent uses Unix programs via `Eval`.
- **No built-in sandbox.** Bring your own. Every `Eval` op goes through the interpreter; inject isolation there.
- **No special multi-agent layer.** Sub-agents are agent programs that emit `Infer` ops. The meta-circular structure is sufficient.

## Crate Structure

```
crates/
  agent-core/   — Op, OpF, interpreter, hydration, provider traits (pure kernel, minimal IO)
  agent/        — CLI binary, session loop, FIFO management
  agent-oauth/  — OAuth flows for claude-code / openai-codex providers
```

`agent-core` has no IO of its own except through the `ChatProvider` trait and the `Eval` op interpreter. It is the pure kernel. The `agent` binary is the shell around it. This boundary is intentional and load-bearing.

## Prior Art

The design originates in `Omni/Agent/Op.hs`, a Haskell prototype that first demonstrated the free monad Op abstraction at production scale. This Rust port is a faithful translation, not a rewrite. The Haskell codebase remains the reference for Op semantics. The meta-circular `Infer`-emitting-`Infer` pattern has direct precedent in SICP's meta-circular evaluator — an `eval` that can call `eval` collapses the interpreter/object language boundary in a structurally identical way.
