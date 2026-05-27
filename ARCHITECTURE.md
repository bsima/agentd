# agentd Architecture

agentd is a Rust runtime for long-running AI agents. It implements several design ideas that are non-obvious in the current landscape of agent frameworks, so this document explains the reasoning behind the key choices.

## Core Idea: Infer/Eval as the Agentic Loop

Most agent frameworks treat LLM calls as the fundamental unit and bolt tool execution on as an afterthought. agentd treats the agent loop as an _interpreter_ over a _program_: the model produces a plan (Infer), the runtime evaluates it (Eval), and the loop recurs.

This is not just a metaphor. The `Op` free monad makes it structural: `Infer` and `Tool` are both `OpF` variants interpreted by the same machinery. The agent is a program; the runtime is an evaluator. This boundary is what makes the system testable, composable, and interpreter-portable.

## Free Monad: Programs as Data

The central architectural constraint is the free monad over `OpF`.

```rust
pub enum OpF<S, A> {
    Infer  { model, prompt, next },
    Tool   { name, args, next },
    Get    { next },
    Put    { state, next },
    Emit   { event, next },
    Par    { ops, next },
    Pure(A),
}
```

An `Op<S, A>` is a program that carries state `S` and produces a value `A`. It does not execute anything. It is data that can be:

- **Interpreted** — run by any interpreter (sequential, parallel, replay, dry-run)
- **Inspected** — tools, model calls, and state transitions are all visible as enum variants
- **Tested** — mock interpreters can validate program semantics without network calls
- **Transformed** — programs can be composed with `and_then`/`map` before any IO happens

Every new capability starts by adding a variant to `OpF` and teaching interpreters how to handle it. This keeps effects explicit and prevents the "async spaghetti" failure mode of most agent codebases.

### Why not just async functions?

Because `async fn` hides effects. Once you have `async fn run_agent(...)`, you cannot inspect what it will do without running it. You cannot substitute a mock provider without thread-local trickery. You cannot replay a run from a trace without re-executing.

The free monad preserves effect structure across the monad boundary. Interpretation is deferred.

### Reference: Haskell origin

This architecture ports `Omni/Agent/Op.hs` from the Haskell prototype. The Rust implementation preserves the same boundary: programs built with `Op` constructors, interpreted by `run_sequential` (or future interpreters). The Haskell version was the first proof that this structure worked at production scale.

## Hydration Sources: Typed Context Injection

Context injection in most systems is a string append. agentd models it as a typed registry of `HydrationSource` implementations:

```rust
pub trait HydrationSource: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind;     // Temporal | Semantic | Knowledge
    fn capabilities(&self) -> SourceCapability;
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult>;
}
```

Sources are registered at startup. Before the first infer, `SourceRegistry::retrieve_all` is called to assemble the system prompt. Sources can be:

- `Temporal` — time-indexed history, recent events
- `Semantic` — embedding-retrieved memory, RAG
- `Knowledge` — static files, workspace context, configs

The current implementation ships `LocalFileSource` (reads a directory into context). The trait is the extension point: a future memory source, vector DB, or hydration-from-checkpoint plugs in without touching the interpreter.

**Why this matters:** Most agent systems have no principled model of _what goes into the context window and why_. Hydration sources make the provenance of every context chunk traceable and swappable.

## Session Model: FIFO + Checkpoints

agentd agents run as long-lived processes. Turn delivery is via a FIFO (named pipe). Each turn is NUL-terminated on the pipe. The agent reads a turn, runs the `agent_loop`, emits a JSONL trace event, writes a checkpoint, and waits for the next turn.

```
agentd send <agent> "message"
  → writes NUL-terminated message to FIFO
  → agent reads, runs Op loop, writes checkpoint JSON
  → trace: ~/.local/share/agent/traces/<run-id>.jsonl
  → checkpoint: <checkpoint-dir>/<run-id>-<seq>.json
```

Checkpoints contain the full conversation history, model, provider URL, and sequence number. An interrupted agent can resume from checkpoint with `--resume`.

This is different from "memory" — checkpoints are structural conversation state. Memory (semantic retrieval) is a hydration source.

## Interpreter Extensibility

The M1 interpreter is `run_sequential`. It pattern-matches `OpF` and executes effects directly. `Par` is run sequentially in M1.

The design allows future interpreters to be added without changing `Op` programs:

| Interpreter | Par behavior | Use case |
|---|---|---|
| `run_sequential` (M1) | sequential | single-agent, debugging |
| `run_parallel` (planned) | concurrent tasks | subagent fan-out |
| `run_replay` (planned) | reads trace events | deterministic replay |
| `run_dryrun` (planned) | validates, no IO | pre-flight checks |
| `run_distributed` (planned) | remote workers | multi-VM campaigns |

The free monad is the core architectural constraint. All future interpreters must preserve the meaning of `Pure`, `and_then`, and state transitions.

## Trace / Event Log

Every interpreter emits structured JSONL events:

```
InferStart { run_id, model, timestamp }
InferEnd   { run_id, tokens, timestamp }
ToolCall   { run_id, name, args, timestamp }
ToolResult { run_id, name, result, timestamp }
AgentDone  { run_id, timestamp }
```

Traces are written to `~/.local/share/agent/traces/<run-id>.jsonl`. They are the basis for replay, cost tracking, and debugging.

## Provider Abstraction

`ChatProvider` is a trait over the OpenAI-compatible `/v1/chat/completions` endpoint. Any provider that implements this interface works: Parasail, OpenRouter, Anthropic (via compat endpoint), local Ollama. No per-provider SDK dependencies.

```rust
#[async_trait]
pub trait ChatProvider: Send + Sync {
    async fn chat(&self, model: &Model, tools: &[ToolSpec], messages: &[ChatMessage]) -> Result<Response>;
}
```

## Design Decisions and Non-Goals

**Non-goal: framework.** agentd is a runtime, not a framework. It does not have a plugin marketplace, a web dashboard, or a YAML config language for "agent pipelines." It has a free monad, an interpreter, and a FIFO.

**Non-goal: LLM-native orchestration.** Approaches that ask the LLM to decide when to spawn subagents or pick tools dynamically are not the design here. The _program_ is the agent; the LLM is one effect within it.

**Non-goal: Python.** The Rust port exists because the system needs to be deployable as a single binary with predictable resource usage. Haskell worked for prototyping; Rust works for production.

## Crate Structure

```
crates/
  agent-core/     — Op, OpF, interpreter, hydration, provider, trace
  agent/          — CLI binary, session loop, standard tools
  agent-oauth/    — optional OAuth flow for claude-code / openai-codex
```

`agent-core` has no IO of its own except through `ChatProvider` and `Tool` traits. It is the pure kernel. `agent` is the executable shell around it.

## Prior Art and Relationship to Haskell Prototype

The Haskell prototype (`Omni/Agent/` in the omnirepo) was the first implementation of this design. The free monad Op abstraction, hydration sources, and sequential interpreter all originate there. The Rust port is a faithful translation of the core ideas, not a rewrite from scratch.

The Haskell codebase remains the reference for Op semantics. When in doubt about the intended behavior of a variant, check `Omni/Agent/Op.hs` and `Omni/Agent/Interpreter/Sequential.hs`.
