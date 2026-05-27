# Rust Agent — M1 Scope: `agent` binary

## Status

M1 implementation status:

- Free monad core is live in `crates/agent-core/src/op.rs`: `OpF`, `Op`, `and_then`, `map`, and effect constructors are implemented.
- Sequential reference interpreter is live in `crates/agent-core/src/interpreter.rs`: it walks the `OpF` tree and executes provider/tool/state/trace effects.
- `Par` is represented in `OpF` and interpreted sequentially for M1 by design; future interpreters can provide concurrent semantics.
- Tool-call loop is represented as data by `agent_loop` in `crates/agent-core/src/op.rs` and executed by `run_sequential` in `crates/agent-core/src/interpreter.rs`.
- Unit tests cover monad laws, Op combinator shape, state/emit round-trips, and agent loop/tool-call behavior.

**Status:** Scoping
**Target:** private Rust agent workspace

## Goal

Port the `agent` binary to Rust as the MVP. This is a standalone agent runner:
given a prompt and provider config, run the tool-call loop until done, emit traces.

`agentd` (the daemon/scheduler) comes in M2.

## Core Design Constraint

*The free monad Op must be preserved in the Rust design.*

The Haskell system represents agent programs as `Free (OpF s)` — programs as data,
separate from interpretation. This is the core innovation enabling:
- Multiple interpreters (sequential, parallel, future: distributed)
- First-class traces/observability
- Composability (par, race, etc.)
- Testability without mocking IO

In Rust, this maps to an enum-based free monad (or equivalent via trait objects).

```rust
// Rough Rust analog
enum OpF<S, A> {
    Infer { model: Model, prompt: Prompt, next: Box<dyn FnOnce(Response) -> Op<S, A>> },
    Par { ops: Vec<Op<S, ()>>, next: Box<dyn FnOnce(Vec<()>) -> Op<S, A>> },
    Get { next: Box<dyn FnOnce(S) -> Op<S, A>> },
    Put { state: S, next: Op<S, A> },
    Tool { name: ToolName, args: Value, next: Box<dyn FnOnce(Value) -> Op<S, A>> },
    Emit { event: Event, next: Op<S, A> },
    Pure(A),
}

type Op<S, A> = Box<OpF<S, A>>;
```

The interpreter (Sequential for M1) walks this tree and executes IO.

## Architecture Docs to Read

Before coding, read:
- `/home/ben/omni/live/Omni/Agent/ARCHITECTURE.md` — Op design rationale
- `/home/ben/omni/live/Omni/Agent/README.md` — API surface and usage patterns
- `/home/ben/omni/live/Omni/Agent/DESIGN.md` — hydration/emission model (north star)
- `/home/ben/omni/live/Omni/Agent/Op.hs` — Op types (use as reference, not to copy)
- `/home/ben/omni/live/Omni/Agent/Interpreter/Sequential.hs` — sequential interpreter

## M1 Scope: `agent` binary

### Included

1. **Cargo workspace**
   - `crates/agent-core/` — Op types, interpreter, provider client
   - `crates/agent/` — CLI binary

2. **Op free monad** (`agent-core`)
   - `OpF` enum with: `Infer`, `Tool`, `Get`, `Put`, `Emit`, `Pure`
   - `Par` can be stubbed as sequential in M1 (real concurrency in M2)
   - Sequential interpreter: walks Op tree, runs IO

3. **Provider client** (OpenAI-compatible only)
   - POST `/v1/chat/completions` with tools
   - Provider config: base URL + API key (from env or config file)
   - Supported out of the box: Parasail, OpenRouter, Anthropic (via oai-compat endpoint)
   - No OAuth, no browser flows

4. **Tool dispatch**
   - Tools are Rust functions registered with the interpreter
   - Standard tools: `bash` (shell exec), `read_file`, `write_file`
   - Tool results fed back into conversation

5. **Trace/event logging**
   - JSONL event log: `InferStart`, `InferEnd`, `ToolCall`, `ToolResult`, `AgentDone`
   - Written to stderr or a log file

6. **CLI surface**
   ```
   agent [--provider <url>] [--model <name>] [--key <api-key>] "<prompt>"
   agent --config <file> "<prompt>"
   ```
   - Provider/model/key can also come from env: `AGENT_PROVIDER`, `AGENT_MODEL`, `AGENT_API_KEY`

7. **Config file** (TOML or YAML)
   ```toml
   [provider]
   url = "https://api.parasail.io/v1"
   model = "parasail-qwen3-235b-a22b-instruct-2507"
   # api_key from AGENT_API_KEY env var
   ```

### Excluded from M1

- Daemon scheduling, systemd integration, FIFO session orchestration
- OAuth providers (claude-code, openai-codex)
- Memory/hydration system (PromptIR)
- Subagent fan-out
- OCI container machinery
- `Par` / `Race` concurrency (stub as sequential)
- Persistent sessions

## Acceptance Criteria

1. `agent "list files in current directory"` runs, calls `bash` tool, returns output
2. Multi-turn tool loop works (agent can call multiple tools before finishing)
3. Works against Parasail API with `AGENT_API_KEY` set
4. Works against OpenRouter with `OPENROUTER_API_KEY`
5. JSONL trace written to `~/.local/share/agent/traces/<run-id>.jsonl`
6. `cargo test` passes

## Rust Stack

- `tokio` for async runtime
- `reqwest` for HTTP
- `serde` / `serde_json` for serialization
- `clap` for CLI
- `anyhow` for error handling
- No opinion on free monad crate — implement directly or use `free` crate

## References

- Existing Op.hs: `/home/ben/omni/live/Omni/Agent/Op.hs`
- Sequential interpreter: `/home/ben/omni/live/Omni/Agent/Interpreter/Sequential.hs`
- Provider.hs (for API shape): `/home/ben/omni/live/Omni/Agent/Provider.hs`
