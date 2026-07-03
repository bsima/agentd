# agentd Architecture

`agentd` is a Rust runtime for long-running AI agents. The main design choice is that agent programs are data, and runtimes interpret that data.

That sounds abstract. The practical payoff is simple: you can inspect, replay, test, sandbox, parallelize, or distribute the same agent program by swapping interpreters.

## The agentic loop is Infer/Eval

Traditional programs do this:

```text
eval(structured_data)
```

Modern models do this:

```text
infer(unstructured_data)
```

An agent needs both. It infers from text, history, files, and command output. Then it evaluates effects against the environment. Then it reads the result and infers again.

The loop is the agent.

`agentd` encodes that loop with a small effect algebra — `Infer`, `Eval`, `Retrieve`, `Store`, `Emit` — carried by two program representations:

| Representation | Module | Status |
|---|---|---|
| **AgentIR** — serializable block/instruction CFG with an explicit machine | `agent-core::ir`, `ir_interpreter` | Stable runtime; the CLI's only runtime |
| **Op** — free monad with Rust closure continuations | `agent-core::op`, `interpreter` | Library-level builder/test API; no CLI runtime mode |

The effect algebra, not either encoding, is the architectural constraint. Interpreters must preserve explicit `Infer`/`Eval`/`Retrieve`/`Store`/`Emit` effects regardless of how programs are authored. (The Op builder covers the `Infer`/`Eval`/`Emit`/`Par` subset; `Retrieve` and `Store` exist only in the IR. Their key-based predecessor, `Get`/`Put`, was deleted — see [docs/MEMORY.md](./docs/MEMORY.md).)

## AgentIR: programs are data, literally

AgentIR is a validated, serializable program representation:

```rust
pub struct Program {
    pub id: ProgramId,
    pub entry: BlockId,
    pub blocks: BTreeMap<BlockId, Block>,   // params, instructions, terminator
}

pub enum Instr {
    Let      { out, expr },
    Infer    { out, model, prompt, policy },  // LLM call: infer(unstructured)
    Eval     { out, request, policy },        // process call: eval(structured)
    Emit     { event },                       // trace
    Retrieve { out, query, kind, max_bytes, policy },  // ranked read from hydration sources
    Store    { out, sink, op, id, item, policy },      // write to a hydration sink
}
```

Execution is an explicit machine — `{program, block, pc, env, budgets}` — stepped by `run_ir_sequential`. Because the machine is plain data:

- programs round-trip through JSON and are validated (block refs, arity, use-before-def, shadowing) before any effect runs
- checkpoints can snapshot a machine **mid-turn** and resume without re-running completed effects
- every effect has a stable id derived from `hash(program_hash, effect_site, dynamic_path)`, so replay keys on program identity rather than incidental sequence numbers, and divergence errors name the block and instruction
- a failed effect closes with an error trace event (`InferError`/`EvalError`/`RetrieveError`/`StoreError`), so failed runs replay as the same failure

The design is specified in [docs/AGENT_IR.md](./docs/AGENT_IR.md).

## The Op layer

The original encoding is a free monad over `OpF`, with `and_then`/`map` and closure continuations:

```rust
pub enum OpF<S, A> {
    Infer { model, prompt, next },
    Eval  { command, next },
    Emit  { event, next },
    Par   { ops, next },
    Pure(A),
}
```

It proved the interpreter boundary (M1) and remains useful as an ergonomic, typed way to compose programs in Rust. But closures are not serializable, hashable, or checkpointable mid-turn — which is why it is not the stable runtime representation. The CLI runs AgentIR exclusively; the Op layer survives as a library builder and test API (`run_sequential` remains the reference interpreter for the effect algebra).

## Infer can call Infer

The important meta-circular move is that agent programs can emit every effect, including `Infer`.

So sub-agents are not a separate orchestration layer. They are just `Infer` effects emitted by another agent program — the IR agent loop exposes this to the model as an `infer` tool. The outer agent can choose the model, prompt, context, and budget for each inner call. The interpreter enforces whatever governance rules we need.

This is the SICP evaluator idea in agent form. `eval` calling `eval` collapses the interpreter/object-language boundary. `Infer` calling `Infer` collapses the agent/orchestrator boundary.

## Why Eval, not Tool

Most frameworks expose tools as named functions with JSON schemas. That is useful as an API shape, but it is the wrong primitive.

A tool eventually becomes process execution, file IO, an HTTP call, or some other environment effect. In Unix terms, the general operation is evaluation against the environment.

So `Eval` is the primitive.

Today `Eval` forks the configured shell with `-c <command>`. The default shell is `$SHELL`, falling back to `/bin/sh`. `Eval` also has interpreter-owned policy: timeout, stdout/stderr caps, cwd, and environment mode. The agent program does not care.

It also gives one sandboxing hook. You do not sandbox each tool. You sandbox the evaluator. The interpreter can wrap every `Eval` with `bwrap`, a container, a VM, a remote worker, or a hermetic PATH.

## Retrieve/Store is the hydration model

`Retrieve` and `Store` are not just state plumbing. They are the interface for context.

A context read is a query, not a key lookup; a context write is a mutation with real semantics, not a blind put:

```text
Retrieve { query, kind?, max_bytes? }
    -> [{source, kind, content, metadata}]   one hit per source, best matches first within it

Store { sink, op: create | update | delete, id?, item }
    -> sink-assigned stable id
```

`Retrieve` fans out to every registered query-capable source, optionally narrowed to one `SourceKind` (`Temporal` | `Semantic` | `Knowledge`), and returns full bodies under `max_bytes` — no second round-trip. `Store` targets one sink by registered name; the runtime — never the program — attaches provenance (run id, effect id, timestamp) to every write, and each sink declares a write policy (`Free` writes execute immediately with the trace as the audit; `RequireApproval` sinks refuse until the approval flow exists).

Both are effects in the full sense: stable effect ids, `RetrieveCall`/`RetrieveResult` and `StoreCall`/`StoreResult` trace events, and deterministic replay. A replayed `Retrieve` returns the recorded hits without touching any source; a replayed `Store` returns the recorded id without mutating the sink — replay never writes.

This gives one model for passive context injection and active recall, split by initiator:

|            | Source (read) | Sink (write) |
|------------|---------------|--------------|
| **Passive** (runtime-initiated) | prompt-assembly hydration before each turn; program-sited `Retrieve` | turn-completion persistence (checkpointing); program-sited `Store` |
| **Active** (model-initiated) | `recall` tool -> `Retrieve` | `remember` tool -> `Store` |

Passive mode is ordinary context construction. The interpreter gathers recent history, semantic matches, workspace facts, or session data before `Infer`.

Active mode is model-driven: the agent loop exposes `remember`/`recall` tools alongside `shell` and `infer`, and its tool-dispatch arm compiles them onto the same `Store`/`Retrieve` effects. The tools appear automatically whenever a memory backend is registered. If the passive window is not enough, the model can ask.

Same operations. Different initiator.

The design, including why the key-based `Get`/`Put` predecessor was deleted, is specified in [docs/MEMORY.md](./docs/MEMORY.md).

## Hydration sources and sinks

The interpreter maps passive hydration, `Retrieve`, and `Store` to `HydrationSource`/`HydrationSink` implementations registered in one `SourceRegistry`:

```rust
pub trait HydrationSource: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind;                 // Temporal | Semantic | Knowledge
    fn capabilities(&self) -> SourceCapability;   // SESSION_CONTEXT | QUERY | WORKSPACE
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult>;
}

pub trait HydrationSink: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind;                 // sinks share the source kind taxonomy
    fn write_policy(&self) -> SinkWritePolicy;    // Free | RequireApproval
    async fn store(&self, item: SinkItem) -> Result<SinkId>;
    async fn update(&self, id: &SinkId, item: SinkItem) -> Result<()>;
    async fn delete(&self, id: &SinkId) -> Result<()>;
}
```

They are deliberately separate traits (the std::io `Read`/`Write` precedent), so writability is a compile-time fact. A backend that persists implements both and can register once via `register_backend` — the file-backed memory backend (`--memory-dir`) registers this way, serving retrieval and writes from one object. The `ChatHistory` session backend also implements both halves: its sink half writes checkpoints, its source half is the recency-windowed temporal reader. The payload schema belongs to each sink, not to the trait or the IR.

`SeqConfig::passive_hydration` selects passive sources before each `Infer`. `Retrieve` dispatches to query-capable sources, optionally filtered by kind; `Store` looks up its sink by name.

Passively hydrated sections are assembled through PromptIR — labeled, sourced, budgeted sections compiled to provider messages — so every context chunk has a key, source, and hash in the trace. See [docs/PROMPT_IR.md](./docs/PROMPT_IR.md).

## Sessions are FIFOs plus checkpoints

An agent session is a long-lived process.

Turn delivery happens through stdin or a FIFO. Each turn is NUL-terminated. The agent reads one turn, runs the loop, writes trace events and checkpoints, then waits for the next turn. A failed turn is traced and reported; the session keeps reading.

A FIFO works because it is boring:

- it is a file
- any language can write to it
- NUL framing composes with standard Unix tools
- kernel backpressure blocks the writer if the agent is busy

No broker is required. No coordinator is required.

Checkpoints are written at turn completion through the `ChatHistory` sink — the passive write channel, not a program effect: it consumes no effect id, a failing sink logs and never fails the turn, and it is suppressed entirely under replay (replay never writes). A crashed agent can restart from the latest checkpoint with history intact; checkpoints with dangling tool calls are repaired on load.

## Provider neutrality

The core message type (`ChatMessage`/`ToolCall`) is provider-neutral: tool calls are `{id, name, arguments: json}`, not any provider's wire shape. Each provider adapts at its serialization edge — the OpenAI-compatible client nests `function` objects and stringifies arguments, the Anthropic client emits `tool_use` content blocks. Persisted state (checkpoints, traces) uses the neutral shape; legacy OpenAI-shaped state still deserializes.

Known limitation: `ChatMessage` is flat text-plus-tool-calls. Content blocks (e.g. provider thinking blocks) cannot round-trip through history yet.

## Interpreters define execution

The interpreter decides what each effect means. Change the interpreter and the same agent program runs under a different execution model.

| Interpreter | `Eval` behavior        | `Infer` behavior         | `Par` behavior       |
|-------------|-------------------------|--------------------------|----------------------|
| IR sequential (CLI) | fork `$SHELL -c`    | HTTP provider call       | not yet implemented (errors) |
| Op sequential (library) | fork `$SHELL -c` | HTTP provider call      | serial execution     |
| Replay      | return recorded result/failure | return recorded result/failure | serial |
| Sandboxed   | wrapped fork            | HTTP provider call       | (future)             |
| Parallel    | fork `$SHELL -c`        | HTTP provider call       | concurrent (future)  |
| Distributed | RPC to worker/sandbox   | RPC/provider pool        | distributed (future) |

`Par` is deliberately unimplemented in the IR runtime until its semantics (store isolation, join merge, failure propagation, trace ordering) are settled — see docs/AGENT_IR.md.

## Resource governance

Because `Infer` is an operation like `Eval`, the interpreter is the natural policy boundary.

Before running an emitted `Infer`, the interpreter can check budget, depth, model allowlist, or tenant quota. Before running `Eval`, it can check sandbox policy, command limits, network policy, or filesystem policy.

One governance hook. Both operations.

## Trace log

Every effect execution appends JSONL events with a run id, operation id, and — in IR mode — a stable effect id:

```json
{"event":"Custom","name":"ir_effect","data":{"effect_id":"sha256:...","kind":"Infer","site":{"block":0,"instruction_index":0}}}
{"event":"InferCall",  "op_id":3, "model":"...", "prompt_preview":"..."}
{"event":"InferResult","op_id":3, "tokens":340, "response_preview":"..."}
{"event":"EvalCall",   "op_id":4, "command":"rg TODO src/"}
{"event":"EvalError",  "op_id":4, "error":"..."}
```

Failures close their call with `InferError`/`EvalError` (or `RetrieveError`/`StoreError` for the hydration effects), so failed runs are as inspectable and replayable as successful ones. Replay mode re-runs the same program and feeds recorded results (or failures) back at matching effect ids instead of calling providers, executing shell commands, or touching sources and sinks.

Full prompts are opt-in (`--trace-full-payloads`); by default traces carry previews, keeping trace growth linear in session length. Two deliberate exceptions: `Retrieve` results are always recorded in full, because replay returns them verbatim, and `Store` records a payload preview plus a content hash, so replay can detect same-site divergence without recording the item.

## Non-goals

`agentd` is not a full-stack agent framework. It is a runtime substrate.

It does not provide YAML pipelines, a plugin marketplace, a dashboard, or a special multi-agent abstraction. It does not include a built-in sandbox. It does not try to hide Linux behind tool schemas.

The model is: agent programs emit operations; interpreters run them; Linux is the environment.

## Crate structure

```text
crates/
  agent-core/   -- effect algebra, AgentIR + machine, Op builder, interpreters,
                   hydration, PromptIR, GC, providers, tracing
  agent/        -- CLI binary, session loop, FIFO management
  agent-oauth/  -- OAuth flows for claude-code / openai-codex providers
```

`agent-core` is the kernel. `agent` is the CLI shell around it. That boundary is intentional.

## Prior art

The design comes from `Omni/Agent/Op.hs`, a Haskell prototype that proved the free monad Op abstraction at production scale. The Rust port keeps the same effect boundary while moving the stable representation from closures to serializable IR.

The meta-circular `Infer`-emitting-`Infer` pattern has direct precedent in SICP's meta-circular evaluator.
