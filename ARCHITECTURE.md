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

`agentd` encodes that loop with a small effect algebra — `Infer`, `Eval`, `Get`, `Put`, `Emit`, `Par` — shared by two program representations:

| Representation | Module | Status |
|---|---|---|
| **AgentIR** — serializable block/instruction CFG with an explicit machine | `agent-core::ir`, `ir_interpreter` | Stable runtime; CLI default (`--runtime ir`) |
| **Op** — free monad with Rust closure continuations | `agent-core::op`, `interpreter` | Deprecated compatibility runtime and builder API |

The effect algebra, not either encoding, is the architectural constraint. Interpreters must preserve explicit `Infer`/`Eval`/`Get`/`Put`/`Emit`/`Par` effects regardless of how programs are authored.

## AgentIR: programs are data, literally

AgentIR is a validated, serializable program representation:

```rust
pub struct Program {
    pub id: ProgramId,
    pub entry: BlockId,
    pub blocks: BTreeMap<BlockId, Block>,   // params, instructions, terminator
}

pub enum Instr {
    Let   { out, expr },
    Infer { out, model, prompt, policy },   // LLM call: infer(unstructured)
    Eval  { out, request, policy },         // process call: eval(structured)
    Get   { out, key },                     // state/context read
    Put   { key, value },                   // state/context write
    Emit  { event },                        // trace
}
```

Execution is an explicit machine — `{program, block, pc, env, budgets}` — stepped by `run_ir_sequential`. Because the machine is plain data:

- programs round-trip through JSON and are validated (block refs, arity, use-before-def, shadowing) before any effect runs
- checkpoints can snapshot a machine **mid-turn** and resume without re-running completed effects
- every effect has a stable id derived from `hash(program_hash, effect_site, dynamic_path)`, so replay keys on program identity rather than incidental sequence numbers, and divergence errors name the block and instruction
- a failed effect closes with an `InferError`/`EvalError` trace event, so failed runs replay as the same failure

The design is specified in [docs/AGENT_IR.md](./docs/AGENT_IR.md).

## The Op layer

The original encoding is a free monad over `OpF`, with `and_then`/`map` and closure continuations:

```rust
pub enum OpF<S, A> {
    Infer { model, prompt, next },
    Eval  { command, next },
    Get   { key, next },
    Put   { key, value, next },
    Emit  { event, next },
    Par   { ops, next },
    Pure(A),
}
```

It proved the interpreter boundary (M1) and remains useful as an ergonomic, typed way to compose programs in Rust. But closures are not serializable, hashable, or checkpointable mid-turn — which is why it is not the stable runtime representation. The CLI still accepts `--runtime op` as a deprecated compatibility mode; new work targets AgentIR.

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

## Get/Put is the hydration model

`Get` and `Put` are not just state plumbing. They are the interface for context.

Every context source is a keyed read:

```text
Get("temporal:history")   -> conversation history
Get("semantic:topic")     -> vector search or other semantic recall
Get("session:state")      -> current checkpoint
Put("session:state", v)   -> write checkpoint
```

The interpreter decides how each key is backed, but the guaranteed namespaces — `session:state`, `temporal:*`, `semantic:*` — have a fixed observable contract that every runtime must satisfy. The contract, including the one deliberate divergence (unknown keys), is specified in [docs/STATE_KEYS.md](./docs/STATE_KEYS.md) and enforced by a conformance test against both runtimes.

This gives one model for passive context injection and active recall:

|            | Passive, interpreter-owned | Active, agent-emitted |
|------------|----------------------------|------------------------|
| `Get`      | inject context before turn  | query a source by key   |
| `Put`      | write checkpoints/traces    | mutate state by key     |

Passive mode is ordinary context construction. The interpreter gathers recent history, semantic matches, workspace facts, or session data before `Infer`.

Active mode is agent-driven recall. If the passive window is not enough, the program can emit `Get("semantic:prior architecture decisions")`.

Same operation. Different timing.

## Hydration sources

The interpreter maps passive hydration and active `Get` keys to `HydrationSource` implementations:

```rust
pub trait HydrationSource: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind;                 // Temporal | Semantic | Knowledge
    fn capabilities(&self) -> SourceCapability;   // SESSION_CONTEXT | QUERY | WORKSPACE
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult>;
}
```

`SeqConfig::passive_hydration` selects passive sources before each `Infer`. Active reads use the same `Get` shape: `Get("semantic:topic")` dispatches to query-capable sources.

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

Checkpoints are written after turns through `Put("session:state", ...)` and mirrored to checkpoint files by the CLI. A crashed agent can restart from the latest checkpoint with history intact; checkpoints with dangling tool calls are repaired on load.

## Provider neutrality

The core message type (`ChatMessage`/`ToolCall`) is provider-neutral: tool calls are `{id, name, arguments: json}`, not any provider's wire shape. Each provider adapts at its serialization edge — the OpenAI-compatible client nests `function` objects and stringifies arguments, the Anthropic client emits `tool_use` content blocks. Persisted state (checkpoints, traces) uses the neutral shape; legacy OpenAI-shaped state still deserializes.

Known limitation: `ChatMessage` is flat text-plus-tool-calls. Content blocks (e.g. provider thinking blocks) cannot round-trip through history yet.

## Interpreters define execution

The interpreter decides what each effect means. Change the interpreter and the same agent program runs under a different execution model.

| Interpreter | `Eval` behavior        | `Infer` behavior         | `Par` behavior       |
|-------------|-------------------------|--------------------------|----------------------|
| IR sequential (default) | fork `$SHELL -c` | HTTP provider call      | not yet implemented (errors) |
| Op sequential (compat)  | fork `$SHELL -c` | HTTP provider call      | serial execution     |
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

Failures close their call with `InferError`/`EvalError`, so failed runs are as inspectable and replayable as successful ones. Replay mode re-runs the same program and feeds recorded results (or failures) back at matching effect ids instead of calling providers or executing shell commands.

Full prompts and `Get` values are opt-in (`--trace-full-payloads`); by default traces carry previews, keeping trace growth linear in session length.

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
