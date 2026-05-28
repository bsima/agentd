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

`agentd` encodes that loop with a free monad over `OpF`. `Infer` and `Eval` are not two unrelated subsystems. They are variants in the same operation language, interpreted by the same runtime.

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

An `Op<S, A>` carries state `S` and eventually produces `A`. It does not execute anything by itself.

## Why a free monad

The free monad preserves the structure of effects.

That is the bigger lever. If the agent is just an `async fn`, you cannot know what it will do without running it. You cannot replay it cleanly. You cannot swap in a dry-run interpreter without plumbing mock objects through everything.

With `Op`, the program is data at the interpreter boundary. It can be:

- interpreted by a sequential, parallel, replay, dry-run, sandboxed, or distributed runtime
- inspected before any IO happens
- tested against a mock provider or no-op evaluator
- transformed with `and_then` and `map`

The Rust DSL still uses closure continuations, so this is not a fully serializable AST. Replay works by re-running the same program and feeding recorded operation results back at matching op IDs.

Adding a capability means adding an `OpF` variant and teaching interpreters how to handle it. Effects stay explicit.

## Infer can call Infer

The important meta-circular move is that agent programs can emit every `OpF` variant, including `Infer`.

So sub-agents are not a separate orchestration layer. They are just `Infer` calls emitted by another agent program. The outer agent can choose the model, prompt, context, and budget for each inner call. The interpreter enforces whatever governance rules we need.

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
Get("temporal-passive")   -> recent chat history
Get("semantic:topic")     -> vector search or other semantic recall
Get("session:state")      -> current checkpoint
Put("session:state", v)   -> write checkpoint
Put("trace:event", e)     -> append event
```

The interpreter decides what each key means.

This gives one model for passive context injection and active recall.

|            | Passive, interpreter-owned | Active, agent-emitted |
|------------|----------------------------|------------------------|
| `Get`      | inject context before turn  | query a source by key   |
| `Put`      | write checkpoints/traces    | mutate state by key     |

Passive mode is ordinary context construction. The interpreter gathers recent history, semantic matches, workspace facts, or session data before `Infer`.

Active mode is agent-driven recall. If the passive window is not enough, the program can emit `Get("temporal:3 weeks ago")` or `Get("semantic:prior architecture decisions")`.

Same operation. Different timing.

The source taxonomy is just a naming convention over keys:

|                | Passive                  | Active |
|----------------|--------------------------|--------|
| Temporal       | recent events/history     | `Get("temporal:query")` |
| Semantic       | similarity/workspace RAG  | `Get("semantic:topic")` |

Sources are still registered at startup. The point is that the agent's interface stays uniform.

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

`SeqConfig::passive_hydration` selects passive sources before each `Infer`. Active reads use the same `Get` shape. Today `Get("semantic:topic")` dispatches to query-capable sources, `Get("session:state")` reads checkpoint JSON, and `Get("temporal:...")` reads interpreter state.

The accurate model here is provenance. Every context chunk has a key and source. The interpreter can decide whether that source is passive, active, local, remote, replayed, or sandboxed.

## Sessions are FIFOs plus checkpoints

An agent session is a long-lived process.

Turn delivery happens through stdin or a FIFO. Each turn is NUL-terminated. The agent reads one turn, runs the loop, writes trace events and checkpoints, then waits for the next turn.

A FIFO works because it is boring:

- it is a file
- any language can write to it
- NUL framing composes with standard Unix tools
- kernel backpressure blocks the writer if the agent is busy

No broker is required. No coordinator is required.

Checkpoints are written after turns through `Put("session:state", ...)` and mirrored to checkpoint files by the CLI. A crashed agent can restart from the latest checkpoint with history intact.

## Interpreters define execution

The interpreter decides what each `OpF` variant means. Change the interpreter and the same agent program runs under a different execution model.

| Interpreter | `Eval` behavior        | `Infer` behavior         | `Par` behavior       |
|-------------|-------------------------|--------------------------|----------------------|
| Sequential  | fork `$SHELL -c`        | HTTP provider call       | serial execution     |
| Sandboxed   | wrapped fork            | HTTP provider call       | serial execution     |
| Parallel    | fork `$SHELL -c`        | HTTP provider call       | concurrent futures   |
| Replay      | return trace result     | return trace result      | serial execution     |
| Distributed | RPC to worker/sandbox   | RPC/provider pool        | distributed dispatch |
| Dry-run     | log intent, no-op       | mock response            | serial execution     |

That is the point of the free monad. The agent program is written once. The runtime changes independently.

## Resource governance

Because `Infer` is an operation like `Eval`, the interpreter is the natural policy boundary.

Before running an emitted `Infer`, the interpreter can check budget, depth, model allowlist, or tenant quota. Before running `Eval`, it can check sandbox policy, command limits, network policy, or filesystem policy.

One governance hook. Both operations.

## Trace log

Every op execution appends JSONL events with a run id and operation id:

```json
{"event":"EvalCall", "run_id":"...", "op_id":2, "command":"rg TODO src/"}
{"event":"EvalResult", "run_id":"...", "op_id":2, "result":{"status":0,"stdout":"..."}}
{"event":"InferCall", "run_id":"...", "op_id":3, "model":"...", "prompt_preview":"..."}
{"event":"InferResult", "run_id":"...", "op_id":3, "tokens":340, "response_preview":"..."}
{"event":"GetCall", "run_id":"...", "op_id":4, "key":"semantic:prior decisions"}
{"event":"GetResult", "run_id":"...", "op_id":4, "source_count":3}
{"event":"PutCall", "run_id":"...", "op_id":5, "key":"session:state"}
```

The log is for debugging and replay. Replay mode re-runs the same program and feeds logged `Eval` and `Infer` results back at matching op IDs instead of calling providers or executing shell commands.

## Non-goals

`agentd` is not a full-stack agent framework. It is a runtime substrate.

It does not provide YAML pipelines, a plugin marketplace, a dashboard, or a special multi-agent abstraction. It does not include a built-in sandbox. It does not try to hide Linux behind tool schemas.

The model is: agent programs emit operations; interpreters run them; Linux is the environment.

## Crate structure

```text
crates/
  agent-core/   -- Op, OpF, interpreter, hydration, provider traits
  agent/        -- CLI binary, session loop, FIFO management
  agent-oauth/  -- OAuth flows for claude-code / openai-codex providers
```

`agent-core` is the kernel. `agent` is the CLI shell around it. That boundary is intentional.

## Prior art

The design comes from `Omni/Agent/Op.hs`, a Haskell prototype that proved the free monad Op abstraction at production scale. This Rust port preserves the same boundary: programs built with `Op` constructors, interpreted by `run_sequential` or future interpreters.

The meta-circular `Infer`-emitting-`Infer` pattern has direct precedent in SICP's meta-circular evaluator.
