# agent-core Architecture

`agent-core` keeps agent programs separate from the runtime that executes them. The core representation is a free monad: programs are ordinary data until an interpreter walks the tree and performs effects.

## Why Free Monad

The free monad gives the Rust port the same architectural boundary as the Haskell agent:

- Programs are data: a plan can be built, transformed, inspected, tested, or traced before any IO happens.
- Interpretation is separate: the same `Op` program can run in a sequential interpreter today and a parallel, distributed, replay, or dry-run interpreter later.
- Effects are explicit: model calls, tools, state, emitted events, and parallel branches are visible as `OpF` variants instead of hidden in arbitrary async code.
- Tests can observe semantics without talking to a provider by interpreting small programs with a mock or no-op backend.

## OpF

`OpF<S, A>` is one layer of the program tree. Each variant represents one effect or a completed value:

- `Infer` asks a chat provider for a model response.
- `Tool` executes a named tool with JSON arguments.
- `Get` reads keyed interpreter/source data such as `session:state`, `semantic:*`, or `temporal:*`.
- `Put` writes keyed interpreter/source data such as `session:state` checkpoints or temporal state.
- `Emit` writes a trace event.
- `Par` describes multiple child operations that can be interpreted together.
- `Pure` contains a completed value with no remaining effects.

Adding a new effect starts by adding a new `OpF` variant and teaching each interpreter how to handle it.

## Op

`Op<S, A>` is a newtype wrapper around `Box<OpF<S, A>>`. It provides the monadic operations:

- `Op::pure(value)` creates a finished program.
- `and_then` is bind: it appends the next program while preserving existing effect nodes.
- `map` transforms a result by binding and returning `Pure`.

Because continuations are `FnOnce` trait objects, `Op` is not structurally comparable. Tests verify the monad laws by interpreting equivalent programs and comparing observed results.

## agent_loop

`agent_loop` is an example `Op` program. It does not perform IO itself. It builds a tree that says:

1. infer from the model,
2. if tool calls are returned, read conversation state,
3. append the assistant tool-call message,
4. execute each tool call,
5. write updated state,
6. recurse until the response is final or the turn limit is reached.

The loop is therefore reusable across interpreters: all runtime behavior comes from the interpreter, not from `agent_loop`.

## run_sequential

`run_sequential` is the M1 reference interpreter. It pattern matches the `OpF` tree and executes effects directly:

- `Infer` builds configured passive hydration into the prompt, then calls the configured `ChatProvider`.
- `Tool` dispatches through the configured tool map.
- `Get` dispatches explicit keys through interpreter state, checkpoint storage, or the configured hydration backend.
- `Put` writes explicit keys through interpreter state or checkpoint storage.
- `Emit` writes to the trace logger.
- `Par` intentionally runs child operations sequentially in M1.
- `Pure` returns the value and current state.

Passive hydration is interpreter-owned context construction. `SeqConfig::passive_hydration` selects the passive sources that are assembled before each `Infer`; agent programs do not need to emit a `Get` for those sources. Active reads remain explicit `Get(key)` operations, for example `Get("semantic:topic")` through the hydration backend or `Get("session:state")` through checkpoint storage.

`Par` being sequential is deliberate for M1. Later milestones can add async or distributed interpreters that preserve the same `Op` program shape while changing scheduling semantics.

## Adding Interpreters

A new interpreter should expose a `run_*` function with the same basic shape as `run_sequential`: accept configuration, state, and `Op<S, A>`, then pattern match every `OpF` variant. The interpreter decides how to execute each effect, but it must preserve the meaning of `Pure`, `and_then`, and state transitions.

Examples of future interpreters:

- a parallel interpreter that executes `Par` branches concurrently,
- a race interpreter that tries multiple inference providers and uses the fastest response,
- a marge iterpreter that uses multiple LLMs and merges them together into one response,
- a replay interpreter that reads prior trace events instead of calling providers,
- a dry-run interpreter that validates tool/model requests without executing them,
- a distributed interpreter that schedules effects across workers.

The free monad is the core architectural constraint. All future ports must preserve it.
