# agent-core Architecture

`agent-core` keeps agent programs separate from the runtime that executes them.

The core representation is a free monad. Programs are data until an interpreter walks the tree and performs effects.

## Why a free monad

The free monad gives the Rust port the same boundary as the Haskell agent:

- programs can be built, transformed, inspected, tested, or traced before IO
- the same `Op` program can run under a sequential, parallel, distributed, replay, or dry-run interpreter
- effects stay visible as `OpF` variants instead of hiding in arbitrary async code
- tests can observe semantics without talking to a real model provider

The bigger lever is interpretation. Change the interpreter and the execution model changes. The agent program does not.

## OpF

`OpF<S, A>` is one layer of the program tree. Each variant is either an effect or a completed value:

- `Infer` asks a chat provider for a model response
- `Eval` executes a command string, currently through `$SHELL -c`
- `Get` reads keyed interpreter/source data such as `session:state`, `semantic:*`, or `temporal:*`
- `Put` writes keyed interpreter/source data such as `session:state` checkpoints or temporal state
- `Emit` writes a trace event
- `Par` describes child operations that can be interpreted together
- `Pure` contains a completed value with no remaining effects

Adding a capability starts by adding an `OpF` variant and teaching each interpreter how to handle it.

## Op

`Op<S, A>` wraps `Box<OpF<S, A>>`. It provides the monadic operations:

- `Op::pure(value)` creates a finished program
- `and_then` is bind: it appends the next program while preserving existing effect nodes
- `map` transforms a result by binding and returning `Pure`

Because continuations are `FnOnce` trait objects, `Op` is not structurally comparable. Tests verify monad laws by interpreting equivalent programs and comparing observed results.

## agent_loop

`agent_loop` is an example `Op` program. It does not perform IO.

It builds a tree that says:

1. infer from the model
2. if tool calls are returned, read conversation state
3. append the assistant tool-call message
4. translate each tool call into an `Eval`
5. write updated state
6. recurse until the response is final or the turn limit is reached

Runtime behavior comes from the interpreter, not from `agent_loop`.

## run_sequential

`run_sequential` is the M1 reference interpreter. It pattern matches the `OpF` tree and executes effects directly:

- `Infer` builds configured passive hydration into the prompt, then calls the configured `ChatProvider`
- `Eval` runs the command through the configured shell
- `Get` dispatches explicit keys through interpreter state, checkpoint storage, or the hydration backend
- `Put` writes explicit keys through interpreter state or checkpoint storage
- `Emit` writes to the trace logger
- `Par` intentionally runs child operations sequentially in M1
- `Pure` returns the value and current state

Passive hydration is interpreter-owned context construction. `SeqConfig::passive_hydration` selects sources assembled before each `Infer`. Agent programs do not need to emit `Get` for those sources.

Active reads remain explicit `Get(key)` operations. Examples: `Get("semantic:topic")` through the hydration backend, or `Get("session:state")` through checkpoint storage.

`Par` being sequential is deliberate for M1. Later interpreters can preserve the same `Op` program shape while changing scheduling semantics.

## Adding interpreters

A new interpreter should have the same basic shape as `run_sequential`: accept configuration, state, and `Op<S, A>`, then pattern match every `OpF` variant.

The interpreter decides how to execute each effect. It must preserve the meaning of `Pure`, `and_then`, and state transitions.

Examples:

- a parallel interpreter that executes `Par` branches concurrently
- a race interpreter that tries multiple inference providers and uses the fastest response
- a merge interpreter that asks multiple models and combines the answers
- a replay interpreter that reads prior trace events instead of calling providers
- a dry-run interpreter that validates model and command requests without executing them
- a distributed interpreter that schedules effects across workers

The free monad is the core architectural constraint. Future ports should preserve it.
