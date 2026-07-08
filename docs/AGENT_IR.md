# AgentIR design

Status: **implemented; the CLI's only runtime.** This document
is the design rationale; where it says "should", check the table below for
whether reality has caught up.

| Design item | Status |
|---|---|
| Serializable `Program`/`Block`/`Instr`/`Terminator`/`Expr` | Implemented (`agent-core::ir`) |
| Validation before execution | Implemented (`validate_program`) |
| Explicit machine, mid-turn checkpoints, step limit | Implemented (`run_ir_steps`); block terminators count toward the step limit |
| Stable effect IDs + replay + divergence locations | Implemented (`IrReplayTrace`) |
| Failure semantics: error events with stable IDs | Implemented (`InferError`/`EvalError` trace events; replay reproduces failures) |
| `agent_loop` ported with feature parity | Implemented (`agent_loop_ir`, including stalled-turn nudge) |
| CLI switched to AgentIR | Implemented; the CLI is IR-only (`--runtime` removed; the Op layer remains a library builder/test API) |
| In-memory **STM** store with transactional Get/Put | Removed (t-1182): Get/Put deleted in favor of the Retrieve/Store hydration effects; `InMemoryStore` now only backs instruction-limit checkpoints. See docs/MEMORY.md |
| Normalization pass for canonical hashing | Implemented (`ir_normalize`): programs normalize to strict SSA (existing params/args preserved, implicit dominator-scoped uses become params) and `program_hash` hashes the canonical form, so alpha-equivalent programs share identity |
| `Par` semantics | Not implemented — the IR runtime rejects `Par` until the open questions below are settled |

AgentIR is the serializable core representation for `agentd` programs.

The current `Op` free monad is useful for M1, but it is not the final runtime representation because continuations are Rust closures:

```rust
next: Box<dyn FnOnce(Response) -> Op<S, A> + Send>
```

That makes the next step executable but not serializable, hashable, inspectable, checkpointable, or portable across interpreters. AgentIR replaces closure continuations with an explicit AST/CFG and an explicit machine state.

## Goal

The stable runtime should be:

```text
surface API / CLI loop
        ↓
serializable AgentIR AST/CFG
        ↓
validated / normalized
        ↓
small-step interpreter
        ↓
effect handlers: Infer, Eval, Retrieve, Store, Emit, Par
```

`Infer` is the novel operation. The interpreter/compiler machinery should otherwise use boring, proven techniques: explicit control flow, variable binding, effect handlers, stable IDs, replay logs, and serializable machine snapshots.

The important boundary is not the free monad. It is the effect algebra. AgentIR should keep agent effects explicit while making control flow inspectable.

## Core shape

AgentIR should be a serializable program representation. A likely shape is block-based ANF/SSA-like IR:

```rust
pub struct Program {
    pub id: ProgramId,
    pub entry: BlockId,
    pub blocks: BTreeMap<BlockId, Block>,
}

pub struct Block {
    pub params: Vec<Var>,
    pub instructions: Vec<Instr>,
    pub terminator: Terminator,
}

pub enum Instr {
    Let { out: Var, expr: Expr },
    Infer { out: Var, model: Expr, prompt: PromptRef, policy: InferPolicy },
    Eval { out: Var, request: EvalRequest, policy: EvalPolicy },
    Retrieve { out: Var, query: Expr, kind: Option<SourceKind>, max_bytes: Option<usize> },
    Store { out: Var, sink: Expr, op: StoreOp, id: Option<Expr>, item: Expr, policy: StorePolicy },
    Emit { event: Expr },
}

pub enum Terminator {
    Goto { block: BlockId, args: Vec<Expr> },
    If { cond: Expr, then_block: BlockId, else_block: BlockId },
    Match { value: Expr, arms: Vec<MatchArm> },
    Return { value: Expr },
    Par { branches: Vec<BlockId>, join: BlockId },
}
```

The first version uses an environment-carrying CFG rather than strict SSA block scope. `Goto` binds target params, but the runtime environment is carried across block transitions so loop builders can keep durable locals without threading every variable through every block. This is simpler for the first interpreter. A later normalization pass can tighten this into stricter SSA if that becomes valuable.

The first version can use `serde_json::Value` for runtime values. A richer type system can come later.

## Validation and normalization

AgentIR should be validated before execution. A malformed program should fail before any `Infer`, `Eval`, `Retrieve`, `Store`, or `Emit` occurs.

At minimum validation should check:

- the entry block exists
- every referenced block exists
- `Goto` argument count matches target block params
- branch and match targets exist
- variables are defined before use
- variables are scoped to block params and prior instruction outputs
- instruction outputs have an explicit shadowing rule, preferably no shadowing within a block
- terminators only occur at block end
- effect inputs have the expected runtime shape where statically knowable
- `PromptRef::Var` values are expected to decode as prompts

Normalization should produce the canonical form used for hashing, trace locations, replay, and inspection. Equivalent programs should not get different identities because of incidental map order or formatting.

## Abstract machine

The interpreter should run an explicit machine, not Rust continuations:

```rust
pub struct Machine {
    pub program: Program,
    pub block: BlockId,
    pub pc: usize,
    pub env: BTreeMap<Var, Value>,
    pub continuation_stack: Vec<Frame>,
    pub budgets: Budgets,
    pub trace: TraceHandle,
}
```

`env` is local execution state: block params, instruction outputs, and temporary values. It is checkpointed because it is needed to resume execution, but it is not the durable session store.

This gives real checkpoints. A checkpoint can serialize the machine in the middle of a turn, not only the chat history between turns.

## State and storage model

AgentIR has local machine state and interpreter-owned durable state.

```text
env   = local execution state inside the machine
hydration backends (sources + sinks) accessed through Retrieve / Store
trace = append-only execution history
```

The store backend is not part of AgentIR. The same program should be able to run with an in-memory STM store, SQLite, a distributed KV store, a replay store, or a read-only inspection store.

`Retrieve` reads from registered hydration sources; `Store` writes to registered hydration sinks (docs/MEMORY.md). The runtime attaches provenance and decides persistence, checkpoint cadence, and replay semantics (replay never mutates a sink).

A serialized `Machine` checkpoint contains enough local execution state to resume the program. It may also contain or reference a store snapshot, depending on the backend.

## Effect IDs and replay

AgentIR uses stable effect IDs derived from program identity and dynamic execution path:

```text
effect_id = sha256(program_hash, effect_kind, effect_site, dynamic_path)

effect_site  = block_id + instruction_index
dynamic_path = { path, transitions, visit }
```

The dynamic path is explicit enough to distinguish repeated visits to the same effect site *and* which control-flow route led there, while staying O(1) per trace line (ids ride on every effect call event):

- **`path`** is the machine's rolling control-flow digest at the moment the effect executed: a sha256 chain the interpreter folds at every terminator transition over `(from_block, arm, to_block)`. The arm records which way the terminator went — `0` for `Goto` and the `If` then-branch, `1` for the else-branch, the arm index for `Match` (its default arm is `arms.len()`). The entry block, before any transition, has the empty path. Because the digest chains, a then-visit and an else-visit to the same downstream site differ even after the paths rejoin, and every trip around a loop back-edge produces a fresh digest — full path sensitivity without storing the path.
- **`transitions`** is how many transitions were folded into `path`: a human-readable depth for divergence errors, where the digest itself is opaque.
- **`visit`** is the per-site execution ordinal (0-based). Within one machine run it is redundant with `path`; it exists because per-site visit counts are carried across session turns (each turn runs a fresh machine whose path restarts at the root), so turn N's entry effect is `(path = root, visit = N-1)` — distinguishable from turn 1's and computable without simulating the machine (`agent ir-effect --visit N-1`, `DynamicPath::at_entry`). It also names loop iterations legibly in errors.

**Par (planned).** The scheme extends to parallel branches without new machinery: branch `b` of a `Par` at block `P` forks the parent path by folding `(P, arm = b, branch_entry)`, so sibling branches derive distinct, deterministic digests from the same parent, independent of scheduling order; the continuation after the join folds `(P, arm = branch_count, join)` onto the parent path. Nested forks compose the same way, since each branch's digest is a prefix-chained value — the parent-frame prefix comes for free. A scaffold test (`par_branches_fork_the_control_path`) documents this.

Replay feeds recorded `Infer`/`Eval`/`Retrieve`/`Store` results back at matching effect IDs and never executes the underlying effect. A divergence fails with an error naming the effect id, its site, its visit, and its control path — and states the id scheme, since a "missing call" is usually an edited program or a different branch/loop path.

## Failure semantics

AgentIR needs an explicit failure model. Effects can fail because of provider errors, timeouts, denied sandbox operations, invalid values, failed evals, canceled branches, or replay divergence.

Each effect (`Infer`/`Store`/`Retrieve`) carries an `on_error` mode on its policy with two settings:

- **`Abort` (default)** — the error propagates and unwinds the program. Correct for the main inference and program-sited effects, where a provider failure is genuinely fatal.
- **`Bind`** — *errors as values* (t-1222): the failure is converted to `{"ok": false, "error": <msg>}` and bound to the effect's `out`, so the surrounding IR branches on it like any other value. The model-initiated tool dispatches (`infer`/`remember`/`recall`) use `Bind`, so a bad tool argument (e.g. a hallucinated model id, or a duplicate memory slug) becomes a tool result the model can read and recover from within the same turn, instead of aborting the whole run.

This deliberately stays within the IR's "no hidden control flow" character: `Bind` adds no new control-flow construct — the error is just data flowing through the existing `Match`/`If` terminators. We chose it over a `Try`/`Catch` terminator precisely because exception-style stack-unwinding is the thing AgentIR avoids.

### Future: resumable handlers

`Bind` lets a program *observe and branch on* a failure, but not *resume the failed effect*. The principled upgrade — when a program (or the model, via a policy) should retry an effect with a corrected argument rather than just see that it failed — is a **resumable handler**, in either of two PL framings:

- **Algebraic effects / handlers** (Koka, OCaml 5, Unison abilities): a handler installed by an enclosing block intercepts the effect failure *holding the delimited continuation*, and may `resume` the computation with a substitute value (e.g. re-run `Infer` against the default model). `Bind` is the degenerate handler that never resumes; `Abort` is the handler that always re-raises.
- **Conditions and restarts** (Common Lisp): the effect site declares named restarts (`use-default-model`, `skip`, `retry`); a handler higher up runs *without unwinding* and selects one at the point of failure — recovery policy decided high, applied low.

Both map cleanly onto the existing effect/handler vocabulary and the stable-effect-id machinery: a resume is a new attempt at the same effect site, recordable and replayable like any other. The migration path is additive — `Abort`/`Bind` remain the common cases; a `Handle { effect, with: BlockId }` construct (or an effect-row on blocks) would layer on top without changing them. Not built yet; recorded here so the IR shape does not preclude it.

The important rule, unchanged across all of these: failure behavior must be visible in traces and replay. A failed effect always has a stable effect ID and an error event (`InferError`/`StoreError`/`RetrieveError`), whether it then aborts, binds, or (in future) resumes — not just a missing result.

## Nested delegation: the `infer` tool

The agent loop exposes an `infer` tool so the model can dispatch a nested `Infer` directly: `{model, prompt, context_refs?}`. The child is a bare single completion — it is offered no tools (`InferPolicy.tools = Some([])`, t-1346), its text feeds back verbatim as the tool result (t-1120), and a failed dispatch (e.g. a hallucinated model id) binds as a readable tool result rather than aborting the turn (t-1222). Trace lineage rides on `parent_op_id` (t-1347).

**Pass-by-reference (t-1344).** `context_refs` is an optional array of tool-call ids — the ids the model itself minted when calling tools, already visible in its own context as `tool_calls[].id` / `tool_call_id`, so nothing extra needs surfacing. At dispatch, the loop program resolves each id against history (`Expr::SelectToolResults`, a pure total expression) and assembles the referenced tool results into the child's messages server-side. The material therefore never transits parent output tokens, and the arguments retained in parent history stay small (refs + prompt) — by-copy delegation of material cost 5x the input rate to emit and then rode history every later turn (see evals/infer-infer/README.md for the measured economics). Results append to history as the tool loop walks one assistant turn, so a batch like `[shell fetch, infer(context_refs=[that shell id])]` resolves within the same turn. An unresolved ref binds as a tool result naming the missing ids; the child is not dispatched.

**Child message structure.** Optional system slot first (`AgentLoopOptions.infer_system_prompt` — owned by the dispatch site, never the model, and part of the program hash like every other loop knob), then one user message per referenced result under a short provenance header, then the instruction prompt. Without refs the child still receives exactly one bare user message: parent history never leaks into the child implicitly.

## `Par` semantics

`Par` should have deterministic semantics before it becomes a core runtime feature.

Open questions to settle before enabling it:

- whether branches get isolated store transactions or shared store access
- how branch writes are merged at the join
- what happens when one branch fails or is canceled
- how branch effects are ordered in the trace
- ~~how stable effect IDs include branch identity~~ — settled: each branch forks the parent control path with its branch index (see "Effect IDs and replay" above), so branch effect ids are deterministic and independent of scheduling order
- whether join result order follows branch declaration order

The default should favor replayability over cleverness. Parallel branches should use isolated transactions unless a specific interpreter provides stronger shared-state semantics.

A demand-first pass over these questions — concrete fan-out patterns, derived requirements (join-all in declaration order, errors-as-values propagation, id assignment at fork, pre-split budgets, Store rejected in branches), and a minimal-Par recommendation — is in [GUIDANCE.md §3](GUIDANCE.md) (t-1356).

## Runtime policy and sandbox boundary

AgentIR does not ask the model to govern itself.

Governance lives outside `Infer` and outside LLM-visible program semantics. The agent is an ordinary Unix process. Lockdown should happen at the interpreter/process boundary using Linux sandboxing primitives: namespaces, seccomp, cgroups, filesystem mounts, network policy, uid/gid isolation, containers, VMs, or remote workers.

`Eval` is the main environment effect. Interpreters may route it through a shell, sandbox, container, VM, or remote executor. The IR records that an eval was requested; the interpreter decides whether and how it is allowed to run.

`EvalRequest` has two variants, one per trust model. `Shell { command }` runs a freeform string via `$SHELL -c` — the model-issued shell tool path, with the quoting/injection surface that implies. `Argv { argv }` execs `argv[0]` directly with `argv[1..]` as arguments — no shell, no re-parsing, so typed tool calls never compile to shell templates (`Eval(argv=["some-tool", "call", tool_id, payload_ref])`). Both share the same env policy (credential stripping), timeout, output caps, and cwd handling; an empty argv is rejected by `validate_program` before any effect runs. Trace `EvalCall` events record the argv verbatim (it is the replay identity for argv Evals) alongside a quoted display `command`.

`Infer` is also interpreter-mediated, but model choice, quota, tenant policy, provider credentials, and network access are runtime concerns. They are not delegated to the LLM.

Inspection can still report what the program syntactically requests, but authorization is enforced by the runtime environment.

## Trace schema relationship

AgentIR execution should map cleanly to trace events.

```text
Instr::Infer -> InferCall / InferResult / InferError
Instr::Eval  -> EvalCall / EvalResult / EvalError
Instr::Retrieve -> RetrieveCall / RetrieveResult / RetrieveError
Instr::Store    -> StoreCall / StoreResult / StoreError
Instr::Emit  -> emitted event plus source location metadata
```

Every effect trace event should include the stable effect ID and source location. Checkpoint events should include enough information to identify the machine snapshot and, if applicable, the store snapshot.

Block enter/exit events can be optional debug traces. Effect traces are required.

## Relationship to PromptIR and Intent

AgentIR is the control/effect IR. PromptIR and Intent are payload IRs.

```text
AgentIR
  Infer(PromptIR) -> Response
  Eval(EvalRequest::Shell | EvalRequest::Argv | EvalRequest::Intent) -> Value
  Retrieve/Store/Emit/Par -> runtime effects
```

PromptIR is the structured payload for `Infer`: sourced, labeled, budgeted context sections that can later be optimized.

Intent is the planned structured payload for `Eval`: deterministic/verifiable work that can replace shell strings for high-assurance workloads. `Argv` (implemented) is the step before it: still an opaque external program, but invoked without a shell.

These should stay separate. AgentIR says when effects happen. PromptIR says what context is given to inference. Intent says what deterministic workload is evaluated.

## Authoring and builder layers

AgentIR is the runtime format, not necessarily the authoring format.

Directly writing blocks is useful for tests and serialization, but normal agent code should be able to use a higher-level builder, DSL, or compiler. The builder can be typed and ergonomic as long as the runtime artifact is validated AgentIR.

The closure-based `Op` representation can remain as a compatibility layer or builder if useful. It should not be the stable runtime representation.

## Migration plan

1. Add `agent-core::ir` with serializable AST/CFG types.
2. Add `run_ir_sequential` beside the current closure-based `run_sequential`.
3. Port the current `agent_loop` into AgentIR with feature parity.
4. Make trace replay use stable AgentIR effect IDs.
5. Make checkpoints serialize `Machine` snapshots.
6. Switch the CLI to AgentIR once the release evals pass under both interpreters.
7. Keep closure-based `Op` as an ergonomic builder or compatibility layer if useful.

## Acceptance for next stable release

- Existing CLI behavior is preserved.
- Existing release evals pass with the AgentIR interpreter.
- AgentIR programs are validated before execution.
- Existing release evals pass against serialized/deserialized AgentIR programs.
- A gated online eval demonstrates model-visible `infer` tool behavior: an agent can request a second `Infer` effect directly, use its result, and produce a correct final answer without routing through shell or a nested agent process.
- Agent programs can be serialized and deserialized before execution.
- Replay is keyed by stable effect IDs, not incidental sequence numbers.
- Replay divergence reports expected and observed effect locations.
- Mid-turn checkpoints can resume without replaying completed effects.
- `Retrieve` / `Store` run against registered hydration sources and sinks (docs/MEMORY.md).
- Docs stop relying on closure-based `Op` as the final program representation.
