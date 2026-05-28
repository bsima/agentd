# AgentIR design

AgentIR is the planned serializable core representation for `agentd` programs.

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
effect handlers: Infer, Eval, Get, Put, Emit, Par
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
    Get { out: Var, key: Expr },
    Put { key: Expr, value: Expr },
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

The first version can use `serde_json::Value` for runtime values. A richer type system can come later.

## Validation and normalization

AgentIR should be validated before execution. A malformed program should fail before any `Infer`, `Eval`, `Get`, `Put`, or `Emit` occurs.

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
store = interpreter-owned state accessed through Get / Put
trace = append-only execution history
```

The store backend is not part of AgentIR. The same program should be able to run with an in-memory STM store, SQLite, a distributed KV store, a replay store, or a read-only inspection store.

The default runtime should use an in-memory STM store. `Get` and `Put` are transactional operations against that store. Interpreters decide the isolation level, persistence, checkpoint cadence, and conflict semantics.

A serialized `Machine` checkpoint contains enough local execution state to resume the program. It may also contain or reference a store snapshot, depending on the backend.

## Effect IDs and replay

Current replay is sequence-based. AgentIR should use stable effect IDs derived from program identity and dynamic execution path:

```text
effect_id = hash(program_hash, effect_site, dynamic_path)

effect_site = block_id + instruction_index
dynamic_path = interpreter-maintained branch/loop/parallel path
```

The dynamic path must be explicit enough to distinguish repeated visits to the same effect site. It should not be an incidental global sequence number. A replay mismatch should say which effect was expected, which effect was observed, and where both are located in the program.

Replay then feeds recorded `Infer` and `Eval` results back at matching effect IDs. If the program diverges, replay should fail with a precise error that names the expected effect and the observed effect.

## Failure semantics

AgentIR needs an explicit failure model. Effects can fail because of provider errors, timeouts, denied sandbox operations, invalid values, failed evals, canceled branches, or replay divergence.

The initial runtime can treat these as typed machine errors that abort execution. If agent programs need to recover from failures, AgentIR should add explicit control-flow support rather than hiding recovery inside interpreters. Likely options are result-shaped effect outputs, error edges, or a `Try`-like terminator.

The important rule is that failure behavior must be visible in traces and replay. A failed effect should have a stable effect ID and an error event, not just a missing result.

## `Par` semantics

`Par` should have deterministic semantics before it becomes a core runtime feature.

Open questions to settle before enabling it:

- whether branches get isolated store transactions or shared store access
- how branch writes are merged at the join
- what happens when one branch fails or is canceled
- how branch effects are ordered in the trace
- how stable effect IDs include branch identity
- whether join result order follows branch declaration order

The default should favor replayability over cleverness. Parallel branches should use isolated transactions unless a specific interpreter provides stronger shared-state semantics.

## Runtime policy and sandbox boundary

AgentIR does not ask the model to govern itself.

Governance lives outside `Infer` and outside LLM-visible program semantics. The agent is an ordinary Unix process. Lockdown should happen at the interpreter/process boundary using Linux sandboxing primitives: namespaces, seccomp, cgroups, filesystem mounts, network policy, uid/gid isolation, containers, VMs, or remote workers.

`Eval` is the main environment effect. Interpreters may route it through a shell, sandbox, container, VM, or remote executor. The IR records that an eval was requested; the interpreter decides whether and how it is allowed to run.

`Infer` is also interpreter-mediated, but model choice, quota, tenant policy, provider credentials, and network access are runtime concerns. They are not delegated to the LLM.

Inspection can still report what the program syntactically requests, but authorization is enforced by the runtime environment.

## Trace schema relationship

AgentIR execution should map cleanly to trace events.

```text
Instr::Infer -> InferCall / InferResult / InferError
Instr::Eval  -> EvalCall / EvalResult / EvalError
Instr::Get   -> GetCall / GetResult / GetError
Instr::Put   -> PutCall / PutResult / PutError
Instr::Emit  -> emitted event plus source location metadata
```

Every effect trace event should include the stable effect ID and source location. Checkpoint events should include enough information to identify the machine snapshot and, if applicable, the store snapshot.

Block enter/exit events can be optional debug traces. Effect traces are required.

## Relationship to PromptIR and Intent

AgentIR is the control/effect IR. PromptIR and Intent are payload IRs.

```text
AgentIR
  Infer(PromptIR) -> Response
  Eval(EvalRequest::Shell | EvalRequest::Intent) -> Value
  Get/Put/Emit/Par -> runtime effects
```

PromptIR is the structured payload for `Infer`: sourced, labeled, budgeted context sections that can later be optimized.

Intent is the structured payload for `Eval`: deterministic/verifiable work that can replace shell strings for high-assurance workloads.

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
- `Get` / `Put` run against the default in-memory STM store.
- Docs stop relying on closure-based `Op` as the final program representation.
