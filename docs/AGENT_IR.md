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

## Abstract machine

The interpreter should run an explicit machine, not Rust continuations:

```rust
pub struct Machine {
    pub program: Program,
    pub block: BlockId,
    pub pc: usize,
    pub env: BTreeMap<Var, Value>,
    pub state: Value,
    pub continuation_stack: Vec<Frame>,
    pub budgets: Budgets,
    pub trace: TraceHandle,
}
```

This gives real checkpoints. A checkpoint can serialize the machine in the middle of a turn, not only the chat history between turns.

## Effect IDs and replay

Current replay is sequence-based. AgentIR should use stable effect IDs derived from program identity and dynamic execution path:

```text
effect_id = hash(program_hash, call_path, block_id, instruction_index, loop_iteration)
```

Replay then feeds recorded `Infer` and `Eval` results back at matching effect IDs. If the program diverges, replay should fail with a precise error that names the expected effect and the observed effect.

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
- A gated online eval demonstrates model-visible `infer` tool behavior: an agent can request a second `Infer` effect directly, use its result, and produce a correct final answer without routing through shell or a nested agent process.
- Agent programs can be serialized and deserialized before execution.
- Replay is keyed by stable effect IDs, not incidental sequence numbers.
- Mid-turn checkpoints can resume without replaying completed effects.
- Docs stop relying on closure-based `Op` as the final program representation.
