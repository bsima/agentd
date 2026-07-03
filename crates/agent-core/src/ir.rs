use crate::op::ChatMessage;
use crate::prompt_ir::PromptIR;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProgramId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Var(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProgramHash(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EffectSite {
    pub block: BlockId,
    pub instruction_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DynamicPath(pub Vec<DynamicPathSegment>);

impl DynamicPath {
    pub fn root() -> Self {
        Self(Vec::new())
    }

    pub fn with_visit(site: EffectSite, visit: u64) -> Self {
        Self(vec![DynamicPathSegment::Visit { site, visit }])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DynamicPathSegment {
    Visit { site: EffectSite, visit: u64 },
    Branch { index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EffectId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EffectKind {
    Infer,
    Eval,
    Emit,
    Retrieve,
    Store,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectLocation {
    pub program_hash: ProgramHash,
    pub effect_id: EffectId,
    pub kind: EffectKind,
    pub site: EffectSite,
    pub dynamic_path: DynamicPath,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub id: ProgramId,
    pub entry: BlockId,
    pub blocks: BTreeMap<BlockId, Block>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    #[serde(default)]
    pub params: Vec<Var>,
    #[serde(default)]
    pub instructions: Vec<Instr>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Instr {
    Let {
        out: Var,
        expr: Expr,
    },
    Infer {
        out: Var,
        model: Expr,
        prompt: PromptRef,
        #[serde(default)]
        policy: InferPolicy,
    },
    Eval {
        out: Var,
        request: EvalRequest,
        #[serde(default)]
        policy: EvalPolicy,
    },
    Emit {
        event: Expr,
    },
    /// Ranked read from registered hydration sources (docs/MEMORY.md).
    /// Full bodies under `max_bytes` by decision; `kind` narrows to one
    /// source kind (e.g. Semantic for the recall tool).
    Retrieve {
        out: Var,
        query: Expr,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kind: Option<crate::hydration::SourceKind>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_bytes: Option<usize>,
        #[serde(default)]
        policy: RetrievePolicy,
    },
    /// Write to a registered hydration sink (docs/MEMORY.md). `sink`
    /// selects the target by registered name; `item` is the sink-schema
    /// payload; the runtime attaches provenance. Replay never mutates.
    Store {
        out: Var,
        sink: Expr,
        op: StoreOp,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<Expr>,
        item: Expr,
        #[serde(default)]
        policy: StorePolicy,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreOp {
    Create,
    Update,
    Delete,
}

impl StoreOp {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

/// What the interpreter does when an effect (Infer/Store/Retrieve) fails.
/// `Abort` (default) propagates the error and unwinds the program — correct
/// for the main inference and program-sited effects, where a provider
/// failure is fatal. `Bind` converts the failure into a value
/// (`{"ok": false, "error": <msg>}`) bound to the effect's `out`, so the
/// surrounding IR can branch on it — this is errors-as-values, used by the
/// model-initiated tool dispatches (infer/remember/recall) so a bad tool
/// argument becomes a tool result the model can recover from instead of
/// killing the whole turn (t-1222). See docs/AGENT_IR.md for the future
/// path to resumable (algebraic-effect / restart) handlers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectErrorMode {
    #[default]
    Abort,
    Bind,
}

/// Per-instruction policy slot, mirroring InferPolicy/EvalPolicy. The
/// per-SINK write policy (Free vs RequireApproval) lives on the sink
/// itself; `on_error` chooses abort-vs-bind for this site.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorePolicy {
    #[serde(default)]
    pub on_error: EffectErrorMode,
}

/// Per-Retrieve policy slot; `on_error` chooses abort-vs-bind for this site.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievePolicy {
    #[serde(default)]
    pub on_error: EffectErrorMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Terminator {
    Goto {
        block: BlockId,
        args: Vec<Expr>,
    },
    If {
        cond: Expr,
        then_block: BlockId,
        #[serde(default)]
        then_args: Vec<Expr>,
        else_block: BlockId,
        #[serde(default)]
        else_args: Vec<Expr>,
    },
    Match {
        value: Expr,
        arms: Vec<MatchArm>,
        default: Option<BlockId>,
        #[serde(default)]
        default_args: Vec<Expr>,
    },
    Return {
        value: Expr,
    },
    Par {
        branches: Vec<BlockId>,
        join: BlockId,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub block: BlockId,
    #[serde(default)]
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    Null,
    Bool(bool),
    String(String),
    Number(serde_json::Number),
    ObjectField {
        field: String,
        pattern: Box<Pattern>,
    },
    Any,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Value(Value),
    Var(Var),
    Field {
        base: Var,
        field: String,
    },
    FieldOr {
        base: Var,
        field: String,
        default: Box<Expr>,
    },
    StringOr {
        value: Box<Expr>,
        default: Box<Expr>,
    },
    If {
        cond: Box<Expr>,
        then_value: Box<Expr>,
        else_value: Box<Expr>,
    },
    Index {
        base: Var,
        index: Box<Expr>,
    },
    Len {
        base: Var,
    },
    IsEmpty {
        base: Var,
    },
    Eq {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Lt {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Or {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    And {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    HasPendingToolCalls {
        base: Var,
    },
    Add {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Sub {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Push {
        base: Var,
        value: Box<Expr>,
    },
    JsonParse {
        value: Box<Expr>,
    },
    JsonParseOr {
        value: Box<Expr>,
        default: Box<Expr>,
    },
    ToString {
        value: Box<Expr>,
    },
    Array(Vec<Expr>),
    Object(BTreeMap<String, Expr>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PromptRef {
    Inline(Vec<ChatMessage>),
    Var(Var),
    PromptIr(Box<PromptIR>),
    PromptIrVar(Var),
}

/// Per-Infer policy slot. `Instr::Infer` is a single provider call — it has
/// no multi-turn semantics, so there is deliberately no turn limit here.
/// Turn budgets belong to the loop *program* that contains the Infer: see
/// the counter threaded through `agent_loop_ir` (ir_agent.rs) and
/// `op::agent_loop` (t-1056). Old serialized programs that still carry a
/// `max_turns` field deserialize fine; serde ignores unknown fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferPolicy {
    /// Abort (default) vs bind-error-as-value for this Infer site.
    #[serde(default)]
    pub on_error: EffectErrorMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvalRequest {
    Shell { command: Expr },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalPolicy {
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    pub program: Program,
    pub block: BlockId,
    pub pc: usize,
    #[serde(default)]
    pub env: BTreeMap<Var, Value>,
    #[serde(default)]
    pub effect_visits: BTreeMap<String, u64>,
    #[serde(default)]
    pub continuation_stack: Vec<Frame>,
    #[serde(default)]
    pub budgets: Budgets,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub block: BlockId,
    pub pc: usize,
    #[serde(default)]
    pub env: BTreeMap<Var, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    pub max_infer_calls: Option<u64>,
    pub max_eval_calls: Option<u64>,
}

pub fn program_hash(program: &Program) -> Result<ProgramHash> {
    crate::ir_normalize::canonical_program_hash(program)
}

pub fn effect_location(
    program_hash: ProgramHash,
    kind: EffectKind,
    site: EffectSite,
    dynamic_path: DynamicPath,
) -> Result<EffectLocation> {
    let bytes = serde_json::to_vec(&(&program_hash, kind, site, &dynamic_path))?;
    let digest = Sha256::digest(bytes);
    Ok(EffectLocation {
        program_hash,
        effect_id: EffectId(format!("sha256:{digest:x}")),
        kind,
        site,
        dynamic_path,
    })
}

pub fn validate_program(program: &Program) -> Result<()> {
    if !program.blocks.contains_key(&program.entry) {
        return Err(anyhow!(
            "AgentIR entry block {:?} does not exist",
            program.entry
        ));
    }

    for (block_id, block) in &program.blocks {
        validate_unique_vars(&block.params, "block params", *block_id)?;
        validate_local_shadowing(block, *block_id)?;
        validate_terminator_block_refs(program, &block.terminator)?;
    }

    let mut inputs = BTreeMap::<BlockId, std::collections::BTreeSet<Var>>::new();
    inputs.insert(
        program.entry,
        program
            .blocks
            .get(&program.entry)
            .expect("entry checked")
            .params
            .iter()
            .cloned()
            .collect(),
    );
    let mut worklist = vec![program.entry];
    while let Some(block_id) = worklist.pop() {
        let block = program.blocks.get(&block_id).expect("queued block exists");
        let mut defined = inputs.get(&block_id).cloned().unwrap_or_default();
        defined.extend(block.params.iter().cloned());
        for instr in &block.instructions {
            validate_instr_vars(instr, &defined, block_id)?;
            if let Some(out) = instr_out(instr) {
                defined.insert(out.clone());
            }
        }
        validate_terminator_vars(&block.terminator, &defined, block_id)?;
        for (target, inherited) in terminator_successors(program, &block.terminator, &defined)? {
            let entry = inputs.entry(target).or_default();
            let old_len = entry.len();
            entry.extend(inherited);
            if entry.len() != old_len {
                worklist.push(target);
            }
        }
    }

    Ok(())
}

fn validate_local_shadowing(block: &Block, block_id: BlockId) -> Result<()> {
    let mut defined = block
        .params
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    for instr in &block.instructions {
        if let Some(out) = instr_out(instr) {
            if !defined.insert(out.clone()) {
                return Err(anyhow!(
                    "AgentIR variable {:?} is shadowed in block {:?}",
                    out,
                    block_id
                ));
            }
        }
    }
    Ok(())
}

fn validate_unique_vars(vars: &[Var], label: &str, block_id: BlockId) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for var in vars {
        if !seen.insert(var) {
            return Err(anyhow!(
                "AgentIR duplicate variable {:?} in {label} for block {:?}",
                var,
                block_id
            ));
        }
    }
    Ok(())
}

fn instr_out(instr: &Instr) -> Option<&Var> {
    match instr {
        Instr::Let { out, .. }
        | Instr::Infer { out, .. }
        | Instr::Eval { out, .. }
        | Instr::Retrieve { out, .. }
        | Instr::Store { out, .. } => Some(out),
        Instr::Emit { .. } => None,
    }
}

fn validate_instr_vars(
    instr: &Instr,
    defined: &std::collections::BTreeSet<Var>,
    block_id: BlockId,
) -> Result<()> {
    match instr {
        Instr::Let { expr, .. } => validate_expr_vars(expr, defined, block_id),
        Instr::Infer { model, prompt, .. } => {
            validate_expr_vars(model, defined, block_id)?;
            validate_prompt_ref_vars(prompt, defined, block_id)
        }
        Instr::Eval { request, .. } => validate_eval_request_vars(request, defined, block_id),
        Instr::Emit { event } => validate_expr_vars(event, defined, block_id),
        Instr::Retrieve { query, .. } => validate_expr_vars(query, defined, block_id),
        Instr::Store { sink, id, item, .. } => {
            validate_expr_vars(sink, defined, block_id)?;
            if let Some(id) = id {
                validate_expr_vars(id, defined, block_id)?;
            }
            validate_expr_vars(item, defined, block_id)
        }
    }
}

fn validate_prompt_ref_vars(
    prompt: &PromptRef,
    defined: &std::collections::BTreeSet<Var>,
    block_id: BlockId,
) -> Result<()> {
    match prompt {
        PromptRef::Inline(_) | PromptRef::PromptIr(_) => Ok(()),
        PromptRef::Var(var) | PromptRef::PromptIrVar(var) => validate_var(var, defined, block_id),
    }
}

fn validate_eval_request_vars(
    request: &EvalRequest,
    defined: &std::collections::BTreeSet<Var>,
    block_id: BlockId,
) -> Result<()> {
    match request {
        EvalRequest::Shell { command } => validate_expr_vars(command, defined, block_id),
    }
}

fn validate_terminator_block_refs(program: &Program, terminator: &Terminator) -> Result<()> {
    match terminator {
        Terminator::Goto { block, .. } => validate_block_ref(program, *block),
        Terminator::If {
            then_block,
            else_block,
            ..
        } => {
            validate_block_ref(program, *then_block)?;
            validate_block_ref(program, *else_block)
        }
        Terminator::Match { arms, default, .. } => {
            for arm in arms {
                validate_block_ref(program, arm.block)?;
            }
            if let Some(default) = default {
                validate_block_ref(program, *default)?;
            }
            Ok(())
        }
        Terminator::Return { .. } => Ok(()),
        Terminator::Par { branches, join } => {
            for branch in branches {
                validate_block_ref(program, *branch)?;
            }
            validate_block_ref(program, *join)
        }
    }
}

fn validate_terminator_vars(
    terminator: &Terminator,
    defined: &std::collections::BTreeSet<Var>,
    block_id: BlockId,
) -> Result<()> {
    match terminator {
        Terminator::Goto { args, .. } => {
            for arg in args {
                validate_expr_vars(arg, defined, block_id)?;
            }
            Ok(())
        }
        Terminator::If {
            cond,
            then_args,
            else_args,
            ..
        } => {
            validate_expr_vars(cond, defined, block_id)?;
            for arg in then_args.iter().chain(else_args) {
                validate_expr_vars(arg, defined, block_id)?;
            }
            Ok(())
        }
        Terminator::Match {
            value,
            arms,
            default_args,
            ..
        } => {
            validate_expr_vars(value, defined, block_id)?;
            for arm in arms {
                for arg in &arm.args {
                    validate_expr_vars(arg, defined, block_id)?;
                }
            }
            for arg in default_args {
                validate_expr_vars(arg, defined, block_id)?;
            }
            Ok(())
        }
        Terminator::Return { value } => validate_expr_vars(value, defined, block_id),
        Terminator::Par { .. } => Ok(()),
    }
}

fn terminator_successors(
    program: &Program,
    terminator: &Terminator,
    defined: &std::collections::BTreeSet<Var>,
) -> Result<Vec<(BlockId, std::collections::BTreeSet<Var>)>> {
    match terminator {
        Terminator::Goto { block, args } => {
            let target_block = program.blocks.get(block).expect("block ref checked");
            if target_block.params.len() != args.len() {
                return Err(anyhow!(
                    "AgentIR Goto to {:?} expected {} args, got {}",
                    block,
                    target_block.params.len(),
                    args.len()
                ));
            }
            let mut inherited = defined.clone();
            inherited.extend(target_block.params.iter().cloned());
            Ok(vec![(*block, inherited)])
        }
        Terminator::If {
            then_block,
            then_args,
            else_block,
            else_args,
            ..
        } => {
            let then_target = program.blocks.get(then_block).expect("block ref checked");
            if then_target.params.len() != then_args.len() {
                return Err(anyhow!(
                    "AgentIR If then branch to {:?} expected {} args, got {}",
                    then_block,
                    then_target.params.len(),
                    then_args.len()
                ));
            }
            let else_target = program.blocks.get(else_block).expect("block ref checked");
            if else_target.params.len() != else_args.len() {
                return Err(anyhow!(
                    "AgentIR If else branch to {:?} expected {} args, got {}",
                    else_block,
                    else_target.params.len(),
                    else_args.len()
                ));
            }
            Ok(vec![
                (*then_block, defined.clone()),
                (*else_block, defined.clone()),
            ])
        }
        Terminator::Match {
            arms,
            default,
            default_args,
            ..
        } => {
            let mut out = Vec::new();
            for arm in arms {
                let target = program.blocks.get(&arm.block).expect("block ref checked");
                if target.params.len() != arm.args.len() {
                    return Err(anyhow!(
                        "AgentIR Match arm to {:?} expected {} args, got {}",
                        arm.block,
                        target.params.len(),
                        arm.args.len()
                    ));
                }
                out.push((arm.block, defined.clone()));
            }
            if let Some(default) = default {
                let target = program.blocks.get(default).expect("block ref checked");
                if target.params.len() != default_args.len() {
                    return Err(anyhow!(
                        "AgentIR Match default to {:?} expected {} args, got {}",
                        default,
                        target.params.len(),
                        default_args.len()
                    ));
                }
                out.push((*default, defined.clone()));
            } else if !default_args.is_empty() {
                return Err(anyhow!(
                    "AgentIR Match default args provided without default block"
                ));
            }
            Ok(out)
        }
        Terminator::Return { .. } => Ok(vec![]),
        Terminator::Par { branches, join } => {
            let mut out = branches
                .iter()
                .map(|branch| (*branch, defined.clone()))
                .collect::<Vec<_>>();
            out.push((*join, defined.clone()));
            Ok(out)
        }
    }
}

fn validate_block_ref(program: &Program, block_id: BlockId) -> Result<()> {
    if program.blocks.contains_key(&block_id) {
        Ok(())
    } else {
        Err(anyhow!(
            "AgentIR referenced block {:?} does not exist",
            block_id
        ))
    }
}

fn validate_expr_vars(
    expr: &Expr,
    defined: &std::collections::BTreeSet<Var>,
    block_id: BlockId,
) -> Result<()> {
    match expr {
        Expr::Value(_) => Ok(()),
        Expr::Var(var) => validate_var(var, defined, block_id),
        Expr::Field { base, .. }
        | Expr::Len { base }
        | Expr::IsEmpty { base }
        | Expr::HasPendingToolCalls { base } => validate_var(base, defined, block_id),
        Expr::FieldOr { base, default, .. } => {
            validate_var(base, defined, block_id)?;
            validate_expr_vars(default, defined, block_id)
        }
        Expr::StringOr { value, default } => {
            validate_expr_vars(value, defined, block_id)?;
            validate_expr_vars(default, defined, block_id)
        }
        Expr::If {
            cond,
            then_value,
            else_value,
        } => {
            validate_expr_vars(cond, defined, block_id)?;
            validate_expr_vars(then_value, defined, block_id)?;
            validate_expr_vars(else_value, defined, block_id)
        }
        Expr::Index { base, index } => {
            validate_var(base, defined, block_id)?;
            validate_expr_vars(index, defined, block_id)
        }
        Expr::Eq { left, right }
        | Expr::Lt { left, right }
        | Expr::Or { left, right }
        | Expr::And { left, right }
        | Expr::Add { left, right }
        | Expr::Sub { left, right } => {
            validate_expr_vars(left, defined, block_id)?;
            validate_expr_vars(right, defined, block_id)
        }
        Expr::Push { base, value } => {
            validate_var(base, defined, block_id)?;
            validate_expr_vars(value, defined, block_id)
        }
        Expr::JsonParseOr { value, default } => {
            validate_expr_vars(value, defined, block_id)?;
            validate_expr_vars(default, defined, block_id)
        }
        Expr::JsonParse { value } | Expr::ToString { value } => {
            validate_expr_vars(value, defined, block_id)
        }
        Expr::Array(items) => {
            for item in items {
                validate_expr_vars(item, defined, block_id)?;
            }
            Ok(())
        }
        Expr::Object(fields) => {
            for value in fields.values() {
                validate_expr_vars(value, defined, block_id)?;
            }
            Ok(())
        }
    }
}

fn validate_var(
    var: &Var,
    defined: &std::collections::BTreeSet<Var>,
    block_id: BlockId,
) -> Result<()> {
    if defined.contains(var) {
        Ok(())
    } else {
        Err(anyhow!(
            "AgentIR variable {:?} is used before definition in block {:?}",
            var,
            block_id
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_round_trips_through_json() {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            Block {
                params: vec![],
                instructions: vec![Instr::Infer {
                    out: Var("response".into()),
                    model: Expr::Value(Value::String("mock".into())),
                    prompt: PromptRef::Inline(vec![ChatMessage::user("hello")]),
                    policy: InferPolicy::default(),
                }],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("response".into())),
                },
            },
        );
        let program = Program {
            id: ProgramId("test".into()),
            entry: BlockId(0),
            blocks,
        };

        let encoded = serde_json::to_string(&program).unwrap();
        let decoded: Program = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, program);
    }

    /// InferPolicy used to carry an unused `max_turns` field (removed in
    /// t-1056). Programs serialized before the removal must still load:
    /// serde ignores unknown fields, so the stale key is dropped silently.
    #[test]
    fn infer_policy_tolerates_legacy_max_turns_field() {
        let legacy = r#"{"max_turns":3,"on_error":"bind"}"#;
        let policy: InferPolicy = serde_json::from_str(legacy).unwrap();
        assert_eq!(policy.on_error, EffectErrorMode::Bind);

        let legacy_null = r#"{"max_turns":null}"#;
        let policy: InferPolicy = serde_json::from_str(legacy_null).unwrap();
        assert_eq!(policy, InferPolicy::default());
    }

    #[test]
    fn stable_effect_ids_are_deterministic_and_visit_sensitive() {
        let program = Program {
            id: ProgramId("ids".into()),
            entry: BlockId(0),
            blocks: BTreeMap::from([(
                BlockId(0),
                Block {
                    params: vec![],
                    instructions: vec![Instr::Retrieve {
                        out: Var("x".into()),
                        query: Expr::Value(Value::String("k".into())),
                        kind: None,
                        max_bytes: None,
                        policy: Default::default(),
                    }],
                    terminator: Terminator::Return {
                        value: Expr::Var(Var("x".into())),
                    },
                },
            )]),
        };
        let hash = program_hash(&program).unwrap();
        let site = EffectSite {
            block: BlockId(0),
            instruction_index: 0,
        };
        let first = effect_location(
            hash.clone(),
            EffectKind::Retrieve,
            site,
            DynamicPath::with_visit(site, 0),
        )
        .unwrap();
        let first_again = effect_location(
            hash.clone(),
            EffectKind::Retrieve,
            site,
            DynamicPath::with_visit(site, 0),
        )
        .unwrap();
        let second_visit = effect_location(
            hash,
            EffectKind::Retrieve,
            site,
            DynamicPath::with_visit(site, 1),
        )
        .unwrap();

        assert_eq!(first.effect_id, first_again.effect_id);
        assert_ne!(first.effect_id, second_visit.effect_id);
    }

    #[test]
    fn machine_snapshot_round_trips_through_json() {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            Block {
                params: vec![],
                instructions: vec![],
                terminator: Terminator::Return {
                    value: Expr::Value(Value::String("done".into())),
                },
            },
        );
        let machine = Machine {
            program: Program {
                id: ProgramId("snapshot-test".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::from([(Var("x".into()), Value::String("y".into()))]),
            effect_visits: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Budgets::default(),
        };

        let encoded = serde_json::to_value(&machine).unwrap();
        let decoded: Machine = serde_json::from_value(encoded).unwrap();

        assert_eq!(decoded, machine);
    }

    #[test]
    fn validation_rejects_missing_entry_block() {
        let program = Program {
            id: ProgramId("bad".into()),
            entry: BlockId(99),
            blocks: BTreeMap::new(),
        };
        let err = validate_program(&program).unwrap_err().to_string();
        assert!(err.contains("entry block"));
    }

    #[test]
    fn validation_rejects_use_before_definition() {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            Block {
                params: vec![],
                instructions: vec![],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("missing".into())),
                },
            },
        );
        let program = Program {
            id: ProgramId("bad".into()),
            entry: BlockId(0),
            blocks,
        };
        let err = validate_program(&program).unwrap_err().to_string();
        assert!(err.contains("used before definition"));
    }

    #[test]
    fn validation_rejects_shadowing() {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            Block {
                params: vec![],
                instructions: vec![
                    Instr::Let {
                        out: Var("x".into()),
                        expr: Expr::Value(Value::Null),
                    },
                    Instr::Let {
                        out: Var("x".into()),
                        expr: Expr::Value(Value::Null),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("x".into())),
                },
            },
        );
        let program = Program {
            id: ProgramId("bad".into()),
            entry: BlockId(0),
            blocks,
        };
        let err = validate_program(&program).unwrap_err().to_string();
        assert!(err.contains("shadowed"));
    }
}
