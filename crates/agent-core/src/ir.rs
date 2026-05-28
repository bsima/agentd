use crate::op::ChatMessage;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProgramId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Var(pub String);

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
    Get {
        out: Var,
        key: Expr,
    },
    Put {
        key: Expr,
        value: Expr,
    },
    Emit {
        event: Expr,
    },
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
        else_block: BlockId,
    },
    Match {
        value: Expr,
        arms: Vec<MatchArm>,
        default: Option<BlockId>,
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
    Field { base: Var, field: String },
    Array(Vec<Expr>),
    Object(BTreeMap<String, Expr>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PromptRef {
    Inline(Vec<ChatMessage>),
    Var(Var),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferPolicy {
    pub max_turns: Option<usize>,
}

impl Default for InferPolicy {
    fn default() -> Self {
        Self { max_turns: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvalRequest {
    Shell { command: Expr },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalPolicy {
    pub timeout_ms: Option<u64>,
}

impl Default for EvalPolicy {
    fn default() -> Self {
        Self { timeout_ms: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    pub program: Program,
    pub block: BlockId,
    pub pc: usize,
    #[serde(default)]
    pub env: BTreeMap<Var, Value>,
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

pub fn validate_program(program: &Program) -> Result<()> {
    if !program.blocks.contains_key(&program.entry) {
        return Err(anyhow!(
            "AgentIR entry block {:?} does not exist",
            program.entry
        ));
    }

    for (block_id, block) in &program.blocks {
        validate_unique_vars(&block.params, "block params", *block_id)?;
        let mut defined = block
            .params
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        for instr in &block.instructions {
            validate_instr_vars(instr, &defined, *block_id)?;
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
        validate_terminator(program, *block_id, &block.terminator, &defined)?;
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
        | Instr::Get { out, .. } => Some(out),
        Instr::Put { .. } | Instr::Emit { .. } => None,
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
        Instr::Get { key, .. } => validate_expr_vars(key, defined, block_id),
        Instr::Put { key, value } => {
            validate_expr_vars(key, defined, block_id)?;
            validate_expr_vars(value, defined, block_id)
        }
        Instr::Emit { event } => validate_expr_vars(event, defined, block_id),
    }
}

fn validate_prompt_ref_vars(
    prompt: &PromptRef,
    defined: &std::collections::BTreeSet<Var>,
    block_id: BlockId,
) -> Result<()> {
    match prompt {
        PromptRef::Inline(_) => Ok(()),
        PromptRef::Var(var) => validate_var(var, defined, block_id),
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

fn validate_terminator(
    program: &Program,
    block_id: BlockId,
    terminator: &Terminator,
    defined: &std::collections::BTreeSet<Var>,
) -> Result<()> {
    match terminator {
        Terminator::Goto { block, args } => {
            validate_block_ref(program, *block)?;
            validate_goto_args(program, *block, args, defined, block_id)
        }
        Terminator::If {
            cond,
            then_block,
            else_block,
        } => {
            validate_expr_vars(cond, defined, block_id)?;
            validate_block_ref(program, *then_block)?;
            validate_block_ref(program, *else_block)
        }
        Terminator::Match {
            value,
            arms,
            default,
        } => {
            validate_expr_vars(value, defined, block_id)?;
            for arm in arms {
                validate_block_ref(program, arm.block)?;
            }
            if let Some(default) = default {
                validate_block_ref(program, *default)?;
            }
            Ok(())
        }
        Terminator::Return { value } => validate_expr_vars(value, defined, block_id),
        Terminator::Par { branches, join } => {
            for branch in branches {
                validate_block_ref(program, *branch)?;
            }
            validate_block_ref(program, *join)
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

fn validate_goto_args(
    program: &Program,
    target: BlockId,
    args: &[Expr],
    defined: &std::collections::BTreeSet<Var>,
    source: BlockId,
) -> Result<()> {
    let target_block = program
        .blocks
        .get(&target)
        .expect("block ref checked before goto args");
    if target_block.params.len() != args.len() {
        return Err(anyhow!(
            "AgentIR Goto from {:?} to {:?} expected {} args, got {}",
            source,
            target,
            target_block.params.len(),
            args.len()
        ));
    }
    for arg in args {
        validate_expr_vars(arg, defined, source)?;
    }
    Ok(())
}

fn validate_expr_vars(
    expr: &Expr,
    defined: &std::collections::BTreeSet<Var>,
    block_id: BlockId,
) -> Result<()> {
    match expr {
        Expr::Value(_) => Ok(()),
        Expr::Var(var) => validate_var(var, defined, block_id),
        Expr::Field { base, .. } => validate_var(base, defined, block_id),
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
