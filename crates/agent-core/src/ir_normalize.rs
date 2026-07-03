use crate::ir::*;
use anyhow::{anyhow, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Normalize a validated AgentIR program into the canonical form used for
/// identity. The returned program is strict SSA: every value used by a block is
/// either one of that block's params or is defined earlier in that block.
///
/// `Program.id` is normalized to the empty string. Program identity is the
/// executable structure, not a caller-chosen label; including `id` would make
/// two otherwise identical programs hash differently for non-semantic reasons.
pub fn normalize_program(program: &Program) -> Result<Program> {
    validate_program(program)?;

    let reachable = reachable_blocks(program);
    let def_order = definition_order(program, &reachable);
    let required = required_block_params(program, &reachable);
    let mut ssa = add_ssa_params(program, &reachable, &required, &def_order)?;
    ssa = renumber_blocks(&ssa);
    ssa = rename_vars(&ssa)?;
    ssa.id = ProgramId(String::new());
    validate_program(&ssa)?;
    validate_strict_ssa_program(&ssa)?;
    Ok(ssa)
}

/// Hash the canonical normalized program form.
pub fn canonical_program_hash(program: &Program) -> Result<ProgramHash> {
    let normalized = normalize_program(program)?;
    let bytes = serde_json::to_vec(&HashProgram::from(&normalized))?;
    let digest = Sha256::digest(bytes);
    Ok(ProgramHash(format!("sha256:{digest:x}")))
}

#[derive(Serialize)]
struct HashProgram<'a> {
    entry: &'a BlockId,
    blocks: &'a BTreeMap<BlockId, Block>,
}

impl<'a> From<&'a Program> for HashProgram<'a> {
    fn from(program: &'a Program) -> Self {
        Self {
            entry: &program.entry,
            blocks: &program.blocks,
        }
    }
}

pub fn validate_strict_ssa_program(program: &Program) -> Result<()> {
    validate_program(program)?;
    for (block_id, block) in &program.blocks {
        let mut defined = block.params.iter().cloned().collect::<BTreeSet<_>>();
        for instr in &block.instructions {
            for var in instr_uses(instr) {
                if !defined.contains(&var) {
                    return Err(anyhow!(
                        "AgentIR strict SSA violation: variable {:?} is not local to block {:?}",
                        var,
                        block_id
                    ));
                }
            }
            if let Some(out) = instr_out_public(instr) {
                defined.insert(out.clone());
            }
        }
        for var in terminator_value_uses(&block.terminator) {
            if !defined.contains(&var) {
                return Err(anyhow!(
                    "AgentIR strict SSA violation: variable {:?} is not local to block {:?}",
                    var,
                    block_id
                ));
            }
        }
        for (target, args) in terminator_edges_with_args(&block.terminator) {
            let expected = program.blocks.get(&target).expect("validated").params.len();
            if args.len() != expected {
                return Err(anyhow!(
                    "AgentIR strict SSA violation: edge from {:?} to {:?} expected {} args, got {}",
                    block_id,
                    target,
                    expected,
                    args.len()
                ));
            }
        }
    }
    Ok(())
}

fn reachable_blocks(program: &Program) -> Vec<BlockId> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    fn dfs(program: &Program, id: BlockId, seen: &mut BTreeSet<BlockId>, out: &mut Vec<BlockId>) {
        if !seen.insert(id) {
            return;
        }
        out.push(id);
        if let Some(block) = program.blocks.get(&id) {
            for succ in terminator_successor_ids(&block.terminator) {
                dfs(program, succ, seen, out);
            }
        }
    }
    dfs(program, program.entry, &mut seen, &mut out);
    out
}

fn definition_order(program: &Program, order: &[BlockId]) -> BTreeMap<Var, usize> {
    let mut map = BTreeMap::new();
    let mut next = 0usize;
    for id in order {
        let block = &program.blocks[id];
        for param in &block.params {
            map.entry(param.clone()).or_insert_with(|| {
                let n = next;
                next += 1;
                n
            });
        }
        for instr in &block.instructions {
            if let Some(out) = instr_out_public(instr) {
                map.entry(out.clone()).or_insert_with(|| {
                    let n = next;
                    next += 1;
                    n
                });
            }
        }
    }
    map
}

fn required_block_params(
    program: &Program,
    reachable: &[BlockId],
) -> BTreeMap<BlockId, BTreeSet<Var>> {
    let reachable_set = reachable.iter().copied().collect::<BTreeSet<_>>();
    let mut required = BTreeMap::<BlockId, BTreeSet<Var>>::new();
    let mut local_defs = BTreeMap::<BlockId, BTreeSet<Var>>::new();
    for id in reachable {
        let block = &program.blocks[id];
        let defs = block
            .instructions
            .iter()
            .filter_map(instr_out_public)
            .cloned()
            .collect::<BTreeSet<_>>();
        local_defs.insert(*id, defs);
        required.insert(*id, direct_free_vars(block));
    }

    let mut changed = true;
    while changed {
        changed = false;
        for id in reachable.iter().rev() {
            let mut add = BTreeSet::new();
            let defs = &local_defs[id];
            let current_params = program.blocks[id]
                .params
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            for succ in terminator_successor_ids(&program.blocks[id].terminator) {
                if !reachable_set.contains(&succ) {
                    continue;
                }
                for var in required.get(&succ).into_iter().flatten() {
                    if !defs.contains(var) && !current_params.contains(var) {
                        add.insert(var.clone());
                    }
                }
            }
            let entry = required.entry(*id).or_default();
            let old = entry.len();
            entry.extend(add);
            changed |= entry.len() != old;
        }
    }
    required
}

fn direct_free_vars(block: &Block) -> BTreeSet<Var> {
    let mut defined = block.params.iter().cloned().collect::<BTreeSet<_>>();
    let mut free = BTreeSet::new();
    for instr in &block.instructions {
        for var in instr_uses(instr) {
            if !defined.contains(&var) {
                free.insert(var);
            }
        }
        if let Some(out) = instr_out_public(instr) {
            defined.insert(out.clone());
        }
    }
    for var in terminator_control_uses(&block.terminator) {
        if !defined.contains(&var) {
            free.insert(var);
        }
    }
    // Existing edge args are exprs evaluated in this block's scope; their vars
    // are uses just like instruction operands.
    for var in terminator_value_uses(&block.terminator) {
        if !defined.contains(&var) {
            free.insert(var);
        }
    }
    free
}

fn add_ssa_params(
    program: &Program,
    reachable: &[BlockId],
    required: &BTreeMap<BlockId, BTreeSet<Var>>,
    def_order: &BTreeMap<Var, usize>,
) -> Result<Program> {
    let reachable_set = reachable.iter().copied().collect::<BTreeSet<_>>();
    let mut blocks = BTreeMap::new();
    for id in reachable {
        let block = &program.blocks[id];
        let params = ordered_required_params(block, &required[id], def_order);
        let terminator = add_edge_args(
            &block.terminator,
            program,
            required,
            def_order,
            &reachable_set,
        )?;
        blocks.insert(
            *id,
            Block {
                params,
                instructions: block.instructions.clone(),
                terminator,
            },
        );
    }
    Ok(Program {
        id: ProgramId(String::new()),
        entry: program.entry,
        blocks,
    })
}

fn ordered_required_params(
    block: &Block,
    req: &BTreeSet<Var>,
    def_order: &BTreeMap<Var, usize>,
) -> Vec<Var> {
    // Existing params are an explicit interface: every in-edge already passes a
    // positional arg for each one, so they are always kept, in order. Only vars
    // the block still needs implicitly (dominator-scoped uses) are appended.
    let mut out = Vec::new();
    for p in &block.params {
        if !out.contains(p) {
            out.push(p.clone());
        }
    }
    let mut rest = req
        .iter()
        .filter(|v| !out.contains(v))
        .cloned()
        .collect::<Vec<_>>();
    rest.sort_by_key(|v| (def_order.get(v).copied().unwrap_or(usize::MAX), v.clone()));
    out.extend(rest);
    out
}

fn add_edge_args(
    term: &Terminator,
    program: &Program,
    required: &BTreeMap<BlockId, BTreeSet<Var>>,
    def_order: &BTreeMap<Var, usize>,
    reachable: &BTreeSet<BlockId>,
) -> Result<Terminator> {
    // Existing args stay bound to the target's existing params positionally;
    // params appended by SSA-ification get by-name Var args appended after.
    let args_for = |target: BlockId, existing: &[Expr]| -> Vec<Expr> {
        if !reachable.contains(&target) {
            return Vec::new();
        }
        let target_block = &program.blocks[&target];
        let ordered = ordered_required_params(target_block, &required[&target], def_order);
        let mut args = existing.to_vec();
        args.extend(
            ordered
                .into_iter()
                .skip(target_block.params.len())
                .map(Expr::Var),
        );
        args
    };
    Ok(match term {
        Terminator::Goto { block, args } => Terminator::Goto {
            block: *block,
            args: args_for(*block, args),
        },
        Terminator::If {
            cond,
            then_block,
            then_args,
            else_block,
            else_args,
        } => Terminator::If {
            cond: cond.clone(),
            then_block: *then_block,
            then_args: args_for(*then_block, then_args),
            else_block: *else_block,
            else_args: args_for(*else_block, else_args),
        },
        Terminator::Match {
            value,
            arms,
            default,
            default_args,
        } => Terminator::Match {
            value: value.clone(),
            arms: arms
                .iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern.clone(),
                    block: arm.block,
                    args: args_for(arm.block, &arm.args),
                })
                .collect(),
            default: *default,
            default_args: default
                .map(|d| args_for(d, default_args))
                .unwrap_or_default(),
        },
        Terminator::Return { value } => Terminator::Return {
            value: value.clone(),
        },
        Terminator::Par { branches, join } => Terminator::Par {
            branches: branches.clone(),
            join: *join,
        },
    })
}

fn renumber_blocks(program: &Program) -> Program {
    let order = reachable_blocks(program);
    let map = order
        .iter()
        .enumerate()
        .map(|(i, old)| (*old, BlockId(i as u32)))
        .collect::<BTreeMap<_, _>>();
    let mut blocks = BTreeMap::new();
    for old in order {
        let mut block = program.blocks[&old].clone();
        block.terminator = map_terminator_blocks(&block.terminator, &map);
        blocks.insert(map[&old], block);
    }
    Program {
        id: ProgramId(String::new()),
        entry: BlockId(0),
        blocks,
    }
}

fn rename_vars(program: &Program) -> Result<Program> {
    let mut env = BTreeMap::<Var, Var>::new();
    let mut next = 0usize;
    let mut blocks = BTreeMap::new();
    for (id, block) in &program.blocks {
        let mut new_params = Vec::new();
        for p in &block.params {
            let nv = Var(format!("v{next}"));
            next += 1;
            env.insert(p.clone(), nv.clone());
            new_params.push(nv);
        }
        let mut instructions = Vec::new();
        for instr in &block.instructions {
            let mut ni = rename_instr_uses(instr, &env)?;
            if let Some(out) = instr_out_mut(&mut ni) {
                let nv = Var(format!("v{next}"));
                next += 1;
                env.insert(out.clone(), nv.clone());
                *out = nv;
            }
            instructions.push(ni);
        }
        let terminator = rename_terminator_uses(&block.terminator, &env)?;
        blocks.insert(
            *id,
            Block {
                params: new_params,
                instructions,
                terminator,
            },
        );
    }
    Ok(Program {
        id: ProgramId(String::new()),
        entry: program.entry,
        blocks,
    })
}

fn rename_var(var: &Var, env: &BTreeMap<Var, Var>) -> Result<Var> {
    env.get(var)
        .cloned()
        .ok_or_else(|| anyhow!("normalization internal error: missing rename for {var:?}"))
}

fn instr_out_public(instr: &Instr) -> Option<&Var> {
    match instr {
        Instr::Let { out, .. }
        | Instr::Infer { out, .. }
        | Instr::Eval { out, .. }
        | Instr::Retrieve { out, .. }
        | Instr::Store { out, .. }
        | Instr::Tool { out, .. } => Some(out),
        Instr::Emit { .. } => None,
    }
}

fn instr_out_mut(instr: &mut Instr) -> Option<&mut Var> {
    match instr {
        Instr::Let { out, .. }
        | Instr::Infer { out, .. }
        | Instr::Eval { out, .. }
        | Instr::Retrieve { out, .. }
        | Instr::Store { out, .. }
        | Instr::Tool { out, .. } => Some(out),
        Instr::Emit { .. } => None,
    }
}

fn instr_uses(instr: &Instr) -> BTreeSet<Var> {
    let mut out = BTreeSet::new();
    match instr {
        Instr::Let { expr, .. } => collect_expr_vars(expr, &mut out),
        Instr::Infer { model, prompt, .. } => {
            collect_expr_vars(model, &mut out);
            collect_prompt_vars(prompt, &mut out);
        }
        Instr::Eval { request, .. } => collect_eval_request_vars(request, &mut out),
        Instr::Emit { event } => collect_expr_vars(event, &mut out),
        Instr::Retrieve { query, .. } => collect_expr_vars(query, &mut out),
        Instr::Store { sink, id, item, .. } => {
            collect_expr_vars(sink, &mut out);
            if let Some(id) = id {
                collect_expr_vars(id, &mut out);
            }
            collect_expr_vars(item, &mut out);
        }
        Instr::Tool { arguments, .. } => collect_expr_vars(arguments, &mut out),
    }
    out
}

fn terminator_control_uses(term: &Terminator) -> BTreeSet<Var> {
    let mut out = BTreeSet::new();
    match term {
        Terminator::If { cond, .. } => collect_expr_vars(cond, &mut out),
        Terminator::Match { value, .. } => collect_expr_vars(value, &mut out),
        Terminator::Return { value } => collect_expr_vars(value, &mut out),
        Terminator::Goto { .. } | Terminator::Par { .. } => {}
    }
    out
}

fn terminator_value_uses(term: &Terminator) -> BTreeSet<Var> {
    let mut out = BTreeSet::new();
    match term {
        Terminator::Goto { args, .. } => {
            for e in args {
                collect_expr_vars(e, &mut out);
            }
        }
        Terminator::If {
            cond,
            then_args,
            else_args,
            ..
        } => {
            collect_expr_vars(cond, &mut out);
            for e in then_args.iter().chain(else_args) {
                collect_expr_vars(e, &mut out);
            }
        }
        Terminator::Match {
            value,
            arms,
            default_args,
            ..
        } => {
            collect_expr_vars(value, &mut out);
            for arm in arms {
                for e in &arm.args {
                    collect_expr_vars(e, &mut out);
                }
            }
            for e in default_args {
                collect_expr_vars(e, &mut out);
            }
        }
        Terminator::Return { value } => collect_expr_vars(value, &mut out),
        Terminator::Par { .. } => {}
    }
    out
}

fn terminator_successor_ids(term: &Terminator) -> Vec<BlockId> {
    match term {
        Terminator::Goto { block, .. } => vec![*block],
        Terminator::If {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        Terminator::Match { arms, default, .. } => {
            let mut v = arms.iter().map(|a| a.block).collect::<Vec<_>>();
            if let Some(d) = default {
                v.push(*d);
            }
            v
        }
        Terminator::Return { .. } => vec![],
        Terminator::Par { branches, join } => {
            let mut v = branches.clone();
            v.push(*join);
            v
        }
    }
}

fn terminator_edges_with_args(term: &Terminator) -> Vec<(BlockId, &[Expr])> {
    match term {
        Terminator::Goto { block, args } => vec![(*block, args.as_slice())],
        Terminator::If {
            then_block,
            then_args,
            else_block,
            else_args,
            ..
        } => vec![
            (*then_block, then_args.as_slice()),
            (*else_block, else_args.as_slice()),
        ],
        Terminator::Match {
            arms,
            default,
            default_args,
            ..
        } => {
            let mut v = arms
                .iter()
                .map(|a| (a.block, a.args.as_slice()))
                .collect::<Vec<_>>();
            if let Some(d) = default {
                v.push((*d, default_args.as_slice()));
            }
            v
        }
        Terminator::Return { .. } => vec![],
        Terminator::Par { branches, join } => {
            let mut v = branches.iter().map(|b| (*b, &[][..])).collect::<Vec<_>>();
            v.push((*join, &[][..]));
            v
        }
    }
}

fn collect_prompt_vars(prompt: &PromptRef, out: &mut BTreeSet<Var>) {
    match prompt {
        PromptRef::Var(v) | PromptRef::PromptIrVar(v) => {
            out.insert(v.clone());
        }
        PromptRef::Inline(_) | PromptRef::PromptIr(_) => {}
    }
}
fn collect_eval_request_vars(req: &EvalRequest, out: &mut BTreeSet<Var>) {
    match req {
        EvalRequest::Shell { command } => collect_expr_vars(command, out),
        EvalRequest::Argv { argv } => {
            for arg in argv {
                collect_expr_vars(arg, out);
            }
        }
    }
}

fn collect_expr_vars(expr: &Expr, out: &mut BTreeSet<Var>) {
    match expr {
        Expr::Value(_) => {}
        Expr::Var(v) => {
            out.insert(v.clone());
        }
        Expr::Field { base, .. }
        | Expr::Len { base }
        | Expr::IsEmpty { base }
        | Expr::HasPendingToolCalls { base } => {
            out.insert(base.clone());
        }
        Expr::FieldOr { base, default, .. } => {
            out.insert(base.clone());
            collect_expr_vars(default, out);
        }
        Expr::StringOr { value, default } => {
            collect_expr_vars(value, out);
            collect_expr_vars(default, out);
        }
        Expr::If {
            cond,
            then_value,
            else_value,
        } => {
            collect_expr_vars(cond, out);
            collect_expr_vars(then_value, out);
            collect_expr_vars(else_value, out);
        }
        Expr::Index { base, index } => {
            out.insert(base.clone());
            collect_expr_vars(index, out);
        }
        Expr::Eq { left, right }
        | Expr::Lt { left, right }
        | Expr::Or { left, right }
        | Expr::And { left, right }
        | Expr::Add { left, right }
        | Expr::Sub { left, right } => {
            collect_expr_vars(left, out);
            collect_expr_vars(right, out);
        }
        Expr::Push { base, value } => {
            out.insert(base.clone());
            collect_expr_vars(value, out);
        }
        Expr::JsonParse { value }
        | Expr::JsonParseOr { value, default: _ }
        | Expr::ToString { value } => {
            collect_expr_vars(value, out);
            if let Expr::JsonParseOr { default, .. } = expr {
                collect_expr_vars(default, out);
            }
        }
        Expr::Array(items) => {
            for e in items {
                collect_expr_vars(e, out);
            }
        }
        Expr::Object(fields) => {
            for e in fields.values() {
                collect_expr_vars(e, out);
            }
        }
    }
}

fn rename_instr_uses(instr: &Instr, env: &BTreeMap<Var, Var>) -> Result<Instr> {
    Ok(match instr {
        Instr::Let { out, expr } => Instr::Let {
            out: out.clone(),
            expr: rename_expr(expr, env)?,
        },
        Instr::Infer {
            out,
            model,
            prompt,
            policy,
        } => Instr::Infer {
            out: out.clone(),
            model: rename_expr(model, env)?,
            prompt: rename_prompt(prompt, env)?,
            policy: policy.clone(),
        },
        Instr::Eval {
            out,
            request,
            policy,
        } => Instr::Eval {
            out: out.clone(),
            request: rename_eval_request(request, env)?,
            policy: policy.clone(),
        },
        Instr::Emit { event } => Instr::Emit {
            event: rename_expr(event, env)?,
        },
        Instr::Retrieve {
            out,
            query,
            kind,
            max_bytes,
            policy,
        } => Instr::Retrieve {
            out: out.clone(),
            query: rename_expr(query, env)?,
            kind: *kind,
            max_bytes: *max_bytes,
            policy: *policy,
        },
        Instr::Store {
            out,
            sink,
            op,
            id,
            item,
            policy,
        } => Instr::Store {
            out: out.clone(),
            sink: rename_expr(sink, env)?,
            op: *op,
            id: id.as_ref().map(|e| rename_expr(e, env)).transpose()?,
            item: rename_expr(item, env)?,
            policy: *policy,
        },
        Instr::Tool {
            out,
            name,
            arguments,
            policy,
        } => Instr::Tool {
            out: out.clone(),
            name: name.clone(),
            arguments: rename_expr(arguments, env)?,
            policy: *policy,
        },
    })
}

fn rename_terminator_uses(term: &Terminator, env: &BTreeMap<Var, Var>) -> Result<Terminator> {
    Ok(match term {
        Terminator::Goto { block, args } => Terminator::Goto {
            block: *block,
            args: rename_exprs(args, env)?,
        },
        Terminator::If {
            cond,
            then_block,
            then_args,
            else_block,
            else_args,
        } => Terminator::If {
            cond: rename_expr(cond, env)?,
            then_block: *then_block,
            then_args: rename_exprs(then_args, env)?,
            else_block: *else_block,
            else_args: rename_exprs(else_args, env)?,
        },
        Terminator::Match {
            value,
            arms,
            default,
            default_args,
        } => Terminator::Match {
            value: rename_expr(value, env)?,
            arms: arms
                .iter()
                .map(|a| {
                    Ok(MatchArm {
                        pattern: a.pattern.clone(),
                        block: a.block,
                        args: rename_exprs(&a.args, env)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            default: *default,
            default_args: rename_exprs(default_args, env)?,
        },
        Terminator::Return { value } => Terminator::Return {
            value: rename_expr(value, env)?,
        },
        Terminator::Par { branches, join } => Terminator::Par {
            branches: branches.clone(),
            join: *join,
        },
    })
}

fn rename_exprs(exprs: &[Expr], env: &BTreeMap<Var, Var>) -> Result<Vec<Expr>> {
    exprs.iter().map(|e| rename_expr(e, env)).collect()
}
fn rename_prompt(prompt: &PromptRef, env: &BTreeMap<Var, Var>) -> Result<PromptRef> {
    Ok(match prompt {
        PromptRef::Inline(v) => PromptRef::Inline(v.clone()),
        PromptRef::Var(v) => PromptRef::Var(rename_var(v, env)?),
        PromptRef::PromptIr(p) => PromptRef::PromptIr(p.clone()),
        PromptRef::PromptIrVar(v) => PromptRef::PromptIrVar(rename_var(v, env)?),
    })
}
fn rename_eval_request(req: &EvalRequest, env: &BTreeMap<Var, Var>) -> Result<EvalRequest> {
    Ok(match req {
        EvalRequest::Shell { command } => EvalRequest::Shell {
            command: rename_expr(command, env)?,
        },
        EvalRequest::Argv { argv } => EvalRequest::Argv {
            argv: rename_exprs(argv, env)?,
        },
    })
}

fn rename_expr(expr: &Expr, env: &BTreeMap<Var, Var>) -> Result<Expr> {
    Ok(match expr {
        Expr::Value(v) => Expr::Value(v.clone()),
        Expr::Var(v) => Expr::Var(rename_var(v, env)?),
        Expr::Field { base, field } => Expr::Field {
            base: rename_var(base, env)?,
            field: field.clone(),
        },
        Expr::FieldOr {
            base,
            field,
            default,
        } => Expr::FieldOr {
            base: rename_var(base, env)?,
            field: field.clone(),
            default: Box::new(rename_expr(default, env)?),
        },
        Expr::StringOr { value, default } => Expr::StringOr {
            value: Box::new(rename_expr(value, env)?),
            default: Box::new(rename_expr(default, env)?),
        },
        Expr::If {
            cond,
            then_value,
            else_value,
        } => Expr::If {
            cond: Box::new(rename_expr(cond, env)?),
            then_value: Box::new(rename_expr(then_value, env)?),
            else_value: Box::new(rename_expr(else_value, env)?),
        },
        Expr::Index { base, index } => Expr::Index {
            base: rename_var(base, env)?,
            index: Box::new(rename_expr(index, env)?),
        },
        Expr::Len { base } => Expr::Len {
            base: rename_var(base, env)?,
        },
        Expr::IsEmpty { base } => Expr::IsEmpty {
            base: rename_var(base, env)?,
        },
        Expr::Eq { left, right } => Expr::Eq {
            left: Box::new(rename_expr(left, env)?),
            right: Box::new(rename_expr(right, env)?),
        },
        Expr::Lt { left, right } => Expr::Lt {
            left: Box::new(rename_expr(left, env)?),
            right: Box::new(rename_expr(right, env)?),
        },
        Expr::Or { left, right } => Expr::Or {
            left: Box::new(rename_expr(left, env)?),
            right: Box::new(rename_expr(right, env)?),
        },
        Expr::And { left, right } => Expr::And {
            left: Box::new(rename_expr(left, env)?),
            right: Box::new(rename_expr(right, env)?),
        },
        Expr::HasPendingToolCalls { base } => Expr::HasPendingToolCalls {
            base: rename_var(base, env)?,
        },
        Expr::Add { left, right } => Expr::Add {
            left: Box::new(rename_expr(left, env)?),
            right: Box::new(rename_expr(right, env)?),
        },
        Expr::Sub { left, right } => Expr::Sub {
            left: Box::new(rename_expr(left, env)?),
            right: Box::new(rename_expr(right, env)?),
        },
        Expr::Push { base, value } => Expr::Push {
            base: rename_var(base, env)?,
            value: Box::new(rename_expr(value, env)?),
        },
        Expr::JsonParse { value } => Expr::JsonParse {
            value: Box::new(rename_expr(value, env)?),
        },
        Expr::JsonParseOr { value, default } => Expr::JsonParseOr {
            value: Box::new(rename_expr(value, env)?),
            default: Box::new(rename_expr(default, env)?),
        },
        Expr::ToString { value } => Expr::ToString {
            value: Box::new(rename_expr(value, env)?),
        },
        Expr::Array(items) => Expr::Array(rename_exprs(items, env)?),
        Expr::Object(fields) => Expr::Object(
            fields
                .iter()
                .map(|(k, v)| Ok((k.clone(), rename_expr(v, env)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        ),
    })
}

fn map_terminator_blocks(term: &Terminator, map: &BTreeMap<BlockId, BlockId>) -> Terminator {
    match term {
        Terminator::Goto { block, args } => Terminator::Goto {
            block: map[block],
            args: args.clone(),
        },
        Terminator::If {
            cond,
            then_block,
            then_args,
            else_block,
            else_args,
        } => Terminator::If {
            cond: cond.clone(),
            then_block: map[then_block],
            then_args: then_args.clone(),
            else_block: map[else_block],
            else_args: else_args.clone(),
        },
        Terminator::Match {
            value,
            arms,
            default,
            default_args,
        } => Terminator::Match {
            value: value.clone(),
            arms: arms
                .iter()
                .map(|a| MatchArm {
                    pattern: a.pattern.clone(),
                    block: map[&a.block],
                    args: a.args.clone(),
                })
                .collect(),
            default: default.map(|d| map[&d]),
            default_args: default_args.clone(),
        },
        Terminator::Return { value } => Terminator::Return {
            value: value.clone(),
        },
        Terminator::Par { branches, join } => Terminator::Par {
            branches: branches.iter().map(|b| map[b]).collect(),
            join: map[join],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn loop_branch_program() -> Program {
        let blocks = BTreeMap::from([
            (
                BlockId(10),
                Block {
                    params: vec![],
                    instructions: vec![Instr::Let {
                        out: Var("n".into()),
                        expr: Expr::Value(json!(0)),
                    }],
                    terminator: Terminator::Goto {
                        block: BlockId(20),
                        args: vec![],
                    },
                },
            ),
            (
                BlockId(20),
                Block {
                    params: vec![],
                    instructions: vec![Instr::Let {
                        out: Var("done".into()),
                        expr: Expr::Lt {
                            left: Box::new(Expr::Value(json!(1))),
                            right: Box::new(Expr::Var(Var("n".into()))),
                        },
                    }],
                    terminator: Terminator::If {
                        cond: Expr::Var(Var("done".into())),
                        then_block: BlockId(40),
                        then_args: vec![],
                        else_block: BlockId(30),
                        else_args: vec![],
                    },
                },
            ),
            (
                BlockId(30),
                Block {
                    params: vec![],
                    instructions: vec![Instr::Let {
                        out: Var("next".into()),
                        expr: Expr::Add {
                            left: Box::new(Expr::Var(Var("n".into()))),
                            right: Box::new(Expr::Value(json!(1))),
                        },
                    }],
                    terminator: Terminator::Goto {
                        block: BlockId(20),
                        args: vec![],
                    },
                },
            ),
            (
                BlockId(40),
                Block {
                    params: vec![],
                    instructions: vec![Instr::Let {
                        out: Var("answer".into()),
                        expr: Expr::Add {
                            left: Box::new(Expr::Var(Var("n".into()))),
                            right: Box::new(Expr::Value(json!(40))),
                        },
                    }],
                    terminator: Terminator::Return {
                        value: Expr::Var(Var("answer".into())),
                    },
                },
            ),
        ]);
        Program {
            id: ProgramId("loop".into()),
            entry: BlockId(10),
            blocks,
        }
    }

    #[test]
    fn normalize_is_idempotent_and_strict_ssa() {
        let p = loop_branch_program();
        let n1 = normalize_program(&p).unwrap();
        let n2 = normalize_program(&n1).unwrap();
        assert_eq!(n1, n2);
        validate_strict_ssa_program(&n1).unwrap();
    }

    #[test]
    fn alpha_equivalent_programs_normalize_and_hash_equal() {
        let mut p1 = Program {
            id: ProgramId("a".into()),
            entry: BlockId(0),
            blocks: BTreeMap::from([(
                BlockId(0),
                Block {
                    params: vec![],
                    instructions: vec![Instr::Let {
                        out: Var("x".into()),
                        expr: Expr::Value(json!(1)),
                    }],
                    terminator: Terminator::Return {
                        value: Expr::Var(Var("x".into())),
                    },
                },
            )]),
        };
        let p2 = Program {
            id: ProgramId("b".into()),
            entry: BlockId(0),
            blocks: BTreeMap::from([(
                BlockId(0),
                Block {
                    params: vec![],
                    instructions: vec![Instr::Let {
                        out: Var("renamed".into()),
                        expr: Expr::Value(json!(1)),
                    }],
                    terminator: Terminator::Return {
                        value: Expr::Var(Var("renamed".into())),
                    },
                },
            )]),
        };
        p1.id = ProgramId("different".into());
        assert_eq!(
            normalize_program(&p1).unwrap(),
            normalize_program(&p2).unwrap()
        );
        assert_eq!(
            canonical_program_hash(&p1).unwrap(),
            canonical_program_hash(&p2).unwrap()
        );
    }

    #[test]
    fn block_renumbering_does_not_change_hash() {
        let p = loop_branch_program();
        let mut map = BTreeMap::new();
        map.insert(BlockId(10), BlockId(300));
        map.insert(BlockId(20), BlockId(100));
        map.insert(BlockId(30), BlockId(400));
        map.insert(BlockId(40), BlockId(200));
        let mut blocks = BTreeMap::new();
        for (old, block) in &p.blocks {
            let mut b = block.clone();
            b.terminator = map_terminator_blocks(&b.terminator, &map);
            blocks.insert(map[old], b);
        }
        let q = Program {
            id: ProgramId("renumbered".into()),
            entry: map[&p.entry],
            blocks,
        };
        assert_eq!(
            canonical_program_hash(&p).unwrap(),
            canonical_program_hash(&q).unwrap()
        );
    }

    #[test]
    fn invalid_program_errors() {
        let p = Program {
            id: ProgramId("bad".into()),
            entry: BlockId(0),
            blocks: BTreeMap::new(),
        };
        assert!(normalize_program(&p).is_err());
    }
}
