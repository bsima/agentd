use crate::interpreter::{
    hydrate_infer_prompt, millis_u64, prompt_preview, response_preview, run_eval, SeqConfig,
};
use crate::ir::{
    effect_location, program_hash, validate_program, BlockId, DynamicPath, EffectKind,
    EffectLocation, EffectSite, EvalRequest, Expr, Instr, Machine, MatchArm, Pattern, ProgramHash,
    PromptRef, Terminator, Var,
};
use crate::op::{ChatMessage, Model, Prompt};
use crate::trace::Event;
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct IrReplayTrace {
    infer_calls: BTreeMap<String, IrInferCall>,
    infer_results: BTreeMap<String, crate::op::Response>,
    eval_calls: BTreeMap<String, IrEvalCall>,
    eval_results: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
struct IrInferCall {
    location: EffectLocation,
    model: String,
}

#[derive(Debug, Clone, PartialEq)]
struct IrEvalCall {
    location: EffectLocation,
    command: String,
}

impl IrReplayTrace {
    pub async fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let events = crate::trace::TraceLogger::read_events(path).await?;
        Self::from_events(&events)
    }

    pub fn from_events(events: &[Event]) -> Result<Self> {
        let mut replay = Self::default();
        let mut last_location: Option<EffectLocation> = None;
        let mut last_infer_id: Option<String> = None;
        let mut last_eval_id: Option<String> = None;

        for event in events {
            match event {
                Event::Custom { name, data, .. } if name == "ir_effect" => {
                    last_location = Some(serde_json::from_value(data.clone())?);
                }
                Event::InferCall { model, .. } => {
                    let location = take_location(&mut last_location, EffectKind::Infer)?;
                    let effect_id = location.effect_id.0.clone();
                    replay.infer_calls.insert(
                        effect_id.clone(),
                        IrInferCall {
                            location,
                            model: model.clone(),
                        },
                    );
                    last_infer_id = Some(effect_id);
                }
                Event::InferResult {
                    response: Some(response),
                    ..
                } => {
                    if let Some(effect_id) = last_infer_id.take() {
                        replay.infer_results.insert(effect_id, response.clone());
                    }
                }
                Event::EvalCall { command, .. } => {
                    let location = take_location(&mut last_location, EffectKind::Eval)?;
                    let effect_id = location.effect_id.0.clone();
                    replay.eval_calls.insert(
                        effect_id.clone(),
                        IrEvalCall {
                            location,
                            command: command.clone(),
                        },
                    );
                    last_eval_id = Some(effect_id);
                }
                Event::EvalResult { result, .. } => {
                    if let Some(effect_id) = last_eval_id.take() {
                        replay.eval_results.insert(effect_id, result.clone());
                    }
                }
                _ => {}
            }
        }
        Ok(replay)
    }

    fn infer_result(&self, location: &EffectLocation, model: &str) -> Result<crate::op::Response> {
        let effect_id = &location.effect_id.0;
        let call = self.infer_calls.get(effect_id).ok_or_else(|| {
            anyhow!(
                "AgentIR replay missing InferCall for effect {} at block {:?} instruction {}",
                effect_id,
                location.site.block,
                location.site.instruction_index
            )
        })?;
        if call.model != model {
            return Err(anyhow!(
                "AgentIR replay diverged at effect {}: expected Infer model {:?} at block {:?} instruction {}, observed {:?}",
                effect_id,
                call.model,
                call.location.site.block,
                call.location.site.instruction_index,
                model
            ));
        }
        self.infer_results
            .get(effect_id)
            .cloned()
            .ok_or_else(|| anyhow!("AgentIR replay missing InferResult for effect {effect_id}"))
    }

    fn eval_result(&self, location: &EffectLocation, command: &str) -> Result<Value> {
        let effect_id = &location.effect_id.0;
        let call = self.eval_calls.get(effect_id).ok_or_else(|| {
            anyhow!(
                "AgentIR replay missing EvalCall for effect {} at block {:?} instruction {}",
                effect_id,
                location.site.block,
                location.site.instruction_index
            )
        })?;
        if call.command != command {
            return Err(anyhow!(
                "AgentIR replay diverged at effect {}: expected Eval command {:?} at block {:?} instruction {}, observed {:?}",
                effect_id,
                call.command,
                call.location.site.block,
                call.location.site.instruction_index,
                command
            ));
        }
        self.eval_results
            .get(effect_id)
            .cloned()
            .ok_or_else(|| anyhow!("AgentIR replay missing EvalResult for effect {effect_id}"))
    }
}

fn take_location(
    location: &mut Option<EffectLocation>,
    expected: EffectKind,
) -> Result<EffectLocation> {
    let location = location.take().ok_or_else(|| {
        anyhow!("AgentIR replay trace missing ir_effect metadata before {expected:?}")
    })?;
    if location.kind != expected {
        return Err(anyhow!(
            "AgentIR replay expected {expected:?} metadata, got {:?} at block {:?} instruction {}",
            location.kind,
            location.site.block,
            location.site.instruction_index
        ));
    }
    Ok(location)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IrCheckpoint {
    pub machine: Machine,
    pub store: InMemoryStore,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IrStepOutcome {
    Complete { value: Value, machine: Machine },
    Suspended { checkpoint: IrCheckpoint },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InMemoryStore {
    values: BTreeMap<String, Value>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Value {
        self.values.get(key).cloned().unwrap_or(Value::Null)
    }

    pub fn put(&mut self, key: impl Into<String>, value: Value) {
        self.values.insert(key.into(), value);
    }
}

pub async fn run_ir_sequential(config: &SeqConfig, machine: Machine) -> Result<(Value, Machine)> {
    let mut store = InMemoryStore::new();
    run_ir_sequential_with_store(config, machine, &mut store).await
}

pub async fn run_ir_sequential_with_store(
    config: &SeqConfig,
    machine: Machine,
    store: &mut InMemoryStore,
) -> Result<(Value, Machine)> {
    match run_ir_steps_with_store_and_replay(config, machine, store, None, None).await? {
        IrStepOutcome::Complete { value, machine } => Ok((value, machine)),
        IrStepOutcome::Suspended { .. } => unreachable!("no instruction limit was set"),
    }
}

pub async fn run_ir_sequential_with_store_and_replay(
    config: &SeqConfig,
    machine: Machine,
    store: &mut InMemoryStore,
    ir_replay: Option<&IrReplayTrace>,
) -> Result<(Value, Machine)> {
    match run_ir_steps_with_store_and_replay(config, machine, store, ir_replay, None).await? {
        IrStepOutcome::Complete { value, machine } => Ok((value, machine)),
        IrStepOutcome::Suspended { .. } => unreachable!("no instruction limit was set"),
    }
}

pub async fn run_ir_steps(
    config: &SeqConfig,
    machine: Machine,
    max_instructions: usize,
) -> Result<IrStepOutcome> {
    let mut store = InMemoryStore::new();
    run_ir_steps_with_store_and_replay(config, machine, &mut store, None, Some(max_instructions))
        .await
}

pub async fn run_ir_steps_with_store_and_replay(
    config: &SeqConfig,
    mut machine: Machine,
    store: &mut InMemoryStore,
    ir_replay: Option<&IrReplayTrace>,
    max_instructions: Option<usize>,
) -> Result<IrStepOutcome> {
    validate_program(&machine.program)?;
    let program_hash = program_hash(&machine.program)?;
    let mut site_visits = HashMap::<EffectSite, u64>::new();
    let mut instructions_executed = 0usize;

    loop {
        if max_instructions.is_some_and(|max| instructions_executed >= max) {
            return Ok(IrStepOutcome::Suspended {
                checkpoint: IrCheckpoint {
                    machine,
                    store: store.clone(),
                },
            });
        }
        let block = machine
            .program
            .blocks
            .get(&machine.block)
            .with_context(|| format!("unknown AgentIR block {:?}", machine.block))?
            .clone();

        if machine.pc < block.instructions.len() {
            let site = EffectSite {
                block: machine.block,
                instruction_index: machine.pc,
            };
            let dynamic_path = DynamicPath::with_visit(site, next_visit(&mut site_visits, site));
            let instr = block.instructions[machine.pc].clone();
            execute_instr(
                config,
                &mut machine,
                store,
                &program_hash,
                site,
                dynamic_path,
                ir_replay,
                instr,
            )
            .await?;
            machine.pc += 1;
            instructions_executed += 1;
            continue;
        }

        match block.terminator {
            Terminator::Return { value } => {
                let value = eval_expr(&machine.env, &value)?;
                return Ok(IrStepOutcome::Complete { value, machine });
            }
            Terminator::Goto { block, args } => {
                goto_block(&mut machine, block, args).await?;
            }
            Terminator::If {
                cond,
                then_block,
                else_block,
            } => {
                let cond = eval_expr(&machine.env, &cond)?;
                let target = match cond {
                    Value::Bool(true) => then_block,
                    Value::Bool(false) => else_block,
                    other => return Err(anyhow!("AgentIR If condition must be bool, got {other}")),
                };
                branch_to_block(&mut machine, target).await?;
            }
            Terminator::Match {
                value,
                arms,
                default,
            } => {
                let value = eval_expr(&machine.env, &value)?;
                let target = match_match_arms(&value, &arms).or(default).ok_or_else(|| {
                    anyhow!("AgentIR Match had no matching arm and no default for {value}")
                })?;
                branch_to_block(&mut machine, target).await?;
            }
            Terminator::Par { .. } => {
                return Err(anyhow!(
                    "AgentIR Par terminator is not implemented in run_ir_sequential yet"
                ));
            }
        }
    }
}

async fn execute_instr(
    config: &SeqConfig,
    machine: &mut Machine,
    store: &mut InMemoryStore,
    program_hash: &ProgramHash,
    site: EffectSite,
    dynamic_path: DynamicPath,
    ir_replay: Option<&IrReplayTrace>,
    instr: Instr,
) -> Result<()> {
    match instr {
        Instr::Let { out, expr } => {
            let value = eval_expr(&machine.env, &expr)?;
            machine.env.insert(out, value);
        }
        Instr::Infer {
            out,
            model,
            prompt,
            policy: _,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Infer,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let model = string_expr(&machine.env, &model, "Infer.model")?;
            let prompt = resolve_prompt(&machine.env, prompt)?;
            let prompt = hydrate_infer_prompt(config, &Value::Null, prompt).await?;
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::InferCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    model: model.clone(),
                    prompt: Some(prompt.clone()),
                    prompt_preview: prompt_preview(&prompt),
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            let response = match ir_replay {
                Some(replay) => replay.infer_result(&location, &model)?,
                None => match &config.replay {
                    Some(replay) => replay.infer_result(op_id, &model)?,
                    None => {
                        config
                            .provider
                            .chat(&Model(model), &ir_tool_specs(), &prompt)
                            .await?
                    }
                },
            };
            config
                .trace
                .emit(&Event::InferResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    response: Some(response.clone()),
                    response_preview: response_preview(&response),
                    tokens: response.tokens,
                    duration_ms: millis_u64(started.elapsed()),
                    timestamp: Utc::now(),
                })
                .await?;
            machine.env.insert(out, serde_json::to_value(response)?);
        }
        Instr::Eval {
            out,
            request,
            policy: _,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Eval,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let command = match request {
                EvalRequest::Shell { command } => {
                    string_expr(&machine.env, &command, "Eval.command")?
                }
            };
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::EvalCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    command: command.clone(),
                    cwd: config
                        .eval
                        .cwd
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    env_policy: config.eval.env.label(),
                    timeout_ms: millis_u64(config.eval.timeout),
                    timestamp: Utc::now(),
                })
                .await?;
            let result = match ir_replay {
                Some(replay) => replay.eval_result(&location, &command)?,
                None => match &config.replay {
                    Some(replay) => replay.eval_result(op_id, &command)?,
                    None => run_eval(&config.eval, &command).await?,
                },
            };
            let truncated_stdout = result
                .get("stdout_truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let truncated_stderr = result
                .get("stderr_truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let duration_ms = result
                .get("duration_ms")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            config
                .trace
                .emit(&Event::EvalResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    command,
                    result: result.clone(),
                    duration_ms,
                    truncated_stdout,
                    truncated_stderr,
                    timestamp: Utc::now(),
                })
                .await?;
            machine.env.insert(out, result);
        }
        Instr::Get { out, key } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Get,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let key = string_expr(&machine.env, &key, "Get.key")?;
            let value = store.get(&key);
            machine.env.insert(out, value);
        }
        Instr::Put { key, value } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Put,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let key = string_expr(&machine.env, &key, "Put.key")?;
            let value = eval_expr(&machine.env, &value)?;
            store.put(key, value);
        }
        Instr::Emit { event } => {
            let location =
                effect_location(program_hash.clone(), EffectKind::Emit, site, dynamic_path)?;
            emit_ir_effect(config, &location).await?;
            let value = eval_expr(&machine.env, &event)?;
            let event: Event =
                serde_json::from_value(value).context("decoding AgentIR Emit event")?;
            config.trace.emit(&event).await?;
        }
    }
    Ok(())
}

async fn branch_to_block(machine: &mut Machine, block_id: BlockId) -> Result<()> {
    let target = machine
        .program
        .blocks
        .get(&block_id)
        .with_context(|| format!("unknown AgentIR block {block_id:?}"))?;
    if !target.params.is_empty() {
        return Err(anyhow!(
            "AgentIR branch to {:?} expected target with no params, got {}",
            block_id,
            target.params.len()
        ));
    }
    machine.block = block_id;
    machine.pc = 0;
    Ok(())
}

async fn goto_block(machine: &mut Machine, block_id: BlockId, args: Vec<Expr>) -> Result<()> {
    let target = machine
        .program
        .blocks
        .get(&block_id)
        .with_context(|| format!("unknown AgentIR block {block_id:?}"))?;
    if target.params.len() != args.len() {
        return Err(anyhow!(
            "AgentIR Goto to {:?} expected {} args, got {}",
            block_id,
            target.params.len(),
            args.len()
        ));
    }
    let mut env = machine.env.clone();
    for (param, arg) in target.params.iter().cloned().zip(args) {
        env.insert(param, eval_expr(&machine.env, &arg)?);
    }
    machine.block = block_id;
    machine.pc = 0;
    machine.env = env;
    Ok(())
}

fn ir_tool_specs() -> Vec<crate::provider::ToolSpec> {
    vec![
        crate::provider::ToolSpec {
            kind: "function".into(),
            function: crate::provider::ToolFunctionSpec {
                name: "shell".into(),
                description: "Execute a command string using the configured shell.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "command": { "type": "string" } },
                    "required": ["command"]
                }),
            },
        },
        crate::provider::ToolSpec {
            kind: "function".into(),
            function: crate::provider::ToolFunctionSpec {
                name: "infer".into(),
                description: "Ask the model a focused sub-question and return its response.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "model": { "type": "string" },
                        "prompt": { "type": "string" }
                    },
                    "required": ["model", "prompt"]
                }),
            },
        },
    ]
}

fn next_visit(site_visits: &mut HashMap<EffectSite, u64>, site: EffectSite) -> u64 {
    let visit = site_visits.entry(site).or_insert(0);
    let current = *visit;
    *visit += 1;
    current
}

async fn emit_ir_effect(config: &SeqConfig, location: &EffectLocation) -> Result<()> {
    config
        .trace
        .emit(&Event::Custom {
            run_id: config.trace.run_id().into(),
            name: "ir_effect".into(),
            data: serde_json::to_value(location)?,
            timestamp: Utc::now(),
        })
        .await
}

fn eval_expr(env: &BTreeMap<Var, Value>, expr: &Expr) -> Result<Value> {
    match expr {
        Expr::Value(value) => Ok(value.clone()),
        Expr::Var(var) => env
            .get(var)
            .cloned()
            .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", var)),
        Expr::Field { base, field } => {
            let value = env
                .get(base)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", base))?;
            value
                .get(field)
                .cloned()
                .ok_or_else(|| anyhow!("AgentIR field {field:?} not found on {value}"))
        }
        Expr::Index { base, index } => {
            let value = env
                .get(base)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", base))?;
            let index = usize_expr(env, index, "Index.index")?;
            value
                .get(index)
                .cloned()
                .ok_or_else(|| anyhow!("AgentIR index {index} not found on {value}"))
        }
        Expr::Len { base } => {
            let value = env
                .get(base)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", base))?;
            match value {
                Value::Array(items) => Ok(Value::Number(items.len().into())),
                Value::String(text) => Ok(Value::Number(text.chars().count().into())),
                other => Err(anyhow!("AgentIR Len expected array or string, got {other}")),
            }
        }
        Expr::IsEmpty { base } => {
            let value = env
                .get(base)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", base))?;
            match value {
                Value::Array(items) => Ok(Value::Bool(items.is_empty())),
                Value::String(text) => Ok(Value::Bool(text.is_empty())),
                other => Err(anyhow!(
                    "AgentIR IsEmpty expected array or string, got {other}"
                )),
            }
        }
        Expr::Eq { left, right } => {
            Ok(Value::Bool(eval_expr(env, left)? == eval_expr(env, right)?))
        }
        Expr::Lt { left, right } => Ok(Value::Bool(
            number_expr(env, left, "Lt.left")? < number_expr(env, right, "Lt.right")?,
        )),
        Expr::Or { left, right } => Ok(Value::Bool(
            bool_expr(env, left, "Or.left")? || bool_expr(env, right, "Or.right")?,
        )),
        Expr::Add { left, right } => Ok(Value::Number(
            (number_expr(env, left, "Add.left")? + number_expr(env, right, "Add.right")?).into(),
        )),
        Expr::Sub { left, right } => Ok(Value::Number(
            (number_expr(env, left, "Sub.left")? - number_expr(env, right, "Sub.right")?).into(),
        )),
        Expr::Push { base, value } => {
            let array = env
                .get(base)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", base))?;
            let mut array = array
                .as_array()
                .cloned()
                .ok_or_else(|| anyhow!("AgentIR Push expected array, got {array}"))?;
            array.push(eval_expr(env, value)?);
            Ok(Value::Array(array))
        }
        Expr::JsonParse { value } => {
            let text = string_expr(env, value, "JsonParse.value")?;
            serde_json::from_str(&text).context("AgentIR JsonParse failed")
        }
        Expr::ToString { value } => {
            let value = eval_expr(env, value)?;
            Ok(Value::String(value.to_string()))
        }
        Expr::Array(items) => items
            .iter()
            .map(|item| eval_expr(env, item))
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        Expr::Object(fields) => {
            let mut object = serde_json::Map::new();
            for (key, expr) in fields {
                object.insert(key.clone(), eval_expr(env, expr)?);
            }
            Ok(Value::Object(object))
        }
    }
}

fn string_expr(env: &BTreeMap<Var, Value>, expr: &Expr, label: &str) -> Result<String> {
    match eval_expr(env, expr)? {
        Value::String(value) => Ok(value),
        other => Err(anyhow!("AgentIR {label} must be string, got {other}")),
    }
}

fn bool_expr(env: &BTreeMap<Var, Value>, expr: &Expr, label: &str) -> Result<bool> {
    match eval_expr(env, expr)? {
        Value::Bool(value) => Ok(value),
        other => Err(anyhow!("AgentIR {label} must be bool, got {other}")),
    }
}

fn number_expr(env: &BTreeMap<Var, Value>, expr: &Expr, label: &str) -> Result<i64> {
    match eval_expr(env, expr)? {
        Value::Number(value) => value
            .as_i64()
            .ok_or_else(|| anyhow!("AgentIR {label} must be i64-compatible, got {value}")),
        other => Err(anyhow!("AgentIR {label} must be number, got {other}")),
    }
}

fn usize_expr(env: &BTreeMap<Var, Value>, expr: &Expr, label: &str) -> Result<usize> {
    let value = number_expr(env, expr, label)?;
    usize::try_from(value).map_err(|_| anyhow!("AgentIR {label} must be non-negative, got {value}"))
}

fn resolve_prompt(env: &BTreeMap<Var, Value>, prompt: PromptRef) -> Result<Prompt> {
    match prompt {
        PromptRef::Inline(prompt) => Ok(prompt),
        PromptRef::Var(var) => {
            let value = env
                .get(&var)
                .cloned()
                .ok_or_else(|| anyhow!("unknown AgentIR prompt var {:?}", var))?;
            serde_json::from_value::<Vec<ChatMessage>>(value).context("decoding AgentIR prompt")
        }
    }
}

fn match_match_arms(value: &Value, arms: &[MatchArm]) -> Option<BlockId> {
    arms.iter()
        .find(|arm| pattern_matches(value, &arm.pattern))
        .map(|arm| arm.block)
}

fn pattern_matches(value: &Value, pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Null => value.is_null(),
        Pattern::Bool(expected) => value.as_bool() == Some(*expected),
        Pattern::String(expected) => value.as_str() == Some(expected.as_str()),
        Pattern::Number(expected) => value.as_number() == Some(expected),
        Pattern::ObjectField { field, pattern } => value
            .get(field)
            .is_some_and(|value| pattern_matches(value, pattern)),
        Pattern::Any => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::{PassiveHydrationConfig, SourceRegistry};
    use crate::interpreter::{EvalConfig, SeqConfig};
    use crate::op::{Response, ResponseToolCall};
    use crate::provider::{ChatProvider, ToolSpec};
    use crate::trace::TraceLogger;
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    struct MockProvider {
        responses: Mutex<Vec<Response>>,
        prompts: Mutex<Vec<Prompt>>,
    }

    impl MockProvider {
        fn new(mut responses: Vec<Response>) -> Self {
            responses.reverse();
            Self {
                responses: Mutex::new(responses),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn prompts(&self) -> Vec<Prompt> {
            self.prompts.lock().unwrap().clone()
        }

        fn prompt_count(&self) -> usize {
            self.prompts.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        async fn chat(
            &self,
            _model: &Model,
            _tools: &[ToolSpec],
            messages: &[ChatMessage],
        ) -> Result<Response> {
            self.prompts.lock().unwrap().push(messages.to_vec());
            self.responses
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow!("mock provider exhausted"))
        }
    }

    fn response(content: &str) -> Response {
        Response {
            content: content.into(),
            tool_calls: Vec::<ResponseToolCall>::new(),
            tokens: 1,
        }
    }

    fn test_trace() -> TraceLogger {
        let path = std::env::temp_dir().join(format!("agent-ir-test-{}.jsonl", Uuid::new_v4()));
        TraceLogger::new(Uuid::new_v4().to_string(), path)
    }

    fn config(provider: Arc<dyn ChatProvider>) -> SeqConfig {
        SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
        }
    }

    fn config_with_trace(provider: Arc<dyn ChatProvider>, trace: TraceLogger) -> SeqConfig {
        SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace,
            eval: EvalConfig::default(),
            replay: None,
        }
    }

    #[tokio::test]
    async fn ir_runs_infer_then_infer_without_rust_continuations() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response("first"),
            response("second"),
        ]));
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![
                    Instr::Infer {
                        out: Var("a".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Inline(vec![ChatMessage::user("first prompt")]),
                        policy: Default::default(),
                    },
                    Instr::Let {
                        out: Var("a_content".into()),
                        expr: Expr::Field {
                            base: Var("a".into()),
                            field: "content".into(),
                        },
                    },
                    Instr::Let {
                        out: Var("second_prompt".into()),
                        expr: Expr::Array(vec![Expr::Object(BTreeMap::from([
                            ("role".into(), Expr::Value(Value::String("user".into()))),
                            ("content".into(), Expr::Var(Var("a_content".into()))),
                        ]))]),
                    },
                    Instr::Infer {
                        out: Var("b".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Var(Var("second_prompt".into())),
                        policy: Default::default(),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Field {
                        base: Var("b".into()),
                        field: "content".into(),
                    },
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("infer-infer".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let (value, _machine) = run_ir_sequential(&config(provider.clone()), machine).await?;

        assert_eq!(value, Value::String("second".into()));
        let prompts = provider.prompts();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[1][0].content.as_deref(), Some("first"));
        Ok(())
    }

    #[tokio::test]
    async fn ir_effect_metadata_is_stable_and_visit_sensitive() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Get {
                    out: Var("a".into()),
                    key: Expr::Value(Value::String("missing".into())),
                }],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("a".into())),
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("effect-ids".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let _ = run_ir_sequential(&config_with_trace(provider, trace), machine).await?;
        let events = TraceLogger::read_events(trace_path).await?;
        let locations = events
            .iter()
            .filter_map(|event| match event {
                Event::Custom { name, data, .. } if name == "ir_effect" => {
                    Some(serde_json::from_value::<EffectLocation>(data.clone()))
                }
                _ => None,
            })
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].kind, EffectKind::Get);
        assert_eq!(locations[0].site.block, BlockId(0));
        assert_eq!(locations[0].site.instruction_index, 0);
        Ok(())
    }

    #[tokio::test]
    async fn ir_replay_uses_stable_effect_ids() -> Result<()> {
        let record_provider = Arc::new(MockProvider::new(vec![response("recorded")]));
        let record_trace = test_trace();
        let record_path = record_trace.path().clone();
        let machine = single_infer_machine("mock");
        let (recorded, _) = run_ir_sequential(
            &config_with_trace(record_provider, record_trace),
            machine.clone(),
        )
        .await?;
        assert_eq!(recorded["content"], Value::String("recorded".into()));

        let replay = IrReplayTrace::load(record_path).await?;
        let replay_provider = Arc::new(MockProvider::new(vec![]));
        let mut store = InMemoryStore::new();
        let (replayed, _) = run_ir_sequential_with_store_and_replay(
            &config(replay_provider.clone()),
            machine,
            &mut store,
            Some(&replay),
        )
        .await?;

        assert_eq!(replayed, recorded);
        assert_eq!(replay_provider.prompt_count(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn ir_replay_divergence_reports_effect_location() -> Result<()> {
        let record_provider = Arc::new(MockProvider::new(vec![response("recorded")]));
        let record_trace = test_trace();
        let record_path = record_trace.path().clone();
        let _ = run_ir_sequential(
            &config_with_trace(record_provider, record_trace),
            single_infer_machine("mock"),
        )
        .await?;
        let replay = IrReplayTrace::load(record_path).await?;
        let mut store = InMemoryStore::new();

        let err = run_ir_sequential_with_store_and_replay(
            &config(Arc::new(MockProvider::new(vec![]))),
            single_infer_machine("other-model"),
            &mut store,
            Some(&replay),
        )
        .await
        .unwrap_err()
        .to_string();

        assert!(err.contains("AgentIR replay missing InferCall"));
        assert!(err.contains("block BlockId(0) instruction 0"));
        Ok(())
    }

    fn single_infer_machine(model: &str) -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Infer {
                    out: Var("response".into()),
                    model: Expr::Value(Value::String(model.into())),
                    prompt: PromptRef::Inline(vec![ChatMessage::user("hello")]),
                    policy: Default::default(),
                }],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("response".into())),
                },
            },
        );
        Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("single-infer".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn ir_checkpoint_resumes_without_replaying_completed_effects() -> Result<()> {
        let first_provider = Arc::new(MockProvider::new(vec![response("first")]));
        let machine = infer_then_infer_machine();
        let outcome = run_ir_steps(&config(first_provider.clone()), machine, 1).await?;
        let checkpoint = match outcome {
            IrStepOutcome::Suspended { checkpoint } => checkpoint,
            IrStepOutcome::Complete { .. } => panic!("expected suspension after one instruction"),
        };
        assert_eq!(first_provider.prompt_count(), 1);

        let encoded = serde_json::to_value(&checkpoint)?;
        let checkpoint: IrCheckpoint = serde_json::from_value(encoded)?;
        let second_provider = Arc::new(MockProvider::new(vec![response("second")]));
        let mut store = checkpoint.store;
        let (value, _machine) = run_ir_sequential_with_store(
            &config(second_provider.clone()),
            checkpoint.machine,
            &mut store,
        )
        .await?;

        assert_eq!(value, Value::String("second".into()));
        assert_eq!(second_provider.prompt_count(), 1);
        Ok(())
    }

    fn infer_then_infer_machine() -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![
                    Instr::Infer {
                        out: Var("a".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Inline(vec![ChatMessage::user("first prompt")]),
                        policy: Default::default(),
                    },
                    Instr::Let {
                        out: Var("a_content".into()),
                        expr: Expr::Field {
                            base: Var("a".into()),
                            field: "content".into(),
                        },
                    },
                    Instr::Let {
                        out: Var("second_prompt".into()),
                        expr: Expr::Array(vec![Expr::Object(BTreeMap::from([
                            ("role".into(), Expr::Value(Value::String("user".into()))),
                            ("content".into(), Expr::Var(Var("a_content".into()))),
                        ]))]),
                    },
                    Instr::Infer {
                        out: Var("b".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Var(Var("second_prompt".into())),
                        policy: Default::default(),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Field {
                        base: Var("b".into()),
                        field: "content".into(),
                    },
                },
            },
        );
        Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("infer-infer".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn ir_get_put_use_interpreter_store_not_machine_state() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![
                    Instr::Put {
                        key: Expr::Value(Value::String("answer".into())),
                        value: Expr::Value(Value::Number(42.into())),
                    },
                    Instr::Get {
                        out: Var("value".into()),
                        key: Expr::Value(Value::String("answer".into())),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("value".into())),
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("store".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };
        let mut store = InMemoryStore::new();

        let (value, _machine) =
            run_ir_sequential_with_store(&config(provider), machine, &mut store).await?;

        assert_eq!(value, Value::Number(42.into()));
        assert_eq!(store.get("answer"), Value::Number(42.into()));
        Ok(())
    }

    #[tokio::test]
    async fn ir_validation_runs_before_effects() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("should-not-run")]));
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Infer {
                    out: Var("response".into()),
                    model: Expr::Value(Value::String("mock".into())),
                    prompt: PromptRef::Inline(vec![ChatMessage::user("do not run")]),
                    policy: Default::default(),
                }],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("missing".into())),
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("invalid".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let err = run_ir_sequential(&config(provider.clone()), machine)
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("used before definition"));
        assert_eq!(provider.prompt_count(), 0);
        Ok(())
    }
}
