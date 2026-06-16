use crate::gc::GcState;
use crate::interpreter::{
    annotate_overflow_failure, catch_overflow_active, collect_for_overflow, hydrate_infer_prompt,
    maybe_collect_prompt, millis_u64, prompt_preview, response_preview, run_eval_with_env,
    SeqConfig, CATCH_OVERFLOW_MAX_CYCLES,
};
use crate::ir::{
    effect_location, program_hash, validate_program, BlockId, DynamicPath, EffectErrorMode,
    EffectKind, EffectLocation, EffectSite, EvalRequest, Expr, Instr, Machine, MatchArm, Pattern,
    ProgramHash, PromptRef, Terminator, Var,
};
use crate::op::{ChatMessage, Model, Prompt};
use crate::prompt_ir::{collect_prompt_ir_sections, compile_prompt_ir, PromptIR};
use crate::trace::Event;
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct IrReplayTrace {
    infer_calls: BTreeMap<String, IrInferCall>,
    infer_results: BTreeMap<String, crate::op::Response>,
    infer_errors: BTreeMap<String, String>,
    eval_calls: BTreeMap<String, IrEvalCall>,
    eval_results: BTreeMap<String, Value>,
    eval_errors: BTreeMap<String, String>,
    retrieve_calls: BTreeMap<String, IrRetrieveCall>,
    retrieve_results: BTreeMap<String, Value>,
    retrieve_errors: BTreeMap<String, String>,
    store_calls: BTreeMap<String, IrStoreCall>,
    store_results: BTreeMap<String, String>,
    store_errors: BTreeMap<String, String>,
}

/// The Retrieve identity recorded for replay-divergence detection. `kind`
/// and `max_bytes` are static program fields (so an edit also changes the
/// program hash and effect id), but recording them makes divergence
/// errors specific rather than a bare "missing call" (t-1182 review).
#[derive(Debug, Clone, PartialEq)]
struct IrRetrieveCall {
    query: String,
    kind: Option<String>,
    max_bytes: Option<usize>,
}

/// The Store identity recorded for replay-divergence detection. `sink` and
/// `id` are dynamic (evaluated from Exprs), so a same-site call computing a
/// different target must be caught here — the payload `content_hash` alone
/// does not cover them.
#[derive(Debug, Clone, PartialEq)]
struct IrStoreCall {
    sink: String,
    op: String,
    id: Option<String>,
    content_hash: String,
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
        let mut last_retrieve_id: Option<String> = None;
        let mut last_store_id: Option<String> = None;

        for event in events {
            match event {
                Event::Custom { name, data, .. } if name == "ir_effect" => {
                    let location: EffectLocation = serde_json::from_value(data.clone())?;
                    last_location = matches!(
                        location.kind,
                        EffectKind::Infer
                            | EffectKind::Eval
                            | EffectKind::Retrieve
                            | EffectKind::Store
                    )
                    .then_some(location);
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
                Event::InferError { error, .. } => {
                    if let Some(effect_id) = last_infer_id.take() {
                        replay.infer_errors.insert(effect_id, error.clone());
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
                Event::EvalError { error, .. } => {
                    if let Some(effect_id) = last_eval_id.take() {
                        replay.eval_errors.insert(effect_id, error.clone());
                    }
                }
                Event::RetrieveCall {
                    query,
                    kind,
                    max_bytes,
                    ..
                } => {
                    let location = take_location(&mut last_location, EffectKind::Retrieve)?;
                    let effect_id = location.effect_id.0.clone();
                    replay.retrieve_calls.insert(
                        effect_id.clone(),
                        IrRetrieveCall {
                            query: query.clone(),
                            kind: kind.clone(),
                            max_bytes: *max_bytes,
                        },
                    );
                    last_retrieve_id = Some(effect_id);
                }
                Event::RetrieveResult { results, .. } => {
                    if let Some(effect_id) = last_retrieve_id.take() {
                        replay.retrieve_results.insert(effect_id, results.clone());
                    }
                }
                Event::RetrieveError { error, .. } => {
                    if let Some(effect_id) = last_retrieve_id.take() {
                        replay.retrieve_errors.insert(effect_id, error.clone());
                    }
                }
                Event::StoreCall {
                    sink,
                    store_op,
                    store_id,
                    content_hash,
                    ..
                } => {
                    let location = take_location(&mut last_location, EffectKind::Store)?;
                    let effect_id = location.effect_id.0.clone();
                    replay.store_calls.insert(
                        effect_id.clone(),
                        IrStoreCall {
                            sink: sink.clone(),
                            op: store_op.clone(),
                            id: store_id.clone(),
                            content_hash: content_hash.clone(),
                        },
                    );
                    last_store_id = Some(effect_id);
                }
                Event::StoreResult { sink_id, .. } => {
                    if let Some(effect_id) = last_store_id.take() {
                        replay.store_results.insert(effect_id, sink_id.clone());
                    }
                }
                Event::StoreError { error, .. } => {
                    if let Some(effect_id) = last_store_id.take() {
                        replay.store_errors.insert(effect_id, error.clone());
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
        if let Some(error) = self.infer_errors.get(effect_id) {
            return Err(anyhow!(
                "AgentIR replaying recorded Infer failure at effect {effect_id}: {error}"
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
        if let Some(error) = self.eval_errors.get(effect_id) {
            return Err(anyhow!(
                "AgentIR replaying recorded Eval failure at effect {effect_id}: {error}"
            ));
        }
        self.eval_results
            .get(effect_id)
            .cloned()
            .ok_or_else(|| anyhow!("AgentIR replay missing EvalResult for effect {effect_id}"))
    }

    fn retrieve_result(&self, location: &EffectLocation, key: &IrRetrieveCall) -> Result<Value> {
        let effect_id = &location.effect_id.0;
        let recorded = self.retrieve_calls.get(effect_id).ok_or_else(|| {
            anyhow!(
                "AgentIR replay missing RetrieveCall for effect {} at block {:?} instruction {}",
                effect_id,
                location.site.block,
                location.site.instruction_index
            )
        })?;
        if recorded != key {
            return Err(anyhow!(
                "AgentIR replay diverged at effect {effect_id}: expected Retrieve {recorded:?}, observed {key:?}"
            ));
        }
        if let Some(error) = self.retrieve_errors.get(effect_id) {
            return Err(anyhow!(
                "AgentIR replaying recorded Retrieve failure at effect {effect_id}: {error}"
            ));
        }
        self.retrieve_results
            .get(effect_id)
            .cloned()
            .ok_or_else(|| anyhow!("AgentIR replay missing RetrieveResult for effect {effect_id}"))
    }

    /// Replay never mutates a sink: the recorded id is returned. The full
    /// Store identity (sink, op, id, payload hash) is checked so a same-site
    /// call computing a different sink/op/id/payload diverges instead of
    /// silently replaying a stale result.
    fn store_result(&self, location: &EffectLocation, key: &IrStoreCall) -> Result<String> {
        let effect_id = &location.effect_id.0;
        let recorded = self.store_calls.get(effect_id).ok_or_else(|| {
            anyhow!(
                "AgentIR replay missing StoreCall for effect {} at block {:?} instruction {}",
                effect_id,
                location.site.block,
                location.site.instruction_index
            )
        })?;
        if recorded != key {
            return Err(anyhow!(
                "AgentIR replay diverged at effect {effect_id}: expected Store {recorded:?}, observed {key:?}"
            ));
        }
        if let Some(error) = self.store_errors.get(effect_id) {
            return Err(anyhow!(
                "AgentIR replaying recorded Store failure at effect {effect_id}: {error}"
            ));
        }
        self.store_results
            .get(effect_id)
            .cloned()
            .ok_or_else(|| anyhow!("AgentIR replay missing StoreResult for effect {effect_id}"))
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

/// Backing store for the IR interpreter's instruction-limit checkpoints
/// (`in_memory_snapshot`). The Get/Put key-value methods were removed with
/// the Get/Put effects (t-1182); session state and retrieval now flow
/// through the passive ChatHistory sink and the Retrieve/Store effects.
#[async_trait]
pub trait IrStore: Send {
    fn in_memory_snapshot(&self) -> Option<InMemoryStore> {
        None
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InMemoryStore {
    values: BTreeMap<String, Value>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_local(&self, key: &str) -> Value {
        self.values.get(key).cloned().unwrap_or(Value::Null)
    }

    pub fn put_local(&mut self, key: impl Into<String>, value: Value) {
        self.values.insert(key.into(), value);
    }
}

#[async_trait]
impl IrStore for InMemoryStore {
    fn in_memory_snapshot(&self) -> Option<InMemoryStore> {
        Some(self.clone())
    }
}

pub async fn run_ir_sequential(config: &SeqConfig, machine: Machine) -> Result<(Value, Machine)> {
    let mut store = InMemoryStore::new();
    run_ir_sequential_with_store(config, machine, &mut store).await
}

pub async fn run_ir_sequential_with_store(
    config: &SeqConfig,
    machine: Machine,
    store: &mut dyn IrStore,
) -> Result<(Value, Machine)> {
    match run_ir_steps_with_store_and_replay(config, machine, store, None, None).await? {
        IrStepOutcome::Complete { value, machine } => Ok((value, machine)),
        IrStepOutcome::Suspended { .. } => unreachable!("no instruction limit was set"),
    }
}

pub async fn run_ir_sequential_with_store_and_replay(
    config: &SeqConfig,
    machine: Machine,
    store: &mut dyn IrStore,
    ir_replay: Option<&IrReplayTrace>,
) -> Result<(Value, Machine)> {
    let mut gc_state = GcState::default();
    run_ir_sequential_with_gc(config, machine, store, ir_replay, &mut gc_state).await
}

/// Like [`run_ir_sequential_with_store_and_replay`], but GC state is owned
/// by the caller so it survives across loop runs. A session turn is one
/// loop run; what GC learned during it — the provider's real context
/// ceiling (`discovered_budget`, t-1151), frame lifecycles, the every-N
/// cadence — is knowledge about the *session*, not the turn. Resetting it
/// per turn made catch-overflow pay one failed provider call per user turn
/// to relearn the same ceiling (t-1162).
pub async fn run_ir_sequential_with_gc(
    config: &SeqConfig,
    machine: Machine,
    store: &mut dyn IrStore,
    ir_replay: Option<&IrReplayTrace>,
    gc_state: &mut GcState,
) -> Result<(Value, Machine)> {
    match run_ir_steps_with_gc(config, machine, store, ir_replay, None, gc_state).await? {
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
    machine: Machine,
    store: &mut dyn IrStore,
    ir_replay: Option<&IrReplayTrace>,
    max_instructions: Option<usize>,
) -> Result<IrStepOutcome> {
    let mut gc_state = GcState::default();
    run_ir_steps_with_gc(
        config,
        machine,
        store,
        ir_replay,
        max_instructions,
        &mut gc_state,
    )
    .await
}

pub async fn run_ir_steps_with_gc(
    config: &SeqConfig,
    mut machine: Machine,
    store: &mut dyn IrStore,
    ir_replay: Option<&IrReplayTrace>,
    max_instructions: Option<usize>,
    gc_state: &mut GcState,
) -> Result<IrStepOutcome> {
    validate_program(&machine.program)?;
    let program_hash = program_hash(&machine.program)?;
    let mut instructions_executed = 0usize;

    loop {
        if max_instructions.is_some_and(|max| instructions_executed >= max) {
            let store = store.in_memory_snapshot().ok_or_else(|| {
                anyhow!("AgentIR instruction-limit checkpoints require an in-memory store snapshot")
            })?;
            return Ok(IrStepOutcome::Suspended {
                checkpoint: IrCheckpoint { machine, store },
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
            let dynamic_path = DynamicPath::with_visit(site, next_visit(&mut machine, site));
            let instr = block.instructions[machine.pc].clone();
            execute_instr(
                config,
                &mut machine,
                &program_hash,
                site,
                dynamic_path,
                ir_replay,
                instr,
                gc_state,
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
        // Block transitions count as steps too: a cycle of blocks with empty
        // instruction lists (pure Goto/If/Match loops) must still hit the
        // instruction limit, or the limit is useless as a watchdog.
        instructions_executed += 1;
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_instr(
    config: &SeqConfig,
    machine: &mut Machine,
    program_hash: &ProgramHash,
    site: EffectSite,
    dynamic_path: DynamicPath,
    ir_replay: Option<&IrReplayTrace>,
    instr: Instr,
    gc_state: &mut GcState,
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
            policy,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Infer,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let model = string_expr(&machine.env, &model, "Infer.model")?;
            let prompt = resolve_prompt(config, &machine.env, prompt)?;
            let prompt = hydrate_infer_prompt(config, &Value::Null, prompt).await?;
            let mut prompt = maybe_collect_prompt(config, prompt, gc_state).await?;
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::InferCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    model: model.clone(),
                    prompt: config.trace_full_payloads.then(|| prompt.clone()),
                    prompt_preview: prompt_preview(&prompt),
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            // Catch-overflow retries stay inside this one Infer instruction:
            // failed attempts surface as gc_collect{trigger:context_overflow}
            // events and the single InferResult/InferError pair reports the
            // outcome, so effect-id replay keeps its one-call-one-result
            // contract (replay branches never engage the retry loop).
            let live = ir_replay.is_none() && config.replay.is_none();
            let mut overflow_cycles = 0usize;
            let result = loop {
                let attempt = match ir_replay {
                    Some(replay) => replay.infer_result(&location, &model),
                    None => match &config.replay {
                        Some(replay) => replay.infer_result(op_id, &model),
                        None => {
                            config
                                .provider
                                .chat(&Model(model.clone()), &ir_tool_specs(config), &prompt)
                                .await
                        }
                    },
                };
                match attempt {
                    Err(err)
                        if live
                            && catch_overflow_active(config)
                            && crate::provider::is_context_overflow_anyhow(&err)
                            && overflow_cycles < CATCH_OVERFLOW_MAX_CYCLES =>
                    {
                        overflow_cycles += 1;
                        let (collected, target) =
                            collect_for_overflow(config, prompt, gc_state, overflow_cycles).await?;
                        prompt = collected;
                        gc_state.discovered_budget = Some(target);
                    }
                    other => break other,
                }
            };
            let response = match result {
                Ok(response) => response,
                Err(err) => {
                    let err = annotate_overflow_failure(err, overflow_cycles);
                    config
                        .trace
                        .emit(&Event::InferError {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id: None,
                            error: format!("{err:#}"),
                            duration_ms: millis_u64(started.elapsed()),
                            timestamp: Utc::now(),
                        })
                        .await?;
                    // Errors-as-values (t-1222): a Bind site converts the
                    // failure into a tool-visible value instead of unwinding
                    // the turn. Abort (default) propagates.
                    if policy.on_error == EffectErrorMode::Bind {
                        machine.env.insert(out, effect_error_value(&err));
                        return Ok(());
                    }
                    return Err(err);
                }
            };
            config
                .trace
                .emit(&Event::InferResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    response: Some(response.clone()),
                    response_preview: response_preview(&response),
                    input_tokens: response.input_tokens,
                    output_tokens: response.output_tokens,
                    total_tokens: response.total_tokens,
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
                    parent_op_id: None,
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
            let started = Instant::now();
            let result = match ir_replay {
                Some(replay) => replay.eval_result(&location, &command),
                None => match &config.replay {
                    Some(replay) => replay.eval_result(op_id, &command),
                    None => {
                        run_eval_with_env(&config.eval, &command, config.trace.trace_context_env())
                            .await
                    }
                },
            };
            let result = match result {
                Ok(result) => result,
                Err(err) => {
                    config
                        .trace
                        .emit(&Event::EvalError {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id: None,
                            command,
                            error: format!("{err:#}"),
                            duration_ms: millis_u64(started.elapsed()),
                            timestamp: Utc::now(),
                        })
                        .await?;
                    return Err(err);
                }
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
                    parent_op_id: None,
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
        Instr::Emit { event } => {
            let location =
                effect_location(program_hash.clone(), EffectKind::Emit, site, dynamic_path)?;
            emit_ir_effect(config, &location).await?;
            let value = eval_expr(&machine.env, &event)?;
            let event: Event =
                serde_json::from_value(value).context("decoding AgentIR Emit event")?;
            config.trace.emit(&event).await?;
        }
        Instr::Retrieve {
            out,
            query,
            kind,
            max_bytes,
            policy,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Retrieve,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let query = string_expr(&machine.env, &query, "Retrieve.query")?;
            let retrieve_key = IrRetrieveCall {
                query: query.clone(),
                kind: kind.map(|kind| format!("{kind:?}")),
                max_bytes,
            };
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::RetrieveCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    query: query.clone(),
                    kind: retrieve_key.kind.clone(),
                    max_bytes,
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            let result = match ir_replay {
                Some(replay) => replay.retrieve_result(&location, &retrieve_key),
                None => {
                    let params = crate::hydration::SourceParams {
                        query: Some(query.clone()),
                        max_bytes,
                    };
                    config
                        .hydration
                        .retrieve_query_of_kind(params, kind)
                        .await
                        .and_then(|results| serde_json::to_value(results).map_err(Into::into))
                }
            };
            let results = match result {
                Ok(results) => results,
                Err(err) => {
                    config
                        .trace
                        .emit(&Event::RetrieveError {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id: None,
                            error: format!("{err:#}"),
                            duration_ms: millis_u64(started.elapsed()),
                            timestamp: Utc::now(),
                        })
                        .await?;
                    if policy.on_error == EffectErrorMode::Bind {
                        machine.env.insert(out, effect_error_value(&err));
                        return Ok(());
                    }
                    return Err(err);
                }
            };
            let rendered = results.to_string();
            config
                .trace
                .emit(&Event::RetrieveResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    results: results.clone(),
                    result_preview: crate::trace::preview(&rendered, 512),
                    source_count: results.as_array().map_or(0, Vec::len),
                    bytes: rendered.len(),
                    duration_ms: millis_u64(started.elapsed()),
                    timestamp: Utc::now(),
                })
                .await?;
            machine.env.insert(out, results);
        }
        Instr::Store {
            out,
            sink,
            op: store_op,
            id,
            item,
            policy,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Store,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let sink_name = string_expr(&machine.env, &sink, "Store.sink")?;
            let item_value = eval_expr(&machine.env, &item)?;
            let id_value = id
                .map(|id| string_expr(&machine.env, &id, "Store.id"))
                .transpose()?;
            // Hash over the payload only (provenance is runtime-attached and
            // legitimately differs between record and replay).
            let content_hash = format!("{:x}", Sha256::digest(item_value.to_string().as_bytes()));
            // The full Store identity: sink and id are dynamic (Expr), so
            // replay must check them, not just the payload hash (t-1182 review).
            let store_key = IrStoreCall {
                sink: sink_name.clone(),
                op: store_op.name().into(),
                id: id_value.clone(),
                content_hash: content_hash.clone(),
            };
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::StoreCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    sink: sink_name.clone(),
                    store_op: store_op.name().into(),
                    store_id: id_value.clone(),
                    item_preview: crate::trace::preview(&item_value.to_string(), 256),
                    content_hash: content_hash.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            let result = match ir_replay {
                // Replay never mutates: the recorded id is the result.
                Some(replay) => replay.store_result(&location, &store_key),
                None => {
                    execute_store_live(
                        config,
                        &location,
                        &sink_name,
                        store_op,
                        id_value.as_deref(),
                        item_value,
                    )
                    .await
                }
            };
            let sink_id = match result {
                Ok(sink_id) => sink_id,
                Err(err) => {
                    config
                        .trace
                        .emit(&Event::StoreError {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id: None,
                            sink: sink_name.clone(),
                            error: format!("{err:#}"),
                            duration_ms: millis_u64(started.elapsed()),
                            timestamp: Utc::now(),
                        })
                        .await?;
                    if policy.on_error == EffectErrorMode::Bind {
                        machine.env.insert(out, effect_error_value(&err));
                        return Ok(());
                    }
                    return Err(err);
                }
            };
            config
                .trace
                .emit(&Event::StoreResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    sink: sink_name,
                    sink_id: sink_id.clone(),
                    duration_ms: millis_u64(started.elapsed()),
                    timestamp: Utc::now(),
                })
                .await?;
            machine.env.insert(out, Value::String(sink_id));
        }
    }
    Ok(())
}

/// The value an effect binds to its `out` when `on_error: Bind` and the
/// effect failed (t-1222). A small, model-legible envelope mirroring the
/// shell tool's failure shape so the agent loop surfaces it as a tool
/// result the model can read and recover from.
fn effect_error_value(err: &anyhow::Error) -> Value {
    serde_json::json!({ "ok": false, "error": format!("{err:#}") })
}

/// Execute a live Store against the registry: resolve the sink, enforce its
/// write policy, attach provenance, run the op. Returns the sink id (for
/// Update/Delete, the caller-supplied id echoed back).
async fn execute_store_live(
    config: &SeqConfig,
    location: &EffectLocation,
    sink_name: &str,
    store_op: crate::ir::StoreOp,
    id: Option<&str>,
    payload: Value,
) -> Result<String> {
    use crate::hydration::{Provenance, SinkId, SinkItem, SinkWritePolicy};

    let sink = config
        .hydration
        .sink(sink_name)
        .ok_or_else(|| anyhow!("no sink {sink_name:?} registered"))?;
    if sink.write_policy() == SinkWritePolicy::RequireApproval {
        return Err(anyhow!(
            "sink {sink_name:?} requires approval; the approval flow is not implemented yet (docs/MEMORY.md)"
        ));
    }
    let item = SinkItem {
        payload,
        provenance: Provenance {
            run_id: config.trace.run_id().into(),
            effect_id: Some(location.effect_id.0.clone()),
            timestamp: Some(Utc::now()),
        },
    };
    match store_op {
        crate::ir::StoreOp::Create => Ok(sink.store(item).await?.0),
        crate::ir::StoreOp::Update => {
            let id = id.ok_or_else(|| anyhow!("Store update requires an id"))?;
            sink.update(&SinkId(id.into()), item).await?;
            Ok(id.into())
        }
        crate::ir::StoreOp::Delete => {
            let id = id.ok_or_else(|| anyhow!("Store delete requires an id"))?;
            sink.delete(&SinkId(id.into())).await?;
            Ok(id.into())
        }
    }
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

/// Tool list shown to the provider: shell + infer always; remember/recall
/// appear automatically whenever a memory backend is registered (settled
/// question 6 — an unreachable sink is a trap, so registration IS the
/// exposure switch).
fn ir_tool_specs(config: &SeqConfig) -> Vec<crate::provider::ToolSpec> {
    let mut specs = base_ir_tool_specs();
    if !config
        .hydration
        .sinks_of_kind(crate::hydration::SourceKind::Semantic)
        .is_empty()
    {
        specs.push(crate::provider::ToolSpec {
            kind: "function".into(),
            function: crate::provider::ToolFunctionSpec {
                name: "remember".into(),
                description: "Save a fact to persistent memory for future sessions. \
                              Use when something is worth keeping beyond this conversation."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "content": { "type": "string", "description": "the fact to keep" },
                        "name": {
                            "type": "string",
                            "description": "optional kebab-case slug; derived from the description or content when omitted"
                        },
                        "description": {
                            "type": "string",
                            "description": "one-line summary used for recall relevance"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["user", "feedback", "project", "reference"]
                        }
                    },
                    "required": ["content"]
                }),
            },
        });
        specs.push(crate::provider::ToolSpec {
            kind: "function".into(),
            function: crate::provider::ToolFunctionSpec {
                name: "recall".into(),
                description: "Search persistent memory by keywords and return matching notes."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }),
            },
        });
    }
    specs
}

fn base_ir_tool_specs() -> Vec<crate::provider::ToolSpec> {
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

fn next_visit(machine: &mut Machine, site: EffectSite) -> u64 {
    let key = format!("{}:{}", site.block.0, site.instruction_index);
    let visit = machine.effect_visits.entry(key).or_insert(0);
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
        Expr::FieldOr {
            base,
            field,
            default,
        } => {
            let value = env
                .get(base)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", base))?;
            match value.get(field).cloned() {
                Some(value) => Ok(value),
                None => eval_expr(env, default),
            }
        }
        Expr::StringOr { value, default } => match eval_expr(env, value)? {
            Value::String(value) => Ok(Value::String(value)),
            _ => eval_expr(env, default),
        },
        Expr::If {
            cond,
            then_value,
            else_value,
        } => {
            if bool_expr(env, cond, "If.cond")? {
                eval_expr(env, then_value)
            } else {
                eval_expr(env, else_value)
            }
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
        Expr::And { left, right } => Ok(Value::Bool(
            bool_expr(env, left, "And.left")? && bool_expr(env, right, "And.right")?,
        )),
        Expr::HasPendingToolCalls { base } => {
            let value = env
                .get(base)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", base))?;
            Ok(Value::Bool(has_pending_tool_calls(value)?))
        }
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
        Expr::JsonParseOr { value, default } => {
            let text = string_expr(env, value, "JsonParseOr.value")?;
            match serde_json::from_str(&text) {
                Ok(value) => Ok(value),
                Err(_) => eval_expr(env, default),
            }
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

fn has_pending_tool_calls(value: &Value) -> Result<bool> {
    let messages = value
        .as_array()
        .ok_or_else(|| anyhow!("AgentIR HasPendingToolCalls expected array, got {value}"))?;
    let mut pending = std::collections::BTreeSet::new();
    for message in messages {
        if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
            pending.extend(tool_calls.iter().filter_map(|call| {
                call.get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            }));
        }
        if let Some(tool_call_id) = message.get("tool_call_id").and_then(Value::as_str) {
            pending.remove(tool_call_id);
        }
    }
    Ok(!pending.is_empty())
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

fn resolve_prompt(
    config: &SeqConfig,
    env: &BTreeMap<Var, Value>,
    prompt: PromptRef,
) -> Result<Prompt> {
    match prompt {
        PromptRef::Inline(prompt) => Ok(prompt),
        PromptRef::Var(var) => {
            let value = env
                .get(&var)
                .cloned()
                .ok_or_else(|| anyhow!("unknown AgentIR prompt var {:?}", var))?;
            serde_json::from_value::<Vec<ChatMessage>>(value).context("decoding AgentIR prompt")
        }
        PromptRef::PromptIr(mut prompt_ir) => {
            if config.gc.is_mark_sweep() {
                collect_prompt_ir_sections(&mut prompt_ir, config.context_budget);
            }
            Ok(compile_prompt_ir(&prompt_ir))
        }
        PromptRef::PromptIrVar(var) => {
            let value = env
                .get(&var)
                .cloned()
                .ok_or_else(|| anyhow!("unknown AgentIR PromptIR var {:?}", var))?;
            let mut prompt_ir =
                serde_json::from_value::<PromptIR>(value).context("decoding AgentIR PromptIR")?;
            if config.gc.is_mark_sweep() {
                collect_prompt_ir_sections(&mut prompt_ir, config.context_budget);
            }
            Ok(compile_prompt_ir(&prompt_ir))
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
    use crate::gc::GcMode;
    use crate::gc::GcTiming;
    use crate::hydration::{PassiveHydrationConfig, SourceRegistry};
    use crate::interpreter::{EvalConfig, SeqConfig};
    use crate::op::{Response, ToolCall};
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
            tool_calls: Vec::<ToolCall>::new(),
            finish_reason: Some(crate::op::FinishReason::Stop),
            input_tokens: 0,
            output_tokens: 1,
            total_tokens: 1,
            metadata: Default::default(),
        }
    }

    fn test_trace() -> TraceLogger {
        let path = std::env::temp_dir().join(format!("agent-ir-test-{}.jsonl", Uuid::new_v4()));
        TraceLogger::new(Uuid::new_v4().to_string(), path)
    }

    fn config(provider: Arc<dyn ChatProvider>) -> SeqConfig {
        config_with_trace(provider, test_trace())
    }

    fn config_with_trace(provider: Arc<dyn ChatProvider>, trace: TraceLogger) -> SeqConfig {
        SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 200_000,
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
            effect_visits: BTreeMap::new(),
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

    /// Fails the first `failures` chat calls with a raw codex-style overflow
    /// message, then serves queued responses (mirrors the smith t-1145 shape).
    struct OverflowProvider {
        failures: Mutex<usize>,
        responses: Mutex<Vec<Response>>,
        prompts: Mutex<Vec<Prompt>>,
    }

    impl OverflowProvider {
        fn new(failures: usize, mut responses: Vec<Response>) -> Self {
            responses.reverse();
            Self {
                failures: Mutex::new(failures),
                responses: Mutex::new(responses),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn prompts(&self) -> Vec<Prompt> {
            self.prompts.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ChatProvider for OverflowProvider {
        async fn chat(
            &self,
            _model: &Model,
            _tools: &[ToolSpec],
            messages: &[ChatMessage],
        ) -> Result<Response> {
            self.prompts.lock().unwrap().push(messages.to_vec());
            let mut failures = self.failures.lock().unwrap();
            if *failures > 0 {
                *failures -= 1;
                return Err(anyhow!(
                    "Codex OAuth provider returned 400 Bad Request: \
                     Your input exceeds the context window of this model."
                ));
            }
            self.responses
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow!("mock provider exhausted"))
        }
    }

    #[tokio::test]
    async fn ir_infer_catch_overflow_collects_and_retries() -> Result<()> {
        let provider = Arc::new(OverflowProvider::new(1, vec![response("recovered")]));
        let mut config = config(provider.clone());
        config.gc = GcMode::Ring(crate::gc::RingGc::default());
        config.gc_timing = crate::gc::GcTiming::CatchOverflow;
        let mut prompt = vec![ChatMessage::system("system")];
        prompt.extend((0..6).map(|i| ChatMessage::user(format!("{i}-{}", "x".repeat(200)))));
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Infer {
                    out: Var("a".into()),
                    model: Expr::Value(Value::String("mock".into())),
                    prompt: PromptRef::Inline(prompt),
                    policy: Default::default(),
                }],
                terminator: Terminator::Return {
                    value: Expr::Field {
                        base: Var("a".into()),
                        field: "content".into(),
                    },
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("overflow-retry".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            effect_visits: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let (value, _machine) = run_ir_sequential(&config, machine).await?;

        assert_eq!(value, Value::String("recovered".into()));
        let prompts = provider.prompts();
        assert_eq!(prompts.len(), 2, "one overflow, one retry");
        assert!(
            prompts[1].len() < prompts[0].len(),
            "the retry prompt must have been collected"
        );
        Ok(())
    }

    fn overflow_test_machine(name: &str, prompt: Vec<ChatMessage>) -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Infer {
                    out: Var("a".into()),
                    model: Expr::Value(Value::String("mock".into())),
                    prompt: PromptRef::Inline(prompt),
                    policy: Default::default(),
                }],
                terminator: Terminator::Return {
                    value: Expr::Field {
                        base: Var("a".into()),
                        field: "content".into(),
                    },
                },
            },
        );
        Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId(name.into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            effect_visits: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn discovered_budget_survives_across_loop_runs() -> Result<()> {
        // Turn 1 overflows once and learns the real ceiling; turn 2 (same
        // caller-owned GcState) must collect proactively and never see a
        // failed provider call (t-1162).
        let provider = Arc::new(OverflowProvider::new(
            1,
            vec![response("turn-one"), response("turn-two")],
        ));
        let mut config = config(provider.clone());
        config.gc = GcMode::Ring(crate::gc::RingGc::default());
        config.gc_timing = crate::gc::GcTiming::CatchOverflow;
        let mut prompt = vec![ChatMessage::system("system")];
        prompt.extend((0..6).map(|i| ChatMessage::user(format!("{i}-{}", "x".repeat(200)))));

        let mut gc_state = crate::gc::GcState::default();
        let mut store = InMemoryStore::new();
        let (value, _) = run_ir_sequential_with_gc(
            &config,
            overflow_test_machine("turn-1", prompt.clone()),
            &mut store,
            None,
            &mut gc_state,
        )
        .await?;
        assert_eq!(value, Value::String("turn-one".into()));
        let ceiling = gc_state
            .discovered_budget
            .expect("turn 1 learned the ceiling");

        let mut store = InMemoryStore::new();
        let (value, _) = run_ir_sequential_with_gc(
            &config,
            overflow_test_machine("turn-2", prompt.clone()),
            &mut store,
            None,
            &mut gc_state,
        )
        .await?;
        assert_eq!(value, Value::String("turn-two".into()));
        assert_eq!(gc_state.discovered_budget, Some(ceiling));

        let prompts = provider.prompts();
        assert_eq!(
            prompts.len(),
            3,
            "turn 1: overflow + retry; turn 2: exactly one call, no relearning"
        );
        assert!(
            prompts[2].len() < prompt.len(),
            "turn 2's prompt was proactively collected to the learned ceiling"
        );
        Ok(())
    }

    fn memory_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("agent-ir-memory-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Store a note into the memory sink, then Retrieve it back, in one
    /// program: the full active write/read round trip of docs/MEMORY.md.
    fn store_then_retrieve_machine() -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![
                    Instr::Store {
                        out: Var("stored".into()),
                        sink: Expr::Value(Value::String("memory".into())),
                        op: crate::ir::StoreOp::Create,
                        id: None,
                        item: Expr::Value(serde_json::json!({
                            "name": "round-trip",
                            "description": "stored by the Store effect",
                            "body": "the remembered fact",
                        })),
                        policy: Default::default(),
                    },
                    Instr::Retrieve {
                        out: Var("hits".into()),
                        query: Expr::Value(Value::String("remembered fact".into())),
                        kind: Some(crate::hydration::SourceKind::Semantic),
                        max_bytes: None,
                        policy: Default::default(),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("hits".into())),
                },
            },
        );
        Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("store-retrieve".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            effect_visits: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn store_and_retrieve_effects_round_trip_against_the_memory_backend() -> Result<()> {
        let dir = memory_dir();
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = config_with_trace(Arc::new(MockProvider::new(vec![])), trace);
        config.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(dir.clone()));

        let (value, _machine) = run_ir_sequential(&config, store_then_retrieve_machine()).await?;

        let written = std::fs::read_to_string(dir.join("round-trip.md"))?;
        assert!(
            written.contains("provenance"),
            "the runtime attaches provenance: {written}"
        );
        let hits = value.as_array().expect("retrieve returns a result list");
        assert_eq!(hits.len(), 1);
        assert!(hits[0]["content"]
            .as_str()
            .unwrap()
            .contains("the remembered fact"));

        let events = crate::trace::TraceLogger::read_events(trace_path).await?;
        let names: Vec<&str> = events.iter().map(|event| event.name()).collect();
        assert!(names.contains(&"Store"), "{names:?}");
        assert!(names.contains(&"Retrieve"), "{names:?}");
        Ok(())
    }

    #[tokio::test]
    async fn replayed_store_and_retrieve_never_touch_the_backend() -> Result<()> {
        let dir = memory_dir();
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = config_with_trace(Arc::new(MockProvider::new(vec![])), trace);
        config.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(dir.clone()));
        let (live_value, _) = run_ir_sequential(&config, store_then_retrieve_machine()).await?;

        // Remove the backend entirely: a replay that touched it would
        // re-create the directory — assert the same result AND an untouched
        // filesystem.
        std::fs::remove_dir_all(&dir)?;
        let events = crate::trace::TraceLogger::read_events(&trace_path).await?;
        let replay = IrReplayTrace::from_events(&events)?;
        let replay_trace = test_trace();
        let mut replay_config =
            config_with_trace(Arc::new(MockProvider::new(vec![])), replay_trace);
        replay_config.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(dir.clone()));

        let mut store = InMemoryStore::new();
        let (replayed_value, _) = run_ir_sequential_with_store_and_replay(
            &replay_config,
            store_then_retrieve_machine(),
            &mut store,
            Some(&replay),
        )
        .await?;

        assert_eq!(replayed_value, live_value);
        assert!(!dir.exists(), "replay must never write to the sink");
        Ok(())
    }

    #[tokio::test]
    async fn replayed_store_detects_payload_divergence() -> Result<()> {
        let dir = memory_dir();
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = config_with_trace(Arc::new(MockProvider::new(vec![])), trace);
        config.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(dir.clone()));
        run_ir_sequential(&config, store_then_retrieve_machine()).await?;

        let _ = trace_path;
        // Same program (same effect ids), different RUNTIME payload: the
        // recorded content hash must catch it. An edited program would
        // change the effect id and miss earlier — dynamic divergence is
        // exactly what the hash exists for.
        let mut diverged = store_then_retrieve_machine();
        if let Some(block) = diverged.program.blocks.get_mut(&BlockId(0)) {
            // Env-seeded vars are declared as entry params.
            block.params = vec![Var("note".into())];
            if let Instr::Store { item, .. } = &mut block.instructions[0] {
                *item = Expr::Var(Var("note".into()));
            }
        }
        let mut recorded = diverged.clone();
        recorded.env.insert(
            Var("note".into()),
            serde_json::json!({
                "name": "dynamic",
                "description": "from env",
                "body": "recorded payload",
            }),
        );
        let dir2 = memory_dir();
        let trace2 = test_trace();
        let trace2_path = trace2.path().clone();
        let mut config2 = config_with_trace(Arc::new(MockProvider::new(vec![])), trace2);
        config2.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(dir2));
        run_ir_sequential(&config2, recorded).await?;
        let replay = IrReplayTrace::from_events(
            &crate::trace::TraceLogger::read_events(&trace2_path).await?,
        )?;

        diverged.env.insert(
            Var("note".into()),
            serde_json::json!({
                "name": "dynamic",
                "description": "from env",
                "body": "a DIFFERENT payload",
            }),
        );
        let mut store = InMemoryStore::new();
        let err =
            run_ir_sequential_with_store_and_replay(&config, diverged, &mut store, Some(&replay))
                .await
                .expect_err("payload divergence must fail replay");
        assert!(err.to_string().contains("hash"), "{err}");
        Ok(())
    }

    /// t-1182 review: the Store sink is dynamic (an Expr), so a same-site
    /// call computing a different sink must diverge on replay even when the
    /// payload (and thus content_hash) is byte-identical — the old hash-only
    /// check would have replayed a stale result against the wrong sink.
    #[tokio::test]
    async fn replayed_store_detects_sink_divergence_with_identical_payload() -> Result<()> {
        fn dynamic_sink_machine() -> Machine {
            let mut blocks = BTreeMap::new();
            blocks.insert(
                BlockId(0),
                crate::ir::Block {
                    params: vec![Var("sink".into())],
                    instructions: vec![Instr::Store {
                        out: Var("id".into()),
                        sink: Expr::Var(Var("sink".into())),
                        op: crate::ir::StoreOp::Create,
                        id: None,
                        item: Expr::Value(serde_json::json!({
                            "name": "fixed",
                            "description": "identical payload",
                            "body": "same bytes either run",
                        })),
                        policy: Default::default(),
                    }],
                    terminator: Terminator::Return {
                        value: Expr::Var(Var("id".into())),
                    },
                },
            );
            Machine {
                program: crate::ir::Program {
                    id: crate::ir::ProgramId("dynamic-sink".into()),
                    entry: BlockId(0),
                    blocks,
                },
                block: BlockId(0),
                pc: 0,
                env: BTreeMap::new(),
                effect_visits: BTreeMap::new(),
                continuation_stack: vec![],
                budgets: Default::default(),
            }
        }

        // Record a run that stores to the "memory" sink.
        let dir = memory_dir();
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = config_with_trace(Arc::new(MockProvider::new(vec![])), trace);
        config.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(dir));
        let mut recorded = dynamic_sink_machine();
        recorded
            .env
            .insert(Var("sink".into()), Value::String("memory".into()));
        run_ir_sequential(&config, recorded).await?;
        let replay = IrReplayTrace::from_events(
            &crate::trace::TraceLogger::read_events(&trace_path).await?,
        )?;

        // Replay the same program/site with the sink computed differently.
        let mut diverged = dynamic_sink_machine();
        diverged
            .env
            .insert(Var("sink".into()), Value::String("elsewhere".into()));
        let mut store = InMemoryStore::new();
        let err =
            run_ir_sequential_with_store_and_replay(&config, diverged, &mut store, Some(&replay))
                .await
                .expect_err("sink divergence must fail replay even with identical payload");
        let message = err.to_string();
        assert!(
            message.contains("memory") && message.contains("elsewhere"),
            "{message}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn store_against_unregistered_sink_fails_with_a_clear_error() -> Result<()> {
        let config = config(Arc::new(MockProvider::new(vec![])));
        let err = run_ir_sequential(&config, store_then_retrieve_machine())
            .await
            .expect_err("no sink registered");
        assert!(
            format!("{err:#}").contains("no sink \"memory\" registered"),
            "{err:#}"
        );
        Ok(())
    }

    #[test]
    fn memory_tools_ride_with_backend_registration() {
        let bare = config(Arc::new(MockProvider::new(vec![])));
        let names: Vec<String> = ir_tool_specs(&bare)
            .iter()
            .map(|spec| spec.function.name.clone())
            .collect();
        assert_eq!(names, vec!["shell", "infer"]);

        let mut with_memory = config(Arc::new(MockProvider::new(vec![])));
        with_memory.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(memory_dir()));
        let names: Vec<String> = ir_tool_specs(&with_memory)
            .iter()
            .map(|spec| spec.function.name.clone())
            .collect();
        assert_eq!(names, vec!["shell", "infer", "remember", "recall"]);
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
                instructions: vec![Instr::Retrieve {
                    out: Var("a".into()),
                    query: Expr::Value(Value::String("missing".into())),
                    kind: None,
                    max_bytes: None,
                    policy: Default::default(),
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
            effect_visits: BTreeMap::new(),
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
        assert_eq!(locations[0].kind, EffectKind::Retrieve);
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
            effect_visits: BTreeMap::new(),
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
            effect_visits: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn ir_step_limit_suspends_goto_only_loops() -> Result<()> {
        // A cycle of blocks with no instructions must still hit the step
        // limit: block transitions count as steps.
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![],
                terminator: Terminator::Goto {
                    block: BlockId(1),
                    args: vec![],
                },
            },
        );
        blocks.insert(
            BlockId(1),
            crate::ir::Block {
                params: vec![],
                instructions: vec![],
                terminator: Terminator::Goto {
                    block: BlockId(0),
                    args: vec![],
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("goto-loop".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            effect_visits: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let outcome =
            run_ir_steps(&config(Arc::new(MockProvider::new(vec![]))), machine, 10).await?;

        assert!(
            matches!(outcome, IrStepOutcome::Suspended { .. }),
            "goto-only loop must suspend at the step limit, got {outcome:?}"
        );
        Ok(())
    }

    #[test]
    fn if_expr_selects_branch_lazily_and_rejects_non_bool_cond() {
        let env = BTreeMap::from([(Var("flag".into()), Value::Bool(true))]);
        let expr = Expr::If {
            cond: Box::new(Expr::Var(Var("flag".into()))),
            then_value: Box::new(Expr::Value(Value::String("yes".into()))),
            // The untaken branch references an unknown var; lazy evaluation
            // must not touch it.
            else_value: Box::new(Expr::Var(Var("missing".into()))),
        };
        assert_eq!(eval_expr(&env, &expr).unwrap(), Value::String("yes".into()));

        let non_bool = Expr::If {
            cond: Box::new(Expr::Value(Value::String("nope".into()))),
            then_value: Box::new(Expr::Value(Value::Null)),
            else_value: Box::new(Expr::Value(Value::Null)),
        };
        let err = eval_expr(&env, &non_bool).unwrap_err().to_string();
        assert!(err.contains("If.cond"), "got: {err}");
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
            effect_visits: BTreeMap::new(),
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
