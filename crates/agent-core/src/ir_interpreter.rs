use crate::gc::GcState;
use crate::interpreter::{
    annotate_overflow_failure, catch_overflow_active, collect_for_overflow, hydrate_infer_prompt,
    maybe_collect_prompt, millis_u64, prompt_preview, response_preview, run_eval_argv_with_env,
    run_eval_with_env, SeqConfig, CATCH_OVERFLOW_MAX_CYCLES,
};
use crate::ir::{
    effect_location, program_hash, validate_program, BlockId, DynamicPath, EffectErrorMode,
    EffectKind, EffectLocation, EffectSite, EvalRequest, Expr, Instr, Machine, Pattern,
    ProgramHash, PromptRef, Terminator, Var,
};
use crate::op::{ChatMessage, Model, Prompt};
use crate::prompt_ir::{collect_prompt_ir_sections, compile_prompt_ir, PromptIR};
use crate::trace::Event;
use anyhow::{anyhow, Context, Result};
use async_recursion::async_recursion;
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
    tool_calls: BTreeMap<String, IrToolCall>,
    tool_results: BTreeMap<String, Value>,
    tool_errors: BTreeMap<String, String>,
    /// Output-contract schema hash recorded by the run (the Custom
    /// `output_contract` event, t-1308.4). Run-identity metadata: replaying
    /// with a different contract must diverge, so the agent-loop driver
    /// compares this against the current contract before executing.
    output_schema_hash: Option<String>,
    /// Approval-gate history (t-1308.10), keyed by effect id: recorded
    /// `ApprovalRequested` payloads and `ApprovalResolved` decisions.
    /// Replay reproduces the pause/decision as data — never pausing a
    /// resolved recording and never prompting.
    approval_requests: BTreeMap<String, IrApprovalRequested>,
    approval_resolutions: BTreeMap<String, IrApprovalResolved>,
}

/// A recorded `ApprovalRequested`: the request payload doubles as the
/// dynamic identity check for effects that never emitted a `*Call` (denied
/// or still-pending), where the usual call-identity comparison cannot run.
#[derive(Debug, Clone, PartialEq)]
struct IrApprovalRequested {
    pending_id: String,
    kind: String,
    request: Value,
}

#[derive(Debug, Clone, PartialEq)]
struct IrApprovalResolved {
    pending_id: String,
    decision: String,
    resolved_by: Option<String>,
    reason: Option<String>,
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

/// The native-tool identity recorded for replay-divergence detection: the
/// registered name plus the full model-supplied arguments (both dynamic at
/// the call site — a same-site call with different arguments must diverge
/// loudly instead of replaying a stale result).
#[derive(Debug, Clone, PartialEq)]
struct IrToolCall {
    location: EffectLocation,
    name: String,
    arguments: Value,
}

/// The Eval identity recorded for replay-divergence detection: the display
/// command plus, for direct-exec Evals, the exact argv. Both are compared —
/// a shell recording never satisfies an argv replay even if the rendered
/// command strings coincide, and a dynamically-computed argv element that
/// changed between record and replay diverges loudly.
#[derive(Debug, Clone, PartialEq)]
struct IrEvalCall {
    location: EffectLocation,
    command: String,
    argv: Option<Vec<String>>,
}

impl IrReplayTrace {
    pub async fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let events = crate::trace::TraceLogger::read_events(path).await?;
        Self::from_events(&events)
    }

    /// Build a replay index from trace events. Effect identity is read from
    /// the `effect` field carried directly on each `*Call` event; results and
    /// errors are paired to their call via `op_id`. No event adjacency is
    /// assumed, so a reordered, filtered, or interleaved trace still indexes
    /// correctly as long as call/result pairs share their op ids.
    pub fn from_events(events: &[Event]) -> Result<Self> {
        let mut replay = Self::default();
        // op_id -> effect_id for every effect call seen so far. Op ids are
        // unique within a run, so one map covers all four effect kinds.
        let mut effect_by_op: BTreeMap<u64, String> = BTreeMap::new();

        for event in events {
            match event {
                Event::InferCall {
                    op_id,
                    model,
                    effect,
                    ..
                } => {
                    let location = require_effect(effect.as_deref(), EffectKind::Infer, *op_id)?;
                    let effect_id = location.effect_id.0.clone();
                    replay.infer_calls.insert(
                        effect_id.clone(),
                        IrInferCall {
                            location: location.clone(),
                            model: model.clone(),
                        },
                    );
                    effect_by_op.insert(*op_id, effect_id);
                }
                Event::InferResult {
                    op_id,
                    response: Some(response),
                    ..
                } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay
                            .infer_results
                            .insert(effect_id.clone(), response.clone());
                    }
                }
                Event::InferError { op_id, error, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay.infer_errors.insert(effect_id.clone(), error.clone());
                    }
                }
                Event::EvalCall {
                    op_id,
                    command,
                    argv,
                    effect,
                    ..
                } => {
                    let location = require_effect(effect.as_deref(), EffectKind::Eval, *op_id)?;
                    let effect_id = location.effect_id.0.clone();
                    replay.eval_calls.insert(
                        effect_id.clone(),
                        IrEvalCall {
                            location: location.clone(),
                            command: command.clone(),
                            argv: argv.clone(),
                        },
                    );
                    effect_by_op.insert(*op_id, effect_id);
                }
                Event::EvalResult { op_id, result, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay
                            .eval_results
                            .insert(effect_id.clone(), result.clone());
                    }
                }
                Event::EvalError { op_id, error, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay.eval_errors.insert(effect_id.clone(), error.clone());
                    }
                }
                Event::RetrieveCall {
                    op_id,
                    query,
                    kind,
                    max_bytes,
                    effect,
                    ..
                } => {
                    let location = require_effect(effect.as_deref(), EffectKind::Retrieve, *op_id)?;
                    let effect_id = location.effect_id.0.clone();
                    replay.retrieve_calls.insert(
                        effect_id.clone(),
                        IrRetrieveCall {
                            query: query.clone(),
                            kind: kind.clone(),
                            max_bytes: *max_bytes,
                        },
                    );
                    effect_by_op.insert(*op_id, effect_id);
                }
                Event::RetrieveResult { op_id, results, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay
                            .retrieve_results
                            .insert(effect_id.clone(), results.clone());
                    }
                }
                Event::RetrieveError { op_id, error, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay
                            .retrieve_errors
                            .insert(effect_id.clone(), error.clone());
                    }
                }
                Event::StoreCall {
                    op_id,
                    sink,
                    store_op,
                    store_id,
                    content_hash,
                    effect,
                    ..
                } => {
                    let location = require_effect(effect.as_deref(), EffectKind::Store, *op_id)?;
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
                    effect_by_op.insert(*op_id, effect_id);
                }
                Event::StoreResult { op_id, sink_id, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay
                            .store_results
                            .insert(effect_id.clone(), sink_id.clone());
                    }
                }
                Event::StoreError { op_id, error, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay.store_errors.insert(effect_id.clone(), error.clone());
                    }
                }
                Event::ToolCall {
                    op_id,
                    name,
                    arguments,
                    effect,
                    ..
                } => {
                    let location = require_effect(effect.as_deref(), EffectKind::Tool, *op_id)?;
                    let effect_id = location.effect_id.0.clone();
                    replay.tool_calls.insert(
                        effect_id.clone(),
                        IrToolCall {
                            location: location.clone(),
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    );
                    effect_by_op.insert(*op_id, effect_id);
                }
                Event::ToolResult { op_id, result, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay
                            .tool_results
                            .insert(effect_id.clone(), result.clone());
                    }
                }
                Event::ToolError { op_id, error, .. } => {
                    if let Some(effect_id) = effect_by_op.get(op_id) {
                        replay.tool_errors.insert(effect_id.clone(), error.clone());
                    }
                }
                Event::Custom { name, data, .. }
                    if name == crate::output_contract::OUTPUT_CONTRACT_EVENT =>
                {
                    if let Some(hash) = data.get("schema_hash").and_then(Value::as_str) {
                        replay.output_schema_hash = Some(hash.to_owned());
                    }
                }
                Event::ApprovalRequested {
                    pending_id,
                    kind,
                    request,
                    effect,
                    ..
                } => {
                    replay.approval_requests.insert(
                        effect.effect_id.0.clone(),
                        IrApprovalRequested {
                            pending_id: pending_id.clone(),
                            kind: kind.clone(),
                            request: request.clone(),
                        },
                    );
                }
                Event::ApprovalResolved {
                    pending_id,
                    effect_id,
                    decision,
                    resolved_by,
                    reason,
                    ..
                } => {
                    replay.approval_resolutions.insert(
                        effect_id.clone(),
                        IrApprovalResolved {
                            pending_id: pending_id.clone(),
                            decision: decision.clone(),
                            resolved_by: resolved_by.clone(),
                            reason: reason.clone(),
                        },
                    );
                }
                _ => {}
            }
        }
        Ok(replay)
    }

    /// The recorded output-contract schema hash, if the run had a contract.
    pub fn output_schema_hash(&self) -> Option<&str> {
        self.output_schema_hash.as_deref()
    }

    fn infer_result(&self, location: &EffectLocation, model: &str) -> Result<crate::op::Response> {
        let effect_id = &location.effect_id.0;
        let call = self.infer_calls.get(effect_id).ok_or_else(|| {
            anyhow!(
                "AgentIR replay missing InferCall for effect {} at {}; {MISSING_EFFECT_HINT}",
                effect_id,
                location_desc(location)
            )
        })?;
        if call.model != model {
            return Err(anyhow!(
                "AgentIR replay diverged at effect {}: expected Infer model {:?} at {}, observed {:?}",
                effect_id,
                call.model,
                location_desc(&call.location),
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

    fn eval_result(
        &self,
        location: &EffectLocation,
        command: &str,
        argv: Option<&[String]>,
    ) -> Result<Value> {
        let effect_id = &location.effect_id.0;
        let call = self.eval_calls.get(effect_id).ok_or_else(|| {
            anyhow!(
                "AgentIR replay missing EvalCall for effect {} at {}; {MISSING_EFFECT_HINT}",
                effect_id,
                location_desc(location)
            )
        })?;
        if call.command != command || call.argv.as_deref() != argv {
            return Err(anyhow!(
                "AgentIR replay diverged at effect {}: expected Eval command {:?} (argv {:?}) at {}, observed {:?} (argv {:?})",
                effect_id,
                call.command,
                call.argv,
                location_desc(&call.location),
                command,
                argv
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
                "AgentIR replay missing RetrieveCall for effect {} at {}; {MISSING_EFFECT_HINT}",
                effect_id,
                location_desc(location)
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
                "AgentIR replay missing StoreCall for effect {} at {}; {MISSING_EFFECT_HINT}",
                effect_id,
                location_desc(location)
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

    /// Replay never invokes a native tool handler: the recorded result is
    /// returned. Name and arguments are checked so a same-site call with a
    /// different payload diverges instead of silently replaying stale data.
    fn tool_result(
        &self,
        location: &EffectLocation,
        name: &str,
        arguments: &Value,
    ) -> Result<Value> {
        let effect_id = &location.effect_id.0;
        let recorded = self.tool_calls.get(effect_id).ok_or_else(|| {
            anyhow!(
                "AgentIR replay missing ToolCall for effect {} at {}; {MISSING_EFFECT_HINT}",
                effect_id,
                location_desc(location)
            )
        })?;
        if recorded.name != name || recorded.arguments != *arguments {
            return Err(anyhow!(
                "AgentIR replay diverged at effect {}: expected tool {:?} with arguments {} at {}, observed {:?} with arguments {}",
                effect_id,
                recorded.name,
                recorded.arguments,
                location_desc(&recorded.location),
                name,
                arguments
            ));
        }
        if let Some(error) = self.tool_errors.get(effect_id) {
            return Err(anyhow!(
                "AgentIR replaying recorded Tool failure at effect {effect_id}: {error}"
            ));
        }
        self.tool_results
            .get(effect_id)
            .cloned()
            .ok_or_else(|| anyhow!("AgentIR replay missing ToolResult for effect {effect_id}"))
    }
}

/// Appended to replay "missing call" errors so a divergence names the id
/// scheme instead of leaving a bare lookup failure.
const MISSING_EFFECT_HINT: &str = "effect ids hash (program hash, kind, site, control path, \
     visit), so an edited program or a different branch/loop path diverges here";

/// Render an effect's site and dynamic path for replay errors: the control
/// path digest is opaque, so also say the visit ordinal and transition
/// count, which humans can map back to loop iterations and turn numbers.
fn location_desc(location: &EffectLocation) -> String {
    let path = if location.dynamic_path.path.is_empty() {
        "entry".to_string()
    } else {
        let digest = &location.dynamic_path.path;
        format!(
            "{}... after {} transitions",
            &digest[..digest.len().min(12)],
            location.dynamic_path.transitions
        )
    };
    format!(
        "block {:?} instruction {} (visit {}, control path {})",
        location.site.block, location.site.instruction_index, location.dynamic_path.visit, path
    )
}

fn require_effect(
    effect: Option<&EffectLocation>,
    expected: EffectKind,
    op_id: u64,
) -> Result<&EffectLocation> {
    let location = effect.ok_or_else(|| {
        anyhow!(
            "AgentIR replay trace {expected:?} call op {op_id} carries no effect identity; \
             only IR-mode traces (whose call events have an `effect` field) are replayable"
        )
    })?;
    if location.kind != expected {
        return Err(anyhow!(
            "AgentIR replay expected {expected:?} effect metadata on op {op_id}, got {:?} at block {:?} instruction {}",
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
    Complete {
        value: Value,
        machine: Machine,
    },
    Suspended {
        checkpoint: IrCheckpoint,
    },
    /// A gated effect was reached with no decision available (t-1308.10,
    /// DR-7): the machine checkpointed mid-turn with the program counter
    /// still at the gated instruction (its visit counter rewound), so
    /// re-entering the checkpoint recomputes the same effect id and — with
    /// a resolution loaded into
    /// [`crate::approval::ApprovalConfig::resolutions`] — executes the
    /// effect (approved) or binds the typed denial value (denied). The
    /// effect did NOT execute; failing closed is the only unattended
    /// behavior.
    AwaitingApproval {
        checkpoint: IrCheckpoint,
        pending: crate::approval::ApprovalRequest,
    },
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
        IrStepOutcome::AwaitingApproval { pending, .. } => Err(awaiting_approval_error(&pending)),
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
        IrStepOutcome::AwaitingApproval { pending, .. } => Err(awaiting_approval_error(&pending)),
    }
}

/// The fail-closed rendering of a pause for entry points that cannot
/// suspend: the gated effect did not execute and no one can approve it
/// here. Drivers that CAN pause (`run_agent_loop_outcome`, the `agent`
/// CLI) handle [`IrStepOutcome::AwaitingApproval`] instead.
pub(crate) fn awaiting_approval_error(pending: &crate::approval::ApprovalRequest) -> anyhow::Error {
    anyhow!(
        "gated {} effect {} requires approval and no resolver is configured; \
         failing closed without executing it (request: {})",
        pending.kind.as_str(),
        pending.effect.effect_id.0,
        pending.request
    )
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
    machine: Machine,
    store: &mut dyn IrStore,
    ir_replay: Option<&IrReplayTrace>,
    max_instructions: Option<usize>,
    gc_state: &mut GcState,
) -> Result<IrStepOutcome> {
    validate_program(&machine.program)?;
    let program_hash = program_hash(&machine.program)?;
    let (outcome, _instructions_used) = run_ir_steps_inner(
        config,
        machine,
        store,
        ir_replay,
        max_instructions,
        gc_state,
        &program_hash,
    )
    .await?;
    Ok(outcome)
}

/// The machine loop proper, shared by top-level runs and Par branch
/// sub-machines (which recurse into it — hence the boxing). The program is
/// validated and hashed once by the public entry point; Par branches run
/// the same program and reuse the parent's hash. Returns the outcome plus
/// how many instructions this machine actually executed, so a Par fork can
/// return a branch's unused pre-split step allocation at the join.
#[async_recursion]
#[allow(clippy::too_many_arguments)]
async fn run_ir_steps_inner(
    config: &SeqConfig,
    mut machine: Machine,
    store: &mut dyn IrStore,
    ir_replay: Option<&'async_recursion IrReplayTrace>,
    max_instructions: Option<usize>,
    gc_state: &mut GcState,
    program_hash: &ProgramHash,
) -> Result<(IrStepOutcome, usize)> {
    let mut instructions_executed = 0usize;
    // The op_id of the most recent default-toolset ("turn") Infer: dispatched
    // child Infers (toolset overridden, e.g. the agent loop's sub-infer) are
    // stamped with it as parent_op_id, so traces carry the parent/child
    // linkage (t-1347). Runtime-only lineage — not part of the checkpoint, so
    // a child re-dispatched after a mid-turn approval pause resumes without a
    // parent link (op_id counters restart with the logger anyway).
    let mut last_turn_infer_op_id: Option<u64> = None;

    loop {
        if max_instructions.is_some_and(|max| instructions_executed >= max) {
            let store = store.in_memory_snapshot().ok_or_else(|| {
                anyhow!("AgentIR instruction-limit checkpoints require an in-memory store snapshot")
            })?;
            return Ok((
                IrStepOutcome::Suspended {
                    checkpoint: IrCheckpoint { machine, store },
                },
                instructions_executed,
            ));
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
            let visit = next_visit(&mut machine, site);
            let dynamic_path = machine.control_path.at_visit(visit);
            let instr = block.instructions[machine.pc].clone();
            if let Some(pending) = execute_instr(
                config,
                &mut machine,
                program_hash,
                site,
                dynamic_path,
                ir_replay,
                instr,
                gc_state,
                &mut last_turn_infer_op_id,
            )
            .await?
            {
                // Pause mid-turn without executing the gated effect: rewind
                // this site's visit counter (taken above by next_visit) so
                // re-entering the checkpoint recomputes the identical
                // effect id, and leave pc pointing at the instruction.
                let key = format!("{}:{}", site.block.0, site.instruction_index);
                if let Some(visits) = machine.effect_visits.get_mut(&key) {
                    *visits = visits.saturating_sub(1);
                }
                let store = store.in_memory_snapshot().ok_or_else(|| {
                    anyhow!("AgentIR approval pauses require an in-memory store snapshot")
                })?;
                return Ok((
                    IrStepOutcome::AwaitingApproval {
                        checkpoint: IrCheckpoint { machine, store },
                        pending,
                    },
                    instructions_executed,
                ));
            }
            machine.pc += 1;
            instructions_executed += 1;
            continue;
        }

        match block.terminator {
            Terminator::Return { value } => {
                let value = eval_expr(&machine.env, &value)?;
                return Ok((
                    IrStepOutcome::Complete { value, machine },
                    instructions_executed,
                ));
            }
            // Every transition folds (from block, arm, to block) into the
            // machine's control path so downstream effect ids encode which
            // way control flow went (branch provenance, loop iterations).
            Terminator::Goto { block, args } => {
                goto_block(&mut machine, block, args, 0).await?;
            }
            Terminator::If {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
            } => {
                let cond = eval_expr(&machine.env, &cond)?;
                let (arm, target, args) = match cond {
                    Value::Bool(true) => (0, then_block, then_args),
                    Value::Bool(false) => (1, else_block, else_args),
                    other => return Err(anyhow!("AgentIR If condition must be bool, got {other}")),
                };
                goto_block(&mut machine, target, args, arm).await?;
            }
            Terminator::Match {
                value,
                arms,
                default,
                default_args,
            } => {
                let value = eval_expr(&machine.env, &value)?;
                let matched = arms
                    .iter()
                    .position(|arm| pattern_matches(&value, &arm.pattern));
                let (arm, target, args) = match matched {
                    Some(index) => (
                        u32::try_from(index).expect("arm count fits in u32"),
                        arms[index].block,
                        arms[index].args.clone(),
                    ),
                    // The default arm's index is arms.len(): distinct from
                    // every explicit arm in the control path.
                    None => match default {
                        Some(target) => (
                            u32::try_from(arms.len()).expect("arm count fits in u32"),
                            target,
                            default_args,
                        ),
                        None => {
                            return Err(anyhow!(
                                "AgentIR Match had no matching arm and no default for {value}"
                            ))
                        }
                    },
                };
                goto_block(&mut machine, target, args, arm).await?;
            }
            // Dynamic-width map-Par (t-1358; docs/GUIDANCE.md §3): fork one
            // branch machine per element of `over`, run them concurrently,
            // and join-all in declaration (element) order.
            Terminator::Par {
                over,
                body,
                body_args,
                join,
                join_args,
            } => {
                let par_block = machine.block;
                let over_value = eval_expr(&machine.env, &over)?;
                let Value::Array(elements) = over_value else {
                    return Err(anyhow!(
                        "AgentIR Par.over must evaluate to a list, got {}",
                        crate::trace::preview(&over_value.to_string(), 128)
                    ));
                };
                let width = elements.len();

                // Pre-split step budget: each branch gets floor(remaining /
                // width), remainder to earlier branches — deterministic, so
                // "which branch hit the limit" never depends on scheduling.
                // Unused allocation is returned at the join (only what a
                // branch actually executed counts against the parent).
                let allocations: Vec<Option<usize>> = match max_instructions {
                    None => vec![None; width],
                    Some(max) => {
                        let remaining = max.saturating_sub(instructions_executed);
                        if width > 0 && remaining < width {
                            if max < width {
                                // A suspend could never make progress: even a
                                // fresh budget of `max` cannot give every
                                // branch one instruction.
                                return Err(anyhow!(
                                    "AgentIR Par width {width} exceeds the whole instruction \
                                     budget {max}; the fork needs at least one instruction \
                                     per branch"
                                ));
                            }
                            // Not enough remaining to fork: checkpoint at the
                            // Par terminator (like the loop-top limit check)
                            // so a resume with a fresh budget re-forks.
                            let store = store.in_memory_snapshot().ok_or_else(|| {
                                anyhow!(
                                    "AgentIR instruction-limit checkpoints require an \
                                     in-memory store snapshot"
                                )
                            })?;
                            return Ok((
                                IrStepOutcome::Suspended {
                                    checkpoint: IrCheckpoint { machine, store },
                                },
                                instructions_executed,
                            ));
                        }
                        let base = remaining / width.max(1);
                        let extra = remaining % width.max(1);
                        (0..width)
                            .map(|index| Some(base + usize::from(index < extra)))
                            .collect()
                    }
                };

                // Bind the body's non-element params once, in the parent env
                // at fork: every branch sees the same values (validation
                // guarantees the arity).
                let body_block = machine
                    .program
                    .blocks
                    .get(&body)
                    .with_context(|| format!("unknown AgentIR block {body:?}"))?;
                let element_param = body_block
                    .params
                    .first()
                    .cloned()
                    .ok_or_else(|| anyhow!("AgentIR Par body {body:?} must declare params"))?;
                let shared_bindings = body_block.params[1..]
                    .iter()
                    .zip(&body_args)
                    .map(|(param, arg)| Ok((param.clone(), eval_expr(&machine.env, arg)?)))
                    .collect::<Result<Vec<_>>>()?;

                // Fork the branch machines BEFORE any branch is scheduled:
                // branch `index` forks the parent control path with the
                // transition (P, arm = index, body), so every effect id a
                // branch will mint derives from the fork point, never from
                // completion order. Branches see a snapshot of the parent
                // env; their mutations stay branch-local and die at the join
                // (only the Return value survives, in the results list).
                let mut branch_runs = Vec::with_capacity(width);
                for (index, element) in elements.iter().enumerate() {
                    let mut env = machine.env.clone();
                    env.insert(element_param.clone(), element.clone());
                    for (param, value) in &shared_bindings {
                        env.insert(param.clone(), value.clone());
                    }
                    let mut control_path = machine.control_path.clone();
                    control_path.transition(
                        par_block,
                        u32::try_from(index).context("AgentIR Par width exceeds u32")?,
                        body,
                    );
                    let branch = Machine {
                        program: machine.program.clone(),
                        block: body,
                        pc: 0,
                        env,
                        effect_visits: machine.effect_visits.clone(),
                        control_path,
                        continuation_stack: vec![],
                        budgets: machine.budgets.clone(),
                    };
                    branch_runs.push(run_par_branch(
                        config,
                        branch,
                        ir_replay,
                        allocations[index],
                        gc_state.discovered_budget,
                        program_hash,
                        index,
                    ));
                }
                // Actual concurrency: all branch futures progress together
                // (their provider calls and Evals overlap); join_all yields
                // outcomes in declaration order regardless of completion
                // order, and never before ALL branches settle — no
                // cancellation in v1.
                let outcomes = futures::future::join_all(branch_runs).await;

                let mut results = Vec::with_capacity(width);
                let mut first_abort: Option<anyhow::Error> = None;
                for outcome in outcomes {
                    instructions_executed += outcome.instructions_used;
                    // Per-site visit counters advanced independently in each
                    // branch from the same fork snapshot; the max per site is
                    // an order-independent merge, keeping post-join visits
                    // deterministic.
                    for (site, visits) in outcome.effect_visits {
                        let merged = machine.effect_visits.entry(site).or_insert(0);
                        *merged = (*merged).max(visits);
                    }
                    match outcome.slot {
                        Ok(value) => results.push(value),
                        // Abort-polarity failures propagate only after all
                        // branches settle (join_all above); the first by
                        // declaration order wins deterministically.
                        Err(err) => {
                            if first_abort.is_none() {
                                first_abort = Some(err);
                            }
                        }
                    }
                }
                if let Some(err) = first_abort {
                    return Err(err);
                }

                // The continuation folds (P, arm = width, join) onto the
                // PARENT path — the branch forks were arms 0..width — at a
                // deterministic moment: after all branches settled.
                machine.control_path.transition(
                    par_block,
                    u32::try_from(width).context("AgentIR Par width exceeds u32")?,
                    join,
                );
                let join_block = machine
                    .program
                    .blocks
                    .get(&join)
                    .with_context(|| format!("unknown AgentIR block {join:?}"))?;
                let results_param = join_block
                    .params
                    .first()
                    .cloned()
                    .ok_or_else(|| anyhow!("AgentIR Par join {join:?} must declare params"))?;
                let mut env = machine.env.clone();
                env.insert(results_param, Value::Array(results));
                for (param, arg) in join_block.params[1..].iter().zip(&join_args) {
                    env.insert(param.clone(), eval_expr(&machine.env, arg)?);
                }
                machine.block = join;
                machine.pc = 0;
                machine.env = env;
            }
        }
        // Block transitions count as steps too: a cycle of blocks with empty
        // instruction lists (pure Goto/If/Match loops) must still hit the
        // instruction limit, or the limit is useless as a watchdog.
        instructions_executed += 1;
    }
}

/// One settled Par branch, in declaration order (t-1358). `slot` is the
/// branch's entry in the join results list: a branch that completed — or
/// whose effect failed under `Bind`, or that exhausted its pre-split step
/// allocation — settles as `Ok(value)` (errors-as-values), while an
/// abort-polarity failure settles as `Err` and propagates only after every
/// sibling has settled, first-by-declaration-order.
struct ParBranchOutcome {
    slot: Result<Value>,
    instructions_used: usize,
    effect_visits: BTreeMap<String, u64>,
}

/// Run one Par branch machine to settlement. Branches are isolated
/// sub-executions: a fresh in-memory store (Store is validation-rejected in
/// bodies, and branch checkpoints never escape), a fresh GC state seeded
/// with the parent's discovered context budget, and the pre-split step
/// allocation as the instruction limit. A branch cannot pause: exhausting
/// its allocation binds as that branch's error value, and reaching an
/// approval gate (only possible dynamically — statically gated effects are
/// validation-rejected in bodies) fails the branch with a clear error.
async fn run_par_branch(
    config: &SeqConfig,
    machine: Machine,
    ir_replay: Option<&IrReplayTrace>,
    allocation: Option<usize>,
    discovered_budget: Option<usize>,
    program_hash: &ProgramHash,
    index: usize,
) -> ParBranchOutcome {
    let mut store = InMemoryStore::new();
    let mut gc_state = GcState {
        discovered_budget,
        ..Default::default()
    };
    match run_ir_steps_inner(
        config,
        machine,
        &mut store,
        ir_replay,
        allocation,
        &mut gc_state,
        program_hash,
    )
    .await
    {
        Ok((IrStepOutcome::Complete { value, machine }, used)) => ParBranchOutcome {
            slot: Ok(value),
            instructions_used: used,
            effect_visits: machine.effect_visits,
        },
        // v1: no mid-Par checkpoints — a branch that hits its pre-split
        // instruction allocation fails as that branch's error value (bind
        // semantics), so siblings and the join still run.
        Ok((IrStepOutcome::Suspended { checkpoint }, used)) => ParBranchOutcome {
            slot: Ok(effect_error_value(&anyhow!(
                "AgentIR Par branch {index} exhausted its instruction allocation of {}",
                allocation.unwrap_or(0)
            ))),
            instructions_used: used,
            effect_visits: checkpoint.machine.effect_visits,
        },
        // v1: approval gates cannot pause inside a Par branch (the machine
        // cannot checkpoint sibling in-flight state). Statically detectable
        // gates are validation-rejected before execution; this is the
        // dynamic case, which fails the branch loudly instead of pausing.
        Ok((IrStepOutcome::AwaitingApproval { pending, .. }, used)) => ParBranchOutcome {
            slot: Err(anyhow!(
                "AgentIR Par branch {index} reached approval-gated {} effect {}; approval \
                 gates cannot pause inside Par branches (v1, t-1358)",
                pending.kind.as_str(),
                pending.effect.effect_id.0
            )),
            instructions_used: used,
            effect_visits: BTreeMap::new(),
        },
        Err(err) => ParBranchOutcome {
            slot: Err(err.context(format!("AgentIR Par branch {index} aborted"))),
            instructions_used: 0,
            effect_visits: BTreeMap::new(),
        },
    }
}

/// Execute one instruction. `Ok(None)` means the instruction ran (or bound
/// an error/denial value); `Ok(Some(pending))` means an approval-gated
/// effect was reached with no decision available — the machine is
/// unmodified and the caller must suspend (t-1308.10).
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
    last_turn_infer_op_id: &mut Option<u64>,
) -> Result<Option<crate::approval::ApprovalRequest>> {
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
            let model = string_expr(&machine.env, &model, "Infer.model")?;
            // The Infer site's policy owns the tool offer (t-1346): the
            // default is the loop's full toolset; an explicit list narrows
            // it to that subset — an empty list offers nothing, which is
            // how the agent loop's sub-infer child stays a bare single
            // completion instead of being teased with tools whose calls
            // would never be dispatched.
            let tool_specs = match &policy.tools {
                None => ir_tool_specs(config),
                Some(names) => ir_tool_specs(config)
                    .into_iter()
                    .filter(|spec| names.contains(&spec.function.name))
                    .collect(),
            };
            let prompt = resolve_prompt(config, &machine.env, prompt)?;
            // Runtime guidance rides ahead of hydration so the fragment
            // lands in the prefix-stable region of the system message,
            // before any per-turn hydrated context.
            let prompt =
                apply_runtime_guidance(config, &machine.program, &tool_specs, prompt).await?;
            let prompt = hydrate_infer_prompt(config, &Value::Null, prompt).await?;
            let mut prompt = maybe_collect_prompt(config, prompt, gc_state).await?;
            let op_id = config.trace.next_op_id();
            // Trace lineage (t-1347): an overridden toolset marks a
            // dispatched child call (the loop's own turn Infers use the
            // default), which carries the dispatching turn Infer's op_id
            // as parent_op_id on all three of its events. Turn Infers
            // update the cursor and carry no parent themselves.
            let parent_op_id = if policy.tools.is_some() {
                *last_turn_infer_op_id
            } else {
                *last_turn_infer_op_id = Some(op_id);
                None
            };
            config
                .trace
                .emit(&Event::InferCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    model: model.clone(),
                    prompt: config.trace_full_payloads.then(|| prompt.clone()),
                    prompt_preview: prompt_preview(&prompt),
                    effect: Some(Box::new(location.clone())),
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
                                .chat(&Model(model.clone()), &tool_specs, &prompt)
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
            let mut response = match result {
                Ok(response) => response,
                Err(err) => {
                    let err = annotate_overflow_failure(err, overflow_cycles);
                    config
                        .trace
                        .emit(&Event::InferError {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id,
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
                        return Ok(None);
                    }
                    return Err(err);
                }
            };
            // Live responses get cost stamped from the registry pricing;
            // replayed responses carry their recorded cost untouched so a
            // replay reproduces the original totals even if today's
            // models.yaml prices differ (t-1334).
            if live {
                crate::cost::price_response(&mut response, &config.pricing, &model);
            }
            config
                .trace
                .emit(&Event::InferResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    response: Some(response.clone()),
                    response_preview: response_preview(&response),
                    input_tokens: response.input_tokens,
                    output_tokens: response.output_tokens,
                    total_tokens: response.total_tokens,
                    cached_input_tokens: response.cached_input_tokens,
                    cost_micro_usd: response.cost_micro_usd,
                    pricing: response.pricing,
                    duration_ms: millis_u64(started.elapsed()),
                    timestamp: Utc::now(),
                })
                .await?;
            machine.env.insert(out, serde_json::to_value(response)?);
        }
        Instr::Eval {
            out,
            request,
            policy,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Eval,
                site,
                dynamic_path.clone(),
            )?;
            // Evaluate the request to its runtime identity: the display
            // command (trace/otel) and, for direct-exec requests, the exact
            // argv (execution + replay identity).
            let (command, argv) = match request {
                EvalRequest::Shell { command } => {
                    (string_expr(&machine.env, &command, "Eval.command")?, None)
                }
                EvalRequest::Argv { argv } => {
                    let argv = argv
                        .iter()
                        .map(|arg| string_expr(&machine.env, arg, "Eval.argv element"))
                        .collect::<Result<Vec<_>>>()?;
                    (crate::op::render_argv(&argv), Some(argv))
                }
            };
            // The approval gate runs before the EvalCall event: a paused or
            // denied effect never dispatches, so it leaves no dangling call
            // in the trace (t-1308.10).
            match approval_gate(
                config,
                ir_replay,
                &location,
                crate::approval::ApprovalKind::Eval,
                serde_json::json!({ "command": command, "argv": argv }),
                policy.require_approval,
            )
            .await?
            {
                GateOutcome::Proceed => {}
                GateOutcome::Deny(denial) => {
                    // Denial is not an abort: bind the typed denial value
                    // (errors-as-values, t-1222) and continue the program.
                    machine.env.insert(out, denial);
                    return Ok(None);
                }
                GateOutcome::Pause(pending) => return Ok(Some(pending)),
            }
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::EvalCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    command: command.clone(),
                    argv: argv.clone(),
                    cwd: config
                        .eval
                        .cwd
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    env_policy: config.eval.env.label(),
                    timeout_ms: millis_u64(config.eval.timeout),
                    effect: Some(Box::new(location.clone())),
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            let result = match ir_replay {
                Some(replay) => replay.eval_result(&location, &command, argv.as_deref()),
                None => match &config.replay {
                    Some(replay) => replay.eval_result(op_id, &command, argv.as_deref()),
                    None => match &argv {
                        Some(argv) => {
                            run_eval_argv_with_env(
                                &config.eval,
                                argv,
                                config.trace.trace_context_env(),
                            )
                            .await
                        }
                        None => {
                            run_eval_with_env(
                                &config.eval,
                                &command,
                                config.trace.trace_context_env(),
                            )
                            .await
                        }
                    },
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
                    effect: Some(Box::new(location.clone())),
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
                        return Ok(None);
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
            // The per-sink write policy is the Store gate (docs/MEMORY.md):
            // a RequireApproval sink pauses exactly like a gated Eval.
            // Replay never consults the live registry — the recorded
            // approval story decides.
            let store_gated = ir_replay.is_none()
                && config.hydration.sink(&sink_name).is_some_and(|sink| {
                    sink.write_policy() == crate::hydration::SinkWritePolicy::RequireApproval
                });
            match approval_gate(
                config,
                ir_replay,
                &location,
                crate::approval::ApprovalKind::Store,
                serde_json::json!({
                    "sink": sink_name,
                    "op": store_op.name(),
                    "id": id_value,
                    "item_preview": crate::trace::preview(&item_value.to_string(), 256),
                    "content_hash": content_hash,
                }),
                store_gated,
            )
            .await?
            {
                GateOutcome::Proceed => {}
                GateOutcome::Deny(denial) => {
                    machine.env.insert(out, denial);
                    return Ok(None);
                }
                GateOutcome::Pause(pending) => return Ok(Some(pending)),
            }
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
                    effect: Some(Box::new(location.clone())),
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
                        return Ok(None);
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
        Instr::Tool {
            out,
            name,
            arguments,
            policy,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Tool,
                site,
                dynamic_path.clone(),
            )?;
            let arguments = eval_expr(&machine.env, &arguments)?;
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::ToolCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    name: name.clone(),
                    arguments: arguments.clone(),
                    effect: Some(Box::new(location.clone())),
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            // Dispatch is registry-only: a Tool effect is executed by its
            // in-process handler or replayed from the trace — never by a
            // shell or any other fallback.
            let result = match ir_replay {
                Some(replay) => replay.tool_result(&location, &name, &arguments),
                None if config.replay.is_some() => Err(anyhow!(
                    "op-layer replay traces do not carry native tool results; \
                     replay this run with an IR replay trace"
                )),
                None => match config.tools.get(&name) {
                    Some(tool) => tool.handler.call(arguments.clone()).await,
                    None => Err(anyhow!(
                        "no native tool {name:?} is registered with the runtime"
                    )),
                },
            };
            let result = match result {
                Ok(result) => result,
                Err(err) => {
                    config
                        .trace
                        .emit(&Event::ToolError {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id: None,
                            name,
                            error: format!("{err:#}"),
                            duration_ms: millis_u64(started.elapsed()),
                            timestamp: Utc::now(),
                        })
                        .await?;
                    // Errors-as-values (t-1222): the loop's dispatch arms use
                    // Bind so a failed handler becomes a tool result the
                    // model can recover from.
                    if policy.on_error == EffectErrorMode::Bind {
                        machine.env.insert(out, effect_error_value(&err));
                        return Ok(None);
                    }
                    return Err(err);
                }
            };
            config
                .trace
                .emit(&Event::ToolResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id: None,
                    name,
                    result: result.clone(),
                    result_preview: crate::trace::preview(&result.to_string(), 512),
                    duration_ms: millis_u64(started.elapsed()),
                    timestamp: Utc::now(),
                })
                .await?;
            machine.env.insert(out, result);
        }
    }
    Ok(None)
}

/// The value an effect binds to its `out` when `on_error: Bind` and the
/// effect failed (t-1222). A small, model-legible envelope mirroring the
/// shell tool's failure shape so the agent loop surfaces it as a tool
/// result the model can read and recover from.
fn effect_error_value(err: &anyhow::Error) -> Value {
    serde_json::json!({ "ok": false, "error": format!("{err:#}") })
}

/// What the approval gate decided for one gated effect.
enum GateOutcome {
    /// Not gated, or approved: dispatch the effect normally.
    Proceed,
    /// Denied: bind this typed denial value to the effect's `out` and
    /// continue — denial is a value, not an abort.
    Deny(Value),
    /// No decision available: suspend the machine without executing
    /// (fail closed).
    Pause(crate::approval::ApprovalRequest),
}

/// The approval gate (t-1308.10, DR-7) — one decision point shared by every
/// gatable effect kind (Eval today, Store via sink policy; an Infer gate
/// would call this same function). Precedence, live:
///
/// 1. not gated → proceed;
/// 2. a pre-loaded resolution for this effect id (resume driver) → emit
///    `ApprovalResolved`, then proceed or bind the denial;
/// 3. the in-process hook → emit `ApprovalRequested`, ask it, emit
///    `ApprovalResolved`, act on the decision;
/// 4. otherwise → emit `ApprovalRequested` and pause. Never auto-approve.
///
/// Under replay the recording is the sole authority: recorded pauses and
/// decisions are reproduced as data (their events re-emitted), a gated
/// effect with no recorded approval story is a divergence, and the recorded
/// request payload is checked against the observed one — that is the only
/// dynamic identity check available for effects that never dispatched
/// (denied or still pending).
async fn approval_gate(
    config: &SeqConfig,
    ir_replay: Option<&IrReplayTrace>,
    location: &EffectLocation,
    kind: crate::approval::ApprovalKind,
    request: Value,
    gated: bool,
) -> Result<GateOutcome> {
    use crate::approval::{denial_value, pending_id_for, ApprovalDecision, ApprovalRequest};

    let effect_id = &location.effect_id.0;
    let run_id: String = config.trace.run_id().into();

    if let Some(replay) = ir_replay {
        let requested = replay.approval_requests.get(effect_id);
        let resolution = replay.approval_resolutions.get(effect_id);
        if requested.is_none() && resolution.is_none() {
            if gated {
                return Err(anyhow!(
                    "AgentIR replay diverged at effect {effect_id}: the effect is \
                     approval-gated but the recording has no approval outcome for it"
                ));
            }
            return Ok(GateOutcome::Proceed);
        }
        if let Some(requested) = requested {
            if requested.request != request {
                return Err(anyhow!(
                    "AgentIR replay diverged at effect {effect_id}: recorded approval \
                     request {} does not match observed request {request}",
                    requested.request
                ));
            }
        }
        let pending_id = requested
            .map(|requested| requested.pending_id.clone())
            .or_else(|| resolution.map(|resolution| resolution.pending_id.clone()))
            .expect("requested or resolution present");
        config
            .trace
            .emit(&Event::ApprovalRequested {
                run_id: run_id.clone(),
                pending_id: pending_id.clone(),
                kind: kind.as_str().into(),
                request: request.clone(),
                effect: Box::new(location.clone()),
                timestamp: Utc::now(),
            })
            .await?;
        let Some(resolution) = resolution else {
            // The recorded run parked here unresolved; the replay reports
            // the same pause as data (the driver never persists or prompts
            // under replay).
            return Ok(GateOutcome::Pause(ApprovalRequest {
                pending_id,
                effect: location.clone(),
                kind,
                request,
            }));
        };
        config
            .trace
            .emit(&Event::ApprovalResolved {
                run_id,
                pending_id: pending_id.clone(),
                effect_id: effect_id.clone(),
                kind: kind.as_str().into(),
                decision: resolution.decision.clone(),
                resolved_by: resolution.resolved_by.clone(),
                reason: resolution.reason.clone(),
                timestamp: Utc::now(),
            })
            .await?;
        return Ok(if resolution.decision == "denied" {
            GateOutcome::Deny(denial_value(
                &pending_id,
                resolution.resolved_by.as_deref(),
                resolution.reason.as_deref(),
            ))
        } else {
            GateOutcome::Proceed
        });
    }

    if !gated {
        return Ok(GateOutcome::Proceed);
    }
    let pending_id = pending_id_for(&run_id, effect_id);
    if let Some(resolution) = config.approvals.resolutions.get(effect_id) {
        // Resume path: the pausing process already emitted the
        // ApprovalRequested into this run's trace; only the resolution is
        // emitted here, at the effect site where it takes effect.
        config
            .trace
            .emit(&Event::ApprovalResolved {
                run_id,
                pending_id: pending_id.clone(),
                effect_id: effect_id.clone(),
                kind: kind.as_str().into(),
                decision: resolution.decision.as_status_str().into(),
                resolved_by: resolution.resolved_by.clone(),
                reason: resolution.reason.clone(),
                timestamp: Utc::now(),
            })
            .await?;
        return Ok(match resolution.decision {
            ApprovalDecision::Approve => GateOutcome::Proceed,
            ApprovalDecision::Deny => GateOutcome::Deny(denial_value(
                &pending_id,
                resolution.resolved_by.as_deref(),
                resolution.reason.as_deref(),
            )),
        });
    }
    config
        .trace
        .emit(&Event::ApprovalRequested {
            run_id: run_id.clone(),
            pending_id: pending_id.clone(),
            kind: kind.as_str().into(),
            request: request.clone(),
            effect: Box::new(location.clone()),
            timestamp: Utc::now(),
        })
        .await?;
    let pending = ApprovalRequest {
        pending_id: pending_id.clone(),
        effect: location.clone(),
        kind,
        request,
    };
    if let Some(hook) = &config.approvals.hook {
        let decision = hook(&pending);
        config
            .trace
            .emit(&Event::ApprovalResolved {
                run_id,
                pending_id: pending_id.clone(),
                effect_id: effect_id.clone(),
                kind: kind.as_str().into(),
                decision: decision.as_status_str().into(),
                resolved_by: Some("hook".into()),
                reason: None,
                timestamp: Utc::now(),
            })
            .await?;
        return Ok(match decision {
            ApprovalDecision::Approve => GateOutcome::Proceed,
            ApprovalDecision::Deny => {
                GateOutcome::Deny(denial_value(&pending_id, Some("hook"), None))
            }
        });
    }
    // Fail closed: no resolution and no hook means the effect does not
    // execute — the machine suspends for a durable, out-of-process decision.
    Ok(GateOutcome::Pause(pending))
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
    use crate::hydration::{Provenance, SinkId, SinkItem};

    // A RequireApproval write policy is enforced by the approval gate ahead
    // of dispatch (t-1308.10): reaching this function means the write is
    // free, approved, or pre-resolved.
    let sink = config
        .hydration
        .sink(sink_name)
        .ok_or_else(|| anyhow!("no sink {sink_name:?} registered"))?;
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

/// Transfer control to `block_id`, binding its params and folding the
/// transition (current block, `arm`, target block) into the machine's
/// control path. `arm` records which way the terminator went — see
/// [`crate::ir::ControlPath::transition`].
async fn goto_block(
    machine: &mut Machine,
    block_id: BlockId,
    args: Vec<Expr>,
    arm: u32,
) -> Result<()> {
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
    machine
        .control_path
        .transition(machine.block, arm, block_id);
    machine.block = block_id;
    machine.pc = 0;
    machine.env = env;
    Ok(())
}

/// Inject the runtime-guidance fragment (t-1359, docs/GUIDANCE.md §4):
/// assemble the capability-keyed operations text from this call's live
/// capabilities and deliver it as a PromptIR Developer/Constraint section
/// (`Static { name: "runtime-guidance" }`). The section's content hash
/// rides the per-InferCall `prompt_ir` trace event exactly like hydration
/// sections, so any trace proves which guidance version the model saw.
///
/// Delivery conditions: the config's guidance switch is on AND the call
/// offers at least one tool — a bare completion (the loop's sub-infer
/// child, `policy.tools: Some([])`) is not operating the runtime and gets
/// no fragment. Within a delivered fragment each block is keyed to its
/// capability; the §2.5 cost block is unconditional.
///
/// Budget-aware delivery (t-1368): the fragment is additionally gated on
/// context-budget headroom — full fragment, the 2-4 sentence minimal
/// core, or nothing, per `guidance::variant_for_budget` (t-1364 showed a
/// ~700-token eviction-protected fragment at 33-44% of a small budget is
/// a priority inversion that destroys the task). Which variant was
/// delivered is trace-visible: the section content hash differs per
/// variant, and the `prompt_ir` event carries `guidance_variant`; a
/// suppressed fragment emits no section and no event, same as opt-out.
///
/// Prompt bytes never affect effect identity or replay (recorded Infer
/// results are matched by effect id), so guidance can be tuned, toggled,
/// or re-keyed without breaking replay of old traces — pinned by
/// `replay_reproduces_results_regardless_of_guidance_setting` below.
async fn apply_runtime_guidance(
    config: &SeqConfig,
    program: &crate::ir::Program,
    tool_specs: &[crate::provider::ToolSpec],
    prompt: Prompt,
) -> Result<Prompt> {
    if !config.guidance.enabled || tool_specs.is_empty() {
        return Ok(prompt);
    }
    let offered = |name: &str| tool_specs.iter().any(|spec| spec.function.name == name);
    let caps = crate::guidance::GuidanceCapabilities {
        infer: offered("infer"),
        memory: offered("remember") && offered("recall"),
        gc: config.gc.enabled(),
        // Strategy-honest citation guidance (GUIDANCE.md gap 6): the
        // protection line renders only where cited-keep actually runs.
        cited_keep: matches!(&config.gc, crate::gc::GcMode::Semantic(gc) if gc.cited_keep),
        approvals: run_has_gated_effects(config, program),
        delegate_models: config.guidance.delegate_models.clone(),
    };
    let Some((fragment, variant)) =
        crate::guidance::budgeted_runtime_guidance_fragment(&caps, config.context_budget)
    else {
        return Ok(prompt);
    };
    let section = crate::guidance::runtime_guidance_section(fragment);
    let prompt_ir = PromptIR::new(prompt, vec![section])?;
    let mut data = serde_json::to_value(prompt_ir.trace(config.trace_full_prompt_ir))?;
    data["guidance_variant"] = variant.name().into();
    config
        .trace
        .emit(&Event::Custom {
            run_id: config.trace.run_id().into(),
            name: "prompt_ir".into(),
            data,
            timestamp: Utc::now(),
        })
        .await?;
    Ok(compile_prompt_ir(&prompt_ir))
}

/// Whether any effect in the run's config is gated behind approval
/// (GUIDANCE.md §2.6's delivery condition): an Eval site with
/// `require_approval`, or a registered sink whose writes require approval.
fn run_has_gated_effects(config: &SeqConfig, program: &crate::ir::Program) -> bool {
    program.blocks.values().any(|block| {
        block
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Eval { policy, .. } if policy.require_approval))
    }) || config
        .hydration
        .sinks()
        .iter()
        .any(|sink| sink.write_policy() == crate::hydration::SinkWritePolicy::RequireApproval)
}

/// Tool list shown to the provider: shell + infer always; remember/recall
/// appear automatically whenever a memory backend is registered (settled
/// question 6 — an unreachable sink is a trap, so registration IS the
/// exposure switch); native tools ride with their registry entries
/// (t-1308.7 — same principle).
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
                description: crate::guidance::REMEMBER_TOOL_DESCRIPTION.into(),
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
                description: crate::guidance::RECALL_TOOL_DESCRIPTION.into(),
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
    specs.extend(config.tools.specs());
    specs
}

fn base_ir_tool_specs() -> Vec<crate::provider::ToolSpec> {
    vec![
        crate::provider::ToolSpec {
            kind: "function".into(),
            function: crate::provider::ToolFunctionSpec {
                name: "shell".into(),
                description: crate::guidance::SHELL_TOOL_DESCRIPTION.into(),
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
                description: crate::guidance::INFER_TOOL_DESCRIPTION.into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "model": { "type": "string" },
                        "prompt": { "type": "string" },
                        "context_refs": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": crate::guidance::INFER_CONTEXT_REFS_DESCRIPTION
                        }
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
        Expr::Concat { left, right } => {
            let left = eval_expr(env, left)?;
            let right = eval_expr(env, right)?;
            match (left, right) {
                (Value::Array(mut left), Value::Array(right)) => {
                    left.extend(right);
                    Ok(Value::Array(left))
                }
                (left, right) => Err(anyhow!(
                    "AgentIR Concat expected two arrays, got {left} and {right}"
                )),
            }
        }
        Expr::SelectToolResults { history, ids } => {
            let history = env
                .get(history)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", history))?;
            let messages = history.as_array().ok_or_else(|| {
                anyhow!("AgentIR SelectToolResults expected array, got {history}")
            })?;
            let ids = eval_expr(env, ids)?;
            Ok(select_tool_results(messages, &ids))
        }
    }
}

/// Resolve tool-call ids against a chat-message history (t-1344): the model
/// references prior tool results by the ids it minted itself
/// (`tool_calls[].id` on assistant messages, echoed as `tool_call_id` on the
/// result). Returns `{"messages": [...], "missing": [...]}`: one user-role
/// message per resolved id — the referenced result verbatim under a short
/// provenance header — in first-occurrence order (duplicates dropped), plus
/// every id (or non-string element, serialized) that resolved to nothing.
/// Total on model-shaped garbage by design: bad refs land in `missing` for
/// the program to answer as a tool result, never an interpreter error.
fn select_tool_results(messages: &[Value], ids: &Value) -> Value {
    // Index the history once: id -> tool name (from the assistant call) and
    // id -> result content (from the tool message).
    let mut names = std::collections::BTreeMap::<&str, &str>::new();
    let mut results = std::collections::BTreeMap::<&str, String>::new();
    for message in messages {
        for call in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let (Some(id), Some(name)) = (
                call.get("id").and_then(Value::as_str),
                call.get("name").and_then(Value::as_str),
            ) {
                names.insert(id, name);
            }
        }
        if let Some(id) = message.get("tool_call_id").and_then(Value::as_str) {
            let content = match message.get("content") {
                Some(Value::String(content)) => content.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            results.insert(id, content);
        }
    }

    let mut resolved = Vec::new();
    let mut missing = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let elements = match ids.as_array() {
        Some(elements) => elements.as_slice(),
        // A non-array `context_refs` is one unresolvable ref: its
        // serialization, so the error result names what was sent.
        None => std::slice::from_ref(ids),
    };
    for element in elements {
        let Some(id) = element.as_str() else {
            missing.push(Value::String(element.to_string()));
            continue;
        };
        if !seen.insert(id) {
            continue;
        }
        match results.get(id) {
            Some(content) => {
                let name = names.get(id).copied().unwrap_or("unknown");
                resolved.push(serde_json::json!({
                    "role": "user",
                    "content": format!(
                        "Referenced result of tool call {id} ({name}):\n{content}"
                    ),
                }));
            }
            None => missing.push(Value::String(id.into())),
        }
    }
    serde_json::json!({ "messages": resolved, "missing": missing })
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
    use crate::ir::InferPolicy;
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
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
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
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
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
            pricing: Default::default(),
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
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        // Guidance off: this test pins exact prompt bytes across the
        // continuation, which the runtime-guidance system section would
        // prepend to (its delivery is pinned by its own tests below).
        let mut config = config(provider.clone());
        config.guidance = crate::guidance::RuntimeGuidance::disabled();
        let (value, _machine) = run_ir_sequential(&config, machine).await?;

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
            control_path: Default::default(),
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
            control_path: Default::default(),
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
            control_path: Default::default(),
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
                control_path: Default::default(),
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

    /// The Infer site's policy owns the provider tool offer (t-1346):
    /// default = the loop's full toolset, an explicit list = exactly that
    /// subset, and an empty list = no tools (the sub-infer child shape —
    /// its tool calls would never be dispatched, so it must not be offered
    /// any).
    #[tokio::test]
    async fn infer_policy_toolset_controls_the_provider_offer() -> Result<()> {
        struct ToolRecordingProvider {
            offers: Mutex<Vec<Vec<String>>>,
        }

        #[async_trait]
        impl ChatProvider for ToolRecordingProvider {
            async fn chat(
                &self,
                _model: &Model,
                tools: &[ToolSpec],
                _messages: &[ChatMessage],
            ) -> Result<Response> {
                self.offers.lock().unwrap().push(
                    tools
                        .iter()
                        .map(|spec| spec.function.name.clone())
                        .collect(),
                );
                Ok(response("ok"))
            }
        }

        let infer = |out: &str, tools: Option<Vec<String>>| Instr::Infer {
            out: Var(out.into()),
            model: Expr::Value(Value::String("mock".into())),
            prompt: PromptRef::Inline(vec![ChatMessage::user(out.to_owned())]),
            policy: crate::ir::InferPolicy {
                tools,
                ..Default::default()
            },
        };
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![
                    infer("full", None),
                    infer("subset", Some(vec!["infer".into()])),
                    infer("bare", Some(vec![])),
                ],
                terminator: Terminator::Return {
                    value: Expr::Value(Value::Null),
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("toolset-policy".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            effect_visits: BTreeMap::new(),
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };
        let provider = Arc::new(ToolRecordingProvider {
            offers: Mutex::new(Vec::new()),
        });
        run_ir_sequential(&config(provider.clone()), machine).await?;

        let offers = provider.offers.lock().unwrap().clone();
        assert_eq!(
            offers,
            vec![
                vec!["shell".to_owned(), "infer".to_owned()],
                vec!["infer".to_owned()],
                Vec::<String>::new(),
            ]
        );
        Ok(())
    }

    // --- runtime guidance delivery (t-1359, docs/GUIDANCE.md §4) ---------

    fn guidance_probe_machine() -> Machine {
        let infer = |out: &str, tools: Option<Vec<String>>| Instr::Infer {
            out: Var(out.into()),
            model: Expr::Value(Value::String("mock".into())),
            prompt: PromptRef::Inline(vec![ChatMessage::user(out.to_owned())]),
            policy: crate::ir::InferPolicy {
                tools,
                ..Default::default()
            },
        };
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![infer("turn", None), infer("bare", Some(vec![]))],
                terminator: Terminator::Return {
                    value: Expr::Field {
                        base: Var("turn".into()),
                        field: "content".into(),
                    },
                },
            },
        );
        Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("guidance-probe".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            effect_visits: BTreeMap::new(),
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    /// Default-on delivery, keyed to the call's tool offer: the turn Infer
    /// (full toolset) gets the fragment as a system section; the bare
    /// sub-infer child (`tools: Some([])`) is not operating the runtime
    /// and gets nothing.
    #[tokio::test]
    async fn runtime_guidance_rides_tool_bearing_infers_only() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("a"), response("b")]));
        run_ir_sequential(&config(provider.clone()), guidance_probe_machine()).await?;

        let prompts = provider.prompts();
        assert_eq!(prompts.len(), 2);
        let system = prompts[0][0].content.as_deref().unwrap_or_default();
        assert_eq!(prompts[0][0].role, "system");
        assert!(system.contains("## runtime-guidance"));
        assert!(system.contains(crate::guidance::DELEGATION_BLOCK));
        assert!(system.contains(crate::guidance::COST_BLOCK));
        // Capabilities not live in this config render no block: no memory
        // backend, no GC, no gated effect.
        assert!(!system.contains("persistent memory"));
        assert!(!system.contains("[frame"));
        assert!(!system.contains("human approval"));

        assert!(
            !prompts[1].iter().any(|message| message
                .content
                .as_deref()
                .unwrap_or_default()
                .contains("runtime-guidance")),
            "a bare completion child must get no fragment"
        );
        Ok(())
    }

    /// Config-level capability keying: memory backend registered, semantic
    /// GC with cited-keep, and a gated Eval in the program (unreachable —
    /// gating is a config fact, not an execution fact) light up their
    /// blocks, including the strategy-conditional cited-keep line.
    #[tokio::test]
    async fn runtime_guidance_blocks_key_to_live_capabilities() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("a"), response("b")]));
        let mut config = config(provider.clone());
        config.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(memory_dir()));
        config.gc = GcMode::Semantic(crate::gc::SemanticGc {
            preserve_prefix: true,
            recent_window: 8,
            similarity_floor: 0.35,
            embedder: None,
            cited_keep: true,
        });
        let mut machine = guidance_probe_machine();
        machine.program.blocks.insert(
            BlockId(9),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Eval {
                    out: Var("gated".into()),
                    request: crate::ir::EvalRequest::Shell {
                        command: Expr::Value(Value::String("true".into())),
                    },
                    policy: crate::ir::EvalPolicy {
                        timeout_ms: None,
                        require_approval: true,
                    },
                }],
                terminator: Terminator::Return {
                    value: Expr::Value(Value::Null),
                },
            },
        );
        run_ir_sequential(&config, machine).await?;

        let prompts = provider.prompts();
        let system = prompts[0][0].content.as_deref().unwrap_or_default();
        // The §2.2 memory-discipline block is demoted to draft (t-1368,
        // failed its t-1364 A/B): memory capability no longer adds it. The
        // §2.4 GC block keeps its memory-keyed variant.
        assert!(!system.contains(crate::guidance::MEMORY_BLOCK));
        assert!(system.contains(crate::guidance::GC_BLOCK_WITH_MEMORY));
        assert!(system.contains(crate::guidance::GC_CITED_KEEP_LINE));
        assert!(system.contains(crate::guidance::APPROVAL_BLOCK));
        Ok(())
    }

    /// Budget-aware delivery end to end (t-1368): the same config at the
    /// three budget regimes ships the full fragment, the minimal core
    /// (with `guidance_variant` on the prompt_ir event), or nothing (no
    /// section, no event — the t-1364 failure regime).
    #[tokio::test]
    async fn runtime_guidance_variant_tracks_budget_headroom() -> Result<()> {
        async fn run_at(budget: usize) -> Result<(String, Vec<Event>)> {
            let provider = Arc::new(MockProvider::new(vec![response("a"), response("b")]));
            let trace = test_trace();
            let trace_path = trace.path().clone();
            let mut config = config_with_trace(provider.clone(), trace);
            config.gc = GcMode::Stack(crate::gc::StackFrameGc {
                preserve_prefix: true,
            });
            config.context_budget = budget;
            run_ir_sequential(&config, guidance_probe_machine()).await?;
            let system = provider.prompts()[0][0]
                .content
                .as_deref()
                .unwrap_or_default()
                .to_string();
            Ok((system, TraceLogger::read_events(&trace_path).await?))
        }
        let variant_attr = |events: &[Event]| -> Option<String> {
            events.iter().find_map(|event| match event {
                Event::Custom { name, data, .. } if name == "prompt_ir" => {
                    data["guidance_variant"].as_str().map(str::to_string)
                }
                _ => None,
            })
        };

        // Deployment-sized budget: the full fragment.
        let (system, events) = run_at(200_000).await?;
        assert!(system.contains(crate::guidance::DELEGATION_BLOCK));
        assert!(system.contains(crate::guidance::GC_BLOCK_WITHOUT_MEMORY));
        assert_eq!(variant_attr(&events).as_deref(), Some("full"));

        // Mid budget: the minimal core replaces the fragment.
        let (system, events) = run_at(8_000).await?;
        assert!(system.contains(crate::guidance::MINIMAL_GC_CORE));
        assert!(!system.contains(crate::guidance::DELEGATION_BLOCK));
        assert!(!system.contains(crate::guidance::COST_BLOCK));
        assert_eq!(variant_attr(&events).as_deref(), Some("minimal"));

        // The t-1364 failure regime: nothing — no section, no event.
        let (system, events) = run_at(2_000).await?;
        assert!(!system.contains("runtime-guidance"));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Custom { name, .. } if name == "prompt_ir")),
            "a suppressed fragment must emit no prompt_ir event"
        );
        Ok(())
    }

    /// The cited-keep line is strategy-honest end to end (GUIDANCE.md gap
    /// 6): the default stack strategy has no citation protection, so the
    /// line must not render under it.
    #[tokio::test]
    async fn runtime_guidance_omits_cited_keep_line_under_stack_gc() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("a"), response("b")]));
        let mut config = config(provider.clone());
        config.gc = GcMode::Stack(crate::gc::StackFrameGc {
            preserve_prefix: true,
        });
        run_ir_sequential(&config, guidance_probe_machine()).await?;

        let system = provider.prompts()[0][0]
            .content
            .as_deref()
            .unwrap_or_default()
            .to_string();
        assert!(system.contains(crate::guidance::GC_BLOCK_WITHOUT_MEMORY));
        assert!(!system.contains("marks that result as load-bearing"));
        Ok(())
    }

    /// Opt-out is explicit and total: no fragment in any prompt, no
    /// `prompt_ir` trace event.
    #[tokio::test]
    async fn runtime_guidance_opt_out_removes_fragment_and_trace_event() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("a"), response("b")]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = config_with_trace(provider.clone(), trace);
        config.guidance = crate::guidance::RuntimeGuidance::disabled();
        run_ir_sequential(&config, guidance_probe_machine()).await?;

        for prompt in provider.prompts() {
            for message in prompt {
                assert!(!message
                    .content
                    .as_deref()
                    .unwrap_or_default()
                    .contains("runtime-guidance"));
            }
        }
        let events = TraceLogger::read_events(&trace_path).await?;
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::Custom { name, .. } if name == "prompt_ir")));
        Ok(())
    }

    /// The fragment's section summary — origin `Static {
    /// "runtime-guidance" }`, content hash — rides a `prompt_ir` Custom
    /// event per delivering InferCall, so any trace proves which guidance
    /// version the model saw (docs/GUIDANCE.md §4).
    #[tokio::test]
    async fn runtime_guidance_section_hash_rides_the_trace() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("a"), response("b")]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = config_with_trace(provider, trace);
        run_ir_sequential(&config, guidance_probe_machine()).await?;

        let events = TraceLogger::read_events(&trace_path).await?;
        let sections: Vec<Value> = events
            .iter()
            .filter_map(|event| match event {
                Event::Custom { name, data, .. } if name == "prompt_ir" => {
                    Some(data["sections"].as_array().cloned().unwrap_or_default())
                }
                _ => None,
            })
            .flatten()
            .collect();
        // Exactly one delivering Infer (the turn call; the bare child gets
        // none), carrying exactly the runtime-guidance section.
        assert_eq!(sections.len(), 1, "{sections:?}");
        assert_eq!(
            sections[0]["source"]["origin"]["Static"]["name"],
            Value::String("runtime-guidance".into())
        );
        assert!(sections[0]["hash"]
            .as_str()
            .unwrap_or_default()
            .starts_with("sha256:"));
        assert_eq!(sections[0]["role"], Value::String("Developer".into()));
        Ok(())
    }

    /// Recorded Infer results replay by effect id regardless of prompt
    /// bytes: a trace recorded WITH guidance replays cleanly with guidance
    /// disabled (and would the other way around) — guidance wording is
    /// tunable without breaking replay of old traces.
    #[tokio::test]
    async fn replay_reproduces_results_regardless_of_guidance_setting() -> Result<()> {
        // One machine value, cloned per run: Inline prompts carry fresh
        // ChatMessage ids per construction, which land in the serialized
        // program (and so its hash) — reusing the value keeps the program
        // identity fixed while the guidance setting varies.
        let machine = guidance_probe_machine();
        let provider = Arc::new(MockProvider::new(vec![response("a"), response("b")]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config_on = config_with_trace(provider, trace);
        let (recorded_value, _machine) = run_ir_sequential(&config_on, machine.clone()).await?;

        let events = TraceLogger::read_events(&trace_path).await?;
        let replay = IrReplayTrace::from_events(&events)?;
        let mut config_off = config(Arc::new(crate::provider::ReplayOnlyProvider));
        config_off.guidance = crate::guidance::RuntimeGuidance::disabled();
        let mut store = InMemoryStore::new();
        let (replayed_value, _machine) = run_ir_sequential_with_store_and_replay(
            &config_off,
            machine,
            &mut store,
            Some(&replay),
        )
        .await?;

        assert_eq!(replayed_value, recorded_value);
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
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let _ = run_ir_sequential(&config_with_trace(provider, trace), machine).await?;
        let events = TraceLogger::read_events(trace_path).await?;
        let locations: Vec<&EffectLocation> = events
            .iter()
            .filter_map(|event| match event {
                Event::RetrieveCall { effect, .. } => effect.as_deref(),
                _ => None,
            })
            .collect();

        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].kind, EffectKind::Retrieve);
        assert_eq!(locations[0].site.block, BlockId(0));
        assert_eq!(locations[0].site.instruction_index, 0);
        // An entry-block effect runs at the control-path root on visit 0 —
        // the one dynamic path computable without simulating the machine
        // (DynamicPath::at_entry, the `agent ir-effect` command).
        assert_eq!(locations[0].dynamic_path, DynamicPath::at_entry(0));
        Ok(())
    }

    /// Effect identity rides directly on the call events (t-1057): no
    /// side-channel `ir_effect` Custom event exists, and the `effect` field
    /// carries the id and location that replay keys on.
    #[tokio::test]
    async fn call_events_carry_effect_identity_directly() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("hi")]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![
                    Instr::Infer {
                        out: Var("a".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Inline(vec![ChatMessage::user("hello")]),
                        policy: Default::default(),
                    },
                    Instr::Eval {
                        out: Var("b".into()),
                        request: EvalRequest::Shell {
                            command: Expr::Value(Value::String("true".into())),
                        },
                        policy: Default::default(),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("b".into())),
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("direct-effect".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            effect_visits: BTreeMap::new(),
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let _ = run_ir_sequential(&config_with_trace(provider, trace), machine).await?;
        let events = TraceLogger::read_events(trace_path).await?;

        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Custom { name, .. } if name == "ir_effect")),
            "the side-channel ir_effect event was removed in favor of direct fields"
        );
        let infer_effect = events
            .iter()
            .find_map(|event| match event {
                Event::InferCall { effect, .. } => effect.as_deref(),
                _ => None,
            })
            .expect("InferCall carries its effect location directly");
        assert_eq!(infer_effect.kind, EffectKind::Infer);
        assert_eq!(infer_effect.site.block, BlockId(0));
        assert_eq!(infer_effect.site.instruction_index, 0);
        assert!(infer_effect.effect_id.0.starts_with("sha256:"));
        let eval_effect = events
            .iter()
            .find_map(|event| match event {
                Event::EvalCall { effect, .. } => effect.as_deref(),
                _ => None,
            })
            .expect("EvalCall carries its effect location directly");
        assert_eq!(eval_effect.kind, EffectKind::Eval);
        assert_eq!(eval_effect.site.instruction_index, 1);
        assert_ne!(infer_effect.effect_id, eval_effect.effect_id);
        Ok(())
    }

    /// One Eval(Argv) whose last element is dynamic (a block param), so
    /// record and replay share a program hash while the argv can differ.
    fn argv_eval_machine(payload: &str) -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![Var("payload".into())],
                instructions: vec![Instr::Eval {
                    out: Var("result".into()),
                    request: EvalRequest::Argv {
                        argv: vec![
                            Expr::Value(Value::String("/bin/sh".into())),
                            Expr::Value(Value::String("-c".into())),
                            Expr::Value(Value::String(r#"printf %s "$0""#.into())),
                            Expr::Var(Var("payload".into())),
                        ],
                    },
                    policy: Default::default(),
                }],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("result".into())),
                },
            },
        );
        machine_with_env(
            "argv-eval",
            blocks,
            BTreeMap::from([(Var("payload".into()), Value::String(payload.into()))]),
        )
    }

    /// Direct-exec Evals: the spaced, `$`-laden argv element arrives at the
    /// child as one verbatim argument (no shell splitting or expansion — the
    /// child /bin/sh just prints its `$0`), and the EvalCall trace event
    /// records the argv faithfully with a quoted display command.
    #[tokio::test]
    async fn ir_argv_eval_executes_directly_and_records_argv() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let machine = argv_eval_machine("one arg $HOME");

        let (value, _) = run_ir_sequential(
            &config_with_trace(Arc::new(MockProvider::new(vec![])), trace),
            machine,
        )
        .await?;
        assert_eq!(value["ok"], Value::Bool(true));
        assert_eq!(value["stdout"], Value::String("one arg $HOME".into()));

        let events = TraceLogger::read_events(trace_path).await?;
        let (command, argv, effect) = events
            .iter()
            .find_map(|event| match event {
                Event::EvalCall {
                    command,
                    argv,
                    effect,
                    ..
                } => Some((command.clone(), argv.clone(), effect.clone())),
                _ => None,
            })
            .expect("argv Eval emits an EvalCall");
        assert_eq!(
            argv.as_deref(),
            Some(
                &[
                    "/bin/sh".to_string(),
                    "-c".into(),
                    r#"printf %s "$0""#.into(),
                    "one arg $HOME".into(),
                ][..]
            )
        );
        assert_eq!(command, r#"/bin/sh -c 'printf %s "$0"' 'one arg $HOME'"#);
        assert_eq!(
            effect.expect("IR EvalCall carries effect identity").kind,
            EffectKind::Eval
        );
        Ok(())
    }

    /// Argv Evals replay exactly like shell Evals: the recorded result is
    /// returned without executing, and a same-site call whose dynamic argv
    /// changed diverges loudly instead of replaying a stale result.
    #[tokio::test]
    async fn ir_argv_eval_replays_recorded_result_and_detects_argv_divergence() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let _ = run_ir_sequential(
            &config_with_trace(Arc::new(MockProvider::new(vec![])), trace),
            argv_eval_machine("recorded-payload"),
        )
        .await?;

        // Swap the recorded EvalResult for a sentinel: if replay returns it,
        // the value provably came from the recording, not a re-execution.
        let events: Vec<Event> = TraceLogger::read_events(trace_path)
            .await?
            .into_iter()
            .map(|event| match event {
                Event::EvalResult {
                    run_id,
                    op_id,
                    parent_op_id,
                    command,
                    result: _,
                    duration_ms,
                    truncated_stdout,
                    truncated_stderr,
                    timestamp,
                } => Event::EvalResult {
                    run_id,
                    op_id,
                    parent_op_id,
                    command,
                    result: serde_json::json!({ "ok": true, "stdout": "from-recording" }),
                    duration_ms,
                    truncated_stdout,
                    truncated_stderr,
                    timestamp,
                },
                other => other,
            })
            .collect();
        let replay = IrReplayTrace::from_events(&events)?;

        let mut store = InMemoryStore::new();
        let (replayed, _) = run_ir_sequential_with_store_and_replay(
            &config(Arc::new(MockProvider::new(vec![]))),
            argv_eval_machine("recorded-payload"),
            &mut store,
            Some(&replay),
        )
        .await?;
        assert_eq!(replayed["stdout"], Value::String("from-recording".into()));

        let mut store = InMemoryStore::new();
        let err = run_ir_sequential_with_store_and_replay(
            &config(Arc::new(MockProvider::new(vec![]))),
            argv_eval_machine("changed-payload"),
            &mut store,
            Some(&replay),
        )
        .await
        .expect_err("a changed argv element must not replay");
        let message = err.to_string();
        assert!(message.contains("AgentIR replay diverged"), "{message}");
        assert!(message.contains("argv"), "{message}");
        assert!(message.contains("changed-payload"), "{message}");
        Ok(())
    }

    fn machine_with_env(
        name: &str,
        blocks: BTreeMap<BlockId, crate::ir::Block>,
        env: BTreeMap<Var, Value>,
    ) -> Machine {
        Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId(name.into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env,
            effect_visits: BTreeMap::new(),
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    fn retrieve_instr(out: &str) -> Instr {
        Instr::Retrieve {
            out: Var(out.into()),
            query: Expr::Value(Value::String("q".into())),
            kind: None,
            max_bytes: None,
            policy: Default::default(),
        }
    }

    async fn recorded_effects(machine: Machine) -> Result<Vec<EffectLocation>> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let _ = run_ir_sequential(
            &config_with_trace(Arc::new(MockProvider::new(vec![])), trace),
            machine,
        )
        .await?;
        Ok(TraceLogger::read_events(trace_path)
            .await?
            .iter()
            .filter_map(|event| match event {
                Event::RetrieveCall { effect, .. } => effect.as_deref().cloned(),
                _ => None,
            })
            .collect())
    }

    /// Loop iterations: the same effect site executed twice around a
    /// back-edge gets distinct ids, with the iteration visible both as the
    /// visit ordinal and as a changed control-path digest (t-1058).
    #[tokio::test]
    async fn loop_iterations_get_distinct_path_sensitive_effect_ids() -> Result<()> {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![Var("i".into())],
                instructions: vec![retrieve_instr("hits")],
                terminator: Terminator::If {
                    cond: Expr::Eq {
                        left: Box::new(Expr::Var(Var("i".into()))),
                        right: Box::new(Expr::Value(Value::Number(1.into()))),
                    },
                    then_block: BlockId(1),
                    then_args: vec![],
                    else_block: BlockId(0),
                    else_args: vec![Expr::Add {
                        left: Box::new(Expr::Var(Var("i".into()))),
                        right: Box::new(Expr::Value(Value::Number(1.into()))),
                    }],
                },
            },
        );
        blocks.insert(
            BlockId(1),
            crate::ir::Block {
                params: vec![],
                instructions: vec![],
                terminator: Terminator::Return {
                    value: Expr::Value(Value::String("done".into())),
                },
            },
        );
        let machine = machine_with_env(
            "loop-ids",
            blocks,
            BTreeMap::from([(Var("i".into()), Value::Number(0.into()))]),
        );

        let effects = recorded_effects(machine).await?;
        assert_eq!(effects.len(), 2, "two loop iterations, two Retrieves");
        assert_eq!(effects[0].site, effects[1].site, "same static site");
        assert_ne!(effects[0].effect_id, effects[1].effect_id);
        assert_eq!(effects[0].dynamic_path.visit, 0);
        assert_eq!(effects[1].dynamic_path.visit, 1);
        // Iteration 0 runs at the root (no transitions yet); iteration 1
        // folded the back-edge into the path.
        assert_eq!(effects[0].dynamic_path.path, "");
        assert_eq!(effects[0].dynamic_path.transitions, 0);
        assert_ne!(effects[1].dynamic_path.path, "");
        assert_eq!(effects[1].dynamic_path.transitions, 1);
        Ok(())
    }

    /// Branch provenance: then vs else visits to the same downstream effect
    /// site (after the paths rejoin!) produce ids that encode which arm was
    /// taken, and replay refuses to feed a then-path recording to an
    /// else-path run (t-1058).
    fn diamond_machine(flag: bool) -> Machine {
        // 0 --If(flag)--> 1 or 2, both Goto 3; 3 holds the effect.
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![Var("flag".into())],
                instructions: vec![],
                terminator: Terminator::If {
                    cond: Expr::Var(Var("flag".into())),
                    then_block: BlockId(1),
                    then_args: vec![],
                    else_block: BlockId(2),
                    else_args: vec![],
                },
            },
        );
        for id in [1u32, 2] {
            blocks.insert(
                BlockId(id),
                crate::ir::Block {
                    params: vec![],
                    instructions: vec![],
                    terminator: Terminator::Goto {
                        block: BlockId(3),
                        args: vec![],
                    },
                },
            );
        }
        blocks.insert(
            BlockId(3),
            crate::ir::Block {
                params: vec![],
                instructions: vec![retrieve_instr("hits")],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("hits".into())),
                },
            },
        );
        machine_with_env(
            "diamond-ids",
            blocks,
            BTreeMap::from([(Var("flag".into()), Value::Bool(flag))]),
        )
    }

    #[tokio::test]
    async fn branch_arms_reaching_the_same_site_get_distinct_effect_ids() -> Result<()> {
        let then_effects = recorded_effects(diamond_machine(true)).await?;
        let else_effects = recorded_effects(diamond_machine(false)).await?;
        assert_eq!((then_effects.len(), else_effects.len()), (1, 1));
        let (then_effect, else_effect) = (&then_effects[0], &else_effects[0]);
        assert_eq!(then_effect.site, else_effect.site, "same join-block site");
        assert_eq!(then_effect.dynamic_path.visit, 0);
        assert_eq!(else_effect.dynamic_path.visit, 0);
        assert_eq!(then_effect.dynamic_path.transitions, 2);
        assert_ne!(
            then_effect.dynamic_path.path, else_effect.dynamic_path.path,
            "the arm taken at block 0 must be encoded in the path"
        );
        assert_ne!(then_effect.effect_id, else_effect.effect_id);
        Ok(())
    }

    #[tokio::test]
    async fn replaying_a_then_path_recording_against_an_else_path_run_diverges() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let _ = run_ir_sequential(
            &config_with_trace(Arc::new(MockProvider::new(vec![])), trace),
            diamond_machine(true),
        )
        .await?;
        let replay = IrReplayTrace::load(trace_path).await?;

        let mut store = InMemoryStore::new();
        let err = run_ir_sequential_with_store_and_replay(
            &config(Arc::new(MockProvider::new(vec![]))),
            diamond_machine(false),
            &mut store,
            Some(&replay),
        )
        .await
        .expect_err("an effect reached along a different branch must not replay");
        let message = err.to_string();
        assert!(
            message.contains("AgentIR replay missing RetrieveCall"),
            "{message}"
        );
        assert!(
            message.contains("control path"),
            "divergence must mention the id scheme: {message}"
        );
        Ok(())
    }

    /// Par control-path forks (t-1058 scaffold, realized by t-1358): branch
    /// `b` of a Par at block `P` forks the parent path with the transition
    /// `(P, arm = b, body)` BEFORE any branch is scheduled, so sibling
    /// branches derive distinct, deterministic, scheduling-independent
    /// digests from the same parent; the continuation after the join folds
    /// `(P, arm = width, join)` onto the parent path.
    #[tokio::test]
    async fn par_branches_fork_the_control_path() -> Result<()> {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![],
                terminator: Terminator::Par {
                    over: Expr::Value(serde_json::json!(["a", "b"])),
                    body: BlockId(1),
                    body_args: vec![],
                    join: BlockId(2),
                    join_args: vec![],
                },
            },
        );
        blocks.insert(
            BlockId(1),
            crate::ir::Block {
                params: vec![Var("element".into())],
                instructions: vec![retrieve_instr("hits")],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("hits".into())),
                },
            },
        );
        blocks.insert(
            BlockId(2),
            crate::ir::Block {
                params: vec![Var("results".into())],
                instructions: vec![retrieve_instr("post_join")],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("results".into())),
                },
            },
        );
        let machine = machine_with_env("par-ids", blocks, BTreeMap::new());

        let effects = recorded_effects(machine).await?;
        assert_eq!(effects.len(), 3, "one effect per branch plus one post-join");

        // The exact digests are computable without running the machine:
        // the fork happens at block 0 (parent path empty) with arm = the
        // element index, and the join continuation folds arm = width.
        let mut fork0 = crate::ir::ControlPath::default();
        fork0.transition(BlockId(0), 0, BlockId(1));
        let mut fork1 = crate::ir::ControlPath::default();
        fork1.transition(BlockId(0), 1, BlockId(1));
        let mut joined = crate::ir::ControlPath::default();
        joined.transition(BlockId(0), 2, BlockId(2));

        // Trace order may interleave branch events, so match by digest.
        let paths = effects
            .iter()
            .map(|effect| effect.dynamic_path.path.as_str())
            .collect::<Vec<_>>();
        for expected in [&fork0, &fork1, &joined] {
            assert!(
                paths.contains(&expected.digest.as_str()),
                "expected digest {} among recorded paths {paths:?}",
                expected.digest
            );
        }
        let mut ids = effects
            .iter()
            .map(|effect| effect.effect_id.clone())
            .collect::<Vec<_>>();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 3, "sibling + post-join effect ids are distinct");
        Ok(())
    }

    /// Provider for Par tests: echoes the last user message back as
    /// "echo:<content>". A `barrier` (a rendezvous sized to the Par width)
    /// proves the branches are actually in flight together — sequential
    /// dispatch would deadlock, which the tests convert to a loud failure
    /// via a timeout. Contents listed in `slow` sleep after the rendezvous
    /// so their branch settles LAST despite being declared first; contents
    /// in `fail` return an error instead of a response.
    struct ParProvider {
        barrier: Option<tokio::sync::Barrier>,
        slow: Vec<String>,
        fail: Vec<String>,
    }

    impl ParProvider {
        fn echo() -> Self {
            Self {
                barrier: None,
                slow: vec![],
                fail: vec![],
            }
        }
    }

    #[async_trait]
    impl ChatProvider for ParProvider {
        async fn chat(
            &self,
            _model: &Model,
            _tools: &[ToolSpec],
            messages: &[ChatMessage],
        ) -> Result<Response> {
            let content = messages
                .last()
                .and_then(|message| message.content.clone())
                .unwrap_or_default();
            if let Some(barrier) = &self.barrier {
                barrier.wait().await;
            }
            if self.slow.contains(&content) {
                tokio::time::sleep(std::time::Duration::from_millis(75)).await;
            }
            if self.fail.contains(&content) {
                return Err(anyhow!("ParProvider failing {content} on purpose"));
            }
            Ok(response(&format!("echo:{content}")))
        }
    }

    /// entry(0) --Par over--> body(1): build a one-message prompt from the
    /// element, Infer, return the full response; join(2) returns the
    /// results list.
    fn infer_par_machine(over: Expr, on_error: EffectErrorMode) -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![],
                terminator: Terminator::Par {
                    over,
                    body: BlockId(1),
                    body_args: vec![],
                    join: BlockId(2),
                    join_args: vec![],
                },
            },
        );
        blocks.insert(
            BlockId(1),
            crate::ir::Block {
                params: vec![Var("element".into())],
                instructions: vec![
                    Instr::Let {
                        out: Var("prompt".into()),
                        expr: Expr::Array(vec![Expr::Object(BTreeMap::from([
                            ("role".into(), Expr::Value(Value::String("user".into()))),
                            ("content".into(), Expr::Var(Var("element".into()))),
                        ]))]),
                    },
                    Instr::Infer {
                        out: Var("reply".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Var(Var("prompt".into())),
                        policy: InferPolicy {
                            on_error,
                            ..Default::default()
                        },
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("reply".into())),
                },
            },
        );
        blocks.insert(
            BlockId(2),
            crate::ir::Block {
                params: vec![Var("results".into())],
                instructions: vec![],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("results".into())),
                },
            },
        );
        machine_with_env("par-infer", blocks, BTreeMap::new())
    }

    /// entry(0) --Par over--> trivial body(1) returning its element;
    /// join(2) returns the results list.
    fn value_par_machine(over: Expr) -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![],
                terminator: Terminator::Par {
                    over,
                    body: BlockId(1),
                    body_args: vec![],
                    join: BlockId(2),
                    join_args: vec![],
                },
            },
        );
        blocks.insert(
            BlockId(1),
            crate::ir::Block {
                params: vec![Var("element".into())],
                instructions: vec![],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("element".into())),
                },
            },
        );
        blocks.insert(
            BlockId(2),
            crate::ir::Block {
                params: vec![Var("results".into())],
                instructions: vec![],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("results".into())),
                },
            },
        );
        machine_with_env("par-values", blocks, BTreeMap::new())
    }

    /// Join-all in declaration order under an adversarial completion order:
    /// branch 0 is forced to settle LAST (both branches rendezvous at a
    /// barrier — proving they run concurrently — then branch 0 sleeps), yet
    /// the results list still follows element order. A sequential dispatch
    /// would deadlock at the barrier; the timeout turns that into a failure.
    #[tokio::test]
    async fn par_joins_in_declaration_order_under_adversarial_completion() -> Result<()> {
        let provider = Arc::new(ParProvider {
            barrier: Some(tokio::sync::Barrier::new(2)),
            slow: vec!["first".into()],
            fail: vec![],
        });
        let machine = infer_par_machine(
            Expr::Value(serde_json::json!(["first", "second"])),
            EffectErrorMode::Abort,
        );
        let (value, _) = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            run_ir_sequential(&config(provider), machine),
        )
        .await
        .expect("branches must dispatch concurrently (sequential dispatch deadlocks here)")?;

        assert_eq!(value[0]["content"], Value::String("echo:first".into()));
        assert_eq!(value[1]["content"], Value::String("echo:second".into()));
        Ok(())
    }

    /// Errors-as-values inside a branch: a Bind-polarity Infer failure
    /// settles as that branch's error value in its declaration-order slot,
    /// and the join still runs with every sibling's result intact.
    #[tokio::test]
    async fn par_bind_failure_lands_in_its_slot_and_join_runs() -> Result<()> {
        let provider = Arc::new(ParProvider {
            barrier: None,
            slow: vec![],
            fail: vec!["bad".into()],
        });
        let machine = infer_par_machine(
            Expr::Value(serde_json::json!(["ok", "bad", "also-ok"])),
            EffectErrorMode::Bind,
        );
        let (value, _) = run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(value[0]["content"], Value::String("echo:ok".into()));
        assert_eq!(value[1]["ok"], Value::Bool(false));
        assert!(
            value[1]["error"]
                .as_str()
                .unwrap_or_default()
                .contains("failing bad on purpose"),
            "{value}"
        );
        assert_eq!(value[2]["content"], Value::String("echo:also-ok".into()));
        Ok(())
    }

    /// Abort-polarity failures propagate only after ALL branches settle,
    /// and the first failure by DECLARATION order is the reported one even
    /// when it completes last: branch 0 fails slowly, branch 1 fails fast,
    /// both error events reach the trace, and the surfaced error names
    /// branch 0.
    #[tokio::test]
    async fn par_abort_reports_first_declared_failure_after_all_settle() -> Result<()> {
        let provider = Arc::new(ParProvider {
            barrier: None,
            slow: vec!["bad-first".into()],
            fail: vec!["bad-first".into(), "bad-second".into()],
        });
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let machine = infer_par_machine(
            Expr::Value(serde_json::json!(["bad-first", "bad-second"])),
            EffectErrorMode::Abort,
        );
        let err = run_ir_sequential(&config_with_trace(provider, trace), machine)
            .await
            .expect_err("both branches abort");
        assert!(
            err.to_string().contains("Par branch 0"),
            "declaration order, not completion order, picks the error: {err:#}"
        );
        assert!(
            format!("{err:#}").contains("failing bad-first on purpose"),
            "{err:#}"
        );

        // Both branches settled before the abort propagated: each recorded
        // its own InferError.
        let errors = TraceLogger::read_events(trace_path)
            .await?
            .iter()
            .filter(|event| matches!(event, Event::InferError { .. }))
            .count();
        assert_eq!(errors, 2, "siblings run to completion; no cancellation");
        Ok(())
    }

    /// Width edges: an empty list forks nothing and the join still runs
    /// (with []); width 1 is the degenerate map.
    #[tokio::test]
    async fn par_width_zero_and_one() -> Result<()> {
        let (empty, _) = run_ir_sequential(
            &config(Arc::new(ParProvider::echo())),
            value_par_machine(Expr::Value(serde_json::json!([]))),
        )
        .await?;
        assert_eq!(empty, serde_json::json!([]));

        let (single, _) = run_ir_sequential(
            &config(Arc::new(ParProvider::echo())),
            value_par_machine(Expr::Value(serde_json::json!(["only"]))),
        )
        .await?;
        assert_eq!(single, serde_json::json!(["only"]));
        Ok(())
    }

    #[tokio::test]
    async fn par_over_must_evaluate_to_a_list() -> Result<()> {
        let err = run_ir_sequential(
            &config(Arc::new(ParProvider::echo())),
            value_par_machine(Expr::Value(Value::String("not-a-list".into()))),
        )
        .await
        .expect_err("a non-list over is a runtime type error");
        assert!(
            err.to_string().contains("Par.over must evaluate to a list"),
            "{err}"
        );
        Ok(())
    }

    /// Effect ids are assigned at fork, before any branch is scheduled: two
    /// runs of the same Par program — with the adversarial slow-first
    /// scheduling — mint identical id sets.
    #[tokio::test]
    async fn par_effect_ids_are_deterministic_across_runs_and_schedules() -> Result<()> {
        let mut id_sets = Vec::new();
        for _ in 0..2 {
            let provider = Arc::new(ParProvider {
                barrier: Some(tokio::sync::Barrier::new(2)),
                slow: vec!["first".into()],
                fail: vec![],
            });
            let trace = test_trace();
            let trace_path = trace.path().clone();
            let machine = infer_par_machine(
                Expr::Value(serde_json::json!(["first", "second"])),
                EffectErrorMode::Abort,
            );
            tokio::time::timeout(
                std::time::Duration::from_secs(30),
                run_ir_sequential(&config_with_trace(provider, trace), machine),
            )
            .await
            .expect("concurrent dispatch")?;
            let mut ids = TraceLogger::read_events(trace_path)
                .await?
                .iter()
                .filter_map(|event| match event {
                    Event::InferCall { effect, .. } => {
                        effect.as_deref().map(|e| e.effect_id.0.clone())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            ids.sort();
            assert_eq!(ids.len(), 2);
            id_sets.push(ids);
        }
        assert_eq!(
            id_sets[0], id_sets[1],
            "branch effect ids must not depend on scheduling"
        );
        Ok(())
    }

    /// Replay is id-keyed and order-independent: a recording whose branch
    /// events are deliberately re-interleaved (branch groups reversed —
    /// each event stays atomic and every result still follows its own
    /// call, which the serialized appender guarantees) replays to the same
    /// value without touching the provider.
    #[tokio::test]
    async fn par_replay_is_order_independent() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let machine = infer_par_machine(
            Expr::Value(serde_json::json!(["a", "b"])),
            EffectErrorMode::Abort,
        );
        let (recorded, _) = run_ir_sequential(
            &config_with_trace(Arc::new(ParProvider::echo()), trace),
            machine.clone(),
        )
        .await?;

        // Shuffle: group events by op id (first-seen order), then emit the
        // groups REVERSED — branch b's call/result now precede branch a's.
        let events = TraceLogger::read_events(trace_path).await?;
        fn op_id_of(event: &Event) -> Option<u64> {
            match event {
                Event::InferCall { op_id, .. }
                | Event::InferResult { op_id, .. }
                | Event::InferError { op_id, .. } => Some(*op_id),
                _ => None,
            }
        }
        let mut groups: Vec<(Option<u64>, Vec<Event>)> = Vec::new();
        for event in events {
            let key = op_id_of(&event);
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, group)) => group.push(event),
                None => groups.push((key, vec![event])),
            }
        }
        groups.reverse();
        let shuffled = groups
            .into_iter()
            .flat_map(|(_, group)| group)
            .collect::<Vec<_>>();
        let replay = IrReplayTrace::from_events(&shuffled)?;

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

    /// Pre-split step budgets: with 3 remaining instructions and width 2 the
    /// split is [2, 1] (floor + remainder to earlier branches). The body
    /// costs one Let + Return: branch 0 completes inside its allocation,
    /// branch 1 exhausts its single step and fails AS A VALUE in its slot
    /// (bind semantics — the join still runs). The used steps count against
    /// the parent, which then suspends at the join block; the resumed
    /// checkpoint completes.
    #[tokio::test]
    async fn par_branch_budget_split_is_deterministic_and_exhaustion_binds() -> Result<()> {
        let mut machine = value_par_machine(Expr::Value(serde_json::json!(["a", "b"])));
        machine
            .program
            .blocks
            .get_mut(&BlockId(1))
            .unwrap()
            .instructions = vec![Instr::Let {
            out: Var("scratch".into()),
            expr: Expr::Value(Value::Null),
        }];

        let config = config(Arc::new(ParProvider::echo()));
        let outcome = run_ir_steps(&config, machine, 3).await?;
        let IrStepOutcome::Suspended { checkpoint } = outcome else {
            panic!("used branch steps count against the parent: {outcome:?}");
        };
        assert_eq!(
            checkpoint.machine.block,
            BlockId(2),
            "the join is reached before the parent suspends"
        );

        let resumed = run_ir_steps(&config, checkpoint.machine, 10).await?;
        let IrStepOutcome::Complete { value, .. } = resumed else {
            panic!("resume completes: {resumed:?}");
        };
        assert_eq!(value[0], Value::String("a".into()));
        assert_eq!(value[1]["ok"], Value::Bool(false));
        assert!(
            value[1]["error"]
                .as_str()
                .unwrap_or_default()
                .contains("exhausted its instruction allocation of 1"),
            "branch 1 (the smaller pre-split share) fails as a value: {value}"
        );
        Ok(())
    }

    /// Unused allocation returns at the join: two zero-instruction branches
    /// each hold an allocation they never use, so a budget that could not
    /// cover full consumption still completes the join block's work.
    #[tokio::test]
    async fn par_unused_branch_allocation_returns_at_the_join() -> Result<()> {
        let mut machine = value_par_machine(Expr::Value(serde_json::json!(["a", "b"])));
        machine
            .program
            .blocks
            .get_mut(&BlockId(2))
            .unwrap()
            .instructions = vec![Instr::Let {
            out: Var("scratch".into()),
            expr: Expr::Value(Value::Null),
        }];

        // Budget 3: fork splits [2, 1]; branches use 0; the Par transition
        // costs 1 and the join's Let costs 1 — this completes only because
        // the unused pre-split steps returned at the join.
        let outcome = run_ir_steps(&config(Arc::new(ParProvider::echo())), machine, 3).await?;
        let IrStepOutcome::Complete { value, .. } = outcome else {
            panic!("unused allocation must return at the join: {outcome:?}");
        };
        assert_eq!(value, serde_json::json!(["a", "b"]));
        Ok(())
    }

    /// A width the whole budget cannot cover is a deterministic error (a
    /// suspend could never progress); a width the REMAINING budget cannot
    /// cover suspends at the Par terminator and re-forks on resume.
    #[tokio::test]
    async fn par_fork_budget_edges_error_or_suspend() -> Result<()> {
        let config = config(Arc::new(ParProvider::echo()));

        let err = run_ir_steps(
            &config,
            value_par_machine(Expr::Value(serde_json::json!(["a", "b"]))),
            1,
        )
        .await
        .expect_err("width 2 cannot fork under a whole budget of 1");
        assert!(
            err.to_string()
                .contains("exceeds the whole instruction budget"),
            "{err}"
        );

        // Two entry instructions eat the budget down to 1 remaining at the
        // fork: suspend (max itself could cover the width), then resume.
        let mut machine = value_par_machine(Expr::Value(serde_json::json!(["a", "b"])));
        machine
            .program
            .blocks
            .get_mut(&BlockId(0))
            .unwrap()
            .instructions = vec![
            Instr::Let {
                out: Var("x".into()),
                expr: Expr::Value(Value::Null),
            },
            Instr::Let {
                out: Var("y".into()),
                expr: Expr::Value(Value::Null),
            },
        ];
        let outcome = run_ir_steps(&config, machine, 3).await?;
        let IrStepOutcome::Suspended { checkpoint } = outcome else {
            panic!("remaining 1 < width 2 suspends before forking: {outcome:?}");
        };
        assert_eq!(checkpoint.machine.block, BlockId(0));
        let resumed = run_ir_steps(&config, checkpoint.machine, 10).await?;
        let IrStepOutcome::Complete { value, .. } = resumed else {
            panic!("resume re-forks: {resumed:?}");
        };
        assert_eq!(value, serde_json::json!(["a", "b"]));
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
            control_path: Default::default(),
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
            IrStepOutcome::Complete { .. } | IrStepOutcome::AwaitingApproval { .. } => {
                panic!("expected suspension after one instruction")
            }
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
            control_path: Default::default(),
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
            control_path: Default::default(),
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
            control_path: Default::default(),
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

    // ---- approval/pause protocol (t-1308.10, DR-7) ----

    use crate::approval::{
        is_denial_value, pending_id_for, ApprovalConfig, ApprovalDecision, ApprovalStore,
        PendingEffectRecord, PendingStatus,
    };

    /// A single-block program: one gated Eval, then a Let that proves the
    /// program continued past the gate, returning both. The command comes
    /// in through the entry block's `cmd` param (seeded via machine env),
    /// so tests can vary the request without changing the program hash.
    fn gated_eval_machine(name: &str) -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![Var("cmd".into())],
                instructions: vec![
                    Instr::Eval {
                        out: Var("eval_out".into()),
                        request: crate::ir::EvalRequest::Shell {
                            command: Expr::Var(Var("cmd".into())),
                        },
                        policy: crate::ir::EvalPolicy {
                            require_approval: true,
                            ..Default::default()
                        },
                    },
                    Instr::Let {
                        out: Var("after".into()),
                        expr: Expr::Value(Value::Bool(true)),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Object(BTreeMap::from([
                        ("eval".into(), Expr::Var(Var("eval_out".into()))),
                        ("after".into(), Expr::Var(Var("after".into()))),
                    ])),
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
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    fn append_command(marker: &std::path::Path) -> String {
        format!("printf ran >> {}", marker.display())
    }

    fn temp_marker(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("agent-approval-{tag}-{}", Uuid::new_v4()))
    }

    fn pending_record_for(
        run_id: &str,
        pending: &crate::approval::ApprovalRequest,
    ) -> PendingEffectRecord {
        PendingEffectRecord {
            pending_id: pending.pending_id.clone(),
            run_id: run_id.into(),
            turn_id: Some(format!("{run_id}-t0")),
            effect_id: pending.effect.effect_id.0.clone(),
            program_hash: pending.effect.program_hash.0.clone(),
            kind: pending.kind,
            request: pending.request.clone(),
            created_ts: Utc::now(),
            status: PendingStatus::AwaitingApproval,
            resolved_ts: None,
            resolved_by: None,
            reason: None,
            runtime: Some(serde_json::json!({ "model": "mock" })),
        }
    }

    /// The full one-shot pause/approve lifecycle across a simulated process
    /// restart: a gated Eval pauses without executing, the pause persists
    /// as a pending record (file shape pinned here — Ben reviews this
    /// before dashboard consumption) plus a machine checkpoint, and a fresh
    /// "process" (all-new store/config objects, state loaded from disk
    /// only) approves, claims, resumes, executes the command exactly once,
    /// and completes the run. Double-resolution and double-claim both fail.
    #[tokio::test]
    async fn gated_eval_pauses_persists_and_approve_after_restart_executes_once() -> Result<()> {
        let marker = temp_marker("approve");
        let dir = std::env::temp_dir().join(format!("agent-approvals-{}", Uuid::new_v4()));
        let run_id = "run-approve";
        let trace_path =
            std::env::temp_dir().join(format!("agent-approval-trace-{}.jsonl", Uuid::new_v4()));

        // --- process 1: run until the gate pauses ---
        let pending = {
            let provider = Arc::new(MockProvider::new(vec![]));
            let config = config_with_trace(
                provider,
                TraceLogger::new(run_id.to_string(), trace_path.clone()),
            );
            let mut machine = gated_eval_machine("gated-approve");
            machine
                .env
                .insert(Var("cmd".into()), Value::String(append_command(&marker)));
            let mut store = InMemoryStore::new();
            let outcome =
                run_ir_steps_with_store_and_replay(&config, machine, &mut store, None, None)
                    .await?;
            let IrStepOutcome::AwaitingApproval {
                checkpoint,
                pending,
            } = outcome
            else {
                panic!("expected AwaitingApproval, got {outcome:?}");
            };
            assert!(!marker.exists(), "gated effect must not have executed");
            assert_eq!(pending.kind, crate::approval::ApprovalKind::Eval);
            assert_eq!(
                pending.pending_id,
                pending_id_for(run_id, &pending.effect.effect_id.0)
            );
            let approvals = ApprovalStore::new(&dir);
            approvals
                .write_pending(&pending_record_for(run_id, &pending), &checkpoint)
                .await?;
            pending
        };

        // The on-disk record shape (the dashboard/API contract).
        let raw: Value = serde_json::from_str(
            &tokio::fs::read_to_string(dir.join(format!("{}.json", pending.pending_id))).await?,
        )?;
        assert_eq!(raw["pending_id"], pending.pending_id.as_str());
        assert_eq!(raw["run_id"], run_id);
        assert_eq!(raw["turn_id"], "run-approve-t0");
        assert_eq!(raw["effect_id"], pending.effect.effect_id.0.as_str());
        assert_eq!(raw["program_hash"], pending.effect.program_hash.0.as_str());
        assert_eq!(raw["kind"], "eval");
        assert_eq!(
            raw["request"]["command"],
            Value::String(append_command(&marker))
        );
        assert_eq!(raw["status"], "awaiting_approval");
        assert!(raw["created_ts"].is_string());
        assert!(raw.get("resolved_ts").is_none(), "unresolved: field absent");
        assert!(
            dir.join(format!("{}.machine.json", pending.pending_id))
                .exists(),
            "mid-turn machine checkpoint persisted alongside"
        );

        // --- "restart": fresh objects, state comes from disk only ---
        let approvals = ApprovalStore::new(&dir);
        let record = approvals
            .resolve(
                &pending.pending_id,
                ApprovalDecision::Approve,
                Some("ben".into()),
                None,
            )
            .await?;
        let checkpoint = approvals.claim_checkpoint(&pending.pending_id).await?;
        let provider = Arc::new(MockProvider::new(vec![]));
        let mut config = config_with_trace(
            provider,
            TraceLogger::new(run_id.to_string(), trace_path.clone()),
        );
        config.approvals = ApprovalConfig::default();
        config.approvals.resolutions.insert(
            record.effect_id.clone(),
            ApprovalStore::resolution_of(&record)?,
        );
        let mut store = checkpoint.store.clone();
        let outcome =
            run_ir_steps_with_store_and_replay(&config, checkpoint.machine, &mut store, None, None)
                .await?;
        let IrStepOutcome::Complete { value, .. } = outcome else {
            panic!("expected completion after approval, got {outcome:?}");
        };
        assert_eq!(value["after"], Value::Bool(true), "program continued");
        assert_eq!(value["eval"]["ok"], Value::Bool(true));
        assert_eq!(
            tokio::fs::read_to_string(&marker).await?,
            "ran",
            "the approved effect executed exactly once"
        );

        // Exactly-once guards: no re-resolution, no re-claim.
        assert!(approvals
            .resolve(&pending.pending_id, ApprovalDecision::Deny, None, None)
            .await
            .is_err());
        assert!(approvals
            .claim_checkpoint(&pending.pending_id)
            .await
            .is_err());

        // Trace: the pause emitted ApprovalRequested; the resume appended
        // ApprovalResolved at the effect site, under the same run id.
        let events = TraceLogger::read_events(&trace_path).await?;
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ApprovalRequested { pending_id, .. } if *pending_id == pending.pending_id
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ApprovalResolved { decision, resolved_by, .. }
                if decision == "approved" && resolved_by.as_deref() == Some("ben")
        )));
        Ok(())
    }

    /// Denial is a value, not an abort: the denied Eval binds the typed
    /// denial (errors-as-values) and the instructions after the gate still
    /// run. The command never executes.
    #[tokio::test]
    async fn denied_gated_eval_binds_typed_value_and_program_continues() -> Result<()> {
        let marker = temp_marker("deny");
        let run_id = "run-deny";
        let trace_path =
            std::env::temp_dir().join(format!("agent-approval-trace-{}.jsonl", Uuid::new_v4()));
        let provider = Arc::new(MockProvider::new(vec![]));
        let config = config_with_trace(
            provider,
            TraceLogger::new(run_id.to_string(), trace_path.clone()),
        );
        let mut machine = gated_eval_machine("gated-deny");
        machine
            .env
            .insert(Var("cmd".into()), Value::String(append_command(&marker)));
        let mut store = InMemoryStore::new();
        let outcome =
            run_ir_steps_with_store_and_replay(&config, machine, &mut store, None, None).await?;
        let IrStepOutcome::AwaitingApproval {
            checkpoint,
            pending,
        } = outcome
        else {
            panic!("expected AwaitingApproval, got {outcome:?}");
        };

        // Resolve as denied (via the durable store, like the CLI) and resume.
        let dir = std::env::temp_dir().join(format!("agent-approvals-{}", Uuid::new_v4()));
        let approvals = ApprovalStore::new(&dir);
        approvals
            .write_pending(&pending_record_for(run_id, &pending), &checkpoint)
            .await?;
        let record = approvals
            .resolve(
                &pending.pending_id,
                ApprovalDecision::Deny,
                Some("ben".into()),
                Some("not on prod".into()),
            )
            .await?;
        let checkpoint = approvals.claim_checkpoint(&pending.pending_id).await?;
        let provider = Arc::new(MockProvider::new(vec![]));
        let mut config = config_with_trace(
            provider,
            TraceLogger::new(run_id.to_string(), trace_path.clone()),
        );
        config.approvals.resolutions.insert(
            record.effect_id.clone(),
            ApprovalStore::resolution_of(&record)?,
        );
        let mut store = checkpoint.store.clone();
        let outcome =
            run_ir_steps_with_store_and_replay(&config, checkpoint.machine, &mut store, None, None)
                .await?;
        let IrStepOutcome::Complete { value, .. } = outcome else {
            panic!("expected completion after denial, got {outcome:?}");
        };
        assert_eq!(value["after"], Value::Bool(true), "program continued");
        assert!(is_denial_value(&value["eval"]), "typed denial: {value}");
        assert_eq!(value["eval"]["ok"], Value::Bool(false));
        assert_eq!(value["eval"]["approval"]["status"], "denied");
        assert_eq!(value["eval"]["approval"]["reason"], "not on prod");
        assert!(!marker.exists(), "denied effect must never execute");
        Ok(())
    }

    /// Unattended fail-closed: no hook, no resolution — the entry points
    /// that cannot pause return an error and the effect does not execute.
    /// Never auto-approve.
    #[tokio::test]
    async fn unattended_gated_eval_fails_closed() -> Result<()> {
        let marker = temp_marker("closed");
        let provider = Arc::new(MockProvider::new(vec![]));
        let mut machine = gated_eval_machine("gated-closed");
        machine
            .env
            .insert(Var("cmd".into()), Value::String(append_command(&marker)));
        let err = run_ir_sequential(&config(provider), machine)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires approval"), "{err}");
        assert!(err.contains("failing closed"), "{err}");
        assert!(!marker.exists(), "fail closed means the effect never ran");
        Ok(())
    }

    /// The in-process hook decides at the effect site: approve executes
    /// inline (no pause), and both events land in the trace.
    #[tokio::test]
    async fn approval_hook_decides_at_the_effect_site() -> Result<()> {
        let marker = temp_marker("hook");
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let provider = Arc::new(MockProvider::new(vec![]));
        let mut config = config_with_trace(provider, trace);
        config.approvals.hook = Some(Arc::new(|_request: &crate::approval::ApprovalRequest| {
            ApprovalDecision::Approve
        }));
        let mut machine = gated_eval_machine("gated-hook");
        machine
            .env
            .insert(Var("cmd".into()), Value::String(append_command(&marker)));
        let (value, _machine) = run_ir_sequential(&config, machine).await?;
        assert_eq!(value["eval"]["ok"], Value::Bool(true));
        assert_eq!(tokio::fs::read_to_string(&marker).await?, "ran");

        let events = TraceLogger::read_events(&trace_path).await?;
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::ApprovalRequested { .. })));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ApprovalResolved { decision, resolved_by, .. }
                if decision == "approved" && resolved_by.as_deref() == Some("hook")
        )));
        Ok(())
    }

    /// Record an approved and a denied gated run, then replay both traces:
    /// the pause/decision are reproduced as data (both events re-emitted),
    /// nothing pauses, nothing prompts, and the approved command is NOT
    /// re-executed (the recorded EvalResult is served by effect id).
    #[tokio::test]
    async fn replay_reproduces_approved_and_denied_outcomes_as_data() -> Result<()> {
        for decision in [ApprovalDecision::Approve, ApprovalDecision::Deny] {
            let marker = temp_marker("replay");
            let record_trace = test_trace();
            let record_path = record_trace.path().clone();
            let provider = Arc::new(MockProvider::new(vec![]));
            let mut config = config_with_trace(provider, record_trace);
            config.approvals.hook = Some(Arc::new(move |_: &crate::approval::ApprovalRequest| {
                decision
            }));
            let mut machine = gated_eval_machine("gated-replay");
            machine
                .env
                .insert(Var("cmd".into()), Value::String(append_command(&marker)));
            let (recorded_value, _machine) = run_ir_sequential(&config, machine.clone()).await?;
            let executions_after_recording = marker.exists() as usize;

            let replay = IrReplayTrace::load(&record_path).await?;
            let replay_trace = test_trace();
            let replay_path = replay_trace.path().clone();
            // No hook and no resolutions: the recording is the sole
            // authority under replay.
            let replay_config =
                config_with_trace(Arc::new(MockProvider::new(vec![])), replay_trace);
            let mut store = InMemoryStore::new();
            let outcome = run_ir_steps_with_store_and_replay(
                &replay_config,
                machine,
                &mut store,
                Some(&replay),
                None,
            )
            .await?;
            let IrStepOutcome::Complete { value, .. } = outcome else {
                panic!("replay must not pause: {outcome:?}");
            };
            assert_eq!(value, recorded_value, "replay reproduces the outcome");
            assert_eq!(
                marker.exists() as usize,
                executions_after_recording,
                "replay never re-executes the effect"
            );
            let events = TraceLogger::read_events(&replay_path).await?;
            assert!(events
                .iter()
                .any(|event| matches!(event, Event::ApprovalRequested { .. })));
            let expected = decision.as_status_str();
            assert!(events.iter().any(|event| matches!(
                event,
                Event::ApprovalResolved { decision, .. } if decision == expected
            )));
        }
        Ok(())
    }

    /// Replay divergence on mismatched effect identity: same program, same
    /// effect id, but the observed gated request differs from the recorded
    /// one (the request payload is the identity check for gated effects).
    #[tokio::test]
    async fn replay_diverges_on_mismatched_gated_request() -> Result<()> {
        let marker = temp_marker("diverge");
        let record_trace = test_trace();
        let record_path = record_trace.path().clone();
        let provider = Arc::new(MockProvider::new(vec![]));
        let mut config = config_with_trace(provider, record_trace);
        config.approvals.hook = Some(Arc::new(|_: &crate::approval::ApprovalRequest| {
            ApprovalDecision::Deny
        }));
        let mut machine = gated_eval_machine("gated-diverge");
        machine
            .env
            .insert(Var("cmd".into()), Value::String(append_command(&marker)));
        run_ir_sequential(&config, machine.clone()).await?;

        // Same program (hash unchanged: the command is env data), different
        // observed request.
        let replay = IrReplayTrace::load(&record_path).await?;
        machine
            .env
            .insert(Var("cmd".into()), Value::String("printf other".to_string()));
        let replay_config = config_with_trace(Arc::new(MockProvider::new(vec![])), test_trace());
        let mut store = InMemoryStore::new();
        let err = run_ir_steps_with_store_and_replay(
            &replay_config,
            machine,
            &mut store,
            Some(&replay),
            None,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("diverged"), "{err}");
        assert!(err.contains("does not match observed request"), "{err}");
        Ok(())
    }

    /// A sink registered with `SinkWritePolicy::RequireApproval` gates the
    /// Store effect exactly like a gated Eval: pause without a resolver,
    /// execute on hook-approve, typed denial on hook-deny.
    struct GatedSink {
        items: Arc<Mutex<Vec<Value>>>,
    }

    #[async_trait]
    impl crate::hydration::HydrationSink for GatedSink {
        fn name(&self) -> &str {
            "gated"
        }

        fn kind(&self) -> crate::hydration::SourceKind {
            crate::hydration::SourceKind::Semantic
        }

        fn write_policy(&self) -> crate::hydration::SinkWritePolicy {
            crate::hydration::SinkWritePolicy::RequireApproval
        }

        async fn store(
            &self,
            item: crate::hydration::SinkItem,
        ) -> Result<crate::hydration::SinkId> {
            self.items.lock().unwrap().push(item.payload);
            Ok(crate::hydration::SinkId("gated-1".into()))
        }

        async fn update(
            &self,
            _id: &crate::hydration::SinkId,
            _item: crate::hydration::SinkItem,
        ) -> Result<()> {
            unimplemented!("not exercised")
        }

        async fn delete(&self, _id: &crate::hydration::SinkId) -> Result<()> {
            unimplemented!("not exercised")
        }
    }

    fn gated_store_machine(name: &str) -> Machine {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Store {
                    out: Var("stored".into()),
                    sink: Expr::Value(Value::String("gated".into())),
                    op: crate::ir::StoreOp::Create,
                    id: None,
                    item: Expr::Value(serde_json::json!({ "note": "hello" })),
                    policy: Default::default(),
                }],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("stored".into())),
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
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn require_approval_sink_gates_store_effects() -> Result<()> {
        // No resolver: pause, nothing written.
        let items = Arc::new(Mutex::new(Vec::new()));
        let mut config = config(Arc::new(MockProvider::new(vec![])));
        config.hydration = SourceRegistry::new().register_sink(GatedSink {
            items: items.clone(),
        });
        let mut store = InMemoryStore::new();
        let outcome = run_ir_steps_with_store_and_replay(
            &config,
            gated_store_machine("gated-store"),
            &mut store,
            None,
            None,
        )
        .await?;
        let IrStepOutcome::AwaitingApproval { pending, .. } = outcome else {
            panic!("expected AwaitingApproval, got {outcome:?}");
        };
        assert_eq!(pending.kind, crate::approval::ApprovalKind::Store);
        assert_eq!(pending.request["sink"], "gated");
        assert_eq!(pending.request["op"], "create");
        assert!(items.lock().unwrap().is_empty(), "no write before approval");

        // Hook approve: the write happens.
        config.approvals.hook = Some(Arc::new(|_: &crate::approval::ApprovalRequest| {
            ApprovalDecision::Approve
        }));
        let (value, _machine) =
            run_ir_sequential(&config, gated_store_machine("gated-store-ok")).await?;
        assert_eq!(value, Value::String("gated-1".into()));
        assert_eq!(items.lock().unwrap().len(), 1);

        // Hook deny: typed denial value, no additional write.
        config.approvals.hook = Some(Arc::new(|_: &crate::approval::ApprovalRequest| {
            ApprovalDecision::Deny
        }));
        let (value, _machine) =
            run_ir_sequential(&config, gated_store_machine("gated-store-no")).await?;
        assert!(is_denial_value(&value), "{value}");
        assert_eq!(items.lock().unwrap().len(), 1);
        Ok(())
    }
}
