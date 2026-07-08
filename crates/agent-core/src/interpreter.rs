use crate::gc::{estimate_tokens, truncate_oversized_message, GcMode, GcState, GcTiming};
use crate::hydration::{PassiveHydrationConfig, PassiveSource, SourceParams, SourceRegistry};
use crate::op::{ChatMessage, Op, OpF, Prompt, Response};
use crate::prompt_ir::{compile_prompt_ir, PromptIR, RetrievalTiming, Section};
use crate::provider::{ChatProvider, ToolFunctionSpec, ToolSpec};
use crate::trace::{preview, Event, TraceLogger};
use anyhow::{anyhow, Result};
use async_recursion::async_recursion;
use chrono::Utc;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Command;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvPolicy {
    /// Inherit the parent environment minus known credential variables
    /// (see [`is_credential_var`]). The agent's own provider key must not be
    /// readable by model-issued shell commands by default.
    Inherit,
    /// Inherit the parent environment unmodified, credentials included.
    /// Explicit opt-in for workflows whose commands need the keys.
    InheritFull,
    Clean {
        vars: BTreeMap<String, String>,
    },
    AllowList {
        names: Vec<String>,
        extra: BTreeMap<String, String>,
    },
}

/// Environment variables stripped from Eval children under
/// [`EnvPolicy::Inherit`]: the variables the agent itself reads for provider
/// auth, plus the `*_API_KEY` convention used by model-registry entries
/// (`api_key: $NAME` in models.yaml). `*_TOKEN` is deliberately NOT stripped:
/// vars like GITHUB_TOKEN are often the agent's working credentials, not the
/// key it runs on.
pub(crate) fn is_credential_var(name: &str) -> bool {
    name == "ANTHROPIC_AUTH_TOKEN" || name.ends_with("_API_KEY")
}

impl EnvPolicy {
    pub(crate) fn label(&self) -> String {
        match self {
            Self::Inherit => "inherit".into(),
            Self::InheritFull => "inherit-full".into(),
            Self::Clean { .. } => "clean".into(),
            Self::AllowList { .. } => "allowlist".into(),
        }
    }

    fn apply(&self, command: &mut Command) {
        self.apply_with_parent_env(command, std::env::vars_os())
    }

    /// Testable core of [`EnvPolicy::apply`]: the parent environment is
    /// injected so tests do not need to mutate process-global env vars.
    fn apply_with_parent_env(
        &self,
        command: &mut Command,
        parent_env: impl Iterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
    ) {
        match self {
            Self::Inherit => {
                command.env_clear();
                for (name, value) in parent_env {
                    let denied = name.to_str().is_some_and(is_credential_var);
                    if !denied {
                        command.env(name, value);
                    }
                }
            }
            Self::InheritFull => {}
            Self::Clean { vars } => {
                command.env_clear();
                command.envs(vars);
            }
            Self::AllowList { names, extra } => {
                command.env_clear();
                for name in names {
                    if let Ok(value) = std::env::var(name) {
                        command.env(name, value);
                    }
                }
                command.envs(extra);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalConfig {
    pub shell: String,
    pub cwd: Option<PathBuf>,
    pub timeout: Duration,
    pub max_stdout_bytes: usize,
    pub max_stderr_bytes: usize,
    pub env: EnvPolicy,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            shell: std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into()),
            cwd: None,
            timeout: Duration::from_secs(120),
            max_stdout_bytes: 1024 * 1024,
            max_stderr_bytes: 1024 * 1024,
            env: EnvPolicy::Inherit,
        }
    }
}

/// The Eval identity recorded for op-layer replay-divergence detection: the
/// display command plus, for direct-exec Evals, the exact argv. Both are
/// compared, so a shell command whose rendering happens to match an argv
/// rendering can never satisfy an argv replay (or vice versa).
#[derive(Clone, Debug, PartialEq, Eq)]
struct RecordedEvalCall {
    command: String,
    argv: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default)]
pub struct ReplayTrace {
    infer_calls: BTreeMap<u64, String>,
    infer_results: BTreeMap<u64, Response>,
    infer_errors: BTreeMap<u64, String>,
    eval_calls: BTreeMap<u64, RecordedEvalCall>,
    eval_results: BTreeMap<u64, Value>,
    eval_errors: BTreeMap<u64, String>,
}

impl ReplayTrace {
    pub fn from_events(events: &[Event]) -> Self {
        let mut replay = Self::default();
        for event in events {
            match event {
                Event::InferCall { op_id, model, .. } => {
                    replay.infer_calls.insert(*op_id, model.clone());
                }
                Event::InferResult {
                    op_id,
                    parent_op_id: None,
                    response: Some(response),
                    ..
                } => {
                    replay.infer_results.insert(*op_id, response.clone());
                }
                Event::InferError { op_id, error, .. } => {
                    replay.infer_errors.insert(*op_id, error.clone());
                }
                Event::EvalCall {
                    op_id,
                    command,
                    argv,
                    ..
                } => {
                    replay.eval_calls.insert(
                        *op_id,
                        RecordedEvalCall {
                            command: command.clone(),
                            argv: argv.clone(),
                        },
                    );
                }
                Event::EvalResult { op_id, result, .. } => {
                    replay.eval_results.insert(*op_id, result.clone());
                }
                Event::EvalError { op_id, error, .. } => {
                    replay.eval_errors.insert(*op_id, error.clone());
                }
                _ => {}
            }
        }
        replay
    }

    pub async fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let events = TraceLogger::read_events(path).await?;
        Ok(Self::from_events(&events))
    }

    pub(crate) fn infer_result(&self, op_id: u64, model: &str) -> Result<Response> {
        if let Some(recorded_model) = self.infer_calls.get(&op_id) {
            if recorded_model != model {
                return Err(anyhow!(
                    "replay diverged at Infer op {op_id}: recorded model '{recorded_model}', requested '{model}'"
                ));
            }
        }
        if let Some(error) = self.infer_errors.get(&op_id) {
            return Err(anyhow!(
                "replaying recorded Infer failure at op {op_id}: {error}"
            ));
        }
        self.infer_results
            .get(&op_id)
            .cloned()
            .ok_or_else(|| anyhow!("replay trace has no InferResult for op {op_id}"))
    }

    pub(crate) fn eval_result(
        &self,
        op_id: u64,
        command: &str,
        argv: Option<&[String]>,
    ) -> Result<Value> {
        if let Some(recorded) = self.eval_calls.get(&op_id) {
            if recorded.command != command || recorded.argv.as_deref() != argv {
                return Err(anyhow!(
                    "replay diverged at Eval op {op_id}: recorded command '{}' (argv {:?}), requested '{command}' (argv {argv:?})",
                    recorded.command,
                    recorded.argv,
                ));
            }
        }
        if let Some(error) = self.eval_errors.get(&op_id) {
            return Err(anyhow!(
                "replaying recorded Eval failure at op {op_id}: {error}"
            ));
        }
        self.eval_results
            .get(&op_id)
            .cloned()
            .ok_or_else(|| anyhow!("replay trace has no EvalResult for op {op_id}"))
    }
}

pub struct SeqConfig {
    pub provider: Arc<dyn ChatProvider>,
    pub hydration: SourceRegistry,
    /// Native tools registered with the runtime (t-1308.7): advertised to
    /// the provider alongside the built-ins and dispatched in-process by
    /// the IR interpreter's Tool effect. Registration is the exposure
    /// switch, mirroring the memory tools.
    pub tools: crate::tool::ToolRegistry,
    pub passive_hydration: PassiveHydrationConfig,
    pub trace: TraceLogger,
    pub eval: EvalConfig,
    pub replay: Option<ReplayTrace>,
    pub trace_full_prompt_ir: bool,
    /// Record full Infer prompts and Get values in trace events. Off by
    /// default: the full prompt repeats the entire conversation on every
    /// call, making traces O(n^2) in session length, and replay only needs
    /// the recorded results. Previews are always recorded.
    pub trace_full_payloads: bool,
    pub gc: GcMode,
    pub gc_threshold: f32,
    pub gc_log: bool,
    pub gc_timing: GcTiming,
    pub context_budget: usize,
    /// Model pricing used to stamp `cost_micro_usd`/`pricing` onto live
    /// InferResults at emission time (t-1334). Keyed by model id string
    /// (registry alias and provider api id). Empty table = usage recorded,
    /// cost omitted. Replayed InferResults never consult this: recorded
    /// costs pass through verbatim.
    pub pricing: crate::cost::PricingTable,
    /// Approval policy for gated effects (t-1308.10, DR-7): pre-loaded
    /// resolutions (resume drivers) and the in-process decision hook (SDK
    /// Runner). Default = neither, which fails closed: a gated effect
    /// pauses the run instead of executing.
    pub approvals: crate::approval::ApprovalConfig,
    /// Runtime-guidance delivery (t-1359, docs/GUIDANCE.md §4): the
    /// capability-keyed operations fragment injected as a PromptIR
    /// Developer/Constraint section on tool-bearing IR Infer calls.
    /// Default-on; opt out (`RuntimeGuidance::disabled()`) for
    /// deterministic prompt-sensitive runs or a hand-written manual.
    pub guidance: crate::guidance::RuntimeGuidance,
}

impl SeqConfig {
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        vec![ToolSpec {
            kind: "function".into(),
            function: ToolFunctionSpec {
                name: "shell".into(),
                description: crate::guidance::SHELL_TOOL_DESCRIPTION.into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "command": { "type": "string" } },
                    "required": ["command"]
                }),
            },
        }]
    }
}

#[async_recursion]
pub async fn run_sequential<S, A>(config: &SeqConfig, state: S, op: Op<S, A>) -> Result<(A, S)>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    A: Send + 'static,
{
    let mut gc_state = GcState::default();
    run_sequential_inner(config, state, op, &mut gc_state, None).await
}

#[async_recursion]
async fn run_sequential_inner<S, A>(
    config: &SeqConfig,
    state: S,
    op: Op<S, A>,
    gc_state: &mut GcState,
    parent_op_id: Option<u64>,
) -> Result<(A, S)>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    A: Send + 'static,
{
    match *op.0 {
        OpF::Pure(value) => Ok((value, state)),
        OpF::Infer {
            model,
            prompt,
            next,
        } => {
            let prompt = hydrate_infer_prompt(config, &state, prompt).await?;
            let mut prompt = maybe_collect_prompt(config, prompt, gc_state).await?;
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::InferCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    model: model.0.clone(),
                    prompt: config.trace_full_payloads.then(|| prompt.clone()),
                    prompt_preview: prompt_preview(&prompt),
                    // Op-layer effects have no IR location; the field stays
                    // absent so op traces serialize unchanged.
                    effect: None,
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            // Catch-overflow retries stay inside this one InferCall: failed
            // attempts surface as gc_collect{trigger:context_overflow} events
            // and the single InferResult/InferError pair reports the outcome,
            // so traces and replay keep their one-call-one-result contract.
            let mut overflow_cycles = 0usize;
            let result = loop {
                let attempt = match &config.replay {
                    Some(replay) => replay.infer_result(op_id, &model.0),
                    None => {
                        config
                            .provider
                            .chat(&model, &config.tool_specs(), &prompt)
                            .await
                    }
                };
                match attempt {
                    Err(err)
                        if config.replay.is_none()
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
                    return Err(err);
                }
            };
            // Live responses get cost stamped from the registry pricing;
            // replayed responses carry their recorded cost untouched so a
            // replay reproduces the original totals even if today's
            // models.yaml prices differ (t-1334).
            if config.replay.is_none() {
                crate::cost::price_response(&mut response, &config.pricing, &model.0);
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
            run_sequential_inner(config, state, next(response), gc_state, parent_op_id).await
        }
        OpF::Eval { request, next } => {
            let command = request.display_command();
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::EvalCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    command: command.clone(),
                    argv: request.argv().map(<[String]>::to_vec),
                    cwd: config
                        .eval
                        .cwd
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    env_policy: config.eval.env.label(),
                    timeout_ms: millis_u64(config.eval.timeout),
                    effect: None,
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            let result = match &config.replay {
                Some(replay) => replay.eval_result(op_id, &command, request.argv()),
                None => {
                    run_eval_request(&config.eval, &request, config.trace.trace_context_env()).await
                }
            };
            let result = match result {
                Ok(result) => result,
                Err(err) => {
                    config
                        .trace
                        .emit(&Event::EvalError {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id,
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
                    parent_op_id,
                    command,
                    result: result.clone(),
                    duration_ms,
                    truncated_stdout,
                    truncated_stderr,
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential_inner(config, state, next(result), gc_state, parent_op_id).await
        }
        OpF::Emit { event, next } => {
            config.trace.emit(&event).await?;
            run_sequential_inner(config, state, next, gc_state, parent_op_id).await
        }
        OpF::Par { ops, next } => {
            let op_id = config.trace.next_op_id();
            let started = Instant::now();
            config
                .trace
                .emit(&Event::ParStart {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    branch_count: ops.len(),
                    timestamp: Utc::now(),
                })
                .await?;
            let branch_count = ops.len();
            let mut values = Vec::with_capacity(branch_count);
            let mut current_state = state;
            for op in ops {
                let (value, new_state) =
                    run_sequential_inner(config, current_state, op, gc_state, Some(op_id)).await?;
                values.push(value);
                current_state = new_state;
            }
            config
                .trace
                .emit(&Event::ParEnd {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    branch_count,
                    duration_ms: millis_u64(started.elapsed()),
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential_inner(config, current_state, next(values), gc_state, parent_op_id).await
        }
    }
}

/// Bounded GC+retry cycles when catch-overflow timing hits a provider
/// context-overflow error. Each cycle halves the target again, so three
/// cycles cover an estimate that is off by up to 8x.
pub(crate) const CATCH_OVERFLOW_MAX_CYCLES: usize = 3;

pub(crate) fn catch_overflow_active(config: &SeqConfig) -> bool {
    config.gc.enabled() && config.gc_timing == GcTiming::CatchOverflow
}

/// Shrink the prompt after the provider rejected it (catch-overflow cycle
/// `cycle`, 1-based): the target is the estimated size halved `cycle` times,
/// because the estimate just proved unreliable in the dangerous direction.
/// Returns the collected prompt and the target it was collected to; the
/// caller records the target as the loop's discovered budget so later turns
/// collect proactively instead of paying a failed request each.
pub(crate) async fn collect_for_overflow(
    config: &SeqConfig,
    prompt: Prompt,
    gc_state: &mut GcState,
    cycle: usize,
) -> Result<(Prompt, usize)> {
    let target_budget = (estimate_tokens(&prompt) >> cycle).max(1);
    let collected = collect_prompt(
        config,
        prompt,
        gc_state,
        target_budget,
        Some(cycle),
        "overflow",
    )
    .await?;
    Ok((collected, target_budget))
}

/// Wrap a turn that still overflowed after GC retry cycles. The message
/// keeps the `context_length_exceeded` prefix so the turn lands in the
/// existing context_overflow taxonomy event, and is non-empty about what
/// was attempted.
pub(crate) fn annotate_overflow_failure(err: anyhow::Error, cycles: usize) -> anyhow::Error {
    if cycles == 0 {
        return err;
    }
    err.context(format!(
        "context_length_exceeded: prompt still overflows the provider context window after {cycles} catch-overflow GC cycle(s)"
    ))
}

pub(crate) async fn maybe_collect_prompt(
    config: &SeqConfig,
    prompt: Prompt,
    gc_state: &mut GcState,
) -> Result<Prompt> {
    if !config.gc.enabled() {
        return Ok(prompt);
    }
    gc_state.infer_calls += 1;
    let before_tokens = estimate_tokens(&prompt);
    let threshold = ((config.context_budget as f32) * config.gc_threshold) as usize;
    let threshold_target = threshold.max(1);
    let (should_collect, target_budget) = match config.gc_timing {
        GcTiming::Threshold => (before_tokens > threshold, threshold_target),
        GcTiming::Eager => (true, threshold_target),
        GcTiming::EveryN(n) => (gc_state.infer_calls.is_multiple_of(n), threshold_target),
        // No estimate-based trigger: the provider is the source of truth.
        // Proactive collection happens only at a ceiling a real overflow
        // already taught us (set by collect_for_overflow).
        GcTiming::CatchOverflow => match gc_state.discovered_budget {
            Some(budget) => (before_tokens > budget, budget.max(1)),
            None => (false, threshold_target),
        },
    };
    if should_collect {
        return collect_prompt(config, prompt, gc_state, target_budget, None, "scheduled").await;
    }
    // Collect-on-overflow backstop (t-1343): no timing policy fired, but the
    // assembled prompt already exceeds the model context budget, so
    // dispatching it would overflow. This is the every:N
    // between-collections gap (and a threshold configured above 1.0) the
    // t-1339 matrix exposed: timing decides when GC runs *proactively*; it
    // never licenses shipping a prompt we already estimate over budget.
    // Collect before dispatch regardless of timing mode; if collection
    // still cannot reach the budget, the existing overflow behavior
    // (catch-overflow retries, provider error) applies unchanged.
    if before_tokens > config.context_budget {
        let backstop_target = threshold_target.min(config.context_budget).max(1);
        return collect_prompt(config, prompt, gc_state, backstop_target, None, "backstop").await;
    }
    Ok(prompt)
}

/// Unconditionally truncate + collect `prompt` to `target_budget`, emitting
/// the gc_truncate/gc_collect events. `overflow_cycle` is set when this
/// collection was triggered reactively by a provider context overflow.
/// `reason` marks why the collection ran on the gc_collect event:
/// `"scheduled"` (the timing policy fired), `"backstop"` (the prompt
/// exceeded the context budget between scheduled collections, t-1343), or
/// `"overflow"` (a provider overflow triggered a catch-overflow cycle).
async fn collect_prompt(
    config: &SeqConfig,
    mut prompt: Prompt,
    gc_state: &mut GcState,
    target_budget: usize,
    overflow_cycle: Option<usize>,
    reason: &'static str,
) -> Result<Prompt> {
    let before_tokens = estimate_tokens(&prompt);
    let truncated_count = truncate_oversized_message(&mut prompt, target_budget);
    if truncated_count > 0 && config.gc_log {
        // Distinct from gc_collect: one or more single messages exceeded the
        // whole budget and were truncated in place (t-1133 overflow taxonomy).
        config
            .trace
            .emit(&Event::Custom {
                run_id: config.trace.run_id().into(),
                name: "gc_truncate".into(),
                data: serde_json::json!({
                    "type": "gc_truncate",
                    "truncated_messages": truncated_count,
                    "budget": target_budget,
                }),
                timestamp: Utc::now(),
            })
            .await?;
    }
    // Semantic embedding pre-pass (t-1350): SemanticGc::collect() consumes
    // only the GcState.embeddings cache, so the async provider call happens
    // here — the layer with async + config access (the same layer the
    // t-1343 backstop landed in, and the shape the t-1166 design note
    // sanctioned). Runs after the truncate pre-pass because the cache keys
    // on message content. Best-effort: a failed embed leaves the cache
    // unchanged and collect() degrades to its deterministic recency
    // heuristic — an embedding outage can never fail a turn. Replay runs
    // are offline by contract and skip the embedder entirely (the same
    // heuristic path as "no embedder configured").
    if let GcMode::Semantic(semantic) = &config.gc {
        if config.replay.is_none() {
            let report = semantic.prime_cache(&prompt, gc_state).await;
            if config.gc_log && (report.embedded > 0 || report.failed) {
                config
                    .trace
                    .emit(&Event::Custom {
                        run_id: config.trace.run_id().into(),
                        name: "gc_semantic_embed".into(),
                        data: serde_json::json!({
                            "type": "gc_semantic_embed",
                            "embedded": report.embedded,
                            "cached": report.cached,
                            "failed": report.failed,
                        }),
                        timestamp: Utc::now(),
                    })
                    .await?;
            }
        }
    }
    // Recall-overlap write-barrier pre-pass (t-1351): a recall result that
    // re-injects content already in (or previously collected from) the
    // window marks that content HOT in GcState.recall_hot. Pure and
    // synchronous, but it lives here rather than inside collect() because
    // "previously collected" needs cross-collection state. Signal-only:
    // no strategy consumes it yet (t-1167 generational input); it is
    // observable on the gc_collect event below.
    let recall_report = crate::gc::record_recall_overlaps(&prompt, gc_state);
    // Snapshot (id, content-key) pairs so the contents of whatever this
    // collection drops can join GcState.collected_hashes afterwards.
    let before_contents: Vec<(uuid::Uuid, String)> = prompt
        .iter()
        .filter_map(|message| {
            message
                .content
                .as_deref()
                .filter(|content| !content.trim().is_empty())
                .map(|content| (message.id, crate::gc::recall_content_key(content)))
        })
        .collect();
    let before_ids: BTreeSet<_> = prompt.iter().map(|message| message.id).collect();
    let collected = config.gc.collect(prompt, target_budget, gc_state);
    let after_ids: BTreeSet<_> = collected.iter().map(|message| message.id).collect();
    let dropped_count = before_ids.difference(&after_ids).count();
    for (id, key) in before_contents {
        if !after_ids.contains(&id) {
            gc_state.collected_hashes.insert(key);
        }
    }
    let after_tokens = estimate_tokens(&collected);
    if config.gc_log {
        let mut data = serde_json::json!({
            "type": "gc_collect",
            "strategy": config.gc.name(),
            "timing": config.gc_timing.name(),
            "reason": reason,
            "target_budget": target_budget,
            "tokens_before": before_tokens,
            "tokens_after": after_tokens,
            "cache_invalidated": gc_state.prefix_invalidated,
            "dropped_count": dropped_count,
            "recall_overlap_events": recall_report.overlap_events,
            "recall_hot": recall_report.hot_total,
        });
        if let Some(cycle) = overflow_cycle {
            let object = data.as_object_mut().expect("gc_collect data is an object");
            object.insert("trigger".into(), "context_overflow".into());
            object.insert("cycle".into(), cycle.into());
        }
        config
            .trace
            .emit(&Event::Custom {
                run_id: config.trace.run_id().into(),
                name: "gc_collect".into(),
                data,
                timestamp: Utc::now(),
            })
            .await?;
    }
    Ok(collected)
}

/// Dispatch an op-layer Eval request to its execution path: shell string via
/// `$SHELL -c`, argv via direct exec. Both share [`run_eval_process`].
pub(crate) async fn run_eval_request(
    config: &EvalConfig,
    request: &crate::op::EvalSpec,
    extra_env: BTreeMap<String, String>,
) -> Result<Value> {
    match request {
        crate::op::EvalSpec::Shell(command) => run_eval_with_env(config, command, extra_env).await,
        crate::op::EvalSpec::Argv(argv) => run_eval_argv_with_env(config, argv, extra_env).await,
    }
}

pub(crate) async fn run_eval_with_env(
    config: &EvalConfig,
    command: &str,
    extra_env: BTreeMap<String, String>,
) -> Result<Value> {
    let mut process = Command::new(&config.shell);
    process.arg("-c").arg(command);
    run_eval_process(config, process, extra_env).await
}

/// Direct exec: `argv[0]` spawned with `argv[1..]` as arguments — no
/// `$SHELL -c`, so nothing re-parses the arguments (SDK DR-3). Env policy,
/// timeout, output caps, and cwd apply exactly as for shell Evals. An empty
/// argv is rejected by `validate_program` (IR) before execution; the check
/// here is a defensive backstop for callers that skip validation.
pub(crate) async fn run_eval_argv_with_env(
    config: &EvalConfig,
    argv: &[String],
    extra_env: BTreeMap<String, String>,
) -> Result<Value> {
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| anyhow!("Eval argv must not be empty"))?;
    let mut process = Command::new(program);
    process.args(args);
    run_eval_process(config, process, extra_env).await
}

/// Shared tail of every Eval execution: cwd, env policy (credential
/// stripping under Inherit), trace-context env, detached stdin, timeout, and
/// capped stdout/stderr decoding.
async fn run_eval_process(
    config: &EvalConfig,
    mut process: Command,
    extra_env: BTreeMap<String, String>,
) -> Result<Value> {
    let started = Instant::now();
    if let Some(cwd) = &config.cwd {
        process.current_dir(cwd);
    }
    config.env.apply(&mut process);
    process.envs(extra_env);
    // Detach the child's stdin so interactive commands (e.g. `git rebase -i`,
    // `git commit` with no -m, `ssh`, prompts) get immediate EOF instead of
    // consuming the agent's own control channel (NUL-framed session/fifo input).
    process.stdin(Stdio::null());
    process.kill_on_drop(true);

    let output = tokio::time::timeout(config.timeout, process.output()).await;
    let duration_ms = millis_u64(started.elapsed());

    match output {
        Ok(output) => {
            let output = output?;
            let (stdout, stdout_truncated) = decode_capped(&output.stdout, config.max_stdout_bytes);
            let (stderr, stderr_truncated) = decode_capped(&output.stderr, config.max_stderr_bytes);
            let status = output.status.code();
            Ok(serde_json::json!({
                "ok": status == Some(0),
                "status": status,
                "timed_out": false,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "duration_ms": duration_ms,
            }))
        }
        Err(_) => Ok(serde_json::json!({
            "ok": false,
            "status": null,
            "timed_out": true,
            "stdout": "",
            "stderr": "",
            "stdout_truncated": false,
            "stderr_truncated": false,
            "duration_ms": duration_ms,
        })),
    }
}

pub(crate) fn millis_u64(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn decode_capped(bytes: &[u8], max_bytes: usize) -> (String, bool) {
    let truncated = bytes.len() > max_bytes;
    let take = bytes.len().min(max_bytes);
    (
        String::from_utf8_lossy(&bytes[..take]).into_owned(),
        truncated,
    )
}

pub(crate) fn prompt_preview(prompt: &Prompt) -> String {
    let rendered = prompt
        .iter()
        .map(|message| {
            format!(
                "{}: {}",
                message.role,
                message.content.as_deref().unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    preview(&rendered, 1024)
}

pub(crate) fn response_preview(response: &Response) -> String {
    preview(&response.content, 1024)
}

fn stable_section_id(prefix: &str, content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    format!("{}-{}", prefix, &hash[..12])
}

pub(crate) async fn hydrate_infer_prompt<S>(
    config: &SeqConfig,
    state: &S,
    mut prompt: Prompt,
) -> Result<Prompt>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    if config.passive_hydration.is_empty() {
        return Ok(prompt);
    }

    let op_id = config.trace.next_op_id();
    let sources = config
        .passive_hydration
        .sources
        .iter()
        .map(|source| format!("{source:?}"))
        .collect::<Vec<_>>();
    config
        .trace
        .emit(&Event::HydrationStart {
            run_id: config.trace.run_id().into(),
            op_id,
            parent_op_id: None,
            sources,
            max_bytes: config.passive_hydration.max_bytes,
            timestamp: Utc::now(),
        })
        .await?;

    let mut prompt_ir_sections = Vec::new();
    let mut section_count = 0;
    let mut total_bytes = 0;
    for source in &config.passive_hydration.sources {
        match source {
            PassiveSource::TemporalHistory => {
                let value = serde_json::to_value(state)?;
                if let Value::Array(messages) = value {
                    for value in messages {
                        if let Ok(message) = serde_json::from_value::<ChatMessage>(value) {
                            if !prompt.contains(&message) {
                                prompt.push(message);
                            }
                        }
                    }
                } else if value != Value::Null {
                    let content = value.to_string();
                    total_bytes += content.len();
                    section_count += 1;
                    config
                        .trace
                        .emit(&Event::HydrationSection {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id: None,
                            source: "temporal-history".into(),
                            kind: "Temporal".into(),
                            bytes: content.len(),
                            content_preview: preview(&content, 512),
                            metadata: serde_json::json!({}),
                            timestamp: Utc::now(),
                        })
                        .await?;
                    let id = stable_section_id("passive-temporal-history", &content);
                    prompt_ir_sections.push(Section::passive_temporal(
                        id,
                        "temporal history",
                        content,
                    ));
                }
            }
            PassiveSource::SessionContext => {
                let params = SourceParams {
                    query: None,
                    max_bytes: config.passive_hydration.max_bytes,
                };
                for result in config.hydration.retrieve_session_context(params).await? {
                    total_bytes += result.content.len();
                    section_count += 1;
                    config
                        .trace
                        .emit(&Event::HydrationSection {
                            run_id: config.trace.run_id().into(),
                            op_id,
                            parent_op_id: None,
                            source: result.source.clone(),
                            kind: format!("{:?}", result.kind),
                            bytes: result.content.len(),
                            content_preview: preview(&result.content, 512),
                            metadata: result.metadata.clone(),
                            timestamp: Utc::now(),
                        })
                        .await?;
                    let id = stable_section_id(
                        &format!("passive-source-{}-{:?}", result.source, result.kind),
                        &result.content,
                    );
                    prompt_ir_sections.push(Section::from_source_result(
                        id,
                        result,
                        RetrievalTiming::Passive,
                        None,
                    ));
                }
            }
        }
    }

    if !prompt_ir_sections.is_empty() {
        let prompt_ir = PromptIR::new(prompt, prompt_ir_sections)?;
        config
            .trace
            .emit(&Event::Custom {
                run_id: config.trace.run_id().into(),
                name: "prompt_ir".into(),
                data: serde_json::to_value(prompt_ir.trace(config.trace_full_prompt_ir))?,
                timestamp: Utc::now(),
            })
            .await?;
        prompt = compile_prompt_ir(&prompt_ir);
    }

    config
        .trace
        .emit(&Event::HydrationEnd {
            run_id: config.trace.run_id().into(),
            op_id,
            parent_op_id: None,
            section_count,
            total_bytes,
            timestamp: Utc::now(),
        })
        .await?;

    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::{HydrationSource, SourceCapability, SourceKind, SourceResult};
    use crate::op::{agent_loop, infer, Model, Response, ToolCall};
    use crate::provider::ToolSpec;
    use async_trait::async_trait;
    use serde_json::{json, Value};
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

        fn prompt_count(&self) -> usize {
            self.prompts.lock().unwrap().len()
        }

        fn prompts(&self) -> Vec<Prompt> {
            self.prompts.lock().unwrap().clone()
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

    /// Fails the first `failures` chat calls with `error_message` (a raw
    /// provider string, the way the codex OAuth path surfaces overflows),
    /// then serves queued responses.
    struct OverflowProvider {
        failures: Mutex<usize>,
        error_message: String,
        responses: Mutex<Vec<Response>>,
        prompts: Mutex<Vec<Prompt>>,
    }

    impl OverflowProvider {
        fn new(failures: usize, error_message: &str, mut responses: Vec<Response>) -> Self {
            responses.reverse();
            Self {
                failures: Mutex::new(failures),
                error_message: error_message.into(),
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
                return Err(anyhow!("{}", self.error_message));
            }
            self.responses
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow!("mock provider exhausted"))
        }
    }

    /// Raw codex backend phrasing: never classified by the old
    /// `context_length_exceeded` prefix check (the smith failure mode).
    const CODEX_OVERFLOW: &str = "Codex OAuth provider returned 400 Bad Request: \
        {\"detail\":\"Your input exceeds the context window of this model. \
        Please adjust your input and try again.\"}";

    fn response(content: &str, tool_calls: Vec<ToolCall>) -> Response {
        Response {
            content: content.into(),
            tool_calls,
            finish_reason: Some(crate::op::FinishReason::Stop),
            input_tokens: 3,
            output_tokens: 4,
            total_tokens: 7,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            metadata: Default::default(),
        }
    }

    fn tool_call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall::new(id, name, arguments)
    }

    fn test_trace() -> TraceLogger {
        let path = std::env::temp_dir().join(format!("agent-core-test-{}.jsonl", Uuid::new_v4()));
        TraceLogger::new(Uuid::new_v4().to_string(), path)
    }

    #[tokio::test]
    async fn gc_collects_to_threshold_budget() -> Result<()> {
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: crate::gc::GcMode::Ring(crate::gc::RingGc::default()),
            gc_threshold: 0.5,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 100,
            pricing: Default::default(),
        };
        // The last user message is hard-protected (t-1367), so the
        // droppable ballast is the OLDER user message; system + last user
        // must fit the threshold budget for convergence to be possible.
        let prompt = vec![
            ChatMessage::system("system"),
            ChatMessage::user("x".repeat(90)),
            ChatMessage::user("y".repeat(60)),
        ];
        assert!(estimate_tokens(&prompt) > 50);
        assert!(estimate_tokens(&prompt) <= 100);

        let mut state = crate::gc::GcState::default();
        let collected = maybe_collect_prompt(&config, prompt, &mut state).await?;

        assert!(estimate_tokens(&collected) <= 50, "collected={collected:?}");
        assert!(collected.iter().any(|message| message.role == "system"));
        Ok(())
    }

    #[tokio::test]
    async fn gc_truncation_emits_distinct_gc_truncate_event() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: crate::gc::GcMode::Ring(crate::gc::RingGc::default()),
            gc_threshold: 0.5,
            gc_log: true,
            gc_timing: GcTiming::Threshold,
            context_budget: 100,
            pricing: Default::default(),
        };
        // One message alone larger than the whole budget: only the truncate
        // pre-pass can shrink it, and that must be visible as its own event.
        let prompt = vec![
            ChatMessage::system("system"),
            ChatMessage::user("x".repeat(2000)),
        ];

        let mut state = crate::gc::GcState::default();
        let collected = maybe_collect_prompt(&config, prompt, &mut state).await?;
        assert!(estimate_tokens(&collected) <= 50);

        let events = TraceLogger::read_events(trace_path).await?;
        assert!(
            events.iter().any(|event| matches!(
                event,
                Event::Custom { name, .. } if name == "gc_truncate"
            )),
            "single-oversized-message truncation must emit gc_truncate: {events:?}"
        );
        assert!(
            events.iter().any(|event| matches!(
                event,
                Event::Custom { name, .. } if name == "gc_collect"
            )),
            "the collection event must still fire"
        );
        Ok(())
    }

    /// Deterministic test embedder for the semantic pre-pass; flippable to
    /// failure to exercise the degrade path.
    struct StubEmbedder {
        fail: bool,
    }

    #[async_trait::async_trait]
    impl crate::embedding::Embedder for StubEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if self.fail {
                anyhow::bail!("embeddings endpoint down");
            }
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }

        fn model_id(&self) -> &str {
            "stub-embedder"
        }
    }

    fn semantic_config(trace: TraceLogger, fail: bool) -> SeqConfig {
        SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: crate::gc::GcMode::Semantic(crate::gc::SemanticGc {
                preserve_prefix: false,
                recent_window: 2,
                embedder: Some(Arc::new(StubEmbedder { fail })),
                ..Default::default()
            }),
            gc_threshold: 0.5,
            gc_log: true,
            gc_timing: GcTiming::Threshold,
            context_budget: 200,
            pricing: Default::default(),
        }
    }

    fn semantic_test_prompt() -> Prompt {
        vec![
            ChatMessage::system("system"),
            ChatMessage::user("x".repeat(90)),
            ChatMessage::user("y".repeat(90)),
            ChatMessage::user("z".repeat(90)),
        ]
    }

    /// The semantic pre-pass primes GcState.embeddings before collect()
    /// runs, emits the gc_semantic_embed event under --gc-log, and the
    /// collection converges (t-1350).
    #[tokio::test]
    async fn semantic_gc_pre_pass_primes_cache_and_collects() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = semantic_config(trace, false);
        let prompt = semantic_test_prompt();

        let mut state = crate::gc::GcState::default();
        let collected = maybe_collect_prompt(&config, prompt, &mut state).await?;

        assert!(
            estimate_tokens(&collected) <= 100,
            "collected={collected:?}"
        );
        assert!(
            !state.embeddings.is_empty(),
            "the pre-pass must populate the cache before collect() consumes it"
        );
        let events = TraceLogger::read_events(trace_path).await?;
        let embed_event = events
            .iter()
            .find_map(|event| match event {
                Event::Custom { name, data, .. } if name == "gc_semantic_embed" => Some(data),
                _ => None,
            })
            .expect("gc_semantic_embed event under --gc-log");
        assert_eq!(embed_event["failed"], false);
        assert!(embed_event["embedded"].as_u64().unwrap_or(0) > 0);
        Ok(())
    }

    /// Embedding failure is a degrade, never an error: the turn proceeds,
    /// collect() uses the recency heuristic, and the failure is visible on
    /// the gc_semantic_embed event.
    #[tokio::test]
    async fn semantic_gc_pre_pass_failure_degrades_to_heuristic() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = semantic_config(trace, true);
        let prompt = semantic_test_prompt();

        let mut state = crate::gc::GcState::default();
        let collected = maybe_collect_prompt(&config, prompt, &mut state).await?;

        assert!(
            estimate_tokens(&collected) <= 100,
            "collected={collected:?}"
        );
        assert!(state.embeddings.is_empty());
        let events = TraceLogger::read_events(trace_path).await?;
        let embed_event = events
            .iter()
            .find_map(|event| match event {
                Event::Custom { name, data, .. } if name == "gc_semantic_embed" => Some(data),
                _ => None,
            })
            .expect("the failure must be observable under --gc-log");
        assert_eq!(embed_event["failed"], true);
        Ok(())
    }

    /// The recall-overlap write-barrier (t-1351) is recorded by the pre-pass
    /// and observable on the gc_collect event, so the behavioral eval
    /// (t-1349) can watch it fire. Ring is the strategy here on purpose:
    /// the signal is strategy-independent and consumed by none yet.
    #[tokio::test]
    async fn gc_collect_reports_recall_overlap_signal() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: crate::gc::GcMode::Ring(crate::gc::RingGc::default()),
            gc_threshold: 0.5,
            gc_log: true,
            gc_timing: GcTiming::Threshold,
            context_budget: 300,
            pricing: Default::default(),
        };
        let note = "the planner fix is raising the statistics target";
        let prompt = vec![
            ChatMessage::system("system"),
            ChatMessage::user(note),
            ChatMessage::assistant(
                None,
                vec![tool_call(
                    "call-recall",
                    "recall",
                    serde_json::json!({ "query": "planner" }),
                )],
            ),
            ChatMessage::tool(
                "call-recall",
                serde_json::json!([{ "source": "memory", "kind": "Semantic", "content": note }])
                    .to_string(),
            ),
            ChatMessage::user("x".repeat(600)),
        ];

        let mut state = crate::gc::GcState::default();
        let _ = maybe_collect_prompt(&config, prompt, &mut state).await?;

        assert!(
            state
                .recall_hot
                .contains(&crate::gc::recall_content_key(note)),
            "the pre-pass must mark the re-injected content hot"
        );
        let events = TraceLogger::read_events(trace_path).await?;
        let collect_event = events
            .iter()
            .find_map(|event| match event {
                Event::Custom { name, data, .. } if name == "gc_collect" => Some(data),
                _ => None,
            })
            .expect("gc_collect event under --gc-log");
        assert_eq!(collect_event["recall_overlap_events"], 1);
        assert_eq!(collect_event["recall_hot"], 1);
        Ok(())
    }

    /// Dropped message contents join GcState.collected_hashes so a later
    /// recall re-injecting them still registers as a write-barrier event.
    #[tokio::test]
    async fn gc_collect_records_dropped_contents_for_the_write_barrier() -> Result<()> {
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: crate::gc::GcMode::Ring(crate::gc::RingGc {
                preserve_prefix: false,
            }),
            gc_threshold: 0.5,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 100,
            pricing: Default::default(),
        };
        let doomed = "z".repeat(90);
        let prompt = vec![
            ChatMessage::system("system"),
            ChatMessage::user(doomed.clone()),
            ChatMessage::user("y".repeat(90)),
        ];

        let mut state = crate::gc::GcState::default();
        let collected = maybe_collect_prompt(&config, prompt, &mut state).await?;

        assert!(collected
            .iter()
            .all(|message| message.content.as_deref() != Some(doomed.as_str())));
        assert!(
            state
                .collected_hashes
                .contains(&crate::gc::recall_content_key(&doomed)),
            "dropped content must be remembered for later recall overlap"
        );
        Ok(())
    }

    fn timing_config(
        provider: Arc<dyn ChatProvider>,
        trace: TraceLogger,
        timing: GcTiming,
    ) -> SeqConfig {
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
            gc: crate::gc::GcMode::Ring(crate::gc::RingGc::default()),
            gc_threshold: 0.85,
            gc_log: true,
            gc_timing: timing,
            context_budget: 200_000,
            pricing: Default::default(),
        }
    }

    async fn gc_collect_events(trace_path: PathBuf) -> Result<Vec<Value>> {
        if !trace_path.exists() {
            // The trace file is created lazily on first emit.
            return Ok(Vec::new());
        }
        let events = TraceLogger::read_events(trace_path).await?;
        Ok(events
            .iter()
            .filter_map(|event| match event {
                Event::Custom { name, data, .. } if name == "gc_collect" => Some(data.clone()),
                _ => None,
            })
            .collect())
    }

    #[tokio::test]
    async fn eager_timing_collects_on_every_infer_call() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = timing_config(Arc::new(MockProvider::new(vec![])), trace, GcTiming::Eager);
        // Far below the threshold trigger: only eager timing collects here.
        let prompt = vec![ChatMessage::system("system"), ChatMessage::user("hi")];

        let mut state = crate::gc::GcState::default();
        for _ in 0..3 {
            maybe_collect_prompt(&config, prompt.clone(), &mut state).await?;
        }

        let collects = gc_collect_events(trace_path).await?;
        assert_eq!(collects.len(), 3, "eager collects every call: {collects:?}");
        assert_eq!(collects[0]["timing"], "eager");
        Ok(())
    }

    #[tokio::test]
    async fn every_n_timing_collects_on_schedule() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = timing_config(
            Arc::new(MockProvider::new(vec![])),
            trace,
            GcTiming::EveryN(2),
        );
        let prompt = vec![ChatMessage::system("system"), ChatMessage::user("hi")];

        let mut state = crate::gc::GcState::default();
        for _ in 0..4 {
            maybe_collect_prompt(&config, prompt.clone(), &mut state).await?;
        }

        let collects = gc_collect_events(trace_path).await?;
        assert_eq!(
            collects.len(),
            2,
            "every:2 collects on calls 2 and 4: {collects:?}"
        );
        assert_eq!(collects[0]["timing"], "every-n");
        assert_eq!(collects[0]["reason"], "scheduled");
        Ok(())
    }

    /// t-1343: `every:N` used to dispatch over-budget prompts between its
    /// scheduled collections (7/15 ring and stack cells in the t-1339
    /// matrix). The backstop collects at any infer point whose prompt
    /// exceeds the full context budget, regardless of the schedule.
    #[tokio::test]
    async fn every_n_backstop_collects_before_over_budget_dispatch() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = timing_config(
            Arc::new(MockProvider::new(vec![])),
            trace,
            GcTiming::EveryN(4),
        );
        config.context_budget = 100;
        // First infer call: 1 % 4 != 0, so the schedule alone would ship
        // this prompt over budget.
        let mut prompt = vec![ChatMessage::system("system")];
        prompt.extend((0..4).map(|i| ChatMessage::user(format!("{i}-{}", "x".repeat(90)))));
        assert!(estimate_tokens(&prompt) > config.context_budget);

        let mut state = crate::gc::GcState::default();
        let collected = maybe_collect_prompt(&config, prompt, &mut state).await?;

        assert!(
            estimate_tokens(&collected) <= config.context_budget,
            "no over-budget dispatch: {} > {}",
            estimate_tokens(&collected),
            config.context_budget
        );
        let collects = gc_collect_events(trace_path).await?;
        assert_eq!(collects.len(), 1, "{collects:?}");
        assert_eq!(collects[0]["reason"], "backstop");
        assert_eq!(collects[0]["timing"], "every-n");
        Ok(())
    }

    /// t-1343: the threshold path has the same exposure when the configured
    /// threshold sits above the full budget (estimate between the two would
    /// dispatch over budget without ever crossing the trigger).
    #[tokio::test]
    async fn threshold_backstop_covers_thresholds_above_budget() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = timing_config(
            Arc::new(MockProvider::new(vec![])),
            trace,
            GcTiming::Threshold,
        );
        config.context_budget = 100;
        config.gc_threshold = 1.5; // trigger at 150 — above the budget
        let mut prompt = vec![ChatMessage::system("system")];
        prompt.extend((0..3).map(|i| ChatMessage::user(format!("{i}-{}", "x".repeat(90)))));
        let before = estimate_tokens(&prompt);
        assert!(
            before > 100 && before < 150,
            "test setup: estimate {before} must sit between budget and threshold"
        );

        let mut state = crate::gc::GcState::default();
        let collected = maybe_collect_prompt(&config, prompt, &mut state).await?;

        assert!(estimate_tokens(&collected) <= config.context_budget);
        let collects = gc_collect_events(trace_path).await?;
        assert_eq!(collects.len(), 1, "{collects:?}");
        assert_eq!(collects[0]["reason"], "backstop");
        assert_eq!(collects[0]["timing"], "threshold");
        // The backstop target never exceeds the budget even though the
        // threshold target does.
        assert_eq!(collects[0]["target_budget"], 100);
        Ok(())
    }

    /// t-1343: a backstop collection is an ordinary collection — under
    /// `--gc-cache preserve` (the strategy default here) it must keep the
    /// pinned prefix byte-stable, exactly like a scheduled one.
    #[tokio::test]
    async fn backstop_respects_preserve_cache_semantics() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = timing_config(
            Arc::new(MockProvider::new(vec![])),
            trace,
            GcTiming::EveryN(4),
        );
        config.context_budget = 100;
        let mut prompt = vec![ChatMessage::system("system")];
        prompt.extend((0..4).map(|i| ChatMessage::user(format!("{i}-{}", "x".repeat(90)))));
        let system_id = prompt[0].id;
        assert!(estimate_tokens(&prompt) > config.context_budget);

        let mut state = crate::gc::GcState::default();
        let collected = maybe_collect_prompt(&config, prompt, &mut state).await?;

        assert!(
            !state.prefix_invalidated,
            "preserve-mode backstop must not invalidate the cached prefix"
        );
        assert_eq!(
            collected.first().map(|message| message.id),
            Some(system_id),
            "the pinned prefix must survive the backstop untouched"
        );
        let collects = gc_collect_events(trace_path).await?;
        assert_eq!(collects.len(), 1, "{collects:?}");
        assert_eq!(collects[0]["reason"], "backstop");
        assert_eq!(collects[0]["cache_invalidated"], false);
        Ok(())
    }

    #[tokio::test]
    async fn catch_overflow_timing_has_no_estimate_trigger() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut config = timing_config(
            Arc::new(MockProvider::new(vec![])),
            trace,
            GcTiming::CatchOverflow,
        );
        // Estimate is over the threshold trigger; threshold timing would
        // collect here, catch-overflow must not (the provider is the truth).
        config.context_budget = 100;
        config.gc_threshold = 0.5;
        let prompt = vec![
            ChatMessage::system("system"),
            ChatMessage::user("x".repeat(90)),
            ChatMessage::user("y".repeat(90)),
        ];
        assert!(estimate_tokens(&prompt) > 50);

        let mut state = crate::gc::GcState::default();
        let kept = maybe_collect_prompt(&config, prompt.clone(), &mut state).await?;

        assert_eq!(kept.len(), prompt.len());
        assert!(gc_collect_events(trace_path).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn catch_overflow_collects_and_retries_the_same_turn() -> Result<()> {
        let provider = Arc::new(OverflowProvider::new(
            1,
            CODEX_OVERFLOW,
            vec![response("recovered", vec![])],
        ));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = timing_config(provider.clone(), trace, GcTiming::CatchOverflow);
        let mut prompt = vec![ChatMessage::system("system")];
        prompt.extend((0..6).map(|i| ChatMessage::user(format!("{i}-{}", "x".repeat(200)))));

        let (result, _state) = run_sequential(
            &config,
            prompt.clone(),
            infer::<Prompt>(Model("mock".into()), prompt),
        )
        .await?;

        assert_eq!(result.content, "recovered");
        let prompts = provider.prompts();
        assert_eq!(prompts.len(), 2, "one overflow, one retry");
        assert!(
            prompts[1].len() < prompts[0].len(),
            "the retry prompt must have been collected: {} -> {}",
            prompts[0].len(),
            prompts[1].len()
        );
        let events = TraceLogger::read_events(trace_path).await?;
        let infer_calls = events
            .iter()
            .filter(|event| matches!(event, Event::InferCall { .. }))
            .count();
        let infer_results = events
            .iter()
            .filter(|event| matches!(event, Event::InferResult { .. }))
            .count();
        assert_eq!(
            (infer_calls, infer_results),
            (1, 1),
            "retries stay inside one InferCall/InferResult pair"
        );
        let collects: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Custom { name, data, .. } if name == "gc_collect" => Some(data),
                _ => None,
            })
            .collect();
        assert_eq!(collects.len(), 1);
        assert_eq!(collects[0]["trigger"], "context_overflow");
        assert_eq!(collects[0]["timing"], "catch-overflow");
        assert_eq!(collects[0]["cycle"], 1);
        Ok(())
    }

    #[tokio::test]
    async fn catch_overflow_gives_up_cleanly_after_bounded_cycles() -> Result<()> {
        let provider = Arc::new(OverflowProvider::new(usize::MAX, CODEX_OVERFLOW, vec![]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = timing_config(provider.clone(), trace, GcTiming::CatchOverflow);
        let mut prompt = vec![ChatMessage::system("system")];
        prompt.extend((0..6).map(|i| ChatMessage::user(format!("{i}-{}", "x".repeat(200)))));

        let err = run_sequential(
            &config,
            prompt.clone(),
            infer::<Prompt>(Model("mock".into()), prompt),
        )
        .await
        .expect_err("provider never stops overflowing");

        // The terminal message is non-empty about what was attempted and
        // keeps the prefix the context_overflow taxonomy keys on.
        let message = err.to_string();
        assert!(
            message.starts_with("context_length_exceeded"),
            "taxonomy prefix preserved: {message}"
        );
        assert!(
            message.contains("3 catch-overflow GC cycle(s)"),
            "{message}"
        );
        assert_eq!(provider.prompts().len(), 1 + CATCH_OVERFLOW_MAX_CYCLES);
        let events = TraceLogger::read_events(trace_path).await?;
        let cycles: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Custom { name, data, .. } if name == "gc_collect" => {
                    data.get("cycle").cloned()
                }
                _ => None,
            })
            .collect();
        assert_eq!(cycles, vec![json!(1), json!(2), json!(3)]);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::InferError { .. }))
                .count(),
            1,
            "one InferError for the instruction, not one per attempt"
        );
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_executes_tool_and_feeds_result_back() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![tool_call(
                    "call-1",
                    "shell",
                    json!({ "command": "printf hello" }),
                )],
            ),
            response("done", vec![]),
        ]));
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
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
        };
        let prompt = vec![ChatMessage::user("use echo")];

        let (result, _state) = run_sequential(
            &config,
            prompt.clone(),
            agent_loop(Model("mock".into()), prompt, 4),
        )
        .await?;

        assert_eq!(result.content, "done");
        assert_eq!(provider.prompt_count(), 2);
        // History threads through the continuation (t-1182): the second
        // inference sees the tool result fed back.
        let prompts = provider.prompts();
        assert!(prompts[1].iter().any(|msg| msg.role == "tool"));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_supports_multiple_tool_turns() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![tool_call(
                    "call-1",
                    "shell",
                    json!({ "command": "printf 1" }),
                )],
            ),
            response(
                "",
                vec![tool_call(
                    "call-2",
                    "shell",
                    json!({ "command": "printf 2" }),
                )],
            ),
            response("finished", vec![]),
        ]));
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
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
        };
        let prompt = vec![ChatMessage::user("do two steps")];

        let (result, _state) = run_sequential(
            &config,
            prompt.clone(),
            agent_loop(Model("mock".into()), prompt, 4),
        )
        .await?;

        assert_eq!(result.content, "finished");
        assert_eq!(provider.prompt_count(), 3);
        // By the final inference both tool results have been fed back.
        let prompts = provider.prompts();
        assert_eq!(
            prompts[2].iter().filter(|msg| msg.role == "tool").count(),
            2
        );
        Ok(())
    }

    #[tokio::test]
    async fn infer_can_call_infer_from_its_continuation() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response("first-infer-result", vec![]),
            response("second-infer-result", vec![]),
        ]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: provider.clone(),
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
        };

        let program = infer(Model("mock".into()), vec![ChatMessage::user("first")]).and_then(
            |first_response| {
                infer(
                    Model("mock".into()),
                    vec![ChatMessage::user(format!(
                        "second saw: {}",
                        first_response.content
                    ))],
                )
            },
        );

        let (result, _) = run_sequential(&config, (), program).await?;

        assert_eq!(result.content, "second-infer-result");
        let prompts = provider.prompts();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0][0].content.as_deref(), Some("first"));
        assert_eq!(
            prompts[1][0].content.as_deref(),
            Some("second saw: first-infer-result")
        );
        let events = TraceLogger::read_events(trace_path).await?;
        let infer_calls = events
            .iter()
            .filter(|event| matches!(event, Event::InferCall { .. }))
            .count();
        let infer_results = events
            .iter()
            .filter(|event| matches!(event, Event::InferResult { .. }))
            .count();
        assert_eq!(infer_calls, 2);
        assert_eq!(infer_results, 2);
        Ok(())
    }

    struct StaticSource {
        name: &'static str,
        kind: SourceKind,
        capabilities: SourceCapability,
        content: &'static str,
        queries: Arc<Mutex<Vec<Option<String>>>>,
    }

    #[async_trait]
    impl HydrationSource for StaticSource {
        fn name(&self) -> &str {
            self.name
        }

        fn kind(&self) -> SourceKind {
            self.kind
        }

        fn capabilities(&self) -> SourceCapability {
            self.capabilities
        }

        async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
            self.queries.lock().unwrap().push(params.query.clone());
            Ok(SourceResult {
                source: self.name.into(),
                kind: self.kind,
                content: self.content.into(),
                metadata: json!({}),
            })
        }
    }

    #[tokio::test]
    async fn infer_injects_configured_passive_context_without_agent_get() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("ok", vec![])]));
        let queries = Arc::new(Mutex::new(Vec::new()));
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: provider.clone(),
            hydration: SourceRegistry::new().register(StaticSource {
                name: "workspace",
                kind: SourceKind::Knowledge,
                capabilities: SourceCapability::SESSION_CONTEXT,
                content: "passive workspace facts",
                queries: queries.clone(),
            }),
            passive_hydration: PassiveHydrationConfig::with_sources([
                PassiveSource::SessionContext,
            ]),
            trace: test_trace(),
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
        };
        let prompt = vec![ChatMessage::user("answer")];

        let (result, _) =
            run_sequential(&config, prompt, infer(Model("mock".into()), vec![])).await?;

        assert_eq!(result.content, "ok");
        let prompts = provider.prompts();
        assert!(prompts[0].iter().any(|message| {
            message.role == "system"
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("passive workspace facts"))
        }));
        assert_eq!(*queries.lock().unwrap(), vec![None]);
        Ok(())
    }

    #[tokio::test]
    async fn eval_timeout_output_cap_cwd_and_clean_env_are_enforced() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let cwd = std::env::temp_dir().join(format!("agent-core-eval-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&cwd).await?;
        let mut clean_vars = std::collections::BTreeMap::new();
        if let Ok(path) = std::env::var("PATH") {
            clean_vars.insert("PATH".into(), path);
        }
        let mut config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
            eval: EvalConfig {
                shell: "/bin/sh".into(),
                cwd: Some(cwd.clone()),
                // Short timeout only for the timed_out assertion below; the
                // remaining evals get a generous timeout, because under
                // parallel-test load even spawning `sh -c printf` can take
                // longer than 50ms and the child dies with empty output.
                timeout: std::time::Duration::from_millis(50),
                max_stdout_bytes: 4,
                max_stderr_bytes: 4,
                env: EnvPolicy::Clean { vars: clean_vars },
            },
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 200_000,
            pricing: Default::default(),
        };

        let (timeout_result, _) = run_sequential(&config, (), crate::op::eval("sleep 1")).await?;
        assert_eq!(timeout_result["timed_out"], json!(true));

        config.eval.timeout = std::time::Duration::from_secs(10);
        let (cap_result, _) = run_sequential(&config, (), crate::op::eval("printf 123456")).await?;
        assert_eq!(cap_result["stdout"], json!("1234"));
        assert_eq!(cap_result["stdout_truncated"], json!(true));

        let _ = run_sequential(&config, (), crate::op::eval("printf cwd > marker")).await?;
        assert_eq!(tokio::fs::read_to_string(cwd.join("marker")).await?, "cwd");

        // Clean policy must not leak inherited vars. Probe with a var that is
        // already present in the parent environment instead of set_var, which
        // races under parallel test execution. `${VAR+leaked}` expands to
        // "leaked" only if VAR is set in the *child* environment.
        let is_shell_identifier = |key: &str| {
            !key.is_empty()
                && key
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        };
        if let Some((inherited, _)) =
            std::env::vars().find(|(key, _)| key != "PATH" && is_shell_identifier(key))
        {
            let (env_result, _) = run_sequential(
                &config,
                (),
                crate::op::eval(format!(r#"printf %s "${{{inherited}+leaked}}""#)),
            )
            .await?;
            assert_eq!(env_result["stdout"], json!(""), "leaked: {inherited}");
        }
        Ok(())
    }

    #[test]
    fn credential_var_classification() {
        // The agent's own provider auth must be stripped from Eval children…
        assert!(is_credential_var("AGENT_API_KEY"));
        assert!(is_credential_var("ANTHROPIC_API_KEY"));
        assert!(is_credential_var("OPENROUTER_API_KEY"));
        assert!(is_credential_var("PARASAIL_API_KEY"));
        assert!(is_credential_var("ANTHROPIC_AUTH_TOKEN"));
        // …but working credentials and ordinary vars must not be.
        assert!(!is_credential_var("GITHUB_TOKEN"));
        assert!(!is_credential_var("PATH"));
        assert!(!is_credential_var("HOME"));
        assert!(!is_credential_var("API_KEYS_DIR"));
    }

    #[tokio::test]
    async fn inherit_policy_strips_credential_vars_but_keeps_the_rest() -> Result<()> {
        use std::ffi::OsString;

        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(
            r#"printf %s "${ANTHROPIC_API_KEY+key-leaked}${ANTHROPIC_AUTH_TOKEN+token-leaked}${SAFE_VAR-safe-missing}""#,
        );
        command.stdin(Stdio::null());
        // Inject a fake parent environment instead of mutating process env.
        let parent = vec![
            (
                OsString::from("PATH"),
                std::env::var_os("PATH").unwrap_or_default(),
            ),
            (
                OsString::from("ANTHROPIC_API_KEY"),
                OsString::from("sk-fake"),
            ),
            (
                OsString::from("ANTHROPIC_AUTH_TOKEN"),
                OsString::from("oauth-fake"),
            ),
            (OsString::from("SAFE_VAR"), OsString::from("visible")),
        ];

        EnvPolicy::Inherit.apply_with_parent_env(&mut command, parent.into_iter());
        let output = command.output().await?;

        assert_eq!(String::from_utf8_lossy(&output.stdout), "visible");
        Ok(())
    }

    #[tokio::test]
    async fn inherit_full_policy_keeps_credentials() -> Result<()> {
        use std::ffi::OsString;

        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(r#"printf %s "${FAKE_TEST_API_KEY-missing}""#);
        command.stdin(Stdio::null());
        // InheritFull leaves the command env untouched; simulate the parent
        // env with an explicit Command-level var.
        command.env("FAKE_TEST_API_KEY", "still-here");

        EnvPolicy::InheritFull
            .apply_with_parent_env(&mut command, std::iter::empty::<(OsString, OsString)>());
        let output = command.output().await?;

        assert_eq!(String::from_utf8_lossy(&output.stdout), "still-here");
        Ok(())
    }

    #[tokio::test]
    async fn eval_propagates_explicit_traceparent_even_with_clean_env() -> Result<()> {
        let traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let mut clean_vars = std::collections::BTreeMap::new();
        if let Ok(path) = std::env::var("PATH") {
            clean_vars.insert("PATH".into(), path);
        }
        let result = run_eval_with_env(
            &EvalConfig {
                shell: "/bin/sh".into(),
                cwd: None,
                timeout: std::time::Duration::from_secs(1),
                max_stdout_bytes: 1024,
                max_stderr_bytes: 1024,
                env: EnvPolicy::Clean { vars: clean_vars },
            },
            "printf %s \"$TRACEPARENT\"",
            BTreeMap::from([("TRACEPARENT".into(), traceparent.into())]),
        )
        .await;

        assert_eq!(result?["stdout"], json!(traceparent));
        Ok(())
    }

    #[tokio::test]
    async fn argv_eval_execs_directly_without_shell_interpretation() -> Result<()> {
        // The spaced, $-laden element must arrive as ONE argv entry, verbatim:
        // no word splitting, no variable expansion — there is no shell between
        // the interpreter and the child. (/bin/sh is the child here only as a
        // portable printf; it prints its $0 argument without re-parsing it.)
        let result = run_eval_argv_with_env(
            &EvalConfig {
                shell: "/nonexistent-shell-must-not-be-used".into(),
                ..EvalConfig::default()
            },
            &[
                "/bin/sh".into(),
                "-c".into(),
                r#"printf %s "$0""#.into(),
                "hello $HOME world".into(),
            ],
            BTreeMap::new(),
        )
        .await?;

        assert_eq!(result["ok"], json!(true));
        assert_eq!(result["stdout"], json!("hello $HOME world"));
        Ok(())
    }

    #[tokio::test]
    async fn argv_eval_rejects_empty_argv() {
        let err = run_eval_argv_with_env(&EvalConfig::default(), &[], BTreeMap::new())
            .await
            .expect_err("empty argv has no program to exec");
        assert!(err.to_string().contains("argv must not be empty"));
    }

    #[tokio::test]
    async fn inherit_policy_strips_credentials_from_argv_children() -> Result<()> {
        // EnvPolicy::Inherit reads the real process env, so plant a uniquely
        // named fake credential there for the duration of this test.
        std::env::set_var("AGENT_TEST_ARGV_STRIP_API_KEY", "leaked");
        let result = run_eval_argv_with_env(
            &EvalConfig {
                env: EnvPolicy::Inherit,
                ..EvalConfig::default()
            },
            &[
                "/bin/sh".into(),
                "-c".into(),
                r#"printf %s "${AGENT_TEST_ARGV_STRIP_API_KEY-stripped}""#.into(),
            ],
            BTreeMap::new(),
        )
        .await;
        std::env::remove_var("AGENT_TEST_ARGV_STRIP_API_KEY");

        assert_eq!(result?["stdout"], json!("stripped"));
        Ok(())
    }

    #[tokio::test]
    async fn op_layer_argv_eval_records_argv_and_replays_with_divergence_detection() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = SeqConfig {
            trace,
            ..seq_config_for_eval()
        };
        let argv = ["/bin/sh", "-c", r#"printf %s "$0""#, "argv-ok"];
        let (recorded, _) = run_sequential(&config, (), crate::op::eval_argv(argv)).await?;
        assert_eq!(recorded["stdout"], json!("argv-ok"));

        let events = TraceLogger::read_events(&trace_path).await?;
        let recorded_argv = events
            .iter()
            .find_map(|event| match event {
                Event::EvalCall { argv, .. } => argv.clone(),
                _ => None,
            })
            .expect("argv Evals record their argv on EvalCall");
        assert_eq!(recorded_argv, argv.map(String::from));

        // Replay returns the recorded result without executing…
        let replay = ReplayTrace::from_events(&events);
        let config = SeqConfig {
            replay: Some(replay.clone()),
            ..seq_config_for_eval()
        };
        let (replayed, _) = run_sequential(&config, (), crate::op::eval_argv(argv)).await?;
        assert_eq!(replayed, recorded);

        // …a different argv diverges…
        let config = SeqConfig {
            replay: Some(replay.clone()),
            ..seq_config_for_eval()
        };
        let err = run_sequential(
            &config,
            (),
            crate::op::eval_argv(["/bin/sh", "-c", r#"printf %s "$0""#, "argv-changed"]),
        )
        .await
        .expect_err("changed argv must not replay");
        assert!(err.to_string().contains("replay diverged at Eval"), "{err}");

        // …and a shell Eval whose rendered command matches the argv rendering
        // still diverges: the identities are distinct.
        let config = SeqConfig {
            replay: Some(replay),
            ..seq_config_for_eval()
        };
        let rendered = crate::op::EvalSpec::Argv(argv.map(String::from).to_vec()).display_command();
        let err = run_sequential(&config, (), crate::op::eval(rendered))
            .await
            .expect_err("a shell Eval must not satisfy an argv recording");
        assert!(err.to_string().contains("replay diverged at Eval"), "{err}");
        Ok(())
    }

    fn seq_config_for_eval() -> SeqConfig {
        SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
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
    async fn eval_child_stdin_is_detached() -> Result<()> {
        // Regression: the child of an Eval op must not inherit the agent's stdin.
        // Otherwise an interactive `read` (or `git rebase -i`, `ssh`, etc.) would
        // consume the agent's own NUL-framed session/fifo control channel. With
        // stdin detached to /dev/null, `read` sees immediate EOF and returns
        // non-zero without blocking or stealing any input.
        let provider = Arc::new(MockProvider::new(vec![]));
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
            eval: EvalConfig {
                shell: "/bin/sh".into(),
                cwd: None,
                // Generous timeout: if stdin were inherited and blocked, the test
                // process has no stdin attached under cargo test so it would also
                // EOF -- but a real inherited terminal would hang. The key assertion
                // is that `read` reports failure (EOF), not success.
                timeout: std::time::Duration::from_secs(5),
                max_stdout_bytes: 64,
                max_stderr_bytes: 64,
                env: EnvPolicy::Inherit,
            },
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 200_000,
            pricing: Default::default(),
        };

        let (result, _) = run_sequential(
            &config,
            (),
            crate::op::eval("if read _x; then echo GOT; else echo EOF; fi"),
        )
        .await?;
        assert_eq!(result["timed_out"], json!(false));
        assert_eq!(result["stdout"], json!("EOF\n"));
        Ok(())
    }

    #[tokio::test]
    async fn trace_can_be_read_back_and_summarized() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("ok", vec![])]));
        let trace = test_trace();
        let path = trace.path().clone();
        let config = SeqConfig {
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
        };

        let _ = run_sequential(
            &config,
            vec![ChatMessage::user("hello")],
            infer(Model("mock".into()), vec![ChatMessage::user("hello")]),
        )
        .await?;
        let events = TraceLogger::read_events(path).await?;
        let summary = crate::trace::TraceSummary::from_events(&events);
        assert_eq!(summary.infer_calls, 1);
        assert_eq!(summary.total_tokens, 7);
        Ok(())
    }

    #[tokio::test]
    async fn replay_trace_feeds_recorded_infer_and_eval_results() -> Result<()> {
        let record_provider = Arc::new(MockProvider::new(vec![response("recorded", vec![])]));
        let record_trace = test_trace();
        let record_path = record_trace.path().clone();
        let record_config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: record_provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: record_trace,
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
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")])
            .and_then(|_| crate::op::eval("printf replayed"));
        let (recorded, _) = run_sequential(&record_config, (), program).await?;
        assert_eq!(recorded["stdout"], json!("replayed"));

        let replay = ReplayTrace::load(record_path).await?;
        let replay_config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: Some(replay),
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 200_000,
            pricing: Default::default(),
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")])
            .and_then(|_| crate::op::eval("printf replayed"));
        let (replayed, _) = run_sequential(&replay_config, (), program).await?;
        assert_eq!(replayed, recorded);
        Ok(())
    }

    /// t-1334 acceptance: a replayed run reproduces the ORIGINAL usage and
    /// cost exactly — the recorded values pass through the replayed
    /// Response, and neither the per-event cost nor the AgentDone rollup is
    /// recomputed from whatever pricing the replaying machine has today.
    #[tokio::test]
    async fn replay_reproduces_recorded_usage_and_cost_totals() -> Result<()> {
        let cost_fields = |event: &Event| match event {
            Event::InferResult {
                input_tokens,
                output_tokens,
                total_tokens,
                cached_input_tokens,
                cost_micro_usd,
                pricing,
                ..
            } => Some((
                *input_tokens,
                *output_tokens,
                *total_tokens,
                *cached_input_tokens,
                *cost_micro_usd,
                *pricing,
            )),
            _ => None,
        };
        let mut provider_response = response("recorded", vec![]);
        provider_response.cached_input_tokens = Some(2);
        let record_trace = test_trace();
        let record_path = record_trace.path().clone();
        let mut record_pricing = crate::cost::PricingTable::default();
        record_pricing.insert("mock", crate::cost::Pricing::from_usd_per_mtok(3.0, 15.0)?);
        let record_config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![provider_response])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: record_trace.clone(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 200_000,
            pricing: record_pricing,
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")]);
        let _ = run_sequential(&record_config, (), program).await?;
        record_trace
            .emit(&Event::AgentDone {
                run_id: record_trace.run_id().into(),
                usage: None,
                timestamp: Utc::now(),
            })
            .await?;

        let recorded_events = TraceLogger::read_events(&record_path).await?;
        let recorded = recorded_events
            .iter()
            .find_map(cost_fields)
            .expect("recorded InferResult");
        // 3 input + 4 output tokens at $3/$15 per Mtok = 69 micro-USD.
        let pricing = crate::cost::Pricing {
            input_micro_usd_per_mtok: 3_000_000,
            output_micro_usd_per_mtok: 15_000_000,
        };
        assert_eq!(recorded, (3, 4, 7, Some(2), Some(69), Some(pricing)));

        // Replay under a WILDLY different pricing table: if anything
        // recomputed, cost would come out at $100/Mtok rates, not 69.
        let mut replay_pricing = crate::cost::PricingTable::default();
        replay_pricing.insert(
            "mock",
            crate::cost::Pricing::from_usd_per_mtok(100.0, 100.0)?,
        );
        let replay_trace = test_trace();
        let replay_path = replay_trace.path().clone();
        let replay_config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: replay_trace.clone(),
            eval: EvalConfig::default(),
            replay: Some(ReplayTrace::from_events(&recorded_events)),
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 200_000,
            pricing: replay_pricing,
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")]);
        let (replayed_response, _) = run_sequential(&replay_config, (), program).await?;
        assert_eq!(replayed_response.cost_micro_usd, Some(69));
        replay_trace
            .emit(&Event::AgentDone {
                run_id: replay_trace.run_id().into(),
                usage: None,
                timestamp: Utc::now(),
            })
            .await?;

        let replayed_events = TraceLogger::read_events(&replay_path).await?;
        let replayed = replayed_events
            .iter()
            .find_map(cost_fields)
            .expect("replayed InferResult");
        assert_eq!(replayed, recorded, "per-event usage/cost must match");
        let rollup = |events: &[Event]| {
            events.iter().find_map(|event| match event {
                Event::AgentDone { usage, .. } => usage.clone(),
                _ => None,
            })
        };
        let recorded_rollup = rollup(&recorded_events).expect("recorded rollup");
        assert_eq!(recorded_rollup.cost_micro_usd, Some(69));
        assert_eq!(
            rollup(&replayed_events),
            Some(recorded_rollup),
            "replayed run rollup must equal the original"
        );
        Ok(())
    }

    #[tokio::test]
    async fn full_payloads_are_omitted_from_traces_by_default() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("ok", vec![])]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = SeqConfig {
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
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")]);

        let _ = run_sequential(&config, vec![ChatMessage::user("hello")], program).await?;

        let events = TraceLogger::read_events(trace_path).await?;
        for event in &events {
            if let Event::InferCall {
                prompt,
                prompt_preview,
                ..
            } = event
            {
                assert!(prompt.is_none(), "full prompt must be opt-in");
                assert!(!prompt_preview.is_empty());
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn provider_failure_emits_infer_error_and_replays_as_failure() -> Result<()> {
        // Record a run whose provider fails terminally.
        let record_trace = test_trace();
        let record_path = record_trace.path().clone();
        let record_config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: record_trace,
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
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")]);
        let err = run_sequential(&record_config, (), program)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("mock provider exhausted"));

        let events = TraceLogger::read_events(&record_path).await?;
        let infer_error = events
            .iter()
            .find_map(|event| match event {
                Event::InferError { op_id, error, .. } => Some((op_id, error.clone())),
                _ => None,
            })
            .expect("failed run must record an InferError event");
        assert!(infer_error.1.contains("mock provider exhausted"));

        // Replaying the failed run reproduces the failure without touching
        // the provider.
        let replay = ReplayTrace::load(record_path).await?;
        let live_provider = Arc::new(MockProvider::new(vec![response("unused", vec![])]));
        let replay_config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: live_provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: Some(replay),
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 200_000,
            pricing: Default::default(),
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")]);
        let err = run_sequential(&replay_config, (), program)
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("replaying recorded Infer failure"), "{err}");
        assert!(err.contains("mock provider exhausted"), "{err}");
        assert_eq!(live_provider.prompt_count(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn eval_spawn_failure_emits_eval_error() -> Result<()> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = SeqConfig {
            approvals: Default::default(),
            guidance: Default::default(),
            tools: Default::default(),
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            trace,
            eval: EvalConfig {
                shell: "/nonexistent-shell-for-eval-error-test".into(),
                ..EvalConfig::default()
            },
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
            context_budget: 200_000,
            pricing: Default::default(),
        };

        let result = run_sequential(&config, (), crate::op::eval("printf hi")).await;

        assert!(result.is_err());
        let events = TraceLogger::read_events(trace_path).await?;
        assert!(
            events.iter().any(
                |event| matches!(event, Event::EvalError { command, .. } if command == "printf hi")
            ),
            "failed eval must record an EvalError event: {events:?}"
        );
        Ok(())
    }
}
