use crate::gc::{estimate_tokens, truncate_oversized_message, GcMode, GcState};
use crate::hydration::{
    PassiveHydrationConfig, PassiveSource, SourceParams, SourceRegistry, SEMANTIC_PREFIX,
    SESSION_STATE_KEY, TEMPORAL_PREFIX,
};
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

#[derive(Clone, Debug, Default)]
pub struct ReplayTrace {
    infer_calls: BTreeMap<u64, String>,
    infer_results: BTreeMap<u64, Response>,
    infer_errors: BTreeMap<u64, String>,
    eval_calls: BTreeMap<u64, String>,
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
                Event::EvalCall { op_id, command, .. } => {
                    replay.eval_calls.insert(*op_id, command.clone());
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

    pub(crate) fn eval_result(&self, op_id: u64, command: &str) -> Result<Value> {
        if let Some(recorded_command) = self.eval_calls.get(&op_id) {
            if recorded_command != command {
                return Err(anyhow!(
                    "replay diverged at Eval op {op_id}: recorded command '{recorded_command}', requested '{command}'"
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
    pub passive_hydration: PassiveHydrationConfig,
    pub checkpoint_path: Option<PathBuf>,
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
    pub context_budget: usize,
}

impl SeqConfig {
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        vec![ToolSpec {
            kind: "function".into(),
            function: ToolFunctionSpec {
                name: "shell".into(),
                description: "Execute a command string using the configured shell.".into(),
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
            let prompt = maybe_collect_prompt(config, prompt, gc_state).await?;
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
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            let result = match &config.replay {
                Some(replay) => replay.infer_result(op_id, &model.0),
                None => {
                    config
                        .provider
                        .chat(&model, &config.tool_specs(), &prompt)
                        .await
                }
            };
            let response = match result {
                Ok(response) => response,
                Err(err) => {
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
                    duration_ms: millis_u64(started.elapsed()),
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential_inner(config, state, next(response), gc_state, parent_op_id).await
        }
        OpF::Eval { command, next } => {
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::EvalCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
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
            let result = match &config.replay {
                Some(replay) => replay.eval_result(op_id, &command),
                None => {
                    run_eval_with_env(&config.eval, &command, config.trace.trace_context_env())
                        .await
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
        OpF::Get { key, next } => {
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::GetCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    key: key.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            let value = dispatch_get(config, &state, &key).await?;
            config
                .trace
                .emit(&Event::GetResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    key,
                    source_count: value.as_array().map(Vec::len).unwrap_or(0),
                    value_preview: preview(&value.to_string(), 512),
                    value: config.trace_full_payloads.then(|| value.clone()),
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential_inner(config, state, next(value), gc_state, parent_op_id).await
        }
        OpF::Put { key, value, next } => {
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::PutCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    key: key.clone(),
                    value_preview: preview(&value.to_string(), 512),
                    timestamp: Utc::now(),
                })
                .await?;
            let state = dispatch_put(config, state, &key, value).await?;
            config
                .trace
                .emit(&Event::PutResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    parent_op_id,
                    key,
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential_inner(config, state, next, gc_state, parent_op_id).await
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

pub(crate) async fn maybe_collect_prompt(
    config: &SeqConfig,
    mut prompt: Prompt,
    gc_state: &mut GcState,
) -> Result<Prompt> {
    if !config.gc.enabled() {
        return Ok(prompt);
    }
    let before_tokens = estimate_tokens(&prompt);
    let threshold = ((config.context_budget as f32) * config.gc_threshold) as usize;
    if before_tokens <= threshold {
        return Ok(prompt);
    }
    let target_budget = threshold.max(1);
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
    let before_ids: BTreeSet<_> = prompt.iter().map(|message| message.id).collect();
    let collected = config.gc.collect(prompt, target_budget, gc_state);
    let after_ids: BTreeSet<_> = collected.iter().map(|message| message.id).collect();
    let dropped_count = before_ids.difference(&after_ids).count();
    let after_tokens = estimate_tokens(&collected);
    if config.gc_log {
        config
            .trace
            .emit(&Event::Custom {
                run_id: config.trace.run_id().into(),
                name: "gc_collect".into(),
                data: serde_json::json!({
                    "type": "gc_collect",
                    "strategy": config.gc.name(),
                    "tokens_before": before_tokens,
                    "tokens_after": after_tokens,
                    "cache_invalidated": gc_state.prefix_invalidated,
                    "dropped_count": dropped_count,
                }),
                timestamp: Utc::now(),
            })
            .await?;
    }
    Ok(collected)
}

pub(crate) async fn run_eval_with_env(
    config: &EvalConfig,
    command: &str,
    extra_env: BTreeMap<String, String>,
) -> Result<Value> {
    let started = Instant::now();
    let mut process = Command::new(&config.shell);
    process.arg("-c").arg(command);
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

pub(crate) async fn dispatch_get<S>(config: &SeqConfig, state: &S, key: &str) -> Result<Value>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    if key.starts_with(TEMPORAL_PREFIX) {
        return serde_json::to_value(state).map_err(Into::into);
    }
    if key.starts_with(SEMANTIC_PREFIX) {
        let query = key.trim_start_matches(SEMANTIC_PREFIX);
        let results = config
            .hydration
            .retrieve_query(SourceParams::new(query))
            .await?;
        return serde_json::to_value(results).map_err(Into::into);
    }
    if key == SESSION_STATE_KEY {
        let Some(path) = &config.checkpoint_path else {
            return Ok(Value::Null);
        };
        match tokio::fs::read_to_string(path).await {
            Ok(content) => serde_json::from_str(&content).map_err(Into::into),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
            Err(err) => Err(err.into()),
        }
    } else {
        Err(anyhow!("unknown Get key: {key}"))
    }
}

pub(crate) async fn dispatch_put<S>(
    config: &SeqConfig,
    state: S,
    key: &str,
    value: Value,
) -> Result<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    if key.starts_with(TEMPORAL_PREFIX) {
        return serde_json::from_value(value).map_err(Into::into);
    }
    if key == SESSION_STATE_KEY {
        let Some(path) = &config.checkpoint_path else {
            return Ok(state);
        };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, serde_json::to_vec_pretty(&value)?).await?;
        Ok(state)
    } else {
        Err(anyhow!("unknown Put key: {key}"))
    }
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

    fn response(content: &str, tool_calls: Vec<ToolCall>) -> Response {
        Response {
            content: content.into(),
            tool_calls,
            finish_reason: Some(crate::op::FinishReason::Stop),
            input_tokens: 3,
            output_tokens: 4,
            total_tokens: 7,
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
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: crate::gc::GcMode::Ring(crate::gc::RingGc::default()),
            gc_threshold: 0.5,
            gc_log: false,
            context_budget: 100,
        };
        let prompt = vec![
            ChatMessage::system("system"),
            ChatMessage::user("x".repeat(90)),
            ChatMessage::user("y".repeat(90)),
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
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: crate::gc::GcMode::Ring(crate::gc::RingGc::default()),
            gc_threshold: 0.5,
            gc_log: true,
            context_budget: 100,
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
            provider: provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };
        let prompt = vec![ChatMessage::user("use echo")];

        let (result, state) = run_sequential(
            &config,
            prompt.clone(),
            agent_loop(Model("mock".into()), prompt, 4),
        )
        .await?;

        assert_eq!(result.content, "done");
        assert_eq!(provider.prompt_count(), 2);
        assert!(state.iter().any(|msg| msg.role == "tool"));
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
            provider: provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };
        let prompt = vec![ChatMessage::user("do two steps")];

        let (result, state) = run_sequential(
            &config,
            prompt.clone(),
            agent_loop(Model("mock".into()), prompt, 4),
        )
        .await?;

        assert_eq!(result.content, "finished");
        assert_eq!(provider.prompt_count(), 3);
        assert_eq!(state.iter().filter(|msg| msg.role == "tool").count(), 2);
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
            provider: provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
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

    #[tokio::test]
    async fn get_put_and_par_are_interpreted_sequentially() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };
        let program = crate::op::put("temporal:history", json!(1))
            .and_then(|_| {
                crate::op::par(vec![
                    crate::op::put("temporal:history", json!(2)),
                    crate::op::put("temporal:history", json!(3)),
                ])
            })
            .and_then(|_| crate::op::get("temporal:history"));

        let (result, state) = run_sequential(&config, 0, program).await?;

        assert_eq!(result, json!(3));
        assert_eq!(state, 3);
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
    async fn session_state_get_reads_checkpoint_json() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let path = std::env::temp_dir().join(format!("agent-core-session-{}.json", Uuid::new_v4()));
        tokio::fs::write(&path, serde_json::to_vec(&json!({ "checkpoint": 7 }))?).await?;
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: Some(path),
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };

        let initial_state = vec![ChatMessage::user("state")];
        let (value, state) = run_sequential(
            &config,
            initial_state.clone(),
            crate::op::get("session:state"),
        )
        .await?;

        assert_eq!(value, json!({ "checkpoint": 7 }));
        assert_eq!(state, initial_state);
        Ok(())
    }

    #[tokio::test]
    async fn session_state_put_writes_checkpoint_json() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let path = std::env::temp_dir().join(format!("agent-core-session-{}.json", Uuid::new_v4()));
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: Some(path.clone()),
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };

        let initial_state = vec![ChatMessage::user("state")];
        let (_, state) = run_sequential(
            &config,
            initial_state.clone(),
            crate::op::put("session:state", json!({ "checkpoint": 8 })),
        )
        .await?;
        let content: Value = serde_json::from_slice(&tokio::fs::read(path).await?)?;

        assert_eq!(content, json!({ "checkpoint": 8 }));
        assert_eq!(state, initial_state);
        Ok(())
    }

    #[tokio::test]
    async fn infer_injects_configured_passive_context_without_agent_get() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("ok", vec![])]));
        let queries = Arc::new(Mutex::new(Vec::new()));
        let config = SeqConfig {
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
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
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
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig {
                shell: "/bin/sh".into(),
                cwd: Some(cwd.clone()),
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
            context_budget: 200_000,
        };

        let (timeout_result, _) = run_sequential(&config, (), crate::op::eval("sleep 1")).await?;
        assert_eq!(timeout_result["timed_out"], json!(true));

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
    async fn eval_child_stdin_is_detached() -> Result<()> {
        // Regression: the child of an Eval op must not inherit the agent's stdin.
        // Otherwise an interactive `read` (or `git rebase -i`, `ssh`, etc.) would
        // consume the agent's own NUL-framed session/fifo control channel. With
        // stdin detached to /dev/null, `read` sees immediate EOF and returns
        // non-zero without blocking or stealing any input.
        let provider = Arc::new(MockProvider::new(vec![]));
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
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
            context_budget: 200_000,
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
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
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
            provider: record_provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: record_trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")])
            .and_then(|_| crate::op::eval("printf replayed"));
        let (recorded, _) = run_sequential(&record_config, (), program).await?;
        assert_eq!(recorded["stdout"], json!("replayed"));

        let replay = ReplayTrace::load(record_path).await?;
        let replay_config = SeqConfig {
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: Some(replay),
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")])
            .and_then(|_| crate::op::eval("printf replayed"));
        let (replayed, _) = run_sequential(&replay_config, (), program).await?;
        assert_eq!(replayed, recorded);
        Ok(())
    }

    #[tokio::test]
    async fn full_payloads_are_omitted_from_traces_by_default() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("ok", vec![])]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };
        let program = infer(Model("mock".into()), vec![ChatMessage::user("hello")])
            .and_then(|_| crate::op::get("temporal:history"));

        let _ = run_sequential(&config, vec![ChatMessage::user("hello")], program).await?;

        let events = TraceLogger::read_events(trace_path).await?;
        for event in &events {
            match event {
                Event::InferCall {
                    prompt,
                    prompt_preview,
                    ..
                } => {
                    assert!(prompt.is_none(), "full prompt must be opt-in");
                    assert!(!prompt_preview.is_empty());
                }
                Event::GetResult {
                    value,
                    value_preview,
                    ..
                } => {
                    assert!(value.is_none(), "full Get value must be opt-in");
                    assert!(!value_preview.is_empty());
                }
                _ => {}
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
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: record_trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
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
            provider: live_provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: Some(replay),
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
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
            provider: Arc::new(MockProvider::new(vec![])),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
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
            context_budget: 200_000,
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

    #[tokio::test]
    async fn active_semantic_get_uses_query_capable_backend() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let queries = Arc::new(Mutex::new(Vec::new()));
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new().register(StaticSource {
                name: "semantic-store",
                kind: SourceKind::Semantic,
                capabilities: SourceCapability::QUERY,
                content: "semantic result",
                queries: queries.clone(),
            }),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        };

        let (value, _) = run_sequential(&config, (), crate::op::get("semantic:topic")).await?;

        assert_eq!(queries.lock().unwrap().as_slice(), &[Some("topic".into())]);
        assert_eq!(value[0]["content"], json!("semantic result"));
        Ok(())
    }
}
