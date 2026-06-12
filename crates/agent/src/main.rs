use agent_core::{
    agent_loop_ir, AgentIdGenerator, AnthropicConfig, AnthropicProvider, ChatMessage, EnvPolicy,
    EvalConfig, Event, GcMode, GcTiming, HydrationSource, InMemoryStore, IrReplayTrace,
    JsonlTraceSink, MarkSweepGc, ModelRegistry, OtelTraceSink, PassiveHydrationConfig,
    PassiveSource, ProviderClient, ProviderConfig, ResolvedModel, RingGc, SeqConfig,
    SourceCapability, SourceKind, SourceParams, SourceRegistry, SourceResult, StackFrameGc,
    TraceContextEnv, TraceLogger,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::{logs::SdkLoggerProvider, trace::SdkTracerProvider, Resource};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Read};
#[cfg(unix)]
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use uuid::Uuid;

mod frontmatter;

/// Soft turn ceiling per session turn (Ben's decision on t-1133). Models like
/// gpt-5.5 issue one tool call per assistant turn, so real inspect/edit loops
/// burn turns fast; 100 is generous enough that no legitimate task dies one
/// step from the line, while still bounding a runaway loop before it burns
/// real spend or hits the context wall. Hitting it is reported (typed
/// TurnBudgetExhausted event + non-empty terminal notice), not fatal-looking.
const DEFAULT_MAX_TURNS: usize = 100;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    provider: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    key: Option<String>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, alias = "json")]
    debug: bool,
    /// Stable run id used for traces/checkpoints.
    #[arg(long, env = "AGENT_RUN_ID")]
    run_id: Option<String>,
    /// Read NUL-terminated session turns from stdin.
    #[arg(long)]
    session: bool,
    /// Read NUL-terminated session turns from this FIFO path.
    #[arg(long, env = "AGENT_FIFO")]
    fifo: Option<PathBuf>,
    /// Write a checkpoint JSON after each completed turn.
    #[arg(long, env = "AGENT_CHECKPOINT_DIR")]
    checkpoint_dir: Option<PathBuf>,
    /// Resume conversation history from a checkpoint JSON.
    #[arg(long, env = "AGENT_RESUME")]
    resume: Option<PathBuf>,
    /// Replay recorded Infer/Eval results from a trace JSONL instead of calling providers or shell.
    #[arg(long, env = "AGENT_REPLAY_TRACE")]
    replay_trace: Option<PathBuf>,
    /// Store full PromptIR section content in traces instead of previews/hashes only.
    #[arg(long, env = "AGENT_TRACE_FULL_PROMPT_IR")]
    trace_full_prompt_ir: bool,
    /// Store full Infer prompts and Get values in traces. Off by default:
    /// full prompts repeat the whole conversation per call (O(n^2) trace
    /// growth) and replay only needs recorded results. Previews are always
    /// stored. Enable when recording fixtures that need full prompts (e.g.
    /// GC eval traces).
    #[arg(long, env = "AGENT_TRACE_FULL_PAYLOADS")]
    trace_full_payloads: bool,
    /// Directory to read into passive hydration context.
    #[arg(long, env = "AGENT_HYDRATION_DIR")]
    hydration_dir: Option<PathBuf>,
    /// Timeout for each Eval shell command.
    #[arg(long, default_value_t = 120)]
    eval_timeout_seconds: u64,
    /// Maximum bytes captured from stdout and stderr for each Eval command.
    #[arg(long)]
    eval_max_output_bytes: Option<usize>,
    /// Working directory for Eval shell commands.
    #[arg(long)]
    eval_cwd: Option<PathBuf>,
    /// Environment policy for Eval shell commands. `inherit` (default)
    /// passes the parent environment minus known credential vars
    /// (ANTHROPIC_AUTH_TOKEN and anything ending in _API_KEY) so model-issued
    /// commands cannot read the key the agent runs on; `inherit-full` passes
    /// everything, credentials included; `clean` passes nothing.
    #[arg(long, value_enum, default_value_t = EvalEnvMode::Inherit)]
    eval_env: EvalEnvMode,
    /// Context GC strategy.
    #[arg(long, value_enum, default_value_t = GcArg::Ring)]
    gc: GcArg,
    /// Trigger GC at this fraction of the model context budget.
    #[arg(long, default_value_t = 0.85)]
    gc_threshold: f32,
    /// Emit gc_collect trace events.
    #[arg(long)]
    gc_log: bool,
    /// When GC runs: `threshold` (default) collects past the estimated
    /// budget fraction; `catch-overflow` trusts the provider instead of the
    /// token estimate — on a context-overflow error it collects to a
    /// shrinking budget and retries the same turn; `eager` collects before
    /// every infer call; `every:N` collects on every Nth infer call.
    #[arg(long, default_value = "threshold")]
    gc_timing: GcTiming,
    /// Enable OpenTelemetry OTLP export to this collector endpoint. Also enabled by OTEL_EXPORTER_OTLP_ENDPOINT.
    #[arg(long, env = "OTEL_EXPORTER_OTLP_ENDPOINT")]
    otel_endpoint: Option<String>,
    /// Prompt-cache policy for GC. `preserve` (default) pins the system
    /// prompt plus the oldest ~25% of the budget as a stable cache prefix and
    /// evicts from the interior, falling back to front-drop only when that
    /// cannot reach the budget; `ignore` maximizes token reclaim and may
    /// invalidate the provider prompt cache on every collection.
    #[arg(long, value_enum, default_value_t = GcCacheArg::Preserve)]
    gc_cache: GcCacheArg,
    /// Accept compaction flag for agentd compatibility; compaction is not implemented yet.
    #[arg(long)]
    enable_compaction: bool,
    /// One-shot prompt text or path to a .md/.markdown prompt file. Omit when using --fifo or NUL-framed stdin sessions.
    prompt: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum GcArg {
    None,
    Ring,
    MarkSweep,
    Stack,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum GcCacheArg {
    Preserve,
    Ignore,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EvalEnvMode {
    Inherit,
    InheritFull,
    Clean,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print GC statistics from a trace JSONL file.
    GcStats { trace: PathBuf },
    /// Print the AgentIR effect-location JSON for the entry Infer of the
    /// built-in agent loop. Eval scripts use it to build replay fixtures
    /// without hardcoding program hashes.
    #[command(hide = true)]
    IrEffect {
        #[arg(long)]
        model: String,
        /// Visit count for the effect site (the Nth Infer of a session).
        #[arg(long, default_value_t = 0)]
        visit: u64,
    },
    #[cfg(feature = "oauth")]
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
}

#[cfg(feature = "oauth")]
#[derive(Debug, Subcommand)]
enum AuthCommand {
    Login { provider: String },
    Status,
}

#[derive(Debug, Deserialize, Default)]
struct FileConfig {
    provider: Option<FileProvider>,
}

#[derive(Debug, Deserialize, Default)]
struct FileProvider {
    url: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    run_id: String,
    sequence: u64,
    model: String,
    provider_url: String,
    messages: Vec<ChatMessage>,
    trace_path: PathBuf,
    timestamp: DateTime<Utc>,
}

struct ReplayOnlyProvider;

#[async_trait]
impl agent_core::ChatProvider for ReplayOnlyProvider {
    async fn chat(
        &self,
        _model: &agent_core::Model,
        _tools: &[agent_core::provider::ToolSpec],
        _messages: &[ChatMessage],
    ) -> Result<agent_core::Response> {
        Err(anyhow!(
            "replay provider was called; trace is missing a recorded InferResult for this op"
        ))
    }
}

struct Runtime {
    config: SeqConfig,
    trace: TraceLogger,
    run_id: String,
    model: agent_core::Model,
    provider_url: String,
    trace_path: PathBuf,
    checkpoint_dir: Option<PathBuf>,
    checkpoint_path: Option<PathBuf>,
    checkpoint_sequence: u64,
    history: Vec<ChatMessage>,
    debug: bool,
    max_turns: usize,
    ir_store: InMemoryStore,
    ir_replay: Option<IrReplayTrace>,
    ir_effect_visits: BTreeMap<String, u64>,
    gc_threshold: f32,
    context_budget: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(command) = args.command.as_ref() {
        return run_command(command).await;
    }
    let file_config = read_config(args.config.as_ref()).await?;
    let provider_file = file_config.provider.unwrap_or_default();

    let loaded_prompt = match args.prompt.as_ref() {
        Some(prompt) => Some(frontmatter::MarkdownPrompt::from_arg(prompt).await?),
        None => None,
    };
    let frontmatter = loaded_prompt
        .as_ref()
        .and_then(|prompt| prompt.frontmatter.as_ref());
    let requested_model = args
        .model
        .clone()
        .or_else(|| frontmatter.and_then(|meta| meta.model.clone()));
    let requested_provider = args
        .provider
        .clone()
        .or_else(|| frontmatter.and_then(|meta| meta.provider.clone()));
    let max_turns = frontmatter
        .and_then(|meta| meta.max_iterations)
        .unwrap_or(DEFAULT_MAX_TURNS);
    let system_prompt_override = match loaded_prompt.as_ref() {
        Some(prompt) => {
            frontmatter::resolve_system_prompt(
                &prompt.base_dir,
                frontmatter.and_then(|meta| meta.system_prompt.as_deref()),
            )
            .await?
        }
        None => None,
    };
    let system_prompt = build_system_prompt(system_prompt_override).await?;

    let resolved_model = resolve_model(requested_model, provider_file.model.clone()).await?;
    let eval_config = EvalConfig {
        cwd: args.eval_cwd.clone(),
        timeout: Duration::from_secs(args.eval_timeout_seconds),
        max_stdout_bytes: args.eval_max_output_bytes.unwrap_or(1024 * 1024),
        max_stderr_bytes: args.eval_max_output_bytes.unwrap_or(1024 * 1024),
        env: match args.eval_env {
            EvalEnvMode::Inherit => EnvPolicy::Inherit,
            EvalEnvMode::InheritFull => EnvPolicy::InheritFull,
            EvalEnvMode::Clean => EnvPolicy::Clean {
                vars: Default::default(),
            },
        },
        ..EvalConfig::default()
    };
    let provider_tag = resolved_model.provider.as_deref();
    let is_anthropic_provider = provider_tag == Some("anthropic");
    let oauth_provider = provider_tag.filter(|provider| is_oauth_provider_tag(provider));
    let url = requested_provider
        .or(provider_file.url)
        .or(resolved_model.base_url.clone())
        .or_else(|| std::env::var("AGENT_PROVIDER").ok())
        .or_else(|| std::env::var("OPENROUTER_BASE_URL").ok())
        .unwrap_or_else(|| {
            if is_anthropic_provider {
                "https://api.anthropic.com/v1".into()
            } else {
                "https://openrouter.ai/api/v1".into()
            }
        });
    let reported_provider_url = reported_provider_url(oauth_provider, &url);
    let replay_enabled = args.replay_trace.is_some();
    let ir_replay = match args.replay_trace.as_ref() {
        Some(path) => Some(IrReplayTrace::load(path).await?),
        None => None,
    };
    let context_budget = resolved_model.context;
    let model = resolved_model.api_id.clone();
    #[cfg(not(feature = "oauth"))]
    if !replay_enabled {
        if let Some(provider) = oauth_provider {
            return Err(anyhow!(
                "model '{}' requires OAuth provider '{provider}', but this agent was built without the 'oauth' feature",
                resolved_model.alias
            ));
        }
    }
    let api_key = if oauth_provider.is_some() || replay_enabled {
        None
    } else {
        Some(
            args.key
                .or(provider_file.api_key)
                .or(resolved_model.api_key.clone())
                .or_else(|| std::env::var("AGENT_API_KEY").ok())
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
                .ok_or_else(|| {
                    anyhow!("missing API key: pass --key, set AGENT_API_KEY/ANTHROPIC_API_KEY/OPENROUTER_API_KEY, or configure api_key in models.yaml")
                })?,
        )
    };

    let checkpoint = match args.resume.as_ref() {
        Some(path) => Some(load_checkpoint(path).await?),
        None => None,
    };
    let run_id = checkpoint
        .as_ref()
        .map(|cp| cp.run_id.clone())
        .or(args.run_id.clone())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let trace_path = trace_path(&run_id)?;
    let otel = init_otel(args.otel_endpoint.as_deref(), &run_id)?;
    let trace = match &otel {
        Some(_) => {
            let context_env = TraceContextEnv::default();
            TraceLogger::with_sinks_and_context(
                run_id.clone(),
                trace_path.clone(),
                vec![
                    Arc::new(JsonlTraceSink::new(trace_path.clone()).mirror_stdout(args.debug)),
                    Arc::new(OtelTraceSink::with_context_env(context_env.clone())),
                ],
                context_env,
            )
        }
        None => TraceLogger::new(run_id.clone(), trace_path.clone()).mirror_stdout(args.debug),
    };
    let provider: Arc<dyn agent_core::ChatProvider> = if replay_enabled {
        Arc::new(ReplayOnlyProvider)
    } else {
        match oauth_provider {
            Some(tag) => {
                #[cfg(feature = "oauth")]
                {
                    agent_oauth::provider_for_tag(tag, agent_core::Model(model.clone()))?
                        .map(Arc::from)
                        .ok_or_else(|| anyhow!("unsupported OAuth provider tag: {tag}"))?
                }
                #[cfg(not(feature = "oauth"))]
                {
                    return Err(anyhow!("unsupported OAuth provider tag: {tag}"));
                }
            }
            None if is_anthropic_provider => Arc::new(AnthropicProvider::new(AnthropicConfig {
                base_url: url.clone(),
                api_key: api_key.expect("api_key is set for non-OAuth providers"),
                model: agent_core::Model(model.clone()),
                max_tokens: resolved_model.max_tokens,
            })),
            None => Arc::new(ProviderClient::new(ProviderConfig {
                url: url.clone(),
                api_key: api_key.expect("api_key is set for non-OAuth providers"),
                model: agent_core::Model(model.clone()),
            })),
        }
    };
    let checkpoint_path = args
        .checkpoint_dir
        .as_ref()
        .map(|dir| dir.join("session-latest.json"));
    let hydration = match args.hydration_dir.as_ref() {
        Some(path) => SourceRegistry::new().register(LocalFileSource::new(path.clone())),
        None => SourceRegistry::new(),
    };
    let (history, checkpoint_sequence) = match checkpoint {
        Some(cp) => (cp.messages, cp.sequence),
        None => (initial_history(system_prompt), 0),
    };
    let config = SeqConfig {
        provider,
        hydration: hydration.clone(),
        passive_hydration: PassiveHydrationConfig::with_sources([
            PassiveSource::TemporalHistory,
            PassiveSource::SessionContext,
        ]),
        checkpoint_path: checkpoint_path.clone(),
        trace: trace.clone(),
        eval: eval_config,
        replay: None,
        trace_full_prompt_ir: args.trace_full_prompt_ir,
        trace_full_payloads: args.trace_full_payloads,
        gc: {
            let preserve_prefix = matches!(args.gc_cache, GcCacheArg::Preserve);
            match args.gc {
                GcArg::None => GcMode::None,
                GcArg::Ring => GcMode::Ring(RingGc { preserve_prefix }),
                GcArg::MarkSweep => GcMode::MarkSweep(MarkSweepGc { preserve_prefix }),
                GcArg::Stack => GcMode::Stack(StackFrameGc { preserve_prefix }),
            }
        },
        gc_threshold: args.gc_threshold,
        gc_log: args.gc_log,
        gc_timing: args.gc_timing,
        context_budget,
    };
    if !config.gc.enabled() && args.gc_timing != GcTiming::Threshold {
        return Err(anyhow!(
            "--gc-timing {} requires a GC strategy; pass --gc ring or --gc mark-sweep",
            args.gc_timing.name()
        ));
    }
    let mut runtime = Runtime {
        config,
        trace,
        run_id: run_id.clone(),
        model: agent_core::Model(model.clone()),
        provider_url: reported_provider_url.clone(),
        trace_path: trace_path.clone(),
        checkpoint_dir: args.checkpoint_dir,
        checkpoint_path,
        checkpoint_sequence,
        history,
        debug: args.debug,
        max_turns,
        ir_store: InMemoryStore::new(),
        ir_replay,
        ir_effect_visits: BTreeMap::new(),
        gc_threshold: args.gc_threshold,
        context_budget,
    };

    tracing::info!(%model, trace = %trace_path.display(), %run_id, provider = %reported_provider_url, "agent runtime starting");
    eprintln!("model: {model}");
    eprintln!("trace: {}", trace_path.display());
    eprintln!("run_id: {run_id}");
    eprintln!("provider: {reported_provider_url}");
    if let Some(prompt) = loaded_prompt.as_ref() {
        eprintln!("prompt: {}", prompt.body);
    }

    let result = match (loaded_prompt, args.fifo, args.session) {
        (Some(prompt), None, false) => {
            let prompt = prompt_with_optional_stdin(prompt.body)?;
            run_one_shot(&mut runtime, prompt).await
        }
        (Some(_), Some(_), _) => Err(anyhow!("provide either a prompt or --fifo, not both")),
        (Some(_), None, true) => Err(anyhow!("provide either a prompt or --session, not both")),
        (None, Some(path), _) => run_fifo_session(&mut runtime, path).await,
        (None, None, _) => run_stdin_session(&mut runtime).await,
    };
    if let Some(otel) = otel {
        otel.shutdown();
    }
    result
}

struct OtelGuard {
    tracer_provider: SdkTracerProvider,
    logger_provider: SdkLoggerProvider,
}

impl OtelGuard {
    fn shutdown(self) {
        let _ = self.tracer_provider.shutdown();
        let _ = self.logger_provider.shutdown();
    }
}

fn init_otel(endpoint: Option<&str>, run_id: &str) -> Result<Option<OtelGuard>> {
    let Some(endpoint) = endpoint.filter(|endpoint| !endpoint.trim().is_empty()) else {
        return Ok(None);
    };
    let mut resource = Resource::builder()
        .with_service_name("agentd")
        .with_attribute(KeyValue::new("agent.run_id", run_id.to_string()));
    if let Ok(agent_name) = std::env::var("AGENT_NAME") {
        resource = resource.with_attribute(KeyValue::new("agent.name", agent_name));
    }
    if let Ok(parent_run_id) = std::env::var("AGENT_PARENT_RUN_ID") {
        resource = resource.with_attribute(KeyValue::new("agent.parent_run_id", parent_run_id));
    }
    let resource = resource.build();
    let span_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(endpoint)
        .build()
        .context("building OTLP span exporter")?;
    let tracer_provider = SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_id_generator(AgentIdGenerator::default())
        .with_batch_exporter(span_exporter)
        .build();
    global::set_tracer_provider(tracer_provider.clone());

    let log_exporter = opentelemetry_otlp::LogExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(endpoint)
        .build()
        .context("building OTLP log exporter")?;
    let logger_provider = SdkLoggerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(log_exporter)
        .build();
    let otel_log_layer =
        opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(&logger_provider);
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(otel_log_layer)
        .try_init()
        .context("initializing tracing subscriber")?;

    Ok(Some(OtelGuard {
        tracer_provider,
        logger_provider,
    }))
}

fn is_oauth_provider_tag(provider: &str) -> bool {
    matches!(
        provider,
        "openai-codex" | "codex-oauth" | "claude-code" | "claude-code-oauth"
    )
}

fn oauth_provider_base_url(provider: &str) -> Option<&'static str> {
    #[cfg(feature = "oauth")]
    {
        agent_oauth::provider_base_url_for_tag(provider)
    }
    #[cfg(not(feature = "oauth"))]
    {
        match provider {
            "openai-codex" | "codex-oauth" => Some("https://chatgpt.com/backend-api"),
            "claude-code" | "claude-code-oauth" => Some("https://api.anthropic.com/v1"),
            _ => None,
        }
    }
}

fn reported_provider_url(oauth_provider: Option<&str>, resolved_url: &str) -> String {
    oauth_provider
        .and_then(oauth_provider_base_url)
        .map(str::to_string)
        .unwrap_or_else(|| resolved_url.to_string())
}

async fn resolve_model(
    args_model: Option<String>,
    file_model: Option<String>,
) -> Result<ResolvedModel> {
    let requested = args_model
        .or(file_model)
        .or_else(|| std::env::var("AGENT_MODEL").ok());
    resolve_model_from(ModelRegistry::load_default().await, requested)
}

/// Pure registry-or-fallback resolution, split from the env/filesystem reads
/// so it is testable without mutating process-global state.
fn resolve_model_from(
    registry: Result<ModelRegistry>,
    requested: Option<String>,
) -> Result<ResolvedModel> {
    match registry {
        Ok(registry) => registry.resolve(requested.as_deref()),
        Err(_err) if requested.is_some() => {
            let model = requested.expect("requested model checked above");
            Ok(ResolvedModel {
                alias: model.clone(),
                provider: None,
                api_id: model,
                base_url: None,
                api_key: None,
                context: 200_000,
                max_tokens: None,
            })
        }
        Err(err) => Err(err.context(
            "loading default model registry; create ~/.config/agent/models.yaml or pass --model with a raw model id",
        )),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct GcFireStats {
    index: usize,
    strategy: String,
    tokens_before: u64,
    tokens_after: u64,
    cache_invalidated: bool,
    dropped_count: Option<u64>,
}

impl GcFireStats {
    fn reclaimed(&self) -> u64 {
        self.tokens_before.saturating_sub(self.tokens_after)
    }

    fn reduction_pct(&self) -> f64 {
        if self.tokens_before == 0 {
            0.0
        } else {
            (self.reclaimed() as f64 / self.tokens_before as f64) * 100.0
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct GcStatsReport {
    fires: Vec<GcFireStats>,
}

impl GcStatsReport {
    fn from_events(events: &[Event]) -> Self {
        let fires = events
            .iter()
            .filter_map(|event| match event {
                Event::Custom { name, data, .. } if name == "gc_collect" => Some(data),
                _ => None,
            })
            .enumerate()
            .map(|(idx, data)| GcFireStats {
                index: idx + 1,
                strategy: data
                    .get("strategy")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown")
                    .to_owned(),
                tokens_before: data
                    .get("tokens_before")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0),
                tokens_after: data
                    .get("tokens_after")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0),
                cache_invalidated: data
                    .get("cache_invalidated")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
                dropped_count: data.get("dropped_count").and_then(|value| value.as_u64()),
            })
            .collect();
        Self { fires }
    }

    fn fire_count(&self) -> usize {
        self.fires.len()
    }

    fn total_reclaimed(&self) -> u64 {
        self.fires.iter().map(GcFireStats::reclaimed).sum()
    }

    fn cache_invalidation_count(&self) -> usize {
        self.fires
            .iter()
            .filter(|fire| fire.cache_invalidated)
            .count()
    }

    fn mean_reduction_pct(&self) -> Option<f64> {
        (!self.fires.is_empty()).then(|| {
            self.fires
                .iter()
                .map(GcFireStats::reduction_pct)
                .sum::<f64>()
                / self.fires.len() as f64
        })
    }

    fn median_reduction_pct(&self) -> Option<f64> {
        if self.fires.is_empty() {
            return None;
        }
        let mut reductions: Vec<f64> = self.fires.iter().map(GcFireStats::reduction_pct).collect();
        reductions.sort_by(f64::total_cmp);
        let mid = reductions.len() / 2;
        Some(if reductions.len().is_multiple_of(2) {
            (reductions[mid - 1] + reductions[mid]) / 2.0
        } else {
            reductions[mid]
        })
    }

    fn render(&self) -> String {
        if self.fires.is_empty() {
            return "GC never fired in this trace\n".to_owned();
        }

        let mut out = String::new();
        out.push_str(&format!("GC fires: {}\n", self.fire_count()));
        out.push_str(
            "\n#  strategy    tokens before -> after  reduction  cache invalidated  dropped\n",
        );
        for fire in &self.fires {
            let dropped = fire
                .dropped_count
                .map(|count| count.to_string())
                .unwrap_or_else(|| "n/a".to_owned());
            out.push_str(&format!(
                "{:<2} {:<11} {:>8} -> {:<8} {:>6.1}%   {:<17} {}\n",
                fire.index,
                fire.strategy,
                fire.tokens_before,
                fire.tokens_after,
                fire.reduction_pct(),
                fire.cache_invalidated,
                dropped
            ));
        }
        out.push_str("\nSummary:\n");
        out.push_str(&format!(
            "  total tokens reclaimed: {}\n",
            self.total_reclaimed()
        ));
        out.push_str(&format!(
            "  cache-invalidating fires: {}\n",
            self.cache_invalidation_count()
        ));
        out.push_str(&format!(
            "  mean reduction: {:.1}%\n",
            self.mean_reduction_pct().unwrap_or(0.0)
        ));
        out.push_str(&format!(
            "  median reduction: {:.1}%\n",
            self.median_reduction_pct().unwrap_or(0.0)
        ));
        out
    }
}

async fn run_gc_stats_command(trace: &Path) -> Result<()> {
    let events = TraceLogger::read_events(trace).await?;
    print!("{}", GcStatsReport::from_events(&events).render());
    Ok(())
}

async fn run_command(command: &Command) -> Result<()> {
    match command {
        Command::GcStats { trace } => run_gc_stats_command(trace).await,
        Command::IrEffect { model, visit } => {
            let machine = agent_loop_ir(agent_core::Model(model.clone()), vec![], 16);
            let hash = agent_core::program_hash(&machine.program)?;
            let site = agent_core::EffectSite {
                block: agent_core::BlockId(0),
                instruction_index: 0,
            };
            let location = agent_core::effect_location(
                hash,
                agent_core::EffectKind::Infer,
                site,
                agent_core::DynamicPath::with_visit(site, *visit),
            )?;
            println!("{}", serde_json::to_string(&location)?);
            Ok(())
        }
        #[cfg(feature = "oauth")]
        Command::Auth { command } => run_auth_command(command).await,
        #[cfg(not(feature = "oauth"))]
        _ => Err(anyhow!("this command requires the 'oauth' feature")),
    }
}

#[cfg(feature = "oauth")]
async fn run_auth_command(command: &AuthCommand) -> Result<()> {
    match command {
        AuthCommand::Login { provider } => {
            let kind = agent_oauth::OAuthProviderKind::from_name(provider)?;
            let token = agent_oauth::login(kind).await?;
            println!(
                "logged in to {}; token expires {}",
                kind.name(),
                token
                    .expires_at
                    .map(|expires_at| expires_at.to_rfc3339())
                    .unwrap_or_else(|| "unknown".into())
            );
        }
        AuthCommand::Status => {
            for status in agent_oauth::status_all().await? {
                let expires = status
                    .expires_at
                    .map(|expires_at| expires_at.to_rfc3339())
                    .unwrap_or_else(|| "unknown".into());
                let state = if status.present {
                    if status.expired {
                        "expired"
                    } else {
                        "valid"
                    }
                } else {
                    "missing"
                };
                println!(
                    "{}: {} (expires: {}, store: {})",
                    status.provider,
                    state,
                    expires,
                    status.path.display()
                );
            }
        }
    }
    Ok(())
}

async fn run_one_shot(runtime: &mut Runtime, prompt: String) -> Result<()> {
    let response = run_turn_with_status(runtime, prompt).await?;
    runtime
        .trace
        .emit(&Event::AgentDone {
            run_id: runtime.run_id.clone(),
            timestamp: Utc::now(),
        })
        .await?;
    if !runtime.debug {
        println!("{}", response.content);
    }
    Ok(())
}

/// Why a NUL-delimited prompt loop stopped reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionStop {
    /// Reader hit EOF or an explicit empty frame.
    Eof,
    /// SIGINT/SIGTERM arrived; the whole session should wind down.
    Shutdown,
}

async fn run_stdin_session(runtime: &mut Runtime) -> Result<()> {
    tracing::info!(run_id = %runtime.run_id, "starting stdin session");
    let reader = BufReader::new(tokio::io::stdin());
    run_nul_delimited_prompt_loop(runtime, reader).await?;
    emit_done(runtime).await
}

async fn run_fifo_session(runtime: &mut Runtime, path: PathBuf) -> Result<()> {
    tracing::info!(run_id = %runtime.run_id, fifo = %path.display(), "starting fifo session");
    validate_fifo_path(&path).await?;

    let mut consecutive_empty_sessions = 0_u32;
    loop {
        // Opening a FIFO for read blocks until a writer appears; keep the
        // shutdown signal selectable during that wait or the process can only
        // be stopped with SIGKILL once the signal handlers are installed.
        let mut open_options = tokio::fs::OpenOptions::new();
        open_options.read(true);
        let file = tokio::select! {
            file = open_options.open(&path) => {
                file.with_context(|| format!("opening fifo {}", path.display()))?
            }
            _ = shutdown_signal() => break,
        };
        let reader = BufReader::new(file);
        let (handled_messages, stop) = run_nul_delimited_prompt_loop(runtime, reader).await?;
        if handled_messages {
            consecutive_empty_sessions = 0;
            emit_done(runtime).await?;
        } else {
            consecutive_empty_sessions = consecutive_empty_sessions.saturating_add(1);
        }
        if stop == SessionStop::Shutdown {
            break;
        }

        let backoff = if consecutive_empty_sessions == 0 {
            Duration::from_millis(50)
        } else {
            Duration::from_millis(50 * 2_u64.pow(consecutive_empty_sessions.min(5)))
        };
        tokio::time::sleep(backoff).await;
    }
    tracing::info!(run_id = %runtime.run_id, "fifo session shutting down");
    Ok(())
}

#[cfg(unix)]
async fn validate_fifo_path(path: &Path) -> Result<()> {
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("checking fifo {}", path.display()))?;
    if !metadata.file_type().is_fifo() {
        return Err(anyhow!(
            "path {} is not a named pipe; create it with mkfifo or use stdin/--session",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
async fn validate_fifo_path(_path: &Path) -> Result<()> {
    Err(anyhow!("--fifo is only supported on Unix platforms"))
}

async fn run_nul_delimited_prompt_loop<R>(
    runtime: &mut Runtime,
    mut reader: R,
) -> Result<(bool, SessionStop)>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut handled_messages = false;
    let stop = loop {
        tokio::select! {
            frame = read_nul_frame(&mut reader) => {
                match frame? {
                    Some(message) if message.is_empty() => break SessionStop::Eof,
                    Some(message) => {
                        handled_messages = true;
                        // A failed turn (provider outage, replay divergence,
                        // context overflow) is already traced via agent_error;
                        // a long-running session survives it and waits for
                        // the next turn instead of crashing.
                        if let Err(err) = write_session_response(runtime, message).await {
                            eprintln!("turn failed: {err:#}");
                        }
                    }
                    None => break SessionStop::Eof,
                }
            }
            _ = shutdown_signal() => break SessionStop::Shutdown,
        }
    };
    Ok((handled_messages, stop))
}

async fn write_session_response(runtime: &mut Runtime, message: String) -> Result<()> {
    let response = run_turn_with_status(runtime, message).await?;
    if !runtime.debug {
        let mut stdout = tokio::io::stdout();
        stdout.write_all(response.content.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }
    Ok(())
}

async fn run_turn_with_status(
    runtime: &mut Runtime,
    message: String,
) -> Result<agent_core::Response> {
    emit_agent_start(runtime).await?;
    match run_turn(runtime, message).await {
        Ok(response) => {
            // A turn-budget stop with empty content must not look like a
            // crash: surface a clear terminal notice instead (t-1133).
            let completion = terminal_response(runtime, &response);
            emit_agent_complete(runtime, &completion).await?;
            Ok(response)
        }
        Err(err) => {
            let message = err.to_string();
            tracing::error!(run_id = %runtime.run_id, error = %message, "agent turn failed");
            if is_context_overflow_error(&message) {
                emit_context_overflow(runtime, &message).await?;
            }
            emit_agent_error(runtime, &message).await?;
            Err(err)
        }
    }
}

async fn run_turn(runtime: &mut Runtime, message: String) -> Result<agent_core::Response> {
    runtime.history.push(ChatMessage::user(message));
    let prompt = runtime.history.clone();
    let (response, mut new_history) = {
        let mut machine = agent_loop_ir(runtime.model.clone(), prompt.clone(), runtime.max_turns);
        machine.effect_visits = runtime.ir_effect_visits.clone();
        let (value, machine) = agent_core::run_ir_sequential_with_store_and_replay(
            &runtime.config,
            machine,
            &mut runtime.ir_store,
            runtime.ir_replay.as_ref(),
        )
        .await?;
        runtime.ir_effect_visits = machine.effect_visits.clone();
        let response: agent_core::Response =
            serde_json::from_value(value).context("decoding AgentIR agent loop response")?;
        let history = machine
            .env
            .get(&agent_core::Var("history".into()))
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or(prompt);
        (response, history)
    };
    if !response.content.is_empty() || !response.tool_calls.is_empty() {
        new_history.push(ChatMessage::assistant(
            (!response.content.is_empty()).then_some(response.content.clone()),
            response.tool_calls.clone(),
        ));
    }
    if response_turn_budget_exhausted(&response) {
        emit_turn_budget_exhausted(runtime, &response).await?;
    }
    // A turn that exhausted its budget mid-tool-call returns unexecuted tool
    // calls; close them with synthetic error results so the next turn is not
    // rejected by the provider's pending-tool-call guard.
    let closed = agent_core::close_pending_tool_calls(&mut new_history);
    if !closed.is_empty() {
        tracing::warn!(
            run_id = %runtime.run_id,
            closed = closed.len(),
            "closed unexecuted tool calls left by an exhausted turn budget"
        );
    }
    runtime.history = new_history;
    put_checkpoint(runtime).await?;
    save_checkpoint(runtime).await?;
    Ok(response)
}

fn response_turn_budget_exhausted(response: &agent_core::Response) -> bool {
    matches!(
        response
            .metadata
            .get("stop_reason")
            .and_then(serde_json::Value::as_str),
        Some("turn_budget_exhausted")
    )
}

/// What agent_complete should carry: the assistant text, or — when the turn
/// budget ran out with nothing left to say — a clear notice naming the limit
/// and what was pending, so budget exhaustion is distinguishable from a crash.
fn terminal_response(runtime: &Runtime, response: &agent_core::Response) -> String {
    if response_turn_budget_exhausted(response) && response.content.is_empty() {
        return turn_budget_message(runtime.max_turns, &response.tool_calls);
    }
    response.content.clone()
}

fn turn_budget_message(max_turns: usize, tool_calls: &[agent_core::ToolCall]) -> String {
    let pending = tool_calls.len();
    let summary = tool_calls
        .first()
        .map(summarize_tool_call)
        .unwrap_or_else(|| "no pending tool call".into());
    format!(
        "[turn budget exhausted after {max_turns} turns; {pending} tool call(s) left unexecuted: {summary}]"
    )
}

fn summarize_tool_call(call: &agent_core::ToolCall) -> String {
    let detail = call
        .arguments
        .get("command")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            call.arguments
                .get("prompt")
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("");
    if detail.is_empty() {
        call.name.clone()
    } else {
        format!("{} {}", call.name, detail)
    }
}

async fn emit_turn_budget_exhausted(
    runtime: &mut Runtime,
    response: &agent_core::Response,
) -> Result<()> {
    runtime
        .trace
        .emit(&Event::TurnBudgetExhausted {
            run_id: runtime.run_id.clone(),
            max_turns: runtime.max_turns,
            pending_tool_calls: response.tool_calls.len(),
            first_tool: response.tool_calls.first().map(summarize_tool_call),
            timestamp: Utc::now(),
        })
        .await
}

async fn read_nul_frame<R>(reader: &mut R) -> Result<Option<String>>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut buf = Vec::new();
    let n = reader.read_until(0, &mut buf).await?;
    if n == 0 {
        return Ok(None);
    }
    if buf.last() == Some(&0) {
        buf.pop();
    }
    Ok(Some(
        String::from_utf8(buf).context("session message was not UTF-8")?,
    ))
}

async fn emit_done(runtime: &mut Runtime) -> Result<()> {
    runtime
        .trace
        .emit(&Event::AgentDone {
            run_id: runtime.run_id.clone(),
            timestamp: Utc::now(),
        })
        .await
}

async fn emit_agent_start(runtime: &mut Runtime) -> Result<()> {
    emit_custom_event(
        runtime,
        "agent_start",
        serde_json::json!({
            "config": {
                "run_id": runtime.run_id,
                "model": runtime.model.0,
                "provider_url": runtime.provider_url,
                "trace_path": runtime.trace_path,
                "checkpoint_dir": runtime.checkpoint_dir,
            }
        }),
    )
    .await
}

async fn emit_agent_complete(runtime: &mut Runtime, response: &str) -> Result<()> {
    emit_custom_event(
        runtime,
        "agent_complete",
        serde_json::json!({ "response": response }),
    )
    .await
}

fn is_context_overflow_error(message: &str) -> bool {
    // Shared heuristic rather than the bare context_length_exceeded prefix:
    // the codex OAuth provider surfaces raw backend text ("your input
    // exceeds the context window of this model") that the prefix check
    // missed, leaving real overflows untagged in the taxonomy (t-1151).
    agent_core::is_context_overflow_message(message)
}

async fn emit_context_overflow(runtime: &mut Runtime, message: &str) -> Result<()> {
    emit_custom_event(
        runtime,
        "context_overflow",
        serde_json::json!({ "message": message }),
    )
    .await
}

async fn emit_agent_error(runtime: &mut Runtime, message: &str) -> Result<()> {
    emit_custom_event(
        runtime,
        "agent_error",
        serde_json::json!({ "message": message }),
    )
    .await
}

async fn emit_custom_event(
    runtime: &mut Runtime,
    custom_type: &str,
    data: serde_json::Value,
) -> Result<()> {
    if !runtime.debug {
        return Ok(());
    }
    let line = serde_json::json!({
        "type": "custom",
        "custom_type": custom_type,
        "data": data,
        "timestamp": Utc::now().to_rfc3339(),
    });
    let mut stdout = tokio::io::stdout();
    stdout
        .write_all(serde_json::to_string(&line)?.as_bytes())
        .await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

async fn put_checkpoint(runtime: &mut Runtime) -> Result<()> {
    let Some(path) = runtime.checkpoint_path.clone() else {
        return Ok(());
    };
    let Some(checkpoint) = checkpoint_from_runtime(runtime, runtime.checkpoint_sequence + 1) else {
        tracing::warn!(run_id = %runtime.run_id, "skipping session:state checkpoint with pending tool calls");
        return Ok(());
    };
    let value = serde_json::to_value(checkpoint)?;
    let config = SeqConfig {
        provider: runtime.config.provider.clone(),
        hydration: runtime.config.hydration.clone(),
        passive_hydration: PassiveHydrationConfig::default(),
        checkpoint_path: Some(path),
        trace: runtime.trace.clone(),
        eval: runtime.config.eval.clone(),
        replay: runtime.config.replay.clone(),
        trace_full_prompt_ir: runtime.config.trace_full_prompt_ir,
        trace_full_payloads: runtime.config.trace_full_payloads,
        gc: GcMode::None,
        gc_threshold: runtime.gc_threshold,
        gc_log: false,
        gc_timing: GcTiming::Threshold,
        context_budget: runtime.context_budget,
    };
    let _ = agent_core::run_sequential(
        &config,
        runtime.history.clone(),
        agent_core::put("session:state", value),
    )
    .await?;
    Ok(())
}

async fn save_checkpoint(runtime: &mut Runtime) -> Result<()> {
    let Some(dir) = &runtime.checkpoint_dir else {
        return Ok(());
    };
    let sequence = runtime.checkpoint_sequence + 1;
    let Some(checkpoint) = checkpoint_from_runtime(runtime, sequence) else {
        tracing::warn!(run_id = %runtime.run_id, "skipping checkpoint with pending tool calls");
        return Ok(());
    };
    runtime.checkpoint_sequence = sequence;
    tokio::fs::create_dir_all(dir).await?;
    let bytes = serde_json::to_vec_pretty(&checkpoint)?;
    let path = dir.join(format!(
        "checkpoint-{:06}-{}.json",
        checkpoint.sequence, runtime.run_id
    ));
    tokio::fs::write(&path, &bytes).await?;
    tokio::fs::write(dir.join("latest.json"), &bytes).await?;
    tokio::fs::write(dir.join("session-latest.json"), bytes).await?;
    runtime
        .trace
        .emit(&Event::Checkpoint {
            run_id: runtime.run_id.clone(),
            name: format!("checkpoint-{:06}", checkpoint.sequence),
            path: Some(path.display().to_string()),
            timestamp: Utc::now(),
        })
        .await?;
    tracing::info!(run_id = %runtime.run_id, checkpoint = %path.display(), "checkpoint saved");
    if runtime.debug {
        eprintln!("checkpoint: {}", path.display());
    }
    Ok(())
}

fn checkpoint_from_runtime(runtime: &Runtime, sequence: u64) -> Option<Checkpoint> {
    if agent_core::has_pending_tool_calls(&runtime.history) {
        return None;
    }
    Some(Checkpoint {
        run_id: runtime.run_id.clone(),
        sequence,
        model: runtime.model.0.clone(),
        provider_url: runtime.provider_url.clone(),
        messages: runtime.history.clone(),
        trace_path: runtime.trace_path.clone(),
        timestamp: Utc::now(),
    })
}

async fn load_checkpoint(path: &Path) -> Result<Checkpoint> {
    tracing::info!(checkpoint = %path.display(), "loading checkpoint");
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("reading checkpoint {}", path.display()))?;
    let mut checkpoint: Checkpoint = serde_json::from_str(&content)
        .with_context(|| format!("parsing checkpoint {}", path.display()))?;
    if agent_core::has_pending_tool_calls(&checkpoint.messages) {
        let original_len = checkpoint.messages.len();
        checkpoint.messages = agent_core::repair_trailing_pending_tool_calls(&checkpoint.messages);
        let repaired_len = checkpoint.messages.len();
        tracing::warn!(
            checkpoint = %path.display(),
            original_len,
            repaired_len,
            "repaired checkpoint with pending tool calls"
        );
        let bytes = serde_json::to_vec_pretty(&checkpoint)?;
        tokio::fs::write(path, bytes)
            .await
            .with_context(|| format!("writing repaired checkpoint {}", path.display()))?;
    }
    Ok(checkpoint)
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn base_system_prompt() -> &'static str {
    "You are a standalone agent runner. Use the shell tool when you need to inspect or change the environment. The shell tool executes command strings with the configured shell inside the current process environment. When finished, answer concisely."
}

async fn build_system_prompt(override_prompt: Option<String>) -> Result<String> {
    let base = override_prompt.unwrap_or_else(|| base_system_prompt().to_string());
    let cwd = std::env::current_dir().context("getting current directory")?;
    Ok(format!(
        "{base}\n\nCurrent date and time: {}\nCurrent working directory: {}",
        Utc::now().to_rfc3339(),
        cwd.display()
    ))
}

fn initial_history(system_prompt: String) -> Vec<ChatMessage> {
    vec![ChatMessage::system(system_prompt)]
}

fn prompt_with_optional_stdin(prompt: String) -> Result<String> {
    if std::io::stdin().is_terminal() {
        return Ok(prompt);
    }

    let mut stdin = String::new();
    std::io::stdin()
        .read_to_string(&mut stdin)
        .context("reading stdin")?;
    if stdin.trim().is_empty() {
        Ok(prompt)
    } else if prompt.trim().is_empty() {
        Ok(stdin)
    } else {
        Ok(format!("{prompt}\n\n--- Input Data ---\n{stdin}"))
    }
}

struct LocalFileSource {
    root: PathBuf,
    max_bytes: usize,
}

impl LocalFileSource {
    fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_bytes: 64 * 1024,
        }
    }
}

#[async_trait]
impl HydrationSource for LocalFileSource {
    fn name(&self) -> &str {
        "local-files"
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Knowledge
    }

    fn capabilities(&self) -> SourceCapability {
        SourceCapability::SESSION_CONTEXT | SourceCapability::WORKSPACE
    }

    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
        let mut entries = tokio::fs::read_dir(&self.root)
            .await
            .with_context(|| format!("reading hydration directory {}", self.root.display()))?;
        let max_bytes = params.max_bytes.unwrap_or(self.max_bytes);
        let mut paths = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                paths.push(entry.path());
            }
        }
        paths.sort();

        let mut remaining = max_bytes;
        let mut files = Vec::new();
        let mut included_paths = Vec::new();
        for path in paths {
            if remaining == 0 {
                break;
            }
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            let bytes = tokio::fs::read(&path)
                .await
                .with_context(|| format!("reading hydration file {}", path.display()))?;
            let take = remaining.min(bytes.len());
            let content = String::from_utf8_lossy(&bytes[..take]);
            files.push(format!("### {name}\n{content}"));
            included_paths.push(path.display().to_string());
            remaining -= take;
        }

        Ok(SourceResult {
            source: self.name().into(),
            kind: self.kind(),
            content: files.join("\n\n"),
            metadata: serde_json::json!({
                "root": self.root,
                "max_bytes": max_bytes,
                "paths": included_paths,
            }),
        })
    }
}

async fn read_config(path: Option<&PathBuf>) -> Result<FileConfig> {
    match path {
        Some(path) => {
            let content = tokio::fs::read_to_string(path)
                .await
                .with_context(|| format!("reading config {}", path.display()))?;
            Ok(toml::from_str(&content)
                .with_context(|| format!("parsing config {}", path.display()))?)
        }
        None => Ok(FileConfig::default()),
    }
}

fn trace_path(run_id: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
    Ok(home
        .join(".local/share/agent/traces")
        .join(format!("{run_id}.jsonl")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget_exhausted_response(tool_calls: Vec<agent_core::ToolCall>) -> agent_core::Response {
        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "stop_reason".into(),
            serde_json::Value::String("turn_budget_exhausted".into()),
        );
        agent_core::Response {
            content: String::new(),
            tool_calls,
            finish_reason: Some(agent_core::FinishReason::Stop),
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            metadata,
        }
    }

    #[test]
    fn turn_budget_exhaustion_is_detected_and_message_is_non_empty() {
        let call = agent_core::ToolCall::new(
            "call-1",
            "shell",
            serde_json::json!({ "command": "cargo test" }),
        );
        let response = budget_exhausted_response(vec![call]);

        assert!(response_turn_budget_exhausted(&response));
        let message = turn_budget_message(100, &response.tool_calls);
        assert!(
            message.contains("turn budget exhausted after 100 turns"),
            "got: {message}"
        );
        assert!(
            message.contains("1 tool call(s) left unexecuted"),
            "got: {message}"
        );
        assert!(message.contains("shell cargo test"), "got: {message}");
    }

    #[test]
    fn natural_responses_are_not_flagged_as_budget_exhausted() {
        let response = agent_core::Response {
            content: "done".into(),
            tool_calls: vec![],
            finish_reason: Some(agent_core::FinishReason::Stop),
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            metadata: Default::default(),
        };
        assert!(!response_turn_budget_exhausted(&response));
    }

    // Overflow taxonomy regression (t-1133): the three budget conditions stay
    // distinct — turn_budget_exhausted (soft turn ceiling), context_overflow
    // (hard provider context window), gc_collect/gc_truncate (token-budget
    // pressure inside GC).
    #[test]
    fn context_overflow_detection_still_matches_provider_sentinel() {
        assert!(is_context_overflow_error(
            "context_length_exceeded: provider returned 400: too long"
        ));
        assert!(!is_context_overflow_error("provider returned 500: boom"));
    }

    #[test]
    fn overflow_event_names_are_distinct() {
        let turn_budget = Event::TurnBudgetExhausted {
            run_id: "r".into(),
            max_turns: 100,
            pending_tool_calls: 1,
            first_tool: Some("shell cargo test".into()),
            timestamp: Utc::now(),
        };
        let custom = |name: &str| Event::Custom {
            run_id: "r".into(),
            name: name.into(),
            data: serde_json::json!({}),
            timestamp: Utc::now(),
        };

        let names = [
            turn_budget.name(),
            custom("context_overflow").name(),
            custom("gc_collect").name(),
            custom("gc_truncate").name(),
        ];
        assert_eq!(
            names,
            [
                "turn_budget_exhausted",
                "context_overflow",
                "gc_collect",
                "gc_truncate"
            ]
        );
    }

    #[test]
    fn reports_oauth_provider_base_url_instead_of_fallback() {
        assert_eq!(
            reported_provider_url(Some("openai-codex"), "https://openrouter.ai/api/v1"),
            "https://chatgpt.com/backend-api"
        );
        assert_eq!(
            reported_provider_url(Some("claude-code"), "https://openrouter.ai/api/v1"),
            "https://api.anthropic.com/v1"
        );
    }

    #[test]
    fn reports_resolved_url_for_non_oauth_provider() {
        assert_eq!(
            reported_provider_url(None, "https://api.example.test/v1"),
            "https://api.example.test/v1"
        );
    }

    #[test]
    fn raw_model_is_resolved_without_model_registry() -> Result<()> {
        // Exercise the registry-missing fallback purely — no env mutation,
        // which races under parallel test execution.
        let missing_registry = Err(anyhow!("no registry on this machine"));

        let resolved = resolve_model_from(missing_registry, Some("openrouter/auto".into()))?;

        assert_eq!(resolved.alias, "openrouter/auto");
        assert_eq!(resolved.api_id, "openrouter/auto");
        assert_eq!(resolved.provider, None);
        Ok(())
    }

    #[test]
    fn missing_registry_without_requested_model_is_an_error() {
        let err = resolve_model_from(Err(anyhow!("no registry")), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("models.yaml"), "got: {err}");
    }

    #[tokio::test]
    async fn load_checkpoint_repairs_trailing_tool_call() -> Result<()> {
        let path = std::env::temp_dir().join(format!("agent-checkpoint-{}.json", Uuid::new_v4()));
        let checkpoint = Checkpoint {
            run_id: "run".into(),
            sequence: 7,
            model: "model".into(),
            provider_url: "https://chatgpt.com/backend-api".into(),
            messages: vec![
                ChatMessage::system("system"),
                ChatMessage::user("inspect"),
                ChatMessage::assistant(
                    None,
                    vec![agent_core::ToolCall::new(
                        "call-1",
                        "shell",
                        serde_json::json!({ "command": "pwd" }),
                    )],
                ),
            ],
            trace_path: path.with_extension("jsonl"),
            timestamp: Utc::now(),
        };
        tokio::fs::write(&path, serde_json::to_vec_pretty(&checkpoint)?).await?;

        let repaired = load_checkpoint(&path).await?;

        assert!(!agent_core::has_pending_tool_calls(&repaired.messages));
        assert_eq!(repaired.messages.len(), 2);
        let on_disk: Checkpoint = serde_json::from_slice(&tokio::fs::read(&path).await?)?;
        assert_eq!(on_disk.messages.len(), 2);
        Ok(())
    }
}

#[cfg(test)]
mod gc_stats_tests {
    use super::*;
    use chrono::Utc;

    fn custom_gc(tokens_before: u64, tokens_after: u64, cache_invalidated: bool) -> Event {
        Event::Custom {
            run_id: "run".into(),
            name: "gc_collect".into(),
            data: serde_json::json!({
                "type": "gc_collect",
                "strategy": "ring",
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "cache_invalidated": cache_invalidated,
                "dropped_count": 2,
            }),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn gc_stats_computes_aggregates_from_trace_events() {
        let events = vec![
            Event::AgentDone {
                run_id: "run".into(),
                timestamp: Utc::now(),
            },
            custom_gc(100, 60, false),
            custom_gc(200, 100, true),
        ];

        let report = GcStatsReport::from_events(&events);

        assert_eq!(report.fire_count(), 2);
        assert_eq!(report.total_reclaimed(), 140);
        assert_eq!(report.cache_invalidation_count(), 1);
        assert_eq!(report.mean_reduction_pct(), Some(45.0));
        assert_eq!(report.median_reduction_pct(), Some(45.0));
        assert!(report.render().contains("GC fires: 2"));
        assert!(report.render().contains("total tokens reclaimed: 140"));
    }

    #[test]
    fn gc_stats_reports_zero_fire_case_clearly() {
        let events = vec![Event::AgentDone {
            run_id: "run".into(),
            timestamp: Utc::now(),
        }];

        let report = GcStatsReport::from_events(&events);

        assert_eq!(report.fire_count(), 0);
        assert_eq!(report.render(), "GC never fired in this trace\n");
    }
}
