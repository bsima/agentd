use agent_core::{
    agent_loop_ir, format_micro_usd, AgentIdGenerator, AnthropicConfig, AnthropicProvider,
    ChatHistory, ChatMessage, Embedder, EmbeddingClient, EnvPolicy, EvalConfig, Event, GcMode,
    GcTiming, HydrationSink, HydrationSource, InMemoryStore, IrReplayTrace, JsonlTraceSink,
    MarkSweepGc, MemorySource, ModelRegistry, OtelTraceSink, PassiveHydrationConfig, PassiveSource,
    PricingTable, ProviderClient, ProviderConfig, ReplayOnlyProvider, ResolvedModel, RingGc,
    RunUsage, SeqConfig, SourceCapability, SourceKind, SourceParams, SourceRegistry, SourceResult,
    StackFrameGc, TemporalSource, TraceContextEnv, TraceLogger,
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
#[command(version)]
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
    /// Path to a JSON Schema file constraining the final response: each
    /// turn's final answer must be a single JSON value conforming to it.
    /// Non-conforming answers get up to 2 repair turns, then the turn
    /// errors. Supported schema subset: type/required/properties/items/enum
    /// (see agent_core::output_contract).
    #[arg(long, env = "AGENT_OUTPUT_SCHEMA")]
    output_schema: Option<PathBuf>,
    /// System prompt override as literal text. Primarily for session modes
    /// (--session/--fifo), which have no prompt file whose frontmatter could
    /// carry one; takes precedence over a prompt file's `system_prompt`.
    /// Ignored on --resume (the checkpoint's history already carries the
    /// session's system message).
    #[arg(long, env = "AGENT_SYSTEM_PROMPT")]
    system_prompt: Option<String>,
    /// Soft turn ceiling per session turn (default 100). Takes precedence
    /// over prompt frontmatter `max_iterations`.
    #[arg(long, env = "AGENT_MAX_TURNS")]
    max_turns: Option<usize>,
    /// Store full PromptIR section content in traces instead of previews/hashes only.
    #[arg(long, env = "AGENT_TRACE_FULL_PROMPT_IR")]
    trace_full_prompt_ir: bool,
    /// Store full Infer prompts and Retrieve results in traces. Off by default:
    /// full prompts repeat the whole conversation per call (O(n^2) trace
    /// growth) and replay only needs recorded results. Previews are always
    /// stored. Enable when recording fixtures that need full prompts (e.g.
    /// GC eval traces).
    #[arg(long, env = "AGENT_TRACE_FULL_PAYLOADS")]
    trace_full_payloads: bool,
    /// Directory to read into passive hydration context.
    #[arg(long, env = "AGENT_HYDRATION_DIR")]
    hydration_dir: Option<PathBuf>,
    /// Memory directory: markdown files with frontmatter served to
    /// the `recall` tool / Retrieve effect via keyword retrieval, and
    /// registered as a write sink for `remember` / Store.
    #[arg(long, env = "AGENT_MEMORY_DIR")]
    memory_dir: Option<PathBuf>,
    /// Checkpoint directory of a PAST or sibling session, served as
    /// recent-turn summaries (passive context + semantic queries) for
    /// cross-session continuity. Do not point it at this session's own
    /// --checkpoint-dir: the live history already holds those turns.
    #[arg(long, env = "AGENT_TEMPORAL_DIR")]
    temporal_dir: Option<PathBuf>,
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
    /// Context GC strategy. `stack` is the default per the t-1339 strategy
    /// matrix: best replay-completion on tool chains, degrades to ring on
    /// chat-heavy windows (see docs/GC.md "Choosing a strategy").
    #[arg(long, value_enum, default_value_t = GcArg::Stack)]
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
    /// Gate the shell tool behind human approval (t-1308.10, DR-7): each
    /// shell command pauses the run until approved. The pause persists as a
    /// pending-approval record plus a mid-turn machine checkpoint under
    /// ~/.local/share/agent/approvals; resolve and resume it — in this or
    /// any later process — with `agent approvals --approve/--deny`.
    #[arg(long, env = "AGENT_REQUIRE_SHELL_APPROVAL")]
    require_shell_approval: bool,
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
    /// Print per-model/per-run token usage and USD cost rollups from a
    /// trace JSONL file (t-1334). Costs sum the integer micro-USD recorded
    /// on each InferResult; infers recorded without pricing are counted as
    /// "uncosted" rather than priced retroactively.
    Cost {
        #[arg(long)]
        trace: PathBuf,
        /// Emit machine-readable JSON instead of the table.
        #[arg(long)]
        json: bool,
    },
    /// Inspect and resolve pending approval gates (t-1308.10, DR-7). A
    /// paused run persists one record + one machine checkpoint per gated
    /// effect under ~/.local/share/agent/approvals; `--approve` records the
    /// decision, re-enters the checkpoint, executes the effect exactly
    /// once, and drives the run to completion (or its next pause); `--deny`
    /// records the denial and re-enters the checkpoint so the program
    /// continues with a typed denial value the model can react to. Both
    /// work after a full process restart — the filesystem is the API.
    Approvals {
        /// List pending and resolved approvals, oldest first.
        #[arg(long)]
        list: bool,
        /// With --list: emit the records as JSON.
        #[arg(long)]
        json: bool,
        /// Approve this pending effect and resume its run.
        #[arg(long, value_name = "PENDING_ID", conflicts_with_all = ["list", "deny"])]
        approve: Option<String>,
        /// Deny this pending effect and resume its run (the effect fails as
        /// a value; the program continues).
        #[arg(long, value_name = "PENDING_ID", conflicts_with = "list")]
        deny: Option<String>,
        /// Who resolved it; recorded on the record and the trace event.
        #[arg(long)]
        by: Option<String>,
        /// Optional reason, recorded and carried on the denial value.
        #[arg(long)]
        reason: Option<String>,
    },
    /// Print the AgentIR effect-location JSON for the entry Infer of the
    /// built-in agent loop. Eval scripts use it to build replay fixtures
    /// without hardcoding program hashes.
    #[command(hide = true)]
    IrEffect {
        #[arg(long)]
        model: String,
        /// Visit ordinal for the entry Infer (the Nth turn of a session,
        /// 0-based). Entry effects run before any block transition, so their
        /// control path is the root; visits along non-root paths (e.g. the
        /// within-turn nudge retry) cannot be computed here — record them.
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
    Login {
        provider: String,
    },
    /// Import credentials from an external CLI's session (codex only:
    /// reads `$CODEX_HOME/auth.json` / `~/.codex/auth.json` written by
    /// `codex login`).
    Import {
        provider: String,
    },
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
    /// The provider context ceiling catch-overflow discovered (t-1151);
    /// absent in checkpoints written before t-1162.
    #[serde(default)]
    discovered_budget: Option<usize>,
}

/// A parsed session turn frame (t-1308.2; docs/SUPERVISOR.md "Turn envelope").
///
/// Classification happens at the frame boundary: a NUL-framed turn that
/// parses as a JSON object with `"v": 1` and a string `"input"` field is a
/// turn envelope; any other frame is a raw prompt, byte-for-byte. A raw
/// prompt that happens to be a valid v1 envelope must be wrapped by the
/// caller (`{"v":1,"input":"<that text>"}`) to be sent literally.
#[derive(Debug, Clone, PartialEq)]
struct TurnFrame {
    /// Caller-supplied correlation id, echoed on the turn's machine events.
    /// `None` means the agent mints one (see `mint_turn_id`).
    turn_id: Option<String>,
    /// The prompt text delivered to the agent loop.
    input: String,
    /// Opaque caller data, echoed verbatim on the turn's `agent_complete`.
    metadata: Option<serde_json::Value>,
}

impl TurnFrame {
    fn raw(input: String) -> Self {
        Self {
            turn_id: None,
            input,
            metadata: None,
        }
    }
}

fn parse_turn_frame(frame: &str) -> TurnFrame {
    let Ok(serde_json::Value::Object(mut object)) = serde_json::from_str(frame) else {
        return TurnFrame::raw(frame.to_owned());
    };
    // `"v"` must be exactly the integer 1: strings, floats, and future
    // versions all fall through to raw-prompt compatibility.
    if object.get("v").and_then(serde_json::Value::as_u64) != Some(1) {
        return TurnFrame::raw(frame.to_owned());
    }
    let Some(serde_json::Value::String(input)) = object.remove("input") else {
        return TurnFrame::raw(frame.to_owned());
    };
    TurnFrame {
        // Classification depends only on `v` and `input`; a malformed
        // (non-string) turn_id is ignored and an id is minted instead of
        // demoting the whole frame to a raw prompt.
        turn_id: object
            .get("turn_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        input,
        metadata: object.remove("metadata"),
    }
}

/// Deterministic per-run turn id for frames that don't supply one:
/// `<run_id>-t<seq>`, where seq is the 0-based turn ordinal within the run.
/// Resumed sessions continue the ordinal from the checkpoint sequence, so a
/// resumed run does not re-mint ids already used by completed turns.
fn mint_turn_id(run_id: &str, seq: u64) -> String {
    format!("{run_id}-t{seq}")
}

struct Runtime {
    config: SeqConfig,
    trace: TraceLogger,
    run_id: String,
    model: agent_core::Model,
    provider_url: String,
    trace_path: PathBuf,
    checkpoint_dir: Option<PathBuf>,
    checkpoint_sequence: u64,
    /// 0-based ordinal of the next turn, used to mint turn ids for frames
    /// that don't carry one (t-1308.2). Seeded from the checkpoint sequence
    /// so resumed sessions keep minting fresh ids; increments on every turn,
    /// supplied id or not, so minted ids always name the turn's ordinal.
    turn_seq: u64,
    history: Vec<ChatMessage>,
    debug: bool,
    max_turns: usize,
    ir_store: InMemoryStore,
    ir_replay: Option<IrReplayTrace>,
    ir_effect_visits: BTreeMap<String, u64>,
    /// Structured final-output contract (t-1308.4): validation with bounded
    /// repair applies to each turn's final response, one-shot and session.
    output_contract: Option<agent_core::OutputContract>,
    /// Session-lived GC state (t-1162): the discovered provider ceiling
    /// (catch-overflow), frame lifecycles, and the every-N cadence survive
    /// across turns instead of being relearned per turn.
    gc_state: agent_core::GcState,
    /// Gate the shell tool's Eval behind the approval protocol (t-1308.10).
    shell_requires_approval: bool,
    /// Driver-owned facts persisted onto every pending-approval record
    /// (`PendingEffectRecord.runtime`) so `agent approvals` can rebuild
    /// this runtime after a full process restart.
    resume_facts: ResumeFacts,
}

/// What `agent approvals` needs to rebuild the runtime that paused: the
/// model alias (registry re-resolution recovers provider/url/pricing; the
/// API key is re-read from flags/env, never persisted), the run's trace
/// path (the resume appends to the same trace), and the loop/eval policies
/// that shape the program and its effects. Serialized as
/// `PendingEffectRecord.runtime` — opaque to agent-core.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResumeFacts {
    model: String,
    trace_path: PathBuf,
    max_turns: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    memory_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hydration_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temporal_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output_schema: Option<PathBuf>,
    #[serde(default)]
    memory_tools: bool,
    #[serde(default)]
    shell_requires_approval: bool,
    eval_timeout_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eval_cwd: Option<PathBuf>,
    eval_env: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eval_max_output_bytes: Option<usize>,
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
    let max_turns = args
        .max_turns
        .or_else(|| frontmatter.and_then(|meta| meta.max_iterations))
        .unwrap_or(DEFAULT_MAX_TURNS);
    let system_prompt_override = match (args.system_prompt.clone(), loaded_prompt.as_ref()) {
        // The explicit flag wins: it is how flag-only launches (sessions
        // spawned by agent-sdk or a supervisor) express instructions that
        // one-shot runs carry in prompt-file frontmatter.
        (Some(text), _) => Some(text),
        (None, Some(prompt)) => {
            frontmatter::resolve_system_prompt(
                &prompt.base_dir,
                frontmatter.and_then(|meta| meta.system_prompt.as_deref()),
            )
            .await?
        }
        (None, None) => None,
    };
    let system_prompt = build_system_prompt(system_prompt_override).await?;

    let (resolved_model, pricing_table, embedder) =
        resolve_model(requested_model, provider_file.model.clone()).await?;
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
    let replay_enabled = args.replay_trace.is_some();
    let ir_replay = match args.replay_trace.as_ref() {
        Some(path) => Some(IrReplayTrace::load(path).await?),
        None => None,
    };
    let output_contract = match args.output_schema.as_ref() {
        Some(path) => Some(load_output_contract(path).await?),
        None => None,
    };
    let context_budget = resolved_model.context;
    let model = resolved_model.api_id.clone();
    let (provider, reported_provider_url) = build_provider(
        &resolved_model,
        requested_provider.or(provider_file.url),
        args.key.clone().or(provider_file.api_key),
        replay_enabled,
    )?;

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
    let hydration = {
        let mut registry = SourceRegistry::new();
        if let Some(path) = args.hydration_dir.as_ref() {
            registry = registry.register(LocalFileSource::new(path.clone()));
        }
        if let Some(path) = args.memory_dir.as_ref() {
            // Both halves: source for retrieval, sink for the Store effect.
            // The embedder is the registry's optional `embeddings` config;
            // None = keyword-only retrieval (t-1340).
            registry = registry
                .register_backend(MemorySource::new(path.clone()).with_embedder(embedder.clone()));
        }
        if let Some(path) = args.temporal_dir.as_ref() {
            registry = registry.register(TemporalSource::new(path.clone()));
        }
        registry
    };
    let (history, checkpoint_sequence, resumed_discovered_budget) = match checkpoint {
        Some(cp) => (cp.messages, cp.sequence, cp.discovered_budget),
        None => (initial_history(system_prompt), 0, None),
    };
    let config = SeqConfig {
        // No in-process approval hook in the CLI: gated effects pause
        // durably and resolve via `agent approvals` (t-1308.10). Resume
        // drivers load resolutions into this config before re-entering.
        approvals: Default::default(),
        tools: Default::default(),
        provider,
        hydration: hydration.clone(),
        passive_hydration: PassiveHydrationConfig::with_sources([
            PassiveSource::TemporalHistory,
            PassiveSource::SessionContext,
        ]),
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
        pricing: pricing_table,
    };
    if !config.gc.enabled() && args.gc_timing != GcTiming::Threshold {
        return Err(anyhow!(
            "--gc-timing {} requires a GC strategy; pass --gc stack, --gc ring, or --gc mark-sweep",
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
        checkpoint_sequence,
        turn_seq: checkpoint_sequence,
        history,
        debug: args.debug,
        max_turns,
        ir_store: InMemoryStore::new(),
        ir_replay,
        ir_effect_visits: BTreeMap::new(),
        output_contract,
        gc_state: agent_core::GcState {
            // The discovered ceiling is knowledge about the provider, not
            // the process: a resumed session keeps it.
            discovered_budget: resumed_discovered_budget,
            ..Default::default()
        },
        shell_requires_approval: args.require_shell_approval,
        resume_facts: ResumeFacts {
            model: resolved_model.alias.clone(),
            trace_path: trace_path.clone(),
            max_turns,
            memory_dir: args.memory_dir.clone(),
            hydration_dir: args.hydration_dir.clone(),
            temporal_dir: args.temporal_dir.clone(),
            output_schema: args.output_schema.clone(),
            memory_tools: args.memory_dir.is_some(),
            shell_requires_approval: args.require_shell_approval,
            eval_timeout_seconds: args.eval_timeout_seconds,
            eval_cwd: args.eval_cwd.clone(),
            eval_env: match args.eval_env {
                EvalEnvMode::Inherit => "inherit".into(),
                EvalEnvMode::InheritFull => "inherit-full".into(),
                EvalEnvMode::Clean => "clean".into(),
            },
            eval_max_output_bytes: args.eval_max_output_bytes,
        },
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

/// Construct the chat provider for a resolved model, mirroring the CLI's
/// conventions (flag/config overrides, then env fallbacks, then defaults).
/// Shared by the main run path and the `agent approvals` resume path
/// (t-1308.10), which must rebuild the exact same provider after a full
/// process restart. Returns the provider plus the reported provider URL.
fn build_provider(
    resolved: &ResolvedModel,
    url_override: Option<String>,
    key_override: Option<String>,
    replay_enabled: bool,
) -> Result<(Arc<dyn agent_core::ChatProvider>, String)> {
    let provider_tag = resolved.provider.as_deref();
    let is_anthropic_provider = provider_tag == Some("anthropic");
    let oauth_provider = provider_tag.filter(|provider| is_oauth_provider_tag(provider));
    let url = url_override
        .or(resolved.base_url.clone())
        .or_else(|| std::env::var("AGENT_PROVIDER").ok())
        .or_else(|| std::env::var("OPENROUTER_BASE_URL").ok())
        .unwrap_or_else(|| {
            if is_anthropic_provider {
                "https://api.anthropic.com/v1".into()
            } else {
                "https://openrouter.ai/api/v1".into()
            }
        });
    let reported = reported_provider_url(oauth_provider, &url);
    #[cfg(not(feature = "oauth"))]
    if !replay_enabled {
        if let Some(provider) = oauth_provider {
            return Err(anyhow!(
                "model '{}' requires OAuth provider '{provider}', but this agent was built without the 'oauth' feature",
                resolved.alias
            ));
        }
    }
    let api_key = if oauth_provider.is_some() || replay_enabled {
        None
    } else {
        Some(
            key_override
                .or(resolved.api_key.clone())
                .or_else(|| std::env::var("AGENT_API_KEY").ok())
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
                .ok_or_else(|| {
                    anyhow!("missing API key: pass --key, set AGENT_API_KEY/ANTHROPIC_API_KEY/OPENROUTER_API_KEY, or configure api_key in models.yaml")
                })?,
        )
    };
    let provider: Arc<dyn agent_core::ChatProvider> = if replay_enabled {
        Arc::new(ReplayOnlyProvider)
    } else {
        match oauth_provider {
            Some(tag) => {
                #[cfg(feature = "oauth")]
                {
                    agent_oauth::provider_for_tag(tag, agent_core::Model(resolved.api_id.clone()))?
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
                model: agent_core::Model(resolved.api_id.clone()),
                max_tokens: resolved.max_tokens,
            })),
            None => Arc::new(ProviderClient::new(ProviderConfig {
                url: url.clone(),
                api_key: api_key.expect("api_key is set for non-OAuth providers"),
                model: agent_core::Model(resolved.api_id.clone()),
            })),
        }
    };
    Ok((provider, reported))
}

/// What model resolution yields: the chat model, the registry's pricing
/// table (t-1334), and the optional memory embedder (t-1340; `None` =
/// keyword-only retrieval).
type ModelResolution = (ResolvedModel, PricingTable, Option<Arc<dyn Embedder>>);

async fn resolve_model(
    args_model: Option<String>,
    file_model: Option<String>,
) -> Result<ModelResolution> {
    let requested = args_model
        .or(file_model)
        .or_else(|| std::env::var("AGENT_MODEL").ok());
    resolve_model_from(ModelRegistry::load_default().await, requested)
}

/// Pure registry-or-fallback resolution, split from the env/filesystem reads
/// so it is testable without mutating process-global state. Also returns the
/// registry's pricing table (t-1334) so cost accounting covers dynamic
/// (tool-supplied) model ids, not just the resolved one; the no-registry
/// fallback has no pricing (usage recorded, cost omitted).
fn resolve_model_from(
    registry: Result<ModelRegistry>,
    requested: Option<String>,
) -> Result<ModelResolution> {
    match registry {
        Ok(registry) => {
            // The optional `embeddings` section (t-1340): absent = None
            // (keyword-only memory retrieval); an invalid section — unknown
            // alias, no base_url — fails the run here, at config load.
            let embedder = EmbeddingClient::from_registry(&registry)?
                .map(|client| Arc::new(client) as Arc<dyn Embedder>);
            Ok((
                registry.resolve(requested.as_deref())?,
                registry.pricing_table()?,
                embedder,
            ))
        }
        Err(_err) if requested.is_some() => {
            let model = requested.expect("requested model checked above");
            Ok((
                ResolvedModel {
                    alias: model.clone(),
                    provider: None,
                    api_id: model,
                    base_url: None,
                    api_key: None,
                    context: 200_000,
                    max_tokens: None,
                    pricing: None,
                },
                PricingTable::default(),
                None,
            ))
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
    // t-1151 fields; absent in traces recorded before timing strategies.
    timing: Option<String>,
    target_budget: Option<u64>,
    trigger: Option<String>,
    cycle: Option<u64>,
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
                timing: data
                    .get("timing")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned),
                target_budget: data.get("target_budget").and_then(|value| value.as_u64()),
                trigger: data
                    .get("trigger")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned),
                cycle: data.get("cycle").and_then(|value| value.as_u64()),
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
            "\n#  strategy    timing          tokens before -> after  reduction  cache invalidated  dropped\n",
        );
        for fire in &self.fires {
            let dropped = fire
                .dropped_count
                .map(|count| count.to_string())
                .unwrap_or_else(|| "n/a".to_owned());
            let overflow = match (&fire.trigger, fire.cycle, fire.target_budget) {
                (Some(trigger), cycle, target) if trigger == "context_overflow" => format!(
                    "  [overflow cycle {} -> target {}]",
                    cycle.map_or_else(|| "?".into(), |cycle| cycle.to_string()),
                    target.map_or_else(|| "?".into(), |target| target.to_string()),
                ),
                _ => String::new(),
            };
            out.push_str(&format!(
                "{:<2} {:<11} {:<15} {:>8} -> {:<8} {:>6.1}%   {:<17} {}{}\n",
                fire.index,
                fire.strategy,
                fire.timing.as_deref().unwrap_or("n/a"),
                fire.tokens_before,
                fire.tokens_after,
                fire.reduction_pct(),
                fire.cache_invalidated,
                dropped,
                overflow
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
        let by_timing = self.fires_by_timing();
        if !by_timing.is_empty() {
            let breakdown = by_timing
                .iter()
                .map(|(timing, count)| format!("{timing}={count}"))
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!("  fires by timing: {breakdown}\n"));
        }
        if let Some(recovery) = self.overflow_recovery_summary() {
            out.push_str(&format!("  overflow recoveries: {recovery}\n"));
        }
        out
    }

    fn fires_by_timing(&self) -> BTreeMap<String, usize> {
        let mut by_timing = BTreeMap::new();
        for fire in &self.fires {
            // Old traces predate the timing field; bucket them visibly
            // rather than silently inventing a mode.
            let timing = fire.timing.clone().unwrap_or_else(|| "unreported".into());
            *by_timing.entry(timing).or_insert(0) += 1;
        }
        by_timing
    }

    /// One line describing the catch-overflow story of this trace: how many
    /// reactive collections fired, the deepest retry cycle reached, and the
    /// budget the shrinking converged to (the provider's real ceiling as
    /// discovered, which is what you tune context_budget against).
    fn overflow_recovery_summary(&self) -> Option<String> {
        let overflow_fires: Vec<&GcFireStats> = self
            .fires
            .iter()
            .filter(|fire| fire.trigger.as_deref() == Some("context_overflow"))
            .collect();
        if overflow_fires.is_empty() {
            return None;
        }
        let deepest_cycle = overflow_fires.iter().filter_map(|fire| fire.cycle).max();
        let final_target = overflow_fires
            .iter()
            .rev()
            .find_map(|fire| fire.target_budget);
        Some(format!(
            "{} fire(s), deepest cycle {}, final target budget {}",
            overflow_fires.len(),
            deepest_cycle.map_or_else(|| "?".into(), |cycle| cycle.to_string()),
            final_target.map_or_else(|| "?".into(), |target| target.to_string()),
        ))
    }
}

async fn run_gc_stats_command(trace: &Path) -> Result<()> {
    let events = TraceLogger::read_events(trace).await?;
    print!("{}", GcStatsReport::from_events(&events).render());
    print!("{}", CalibrationReport::from_events(&events).render());
    Ok(())
}

/// Cost rollup over a trace (t-1334): per-run and per-model token/cost
/// sums built purely from the recorded `InferResult` fields — the exact
/// integer micro-USD values stamped at emission time. Nothing here consults
/// models.yaml, so the same trace always rolls up to the same totals,
/// including replayed traces.
#[derive(Debug, Clone, Default, PartialEq)]
struct CostReport {
    runs: Vec<RunCostReport>,
    total: RunUsage,
}

#[derive(Debug, Clone, PartialEq)]
struct RunCostReport {
    run_id: String,
    /// Keyed by the model recorded on the paired InferCall (joined via
    /// op_id); InferResults whose call is missing from the trace bucket
    /// under "(unknown)".
    per_model: BTreeMap<String, RunUsage>,
    total: RunUsage,
}

impl CostReport {
    fn from_events(events: &[Event]) -> Self {
        // (run_id, op_id) -> model, from the InferCall side of each pair.
        let mut model_by_op: BTreeMap<(String, u64), String> = BTreeMap::new();
        for event in events {
            if let Event::InferCall {
                run_id,
                op_id,
                model,
                ..
            } = event
            {
                model_by_op.insert((run_id.clone(), *op_id), model.clone());
            }
        }
        let mut report = Self::default();
        let mut run_index: BTreeMap<String, usize> = BTreeMap::new();
        for event in events {
            // Successful calls carry usage/cost; failed attempts
            // (InferError, t-1347) carry none — the provider error path
            // returns no Response — so they are counted, never summed.
            let (run_id, op_id, outcome) = match event {
                Event::InferResult {
                    run_id,
                    op_id,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cached_input_tokens,
                    cost_micro_usd,
                    ..
                } => (
                    run_id,
                    op_id,
                    Some((
                        *input_tokens,
                        *output_tokens,
                        *total_tokens,
                        *cached_input_tokens,
                        *cost_micro_usd,
                    )),
                ),
                Event::InferError { run_id, op_id, .. } => (run_id, op_id, None),
                _ => continue,
            };
            let index = *run_index.entry(run_id.clone()).or_insert_with(|| {
                report.runs.push(RunCostReport {
                    run_id: run_id.clone(),
                    per_model: BTreeMap::new(),
                    total: RunUsage::default(),
                });
                report.runs.len() - 1
            });
            let run = &mut report.runs[index];
            let model = model_by_op
                .get(&(run_id.clone(), *op_id))
                .cloned()
                .unwrap_or_else(|| "(unknown)".into());
            for usage in [
                run.per_model.entry(model).or_default(),
                &mut run.total,
                &mut report.total,
            ] {
                match outcome {
                    Some((input, output, total, cached, cost)) => {
                        usage.observe_infer(input, output, total, cached, cost);
                    }
                    None => usage.observe_infer_error(),
                }
            }
        }
        report
    }

    fn render(&self) -> String {
        if self.runs.is_empty() {
            return "no InferResult/InferError events in this trace\n".to_owned();
        }
        let mut out = String::new();
        for run in &self.runs {
            // The failed column appears only when the run has failed
            // attempts (t-1347), so all-success tables keep their shape.
            let show_failed = run.total.failed_infer_calls > 0;
            out.push_str(&format!("Run {}\n", run.run_id));
            out.push_str(&format!(
                "  {:<40} {:>7} {:>12} {:>12} {:>10} {:>14}",
                "model", "infers", "input tok", "output tok", "cached", "cost (USD)"
            ));
            if show_failed {
                out.push_str(&format!(" {:>7}", "failed"));
            }
            out.push('\n');
            for (model, usage) in &run.per_model {
                out.push_str(&render_usage_row(model, usage, show_failed));
            }
            out.push_str(&render_usage_row("run total", &run.total, show_failed));
            if run.total.uncosted_infer_calls > 0 {
                out.push_str(&format!(
                    "  uncosted infer calls: {} (no pricing recorded; cost total is partial)\n",
                    run.total.uncosted_infer_calls
                ));
            }
            if show_failed {
                out.push_str(&format!(
                    "  failed infer calls: {} (no usage recorded; attempts only)\n",
                    run.total.failed_infer_calls
                ));
            }
            out.push('\n');
        }
        if self.runs.len() > 1 {
            out.push_str(&format!(
                "Total: {} infer(s), {} input + {} output = {} tokens, cost {}{}{}\n",
                self.total.infer_calls,
                self.total.input_tokens,
                self.total.output_tokens,
                self.total.total_tokens,
                self.total
                    .cost_micro_usd
                    .map(format_micro_usd)
                    .unwrap_or_else(|| "n/a".into()),
                if self.total.uncosted_infer_calls > 0 {
                    format!(" ({} uncosted)", self.total.uncosted_infer_calls)
                } else {
                    String::new()
                },
                if self.total.failed_infer_calls > 0 {
                    format!(" ({} failed)", self.total.failed_infer_calls)
                } else {
                    String::new()
                },
            ));
        }
        out
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "runs": self
                .runs
                .iter()
                .map(|run| {
                    serde_json::json!({
                        "run_id": run.run_id,
                        "models": run
                            .per_model
                            .iter()
                            .map(|(model, usage)| (model.clone(), usage_json(usage)))
                            .collect::<serde_json::Map<String, serde_json::Value>>(),
                        "total": usage_json(&run.total),
                    })
                })
                .collect::<Vec<_>>(),
            "total": usage_json(&self.total),
        })
    }
}

fn render_usage_row(label: &str, usage: &RunUsage, show_failed: bool) -> String {
    let mut row = format!(
        "  {:<40} {:>7} {:>12} {:>12} {:>10} {:>14}",
        label,
        usage.infer_calls,
        usage.input_tokens,
        usage.output_tokens,
        usage
            .cached_input_tokens
            .map(|cached| cached.to_string())
            .unwrap_or_else(|| "n/a".into()),
        usage
            .cost_micro_usd
            .map(format_micro_usd)
            .unwrap_or_else(|| "n/a".into()),
    );
    if show_failed {
        row.push_str(&format!(" {:>7}", usage.failed_infer_calls));
    }
    row.push('\n');
    row
}

/// JSON rendering of a rollup: the canonical integer `cost_micro_usd` plus
/// a derived exact-decimal `cost_usd` string for humans/spreadsheets.
fn usage_json(usage: &RunUsage) -> serde_json::Value {
    let mut value = serde_json::to_value(usage).expect("RunUsage serializes");
    if let (Some(cost), Some(object)) = (usage.cost_micro_usd, value.as_object_mut()) {
        object.insert(
            "cost_usd".into(),
            serde_json::Value::String(format_micro_usd(cost).trim_start_matches('$').to_owned()),
        );
    }
    value
}

async fn run_cost_command(trace: &Path, json: bool) -> Result<()> {
    // read_events fails on any malformed JSONL line, which exits nonzero:
    // a broken trace must never roll up to a silently-wrong total.
    let events = TraceLogger::read_events(trace)
        .await
        .with_context(|| format!("reading trace {}", trace.display()))?;
    let report = CostReport::from_events(&events);
    if json {
        println!("{}", serde_json::to_string_pretty(&report.to_json())?);
    } else {
        print!("{}", report.render());
    }
    Ok(())
}

/// `agent approvals` (t-1308.10, DR-7): list pending/resolved approval
/// records, or resolve one and resume its run from the persisted mid-turn
/// checkpoint. Everything flows through the flat approvals directory
/// (`~/.local/share/agent/approvals`) — the filesystem is the API, so
/// resolution works from any process, including after a full restart of
/// the machine that paused.
async fn run_approvals_command(
    list: bool,
    json: bool,
    approve: Option<&str>,
    deny: Option<&str>,
    by: Option<String>,
    reason: Option<String>,
) -> Result<()> {
    let store = agent_core::ApprovalStore::new(agent_core::ApprovalStore::default_dir()?);
    match (list, approve, deny) {
        (true, None, None) => {
            let records = store.list().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&records)?);
                return Ok(());
            }
            if records.is_empty() {
                println!("no pending approvals under {}", store.dir().display());
                return Ok(());
            }
            println!(
                "PENDING_ID       STATUS             KIND   RUN_ID                                 REQUEST"
            );
            for record in records {
                println!(
                    "{:<16} {:<18} {:<6} {:<38} {}",
                    record.pending_id,
                    match record.status {
                        agent_core::PendingStatus::AwaitingApproval => "awaiting_approval",
                        agent_core::PendingStatus::Approved => "approved",
                        agent_core::PendingStatus::Denied => "denied",
                    },
                    record.kind.as_str(),
                    record.run_id,
                    record.request
                );
            }
            Ok(())
        }
        (false, Some(pending_id), None) => {
            resolve_and_resume(
                &store,
                pending_id,
                agent_core::ApprovalDecision::Approve,
                by,
                reason,
            )
            .await
        }
        (false, None, Some(pending_id)) => {
            resolve_and_resume(
                &store,
                pending_id,
                agent_core::ApprovalDecision::Deny,
                by,
                reason,
            )
            .await
        }
        _ => Err(anyhow!(
            "pass exactly one of --list, --approve <pending_id>, or --deny <pending_id>"
        )),
    }
}

/// Record the decision (exactly once), claim the mid-turn checkpoint
/// (atomically — a second resume attempt fails instead of re-executing the
/// effect), rebuild the paused runtime from the record's persisted facts,
/// and drive the run to completion or to its next pause. A record already
/// resolved with the SAME decision (e.g. via the SDK's
/// `PendingApproval::approve`) skips straight to the resume; a
/// contradicting decision is refused.
async fn resolve_and_resume(
    store: &agent_core::ApprovalStore,
    pending_id: &str,
    decision: agent_core::ApprovalDecision,
    by: Option<String>,
    reason: Option<String>,
) -> Result<()> {
    let record = store.load(pending_id).await?;
    let record = if record.is_awaiting() {
        store
            .resolve(
                pending_id,
                decision,
                Some(by.unwrap_or_else(|| "agent-approvals".into())),
                reason,
            )
            .await?
    } else {
        let recorded = agent_core::ApprovalStore::resolution_of(&record)?;
        if recorded.decision != decision {
            return Err(anyhow!(
                "pending approval {pending_id} is already resolved as {}; refusing to overturn a made decision",
                recorded.decision.as_status_str()
            ));
        }
        record
    };
    let resolution = agent_core::ApprovalStore::resolution_of(&record)?;
    let checkpoint = store.claim_checkpoint(pending_id).await?;
    let facts: ResumeFacts = serde_json::from_value(record.runtime.clone().ok_or_else(|| {
        anyhow!(
            "pending approval {pending_id} carries no runtime facts; it was not persisted by \
             this CLI and cannot be resumed here"
        )
    })?)
    .with_context(|| format!("parsing runtime facts of pending approval {pending_id}"))?;
    resume_run(store, &record, resolution, checkpoint, facts).await
}

/// Re-enter a claimed checkpoint: same model resolution, provider
/// construction, hydration registry, and eval policies as the run that
/// paused (rebuilt from [`ResumeFacts`]; the API key comes from the
/// environment, never from disk), with the decision pre-loaded so the gate
/// consumes it at the effect site. Trace events append to the run's
/// original trace file under its original run id. The resumed turn's
/// final response prints to stdout; it is not folded back into a session
/// checkpoint (wave-1 limitation, see docs/TRACE_SCHEMA.md 1.4 notes).
async fn resume_run(
    store: &agent_core::ApprovalStore,
    record: &agent_core::PendingEffectRecord,
    resolution: agent_core::ApprovalResolution,
    checkpoint: agent_core::IrCheckpoint,
    facts: ResumeFacts,
) -> Result<()> {
    let (resolved_model, pricing_table, embedder) = resolve_model_from(
        ModelRegistry::load_default().await,
        Some(facts.model.clone()),
    )?;
    let (provider, _provider_url) = build_provider(&resolved_model, None, None, false)?;
    let trace = TraceLogger::new(record.run_id.clone(), facts.trace_path.clone());
    let hydration = {
        let mut registry = SourceRegistry::new();
        if let Some(path) = facts.hydration_dir.as_ref() {
            registry = registry.register(LocalFileSource::new(path.clone()));
        }
        if let Some(path) = facts.memory_dir.as_ref() {
            registry = registry
                .register_backend(MemorySource::new(path.clone()).with_embedder(embedder.clone()));
        }
        if let Some(path) = facts.temporal_dir.as_ref() {
            registry = registry.register(TemporalSource::new(path.clone()));
        }
        registry
    };
    let output_contract = match facts.output_schema.as_ref() {
        Some(path) => Some(load_output_contract(path).await?),
        None => None,
    };
    let mut approvals = agent_core::ApprovalConfig::default();
    approvals
        .resolutions
        .insert(record.effect_id.clone(), resolution);
    let config = SeqConfig {
        approvals,
        tools: Default::default(),
        provider,
        hydration,
        passive_hydration: PassiveHydrationConfig::with_sources([
            PassiveSource::TemporalHistory,
            PassiveSource::SessionContext,
        ]),
        trace: trace.clone(),
        eval: EvalConfig {
            cwd: facts.eval_cwd.clone(),
            timeout: Duration::from_secs(facts.eval_timeout_seconds),
            max_stdout_bytes: facts.eval_max_output_bytes.unwrap_or(1024 * 1024),
            max_stderr_bytes: facts.eval_max_output_bytes.unwrap_or(1024 * 1024),
            env: match facts.eval_env.as_str() {
                "inherit-full" => EnvPolicy::InheritFull,
                "clean" => EnvPolicy::Clean {
                    vars: Default::default(),
                },
                _ => EnvPolicy::Inherit,
            },
            ..EvalConfig::default()
        },
        replay: None,
        trace_full_prompt_ir: false,
        trace_full_payloads: false,
        // Mirrors the CLI defaults (--gc stack, --gc-cache preserve): the
        // resumed run should collect the way a fresh run would.
        gc: GcMode::Stack(StackFrameGc {
            preserve_prefix: true,
        }),
        gc_threshold: 0.85,
        gc_log: false,
        gc_timing: GcTiming::Threshold,
        context_budget: resolved_model.context,
        pricing: pricing_table,
    };
    let options = agent_core::AgentLoopOptions {
        memory_tools: facts.memory_tools,
        tool_names: vec![],
        output_contract,
        shell_requires_approval: facts.shell_requires_approval,
        infer_system_prompt: None,
    };
    let mut ir_store = checkpoint.store.clone();
    let mut gc_state = agent_core::GcState::default();
    eprintln!(
        "resuming run {} at effect {} ({})",
        record.run_id,
        record.effect_id,
        resolution_status(record)
    );
    match agent_core::resume_agent_loop_outcome(
        &config,
        &mut ir_store,
        &mut gc_state,
        agent_core::Model(resolved_model.api_id.clone()),
        facts.max_turns,
        &options,
        checkpoint.machine,
    )
    .await?
    {
        agent_core::AgentLoopOutcome::Complete { value, .. } => {
            if let Some(failure) = agent_core::output_contract_failure(&value) {
                return Err(anyhow!(
                    "final output failed the output schema after {} attempt(s): {}",
                    failure.attempts,
                    failure.errors.join("; ")
                ));
            }
            let response: agent_core::Response =
                serde_json::from_value(value).context("decoding AgentIR agent loop response")?;
            trace
                .emit(&Event::AgentDone {
                    run_id: record.run_id.clone(),
                    usage: None,
                    timestamp: Utc::now(),
                })
                .await?;
            println!("{}", response.content);
            Ok(())
        }
        agent_core::AgentLoopOutcome::AwaitingApproval {
            checkpoint,
            pending,
        } => {
            // The resumed run reached its NEXT gated effect: persist the
            // new pause with the same runtime facts and report it.
            let next = agent_core::PendingEffectRecord {
                pending_id: pending.pending_id.clone(),
                run_id: record.run_id.clone(),
                turn_id: record.turn_id.clone(),
                effect_id: pending.effect.effect_id.0.clone(),
                program_hash: pending.effect.program_hash.0.clone(),
                kind: pending.kind,
                request: pending.request.clone(),
                created_ts: Utc::now(),
                status: agent_core::PendingStatus::AwaitingApproval,
                resolved_ts: None,
                resolved_by: None,
                reason: None,
                runtime: record.runtime.clone(),
            };
            store.write_pending(&next, &checkpoint).await?;
            println!(
                "run paused again awaiting approval {}: gated {} effect (request: {}); resolve \
                 with `agent approvals --approve {}` or `agent approvals --deny {}`",
                next.pending_id,
                next.kind.as_str(),
                next.request,
                next.pending_id,
                next.pending_id
            );
            Ok(())
        }
    }
}

fn resolution_status(record: &agent_core::PendingEffectRecord) -> &'static str {
    match record.status {
        agent_core::PendingStatus::AwaitingApproval => "awaiting_approval",
        agent_core::PendingStatus::Approved => "approved",
        agent_core::PendingStatus::Denied => "denied",
    }
}

/// estimate_tokens drift against provider-reported usage (t-1163). Every
/// InferResult carries the provider's real input_tokens; when the trace
/// also recorded full prompts (--trace-full-payloads) we can re-estimate
/// each one and measure how far off the estimator is per model. This is
/// the proactive complement to catch-overflow: a calibrated estimator
/// makes threshold timing trustworthy again. Measurement only — no
/// correction factor is applied anywhere automatically.
#[derive(Debug, Clone, PartialEq)]
struct CalibrationReport {
    /// (model, estimated, provider-reported), one per usable infer pair.
    samples: Vec<(String, u64, u64)>,
}

impl CalibrationReport {
    fn from_events(events: &[Event]) -> Self {
        let mut estimates: BTreeMap<u64, (String, u64)> = BTreeMap::new();
        for event in events {
            if let Event::InferCall {
                op_id,
                model,
                prompt: Some(prompt),
                ..
            } = event
            {
                estimates.insert(
                    *op_id,
                    (model.clone(), agent_core::estimate_tokens(prompt) as u64),
                );
            }
        }
        let samples = events
            .iter()
            .filter_map(|event| match event {
                Event::InferResult {
                    op_id,
                    input_tokens,
                    ..
                } if *input_tokens > 0 => estimates
                    .get(op_id)
                    .map(|(model, estimate)| (model.clone(), *estimate, u64::from(*input_tokens))),
                _ => None,
            })
            .collect();
        Self { samples }
    }

    fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("\nEstimator calibration (estimate_tokens vs provider input_tokens):\n");
        if self.samples.is_empty() {
            out.push_str(
                "  no samples: trace lacks full prompts (record with --trace-full-payloads) \
                 or the provider reported no input_tokens\n",
            );
            return out;
        }
        let mut by_model: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
        for (model, estimated, actual) in &self.samples {
            by_model
                .entry(model.as_str())
                .or_default()
                .push((*estimated, *actual));
        }
        for (model, pairs) in by_model {
            let ratios: Vec<f64> = pairs
                .iter()
                .map(|(estimated, actual)| *estimated as f64 / *actual as f64)
                .collect();
            let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
            let min = ratios.iter().copied().fold(f64::INFINITY, f64::min);
            let max = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            // The factor a tuned estimator would multiply by; reported for
            // a human to consider, never applied.
            let correction = if mean > 0.0 { 1.0 / mean } else { 0.0 };
            out.push_str(&format!(
                "  {model}: {} sample(s), est/actual mean {mean:.2} (min {min:.2}, max {max:.2}), suggested correction x{correction:.2}\n",
                pairs.len(),
            ));
        }
        out
    }
}

async fn run_command(command: &Command) -> Result<()> {
    match command {
        Command::GcStats { trace } => run_gc_stats_command(trace).await,
        Command::Cost { trace, json } => run_cost_command(trace, *json).await,
        Command::Approvals {
            list,
            json,
            approve,
            deny,
            by,
            reason,
        } => {
            run_approvals_command(
                *list,
                *json,
                approve.as_deref(),
                deny.as_deref(),
                by.clone(),
                reason.clone(),
            )
            .await
        }
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
                agent_core::DynamicPath::at_entry(*visit),
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
        AuthCommand::Import { provider } => {
            let kind = agent_oauth::OAuthProviderKind::from_name(provider)?;
            if kind != agent_oauth::OAuthProviderKind::Codex {
                return Err(anyhow!(
                    "auth import is only supported for codex (reads the Codex CLI session)"
                ));
            }
            let token = agent_oauth::import_codex_cli().await?;
            println!(
                "imported codex credentials; token expires {}",
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
    let response = run_turn_with_status(runtime, TurnFrame::raw(prompt)).await?;
    runtime
        .trace
        .emit(&Event::AgentDone {
            run_id: runtime.run_id.clone(),
            usage: None,
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
    let response = run_turn_with_status(runtime, parse_turn_frame(&message)).await?;
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
    frame: TurnFrame,
) -> Result<agent_core::Response> {
    let TurnFrame {
        turn_id,
        input,
        metadata,
    } = frame;
    // Every turn boundary carries a turn id, supplied or minted, so the
    // supervisor can correlate agent_complete/agent_error to its send.
    let turn_id = turn_id.unwrap_or_else(|| mint_turn_id(&runtime.run_id, runtime.turn_seq));
    runtime.turn_seq += 1;
    emit_agent_start(runtime, &turn_id).await?;
    match run_turn(runtime, input, &turn_id).await {
        Ok(response) => {
            // A turn-budget stop with empty content must not look like a
            // crash: surface a clear terminal notice instead (t-1133).
            let completion = terminal_response(runtime, &response);
            emit_custom_event(
                runtime,
                "agent_complete",
                agent_complete_data(&completion, &turn_id, metadata.as_ref()),
            )
            .await?;
            Ok(response)
        }
        Err(err) => {
            let message = err.to_string();
            tracing::error!(run_id = %runtime.run_id, %turn_id, error = %message, "agent turn failed");
            if is_context_overflow_error(&message) {
                emit_context_overflow(runtime, &message).await?;
            }
            emit_custom_event(runtime, "agent_error", agent_error_data(&message, &turn_id)).await?;
            Err(err)
        }
    }
}

async fn run_turn(
    runtime: &mut Runtime,
    message: String,
    turn_id: &str,
) -> Result<agent_core::Response> {
    runtime.history.push(ChatMessage::user(message));
    let prompt = runtime.history.clone();
    let (response, mut new_history) = {
        // The remember/recall tools ride with the memory backend (settled
        // question 6): registering --memory-dir changes the loop program.
        let memory_tools = !runtime
            .config
            .hydration
            .sinks_of_kind(agent_core::SourceKind::Semantic)
            .is_empty();
        let options = agent_core::AgentLoopOptions {
            memory_tools,
            tool_names: runtime.config.tools.names(),
            output_contract: runtime.output_contract.clone(),
            shell_requires_approval: runtime.shell_requires_approval,
            infer_system_prompt: None,
        };
        let outcome = agent_core::run_agent_loop_outcome(
            &runtime.config,
            &mut runtime.ir_store,
            runtime.ir_replay.as_ref(),
            &mut runtime.gc_state,
            runtime.model.clone(),
            prompt.clone(),
            runtime.max_turns,
            &options,
            runtime.ir_effect_visits.clone(),
        )
        .await?;
        let (value, machine) = match outcome {
            agent_core::AgentLoopOutcome::Complete { value, machine } => (value, machine),
            agent_core::AgentLoopOutcome::AwaitingApproval {
                checkpoint,
                pending,
            } => {
                return Err(pause_turn(runtime, turn_id, checkpoint, pending).await?);
            }
        };
        runtime.ir_effect_visits = machine.effect_visits.clone();
        // Exhausted output-schema repairs come back as a typed value (the
        // loop's errors-as-values convention); surface them as a failed
        // turn — agent_error in session mode, nonzero exit one-shot —
        // rather than emitting non-conforming output as a completion.
        if let Some(failure) = agent_core::output_contract_failure(&value) {
            return Err(anyhow!(
                "final output failed the output schema after {} attempt(s): {}",
                failure.attempts,
                failure.errors.join("; ")
            ));
        }
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
    persist_session(runtime).await;
    Ok(response)
}

/// Persist an approval pause durably (the pending record plus the mid-turn
/// machine checkpoint, under `~/.local/share/agent/approvals`) and return
/// the error that reports it. The turn does NOT complete: in one-shot mode
/// the process exits nonzero with this message; in session mode it rides
/// the turn's `agent_error` machine event (naming the pending id) and the
/// session stays alive for further turns. Resolution and resumption — in
/// this or any later process — are `agent approvals --approve/--deny`.
async fn pause_turn(
    runtime: &Runtime,
    turn_id: &str,
    checkpoint: agent_core::IrCheckpoint,
    pending: agent_core::ApprovalRequest,
) -> Result<anyhow::Error> {
    // Replay reproduces a recorded pause as data (the gate already
    // re-emitted its events); persisting or waiting would make replay
    // side-effecting.
    if runtime.ir_replay.is_some() {
        return Ok(anyhow!(
            "replay reproduced an approval pause: effect {} (pending {}) was recorded awaiting \
             approval and never resolved",
            pending.effect.effect_id.0,
            pending.pending_id
        ));
    }
    let store = agent_core::ApprovalStore::new(agent_core::ApprovalStore::default_dir()?);
    let record = agent_core::PendingEffectRecord {
        pending_id: pending.pending_id.clone(),
        run_id: runtime.run_id.clone(),
        turn_id: Some(turn_id.to_owned()),
        effect_id: pending.effect.effect_id.0.clone(),
        program_hash: pending.effect.program_hash.0.clone(),
        kind: pending.kind,
        request: pending.request.clone(),
        created_ts: Utc::now(),
        status: agent_core::PendingStatus::AwaitingApproval,
        resolved_ts: None,
        resolved_by: None,
        reason: None,
        runtime: Some(serde_json::to_value(&runtime.resume_facts)?),
    };
    store.write_pending(&record, &checkpoint).await?;
    tracing::info!(
        run_id = %runtime.run_id,
        %turn_id,
        pending_id = %record.pending_id,
        "turn paused awaiting approval"
    );
    Ok(anyhow!(
        "turn paused awaiting approval {id}: gated {kind} effect did not execute \
         (request: {request}); resolve with `agent approvals --approve {id}` or \
         `agent approvals --deny {id}` (state under {dir})",
        id = record.pending_id,
        kind = record.kind.as_str(),
        request = record.request,
        dir = store.dir().display(),
    ))
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
            usage: None,
            timestamp: Utc::now(),
        })
        .await
}

async fn emit_agent_start(runtime: &mut Runtime, turn_id: &str) -> Result<()> {
    emit_custom_event(
        runtime,
        "agent_start",
        serde_json::json!({
            "turn_id": turn_id,
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

/// agent_complete payload: the terminal response plus the turn id, and the
/// envelope's opaque metadata echoed back when the caller supplied one.
fn agent_complete_data(
    response: &str,
    turn_id: &str,
    metadata: Option<&serde_json::Value>,
) -> serde_json::Value {
    let mut data = serde_json::json!({ "response": response, "turn_id": turn_id });
    if let Some(metadata) = metadata {
        data["metadata"] = metadata.clone();
    }
    data
}

/// agent_error payload: an errored turn must still correlate, so the turn id
/// rides on the error event exactly like on agent_complete.
fn agent_error_data(message: &str, turn_id: &str) -> serde_json::Value {
    serde_json::json!({ "message": message, "turn_id": turn_id })
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

/// Persist the session at turn completion via the ChatHistory sink
/// (docs/MEMORY.md, t-1181). This is the design's passive-sink channel:
/// runtime-initiated at a lifecycle point, outside the program's effect
/// stream, suppressed under replay, and failures log-and-continue — a
/// failing sink must never fail the turn. It absorbs the former
/// put_checkpoint (a redundant session:state Put, removed) and writes the
/// exact same files and schema as before, so the Haskell agentd and the
/// persistence eval are unaffected.
async fn persist_session(runtime: &mut Runtime) {
    let Some(dir) = runtime.checkpoint_dir.clone() else {
        return;
    };
    // Replay re-runs a recorded session deterministically; writing would
    // clobber the real session's checkpoint with replayed state.
    if runtime.ir_replay.is_some() || runtime.config.replay.is_some() {
        return;
    }
    let sequence = runtime.checkpoint_sequence + 1;
    let Some(checkpoint) = checkpoint_from_runtime(runtime, sequence) else {
        tracing::warn!(run_id = %runtime.run_id, "skipping checkpoint with pending tool calls");
        return;
    };
    let payload = match serde_json::to_value(&checkpoint) {
        Ok(payload) => payload,
        Err(err) => {
            tracing::error!(run_id = %runtime.run_id, error = %err, "serializing checkpoint");
            return;
        }
    };
    let sink = ChatHistory::new(dir);
    match sink
        .store(agent_core::SinkItem {
            payload,
            provenance: agent_core::Provenance {
                run_id: runtime.run_id.clone(),
                effect_id: None,
                timestamp: Some(Utc::now()),
            },
        })
        .await
    {
        Ok(_) => {
            runtime.checkpoint_sequence = sequence;
            let name = format!("checkpoint-{sequence:06}");
            if let Err(err) = runtime
                .trace
                .emit(&Event::Checkpoint {
                    run_id: runtime.run_id.clone(),
                    name: name.clone(),
                    path: runtime
                        .checkpoint_dir
                        .as_ref()
                        .map(|dir| dir.join("session-latest.json").display().to_string()),
                    timestamp: Utc::now(),
                })
                .await
            {
                tracing::error!(run_id = %runtime.run_id, error = %err, "emitting checkpoint event");
            }
            tracing::info!(run_id = %runtime.run_id, %name, "checkpoint saved");
            if runtime.debug {
                eprintln!("checkpoint: {name}");
            }
        }
        Err(err) => {
            // Log-and-continue: a failed passive write must not fail the turn.
            tracing::error!(run_id = %runtime.run_id, error = %format!("{err:#}"), "checkpoint write failed");
        }
    }
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
        discovered_budget: runtime.gc_state.discovered_budget,
    })
}

async fn load_output_contract(path: &Path) -> Result<agent_core::OutputContract> {
    let text = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("reading output schema {}", path.display()))?;
    parse_output_contract(&text)
        .with_context(|| format!("parsing output schema {}", path.display()))
}

/// Pure parsing seam (like `parse_turn_frame`): output-schema file text to
/// contract. The document must be a JSON object — a bare `true`/`false`
/// schema is technically valid JSON Schema but useless as an output
/// contract, and anything else is a caller error worth failing loudly.
fn parse_output_contract(text: &str) -> Result<agent_core::OutputContract> {
    let schema: serde_json::Value =
        serde_json::from_str(text).context("output schema is not valid JSON")?;
    if !schema.is_object() {
        return Err(anyhow!(
            "output schema must be a JSON object (a JSON Schema document), got {}",
            match schema {
                serde_json::Value::Array(_) => "an array",
                serde_json::Value::String(_) => "a string",
                serde_json::Value::Bool(_) => "a boolean",
                serde_json::Value::Number(_) => "a number",
                serde_json::Value::Null => "null",
                serde_json::Value::Object(_) => unreachable!(),
            }
        ));
    }
    Ok(agent_core::OutputContract::new(schema))
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
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            metadata,
        }
    }

    // Turn envelope classification (t-1308.2): the frame boundary decides
    // envelope vs raw prompt, and the decision must be byte-preserving for
    // every pre-envelope sender.
    #[test]
    fn plain_text_frame_is_a_raw_prompt() {
        assert_eq!(
            parse_turn_frame("run the tests"),
            TurnFrame::raw("run the tests".into())
        );
    }

    #[test]
    fn json_looking_prompts_without_the_envelope_shape_stay_raw() {
        // Valid JSON objects that lack v:1 + string input are prompts,
        // byte-for-byte: the model was probably meant to see them.
        for frame in [
            r#"{"input":"hi"}"#,                     // no version tag
            r#"{"v":2,"input":"hi"}"#,               // future version
            r#"{"v":"1","input":"hi"}"#,             // version is a string
            r#"{"v":1.0,"input":"hi"}"#,             // version is a float
            r#"{"v":1}"#,                            // no input
            r#"{"v":1,"input":42}"#,                 // input is not a string
            r#"[{"v":1,"input":"hi"}]"#,             // not an object
            r#"{"v":1,"input":"hi""#,                // not JSON at all
            "explain this JSON: {\"v\":1, \"x\":2}", // prose containing JSON
        ] {
            assert_eq!(
                parse_turn_frame(frame),
                TurnFrame::raw(frame.into()),
                "frame: {frame}"
            );
        }
    }

    #[test]
    fn envelope_frame_carries_id_input_and_metadata() {
        let frame = parse_turn_frame(
            r#"{"v":1,"turn_id":"send-7","input":"run tests","metadata":{"sender":"ben"}}"#,
        );
        assert_eq!(
            frame,
            TurnFrame {
                turn_id: Some("send-7".into()),
                input: "run tests".into(),
                metadata: Some(serde_json::json!({"sender":"ben"})),
            }
        );
    }

    #[test]
    fn envelope_turn_id_and_metadata_are_optional() {
        assert_eq!(
            parse_turn_frame(r#"{"v":1,"input":"hi"}"#),
            TurnFrame {
                turn_id: None,
                input: "hi".into(),
                metadata: None,
            }
        );
    }

    #[test]
    fn envelope_with_malformed_turn_id_still_parses_as_envelope() {
        // Classification depends only on v + input; a non-string turn_id is
        // dropped (an id gets minted) rather than demoting the frame to a
        // raw prompt.
        assert_eq!(
            parse_turn_frame(r#"{"v":1,"turn_id":42,"input":"hi"}"#),
            TurnFrame {
                turn_id: None,
                input: "hi".into(),
                metadata: None,
            }
        );
    }

    #[test]
    fn envelope_shaped_prompt_must_be_wrapped_to_be_literal() {
        // The documented edge: to deliver text that IS a valid v1 envelope
        // as a prompt, wrap it in an envelope. The outer envelope parses;
        // the inner text arrives untouched as the input.
        let literal = r#"{"v":1,"input":"hi"}"#;
        let wrapped = serde_json::json!({"v": 1, "input": literal}).to_string();
        assert_eq!(parse_turn_frame(&wrapped).input, literal);
        // Unwrapped, the same bytes are treated as an envelope — the caller
        // contract, not a bug.
        assert_eq!(parse_turn_frame(literal).input, "hi");
    }

    #[test]
    fn minted_turn_ids_are_deterministic_per_run_ordinal() {
        assert_eq!(mint_turn_id("run-abc", 0), "run-abc-t0");
        assert_eq!(mint_turn_id("run-abc", 7), "run-abc-t7");
        // Same run + ordinal always mints the same id.
        assert_eq!(mint_turn_id("run-abc", 0), mint_turn_id("run-abc", 0));
    }

    #[test]
    fn completion_and_error_events_both_carry_the_turn_id() {
        // A turn that errors must still correlate to its sender.
        let complete = agent_complete_data("done", "send-7", None);
        assert_eq!(complete["turn_id"], "send-7");
        assert_eq!(complete["response"], "done");
        assert!(complete.get("metadata").is_none());

        let error = agent_error_data("provider exploded", "send-7");
        assert_eq!(error["turn_id"], "send-7");
        assert_eq!(error["message"], "provider exploded");
    }

    #[test]
    fn completion_event_echoes_envelope_metadata() {
        let metadata = serde_json::json!({"sender":"eval","nested":{"k":[1,2]}});
        let complete = agent_complete_data("done", "send-7", Some(&metadata));
        assert_eq!(complete["metadata"], metadata);
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
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
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

        let (resolved, pricing, embedder) =
            resolve_model_from(missing_registry, Some("openrouter/auto".into()))?;
        assert!(embedder.is_none(), "no registry, no embedder");

        assert_eq!(resolved.alias, "openrouter/auto");
        assert_eq!(resolved.api_id, "openrouter/auto");
        assert_eq!(resolved.provider, None);
        assert_eq!(resolved.pricing, None);
        assert!(pricing.is_empty(), "fallback has no pricing to guess from");
        Ok(())
    }

    #[test]
    fn missing_registry_without_requested_model_is_an_error() {
        // No unwrap_err: the Ok tuple holds an Arc<dyn Embedder>, which has
        // no Debug impl.
        let Err(err) = resolve_model_from(Err(anyhow!("no registry")), None) else {
            panic!("missing registry without a requested model must error");
        };
        assert!(err.to_string().contains("models.yaml"), "got: {err}");
    }

    #[test]
    fn checkpoint_discovered_budget_round_trips_and_old_checkpoints_load() {
        // A pre-t-1162 checkpoint has no discovered_budget field.
        let old = serde_json::json!({
            "run_id": "run", "sequence": 1, "model": "m",
            "provider_url": "https://example.com",
            "messages": [], "trace_path": "/tmp/t.jsonl",
            "timestamp": "2026-06-12T00:00:00Z",
        });
        let loaded: Checkpoint = serde_json::from_value(old).unwrap();
        assert_eq!(loaded.discovered_budget, None);

        let mut with_budget = loaded;
        with_budget.discovered_budget = Some(120_000);
        let round_tripped: Checkpoint =
            serde_json::from_str(&serde_json::to_string(&with_budget).unwrap()).unwrap();
        assert_eq!(round_tripped.discovered_budget, Some(120_000));
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
            discovered_budget: None,
        };
        tokio::fs::write(&path, serde_json::to_vec_pretty(&checkpoint)?).await?;

        let repaired = load_checkpoint(&path).await?;

        assert!(!agent_core::has_pending_tool_calls(&repaired.messages));
        assert_eq!(repaired.messages.len(), 2);
        let on_disk: Checkpoint = serde_json::from_slice(&tokio::fs::read(&path).await?)?;
        assert_eq!(on_disk.messages.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn chat_history_sink_write_resumes_through_load_checkpoint() -> Result<()> {
        // The cross-boundary contract of t-1181: a checkpoint written by the
        // ChatHistory sink (the passive turn-completion path) is byte-schema
        // compatible with the resume reader, so --resume keeps working.
        let dir = std::env::temp_dir().join(format!("agent-resume-{}", Uuid::new_v4()));
        let checkpoint = Checkpoint {
            run_id: "run-resume".into(),
            sequence: 4,
            model: "model".into(),
            provider_url: "https://chatgpt.com/backend-api".into(),
            messages: vec![
                ChatMessage::system("system"),
                ChatMessage::user("remember this"),
                ChatMessage::assistant(Some("noted".into()), vec![]),
            ],
            trace_path: dir.join("trace.jsonl"),
            timestamp: Utc::now(),
            discovered_budget: Some(123_000),
        };

        let sink = ChatHistory::new(dir.clone());
        sink.store(agent_core::SinkItem {
            payload: serde_json::to_value(&checkpoint)?,
            provenance: Default::default(),
        })
        .await?;

        // session-latest.json is what --resume reads.
        let resumed = load_checkpoint(&dir.join("session-latest.json")).await?;
        assert_eq!(resumed.run_id, "run-resume");
        assert_eq!(resumed.sequence, 4);
        assert_eq!(resumed.discovered_budget, Some(123_000));
        assert_eq!(resumed.messages.len(), 3);
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
                usage: None,
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
            usage: None,
            timestamp: Utc::now(),
        }];

        let report = GcStatsReport::from_events(&events);

        assert_eq!(report.fire_count(), 0);
        assert_eq!(report.render(), "GC never fired in this trace\n");
    }

    fn custom_gc_with(data: serde_json::Value) -> Event {
        Event::Custom {
            run_id: "run".into(),
            name: "gc_collect".into(),
            data,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn gc_stats_surfaces_timing_and_overflow_recovery_fields() {
        let events = vec![
            custom_gc_with(serde_json::json!({
                "type": "gc_collect", "strategy": "ring", "timing": "catch-overflow",
                "target_budget": 120_000, "trigger": "context_overflow", "cycle": 1,
                "tokens_before": 240_000, "tokens_after": 110_000,
                "cache_invalidated": true, "dropped_count": 12,
            })),
            custom_gc_with(serde_json::json!({
                "type": "gc_collect", "strategy": "ring", "timing": "catch-overflow",
                "target_budget": 60_000, "trigger": "context_overflow", "cycle": 2,
                "tokens_before": 110_000, "tokens_after": 55_000,
                "cache_invalidated": true, "dropped_count": 9,
            })),
            custom_gc_with(serde_json::json!({
                "type": "gc_collect", "strategy": "ring", "timing": "threshold",
                "target_budget": 170_000,
                "tokens_before": 200_000, "tokens_after": 150_000,
                "cache_invalidated": false, "dropped_count": 4,
            })),
        ];

        let rendered = GcStatsReport::from_events(&events).render();

        assert!(rendered.contains("catch-overflow"), "{rendered}");
        assert!(
            rendered.contains("[overflow cycle 2 -> target 60000]"),
            "{rendered}"
        );
        assert!(
            rendered.contains("fires by timing: catch-overflow=2 threshold=1"),
            "{rendered}"
        );
        assert!(
            rendered.contains(
                "overflow recoveries: 2 fire(s), deepest cycle 2, final target budget 60000"
            ),
            "{rendered}"
        );
    }

    #[test]
    fn calibration_reports_per_model_drift_from_paired_infer_events() {
        let prompt = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("x".repeat(400)),
        ];
        let estimate = agent_core::estimate_tokens(&prompt) as u64;
        let infer_call = |op_id: u64, model: &str| Event::InferCall {
            run_id: "run".into(),
            op_id,
            parent_op_id: None,
            model: model.into(),
            prompt: Some(prompt.clone()),
            prompt_preview: String::new(),
            effect: None,
            timestamp: Utc::now(),
        };
        let infer_result = |op_id: u64, input_tokens: u32| Event::InferResult {
            run_id: "run".into(),
            op_id,
            parent_op_id: None,
            response: None,
            response_preview: String::new(),
            input_tokens,
            output_tokens: 1,
            total_tokens: input_tokens + 1,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            duration_ms: 1,
            timestamp: Utc::now(),
        };
        let actual = (estimate * 2) as u32; // estimator reads half the truth
        let events = vec![
            infer_call(1, "gpt-5.5"),
            infer_result(1, actual),
            infer_call(2, "gpt-5.5"),
            infer_result(2, actual),
            // Preview-only call (no full prompt): contributes no sample.
            Event::InferCall {
                run_id: "run".into(),
                op_id: 3,
                parent_op_id: None,
                model: "gpt-5.5".into(),
                prompt: None,
                prompt_preview: String::new(),
                effect: None,
                timestamp: Utc::now(),
            },
            infer_result(3, 100),
        ];

        let report = CalibrationReport::from_events(&events);
        assert_eq!(report.samples.len(), 2);
        let rendered = report.render();
        assert!(
            rendered.contains("gpt-5.5: 2 sample(s), est/actual mean 0.50"),
            "{rendered}"
        );
        assert!(
            rendered.contains("suggested correction x2.00"),
            "{rendered}"
        );
    }

    #[test]
    fn calibration_without_full_prompts_explains_itself() {
        let rendered = CalibrationReport::from_events(&[]).render();
        assert!(rendered.contains("--trace-full-payloads"), "{rendered}");
    }

    #[test]
    fn gc_stats_handles_pre_t1151_traces_without_timing_fields() {
        let rendered = GcStatsReport::from_events(&[custom_gc(100, 60, false)]).render();

        assert!(rendered.contains("n/a"), "{rendered}");
        assert!(
            rendered.contains("fires by timing: unreported=1"),
            "{rendered}"
        );
        assert!(!rendered.contains("overflow recoveries"), "{rendered}");
    }

    #[test]
    fn parse_output_contract_accepts_a_schema_object_with_default_repairs() {
        let contract =
            parse_output_contract(r#"{ "type": "object", "required": ["answer"] }"#).unwrap();
        assert_eq!(contract.max_repairs, agent_core::DEFAULT_MAX_REPAIRS);
        assert_eq!(contract.schema["type"], "object");
    }

    #[test]
    fn parse_output_contract_rejects_invalid_json() {
        let err = parse_output_contract("{ nope").unwrap_err();
        assert!(err.to_string().contains("not valid JSON"), "{err:#}");
    }

    #[test]
    fn parse_output_contract_rejects_non_object_documents() {
        for (text, kind) in [
            ("true", "a boolean"),
            ("[1, 2]", "an array"),
            ("\"x\"", "a string"),
            ("null", "null"),
        ] {
            let err = parse_output_contract(text).unwrap_err();
            assert!(err.to_string().contains(kind), "{text}: {err:#}");
        }
    }
}

#[cfg(test)]
mod cost_tests {
    use super::*;
    use chrono::Utc;

    fn infer_call(run_id: &str, op_id: u64, model: &str) -> Event {
        Event::InferCall {
            run_id: run_id.into(),
            op_id,
            parent_op_id: None,
            model: model.into(),
            prompt: None,
            prompt_preview: String::new(),
            effect: None,
            timestamp: Utc::now(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn infer_result(
        run_id: &str,
        op_id: u64,
        input_tokens: u32,
        output_tokens: u32,
        cached_input_tokens: Option<u32>,
        cost_micro_usd: Option<u64>,
    ) -> Event {
        Event::InferResult {
            run_id: run_id.into(),
            op_id,
            parent_op_id: None,
            response: None,
            response_preview: String::new(),
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens,
            cost_micro_usd,
            pricing: None,
            duration_ms: 1,
            timestamp: Utc::now(),
        }
    }

    fn two_run_events() -> Vec<Event> {
        vec![
            infer_call("run-a", 1, "claude-sonnet-4-6"),
            infer_result("run-a", 1, 1000, 200, Some(100), Some(9_000)),
            infer_call("run-a", 2, "claude-sonnet-4-6"),
            infer_result("run-a", 2, 2000, 300, None, Some(10_500)),
            // Recorded without pricing: counted, tokens summed, no cost.
            infer_call("run-a", 3, "gpt-5.5"),
            infer_result("run-a", 3, 50, 5, None, None),
            // Result whose call is missing from the trace (filtered file).
            infer_result("run-b", 7, 10, 1, None, Some(33)),
        ]
    }

    #[test]
    fn cost_report_aggregates_per_model_and_per_run_from_recorded_integers() {
        let report = CostReport::from_events(&two_run_events());

        assert_eq!(report.runs.len(), 2);
        let run_a = &report.runs[0];
        assert_eq!(run_a.run_id, "run-a");
        let sonnet = &run_a.per_model["claude-sonnet-4-6"];
        assert_eq!(sonnet.infer_calls, 2);
        assert_eq!(sonnet.input_tokens, 3000);
        assert_eq!(sonnet.output_tokens, 500);
        assert_eq!(sonnet.cached_input_tokens, Some(100));
        assert_eq!(sonnet.cost_micro_usd, Some(19_500));
        assert_eq!(sonnet.uncosted_infer_calls, 0);
        let gpt = &run_a.per_model["gpt-5.5"];
        assert_eq!(gpt.cost_micro_usd, None, "absent means absent, not zero");
        assert_eq!(gpt.uncosted_infer_calls, 1);
        assert_eq!(run_a.total.infer_calls, 3);
        assert_eq!(run_a.total.cost_micro_usd, Some(19_500));
        assert_eq!(run_a.total.uncosted_infer_calls, 1);

        let run_b = &report.runs[1];
        assert!(run_b.per_model.contains_key("(unknown)"));
        assert_eq!(run_b.total.cost_micro_usd, Some(33));

        assert_eq!(report.total.infer_calls, 4);
        assert_eq!(report.total.total_tokens, 3566);
        assert_eq!(report.total.cost_micro_usd, Some(19_533));
        assert_eq!(report.total.uncosted_infer_calls, 1);
    }

    /// Golden rendering: pins the human-readable table so accidental format
    /// churn is visible in review.
    #[test]
    fn cost_report_renders_golden_table() {
        let rendered = CostReport::from_events(&two_run_events()).render();
        let expected = "\
Run run-a
  model                                     infers    input tok   output tok     cached     cost (USD)
  claude-sonnet-4-6                              2         3000          500        100      $0.019500
  gpt-5.5                                        1           50            5        n/a            n/a
  run total                                      3         3050          505        100      $0.019500
  uncosted infer calls: 1 (no pricing recorded; cost total is partial)

Run run-b
  model                                     infers    input tok   output tok     cached     cost (USD)
  (unknown)                                      1           10            1        n/a      $0.000033
  run total                                      1           10            1        n/a      $0.000033

Total: 4 infer(s), 3060 input + 506 output = 3566 tokens, cost $0.019533 (1 uncosted)
";
        assert_eq!(rendered, expected, "actual:\n{rendered}");
    }

    #[test]
    fn cost_report_json_carries_canonical_micro_usd_and_derived_decimal() {
        let json = CostReport::from_events(&two_run_events()).to_json();
        assert_eq!(json["total"]["cost_micro_usd"], 19_533);
        assert_eq!(json["total"]["cost_usd"], "0.019533");
        assert_eq!(json["total"]["uncosted_infer_calls"], 1);
        let run_a = &json["runs"][0];
        assert_eq!(run_a["run_id"], "run-a");
        assert_eq!(
            run_a["models"]["claude-sonnet-4-6"]["cost_micro_usd"],
            19_500
        );
        assert_eq!(
            run_a["models"]["claude-sonnet-4-6"]["cached_input_tokens"],
            100
        );
        // Unpriced models carry no cost keys at all.
        assert!(run_a["models"]["gpt-5.5"]
            .as_object()
            .unwrap()
            .get("cost_micro_usd")
            .is_none());
    }

    #[test]
    fn cost_report_on_infer_less_trace_says_so() {
        let rendered = CostReport::from_events(&[]).render();
        assert!(
            rendered.contains("no InferResult/InferError events"),
            "{rendered}"
        );
    }

    /// Failed attempts (InferError, t-1347) are counted per model and per
    /// run — attempts only, no usage exists for them — and surface as a
    /// `failed` column exactly when a run has any.
    #[test]
    fn cost_report_counts_failed_calls_and_shows_the_column_when_nonzero() {
        let mut events = two_run_events();
        events.push(infer_call("run-a", 9, "eval-dead-model"));
        events.push(Event::InferError {
            run_id: "run-a".into(),
            op_id: 9,
            parent_op_id: Some(1),
            error: "model not found (404)".into(),
            duration_ms: 1,
            timestamp: Utc::now(),
        });

        let report = CostReport::from_events(&events);
        let run_a = &report.runs[0];
        let dead = &run_a.per_model["eval-dead-model"];
        assert_eq!(dead.failed_infer_calls, 1);
        assert_eq!(dead.infer_calls, 0, "no InferResult, no successful call");
        assert_eq!(dead.total_tokens, 0, "attempts carry no usage");
        assert_eq!(run_a.total.failed_infer_calls, 1);
        assert_eq!(report.total.failed_infer_calls, 1);
        // The all-success run is untouched.
        assert_eq!(report.runs[1].total.failed_infer_calls, 0);

        let rendered = report.render();
        assert!(rendered.contains("failed"), "{rendered}");
        assert!(
            rendered.contains("failed infer calls: 1 (no usage recorded; attempts only)"),
            "{rendered}"
        );
        assert!(rendered.contains("(1 failed)"), "{rendered}");
        // run-b has no failures: its table keeps the pre-t-1347 shape.
        let run_b_table = rendered
            .split("Run run-b")
            .nth(1)
            .unwrap()
            .split("Total:")
            .next()
            .unwrap();
        assert!(
            !run_b_table.contains("failed"),
            "failed column must be absent for all-success runs: {run_b_table}"
        );

        // JSON follows RunUsage serialization: key present only when nonzero.
        let json = report.to_json();
        assert_eq!(json["total"]["failed_infer_calls"], 1);
        assert!(json["runs"][1]["total"]
            .as_object()
            .unwrap()
            .get("failed_infer_calls")
            .is_none());
    }
}
