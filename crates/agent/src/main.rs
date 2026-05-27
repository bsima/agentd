use agent_core::{
    agent_loop, ChatMessage, Event, HydrationSource, ModelRegistry, PassiveHydrationConfig,
    PassiveSource, ProviderClient, ProviderConfig, SeqConfig, SourceCapability, SourceKind,
    SourceParams, SourceRegistry, SourceResult, TraceLogger,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use uuid::Uuid;

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
    /// Directory to read into startup hydration context.
    #[arg(long, env = "AGENT_HYDRATION_DIR")]
    hydration_dir: Option<PathBuf>,
    /// Accept compaction flag for agentd compatibility; compaction is not implemented yet.
    #[arg(long)]
    enable_compaction: bool,
    /// One-shot prompt. Omit when using --fifo or NUL-framed stdin sessions.
    prompt: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(command) = args.command.as_ref() {
        return run_command(command).await;
    }
    let file_config = read_config(args.config.as_ref()).await?;
    let provider_file = file_config.provider.unwrap_or_default();
    let registry = ModelRegistry::load_default().await?;

    let requested_model = args
        .model
        .or(provider_file.model)
        .or_else(|| std::env::var("AGENT_MODEL").ok());
    let resolved_model = registry.resolve(requested_model.as_deref())?;
    let url = args
        .provider
        .or(provider_file.url)
        .or(resolved_model.base_url.clone())
        .or_else(|| std::env::var("AGENT_PROVIDER").ok())
        .or_else(|| std::env::var("OPENROUTER_BASE_URL").ok())
        .unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
    let model = resolved_model.api_id.clone();
    let oauth_provider = resolved_model
        .provider
        .as_deref()
        .filter(|provider| matches!(*provider, "codex-oauth" | "claude-code-oauth"));
    #[cfg(not(feature = "oauth"))]
    if let Some(provider) = oauth_provider {
        return Err(anyhow!(
            "model '{}' requires OAuth provider '{provider}', but this agent was built without the 'oauth' feature",
            resolved_model.alias
        ));
    }
    let api_key = if oauth_provider.is_some() {
        None
    } else {
        Some(
            args.key
                .or(provider_file.api_key)
                .or(resolved_model.api_key.clone())
                .or_else(|| std::env::var("AGENT_API_KEY").ok())
                .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
                .ok_or_else(|| {
                    anyhow!("missing API key: pass --key, set AGENT_API_KEY/OPENROUTER_API_KEY, or configure api_key in models.yaml")
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
    let trace = TraceLogger::new(run_id.clone(), trace_path.clone());
    let provider: Arc<dyn agent_core::ChatProvider> = match oauth_provider {
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
        None => Arc::new(ProviderClient::new(ProviderConfig {
            url: url.clone(),
            api_key: api_key.expect("api_key is set for non-OAuth providers"),
            model: agent_core::Model(model.clone()),
        })),
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
        None => (initial_history(&hydration).await?, 0),
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
    };
    let mut runtime = Runtime {
        config,
        trace,
        run_id: run_id.clone(),
        model: agent_core::Model(model.clone()),
        provider_url: url.clone(),
        trace_path: trace_path.clone(),
        checkpoint_dir: args.checkpoint_dir,
        checkpoint_path,
        checkpoint_sequence,
        history,
        debug: args.debug,
    };

    if args.debug {
        eprintln!("run_id: {run_id}");
        eprintln!("model: {model}");
        eprintln!("provider: {url}");
        eprintln!("trace: {}", trace_path.display());
    }

    match (args.prompt, args.fifo, args.session) {
        (Some(prompt), None, false) => run_one_shot(&mut runtime, prompt).await,
        (Some(_), Some(_), _) => Err(anyhow!("provide either a prompt or --fifo, not both")),
        (Some(_), None, true) => Err(anyhow!("provide either a prompt or --session, not both")),
        (None, Some(path), _) => run_fifo_session(&mut runtime, path).await,
        (None, None, _) => run_stdin_session(&mut runtime).await,
    }
}

async fn run_command(command: &Command) -> Result<()> {
    match command {
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
    println!("{}", response.content);
    Ok(())
}

async fn run_stdin_session(runtime: &mut Runtime) -> Result<()> {
    let reader = BufReader::new(tokio::io::stdin());
    run_nul_delimited_prompt_loop(runtime, reader).await
}

async fn run_fifo_session(runtime: &mut Runtime, path: PathBuf) -> Result<()> {
    loop {
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .open(&path)
            .await
            .with_context(|| format!("opening fifo {}", path.display()))?;
        let reader = BufReader::new(file);
        run_nul_delimited_prompt_loop(runtime, reader).await?;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

async fn run_nul_delimited_prompt_loop<R>(runtime: &mut Runtime, mut reader: R) -> Result<()>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    loop {
        tokio::select! {
            frame = read_nul_frame(&mut reader) => {
                match frame? {
                    Some(message) if message.is_empty() => break,
                    Some(message) => write_session_response(runtime, message).await?,
                    None => break,
                }
            }
            _ = shutdown_signal() => break,
        }
    }
    emit_done(runtime).await
}

async fn write_session_response(runtime: &mut Runtime, message: String) -> Result<()> {
    let response = run_turn_with_status(runtime, message).await?;
    let mut stdout = tokio::io::stdout();
    stdout.write_all(response.content.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

async fn run_turn_with_status(
    runtime: &mut Runtime,
    message: String,
) -> Result<agent_core::Response> {
    emit_agent_start(runtime).await?;
    match run_turn(runtime, message).await {
        Ok(response) => {
            emit_agent_complete(runtime, &response.content).await?;
            Ok(response)
        }
        Err(err) => {
            let message = err.to_string();
            emit_agent_error(runtime, &message).await?;
            Err(err)
        }
    }
}

async fn run_turn(runtime: &mut Runtime, message: String) -> Result<agent_core::Response> {
    runtime.history.push(ChatMessage::user(message));
    let prompt = runtime.history.clone();
    let program = agent_loop(runtime.model.clone(), prompt.clone(), 16);
    let (response, mut new_history) =
        agent_core::run_sequential(&runtime.config, prompt, program).await?;
    if !response.content.is_empty() || response.tool_calls.is_empty() {
        new_history.push(ChatMessage::assistant(
            (!response.content.is_empty()).then_some(response.content.clone()),
            response.tool_calls.clone(),
        ));
    }
    runtime.history = new_history;
    put_checkpoint(runtime).await?;
    save_checkpoint(runtime).await?;
    Ok(response)
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
    let checkpoint = Checkpoint {
        run_id: runtime.run_id.clone(),
        sequence: runtime.checkpoint_sequence + 1,
        model: runtime.model.0.clone(),
        provider_url: runtime.provider_url.clone(),
        messages: runtime.history.clone(),
        trace_path: runtime.trace_path.clone(),
        timestamp: Utc::now(),
    };
    let value = serde_json::to_value(checkpoint)?;
    let config = SeqConfig {
        provider: runtime.config.provider.clone(),
        hydration: runtime.config.hydration.clone(),
        passive_hydration: PassiveHydrationConfig::default(),
        checkpoint_path: Some(path),
        trace: runtime.trace.clone(),
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
    runtime.checkpoint_sequence += 1;
    tokio::fs::create_dir_all(dir).await?;
    let checkpoint = Checkpoint {
        run_id: runtime.run_id.clone(),
        sequence: runtime.checkpoint_sequence,
        model: runtime.model.0.clone(),
        provider_url: runtime.provider_url.clone(),
        messages: runtime.history.clone(),
        trace_path: runtime.trace_path.clone(),
        timestamp: Utc::now(),
    };
    let bytes = serde_json::to_vec_pretty(&checkpoint)?;
    let path = dir.join(format!(
        "checkpoint-{:06}-{}.json",
        checkpoint.sequence, runtime.run_id
    ));
    tokio::fs::write(&path, &bytes).await?;
    tokio::fs::write(dir.join("latest.json"), &bytes).await?;
    tokio::fs::write(dir.join("session-latest.json"), bytes).await?;
    if runtime.debug {
        eprintln!("checkpoint: {}", path.display());
    }
    Ok(())
}

async fn load_checkpoint(path: &Path) -> Result<Checkpoint> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("reading checkpoint {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("parsing checkpoint {}", path.display()))
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
    "You are a standalone agent runner. Use the shell tool when you need to inspect or change the environment. The shell tool executes command strings with the SHELL environment variable inside the sandboxed PATH. When finished, answer concisely."
}

async fn initial_history(sources: &SourceRegistry) -> Result<Vec<ChatMessage>> {
    let mut prompt = base_system_prompt().to_string();
    let hydration = sources.retrieve_all(SourceParams::default()).await?;

    if !hydration.is_empty() {
        prompt.push_str("\n\nHydrated context:\n");
        for result in hydration {
            prompt.push_str(&format!(
                "\n## {} ({:?})\n{}\n",
                result.source, result.kind, result.content
            ));
        }
    }

    Ok(vec![ChatMessage::system(prompt)])
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
        let mut remaining = max_bytes;
        let mut files = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            if remaining == 0 {
                break;
            }
            let path = entry.path();
            if !entry.file_type().await?.is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let bytes = tokio::fs::read(&path)
                .await
                .with_context(|| format!("reading hydration file {}", path.display()))?;
            let take = remaining.min(bytes.len());
            let content = String::from_utf8_lossy(&bytes[..take]);
            files.push(format!("### {name}\n{content}"));
            remaining -= take;
        }

        Ok(SourceResult {
            source: self.name().into(),
            kind: self.kind(),
            content: files.join("\n\n"),
            metadata: serde_json::json!({ "root": self.root, "max_bytes": max_bytes }),
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
