use agent_core::{
    agent_loop, standard_tools, ChatMessage, Event, ProviderClient, ProviderConfig, SeqConfig,
    TraceLogger,
};
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
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
    #[arg(long)]
    debug: bool,
    /// Read NUL-terminated session turns from this FIFO path.
    #[arg(long, env = "AGENT_FIFO")]
    fifo: Option<PathBuf>,
    /// Write a checkpoint JSON after each completed turn.
    #[arg(long, env = "AGENT_CHECKPOINT_DIR")]
    checkpoint_dir: Option<PathBuf>,
    /// Resume conversation history from a checkpoint JSON.
    #[arg(long, env = "AGENT_RESUME")]
    resume: Option<PathBuf>,
    /// One-shot prompt. Omit when using --fifo or NUL-framed stdin sessions.
    prompt: Option<String>,
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
    checkpoint_sequence: u64,
    history: Vec<ChatMessage>,
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let file_config = read_config(args.config.as_ref()).await?;
    let provider_file = file_config.provider.unwrap_or_default();

    let url = args
        .provider
        .or(provider_file.url)
        .or_else(|| std::env::var("AGENT_PROVIDER").ok())
        .or_else(|| std::env::var("OPENROUTER_BASE_URL").ok())
        .unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
    let model = args
        .model
        .or(provider_file.model)
        .or_else(|| std::env::var("AGENT_MODEL").ok())
        .unwrap_or_else(|| "openai/gpt-4o-mini".into());
    let api_key = args
        .key
        .or(provider_file.api_key)
        .or_else(|| std::env::var("AGENT_API_KEY").ok())
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .ok_or_else(|| {
            anyhow!("missing API key: pass --key or set AGENT_API_KEY/OPENROUTER_API_KEY")
        })?;

    let checkpoint = match args.resume.as_ref() {
        Some(path) => Some(load_checkpoint(path).await?),
        None => None,
    };
    let run_id = checkpoint
        .as_ref()
        .map(|cp| cp.run_id.clone())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let trace_path = trace_path(&run_id)?;
    let trace = TraceLogger::new(run_id.clone(), trace_path.clone());
    let provider = ProviderClient::new(ProviderConfig {
        url: url.clone(),
        api_key,
        model: agent_core::Model(model.clone()),
    });
    let config = SeqConfig {
        provider: Arc::new(provider),
        tools: standard_tools(),
        trace: trace.clone(),
    };

    let (history, checkpoint_sequence) = checkpoint.map_or_else(
        || (initial_history(), 0),
        |cp| {
            let sequence = cp.sequence;
            (cp.messages, sequence)
        },
    );
    let mut runtime = Runtime {
        config,
        trace,
        run_id: run_id.clone(),
        model: agent_core::Model(model.clone()),
        provider_url: url.clone(),
        trace_path: trace_path.clone(),
        checkpoint_dir: args.checkpoint_dir,
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

    match (args.prompt, args.fifo) {
        (Some(prompt), None) => run_one_shot(&mut runtime, prompt).await,
        (Some(_), Some(_)) => Err(anyhow!("provide either a prompt or --fifo, not both")),
        (None, Some(path)) => run_fifo_session(&mut runtime, path).await,
        (None, None) => run_stdin_session(&mut runtime).await,
    }
}

async fn run_one_shot(runtime: &mut Runtime, prompt: String) -> Result<()> {
    let response = run_turn(runtime, prompt).await?;
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
    let mut reader = BufReader::new(tokio::io::stdin());
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

async fn run_fifo_session(runtime: &mut Runtime, path: PathBuf) -> Result<()> {
    loop {
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .open(&path)
            .await
            .with_context(|| format!("opening fifo {}", path.display()))?;
        let mut reader = BufReader::new(file);
        loop {
            tokio::select! {
                frame = read_nul_frame(&mut reader) => {
                    match frame? {
                        Some(message) if message.is_empty() => return emit_done(runtime).await,
                        Some(message) => write_session_response(runtime, message).await?,
                        None => break,
                    }
                }
                _ = shutdown_signal() => return emit_done(runtime).await,
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

async fn write_session_response(runtime: &mut Runtime, message: String) -> Result<()> {
    let response = run_turn(runtime, message).await?;
    let mut stdout = tokio::io::stdout();
    stdout.write_all(response.content.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
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

async fn emit_done(runtime: &Runtime) -> Result<()> {
    runtime
        .trace
        .emit(&Event::AgentDone {
            run_id: runtime.run_id.clone(),
            timestamp: Utc::now(),
        })
        .await
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
    tokio::fs::write(dir.join("latest.json"), bytes).await?;
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

fn initial_history() -> Vec<ChatMessage> {
    vec![ChatMessage::system("You are a standalone agent runner. Use tools when needed. For filesystem listing or shell tasks, call the bash tool. When finished, answer concisely.")]
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
