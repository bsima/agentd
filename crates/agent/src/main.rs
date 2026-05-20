use agent_core::{
    agent_loop, standard_tools, ChatMessage, Event, ProviderClient, ProviderConfig, SeqConfig,
    TraceLogger,
};
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;
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
    prompt: String,
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

    let run_id = Uuid::new_v4().to_string();
    let trace_path = trace_path(&run_id)?;
    let trace = TraceLogger::new(run_id.clone(), trace_path.clone());
    let provider = ProviderClient::new(ProviderConfig {
        url,
        api_key,
        model: agent_core::Model(model),
    });
    let config = SeqConfig {
        provider: provider.clone(),
        tools: standard_tools(),
        trace: trace.clone(),
    };

    let system = "You are a standalone agent runner. Use tools when needed. For filesystem listing or shell tasks, call the bash tool. When finished, answer concisely.";
    let prompt = vec![ChatMessage::system(system), ChatMessage::user(args.prompt)];
    let program = agent_loop(provider.model(), prompt.clone(), 16);
    let (response, _) = agent_core::run_sequential(&config, prompt, program).await?;
    trace
        .emit(&Event::AgentDone {
            run_id,
            timestamp: chrono::Utc::now(),
        })
        .await?;

    println!("{}", response.content);
    eprintln!("trace: {}", trace_path.display());
    Ok(())
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
