//! Minimal one-shot run.
//!
//! Uses a scripted provider so it runs without credentials:
//!
//! ```sh
//! cargo run -p agent-sdk --example one_shot
//! ```
//!
//! For a real model, drop the `.provider(...)` line and pass a model alias
//! from `~/.config/agent/models.yaml` (with its API key configured or in
//! `AGENT_API_KEY`/`ANTHROPIC_API_KEY`/`OPENROUTER_API_KEY`).

use agent_sdk::testing::ScriptedProvider;
use agent_sdk::{Agent, Runner};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ScriptedProvider::new().text("Hello! (from the scripted model)");

    let agent = Agent::builder("mock-model")
        .name("hello-bot")
        .instructions("You are a terse assistant.")
        .provider(Arc::new(provider))
        .trace_dir(std::env::temp_dir().join("agent-sdk-examples"))
        .build()?;

    let result = Runner::run(&agent, "Say hello.").await?;

    println!("text:   {}", result.text);
    println!("run_id: {}", result.run_id);
    println!("trace:  {}", result.trace_path.display());
    Ok(())
}
