//! A typed native tool plus live public-event streaming.
//!
//! The scripted model calls `get_weather`, the SDK dispatches the in-process
//! handler through the agent loop's tool-dispatch arm (a first-class traced
//! effect — never a shell command), and the event stream shows the public
//! `tool.requested`/`tool.completed` lifecycle live.
//!
//! ```sh
//! cargo run -p agent-sdk --example typed_tool
//! ```

use agent_sdk::testing::ScriptedProvider;
use agent_sdk::{Agent, Runner, Tool};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let weather = Tool::new(
        "get_weather",
        "Current weather for a city.",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
        |arguments| async move {
            let city = arguments["city"].as_str().unwrap_or("nowhere").to_owned();
            Ok(json!({ "city": city, "forecast": "sunny", "temp_c": 21 }))
        },
    );

    let provider = ScriptedProvider::new()
        .tool_call("get_weather", json!({ "city": "san francisco" }))
        .text("It's sunny and 21C in San Francisco.");

    let agent = Agent::builder("mock-model")
        .name("weather-bot")
        .instructions("You are a weather assistant. Use the get_weather tool.")
        .tool(weather)
        .provider(Arc::new(provider))
        .trace_dir(std::env::temp_dir().join("agent-sdk-examples"))
        .build()?;

    let mut handle = Runner::start(&agent, "What's the weather in SF?").await?;
    let mut events = handle.events().expect("event stream");
    while let Some(event) = events.next().await {
        println!(
            "[{}] {}{}",
            event.event,
            event
                .attrs
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or(""),
            event
                .payload_preview
                .as_deref()
                .map(|preview| format!(" — {preview}"))
                .unwrap_or_default()
        );
    }
    let result = handle.wait().await?;

    println!();
    println!("text:  {}", result.text);
    println!("trace: {}", result.trace_path.display());
    Ok(())
}
