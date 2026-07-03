//! Structured output: attach a JSON Schema to the agent's final response.
//!
//! The first scripted answer is prose, so the loop appends a repair turn
//! quoting the validation errors; the second answer conforms and comes back
//! parsed and validated on `RunResult::output`.
//!
//! ```sh
//! cargo run -p agent-sdk --example output_schema
//! ```

use agent_sdk::testing::ScriptedProvider;
use agent_sdk::{Agent, Runner};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ScriptedProvider::new()
        .text("The answer to your question is forty-two!") // fails the schema
        .text(r#"{"answer": 42, "confidence": 0.99}"#); // repair turn conforms

    let agent = Agent::builder("mock-model")
        .instructions("Answer with JSON only.")
        .output_schema(json!({
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer": { "type": "integer" },
                "confidence": { "type": "number" }
            }
        }))
        .provider(Arc::new(provider))
        .trace_dir(std::env::temp_dir().join("agent-sdk-examples"))
        .build()?;

    let result = Runner::run(&agent, "What is the answer?").await?;

    let output = result.output.expect("schema set, so output is present");
    println!("validated output: {output}");
    println!("answer field:     {}", output["answer"]);
    println!("trace:            {}", result.trace_path.display());
    Ok(())
}
