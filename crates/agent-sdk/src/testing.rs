//! Test and example utilities: run agents without credentials or a live
//! provider.

use agent_core::provider::ToolSpec;
use agent_core::{ChatMessage, ChatProvider, FinishReason, Model, Response, ToolCall};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Mutex;

/// A provider that replays a fixed script of responses, one per model
/// call, in order — the same pattern agent-core's own loop tests use.
/// Inject it with [`crate::AgentBuilder::provider`] to run agents (and the
/// crate's examples) without credentials.
///
/// ```
/// use agent_sdk::testing::ScriptedProvider;
/// let provider = ScriptedProvider::new()
///     .tool_call("get_weather", serde_json::json!({ "city": "sf" }))
///     .text("It's sunny in SF.");
/// ```
#[derive(Default)]
pub struct ScriptedProvider {
    responses: Mutex<VecDeque<Response>>,
    scripted: usize,
}

impl ScriptedProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Script a final text response (no tool calls: ends the turn).
    pub fn text(self, content: impl Into<String>) -> Self {
        self.push(response(content.into(), Vec::new()))
    }

    /// Script a response that calls one tool. Tool-call ids are minted as
    /// `call-1`, `call-2`, ... in script order.
    pub fn tool_call(self, tool: impl Into<String>, arguments: Value) -> Self {
        let id = format!("call-{}", self.scripted + 1);
        self.push(response(
            String::new(),
            vec![ToolCall::new(id, tool, arguments)],
        ))
    }

    fn push(mut self, response: Response) -> Self {
        self.responses.get_mut().unwrap().push_back(response);
        self.scripted += 1;
        self
    }
}

fn response(content: String, tool_calls: Vec<ToolCall>) -> Response {
    Response {
        content,
        tool_calls,
        finish_reason: Some(FinishReason::Stop),
        input_tokens: 0,
        output_tokens: 1,
        total_tokens: 1,
        metadata: Default::default(),
    }
}

#[async_trait]
impl ChatProvider for ScriptedProvider {
    async fn chat(
        &self,
        _model: &Model,
        _tools: &[ToolSpec],
        _messages: &[ChatMessage],
    ) -> Result<Response> {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("scripted provider exhausted: no more responses queued"))
    }
}
