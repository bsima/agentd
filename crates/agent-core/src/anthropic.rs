use crate::op::{ChatMessage, FinishReason, Model, Response, ToolCall};
use crate::provider::{
    chat_with_retries, is_context_overflow, retry_after_delay, ChatProvider, ProviderError,
    ToolSpec, CONTINUE_NUDGE,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8192;

#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: Model,
    /// Per-response output token cap. Defaults to `DEFAULT_MAX_TOKENS` when
    /// unset; configure via the model registry (`max_tokens` in models.yaml)
    /// for models that need longer completions.
    pub max_tokens: Option<u32>,
}

#[derive(Clone)]
pub struct AnthropicProvider {
    client: Client,
    config: AnthropicConfig,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("building Anthropic HTTP client"),
            config,
        }
    }

    pub fn model(&self) -> Model {
        self.config.model.clone()
    }
}

#[async_trait]
impl ChatProvider for AnthropicProvider {
    async fn chat(
        &self,
        model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        if crate::op::has_pending_tool_calls(messages) {
            return Err(anyhow::anyhow!(
                "refusing to send malformed transcript to provider: assistant tool_call is missing a matching tool result; resume from a repaired checkpoint or reset the session"
            ));
        }
        let url = format!("{}/messages", self.config.base_url.trim_end_matches('/'));
        let max_tokens = self.config.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        chat_with_retries(
            |nudge| {
                if nudge {
                    let mut messages = messages.to_vec();
                    messages.push(ChatMessage::user(CONTINUE_NUDGE));
                    build_messages_body(model, tools, &messages, max_tokens)
                } else {
                    build_messages_body(model, tools, messages, max_tokens)
                }
            },
            |body| {
                let url = url.clone();
                async move { self.send_messages_request(&url, &body).await }
            },
        )
        .await
        .map_err(ProviderError::into_anyhow)
    }
}

impl AnthropicProvider {
    async fn send_messages_request(
        &self,
        url: &str,
        body: &Value,
    ) -> std::result::Result<Response, ProviderError> {
        let response = self
            .client
            .post(url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(body)
            .send()
            .await
            .map_err(ProviderError::transport)?;
        let status = response.status();
        let retry_after = retry_after_delay(&response);
        let text = response
            .text()
            .await
            .map_err(|source| ProviderError::Transport {
                source,
                context: "reading Anthropic response",
            })?;
        if !status.is_success() {
            if is_context_overflow(status, &text) {
                return Err(ProviderError::ContextOverflow { status, text });
            }
            return Err(ProviderError::Http {
                status,
                text,
                retry_after,
            });
        }
        let response = parse_messages_response(&text).map_err(ProviderError::Other)?;
        // Same guard as the OpenAI-compatible path (t-1071): a non-stop turn
        // with neither text nor tool_use blocks would silently terminate an
        // active run. Retry it (with a continuation nudge) instead.
        if response.content.trim().is_empty()
            && response.tool_calls.is_empty()
            && !matches!(response.finish_reason.as_ref(), Some(FinishReason::Stop))
        {
            tracing::warn!(raw_response = %text, "provider returned empty completion");
            return Err(ProviderError::EmptyCompletion { raw: text });
        }
        Ok(response)
    }
}

fn build_messages_body(
    model: &Model,
    tools: &[ToolSpec],
    messages: &[ChatMessage],
    max_tokens: u32,
) -> Value {
    let system = messages
        .iter()
        .filter(|message| message.role == "system")
        .filter_map(|message| message.content.as_deref())
        .collect::<Vec<_>>()
        .join("\n\n");

    let anthropic_messages: Vec<Value> = messages
        .iter()
        .filter(|message| message.role != "system")
        .map(message_to_anthropic)
        .collect();

    let mut body = json!({
        "model": model.0,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    });
    if !system.is_empty() {
        // The system prompt is the long stable prefix of every request in a
        // session - mark it cacheable so repeated turns hit the prompt cache.
        body["system"] = json!([{
            "type": "text",
            "text": system,
            "cache_control": { "type": "ephemeral" },
        }]);
    }
    if !tools.is_empty() {
        body["tools"] = json!(tools.iter().map(tool_to_anthropic).collect::<Vec<_>>());
        body["tool_choice"] = json!({ "type": "auto" });
    }
    body
}

fn message_to_anthropic(message: &ChatMessage) -> Value {
    match message.role.as_str() {
        "assistant" => {
            let mut content = Vec::new();
            if let Some(text) = message.content.as_deref().filter(|text| !text.is_empty()) {
                content.push(json!({ "type": "text", "text": text }));
            }
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                content.push(json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }));
            }
            json!({ "role": "assistant", "content": content })
        }
        "tool" => json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": message.tool_call_id.as_deref().unwrap_or_default(),
                "content": message.content.as_deref().unwrap_or_default(),
            }]
        }),
        _ => json!({
            "role": "user",
            "content": [{
                "type": "text",
                "text": message.content.as_deref().unwrap_or_default(),
            }]
        }),
    }
}

fn tool_to_anthropic(tool: &ToolSpec) -> Value {
    json!({
        "name": tool.function.name,
        "description": tool.function.description,
        "input_schema": tool.function.parameters,
    })
}

fn parse_messages_response(text: &str) -> Result<Response> {
    let response: AnthropicResponse =
        serde_json::from_str(text).context("parsing Anthropic response")?;
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    for block in response.content {
        match block {
            AnthropicContentBlock::Text { text } => content.push_str(&text),
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall::new(id, name, input));
            }
            AnthropicContentBlock::Other => {}
        }
    }
    Ok(Response {
        content,
        tool_calls,
        finish_reason: response
            .stop_reason
            .as_deref()
            .map(FinishReason::from_provider),
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        total_tokens: response
            .usage
            .input_tokens
            .saturating_add(response.usage.output_tokens),
        metadata: Default::default(),
    })
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ToolFunctionSpec, ToolSpec};

    #[test]
    fn parses_text_and_tool_use_response() -> Result<()> {
        let response = parse_messages_response(
            r#"{
              "id": "msg_123",
              "type": "message",
              "role": "assistant",
              "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "toolu_123", "name": "lookup", "input": {"query": "rust"}}
              ],
              "model": "claude-opus-4-8",
              "stop_reason": "tool_use",
              "usage": {"input_tokens": 10, "output_tokens": 5}
            }"#,
        )?;

        assert_eq!(response.content, "Hello");
        assert_eq!(response.input_tokens, 10);
        assert_eq!(response.output_tokens, 5);
        assert_eq!(response.total_tokens, 15);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "toolu_123");
        assert_eq!(response.tool_calls[0].name, "lookup");
        assert_eq!(response.tool_calls[0].arguments, json!({"query": "rust"}));
        Ok(())
    }

    #[test]
    fn maps_messages_to_anthropic_body() {
        let messages = vec![
            ChatMessage::system("system one"),
            ChatMessage::system("system two"),
            ChatMessage::user("hello"),
            ChatMessage::tool("toolu_123", "tool output"),
        ];
        let tools = vec![ToolSpec {
            kind: "function".into(),
            function: ToolFunctionSpec {
                name: "lookup".into(),
                description: "Lookup things".into(),
                parameters: json!({"type": "object", "properties": {"query": {"type": "string"}}}),
            },
        }];

        let body = build_messages_body(
            &Model("claude-opus-4-8".into()),
            &tools,
            &messages,
            DEFAULT_MAX_TOKENS,
        );

        assert_eq!(body["model"], "claude-opus-4-8");
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(
            body["system"],
            json!([{
                "type": "text",
                "text": "system one\n\nsystem two",
                "cache_control": { "type": "ephemeral" },
            }])
        );
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_123", "content": "tool output"}]}
            ])
        );
        assert_eq!(
            body["tools"],
            json!([{"name": "lookup", "description": "Lookup things", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}])
        );
        assert_eq!(body["tool_choice"], json!({"type": "auto"}));
    }
}
