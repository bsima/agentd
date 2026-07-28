use crate::op::{ChatMessage, FinishReason, Model, Response, ToolCall};
use crate::provider::{
    chat_with_retries, is_context_overflow, is_model_not_found, retry_after_delay, ChatProvider,
    ProviderError, TextDeltaFn, ToolSpec, CONTINUE_NUDGE,
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

    async fn chat_streamed(
        &self,
        model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
        on_delta: &TextDeltaFn,
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
                let mut body = if nudge {
                    let mut messages = messages.to_vec();
                    messages.push(ChatMessage::user(CONTINUE_NUDGE));
                    build_messages_body(model, tools, &messages, max_tokens)
                } else {
                    build_messages_body(model, tools, messages, max_tokens)
                };
                body["stream"] = json!(true);
                body
            },
            |body| {
                let url = url.clone();
                async move {
                    self.send_messages_request_streamed(&url, &body, on_delta)
                        .await
                }
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
            if is_model_not_found(status, &text) {
                return Err(ProviderError::ModelNotFound { status, text });
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

    async fn send_messages_request_streamed(
        &self,
        url: &str,
        body: &Value,
        on_delta: &TextDeltaFn,
    ) -> std::result::Result<Response, ProviderError> {
        use futures::StreamExt;
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
        // Errors arrive as a non-2xx status before any SSE bytes; the
        // classification matches the buffered path exactly.
        if !status.is_success() {
            let retry_after = retry_after_delay(&response);
            let text = response
                .text()
                .await
                .map_err(|source| ProviderError::Transport {
                    source,
                    context: "reading Anthropic response",
                })?;
            if is_context_overflow(status, &text) {
                return Err(ProviderError::ContextOverflow { status, text });
            }
            if is_model_not_found(status, &text) {
                return Err(ProviderError::ModelNotFound { status, text });
            }
            return Err(ProviderError::Http {
                status,
                text,
                retry_after,
            });
        }

        let mut decoder = crate::sse::SseDecoder::new();
        let mut stream = response.bytes_stream();
        let mut accum = AnthropicStreamAccum::default();
        'outer: while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|source| ProviderError::Transport {
                source,
                context: "reading Anthropic stream",
            })?;
            for event in decoder.feed(&chunk) {
                let parsed: AnthropicStreamEvent = serde_json::from_str(&event.data)
                    .context("parsing Anthropic stream event")
                    .map_err(ProviderError::Other)?;
                if accum.on_event(parsed, on_delta)? {
                    break 'outer;
                }
            }
        }
        accum.into_response()
    }
}

/// Accumulates the documented `/v1/messages` SSE event sequence into the
/// same [`Response`] that [`parse_messages_response`] builds from the
/// buffered body, forwarding `text_delta` fragments through `on_delta`.
#[derive(Default)]
struct AnthropicStreamAccum {
    /// `message_stop` arrived. EOF without it means the stream was
    /// truncated mid-response; partial content is never accepted.
    stopped: bool,
    content: String,
    /// Open/finished tool_use blocks keyed by content-block index; the
    /// `input_json_delta` fragments buffer until `content_block_stop`.
    tool_blocks: std::collections::BTreeMap<usize, (String, String, String)>,
    stop_reason: Option<String>,
    input_tokens: u32,
    output_tokens: u32,
    cached_input_tokens: Option<u32>,
}

impl AnthropicStreamAccum {
    /// Absorb one stream event; returns `true` on `message_stop`.
    fn on_event(
        &mut self,
        event: AnthropicStreamEvent,
        on_delta: &TextDeltaFn,
    ) -> std::result::Result<bool, ProviderError> {
        match event {
            AnthropicStreamEvent::MessageStart { message } => {
                self.input_tokens = message.usage.input_tokens;
                self.cached_input_tokens = message.usage.cache_read_input_tokens;
            }
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if let AnthropicStreamBlock::ToolUse { id, name } = content_block {
                    self.tool_blocks.insert(index, (id, name, String::new()));
                }
            }
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
                AnthropicStreamDelta::TextDelta { text } => {
                    if !text.is_empty() {
                        on_delta(&text);
                        self.content.push_str(&text);
                    }
                }
                AnthropicStreamDelta::InputJsonDelta { partial_json } => {
                    if let Some((_, _, buffer)) = self.tool_blocks.get_mut(&index) {
                        buffer.push_str(&partial_json);
                    }
                }
                AnthropicStreamDelta::Other => {}
            },
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = delta.stop_reason {
                    self.stop_reason = Some(reason);
                }
                if let Some(usage) = usage {
                    self.output_tokens = usage.output_tokens;
                }
            }
            AnthropicStreamEvent::MessageStop => {
                self.stopped = true;
                return Ok(true);
            }
            AnthropicStreamEvent::Error { error } => {
                let text = error.to_string();
                // Mid-stream overloads are retryable per the API docs; the
                // 529-shaped Http variant routes them through the existing
                // backoff loop.
                if text.contains("overloaded_error") {
                    return Err(ProviderError::Http {
                        status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                        text,
                        retry_after: None,
                    });
                }
                return Err(ProviderError::Other(anyhow::anyhow!(
                    "Anthropic stream error event: {text}"
                )));
            }
            AnthropicStreamEvent::ContentBlockStop { .. } | AnthropicStreamEvent::Other => {}
        }
        Ok(false)
    }

    fn into_response(self) -> std::result::Result<Response, ProviderError> {
        if !self.stopped {
            return Err(ProviderError::TruncatedStream {
                context: "Anthropic stream ended before message_stop",
            });
        }
        let tool_calls: Vec<ToolCall> = self
            .tool_blocks
            .into_values()
            .map(|(id, name, buffer)| {
                // Same convention as the buffered parser: an unparsable
                // argument buffer degrades to `{"raw": ...}`; an empty one
                // (tool_use with no input deltas) is an empty object.
                let arguments: Value = if buffer.trim().is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&buffer).unwrap_or_else(|_| json!({ "raw": buffer }))
                };
                ToolCall::new(id, name, arguments)
            })
            .collect();
        let finish_reason = self.stop_reason.as_deref().map(FinishReason::from_provider);
        if self.content.trim().is_empty()
            && tool_calls.is_empty()
            && !matches!(finish_reason.as_ref(), Some(FinishReason::Stop))
        {
            tracing::warn!("Anthropic stream ended with empty completion");
            return Err(ProviderError::EmptyCompletion {
                raw: "<streamed response with no content or tool_use blocks>".into(),
            });
        }
        Ok(Response {
            content: self.content,
            tool_calls,
            finish_reason,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            total_tokens: self.input_tokens.saturating_add(self.output_tokens),
            cached_input_tokens: self.cached_input_tokens,
            cost_micro_usd: None,
            pricing: None,
            metadata: Default::default(),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicStreamMessage },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: AnthropicStreamBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: AnthropicStreamDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {},
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: Option<AnthropicDeltaUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error { error: Value },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamMessage {
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamBlock {
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageDelta {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDeltaUsage {
    output_tokens: u32,
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
        cached_input_tokens: response.usage.cache_read_input_tokens,
        // Cost is stamped at trace-emission time (crate::cost), not here.
        cost_micro_usd: None,
        pricing: None,
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
    /// Prompt-cache read tokens; recorded when the API reports them,
    /// never fabricated (t-1334).
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
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
    fn stream_accumulator_rebuilds_the_buffered_response() -> Result<()> {
        use std::sync::{Arc, Mutex};
        let events = [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":4}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"rust\"}"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let on_delta: TextDeltaFn = Arc::new(move |text: &str| {
            sink.lock().unwrap().push(text.to_owned());
        });
        let mut accum = AnthropicStreamAccum::default();
        let mut stopped = false;
        for event in events {
            let parsed: AnthropicStreamEvent = serde_json::from_str(event)?;
            if accum
                .on_event(parsed, &on_delta)
                .map_err(|e| e.into_anyhow())?
            {
                stopped = true;
                break;
            }
        }
        assert!(stopped, "message_stop terminates the stream");
        assert_eq!(*seen.lock().unwrap(), vec!["Hel".to_owned(), "lo".into()]);
        let response = accum.into_response().map_err(|e| e.into_anyhow())?;
        assert_eq!(response.content, "Hello");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].arguments, json!({"query": "rust"}));
        assert_eq!(response.input_tokens, 10);
        assert_eq!(response.output_tokens, 5);
        assert_eq!(response.total_tokens, 15);
        assert_eq!(response.cached_input_tokens, Some(4));
        Ok(())
    }

    #[test]
    fn truncated_stream_without_message_stop_is_rejected() {
        use std::sync::Arc;
        let events = [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial answ"}}"#,
            // EOF here: no message_delta, no message_stop.
        ];
        let on_delta: TextDeltaFn = Arc::new(|_| {});
        let mut accum = AnthropicStreamAccum::default();
        for event in events {
            let parsed: AnthropicStreamEvent = serde_json::from_str(event).unwrap();
            assert!(!accum.on_event(parsed, &on_delta).unwrap());
        }
        let err = accum.into_response().map(|_| ()).unwrap_err();
        assert!(
            err.is_retryable(),
            "a cut stream is a transport interruption, not a terminal failure"
        );
        assert!(matches!(err, ProviderError::TruncatedStream { .. }));
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
