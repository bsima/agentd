use crate::op::{ChatMessage, FinishReason, Model, Response, ToolCall};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use reqwest::{header::RETRY_AFTER, Client, StatusCode};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

/// Number of attempts (initial + retries) the chat retry loop makes before
/// surfacing a terminal error.
const MAX_ATTEMPTS: usize = 3;

/// Synthetic continuation message appended to the *retry* request after the
/// provider returns an empty completion. gpt-5.5 has been observed returning a
/// 200 OK with empty content AND empty tool_calls *repeatably* for a given
/// context (most often the turn right after a tool result, e.g. the final
/// squash-commit). Resending the identical request hits the same wall, so the
/// retry mutates the request with this nudge to break the model out of the
/// empty-turn collapse. See t-1071.
pub(crate) const CONTINUE_NUDGE: &str =
    "Your previous response was empty. Continue the task: if work remains, issue \
     the next tool call; if the task is complete, say so explicitly.";

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub url: String,
    pub api_key: String,
    pub model: Model,
}

#[async_trait]
pub trait ChatProvider: Send + Sync {
    async fn chat(
        &self,
        model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response>;
}

#[derive(Clone)]
pub struct ProviderClient {
    client: Client,
    config: ProviderConfig,
}

impl ProviderClient {
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("building provider HTTP client"),
            config,
        }
    }

    pub fn model(&self) -> Model {
        self.config.model.clone()
    }
}

struct WireChatMessage<'a>(&'a ChatMessage);

/// Adapt a neutral [`ToolCall`] to the OpenAI chat-completions wire shape:
/// nested `function` object with the arguments JSON-encoded as a string.
fn tool_call_to_openai_wire(call: &crate::op::ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments.to_string(),
        },
    })
}

impl Serialize for WireChatMessage<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let message = self.0;
        // An empty tool_calls array is rejected by OpenAI-compatible
        // providers; omit the field entirely rather than sending [].
        let tool_calls: Option<Vec<Value>> = message
            .tool_calls
            .as_ref()
            .filter(|calls| !calls.is_empty())
            .map(|calls| calls.iter().map(tool_call_to_openai_wire).collect());
        let mut fields = 2;
        if message.tool_call_id.is_some() {
            fields += 1;
        }
        if tool_calls.is_some() {
            fields += 1;
        }
        let mut state = serializer.serialize_struct("ChatMessage", fields)?;
        state.serialize_field("role", &message.role)?;
        state.serialize_field("content", &message.content)?;
        if let Some(tool_call_id) = &message.tool_call_id {
            state.serialize_field("tool_call_id", tool_call_id)?;
        }
        if let Some(tool_calls) = &tool_calls {
            state.serialize_field("tool_calls", tool_calls)?;
        }
        state.end()
    }
}

#[async_trait]
impl ChatProvider for ProviderClient {
    async fn chat(
        &self,
        model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        openai_compatible_chat(
            &self.client,
            &self.config.url,
            &self.config.api_key,
            model,
            tools,
            messages,
        )
        .await
    }
}

/// Full OpenAI-compatible chat call: pending-tool-call guard, wire
/// adaptation, bounded retry/backoff, and the empty-completion nudge.
/// Shared by [`ProviderClient`] and the OAuth providers in `agent-oauth` so
/// every OpenAI-shaped path gets the same transport behavior.
pub async fn openai_compatible_chat(
    client: &Client,
    base_url: &str,
    bearer_token: &str,
    model: &Model,
    tools: &[ToolSpec],
    messages: &[ChatMessage],
) -> Result<Response> {
    if crate::op::has_pending_tool_calls(messages) {
        return Err(anyhow::anyhow!(
            "refusing to send malformed transcript to provider: assistant tool_call is missing a matching tool result; resume from a repaired checkpoint or reset the session"
        ));
    }
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    chat_with_retries(
        |nudge| build_chat_body(model, tools, messages, nudge),
        |body| {
            let url = url.clone();
            async move { send_chat_request(client, bearer_token, &url, &body).await }
        },
    )
    .await
    .map_err(ProviderError::into_anyhow)
}

/// Build the JSON request body for an OpenAI-compatible chat completion.
/// Shared by [`ProviderClient`] and the OAuth providers in `agent-oauth`.
///
/// Messages are serialized through [`WireChatMessage`] so internal stable ids
/// (used for GC/trace state) never reach the provider and neutral tool calls
/// are adapted to the OpenAI wire shape.
pub fn openai_chat_body(model: &Model, tools: &[ToolSpec], messages: &[ChatMessage]) -> Value {
    build_chat_body(model, tools, messages, false)
}

/// Like [`openai_chat_body`], but when `nudge` is set a synthetic user
/// continuation message is appended so that a *retry* after an empty
/// completion differs from the request that produced it.
fn build_chat_body(
    model: &Model,
    tools: &[ToolSpec],
    messages: &[ChatMessage],
    nudge: bool,
) -> Value {
    let nudge_msg = nudge.then(|| ChatMessage::user(CONTINUE_NUDGE));
    let mut wire_messages: Vec<WireChatMessage<'_>> =
        messages.iter().map(WireChatMessage).collect();
    if let Some(msg) = nudge_msg.as_ref() {
        wire_messages.push(WireChatMessage(msg));
    }
    json!({
        "model": model.0,
        "messages": wire_messages,
        "tools": tools,
        "tool_choice": "auto",
    })
}

/// Drive a chat request through the bounded backoff loop.
///
/// `build_body` receives a nudge flag: on a retryable
/// [`ProviderError::EmptyCompletion`], the *next* request is mutated with a
/// continuation nudge rather than resending the identical body (identical
/// resends have not recovered in observed gpt-5.5 cases — t-1071). If every
/// attempt is exhausted the last error is returned so the run terminates
/// with a descriptive message instead of a silent empty completion.
pub(crate) async fn chat_with_retries<B, F, Fut>(
    mut build_body: B,
    mut send: F,
) -> std::result::Result<Response, ProviderError>
where
    B: FnMut(bool) -> Value,
    F: FnMut(Value) -> Fut,
    Fut: std::future::Future<Output = std::result::Result<Response, ProviderError>>,
{
    let mut delay = Duration::from_secs(1);
    let mut nudge = false;
    for attempt in 0..MAX_ATTEMPTS {
        let body = build_body(nudge);
        match send(body).await {
            Ok(response) => return Ok(response),
            Err(err) if attempt + 1 < MAX_ATTEMPTS && err.is_retryable() => {
                // An empty completion is deterministic for a given context, so
                // mutate the next request to break the loop.
                if matches!(err, ProviderError::EmptyCompletion { .. }) {
                    nudge = true;
                }
                tokio::time::sleep(err.retry_after().unwrap_or(delay)).await;
                delay *= 2;
            }
            Err(err) => return Err(err),
        }
    }
    unreachable!("retry loop always returns")
}

async fn send_chat_request(
    client: &Client,
    bearer_token: &str,
    url: &str,
    body: &Value,
) -> std::result::Result<Response, ProviderError> {
    {
        let response = client
            .post(url)
            .bearer_auth(bearer_token)
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
                context: "reading provider response",
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

        let completion: ChatCompletion = serde_json::from_str(&text)
            .context("parsing provider response")
            .map_err(ProviderError::Other)?;
        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Other(anyhow!("provider returned no choices")))?;
        let usage = completion.usage.unwrap_or_default();
        let input_tokens = usage.prompt_tokens.unwrap_or_default();
        let output_tokens = usage.completion_tokens.unwrap_or_default();
        let total_tokens = usage
            .total_tokens
            .unwrap_or_else(|| input_tokens.saturating_add(output_tokens));
        let finish_reason = choice
            .finish_reason
            .as_deref()
            .map(FinishReason::from_provider);
        let content = choice.message.content.unwrap_or_default();
        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| {
                let arguments: Value = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| json!({ "raw": call.function.arguments }));
                ToolCall::new(call.id, call.function.name, arguments)
            })
            .collect();
        // Some providers (observed with gpt-5.5) occasionally return a 200 OK
        // with an empty completion: no content AND no tool calls. The agent
        // loop treats an empty tool_calls list as "the model is done", so such
        // a turn would be surfaced as a final, empty response
        // (`agent_complete{response:""}`), silently terminating an otherwise
        // active run. Treat this as a retryable error so the backoff loop
        // re-requests the turn (with a continuation nudge — t-1071).
        if content.trim().is_empty()
            && tool_calls.is_empty()
            && !matches!(finish_reason.as_ref(), Some(FinishReason::Stop))
        {
            // Log the full raw body so we can confirm it is genuinely empty
            // rather than a parse/serialization bug on our side (t-1071).
            tracing::warn!(raw_response = %text, "provider returned empty completion");
            eprintln!("provider returned empty completion; raw response body: {text}");
            return Err(ProviderError::EmptyCompletion { raw: text });
        }
        Ok(Response {
            content,
            tool_calls,
            finish_reason,
            input_tokens,
            output_tokens,
            total_tokens,
            metadata: Default::default(),
        })
    }
}

#[derive(Debug)]
pub(crate) enum ProviderError {
    Transport {
        source: reqwest::Error,
        context: &'static str,
    },
    Http {
        status: StatusCode,
        text: String,
        retry_after: Option<Duration>,
    },
    ContextOverflow {
        status: StatusCode,
        text: String,
    },
    /// Provider returned a 200 OK with neither content nor tool calls. Carries
    /// the raw response body for diagnostics.
    EmptyCompletion {
        raw: String,
    },
    Other(anyhow::Error),
}

impl ProviderError {
    pub(crate) fn transport(source: reqwest::Error) -> Self {
        Self::Transport {
            source,
            context: "provider request failed",
        }
    }

    pub(crate) fn is_retryable(&self) -> bool {
        match self {
            Self::Transport { .. } => true,
            Self::Http { status, .. } => {
                *status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
            }
            Self::ContextOverflow { .. } => false,
            Self::EmptyCompletion { .. } => true,
            Self::Other(_) => false,
        }
    }

    pub(crate) fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Http { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    pub(crate) fn into_anyhow(self) -> anyhow::Error {
        match self {
            Self::Transport { source, context } => anyhow::Error::new(source).context(context),
            Self::Http { status, text, .. } => anyhow!("provider returned {status}: {text}"),
            Self::ContextOverflow { status, text } => {
                anyhow::Error::new(ContextOverflowError { status, text })
            }
            Self::EmptyCompletion { raw } => {
                anyhow!("provider returned an empty completion (no content or tool calls) after retries; raw response body: {raw}")
            }
            Self::Other(err) => err,
        }
    }
}

/// Typed context-overflow error so interpreters can classify it without
/// string-matching (t-1151). The Display string keeps the historical
/// `context_length_exceeded:` prefix that traces and tooling key on.
#[derive(Debug)]
pub struct ContextOverflowError {
    pub status: StatusCode,
    pub text: String,
}

impl std::fmt::Display for ContextOverflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "context_length_exceeded: provider returned {}: {}",
            self.status, self.text
        )
    }
}

impl std::error::Error for ContextOverflowError {}

/// Classify a provider error as a context overflow. Typed when the provider
/// constructed [`ContextOverflowError`]; falls back to message heuristics for
/// providers that surface raw backend text (the codex OAuth path returns
/// `anyhow!("Codex OAuth provider returned {status}: {text}")` unclassified,
/// which is how smith's overflow escaped the t-1133 taxonomy entirely).
pub fn is_context_overflow_anyhow(err: &anyhow::Error) -> bool {
    if err
        .chain()
        .any(|cause| cause.downcast_ref::<ContextOverflowError>().is_some())
    {
        return true;
    }
    is_context_overflow_message(&format!("{err:#}"))
}

/// Message-level overflow heuristic covering observed provider phrasings:
/// OpenAI-compat (`context_length_exceeded`), the codex/Responses backend
/// ("your input exceeds the context window of this model"), and Anthropic
/// ("prompt is too long: N tokens > M maximum").
pub fn is_context_overflow_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("context_length_exceeded")
        || lower.contains("prompt is too long")
        || (lower.contains("context window")
            && (lower.contains("exceed") || lower.contains("too long")))
        || (lower.contains("context")
            && lower.contains("length")
            && (lower.contains("exceed") || lower.contains("maximum")))
}

pub(crate) fn is_context_overflow(status: StatusCode, text: &str) -> bool {
    if status != StatusCode::BAD_REQUEST {
        return false;
    }
    let lower = text.to_ascii_lowercase();
    lower.contains("context_length_exceeded")
        || (lower.contains("context")
            && (lower.contains("limit")
                || lower.contains("length")
                || lower.contains("too long")
                || lower.contains("maximum")))
}

pub(crate) fn retry_after_delay(response: &reqwest::Response) -> Option<Duration> {
    let header = response.headers().get(RETRY_AFTER)?.to_str().ok()?;
    let seconds = header.parse::<u64>().ok()?;
    Some(Duration::from_secs(seconds))
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunctionSpec,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunctionSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Deserialize)]
struct ChatCompletion {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AssistantMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ApiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ApiToolCall {
    id: String,
    function: ApiToolFunction,
}

#[derive(Debug, Deserialize)]
struct ApiToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Default, Deserialize)]
struct Usage {
    total_tokens: Option<u32>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn ok(content: &str) -> Response {
        Response {
            content: content.into(),
            tool_calls: Vec::new(),
            finish_reason: Some(FinishReason::Stop),
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            metadata: Default::default(),
        }
    }

    fn empty_err() -> ProviderError {
        ProviderError::EmptyCompletion {
            raw: r#"{"choices":[{"message":{"content":""}}]}"#.into(),
        }
    }

    #[test]
    fn typed_context_overflow_survives_anyhow_conversion() {
        let err = ProviderError::ContextOverflow {
            status: StatusCode::BAD_REQUEST,
            text: "context_length_exceeded".into(),
        }
        .into_anyhow();
        assert!(is_context_overflow_anyhow(&err));
        // Traces and tooling key on this prefix; keep it stable.
        assert!(err.to_string().starts_with("context_length_exceeded"));
    }

    #[test]
    fn raw_provider_messages_classify_as_overflow() {
        // The codex OAuth path surfaces the backend text unclassified —
        // exactly what the t-1145 smith overflow looked like.
        assert!(is_context_overflow_anyhow(&anyhow!(
            "Codex OAuth provider returned 400 Bad Request: \
             Your input exceeds the context window of this model."
        )));
        // Anthropic phrasing.
        assert!(is_context_overflow_message(
            "prompt is too long: 215000 tokens > 200000 maximum"
        ));
        // OpenAI-compat error code.
        assert!(is_context_overflow_message(
            "context_length_exceeded: provider returned 400: ..."
        ));
        assert!(!is_context_overflow_anyhow(&anyhow!(
            "provider returned 500: boom"
        )));
        assert!(!is_context_overflow_message(
            "shell command printed 'context' and 'length' is fine"
        ));
    }

    /// Count how many `user` messages a serialized request body carries. The
    /// nudge injected on retry is a synthetic `user` message, so this lets a
    /// test assert that the retry request was actually mutated.
    fn user_message_count(body: &Value) -> usize {
        body["messages"]
            .as_array()
            .map(|msgs| msgs.iter().filter(|m| m["role"] == "user").count())
            .unwrap_or(0)
    }

    fn nudged(body: &Value) -> bool {
        body["messages"]
            .as_array()
            .map(|msgs| {
                msgs.iter().any(|m| {
                    m["content"]
                        .as_str()
                        .map(|c| c.contains("previous response was empty"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    }

    fn model() -> Model {
        Model("gpt-5.5".into())
    }

    fn convo() -> Vec<ChatMessage> {
        vec![
            ChatMessage::user("do the task"),
            ChatMessage::assistant(
                None,
                vec![ToolCall::new(
                    "call_1",
                    "shell",
                    json!({"command": "git commit -m x"}),
                )],
            ),
            ChatMessage::tool("call_1", "committed"),
        ]
    }

    #[test]
    fn wire_messages_strip_ids_and_adapt_tool_calls_to_openai_shape() {
        let messages = vec![
            ChatMessage::assistant(
                Some("text".into()),
                vec![ToolCall::new("call_1", "shell", json!({"command": "pwd"}))],
            ),
            ChatMessage::tool("call_1", "ok"),
        ];

        let body = openai_chat_body(&model(), &[], &messages);

        let wire = body["messages"].as_array().unwrap();
        assert!(
            wire.iter().all(|message| message.get("id").is_none()),
            "internal ids must not reach the wire: {wire:?}"
        );
        assert_eq!(wire[0]["tool_calls"][0]["id"], json!("call_1"));
        assert_eq!(wire[0]["tool_calls"][0]["type"], json!("function"));
        assert_eq!(wire[0]["tool_calls"][0]["function"]["name"], json!("shell"));
        assert_eq!(
            wire[0]["tool_calls"][0]["function"]["arguments"],
            json!(r#"{"command":"pwd"}"#)
        );
        assert_eq!(wire[1]["tool_call_id"], json!("call_1"));
    }

    #[test]
    fn wire_messages_omit_empty_tool_calls_array() {
        // OpenAI-compatible providers reject assistant messages carrying an
        // empty tool_calls array, so the wire layer must drop the field.
        let mut message = ChatMessage::assistant(Some("text".into()), Vec::new());
        message.tool_calls = Some(Vec::new());

        let body = openai_chat_body(&model(), &[], &[message]);

        assert!(body["messages"][0].get("tool_calls").is_none());
    }

    #[test]
    fn empty_completion_is_retryable() {
        assert!(empty_err().is_retryable());
    }

    #[test]
    fn http_429_and_5xx_are_retryable_but_4xx_is_not() {
        let make = |status| ProviderError::Http {
            status,
            text: String::new(),
            retry_after: None,
        };
        assert!(make(StatusCode::TOO_MANY_REQUESTS).is_retryable());
        assert!(make(StatusCode::INTERNAL_SERVER_ERROR).is_retryable());
        assert!(!make(StatusCode::BAD_REQUEST).is_retryable());
    }

    #[tokio::test]
    async fn provider_refuses_dangling_tool_call_before_http() {
        let provider = ProviderClient::new(ProviderConfig {
            url: "http://127.0.0.1:1".into(),
            api_key: "test".into(),
            model: model(),
        });
        let messages = vec![
            ChatMessage::user("do the task"),
            ChatMessage::assistant(
                None,
                vec![ToolCall::new("call_1", "shell", json!({"command": "pwd"}))],
            ),
        ];

        let err = provider.chat(&model(), &[], &messages).await.unwrap_err();

        assert!(
            err.to_string()
                .contains("refusing to send malformed transcript"),
            "got: {err}"
        );
    }

    #[test]
    fn empty_completion_terminal_error_includes_raw_body() {
        let msg = empty_err().into_anyhow().to_string();
        assert!(msg.contains("empty completion"), "got: {msg}");
        // Raw body must be surfaced so we can confirm it is genuinely empty
        // rather than a parse bug (t-1071 acceptance criterion).
        assert!(msg.contains("raw response body"), "got: {msg}");
        assert!(msg.contains("choices"), "got: {msg}");
    }

    // Regression for t-1071: the empty-after-a-tool-call stall. The provider
    // returns empty on every attempt; the loop must exhaust its retries and
    // surface the descriptive terminal error rather than a silent empty
    // completion (`agent_complete{response:""}`).
    #[tokio::test(start_paused = true)]
    async fn exhausts_retries_then_returns_descriptive_error() {
        let calls = RefCell::new(0usize);
        let result = chat_with_retries(
            |nudge| build_chat_body(&model(), &[], &convo(), nudge),
            |_body| {
                *calls.borrow_mut() += 1;
                async { Err(empty_err()) }
            },
        )
        .await;

        assert_eq!(
            *calls.borrow(),
            MAX_ATTEMPTS,
            "must use the full retry budget"
        );
        let err = result.expect_err("all-empty must be terminal");
        let msg = err.into_anyhow().to_string();
        assert!(msg.contains("empty completion"), "got: {msg}");
    }

    // The core of t-1071: an identical resend does not recover, so the retry
    // must MUTATE the request with a continuation nudge.
    #[tokio::test(start_paused = true)]
    async fn retry_after_empty_injects_continuation_nudge() {
        let bodies: RefCell<Vec<Value>> = RefCell::new(Vec::new());
        let _ = chat_with_retries(
            |nudge| build_chat_body(&model(), &[], &convo(), nudge),
            |body| {
                bodies.borrow_mut().push(body);
                async { Err(empty_err()) }
            },
        )
        .await;

        let bodies = bodies.borrow();
        assert_eq!(bodies.len(), MAX_ATTEMPTS);
        // First request is the original context, unmutated.
        assert!(!nudged(&bodies[0]), "first attempt must not be nudged");
        let base_users = user_message_count(&bodies[0]);
        // Every retry after an empty completion carries the nudge.
        for body in &bodies[1..] {
            assert!(nudged(body), "retry must inject the continuation nudge");
            assert_eq!(
                user_message_count(body),
                base_users + 1,
                "nudge must be an added user message, not a replacement"
            );
        }
    }

    // A nudged retry that succeeds returns that response (the run continues).
    #[tokio::test(start_paused = true)]
    async fn nudged_retry_recovers() {
        let calls = RefCell::new(0usize);
        let result = chat_with_retries(
            |nudge| build_chat_body(&model(), &[], &convo(), nudge),
            |body| {
                let n = {
                    let mut c = calls.borrow_mut();
                    *c += 1;
                    *c
                };
                // First call empty; the nudged retry yields a real completion.
                let out = if n == 1 {
                    assert!(!nudged(&body));
                    Err(empty_err())
                } else {
                    assert!(nudged(&body), "recovery attempt should be nudged");
                    Ok(ok("done"))
                };
                async move { out }
            },
        )
        .await;

        assert_eq!(*calls.borrow(), 2);
        assert_eq!(result.expect("should recover").content, "done");
    }

    // Non-retryable errors short-circuit immediately (no nudge, no extra calls).
    #[tokio::test(start_paused = true)]
    async fn non_retryable_error_is_not_retried() {
        let calls = RefCell::new(0usize);
        let result = chat_with_retries(
            |nudge| build_chat_body(&model(), &[], &convo(), nudge),
            |_body| {
                *calls.borrow_mut() += 1;
                async {
                    Err(ProviderError::Http {
                        status: StatusCode::BAD_REQUEST,
                        text: "bad".into(),
                        retry_after: None,
                    })
                }
            },
        )
        .await;

        assert_eq!(*calls.borrow(), 1, "4xx must not be retried");
        assert!(result.is_err());
    }
}
