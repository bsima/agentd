use crate::op::{ChatMessage, Model, Response, ResponseToolCall};
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
const CONTINUE_NUDGE: &str =
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

impl Serialize for WireChatMessage<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let message = self.0;
        let mut fields = 2;
        if message.tool_call_id.is_some() {
            fields += 1;
        }
        if message.tool_calls.is_some() {
            fields += 1;
        }
        let mut state = serializer.serialize_struct("ChatMessage", fields)?;
        state.serialize_field("role", &message.role)?;
        state.serialize_field("content", &message.content)?;
        if let Some(tool_call_id) = &message.tool_call_id {
            state.serialize_field("tool_call_id", tool_call_id)?;
        }
        if let Some(tool_calls) = &message.tool_calls {
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
        let url = format!("{}/chat/completions", self.config.url.trim_end_matches('/'));
        chat_with_retries(model, tools, messages, |body| {
            let url = url.clone();
            async move { self.send_chat_request(&url, &body).await }
        })
        .await
        .map_err(ProviderError::into_anyhow)
    }
}

/// Build the JSON request body for a chat completion. When `nudge` is set, a
/// synthetic user continuation message is appended so that a *retry* after an
/// empty completion differs from the request that produced it.
///
/// Messages are serialized through [`WireChatMessage`] so internal stable ids
/// (used for GC/trace state) never reach the provider.
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

/// Drive the chat request through the bounded backoff loop.
///
/// On a retryable [`ProviderError::EmptyCompletion`], the *next* request is
/// mutated with [`CONTINUE_NUDGE`] rather than resending the identical body
/// (identical resends have not recovered in observed gpt-5.5 cases — t-1071).
/// If every attempt is exhausted the last error is returned so the run
/// terminates with a descriptive message instead of a silent empty completion.
async fn chat_with_retries<F, Fut>(
    model: &Model,
    tools: &[ToolSpec],
    messages: &[ChatMessage],
    mut send: F,
) -> std::result::Result<Response, ProviderError>
where
    F: FnMut(Value) -> Fut,
    Fut: std::future::Future<Output = std::result::Result<Response, ProviderError>>,
{
    let mut delay = Duration::from_secs(1);
    let mut nudge = false;
    for attempt in 0..MAX_ATTEMPTS {
        let body = build_chat_body(model, tools, messages, nudge);
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

impl ProviderClient {
    async fn send_chat_request(
        &self,
        url: &str,
        body: &Value,
    ) -> std::result::Result<Response, ProviderError> {
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
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
        let tokens = completion.usage.map(|u| u.total_tokens).unwrap_or_default();
        let content = choice.message.content.unwrap_or_default();
        let tool_calls: Vec<ResponseToolCall> = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| {
                let arguments: Value = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| json!({ "raw": call.function.arguments }));
                ResponseToolCall::new(call.id, call.function.name, arguments)
            })
            .collect();
        // Some providers (observed with gpt-5.5) occasionally return a 200 OK
        // with an empty completion: no content AND no tool calls. The agent
        // loop treats an empty tool_calls list as "the model is done", so such
        // a turn would be surfaced as a final, empty response
        // (`agent_complete{response:""}`), silently terminating an otherwise
        // active run. Treat this as a retryable error so the backoff loop
        // re-requests the turn (with a continuation nudge — t-1071).
        if content.trim().is_empty() && tool_calls.is_empty() {
            // Log the full raw body so we can confirm it is genuinely empty
            // rather than a parse/serialization bug on our side (t-1071).
            eprintln!("provider returned empty completion; raw response body: {text}");
            return Err(ProviderError::EmptyCompletion { raw: text });
        }
        Ok(Response {
            content,
            tool_calls,
            tokens,
        })
    }
}

#[derive(Debug)]
enum ProviderError {
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
    fn transport(source: reqwest::Error) -> Self {
        Self::Transport {
            source,
            context: "provider request failed",
        }
    }

    fn is_retryable(&self) -> bool {
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

    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Http { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    fn into_anyhow(self) -> anyhow::Error {
        match self {
            Self::Transport { source, context } => anyhow::Error::new(source).context(context),
            Self::Http { status, text, .. } => anyhow!("provider returned {status}: {text}"),
            Self::ContextOverflow { status, text } => {
                anyhow!("context_length_exceeded: provider returned {status}: {text}")
            }
            Self::EmptyCompletion { raw } => {
                anyhow!("provider returned an empty completion (no content or tool calls) after retries; raw response body: {raw}")
            }
            Self::Other(err) => err,
        }
    }
}

fn is_context_overflow(status: StatusCode, text: &str) -> bool {
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

fn retry_after_delay(response: &reqwest::Response) -> Option<Duration> {
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

#[derive(Debug, Deserialize)]
struct Usage {
    total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn ok(content: &str) -> Response {
        Response {
            content: content.into(),
            tool_calls: Vec::new(),
            tokens: 0,
        }
    }

    fn empty_err() -> ProviderError {
        ProviderError::EmptyCompletion {
            raw: r#"{"choices":[{"message":{"content":""}}]}"#.into(),
        }
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
                vec![ResponseToolCall::new(
                    "call_1",
                    "shell",
                    json!({"command": "git commit -m x"}),
                )],
            ),
            ChatMessage::tool("call_1", "committed"),
        ]
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
        let result = chat_with_retries(&model(), &[], &convo(), |_body| {
            *calls.borrow_mut() += 1;
            async { Err(empty_err()) }
        })
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
        let _ = chat_with_retries(&model(), &[], &convo(), |body| {
            bodies.borrow_mut().push(body);
            async { Err(empty_err()) }
        })
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
        let result = chat_with_retries(&model(), &[], &convo(), |body| {
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
        })
        .await;

        assert_eq!(*calls.borrow(), 2);
        assert_eq!(result.expect("should recover").content, "done");
    }

    // Non-retryable errors short-circuit immediately (no nudge, no extra calls).
    #[tokio::test(start_paused = true)]
    async fn non_retryable_error_is_not_retried() {
        let calls = RefCell::new(0usize);
        let result = chat_with_retries(&model(), &[], &convo(), |_body| {
            *calls.borrow_mut() += 1;
            async {
                Err(ProviderError::Http {
                    status: StatusCode::BAD_REQUEST,
                    text: "bad".into(),
                    retry_after: None,
                })
            }
        })
        .await;

        assert_eq!(*calls.borrow(), 1, "4xx must not be retried");
        assert!(result.is_err());
    }
}
