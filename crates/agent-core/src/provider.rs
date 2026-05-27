use crate::op::{ChatMessage, Model, Response, ResponseToolCall};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use reqwest::{header::RETRY_AFTER, Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

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

#[async_trait]
impl ChatProvider for ProviderClient {
    async fn chat(
        &self,
        model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        let url = format!("{}/chat/completions", self.config.url.trim_end_matches('/'));
        let body = json!({
            "model": model.0,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        });

        let mut delay = Duration::from_secs(1);
        for attempt in 0..3 {
            match self.send_chat_request(&url, &body).await {
                Ok(response) => return Ok(response),
                Err(err) if attempt < 2 && err.is_retryable() => {
                    tokio::time::sleep(err.retry_after().unwrap_or(delay)).await;
                    delay *= 2;
                }
                Err(err) => return Err(err.into_anyhow()),
            }
        }
        unreachable!("retry loop always returns")
    }
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
        Ok(Response {
            content: choice.message.content.unwrap_or_default(),
            tool_calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    let arguments: Value = serde_json::from_str(&call.function.arguments)
                        .unwrap_or_else(|_| json!({ "raw": call.function.arguments }));
                    ResponseToolCall::new(call.id, call.function.name, arguments)
                })
                .collect(),
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
            Self::Other(err) => err,
        }
    }
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
