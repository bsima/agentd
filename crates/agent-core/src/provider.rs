use crate::op::{ChatMessage, Model, Response, ResponseToolCall};
use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub url: String,
    pub api_key: String,
    pub model: Model,
}

#[derive(Clone)]
pub struct ProviderClient {
    client: Client,
    config: ProviderConfig,
}

impl ProviderClient {
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn model(&self) -> Model {
        self.config.model.clone()
    }

    pub async fn chat(
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

        let response = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .await
            .context("provider request failed")?;
        let status = response.status();
        let text = response.text().await.context("reading provider response")?;
        if !status.is_success() {
            return Err(anyhow!("provider returned {status}: {text}"));
        }

        let completion: ChatCompletion =
            serde_json::from_str(&text).context("parsing provider response")?;
        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("provider returned no choices"))?;
        let tokens = completion.usage.map(|u| u.total_tokens).unwrap_or_default();
        Ok(Response {
            content: choice.message.content.unwrap_or_default(),
            tool_calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| ResponseToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: serde_json::from_str(&call.function.arguments)
                        .unwrap_or_else(|_| json!({ "raw": call.function.arguments })),
                })
                .collect(),
            tokens,
        })
    }
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
