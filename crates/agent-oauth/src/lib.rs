use agent_core::provider::ToolSpec;
use agent_core::{ChatMessage, ChatProvider, Model, Response, ResponseToolCall};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthProviderKind {
    Codex,
    ClaudeCode,
}

impl OAuthProviderKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::ClaudeCode => "claude-code",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Codex => "openai-codex",
            Self::ClaudeCode => "anthropic",
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "codex" | "codex-oauth" | "openai-codex" => Ok(Self::Codex),
            "claude-code" | "claude-code-oauth" | "claude" | "anthropic" => Ok(Self::ClaudeCode),
            other => Err(anyhow!("unknown OAuth provider: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthToken {
    #[serde(rename = "access")]
    pub access_token: String,
    #[serde(rename = "refresh")]
    pub refresh_token: Option<String>,
    #[serde(rename = "expires", with = "expires_millis")]
    pub expires_at: Option<DateTime<Utc>>,
    #[serde(skip)]
    pub token_type: Option<String>,
}

impl OAuthToken {
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|expires_at| expires_at <= Utc::now() + Duration::minutes(1))
            .unwrap_or(false)
    }
}

mod expires_millis {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(
        value: &Option<DateTime<Utc>>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(value) => serializer.serialize_some(&value.timestamp_millis()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> std::result::Result<Option<DateTime<Utc>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = Option::<i64>::deserialize(deserializer)?;
        Ok(millis.and_then(|millis| Utc.timestamp_millis_opt(millis).single()))
    }
}

#[derive(Debug, Clone)]
pub struct TokenStatus {
    pub provider: String,
    pub path: PathBuf,
    pub present: bool,
    pub expires_at: Option<DateTime<Utc>>,
    pub expired: bool,
}

#[derive(Debug, Clone)]
pub struct TokenStore {
    provider: OAuthProviderKind,
    path: PathBuf,
}

type AuthFile = BTreeMap<String, OAuthToken>;

impl TokenStore {
    pub fn new(provider: OAuthProviderKind) -> Result<Self> {
        Ok(Self {
            path: credentials_file()?,
            provider,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn load(&self) -> Result<Option<OAuthToken>> {
        let auth = load_auth_file(&self.path).await?;
        Ok(auth.get(self.provider.key()).cloned())
    }

    pub async fn save(&self, token: &OAuthToken) -> Result<()> {
        let mut auth = load_auth_file(&self.path).await?;
        auth.insert(self.provider.key().into(), token.clone());
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let content = serde_json::to_vec_pretty(&auth)?;
        write_private(&self.path, &content).await
    }

    pub async fn status(&self) -> Result<TokenStatus> {
        let token = self.load().await?;
        Ok(TokenStatus {
            provider: self.provider.key().into(),
            path: self.path.clone(),
            present: token.is_some(),
            expires_at: token.as_ref().and_then(|token| token.expires_at),
            expired: token.as_ref().map(OAuthToken::is_expired).unwrap_or(false),
        })
    }
}

pub fn credentials_file() -> Result<PathBuf> {
    let data_home = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".local/share")))
        .ok_or_else(|| anyhow!("could not determine XDG data directory"))?;
    Ok(data_home.join("agent/auth.json"))
}

async fn load_auth_file(path: &Path) -> Result<AuthFile> {
    match tokio::fs::read_to_string(path).await {
        Ok(content) => serde_json::from_str(&content)
            .with_context(|| format!("parsing token store {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(AuthFile::new()),
        Err(err) => Err(err).with_context(|| format!("reading token store {}", path.display())),
    }
}

#[cfg(unix)]
async fn write_private(path: &Path, content: &[u8]) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    use tokio::io::AsyncWriteExt;

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .await
        .with_context(|| format!("opening token store {}", path.display()))?;
    file.write_all(content).await?;
    file.flush().await?;
    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).await?;
    Ok(())
}

#[cfg(not(unix))]
async fn write_private(path: &Path, content: &[u8]) -> Result<()> {
    tokio::fs::write(path, content)
        .await
        .with_context(|| format!("writing token store {}", path.display()))
}

#[async_trait]
pub trait OAuthProvider: ChatProvider {
    async fn refresh_token(&self) -> Result<OAuthToken>;
    async fn login(&self) -> Result<OAuthToken>;
    fn token_store(&self) -> &TokenStore;
}

#[derive(Clone)]
pub struct CodexOAuthProvider {
    inner: OAuthChatProvider,
}

impl CodexOAuthProvider {
    pub fn new(model: Model) -> Result<Self> {
        Ok(Self {
            inner: OAuthChatProvider::new(OAuthProviderKind::Codex, model)?,
        })
    }
}

#[derive(Clone)]
pub struct ClaudeCodeOAuthProvider {
    inner: OAuthChatProvider,
}

impl ClaudeCodeOAuthProvider {
    pub fn new(model: Model) -> Result<Self> {
        Ok(Self {
            inner: OAuthChatProvider::new(OAuthProviderKind::ClaudeCode, model)?,
        })
    }
}

#[derive(Clone)]
struct OAuthChatProvider {
    kind: OAuthProviderKind,
    client: Client,
    store: TokenStore,
    base_url: String,
}

impl OAuthChatProvider {
    fn new(kind: OAuthProviderKind, _model: Model) -> Result<Self> {
        let base_url = match kind {
            OAuthProviderKind::Codex => "https://api.openai.com/v1",
            OAuthProviderKind::ClaudeCode => "https://api.anthropic.com/v1",
        };
        Ok(Self {
            kind,
            client: Client::new(),
            store: TokenStore::new(kind)?,
            base_url: base_url.into(),
        })
    }

    async fn access_token(&self) -> Result<String> {
        let token = self.store.load().await?.ok_or_else(|| {
            anyhow!(
                "missing OAuth token for {}; run `agent auth login {}`",
                self.kind.key(),
                self.kind.name()
            )
        })?;
        if token.is_expired() {
            return Ok(self.refresh_token().await?.access_token);
        }
        Ok(token.access_token)
    }

    async fn refresh_token(&self) -> Result<OAuthToken> {
        let current = self.store.load().await?.ok_or_else(|| {
            anyhow!(
                "missing OAuth token for {}; run `agent auth login {}`",
                self.kind.key(),
                self.kind.name()
            )
        })?;
        let Some(refresh_token) = current.refresh_token else {
            return Err(anyhow!(
                "OAuth token for {} has no refresh token; run `agent auth login {}`",
                self.kind.key(),
                self.kind.name()
            ));
        };
        let token = self.exchange_refresh_token(&refresh_token).await?;
        self.store.save(&token).await?;
        Ok(token)
    }

    async fn login(&self) -> Result<OAuthToken> {
        let device = self.start_device_login().await?;
        eprintln!(
            "Open this URL to authorize {}: {}",
            self.kind.name(),
            device
                .verification_uri_complete
                .as_deref()
                .unwrap_or(&device.verification_uri)
        );
        if let Some(code) = &device.user_code {
            eprintln!("Enter code: {code}");
        }
        let token = self.poll_device_token(&device).await?;
        self.store.save(&token).await?;
        Ok(token)
    }

    async fn start_device_login(&self) -> Result<DeviceAuthorization> {
        let endpoint = match self.kind {
            OAuthProviderKind::Codex => "https://auth.openai.com/oauth/device/code",
            OAuthProviderKind::ClaudeCode => "https://claude.ai/oauth/device/code",
        };
        let response = self.client.post(endpoint).send().await;
        match response {
            Ok(response) if response.status().is_success() => response
                .json::<DeviceAuthorization>()
                .await
                .context("parsing OAuth device authorization"),
            Ok(response) => Err(anyhow!(
                "OAuth login for {} returned {}: {}",
                self.kind.name(),
                response.status(),
                response.text().await.unwrap_or_default()
            )),
            Err(err) => Err(err).context("starting OAuth login"),
        }
    }

    async fn poll_device_token(&self, device: &DeviceAuthorization) -> Result<OAuthToken> {
        let endpoint = match self.kind {
            OAuthProviderKind::Codex => "https://auth.openai.com/oauth/token",
            OAuthProviderKind::ClaudeCode => "https://claude.ai/oauth/token",
        };
        let interval = device.interval.unwrap_or(5);
        loop {
            let response = self
                .client
                .post(endpoint)
                .json(&json!({
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device.device_code,
                }))
                .send()
                .await
                .context("polling OAuth token")?;
            let status = response.status();
            let text = response.text().await?;
            if status.is_success() {
                return parse_token(&text);
            }
            if !text.contains("authorization_pending") && !text.contains("slow_down") {
                return Err(anyhow!("OAuth token polling returned {status}: {text}"));
            }
            tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        }
    }

    async fn exchange_refresh_token(&self, refresh_token: &str) -> Result<OAuthToken> {
        let endpoint = match self.kind {
            OAuthProviderKind::Codex => "https://auth.openai.com/oauth/token",
            OAuthProviderKind::ClaudeCode => "https://claude.ai/oauth/token",
        };
        let response = self
            .client
            .post(endpoint)
            .json(&json!({
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }))
            .send()
            .await
            .context("refreshing OAuth token")?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(anyhow!("OAuth refresh returned {status}: {text}"));
        }
        parse_token(&text)
    }
}

#[async_trait]
impl ChatProvider for OAuthChatProvider {
    async fn chat(
        &self,
        model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        let token = self.access_token().await?;
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = json!({
            "model": model.0,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        });
        let response = self
            .client
            .post(url)
            .bearer_auth(token)
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(anyhow!("OAuth provider returned {status}: {text}"));
        }
        parse_chat_response(&text)
    }
}

macro_rules! delegate_provider {
    ($ty:ty) => {
        #[async_trait]
        impl ChatProvider for $ty {
            async fn chat(
                &self,
                model: &Model,
                tools: &[ToolSpec],
                messages: &[ChatMessage],
            ) -> Result<Response> {
                self.inner.chat(model, tools, messages).await
            }
        }

        #[async_trait]
        impl OAuthProvider for $ty {
            async fn refresh_token(&self) -> Result<OAuthToken> {
                self.inner.refresh_token().await
            }

            async fn login(&self) -> Result<OAuthToken> {
                self.inner.login().await
            }

            fn token_store(&self) -> &TokenStore {
                &self.inner.store
            }
        }
    };
}

delegate_provider!(CodexOAuthProvider);
delegate_provider!(ClaudeCodeOAuthProvider);

pub async fn login(provider: OAuthProviderKind) -> Result<OAuthToken> {
    match provider {
        OAuthProviderKind::Codex => {
            CodexOAuthProvider::new(Model("gpt-5".into()))?
                .login()
                .await
        }
        OAuthProviderKind::ClaudeCode => {
            ClaudeCodeOAuthProvider::new(Model("claude-sonnet-4-5".into()))?
                .login()
                .await
        }
    }
}

pub async fn status_all() -> Result<Vec<TokenStatus>> {
    let mut statuses = Vec::new();
    for kind in [OAuthProviderKind::Codex, OAuthProviderKind::ClaudeCode] {
        statuses.push(TokenStore::new(kind)?.status().await?);
    }
    Ok(statuses)
}

pub fn provider_for_tag(tag: &str, model: Model) -> Result<Option<Box<dyn ChatProvider>>> {
    match tag {
        "codex-oauth" => Ok(Some(Box::new(CodexOAuthProvider::new(model)?))),
        "claude-code-oauth" => Ok(Some(Box::new(ClaudeCodeOAuthProvider::new(model)?))),
        _ => Ok(None),
    }
}

pub struct PkcePair {
    pub verifier: String,
    pub challenge: String,
}

pub fn generate_pkce() -> PkcePair {
    let verifier = base64url_no_pad(Uuid::new_v4().as_bytes());
    let mut challenge_bytes = *Uuid::new_v4().as_bytes();
    challenge_bytes[0] ^= verifier.as_bytes()[0];
    PkcePair {
        verifier: fixed_43(verifier),
        challenge: fixed_43(base64url_no_pad(&challenge_bytes)),
    }
}

pub fn anthropic_auth_url(pkce: &PkcePair) -> String {
    let redirect = "https://console.anthropic.com/oauth/code/callback";
    format!(
        "https://claude.ai/oauth/authorize?response_type=code&client_id=claude-code&redirect_uri={}&code_challenge={}&code_challenge_method=S256",
        percent_encode(redirect),
        pkce.challenge
    )
}

fn fixed_43(mut value: String) -> String {
    while value.len() < 43 {
        value.push('A');
    }
    value.truncate(43);
    value
}

fn base64url_no_pad(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        }
    }
    out
}

fn percent_encode(input: &str) -> String {
    let mut out = String::new();
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            byte => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[derive(Debug, Deserialize)]
struct DeviceAuthorization {
    device_code: String,
    user_code: Option<String>,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
    expires_at: Option<DateTime<Utc>>,
    token_type: Option<String>,
}

fn parse_token(text: &str) -> Result<OAuthToken> {
    let token: TokenResponse = serde_json::from_str(text).context("parsing OAuth token")?;
    Ok(OAuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at: token.expires_at.or_else(|| {
            token
                .expires_in
                .map(|seconds| Utc::now() + Duration::seconds(seconds))
        }),
        token_type: token.token_type,
    })
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

fn parse_chat_response(text: &str) -> Result<Response> {
    let completion: ChatCompletion =
        serde_json::from_str(text).context("parsing OAuth provider response")?;
    let choice = completion
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("OAuth provider returned no choices"))?;
    let tokens = completion
        .usage
        .map(|usage| usage.total_tokens)
        .unwrap_or_default();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_generation_has_expected_lengths() {
        let pkce = generate_pkce();
        assert_eq!(pkce.verifier.len(), 43);
        assert_eq!(pkce.challenge.len(), 43);
        assert_ne!(pkce.verifier, pkce.challenge);
    }

    #[test]
    fn credentials_file_returns_valid_path() -> Result<()> {
        let path = credentials_file()?;
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("auth.json")
        );
        Ok(())
    }

    #[test]
    fn anthropic_auth_url_encodes_redirect_uri() {
        let pkce = PkcePair {
            verifier: "v".repeat(43),
            challenge: "c".repeat(43),
        };
        let url = anthropic_auth_url(&pkce);
        assert!(url.contains(
            "redirect_uri=https%3A%2F%2Fconsole.anthropic.com%2Foauth%2Fcode%2Fcallback"
        ));
        assert!(!url.contains("redirect_uri=https://console.anthropic.com/oauth/code/callback"));
    }

    #[test]
    fn token_serializes_like_haskell_credentials() -> Result<()> {
        let token = OAuthToken {
            access_token: "access-token".into(),
            refresh_token: Some("refresh-token".into()),
            expires_at: Some(Utc.timestamp_millis_opt(1_234_567_890_000).unwrap()),
            token_type: Some("Bearer".into()),
        };
        let value = serde_json::to_value(&token)?;
        assert_eq!(
            value,
            json!({
                "access": "access-token",
                "refresh": "refresh-token",
                "expires": 1234567890000i64,
            })
        );
        Ok(())
    }
}
