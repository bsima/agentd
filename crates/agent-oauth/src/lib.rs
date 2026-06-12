use agent_core::provider::ToolSpec;
use agent_core::{ChatMessage, ChatProvider, FinishReason, Model, Response, ToolCall};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rand::RngCore;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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

    /// A store backed by an explicit file instead of the XDG default.
    /// Lets tests exercise import/save semantics without touching the
    /// process environment or the real credential file.
    pub fn with_path(provider: OAuthProviderKind, path: PathBuf) -> Self {
        Self { provider, path }
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
            OAuthProviderKind::Codex => "https://chatgpt.com/backend-api",
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
        let token = match self.store.load().await? {
            Some(token) => token,
            // No stored codex token: fall through to the Codex CLI's
            // session before failing — avoids token drift between the two
            // tools and makes a fresh `codex login` just work (t-1170).
            None if self.kind == OAuthProviderKind::Codex => {
                let source = codex_cli_auth_file()?;
                import_codex_cli_from(&source, &self.store)
                    .await
                    .with_context(|| {
                        format!(
                            "missing OAuth token for {} and no Codex CLI session to import",
                            self.kind.key()
                        )
                    })?
            }
            None => {
                return Err(anyhow!(
                    "missing OAuth token for {}; run `agent auth login {}`",
                    self.kind.key(),
                    self.kind.name()
                ))
            }
        };
        if token.is_expired() {
            let refreshed = self.refresh_token().await;
            return match refreshed {
                Ok(token) => Ok(token.access_token),
                // Refresh failed: the Codex CLI may hold a fresher session
                // (the user refreshes it whenever they use the CLI).
                Err(refresh_err) if self.kind == OAuthProviderKind::Codex => {
                    let source = codex_cli_auth_file()?;
                    match import_codex_cli_from(&source, &self.store).await {
                        Ok(imported) if !imported.is_expired() => Ok(imported.access_token),
                        _ => Err(refresh_err),
                    }
                }
                Err(err) => Err(err),
            };
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
        if self.kind == OAuthProviderKind::Codex {
            // The OpenAI device-code endpoint this command used to call
            // returns 404 (t-1170). The supported path is the Codex CLI's
            // own login flow; we import its session rather than guessing
            // at unverified endpoints and client params.
            let source = codex_cli_auth_file()?;
            let token = import_codex_cli_from(&source, &self.store).await?;
            eprintln!(
                "imported Codex CLI credentials from {} into {}",
                source.display(),
                self.store.path().display()
            );
            return Ok(token);
        }
        tracing::info!(provider = self.kind.name(), "starting OAuth device login");
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
        tracing::info!(provider = self.kind.name(), "OAuth device login completed");
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
        if agent_core::has_pending_tool_calls(messages) {
            return Err(anyhow!(
                "refusing to send malformed transcript to provider: assistant tool_call is missing a matching tool result; resume from a repaired checkpoint or reset the session"
            ));
        }
        let token = self.access_token().await?;
        match self.kind {
            OAuthProviderKind::Codex => self.chat_codex(&token, model, tools, messages).await,
            OAuthProviderKind::ClaudeCode => {
                self.chat_openai_compatible(&token, model, tools, messages)
                    .await
            }
        }
    }
}

impl OAuthChatProvider {
    async fn chat_openai_compatible(
        &self,
        token: &str,
        model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        // Shared transport: wire adaptation (no internal ids), bounded
        // retry/backoff, and the empty-completion nudge — identical to the
        // API-key provider path.
        agent_core::provider::openai_compatible_chat(
            &self.client,
            &self.base_url,
            token,
            model,
            tools,
            messages,
        )
        .await
    }

    async fn chat_codex(
        &self,
        token: &str,
        model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        let account_id = extract_codex_account_id(token).ok_or_else(|| {
            anyhow!("failed to extract chatgpt_account_id from OpenAI Codex token")
        })?;
        let url = format!("{}/codex/responses", self.base_url.trim_end_matches('/'));
        let body = build_codex_request(model, tools, messages);
        let response = self
            .client
            .post(url)
            .bearer_auth(token)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", "codex_cli_rs")
            .header("chatgpt-account-id", account_id)
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(anyhow!("Codex OAuth provider returned {status}: {text}"));
        }
        parse_codex_sse_response(&text)
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

/// Where the Codex CLI keeps its credentials: `$CODEX_HOME/auth.json`,
/// defaulting to `~/.codex/auth.json`.
pub fn codex_cli_auth_file() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(home).join("auth.json"));
    }
    dirs::home_dir()
        .map(|home| home.join(".codex/auth.json"))
        .ok_or_else(|| anyhow!("could not determine home directory"))
}

/// Import the Codex CLI's credentials (`codex login` writes them) into the
/// agent token store under `openai-codex`, preserving every other entry.
/// This replaced device-code login for codex (t-1170): the device endpoint
/// agentd used returned 404, and reusing the CLI's session avoids token
/// drift between the two tools anyway.
pub async fn import_codex_cli() -> Result<OAuthToken> {
    let source = codex_cli_auth_file()?;
    let store = TokenStore::new(OAuthProviderKind::Codex)?;
    import_codex_cli_from(&source, &store).await
}

pub async fn import_codex_cli_from(source: &Path, store: &TokenStore) -> Result<OAuthToken> {
    let raw = match tokio::fs::read_to_string(source).await {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(anyhow!(
                "no Codex CLI credentials at {}; run `codex login` first, then `agent auth import codex`",
                source.display()
            ));
        }
        Err(err) => {
            return Err(err)
                .with_context(|| format!("reading Codex CLI auth file {}", source.display()))
        }
    };
    let token = token_from_codex_cli(&raw)
        .with_context(|| format!("parsing Codex CLI auth file {}", source.display()))?;
    store.save(&token).await?;
    Ok(token)
}

/// Parse the Codex CLI auth.json shape: `tokens.access_token` and
/// `tokens.refresh_token`, with expiry derived from the access token's JWT
/// `exp` claim (the CLI file does not store an expiry of its own).
fn token_from_codex_cli(raw: &str) -> Result<OAuthToken> {
    let value: Value = serde_json::from_str(raw).context("decoding JSON")?;
    let tokens = value
        .get("tokens")
        .ok_or_else(|| anyhow!("missing `tokens` object"))?;
    let access_token = tokens
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing `tokens.access_token`"))?
        .to_string();
    let refresh_token = tokens
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(OAuthToken {
        expires_at: extract_jwt_expiry(&access_token),
        access_token,
        refresh_token,
        token_type: None,
    })
}

/// The `exp` claim (seconds since epoch) from a JWT's payload, if any.
fn extract_jwt_expiry(token: &str) -> Option<DateTime<Utc>> {
    let payload = token.split('.').nth(1)?;
    let decoded = base64_url_decode(payload)?;
    let value: Value = serde_json::from_slice(&decoded).ok()?;
    let exp = value.get("exp")?.as_i64()?;
    Utc.timestamp_opt(exp, 0).single()
}

pub fn provider_base_url_for_tag(tag: &str) -> Option<&'static str> {
    match tag {
        "openai-codex" | "codex-oauth" => Some("https://chatgpt.com/backend-api"),
        "claude-code" | "claude-code-oauth" => Some("https://api.anthropic.com/v1"),
        _ => None,
    }
}

pub fn provider_for_tag(tag: &str, model: Model) -> Result<Option<Box<dyn ChatProvider>>> {
    match tag {
        "openai-codex" | "codex-oauth" => Ok(Some(Box::new(CodexOAuthProvider::new(model)?))),
        "claude-code" | "claude-code-oauth" => {
            Ok(Some(Box::new(ClaudeCodeOAuthProvider::new(model)?)))
        }
        _ => Ok(None),
    }
}

pub struct PkcePair {
    pub verifier: String,
    pub challenge: String,
}

pub fn generate_pkce() -> PkcePair {
    let mut verifier_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut verifier_bytes);
    let verifier = base64url_no_pad(&verifier_bytes);
    let challenge = base64url_no_pad(&Sha256::digest(verifier.as_bytes()));
    PkcePair {
        verifier,
        challenge,
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

const CODEX_DEFAULT_INSTRUCTIONS: &str =
    "You are Codex, based on GPT-5. You are running as a coding agent in a CLI harness.";

fn build_codex_request(model: &Model, tools: &[ToolSpec], messages: &[ChatMessage]) -> Value {
    let mut input = Vec::new();
    let mut system = Vec::new();
    for message in messages {
        if message.role == "system" {
            if let Some(content) = message
                .content
                .as_deref()
                .filter(|content| !content.is_empty())
            {
                system.push(content);
            }
        }
    }
    if !system.is_empty() {
        input.extend(codex_message("developer", &system.join("\n\n")));
    }
    for message in messages {
        if message.role != "system" {
            input.extend(message_to_codex_input(message));
        }
    }

    let mut body = json!({
        "model": model.0,
        "instructions": CODEX_DEFAULT_INSTRUCTIONS,
        "input": input,
        "stream": true,
        "store": false,
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools.iter().map(tool_spec_to_codex).collect());
    }
    body
}

fn message_to_codex_input(message: &ChatMessage) -> Vec<Value> {
    match message.role.as_str() {
        "user" => codex_message("user", message.content.as_deref().unwrap_or_default()),
        "assistant" => {
            let mut items =
                codex_message("assistant", message.content.as_deref().unwrap_or_default());
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                items.push(json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments.to_string(),
                }));
            }
            items
        }
        "tool" => message
            .tool_call_id
            .as_ref()
            .map(|call_id| {
                vec![json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.content.as_deref().unwrap_or_default(),
                })]
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn codex_message(role: &str, content: &str) -> Vec<Value> {
    if content.is_empty() {
        return Vec::new();
    }
    let content_type = if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    };
    vec![json!({
        "type": "message",
        "role": role,
        "content": [{ "type": content_type, "text": content }]
    })]
}

fn tool_spec_to_codex(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": tool.function.name,
        "description": tool.function.description,
        "parameters": tool.function.parameters,
        "strict": null,
    })
}

fn parse_codex_sse_response(text: &str) -> Result<Response> {
    let mut content = String::new();
    let mut current_tool: Option<CodexToolAccum> = None;
    let mut tool_calls = Vec::new();
    let mut input_tokens = 0_u32;
    let mut output_tokens = 0_u32;
    let mut total_tokens = 0_u32;

    for event_text in parse_sse_events(text) {
        let Some(event) = parse_sse_event_json(event_text) else {
            continue;
        };
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    content.push_str(delta);
                }
            }
            Some("response.function_call_arguments.delta") => {
                if let (Some(tool), Some(delta)) = (
                    &mut current_tool,
                    event.get("delta").and_then(Value::as_str),
                ) {
                    tool.arguments.push_str(delta);
                }
            }
            Some("response.output_item.added") => {
                if let Some(item) = event.get("item").filter(|item| {
                    item.get("type").and_then(Value::as_str) == Some("function_call")
                }) {
                    current_tool = Some(CodexToolAccum::from_item(item));
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    match item.get("type").and_then(Value::as_str) {
                        Some("function_call") => {
                            let mut accum = CodexToolAccum::from_item(item);
                            if accum.arguments.is_empty() {
                                if let Some(current) = current_tool.take() {
                                    accum.arguments = current.arguments;
                                }
                            } else {
                                current_tool = None;
                            }
                            tool_calls.push(accum.into_tool_call());
                        }
                        Some("message") if content.is_empty() => {
                            content.push_str(&extract_codex_output_text(item));
                        }
                        _ => {}
                    }
                }
            }
            Some("response.completed" | "response.done") => {
                if let Some((input, output, total)) = event
                    .get("response")
                    .and_then(|response| response.get("usage"))
                    .and_then(codex_token_usage)
                {
                    input_tokens = input;
                    output_tokens = output;
                    total_tokens = total;
                }
            }
            Some("error" | "response.failed") => {
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        event
                            .get("error")
                            .and_then(|error| error.get("message"))
                            .and_then(Value::as_str)
                    })
                    .unwrap_or("Codex error");
                return Err(anyhow!(message.to_string()));
            }
            _ => {}
        }
    }

    if let Some(current) = current_tool {
        tool_calls.push(current.into_tool_call());
    }

    // The codex harness contract (t-1134): a turn ends when the assistant
    // returns a final response with no pending tool calls — there is no done
    // tool and no explicit end_turn marker in the SSE stream. Derive the
    // turn state from the parsed shape instead of hardcoding Stop: reporting
    // Stop for tool-call turns made every gpt-5.5 turn look like
    // "finish=stop + tool_call", masking the model's native end-of-turn
    // signal from the agent loop (the smith/forge empty-response crashes).
    // TODO(t-1134, Ben): live-verify against a real codex session that a
    // tool-call-free final message arrives intact through this parser.
    let finish_reason = if tool_calls.is_empty() {
        FinishReason::Stop
    } else {
        FinishReason::ToolCalls
    };
    Ok(Response {
        content,
        tool_calls,
        finish_reason: Some(finish_reason),
        input_tokens,
        output_tokens,
        total_tokens,
        metadata: Default::default(),
    })
}

#[derive(Debug, Clone)]
struct CodexToolAccum {
    id: String,
    name: String,
    arguments: String,
}

impl CodexToolAccum {
    fn from_item(item: &Value) -> Self {
        Self {
            id: item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            arguments: item
                .get("arguments")
                .map(|arguments| match arguments {
                    Value::String(s) => s.clone(),
                    value => value.to_string(),
                })
                .unwrap_or_default(),
        }
    }

    fn into_tool_call(self) -> ToolCall {
        let arguments = serde_json::from_str(&self.arguments)
            .unwrap_or_else(|_| json!({ "raw": self.arguments }));
        ToolCall::new(self.id, self.name, arguments)
    }
}

fn parse_sse_events(text: &str) -> impl Iterator<Item = &str> {
    text.split("\n\n")
}

fn parse_sse_event_json(event_text: &str) -> Option<Value> {
    let json_text = event_text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n");
    if json_text.is_empty() || json_text == "[DONE]" {
        return None;
    }
    serde_json::from_str(&json_text).ok()
}

fn extract_codex_output_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

fn codex_token_usage(usage: &Value) -> Option<(u32, u32, u32)> {
    let input = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .and_then(|tokens| u32::try_from(tokens).ok())
        .unwrap_or_default();
    let output = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .and_then(|tokens| u32::try_from(tokens).ok())
        .unwrap_or_default();
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .and_then(|tokens| u32::try_from(tokens).ok())
        .unwrap_or_else(|| input.saturating_add(output));
    Some((input, output, total))
}

fn extract_codex_account_id(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let decoded = base64_url_decode(payload)?;
    let value: Value = serde_json::from_slice(&decoded).ok()?;
    value
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()
        .map(str::to_string)
}

fn base64_url_decode(input: &str) -> Option<Vec<u8>> {
    let mut normalized = input.replace('-', "+").replace('_', "/");
    let pad_len = (4 - (normalized.len() % 4)) % 4;
    normalized.extend(std::iter::repeat_n('=', pad_len));
    base64_decode(&normalized)
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::new();
    for chunk in bytes.chunks(4) {
        let a = base64_value(chunk[0])?;
        let b = base64_value(chunk[1])?;
        let c = base64_value(chunk[2])?;
        let d = base64_value(chunk[3])?;
        if a < 0 || b < 0 {
            return None;
        }
        out.push(((a << 2) | (b >> 4)) as u8);
        if c >= 0 {
            out.push((((b & 0x0f) << 4) | (c >> 2)) as u8);
        }
        if c >= 0 && d >= 0 {
            out.push((((c & 0x03) << 6) | d) as u8);
        }
    }
    Some(out)
}

fn base64_value(byte: u8) -> Option<i16> {
    match byte {
        b'A'..=b'Z' => Some(i16::from(byte - b'A')),
        b'a'..=b'z' => Some(i16::from(byte - b'a') + 26),
        b'0'..=b'9' => Some(i16::from(byte - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => Some(-1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A syntactically valid JWT whose payload carries the given claims.
    fn fake_jwt(payload: Value) -> String {
        let header = base64url_no_pad(br#"{"alg":"none","typ":"JWT"}"#);
        let body = base64url_no_pad(payload.to_string().as_bytes());
        format!("{header}.{body}.sig")
    }

    fn codex_cli_auth_json(access: &str) -> String {
        json!({
            "OPENAI_API_KEY": null,
            "tokens": {
                "id_token": "unused",
                "access_token": access,
                "refresh_token": "refresh-123",
                "account_id": "acct-1",
            },
            "last_refresh": "2026-06-12T00:00:00Z",
        })
        .to_string()
    }

    #[test]
    fn codex_cli_token_extracts_tokens_and_jwt_expiry() {
        let access = fake_jwt(json!({ "exp": 1893456000i64 }));
        let token = token_from_codex_cli(&codex_cli_auth_json(&access)).unwrap();

        assert_eq!(token.access_token, access);
        assert_eq!(token.refresh_token.as_deref(), Some("refresh-123"));
        assert_eq!(
            token.expires_at,
            Utc.timestamp_opt(1893456000, 0).single(),
            "expiry comes from the JWT exp claim"
        );
    }

    #[test]
    fn codex_cli_token_without_exp_claim_has_no_expiry() {
        let access = fake_jwt(json!({ "sub": "user" }));
        let token = token_from_codex_cli(&codex_cli_auth_json(&access)).unwrap();
        assert!(token.expires_at.is_none());
    }

    #[test]
    fn codex_cli_token_requires_access_token() {
        let err = token_from_codex_cli(r#"{"tokens":{}}"#).unwrap_err();
        assert!(err.to_string().contains("access_token"), "{err}");
    }

    #[tokio::test]
    async fn import_replaces_only_the_codex_entry() {
        let dir = std::env::temp_dir().join(format!("agent-oauth-test-{}", uuid_suffix()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let store_path = dir.join("auth.json");
        // Pre-seed the agent store with an unrelated entry and a stale
        // codex entry; import must replace only the latter.
        tokio::fs::write(
            &store_path,
            json!({
                "anthropic": { "access": "claude-token", "refresh": null, "expires": null },
                "openai-codex": { "access": "stale", "refresh": null, "expires": null },
            })
            .to_string(),
        )
        .await
        .unwrap();

        let source = dir.join("codex-cli-auth.json");
        let access = fake_jwt(json!({ "exp": 1893456000i64 }));
        tokio::fs::write(&source, codex_cli_auth_json(&access))
            .await
            .unwrap();

        let store = TokenStore::with_path(OAuthProviderKind::Codex, store_path.clone());
        let imported = import_codex_cli_from(&source, &store).await.unwrap();
        assert_eq!(imported.access_token, access);

        let written: Value =
            serde_json::from_str(&tokio::fs::read_to_string(&store_path).await.unwrap()).unwrap();
        assert_eq!(
            written["anthropic"]["access"], "claude-token",
            "unrelated entries are preserved"
        );
        assert_eq!(written["openai-codex"]["access"], access.as_str());
        assert_eq!(written["openai-codex"]["refresh"], "refresh-123");
    }

    #[tokio::test]
    async fn import_with_missing_source_says_run_codex_login() {
        let dir = std::env::temp_dir().join(format!("agent-oauth-test-{}", uuid_suffix()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let store = TokenStore::with_path(OAuthProviderKind::Codex, dir.join("auth.json"));

        let err = import_codex_cli_from(&dir.join("missing.json"), &store)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("codex login"), "{err}");
    }

    fn uuid_suffix() -> String {
        let mut bytes = [0u8; 8];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        base64url_no_pad(&bytes)
    }

    #[test]
    fn pkce_generation_has_expected_lengths() {
        let pkce = generate_pkce();
        assert_eq!(pkce.verifier.len(), 43);
        assert_eq!(pkce.challenge.len(), 43);
        assert_ne!(pkce.verifier, pkce.challenge);
        assert!(pkce
            .verifier
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')));
        assert!(pkce
            .challenge
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')));
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
    fn extracts_codex_account_id_from_token() {
        let payload = serde_json::to_vec(&json!({
            "https://api.openai.com/auth": { "chatgpt_account_id": "acct_123" }
        }))
        .unwrap();
        let token = format!("header.{}.sig", base64url_no_pad(&payload));
        assert_eq!(
            extract_codex_account_id(&token).as_deref(),
            Some("acct_123")
        );
    }

    #[test]
    fn parses_codex_sse_text_and_tool_calls() -> Result<()> {
        let sse = r#"data: {"type":"response.output_text.delta","delta":"hello "}


data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"call-1","name":"shell","arguments":""}}


data: {"type":"response.function_call_arguments.delta","delta":"{\"command\":"}


data: {"type":"response.function_call_arguments.delta","delta":"\"printf ok\"}"}


data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call-1","name":"shell"}}


data: {"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}

"#;
        let response = parse_codex_sse_response(sse)?;
        assert_eq!(response.content, "hello ");
        assert_eq!(response.input_tokens, 2);
        assert_eq!(response.output_tokens, 3);
        assert_eq!(response.total_tokens, 5);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "shell");
        assert_eq!(
            response.tool_calls[0].arguments["command"],
            json!("printf ok")
        );
        // t-1134: a tool-call turn must report its real turn state, not a
        // hardcoded Stop that masks the model's end-of-turn signal.
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        Ok(())
    }

    #[test]
    fn codex_tool_call_free_final_reports_end_turn() -> Result<()> {
        let sse = r#"data: {"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"the final answer"}]}}

data: {"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}
"#;
        let response = parse_codex_sse_response(sse)?;
        assert_eq!(response.content, "the final answer");
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        Ok(())
    }

    #[test]
    fn provider_for_tag_accepts_haskell_and_rust_oauth_tags() -> Result<()> {
        assert!(provider_for_tag("openai-codex", Model("gpt-5".into()))?.is_some());
        assert!(provider_for_tag("codex-oauth", Model("gpt-5".into()))?.is_some());
        assert!(provider_for_tag("claude-code", Model("claude".into()))?.is_some());
        assert!(provider_for_tag("claude-code-oauth", Model("claude".into()))?.is_some());
        assert!(provider_for_tag("openai-compatible", Model("model".into()))?.is_none());
        Ok(())
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
