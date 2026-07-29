//! ACP (Agent Client Protocol) server mode: `agent --acp` speaks
//! agentclientprotocol.com JSON-RPC over stdio so frontends like Paseo can
//! spawn the binary as a custom provider and drive sessions from their UIs.
//!
//! stdout carries JSON-RPC frames exclusively in this mode (`--acp` conflicts
//! with `--debug`); diagnostics go to stderr like every other mode. Each
//! `session/new` builds its own `Runtime` (one agent loop per ACP session)
//! whose trace events a bridge sink maps into `session/update` notifications;
//! `session/load` rebuilds a runtime from the session's checkpoint.

mod bridge;
mod registry;
mod session;
mod turn;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, ContentBlock, Implementation, InitializeRequest,
    InitializeResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
    NewSessionResponse, PromptRequest, SessionConfigOption, SessionId, SessionModeState,
    SessionNotification, SessionUpdate, SetSessionConfigOptionRequest, SetSessionModeRequest,
};
use agent_client_protocol::{
    on_receive_notification, on_receive_request, Agent, Client, ConnectionTo, Error, Stdio,
};
use anyhow::{anyhow, Result};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{build_runtime, Args, Checkpoint, SessionParams, DEFAULT_MAX_TURNS};

struct AcpServer {
    args: Arc<Args>,
    otel_active: bool,
    sessions: Mutex<HashMap<String, session::SessionHandle>>,
}

/// Session state lives beside the other agent data:
/// `~/.local/share/agent/acp/<session_id>/` holds the checkpoints that back
/// `session/load`.
fn acp_session_dir(session_id: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
    Ok(home.join(".local/share/agent/acp").join(session_id))
}

fn internal_error(err: impl std::fmt::Display) -> Error {
    Error::new(-32603, err.to_string())
}

fn unknown_session() -> Error {
    Error::invalid_params().data(serde_json::json!("unknown session"))
}

/// MCP passthrough is not implemented: clients that offer servers anyway
/// (initialize advertises all-false `McpCapabilities`) lose those tools, so
/// say so on stderr instead of discarding them silently. Paseo users should
/// disable MCP injection for this provider — see docs/ACP.md.
fn warn_ignored_mcp_servers(count: usize) {
    if count > 0 {
        eprintln!(
            "warning: ignoring {count} MCP server(s) offered by the ACP client — \
             MCP passthrough is not implemented (see docs/ACP.md)"
        );
    }
}

/// Concatenate the prompt's text blocks. Image/audio/resource blocks are
/// rejected: initialize advertised text-only prompt capabilities.
fn prompt_text(request: &PromptRequest) -> std::result::Result<String, Error> {
    let mut parts = Vec::new();
    for block in &request.prompt {
        match block {
            ContentBlock::Text(text) => parts.push(text.text.as_str()),
            _ => {
                return Err(Error::invalid_params()
                    .data(serde_json::json!("only text content blocks are supported")))
            }
        }
    }
    if parts.is_empty() {
        return Err(Error::invalid_params().data(serde_json::json!("empty prompt")));
    }
    Ok(parts.join("\n\n"))
}

impl AcpServer {
    /// Build a session runtime (fresh or from a checkpoint), wire the event
    /// bridge and streaming tap, spawn the actor, and register the handle.
    /// Returns what both `session/new` and `session/load` report back:
    /// modes and config options.
    async fn spawn_session(
        &self,
        session_id: String,
        cwd: PathBuf,
        checkpoint: Option<Checkpoint>,
        cx: &ConnectionTo<Client>,
    ) -> Result<(SessionModeState, Vec<SessionConfigOption>)> {
        let checkpoint_dir = acp_session_dir(&session_id)?;
        let (update_tx, mut update_rx) = mpsc::unbounded_channel();
        let streamed: bridge::StreamedText = Arc::new(Mutex::new(HashMap::new()));
        let sink = Arc::new(bridge::AcpTraceSink::new(
            update_tx.clone(),
            streamed.clone(),
        ));
        let require_shell_approval =
            self.args.require_shell_approval || self.args.acp_shell_approval;
        let params = SessionParams {
            requested_model: self.args.model.clone(),
            requested_provider: self.args.provider.clone(),
            max_turns: self.args.max_turns.unwrap_or(DEFAULT_MAX_TURNS),
            system_prompt_override: self.args.system_prompt.clone(),
            cwd: Some(cwd),
            checkpoint_dir: Some(checkpoint_dir),
            checkpoint,
            run_id: session_id.clone(),
            require_shell_approval,
            trace_sinks_extra: vec![sink],
            otel_active: self.otel_active,
        };
        let mut runtime = build_runtime(&self.args, params).await?;
        // Streaming tap: text deltas go straight to the client as
        // agent_message_chunk updates AND accumulate per op so the bridge
        // can suppress the duplicate whole-message chunk at InferResult.
        let tap_tx = update_tx.clone();
        let tap_streamed = streamed.clone();
        runtime.config.on_infer_delta = Some(Arc::new(move |delta: agent_core::InferDelta| {
            let _ = tap_tx.send(bridge::ForwarderMsg::update(
                SessionUpdate::AgentMessageChunk(bridge::text_chunk(delta.text.clone())),
            ));
            tap_streamed
                .lock()
                .expect("streamed text map poisoned")
                .entry(delta.op_id)
                .or_default()
                .push_str(&delta.text);
        }));
        let modes = registry::session_modes(runtime.shell_requires_approval);
        let config_options = registry::session_config_options(
            &runtime.resume_facts.model,
            &runtime.config.gc,
            runtime.config.gc_threshold,
        )
        .await;

        // Forwarder: drains the bridge channel into session/update
        // notifications. Tied to the connection so it winds down with it.
        // Flush markers ack once everything enqueued before them has been
        // handed to the connection (the actor's pre-response turn barrier).
        let sid = SessionId::new(session_id.clone());
        let notify_cx = cx.clone();
        let notify_sid = sid.clone();
        cx.spawn(async move {
            while let Some(message) = update_rx.recv().await {
                match message {
                    bridge::ForwarderMsg::Update(update) => {
                        notify_cx.send_notification(SessionNotification::new(
                            notify_sid.clone(),
                            *update,
                        ))?;
                    }
                    bridge::ForwarderMsg::Flush(ack) => {
                        let _ = ack.send(());
                    }
                }
            }
            Ok(())
        })
        .map_err(|err| anyhow!("spawning session/update forwarder: {err}"))?;

        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(0_u64);
        tokio::spawn(session::session_actor(
            runtime,
            cmd_rx,
            session::SessionContext {
                cx: cx.clone(),
                session_id: sid,
                cancel_rx,
                streamed,
                update_tx,
                args: self.args.clone(),
            },
        ));
        // Replacing an existing handle (re-loading a live session) closes
        // the old actor's mailbox; it winds down after any in-flight turn.
        self.sessions
            .lock()
            .expect("session map poisoned")
            .insert(session_id, session::SessionHandle { cmd_tx, cancel_tx });
        Ok((modes, config_options))
    }

    async fn new_session(
        &self,
        request: NewSessionRequest,
        cx: &ConnectionTo<Client>,
    ) -> Result<NewSessionResponse> {
        warn_ignored_mcp_servers(request.mcp_servers.len());
        let session_id = Uuid::new_v4().to_string();
        let (modes, config_options) = self
            .spawn_session(session_id.clone(), request.cwd, None, cx)
            .await?;
        Ok(NewSessionResponse::new(SessionId::new(session_id))
            .modes(modes)
            .config_options(config_options))
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
        cx: &ConnectionTo<Client>,
    ) -> Result<LoadSessionResponse> {
        warn_ignored_mcp_servers(request.mcp_servers.len());
        let session_id = request.session_id.0.to_string();
        let checkpoint_path = acp_session_dir(&session_id)?.join("session-latest.json");
        let checkpoint = crate::load_checkpoint(&checkpoint_path).await?;
        // Per protocol, the conversation replays as session/update
        // notifications before the load response. Text-only in v1: tool
        // traffic from past turns is not reconstructed.
        for message in &checkpoint.messages {
            let Some(content) = message.content.as_deref().filter(|text| !text.is_empty()) else {
                continue;
            };
            let update = match message.role.as_str() {
                "user" => SessionUpdate::UserMessageChunk(bridge::text_chunk(content)),
                "assistant" => SessionUpdate::AgentMessageChunk(bridge::text_chunk(content)),
                _ => continue,
            };
            cx.send_notification(SessionNotification::new(request.session_id.clone(), update))
                .map_err(|err| anyhow!("replaying session history: {err}"))?;
        }
        let (modes, config_options) = self
            .spawn_session(session_id, request.cwd, Some(checkpoint), cx)
            .await?;
        Ok(LoadSessionResponse::new()
            .modes(modes)
            .config_options(config_options))
    }
}

pub(crate) async fn run(args: Args) -> Result<()> {
    // One process-wide OTel provider; sessions attach their own OTel trace
    // sinks when it is active (the global provider is not per-session).
    let otel = crate::init_otel(
        args.otel_endpoint.as_deref(),
        &format!("acp-{}", Uuid::new_v4()),
    )?;
    let server = Arc::new(AcpServer {
        otel_active: otel.is_some(),
        args: Arc::new(args),
        sessions: Mutex::new(HashMap::new()),
    });
    let new_session_server = server.clone();
    let load_session_server = server.clone();
    let prompt_server = server.clone();
    let set_mode_server = server.clone();
    let set_config_server = server.clone();
    let cancel_server = server.clone();

    let result = Agent
        .builder()
        .name("agentd")
        .on_receive_request(
            async move |initialize: InitializeRequest, responder, _cx| {
                responder.respond(
                    InitializeResponse::new(initialize.protocol_version)
                        .agent_capabilities(AgentCapabilities::new().load_session(true))
                        .agent_info(Implementation::new("agentd", env!("CARGO_PKG_VERSION"))),
                )
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |request: NewSessionRequest, responder, cx| match new_session_server
                .new_session(request, &cx)
                .await
            {
                Ok(response) => responder.respond(response),
                Err(err) => responder.respond_with_error(internal_error(err)),
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |request: LoadSessionRequest, responder, cx| match load_session_server
                .load_session(request, &cx)
                .await
            {
                Ok(response) => responder.respond(response),
                Err(err) => responder.respond_with_error(internal_error(err)),
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |request: PromptRequest, responder, _cx| {
                let text = match prompt_text(&request) {
                    Ok(text) => text,
                    Err(err) => return responder.respond_with_error(err),
                };
                let sessions = prompt_server.sessions.lock().expect("session map poisoned");
                let Some(handle) = sessions.get(request.session_id.0.as_ref()) else {
                    drop(sessions);
                    return responder.respond_with_error(unknown_session());
                };
                // The actor answers via the moved responder when the turn
                // finishes; the dispatch loop stays free meanwhile.
                if let Err(mpsc::error::SendError(session::SessionCommand::Prompt {
                    responder,
                    ..
                })) = handle
                    .cmd_tx
                    .send(session::SessionCommand::Prompt { text, responder })
                {
                    return responder.respond_with_error(internal_error("session is gone"));
                }
                Ok(())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |request: SetSessionModeRequest, responder, _cx| {
                let sessions = set_mode_server
                    .sessions
                    .lock()
                    .expect("session map poisoned");
                let Some(handle) = sessions.get(request.session_id.0.as_ref()) else {
                    drop(sessions);
                    return responder.respond_with_error(unknown_session());
                };
                if let Err(mpsc::error::SendError(session::SessionCommand::SetMode {
                    responder,
                    ..
                })) = handle.cmd_tx.send(session::SessionCommand::SetMode {
                    mode_id: request.mode_id,
                    responder,
                }) {
                    return responder.respond_with_error(internal_error("session is gone"));
                }
                Ok(())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |request: SetSessionConfigOptionRequest, responder, _cx| {
                let config_id = request.config_id.0.to_string();
                if !matches!(
                    config_id.as_str(),
                    registry::MODEL_CONFIG_ID
                        | registry::GC_CONFIG_ID
                        | registry::GC_THRESHOLD_CONFIG_ID
                ) {
                    return responder.respond_with_error(
                        Error::invalid_params().data(serde_json::json!("unknown config option")),
                    );
                }
                let Some(value) = request.value.as_value_id().map(|id| id.0.to_string()) else {
                    return responder.respond_with_error(
                        Error::invalid_params().data(serde_json::json!("expected a value id")),
                    );
                };
                let sessions = set_config_server
                    .sessions
                    .lock()
                    .expect("session map poisoned");
                let Some(handle) = sessions.get(request.session_id.0.as_ref()) else {
                    drop(sessions);
                    return responder.respond_with_error(unknown_session());
                };
                if let Err(mpsc::error::SendError(session::SessionCommand::SetConfig {
                    responder,
                    ..
                })) = handle.cmd_tx.send(session::SessionCommand::SetConfig {
                    config_id,
                    value,
                    responder,
                }) {
                    return responder.respond_with_error(internal_error("session is gone"));
                }
                Ok(())
            },
            on_receive_request!(),
        )
        .on_receive_notification(
            async move |notification: CancelNotification, _cx| {
                let sessions = cancel_server.sessions.lock().expect("session map poisoned");
                if let Some(handle) = sessions.get(notification.session_id.0.as_ref()) {
                    // The actor's select! observes the bump and drops the
                    // in-flight turn; a cancel with no turn running is a
                    // no-op (the next turn re-baselines the generation).
                    handle.cancel_tx.send_modify(|generation| *generation += 1);
                }
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_to(Stdio::new())
        .await
        .map_err(|err| anyhow!("acp connection failed: {err}"));
    if let Some(otel) = otel {
        otel.shutdown();
    }
    result
}
