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
mod commands;
mod list;
mod profiles;
mod registry;
mod replay;
mod session;
mod todo;
mod turn;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, CloseSessionRequest, CloseSessionResponse, ContentBlock,
    Implementation, InitializeRequest, InitializeResponse, ListSessionsRequest, LoadSessionRequest,
    LoadSessionResponse, NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest,
    SessionCapabilities, SessionCloseCapabilities, SessionConfigOption, SessionId,
    SessionListCapabilities, SessionModeState, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionModeRequest,
};
use agent_client_protocol::{
    on_receive_notification, on_receive_request, Agent, Client, ConnectionTo, Error, Stdio,
};
use anyhow::{anyhow, Result};
use tokio::sync::mpsc;
use uuid::Uuid;

use agent_core::{ChatMessage, ImageContent};

use crate::{build_runtime, Args, Checkpoint, SessionParams};

struct AcpServer {
    args: Arc<Args>,
    otel_active: bool,
    sessions: Mutex<HashMap<String, session::SessionHandle>>,
    profiles: Arc<profiles::Profiles>,
}

/// Session state lives beside the other agent data:
/// `~/.local/share/agent/acp/<session_id>/` holds the checkpoints that back
/// `session/load`.
fn acp_session_dir(session_id: &str, state_dir: Option<&std::path::Path>) -> Result<PathBuf> {
    if let Some(root) = state_dir {
        return Ok(root.join("acp").join(session_id));
    }
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

/// Convert ACP text and image blocks to the provider-neutral user message.
fn prompt_message(request: &PromptRequest) -> std::result::Result<ChatMessage, Error> {
    let mut parts = Vec::new();
    let mut images = Vec::new();
    for block in &request.prompt {
        match block {
            ContentBlock::Text(text) => parts.push(text.text.as_str()),
            ContentBlock::Image(image) => images.push(ImageContent::new(
                image.mime_type.clone(),
                image.data.clone(),
            )),
            _ => {
                return Err(Error::invalid_params().data(serde_json::json!(
                    "only text and image content blocks are supported"
                )))
            }
        }
    }
    if parts.is_empty() && images.is_empty() {
        return Err(Error::invalid_params().data(serde_json::json!("empty prompt")));
    }
    let text = (!parts.is_empty()).then(|| parts.join("\n\n"));
    Ok(ChatMessage::user_with_images(text, images))
}

fn close_handle(sessions: &mut HashMap<String, session::SessionHandle>, id: &str) {
    if let Some(handle) = sessions.remove(id) {
        handle.cancel_tx.send_modify(|generation| *generation += 1);
        // Dropping the sender closes the mailbox; the actor drops its runtime.
    }
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
        let checkpoint_dir = acp_session_dir(&session_id, self.args.state_dir.as_deref())?;
        if checkpoint.is_none() {
            list::write_metadata(
                &checkpoint_dir,
                &list::SessionMetadata {
                    cwd: cwd.clone(),
                    title: None,
                },
            )
            .await?;
        }
        let (update_tx, mut update_rx) = mpsc::unbounded_channel();
        let streamed: bridge::StreamedText = Arc::new(Mutex::new(HashMap::new()));
        let context_budget: bridge::ContextBudget = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let sink = Arc::new(bridge::AcpTraceSink::new(
            update_tx.clone(),
            streamed.clone(),
            context_budget.clone(),
        ));
        let usage = sink.usage.clone();
        let require_shell_approval =
            self.args.require_shell_approval || self.args.acp_shell_approval;
        let params = SessionParams {
            requested_model: self.args.model.clone(),
            requested_provider: self.args.provider.clone(),
            max_turns: agent_core::turn_budget(self.args.max_turns),
            system_prompt_override: self.args.system_prompt.clone(),
            cwd: Some(cwd),
            checkpoint_dir: Some(checkpoint_dir),
            checkpoint,
            fork_source_run_id: None,
            restore_checkpoint_config: true,
            suspend: None,
            run_id: session_id.clone(),
            session_id: Some(Uuid::parse_str(&session_id)?),
            parent_session_id: None,
            require_shell_approval,
            trace_sinks_extra: vec![sink],
            otel_active: self.otel_active,
        };
        let mut runtime = build_runtime(&self.args, params).await?;
        session::restore_mode(&mut runtime, &self.profiles)?;
        // ACP clients can answer questions (session/request_permission
        // choosers), so the loop offers `ask_user` here and nowhere else.
        runtime.ask_user_tool = true;
        context_budget.store(
            runtime.config.context_budget as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        // Streaming tap: text deltas go straight to the client as
        // agent_message_chunk updates AND accumulate per op so the bridge
        // can suppress the duplicate whole-message chunk at InferResult.
        // Delegated child Infers are not the assistant's reply: the bridge
        // renders their answer inside the `infer` tool card at InferResult,
        // so their deltas are dropped here.
        let tap_tx = update_tx.clone();
        let tap_streamed = streamed.clone();
        runtime.config.on_infer_delta = Some(Arc::new(move |delta: agent_core::InferDelta| {
            if delta.parent_op_id.is_some() {
                return;
            }
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
        let modes = registry::session_modes(
            runtime.shell_requires_approval,
            runtime.acp_mode.as_deref(),
            &self.profiles,
        );
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
                context_budget,
                usage,
                args: self.args.clone(),
                profiles: self.profiles.clone(),
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

    fn close_session(&self, id: &str) {
        close_handle(&mut self.sessions.lock().expect("session map poisoned"), id);
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
        commands::advertise(cx, SessionId::new(session_id.clone()))?;
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
        let checkpoint_path = acp_session_dir(&session_id, self.args.state_dir.as_deref())?
            .join("session-latest.json");
        let checkpoint = crate::load_checkpoint(&checkpoint_path).await?;
        // Replay committed trace effects in order before the load response.
        for update in replay::history_updates(&checkpoint).await {
            cx.send_notification(SessionNotification::new(request.session_id.clone(), update))
                .map_err(|err| anyhow!("replaying session history: {err}"))?;
        }
        // The trace records validated tool results even across GC compaction.
        // Replay only the last successful replacement, including an empty list.
        // A missing trace is not fatal: history replay already fell back to text.
        if let Ok(events) = agent_core::TraceLogger::read_events(&checkpoint.trace_path).await {
            if let Some(plan) = todo::latest_plan(&events) {
                cx.send_notification(SessionNotification::new(request.session_id.clone(), plan))
                    .map_err(|err| anyhow!("replaying todo plan: {err}"))?;
            }
        }
        let dir = acp_session_dir(&session_id, self.args.state_dir.as_deref())?;
        if list::read_metadata(&dir).await.is_err() {
            list::write_metadata(
                &dir,
                &list::SessionMetadata {
                    cwd: request.cwd.clone(),
                    title: list::checkpoint_title(&checkpoint),
                },
            )
            .await?;
        }
        let (modes, config_options) = self
            .spawn_session(session_id, request.cwd, Some(checkpoint), cx)
            .await?;
        commands::advertise(cx, request.session_id.clone())?;
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
    let profiles = Arc::new(profiles::Profiles::load(args.acp_modes_dir.as_deref())?);
    let server = Arc::new(AcpServer {
        profiles,
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
    let list_server = server.clone();
    let close_server = server.clone();

    let result = Agent
        .builder()
        .name("agentd")
        .on_receive_request(
            async move |initialize: InitializeRequest, responder, _cx| {
                responder.respond(
                    InitializeResponse::new(initialize.protocol_version)
                        .agent_capabilities(
                            AgentCapabilities::new()
                                .load_session(true)
                                .session_capabilities(
                                    SessionCapabilities::new()
                                        .list(SessionListCapabilities::new())
                                        .close(SessionCloseCapabilities::new()),
                                )
                                .prompt_capabilities(PromptCapabilities::new().image(true)),
                        )
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
            async move |request: ListSessionsRequest, responder, _cx| match acp_session_dir(
                "",
                list_server.args.state_dir.as_deref(),
            ) {
                Ok(root) => match list::list(&root, request).await {
                    Ok(response) => responder.respond(response),
                    Err(err) => responder.respond_with_error(internal_error(err)),
                },
                Err(err) => responder.respond_with_error(internal_error(err)),
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |request: CloseSessionRequest, responder, _cx| {
                close_server.close_session(request.session_id.0.as_ref());
                responder.respond(CloseSessionResponse::new())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |request: PromptRequest, responder, _cx| {
                let command = commands::parse(&request.prompt);
                let message = match prompt_message(&request) {
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
                })) = handle.cmd_tx.send(session::SessionCommand::Prompt {
                    message,
                    command,
                    responder,
                }) {
                    return responder.respond_with_error(internal_error("session is gone"));
                }
                Ok(())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |request: SetSessionModeRequest, responder, _cx| {
                if !registry::is_session_mode(request.mode_id.0.as_ref())
                    && set_mode_server
                        .profiles
                        .get(request.mode_id.0.as_ref())
                        .is_none()
                {
                    return responder.respond_with_error(
                        Error::invalid_params().data(serde_json::json!("unknown session mode")),
                    );
                }
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

#[cfg(test)]
mod close_tests {
    use super::*;
    #[test]
    fn close_cancels_and_removes_and_is_idempotent() {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(0);
        let mut sessions = HashMap::from([(
            "session".into(),
            session::SessionHandle { cmd_tx, cancel_tx },
        )]);
        close_handle(&mut sessions, "unknown");
        assert_eq!(*cancel_rx.borrow_and_update(), 0);
        close_handle(&mut sessions, "session");
        assert!(sessions.is_empty());
        assert_eq!(*cancel_rx.borrow_and_update(), 1);
        assert!(cmd_rx.is_closed());
        close_handle(&mut sessions, "session");
    }
}
