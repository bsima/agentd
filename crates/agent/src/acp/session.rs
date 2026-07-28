//! Per-session actor: owns the session's `Runtime` and serializes turns.
//! ACP allows one in-flight `session/prompt` per session; the actor's mailbox
//! enforces that while the connection's dispatch loop stays free for other
//! sessions and cancellation. Mode and model changes ride the same mailbox,
//! so they apply between turns, never mid-turn.

use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    CurrentModeUpdate, PromptResponse, SessionConfigOption, SessionId, SessionModeId,
    SessionNotification, SessionUpdate, SetSessionConfigOptionResponse, SetSessionModeResponse,
    StopReason,
};
use agent_client_protocol::{Client, ConnectionTo, Responder};
use anyhow::Result;
use tokio::sync::{mpsc, watch};

use super::bridge::{ForwarderMsg, StreamedText};
use super::registry;
use super::turn::{run_acp_turn, TurnOutcome};
use crate::{response_turn_budget_exhausted, Args, Runtime};

pub(crate) enum SessionCommand {
    Prompt {
        text: String,
        responder: Responder<PromptResponse>,
    },
    SetMode {
        mode_id: SessionModeId,
        responder: Responder<SetSessionModeResponse>,
    },
    SetModel {
        alias: String,
        responder: Responder<SetSessionConfigOptionResponse>,
    },
}

pub(crate) struct SessionHandle {
    pub(crate) cmd_tx: mpsc::UnboundedSender<SessionCommand>,
    /// Bumped by the `session/cancel` handler; the actor's in-flight turn
    /// select!s on it and drops the turn future when it fires.
    pub(crate) cancel_tx: watch::Sender<u64>,
}

/// Everything the actor needs beyond the runtime and its mailbox.
pub(crate) struct SessionContext {
    pub(crate) cx: ConnectionTo<Client>,
    pub(crate) session_id: SessionId,
    pub(crate) cancel_rx: watch::Receiver<u64>,
    pub(crate) streamed: StreamedText,
    pub(crate) update_tx: mpsc::UnboundedSender<ForwarderMsg>,
    pub(crate) args: Arc<Args>,
}

pub(crate) async fn session_actor(
    mut runtime: Runtime,
    mut cmd_rx: mpsc::UnboundedReceiver<SessionCommand>,
    ctx: SessionContext,
) {
    let SessionContext {
        cx,
        session_id,
        mut cancel_rx,
        streamed,
        update_tx,
        args,
    } = ctx;
    while let Some(command) = cmd_rx.recv().await {
        match command {
            SessionCommand::Prompt { text, responder } => {
                // A cancel that raced in between turns applies to nothing:
                // observe the current generation so only cancels arriving
                // DURING this turn fire the select arm below.
                cancel_rx.borrow_and_update();
                tokio::select! {
                    result = run_acp_turn(&mut runtime, text, &cx, &session_id) => {
                        // Turn barrier: ACP clients treat the prompt
                        // response as the turn boundary, so every update the
                        // turn enqueued must reach the connection first.
                        flush_updates(&update_tx).await;
                        match result {
                            Ok(TurnOutcome::Done(response)) => {
                                let stop_reason = if response_turn_budget_exhausted(&response) {
                                    StopReason::MaxTurnRequests
                                } else {
                                    StopReason::EndTurn
                                };
                                let _ = responder.respond(PromptResponse::new(stop_reason));
                            }
                            Ok(TurnOutcome::Cancelled) => {
                                let _ = responder
                                    .respond(PromptResponse::new(StopReason::Cancelled));
                            }
                            Err(err) => {
                                let _ = responder.respond_with_error(
                                    agent_client_protocol::Error::new(-32603, err.to_string()),
                                );
                            }
                        }
                    }
                    // Coarse v1 cancellation: dropping the turn future closes
                    // any in-flight provider stream and abandons the loop
                    // machine mid-instruction. The session survives — history
                    // keeps the user message (like a failed turn) and the
                    // next prompt starts a fresh machine. An in-flight shell
                    // child is not killed synchronously (it runs to its
                    // timeout); the cancel-token follow-up tightens that.
                    _ = cancel_rx.changed() => {
                        streamed.lock().expect("streamed text map poisoned").clear();
                        flush_updates(&update_tx).await;
                        let _ = responder.respond(PromptResponse::new(StopReason::Cancelled));
                    }
                }
            }
            SessionCommand::SetMode { mode_id, responder } => {
                runtime.shell_requires_approval = mode_id.0.as_ref() != registry::MODE_YOLO;
                // Announce before acking so clients that stop reading at the
                // response still observe the mode change.
                let _ = cx.send_notification(SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode_id)),
                ));
                let _ = responder.respond(SetSessionModeResponse::new());
            }
            SessionCommand::SetModel { alias, responder } => {
                match set_model(&mut runtime, &args, &alias).await {
                    Ok(config_options) => {
                        let _ =
                            responder.respond(SetSessionConfigOptionResponse::new(config_options));
                    }
                    Err(err) => {
                        let _ = responder.respond_with_error(agent_client_protocol::Error::new(
                            -32603,
                            err.to_string(),
                        ));
                    }
                }
            }
        }
    }
}

/// Wait until the forwarder has handed every already-enqueued update to the
/// connection. A dead forwarder (connection wound down) resolves
/// immediately — the response send will fail harmlessly on the same
/// closed connection.
async fn flush_updates(update_tx: &mpsc::UnboundedSender<ForwarderMsg>) {
    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    if update_tx.send(ForwarderMsg::Flush(ack_tx)).is_ok() {
        let _ = ack_rx.await;
    }
}

/// Re-resolve the model alias against the registry and swap the runtime's
/// provider, context budget, and pricing in place — the same helpers the
/// approvals-resume path already drives, so this is pure plumbing.
async fn set_model(
    runtime: &mut Runtime,
    args: &Args,
    alias: &str,
) -> Result<Vec<SessionConfigOption>> {
    let (resolved, pricing, _embedder) = crate::resolve_model(Some(alias.to_owned()), None).await?;
    let (provider, provider_url) = crate::build_provider(
        &resolved,
        args.provider.clone(),
        args.key.clone(),
        runtime.ir_replay.is_some(),
    )?;
    runtime.config.provider = provider;
    runtime.config.context_budget = resolved.context;
    runtime.config.pricing = pricing;
    runtime.model = agent_core::Model(resolved.api_id.clone());
    runtime.provider_url = provider_url;
    runtime.resume_facts.model = resolved.alias.clone();
    Ok(registry::model_config_options(&resolved.alias).await)
}
