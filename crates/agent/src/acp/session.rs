//! Per-session actor: owns the session's `Runtime` and serializes turns.
//! ACP allows one in-flight `session/prompt` per session; the actor's mailbox
//! enforces that while the connection's dispatch loop stays free for other
//! sessions and cancellation. Mode and model changes ride the same mailbox,
//! so they apply between turns, never mid-turn.

use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    CurrentModeUpdate, PromptResponse, SessionId, SessionModeId, SessionNotification,
    SessionUpdate, SetSessionConfigOptionResponse, SetSessionModeResponse, StopReason,
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
    /// A `session/set_config_option` write: model, GC strategy, or GC
    /// threshold, dispatched by config id in the actor so all of them
    /// mutate the runtime between turns, never mid-turn.
    SetConfig {
        config_id: String,
        value: String,
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
            SessionCommand::SetConfig {
                config_id,
                value,
                responder,
            } => {
                let result = match config_id.as_str() {
                    registry::MODEL_CONFIG_ID => set_model(&mut runtime, &args, &value).await,
                    registry::GC_CONFIG_ID => set_gc(&mut runtime, &args, &value).await,
                    registry::GC_THRESHOLD_CONFIG_ID => set_gc_threshold(&mut runtime, &value),
                    other => Err(anyhow::anyhow!("unknown config option: {other}")),
                };
                match result {
                    Ok(()) => {
                        // Config is session state, not connection state. Write
                        // it immediately so session/load preserves a change
                        // even when the client disconnects before another turn.
                        crate::persist_session(&mut runtime).await;
                        let config_options = registry::session_config_options(
                            &runtime.resume_facts.model,
                            &runtime.config.gc,
                            runtime.config.gc_threshold,
                        )
                        .await;
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
async fn set_model(runtime: &mut Runtime, args: &Args, alias: &str) -> Result<()> {
    // Same url/key precedence as build_runtime: flags win, then the
    // --config file's provider block — a session that started against a
    // configured endpoint must not silently switch endpoints on a model
    // change.
    let file_config = crate::read_config(args.config.as_ref()).await?;
    let provider_file = file_config.provider.unwrap_or_default();
    let (resolved, pricing, _embedder) = crate::resolve_model(Some(alias.to_owned()), None).await?;
    let (provider, provider_url) = crate::build_provider(
        &resolved,
        args.provider.clone().or(provider_file.url),
        args.key.clone().or(provider_file.api_key),
        runtime.ir_replay.is_some(),
    )?;
    runtime.config.provider = provider;
    // Learned overflow ceilings are specific to the old provider/model.
    // Preserve the rest of GcState (lifecycles, frames, hot-set history).
    runtime.gc_state.discovered_budget = None;
    runtime.config.context_budget = resolved.context;
    runtime.config.pricing = pricing;
    runtime.model = agent_core::Model(resolved.api_id.clone());
    runtime.provider_url = provider_url;
    runtime.resume_facts.model = resolved.alias.clone();
    Ok(())
}

/// Swap the GC strategy between turns. Safe mid-session: GC mode is not
/// part of program identity (no effect-id or replay impact) and the
/// persistent `gc_state` (discovered budget, frame lifecycles, hot set) is
/// strategy-agnostic. The process-level `--gc-*` knobs (cache policy,
/// windows, floors) carry over into the new strategy.
async fn set_gc(runtime: &mut Runtime, args: &Args, value: &str) -> Result<()> {
    let choice = <crate::GcArg as clap::ValueEnum>::from_str(value, true)
        .map_err(|_| anyhow::anyhow!("unknown gc strategy: {value}"))?;
    // Mirror build_runtime's guard: non-threshold timing needs a strategy.
    if matches!(choice, crate::GcArg::None) && args.gc_timing != agent_core::GcTiming::Threshold {
        return Err(anyhow::anyhow!(
            "--gc-timing {} requires a GC strategy; gc cannot be turned off for this session",
            args.gc_timing.name()
        ));
    }
    // The embedder for semantic/generational comes from the registry's
    // `embeddings` entry, exactly as at session start; absent registry or
    // entry degrades to the heuristic/citation-only modes with a stderr
    // warning (same as the flags).
    let embedder = match crate::resolve_model(Some(runtime.resume_facts.model.clone()), None).await
    {
        Ok((_, _, embedder)) => embedder,
        Err(_) => None,
    };
    runtime.config.gc = crate::gc_mode_from_choice(args, choice, &embedder);
    Ok(())
}

fn set_gc_threshold(runtime: &mut Runtime, value: &str) -> Result<()> {
    let threshold: f32 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("gc threshold must be a number, got {value:?}"))?;
    if !(0.1..=1.0).contains(&threshold) {
        return Err(anyhow::anyhow!(
            "gc threshold must be within 0.1..=1.0, got {threshold}"
        ));
    }
    runtime.config.gc_threshold = threshold;
    Ok(())
}
