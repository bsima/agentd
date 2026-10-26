//! Per-session actor: owns the session's `Runtime` and serializes turns.
//! ACP allows one in-flight `session/prompt` per session; the actor's mailbox
//! enforces that while the connection's dispatch loop stays free for other
//! sessions and cancellation. Mode and model changes ride the same mailbox,
//! so they apply between turns, never mid-turn.

use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    ConfigOptionUpdate, CurrentModeUpdate, PromptResponse, SessionId, SessionModeId,
    SessionNotification, SessionUpdate, SetSessionConfigOptionResponse, SetSessionModeResponse,
    StopReason, Usage,
};
use agent_client_protocol::{Client, ConnectionTo, Responder};
use anyhow::Result;
use tokio::sync::{mpsc, watch};

use super::bridge::{ForwarderMsg, StreamedText};
use super::commands;
use super::registry;
use super::turn::{run_acp_turn, TurnOutcome};
use crate::{response_turn_budget_exhausted, Args, Runtime};

pub(crate) enum SessionCommand {
    Prompt {
        message: agent_core::ChatMessage,
        command: Option<commands::Command>,
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
    pub(crate) context_budget: super::bridge::ContextBudget,
    pub(crate) usage: Arc<std::sync::Mutex<agent_core::RunUsage>>,
    pub(crate) args: Arc<Args>,
    pub(crate) profiles: Arc<super::profiles::Profiles>,
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
        context_budget,
        usage,
        args,
        profiles,
    } = ctx;
    while let Some(command) = cmd_rx.recv().await {
        if cmd_rx.is_closed() {
            break;
        }
        match command {
            SessionCommand::Prompt {
                message,
                command,
                responder,
            } => {
                if let Some(command) = command {
                    match commands::execute(command, &runtime).await {
                        Ok(text) => {
                            let _ = update_tx.send(ForwarderMsg::update(
                                SessionUpdate::AgentMessageChunk(super::bridge::text_chunk(text)),
                            ));
                            flush_updates(&update_tx).await;
                            let _ = responder.respond(PromptResponse::new(StopReason::EndTurn));
                        }
                        Err(err) => {
                            let _ = responder.respond_with_error(
                                agent_client_protocol::Error::new(-32603, err.to_string()),
                            );
                        }
                    }
                    continue;
                }
                // A cancel that raced in between turns applies to nothing:
                // observe the current generation so only cancels arriving
                // DURING this turn fire the select arm below.
                cancel_rx.borrow_and_update();
                // First user prompt owns the title; persist it independently
                // of turn completion (a cancelled first turn is still named).
                if let Some(dir) = runtime.checkpoint_dir.as_deref() {
                    if let Ok(mut info) = super::list::read_metadata(dir).await {
                        if info.title.is_none() {
                            if let Some(title) =
                                message.content.as_deref().and_then(super::list::title)
                            {
                                info.title = Some(title.clone());
                                if let Err(err) = super::list::write_metadata(dir, &info).await {
                                    tracing::warn!(%err, "saving ACP session title");
                                }
                                let _ = cx.send_notification(SessionNotification::new(
                                    session_id.clone(),
                                    SessionUpdate::SessionInfoUpdate(
                                        agent_client_protocol::schema::v1::SessionInfoUpdate::new()
                                            .title(title),
                                    ),
                                ));
                            }
                        }
                    }
                }
                tokio::select! {
                    result = run_acp_turn(&mut runtime, message, &cx, &session_id) => {
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
                                let totals = usage.lock().expect("ACP usage poisoned");
                                let reply = PromptResponse::new(stop_reason).usage(
                                    Usage::new(totals.total_tokens, totals.input_tokens, totals.output_tokens)
                                        .cached_read_tokens(totals.cached_input_tokens),
                                );
                                let _ = responder.respond(reply);
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
                        if cmd_rx.is_closed() { break; }
                    }
                }
            }
            SessionCommand::SetMode { mode_id, responder } => {
                if let Err(err) = apply_mode(&mut runtime, mode_id.0.as_ref(), &profiles) {
                    let _ = responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params()
                            .data(serde_json::json!(err.to_string())),
                    );
                    continue;
                }
                crate::persist_session(&mut runtime).await;
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
                    registry::MODEL_CONFIG_ID => {
                        set_model(&mut runtime, &args, &value, &context_budget).await
                    }
                    registry::THOUGHT_CONFIG_ID => value
                        .parse::<agent_core::Effort>()
                        .map_err(|_| anyhow::anyhow!("unknown effort level: {value}"))
                        .and_then(|effort| {
                            runtime.set_inference_settings(
                                Some(effort),
                                runtime.config.inference_settings.fast,
                            )
                        }),
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
                            &runtime.model_capabilities.0,
                            runtime.config.inference_settings.effort,
                        )
                        .await;
                        if config_id == registry::MODEL_CONFIG_ID {
                            let _ = cx.send_notification(SessionNotification::new(
                                session_id.clone(),
                                SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(
                                    config_options.clone(),
                                )),
                            ));
                        }
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
    context_budget: &super::bridge::ContextBudget,
) -> Result<()> {
    // Same url/key precedence as build_runtime: flags win, then the
    // --config file's provider block — a session that started against a
    // configured endpoint must not silently switch endpoints on a model
    // change.
    let file_config = crate::read_config(args.config.as_ref()).await?;
    let provider_file = file_config.provider.unwrap_or_default();
    let (resolved, pricing, _embedder, registry) =
        crate::resolve_model(Some(alias.to_owned()), None).await?;
    let (provider, provider_url) = crate::build_provider(
        &resolved,
        args.provider.clone().or(provider_file.url),
        args.key.clone().or(provider_file.api_key),
        runtime.ir_replay.is_some(),
    )?;
    runtime.revalidate_inference_settings(&resolved);
    runtime.config.provider = provider;
    // Learned overflow ceilings are specific to the old provider/model.
    // Preserve the rest of GcState (lifecycles, frames, hot-set history).
    runtime.gc_state.discovered_budget = None;
    runtime.config.context_budget = resolved.context;
    context_budget.store(
        resolved.context as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    runtime.config.pricing = pricing;
    runtime.config.guidance.infer_models = registry.map(|r| (r, resolved.clone()));
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
        Ok((_, _, embedder, _)) => embedder,
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

/// Apply the policy and prompt only between turns; the profile catalog is
/// immutable for the lifetime of the ACP process.
pub(super) fn restore_mode(
    runtime: &mut Runtime,
    profiles: &super::profiles::Profiles,
) -> Result<()> {
    if let Some(id) = runtime.acp_mode.clone() {
        if restored_mode_id(&id, profiles) != id {
            eprintln!(
                "warning: mode profile {id:?} is unavailable; restoring ACP session in ask mode"
            );
            apply_mode(runtime, registry::MODE_ASK, profiles)?;
        } else {
            apply_mode(runtime, &id, profiles)?;
        }
    }
    Ok(())
}

fn restored_mode_id<'a>(id: &'a str, profiles: &super::profiles::Profiles) -> &'a str {
    if registry::is_session_mode(id) || profiles.get(id).is_some() {
        id
    } else {
        registry::MODE_ASK
    }
}

fn apply_mode(runtime: &mut Runtime, id: &str, profiles: &super::profiles::Profiles) -> Result<()> {
    use super::profiles::ShellApproval;
    // Only the active profile can offer session-local tools. The process's
    // --todo registration remains available in every mode.
    let profile = if registry::is_session_mode(id) {
        None
    } else {
        Some(
            profiles
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("mode profile {id:?} is unavailable"))?,
        )
    };
    if !runtime.todo_process_wide {
        runtime.config.tools.unregister(agent_core::todo::TODO_TOOL);
        if profile.is_some_and(|p| {
            p.tools
                .iter()
                .any(|tool| tool == agent_core::todo::TODO_TOOL)
        }) {
            runtime
                .config
                .tools
                .register(agent_core::todo::todo_tool())?;
        }
    }
    runtime.acp_mode_prompt = None;
    runtime.acp_shell_always = false;
    match id {
        registry::MODE_ASK => runtime.shell_requires_approval = true,
        registry::MODE_YOLO => runtime.shell_requires_approval = false,
        _ => {
            let profile = profile.expect("validated profile");
            runtime.shell_requires_approval = profile.shell_approval != ShellApproval::Never;
            runtime.acp_shell_always = profile.shell_approval == ShellApproval::Always;
            runtime.acp_mode_prompt = Some(profile.instructions.clone());
        }
    }
    runtime.acp_mode = Some(id.to_owned());
    Ok(())
}

#[cfg(test)]
mod mode_tests {
    use super::super::profiles::{Profile, Profiles, ShellApproval};
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn missing_profile_restores_to_ask() {
        let profiles = Profiles::default();
        assert_eq!(restored_mode_id("removed", &profiles), registry::MODE_ASK);
        assert_eq!(restored_mode_id("yolo", &profiles), registry::MODE_YOLO);
    }

    #[test]
    fn profile_tool_is_turn_scoped_and_falls_back_on_load() {
        let profiles = Profiles(BTreeMap::from([(
            "planning".into(),
            Profile {
                id: "planning".into(),
                name: "Planning".into(),
                description: None,
                shell_approval: ShellApproval::Always,
                instructions: "Read only".into(),
                tools: vec!["todo".into()],
            },
        )]));
        // The registry transition is the same one used on live mode changes
        // and checkpoint restoration; the loop sees only registered names.
        let mut tools = agent_core::ToolRegistry::new();
        tools.register(agent_core::todo::todo_tool()).unwrap();
        assert_eq!(tools.names(), ["todo"]);
        tools.unregister("todo");
        assert!(tools.names().is_empty());
        assert_eq!(profiles.get("planning").unwrap().tools, ["todo"]);
        let state = registry::session_modes(true, Some("planning"), &profiles);
        let wire = serde_json::to_value(state).unwrap();
        assert_eq!(wire["currentModeId"], "planning");
        assert_eq!(wire["availableModes"][0]["id"], "ask");
    }

    #[test]
    fn profile_changes_program_identity() {
        let base = agent_core::AgentLoopOptions {
            shell_requires_approval: true,
            ask_user: true,
            ..Default::default()
        };
        let mut research = base.clone();
        research.infer_system_prompt = Some("ACP mode research: Read first".into());
        let hash = |options: &agent_core::AgentLoopOptions| {
            let machine = agent_core::agent_loop_ir_for_options(
                agent_core::Model("m".into()),
                vec![],
                2,
                options,
            );
            agent_core::program_hash(&machine.program).unwrap()
        };
        assert_ne!(hash(&base), hash(&research));
        assert_eq!(hash(&base), hash(&base));
        let mut with_todo = base.clone();
        with_todo.tool_names.push("todo".into());
        assert_ne!(hash(&base), hash(&with_todo));
    }

    #[test]
    fn profiles_are_advertised_with_metadata() {
        let profiles = Profiles(BTreeMap::from([(
            "research".into(),
            Profile {
                id: "research".into(),
                name: "Research".into(),
                description: Some("Investigate".into()),
                shell_approval: ShellApproval::Always,
                instructions: "Read first".into(),
                tools: vec![],
            },
        )]));
        let state = registry::session_modes(true, Some("research"), &profiles);
        let wire = serde_json::to_value(state).unwrap();
        assert_eq!(wire["currentModeId"], "research");
        assert_eq!(wire["availableModes"][2]["name"], "Research");
        assert_eq!(wire["availableModes"][2]["description"], "Investigate");
    }
}
