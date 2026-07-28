//! The ACP turn driver: runs one prompt turn on a session's runtime,
//! reusing the CLI turn spine's loop entry and `finish_turn` bookkeeping.
//!
//! Approval-gated effects resolve fully in-process: instead of the CLI's
//! durable filesystem pause (`pause_turn` + `agent approvals`), an
//! `AwaitingApproval` outcome becomes a `session/request_permission`
//! round-trip to the ACP client, the decision lands in
//! `ApprovalConfig::resolutions`, and `resume_agent_loop_outcome` re-enters
//! the checkpoint — the same tested resume path the approvals CLI drives.

use agent_client_protocol::schema::v1::{
    PermissionOption, PermissionOptionKind, RequestPermissionOutcome, RequestPermissionRequest,
    SessionId, ToolCallId, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Client, ConnectionTo};
use agent_core::{ApprovalDecision, ApprovalResolution, ChatMessage};
use anyhow::{anyhow, Result};

use crate::{agent_loop_options, finish_turn, Runtime};

pub(crate) enum TurnOutcome {
    Done(Box<agent_core::Response>),
    /// The client answered a permission request with `cancelled` (part of
    /// the session/cancel contract): the turn ends with no response.
    Cancelled,
}

pub(crate) async fn run_acp_turn(
    runtime: &mut Runtime,
    message: String,
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
) -> Result<TurnOutcome> {
    runtime.turn_seq += 1;
    // Approval resolutions are strictly turn-local. They key on effect ids,
    // and a CANCELLED turn never reaches `finish_turn`, so it does not
    // advance `ir_effect_visits` — a later turn can then mint the very same
    // effect id its gated command had. A stale `allow-once` surviving here
    // would silently approve a different command without a prompt.
    runtime.config.approvals.resolutions.clear();
    runtime.history.push(ChatMessage::user(message));
    let prompt = runtime.history.clone();
    // Options are fixed for the whole turn: they shape the loop program and
    // therefore its effect ids, so approval resumes must re-enter with the
    // program identity the checkpoint was minted under.
    let options = agent_loop_options(runtime);
    // `allow always` stops gating future turns (program-level) AND skips
    // further prompts within this turn's already-gated program.
    let mut allow_all = false;
    let mut outcome = agent_core::run_agent_loop_outcome(
        &runtime.config,
        &mut runtime.ir_store,
        runtime.ir_replay.as_ref(),
        &mut runtime.gc_state,
        runtime.model.clone(),
        prompt.clone(),
        runtime.max_turns,
        &options,
        runtime.ir_effect_visits.clone(),
    )
    .await?;
    loop {
        match outcome {
            agent_core::AgentLoopOutcome::Complete { value, machine } => {
                return Ok(TurnOutcome::Done(Box::new(
                    finish_turn(runtime, value, machine, prompt).await?,
                )));
            }
            agent_core::AgentLoopOutcome::AwaitingApproval {
                checkpoint,
                pending,
            } => {
                let decision = if allow_all {
                    ApprovalDecision::Approve
                } else {
                    match request_permission(cx, session_id, &pending).await? {
                        PermissionReply::AllowOnce => ApprovalDecision::Approve,
                        PermissionReply::AllowAlways => {
                            allow_all = true;
                            runtime.shell_requires_approval = false;
                            ApprovalDecision::Approve
                        }
                        PermissionReply::Reject => ApprovalDecision::Deny,
                        PermissionReply::Cancelled => return Ok(TurnOutcome::Cancelled),
                    }
                };
                runtime.config.approvals.resolutions.insert(
                    pending.effect.effect_id.0.clone(),
                    ApprovalResolution {
                        decision,
                        resolved_by: Some("acp-client".into()),
                        reason: None,
                    },
                );
                runtime.ir_store = checkpoint.store.clone();
                outcome = agent_core::resume_agent_loop_outcome(
                    &runtime.config,
                    &mut runtime.ir_store,
                    &mut runtime.gc_state,
                    runtime.model.clone(),
                    runtime.max_turns,
                    &options,
                    checkpoint.machine,
                )
                .await?;
            }
        }
    }
}

enum PermissionReply {
    AllowOnce,
    AllowAlways,
    Reject,
    Cancelled,
}

async fn request_permission(
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    pending: &agent_core::ApprovalRequest,
) -> Result<PermissionReply> {
    // For Eval gates the request preview carries the shell command; Store
    // gates fall back to the kind name. The tool-call id is the effect id,
    // matching the Pending tool_call the bridge already surfaced from the
    // ApprovalRequested trace event.
    let title = pending
        .request
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| pending.kind.as_str().to_owned());
    let tool_call = ToolCallUpdate::new(
        ToolCallId::new(pending.effect.effect_id.0.clone()),
        ToolCallUpdateFields::new()
            .kind(ToolKind::Execute)
            .status(ToolCallStatus::Pending)
            .title(title)
            .raw_input(pending.request.clone()),
    );
    let request = RequestPermissionRequest::new(
        session_id.clone(),
        tool_call,
        vec![
            PermissionOption::new("allow-once", "Allow once", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                "allow-always",
                "Allow for this session",
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new("reject-once", "Reject", PermissionOptionKind::RejectOnce),
        ],
    );
    let response = cx
        .send_request(request)
        .block_task()
        .await
        .map_err(|err| anyhow!("session/request_permission failed: {err}"))?;
    Ok(match response.outcome {
        RequestPermissionOutcome::Selected(selected) => match selected.option_id.0.as_ref() {
            "allow-once" => PermissionReply::AllowOnce,
            "allow-always" => PermissionReply::AllowAlways,
            _ => PermissionReply::Reject,
        },
        RequestPermissionOutcome::Cancelled => PermissionReply::Cancelled,
        // Non-exhaustive enum: any future outcome fails safe.
        _ => PermissionReply::Reject,
    })
}
