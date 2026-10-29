//! The ACP turn driver: runs one prompt turn on a session's runtime,
//! reusing the CLI turn spine's loop entry and `finish_turn` bookkeeping.
//!
//! Approval-gated effects resolve fully in-process: instead of the CLI's
//! durable filesystem pause (`pause_turn` + `agent approvals`), an
//! `AwaitingApproval` outcome becomes a `session/request_permission`
//! round-trip to the ACP client, the decision lands in
//! `ApprovalConfig::resolutions`, and `resume_agent_loop_outcome` re-enters
//! the checkpoint — the same tested resume path the approvals CLI drives.
//!
//! An `ask_user` question (t-1594) is the same pause with kind `question`:
//! it becomes a chooser-shaped permission request (one option per answer
//! plus a dismiss), and the chosen answer rides the resolution back into
//! the machine as the tool result.

use agent_client_protocol::schema::v1::{
    ContentBlock, PermissionOption, PermissionOptionKind, RequestPermissionOutcome,
    RequestPermissionRequest, SessionId, TextContent, ToolCallContent, ToolCallId, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Client, ConnectionTo};
use agent_core::{ApprovalDecision, ApprovalKind, ApprovalResolution, AskUserRequest, ChatMessage};
use anyhow::{anyhow, Result};

use crate::{agent_loop_options, finish_turn, Runtime};

pub(crate) enum TurnOutcome {
    Done(Box<agent_core::Response>, Option<String>),
    /// The client answered a permission request with `cancelled` (part of
    /// the session/cancel contract): the turn ends with no response.
    Cancelled,
}

pub(crate) async fn run_acp_turn(
    runtime: &mut Runtime,
    message: ChatMessage,
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
    // ACP-only boundary: trace effects have no turn id, and checkpoints may
    // omit old prompts after GC. The message id links each turn to retained
    // checkpoint history without changing the default CLI trace shape.
    let user_id = message.id;
    runtime.history.push(message);
    runtime
        .trace
        .emit(&agent_core::Event::Custom {
            run_id: runtime.run_id.clone(),
            name: "acp_turn_start".into(),
            data: serde_json::json!({ "user_message_id": user_id }),
            timestamp: chrono::Utc::now(),
        })
        .await?;
    let mut prompt = runtime.history.clone();
    // A separate system message is turn-local: switching modes never mutates
    // saved conversation history or accumulates stale instructions.
    runtime.acp_mode_message_id = None;
    if let Some(instructions) = runtime.acp_mode_prompt.as_deref().filter(|s| !s.is_empty()) {
        let injected = ChatMessage::system(instructions);
        runtime.acp_mode_message_id = Some(injected.id);
        prompt.insert(1.min(prompt.len()), injected);
    }
    // Options are fixed for the whole turn: they shape the loop program and
    // therefore its effect ids, so approval resumes must re-enter with the
    // program identity the checkpoint was minted under.
    let options = agent_loop_options(runtime);
    // `allow always` stops gating future turns (program-level) AND skips
    // further prompts within this turn's already-gated program.
    let mut allow_all = false;
    let mut approved_mode = None;
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
            agent_core::AgentLoopOutcome::SignalSuspended { .. } => {
                return Err(anyhow::anyhow!("ACP does not support signal suspension"));
            }
            agent_core::AgentLoopOutcome::Complete { value, machine } => {
                return Ok(TurnOutcome::Done(
                    Box::new(finish_turn(runtime, value, machine, prompt).await?),
                    approved_mode,
                ));
            }
            agent_core::AgentLoopOutcome::AwaitingApproval {
                checkpoint,
                pending,
            } => {
                // A question always reaches the user: `allow always` is a
                // verdict on shell commands and cannot pick an answer.
                let (decision, answer) = if pending.kind == ApprovalKind::Mode {
                    match request_mode_permission(cx, session_id, &pending).await? {
                        PermissionReply::AllowOnce => {
                            approved_mode = pending
                                .request
                                .get("mode")
                                .and_then(|v| v.as_str())
                                .map(str::to_owned);
                            (ApprovalDecision::Approve, None)
                        }
                        PermissionReply::Reject | PermissionReply::AllowAlways => {
                            approved_mode = None;
                            (ApprovalDecision::Deny, None)
                        }
                        PermissionReply::Cancelled => return Ok(TurnOutcome::Cancelled),
                    }
                } else if pending.kind == ApprovalKind::Question {
                    match ask_question(cx, session_id, &pending).await? {
                        QuestionReply::Answer(answer) => (ApprovalDecision::Approve, Some(answer)),
                        QuestionReply::Dismissed => (ApprovalDecision::Deny, None),
                        QuestionReply::Cancelled => return Ok(TurnOutcome::Cancelled),
                    }
                } else if allow_all {
                    (ApprovalDecision::Approve, None)
                } else {
                    match request_permission(cx, session_id, &pending).await? {
                        PermissionReply::AllowOnce => (ApprovalDecision::Approve, None),
                        PermissionReply::AllowAlways => {
                            if !runtime.acp_shell_always {
                                allow_all = true;
                                runtime.shell_requires_approval = false;
                            }
                            (ApprovalDecision::Approve, None)
                        }
                        PermissionReply::Reject => (ApprovalDecision::Deny, None),
                        PermissionReply::Cancelled => return Ok(TurnOutcome::Cancelled),
                    }
                };
                runtime.config.approvals.resolutions.insert(
                    pending.effect.effect_id.0.clone(),
                    ApprovalResolution {
                        decision,
                        resolved_by: Some("acp-client".into()),
                        reason: None,
                        answer,
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

/// A single allow_once option avoids Paseo's chooser semantics and never
/// inherits shell's allow-always or yolo bypass.
fn mode_permission_request(
    session_id: &SessionId,
    pending: &agent_core::ApprovalRequest,
) -> RequestPermissionRequest {
    let name = pending.request["name"].as_str().unwrap_or("requested");
    let current = pending.request["current"].as_str().unwrap_or("current");
    let title = format!("Switch to {name} mode");
    RequestPermissionRequest::new(
        session_id.clone(),
        ToolCallUpdate::new(
            ToolCallId::new(pending.effect.effect_id.0.clone()),
            ToolCallUpdateFields::new()
                .kind(ToolKind::SwitchMode)
                .status(ToolCallStatus::Pending)
                .title(title)
                .raw_input(pending.request.clone()),
        ),
        vec![
            PermissionOption::new(
                "allow-once",
                format!("Switch to {name}"),
                PermissionOptionKind::AllowOnce,
            ),
            PermissionOption::new(
                "reject-once",
                format!("Stay in {current}"),
                PermissionOptionKind::RejectOnce,
            ),
        ],
    )
}

async fn request_mode_permission(
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    pending: &agent_core::ApprovalRequest,
) -> Result<PermissionReply> {
    let response = cx
        .send_request(mode_permission_request(session_id, pending))
        .block_task()
        .await
        .map_err(|err| anyhow!("session/request_permission failed: {err}"))?;
    Ok(match response.outcome {
        RequestPermissionOutcome::Selected(selected)
            if selected.option_id.0.as_ref() == "allow-once" =>
        {
            PermissionReply::AllowOnce
        }
        RequestPermissionOutcome::Cancelled => PermissionReply::Cancelled,
        _ => PermissionReply::Reject,
    })
}

enum QuestionReply {
    Answer(String),
    Dismissed,
    Cancelled,
}

/// Option id of the chooser's dismiss button. Answer options use their
/// index ("0", "1", ...), which can never collide with it.
const DISMISS_OPTION_ID: &str = "dismiss";

/// The `session/request_permission` for an `ask_user` question (t-1594),
/// shaped for Paseo's chooser path: one `allow_once` option per answer
/// (two or more same-kind allow options make Paseo render a chooser, and
/// it never auto-accepts choosers), plus one `reject_once` dismiss. Paseo
/// shows the tool call's text content as the prompt under its title.
fn question_permission_request(
    session_id: &SessionId,
    pending: &agent_core::ApprovalRequest,
    question: &AskUserRequest,
) -> RequestPermissionRequest {
    let tool_call = ToolCallUpdate::new(
        ToolCallId::new(pending.effect.effect_id.0.clone()),
        ToolCallUpdateFields::new()
            .kind(ToolKind::Other)
            .status(ToolCallStatus::Pending)
            .title(question.question.clone())
            .content(vec![ToolCallContent::from(ContentBlock::Text(
                TextContent::new(question.question.clone()),
            ))])
            .raw_input(pending.request.clone()),
    );
    let mut options: Vec<PermissionOption> = question
        .options
        .iter()
        .enumerate()
        .map(|(index, answer)| {
            PermissionOption::new(
                index.to_string(),
                answer.clone(),
                PermissionOptionKind::AllowOnce,
            )
        })
        .collect();
    options.push(PermissionOption::new(
        DISMISS_OPTION_ID,
        "Dismiss",
        PermissionOptionKind::RejectOnce,
    ));
    RequestPermissionRequest::new(session_id.clone(), tool_call, options)
}

/// Map the client's choice back to an answer. An unknown option id is a
/// dismissal, never a guessed answer.
fn question_reply(question: &AskUserRequest, outcome: RequestPermissionOutcome) -> QuestionReply {
    match outcome {
        RequestPermissionOutcome::Selected(selected) => selected
            .option_id
            .0
            .parse::<usize>()
            .ok()
            .and_then(|index| question.options.get(index))
            .map_or(QuestionReply::Dismissed, |answer| {
                QuestionReply::Answer(answer.clone())
            }),
        RequestPermissionOutcome::Cancelled => QuestionReply::Cancelled,
        // Non-exhaustive enum: any future outcome is a dismissal.
        _ => QuestionReply::Dismissed,
    }
}

async fn ask_question(
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    pending: &agent_core::ApprovalRequest,
) -> Result<QuestionReply> {
    let question = AskUserRequest::parse(&pending.request)
        .map_err(|err| anyhow!("malformed ask_user question payload: {err}"))?;
    let request = question_permission_request(session_id, pending, &question);
    let response = cx
        .send_request(request)
        .block_task()
        .await
        .map_err(|err| anyhow!("session/request_permission failed: {err}"))?;
    Ok(question_reply(&question, response.outcome))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::SelectedPermissionOutcome;
    use agent_core::{
        effect_location, program_hash, BlockId, DynamicPath, EffectKind, EffectSite, Model,
    };

    #[test]
    fn injected_mode_message_is_removed_even_if_history_moves() {
        let injected = ChatMessage::system("mode instructions");
        let user_system = ChatMessage::system("mode instructions");
        let mut history = vec![
            ChatMessage::user("first"),
            user_system.clone(),
            ChatMessage::user("second"),
            injected.clone(),
        ];
        crate::strip_mode_message(&mut history, injected.id);
        assert_eq!(history.len(), 3);
        assert_eq!(history[1].id, user_system.id);
    }

    fn pending_question() -> (agent_core::ApprovalRequest, AskUserRequest) {
        let machine = agent_core::agent_loop_ir(Model("test-model".into()), vec![], 4);
        let hash = program_hash(&machine.program).unwrap();
        let site = EffectSite {
            block: BlockId(0),
            instruction_index: 0,
        };
        let effect =
            effect_location(hash, EffectKind::Tool, site, DynamicPath::at_entry(0)).unwrap();
        let question = AskUserRequest::parse(&serde_json::json!({
            "question": "Which database should the service use?",
            "options": ["Postgres", "SQLite", "Keep both"]
        }))
        .unwrap();
        let pending = agent_core::ApprovalRequest {
            pending_id: "pa-1".into(),
            effect,
            kind: ApprovalKind::Question,
            request: question.request_value(),
        };
        (pending, question)
    }

    #[test]
    fn question_request_is_a_paseo_chooser() {
        let (pending, question) = pending_question();
        let request = question_permission_request(&SessionId::new("s-1"), &pending, &question);
        let wire = serde_json::to_value(&request).unwrap();

        assert_eq!(
            wire["toolCall"]["toolCallId"],
            pending.effect.effect_id.0.as_str()
        );
        assert_eq!(
            wire["toolCall"]["title"],
            "Which database should the service use?"
        );
        assert_eq!(
            wire["toolCall"]["content"][0]["content"]["text"],
            "Which database should the service use?",
            "Paseo renders the text content as the prompt: {wire}"
        );
        let summary: Vec<(&str, &str, &str)> = wire["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|option| {
                (
                    option["optionId"].as_str().unwrap(),
                    option["name"].as_str().unwrap(),
                    option["kind"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                ("0", "Postgres", "allow_once"),
                ("1", "SQLite", "allow_once"),
                ("2", "Keep both", "allow_once"),
                ("dismiss", "Dismiss", "reject_once"),
            ],
            "two or more same-kind allow options make Paseo render a chooser"
        );
    }

    #[test]
    fn question_outcomes_map_to_answer_dismissal_and_cancel() {
        let (_, question) = pending_question();
        let selected = |id: &str| {
            question_reply(
                &question,
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id.to_owned())),
            )
        };
        assert!(matches!(selected("1"), QuestionReply::Answer(answer) if answer == "SQLite"));
        assert!(matches!(selected("dismiss"), QuestionReply::Dismissed));
        assert!(
            matches!(selected("7"), QuestionReply::Dismissed),
            "an out-of-range id is never a guessed answer"
        );
        assert!(matches!(
            question_reply(&question, RequestPermissionOutcome::Cancelled),
            QuestionReply::Cancelled
        ));
    }
}
