//! Bridge from agent-core runtime trace events to ACP `session/update`
//! notifications. The sink taps the session runtime's `TraceLogger` (full,
//! untruncated payloads — unlike the public_trace projection) and forwards
//! mapped updates over a channel; a per-connection forwarder task delivers
//! them as `SessionNotification`s so `TraceSink::emit` never blocks a turn.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, SessionUpdate, TextContent, ToolCall, ToolCallContent, ToolCallId,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_core::Event;
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;

/// Correlates runtime effect executions to stable ACP tool-call ids. Calls
/// carry an `EffectLocation` (stable across an approval pause and the later
/// execution of the same effect) and an `op_id`; results carry only the
/// `op_id`.
#[derive(Default)]
pub(crate) struct ToolCallIds {
    by_effect: HashMap<String, ToolCallId>,
    by_op: HashMap<u64, ToolCallId>,
}

impl ToolCallIds {
    /// Id for an effect-bearing call event: reuse the effect's id (minted at
    /// e.g. `ApprovalRequested`) and register the op for result correlation.
    fn for_call(&mut self, effect_id: Option<&str>, op_id: u64) -> ToolCallId {
        let id = match effect_id {
            Some(effect_id) => self
                .by_effect
                .entry(effect_id.to_owned())
                .or_insert_with(|| ToolCallId::new(effect_id.to_owned()))
                .clone(),
            None => ToolCallId::new(format!("op-{op_id}")),
        };
        self.by_op.insert(op_id, id.clone());
        id
    }

    /// Id for an effect-only event (`ApprovalRequested` has no op yet).
    fn for_effect(&mut self, effect_id: &str) -> ToolCallId {
        self.by_effect
            .entry(effect_id.to_owned())
            .or_insert_with(|| ToolCallId::new(effect_id.to_owned()))
            .clone()
    }

    /// Id for a result event: the matching call registered the op.
    fn for_result(&mut self, op_id: u64) -> ToolCallId {
        self.by_op
            .remove(&op_id)
            .unwrap_or_else(|| ToolCallId::new(format!("op-{op_id}")))
    }
}

pub(crate) fn text_chunk(text: impl Into<String>) -> ContentChunk {
    ContentChunk::new(ContentBlock::Text(TextContent::new(text.into())))
}

fn tool_text(text: impl Into<String>) -> ToolCallContent {
    ToolCallContent::from(ContentBlock::Text(TextContent::new(text.into())))
}

fn effect_id(effect: &Option<Box<agent_core::EffectLocation>>) -> Option<&str> {
    effect
        .as_deref()
        .map(|location| location.effect_id.0.as_str())
}

/// Native-tool name → the closest ACP tool kind, so clients pick sensible
/// icons. The shell tool is Eval (Execute); memory reads are searches.
fn tool_kind(name: &str) -> ToolKind {
    match name {
        "recall" => ToolKind::Search,
        _ => ToolKind::Other,
    }
}

/// Streamed text per live Infer op, appended by the session's
/// `on_infer_delta` tap and reconciled (and removed) at `InferResult` time
/// by the suppression rule below. Shared between the tap closure and the
/// sink.
pub(crate) type StreamedText = Arc<Mutex<HashMap<u64, String>>>;

/// What flows to the per-session forwarder task. `Flush` is the turn
/// barrier: the forwarder acks it only after every update enqueued before
/// it has been handed to the connection, so the actor can guarantee all of
/// a turn's `session/update` notifications precede its `session/prompt`
/// response (ACP clients treat that response as the turn boundary).
pub(crate) enum ForwarderMsg {
    Update(Box<SessionUpdate>),
    Flush(tokio::sync::oneshot::Sender<()>),
}

impl ForwarderMsg {
    pub(crate) fn update(update: SessionUpdate) -> Self {
        Self::Update(Box::new(update))
    }
}

/// Map one runtime trace event to its ACP session updates. Pure — the sink
/// owns the id map and the channel. Runtime-internal events (hydration, GC,
/// checkpoints, par bookkeeping) map to nothing.
pub(crate) fn map_event(
    event: &Event,
    ids: &mut ToolCallIds,
    streamed: &mut HashMap<u64, String>,
) -> Vec<SessionUpdate> {
    match event {
        // Whole-message assistant text: every completed inference with
        // content is a message chunk. `PromptResponse` carries no content in
        // ACP, so this is the only channel delivering the answer itself.
        // Suppression rule when the streaming tap already delivered deltas
        // for this op: identical accumulated text ⇒ emit nothing; divergent
        // text (a mid-stream transport retry re-emitted fragments) ⇒ one
        // corrective whole-message chunk with the authoritative content —
        // the client may briefly show duplicated text, which beats showing
        // wrong text. No streamed text (non-streaming provider, replay,
        // default `chat_streamed` fallback) ⇒ the whole-message chunk,
        // exactly the pre-streaming behavior.
        Event::InferResult {
            op_id, response, ..
        } => {
            let streamed_text = streamed.remove(op_id);
            let Some(response) = response else {
                return Vec::new();
            };
            if response.content.is_empty() {
                return Vec::new();
            }
            match streamed_text {
                Some(text) if text == response.content => Vec::new(),
                _ => vec![SessionUpdate::AgentMessageChunk(text_chunk(
                    response.content.clone(),
                ))],
            }
        }
        Event::EvalCall {
            op_id,
            command,
            cwd,
            effect,
            ..
        } => {
            let id = ids.for_call(effect_id(effect), *op_id);
            vec![SessionUpdate::ToolCall(
                ToolCall::new(id, command.clone())
                    .kind(ToolKind::Execute)
                    .status(ToolCallStatus::InProgress)
                    .raw_input(serde_json::json!({ "command": command, "cwd": cwd })),
            )]
        }
        Event::EvalResult { op_id, result, .. } => {
            let id = ids.for_result(*op_id);
            let stdout = result
                .get("stdout")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let mut fields = ToolCallUpdateFields::new()
                .status(ToolCallStatus::Completed)
                .raw_output(result.clone());
            if !stdout.is_empty() {
                fields = fields.content(vec![tool_text(stdout)]);
            }
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id, fields,
            ))]
        }
        Event::EvalError { op_id, error, .. } => {
            let id = ids.for_result(*op_id);
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Failed)
                    .content(vec![tool_text(error.clone())]),
            ))]
        }
        Event::ToolCall {
            op_id,
            name,
            arguments,
            effect,
            ..
        } => {
            let id = ids.for_call(effect_id(effect), *op_id);
            vec![SessionUpdate::ToolCall(
                ToolCall::new(id, name.clone())
                    .kind(tool_kind(name))
                    .status(ToolCallStatus::InProgress)
                    .raw_input(arguments.clone()),
            )]
        }
        Event::ToolResult { op_id, result, .. } => {
            let id = ids.for_result(*op_id);
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Completed)
                    .raw_output(result.clone()),
            ))]
        }
        Event::ToolError { op_id, error, .. } => {
            let id = ids.for_result(*op_id);
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Failed)
                    .content(vec![tool_text(error.clone())]),
            ))]
        }
        Event::RetrieveCall {
            op_id,
            query,
            effect,
            ..
        } => {
            let id = ids.for_call(effect_id(effect), *op_id);
            vec![SessionUpdate::ToolCall(
                ToolCall::new(id, format!("recall: {query}"))
                    .kind(ToolKind::Search)
                    .status(ToolCallStatus::InProgress)
                    .raw_input(serde_json::json!({ "query": query })),
            )]
        }
        Event::RetrieveResult { op_id, results, .. } => {
            let id = ids.for_result(*op_id);
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Completed)
                    .raw_output(results.clone()),
            ))]
        }
        Event::RetrieveError { op_id, error, .. } => {
            let id = ids.for_result(*op_id);
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Failed)
                    .content(vec![tool_text(error.clone())]),
            ))]
        }
        Event::StoreCall {
            op_id,
            sink,
            store_op,
            item_preview,
            effect,
            ..
        } => {
            let id = ids.for_call(effect_id(effect), *op_id);
            vec![SessionUpdate::ToolCall(
                ToolCall::new(id, format!("remember ({store_op}): {sink}"))
                    .status(ToolCallStatus::InProgress)
                    .raw_input(serde_json::json!({
                        "sink": sink,
                        "op": store_op,
                        "item_preview": item_preview,
                    })),
            )]
        }
        Event::StoreResult { op_id, sink_id, .. } => {
            let id = ids.for_result(*op_id);
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Completed)
                    .raw_output(serde_json::json!({ "id": sink_id })),
            ))]
        }
        Event::StoreError { op_id, error, .. } => {
            let id = ids.for_result(*op_id);
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new()
                    .status(ToolCallStatus::Failed)
                    .content(vec![tool_text(error.clone())]),
            ))]
        }
        // The gated call surfaces before (and alongside) the permission
        // prompt; the later EvalCall for the same effect reuses this id.
        Event::ApprovalRequested {
            kind,
            request,
            effect,
            ..
        } => {
            let id = ids.for_effect(effect.effect_id.0.as_str());
            let title = request
                .get("command")
                .and_then(|value| value.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| kind.clone());
            vec![SessionUpdate::ToolCall(
                ToolCall::new(id, title)
                    .kind(ToolKind::Execute)
                    .status(ToolCallStatus::Pending)
                    .raw_input(request.clone()),
            )]
        }
        Event::ApprovalResolved {
            effect_id,
            decision,
            ..
        } => {
            let id = ids.for_effect(effect_id.as_str());
            let status = if decision == "approve" {
                ToolCallStatus::InProgress
            } else {
                ToolCallStatus::Failed
            };
            vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                id,
                ToolCallUpdateFields::new().status(status),
            ))]
        }
        Event::TurnBudgetExhausted {
            max_turns,
            pending_tool_calls,
            ..
        } => {
            let mut notice = format!("[turn budget exhausted: stopped after {max_turns} turns");
            if *pending_tool_calls > 0 {
                notice.push_str(&format!(
                    "; {pending_tool_calls} pending tool call(s) were not executed"
                ));
            }
            notice.push(']');
            vec![SessionUpdate::AgentMessageChunk(text_chunk(notice))]
        }
        // Runtime-internal: never surfaced to ACP clients.
        Event::InferCall { .. }
        | Event::InferError { .. }
        | Event::HydrationStart { .. }
        | Event::HydrationSection { .. }
        | Event::HydrationEnd { .. }
        | Event::ParStart { .. }
        | Event::ParEnd { .. }
        | Event::Checkpoint { .. }
        | Event::AgentDone { .. }
        | Event::Custom { .. } => Vec::new(),
    }
}

/// Trace sink observing one session's runtime events. Mapped updates go to
/// the connection forwarder over an unbounded channel: emission never fails
/// the turn, even after the client is gone.
pub(crate) struct AcpTraceSink {
    tx: UnboundedSender<ForwarderMsg>,
    ids: Mutex<ToolCallIds>,
    streamed: StreamedText,
}

impl AcpTraceSink {
    pub(crate) fn new(tx: UnboundedSender<ForwarderMsg>, streamed: StreamedText) -> Self {
        Self {
            tx,
            ids: Mutex::new(ToolCallIds::default()),
            streamed,
        }
    }
}

#[async_trait]
impl agent_core::TraceSink for AcpTraceSink {
    async fn emit(&self, event: &Event) -> Result<()> {
        let updates = {
            let mut ids = self.ids.lock().expect("tool-call id map poisoned");
            let mut streamed = self.streamed.lock().expect("streamed text map poisoned");
            map_event(event, &mut ids, &mut streamed)
        };
        for update in updates {
            let _ = self.tx.send(ForwarderMsg::update(update));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{
        effect_location, program_hash, BlockId, DynamicPath, EffectKind, EffectSite, Model,
    };
    use chrono::Utc;

    fn test_effect() -> agent_core::EffectLocation {
        let machine = agent_core::agent_loop_ir(Model("test-model".into()), vec![], 4);
        let hash = program_hash(&machine.program).unwrap();
        let site = EffectSite {
            block: BlockId(0),
            instruction_index: 0,
        };
        effect_location(hash, EffectKind::Eval, site, DynamicPath::at_entry(0)).unwrap()
    }

    fn update_names(updates: &[SessionUpdate]) -> Vec<&'static str> {
        updates
            .iter()
            .map(|update| match update {
                SessionUpdate::AgentMessageChunk(_) => "agent_message_chunk",
                SessionUpdate::ToolCall(_) => "tool_call",
                SessionUpdate::ToolCallUpdate(_) => "tool_call_update",
                _ => "other",
            })
            .collect()
    }

    #[test]
    fn infer_result_maps_to_whole_message_chunk() {
        let mut ids = ToolCallIds::default();
        let event = Event::InferResult {
            run_id: "r".into(),
            op_id: 2,
            parent_op_id: None,
            response: Some(agent_core::Response {
                content: "hello".into(),
                tool_calls: vec![],
                finish_reason: None,
                input_tokens: 1,
                output_tokens: 1,
                total_tokens: 2,
                cached_input_tokens: None,
                cost_micro_usd: None,
                pricing: None,
                metadata: Default::default(),
            }),
            response_preview: "hello".into(),
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            duration_ms: 1,
            timestamp: Utc::now(),
        };
        let updates = map_event(&event, &mut ids, &mut HashMap::new());
        assert_eq!(update_names(&updates), ["agent_message_chunk"]);
        let SessionUpdate::AgentMessageChunk(chunk) = &updates[0] else {
            unreachable!()
        };
        let ContentBlock::Text(text) = &chunk.content else {
            panic!("expected text chunk")
        };
        assert_eq!(text.text, "hello");
    }

    #[test]
    fn fully_streamed_infer_result_is_suppressed() {
        let mut ids = ToolCallIds::default();
        let mut streamed = HashMap::from([(2_u64, "hello".to_owned())]);
        let event = infer_result_event("hello");
        assert!(
            map_event(&event, &mut ids, &mut streamed).is_empty(),
            "deltas already delivered the text"
        );
        assert!(streamed.is_empty(), "reconciled entries are removed");
    }

    #[test]
    fn divergent_streamed_text_emits_a_corrective_chunk() {
        let mut ids = ToolCallIds::default();
        // A mid-stream retry re-emitted fragments: accumulated text differs
        // from the authoritative response.
        let mut streamed = HashMap::from([(2_u64, "helhello".to_owned())]);
        let event = infer_result_event("hello");
        let updates = map_event(&event, &mut ids, &mut streamed);
        assert_eq!(update_names(&updates), ["agent_message_chunk"]);
        let SessionUpdate::AgentMessageChunk(chunk) = &updates[0] else {
            unreachable!()
        };
        let ContentBlock::Text(text) = &chunk.content else {
            panic!("expected text chunk")
        };
        assert_eq!(text.text, "hello", "the authoritative content wins");
    }

    fn infer_result_event(content: &str) -> Event {
        Event::InferResult {
            run_id: "r".into(),
            op_id: 2,
            parent_op_id: None,
            response: Some(agent_core::Response {
                content: content.into(),
                tool_calls: vec![],
                finish_reason: None,
                input_tokens: 1,
                output_tokens: 1,
                total_tokens: 2,
                cached_input_tokens: None,
                cost_micro_usd: None,
                pricing: None,
                metadata: Default::default(),
            }),
            response_preview: content.into(),
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            duration_ms: 1,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn empty_infer_result_maps_to_nothing() {
        let mut ids = ToolCallIds::default();
        let event = Event::InferResult {
            run_id: "r".into(),
            op_id: 2,
            parent_op_id: None,
            response: None,
            response_preview: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            duration_ms: 1,
            timestamp: Utc::now(),
        };
        assert!(map_event(&event, &mut ids, &mut HashMap::new()).is_empty());
    }

    #[test]
    fn eval_call_and_result_correlate_by_op() {
        let mut ids = ToolCallIds::default();
        let call = Event::EvalCall {
            run_id: "r".into(),
            op_id: 5,
            parent_op_id: None,
            command: "ls".into(),
            argv: None,
            cwd: Some("/tmp".into()),
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: Some(Box::new(test_effect())),
            timestamp: Utc::now(),
        };
        let updates = map_event(&call, &mut ids, &mut HashMap::new());
        assert_eq!(update_names(&updates), ["tool_call"]);
        let SessionUpdate::ToolCall(tool_call) = &updates[0] else {
            unreachable!()
        };
        assert_eq!(tool_call.title, "ls");
        assert_eq!(tool_call.kind, ToolKind::Execute);
        assert_eq!(tool_call.status, ToolCallStatus::InProgress);
        let call_id = tool_call.tool_call_id.clone();

        let result = Event::EvalResult {
            run_id: "r".into(),
            op_id: 5,
            parent_op_id: None,
            command: "ls".into(),
            result: serde_json::json!({"ok": true, "status": 0, "stdout": "file\n", "stderr": ""}),
            duration_ms: 3,
            truncated_stdout: false,
            truncated_stderr: false,
            timestamp: Utc::now(),
        };
        let updates = map_event(&result, &mut ids, &mut HashMap::new());
        assert_eq!(update_names(&updates), ["tool_call_update"]);
        let SessionUpdate::ToolCallUpdate(update) = &updates[0] else {
            unreachable!()
        };
        assert_eq!(
            update.tool_call_id, call_id,
            "result correlates to the call id"
        );
        assert_eq!(update.fields.status, Some(ToolCallStatus::Completed));
        assert!(update.fields.raw_output.is_some());
    }

    #[test]
    fn approval_then_eval_share_the_effect_id() {
        let mut ids = ToolCallIds::default();
        let effect = test_effect();
        let requested = Event::ApprovalRequested {
            run_id: "r".into(),
            pending_id: "pa-1".into(),
            kind: "eval".into(),
            request: serde_json::json!({"command": "rm -rf /tmp/x"}),
            effect: Box::new(effect.clone()),
            timestamp: Utc::now(),
        };
        let updates = map_event(&requested, &mut ids, &mut HashMap::new());
        let SessionUpdate::ToolCall(pending_call) = &updates[0] else {
            panic!("expected tool_call, got {updates:?}")
        };
        assert_eq!(pending_call.status, ToolCallStatus::Pending);
        assert_eq!(pending_call.title, "rm -rf /tmp/x");

        let resolved = Event::ApprovalResolved {
            run_id: "r".into(),
            pending_id: "pa-1".into(),
            effect_id: effect.effect_id.0.clone(),
            kind: "eval".into(),
            decision: "approve".into(),
            resolved_by: Some("paseo".into()),
            reason: None,
            timestamp: Utc::now(),
        };
        let updates = map_event(&resolved, &mut ids, &mut HashMap::new());
        let SessionUpdate::ToolCallUpdate(update) = &updates[0] else {
            panic!("expected tool_call_update, got {updates:?}")
        };
        assert_eq!(update.tool_call_id, pending_call.tool_call_id);
        assert_eq!(update.fields.status, Some(ToolCallStatus::InProgress));

        let eval = Event::EvalCall {
            run_id: "r".into(),
            op_id: 9,
            parent_op_id: None,
            command: "rm -rf /tmp/x".into(),
            argv: None,
            cwd: None,
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: Some(Box::new(effect)),
            timestamp: Utc::now(),
        };
        let updates = map_event(&eval, &mut ids, &mut HashMap::new());
        let SessionUpdate::ToolCall(executing) = &updates[0] else {
            panic!("expected tool_call, got {updates:?}")
        };
        assert_eq!(
            executing.tool_call_id, pending_call.tool_call_id,
            "the executing eval reuses the pending approval's tool-call id"
        );
    }

    #[test]
    fn internal_events_map_to_nothing() {
        let mut ids = ToolCallIds::default();
        let event = Event::AgentDone {
            run_id: "r".into(),
            usage: None,
            timestamp: Utc::now(),
        };
        assert!(map_event(&event, &mut ids, &mut HashMap::new()).is_empty());
    }
}
