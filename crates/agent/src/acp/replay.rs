//! Reconstruct load notifications from the committed trace, using the live
//! event bridge. Checkpoint history supplies user messages (not trace events).

use std::collections::HashMap;

use agent_client_protocol::schema::v1::SessionUpdate;
use agent_core::{ChatMessage, Event, TraceLogger};
use anyhow::{anyhow, Result};

use super::bridge::{self, ToolCallIds};
use crate::Checkpoint;

pub(super) async fn history_updates(checkpoint: &Checkpoint) -> Vec<SessionUpdate> {
    match TraceLogger::read_events(&checkpoint.trace_path).await {
        Ok(events) => match trace_updates(checkpoint, &events) {
            Ok(updates) => updates,
            Err(err) => {
                eprintln!("warning: ACP session/load trace replay failed ({err}); using text-only history");
                text_history(&checkpoint.messages)
            }
        },
        Err(err) => {
            eprintln!(
                "warning: ACP session/load cannot read trace ({}: {err}); using text-only history",
                checkpoint.trace_path.display()
            );
            text_history(&checkpoint.messages)
        }
    }
}

fn text_history(messages: &[ChatMessage]) -> Vec<SessionUpdate> {
    messages
        .iter()
        .filter_map(|message| {
            let text = message.content.as_deref().filter(|text| !text.is_empty())?;
            match message.role.as_str() {
                "user" => Some(SessionUpdate::UserMessageChunk(bridge::text_chunk(text))),
                "assistant" => Some(SessionUpdate::AgentMessageChunk(bridge::text_chunk(text))),
                _ => None,
            }
        })
        .collect()
}

fn trace_updates(checkpoint: &Checkpoint, events: &[Event]) -> Result<Vec<SessionUpdate>> {
    // The file may contain a cancelled or failed turn after the last saved
    // checkpoint. Never replay effects absent from the loaded history.
    let marker = format!("checkpoint-{:06}", checkpoint.sequence);
    let end = events
        .iter()
        .rposition(|event| {
            matches!(event,
                Event::Checkpoint { name, .. } if name == &marker
            )
        })
        .ok_or_else(|| anyhow!("checkpoint marker {marker} absent from trace"))?;
    let mut users = checkpoint.messages.iter().filter(|m| m.role == "user");
    let mut updates = Vec::new();
    let mut ids = ToolCallIds::default();
    let mut streamed = HashMap::new(); // load never replays streaming deltas
    let mut turn_started = false;
    for event in &events[..=end] {
        match event {
            Event::Checkpoint { .. } => turn_started = false,
            Event::InferCall {
                parent_op_id: None, ..
            } if !turn_started => {
                if let Some(user) = users.next() {
                    if let Some(text) = user.content.as_deref().filter(|text| !text.is_empty()) {
                        updates.push(SessionUpdate::UserMessageChunk(bridge::text_chunk(text)));
                    }
                }
                turn_started = true;
            }
            _ => {}
        }
        updates.extend(bridge::map_event(event, &mut ids, &mut streamed));
    }
    // A cancelled turn can leave a user message in the checkpoint without
    // any InferResult (or even InferCall). Preserve it as text.
    for user in users {
        if let Some(text) = user.content.as_deref().filter(|text| !text.is_empty()) {
            updates.push(SessionUpdate::UserMessageChunk(bridge::text_chunk(text)));
        }
    }
    Ok(updates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::ToolCallStatus;
    use chrono::Utc;

    #[tokio::test]
    async fn missing_trace_falls_back_to_checkpoint_text() {
        let checkpoint: Checkpoint = serde_json::from_value(serde_json::json!({
            "run_id":"r", "sequence":1,"model":"m","provider_url":"",
            "messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}],
            "trace_path":format!("/nonexistent/acp-{}.jsonl", uuid::Uuid::new_v4()),
            "timestamp":Utc::now()
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(history_updates(&checkpoint).await).unwrap(),
            serde_json::to_value(text_history(&checkpoint.messages)).unwrap()
        );
    }

    #[tokio::test]
    async fn load_reuses_live_cards_in_order_without_permission_requests() {
        let dir = std::env::temp_dir().join(format!("acp-load-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.jsonl");
        let time = Utc::now();
        let events = [
            serde_json::json!({"event":"InferCall","run_id":"r","op_id":1,"model":"m","prompt_preview":"","timestamp":time}),
            serde_json::json!({"event":"InferResult","run_id":"r","op_id":1,"response":{"content":"Running shell","tool_calls":[],"input_tokens":1,"output_tokens":1,"total_tokens":2},"response_preview":"","input_tokens":1,"output_tokens":1,"total_tokens":2,"duration_ms":1,"timestamp":time}),
            serde_json::json!({"event":"ApprovalRequested","run_id":"r","pending_id":"p","kind":"eval","request":{"command":"echo hi"},"effect":{
                "effect_id":"e", "kind":"Eval", "program_hash":"h", "site":{"block":0,"instruction_index":0}, "dynamic_path":{"path":"","transitions":0,"visit":0}
            },"timestamp":time}),
            serde_json::json!({"event":"ApprovalResolved","run_id":"r","pending_id":"p","effect_id":"e","kind":"eval","decision":"approved","timestamp":time}),
            serde_json::json!({"event":"EvalCall","run_id":"r","op_id":2,"command":"echo hi","cwd":null,"env_policy":"inherit","timeout_ms":1000,"effect":{
                "effect_id":"e", "kind":"Eval", "program_hash":"h", "site":{"block":0,"instruction_index":0}, "dynamic_path":{"path":"","transitions":0,"visit":0}
            },"timestamp":time}),
            serde_json::json!({"event":"EvalResult","run_id":"r","op_id":2,"command":"echo hi","result":{"stdout":"hi\n"},"duration_ms":1,"truncated_stdout":false,"truncated_stderr":false,"timestamp":time}),
            serde_json::json!({"event":"InferCall","run_id":"r","op_id":3,"parent_op_id":1,"model":"child","prompt_preview":"user: summarize","timestamp":time}),
            serde_json::json!({"event":"InferResult","run_id":"r","op_id":3,"parent_op_id":1,"response":{"content":"summary","tool_calls":[],"input_tokens":1,"output_tokens":1,"total_tokens":2},"response_preview":"summary","input_tokens":1,"output_tokens":1,"total_tokens":2,"duration_ms":1,"timestamp":time}),
            serde_json::json!({"event":"ApprovalRequested","run_id":"r","pending_id":"q","kind":"question","request":{"question":"Choose?","options":["a","b"]},"effect":{
                "effect_id":"q", "kind":"Tool", "program_hash":"h", "site":{"block":0,"instruction_index":0}, "dynamic_path":{"path":"","transitions":0,"visit":0}
            },"timestamp":time}),
            serde_json::json!({"event":"ApprovalResolved","run_id":"r","pending_id":"q","effect_id":"q","kind":"question","decision":"approved","timestamp":time}),
            serde_json::json!({"event":"ToolCall","run_id":"r","op_id":4,"name":"ask_user","arguments":{"question":"Choose?","options":["a","b"]},"effect":{
                "effect_id":"q", "kind":"Tool", "program_hash":"h", "site":{"block":0,"instruction_index":0}, "dynamic_path":{"path":"","transitions":0,"visit":0}
            },"timestamp":time}),
            serde_json::json!({"event":"ToolResult","run_id":"r","op_id":4,"name":"ask_user","result":"b","duration_ms":1,"timestamp":time}),
            serde_json::json!({"event":"Checkpoint","run_id":"r","name":"checkpoint-000001","timestamp":time}),
        ];
        std::fs::write(
            &path,
            events.iter().map(|e| format!("{e}\n")).collect::<String>(),
        )
        .unwrap();
        let checkpoint: Checkpoint = serde_json::from_value(serde_json::json!({
            "run_id":"r", "sequence":1,"model":"m","provider_url":"", "messages":[{"role":"user","content":"please run shell"}],
            "trace_path":path,"timestamp":time
        })).unwrap();
        let parsed = TraceLogger::read_events(&path).await.unwrap();
        let updates = history_updates(&checkpoint).await;
        let mut expected = vec![SessionUpdate::UserMessageChunk(bridge::text_chunk(
            "please run shell",
        ))];
        let mut ids = ToolCallIds::default();
        let mut streamed = HashMap::new();
        for event in &parsed[..parsed.len() - 1] {
            expected.extend(bridge::map_event(event, &mut ids, &mut streamed));
        }
        assert_eq!(
            serde_json::to_value(&updates).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert!(
            matches!(&updates[2], SessionUpdate::ToolCall(card) if card.status == ToolCallStatus::Pending)
        );
        assert!(
            matches!(&updates[5], SessionUpdate::ToolCallUpdate(update) if update.fields.status == Some(ToolCallStatus::Completed))
        );
        assert!(
            matches!(&updates[6], SessionUpdate::ToolCall(card) if card.title == "infer (child): summarize")
        );
        assert!(
            matches!(&updates[8], SessionUpdate::ToolCall(card) if card.title == "Choose?" && card.status == ToolCallStatus::Pending)
        );
        assert!(
            matches!(&updates[11], SessionUpdate::ToolCallUpdate(update) if update.fields.status == Some(ToolCallStatus::Completed))
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
