//! Reconstruct load notifications from the committed trace, using the live
//! event bridge. Checkpoint history supplies user messages (not trace events).

use std::collections::HashMap;

use agent_client_protocol::schema::v1::SessionUpdate;
use agent_core::{ChatMessage, Event, TraceLogger};
use anyhow::{anyhow, Result};

use super::bridge::{self, ToolCallIds};
use crate::Checkpoint;

pub(super) async fn load_updates(
    checkpoint: &Checkpoint,
) -> (Vec<SessionUpdate>, Option<SessionUpdate>) {
    let events = match TraceLogger::read_events(&checkpoint.trace_path).await {
        Ok(events) => events,
        Err(err) => {
            eprintln!(
                "warning: ACP session/load cannot read trace ({}: {err}); using text-only history",
                checkpoint.trace_path.display()
            );
            return (text_history(&checkpoint.messages), None);
        }
    };
    let committed = match committed_events(checkpoint, &events) {
        Ok(committed) => committed,
        Err(err) => {
            eprintln!(
                "warning: ACP session/load trace replay failed ({err}); using text-only history"
            );
            return (text_history(&checkpoint.messages), None);
        }
    };
    let plan = super::todo::latest_plan(committed);
    let history = trace_updates(checkpoint, committed).unwrap_or_else(|err| {
        eprintln!("warning: ACP session/load trace replay failed ({err}); using text-only history");
        text_history(&checkpoint.messages)
    });
    (history, plan)
}

fn committed_events<'a>(checkpoint: &Checkpoint, events: &'a [Event]) -> Result<&'a [Event]> {
    let marker = format!("checkpoint-{:06}", checkpoint.sequence);
    let end = events
        .iter()
        .rposition(|event| matches!(event, Event::Checkpoint { name, .. } if name == &marker))
        .ok_or_else(|| anyhow!("checkpoint marker {marker} absent from trace"))?;
    Ok(&events[..=end])
}

fn user_update(message: &ChatMessage) -> Option<SessionUpdate> {
    if message.role != "user" {
        return None;
    }
    let text = message.content.as_deref().unwrap_or_default();
    let image = (!message.images.is_empty()).then_some("[image]");
    if text.is_empty() && image.is_none() {
        return None;
    }
    Some(SessionUpdate::UserMessageChunk(bridge::text_chunk(
        match image {
            Some(image) if text.is_empty() => image.to_owned(),
            Some(image) => format!("{text}\n{image}"),
            None => text.to_owned(),
        },
    )))
}

fn text_history(messages: &[ChatMessage]) -> Vec<SessionUpdate> {
    messages
        .iter()
        .filter_map(|message| {
            user_update(message).or_else(|| {
                (message.role == "assistant")
                    .then(|| message.content.as_deref().filter(|s| !s.is_empty()))
                    .flatten()
                    .map(|text| SessionUpdate::AgentMessageChunk(bridge::text_chunk(text)))
            })
        })
        .collect()
}

fn trace_updates(checkpoint: &Checkpoint, events: &[Event]) -> Result<Vec<SessionUpdate>> {
    let visible: Vec<&ChatMessage> = checkpoint
        .messages
        .iter()
        .filter(|m| {
            user_update(m).is_some()
                || (m.role == "assistant" && m.content.as_deref().is_some_and(|s| !s.is_empty()))
        })
        .collect();
    let mut cursor = 0;
    let mut updates = Vec::new();
    let mut ids = ToolCallIds::default();
    let mut streamed = HashMap::new();
    let mut active = false;
    let mut saw_boundary = false;
    let retained_ids: std::collections::HashSet<_> = visible
        .iter()
        .filter(|m| m.role == "user")
        .map(|m| m.id.to_string())
        .collect();
    for event in events {
        if let Event::Custom { name, data, .. } = event {
            if name == "acp_turn_start" {
                saw_boundary = true;
                let marker_id = data
                    .get("user_message_id")
                    .and_then(serde_json::Value::as_str);
                if !marker_id.is_some_and(|id| retained_ids.contains(id)) {
                    // GC-evicted or cancelled turns own all effects up to the
                    // next boundary. Do not assign their cards to a later turn.
                    active = false;
                    continue;
                }
                let Some(message) = visible.get(cursor) else {
                    return Err(anyhow!("trace turn is out of checkpoint order"));
                };
                if message.role != "user" || marker_id != Some(message.id.to_string().as_str()) {
                    return Err(anyhow!("trace turn does not match checkpoint prompt"));
                }
                active = true;
                updates.push(user_update(message).expect("visible user message"));
                cursor += 1;
                continue;
            }
        }
        if !active {
            continue;
        }
        for update in bridge::map_event(event, &mut ids, &mut streamed) {
            if let SessionUpdate::AgentMessageChunk(chunk) = &update {
                let agent_client_protocol::schema::v1::ContentBlock::Text(text) = &chunk.content
                else {
                    return Err(anyhow!("non-text assistant chunk"));
                };
                // Intermediate inference text need not survive the agent
                // loop's saved history; only emit it if it matches the next
                // retained assistant message.
                if visible.get(cursor).is_some_and(|m| {
                    m.role == "assistant" && m.content.as_deref() == Some(text.text.as_str())
                }) {
                    cursor += 1;
                } else {
                    continue;
                }
            }
            updates.push(update);
        }
    }
    if !saw_boundary || cursor != visible.len() {
        return Err(anyhow!(
            "trace does not account for checkpoint conversation"
        ));
    }
    Ok(updates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::ToolCallStatus;
    use chrono::Utc;

    fn checkpoint(messages: Vec<ChatMessage>) -> Checkpoint {
        let mut cp: Checkpoint = serde_json::from_value(serde_json::json!({
            "run_id":"r", "sequence":1, "model":"m", "provider_url":"",
            "messages":[], "trace_path":"/unused", "timestamp":Utc::now()
        }))
        .unwrap();
        cp.messages = messages;
        cp
    }

    fn start(message: &ChatMessage) -> Event {
        Event::Custom {
            run_id: "r".into(),
            name: "acp_turn_start".into(),
            data: serde_json::json!({"user_message_id":message.id}),
            timestamp: Utc::now(),
        }
    }

    fn answer(content: &str, op_id: u64) -> Event {
        serde_json::from_value(serde_json::json!({
            "event":"InferResult", "run_id":"r", "op_id":op_id,
            "response":{"content":content,"tool_calls":[],"input_tokens":1,"output_tokens":1,"total_tokens":2},
            "response_preview":content,"input_tokens":1,"output_tokens":1,"total_tokens":2,
            "duration_ms":1,"timestamp":Utc::now()
        })).unwrap()
    }

    fn wire(updates: &[SessionUpdate]) -> serde_json::Value {
        serde_json::to_value(updates).unwrap()
    }

    #[test]
    fn image_only_prompt_keeps_its_tool_card_on_replay() {
        let image = ChatMessage::user_with_images(
            None,
            vec![agent_core::ImageContent::new("image/png", "aGVsbG8=")],
        );
        let cp = checkpoint(vec![
            image.clone(),
            ChatMessage::assistant(Some("seen".into()), vec![]),
        ]);
        let card: Event = serde_json::from_value(serde_json::json!({
            "event":"ToolCall", "run_id":"r", "op_id":4,"name":"todo",
            "arguments":{"items":[]},"timestamp":Utc::now()
        }))
        .unwrap();
        let updates = trace_updates(&cp, &[start(&image), card, answer("seen", 5)]).unwrap();
        let rendered = wire(&updates);
        assert_eq!(rendered[0]["content"]["text"], "[image]");
        assert_eq!(rendered[1]["sessionUpdate"], "tool_call");
        assert_eq!(rendered[2]["content"]["text"], "seen");
        assert_eq!(
            wire(&text_history(&cp.messages))[0]["content"]["text"],
            "[image]"
        );
    }

    #[test]
    fn gc_removed_early_prompt_does_not_steal_later_tool_cards() {
        let old = ChatMessage::user("removed");
        let later = ChatMessage::user("retained");
        let cp = checkpoint(vec![
            later.clone(),
            ChatMessage::assistant(Some("reply".into()), vec![]),
        ]);
        let card: Event = serde_json::from_value(serde_json::json!({
            "event":"ToolCall", "run_id":"r", "op_id":4,"name":"shell",
            "arguments":{"command":"old"},"timestamp":Utc::now()
        }))
        .unwrap();
        let later_card: Event = serde_json::from_value(serde_json::json!({
            "event":"ToolCall", "run_id":"r", "op_id":6,"name":"shell",
            "arguments":{"command":"retained"},"timestamp":Utc::now()
        }))
        .unwrap();
        let events = [
            start(&old),
            card,
            start(&later),
            later_card,
            answer("reply", 5),
        ];
        let updates = wire(&trace_updates(&cp, &events).unwrap());
        assert_eq!(updates.as_array().unwrap().len(), 3);
        assert_eq!(updates[1]["rawInput"]["command"], "retained");
        assert_eq!(
            wire(&text_history(&cp.messages))[0]["content"]["text"],
            "retained"
        );
    }

    #[test]
    fn cancelled_turn_without_checkpoint_stays_before_completed_turn() {
        let cancelled = ChatMessage::user("cancelled");
        let completed = ChatMessage::user("completed");
        let cp = checkpoint(vec![
            cancelled.clone(),
            completed.clone(),
            ChatMessage::assistant(Some("done".into()), vec![]),
        ]);
        let cancelled_card: Event = serde_json::from_value(serde_json::json!({
            "event":"ToolCall", "run_id":"r", "op_id":1,"name":"shell",
            "arguments":{"command":"cancelled tool"},"timestamp":Utc::now()
        }))
        .unwrap();
        let completed_card: Event = serde_json::from_value(serde_json::json!({
            "event":"ToolCall", "run_id":"r", "op_id":3,"name":"shell",
            "arguments":{"command":"completed tool"},"timestamp":Utc::now()
        }))
        .unwrap();
        let updates = trace_updates(
            &cp,
            &[
                start(&cancelled),
                cancelled_card,
                start(&completed),
                completed_card,
                answer("done", 2),
            ],
        )
        .unwrap();
        let wire = wire(&updates);
        assert_eq!(wire[0]["content"]["text"], "cancelled");
        assert_eq!(wire[1]["title"], "shell");
        assert_eq!(wire[2]["content"]["text"], "completed");
        assert_eq!(wire[3]["rawInput"]["command"], "completed tool");
        assert_eq!(wire[4]["content"]["text"], "done");
    }

    #[test]
    fn cancelled_live_load_turn_does_not_hide_later_cards() {
        let cancelled = ChatMessage::user("cancelled by load");
        let completed = ChatMessage::user("completed");
        let cp = checkpoint(vec![
            completed.clone(),
            ChatMessage::assistant(Some("done".into()), vec![]),
        ]);
        let card: Event = serde_json::from_value(serde_json::json!({
            "event":"ToolCall", "run_id":"r", "op_id":3,"name":"shell",
            "arguments":{"command":"completed tool"},"timestamp":Utc::now()
        }))
        .unwrap();
        let updates = wire(
            &trace_updates(
                &cp,
                &[
                    start(&cancelled),
                    start(&completed),
                    card,
                    answer("done", 2),
                ],
            )
            .unwrap(),
        );
        assert_eq!(updates.as_array().unwrap().len(), 3);
        assert_eq!(updates[1]["rawInput"]["command"], "completed tool");
    }

    #[test]
    fn truncated_trace_falls_back_instead_of_dropping_old_assistant() {
        let old = ChatMessage::user("old");
        let recent = ChatMessage::user("recent");
        let cp = checkpoint(vec![
            old,
            ChatMessage::assistant(Some("old answer".into()), vec![]),
            recent.clone(),
            ChatMessage::assistant(Some("new answer".into()), vec![]),
        ]);
        let events = [start(&recent), answer("new answer", 2)];
        assert!(trace_updates(&cp, &events).is_err());
        assert_eq!(
            wire(&text_history(&cp.messages))[1]["content"]["text"],
            "old answer"
        );
    }

    #[tokio::test]
    async fn plan_is_bounded_by_the_same_checkpoint_as_history() {
        let cp = checkpoint(vec![]);
        let todo = |content: &str| {
            serde_json::from_value::<Event>(serde_json::json!({
                "event":"ToolResult","run_id":"r","op_id":1,"name":"todo",
                "result":{"items":[{"content":content,"status":"pending"}]},
                "result_preview":"","duration_ms":0,"timestamp":Utc::now()
            }))
            .unwrap()
        };
        let marker: Event = serde_json::from_value(serde_json::json!({
            "event":"Checkpoint","run_id":"r","name":"checkpoint-000001","timestamp":Utc::now()
        }))
        .unwrap();
        let events = [todo("committed"), marker, todo("uncommitted")];
        let dir = std::env::temp_dir().join(format!("acp-plan-load-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.jsonl");
        std::fs::write(
            &path,
            events
                .iter()
                .map(|event| format!("{}\n", serde_json::to_string(event).unwrap()))
                .collect::<String>(),
        )
        .unwrap();
        let mut cp = cp;
        cp.trace_path = path;
        let (history, plan) = load_updates(&cp).await;
        assert!(history.is_empty());
        assert_eq!(
            serde_json::to_value(plan.unwrap()).unwrap()["entries"][0]["content"],
            "committed"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

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
            serde_json::to_value(load_updates(&checkpoint).await.0).unwrap(),
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
            serde_json::json!({"event":"Custom","run_id":"r","name":"acp_turn_start","data":{"user_message_id":"00000000-0000-0000-0000-000000000001"},"timestamp":time}),
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
            "run_id":"r", "sequence":1,"model":"m","provider_url":"", "messages":[{"id":"00000000-0000-0000-0000-000000000001","role":"user","content":"please run shell"},{"role":"assistant","content":"Running shell"}],
            "trace_path":path,"timestamp":time
        })).unwrap();
        let parsed = TraceLogger::read_events(&path).await.unwrap();
        let updates = load_updates(&checkpoint).await.0;
        let mut expected = vec![SessionUpdate::UserMessageChunk(bridge::text_chunk(
            "please run shell",
        ))];
        let mut ids = ToolCallIds::default();
        let mut streamed = HashMap::new();
        for event in &parsed[1..parsed.len() - 1] {
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
