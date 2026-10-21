//! ACP projection of successful `todo` tool results. Only results update the
//! checklist: malformed calls leave the previous plan intact.
use agent_client_protocol::schema::v1::{
    Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus, SessionUpdate,
};
use agent_core::Event;

pub(super) fn plan_from_result(result: &serde_json::Value) -> Option<SessionUpdate> {
    let items = result.get("items")?.as_array()?;
    let entries = items
        .iter()
        .map(|item| {
            let status = match item.get("status")?.as_str()? {
                "pending" => PlanEntryStatus::Pending,
                "in_progress" => PlanEntryStatus::InProgress,
                "completed" => PlanEntryStatus::Completed,
                _ => return None,
            };
            Some(PlanEntry::new(
                item.get("content")?.as_str()?,
                PlanEntryPriority::Medium,
                status,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(SessionUpdate::Plan(Plan::new(entries)))
}

pub(super) fn latest_plan(events: &[Event]) -> Option<SessionUpdate> {
    events.iter().rev().find_map(|event| match event {
        Event::ToolResult { name, result, .. } if name == agent_core::todo::TODO_TOOL => {
            plan_from_result(result)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn latest_successful_result_replaces_the_list_even_when_empty() {
        let event = |items| Event::ToolResult {
            run_id: "r".into(),
            op_id: 1,
            parent_op_id: None,
            name: "todo".into(),
            result: json!({"items": items}),
            result_preview: String::new(),
            duration_ms: 0,
            timestamp: Utc::now(),
        };
        let first = event(json!([{"content":"Write tests","status":"completed"}]));
        let plan = latest_plan(std::slice::from_ref(&first)).unwrap();
        let wire = serde_json::to_value(plan).unwrap();
        assert_eq!(wire["entries"][0]["content"], "Write tests");
        assert_eq!(wire["entries"][0]["priority"], "medium");
        assert_eq!(wire["entries"][0]["status"], "completed");
        assert_eq!(
            serde_json::to_value(latest_plan(&[first, event(json!([]))]).unwrap()).unwrap()
                ["entries"],
            json!([])
        );
    }
}
