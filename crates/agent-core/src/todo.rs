//! Opt-in, in-process step list. Tool results are traced as ordinary native
//! tool results so clients can reconstruct the last successful list on load.

use crate::tool::NativeTool;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};

pub const TODO_TOOL: &str = "todo";

/// Validate the complete replacement list. An empty list clears the plan;
/// at most one item may be in progress at a time.
pub fn todo_items(arguments: &Value) -> Result<Vec<Value>> {
    let items = arguments
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("todo requires `items`: an array of steps"))?;
    let mut active = 0;
    let mut validated = Vec::with_capacity(items.len());
    for item in items {
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("todo item content must be a non-empty string"))?;
        let status = item.get("status").and_then(Value::as_str).unwrap_or("");
        if !matches!(status, "pending" | "in_progress" | "completed") {
            return Err(anyhow!(
                "todo item status must be pending, in_progress, or completed"
            ));
        }
        active += usize::from(status == "in_progress");
        if active > 1 {
            return Err(anyhow!("todo allows at most one in_progress item"));
        }
        validated.push(json!({ "content": content, "status": status }));
    }
    Ok(validated)
}

pub fn todo_tool() -> NativeTool {
    NativeTool::from_fn(
        TODO_TOOL,
        crate::guidance::TODO_TOOL_DESCRIPTION,
        json!({
            "type": "object",
            "properties": {"items": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                },
                "required": ["content", "status"]
            }}},
            "required": ["items"]
        }),
        |arguments| async move {
            let items = todo_items(&arguments)?;
            Ok(json!({ "message": "Plan updated", "items": items }))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn handler_returns_confirmation_and_recoverable_error() {
        let tool = todo_tool();
        let result = tool
            .handler
            .call(json!({"items": [{"content": "test", "status": "pending"}]}))
            .await
            .unwrap();
        assert_eq!(result["message"], "Plan updated");
        assert_eq!(result["items"][0]["content"], "test");
        assert!(tool
            .handler
            .call(json!({"items": [{"content": "", "status": "pending"}]}))
            .await
            .is_err());
    }

    #[test]
    fn validates_replacement_and_clear() {
        assert_eq!(
            todo_items(&json!({"items": []})).unwrap(),
            Vec::<Value>::new()
        );
        assert_eq!(
            todo_items(&json!({"items": [{"content": "  step  ", "status": "in_progress"}]}))
                .unwrap(),
            vec![json!({"content":"step", "status":"in_progress"})]
        );
        for args in [
            json!({}),
            json!({"items": [{"content": " ", "status": "pending"}]}),
            json!({"items": [{"content": "x", "status": "wrong"}]}),
            json!({"items": [{"content": "x", "status": "in_progress"}, {"content": "y", "status": "in_progress"}]}),
        ] {
            assert!(todo_items(&args).is_err(), "{args}");
        }
    }
}
