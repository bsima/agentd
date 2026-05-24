use crate::interpreter::{Tool, ToolMap};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::process::Command;

pub fn standard_tools() -> ToolMap {
    let mut tools: ToolMap = HashMap::new();
    insert(&mut tools, ShellTool);
    insert(&mut tools, SkillTool);
    insert(&mut tools, NotifyTool);
    insert(&mut tools, StopTool);
    tools
}

fn insert<T: Tool + 'static>(tools: &mut ToolMap, tool: T) {
    tools.insert(tool.name().into(), Arc::new(tool));
}

struct ShellTool;
struct SkillTool;
struct NotifyTool;
struct StopTool;

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a command string using the SHELL environment variable."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let command = args
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("shell.command must be a string"))?;
        let shell = std::env::var("SHELL").map_err(|_| {
            anyhow!("SHELL is not set; agentd must provide a shell in the sandbox environment")
        })?;
        let output = Command::new(shell).arg("-c").arg(command).output().await?;
        Ok(json!({
            "status": output.status.code(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr)
        }))
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        "skill"
    }

    fn description(&self) -> &str {
        "agentd integration hook stub for skill discovery/loading."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": { "type": "string" },
                "query": { "type": "string" }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        Ok(json!({ "ok": false, "stub": true, "tool": "skill", "args": args }))
    }
}

#[async_trait]
impl Tool for NotifyTool {
    fn name(&self) -> &str {
        "notify"
    }

    fn description(&self) -> &str {
        "agentd integration hook stub for progress notifications."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "level": { "type": "string" }
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        Ok(json!({ "ok": true, "stub": true, "tool": "notify", "args": args }))
    }
}

#[async_trait]
impl Tool for StopTool {
    fn name(&self) -> &str {
        "stop"
    }

    fn description(&self) -> &str {
        "agentd integration hook stub for explicit done/waiting/blocked stop signals."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "reason": { "type": "string" },
                "message": { "type": "string" }
            },
            "required": ["reason"]
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        Ok(json!({ "ok": true, "stub": true, "tool": "stop", "args": args }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[tokio::test]
    async fn shell_requires_shell_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old_shell = std::env::var_os("SHELL");
        std::env::remove_var("SHELL");

        let result = ShellTool.execute(json!({ "command": "echo hi" })).await;

        if let Some(value) = old_shell {
            std::env::set_var("SHELL", value);
        }
        assert!(result.unwrap_err().to_string().contains("SHELL is not set"));
    }

    #[tokio::test]
    async fn shell_executes_configured_shell() -> Result<()> {
        let _guard = ENV_LOCK.lock().unwrap();
        let old_shell = std::env::var_os("SHELL");
        std::env::set_var("SHELL", "/bin/sh");

        let result = ShellTool
            .execute(json!({ "command": "printf tool-ok" }))
            .await?;

        if let Some(value) = old_shell {
            std::env::set_var("SHELL", value);
        } else {
            std::env::remove_var("SHELL");
        }
        assert_eq!(
            result.get("stdout").and_then(Value::as_str),
            Some("tool-ok")
        );
        Ok(())
    }

    #[test]
    fn standard_tools_are_shell_plus_agentd_hooks() {
        let tools = standard_tools();
        let mut names = tools.keys().cloned().collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, ["notify", "shell", "skill", "stop"]);
    }
}
