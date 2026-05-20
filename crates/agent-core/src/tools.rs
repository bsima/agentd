use crate::interpreter::{Tool, ToolMap};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::process::Command;

pub fn standard_tools() -> ToolMap {
    let mut tools: ToolMap = HashMap::new();
    insert(&mut tools, BashTool);
    insert(&mut tools, ReadFileTool);
    insert(&mut tools, WriteFileTool);
    tools
}

fn insert<T: Tool + 'static>(tools: &mut ToolMap, tool: T) {
    tools.insert(tool.name().into(), Arc::new(tool));
}

struct BashTool;
struct ReadFileTool;
struct WriteFileTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command in the current working directory."
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
            .ok_or_else(|| anyhow!("bash.command must be a string"))?;
        let output = Command::new("/bin/sh")
            .arg("-lc")
            .arg(command)
            .output()
            .await?;
        Ok(json!({
            "status": output.status.code(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr)
        }))
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a UTF-8 text file."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("read_file.path must be a string"))?;
        let content = tokio::fs::read_to_string(path).await?;
        Ok(json!({ "content": content }))
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write UTF-8 text content to a file, creating parent directories if needed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("write_file.path must be a string"))?;
        let content = args
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("write_file.content must be a string"))?;
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        tokio::fs::write(path, content).await?;
        Ok(json!({ "ok": true }))
    }
}
