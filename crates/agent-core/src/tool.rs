//! Native (typed) tools for the agent loop (t-1308.7, SDK PRD).
//!
//! A native tool is an in-process async handler registered with the runtime
//! under a name, with a description and JSON-Schema parameters. Registered
//! tools ride the same mechanism as the built-in shell/infer/remember/recall
//! tools: the loop program grows a dispatch arm per registered name (see
//! [`crate::ir_agent::agent_loop_ir_with_tools`]), the tool list shown to
//! the provider includes their specs, and each invocation executes as a
//! first-class IR effect ([`crate::ir::Instr::Tool`]) with stable effect
//! identity, `ToolCall`/`ToolResult`/`ToolError` trace events, and
//! effect-id replay that returns the recorded result without invoking the
//! handler.
//!
//! A registered tool is dispatched by exact name match *before* the
//! unknown-tool fallthrough and entirely apart from the shell tool: a native
//! tool call never becomes a `$SHELL -c` invocation.

use crate::provider::{ToolFunctionSpec, ToolSpec};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

/// Tool names owned by the built-in loop dispatch arms. A native tool may
/// not shadow them: the built-ins are matched first, so a same-named
/// registration would silently never fire.
pub const RESERVED_TOOL_NAMES: &[&str] = &["shell", "infer", "remember", "recall"];

/// An async native tool handler: JSON arguments in, JSON result out. Errors
/// follow the runtime's errors-as-values convention at Bind sites (the agent
/// loop's dispatch arms): a failed handler becomes a tool result the model
/// can read and recover from, not a turn abort.
#[async_trait]
pub trait ToolHandler: Send + Sync {
    async fn call(&self, arguments: Value) -> Result<Value>;
}

struct FnToolHandler<F> {
    f: F,
}

#[async_trait]
impl<F, Fut> ToolHandler for FnToolHandler<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Value>> + Send,
{
    async fn call(&self, arguments: Value) -> Result<Value> {
        (self.f)(arguments).await
    }
}

/// A registered native tool: the spec advertised to the model plus the
/// handler the interpreter dispatches to. Cloning shares the handler.
#[derive(Clone)]
pub struct NativeTool {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments (the provider-facing
    /// `parameters` object).
    pub parameters: Value,
    pub handler: Arc<dyn ToolHandler>,
}

impl std::fmt::Debug for NativeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl NativeTool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        handler: Arc<dyn ToolHandler>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            handler,
        }
    }

    /// Register an async closure as the handler.
    pub fn from_fn<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        f: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        Self::new(name, description, parameters, Arc::new(FnToolHandler { f }))
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            kind: "function".into(),
            function: ToolFunctionSpec {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
            },
        }
    }
}

/// The runtime's native tool table, carried on
/// [`crate::interpreter::SeqConfig`]. Registration is the exposure switch
/// (same principle as the memory tools): a registered tool is advertised to
/// the model and dispatchable; nothing else is.
#[derive(Clone, Debug, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, NativeTool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool. Rejects empty names, names reserved by the built-in
    /// dispatch arms, and duplicates.
    pub fn register(&mut self, tool: NativeTool) -> Result<()> {
        if tool.name.trim().is_empty() {
            return Err(anyhow!("native tool name must be non-empty"));
        }
        if RESERVED_TOOL_NAMES.contains(&tool.name.as_str()) {
            return Err(anyhow!(
                "native tool name {:?} is reserved by a built-in tool ({})",
                tool.name,
                RESERVED_TOOL_NAMES.join(", ")
            ));
        }
        if self.tools.contains_key(&tool.name) {
            return Err(anyhow!("native tool {:?} is already registered", tool.name));
        }
        self.tools.insert(tool.name.clone(), tool);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&NativeTool> {
        self.tools.get(name)
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Registered names in registry (sorted) order — the order dispatch arms
    /// are generated in and specs are advertised in.
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Provider-facing specs for every registered tool, in name order.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(NativeTool::spec).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> NativeTool {
        NativeTool::from_fn(
            name,
            "test tool",
            serde_json::json!({"type": "object"}),
            |_| async { Ok(Value::Null) },
        )
    }

    #[test]
    fn registry_rejects_reserved_duplicate_and_empty_names() {
        let mut registry = ToolRegistry::new();
        for reserved in RESERVED_TOOL_NAMES {
            let err = registry.register(tool(reserved)).unwrap_err().to_string();
            assert!(err.contains("reserved"), "{err}");
        }
        let err = registry.register(tool("")).unwrap_err().to_string();
        assert!(err.contains("non-empty"), "{err}");
        registry.register(tool("lookup")).unwrap();
        let err = registry.register(tool("lookup")).unwrap_err().to_string();
        assert!(err.contains("already registered"), "{err}");
    }

    #[tokio::test]
    async fn from_fn_handler_round_trips() -> Result<()> {
        let tool = NativeTool::from_fn(
            "echo",
            "echo the arguments",
            serde_json::json!({"type": "object"}),
            |arguments| async move { Ok(serde_json::json!({ "echoed": arguments })) },
        );
        let out = tool.handler.call(serde_json::json!({"x": 1})).await?;
        assert_eq!(out, serde_json::json!({"echoed": {"x": 1}}));
        Ok(())
    }

    #[test]
    fn specs_are_name_ordered() {
        let mut registry = ToolRegistry::new();
        registry.register(tool("zeta")).unwrap();
        registry.register(tool("alpha")).unwrap();
        assert_eq!(registry.names(), vec!["alpha", "zeta"]);
        let specs = registry.specs();
        assert_eq!(specs[0].function.name, "alpha");
        assert_eq!(specs[1].function.name, "zeta");
    }
}
