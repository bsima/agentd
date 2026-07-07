//! Agent definition: what to run (model, instructions, tools, output
//! contract) and under which policies (turn budget, eval env/timeout,
//! memory). Built once, reusable across [`crate::Runner`] calls.

use crate::error::SdkError;
use agent_core::approval::{ApprovalDecision, ApprovalHookFn, ApprovalRequest};
use agent_core::tool::{NativeTool, ToolHandler, ToolRegistry};
use agent_core::{ChatProvider, EnvPolicy, OutputContract};
use serde_json::Value;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// A declarative native tool description: everything except the handler.
/// Pair it with a handler via [`Tool::from_def`] — useful when tool schemas
/// are data (loaded from config, generated) rather than code.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments.
    pub parameters: Value,
}

/// A typed native tool: name, description, JSON-Schema parameters, and an
/// in-process async handler. Registered tools are advertised to the model
/// alongside the built-ins and dispatched through the agent loop's
/// tool-dispatch arm as first-class effects — traced
/// (`tool.requested`/`tool.completed`/`tool.failed`), replayable by effect
/// id, and never executed via a shell.
#[derive(Clone, Debug)]
pub struct Tool {
    pub(crate) inner: NativeTool,
}

impl Tool {
    /// Register an async closure: `Fn(Value) -> Future<Output =
    /// anyhow::Result<Value>>`. A handler error becomes a tool result the
    /// model can read and recover from (errors-as-values), surfaced in the
    /// trace as `tool.failed`.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<Value>> + Send + 'static,
    {
        Self {
            inner: NativeTool::from_fn(name, description, parameters, handler),
        }
    }

    /// Register a trait-object handler (for handlers with state or
    /// non-closure implementations).
    pub fn from_handler(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        handler: Arc<dyn ToolHandler>,
    ) -> Self {
        Self {
            inner: NativeTool::new(name, description, parameters, handler),
        }
    }

    /// Pair a declarative [`ToolDef`] with a handler.
    pub fn from_def(def: ToolDef, handler: Arc<dyn ToolHandler>) -> Self {
        Self::from_handler(def.name, def.description, def.parameters, handler)
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }
}

/// An agent definition. Create via [`Agent::builder`]; run via
/// [`crate::Runner`]. Cloning is cheap (tool handlers are shared).
#[derive(Clone)]
pub struct Agent {
    pub(crate) name: Option<String>,
    pub(crate) model: String,
    pub(crate) instructions: Option<String>,
    pub(crate) tools: ToolRegistry,
    pub(crate) output_contract: Option<OutputContract>,
    pub(crate) max_turns: usize,
    pub(crate) eval_timeout: Duration,
    pub(crate) eval_env: EnvPolicy,
    pub(crate) eval_cwd: Option<PathBuf>,
    pub(crate) memory_dir: Option<PathBuf>,
    pub(crate) trace_dir: Option<PathBuf>,
    pub(crate) provider: Option<Arc<dyn ChatProvider>>,
    pub(crate) require_shell_approval: bool,
    pub(crate) on_approval: Option<ApprovalHookFn>,
}

impl Agent {
    /// Start building an agent for `model` — a model registry alias
    /// (`~/.config/agent/models.yaml`, the same registry the `agent` CLI
    /// uses), or any model string when a custom provider is injected with
    /// [`AgentBuilder::provider`].
    pub fn builder(model: impl Into<String>) -> AgentBuilder {
        AgentBuilder {
            name: None,
            model: model.into(),
            instructions: None,
            tools: Vec::new(),
            output_contract: None,
            max_turns: DEFAULT_MAX_TURNS,
            eval_timeout: Duration::from_secs(120),
            eval_env: EnvPolicy::Inherit,
            eval_cwd: None,
            memory_dir: None,
            trace_dir: None,
            provider: None,
            require_shell_approval: false,
            on_approval: None,
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Registered native tool names, in advertisement/dispatch order.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.names()
    }
}

/// Turn-budget safety ceiling, matching the `agent` CLI default.
pub const DEFAULT_MAX_TURNS: usize = 100;

/// Builder for [`Agent`]. All knobs map onto existing runtime policies —
/// the SDK adds no configuration of its own.
pub struct AgentBuilder {
    name: Option<String>,
    model: String,
    instructions: Option<String>,
    tools: Vec<Tool>,
    output_contract: Option<OutputContract>,
    max_turns: usize,
    eval_timeout: Duration,
    eval_env: EnvPolicy,
    eval_cwd: Option<PathBuf>,
    memory_dir: Option<PathBuf>,
    trace_dir: Option<PathBuf>,
    provider: Option<Arc<dyn ChatProvider>>,
    require_shell_approval: bool,
    on_approval: Option<ApprovalHookFn>,
}

impl AgentBuilder {
    /// Display name (also exported as `AGENT_NAME`-style metadata later; not
    /// interpreted by the runtime).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// System instructions. When set, the run's history starts with a
    /// system message carrying exactly this text.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Register a native tool. Order of registration is not significant:
    /// tools dispatch by exact name.
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Require the final response to validate against this JSON Schema,
    /// with the default bounded repair budget (see
    /// [`agent_core::OutputContract`]). The validated value is returned on
    /// [`crate::RunResult::output`]; exhausted repairs become
    /// [`SdkError::OutputContract`].
    pub fn output_schema(self, schema: Value) -> Self {
        self.output_contract(OutputContract::new(schema))
    }

    /// Like [`AgentBuilder::output_schema`], with explicit contract knobs
    /// (`max_repairs`).
    pub fn output_contract(mut self, contract: OutputContract) -> Self {
        self.output_contract = Some(contract);
        self
    }

    /// Turn-budget safety ceiling (default 100, the CLI's default).
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Timeout for each shell-tool execution (default 120s).
    pub fn eval_timeout(mut self, timeout: Duration) -> Self {
        self.eval_timeout = timeout;
        self
    }

    /// Environment policy for shell-tool executions (default
    /// [`EnvPolicy::Inherit`], which strips credential-shaped variables).
    pub fn eval_env(mut self, env: EnvPolicy) -> Self {
        self.eval_env = env;
        self
    }

    /// Working directory for shell-tool executions.
    pub fn eval_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.eval_cwd = Some(cwd.into());
        self
    }

    /// Enable persistent memory backed by this directory: the runtime
    /// registers the memory backend and the model-initiated
    /// `remember`/`recall` tools ride with it.
    pub fn memory_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.memory_dir = Some(dir.into());
        self
    }

    /// Where run traces are written (default: `~/.local/share/agent/traces`,
    /// the CLI's convention).
    pub fn trace_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.trace_dir = Some(dir.into());
        self
    }

    /// Gate the built-in shell tool behind human approval (t-1308.10,
    /// DR-7): each shell command pauses at the approval gate until a
    /// decision arrives. In-process runs decide via
    /// [`AgentBuilder::on_approval`]; a gated run with **no** hook FAILS
    /// CLOSED — the command does not execute and the run errors with
    /// [`SdkError::ApprovalRequired`]. There is no auto-approval and no
    /// timeout-approval.
    pub fn require_shell_approval(mut self) -> Self {
        self.require_shell_approval = true;
        self
    }

    /// Synchronous approval hook for gated effects (the in-process arm of
    /// the DR-7 protocol). Called at the effect site with the pending
    /// request (pending id, effect identity, request preview); returns
    /// [`ApprovalDecision::Approve`] to execute the effect or
    /// [`ApprovalDecision::Deny`] to bind a typed denial value the model
    /// can react to (the run continues either way). Both outcomes are
    /// traced as `approval.requested`/`approval.resolved` and reproduced as
    /// data on replay.
    pub fn on_approval<F>(mut self, hook: F) -> Self
    where
        F: Fn(&ApprovalRequest) -> ApprovalDecision + Send + Sync + 'static,
    {
        self.on_approval = Some(Arc::new(hook));
        self
    }

    /// Inject a custom [`ChatProvider`], bypassing model-registry
    /// resolution — the model string is passed to the provider verbatim.
    /// This is the seam for scripted/mock providers
    /// ([`crate::testing::ScriptedProvider`]) and custom transports.
    pub fn provider(mut self, provider: Arc<dyn ChatProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Validate and build the [`Agent`]. Fails on an empty model, a zero
    /// turn budget, or invalid tool registrations (duplicate names, or
    /// names reserved by the built-in shell/infer/remember/recall tools).
    pub fn build(self) -> Result<Agent, SdkError> {
        if self.model.trim().is_empty() {
            return Err(SdkError::Config("model must be non-empty".into()));
        }
        if self.max_turns == 0 {
            return Err(SdkError::Config("max_turns must be at least 1".into()));
        }
        let mut registry = ToolRegistry::new();
        for tool in self.tools {
            registry
                .register(tool.inner)
                .map_err(|err| SdkError::Config(format!("{err:#}")))?;
        }
        Ok(Agent {
            name: self.name,
            model: self.model,
            instructions: self.instructions,
            tools: registry,
            output_contract: self.output_contract,
            max_turns: self.max_turns,
            eval_timeout: self.eval_timeout,
            eval_env: self.eval_env,
            eval_cwd: self.eval_cwd,
            memory_dir: self.memory_dir,
            trace_dir: self.trace_dir,
            provider: self.provider,
            require_shell_approval: self.require_shell_approval,
            on_approval: self.on_approval,
        })
    }
}
