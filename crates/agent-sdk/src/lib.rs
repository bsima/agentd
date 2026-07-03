//! # agent-sdk
//!
//! Embed [agentd](https://github.com/bsima/agentd) agents in Rust — the
//! Rust core the Python/TS bindings wrap (t-1308). The SDK drives the same
//! `agent_core::run_agent_loop` the `agent` CLI uses, in-process:
//!
//! - **[`Agent`]** — declarative definition: model (registry alias or
//!   custom provider), instructions, typed [`Tool`]s, an output schema
//!   ([`OutputContract`]), and the runtime's existing policy knobs (turn
//!   budget, eval env/timeout, memory directory).
//! - **[`Runner`]** — one-shot [`Runner::run`], live-observed
//!   [`Runner::start`] (public trace events via [`EventStream`]), and
//!   deterministic [`Runner::replay`] of a recorded trace.
//! - **Typed tools** — native async handlers dispatched through the agent
//!   loop's tool-dispatch arm as first-class effects: advertised to the
//!   model like the built-ins, traced as
//!   `tool.requested`/`tool.completed`/`tool.failed` with stable effect
//!   identity, replayed by effect id without invoking the handler, and
//!   never executed via a shell.
//! - **[`SdkError`]** — the single typed error surface.
//!
//! ```no_run
//! use agent_sdk::{Agent, Runner, Tool};
//!
//! # async fn example() -> Result<(), agent_sdk::SdkError> {
//! let weather = Tool::new(
//!     "get_weather",
//!     "Current weather for a city.",
//!     serde_json::json!({
//!         "type": "object",
//!         "properties": { "city": { "type": "string" } },
//!         "required": ["city"]
//!     }),
//!     |args| async move { Ok(serde_json::json!({ "forecast": "sunny", "args": args })) },
//! );
//! let agent = Agent::builder("claude-sonnet") // a models.yaml alias
//!     .instructions("You are a weather assistant.")
//!     .tool(weather)
//!     .build()?;
//! let result = Runner::run(&agent, "What's the weather in SF?").await?;
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```
//!
//! Provider configuration reuses the CLI's conventions: the model registry
//! at `~/.config/agent/models.yaml` plus the
//! `AGENT_PROVIDER`/`AGENT_API_KEY`/`ANTHROPIC_API_KEY`/
//! `OPENROUTER_API_KEY` environment fallbacks. No new config files. For
//! credential-free runs (tests, the crate's `examples/`), inject a
//! [`testing::ScriptedProvider`].

mod agent;
mod error;
mod runner;
pub mod testing;

pub use agent::{Agent, AgentBuilder, Tool, ToolDef, DEFAULT_MAX_TURNS};
pub use error::SdkError;
pub use runner::{EventStream, RunHandle, RunResult, Runner};

// The SDK's event vocabulary is agent-core's public trace schema
// (docs/TRACE_SCHEMA.md), re-exported so consumers need only this crate.
pub use agent_core::public_trace::{
    PublicDynamicPath, PublicEffect, PublicEffectSite, PublicEvent, PublicStatus,
    PUBLIC_SCHEMA_VERSION,
};

// Runtime types that appear on the SDK surface: output contracts, eval env
// policy, the provider trait (for custom providers), and the tool handler
// trait (for stateful tool implementations).
pub use agent_core::tool::{ToolHandler, RESERVED_TOOL_NAMES};
pub use agent_core::{ChatProvider, EnvPolicy, OutputContract};
