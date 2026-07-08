//! # agent-sdk
//!
//! Embed [agentd](https://github.com/bsima/agentd) agents in Rust — the
//! Rust core the Python/TS bindings wrap (t-1308). Three execution flows
//! share one [`Agent`] definition:
//!
//! 1. **One-shot, in-process** — [`Runner`] drives the same
//!    `agent_core::run_agent_loop` the `agent` CLI uses, inside your
//!    process. Native [`Tool`]s, injected providers, and live
//!    [`EventStream`]s all work here.
//! 2. **Persistent session, child-process** — [`Session`] spawns the
//!    `agent` binary (`agent --session --json`), delivers NUL-framed v1
//!    turn envelopes on its stdin, and correlates results strictly by
//!    `turn_id` (never by order). The child owns its trace and per-turn
//!    checkpoints on disk in the CLI/supervisor layout
//!    (docs/SUPERVISOR.md), so a crashed or killed session resumes with
//!    history intact via [`Session::resume`]. **Native SDK tools and
//!    injected providers do not cross the process boundary in this wave**
//!    — see the [`Session`] docs for the limitations and rationale.
//! 3. **Replay** — deterministic, credential-free re-execution of a
//!    recorded trace: [`Runner::replay`] in-process, or
//!    [`SessionOptions::replay_trace`] for sessions (build fixtures with
//!    [`testing::SessionReplayFixture`]).
//!
//! The pieces:
//!
//! - **[`Agent`]** — declarative definition: model (registry alias or
//!   custom provider), instructions, typed [`Tool`]s, an output schema
//!   ([`OutputContract`]), and the runtime's existing policy knobs (turn
//!   budget, eval env/timeout, memory directory).
//! - **[`Runner`]** — one-shot [`Runner::run`], live-observed
//!   [`Runner::start`] (public trace events via [`EventStream`]), and
//!   deterministic [`Runner::replay`] of a recorded trace.
//! - **[`Session`]** — persistent sessions over the agent binary:
//!   [`Session::send`] / [`Session::send_with`] (a caller-side timeout
//!   leaves the turn running; [`Session::attach`] retrieves it later),
//!   [`Session::events`], [`Session::status`], graceful [`Session::stop`],
//!   and crash-safe [`Session::resume`].
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
//! And the session flow (the same agent minus the native tool, which
//! cannot cross the process boundary yet):
//!
//! ```no_run
//! use agent_sdk::{Agent, Session, SessionOptions};
//!
//! # async fn example() -> Result<(), agent_sdk::SdkError> {
//! let agent = Agent::builder("claude-sonnet")
//!     .instructions("You are a weather assistant.")
//!     .build()?;
//! let options = SessionOptions {
//!     name: Some("weather".into()), // names ~/.local/share/agentd/weather
//!     ..SessionOptions::default()
//! };
//! let session = Session::start(&agent, options.clone()).await?;
//! let turn = session.send("What's the weather in SF?").await?;
//! println!("[{}] {}", turn.turn_id, turn.text);
//! session.stop().await?;
//! // Later — even after a crash — pick the session back up:
//! let session = Session::resume(&agent, options).await?;
//! # let _ = session;
//! # Ok(())
//! # }
//! ```
//!
//! Provider configuration reuses the CLI's conventions: the model registry
//! at `~/.config/agent/models.yaml` plus the
//! `AGENT_PROVIDER`/`AGENT_API_KEY`/`ANTHROPIC_API_KEY`/
//! `OPENROUTER_API_KEY` environment fallbacks. No new config files. For
//! credential-free runs (tests, the crate's `examples/`), inject a
//! [`testing::ScriptedProvider`] (one-shot) or drive a session from a
//! [`testing::SessionReplayFixture`].

mod agent;
mod error;
mod runner;
mod session;
pub mod testing;

pub use agent::{Agent, AgentBuilder, Tool, ToolDef, DEFAULT_MAX_TURNS};
pub use error::SdkError;
pub use runner::{EventStream, RunHandle, RunResult, Runner};
pub use session::{
    PendingApproval, Session, SessionOptions, SessionStatus, TurnOptions, TurnResult,
};

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

// The hydration-provider authoring surface (docs/PROVIDERS.md): implement
// [`HydrationSource`] and register it with
// [`AgentBuilder::hydration_source`]. See
// `examples/custom_source.rs` for a complete out-of-tree provider. The
// write-side trait (`HydrationSink`) is intentionally not re-exported —
// custom sinks are unreachable from the built-in loop; see the builder
// docs.
pub use agent_core::hydration::{
    HydrationSource, SourceCapability, SourceKind, SourceParams, SourceResult,
};

// The approval/pause protocol (t-1308.10, DR-7): the hook types for
// AgentBuilder::on_approval and the on-disk record shape surfaced by
// Session::next_approval.
pub use agent_core::approval::{
    ApprovalDecision, ApprovalKind, ApprovalRequest, PendingEffectRecord, PendingStatus,
};
