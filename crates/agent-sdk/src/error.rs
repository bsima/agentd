//! The SDK's single typed error surface.
//!
//! agent-core uses `anyhow` internally (and at its own edges); the SDK's
//! public API is typed so language bindings (t-1308.8) can map each variant
//! to a native exception class. Every variant carries owned data only.

/// Everything a [`crate::Runner`] call can fail with.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SdkError {
    /// The [`crate::Agent`] configuration is invalid (bad tool registration,
    /// empty model, zero turn budget, ...). Raised by
    /// [`crate::AgentBuilder::build`].
    #[error("invalid agent configuration: {0}")]
    Config(String),

    /// Model or provider resolution failed: unknown registry alias, missing
    /// API key, unsupported provider tag, unreadable `models.yaml`.
    #[error("model resolution failed: {0}")]
    Model(String),

    /// The run's trace file could not be created or written.
    #[error("trace I/O failed: {0}")]
    Trace(String),

    /// A replay trace could not be loaded, or replay diverged from the
    /// recording (edited program, changed tool set or output schema,
    /// different effect results).
    #[error("replay failed: {0}")]
    Replay(String),

    /// The agent completed but its final output failed the output schema
    /// even after the bounded repair turns. `content` is the last
    /// (non-conforming) response text.
    #[error("output contract violated after {attempts} attempt(s): {}", errors.join("; "))]
    OutputContract {
        attempts: usize,
        errors: Vec<String>,
        content: String,
    },

    /// The agent loop itself failed (provider error after retries, effect
    /// failure at an aborting site, task panic).
    #[error("agent run failed: {0}")]
    Run(String),

    /// The requested feature cannot be expressed in this execution mode.
    /// Today this is the [`crate::Session`] facade rejecting agent config
    /// that cannot cross the child-process boundary (native tools, injected
    /// providers) until the supervisor/tool-host work lands (wave 3+).
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Session plumbing failed: the `agent` binary could not be spawned,
    /// the child exited unexpectedly, or the machine-event protocol broke.
    #[error("session error: {0}")]
    Session(String),

    /// A [`crate::Session::send_with`] caller-side timeout elapsed. The turn
    /// itself is NOT cancelled: it keeps running in the child (DR-10
    /// semantics), and its result stays retrievable via
    /// [`crate::Session::attach`] with the same `turn_id`.
    #[error("send timed out waiting for turn {turn_id} (turn still running: {still_running}); attach with the turn id to retrieve the result")]
    SendTimeout {
        turn_id: String,
        /// Whether the child process was still alive when the timeout fired.
        still_running: bool,
    },

    /// A session turn failed in the child (`agent_error` machine event).
    /// The session survives a failed turn; later sends still work.
    #[error("turn {turn_id} failed: {message}")]
    Turn { turn_id: String, message: String },
}
