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
}
