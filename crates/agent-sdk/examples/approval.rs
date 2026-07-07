//! The approval/pause protocol (t-1308.10, DR-7), in-process arm — runnable
//! credential-free: a scripted provider stands in for the model, so nothing
//! here needs an API key.
//!
//! Gating the shell tool (`require_shell_approval`) means every shell
//! command stops at the approval gate before executing. In-process runs
//! decide with the `on_approval` hook; a gated run with NO hook fails
//! closed — the command does not execute, ever, and the run errors with
//! `SdkError::ApprovalRequired`. (The durable arm — pause to disk, resolve
//! with `agent approvals --approve/--deny`, resume across restarts — lives
//! on the `agent` CLI and `Session::next_approval`.)
//!
//! Run with: `cargo run -p agent-sdk --example approval`

use agent_sdk::testing::ScriptedProvider;
use agent_sdk::{Agent, AgentBuilder, ApprovalDecision, Runner, SdkError};
use serde_json::json;
use std::sync::Arc;

/// A gated agent whose scripted "model" asks to run one shell command and
/// then answers with `final_text`.
fn gated_agent(command: &str, final_text: &str) -> AgentBuilder {
    let provider = ScriptedProvider::new()
        .tool_call("shell", json!({ "command": command }))
        .text(final_text);
    Agent::builder("mock-model")
        .provider(Arc::new(provider))
        .require_shell_approval()
}

#[tokio::main]
async fn main() -> Result<(), SdkError> {
    // 1. Approve: the hook is called at the effect site with the pending
    //    request; approving executes the command exactly once.
    let agent = gated_agent("echo hello-from-an-approved-command", "The command ran.")
        .on_approval(|request| {
            println!(
                "[hook] approval requested ({}): {}",
                request.kind.as_str(),
                request.request
            );
            ApprovalDecision::Approve
        })
        .build()?;
    let result = Runner::run(&agent, "run the greeting command").await?;
    println!("approved run finished: {}\n", result.text);

    // 2. Deny: denial is a VALUE, not an abort — the model reads the typed
    //    denial as the tool result and the run still completes.
    let agent = gated_agent("rm -rf /important", "Understood, I will not run that.")
        .on_approval(|request| {
            println!(
                "[hook] denying ({}): {}",
                request.kind.as_str(),
                request.request
            );
            ApprovalDecision::Deny
        })
        .build()?;
    let result = Runner::run(&agent, "clean everything up").await?;
    println!("denied run finished: {}\n", result.text);

    // 3. Unattended: gated + no hook = fail closed. No auto-approval, no
    //    timeout-approval; the command never executes.
    let agent = gated_agent("echo never-runs", "unreachable").build()?;
    match Runner::run(&agent, "run it").await {
        Err(SdkError::ApprovalRequired {
            pending_id, kind, ..
        }) => {
            println!("unattended gated run failed closed: pending {pending_id} ({kind})");
        }
        Err(other) => return Err(other),
        Ok(_) => unreachable!("a gated run with no hook must not complete"),
    }
    Ok(())
}
