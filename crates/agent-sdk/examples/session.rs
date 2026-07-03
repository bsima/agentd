//! Persistent session over the `agent` binary, driven credential-free by a
//! replay fixture (no provider, no API key).
//!
//! Build the binary first, then run the example:
//!
//! ```sh
//! cargo build -p agent
//! cargo run -p agent-sdk --example session
//! ```
//!
//! Everything (session home, trace, fixture) lives under a scratch
//! directory that is printed at the end so you can inspect the artifacts.

use agent_sdk::testing::SessionReplayFixture;
use agent_sdk::{Agent, Session, SessionOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let root = std::env::temp_dir().join(format!(
        "agent-sdk-session-example-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&root)?;

    // Two scripted turns, recorded at the loop's stable effect locations —
    // the child replays them instead of calling a provider.
    let fixture = root.join("replay.jsonl");
    SessionReplayFixture::new("example-model")
        .turn("Hi! I looked around: this project builds with cargo.")
        .turn("Yes — and `cargo test --workspace` is green.")
        .write(&fixture)
        .await?;

    let agent = Agent::builder("example-model")
        .instructions("You are a build assistant.")
        .build()?;
    let session = Session::start(
        &agent,
        SessionOptions {
            name: Some("example".into()),
            home: Some(root.join("session")),
            replay_trace: Some(fixture),
            // Hermetic: the child's traces/config lookups stay in scratch.
            env: vec![("HOME".into(), root.display().to_string())],
            ..SessionOptions::default()
        },
    )
    .await?;

    let first = session.send("What kind of project is this?").await?;
    println!("[{}] {}", first.turn_id, first.text);
    let second = session.send("Do the tests pass?").await?;
    println!("[{}] {}", second.turn_id, second.text);

    let status = session.status().await;
    println!(
        "session alive={} pid={:?} run_id={} trace={}",
        status.alive,
        status.pid,
        status.run_id,
        status.trace_path.display()
    );

    let exit = session.stop().await?;
    println!("stopped: {exit}");
    println!("artifacts under: {}", root.display());
    Ok(())
}
