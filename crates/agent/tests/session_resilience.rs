use std::io::Write;
use std::process::{Command, Stdio};
use uuid::Uuid;

// A long-running session must survive a failed turn (provider outage, replay
// divergence, context overflow) instead of crashing. Drive a stdin session
// with a replay trace that has no recorded results: every turn errors, but
// the session should keep reading frames and exit cleanly at EOF.
#[test]
fn stdin_session_survives_failed_turns() {
    let root = std::env::temp_dir().join(format!("agent-session-resilience-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let replay_path = root.join("empty-replay.jsonl");
    std::fs::write(&replay_path, "").unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_agent"))
        .arg("--session")
        .arg("--model")
        .arg("test-model")
        .arg("--replay-trace")
        .arg(&replay_path)
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Two turns, both of which will fail (no recorded InferResult), then EOF.
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(b"first turn\0second turn\0").unwrap();
    drop(stdin);

    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "session must survive failed turns and exit cleanly at EOF\nstderr:\n{stderr}"
    );
    assert_eq!(
        stderr.matches("turn failed:").count(),
        2,
        "both failed turns should be reported on stderr\nstderr:\n{stderr}"
    );
}
