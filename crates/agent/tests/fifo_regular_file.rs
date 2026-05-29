use std::process::{Command, Stdio};
use uuid::Uuid;

#[test]
fn fifo_rejects_regular_file() {
    let root = std::env::temp_dir().join(format!("agent-fifo-regular-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let fifo_path = root.join("agent.fifo");
    std::fs::write(&fifo_path, b"stale input\0").unwrap();
    let replay_path = root.join("replay.jsonl");
    std::fs::write(&replay_path, "").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_agent"))
        .arg("--fifo")
        .arg(&fifo_path)
        .arg("--model")
        .arg("test-model")
        .arg("--replay-trace")
        .arg(&replay_path)
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env_remove("AGENT_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "agent unexpectedly succeeded\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("is not a named pipe"),
        "stderr did not explain regular file rejection: {stderr}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("AgentDone"),
        "regular file path emitted AgentDone events: {stdout}"
    );
}
