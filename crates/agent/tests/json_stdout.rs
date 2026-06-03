use std::process::{Command, Stdio};
use uuid::Uuid;

// Verifies the t-1031 contract: in --json/--debug mode, stdout is pure JSONL
// (machine events only) and the human-readable response is NOT emitted bare.
//
// The agent always runs passive hydration (TemporalHistory + SessionContext)
// before the first inference, which consumes op_id 1 (the HydrationStart op).
// The real Infer call is therefore op_id 2, so the replay fixture records its
// InferResult at op_id 2 to match the deterministic op numbering.
#[test]
fn json_mode_stdout_is_parseable_jsonl() {
    let root = std::env::temp_dir().join(format!("agent-json-stdout-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let replay_path = root.join("replay.jsonl");
    let timestamp = "2026-05-29T00:00:00Z";
    let replay = format!(
        r#"{{"event":"Custom","run_id":"replay","op_id":0,"name":"ir_effect","data":{{"program_hash":"sha256:5c71cc50336006d39c93cda3a28e2e5806f39985adfa1388ac267c0d0f7872f4","effect_id":"sha256:2d2746e54d1d7a94aa1580f358faea32b94d6c7e3af6156aea1db4dcb561b47d","kind":"Infer","site":{{"block":0,"instruction_index":0}},"dynamic_path":[{{"Visit":{{"site":{{"block":0,"instruction_index":0}},"visit":0}}}}]}},"timestamp":"{timestamp}"}}
{{"event":"InferCall","run_id":"replay","op_id":2,"model":"test-model","prompt_preview":"","timestamp":"{timestamp}"}}
{{"event":"InferResult","run_id":"replay","op_id":2,"response":{{"content":"hello human","tool_calls":[],"input_tokens":3,"output_tokens":4,"total_tokens":7}},"response_preview":"hello human","input_tokens":3,"output_tokens":4,"total_tokens":7,"duration_ms":1,"timestamp":"{timestamp}"}}
"#
    );
    std::fs::write(&replay_path, replay).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_agent"))
        .arg("--json")
        .arg("--model")
        .arg("test-model")
        .arg("--replay-trace")
        .arg(&replay_path)
        .arg("hello")
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "agent failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.trim().is_empty(), "expected JSONL on stdout");
    for (index, line) in stdout.lines().enumerate() {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|err| panic!("stdout line {} is not JSON: {err}: {line}", index + 1));
    }
    assert!(
        !stdout.lines().any(|line| line == "hello human"),
        "human response was emitted bare on stdout: {stdout}"
    );
}
