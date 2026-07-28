#![cfg(feature = "acp")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use uuid::Uuid;

// The `agent --acp` mode speaks agentclientprotocol.com JSON-RPC over stdio:
// newline-delimited frames, stdout reserved for the protocol. These tests
// drive the binary the way an ACP client (e.g. Paseo) does.

fn spawn_acp_agent(root: &std::path::Path, extra_args: &[&str]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_agent"))
        .arg("--acp")
        .args(extra_args)
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn agent --acp")
}

fn send_line(child: &mut Child, line: &str) {
    let stdin = child.stdin.as_mut().expect("child stdin");
    stdin.write_all(line.as_bytes()).expect("write frame");
    stdin.write_all(b"\n").expect("write newline");
    stdin.flush().expect("flush");
}

/// Wait for the child to exit on its own, or kill it and fail: an ACP agent
/// must shut down when the client closes its stdin.
fn expect_exit(child: &mut Child) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().expect("try_wait") {
            assert!(status.success(), "agent --acp exited nonzero: {status}");
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    child.kill().ok();
    panic!("agent --acp did not exit after stdin EOF");
}

/// Replay fixture for a single text-answer turn. The agent always runs
/// passive hydration before the first inference (op 1), so the entry Infer
/// records at op 2 — same op numbering contract as tests/json_stdout.rs.
fn write_replay_fixture(root: &std::path::Path, content: &str) -> std::path::PathBuf {
    use agent_core::{
        agent_loop_ir_for_options, effect_location, program_hash, AgentLoopOptions, BlockId,
        DynamicPath, EffectKind, EffectSite, Model,
    };
    // ACP mode gates shell Evals behind approval by default, and the gate is
    // part of the loop program — the fixture's effect ids must be computed
    // under the same program identity.
    let options = AgentLoopOptions {
        memory_tools: false,
        tool_names: vec![],
        output_contract: None,
        shell_requires_approval: true,
        infer_system_prompt: None,
    };
    let machine = agent_loop_ir_for_options(Model("test-model".into()), vec![], 16, &options);
    let hash = program_hash(&machine.program).unwrap();
    let site = EffectSite {
        block: BlockId(0),
        instruction_index: 0,
    };
    let location =
        effect_location(hash, EffectKind::Infer, site, DynamicPath::at_entry(0)).unwrap();
    let ir_effect = serde_json::to_string(&location).unwrap();
    let timestamp = "2026-05-29T00:00:00Z";
    let replay = format!(
        r#"{{"event":"InferCall","run_id":"replay","op_id":2,"model":"test-model","prompt_preview":"","effect":{ir_effect},"timestamp":"{timestamp}"}}
{{"event":"InferResult","run_id":"replay","op_id":2,"response":{{"content":{content},"tool_calls":[],"finish_reason":"stop","input_tokens":3,"output_tokens":4,"total_tokens":7}},"response_preview":"","input_tokens":3,"output_tokens":4,"total_tokens":7,"duration_ms":1,"timestamp":"{timestamp}"}}
"#,
        content = serde_json::to_string(content).unwrap(),
    );
    let path = root.join("replay.jsonl");
    std::fs::write(&path, replay).unwrap();
    path
}

/// Read JSON-RPC frames until the response with `id` arrives; returns the
/// response and every `session/update` notification seen on the way.
fn read_until_response(
    reader: &mut impl BufRead,
    id: u64,
) -> (serde_json::Value, Vec<serde_json::Value>) {
    let mut updates = Vec::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).expect("read frame");
        assert!(n > 0, "stream closed before response {id} arrived");
        let frame: serde_json::Value =
            serde_json::from_str(&line).unwrap_or_else(|err| panic!("bad frame: {err}: {line}"));
        if frame["id"] == id {
            return (frame, updates);
        }
        if frame["method"] == "session/update" {
            updates.push(frame["params"].clone());
        }
    }
}

/// stdout belongs to JSON-RPC in ACP mode: mixing in the --debug machine
/// event JSONL would corrupt the frame stream, so the flags conflict.
#[test]
fn acp_conflicts_with_debug() {
    let output = Command::new(env!("CARGO_BIN_EXE_agent"))
        .args(["--acp", "--debug"])
        .stdin(Stdio::null())
        .output()
        .expect("run agent");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cannot be used with"),
        "clap rejects the combination: {stderr}"
    );
}

#[test]
fn initialize_handshake_reports_capabilities() {
    let root = std::env::temp_dir().join(format!("agent-acp-init-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();

    let mut child = spawn_acp_agent(&root, &[]);
    send_line(
        &mut child,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
    );

    let stdout = child.stdout.take().expect("child stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line).expect("read response line");
    let response: serde_json::Value = serde_json::from_str(&line)
        .unwrap_or_else(|err| panic!("bad JSON-RPC frame: {err}: {line}"));

    assert_eq!(
        response["id"], 1,
        "response correlates to the request: {response}"
    );
    let result = &response["result"];
    assert_eq!(
        result["protocolVersion"], 1,
        "mirrors the client version: {response}"
    );
    assert_eq!(
        result["agentCapabilities"]["loadSession"], true,
        "advertises session/load: {response}"
    );
    assert_eq!(
        result["agentInfo"]["name"], "agentd",
        "names the agent: {response}"
    );

    drop(child.stdin.take());
    expect_exit(&mut child);
}

#[test]
fn prompt_turn_streams_message_and_ends_turn() {
    let root = std::env::temp_dir().join(format!("agent-acp-prompt-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let replay = write_replay_fixture(&root, "hello human");

    let mut child = spawn_acp_agent(
        &root,
        &[
            "--model",
            "test-model",
            "--replay-trace",
            replay.to_str().unwrap(),
        ],
    );
    let stdout = child.stdout.take().expect("child stdout");
    let mut reader = BufReader::new(stdout);

    send_line(
        &mut child,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
    );
    let (_, _) = read_until_response(&mut reader, 1);

    let cwd = root.to_str().unwrap();
    send_line(
        &mut child,
        &format!(
            r#"{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{"cwd":"{cwd}","mcpServers":[]}}}}"#
        ),
    );
    let (new_session, _) = read_until_response(&mut reader, 2);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("no sessionId in {new_session}"))
        .to_owned();

    send_line(
        &mut child,
        &format!(
            r#"{{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{{"sessionId":"{session_id}","prompt":[{{"type":"text","text":"hello"}}]}}}}"#
        ),
    );
    let (prompt_response, updates) = read_until_response(&mut reader, 3);
    assert_eq!(
        prompt_response["result"]["stopReason"], "end_turn",
        "turn completes: {prompt_response}"
    );
    let chunks: Vec<&serde_json::Value> = updates
        .iter()
        .filter(|update| update["update"]["sessionUpdate"] == "agent_message_chunk")
        .collect();
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["update"]["content"]["text"] == "hello human"),
        "the replayed answer arrives as an agent_message_chunk before the \
         prompt response; updates: {updates:?}"
    );
    assert!(
        updates
            .iter()
            .all(|update| update["sessionId"] == session_id.as_str()),
        "updates carry the session id: {updates:?}"
    );

    drop(child.stdin.take());
    expect_exit(&mut child);
}
