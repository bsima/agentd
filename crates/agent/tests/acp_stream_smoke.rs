#![cfg(feature = "acp")]

//! Live-ish ACP tests against a mock OpenAI-compatible SSE provider:
//! streaming deltas, and the shell-approval `session/request_permission`
//! round-trip (allow and reject paths).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use uuid::Uuid;

const TEXT_TURN: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}\n\n\
data: [DONE]\n\n";

const SHELL_TURN: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"shell\",\"arguments\":\"{\\\"command\\\":\\\"echo acp-approved\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: [DONE]\n\n";

const DONE_TURN: &str =
    "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";

/// Marker body: read the request, then stall without answering (the turn
/// hangs until the client cancels and the agent drops the connection).
const STALL: &str = "<stall>";

/// Mock provider serving one canned SSE body per request, in order.
fn spawn_mock_sse_server(bodies: Vec<&'static str>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock provider");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for body in bodies {
            let Ok((mut socket, _)) = listener.accept() else {
                return;
            };
            let mut reader = BufReader::new(socket.try_clone().expect("clone socket"));
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                let line = line.trim_end();
                if let Some(value) = line
                    .to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(str::trim)
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    content_length = value;
                }
                if line.is_empty() {
                    break;
                }
            }
            let mut request_body = vec![0u8; content_length];
            reader
                .read_exact(&mut request_body)
                .expect("read request body");
            let request_body = String::from_utf8_lossy(&request_body);
            assert!(
                request_body.contains("\"stream\":true"),
                "provider request must ask for streaming: {request_body}"
            );
            if body == STALL {
                // Hold the socket open until the peer goes away.
                let mut probe = [0u8; 1];
                let _ = reader.read(&mut probe);
                continue;
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
            );
            socket.write_all(response.as_bytes()).expect("write SSE");
            socket.flush().ok();
        }
    });
    port
}

fn spawn_acp_agent(root: &std::path::Path, port: u16) -> Child {
    Command::new(env!("CARGO_BIN_EXE_agent"))
        .arg("--acp")
        .args(["--model", "test-model"])
        .args(["--provider", &format!("http://127.0.0.1:{port}/v1")])
        .args(["--key", "test-key"])
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn agent --acp")
}

struct AcpClient {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
}

impl AcpClient {
    fn start(root: &std::path::Path, port: u16) -> Self {
        let mut child = spawn_acp_agent(root, port);
        let stdout = child.stdout.take().expect("child stdout");
        Self {
            child,
            reader: BufReader::new(stdout),
        }
    }

    fn send(&mut self, frame: &str) {
        let stdin = self.child.stdin.as_mut().expect("child stdin");
        writeln!(stdin, "{frame}").expect("write frame");
        stdin.flush().expect("flush");
    }

    /// Read frames until the response to client-request `id` arrives.
    /// Agent-originated `session/request_permission` requests are answered
    /// with `permission_option` (panics if one arrives with no option set).
    /// Returns the response, collected updates, and the permission requests
    /// seen on the way.
    fn read_until_response(
        &mut self,
        id: u64,
        permission_option: Option<&str>,
    ) -> (
        serde_json::Value,
        Vec<serde_json::Value>,
        Vec<serde_json::Value>,
    ) {
        let mut updates = Vec::new();
        let mut permissions = Vec::new();
        loop {
            let mut line = String::new();
            let n = self.reader.read_line(&mut line).expect("read frame");
            assert!(n > 0, "stream closed before response {id} arrived");
            let frame: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|err| panic!("bad frame: {err}: {line}"));
            let method = frame.get("method").and_then(|m| m.as_str());
            match method {
                None if frame["id"] == id => return (frame, updates, permissions),
                None => {} // response to some other request; ignore
                Some("session/update") => updates.push(frame["params"].clone()),
                Some("session/request_permission") => {
                    let request_id = frame["id"].clone();
                    let option = permission_option
                        .unwrap_or_else(|| panic!("unexpected permission request: {frame}"));
                    permissions.push(frame["params"].clone());
                    self.send(&format!(
                        r#"{{"jsonrpc":"2.0","id":{request_id},"result":{{"outcome":{{"outcome":"selected","optionId":"{option}"}}}}}}"#
                    ));
                }
                Some(_) => {}
            }
        }
    }

    fn handshake_and_new_session(&mut self, cwd: &str) -> String {
        self.send(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
        );
        self.read_until_response(1, None);
        self.send(&format!(
            r#"{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{"cwd":"{cwd}","mcpServers":[]}}}}"#
        ));
        let (new_session, _, _) = self.read_until_response(2, None);
        new_session["result"]["sessionId"]
            .as_str()
            .unwrap_or_else(|| panic!("no sessionId in {new_session}"))
            .to_owned()
    }

    fn prompt(&mut self, id: u64, session_id: &str, text: &str) {
        self.send(&format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"session/prompt","params":{{"sessionId":"{session_id}","prompt":[{{"type":"text","text":"{text}"}}]}}}}"#
        ));
    }
}

impl Drop for AcpClient {
    fn drop(&mut self) {
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

fn temp_root(tag: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!("agent-acp-{tag}-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn chunk_texts(updates: &[serde_json::Value]) -> Vec<&str> {
    updates
        .iter()
        .filter(|update| update["update"]["sessionUpdate"] == "agent_message_chunk")
        .filter_map(|update| update["update"]["content"]["text"].as_str())
        .collect()
}

#[test]
fn prompt_turn_streams_deltas_before_the_response() {
    let root = temp_root("stream");
    let port = spawn_mock_sse_server(vec![TEXT_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "greet");
    let (response, updates, _) = client.read_until_response(3, None);

    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "turn completes: {response}"
    );
    let chunks = chunk_texts(&updates);
    assert!(
        chunks.len() >= 2,
        "expected at least two streamed chunks before the response, got {chunks:?}"
    );
    assert_eq!(
        chunks.concat(),
        "Hello world",
        "concatenated deltas equal the full text with no duplicate \
         whole-message chunk: {chunks:?}"
    );
}

#[test]
fn session_cancel_mid_turn_returns_cancelled() {
    let root = temp_root("cancel");
    let port = spawn_mock_sse_server(vec![STALL]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "never finishes");
    // Give the turn a moment to reach the stalled provider call, then cancel.
    std::thread::sleep(std::time::Duration::from_millis(200));
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","method":"session/cancel","params":{{"sessionId":"{session_id}"}}}}"#
    ));
    let (response, _, _) = client.read_until_response(3, None);
    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "cancel mid-turn resolves the prompt with the cancelled stop reason: {response}"
    );
}

#[test]
fn shell_approval_allow_once_executes_the_command() {
    let root = temp_root("allow");
    let port = spawn_mock_sse_server(vec![SHELL_TURN, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "run it");
    let (response, updates, permissions) = client.read_until_response(3, Some("allow-once"));

    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    assert_eq!(permissions.len(), 1, "one gated command, one prompt");
    assert_eq!(
        permissions[0]["toolCall"]["title"], "echo acp-approved",
        "the permission request names the command: {}",
        permissions[0]
    );
    assert!(
        permissions[0]["options"]
            .as_array()
            .is_some_and(|options| options.len() == 3),
        "allow once / allow always / reject: {}",
        permissions[0]
    );
    let executed = updates.iter().any(|update| {
        update["update"]["sessionUpdate"] == "tool_call_update"
            && update["update"]["status"] == "completed"
            && update["update"]["rawOutput"]["stdout"]
                .as_str()
                .is_some_and(|out| out.contains("acp-approved"))
    });
    assert!(
        executed,
        "the approved shell command ran and its output surfaced: {updates:?}"
    );
    assert!(
        chunk_texts(&updates).concat().contains("done"),
        "the follow-up turn streamed its answer: {updates:?}"
    );
}

#[test]
fn new_session_reports_modes_and_model_options() {
    let root = temp_root("modes");
    let port = spawn_mock_sse_server(vec![]);
    let mut client = AcpClient::start(&root, port);
    client.send(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
    );
    client.read_until_response(1, None);
    let cwd = root.to_str().unwrap().to_owned();
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{"cwd":"{cwd}","mcpServers":[]}}}}"#
    ));
    let (new_session, _, _) = client.read_until_response(2, None);
    let result = &new_session["result"];
    assert_eq!(
        result["modes"]["currentModeId"], "ask",
        "approval gating is the default mode: {result}"
    );
    let mode_ids: Vec<&str> = result["modes"]["availableModes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|mode| mode["id"].as_str())
        .collect();
    assert_eq!(mode_ids, ["ask", "yolo"], "{result}");
    assert_eq!(
        result["configOptions"][0]["id"], "model",
        "a model picker is advertised: {result}"
    );
}

#[test]
fn gc_config_options_are_advertised_and_switchable() {
    let root = temp_root("gc-config");
    let port = spawn_mock_sse_server(vec![TEXT_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    // session/new already advertised the GC options; fetch them via a
    // strategy switch and assert the refreshed list reflects it.
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/set_config_option","params":{{"sessionId":"{session_id}","configId":"gc","value":"semantic"}}}}"#
    ));
    let (response, _, _) = client.read_until_response(3, None);
    let options = response["result"]["configOptions"]
        .as_array()
        .unwrap_or_else(|| panic!("no configOptions in {response}"));
    let gc = options
        .iter()
        .find(|option| option["id"] == "gc")
        .unwrap_or_else(|| panic!("no gc option in {response}"));
    assert_eq!(gc["currentValue"], "semantic", "{response}");

    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":4,"method":"session/set_config_option","params":{{"sessionId":"{session_id}","configId":"gc-threshold","value":"0.7"}}}}"#
    ));
    let (response, _, _) = client.read_until_response(4, None);
    let threshold = response["result"]["configOptions"]
        .as_array()
        .and_then(|options| options.iter().find(|option| option["id"] == "gc-threshold"))
        .unwrap_or_else(|| panic!("no gc-threshold option in {response}"));
    assert_eq!(threshold["currentValue"], "0.7", "{response}");

    // Bad values are rejected without killing the session.
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":5,"method":"session/set_config_option","params":{{"sessionId":"{session_id}","configId":"gc","value":"bogus"}}}}"#
    ));
    let (response, _, _) = client.read_until_response(5, None);
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("unknown gc strategy")),
        "{response}"
    );

    // The session still takes turns after the config churn.
    client.prompt(6, &session_id, "greet");
    let (response, _, _) = client.read_until_response(6, None);
    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
}

#[test]
fn gc_config_survives_session_load_without_an_intervening_turn() {
    let root = temp_root("gc-config-load");
    let port = spawn_mock_sse_server(vec![]);
    let session_id;
    {
        let mut client = AcpClient::start(&root, port);
        session_id = client.handshake_and_new_session(root.to_str().unwrap());
        client.send(&format!(
            r#"{{"jsonrpc":"2.0","id":3,"method":"session/set_config_option","params":{{"sessionId":"{session_id}","configId":"gc","value":"semantic"}}}}"#
        ));
        client.read_until_response(3, None);
        client.send(&format!(
            r#"{{"jsonrpc":"2.0","id":4,"method":"session/set_config_option","params":{{"sessionId":"{session_id}","configId":"gc-threshold","value":"0.7"}}}}"#
        ));
        client.read_until_response(4, None);
    }

    let port = spawn_mock_sse_server(vec![]);
    let mut client = AcpClient::start(&root, port);
    client.send(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
    );
    client.read_until_response(1, None);
    let cwd = root.to_str().unwrap();
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{session_id}","cwd":"{cwd}","mcpServers":[]}}}}"#
    ));
    let (response, _, _) = client.read_until_response(2, None);
    let options = response["result"]["configOptions"]
        .as_array()
        .unwrap_or_else(|| panic!("no configOptions in {response}"));
    let current = |id: &str| {
        options
            .iter()
            .find(|option| option["id"] == id)
            .and_then(|option| option["currentValue"].as_str())
    };
    assert_eq!(current("gc"), Some("semantic"), "{response}");
    assert_eq!(current("gc-threshold"), Some("0.7"), "{response}");
}

#[test]
fn model_switch_clears_the_discovered_gc_ceiling() {
    let root = temp_root("model-gc-ceiling");
    let config_dir = root.join("config/agent");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("models.yaml"),
        r#"default_model: test-model
models:
  - name: test-model
    provider: openai-compatible
    api_id: test-model
    context: 1000
  - name: larger-model
    provider: openai-compatible
    api_id: larger-api-model
    context: 2000
"#,
    )
    .unwrap();

    let port = spawn_mock_sse_server(vec![]);
    let session_id;
    {
        let mut client = AcpClient::start(&root, port);
        session_id = client.handshake_and_new_session(root.to_str().unwrap());
        // Force an initial checkpoint without needing a provider turn.
        client.send(&format!(
            r#"{{"jsonrpc":"2.0","id":3,"method":"session/set_config_option","params":{{"sessionId":"{session_id}","configId":"gc-threshold","value":"0.7"}}}}"#
        ));
        client.read_until_response(3, None);
    }
    let checkpoint = root
        .join(".local/share/agent/acp")
        .join(&session_id)
        .join("session-latest.json");
    let mut saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&checkpoint).unwrap()).unwrap();
    saved["discovered_budget"] = serde_json::json!(777);
    std::fs::write(&checkpoint, serde_json::to_vec_pretty(&saved).unwrap()).unwrap();

    let port = spawn_mock_sse_server(vec![]);
    let mut client = AcpClient::start(&root, port);
    client.send(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
    );
    client.read_until_response(1, None);
    let cwd = root.to_str().unwrap();
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{session_id}","cwd":"{cwd}","mcpServers":[]}}}}"#
    ));
    client.read_until_response(2, None);
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/set_config_option","params":{{"sessionId":"{session_id}","configId":"model","value":"larger-model"}}}}"#
    ));
    let (response, _, _) = client.read_until_response(3, None);
    let model = response["result"]["configOptions"]
        .as_array()
        .and_then(|options| options.iter().find(|option| option["id"] == "model"))
        .unwrap_or_else(|| panic!("no model option in {response}"));
    assert_eq!(model["currentValue"], "larger-model", "{response}");

    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&checkpoint).unwrap()).unwrap();
    assert_eq!(saved["model"], "larger-api-model");
    assert_eq!(saved["model_alias"], "larger-model");
    assert!(saved["discovered_budget"].is_null(), "{saved}");
    drop(client);

    // A fresh ACP process must resolve the saved alias rather than reverting
    // to its process-level `--model test-model` argument.
    let port = spawn_mock_sse_server(vec![]);
    let mut client = AcpClient::start(&root, port);
    client.send(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
    );
    client.read_until_response(1, None);
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{session_id}","cwd":"{cwd}","mcpServers":[]}}}}"#
    ));
    let (response, _, _) = client.read_until_response(2, None);
    let model = response["result"]["configOptions"]
        .as_array()
        .and_then(|options| options.iter().find(|option| option["id"] == "model"))
        .unwrap_or_else(|| panic!("no model option in {response}"));
    assert_eq!(model["currentValue"], "larger-model", "{response}");
}

#[test]
fn unknown_mode_is_rejected_without_changing_the_current_mode() {
    let root = temp_root("unknown-mode");
    let port = spawn_mock_sse_server(vec![]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/set_mode","params":{{"sessionId":"{session_id}","modeId":"bogus"}}}}"#
    ));
    let (response, updates, _) = client.read_until_response(3, None);
    assert_eq!(response["error"]["code"], -32602, "{response}");
    assert!(
        updates.is_empty(),
        "rejection must emit no update: {updates:?}"
    );

    // Switching to the previous/default mode still succeeds, proving the
    // actor remains in the advertised `ask` state after rejection.
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":4,"method":"session/set_mode","params":{{"sessionId":"{session_id}","modeId":"ask"}}}}"#
    ));
    let (response, updates, _) = client.read_until_response(4, None);
    assert!(response.get("result").is_some(), "{response}");
    assert!(updates
        .iter()
        .any(|u| u["update"]["currentModeId"] == "ask"));
}

#[test]
fn yolo_mode_runs_shell_without_permission_prompts() {
    let root = temp_root("yolo");
    let port = spawn_mock_sse_server(vec![SHELL_TURN, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/set_mode","params":{{"sessionId":"{session_id}","modeId":"yolo"}}}}"#
    ));
    let (_, updates, _) = client.read_until_response(3, None);
    assert!(
        updates
            .iter()
            .any(|update| update["update"]["currentModeId"] == "yolo"),
        "the mode change is announced: {updates:?}"
    );

    // permission_option None: any session/request_permission would panic.
    client.prompt(4, &session_id, "run it");
    let (response, updates, _) = client.read_until_response(4, None);
    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    let executed = updates.iter().any(|update| {
        update["update"]["sessionUpdate"] == "tool_call_update"
            && update["update"]["rawOutput"]["stdout"]
                .as_str()
                .is_some_and(|out| out.contains("acp-approved"))
    });
    assert!(
        executed,
        "yolo mode runs the command unprompted: {updates:?}"
    );
}

#[test]
fn session_load_replays_history_and_continues() {
    let root = temp_root("load");
    let port = spawn_mock_sse_server(vec![TEXT_TURN]);
    let session_id;
    {
        let mut client = AcpClient::start(&root, port);
        session_id = client.handshake_and_new_session(root.to_str().unwrap());
        client.prompt(3, &session_id, "greet");
        let (response, _, _) = client.read_until_response(3, None);
        assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    } // client drops: agent exits, checkpoint persists

    let port = spawn_mock_sse_server(vec![DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    client.send(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
    );
    client.read_until_response(1, None);
    let cwd = root.to_str().unwrap().to_owned();
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{session_id}","cwd":"{cwd}","mcpServers":[]}}}}"#
    ));
    let (load_response, updates, _) = client.read_until_response(2, None);
    assert!(
        load_response["result"]["modes"]["currentModeId"] == "ask",
        "load reports modes too: {load_response}"
    );
    let replayed_user = updates.iter().any(|update| {
        update["update"]["sessionUpdate"] == "user_message_chunk"
            && update["update"]["content"]["text"] == "greet"
    });
    let replayed_agent = updates.iter().any(|update| {
        update["update"]["sessionUpdate"] == "agent_message_chunk"
            && update["update"]["content"]["text"] == "Hello world"
    });
    assert!(
        replayed_user && replayed_agent,
        "history replays before the load response: {updates:?}"
    );

    client.prompt(3, &session_id, "continue");
    let (response, updates, _) = client.read_until_response(3, None);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "the loaded session takes new turns: {response}"
    );
    assert!(
        chunk_texts(&updates).concat().contains("done"),
        "{updates:?}"
    );
}

/// Regression: an `allow-once` given during a turn that is later CANCELLED
/// must not survive into the next turn. The cancelled turn never advances
/// the effect-visit counters, so the next turn's gated command mints the
/// same effect id — a stale resolution would approve a different command
/// with no prompt.
#[test]
fn allow_once_does_not_leak_across_cancellation() {
    let root = temp_root("approval-leak");
    // Turn 1: tool call (approved) → follow-up model call stalls → cancel.
    // Turn 2: tool call again → must prompt again → reject → final text.
    let port = spawn_mock_sse_server(vec![SHELL_TURN, STALL, SHELL_TURN, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "run it");
    // Drive turn 1 by hand: allow the permission, then cancel once the
    // approved command has executed and the turn is stalled in the
    // follow-up inference.
    let mut cancelled = false;
    loop {
        let mut line = String::new();
        let n = client.reader.read_line(&mut line).expect("read frame");
        assert!(n > 0, "stream closed during turn 1");
        let frame: serde_json::Value = serde_json::from_str(&line).expect("frame");
        match frame.get("method").and_then(|m| m.as_str()) {
            Some("session/request_permission") => {
                let request_id = frame["id"].clone();
                client.send(&format!(
                    r#"{{"jsonrpc":"2.0","id":{request_id},"result":{{"outcome":{{"outcome":"selected","optionId":"allow-once"}}}}}}"#
                ));
            }
            Some("session/update")
                if !cancelled
                    && frame["params"]["update"]["sessionUpdate"] == "tool_call_update"
                    && frame["params"]["update"]["status"] == "completed" =>
            {
                // The approved command ran; give the turn a moment to reach
                // the stalled follow-up model call (keeping the mock's
                // request/body sequence aligned), then cancel it.
                std::thread::sleep(std::time::Duration::from_millis(200));
                client.send(&format!(
                    r#"{{"jsonrpc":"2.0","method":"session/cancel","params":{{"sessionId":"{session_id}"}}}}"#
                ));
                cancelled = true;
            }
            None if frame["id"] == 3 => {
                assert_eq!(
                    frame["result"]["stopReason"], "cancelled",
                    "turn 1 ends cancelled: {frame}"
                );
                break;
            }
            _ => {}
        }
    }
    assert!(
        cancelled,
        "the test cancelled after the approved command ran"
    );

    // Turn 2: the gated command must prompt AGAIN — reject it this time and
    // verify the denial (not the stale approval) is what the model sees.
    client.prompt(4, &session_id, "run it again");
    let (response, updates, permissions) = client.read_until_response(4, Some("reject-once"));
    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    assert_eq!(
        permissions.len(),
        1,
        "a fresh permission request is required after the cancelled turn"
    );
    let executed = updates.iter().any(|update| {
        update["update"]["sessionUpdate"] == "tool_call_update"
            && update["update"]["rawOutput"]["stdout"]
                .as_str()
                .is_some_and(|out| out.contains("acp-approved"))
    });
    assert!(
        !executed,
        "the rejected command must not run on the stale approval: {updates:?}"
    );
}

#[test]
fn shell_approval_reject_denies_without_running() {
    let root = temp_root("reject");
    let port = spawn_mock_sse_server(vec![SHELL_TURN, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "run it");
    let (response, updates, permissions) = client.read_until_response(3, Some("reject-once"));

    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "a denial is non-fatal — the model sees it and finishes: {response}"
    );
    assert_eq!(permissions.len(), 1);
    let executed = updates.iter().any(|update| {
        update["update"]["sessionUpdate"] == "tool_call_update"
            && update["update"]["rawOutput"]["stdout"]
                .as_str()
                .is_some_and(|out| out.contains("acp-approved"))
    });
    assert!(!executed, "the denied command must not run: {updates:?}");
    let denied = updates.iter().any(|update| {
        update["update"]["sessionUpdate"] == "tool_call_update"
            && update["update"]["status"] == "failed"
    });
    assert!(
        denied,
        "the denial surfaces as a failed tool_call_update: {updates:?}"
    );
}
