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

/// Parent turn delegating to a child model via the `infer` tool.
const INFER_TURN: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"infer\",\"arguments\":\"{\\\"model\\\":\\\"test-model\\\",\\\"prompt\\\":\\\"summarize the logs\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: [DONE]\n\n";

/// The delegated child's streamed answer.
const CHILD_TURN: &str =
    "data: {\"choices\":[{\"delta\":{\"content\":\"child \"},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";

/// Marker body: read the request, then stall without answering (the turn
/// hangs until the client cancels and the agent drops the connection).
const STALL: &str = "<stall>";

/// An `ask_user` call (t-1594) with two options.
const ASK_TURN: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_ask\",\"function\":{\"name\":\"ask_user\",\"arguments\":\"{\\\"question\\\":\\\"Which database?\\\",\\\"options\\\":[\\\"Postgres\\\",\\\"SQLite\\\"]}\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: [DONE]\n\n";

const MODE_TURN: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_mode\",\"function\":{\"name\":\"switch_mode\",\"arguments\":\"{\\\"mode\\\":\\\"ask\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
const YOLO_MODE_TURN: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_yolo\",\"function\":{\"name\":\"switch_mode\",\"arguments\":\"{\\\"mode\\\":\\\"yolo\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
const CURRENT_MODE_TURN: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_mode\",\"function\":{\"name\":\"switch_mode\",\"arguments\":\"{\\\"mode\\\":\\\"work\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
const UNKNOWN_MODE_TURN: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_mode\",\"function\":{\"name\":\"switch_mode\",\"arguments\":\"{\\\"mode\\\":\\\"absent\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";

/// Permission-option marker: answer the request with the `cancelled`
/// outcome instead of selecting an option.
const CANCEL_PERMISSION: &str = "<cancelled>";

/// Request bodies the mock provider received, in arrival order.
type RecordedRequests = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

/// Mock provider serving one canned SSE body per request, in order.
fn spawn_mock_sse_server(bodies: Vec<&'static str>) -> u16 {
    spawn_recording_mock_sse_server(bodies).0
}

/// [`spawn_mock_sse_server`] that also records every request body.
fn spawn_recording_mock_sse_server(bodies: Vec<&'static str>) -> (u16, RecordedRequests) {
    let recorded = RecordedRequests::default();
    let sink = recorded.clone();
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
            sink.lock().unwrap().push(request_body.to_string());
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
    (port, recorded)
}

fn spawn_acp_agent(root: &std::path::Path, port: u16) -> Child {
    spawn_acp_agent_with_model(root, port, "test-model")
}

fn spawn_acp_agent_with_model(root: &std::path::Path, port: u16, model: &str) -> Child {
    Command::new(env!("CARGO_BIN_EXE_agent"))
        .arg("--acp")
        .args(["--model", model])
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
        Self::from_child(spawn_acp_agent(root, port))
    }

    fn from_child(mut child: Child) -> Self {
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
                    if option == CANCEL_PERMISSION {
                        self.send(&format!(
                            r#"{{"jsonrpc":"2.0","id":{request_id},"result":{{"outcome":{{"outcome":"cancelled"}}}}}}"#
                        ));
                        continue;
                    }
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
fn prompt_response_reports_cumulative_token_usage_for_paseo() {
    let root = temp_root("usage");
    let port = spawn_mock_sse_server(vec![TEXT_TURN, TEXT_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    for (id, multiplier) in [(3, 1), (4, 2)] {
        client.prompt(id, &session_id, "greet");
        let (response, updates, _) = client.read_until_response(id, None);
        assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
        let usage = &response["result"]["usage"];
        assert_eq!(usage["inputTokens"], 3 * multiplier, "{response}");
        assert_eq!(usage["outputTokens"], 4 * multiplier, "{response}");
        assert_eq!(usage["totalTokens"], 7 * multiplier, "{response}");
        assert!(
            updates.iter().any(|update| {
                update["update"]["sessionUpdate"] == "usage_update"
                    && update["update"]["size"].as_u64().unwrap_or(0) > 0
            }),
            "standard ACP context updates still arrive: {updates:?}"
        );
    }
}

#[test]
fn load_without_checkpoint_leaves_live_session_usable() {
    let root = temp_root("load-before-checkpoint");
    let port = spawn_mock_sse_server(vec![TEXT_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/load","params":{{"sessionId":"{session_id}","cwd":"{}","mcpServers":[]}}}}"#,
        root.display()
    ));
    assert!(client.read_until_response(3, None).0.get("error").is_some());
    client.prompt(4, &session_id, "still here");
    let (response, _, _) = client.read_until_response(4, None);
    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
}

#[test]
fn loading_live_session_cancels_old_turn_before_restoring_checkpoint() {
    let root = temp_root("live-reload");
    let (port, requests) = spawn_recording_mock_sse_server(vec![TEXT_TURN, STALL, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());
    client.prompt(3, &session_id, "committed");
    assert_eq!(
        client.read_until_response(3, None).0["result"]["stopReason"],
        "end_turn"
    );
    let checkpoint_path = root
        .join(".local/share/agent/acp")
        .join(&session_id)
        .join("session-latest.json");
    let before: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&checkpoint_path).unwrap()).unwrap();
    client.prompt(4, &session_id, "unfinished");
    for _ in 0..200 {
        if requests.lock().unwrap().len() == 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "old turn reached provider"
    );
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":5,"method":"session/load","params":{{"sessionId":"{session_id}","cwd":"{}","mcpServers":[]}}}}"#,
        root.display()
    ));
    let (loaded, updates, _) = client.read_until_response(5, None);
    assert!(loaded.get("result").is_some(), "{loaded}");
    assert!(updates
        .iter()
        .any(|u| u["update"]["content"]["text"] == "committed"));
    assert!(!updates
        .iter()
        .any(|u| u["update"]["content"]["text"] == "unfinished"));
    let after: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&checkpoint_path).unwrap()).unwrap();
    assert_eq!(before["sequence"], after["sequence"]);
    assert_eq!(before["messages"], after["messages"]);
    client.prompt(6, &session_id, "after load");
    let (response, _, _) = client.read_until_response(6, None);
    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    let final_cp: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint_path).unwrap()).unwrap();
    let history = final_cp["messages"].to_string();
    assert!(
        history.contains("committed") && history.contains("after load"),
        "{history}"
    );
    assert!(!history.contains("unfinished"), "{history}");
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
fn infer_delegation_output_stays_in_its_tool_card() {
    let root = temp_root("infer-card");
    let port = spawn_mock_sse_server(vec![INFER_TURN, CHILD_TURN, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "delegate it");
    let (response, updates, _) = client.read_until_response(3, None);

    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    assert_eq!(
        chunk_texts(&updates).concat(),
        "done",
        "only the root answer reaches the assistant reply: {updates:?}"
    );
    let card = updates
        .iter()
        .find(|update| update["update"]["sessionUpdate"] == "tool_call")
        .unwrap_or_else(|| panic!("the delegation opens a tool card: {updates:?}"));
    let card_id = &card["update"]["toolCallId"];
    assert!(
        card["update"]["title"].as_str().is_some_and(
            |title| title.contains("test-model") && title.contains("summarize the logs")
        ),
        "the card names the child model and task: {card}"
    );
    let completed = updates.iter().any(|update| {
        update["update"]["sessionUpdate"] == "tool_call_update"
            && &update["update"]["toolCallId"] == card_id
            && update["update"]["status"] == "completed"
            && update["update"]["content"][0]["content"]["text"] == "child answer"
    });
    assert!(
        completed,
        "the child's answer completes the card: {updates:?}"
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

/// The `ask_user` tool message the model received in provider request
/// `index`, parsed from the recorded request body.
fn ask_user_tool_result(requests: &RecordedRequests, index: usize) -> String {
    let requests = requests.lock().unwrap();
    let body: serde_json::Value = serde_json::from_str(&requests[index]).unwrap();
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["tool_call_id"] == "call_ask")
        .and_then(|message| message["content"].as_str())
        .unwrap_or_else(|| panic!("no ask_user tool result in request {index}: {body}"))
        .to_owned()
}

#[test]
fn ask_user_is_a_paseo_chooser_and_the_answer_reaches_the_model() {
    let root = temp_root("ask-answer");
    let (port, requests) = spawn_recording_mock_sse_server(vec![ASK_TURN, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "set up the service");
    let (response, updates, permissions) = client.read_until_response(3, Some("1"));

    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    assert!(
        requests.lock().unwrap()[0].contains("\"ask_user\""),
        "ACP sessions advertise ask_user to the model"
    );
    assert_eq!(permissions.len(), 1, "one question, one prompt");
    let request = &permissions[0];
    assert_eq!(request["toolCall"]["title"], "Which database?", "{request}");
    assert_eq!(
        request["toolCall"]["content"][0]["content"]["text"], "Which database?",
        "{request}"
    );
    let options: Vec<(&str, &str, &str)> = request["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|option| {
            (
                option["optionId"].as_str().unwrap(),
                option["name"].as_str().unwrap(),
                option["kind"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        options,
        [
            ("0", "Postgres", "allow_once"),
            ("1", "SQLite", "allow_once"),
            ("dismiss", "Dismiss", "reject_once"),
        ],
        "same-kind allow options make Paseo render a chooser: {request}"
    );
    assert_eq!(ask_user_tool_result(&requests, 1), "SQLite");
    assert!(
        updates.iter().any(|update| {
            update["update"]["sessionUpdate"] == "tool_call_update"
                && update["update"]["status"] == "completed"
                && update["update"]["rawOutput"] == "SQLite"
        }),
        "the answered question completes its tool call: {updates:?}"
    );
}

#[test]
fn ask_user_dismissal_reaches_the_model_and_the_turn_continues() {
    let root = temp_root("ask-dismiss");
    let (port, requests) = spawn_recording_mock_sse_server(vec![ASK_TURN, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "set up the service");
    let (response, _, permissions) = client.read_until_response(3, Some("dismiss"));

    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    assert_eq!(permissions.len(), 1);
    let result = ask_user_tool_result(&requests, 1);
    assert!(
        result.contains("dismissed the question"),
        "the model reads the dismissal: {result}"
    );
}

/// yolo silences shell approvals, never questions: a question needs an
/// answer only the user can give.
#[test]
fn ask_user_still_asks_in_yolo_mode() {
    let root = temp_root("ask-yolo");
    let (port, requests) = spawn_recording_mock_sse_server(vec![ASK_TURN, DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());
    client.send(&format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/set_mode","params":{{"sessionId":"{session_id}","modeId":"yolo"}}}}"#
    ));
    client.read_until_response(3, None);

    client.prompt(4, &session_id, "set up the service");
    let (response, _, permissions) = client.read_until_response(4, Some("0"));
    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    assert_eq!(permissions.len(), 1, "the question still reaches the user");
    assert_eq!(ask_user_tool_result(&requests, 1), "Postgres");
}

#[test]
fn ask_user_cancelled_permission_cancels_the_turn() {
    let root = temp_root("ask-cancel");
    let (port, requests) = spawn_recording_mock_sse_server(vec![ASK_TURN]);
    let mut client = AcpClient::start(&root, port);
    let session_id = client.handshake_and_new_session(root.to_str().unwrap());

    client.prompt(3, &session_id, "set up the service");
    let (response, _, permissions) = client.read_until_response(3, Some(CANCEL_PERMISSION));

    assert_eq!(response["result"]["stopReason"], "cancelled", "{response}");
    assert_eq!(permissions.len(), 1);
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "a cancelled question never resumes the loop"
    );
}

#[test]
fn thought_level_switches_request_shape_and_model_choices() {
    let root = temp_root("thought-config");
    let config = root.join("config/agent");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(
        config.join("models.yaml"),
        r#"default_model: test-model
models:
  - name: test-model
    provider: openai-compatible
    api_id: test-model
    context: 10000
    thinking: true
    efforts: [low, high]
    effort: low
  - name: second-model
    provider: openai-compatible
    api_id: second-model
    context: 10000
    thinking: true
    efforts: [medium, max]
    effort: medium
  - name: plain-model
    provider: openai-compatible
    api_id: plain-model
    context: 10000
"#,
    )
    .unwrap();
    let (port, requests) = spawn_recording_mock_sse_server(vec![TEXT_TURN, TEXT_TURN]);
    let mut client = AcpClient::start(&root, port);
    client.send(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#);
    client.read_until_response(1, None);
    client.send(&format!(r#"{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{"cwd":"{}","mcpServers":[]}}}}"#, root.display()));
    let (new, _, _) = client.read_until_response(2, None);
    let sid = new["result"]["sessionId"].as_str().unwrap();
    let opts = new["result"]["configOptions"].as_array().unwrap();
    let thought = opts.iter().find(|o| o["id"] == "thought_level").unwrap();
    assert_eq!(thought["category"], "thought_level");
    assert_eq!(thought["currentValue"], "low");
    assert_eq!(thought["options"].as_array().unwrap().len(), 2);
    let sid = sid.to_owned();
    let set = |client: &mut AcpClient, id, config_id: &str, value: &str| {
        client.send(&format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"session/set_config_option","params":{{"sessionId":"{sid}","configId":"{config_id}","value":"{value}"}}}}"#));
        client.read_until_response(id, None).0
    };
    let response = set(&mut client, 3, "thought_level", "high");
    assert_eq!(
        response["result"]["configOptions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["id"] == "thought_level")
            .unwrap()["currentValue"],
        "high"
    );
    client.prompt(4, &sid, "hello");
    assert_eq!(
        client.read_until_response(4, None).0["result"]["stopReason"],
        "end_turn"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&requests.lock().unwrap()[0]).unwrap()
            ["reasoning_effort"],
        "high"
    );
    let response = set(&mut client, 5, "model", "second-model");
    let thought = response["result"]["configOptions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["id"] == "thought_level")
        .unwrap();
    assert_eq!(thought["currentValue"], "medium");
    assert_eq!(thought["options"].as_array().unwrap().len(), 2);
    client.prompt(6, &sid, "again");
    client.read_until_response(6, None);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&requests.lock().unwrap()[1]).unwrap()
            ["reasoning_effort"],
        "medium"
    );
    // The selection is checkpointed even without another provider turn.
    drop(client);
    let mut client = AcpClient::start(&root, port);
    client.send(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#);
    client.read_until_response(1, None);
    client.send(&format!(r#"{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{sid}","cwd":"{}","mcpServers":[]}}}}"#, root.display()));
    let (loaded, _, _) = client.read_until_response(2, None);
    assert_eq!(
        loaded["result"]["configOptions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["id"] == "thought_level")
            .unwrap()["currentValue"],
        "medium",
        "{loaded}"
    );
    let response = set(&mut client, 7, "model", "plain-model");
    assert!(response["result"]["configOptions"]
        .as_array()
        .unwrap()
        .iter()
        .all(|o| o["id"] != "thought_level"));
}

#[test]
fn native_reasoning_deltas_arrive_as_thought_chunks() {
    const ANTHROPIC_TURN: &str = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":2}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"consider\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let root = temp_root("reasoning");
    let config = root.join("config/agent");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(config.join("models.yaml"), "default_model: test-model\nmodels:\n  - name: test-model\n    provider: anthropic\n    api_id: test-model\n    context: 10000\n    thinking: true\n    efforts: [low, high]\n    effort: low\n").unwrap();
    let port = spawn_mock_sse_server(vec![ANTHROPIC_TURN]);
    let mut client = AcpClient::from_child(spawn_acp_agent_with_model(&root, port, "test-model"));
    let sid = client.handshake_and_new_session(root.to_str().unwrap());
    client.prompt(3, &sid, "reason");
    let (response, updates, _) = client.read_until_response(3, None);
    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
    assert!(
        updates
            .iter()
            .any(|u| u["update"]["sessionUpdate"] == "agent_thought_chunk"
                && u["update"]["content"]["text"] == "consider"),
        "{updates:?}"
    );
    assert!(chunk_texts(&updates).contains(&"done"));
}

fn mode_profile(root: &std::path::Path, tools: &str) {
    let dir = root.join("config/agent/modes");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("work.md"),
        format!("---\nname: Work\nshell_approval: never\ntools: [{tools}]\n---\nWork."),
    )
    .unwrap();
}

fn select_mode(client: &mut AcpClient, sid: &str) {
    client.send(&format!(r#"{{"jsonrpc":"2.0","id":3,"method":"session/set_mode","params":{{"sessionId":"{sid}","modeId":"work"}}}}"#));
    let (response, _, _) = client.read_until_response(3, None);
    assert!(response.get("result").is_some(), "{response}");
}

#[test]
fn switch_mode_requires_consent_then_applies_after_turn_and_survives_load() {
    let root = temp_root("switch-mode");
    mode_profile(&root, "switch_mode");
    let (port, requests) = spawn_recording_mock_sse_server(vec![MODE_TURN, DONE_TURN]);
    let sid;
    {
        let mut client = AcpClient::start(&root, port);
        sid = client.handshake_and_new_session(root.to_str().unwrap());
        select_mode(&mut client, &sid);
        client.prompt(4, &sid, "proceed");
        let (response, updates, permissions) = client.read_until_response(4, Some("allow-once"));
        assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
        assert_eq!(permissions.len(), 1, "ungated shell mode must still ask");
        assert_eq!(permissions[0]["toolCall"]["kind"], "switch_mode");
        assert_eq!(permissions[0]["options"][0]["kind"], "allow_once");
        assert_eq!(permissions[0]["options"][1]["kind"], "reject_once");
        let position = |kind: &str| {
            updates
                .iter()
                .position(|u| u["update"]["sessionUpdate"] == kind)
                .unwrap()
        };
        assert!(position("tool_call") < position("current_mode_update"));
        assert!(requests.lock().unwrap()[1].contains("switch_mode"));
    }
    let port = spawn_mock_sse_server(vec![DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    client.send(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#);
    client.read_until_response(1, None);
    client.send(&format!(r#"{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{sid}","cwd":"{}","mcpServers":[]}}}}"#, root.display()));
    let (loaded, updates, _) = client.read_until_response(2, None);
    assert_eq!(loaded["result"]["modes"]["currentModeId"], "ask");
    assert!(updates.iter().any(
        |u| u["update"]["sessionUpdate"] == "tool_call" && u["update"]["kind"] == "switch_mode"
    ));
    client.prompt(3, &sid, "continue");
    let (response, _, _) = client.read_until_response(3, None);
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

#[test]
fn switch_mode_rejection_and_invalid_target_do_not_change_mode() {
    for (body, option, expected_prompts) in [
        (MODE_TURN, "reject-once", 1),
        (UNKNOWN_MODE_TURN, "reject-once", 0),
        (CURRENT_MODE_TURN, "reject-once", 0),
    ] {
        let root = temp_root("switch-reject");
        mode_profile(&root, "switch_mode");
        let (port, requests) = spawn_recording_mock_sse_server(vec![body, DONE_TURN]);
        let mut client = AcpClient::start(&root, port);
        let sid = client.handshake_and_new_session(root.to_str().unwrap());
        select_mode(&mut client, &sid);
        client.prompt(4, &sid, "proceed");
        let (response, updates, permissions) = client.read_until_response(4, Some(option));
        assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");
        assert_eq!(permissions.len(), expected_prompts);
        assert!(!updates
            .iter()
            .any(|u| u["update"]["sessionUpdate"] == "current_mode_update"));
        assert_eq!(requests.lock().unwrap().len(), 2);
        let body: serde_json::Value = serde_json::from_str(&requests.lock().unwrap()[1]).unwrap();
        let result = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["tool_call_id"] == "call_mode")
            .unwrap()["content"]
            .as_str()
            .unwrap();
        if expected_prompts == 1 {
            assert!(result.contains("denied"), "{result}");
        } else {
            assert!(
                result.contains("error")
                    && (result.contains("unknown") || result.contains("already")),
                "{result}"
            );
        }
    }
}

#[test]
fn latest_mode_decision_wins_within_a_turn() {
    for (second, expected) in [("reject-once", "work"), ("allow-once", "yolo")] {
        let root = temp_root("switch-latest");
        mode_profile(&root, "switch_mode");
        let port = spawn_mock_sse_server(vec![MODE_TURN, YOLO_MODE_TURN, DONE_TURN]);
        let mut client = AcpClient::start(&root, port);
        let sid = client.handshake_and_new_session(root.to_str().unwrap());
        select_mode(&mut client, &sid);
        client.prompt(4, &sid, "choose");
        let mut updates = Vec::new();
        let mut count = 0;
        loop {
            let mut line = String::new();
            assert!(client.reader.read_line(&mut line).unwrap() > 0);
            let frame: serde_json::Value = serde_json::from_str(&line).unwrap();
            match frame["method"].as_str() {
                Some("session/update") => updates.push(frame["params"].clone()),
                Some("session/request_permission") => {
                    let option = if count == 0 { "allow-once" } else { second };
                    count += 1;
                    client.send(&format!(r#"{{"jsonrpc":"2.0","id":{},"result":{{"outcome":{{"outcome":"selected","optionId":"{option}"}}}}}}"#, frame["id"]));
                }
                _ if frame["id"] == 4 => {
                    assert_eq!(frame["result"]["stopReason"], "end_turn");
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(count, 2);
        assert_eq!(
            updates
                .iter()
                .filter(|u| u["update"]["sessionUpdate"] == "current_mode_update")
                .count(),
            usize::from(expected == "yolo")
        );
        // The turn's final mode is independently observable through session/load.
        drop(client);
        let port = spawn_mock_sse_server(vec![]);
        let mut client = AcpClient::start(&root, port);
        client.send(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#);
        client.read_until_response(1, None);
        client.send(&format!(r#"{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{sid}","cwd":"{}","mcpServers":[]}}}}"#, root.display()));
        let (loaded, _, _) = client.read_until_response(2, None);
        assert_eq!(loaded["result"]["modes"]["currentModeId"], expected);
    }
}

#[test]
fn switch_mode_is_not_offered_without_profile_opt_in() {
    let root = temp_root("switch-unoffered");
    mode_profile(&root, "");
    let (port, requests) = spawn_recording_mock_sse_server(vec![DONE_TURN]);
    let mut client = AcpClient::start(&root, port);
    let sid = client.handshake_and_new_session(root.to_str().unwrap());
    select_mode(&mut client, &sid);
    client.prompt(4, &sid, "hello");
    client.read_until_response(4, None);
    let body: serde_json::Value = serde_json::from_str(&requests.lock().unwrap()[0]).unwrap();
    assert!(!body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["function"]["name"] == "switch_mode"));
}
