//! Session facade integration tests: spawn the real `agent` binary as a
//! child process, credential-free. Turns are driven either by replay
//! fixtures (`--replay-trace`, built with `testing::SessionReplayFixture`)
//! or by a local OpenAI-compatible stub server when the test needs real
//! checkpoint writes (replay sessions deliberately do not checkpoint).

use agent_sdk::testing::SessionReplayFixture;
use agent_sdk::{Agent, SdkError, Session, SessionOptions, Tool, TurnOptions};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The fixture/raw model id every test uses; resolves via the binary's
/// registry-missing fallback under a hermetic HOME.
const MODEL: &str = "test-model";

/// Locate the workspace-built `agent` binary (target/debug/agent),
/// building it once if this test target was compiled without it
/// (e.g. `cargo test -p agent-sdk`).
fn agent_binary() -> PathBuf {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        let exe = std::env::current_exe().expect("test executable path");
        let debug_dir = exe
            .parent()
            .and_then(|deps| deps.parent())
            .expect("target/debug directory")
            .to_path_buf();
        let bin = debug_dir.join(format!("agent{}", std::env::consts::EXE_SUFFIX));
        if !bin.is_file() {
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../agent/Cargo.toml");
            let status = std::process::Command::new(env!("CARGO"))
                .args(["build", "--bin", "agent", "--manifest-path"])
                .arg(&manifest)
                .status()
                .expect("running cargo build for the agent binary");
            assert!(status.success(), "building the agent binary failed");
        }
        assert!(bin.is_file(), "agent binary not found at {}", bin.display());
        bin
    })
    .clone()
}

/// A fresh scratch root; the child's HOME points here so traces, config
/// lookups, and the model registry never touch the real user environment.
fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("agent-sdk-session-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn hermetic_env(root: &std::path::Path) -> Vec<(String, String)> {
    vec![
        ("HOME".into(), root.display().to_string()),
        (
            "XDG_CONFIG_HOME".into(),
            root.join("config").display().to_string(),
        ),
    ]
}

fn agent() -> Agent {
    Agent::builder(MODEL).build().unwrap()
}

fn replay_options(root: &std::path::Path, fixture: PathBuf) -> SessionOptions {
    SessionOptions {
        home: Some(root.join("session")),
        binary: Some(agent_binary()),
        replay_trace: Some(fixture),
        env: hermetic_env(root),
        ..SessionOptions::default()
    }
}

// ---------------------------------------------------------------------------
// Minimal OpenAI-compatible stub provider over TCP, for tests that need the
// child to write real checkpoints (crash recovery) or to observe a slow
// turn (send timeouts). Responses close the connection, so each chat call
// is one accept.
// ---------------------------------------------------------------------------

struct StubProvider {
    url: String,
    /// The JSON bodies of every /chat/completions request received.
    requests: Arc<tokio::sync::Mutex<Vec<Value>>>,
}

/// Serve `responses` in order: (delay before answering, response text).
async fn stub_provider(responses: Vec<(Duration, String)>) -> StubProvider {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let queue = Arc::new(tokio::sync::Mutex::new(VecDeque::from(responses)));
    let seen = requests.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            // Read headers.
            let mut buf = Vec::new();
            let header_end = loop {
                let mut byte = [0u8; 1024];
                let Ok(n) = socket.read(&mut byte).await else {
                    break None;
                };
                if n == 0 {
                    break None;
                }
                buf.extend_from_slice(&byte[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break Some(pos + 4);
                }
            };
            let Some(header_end) = header_end else {
                continue;
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
            let content_length: usize = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().ok())?
                })
                .unwrap_or(0);
            while buf.len() < header_end + content_length {
                let mut byte = [0u8; 4096];
                let Ok(n) = socket.read(&mut byte).await else {
                    break;
                };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&byte[..n]);
            }
            if let Ok(body) = serde_json::from_slice::<Value>(&buf[header_end..]) {
                seen.lock().await.push(body);
            }
            let (delay, text) = queue
                .lock()
                .await
                .pop_front()
                .unwrap_or((Duration::ZERO, "stub exhausted".into()));
            tokio::time::sleep(delay).await;
            let body = json!({
                "choices": [{
                    "message": { "content": text },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    });
    StubProvider { url, requests }
}

fn stub_options(root: &std::path::Path, stub: &StubProvider) -> SessionOptions {
    let mut env = hermetic_env(root);
    env.push(("AGENT_PROVIDER".into(), stub.url.clone()));
    env.push(("AGENT_API_KEY".into(), "stub-key".into()));
    SessionOptions {
        home: Some(root.join("session")),
        binary: Some(agent_binary()),
        env,
        ..SessionOptions::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sends_correlate_by_turn_id_not_order() {
    let root = scratch();
    let fixture = root.join("replay.jsonl");
    SessionReplayFixture::new(MODEL)
        .turn("alpha-reply")
        .turn("beta-reply")
        .write(&fixture)
        .await
        .unwrap();

    let session = Arc::new(
        Session::start(&agent(), replay_options(&root, fixture))
            .await
            .unwrap(),
    );

    // Queue two overlapping sends with caller-supplied ids; the agent
    // processes them in frame order (alpha then beta), but we await them in
    // REVERSE order — only id-based correlation gives each caller its own
    // response.
    let first = {
        let session = session.clone();
        tokio::spawn(async move {
            session
                .send_with(
                    "first prompt",
                    TurnOptions {
                        turn_id: Some("id-A".into()),
                        metadata: Some(json!({"sender": "test-a"})),
                        ..TurnOptions::default()
                    },
                )
                .await
        })
    };
    // Ensure the first frame is written before the second is queued.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let second = session
        .send_with(
            "second prompt",
            TurnOptions {
                turn_id: Some("id-B".into()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    let first = first.await.unwrap().unwrap();

    assert_eq!(first.turn_id, "id-A");
    assert_eq!(first.text, "alpha-reply");
    assert_eq!(first.metadata, Some(json!({"sender": "test-a"})));
    assert_eq!(second.turn_id, "id-B");
    assert_eq!(second.text, "beta-reply");

    session.stop().await.unwrap();
}

#[tokio::test]
async fn sdk_mints_a_turn_id_when_none_is_supplied() {
    let root = scratch();
    let fixture = root.join("replay.jsonl");
    SessionReplayFixture::new(MODEL)
        .turn("minted-reply")
        .write(&fixture)
        .await
        .unwrap();

    let session = Session::start(&agent(), replay_options(&root, fixture))
        .await
        .unwrap();
    let result = session.send("hello").await.unwrap();

    assert_eq!(result.text, "minted-reply");
    assert!(
        result.turn_id.starts_with("sdk-"),
        "SDK-minted id expected, got {}",
        result.turn_id
    );
    session.stop().await.unwrap();
}

#[tokio::test]
async fn stop_status_events_and_send_after_stop() {
    let root = scratch();
    let fixture = root.join("replay.jsonl");
    SessionReplayFixture::new(MODEL)
        .turn("one-reply")
        .write(&fixture)
        .await
        .unwrap();

    let session = Session::start(&agent(), replay_options(&root, fixture))
        .await
        .unwrap();
    let mut events = session.events();

    let status = session.status().await;
    assert!(status.alive);
    assert!(status.pid.is_some());
    assert_eq!(status.run_id, session.run_id());
    assert_eq!(status.trace_path, session.trace_path());

    session.send("go").await.unwrap();

    // Graceful stop: stdin close is the session loop's clean EOF path.
    let exit = session.stop().await.unwrap();
    assert!(exit.success(), "graceful stop should exit cleanly: {exit}");

    let status = session.status().await;
    assert!(!status.alive);
    assert!(status.last_event_ts.is_some());

    // The public event stream tails the trace to completion and ends.
    let mut names = Vec::new();
    while let Some(event) = events.next().await {
        assert_eq!(event.run_id, session.run_id());
        names.push(event.event);
    }
    assert!(
        names.iter().any(|name| name == "infer.completed"),
        "expected an infer.completed public event, got: {names:?}"
    );
    assert_eq!(
        names.last().map(String::as_str),
        Some("run.completed"),
        "session end must project run.completed: {names:?}"
    );

    // A stopped session refuses further turns, typed.
    let err = session.send("too late").await.unwrap_err();
    assert!(matches!(err, SdkError::Session(_)), "got: {err}");
}

#[tokio::test]
async fn send_timeout_leaves_the_turn_running_and_attach_recovers_it() {
    let root = scratch();
    let stub = stub_provider(vec![
        (Duration::from_millis(1500), "slow-reply".into()),
        (Duration::ZERO, "fast-reply".into()),
    ])
    .await;

    let session = Session::start(&agent(), stub_options(&root, &stub))
        .await
        .unwrap();

    // The caller times out; the turn does not.
    let err = session
        .send_with(
            "slow question",
            TurnOptions {
                turn_id: Some("slow-turn".into()),
                timeout: Some(Duration::from_millis(200)),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap_err();
    match err {
        SdkError::SendTimeout {
            ref turn_id,
            still_running,
        } => {
            assert_eq!(turn_id, "slow-turn");
            assert!(still_running, "the child must survive a send timeout");
        }
        other => panic!("expected SendTimeout, got: {other}"),
    }

    // Attach retrieves the timed-out turn's eventual result...
    let recovered = session.attach("slow-turn").await.unwrap();
    assert_eq!(recovered.text, "slow-reply");
    assert_eq!(recovered.turn_id, "slow-turn");

    // ...and a later send still works.
    let next = session.send("next question").await.unwrap();
    assert_eq!(next.text, "fast-reply");

    session.stop().await.unwrap();
}

#[tokio::test]
async fn kill_dash_nine_then_resume_keeps_history_intact() {
    let root = scratch();
    let stub = stub_provider(vec![
        (Duration::ZERO, "first-reply".into()),
        (Duration::ZERO, "second-reply".into()),
    ])
    .await;
    let options = stub_options(&root, &stub);

    // Instructions ride to the child via --system-prompt; the stub's
    // captured request proves they reached the session's history.
    let instructed = Agent::builder(MODEL)
        .instructions("You are the pineapple keeper.")
        .build()
        .unwrap();
    let session = Session::start(&instructed, options.clone()).await.unwrap();
    let run_id = session.run_id().to_owned();
    let first = session.send("remember the word pineapple").await.unwrap();
    assert_eq!(first.text, "first-reply");

    // The turn checkpointed before completing; SIGKILL loses nothing.
    let checkpoint = options
        .home
        .as_ref()
        .unwrap()
        .join("checkpoints/session-latest.json");
    assert!(checkpoint.is_file(), "turn completion implies a checkpoint");
    session.kill().await.unwrap();
    assert!(!session.status().await.alive);

    // Resume: same run id, and the next provider call carries the full
    // pre-crash conversation — history intact.
    let resumed = Session::resume(&agent(), options).await.unwrap();
    assert_eq!(resumed.run_id(), run_id, "resume keeps the run id");
    let second = resumed.send("what was the word?").await.unwrap();
    assert_eq!(second.text, "second-reply");

    let requests = stub.requests.lock().await;
    assert_eq!(requests.len(), 2);
    assert!(
        requests[0]["messages"]
            .to_string()
            .contains("pineapple keeper"),
        "Agent instructions must reach the child via --system-prompt: {}",
        requests[0]["messages"]
    );
    let resumed_messages = requests[1]["messages"].to_string();
    assert!(
        resumed_messages.contains("remember the word pineapple"),
        "resumed history must contain the pre-crash user turn: {resumed_messages}"
    );
    assert!(
        resumed_messages.contains("first-reply"),
        "resumed history must contain the pre-crash assistant turn: {resumed_messages}"
    );
    drop(requests);

    resumed.stop().await.unwrap();
}

#[tokio::test]
async fn resume_continues_agent_minted_turn_ordinals() {
    let root = scratch();
    let home = root.join("session");
    std::fs::create_dir_all(home.join("checkpoints")).unwrap();

    // A checkpoint as the agent would have written it after 3 turns.
    let messages = vec![
        agent_core::ChatMessage::system("sys"),
        agent_core::ChatMessage::user("hello"),
        agent_core::ChatMessage::assistant(Some("hi".into()), vec![]),
    ];
    let checkpoint = json!({
        "run_id": "sess-ordinal",
        "sequence": 3,
        "model": MODEL,
        "provider_url": "http://replay.invalid",
        "messages": serde_json::to_value(&messages).unwrap(),
        "trace_path": root.join("trace.jsonl"),
        "timestamp": "2026-01-01T00:00:00Z",
    });
    std::fs::write(
        home.join("checkpoints/session-latest.json"),
        serde_json::to_vec_pretty(&checkpoint).unwrap(),
    )
    .unwrap();

    let fixture = root.join("replay.jsonl");
    SessionReplayFixture::new(MODEL)
        .turn("resumed-reply")
        .write(&fixture)
        .await
        .unwrap();

    let session = Session::resume(&agent(), replay_options(&root, fixture))
        .await
        .unwrap();
    assert_eq!(session.run_id(), "sess-ordinal");

    // An unkeyed turn makes the agent mint the id: the ordinal continues
    // from the checkpoint sequence instead of restarting at t0 — the
    // envelope work's resume-safety guarantee.
    let result = session.send_unkeyed("continue").await.unwrap();
    assert_eq!(result.turn_id, "sess-ordinal-t3");
    assert_eq!(result.text, "resumed-reply");

    session.stop().await.unwrap();
}

#[tokio::test]
async fn resume_repairs_a_checkpoint_with_dangling_tool_calls() {
    let root = scratch();
    let home = root.join("session");
    std::fs::create_dir_all(home.join("checkpoints")).unwrap();

    // A crash mid-tool-call: the transcript ends on an unexecuted
    // assistant tool call, which providers reject.
    let messages = vec![
        agent_core::ChatMessage::system("sys"),
        agent_core::ChatMessage::user("inspect"),
        agent_core::ChatMessage::assistant(
            None,
            vec![agent_core::ToolCall::new(
                "call-1",
                "shell",
                json!({ "command": "pwd" }),
            )],
        ),
    ];
    let checkpoint = json!({
        "run_id": "sess-repair",
        "sequence": 1,
        "model": MODEL,
        "provider_url": "http://replay.invalid",
        "messages": serde_json::to_value(&messages).unwrap(),
        "trace_path": root.join("trace.jsonl"),
        "timestamp": "2026-01-01T00:00:00Z",
    });
    let checkpoint_path = home.join("checkpoints/session-latest.json");
    std::fs::write(
        &checkpoint_path,
        serde_json::to_vec_pretty(&checkpoint).unwrap(),
    )
    .unwrap();

    let fixture = root.join("replay.jsonl");
    SessionReplayFixture::new(MODEL)
        .turn("after-repair")
        .write(&fixture)
        .await
        .unwrap();

    let session = Session::resume(&agent(), replay_options(&root, fixture))
        .await
        .unwrap();

    // The repair happened on load: the dangling tool-call tail is gone
    // from the on-disk checkpoint, and the session takes turns normally.
    let on_disk: Value = serde_json::from_slice(&std::fs::read(&checkpoint_path).unwrap()).unwrap();
    assert_eq!(
        on_disk["messages"].as_array().map(Vec::len),
        Some(2),
        "trailing dangling tool call must be repaired away"
    );
    let result = session.send("go on").await.unwrap();
    assert_eq!(result.text, "after-repair");

    session.stop().await.unwrap();
}

#[tokio::test]
async fn native_tools_and_injected_providers_are_typed_unsupported() {
    let tool_agent = Agent::builder(MODEL)
        .tool(Tool::new(
            "get_weather",
            "weather",
            json!({"type": "object"}),
            |_args| async move { Ok(json!("sunny")) },
        ))
        .build()
        .unwrap();
    let err = Session::start(&tool_agent, SessionOptions::default())
        .await
        .unwrap_err();
    match err {
        SdkError::Unsupported(message) => {
            assert!(message.contains("get_weather"), "{message}");
            assert!(message.contains("wave 3+"), "{message}");
        }
        other => panic!("expected Unsupported, got: {other}"),
    }

    let provider_agent = Agent::builder(MODEL)
        .provider(Arc::new(agent_sdk::testing::ScriptedProvider::new()))
        .build()
        .unwrap();
    let err = Session::start(&provider_agent, SessionOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, SdkError::Unsupported(_)), "got: {err}");
}
