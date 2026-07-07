//! Supervisor integration tests: drive the real `agentd` and `agent`
//! binaries, credential-free. Instant turns ride replay-trace fixtures
//! (`agent-sdk`'s `SessionReplayFixture`, the same recipe as
//! `evals/session.sh`); tests that need slow turns or real checkpoint
//! writes use a local OpenAI-compatible stub server (replay sessions
//! deliberately do not checkpoint).

use agent_sdk::testing::SessionReplayFixture;
use serde_json::{json, Value};
use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

const MODEL: &str = "test-model";

fn agentd_bin() -> &'static str {
    env!("CARGO_BIN_EXE_agentd")
}

/// Locate the workspace-built `agent` binary, building it once if this
/// test target was compiled without it (e.g. `cargo test -p agentd`).
fn agent_bin() -> PathBuf {
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
            let status = Command::new(env!("CARGO"))
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

/// A fresh scratch root: HOME (traces, approvals, config lookups) and
/// AGENTD_HOME both live here, so nothing touches the real environment.
struct Scratch {
    root: PathBuf,
    /// Session names to stop on drop so failed tests do not leak agents.
    sessions: Mutex<Vec<String>>,
}

impl Scratch {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("agentd-it-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        Self {
            root,
            sessions: Mutex::new(Vec::new()),
        }
    }

    fn agentd_home(&self) -> PathBuf {
        self.root.join("agentd")
    }

    /// An `agentd` invocation with a hermetic environment. Extra
    /// (env, value) pairs layer on top (e.g. the stub provider).
    fn agentd(&self, args: &[&str], env: &[(&str, &str)]) -> Command {
        let mut cmd = Command::new(agentd_bin());
        cmd.args(args)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", &self.root)
            .env("XDG_CONFIG_HOME", self.root.join("config"))
            .env("AGENTD_HOME", self.agentd_home())
            .env("AGENTD_AGENT_BIN", agent_bin());
        for (key, value) in env {
            cmd.env(key, value);
        }
        cmd
    }

    fn run(&self, args: &[&str], env: &[(&str, &str)]) -> Output {
        if args.first() == Some(&"start") || args.first() == Some(&"resume") {
            if let Some(name) = args.get(1) {
                self.sessions.lock().unwrap().push(name.to_string());
            }
        }
        self.agentd(args, env).output().expect("running agentd")
    }

    fn run_ok(&self, args: &[&str], env: &[(&str, &str)]) -> String {
        let output = self.run(args, env);
        assert!(
            output.status.success(),
            "agentd {args:?} failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).to_string()
    }

    fn status_json(&self, name: &str) -> Value {
        let out = self.run_ok(&["status", name, "--json"], &[]);
        let statuses: Value = serde_json::from_str(&out).expect("status --json parses");
        statuses[0].clone()
    }

    fn session_file(&self, name: &str, file: &str) -> PathBuf {
        self.agentd_home().join(name).join(file)
    }

    fn write_replay_fixture(&self, turns: &[&str]) -> PathBuf {
        let mut fixture = SessionReplayFixture::new(MODEL);
        for turn in turns {
            fixture = fixture.turn(*turn);
        }
        let mut lines = String::new();
        for event in fixture.events().expect("fixture events") {
            lines.push_str(&serde_json::to_string(&event).unwrap());
            lines.push('\n');
        }
        let path = self.root.join("replay.jsonl");
        std::fs::write(&path, lines).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for name in self.sessions.lock().unwrap().iter() {
            let _ = self
                .agentd(&["stop", name, "--grace", "1"], &[])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

// ---------------------------------------------------------------------------
// Minimal OpenAI-compatible stub provider (sync twin of the one in
// agent-sdk's session tests): serves scripted (delay, text) responses in
// order and records every request body.
// ---------------------------------------------------------------------------

struct Stub {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
}

fn stub_provider(responses: Vec<(Duration, &str)>) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    let responses: Vec<(Duration, String)> = responses
        .into_iter()
        .map(|(delay, text)| (delay, text.to_string()))
        .collect();
    std::thread::spawn(move || {
        let mut queue = responses.into_iter();
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let Ok(n) = socket.read(&mut chunk) else {
                    break None;
                };
                if n == 0 {
                    break None;
                }
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break Some(pos + 4);
                }
            };
            let Some(header_end) = header_end else {
                continue;
            };
            let content_length: usize = String::from_utf8_lossy(&buf[..header_end])
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().ok())?
                })
                .unwrap_or(0);
            while buf.len() < header_end + content_length {
                let Ok(n) = socket.read(&mut chunk) else {
                    break;
                };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            if let Ok(body) = serde_json::from_slice::<Value>(&buf[header_end..]) {
                seen.lock().unwrap().push(body);
            }
            let (delay, text) = queue
                .next()
                .unwrap_or((Duration::ZERO, "stub exhausted".into()));
            std::thread::sleep(delay);
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
            let _ = socket.write_all(response.as_bytes());
        }
    });
    Stub { url, requests }
}

fn stub_env(stub: &Stub) -> Vec<(&'static str, String)> {
    vec![
        ("AGENT_PROVIDER", stub.url.clone()),
        ("AGENT_API_KEY", "stub-key".to_string()),
    ]
}

fn as_env<'a>(pairs: &'a [(&'static str, String)]) -> Vec<(&'static str, &'a str)> {
    pairs.iter().map(|(k, v)| (*k, v.as_str())).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// start → send (turn-id correlated) → status → errored turn → stop.
#[test]
fn start_send_status_stop_roundtrip() {
    let scratch = Scratch::new();
    let fixture = scratch.write_replay_fixture(&["alpha-reply", "beta-reply"]);
    let fixture = fixture.display().to_string();

    scratch.run_ok(
        &[
            "start",
            "s1",
            "--model",
            MODEL,
            "--",
            "--replay-trace",
            &fixture,
        ],
        &[],
    );
    let status = scratch.status_json("s1");
    assert_eq!(status["running"], json!(true));
    assert_eq!(status["model"], json!(MODEL));
    assert_eq!(status["pending_approvals"], json!(0));
    assert!(status["pid"].as_i64().is_some());

    // Sequential sends with supplied turn ids get their own responses.
    let first = scratch.run_ok(
        &[
            "send",
            "s1",
            "first prompt",
            "--turn-id",
            "turn-a",
            "--timeout",
            "30",
        ],
        &[],
    );
    assert_eq!(first.trim(), "alpha-reply");
    let second = scratch.run_ok(
        &[
            "send",
            "s1",
            "second prompt",
            "--turn-id",
            "turn-b",
            "--timeout",
            "30",
        ],
        &[],
    );
    assert_eq!(second.trim(), "beta-reply");

    // A third turn has no recorded replay result: the agent emits
    // agent_error and send exits 1 (distinct from the 124 timeout).
    let errored = scratch.run(&["send", "s1", "third", "--timeout", "30"], &[]);
    assert_eq!(errored.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&errored.stderr).contains("failed"),
        "stderr: {}",
        String::from_utf8_lossy(&errored.stderr)
    );

    // logs --raw shows the machine events.
    let raw = scratch.run_ok(&["logs", "s1", "--raw", "-n", "200"], &[]);
    assert!(raw.contains("agent_complete"), "raw logs: {raw}");

    let stop_out = scratch.run_ok(&["stop", "s1"], &[]);
    assert!(stop_out.contains("stopped 's1'"));
    let status = scratch.status_json("s1");
    assert_eq!(status["running"], json!(false));
    assert!(status["pid"].is_null());

    // Stopping again is a no-op, not an error.
    let again = scratch.run_ok(&["stop", "s1"], &[]);
    assert!(again.contains("not running"));
}

/// Two overlapping senders: the first turn is slow, the second sender's
/// tail must skip the first sender's completion (id filter) and both must
/// get exactly their own response.
#[test]
fn concurrent_sends_correlate_by_turn_id() {
    let scratch = Scratch::new();
    let stub = stub_provider(vec![
        (Duration::from_millis(1200), "slow-reply"),
        (Duration::ZERO, "fast-reply"),
    ]);
    let env_pairs = stub_env(&stub);
    scratch.run_ok(&["start", "s2", "--model", MODEL], &as_env(&env_pairs));

    let slow = scratch
        .agentd(
            &[
                "send",
                "s2",
                "slow question",
                "--turn-id",
                "id-slow",
                "--timeout",
                "30",
            ],
            &as_env(&env_pairs),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Let the slow frame land first, then queue the second send while the
    // first turn is still inflight.
    std::thread::sleep(Duration::from_millis(400));
    let fast = scratch.run_ok(
        &[
            "send",
            "s2",
            "fast question",
            "--turn-id",
            "id-fast",
            "--timeout",
            "30",
        ],
        &as_env(&env_pairs),
    );
    let slow = slow.wait_with_output().unwrap();
    assert!(
        slow.status.success(),
        "slow send failed: {}",
        String::from_utf8_lossy(&slow.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&slow.stdout).trim(), "slow-reply");
    assert_eq!(fast.trim(), "fast-reply");

    scratch.run_ok(&["stop", "s2"], &[]);
}

/// `send --timeout` times out the CALLER only (exit 124, re-attach hint);
/// the turn keeps running and `agentd attach` retrieves its result.
#[test]
fn send_timeout_leaves_turn_running_and_attach_recovers() {
    let scratch = Scratch::new();
    let stub = stub_provider(vec![
        (Duration::from_millis(2500), "slow-reply"),
        (Duration::ZERO, "next-reply"),
    ]);
    let env_pairs = stub_env(&stub);
    scratch.run_ok(&["start", "s3", "--model", MODEL], &as_env(&env_pairs));

    let timed_out = scratch.run(
        &[
            "send",
            "s3",
            "slow question",
            "--turn-id",
            "slow-turn",
            "--timeout",
            "1",
        ],
        &as_env(&env_pairs),
    );
    assert_eq!(
        timed_out.status.code(),
        Some(124),
        "timeout must exit 124, got {:?}\nstderr: {}",
        timed_out.status.code(),
        String::from_utf8_lossy(&timed_out.stderr)
    );
    let stderr = String::from_utf8_lossy(&timed_out.stderr);
    assert!(
        stderr.contains("agentd attach s3 slow-turn"),
        "stderr must say how to re-attach: {stderr}"
    );

    // No default kill: the session survived the caller's timeout.
    assert_eq!(scratch.status_json("s3")["running"], json!(true));

    // Attach waits out the turn and prints its response.
    let recovered = scratch.run_ok(&["attach", "s3", "slow-turn", "--timeout", "30"], &[]);
    assert_eq!(recovered.trim(), "slow-reply");

    // Attach again: the result is already on disk, no waiting.
    let replayed = scratch.run_ok(&["attach", "s3", "slow-turn", "--timeout", "5"], &[]);
    assert_eq!(replayed.trim(), "slow-reply");

    // The session still takes turns.
    let next = scratch.run_ok(
        &["send", "s3", "next question", "--timeout", "30"],
        &as_env(&env_pairs),
    );
    assert_eq!(next.trim(), "next-reply");

    scratch.run_ok(&["stop", "s3"], &[]);
}

/// kill -9 then `agentd resume`: same run id, and the next provider call
/// carries the full pre-crash conversation — history intact.
#[test]
fn kill_dash_nine_then_resume_keeps_history_intact() {
    let scratch = Scratch::new();
    let stub = stub_provider(vec![
        (Duration::ZERO, "first-reply"),
        (Duration::ZERO, "second-reply"),
    ]);
    let env_pairs = stub_env(&stub);
    scratch.run_ok(&["start", "s4", "--model", MODEL], &as_env(&env_pairs));

    let first = scratch.run_ok(
        &[
            "send",
            "s4",
            "remember the word pineapple",
            "--timeout",
            "30",
        ],
        &as_env(&env_pairs),
    );
    assert_eq!(first.trim(), "first-reply");
    assert!(
        scratch
            .session_file("s4", "checkpoints/session-latest.json")
            .is_file(),
        "turn completion implies a checkpoint"
    );

    let status = scratch.status_json("s4");
    let run_id = status["run_id"].as_str().unwrap().to_string();
    let pid = status["pid"].as_i64().unwrap();
    let killed = Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    // Wait for the process to be reaped.
    for _ in 0..100 {
        if scratch.status_json("s4")["running"] == json!(false) {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(scratch.status_json("s4")["running"], json!(false));

    let resumed = scratch.run_ok(&["resume", "s4"], &as_env(&env_pairs));
    assert!(
        resumed.contains(&run_id),
        "resume keeps the run id: {resumed}"
    );
    let second = scratch.run_ok(
        &["send", "s4", "what was the word?", "--timeout", "30"],
        &as_env(&env_pairs),
    );
    assert_eq!(second.trim(), "second-reply");
    assert_eq!(
        scratch.status_json("s4")["run_id"].as_str(),
        Some(run_id.as_str())
    );

    let requests = stub.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
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

    scratch.run_ok(&["stop", "s4"], &[]);
}

/// The t-1105 acceptance path: the spec file is canonical. `set-model`
/// edits agent.md in place, `resume` reads it fresh, and the next turn's
/// agent_start machine event proves the new model is live. Start flags on
/// an existing spec are refused (they could drift from the file).
#[test]
fn set_model_then_resume_shows_new_model_in_agent_start() {
    let scratch = Scratch::new();
    let stub = stub_provider(vec![
        (Duration::ZERO, "old-model-reply"),
        (Duration::ZERO, "new-model-reply"),
    ]);
    let env_pairs = stub_env(&stub);
    scratch.run_ok(&["start", "s5", "--model", MODEL], &as_env(&env_pairs));
    scratch.run_ok(
        &["send", "s5", "hello", "--timeout", "30"],
        &as_env(&env_pairs),
    );
    scratch.run_ok(&["stop", "s5"], &[]);

    // Config flags cannot bypass the canonical spec.
    let refused = scratch.run(&["start", "s5", "--model", "other-model"], &[]);
    assert!(!refused.status.success());
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(stderr.contains("canonical"), "stderr: {stderr}");
    assert!(stderr.contains("set-model"), "stderr: {stderr}");

    let set_out = scratch.run_ok(&["set-model", "s5", "other-model"], &[]);
    assert!(set_out.contains("set model = other-model"), "{set_out}");
    let spec = std::fs::read_to_string(scratch.session_file("s5", "agent.md")).unwrap();
    assert!(spec.contains("model: other-model"), "spec: {spec}");

    scratch.run_ok(&["resume", "s5"], &as_env(&env_pairs));
    let reply = scratch.run_ok(
        &["send", "s5", "hello again", "--timeout", "30"],
        &as_env(&env_pairs),
    );
    assert_eq!(reply.trim(), "new-model-reply");

    // The startup banner of record: agent_start events carry the live
    // model. The pre-edit turn ran the old model, the post-resume turn the
    // new one.
    let stdout = std::fs::read_to_string(scratch.session_file("s5", "stdout.jsonl")).unwrap();
    let models: Vec<String> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["custom_type"] == json!("agent_start"))
        .filter_map(|v| v["data"]["config"]["model"].as_str().map(str::to_string))
        .collect();
    assert_eq!(models.first().map(String::as_str), Some(MODEL));
    assert_eq!(models.last().map(String::as_str), Some("other-model"));

    scratch.run_ok(&["stop", "s5"], &[]);
}

/// gen-systemd: GENERATED header, restart policy, and lifecycle through
/// `agentd resume`/`agentd stop` (golden checks; `systemd-analyze verify`
/// runs in evals/agentd-persistent.sh when available).
#[test]
fn gen_systemd_emits_a_generated_forking_unit() {
    let scratch = Scratch::new();
    // A session exists if its directory exists; no process needed.
    std::fs::create_dir_all(scratch.agentd_home().join("s6")).unwrap();
    std::fs::write(
        scratch.session_file("s6", "agent.md"),
        "---\nmodel: sonnet\n---\n",
    )
    .unwrap();

    let unit = scratch.run_ok(&["gen-systemd", "s6"], &[]);
    assert!(unit.starts_with("# GENERATED"), "unit: {unit}");
    assert!(unit.contains("edits are overwritten by agentd"));
    assert!(unit.contains("Type=forking"));
    assert!(unit.contains("Restart=on-failure"));
    assert!(unit.contains(&format!(
        "PIDFile={}",
        scratch.session_file("s6", "pid").display()
    )));
    assert!(unit.contains("resume s6"), "ExecStart goes through resume");
    assert!(
        unit.contains("stop s6"),
        "ExecStop goes through agentd stop"
    );
    assert!(unit.contains(&format!(
        "Environment=AGENTD_HOME={}",
        scratch.agentd_home().display()
    )));

    // --output writes the same content to a file.
    let path = scratch.root.join("agentd-s6.service");
    scratch.run_ok(
        &["gen-systemd", "s6", "--output", &path.display().to_string()],
        &[],
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), unit);
}
