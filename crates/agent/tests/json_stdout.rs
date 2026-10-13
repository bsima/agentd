use agent_core::{
    agent_loop_ir, effect_location, program_hash, BlockId, DynamicPath, EffectKind, EffectSite,
    Model,
};
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

    // Compute the stable effect id for the entry-block Infer of the IR agent
    // loop, instead of hardcoding hashes that break whenever the program
    // changes. This mirrors the `effect` field the IR interpreter attaches
    // to each InferCall event.
    let machine = agent_loop_ir(Model("test-model".into()), vec![], 16);
    let hash = program_hash(&machine.program).unwrap();
    let site = EffectSite {
        block: BlockId(0),
        instruction_index: 0,
    };
    let location =
        effect_location(hash, EffectKind::Infer, site, DynamicPath::at_entry(0)).unwrap();
    let ir_effect = serde_json::to_string(&location).unwrap();

    let replay = format!(
        r#"{{"event":"InferCall","run_id":"replay","op_id":2,"model":"test-model","prompt_preview":"","effect":{ir_effect},"timestamp":"{timestamp}"}}
{{"event":"InferResult","run_id":"replay","op_id":2,"response":{{"content":"hello human","tool_calls":[],"finish_reason":"stop","input_tokens":3,"output_tokens":4,"total_tokens":7}},"response_preview":"hello human","input_tokens":3,"output_tokens":4,"total_tokens":7,"duration_ms":1,"timestamp":"{timestamp}"}}
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

/// Minimal OpenAI-compatible (non-streaming) provider: the first three
/// requests answer with a `shell` call whose output is large enough to push
/// a 2000-token window past the GC threshold; the fourth answers "done".
fn spawn_gc_pressure_provider() -> u16 {
    use std::io::{BufRead, BufReader, Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind mock provider");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for turn in 0.. {
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
                let line = line.trim_end().to_ascii_lowercase();
                if let Some(value) = line.strip_prefix("content-length:") {
                    content_length = value.trim().parse().unwrap_or(0);
                }
                if line.is_empty() {
                    break;
                }
            }
            let mut request = vec![0u8; content_length];
            let _ = reader.read_exact(&mut request);
            let (message, finish) = if turn < 3 {
                (
                    serde_json::json!({"role": "assistant", "content": null, "tool_calls": [{
                        "id": format!("call_{turn}"), "type": "function",
                        "function": {"name": "shell", "arguments": "{\"command\":\"seq 1 600\"}"}
                    }]}),
                    "tool_calls",
                )
            } else {
                (
                    serde_json::json!({"role": "assistant", "content": "done"}),
                    "stop",
                )
            };
            let body = serde_json::json!({
                "id": format!("r{turn}"), "object": "chat.completion", "model": "tiny",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes());
        }
    });
    port
}

/// Runs the omni-style one-shot `agent --model=tiny --json [extra] -- PROMPT`
/// against a GC-pressure provider; returns (stdout, stderr, trace file).
fn run_one_shot_under_gc_pressure(extra: &[&str]) -> (String, String, String) {
    let root = std::env::temp_dir().join(format!("agent-json-gc-{}", Uuid::new_v4()));
    let config_dir = root.join("config/agent");
    std::fs::create_dir_all(&config_dir).unwrap();
    let port = spawn_gc_pressure_provider();
    std::fs::write(
        config_dir.join("models.yaml"),
        format!(
            "default_model: tiny\nmodels:\n  - name: tiny\n    provider: openai-compatible\n    api_id: tiny\n    base_url: http://127.0.0.1:{port}/v1\n    api_key: test-key\n    context: 2000\n"
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_agent"))
        .arg("--model=tiny")
        .arg("--json")
        .args(extra)
        .arg("--")
        .arg("list numbers")
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env_remove("AGENT_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        output.status.success(),
        "agent failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // The trace path reaches the consumer twice: on stderr ("trace: ...")
    // and in the agent_start machine event on stdout.
    let start = stdout
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .find(|event| event["custom_type"] == "agent_start")
        .expect("agent_start on stdout");
    let trace_path = start["data"]["config"]["trace_path"]
        .as_str()
        .expect("agent_start carries trace_path")
        .to_owned();
    assert!(trace_path.starts_with(root.join(".local/share/agent/traces").to_str().unwrap()));
    assert!(stderr.contains(&format!("trace: {trace_path}")), "{stderr}");
    let trace = std::fs::read_to_string(&trace_path).unwrap();
    (stdout, stderr, trace)
}

fn count_custom(jsonl: &str, name: &str) -> usize {
    jsonl
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|event| event["event"] == "Custom" && event["name"] == name)
        .count()
}

/// t-1608: GC telemetry is on by default. In one-shot `--json` mode the
/// gc_collect events land in the trace file AND on stdout (the --json
/// stream mirrors every trace event), never on stderr.
#[test]
fn json_one_shot_streams_gc_collect_by_default() {
    let (stdout, stderr, trace) = run_one_shot_under_gc_pressure(&[]);
    let on_stdout = count_custom(&stdout, "gc_collect");
    assert!(on_stdout > 0, "no gc_collect on stdout:\n{stdout}");
    assert_eq!(on_stdout, count_custom(&trace, "gc_collect"));
    assert!(!stderr.contains("gc_collect"), "{stderr}");
}

/// `--gc-log=false` is the explicit opt-out.
#[test]
fn json_one_shot_gc_log_false_suppresses_gc_events() {
    let (stdout, _, trace) = run_one_shot_under_gc_pressure(&["--gc-log=false"]);
    assert_eq!(count_custom(&stdout, "gc_collect"), 0, "{stdout}");
    assert_eq!(count_custom(&trace, "gc_collect"), 0, "{trace}");
}
