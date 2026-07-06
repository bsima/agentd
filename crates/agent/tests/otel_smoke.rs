use agent_core::{
    agent_loop_ir, run_ir_sequential, ChatMessage, ChatProvider, EvalConfig, FinishReason, GcMode,
    GcTiming, Model, PassiveHydrationConfig, Response, SeqConfig, SourceRegistry, ToolCall,
    TraceLogger,
};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Serves the queued responses in order; the fixture recorder below drives
/// the real IR agent loop with it so the recorded trace carries the true
/// path-sensitive effect ids (the nudge retry after the non-stop "thinking"
/// turn re-visits the entry Infer along a non-root control path, which
/// cannot be hand-computed).
struct ScriptedProvider(Mutex<Vec<Response>>);

#[async_trait::async_trait]
impl ChatProvider for ScriptedProvider {
    async fn chat(
        &self,
        _model: &Model,
        _tools: &[agent_core::provider::ToolSpec],
        _messages: &[ChatMessage],
    ) -> anyhow::Result<Response> {
        self.0
            .lock()
            .unwrap()
            .pop()
            .ok_or_else(|| anyhow::anyhow!("scripted provider exhausted"))
    }
}

fn response(content: &str, finish_reason: FinishReason) -> Response {
    Response {
        content: content.into(),
        tool_calls: Vec::<ToolCall>::new(),
        finish_reason: Some(finish_reason),
        input_tokens: 0,
        output_tokens: 1,
        total_tokens: 1,
        cached_input_tokens: None,
        cost_micro_usd: None,
        pricing: None,
        metadata: Default::default(),
    }
}

/// Record a replay fixture by running the built-in agent loop in-process
/// against a scripted provider. IR replay keys on stable effect ids that
/// encode the dynamic control path, so recording is the way to produce a
/// correct fixture — hardcoding ids (or hand-simulating the loop's block
/// transitions) breaks whenever the program changes.
fn record_replay_fixture(path: &std::path::Path, responses: Vec<Response>) {
    let mut responses = responses;
    responses.reverse();
    let config = SeqConfig {
        tools: Default::default(),
        provider: Arc::new(ScriptedProvider(Mutex::new(responses))),
        hydration: SourceRegistry::new(),
        passive_hydration: PassiveHydrationConfig::default(),
        trace: TraceLogger::new("replay", path.to_path_buf()),
        eval: EvalConfig::default(),
        replay: None,
        trace_full_prompt_ir: false,
        trace_full_payloads: false,
        gc: GcMode::None,
        gc_threshold: 0.85,
        gc_log: false,
        gc_timing: GcTiming::Threshold,
        context_budget: 200_000,
        pricing: Default::default(),
    };
    let machine = agent_loop_ir(Model("mock".into()), vec![ChatMessage::user("hello")], 16);
    tokio::runtime::Runtime::new()
        .expect("create tokio runtime")
        .block_on(run_ir_sequential(&config, machine))
        .expect("record replay fixture");
}

#[test]
fn otel_endpoint_smoke_preserves_replay_and_jsonl_trace() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind OTLP mock");
    listener
        .set_nonblocking(true)
        .expect("set OTLP mock nonblocking");
    let addr = listener.local_addr().expect("OTLP mock local addr");
    let stop = Arc::new(AtomicBool::new(false));
    let request_count = Arc::new(AtomicUsize::new(0));
    let server_stop = stop.clone();
    let server_count = request_count.clone();
    let server = std::thread::spawn(move || {
        let started = Instant::now();
        while !server_stop.load(Ordering::SeqCst) && started.elapsed() < Duration::from_secs(10) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    server_count.fetch_add(1, Ordering::SeqCst);
                    let mut buf = [0_u8; 8192];
                    let _ = stream.read(&mut buf);
                    let _ = stream.write_all(
                        b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    );
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        }
    });

    let dir = std::env::temp_dir().join(format!("agent-otel-smoke-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let replay_path = dir.join("replay.jsonl");
    // The first response is non-stop ("thinking", finish_reason length), so
    // the loop nudges and re-Infers; the second ends the turn.
    record_replay_fixture(
        &replay_path,
        vec![
            response("thinking", FinishReason::Length),
            response("ok", FinishReason::Stop),
        ],
    );
    let run_id = format!("otel-smoke-{}", uuid::Uuid::new_v4());

    let output = Command::new(env!("CARGO_BIN_EXE_agent"))
        .env("HOME", &dir)
        .env("AGENT_RUN_ID", &run_id)
        .arg("--model")
        .arg("mock")
        .arg("--replay-trace")
        .arg(&replay_path)
        .arg("--otel-endpoint")
        .arg(format!("http://{addr}"))
        .arg("hello")
        .output()
        .expect("run agent");

    stop.store(true, Ordering::SeqCst);
    let _ = server.join();

    assert!(
        output.status.success(),
        "status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "ok");
    let trace_path = dir
        .join(".local/share/agent/traces")
        .join(format!("{run_id}.jsonl"));
    let trace = std::fs::read_to_string(&trace_path).expect("read JSONL trace");
    assert!(trace.contains("\"event\":\"InferCall\""));
    assert!(
        request_count.load(Ordering::SeqCst) > 0,
        "mock OTLP endpoint did not receive any requests"
    );
}
