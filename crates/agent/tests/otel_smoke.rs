use agent_core::{
    agent_loop_ir, effect_location, program_hash, BlockId, DynamicPath, EffectKind, EffectSite,
    Model,
};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    // IR replay keys on stable effect ids: compute the entry-Infer location
    // for each visit (the second Infer is the nudge retry after the non-stop
    // "thinking" turn) instead of hardcoding hashes.
    let machine = agent_loop_ir(Model("mock".into()), vec![], 16);
    let hash = program_hash(&machine.program).unwrap();
    let site = EffectSite {
        block: BlockId(0),
        instruction_index: 0,
    };
    let effect = |visit: u64| {
        serde_json::to_string(
            &effect_location(
                hash.clone(),
                EffectKind::Infer,
                site,
                DynamicPath::with_visit(site, visit),
            )
            .unwrap(),
        )
        .unwrap()
    };
    let (effect_0, effect_1) = (effect(0), effect(1));
    std::fs::write(
        &replay_path,
        format!(
            r#"{{"event":"Custom","run_id":"replay","op_id":0,"name":"ir_effect","data":{effect_0},"timestamp":"2026-06-02T00:00:00Z"}}
{{"event":"InferCall","run_id":"replay","op_id":1,"model":"mock","prompt_preview":"","timestamp":"2026-06-02T00:00:00Z"}}
{{"event":"InferResult","run_id":"replay","op_id":1,"response":{{"content":"thinking","tool_calls":[],"finish_reason":"length","input_tokens":0,"output_tokens":1,"total_tokens":1}},"response_preview":"thinking","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":1,"timestamp":"2026-06-02T00:00:00Z"}}
{{"event":"Custom","run_id":"replay","op_id":0,"name":"ir_effect","data":{effect_1},"timestamp":"2026-06-02T00:00:00Z"}}
{{"event":"InferCall","run_id":"replay","op_id":2,"model":"mock","prompt_preview":"","timestamp":"2026-06-02T00:00:00Z"}}
{{"event":"InferResult","run_id":"replay","op_id":2,"response":{{"content":"ok","tool_calls":[],"finish_reason":"stop","input_tokens":0,"output_tokens":1,"total_tokens":1}},"response_preview":"ok","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":1,"timestamp":"2026-06-02T00:00:00Z"}}
"#
        ),
    )
    .expect("write replay trace");
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
