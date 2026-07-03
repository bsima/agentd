//! End-to-end SDK tests over a scripted provider (no credentials, no
//! network): the PRD MVP flows — one-shot text, typed tools with effect
//! identity, output contracts, replay, and the public event stream.

use agent_sdk::testing::ScriptedProvider;
use agent_sdk::{Agent, PublicStatus, Runner, SdkError, Tool, PUBLIC_SCHEMA_VERSION};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn trace_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("agent-sdk-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn weather_tool(calls: Arc<AtomicUsize>) -> Tool {
    Tool::new(
        "get_weather",
        "Current weather for a city.",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
        move |arguments| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let city = arguments["city"].as_str().unwrap_or("nowhere").to_owned();
                Ok(Value::String(format!("sunny in {city}")))
            }
        },
    )
}

fn weather_agent(provider: ScriptedProvider, calls: Arc<AtomicUsize>) -> Agent {
    Agent::builder("mock-model")
        .name("weather-bot")
        .instructions("You are a weather assistant.")
        .tool(weather_tool(calls))
        .provider(Arc::new(provider))
        .trace_dir(trace_dir())
        .build()
        .unwrap()
}

#[tokio::test]
async fn one_shot_run_returns_text() {
    let agent = Agent::builder("mock-model")
        .instructions("Be terse.")
        .provider(Arc::new(
            ScriptedProvider::new().text("hello from the mock"),
        ))
        .trace_dir(trace_dir())
        .build()
        .unwrap();

    let result = Runner::run(&agent, "say hello").await.unwrap();

    assert_eq!(result.text, "hello from the mock");
    assert!(result.output.is_none(), "no schema, no structured output");
    assert!(!result.run_id.is_empty());
    assert!(result.trace_path.exists(), "trace file written");
}

#[tokio::test]
async fn typed_tool_round_trips_through_dispatch_with_effect_identity() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = ScriptedProvider::new()
        .tool_call("get_weather", json!({ "city": "sf" }))
        .text("done: sunny in sf");
    let agent = weather_agent(provider, calls.clone());

    let result = Runner::run(&agent, "weather in sf?").await.unwrap();

    assert_eq!(result.text, "done: sunny in sf");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "handler invoked once");

    // The dispatch left a ToolCall with stable effect identity in the
    // runtime trace — and never touched the shell (no Eval events).
    let events = agent_core::TraceLogger::read_events(&result.trace_path)
        .await
        .unwrap();
    let effect = events
        .iter()
        .find_map(|event| match event {
            agent_core::Event::ToolCall {
                name,
                arguments,
                effect,
                ..
            } if name == "get_weather" => {
                assert_eq!(arguments, &json!({ "city": "sf" }));
                effect.clone()
            }
            _ => None,
        })
        .expect("ToolCall event with effect identity");
    assert_eq!(effect.kind, agent_core::EffectKind::Tool);
    assert!(effect.effect_id.0.starts_with("sha256:"));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, agent_core::Event::EvalCall { .. })),
        "a typed tool must never become a shell execution"
    );
}

#[tokio::test]
async fn output_contract_returns_validated_value() {
    let agent = Agent::builder("mock-model")
        .output_schema(json!({
            "type": "object",
            "required": ["answer"],
            "properties": { "answer": { "type": "integer" } }
        }))
        .provider(Arc::new(ScriptedProvider::new().text(r#"{"answer": 42}"#)))
        .trace_dir(trace_dir())
        .build()
        .unwrap();

    let result = Runner::run(&agent, "the answer?").await.unwrap();

    assert_eq!(result.output, Some(json!({ "answer": 42 })));
    assert_eq!(result.text, r#"{"answer": 42}"#);
}

#[tokio::test]
async fn output_contract_violation_is_a_typed_failure() {
    // The scripted model never produces conforming JSON; after the default
    // repair budget (2 repairs = 3 attempts) the run fails with the typed
    // contract error, not a panic and not non-conforming output.
    let provider = ScriptedProvider::new()
        .text("not json")
        .text("still not json")
        .text("never json");
    let agent = Agent::builder("mock-model")
        .output_schema(json!({
            "type": "object",
            "required": ["answer"],
            "properties": { "answer": { "type": "integer" } }
        }))
        .provider(Arc::new(provider))
        .trace_dir(trace_dir())
        .build()
        .unwrap();

    let err = Runner::run(&agent, "the answer?").await.unwrap_err();

    match err {
        SdkError::OutputContract {
            attempts,
            errors,
            content,
        } => {
            assert_eq!(attempts, 3);
            assert_eq!(content, "never json");
            assert!(
                errors.iter().any(|error| error.contains("not valid JSON")),
                "{errors:?}"
            );
        }
        other => panic!("expected OutputContract error, got: {other}"),
    }
}

#[tokio::test]
async fn replay_reproduces_the_result_without_provider_or_handler() {
    // Record a run with a live handler and scripted provider...
    let record_calls = Arc::new(AtomicUsize::new(0));
    let provider = ScriptedProvider::new()
        .tool_call("get_weather", json!({ "city": "sf" }))
        .text("done: sunny in sf");
    let agent = weather_agent(provider, record_calls.clone());
    let recorded = Runner::run(&agent, "weather in sf?").await.unwrap();
    assert_eq!(record_calls.load(Ordering::SeqCst), 1);

    // ...then replay it against an agent whose provider is EXHAUSTED (any
    // call would error) and whose handler must not run.
    let replay_calls = Arc::new(AtomicUsize::new(0));
    let replay_agent = weather_agent(ScriptedProvider::new(), replay_calls.clone());
    let replayed = Runner::replay(&recorded.trace_path, &replay_agent, "weather in sf?")
        .await
        .unwrap();

    assert_eq!(replayed.text, recorded.text);
    assert_eq!(
        replay_calls.load(Ordering::SeqCst),
        0,
        "replay must not invoke tool handlers"
    );
    assert_ne!(replayed.run_id, recorded.run_id, "replay is a fresh run");

    // Replaying with a changed tool set is a divergence, not a silent
    // partial replay: the tool set is part of the program identity.
    let different_agent = Agent::builder("mock-model")
        .instructions("You are a weather assistant.")
        .provider(Arc::new(ScriptedProvider::new()))
        .trace_dir(trace_dir())
        .build()
        .unwrap();
    let err = Runner::replay(&recorded.trace_path, &different_agent, "weather in sf?")
        .await
        .unwrap_err();
    assert!(matches!(err, SdkError::Replay(_)), "got: {err}");
}

#[tokio::test]
async fn event_stream_sees_the_public_dotted_events() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = ScriptedProvider::new()
        .tool_call("get_weather", json!({ "city": "sf" }))
        .text("done: sunny in sf");
    let agent = weather_agent(provider, calls);

    let mut handle = Runner::start(&agent, "weather in sf?").await.unwrap();
    assert!(!handle.run_id().is_empty());
    let mut stream = handle.events().expect("events taken once");
    assert!(handle.events().is_none(), "stream is take-once");

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        assert_eq!(event.schema_version, PUBLIC_SCHEMA_VERSION);
        assert_eq!(event.run_id, handle.run_id());
        events.push(event);
    }
    let result = handle.wait().await.unwrap();
    assert_eq!(result.text, "done: sunny in sf");

    let names: Vec<&str> = events.iter().map(|event| event.event.as_str()).collect();
    // Two model turns around one native tool dispatch, then run close.
    assert_eq!(
        names,
        vec![
            "infer.started",
            "infer.completed",
            "tool.requested",
            "tool.completed",
            "infer.started",
            "infer.completed",
            "run.completed",
        ],
        "full public sequence: {names:?}"
    );
    let requested = &events[2];
    assert_eq!(requested.status, PublicStatus::Started);
    assert_eq!(
        requested.attrs.get("name"),
        Some(&Value::String("get_weather".into()))
    );
    assert!(
        requested.effect.is_some(),
        "tool.requested carries effect identity"
    );
}

#[test]
fn builder_rejects_reserved_and_duplicate_tool_names() {
    let reserved = Agent::builder("mock-model")
        .tool(Tool::new("shell", "nope", json!({}), |_| async {
            Ok(Value::Null)
        }))
        .build();
    assert!(matches!(reserved, Err(SdkError::Config(message)) if message.contains("reserved")));

    let duplicate = Agent::builder("mock-model")
        .tool(Tool::new("lookup", "a", json!({}), |_| async {
            Ok(Value::Null)
        }))
        .tool(Tool::new("lookup", "b", json!({}), |_| async {
            Ok(Value::Null)
        }))
        .build();
    assert!(
        matches!(duplicate, Err(SdkError::Config(message)) if message.contains("already registered"))
    );
}
