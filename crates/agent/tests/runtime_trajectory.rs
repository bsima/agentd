//! LIVE-LLM trajectory tests for agent runtime primitive use.
//!
//! These are opt-in integration tests: run them explicitly with
//! `cargo test -p agent --test runtime_trajectory -- --ignored` and a real API
//! key. They intentionally do **not** use ReplayOnlyProvider, `--model mock`, or
//! `--replay-trace`: replaying a canned trace would pre-decide the trajectory we
//! are trying to measure and make the test circular. Instead, each scenario runs
//! the `agent` binary against a live provider and asserts over the emitted trace
//! `Event` sequence, not over final prose.

use agent_core::{Event, TraceLogger};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use uuid::Uuid;

const DEFAULT_MODEL: &str = "gpt-5.5";
const RUNS_PER_SCENARIO: usize = 1;
const PASS_THRESHOLD: usize = 1;

#[test]
#[ignore = "live LLM trajectory eval; requires API key and opt-in --ignored"]
fn infer_fanout_uses_nested_infer() {
    let Some(config) = LiveConfig::from_env() else {
        return;
    };
    let scenario = Scenario {
        name: "infer-fanout",
        prompt: format!(
            "You have a tool named infer. Use it now to ask at least one focused \
             sub-question before answering. Decompose independently: estimate the \
             key risk of choosing Rust for a tiny CLI and the key risk of choosing \
             Python for the same CLI, then synthesize one sentence. When calling \
             infer, pass model exactly '{}'.",
            config.model
        ),
        predicate: has_nested_infer_call,
        runtime: Runtime::Ir,
    };
    assert_scenario_passes(&config, scenario, RUNS_PER_SCENARIO, PASS_THRESHOLD);
}

#[test]
#[ignore = "live LLM trajectory eval; requires API key and opt-in --ignored"]
fn state_path_put_then_get_same_key() {
    let Some(config) = LiveConfig::from_env() else {
        return;
    };
    let scenario = Scenario {
        name: "state-put-get",
        prompt: "Use the shell tool once for a tiny deterministic check: run `printf runtime_state_probe`. Then finish with a concise answer. This should exercise the agent's state path around tool execution.".to_owned(),
        predicate: has_put_followed_by_get_same_key,
        runtime: Runtime::Op,
    };
    assert_scenario_passes(&config, scenario, RUNS_PER_SCENARIO, PASS_THRESHOLD);
}

#[test]
#[ignore = "live LLM trajectory eval; requires API key and opt-in --ignored"]
fn trivial_prompt_does_not_over_decompose() {
    let Some(config) = LiveConfig::from_env() else {
        return;
    };
    let scenario = Scenario {
        name: "no-over-decompose",
        prompt: "Answer directly with exactly: ok".to_owned(),
        predicate: |events| !has_nested_infer_call(events),
        runtime: Runtime::Ir,
    };
    assert_scenario_passes(&config, scenario, RUNS_PER_SCENARIO, PASS_THRESHOLD);
}

#[test]
#[ignore = "live LLM trajectory eval; requires API key and opt-in --ignored"]
fn shell_prompt_uses_eval() {
    let Some(config) = LiveConfig::from_env() else {
        return;
    };
    let scenario = Scenario {
        name: "shell-use",
        prompt:
            "Use the shell tool to run `printf shell_probe_1088`, then report the exact output."
                .to_owned(),
        predicate: has_eval_call,
        runtime: Runtime::Ir,
    };
    assert_scenario_passes(&config, scenario, RUNS_PER_SCENARIO, PASS_THRESHOLD);
}

struct LiveConfig {
    model: String,
}

impl LiveConfig {
    fn from_env() -> Option<Self> {
        if std::env::var_os("AGENT_API_KEY").is_none()
            && std::env::var_os("ANTHROPIC_API_KEY").is_none()
            && std::env::var_os("OPENROUTER_API_KEY").is_none()
        {
            eprintln!(
                "skipping live trajectory test: set AGENT_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
            );
            return None;
        }
        let model = std::env::var("AGENT_EVAL_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        eprintln!("live trajectory model: {model}");
        Some(Self { model })
    }
}

#[derive(Clone, Copy)]
enum Runtime {
    Op,
    Ir,
}

impl Runtime {
    fn arg(self) -> &'static str {
        match self {
            Self::Op => "op",
            Self::Ir => "ir",
        }
    }
}

struct Scenario {
    name: &'static str,
    prompt: String,
    predicate: fn(&[Event]) -> bool,
    runtime: Runtime,
}

fn assert_scenario_passes(
    config: &LiveConfig,
    scenario: Scenario,
    runs: usize,
    pass_threshold: usize,
) {
    assert!(runs > 0, "scenario must run at least once");
    assert!(
        pass_threshold <= runs,
        "pass threshold must not exceed run count"
    );

    let mut passes = 0;
    let mut failures = Vec::new();
    for run_index in 0..runs {
        let run = run_agent(config, &scenario, run_index);
        if (scenario.predicate)(&run.events) {
            passes += 1;
        } else {
            failures.push(format!(
                "run_id={} trace={} ops={}",
                run.run_id,
                run.trace_path.display(),
                summarize_ops(&run.events)
            ));
        }
    }

    assert!(
        passes >= pass_threshold,
        "scenario '{}' passed {passes}/{runs}, below threshold {pass_threshold}; failures:\n{}",
        scenario.name,
        failures.join("\n")
    );
}

struct AgentRun {
    run_id: String,
    trace_path: PathBuf,
    events: Vec<Event>,
}

fn run_agent(config: &LiveConfig, scenario: &Scenario, run_index: usize) -> AgentRun {
    let home = std::env::temp_dir().join(format!(
        "agent-runtime-trajectory-{}-{}-{}",
        scenario.name,
        run_index,
        Uuid::new_v4()
    ));
    std::fs::create_dir_all(&home).expect("create HOME for live trajectory run");
    let run_id = format!("runtime-trajectory-{}-{}", scenario.name, Uuid::new_v4());
    let checkpoint_dir = home.join("checkpoints");

    let output = Command::new(env!("CARGO_BIN_EXE_agent"))
        .env("HOME", &home)
        .env("AGENT_RUN_ID", &run_id)
        .arg("--model")
        .arg(&config.model)
        .arg("--runtime")
        .arg(scenario.runtime.arg())
        .arg("--checkpoint-dir")
        .arg(&checkpoint_dir)
        .arg(&scenario.prompt)
        .output()
        .expect("run live agent");

    assert!(
        output.status.success(),
        "agent failed for scenario '{}' with status {:?}\nstdout:\n{}\nstderr:\n{}",
        scenario.name,
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let trace_path = trace_path(&home, &run_id);
    let events = read_trace_events(&trace_path);
    AgentRun {
        run_id,
        trace_path,
        events,
    }
}

fn trace_path(home: &Path, run_id: &str) -> PathBuf {
    home.join(".local/share/agent/traces")
        .join(format!("{run_id}.jsonl"))
}

fn read_trace_events(path: &Path) -> Vec<Event> {
    tokio::runtime::Runtime::new()
        .expect("create tokio runtime")
        .block_on(TraceLogger::read_events(path))
        .unwrap_or_else(|err| panic!("read trace {}: {err}", path.display()))
}

fn has_nested_infer_call(events: &[Event]) -> bool {
    let mut in_flight_infers = BTreeSet::new();
    for event in events {
        match event {
            Event::InferCall {
                op_id,
                parent_op_id,
                ..
            } => {
                if parent_op_id.is_some_and(|parent| in_flight_infers.contains(&parent)) {
                    return true;
                }
                in_flight_infers.insert(*op_id);
            }
            Event::InferResult { op_id, .. } => {
                in_flight_infers.remove(op_id);
            }
            _ => {}
        }
    }
    false
}

fn has_put_followed_by_get_same_key(events: &[Event]) -> bool {
    let mut put_keys = BTreeSet::new();
    for event in events {
        match event {
            Event::PutCall { key, .. } => {
                put_keys.insert(key.clone());
            }
            Event::GetCall { key, .. } if put_keys.contains(key) => return true,
            _ => {}
        }
    }
    false
}

fn has_eval_call(events: &[Event]) -> bool {
    events
        .iter()
        .any(|event| matches!(event, Event::EvalCall { .. }))
}

fn summarize_ops(events: &[Event]) -> String {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for event in events {
        let name = match event {
            Event::InferCall { .. } => "InferCall",
            Event::InferResult { .. } => "InferResult",
            Event::InferError { .. } => "InferError",
            Event::EvalCall { .. } => "EvalCall",
            Event::EvalResult { .. } => "EvalResult",
            Event::EvalError { .. } => "EvalError",
            Event::GetCall { .. } => "GetCall",
            Event::GetResult { .. } => "GetResult",
            Event::PutCall { .. } => "PutCall",
            Event::PutResult { .. } => "PutResult",
            Event::HydrationStart { .. } => "HydrationStart",
            Event::HydrationSection { .. } => "HydrationSection",
            Event::HydrationEnd { .. } => "HydrationEnd",
            Event::ParStart { .. } => "ParStart",
            Event::ParEnd { .. } => "ParEnd",
            Event::Checkpoint { .. } => "Checkpoint",
            Event::AgentDone { .. } => "AgentDone",
            Event::Custom { .. } => "Custom",
        };
        *counts.entry(name).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(name, count)| format!("{name}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}
