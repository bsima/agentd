use agent_core::{
    estimate_tokens, truncate_oversized_message, ChatMessage, ContextGc, GcState, MarkSweepGc,
    RingGc,
};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const EVAL_BUDGET_FRACTION: f64 = 0.75;

#[derive(Debug, Clone)]
struct TraceCase {
    name: String,
    prompt: Vec<ChatMessage>,
}

#[derive(Debug, Clone)]
struct EvalMetrics {
    strategy: &'static str,
    trace: String,
    budget: usize,
    tokens_before: usize,
    tokens_after: usize,
    token_reduction_pct: f64,
    messages_before: usize,
    messages_after: usize,
    tool_results_before: usize,
    tool_results_after: usize,
}

#[test]
fn gc_strategy_evals() -> Result<()> {
    let traces = load_trace_cases()?;
    assert!(!traces.is_empty(), "expected at least one eval trace");

    let mut ring_metrics = Vec::new();
    let mut challenger_metrics = Vec::new();

    for trace in &traces {
        let budget = eval_budget(&trace.prompt);
        let ring = evaluate_strategy(
            trace,
            budget,
            RingGc {
                preserve_prefix: false,
            },
        )?;
        print_metrics(&ring);
        ring_metrics.push(ring);

        let mark_sweep = evaluate_strategy(
            trace,
            budget,
            MarkSweepGc {
                preserve_prefix: false,
            },
        )?;
        print_metrics(&mark_sweep);
        challenger_metrics.push(mark_sweep);
    }

    assert_improves_over_ring(&ring_metrics, &challenger_metrics);
    Ok(())
}

fn load_trace_cases() -> Result<Vec<TraceCase>> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("could not resolve repo root"))?;
    let eval_dir = repo_root.join("evals/gc");
    let mut paths = fs::read_dir(&eval_dir)
        .with_context(|| format!("reading {}", eval_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<PathBuf>>>()?;
    paths.sort();

    let mut cases = Vec::new();
    for path in paths
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
    {
        let content = fs::read_to_string(&path)
            .with_context(|| format!("reading trace fixture {}", path.display()))?;
        for (line_idx, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line)
                .with_context(|| format!("decoding {} line {}", path.display(), line_idx + 1))?;
            if value.get("event").and_then(Value::as_str) != Some("InferCall") {
                continue;
            }
            let Some(prompt) = value.get("prompt") else {
                continue;
            };
            let prompt: Vec<ChatMessage> =
                serde_json::from_value(prompt.clone()).with_context(|| {
                    format!(
                        "decoding prompt in {} line {}",
                        path.display(),
                        line_idx + 1
                    )
                })?;
            if prompt_has_tool_chain(&prompt) {
                cases.push(TraceCase {
                    name: path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    prompt,
                });
            }
        }
    }
    Ok(cases)
}

fn prompt_has_tool_chain(prompt: &[ChatMessage]) -> bool {
    let tool_results = prompt
        .iter()
        .filter(|message| message.role == "tool")
        .count();
    let tool_calls = prompt
        .iter()
        .map(|message| message.tool_calls.as_ref().map_or(0, Vec::len))
        .sum::<usize>();
    tool_results >= 3 && tool_calls >= 3
}

fn eval_budget(prompt: &[ChatMessage]) -> usize {
    let before = estimate_tokens(prompt);
    ((before as f64) * EVAL_BUDGET_FRACTION).floor() as usize
}

fn evaluate_strategy<G>(trace: &TraceCase, budget: usize, gc: G) -> Result<EvalMetrics>
where
    G: ContextGc + Copy,
{
    let mut input = trace.prompt.clone();
    let tokens_before = estimate_tokens(&input);
    let messages_before = input.len();
    let tool_results_before = count_tool_results(&input);
    truncate_oversized_message(&mut input, budget);

    let mut state_a = GcState::default();
    let collected = gc.collect(input.clone(), budget, &mut state_a);
    let mut state_b = GcState::default();
    let collected_again = gc.collect(input, budget, &mut state_b);

    assert_eq!(
        collected,
        collected_again,
        "{} on {} must be deterministic across two runs",
        gc.name(),
        trace.name
    );
    assert_structurally_valid(&trace.prompt, &collected, gc.name(), &trace.name, budget);

    let tokens_after = estimate_tokens(&collected);
    Ok(EvalMetrics {
        strategy: gc.name(),
        trace: trace.name.clone(),
        budget,
        tokens_before,
        tokens_after,
        token_reduction_pct: reduction_pct(tokens_before, tokens_after),
        messages_before,
        messages_after: collected.len(),
        tool_results_before,
        tool_results_after: count_tool_results(&collected),
    })
}

fn count_tool_results(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .filter(|message| message.role == "tool")
        .count()
}

fn reduction_pct(before: usize, after: usize) -> f64 {
    if before == 0 {
        0.0
    } else {
        ((before.saturating_sub(after) as f64) / (before as f64)) * 100.0
    }
}

fn assert_structurally_valid(
    original: &[ChatMessage],
    collected: &[ChatMessage],
    strategy: &str,
    trace: &str,
    budget: usize,
) {
    let tokens_after = estimate_tokens(collected);
    assert!(
        tokens_after <= budget,
        "{strategy} on {trace} must converge under budget in one pass: {tokens_after} > {budget}"
    );

    let original_system = original
        .iter()
        .filter(|message| message.role == "system")
        .collect::<Vec<_>>();
    for system in original_system {
        assert!(
            collected
                .iter()
                .any(|message| message.id == system.id && message.role == "system"),
            "{strategy} on {trace} dropped pinned/system message {}",
            system.id
        );
    }

    let live_call_ids = collected
        .iter()
        .flat_map(|message| message.tool_calls.as_deref().unwrap_or_default())
        .map(|call| call.id.as_str())
        .collect::<BTreeSet<_>>();
    let live_result_ids = collected
        .iter()
        .filter_map(|message| message.tool_call_id.as_deref())
        .collect::<BTreeSet<_>>();

    assert_eq!(
        live_call_ids, live_result_ids,
        "{strategy} on {trace} must preserve tool-call/result pair atomicity"
    );
}

fn assert_improves_over_ring(ring: &[EvalMetrics], challengers: &[EvalMetrics]) {
    assert_eq!(
        ring.len(),
        challengers.len(),
        "expected one challenger result per ring baseline"
    );

    for (baseline, challenger) in ring.iter().zip(challengers) {
        assert_eq!(baseline.trace, challenger.trace);
        assert!(
            challenger.messages_after > baseline.messages_after
                || challenger.tool_results_after > baseline.tool_results_after,
            "{} on {} must retain more coherent structure than RingGc; ring kept {} msgs/{} tool results, challenger kept {} msgs/{} tool results",
            challenger.strategy,
            challenger.trace,
            baseline.messages_after,
            baseline.tool_results_after,
            challenger.messages_after,
            challenger.tool_results_after
        );
    }
}

fn print_metrics(metrics: &EvalMetrics) {
    println!(
        "gc_eval trace={} strategy={} budget={} tokens_before={} tokens_after={} reduction={:.1}% messages={}/{} tool_results={}/{}",
        metrics.trace,
        metrics.strategy,
        metrics.budget,
        metrics.tokens_before,
        metrics.tokens_after,
        metrics.token_reduction_pct,
        metrics.messages_after,
        metrics.messages_before,
        metrics.tool_results_after,
        metrics.tool_results_before
    );
}

/// Compare --gc-cache preserve against ignore on the fixture set: preserve
/// must keep a stable leading prefix (provider prompt caches key on it) at
/// least as long as ignore's, without invalidating it, while still reclaiming
/// tokens. Gate for the preserve implementation per docs/GC.md.
#[test]
fn gc_cache_preserve_keeps_prefix_stable() -> Result<()> {
    let traces = load_trace_cases()?;
    assert!(!traces.is_empty(), "expected at least one eval trace");

    type StrategyPair = (&'static str, Box<dyn ContextGc>, Box<dyn ContextGc>);
    let strategies: Vec<StrategyPair> = vec![
        (
            "ring",
            Box::new(RingGc {
                preserve_prefix: true,
            }),
            Box::new(RingGc {
                preserve_prefix: false,
            }),
        ),
        (
            "mark-sweep",
            Box::new(MarkSweepGc {
                preserve_prefix: true,
            }),
            Box::new(MarkSweepGc {
                preserve_prefix: false,
            }),
        ),
    ];

    for trace in &traces {
        let budget = eval_budget(&trace.prompt);
        let mut input = trace.prompt.clone();
        truncate_oversized_message(&mut input, budget);

        for (name, preserve, ignore) in &strategies {
            let mut preserve_state = GcState::default();
            let preserved = preserve.collect(input.clone(), budget, &mut preserve_state);
            let mut ignore_state = GcState::default();
            let ignored = ignore.collect(input.clone(), budget, &mut ignore_state);

            let preserve_prefix = stable_prefix_len(&input, &preserved);
            let ignore_prefix = stable_prefix_len(&input, &ignored);
            println!(
                "gc_cache_eval trace={} strategy={name} budget={budget} \
                 preserve: tokens={} stable_prefix={preserve_prefix} invalidated={} | \
                 ignore: tokens={} stable_prefix={ignore_prefix} invalidated={}",
                trace.name,
                estimate_tokens(&preserved),
                preserve_state.prefix_invalidated,
                estimate_tokens(&ignored),
                ignore_state.prefix_invalidated,
            );

            assert!(
                !preserve_state.prefix_invalidated,
                "{name} preserve on {} must not invalidate the cached prefix",
                trace.name
            );
            assert!(
                preserve_prefix >= ignore_prefix,
                "{name} preserve on {} must keep at least as long a stable prefix \
                 (preserve={preserve_prefix}, ignore={ignore_prefix})",
                trace.name
            );
            assert!(
                estimate_tokens(&preserved) < estimate_tokens(&input),
                "{name} preserve on {} must still reclaim tokens",
                trace.name
            );
            if *name == "ring" {
                // The front-drop fallback guarantees ring converges whenever
                // classic ring would; mark-sweep stays best-effort (it only
                // evicts complete/evictable lifecycles).
                assert!(
                    estimate_tokens(&preserved) <= budget,
                    "ring preserve on {} must converge under budget",
                    trace.name
                );
            }
        }
    }
    Ok(())
}

/// Longest run of leading messages the collection left untouched.
fn stable_prefix_len(original: &[ChatMessage], collected: &[ChatMessage]) -> usize {
    original
        .iter()
        .zip(collected)
        .take_while(|(before, after)| before == after)
        .count()
}
