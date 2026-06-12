//! GC strategy comparison harness (docs/GC.md "Eval Harness", t-1159).
//!
//! Every strategy must be benchmarked before promotion to default. Cases are
//! real recorded traces (`evals/gc/*.jsonl`, see `evals/gc/README.md` for how
//! to record more) plus synthetic shapes covering what the recorded set does
//! not yet: chat-heavy windows, open-tail tool chains, mixed sessions. The
//! matrix runs strategy x cache-policy x budget-pressure and prints one row
//! per combination; structural invariants are asserted, comparative quality
//! is asserted only where docs/GC.md commits to it (challengers must beat
//! ring on retained structure for tool-chain windows).

use agent_core::{
    estimate_tokens, truncate_oversized_message, ChatMessage, ContextGc, GcState, MarkSweepGc,
    RingGc, StackFrameGc, ToolCall,
};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Budget pressures the matrix samples: light (the historical eval point),
/// medium, and heavy. Heavier pressure exposes degrade paths (front-drop,
/// ring fallback) that light pressure never reaches.
const PRESSURES: [f64; 3] = [0.75, 0.5, 0.35];

/// The pressure at which the promotion gate (improves-over-ring) is judged;
/// matches the original single-point harness.
const GATE_PRESSURE: f64 = 0.75;

#[derive(Debug, Clone)]
struct TraceCase {
    name: String,
    prompt: Vec<ChatMessage>,
    /// Tool-chain windows are where mark-sweep/stack have structure to
    /// exploit; the improvement gate only applies to these.
    tool_chain: bool,
}

#[derive(Debug, Clone)]
struct EvalMetrics {
    strategy: &'static str,
    cache: &'static str,
    trace: String,
    pressure: f64,
    budget: usize,
    tokens_before: usize,
    tokens_after: usize,
    token_reduction_pct: f64,
    messages_before: usize,
    messages_after: usize,
    tool_results_before: usize,
    tool_results_after: usize,
    frames_popped: usize,
    stable_prefix: usize,
    cache_invalidated: bool,
    converged: bool,
    last_user_retained: bool,
    last_message_retained: bool,
}

enum Strategy {
    Ring,
    MarkSweep,
    Stack,
}

impl Strategy {
    fn build(&self, preserve_prefix: bool) -> Box<dyn ContextGc> {
        match self {
            Self::Ring => Box::new(RingGc { preserve_prefix }),
            Self::MarkSweep => Box::new(MarkSweepGc { preserve_prefix }),
            Self::Stack => Box::new(StackFrameGc { preserve_prefix }),
        }
    }
}

const STRATEGIES: [Strategy; 3] = [Strategy::Ring, Strategy::MarkSweep, Strategy::Stack];

/// The full comparison matrix: every case x pressure x strategy x cache
/// policy. Structural invariants are asserted on every cell; quality numbers
/// are printed for human comparison and for the promotion gate below.
#[test]
fn gc_strategy_matrix() -> Result<()> {
    let cases = all_cases()?;
    assert!(!cases.is_empty(), "expected at least one eval case");

    println!(
        "{:<28} {:>5} {:<10} {:<8} {:>7} {:>7}->{:<7} {:>5} {:>9} {:>6} {:>6} {:>6} {:>5} {:>4}",
        "case",
        "press",
        "strategy",
        "cache",
        "budget",
        "tok",
        "tok",
        "red%",
        "msgs",
        "tools",
        "frames",
        "prefix",
        "inval",
        "conv"
    );
    for case in &cases {
        for pressure in PRESSURES {
            for strategy in &STRATEGIES {
                for preserve in [true, false] {
                    let metrics = evaluate(case, pressure, strategy, preserve)?;
                    print_metrics(&metrics);
                }
            }
        }
    }
    Ok(())
}

/// The promotion gate from docs/GC.md: on tool-chain windows, challengers
/// must retain more coherent structure than ring at the gate pressure.
#[test]
fn gc_challengers_improve_over_ring_on_tool_chains() -> Result<()> {
    let cases = all_cases()?;
    let tool_chains: Vec<_> = cases.iter().filter(|case| case.tool_chain).collect();
    assert!(
        !tool_chains.is_empty(),
        "expected at least one tool-chain case"
    );

    for case in tool_chains {
        let ring = evaluate(case, GATE_PRESSURE, &Strategy::Ring, false)?;
        for challenger_kind in [Strategy::MarkSweep, Strategy::Stack] {
            let challenger = evaluate(case, GATE_PRESSURE, &challenger_kind, false)?;
            assert!(
                challenger.messages_after > ring.messages_after
                    || challenger.tool_results_after > ring.tool_results_after,
                "{} on {} must retain more coherent structure than RingGc; \
                 ring kept {} msgs/{} tool results, challenger kept {} msgs/{} tool results",
                challenger.strategy,
                challenger.trace,
                ring.messages_after,
                ring.tool_results_after,
                challenger.messages_after,
                challenger.tool_results_after
            );
        }
    }
    Ok(())
}

fn evaluate(
    case: &TraceCase,
    pressure: f64,
    strategy: &Strategy,
    preserve_prefix: bool,
) -> Result<EvalMetrics> {
    let gc = strategy.build(preserve_prefix);
    let tokens_before = estimate_tokens(&case.prompt);
    let budget = ((tokens_before as f64) * pressure).floor() as usize;

    let mut input = case.prompt.clone();
    let messages_before = input.len();
    let tool_results_before = count_tool_results(&input);
    truncate_oversized_message(&mut input, budget);

    let mut state = GcState::default();
    let collected = gc.collect(input.clone(), budget, &mut state);
    let mut state_again = GcState::default();
    let collected_again = gc.collect(input.clone(), budget, &mut state_again);
    assert_eq!(
        collected,
        collected_again,
        "{} on {} must be deterministic across two runs",
        gc.name(),
        case.name
    );

    let tokens_after = estimate_tokens(&collected);
    let converged = tokens_after <= budget;
    assert_invariants(&case.prompt, &collected, gc.name(), &case.name);
    // Ring and stack carry the front-drop degrade path and must always
    // converge; mark-sweep only evicts complete/evictable lifecycles, so its
    // convergence is best-effort and reported rather than asserted.
    if !matches!(strategy, Strategy::MarkSweep) {
        assert!(
            converged,
            "{} on {} must converge under budget: {tokens_after} > {budget}",
            gc.name(),
            case.name
        );
    }

    Ok(EvalMetrics {
        strategy: gc.name(),
        cache: if preserve_prefix {
            "preserve"
        } else {
            "ignore"
        },
        trace: case.name.clone(),
        pressure,
        budget,
        tokens_before,
        tokens_after,
        token_reduction_pct: reduction_pct(tokens_before, tokens_after),
        messages_before,
        messages_after: collected.len(),
        tool_results_before,
        tool_results_after: count_tool_results(&collected),
        frames_popped: count_frame_annotations(&collected),
        stable_prefix: stable_prefix_len(&input, &collected),
        cache_invalidated: state.prefix_invalidated,
        converged,
        last_user_retained: last_user_retained(&input, &collected),
        // Ring legitimately violates this when the tail is a tool result
        // paired to an old call (pair atomicity drags it out); the table
        // makes that visible instead of an assert hiding it.
        last_message_retained: input
            .last()
            .zip(collected.last())
            .is_none_or(|(before, after)| before.id == after.id),
    })
}

/// Invariants every strategy owes every window (docs/GC.md): system messages
/// survive and tool-call/result pairs stay atomic. A call whose result never
/// existed in the window (an open frame, mid-tool-turn) is legitimately
/// unanswered; a call whose result existed must keep it, and no result may
/// outlive its call.
fn assert_invariants(
    original: &[ChatMessage],
    collected: &[ChatMessage],
    strategy: &str,
    trace: &str,
) {
    for system in original.iter().filter(|message| message.role == "system") {
        assert!(
            collected
                .iter()
                .any(|message| message.id == system.id && message.role == "system"),
            "{strategy} on {trace} dropped pinned/system message {}",
            system.id
        );
    }

    let original_result_ids = original
        .iter()
        .filter_map(|message| message.tool_call_id.as_deref())
        .collect::<BTreeSet<_>>();
    let live_call_ids = collected
        .iter()
        .flat_map(|message| message.tool_calls.as_deref().unwrap_or_default())
        .map(|call| call.id.as_str())
        .collect::<BTreeSet<_>>();
    let live_result_ids = collected
        .iter()
        .filter_map(|message| message.tool_call_id.as_deref())
        .collect::<BTreeSet<_>>();
    for result_id in &live_result_ids {
        assert!(
            live_call_ids.contains(result_id),
            "{strategy} on {trace} kept tool result {result_id} without its call"
        );
    }
    for call_id in &live_call_ids {
        if original_result_ids.contains(call_id) {
            assert!(
                live_result_ids.contains(call_id),
                "{strategy} on {trace} split frame {call_id}: kept the call, dropped the result"
            );
        }
    }
}

// --- cases -----------------------------------------------------------------

fn all_cases() -> Result<Vec<TraceCase>> {
    let mut cases = load_trace_cases()?;
    cases.extend(synthetic_cases());
    Ok(cases)
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
                    tool_chain: true,
                });
            }
        }
    }
    Ok(cases)
}

/// Deterministic shapes the recorded fixture set does not cover yet. These
/// stand in until more real traces are recorded (evals/gc/README.md); they
/// are labeled `synthetic:` so table readers can weigh them accordingly.
fn synthetic_cases() -> Vec<TraceCase> {
    vec![
        TraceCase {
            name: "synthetic:chat-heavy".into(),
            prompt: chat_heavy_prompt(),
            tool_chain: false,
        },
        TraceCase {
            name: "synthetic:tool-chain-open-tail".into(),
            prompt: tool_chain_open_tail_prompt(),
            tool_chain: true,
        },
        TraceCase {
            name: "synthetic:mixed-session".into(),
            prompt: mixed_session_prompt(),
            tool_chain: true,
        },
    ]
}

/// Pure conversation, no tool structure: mark-sweep has nothing to evict and
/// stack has nothing to pop, so this case exercises the fallback paths.
fn chat_heavy_prompt() -> Vec<ChatMessage> {
    let mut prompt = vec![ChatMessage::system(
        "You are a helpful assistant for a long design discussion.",
    )];
    for index in 0..24 {
        prompt.push(ChatMessage::user(format!(
            "question {index}: {}",
            lorem(index, 180)
        )));
        prompt.push(ChatMessage::assistant(
            Some(format!("answer {index}: {}", lorem(index + 100, 220))),
            vec![],
        ));
    }
    prompt
}

/// A long chain of completed shell frames with an open frame at the tail —
/// the model is mid-tool-turn. The open frame must survive every strategy.
fn tool_chain_open_tail_prompt() -> Vec<ChatMessage> {
    let mut prompt = vec![
        ChatMessage::system("You are a coding agent."),
        ChatMessage::user("please refactor the parser and keep tests green"),
    ];
    for index in 0..10 {
        push_frame(&mut prompt, index, 900);
        prompt.push(ChatMessage::assistant(
            Some(format!("step {index} done: {}", lorem(index, 80))),
            vec![],
        ));
    }
    prompt.push(ChatMessage::assistant(
        None,
        vec![ToolCall::new(
            "call-open",
            "shell",
            serde_json::json!({ "command": "cargo test --workspace" }),
        )],
    ));
    prompt
}

/// Narration-heavy session with frames interleaved: both chat and frame
/// structure available to collect.
fn mixed_session_prompt() -> Vec<ChatMessage> {
    let mut prompt = vec![
        ChatMessage::system("You are a research assistant."),
        ChatMessage::user("survey the codebase and summarize the GC design"),
    ];
    for index in 0..8 {
        prompt.push(ChatMessage::user(format!(
            "follow-up {index}: {}",
            lorem(index + 50, 150)
        )));
        push_frame(&mut prompt, index + 100, 700);
        prompt.push(ChatMessage::assistant(
            Some(format!("synthesis {index}: {}", lorem(index + 200, 250))),
            vec![],
        ));
    }
    prompt
}

/// One completed frame: assistant tool call + fat tool result.
fn push_frame(prompt: &mut Vec<ChatMessage>, index: usize, result_chars: usize) {
    let call_id = format!("call-{index}");
    prompt.push(ChatMessage::assistant(
        Some(format!("running step {index}")),
        vec![ToolCall::new(
            &call_id,
            "shell",
            serde_json::json!({ "command": format!("make step-{index}") }),
        )],
    ));
    prompt.push(ChatMessage::tool(
        call_id,
        format!("output {index}: {}", lorem(index + 300, result_chars)),
    ));
}

/// Deterministic filler: repeatable pseudo-prose, no RNG.
fn lorem(seed: usize, chars: usize) -> String {
    const WORDS: [&str; 8] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    ];
    let mut out = String::new();
    let mut cursor = seed;
    while out.len() < chars {
        out.push_str(WORDS[cursor % WORDS.len()]);
        out.push(' ');
        cursor = cursor.wrapping_mul(31).wrapping_add(7);
    }
    out.truncate(chars);
    out
}

// --- metric helpers ---------------------------------------------------------

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

fn count_tool_results(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .filter(|message| message.role == "tool")
        .count()
}

fn count_frame_annotations(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .filter(|message| {
            message
                .content
                .as_deref()
                .is_some_and(|content| content.contains("[frame: "))
        })
        .count()
}

/// Continuation-viability proxy: did the most recent *user* message — the
/// statement of the task the model is currently doing — survive collection?
/// Reported per cell rather than asserted: losing it under heavy pressure is
/// exactly the failure mode the table exists to make visible.
fn last_user_retained(original: &[ChatMessage], collected: &[ChatMessage]) -> bool {
    let Some(last_user) = original.iter().rev().find(|message| message.role == "user") else {
        return true;
    };
    collected.iter().any(|message| message.id == last_user.id)
}

fn reduction_pct(before: usize, after: usize) -> f64 {
    if before == 0 {
        0.0
    } else {
        ((before.saturating_sub(after) as f64) / (before as f64)) * 100.0
    }
}

/// Longest run of leading messages the collection left untouched.
fn stable_prefix_len(original: &[ChatMessage], collected: &[ChatMessage]) -> usize {
    original
        .iter()
        .zip(collected)
        .take_while(|(before, after)| before == after)
        .count()
}

fn print_metrics(metrics: &EvalMetrics) {
    println!(
        "{:<28} {:>5.2} {:<10} {:<8} {:>7} {:>7}->{:<7} {:>4.1}% {:>4}/{:<4} {:>3}/{:<3} {:>6} {:>6} {:>5} {:>4}{}",
        metrics.trace,
        metrics.pressure,
        metrics.strategy,
        metrics.cache,
        metrics.budget,
        metrics.tokens_before,
        metrics.tokens_after,
        metrics.token_reduction_pct,
        metrics.messages_after,
        metrics.messages_before,
        metrics.tool_results_after,
        metrics.tool_results_before,
        metrics.frames_popped,
        metrics.stable_prefix,
        metrics.cache_invalidated,
        metrics.converged,
        match (metrics.last_user_retained, metrics.last_message_retained) {
            (true, true) => "",
            (false, true) => "  !last-user-dropped",
            (true, false) => "  !tail-dropped",
            (false, false) => "  !last-user-dropped !tail-dropped",
        }
    );
}

/// Compare --gc-cache preserve against ignore on the fixture set: preserve
/// must keep a stable leading prefix (provider prompt caches key on it) at
/// least as long as ignore's, without invalidating it, while still reclaiming
/// tokens. Gate for the preserve implementation per docs/GC.md.
#[test]
fn gc_cache_preserve_keeps_prefix_stable() -> Result<()> {
    let traces: Vec<_> = all_cases()?
        .into_iter()
        .filter(|case| case.tool_chain)
        .collect();
    assert!(!traces.is_empty(), "expected at least one eval trace");

    for trace in &traces {
        let budget = ((estimate_tokens(&trace.prompt) as f64) * GATE_PRESSURE).floor() as usize;
        let mut input = trace.prompt.clone();
        truncate_oversized_message(&mut input, budget);

        for strategy in &STRATEGIES {
            let preserve = strategy.build(true);
            let ignore = strategy.build(false);
            let name = preserve.name();
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
            if name != "mark-sweep" {
                // Ring and stack carry the front-drop fallback and must
                // converge; mark-sweep stays best-effort (it only evicts
                // complete/evictable lifecycles).
                assert!(
                    estimate_tokens(&preserved) <= budget,
                    "{name} preserve on {} must converge under budget",
                    trace.name
                );
            }
        }
    }
    Ok(())
}
