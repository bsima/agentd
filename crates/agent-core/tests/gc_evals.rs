//! GC strategy comparison harness (docs/GC.md "Eval Harness", t-1159).
//!
//! Every strategy must be benchmarked before promotion to default. Cases are
//! real recorded traces (`evals/gc/*.jsonl`, see `evals/gc/README.md` for how
//! to record more) plus synthetic shapes covering what the recorded set does
//! not yet: chat-heavy windows, open-tail tool chains, mixed sessions, long
//! tool-heavy sessions. The matrix runs strategy x timing x cache-policy x
//! budget-pressure and prints one row per combination; structural invariants
//! are asserted, comparative quality is asserted only where docs/GC.md
//! commits to it (challengers must beat ring on retained structure for
//! tool-chain windows).
//!
//! The timing axis simulates *when* GC runs over the life of the session
//! (mirroring `--gc-timing`, see `interpreter::maybe_collect_prompt`):
//!
//! - `final`: one collection on the full recorded window — what the first
//!   catch-overflow cycle sees.
//! - `threshold`: replay the session growing message-by-message; before each
//!   assistant turn (an infer point) collect iff the estimate exceeds the
//!   budget.
//! - `eager`: collect at every infer point.
//! - `every:4`: collect at every 4th infer point.
//!
//! All incremental timings compose with the runtime's collect-on-overflow
//! backstop (t-1343): an infer point whose window estimate exceeds the
//! budget collects regardless of the timing policy, so no timing dispatches
//! an over-budget window.
//!
//! Incremental timings thread one `GcState` across all collections, so
//! cross-turn metadata (frame status, lifecycle tags, infer counts) behaves
//! as it does in the runtime loop.
//!
//! The optional `judge` column (t-1168) is an LLM semantic-coherence score,
//! ONLINE-GATED behind `RUN_AGENT_ONLINE_EVAL=1` with recorded-judge replay
//! by default — see the judge section at the bottom of this file.

use agent_core::gc::{message_embedding_text, CitationGraph, SemanticGc};
use agent_core::{
    content_hash, estimate_tokens, truncate_oversized_message, ChatMessage, ChatProvider,
    ContextGc, GcState, MarkSweepGc, Model, ProviderClient, ProviderConfig, RingGc, StackFrameGc,
    ToolCall,
};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
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
    timing: Timing,
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
    /// How many collections ran (1 for `final`; up to one per infer point
    /// for the incremental timings).
    collections: usize,
    /// How many of those collections invalidated the cached prefix — each
    /// one is a full-window re-read at the provider.
    invalidations: usize,
    converged: bool,
    last_user_retained: bool,
    last_message_retained: bool,
    /// Semantic-coherence judge score (t-1168), e.g. "3/3". None when no
    /// recording exists and the run is offline, or when the judge response
    /// did not parse.
    judge: Option<String>,
}

enum Strategy {
    Ring,
    MarkSweep,
    Stack,
    Semantic,
}

impl Strategy {
    fn build(&self, preserve_prefix: bool) -> Box<dyn ContextGc> {
        match self {
            Self::Ring => Box::new(RingGc { preserve_prefix }),
            Self::MarkSweep => Box::new(MarkSweepGc { preserve_prefix }),
            Self::Stack => Box::new(StackFrameGc { preserve_prefix }),
            Self::Semantic => Box::new(SemanticGc {
                preserve_prefix,
                ..Default::default()
            }),
        }
    }
}

const STRATEGIES: [Strategy; 4] = [
    Strategy::Ring,
    Strategy::MarkSweep,
    Strategy::Stack,
    Strategy::Semantic,
];

// --- deterministic offline embedder (t-1350) ---------------------------------
//
// Semantic cells score against GcState.embeddings, which the runtime's async
// pre-pass fills from a real embeddings endpoint. The harness mirrors that
// pre-pass with a deterministic mock — the same recorded-replay stance as the
// judge column: offline runs never touch a provider and produce identical
// vectors (and therefore identical collections) every run.

/// Bag-of-tokens vector: each token FNV-hashes into one of 64 buckets, so
/// cosine similarity reflects vocabulary overlap. Deterministic, no RNG.
fn mock_vector(text: &str) -> Vec<f32> {
    const DIMS: u64 = 64;
    let mut vector = vec![0.0f32; DIMS as usize];
    for token in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in token.to_ascii_lowercase().bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        vector[(hash % DIMS) as usize] += 1.0;
    }
    vector
}

/// Mirror of `SemanticGc::prime_cache` (interpreter pre-pass): cache a vector
/// for every window message, keyed exactly as the runtime keys them.
fn prime_semantic_cache(window: &[ChatMessage], state: &mut GcState) {
    for message in window {
        let text = message_embedding_text(message);
        state
            .embeddings
            .entry(content_hash(&text))
            .or_insert_with(|| mock_vector(&text));
    }
}

/// When GC runs over the simulated session (see module docs). `Final`
/// approximates catch-overflow's first cycle; the rest mirror `--gc-timing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Timing {
    Final,
    Threshold,
    Eager,
    EveryN(u64),
}

impl Timing {
    fn label(&self) -> String {
        match self {
            Self::Final => "final".into(),
            Self::Threshold => "threshold".into(),
            Self::Eager => "eager".into(),
            Self::EveryN(n) => format!("every:{n}"),
        }
    }
}

const TIMINGS: [Timing; 4] = [
    Timing::Final,
    Timing::Threshold,
    Timing::Eager,
    Timing::EveryN(4),
];

/// The full comparison matrix: every case x pressure x timing x strategy x
/// cache policy. Structural invariants are asserted on every cell; quality
/// numbers are printed for human comparison and for the promotion gate below.
#[test]
fn gc_strategy_matrix() -> Result<()> {
    let cases = all_cases()?;
    assert!(!cases.is_empty(), "expected at least one eval case");

    let mut judge = JudgeBook::load_default()?;
    print_header();
    for case in &cases {
        for pressure in PRESSURES {
            for timing in TIMINGS {
                for strategy in &STRATEGIES {
                    for preserve in [true, false] {
                        let metrics =
                            evaluate(case, pressure, timing, strategy, preserve, Some(&mut judge))?;
                        print_metrics(&metrics);
                    }
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
        let ring = evaluate(
            case,
            GATE_PRESSURE,
            Timing::Final,
            &Strategy::Ring,
            false,
            None,
        )?;
        for challenger_kind in [Strategy::MarkSweep, Strategy::Stack] {
            let challenger = evaluate(
                case,
                GATE_PRESSURE,
                Timing::Final,
                &challenger_kind,
                false,
                None,
            )?;
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

/// How a strategy handled the tangent fixture: what fraction of the tangent
/// it dropped, and what fraction of the relevant (non-system, non-tangent)
/// messages it retained. "Retained" counts a message as surviving even if
/// rewritten in place (stack's frame annotations keep the stable id) —
/// generous to stack, so the comparison is honest.
#[derive(Debug, Clone, Copy)]
struct TangentMetrics {
    tangent_dropped: f64,
    relevant_retained: f64,
}

fn tangent_metrics(
    original: &[ChatMessage],
    collected: &[ChatMessage],
    tangent: &[usize],
) -> TangentMetrics {
    let survived = |index: &usize| {
        collected
            .iter()
            .any(|message| message.id == original[*index].id)
    };
    let tangent_kept = tangent.iter().filter(|index| survived(index)).count();
    let relevant: Vec<usize> = (0..original.len())
        .filter(|index| !tangent.contains(index) && original[*index].role != "system")
        .collect();
    let relevant_kept = relevant.iter().filter(|index| survived(index)).count();
    TangentMetrics {
        tangent_dropped: 1.0 - (tangent_kept as f64) / (tangent.len() as f64),
        relevant_retained: (relevant_kept as f64) / (relevant.len() as f64),
    }
}

/// The t-1350 promotion bar, half one: on the conversational-dead-end
/// fixture, semantic must drop more of the abandoned tangent than stack
/// while retaining at least as much of the relevant thread. Stack pops
/// frames oldest-first, so under pressure it evicts the EARLY on-topic
/// work and keeps the newer tangent; semantic scores the tangent as
/// distant from the recent thread and drops it first.
#[test]
fn gc_semantic_drops_the_tangent_and_keeps_the_relevant_thread() -> Result<()> {
    let (prompt, tangent) = tangent_abandoned_case();
    for preserve in [true, false] {
        let budget = ((estimate_tokens(&prompt) as f64) * GATE_PRESSURE).floor() as usize;
        let mut per_strategy = Vec::new();
        for strategy in &STRATEGIES {
            let gc = strategy.build(preserve);
            let run = run_timed(&prompt, budget, gc.as_ref(), Timing::Final);
            let metrics = tangent_metrics(&prompt, &run.collected, &tangent);
            println!(
                "tangent_eval strategy={} cache={} tangent_dropped={:.2} relevant_retained={:.2}",
                gc.name(),
                if preserve { "preserve" } else { "ignore" },
                metrics.tangent_dropped,
                metrics.relevant_retained,
            );
            per_strategy.push((gc.name(), metrics));
        }
        let find = |name: &str| {
            per_strategy
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, metrics)| *metrics)
                .expect("strategy evaluated")
        };
        let semantic = find("semantic");
        let stack = find("stack");
        assert!(
            semantic.tangent_dropped > stack.tangent_dropped,
            "semantic must drop more of the tangent than stack (preserve={preserve}): \
             semantic={:.2}, stack={:.2}",
            semantic.tangent_dropped,
            stack.tangent_dropped
        );
        assert!(
            semantic.relevant_retained >= stack.relevant_retained,
            "semantic must retain at least as much of the relevant thread as stack \
             (preserve={preserve}): semantic={:.2}, stack={:.2}",
            semantic.relevant_retained,
            stack.relevant_retained
        );
        assert!(
            semantic.tangent_dropped >= 0.75,
            "semantic should evict most of the tangent (preserve={preserve}): {:.2}",
            semantic.tangent_dropped
        );
    }
    Ok(())
}

/// The t-1350 promotion bar, half two: winning on tangents must not cost
/// the existing fixture classes. Replay-completion proxy: wherever stack
/// retains the last user message (the statement of the current task),
/// semantic must too — at every pressure, on every case, under both cache
/// policies. Convergence and determinism are asserted inside evaluate()
/// for every cell.
#[test]
fn gc_semantic_no_regression_vs_stack_on_replay_completion() -> Result<()> {
    for case in &all_cases()? {
        for pressure in PRESSURES {
            for preserve in [true, false] {
                let stack = evaluate(
                    case,
                    pressure,
                    Timing::Final,
                    &Strategy::Stack,
                    preserve,
                    None,
                )?;
                let semantic = evaluate(
                    case,
                    pressure,
                    Timing::Final,
                    &Strategy::Semantic,
                    preserve,
                    None,
                )?;
                if stack.last_user_retained {
                    assert!(
                        semantic.last_user_retained,
                        "semantic dropped the last user message where stack kept it: \
                         case={} pressure={pressure} preserve={preserve}",
                        case.name
                    );
                }
            }
        }
    }
    Ok(())
}

/// The t-1351 matrix row (docs/GC.md "Citation signals", the cited+distant
/// cell of the 2x2): on the cited-distant fixture, SemanticGc WITHOUT
/// citations drops the old, semantically distant, explicitly cited tool
/// result — that deficiency is pinned here as the baseline — while
/// SemanticGc with `cited-keep` (the default) retains it and pays with the
/// uncited noise frames instead. Both must converge, under both cache
/// policies.
#[test]
fn gc_semantic_cited_keep_retains_the_cited_distant_result() -> Result<()> {
    let (prompt, cited, noise) = cited_distant_case();
    // Fixture honesty: the frame that must survive is cited; the noise
    // frames — same topic, same size, same age class — are not. The
    // citation is the only distinguishing signal.
    let citations = CitationGraph::extract(&prompt);
    let cited_result = cited
        .iter()
        .copied()
        .find(|index| prompt[*index].role == "tool")
        .expect("the cited frame has a tool result");
    assert!(
        citations.is_cited(&prompt[cited_result].id),
        "fixture honesty: the audit result must be cited"
    );
    for index in &noise {
        assert!(
            !citations.is_cited(&prompt[*index].id),
            "fixture honesty: noise frame at {index} must be uncited"
        );
    }

    for preserve in [true, false] {
        let budget = ((estimate_tokens(&prompt) as f64) * GATE_PRESSURE).floor() as usize;
        let survived = |collected: &[ChatMessage], index: usize| {
            collected
                .iter()
                .any(|message| message.id == prompt[index].id)
        };

        let baseline = SemanticGc {
            preserve_prefix: preserve,
            cited_keep: false,
            ..Default::default()
        };
        let baseline_run = run_timed(&prompt, budget, &baseline, Timing::Final);
        let cited_keep = SemanticGc {
            preserve_prefix: preserve,
            ..Default::default()
        };
        let cited_run = run_timed(&prompt, budget, &cited_keep, Timing::Final);

        let noise_retained = noise
            .iter()
            .filter(|index| survived(&cited_run.collected, **index))
            .count();
        println!(
            "cited_eval cache={} budget={budget} \
             baseline(no citations): cited_retained={} tokens={} | \
             cited-keep: cited_retained={} noise_retained={}/{} tokens={}",
            if preserve { "preserve" } else { "ignore" },
            survived(&baseline_run.collected, cited_result),
            estimate_tokens(&baseline_run.collected),
            survived(&cited_run.collected, cited_result),
            noise_retained,
            noise.len(),
            estimate_tokens(&cited_run.collected),
        );

        assert!(estimate_tokens(&baseline_run.collected) <= budget);
        assert!(
            estimate_tokens(&cited_run.collected) <= budget,
            "cited-keep must still converge (preserve={preserve})"
        );
        // The baseline deficiency, pinned: pure similarity cannot tell the
        // load-bearing frame from the noise around it.
        assert!(
            !survived(&baseline_run.collected, cited_result),
            "semantic-without-citations must drop the cited-but-distant \
             result (preserve={preserve}) — if this starts passing, the \
             fixture no longer isolates the citation signal"
        );
        // The fix: the citation keeps it.
        assert!(
            survived(&cited_run.collected, cited_result),
            "semantic+cited-keep must retain the cited result (preserve={preserve})"
        );
    }
    Ok(())
}

/// No regression on the tangent fixture class: the tangent is uncited by
/// construction (asserted — fixture honesty), so cited-keep must be exactly
/// inert there: identical collections with the modifier on and off.
#[test]
fn gc_cited_keep_is_inert_on_the_uncited_tangent_fixture() -> Result<()> {
    let (prompt, tangent) = tangent_abandoned_case();
    let citations = CitationGraph::extract(&prompt);
    for index in &tangent {
        assert!(
            !citations.is_cited(&prompt[*index].id),
            "fixture honesty: the tangent at {index} must be uncited"
        );
    }
    for preserve in [true, false] {
        let budget = ((estimate_tokens(&prompt) as f64) * GATE_PRESSURE).floor() as usize;
        let off = SemanticGc {
            preserve_prefix: preserve,
            cited_keep: false,
            ..Default::default()
        };
        let on = SemanticGc {
            preserve_prefix: preserve,
            ..Default::default()
        };
        assert_eq!(
            run_timed(&prompt, budget, &off, Timing::Final).collected,
            run_timed(&prompt, budget, &on, Timing::Final).collected,
            "cited-keep must be a no-op on an uncited window (preserve={preserve})"
        );
    }
    Ok(())
}

fn evaluate(
    case: &TraceCase,
    pressure: f64,
    timing: Timing,
    strategy: &Strategy,
    preserve_prefix: bool,
    judge: Option<&mut JudgeBook>,
) -> Result<EvalMetrics> {
    let gc = strategy.build(preserve_prefix);
    let tokens_before = estimate_tokens(&case.prompt);
    let budget = ((tokens_before as f64) * pressure).floor() as usize;

    let messages_before = case.prompt.len();
    let tool_results_before = count_tool_results(&case.prompt);

    let run = run_timed(&case.prompt, budget, gc.as_ref(), timing);
    let run_again = run_timed(&case.prompt, budget, gc.as_ref(), timing);
    assert_eq!(
        run.collected,
        run_again.collected,
        "{} ({}) on {} must be deterministic across two runs",
        gc.name(),
        timing.label(),
        case.name
    );

    let collected = run.collected;
    let tokens_after = estimate_tokens(&collected);
    let converged = tokens_after <= budget;
    assert_invariants(&case.prompt, &collected, gc.name(), &case.name);
    // Ring and stack carry the front-drop degrade path and must converge on
    // every timing: since the collect-on-overflow backstop (t-1343), every
    // infer point over budget collects — `every:N` can no longer end over
    // budget between collections. Mark-sweep only evicts complete/evictable
    // lifecycles, so its convergence is best-effort and reported rather
    // than asserted.
    if !matches!(strategy, Strategy::MarkSweep) {
        assert!(
            converged,
            "{} ({}) on {} must converge under budget: {tokens_after} > {budget}",
            gc.name(),
            timing.label(),
            case.name
        );
    }

    let cache = if preserve_prefix {
        "preserve"
    } else {
        "ignore"
    };
    let judge_score = match judge {
        Some(book) => {
            let cell = format!(
                "{}|{:.2}|{}|{}|{}",
                case.name,
                pressure,
                timing.label(),
                gc.name(),
                cache
            );
            book.verdict(&cell, &case.prompt, &collected)?
                .map(|verdict| verdict.display())
        }
        None => None,
    };

    Ok(EvalMetrics {
        strategy: gc.name(),
        timing,
        cache,
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
        stable_prefix: stable_prefix_len(&case.prompt, &collected),
        collections: run.collections,
        invalidations: run.invalidations,
        converged,
        last_user_retained: last_user_retained(&case.prompt, &collected),
        // Ring legitimately violates this when the tail is a tool result
        // paired to an old call (pair atomicity drags it out); the table
        // makes that visible instead of an assert hiding it.
        last_message_retained: case
            .prompt
            .last()
            .zip(collected.last())
            .is_none_or(|(before, after)| before.id == after.id),
        judge: judge_score,
    })
}

struct TimedRun {
    collected: Vec<ChatMessage>,
    collections: usize,
    invalidations: usize,
}

/// Apply `gc` to the case window under the given timing. `Final` is the
/// historical single-shot collection; the incremental timings replay the
/// session growing message-by-message and fire at infer points (right
/// before each assistant message, plus once on the full window — the
/// recorded window is itself the prompt of the next infer call), mirroring
/// `interpreter::maybe_collect_prompt`.
fn run_timed(
    prompt: &[ChatMessage],
    budget: usize,
    gc: &dyn ContextGc,
    timing: Timing,
) -> TimedRun {
    let mut state = GcState::default();
    let mut run = TimedRun {
        collected: Vec::new(),
        collections: 0,
        invalidations: 0,
    };
    if timing == Timing::Final {
        let mut window = prompt.to_vec();
        truncate_oversized_message(&mut window, budget);
        // Semantic pre-pass mirror (after truncation: the cache keys on
        // content, exactly as interpreter::collect_prompt orders it).
        if gc.name() == "semantic" {
            prime_semantic_cache(&window, &mut state);
        }
        run.collected = gc.collect(window, budget, &mut state);
        run.collections = 1;
        run.invalidations = usize::from(state.prefix_invalidated);
        return run;
    }

    let mut window: Vec<ChatMessage> = Vec::new();
    for message in prompt {
        if message.role == "assistant" {
            infer_point(&mut window, budget, gc, timing, &mut state, &mut run);
        }
        window.push(message.clone());
    }
    infer_point(&mut window, budget, gc, timing, &mut state, &mut run);
    run.collected = window;
    run
}

/// One infer point: decide per the timing policy whether to collect, exactly
/// as `maybe_collect_prompt` does (with the harness budget standing in for
/// `context_budget * gc_threshold`).
fn infer_point(
    window: &mut Vec<ChatMessage>,
    budget: usize,
    gc: &dyn ContextGc,
    timing: Timing,
    state: &mut GcState,
    run: &mut TimedRun,
) {
    state.infer_calls += 1;
    let fire = match timing {
        Timing::Threshold => estimate_tokens(window) > budget,
        Timing::Eager => true,
        Timing::EveryN(n) => state.infer_calls.is_multiple_of(n),
        Timing::Final => unreachable!("final timing never reaches an infer point"),
    };
    // Collect-on-overflow backstop (t-1343), mirroring
    // `maybe_collect_prompt`: no timing policy ever dispatches a window
    // whose estimate exceeds the budget without collecting first, so
    // `every:N` can no longer leave the window over budget between its
    // scheduled collections.
    let fire = fire || estimate_tokens(window) > budget;
    if !fire {
        return;
    }
    truncate_oversized_message(window, budget);
    if gc.name() == "semantic" {
        prime_semantic_cache(window, state);
    }
    *window = gc.collect(std::mem::take(window), budget, state);
    run.collections += 1;
    run.invalidations += usize::from(state.prefix_invalidated);
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
        TraceCase {
            name: "synthetic:tool-heavy-long".into(),
            prompt: tool_heavy_long_prompt(),
            tool_chain: true,
        },
        TraceCase {
            name: "synthetic:tangent-abandoned".into(),
            prompt: tangent_abandoned_case().0,
            tool_chain: true,
        },
        TraceCase {
            name: "synthetic:cited-distant".into(),
            prompt: cited_distant_case().0,
            tool_chain: true,
        },
    ]
}

/// Vocabulary pools for the tangent fixture: the mock embedder scores by
/// token overlap, so distinct pools = semantically distant, exactly like
/// distinct topics under a real embedding model.
const PLANNER_WORDS: [&str; 12] = [
    "planner",
    "join",
    "index",
    "statistics",
    "selectivity",
    "postgres",
    "analyze",
    "btree",
    "rows",
    "estimate",
    "scan",
    "vacuum",
];

const CACHE_WORDS: [&str; 12] = [
    "redis",
    "cache",
    "invalidation",
    "websocket",
    "dashboard",
    "frontend",
    "javascript",
    "payload",
    "subscription",
    "broker",
    "eviction",
    "socket",
];

/// Deterministic topical filler: repeatable pseudo-prose drawn from one
/// vocabulary pool, no RNG (the topic-specific sibling of `lorem`).
fn topical_filler(words: &[&str], seed: usize, chars: usize) -> String {
    let mut out = String::new();
    let mut cursor = seed;
    while out.len() < chars {
        out.push_str(words[cursor % words.len()]);
        out.push(' ');
        cursor = cursor.wrapping_mul(31).wrapping_add(7);
    }
    out.truncate(chars);
    out
}

/// One completed frame on a given topic vocabulary.
fn push_topical_frame(
    prompt: &mut Vec<ChatMessage>,
    words: &[&str],
    call_id: &str,
    command: &str,
    narration: &str,
    seed: usize,
    result_chars: usize,
) {
    prompt.push(ChatMessage::assistant(
        Some(narration.into()),
        vec![ToolCall::new(
            call_id,
            "shell",
            serde_json::json!({ "command": command }),
        )],
    ));
    prompt.push(ChatMessage::tool(
        call_id.to_string(),
        topical_filler(words, seed, result_chars),
    ));
}

/// The conversational-dead-end fixture class (t-1350): a session explores a
/// wrong approach for several turns (rewriting the reporting layer around a
/// cache service), abandons it, and continues correctly on the original
/// problem (the query planner). The tangent sits in the MIDDLE of the
/// window — newer than the early on-topic work — so position-based
/// strategies (ring's front-drop, stack's oldest-frame-first) evict the
/// wrong messages first, while the tangent is semantically distant from the
/// recent thread. Returns the prompt and the indices of the tangent
/// messages.
fn tangent_abandoned_case() -> (Vec<ChatMessage>, Vec<usize>) {
    let mut prompt = vec![
        ChatMessage::system("You are a database engineering agent."),
        ChatMessage::user(
            "The orders report query is slow: the planner picks a bad join order. Fix the query plan.",
        ),
    ];
    // On-topic exploration: plans, statistics, index selectivity.
    for step in 0..4 {
        push_topical_frame(
            &mut prompt,
            &PLANNER_WORDS,
            &format!("call-plan-{step}"),
            &format!("psql -c 'EXPLAIN ANALYZE SELECT ...' # step {step}"),
            &format!("Inspecting the query plan: join order and index selectivity, step {step}."),
            step,
            500,
        );
    }

    // The tangent: rewrite the dashboard around a cache service instead.
    let tangent_start = prompt.len();
    prompt.push(ChatMessage::assistant(
        Some(
            "Different idea: skip the planner entirely and build a redis cache service \
             with websocket push to the dashboard frontend."
                .into(),
        ),
        vec![],
    ));
    for step in 0..3 {
        push_topical_frame(
            &mut prompt,
            &CACHE_WORDS,
            &format!("call-cache-{step}"),
            &format!("redis-cli --scan # cache layer step {step}"),
            &format!(
                "Sketching the cache invalidation and websocket subscription flow, step {step}."
            ),
            step + 40,
            800,
        );
    }
    prompt.push(ChatMessage::assistant(
        Some("Cache service and dashboard websocket push sketched out.".into()),
        vec![],
    ));
    let tangent: Vec<usize> = (tangent_start..prompt.len()).collect();

    // Abandonment + correct continuation.
    prompt.push(ChatMessage::user(
        "Stop — the cache rewrite is out of scope. Go back to fixing the query planner.",
    ));
    for step in 0..3 {
        push_topical_frame(
            &mut prompt,
            &PLANNER_WORDS,
            &format!("call-fix-{step}"),
            &format!("psql -c 'ALTER TABLE orders ...; ANALYZE orders' # fix step {step}"),
            &format!("Raising the statistics target and adding the composite index, step {step}."),
            step + 80,
            500,
        );
    }
    prompt.push(ChatMessage::assistant(
        Some(
            "Composite index created and statistics refreshed: the planner now picks the \
             index scan and the query runs in 40ms."
                .into(),
        ),
        vec![],
    ));
    prompt.push(ChatMessage::user(
        "Great — verify the weekly report query also uses the new index.",
    ));
    (prompt, tangent)
}

/// Third vocabulary pool for the cited-distant fixture (t-1351): a security
/// audit — semantically distant from the planner thread under the mock
/// embedder, exactly like a distinct topic under a real one.
const AUDIT_WORDS: [&str; 12] = [
    "audit",
    "vulnerability",
    "dependency",
    "advisory",
    "libfoo",
    "cve",
    "signature",
    "checksum",
    "pinning",
    "sbom",
    "license",
    "transitive",
];

/// The cited-distant fixture class (t-1351, docs/GC.md "Citation signals"
/// 2x2, the cited+distant cell): an OLD tool result that is semantically
/// distant from the recent thread — a dependency-audit lookup in the middle
/// of query-planner work — but explicitly cited by a recent message ("Per
/// the output of call-audit-0, ..."). Pure similarity scoring drops it with
/// the uncited audit noise around it; cited-keep must not. The noise frames
/// share the audit vocabulary and are UNcited by construction, so the
/// citation — not the topic — is the only thing distinguishing the frame
/// that must survive. Returns the prompt, the indices of the cited frame
/// (call + result), and the indices of the uncited noise frames.
fn cited_distant_case() -> (Vec<ChatMessage>, Vec<usize>, Vec<usize>) {
    let mut prompt = vec![
        ChatMessage::system("You are a database engineering agent."),
        ChatMessage::user(
            "The orders report query is slow: the planner picks a bad join order. Fix the query plan.",
        ),
    ];
    // Early on-topic work: fills the preserve-mode prefix allowance so the
    // audit frames below sit in the evictable interior.
    for step in 0..2 {
        push_topical_frame(
            &mut prompt,
            &PLANNER_WORDS,
            &format!("call-plan-{step}"),
            &format!("psql -c 'EXPLAIN ANALYZE SELECT ...' # step {step}"),
            &format!("Inspecting the query plan, step {step}."),
            step,
            500,
        );
    }
    // The audit sidebar: one frame a later message will cite...
    // All four audit results share one seed so their embedding texts — and
    // therefore their similarity scores — tie exactly; the sweep's
    // oldest-first tie-break then makes the CITED frame (deliberately the
    // oldest) the first thing pure similarity kills. The citation is the
    // only signal distinguishing it from the noise.
    push_topical_frame(
        &mut prompt,
        &AUDIT_WORDS,
        "call-audit-0",
        "cargo audit --json # dependency check",
        "Side check: auditing the dependency tree before touching the schema.",
        7,
        800,
    );
    let cited: Vec<usize> = vec![prompt.len() - 2, prompt.len() - 1];
    // ...and noise frames on the same distant topic that nothing ever cites.
    let noise_start = prompt.len();
    for step in 0..3 {
        push_topical_frame(
            &mut prompt,
            &AUDIT_WORDS,
            &format!("call-noise-{step}"),
            "cargo audit --json # dependency check",
            "Side check: auditing the dependency tree before touching the schema.",
            7,
            800,
        );
    }
    let noise: Vec<usize> = (noise_start..prompt.len()).collect();
    // Recent on-topic work + the citation: the model builds on the audit
    // output while doing planner work.
    for step in 0..3 {
        push_topical_frame(
            &mut prompt,
            &PLANNER_WORDS,
            &format!("call-fix-{step}"),
            &format!("psql -c 'ALTER TABLE orders ...; ANALYZE orders' # fix step {step}"),
            &format!("Raising the statistics target and adding the index, step {step}."),
            step + 80,
            500,
        );
    }
    prompt.push(ChatMessage::assistant(
        Some(
            "Per the output of call-audit-0, the slow join comes through the unpatched \
             libfoo dependency — applying the planner fix with that version pinned."
                .into(),
        ),
        vec![],
    ));
    prompt.push(ChatMessage::user(
        "Good — apply it that way and rerun the orders report.",
    ));
    (prompt, cited, noise)
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

/// A long tool-heavy coding session: many fat completed frames, minimal
/// narration, a live user question at the tail. This is the fixture class
/// where strategies should differ most — stack can pop dozens of dead
/// frames to annotations, ring can only amputate history wholesale — and it
/// stands in for the "long coding session" gap in evals/gc/README.md until
/// a real trace of that shape is recorded.
fn tool_heavy_long_prompt() -> Vec<ChatMessage> {
    let mut prompt = vec![
        ChatMessage::system("You are a coding agent working through a large migration."),
        ChatMessage::user(
            "migrate every module to the new error type, running tests after each module",
        ),
    ];
    for index in 0..28 {
        push_frame(&mut prompt, index + 500, 1200);
        if index % 7 == 6 {
            prompt.push(ChatMessage::assistant(
                Some(format!(
                    "checkpoint {index}: modules migrated so far, tests green. {}",
                    lorem(index + 700, 120)
                )),
                vec![],
            ));
        }
    }
    prompt.push(ChatMessage::user(
        "before you continue: which modules are left, and did anything regress?",
    ));
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

fn print_header() {
    println!(
        "{:<28} {:>5} {:<9} {:<10} {:<8} {:>7} {:>7}->{:<7} {:>5} {:>9} {:>6} {:>6} {:>6} {:>4} {:>5} {:>4} {:>5}",
        "case",
        "press",
        "timing",
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
        "coll",
        "inval",
        "conv",
        "judge"
    );
}

fn print_metrics(metrics: &EvalMetrics) {
    println!(
        "{:<28} {:>5.2} {:<9} {:<10} {:<8} {:>7} {:>7}->{:<7} {:>4.1}% {:>4}/{:<4} {:>3}/{:<3} {:>6} {:>6} {:>4} {:>5} {:>4} {:>5}{}",
        metrics.trace,
        metrics.pressure,
        metrics.timing.label(),
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
        metrics.collections,
        metrics.invalidations,
        metrics.converged,
        metrics.judge.as_deref().unwrap_or("-"),
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

// --- semantic-coherence judge (t-1168) ---------------------------------------
//
// An LLM scores whether a GC-collected window preserves what is needed to
// continue the session coherently. ONLINE-GATED: the offline matrix never
// talks to a provider. Judge responses are recorded to
// `evals/gc/judge/recorded.jsonl` keyed by a content hash of the judge
// prompt and replayed from there by default, so reruns are comparable
// offline. Set RUN_AGENT_ONLINE_EVAL=1 (the evals/ convention) to score
// unrecorded cells against a real model, appending the recordings.

/// Rubric the judge scores against. Fixed string: the judge prompt must be
/// deterministic so its hash is a stable recording key.
const JUDGE_RUBRIC: &str = "You are auditing a context-window garbage collection. \
You are shown a FULL conversation window and the COLLECTED window an agent will \
actually continue from. Decide whether the collected window preserves what is \
needed to continue the session coherently.\n\
Score three booleans:\n\
- task_goal_retained: the task/goal the assistant is currently working on is \
still stated or reconstructible from the collected window.\n\
- open_threads_retained: work that was still in flight (unanswered questions, \
pending tool activity, unfinished steps) is still visible.\n\
- no_orphaned_references: the collected window does not depend on dropped \
content (e.g. 'as shown above' with the referent gone, discussion of results \
that no longer appear).\n\
Reply with ONLY a JSON object, no prose: \
{\"task_goal_retained\": bool, \"open_threads_retained\": bool, \"no_orphaned_references\": bool}";

/// Cap for each rendered message body in the judge prompt: keeps prompts
/// bounded and deterministic.
const JUDGE_RENDER_CHARS: usize = 280;

/// Deterministic, comparable score shape: three rubric booleans, displayed
/// as "<sum>/3".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct JudgeVerdict {
    task_goal_retained: bool,
    open_threads_retained: bool,
    no_orphaned_references: bool,
}

impl JudgeVerdict {
    fn score(&self) -> u8 {
        u8::from(self.task_goal_retained)
            + u8::from(self.open_threads_retained)
            + u8::from(self.no_orphaned_references)
    }

    fn display(&self) -> String {
        format!("{}/3", self.score())
    }
}

/// One recorded judge exchange. `cell` and `model` are provenance for human
/// readers; lookup is purely by `key`. `note` marks hand-written entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JudgeRecord {
    key: String,
    cell: String,
    model: String,
    response: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

/// Recorded-judge replay book. Offline (the default) it only serves
/// recordings; online it scores misses against a real provider and appends
/// them to the recording file.
struct JudgeBook {
    path: PathBuf,
    recordings: HashMap<String, String>,
    online: Option<JudgeClient>,
}

struct JudgeClient {
    client: ProviderClient,
    model: Model,
    runtime: tokio::runtime::Runtime,
}

impl JudgeBook {
    /// Load the shipped recording file; go online only under
    /// RUN_AGENT_ONLINE_EVAL=1 (the evals/ convention).
    fn load_default() -> Result<Self> {
        let path = judge_recording_path()?;
        let online = std::env::var("RUN_AGENT_ONLINE_EVAL").is_ok_and(|v| v == "1");
        Self::load(path, online)
    }

    fn load(path: PathBuf, online: bool) -> Result<Self> {
        let mut recordings = HashMap::new();
        if path.exists() {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("reading judge recordings {}", path.display()))?;
            for (line_idx, line) in content.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let record: JudgeRecord = serde_json::from_str(line).with_context(|| {
                    format!("decoding {} line {}", path.display(), line_idx + 1)
                })?;
                recordings.insert(record.key, record.response);
            }
        }
        let online = if online {
            Some(JudgeClient::from_env()?)
        } else {
            None
        };
        Ok(Self {
            path,
            recordings,
            online,
        })
    }

    /// Score one matrix cell. Replays a recording when one exists; otherwise
    /// judges online (recording the response for future replays) or returns
    /// None offline.
    fn verdict(
        &mut self,
        cell: &str,
        original: &[ChatMessage],
        collected: &[ChatMessage],
    ) -> Result<Option<JudgeVerdict>> {
        let prompt = judge_prompt(original, collected);
        let key = judge_key(&prompt);
        if let Some(response) = self.recordings.get(&key) {
            return Ok(parse_judge_response(response));
        }
        let Some(client) = &self.online else {
            return Ok(None);
        };
        let response = client.judge(&prompt)?;
        self.append_record(&JudgeRecord {
            key: key.clone(),
            cell: cell.to_string(),
            model: client.model.0.clone(),
            response: response.clone(),
            note: None,
        })?;
        self.recordings.insert(key, response.clone());
        Ok(parse_judge_response(&response))
    }

    fn append_record(&self, record: &JudgeRecord) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut line = serde_json::to_string(record)?;
        line.push('\n');
        use std::io::Write as _;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("appending judge recording {}", self.path.display()))?;
        file.write_all(line.as_bytes())?;
        Ok(())
    }
}

impl JudgeClient {
    /// Provider config from the environment, following crates/agent
    /// conventions: AGENT_JUDGE_MODEL (or AGENT_ONLINE_MODEL) against an
    /// OpenAI-compatible endpoint, key from
    /// AGENT_API_KEY/ANTHROPIC_API_KEY/OPENROUTER_API_KEY.
    fn from_env() -> Result<Self> {
        let url = std::env::var("AGENT_JUDGE_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".into());
        let api_key = std::env::var("AGENT_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
            .or_else(|_| std::env::var("OPENROUTER_API_KEY"))
            .map_err(|_| {
                anyhow!(
                    "RUN_AGENT_ONLINE_EVAL=1 needs AGENT_API_KEY/ANTHROPIC_API_KEY/OPENROUTER_API_KEY"
                )
            })?;
        let model = Model(
            std::env::var("AGENT_JUDGE_MODEL")
                .or_else(|_| std::env::var("AGENT_ONLINE_MODEL"))
                .unwrap_or_else(|_| "openrouter/auto".into()),
        );
        let client = ProviderClient::new(ProviderConfig {
            url,
            api_key,
            model: model.clone(),
        });
        let runtime = tokio::runtime::Runtime::new()?;
        Ok(Self {
            client,
            model,
            runtime,
        })
    }

    fn judge(&self, prompt: &[ChatMessage]) -> Result<String> {
        let response = self
            .runtime
            .block_on(self.client.chat(&self.model, &[], prompt))?;
        Ok(response.content)
    }
}

fn judge_recording_path() -> Result<PathBuf> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("could not resolve repo root"))?;
    Ok(repo_root.join("evals/gc/judge/recorded.jsonl"))
}

/// Deterministic judge prompt: rubric + rendered before/after windows. No
/// message UUIDs, timestamps, or map iteration order leak in, so the prompt
/// (and therefore the recording key) is stable across runs.
fn judge_prompt(original: &[ChatMessage], collected: &[ChatMessage]) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(JUDGE_RUBRIC),
        ChatMessage::user(format!(
            "== FULL WINDOW (before GC) ==\n{}== COLLECTED WINDOW (after GC) ==\n{}",
            render_window(original),
            render_window(collected),
        )),
    ]
}

fn render_window(messages: &[ChatMessage]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (index, message) in messages.iter().enumerate() {
        let _ = write!(out, "#{index} {}", message.role);
        if let Some(id) = &message.tool_call_id {
            let _ = write!(out, " (result for {id})");
        }
        for call in message.tool_calls.as_deref().unwrap_or_default() {
            let _ = write!(
                out,
                " [calls {} {} as {}]",
                call.name,
                judge_preview(&call.arguments.to_string()),
                call.id
            );
        }
        out.push_str(": ");
        out.push_str(&judge_preview(message.content.as_deref().unwrap_or("")));
        out.push('\n');
    }
    out
}

fn judge_preview(input: &str) -> String {
    let mut out: String = input.chars().take(JUDGE_RENDER_CHARS).collect();
    if input.chars().count() > JUDGE_RENDER_CHARS {
        out.push_str("…[truncated]");
    }
    out
}

/// Recording key: content hash of the judge prompt text (roles + rendered
/// content only — never message UUIDs, which are freshly assigned every
/// run).
fn judge_key(prompt: &[ChatMessage]) -> String {
    let mut hasher = Sha256::new();
    for message in prompt {
        hasher.update(message.role.as_bytes());
        hasher.update([0]);
        hasher.update(message.content.as_deref().unwrap_or("").as_bytes());
        hasher.update([0]);
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Lenient JSON extraction: the strict-JSON instruction notwithstanding,
/// models wrap answers in prose/fences often enough that we take the
/// outermost brace span that parses.
fn parse_judge_response(response: &str) -> Option<JudgeVerdict> {
    let start = response.find('{')?;
    let end = response.rfind('}')?;
    serde_json::from_str(&response[start..=end]).ok()
}

// --- judge plumbing tests (offline; recorded/hand-written fixtures only) ----

/// The judge key must be stable across runs and independent of message
/// UUIDs (which are freshly assigned on every construction).
#[test]
fn gc_judge_key_is_deterministic_and_id_independent() {
    let build = || {
        (
            vec![
                ChatMessage::system("sys"),
                ChatMessage::user("do the thing"),
            ],
            vec![ChatMessage::system("sys")],
        )
    };
    let (original_a, collected_a) = build();
    let (original_b, collected_b) = build();
    // Same structural content, different UUIDs.
    assert_ne!(original_a[0].id, original_b[0].id);
    let key_a = judge_key(&judge_prompt(&original_a, &collected_a));
    let key_b = judge_key(&judge_prompt(&original_b, &collected_b));
    assert_eq!(key_a, key_b);

    let key_c = judge_key(&judge_prompt(&original_a, &original_a.clone()));
    assert_ne!(key_a, key_c, "different collected window, different key");
}

#[test]
fn gc_judge_parses_strict_and_wrapped_json_rejects_garbage() {
    let verdict = parse_judge_response(
        r#"{"task_goal_retained": true, "open_threads_retained": false, "no_orphaned_references": true}"#,
    )
    .expect("strict JSON parses");
    assert_eq!(verdict.score(), 2);
    assert_eq!(verdict.display(), "2/3");

    let wrapped = parse_judge_response(
        "Here is my assessment:\n```json\n{\"task_goal_retained\": true, \
         \"open_threads_retained\": true, \"no_orphaned_references\": true}\n```",
    )
    .expect("fenced JSON parses");
    assert_eq!(wrapped.display(), "3/3");

    assert_eq!(parse_judge_response("I think it looks fine."), None);
    assert_eq!(parse_judge_response("{\"task_goal_retained\": 3}"), None);
}

/// Round-trip through the recording format: a record written the way the
/// online path writes it is served back by a fresh offline book.
#[test]
fn gc_judge_offline_replays_recorded_response_and_misses_return_none() -> Result<()> {
    let original = vec![
        ChatMessage::system("You are a test agent."),
        ChatMessage::user("finish the report"),
        ChatMessage::assistant(Some("working on it".into()), vec![]),
    ];
    let collected = original[..2].to_vec();
    let key = judge_key(&judge_prompt(&original, &collected));

    let dir = std::env::temp_dir().join(format!("gc-judge-test-{}", std::process::id()));
    fs::create_dir_all(&dir)?;
    let path = dir.join("recorded.jsonl");
    let record = JudgeRecord {
        key,
        cell: "test|0.50|final|ring|ignore".into(),
        model: "test-judge".into(),
        response: r#"{"task_goal_retained": true, "open_threads_retained": true, "no_orphaned_references": false}"#.into(),
        note: Some("written by gc_judge_offline_replays_recorded_response".into()),
    };
    fs::write(&path, format!("{}\n", serde_json::to_string(&record)?))?;

    // Offline book (online=false regardless of environment).
    let mut book = JudgeBook::load(path.clone(), false)?;
    let verdict = book
        .verdict("test|0.50|final|ring|ignore", &original, &collected)?
        .expect("recorded response replays offline");
    assert_eq!(verdict.display(), "2/3");
    assert!(!verdict.no_orphaned_references);

    // A different collected window has no recording: offline miss is None,
    // never a provider call.
    let missing = book.verdict("test|other", &original, &original.clone())?;
    assert_eq!(missing, None);

    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// The shipped hand-written fixture (evals/gc/judge/recorded.jsonl) must
/// stay in sync with the judge prompt format: it pins the matrix cells it
/// covers, so this fails loudly (printing the expected keys) if the prompt
/// rendering or the synthetic cases change. Recorded entries are replayed
/// into the matrix judge column; every shipped entry must parse.
#[test]
fn gc_judge_shipped_fixture_replays_into_matrix_cells() -> Result<()> {
    let path = judge_recording_path()?;
    let content = fs::read_to_string(&path)
        .with_context(|| format!("reading shipped judge fixture {}", path.display()))?;
    let mut shipped_keys = BTreeSet::new();
    for line in content.lines().filter(|line| !line.trim().is_empty()) {
        let record: JudgeRecord = serde_json::from_str(line)?;
        assert!(
            parse_judge_response(&record.response).is_some(),
            "shipped judge recording {} must parse into a verdict",
            record.cell
        );
        shipped_keys.insert(record.key);
    }
    assert!(
        !shipped_keys.is_empty(),
        "expected at least one shipped judge recording"
    );

    // The cells the hand-written fixture pins: ring vs mark-sweep vs stack
    // on the long tool-heavy window at heavy pressure (cache=ignore,
    // timing=final).
    let mut book = JudgeBook::load(path, false)?;
    let cases = all_cases()?;
    let case = cases
        .iter()
        .find(|case| case.name == "synthetic:tool-heavy-long")
        .expect("synthetic:tool-heavy-long case exists");
    for strategy in &STRATEGIES {
        let metrics = evaluate(case, 0.35, Timing::Final, strategy, false, Some(&mut book))?;
        let prompt_key = {
            let gc = strategy.build(false);
            let run = run_timed(&case.prompt, metrics.budget, gc.as_ref(), Timing::Final);
            judge_key(&judge_prompt(&case.prompt, &run.collected))
        };
        assert!(
            metrics.judge.is_some(),
            "expected a shipped judge recording for cell {}|0.35|final|{}|ignore \
             (prompt key {prompt_key}); re-generate the hand-written fixture",
            case.name,
            metrics.strategy,
        );
    }
    Ok(())
}
