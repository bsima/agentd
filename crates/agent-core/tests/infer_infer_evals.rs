//! Infer-calls-infer cost/benefit eval harness (t-1342).
//!
//! The IR agent loop exposes an `infer` tool so the model can make a nested
//! Infer call directly (crates/agent-core/src/ir_agent.rs, `infer_tool` /
//! `infer_eval` blocks). This harness measures whether that multi-agent
//! structure *earns its cost*: each fixture task runs in two arms —
//!
//! - **single**: the parent model does everything itself;
//! - **sub-infer**: the parent delegates subtasks to a cheaper model via the
//!   `infer` tool;
//!
//! — and the score is read from the TRACE, not estimated: per-`InferResult`
//! token usage and `cost_micro_usd` (stamped from a fixture pricing table,
//! t-1334), the `RunUsage` rollup on `AgentDone`, and effect counts
//! (Infer/Eval/InferError). Parent-loop infers are distinguished from
//! sub-infers by `parent_op_id` (t-1347): sub-infer events carry the
//! dispatching turn Infer's op_id, parent-loop events carry none — see the
//! harness findings in evals/infer-infer/README.md.
//!
//! Offline (the default, credential-free, deterministic): both arms run
//! against a scripted provider that meters usage from the *actual prompts*
//! it receives (`estimate_tokens`, the same chars/3 estimator the runtime
//! budgets with), so context growth — tool-call argument duplication,
//! history re-send per turn — is what drives the numbers, exactly the
//! quantities the mechanism changes. Online
//! (`RUN_AGENT_ONLINE_EVAL=1`, the evals/ convention): the same fixtures run
//! against a real provider with every exchange recorded to
//! `evals/infer-infer/recorded.jsonl` keyed by a content hash of
//! (model + prompt), and replayed from there by default — the same
//! record/replay pattern as the GC judge (evals/gc/judge/recorded.jsonl).
//!
//! The fixture set intentionally contains shapes where decomposition
//! plausibly helps AND where it plausibly hurts; the expected winner is
//! asserted per fixture so the structural economics of the mechanism are
//! pinned, not just printed.

use agent_core::FinishReason;
use agent_core::{
    agent_loop_ir, estimate_tokens, run_ir_sequential, ChatMessage, ChatProvider, EvalConfig,
    Event, GcMode, GcTiming, Model, PassiveHydrationConfig, Pricing, PricingTable, ProviderClient,
    ProviderConfig, Response, RunUsage, SeqConfig, SourceRegistry, ToolCall, TraceLogger,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Model ids used by the offline arms. They exist only in the scripted
/// provider and the fixture pricing table below.
const PARENT_MODEL: &str = "eval-parent";
const CHILD_MODEL: &str = "eval-child";
/// A plausibly-hallucinated model id: the scripted provider fails it, so
/// the error-binding path (t-1222) is exercised end-to-end.
const DEAD_MODEL: &str = "eval-dead-model";

const MAX_TURNS: usize = 8;

/// Fixture pricing: a frontier-class parent vs a cheap delegate, USD per
/// Mtok. The 5x input/output spread within a model and the 20-25x spread
/// between models are what make delegation economics non-trivial.
fn pricing_table() -> PricingTable {
    let mut table = PricingTable::default();
    table.insert(PARENT_MODEL, Pricing::from_usd_per_mtok(3.0, 15.0).unwrap());
    table.insert(CHILD_MODEL, Pricing::from_usd_per_mtok(0.15, 0.60).unwrap());
    table
}

// --- metered scripted provider ----------------------------------------------

/// One scripted provider turn. Usage is metered at call time: input tokens
/// from the actual received prompt, output tokens from the scripted content
/// plus serialized tool-call arguments (providers bill emitted tool calls
/// as output), so arm comparisons reflect real context growth.
#[derive(Debug, Clone)]
struct ScriptTurn {
    content: String,
    tool_calls: Vec<ToolCall>,
    error: Option<String>,
}

fn text(content: impl Into<String>) -> ScriptTurn {
    ScriptTurn {
        content: content.into(),
        tool_calls: Vec::new(),
        error: None,
    }
}

fn calls(tool_calls: Vec<ToolCall>) -> ScriptTurn {
    ScriptTurn {
        content: String::new(),
        tool_calls,
        error: None,
    }
}

fn text_and_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> ScriptTurn {
    ScriptTurn {
        content: content.into(),
        tool_calls,
        error: None,
    }
}

fn provider_error(message: impl Into<String>) -> ScriptTurn {
    ScriptTurn {
        content: String::new(),
        tool_calls: Vec::new(),
        error: Some(message.into()),
    }
}

fn infer_call(id: &str, model: &str, prompt: String) -> ToolCall {
    ToolCall::new(
        id,
        "infer",
        serde_json::json!({ "model": model, "prompt": prompt }),
    )
}

/// A by-reference delegation (t-1344): the material is named by the ids of
/// prior tool calls, never copied into the arguments.
fn infer_ref_call(id: &str, model: &str, prompt: &str, context_refs: &[&str]) -> ToolCall {
    ToolCall::new(
        id,
        "infer",
        serde_json::json!({ "model": model, "prompt": prompt, "context_refs": context_refs }),
    )
}

fn shell_call(id: &str, command: &str) -> ToolCall {
    ToolCall::new(id, "shell", serde_json::json!({ "command": command }))
}

/// What one provider call actually received — kept for the mechanism
/// probes (child context assembly, advertised toolset).
#[derive(Debug, Clone)]
struct RecordedCall {
    model: String,
    messages: Vec<ChatMessage>,
    tools: Vec<agent_core::provider::ToolSpec>,
}

struct MeteredProvider {
    scripts: Mutex<BTreeMap<String, VecDeque<ScriptTurn>>>,
    calls: Mutex<Vec<RecordedCall>>,
}

impl MeteredProvider {
    fn new(script: &[(String, ScriptTurn)]) -> Self {
        let mut scripts: BTreeMap<String, VecDeque<ScriptTurn>> = BTreeMap::new();
        for (model, turn) in script {
            scripts
                .entry(model.clone())
                .or_default()
                .push_back(turn.clone());
        }
        Self {
            scripts: Mutex::new(scripts),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn recorded_calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }
}

fn approx_output_tokens(turn: &ScriptTurn) -> u32 {
    let mut chars = turn.content.chars().count();
    for call in &turn.tool_calls {
        chars += call.name.chars().count() + call.arguments.to_string().chars().count();
    }
    chars.div_ceil(3).max(1) as u32
}

#[async_trait]
impl ChatProvider for MeteredProvider {
    async fn chat(
        &self,
        model: &Model,
        tools: &[agent_core::provider::ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        self.calls.lock().unwrap().push(RecordedCall {
            model: model.0.clone(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
        });
        let turn = self
            .scripts
            .lock()
            .unwrap()
            .get_mut(&model.0)
            .and_then(VecDeque::pop_front)
            .ok_or_else(|| anyhow!("script exhausted for model {}", model.0))?;
        if let Some(error) = turn.error {
            return Err(anyhow!("{error}"));
        }
        let input_tokens = estimate_tokens(messages) as u32;
        let output_tokens = approx_output_tokens(&turn);
        Ok(Response {
            content: turn.content,
            tool_calls: turn.tool_calls,
            finish_reason: Some(FinishReason::Stop),
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            metadata: Default::default(),
        })
    }
}

// --- fixtures ----------------------------------------------------------------

/// The structural cost expectation a fixture pins.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Expected {
    /// The sub-infer arm must come out cheaper.
    SubWins,
    /// The single arm must come out cheaper.
    SingleWins,
    /// The single arm still wins, but the sub arm must land within this
    /// ratio of it — delegation as a rounding error, not a structural tax.
    SingleWinsWithin(f64),
}

struct Fixture {
    name: &'static str,
    single_prompt: Vec<ChatMessage>,
    sub_prompt: Vec<ChatMessage>,
    single_script: Vec<(String, ScriptTurn)>,
    sub_script: Vec<(String, ScriptTurn)>,
    /// The final answer must contain every needle (fixture-defined task
    /// success; offline this validates the wiring, online it scores the
    /// real model).
    success_needles: Vec<&'static str>,
    /// The structural expectation this fixture pins.
    expected: Expected,
    /// How many sub-infer calls are expected to FAIL in the sub arm
    /// (error-binding fixtures).
    expected_sub_infer_errors: usize,
    rationale: &'static str,
}

/// Deterministic filler: repeatable pseudo-prose, no RNG (same generator
/// as gc_evals).
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

fn quarterly_report(index: usize, fact: &str) -> String {
    format!(
        "=== Quarterly report {index} ===\n{}\nKEY FACT: {fact}\n{}",
        lorem(index * 7 + 1, 1000),
        lorem(index * 13 + 5, 1200),
    )
}

/// Shell command producing quarterly report `index` on stdout (~3KB of
/// deterministic filler around one KEY FACT line). By-reference fixtures
/// fetch material through the shell tool so it enters the conversation as a
/// tool result with a model-minted id — the thing `context_refs` can name.
/// The command is short; its OUTPUT is the fat object.
fn report_command(index: usize, fact: &str) -> String {
    format!(
        "seq -f \"report {index} filler paragraph %g alpha bravo charlie delta echo foxtrot\" 1 30 \
         && echo \"KEY FACT: {fact}\" \
         && seq -f \"report {index} appendix filler %g golf hotel india juliet kilo lima\" 1 30"
    )
}

/// Shell command standing in for migration step 7: ~11KB of repeated noise
/// followed by the one RESULT line that matters.
fn migration_step7_command() -> String {
    "seq -f \"ERROR[%03g]: frobnicator stage %g panicked: widget overflow (retrying)\" 0 159 \
     && echo \"RESULT: step7=ok\""
        .into()
}

fn brainstorm_evaluation() -> String {
    const CANDIDATES: [&str; 20] = [
        "aurora", "basil", "cinder", "dune", "ember", "flint", "gale", "harbor", "iris", "juniper",
        "krill", "lumen", "maple", "nimbus", "onyx", "prism", "quarry", "reef", "sable", "zephyr",
    ];
    let mut out = String::from("Evaluation of all 20 candidates against the constraints:\n");
    for (index, name) in CANDIDATES.iter().enumerate() {
        out.push_str(&format!("- {name}: {}\n", lorem(index * 11 + 3, 260)));
    }
    out.push_str("CHOSEN: zephyr — short, unambiguous, memorable.\n");
    out
}

/// Build the fixture set. `child_model` is the delegate model id the
/// sub-infer arm instructs the parent to use: the offline matrix passes
/// [`CHILD_MODEL`]; the recorded/online matrix passes the real cheap model
/// id so prompts (and therefore recording keys) name a model that exists.
fn fixtures(child_model: &str) -> Vec<Fixture> {
    let mut fixtures = Vec::new();

    // (b in the task statement) simple question: sub-infer indirection is
    // pure overhead.
    {
        let question = "What is the capital of Portugal? Answer with just the city name.";
        let single_prompt = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user(question),
        ];
        let sub_prompt = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user(format!(
                "{question} Delegate the question to the cheaper model \
                 \"{child_model}\" via the infer tool, then relay its answer."
            )),
        ];
        fixtures.push(Fixture {
            name: "simple-question",
            single_prompt,
            sub_prompt,
            single_script: vec![(PARENT_MODEL.into(), text("Lisbon"))],
            sub_script: vec![
                (
                    PARENT_MODEL.into(),
                    calls(vec![infer_call("call-sq-1", child_model, question.into())]),
                ),
                (child_model.into(), text("Lisbon")),
                (PARENT_MODEL.into(), text("Lisbon")),
            ],
            success_needles: vec!["Lisbon"],
            expected: Expected::SingleWins,
            expected_sub_infer_errors: 0,
            rationale: "delegation round-trip (args out + extra turn) on a one-shot answer",
        });
    }

    // (a) synthesis over 3 long documents, BY COPY — the pre-t-1344 path,
    // kept on purpose: passing material through the prompt still works and
    // still costs. The by-reference variant below is the comparison.
    {
        let docs = [
            quarterly_report(1, "revenue grew 12%"),
            quarterly_report(2, "churn fell to 3%"),
            quarterly_report(3, "headcount stayed at 84"),
        ];
        let corpus = docs.join("\n\n");
        let task = format!(
            "Synthesize the three quarterly reports into a short paragraph \
             covering each report's key fact.\n\n{corpus}"
        );
        let synthesis = "Synthesis: revenue grew 12%, churn fell to 3%, and headcount \
                         stayed at 84. The quarter improved on both growth and retention.";
        let doc_prompt = |doc: &str| {
            format!("Summarize this report in one sentence, preserving its key fact.\n\n{doc}")
        };
        fixtures.push(Fixture {
            name: "doc-synthesis-by-copy",
            single_prompt: vec![
                ChatMessage::system("You are a research assistant."),
                ChatMessage::user(task.clone()),
            ],
            sub_prompt: vec![
                ChatMessage::system("You are a research assistant."),
                ChatMessage::user(format!(
                    "{task}\n\nDelegate each report to the cheaper model \
                     \"{child_model}\": call the infer tool once per report with the \
                     full report text in the prompt, then synthesize the summaries."
                )),
            ],
            single_script: vec![(PARENT_MODEL.into(), text(synthesis))],
            sub_script: vec![
                (
                    PARENT_MODEL.into(),
                    calls(vec![
                        infer_call("call-doc-1", child_model, doc_prompt(&docs[0])),
                        infer_call("call-doc-2", child_model, doc_prompt(&docs[1])),
                        infer_call("call-doc-3", child_model, doc_prompt(&docs[2])),
                    ]),
                ),
                (
                    child_model.into(),
                    text("Report 1 key point: revenue grew 12%."),
                ),
                (
                    child_model.into(),
                    text("Report 2 key point: churn fell to 3%."),
                ),
                (
                    child_model.into(),
                    text("Report 3 key point: headcount stayed at 84."),
                ),
                (PARENT_MODEL.into(), text(synthesis)),
            ],
            success_needles: vec!["12%", "3%", "84"],
            expected: Expected::SingleWins,
            expected_sub_infer_errors: 0,
            rationale: "BY COPY, the old path: each doc is copied into the tool-call \
                        arguments (billed as parent output at 5x input rate) and then \
                        rides in parent history every later turn",
        });
    }

    // (a') the same synthesis BY REFERENCE (t-1344): the reports are fetched
    // through the shell tool (fat tool results with model-minted ids) and
    // delegated via context_refs in the same assistant turn — refs resolve
    // against results appended earlier in the batch. The material never
    // transits parent output and never rides twice in parent history, so
    // delegating the reading costs a rounding error instead of 7.7x.
    {
        let fetches = [
            report_command(1, "revenue grew 12%"),
            report_command(2, "churn fell to 3%"),
            report_command(3, "headcount stayed at 84"),
        ];
        let task = format!(
            "Fetch the three quarterly reports by running each of these shell \
             commands as its own shell tool call:\n1. {}\n2. {}\n3. {}\n\
             Then synthesize the reports into a short paragraph covering each \
             report's key fact.",
            fetches[0], fetches[1], fetches[2]
        );
        let synthesis = "Synthesis: revenue grew 12%, churn fell to 3%, and headcount \
                         stayed at 84. The quarter improved on both growth and retention.";
        let child_prompt = "Summarize the referenced report in one sentence, preserving \
                            its KEY FACT line.";
        let batch = |with_infers: bool| {
            let mut batch = vec![
                shell_call("call-ds-r1", &fetches[0]),
                shell_call("call-ds-r2", &fetches[1]),
                shell_call("call-ds-r3", &fetches[2]),
            ];
            if with_infers {
                batch.extend([
                    infer_ref_call("call-ds-i1", child_model, child_prompt, &["call-ds-r1"]),
                    infer_ref_call("call-ds-i2", child_model, child_prompt, &["call-ds-r2"]),
                    infer_ref_call("call-ds-i3", child_model, child_prompt, &["call-ds-r3"]),
                ]);
            }
            batch
        };
        fixtures.push(Fixture {
            name: "doc-synthesis",
            single_prompt: vec![
                ChatMessage::system("You are a research assistant."),
                ChatMessage::user(task.clone()),
            ],
            sub_prompt: vec![
                ChatMessage::system("You are a research assistant."),
                ChatMessage::user(format!(
                    "{task}\n\nDelegate each report's summary to the cheaper model \
                     \"{child_model}\" in the same turn as the fetches: call the infer \
                     tool once per report with context_refs naming the shell call id \
                     that fetched it. Never paste report text into the prompt. Then \
                     synthesize the summaries."
                )),
            ],
            single_script: vec![
                (PARENT_MODEL.into(), calls(batch(false))),
                (PARENT_MODEL.into(), text(synthesis)),
            ],
            sub_script: vec![
                (PARENT_MODEL.into(), calls(batch(true))),
                (
                    child_model.into(),
                    text("Report 1 key point: revenue grew 12%."),
                ),
                (
                    child_model.into(),
                    text("Report 2 key point: churn fell to 3%."),
                ),
                (
                    child_model.into(),
                    text("Report 3 key point: headcount stayed at 84."),
                ),
                (PARENT_MODEL.into(), text(synthesis)),
            ],
            success_needles: vec!["12%", "3%", "84"],
            expected: Expected::SingleWinsWithin(1.3),
            expected_sub_infer_errors: 0,
            rationale: "BY REFERENCE: the corpus enters history once as tool results \
                        (parent input rate, both arms) and reaches the children without \
                        transiting parent output — the remaining sub overhead is the \
                        cheap child reads, not a structural copy tax",
        });
    }

    // (c) multi-step task with a noisy middle step, BY REFERENCE (t-1344):
    // the noisy dump is a shell tool result; the sub arm hands it to a
    // cheap child via context_refs instead of the parent analyzing it
    // inline. Containment is real now — the dump never transits parent
    // output, and the parent's context gains a one-line status instead of
    // a verbose inline digest.
    {
        let step7 = migration_step7_command();
        let task = format!(
            "Run migration step 7 via the shell tool: {step7}\n\
             Determine step 7's status from its output, then run step 8 \
             (`echo step8-done`) and step 9 (`echo step9-done`) via the shell \
             tool, then report the status of steps 7, 8, and 9."
        );
        let digest = format!(
            "The log is 160 repeated frobnicator panics (widget overflow, all retried) \
             followed by a terminal RESULT line reporting step7=ok, so the noise is \
             benign and step 7 succeeded. {}",
            lorem(97, 700)
        );
        let final_report = "Migration status: step7=ok (from the log), step8-done, step9-done.";
        fixtures.push(Fixture {
            name: "noisy-middle-step",
            single_prompt: vec![
                ChatMessage::system("You are an operations agent."),
                ChatMessage::user(task.clone()),
            ],
            sub_prompt: vec![
                ChatMessage::system("You are an operations agent."),
                ChatMessage::user(format!(
                    "{task} Delegate the log analysis to the cheaper model \
                     \"{child_model}\" with the infer tool, passing the log by \
                     reference: context_refs naming the step-7 shell call id, never \
                     the log text itself. Dispatch the delegation and steps 8 and 9 \
                     in the same turn."
                )),
            ],
            single_script: vec![
                (
                    PARENT_MODEL.into(),
                    calls(vec![shell_call("call-nm-s7", &step7)]),
                ),
                // The parent analyzes 160 noise lines inline: a verbose
                // digest at parent OUTPUT rates, which then rides history.
                (
                    PARENT_MODEL.into(),
                    text_and_calls(
                        digest,
                        vec![
                            shell_call("call-nm-s8", "echo step8-done"),
                            shell_call("call-nm-s9", "echo step9-done"),
                        ],
                    ),
                ),
                (PARENT_MODEL.into(), text(final_report)),
            ],
            sub_script: vec![
                (
                    PARENT_MODEL.into(),
                    calls(vec![shell_call("call-nm-s7", &step7)]),
                ),
                (
                    PARENT_MODEL.into(),
                    calls(vec![
                        infer_ref_call(
                            "call-nm-digest",
                            child_model,
                            "Report only the final RESULT line status from the \
                             referenced migration log.",
                            &["call-nm-s7"],
                        ),
                        shell_call("call-nm-s8", "echo step8-done"),
                        shell_call("call-nm-s9", "echo step9-done"),
                    ]),
                ),
                (child_model.into(), text("step7=ok")),
                (PARENT_MODEL.into(), text(final_report)),
            ],
            success_needles: vec!["step7=ok", "step8", "step9"],
            expected: Expected::SubWins,
            expected_sub_infer_errors: 0,
            rationale: "BY REFERENCE the containment is real: the dump stays a tool \
                        result (input rate, both arms), the child digests it at cheap \
                        rates, and the parent trades a verbose inline digest (output \
                        rate + history residue) for a one-line status",
        });
    }

    // Generation offload: the one shape where the mechanism wins — the
    // delegated subtask is generation-heavy with a SHORT prompt, so the
    // expensive model never pays output rates for the long text, only
    // input rates to read it back.
    {
        let instruction = "Evaluate all 20 candidate names (aurora, basil, cinder, dune, \
                           ember, flint, gale, harbor, iris, juniper, krill, lumen, maple, \
                           nimbus, onyx, prism, quarry, reef, sable, zephyr) against the \
                           constraints: memorable, short, unambiguous, no trademark \
                           collisions. Write the full per-candidate evaluation, then end \
                           with 'CHOSEN: <name> — <reason>'.";
        let brainstorm = brainstorm_evaluation();
        fixtures.push(Fixture {
            name: "generation-offload",
            single_prompt: vec![
                ChatMessage::system("You are a naming consultant."),
                ChatMessage::user(instruction),
            ],
            sub_prompt: vec![
                ChatMessage::system("You are a naming consultant."),
                ChatMessage::user(format!(
                    "{instruction} Delegate the full written evaluation to the cheaper \
                     model \"{child_model}\" via one infer call, then state the winner."
                )),
            ],
            single_script: vec![(PARENT_MODEL.into(), text(brainstorm.clone()))],
            sub_script: vec![
                (
                    PARENT_MODEL.into(),
                    calls(vec![infer_call(
                        "call-gen-1",
                        child_model,
                        instruction.to_string(),
                    )]),
                ),
                (child_model.into(), text(brainstorm)),
                (
                    PARENT_MODEL.into(),
                    text("CHOSEN: zephyr — short, unambiguous, memorable."),
                ),
            ],
            success_needles: vec!["zephyr"],
            expected: Expected::SubWins,
            expected_sub_infer_errors: 0,
            rationale: "output-rate arbitrage: the long text is generated at cheap output \
                        rates and only READ back at parent input rates; the delegation \
                        prompt itself is short so argument-copying is negligible",
        });
    }

    // Error binding: the parent delegates to a dead model id, the failure
    // binds as a tool result (t-1222), and the parent recovers. Measures
    // what a hallucinated model id costs.
    {
        let question = "What is the capital of Portugal? Answer with just the city name.";
        fixtures.push(Fixture {
            name: "child-error-recovery",
            single_prompt: vec![
                ChatMessage::system("You are a helpful assistant."),
                ChatMessage::user(question),
            ],
            sub_prompt: vec![
                ChatMessage::system("You are a helpful assistant."),
                ChatMessage::user(format!(
                    "{question} Delegate to the fast model \"{DEAD_MODEL}\" via the \
                     infer tool; if that fails, retry with \"{child_model}\"."
                )),
            ],
            single_script: vec![(PARENT_MODEL.into(), text("Lisbon"))],
            sub_script: vec![
                (
                    PARENT_MODEL.into(),
                    calls(vec![infer_call("call-er-1", DEAD_MODEL, question.into())]),
                ),
                (
                    DEAD_MODEL.into(),
                    provider_error(format!(
                        "model not found (404): unknown model id {DEAD_MODEL}"
                    )),
                ),
                (
                    PARENT_MODEL.into(),
                    calls(vec![infer_call("call-er-2", child_model, question.into())]),
                ),
                (child_model.into(), text("Lisbon")),
                (PARENT_MODEL.into(), text("Lisbon")),
            ],
            success_needles: vec!["Lisbon"],
            expected: Expected::SingleWins,
            expected_sub_infer_errors: 1,
            rationale: "a hallucinated model id costs a full delegation round-trip (args \
                        out, error tool result, retry turn) before the recovery",
        });
    }

    fixtures
}

// --- arm runner + trace-derived metrics ---------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Single,
    SubInfer,
}

impl Arm {
    fn label(&self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::SubInfer => "sub-infer",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ArmMetrics {
    arm: &'static str,
    /// Parent-loop infer calls (= turns taken).
    parent_infers: usize,
    /// Nested infer-tool calls.
    sub_infers: usize,
    eval_calls: usize,
    infer_errors: usize,
    usage: RunUsage,
    parent_cost_micro_usd: u64,
    sub_cost_micro_usd: u64,
    success: bool,
}

/// Run one arm of one fixture: build the loop machine, execute it against
/// the given provider, emit `AgentDone` through the same logger (so the
/// t-1334 rollup stamps it), and return the final answer text plus the
/// full trace.
async fn run_arm(
    provider: Arc<dyn ChatProvider>,
    parent_model: &str,
    prompt: Vec<ChatMessage>,
    pricing: PricingTable,
) -> Result<(String, Vec<Event>)> {
    let trace_path =
        std::env::temp_dir().join(format!("infer-infer-eval-{}.jsonl", Uuid::new_v4()));
    let trace = TraceLogger::new(Uuid::new_v4().to_string(), trace_path.clone());
    let config = SeqConfig {
        approvals: Default::default(),
        tools: Default::default(),
        provider,
        hydration: SourceRegistry::new(),
        passive_hydration: PassiveHydrationConfig::default(),
        trace: trace.clone(),
        eval: EvalConfig::default(),
        replay: None,
        trace_full_prompt_ir: false,
        trace_full_payloads: false,
        gc: GcMode::None,
        gc_threshold: 0.85,
        gc_log: false,
        gc_timing: GcTiming::Threshold,
        context_budget: 200_000,
        pricing,
    };
    let machine = agent_loop_ir(Model(parent_model.into()), prompt, MAX_TURNS);
    let (value, _machine) = run_ir_sequential(&config, machine).await?;
    trace
        .emit(&Event::AgentDone {
            run_id: trace.run_id().into(),
            usage: None,
            timestamp: Utc::now(),
        })
        .await?;
    let events = TraceLogger::read_events(&trace_path).await?;
    let _ = fs::remove_file(&trace_path);
    let content = value["content"].as_str().unwrap_or_default().to_string();
    Ok((content, events))
}

fn metrics_from_events(
    arm: Arm,
    events: &[Event],
    final_content: &str,
    success_needles: &[&str],
) -> Result<ArmMetrics> {
    let mut sub_by_op: HashMap<u64, bool> = HashMap::new();
    let mut metrics = ArmMetrics {
        arm: arm.label(),
        parent_infers: 0,
        sub_infers: 0,
        eval_calls: 0,
        infer_errors: 0,
        usage: RunUsage::default(),
        parent_cost_micro_usd: 0,
        sub_cost_micro_usd: 0,
        success: success_needles
            .iter()
            .all(|needle| final_content.contains(needle)),
    };
    let mut done_usage: Option<RunUsage> = None;
    for event in events {
        match event {
            // A sub-infer carries the dispatching parent Infer's op_id as
            // parent_op_id (t-1347); parent-loop infers carry none. The
            // linkage must point at a parent-loop InferCall already seen —
            // attribution is structural, not effect-site decoding.
            Event::InferCall {
                op_id,
                parent_op_id,
                ..
            } => {
                let sub = match parent_op_id {
                    Some(parent) => {
                        anyhow::ensure!(
                            sub_by_op.get(parent) == Some(&false),
                            "sub-infer {op_id} must link to a parent-loop InferCall, \
                             got parent_op_id {parent}"
                        );
                        true
                    }
                    None => false,
                };
                sub_by_op.insert(*op_id, sub);
                if sub {
                    metrics.sub_infers += 1;
                } else {
                    metrics.parent_infers += 1;
                }
            }
            Event::InferResult {
                op_id,
                cost_micro_usd,
                ..
            } => {
                let sub = *sub_by_op
                    .get(op_id)
                    .ok_or_else(|| anyhow!("InferResult {op_id} without InferCall"))?;
                if let Some(cost) = cost_micro_usd {
                    if sub {
                        metrics.sub_cost_micro_usd += cost;
                    } else {
                        metrics.parent_cost_micro_usd += cost;
                    }
                }
            }
            Event::InferError { .. } => metrics.infer_errors += 1,
            Event::EvalCall { .. } => metrics.eval_calls += 1,
            Event::AgentDone { usage, .. } => done_usage = usage.clone(),
            _ => {}
        }
    }
    metrics.usage = done_usage
        .ok_or_else(|| anyhow!("trace has no AgentDone usage rollup (t-1334 instrument)"))?;
    Ok(metrics)
}

async fn run_offline_arm(fixture: &Fixture, arm: Arm) -> Result<ArmMetrics> {
    let (prompt, script) = match arm {
        Arm::Single => (fixture.single_prompt.clone(), &fixture.single_script),
        Arm::SubInfer => (fixture.sub_prompt.clone(), &fixture.sub_script),
    };
    let provider = Arc::new(MeteredProvider::new(script));
    let (content, events) = run_arm(provider, PARENT_MODEL, prompt, pricing_table()).await?;
    metrics_from_events(arm, &events, &content, &fixture.success_needles)
}

// --- table -------------------------------------------------------------------

fn print_header() {
    println!(
        "{:<22} {:<10} {:>5} {:>4} {:>5} {:>4} {:>8} {:>8} {:>8} {:>11} {:>11} {:>11} {:>3}",
        "fixture",
        "arm",
        "turns",
        "sub",
        "evals",
        "errs",
        "in_tok",
        "out_tok",
        "tot_tok",
        "cost",
        "parent$",
        "sub$",
        "ok"
    );
}

fn format_cost(cost: Option<u64>) -> String {
    cost.map_or_else(|| "-".into(), agent_core::format_micro_usd)
}

fn print_metrics(fixture: &str, metrics: &ArmMetrics) {
    println!(
        "{:<22} {:<10} {:>5} {:>4} {:>5} {:>4} {:>8} {:>8} {:>8} {:>11} {:>11} {:>11} {:>3}",
        fixture,
        metrics.arm,
        metrics.parent_infers,
        metrics.sub_infers,
        metrics.eval_calls,
        metrics.infer_errors,
        metrics.usage.input_tokens,
        metrics.usage.output_tokens,
        metrics.usage.total_tokens,
        format_cost(metrics.usage.cost_micro_usd),
        format_cost(Some(metrics.parent_cost_micro_usd)),
        format_cost(Some(metrics.sub_cost_micro_usd)),
        if metrics.success { "yes" } else { "NO" }
    );
}

fn print_verdict(fixture: &Fixture, single: &ArmMetrics, sub: &ArmMetrics) {
    let (single_cost, sub_cost) = (
        single.usage.cost_micro_usd.unwrap_or(0),
        sub.usage.cost_micro_usd.unwrap_or(0),
    );
    let (winner, cheap, dear) = if sub_cost < single_cost {
        ("sub-infer", sub_cost, single_cost)
    } else {
        ("single", single_cost, sub_cost)
    };
    let ratio = if cheap == 0 {
        f64::NAN
    } else {
        dear as f64 / cheap as f64
    };
    println!(
        "  -> {winner} wins on cost: {ratio:.1}x cheaper (saves {}); tokens {} vs {}\n     {}",
        agent_core::format_micro_usd(dear - cheap),
        single.usage.total_tokens,
        sub.usage.total_tokens,
        fixture.rationale
    );
}

// --- the offline matrix --------------------------------------------------------

/// The comparison matrix: every fixture x arm, scored from the trace.
/// Determinism is asserted (two runs, identical metrics), fixture success
/// is asserted for both arms (the wiring must model a *completed* task),
/// and the expected cost winner is asserted so the structural economics
/// of the mechanism are pinned.
#[tokio::test]
async fn infer_infer_cost_matrix() -> Result<()> {
    print_header();
    for fixture in fixtures(CHILD_MODEL) {
        let single = run_offline_arm(&fixture, Arm::Single).await?;
        let single_again = run_offline_arm(&fixture, Arm::Single).await?;
        assert_eq!(
            single, single_again,
            "{}: single arm must be deterministic",
            fixture.name
        );
        let sub = run_offline_arm(&fixture, Arm::SubInfer).await?;
        let sub_again = run_offline_arm(&fixture, Arm::SubInfer).await?;
        assert_eq!(
            sub, sub_again,
            "{}: sub-infer arm must be deterministic",
            fixture.name
        );

        print_metrics(fixture.name, &single);
        print_metrics(fixture.name, &sub);
        print_verdict(&fixture, &single, &sub);

        assert!(
            single.success && sub.success,
            "{}: both arms must complete the task (single={}, sub={})",
            fixture.name,
            single.success,
            sub.success
        );
        assert_eq!(
            single.sub_infers, 0,
            "{}: the single arm must not delegate",
            fixture.name
        );
        assert!(
            sub.sub_infers >= 1,
            "{}: the sub-infer arm must delegate at least once",
            fixture.name
        );
        assert_eq!(
            sub.infer_errors, fixture.expected_sub_infer_errors,
            "{}: expected {} bound child errors",
            fixture.name, fixture.expected_sub_infer_errors
        );
        assert_eq!(single.infer_errors, 0, "{}", fixture.name);

        // Cost integrity: every successful call is costed (the fixture
        // pricing table covers every scripted model) and the per-class
        // split sums to the AgentDone rollup.
        for metrics in [&single, &sub] {
            assert_eq!(
                metrics.usage.uncosted_infer_calls, 0,
                "{}: every InferResult must be costed",
                fixture.name
            );
            assert_eq!(
                metrics.usage.cost_micro_usd,
                Some(metrics.parent_cost_micro_usd + metrics.sub_cost_micro_usd),
                "{}: per-class cost split must sum to the rollup",
                fixture.name
            );
            // Failed calls have no InferResult, so the success count
            // excludes them — and since t-1347 the attempts are counted
            // apart in failed_infer_calls, so nothing vanishes.
            assert_eq!(
                metrics.usage.infer_calls as usize,
                metrics.parent_infers + metrics.sub_infers - metrics.infer_errors,
                "{}: infer_calls counts successful infers",
                fixture.name
            );
            assert_eq!(
                metrics.usage.failed_infer_calls as usize, metrics.infer_errors,
                "{}: failed attempts are counted in the rollup",
                fixture.name
            );
        }

        let single_cost = single.usage.cost_micro_usd.unwrap();
        let sub_cost = sub.usage.cost_micro_usd.unwrap();
        match fixture.expected {
            Expected::SubWins => assert!(
                sub_cost < single_cost,
                "{}: expected the sub-infer arm to win on cost ({} vs {})",
                fixture.name,
                sub_cost,
                single_cost
            ),
            Expected::SingleWins => assert!(
                sub_cost > single_cost,
                "{}: expected the single arm to win on cost ({} vs {})",
                fixture.name,
                single_cost,
                sub_cost
            ),
            Expected::SingleWinsWithin(ratio) => {
                assert!(
                    sub_cost > single_cost,
                    "{}: expected the single arm to win on cost ({} vs {})",
                    fixture.name,
                    single_cost,
                    sub_cost
                );
                let actual = sub_cost as f64 / single_cost as f64;
                assert!(
                    actual <= ratio,
                    "{}: expected the sub-infer arm within {ratio}x of single, got {actual:.2}x \
                     ({sub_cost} vs {single_cost})",
                    fixture.name
                );
            }
        }
    }
    Ok(())
}

// --- mechanism probes ----------------------------------------------------------
//
// These pin what the sub-infer mechanism actually gives the model today.
// Where a probe pins a deficiency, the assertion documents CURRENT
// behavior on purpose: fixing the mechanism should flip the probe, and the
// probe failing is the signal to update the eval alongside the fix.

/// Without `context_refs` the child's context is still exactly one bare
/// user message built from the arguments — the by-copy path is unchanged
/// (and still costs; see the doc-synthesis-by-copy fixture). Parent history
/// never leaks into the child implicitly.
#[tokio::test]
async fn probe_child_context_without_refs_is_one_bare_user_message() -> Result<()> {
    let script = vec![
        (
            PARENT_MODEL.to_string(),
            calls(vec![infer_call(
                "call-p1",
                CHILD_MODEL,
                "sub question".into(),
            )]),
        ),
        (CHILD_MODEL.to_string(), text("sub answer")),
        (PARENT_MODEL.to_string(), text("done")),
    ];
    let provider = Arc::new(MeteredProvider::new(&script));
    let prompt = vec![
        ChatMessage::system("parent system prompt"),
        ChatMessage::user("use infer"),
    ];
    let (content, _events) =
        run_arm(provider.clone(), PARENT_MODEL, prompt, pricing_table()).await?;
    assert_eq!(content, "done");

    let child_call = provider
        .recorded_calls()
        .into_iter()
        .find(|call| call.model == CHILD_MODEL)
        .expect("child call recorded");
    assert_eq!(
        child_call.messages.len(),
        1,
        "child prompt is a single message"
    );
    assert_eq!(child_call.messages[0].role, "user");
    assert_eq!(
        child_call.messages[0].content.as_deref(),
        Some("sub question")
    );
    assert!(
        !child_call
            .messages
            .iter()
            .any(|message| message.role == "system"),
        "no system prompt travels to the child"
    );
    Ok(())
}

/// Fix pin (t-1344, findings 1+2): with `context_refs` the referenced tool
/// result is assembled into the child's messages at dispatch — the material
/// reaches the child WITHOUT transiting parent output tokens, and the infer
/// arguments retained in parent history stay small (refs + prompt), so the
/// material lives in parent history exactly once (the original tool
/// result).
#[tokio::test]
async fn probe_context_refs_deliver_material_without_argument_copies() -> Result<()> {
    let material_command = "seq -f \"needle-material line %g\" 1 40";
    let script = vec![
        (
            PARENT_MODEL.to_string(),
            calls(vec![shell_call("call-sh", material_command)]),
        ),
        (
            PARENT_MODEL.to_string(),
            calls(vec![infer_ref_call(
                "call-inf",
                CHILD_MODEL,
                "summarize the referenced output",
                &["call-sh"],
            )]),
        ),
        (CHILD_MODEL.to_string(), text("digest")),
        (PARENT_MODEL.to_string(), text("done")),
    ];
    let provider = Arc::new(MeteredProvider::new(&script));
    let prompt = vec![ChatMessage::system("system"), ChatMessage::user("go")];
    let (content, _events) =
        run_arm(provider.clone(), PARENT_MODEL, prompt, pricing_table()).await?;
    assert_eq!(content, "done");

    let calls = provider.recorded_calls();
    let child_call = calls
        .iter()
        .find(|call| call.model == CHILD_MODEL)
        .expect("child call recorded");
    // The child gets a proper message structure: referenced material first,
    // instruction last.
    assert_eq!(
        child_call.messages.len(),
        2,
        "referenced material + instruction: {:?}",
        child_call.messages
    );
    let referenced = child_call.messages[0].content.as_deref().unwrap_or("");
    assert!(
        referenced.starts_with("Referenced result of tool call call-sh (shell):")
            && referenced.contains("needle-material line 40"),
        "the material travels to the child by reference: {referenced}"
    );
    assert_eq!(
        child_call.messages[1].content.as_deref(),
        Some("summarize the referenced output")
    );

    // Parent history hygiene: in the FINAL parent prompt the material
    // appears exactly once — in the shell tool result — and the infer
    // tool-call arguments retained by prepare_tools contain the ref id,
    // never the material.
    let final_parent = calls
        .iter()
        .rfind(|call| call.model == PARENT_MODEL)
        .expect("final parent call recorded");
    let carriers = final_parent
        .messages
        .iter()
        .filter(|message| {
            message
                .content
                .as_deref()
                .is_some_and(|content| content.contains("needle-material"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        carriers.len(),
        1,
        "material lives in parent history exactly once"
    );
    assert_eq!(carriers[0].role, "tool");
    assert_eq!(carriers[0].tool_call_id.as_deref(), Some("call-sh"));
    let infer_args = final_parent
        .messages
        .iter()
        .flat_map(|message| message.tool_calls.as_deref().unwrap_or_default())
        .find(|call| call.name == "infer")
        .map(|call| call.arguments.to_string())
        .expect("infer tool call retained in history");
    assert!(
        infer_args.contains("call-sh") && !infer_args.contains("needle-material"),
        "retained infer arguments are refs, not copies: {infer_args}"
    );
    Ok(())
}

/// The child is offered NO tools (t-1346): it is a single completion whose
/// tool calls would never be dispatched, so the sub-infer site declares an
/// empty toolset (`InferPolicy.tools`, ir_agent.rs `infer_eval`) instead of
/// the parent's full set. The `infer` tool schema itself (as advertised to
/// the PARENT) still gives the model no model catalog, pricing, or budget
/// guidance for choosing a delegate — that deficiency stands.
#[tokio::test]
async fn probe_child_toolset_and_infer_schema_guidance() -> Result<()> {
    let script = vec![
        (
            PARENT_MODEL.to_string(),
            calls(vec![infer_call(
                "call-p1",
                CHILD_MODEL,
                "sub question".into(),
            )]),
        ),
        (CHILD_MODEL.to_string(), text("sub answer")),
        (PARENT_MODEL.to_string(), text("done")),
    ];
    let provider = Arc::new(MeteredProvider::new(&script));
    let prompt = vec![ChatMessage::system("system"), ChatMessage::user("go")];
    run_arm(provider.clone(), PARENT_MODEL, prompt, pricing_table()).await?;

    let calls = provider.recorded_calls();
    let child_call = calls
        .iter()
        .find(|call| call.model == CHILD_MODEL)
        .expect("child call recorded");
    // Fix pin (t-1346): the child cannot execute tools (its tool calls are
    // never dispatched), so it is offered none.
    assert!(
        child_call.tools.is_empty(),
        "child must be offered no tools, got {:?}",
        child_call
            .tools
            .iter()
            .map(|spec| &spec.function.name)
            .collect::<Vec<_>>()
    );

    // Deficiency pin: no model catalog / budget knobs in the infer schema
    // the PARENT is offered.
    let parent_call = calls
        .iter()
        .find(|call| call.model == PARENT_MODEL)
        .expect("parent call recorded");
    let infer_spec = parent_call
        .tools
        .iter()
        .find(|spec| spec.function.name == "infer")
        .expect("infer spec present");
    let model_property = &infer_spec.function.parameters["properties"]["model"];
    assert!(
        model_property.get("enum").is_none() && model_property.get("description").is_none(),
        "the infer schema's model parameter carries no guidance today: {model_property}"
    );
    assert!(
        infer_spec.function.parameters["properties"]
            .get("max_tokens")
            .is_none(),
        "the infer schema has no budget knob today"
    );
    // Fix pin (t-1344): the schema advertises pass-by-reference, and it is
    // optional — by-copy calls stay valid.
    let refs_property = &infer_spec.function.parameters["properties"]["context_refs"];
    assert_eq!(refs_property["type"], "array", "{refs_property}");
    let required = infer_spec.function.parameters["required"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        !required.iter().any(|name| name == "context_refs"),
        "context_refs must stay optional: {required:?}"
    );
    Ok(())
}

/// t-1120 decided the sub-response TEXT is fed back, not the Response
/// envelope. The envelope fallback used to resurface whenever the child
/// answered with tool calls (content empty); since t-1346 the child is
/// offered no tools, so its response is a single text completion and the
/// parent reads exactly that text — no usage fields, no tool-call JSON.
/// (The `infer_eval` fallback still exists, but as the readable surface
/// for bound child errors — see `probe_failed_child_binds_...` below.)
#[tokio::test]
async fn probe_child_response_feeds_back_as_bare_text() -> Result<()> {
    let script = vec![
        (
            PARENT_MODEL.to_string(),
            calls(vec![infer_call(
                "call-p1",
                CHILD_MODEL,
                "sub question".into(),
            )]),
        ),
        (CHILD_MODEL.to_string(), text("sub answer")),
        (PARENT_MODEL.to_string(), text("done")),
    ];
    let provider = Arc::new(MeteredProvider::new(&script));
    let prompt = vec![ChatMessage::system("system"), ChatMessage::user("go")];
    run_arm(provider.clone(), PARENT_MODEL, prompt, pricing_table()).await?;

    let calls = provider.recorded_calls();
    let child_call = calls
        .iter()
        .find(|call| call.model == CHILD_MODEL)
        .expect("child call recorded");
    assert!(
        child_call.tools.is_empty(),
        "the child is a bare single completion: no tools offered"
    );
    let final_parent_prompt = calls
        .iter()
        .rfind(|call| call.model == PARENT_MODEL)
        .expect("second parent call recorded");
    let tool_result = final_parent_prompt
        .messages
        .iter()
        .find(|message| message.role == "tool")
        .expect("infer tool result present");
    // Fix pin (t-1346/t-1120): the parent reads the child's text verbatim,
    // never the serialized Response envelope.
    assert_eq!(tool_result.content.as_deref(), Some("sub answer"));
    Ok(())
}

/// A failed sub-infer emits InferError (no InferResult), binds as a
/// recoverable tool value (t-1222) — and since t-1347 the attempt is
/// counted in the AgentDone rollup (`failed_infer_calls`) and the
/// InferError carries the dispatching parent's op_id. Token usage for the
/// attempt remains structurally unavailable (the provider error path
/// returns no Response), so it is a count, never a sum.
#[tokio::test]
async fn probe_failed_child_binds_and_counts_in_usage() -> Result<()> {
    let script = vec![
        (
            PARENT_MODEL.to_string(),
            calls(vec![infer_call(
                "call-p1",
                DEAD_MODEL,
                "sub question".into(),
            )]),
        ),
        (
            DEAD_MODEL.to_string(),
            provider_error("model not found (404)"),
        ),
        (PARENT_MODEL.to_string(), text("recovered")),
    ];
    let provider = Arc::new(MeteredProvider::new(&script));
    let prompt = vec![ChatMessage::system("system"), ChatMessage::user("go")];
    let (content, events) =
        run_arm(provider.clone(), PARENT_MODEL, prompt, pricing_table()).await?;
    assert_eq!(content, "recovered");

    // The error came back to the model as a readable tool result.
    let final_parent_prompt = provider
        .recorded_calls()
        .into_iter()
        .rfind(|call| call.model == PARENT_MODEL)
        .unwrap();
    let tool_result = final_parent_prompt
        .messages
        .iter()
        .find(|message| message.role == "tool")
        .expect("bound error tool result present");
    let tool_content = tool_result.content.as_deref().unwrap_or_default();
    assert!(
        tool_content.contains("\"ok\":false") && tool_content.contains("model not found"),
        "bound error is model-readable: {tool_content}"
    );

    let metrics = metrics_from_events(Arm::SubInfer, &events, &content, &["recovered"])?;
    assert_eq!(metrics.infer_errors, 1);
    assert_eq!(metrics.parent_infers, 2);
    assert_eq!(
        metrics.sub_infers, 1,
        "the attempt IS in the trace as a call"
    );
    // Fix pin (t-1347): successes and failed attempts are counted apart —
    // the failed attempt no longer vanishes from AgentDone usage. Its
    // tokens stay unmeasured (no Response exists to read them from).
    assert_eq!(metrics.usage.infer_calls, 2);
    assert_eq!(metrics.usage.failed_infer_calls, 1);
    assert_eq!(metrics.usage.uncosted_infer_calls, 0);

    // Fix pin (t-1347): the InferError is attributed to its dispatching
    // parent Infer via parent_op_id, same as a successful sub-infer.
    let first_parent_op = events
        .iter()
        .find_map(|event| match event {
            Event::InferCall {
                op_id,
                parent_op_id: None,
                ..
            } => Some(*op_id),
            _ => None,
        })
        .expect("parent-loop InferCall present");
    let error_parent = events
        .iter()
        .find_map(|event| match event {
            Event::InferError { parent_op_id, .. } => Some(*parent_op_id),
            _ => None,
        })
        .expect("InferError present");
    assert_eq!(error_parent, Some(first_parent_op));
    Ok(())
}

// --- recorded / online matrix ---------------------------------------------------
//
// The same fixtures against a real provider. ONLINE-GATED behind
// RUN_AGENT_ONLINE_EVAL=1 (the evals/ convention): online, every provider
// exchange is recorded to evals/infer-infer/recorded.jsonl keyed by a
// content hash of (model + prompt), so subsequent offline runs replay the
// recorded session deterministically — the GC judge's record/replay
// pattern, applied to whole agent runs. Offline with no recordings the
// test is a no-op pass.

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecordedMeta {
    parent_model: String,
    child_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecordedExchange {
    key: String,
    /// Provenance for human readers; lookup is purely by `key`.
    cell: String,
    model: String,
    response: Response,
}

fn recordings_path() -> Result<PathBuf> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("could not resolve repo root"))?;
    Ok(repo_root.join("evals/infer-infer/recorded.jsonl"))
}

/// Recording key: content hash of the model id plus the prompt's structural
/// content (roles, content, tool linkage — never message UUIDs, which are
/// freshly assigned every run).
fn exchange_key(model: &Model, messages: &[ChatMessage]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model.0.as_bytes());
    hasher.update([0]);
    for message in messages {
        hasher.update(message.role.as_bytes());
        hasher.update([0]);
        hasher.update(message.content.as_deref().unwrap_or("").as_bytes());
        hasher.update([0]);
        hasher.update(message.tool_call_id.as_deref().unwrap_or("").as_bytes());
        hasher.update([0]);
        for call in message.tool_calls.as_deref().unwrap_or_default() {
            hasher.update(call.id.as_bytes());
            hasher.update([0]);
            hasher.update(call.name.as_bytes());
            hasher.update([0]);
            hasher.update(call.arguments.to_string().as_bytes());
            hasher.update([0]);
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Replay-by-content-hash provider. Offline it only serves recordings (a
/// miss is an error, which the caller reports as a skipped cell); online
/// it forwards misses to the real provider and appends the recording.
struct RecordedProvider {
    path: PathBuf,
    recordings: Mutex<HashMap<String, Response>>,
    online: Option<ProviderClient>,
    cell: Mutex<String>,
}

impl RecordedProvider {
    fn load(path: PathBuf, online: bool) -> Result<(Option<RecordedMeta>, Self)> {
        let mut recordings = HashMap::new();
        let mut meta = None;
        if path.exists() {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("reading recordings {}", path.display()))?;
            for (line_idx, line) in content.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let value: serde_json::Value = serde_json::from_str(line).with_context(|| {
                    format!("decoding {} line {}", path.display(), line_idx + 1)
                })?;
                if let Some(found) = value.get("meta") {
                    meta = Some(serde_json::from_value(found.clone())?);
                    continue;
                }
                let exchange: RecordedExchange = serde_json::from_value(value)?;
                recordings.entry(exchange.key).or_insert(exchange.response);
            }
        }
        let online = if online { Some(online_client()?) } else { None };
        Ok((
            meta,
            Self {
                path,
                recordings: Mutex::new(recordings),
                online,
                cell: Mutex::new(String::new()),
            },
        ))
    }

    fn set_cell(&self, cell: String) {
        *self.cell.lock().unwrap() = cell;
    }

    fn append_line(&self, value: &serde_json::Value) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut line = serde_json::to_string(value)?;
        line.push('\n');
        use std::io::Write as _;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("appending recording {}", self.path.display()))?;
        file.write_all(line.as_bytes())?;
        Ok(())
    }

    fn write_meta(&self, meta: &RecordedMeta) -> Result<()> {
        self.append_line(&serde_json::json!({ "meta": meta }))
    }
}

#[async_trait]
impl ChatProvider for RecordedProvider {
    async fn chat(
        &self,
        model: &Model,
        tools: &[agent_core::provider::ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        let key = exchange_key(model, messages);
        if let Some(response) = self.recordings.lock().unwrap().get(&key) {
            return Ok(response.clone());
        }
        let cell = self.cell.lock().unwrap().clone();
        let Some(client) = &self.online else {
            return Err(anyhow!(
                "no recording for cell {cell} (key {key}); \
                 run with RUN_AGENT_ONLINE_EVAL=1 to record"
            ));
        };
        let response = client.chat(model, tools, messages).await?;
        self.append_line(&serde_json::to_value(RecordedExchange {
            key: key.clone(),
            cell,
            model: model.0.clone(),
            response: response.clone(),
        })?)?;
        self.recordings
            .lock()
            .unwrap()
            .insert(key, response.clone());
        Ok(response)
    }
}

/// Provider config from the environment, following the GC judge / evals
/// conventions: an OpenAI-compatible endpoint (default OpenRouter), key
/// from AGENT_API_KEY/ANTHROPIC_API_KEY/OPENROUTER_API_KEY.
fn online_client() -> Result<ProviderClient> {
    let url =
        std::env::var("AGENT_EVAL_URL").unwrap_or_else(|_| "https://openrouter.ai/api/v1".into());
    let api_key = std::env::var("AGENT_API_KEY")
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .or_else(|_| std::env::var("OPENROUTER_API_KEY"))
        .map_err(|_| {
            anyhow!(
                "RUN_AGENT_ONLINE_EVAL=1 needs AGENT_API_KEY/ANTHROPIC_API_KEY/OPENROUTER_API_KEY"
            )
        })?;
    Ok(ProviderClient::new(ProviderConfig {
        url,
        api_key,
        model: Model(env_parent_model()),
    }))
}

fn env_parent_model() -> String {
    std::env::var("AGENT_EVAL_PARENT_MODEL")
        .or_else(|_| std::env::var("AGENT_ONLINE_MODEL"))
        .unwrap_or_else(|_| "anthropic/claude-sonnet-4.5".into())
}

fn env_child_model() -> String {
    std::env::var("AGENT_EVAL_CHILD_MODEL").unwrap_or_else(|_| "anthropic/claude-haiku-4.5".into())
}

/// The recorded/online matrix. Reports rows for whatever recordings exist
/// (or records fresh ones online); asserts nothing about winners — real
/// model behavior is data here, and the fixture success column shows
/// whether the model completed the task and actually delegated. Structural
/// mechanics (attribution, rollup integrity) are still checked on every
/// completed row.
#[tokio::test]
async fn infer_infer_recorded_matrix() -> Result<()> {
    let path = recordings_path()?;
    let online = std::env::var("RUN_AGENT_ONLINE_EVAL").is_ok_and(|value| value == "1");
    let (meta, provider) = RecordedProvider::load(path.clone(), online)?;
    let meta = match meta {
        Some(meta) => meta,
        None if online => {
            let meta = RecordedMeta {
                parent_model: env_parent_model(),
                child_model: env_child_model(),
            };
            provider.write_meta(&meta)?;
            meta
        }
        None => {
            println!(
                "infer_infer_recorded_matrix: no recordings at {} — offline no-op; \
                 run with RUN_AGENT_ONLINE_EVAL=1 to record",
                path.display()
            );
            return Ok(());
        }
    };
    let provider = Arc::new(provider);

    print_header();
    for fixture in fixtures(&meta.child_model) {
        for arm in [Arm::Single, Arm::SubInfer] {
            let prompt = match arm {
                Arm::Single => fixture.single_prompt.clone(),
                Arm::SubInfer => fixture.sub_prompt.clone(),
            };
            provider.set_cell(format!("{}|{}", fixture.name, arm.label()));
            match run_arm(
                provider.clone(),
                &meta.parent_model,
                prompt,
                PricingTable::default(),
            )
            .await
            {
                Ok((content, events)) => {
                    let metrics =
                        metrics_from_events(arm, &events, &content, &fixture.success_needles)?;
                    print_metrics(fixture.name, &metrics);
                }
                Err(err) => {
                    println!("{:<22} {:<10} skipped: {err:#}", fixture.name, arm.label());
                }
            }
        }
    }
    Ok(())
}

// --- harness plumbing tests ------------------------------------------------------

/// The recording key must be stable across runs (message UUIDs are fresh
/// every construction) and sensitive to model and content.
#[test]
fn exchange_key_is_deterministic_and_id_independent() {
    let build = || {
        vec![
            ChatMessage::system("sys"),
            ChatMessage::user("do the thing"),
        ]
    };
    let (a, b) = (build(), build());
    assert_ne!(a[0].id, b[0].id, "UUIDs differ by construction");
    let model = Model("m".into());
    assert_eq!(exchange_key(&model, &a), exchange_key(&model, &b));
    assert_ne!(
        exchange_key(&model, &a),
        exchange_key(&Model("other".into()), &a)
    );
    let mut with_tool = build();
    with_tool.push(ChatMessage::assistant(
        None,
        vec![ToolCall::new(
            "id-1",
            "infer",
            serde_json::json!({"prompt": "x"}),
        )],
    ));
    assert_ne!(exchange_key(&model, &a), exchange_key(&model, &with_tool));
}

/// A recorded exchange written the way the online path writes it is served
/// back by a fresh offline provider, and offline misses are errors (never
/// a provider call).
#[tokio::test]
async fn recorded_provider_replays_and_fails_closed_offline() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("infer-infer-recorded-{}", Uuid::new_v4()));
    fs::create_dir_all(&dir)?;
    let path = dir.join("recorded.jsonl");

    let model = Model("rec-model".into());
    let prompt = vec![ChatMessage::user("recorded question")];
    let response = Response {
        content: "recorded answer".into(),
        tool_calls: Vec::new(),
        finish_reason: Some(FinishReason::Stop),
        input_tokens: 11,
        output_tokens: 3,
        total_tokens: 14,
        cached_input_tokens: None,
        cost_micro_usd: Some(42),
        pricing: None,
        metadata: Default::default(),
    };
    let meta = RecordedMeta {
        parent_model: "rec-model".into(),
        child_model: "rec-child".into(),
    };
    let exchange = RecordedExchange {
        key: exchange_key(&model, &prompt),
        cell: "test|single".into(),
        model: model.0.clone(),
        response: response.clone(),
    };
    fs::write(
        &path,
        format!(
            "{}\n{}\n",
            serde_json::json!({ "meta": meta }),
            serde_json::to_string(&exchange)?
        ),
    )?;

    let (loaded_meta, provider) = RecordedProvider::load(path, false)?;
    assert_eq!(loaded_meta.unwrap().child_model, "rec-child");
    let replayed = provider.chat(&model, &[], &prompt).await?;
    assert_eq!(replayed.content, "recorded answer");
    assert_eq!(
        replayed.cost_micro_usd,
        Some(42),
        "recorded cost passes through verbatim"
    );

    let miss = provider
        .chat(&model, &[], &[ChatMessage::user("unrecorded")])
        .await;
    assert!(miss.is_err(), "offline miss fails closed");
    assert!(format!("{:#}", miss.unwrap_err()).contains("no recording"));

    fs::remove_dir_all(&dir)?;
    Ok(())
}
