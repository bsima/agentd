//! Delegation-behavior eval harness (t-1354).
//!
//! t-1342/t-1344 (crates/agent-core/tests/infer_infer_evals.rs) validated the
//! COST mechanics of the `infer` tool with scripted arms — the model never
//! chose anything. This harness asks the behavioral question: given the tool,
//! does a REAL model use it effectively? Concretely (the questions the
//! findings section of evals/delegation/README.md answers):
//!
//! 1. does the model use the tool at all, unprompted?
//! 2. does guidance change delegation quality (guided vs unprompted delta)?
//! 3. does it show restraint where delegation is pure overhead?
//! 4. is "exploration via Infer" redundant given internal reasoning?
//! 5. can a subagent-as-PROCESS (`agent` one-shot via the shell tool) do
//!    real work, including tool work the infer child cannot do (t-1346:
//!    infer children get no tools)?
//!
//! Arms (same task text, same parent model; only the system prompt and the
//! advertised toolset differ):
//!
//! - **baseline**       — no `infer` tool advertised (provider-side filter);
//! - **tool-unprompted** — `infer` advertised, no guidance;
//! - **tool-guided**    — `infer` advertised + delegation guidance (when to
//!   delegate, `context_refs`, self-contained child prompts, when NOT to);
//! - **subagent-process** — no `infer` tool; guidance says to delegate by
//!   running `agent --model eval-child '<prompt>'` through the shell tool.
//!   The child is a FULL agent: its own loop, its own shell tool.
//!
//! Scoring is read from the trace, never estimated: `RunUsage` on
//! `AgentDone` (t-1334), per-`InferResult` cost, `parent_op_id` lineage
//! (t-1347) to count sub-infers, `EvalCall` commands to count process
//! delegations, plus per-fixture programmatic success needles.
//!
//! Online/offline: online (`RUN_AGENT_ONLINE_EVAL=1`, the evals/ convention)
//! runs each (fixture, arm) cell against a real provider and records the
//! cell's FULL event trace to `evals/delegation/recordings/`. Offline (the
//! default) replays those traces through the interpreter's effect-id replay
//! (`IrReplayTrace`) — Infer responses AND Eval (shell/child-process)
//! results come from the recording, so no provider, no key, no subprocess,
//! and recorded costs pass through verbatim. Unlike the GC judge fixtures
//! there are NO hand-written behavioral recordings: faking model choices
//! would corrupt the eval's purpose, so offline-without-recordings is a
//! documented no-op and the always-on tests below are plumbing-only
//! (scripted providers that validate the harness, marked as such).
//!
//! Arm-D child environment (documented contract): the harness gives shell
//! children an allowlist env — PATH prefixed with a dir holding the built
//! `agent` binary, HOME pointing at a per-cell scratch dir (so child traces
//! land where the harness can sweep them for cost), XDG_CONFIG_HOME
//! pointing at evals/delegation/config (the EVAL's own models.yaml defining
//! the `eval-child` alias), and — online only, arm D only —
//! OPENROUTER_API_KEY. Note the runtime's default Eval env policy strips
//! `*_API_KEY` precisely so model-issued commands cannot read the parent's
//! key; process-children needing a key is an explicit opt-in here and a
//! real deployment consideration for subagent-as-process.

use agent_core::{
    agent_loop_ir, run_ir_sequential_with_store_and_replay, ChatMessage, ChatProvider, EnvPolicy,
    EvalConfig, Event, GcMode, GcTiming, InMemoryStore, IrReplayTrace, Model,
    PassiveHydrationConfig, Pricing, PricingTable, ProviderClient, ProviderConfig,
    ReplayOnlyProvider, Response, RunUsage, SeqConfig, SourceRegistry, ToolCall, TraceLogger,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

const MAX_TURNS: usize = 12;
/// Cheap-but-capable default parent; overridable with
/// AGENT_EVAL_PARENT_MODEL. OpenRouter id.
const DEFAULT_PARENT_MODEL: &str = "anthropic/claude-haiku-4.5";
/// Cheaper delegate; overridable with AGENT_EVAL_CHILD_MODEL. OpenRouter id.
const DEFAULT_CHILD_MODEL: &str = "openai/gpt-4o-mini";
/// The registry alias arm-D children run as; defined in
/// evals/delegation/config/agent/models.yaml (the eval's own registry).
const PROCESS_CHILD_ALIAS: &str = "eval-child";

fn repo_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("could not resolve repo root"))
}

fn recordings_dir() -> Result<PathBuf> {
    Ok(repo_root()?.join("evals/delegation/recordings"))
}

/// XDG_CONFIG_HOME for arm-D children: contains agent/models.yaml defining
/// the `eval-child` alias against OpenRouter with a pricing block, so child
/// traces carry costed InferResults (t-1334).
fn eval_config_home() -> Result<PathBuf> {
    Ok(repo_root()?.join("evals/delegation/config"))
}

/// Fixture pricing for the DEFAULT model ids (OpenRouter list prices, USD
/// per Mtok). Env-overridden models run uncosted (visible as
/// `uncosted_infer_calls` in the rollup) — absent pricing is never guessed.
fn pricing_table() -> PricingTable {
    let mut table = PricingTable::default();
    table.insert(
        DEFAULT_PARENT_MODEL,
        Pricing::from_usd_per_mtok(1.0, 5.0).unwrap(),
    );
    table.insert(
        DEFAULT_CHILD_MODEL,
        Pricing::from_usd_per_mtok(0.15, 0.60).unwrap(),
    );
    table
}

// --- arms ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Baseline,
    ToolUnprompted,
    ToolGuided,
    SubagentProcess,
}

impl Arm {
    const ALL: [Arm; 4] = [
        Arm::Baseline,
        Arm::ToolUnprompted,
        Arm::ToolGuided,
        Arm::SubagentProcess,
    ];

    fn label(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ToolUnprompted => "tool-unprompted",
            Self::ToolGuided => "tool-guided",
            Self::SubagentProcess => "subagent-process",
        }
    }

    fn from_label(label: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|arm| arm.label() == label)
            .ok_or_else(|| anyhow!("unknown arm label {label}"))
    }

    /// Whether the `infer` tool is advertised to the parent. The loop
    /// program always contains the infer arm; exposure is filtered at the
    /// provider boundary ([`WithoutInferTool`]), which is invisible to
    /// replay (recorded responses drive the run).
    fn advertises_infer(&self) -> bool {
        matches!(self, Self::ToolUnprompted | Self::ToolGuided)
    }

    fn system_prompt(&self, _child_model: &str) -> String {
        let base = "You are a capable software agent with a shell tool, working in the \
                    current directory. Be efficient: use the fewest steps that complete \
                    the task correctly, then give the final answer.";
        match self {
            // The guided arm's delegation text is no longer handwritten
            // here: it is the SHIPPED runtime-guidance fragment (t-1359),
            // delivered by the runtime itself (see Arm::guidance) — the
            // arms measure exactly what ships.
            Self::Baseline | Self::ToolUnprompted | Self::ToolGuided => base.to_string(),
            Self::SubagentProcess => format!(
                "{base}\n\nYou can delegate subtasks by launching a child agent through \
                 the shell tool: run `agent --model {PROCESS_CHILD_ALIAS} 'CHILD \
                 PROMPT'` (one single-quoted argument). The child is a full agent with \
                 its own shell tool, runs in this same working directory, and prints \
                 its final answer to stdout. Delegate when a subtask is \
                 generation-heavy, needs bulky material read or digested, or is \
                 mechanical tool work. Do NOT delegate direct questions you can already \
                 answer: a child process costs a full agent run. Child prompts must be \
                 self-contained — the child cannot see this conversation."
            ),
        }
    }

    /// Runtime-guidance config per arm (t-1359). The guided arm runs the
    /// SHIPPED fragment default-on, with the interim delegate catalog
    /// naming the eval's child model (the same id the old handwritten
    /// text named) — so the arm measures exactly what a deployment that
    /// supplies its delegate catalog gets. Every other arm keeps guidance
    /// off: baseline/tool-unprompted measure the runtime WITHOUT its
    /// guidance layer (that contrast is this eval's question), and the
    /// subagent-process arm keeps its handwritten process-delegation
    /// prompt, which the fragment does not cover.
    fn guidance(&self, child_model: &str) -> agent_core::RuntimeGuidance {
        match self {
            Self::ToolGuided => agent_core::RuntimeGuidance {
                enabled: true,
                delegate_models: vec![agent_core::DelegateModel {
                    id: child_model.to_string(),
                    pricing: pricing_table().get(child_model).copied(),
                }],
            },
            _ => agent_core::RuntimeGuidance::disabled(),
        }
    }
}

/// Provider middleware that hides the `infer` tool from the model (arms
/// that must not see it). Filtering happens at the provider boundary so
/// the loop program — and therefore effect ids and replay — is identical
/// across arms.
struct WithoutInferTool(Arc<dyn ChatProvider>);

#[async_trait]
impl ChatProvider for WithoutInferTool {
    async fn chat(
        &self,
        model: &Model,
        tools: &[agent_core::provider::ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        let filtered: Vec<agent_core::provider::ToolSpec> = tools
            .iter()
            .filter(|spec| spec.function.name != "infer")
            .cloned()
            .collect();
        self.0.chat(model, &filtered, messages).await
    }
}

// --- fixtures -----------------------------------------------------------------

/// What effective delegation looks like on this fixture — drives the
/// appropriateness flag, never a hard assertion (real model behavior is
/// data; the findings section interprets it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stance {
    /// Any delegation is pure overhead; effective use = NOT delegating.
    Restraint,
    /// Delegation plausibly helps; not delegating is a missed opportunity
    /// (scored, not failed — that is a finding).
    Help,
    /// Exploration shape (tempting wrong path): delegation vs internal
    /// reasoning is exactly the open question, so no flag either way.
    Exploration,
    /// Tool work a process child can do but an infer child cannot
    /// (t-1346): the arm-D capability probe.
    ToolWork,
}

struct Fixture {
    name: &'static str,
    stance: Stance,
    /// Identical across arms — the model, not the task, chooses delegation.
    task: String,
    /// Programmatic success needles (v1: no judge). Pure-numeric needles
    /// match on token boundaries, text needles by substring.
    needles: Vec<&'static str>,
    /// Deterministic working-directory content the shell tool sees.
    files: Vec<(&'static str, String)>,
}

/// Shell command printing quarterly report `index` (~2.5KB of deterministic
/// filler around one KEY FACT line) — same shape as the t-1342 fixtures,
/// but here NO arm is told to delegate the reading; the model chooses.
fn report_command(index: usize, fact: &str) -> String {
    format!(
        "seq -f \"report {index} filler paragraph %g alpha bravo charlie delta echo foxtrot\" 1 30 \
         && echo \"KEY FACT: {fact}\" \
         && seq -f \"report {index} appendix filler %g golf hotel india juliet kilo lima\" 1 30"
    )
}

fn fixtures() -> Vec<Fixture> {
    let mut fixtures = Vec::new();

    // Task class 1 — delegation-should-help: synthesis across three long
    // documents (the t-1342 shape, model's choice). Effective delegation =
    // fetch via shell, hand each report to the cheap child by context_refs.
    {
        let commands = [
            report_command(1, "revenue grew 12%"),
            report_command(2, "churn fell to 3%"),
            report_command(3, "headcount stayed at 84"),
        ];
        fixtures.push(Fixture {
            name: "doc-synthesis",
            stance: Stance::Help,
            task: format!(
                "Fetch the three quarterly reports by running each of these shell \
                 commands as its own shell tool call:\n1. {}\n2. {}\n3. {}\nThen \
                 synthesize the reports into a short paragraph covering each report's \
                 key fact.",
                commands[0], commands[1], commands[2]
            ),
            needles: vec!["12%", "3%", "84"],
            files: Vec::new(),
        });
    }

    // Task class 2 — exploration with a dead end: lib.sh carries a tempting
    // wrong lead (a comment blaming its arithmetic); the actual bug is a
    // config value. Measures whether Infer-based exploration happens and
    // helps versus internal reasoning — the harness must be able to
    // FALSIFY the value of delegated exploration, not assume it.
    {
        fixtures.push(Fixture {
            name: "dead-end-debugging",
            stance: Stance::Exploration,
            task: "This directory contains run.sh, lib.sh, and config.env. The billing \
                   spec says RATE=21 and HOURS=20, so `sh run.sh` should print \
                   `total: 420`, but it prints `total: 400`. Investigate and find the \
                   actual bug. Answer with just the name of the one file that must be \
                   fixed."
                .into(),
            needles: vec!["config.env"],
            files: vec![
                (
                    "run.sh",
                    "#!/bin/sh\n. ./config.env\n. ./lib.sh\ncompute\n".into(),
                ),
                (
                    "lib.sh",
                    "# invoice computation library\n\
                     # TODO: totals have been reported low before. The arithmetic \n\
                     # below has been flagged as the likely culprit — double-check \n\
                     # the multiplication (possible off-by-one in HOURS handling?).\n\
                     compute() {\n    echo \"total: $((RATE * HOURS))\"\n}\n"
                        .into(),
                ),
                (
                    "config.env",
                    "# billing configuration (synced from billing-spec v7)\n\
                     CURRENCY=USD\nREGION=us-east\nRATE=20\nHOURS=20\nTAX_PCT=0\n"
                        .into(),
                ),
            ],
        });
    }

    // Task class 1b — generation offload: the ONE shape where t-1342
    // measured the infer mechanism winning outright (~2.7x, output-rate
    // arbitrage: a short prompt, a long cheap generation, read back at
    // input rates). If a model ever delegates by choice, it should be
    // here.
    {
        fixtures.push(Fixture {
            name: "generation-offload",
            stance: Stance::Help,
            task: "Evaluate all 20 candidate product names (aurora, basil, cinder, \
                   dune, ember, flint, gale, harbor, iris, juniper, krill, lumen, \
                   maple, nimbus, onyx, prism, quarry, reef, sable, zephyr) against \
                   these constraints: memorable, short, unambiguous, no obvious \
                   trademark collisions. Produce the full written evaluation — one or \
                   two sentences per candidate, all 20 covered — and end with a final \
                   line 'CHOSEN: <name> — <reason>'."
                .into(),
            needles: vec![
                "aurora", "basil", "cinder", "dune", "ember", "flint", "gale", "harbor", "iris",
                "juniper", "krill", "lumen", "maple", "nimbus", "onyx", "prism", "quarry", "reef",
                "sable", "zephyr", "CHOSEN:",
            ],
            files: Vec::new(),
        });
    }

    // Task class 3 — restraint: a direct question where any delegation is
    // pure overhead. Effective use of the mechanism = not using it.
    {
        fixtures.push(Fixture {
            name: "restraint-direct",
            stance: Stance::Restraint,
            task: "What is 17 * 23? Answer with just the number.".into(),
            needles: vec!["391"],
            files: Vec::new(),
        });
    }

    // Task class 4 — tool work (the arm-D probe): counting across files
    // needs a tool-using child. An infer child has no tools (t-1346), so
    // infer arms must do the counting themselves; a process child is a
    // full agent and can be handed the whole subtask.
    {
        let log = |lines: &[&str]| lines.join("\n") + "\n";
        fixtures.push(Fixture {
            name: "count-in-files",
            stance: Stance::ToolWork,
            task: "Count how many lines across the files logs/a.log, logs/b.log, and \
                   logs/c.log contain the exact uppercase string ERROR. Answer with \
                   just the number."
                .into(),
            needles: vec!["7"],
            files: vec![
                (
                    "logs/a.log",
                    log(&[
                        "2026-01-01 startup ok",
                        "2026-01-01 ERROR disk quota exceeded",
                        "2026-01-01 warn retrying (lowercase error noise)",
                        "2026-01-02 ERROR disk quota exceeded",
                        "2026-01-02 notice compaction done",
                        "2026-01-03 ERROR replica lag",
                    ]),
                ),
                (
                    "logs/b.log",
                    log(&[
                        "2026-02-01 startup ok",
                        "2026-02-01 warn slow query",
                        "2026-02-02 ERROR timeout upstream",
                        "2026-02-03 error lowercase decoy",
                        "2026-02-03 ERROR timeout upstream",
                    ]),
                ),
                (
                    "logs/c.log",
                    log(&[
                        "2026-03-01 startup ok",
                        "2026-03-02 ERROR checksum mismatch",
                        "2026-03-02 notice repaired",
                        "2026-03-03 ERROR checksum mismatch",
                        "2026-03-04 shutdown clean",
                    ]),
                ),
            ],
        });
    }

    fixtures
}

/// Needle matching: pure-numeric needles must sit on token boundaries
/// (so the answer "27" never satisfies needle "7"); everything else is a
/// case-insensitive substring check (models legitimately capitalize —
/// "Aurora" satisfies "aurora"; the first recording pass scored three
/// correct generation-offload answers as failures on exactly this).
fn needle_present(content: &str, needle: &str) -> bool {
    if needle.is_empty() || !needle.chars().all(|c| c.is_ascii_digit()) {
        return content.to_lowercase().contains(&needle.to_lowercase());
    }
    let bytes = content.as_bytes();
    let mut from = 0;
    while let Some(pos) = content[from..].find(needle) {
        let at = from + pos;
        let end = at + needle.len();
        // A neighboring '.' only breaks the boundary when it makes the
        // match part of a decimal ("39.1", "7.5") — a sentence-ending
        // period ("is 391.") does not.
        let before_ok = at == 0 || {
            let prev = bytes[at - 1];
            !(prev.is_ascii_alphanumeric()
                || (prev == b'.' && at >= 2 && bytes[at - 2].is_ascii_digit()))
        };
        let after_ok = end >= bytes.len() || {
            let next = bytes[end];
            !(next.is_ascii_alphanumeric()
                || (next == b'.' && end + 1 < bytes.len() && bytes[end + 1].is_ascii_digit()))
        };
        if before_ok && after_ok {
            return true;
        }
        from = at + 1;
    }
    false
}

// --- cell runner ---------------------------------------------------------------

/// Extra env granted to shell children (EnvPolicy::AllowList — children see
/// exactly this, nothing inherited).
fn child_env(
    bin_dir: &Path,
    child_home: &Path,
    api_key: Option<&str>,
) -> Result<BTreeMap<String, String>> {
    let mut env = BTreeMap::new();
    let path = std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".into());
    env.insert("PATH".into(), format!("{}:{path}", bin_dir.display()));
    env.insert("HOME".into(), child_home.display().to_string());
    env.insert(
        "XDG_CONFIG_HOME".into(),
        eval_config_home()?.display().to_string(),
    );
    if let Some(key) = api_key {
        env.insert("OPENROUTER_API_KEY".into(), key.to_string());
    }
    Ok(env)
}

/// A dir containing an `agent` symlink to the binary cargo built for this
/// test run, prepended to the children's PATH so the guidance command
/// `agent --model eval-child '...'` resolves without embedding paths in
/// prompts (prompt text must be byte-stable across record/replay).
fn agent_bin_dir() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("delegation-eval-bin-{}", std::process::id()));
    fs::create_dir_all(&dir)?;
    let link = dir.join("agent");
    if !link.exists() {
        std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_agent"), &link)?;
    }
    Ok(dir)
}

struct CellRun {
    content: String,
    events: Vec<Event>,
    wall_ms: u64,
}

async fn run_cell(
    provider: Arc<dyn ChatProvider>,
    replay: Option<&IrReplayTrace>,
    parent_model: &str,
    prompt: Vec<ChatMessage>,
    workdir: &Path,
    env: BTreeMap<String, String>,
    guidance: agent_core::RuntimeGuidance,
) -> Result<CellRun> {
    let trace_path = std::env::temp_dir().join(format!("delegation-eval-{}.jsonl", Uuid::new_v4()));
    let trace = TraceLogger::new(Uuid::new_v4().to_string(), trace_path.clone());
    let config = SeqConfig {
        approvals: Default::default(),
        guidance,
        tools: Default::default(),
        provider,
        hydration: SourceRegistry::new(),
        passive_hydration: PassiveHydrationConfig::default(),
        trace: trace.clone(),
        eval: EvalConfig {
            shell: "/bin/sh".into(),
            cwd: Some(workdir.to_path_buf()),
            timeout: Duration::from_secs(300),
            env: EnvPolicy::AllowList {
                names: Vec::new(),
                extra: env,
            },
            ..EvalConfig::default()
        },
        replay: None,
        trace_full_prompt_ir: false,
        trace_full_payloads: false,
        gc: GcMode::None,
        gc_threshold: 0.85,
        gc_log: false,
        gc_timing: GcTiming::Threshold,
        context_budget: 200_000,
        pricing: pricing_table(),
    };
    let machine = agent_loop_ir(Model(parent_model.into()), prompt, MAX_TURNS);
    let started = Instant::now();
    let mut store = InMemoryStore::new();
    let (value, _machine) =
        run_ir_sequential_with_store_and_replay(&config, machine, &mut store, replay).await?;
    let wall_ms = started.elapsed().as_millis() as u64;
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
    Ok(CellRun {
        content,
        events,
        wall_ms,
    })
}

/// Create the fixture's working directory content under a fresh temp dir.
fn materialize_fixture(fixture: &Fixture) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("delegation-eval-fx-{}", Uuid::new_v4()));
    fs::create_dir_all(&dir)?;
    for (rel, content) in &fixture.files {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)?;
    }
    Ok(dir)
}

// --- metrics -------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct CellMetrics {
    /// Parent-loop infer calls (= provider turns taken).
    turns: usize,
    /// Nested infer-tool delegations (parent_op_id lineage, t-1347).
    sub_infers: usize,
    /// Process delegations: shell Evals invoking the `agent` binary.
    proc_delegations: usize,
    eval_calls: usize,
    infer_errors: usize,
    usage: RunUsage,
    success: bool,
}

fn is_agent_invocation(command: &str) -> bool {
    let trimmed = command.trim_start();
    trimmed == "agent" || trimmed.starts_with("agent ")
}

fn metrics_from_events(events: &[Event], content: &str, needles: &[&str]) -> Result<CellMetrics> {
    let mut sub_by_op: HashMap<u64, bool> = HashMap::new();
    let mut metrics = CellMetrics {
        turns: 0,
        sub_infers: 0,
        proc_delegations: 0,
        eval_calls: 0,
        infer_errors: 0,
        usage: RunUsage::default(),
        success: needles.iter().all(|needle| needle_present(content, needle)),
    };
    let mut done_usage: Option<RunUsage> = None;
    for event in events {
        match event {
            Event::InferCall {
                op_id,
                parent_op_id,
                ..
            } => {
                let sub = parent_op_id.is_some();
                sub_by_op.insert(*op_id, sub);
                if sub {
                    metrics.sub_infers += 1;
                } else {
                    metrics.turns += 1;
                }
            }
            Event::InferError { .. } => metrics.infer_errors += 1,
            Event::EvalCall { command, .. } => {
                metrics.eval_calls += 1;
                if is_agent_invocation(command) {
                    metrics.proc_delegations += 1;
                }
            }
            Event::AgentDone { usage, .. } => done_usage = usage.clone(),
            _ => {}
        }
    }
    metrics.usage = done_usage
        .ok_or_else(|| anyhow!("trace has no AgentDone usage rollup (t-1334 instrument)"))?;
    Ok(metrics)
}

/// Appropriateness flag from the fixture stance. Informational — printed
/// and interpreted in the README findings, never asserted (behavior is the
/// data this eval exists to collect). The baseline arm has no delegation
/// mechanism, so no flag can apply to it.
fn appropriateness_flag(arm: Arm, stance: Stance, metrics: &CellMetrics) -> &'static str {
    if arm == Arm::Baseline {
        return "-";
    }
    let delegated = metrics.sub_infers > 0 || metrics.proc_delegations > 0;
    match stance {
        Stance::Restraint if delegated => "OVER-DELEGATED",
        Stance::Help if !delegated => "missed-op",
        _ => "-",
    }
}

// --- recording format ------------------------------------------------------------
//
// One JSONL file per (fixture, arm) cell: a meta first line, then one
// {"event": ...} line per trace event of the online run. Offline, the
// events rebuild an IrReplayTrace (effect-id replay) AND serve as the
// reference metrics the replayed run must reproduce. Recordings must be
// credential-free: the online writer asserts the key never appears, and
// `recordings_are_credential_free` re-checks whatever is committed.

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CellMeta {
    fixture: String,
    arm: String,
    parent_model: String,
    child_model: String,
    /// Online wall time — replays report this, not their own.
    wall_ms: u64,
    /// The online run's final answer; replay must reproduce it.
    final_content: String,
    /// Arm-D process children: usage summed from the child agents' own
    /// traces (swept from the per-cell child HOME). Zero elsewhere.
    child_runs: usize,
    child_cost_micro_usd: u64,
    child_total_tokens: u64,
    recorded_at: String,
}

fn cell_path(dir: &Path, fixture: &str, arm: Arm) -> PathBuf {
    dir.join(format!("{fixture}--{}.jsonl", arm.label()))
}

fn write_cell_recording(path: &Path, meta: &CellMeta, events: &[Event]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = serde_json::to_string(&serde_json::json!({ "meta": meta }))?;
    out.push('\n');
    for event in events {
        out.push_str(&serde_json::to_string(
            &serde_json::json!({ "event": event }),
        )?);
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

fn load_cell_recording(path: &Path) -> Result<(CellMeta, Vec<Event>)> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut meta = None;
    let mut events = Vec::new();
    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("decoding {} line {}", path.display(), index + 1))?;
        if let Some(found) = value.get("meta") {
            meta = Some(serde_json::from_value(found.clone())?);
        } else if let Some(found) = value.get("event") {
            events.push(serde_json::from_value(found.clone())?);
        } else {
            return Err(anyhow!(
                "{} line {}: neither meta nor event",
                path.display(),
                index + 1
            ));
        }
    }
    Ok((
        meta.ok_or_else(|| anyhow!("{} has no meta line", path.display()))?,
        events,
    ))
}

/// Sum usage from the child agent traces a subagent-process cell left in
/// its per-cell HOME. Children stamp their own InferResults (the eval
/// registry carries pricing), so this is read, not estimated.
fn sweep_child_traces(child_home: &Path) -> Result<(usize, u64, u64)> {
    let traces = child_home.join(".local/share/agent/traces");
    let mut runs = 0;
    let (mut cost, mut tokens) = (0u64, 0u64);
    if !traces.exists() {
        return Ok((0, 0, 0));
    }
    for entry in fs::read_dir(&traces)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        runs += 1;
        let content = fs::read_to_string(&path)?;
        for line in content.lines().filter(|line| !line.trim().is_empty()) {
            if let Ok(Event::InferResult {
                cost_micro_usd,
                total_tokens,
                ..
            }) = serde_json::from_str::<Event>(line)
            {
                cost += cost_micro_usd.unwrap_or(0);
                tokens += u64::from(total_tokens);
            }
        }
    }
    Ok((runs, cost, tokens))
}

// --- table ----------------------------------------------------------------------

fn print_header() {
    println!(
        "{:<20} {:<17} {:>5} {:>4} {:>5} {:>5} {:>4} {:>8} {:>8} {:>10} {:>10} {:>7} {:>3}  flag",
        "fixture",
        "arm",
        "turns",
        "sub",
        "proc",
        "evals",
        "errs",
        "in_tok",
        "out_tok",
        "cost",
        "child$",
        "wall_s",
        "ok",
    );
}

fn print_row(fixture: &Fixture, arm: Arm, metrics: &CellMetrics, meta: &CellMeta) {
    println!(
        "{:<20} {:<17} {:>5} {:>4} {:>5} {:>5} {:>4} {:>8} {:>8} {:>10} {:>10} {:>7.1} {:>3}  {}",
        fixture.name,
        arm.label(),
        metrics.turns,
        metrics.sub_infers,
        metrics.proc_delegations,
        metrics.eval_calls,
        metrics.infer_errors,
        metrics.usage.input_tokens,
        metrics.usage.output_tokens,
        metrics
            .usage
            .cost_micro_usd
            .map_or_else(|| "-".into(), agent_core::format_micro_usd),
        if meta.child_runs > 0 {
            agent_core::format_micro_usd(meta.child_cost_micro_usd)
        } else {
            "-".into()
        },
        meta.wall_ms as f64 / 1000.0,
        if metrics.success { "yes" } else { "NO" },
        appropriateness_flag(arm, fixture.stance, metrics)
    );
}

// --- online provider ---------------------------------------------------------------

fn env_parent_model() -> String {
    std::env::var("AGENT_EVAL_PARENT_MODEL").unwrap_or_else(|_| DEFAULT_PARENT_MODEL.into())
}

fn env_child_model() -> String {
    std::env::var("AGENT_EVAL_CHILD_MODEL").unwrap_or_else(|_| DEFAULT_CHILD_MODEL.into())
}

fn online_api_key() -> Result<String> {
    std::env::var("AGENT_API_KEY")
        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        .or_else(|_| std::env::var("OPENROUTER_API_KEY"))
        .map_err(|_| {
            anyhow!(
                "RUN_AGENT_ONLINE_EVAL=1 needs AGENT_API_KEY/ANTHROPIC_API_KEY/OPENROUTER_API_KEY"
            )
        })
}

fn online_client(parent_model: &str) -> Result<ProviderClient> {
    let url =
        std::env::var("AGENT_EVAL_URL").unwrap_or_else(|_| "https://openrouter.ai/api/v1".into());
    Ok(ProviderClient::new(ProviderConfig {
        url,
        api_key: online_api_key()?,
        model: Model(parent_model.into()),
    }))
}

// --- the matrix ------------------------------------------------------------------

/// The delegation-behavior matrix: every fixture x arm.
///
/// Offline (default): replays each cell's recorded trace through effect-id
/// replay, asserts the replay reproduces the recording (final answer and
/// metrics), and prints the table. Cells without recordings are reported
/// and skipped; a wholly-absent recordings dir is a clean no-op (there are
/// deliberately no hand-written behavioral recordings — see module docs).
///
/// Online (RUN_AGENT_ONLINE_EVAL=1): records any missing cells against the
/// real provider first (arm-D cells spawn real `agent` child processes),
/// then replays everything just like offline — so a recording run IS a
/// replay verification run.
#[tokio::test]
async fn delegation_matrix() -> Result<()> {
    let online = std::env::var("RUN_AGENT_ONLINE_EVAL").is_ok_and(|value| value == "1");
    let dir = recordings_dir()?;

    if online {
        record_missing_cells(&dir).await?;
    } else if !dir.exists() {
        println!(
            "delegation_matrix: no recordings at {} — offline no-op; \
             run with RUN_AGENT_ONLINE_EVAL=1 to record (see evals/delegation/README.md)",
            dir.display()
        );
        return Ok(());
    }

    print_header();
    for fixture in fixtures() {
        for arm in Arm::ALL {
            let path = cell_path(&dir, fixture.name, arm);
            if !path.exists() {
                println!(
                    "{:<20} {:<17} skipped: no recording ({})",
                    fixture.name,
                    arm.label(),
                    path.display()
                );
                continue;
            }
            let (meta, metrics) = replay_cell(&path, &fixture).await?;
            print_row(&fixture, Arm::from_label(&meta.arm)?, &metrics, &meta);
        }
    }
    Ok(())
}

/// Replay one recorded cell and assert it reproduces the recording: same
/// final answer, same trace-derived metrics as the recorded events.
async fn replay_cell(path: &Path, fixture: &Fixture) -> Result<(CellMeta, CellMetrics)> {
    let (meta, recorded_events) = load_cell_recording(path)?;
    anyhow::ensure!(
        meta.fixture == fixture.name,
        "{}: recording is for fixture {}",
        path.display(),
        meta.fixture
    );
    let arm = Arm::from_label(&meta.arm)?;
    let replay = IrReplayTrace::from_events(&recorded_events)
        .with_context(|| format!("building replay from {}", path.display()))?;
    let prompt = vec![
        ChatMessage::system(arm.system_prompt(&meta.child_model)),
        ChatMessage::user(fixture.task.clone()),
    ];
    let workdir = materialize_fixture(fixture)?;
    let run = run_cell(
        Arc::new(ReplayOnlyProvider),
        Some(&replay),
        &meta.parent_model,
        prompt,
        &workdir,
        BTreeMap::new(),
        arm.guidance(&meta.child_model),
    )
    .await
    .with_context(|| format!("replaying {}", path.display()))?;
    let _ = fs::remove_dir_all(&workdir);

    assert_eq!(
        run.content,
        meta.final_content,
        "{}: replay must reproduce the recorded final answer",
        path.display()
    );
    let replayed = metrics_from_events(&run.events, &run.content, &fixture.needles)?;
    let recorded = metrics_from_events(&recorded_events, &meta.final_content, &fixture.needles)?;
    assert_eq!(
        replayed,
        recorded,
        "{}: replayed metrics must reproduce the recording",
        path.display()
    );
    Ok((meta, replayed))
}

/// Record every cell that has no recording yet. Requires a key; spends real
/// money (small fixtures, cheap models — see README for the measured total).
async fn record_missing_cells(dir: &Path) -> Result<()> {
    let parent_model = env_parent_model();
    let child_model = env_child_model();
    let api_key = online_api_key()?;
    let client: Arc<dyn ChatProvider> = Arc::new(online_client(&parent_model)?);
    let bin_dir = agent_bin_dir()?;

    for fixture in fixtures() {
        for arm in Arm::ALL {
            let path = cell_path(dir, fixture.name, arm);
            if path.exists() {
                continue;
            }
            println!("recording {} / {} ...", fixture.name, arm.label());
            let provider: Arc<dyn ChatProvider> = if arm.advertises_infer() {
                client.clone()
            } else {
                Arc::new(WithoutInferTool(client.clone()))
            };
            let prompt = vec![
                ChatMessage::system(arm.system_prompt(&child_model)),
                ChatMessage::user(fixture.task.clone()),
            ];
            let workdir = materialize_fixture(&fixture)?;
            let child_home =
                std::env::temp_dir().join(format!("delegation-eval-home-{}", Uuid::new_v4()));
            fs::create_dir_all(child_home.join(".local/share/agent/traces"))?;
            // The key goes only to arm-D children (they dial the provider
            // themselves); other arms' shell children get none.
            let key_for_children = (arm == Arm::SubagentProcess).then_some(api_key.as_str());
            let env = child_env(&bin_dir, &child_home, key_for_children)?;
            let run = run_cell(
                provider,
                None,
                &parent_model,
                prompt,
                &workdir,
                env,
                arm.guidance(&child_model),
            )
            .await
            .with_context(|| format!("online cell {} / {}", fixture.name, arm.label()))?;
            let (child_runs, child_cost, child_tokens) = sweep_child_traces(&child_home)?;
            let _ = fs::remove_dir_all(&workdir);
            let _ = fs::remove_dir_all(&child_home);

            let meta = CellMeta {
                fixture: fixture.name.into(),
                arm: arm.label().into(),
                parent_model: parent_model.clone(),
                child_model: child_model.clone(),
                wall_ms: run.wall_ms,
                final_content: run.content.clone(),
                child_runs,
                child_cost_micro_usd: child_cost,
                child_total_tokens: child_tokens,
                recorded_at: Utc::now().to_rfc3339(),
            };
            write_cell_recording(&path, &meta, &run.events)?;
            // Credential hygiene: the recording must not embed the key
            // (events carry prompts, tool commands, and child stdout —
            // none of which should ever contain it).
            let written = fs::read_to_string(&path)?;
            anyhow::ensure!(
                !written.contains(api_key.as_str()),
                "{}: recording embeds the API key — deleted; do not commit",
                path.display()
            );
        }
    }
    Ok(())
}

// --- process-child capability probe (Ben's Q5) -------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProbeRecord {
    child_command: Vec<String>,
    stdout: String,
    exit_ok: bool,
    wall_ms: u64,
    child_runs: usize,
    child_cost_micro_usd: u64,
    child_total_tokens: u64,
    recorded_at: String,
}

fn probe_path() -> Result<PathBuf> {
    Ok(recordings_dir()?.join("child-capability.json"))
}

/// Can a subagent-as-process do REAL work — its own loop, its own Eval —
/// when handed a self-contained subtask? The matrix shows a competent
/// parent rationally does trivial tool work itself instead of delegating,
/// so this probe runs the exact invocation the arm-D guidance teaches
/// (`agent --model eval-child '<prompt>'`) directly against the
/// count-in-files fixture, decoupled from parent inclination. Online it
/// spawns the real binary with the same allowlist env arm D provides
/// (PATH, HOME, XDG_CONFIG_HOME -> the eval registry, OPENROUTER_API_KEY);
/// offline it reports the recording; no recording = documented no-op.
#[tokio::test]
async fn process_child_capability_probe() -> Result<()> {
    let online = std::env::var("RUN_AGENT_ONLINE_EVAL").is_ok_and(|value| value == "1");
    let path = probe_path()?;
    let child_prompt = "Count how many lines across the files logs/a.log, logs/b.log, \
                        and logs/c.log in the current directory contain the exact \
                        uppercase string ERROR. Answer with just the number.";

    let record: ProbeRecord = if path.exists() && !online {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else if !online {
        println!(
            "process_child_capability_probe: no recording at {} — offline no-op",
            path.display()
        );
        return Ok(());
    } else if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        let api_key = online_api_key()?;
        let fixture = fixtures()
            .into_iter()
            .find(|fixture| fixture.name == "count-in-files")
            .expect("count-in-files fixture exists");
        let workdir = materialize_fixture(&fixture)?;
        let child_home =
            std::env::temp_dir().join(format!("delegation-eval-home-{}", Uuid::new_v4()));
        fs::create_dir_all(child_home.join(".local/share/agent/traces"))?;
        let env = child_env(&agent_bin_dir()?, &child_home, Some(&api_key))?;
        let command = vec![
            "agent".to_string(),
            "--model".to_string(),
            PROCESS_CHILD_ALIAS.to_string(),
            child_prompt.to_string(),
        ];
        let started = Instant::now();
        // stdin MUST be detached: a one-shot `agent` with non-terminal
        // stdin reads it to EOF as optional input data
        // (main.rs `prompt_with_optional_stdin`), so an inherited
        // never-closing stdin hangs the child forever. The runtime's own
        // Eval effect already detaches child stdin (interpreter.rs), which
        // is why model-issued `agent ...` shell commands do not hit this.
        let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_agent"))
            .args(&command[1..])
            .current_dir(&workdir)
            .env_clear()
            .envs(&env)
            .stdin(std::process::Stdio::null())
            .output()
            .await
            .context("spawning the process child")?;
        let (child_runs, child_cost_micro_usd, child_total_tokens) =
            sweep_child_traces(&child_home)?;
        let record = ProbeRecord {
            child_command: command,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            exit_ok: output.status.success(),
            wall_ms: started.elapsed().as_millis() as u64,
            child_runs,
            child_cost_micro_usd,
            child_total_tokens,
            recorded_at: Utc::now().to_rfc3339(),
        };
        let _ = fs::remove_dir_all(&workdir);
        let _ = fs::remove_dir_all(&child_home);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let serialized = serde_json::to_string_pretty(&record)?;
        anyhow::ensure!(
            !serialized.contains(api_key.as_str()),
            "probe recording embeds the API key; not writing it"
        );
        fs::write(&path, serialized)?;
        record
    };

    let counted_correctly = needle_present(record.stdout.trim(), "7");
    println!(
        "process child capability: exit_ok={} answer={:?} correct={} cost={} tokens={} wall={:.1}s",
        record.exit_ok,
        record.stdout.trim(),
        counted_correctly,
        agent_core::format_micro_usd(record.child_cost_micro_usd),
        record.child_total_tokens,
        record.wall_ms as f64 / 1000.0,
    );
    Ok(())
}

// --- plumbing tests (always on, offline, scripted providers) ----------------------
//
// PLUMBING-ONLY: these validate the harness machinery with scripted model
// turns. They say nothing about real model behavior — that is exactly what
// scripted arms cannot measure (the t-1354 point) and what the recorded
// matrix above exists for.

struct ScriptedProvider {
    turns: Mutex<VecDeque<Response>>,
}

impl ScriptedProvider {
    fn new(turns: Vec<Response>) -> Self {
        Self {
            turns: Mutex::new(turns.into()),
        }
    }

    fn text(content: &str) -> Response {
        Self::response(content, Vec::new())
    }

    fn calls(tool_calls: Vec<ToolCall>) -> Response {
        Self::response("", tool_calls)
    }

    fn response(content: &str, tool_calls: Vec<ToolCall>) -> Response {
        Response {
            content: content.into(),
            tool_calls,
            finish_reason: Some(agent_core::FinishReason::Stop),
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            metadata: Default::default(),
        }
    }
}

#[async_trait]
impl ChatProvider for ScriptedProvider {
    async fn chat(
        &self,
        _model: &Model,
        _tools: &[agent_core::provider::ToolSpec],
        _messages: &[ChatMessage],
    ) -> Result<Response> {
        self.turns
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("scripted provider exhausted"))
    }
}

/// Harness counting: sub-infer attribution (parent_op_id lineage), flags,
/// and needle scoring — on a scripted delegation.
#[tokio::test]
async fn plumbing_counts_infer_delegation_and_flags() -> Result<()> {
    let provider = Arc::new(ScriptedProvider::new(vec![
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-1",
            "infer",
            serde_json::json!({ "model": DEFAULT_CHILD_MODEL, "prompt": "what is 17*23?" }),
        )]),
        ScriptedProvider::text("391"),
        ScriptedProvider::text("391"),
    ]));
    let restraint = fixtures()
        .into_iter()
        .find(|fixture| fixture.name == "restraint-direct")
        .expect("restraint fixture exists");
    let workdir = materialize_fixture(&restraint)?;
    let run = run_cell(
        provider,
        None,
        DEFAULT_PARENT_MODEL,
        vec![
            ChatMessage::system("plumbing"),
            ChatMessage::user("What is 17 * 23? Answer with just the number."),
        ],
        &workdir,
        BTreeMap::new(),
        agent_core::RuntimeGuidance::disabled(),
    )
    .await?;
    let _ = fs::remove_dir_all(&workdir);
    let metrics = metrics_from_events(&run.events, &run.content, &["391"])?;
    assert!(metrics.success);
    assert_eq!(metrics.turns, 2, "two parent-loop provider turns");
    assert_eq!(metrics.sub_infers, 1, "one delegation, by lineage");
    assert_eq!(metrics.proc_delegations, 0);
    assert_eq!(
        appropriateness_flag(Arm::ToolGuided, Stance::Restraint, &metrics),
        "OVER-DELEGATED"
    );
    assert_eq!(
        appropriateness_flag(Arm::ToolGuided, Stance::Help, &metrics),
        "-"
    );
    assert_eq!(
        appropriateness_flag(Arm::Baseline, Stance::Restraint, &metrics),
        "-",
        "the baseline arm has no mechanism to misuse"
    );

    // Costed: both models are in the fixture pricing table.
    assert_eq!(metrics.usage.uncosted_infer_calls, 0);
    assert!(metrics.usage.cost_micro_usd.is_some());
    Ok(())
}

/// Harness plumbing for arm D: a scripted parent invokes the REAL `agent`
/// binary through the shell tool (`--version`: no network, no key). Pins
/// that the PATH symlink + allowlist env deliver a working `agent` command
/// to shell children, and that the invocation is counted as a process
/// delegation.
#[tokio::test]
async fn plumbing_process_delegation_runs_agent_binary() -> Result<()> {
    let provider = Arc::new(ScriptedProvider::new(vec![
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-1",
            "shell",
            serde_json::json!({ "command": "agent --version" }),
        )]),
        ScriptedProvider::text("done"),
    ]));
    let bin_dir = agent_bin_dir()?;
    let child_home = std::env::temp_dir().join(format!("delegation-eval-home-{}", Uuid::new_v4()));
    fs::create_dir_all(&child_home)?;
    let workdir = std::env::temp_dir().join(format!("delegation-eval-fx-{}", Uuid::new_v4()));
    fs::create_dir_all(&workdir)?;
    let run = run_cell(
        provider,
        None,
        DEFAULT_PARENT_MODEL,
        vec![ChatMessage::system("plumbing"), ChatMessage::user("go")],
        &workdir,
        child_env(&bin_dir, &child_home, None)?,
        agent_core::RuntimeGuidance::disabled(),
    )
    .await?;
    let _ = fs::remove_dir_all(&workdir);
    let _ = fs::remove_dir_all(&child_home);
    assert_eq!(run.content, "done");
    let metrics = metrics_from_events(&run.events, &run.content, &[])?;
    assert_eq!(metrics.eval_calls, 1);
    assert_eq!(
        metrics.proc_delegations, 1,
        "agent invocations are counted as process delegations"
    );
    // The binary actually ran: its version output came back as the tool
    // result (a failed spawn would have bound an error instead).
    let eval_result = run.events.iter().find_map(|event| match event {
        Event::EvalResult { result, .. } => Some(result.clone()),
        _ => None,
    });
    let stdout = eval_result
        .as_ref()
        .and_then(|value| value["stdout"].as_str())
        .unwrap_or_default()
        .to_string();
    assert!(
        stdout.contains("agent"),
        "child `agent --version` ran via allowlisted PATH: {eval_result:?}"
    );
    Ok(())
}

/// Recording round-trip: what the online writer persists, the offline
/// loader restores byte-faithfully enough to score.
#[tokio::test]
async fn plumbing_recording_roundtrip() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("delegation-eval-rec-{}", Uuid::new_v4()));
    let path = cell_path(&dir, "roundtrip", Arm::ToolGuided);
    let meta = CellMeta {
        fixture: "roundtrip".into(),
        arm: Arm::ToolGuided.label().into(),
        parent_model: DEFAULT_PARENT_MODEL.into(),
        child_model: DEFAULT_CHILD_MODEL.into(),
        wall_ms: 1234,
        final_content: "391".into(),
        child_runs: 0,
        child_cost_micro_usd: 0,
        child_total_tokens: 0,
        recorded_at: Utc::now().to_rfc3339(),
    };
    let events = vec![Event::AgentDone {
        run_id: "r".into(),
        usage: Some(RunUsage::default()),
        timestamp: Utc::now(),
    }];
    write_cell_recording(&path, &meta, &events)?;
    let (loaded_meta, loaded_events) = load_cell_recording(&path)?;
    assert_eq!(loaded_meta.final_content, meta.final_content);
    assert_eq!(loaded_meta.wall_ms, meta.wall_ms);
    assert_eq!(loaded_events, events);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Committed recordings must be credential-free. Checks the OpenRouter key
/// prefix and, when a key is present in the environment, the key itself.
#[test]
fn recordings_are_credential_free() -> Result<()> {
    let dir = recordings_dir()?;
    if !dir.exists() {
        return Ok(());
    }
    let live_key = std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("AGENT_API_KEY"))
        .ok();
    for entry in fs::read_dir(&dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let content = fs::read_to_string(&path)?;
        assert!(
            !content.contains("sk-or-"),
            "{}: contains an OpenRouter key marker",
            path.display()
        );
        if let Some(key) = live_key.as_deref() {
            assert!(
                !content.contains(key),
                "{}: contains the live API key",
                path.display()
            );
        }
    }
    Ok(())
}

/// Task text is arm-independent by construction (the model, not the
/// prompt, chooses delegation) — pin it so future edits keep the contract.
#[test]
fn fixture_tasks_never_mention_delegation() {
    for fixture in fixtures() {
        for banned in ["infer", "delegate", "child model", "agent --model"] {
            assert!(
                !fixture.task.to_lowercase().contains(banned),
                "{}: task text must not steer delegation (found {banned:?})",
                fixture.name
            );
        }
    }
}

#[test]
fn needle_matching_is_boundary_aware_for_numbers() {
    assert!(needle_present("the answer is 391.", "391"));
    assert!(needle_present("391", "391"));
    assert!(!needle_present("13917", "391"));
    assert!(!needle_present("39.1", "391"));
    assert!(needle_present("total was 7 lines", "7"));
    assert!(!needle_present("27 lines", "7"));
    assert!(!needle_present("7.5 lines", "7"));
    assert!(needle_present("revenue grew 12% here", "12%"));
    assert!(needle_present("fix config.env now", "config.env"));
}
