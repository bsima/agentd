//! Online behavioral GC eval (t-1349).
//!
//! The offline matrix (gc_evals.rs) measures RETENTION arithmetic: what a
//! strategy keeps, on frozen windows. This harness asks the behavioral
//! question: what does garbage collection do to a REAL agent mid-task —
//! completion, coherence, recovery — when collections fire under genuine
//! context pressure? Concretely (the questions the behavioral section of
//! evals/gc/README.md answers):
//!
//! 1. does the offline ranking (stack default, semantic for tangents)
//!    survive contact with real behavior?
//! 2. when a strategy drops an early tool result the task needs again late,
//!    does the model recover (re-fetch, recall) — and what does recovery
//!    cost in turns and tokens?
//! 3. after an on-task tangent is collected, does the model return to the
//!    thread coherently?
//! 4. does memory discipline (remember early, recall late) survive GC
//!    pressure, and does the recall-overlap write-barrier (t-1351) fire in
//!    real sessions?
//!
//! Arms = GC strategy, everything else identical: `none` (control), `ring`,
//! `mark-sweep`, `stack`, `semantic` (cited-keep on, the default) — all at
//! the runtime defaults otherwise (`--gc-cache preserve`, `--gc-timing
//! threshold`, threshold 0.85) with a context budget SMALL enough that
//! collections fire mid-session. That firing is asserted per cell: a GC
//! cell where no collection ran measures nothing.
//!
//! Semantic cells use a deterministic bag-of-tokens embedder (the offline
//! harness's stance, evals/gc/README.md "Semantic strategy cells"):
//! OpenRouter has no embeddings endpoint and record/replay requires
//! identical vectors both sides, so the mock — vocabulary overlap as
//! cosine similarity — stands in for a real embedding model. Semantic
//! rows measure the strategy under that stand-in, and the README says so.
//!
//! Scoring is read from the trace, never estimated: programmatic needles on
//! the final answer, re-fetch counts (EvalCall commands touching the
//! needle's source beyond the task's own allowance — the coherence proxy:
//! did the model have to re-acquire something it had been given?),
//! gc_collect events (count, reason markers, dropped counts,
//! recall-overlap write-barrier fields), remember/recall usage
//! (StoreCall/RetrieveCall), and the RunUsage rollup on AgentDone (t-1334).
//! An LLM judge scores coherence where the programmatic checks are crude
//! (staying on task, redundant work, grounding); judge responses are REAL
//! recorded model verdicts in `evals/gc/judge/behavioral.jsonl` — unlike
//! the hand-written placeholders shipped for the offline matrix — and
//! replay offline by content-hash key, the existing recorded-judge stance.
//!
//! Online/offline: online (`RUN_AGENT_ONLINE_EVAL=1`) runs each
//! (fixture, arm) cell against a real provider and records the FULL event
//! trace to `evals/gc/recordings/`. Offline (the default) replays those
//! traces through effect-id replay (`IrReplayTrace`) — Infer, Eval, Store,
//! and Retrieve results come from the recording — and asserts the replay
//! reproduces the recording per cell: same final answer, same metrics,
//! including the gc_collect stream (GC re-runs deterministically during
//! replay: same windows, same mock embedder, same GcState threading).
//! There are NO hand-written behavioral recordings; offline without
//! recordings is a documented no-op and the always-on tests below are
//! plumbing-only.
//!
//! Guidance axis (t-1364): t-1349 found the offline champion (stack)
//! thrashing and three of four strategies confabulating evicted content —
//! and remember/recall discipline rescuing everything. The follow-up
//! hypothesis: GUIDANCE dominates STRATEGY — the shipped runtime-guidance
//! fragment (t-1359: GC-awareness §2.4 + memory-discipline §2.2 blocks)
//! changes behavior more than swapping collectors does, and eliminates
//! confabulation. Each t-1364 cell is (fixture, arm, guided, sample):
//! guided cells run the SHIPPED `RuntimeGuidance::default()`, unguided
//! cells `RuntimeGuidance::disabled()`. The t-1349 recordings could NOT be
//! reused as the unguided arms: commit a6592f8 (t-1359 step 1) rewrote the
//! shell/remember/recall/infer tool descriptions, which enter every cell's
//! provider offer — so the unguided arms were re-recorded on the current
//! descriptions, and the legacy recordings are replayed separately (still
//! a valid regression check; no longer a valid comparison arm). Guidance
//! is prompt-bytes only, so replay works either way — but GC re-runs the
//! live token-sensitive collector during replay, so each cell's replay
//! must run the guidance setting it was recorded under (meta carries it).
//! New metrics for the t-1364 question: `prem` (remember calls BEFORE the
//! first collection — proactive saves, the §2.2 behavior) and `cfab` (the
//! final answer asserts the fixture's claim marker with a wrong value —
//! fabricated content for evicted material; needle-absence programmatic
//! check, corroborated by the judge's grounded_final_answer).

use agent_core::gc::SemanticGc;
use agent_core::{
    agent_loop_ir_with_options, run_ir_sequential_with_store_and_replay, ChatMessage, ChatProvider,
    Embedder, EnvPolicy, EvalConfig, Event, GcMode, GcTiming, InMemoryStore, IrReplayTrace,
    MarkSweepGc, MemorySource, Model, PassiveHydrationConfig, Pricing, PricingTable,
    ProviderClient, ProviderConfig, ReplayOnlyProvider, Response, RingGc, RunUsage,
    RuntimeGuidance, SeqConfig, SourceRegistry, StackFrameGc, ToolCall, TraceLogger,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Generous ceiling: the longest fixture script is 9 tool steps + answer,
/// and models spend extra text-only turns on commentary; the margin absorbs
/// those without runaway spend (the first recording pass clipped a control
/// cell at 20).
const MAX_TURNS: usize = 26;
/// Cheap-but-capable model (the t-1354 choice); overridable with
/// AGENT_EVAL_PARENT_MODEL. OpenRouter id.
const DEFAULT_MODEL: &str = "anthropic/claude-haiku-4.5";
/// Coherence judge; overridable with AGENT_JUDGE_MODEL. OpenRouter id.
const DEFAULT_JUDGE_MODEL: &str = "anthropic/claude-haiku-4.5";
/// Runtime GC defaults the cells run under (docs/GC.md).
const GC_THRESHOLD: f32 = 0.85;

fn repo_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("could not resolve repo root"))
}

fn recordings_dir() -> Result<PathBuf> {
    Ok(repo_root()?.join("evals/gc/recordings"))
}

fn judge_book_path() -> Result<PathBuf> {
    Ok(repo_root()?.join("evals/gc/judge/behavioral.jsonl"))
}

/// Fixture pricing for the default model id (OpenRouter list price, USD per
/// Mtok). Env-overridden models run uncosted — absent pricing is never
/// guessed.
fn pricing_table() -> PricingTable {
    let mut table = PricingTable::default();
    table.insert(DEFAULT_MODEL, Pricing::from_usd_per_mtok(1.0, 5.0).unwrap());
    table
}

// --- deterministic embedder for semantic cells --------------------------------

/// Bag-of-tokens embedder: each token FNV-hashes into one of 64 buckets, so
/// cosine similarity is vocabulary overlap. The same mock the offline
/// harness primes GcState with (gc_evals.rs, t-1350) — deterministic, no
/// RNG, identical vectors in record and replay.
struct BagOfTokensEmbedder;

fn bag_of_tokens_vector(text: &str) -> Vec<f32> {
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

#[async_trait]
impl Embedder for BagOfTokensEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| bag_of_tokens_vector(text))
            .collect())
    }

    fn model_id(&self) -> &str {
        "eval-bag-of-tokens-64"
    }
}

// --- arms ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    NoGc,
    Ring,
    MarkSweep,
    Stack,
    Semantic,
}

impl Arm {
    const ALL: [Arm; 5] = [
        Arm::NoGc,
        Arm::Ring,
        Arm::MarkSweep,
        Arm::Stack,
        Arm::Semantic,
    ];

    fn label(&self) -> &'static str {
        match self {
            Self::NoGc => "none",
            Self::Ring => "ring",
            Self::MarkSweep => "mark-sweep",
            Self::Stack => "stack",
            Self::Semantic => "semantic",
        }
    }

    fn from_label(label: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|arm| arm.label() == label)
            .ok_or_else(|| anyhow!("unknown arm label {label}"))
    }

    /// The strategy under the runtime defaults: cache preserve, and for
    /// semantic the cited-keep default plus the deterministic embedder.
    fn gc_mode(&self) -> GcMode {
        match self {
            Self::NoGc => GcMode::None,
            Self::Ring => GcMode::Ring(RingGc {
                preserve_prefix: true,
            }),
            Self::MarkSweep => GcMode::MarkSweep(MarkSweepGc {
                preserve_prefix: true,
            }),
            Self::Stack => GcMode::Stack(StackFrameGc {
                preserve_prefix: true,
            }),
            Self::Semantic => GcMode::Semantic(SemanticGc {
                preserve_prefix: true,
                embedder: Some(Arc::new(BagOfTokensEmbedder)),
                ..Default::default()
            }),
        }
    }

    fn collects(&self) -> bool {
        !matches!(self, Self::NoGc)
    }
}

// --- fixtures ------------------------------------------------------------------

struct Fixture {
    name: &'static str,
    /// Identical across arms — the strategy, not the task, is the variable.
    task: String,
    /// The SMALL context budget (tokens) the cell runs under; sized so the
    /// full session overflows it 2x-3x and collections fire mid-session.
    context_budget: usize,
    /// Programmatic success needles on the final answer. Pure-numeric
    /// needles match on token boundaries, text needles by substring.
    /// CONTRACT: `needles[0]` is the evictable claim value (the early tool
    /// result the task needs again late) — the confabulation check keys on
    /// it.
    needles: Vec<&'static str>,
    /// The final-line marker the task's answer format requires ("ACCESS",
    /// "CATEGORIES", "DEPLOY"). Confabulation = the marker is present (the
    /// model ASSERTED an answer) while the claim value is wrong/absent —
    /// fabricated content standing in for evicted material. A cell that
    /// gives no final line (thrash, turn-cap clip) is a non-answer, not a
    /// confabulation.
    claim_marker: &'static str,
    /// Ordered needles that must appear in this order after the final
    /// answer marker (F2's category ranking); empty = no order check.
    ordered_needles: Vec<&'static str>,
    /// Substring identifying shell commands that touch the needle's source
    /// (e.g. the access-code file). Occurrences beyond `probe_allowance`
    /// count as re-fetches — the model re-acquiring something it had been
    /// given, the coherence-cost signal.
    probe: &'static str,
    /// How many probe-matching commands the task itself asks for.
    probe_allowance: usize,
    /// Deterministic working-directory content.
    files: Vec<(&'static str, String)>,
}

/// Deterministic topical filler: repeatable pseudo-prose from one
/// vocabulary pool, no RNG (the gc_evals pattern — distinct pools are
/// semantically distant under the bag-of-tokens embedder, like distinct
/// topics under a real one).
fn filler(words: &[&str], seed: usize, chars: usize) -> String {
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

/// Filler broken into lines, so tool output looks like a real file.
fn filler_lines(words: &[&str], seed: usize, lines: usize, prefix: &str) -> String {
    (0..lines)
        .map(|line| format!("{prefix}{}", filler(words, seed + line * 13, 64)))
        .collect::<Vec<_>>()
        .join("\n")
}

const MANUAL_WORDS: [&str; 12] = [
    "batch",
    "pipeline",
    "operator",
    "manual",
    "procedure",
    "checklist",
    "throughput",
    "queue",
    "schedule",
    "rotation",
    "shift",
    "handoff",
];

const INGEST_WORDS: [&str; 12] = [
    "record",
    "ingest",
    "parser",
    "payload",
    "sequence",
    "offset",
    "commit",
    "partition",
    "segment",
    "flush",
    "worker",
    "cursor",
];

const APP_WORDS: [&str; 12] = [
    "request", "handler", "session", "router", "upstream", "latency", "retry", "gateway", "socket",
    "buffer", "thread", "dispatch",
];

const POEM_WORDS: [&str; 12] = [
    "moonlight",
    "meadow",
    "sorrow",
    "blossom",
    "whisper",
    "autumn",
    "ember",
    "lullaby",
    "river",
    "twilight",
    "petal",
    "longing",
];

const BUILD_WORDS: [&str; 12] = [
    "compile",
    "linker",
    "artifact",
    "module",
    "dependency",
    "target",
    "cache",
    "codegen",
    "symbol",
    "binary",
    "toolchain",
    "incremental",
];

const ACCESS_CODE: &str = "MX-7749-KESTREL";
const DEPLOY_TOKEN: &str = "TOKEN-9QX-RAVEN-7734";

/// Batch log `index` with exactly `ok_lines` lines reading exactly
/// "STATUS: OK" amid bulky deterministic filler. The OK line is the WHOLE
/// line so "count lines that read exactly `STATUS: OK`" has one right
/// answer under every plausible reading (`grep -c` and `grep -cx` agree) —
/// the first recording pass showed models interpreting "containing exactly"
/// as exact-line-match and scoring 0 against suffix-decorated OK lines.
fn batch_log(index: usize, ok_lines: usize) -> String {
    let mut out = String::new();
    for line in 0..20 {
        out.push_str(&format!(
            "batch-{index} {}\n",
            filler(&INGEST_WORDS, index * 100 + line * 7, 64)
        ));
        if line < ok_lines {
            out.push_str("STATUS: OK\n");
        }
    }
    out
}

/// Build log `index` with exactly `warn_lines` "WARN:" lines amid filler.
fn build_log(index: usize, warn_lines: usize) -> String {
    let mut out = String::new();
    for line in 0..22 {
        out.push_str(&format!(
            "step-{index} {}\n",
            filler(&BUILD_WORDS, index * 90 + line * 11, 64)
        ));
        if line < warn_lines {
            out.push_str(&format!(
                "WARN: deprecated symbol in module m{index}{line}\n"
            ));
        }
    }
    out
}

/// app.log: ERROR lines in three categories (timeout 7, checksum 4,
/// quota 2) interleaved with INFO filler — the ranking the final answer
/// must reproduce.
fn app_log() -> String {
    let mut lines: Vec<String> = Vec::new();
    for index in 0..7 {
        lines.push(format!(
            "ERROR timeout upstream while {}",
            filler(&APP_WORDS, index * 17, 48)
        ));
    }
    for index in 0..4 {
        lines.push(format!("ERROR checksum mismatch in segment s{index}"));
    }
    for index in 0..2 {
        lines.push(format!("ERROR quota exceeded for tenant t{index}"));
    }
    for index in 0..11 {
        lines.push(format!("INFO {}", filler(&APP_WORDS, 500 + index * 19, 56)));
    }
    // Deterministic interleave: rotate by a fixed stride so categories mix.
    let mut out = Vec::with_capacity(lines.len());
    let mut cursor = 0usize;
    let mut taken = vec![false; lines.len()];
    for _ in 0..lines.len() {
        while taken[cursor % lines.len()] {
            cursor += 1;
        }
        out.push(lines[cursor % lines.len()].clone());
        taken[cursor % lines.len()] = true;
        cursor += 7;
    }
    out.join("\n") + "\n"
}

fn fixtures() -> Vec<Fixture> {
    vec![
        // Fixture 1 — early-needle: a long multi-step task where an early
        // tool result (the access code) is needed again at the very end.
        // manual.txt is bulky ballast BEFORE the needle so the needle frame
        // sits in the evictable interior, not the preserve-mode pinned
        // prefix. OK counts: 3+5+2+6+4+1 = 21.
        Fixture {
            name: "early-needle",
            task: "Complete these steps strictly in order, one shell command per step, \
                   with at most one short sentence of commentary each:\n\
                   1. Run `cat manual.txt` (the operations manual; no action needed).\n\
                   2. Run `cat config/access-code.txt` — it prints the batch access code.\n\
                   3-8. For each of logs/batch-1.log through logs/batch-6.log in order, \
                   run `cat <file>` and count the lines that read exactly `STATUS: OK`.\n\
                   9. Finish with one line exactly of the form: \
                   `ACCESS <code> TOTAL <sum>` — the access code from step 2 and the \
                   sum of the six OK-line counts."
                .into(),
            context_budget: 2000,
            needles: vec![ACCESS_CODE, "21"],
            ordered_needles: vec![],
            claim_marker: "ACCESS",
            probe: "access-code",
            probe_allowance: 1,
            files: vec![
                ("manual.txt", filler_lines(&MANUAL_WORDS, 1, 36, "manual: ")),
                ("config/access-code.txt", format!("{ACCESS_CODE}\n")),
                ("logs/batch-1.log", batch_log(1, 3)),
                ("logs/batch-2.log", batch_log(2, 5)),
                ("logs/batch-3.log", batch_log(3, 2)),
                ("logs/batch-4.log", batch_log(4, 6)),
                ("logs/batch-5.log", batch_log(5, 4)),
                ("logs/batch-6.log", batch_log(6, 1)),
            ],
        },
        // Fixture 2 — tangent-return: analyze a log, take a deliberate
        // bulky tangent (two poems, distinct vocabulary — semantic's home
        // turf), then return to the thread and answer from the EARLY log
        // read. The reminder + grep steps re-engage the original topic so
        // the recent window is not pure tangent at the final collections.
        Fixture {
            name: "tangent-return",
            task: "Complete these steps strictly in order, one shell command per step, \
                   with at most one short sentence of commentary each:\n\
                   1. Run `cat app.log` and note how many ERROR lines mention each \
                   category: timeout, checksum, quota.\n\
                   2. Sidebar: run `cat poems/verse-1.txt` and summarize the poem in \
                   one sentence.\n\
                   3. Run `cat poems/verse-2.txt` and summarize it in one sentence.\n\
                   4. Back to the log work: run `cat notes/reminder.txt`.\n\
                   5. Run `grep -c ERROR app.log` as a sanity check of the total.\n\
                   6. Finish with one line exactly of the form: \
                   `CATEGORIES: <first>,<second>,<third>` — the three categories \
                   ordered by how many ERROR lines mention each (most first), using \
                   your counts from step 1."
                .into(),
            context_budget: 1600,
            needles: vec!["CATEGORIES", "timeout", "checksum", "quota"],
            ordered_needles: vec!["timeout", "checksum", "quota"],
            claim_marker: "CATEGORIES",
            probe: "app.log",
            probe_allowance: 2,
            files: vec![
                ("app.log", app_log()),
                ("poems/verse-1.txt", filler_lines(&POEM_WORDS, 3, 52, "")),
                ("poems/verse-2.txt", filler_lines(&POEM_WORDS, 211, 52, "")),
                (
                    "notes/reminder.txt",
                    "Reminder: the deliverable is the ERROR category ranking from \
                     app.log — timeout, checksum, quota — ordered by how many ERROR \
                     lines mention each. Use the counts you already collected in \
                     step 1.\n"
                        .into(),
                ),
            ],
        },
        // Fixture 3 — memory-discipline: remember early, recall late,
        // across GC pressure. build/toolchain.txt is ballast before the
        // token read (same pinned-prefix reasoning as fixture 1). WARN
        // counts: 2+0+3+1 = 6.
        Fixture {
            name: "memory-discipline",
            task: "Complete these steps strictly in order, one tool call per step, \
                   with at most one short sentence of commentary each:\n\
                   1. Run `cat build/toolchain.txt` (the build manifest; no action \
                   needed).\n\
                   2. Run `cat secrets/deploy-token.txt` — it prints the deploy token.\n\
                   3. Immediately save the token with the `remember` tool: name it \
                   `deploy-token` and make the content exactly the token string.\n\
                   4-7. For each of build/step-1.log through build/step-4.log in \
                   order, run `cat <file>` and count the lines containing `WARN:`.\n\
                   8. Use the `recall` tool with query `deploy token` to retrieve the \
                   token.\n\
                   9. Finish with one line exactly of the form: \
                   `DEPLOY <token> WARNS <total>` — the deploy token and the sum of \
                   the four WARN-line counts."
                .into(),
            context_budget: 1700,
            needles: vec![DEPLOY_TOKEN, "6"],
            ordered_needles: vec![],
            claim_marker: "DEPLOY",
            probe: "deploy-token",
            probe_allowance: 1,
            files: vec![
                (
                    "build/toolchain.txt",
                    filler_lines(&BUILD_WORDS, 5, 30, "toolchain: "),
                ),
                ("secrets/deploy-token.txt", format!("{DEPLOY_TOKEN}\n")),
                ("build/step-1.log", build_log(1, 2)),
                ("build/step-2.log", build_log(2, 0)),
                ("build/step-3.log", build_log(3, 3)),
                ("build/step-4.log", build_log(4, 1)),
            ],
        },
    ]
}

fn system_prompt() -> &'static str {
    "You are a careful software agent with shell, remember, and recall tools, \
     working in the current directory. Follow the task steps exactly and in \
     order, keep commentary to one short sentence per step, and end with the \
     exact final line the task requires."
}

/// Needle matching (the delegation-eval rule): pure-numeric needles must sit
/// on token boundaries (so "21" never matches inside "213"); everything else
/// is a case-insensitive substring check.
fn needle_present(content: &str, needle: &str) -> bool {
    if needle.is_empty() || !needle.chars().all(|c| c.is_ascii_digit()) {
        return content.to_lowercase().contains(&needle.to_lowercase());
    }
    let bytes = content.as_bytes();
    let mut from = 0;
    while let Some(pos) = content[from..].find(needle) {
        let at = from + pos;
        let end = at + needle.len();
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

/// The F2 order check: after the last "CATEGORIES" marker, the ordered
/// needles must each appear, in order.
fn ordered_needles_present(content: &str, ordered: &[&str]) -> bool {
    if ordered.is_empty() {
        return true;
    }
    let lower = content.to_lowercase();
    let Some(marker) = lower.rfind("categories") else {
        return false;
    };
    let tail = &lower[marker..];
    let mut from = 0;
    for needle in ordered {
        match tail[from..].find(&needle.to_lowercase()) {
            Some(pos) => from += pos + needle.len(),
            None => return false,
        }
    }
    true
}

fn fixture_success(fixture: &Fixture, content: &str) -> bool {
    fixture
        .needles
        .iter()
        .all(|needle| needle_present(content, needle))
        && ordered_needles_present(content, &fixture.ordered_needles)
}

/// The confabulation flag (t-1364): the final answer ASSERTS the fixture's
/// answer format (claim marker present) while the claim itself is wrong —
/// for order fixtures the category order, otherwise `needles[0]` (the
/// evicted early tool result). This is t-1349 finding 3 made programmatic:
/// a fabricated `ACCESS CDBH92 ...` flags, a thrash cell that never
/// answers does not, and a wrong-arithmetic-but-right-code answer does not
/// (arithmetic slips are not fabricated recall of evicted content).
fn confabulated(fixture: &Fixture, content: &str) -> bool {
    if !content
        .to_lowercase()
        .contains(&fixture.claim_marker.to_lowercase())
    {
        return false;
    }
    let claim_ok = if fixture.ordered_needles.is_empty() {
        needle_present(content, fixture.needles[0])
    } else {
        ordered_needles_present(content, &fixture.ordered_needles)
    };
    !claim_ok
}

// --- cell runner -----------------------------------------------------------------

fn materialize_fixture(fixture: &Fixture) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("gc-behavior-fx-{}", Uuid::new_v4()));
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

struct CellRun {
    content: String,
    events: Vec<Event>,
    wall_ms: u64,
}

/// The per-cell knobs `run_cell` needs beyond provider/replay/dirs.
struct CellSpec {
    model: String,
    gc: GcMode,
    context_budget: usize,
    prompt: Vec<ChatMessage>,
    /// The t-1364 axis: guided cells run the SHIPPED
    /// `RuntimeGuidance::default()` (empty delegate catalog — exactly what
    /// a stock deployment gets), unguided cells run `disabled()`. Replay
    /// must use the recording's setting: guidance is prompt bytes, and the
    /// live collector re-run during replay is token-sensitive.
    guidance: RuntimeGuidance,
}

fn cell_guidance(guided: bool) -> RuntimeGuidance {
    if guided {
        RuntimeGuidance::default()
    } else {
        RuntimeGuidance::disabled()
    }
}

/// One session: the memory-enabled agent loop under the arm's GC mode at
/// the fixture's small budget, gc_log on so gc_collect events land in the
/// trace. Shell children get an allowlist env of PATH only — never a key —
/// so recordings are credential-free by construction.
async fn run_cell(
    provider: Arc<dyn ChatProvider>,
    replay: Option<&IrReplayTrace>,
    spec: CellSpec,
    workdir: &Path,
    memory_dir: &Path,
) -> Result<CellRun> {
    let trace_path = std::env::temp_dir().join(format!("gc-behavior-{}.jsonl", Uuid::new_v4()));
    let trace = TraceLogger::new(Uuid::new_v4().to_string(), trace_path.clone());
    let mut extra_env = BTreeMap::new();
    extra_env.insert(
        "PATH".into(),
        std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".into()),
    );
    let config = SeqConfig {
        approvals: Default::default(),
        // Per-cell (t-1364): the guided arms run the shipped fragment, the
        // unguided arms and all t-1349 legacy replays run disabled — GC
        // replay re-runs the live token-sensitive collector, so the setting
        // must match what the recording ran under (CellMeta.guided).
        guidance: spec.guidance.clone(),
        tools: Default::default(),
        provider,
        hydration: SourceRegistry::new().register_backend(MemorySource::new(memory_dir.into())),
        passive_hydration: PassiveHydrationConfig::default(),
        trace: trace.clone(),
        eval: EvalConfig {
            shell: "/bin/sh".into(),
            cwd: Some(workdir.to_path_buf()),
            timeout: Duration::from_secs(120),
            env: EnvPolicy::AllowList {
                names: Vec::new(),
                extra: extra_env,
            },
            ..EvalConfig::default()
        },
        replay: None,
        trace_full_prompt_ir: false,
        trace_full_payloads: false,
        gc: spec.gc,
        gc_threshold: GC_THRESHOLD,
        gc_log: true,
        gc_timing: GcTiming::Threshold,
        context_budget: spec.context_budget,
        pricing: pricing_table(),
    };
    let machine = agent_loop_ir_with_options(Model(spec.model), spec.prompt, MAX_TURNS, true);
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

// --- metrics ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct CellMetrics {
    /// Parent-loop provider turns (InferCall without a parent).
    turns: usize,
    eval_calls: usize,
    /// Shell commands identical to one already executed this session.
    repeat_evals: usize,
    /// Probe-matching commands beyond the fixture's allowance: the model
    /// re-acquiring the needle it had been given.
    needle_refetches: usize,
    remember_calls: usize,
    /// Remember calls issued BEFORE the first collection fired — proactive
    /// saves (the §2.2 guided behavior), as opposed to saves scrambled
    /// together after eviction already happened. On a `none` cell every
    /// remember is trivially proactive.
    proactive_remembers: usize,
    recall_calls: usize,
    /// gc_collect events: how many collections actually fired.
    collections: usize,
    /// gc_collect reason distribution (scheduled / backstop / overflow).
    reasons: BTreeMap<String, usize>,
    /// Sum of gc_collect dropped_count: messages evicted across the session.
    dropped_total: u64,
    /// Sum of gc_collect recall_overlap_events: recall-overlap write-barrier
    /// firings (t-1351).
    overlap_total: u64,
    /// Max gc_collect recall_hot: hot-set size high-water mark.
    recall_hot_max: u64,
    /// Max gc_collect markers: in-window eviction-marker high-water mark
    /// (t-1360). 0 on recordings made before the marker mechanism (the
    /// field is absent there — see the pre-marker replay note in
    /// `replay_cell`).
    markers_max: u64,
    /// gc_collect events whose marker emission was suppressed (no room for
    /// even the coalesced line at this budget).
    markers_suppressed: u64,
    usage: RunUsage,
    success: bool,
    /// The final answer asserts the claim marker with a wrong claim value
    /// (see [`confabulated`]) — fabricated content for evicted material.
    confabulated: bool,
}

fn metrics_from_events(events: &[Event], content: &str, fixture: &Fixture) -> Result<CellMetrics> {
    let mut metrics = CellMetrics {
        turns: 0,
        eval_calls: 0,
        repeat_evals: 0,
        needle_refetches: 0,
        remember_calls: 0,
        proactive_remembers: 0,
        recall_calls: 0,
        collections: 0,
        reasons: BTreeMap::new(),
        dropped_total: 0,
        overlap_total: 0,
        recall_hot_max: 0,
        markers_max: 0,
        markers_suppressed: 0,
        usage: RunUsage::default(),
        success: fixture_success(fixture, content),
        confabulated: confabulated(fixture, content),
    };
    let mut seen_commands: BTreeSet<String> = BTreeSet::new();
    let mut probe_hits = 0usize;
    let mut done_usage: Option<RunUsage> = None;
    for event in events {
        match event {
            Event::InferCall { parent_op_id, .. } => {
                if parent_op_id.is_none() {
                    metrics.turns += 1;
                }
            }
            Event::EvalCall { command, .. } => {
                metrics.eval_calls += 1;
                if !seen_commands.insert(command.trim().to_string()) {
                    metrics.repeat_evals += 1;
                }
                if command.contains(fixture.probe) {
                    probe_hits += 1;
                }
            }
            Event::StoreCall { .. } => {
                metrics.remember_calls += 1;
                if metrics.collections == 0 {
                    metrics.proactive_remembers += 1;
                }
            }
            Event::RetrieveCall { .. } => metrics.recall_calls += 1,
            Event::Custom { name, data, .. } if name == "gc_collect" => {
                metrics.collections += 1;
                let reason = data["reason"].as_str().unwrap_or("unknown").to_string();
                *metrics.reasons.entry(reason).or_insert(0) += 1;
                metrics.dropped_total += data["dropped_count"].as_u64().unwrap_or(0);
                metrics.overlap_total += data["recall_overlap_events"].as_u64().unwrap_or(0);
                metrics.recall_hot_max = metrics
                    .recall_hot_max
                    .max(data["recall_hot"].as_u64().unwrap_or(0));
                // Eviction-marker needles (t-1360): in-window marker
                // high-water mark and suppression count, for scoring
                // marker-driven recovery vs re-derivation vs fabrication.
                // Absent on pre-marker recordings (replayed leniently).
                metrics.markers_max = metrics
                    .markers_max
                    .max(data["markers"].as_u64().unwrap_or(0));
                if data["markers_suppressed"].as_bool().unwrap_or(false) {
                    metrics.markers_suppressed += 1;
                }
                // Structural: the write-barrier fields must be present on
                // every gc_collect event (t-1351) — absent fields would
                // silently zero the behavioral signal.
                anyhow::ensure!(
                    data.get("recall_overlap_events").is_some() && data.get("recall_hot").is_some(),
                    "gc_collect event missing recall-overlap write-barrier fields: {data}"
                );
            }
            Event::AgentDone { usage, .. } => done_usage = usage.clone(),
            _ => {}
        }
    }
    metrics.needle_refetches = probe_hits.saturating_sub(fixture.probe_allowance);
    metrics.usage = done_usage
        .ok_or_else(|| anyhow!("trace has no AgentDone usage rollup (t-1334 instrument)"))?;
    Ok(metrics)
}

// --- recording format --------------------------------------------------------------
//
// One JSONL file per (fixture, arm) cell: a meta first line, then one
// {"event": ...} line per trace event of the online run. Offline, the
// events rebuild an IrReplayTrace (effect-id replay) AND serve as the
// reference the replayed run must reproduce. Recordings must be
// credential-free: the online writer asserts the key never appears, and
// `recordings_are_credential_free` re-checks whatever is committed.

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CellMeta {
    fixture: String,
    arm: String,
    model: String,
    context_budget: usize,
    /// The t-1364 guidance axis (defaults false: the legacy t-1349
    /// recordings predate it and ran unguided).
    #[serde(default)]
    guided: bool,
    /// Sample index within the cell (n=2 on the hypothesis-deciding cells).
    #[serde(default = "default_sample")]
    sample: u32,
    /// Online wall time — replays report this, not their own.
    wall_ms: u64,
    /// The online run's final answer; replay must reproduce it.
    final_content: String,
    recorded_at: String,
}

fn default_sample() -> u32 {
    1
}

/// t-1349 legacy recording path: pre-a6592f8 tool descriptions, guidance
/// off. Kept replayable, but no longer a valid comparison arm (module
/// docs).
fn legacy_cell_path(dir: &Path, fixture: &str, arm: Arm) -> PathBuf {
    dir.join(format!("{fixture}--{}.jsonl", arm.label()))
}

/// t-1364 cell path: the guidance axis and sample index are part of the
/// cell identity.
fn cell_path(dir: &Path, fixture: &str, arm: Arm, guided: bool, sample: u32) -> PathBuf {
    dir.join(format!(
        "{fixture}--{}--{}-s{sample}.jsonl",
        arm.label(),
        if guided { "guided" } else { "unguided" }
    ))
}

/// One planned t-1364 cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CellId {
    fixture: &'static str,
    arm: Arm,
    guided: bool,
    sample: u32,
}

/// The recording plan, in SPEND-PRIORITY order. Two generations:
///
/// t-1364 recorded the guidance x strategy matrix. Its guided recordings
/// were later invalidated in two waves — the t-1367 last-user hard guard
/// (ring/stack cells whose recorded gc_collect streams embodied the
/// task-eviction the guard bans) and the t-1368 budget gate (the fragment
/// those cells recorded under no longer renders at these budgets) — and
/// deleted; their rows live on in the README as the historical record.
/// The guided mark-sweep/semantic cells are deliberately NOT re-planned:
/// at these budgets guided arms now deliver no fragment, so re-recording
/// them answers no open question the t-1367 re-run does not.
///
/// t-1367 verification re-run (first, so the spend cap can never cut it):
/// the 8 deciding cells — ring/stack x guided x all three fixtures at the
/// t-1364 sample counts — re-recorded with the hard guard and the budget
/// gate live. Expected: no more 2-turn task-eviction non-answers; guided
/// ring/stack attempt the tasks like their unguided twins (the fragment
/// is suppressed at these budgets, so the guided axis now measures the
/// gate itself plus sampling variance).
///
/// Then the t-1364 unguided coverage and `none` baselines, all recorded.
fn planned_cells() -> Vec<CellId> {
    let mut cells = Vec::new();
    // t-1367 re-run: the deciding guided ring/stack cells.
    for fixture in ["early-needle", "tangent-return", "memory-discipline"] {
        for arm in [Arm::Ring, Arm::Stack] {
            cells.push(CellId {
                fixture,
                arm,
                guided: true,
                sample: 1,
            });
            // Stack — the shipped default on trial — keeps its n=2 on the
            // fixtures where t-1364 saw the deciding failures.
            if arm == Arm::Stack && fixture != "memory-discipline" {
                cells.push(CellId {
                    fixture,
                    arm,
                    guided: true,
                    sample: 2,
                });
            }
        }
    }
    // t-1364 unguided coverage, at its recorded sample counts.
    for sample in [1, 2] {
        for fixture in ["early-needle", "tangent-return"] {
            for arm in [Arm::Stack, Arm::MarkSweep] {
                cells.push(CellId {
                    fixture,
                    arm,
                    guided: false,
                    sample,
                });
            }
        }
    }
    for fixture in ["early-needle", "tangent-return", "memory-discipline"] {
        cells.push(CellId {
            fixture,
            arm: Arm::NoGc,
            guided: false,
            sample: 1,
        });
        for arm in [Arm::Ring, Arm::Stack, Arm::MarkSweep, Arm::Semantic] {
            let cell = CellId {
                fixture,
                arm,
                guided: false,
                sample: 1,
            };
            if !cells.contains(&cell) {
                cells.push(cell);
            }
        }
    }
    cells
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

// --- coherence judge (real recorded verdicts) ---------------------------------------
//
// The recorded-judge stance from gc_evals.rs, applied to BEHAVIOR: the
// judge reads the task, a rendered transcript of what the agent actually
// did (from the recorded events), and the final answer, and scores three
// booleans. Responses recorded to evals/gc/judge/behavioral.jsonl are real
// model verdicts (unlike the offline matrix's hand-written placeholders)
// and replay offline by content-hash key.

const JUDGE_RUBRIC: &str = "You are auditing the behavior of a software agent that worked \
under context-window garbage collection (parts of its conversation were evicted \
mid-session). You are shown the TASK it was given, a TRANSCRIPT of its actions \
(assistant commentary, tool commands, tool-output previews, memory tool use), and its \
FINAL ANSWER.\n\
Score three booleans:\n\
- stayed_on_task: the agent followed the task's steps and never wandered into work \
the task did not ask for.\n\
- no_redundant_work: the agent did not repeat commands or re-acquire information it \
had already been given, beyond what the task itself asks for.\n\
- grounded_final_answer: the final answer is consistent with what the transcript \
actually observed (not invented, not contradicting the tool outputs).\n\
Reply with ONLY a JSON object, no prose: \
{\"stayed_on_task\": bool, \"no_redundant_work\": bool, \"grounded_final_answer\": bool}";

/// Cap for each rendered line in the judge transcript: bounded and
/// deterministic.
const JUDGE_RENDER_CHARS: usize = 240;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct JudgeVerdict {
    stayed_on_task: bool,
    no_redundant_work: bool,
    grounded_final_answer: bool,
}

impl JudgeVerdict {
    fn display(&self) -> String {
        let score = u8::from(self.stayed_on_task)
            + u8::from(self.no_redundant_work)
            + u8::from(self.grounded_final_answer);
        format!("{score}/3")
    }
}

/// One recorded judge exchange. `cell` and `model` are provenance; lookup is
/// purely by `key`. Token counts record what the verdict cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JudgeRecord {
    key: String,
    cell: String,
    model: String,
    response: String,
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    recorded_at: String,
}

fn judge_preview(input: &str) -> String {
    let mut out: String = input.chars().take(JUDGE_RENDER_CHARS).collect();
    if input.chars().count() > JUDGE_RENDER_CHARS {
        out.push_str("…[truncated]");
    }
    out
}

/// Deterministic transcript render from recorded events: assistant turns,
/// tool commands with output previews, memory tool use. No UUIDs, op ids,
/// or timestamps leak in, so the render (and the recording key) is stable.
fn render_transcript(events: &[Event]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for event in events {
        match event {
            Event::InferResult {
                response: Some(response),
                ..
            } => {
                if !response.content.trim().is_empty() {
                    let _ = writeln!(out, "assistant: {}", judge_preview(&response.content));
                }
                for call in &response.tool_calls {
                    let _ = writeln!(
                        out,
                        "assistant calls {} {}",
                        call.name,
                        judge_preview(&call.arguments.to_string())
                    );
                }
            }
            Event::EvalResult {
                command, result, ..
            } => {
                let stdout = result["stdout"].as_str().unwrap_or_default();
                let _ = writeln!(
                    out,
                    "$ {}\n  -> {}",
                    judge_preview(command),
                    judge_preview(stdout)
                );
            }
            Event::StoreResult { sink_id, .. } => {
                let _ = writeln!(out, "remember stored: {}", judge_preview(sink_id));
            }
            Event::RetrieveResult { result_preview, .. } => {
                let _ = writeln!(out, "recall returned: {}", judge_preview(result_preview));
            }
            _ => {}
        }
    }
    out
}

fn judge_prompt(task: &str, events: &[Event], final_content: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(JUDGE_RUBRIC),
        ChatMessage::user(format!(
            "== TASK ==\n{task}\n\n== TRANSCRIPT ==\n{}\n== FINAL ANSWER ==\n{}",
            render_transcript(events),
            judge_preview(final_content),
        )),
    ]
}

/// Recording key: content hash of the judge prompt text (roles + content
/// only — never message UUIDs).
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

/// Lenient JSON extraction (the gc_evals judge rule): take the outermost
/// brace span that parses.
fn parse_judge_response(response: &str) -> Option<JudgeVerdict> {
    let start = response.find('{')?;
    let end = response.rfind('}')?;
    serde_json::from_str(&response[start..=end]).ok()
}

struct JudgeBook {
    path: PathBuf,
    recordings: HashMap<String, String>,
    online: Option<(ProviderClient, Model)>,
}

impl JudgeBook {
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
            let model = Model(
                std::env::var("AGENT_JUDGE_MODEL").unwrap_or_else(|_| DEFAULT_JUDGE_MODEL.into()),
            );
            let url = std::env::var("AGENT_JUDGE_URL")
                .unwrap_or_else(|_| "https://openrouter.ai/api/v1".into());
            let client = ProviderClient::new(ProviderConfig {
                url,
                api_key: online_api_key()?,
                model: model.clone(),
            });
            Some((client, model))
        } else {
            None
        };
        Ok(Self {
            path,
            recordings,
            online,
        })
    }

    async fn verdict(
        &mut self,
        cell: &str,
        task: &str,
        events: &[Event],
        final_content: &str,
    ) -> Result<Option<JudgeVerdict>> {
        let prompt = judge_prompt(task, events, final_content);
        let key = judge_key(&prompt);
        if let Some(response) = self.recordings.get(&key) {
            return Ok(parse_judge_response(response));
        }
        let Some((client, model)) = &self.online else {
            return Ok(None);
        };
        let response = client.chat(model, &[], &prompt).await?;
        let record = JudgeRecord {
            key: key.clone(),
            cell: cell.to_string(),
            model: model.0.clone(),
            response: response.content.clone(),
            input_tokens: response.input_tokens,
            output_tokens: response.output_tokens,
            recorded_at: Utc::now().to_rfc3339(),
        };
        self.append_record(&record)?;
        self.recordings.insert(key, response.content.clone());
        Ok(parse_judge_response(&response.content))
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

// --- table -----------------------------------------------------------------------

fn print_header() {
    println!(
        "{:<18} {:<10} {:<4} {:>1} {:>5} {:>5} {:>4} {:>4} {:>3} {:>4} {:>3} {:>4} {:<11} {:>4} {:>3} {:>3} {:>8} {:>8} {:>10} {:>6} {:>3} {:>4} {:>5}",
        "fixture",
        "arm",
        "guid",
        "s",
        "turns",
        "evals",
        "rpt",
        "refx",
        "rem",
        "prem",
        "rec",
        "coll",
        "reasons",
        "drop",
        "ovl",
        "mkr",
        "in_tok",
        "out_tok",
        "cost",
        "wall_s",
        "ok",
        "cfab",
        "judge",
    );
}

fn reasons_label(reasons: &BTreeMap<String, usize>) -> String {
    if reasons.is_empty() {
        return "-".into();
    }
    reasons
        .iter()
        .map(|(reason, count)| format!("{}:{count}", &reason[..1]))
        .collect::<Vec<_>>()
        .join("/")
}

fn print_row(fixture: &str, arm: Arm, metrics: &CellMetrics, meta: &CellMeta, judge: &str) {
    println!(
        "{:<18} {:<10} {:<4} {:>1} {:>5} {:>5} {:>4} {:>4} {:>3} {:>4} {:>3} {:>4} {:<11} {:>4} {:>3} {:>3} {:>8} {:>8} {:>10} {:>6.1} {:>3} {:>4} {:>5}",
        fixture,
        arm.label(),
        if meta.guided { "on" } else { "off" },
        meta.sample,
        metrics.turns,
        metrics.eval_calls,
        metrics.repeat_evals,
        metrics.needle_refetches,
        metrics.remember_calls,
        metrics.proactive_remembers,
        metrics.recall_calls,
        metrics.collections,
        reasons_label(&metrics.reasons),
        metrics.dropped_total,
        metrics.overlap_total,
        metrics.markers_max,
        metrics.usage.input_tokens,
        metrics.usage.output_tokens,
        metrics
            .usage
            .cost_micro_usd
            .map_or_else(|| "-".into(), agent_core::format_micro_usd),
        meta.wall_ms as f64 / 1000.0,
        if metrics.success { "yes" } else { "NO" },
        if metrics.confabulated { "YES" } else { "-" },
        judge,
    );
}

// --- online provider ----------------------------------------------------------------

fn env_model() -> String {
    std::env::var("AGENT_EVAL_PARENT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into())
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

fn online_client(model: &str) -> Result<ProviderClient> {
    let url =
        std::env::var("AGENT_EVAL_URL").unwrap_or_else(|_| "https://openrouter.ai/api/v1".into());
    Ok(ProviderClient::new(ProviderConfig {
        url,
        api_key: online_api_key()?,
        model: Model(model.into()),
    }))
}

// --- the matrix -----------------------------------------------------------------------

/// The behavioral matrix: every fixture x arm.
///
/// Offline (default): replays each cell's recorded trace through effect-id
/// replay, asserts the replay reproduces the recording (final answer,
/// metrics, and the gc_collect stream — GC re-runs deterministically during
/// replay), and prints the table. Cells without recordings are reported and
/// skipped; a wholly-absent recordings dir is a clean no-op (there are
/// deliberately no hand-written behavioral recordings — see module docs).
///
/// Online (RUN_AGENT_ONLINE_EVAL=1): records any missing cells against the
/// real provider first, then replays everything just like offline — so a
/// recording run IS a replay verification run. Judge verdicts for cells
/// missing from the judge book are recorded in the same pass.
#[tokio::test]
async fn gc_behavior_matrix() -> Result<()> {
    let online = std::env::var("RUN_AGENT_ONLINE_EVAL").is_ok_and(|value| value == "1");
    let dir = recordings_dir()?;

    if online {
        record_missing_cells(&dir).await?;
    } else if !dir.exists() {
        println!(
            "gc_behavior_matrix: no recordings at {} — offline no-op; \
             run with RUN_AGENT_ONLINE_EVAL=1 to record (see evals/gc/README.md)",
            dir.display()
        );
        return Ok(());
    }

    let mut judge = JudgeBook::load(judge_book_path()?, online)?;
    let fixtures = fixtures();

    // Section 1 — t-1349 legacy recordings: pre-a6592f8 tool descriptions,
    // guidance off. Still replayed (regression: old recordings must keep
    // replaying), printed apart because they are NOT comparable with the
    // t-1364 rows — the remember/recall/shell/infer descriptions the model
    // saw differ.
    println!("== t-1349 legacy cells (pre-guidance tool descriptions) ==");
    print_header();
    for fixture in &fixtures {
        for arm in Arm::ALL {
            let path = legacy_cell_path(&dir, fixture.name, arm);
            if !path.exists() {
                println!(
                    "{:<18} {:<10} skipped: no recording ({})",
                    fixture.name,
                    arm.label(),
                    path.display()
                );
                continue;
            }
            replay_and_print(&path, fixture, arm, false, &mut judge).await?;
        }
    }

    // Section 2 — the t-1364 guidance x strategy matrix, all cells on the
    // current tool descriptions. `none`+guided is deliberately unplanned
    // (guidance without GC is not this run's hypothesis; budget went to
    // n=2 on the deciding cells instead).
    println!();
    println!("== t-1364 guidance x strategy (current tool descriptions) ==");
    print_header();
    let planned = planned_cells();
    for fixture in &fixtures {
        for arm in Arm::ALL {
            for guided in [false, true] {
                for sample in [1, 2] {
                    let path = cell_path(&dir, fixture.name, arm, guided, sample);
                    if !path.exists() {
                        let cell = CellId {
                            fixture: fixture.name,
                            arm,
                            guided,
                            sample,
                        };
                        if planned.contains(&cell) {
                            println!(
                                "{:<18} {:<10} {:<4} {} skipped: planned cell not recorded",
                                fixture.name,
                                arm.label(),
                                if guided { "on" } else { "off" },
                                sample,
                            );
                        }
                        continue;
                    }
                    replay_and_print(&path, fixture, arm, guided, &mut judge).await?;
                }
            }
        }
    }
    Ok(())
}

/// Replay one cell, assert the firing invariant, judge it, print the row.
async fn replay_and_print(
    path: &Path,
    fixture: &Fixture,
    arm: Arm,
    guided: bool,
    judge: &mut JudgeBook,
) -> Result<()> {
    let (meta, metrics, events) = replay_cell(path, fixture, guided).await?;
    // The point of the small budget: collections must actually have fired
    // in the recorded session, or the cell measures nothing.
    if arm.collects() {
        assert!(
            metrics.collections > 0,
            "{}/{}: GC never fired — the cell measures nothing; \
             shrink the budget or fatten the fixture",
            fixture.name,
            arm.label()
        );
    } else {
        assert_eq!(
            metrics.collections, 0,
            "{}/none: control arm must not collect",
            fixture.name
        );
    }
    let cell = format!(
        "{}|{}|{}|s{}",
        fixture.name,
        arm.label(),
        if meta.guided { "guided" } else { "unguided" },
        meta.sample
    );
    let verdict = judge
        .verdict(&cell, &fixture.task, &events, &meta.final_content)
        .await?
        .map(|verdict| verdict.display());
    print_row(
        fixture.name,
        arm,
        &metrics,
        &meta,
        verdict.as_deref().unwrap_or("-"),
    );
    Ok(())
}

/// Replay one recorded cell and assert it reproduces the recording: same
/// final answer, same trace-derived metrics (including the gc_collect
/// stream) as the recorded events. Returns the recorded events for judge
/// rendering (key stability: the judge scores the RECORDING, which the
/// replay was just proven to reproduce).
async fn replay_cell(
    path: &Path,
    fixture: &Fixture,
    expect_guided: bool,
) -> Result<(CellMeta, CellMetrics, Vec<Event>)> {
    let (meta, recorded_events) = load_cell_recording(path)?;
    anyhow::ensure!(
        meta.fixture == fixture.name,
        "{}: recording is for fixture {}",
        path.display(),
        meta.fixture
    );
    anyhow::ensure!(
        meta.guided == expect_guided,
        "{}: recording's guidance setting ({}) does not match its cell ({})",
        path.display(),
        meta.guided,
        expect_guided
    );
    let arm = Arm::from_label(&meta.arm)?;
    let replay = IrReplayTrace::from_events(&recorded_events)
        .with_context(|| format!("building replay from {}", path.display()))?;
    let prompt = vec![
        ChatMessage::system(system_prompt()),
        ChatMessage::user(fixture.task.clone()),
    ];
    let workdir = materialize_fixture(fixture)?;
    let memory_dir = std::env::temp_dir().join(format!("gc-behavior-mem-{}", Uuid::new_v4()));
    fs::create_dir_all(&memory_dir)?;
    let run = run_cell(
        Arc::new(ReplayOnlyProvider),
        Some(&replay),
        CellSpec {
            model: meta.model.clone(),
            gc: arm.gc_mode(),
            context_budget: meta.context_budget,
            prompt,
            guidance: cell_guidance(meta.guided),
        },
        &workdir,
        &memory_dir,
    )
    .await
    .with_context(|| format!("replaying {}", path.display()))?;
    let _ = fs::remove_dir_all(&workdir);
    let _ = fs::remove_dir_all(&memory_dir);

    assert_eq!(
        run.content,
        meta.final_content,
        "{}: replay must reproduce the recorded final answer",
        path.display()
    );
    let replayed = metrics_from_events(&run.events, &run.content, fixture)?;
    let recorded = metrics_from_events(&recorded_events, &meta.final_content, fixture)?;
    // Bound effect errors (t-1222) do not replay byte-identically: the
    // replayed BOUND VALUE carries the "AgentIR replaying recorded ...
    // failure" wrapper, so window content differs by a few tokens and
    // content-sensitive GC (semantic) can drop marginally differently
    // (observed: memory-discipline/semantic, two duplicate-slug
    // StoreErrors, 190 vs 188 dropped). For those cells the gc-derived
    // fields are reported from the recording and compared leniently;
    // everything else — answers, turns, tool counts, usage — must still
    // reproduce exactly. Runtime gap noted in evals/gc/README.md.
    let bound_errors = recorded_events.iter().any(|event| {
        matches!(
            event,
            Event::StoreError { .. }
                | Event::RetrieveError { .. }
                | Event::EvalError { .. }
                | Event::InferError { .. }
        )
    });
    // Pre-marker recordings (before t-1360's eviction markers): GC re-runs
    // live during replay, and the marker-era collector leaves marker lines
    // in the window, so the gc stream (dropped counts, tokens, marker
    // fields) cannot reproduce a recording made without them. Provider
    // effects still replay by effect id, so answers, turns, tool counts,
    // and usage must reproduce exactly; the gc-derived fields are reported
    // from the recording and compared leniently until the cell is
    // re-recorded (see evals/gc/README.md — fresh recordings are batched
    // with the next online eval round).
    let pre_marker_recording = recorded_events.iter().any(|event| {
        matches!(event, Event::Custom { name, data, .. }
            if name == "gc_collect" && data.get("markers").is_none())
    });
    let mut replayed_cmp = replayed.clone();
    if bound_errors || pre_marker_recording {
        replayed_cmp.collections = recorded.collections;
        replayed_cmp.reasons = recorded.reasons.clone();
        replayed_cmp.dropped_total = recorded.dropped_total;
        replayed_cmp.overlap_total = recorded.overlap_total;
        replayed_cmp.recall_hot_max = recorded.recall_hot_max;
        replayed_cmp.markers_max = recorded.markers_max;
        replayed_cmp.markers_suppressed = recorded.markers_suppressed;
        assert!(
            replayed.collections > 0 || recorded.collections == 0,
            "{}: replay lost the gc_collect stream entirely",
            path.display()
        );
    }
    assert_eq!(
        replayed_cmp,
        recorded,
        "{}: replayed metrics (incl. the gc_collect stream) must reproduce the recording",
        path.display()
    );
    Ok((meta, recorded, recorded_events))
}

/// The recording pass's spend ceiling (USD), enforced across the run from
/// each cell's AgentDone rollup; override with AGENT_EVAL_SPEND_CAP_USD.
/// `planned_cells` is priority-ordered so hitting the cap drops coverage
/// cells, never the hypothesis-deciding ones.
const DEFAULT_SPEND_CAP_USD: f64 = 2.0;

/// Record every planned cell that has no recording yet, in plan order,
/// under the spend cap. Requires a key; spends real money (small
/// fixtures, tiny windows, a cheap model — see README for the measured
/// totals). Legacy t-1349 recordings are never re-recorded.
async fn record_missing_cells(dir: &Path) -> Result<()> {
    let model = env_model();
    let api_key = online_api_key()?;
    let client: Arc<dyn ChatProvider> = Arc::new(online_client(&model)?);
    let cap_usd: f64 = std::env::var("AGENT_EVAL_SPEND_CAP_USD")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SPEND_CAP_USD);
    let mut spent_micro: u64 = 0;
    let fixtures = fixtures();

    for cell in planned_cells() {
        let path = cell_path(dir, cell.fixture, cell.arm, cell.guided, cell.sample);
        if path.exists() {
            continue;
        }
        let label = format!(
            "{} / {} / {} / s{}",
            cell.fixture,
            cell.arm.label(),
            if cell.guided { "guided" } else { "unguided" },
            cell.sample
        );
        if spent_micro as f64 / 1e6 >= cap_usd {
            println!("SKIPPING {label}: spend cap ${cap_usd} reached");
            continue;
        }
        let fixture = fixtures
            .iter()
            .find(|fixture| fixture.name == cell.fixture)
            .ok_or_else(|| anyhow!("planned cell names unknown fixture {}", cell.fixture))?;
        println!("recording {label} ...");
        let prompt = vec![
            ChatMessage::system(system_prompt()),
            ChatMessage::user(fixture.task.clone()),
        ];
        let workdir = materialize_fixture(fixture)?;
        let memory_dir = std::env::temp_dir().join(format!("gc-behavior-mem-{}", Uuid::new_v4()));
        fs::create_dir_all(&memory_dir)?;
        let run = run_cell(
            client.clone(),
            None,
            CellSpec {
                model: model.clone(),
                gc: cell.arm.gc_mode(),
                context_budget: fixture.context_budget,
                prompt,
                guidance: cell_guidance(cell.guided),
            },
            &workdir,
            &memory_dir,
        )
        .await
        .with_context(|| format!("online cell {label}"))?;
        let _ = fs::remove_dir_all(&workdir);
        let _ = fs::remove_dir_all(&memory_dir);

        let cell_micro = run
            .events
            .iter()
            .rev()
            .find_map(|event| match event {
                Event::AgentDone {
                    usage: Some(usage), ..
                } => usage.cost_micro_usd,
                _ => None,
            })
            .unwrap_or(0);
        spent_micro += cell_micro;
        println!(
            "  recorded {label}: {} (cumulative {})",
            agent_core::format_micro_usd(cell_micro),
            agent_core::format_micro_usd(spent_micro),
        );

        let meta = CellMeta {
            fixture: fixture.name.into(),
            arm: cell.arm.label().into(),
            model: model.clone(),
            context_budget: fixture.context_budget,
            guided: cell.guided,
            sample: cell.sample,
            wall_ms: run.wall_ms,
            final_content: run.content.clone(),
            recorded_at: Utc::now().to_rfc3339(),
        };
        write_cell_recording(&path, &meta, &run.events)?;
        // Credential hygiene: the recording must not embed the key
        // (shell children get an allowlist env of PATH only, but the
        // check is unconditional).
        let written = fs::read_to_string(&path)?;
        anyhow::ensure!(
            !written.contains(api_key.as_str()),
            "{}: recording embeds the API key — do not commit",
            path.display()
        );
    }
    Ok(())
}

// --- plumbing tests (always on, offline, scripted providers) ------------------------
//
// PLUMBING-ONLY: these validate the harness machinery with scripted model
// turns. They say nothing about real model behavior — that is what the
// recorded matrix above exists for.

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

/// A tiny scripted session under a tiny budget: GC fires, the gc_collect
/// events carry the write-barrier fields, repeats and re-fetches are
/// counted, and remember/recall land as Store/Retrieve calls.
#[tokio::test]
async fn plumbing_gc_fires_and_metrics_count() -> Result<()> {
    let fixture = Fixture {
        name: "plumbing",
        task: "plumbing".into(),
        context_budget: 400,
        needles: vec!["391"],
        ordered_needles: vec![],
        claim_marker: "RESULT",
        probe: "fat.txt",
        probe_allowance: 1,
        files: vec![("fat.txt", filler(&MANUAL_WORDS, 9, 6000))],
    };
    let provider = Arc::new(ScriptedProvider::new(vec![
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-1",
            "shell",
            serde_json::json!({ "command": "cat fat.txt" }),
        )]),
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-2",
            "shell",
            serde_json::json!({ "command": "cat fat.txt" }),
        )]),
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-3",
            "remember",
            serde_json::json!({ "name": "answer", "content": "391" }),
        )]),
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-4",
            "recall",
            serde_json::json!({ "query": "answer" }),
        )]),
        ScriptedProvider::text("391"),
    ]));
    let workdir = materialize_fixture(&fixture)?;
    let memory_dir = std::env::temp_dir().join(format!("gc-behavior-mem-{}", Uuid::new_v4()));
    fs::create_dir_all(&memory_dir)?;
    let run = run_cell(
        provider,
        None,
        CellSpec {
            model: DEFAULT_MODEL.into(),
            gc: Arm::Ring.gc_mode(),
            context_budget: fixture.context_budget,
            prompt: vec![
                ChatMessage::system("plumbing"),
                ChatMessage::user("what is 17 * 23?"),
            ],
            // Guided, so plumbing also proves run_cell works under the
            // shipped fragment (the t-1364 guided arms' configuration).
            guidance: cell_guidance(true),
        },
        &workdir,
        &memory_dir,
    )
    .await?;
    let _ = fs::remove_dir_all(&workdir);
    let _ = fs::remove_dir_all(&memory_dir);

    let metrics = metrics_from_events(&run.events, &run.content, &fixture)?;
    assert!(metrics.success);
    assert_eq!(metrics.turns, 5);
    assert_eq!(metrics.eval_calls, 2);
    assert_eq!(metrics.repeat_evals, 1, "identical command counted once");
    assert_eq!(
        metrics.needle_refetches, 1,
        "second probe hit is a re-fetch"
    );
    assert_eq!(metrics.remember_calls, 1);
    assert_eq!(
        metrics.proactive_remembers, 0,
        "the remember fired after collections had already started"
    );
    assert_eq!(metrics.recall_calls, 1);
    assert!(
        !metrics.confabulated,
        "correct answer is not a confabulation"
    );
    assert!(
        metrics.collections > 0,
        "two 6KB tool results under a 400-token budget must collect"
    );
    assert!(
        metrics
            .reasons
            .keys()
            .all(|reason| ["scheduled", "backstop", "overflow"].contains(&reason.as_str())),
        "unexpected gc_collect reason markers: {:?}",
        metrics.reasons
    );
    assert!(metrics.usage.cost_micro_usd.is_some());
    Ok(())
}

/// The none arm never collects and the metrics say so.
#[tokio::test]
async fn plumbing_none_arm_never_collects() -> Result<()> {
    let fixture = Fixture {
        name: "plumbing-none",
        task: "plumbing".into(),
        context_budget: 400,
        needles: vec![],
        ordered_needles: vec![],
        claim_marker: "RESULT",
        probe: "fat.txt",
        probe_allowance: 1,
        files: vec![("fat.txt", filler(&MANUAL_WORDS, 9, 6000))],
    };
    let provider = Arc::new(ScriptedProvider::new(vec![
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-1",
            "shell",
            serde_json::json!({ "command": "cat fat.txt" }),
        )]),
        ScriptedProvider::text("done"),
    ]));
    let workdir = materialize_fixture(&fixture)?;
    let memory_dir = std::env::temp_dir().join(format!("gc-behavior-mem-{}", Uuid::new_v4()));
    fs::create_dir_all(&memory_dir)?;
    let run = run_cell(
        provider,
        None,
        CellSpec {
            model: DEFAULT_MODEL.into(),
            gc: Arm::NoGc.gc_mode(),
            context_budget: fixture.context_budget,
            prompt: vec![ChatMessage::system("plumbing"), ChatMessage::user("go")],
            guidance: cell_guidance(false),
        },
        &workdir,
        &memory_dir,
    )
    .await?;
    let _ = fs::remove_dir_all(&workdir);
    let _ = fs::remove_dir_all(&memory_dir);
    let metrics = metrics_from_events(&run.events, &run.content, &fixture)?;
    assert_eq!(metrics.collections, 0);
    assert_eq!(metrics.dropped_total, 0);
    Ok(())
}

/// Recording round-trip: what the online writer persists, the offline
/// loader restores faithfully enough to score.
#[tokio::test]
async fn plumbing_recording_roundtrip() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("gc-behavior-rec-{}", Uuid::new_v4()));
    let path = cell_path(&dir, "roundtrip", Arm::Stack, true, 2);
    let meta = CellMeta {
        fixture: "roundtrip".into(),
        arm: Arm::Stack.label().into(),
        model: DEFAULT_MODEL.into(),
        context_budget: 2000,
        guided: true,
        sample: 2,
        wall_ms: 1234,
        final_content: "ACCESS X TOTAL 21".into(),
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
    assert_eq!(loaded_meta.context_budget, meta.context_budget);
    assert!(loaded_meta.guided);
    assert_eq!(loaded_meta.sample, 2);
    assert_eq!(loaded_events, events);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Legacy t-1349 meta lines (no guided/sample fields) must keep loading:
/// guidance defaults off, sample defaults 1 — exactly the configuration
/// those cells ran under.
#[test]
fn plumbing_legacy_meta_defaults() -> Result<()> {
    let legacy = r#"{"fixture":"early-needle","arm":"stack","model":"m","context_budget":2000,"wall_ms":1,"final_content":"x","recorded_at":"t"}"#;
    let meta: CellMeta = serde_json::from_str(legacy)?;
    assert!(!meta.guided);
    assert_eq!(meta.sample, 1);
    Ok(())
}

/// The confabulation flag (t-1364): asserted-but-wrong flags, silence and
/// correct answers do not, and the order fixture keys on category order.
#[test]
fn confabulation_flag_detects_fabricated_claims() {
    let fixtures = fixtures();
    let early = fixtures.iter().find(|f| f.name == "early-needle").unwrap();
    // t-1349's actual stack hallucination shape:
    assert!(confabulated(early, "ACCESS CDBH92 TOTAL 21"));
    assert!(!confabulated(early, "ACCESS MX-7749-KESTREL TOTAL 21"));
    // Wrong arithmetic with the right code is a slip, not fabrication:
    assert!(!confabulated(early, "ACCESS MX-7749-KESTREL TOTAL 19"));
    // A thrash cell that never answers is a non-answer, not a confabulation:
    assert!(!confabulated(early, ""));
    assert!(!confabulated(early, "I could not finish the steps."));

    let tangent = fixtures
        .iter()
        .find(|f| f.name == "tangent-return")
        .unwrap();
    assert!(confabulated(tangent, "CATEGORIES: checksum,timeout,quota"));
    assert!(!confabulated(tangent, "CATEGORIES: timeout,checksum,quota"));

    let memory = fixtures
        .iter()
        .find(|f| f.name == "memory-discipline")
        .unwrap();
    assert!(confabulated(memory, "DEPLOY TOKEN-1234-FAKE WARNS 6"));
    assert!(!confabulated(memory, "DEPLOY TOKEN-9QX-RAVEN-7734 WARNS 6"));
}

/// Judge book round-trip: a record written the way the online path writes
/// it is served back by a fresh offline book; misses return None offline.
#[tokio::test]
async fn plumbing_judge_book_replays_offline() -> Result<()> {
    let events = vec![Event::AgentDone {
        run_id: "r".into(),
        usage: None,
        timestamp: Utc::now(),
    }];
    let prompt = judge_prompt("task", &events, "final");
    let key = judge_key(&prompt);
    let dir = std::env::temp_dir().join(format!("gc-behavior-judge-{}", Uuid::new_v4()));
    fs::create_dir_all(&dir)?;
    let path = dir.join("behavioral.jsonl");
    let record = JudgeRecord {
        key,
        cell: "plumbing|stack".into(),
        model: "test-judge".into(),
        response:
            r#"{"stayed_on_task": true, "no_redundant_work": false, "grounded_final_answer": true}"#
                .into(),
        input_tokens: 100,
        output_tokens: 20,
        recorded_at: Utc::now().to_rfc3339(),
    };
    fs::write(&path, format!("{}\n", serde_json::to_string(&record)?))?;

    let mut book = JudgeBook::load(path, false)?;
    let verdict = book
        .verdict("plumbing|stack", "task", &events, "final")
        .await?
        .expect("recorded response replays offline");
    assert_eq!(verdict.display(), "2/3");
    assert!(!verdict.no_redundant_work);

    let miss = book
        .verdict("plumbing|other", "different task", &events, "final")
        .await?;
    assert_eq!(miss, None, "offline miss is None, never a provider call");
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Judge keys are stable across runs and independent of message UUIDs.
#[test]
fn plumbing_judge_key_is_deterministic() {
    let events = vec![Event::AgentDone {
        run_id: "r".into(),
        usage: None,
        timestamp: Utc::now(),
    }];
    let key_a = judge_key(&judge_prompt("task", &events, "final"));
    let key_b = judge_key(&judge_prompt("task", &events, "final"));
    assert_eq!(key_a, key_b);
    let key_c = judge_key(&judge_prompt("task", &events, "different final"));
    assert_ne!(key_a, key_c);
}

/// Committed recordings (cells and judge book) must be credential-free.
#[test]
fn recordings_are_credential_free() -> Result<()> {
    let live_key = std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("AGENT_API_KEY"))
        .ok();
    let mut paths: Vec<PathBuf> = Vec::new();
    let dir = recordings_dir()?;
    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                paths.push(path);
            }
        }
    }
    let judge_path = judge_book_path()?;
    if judge_path.exists() {
        paths.push(judge_path);
    }
    for path in paths {
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

/// Fixture honesty: needles behave (boundary-aware numerics, order check).
#[test]
fn needle_and_order_matching() {
    assert!(needle_present("ACCESS MX-7749-KESTREL TOTAL 21", "21"));
    assert!(!needle_present("TOTAL 213", "21"));
    assert!(needle_present("total was 21.", "21"));
    assert!(!needle_present("21.5", "21"));
    assert!(ordered_needles_present(
        "CATEGORIES: timeout,checksum,quota",
        &["timeout", "checksum", "quota"]
    ));
    assert!(ordered_needles_present(
        "the answer.\nCATEGORIES: Timeout, Checksum, Quota",
        &["timeout", "checksum", "quota"]
    ));
    assert!(!ordered_needles_present(
        "CATEGORIES: checksum,timeout,quota",
        &["timeout", "checksum", "quota"]
    ));
    assert!(!ordered_needles_present(
        "timeout checksum quota but no marker",
        &["timeout", "checksum", "quota"]
    ));
}

/// The fixture files really contain what the needles assume: OK/WARN/ERROR
/// counts add up, and the needle values sit in exactly one source file.
#[test]
fn fixture_arithmetic_is_honest() {
    for fixture in fixtures() {
        match fixture.name {
            "early-needle" => {
                let total: usize = fixture
                    .files
                    .iter()
                    .filter(|(name, _)| name.starts_with("logs/"))
                    .map(|(_, content)| content.matches("STATUS: OK").count())
                    .sum();
                assert_eq!(total, 21);
                let code = fixture
                    .files
                    .iter()
                    .find(|(name, _)| *name == "config/access-code.txt")
                    .expect("access code file");
                assert!(code.1.contains(ACCESS_CODE));
            }
            "tangent-return" => {
                let log = fixture
                    .files
                    .iter()
                    .find(|(name, _)| *name == "app.log")
                    .expect("app.log");
                assert_eq!(log.1.matches("ERROR timeout").count(), 7);
                assert_eq!(log.1.matches("ERROR checksum").count(), 4);
                assert_eq!(log.1.matches("ERROR quota").count(), 2);
            }
            "memory-discipline" => {
                let total: usize = fixture
                    .files
                    .iter()
                    .filter(|(name, _)| name.starts_with("build/step-"))
                    .map(|(_, content)| content.matches("WARN:").count())
                    .sum();
                assert_eq!(total, 6);
                let token = fixture
                    .files
                    .iter()
                    .find(|(name, _)| *name == "secrets/deploy-token.txt")
                    .expect("token file");
                assert!(token.1.contains(DEPLOY_TOKEN));
            }
            other => panic!("unknown fixture {other}"),
        }
        // Sessions must actually overflow the budget: total fixture bytes
        // alone (before narration and tool-call overhead) must exceed the
        // budget in token terms, or collections cannot fire.
        let bytes: usize = fixture.files.iter().map(|(_, content)| content.len()).sum();
        assert!(
            bytes / 4 > fixture.context_budget,
            "{}: fixture too small ({} bytes) for budget {} — GC would never fire",
            fixture.name,
            bytes,
            fixture.context_budget
        );
    }
}
