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
//!
//! Marker axis (t-1369): t-1360 gave every strategy eviction markers — a
//! deterministic `[gc: ...]` line naming what was dropped (tool-call id,
//! recall query, turn ordinal) and the recovery affordance ("re-run the
//! call" / "recall the memory" / "ask the user again" — always "do not
//! guess"), and stack's `[frame ...]` annotations an explicit
//! "evicted; re-run to recover" clause. The deciding question: do the
//! early-needle fabricators (t-1349 finding 3, reproduced twice) flip to
//! honest recovery now that eviction is named instead of silent? At these
//! budgets the t-1368 gate suppresses the guidance fragment entirely, so
//! the markers themselves are the intervention — a guided cell's prompt
//! differs from its unguided twin by nothing, which the early-needle
//! stack guided/unguided pairs measure directly (any delta is sampling
//! variance, not text effect). Marker-reaction metrics, all from traces:
//! `mkref` (assistant texts quoting marker syntax — the literal `[gc` /
//! `[frame` strings, which only an in-window marker can supply; the
//! `remember` tool description says "evicted", so prose-level mentions
//! are deliberately NOT counted), `rcov` (recovery action: re-ran the
//! probe command beyond the task's allowance, or recall beyond the
//! fixture's scripted count), and `admt` (the final answer admits the
//! value is unavailable instead of asserting one — admission-phrase
//! check on failed cells, a lower bound).
//!
//! Ledger axis (t-1373): the restart loop — the dominant failure across
//! all five recording generations (t-1349 finding 2 through t-1371's
//! curation refutation) — got its designed fix: the progress ledger, a
//! deterministic in-window `[gc-ledger]` digest of the session's
//! completed tool calls, rebuilt by every collection (docs/GC.md "The
//! progress ledger"). Offline, this harness validates it against the
//! EXISTING five-generation corpus: GC re-runs live during replay, so
//! the ledger builder is driven by the real recorded histories, and the
//! replayed gc stream's ledger fields are sanity-asserted per cell
//! (present-or-suppressed on evicting GC arms, entry counts within the
//! cap, absent on controls). Recordings predate the ledger, so their
//! gc-derived fields compare leniently (absent `ledger_present` field —
//! the pre-marker pattern); everything effect-replayed reproduces
//! exactly. New restart-loop needles for the FUTURE recording round:
//! `ldg` (in-window itemized-entry high-water) and `rptl` (repeats of a
//! command whose earlier call id the then-current ledger itemized — a
//! re-run issued AGAINST the model's own progress record, the loop
//! signature the ledger exists to break).
//!
//! Curation axis (t-1371): every round above ran the STARVATION regime
//! (1.6-2k-token budgets, GC forced). The pre-registered hypothesis
//! (evals/gc/README.md "GC as curation — PRE-REGISTRATION", committed
//! before any recording) is that well-tuned GC IMPROVES accuracy in the
//! CURATION regime: generous budgets (8k tokens — nothing forced, the
//! t-1368 gate delivers the MINIMAL guidance variant), threshold-triggered
//! collections, and sessions long enough to curate. Two new fixture
//! classes with OPPOSITE predictions: `distractor-update` (an early value
//! superseded later + an abandoned approach + an irrelevant tangent;
//! prediction: semantic beats the control by evicting the stale/dead
//! content, the control risks context rot) and `clean-long` (same length
//! and structure, everything relevant and consistent, broad recall needed;
//! prediction: GC arms MATCH the control and never beat it — a GC win here
//! is a fixture artifact that discounts the class-1 win). New metric:
//! `rot` — the final answer asserts the claim with the STALE needle while
//! the updated one is absent (context-rot failure made programmatic; the
//! success needles score the updated value, the rot needles the stale
//! one, so every cell is scoreable both ways).

use agent_core::gc::SemanticGc;
use agent_core::{
    agent_loop_ir_with_options, run_ir_sequential_with_store_and_replay, ChatMessage, ChatProvider,
    Embedder, EnvPolicy, EvalConfig, Event, GcMode, GcTiming, GenerationalGc, GenerationalReport,
    InMemoryStore, IrReplayTrace, MarkSweepGc, MemorySource, Model, PassiveHydrationConfig,
    Pricing, PricingTable, ProviderClient, ProviderConfig, ReplayOnlyProvider, Response, RingGc,
    RunUsage, RuntimeGuidance, SeqConfig, SourceRegistry, StackFrameGc, ToolCall, TraceLogger,
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
    Generational,
}

impl Arm {
    const ALL: [Arm; 6] = [
        Arm::NoGc,
        Arm::Ring,
        Arm::MarkSweep,
        Arm::Stack,
        Arm::Semantic,
        Arm::Generational,
    ];

    fn label(&self) -> &'static str {
        match self {
            Self::NoGc => "none",
            Self::Ring => "ring",
            Self::MarkSweep => "mark-sweep",
            Self::Stack => "stack",
            Self::Semantic => "semantic",
            Self::Generational => "generational",
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
                hot_keep: true,
                preserve_prefix: true,
            }),
            Self::MarkSweep => GcMode::MarkSweep(MarkSweepGc {
                hot_keep: true,
                preserve_prefix: true,
            }),
            Self::Stack => GcMode::Stack(StackFrameGc {
                hot_keep: true,
                preserve_prefix: true,
            }),
            Self::Semantic => GcMode::Semantic(SemanticGc {
                preserve_prefix: true,
                embedder: Some(Arc::new(BagOfTokensEmbedder)),
                ..Default::default()
            }),
            // The t-1167 synthesis strategy, with the same deterministic
            // embedder stance as semantic (identical vectors in record and
            // replay; without it the warm tier would be citation-only).
            Self::Generational => GcMode::Generational(GenerationalGc {
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
    /// Context-rot needles (t-1371): STALE values a distractor fixture
    /// plants and later supersedes. The `rot` flag fires when the final
    /// answer asserts the claim marker with a rot needle while the updated
    /// value (`needles[0]`) is absent — the context-rot failure made
    /// programmatic. Empty on fixtures with nothing stale to quote.
    rot_needles: Vec<&'static str>,
    /// Substring identifying shell commands that touch the needle's source
    /// (e.g. the access-code file). Occurrences beyond `probe_allowance`
    /// count as re-fetches — the model re-acquiring something it had been
    /// given, the coherence-cost signal.
    probe: &'static str,
    /// How many probe-matching commands the task itself asks for.
    probe_allowance: usize,
    /// How many `recall` calls the task script itself asks for; recalls
    /// beyond this count as recovery actions (t-1369 `rcov`).
    scripted_recalls: usize,
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

const CATALOG_WORDS: [&str; 12] = [
    "catalog",
    "listing",
    "wholesale",
    "invoice",
    "quotation",
    "discount",
    "surcharge",
    "freight",
    "tariff",
    "bundle",
    "voucher",
    "ledger",
];

const DEPOT_WORDS: [&str; 12] = [
    "depot",
    "pallet",
    "dockside",
    "forklift",
    "manifest",
    "carton",
    "staging",
    "inbound",
    "outbound",
    "bay",
    "consignment",
    "waybill",
];

const ACCESS_CODE: &str = "MX-7749-KESTREL";
const DEPLOY_TOKEN: &str = "TOKEN-9QX-RAVEN-7734";
/// t-1371 distractor-update values: the stale price and the update.
const STALE_UNIT_PRICE: &str = "42";
const CURRENT_UNIT_PRICE: &str = "57";

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

/// Shipment day-log `index` with exactly `ok_lines` lines reading exactly
/// `SHIPPED: OK` amid `bulk_lines` of deterministic filler (the batch_log
/// whole-line lesson: one right answer under every plausible reading).
fn shipment_log(index: usize, ok_lines: usize, bulk_lines: usize) -> String {
    let mut out = String::new();
    for line in 0..bulk_lines {
        out.push_str(&format!(
            "day-{index} {}\n",
            filler(&DEPOT_WORDS, index * 100 + line * 7, 64)
        ));
        if line < ok_lines {
            out.push_str("SHIPPED: OK\n");
        }
    }
    out
}

/// A bulky prose file with one load-bearing `value_line` embedded mid-file
/// (t-1371): filler above and below, so the value sits inside ordinary
/// content rather than alone at a file boundary.
fn value_file(words: &[&str], seed: usize, lines: usize, prefix: &str, value_line: &str) -> String {
    format!(
        "{}\n{value_line}\n{}\n",
        filler_lines(words, seed, lines / 2, prefix),
        filler_lines(words, seed + 1000, lines - lines / 2, prefix),
    )
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
            rot_needles: vec![],
            claim_marker: "ACCESS",
            probe: "access-code",
            probe_allowance: 1,
            scripted_recalls: 0,
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
            rot_needles: vec![],
            claim_marker: "CATEGORIES",
            probe: "app.log",
            probe_allowance: 2,
            scripted_recalls: 0,
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
            rot_needles: vec![],
            claim_marker: "DEPLOY",
            probe: "deploy-token",
            probe_allowance: 1,
            scripted_recalls: 1,
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

/// The t-1371 curation-regime fixtures (pre-registered in
/// evals/gc/README.md BEFORE recording). Budget 8000 tokens — generous:
/// the largest single tool result is well under budget, so nothing is
/// forced; the ~40 KB sessions overflow it ~2x, so threshold collections
/// fire mid-session with real material to curate. Both fixtures share
/// length and structure (8 cat steps + a final line); they differ only in
/// whether the content contains distractors — that is the experiment.
fn curation_fixtures() -> Vec<Fixture> {
    vec![
        // Class 1 — distractor-update (hypothesis home turf): a stale
        // value superseded later, an explicitly abandoned approach, and an
        // irrelevant tangent. Scoreable both ways: success needles score
        // the updated price, rot needles the stale one.
        Fixture {
            name: "distractor-update",
            task: "Complete these steps strictly in order, one shell command per step, \
                   with at most one short sentence of commentary each:\n\
                   1. Run `cat briefing/notes.txt` (the ops briefing; no action needed).\n\
                   2. Run `cat catalog/pricing-v1.txt` — the price list.\n\
                   3. Run `cat plans/approach-a.txt` — a draft quoting procedure.\n\
                   4. Run `cat plans/approach-a-review.txt` — the review of that draft.\n\
                   5. Run `cat archive/clippings-1.txt` and summarize it in one sentence.\n\
                   6. Run `cat archive/clippings-2.txt` and summarize it in one sentence.\n\
                   7. Run `cat catalog/pricing-v2.txt` — the price list again.\n\
                   8. Run `cat orders/current.txt` — the confirmed order count.\n\
                   9. Finish with one line exactly of the form: \
                   `QUOTE UNIT <price> TOTAL <total>` — the current unit price in \
                   whole dollars (digits only, no symbols or decimals) and that \
                   price times the number of units ordered."
                .into(),
            context_budget: 8000,
            needles: vec![CURRENT_UNIT_PRICE, "570"],
            ordered_needles: vec![],
            rot_needles: vec![STALE_UNIT_PRICE, "420"],
            claim_marker: "QUOTE",
            probe: "pricing",
            probe_allowance: 2,
            scripted_recalls: 0,
            files: vec![
                (
                    "briefing/notes.txt",
                    filler_lines(&MANUAL_WORDS, 11, 120, "briefing: "),
                ),
                (
                    "catalog/pricing-v1.txt",
                    format!(
                        "PRICE LIST v1\n{}",
                        value_file(
                            &CATALOG_WORDS,
                            21,
                            56,
                            "item: ",
                            &format!("unit price: ${STALE_UNIT_PRICE}"),
                        )
                    ),
                ),
                (
                    "plans/approach-a.txt",
                    format!(
                        "DRAFT quoting procedure (approach A)\n{}",
                        value_file(
                            &BUILD_WORDS,
                            31,
                            118,
                            "draft: ",
                            "step: multiply the legacy base rate $38 by the seasonal \
                             factor 1.2, then add the handling surcharge",
                        )
                    ),
                ),
                (
                    "plans/approach-a-review.txt",
                    "REVIEW: Approach A is REJECTED. The draft procedure and every \
                     number in it (base rate, seasonal factor, surcharge) are \
                     obsolete — do not use them. Produce the quote from the current \
                     price list instead.\n"
                        .into(),
                ),
                (
                    "archive/clippings-1.txt",
                    filler_lines(&POEM_WORDS, 7, 110, ""),
                ),
                (
                    "archive/clippings-2.txt",
                    filler_lines(&POEM_WORDS, 401, 110, ""),
                ),
                (
                    "catalog/pricing-v2.txt",
                    format!(
                        "PRICE LIST v2 (CURRENT — supersedes v1; v1 prices are obsolete)\n{}",
                        value_file(
                            &CATALOG_WORDS,
                            61,
                            56,
                            "item: ",
                            &format!("unit price: ${CURRENT_UNIT_PRICE}"),
                        )
                    ),
                ),
                (
                    "orders/current.txt",
                    "confirmed orders this cycle: 10 units\n".into(),
                ),
            ],
        },
        // Class 2 — clean-long (refutation control): same length and
        // structure, every byte relevant and consistent, and the final
        // answer needs broad recall (early + middle + late values). No rot
        // needles: there is nothing stale to quote. Predicted: GC arms
        // match the control, never beat it — a GC win here is a fixture
        // artifact that discounts the class-1 result.
        Fixture {
            name: "clean-long",
            task: "Complete these steps strictly in order, one shell command per step, \
                   with at most one short sentence of commentary each:\n\
                   1. Run `cat briefing/overview.txt` (the depot background; no action \
                   needed).\n\
                   2. Run `cat depot/region.txt` — it states the depot's region code.\n\
                   3. Run `cat procedures/receiving.txt` (the receiving procedure).\n\
                   4. Run `cat procedures/receiving-checklist.txt`.\n\
                   5-7. For each of shipments/day-1.log through shipments/day-3.log in \
                   order, run `cat <file>` and count the lines that read exactly \
                   `SHIPPED: OK`.\n\
                   8. Run `cat depot/audit.txt` — it states the audit id.\n\
                   9. Finish with one line exactly of the form: \
                   `REGION <code> SHIPPED <total> AUDIT <id>` — the region code from \
                   step 2, the sum of the three SHIPPED-line counts, and the audit id \
                   from step 8."
                .into(),
            context_budget: 8000,
            needles: vec!["NORTH-7", "8", "AUD-4413"],
            ordered_needles: vec![],
            rot_needles: vec![],
            claim_marker: "REGION",
            probe: "region",
            probe_allowance: 1,
            scripted_recalls: 0,
            files: vec![
                (
                    "briefing/overview.txt",
                    filler_lines(&MANUAL_WORDS, 17, 120, "overview: "),
                ),
                (
                    "depot/region.txt",
                    value_file(&DEPOT_WORDS, 23, 56, "site: ", "region code: NORTH-7"),
                ),
                (
                    "procedures/receiving.txt",
                    filler_lines(&BUILD_WORDS, 37, 118, "procedure: "),
                ),
                (
                    "procedures/receiving-checklist.txt",
                    "CHECKLIST: the deliverable is the day-log shipment count — for \
                     each shipments/day-N.log, count the lines that read exactly \
                     `SHIPPED: OK`, and report the region code and audit id with the \
                     total.\n"
                        .into(),
                ),
                ("shipments/day-1.log", shipment_log(1, 3, 110)),
                ("shipments/day-2.log", shipment_log(2, 1, 110)),
                ("shipments/day-3.log", shipment_log(3, 4, 56)),
                ("depot/audit.txt", "audit id: AUD-4413\n".into()),
            ],
        },
    ]
}

/// Every fixture the harness knows: the starvation-regime set plus the
/// t-1371 curation-regime set (recording lookup + honesty checks).
fn all_fixtures() -> Vec<Fixture> {
    let mut all = fixtures();
    all.extend(curation_fixtures());
    all
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

/// The context-rot flag (t-1371): the final answer asserts the fixture's
/// claim marker, the updated value (`needles[0]`) is absent, and a STALE
/// needle is present — the model quoted the superseded value the session
/// later corrected. A prose mention of the old value alongside a correct
/// claim does NOT flag (the updated value being present clears it), and a
/// cell that never answers cannot rot. Always false on fixtures with no
/// rot needles.
fn context_rot(fixture: &Fixture, content: &str) -> bool {
    if fixture.rot_needles.is_empty()
        || !content
            .to_lowercase()
            .contains(&fixture.claim_marker.to_lowercase())
        || needle_present(content, fixture.needles[0])
    {
        return false;
    }
    fixture
        .rot_needles
        .iter()
        .any(|needle| needle_present(content, needle))
}

/// Marker-reaction needle (t-1369): does an assistant text quote eviction-
/// marker syntax? Keyed on the literal `[gc` / `[frame` strings, which only
/// an in-window marker (t-1360) can supply. Deliberately NOT keyed on
/// "evicted"/"re-run" prose: the shipped `remember`/`recall` tool
/// descriptions use those words in every cell's offer, guided or not, so a
/// prose match cannot be attributed to a marker. A lower bound — a model
/// reacting to a marker without quoting it is not counted.
fn mentions_marker(content: &str) -> bool {
    content.contains("[gc") || content.contains("[frame")
}

/// Honest-loss admission (t-1369): the final answer flags the value as
/// unavailable instead of asserting one. Phrase check on a short final
/// answer — a lower-bound heuristic, only meaningful on failed cells
/// (a successful answer has nothing to admit).
fn admits_loss(content: &str) -> bool {
    const ADMISSIONS: [&str; 12] = [
        "evicted",
        "was lost",
        "lost from",
        "no longer",
        "not available",
        "unavailable",
        "unable to",
        "cannot",
        "can't",
        "could not",
        "couldn't",
        "do not have",
    ];
    let lower = content.to_lowercase();
    ADMISSIONS.iter().any(|phrase| lower.contains(phrase))
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
    /// Record full prompts on InferCall events (plumbing-only, t-1373:
    /// lets a test inspect the exact window a turn saw — e.g. that the
    /// ledger named a call at the moment it was repeated). Recorded cells
    /// keep this OFF (prompt payloads grow O(n^2)).
    full_payloads: bool,
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
        trace_full_payloads: spec.full_payloads,
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
    /// Max gc_collect hot_kept: messages the hot-keep consumer (t-1362)
    /// was protecting in a collected window. 0 on recordings made before
    /// hot-keep (the field is absent there — replayed leniently, like the
    /// pre-marker era).
    hot_kept_max: u64,
    /// Sum of gc_collect reevictions (t-1370): evictions of content
    /// already evicted before — the re-fetch-loss loop signal hot-keep
    /// should drive to ~0. Absent (0) on pre-t-1370 recordings.
    reevictions_total: u64,
    /// Max gc_collect markers_escalated (t-1370): in-window escalated
    /// honest-exit markers. Absent (0) on pre-t-1370 recordings.
    escalated_max: u64,
    /// Max gc_collect markers: in-window eviction-marker high-water mark
    /// (t-1360). 0 on recordings made before the marker mechanism (the
    /// field is absent there — see the pre-marker replay note in
    /// `replay_cell`).
    markers_max: u64,
    /// gc_collect events whose marker emission was suppressed (no room for
    /// even the coalesced line at this budget).
    markers_suppressed: u64,
    /// Assistant texts quoting eviction-marker syntax (`[gc` / `[frame`) —
    /// the model demonstrably read a marker (t-1369; lower bound, see
    /// [`mentions_marker`]).
    marker_mentions: usize,
    /// Recovery action taken (t-1369): probe re-fetch beyond the task's
    /// allowance, or recall beyond the fixture's scripted count — the
    /// marker affordances ("re-run the call", "recall the memory") acted
    /// on, whatever prompted them.
    recovered: bool,
    /// Failed cell whose final answer admits the value is unavailable
    /// instead of asserting one (t-1369; see [`admits_loss`]).
    admitted: bool,
    /// Max gc_collect ledger_entries (t-1373): itemized-entry high-water
    /// of the in-window progress ledger. 0 on pre-ledger recordings (the
    /// field is absent there — replayed leniently).
    ledger_entries_max: u64,
    /// Any gc_collect event reported an in-window ledger (t-1373).
    ledger_present_any: bool,
    /// gc_collect events whose ledger was suppressed (no room at this
    /// budget; recorded, never silent).
    ledger_suppressed_total: u64,
    /// Repeated commands issued while the then-current ledger itemized an
    /// earlier identical call (t-1373): the model re-ran work AGAINST its
    /// own in-window progress record — the restart-loop needle for the
    /// next recording round. Always <= repeat_evals.
    rpt_ledger_named: usize,
    /// Repeats of a command whose earlier call id ANY prior in-window
    /// ledger had itemized (t-1374): a ledger naming the call was in a
    /// prompt the model already saw — even if the current ledger has
    /// since coalesced the entry away — and it re-ran the command anyway.
    /// The ledger-obedience metric; rpt_ledger_named <= this <=
    /// repeat_evals.
    post_ledger_repeats: usize,
    /// Repeated commands issued after the first collection whose window
    /// carried an escalated marker (t-1370's "do not re-fetch again").
    /// A coarse escalation-obedience bound: gc_collect events do not name
    /// WHICH content escalated, so this counts every post-escalation
    /// repeat, not only repeats of the escalated content.
    repeats_after_escalation: usize,
    usage: RunUsage,
    success: bool,
    /// The final answer asserts the claim marker with a wrong claim value
    /// (see [`confabulated`]) — fabricated content for evicted material.
    confabulated: bool,
    /// The final answer asserts the claim with a STALE needle while the
    /// updated value is absent (see [`context_rot`], t-1371).
    context_rot: bool,
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
        hot_kept_max: 0,
        reevictions_total: 0,
        escalated_max: 0,
        markers_max: 0,
        markers_suppressed: 0,
        marker_mentions: 0,
        ledger_entries_max: 0,
        ledger_present_any: false,
        ledger_suppressed_total: 0,
        rpt_ledger_named: 0,
        post_ledger_repeats: 0,
        repeats_after_escalation: 0,
        recovered: false,
        admitted: false,
        usage: RunUsage::default(),
        success: fixture_success(fixture, content),
        confabulated: confabulated(fixture, content),
        context_rot: context_rot(fixture, content),
    };
    let mut seen_commands: BTreeSet<String> = BTreeSet::new();
    let mut probe_hits = 0usize;
    let mut done_usage: Option<RunUsage> = None;
    // t-1373 restart-loop needle state: which call ids the most recent
    // collection's ledger itemized, and which call ids ran which command
    // (from InferResult tool_calls — EvalCall events carry no call id).
    let mut ledger_calls: BTreeSet<String> = BTreeSet::new();
    // t-1374 obedience needles: every call id any ledger has EVER itemized
    // (the model has seen its own record name the call), and whether an
    // escalated marker has entered the window yet.
    let mut ever_ledger_calls: BTreeSet<String> = BTreeSet::new();
    let mut escalation_seen = false;
    let mut command_calls: HashMap<String, Vec<String>> = HashMap::new();
    for event in events {
        match event {
            Event::InferCall { parent_op_id, .. } => {
                if parent_op_id.is_none() {
                    metrics.turns += 1;
                }
            }
            Event::InferResult {
                response: Some(response),
                ..
            } => {
                // Marker-reaction needle (t-1369): the model's own text
                // quoting `[gc` / `[frame` — it demonstrably read a marker.
                if mentions_marker(&response.content) {
                    metrics.marker_mentions += 1;
                }
                // t-1373: map shell commands to the call ids that ran
                // them, so a later repeat can be checked against the
                // ledger's itemized call ids.
                for call in &response.tool_calls {
                    if call.name != "shell" {
                        continue;
                    }
                    if let Some(command) = call
                        .arguments
                        .get("command")
                        .and_then(serde_json::Value::as_str)
                    {
                        command_calls
                            .entry(command.trim().to_string())
                            .or_default()
                            .push(call.id.clone());
                    }
                }
            }
            Event::EvalCall { command, .. } => {
                metrics.eval_calls += 1;
                let trimmed = command.trim().to_string();
                if !seen_commands.insert(trimmed.clone()) {
                    metrics.repeat_evals += 1;
                    // t-1373: a repeat issued while the in-window ledger
                    // itemized an earlier call running this exact command
                    // — a re-run against the model's own progress record.
                    let named = command_calls
                        .get(&trimmed)
                        .into_iter()
                        .flatten()
                        .any(|call_id| ledger_calls.contains(call_id));
                    if named {
                        metrics.rpt_ledger_named += 1;
                    }
                    // t-1374: the model was ALREADY shown a ledger naming
                    // this call (in this or an earlier window) and re-ran
                    // the command anyway — the obedience metric.
                    let ever_named = command_calls
                        .get(&trimmed)
                        .into_iter()
                        .flatten()
                        .any(|call_id| ever_ledger_calls.contains(call_id));
                    if ever_named {
                        metrics.post_ledger_repeats += 1;
                    }
                    if escalation_seen {
                        metrics.repeats_after_escalation += 1;
                    }
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
                metrics.hot_kept_max = metrics
                    .hot_kept_max
                    .max(data["hot_kept"].as_u64().unwrap_or(0));
                metrics.reevictions_total += data["reevictions"].as_u64().unwrap_or(0);
                metrics.escalated_max = metrics
                    .escalated_max
                    .max(data["markers_escalated"].as_u64().unwrap_or(0));
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
                // Progress-ledger needles (t-1373): presence, itemized
                // entries, recorded suppression, and the itemized call ids
                // (each collection REPLACES the ledger, so the latest
                // event's list is exactly what the window holds). Absent
                // on pre-ledger recordings (replayed leniently).
                metrics.ledger_entries_max = metrics
                    .ledger_entries_max
                    .max(data["ledger_entries"].as_u64().unwrap_or(0));
                metrics.ledger_present_any |= data["ledger_present"].as_bool().unwrap_or(false);
                if data["ledger_suppressed"].as_bool().unwrap_or(false) {
                    metrics.ledger_suppressed_total += 1;
                }
                ledger_calls = data["ledger_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect();
                ever_ledger_calls.extend(ledger_calls.iter().cloned());
                if data["markers_escalated"].as_u64().unwrap_or(0) > 0 {
                    escalation_seen = true;
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
    // Recovery actions (t-1369): the marker affordances acted on — re-run
    // the named call (probe re-fetch) or recall beyond the script.
    metrics.recovered =
        metrics.needle_refetches > 0 || metrics.recall_calls > fixture.scripted_recalls;
    metrics.admitted = !metrics.success && admits_loss(content);
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

/// The recording plan, in SPEND-PRIORITY order. Five generations:
///
/// t-1374 ledger recording round (first, so the spend cap can never cut
/// the deciding cells): does the t-1373 progress ledger break the restart
/// loop when a real model sees it?
///
/// 1. the loop-breakers — the worst restart-loop cells across the prior
///    generations (rpt 19-25 at the turn cap): tangent-return stack
///    guided n=2, distractor-update semantic guided n=2, and
///    distractor-update stack guided n=1;
/// 2. the four cells the t-1371 hard cap left unfunded: distractor-update
///    mark-sweep (the strategy that historically completes), clean-long
///    stack + mark-sweep (P3's missing halves), and the class-3
///    tangent-return semantic regime contrast;
/// 3. early-needle stack guided n=1 — the t-1369 recovery loop; hot-keep
///    mostly fixed the value loss, the ledger is belt-and-braces, so the
///    cell measures the composition;
/// 4. early-needle ring guided n=1 — the narrate-to-cap cell — last.
///
/// Earlier generations:
///
/// t-1371 curation-regime cells (the pre-registered plan order —
/// evals/gc/README.md "GC as curation — PRE-REGISTRATION" — puts the
/// hypothesis-deciding cells ahead of everything, so the spend cap can
/// never cut them):
///
/// 1. class-1 deciding cells — `distractor-update` semantic + control,
///    n=2 (the strong form's accuracy comparison);
/// 2. class-2 control pair — `clean-long` semantic + control, n=2 (the
///    refutation control that makes a class-1 win believable);
/// 3. remaining arms n=1 — stack and mark-sweep on both new fixtures;
/// 4. class-3 regime contrast — `tangent-return` semantic guided s1 at
///    the starvation budget (the tuned curator in the regime where its
///    arm has thrash-looped every generation).
///
/// All guided (the shipped default; at budget 8000 the t-1368 gate
/// delivers the MINIMAL fragment variant — verified before
/// pre-registration and part of the tested configuration). Earlier
/// generations:
///
/// t-1369 marker-era re-record (first, so the spend cap can never cut it):
/// t-1360 gave every strategy eviction markers, which invalidated the
/// pre-marker recordings' gc streams (replayed leniently, but no basis
/// for any behavioral claim about markers). The deciding cells:
///
/// 1. early-needle x all four strategies x guided, n=2 — do the
///    fabricators flip to honest recovery (re-run / recall / admit) now
///    that markers name what was evicted and how to get it back? Guided
///    is the shipped default; at these budgets the t-1368 gate suppresses
///    the fragment, so the markers are the whole intervention.
/// 2. early-needle stack unguided, n=2 — the marker-vs-text isolation
///    pair: with the fragment suppressed, guided and unguided prompts are
///    byte-identical, so any guided/unguided delta here bounds sampling
///    variance rather than measuring a text effect.
/// 3. tangent-return stack + mark-sweep guided, n=1 — does marker
///    presence change the thrash loop?
/// 4. memory-discipline ring + stack guided, n=1 — spot cells: does a
///    marker naming an evicted recall change memory discipline?
///
/// The stale pre-marker recordings at these paths were deleted with this
/// change (the t-1364/t-1367 tables in evals/gc/README.md are the
/// historical record); the remaining pre-marker cells keep replaying
/// leniently. Earlier generations:
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
    // t-1167 generational round, priority order (first, so the spend cap
    // can never cut the deciding cells): the behavioral canon vs the
    // existing baselines — does generational match mark-sweep's
    // behavioral wins (the only tangent-return completer across
    // generations; the honest early-needle answers) while beating its
    // starvation-budget ledger-suppression ceiling and retention
    // weaknesses? n=2 where the canon's deciding failures live
    // (early-needle confabulation/recovery; tangent-return restart
    // loop), n=1 on the curation fixtures. All guided (the shipped
    // default; at the starvation budgets the t-1368 gate suppresses the
    // fragment, at 8000 the minimal core renders — identical delivery to
    // every baseline recording).
    for fixture in ["early-needle", "tangent-return"] {
        for sample in [1, 2] {
            cells.push(CellId {
                fixture,
                arm: Arm::Generational,
                guided: true,
                sample,
            });
        }
    }
    for fixture in ["distractor-update", "clean-long"] {
        cells.push(CellId {
            fixture,
            arm: Arm::Generational,
            guided: true,
            sample: 1,
        });
    }
    // t-1374 ledger round, priority order (see the doc comment above).
    for sample in [1, 2] {
        cells.push(CellId {
            fixture: "tangent-return",
            arm: Arm::Stack,
            guided: true,
            sample,
        });
    }
    for sample in [1, 2] {
        cells.push(CellId {
            fixture: "distractor-update",
            arm: Arm::Semantic,
            guided: true,
            sample,
        });
    }
    for (fixture, arm) in [
        ("distractor-update", Arm::Stack),
        ("distractor-update", Arm::MarkSweep),
        ("clean-long", Arm::Stack),
        ("clean-long", Arm::MarkSweep),
        ("tangent-return", Arm::Semantic),
        ("early-needle", Arm::Stack),
        ("early-needle", Arm::Ring),
    ] {
        cells.push(CellId {
            fixture,
            arm,
            guided: true,
            sample: 1,
        });
    }
    // t-1371 curation-regime cells, in pre-registered priority order.
    for fixture in ["distractor-update", "clean-long"] {
        for sample in [1, 2] {
            for arm in [Arm::Semantic, Arm::NoGc] {
                cells.push(CellId {
                    fixture,
                    arm,
                    guided: true,
                    sample,
                });
            }
        }
    }
    for fixture in ["distractor-update", "clean-long"] {
        for arm in [Arm::Stack, Arm::MarkSweep] {
            cells.push(CellId {
                fixture,
                arm,
                guided: true,
                sample: 1,
            });
        }
    }
    cells.push(CellId {
        fixture: "tangent-return",
        arm: Arm::Semantic,
        guided: true,
        sample: 1,
    });
    // t-1369 marker-era cells, in deciding-question-first order.
    for arm in [Arm::Ring, Arm::Stack, Arm::MarkSweep, Arm::Semantic] {
        for sample in [1, 2] {
            cells.push(CellId {
                fixture: "early-needle",
                arm,
                guided: true,
                sample,
            });
        }
    }
    for sample in [1, 2] {
        cells.push(CellId {
            fixture: "early-needle",
            arm: Arm::Stack,
            guided: false,
            sample,
        });
    }
    for arm in [Arm::Stack, Arm::MarkSweep] {
        cells.push(CellId {
            fixture: "tangent-return",
            arm,
            guided: true,
            sample: 1,
        });
    }
    for arm in [Arm::Ring, Arm::Stack] {
        cells.push(CellId {
            fixture: "memory-discipline",
            arm,
            guided: true,
            sample: 1,
        });
    }
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
    // Later generations re-plan cells earlier ones already carry (the
    // t-1369 block owns several t-1367/t-1364 paths): first occurrence —
    // highest priority — wins.
    let mut unique: Vec<CellId> = Vec::with_capacity(cells.len());
    for cell in cells {
        if !unique.contains(&cell) {
            unique.push(cell);
        }
    }
    unique
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
        "{:<18} {:<10} {:<4} {:>1} {:>5} {:>5} {:>4} {:>4} {:>4} {:>4} {:>4} {:>3} {:>4} {:>3} {:>4} {:<11} {:>4} {:>3} {:>3} {:>4} {:>3} {:>3} {:>5} {:>3} {:>8} {:>8} {:>10} {:>6} {:>3} {:>4} {:>3} {:>4} {:>4} {:>5}",
        "fixture",
        "arm",
        "guid",
        "s",
        "turns",
        "evals",
        "rpt",
        "rptl",
        "pldr",
        "rpte",
        "refx",
        "rem",
        "prem",
        "rec",
        "coll",
        "reasons",
        "drop",
        "ovl",
        "hot",
        "reev",
        "esc",
        "mkr",
        "mkref",
        "ldg",
        "in_tok",
        "out_tok",
        "cost",
        "wall_s",
        "ok",
        "cfab",
        "rot",
        "rcov",
        "admt",
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
        "{:<18} {:<10} {:<4} {:>1} {:>5} {:>5} {:>4} {:>4} {:>4} {:>4} {:>4} {:>3} {:>4} {:>3} {:>4} {:<11} {:>4} {:>3} {:>3} {:>4} {:>3} {:>3} {:>5} {:>3} {:>8} {:>8} {:>10} {:>6.1} {:>3} {:>4} {:>3} {:>4} {:>4} {:>5}",
        fixture,
        arm.label(),
        if meta.guided { "on" } else { "off" },
        meta.sample,
        metrics.turns,
        metrics.eval_calls,
        metrics.repeat_evals,
        metrics.rpt_ledger_named,
        metrics.post_ledger_repeats,
        metrics.repeats_after_escalation,
        metrics.needle_refetches,
        metrics.remember_calls,
        metrics.proactive_remembers,
        metrics.recall_calls,
        metrics.collections,
        reasons_label(&metrics.reasons),
        metrics.dropped_total,
        metrics.overlap_total,
        metrics.hot_kept_max,
        metrics.reevictions_total,
        metrics.escalated_max,
        metrics.markers_max,
        metrics.marker_mentions,
        metrics.ledger_entries_max,
        metrics.usage.input_tokens,
        metrics.usage.output_tokens,
        metrics
            .usage
            .cost_micro_usd
            .map_or_else(|| "-".into(), agent_core::format_micro_usd),
        meta.wall_ms as f64 / 1000.0,
        if metrics.success { "yes" } else { "NO" },
        if metrics.confabulated { "YES" } else { "-" },
        if metrics.context_rot { "YES" } else { "-" },
        if metrics.recovered { "yes" } else { "-" },
        if metrics.admitted { "yes" } else { "-" },
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
        // Generational (t-1167) postdates the legacy era; no legacy cell
        // ever existed for it.
        for arm in Arm::ALL.into_iter().filter(|arm| *arm != Arm::Generational) {
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

    // Section 3 — the t-1371 curation-regime matrix (pre-registered; see
    // evals/gc/README.md). All cells guided (the shipped default; the
    // MINIMAL fragment variant renders at budget 8000). Ring is not an
    // arm here — the pre-registration tests the tuned stack (stack,
    // semantic+cited-keep+hot-keep, mark-sweep) against the control.
    println!();
    println!("== t-1371 curation regime (budget 8000, guided/minimal fragment) ==");
    print_header();
    for fixture in &curation_fixtures() {
        for arm in [
            Arm::NoGc,
            Arm::Stack,
            Arm::MarkSweep,
            Arm::Semantic,
            Arm::Generational,
        ] {
            for sample in [1, 2] {
                let path = cell_path(&dir, fixture.name, arm, true, sample);
                if !path.exists() {
                    let cell = CellId {
                        fixture: fixture.name,
                        arm,
                        guided: true,
                        sample,
                    };
                    if planned.contains(&cell) {
                        println!(
                            "{:<18} {:<10} on   {} skipped: planned cell not recorded (UNFUNDED)",
                            fixture.name,
                            arm.label(),
                            sample,
                        );
                    }
                    continue;
                }
                replay_and_print(&path, fixture, arm, true, &mut judge).await?;
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
            full_payloads: false,
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
    // Pre-hot-keep recordings (before t-1362's chunk-normalized write
    // barrier + hot-keep consumer): the replayed collector now protects
    // re-acquired content and reports `hot_kept`, so the gc stream cannot
    // reproduce a recording made without it — the same lenient stance as
    // the pre-marker era, detected by the absent `hot_kept` field.
    // Everything effect-replayed (answers, turns, tool counts, usage)
    // still reproduces exactly.
    let pre_hotkeep_recording = recorded_events.iter().any(|event| {
        matches!(event, Event::Custom { name, data, .. }
            if name == "gc_collect" && data.get("hot_kept").is_none())
    });
    // Pre-ledger recordings (before t-1373's progress ledger): the
    // replayed collector now maintains an in-window `[gc-ledger]` digest,
    // whose tokens shift what content-sensitive collection drops, so the
    // gc stream cannot reproduce a recording made without it — the same
    // lenient stance as the two eras above, detected by the absent
    // `ledger_present` field. Everything effect-replayed (answers, turns,
    // tool counts, usage) still reproduces exactly.
    let pre_ledger_recording = recorded_events.iter().any(|event| {
        matches!(event, Event::Custom { name, data, .. }
            if name == "gc_collect" && data.get("ledger_present").is_none())
    });
    let mut replayed_cmp = replayed.clone();
    if bound_errors || pre_marker_recording || pre_hotkeep_recording || pre_ledger_recording {
        replayed_cmp.collections = recorded.collections;
        replayed_cmp.reasons = recorded.reasons.clone();
        replayed_cmp.dropped_total = recorded.dropped_total;
        replayed_cmp.overlap_total = recorded.overlap_total;
        replayed_cmp.recall_hot_max = recorded.recall_hot_max;
        replayed_cmp.hot_kept_max = recorded.hot_kept_max;
        replayed_cmp.reevictions_total = recorded.reevictions_total;
        replayed_cmp.escalated_max = recorded.escalated_max;
        replayed_cmp.markers_max = recorded.markers_max;
        replayed_cmp.markers_suppressed = recorded.markers_suppressed;
        replayed_cmp.ledger_entries_max = recorded.ledger_entries_max;
        replayed_cmp.ledger_present_any = recorded.ledger_present_any;
        replayed_cmp.ledger_suppressed_total = recorded.ledger_suppressed_total;
        replayed_cmp.rpt_ledger_named = recorded.rpt_ledger_named;
        replayed_cmp.post_ledger_repeats = recorded.post_ledger_repeats;
        replayed_cmp.repeats_after_escalation = recorded.repeats_after_escalation;
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
    // t-1362 promotion assertion: on the recordings whose traces show
    // re-injection (a recovery action — the t-1369 re-fetch/recall cells,
    // where t-1351's exact-hash barrier fired 0/15), the chunk-normalized
    // barrier must fire during replay. GC re-runs live under replay, so
    // this drives the new matcher over the REAL recorded sessions.
    // (Observed on the full recording set: every GC cell fires — 3..6336
    // overlap events — and every control cell stays at 0.)
    if pre_hotkeep_recording && recorded.collections > 0 && recorded.recovered {
        assert!(
            replayed.overlap_total > 0,
            "{}: recorded session re-injects evicted content but the \
             write-barrier did not fire on replay (t-1362 matcher regression)",
            path.display()
        );
    }
    // t-1373 replayed-corpus sanity: GC re-runs live under replay, so the
    // ledger builder just ran against this REAL recorded history. Every
    // GC cell that evicted content after completing tool calls must show
    // an in-window ledger (or a recorded suppression) in the replayed gc
    // stream, bounded by the entry cap; control cells must show nothing.
    let arm_collects = Arm::from_label(&meta.arm)?.collects();
    if arm_collects && replayed.collections > 0 && replayed.eval_calls > 0 {
        assert!(
            replayed.ledger_present_any || replayed.ledger_suppressed_total > 0,
            "{}: replayed collections over a tool-bearing session produced \
             neither an in-window ledger nor a recorded suppression (t-1373)",
            path.display()
        );
    }
    assert!(
        replayed.ledger_entries_max <= agent_core::MAX_LEDGER_ENTRIES as u64,
        "{}: replayed ledger itemized {} entries — over the cap",
        path.display(),
        replayed.ledger_entries_max
    );
    if !arm_collects {
        assert!(
            !replayed.ledger_present_any && replayed.ledger_suppressed_total == 0,
            "{}: control arm must never carry a ledger",
            path.display()
        );
    }
    // The t-1373 deciding needle, driven by the real histories: on every
    // recording with the full restart-loop signature (>= 20 repeated
    // commands — the 25-collection turn-cap loops of the tangent-return
    // thrash cells and their kin), the replayed ledger must have NAMED at
    // least one repeated call in its in-window digest at the moment of
    // repetition. This is the offline form of the future recording
    // round's question: the record was there; whether models consult it
    // is what the online round measures. A lower bound, not a universal:
    // at starvation budgets a heavily-suppressed ledger can miss milder
    // repeat patterns (observed: early-needle stack guided s2, rpt 17,
    // suppressed 17/22 collections). Observed on the loop cells proper:
    // rpt_ledger_named 11-22.
    if arm_collects && recorded.repeat_evals >= 20 {
        assert!(
            replayed.rpt_ledger_named > 0,
            "{}: a restart-loop recording ({} repeats) replayed with no \
             ledger-named repeat — the digest failed to name the loop (t-1373)",
            path.display(),
            recorded.repeat_evals,
        );
    }
    if arm_collects {
        // Diagnostic (visible under --nocapture): what the live ledger
        // builder produced when driven by this recorded history — the
        // t-1373 replayed-corpus evidence. The printed table shows the
        // RECORDED metrics (pre-ledger recordings have none).
        println!(
            "  [t-1373 replay] {}: ledger present={} suppressed={} entries_max={} rpt_ledger_named={}",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            replayed.ledger_present_any,
            replayed.ledger_suppressed_total,
            replayed.ledger_entries_max,
            replayed.rpt_ledger_named,
        );
    }
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
    let fixtures = all_fixtures();

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
                full_payloads: false,
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
        rot_needles: vec![],
        claim_marker: "RESULT",
        probe: "fat.txt",
        probe_allowance: 1,
        scripted_recalls: 0,
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
            full_payloads: false,
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
        metrics.recovered,
        "re-fetch + unscripted recall count as recovery actions (t-1369)"
    );
    assert!(
        !metrics.admitted,
        "a successful answer has nothing to admit"
    );
    assert_eq!(
        metrics.marker_mentions, 0,
        "scripted turns never quote marker syntax"
    );
    assert!(
        metrics.collections > 0,
        "two 6KB tool results under a 400-token budget must collect"
    );
    // Progress-ledger plumbing (t-1373): an evicting, tool-bearing session
    // ends every collection with an in-window ledger or a recorded
    // suppression, entries stay under the cap, and the turn-2 repeat of
    // `cat fat.txt` was issued while the ledger itemized call-1 — the
    // restart-loop needle counts it.
    assert!(
        metrics.ledger_present_any || metrics.ledger_suppressed_total > 0,
        "collections fired but no ledger and no recorded suppression"
    );
    assert!(metrics.ledger_entries_max <= agent_core::MAX_LEDGER_ENTRIES as u64);
    // The turn-2 repeat preceded the first eviction (the first collection
    // only truncated the oversized result in place, dropping nothing), so
    // it is NOT ledger-named: the needle counts re-runs issued against
    // the model's own in-window record, never all repeats. Same for the
    // cumulative t-1374 obedience needle — no ledger had named the call
    // yet when the repeat was issued.
    assert_eq!(metrics.rpt_ledger_named, 0);
    assert_eq!(metrics.post_ledger_repeats, 0);
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

/// The restart-loop needle, end to end (t-1373): when the model repeats a
/// command after a collection, the WINDOW of the repeating turn already
/// contained a `[gc-ledger]` digest naming the earlier identical call —
/// i.e. the ledger would have told the model this step was done at the
/// moment it repeated it. Full prompt payloads are recorded so the test
/// inspects the exact window the turn saw; `rpt_ledger_named` counts the
/// same fact from the trace alone (what the future recording round's
/// table reports as `rptl`).
#[tokio::test]
async fn plumbing_ledger_names_the_repeated_call_at_the_moment_of_repetition() -> Result<()> {
    let fixture = Fixture {
        name: "plumbing-ledger",
        task: "plumbing".into(),
        context_budget: 700,
        needles: vec!["391"],
        ordered_needles: vec![],
        rot_needles: vec![],
        claim_marker: "RESULT",
        probe: "a.txt",
        probe_allowance: 1,
        scripted_recalls: 0,
        files: vec![
            // Medium files: small enough that no single result trips the
            // truncate pre-pass (which is not an eviction), big enough
            // that the accumulated window forces real drops.
            ("a.txt", filler(&MANUAL_WORDS, 9, 1600)),
            ("b.txt", filler(&INGEST_WORDS, 5, 1600)),
        ],
    };
    let provider = Arc::new(ScriptedProvider::new(vec![
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-1",
            "shell",
            serde_json::json!({ "command": "cat a.txt" }),
        )]),
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-2",
            "shell",
            serde_json::json!({ "command": "cat b.txt" }),
        )]),
        // The restart-loop signature: the model re-runs the exact command
        // it already ran, after a collection evicted the result.
        ScriptedProvider::calls(vec![ToolCall::new(
            "call-3",
            "shell",
            serde_json::json!({ "command": "cat a.txt" }),
        )]),
        ScriptedProvider::text("RESULT 391"),
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
                ChatMessage::user("read fat.txt and answer"),
            ],
            guidance: cell_guidance(false),
            full_payloads: true,
        },
        &workdir,
        &memory_dir,
    )
    .await?;
    let _ = fs::remove_dir_all(&workdir);
    let _ = fs::remove_dir_all(&memory_dir);

    let metrics = metrics_from_events(&run.events, &run.content, &fixture)?;
    assert_eq!(metrics.repeat_evals, 1);
    assert_eq!(
        metrics.rpt_ledger_named, 1,
        "the repeat must count as issued against the ledger's own record"
    );
    assert_eq!(
        metrics.post_ledger_repeats, 1,
        "a ledger-named repeat is also a post-ledger repeat (t-1374)"
    );
    assert!(metrics.ledger_present_any);

    // The stronger form, from the recorded window itself: the prompt of
    // the repeating turn (the second parent Infer) carried a [gc-ledger]
    // message naming call-1 and its command.
    let prompts: Vec<&Vec<ChatMessage>> = run
        .events
        .iter()
        .filter_map(|event| match event {
            Event::InferCall {
                parent_op_id: None,
                prompt: Some(prompt),
                ..
            } => Some(prompt),
            _ => None,
        })
        .collect();
    assert!(prompts.len() >= 3, "expected full prompts on parent turns");
    let repeat_window = prompts[2];
    let ledger = repeat_window
        .iter()
        .find(|message| agent_core::is_gc_ledger(message))
        .expect("the repeating turn's window carries the progress ledger");
    let content = ledger.content.as_deref().unwrap_or("");
    assert!(
        content.contains("call-1") && content.contains("cat a.txt"),
        "the ledger names the completed call and its command: {content}"
    );
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
        rot_needles: vec![],
        claim_marker: "RESULT",
        probe: "fat.txt",
        probe_allowance: 1,
        scripted_recalls: 0,
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
            full_payloads: false,
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
    assert!(
        !metrics.ledger_present_any && metrics.ledger_suppressed_total == 0,
        "no collections = no ledger (t-1373)"
    );
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

/// The context-rot flag (t-1371): fires exactly when the claim is asserted
/// with the stale value and without the updated one; correct answers,
/// non-answers, prose mentions of the old value alongside a correct claim,
/// and non-stale wrong answers (plain confabulation) do not flag.
#[test]
fn context_rot_flag_detects_stale_claims() {
    let fixtures = curation_fixtures();
    let distractor = fixtures
        .iter()
        .find(|f| f.name == "distractor-update")
        .unwrap();
    // The context-rot failure: the stale price used in the claim.
    assert!(context_rot(distractor, "QUOTE UNIT 42 TOTAL 420"));
    assert!(confabulated(distractor, "QUOTE UNIT 42 TOTAL 420"));
    // Correct answer: no rot, no confabulation.
    assert!(!context_rot(distractor, "QUOTE UNIT 57 TOTAL 570"));
    assert!(!confabulated(distractor, "QUOTE UNIT 57 TOTAL 570"));
    // Prose mention of the superseded value next to a correct claim: clean.
    assert!(!context_rot(
        distractor,
        "The v1 price ($42) was superseded. QUOTE UNIT 57 TOTAL 570"
    ));
    // Wrong but not stale: confabulation without rot (the flags separate
    // "invented" from "quoted the superseded value").
    assert!(!context_rot(distractor, "QUOTE UNIT 99 TOTAL 990"));
    assert!(confabulated(distractor, "QUOTE UNIT 99 TOTAL 990"));
    // A non-answer cannot rot.
    assert!(!context_rot(distractor, ""));
    assert!(!context_rot(distractor, "I could not finish the steps."));
    // The dead approach's numbers in the claim: confabulation, and not
    // counted as rot (the rot needles are the superseded price only).
    assert!(!context_rot(distractor, "QUOTE UNIT 38 TOTAL 380"));
    assert!(confabulated(distractor, "QUOTE UNIT 38 TOTAL 380"));

    let clean = fixtures.iter().find(|f| f.name == "clean-long").unwrap();
    // No rot needles: the flag can never fire, even on wrong answers.
    assert!(!context_rot(
        clean,
        "REGION SOUTH-2 SHIPPED 9 AUDIT AUD-0000"
    ));
    assert!(confabulated(
        clean,
        "REGION SOUTH-2 SHIPPED 9 AUDIT AUD-0000"
    ));
    assert!(!confabulated(
        clean,
        "REGION NORTH-7 SHIPPED 8 AUDIT AUD-4413"
    ));
}

/// Marker-reaction needles (t-1369): syntax quotes count, prose does not
/// (the shipped `remember` description says "evicted" in every cell's
/// offer, so prose can never be attributed to a marker); admissions flag
/// unavailable-value answers, not asserted ones.
#[test]
fn marker_reaction_needles() {
    assert!(mentions_marker(
        "A [gc: 3 evicted — shell call-2; recover: re-run the call] line is present."
    ));
    assert!(mentions_marker(
        "I see [frame call-1: shell(cat manual.txt) -> ... — evicted; re-run to recover]."
    ));
    assert!(!mentions_marker(
        "old tool output may be evicted from your context — I will re-run the command"
    ));
    assert!(admits_loss(
        "The access code was evicted from context and I cannot recover it."
    ));
    assert!(admits_loss("ACCESS <no longer available> TOTAL 21"));
    assert!(!admits_loss("ACCESS MX-7749-KESTREL TOTAL 21"));
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
    for fixture in all_fixtures() {
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
            "distractor-update" => {
                let find = |name: &str| {
                    fixture
                        .files
                        .iter()
                        .find(|(file, _)| *file == name)
                        .unwrap_or_else(|| panic!("missing {name}"))
                        .1
                        .as_str()
                };
                let v1 = find("catalog/pricing-v1.txt");
                let v2 = find("catalog/pricing-v2.txt");
                assert!(v1.contains("unit price: $42"), "stale value in v1");
                assert!(!v1.contains("57"), "v1 must not leak the updated price");
                assert!(v2.contains("unit price: $57"), "updated value in v2");
                assert!(v2.contains("supersedes v1"), "v2 must declare supersession");
                assert!(!v2.contains("42"), "v2 must not repeat the stale price");
                assert!(find("orders/current.txt").contains("10 units"));
                // Scoreable both ways: the success and rot needle sets are
                // non-empty and disjoint.
                assert!(!fixture.rot_needles.is_empty());
                for rot in &fixture.rot_needles {
                    assert!(!fixture.needles.contains(rot));
                }
            }
            "clean-long" => {
                let total: usize = fixture
                    .files
                    .iter()
                    .filter(|(name, _)| name.starts_with("shipments/"))
                    .map(|(_, content)| content.matches("SHIPPED: OK").count())
                    .sum();
                assert_eq!(total, 8);
                let region = fixture
                    .files
                    .iter()
                    .find(|(name, _)| *name == "depot/region.txt")
                    .expect("region file");
                assert!(region.1.contains("region code: NORTH-7"));
                let audit = fixture
                    .files
                    .iter()
                    .find(|(name, _)| *name == "depot/audit.txt")
                    .expect("audit file");
                assert!(audit.1.contains("AUD-4413"));
                assert!(fixture.rot_needles.is_empty(), "nothing stale to quote");
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

/// t-1167 offline corpus check, the t-1373 pattern: GC re-runs live during
/// replay, so substituting GenerationalGc for every recorded cell's arm
/// drives its tier assigner and collector over the REAL histories of all
/// six behavioral-eval generations (provider/tool effects replay by effect
/// id, so the sessions reproduce regardless of window bytes). Asserted per
/// recording:
///
/// - effect-replay integrity: the final answer reproduces;
/// - the tiny budgets force collections, and every gc_collect carries the
///   `tiers` object;
/// - tier sanity on real windows: assignments cover the window, a nursery
///   always exists, and the ladder accounting holds — no nursery eviction
///   without the floor-relax rung, no hot eviction without a degrade rung,
///   no warm eviction without warm-relax (cold never captures protected
///   content: anything hot/cited is by construction not cold, and the
///   accounting proves the protected tiers were not silently reclaimed);
/// - the ledger discipline holds for the new strategy (present or
///   recorded-suppressed whenever a tool-bearing session evicted).
#[tokio::test]
async fn generational_replays_the_six_generation_corpus_with_sane_tiers() -> Result<()> {
    let dir = recordings_dir()?;
    if !dir.exists() {
        println!("no recordings — offline no-op");
        return Ok(());
    }
    let fixtures = all_fixtures();
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "recordings dir exists but is empty");

    let mut replayed_cells = 0usize;
    for path in paths {
        let (meta, recorded_events) = load_cell_recording(&path)?;
        let Some(fixture) = fixtures.iter().find(|fixture| fixture.name == meta.fixture) else {
            panic!(
                "{}: recording names unknown fixture {}",
                path.display(),
                meta.fixture
            );
        };
        let replay = IrReplayTrace::from_events(&recorded_events)
            .with_context(|| format!("building replay from {}", path.display()))?;
        let prompt = vec![
            ChatMessage::system(system_prompt()),
            ChatMessage::user(fixture.task.clone()),
        ];
        let workdir = materialize_fixture(fixture)?;
        let memory_dir = std::env::temp_dir().join(format!("gc-gen-corpus-{}", Uuid::new_v4()));
        fs::create_dir_all(&memory_dir)?;
        let run = run_cell(
            Arc::new(ReplayOnlyProvider),
            Some(&replay),
            CellSpec {
                model: meta.model.clone(),
                gc: Arm::Generational.gc_mode(),
                context_budget: meta.context_budget,
                prompt,
                guidance: cell_guidance(meta.guided),
                full_payloads: false,
            },
            &workdir,
            &memory_dir,
        )
        .await
        .with_context(|| format!("replaying {} under generational", path.display()))?;
        let _ = fs::remove_dir_all(&workdir);
        let _ = fs::remove_dir_all(&memory_dir);

        assert_eq!(
            run.content,
            meta.final_content,
            "{}: effect-id replay must reproduce the recorded final answer \
             regardless of the substituted collector",
            path.display()
        );

        let mut collections = 0usize;
        let mut ledger_ok = 0usize;
        let mut evicting = 0usize;
        for event in &run.events {
            let Event::Custom { name, data, .. } = event else {
                continue;
            };
            if name != "gc_collect" {
                continue;
            }
            collections += 1;
            let tiers: GenerationalReport = serde_json::from_value(
                data.get("tiers")
                    .unwrap_or_else(|| {
                        panic!(
                            "{}: generational gc_collect event without a tiers object",
                            path.display()
                        )
                    })
                    .clone(),
            )?;
            let assigned = tiers.nursery + tiers.hot + tiers.warm + tiers.cold;
            assert!(assigned > 0, "{}: empty tier assignment", path.display());
            assert!(
                tiers.nursery > 0,
                "{}: a live window always has a nursery",
                path.display()
            );
            if !tiers.floor_relaxed && !tiers.prefix_relaxed {
                assert_eq!(
                    tiers.evicted_nursery,
                    0,
                    "{}: nursery evicted without the floor-relax rung: {tiers:?}",
                    path.display()
                );
            }
            if !tiers.hot_relaxed && !tiers.floor_relaxed && !tiers.prefix_relaxed {
                assert_eq!(
                    tiers.evicted_hot,
                    0,
                    "{}: hot evicted without a degrade rung: {tiers:?}",
                    path.display()
                );
            }
            if !tiers.warm_relaxed {
                assert_eq!(
                    tiers.evicted_warm,
                    0,
                    "{}: warm evicted without the warm-relax rung: {tiers:?}",
                    path.display()
                );
            }
            let dropped = data
                .get("dropped_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if dropped > 0 {
                evicting += 1;
                let present = data
                    .get("ledger_present")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let suppressed = data
                    .get("ledger_suppressed")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                if present || suppressed {
                    ledger_ok += 1;
                }
            }
        }
        assert!(
            collections > 0,
            "{}: the corpus budgets must force generational to collect",
            path.display()
        );
        assert_eq!(
            evicting,
            ledger_ok,
            "{}: every evicting collection must carry a ledger or a recorded \
             suppression",
            path.display()
        );
        replayed_cells += 1;
        println!(
            "  [t-1167 corpus] {}: collections={collections} evicting={evicting}",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
        );
    }
    println!("t-1167 corpus: {replayed_cells} recorded cells replayed under generational");
    Ok(())
}
