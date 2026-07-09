//! Runtime operations guidance (t-1359, docs/GUIDANCE.md).
//!
//! The per-tool operational text the runtime ships to its models. Guidance
//! has the same status as a tool schema: it is part of the runtime's
//! model-facing API surface (docs/GUIDANCE.md §1), so its text lives here —
//! versioned in-repo — and is kept **verbatim-synced** with the shipped-text
//! record in docs/GUIDANCE.md §4. Any edit to a constant below must update
//! the doc in the same commit, and (per the §5 promotion gate) invalidates
//! the recording that validated the text, requiring a cheap re-record.
//!
//! None of this text enters the program hash or effect identity: tool
//! descriptions are provider-offer material assembled from `SeqConfig` at
//! dispatch time (`ir_tool_specs`), not program content, and recorded Infer
//! results replay by effect id regardless of prompt bytes. Pinned by test.

/// Shipped `shell` tool description. Carries the operational sentences
/// that used to live only in the CLI's default system prompt — where any
/// `--system-prompt` override silently deleted them (GUIDANCE.md §4
/// migration step 3). As a tool description it survives every prompt
/// configuration.
pub const SHELL_TOOL_DESCRIPTION: &str =
    "Execute a command string with the configured shell, inside the \
     runtime's current process environment. Use it when you need to \
     inspect or change the environment.";

/// Shipped `infer` tool description (docs/GUIDANCE.md §2.1 delegation +
/// §2.3 by-reference digestion, condensed to schema-description prose).
pub const INFER_TOOL_DESCRIPTION: &str =
    "Delegate a subtask to another (usually cheaper) model and return its \
     response. Delegate when the subtask is generation-heavy (long \
     boilerplate, many candidates to write up) or requires digesting bulky \
     material you have already fetched — pass that material by reference \
     via context_refs; never paste large tool output into the prompt \
     (copying it costs output-rate tokens and the copy rides your history \
     afterward). Do NOT delegate questions you can answer directly, or work \
     a one-line command can do: a delegation costs a full provider \
     round-trip of serialized latency. Child prompts must be \
     self-contained — the child sees only your prompt and the referenced \
     tool results, never this conversation, and has no tools.";

/// Shipped `context_refs` parameter description (docs/GUIDANCE.md §2.3:
/// digest bulky command output by reference, fetch and delegate in one
/// turn).
pub const INFER_CONTEXT_REFS_DESCRIPTION: &str =
    "ids of prior tool calls from this conversation (e.g. a shell call's \
     id): each referenced call's result is delivered to the sub-model \
     directly, ahead of the prompt, without being repeated here. Use this \
     for bulky output you only need distilled — you can issue the fetch \
     and the delegation in the same turn.";

/// Shipped `remember` tool description (docs/GUIDANCE.md §2.2). The
/// previous text framed memory as cross-session only ("worth keeping
/// beyond this conversation") — precisely the framing §2.2 identifies as
/// the unguided failure mode, since GC destroys intra-session
/// intermediates.
pub const REMEMBER_TOOL_DESCRIPTION: &str =
    "Save a distilled fact to persistent memory. Save load-bearing \
     intermediate results as soon as you produce them — decisions, \
     distilled findings, computed values later steps depend on — because \
     old tool output may be evicted from your context and a fact you did \
     not save may be gone when you need it. Save the distilled fact, not \
     raw output: one or two sentences with the concrete values in them. \
     Anything worth keeping beyond this session — user preferences, \
     project conventions, decisions — also belongs here.";

/// Shipped `recall` tool description (docs/GUIDANCE.md §2.2).
pub const RECALL_TOOL_DESCRIPTION: &str =
    "Search persistent memory by keywords and return matching notes. When \
     you need something you saw earlier and it is no longer in view, \
     recall it instead of re-running commands or guessing.";

// --- the runtime-guidance fragment (docs/GUIDANCE.md §4 step 2) -----------
//
// Cross-tool workflow text, assembled at prompt build from live
// capabilities and delivered as a PromptIR Developer/Constraint section
// (`SectionOrigin::Static { name: "runtime-guidance" }`). Each block ships
// only when its capability is live — guidance about a tool the model does
// not have is noise, and noise is the failure mode of guidance itself
// (§2's design invariant). Block text below is the §2 drafted guidance
// verbatim; the shipped-fragment record in docs/GUIDANCE.md is the same
// text by contract.

/// §2.1 delegation block (including the §2.3 by-reference chaining bullet,
/// which ships inside it). Delivered only when the `infer` tool is in the
/// call's tool offer.
pub const DELEGATION_BLOCK: &str = "\
You can delegate subtasks to another model with the `infer` tool.

- Delegate when a subtask is generation-heavy (long boilerplate, many \
candidates to write up) or requires digesting bulky material you have \
already fetched.
- Pass bulky material by reference: set `context_refs` to the ids of the \
tool calls that produced it. Never paste large tool output into the child \
prompt — copying it costs output-rate tokens and the copy rides your \
history afterward.
- Do NOT delegate questions you can answer directly, or work a one-line \
command can do: a delegation costs a full provider round-trip.
- Delegation trades money for time: the child round-trip is serialized \
latency. Prefer doing it yourself when wall-clock matters.
- Child prompts must be self-contained. The child sees only your prompt \
and the referenced tool results — never this conversation — and has no \
tools.
- When a command produces bulky output you only need distilled (logs, \
dumps, fetched documents), do not read it line-by-line into your reply: \
call `infer` with `context_refs` naming that shell call's id and ask for \
the distilled answer. You can issue the fetch and the delegation in the \
same turn.";

/// §2.2 store/retrieve discipline block — **DRAFT, not shipped** (demoted
/// t-1368). Shipped default-on with t-1359; the t-1364 A/B failed the
/// promotion gate's "target behavior moved" requirement: zero proactive
/// saves (rem=0, prem=0, rec=0) in all 12 guided cells outside the
/// scripted memory fixture, and no improvement on the memory fixture
/// itself. The remember/recall TOOL descriptions (step 1) remain shipped;
/// it is this fragment block that failed. The constant stays as the draft
/// text (docs/GUIDANCE.md §2.2) pending a rework + re-record. The drafted
/// "retained preferentially" sentence is no longer mechanism-blocked —
/// hot-keep (t-1362) consumes `recall_hot` in every strategy — but stays
/// out with the rest of the draft block until the rework's re-record
/// passes the promotion gate.
pub const MEMORY_BLOCK: &str = "\
You have persistent memory via `remember` and `recall`.

- Save load-bearing intermediate results as soon as you produce them: \
decisions, distilled findings, computed values later steps depend on. \
Your context window is managed — old tool output is evicted under \
pressure, and a fact you did not save may be gone when you need it.
- Save the distilled fact, not raw output: one or two sentences with the \
concrete values in them.
- When you need something you saw earlier and it is no longer in view, \
`recall` it instead of re-running commands or guessing.
- Anything worth keeping beyond this session — user preferences, project \
conventions, decisions — always belongs in memory.";

/// §2.4 GC-awareness block, memory-tools variant. Delivered only when a GC
/// strategy is active AND the memory tools are offered (the
/// `remember`/`recall` cross-references would otherwise name tools the
/// model does not have). The marker sentences describe the REAL t-1360
/// mechanism (gc.rs eviction markers + frame annotations) — mechanism
/// first, text describing it second, per the strategy-honesty rule.
pub const GC_BLOCK_WITH_MEMORY: &str = "\
Your context window is managed. In long sessions, old tool results are \
collapsed to one-line `[frame ...]` annotations or replaced by `[gc: ...]` \
eviction markers.

- Extract what matters from a result when you see it — into your reply, \
or into memory with `remember`. Do not plan to re-read old output \
verbatim later.
- A `[frame ...]` or `[gc: ...]` line means that content is gone. The \
marker names what was evicted and how to recover it: re-run the named \
tool call, `recall` the named memory, or ask the user again — do not \
guess at what it contained.
- A marker saying content \"cannot stay in context\" means re-fetching \
will not help: summarize what you need into memory with `remember` or \
ask the user, then move on.";

/// §2.4 GC-awareness block, no-memory variant: the same text with the
/// `remember`/`recall` cross-references removed (recorded as the shipped
/// no-memory variant in docs/GUIDANCE.md).
pub const GC_BLOCK_WITHOUT_MEMORY: &str = "\
Your context window is managed. In long sessions, old tool results are \
collapsed to one-line `[frame ...]` annotations or replaced by `[gc: ...]` \
eviction markers.

- Extract what matters from a result when you see it — into your reply. \
Do not plan to re-read old output verbatim later.
- A `[frame ...]` or `[gc: ...]` line means that content is gone. The \
marker names what was evicted and how to recover it: re-run the named \
tool call or ask the user again — do not guess at what it contained.
- A marker saying content \"cannot stay in context\" means re-fetching \
will not help: ask the user for what you need, then move on.";

/// §2.4 citation-protection line. Strategy-conditional (GUIDANCE.md gap
/// 6): cited-keep is implemented for the `semantic` strategy only, so this
/// line renders ONLY under `semantic` with `cited_keep` on — under any
/// other strategy it would be a false promise.
pub const GC_CITED_KEEP_LINE: &str = "\
- Referring to a tool call by its id in your text (for example, \"per the \
output of call-12\") marks that result as load-bearing and protects it \
from eviction.";

/// §2.6 approval-awareness block. Delivered only when some effect in the
/// run's config is gated.
pub const APPROVAL_BLOCK: &str = "\
Some actions require human approval before they run. When you request \
one, the run may pause — possibly for a long time — and resumes when a \
person decides. A pause is not an error: do not retry or rephrase a \
pending action.

If an action is denied, you will see a denial result. Respect it: do not \
re-attempt that action or an equivalent of it. Pursue an alternative, or \
report clearly what you could not do and why.";

// --- budget-aware delivery (t-1368) ----------------------------------------
//
// t-1364's load-bearing caveat: at 1600-2000-token context budgets the
// ~700-token fragment was 33-44% of the WHOLE budget — eviction-protected
// meta-text crowding out the task, a priority inversion. It fattened the
// system message, tripped the GC threshold on turn 1-2, and (before
// t-1367) evicted the live task; the strategies that kept the task
// thrashed instead (16-26 collections/cell). Delivery is therefore gated
// on headroom: the full fragment only when it costs at most
// [`FULL_FRAGMENT_BUDGET_SHARE`] of `context_budget`, the minimal variant
// (the do-not-guess + remember-distilled core) up to
// [`MINIMAL_FRAGMENT_BUDGET_SHARE`], and nothing above that — a budget
// that small has no room for operations prose; the task must win. Real
// deployments (100k+ windows, fragment <1%) always get the full fragment.

/// Full-fragment ceiling: 5% of the context budget. At the shipped
/// fragment's ~700 tokens this delivers full guidance on budgets >=
/// ~14k tokens — an order of magnitude above the t-1364 failure regime,
/// an order of magnitude below real deployment windows.
pub const FULL_FRAGMENT_BUDGET_SHARE: f32 = 0.05;

/// Minimal-variant ceiling: 15% of the context budget. Between the two
/// thresholds the fragment is replaced by the 2-4 sentence core; above
/// it nothing ships.
pub const MINIMAL_FRAGMENT_BUDGET_SHARE: f32 = 0.15;

/// Minimal-variant GC core (§2.4 distilled; ships when a GC strategy is
/// active and the full fragment does not fit the budget). One sentence on
/// eviction markers (t-1360) — small budgets are exactly where GC fires
/// most, so the marker format cannot be full-variant-only.
pub const MINIMAL_GC_CORE: &str = "\
Your context window is managed: old tool results are collapsed or dropped \
under pressure, and a `[gc: ...]` or `[frame ...]` marker names what was \
evicted and how to recover it. Extract what matters from a result when \
you see it; if a result you need is gone, re-run the command — do not \
guess at what it contained.";

/// Minimal-variant memory core (ships when the memory tools are offered
/// and the full fragment does not fit the budget). One sentence — this is
/// NOT the demoted §2.2 block; it is the remember-distilled core the
/// t-1368 minimal variant carries, unvalidated pending its own A/B.
pub const MINIMAL_MEMORY_CORE: &str = "\
Save load-bearing values with `remember` as you produce them — the \
distilled fact, not raw output — and `recall` them instead of re-running \
commands or guessing.";

/// Which guidance rendering one Infer call's budget allows. Trace-visible:
/// the delivered section's content hash differs per variant, and the
/// variant name rides the `prompt_ir` event (`guidance_variant`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidanceVariant {
    /// The whole capability-keyed fragment.
    Full,
    /// The 2-4 sentence do-not-guess + remember-distilled core.
    Minimal,
    /// Nothing: the budget is too small for meta-text.
    Suppressed,
}

impl GuidanceVariant {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Minimal => "minimal",
            Self::Suppressed => "suppressed",
        }
    }
}

/// Pick the variant for one call: the FULL fragment's estimated tokens
/// against the run's context budget (the same conservative estimator GC
/// budgets use).
pub fn variant_for_budget(full_fragment_tokens: usize, context_budget: usize) -> GuidanceVariant {
    let share = |ratio: f32| ((context_budget as f32) * ratio) as usize;
    if full_fragment_tokens <= share(FULL_FRAGMENT_BUDGET_SHARE) {
        GuidanceVariant::Full
    } else if full_fragment_tokens <= share(MINIMAL_FRAGMENT_BUDGET_SHARE) {
        GuidanceVariant::Minimal
    } else {
        GuidanceVariant::Suppressed
    }
}

/// The minimal variant's text for one call's capabilities: GC core and/or
/// memory core. Empty when neither applies (no GC, no memory tools) —
/// the caller ships nothing rather than padding.
pub fn minimal_runtime_guidance_fragment(caps: &GuidanceCapabilities) -> String {
    let mut blocks: Vec<&str> = Vec::new();
    if caps.gc {
        blocks.push(MINIMAL_GC_CORE);
    }
    if caps.memory {
        blocks.push(MINIMAL_MEMORY_CORE);
    }
    blocks.join("\n\n")
}

/// Assemble the fragment a call's budget allows: the full capability-keyed
/// fragment, the minimal core, or `None` (suppressed, or the selected
/// variant renders empty for these capabilities).
pub fn budgeted_runtime_guidance_fragment(
    caps: &GuidanceCapabilities,
    context_budget: usize,
) -> Option<(String, GuidanceVariant)> {
    let full = runtime_guidance_fragment(caps);
    let variant = variant_for_budget(crate::gc::estimate_text_tokens(&full), context_budget);
    let fragment = match variant {
        GuidanceVariant::Full => full,
        GuidanceVariant::Minimal => minimal_runtime_guidance_fragment(caps),
        GuidanceVariant::Suppressed => return None,
    };
    if fragment.is_empty() {
        return None;
    }
    Some((fragment, variant))
}

/// §2.5 cost-awareness block. Unconditional within the fragment (no
/// capability gate); the fragment itself is delivered only on tool-bearing
/// Infer calls.
pub const COST_BLOCK: &str = "\
Be economical. Use the fewest steps that complete the task correctly. A \
one-line command beats delegation; delegating long generation to a cheap \
model beats writing it yourself; nothing beats not doing the work twice — \
do not re-fetch or re-compute what you already have. When several tool \
calls do not depend on each other, issue them together in a single turn.";

/// Runtime-guidance delivery config, carried on
/// [`crate::interpreter::SeqConfig`]. Ships default-on; opt-out is
/// explicit and total (`--no-runtime-guidance` / SDK
/// `.runtime_guidance(false)`) — for deterministic evals and for users who
/// ship their own manual.
#[derive(Debug, Clone)]
pub struct RuntimeGuidance {
    pub enabled: bool,
    /// The interim delegate catalog (GUIDANCE.md §2.1, pending t-1345):
    /// when non-empty, the delegation block ends with an "Available
    /// delegate models: ..." line naming these ids (and rates where
    /// known). Empty by default — the runtime cannot yet promise which
    /// model ids the run's provider resolves, so deployments that know
    /// (evals, fixed-provider setups) supply it explicitly.
    pub delegate_models: Vec<DelegateModel>,
}

impl Default for RuntimeGuidance {
    fn default() -> Self {
        Self {
            enabled: true,
            delegate_models: Vec::new(),
        }
    }
}

impl RuntimeGuidance {
    /// Guidance off entirely: no fragment on any call. Tool descriptions
    /// (which survive any prompt configuration by design) are unaffected.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            delegate_models: Vec::new(),
        }
    }
}

/// One delegate-catalog entry: a model id the run's provider is known to
/// resolve, plus optional rates for the "($in/$out per Mtok)" suffix.
#[derive(Debug, Clone)]
pub struct DelegateModel {
    pub id: String,
    pub pricing: Option<crate::cost::Pricing>,
}

/// The live capabilities a fragment is assembled from — each field gates
/// its block (docs/GUIDANCE.md §2 conditionality invariant).
#[derive(Debug, Clone, Default)]
pub struct GuidanceCapabilities {
    /// The `infer` tool is in this call's tool offer.
    pub infer: bool,
    /// The `remember`/`recall` tools are in this call's tool offer.
    pub memory: bool,
    /// A GC strategy is active for this run.
    pub gc: bool,
    /// The active strategy is `semantic` with cited-keep on — the only
    /// configuration where the citation-protection line is true.
    pub cited_keep: bool,
    /// Some effect in the run's config is gated behind approval.
    pub approvals: bool,
    /// Interim delegate catalog (only rendered when `infer` is set).
    pub delegate_models: Vec<DelegateModel>,
}

/// Assemble the runtime-guidance fragment for one Infer call. `None` when
/// nothing applies (callers additionally gate on `RuntimeGuidance.enabled`
/// and on the call offering at least one tool — a bare completion child
/// gets no fragment). The cost block is unconditional per §2.5, so a
/// delivered fragment is never empty.
pub fn runtime_guidance_fragment(caps: &GuidanceCapabilities) -> String {
    let mut blocks: Vec<String> = Vec::new();
    if caps.infer {
        let mut block = DELEGATION_BLOCK.to_string();
        if !caps.delegate_models.is_empty() {
            block.push_str("\n\nAvailable delegate models: ");
            block.push_str(&delegate_catalog_line(&caps.delegate_models));
            block.push('.');
        }
        blocks.push(block);
    }
    // The §2.2 memory-discipline block is deliberately absent: demoted to
    // draft by t-1368 after its t-1364 A/B moved nothing (see MEMORY_BLOCK).
    // The §2.4 GC block below keeps its memory-keyed variant — that block's
    // remember/recall cross-references are recovery instructions, not the
    // failed save-discipline prose.
    if caps.gc {
        let mut block = if caps.memory {
            GC_BLOCK_WITH_MEMORY.to_string()
        } else {
            GC_BLOCK_WITHOUT_MEMORY.to_string()
        };
        if caps.cited_keep {
            block.push('\n');
            block.push_str(GC_CITED_KEEP_LINE);
        }
        blocks.push(block);
    }
    if caps.approvals {
        blocks.push(APPROVAL_BLOCK.into());
    }
    blocks.push(COST_BLOCK.into());
    blocks.join("\n\n")
}

fn delegate_catalog_line(models: &[DelegateModel]) -> String {
    models
        .iter()
        .map(|model| match &model.pricing {
            Some(pricing) => format!(
                "`{}` (${}/${} per Mtok)",
                model.id,
                format_usd_per_mtok(pricing.input_micro_usd_per_mtok),
                format_usd_per_mtok(pricing.output_micro_usd_per_mtok),
            ),
            None => format!("`{}`", model.id),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Render a micro-USD-per-Mtok rate as a short dollar figure ("1", "0.15").
fn format_usd_per_mtok(micro: u64) -> String {
    let rendered = format!("{:.6}", micro as f64 / 1_000_000.0);
    let trimmed = rendered.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".into()
    } else {
        trimmed.into()
    }
}

/// The stable name identifying the fragment's section in PromptIR traces.
pub const RUNTIME_GUIDANCE_SECTION: &str = "runtime-guidance";

/// Wrap an assembled fragment in its PromptIR section: Developer role,
/// Constraint composition, High priority, `Static { name:
/// "runtime-guidance" }` origin (docs/GUIDANCE.md §4). That placement makes
/// the fragment instruction-authority content, structurally distinct from
/// retrieved data, and its content hash rides the per-InferCall PromptIR
/// trace like every other section.
pub fn runtime_guidance_section(content: String) -> crate::prompt_ir::Section {
    crate::prompt_ir::Section::new(
        RUNTIME_GUIDANCE_SECTION,
        RUNTIME_GUIDANCE_SECTION,
        crate::prompt_ir::SectionSource {
            origin: crate::prompt_ir::SectionOrigin::Static {
                name: RUNTIME_GUIDANCE_SECTION.into(),
            },
            timing: crate::prompt_ir::RetrievalTiming::Passive,
            metadata: serde_json::Value::Null,
        },
        crate::prompt_ir::SectionRole::Developer,
        content,
        crate::prompt_ir::Priority::High,
        crate::prompt_ir::CompositionMode::Constraint,
        None,
        serde_json::Value::Null,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::program_hash;
    use crate::ir_agent::agent_loop_ir_with_options;
    use crate::op::Model;

    /// The §2 conditionality invariant, block by block: each block renders
    /// exactly under its capability, and the §2.5 cost block always.
    #[test]
    fn fragment_blocks_are_capability_keyed() {
        let none = runtime_guidance_fragment(&GuidanceCapabilities::default());
        assert_eq!(none, COST_BLOCK, "no capabilities = cost block only");

        let infer = runtime_guidance_fragment(&GuidanceCapabilities {
            infer: true,
            ..Default::default()
        });
        assert!(infer.contains(DELEGATION_BLOCK));
        assert!(!infer.contains("persistent memory"));
        assert!(!infer.contains("[frame"));
        assert!(!infer.contains("[gc:"));
        assert!(!infer.contains("human approval"));
        assert!(infer.contains(COST_BLOCK));

        // The §2.2 memory-discipline block is demoted to draft (t-1368):
        // memory capability alone adds NOTHING to the full fragment (the
        // tool descriptions carry the discipline; the block failed its
        // t-1364 A/B).
        let memory = runtime_guidance_fragment(&GuidanceCapabilities {
            memory: true,
            ..Default::default()
        });
        assert!(!memory.contains(MEMORY_BLOCK));
        assert_eq!(memory, COST_BLOCK);
        assert!(!memory.contains(DELEGATION_BLOCK));

        let approvals = runtime_guidance_fragment(&GuidanceCapabilities {
            approvals: true,
            ..Default::default()
        });
        assert!(approvals.contains(APPROVAL_BLOCK));
    }

    /// §2.4's memory cross-references (`remember`/`recall`) only render
    /// when the memory tools are actually offered — guidance about a tool
    /// the model does not have is noise.
    #[test]
    fn gc_block_matches_memory_availability() {
        let with_memory = runtime_guidance_fragment(&GuidanceCapabilities {
            gc: true,
            memory: true,
            ..Default::default()
        });
        assert!(with_memory.contains(GC_BLOCK_WITH_MEMORY));

        let without_memory = runtime_guidance_fragment(&GuidanceCapabilities {
            gc: true,
            ..Default::default()
        });
        assert!(without_memory.contains(GC_BLOCK_WITHOUT_MEMORY));
        assert!(!without_memory.contains("`remember`"));
        assert!(!without_memory.contains("`recall`"));
    }

    /// Strategy-honest citation guidance (GUIDANCE.md gap 6): the
    /// protection line renders ONLY under semantic + cited-keep. Under any
    /// other strategy the sentence would promise protection the collector
    /// does not implement.
    #[test]
    fn cited_keep_line_is_strategy_conditional() {
        let cited = runtime_guidance_fragment(&GuidanceCapabilities {
            gc: true,
            cited_keep: true,
            ..Default::default()
        });
        assert!(cited.contains(GC_CITED_KEEP_LINE));

        let plain = runtime_guidance_fragment(&GuidanceCapabilities {
            gc: true,
            cited_keep: false,
            ..Default::default()
        });
        assert!(!plain.contains("marks that result as load-bearing"));

        // cited_keep without gc renders no GC block at all (the line rides
        // inside it).
        let no_gc = runtime_guidance_fragment(&GuidanceCapabilities {
            cited_keep: true,
            ..Default::default()
        });
        assert!(!no_gc.contains("marks that result as load-bearing"));
    }

    /// The §2.2 "retained preferentially" claim stays out of both the
    /// draft memory block and the shipped minimal core: the mechanism now
    /// exists (hot-keep, t-1362, consumes `recall_hot` in every strategy),
    /// but the sentence ships only with the §2.2 rework once its re-record
    /// passes the promotion gate — text follows validated mechanism.
    #[test]
    fn memory_block_does_not_promise_preferential_retention_yet() {
        assert!(!MEMORY_BLOCK.contains("retained preferentially"));
        assert!(!MINIMAL_MEMORY_CORE.contains("retained preferentially"));
    }

    // --- budget-aware delivery (t-1368) -----------------------------------

    /// Variant selection at the three budget regimes, measured against the
    /// real shipped fragment size: a t-1364-sized budget (2k) suppresses,
    /// a mid budget delivers the minimal core, a deployment-sized budget
    /// delivers the full fragment.
    #[test]
    fn variant_tracks_budget_headroom() {
        let caps = GuidanceCapabilities {
            infer: true,
            memory: true,
            gc: true,
            ..Default::default()
        };
        let full_tokens = crate::gc::estimate_text_tokens(&runtime_guidance_fragment(&caps));
        assert_eq!(
            variant_for_budget(full_tokens, 2_000),
            GuidanceVariant::Suppressed,
            "the t-1364 failure regime must suppress the fragment \
             (fragment = {full_tokens} tokens)"
        );
        assert_eq!(
            variant_for_budget(full_tokens, 8_000),
            GuidanceVariant::Minimal
        );
        assert_eq!(
            variant_for_budget(full_tokens, 200_000),
            GuidanceVariant::Full,
            "deployment-sized budgets always get the full fragment"
        );

        // The boundaries are shares of the budget, exact at the estimator.
        assert_eq!(variant_for_budget(100, 2_000), GuidanceVariant::Full);
        assert_eq!(variant_for_budget(101, 2_000), GuidanceVariant::Minimal);
        assert_eq!(variant_for_budget(300, 2_000), GuidanceVariant::Minimal);
        assert_eq!(variant_for_budget(301, 2_000), GuidanceVariant::Suppressed);
    }

    /// The minimal variant is capability-keyed like the full fragment: GC
    /// core under GC, memory core when the tools are offered, empty (=
    /// nothing delivered) when neither applies.
    #[test]
    fn minimal_fragment_is_capability_keyed() {
        let both = minimal_runtime_guidance_fragment(&GuidanceCapabilities {
            gc: true,
            memory: true,
            ..Default::default()
        });
        assert!(both.contains(MINIMAL_GC_CORE));
        assert!(both.contains(MINIMAL_MEMORY_CORE));

        let gc_only = minimal_runtime_guidance_fragment(&GuidanceCapabilities {
            gc: true,
            ..Default::default()
        });
        assert!(gc_only.contains(MINIMAL_GC_CORE));
        assert!(!gc_only.contains("`remember`"));

        let neither = minimal_runtime_guidance_fragment(&GuidanceCapabilities::default());
        assert!(neither.is_empty());
        // A budget in the minimal regime for the cost-block-only fragment
        // (~120 tokens: 5% < 120/1000 <= 15%): the minimal variant renders
        // empty for these capabilities, so nothing is delivered.
        assert_eq!(
            variant_for_budget(
                crate::gc::estimate_text_tokens(&runtime_guidance_fragment(
                    &GuidanceCapabilities::default()
                )),
                1_000
            ),
            GuidanceVariant::Minimal
        );
        assert_eq!(
            budgeted_runtime_guidance_fragment(&GuidanceCapabilities::default(), 1_000),
            None,
            "an empty minimal variant delivers nothing, not padding"
        );
    }

    /// Which variant was delivered is trace-visible through the section
    /// content hash alone: full and minimal hash differently on the same
    /// capabilities.
    #[test]
    fn variant_changes_the_section_hash() {
        let caps = GuidanceCapabilities {
            infer: true,
            memory: true,
            gc: true,
            ..Default::default()
        };
        let (full, full_variant) = budgeted_runtime_guidance_fragment(&caps, 200_000).unwrap();
        let (minimal, minimal_variant) = budgeted_runtime_guidance_fragment(&caps, 8_000).unwrap();
        assert_eq!(full_variant, GuidanceVariant::Full);
        assert_eq!(minimal_variant, GuidanceVariant::Minimal);
        assert_ne!(
            runtime_guidance_section(full).hash,
            runtime_guidance_section(minimal).hash
        );
        assert_eq!(budgeted_runtime_guidance_fragment(&caps, 2_000), None);
    }

    /// The interim delegate catalog (§2.1, pending t-1345): rendered only
    /// when the deployment supplies ids, with rates where known.
    #[test]
    fn delegate_catalog_line_renders_ids_and_rates() {
        let fragment = runtime_guidance_fragment(&GuidanceCapabilities {
            infer: true,
            delegate_models: vec![
                DelegateModel {
                    id: "openai/gpt-4o-mini".into(),
                    pricing: Some(crate::cost::Pricing::from_usd_per_mtok(0.15, 0.60).unwrap()),
                },
                DelegateModel {
                    id: "bare-model".into(),
                    pricing: None,
                },
            ],
            ..Default::default()
        });
        assert!(fragment.contains(
            "Available delegate models: `openai/gpt-4o-mini` ($0.15/$0.6 per Mtok), `bare-model`."
        ));
    }

    /// The fragment's PromptIR placement per docs/GUIDANCE.md §4:
    /// Developer role, Constraint composition, High priority, Static
    /// origin named "runtime-guidance" — with a content hash, so every
    /// InferCall's PromptIR trace proves which guidance the model saw.
    #[test]
    fn section_has_the_specified_shape() {
        let section = runtime_guidance_section("text".into());
        assert_eq!(section.role, crate::prompt_ir::SectionRole::Developer);
        assert_eq!(
            section.composition,
            crate::prompt_ir::CompositionMode::Constraint
        );
        assert_eq!(section.priority, crate::prompt_ir::Priority::High);
        assert_eq!(
            section.source.origin,
            crate::prompt_ir::SectionOrigin::Static {
                name: RUNTIME_GUIDANCE_SECTION.into()
            }
        );
        assert!(section.hash.0.starts_with("sha256:"));
    }

    /// Guidance text is provider-offer material, not program content: no
    /// description string appears anywhere in the serialized program, so
    /// the program hash — and with it every effect id — is independent of
    /// guidance wording. This is the property that lets the text be tuned
    /// without breaking replay of old traces (recorded Infer results are
    /// matched by effect id, never by prompt or tool-spec bytes).
    #[test]
    fn tool_descriptions_never_enter_the_program_or_its_hash() {
        let machine = agent_loop_ir_with_options(Model("m".into()), Vec::new(), 4, true);
        let serialized = serde_json::to_string(&machine.program).unwrap();
        for text in [
            INFER_TOOL_DESCRIPTION,
            INFER_CONTEXT_REFS_DESCRIPTION,
            REMEMBER_TOOL_DESCRIPTION,
            RECALL_TOOL_DESCRIPTION,
        ] {
            assert!(
                !serialized.contains(text),
                "guidance text leaked into the program (and so its hash)"
            );
        }
        // And the hash is a pure function of that same serialized program:
        // hashing twice is stable, with no config or tool-spec input at all.
        assert_eq!(
            program_hash(&machine.program).unwrap(),
            program_hash(&machine.program).unwrap()
        );
    }
}
