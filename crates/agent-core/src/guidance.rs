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

/// §2.2 store/retrieve discipline block. Delivered only when the memory
/// tools are in the call's tool offer. The drafted "retained
/// preferentially" sentence is intentionally absent until a GC strategy
/// consumes `recall_hot` (t-1167) — guidance must never claim retention
/// the active strategy does not implement.
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
/// model does not have).
pub const GC_BLOCK_WITH_MEMORY: &str = "\
Your context window is managed. In long sessions, old tool results are \
collapsed to one-line `[frame: ...]` annotations or dropped entirely.

- Extract what matters from a result when you see it — into your reply, \
or into memory with `remember`. Do not plan to re-read old output \
verbatim later.
- A `[frame: ...]` annotation means the result body is gone. If you need \
it, re-run the command or `recall` the saved fact — do not guess at what \
it contained.";

/// §2.4 GC-awareness block, no-memory variant: the same text with the
/// `remember`/`recall` cross-references removed (recorded as the shipped
/// no-memory variant in docs/GUIDANCE.md).
pub const GC_BLOCK_WITHOUT_MEMORY: &str = "\
Your context window is managed. In long sessions, old tool results are \
collapsed to one-line `[frame: ...]` annotations or dropped entirely.

- Extract what matters from a result when you see it — into your reply. \
Do not plan to re-read old output verbatim later.
- A `[frame: ...]` annotation means the result body is gone. If you need \
it, re-run the command — do not guess at what it contained.";

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
    if caps.memory {
        blocks.push(MEMORY_BLOCK.into());
    }
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
        assert!(!infer.contains("[frame:"));
        assert!(!infer.contains("human approval"));
        assert!(infer.contains(COST_BLOCK));

        let memory = runtime_guidance_fragment(&GuidanceCapabilities {
            memory: true,
            ..Default::default()
        });
        assert!(memory.contains(MEMORY_BLOCK));
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

    /// The §2.2 "retained preferentially" claim ships only when a GC
    /// strategy consumes `recall_hot` (t-1167) — until then the memory
    /// block must not promise preferential retention.
    #[test]
    fn memory_block_does_not_promise_preferential_retention_yet() {
        assert!(!MEMORY_BLOCK.contains("retained preferentially"));
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
