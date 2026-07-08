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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::program_hash;
    use crate::ir_agent::agent_loop_ir_with_options;
    use crate::op::Model;

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
