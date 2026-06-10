# PromptIR design

Status: **v1 implemented as a traceable representation** (`agent-core::prompt_ir`).
Migration steps 1–7 from the plan below are done: hydration builds PromptIR,
compiles it before provider calls, and emits hash + section summaries per
`InferCall`; `PromptRef::PromptIr`/`PromptIrVar` exist in AgentIR. PromptIR is
constructed when hydration produces sections — bare prompts skip it. The
optimization passes listed under Non-goals remain future work.

PromptIR is the structured payload for `AgentIR::Infer`.

AgentIR answers when effects happen. PromptIR answers what context the model saw, where it came from, how it was budgeted, and how it should be traced.

The prior Haskell implementation lives at `~/omni/live/Omni/Agent/Prompt/IR.hs`. The useful parts to port are the section/provenance/budget model. The graph-expression part should not be ported directly because AgentIR now owns control flow, effect scheduling, and parallelism.

## Goal

The stable inference path should become:

```text
ContextRequest
        ↓ hydrate from registered sources
PromptIR
        ↓ validate / budget / normalize
PromptIR snapshot + hash in trace
        ↓ compile
Vec<ChatMessage> + tool specs
        ↓
AgentIR Infer
```

The first version should preserve current prompt behavior. PromptIR should initially be an inspectable representation and trace format, not an optimization engine.

## Core model

A prompt is not an append-only string. It is a set of sourced sections composed into a provider prompt.

```rust
pub struct PromptIR {
    pub id: PromptId,
    pub sections: Vec<Section>,
    pub tools: Vec<ToolDef>,
    pub observation: Option<Observation>,
    pub meta: PromptMeta,
}

pub struct Section {
    pub id: SectionId,
    pub label: String,
    pub source: SectionSource,
    pub role: SectionRole,
    pub content: String,
    pub tokens: TokenEstimate,
    pub priority: Priority,
    pub composition: CompositionMode,
    pub relevance: Option<f32>,
    pub recency: Option<DateTime<Utc>>,
    pub hash: ContentHash,
    pub metadata: serde_json::Value,
}
```

The important invariant is that every piece of model-visible context has a section ID, source, priority, and hash.

## Section source

PromptIR should preserve the context model from the README: agents build context in two ways, temporal lookup and semantic lookup. Everything else is a backend/corpus plugged into one or both modes.

```rust
pub enum RetrievalMode {
    Temporal,
    Semantic,
}

pub enum RetrievalTiming {
    Passive,
    Active,
}

pub struct SectionSource {
    pub origin: SectionOrigin,
    pub timing: RetrievalTiming,
    pub metadata: serde_json::Value,
}

pub enum SectionOrigin {
    Static { name: String },
    Retrieval {
        backend: String,
        mode: RetrievalMode,
        query: Option<String>,
        key: Option<String>,
        score: Option<f32>,
    },
    State { key: String },
    User,
    ToolResult,
}
```

A backend can support temporal lookup, semantic lookup, or both. For example, a workspace backend can expose recent edits temporally and file/symbol search semantically. A memory backend can expose chronological events temporally and remembered facts semantically.

Workspace, knowledge, docs, issue trackers, and long-term memory should not become top-level source kinds. They are backend names or metadata on `SectionOrigin::Retrieval`.

`SourceResult -> Section` should be the main adapter between current hydration and PromptIR. The source registry remains the backend boundary.

## Section role and composition

Provider APIs still need chat messages. PromptIR should not lose that structure.

```rust
pub enum SectionRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Context,
}

pub enum CompositionMode {
    Hierarchical, // system/developer hyperprior
    Constraint,   // hard instruction or policy text
    Additive,     // ordinary retrievable context
    Contextual,   // current observation / recent conversation
}

pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}
```

Compilation order should be deterministic:

```text
Hierarchical -> Constraint -> Additive -> Contextual -> current observation
```

Within each group, preserve insertion order unless a later budget pass explicitly reorders by priority/relevance.

## Budget and strategy

PromptIR should include budget inputs and budget decisions in metadata.

```rust
pub struct TokenBudget {
    pub total: usize,
    pub reserve_ratio: f32,
    pub allocation: BudgetAllocation,
}

pub enum BudgetAllocation {
    FixedRatios { system: f32, context: f32, observation: f32 },
    RelevanceWeighted,
    InformationWeighted,
}

pub struct ContextStrategy {
    pub temporal_window: usize,
    pub semantic_limit: usize,
    pub semantic_threshold: f32,
    pub enabled_backends: Vec<String>,
}

pub struct ContextRequest {
    pub observation: String,
    pub goal: Option<String>,
    pub strategy: ContextStrategy,
    pub budget: TokenBudget,
}
```

For v1, token counts can be approximate. The point is to make budget choices visible and stable in traces.

`enabled_backends` selects corpora/backends, not retrieval semantics. The retrieval semantics remain temporal and semantic. If empty, the interpreter can use its configured default backends.

## Trace shape

Every `InferCall` should be associated with a PromptIR hash and section summaries.

Trace metadata should include:

```text
prompt_id
prompt_hash
total_tokens_estimate
budget
strategy
sections: [
  section_id,
  label,
  source,
  role,
  priority,
  composition,
  tokens,
  hash,
  content_preview
]
```

The trace should not need to store full section content by default. It should store hashes and previews. Full snapshots should be available for debugging through a flag/config setting. The CLI setting is `--trace-full-prompt-ir` / `AGENT_TRACE_FULL_PROMPT_IR`.

## Compilation

PromptIR compiles to the existing provider shape:

```rust
pub fn compile_prompt_ir(ir: &PromptIR) -> Result<Vec<ChatMessage>>;
```

Initial compilation should be boring and compatibility-preserving:

- system/developer sections become system messages or the current provider equivalent
- user/assistant/tool sections preserve their chat roles
- context sections are rendered as labeled markdown blocks
- observation is last
- tools compile separately to provider tool specs

The acceptance bar for v1 is semantic equivalence with current hydration, not prompt optimization.

## Relationship to AgentIR

AgentIR should refer to PromptIR as the payload for inference:

```rust
Instr::Infer {
    out,
    model,
    prompt: PromptRef,
    policy,
}

pub enum PromptRef {
    Inline(Vec<ChatMessage>),       // compatibility
    Var(Var),                       // compatibility
    PromptIr(PromptIR),             // stable path
    PromptIrVar(Var),               // dynamic path
}
```

PromptIR should not introduce a second control-flow graph. The graph/scheduling ideas from the Haskell PromptIR module belong in AgentIR now.

## Migration plan

1. Add `agent-core::prompt_ir` with serializable types and round-trip tests.
2. Add `SourceResult -> Section` adapters.
3. Add `compile_prompt_ir` to produce current `Vec<ChatMessage>` prompts.
4. Change passive hydration to build PromptIR, then compile it before provider calls.
5. Emit PromptIR metadata before every `InferCall`.
6. Add `PromptRef::PromptIr` / `PromptIrVar` once the v1 compiler is stable.
7. Preserve `PromptRef::Inline` and `PromptRef::Var` as compatibility inputs.

## Non-goals for v1

- automatic compression
- embedding-based optimization
- prompt graph scheduling
- learned section ranking
- provider-specific prompt tuning

Those should come after PromptIR is stable as a traceable context representation.

## Acceptance

- Existing release evals pass with PromptIR-enabled hydration.
- Flat compiled prompts are semantically equivalent to current prompts.
- Every hydrated context section has source provenance in the trace.
- PromptIR provenance preserves temporal-vs-semantic lookup mode and passive-vs-active timing.
- Every `InferCall` can be linked to a PromptIR hash and section summaries.
- The source registry remains the swappable hydration backend.
