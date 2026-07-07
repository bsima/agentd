pub mod anthropic;
pub mod approval;
pub mod chat_history;
pub mod cost;
pub mod gc;
pub mod hydration;
pub mod interpreter;
pub mod ir;
pub mod ir_agent;
pub mod ir_interpreter;
pub mod ir_normalize;
pub mod memory;
pub mod models;
pub mod op;
pub mod output_contract;
pub mod prompt_ir;
pub mod provider;
pub mod public_trace;
pub mod temporal;
pub mod tool;
pub mod trace;

pub use anthropic::{AnthropicConfig, AnthropicProvider};
pub use approval::{
    denial_value, is_denial_value, pending_id_for, ApprovalConfig, ApprovalDecision,
    ApprovalHookFn, ApprovalKind, ApprovalRequest, ApprovalResolution, ApprovalStore,
    PendingEffectRecord, PendingStatus,
};
pub use chat_history::ChatHistory;
pub use cost::{format_micro_usd, Pricing, PricingTable, RunUsage};
pub use gc::{
    estimate_tokens, truncate_oversized_message, ContextGc, FrameId, FrameStatus, GcMode, GcState,
    GcTiming, LifecycleState, MarkSweepGc, MsgId, RingGc, StackFrameGc,
};
pub use hydration::{
    HydrationSink, HydrationSource, PassiveHydrationConfig, PassiveSource, Provenance, SinkId,
    SinkItem, SinkWritePolicy, SourceCapability, SourceKind, SourceParams, SourceRegistry,
    SourceResult,
};
pub use interpreter::{run_sequential, EnvPolicy, EvalConfig, ReplayTrace, SeqConfig};
pub use ir::{
    effect_location, program_hash, validate_program, Block, BlockId, Budgets, ControlPath,
    DynamicPath, EffectErrorMode, EffectId, EffectKind, EffectLocation, EffectSite, EvalPolicy,
    EvalRequest, Expr, Frame, InferPolicy, Instr, Machine, MatchArm, Pattern, Program, ProgramHash,
    ProgramId, PromptRef, RetrievePolicy, StoreOp, StorePolicy, Terminator, ToolPolicy, Var,
};
pub use ir_agent::{
    agent_loop_ir, agent_loop_ir_with_options, agent_loop_ir_with_policies,
    agent_loop_ir_with_tools, resume_agent_loop_outcome, run_agent_loop, run_agent_loop_outcome,
    AgentLoopOptions, AgentLoopOutcome,
};
pub use ir_interpreter::{
    run_ir_sequential, run_ir_sequential_with_gc, run_ir_sequential_with_store,
    run_ir_sequential_with_store_and_replay, run_ir_steps, run_ir_steps_with_gc,
    run_ir_steps_with_store_and_replay, InMemoryStore, IrCheckpoint, IrReplayTrace, IrStepOutcome,
    IrStore,
};
pub use ir_normalize::{normalize_program, validate_strict_ssa_program};
pub use memory::MemorySource;
pub use models::{ModelEntry, ModelRegistry, PricingEntry, ResolvedModel};
pub use op::{
    agent_loop, close_pending_tool_calls, emit, eval, eval_argv, has_pending_tool_calls, infer,
    par, repair_trailing_pending_tool_calls, ChatMessage, EvalSpec, FinishReason, Model, Op, OpF,
    Prompt, Response, ToolCall,
};
pub use output_contract::{
    output_contract_failure, validate as validate_output, OutputContract, OutputContractFailure,
    DEFAULT_MAX_REPAIRS, OUTPUT_CONTRACT_ERROR, OUTPUT_CONTRACT_EVENT,
    OUTPUT_VALIDATION_FAILED_EVENT,
};
pub use prompt_ir::{
    collect_prompt_ir_sections, compile_prompt_ir, BudgetAllocation, CompositionMode, ContentHash,
    ContextRequest, ContextStrategy, Observation, Priority, PromptIR, PromptIRTrace, PromptId,
    RetrievalMode, RetrievalTiming, Section, SectionId, SectionOrigin, SectionRole, SectionSource,
    SectionSummary, TokenBudget, TokenEstimate, ToolDef,
};
pub use provider::{
    is_context_overflow_anyhow, is_context_overflow_message, ChatProvider, ContextOverflowError,
    ProviderClient, ProviderConfig, ReplayOnlyProvider,
};
pub use public_trace::{
    public_event, PublicDynamicPath, PublicEffect, PublicEffectSite, PublicEvent, PublicStatus,
    PUBLIC_SCHEMA_VERSION,
};
pub use temporal::TemporalSource;
pub use tool::{NativeTool, ToolHandler, ToolRegistry, RESERVED_TOOL_NAMES};
pub use trace::{
    AgentIdGenerator, Event, JsonlTraceSink, OtelTraceSink, TraceContextEnv, TraceLogger,
    TraceSink, TraceSummary,
};
