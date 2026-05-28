pub mod hydration;
pub mod interpreter;
pub mod ir;
pub mod ir_agent;
pub mod ir_interpreter;
pub mod models;
pub mod op;
pub mod provider;
pub mod trace;

pub use hydration::{
    HydrationSource, PassiveHydrationConfig, PassiveSource, SourceCapability, SourceKind,
    SourceParams, SourceRegistry, SourceResult, SEMANTIC_PREFIX, SESSION_STATE_KEY,
    TEMPORAL_PREFIX,
};
pub use interpreter::{run_sequential, EnvPolicy, EvalConfig, ReplayTrace, SeqConfig};
pub use ir::{
    effect_location, program_hash, validate_program, Block, BlockId, Budgets, DynamicPath,
    DynamicPathSegment, EffectId, EffectKind, EffectLocation, EffectSite, EvalPolicy, EvalRequest,
    Expr, Frame, InferPolicy, Instr, Machine, MatchArm, Pattern, Program, ProgramHash, ProgramId,
    PromptRef, Terminator, Var,
};
pub use ir_agent::agent_loop_ir;
pub use ir_interpreter::{
    run_ir_sequential, run_ir_sequential_with_store, run_ir_sequential_with_store_and_replay,
    run_ir_steps, run_ir_steps_with_store_and_replay, InMemoryStore, IrCheckpoint, IrReplayTrace,
    IrStepOutcome,
};
pub use models::{ModelEntry, ModelRegistry, ResolvedModel};
pub use op::{
    agent_loop, emit, eval, get, infer, par, put, ChatMessage, Model, Op, OpF, Prompt, Response,
    ResponseToolCall,
};
pub use provider::{ChatProvider, ProviderClient, ProviderConfig};
pub use trace::{Event, TraceLogger, TraceSummary};
