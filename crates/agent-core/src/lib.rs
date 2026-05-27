pub mod hydration;
pub mod interpreter;
pub mod models;
pub mod op;
pub mod provider;
pub mod trace;

pub use hydration::{
    HydrationSource, PassiveHydrationConfig, PassiveSource, SourceCapability, SourceKind,
    SourceParams, SourceRegistry, SourceResult, SEMANTIC_PREFIX, SESSION_STATE_KEY,
    TEMPORAL_PREFIX,
};
pub use interpreter::{run_sequential, SeqConfig};
pub use models::{ModelEntry, ModelRegistry, ResolvedModel};
pub use op::{
    agent_loop, emit, eval, get, infer, par, put, ChatMessage, Model, Op, OpF, Prompt, Response,
    ResponseToolCall,
};
pub use provider::{ChatProvider, ProviderClient, ProviderConfig};
pub use trace::{Event, TraceLogger};
