//! Conformance tests for the Get/Put key-namespace contract
//! (docs/STATE_KEYS.md). The guaranteed namespaces must behave the same under
//! the Op and IR runtimes; where the runtimes deliberately diverge, the
//! divergence is pinned here so it cannot drift silently.

use agent_core::{Block, BlockId, Expr, Instr, Program, ProgramId, Terminator, Var};
use agent_core::{
    ChatMessage, ChatProvider, EvalConfig, GcMode, HydrationSource, InMemoryStore, Machine, Model,
    PassiveHydrationConfig, Response, SeqConfig, SourceCapability, SourceKind, SourceParams,
    SourceRegistry, SourceResult, TraceLogger,
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

struct NoopProvider;

#[async_trait]
impl ChatProvider for NoopProvider {
    async fn chat(
        &self,
        _model: &Model,
        _tools: &[agent_core::provider::ToolSpec],
        _messages: &[ChatMessage],
    ) -> Result<Response> {
        Err(anyhow!("state-key conformance tests never call Infer"))
    }
}

struct SemanticSource;

#[async_trait]
impl HydrationSource for SemanticSource {
    fn name(&self) -> &str {
        "semantic-store"
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Semantic
    }

    fn capabilities(&self) -> SourceCapability {
        SourceCapability::QUERY
    }

    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
        Ok(SourceResult {
            source: self.name().into(),
            kind: self.kind(),
            content: format!("result for {}", params.query.unwrap_or_default()),
            metadata: json!({}),
        })
    }
}

fn config(checkpoint_path: Option<PathBuf>) -> SeqConfig {
    let path = std::env::temp_dir().join(format!("state-keys-{}.jsonl", Uuid::new_v4()));
    SeqConfig {
        provider: Arc::new(NoopProvider),
        hydration: SourceRegistry::new().register(SemanticSource),
        passive_hydration: PassiveHydrationConfig::default(),
        checkpoint_path,
        trace: TraceLogger::new(Uuid::new_v4().to_string(), path),
        eval: EvalConfig::default(),
        replay: None,
        trace_full_prompt_ir: false,
        trace_full_payloads: false,
        gc: GcMode::None,
        gc_threshold: 0.85,
        gc_log: false,
        context_budget: 200_000,
    }
}

/// Build a one-block IR machine that runs `instructions` and returns `value`.
fn ir_machine(instructions: Vec<Instr>, value: Expr) -> Machine {
    let mut blocks = BTreeMap::new();
    blocks.insert(
        BlockId(0),
        Block {
            params: vec![],
            instructions,
            terminator: Terminator::Return { value },
        },
    );
    Machine {
        program: Program {
            id: ProgramId("state-keys".into()),
            entry: BlockId(0),
            blocks,
        },
        block: BlockId(0),
        pc: 0,
        env: BTreeMap::new(),
        effect_visits: BTreeMap::new(),
        continuation_stack: vec![],
        budgets: Default::default(),
    }
}

fn ir_put_get(key: &str, value: Value) -> Machine {
    ir_machine(
        vec![
            Instr::Put {
                key: Expr::Value(Value::String(key.into())),
                value: Expr::Value(value),
            },
            Instr::Get {
                out: Var("out".into()),
                key: Expr::Value(Value::String(key.into())),
            },
        ],
        Expr::Var(Var("out".into())),
    )
}

fn ir_get(key: &str) -> Machine {
    ir_machine(
        vec![Instr::Get {
            out: Var("out".into()),
            key: Expr::Value(Value::String(key.into())),
        }],
        Expr::Var(Var("out".into())),
    )
}

// `session:state` — durable checkpoint storage. Put writes the checkpoint
// file; Get reads it back. Guaranteed in both runtimes.
#[tokio::test]
async fn session_state_round_trips_in_both_runtimes() -> Result<()> {
    let value = json!({ "checkpoint": 7 });

    let path = std::env::temp_dir().join(format!("state-keys-op-{}.json", Uuid::new_v4()));
    let op_config = config(Some(path));
    let program = agent_core::put::<()>("session:state", value.clone())
        .and_then(|_| agent_core::get("session:state"));
    let (observed, _) = agent_core::run_sequential(&op_config, (), program).await?;
    assert_eq!(observed, value);

    let path = std::env::temp_dir().join(format!("state-keys-ir-{}.json", Uuid::new_v4()));
    let ir_config = config(Some(path));
    let (observed, _) =
        agent_core::run_ir_sequential(&ir_config, ir_put_get("session:state", value.clone()))
            .await?;
    assert_eq!(observed, value);
    Ok(())
}

// `temporal:*` — Put then Get returns the last put value in both runtimes
// (the Op runtime backs it with interpreter state, the IR runtime with the
// session-local store; the observable contract is the same).
#[tokio::test]
async fn temporal_put_then_get_returns_last_value_in_both_runtimes() -> Result<()> {
    let op_config = config(None);
    let program = agent_core::put::<i64>("temporal:history", json!(42))
        .and_then(|_| agent_core::get("temporal:history"));
    let (observed, _) = agent_core::run_sequential(&op_config, 0_i64, program).await?;
    assert_eq!(observed, json!(42));

    let ir_config = config(None);
    let (observed, _) =
        agent_core::run_ir_sequential(&ir_config, ir_put_get("temporal:history", json!(42)))
            .await?;
    assert_eq!(observed, json!(42));
    Ok(())
}

// `semantic:<query>` — Get dispatches the query to QUERY-capable hydration
// sources in both runtimes.
#[tokio::test]
async fn semantic_get_queries_registered_sources_in_both_runtimes() -> Result<()> {
    let op_config = config(None);
    let (observed, _) =
        agent_core::run_sequential(&op_config, (), agent_core::get::<()>("semantic:topic")).await?;
    assert_eq!(observed[0]["content"], json!("result for topic"));

    let ir_config = config(None);
    let (observed, _) = agent_core::run_ir_sequential(&ir_config, ir_get("semantic:topic")).await?;
    assert_eq!(observed[0]["content"], json!("result for topic"));
    Ok(())
}

// Keys outside the guaranteed namespaces: the Op runtime rejects them (typed
// state, nowhere to put arbitrary keys); the IR runtime treats them as
// session-local KV. This is a *pinned divergence* — if either side changes,
// update docs/STATE_KEYS.md.
#[tokio::test]
async fn unknown_keys_error_in_op_and_are_session_local_kv_in_ir() -> Result<()> {
    let op_config = config(None);
    let err = agent_core::run_sequential(&op_config, (), agent_core::get::<()>("custom:key"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("unknown Get key"), "got: {err}");

    let ir_config = config(None);
    let mut store = InMemoryStore::new();
    let (observed, _) = agent_core::run_ir_sequential_with_store(
        &ir_config,
        ir_put_get("custom:key", json!("local")),
        &mut store,
    )
    .await?;
    assert_eq!(observed, json!("local"));
    assert_eq!(store.get_local("custom:key"), json!("local"));

    // An unknown key that was never Put reads as null in the IR runtime.
    let ir_config = config(None);
    let (observed, _) =
        agent_core::run_ir_sequential(&ir_config, ir_get("custom:never-put")).await?;
    assert_eq!(observed, Value::Null);
    Ok(())
}
