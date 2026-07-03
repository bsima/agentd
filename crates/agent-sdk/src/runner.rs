//! Running agents: one-shot runs, deterministic replay, and live public
//! event streaming. `Runner` wraps `agent_core::run_agent_loop` — the same
//! loop entry the `agent` CLI drives — with the SDK's provider resolution
//! and trace plumbing.

use crate::agent::Agent;
use crate::error::SdkError;
use agent_core::ir_interpreter::InMemoryStore;
use agent_core::public_trace::{public_event, PublicEvent};
use agent_core::trace::TraceSink;
use agent_core::{
    output_contract_failure, run_agent_loop, AgentLoopOptions, AnthropicConfig, AnthropicProvider,
    ChatMessage, ChatProvider, EvalConfig, Event, GcMode, GcState, GcTiming, IrReplayTrace,
    JsonlTraceSink, MemorySource, Model, ModelRegistry, PassiveHydrationConfig, ProviderClient,
    ProviderConfig, ReplayOnlyProvider, Response, SeqConfig, SourceRegistry, TraceLogger,
};
use anyhow::Result as AnyResult;
use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

/// Context budget assumed when the model registry is bypassed (injected
/// provider) or unavailable; matches the registry's own default.
const DEFAULT_CONTEXT_BUDGET: usize = 200_000;

/// The outcome of a completed run.
#[derive(Clone, Debug)]
pub struct RunResult {
    /// The agent's final response text.
    pub text: String,
    /// When the agent has an output schema: the final response parsed as
    /// JSON, already validated against the schema. `None` when no schema is
    /// set.
    pub output: Option<Value>,
    /// The run id, unique per run; every trace event carries it.
    pub run_id: String,
    /// The JSONL runtime trace recorded for this run — the input to
    /// [`Runner::replay`].
    pub trace_path: PathBuf,
}

/// Live feed of public trace events (docs/TRACE_SCHEMA.md) for one run.
///
/// This is a channel with an async `next()`, not a `futures::Stream`, on
/// purpose: the upcoming Python/TS bindings (t-1308.8) map an async
/// poll-next method directly onto their native async iterators, whereas a
/// poll-based `Stream` needs pinning and combinator glue that neither PyO3
/// nor napi-rs can express cleanly. The channel is unbounded; a slow
/// consumer never backpressures the run.
pub struct EventStream {
    rx: mpsc::UnboundedReceiver<PublicEvent>,
}

impl EventStream {
    /// The next public event, or `None` once the run has finished and the
    /// feed is drained. The final event of a successful run is
    /// `run.completed`.
    pub async fn next(&mut self) -> Option<PublicEvent> {
        self.rx.recv().await
    }
}

/// A run started with [`Runner::start`]: poll events while it executes,
/// then [`RunHandle::wait`] for the result.
pub struct RunHandle {
    run_id: String,
    trace_path: PathBuf,
    events: Option<EventStream>,
    task: tokio::task::JoinHandle<Result<RunResult, SdkError>>,
}

impl RunHandle {
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn trace_path(&self) -> &Path {
        &self.trace_path
    }

    /// Take the public event stream. Yields `Some` exactly once; the stream
    /// is owned by the caller afterwards (dropping it never affects the
    /// run).
    pub fn events(&mut self) -> Option<EventStream> {
        self.events.take()
    }

    /// Wait for the run to finish and return its result.
    pub async fn wait(self) -> Result<RunResult, SdkError> {
        self.task
            .await
            .map_err(|err| SdkError::Run(format!("run task panicked: {err}")))?
    }
}

/// Entry points for executing an [`Agent`].
pub struct Runner;

impl Runner {
    /// One-shot run: send `prompt`, drive the agent loop (tool dispatch,
    /// repairs, budgets) to completion, return the final result.
    pub async fn run(agent: &Agent, prompt: impl Into<String>) -> Result<RunResult, SdkError> {
        let prepared = Prepared::new(agent, None)?;
        execute(agent.clone(), prompt.into(), None, prepared).await
    }

    /// Like [`Runner::run`], but returns immediately with a [`RunHandle`]
    /// whose [`EventStream`] observes the run live.
    pub async fn start(agent: &Agent, prompt: impl Into<String>) -> Result<RunHandle, SdkError> {
        let (tx, rx) = mpsc::unbounded_channel();
        let prepared = Prepared::new(agent, Some(tx))?;
        let run_id = prepared.run_id.clone();
        let trace_path = prepared.trace_path.clone();
        let agent = agent.clone();
        let prompt = prompt.into();
        let task = tokio::spawn(async move { execute(agent, prompt, None, prepared).await });
        Ok(RunHandle {
            run_id,
            trace_path,
            events: Some(EventStream { rx }),
            task,
        })
    }

    /// Deterministic replay of a recorded run: every effect result (model
    /// responses, tool results, shell output, memory reads/writes) is
    /// served from the trace by stable effect id — no provider call and no
    /// tool handler invocation happens. The agent must match the recording
    /// (same model, tool set, and output schema); divergence fails loudly.
    /// The replay itself is traced as a fresh run.
    pub async fn replay(
        trace_path: impl AsRef<Path>,
        agent: &Agent,
        prompt: impl Into<String>,
    ) -> Result<RunResult, SdkError> {
        let replay = IrReplayTrace::load(trace_path.as_ref())
            .await
            .map_err(|err| SdkError::Replay(format!("{err:#}")))?;
        let prepared = Prepared::new(agent, None)?;
        execute(agent.clone(), prompt.into(), Some(replay), prepared).await
    }
}

/// Trace plumbing created before the run starts, so `start` can hand out
/// the run id, trace path, and event stream immediately.
struct Prepared {
    run_id: String,
    trace_path: PathBuf,
    trace: TraceLogger,
}

impl Prepared {
    fn new(
        agent: &Agent,
        events_tx: Option<mpsc::UnboundedSender<PublicEvent>>,
    ) -> Result<Self, SdkError> {
        let run_id = Uuid::new_v4().to_string();
        let trace_dir = agent.trace_dir.clone().unwrap_or_else(default_trace_dir);
        std::fs::create_dir_all(&trace_dir).map_err(|err| {
            SdkError::Trace(format!(
                "creating trace directory {}: {err}",
                trace_dir.display()
            ))
        })?;
        let trace_path = trace_dir.join(format!("{run_id}.jsonl"));
        let mut sinks: Vec<Arc<dyn TraceSink>> =
            vec![Arc::new(JsonlTraceSink::new(trace_path.clone()))];
        if let Some(tx) = events_tx {
            sinks.push(Arc::new(PublicEventSink { tx }));
        }
        let trace = TraceLogger::with_sinks(run_id.clone(), trace_path.clone(), sinks);
        Ok(Self {
            run_id,
            trace_path,
            trace,
        })
    }
}

/// Trace sink that projects runtime events through
/// [`agent_core::public_trace::public_event`] and feeds the survivors into
/// the run's [`EventStream`]. The public projection is the ONLY event
/// vocabulary the SDK exposes; runtime-internal events project to `None`
/// and never reach consumers. Send failures (consumer dropped the stream)
/// are ignored: observation must never fail the run.
struct PublicEventSink {
    tx: mpsc::UnboundedSender<PublicEvent>,
}

#[async_trait]
impl TraceSink for PublicEventSink {
    async fn emit(&self, event: &Event) -> AnyResult<()> {
        if let Some(public) = public_event(event) {
            let _ = self.tx.send(public);
        }
        Ok(())
    }
}

fn default_trace_dir() -> PathBuf {
    dirs::home_dir()
        .map(|home| home.join(".local/share/agent/traces"))
        .unwrap_or_else(|| std::env::temp_dir().join("agent-sdk-traces"))
}

/// Resolve the provider and provider-facing model id, mirroring the CLI's
/// conventions: an injected provider wins; otherwise the model registry
/// (`~/.config/agent/models.yaml`) resolves the alias, with
/// `AGENT_PROVIDER`/`OPENROUTER_BASE_URL` and
/// `AGENT_API_KEY`/`ANTHROPIC_API_KEY`/`OPENROUTER_API_KEY` env fallbacks.
async fn resolve_provider(
    agent: &Agent,
) -> Result<(Arc<dyn ChatProvider>, String, usize), SdkError> {
    if let Some(provider) = &agent.provider {
        return Ok((
            provider.clone(),
            agent.model.clone(),
            DEFAULT_CONTEXT_BUDGET,
        ));
    }
    let registry = ModelRegistry::load_default()
        .await
        .map_err(|err| SdkError::Model(format!("{err:#}")))?;
    let resolved = registry
        .resolve(Some(&agent.model))
        .map_err(|err| SdkError::Model(format!("{err:#}")))?;
    let tag = resolved.provider.as_deref();
    if matches!(
        tag,
        Some("openai-codex" | "codex-oauth" | "claude-code" | "claude-code-oauth")
    ) {
        return Err(SdkError::Model(format!(
            "model {:?} uses OAuth provider {:?}, which agent-sdk does not support yet; \
             run it through the agent CLI or use an API-key model",
            resolved.alias,
            tag.unwrap_or_default()
        )));
    }
    let is_anthropic = tag == Some("anthropic");
    let url = resolved
        .base_url
        .clone()
        .or_else(|| std::env::var("AGENT_PROVIDER").ok())
        .or_else(|| std::env::var("OPENROUTER_BASE_URL").ok())
        .unwrap_or_else(|| {
            if is_anthropic {
                "https://api.anthropic.com/v1".into()
            } else {
                "https://openrouter.ai/api/v1".into()
            }
        });
    let api_key = resolved
        .api_key
        .clone()
        .or_else(|| std::env::var("AGENT_API_KEY").ok())
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .ok_or_else(|| {
            SdkError::Model(
                "missing API key: set AGENT_API_KEY/ANTHROPIC_API_KEY/OPENROUTER_API_KEY \
                 or configure api_key in models.yaml"
                    .into(),
            )
        })?;
    let provider: Arc<dyn ChatProvider> = if is_anthropic {
        Arc::new(AnthropicProvider::new(AnthropicConfig {
            base_url: url,
            api_key,
            model: Model(resolved.api_id.clone()),
            max_tokens: resolved.max_tokens,
        }))
    } else {
        Arc::new(ProviderClient::new(ProviderConfig {
            url,
            api_key,
            model: Model(resolved.api_id.clone()),
        }))
    };
    Ok((provider, resolved.api_id, resolved.context))
}

/// The provider-facing model id for a replay run — no credentials needed:
/// registry resolution is best-effort (the recorded model string must
/// match, and a registry alias records its api_id), falling back to the
/// agent's model string verbatim (the injected-provider case).
async fn replay_model(agent: &Agent) -> String {
    if agent.provider.is_some() {
        return agent.model.clone();
    }
    match ModelRegistry::load_default().await {
        Ok(registry) => registry
            .resolve(Some(&agent.model))
            .map(|resolved| resolved.api_id)
            .unwrap_or_else(|_| agent.model.clone()),
        Err(_) => agent.model.clone(),
    }
}

async fn execute(
    agent: Agent,
    prompt: String,
    replay: Option<IrReplayTrace>,
    prepared: Prepared,
) -> Result<RunResult, SdkError> {
    let Prepared {
        run_id,
        trace_path,
        trace,
    } = prepared;

    let (provider, model, context_budget): (Arc<dyn ChatProvider>, String, usize) =
        if replay.is_some() {
            // Replay never calls a provider: recorded results are served by
            // effect id, and any call reaching the provider is a divergence.
            (
                Arc::new(ReplayOnlyProvider),
                replay_model(&agent).await,
                DEFAULT_CONTEXT_BUDGET,
            )
        } else {
            resolve_provider(&agent).await?
        };

    let mut hydration = SourceRegistry::new();
    if let Some(dir) = &agent.memory_dir {
        hydration = hydration.register_backend(MemorySource::new(dir.clone()));
    }
    let memory_tools = agent.memory_dir.is_some();

    let config = SeqConfig {
        provider,
        hydration,
        tools: agent.tools.clone(),
        passive_hydration: PassiveHydrationConfig::default(),
        trace: trace.clone(),
        eval: EvalConfig {
            cwd: agent.eval_cwd.clone(),
            timeout: agent.eval_timeout,
            env: agent.eval_env.clone(),
            ..EvalConfig::default()
        },
        replay: None,
        trace_full_prompt_ir: false,
        trace_full_payloads: false,
        gc: GcMode::None,
        gc_threshold: 0.85,
        gc_log: false,
        gc_timing: GcTiming::Threshold,
        context_budget,
    };
    let options = AgentLoopOptions {
        memory_tools,
        tool_names: agent.tools.names(),
        output_contract: agent.output_contract.clone(),
    };
    let mut history = Vec::new();
    if let Some(instructions) = &agent.instructions {
        history.push(ChatMessage::system(instructions.clone()));
    }
    history.push(ChatMessage::user(prompt));

    let mut store = InMemoryStore::new();
    let mut gc_state = GcState::default();
    let (value, _machine) = run_agent_loop(
        &config,
        &mut store,
        replay.as_ref(),
        &mut gc_state,
        Model(model),
        history,
        agent.max_turns,
        &options,
        BTreeMap::new(),
    )
    .await
    .map_err(|err| {
        let rendered = format!("{err:#}");
        if replay.is_some() {
            SdkError::Replay(rendered)
        } else {
            SdkError::Run(rendered)
        }
    })?;

    // Exhausted output-schema repairs come back as a typed value (the
    // loop's errors-as-values convention); surface them as the typed SDK
    // error rather than handing back non-conforming output.
    if let Some(failure) = output_contract_failure(&value) {
        return Err(SdkError::OutputContract {
            attempts: failure.attempts,
            errors: failure.errors,
            content: failure.content,
        });
    }
    let response: Response = serde_json::from_value(value)
        .map_err(|err| SdkError::Run(format!("decoding agent loop response: {err}")))?;

    // Close the run in the trace: consumers see the public `run.completed`.
    trace
        .emit(&Event::AgentDone {
            run_id: run_id.clone(),
            timestamp: Utc::now(),
        })
        .await
        .map_err(|err| SdkError::Trace(format!("{err:#}")))?;

    // With a contract set the text already validated; parse it once for the
    // structured result.
    let output = match &agent.output_contract {
        Some(_) => Some(serde_json::from_str(&response.content).map_err(|err| {
            SdkError::Run(format!("validated output failed to parse as JSON: {err}"))
        })?),
        None => None,
    };
    Ok(RunResult {
        text: response.content,
        output,
        run_id,
        trace_path,
    })
}
