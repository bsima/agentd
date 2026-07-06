//! Test and example utilities: run agents without credentials or a live
//! provider.

use agent_core::provider::ToolSpec;
use agent_core::{ChatMessage, ChatProvider, FinishReason, Model, Response, ToolCall};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;

/// A provider that replays a fixed script of responses, one per model
/// call, in order — the same pattern agent-core's own loop tests use.
/// Inject it with [`crate::AgentBuilder::provider`] to run agents (and the
/// crate's examples) without credentials.
///
/// ```
/// use agent_sdk::testing::ScriptedProvider;
/// let provider = ScriptedProvider::new()
///     .tool_call("get_weather", serde_json::json!({ "city": "sf" }))
///     .text("It's sunny in SF.");
/// ```
#[derive(Default)]
pub struct ScriptedProvider {
    responses: Mutex<VecDeque<Response>>,
    scripted: usize,
}

impl ScriptedProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Script a final text response (no tool calls: ends the turn).
    pub fn text(self, content: impl Into<String>) -> Self {
        self.push(response(content.into(), Vec::new()))
    }

    /// Script a response that calls one tool. Tool-call ids are minted as
    /// `call-1`, `call-2`, ... in script order.
    pub fn tool_call(self, tool: impl Into<String>, arguments: Value) -> Self {
        let id = format!("call-{}", self.scripted + 1);
        self.push(response(
            String::new(),
            vec![ToolCall::new(id, tool, arguments)],
        ))
    }

    fn push(mut self, response: Response) -> Self {
        self.responses.get_mut().unwrap().push_back(response);
        self.scripted += 1;
        self
    }
}

fn response(content: String, tool_calls: Vec<ToolCall>) -> Response {
    Response {
        content,
        tool_calls,
        finish_reason: Some(FinishReason::Stop),
        input_tokens: 0,
        output_tokens: 1,
        total_tokens: 1,
        cached_input_tokens: None,
        cost_micro_usd: None,
        pricing: None,
        metadata: Default::default(),
    }
}

/// Build a replay-trace fixture that drives an `agent --session
/// --replay-trace` child without credentials — the Rust twin of the shell
/// recipe in `evals/session.sh`. One scripted text response per turn, each
/// recorded at the stable effect location of that turn's entry `Infer`
/// (program hash + site + visit ordinal), which is what replay matches on.
///
/// The fixture assumes the child's default loop program: `model` must equal
/// the model string the child resolves (a raw model id when no registry
/// entry exists), and the agent must have no memory dir (the memory tools
/// change the program hash).
///
/// ```no_run
/// # async fn example() -> anyhow::Result<()> {
/// use agent_sdk::testing::SessionReplayFixture;
/// SessionReplayFixture::new("test-model")
///     .turn("first reply")
///     .turn("second reply")
///     .write("/tmp/replay.jsonl")
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct SessionReplayFixture {
    model: String,
    turns: Vec<String>,
}

impl SessionReplayFixture {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            turns: Vec::new(),
        }
    }

    /// Script the final text response of the next session turn.
    pub fn turn(mut self, response: impl Into<String>) -> Self {
        self.turns.push(response.into());
        self
    }

    /// The fixture as runtime trace events (one Hydration pair plus an
    /// `InferCall`/`InferResult` pair per turn).
    pub fn events(&self) -> Result<Vec<agent_core::Event>> {
        // Mirrors `agent ir-effect`: the entry Infer of the built-in loop
        // lives at block 0, instruction 0; its Nth session turn is visit N
        // on the root control path. Prompt content and turn budgets are
        // data, not program, so they do not perturb the hash.
        let machine = agent_core::agent_loop_ir(Model(self.model.clone()), Vec::new(), 16);
        let hash = agent_core::program_hash(&machine.program)?;
        let site = agent_core::EffectSite {
            block: agent_core::BlockId(0),
            instruction_index: 0,
        };
        let run_id = "agent-sdk-fixture".to_owned();
        let mut events = Vec::new();
        for (visit, content) in self.turns.iter().enumerate() {
            let op_base = (visit as u64) * 2;
            let location = agent_core::effect_location(
                hash.clone(),
                agent_core::EffectKind::Infer,
                site,
                agent_core::DynamicPath::at_entry(visit as u64),
            )?;
            events.push(agent_core::Event::HydrationStart {
                run_id: run_id.clone(),
                op_id: op_base + 1,
                parent_op_id: None,
                sources: vec!["TemporalHistory".into(), "SessionContext".into()],
                max_bytes: None,
                timestamp: Utc::now(),
            });
            events.push(agent_core::Event::HydrationEnd {
                run_id: run_id.clone(),
                op_id: op_base + 1,
                parent_op_id: None,
                section_count: 0,
                total_bytes: 0,
                timestamp: Utc::now(),
            });
            events.push(agent_core::Event::InferCall {
                run_id: run_id.clone(),
                op_id: op_base + 2,
                parent_op_id: None,
                model: self.model.clone(),
                prompt: Some(Vec::new()),
                prompt_preview: content.clone(),
                effect: Some(Box::new(location)),
                timestamp: Utc::now(),
            });
            events.push(agent_core::Event::InferResult {
                run_id: run_id.clone(),
                op_id: op_base + 2,
                parent_op_id: None,
                response: Some(response(content.clone(), Vec::new())),
                response_preview: content.clone(),
                input_tokens: 0,
                output_tokens: 1,
                total_tokens: 1,
                cached_input_tokens: None,
                cost_micro_usd: None,
                pricing: None,
                duration_ms: 0,
                timestamp: Utc::now(),
            });
        }
        Ok(events)
    }

    /// Write the fixture as trace JSONL at `path` (parent directories are
    /// created), ready for `--replay-trace` /
    /// [`crate::SessionOptions::replay_trace`].
    pub async fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut lines = String::new();
        for event in self.events()? {
            lines.push_str(&serde_json::to_string(&event)?);
            lines.push('\n');
        }
        tokio::fs::write(path, lines).await?;
        Ok(())
    }
}

#[async_trait]
impl ChatProvider for ScriptedProvider {
    async fn chat(
        &self,
        _model: &Model,
        _tools: &[ToolSpec],
        _messages: &[ChatMessage],
    ) -> Result<Response> {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("scripted provider exhausted: no more responses queued"))
    }
}
