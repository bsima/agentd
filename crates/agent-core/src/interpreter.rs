use crate::hydration::{
    PassiveHydrationConfig, PassiveSource, SourceParams, SourceRegistry, SEMANTIC_PREFIX,
    SESSION_STATE_KEY, TEMPORAL_PREFIX,
};
use crate::op::{ChatMessage, Op, OpF, Prompt};
use crate::provider::{ChatProvider, ToolFunctionSpec, ToolSpec};
use crate::trace::{Event, TraceLogger};
use anyhow::{anyhow, Result};
use async_recursion::async_recursion;
use chrono::Utc;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Command;

pub struct SeqConfig {
    pub provider: Arc<dyn ChatProvider>,
    pub hydration: SourceRegistry,
    pub passive_hydration: PassiveHydrationConfig,
    pub checkpoint_path: Option<PathBuf>,
    pub trace: TraceLogger,
}

impl SeqConfig {
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        vec![ToolSpec {
            kind: "function".into(),
            function: ToolFunctionSpec {
                name: "shell".into(),
                description: "Execute a command string using the SHELL environment variable."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "command": { "type": "string" } },
                    "required": ["command"]
                }),
            },
        }]
    }
}

#[async_recursion]
pub async fn run_sequential<S, A>(config: &SeqConfig, state: S, op: Op<S, A>) -> Result<(A, S)>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    A: Send + 'static,
{
    match *op.0 {
        OpF::Pure(value) => Ok((value, state)),
        OpF::Infer {
            model,
            prompt,
            next,
        } => {
            let prompt = hydrate_infer_prompt(config, &state, prompt).await?;
            config
                .trace
                .emit(&Event::InferStart {
                    run_id: config.trace.run_id().into(),
                    model: model.0.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            let response = config
                .provider
                .chat(&model, &config.tool_specs(), &prompt)
                .await?;
            config
                .trace
                .emit(&Event::InferEnd {
                    run_id: config.trace.run_id().into(),
                    tokens: response.tokens,
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential(config, state, next(response)).await
        }
        OpF::Eval { command, next } => {
            config
                .trace
                .emit(&Event::EvalCall {
                    run_id: config.trace.run_id().into(),
                    command: command.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            let shell = std::env::var("SHELL").map_err(|_| {
                anyhow!("SHELL is not set; set it to the shell used for command execution")
            })?;
            let output = Command::new(shell).arg("-c").arg(&command).output().await?;
            let result = serde_json::json!({
                "status": output.status.code(),
                "stdout": String::from_utf8_lossy(&output.stdout),
                "stderr": String::from_utf8_lossy(&output.stderr)
            });
            config
                .trace
                .emit(&Event::EvalResult {
                    run_id: config.trace.run_id().into(),
                    command,
                    result: result.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential(config, state, next(result)).await
        }
        OpF::Get { key, next } => {
            let value = dispatch_get(config, &state, &key).await?;
            run_sequential(config, state, next(value)).await
        }
        OpF::Put { key, value, next } => {
            let state = dispatch_put(config, state, &key, value).await?;
            run_sequential(config, state, next).await
        }
        OpF::Emit { event, next } => {
            config.trace.emit(&event).await?;
            run_sequential(config, state, next).await
        }
        OpF::Par { ops, next } => {
            let mut values = Vec::with_capacity(ops.len());
            let mut current_state = state;
            for op in ops {
                let (value, new_state) = run_sequential(config, current_state, op).await?;
                values.push(value);
                current_state = new_state;
            }
            run_sequential(config, current_state, next(values)).await
        }
    }
}

async fn hydrate_infer_prompt<S>(
    config: &SeqConfig,
    state: &S,
    mut prompt: Prompt,
) -> Result<Prompt>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    if config.passive_hydration.is_empty() {
        return Ok(prompt);
    }

    let mut sections = Vec::new();
    for source in &config.passive_hydration.sources {
        match source {
            PassiveSource::TemporalHistory => {
                let value = serde_json::to_value(state)?;
                if let Value::Array(messages) = value {
                    for value in messages {
                        if let Ok(message) = serde_json::from_value::<ChatMessage>(value) {
                            if !prompt.contains(&message) {
                                prompt.push(message);
                            }
                        }
                    }
                } else if value != Value::Null {
                    sections.push(format!("## temporal history\n{}", value));
                }
            }
            PassiveSource::SessionContext => {
                let params = SourceParams {
                    query: None,
                    max_bytes: config.passive_hydration.max_bytes,
                };
                for result in config.hydration.retrieve_session_context(params).await? {
                    sections.push(format!(
                        "## {} ({:?})\n{}",
                        result.source, result.kind, result.content
                    ));
                }
            }
        }
    }

    if !sections.is_empty() {
        inject_context_sections(&mut prompt, sections.join("\n\n"));
    }
    Ok(prompt)
}

fn inject_context_sections(prompt: &mut Prompt, context: String) {
    let block = format!("Hydrated context:\n\n{context}");
    if let Some(system) = prompt.iter_mut().find(|message| message.role == "system") {
        match &mut system.content {
            Some(content) if !content.is_empty() => {
                content.push_str("\n\n");
                content.push_str(&block);
            }
            _ => system.content = Some(block),
        }
    } else {
        prompt.insert(0, ChatMessage::system(block));
    }
}

async fn dispatch_get<S>(config: &SeqConfig, state: &S, key: &str) -> Result<Value>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    if key.starts_with(TEMPORAL_PREFIX) {
        return serde_json::to_value(state).map_err(Into::into);
    }
    if key.starts_with(SEMANTIC_PREFIX) {
        let query = key.trim_start_matches(SEMANTIC_PREFIX);
        let results = config
            .hydration
            .retrieve_query(SourceParams::new(query))
            .await?;
        return serde_json::to_value(results).map_err(Into::into);
    }
    if key == SESSION_STATE_KEY {
        let Some(path) = &config.checkpoint_path else {
            return Ok(Value::Null);
        };
        match tokio::fs::read_to_string(path).await {
            Ok(content) => serde_json::from_str(&content).map_err(Into::into),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
            Err(err) => Err(err.into()),
        }
    } else {
        Err(anyhow!("unknown Get key: {key}"))
    }
}

async fn dispatch_put<S>(config: &SeqConfig, state: S, key: &str, value: Value) -> Result<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    if key.starts_with(TEMPORAL_PREFIX) {
        return serde_json::from_value(value).map_err(Into::into);
    }
    if key == SESSION_STATE_KEY {
        let Some(path) = &config.checkpoint_path else {
            return Ok(state);
        };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, serde_json::to_vec_pretty(&value)?).await?;
        Ok(state)
    } else {
        Err(anyhow!("unknown Put key: {key}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::{HydrationSource, SourceCapability, SourceKind, SourceResult};
    use crate::op::{agent_loop, infer, Model, Response, ResponseToolCall};
    use crate::provider::ToolSpec;
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    struct MockProvider {
        responses: Mutex<Vec<Response>>,
        prompts: Mutex<Vec<Prompt>>,
    }

    impl MockProvider {
        fn new(mut responses: Vec<Response>) -> Self {
            responses.reverse();
            Self {
                responses: Mutex::new(responses),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn prompt_count(&self) -> usize {
            self.prompts.lock().unwrap().len()
        }

        fn prompts(&self) -> Vec<Prompt> {
            self.prompts.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        async fn chat(
            &self,
            _model: &Model,
            _tools: &[ToolSpec],
            messages: &[ChatMessage],
        ) -> Result<Response> {
            self.prompts.lock().unwrap().push(messages.to_vec());
            self.responses
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow!("mock provider exhausted"))
        }
    }

    fn response(content: &str, tool_calls: Vec<ResponseToolCall>) -> Response {
        Response {
            content: content.into(),
            tool_calls,
            tokens: 7,
        }
    }

    fn tool_call(id: &str, name: &str, arguments: Value) -> ResponseToolCall {
        ResponseToolCall::new(id, name, arguments)
    }

    fn test_trace() -> TraceLogger {
        let path = std::env::temp_dir().join(format!("agent-core-test-{}.jsonl", Uuid::new_v4()));
        TraceLogger::new(Uuid::new_v4().to_string(), path)
    }

    #[tokio::test]
    async fn agent_loop_executes_tool_and_feeds_result_back() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![tool_call("call-1", "echo", json!({ "value": "hello" }))],
            ),
            response("done", vec![]),
        ]));
        let config = SeqConfig {
            provider: provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
        };
        let prompt = vec![ChatMessage::user("use echo")];

        let (result, state) = run_sequential(
            &config,
            prompt.clone(),
            agent_loop(Model("mock".into()), prompt, 4),
        )
        .await?;

        assert_eq!(result.content, "done");
        assert_eq!(provider.prompt_count(), 2);
        assert!(state.iter().any(|msg| msg.role == "tool"));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_supports_multiple_tool_turns() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response("", vec![tool_call("call-1", "echo", json!({ "step": 1 }))]),
            response("", vec![tool_call("call-2", "echo", json!({ "step": 2 }))]),
            response("finished", vec![]),
        ]));
        let config = SeqConfig {
            provider: provider.clone(),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
        };
        let prompt = vec![ChatMessage::user("do two steps")];

        let (result, state) = run_sequential(
            &config,
            prompt.clone(),
            agent_loop(Model("mock".into()), prompt, 4),
        )
        .await?;

        assert_eq!(result.content, "finished");
        assert_eq!(provider.prompt_count(), 3);
        assert_eq!(state.iter().filter(|msg| msg.role == "tool").count(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn get_put_and_par_are_interpreted_sequentially() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
        };
        let program = crate::op::put("temporal:history", json!(1))
            .and_then(|_| {
                crate::op::par(vec![
                    crate::op::put("temporal:history", json!(2)),
                    crate::op::put("temporal:history", json!(3)),
                ])
            })
            .and_then(|_| crate::op::get("temporal:history"));

        let (result, state) = run_sequential(&config, 0, program).await?;

        assert_eq!(result, json!(3));
        assert_eq!(state, 3);
        Ok(())
    }
    struct StaticSource {
        name: &'static str,
        kind: SourceKind,
        capabilities: SourceCapability,
        content: &'static str,
        queries: Arc<Mutex<Vec<Option<String>>>>,
    }

    #[async_trait]
    impl HydrationSource for StaticSource {
        fn name(&self) -> &str {
            self.name
        }

        fn kind(&self) -> SourceKind {
            self.kind
        }

        fn capabilities(&self) -> SourceCapability {
            self.capabilities
        }

        async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
            self.queries.lock().unwrap().push(params.query.clone());
            Ok(SourceResult {
                source: self.name.into(),
                kind: self.kind,
                content: self.content.into(),
                metadata: json!({}),
            })
        }
    }

    #[tokio::test]
    async fn session_state_get_reads_checkpoint_json() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let path = std::env::temp_dir().join(format!("agent-core-session-{}.json", Uuid::new_v4()));
        tokio::fs::write(&path, serde_json::to_vec(&json!({ "checkpoint": 7 }))?).await?;
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: Some(path),
            trace: test_trace(),
        };

        let (value, state) = run_sequential(
            &config,
            vec![ChatMessage::user("state")],
            crate::op::get("session:state"),
        )
        .await?;

        assert_eq!(value, json!({ "checkpoint": 7 }));
        assert_eq!(state, vec![ChatMessage::user("state")]);
        Ok(())
    }

    #[tokio::test]
    async fn session_state_put_writes_checkpoint_json() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let path = std::env::temp_dir().join(format!("agent-core-session-{}.json", Uuid::new_v4()));
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: Some(path.clone()),
            trace: test_trace(),
        };

        let (_, state) = run_sequential(
            &config,
            vec![ChatMessage::user("state")],
            crate::op::put("session:state", json!({ "checkpoint": 8 })),
        )
        .await?;
        let content: Value = serde_json::from_slice(&tokio::fs::read(path).await?)?;

        assert_eq!(content, json!({ "checkpoint": 8 }));
        assert_eq!(state, vec![ChatMessage::user("state")]);
        Ok(())
    }

    #[tokio::test]
    async fn infer_injects_configured_passive_context_without_agent_get() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("ok", vec![])]));
        let queries = Arc::new(Mutex::new(Vec::new()));
        let config = SeqConfig {
            provider: provider.clone(),
            hydration: SourceRegistry::new().register(StaticSource {
                name: "workspace",
                kind: SourceKind::Knowledge,
                capabilities: SourceCapability::SESSION_CONTEXT,
                content: "passive workspace facts",
                queries: queries.clone(),
            }),
            passive_hydration: PassiveHydrationConfig::with_sources([
                PassiveSource::SessionContext,
            ]),
            checkpoint_path: None,
            trace: test_trace(),
        };
        let prompt = vec![ChatMessage::user("answer")];

        let (result, _) =
            run_sequential(&config, prompt, infer(Model("mock".into()), vec![])).await?;

        assert_eq!(result.content, "ok");
        let prompts = provider.prompts();
        assert!(prompts[0].iter().any(|message| {
            message.role == "system"
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("passive workspace facts"))
        }));
        assert_eq!(*queries.lock().unwrap(), vec![None]);
        Ok(())
    }

    #[tokio::test]
    async fn active_semantic_get_uses_query_capable_backend() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let queries = Arc::new(Mutex::new(Vec::new()));
        let config = SeqConfig {
            provider,
            hydration: SourceRegistry::new().register(StaticSource {
                name: "semantic-store",
                kind: SourceKind::Semantic,
                capabilities: SourceCapability::QUERY,
                content: "semantic result",
                queries: queries.clone(),
            }),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
        };

        let (value, _) = run_sequential(&config, (), crate::op::get("semantic:topic")).await?;

        assert_eq!(queries.lock().unwrap().as_slice(), &[Some("topic".into())]);
        assert_eq!(value[0]["content"], json!("semantic result"));
        Ok(())
    }
}
