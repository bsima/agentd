use crate::op::{Op, OpF};
use crate::provider::{ChatProvider, ToolFunctionSpec, ToolSpec};
use crate::trace::{Event, TraceLogger};
use anyhow::{anyhow, Result};
use async_recursion::async_recursion;
use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub type ToolMap = HashMap<String, Arc<dyn Tool>>;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn execute(&self, args: Value) -> Result<Value>;
}

pub struct SeqConfig {
    pub provider: Arc<dyn ChatProvider>,
    pub tools: ToolMap,
    pub trace: TraceLogger,
}

impl SeqConfig {
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|tool| ToolSpec {
                kind: "function".into(),
                function: ToolFunctionSpec {
                    name: tool.name().into(),
                    description: tool.description().into(),
                    parameters: tool.parameters(),
                },
            })
            .collect()
    }
}

#[async_recursion]
pub async fn run_sequential<S, A>(config: &SeqConfig, state: S, op: Op<S, A>) -> Result<(A, S)>
where
    S: Clone + Send + 'static,
    A: Send + 'static,
{
    match *op.0 {
        OpF::Pure(value) => Ok((value, state)),
        OpF::Infer {
            model,
            prompt,
            next,
        } => {
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
        OpF::Tool { name, args, next } => {
            config
                .trace
                .emit(&Event::ToolCall {
                    run_id: config.trace.run_id().into(),
                    name: name.clone(),
                    args: args.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            let tool = config
                .tools
                .get(&name)
                .ok_or_else(|| anyhow!("unknown tool: {name}"))?;
            let result = tool.execute(args).await?;
            config
                .trace
                .emit(&Event::ToolResult {
                    run_id: config.trace.run_id().into(),
                    name,
                    result: result.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential(config, state, next(result)).await
        }
        OpF::Get { next } => run_sequential(config, state.clone(), next(state.clone())).await,
        OpF::Put {
            state: new_state,
            next,
        } => run_sequential(config, new_state, next).await,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{agent_loop, ChatMessage, Model, Prompt, Response, ResponseToolCall};
    use crate::provider::ToolSpec;
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

    struct EchoTool {
        calls: Arc<Mutex<Vec<Value>>>,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo test args"
        }

        fn parameters(&self) -> Value {
            json!({ "type": "object" })
        }

        async fn execute(&self, args: Value) -> Result<Value> {
            self.calls.lock().unwrap().push(args.clone());
            Ok(json!({ "echoed": args }))
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
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolMap::new();
        tools.insert(
            "echo".into(),
            Arc::new(EchoTool {
                calls: calls.clone(),
            }),
        );
        let config = SeqConfig {
            provider: provider.clone(),
            tools,
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
        assert_eq!(calls.lock().unwrap().len(), 1);
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
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolMap::new();
        tools.insert(
            "echo".into(),
            Arc::new(EchoTool {
                calls: calls.clone(),
            }),
        );
        let config = SeqConfig {
            provider: provider.clone(),
            tools,
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
        assert_eq!(calls.lock().unwrap().len(), 2);
        assert_eq!(state.iter().filter(|msg| msg.role == "tool").count(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn get_put_and_par_are_interpreted_sequentially() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let config = SeqConfig {
            provider,
            tools: ToolMap::new(),
            trace: test_trace(),
        };
        let program = crate::op::put(1)
            .and_then(|_| crate::op::par(vec![crate::op::put(2), crate::op::put(3)]))
            .and_then(|_| crate::op::get());

        let (result, state) = run_sequential(&config, 0, program).await?;

        assert_eq!(result, 3);
        assert_eq!(state, 3);
        Ok(())
    }
}
