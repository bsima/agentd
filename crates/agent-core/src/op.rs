use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

pub type BoxFutureOp<S, A> = Pin<Box<dyn Future<Output = Op<S, A>> + Send>>;
pub type Prompt = Vec<ChatMessage>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: Option<String>, tool_calls: Vec<ResponseToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content,
            tool_call_id: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub content: String,
    pub tool_calls: Vec<ResponseToolCall>,
    pub tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type", default = "tool_call_type")]
    pub kind: String,
    pub function: ResponseToolFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseToolFunction {
    pub name: String,
    pub arguments: String,
}

impl ResponseToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            kind: tool_call_type(),
            function: ResponseToolFunction {
                name: name.into(),
                arguments: arguments.to_string(),
            },
        }
    }

    pub fn name(&self) -> &str {
        &self.function.name
    }

    pub fn arguments(&self) -> Value {
        serde_json::from_str(&self.function.arguments)
            .unwrap_or_else(|_| serde_json::json!({ "raw": self.function.arguments }))
    }
}

fn tool_call_type() -> String {
    "function".into()
}

pub enum OpF<S, A> {
    Infer {
        model: Model,
        prompt: Prompt,
        next: Box<dyn FnOnce(Response) -> Op<S, A> + Send>,
    },
    Eval {
        command: String,
        next: Box<dyn FnOnce(Value) -> Op<S, A> + Send>,
    },
    Get {
        key: String,
        next: Box<dyn FnOnce(Value) -> Op<S, A> + Send>,
    },
    Put {
        key: String,
        value: Value,
        next: Op<S, A>,
    },
    Emit {
        event: crate::trace::Event,
        next: Op<S, A>,
    },
    Par {
        ops: Vec<Op<S, ()>>,
        next: Box<dyn FnOnce(Vec<()>) -> Op<S, A> + Send>,
    },
    Pure(A),
}

pub struct Op<S, A>(pub Box<OpF<S, A>>);

impl<S: Send + 'static, A: Send + 'static> Op<S, A> {
    pub fn pure(value: A) -> Self {
        Self(Box::new(OpF::Pure(value)))
    }

    pub fn and_then<B, F>(self, f: F) -> Op<S, B>
    where
        B: Send + 'static,
        F: FnOnce(A) -> Op<S, B> + Send + 'static,
    {
        match *self.0 {
            OpF::Pure(a) => f(a),
            OpF::Infer {
                model,
                prompt,
                next,
            } => Op(Box::new(OpF::Infer {
                model,
                prompt,
                next: Box::new(move |r| next(r).and_then(f)),
            })),
            OpF::Eval { command, next } => Op(Box::new(OpF::Eval {
                command,
                next: Box::new(move |v| next(v).and_then(f)),
            })),
            OpF::Get { key, next } => Op(Box::new(OpF::Get {
                key,
                next: Box::new(move |v| next(v).and_then(f)),
            })),
            OpF::Put { key, value, next } => Op(Box::new(OpF::Put {
                key,
                value,
                next: next.and_then(f),
            })),
            OpF::Emit { event, next } => Op(Box::new(OpF::Emit {
                event,
                next: next.and_then(f),
            })),
            OpF::Par { ops, next } => Op(Box::new(OpF::Par {
                ops,
                next: Box::new(move |values| next(values).and_then(f)),
            })),
        }
    }

    pub fn map<B, F>(self, f: F) -> Op<S, B>
    where
        B: Send + 'static,
        F: FnOnce(A) -> B + Send + 'static,
    {
        self.and_then(|a| Op::pure(f(a)))
    }
}

pub fn infer<S: Send + 'static>(model: Model, prompt: Prompt) -> Op<S, Response> {
    Op(Box::new(OpF::Infer {
        model,
        prompt,
        next: Box::new(Op::pure),
    }))
}

pub fn eval<S: Send + 'static>(command: impl Into<String>) -> Op<S, Value> {
    Op(Box::new(OpF::Eval {
        command: command.into(),
        next: Box::new(Op::pure),
    }))
}

pub fn get<S: Send + 'static>(key: impl Into<String>) -> Op<S, Value> {
    Op(Box::new(OpF::Get {
        key: key.into(),
        next: Box::new(Op::pure),
    }))
}

pub fn put<S: Send + 'static>(key: impl Into<String>, value: Value) -> Op<S, ()> {
    Op(Box::new(OpF::Put {
        key: key.into(),
        value,
        next: Op::pure(()),
    }))
}

pub fn emit<S: Send + 'static>(event: crate::trace::Event) -> Op<S, ()> {
    Op(Box::new(OpF::Emit {
        event,
        next: Op::pure(()),
    }))
}

pub fn par<S: Send + 'static>(ops: Vec<Op<S, ()>>) -> Op<S, Vec<()>> {
    Op(Box::new(OpF::Par {
        ops,
        next: Box::new(Op::pure),
    }))
}

pub fn agent_loop(model: Model, prompt: Prompt, max_turns: usize) -> Op<Prompt, Response> {
    infer(model.clone(), prompt.clone()).and_then(move |response| {
        if response.tool_calls.is_empty() || max_turns == 0 {
            Op::pure(response)
        } else {
            let calls = response.tool_calls.clone();
            get("temporal:history").and_then(move |value| {
                let mut history: Prompt =
                    serde_json::from_value(value).unwrap_or_else(|_| prompt.clone());
                history.push(ChatMessage::assistant(
                    (!response.content.is_empty()).then_some(response.content.clone()),
                    response.tool_calls.clone(),
                ));

                let mut program = Op::pure(history);
                for call in calls {
                    program = program.and_then(move |mut acc| {
                        let id = call.id.clone();
                        match command_from_tool_call(&call) {
                            Ok(command) => eval(command).map(move |result| {
                                acc.push(ChatMessage::tool(id, result.to_string()));
                                acc
                            }),
                            Err(error) => Op::pure({
                                acc.push(ChatMessage::tool(id, error.to_string()));
                                acc
                            }),
                        }
                    });
                }

                program.and_then(move |history| {
                    let value = serde_json::to_value(&history).unwrap_or(Value::Null);
                    put("temporal:history", value)
                        .and_then(move |_| agent_loop(model, history, max_turns - 1))
                })
            })
        }
    })
}

fn command_from_tool_call(call: &ResponseToolCall) -> std::result::Result<String, Value> {
    if call.name() != "shell" {
        return Err(serde_json::json!({
            "ok": false,
            "error": "unknown_tool",
            "tool": call.name(),
            "message": "unknown tool; available tools: shell"
        }));
    }

    let args = call.arguments();
    match args.get("command").and_then(Value::as_str) {
        Some(command) if !command.trim().is_empty() => Ok(command.to_string()),
        _ => Err(serde_json::json!({
            "ok": false,
            "error": "invalid_arguments",
            "message": "shell requires non-empty string argument: command"
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::GcMode;
    use crate::hydration::{PassiveHydrationConfig, SourceRegistry};
    use crate::interpreter::{run_sequential, SeqConfig};
    use crate::provider::{ChatProvider, ToolSpec};
    use crate::trace::TraceLogger;
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use std::sync::Arc;
    use uuid::Uuid;

    struct NoopProvider;

    #[async_trait]
    impl ChatProvider for NoopProvider {
        async fn chat(
            &self,
            _model: &Model,
            _tools: &[ToolSpec],
            _messages: &[ChatMessage],
        ) -> Result<Response> {
            Err(anyhow!(
                "noop provider should not be called by pure Op tests"
            ))
        }
    }

    fn seq_config() -> SeqConfig {
        let path =
            std::env::temp_dir().join(format!("agent-core-op-test-{}.jsonl", Uuid::new_v4()));
        SeqConfig {
            provider: Arc::new(NoopProvider),
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: TraceLogger::new(Uuid::new_v4().to_string(), path),
            eval: crate::interpreter::EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        }
    }

    async fn observe(op: Op<i32, i32>, state: i32) -> Result<(i32, i32)> {
        run_sequential(&seq_config(), state, op).await
    }

    #[tokio::test]
    async fn monad_left_identity_holds_for_pure() -> Result<()> {
        fn f(value: i32) -> Op<i32, i32> {
            Op::pure(value + 1)
        }

        let left = observe(Op::pure(41).and_then(f), 0).await?;
        let right = observe(f(41), 0).await?;

        assert_eq!(left, right);
        Ok(())
    }

    #[tokio::test]
    async fn monad_right_identity_holds_for_pure() -> Result<()> {
        let left = observe(Op::pure(42).and_then(Op::pure), 7).await?;
        let right = observe(Op::pure(42), 7).await?;

        assert_eq!(left, right);
        Ok(())
    }

    #[tokio::test]
    async fn monad_associativity_holds_for_pure() -> Result<()> {
        fn f(value: i32) -> Op<i32, i32> {
            Op::pure(value + 2)
        }
        fn g(value: i32) -> Op<i32, i32> {
            Op::pure(value * 3)
        }

        let left = observe(Op::pure(10).and_then(f).and_then(g), 0).await?;
        let right = observe(Op::pure(10).and_then(|a| f(a).and_then(g)), 0).await?;

        assert_eq!(left, right);
        Ok(())
    }

    #[test]
    fn infer_wraps_in_infer_node() {
        let prompt = vec![ChatMessage::user("hello")];
        match *infer::<()>(Model("model".into()), prompt.clone()).0 {
            OpF::Infer {
                model,
                prompt: actual,
                ..
            } => {
                assert_eq!(model, Model("model".into()));
                assert_eq!(actual, prompt);
            }
            _ => panic!("infer() did not create OpF::Infer"),
        }
    }

    #[test]
    fn eval_wraps_in_eval_node() {
        match *eval::<()>("printf ok").0 {
            OpF::Eval { command, .. } => assert_eq!(command, "printf ok"),
            _ => panic!("eval() did not create OpF::Eval"),
        }
    }

    #[test]
    fn par_wraps_in_par_node() {
        match *par::<()>(vec![Op::pure(())]).0 {
            OpF::Par { ops, .. } => assert_eq!(ops.len(), 1),
            _ => panic!("par() did not create OpF::Par"),
        }
    }

    #[test]
    fn shell_tool_call_parser_accepts_only_non_empty_shell_command() {
        let valid = ResponseToolCall::new(
            "call-1",
            "shell",
            serde_json::json!({ "command": "printf ok" }),
        );
        assert_eq!(command_from_tool_call(&valid).unwrap(), "printf ok");

        let unknown = ResponseToolCall::new(
            "call-2",
            "echo",
            serde_json::json!({ "command": "printf bad" }),
        );
        assert_eq!(
            command_from_tool_call(&unknown).unwrap_err()["error"],
            serde_json::json!("unknown_tool")
        );

        let missing = ResponseToolCall::new("call-3", "shell", serde_json::json!({}));
        assert_eq!(
            command_from_tool_call(&missing).unwrap_err()["error"],
            serde_json::json!("invalid_arguments")
        );

        let empty =
            ResponseToolCall::new("call-4", "shell", serde_json::json!({ "command": "   " }));
        assert_eq!(
            command_from_tool_call(&empty).unwrap_err()["error"],
            serde_json::json!("invalid_arguments")
        );
    }

    #[tokio::test]
    async fn get_put_and_emit_round_trip_through_sequential_interpreter() -> Result<()> {
        let event = crate::trace::Event::AgentDone {
            run_id: "op-test".into(),
            timestamp: chrono::Utc::now(),
        };
        let program = get::<i32>("temporal:history").and_then(move |value| {
            let state: i32 = serde_json::from_value(value).unwrap();
            put("temporal:history", serde_json::json!(state + 1))
                .and_then(move |_| emit(event).and_then(move |_| get::<i32>("temporal:history")))
                .map(|value| serde_json::from_value(value).unwrap())
        });

        let observed = run_sequential(&seq_config(), 41, program).await?;
        assert_eq!(observed, (42, 42));
        Ok(())
    }
}
