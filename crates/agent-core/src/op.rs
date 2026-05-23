use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

pub type BoxFutureOp<S, A> = Pin<Box<dyn Future<Output = Op<S, A>> + Send>>;
pub type Prompt = Vec<ChatMessage>;
pub type ToolName = String;

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
    Tool {
        name: ToolName,
        args: Value,
        next: Box<dyn FnOnce(Value) -> Op<S, A> + Send>,
    },
    Get {
        next: Box<dyn FnOnce(S) -> Op<S, A> + Send>,
    },
    Put {
        state: S,
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
            OpF::Tool { name, args, next } => Op(Box::new(OpF::Tool {
                name,
                args,
                next: Box::new(move |v| next(v).and_then(f)),
            })),
            OpF::Get { next } => Op(Box::new(OpF::Get {
                next: Box::new(move |s| next(s).and_then(f)),
            })),
            OpF::Put { state, next } => Op(Box::new(OpF::Put {
                state,
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

pub fn tool<S: Send + 'static>(name: impl Into<ToolName>, args: Value) -> Op<S, Value> {
    Op(Box::new(OpF::Tool {
        name: name.into(),
        args,
        next: Box::new(Op::pure),
    }))
}

pub fn get<S: Clone + Send + 'static>() -> Op<S, S> {
    Op(Box::new(OpF::Get {
        next: Box::new(Op::pure),
    }))
}

pub fn put<S: Send + 'static>(state: S) -> Op<S, ()> {
    Op(Box::new(OpF::Put {
        state,
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
    infer(model.clone(), prompt).and_then(move |response| {
        if response.tool_calls.is_empty() || max_turns == 0 {
            Op::pure(response)
        } else {
            let calls = response.tool_calls.clone();
            get().and_then(move |mut history: Prompt| {
                history.push(ChatMessage::assistant(
                    (!response.content.is_empty()).then_some(response.content.clone()),
                    response.tool_calls.clone(),
                ));

                let mut program = Op::pure(history);
                for call in calls {
                    program = program.and_then(move |mut acc| {
                        let id = call.id.clone();
                        tool(call.name().to_string(), call.arguments()).map(move |result| {
                            acc.push(ChatMessage::tool(id, result.to_string()));
                            acc
                        })
                    });
                }

                program.and_then(move |history| {
                    put(history.clone())
                        .and_then(move |_| agent_loop(model, history, max_turns - 1))
                })
            })
        }
    })
}
