use crate::ir::{
    Block, BlockId, Expr, Instr, Machine, Program, ProgramId, PromptRef, Terminator, Var,
};
use crate::op::{Model, Prompt};
use serde_json::Value;
use std::collections::BTreeMap;

pub fn agent_loop_ir(model: Model, prompt: Prompt, max_turns: usize) -> Machine {
    let entry = BlockId(0);
    let done = BlockId(1);
    let prepare_tools = BlockId(2);
    let tool_loop = BlockId(3);
    let tool_body = BlockId(4);
    let next_turn = BlockId(5);
    let shell_tool = BlockId(6);
    let infer_tool = BlockId(7);
    let append_tool = BlockId(8);

    let history = Var("history".into());
    let turns_left = Var("turns_left".into());
    let response = Var("response".into());
    let tool_calls = Var("tool_calls".into());
    let no_tool_calls = Var("no_tool_calls".into());
    let no_turns_left = Var("no_turns_left".into());
    let should_return = Var("should_return".into());
    let history_with_assistant = Var("history_with_assistant".into());
    let i = Var("i".into());
    let keep_looping = Var("keep_looping".into());
    let call = Var("call".into());
    let function = Var("function".into());
    let raw_arguments = Var("raw_arguments".into());
    let arguments = Var("arguments".into());
    let function_name = Var("function_name".into());
    let is_infer_tool = Var("is_infer_tool".into());
    let command = Var("command".into());
    let eval_result = Var("eval_result".into());
    let infer_model = Var("infer_model".into());
    let infer_prompt_text = Var("infer_prompt_text".into());
    let infer_prompt = Var("infer_prompt".into());
    let infer_result = Var("infer_result".into());
    let tool_content = Var("tool_content".into());
    let tool_message = Var("tool_message".into());
    let next_history = Var("next_history".into());
    let next_i = Var("next_i".into());
    let next_turns_left = Var("next_turns_left".into());

    let mut blocks = BTreeMap::new();
    blocks.insert(
        entry,
        Block {
            params: vec![history.clone(), turns_left.clone()],
            instructions: vec![
                Instr::Infer {
                    out: response.clone(),
                    model: Expr::Value(Value::String(model.0)),
                    prompt: PromptRef::Var(history.clone()),
                    policy: Default::default(),
                },
                Instr::Let {
                    out: tool_calls.clone(),
                    expr: Expr::Field {
                        base: response.clone(),
                        field: "tool_calls".into(),
                    },
                },
                Instr::Let {
                    out: no_tool_calls.clone(),
                    expr: Expr::IsEmpty {
                        base: tool_calls.clone(),
                    },
                },
                Instr::Let {
                    out: no_turns_left.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(turns_left.clone())),
                        right: Box::new(Expr::Value(Value::Number(0.into()))),
                    },
                },
                Instr::Let {
                    out: should_return.clone(),
                    expr: Expr::Or {
                        left: Box::new(Expr::Var(no_tool_calls)),
                        right: Box::new(Expr::Var(no_turns_left)),
                    },
                },
            ],
            terminator: Terminator::If {
                cond: Expr::Var(should_return),
                then_block: done,
                else_block: prepare_tools,
            },
        },
    );

    blocks.insert(
        done,
        Block {
            params: vec![],
            instructions: vec![],
            terminator: Terminator::Return {
                value: Expr::Var(response.clone()),
            },
        },
    );

    blocks.insert(
        prepare_tools,
        Block {
            params: vec![],
            instructions: vec![Instr::Let {
                out: history_with_assistant.clone(),
                expr: Expr::Push {
                    base: history.clone(),
                    value: Box::new(Expr::Object(BTreeMap::from([
                        (
                            "role".into(),
                            Expr::Value(Value::String("assistant".into())),
                        ),
                        (
                            "content".into(),
                            Expr::Field {
                                base: response.clone(),
                                field: "content".into(),
                            },
                        ),
                        (
                            "tool_calls".into(),
                            Expr::Field {
                                base: response.clone(),
                                field: "tool_calls".into(),
                            },
                        ),
                    ]))),
                },
            }],
            terminator: Terminator::Goto {
                block: tool_loop,
                args: vec![
                    Expr::Var(history_with_assistant),
                    Expr::Var(tool_calls.clone()),
                    Expr::Value(Value::Number(0.into())),
                    Expr::Var(turns_left.clone()),
                ],
            },
        },
    );

    blocks.insert(
        tool_loop,
        Block {
            params: vec![
                history.clone(),
                tool_calls.clone(),
                i.clone(),
                turns_left.clone(),
            ],
            instructions: vec![Instr::Let {
                out: keep_looping.clone(),
                expr: Expr::Lt {
                    left: Box::new(Expr::Var(i.clone())),
                    right: Box::new(Expr::Len {
                        base: tool_calls.clone(),
                    }),
                },
            }],
            terminator: Terminator::If {
                cond: Expr::Var(keep_looping),
                then_block: tool_body,
                else_block: next_turn,
            },
        },
    );

    blocks.insert(
        tool_body,
        Block {
            params: vec![],
            instructions: vec![
                Instr::Let {
                    out: call.clone(),
                    expr: Expr::Index {
                        base: tool_calls.clone(),
                        index: Box::new(Expr::Var(i.clone())),
                    },
                },
                Instr::Let {
                    out: function.clone(),
                    expr: Expr::Field {
                        base: call.clone(),
                        field: "function".into(),
                    },
                },
                Instr::Let {
                    out: function_name.clone(),
                    expr: Expr::Field {
                        base: function.clone(),
                        field: "name".into(),
                    },
                },
                Instr::Let {
                    out: raw_arguments.clone(),
                    expr: Expr::Field {
                        base: function.clone(),
                        field: "arguments".into(),
                    },
                },
                Instr::Let {
                    out: arguments.clone(),
                    expr: Expr::JsonParse {
                        value: Box::new(Expr::Var(raw_arguments)),
                    },
                },
                Instr::Let {
                    out: is_infer_tool.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(function_name)),
                        right: Box::new(Expr::Value(Value::String("infer".into()))),
                    },
                },
            ],
            terminator: Terminator::If {
                cond: Expr::Var(is_infer_tool),
                then_block: infer_tool,
                else_block: shell_tool,
            },
        },
    );

    blocks.insert(
        shell_tool,
        Block {
            params: vec![],
            instructions: vec![
                Instr::Let {
                    out: command.clone(),
                    expr: Expr::Field {
                        base: arguments.clone(),
                        field: "command".into(),
                    },
                },
                Instr::Eval {
                    out: eval_result.clone(),
                    request: crate::ir::EvalRequest::Shell {
                        command: Expr::Var(command),
                    },
                    policy: Default::default(),
                },
                Instr::Let {
                    out: tool_content.clone(),
                    expr: Expr::ToString {
                        value: Box::new(Expr::Var(eval_result)),
                    },
                },
            ],
            terminator: Terminator::Goto {
                block: append_tool,
                args: vec![Expr::Var(tool_content.clone())],
            },
        },
    );

    blocks.insert(
        infer_tool,
        Block {
            params: vec![],
            instructions: vec![
                Instr::Let {
                    out: infer_model.clone(),
                    expr: Expr::Field {
                        base: arguments.clone(),
                        field: "model".into(),
                    },
                },
                Instr::Let {
                    out: infer_prompt_text.clone(),
                    expr: Expr::Field {
                        base: arguments.clone(),
                        field: "prompt".into(),
                    },
                },
                Instr::Let {
                    out: infer_prompt.clone(),
                    expr: Expr::Array(vec![Expr::Object(BTreeMap::from([
                        ("role".into(), Expr::Value(Value::String("user".into()))),
                        ("content".into(), Expr::Var(infer_prompt_text)),
                    ]))]),
                },
                Instr::Infer {
                    out: infer_result.clone(),
                    model: Expr::Var(infer_model),
                    prompt: PromptRef::Var(infer_prompt),
                    policy: Default::default(),
                },
                Instr::Let {
                    out: tool_content.clone(),
                    expr: Expr::ToString {
                        value: Box::new(Expr::Var(infer_result)),
                    },
                },
            ],
            terminator: Terminator::Goto {
                block: append_tool,
                args: vec![Expr::Var(tool_content.clone())],
            },
        },
    );

    blocks.insert(
        append_tool,
        Block {
            params: vec![tool_content.clone()],
            instructions: vec![
                Instr::Let {
                    out: tool_message.clone(),
                    expr: Expr::Object(BTreeMap::from([
                        ("role".into(), Expr::Value(Value::String("tool".into()))),
                        (
                            "tool_call_id".into(),
                            Expr::Field {
                                base: call.clone(),
                                field: "id".into(),
                            },
                        ),
                        ("content".into(), Expr::Var(tool_content)),
                    ])),
                },
                Instr::Let {
                    out: next_history.clone(),
                    expr: Expr::Push {
                        base: history.clone(),
                        value: Box::new(Expr::Var(tool_message)),
                    },
                },
                Instr::Let {
                    out: next_i.clone(),
                    expr: Expr::Add {
                        left: Box::new(Expr::Var(i.clone())),
                        right: Box::new(Expr::Value(Value::Number(1.into()))),
                    },
                },
            ],
            terminator: Terminator::Goto {
                block: tool_loop,
                args: vec![
                    Expr::Var(next_history),
                    Expr::Var(tool_calls),
                    Expr::Var(next_i),
                    Expr::Var(turns_left.clone()),
                ],
            },
        },
    );

    blocks.insert(
        next_turn,
        Block {
            params: vec![],
            instructions: vec![Instr::Let {
                out: next_turns_left.clone(),
                expr: Expr::Sub {
                    left: Box::new(Expr::Var(turns_left)),
                    right: Box::new(Expr::Value(Value::Number(1.into()))),
                },
            }],
            terminator: Terminator::Goto {
                block: entry,
                args: vec![Expr::Var(history), Expr::Var(next_turns_left)],
            },
        },
    );

    Machine {
        program: Program {
            id: ProgramId("agent-loop".into()),
            entry,
            blocks,
        },
        block: entry,
        pc: 0,
        env: BTreeMap::from([
            (
                Var("history".into()),
                serde_json::to_value(prompt).expect("prompt serializes"),
            ),
            (
                Var("turns_left".into()),
                Value::Number((max_turns as u64).into()),
            ),
        ]),
        effect_visits: BTreeMap::new(),
        continuation_stack: vec![],
        budgets: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::{PassiveHydrationConfig, SourceRegistry};
    use crate::interpreter::{EvalConfig, SeqConfig};
    use crate::ir::validate_program;
    use crate::op::{ChatMessage, Response, ResponseToolCall};
    use crate::provider::{ChatProvider, ToolSpec};
    use crate::trace::TraceLogger;
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    struct MockProvider {
        responses: Mutex<Vec<Response>>,
        prompt_count: Mutex<usize>,
    }

    impl MockProvider {
        fn new(mut responses: Vec<Response>) -> Self {
            responses.reverse();
            Self {
                responses: Mutex::new(responses),
                prompt_count: Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        async fn chat(
            &self,
            _model: &Model,
            _tools: &[ToolSpec],
            _messages: &[ChatMessage],
        ) -> Result<Response> {
            *self.prompt_count.lock().unwrap() += 1;
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
            tokens: 1,
        }
    }

    fn config(provider: Arc<dyn ChatProvider>) -> SeqConfig {
        SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: TraceLogger::new(
                Uuid::new_v4().to_string(),
                std::env::temp_dir().join(format!("agent-ir-loop-{}.jsonl", Uuid::new_v4())),
            ),
            eval: EvalConfig::default(),
            replay: None,
        }
    }

    #[test]
    fn agent_loop_ir_validates() {
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("hello")],
            4,
        );
        validate_program(&machine.program).unwrap();
    }

    #[tokio::test]
    async fn agent_loop_ir_executes_infer_tool_directly() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ResponseToolCall::new(
                    "call-1",
                    "infer",
                    serde_json::json!({ "model": "mock", "prompt": "sub question" }),
                )],
            ),
            response("sub answer", vec![]),
            response("done", vec![]),
        ]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![
                ChatMessage::system("system"),
                ChatMessage::user("use infer"),
            ],
            4,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(value["content"], Value::String("done".into()));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_executes_shell_tool_then_finishes() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ResponseToolCall::new(
                    "call-1",
                    "shell",
                    serde_json::json!({ "command": "printf ir-loop" }),
                )],
            ),
            response("done", vec![]),
        ]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![
                ChatMessage::system("system"),
                ChatMessage::user("use shell"),
            ],
            4,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(value["content"], Value::String("done".into()));
        Ok(())
    }
}
