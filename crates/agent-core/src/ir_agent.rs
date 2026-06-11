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
    let shell_dispatch = BlockId(6);
    let shell_tool = BlockId(7);
    let infer_tool = BlockId(8);
    let append_tool = BlockId(9);
    let invalid_tool = BlockId(10);
    let invalid_arguments = BlockId(11);
    let shell_eval = BlockId(12);
    let infer_eval = BlockId(13);
    let route = BlockId(14);
    let nudge_turn = BlockId(15);
    let route_done = BlockId(16);
    let budget_done = BlockId(17);

    let history = Var("history".into());
    let turns_left = Var("turns_left".into());
    let response = Var("response".into());
    let tool_calls = Var("tool_calls".into());
    let no_tool_calls = Var("no_tool_calls".into());
    let finish_reason = Var("finish_reason".into());
    let is_stop = Var("is_stop".into());
    let has_pending_tool_calls = Var("has_pending_tool_calls".into());
    let no_pending_tool_calls = Var("no_pending_tool_calls".into());
    let can_stop = Var("can_stop".into());
    let no_turns_left = Var("no_turns_left".into());
    let should_return = Var("should_return".into());
    let content = Var("content".into());
    let content_empty = Var("content_empty".into());
    let history_for_nudge = Var("history_for_nudge".into());
    let nudged_history = Var("nudged_history".into());
    let nudge_turns_left = Var("nudge_turns_left".into());
    let budget_response = Var("budget_response".into());
    let history_with_assistant = Var("history_with_assistant".into());
    let i = Var("i".into());
    let keep_looping = Var("keep_looping".into());
    let call = Var("call".into());
    let arguments = Var("arguments".into());
    let function_name = Var("function_name".into());
    let is_infer_tool = Var("is_infer_tool".into());
    let is_shell_tool = Var("is_shell_tool".into());
    let missing_command = Var("missing_command".into());
    let missing_infer_model = Var("missing_infer_model".into());
    let missing_infer_prompt = Var("missing_infer_prompt".into());
    let invalid_infer_arguments = Var("invalid_infer_arguments".into());
    let invalid_message = Var("invalid_message".into());
    let command = Var("command".into());
    let eval_result = Var("eval_result".into());
    let infer_model = Var("infer_model".into());
    let infer_prompt_text = Var("infer_prompt_text".into());
    let infer_prompt = Var("infer_prompt".into());
    let infer_result = Var("infer_result".into());
    let infer_content = Var("infer_content".into());
    let infer_content_empty = Var("infer_content_empty".into());
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
                    out: finish_reason.clone(),
                    expr: Expr::FieldOr {
                        base: response.clone(),
                        field: "finish_reason".into(),
                        default: Box::new(Expr::Value(Value::Null)),
                    },
                },
                Instr::Let {
                    out: is_stop.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(finish_reason)),
                        right: Box::new(Expr::Value(Value::String("stop".into()))),
                    },
                },
                Instr::Let {
                    out: has_pending_tool_calls.clone(),
                    expr: Expr::HasPendingToolCalls {
                        base: history.clone(),
                    },
                },
                Instr::Let {
                    out: no_pending_tool_calls.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(has_pending_tool_calls)),
                        right: Box::new(Expr::Value(Value::Bool(false))),
                    },
                },
                Instr::Let {
                    out: can_stop.clone(),
                    expr: Expr::And {
                        left: Box::new(Expr::And {
                            left: Box::new(Expr::Var(is_stop)),
                            right: Box::new(Expr::Var(no_tool_calls.clone())),
                        }),
                        right: Box::new(Expr::Var(no_pending_tool_calls)),
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
                        left: Box::new(Expr::Var(can_stop.clone())),
                        right: Box::new(Expr::Var(no_turns_left)),
                    },
                },
                Instr::Let {
                    out: content.clone(),
                    expr: Expr::StringOr {
                        value: Box::new(Expr::FieldOr {
                            base: response.clone(),
                            field: "content".into(),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        }),
                        default: Box::new(Expr::Value(Value::String("".into()))),
                    },
                },
                Instr::Let {
                    out: content_empty.clone(),
                    expr: Expr::IsEmpty {
                        base: content.clone(),
                    },
                },
            ],
            terminator: Terminator::If {
                cond: Expr::Var(should_return),
                then_block: route_done,
                else_block: route,
            },
        },
    );

    // A natural stop returns the response verbatim; a turn-budget stop must
    // be distinguishable downstream (t-1133), so it returns the response
    // annotated with metadata.stop_reason = "turn_budget_exhausted". When
    // both conditions hold, the natural stop wins: the budget never fired.
    blocks.insert(
        route_done,
        Block {
            params: vec![],
            instructions: vec![],
            terminator: Terminator::If {
                cond: Expr::Var(can_stop),
                then_block: done,
                else_block: budget_done,
            },
        },
    );

    blocks.insert(
        budget_done,
        Block {
            params: vec![],
            instructions: vec![Instr::Let {
                out: budget_response.clone(),
                expr: Expr::Object(BTreeMap::from([
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
                    (
                        "finish_reason".into(),
                        Expr::FieldOr {
                            base: response.clone(),
                            field: "finish_reason".into(),
                            default: Box::new(Expr::Value(Value::Null)),
                        },
                    ),
                    (
                        "input_tokens".into(),
                        Expr::Field {
                            base: response.clone(),
                            field: "input_tokens".into(),
                        },
                    ),
                    (
                        "output_tokens".into(),
                        Expr::Field {
                            base: response.clone(),
                            field: "output_tokens".into(),
                        },
                    ),
                    (
                        "total_tokens".into(),
                        Expr::Field {
                            base: response.clone(),
                            field: "total_tokens".into(),
                        },
                    ),
                    (
                        "metadata".into(),
                        Expr::Object(BTreeMap::from([(
                            "stop_reason".into(),
                            Expr::Value(Value::String("turn_budget_exhausted".into())),
                        )])),
                    ),
                ])),
            }],
            terminator: Terminator::Return {
                value: Expr::Var(budget_response),
            },
        },
    );

    blocks.insert(
        route,
        Block {
            params: vec![],
            instructions: vec![],
            terminator: Terminator::If {
                cond: Expr::Var(no_tool_calls.clone()),
                then_block: nudge_turn,
                else_block: prepare_tools,
            },
        },
    );

    // Mirror of the Op loop's stalled-turn recovery: a non-stop response with
    // no tool calls gets the assistant text (when present) plus a synthetic
    // user continuation nudge appended, then the loop re-infers.
    blocks.insert(
        nudge_turn,
        Block {
            params: vec![],
            instructions: vec![
                Instr::Let {
                    out: history_for_nudge.clone(),
                    expr: Expr::If {
                        cond: Box::new(Expr::Var(content_empty.clone())),
                        then_value: Box::new(Expr::Var(history.clone())),
                        else_value: Box::new(Expr::Push {
                            base: history.clone(),
                            value: Box::new(Expr::Object(BTreeMap::from([
                                (
                                    "role".into(),
                                    Expr::Value(Value::String("assistant".into())),
                                ),
                                ("content".into(), Expr::Var(content.clone())),
                            ]))),
                        }),
                    },
                },
                Instr::Let {
                    out: nudged_history.clone(),
                    expr: Expr::Push {
                        base: history_for_nudge.clone(),
                        value: Box::new(Expr::Object(BTreeMap::from([
                            ("role".into(), Expr::Value(Value::String("user".into()))),
                            (
                                "content".into(),
                                Expr::Value(Value::String(crate::op::CONTINUE_NUDGE.into())),
                            ),
                        ]))),
                    },
                },
                Instr::Let {
                    out: nudge_turns_left.clone(),
                    expr: Expr::Sub {
                        left: Box::new(Expr::Var(turns_left.clone())),
                        right: Box::new(Expr::Value(Value::Number(1.into()))),
                    },
                },
            ],
            terminator: Terminator::Goto {
                block: entry,
                args: vec![Expr::Var(nudged_history), Expr::Var(nudge_turns_left)],
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
                            // Parity with ChatMessage::assistant: empty text
                            // becomes null content, not an empty string.
                            "content".into(),
                            Expr::If {
                                cond: Box::new(Expr::Var(content_empty.clone())),
                                then_value: Box::new(Expr::Value(Value::Null)),
                                else_value: Box::new(Expr::Var(content.clone())),
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
                    out: function_name.clone(),
                    expr: Expr::StringOr {
                        value: Box::new(Expr::FieldOr {
                            base: call.clone(),
                            field: "name".into(),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        }),
                        default: Box::new(Expr::Value(Value::String("".into()))),
                    },
                },
                Instr::Let {
                    out: arguments.clone(),
                    expr: Expr::FieldOr {
                        base: call.clone(),
                        field: "arguments".into(),
                        default: Box::new(Expr::Object(BTreeMap::new())),
                    },
                },
                Instr::Let {
                    out: is_infer_tool.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(function_name.clone())),
                        right: Box::new(Expr::Value(Value::String("infer".into()))),
                    },
                },
            ],
            terminator: Terminator::If {
                cond: Expr::Var(is_infer_tool),
                then_block: infer_tool,
                else_block: shell_dispatch,
            },
        },
    );

    blocks.insert(
        shell_dispatch,
        Block {
            params: vec![],
            instructions: vec![Instr::Let {
                out: is_shell_tool.clone(),
                expr: Expr::Eq {
                    left: Box::new(Expr::Var(function_name.clone())),
                    right: Box::new(Expr::Value(Value::String("shell".into()))),
                },
            }],
            terminator: Terminator::If {
                cond: Expr::Var(is_shell_tool),
                then_block: shell_tool,
                else_block: invalid_tool,
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
                    expr: Expr::StringOr {
                        value: Box::new(Expr::FieldOr {
                            base: arguments.clone(),
                            field: "command".into(),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        }),
                        default: Box::new(Expr::Value(Value::String("".into()))),
                    },
                },
                Instr::Let {
                    out: missing_command.clone(),
                    expr: Expr::IsEmpty {
                        base: command.clone(),
                    },
                },
            ],
            terminator: Terminator::If {
                cond: Expr::Var(missing_command),
                then_block: invalid_arguments,
                else_block: shell_eval,
            },
        },
    );

    blocks.insert(
        shell_eval,
        Block {
            params: vec![],
            instructions: vec![
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
                    expr: Expr::StringOr {
                        value: Box::new(Expr::FieldOr {
                            base: arguments.clone(),
                            field: "model".into(),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        }),
                        default: Box::new(Expr::Value(Value::String("".into()))),
                    },
                },
                Instr::Let {
                    out: infer_prompt_text.clone(),
                    expr: Expr::StringOr {
                        value: Box::new(Expr::FieldOr {
                            base: arguments.clone(),
                            field: "prompt".into(),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        }),
                        default: Box::new(Expr::Value(Value::String("".into()))),
                    },
                },
                Instr::Let {
                    out: missing_infer_model.clone(),
                    expr: Expr::IsEmpty {
                        base: infer_model.clone(),
                    },
                },
                Instr::Let {
                    out: missing_infer_prompt.clone(),
                    expr: Expr::IsEmpty {
                        base: infer_prompt_text.clone(),
                    },
                },
                Instr::Let {
                    out: invalid_infer_arguments.clone(),
                    expr: Expr::Or {
                        left: Box::new(Expr::Var(missing_infer_model)),
                        right: Box::new(Expr::Var(missing_infer_prompt.clone())),
                    },
                },
            ],
            terminator: Terminator::If {
                cond: Expr::Var(invalid_infer_arguments),
                then_block: invalid_arguments,
                else_block: infer_eval,
            },
        },
    );

    blocks.insert(
        infer_eval,
        Block {
            params: vec![],
            instructions: vec![
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
                // Feed back the sub-response *text*, not the serialized
                // Response envelope (token counts, finish_reason, ids) — the
                // envelope wastes context and teaches the model to imitate
                // it. Fall back to the envelope only when there is no text
                // (e.g. the sub-infer answered with tool calls).
                Instr::Let {
                    out: infer_content.clone(),
                    expr: Expr::StringOr {
                        value: Box::new(Expr::FieldOr {
                            base: infer_result.clone(),
                            field: "content".into(),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        }),
                        default: Box::new(Expr::Value(Value::String("".into()))),
                    },
                },
                Instr::Let {
                    out: infer_content_empty.clone(),
                    expr: Expr::IsEmpty {
                        base: infer_content.clone(),
                    },
                },
                Instr::Let {
                    out: tool_content.clone(),
                    expr: Expr::If {
                        cond: Box::new(Expr::Var(infer_content_empty)),
                        then_value: Box::new(Expr::ToString {
                            value: Box::new(Expr::Var(infer_result)),
                        }),
                        else_value: Box::new(Expr::Var(infer_content)),
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
        invalid_tool,
        Block {
            params: vec![],
            instructions: vec![Instr::Let {
                out: invalid_message.clone(),
                expr: Expr::Value(serde_json::json!({
                    "ok": false,
                    "error": "unknown_tool",
                    "message": "unknown tool; available tools: shell, infer"
                })),
            }],
            terminator: Terminator::Goto {
                block: append_tool,
                args: vec![Expr::ToString {
                    value: Box::new(Expr::Var(invalid_message.clone())),
                }],
            },
        },
    );

    blocks.insert(
        invalid_arguments,
        Block {
            params: vec![],
            instructions: vec![Instr::Let {
                out: invalid_message.clone(),
                expr: Expr::Value(serde_json::json!({
                    "ok": false,
                    "error": "invalid_arguments",
                    "message": "tool requires non-empty string arguments"
                })),
            }],
            terminator: Terminator::Goto {
                block: append_tool,
                args: vec![Expr::ToString {
                    value: Box::new(Expr::Var(invalid_message.clone())),
                }],
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
    use crate::gc::GcMode;
    use crate::hydration::{PassiveHydrationConfig, SourceRegistry};
    use crate::interpreter::{EvalConfig, SeqConfig};
    use crate::ir::validate_program;
    use crate::op::{ChatMessage, FinishReason, Prompt, Response, ToolCall};
    use crate::provider::{ChatProvider, ToolSpec};
    use crate::trace::{Event, TraceLogger, TraceSummary};
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
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

    fn response(content: &str, tool_calls: Vec<ToolCall>) -> Response {
        response_with_finish(content, tool_calls, Some(FinishReason::Stop))
    }

    fn response_with_finish(
        content: &str,
        tool_calls: Vec<ToolCall>,
        finish_reason: Option<FinishReason>,
    ) -> Response {
        Response {
            content: content.into(),
            tool_calls,
            finish_reason,
            input_tokens: 0,
            output_tokens: 1,
            total_tokens: 1,
            metadata: Default::default(),
        }
    }

    fn test_trace() -> TraceLogger {
        TraceLogger::new(
            Uuid::new_v4().to_string(),
            std::env::temp_dir().join(format!("agent-ir-loop-{}.jsonl", Uuid::new_v4())),
        )
    }

    fn config(provider: Arc<dyn ChatProvider>) -> SeqConfig {
        config_with_trace(provider, test_trace())
    }

    fn config_with_trace(provider: Arc<dyn ChatProvider>, trace: TraceLogger) -> SeqConfig {
        SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace,
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
                vec![ToolCall::new(
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
            crate::ir_interpreter::run_ir_sequential(&config(provider.clone()), machine).await?;

        assert_eq!(value["content"], Value::String("done".into()));
        // The tool result fed back to the model must be the sub-response
        // text, not the serialized Response envelope.
        let prompts = provider.prompts();
        let tool_message = prompts
            .last()
            .unwrap()
            .iter()
            .find(|message| message.role == "tool")
            .expect("infer tool result in final prompt");
        assert_eq!(tool_message.content.as_deref(), Some("sub answer"));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_turn_budget_exhaustion_annotates_response() -> Result<()> {
        // A model that always answers finish=stop + a tool call never reaches
        // can_stop; with max_turns=2 the loop must return via no_turns_left
        // and annotate the response so the runtime can tell a budget stop
        // from a natural one (t-1133).
        let tool_turn = || {
            response(
                "",
                vec![ToolCall::new(
                    "call-budget",
                    "shell",
                    serde_json::json!({ "command": "printf spin" }),
                )],
            )
        };
        let provider = Arc::new(MockProvider::new(vec![
            tool_turn(),
            tool_turn(),
            tool_turn(),
        ]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("spin")],
            2,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(
            value["metadata"]["stop_reason"],
            Value::String("turn_budget_exhausted".into()),
            "budget stop must be annotated: {value}"
        );
        assert!(
            !value["tool_calls"].as_array().unwrap().is_empty(),
            "the unexecuted tool calls must survive in the response"
        );
        // The annotated value still decodes as a Response.
        let decoded: Response = serde_json::from_value(value)?;
        assert_eq!(
            decoded.metadata.get("stop_reason").and_then(Value::as_str),
            Some("turn_budget_exhausted")
        );
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_natural_stop_has_no_budget_metadata() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("done", vec![])]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("hi")],
            2,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(value["content"], Value::String("done".into()));
        assert!(
            value.get("metadata").is_none(),
            "natural stop must not carry budget metadata: {value}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_nudges_non_stop_turn_without_tool_calls() -> Result<()> {
        // Parity with the Op loop: a non-stop response with no tool calls must
        // not loop on the same context — the assistant text plus a synthetic
        // user continuation nudge are appended before the next infer.
        let provider = Arc::new(MockProvider::new(vec![
            response_with_finish("partial answer", vec![], Some(FinishReason::Length)),
            response("done", vec![]),
        ]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("work")],
            4,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider.clone()), machine).await?;

        assert_eq!(value["content"], Value::String("done".into()));
        let prompts = provider.prompts();
        assert_eq!(prompts.len(), 2);
        let second = &prompts[1];
        let assistant = second
            .iter()
            .find(|message| message.role == "assistant")
            .expect("assistant turn carried into nudged prompt");
        assert_eq!(assistant.content.as_deref(), Some("partial answer"));
        assert!(
            assistant.tool_calls.is_none(),
            "nudged assistant message must not carry a tool_calls field"
        );
        let last = second.last().expect("nudged prompt is non-empty");
        assert_eq!(last.role, "user");
        assert_eq!(last.content.as_deref(), Some(crate::op::CONTINUE_NUDGE));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_tool_turn_normalizes_empty_content_to_null() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
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
            crate::ir_interpreter::run_ir_sequential(&config(provider.clone()), machine).await?;

        assert_eq!(value["content"], Value::String("done".into()));
        let prompts = provider.prompts();
        let assistant = prompts[1]
            .iter()
            .find(|message| message.role == "assistant")
            .expect("assistant tool-call turn in second prompt");
        assert_eq!(
            assistant.content, None,
            "empty assistant text must be null, not an empty string"
        );
        assert!(assistant
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty()));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_malformed_tool_call_does_not_abort_turn() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new("call-1", "shell", serde_json::json!({}))],
            ),
            response("recovered", vec![]),
        ]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("bad tool")],
            4,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(value["content"], Value::String("recovered".into()));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_executes_shell_tool_then_finishes() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
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

    #[tokio::test]
    async fn agent_loop_ir_stop_with_tool_call_dispatches_before_finishing() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "shell",
                    serde_json::json!({ "command": "printf ir-loop" }),
                )],
            ),
            response("done", vec![]),
        ]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![
                ChatMessage::system("system"),
                ChatMessage::user("use shell"),
            ],
            4,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config_with_trace(provider, trace), machine)
                .await?;

        let events = TraceLogger::read_events(trace_path).await?;
        let summary = TraceSummary::from_events(&events);
        assert_eq!(summary.eval_calls, 1);
        assert!(!events.iter().any(|event| matches!(
            event,
            Event::Custom { name, .. } if name == "agent_complete"
        )));
        assert_eq!(value["content"], Value::String("done".into()));
        Ok(())
    }
}
