use crate::ir::{
    Block, BlockId, Expr, Instr, Machine, Program, ProgramId, PromptRef, Terminator, Var,
};
use crate::op::{Model, Prompt};
use serde_json::Value;
use std::collections::BTreeMap;

pub fn agent_loop_ir(model: Model, prompt: Prompt, max_turns: usize) -> Machine {
    agent_loop_ir_with_options(model, prompt, max_turns, false)
}

/// The agent loop with the model-initiated memory tools (docs/MEMORY.md
/// settled question 6) toggled by `memory_tools`. Including the tools
/// changes the program (and so its hash) — callers decide based on whether
/// a memory backend is registered, and the plain [`agent_loop_ir`] stays
/// byte-identical for existing fixtures.
pub fn agent_loop_ir_with_options(
    model: Model,
    prompt: Prompt,
    max_turns: usize,
    memory_tools: bool,
) -> Machine {
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
    let remember_dispatch = BlockId(18);
    let remember_tool = BlockId(19);
    let recall_dispatch = BlockId(20);
    let recall_tool = BlockId(21);
    let remember_store = BlockId(22);
    let recall_retrieve = BlockId(23);

    let history = Var("history".into());
    let turns_left = Var("turns_left".into());
    let response = Var("response".into());
    let tool_calls = Var("tool_calls".into());
    let no_tool_calls = Var("no_tool_calls".into());
    let finish_reason = Var("finish_reason".into());
    let is_truncated = Var("is_truncated".into());
    let not_truncated = Var("not_truncated".into());
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
    let is_remember_tool = Var("is_remember_tool".into());
    let is_recall_tool = Var("is_recall_tool".into());
    let memory_name = Var("memory_name".into());
    let memory_content = Var("memory_content".into());
    let missing_memory_name = Var("missing_memory_name".into());
    let missing_memory_content = Var("missing_memory_content".into());
    let invalid_memory_arguments = Var("invalid_memory_arguments".into());
    let memory_item = Var("memory_item".into());
    let stored_id = Var("stored_id".into());
    let recall_query = Var("recall_query".into());
    let missing_recall_query = Var("missing_recall_query".into());
    let recall_hits = Var("recall_hits".into());

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
                // Claude-Code-style turn completion (t-1134): "done" is the
                // ABSENCE of pending tool calls, not a token or a counter. A
                // response carrying tool calls is never terminal; a response
                // without them ends the turn. finish_reason is demoted to a
                // truncation hint: only positive evidence of truncation
                // ("length") routes to the continuation nudge instead of
                // ending the turn. The turn budget (max_turns) is a pure
                // safety ceiling — its exhaustion path is the annotated
                // budget_done branch, never the normal exit.
                Instr::Let {
                    out: is_truncated.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(finish_reason)),
                        right: Box::new(Expr::Value(Value::String("length".into()))),
                    },
                },
                Instr::Let {
                    out: not_truncated.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(is_truncated)),
                        right: Box::new(Expr::Value(Value::Bool(false))),
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
                            left: Box::new(Expr::Var(no_tool_calls.clone())),
                            right: Box::new(Expr::Var(not_truncated)),
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
                else_block: if memory_tools {
                    remember_dispatch
                } else {
                    invalid_tool
                },
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
                    "message": if memory_tools {
                        "unknown tool; available tools: shell, infer, remember, recall"
                    } else {
                        "unknown tool; available tools: shell, infer"
                    }
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
                        ("content".into(), Expr::Var(tool_content.clone())),
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

    if memory_tools {
        blocks.insert(
            remember_dispatch,
            Block {
                params: vec![],
                instructions: vec![Instr::Let {
                    out: is_remember_tool.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(function_name.clone())),
                        right: Box::new(Expr::Value(Value::String("remember".into()))),
                    },
                }],
                terminator: Terminator::If {
                    cond: Expr::Var(is_remember_tool),
                    then_block: remember_tool,
                    else_block: recall_dispatch,
                },
            },
        );
        blocks.insert(
            remember_tool,
            Block {
                params: vec![],
                instructions: vec![
                    Instr::Let {
                        out: memory_name.clone(),
                        expr: Expr::StringOr {
                            value: Box::new(Expr::FieldOr {
                                base: arguments.clone(),
                                field: "name".into(),
                                default: Box::new(Expr::Value(Value::String("".into()))),
                            }),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        },
                    },
                    Instr::Let {
                        out: memory_content.clone(),
                        expr: Expr::StringOr {
                            value: Box::new(Expr::FieldOr {
                                base: arguments.clone(),
                                field: "content".into(),
                                default: Box::new(Expr::Value(Value::String("".into()))),
                            }),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        },
                    },
                    Instr::Let {
                        out: missing_memory_name.clone(),
                        expr: Expr::IsEmpty {
                            base: memory_name.clone(),
                        },
                    },
                    Instr::Let {
                        out: missing_memory_content.clone(),
                        expr: Expr::IsEmpty {
                            base: memory_content.clone(),
                        },
                    },
                    Instr::Let {
                        out: invalid_memory_arguments.clone(),
                        expr: Expr::Or {
                            left: Box::new(Expr::Var(missing_memory_name)),
                            right: Box::new(Expr::Var(missing_memory_content.clone())),
                        },
                    },
                ],
                terminator: Terminator::If {
                    cond: Expr::Var(invalid_memory_arguments),
                    then_block: invalid_arguments,
                    else_block: remember_store,
                },
            },
        );
        blocks.insert(
            remember_store,
            Block {
                params: vec![],
                instructions: vec![
                    // The tool schema maps onto the memory sink's payload
                    // schema; description defaults to the empty string and
                    // type passes through when given.
                    Instr::Let {
                        out: memory_item.clone(),
                        expr: Expr::Object(BTreeMap::from([
                            ("name".into(), Expr::Var(memory_name.clone())),
                            (
                                "description".into(),
                                Expr::StringOr {
                                    value: Box::new(Expr::FieldOr {
                                        base: arguments.clone(),
                                        field: "description".into(),
                                        default: Box::new(Expr::Value(Value::String("".into()))),
                                    }),
                                    default: Box::new(Expr::Value(Value::String("".into()))),
                                },
                            ),
                            (
                                "type".into(),
                                Expr::FieldOr {
                                    base: arguments.clone(),
                                    field: "type".into(),
                                    default: Box::new(Expr::Value(Value::Null)),
                                },
                            ),
                            ("body".into(), Expr::Var(memory_content.clone())),
                        ])),
                    },
                    Instr::Store {
                        out: stored_id.clone(),
                        sink: Expr::Value(Value::String("memory".into())),
                        op: crate::ir::StoreOp::Create,
                        id: None,
                        item: Expr::Var(memory_item),
                        policy: Default::default(),
                    },
                    Instr::Let {
                        out: tool_content.clone(),
                        expr: Expr::ToString {
                            value: Box::new(Expr::Var(stored_id)),
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
            recall_dispatch,
            Block {
                params: vec![],
                instructions: vec![Instr::Let {
                    out: is_recall_tool.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(function_name.clone())),
                        right: Box::new(Expr::Value(Value::String("recall".into()))),
                    },
                }],
                terminator: Terminator::If {
                    cond: Expr::Var(is_recall_tool),
                    then_block: recall_tool,
                    else_block: invalid_tool,
                },
            },
        );
        blocks.insert(
            recall_tool,
            Block {
                params: vec![],
                instructions: vec![
                    Instr::Let {
                        out: recall_query.clone(),
                        expr: Expr::StringOr {
                            value: Box::new(Expr::FieldOr {
                                base: arguments.clone(),
                                field: "query".into(),
                                default: Box::new(Expr::Value(Value::String("".into()))),
                            }),
                            default: Box::new(Expr::Value(Value::String("".into()))),
                        },
                    },
                    Instr::Let {
                        out: missing_recall_query.clone(),
                        expr: Expr::IsEmpty {
                            base: recall_query.clone(),
                        },
                    },
                ],
                terminator: Terminator::If {
                    cond: Expr::Var(missing_recall_query),
                    then_block: invalid_arguments,
                    else_block: recall_retrieve,
                },
            },
        );
        blocks.insert(
            recall_retrieve,
            Block {
                params: vec![],
                instructions: vec![
                    Instr::Retrieve {
                        out: recall_hits.clone(),
                        query: Expr::Var(recall_query),
                        kind: Some(crate::hydration::SourceKind::Semantic),
                        max_bytes: Some(16 * 1024),
                    },
                    Instr::Let {
                        out: tool_content.clone(),
                        expr: Expr::ToString {
                            value: Box::new(Expr::Var(recall_hits)),
                        },
                    },
                ],
                terminator: Terminator::Goto {
                    block: append_tool,
                    args: vec![Expr::Var(tool_content.clone())],
                },
            },
        );
    } else {
        let _ = (
            remember_dispatch,
            remember_tool,
            recall_dispatch,
            recall_tool,
            remember_store,
            recall_retrieve,
            is_remember_tool,
            is_recall_tool,
            memory_name,
            memory_content,
            missing_memory_name,
            missing_memory_content,
            invalid_memory_arguments,
            memory_item,
            stored_id,
            recall_query,
            missing_recall_query,
            recall_hits,
        );
    }

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
    use crate::gc::GcTiming;
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
            trace,
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: GcTiming::Threshold,
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
    async fn agent_loop_ir_ends_turn_on_tool_call_free_response_without_finish_reason() -> Result<()>
    {
        // t-1134: "done" is the absence of tool calls, not a finish_reason
        // token. A provider that omits finish_reason entirely must still end
        // the turn instead of burning the budget on nudges. MockProvider has
        // exactly one response: a second infer would error.
        let provider = Arc::new(MockProvider::new(vec![response_with_finish(
            "final answer",
            vec![],
            None,
        )]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("hi")],
            4,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(value["content"], Value::String("final answer".into()));
        assert!(value.get("metadata").is_none(), "natural end_turn: {value}");
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_ends_turn_on_unrecognized_finish_reason() -> Result<()> {
        // Only positive truncation evidence ("length") keeps the turn alive;
        // anything else with no tool calls is a final answer.
        let provider = Arc::new(MockProvider::new(vec![response_with_finish(
            "filtered but final",
            vec![],
            Some(FinishReason::ContentFilter),
        )]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("hi")],
            4,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(value["content"], Value::String("filtered but final".into()));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_never_treats_stop_with_tool_calls_as_terminal() -> Result<()> {
        // The June-9 'finalize on stop+tool_calls' bug, made structurally
        // unrepresentable (t-1134): a response carrying a tool call always
        // executes and loops, whatever its finish_reason says.
        let provider = Arc::new(MockProvider::new(vec![
            response_with_finish(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "shell",
                    serde_json::json!({ "command": "printf t1134" }),
                )],
                Some(FinishReason::Stop),
            ),
            response("done after tool", vec![]),
        ]));
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("run it")],
            4,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider), machine).await?;

        assert_eq!(
            value["content"],
            Value::String("done after tool".into()),
            "the tool-call turn must execute and loop, not finalize: {value}"
        );
        assert!(value.get("metadata").is_none());
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

    fn memory_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("agent-loop-memory-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn memory_config(provider: Arc<dyn ChatProvider>, dir: std::path::PathBuf) -> SeqConfig {
        let mut config = config(provider);
        config.hydration =
            SourceRegistry::new().register_backend(crate::memory::MemorySource::new(dir));
        config
    }

    #[tokio::test]
    async fn agent_loop_ir_remember_then_recall_round_trips() -> Result<()> {
        let dir = memory_dir();
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "remember",
                    serde_json::json!({
                        "name": "deploy-window",
                        "description": "when deploys are allowed",
                        "type": "project",
                        "content": "deploys only on tuesdays",
                    }),
                )],
            ),
            response(
                "",
                vec![ToolCall::new(
                    "call-2",
                    "recall",
                    serde_json::json!({ "query": "when can we deploy" }),
                )],
            ),
            response("deploys happen on tuesdays", vec![]),
        ]));
        let config = memory_config(provider.clone(), dir.clone());
        let machine = agent_loop_ir_with_options(
            Model("mock".into()),
            vec![ChatMessage::user("note the deploy window, then check it")],
            8,
            true,
        );

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
        let response: Response = serde_json::from_value(value)?;
        assert_eq!(response.content, "deploys happen on tuesdays");

        // The remember call wrote a real memory file...
        assert!(dir.join("deploy-window.md").exists());
        // ...the remember tool result echoed the sink id...
        let prompts = provider.prompts();
        let remember_result = prompts[1]
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("remember tool result fed back");
        assert!(
            remember_result
                .content
                .as_deref()
                .unwrap()
                .contains("deploy-window"),
            "{remember_result:?}"
        );
        // ...and the recall tool result carried the stored fact back in.
        let recall_result = prompts[2]
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-2"))
            .expect("recall tool result fed back");
        assert!(
            recall_result
                .content
                .as_deref()
                .unwrap()
                .contains("deploys only on tuesdays"),
            "{recall_result:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_remember_with_missing_arguments_does_not_abort_turn() -> Result<()> {
        let dir = memory_dir();
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "remember",
                    serde_json::json!({ "content": "fact without a name" }),
                )],
            ),
            response("noted the problem", vec![]),
        ]));
        let config = memory_config(provider.clone(), dir.clone());
        let machine = agent_loop_ir_with_options(
            Model("mock".into()),
            vec![ChatMessage::user("remember badly")],
            4,
            true,
        );

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
        let response: Response = serde_json::from_value(value)?;
        assert_eq!(response.content, "noted the problem");

        let prompts = provider.prompts();
        let tool_result = prompts[1]
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("invalid-arguments result fed back");
        assert!(
            tool_result
                .content
                .as_deref()
                .unwrap()
                .contains("invalid_arguments"),
            "{tool_result:?}"
        );
        assert!(
            std::fs::read_dir(&dir)?.next().is_none(),
            "nothing was written"
        );
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_without_memory_tools_treats_remember_as_unknown() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "remember",
                    serde_json::json!({ "name": "x", "content": "y" }),
                )],
            ),
            response("ok", vec![]),
        ]));
        let config = config(provider.clone());
        let machine = agent_loop_ir(
            Model("mock".into()),
            vec![ChatMessage::user("hallucinate a tool")],
            4,
        );

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
        let response: Response = serde_json::from_value(value)?;
        assert_eq!(response.content, "ok");

        let prompts = provider.prompts();
        let tool_result = prompts[1]
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("unknown-tool result fed back");
        assert!(
            tool_result
                .content
                .as_deref()
                .unwrap()
                .contains("unknown_tool"),
            "{tool_result:?}"
        );
        Ok(())
    }

    #[test]
    fn memory_tools_only_change_the_program_when_enabled() {
        let plain = agent_loop_ir(Model("m".into()), vec![], 4);
        let plain_again = agent_loop_ir_with_options(Model("m".into()), vec![], 4, false);
        assert_eq!(
            crate::ir::program_hash(&plain.program).unwrap(),
            crate::ir::program_hash(&plain_again.program).unwrap(),
            "the default loop is byte-identical with tools off"
        );
        let with_tools = agent_loop_ir_with_options(Model("m".into()), vec![], 4, true);
        assert_ne!(
            crate::ir::program_hash(&plain.program).unwrap(),
            crate::ir::program_hash(&with_tools.program).unwrap(),
        );
        validate_program(&with_tools.program).expect("tool variant validates");
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
