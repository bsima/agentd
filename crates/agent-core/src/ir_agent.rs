use crate::gc::GcState;
use crate::interpreter::SeqConfig;
use crate::ir::{
    Block, BlockId, EffectErrorMode, Expr, InferPolicy, Instr, Machine, Program, ProgramId,
    PromptRef, RetrievePolicy, StorePolicy, Terminator, ToolPolicy, Var,
};
use crate::ir_interpreter::{IrReplayTrace, IrStore};
use crate::op::{Model, Prompt};
use crate::output_contract::{
    OutputContract, OutputContractFailure, MAX_TRACE_ERRORS, OUTPUT_CONTRACT_EVENT,
    OUTPUT_VALIDATION_FAILED_EVENT,
};
use crate::trace::Event;
use anyhow::{anyhow, Result};
use chrono::Utc;
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
    agent_loop_ir_with_tools(model, prompt, max_turns, memory_tools, &[])
}

/// The agent loop with native tool dispatch arms (t-1308.7). Each name in
/// `native_tools` — the registered [`crate::tool::ToolRegistry`] names, in
/// registry order — gets its own dispatch arm ahead of the unknown-tool
/// fallthrough, executing as an [`Instr::Tool`] effect with `on_error:
/// Bind` (a failed handler is a tool result the model can read, not a turn
/// abort — t-1222). Like the memory tools, the tool set is part of the
/// program: changing it changes the program hash, so replay against a
/// different tool set diverges. Dispatch is by exact name match, entirely
/// apart from the shell arm: a native tool call is never executed via
/// `$SHELL -c`.
pub fn agent_loop_ir_with_tools(
    model: Model,
    prompt: Prompt,
    max_turns: usize,
    memory_tools: bool,
    native_tools: &[String],
) -> Machine {
    agent_loop_ir_with_policies(model, prompt, max_turns, memory_tools, native_tools, false)
}

/// The agent loop with per-effect policy knobs (t-1308.10). Today the one
/// knob is `shell_requires_approval`: gate the shell tool's Eval behind the
/// approval protocol (DR-7) by setting `require_approval` on its
/// [`crate::ir::EvalPolicy`]. Gating is part of the program — enabling it
/// changes the program hash, so replay against a differently-gated loop
/// diverges; the ungated loop hashes identically to before the field
/// existed (the policy field serializes only when true).
pub fn agent_loop_ir_with_policies(
    model: Model,
    prompt: Prompt,
    max_turns: usize,
    memory_tools: bool,
    native_tools: &[String],
    shell_requires_approval: bool,
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
    // Native tool dispatch arms (t-1308.7): a dispatch/body block pair per
    // registered tool, from block id 24 up.
    let native_base = 24u32;
    let native_dispatch = |index: usize| BlockId(native_base + 2 * index as u32);
    let native_body = |index: usize| BlockId(native_base + 2 * index as u32 + 1);
    // Where an unmatched name falls through to after the built-in arms:
    // the first native dispatch arm when tools are registered, else the
    // unknown-tool response.
    let first_native_or_invalid = if native_tools.is_empty() {
        invalid_tool
    } else {
        native_dispatch(0)
    };

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
    let missing_memory_content = Var("missing_memory_content".into());
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
                then_args: vec![],
                else_block: route,
                else_args: vec![],
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
                then_args: vec![],
                else_block: budget_done,
                else_args: vec![],
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
                then_args: vec![],
                else_block: prepare_tools,
                else_args: vec![],
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
                then_args: vec![],
                else_block: next_turn,
                else_args: vec![],
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
                then_args: vec![],
                else_block: shell_dispatch,
                else_args: vec![],
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
                then_args: vec![],
                else_block: if memory_tools {
                    remember_dispatch
                } else {
                    first_native_or_invalid
                },
                else_args: vec![],
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
                then_args: vec![],
                else_block: shell_eval,
                else_args: vec![],
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
                    policy: crate::ir::EvalPolicy {
                        require_approval: shell_requires_approval,
                        ..Default::default()
                    },
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
                then_args: vec![],
                else_block: infer_eval,
                else_args: vec![],
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
                    // Errors-as-values (t-1222): a bad sub-infer model (e.g.
                    // a hallucinated id) becomes a tool result the model can
                    // recover from, not a turn-aborting error.
                    policy: InferPolicy {
                        on_error: EffectErrorMode::Bind,
                    },
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

    let available_tools = {
        let mut names: Vec<&str> = vec!["shell", "infer"];
        if memory_tools {
            names.push("remember");
            names.push("recall");
        }
        names.extend(native_tools.iter().map(String::as_str));
        names.join(", ")
    };
    blocks.insert(
        invalid_tool,
        Block {
            params: vec![],
            instructions: vec![Instr::Let {
                out: invalid_message.clone(),
                expr: Expr::Value(serde_json::json!({
                    "ok": false,
                    "error": "unknown_tool",
                    "message": format!("unknown tool; available tools: {available_tools}")
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
                    then_args: vec![],
                    else_block: recall_dispatch,
                    else_args: vec![],
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
                    // Only `content` is required (docs/MEMORY.md tool surface
                    // remember {content, name?, type?}): an absent name is
                    // slugged from the description/body by the memory sink.
                    Instr::Let {
                        out: missing_memory_content.clone(),
                        expr: Expr::IsEmpty {
                            base: memory_content.clone(),
                        },
                    },
                ],
                terminator: Terminator::If {
                    cond: Expr::Var(missing_memory_content.clone()),
                    then_block: invalid_arguments,
                    then_args: vec![],
                    else_block: remember_store,
                    else_args: vec![],
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
                        // Errors-as-values (t-1222): a rejected write (e.g. a
                        // duplicate slug) becomes a tool result, not a fatal turn.
                        policy: StorePolicy {
                            on_error: EffectErrorMode::Bind,
                        },
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
                    then_args: vec![],
                    else_block: first_native_or_invalid,
                    else_args: vec![],
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
                    then_args: vec![],
                    else_block: recall_retrieve,
                    else_args: vec![],
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
                        // Errors-as-values (t-1222): a failed recall becomes a
                        // tool result, not a turn-aborting error.
                        policy: RetrievePolicy {
                            on_error: EffectErrorMode::Bind,
                        },
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
            missing_memory_content,
            memory_item,
            stored_id,
            recall_query,
            missing_recall_query,
            recall_hits,
        );
    }

    let is_native_tool = Var("is_native_tool".into());
    let native_tool_result = Var("native_tool_result".into());
    for (index, tool_name) in native_tools.iter().enumerate() {
        let fallthrough = if index + 1 < native_tools.len() {
            native_dispatch(index + 1)
        } else {
            invalid_tool
        };
        blocks.insert(
            native_dispatch(index),
            Block {
                params: vec![],
                instructions: vec![Instr::Let {
                    out: is_native_tool.clone(),
                    expr: Expr::Eq {
                        left: Box::new(Expr::Var(function_name.clone())),
                        right: Box::new(Expr::Value(Value::String(tool_name.clone()))),
                    },
                }],
                terminator: Terminator::If {
                    cond: Expr::Var(is_native_tool.clone()),
                    then_block: native_body(index),
                    then_args: vec![],
                    else_block: fallthrough,
                    else_args: vec![],
                },
            },
        );
        blocks.insert(
            native_body(index),
            Block {
                params: vec![],
                instructions: vec![
                    Instr::Tool {
                        out: native_tool_result.clone(),
                        name: tool_name.clone(),
                        arguments: Expr::Var(arguments.clone()),
                        // Errors-as-values (t-1222): a failed handler becomes
                        // a tool result the model can recover from.
                        policy: ToolPolicy {
                            on_error: EffectErrorMode::Bind,
                        },
                    },
                    // String results feed back verbatim; anything else is
                    // serialized JSON (a bare ToString of a string would
                    // hand the model a quoted literal).
                    Instr::Let {
                        out: tool_content.clone(),
                        expr: Expr::StringOr {
                            value: Box::new(Expr::Var(native_tool_result.clone())),
                            default: Box::new(Expr::ToString {
                                value: Box::new(Expr::Var(native_tool_result.clone())),
                            }),
                        },
                    },
                ],
                terminator: Terminator::Goto {
                    block: append_tool,
                    args: vec![Expr::Var(tool_content.clone())],
                },
            },
        );
    }
    let _ = (is_native_tool, native_tool_result);

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
        control_path: Default::default(),
        continuation_stack: vec![],
        budgets: Default::default(),
    }
}

/// Loop-level options for [`run_agent_loop`]. The output contract lives
/// here — not in `InferPolicy` — per the SDK PRD decision (DR-2): final
/// output validation is post-processing over the whole loop run, migrated
/// into the IR only when AgentIR can host it cleanly.
#[derive(Debug, Clone, Default)]
pub struct AgentLoopOptions {
    /// Include the model-initiated remember/recall tools (docs/MEMORY.md);
    /// see [`agent_loop_ir_with_options`].
    pub memory_tools: bool,
    /// Registered native tool names (t-1308.7), in registry order — pass
    /// [`crate::tool::ToolRegistry::names`] of the registry carried on the
    /// `SeqConfig`. Each name gets a dispatch arm in the loop program; see
    /// [`agent_loop_ir_with_tools`].
    pub tool_names: Vec<String>,
    /// Validate the loop's final response against a JSON Schema, with
    /// bounded repair (t-1308.4, DR-2).
    pub output_contract: Option<OutputContract>,
    /// Gate the shell tool's Eval behind the approval protocol (t-1308.10,
    /// DR-7): see [`agent_loop_ir_with_policies`]. Off by default; without
    /// a resolver (hook or pre-loaded resolution) a gated shell command
    /// pauses the run instead of executing.
    pub shell_requires_approval: bool,
}

/// Run the agent loop with loop-level post-processing (t-1308.4): build the
/// loop program, execute it, and — when an [`OutputContract`] is present —
/// validate the natural final response as JSON against the schema. Each
/// failure appends a repair turn (a user message quoting the validation
/// errors) and re-enters the loop, up to `max_repairs` times; exhaustion
/// returns an [`OutputContractFailure`] rendered as a value (errors-as-
/// values, t-1222), never a process abort. Detect it with
/// [`crate::output_contract::output_contract_failure`].
///
/// Scope of validation: natural completions only. A turn-budget-exhausted
/// response (`metadata.stop_reason = "turn_budget_exhausted"`) is returned
/// as-is — the turn never produced a final answer to validate.
///
/// `effect_visits` seeds the machine's per-site visit counters (thread the
/// returned machine's counters back in on the next session turn, exactly as
/// with the raw builder); repair runs inside this call carry them forward
/// automatically, so effect ids stay unique across repairs.
///
/// Replay: when a contract is present its `schema_hash` is recorded in the
/// trace (a Custom `output_contract` event). Replaying against a trace whose
/// recorded hash differs from the current contract's (including one side
/// having no contract at all) fails fast with a divergence error.
#[allow(clippy::too_many_arguments)] // mirrors run_ir_steps_with_gc's seams: config, state, and loop identity are distinct axes
pub async fn run_agent_loop(
    config: &SeqConfig,
    store: &mut dyn IrStore,
    ir_replay: Option<&IrReplayTrace>,
    gc_state: &mut GcState,
    model: Model,
    prompt: Prompt,
    max_turns: usize,
    options: &AgentLoopOptions,
    effect_visits: BTreeMap<String, u64>,
) -> Result<(Value, Machine)> {
    match run_agent_loop_outcome(
        config,
        store,
        ir_replay,
        gc_state,
        model,
        prompt,
        max_turns,
        options,
        effect_visits,
    )
    .await?
    {
        AgentLoopOutcome::Complete { value, machine } => Ok((value, machine)),
        // This entry point cannot suspend, so a pause with no resolver is
        // the fail-closed error: the gated effect did not execute (DR-7).
        AgentLoopOutcome::AwaitingApproval { pending, .. } => {
            Err(crate::ir_interpreter::awaiting_approval_error(&pending))
        }
    }
}

/// The outcome of one agent-loop run for drivers that can pause (t-1308.10).
#[derive(Debug, Clone, PartialEq)]
pub enum AgentLoopOutcome {
    /// The loop ran to completion (including any output-contract repairs).
    Complete { value: Value, machine: Machine },
    /// An approval-gated effect was reached with no decision available: the
    /// machine checkpointed mid-turn without executing it. Persist the
    /// checkpoint alongside a [`crate::approval::PendingEffectRecord`]
    /// (see [`crate::approval::ApprovalStore`]) and, once resolved, re-enter
    /// it with [`resume_agent_loop_outcome`].
    AwaitingApproval {
        checkpoint: crate::ir_interpreter::IrCheckpoint,
        pending: crate::approval::ApprovalRequest,
    },
}

/// [`run_agent_loop`] for drivers that can pause on approval gates: instead
/// of failing closed with an error, a gated effect with no resolver returns
/// [`AgentLoopOutcome::AwaitingApproval`] carrying the mid-turn checkpoint.
#[allow(clippy::too_many_arguments)] // mirrors run_agent_loop's axes
pub async fn run_agent_loop_outcome(
    config: &SeqConfig,
    store: &mut dyn IrStore,
    ir_replay: Option<&IrReplayTrace>,
    gc_state: &mut GcState,
    model: Model,
    prompt: Prompt,
    max_turns: usize,
    options: &AgentLoopOptions,
    effect_visits: BTreeMap<String, u64>,
) -> Result<AgentLoopOutcome> {
    if let Some(replay) = ir_replay {
        check_output_schema_hash(replay, options.output_contract.as_ref())?;
    }
    if let Some(contract) = &options.output_contract {
        // Run-identity metadata: the schema hash rides in the trace the way
        // program_hash rides in every effect id, so replay can detect a
        // changed contract (checked above on the replay side).
        config
            .trace
            .emit(&Event::Custom {
                run_id: config.trace.run_id().into(),
                name: OUTPUT_CONTRACT_EVENT.into(),
                data: serde_json::json!({
                    "schema_hash": contract.schema_hash(),
                    "max_repairs": contract.max_repairs,
                }),
                timestamp: Utc::now(),
            })
            .await?;
    }

    let mut machine = agent_loop_ir_with_policies(
        model.clone(),
        prompt,
        max_turns,
        options.memory_tools,
        &options.tool_names,
        options.shell_requires_approval,
    );
    machine.effect_visits = effect_visits;
    drive_agent_loop(
        config, store, ir_replay, gc_state, model, max_turns, options, machine,
    )
    .await
}

/// Re-enter an approval pause (t-1308.10): run the checkpointed machine —
/// program counter still at the gated instruction — to completion or to the
/// next pause. The caller seeds `store` from the checkpoint's store
/// snapshot and loads the durable decision into
/// [`crate::approval::ApprovalConfig::resolutions`] on `config` (see
/// [`crate::approval::ApprovalStore::resolution_of`]); the gate consumes it
/// at the effect site, executing the effect (approved) or binding the typed
/// denial value (denied). `model`/`max_turns`/`options` must match the
/// original run: they rebuild the loop for output-contract repair turns.
#[allow(clippy::too_many_arguments)] // mirrors run_agent_loop's axes
pub async fn resume_agent_loop_outcome(
    config: &SeqConfig,
    store: &mut dyn IrStore,
    gc_state: &mut GcState,
    model: Model,
    max_turns: usize,
    options: &AgentLoopOptions,
    machine: Machine,
) -> Result<AgentLoopOutcome> {
    // No replay: a resume is a live continuation of a live run. The
    // output-contract identity event was already emitted by the run that
    // paused, so the drive loop is entered directly.
    drive_agent_loop(
        config, store, None, gc_state, model, max_turns, options, machine,
    )
    .await
}

/// The agent loop's execute/validate/repair cycle, shared by fresh runs and
/// approval resumes. Repair attempts count from zero per entry — a resumed
/// run's earlier repair attempts are not carried across the pause (wave-1
/// simplification; the contract budget still bounds each entry).
#[allow(clippy::too_many_arguments)] // mirrors run_agent_loop's axes
async fn drive_agent_loop(
    config: &SeqConfig,
    store: &mut dyn IrStore,
    ir_replay: Option<&IrReplayTrace>,
    gc_state: &mut GcState,
    model: Model,
    max_turns: usize,
    options: &AgentLoopOptions,
    mut machine: Machine,
) -> Result<AgentLoopOutcome> {
    let mut attempt = 0usize;
    loop {
        let (value, done) = match crate::ir_interpreter::run_ir_steps_with_gc(
            config, machine, store, ir_replay, None, gc_state,
        )
        .await?
        {
            crate::ir_interpreter::IrStepOutcome::Complete { value, machine } => (value, machine),
            crate::ir_interpreter::IrStepOutcome::AwaitingApproval {
                checkpoint,
                pending,
            } => {
                return Ok(AgentLoopOutcome::AwaitingApproval {
                    checkpoint,
                    pending,
                })
            }
            crate::ir_interpreter::IrStepOutcome::Suspended { .. } => {
                unreachable!("no instruction limit was set")
            }
        };
        let Some(contract) = &options.output_contract else {
            return Ok(AgentLoopOutcome::Complete {
                value,
                machine: done,
            });
        };
        if turn_budget_exhausted(&value) {
            return Ok(AgentLoopOutcome::Complete {
                value,
                machine: done,
            });
        }
        let content = value
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let errors = contract.validate_text(&content);
        if errors.is_empty() {
            return Ok(AgentLoopOutcome::Complete {
                value,
                machine: done,
            });
        }
        attempt += 1;
        config
            .trace
            .emit(&Event::Custom {
                run_id: config.trace.run_id().into(),
                name: OUTPUT_VALIDATION_FAILED_EVENT.into(),
                data: serde_json::json!({
                    "attempt": attempt,
                    "errors": errors.iter().take(MAX_TRACE_ERRORS).collect::<Vec<_>>(),
                    "preview": crate::trace::preview(&content, 256),
                }),
                timestamp: Utc::now(),
            })
            .await?;
        if attempt > contract.max_repairs {
            let failure = OutputContractFailure {
                attempts: attempt,
                errors,
                content,
            };
            return Ok(AgentLoopOutcome::Complete {
                value: failure.into_value(),
                machine: done,
            });
        }
        machine = repair_machine(&done, &content, &errors, model.clone(), max_turns, options);
    }
}

fn turn_budget_exhausted(value: &Value) -> bool {
    value
        .get("metadata")
        .and_then(|metadata| metadata.get("stop_reason"))
        .and_then(Value::as_str)
        == Some("turn_budget_exhausted")
}

/// A fresh loop machine for one repair turn: the completed machine's
/// history plus the invalid assistant answer and a user repair message,
/// with a full turn budget and the effect-visit counters carried forward
/// (same shape as a new session turn in the runtime).
fn repair_machine(
    done: &Machine,
    content: &str,
    errors: &[String],
    model: Model,
    max_turns: usize,
    options: &AgentLoopOptions,
) -> Machine {
    let mut history = done
        .env
        .get(&Var("history".into()))
        .cloned()
        .unwrap_or_else(|| Value::Array(vec![]));
    if let Value::Array(messages) = &mut history {
        if !content.is_empty() {
            messages.push(serde_json::json!({ "role": "assistant", "content": content }));
        }
        messages.push(serde_json::json!({ "role": "user", "content": repair_prompt(errors) }));
    }
    let mut machine = agent_loop_ir_with_policies(
        model,
        vec![],
        max_turns,
        options.memory_tools,
        &options.tool_names,
        options.shell_requires_approval,
    );
    machine.env.insert(Var("history".into()), history);
    machine.effect_visits = done.effect_visits.clone();
    machine
}

fn repair_prompt(errors: &[String]) -> String {
    let listed = errors
        .iter()
        .take(MAX_TRACE_ERRORS)
        .map(|error| format!("- {error}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Your final response failed validation against the required output schema:\n{listed}\n\n\
         Respond again with ONLY a single JSON value that conforms to the output schema. \
         No prose, no code fences, no explanation."
    )
}

/// Replay-divergence check for the output contract: the recorded
/// `schema_hash` (from the trace's `output_contract` Custom event) must
/// match the current contract's, including presence — a run recorded with a
/// contract cannot be replayed without one, and vice versa.
fn check_output_schema_hash(
    replay: &IrReplayTrace,
    contract: Option<&OutputContract>,
) -> Result<()> {
    let recorded = replay.output_schema_hash();
    let current = contract.map(OutputContract::schema_hash);
    if recorded != current.as_deref() {
        return Err(anyhow!(
            "AgentIR replay diverged: output contract schema_hash mismatch \
             (recorded {}, current {}); replay requires the same output schema \
             the run was recorded with",
            recorded.unwrap_or("none"),
            current.as_deref().unwrap_or("none")
        ));
    }
    Ok(())
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
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
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
            approvals: Default::default(),
            tools: Default::default(),
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
            pricing: Default::default(),
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
    async fn agent_loop_ir_remember_without_name_derives_a_slug() -> Result<()> {
        // The documented surface is remember {content, name?}: a content-only
        // call must succeed, deriving the slug from the description/content
        // (t-1182 review finding #3).
        let dir = memory_dir();
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "remember",
                    serde_json::json!({
                        "content": "deploys only on tuesdays",
                        "description": "Deploy Window",
                    }),
                )],
            ),
            response("noted", vec![]),
        ]));
        let config = memory_config(provider.clone(), dir.clone());
        let machine = agent_loop_ir_with_options(
            Model("mock".into()),
            vec![ChatMessage::user("remember the deploy window")],
            4,
            true,
        );

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
        let response: Response = serde_json::from_value(value)?;
        assert_eq!(response.content, "noted");

        // A memory file was written under the derived slug, and the tool
        // result echoed that slug back (no invalid_arguments).
        assert!(
            dir.join("deploy-window.md").exists(),
            "slug derived from description"
        );
        let prompts = provider.prompts();
        let tool_result = prompts[1]
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("remember tool result fed back");
        let content = tool_result.content.as_deref().unwrap();
        assert!(content.contains("deploy-window"), "{tool_result:?}");
        assert!(!content.contains("invalid_arguments"), "{tool_result:?}");
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_remember_without_content_is_invalid() -> Result<()> {
        // content is the one required field; its absence is the only
        // invalid-arguments case for remember.
        let dir = memory_dir();
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "remember",
                    serde_json::json!({ "name": "no-body" }),
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

    /// Provider that 404s for one model id (a dead/hallucinated model) and
    /// serves queued responses otherwise — mirrors the t-1221 trigger.
    struct DeadModelProvider {
        dead_model: String,
        responses: Mutex<Vec<Response>>,
        prompts: Mutex<Vec<Prompt>>,
    }

    #[async_trait]
    impl ChatProvider for DeadModelProvider {
        async fn chat(
            &self,
            model: &Model,
            _tools: &[ToolSpec],
            messages: &[ChatMessage],
        ) -> Result<Response> {
            if model.0 == self.dead_model {
                return Err(anyhow!(
                    "provider returned 404 Not Found: model: {}",
                    self.dead_model
                ));
            }
            self.prompts.lock().unwrap().push(messages.to_vec());
            self.responses
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow!("mock provider exhausted"))
        }
    }

    #[tokio::test]
    async fn agent_loop_ir_infer_tool_failure_becomes_a_recoverable_tool_result() -> Result<()> {
        // t-1222: the model calls the infer sub-tool with a dead model id.
        // The 404 must become a tool result the model sees, and the turn
        // must continue — not abort the whole run (the t-1221 impact).
        let mut queued = vec![
            // turn 1: call the dead-model infer tool.
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "infer",
                    serde_json::json!({
                        "model": "claude-sonnet-4-20250514",
                        "prompt": "sub-question",
                    }),
                )],
            ),
            // turn 2: having seen the error tool result, finish.
            response("recovered from the bad infer", vec![]),
        ];
        queued.reverse();
        let provider = Arc::new(DeadModelProvider {
            dead_model: "claude-sonnet-4-20250514".into(),
            responses: Mutex::new(queued),
            prompts: Mutex::new(Vec::new()),
        });
        let machine = agent_loop_ir(
            Model("mock-main".into()),
            vec![ChatMessage::user("use infer, then answer")],
            6,
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider.clone()), machine).await?;
        let response: Response = serde_json::from_value(value)?;

        // The run completed instead of aborting on the 404.
        assert_eq!(response.content, "recovered from the bad infer");
        // The infer tool's failure was fed back as a tool result the model
        // saw on turn 2 (errors-as-values, not a turn-aborting error). Only
        // the two main-model calls are recorded; the dead-model sub-infer
        // errored before recording, so turn 2's prompt is index 1.
        let turn_two = provider.prompts.lock().unwrap()[1].clone();
        let tool_result = turn_two
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("the infer failure was fed back as a tool result");
        let content = tool_result.content.as_deref().unwrap();
        assert!(content.contains("\"ok\":false"), "{content}");
        assert!(content.contains("404"), "{content}");
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_remember_store_failure_becomes_a_recoverable_tool_result() -> Result<()>
    {
        // t-1222 (Store side): a rejected write — here a duplicate slug —
        // becomes a tool result the model can react to, not a fatal turn.
        let dir = memory_dir();
        tokio::fs::write(
            dir.join("taken.md"),
            "---\nname: taken\n---\n\nalready here",
        )
        .await
        .unwrap();
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "remember",
                    serde_json::json!({ "name": "taken", "content": "collides with an existing slug" }),
                )],
            ),
            response("acknowledged the collision", vec![]),
        ]));
        let config = memory_config(provider.clone(), dir);
        let machine = agent_loop_ir_with_options(
            Model("mock".into()),
            vec![ChatMessage::user("remember a colliding note, then answer")],
            6,
            true,
        );

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
        let response: Response = serde_json::from_value(value)?;
        assert_eq!(response.content, "acknowledged the collision");

        let turn_two = &provider.prompts()[1];
        let tool_result = turn_two
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("the store failure was fed back as a tool result");
        let content = tool_result.content.as_deref().unwrap();
        assert!(content.contains("\"ok\":false"), "{content}");
        assert!(content.contains("already exists"), "{content}");
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

    // ---- native tools (t-1308.7) ----

    fn native_tool_config(
        provider: Arc<dyn ChatProvider>,
        trace: TraceLogger,
        tool: crate::tool::NativeTool,
    ) -> SeqConfig {
        let mut config = config_with_trace(provider, trace);
        config.tools.register(tool).unwrap();
        config
    }

    fn weather_tool(calls: Arc<std::sync::atomic::AtomicUsize>) -> crate::tool::NativeTool {
        crate::tool::NativeTool::from_fn(
            "get_weather",
            "Current weather for a city.",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
            move |arguments| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let city = arguments["city"].as_str().unwrap_or("nowhere").to_owned();
                    Ok(Value::String(format!("sunny in {city}")))
                }
            },
        )
    }

    fn weather_turns() -> Vec<Response> {
        vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "get_weather",
                    serde_json::json!({ "city": "sf" }),
                )],
            ),
            response("done: sunny in sf", vec![]),
        ]
    }

    #[tokio::test]
    async fn agent_loop_ir_dispatches_native_tool_with_effect_identity() -> Result<()> {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let provider = Arc::new(MockProvider::new(weather_turns()));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = native_tool_config(provider.clone(), trace, weather_tool(calls.clone()));
        let machine = agent_loop_ir_with_tools(
            Model("mock".into()),
            vec![ChatMessage::user("what's the weather in sf?")],
            4,
            false,
            &["get_weather".into()],
        );
        validate_program(&machine.program)?;

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;

        assert_eq!(value["content"], Value::String("done: sunny in sf".into()));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        // The handler's string result feeds back verbatim (not JSON-quoted).
        let prompts = provider.prompts();
        let tool_message = prompts[1]
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("native tool result fed back");
        assert_eq!(tool_message.content.as_deref(), Some("sunny in sf"));

        // The dispatch rode a Tool effect with stable identity — and never
        // the shell: no Eval events at all.
        let events = TraceLogger::read_events(&trace_path).await?;
        let tool_call = events
            .iter()
            .find_map(|event| match event {
                Event::ToolCall {
                    name,
                    arguments,
                    effect,
                    ..
                } => Some((name.clone(), arguments.clone(), effect.clone())),
                _ => None,
            })
            .expect("ToolCall event recorded");
        assert_eq!(tool_call.0, "get_weather");
        assert_eq!(tool_call.1, serde_json::json!({ "city": "sf" }));
        let effect = tool_call.2.expect("Tool effect identity recorded");
        assert_eq!(effect.kind, crate::ir::EffectKind::Tool);
        assert!(effect.effect_id.0.starts_with("sha256:"));
        let tool_result = events
            .iter()
            .find_map(|event| match event {
                Event::ToolResult { name, result, .. } => Some((name.clone(), result.clone())),
                _ => None,
            })
            .expect("ToolResult event recorded");
        assert_eq!(tool_result.0, "get_weather");
        assert_eq!(tool_result.1, Value::String("sunny in sf".into()));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::EvalCall { .. })),
            "a native tool call must never become an Eval/shell execution"
        );
        let _ = std::fs::remove_file(&trace_path);
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_native_tool_replays_by_effect_id_without_invoking_handler() -> Result<()>
    {
        // Record a run with a live handler...
        let record_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let recorded_value = {
            let provider = Arc::new(MockProvider::new(weather_turns()));
            let config = native_tool_config(provider, trace, weather_tool(record_calls.clone()));
            let machine = agent_loop_ir_with_tools(
                Model("mock".into()),
                vec![ChatMessage::user("what's the weather in sf?")],
                4,
                false,
                &["get_weather".into()],
            );
            let (value, _machine) =
                crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
            value
        };
        assert_eq!(record_calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // ...then replay: same program, exhausted provider, a handler that
        // must not run. The recorded result comes back by effect id.
        let events = TraceLogger::read_events(&trace_path).await?;
        let replay = IrReplayTrace::from_events(&events)?;
        let replay_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let provider = Arc::new(MockProvider::new(vec![]));
        let config = native_tool_config(provider, test_trace(), weather_tool(replay_calls.clone()));
        let machine = agent_loop_ir_with_tools(
            Model("mock".into()),
            vec![ChatMessage::user("what's the weather in sf?")],
            4,
            false,
            &["get_weather".into()],
        );
        let mut store = crate::ir_interpreter::InMemoryStore::new();
        let (replayed, _machine) = crate::ir_interpreter::run_ir_sequential_with_store_and_replay(
            &config,
            machine,
            &mut store,
            Some(&replay),
        )
        .await?;

        assert_eq!(replayed, recorded_value);
        assert_eq!(
            replay_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "replay must not invoke the native tool handler"
        );
        let _ = std::fs::remove_file(&trace_path);
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_native_tool_failure_is_a_recoverable_tool_result() -> Result<()> {
        let tool = crate::tool::NativeTool::from_fn(
            "get_weather",
            "always fails",
            serde_json::json!({ "type": "object" }),
            |_| async { Err(anyhow!("weather service is down")) },
        );
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new(
                    "call-1",
                    "get_weather",
                    serde_json::json!({}),
                )],
            ),
            response("recovered from tool failure", vec![]),
        ]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = native_tool_config(provider.clone(), trace, tool);
        let machine = agent_loop_ir_with_tools(
            Model("mock".into()),
            vec![ChatMessage::user("try the tool")],
            4,
            false,
            &["get_weather".into()],
        );

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;

        assert_eq!(
            value["content"],
            Value::String("recovered from tool failure".into())
        );
        // Errors-as-values: the failure reached the model as a tool result...
        let prompts = provider.prompts();
        let tool_message = prompts[1]
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("failure tool result fed back");
        let content = tool_message.content.as_deref().unwrap();
        assert!(content.contains("\"ok\":false"), "{content}");
        assert!(content.contains("weather service is down"), "{content}");
        // ...and the trace closed the call with a ToolError event.
        let events = TraceLogger::read_events(&trace_path).await?;
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolError { name, error, .. }
                if name == "get_weather" && error.contains("weather service is down")
        )));
        let _ = std::fs::remove_file(&trace_path);
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_ir_unregistered_native_tool_binds_error_result() -> Result<()> {
        // The program advertises the arm but the registry has no handler
        // (a caller wiring bug): the model sees the error, the turn lives.
        let provider = Arc::new(MockProvider::new(vec![
            response(
                "",
                vec![ToolCall::new("call-1", "ghost", serde_json::json!({}))],
            ),
            response("noted the missing tool", vec![]),
        ]));
        let machine = agent_loop_ir_with_tools(
            Model("mock".into()),
            vec![ChatMessage::user("call the ghost tool")],
            4,
            false,
            &["ghost".into()],
        );

        let (value, _machine) =
            crate::ir_interpreter::run_ir_sequential(&config(provider.clone()), machine).await?;

        assert_eq!(
            value["content"],
            Value::String("noted the missing tool".into())
        );
        let prompts = provider.prompts();
        let tool_message = prompts[1]
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call-1"))
            .expect("missing-registration result fed back");
        let content = tool_message.content.as_deref().unwrap();
        assert!(content.contains("no native tool"), "{content}");
        Ok(())
    }

    #[test]
    fn native_tools_change_the_program_hash_and_unknown_message() {
        let plain = agent_loop_ir(Model("m".into()), vec![], 4);
        let with_native =
            agent_loop_ir_with_tools(Model("m".into()), vec![], 4, false, &["get_weather".into()]);
        assert_ne!(
            crate::ir::program_hash(&plain.program).unwrap(),
            crate::ir::program_hash(&with_native.program).unwrap(),
            "the tool set is part of the program identity"
        );
        validate_program(&with_native.program).expect("native-tool variant validates");
        // Native + memory arms coexist.
        let both = agent_loop_ir_with_tools(
            Model("m".into()),
            vec![],
            4,
            true,
            &["get_weather".into(), "lookup".into()],
        );
        validate_program(&both.program).expect("memory + native variant validates");
        let rendered = serde_json::to_string(&both.program).unwrap();
        assert!(
            rendered.contains(
                "unknown tool; available tools: shell, infer, remember, recall, get_weather, lookup"
            ),
            "unknown-tool message names the full tool set"
        );
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

    // ---- run_agent_loop / OutputContract (t-1308.4) ----

    fn answer_schema() -> Value {
        serde_json::json!({
            "type": "object",
            "required": ["answer"],
            "properties": { "answer": { "type": "integer" } }
        })
    }

    fn contract_with_repairs(schema: Value, max_repairs: usize) -> OutputContract {
        OutputContract {
            schema,
            max_repairs,
        }
    }

    /// Drive run_agent_loop with a scripted provider and return the loop
    /// value, the final machine, and the runtime trace events.
    async fn run_contract_loop(
        provider: Arc<MockProvider>,
        contract: Option<OutputContract>,
        max_turns: usize,
    ) -> Result<(Value, Machine, Vec<Event>)> {
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let config = config_with_trace(provider, trace);
        let mut store = crate::ir_interpreter::InMemoryStore::new();
        let mut gc_state = GcState::default();
        let options = AgentLoopOptions {
            memory_tools: false,
            tool_names: Vec::new(),
            output_contract: contract,
            shell_requires_approval: false,
        };
        let (value, machine) = run_agent_loop(
            &config,
            &mut store,
            None,
            &mut gc_state,
            Model("mock".into()),
            vec![ChatMessage::system("system"), ChatMessage::user("go")],
            max_turns,
            &options,
            BTreeMap::new(),
        )
        .await?;
        let events = TraceLogger::read_events(&trace_path).await?;
        let _ = std::fs::remove_file(&trace_path);
        Ok((value, machine, events))
    }

    fn validation_failures(events: &[Event]) -> Vec<&Value> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Custom { name, data, .. }
                    if name == crate::output_contract::OUTPUT_VALIDATION_FAILED_EVENT =>
                {
                    Some(data)
                }
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn output_contract_valid_response_passes_through() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response(
            r#"{"answer": 42}"#,
            vec![],
        )]));
        let contract = contract_with_repairs(answer_schema(), 2);
        let schema_hash = contract.schema_hash();
        let (value, _machine, events) = run_contract_loop(provider, Some(contract), 4).await?;

        assert_eq!(value["content"], Value::String(r#"{"answer": 42}"#.into()));
        assert!(validation_failures(&events).is_empty());
        // The schema hash rides in the trace as run-identity metadata.
        let recorded = events.iter().find_map(|event| match event {
            Event::Custom { name, data, .. }
                if name == crate::output_contract::OUTPUT_CONTRACT_EVENT =>
            {
                data.get("schema_hash")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }
            _ => None,
        });
        assert_eq!(recorded.as_deref(), Some(schema_hash.as_str()));
        Ok(())
    }

    #[tokio::test]
    async fn output_contract_invalid_then_repaired() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response("not json at all", vec![]),
            response(r#"{"answer": 7}"#, vec![]),
        ]));
        let (value, machine, events) = run_contract_loop(
            provider.clone(),
            Some(contract_with_repairs(answer_schema(), 2)),
            4,
        )
        .await?;

        assert_eq!(value["content"], Value::String(r#"{"answer": 7}"#.into()));
        // Exactly one failed attempt was traced, with the parse error.
        let failures = validation_failures(&events);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0]["attempt"], Value::from(1));
        assert!(
            failures[0]["errors"][0]
                .as_str()
                .unwrap()
                .contains("not valid JSON"),
            "{failures:?}"
        );
        assert_eq!(
            failures[0]["preview"],
            Value::String("not json at all".into())
        );
        // The repair turn carried the invalid answer plus a user message
        // quoting the errors and demanding JSON only.
        let prompts = provider.prompts();
        assert_eq!(prompts.len(), 2);
        let repair = &prompts[1];
        let assistant = repair
            .iter()
            .find(|message| message.role == "assistant")
            .expect("invalid assistant answer in repair prompt");
        assert_eq!(assistant.content.as_deref(), Some("not json at all"));
        let last = repair.last().unwrap();
        assert_eq!(last.role, "user");
        let repair_text = last.content.as_deref().unwrap();
        assert!(repair_text.contains("not valid JSON"), "{repair_text}");
        assert!(
            repair_text.contains("ONLY a single JSON value"),
            "{repair_text}"
        );
        // The final machine's history contains the repair exchange, so a
        // session persists it like any other turn.
        let history = machine.env.get(&Var("history".into())).unwrap();
        assert!(
            serde_json::to_string(history)?.contains("failed validation"),
            "repair exchange survives in history"
        );
        Ok(())
    }

    #[tokio::test]
    async fn output_contract_exhausted_repairs_return_typed_error_value() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response(r#"{"answer": "nope"}"#, vec![]),
            response(r#"{"wrong": true}"#, vec![]),
        ]));
        let (value, _machine, events) =
            run_contract_loop(provider, Some(contract_with_repairs(answer_schema(), 1)), 4).await?;

        // The loop returned (no abort) with the typed failure value.
        let failure = crate::output_contract::output_contract_failure(&value)
            .expect("exhausted repairs yield a contract failure value");
        assert_eq!(failure.attempts, 2);
        assert_eq!(failure.content, r#"{"wrong": true}"#);
        assert!(
            failure.errors.iter().any(|error| error.contains("answer")),
            "last validation errors preserved: {failure:?}"
        );
        let failures = validation_failures(&events);
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0]["attempt"], Value::from(1));
        assert_eq!(failures[1]["attempt"], Value::from(2));
        Ok(())
    }

    #[tokio::test]
    async fn output_contract_skips_budget_exhausted_turns() -> Result<()> {
        // A turn-budget stop never produced a final answer; it is returned
        // annotated, not validated (and not "repaired").
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
        let provider = Arc::new(MockProvider::new(vec![tool_turn(), tool_turn()]));
        let (value, _machine, events) =
            run_contract_loop(provider, Some(contract_with_repairs(answer_schema(), 2)), 1).await?;

        assert_eq!(
            value["metadata"]["stop_reason"],
            Value::String("turn_budget_exhausted".into())
        );
        assert!(validation_failures(&events).is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn output_contract_absent_leaves_loop_untouched() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("plain text", vec![])]));
        let (value, _machine, events) = run_contract_loop(provider, None, 4).await?;
        assert_eq!(value["content"], Value::String("plain text".into()));
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::Custom { .. })));
        Ok(())
    }

    #[tokio::test]
    async fn output_contract_schema_hash_gates_replay() -> Result<()> {
        // Record a run with contract A...
        let contract_a = contract_with_repairs(answer_schema(), 2);
        let trace = test_trace();
        let trace_path = trace.path().clone();
        {
            let provider = Arc::new(MockProvider::new(vec![response(
                r#"{"answer": 1}"#,
                vec![],
            )]));
            let config = config_with_trace(provider, trace);
            let mut store = crate::ir_interpreter::InMemoryStore::new();
            let mut gc_state = GcState::default();
            let options = AgentLoopOptions {
                memory_tools: false,
                tool_names: Vec::new(),
                output_contract: Some(contract_a.clone()),
                shell_requires_approval: false,
            };
            let (value, _machine) = run_agent_loop(
                &config,
                &mut store,
                None,
                &mut gc_state,
                Model("mock".into()),
                vec![ChatMessage::user("go")],
                4,
                &options,
                BTreeMap::new(),
            )
            .await?;
            assert_eq!(value["content"], Value::String(r#"{"answer": 1}"#.into()));
        }
        let events = TraceLogger::read_events(&trace_path).await?;
        let replay = IrReplayTrace::from_events(&events)?;
        assert_eq!(
            replay.output_schema_hash(),
            Some(contract_a.schema_hash().as_str())
        );

        let run_replay = |contract: Option<OutputContract>| {
            let events = events.clone();
            async move {
                let replay = IrReplayTrace::from_events(&events)?;
                let provider = Arc::new(MockProvider::new(vec![]));
                let config = config(provider);
                let mut store = crate::ir_interpreter::InMemoryStore::new();
                let mut gc_state = GcState::default();
                let options = AgentLoopOptions {
                    memory_tools: false,
                    tool_names: Vec::new(),
                    output_contract: contract,
                    shell_requires_approval: false,
                };
                run_agent_loop(
                    &config,
                    &mut store,
                    Some(&replay),
                    &mut gc_state,
                    Model("mock".into()),
                    vec![ChatMessage::user("go")],
                    4,
                    &options,
                    BTreeMap::new(),
                )
                .await
                .map(|(value, _machine)| value)
            }
        };

        // Same contract: replays cleanly from the recorded results.
        let value = run_replay(Some(contract_a.clone())).await?;
        assert_eq!(value["content"], Value::String(r#"{"answer": 1}"#.into()));

        // A changed schema diverges with a clear error before any effect.
        let contract_b = contract_with_repairs(serde_json::json!({ "type": "array" }), 2);
        let err = run_replay(Some(contract_b)).await.unwrap_err();
        assert!(err.to_string().contains("schema_hash"), "{err:#}");

        // Dropping the contract entirely also diverges.
        let err = run_replay(None).await.unwrap_err();
        assert!(err.to_string().contains("schema_hash"), "{err:#}");

        let _ = std::fs::remove_file(&trace_path);
        Ok(())
    }
}
