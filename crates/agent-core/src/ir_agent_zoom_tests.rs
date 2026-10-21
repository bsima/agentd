//! The agent loop's GC zoom arm (t-1609): program shape, dispatch,
//! addressing through the real loop, child isolation, and replay.

use super::*;
use crate::gc::{GcMode, GcTiming, StackFrameGc};
use crate::hydration::{PassiveHydrationConfig, SourceRegistry};
use crate::interpreter::EvalConfig;
use crate::ir::validate_program;
use crate::op::{ChatMessage, FinishReason, Response, ToolCall};
use crate::provider::{ChatProvider, ToolSpec};
use crate::trace::TraceLogger;
use crate::zoom::{ZoomOptions, ZOOM_TOOL};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Scripted provider that records each call's prompt and offered tools.
struct Recorder {
    responses: Mutex<Vec<Response>>,
    calls: Mutex<Vec<(Vec<ChatMessage>, Vec<String>)>>,
}

impl Recorder {
    fn new(mut responses: Vec<Response>) -> Arc<Self> {
        responses.reverse();
        Arc::new(Self {
            responses: Mutex::new(responses),
            calls: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> Vec<(Vec<ChatMessage>, Vec<String>)> {
        self.calls.lock().unwrap().clone()
    }

    /// The tool result for `call_id` in the last prompt.
    fn result(&self, call_id: &str) -> String {
        let calls = self.calls();
        let (prompt, _) = calls.last().expect("a provider call");
        prompt
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some(call_id))
            .and_then(|message| message.content.clone())
            .unwrap_or_else(|| panic!("no result for {call_id} in {prompt:#?}"))
    }
}

#[async_trait]
impl ChatProvider for Recorder {
    async fn chat(
        &self,
        _model: &Model,
        tools: &[ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        self.calls.lock().unwrap().push((
            messages.to_vec(),
            tools
                .iter()
                .map(|tool| tool.function.name.clone())
                .collect(),
        ));
        self.responses
            .lock()
            .unwrap()
            .pop()
            .ok_or_else(|| anyhow!("recorder exhausted"))
    }
}

fn response(content: &str, tool_calls: Vec<ToolCall>) -> Response {
    Response {
        content: content.into(),
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

fn trace() -> TraceLogger {
    TraceLogger::new(
        Uuid::new_v4().to_string(),
        std::env::temp_dir().join(format!("agent-ir-zoom-{}.jsonl", Uuid::new_v4())),
    )
}

fn config(provider: Arc<dyn ChatProvider>, trace: TraceLogger, gc: GcMode) -> SeqConfig {
    SeqConfig {
        on_infer_delta: None,
        on_reasoning_delta: None,
        inference_settings: Default::default(),
        approvals: Default::default(),
        guidance: Default::default(),
        tools: Default::default(),
        provider,
        hydration: SourceRegistry::new(),
        passive_hydration: PassiveHydrationConfig::default(),
        trace,
        eval: EvalConfig::default(),
        replay: None,
        trace_full_prompt_ir: false,
        trace_full_payloads: false,
        gc,
        gc_threshold: 0.85,
        gc_log: false,
        gc_timing: GcTiming::Threshold,
        context_budget: 200_000,
        pricing: Default::default(),
    }
}

fn zoom_options() -> AgentLoopOptions {
    AgentLoopOptions {
        zoom: Some(ZoomOptions::for_context_budget(200_000)),
        ..Default::default()
    }
}

fn program(options: &AgentLoopOptions) -> crate::ir::Program {
    agent_loop_ir_for_options(
        Model("mock".into()),
        vec![ChatMessage::user("hello")],
        4,
        options,
    )
    .program
}

fn has_zoom_read(program: &crate::ir::Program) -> bool {
    program.blocks.values().any(|block| {
        block.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::Let {
                    expr: Expr::Zoom { .. },
                    ..
                }
            )
        })
    })
}

/// Off is the default program byte-for-byte (so every recorded fixture's
/// effect ids keep their program hash); on adds a validated arm.
#[test]
fn zoom_option_shapes_the_program_only_when_set() {
    let default_program = program(&AgentLoopOptions::default());
    let legacy = agent_loop_ir(Model("mock".into()), vec![ChatMessage::user("hello")], 4).program;
    assert_eq!(default_program, legacy);
    assert_eq!(
        crate::ir::program_hash(&default_program).unwrap(),
        crate::ir::program_hash(&legacy).unwrap()
    );
    assert!(!has_zoom_read(&default_program));

    let with_zoom = program(&zoom_options());
    validate_program(&with_zoom).unwrap();
    crate::ir_normalize::normalize_program(&with_zoom).unwrap();
    assert!(has_zoom_read(&with_zoom));
    assert_ne!(
        crate::ir::program_hash(&default_program).unwrap(),
        crate::ir::program_hash(&with_zoom).unwrap()
    );
    // The page size is program: a different budget is a different program.
    let small = program(&AgentLoopOptions {
        zoom: Some(ZoomOptions::for_context_budget(8_000)),
        ..Default::default()
    });
    assert_ne!(
        crate::ir::program_hash(&small).unwrap(),
        crate::ir::program_hash(&with_zoom).unwrap()
    );
}

/// Appended after every existing arm: with native tools, memory tools, and
/// ask_user all on, adding zoom keeps every existing block and only
/// re-points the last fallthrough (ask_user's dispatch) at the zoom arm
/// and names zoom in the unknown-tool message.
#[test]
fn zoom_arm_appends_after_every_existing_block() {
    let base = AgentLoopOptions {
        memory_tools: true,
        tool_names: vec!["weather".into()],
        ask_user: true,
        ..Default::default()
    };
    let without = program(&base);
    let with = program(&AgentLoopOptions {
        zoom: Some(ZoomOptions::for_context_budget(200_000)),
        ..base
    });
    // Changed on purpose: the last native dispatch now falls through to
    // zoom, and the unknown-tool message lists zoom among the tools.
    let last_native_dispatch = BlockId(26 + 2);
    let invalid_tool = BlockId(10);
    for (id, block) in &without.blocks {
        if *id == last_native_dispatch || *id == invalid_tool {
            continue;
        }
        assert_eq!(with.blocks.get(id), Some(block), "block {id:?} changed");
    }
    assert_eq!(with.blocks.len(), without.blocks.len() + 2);
    assert!(with.blocks.contains_key(&BlockId(26 + 4)));
    assert!(with.blocks.contains_key(&BlockId(26 + 5)));
}

#[tokio::test]
async fn zoom_reads_history_exactly_by_position_and_alias() -> Result<()> {
    let provider = Recorder::new(vec![
        response(
            "",
            vec![ToolCall::new(
                "call-1",
                "shell",
                serde_json::json!({ "command": "printf 'counter=7\\n'" }),
            )],
        ),
        response(
            "",
            vec![
                // History: #0 user, #1 call-1, #2 its result.
                ToolCall::new("call-2", ZOOM_TOOL, serde_json::json!({ "address": "#2" })),
                ToolCall::new(
                    "call-3",
                    ZOOM_TOOL,
                    serde_json::json!({ "address": "call-1" }),
                ),
                ToolCall::new("call-4", ZOOM_TOOL, serde_json::json!({ "address": "#0" })),
                ToolCall::new("call-5", ZOOM_TOOL, serde_json::json!({ "address": "#99" })),
            ],
        ),
        response("done", vec![]),
    ]);
    let config = config(provider.clone(), trace(), GcMode::None);
    let machine = agent_loop_ir_for_options(
        Model("mock".into()),
        vec![ChatMessage::user("read the counter")],
        6,
        &zoom_options(),
    );
    let (value, machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
    assert_eq!(value["content"], "done");

    let history = machine.env[&Var("history".into())]
        .as_array()
        .unwrap()
        .clone();
    let original = history[2]["content"].as_str().unwrap().to_string();
    assert!(original.contains("counter=7"), "{original}");
    assert_eq!(
        provider.result("call-2"),
        format!("#2 tool result of shell call-1:\n{original}")
    );
    assert_eq!(provider.result("call-3"), provider.result("call-2"));
    assert_eq!(
        provider.result("call-4"),
        "#0 user message:\nread the counter"
    );
    assert!(provider
        .result("call-5")
        .contains("\"error\":\"address_out_of_range\""));
    // The tool is offered to the loop's turn Infers.
    assert!(provider
        .calls()
        .iter()
        .all(|(_, tools)| tools.iter().any(|tool| tool == ZOOM_TOOL)));
    Ok(())
}

#[tokio::test]
async fn without_the_option_zoom_is_an_unknown_tool() -> Result<()> {
    let provider = Recorder::new(vec![
        response(
            "",
            vec![ToolCall::new(
                "call-1",
                ZOOM_TOOL,
                serde_json::json!({ "address": "#0" }),
            )],
        ),
        response("done", vec![]),
    ]);
    let config = config(provider.clone(), trace(), GcMode::None);
    let machine = agent_loop_ir(Model("mock".into()), vec![ChatMessage::user("hi")], 4);
    crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
    assert!(provider
        .result("call-1")
        .contains("\"error\":\"unknown_tool\""));
    assert!(provider
        .calls()
        .iter()
        .all(|(_, tools)| !tools.iter().any(|tool| tool == ZOOM_TOOL)));
    Ok(())
}

/// Delegated children are bare completions: no tools, so no zoom — and
/// their prompts are not history, so no address book (markers in a child
/// collection keep the zoom-free text).
#[tokio::test]
async fn sub_infer_children_get_no_zoom() -> Result<()> {
    let provider = Recorder::new(vec![
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
    ]);
    let config = config(
        provider.clone(),
        trace(),
        GcMode::Stack(StackFrameGc::default()),
    );
    let machine = agent_loop_ir_for_options(
        Model("mock".into()),
        vec![ChatMessage::user("delegate")],
        4,
        &zoom_options(),
    );
    let mut gc_state = GcState::default();
    let mut store = crate::ir_interpreter::InMemoryStore::new();
    crate::ir_interpreter::run_ir_sequential_with_gc(
        &config,
        machine,
        &mut store,
        None,
        &mut gc_state,
    )
    .await?;
    let calls = provider.calls();
    assert_eq!(calls.len(), 3);
    assert!(calls[0].1.iter().any(|tool| tool == ZOOM_TOOL));
    assert!(calls[1].1.is_empty(), "the child is offered no tools");
    assert!(calls[2].1.iter().any(|tool| tool == ZOOM_TOOL));
    assert!(gc_state.zoom.is_some(), "the last turn Infer offered zoom");
    Ok(())
}

/// A zoom is a pure read: replay reproduces the run from the trace alone,
/// and the trace holds no zoom effect.
#[tokio::test]
async fn zoom_turns_replay_from_the_trace() -> Result<()> {
    let trace = trace();
    let trace_path = trace.path().clone();
    let provider = Recorder::new(vec![
        response(
            "",
            vec![ToolCall::new(
                "call-1",
                "shell",
                serde_json::json!({ "command": "printf 'ticket=4101\\n'" }),
            )],
        ),
        response(
            "",
            vec![ToolCall::new(
                "call-2",
                ZOOM_TOOL,
                serde_json::json!({ "address": "#2" }),
            )],
        ),
        response("TICKET 4101", vec![]),
    ]);
    let live = config(provider.clone(), trace, GcMode::None);
    let options = zoom_options();
    let prompt = vec![ChatMessage::user("issue a ticket")];
    let (recorded, recorded_machine) = run_agent_loop(
        &live,
        &mut crate::ir_interpreter::InMemoryStore::new(),
        None,
        &mut GcState::default(),
        Model("mock".into()),
        prompt.clone(),
        6,
        &options,
        BTreeMap::new(),
    )
    .await?;

    let events = TraceLogger::read_events(&trace_path).await?;
    for event in &events {
        let line = serde_json::to_string(event)?;
        if line.contains("\"effect\"") {
            assert!(
                !line.contains("\"zoom\""),
                "zoom recorded as an effect: {line}"
            );
        }
    }

    let replay = IrReplayTrace::load(&trace_path).await?;
    let replay_provider = Recorder::new(vec![]);
    let (replayed, replayed_machine) = run_agent_loop(
        &config(replay_provider.clone(), self::trace(), GcMode::None),
        &mut crate::ir_interpreter::InMemoryStore::new(),
        Some(&replay),
        &mut GcState::default(),
        Model("mock".into()),
        prompt.clone(),
        6,
        &options,
        BTreeMap::new(),
    )
    .await?;
    assert_eq!(replayed, recorded);
    assert_eq!(
        replayed_machine.env[&Var("history".into())],
        recorded_machine.env[&Var("history".into())],
        "the zoom result in history is byte-identical under replay"
    );
    assert!(replay_provider.calls().is_empty(), "replay never infers");
    let _ = std::fs::remove_file(&trace_path);
    Ok(())
}
