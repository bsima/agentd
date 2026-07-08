//! t-1359 step 3: user system prompts compose with — never replace — the
//! runtime-guidance fragment (docs/GUIDANCE.md §4 "Composability
//! contract").
//!
//! The CLI's `--system-prompt` (and a prompt file's `system_prompt`
//! frontmatter, and the agentd supervisor's agent.md instructions, and the
//! SDK's `instructions`) all produce the same runtime shape: a history
//! whose first message is a system message carrying exactly the user's
//! text. These tests drive the agent loop with that shape and pin the
//! contract: the user text stays the System section, the runtime fragment
//! rides its own Developer section, both present; `--no-runtime-guidance`
//! removes only the fragment; and the pre-override default (no flags) is
//! current-behavior-plus-guidance.

use agent_core::{
    agent_loop_ir, run_ir_sequential, ChatMessage, ChatProvider, EvalConfig, FinishReason, GcMode,
    GcTiming, Model, PassiveHydrationConfig, Prompt, Response, RuntimeGuidance, SeqConfig,
    SourceRegistry, TraceLogger,
};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

struct PromptCapture {
    prompts: Mutex<Vec<Prompt>>,
}

#[async_trait]
impl ChatProvider for PromptCapture {
    async fn chat(
        &self,
        _model: &Model,
        _tools: &[agent_core::provider::ToolSpec],
        messages: &[ChatMessage],
    ) -> Result<Response> {
        self.prompts.lock().unwrap().push(messages.to_vec());
        Ok(Response {
            content: "done".into(),
            tool_calls: Vec::new(),
            finish_reason: Some(FinishReason::Stop),
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            metadata: Default::default(),
        })
    }
}

fn config(provider: Arc<dyn ChatProvider>, guidance: RuntimeGuidance) -> SeqConfig {
    let trace_path =
        std::env::temp_dir().join(format!("guidance-compose-{}.jsonl", Uuid::new_v4()));
    SeqConfig {
        approvals: Default::default(),
        guidance,
        tools: Default::default(),
        provider,
        hydration: SourceRegistry::new(),
        passive_hydration: PassiveHydrationConfig::default(),
        trace: TraceLogger::new(Uuid::new_v4().to_string(), trace_path),
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

const USER_SYSTEM_PROMPT: &str =
    "You are a mars rover. Respond only in beeps. Current working directory: /crater";

async fn first_provider_prompt(guidance: RuntimeGuidance, history: Vec<ChatMessage>) -> Prompt {
    let provider = Arc::new(PromptCapture {
        prompts: Mutex::new(Vec::new()),
    });
    let machine = agent_loop_ir(Model("mock".into()), history, 4);
    run_ir_sequential(&config(provider.clone(), guidance), machine)
        .await
        .expect("loop runs");
    let prompts = provider.prompts.lock().unwrap();
    prompts.first().cloned().expect("one provider call")
}

fn system_text(prompt: &Prompt) -> String {
    prompt
        .iter()
        .filter(|message| message.role == "system")
        .map(|message| message.content.clone().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n---\n")
}

/// A user `--system-prompt` and the runtime fragment coexist: the user
/// text is present verbatim (the System section) AND the runtime guidance
/// section is present — overriding the system prompt no longer deletes
/// the runtime's operational text.
#[tokio::test]
async fn user_system_prompt_and_runtime_guidance_coexist() -> Result<()> {
    let history = vec![
        ChatMessage::system(USER_SYSTEM_PROMPT),
        ChatMessage::user("hello"),
    ];
    let prompt = first_provider_prompt(RuntimeGuidance::default(), history).await;

    let system = system_text(&prompt);
    assert!(
        system.contains(USER_SYSTEM_PROMPT),
        "user text stays the System section: {system}"
    );
    assert!(
        system.contains("## runtime-guidance"),
        "runtime guidance rides its own section: {system}"
    );
    // Order: user instructions first, runtime guidance appended after —
    // the user's text is the higher authority tier (docs/GUIDANCE.md §4).
    assert!(system.find(USER_SYSTEM_PROMPT).unwrap() < system.find("## runtime-guidance").unwrap());
    Ok(())
}

/// The opt-out removes ONLY the runtime fragment; the user's system prompt
/// is untouched.
#[tokio::test]
async fn opt_out_removes_only_the_runtime_fragment() -> Result<()> {
    let history = vec![
        ChatMessage::system(USER_SYSTEM_PROMPT),
        ChatMessage::user("hello"),
    ];
    let prompt = first_provider_prompt(RuntimeGuidance::disabled(), history).await;

    let system = system_text(&prompt);
    assert!(system.contains(USER_SYSTEM_PROMPT));
    assert!(!system.contains("runtime-guidance"));
    Ok(())
}

/// No flags = current behavior plus guidance: a default-shaped history
/// (the CLI's base system prompt) still gets the fragment, and a history
/// with no system message at all gets one synthesized for the fragment
/// (the SDK-without-instructions shape).
#[tokio::test]
async fn default_history_gets_guidance_added() -> Result<()> {
    let base_shaped = vec![
        ChatMessage::system("You are a standalone agent runner. When finished, answer concisely."),
        ChatMessage::user("hello"),
    ];
    let prompt = first_provider_prompt(RuntimeGuidance::default(), base_shaped).await;
    let system = system_text(&prompt);
    assert!(system.contains("standalone agent runner"));
    assert!(system.contains("## runtime-guidance"));

    let no_system = vec![ChatMessage::user("hello")];
    let prompt = first_provider_prompt(RuntimeGuidance::default(), no_system).await;
    assert!(system_text(&prompt).contains("## runtime-guidance"));
    Ok(())
}
