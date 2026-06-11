use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use uuid::Uuid;

pub type BoxFutureOp<S, A> = Pin<Box<dyn Future<Output = Op<S, A>> + Send>>;
pub type Prompt = Vec<ChatMessage>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Other(String),
}

impl FinishReason {
    pub fn from_provider(value: impl AsRef<str>) -> Self {
        match value.as_ref() {
            "stop" | "end_turn" => Self::Stop,
            "tool_calls" | "tool_use" => Self::ToolCalls,
            "length" | "max_tokens" => Self::Length,
            "content_filter" => Self::ContentFilter,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn is_stop(&self) -> bool {
        matches!(self, Self::Stop)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl PartialEq for ChatMessage {
    fn eq(&self, other: &Self) -> bool {
        self.role == other.role
            && self.content == other.content
            && self.tool_call_id == other.tool_call_id
            && self.tool_calls == other.tool_calls
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            role: "system".into(),
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            role: "user".into(),
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            id: Uuid::new_v4(),
            role: "assistant".into(),
            content,
            tool_call_id: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
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
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    /// Runtime annotations that travel with the response (e.g.
    /// stop_reason = "turn_budget_exhausted" when the agent loop returned
    /// because max_turns ran out rather than a natural stop — t-1133).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, Value>,
}

/// A provider-neutral tool invocation. `arguments` is the parsed JSON value,
/// not a provider-specific encoding. Providers adapt this to their own wire
/// shape at the serialization edge (see `provider`, `anthropic`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Checkpoints, traces, and eval fixtures written before the neutral
        // shape carry OpenAI-wire tool calls
        // ({"id", "type", "function": {"name", "arguments": "<json string>"}}).
        // Accept both so old persisted state keeps loading; serialization is
        // always the neutral shape.
        #[derive(Deserialize)]
        struct LegacyOpenAiFunction {
            name: String,
            arguments: String,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Compat {
            Neutral {
                id: String,
                name: String,
                arguments: Value,
            },
            LegacyOpenAi {
                id: String,
                function: LegacyOpenAiFunction,
            },
        }

        Ok(match Compat::deserialize(deserializer)? {
            Compat::Neutral {
                id,
                name,
                arguments,
            } => Self {
                id,
                name,
                arguments,
            },
            Compat::LegacyOpenAi { id, function } => Self {
                id,
                name: function.name,
                arguments: serde_json::from_str(&function.arguments)
                    .unwrap_or_else(|_| serde_json::json!({ "raw": function.arguments })),
            },
        })
    }
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
        let can_stop = response
            .finish_reason
            .as_ref()
            .is_some_and(FinishReason::is_stop)
            && response.tool_calls.is_empty()
            && !has_pending_tool_calls(&prompt);
        if can_stop || max_turns == 0 {
            Op::pure(response)
        } else if response.tool_calls.is_empty() {
            let mut history = prompt.clone();
            if !response.content.is_empty() {
                history.push(ChatMessage::assistant(
                    Some(response.content.clone()),
                    Vec::new(),
                ));
            }
            history.push(ChatMessage::user(CONTINUE_NUDGE));
            let value = serde_json::to_value(&history).unwrap_or(Value::Null);
            put("temporal:history", value)
                .and_then(move |_| agent_loop(model, history, max_turns - 1))
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

/// Synthetic user message appended when the model returns a non-stop turn with
/// no tool calls. Shared by the Op and IR agent loops so both runtimes recover
/// from stalled turns identically.
pub(crate) const CONTINUE_NUDGE: &str =
    "Your previous response did not finish the turn. Continue the task: if work remains, issue \
     the next tool call; if the task is complete, say so explicitly.";

pub fn has_pending_tool_calls(prompt: &[ChatMessage]) -> bool {
    let mut pending = std::collections::BTreeSet::new();
    for message in prompt {
        if let Some(tool_calls) = &message.tool_calls {
            pending.extend(tool_calls.iter().map(|call| call.id.clone()));
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            pending.remove(tool_call_id);
        }
    }
    !pending.is_empty()
}

pub fn repair_trailing_pending_tool_calls(prompt: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut pending = std::collections::BTreeSet::new();
    let mut latest_clean_len = 0;
    for (index, message) in prompt.iter().enumerate() {
        if let Some(tool_calls) = &message.tool_calls {
            pending.extend(tool_calls.iter().map(|call| call.id.clone()));
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            pending.remove(tool_call_id);
        }
        if pending.is_empty() {
            latest_clean_len = index + 1;
        }
    }
    prompt[..latest_clean_len].to_vec()
}

/// Append a synthetic error tool result for every assistant tool call that has
/// no matching result, so the transcript stays well-formed for providers.
///
/// The agent loop returns its final response with unexecuted tool calls when
/// the turn budget runs out. Once that assistant message is appended to the
/// live history, every later provider call would be rejected by the
/// pending-tool-call guard, wedging the session until an operator restarts it
/// from a repaired checkpoint. Closing the calls keeps the session usable and
/// tells the model why its calls never ran. Returns the closed call ids.
pub fn close_pending_tool_calls(history: &mut Vec<ChatMessage>) -> Vec<String> {
    let mut pending: Vec<String> = Vec::new();
    for message in history.iter() {
        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                if !pending.contains(&call.id) {
                    pending.push(call.id.clone());
                }
            }
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            pending.retain(|id| id != tool_call_id);
        }
    }
    for id in &pending {
        let content = serde_json::json!({
            "ok": false,
            "error": "turn_budget_exhausted",
            "message": "tool call was not executed: the turn budget ran out before dispatch",
        });
        history.push(ChatMessage::tool(id, content.to_string()));
    }
    pending
}

fn command_from_tool_call(call: &ToolCall) -> std::result::Result<String, Value> {
    if call.name != "shell" {
        return Err(serde_json::json!({
            "ok": false,
            "error": "unknown_tool",
            "tool": call.name,
            "message": "unknown tool; available tools: shell"
        }));
    }

    match call.arguments.get("command").and_then(Value::as_str) {
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
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    struct NoopProvider;

    struct QueueProvider {
        responses: Mutex<VecDeque<Response>>,
    }

    impl QueueProvider {
        fn new(responses: Vec<Response>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
            }
        }
    }

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

    #[async_trait]
    impl ChatProvider for QueueProvider {
        async fn chat(
            &self,
            _model: &Model,
            _tools: &[ToolSpec],
            _messages: &[ChatMessage],
        ) -> Result<Response> {
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| anyhow!("no queued response"))
        }
    }

    fn response_with_finish(content: &str, finish_reason: Option<FinishReason>) -> Response {
        Response {
            content: content.into(),
            tool_calls: Vec::new(),
            finish_reason,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            metadata: Default::default(),
        }
    }

    #[test]
    fn tool_call_serializes_neutral_and_deserializes_both_shapes() {
        let call = ToolCall::new("call-1", "shell", serde_json::json!({ "command": "pwd" }));
        let encoded = serde_json::to_value(&call).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({
                "id": "call-1",
                "name": "shell",
                "arguments": { "command": "pwd" }
            })
        );
        let decoded: ToolCall = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, call);

        // Legacy OpenAI wire shape from pre-neutral checkpoints/traces.
        let legacy = serde_json::json!({
            "id": "call-1",
            "type": "function",
            "function": { "name": "shell", "arguments": "{\"command\":\"pwd\"}" }
        });
        let decoded: ToolCall = serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded, call);
    }

    #[test]
    fn legacy_tool_call_with_unparseable_arguments_falls_back_to_raw() {
        let legacy = serde_json::json!({
            "id": "call-1",
            "type": "function",
            "function": { "name": "shell", "arguments": "not json" }
        });
        let decoded: ToolCall = serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.name, "shell");
        assert_eq!(decoded.arguments, serde_json::json!({ "raw": "not json" }));
    }

    #[test]
    fn chat_message_with_legacy_tool_calls_deserializes() {
        // A checkpoint written before the neutral ToolCall shape.
        let message = serde_json::json!({
            "id": "f3b9c2d8-1111-2222-3333-444455556666",
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": { "name": "shell", "arguments": "{\"command\":\"pwd\"}" }
            }]
        });
        let decoded: ChatMessage = serde_json::from_value(message).unwrap();
        let calls = decoded.tool_calls.unwrap();
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments, serde_json::json!({ "command": "pwd" }));
    }

    #[test]
    fn chat_message_equality_ignores_stable_id() {
        let a = ChatMessage::user("same content");
        let b = ChatMessage::user("same content");

        assert_ne!(a.id, b.id);
        assert_eq!(a, b);
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
            trace_full_payloads: false,
            gc: GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            context_budget: 200_000,
        }
    }

    fn seq_config_with_provider(provider: Arc<dyn ChatProvider>) -> SeqConfig {
        SeqConfig {
            provider,
            ..seq_config()
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
        let valid = ToolCall::new(
            "call-1",
            "shell",
            serde_json::json!({ "command": "printf ok" }),
        );
        assert_eq!(command_from_tool_call(&valid).unwrap(), "printf ok");

        let unknown = ToolCall::new(
            "call-2",
            "echo",
            serde_json::json!({ "command": "printf bad" }),
        );
        assert_eq!(
            command_from_tool_call(&unknown).unwrap_err()["error"],
            serde_json::json!("unknown_tool")
        );

        let missing = ToolCall::new("call-3", "shell", serde_json::json!({}));
        assert_eq!(
            command_from_tool_call(&missing).unwrap_err()["error"],
            serde_json::json!("invalid_arguments")
        );

        let empty = ToolCall::new("call-4", "shell", serde_json::json!({ "command": "   " }));
        assert_eq!(
            command_from_tool_call(&empty).unwrap_err()["error"],
            serde_json::json!("invalid_arguments")
        );
    }

    #[test]
    fn close_pending_tool_calls_appends_error_results_for_dangling_calls() {
        let call_1 = ToolCall::new("call-1", "shell", serde_json::json!({ "command": "pwd" }));
        let call_2 = ToolCall::new("call-2", "shell", serde_json::json!({ "command": "ls" }));
        let mut history = vec![
            ChatMessage::user("work"),
            ChatMessage::assistant(None, vec![call_1]),
            ChatMessage::tool("call-1", "ok"),
            ChatMessage::assistant(Some("one more".into()), vec![call_2]),
        ];

        let closed = close_pending_tool_calls(&mut history);

        assert_eq!(closed, vec!["call-2".to_string()]);
        assert!(!has_pending_tool_calls(&history));
        let result = history.last().unwrap();
        assert_eq!(result.role, "tool");
        assert_eq!(result.tool_call_id.as_deref(), Some("call-2"));
        assert!(result
            .content
            .as_deref()
            .unwrap()
            .contains("turn_budget_exhausted"));
    }

    #[test]
    fn close_pending_tool_calls_is_a_noop_on_clean_history() {
        let mut history = vec![
            ChatMessage::user("work"),
            ChatMessage::assistant(Some("done".into()), Vec::new()),
        ];
        let before = history.clone();

        assert!(close_pending_tool_calls(&mut history).is_empty());
        assert_eq!(history, before);
    }

    #[test]
    fn repair_trailing_pending_tool_calls_drops_to_clean_prefix() {
        let call_1 = ToolCall::new(
            "call-1",
            "shell",
            serde_json::json!({ "command": "printf ok" }),
        );
        let call_2 = ToolCall::new("call-2", "shell", serde_json::json!({ "command": "pwd" }));
        let prompt = vec![
            ChatMessage::system("system"),
            ChatMessage::user("first"),
            ChatMessage::assistant(None, vec![call_1]),
            ChatMessage::tool("call-1", "ok"),
            ChatMessage::user("second"),
            ChatMessage::assistant(None, vec![call_2]),
        ];

        let repaired = repair_trailing_pending_tool_calls(&prompt);

        assert!(!has_pending_tool_calls(&repaired));
        assert_eq!(repaired.len(), 5);
        assert_eq!(repaired.last().unwrap().role, "user");
    }

    #[tokio::test]
    async fn agent_loop_nudges_empty_non_stop_turn_instead_of_completing() -> Result<()> {
        let provider = Arc::new(QueueProvider::new(vec![
            response_with_finish("", Some(FinishReason::Other("length".into()))),
            response_with_finish("done", Some(FinishReason::Stop)),
        ]));
        let op = agent_loop(Model("model".into()), vec![ChatMessage::user("work")], 3);

        let (response, _state) = run_sequential(
            &seq_config_with_provider(provider),
            vec![ChatMessage::user("work")],
            op,
        )
        .await?;

        assert_eq!(response.content, "done");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        Ok(())
    }

    #[tokio::test]
    async fn agent_loop_allows_clean_empty_stop_to_complete() -> Result<()> {
        let provider = Arc::new(QueueProvider::new(vec![response_with_finish(
            "",
            Some(FinishReason::Stop),
        )]));
        let op = agent_loop(Model("model".into()), vec![ChatMessage::user("work")], 3);

        let (response, _state) = run_sequential(
            &seq_config_with_provider(provider),
            vec![ChatMessage::user("work")],
            op,
        )
        .await?;

        assert_eq!(response.content, "");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        Ok(())
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
