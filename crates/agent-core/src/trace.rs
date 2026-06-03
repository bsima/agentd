use crate::op::{Prompt, Response};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::{Span, SpanKind, Status, Tracer};
use opentelemetry::{global, KeyValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "PascalCase")]
pub enum Event {
    InferCall {
        run_id: String,
        op_id: u64,
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt: Option<Prompt>,
        prompt_preview: String,
        timestamp: DateTime<Utc>,
    },
    InferResult {
        run_id: String,
        op_id: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<Response>,
        response_preview: String,
        tokens: u32,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    EvalCall {
        run_id: String,
        op_id: u64,
        command: String,
        cwd: Option<String>,
        env_policy: String,
        timeout_ms: u64,
        timestamp: DateTime<Utc>,
    },
    EvalResult {
        run_id: String,
        op_id: u64,
        command: String,
        result: Value,
        duration_ms: u64,
        truncated_stdout: bool,
        truncated_stderr: bool,
        timestamp: DateTime<Utc>,
    },
    GetCall {
        run_id: String,
        op_id: u64,
        key: String,
        timestamp: DateTime<Utc>,
    },
    GetResult {
        run_id: String,
        op_id: u64,
        key: String,
        value: Value,
        value_preview: String,
        source_count: usize,
        timestamp: DateTime<Utc>,
    },
    PutCall {
        run_id: String,
        op_id: u64,
        key: String,
        value_preview: String,
        timestamp: DateTime<Utc>,
    },
    PutResult {
        run_id: String,
        op_id: u64,
        key: String,
        timestamp: DateTime<Utc>,
    },
    HydrationStart {
        run_id: String,
        op_id: u64,
        sources: Vec<String>,
        max_bytes: Option<usize>,
        timestamp: DateTime<Utc>,
    },
    HydrationSection {
        run_id: String,
        op_id: u64,
        source: String,
        kind: String,
        bytes: usize,
        content_preview: String,
        metadata: Value,
        timestamp: DateTime<Utc>,
    },
    HydrationEnd {
        run_id: String,
        op_id: u64,
        section_count: usize,
        total_bytes: usize,
        timestamp: DateTime<Utc>,
    },
    ParStart {
        run_id: String,
        op_id: u64,
        branch_count: usize,
        timestamp: DateTime<Utc>,
    },
    ParEnd {
        run_id: String,
        op_id: u64,
        branch_count: usize,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    Checkpoint {
        run_id: String,
        name: String,
        path: Option<String>,
        timestamp: DateTime<Utc>,
    },
    AgentDone {
        run_id: String,
        timestamp: DateTime<Utc>,
    },
    Custom {
        run_id: String,
        name: String,
        data: Value,
        timestamp: DateTime<Utc>,
    },
}

impl Event {
    pub fn run_id(&self) -> &str {
        match self {
            Self::InferCall { run_id, .. }
            | Self::InferResult { run_id, .. }
            | Self::EvalCall { run_id, .. }
            | Self::EvalResult { run_id, .. }
            | Self::GetCall { run_id, .. }
            | Self::GetResult { run_id, .. }
            | Self::PutCall { run_id, .. }
            | Self::PutResult { run_id, .. }
            | Self::HydrationStart { run_id, .. }
            | Self::HydrationSection { run_id, .. }
            | Self::HydrationEnd { run_id, .. }
            | Self::ParStart { run_id, .. }
            | Self::ParEnd { run_id, .. }
            | Self::Checkpoint { run_id, .. }
            | Self::AgentDone { run_id, .. }
            | Self::Custom { run_id, .. } => run_id,
        }
    }

    pub fn op_id(&self) -> Option<u64> {
        match self {
            Self::InferCall { op_id, .. }
            | Self::InferResult { op_id, .. }
            | Self::EvalCall { op_id, .. }
            | Self::EvalResult { op_id, .. }
            | Self::GetCall { op_id, .. }
            | Self::GetResult { op_id, .. }
            | Self::PutCall { op_id, .. }
            | Self::PutResult { op_id, .. }
            | Self::HydrationStart { op_id, .. }
            | Self::HydrationSection { op_id, .. }
            | Self::HydrationEnd { op_id, .. }
            | Self::ParStart { op_id, .. }
            | Self::ParEnd { op_id, .. } => Some(*op_id),
            Self::Checkpoint { .. } | Self::AgentDone { .. } | Self::Custom { .. } => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::InferCall { .. } | Self::InferResult { .. } => "Infer",
            Self::EvalCall { .. } | Self::EvalResult { .. } => "Eval",
            Self::GetCall { .. } | Self::GetResult { .. } => "Get",
            Self::PutCall { .. } | Self::PutResult { .. } => "Put",
            Self::HydrationStart { .. } | Self::HydrationEnd { .. } => "Hydration",
            Self::HydrationSection { .. } => "HydrationSection",
            Self::ParStart { .. } | Self::ParEnd { .. } => "Par",
            Self::Checkpoint { .. } => "Checkpoint",
            Self::AgentDone { .. } => "AgentDone",
            Self::Custom { name, .. } => match name.as_str() {
                "agent_error" => "agent_error",
                "agent_response" => "agent_response",
                "gc_collect" => "gc_collect",
                "context_overflow" => "context_overflow",
                _ => "Custom",
            },
        }
    }

    fn is_start(&self) -> bool {
        matches!(
            self,
            Self::InferCall { .. }
                | Self::EvalCall { .. }
                | Self::GetCall { .. }
                | Self::PutCall { .. }
                | Self::HydrationStart { .. }
                | Self::ParStart { .. }
        )
    }

    fn is_end(&self) -> bool {
        matches!(
            self,
            Self::InferResult { .. }
                | Self::EvalResult { .. }
                | Self::GetResult { .. }
                | Self::PutResult { .. }
                | Self::HydrationEnd { .. }
                | Self::ParEnd { .. }
        )
    }

    fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::InferCall { timestamp, .. }
            | Self::InferResult { timestamp, .. }
            | Self::EvalCall { timestamp, .. }
            | Self::EvalResult { timestamp, .. }
            | Self::GetCall { timestamp, .. }
            | Self::GetResult { timestamp, .. }
            | Self::PutCall { timestamp, .. }
            | Self::PutResult { timestamp, .. }
            | Self::HydrationStart { timestamp, .. }
            | Self::HydrationSection { timestamp, .. }
            | Self::HydrationEnd { timestamp, .. }
            | Self::ParStart { timestamp, .. }
            | Self::ParEnd { timestamp, .. }
            | Self::Checkpoint { timestamp, .. }
            | Self::AgentDone { timestamp, .. }
            | Self::Custom { timestamp, .. } => *timestamp,
        }
    }

    fn otel_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("agent.run_id", self.run_id().to_string()),
            KeyValue::new("agent.event", self.name()),
        ];
        if let Some(op_id) = self.op_id() {
            attrs.push(KeyValue::new("agent.op_id", op_id as i64));
        }
        match self {
            Self::InferCall {
                model,
                prompt_preview,
                ..
            } => {
                attrs.push(KeyValue::new("agent.model", model.clone()));
                attrs.push(KeyValue::new(
                    "agent.prompt_preview",
                    prompt_preview.clone(),
                ));
            }
            Self::InferResult {
                tokens,
                duration_ms,
                response_preview,
                ..
            } => {
                attrs.push(KeyValue::new("agent.tokens", i64::from(*tokens)));
                attrs.push(KeyValue::new("agent.duration_ms", *duration_ms as i64));
                attrs.push(KeyValue::new(
                    "agent.response_preview",
                    response_preview.clone(),
                ));
            }
            Self::EvalCall {
                command,
                cwd,
                env_policy,
                timeout_ms,
                ..
            } => {
                attrs.push(KeyValue::new("agent.command", command.clone()));
                if let Some(cwd) = cwd {
                    attrs.push(KeyValue::new("agent.cwd", cwd.clone()));
                }
                attrs.push(KeyValue::new("agent.env_policy", env_policy.clone()));
                attrs.push(KeyValue::new("agent.timeout_ms", *timeout_ms as i64));
            }
            Self::EvalResult {
                duration_ms,
                truncated_stdout,
                truncated_stderr,
                ..
            } => {
                attrs.push(KeyValue::new("agent.duration_ms", *duration_ms as i64));
                attrs.push(KeyValue::new("agent.stdout_truncated", *truncated_stdout));
                attrs.push(KeyValue::new("agent.stderr_truncated", *truncated_stderr));
            }
            Self::GetCall { key, .. }
            | Self::GetResult { key, .. }
            | Self::PutCall { key, .. }
            | Self::PutResult { key, .. } => attrs.push(KeyValue::new("agent.key", key.clone())),
            Self::HydrationStart {
                sources, max_bytes, ..
            } => {
                attrs.push(KeyValue::new("agent.sources", sources.join(",")));
                if let Some(max_bytes) = max_bytes {
                    attrs.push(KeyValue::new("agent.max_bytes", *max_bytes as i64));
                }
            }
            Self::HydrationSection {
                source,
                kind,
                bytes,
                content_preview,
                ..
            } => {
                attrs.push(KeyValue::new("agent.source", source.clone()));
                attrs.push(KeyValue::new("agent.kind", kind.clone()));
                attrs.push(KeyValue::new("agent.bytes", *bytes as i64));
                attrs.push(KeyValue::new(
                    "agent.content_preview",
                    content_preview.clone(),
                ));
            }
            Self::HydrationEnd {
                section_count,
                total_bytes,
                ..
            } => {
                attrs.push(KeyValue::new("agent.section_count", *section_count as i64));
                attrs.push(KeyValue::new("agent.total_bytes", *total_bytes as i64));
            }
            Self::ParStart { branch_count, .. } | Self::ParEnd { branch_count, .. } => {
                attrs.push(KeyValue::new("agent.branch_count", *branch_count as i64));
            }
            Self::Checkpoint { name, path, .. } => {
                attrs.push(KeyValue::new("agent.checkpoint", name.clone()));
                if let Some(path) = path {
                    attrs.push(KeyValue::new("agent.path", path.clone()));
                }
            }
            Self::Custom { name, data, .. } => {
                attrs.push(KeyValue::new("agent.custom_name", name.clone()));
                attrs.push(KeyValue::new("agent.data", data.to_string()));
            }
            Self::AgentDone { .. } => {}
        }
        attrs
    }
}

#[async_trait]
pub trait TraceSink: Send + Sync {
    async fn emit(&self, event: &Event) -> Result<()>;
}

#[derive(Clone)]
pub struct JsonlTraceSink {
    path: PathBuf,
    mirror_stdout: bool,
}

impl JsonlTraceSink {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            mirror_stdout: false,
        }
    }

    pub fn mirror_stdout(mut self, mirror_stdout: bool) -> Self {
        self.mirror_stdout = mirror_stdout;
        self
    }
}

#[async_trait]
impl TraceSink for JsonlTraceSink {
    async fn emit(&self, event: &Event) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;
        let line = serde_json::to_string(event)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        if self.mirror_stdout {
            let mut stdout = tokio::io::stdout();
            stdout.write_all(line.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct OtelTraceSink {
    spans: Mutex<HashMap<u64, BoxedSpan>>,
}

impl OtelTraceSink {
    pub fn new() -> Self {
        Self::default()
    }

    fn start_span(&self, event: &Event) {
        let Some(op_id) = event.op_id() else {
            return;
        };
        let tracer = global::tracer("agentd");
        let builder = tracer
            .span_builder(event.name())
            .with_kind(SpanKind::Internal)
            .with_start_time(event.timestamp())
            .with_attributes(event.otel_attributes());
        let span = tracer.build(builder);
        self.spans.lock().unwrap().insert(op_id, span);
    }

    fn finish_span(&self, event: &Event) {
        let Some(op_id) = event.op_id() else {
            self.emit_instant_event(event);
            return;
        };
        let mut spans = self.spans.lock().unwrap();
        if let Some(mut span) = spans.remove(&op_id) {
            span.set_attributes(event.otel_attributes());
            span.set_status(Status::Ok);
            span.end_with_timestamp(event.timestamp().into());
        } else {
            drop(spans);
            self.emit_instant_event(event);
        }
    }

    fn emit_instant_event(&self, event: &Event) {
        let tracer = global::tracer("agentd");
        let builder = tracer
            .span_builder(event.name())
            .with_kind(SpanKind::Internal)
            .with_start_time(event.timestamp())
            .with_attributes(event.otel_attributes());
        let mut span = tracer.build(builder);
        span.set_status(Status::Ok);
        span.end_with_timestamp(event.timestamp().into());
    }
}

#[async_trait]
impl TraceSink for OtelTraceSink {
    async fn emit(&self, event: &Event) -> Result<()> {
        if event.is_start() {
            self.start_span(event);
        } else if event.is_end() {
            self.finish_span(event);
        } else {
            self.emit_instant_event(event);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct TraceLogger {
    run_id: String,
    path: PathBuf,
    next_op_id: Arc<AtomicU64>,
    sinks: Arc<Vec<Arc<dyn TraceSink>>>,
}

impl TraceLogger {
    pub fn new(run_id: impl Into<String>, path: PathBuf) -> Self {
        let sink = JsonlTraceSink::new(path.clone());
        Self::with_sinks(run_id, path, vec![Arc::new(sink)])
    }

    pub fn with_sinks(
        run_id: impl Into<String>,
        path: PathBuf,
        sinks: Vec<Arc<dyn TraceSink>>,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            path,
            next_op_id: Arc::new(AtomicU64::new(1)),
            sinks: Arc::new(sinks),
        }
    }

    pub fn mirror_stdout(mut self, mirror_stdout: bool) -> Self {
        let sink = JsonlTraceSink::new(self.path.clone()).mirror_stdout(mirror_stdout);
        self.sinks = Arc::new(vec![Arc::new(sink)]);
        self
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn next_op_id(&self) -> u64 {
        self.next_op_id.fetch_add(1, Ordering::Relaxed)
    }

    pub async fn emit(&self, event: &Event) -> Result<()> {
        for sink in self.sinks.iter() {
            sink.emit(event).await?;
        }
        Ok(())
    }

    pub async fn read_events(path: impl AsRef<Path>) -> Result<Vec<Event>> {
        let file = tokio::fs::File::open(path).await?;
        let mut lines = tokio::io::BufReader::new(file).lines();
        let mut events = Vec::new();
        while let Some(line) = lines.next_line().await? {
            if !line.trim().is_empty() {
                events.push(serde_json::from_str(&line)?);
            }
        }
        Ok(events)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraceSummary {
    pub total_tokens: u32,
    pub infer_calls: usize,
    pub eval_calls: usize,
    pub get_calls: usize,
    pub put_calls: usize,
}

impl TraceSummary {
    pub fn from_events(events: &[Event]) -> Self {
        let mut summary = Self::default();
        for event in events {
            match event {
                Event::InferCall { .. } => summary.infer_calls += 1,
                Event::InferResult { tokens, .. } => summary.total_tokens += *tokens,
                Event::EvalCall { .. } => summary.eval_calls += 1,
                Event::GetCall { .. } => summary.get_calls += 1,
                Event::PutCall { .. } => summary.put_calls += 1,
                _ => {}
            }
        }
        summary
    }
}

pub fn preview(input: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in input.chars().take(max_chars) {
        out.push(ch);
    }
    if input.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use uuid::Uuid;

    #[derive(Default)]
    struct RecordingSink {
        events: Mutex<Vec<Event>>,
    }

    #[async_trait]
    impl TraceSink for RecordingSink {
        async fn emit(&self, event: &Event) -> Result<()> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn done_event(run_id: &str) -> Event {
        Event::AgentDone {
            run_id: run_id.into(),
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn trace_logger_emits_to_all_sinks_and_preserves_jsonl_readback() -> Result<()> {
        let run_id = Uuid::new_v4().to_string();
        let path = std::env::temp_dir().join(format!("trace-sink-test-{run_id}.jsonl"));
        let recording = Arc::new(RecordingSink::default());
        let logger = TraceLogger::with_sinks(
            run_id.clone(),
            path.clone(),
            vec![
                Arc::new(JsonlTraceSink::new(path.clone())),
                recording.clone(),
            ],
        );
        let event = done_event(&run_id);

        logger.emit(&event).await?;

        assert_eq!(
            recording.events.lock().unwrap().as_slice(),
            std::slice::from_ref(&event)
        );
        assert_eq!(TraceLogger::read_events(&path).await?, vec![event]);
        let _ = tokio::fs::remove_file(path).await;
        Ok(())
    }
}
