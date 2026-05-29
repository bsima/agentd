use crate::op::{Prompt, Response};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Clone)]
pub struct TraceLogger {
    run_id: String,
    path: PathBuf,
    next_op_id: Arc<AtomicU64>,
    mirror_stdout: bool,
}

impl TraceLogger {
    pub fn new(run_id: impl Into<String>, path: PathBuf) -> Self {
        Self {
            run_id: run_id.into(),
            path,
            next_op_id: Arc::new(AtomicU64::new(1)),
            mirror_stdout: false,
        }
    }

    pub fn mirror_stdout(mut self, mirror_stdout: bool) -> Self {
        self.mirror_stdout = mirror_stdout;
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
