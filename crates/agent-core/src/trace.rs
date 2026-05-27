use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "PascalCase")]
pub enum Event {
    InferStart {
        run_id: String,
        model: String,
        timestamp: DateTime<Utc>,
    },
    InferEnd {
        run_id: String,
        tokens: u32,
        timestamp: DateTime<Utc>,
    },
    EvalCall {
        run_id: String,
        command: String,
        timestamp: DateTime<Utc>,
    },
    EvalResult {
        run_id: String,
        command: String,
        result: Value,
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
}

impl TraceLogger {
    pub fn new(run_id: impl Into<String>, path: PathBuf) -> Self {
        Self {
            run_id: run_id.into(),
            path,
        }
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
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
        file.write_all(serde_json::to_string(event)?.as_bytes())
            .await?;
        file.write_all(b"\n").await?;
        Ok(())
    }
}
