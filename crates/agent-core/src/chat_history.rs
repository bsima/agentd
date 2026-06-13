//! Chat-history backend (t-1181): the session's own conversation as a
//! hydration backend, unifying the two halves the design (docs/MEMORY.md)
//! calls for in one type.
//!
//! - **Sink half** (`HydrationSink`): persists a session checkpoint to a
//!   directory. This is what checkpointing *is* once it stops being a
//!   bespoke mechanism — the runtime writes it passively at turn
//!   completion. The on-disk layout (numbered `checkpoint-NNNNNN-<run>.json`
//!   plus `latest.json` and `session-latest.json` pointers) and the
//!   payload schema are unchanged from the pre-t-1181 writer, so the
//!   Haskell agentd and `evals/agentd-persistent.sh` keep working with no
//!   change (Ben, 2026-06-13: clean break is allowed but a no-op schema is
//!   preferred while Haskell compat is maintained).
//! - **Source half** (`HydrationSource`): recency-windowed turn summaries,
//!   folding in t-1164's `TemporalSource` by delegation — same mechanism,
//!   so a future consolidation is cosmetic.
//!
//! The sink writes the payload verbatim; provenance is available on the
//! `SinkItem` but deliberately not persisted, because the checkpoint schema
//! is fixed for compatibility. Other sinks (memory) persist it.

use crate::hydration::{
    HydrationSink, HydrationSource, SinkId, SinkItem, SourceCapability, SourceKind, SourceParams,
    SourceResult,
};
use crate::temporal::TemporalSource;
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::path::PathBuf;

pub struct ChatHistory {
    dir: PathBuf,
    reader: TemporalSource,
}

impl ChatHistory {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            reader: TemporalSource::new(dir.clone()),
            dir,
        }
    }

    /// Write `payload` to all three checkpoint files (numbered history +
    /// the `latest`/`session-latest` pointers), byte-identical to the
    /// pre-t-1181 writer. Returns the numbered checkpoint's id.
    async fn persist(&self, payload: &Value) -> Result<SinkId> {
        let run_id = payload
            .get("run_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("checkpoint payload missing run_id"))?;
        let sequence = payload
            .get("sequence")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("checkpoint payload missing sequence"))?;
        tokio::fs::create_dir_all(&self.dir)
            .await
            .with_context(|| format!("creating checkpoint dir {}", self.dir.display()))?;
        let bytes = serde_json::to_vec_pretty(payload)?;
        let numbered = format!("checkpoint-{sequence:06}-{run_id}.json");
        tokio::fs::write(self.dir.join(&numbered), &bytes).await?;
        tokio::fs::write(self.dir.join("latest.json"), &bytes).await?;
        tokio::fs::write(self.dir.join("session-latest.json"), &bytes).await?;
        Ok(SinkId(format!("checkpoint-{sequence:06}")))
    }
}

#[async_trait]
impl HydrationSink for ChatHistory {
    fn name(&self) -> &str {
        "chat-history"
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Temporal
    }

    async fn store(&self, item: SinkItem) -> Result<SinkId> {
        // A session snapshot is meant to be overwritten each turn, so unlike
        // the memory sink there is no create-vs-update distinction or
        // duplicate refusal.
        self.persist(&item.payload).await
    }

    async fn update(&self, _id: &SinkId, item: SinkItem) -> Result<()> {
        self.persist(&item.payload).await.map(|_| ())
    }

    async fn delete(&self, id: &SinkId) -> Result<()> {
        // Best-effort removal of the numbered checkpoint; the pointers are
        // left for the next snapshot to overwrite.
        let path = self.dir.join(format!("{}.json", id.0));
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err).with_context(|| format!("deleting checkpoint {}", path.display())),
        }
    }
}

#[async_trait]
impl HydrationSource for ChatHistory {
    fn name(&self) -> &str {
        "chat-history"
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Temporal
    }

    fn capabilities(&self) -> SourceCapability {
        SourceCapability::SESSION_CONTEXT | SourceCapability::QUERY
    }

    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
        // The read half is t-1164's recency window, by delegation.
        self.reader.retrieve(params).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("agent-chat-history-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn checkpoint(run_id: &str, sequence: u64, turns: &[(&str, &str)]) -> Value {
        let messages: Vec<Value> = turns
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect();
        serde_json::json!({
            "run_id": run_id,
            "sequence": sequence,
            "model": "m",
            "provider_url": "https://example.com",
            "messages": messages,
            "trace_path": "/tmp/t.jsonl",
            "timestamp": "2026-06-13T00:00:00Z",
        })
    }

    fn item(payload: Value) -> SinkItem {
        SinkItem {
            payload,
            provenance: Default::default(),
        }
    }

    #[tokio::test]
    async fn sink_writes_numbered_history_and_pointers_verbatim() {
        let dir = temp_dir();
        let backend = ChatHistory::new(dir.clone());
        let payload = checkpoint("run-a", 7, &[("user", "hi"), ("assistant", "hello")]);

        let id = backend.store(item(payload.clone())).await.unwrap();
        assert_eq!(id, SinkId("checkpoint-000007".into()));

        // The schema is unchanged: each file is the payload, pretty-printed,
        // and parses straight back.
        let expected = serde_json::to_vec_pretty(&payload).unwrap();
        for file in [
            "checkpoint-000007-run-a.json",
            "latest.json",
            "session-latest.json",
        ] {
            let bytes = tokio::fs::read(dir.join(file)).await.unwrap();
            assert_eq!(bytes, expected, "{file} must be the payload verbatim");
        }
    }

    #[tokio::test]
    async fn sink_overwrites_pointers_each_turn() {
        let dir = temp_dir();
        let backend = ChatHistory::new(dir.clone());
        backend
            .store(item(checkpoint("run-a", 1, &[("user", "first")])))
            .await
            .unwrap();
        backend
            .store(item(checkpoint("run-a", 2, &[("user", "second")])))
            .await
            .unwrap();

        let latest: Value = serde_json::from_slice(
            &tokio::fs::read(dir.join("session-latest.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(latest["sequence"], 2);
        // Both numbered snapshots are retained as history.
        assert!(dir.join("checkpoint-000001-run-a.json").exists());
        assert!(dir.join("checkpoint-000002-run-a.json").exists());
    }

    #[tokio::test]
    async fn one_backend_round_trips_write_then_read() {
        let dir = temp_dir();
        let backend = ChatHistory::new(dir);
        backend
            .store(item(checkpoint(
                "run-a",
                3,
                &[("user", "refactor the parser"), ("assistant", "done")],
            )))
            .await
            .unwrap();

        // The same object's source half reads the recency window back.
        let result = backend.retrieve(SourceParams::default()).await.unwrap();
        assert!(
            result.content.contains("refactor the parser"),
            "{}",
            result.content
        );
        assert_eq!(result.kind, SourceKind::Temporal);
    }

    #[tokio::test]
    async fn missing_run_id_or_sequence_is_an_error() {
        let backend = ChatHistory::new(temp_dir());
        let err = backend
            .store(item(serde_json::json!({ "messages": [] })))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("run_id"), "{err}");
    }
}
