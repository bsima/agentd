//! Checkpoint-backed temporal hydration source (t-1164): cross-session
//! continuity served from the agent's own persisted history.
//!
//! Points at a directory of checkpoint JSONs (the `--checkpoint-dir` shape)
//! belonging to *past or sibling* sessions and serves recency-windowed turn
//! summaries from the newest one. Deliberately NOT auto-registered for the
//! running session's own checkpoint dir — the live history already contains
//! those messages, and passive re-injection would duplicate the window.
//! Wire it with `--temporal-dir` at a supervisor session dir (SUPERVISOR.md)
//! or any archived checkpoint dir.
//!
//! Retrieval v1 is recency + keyword filtering, deterministic (checkpoint
//! `sequence` then filename; no clocks). Designed against t-1139 (time2vec
//! temporal conditioning, omni-side): staleness-aware scoring can replace
//! the plain recency window later without changing the source contract.

use crate::hydration::{HydrationSource, SourceCapability, SourceKind, SourceParams, SourceResult};
use crate::op::ChatMessage;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::path::PathBuf;

/// Turns served per retrieval; callers shrink the rendering further via
/// `SourceParams.max_bytes`.
const DEFAULT_MAX_TURNS: usize = 20;
const DEFAULT_MAX_BYTES: usize = 8 * 1024;
const TURN_PREVIEW_CHARS: usize = 200;

pub struct TemporalSource {
    root: PathBuf,
    max_turns: usize,
    max_bytes: usize,
}

impl TemporalSource {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_turns: DEFAULT_MAX_TURNS,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// The newest checkpoint in the directory: highest `sequence`, filename
    /// as the deterministic tie-break. Malformed or non-checkpoint JSON is
    /// skipped — archived session dirs are long-lived and hand-touched.
    async fn latest_checkpoint(&self) -> Result<Option<CheckpointView>> {
        let mut entries = match tokio::fs::read_dir(&self.root).await {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("reading temporal directory {}", self.root.display()))
            }
        };
        let mut paths = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if entry.file_type().await?.is_file()
                && path.extension().is_some_and(|ext| ext == "json")
            {
                paths.push(path);
            }
        }
        paths.sort();

        let mut latest: Option<CheckpointView> = None;
        for path in paths {
            let Ok(raw) = tokio::fs::read_to_string(&path).await else {
                continue;
            };
            let Some(view) = CheckpointView::parse(&raw) else {
                continue;
            };
            // paths are sorted, so >= keeps the lexicographically-last file
            // among equal sequences: deterministic without mtimes.
            if latest
                .as_ref()
                .is_none_or(|current| view.sequence >= current.sequence)
            {
                latest = Some(view);
            }
        }
        Ok(latest)
    }
}

struct CheckpointView {
    run_id: String,
    sequence: u64,
    messages: Vec<ChatMessage>,
}

impl CheckpointView {
    fn parse(raw: &str) -> Option<Self> {
        let value: Value = serde_json::from_str(raw).ok()?;
        let messages = value.get("messages")?.clone();
        Some(Self {
            run_id: value.get("run_id")?.as_str()?.to_string(),
            sequence: value.get("sequence")?.as_u64()?,
            messages: serde_json::from_value(messages).ok()?,
        })
    }
}

#[async_trait]
impl HydrationSource for TemporalSource {
    fn name(&self) -> &str {
        "temporal-checkpoints"
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Temporal
    }

    fn capabilities(&self) -> SourceCapability {
        SourceCapability::SESSION_CONTEXT | SourceCapability::QUERY
    }

    /// Without a query: the last `max_turns` turns of the newest checkpoint,
    /// previewed one line each, most recent kept under the byte cap. With a
    /// query: the same window over only the turns containing a query word
    /// (>= 3 chars, case-insensitive).
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
        let Some(checkpoint) = self.latest_checkpoint().await? else {
            return Ok(SourceResult {
                source: self.name().into(),
                kind: SourceKind::Temporal,
                content: String::new(),
                metadata: serde_json::json!({ "turns": 0 }),
            });
        };

        let words: Vec<String> = params
            .query
            .as_deref()
            .unwrap_or_default()
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| word.len() >= 3)
            .map(str::to_string)
            .collect();

        let max_bytes = params.max_bytes.unwrap_or(self.max_bytes);
        let mut lines: Vec<String> = Vec::new();
        let mut spent = 0usize;
        // Walk newest-first so the byte cap always sacrifices the oldest.
        for message in checkpoint.messages.iter().rev() {
            if lines.len() >= self.max_turns {
                break;
            }
            let content = message.content.as_deref().unwrap_or_default();
            if content.trim().is_empty() {
                continue;
            }
            if !words.is_empty() {
                let lower = content.to_lowercase();
                if !words.iter().any(|word| lower.contains(word.as_str())) {
                    continue;
                }
            }
            let preview: String = content.trim().chars().take(TURN_PREVIEW_CHARS).collect();
            let line = format!("- {}: {preview}", message.role);
            if spent + line.len() + 1 > max_bytes {
                break;
            }
            spent += line.len() + 1;
            lines.push(line);
        }
        lines.reverse();

        let turns = lines.len();
        let content = if lines.is_empty() {
            String::new()
        } else {
            format!(
                "### recent session turns (run {}, checkpoint {})\n{}",
                checkpoint.run_id,
                checkpoint.sequence,
                lines.join("\n")
            )
        };
        Ok(SourceResult {
            source: self.name().into(),
            kind: SourceKind::Temporal,
            content,
            metadata: serde_json::json!({
                "turns": turns,
                "run_id": checkpoint.run_id,
                "sequence": checkpoint.sequence,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("agent-temporal-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn checkpoint_json(run_id: &str, sequence: u64, turns: &[(&str, &str)]) -> String {
        let messages: Vec<Value> = turns
            .iter()
            .map(|(role, content)| {
                serde_json::to_value(match *role {
                    "user" => ChatMessage::user(*content),
                    "system" => ChatMessage::system(*content),
                    _ => ChatMessage::assistant(Some((*content).into()), vec![]),
                })
                .unwrap()
            })
            .collect();
        serde_json::json!({
            "run_id": run_id, "sequence": sequence, "model": "m",
            "provider_url": "https://example.com", "messages": messages,
            "trace_path": "/tmp/t.jsonl", "timestamp": "2026-06-12T00:00:00Z",
        })
        .to_string()
    }

    #[tokio::test]
    async fn serves_recent_turns_from_the_newest_checkpoint() {
        let dir = temp_dir();
        std::fs::write(
            dir.join("old.json"),
            checkpoint_json("run-a", 3, &[("user", "ancient question")]),
        )
        .unwrap();
        std::fs::write(
            dir.join("latest.json"),
            checkpoint_json(
                "run-a",
                7,
                &[
                    ("user", "refactor the parser"),
                    ("assistant", "done, tests pass"),
                ],
            ),
        )
        .unwrap();

        let result = TemporalSource::new(dir)
            .retrieve(SourceParams::default())
            .await
            .unwrap();

        assert!(
            result.content.contains("checkpoint 7"),
            "{}",
            result.content
        );
        assert!(result.content.contains("- user: refactor the parser"));
        assert!(result.content.contains("- assistant: done, tests pass"));
        assert!(!result.content.contains("ancient"));
        assert_eq!(result.metadata["sequence"], 7);
    }

    #[tokio::test]
    async fn query_filters_turns_by_keyword() {
        let dir = temp_dir();
        std::fs::write(
            dir.join("cp.json"),
            checkpoint_json(
                "run-b",
                1,
                &[
                    ("user", "talk about parsers"),
                    ("assistant", "parsers are fine"),
                    ("user", "now deploy it"),
                ],
            ),
        )
        .unwrap();

        let result = TemporalSource::new(dir)
            .retrieve(SourceParams::new("parser"))
            .await
            .unwrap();

        assert!(result.content.contains("talk about parsers"));
        assert!(result.content.contains("parsers are fine"));
        assert!(!result.content.contains("deploy"));
        assert_eq!(result.metadata["turns"], 2);
    }

    #[tokio::test]
    async fn byte_cap_sacrifices_the_oldest_turns() {
        let dir = temp_dir();
        std::fs::write(
            dir.join("cp.json"),
            checkpoint_json(
                "run-c",
                1,
                &[
                    ("user", &"old ".repeat(40)),
                    ("assistant", &"mid ".repeat(40)),
                    ("user", "newest question"),
                ],
            ),
        )
        .unwrap();

        let params = SourceParams {
            max_bytes: Some(80),
            ..Default::default()
        };
        let result = TemporalSource::new(dir).retrieve(params).await.unwrap();

        assert!(result.content.contains("newest question"));
        assert!(!result.content.contains("old old"));
    }

    #[tokio::test]
    async fn missing_dir_and_malformed_files_yield_empty_content() {
        let missing = TemporalSource::new(PathBuf::from("/nonexistent/temporal"))
            .retrieve(SourceParams::default())
            .await
            .unwrap();
        assert!(missing.content.is_empty());

        let dir = temp_dir();
        std::fs::write(dir.join("junk.json"), "not json").unwrap();
        let junk = TemporalSource::new(dir)
            .retrieve(SourceParams::default())
            .await
            .unwrap();
        assert!(junk.content.is_empty());
        assert_eq!(junk.metadata["turns"], 0);
    }
}
