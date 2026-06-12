//! File-backed memory source (t-1160): the first real backend for the
//! hydration system's `semantic:` namespace.
//!
//! A memory is one markdown file holding one fact, with YAML frontmatter
//! (`name`, `description`, optional `metadata.type`) — the same shape Claude
//! Code memories use, so a memory directory is human-curated and
//! agent-readable with no migration. Retrieval is deterministic keyword
//! scoring over name/description/body: no embeddings, no network, evaluable
//! offline. READ-ONLY by design — the write path is gated on the Get/Put v2
//! design (t-1165); until then, agents read memories and humans write them.

use crate::hydration::{HydrationSource, SourceCapability, SourceKind, SourceParams, SourceResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use std::path::PathBuf;

/// Default cap on rendered memory bytes per retrieval; callers override via
/// `SourceParams.max_bytes`.
const DEFAULT_MAX_BYTES: usize = 16 * 1024;

/// Score weights: a query word matching the memory's name is worth more
/// than one matching the description, which beats one buried in the body.
const NAME_WEIGHT: f64 = 3.0;
const DESCRIPTION_WEIGHT: f64 = 2.0;
const BODY_WEIGHT: f64 = 1.0;

pub struct MemorySource {
    root: PathBuf,
    max_bytes: usize,
}

impl MemorySource {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    async fn load_memories(&self) -> Result<Vec<Memory>> {
        let mut entries = match tokio::fs::read_dir(&self.root).await {
            Ok(entries) => entries,
            // A missing memory directory is an empty memory, not an error:
            // fresh agents have nothing to remember yet.
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("reading memory directory {}", self.root.display()))
            }
        };
        let mut memories = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if !entry.file_type().await?.is_file() || path.extension().is_none_or(|ext| ext != "md")
            {
                continue;
            }
            let raw = tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("reading memory file {}", path.display()))?;
            let fallback_name = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            memories.push(Memory::parse(&raw, fallback_name));
        }
        // Deterministic base order; scoring ties break on this.
        memories.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(memories)
    }
}

#[derive(Debug, Clone)]
struct Memory {
    name: String,
    description: String,
    memory_type: Option<String>,
    body: String,
}

#[derive(Debug, Default, Deserialize)]
struct Frontmatter {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    metadata: FrontmatterMetadata,
}

#[derive(Debug, Default, Deserialize)]
struct FrontmatterMetadata {
    #[serde(rename = "type", default)]
    memory_type: Option<String>,
}

impl Memory {
    /// Frontmatter is best-effort: a file without it (or with YAML that does
    /// not parse) is still a memory — name from the filename, whole content
    /// as body. Memory directories are hand-edited; one malformed file must
    /// not blank the agent's memory.
    fn parse(raw: &str, fallback_name: String) -> Self {
        let (front, body) = split_frontmatter(raw);
        let parsed = front
            .and_then(|yaml| serde_yaml::from_str::<Frontmatter>(yaml).ok())
            .unwrap_or_default();
        Self {
            name: parsed.name.unwrap_or(fallback_name),
            description: parsed.description.unwrap_or_default(),
            memory_type: parsed.metadata.memory_type,
            body: body.trim().to_string(),
        }
    }

    fn score(&self, query_words: &[String]) -> f64 {
        let name = self.name.to_lowercase();
        let description = self.description.to_lowercase();
        let body = self.body.to_lowercase();
        query_words
            .iter()
            .map(|word| {
                let mut score = 0.0;
                if name.contains(word.as_str()) {
                    score += NAME_WEIGHT;
                }
                if description.contains(word.as_str()) {
                    score += DESCRIPTION_WEIGHT;
                }
                if body.contains(word.as_str()) {
                    score += BODY_WEIGHT;
                }
                score
            })
            .sum()
    }

    fn render(&self) -> String {
        let mut header = format!("### {}", self.name);
        if let Some(memory_type) = &self.memory_type {
            header.push_str(&format!(" ({memory_type})"));
        }
        if !self.description.is_empty() {
            header.push_str(&format!(" — {}", self.description));
        }
        format!("{header}\n{}", self.body)
    }
}

/// Split a leading `---\n...\n---` block from the body. Returns
/// `(frontmatter_yaml, body)`; no frontmatter means the whole input is body.
fn split_frontmatter(raw: &str) -> (Option<&str>, &str) {
    let Some(rest) = raw.strip_prefix("---\n").or(raw.strip_prefix("---\r\n")) else {
        return (None, raw);
    };
    for terminator in ["\n---\n", "\n---\r\n"] {
        if let Some(end) = rest.find(terminator) {
            return (Some(&rest[..end]), &rest[end + terminator.len()..]);
        }
    }
    if let Some(front) = rest.strip_suffix("\n---") {
        return (Some(front), "");
    }
    (None, raw)
}

#[async_trait]
impl HydrationSource for MemorySource {
    fn name(&self) -> &str {
        "memory"
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Semantic
    }

    fn capabilities(&self) -> SourceCapability {
        SourceCapability::QUERY
    }

    /// With a query: keyword-scored memories, best first, zero-score
    /// memories omitted. Without one: the index (name + description per
    /// memory), so a passive caller can see what is rememberable. Both
    /// respect the byte cap and are deterministic (score desc, then name).
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
        let memories = self.load_memories().await?;
        let max_bytes = params.max_bytes.unwrap_or(self.max_bytes);

        let (selected, index_only) = match params.query.as_deref() {
            Some(query) if !query.trim().is_empty() => {
                // Words under 3 chars ("i", "a", "of") match everything and
                // make every memory relevant to every query; drop them.
                let words: Vec<String> = query
                    .to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|word| word.len() >= 3)
                    .map(str::to_string)
                    .collect();
                let mut scored: Vec<(f64, &Memory)> = memories
                    .iter()
                    .map(|memory| (memory.score(&words), memory))
                    .filter(|(score, _)| *score > 0.0)
                    .collect();
                scored.sort_by(|(score_a, mem_a), (score_b, mem_b)| {
                    score_b
                        .partial_cmp(score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| mem_a.name.cmp(&mem_b.name))
                });
                (
                    scored.into_iter().map(|(_, memory)| memory).collect(),
                    false,
                )
            }
            _ => (memories.iter().collect::<Vec<_>>(), true),
        };

        let mut content = String::new();
        let mut included = Vec::new();
        for memory in &selected {
            let rendered = if index_only {
                format!("- {} — {}", memory.name, memory.description)
            } else {
                memory.render()
            };
            let separator = if content.is_empty() { 0 } else { 2 };
            if content.len() + separator + rendered.len() > max_bytes {
                break;
            }
            if separator > 0 {
                content.push_str("\n\n");
            }
            content.push_str(&rendered);
            included.push(memory.name.clone());
        }

        Ok(SourceResult {
            source: "memory".into(),
            kind: SourceKind::Semantic,
            content,
            metadata: serde_json::json!({
                "memories": included,
                "total": selected.len(),
                "index_only": index_only,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn write_memory(dir: &std::path::Path, file: &str, content: &str) {
        tokio::fs::write(dir.join(file), content).await.unwrap();
    }

    fn temp_memory_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("agent-memory-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const RUST_MEMORY: &str = "---\nname: prefers-rust-idioms\ndescription: Ben wants code to match surrounding Rust idioms\nmetadata:\n  type: feedback\n---\n\nMatch comment density and naming of the file you edit.";
    const DEPLOY_MEMORY: &str = "---\nname: deploy-runbook\ndescription: production deploy steps for agentd\nmetadata:\n  type: reference\n---\n\nRun the release evals, then push to the deploy branch.";

    #[tokio::test]
    async fn query_ranks_by_keyword_relevance_and_omits_misses() {
        let dir = temp_memory_dir();
        write_memory(&dir, "rust.md", RUST_MEMORY).await;
        write_memory(&dir, "deploy.md", DEPLOY_MEMORY).await;

        let source = MemorySource::new(dir);
        let result = source
            .retrieve(SourceParams::new("how should I deploy agentd"))
            .await
            .unwrap();

        assert!(result.content.starts_with("### deploy-runbook"));
        assert!(
            !result.content.contains("prefers-rust-idioms"),
            "zero-score memories are omitted: {}",
            result.content
        );
        assert_eq!(result.metadata["memories"][0], "deploy-runbook");
    }

    #[tokio::test]
    async fn empty_query_returns_the_index() {
        let dir = temp_memory_dir();
        write_memory(&dir, "rust.md", RUST_MEMORY).await;
        write_memory(&dir, "deploy.md", DEPLOY_MEMORY).await;

        let source = MemorySource::new(dir);
        let result = source.retrieve(SourceParams::default()).await.unwrap();

        assert!(result.metadata["index_only"].as_bool().unwrap());
        assert!(result
            .content
            .contains("- deploy-runbook — production deploy steps"));
        assert!(result.content.contains("- prefers-rust-idioms"));
        assert!(!result.content.contains("Match comment density"));
    }

    #[tokio::test]
    async fn byte_cap_limits_rendered_memories_deterministically() {
        let dir = temp_memory_dir();
        write_memory(&dir, "rust.md", RUST_MEMORY).await;
        write_memory(&dir, "deploy.md", DEPLOY_MEMORY).await;

        let source = MemorySource::new(dir);
        // Room for exactly one rendered memory; the runner-up must be cut.
        let mut params = SourceParams::new("agentd deploy rust idioms");
        params.max_bytes = Some(200);
        let result = source.retrieve(params).await.unwrap();

        assert!(result.content.len() <= 200);
        assert_eq!(
            result.metadata["memories"].as_array().unwrap().len(),
            1,
            "only the top-scored memory fits under the cap"
        );
    }

    #[tokio::test]
    async fn missing_directory_is_an_empty_memory_not_an_error() {
        let source = MemorySource::new(PathBuf::from("/nonexistent/memory/dir"));
        let result = source
            .retrieve(SourceParams::new("anything"))
            .await
            .unwrap();
        assert!(result.content.is_empty());
        assert_eq!(result.metadata["total"], 0);
    }

    #[tokio::test]
    async fn malformed_frontmatter_degrades_to_filename_and_body() {
        let dir = temp_memory_dir();
        write_memory(
            &dir,
            "broken.md",
            "---\n: not yaml ::\n---\n\nthe fact survives",
        )
        .await;
        write_memory(&dir, "plain.md", "no frontmatter at all").await;

        let source = MemorySource::new(dir);
        let result = source
            .retrieve(SourceParams::new("fact survives frontmatter"))
            .await
            .unwrap();

        assert!(result.content.contains("### broken"));
        assert!(result.content.contains("the fact survives"));
        assert!(result.content.contains("### plain"));
    }

    #[tokio::test]
    async fn non_markdown_files_are_ignored() {
        let dir = temp_memory_dir();
        write_memory(&dir, "rust.md", RUST_MEMORY).await;
        write_memory(&dir, "notes.txt", "rust rust rust").await;

        let source = MemorySource::new(dir);
        let result = source.retrieve(SourceParams::new("rust")).await.unwrap();

        assert_eq!(result.metadata["total"], 1);
    }
}
