//! File-backed memory source (t-1160): the first real backend for the
//! hydration system's `semantic:` namespace.
//!
//! A memory is one markdown file holding one fact, with YAML frontmatter
//! (`name`, `description`, optional `metadata.type`) — the same shape Claude
//! Code memories use, so a memory directory is human-curated and
//! agent-readable with no migration. Retrieval is deterministic keyword
//! scoring over name/description/body: no embeddings, no network, evaluable
//! offline.
//!
//! The write half (t-1178, per the approved docs/MEMORY.md design)
//! implements [`HydrationSink`]: the payload schema is
//! `{ name?, description?, type?, body }` — validated HERE, by the backend,
//! not by the trait or the IR. The slug doubles as the [`SinkId`]; it is
//! optional and, when absent, derived deterministically from the
//! description (else body) so the `remember { content, name? }` surface
//! works and replay stays deterministic. Hard delete by decision (memory
//! dirs live in git; history is the tombstone); write policy is Free
//! (trace-visible) by decision.

use crate::hydration::{
    HydrationSink, HydrationSource, Provenance, SinkId, SinkItem, SinkWritePolicy,
    SourceCapability, SourceKind, SourceParams, SourceResult,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
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

/// The memory sink's payload schema (docs/MEMORY.md): validated by the
/// backend, not the trait. `name` is the slug and the [`SinkId`]; it is
/// optional — when absent, the slug is derived deterministically from the
/// description (else the body), so the documented `remember { content,
/// name? }` surface works (t-1180) and replay stays deterministic.
#[derive(Debug, Deserialize)]
struct MemoryPayload {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: String,
    #[serde(rename = "type", default)]
    memory_type: Option<String>,
    body: String,
}

/// A valid memory slug: the filename stem, so it must be kebab-case and
/// can never traverse out of the memory dir. Enforced on every
/// path-building operation — store, update, AND delete (t-1182 review:
/// delete previously trusted the id and `SinkId("../foo")` escaped root).
fn is_valid_slug(slug: &str) -> bool {
    !slug.is_empty()
        && slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !slug.starts_with('-')
        && !slug.ends_with('-')
}

fn validate_slug(slug: &str) -> Result<()> {
    if is_valid_slug(slug) {
        Ok(())
    } else {
        Err(anyhow!(
            "memory id {slug:?} must be a kebab-case slug ([a-z0-9-], no leading/trailing dash)"
        ))
    }
}

/// Deterministically derive a kebab slug from free text: lowercase, runs of
/// non-alphanumerics collapse to single dashes, trimmed, capped. Determinism
/// matters — the slug becomes the recorded SinkId, and a non-deterministic
/// one (random/timestamp) would diverge on replay.
fn slugify(text: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = true; // suppress leading dash
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.extend(ch.to_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
        if slug.len() >= 48 {
            break;
        }
    }
    slug.trim_matches('-').to_string()
}

/// Frontmatter written by the sink; matches what the source half parses,
/// plus runtime provenance under metadata.
#[derive(Debug, Serialize)]
struct FrontmatterOut<'a> {
    name: &'a str,
    description: &'a str,
    metadata: FrontmatterMetadataOut<'a>,
}

#[derive(Debug, Serialize)]
struct FrontmatterMetadataOut<'a> {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    memory_type: Option<&'a str>,
    provenance: &'a Provenance,
}

impl MemoryPayload {
    fn parse(item: &SinkItem) -> Result<Self> {
        serde_json::from_value(item.payload.clone())
            .context("memory payload must be { name?, description?, type?, body }")
    }

    /// The slug for this payload: the given `name` if present, else derived
    /// from the description, else the body. Validated to kebab.
    fn slug(&self) -> Result<String> {
        let slug = match self.name.as_deref() {
            Some(name) if !name.trim().is_empty() => name.to_string(),
            _ => {
                let derived = slugify(&self.description);
                if derived.is_empty() {
                    slugify(&self.body)
                } else {
                    derived
                }
            }
        };
        if slug.is_empty() {
            return Err(anyhow!(
                "memory has no name and none could be derived from its description or body"
            ));
        }
        validate_slug(&slug)?;
        Ok(slug)
    }

    fn render(&self, name: &str, provenance: &Provenance) -> Result<String> {
        let front = serde_yaml::to_string(&FrontmatterOut {
            name,
            description: &self.description,
            metadata: FrontmatterMetadataOut {
                memory_type: self.memory_type.as_deref(),
                provenance,
            },
        })?;
        Ok(format!("---\n{front}---\n\n{}\n", self.body.trim()))
    }
}

impl MemorySource {
    fn memory_path(&self, id: &SinkId) -> PathBuf {
        self.root.join(format!("{}.md", id.0))
    }
}

#[async_trait]
impl HydrationSink for MemorySource {
    fn name(&self) -> &str {
        "memory"
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Semantic
    }

    fn write_policy(&self) -> SinkWritePolicy {
        // Settled question 1: free but trace-visible; the trace is the audit.
        SinkWritePolicy::Free
    }

    async fn store(&self, item: SinkItem) -> Result<SinkId> {
        let payload = MemoryPayload::parse(&item)?;
        let slug = payload.slug()?;
        let id = SinkId(slug.clone());
        let path = self.memory_path(&id);
        if tokio::fs::try_exists(&path).await? {
            return Err(anyhow!(
                "memory {slug:?} already exists; use Update to revise it"
            ));
        }
        tokio::fs::create_dir_all(&self.root).await?;
        tokio::fs::write(&path, payload.render(&slug, &item.provenance)?)
            .await
            .with_context(|| format!("writing memory {}", path.display()))?;
        Ok(id)
    }

    async fn update(&self, id: &SinkId, item: SinkItem) -> Result<()> {
        validate_slug(&id.0)?;
        let payload = MemoryPayload::parse(&item)?;
        // If the payload names a slug, it must match the target id; an empty
        // name updates in place under the given id.
        if let Some(name) = payload.name.as_deref() {
            if !name.trim().is_empty() && name != id.0 {
                return Err(anyhow!(
                    "memory update payload renames {:?} to {name:?}; delete and re-create instead",
                    id.0
                ));
            }
        }
        let path = self.memory_path(id);
        if !tokio::fs::try_exists(&path).await? {
            return Err(anyhow!("no memory {:?} to update", id.0));
        }
        tokio::fs::write(&path, payload.render(&id.0, &item.provenance)?)
            .await
            .with_context(|| format!("rewriting memory {}", path.display()))
    }

    async fn delete(&self, id: &SinkId) -> Result<()> {
        // Validate before building the path: an unvalidated id like
        // "../foo" would otherwise delete files outside the memory dir
        // (t-1182 review). Hard delete by decision — git is the tombstone.
        validate_slug(&id.0)?;
        let path = self.memory_path(id);
        tokio::fs::remove_file(&path)
            .await
            .with_context(|| format!("deleting memory {:?}", id.0))
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

    fn item(payload: serde_json::Value) -> SinkItem {
        SinkItem {
            payload,
            provenance: Provenance {
                run_id: "run-test".into(),
                effect_id: Some("effect-1".into()),
                timestamp: None,
            },
        }
    }

    fn note(name: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "description": "a test note",
            "type": "project",
            "body": "the fact to keep",
        })
    }

    #[tokio::test]
    async fn store_then_retrieve_round_trips_with_provenance() {
        let dir = temp_memory_dir();
        let backend = MemorySource::new(dir.clone());

        let id = backend.store(item(note("test-note"))).await.unwrap();
        assert_eq!(id, SinkId("test-note".into()));

        let written = tokio::fs::read_to_string(dir.join("test-note.md"))
            .await
            .unwrap();
        assert!(written.contains("run_id: run-test"), "{written}");
        assert!(written.contains("effect_id: effect-1"), "{written}");

        let result = backend
            .retrieve(SourceParams::new("fact to keep"))
            .await
            .unwrap();
        assert!(result
            .content
            .contains("### test-note (project) — a test note"));
        assert!(result.content.contains("the fact to keep"));
    }

    #[tokio::test]
    async fn store_refuses_duplicates_and_update_revises_in_place() {
        let backend = MemorySource::new(temp_memory_dir());
        let id = backend.store(item(note("twice"))).await.unwrap();

        let err = backend.store(item(note("twice"))).await.unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");

        let mut revised = note("twice");
        revised["body"] = "the revised fact".into();
        backend.update(&id, item(revised)).await.unwrap();
        let result = backend
            .retrieve(SourceParams::new("revised"))
            .await
            .unwrap();
        assert!(result.content.contains("the revised fact"));

        let rename = backend
            .update(&id, item(note("renamed")))
            .await
            .unwrap_err();
        assert!(rename.to_string().contains("renames"), "{rename}");
        let missing = backend
            .update(&SinkId("ghost".into()), item(note("ghost")))
            .await
            .unwrap_err();
        assert!(missing.to_string().contains("no memory"), "{missing}");
    }

    #[tokio::test]
    async fn delete_is_hard_and_unrelated_memories_survive() {
        let dir = temp_memory_dir();
        write_memory(&dir, "keeper.md", RUST_MEMORY).await;
        let backend = MemorySource::new(dir.clone());
        let id = backend.store(item(note("goner"))).await.unwrap();

        backend.delete(&id).await.unwrap();

        assert!(!dir.join("goner.md").exists());
        assert!(dir.join("keeper.md").exists());
        assert!(backend.delete(&id).await.is_err(), "double delete errors");
    }

    #[tokio::test]
    async fn delete_rejects_path_traversal_ids() {
        // t-1182 review: delete must validate the id before building a path,
        // or SinkId("../victim") escapes the memory dir. Use a dedicated
        // enclosing dir so the sibling "victim.md" can't collide with other
        // (parallel) tests sharing the temp root.
        let enclosing =
            std::env::temp_dir().join(format!("agent-memory-traversal-{}", uuid::Uuid::new_v4()));
        let dir = enclosing.join("memory");
        std::fs::create_dir_all(&dir).unwrap();
        let outside = enclosing.join("victim.md");
        tokio::fs::write(&outside, "do not delete me")
            .await
            .unwrap();

        let backend = MemorySource::new(dir);
        let err = backend
            .delete(&SinkId("../victim".into()))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("kebab-case"), "{err}");
        assert!(outside.exists(), "the file outside the memory dir survives");
        std::fs::remove_dir_all(&enclosing).ok();
    }

    #[tokio::test]
    async fn name_is_optional_and_slugged_from_description() {
        let dir = temp_memory_dir();
        let backend = MemorySource::new(dir.clone());

        // No name: the slug is derived from the description, deterministically.
        let id = backend
            .store(item(serde_json::json!({
                "description": "Deploy Window Rules!",
                "body": "deploys only on tuesdays",
            })))
            .await
            .unwrap();
        assert_eq!(id, SinkId("deploy-window-rules".into()));
        assert!(dir.join("deploy-window-rules.md").exists());

        // Re-deriving from the same inputs yields the same slug (replay-safe).
        let again = backend
            .store(item(serde_json::json!({
                "description": "Deploy Window Rules!",
                "body": "x",
            })))
            .await
            .unwrap_err();
        assert!(again.to_string().contains("already exists"), "{again}");
    }

    #[tokio::test]
    async fn payload_schema_and_slug_are_validated() {
        let backend = MemorySource::new(temp_memory_dir());

        // body is the only required field.
        let bad_schema = backend
            .store(item(serde_json::json!({ "description": "x" })))
            .await
            .unwrap_err();
        assert!(
            format!("{bad_schema:#}").contains("name?, description?, type?, body"),
            "{bad_schema:#}"
        );

        // A provided name must be a valid slug (an empty/absent name instead
        // derives one, covered above).
        for bad_name in ["../escape", "Has Spaces", "-lead", "trail-"] {
            let mut payload = note("placeholder");
            payload["name"] = bad_name.into();
            let err = backend.store(item(payload)).await.unwrap_err();
            assert!(
                err.to_string().contains("kebab-case"),
                "{bad_name:?} must be rejected: {err}"
            );
        }
    }

    #[tokio::test]
    async fn register_backend_serves_both_halves() {
        use crate::hydration::SourceRegistry;

        let registry = SourceRegistry::new().register_backend(MemorySource::new(temp_memory_dir()));

        assert_eq!(registry.sources().len(), 1);
        assert_eq!(registry.sinks().len(), 1);
        let sink = registry.sink("memory").expect("sink registered by name");
        assert_eq!(sink.kind(), SourceKind::Semantic);
        assert_eq!(sink.write_policy(), SinkWritePolicy::Free);
        assert_eq!(registry.sinks_of_kind(SourceKind::Semantic).len(), 1);
        assert!(registry.sinks_of_kind(SourceKind::Temporal).is_empty());
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
