//! File-backed memory source (t-1160): the first real backend for the
//! hydration system's `semantic:` namespace.
//!
//! A memory is one markdown file holding one fact, with YAML frontmatter
//! (`name`, `description`, optional `metadata.type`) — the same shape Claude
//! Code memories use, so a memory directory is human-curated and
//! agent-readable with no migration. Retrieval is deterministic keyword
//! scoring over name/description/body, optionally blended with embedding
//! cosine similarity when an [`Embedder`] is configured (t-1340, see
//! [`crate::embedding`]): the keyword path needs no network and stays
//! evaluable offline, and every embedding failure degrades back to it.
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

use crate::embedding::{content_hash, cosine, Embedder, EmbeddingIndex};
use crate::hydration::{
    HydrationSink, HydrationSource, Provenance, SinkId, SinkItem, SinkWritePolicy,
    SourceCapability, SourceKind, SourceParams, SourceResult,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

/// Default cap on rendered memory bytes per retrieval; callers override via
/// `SourceParams.max_bytes`.
const DEFAULT_MAX_BYTES: usize = 16 * 1024;

/// Score weights: a query word matching the memory's name is worth more
/// than one matching the description, which beats one buried in the body.
const NAME_WEIGHT: f64 = 3.0;
const DESCRIPTION_WEIGHT: f64 = 2.0;
const BODY_WEIGHT: f64 = 1.0;

/// Ranking blend (t-1340): `score = keyword + SEMANTIC_WEIGHT * cosine`
/// (cosine clamped to `[0, 1]`). The blend is keyword-dominant by design —
/// a perfect semantic match is worth about one name+body keyword hit — so
/// embeddings act as a tie-breaker among keyword matches and as a recall
/// net for paraphrased queries, without letting a fuzzy cosine outvote an
/// exact term match. Chosen over a hard gate so the two signals compose in
/// one deterministic ordering.
const SEMANTIC_WEIGHT: f64 = 4.0;

/// A memory with zero keyword overlap is only included when its cosine
/// clears this floor. Real embedding spaces score *unrelated* text well
/// above zero, so "any positive cosine" would make every memory relevant
/// to every query — the floor is what preserves the "misses are omitted"
/// contract on the semantic path.
const SEMANTIC_FLOOR: f64 = 0.30;

pub struct MemorySource {
    root: PathBuf,
    max_bytes: usize,
    /// Absent = keyword-only retrieval, the zero-cost default; nothing on
    /// that path touches the index or the network.
    embedder: Option<Arc<dyn Embedder>>,
    /// Serializes read-modify-write cycles on the index file within this
    /// process. The index is a rebuildable cache, so a cross-process race
    /// at worst wastes a re-embed.
    index_lock: tokio::sync::Mutex<()>,
}

impl MemorySource {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_bytes: DEFAULT_MAX_BYTES,
            embedder: None,
            index_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Attach the optional embedder (t-1340). `None` is the keyword-only
    /// configuration and is exactly `new()` — call sites can pass their
    /// `Option` straight through.
    pub fn with_embedder(mut self, embedder: Option<Arc<dyn Embedder>>) -> Self {
        self.embedder = embedder;
        self
    }

    /// The vector index sidecar: `<root>/.index/embeddings.json`. Lives
    /// inside the memory dir so it travels with it, but under a dot-dir the
    /// memory loader already ignores.
    fn index_path(&self) -> PathBuf {
        self.root.join(".index").join("embeddings.json")
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

    /// The text a memory is embedded from. Must stay in lockstep with
    /// [`embed_text_parts`] (the write half embeds from the payload before
    /// the file is re-read) or content hashes will never match and every
    /// query re-embeds everything.
    fn embed_text(&self) -> String {
        embed_text_parts(&self.name, &self.description, &self.body)
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

/// Canonical embeddable text for a memory: name, description, and body,
/// newline-joined. One definition shared by the read half (parsed files)
/// and the write half (sink payloads) so the (id, content hash) index key
/// agrees across both.
fn embed_text_parts(name: &str, description: &str, body: &str) -> String {
    format!("{name}\n{description}\n{body}")
}

impl MemorySource {
    /// Semantic scores by memory name for one query: prune vectors of
    /// deleted memories, lazily backfill un-embedded ones, embed the query,
    /// and cosine against the index — all in one batch `embed` call.
    ///
    /// Best-effort throughout: no embedder, no memories, an endpoint
    /// failure, or a malformed response all return an empty map, which the
    /// caller treats as "keyword ranking only". An embedding outage must
    /// never fail a Retrieve (docs/MEMORY.md).
    async fn semantic_scores(&self, query: &str, memories: &[Memory]) -> HashMap<String, f64> {
        let Some(embedder) = &self.embedder else {
            return HashMap::new();
        };
        if memories.is_empty() {
            return HashMap::new();
        }
        let _guard = self.index_lock.lock().await;
        let path = self.index_path();
        let mut index = EmbeddingIndex::load(&path, embedder.model_id()).await;

        // Drop vectors of memories that no longer exist (covers files
        // deleted out-of-band, where the delete() hook never ran).
        let live: HashSet<&str> = memories.iter().map(|memory| memory.name.as_str()).collect();
        let mut dirty = index.retain_ids(&live);

        // One batch call embeds the query plus every stale/missing memory.
        let hashed: Vec<(&Memory, String, String)> = memories
            .iter()
            .map(|memory| {
                let text = memory.embed_text();
                let hash = content_hash(&text);
                (memory, hash, text)
            })
            .collect();
        let missing: Vec<&(&Memory, String, String)> = hashed
            .iter()
            .filter(|(memory, hash, _)| index.vector(&memory.name, hash).is_none())
            .collect();
        let mut inputs = Vec::with_capacity(1 + missing.len());
        inputs.push(query.to_string());
        inputs.extend(missing.iter().map(|(_, _, text)| text.clone()));

        let query_vector = match embedder.embed(&inputs).await {
            Ok(mut vectors) if vectors.len() == inputs.len() => {
                let backfill = vectors.split_off(1);
                for ((memory, hash, _), vector) in missing.into_iter().zip(backfill) {
                    index.insert(memory.name.clone(), hash.clone(), vector);
                    dirty = true;
                }
                vectors.pop().expect("vectors[0] is the query embedding")
            }
            Ok(vectors) => {
                tracing::warn!(
                    got = vectors.len(),
                    expected = inputs.len(),
                    "embedder returned wrong vector count; keyword ranking only"
                );
                if dirty {
                    save_index_best_effort(&index, &path).await;
                }
                return HashMap::new();
            }
            Err(err) => {
                tracing::warn!(error = %format!("{err:#}"), "embedding failed; keyword ranking only");
                if dirty {
                    save_index_best_effort(&index, &path).await;
                }
                return HashMap::new();
            }
        };
        if dirty {
            save_index_best_effort(&index, &path).await;
        }

        hashed
            .into_iter()
            .filter_map(|(memory, hash, _)| {
                index.vector(&memory.name, &hash).map(|vector| {
                    (
                        memory.name.clone(),
                        f64::from(cosine(&query_vector, vector)),
                    )
                })
            })
            .collect()
    }

    /// Write-side hook: embed one memory and upsert its vector, keyed by
    /// (name, content hash) so unchanged content is a no-op and changed
    /// content re-embeds. Best-effort — a Store/Update must succeed even
    /// with the embedding endpoint down (the query path backfills later).
    async fn embed_memory_best_effort(&self, name: &str, description: &str, body: &str) {
        let Some(embedder) = &self.embedder else {
            return;
        };
        let text = embed_text_parts(name, description, body);
        let hash = content_hash(&text);
        let _guard = self.index_lock.lock().await;
        let path = self.index_path();
        let mut index = EmbeddingIndex::load(&path, embedder.model_id()).await;
        if index.vector(name, &hash).is_some() {
            return;
        }
        match embedder.embed(std::slice::from_ref(&text)).await {
            Ok(mut vectors) if vectors.len() == 1 => {
                index.insert(name.to_string(), hash, vectors.remove(0));
                save_index_best_effort(&index, &path).await;
            }
            Ok(vectors) => tracing::warn!(
                memory = name,
                got = vectors.len(),
                "embedder returned wrong vector count; memory left un-embedded"
            ),
            Err(err) => tracing::warn!(
                memory = name,
                error = %format!("{err:#}"),
                "embedding memory failed; will backfill on a later query"
            ),
        }
    }

    /// Write-side hook for delete: drop the memory's vector. Best-effort;
    /// the query path also prunes dead ids, so a miss here self-heals.
    async fn drop_vector_best_effort(&self, name: &str) {
        let Some(embedder) = &self.embedder else {
            return;
        };
        let _guard = self.index_lock.lock().await;
        let path = self.index_path();
        let mut index = EmbeddingIndex::load(&path, embedder.model_id()).await;
        if index.remove(name) {
            save_index_best_effort(&index, &path).await;
        }
    }
}

/// The index is a rebuildable cache: failing to persist it costs a future
/// re-embed, never the operation that produced it.
async fn save_index_best_effort(index: &EmbeddingIndex, path: &std::path::Path) {
    if let Err(err) = index.save(path).await {
        tracing::warn!(error = %format!("{err:#}"), "saving embedding index failed");
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

    /// With a query: scored memories, best first, misses omitted — keyword
    /// scoring blended with embedding cosine when an embedder is configured
    /// (see [`SEMANTIC_WEIGHT`]/[`SEMANTIC_FLOOR`]; every embedding failure
    /// silently reverts to pure keyword ranking). Without a query: the
    /// index (name + description per memory), so a passive caller can see
    /// what is rememberable. Both respect the byte cap and are
    /// deterministic given the same index state (score desc, then name).
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
        let memories = self.load_memories().await?;
        let max_bytes = params.max_bytes.unwrap_or(self.max_bytes);

        let (selected, index_only, semantic) = match params.query.as_deref() {
            Some(query) if !query.trim().is_empty() => {
                // Words under 3 chars ("i", "a", "of") match everything and
                // make every memory relevant to every query; drop them.
                let words: Vec<String> = query
                    .to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|word| word.len() >= 3)
                    .map(str::to_string)
                    .collect();
                // Empty on any failure => the blend below degrades to
                // exactly the old keyword-only ranking.
                let semantic = self.semantic_scores(query, &memories).await;
                let mut scored: Vec<(f64, &Memory)> = memories
                    .iter()
                    .filter_map(|memory| {
                        let keyword = memory.score(&words);
                        let similarity = semantic
                            .get(&memory.name)
                            .copied()
                            .unwrap_or(0.0)
                            .clamp(0.0, 1.0);
                        (keyword > 0.0 || similarity >= SEMANTIC_FLOOR)
                            .then_some((keyword + SEMANTIC_WEIGHT * similarity, memory))
                    })
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
                    !semantic.is_empty(),
                )
            }
            _ => (memories.iter().collect::<Vec<_>>(), true, false),
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
                // Whether embedding similarity contributed to this ranking;
                // false = keyword-only (no embedder, or it degraded).
                "semantic": semantic,
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
        self.embed_memory_best_effort(&slug, &payload.description, payload.body.trim())
            .await;
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
            .with_context(|| format!("rewriting memory {}", path.display()))?;
        // Keyed by content hash, so unchanged content is a no-op and
        // changed content re-embeds.
        self.embed_memory_best_effort(&id.0, &payload.description, payload.body.trim())
            .await;
        Ok(())
    }

    async fn delete(&self, id: &SinkId) -> Result<()> {
        // Validate before building the path: an unvalidated id like
        // "../foo" would otherwise delete files outside the memory dir
        // (t-1182 review). Hard delete by decision — git is the tombstone.
        validate_slug(&id.0)?;
        let path = self.memory_path(id);
        tokio::fs::remove_file(&path)
            .await
            .with_context(|| format!("deleting memory {:?}", id.0))?;
        self.drop_vector_best_effort(&id.0).await;
        Ok(())
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

    // ---- semantic retrieval (t-1340): all offline via MockEmbedder ----

    use std::sync::atomic::{AtomicBool, Ordering};

    /// Deterministic offline embedder: each dimension counts occurrences of
    /// a synonym group, so tests control cosine exactly — and, crucially,
    /// semantic similarity can exist without keyword overlap ("feline"
    /// embeds onto the same axis as "cat").
    struct MockEmbedder {
        fail: AtomicBool,
        calls: std::sync::Mutex<Vec<Vec<String>>>,
    }

    fn mock_vector(text: &str) -> Vec<f32> {
        const AXES: [&[&str]; 3] = [
            &["cat", "feline"],
            &["deploy", "release"],
            &["rust", "crab"],
        ];
        let lower = text.to_lowercase();
        AXES.iter()
            .map(|synonyms| {
                synonyms
                    .iter()
                    .map(|synonym| lower.matches(synonym).count())
                    .sum::<usize>() as f32
            })
            .collect()
    }

    impl MockEmbedder {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                fail: AtomicBool::new(false),
                calls: std::sync::Mutex::new(Vec::new()),
            })
        }

        fn failing() -> Arc<Self> {
            let embedder = Self::new();
            embedder.fail.store(true, Ordering::SeqCst);
            embedder
        }

        fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl crate::embedding::Embedder for MockEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.to_vec());
            if self.fail.load(Ordering::SeqCst) {
                return Err(anyhow!("mock embedding endpoint down"));
            }
            Ok(texts.iter().map(|text| mock_vector(text)).collect())
        }

        fn model_id(&self) -> &str {
            "mock-embedder"
        }
    }

    fn semantic_backend(dir: PathBuf, embedder: &Arc<MockEmbedder>) -> MemorySource {
        MemorySource::new(dir).with_embedder(Some(embedder.clone() as Arc<dyn Embedder>))
    }

    fn cat_note() -> serde_json::Value {
        serde_json::json!({
            "name": "cat-care",
            "description": "feeding schedule",
            "body": "the cat eats at dawn",
        })
    }

    #[tokio::test]
    async fn store_embeds_incrementally_and_query_reuses_the_vector() {
        let dir = temp_memory_dir();
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir.clone(), &embedder);

        backend.store(item(cat_note())).await.unwrap();

        // Exactly one embed call, with exactly the memory's canonical text,
        // and the vector lands in the sidecar under (id, content hash).
        let calls = embedder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0],
            vec![embed_text_parts(
                "cat-care",
                "feeding schedule",
                "the cat eats at dawn"
            )]
        );
        let index =
            EmbeddingIndex::load(&dir.join(".index").join("embeddings.json"), "mock-embedder")
                .await;
        assert_eq!(index.len(), 1);
        assert_eq!(
            index.vector("cat-care", &content_hash(&calls[0][0])),
            Some(mock_vector(&calls[0][0]).as_slice())
        );

        // A query embeds only itself — the stored memory is already
        // indexed — and ranks the semantic match despite zero keyword
        // overlap ("feline" never appears in the memory).
        let result = backend.retrieve(SourceParams::new("feline")).await.unwrap();
        assert_eq!(embedder.calls().len(), 2);
        assert_eq!(embedder.calls()[1], vec!["feline".to_string()]);
        assert_eq!(result.metadata["semantic"], true);
        assert!(
            result.content.contains("### cat-care"),
            "{}",
            result.content
        );
    }

    #[tokio::test]
    async fn update_reembeds_changed_content_and_skips_unchanged() {
        let dir = temp_memory_dir();
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir.clone(), &embedder);
        let id = backend.store(item(cat_note())).await.unwrap();

        let mut revised = cat_note();
        revised["body"] = "the cat eats at dusk".into();
        backend.update(&id, item(revised.clone())).await.unwrap();
        assert_eq!(embedder.calls().len(), 2, "changed content re-embeds");

        let index =
            EmbeddingIndex::load(&dir.join(".index").join("embeddings.json"), "mock-embedder")
                .await;
        let new_text = embed_text_parts("cat-care", "feeding schedule", "the cat eats at dusk");
        assert_eq!(
            index.vector("cat-care", &content_hash(&new_text)),
            Some(mock_vector(&new_text).as_slice()),
            "index holds the new content's vector"
        );

        // Same content again: the (id, hash) key hits, so no embed call.
        backend.update(&id, item(revised)).await.unwrap();
        assert_eq!(embedder.calls().len(), 2, "unchanged content is a no-op");
    }

    #[tokio::test]
    async fn delete_drops_the_vector() {
        let dir = temp_memory_dir();
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir.clone(), &embedder);
        let id = backend.store(item(cat_note())).await.unwrap();

        backend.delete(&id).await.unwrap();

        let index =
            EmbeddingIndex::load(&dir.join(".index").join("embeddings.json"), "mock-embedder")
                .await;
        assert!(index.is_empty());
    }

    #[tokio::test]
    async fn first_query_lazily_backfills_hand_written_memories() {
        let dir = temp_memory_dir();
        write_memory(&dir, "rust.md", RUST_MEMORY).await;
        write_memory(&dir, "deploy.md", DEPLOY_MEMORY).await;
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir.clone(), &embedder);

        backend
            .retrieve(SourceParams::new("release the deploy"))
            .await
            .unwrap();

        // One batch call: the query plus both un-embedded memories.
        let calls = embedder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 3, "query + 2 backfills: {:?}", calls[0]);
        let index =
            EmbeddingIndex::load(&dir.join(".index").join("embeddings.json"), "mock-embedder")
                .await;
        assert_eq!(index.len(), 2);

        // The second query finds everything cached.
        backend
            .retrieve(SourceParams::new("release the deploy"))
            .await
            .unwrap();
        assert_eq!(embedder.calls()[1].len(), 1, "only the query re-embeds");
    }

    #[tokio::test]
    async fn blended_score_breaks_keyword_ties_semantically() {
        let dir = temp_memory_dir();
        // Symmetric keyword profiles ("notes" in name and description of
        // both), asymmetric semantics (only one is about cats).
        write_memory(
            &dir,
            "a.md",
            "---\nname: cat-notes\ndescription: notes about my cat\n---\n\nthe cat purrs",
        )
        .await;
        write_memory(
            &dir,
            "b.md",
            "---\nname: dog-notes\ndescription: notes about my dog\n---\n\nthe dog barks",
        )
        .await;
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir, &embedder);

        // "feline" shares no keyword with either memory; "notes" ties them.
        let result = backend
            .retrieve(SourceParams::new("feline notes"))
            .await
            .unwrap();

        assert_eq!(result.metadata["semantic"], true);
        assert_eq!(result.metadata["memories"][0], "cat-notes");
        assert_eq!(result.metadata["memories"][1], "dog-notes");
        assert!(
            result.content.starts_with("### cat-notes"),
            "{}",
            result.content
        );
    }

    #[tokio::test]
    async fn semantic_only_match_needs_the_floor_and_misses_stay_omitted() {
        let dir = temp_memory_dir();
        write_memory(
            &dir,
            "cat.md",
            "---\nname: cat-care\ndescription: pet routine\n---\n\nthe cat eats at dawn",
        )
        .await;
        write_memory(&dir, "deploy.md", DEPLOY_MEMORY).await;
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir, &embedder);

        // Zero keyword overlap with both; cosine 1.0 with the cat memory,
        // cosine 0.0 (below SEMANTIC_FLOOR) with the deploy memory.
        let result = backend.retrieve(SourceParams::new("feline")).await.unwrap();

        assert!(
            result.content.contains("### cat-care"),
            "{}",
            result.content
        );
        assert!(
            !result.content.contains("deploy-runbook"),
            "below-floor memories stay omitted: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn embedder_outage_degrades_every_path_without_erroring() {
        let dir = temp_memory_dir();
        write_memory(&dir, "rust.md", RUST_MEMORY).await;
        write_memory(&dir, "deploy.md", DEPLOY_MEMORY).await;
        let embedder = MockEmbedder::failing();
        let backend = semantic_backend(dir.clone(), &embedder);

        // Store succeeds with the endpoint down; the memory is just left
        // un-embedded (a later query backfills).
        backend.store(item(cat_note())).await.unwrap();
        assert!(dir.join("cat-care.md").exists());

        // Retrieve degrades to exactly the keyword ranking.
        let result = backend
            .retrieve(SourceParams::new("how should I deploy agentd"))
            .await
            .unwrap();
        assert_eq!(result.metadata["semantic"], false);
        assert!(result.content.starts_with("### deploy-runbook"));

        // Update and delete likewise never surface the outage.
        let mut revised = cat_note();
        revised["body"] = "revised".into();
        backend
            .update(&SinkId("cat-care".into()), item(revised))
            .await
            .unwrap();
        backend.delete(&SinkId("cat-care".into())).await.unwrap();
    }

    #[tokio::test]
    async fn corrupt_or_empty_index_is_rebuilt_not_fatal() {
        let dir = temp_memory_dir();
        write_memory(
            &dir,
            "cat.md",
            "---\nname: cat-care\ndescription: pet routine\n---\n\nthe cat eats at dawn",
        )
        .await;
        let index_path = dir.join(".index").join("embeddings.json");
        tokio::fs::create_dir_all(index_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&index_path, "{corrupt").await.unwrap();
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir, &embedder);

        let result = backend.retrieve(SourceParams::new("feline")).await.unwrap();

        assert_eq!(result.metadata["semantic"], true);
        assert!(result.content.contains("### cat-care"));
        let rebuilt = EmbeddingIndex::load(&index_path, "mock-embedder").await;
        assert_eq!(rebuilt.len(), 1, "corrupt index was rebuilt in place");
    }

    #[tokio::test]
    async fn query_prunes_vectors_of_memories_deleted_out_of_band() {
        let dir = temp_memory_dir();
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir.clone(), &embedder);
        backend.store(item(cat_note())).await.unwrap();
        backend.store(item(note("keeper"))).await.unwrap();

        // Delete the file directly — the delete() hook never runs.
        tokio::fs::remove_file(dir.join("cat-care.md"))
            .await
            .unwrap();

        backend.retrieve(SourceParams::new("keep")).await.unwrap();
        let index =
            EmbeddingIndex::load(&dir.join(".index").join("embeddings.json"), "mock-embedder")
                .await;
        assert_eq!(index.len(), 1, "dead id pruned on query");
        assert!(index.vector("cat-care", "any").is_none());
    }

    #[tokio::test]
    async fn empty_memory_dir_with_embedder_never_calls_it() {
        let dir = temp_memory_dir();
        let embedder = MockEmbedder::new();
        let backend = semantic_backend(dir, &embedder);

        let result = backend
            .retrieve(SourceParams::new("anything"))
            .await
            .unwrap();

        assert!(result.content.is_empty());
        assert_eq!(result.metadata["semantic"], false);
        assert!(embedder.calls().is_empty(), "no memories, no embed call");
    }

    #[tokio::test]
    async fn no_embedder_is_zero_cost_and_keyword_only() {
        let dir = temp_memory_dir();
        let backend = MemorySource::new(dir.clone());
        backend.store(item(cat_note())).await.unwrap();

        let result = backend.retrieve(SourceParams::new("cat")).await.unwrap();

        assert_eq!(result.metadata["semantic"], false);
        assert!(result.content.contains("### cat-care"));
        assert!(
            !dir.join(".index").exists(),
            "no embedder => no index sidecar is ever created"
        );
    }
}
