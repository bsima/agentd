//! Embedding-based semantic retrieval for the memory backend (t-1340).
//!
//! Three pieces, all backend-internal to [`crate::memory::MemorySource`]:
//!
//! - [`Embedder`] / [`EmbeddingClient`]: an OpenAI-compatible
//!   `POST /embeddings` client (OpenRouter, OpenAI, and local servers all
//!   speak this shape), configured through the model registry's optional
//!   `embeddings` section (see [`crate::models::ModelRegistry`]).
//! - [`EmbeddingIndex`]: a local, file-based vector index — one JSON file
//!   under the memory dir's `.index/` sidecar. Vectors are keyed by
//!   (memory id, content hash), so an edited memory re-embeds and a
//!   deleted memory drops its vector. JSON over a binary format by
//!   decision: memory dirs are small (tens of memories, not millions),
//!   human-inspectable, and diffable; the index is a cache that can be
//!   deleted at any time and lazily rebuilt.
//! - [`cosine`]: similarity over raw `f32` vectors — no external vector
//!   DB, no heavy deps.
//!
//! **Replay/effects note (docs/MEMORY.md):** embedding HTTP calls are
//! backend-internal, like a vector DB's internals. They are NOT IR
//! effects, are not cost-attributed, and are never replayed; the
//! `Retrieve` effect records its *results*, which is what keeps replay
//! sound. Every failure here degrades to keyword ranking — an embedding
//! outage must never fail a Retrieve.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

/// Producer of embedding vectors. The one real implementation is
/// [`EmbeddingClient`]; tests substitute deterministic mocks so nothing
/// in the retrieval path needs a network.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts; one vector per input, same order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Identifies the embedding space. Recorded in the index so switching
    /// models invalidates every cached vector (vectors from different
    /// models are not comparable).
    fn model_id(&self) -> &str;
}

/// OpenAI-compatible embeddings client: `POST {base_url}/embeddings` with
/// a bearer token, same transport conventions as `provider.rs`. No retry
/// loop by design — the caller treats any failure as "no semantic ranking
/// this query" and falls back to keywords, so retries would only add
/// latency to a degraded path.
pub struct EmbeddingClient {
    client: reqwest::Client,
    url: String,
    api_key: String,
    model: String,
}

impl EmbeddingClient {
    pub fn new(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("building embeddings HTTP client"),
            url: format!("{}/embeddings", base_url.as_ref().trim_end_matches('/')),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// Build the client from the registry's optional `embeddings` section:
    /// `Ok(None)` when the section is absent (keyword-only retrieval);
    /// fails closed when it names an unknown alias or an entry the client
    /// cannot use (no base_url).
    pub fn from_registry(registry: &crate::models::ModelRegistry) -> Result<Option<Self>> {
        let Some(resolved) = registry.resolve_embeddings()? else {
            return Ok(None);
        };
        let base_url = resolved.base_url.as_deref().ok_or_else(|| {
            anyhow!(
                "embeddings model {:?} has no base_url; the embeddings client is \
                 OpenAI-compatible and needs one",
                resolved.alias
            )
        })?;
        Ok(Some(Self::new(
            base_url,
            resolved.api_key.unwrap_or_default(),
            resolved.api_id,
        )))
    }
}

#[async_trait]
impl Embedder for EmbeddingClient {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let body = json!({ "model": self.model, "input": texts });
        let response = self
            .client
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("sending embeddings request")?;
        let status = response.status();
        let text = response
            .text()
            .await
            .context("reading embeddings response")?;
        if !status.is_success() {
            return Err(anyhow!("embeddings endpoint returned {status}: {text}"));
        }
        parse_embeddings_response(&text, texts.len())
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

/// Parse the OpenAI-compatible embeddings response body. Reorders by the
/// declared `index` (providers may return data out of order) and insists
/// on exactly one vector per input.
pub(crate) fn parse_embeddings_response(text: &str, expected: usize) -> Result<Vec<Vec<f32>>> {
    let parsed: EmbeddingsResponse =
        serde_json::from_str(text).context("parsing embeddings response")?;
    if parsed.data.len() != expected {
        return Err(anyhow!(
            "embeddings endpoint returned {} vectors for {expected} inputs",
            parsed.data.len()
        ));
    }
    let mut out: Vec<Option<Vec<f32>>> = vec![None; expected];
    for datum in parsed.data {
        let slot = out
            .get_mut(datum.index)
            .ok_or_else(|| anyhow!("embeddings response index {} out of range", datum.index))?;
        if slot.replace(datum.embedding).is_some() {
            return Err(anyhow!("embeddings response repeats index {}", datum.index));
        }
    }
    // Length matched and indexes were unique and in range, so every slot
    // is filled.
    Ok(out.into_iter().map(|slot| slot.unwrap()).collect())
}

/// Cosine similarity over raw f32 vectors. Mismatched lengths and zero
/// vectors score 0.0 (unrelated) rather than erroring — a degenerate
/// vector must never fail a retrieval.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Hex SHA-256 of a memory's embedded text; the "content" half of the
/// (memory id, content hash) index key.
pub fn content_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

const INDEX_VERSION: u32 = 1;

/// The on-disk vector index: `{ version, model, entries: { <memory id>:
/// { hash, vector } } }` as JSON. The index is a cache — corrupt, missing,
/// wrong-version, or wrong-model files all load as empty and the caller
/// lazily re-embeds. `BTreeMap` keeps serialization deterministic.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingIndex {
    version: u32,
    model: String,
    entries: BTreeMap<String, IndexEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexEntry {
    hash: String,
    vector: Vec<f32>,
}

impl EmbeddingIndex {
    pub fn empty(model: impl Into<String>) -> Self {
        Self {
            version: INDEX_VERSION,
            model: model.into(),
            entries: BTreeMap::new(),
        }
    }

    /// Load the index for `model`, treating every failure (missing file,
    /// unparsable JSON, version bump, different embedding model) as an
    /// empty index: the vectors are a rebuildable cache, so the only wrong
    /// answer is an error.
    pub async fn load(path: &Path, model: &str) -> Self {
        let Ok(raw) = tokio::fs::read_to_string(path).await else {
            return Self::empty(model);
        };
        match serde_json::from_str::<Self>(&raw) {
            Ok(index) if index.version == INDEX_VERSION && index.model == model => index,
            _ => Self::empty(model),
        }
    }

    /// Persist to `path` via write-to-temp + rename so a crash mid-write
    /// leaves the previous index intact rather than truncated JSON.
    pub async fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating index dir {}", parent.display()))?;
        }
        let bytes = serde_json::to_vec(self)?;
        let tmp = path.with_extension("json.tmp");
        tokio::fs::write(&tmp, &bytes)
            .await
            .with_context(|| format!("writing embedding index {}", tmp.display()))?;
        tokio::fs::rename(&tmp, path)
            .await
            .with_context(|| format!("installing embedding index {}", path.display()))
    }

    /// The vector for `id`, but only if it was embedded from content with
    /// this exact `hash` — an edited memory misses and re-embeds.
    pub fn vector(&self, id: &str, hash: &str) -> Option<&[f32]> {
        self.entries
            .get(id)
            .filter(|entry| entry.hash == hash)
            .map(|entry| entry.vector.as_slice())
    }

    pub fn insert(&mut self, id: String, hash: String, vector: Vec<f32>) {
        self.entries.insert(id, IndexEntry { hash, vector });
    }

    /// Drop `id`'s vector. Returns whether anything was removed.
    pub fn remove(&mut self, id: &str) -> bool {
        self.entries.remove(id).is_some()
    }

    /// Drop vectors for memories that no longer exist (deleted files).
    /// Returns whether anything was pruned.
    pub fn retain_ids(&mut self, live: &std::collections::HashSet<&str>) -> bool {
        let before = self.entries.len();
        self.entries.retain(|id, _| live.contains(id.as_str()));
        self.entries.len() != before
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn temp_index_path() -> PathBuf {
        std::env::temp_dir()
            .join(format!("agent-embedding-index-{}", uuid::Uuid::new_v4()))
            .join(".index")
            .join("embeddings.json")
    }

    #[test]
    fn cosine_orders_by_angle_and_tolerates_degenerate_input() {
        let a = [1.0f32, 0.0, 0.0];
        let close = [0.9f32, 0.1, 0.0];
        let far = [0.0f32, 1.0, 0.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        assert!(cosine(&a, &close) > cosine(&a, &far));
        assert_eq!(cosine(&a, &far), 0.0);
        // Degenerate inputs score 0 instead of erroring or NaN-ing.
        assert_eq!(cosine(&a, &[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(cosine(&a, &[1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn content_hash_is_stable_and_content_sensitive() {
        assert_eq!(content_hash("abc"), content_hash("abc"));
        assert_ne!(content_hash("abc"), content_hash("abd"));
        assert_eq!(content_hash("abc").len(), 64);
    }

    #[tokio::test]
    async fn index_round_trips_and_misses_on_stale_hash() {
        let path = temp_index_path();
        let mut index = EmbeddingIndex::empty("mock-model");
        index.insert("note".into(), content_hash("v1"), vec![1.0, 2.0]);
        index.save(&path).await.unwrap();

        let loaded = EmbeddingIndex::load(&path, "mock-model").await;
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded.vector("note", &content_hash("v1")),
            Some([1.0f32, 2.0].as_slice())
        );
        // Same id, different content: the stale vector must not be served.
        assert_eq!(loaded.vector("note", &content_hash("v2")), None);
        assert_eq!(loaded.vector("other", &content_hash("v1")), None);
    }

    #[tokio::test]
    async fn model_change_corruption_and_absence_all_load_empty() {
        let path = temp_index_path();
        let mut index = EmbeddingIndex::empty("model-a");
        index.insert("note".into(), content_hash("v1"), vec![1.0]);
        index.save(&path).await.unwrap();

        // A different embedding model cannot reuse model-a's vectors.
        assert!(EmbeddingIndex::load(&path, "model-b").await.is_empty());
        // Corrupt JSON is an empty cache, not an error.
        tokio::fs::write(&path, "{not json").await.unwrap();
        assert!(EmbeddingIndex::load(&path, "model-a").await.is_empty());
        // Missing file likewise.
        let missing = path.with_file_name("nonexistent.json");
        assert!(EmbeddingIndex::load(&missing, "model-a").await.is_empty());
    }

    #[test]
    fn retain_and_remove_drop_vectors() {
        let mut index = EmbeddingIndex::empty("m");
        index.insert("keep".into(), "h1".into(), vec![1.0]);
        index.insert("gone".into(), "h2".into(), vec![2.0]);

        assert!(index.remove("gone"));
        assert!(!index.remove("gone"), "second remove is a no-op");

        index.insert("dead".into(), "h3".into(), vec![3.0]);
        let live: HashSet<&str> = ["keep"].into_iter().collect();
        assert!(index.retain_ids(&live));
        assert!(!index.retain_ids(&live), "nothing left to prune");
        assert_eq!(index.len(), 1);
        assert_eq!(index.vector("keep", "h1"), Some([1.0f32].as_slice()));
    }

    /// `from_registry` construction: absent section is `None` (keyword-only,
    /// zero-cost), a resolvable alias yields a client aimed at
    /// `{base_url}/embeddings` with the entry's api_id, and an entry the
    /// client cannot use (no base_url) fails closed.
    #[test]
    fn from_registry_absent_valid_and_unusable_entry() {
        let base = r#"
default_model: chat
models:
- name: chat
  provider: openai-compatible
  base_url: https://example.test/v1
- name: embed
  provider: openai-compatible
  base_url: https://example.test/v1/
  api_key: test-key
  api_id: text-embedding-tiny
- name: no-url
  provider: claude-code
"#;
        let registry = crate::models::ModelRegistry::from_yaml_str(base).unwrap();
        assert!(EmbeddingClient::from_registry(&registry).unwrap().is_none());

        let yaml = format!("{base}embeddings:\n  model: embed\n");
        let registry = crate::models::ModelRegistry::from_yaml_str(&yaml).unwrap();
        let client = EmbeddingClient::from_registry(&registry)
            .unwrap()
            .expect("embeddings configured");
        assert_eq!(client.model_id(), "text-embedding-tiny");
        assert_eq!(client.url, "https://example.test/v1/embeddings");

        let yaml = format!("{base}embeddings:\n  model: no-url\n");
        let registry = crate::models::ModelRegistry::from_yaml_str(&yaml).unwrap();
        // No unwrap_err: EmbeddingClient deliberately has no Debug impl
        // (it holds an api_key).
        let Err(err) = EmbeddingClient::from_registry(&registry) else {
            panic!("an entry without base_url must fail closed");
        };
        assert!(err.to_string().contains("no base_url"), "{err}");
    }

    #[test]
    fn parses_openai_shape_reordering_by_index() {
        let body = r#"{
            "object": "list",
            "data": [
                { "object": "embedding", "index": 1, "embedding": [0.5, 0.5] },
                { "object": "embedding", "index": 0, "embedding": [1.0, 0.0] }
            ],
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 }
        }"#;
        let vectors = parse_embeddings_response(body, 2).unwrap();
        assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.5, 0.5]]);
    }

    #[test]
    fn response_shape_violations_are_errors() {
        let one = r#"{ "data": [ { "index": 0, "embedding": [1.0] } ] }"#;
        assert!(parse_embeddings_response(one, 2)
            .unwrap_err()
            .to_string()
            .contains("1 vectors for 2 inputs"));

        let oob = r#"{ "data": [ { "index": 5, "embedding": [1.0] } ] }"#;
        assert!(parse_embeddings_response(oob, 1)
            .unwrap_err()
            .to_string()
            .contains("out of range"));

        let dup = r#"{ "data": [ { "index": 0, "embedding": [1.0] }, { "index": 0, "embedding": [2.0] } ] }"#;
        assert!(parse_embeddings_response(dup, 2)
            .unwrap_err()
            .to_string()
            .contains("repeats index"));

        assert!(parse_embeddings_response("not json", 1).is_err());
    }
}
