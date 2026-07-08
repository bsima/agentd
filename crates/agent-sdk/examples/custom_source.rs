//! A complete out-of-tree hydration provider (docs/PROVIDERS.md): a
//! read-only `WorkspaceSource` that answers Semantic queries by substring
//! search over text files in a directory, wired into an agent and
//! exercised through a `recall` round-trip.
//!
//! Everything here uses only the SDK's public surface — the provider type
//! lives in this file, not in agent-core — which is the point: the
//! hydration traits are implementable outside the runtime crate.
//!
//! Uses a scripted provider so it runs without credentials:
//!
//! ```sh
//! cargo run -p agent-sdk --example custom_source
//! ```
//!
//! The same round-trip runs in CI via `tests/custom_source_example.rs`,
//! which includes this file and calls [`run_demo`].
//!
//! The search is deliberately naive (case-insensitive substring over
//! lines): the example demonstrates the trait mechanics, not retrieval
//! quality. An embedding-backed version would keep exactly the same trait
//! surface and differ only inside `retrieve()` — embed the corpus with the
//! shipped `agent_core::embedding` infrastructure (`Embedder` for the
//! vectors, `EmbeddingIndex` for the `(id, content-hash)`-keyed sidecar
//! cache, `cosine` for similarity) and rank by blended score the way the
//! in-tree `MemorySource` does. Retrieval internals are backend-private:
//! nothing about the trait, the `Retrieve` effect, or replay changes.

use agent_sdk::testing::ScriptedProvider;
use agent_sdk::{
    Agent, HydrationSource, Runner, SourceCapability, SourceKind, SourceParams, SourceResult,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

/// Read-only workspace search: every `.md`/`.txt` file directly under
/// `root`, matched line-by-line against the query.
struct WorkspaceSource {
    root: PathBuf,
    /// Default result budget; callers override per-retrieval via
    /// `SourceParams::max_bytes`.
    max_bytes: usize,
}

impl WorkspaceSource {
    fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_bytes: 8 * 1024,
        }
    }
}

#[async_trait]
impl HydrationSource for WorkspaceSource {
    /// Stable, human-readable id: lands in `SourceResult.source`, trace
    /// events, and PromptIR provenance.
    fn name(&self) -> &str {
        "workspace"
    }

    /// Semantic is the kind the built-in `recall` tool queries.
    fn kind(&self) -> SourceKind {
        SourceKind::Semantic
    }

    /// QUERY = reachable from `Retrieve` effects (and so from `recall`).
    fn capabilities(&self) -> SourceCapability {
        SourceCapability::QUERY
    }

    /// One `SourceResult` per retrieval, matches aggregated into
    /// `content`. Empty result sets are a normal outcome, not an error —
    /// reserve `Err` for genuine failures (here: an unreadable root).
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
        let query = params.query.unwrap_or_default().to_lowercase();
        let words: Vec<&str> = query.split_whitespace().collect();
        let max_bytes = params.max_bytes.unwrap_or(self.max_bytes);

        let mut entries = tokio::fs::read_dir(&self.root)
            .await
            .with_context(|| format!("reading workspace root {}", self.root.display()))?;
        let mut paths = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let text_file = path
                .extension()
                .is_some_and(|ext| ext == "md" || ext == "txt");
            if entry.file_type().await?.is_file() && text_file {
                paths.push(path);
            }
        }
        // Deterministic scan order. Not required by the runtime (Retrieve
        // results are recorded; replay serves the recording), but it keeps
        // fresh runs reproducible and testable.
        paths.sort();

        let mut hits = Vec::new();
        let mut matched = 0usize;
        let mut budget = max_bytes;
        for path in &paths {
            let file = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            let text = tokio::fs::read_to_string(path)
                .await
                .with_context(|| format!("reading workspace file {}", path.display()))?;
            for (index, line) in text.lines().enumerate() {
                let haystack = line.to_lowercase();
                if words.iter().any(|word| haystack.contains(word)) {
                    matched += 1;
                    let hit = format!("{file}:{}: {}", index + 1, line.trim());
                    if hit.len() > budget {
                        continue;
                    }
                    budget -= hit.len();
                    hits.push(hit);
                }
            }
        }

        Ok(SourceResult {
            source: self.name().into(),
            kind: self.kind(),
            content: if hits.is_empty() {
                "no workspace matches".into()
            } else {
                hits.join("\n")
            },
            // Provider-defined; a numeric `score` field, when present, is
            // surfaced as retrieval provenance in the PromptIR.
            metadata: json!({ "files_scanned": paths.len(), "matches": matched }),
        })
    }
}

/// The round-trip, callable from both `main` and the CI test: build a tiny
/// workspace, register the source, let a scripted model call `recall`, and
/// return `(final_text, retrieved_content)` — where `retrieved_content` is
/// what the `Retrieve` effect actually recorded and fed back to the model
/// as the tool result.
pub async fn run_demo() -> Result<(String, String)> {
    // A throwaway workspace with one relevant fact and one distractor.
    let workspace =
        std::env::temp_dir().join(format!("agent-sdk-workspace-{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&workspace).await?;
    tokio::fs::write(
        workspace.join("deploy.md"),
        "# Ops notes\nThe deploy checklist: run migrations before restarting the API.\n",
    )
    .await?;
    tokio::fs::write(
        workspace.join("lunch.txt"),
        "Team lunch is Thursday at the taqueria.\n",
    )
    .await?;

    // Scripted model (credential-free): first response calls `recall`
    // — which the loop compiles onto a `Retrieve` effect that our source
    // answers — second response ends the turn.
    let provider = ScriptedProvider::new()
        .tool_call("recall", json!({ "query": "deploy checklist" }))
        .text("Deploy checklist: run migrations before restarting the API.");

    let agent = Agent::builder("mock-model")
        .name("workspace-bot")
        .instructions("Answer from the workspace; use recall to search it.")
        .hydration_source(WorkspaceSource::new(&workspace))
        .provider(Arc::new(provider))
        .trace_dir(std::env::temp_dir().join("agent-sdk-examples"))
        .build()?;

    let result = Runner::run(&agent, "What's on the deploy checklist?").await?;

    // The RetrieveResult trace event carries the full result set the
    // effect bound — the same JSON the loop rendered into the recall tool
    // message the model read before answering.
    let events = agent_core::TraceLogger::read_events(&result.trace_path).await?;
    let retrieved = events
        .iter()
        .find_map(|event| match event {
            agent_core::Event::RetrieveResult { results, .. } => Some(results.to_string()),
            _ => None,
        })
        .context("no RetrieveResult event in the trace")?;

    anyhow::ensure!(
        retrieved.contains("run migrations before restarting the API"),
        "workspace content missing from retrieval: {retrieved}"
    );
    anyhow::ensure!(
        retrieved.contains("deploy.md"),
        "retrieval lost file provenance: {retrieved}"
    );
    Ok((result.text, retrieved))
}

#[tokio::main]
async fn main() -> Result<()> {
    let (text, retrieved) = run_demo().await?;
    println!("retrieved: {retrieved}");
    println!("answer:    {text}");
    Ok(())
}
