use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ops::{BitOr, BitOrAssign};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    Temporal,
    Semantic,
    Knowledge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceCapability(u32);

impl SourceCapability {
    pub const NONE: Self = Self(0);
    pub const SESSION_CONTEXT: Self = Self(1 << 0);
    pub const QUERY: Self = Self(1 << 1);
    pub const WORKSPACE: Self = Self(1 << 2);

    pub fn empty() -> Self {
        Self::NONE
    }

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn bits(self) -> u32 {
        self.0
    }
}

impl BitOr for SourceCapability {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for SourceCapability {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceParams {
    pub query: Option<String>,
    pub max_bytes: Option<usize>,
}

impl SourceParams {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: Some(query.into()),
            max_bytes: None,
        }
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceResult {
    pub source: String,
    pub kind: SourceKind,
    pub content: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassiveSource {
    TemporalHistory,
    SessionContext,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassiveHydrationConfig {
    pub sources: Vec<PassiveSource>,
    pub max_bytes: Option<usize>,
}

impl PassiveHydrationConfig {
    pub fn none() -> Self {
        Self {
            sources: Vec::new(),
            max_bytes: None,
        }
    }

    pub fn with_sources(sources: impl Into<Vec<PassiveSource>>) -> Self {
        Self {
            sources: sources.into(),
            max_bytes: None,
        }
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl Default for PassiveHydrationConfig {
    fn default() -> Self {
        Self::none()
    }
}

#[async_trait]
pub trait HydrationSource: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> SourceKind;
    fn capabilities(&self) -> SourceCapability;
    async fn retrieve(&self, params: SourceParams) -> Result<SourceResult>;
}

/// Stable identifier assigned by a sink to a stored item (the memory file
/// backend uses the slug). Opaque to the runtime; only the assigning sink
/// can interpret it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SinkId(pub String);

/// Provenance the RUNTIME attaches to every sink write (docs/MEMORY.md):
/// which run wrote it, through which effect, and when. Universal across
/// sinks, never the program's job to supply.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Provenance {
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

/// A sink write: the payload is sink-defined JSON the sink validates
/// against its own schema, plus runtime-attached provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkItem {
    pub payload: Value,
    #[serde(default)]
    pub provenance: Provenance,
}

/// Per-sink write policy (docs/MEMORY.md settled question 1): the policy
/// hook lives at the effect/dispatch layer, not in the model's prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkWritePolicy {
    /// Writes execute immediately; the trace is the audit.
    Free,
    /// Writes require harness-level approval (e.g. the self-prompt sink).
    RequireApproval,
}

/// Write side of a backend (docs/MEMORY.md, t-1165). Deliberately a
/// separate trait from [`HydrationSource`] — std::io `Read`/`Write`
/// precedent — so writability is a compile-time fact; backends that
/// persist implement both and register via
/// [`SourceRegistry::register_backend`].
#[async_trait]
pub trait HydrationSink: Send + Sync {
    fn name(&self) -> &str;
    /// Sinks share the source kind taxonomy.
    fn kind(&self) -> SourceKind;
    fn write_policy(&self) -> SinkWritePolicy {
        SinkWritePolicy::Free
    }
    async fn store(&self, item: SinkItem) -> Result<SinkId>;
    async fn update(&self, id: &SinkId, item: SinkItem) -> Result<()>;
    async fn delete(&self, id: &SinkId) -> Result<()>;
}

#[derive(Clone, Default)]
pub struct SourceRegistry {
    sources: Vec<Arc<dyn HydrationSource>>,
    sinks: Vec<Arc<dyn HydrationSink>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(mut self, source: T) -> Self
    where
        T: HydrationSource + 'static,
    {
        self.sources.push(Arc::new(source));
        self
    }

    pub fn register_arc(mut self, source: Arc<dyn HydrationSource>) -> Self {
        self.sources.push(source);
        self
    }

    /// Register a write-only sink.
    pub fn register_sink<T>(mut self, sink: T) -> Self
    where
        T: HydrationSink + 'static,
    {
        self.sinks.push(Arc::new(sink));
        self
    }

    /// Register a backend that is both source and sink: one object, one
    /// `Arc`, coerced into both lists.
    pub fn register_backend<T>(mut self, backend: T) -> Self
    where
        T: HydrationSource + HydrationSink + 'static,
    {
        let backend = Arc::new(backend);
        self.sources.push(backend.clone());
        self.sinks.push(backend);
        self
    }

    pub fn sources(&self) -> &[Arc<dyn HydrationSource>] {
        &self.sources
    }

    pub fn sinks(&self) -> &[Arc<dyn HydrationSink>] {
        &self.sinks
    }

    /// The sink registered under `name`, if any.
    pub fn sink(&self, name: &str) -> Option<Arc<dyn HydrationSink>> {
        self.sinks.iter().find(|sink| sink.name() == name).cloned()
    }

    /// All sinks of a kind (e.g. the memory sinks the `remember` tool
    /// targets).
    pub fn sinks_of_kind(&self, kind: SourceKind) -> Vec<Arc<dyn HydrationSink>> {
        self.sinks
            .iter()
            .filter(|sink| sink.kind() == kind)
            .cloned()
            .collect()
    }

    pub async fn retrieve_all(&self, params: SourceParams) -> Result<Vec<SourceResult>> {
        self.retrieve_matching(params, |_| true).await
    }

    pub async fn retrieve_session_context(
        &self,
        params: SourceParams,
    ) -> Result<Vec<SourceResult>> {
        self.retrieve_matching(params, |source| {
            source
                .capabilities()
                .contains(SourceCapability::SESSION_CONTEXT)
        })
        .await
    }

    pub async fn retrieve_query(&self, params: SourceParams) -> Result<Vec<SourceResult>> {
        self.retrieve_matching(params, |source| {
            source.capabilities().contains(SourceCapability::QUERY)
        })
        .await
    }

    /// QUERY-capable sources, optionally narrowed to one kind — the
    /// Retrieve effect's dispatch (docs/MEMORY.md).
    pub async fn retrieve_query_of_kind(
        &self,
        params: SourceParams,
        kind: Option<SourceKind>,
    ) -> Result<Vec<SourceResult>> {
        self.retrieve_matching(params, |source| {
            source.capabilities().contains(SourceCapability::QUERY)
                && kind.is_none_or(|kind| source.kind() == kind)
        })
        .await
    }

    async fn retrieve_matching<F>(
        &self,
        params: SourceParams,
        mut keep: F,
    ) -> Result<Vec<SourceResult>>
    where
        F: FnMut(&dyn HydrationSource) -> bool,
    {
        let mut results = Vec::new();
        for source in &self.sources {
            if keep(source.as_ref()) {
                results.push(source.retrieve(params.clone()).await?);
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticSource;

    #[async_trait]
    impl HydrationSource for StaticSource {
        fn name(&self) -> &str {
            "static"
        }

        fn kind(&self) -> SourceKind {
            SourceKind::Knowledge
        }

        fn capabilities(&self) -> SourceCapability {
            SourceCapability::SESSION_CONTEXT | SourceCapability::QUERY
        }

        async fn retrieve(&self, params: SourceParams) -> Result<SourceResult> {
            Ok(SourceResult {
                source: self.name().into(),
                kind: self.kind(),
                content: params.query.unwrap_or_else(|| "default".into()),
                metadata: serde_json::json!({ "capabilities": self.capabilities().bits() }),
            })
        }
    }

    #[tokio::test]
    async fn registry_retrieves_from_registered_sources() -> Result<()> {
        let registry = SourceRegistry::new().register(StaticSource);
        let results = registry.retrieve_all(SourceParams::new("hello")).await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "static");
        assert_eq!(results[0].kind, SourceKind::Knowledge);
        assert_eq!(results[0].content, "hello");
        Ok(())
    }

    #[test]
    fn capabilities_are_composable_flags() {
        let caps = SourceCapability::SESSION_CONTEXT | SourceCapability::WORKSPACE;
        assert!(caps.contains(SourceCapability::SESSION_CONTEXT));
        assert!(caps.contains(SourceCapability::WORKSPACE));
        assert!(!caps.contains(SourceCapability::QUERY));
    }
}
