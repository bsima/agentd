use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ops::{BitOr, BitOrAssign};
use std::sync::Arc;

pub const TEMPORAL_PREFIX: &str = "temporal:";
pub const SEMANTIC_PREFIX: &str = "semantic:";
pub const SESSION_STATE_KEY: &str = "session:state";

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

#[derive(Clone, Default)]
pub struct SourceRegistry {
    sources: Vec<Arc<dyn HydrationSource>>,
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

    pub fn sources(&self) -> &[Arc<dyn HydrationSource>] {
        &self.sources
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
