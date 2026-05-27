use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub provider: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_id: Option<String>,
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default = "default_context")]
    pub context: usize,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub vision: bool,
    pub display: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelRegistry {
    pub default_model: String,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub alias: String,
    pub provider: Option<String>,
    pub api_id: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}

impl ModelRegistry {
    pub fn from_yaml_str(input: &str) -> Result<Self> {
        serde_yaml::from_str(input).context("parsing models.yaml")
    }

    pub async fn load_default() -> Result<Self> {
        Self::load(default_models_path()?).await
    }

    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("reading model registry {}", path.display()))?;
        Self::from_yaml_str(&content)
            .with_context(|| format!("loading model registry {}", path.display()))
    }

    pub fn resolve(&self, requested: Option<&str>) -> Result<ResolvedModel> {
        let alias = requested.unwrap_or(&self.default_model);
        match self.models.iter().find(|entry| entry.name == alias) {
            Some(entry) => Ok(ResolvedModel {
                alias: entry.name.clone(),
                provider: Some(entry.provider.clone()),
                api_id: entry.api_id.clone().unwrap_or_else(|| entry.name.clone()),
                base_url: entry.base_url.clone(),
                api_key: expand_api_key(entry.api_key.as_deref())?,
            }),
            None => Ok(ResolvedModel {
                alias: alias.to_string(),
                provider: None,
                api_id: alias.to_string(),
                base_url: None,
                api_key: None,
            }),
        }
    }
}

fn default_context() -> usize {
    200_000
}

fn expand_api_key(value: Option<&str>) -> Result<Option<String>> {
    match value {
        Some(raw) if raw.starts_with('$') => {
            let name = raw.trim_start_matches('$');
            if name.is_empty() {
                return Err(anyhow!(
                    "empty environment variable reference in model api_key"
                ));
            }
            std::env::var(name)
                .map(Some)
                .with_context(|| format!("resolving model api_key env var {name}"))
        }
        Some(raw) => Ok(Some(raw.to_string())),
        None => Ok(None),
    }
}

fn default_models_path() -> Result<PathBuf> {
    let config_home = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".config")))
        .ok_or_else(|| anyhow!("could not determine config directory"))?;
    Ok(config_home.join("agent/models.yaml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS: &str = r#"
default_model: parasail/qwen3-235b
models:
- name: parasail/qwen3-235b
  provider: openai-compatible
  base_url: https://api.parasail.io/v1
  api_key: literal-key
  api_id: parasail-qwen3-235b-a22b-instruct-2507
  input: 0.5
  output: 2.0
  context: 131072
  thinking: false
  vision: false
  display: Qwen3 235B (parasail)
- name: direct-name
  provider: openai-compatible
  input: 1.0
  output: 1.0
"#;

    #[test]
    fn parses_and_resolves_alias_to_api_id() -> Result<()> {
        let registry = ModelRegistry::from_yaml_str(MODELS)?;
        let resolved = registry.resolve(Some("parasail/qwen3-235b"))?;

        assert_eq!(resolved.alias, "parasail/qwen3-235b");
        assert_eq!(resolved.api_id, "parasail-qwen3-235b-a22b-instruct-2507");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://api.parasail.io/v1")
        );
        assert_eq!(resolved.api_key.as_deref(), Some("literal-key"));
        Ok(())
    }

    #[test]
    fn default_model_and_unknown_alias_are_supported() -> Result<()> {
        let registry = ModelRegistry::from_yaml_str(MODELS)?;
        assert_eq!(
            registry.resolve(None)?.api_id,
            "parasail-qwen3-235b-a22b-instruct-2507"
        );
        assert_eq!(
            registry.resolve(Some("unknown/model"))?.api_id,
            "unknown/model"
        );
        assert_eq!(registry.resolve(Some("direct-name"))?.api_id, "direct-name");
        Ok(())
    }
}
