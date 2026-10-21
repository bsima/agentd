use crate::cost::{Pricing, PricingTable};
use crate::reasoning::Effort;
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Optional cost-accounting rates for a model entry, in USD per million
/// tokens (t-1334). Absent pricing means Infer usage is recorded but cost
/// is omitted — never guessed, never zero.
#[derive(Debug, Clone, Deserialize)]
pub struct PricingEntry {
    pub input_per_mtok_usd: f64,
    pub output_per_mtok_usd: f64,
    pub fast_input_per_mtok_usd: Option<f64>,
    pub fast_output_per_mtok_usd: Option<f64>,
}

impl PricingEntry {
    fn fast_pricing(&self, model: &str) -> Result<Option<Pricing>> {
        match (self.fast_input_per_mtok_usd, self.fast_output_per_mtok_usd) {
            (Some(input), Some(output)) => Pricing::from_usd_per_mtok(input, output)
                .with_context(|| format!("invalid fast pricing for model {model:?}"))
                .map(Some),
            (None, None) => Ok(None),
            _ => Err(anyhow!(
                "model {model:?}: both fast pricing rates are required"
            )),
        }
    }
    fn to_pricing(&self, model: &str) -> Result<Pricing> {
        Pricing::from_usd_per_mtok(self.input_per_mtok_usd, self.output_per_mtok_usd)
            .with_context(|| format!("invalid pricing for model {model:?}"))
    }
}

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
    /// Optional per-response output token cap for providers that need one
    /// (the anthropic provider defaults to 8192 when unset).
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub thinking: bool,
    pub effort: Option<Effort>,
    #[serde(default)]
    pub efforts: Vec<Effort>,
    #[serde(default)]
    pub fast: bool,
    #[serde(default)]
    pub vision: bool,
    pub display: Option<String>,
    /// Cost-accounting rates (USD per Mtok); serde-optional so existing
    /// models.yaml files load unchanged. The legacy flat `input`/`output`
    /// fields above are NOT read for cost accounting (their zero default
    /// cannot distinguish "free" from "unknown").
    #[serde(default)]
    pub pricing: Option<PricingEntry>,
}

/// Optional `embeddings` section (t-1340): selects, by registry alias, the
/// model the memory backend embeds with. The alias must name an entry in
/// `models`, which supplies the base_url/api_key/api_id exactly like a chat
/// model — one resolution path, one fail-closed behavior.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsEntry {
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelRegistry {
    #[serde(default)]
    pub default_model: Option<String>,
    pub models: Vec<ModelEntry>,
    /// Serde-optional so existing models.yaml files load unchanged; absent
    /// means keyword-only memory retrieval.
    #[serde(default)]
    pub embeddings: Option<EmbeddingsEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub alias: String,
    pub provider: Option<String>,
    pub api_id: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub context: usize,
    pub max_tokens: Option<u32>,
    pub effort: Option<Effort>,
    pub efforts: Vec<Effort>,
    pub fast: bool,
    /// Registry pricing for this model, already validated/converted to the
    /// exact integer representation. None = no pricing configured.
    pub pricing: Option<Pricing>,
}

impl ModelRegistry {
    pub fn from_yaml_str(input: &str) -> Result<Self> {
        let registry: Self = serde_yaml::from_str(input).context("parsing models.yaml")?;
        for entry in &registry.models {
            if entry
                .effort
                .is_some_and(|effort| !entry.efforts.contains(&effort))
            {
                return Err(anyhow!(
                    "model {:?}: default effort must be in efforts",
                    entry.name
                ));
            }
            if (!entry.thinking && (!entry.efforts.is_empty() || entry.effort.is_some()))
                || (entry.thinking && entry.efforts.is_empty() && entry.effort.is_some())
            {
                return Err(anyhow!(
                    "model {:?}: effort requires thinking and an allowed efforts list",
                    entry.name
                ));
            }
        }
        Ok(registry)
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
        let alias = requested.or(self.default_model.as_deref()).ok_or_else(|| {
            anyhow!("models.yaml has no default_model; select a model with --model")
        })?;
        let Some(entry) = self.models.iter().find(|entry| entry.name == alias) else {
            let mut known = self
                .models
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>();
            known.sort_unstable();
            return Err(anyhow!(
                "unknown model alias {alias:?}; known model names: {}",
                known.join(", ")
            ));
        };

        Ok(ResolvedModel {
            alias: entry.name.clone(),
            provider: Some(entry.provider.clone()),
            api_id: entry.api_id.clone().unwrap_or_else(|| entry.name.clone()),
            base_url: entry.base_url.clone(),
            api_key: expand_api_key(entry.api_key.as_deref())?,
            context: entry.context,
            max_tokens: entry.max_tokens,
            effort: entry.effort,
            efforts: entry.efforts.clone(),
            fast: entry.fast,
            pricing: entry
                .pricing
                .as_ref()
                .map(|pricing| pricing.to_pricing(&entry.name))
                .transpose()?,
        })
    }

    /// Resolve the optional `embeddings` section through the same alias
    /// resolution as chat models: `Ok(None)` when absent (keyword-only
    /// retrieval), and a fail-closed error when the section names an
    /// unknown alias (26c02ea) — a typo'd embeddings model must surface at
    /// config load, not silently degrade semantic retrieval.
    pub fn resolve_embeddings(&self) -> Result<Option<ResolvedModel>> {
        self.embeddings
            .as_ref()
            .map(|entry| {
                self.resolve(Some(&entry.model))
                    .context("resolving embeddings model")
            })
            .transpose()
    }

    /// Pricing lookup table over the whole registry, keyed by both the
    /// registry alias (`name`) and the provider `api_id` — trace events
    /// record the model id the Infer actually ran with, which is the api id
    /// for registry-resolved models and may be either for dynamic
    /// (tool-supplied) models. First entry wins on key collisions, matching
    /// the resolve order of the file.
    pub fn pricing_table(&self) -> Result<PricingTable> {
        let mut table = PricingTable::default();
        for entry in &self.models {
            let Some(pricing) = entry.pricing.as_ref() else {
                continue;
            };
            let fast = pricing.fast_pricing(&entry.name)?;
            let pricing = pricing.to_pricing(&entry.name)?;
            table.insert(entry.name.clone(), pricing);
            if let Some(fast) = fast {
                table.insert_fast(entry.name.clone(), fast);
            }
            if let Some(api_id) = entry.api_id.as_deref() {
                table.insert(api_id.to_owned(), pricing);
                if let Some(fast) = fast {
                    table.insert_fast(api_id.to_owned(), fast);
                }
            }
        }
        Ok(table)
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
  max_tokens: 32000
  thinking: false
  vision: false
  display: Qwen3 235B (parasail)
  pricing:
    input_per_mtok_usd: 0.5
    output_per_mtok_usd: 2.0
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
        assert_eq!(resolved.max_tokens, Some(32000));
        // Entries without the field resolve to None (provider default).
        assert_eq!(registry.resolve(Some("direct-name"))?.max_tokens, None);
        Ok(())
    }

    #[test]
    fn default_model_is_optional_when_model_is_explicitly_requested() -> Result<()> {
        let registry = ModelRegistry::from_yaml_str(
            "models:\n- name: explicit/model\n  provider: openai-compatible\n",
        )?;
        assert_eq!(
            registry.resolve(Some("explicit/model"))?.api_id,
            "explicit/model"
        );
        let err = registry.resolve(None).unwrap_err().to_string();
        assert!(err.contains("no default_model"), "got: {err}");
        assert!(err.contains("--model"), "got: {err}");
        Ok(())
    }

    #[test]
    fn default_model_known_alias_and_unknown_alias_error() -> Result<()> {
        let registry = ModelRegistry::from_yaml_str(MODELS)?;
        assert_eq!(
            registry.resolve(None)?.api_id,
            "parasail-qwen3-235b-a22b-instruct-2507"
        );
        let err = registry
            .resolve(Some("unknown/model"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown/model"), "got: {err}");
        assert!(err.contains("direct-name"), "got: {err}");
        assert!(err.contains("parasail/qwen3-235b"), "got: {err}");
        assert_eq!(registry.resolve(Some("direct-name"))?.api_id, "direct-name");
        Ok(())
    }

    /// Pricing is optional per entry: present entries resolve to the exact
    /// integer rates, absent entries resolve to None (usage recorded, cost
    /// omitted — the legacy flat input/output fields are deliberately NOT
    /// read as pricing).
    #[test]
    fn pricing_is_optional_and_resolves_to_integer_rates() -> Result<()> {
        let registry = ModelRegistry::from_yaml_str(MODELS)?;
        let priced = registry.resolve(Some("parasail/qwen3-235b"))?;
        assert_eq!(
            priced.pricing,
            Some(Pricing {
                input_micro_usd_per_mtok: 500_000,
                output_micro_usd_per_mtok: 2_000_000,
            })
        );
        // `input: 1.0 / output: 1.0` (legacy fields) must not leak into
        // pricing: absent means absent.
        assert_eq!(registry.resolve(Some("direct-name"))?.pricing, None);
        Ok(())
    }

    #[test]
    fn pricing_table_is_keyed_by_alias_and_api_id() -> Result<()> {
        let registry = ModelRegistry::from_yaml_str(MODELS)?;
        let table = registry.pricing_table()?;
        let expected = Pricing {
            input_micro_usd_per_mtok: 500_000,
            output_micro_usd_per_mtok: 2_000_000,
        };
        assert_eq!(table.get("parasail/qwen3-235b"), Some(&expected));
        assert_eq!(
            table.get("parasail-qwen3-235b-a22b-instruct-2507"),
            Some(&expected)
        );
        assert_eq!(table.get("direct-name"), None);
        Ok(())
    }

    /// The `embeddings` section (t-1340) is serde-optional (existing
    /// configs load unchanged), resolves through the same alias path as
    /// chat models, and fails closed on an unknown alias or a malformed
    /// section.
    #[test]
    fn embeddings_section_absent_present_and_invalid() -> Result<()> {
        // Absent: keyword-only retrieval, not an error.
        let registry = ModelRegistry::from_yaml_str(MODELS)?;
        assert!(registry.resolve_embeddings()?.is_none());

        // Present, valid alias: resolves like any chat model.
        let yaml = format!("{MODELS}\nembeddings:\n  model: parasail/qwen3-235b\n");
        let resolved = ModelRegistry::from_yaml_str(&yaml)?
            .resolve_embeddings()?
            .expect("embeddings configured");
        assert_eq!(resolved.api_id, "parasail-qwen3-235b-a22b-instruct-2507");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://api.parasail.io/v1")
        );

        // Unknown alias: fails closed with the alias in the error.
        let yaml = format!("{MODELS}\nembeddings:\n  model: nope/missing\n");
        let err = ModelRegistry::from_yaml_str(&yaml)?
            .resolve_embeddings()
            .unwrap_err();
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("resolving embeddings model"),
            "{rendered}"
        );
        assert!(rendered.contains("nope/missing"), "{rendered}");

        // Malformed section (no `model` field): fails at parse.
        let yaml = format!("{MODELS}\nembeddings: {{}}\n");
        assert!(ModelRegistry::from_yaml_str(&yaml).is_err());
        Ok(())
    }

    #[test]
    fn effort_validation() {
        let base = "models:\n- name: m\n  provider: anthropic\n  thinking: true\n";
        assert!(ModelRegistry::from_yaml_str(&format!(
            "{base}  effort: turbo\n  efforts: [turbo]\n"
        ))
        .is_err());
        assert!(
            ModelRegistry::from_yaml_str(&format!("{base}  effort: high\n  efforts: [low]\n"))
                .is_err()
        );
        let registry = ModelRegistry::from_yaml_str(&format!(
            "{base}  effort: high\n  efforts: [low, high]\n  fast: true\n"
        ))
        .unwrap();
        let model = registry.resolve(Some("m")).unwrap();
        assert_eq!(model.effort, Some(Effort::High));
        assert!(model.fast);
    }

    #[test]
    fn negative_pricing_fails_closed() {
        let yaml = r#"
default_model: m
models:
- name: m
  provider: openai-compatible
  pricing:
    input_per_mtok_usd: -1.0
    output_per_mtok_usd: 2.0
"#;
        let registry = ModelRegistry::from_yaml_str(yaml).expect("parses");
        let err = registry.resolve(Some("m")).unwrap_err().to_string();
        assert!(err.contains("invalid pricing"), "got: {err}");
        assert!(registry.pricing_table().is_err());
    }
}
