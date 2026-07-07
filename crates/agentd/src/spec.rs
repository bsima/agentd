//! The canonical session spec: `<name>/agent.md` (docs/SUPERVISOR.md,
//! t-1105 design constraint).
//!
//! The on-disk file is the single source of truth for launch config. It uses
//! the same markdown-prompt conventions the `agent` binary parses for
//! one-shot prompt files: YAML frontmatter delimited by `---` lines carrying
//! `model` / `provider` / `system_prompt` / `max_iterations`, followed by a
//! markdown body. `agentd start`/`resume` read it fresh on every launch and
//! translate it to `agent` flags; `agentd set-*` edits it in place; hand
//! edits are equally valid — same file, same effect. Nothing regenerates it
//! from a shadow store.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const SPEC_FILE: &str = "agent.md";

/// The typed view of the frontmatter fields the supervisor understands.
/// Unknown keys are preserved on edit (see [`Spec::set`]) but ignored here,
/// so users can annotate their specs freely.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct SpecConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    /// The agent's frontmatter name for the turn ceiling; `max_turns` is
    /// accepted as an alias when reading (writes use `max_iterations`).
    #[serde(alias = "max_turns")]
    pub max_iterations: Option<usize>,
    /// Literal text, or a path resolved relative to the session directory
    /// (the same convention as the agent's prompt-file frontmatter).
    pub system_prompt: Option<String>,
    /// Extra `agent` CLI arguments appended verbatim at launch (e.g.
    /// `["--memory-dir", "memory"]`). Relative paths are the child's cwd.
    #[serde(default)]
    pub args: Vec<String>,
}

/// A parsed spec file: frontmatter as an order-preserving YAML mapping (so
/// edits keep unknown keys and their order) plus the markdown body.
#[derive(Debug, Clone, Default)]
pub struct Spec {
    mapping: serde_yaml::Mapping,
    body: String,
}

impl Spec {
    /// Parse spec text with the agent's markdown-prompt conventions: an
    /// opening `---` line starts YAML frontmatter, closed by the next `---`
    /// line; anything else is body. Unclosed frontmatter is an error.
    pub fn parse(content: &str) -> Result<Self> {
        let mut lines = content.lines();
        let Some(first) = lines.next() else {
            return Ok(Self::default());
        };
        if first.trim() != "---" {
            return Ok(Self {
                mapping: serde_yaml::Mapping::new(),
                body: content.to_string(),
            });
        }
        let mut yaml_lines = Vec::new();
        let mut found_end = false;
        for line in lines.by_ref() {
            if line.trim() == "---" {
                found_end = true;
                break;
            }
            yaml_lines.push(line);
        }
        if !found_end {
            bail!("spec has unclosed YAML frontmatter (missing the closing `---` line)");
        }
        let body = lines.collect::<Vec<_>>().join("\n");
        let yaml = yaml_lines.join("\n");
        let mapping = if yaml.trim().is_empty() {
            serde_yaml::Mapping::new()
        } else {
            serde_yaml::from_str::<serde_yaml::Mapping>(&yaml)
                .context("parsing spec frontmatter (expected a YAML mapping)")?
        };
        Ok(Self { mapping, body })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading spec {}", path.display()))?;
        Self::parse(&content).with_context(|| format!("parsing spec {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.render())
            .with_context(|| format!("writing spec {}", path.display()))
    }

    /// Render back to markdown. Frontmatter is always emitted (even when
    /// empty) so the file self-documents where config lives.
    pub fn render(&self) -> String {
        let yaml = if self.mapping.is_empty() {
            String::new()
        } else {
            serde_yaml::to_string(&serde_yaml::Value::Mapping(self.mapping.clone()))
                .expect("YAML mapping serializes")
        };
        let mut out = format!("---\n{yaml}---\n");
        if !self.body.is_empty() {
            out.push_str(&self.body);
            if !self.body.ends_with('\n') {
                out.push('\n');
            }
        }
        out
    }

    /// Set (or overwrite) one frontmatter key, preserving every other key
    /// and the body byte-for-byte modulo YAML re-serialization.
    pub fn set(&mut self, key: &str, value: serde_yaml::Value) {
        self.mapping
            .insert(serde_yaml::Value::String(key.to_string()), value);
    }

    pub fn body(&self) -> &str {
        &self.body
    }

    /// The typed config view of the frontmatter.
    pub fn config(&self) -> Result<SpecConfig> {
        serde_yaml::from_value(serde_yaml::Value::Mapping(self.mapping.clone()))
            .context("spec frontmatter does not match the expected fields")
    }
}

/// Resolve the effective system prompt for a launch: the frontmatter
/// `system_prompt` (literal text, or a path read relative to the session
/// directory — the agent's own convention), falling back to a non-empty
/// spec body. The markdown body of `agent.md` *describes the agent*, which
/// is exactly what a session's system prompt is.
pub fn resolve_system_prompt(session_dir: &Path, spec: &Spec) -> Result<Option<String>> {
    let config = spec.config()?;
    if let Some(value) = config.system_prompt {
        if !looks_like_path(&value) {
            return Ok(Some(value));
        }
        let path = PathBuf::from(&value);
        let full = if path.is_absolute() {
            path
        } else {
            session_dir.join(path)
        };
        if full.is_file() {
            let text = std::fs::read_to_string(&full)
                .with_context(|| format!("reading system prompt {}", full.display()))?;
            return Ok(Some(text));
        }
        return Ok(Some(value));
    }
    let body = spec.body().trim();
    if body.is_empty() {
        Ok(None)
    } else {
        Ok(Some(body.to_string()))
    }
}

/// Mirror of the agent's `frontmatter::looks_like_path`.
fn looks_like_path(value: &str) -> bool {
    value.contains('/')
        || value.ends_with(".md")
        || value.ends_with(".markdown")
        || value.ends_with(".txt")
}

/// Frontmatter keys `agentd set-*` may edit, with their value parsers.
pub fn set_value_for(key: &str, raw: &str) -> Result<serde_yaml::Value> {
    match key {
        "model" | "provider" | "system_prompt" => Ok(serde_yaml::Value::String(raw.to_string())),
        "max_iterations" => {
            let n: u64 = raw
                .parse()
                .map_err(|_| anyhow!("max-turns must be a positive integer, got {raw:?}"))?;
            Ok(serde_yaml::Value::Number(n.into()))
        }
        other => bail!("unsupported spec key {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter_and_body() -> Result<()> {
        let spec = Spec::parse("---\nmodel: sonnet\nmax_iterations: 3\n---\nBe helpful.")?;
        let config = spec.config()?;
        assert_eq!(config.model.as_deref(), Some("sonnet"));
        assert_eq!(config.max_iterations, Some(3));
        assert_eq!(spec.body(), "Be helpful.");
        Ok(())
    }

    #[test]
    fn accepts_max_turns_alias() -> Result<()> {
        let spec = Spec::parse("---\nmax_turns: 7\n---\n")?;
        assert_eq!(spec.config()?.max_iterations, Some(7));
        Ok(())
    }

    #[test]
    fn edit_preserves_unknown_keys_and_body() -> Result<()> {
        let mut spec = Spec::parse(
            "---\nmodel: old-model\nowner: ben # keep me\ntags:\n  - prod\n---\nSystem prompt body.\n",
        )?;
        spec.set("model", set_value_for("model", "new-model")?);
        let rendered = spec.render();
        let reparsed = Spec::parse(&rendered)?;
        assert_eq!(reparsed.config()?.model.as_deref(), Some("new-model"));
        assert!(
            rendered.contains("owner: ben"),
            "unknown key kept: {rendered}"
        );
        assert!(rendered.contains("- prod"), "nested value kept: {rendered}");
        assert_eq!(reparsed.body().trim(), "System prompt body.");
        Ok(())
    }

    #[test]
    fn empty_and_plain_markdown_specs_parse() -> Result<()> {
        assert!(Spec::parse("")?.config()?.model.is_none());
        let plain = Spec::parse("Just a body.")?;
        assert_eq!(plain.body(), "Just a body.");
        Ok(())
    }

    #[test]
    fn unclosed_frontmatter_is_an_error() {
        let err = Spec::parse("---\nmodel: x\nbody").unwrap_err();
        assert!(err.to_string().contains("unclosed"));
    }

    #[test]
    fn body_falls_back_as_system_prompt() -> Result<()> {
        let dir = std::env::temp_dir();
        let spec = Spec::parse("---\nmodel: m\n---\nYou are the keeper.\n")?;
        assert_eq!(
            resolve_system_prompt(&dir, &spec)?.as_deref(),
            Some("You are the keeper.")
        );
        let literal = Spec::parse("---\nsystem_prompt: inline wins\n---\nbody ignored\n")?;
        assert_eq!(
            resolve_system_prompt(&dir, &literal)?.as_deref(),
            Some("inline wins")
        );
        Ok(())
    }

    #[test]
    fn system_prompt_path_resolves_relative_to_session_dir() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("agentd-spec-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("system.md"), "From the file.")?;
        let spec = Spec::parse("---\nsystem_prompt: ./system.md\n---\n")?;
        assert_eq!(
            resolve_system_prompt(&dir, &spec)?.as_deref(),
            Some("From the file.")
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn set_value_for_validates_max_iterations() {
        assert!(set_value_for("max_iterations", "12").is_ok());
        assert!(set_value_for("max_iterations", "twelve").is_err());
        assert!(set_value_for("nonsense", "x").is_err());
    }
}
