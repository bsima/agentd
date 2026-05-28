use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct Frontmatter {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub max_iterations: Option<usize>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownPrompt {
    pub body: String,
    pub path: Option<PathBuf>,
    pub base_dir: PathBuf,
    pub frontmatter: Option<Frontmatter>,
}

impl MarkdownPrompt {
    pub async fn from_arg(arg: &str) -> Result<Self> {
        let path = PathBuf::from(arg);
        if is_markdown_path(&path) && tokio::fs::try_exists(&path).await? {
            let content = tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("reading markdown prompt {}", path.display()))?;
            let content = strip_shebang(&content);
            let (frontmatter, body) = parse_markdown_prompt(&content)?;
            let full_path = std::fs::canonicalize(&path)
                .with_context(|| format!("resolving markdown prompt {}", path.display()))?;
            let base_dir = full_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            Ok(Self {
                body: body.trim().to_string(),
                path: Some(full_path),
                base_dir,
                frontmatter,
            })
        } else {
            Ok(Self {
                body: arg.to_string(),
                path: None,
                base_dir: std::env::current_dir().context("getting current directory")?,
                frontmatter: None,
            })
        }
    }
}

pub fn is_markdown_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext, "md" | "markdown"))
}

pub fn strip_shebang(content: &str) -> String {
    if let Some(rest) = content.strip_prefix("#!") {
        match rest.find('\n') {
            Some(idx) => rest[idx + 1..].to_string(),
            None => String::new(),
        }
    } else {
        content.to_string()
    }
}

pub fn parse_markdown_prompt(content: &str) -> Result<(Option<Frontmatter>, String)> {
    let mut lines = content.lines();
    let Some(first) = lines.next() else {
        return Ok((None, String::new()));
    };

    if first.trim() != "---" {
        return Ok((None, content.to_string()));
    }

    let mut yaml_lines = Vec::new();
    let mut body_lines = Vec::new();
    let mut found_end = false;
    for line in lines.by_ref() {
        if line.trim() == "---" {
            found_end = true;
            break;
        }
        yaml_lines.push(line);
    }
    if !found_end {
        return Err(anyhow!("markdown prompt has unclosed YAML frontmatter"));
    }
    body_lines.extend(lines);

    let yaml = yaml_lines.join("\n");
    let body = body_lines.join("\n");
    if yaml.trim().is_empty() {
        return Ok((None, body));
    }
    let frontmatter = serde_yaml::from_str::<Frontmatter>(&yaml)
        .with_context(|| "parsing markdown prompt frontmatter")?;
    Ok((Some(frontmatter), body))
}

pub async fn resolve_system_prompt(base_dir: &Path, value: Option<&str>) -> Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !looks_like_path(value) {
        return Ok(Some(value.to_string()));
    }

    let path = PathBuf::from(value);
    let full_path = if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    };
    if tokio::fs::try_exists(&full_path).await? {
        let content = tokio::fs::read_to_string(&full_path)
            .await
            .with_context(|| format!("reading system prompt {}", full_path.display()))?;
        Ok(Some(strip_shebang(&content)))
    } else {
        Ok(Some(value.to_string()))
    }
}

fn looks_like_path(value: &str) -> bool {
    value.contains('/')
        || value.ends_with(".md")
        || value.ends_with(".markdown")
        || value.ends_with(".txt")
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn parses_frontmatter_and_body() -> Result<()> {
        let content =
            "---\nmodel: test-model\nmax_iterations: 3\nsystem_prompt: ./system.md\n---\nDo it.";
        let (frontmatter, body) = parse_markdown_prompt(content)?;
        let frontmatter = frontmatter.expect("frontmatter");
        assert_eq!(frontmatter.model.as_deref(), Some("test-model"));
        assert_eq!(frontmatter.max_iterations, Some(3));
        assert_eq!(frontmatter.system_prompt.as_deref(), Some("./system.md"));
        assert_eq!(body, "Do it.");
        Ok(())
    }

    #[test]
    fn leaves_plain_markdown_as_body() -> Result<()> {
        let (frontmatter, body) = parse_markdown_prompt("# Task\n\nDo it.")?;
        assert!(frontmatter.is_none());
        assert_eq!(body, "# Task\n\nDo it.");
        Ok(())
    }

    #[test]
    fn rejects_unclosed_frontmatter() {
        let err = parse_markdown_prompt("---\nmodel: x\nbody").unwrap_err();
        assert!(err.to_string().contains("unclosed YAML frontmatter"));
    }

    #[test]
    fn strips_shebang_line() {
        assert_eq!(strip_shebang("#!/usr/bin/env agent\nDo it."), "Do it.");
        assert_eq!(strip_shebang("Do it."), "Do it.");
    }

    #[tokio::test]
    async fn resolves_system_prompt_path_relative_to_markdown() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("agent-frontmatter-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await?;
        tokio::fs::write(dir.join("system.md"), "Custom system").await?;
        let resolved = resolve_system_prompt(&dir, Some("system.md")).await?;
        assert_eq!(resolved.as_deref(), Some("Custom system"));
        Ok(())
    }
}
