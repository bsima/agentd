//! Profile-backed ACP modes. Loaded once at server startup, never from a turn.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use super::registry::{MODE_ASK, MODE_YOLO};

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ShellApproval {
    #[default]
    Ask,
    Always,
    Never,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    name: Option<String>,
    description: Option<String>,
    #[serde(default)]
    shell_approval: ShellApproval,
    #[serde(default)]
    tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct Profile {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub shell_approval: ShellApproval,
    pub instructions: String,
}

#[derive(Debug, Default)]
pub(crate) struct Profiles(pub BTreeMap<String, Profile>);

// No native opt-in tools are currently available: ask_user is always on in
// ACP, and zoom depends on the GC strategy. Extend this registry when an
// independently switchable tool ships.
const OPT_IN_TOOLS: &[&str] = &[];

impl Profiles {
    pub fn get(&self, id: &str) -> Option<&Profile> {
        self.0.get(id)
    }

    pub fn load(dir: Option<&Path>) -> Result<Self> {
        let path = match dir {
            Some(path) => path.to_path_buf(),
            None => {
                let home = std::env::var_os("XDG_CONFIG_HOME")
                    .map(PathBuf::from)
                    .or_else(|| dirs::home_dir().map(|home| home.join(".config")))
                    .ok_or_else(|| anyhow!("could not determine config directory"))?;
                home.join("agent/modes")
            }
        };
        if dir.is_none() && !path.exists() {
            return Ok(Self::default());
        }
        let mut profiles = BTreeMap::new();
        for entry in std::fs::read_dir(&path)
            .with_context(|| format!("reading modes directory {}", path.display()))?
        {
            let entry = entry?;
            let file = entry.path();
            if file.extension().is_none_or(|ext| ext != "md") {
                continue;
            }
            let profile =
                load_profile(&file).with_context(|| format!("mode profile {}", file.display()))?;
            if profiles.insert(profile.id.clone(), profile).is_some() {
                return Err(anyhow!("duplicate mode profile id in {}", path.display()));
            }
        }
        Ok(Self(profiles))
    }
}

fn load_profile(file: &Path) -> Result<Profile> {
    let id = file
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("invalid mode file name"))?
        .to_owned();
    if matches!(id.as_str(), MODE_ASK | MODE_YOLO) {
        return Err(anyhow!("mode id {id:?} collides with a built-in mode"));
    }
    let content = std::fs::read_to_string(file)?;
    let content = crate::frontmatter::strip_shebang(&content);
    let (yaml, body) = split_frontmatter(&content)?;
    let meta: Metadata = serde_yaml::from_str(yaml).context("invalid YAML frontmatter")?;
    for tool in &meta.tools {
        if !OPT_IN_TOOLS.contains(&tool.as_str()) {
            return Err(anyhow!("unknown opt-in tool {tool:?}"));
        }
    }
    Ok(Profile {
        name: meta.name.unwrap_or_else(|| id.clone()),
        id,
        description: meta.description,
        shell_approval: meta.shell_approval,
        instructions: body.trim().to_owned(),
    })
}

fn split_frontmatter(content: &str) -> Result<(&str, &str)> {
    let rest = content
        .strip_prefix("---\n")
        .ok_or_else(|| anyhow!("missing YAML frontmatter"))?;
    let (yaml, body) = rest
        .split_once("\n---\n")
        .or_else(|| rest.strip_suffix("\n---").map(|yaml| (yaml, "")))
        .ok_or_else(|| anyhow!("unclosed YAML frontmatter"))?;
    Ok((yaml, body))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_profiles_and_rejects_bad_policy() {
        let dir = std::env::temp_dir().join(format!("acp-modes-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let file = dir.join("research.md");
        std::fs::write(
            &file,
            "---\nname: Research\nshell_approval: always\n---\nInvestigate.",
        )
        .unwrap();
        let profiles = Profiles::load(Some(&dir)).unwrap();
        let p = profiles.get("research").unwrap();
        assert_eq!(p.name, "Research");
        assert_eq!(p.shell_approval, ShellApproval::Always);
        assert_eq!(p.instructions, "Investigate.");
        std::fs::write(&file, "---\ntools: [missing]\n---\nBody").unwrap();
        let err = Profiles::load(Some(&dir)).unwrap_err().to_string();
        assert!(err.contains("research.md"), "{err}");
        std::fs::write(dir.join("ask.md"), "---\nname: Ask\n---\nBody").unwrap();
        std::fs::remove_file(&file).unwrap();
        assert!(Profiles::load(Some(&dir))
            .unwrap_err()
            .to_string()
            .contains("ask.md"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
