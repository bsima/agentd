//! Cheap disk-backed ACP session index. No runtime or provider is built to list.
use std::path::{Path, PathBuf};

use agent_client_protocol::schema::v1::{ListSessionsRequest, ListSessionsResponse, SessionInfo};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::Checkpoint;

const PAGE_SIZE: usize = 50;
const INFO_FILE: &str = "acp-info.json";

#[derive(Serialize, Deserialize)]
pub(super) struct SessionMetadata {
    pub cwd: PathBuf,
    pub title: Option<String>,
}

pub(super) fn title(text: &str) -> Option<String> {
    let line = text.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    let mut result: String = line.chars().take(60).collect();
    if line.chars().count() > 60 {
        result.push('…');
    }
    Some(result)
}

pub(super) fn checkpoint_title(checkpoint: &Checkpoint) -> Option<String> {
    checkpoint
        .messages
        .iter()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref())
        .and_then(title)
}

pub(super) async fn write_metadata(dir: &Path, metadata: &SessionMetadata) -> Result<()> {
    tokio::fs::create_dir_all(dir).await?;
    // Atomic replacement: list never sees a partially written document.
    let tmp = dir.join(format!("{INFO_FILE}.{}", uuid::Uuid::new_v4()));
    tokio::fs::write(&tmp, serde_json::to_vec(metadata)?).await?;
    if let Err(err) = tokio::fs::rename(&tmp, dir.join(INFO_FILE)).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(err.into());
    }
    Ok(())
}

pub(super) async fn read_metadata(dir: &Path) -> Result<SessionMetadata> {
    Ok(serde_json::from_slice(
        &tokio::fs::read(dir.join(INFO_FILE)).await?,
    )?)
}

pub(super) async fn list(
    root: &Path,
    request: ListSessionsRequest,
) -> Result<ListSessionsResponse> {
    let mut rows = Vec::new();
    let mut entries = match tokio::fs::read_dir(root).await {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ListSessionsResponse::new(vec![]))
        }
        Err(err) => return Err(err.into()),
    };
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name();
        let Some(id) = name.to_str() else { continue };
        if uuid::Uuid::parse_str(id).is_err() || !entry.file_type().await?.is_dir() {
            continue;
        }
        let dir = entry.path();
        let checkpoint_path = dir.join("session-latest.json");
        let Ok(file) = tokio::fs::metadata(&checkpoint_path).await else {
            continue;
        };
        let Ok(mut info) = read_metadata(&dir).await else {
            continue;
        };
        if request.cwd.as_ref().is_some_and(|cwd| cwd != &info.cwd) {
            continue;
        }
        if info.title.is_none() {
            if let Ok(bytes) = tokio::fs::read(&checkpoint_path).await {
                if let Ok(checkpoint) = serde_json::from_slice::<Checkpoint>(&bytes) {
                    info.title = checkpoint_title(&checkpoint);
                }
            }
        }
        let modified = file.modified().unwrap_or(std::time::UNIX_EPOCH);
        let updated = chrono::DateTime::<chrono::Utc>::from(modified).to_rfc3339();
        rows.push((
            modified,
            id.to_owned(),
            SessionInfo::new(id.to_owned(), info.cwd)
                .title(info.title)
                .updated_at(updated),
        ));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    // The cursor is an exclusive sort key (timestamp nanos + UUID). This
    // remains stable if newer sessions appear between pages.
    let start = if let Some(cursor) = request.cursor {
        let (time, id) = cursor
            .split_once(':')
            .ok_or_else(|| anyhow!("invalid session cursor"))?;
        let nanos: i128 = time
            .parse()
            .map_err(|_| anyhow!("invalid session cursor"))?;
        if uuid::Uuid::parse_str(id).is_err() {
            return Err(anyhow!("invalid session cursor"));
        }
        rows.partition_point(|(modified, row_id, _)| {
            let t = modified
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as i128;
            (t, row_id.as_str()) >= (nanos, id)
        })
    } else {
        0
    };
    let end = (start + PAGE_SIZE).min(rows.len());
    let next = if end < rows.len() {
        let (time, id, _) = &rows[end - 1];
        Some(format!(
            "{}:{id}",
            time.duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    } else {
        None
    };
    Ok(ListSessionsResponse::new(
        rows.into_iter()
            .skip(start)
            .take(PAGE_SIZE)
            .map(|(_, _, row)| row)
            .collect(),
    )
    .next_cursor(next))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn title_is_first_line_unicode_safe() {
        assert_eq!(
            title("  hello world  \nignored"),
            Some("hello world".into())
        );
        assert_eq!(title("\nsecond"), None);
        assert_eq!(title(&"é".repeat(61)).unwrap().chars().count(), 61);
    }
    #[tokio::test]
    async fn pages_and_filters() {
        let root = std::env::temp_dir().join(format!("acp-list-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        for i in 0..102 {
            let id = uuid::Uuid::new_v4();
            let dir = root.join(id.to_string());
            write_metadata(
                &dir,
                &SessionMetadata {
                    cwd: PathBuf::from(if i % 2 == 0 { "/a" } else { "/b" }),
                    title: Some(format!("title {i}")),
                },
            )
            .await
            .unwrap();
            tokio::fs::write(dir.join("session-latest.json"), b"{}")
                .await
                .unwrap();
        }
        let mut cursor = None;
        let mut ids = std::collections::HashSet::new();
        loop {
            let page = list(
                &root,
                ListSessionsRequest::new()
                    .cwd(PathBuf::from("/a"))
                    .cursor(cursor),
            )
            .await
            .unwrap();
            assert!(page.sessions.len() <= PAGE_SIZE);
            for row in page.sessions {
                assert_eq!(row.cwd, PathBuf::from("/a"));
                assert!(ids.insert(row.session_id.0.to_string()));
            }
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(ids.len(), 51);
        tokio::fs::remove_dir_all(root).await.unwrap();
    }
}
