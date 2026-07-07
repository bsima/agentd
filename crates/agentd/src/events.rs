//! Reading the supervised child's captured `--json` stdout
//! (`<name>/stdout.jsonl`): the machine events (`agent_start`,
//! `agent_complete`, `agent_error`) are the supervisor's response path
//! (docs/SUPERVISOR.md "Turn delivery and response correlation").
//!
//! Under `--json` the child also mirrors runtime trace events to stdout;
//! anything that is not a machine event is skipped, and half-written lines
//! (a tail racing the writer) are left unconsumed until their newline
//! arrives.

use anyhow::{Context, Result};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// A machine event the supervisor acts on, parsed from one stdout line.
#[derive(Debug, Clone, PartialEq)]
pub enum MachineEvent {
    Start {
        turn_id: String,
        model: Option<String>,
    },
    Complete {
        turn_id: String,
        response: String,
    },
    Error {
        turn_id: String,
        message: String,
    },
}

impl MachineEvent {
    pub fn turn_id(&self) -> &str {
        match self {
            MachineEvent::Start { turn_id, .. }
            | MachineEvent::Complete { turn_id, .. }
            | MachineEvent::Error { turn_id, .. } => turn_id,
        }
    }
}

/// Parse one stdout line into a machine event; `None` for trace mirrors,
/// partial garbage, or machine events we do not act on.
pub fn parse_machine_event(line: &str) -> Option<MachineEvent> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    if value.get("type").and_then(serde_json::Value::as_str) != Some("custom") {
        return None;
    }
    let data = value.get("data")?;
    let turn_id = data
        .get("turn_id")
        .and_then(serde_json::Value::as_str)?
        .to_string();
    match value
        .get("custom_type")
        .and_then(serde_json::Value::as_str)?
    {
        "agent_start" => Some(MachineEvent::Start {
            turn_id,
            model: data
                .pointer("/config/model")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        }),
        "agent_complete" => Some(MachineEvent::Complete {
            turn_id,
            response: data
                .get("response")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        }),
        "agent_error" => Some(MachineEvent::Error {
            turn_id,
            message: data
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        }),
        _ => None,
    }
}

/// An offset-tail over a JSONL file. Only complete (newline-terminated)
/// lines are consumed; the offset never advances past a partial line, so a
/// poll that races the writer re-reads the fragment next time.
#[derive(Debug)]
pub struct Tail {
    path: PathBuf,
    offset: u64,
}

impl Tail {
    pub fn from_offset(path: impl Into<PathBuf>, offset: u64) -> Self {
        Self {
            path: path.into(),
            offset,
        }
    }

    /// New complete lines appended since the last poll. A missing file is
    /// an empty poll, not an error (the child may not have started writing).
    pub fn read_new_lines(&mut self) -> Result<Vec<String>> {
        let mut file = match std::fs::File::open(&self.path) {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(err).with_context(|| format!("opening {}", self.path.display()))
            }
        };
        file.seek(SeekFrom::Start(self.offset))
            .with_context(|| format!("seeking {}", self.path.display()))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .with_context(|| format!("reading {}", self.path.display()))?;
        let Some(last_newline) = buf.iter().rposition(|&b| b == b'\n') else {
            return Ok(Vec::new());
        };
        let complete = &buf[..=last_newline];
        self.offset += complete.len() as u64;
        Ok(String::from_utf8_lossy(complete)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_string)
            .collect())
    }

    /// New machine events appended since the last poll.
    pub fn read_new_events(&mut self) -> Result<Vec<MachineEvent>> {
        Ok(self
            .read_new_lines()?
            .iter()
            .filter_map(|line| parse_machine_event(line))
            .collect())
    }
}

/// The last parseable JSON line of a file, for `agentd status`'s "last
/// event" column. Reads only a bounded tail so status stays cheap on big
/// session logs.
pub fn last_event_summary(path: &Path) -> Option<(String, Option<String>)> {
    const TAIL_BYTES: u64 = 16 * 1024;
    let mut file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    file.seek(SeekFrom::Start(len.saturating_sub(TAIL_BYTES)))
        .ok()?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).ok()?;
    for line in buf.lines().rev() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let name = value
            .get("custom_type")
            .or_else(|| value.get("event"))
            .or_else(|| value.get("type"))
            .and_then(serde_json::Value::as_str)?
            .to_string();
        let timestamp = value
            .get("timestamp")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        return Some((name, timestamp));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_machine_events_and_skips_trace_mirrors() {
        let complete = r#"{"type":"custom","custom_type":"agent_complete","data":{"response":"hi","turn_id":"t-1"},"timestamp":"2026-01-01T00:00:00Z"}"#;
        assert_eq!(
            parse_machine_event(complete),
            Some(MachineEvent::Complete {
                turn_id: "t-1".into(),
                response: "hi".into()
            })
        );
        let error = r#"{"type":"custom","custom_type":"agent_error","data":{"message":"boom","turn_id":"t-2"}}"#;
        assert_eq!(
            parse_machine_event(error),
            Some(MachineEvent::Error {
                turn_id: "t-2".into(),
                message: "boom".into()
            })
        );
        let start = r#"{"type":"custom","custom_type":"agent_start","data":{"turn_id":"t-3","config":{"model":"sonnet"}}}"#;
        assert_eq!(
            parse_machine_event(start),
            Some(MachineEvent::Start {
                turn_id: "t-3".into(),
                model: Some("sonnet".into())
            })
        );
        let trace_mirror = r#"{"event":"InferCall","run_id":"r","op_id":2}"#;
        assert_eq!(parse_machine_event(trace_mirror), None);
        assert_eq!(parse_machine_event("not json"), None);
    }

    #[test]
    fn tail_consumes_only_complete_lines() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("agentd-tail-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("stdout.jsonl");
        let mut tail = Tail::from_offset(&path, 0);
        assert!(tail.read_new_lines()?.is_empty(), "missing file is empty");

        std::fs::write(&path, "{\"a\":1}\n{\"b\":")?;
        assert_eq!(tail.read_new_lines()?, vec!["{\"a\":1}".to_string()]);
        assert!(tail.read_new_lines()?.is_empty(), "partial line unconsumed");

        std::fs::write(&path, "{\"a\":1}\n{\"b\":2}\n")?;
        assert_eq!(tail.read_new_lines()?, vec!["{\"b\":2}".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn last_event_summary_reads_the_final_line() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("agentd-last-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("stdout.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"event":"InferCall","timestamp":"2026-01-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"custom","custom_type":"agent_complete","data":{"turn_id":"t"},"timestamp":"2026-01-02T00:00:00Z"}"#,
                "\n",
            ),
        )?;
        let (name, ts) = last_event_summary(&path).expect("an event");
        assert_eq!(name, "agent_complete");
        assert_eq!(ts.as_deref(), Some("2026-01-02T00:00:00Z"));
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
