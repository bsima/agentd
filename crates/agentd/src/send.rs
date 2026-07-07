//! Turn delivery and response correlation (docs/SUPERVISOR.md):
//!
//! 1. record the current `stdout.jsonl` offset,
//! 2. write the NUL-framed v1 turn envelope — carrying a
//!    supervisor-generated `turn_id` — to the session FIFO, under the
//!    per-session flock (concurrent FIFO writers interleave beyond
//!    `PIPE_BUF`),
//! 3. tail `stdout.jsonl` from the offset until the
//!    `agent_complete`/`agent_error` event carrying that `turn_id`.
//!
//! `--timeout` times out the CALLER only: the turn keeps running and
//! `agentd attach <name> <turn_id>` re-attaches to its eventual result.

use crate::events::{MachineEvent, Tail};
use crate::session::Session;
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::{Duration, Instant};

/// How a wait for a turn ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Complete {
        response: String,
    },
    /// The turn errored agent-side (`agent_error`).
    Error {
        message: String,
    },
    /// The CALLER hit its `--timeout`; the turn is still running.
    TimedOut,
}

/// Supervisor-minted turn ids, recomputable/persistable by the sender
/// (spec example: `send-4f2a`).
pub fn mint_turn_id() -> String {
    let uuid = uuid::Uuid::new_v4().simple().to_string();
    format!("send-{}", &uuid[..12])
}

/// The v1 turn envelope frame (shipped agent-side, t-1308.2).
pub fn envelope(turn_id: &str, input: &str, metadata: Option<&serde_json::Value>) -> String {
    let mut frame = serde_json::json!({ "v": 1, "turn_id": turn_id, "input": input });
    if let Some(metadata) = metadata {
        frame["metadata"] = metadata.clone();
    }
    frame.to_string()
}

/// Deliver one envelope frame to the session FIFO and return the
/// `stdout.jsonl` offset recorded before delivery — the point to tail from.
pub fn deliver(
    session: &Session,
    turn_id: &str,
    input: &str,
    metadata: Option<&serde_json::Value>,
    deadline: Option<Instant>,
) -> Result<u64> {
    session.require_exists()?;
    let Some(pid) = session.running() else {
        bail!(
            "session '{}' is not running; start it with `agentd start {}` or `agentd resume {}`",
            session.name,
            session.name,
            session.name
        );
    };
    // Per-session flock: guarantees frame atomicity beyond PIPE_BUF for
    // concurrent senders. Held across offset + open + write so a frame is
    // never interleaved and the offset is always at-or-before our events.
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(session.send_lock_path())
        .with_context(|| format!("opening {}", session.send_lock_path().display()))?;
    flock_exclusive(&lock)?;
    let offset = std::fs::metadata(session.stdout_path())
        .map(|meta| meta.len())
        .unwrap_or(0);
    let mut fifo = open_fifo_writer(&session.fifo_path(), session, pid, deadline)?;
    fifo.write_all(envelope(turn_id, input, metadata).as_bytes())
        .context("writing turn frame to fifo")?;
    fifo.write_all(&[0]).context("writing frame terminator")?;
    fifo.flush().context("flushing fifo")?;
    // Dropping `fifo` closes the write end; dropping `lock` releases the
    // flock.
    Ok(offset)
}

/// Tail `stdout.jsonl` from `offset` until the completion/error event
/// carrying `turn_id`, the deadline passes, or the session dies.
pub fn await_turn(
    session: &Session,
    turn_id: &str,
    offset: u64,
    deadline: Option<Instant>,
) -> Result<Outcome> {
    let mut tail = Tail::from_offset(session.stdout_path(), offset);
    loop {
        if let Some(outcome) = match_events(&mut tail, turn_id)? {
            return Ok(outcome);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Ok(Outcome::TimedOut);
        }
        if session.running().is_none() {
            // Final drain: the completion may have landed between the last
            // poll and the process exiting.
            if let Some(outcome) = match_events(&mut tail, turn_id)? {
                return Ok(outcome);
            }
            bail!(
                "session '{}' exited before turn {turn_id} completed; resume it \
                 (`agentd resume {}`) and check `agentd logs {} --raw`",
                session.name,
                session.name,
                session.name
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn match_events(tail: &mut Tail, turn_id: &str) -> Result<Option<Outcome>> {
    for event in tail.read_new_events()? {
        if event.turn_id() != turn_id {
            continue;
        }
        match event {
            MachineEvent::Complete { response, .. } => {
                return Ok(Some(Outcome::Complete { response }))
            }
            MachineEvent::Error { message, .. } => return Ok(Some(Outcome::Error { message })),
            MachineEvent::Start { .. } => {}
        }
    }
    Ok(None)
}

/// Open the FIFO for writing without wedging forever: `O_NONBLOCK` fails
/// with ENXIO until the agent holds the read end (it re-opens between
/// bursts), so retry while the child is alive and the deadline allows,
/// then clear `O_NONBLOCK` so the actual frame write blocks normally.
fn open_fifo_writer(
    path: &Path,
    session: &Session,
    pid: i32,
    deadline: Option<Instant>,
) -> Result<File> {
    loop {
        match File::options()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
        {
            Ok(fifo) => {
                clear_nonblock(&fifo)?;
                return Ok(fifo);
            }
            Err(err) if err.raw_os_error() == Some(libc::ENXIO) => {
                if !crate::session::pid_alive(pid) {
                    bail!(
                        "session '{}' died before accepting the turn (no FIFO reader)",
                        session.name
                    );
                }
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    bail!(
                        "timed out waiting for session '{}' to accept the turn (no FIFO reader)",
                        session.name
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(err) => {
                return Err(err).with_context(|| format!("opening fifo {}", path.display()))
            }
        }
    }
}

fn flock_exclusive(file: &File) -> Result<()> {
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(std::io::Error::last_os_error()).context("taking the per-session send lock");
    }
    Ok(())
}

fn clear_nonblock(file: &File) -> Result<()> {
    let fd = file.as_raw_fd();
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error()).context("reading fifo flags");
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error()).context("clearing O_NONBLOCK on the fifo");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_matches_the_v1_contract() {
        let frame = envelope(
            "send-4f2a",
            "run tests",
            Some(&serde_json::json!({"sender": "ben"})),
        );
        let value: serde_json::Value = serde_json::from_str(&frame).unwrap();
        assert_eq!(value["v"], 1);
        assert_eq!(value["turn_id"], "send-4f2a");
        assert_eq!(value["input"], "run tests");
        assert_eq!(value["metadata"]["sender"], "ben");

        let bare: serde_json::Value = serde_json::from_str(&envelope("id", "hi", None)).unwrap();
        assert!(bare.get("metadata").is_none());
    }

    #[test]
    fn minted_turn_ids_are_prefixed_and_unique() {
        let a = mint_turn_id();
        let b = mint_turn_id();
        assert!(a.starts_with("send-"));
        assert_ne!(a, b);
    }
}
