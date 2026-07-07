//! Session layout and lifecycle: a session is a directory under
//! `$AGENTD_HOME`, and it is "running" when the pid in `<name>/pid` is
//! live. No registry database — two `agentd` invocations coordinate through
//! the filesystem only (docs/SUPERVISOR.md "Design stance").

use crate::spec::{self, Spec, SPEC_FILE};
use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::FileTypeExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Resolve `$AGENTD_HOME` (CLI flag/env already merged by clap), defaulting
/// to `~/.local/share/agentd`.
pub fn agentd_home(cli: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(home) = cli {
        return Ok(home);
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
    Ok(home.join(".local/share/agentd"))
}

/// One named session's paths. Everything lives under `dir()`; paths are
/// derived, never stored, so the layout stays relocatable with
/// `$AGENTD_HOME` (the M5 multi-host constraint).
#[derive(Debug, Clone)]
pub struct Session {
    pub name: String,
    pub home: PathBuf,
}

impl Session {
    pub fn new(home: PathBuf, name: &str) -> Result<Self> {
        validate_name(name)?;
        Ok(Self {
            name: name.to_string(),
            home,
        })
    }

    pub fn dir(&self) -> PathBuf {
        self.home.join(&self.name)
    }

    pub fn spec_path(&self) -> PathBuf {
        self.dir().join(SPEC_FILE)
    }

    pub fn fifo_path(&self) -> PathBuf {
        self.dir().join("fifo")
    }

    pub fn pid_path(&self) -> PathBuf {
        self.dir().join("pid")
    }

    pub fn run_id_path(&self) -> PathBuf {
        self.dir().join("run-id")
    }

    pub fn stdout_path(&self) -> PathBuf {
        self.dir().join("stdout.jsonl")
    }

    pub fn stderr_path(&self) -> PathBuf {
        self.dir().join("stderr.log")
    }

    pub fn checkpoints_dir(&self) -> PathBuf {
        self.dir().join("checkpoints")
    }

    pub fn send_lock_path(&self) -> PathBuf {
        self.dir().join("send.lock")
    }

    /// A session exists if its directory exists.
    pub fn exists(&self) -> bool {
        self.dir().is_dir()
    }

    pub fn require_exists(&self) -> Result<()> {
        if !self.exists() {
            bail!(
                "no session named '{}' under {} (create one with `agentd start {}`)",
                self.name,
                self.home.display(),
                self.name
            );
        }
        Ok(())
    }

    pub fn pid(&self) -> Option<i32> {
        std::fs::read_to_string(self.pid_path())
            .ok()?
            .trim()
            .parse()
            .ok()
    }

    /// The live pid, or `None` when stopped/crashed (pid file missing,
    /// unparseable, or pointing at a dead process).
    pub fn running(&self) -> Option<i32> {
        self.pid().filter(|&pid| pid_alive(pid))
    }

    pub fn run_id(&self) -> Option<String> {
        let id = std::fs::read_to_string(self.run_id_path()).ok()?;
        let id = id.trim().to_string();
        (!id.is_empty()).then_some(id)
    }

    /// The newest checkpoint to resume from: the `session-latest.json`
    /// pointer the agent's checkpoint sink maintains, with fallbacks for
    /// partially-written directories.
    pub fn latest_checkpoint(&self) -> Option<PathBuf> {
        let dir = self.checkpoints_dir();
        for pointer in ["session-latest.json", "latest.json"] {
            let path = dir.join(pointer);
            if path.is_file() {
                return Some(path);
            }
        }
        // Newest numbered checkpoint by mtime.
        let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
        for entry in std::fs::read_dir(&dir).ok()?.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("checkpoint-") || !name.ends_with(".json") {
                continue;
            }
            let mtime = entry.metadata().ok()?.modified().ok()?;
            if newest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                newest = Some((mtime, path));
            }
        }
        newest.map(|(_, path)| path)
    }
}

/// Session names become directory names and systemd unit names; keep them
/// boring. Rejects path traversal, hidden dirs, and flag-looking names.
fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && !name.starts_with(['.', '-'])
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !ok {
        bail!(
            "invalid session name {name:?}: use ASCII letters, digits, '-', '_' or '.' \
             (must not start with '.' or '-')"
        );
    }
    Ok(())
}

pub fn pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

fn ensure_fifo(path: &Path) -> Result<()> {
    match std::fs::metadata(path) {
        Ok(meta) if meta.file_type().is_fifo() => Ok(()),
        Ok(_) => bail!("{} exists and is not a FIFO", path.display()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let cpath = CString::new(path.as_os_str().as_bytes())
                .context("fifo path contains a NUL byte")?;
            if unsafe { libc::mkfifo(cpath.as_ptr(), 0o600) } != 0 {
                return Err(std::io::Error::last_os_error())
                    .with_context(|| format!("mkfifo {}", path.display()));
            }
            Ok(())
        }
        Err(err) => Err(err).with_context(|| format!("checking fifo {}", path.display())),
    }
}

/// Config flags `agentd start` accepts only to SEED a missing spec file.
/// Once `<name>/agent.md` exists it is canonical (t-1105): launches read it
/// fresh, and config changes go through the file (`agentd set-*` or hand
/// edits), never through launch flags that could drift from it.
#[derive(Debug, Clone, Default)]
pub struct SpecSeed {
    pub model: Option<String>,
    pub provider: Option<String>,
    pub system_prompt: Option<String>,
    pub max_turns: Option<usize>,
}

impl SpecSeed {
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.provider.is_none()
            && self.system_prompt.is_none()
            && self.max_turns.is_none()
    }

    fn into_spec(self) -> Spec {
        let mut spec = Spec::default();
        if let Some(model) = self.model {
            spec.set("model", serde_yaml::Value::String(model));
        }
        if let Some(provider) = self.provider {
            spec.set("provider", serde_yaml::Value::String(provider));
        }
        if let Some(prompt) = self.system_prompt {
            spec.set("system_prompt", serde_yaml::Value::String(prompt));
        }
        if let Some(turns) = self.max_turns {
            spec.set(
                "max_iterations",
                serde_yaml::Value::Number((turns as u64).into()),
            );
        }
        spec
    }
}

pub struct Launch<'a> {
    pub agent_bin: &'a Path,
    pub seed: SpecSeed,
    /// Resume from the latest checkpoint instead of starting fresh.
    pub resume: bool,
    /// Extra `agent` argv appended after the spec-derived flags (after
    /// `--` on the CLI). Ideal for eval/test plumbing like
    /// `--replay-trace`.
    pub extra_args: &'a [String],
}

/// Spawn the supervised `agent` child for `start` (fresh) or `resume`
/// (latest checkpoint). Reads the spec fresh, mkfifos, captures `--json`
/// stdout to `stdout.jsonl`, detaches the child into its own session, and
/// writes pid/run-id. Returns (pid, run_id).
pub fn launch(session: &Session, launch: Launch<'_>) -> Result<(u32, String)> {
    if let Some(pid) = session.running() {
        bail!(
            "session '{}' is already running (pid {pid}); stop it first with `agentd stop {}`",
            session.name,
            session.name
        );
    }
    if launch.resume {
        session.require_exists()?;
    }
    std::fs::create_dir_all(session.checkpoints_dir())
        .with_context(|| format!("creating {}", session.checkpoints_dir().display()))?;

    // The spec file is canonical: seed it only if absent, refuse config
    // flags when it exists (they would silently disagree with the file).
    let spec_path = session.spec_path();
    if spec_path.is_file() {
        if !launch.seed.is_empty() {
            bail!(
                "spec {} already exists and is canonical; edit it directly or use \
                 `agentd set-model`/`agentd set-*` instead of start flags",
                spec_path.display()
            );
        }
    } else {
        launch.seed.into_spec().save(&spec_path)?;
    }
    let spec = Spec::load(&spec_path)?;
    let config = spec.config()?;
    let system_prompt = spec::resolve_system_prompt(&session.dir(), &spec)?;

    ensure_fifo(&session.fifo_path())?;

    // Resume goes through the latest checkpoint when one exists; a resume
    // with no checkpoint yet (e.g. systemd's first ExecStart) degrades to a
    // fresh start — the checkpoint model makes both idempotent to retry.
    let resume_from = if launch.resume {
        session.latest_checkpoint()
    } else {
        None
    };
    if launch.resume && resume_from.is_none() {
        eprintln!(
            "note: session '{}' has no checkpoint yet; starting fresh",
            session.name
        );
    }
    // The run id is stable for traces: a resumed run keeps the checkpoint's
    // id (the agent enforces this; we mirror it into run-id for status/logs).
    let run_id = match resume_from.as_deref() {
        Some(checkpoint) => {
            checkpoint_run_id(checkpoint).unwrap_or_else(|| mint_run_id(&session.name))
        }
        None => mint_run_id(&session.name),
    };
    std::fs::write(session.run_id_path(), format!("{run_id}\n"))
        .with_context(|| format!("writing {}", session.run_id_path().display()))?;

    let stdout = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(session.stdout_path())
        .with_context(|| format!("opening {}", session.stdout_path().display()))?;
    let stderr = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(session.stderr_path())
        .with_context(|| format!("opening {}", session.stderr_path().display()))?;

    let mut cmd = Command::new(launch.agent_bin);
    cmd.arg("--fifo")
        .arg(session.fifo_path())
        .arg("--json")
        .arg("--checkpoint-dir")
        .arg(session.checkpoints_dir())
        .arg("--run-id")
        .arg(&run_id);
    if let Some(model) = &config.model {
        cmd.args(["--model", model]);
    }
    if let Some(provider) = &config.provider {
        cmd.args(["--provider", provider]);
    }
    if let Some(max) = config.max_iterations {
        cmd.args(["--max-turns", &max.to_string()]);
    }
    if let Some(prompt) = &system_prompt {
        cmd.args(["--system-prompt", prompt]);
    }
    if let Some(checkpoint) = &resume_from {
        cmd.arg("--resume").arg(checkpoint);
    }
    cmd.args(&config.args);
    cmd.args(launch.extra_args);
    cmd.env("AGENT_NAME", &session.name)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    // Detach: the child must outlive this CLI and ignore the launching
    // terminal's signals — supervision-as-such belongs to the init system.
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", launch.agent_bin.display()))?;
    let pid = child.id();
    std::fs::write(session.pid_path(), format!("{pid}\n"))
        .with_context(|| format!("writing {}", session.pid_path().display()))?;
    Ok((pid, run_id))
}

fn mint_run_id(name: &str) -> String {
    format!("{name}-{}", uuid::Uuid::new_v4())
}

fn checkpoint_run_id(path: &Path) -> Option<String> {
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    value
        .get("run_id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// SIGTERM, escalate to SIGKILL after `grace` (docs/SUPERVISOR.md
/// lifecycle table). Returns whether escalation happened.
pub fn stop(session: &Session, grace: Duration) -> Result<Option<bool>> {
    session.require_exists()?;
    let Some(pid) = session.running() else {
        // Clean up a stale pid file so `status` stays truthful.
        let _ = std::fs::remove_file(session.pid_path());
        return Ok(None);
    };
    signal(pid, libc::SIGTERM)?;
    let deadline = Instant::now() + grace;
    while pid_alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    let escalated = pid_alive(pid);
    if escalated {
        signal(pid, libc::SIGKILL)?;
        while pid_alive(pid) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    let _ = std::fs::remove_file(session.pid_path());
    Ok(Some(escalated))
}

fn signal(pid: i32, sig: i32) -> Result<()> {
    if unsafe { libc::kill(pid, sig) } != 0 {
        let err = std::io::Error::last_os_error();
        // The process exiting between liveness check and kill is a win,
        // not an error.
        if err.raw_os_error() != Some(libc::ESRCH) {
            return Err(err).with_context(|| format!("signaling pid {pid}"));
        }
    }
    Ok(())
}

/// One session's status snapshot (`agentd status`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Status {
    pub name: String,
    pub running: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_checkpoint: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_ts: Option<String>,
    /// Pending human approvals for this session's run, surfaced from the
    /// agent's approvals directory (`agent approvals` resolves them).
    pub pending_approvals: usize,
}

pub fn status(session: &Session) -> Result<Status> {
    session.require_exists()?;
    let running = session.running();
    let run_id = session.run_id();
    let model = Spec::load(&session.spec_path())
        .ok()
        .and_then(|spec| spec.config().ok())
        .and_then(|config| config.model);
    let last_checkpoint = session
        .latest_checkpoint()
        .and_then(|path| std::fs::metadata(path).ok())
        .and_then(|meta| meta.modified().ok())
        .map(DateTime::<Utc>::from);
    let (last_event, last_event_ts) = crate::events::last_event_summary(&session.stdout_path())
        .map(|(name, ts)| (Some(name), ts))
        .unwrap_or((None, None));
    let pending_approvals = run_id
        .as_deref()
        .map(pending_approvals_for_run)
        .unwrap_or(0);
    Ok(Status {
        name: session.name.clone(),
        running: running.is_some(),
        pid: running,
        model,
        run_id,
        last_checkpoint,
        last_event,
        last_event_ts,
        pending_approvals,
    })
}

/// All session names under `$AGENTD_HOME`, sorted.
pub fn list_sessions(home: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    let entries = match std::fs::read_dir(home) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(names),
        Err(err) => return Err(err).with_context(|| format!("listing {}", home.display())),
    };
    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        if let Some(name) = entry.file_name().to_str() {
            if validate_name(name).is_ok() {
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

/// Count records awaiting approval for `run_id` in the agent's flat
/// approvals directory (`~/.local/share/agent/approvals`). Read directly:
/// the records are the same filesystem API `agent approvals` uses.
fn pending_approvals_for_run(run_id: &str) -> usize {
    let Some(home) = dirs::home_dir() else {
        return 0;
    };
    let dir = home.join(".local/share/agent/approvals");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".json") && !name.contains(".machine")
        })
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .filter(|record| {
            record.get("run_id").and_then(serde_json::Value::as_str) == Some(run_id)
                && record.get("status").and_then(serde_json::Value::as_str)
                    == Some("awaiting_approval")
        })
        .count()
}

/// The trace JSONL the agent writes for a run (`agentd logs` default view).
pub fn trace_path(run_id: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
    Ok(home
        .join(".local/share/agent/traces")
        .join(format!("{run_id}.jsonl")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_validation_rejects_traversal_and_flags() {
        for bad in ["", "..", "a/b", ".hidden", "-flag", "a b", "ü"] {
            assert!(validate_name(bad).is_err(), "{bad:?} should be rejected");
        }
        for good in ["coder", "gc-coder", "a.b_c-2"] {
            assert!(validate_name(good).is_ok(), "{good:?} should be accepted");
        }
    }

    #[test]
    fn dead_pid_is_not_running() {
        let dir = std::env::temp_dir().join(format!("agentd-sess-{}", uuid::Uuid::new_v4()));
        let session = Session::new(dir.clone(), "s").unwrap();
        std::fs::create_dir_all(session.dir()).unwrap();
        // A pid that certainly isn't ours and (on Linux) is beyond pid_max
        // defaults; if it happens to exist the assertion is skipped.
        std::fs::write(session.pid_path(), "999999999\n").unwrap();
        assert!(session.pid().is_some());
        if !pid_alive(999_999_999) {
            assert!(session.running().is_none());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn seed_round_trips_through_the_spec() -> Result<()> {
        let seed = SpecSeed {
            model: Some("sonnet".into()),
            provider: None,
            system_prompt: Some("Be terse.".into()),
            max_turns: Some(12),
        };
        let spec = seed.into_spec();
        let config = Spec::parse(&spec.render())?.config()?;
        assert_eq!(config.model.as_deref(), Some("sonnet"));
        assert_eq!(config.system_prompt.as_deref(), Some("Be terse."));
        assert_eq!(config.max_iterations, Some(12));
        Ok(())
    }
}
