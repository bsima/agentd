//! Persistent sessions: the SDK facade over `agent --session --json`.
//!
//! A [`Session`] runs the **agent binary as a child process** — deliberately
//! not in-process. Sessions are exactly the durable-Unix-process story this
//! runtime sells: the child owns its loop, trace, and checkpoints on disk in
//! the same layout the CLI and the upcoming `agentd` supervisor
//! (docs/SUPERVISOR.md) use, so a session started here is inspectable and
//! resumable by any other tool. An in-process session would fork those
//! semantics.
//!
//! ## The three flows
//!
//! - **One-shot** ([`crate::Runner::run`] / [`crate::Runner::start`]) — the
//!   agent loop runs in-process; native [`crate::Tool`]s, injected
//!   providers, and live [`crate::EventStream`]s all work.
//! - **Session** ([`Session::start`] / [`Session::resume`]) — the loop runs
//!   in a spawned `agent` child; turns are NUL-framed v1 turn envelopes on
//!   the child's stdin, correlated by `turn_id` on the `--json` machine
//!   events (never by order). Checkpoints land under the session home after
//!   every turn; `kill -9` loses nothing that was checkpointed.
//! - **Replay** ([`crate::Runner::replay`], or a session with
//!   [`SessionOptions::replay_trace`]) — deterministic re-execution from a
//!   recorded trace; no provider, no credentials.
//!
//! ## Limitations in this wave (child-process boundary)
//!
//! - **Native SDK tools are NOT available on sessions.** A [`crate::Tool`]
//!   handler is an in-process closure; it cannot cross into the child.
//!   [`Session::start`] returns [`SdkError::Unsupported`] for agents with
//!   registered tools until the supervisor/tool-host work (wave 3+) gives
//!   the child a way to call back into the SDK process. The child's
//!   built-in `shell`/`infer` (and `remember`/`recall` with
//!   [`crate::AgentBuilder::memory_dir`]) tools work as usual.
//! - **Injected providers** ([`crate::AgentBuilder::provider`]) cannot
//!   cross the process boundary either: the child resolves its provider
//!   from the model registry and environment, exactly like the CLI. (This
//!   also means OAuth-provider models, which the in-process Runner rejects,
//!   DO work on sessions.) For credential-free sessions use
//!   [`SessionOptions::replay_trace`].
//! - **Turn delivery is the child's stdin pipe**, so sends only work from
//!   the process that spawned the session. FIFO delivery — which makes a
//!   session addressable by any process and lets it outlive the SDK caller
//!   — is the supervisor-era upgrade (docs/SUPERVISOR.md). Note that
//!   dropping the [`Session`] closes the pipe, which the child treats as a
//!   graceful end-of-session (it checkpoints after every turn, so nothing
//!   is lost; [`Session::resume`] continues it).
//! - The session's runtime trace lives at the CLI's conventional path
//!   (`~/.local/share/agent/traces/<run_id>.jsonl`, honoring the child's
//!   `HOME`); [`crate::AgentBuilder::trace_dir`] does not apply. The real
//!   path is always available via [`Session::trace_path`].

use crate::agent::Agent;
use crate::error::SdkError;
use crate::runner::EventStream;
use agent_core::approval::{ApprovalDecision, ApprovalKind, ApprovalStore, PendingEffectRecord};
use agent_core::public_trace::public_event;
use agent_core::{EnvPolicy, Event, DEFAULT_MAX_REPAIRS};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};
use uuid::Uuid;

/// How long [`Session::start`] waits for the child's startup banner before
/// declaring the spawn failed.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Default grace period [`Session::stop`] allows for each escalation step
/// (stdin close -> SIGTERM -> SIGKILL, per docs/SUPERVISOR.md).
const DEFAULT_STOP_GRACE: Duration = Duration::from_secs(5);

/// Options for [`Session::start`] / [`Session::resume`].
#[derive(Clone, Debug, Default)]
pub struct SessionOptions {
    /// Session name; names the default session home directory
    /// (`~/.local/share/agentd/<name>`, the supervisor's layout).
    pub name: Option<String>,
    /// Session home directory (holds `checkpoints/`, `run-id`, `pid`, and a
    /// generated `output-schema.json` when the agent has one). Overrides
    /// the name-derived default. Created if missing.
    pub home: Option<PathBuf>,
    /// Path to the `agent` binary. Default resolution order:
    /// this option, `$AGENT_SDK_BIN`, a sibling of the current executable
    /// (`target/debug/agent` in dev builds), then `agent` on `PATH`.
    pub binary: Option<PathBuf>,
    /// Run the session in replay mode (`--replay-trace`): every recorded
    /// effect result is served from this trace, no provider is called, and
    /// no credentials are needed. Replay sessions do not write checkpoints.
    pub replay_trace: Option<PathBuf>,
    /// Extra environment variables for the child (e.g. `AGENT_PROVIDER`,
    /// `AGENT_API_KEY`, or `HOME` for hermetic tests). Applied last, so
    /// they win over the SDK's own scrubbing.
    pub env: Vec<(String, String)>,
    /// Grace period per stop-escalation step (default 5s).
    pub stop_grace: Option<Duration>,
}

/// Per-send options for [`Session::send_with`].
#[derive(Clone, Debug, Default)]
pub struct TurnOptions {
    /// Caller-supplied turn id (echoed on the turn's machine events). When
    /// `None` the SDK generates a fresh `sdk-<uuid>` id.
    pub turn_id: Option<String>,
    /// Opaque JSON echoed verbatim on the turn's `agent_complete` event and
    /// returned in [`TurnResult::metadata`].
    pub metadata: Option<Value>,
    /// Caller-side wait budget. On expiry the send returns
    /// [`SdkError::SendTimeout`] and the turn KEEPS RUNNING in the child;
    /// retrieve its result later with [`Session::attach`].
    pub timeout: Option<Duration>,
}

/// The outcome of one completed session turn.
#[derive(Clone, Debug)]
pub struct TurnResult {
    /// The turn's final response text.
    pub text: String,
    /// The turn id the completion correlated on (supplied, SDK-generated,
    /// or agent-minted for [`Session::send_unkeyed`]).
    pub turn_id: String,
    /// When the agent has an output schema: `text` parsed as JSON (already
    /// validated by the child). `None` when no schema is set.
    pub output: Option<Value>,
    /// The envelope metadata echoed back by the child, when supplied.
    pub metadata: Option<Value>,
}

/// A point-in-time snapshot of session health.
#[derive(Clone, Debug)]
pub struct SessionStatus {
    /// Whether the child process is running.
    pub alive: bool,
    /// The child's OS pid.
    pub pid: Option<u32>,
    /// The session's stable run id (shared across resumes).
    pub run_id: String,
    /// Timestamp of the most recent machine event seen on the child's
    /// stdout, if any.
    pub last_event_ts: Option<DateTime<Utc>>,
    /// The child's runtime trace JSONL.
    pub trace_path: PathBuf,
    /// The session home directory (checkpoints live in `checkpoints/`).
    pub session_dir: PathBuf,
}

/// What one turn resolved to, before SDK-error shaping.
#[derive(Clone, Debug)]
enum TurnOutcome {
    Complete {
        text: String,
        metadata: Option<Value>,
    },
    Error {
        message: String,
    },
}

/// Shared state between the Session handle and its stdout-router task.
#[derive(Default)]
struct RouterState {
    /// Waiters keyed by turn id; the router completes them.
    pending: HashMap<String, oneshot::Sender<TurnOutcome>>,
    /// Outcomes that arrived with no live waiter (send timeout, or a
    /// completion racing an [`Session::attach`] registration). Claimed by
    /// the next `attach` for that id.
    unclaimed: HashMap<String, TurnOutcome>,
    /// FIFO of [`Session::send_unkeyed`] callers waiting to learn the
    /// agent-minted turn id from the turn's `agent_start` event.
    unkeyed_watchers: VecDeque<oneshot::Sender<String>>,
    last_event_ts: Option<DateTime<Utc>>,
    /// Set when the child's stdout reaches EOF (the child exited).
    closed: bool,
}

struct Shared {
    state: Mutex<RouterState>,
    /// Ring buffer of the child's most recent stderr lines, for error
    /// context when the child dies.
    stderr_tail: Mutex<VecDeque<String>>,
}

impl Shared {
    fn stderr_context(&self) -> String {
        let tail = self.stderr_tail.lock().expect("stderr tail lock");
        if tail.is_empty() {
            String::new()
        } else {
            format!(
                "; recent child stderr:\n{}",
                tail.iter().cloned().collect::<Vec<_>>().join("\n")
            )
        }
    }
}

/// A persistent agent session backed by an `agent --session --json` child
/// process. See the module docs for the full story and limitations.
pub struct Session {
    agent: Agent,
    session_dir: PathBuf,
    run_id: String,
    trace_path: PathBuf,
    pid: Option<u32>,
    stop_grace: Duration,
    child: AsyncMutex<Child>,
    /// `None` once [`Session::stop`] has closed it.
    stdin: AsyncMutex<Option<ChildStdin>>,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("run_id", &self.run_id)
            .field("session_dir", &self.session_dir)
            .field("trace_path", &self.trace_path)
            .field("pid", &self.pid)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Spawn a fresh session for `agent`. The child reads NUL-framed turn
    /// envelopes from its stdin and reports completions as `--json` machine
    /// events; checkpoints are written under the session home after every
    /// turn.
    pub async fn start(agent: &Agent, options: SessionOptions) -> Result<Session, SdkError> {
        launch(agent, options, LaunchMode::Start).await
    }

    /// Restart a session from its latest checkpoint
    /// (`<home>/checkpoints/session-latest.json`, via the binary's
    /// `--resume`). The run id, conversation history, and turn ordinals all
    /// continue: turn ids minted by the agent after a resume pick up at the
    /// checkpointed sequence instead of restarting at `t0`, and a
    /// checkpoint left dangling by a mid-tool-call crash is repaired on
    /// load (trailing unexecuted tool calls dropped) before the first turn.
    pub async fn resume(agent: &Agent, options: SessionOptions) -> Result<Session, SdkError> {
        launch(agent, options, LaunchMode::Resume).await
    }

    /// The session's stable run id (constant across resumes).
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The child's runtime trace JSONL (as reported by the child itself).
    pub fn trace_path(&self) -> &Path {
        &self.trace_path
    }

    /// The session home directory.
    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }

    /// Send one turn and wait for its result. Equivalent to
    /// [`Session::send_with`] with default [`TurnOptions`]: a fresh
    /// SDK-generated turn id and no caller timeout.
    pub async fn send(&self, prompt: impl Into<String>) -> Result<TurnResult, SdkError> {
        self.send_with(prompt, TurnOptions::default()).await
    }

    /// Send one turn as a v1 turn envelope and wait for the machine event
    /// carrying the SAME turn id (`agent_complete` -> `Ok`, `agent_error`
    /// -> [`SdkError::Turn`]). Correlation is strictly by id, never by
    /// response order, so overlapping sends resolve correctly.
    ///
    /// With [`TurnOptions::timeout`] set, expiry returns
    /// [`SdkError::SendTimeout`] while the turn keeps running in the child;
    /// [`Session::attach`] with the same id retrieves the eventual result,
    /// and later sends are unaffected.
    pub async fn send_with(
        &self,
        prompt: impl Into<String>,
        options: TurnOptions,
    ) -> Result<TurnResult, SdkError> {
        let turn_id = options
            .turn_id
            .unwrap_or_else(|| format!("sdk-{}", Uuid::new_v4()));
        // Register before writing the frame so a fast completion can never
        // race past its waiter.
        let waiter = match self.claim_or_register(&turn_id)? {
            Claim::Ready(outcome) => return self.shape_outcome(&turn_id, outcome),
            Claim::Wait(rx) => rx,
        };
        let mut frame = serde_json::json!({
            "v": 1,
            "turn_id": &turn_id,
            "input": prompt.into(),
        });
        if let Some(metadata) = options.metadata {
            frame["metadata"] = metadata;
        }
        self.write_frame(&frame).await?;
        let outcome = match options.timeout {
            None => self.await_outcome(&turn_id, waiter).await?,
            Some(budget) => match tokio::time::timeout(budget, waiter).await {
                Ok(received) => self.recv_outcome(&turn_id, received)?,
                Err(_elapsed) => {
                    // The pending entry stays registered: when the turn
                    // eventually completes, the router parks the outcome in
                    // `unclaimed` for a later attach.
                    return Err(SdkError::SendTimeout {
                        still_running: self.alive().await,
                        turn_id,
                    });
                }
            },
        };
        self.shape_outcome(&turn_id, outcome)
    }

    /// Wait for the result of a turn by id — typically one whose
    /// [`Session::send_with`] timed out. Works whether the turn is still
    /// running or already finished (finished-but-unclaimed outcomes are
    /// parked until attached). Fails with [`SdkError::Session`] if the
    /// child exits before the turn completes.
    pub async fn attach(&self, turn_id: &str) -> Result<TurnResult, SdkError> {
        let outcome = match self.claim_or_register(turn_id)? {
            Claim::Ready(outcome) => outcome,
            Claim::Wait(rx) => self.await_outcome(turn_id, rx).await?,
        };
        self.shape_outcome(turn_id, outcome)
    }

    /// Send one turn as an envelope WITHOUT a turn id, letting the agent
    /// mint its deterministic `<run_id>-t<seq>` id (`seq` continues across
    /// resumes from the checkpoint sequence). The minted id is learned from
    /// the turn's `agent_start` event and returned on the result.
    ///
    /// This is the supervisor-parity path; prefer [`Session::send`], which
    /// correlates without depending on frame order. Unkeyed sends are
    /// matched to `agent_start` events FIFO, so do not overlap several of
    /// them.
    pub async fn send_unkeyed(&self, prompt: impl Into<String>) -> Result<TurnResult, SdkError> {
        let learn = {
            let mut state = self.lock_state();
            if state.closed {
                return Err(self.closed_error());
            }
            let (tx, rx) = oneshot::channel();
            state.unkeyed_watchers.push_back(tx);
            rx
        };
        let frame = serde_json::json!({ "v": 1, "input": prompt.into() });
        self.write_frame(&frame).await?;
        let turn_id = learn.await.map_err(|_| self.closed_error())?;
        self.attach(&turn_id).await
    }

    /// The next unresolved approval pause of this session, oldest first
    /// (t-1308.10, DR-7), by polling the run's approvals directory — the
    /// same on-disk records `agent approvals --list` reads, derived from
    /// the child's trace path (`<data-dir>/approvals`, a sibling of
    /// `traces/`). Returns `None` when nothing is awaiting approval.
    ///
    /// Wave-1 scope: a session turn that hits an approval gate FAILS with
    /// [`SdkError::Turn`] (the child reports the pause on its `agent_error`
    /// machine event, naming the pending id) after durably persisting the
    /// pause. [`PendingApproval::approve`] / [`PendingApproval::deny`]
    /// record the decision with the same shared agent-core resolution
    /// functions the CLI uses; re-entering the paused turn's checkpoint is
    /// then `agent approvals --approve/--deny <pending_id>` (which finds
    /// the decision already recorded and resumes), and its outcome lands in
    /// the run's trace — not in this session's in-child history.
    pub async fn next_approval(&self) -> Result<Option<PendingApproval>, SdkError> {
        let Some(dir) = self.approvals_dir() else {
            return Ok(None);
        };
        let store = ApprovalStore::new(dir);
        let records = store
            .list()
            .await
            .map_err(|err| SdkError::Session(format!("listing pending approvals: {err:#}")))?;
        Ok(records
            .into_iter()
            .find(|record| record.run_id == self.run_id && record.is_awaiting())
            .map(|record| PendingApproval { record, store }))
    }

    /// The approvals directory serving this session's run: a sibling of the
    /// trace directory reported by the child (`.../agent/traces/<run>.jsonl`
    /// -> `.../agent/approvals`), so it honors the child's `HOME`.
    fn approvals_dir(&self) -> Option<PathBuf> {
        Some(self.trace_path.parent()?.parent()?.join("approvals"))
    }

    /// A live [`EventStream`] of the session's public trace events
    /// (docs/TRACE_SCHEMA.md), fed by tailing the child's trace file from
    /// the beginning. Each call returns an independent stream; the stream
    /// ends after the child has exited and the tail is drained. Because the
    /// trace file persists (and a resume appends to it under the same run
    /// id), a stream opened after resume replays the whole session history.
    pub fn events(&self) -> EventStream {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(tail_trace(self.trace_path.clone(), self.shared.clone(), tx));
        EventStream::new(rx)
    }

    /// Point-in-time session health.
    pub async fn status(&self) -> SessionStatus {
        let alive = self.alive().await;
        let last_event_ts = self.lock_state().last_event_ts;
        SessionStatus {
            alive,
            pid: self.pid,
            run_id: self.run_id.clone(),
            last_event_ts,
            trace_path: self.trace_path.clone(),
            session_dir: self.session_dir.clone(),
        }
    }

    /// Graceful shutdown, escalating per docs/SUPERVISOR.md: close stdin
    /// (the session loop's clean EOF path — the child finishes the current
    /// turn, checkpoints, and exits), then SIGTERM, then SIGKILL, each
    /// after the configured grace period. Returns the child's exit status.
    pub async fn stop(&self) -> Result<std::process::ExitStatus, SdkError> {
        // Closing stdin is the graceful path; the child exits its
        // NUL-frame loop at EOF after finishing in-flight turns.
        {
            let mut stdin = self.stdin.lock().await;
            *stdin = None;
        }
        let mut child = self.child.lock().await;
        if let Ok(status) = tokio::time::timeout(self.stop_grace, child.wait()).await {
            return status.map_err(|err| SdkError::Session(format!("waiting for child: {err}")));
        }
        // Escalate: SIGTERM (the child's signal handler also exits the
        // session loop cleanly), then SIGKILL.
        #[cfg(unix)]
        if let Some(pid) = self.pid {
            let _ = std::process::Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status();
            if let Ok(status) = tokio::time::timeout(self.stop_grace, child.wait()).await {
                return status
                    .map_err(|err| SdkError::Session(format!("waiting for child: {err}")));
            }
        }
        child
            .kill()
            .await
            .map_err(|err| SdkError::Session(format!("killing child: {err}")))?;
        child
            .wait()
            .await
            .map_err(|err| SdkError::Session(format!("waiting for killed child: {err}")))
    }

    /// Hard-kill the child (SIGKILL, no grace, no checkpoint flush beyond
    /// what already landed). The checkpoint model makes this safe to
    /// recover from: [`Session::resume`] restarts from the last completed
    /// turn.
    pub async fn kill(&self) -> Result<(), SdkError> {
        let mut child = self.child.lock().await;
        child
            .kill()
            .await
            .map_err(|err| SdkError::Session(format!("killing child: {err}")))
    }

    async fn alive(&self) -> bool {
        let mut child = self.child.lock().await;
        matches!(child.try_wait(), Ok(None))
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, RouterState> {
        self.shared.state.lock().expect("session router state lock")
    }

    fn closed_error(&self) -> SdkError {
        SdkError::Session(format!(
            "session child exited{}",
            self.shared.stderr_context()
        ))
    }

    /// Claim an already-parked outcome for `turn_id`, or register a waiter.
    fn claim_or_register(&self, turn_id: &str) -> Result<Claim, SdkError> {
        let mut state = self.lock_state();
        if let Some(outcome) = state.unclaimed.remove(turn_id) {
            return Ok(Claim::Ready(outcome));
        }
        if state.closed {
            return Err(self.closed_error());
        }
        let (tx, rx) = oneshot::channel();
        state.pending.insert(turn_id.to_owned(), tx);
        Ok(Claim::Wait(rx))
    }

    async fn await_outcome(
        &self,
        turn_id: &str,
        rx: oneshot::Receiver<TurnOutcome>,
    ) -> Result<TurnOutcome, SdkError> {
        let received = rx.await;
        self.recv_outcome(turn_id, received)
    }

    fn recv_outcome(
        &self,
        turn_id: &str,
        received: Result<TurnOutcome, oneshot::error::RecvError>,
    ) -> Result<TurnOutcome, SdkError> {
        received.map_err(|_| {
            SdkError::Session(format!(
                "session child exited before turn {turn_id} completed{}",
                self.shared.stderr_context()
            ))
        })
    }

    fn shape_outcome(&self, turn_id: &str, outcome: TurnOutcome) -> Result<TurnResult, SdkError> {
        match outcome {
            TurnOutcome::Complete { text, metadata } => {
                let output = match &self.agent.output_contract {
                    Some(_) => Some(serde_json::from_str(&text).map_err(|err| {
                        SdkError::Run(format!("validated output failed to parse as JSON: {err}"))
                    })?),
                    None => None,
                };
                Ok(TurnResult {
                    text,
                    turn_id: turn_id.to_owned(),
                    output,
                    metadata,
                })
            }
            TurnOutcome::Error { message } => Err(SdkError::Turn {
                turn_id: turn_id.to_owned(),
                message,
            }),
        }
    }

    async fn write_frame(&self, frame: &Value) -> Result<(), SdkError> {
        let mut bytes = serde_json::to_vec(frame)
            .map_err(|err| SdkError::Session(format!("encoding turn envelope: {err}")))?;
        bytes.push(0);
        let mut guard = self.stdin.lock().await;
        let stdin = guard.as_mut().ok_or_else(|| {
            SdkError::Session("session is stopped (stdin closed); no further turns".into())
        })?;
        stdin
            .write_all(&bytes)
            .await
            .map_err(|err| SdkError::Session(format!("writing turn to session stdin: {err}")))?;
        stdin
            .flush()
            .await
            .map_err(|err| SdkError::Session(format!("flushing session stdin: {err}")))
    }
}

enum Claim {
    Ready(TurnOutcome),
    Wait(oneshot::Receiver<TurnOutcome>),
}

/// One unresolved approval pause, discovered by [`Session::next_approval`].
/// Deciding it calls the same durable agent-core resolution the `agent
/// approvals` CLI uses; the decision is made exactly once (a second
/// resolution of the same pending id fails), and neither method re-enters
/// the paused machine — see [`Session::next_approval`] for the wave-1
/// resume story.
#[derive(Debug)]
pub struct PendingApproval {
    record: PendingEffectRecord,
    store: ApprovalStore,
}

impl PendingApproval {
    pub fn pending_id(&self) -> &str {
        &self.record.pending_id
    }

    /// What kind of effect is gated (`Eval` or `Store`).
    pub fn kind(&self) -> ApprovalKind {
        self.record.kind
    }

    /// The gated request payload preview: for Eval `{command, argv}`, for
    /// Store `{sink, op, id, item_preview, content_hash}`.
    pub fn request(&self) -> &Value {
        &self.record.request
    }

    /// The full on-disk pending record.
    pub fn record(&self) -> &PendingEffectRecord {
        &self.record
    }

    /// Durably approve the gated effect: when its checkpoint is re-entered
    /// the effect executes exactly once.
    pub async fn approve(self) -> Result<(), SdkError> {
        self.resolve(ApprovalDecision::Approve, None).await
    }

    /// Durably deny the gated effect: when its checkpoint is re-entered the
    /// effect binds a typed denial value (errors-as-values) and the program
    /// continues.
    pub async fn deny(self, reason: Option<String>) -> Result<(), SdkError> {
        self.resolve(ApprovalDecision::Deny, reason).await
    }

    async fn resolve(
        self,
        decision: ApprovalDecision,
        reason: Option<String>,
    ) -> Result<(), SdkError> {
        self.store
            .resolve(
                &self.record.pending_id,
                decision,
                Some("sdk".into()),
                reason,
            )
            .await
            .map(|_| ())
            .map_err(|err| SdkError::Session(format!("resolving pending approval: {err:#}")))
    }
}

enum LaunchMode {
    Start,
    Resume,
}

/// Startup facts the child prints on stderr before entering its loop.
struct Banner {
    run_id: String,
    trace_path: PathBuf,
}

async fn launch(
    agent: &Agent,
    options: SessionOptions,
    mode: LaunchMode,
) -> Result<Session, SdkError> {
    reject_unsupported(agent)?;
    let session_dir = resolve_session_dir(&options)?;
    let checkpoint_dir = session_dir.join("checkpoints");
    tokio::fs::create_dir_all(&checkpoint_dir)
        .await
        .map_err(|err| {
            SdkError::Session(format!(
                "creating session home {}: {err}",
                session_dir.display()
            ))
        })?;

    let binary = resolve_binary(&options);
    let mut cmd = Command::new(&binary);
    cmd.arg("--session")
        .arg("--json")
        .arg("--model")
        .arg(&agent.model)
        .arg("--checkpoint-dir")
        .arg(&checkpoint_dir)
        .arg("--max-turns")
        .arg(agent.max_turns.to_string())
        .arg("--eval-timeout-seconds")
        .arg(agent.eval_timeout.as_secs().max(1).to_string());
    if agent.require_shell_approval {
        cmd.arg("--require-shell-approval");
    }
    match mode {
        LaunchMode::Start => {
            cmd.arg("--run-id").arg(Uuid::new_v4().to_string());
        }
        LaunchMode::Resume => {
            let checkpoint = checkpoint_dir.join("session-latest.json");
            if !checkpoint.is_file() {
                return Err(SdkError::Session(format!(
                    "no checkpoint to resume from at {}",
                    checkpoint.display()
                )));
            }
            cmd.arg("--resume").arg(&checkpoint);
        }
    }
    if let Some(instructions) = &agent.instructions {
        cmd.arg("--system-prompt").arg(instructions);
    }
    if let Some(dir) = &agent.memory_dir {
        cmd.arg("--memory-dir").arg(dir);
    }
    if let Some(cwd) = &agent.eval_cwd {
        cmd.arg("--eval-cwd").arg(cwd);
    }
    match &agent.eval_env {
        EnvPolicy::Inherit => {}
        EnvPolicy::InheritFull => {
            cmd.arg("--eval-env").arg("inherit-full");
        }
        EnvPolicy::Clean { vars } if vars.is_empty() => {
            cmd.arg("--eval-env").arg("clean");
        }
        other => {
            return Err(SdkError::Unsupported(format!(
                "eval env policy {other:?} cannot be expressed as agent CLI flags; \
                 sessions support Inherit, InheritFull, and empty Clean"
            )));
        }
    }
    if let Some(contract) = &agent.output_contract {
        if contract.max_repairs != DEFAULT_MAX_REPAIRS {
            return Err(SdkError::Unsupported(format!(
                "output contract max_repairs={} cannot be expressed as agent CLI flags \
                 (the binary uses the default of {DEFAULT_MAX_REPAIRS})",
                contract.max_repairs
            )));
        }
        let schema_path = session_dir.join("output-schema.json");
        let bytes = serde_json::to_vec_pretty(&contract.schema)
            .map_err(|err| SdkError::Session(format!("encoding output schema: {err}")))?;
        tokio::fs::write(&schema_path, bytes).await.map_err(|err| {
            SdkError::Session(format!(
                "writing output schema {}: {err}",
                schema_path.display()
            ))
        })?;
        cmd.arg("--output-schema").arg(&schema_path);
    }
    if let Some(trace) = &options.replay_trace {
        cmd.arg("--replay-trace").arg(trace);
    }
    // Scrub the binary's own env-linked knobs so ambient state on the SDK
    // host cannot silently change the session's mode; explicit options.env
    // entries are applied after and win.
    for var in [
        "AGENT_FIFO",
        "AGENT_RUN_ID",
        "AGENT_RESUME",
        "AGENT_REPLAY_TRACE",
        "AGENT_CHECKPOINT_DIR",
        "AGENT_OUTPUT_SCHEMA",
        "AGENT_SYSTEM_PROMPT",
        "AGENT_MAX_TURNS",
        "AGENT_MODEL",
        "AGENT_MEMORY_DIR",
        "AGENT_HYDRATION_DIR",
        "AGENT_TEMPORAL_DIR",
        "AGENT_REQUIRE_SHELL_APPROVAL",
    ] {
        cmd.env_remove(var);
    }
    cmd.envs(options.env.iter().cloned());
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|err| {
        SdkError::Session(format!(
            "spawning agent binary {}: {err}; set SessionOptions.binary or $AGENT_SDK_BIN, \
             or put `agent` on PATH",
            binary.display()
        ))
    })?;
    let pid = child.id();
    let stdin = child.stdin.take();
    let stdout = child.stdout.take().expect("child stdout piped");
    let stderr = child.stderr.take().expect("child stderr piped");

    let shared = Arc::new(Shared {
        state: Mutex::new(RouterState::default()),
        stderr_tail: Mutex::new(VecDeque::new()),
    });
    let (banner_tx, banner_rx) = oneshot::channel();
    tokio::spawn(drain_stderr(stderr, shared.clone(), banner_tx));
    tokio::spawn(route_stdout(stdout, shared.clone()));

    let banner = match tokio::time::timeout(STARTUP_TIMEOUT, banner_rx).await {
        Ok(Ok(banner)) => banner,
        Ok(Err(_)) | Err(_) => {
            // No banner: the child died during startup (bad flags, missing
            // credentials, unreadable checkpoint) or wedged. Reap and report.
            let _ = child.kill().await;
            let status = child.wait().await.ok();
            return Err(SdkError::Session(format!(
                "agent child failed to start (exit: {}){}",
                status.map_or_else(|| "unknown".into(), |st| st.to_string()),
                shared.stderr_context()
            )));
        }
    };

    // Supervisor-layout breadcrumbs (best-effort; the session works without
    // them, but `agentd status`-style tooling reads them).
    let _ = tokio::fs::write(session_dir.join("run-id"), &banner.run_id).await;
    if let Some(pid) = pid {
        let _ = tokio::fs::write(session_dir.join("pid"), pid.to_string()).await;
    }

    Ok(Session {
        agent: agent.clone(),
        session_dir,
        run_id: banner.run_id,
        trace_path: banner.trace_path,
        pid,
        stop_grace: options.stop_grace.unwrap_or(DEFAULT_STOP_GRACE),
        child: AsyncMutex::new(child),
        stdin: AsyncMutex::new(stdin),
        shared,
    })
}

fn reject_unsupported(agent: &Agent) -> Result<(), SdkError> {
    let tools = agent.tools.names();
    if !tools.is_empty() {
        return Err(SdkError::Unsupported(format!(
            "native SDK tools ({}) are not available on child-process sessions: the handler \
             lives in this process and the agent loop runs in the spawned `agent` binary. \
             This needs the supervisor/tool-host work (wave 3+). Use Runner::run for native \
             tools, or the child's built-in shell/infer/memory tools in sessions",
            tools.join(", ")
        )));
    }
    if agent.on_approval.is_some() {
        return Err(SdkError::Unsupported(
            "on_approval hooks are in-process closures and cannot cross the session's process \
             boundary; session pauses persist durably instead — poll Session::next_approval and \
             resolve with PendingApproval::approve()/deny() (resume via `agent approvals`)"
                .into(),
        ));
    }
    if agent.provider.is_some() {
        return Err(SdkError::Unsupported(
            "injected providers cannot cross the session's process boundary; the child \
             resolves its provider from the model registry and environment (models.yaml, \
             AGENT_PROVIDER/AGENT_API_KEY). For credential-free sessions use \
             SessionOptions.replay_trace"
                .into(),
        ));
    }
    Ok(())
}

fn resolve_session_dir(options: &SessionOptions) -> Result<PathBuf, SdkError> {
    if let Some(home) = &options.home {
        return Ok(home.clone());
    }
    let base = dirs::home_dir()
        .map(|home| home.join(".local/share/agentd"))
        .ok_or_else(|| {
            SdkError::Session("could not determine a home directory for the session".into())
        })?;
    let name = options
        .name
        .clone()
        .unwrap_or_else(|| format!("sdk-{}", Uuid::new_v4()));
    Ok(base.join(name))
}

fn resolve_binary(options: &SessionOptions) -> PathBuf {
    if let Some(binary) = &options.binary {
        return binary.clone();
    }
    if let Some(binary) = std::env::var_os("AGENT_SDK_BIN") {
        return PathBuf::from(binary);
    }
    // Dev convenience: a workspace build puts `agent` next to (or one level
    // above) the current executable — target/debug/{agent,examples/,deps/}.
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        for _ in 0..2 {
            let Some(current) = dir else { break };
            let candidate = current.join(format!("agent{}", std::env::consts::EXE_SUFFIX));
            if candidate.is_file() {
                return candidate;
            }
            dir = current.parent();
        }
    }
    PathBuf::from("agent")
}

/// Read the child's stderr: parse the startup banner (`trace:`/`run_id:`
/// lines, printed before the session loop starts), then keep draining into
/// the shared tail ring so the pipe never blocks the child and errors have
/// context.
async fn drain_stderr(
    stderr: tokio::process::ChildStderr,
    shared: Arc<Shared>,
    banner_tx: oneshot::Sender<Banner>,
) {
    let mut lines = BufReader::new(stderr).lines();
    let mut banner_tx = Some(banner_tx);
    let mut run_id: Option<String> = None;
    let mut trace_path: Option<PathBuf> = None;
    while let Ok(Some(line)) = lines.next_line().await {
        if banner_tx.is_some() {
            if let Some(value) = line.strip_prefix("run_id: ") {
                run_id = Some(value.trim().to_owned());
            } else if let Some(value) = line.strip_prefix("trace: ") {
                trace_path = Some(PathBuf::from(value.trim()));
            }
            if let (Some(id), Some(path)) = (&run_id, &trace_path) {
                let banner = Banner {
                    run_id: id.clone(),
                    trace_path: path.clone(),
                };
                if let Some(tx) = banner_tx.take() {
                    let _ = tx.send(banner);
                }
            }
        }
        let mut tail = shared.stderr_tail.lock().expect("stderr tail lock");
        if tail.len() >= 50 {
            tail.pop_front();
        }
        tail.push_back(line);
    }
    // EOF without a banner drops banner_tx, failing the startup wait.
}

/// Route the child's `--json` stdout: machine events (`agent_start`,
/// `agent_complete`, `agent_error`) resolve waiters by turn id; mirrored
/// trace lines only refresh the last-event timestamp.
async fn route_stdout(stdout: tokio::process::ChildStdout, shared: Arc<Shared>) {
    let mut lines = BufReader::new(stdout).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let mut state = shared.state.lock().expect("session router state lock");
        state.last_event_ts = Some(Utc::now());
        let Some(custom_type) = value.get("custom_type").and_then(Value::as_str) else {
            continue;
        };
        let turn_id = value
            .get("data")
            .and_then(|data| data.get("turn_id"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let Some(turn_id) = turn_id else { continue };
        match custom_type {
            "agent_start" => {
                // Keyed sends registered their waiter before writing the
                // frame, so an id we do not know belongs to an unkeyed
                // (agent-minted) send; hand the learned id to the next
                // unkeyed watcher, FIFO (frames are processed in order).
                if !state.pending.contains_key(&turn_id) {
                    if let Some(watcher) = state.unkeyed_watchers.pop_front() {
                        let _ = watcher.send(turn_id);
                    }
                }
            }
            "agent_complete" => {
                let outcome = TurnOutcome::Complete {
                    text: value
                        .get("data")
                        .and_then(|data| data.get("response"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    metadata: value
                        .get("data")
                        .and_then(|data| data.get("metadata"))
                        .cloned(),
                };
                deliver(&mut state, turn_id, outcome);
            }
            "agent_error" => {
                let outcome = TurnOutcome::Error {
                    message: value
                        .get("data")
                        .and_then(|data| data.get("message"))
                        .and_then(Value::as_str)
                        .unwrap_or("unknown turn error")
                        .to_owned(),
                };
                deliver(&mut state, turn_id, outcome);
            }
            _ => {}
        }
    }
    // Child stdout EOF: the process exited. Fail live waiters (dropping the
    // senders wakes them with RecvError) and refuse new registrations.
    let mut state = shared.state.lock().expect("session router state lock");
    state.closed = true;
    state.pending.clear();
    state.unkeyed_watchers.clear();
}

/// Hand `outcome` to the registered waiter, or park it for a later attach
/// (the waiter may have timed out and dropped its receiver).
fn deliver(state: &mut RouterState, turn_id: String, outcome: TurnOutcome) {
    if let Some(waiter) = state.pending.remove(&turn_id) {
        if let Err(outcome) = waiter.send(outcome) {
            state.unclaimed.insert(turn_id, outcome);
        }
    } else {
        state.unclaimed.insert(turn_id, outcome);
    }
}

/// Tail the session's trace JSONL, projecting each runtime event through
/// the public schema (the same projection [`crate::Runner::start`] uses)
/// into `tx`. Ends when the consumer drops the stream, or when the child
/// has exited and the file is drained.
async fn tail_trace(
    path: PathBuf,
    shared: Arc<Shared>,
    tx: mpsc::UnboundedSender<agent_core::public_trace::PublicEvent>,
) {
    let mut offset: u64 = 0;
    let mut partial: Vec<u8> = Vec::new();
    loop {
        let closed = shared
            .state
            .lock()
            .expect("session router state lock")
            .closed;
        let mut new_bytes = Vec::new();
        if let Ok(mut file) = tokio::fs::File::open(&path).await {
            if file.seek(std::io::SeekFrom::Start(offset)).await.is_ok() {
                let _ = file.read_to_end(&mut new_bytes).await;
            }
        }
        offset += new_bytes.len() as u64;
        partial.extend_from_slice(&new_bytes);
        while let Some(newline) = partial.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = partial.drain(..=newline).collect();
            let Ok(text) = std::str::from_utf8(&line) else {
                continue;
            };
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Skip lines that do not parse as runtime events (forward
            // compatibility with future event kinds) rather than dying.
            let Ok(event) = serde_json::from_str::<Event>(trimmed) else {
                continue;
            };
            if let Some(public) = public_event(&event) {
                if tx.send(public).is_err() {
                    return; // consumer dropped the stream
                }
            }
        }
        if closed && new_bytes.is_empty() {
            return; // child gone and file drained: end of stream
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
