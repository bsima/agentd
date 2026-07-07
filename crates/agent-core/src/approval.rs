//! The approval/pause protocol (t-1308.10, PRD DR-7): durable
//! human-in-the-loop gates over IR effects.
//!
//! An approval-gated effect (an `Instr::Eval` whose policy sets
//! `require_approval`, or an `Instr::Store` targeting a sink whose write
//! policy is [`crate::hydration::SinkWritePolicy::RequireApproval`]) does
//! not execute until someone says so. When the interpreter reaches one with
//! no decision available it **fails closed**: the effect is not executed,
//! the machine checkpoints mid-turn (program counter still pointing at the
//! gated instruction), and the run suspends as
//! [`crate::ir_interpreter::IrStepOutcome::AwaitingApproval`]. The driver
//! persists the pause as a [`PendingEffectRecord`] plus the machine
//! checkpoint under an approvals directory ([`ApprovalStore`]), so
//! resolution survives full process restarts — the filesystem is the API.
//!
//! Decisions arrive one of three ways, all funneling through the same gate
//! in the interpreter:
//!
//! 1. **In-process hook** ([`ApprovalConfig::hook`], the SDK Runner's
//!    `on_approval`): a synchronous callback decides at the effect site; no
//!    pause happens. No hook and no resolution means pause (or, for drivers
//!    that cannot pause, a closed failure) — never auto-approval.
//! 2. **Pre-resolved** ([`ApprovalConfig::resolutions`]): a resume driver
//!    (the `agent approvals` CLI) loads a resolved record, maps the effect
//!    id to its decision, and re-enters the checkpointed machine; the gate
//!    consumes the decision, emits `ApprovalResolved`, and either executes
//!    the effect or binds the typed denial value.
//! 3. **Replay**: recorded `ApprovalRequested`/`ApprovalResolved` trace
//!    events reproduce the pause or the decision as data — replay never
//!    prompts, never pauses a resolved recording, and diverges loudly when
//!    the recorded decision's effect identity does not match.
//!
//! Denial is not an abort: the gated effect's `out` binds the typed
//! [`denial_value`] (the errors-as-values convention, t-1222) and the
//! program continues, so the model can read the denial and react.

use crate::ir::EffectLocation;
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What can be gated. Extending the gate to another effect kind (Infer is
/// the anticipated next) means adding a variant here and calling the same
/// interpreter gate at its site — no protocol redesign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalKind {
    Eval,
    Store,
}

impl ApprovalKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eval => "eval",
            Self::Store => "store",
        }
    }
}

/// A resolver's verdict on one gated effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approve,
    Deny,
}

impl ApprovalDecision {
    /// The wire/status string recorded on `ApprovalResolved` events and in
    /// pending records: `"approved"` / `"denied"`.
    pub fn as_status_str(&self) -> &'static str {
        match self {
            Self::Approve => "approved",
            Self::Deny => "denied",
        }
    }
}

/// Lifecycle of a [`PendingEffectRecord`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingStatus {
    AwaitingApproval,
    Approved,
    Denied,
}

/// A decision plus resolver metadata, keyed by effect id in
/// [`ApprovalConfig::resolutions`] for resume drivers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalResolution {
    pub decision: ApprovalDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// What the interpreter hands the driver (and the in-process hook) when a
/// gated effect is reached: enough to identify the effect and preview the
/// request. Re-execution state is the machine checkpoint, not this value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Deterministic pause id: `pa-` + 12 hex of sha256(run_id, effect_id).
    /// Stable across a crash-and-repause of the same effect in the same
    /// run; distinct across runs and across visits (the effect id encodes
    /// the visit ordinal and control path).
    pub pending_id: String,
    pub effect: EffectLocation,
    pub kind: ApprovalKind,
    /// The effect's request payload preview: for Eval the display command
    /// and (when direct-exec) argv; for Store the sink, op, id, item
    /// preview, and content hash.
    pub request: Value,
}

/// Synchronous in-process approval policy hook (SDK Runner `on_approval`).
/// Absent hook + absent resolution = the effect does not execute.
pub type ApprovalHookFn = Arc<dyn Fn(&ApprovalRequest) -> ApprovalDecision + Send + Sync>;

/// Approval policy carried on [`crate::interpreter::SeqConfig`].
#[derive(Clone, Default)]
pub struct ApprovalConfig {
    /// Durable decisions keyed by effect id, loaded by resume drivers from
    /// resolved [`PendingEffectRecord`]s.
    pub resolutions: BTreeMap<String, ApprovalResolution>,
    /// In-process decision hook; consulted only when no resolution is
    /// pre-loaded for the effect.
    pub hook: Option<ApprovalHookFn>,
}

impl std::fmt::Debug for ApprovalConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApprovalConfig")
            .field("resolutions", &self.resolutions)
            .field("hook", &self.hook.as_ref().map(|_| "Fn"))
            .finish()
    }
}

/// The durable pause record — one JSON file per gated effect under the
/// approvals directory (see [`ApprovalStore`]). This file shape is consumed
/// by `agent approvals --list/--approve/--deny`, the SDK's
/// `Session::next_approval`, and (eventually) the dashboard; treat it as
/// additive-only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingEffectRecord {
    pub pending_id: String,
    pub run_id: String,
    /// The session turn that paused, when the driver knows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    pub effect_id: String,
    pub program_hash: String,
    pub kind: ApprovalKind,
    /// Full effect payload preview (see [`ApprovalRequest::request`]);
    /// re-execution state lives in the sibling machine checkpoint.
    pub request: Value,
    pub created_ts: DateTime<Utc>,
    pub status: PendingStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_ts: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Driver-owned facts for rebuilding the runtime on resume (the CLI
    /// records model, trace path, checkpoint dir, eval policy, ...). Opaque
    /// to agent-core.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<Value>,
}

impl PendingEffectRecord {
    pub fn is_awaiting(&self) -> bool {
        self.status == PendingStatus::AwaitingApproval
    }
}

/// Deterministic pending id for a gated effect in a run. Stable so a
/// crash-and-repause overwrites its own record instead of accumulating
/// duplicates; run-scoped so identical one-shot runs do not collide in a
/// shared approvals directory.
pub fn pending_id_for(run_id: &str, effect_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(run_id.as_bytes());
    hasher.update([0]);
    hasher.update(effect_id.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    format!("pa-{}", &digest[..12])
}

/// The typed denial value bound to a denied effect's `out` — the
/// errors-as-values convention (t-1222) extended with a machine-checkable
/// `approval` envelope the program (and the model, when the value surfaces
/// as a tool result) can react to.
pub fn denial_value(pending_id: &str, resolved_by: Option<&str>, reason: Option<&str>) -> Value {
    let mut error = String::from("approval denied");
    if let Some(reason) = reason {
        error.push_str(": ");
        error.push_str(reason);
    }
    serde_json::json!({
        "ok": false,
        "error": error,
        "approval": {
            "pending_id": pending_id,
            "status": "denied",
            "resolved_by": resolved_by,
            "reason": reason,
        }
    })
}

/// True when `value` is a typed approval denial produced by
/// [`denial_value`].
pub fn is_denial_value(value: &Value) -> bool {
    value
        .get("approval")
        .and_then(|approval| approval.get("status"))
        .and_then(Value::as_str)
        == Some("denied")
}

/// Filesystem store for pending-approval state: one `<pending_id>.json`
/// record plus one `<pending_id>.machine.json` mid-turn checkpoint per
/// gated effect, under a single approvals directory. No database — two
/// processes (the paused agent and the `agent approvals` resolver)
/// coordinate through these files only.
#[derive(Debug, Clone)]
pub struct ApprovalStore {
    dir: PathBuf,
}

impl ApprovalStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The conventional approvals directory, a sibling of the trace
    /// directory: `~/.local/share/agent/approvals`. Flat, like `traces/`:
    /// one record (+ one machine checkpoint) per pending effect, named by
    /// the run-scoped pending id; the record carries its `run_id` for
    /// per-run filtering.
    pub fn default_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
        Ok(home.join(".local/share/agent/approvals"))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn record_path(&self, pending_id: &str) -> PathBuf {
        self.dir.join(format!("{pending_id}.json"))
    }

    pub fn checkpoint_path(&self, pending_id: &str) -> PathBuf {
        self.dir.join(format!("{pending_id}.machine.json"))
    }

    /// Where a claimed (resume-in-progress or resumed) checkpoint lives;
    /// see [`ApprovalStore::claim_checkpoint`].
    pub fn claimed_checkpoint_path(&self, pending_id: &str) -> PathBuf {
        self.dir.join(format!("{pending_id}.machine.claimed.json"))
    }

    /// Persist a fresh pause: the record (status `awaiting_approval`) and
    /// the mid-turn machine checkpoint, both written atomically
    /// (tmp + rename). The checkpoint lands first so a record is never
    /// visible without its resume state.
    pub async fn write_pending(
        &self,
        record: &PendingEffectRecord,
        checkpoint: &crate::ir_interpreter::IrCheckpoint,
    ) -> Result<()> {
        tokio::fs::create_dir_all(&self.dir)
            .await
            .with_context(|| format!("creating approvals dir {}", self.dir.display()))?;
        write_atomic(
            &self.checkpoint_path(&record.pending_id),
            &serde_json::to_vec_pretty(checkpoint)?,
        )
        .await?;
        write_atomic(
            &self.record_path(&record.pending_id),
            &serde_json::to_vec_pretty(record)?,
        )
        .await
    }

    /// All records in the store, oldest first. A missing directory is an
    /// empty store, not an error.
    pub async fn list(&self) -> Result<Vec<PendingEffectRecord>> {
        let mut records = Vec::new();
        let mut entries = match tokio::fs::read_dir(&self.dir).await {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(records),
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("reading approvals dir {}", self.dir.display()))
            }
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // Only pending records: skip machine checkpoints (claimed or
            // not) and in-flight temp files.
            if !name.ends_with(".json") || name.contains(".machine.") {
                continue;
            }
            let text = tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("reading pending record {}", path.display()))?;
            let record: PendingEffectRecord = serde_json::from_str(&text)
                .with_context(|| format!("parsing pending record {}", path.display()))?;
            records.push(record);
        }
        records.sort_by(|a, b| {
            a.created_ts
                .cmp(&b.created_ts)
                .then_with(|| a.pending_id.cmp(&b.pending_id))
        });
        Ok(records)
    }

    pub async fn load(&self, pending_id: &str) -> Result<PendingEffectRecord> {
        let path = self.record_path(pending_id);
        let text = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("no pending approval {pending_id} at {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing pending record {}", path.display()))
    }

    pub async fn load_checkpoint(
        &self,
        pending_id: &str,
    ) -> Result<crate::ir_interpreter::IrCheckpoint> {
        let path = self.checkpoint_path(pending_id);
        let text = tokio::fs::read_to_string(&path).await.with_context(|| {
            format!(
                "no machine checkpoint for {pending_id} at {}",
                path.display()
            )
        })?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing machine checkpoint {}", path.display()))
    }

    /// Claim the machine checkpoint for resumption by atomically renaming
    /// it aside, then load it. Exactly one claimer can win: a second
    /// claim (a concurrent resumer, or a retry after a resume already ran)
    /// fails here instead of re-executing the effect. The claimed file is
    /// kept for inspection; recovering a resume that crashed after its
    /// claim is a manual step on purpose — the gated effect may already
    /// have executed.
    pub async fn claim_checkpoint(
        &self,
        pending_id: &str,
    ) -> Result<crate::ir_interpreter::IrCheckpoint> {
        let path = self.checkpoint_path(pending_id);
        let claimed = self.claimed_checkpoint_path(pending_id);
        tokio::fs::rename(&path, &claimed).await.map_err(|err| {
            if claimed.exists() {
                anyhow!(
                    "pending approval {pending_id} was already claimed for resumption \
                     ({} exists); refusing to resume it twice",
                    claimed.display()
                )
            } else {
                anyhow!(
                    "claiming machine checkpoint {} for {pending_id}: {err}",
                    path.display()
                )
            }
        })?;
        let text = tokio::fs::read_to_string(&claimed)
            .await
            .with_context(|| format!("reading claimed checkpoint {}", claimed.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing machine checkpoint {}", claimed.display()))
    }

    /// Durably resolve a pending record. Fails if the record is already
    /// resolved — a decision is made exactly once; re-approving cannot
    /// re-execute the effect.
    pub async fn resolve(
        &self,
        pending_id: &str,
        decision: ApprovalDecision,
        resolved_by: Option<String>,
        reason: Option<String>,
    ) -> Result<PendingEffectRecord> {
        let mut record = self.load(pending_id).await?;
        if !record.is_awaiting() {
            return Err(anyhow!(
                "pending approval {pending_id} is already resolved ({})",
                match record.status {
                    PendingStatus::Approved => "approved",
                    PendingStatus::Denied => "denied",
                    PendingStatus::AwaitingApproval => unreachable!(),
                }
            ));
        }
        record.status = match decision {
            ApprovalDecision::Approve => PendingStatus::Approved,
            ApprovalDecision::Deny => PendingStatus::Denied,
        };
        record.resolved_ts = Some(Utc::now());
        record.resolved_by = resolved_by;
        record.reason = reason;
        write_atomic(
            &self.record_path(pending_id),
            &serde_json::to_vec_pretty(&record)?,
        )
        .await?;
        Ok(record)
    }

    /// The [`ApprovalResolution`] a resume driver feeds into
    /// [`ApprovalConfig::resolutions`] for a resolved record.
    pub fn resolution_of(record: &PendingEffectRecord) -> Result<ApprovalResolution> {
        let decision = match record.status {
            PendingStatus::Approved => ApprovalDecision::Approve,
            PendingStatus::Denied => ApprovalDecision::Deny,
            PendingStatus::AwaitingApproval => {
                return Err(anyhow!(
                    "pending approval {} is not resolved yet",
                    record.pending_id
                ))
            }
        };
        Ok(ApprovalResolution {
            decision,
            resolved_by: record.resolved_by.clone(),
            reason: record.reason.clone(),
        })
    }
}

async fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    tokio::fs::write(&tmp, bytes)
        .await
        .with_context(|| format!("writing {}", tmp.display()))?;
    tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("renaming {} into place", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_interpreter::{InMemoryStore, IrCheckpoint};

    fn record(pending_id: &str) -> PendingEffectRecord {
        PendingEffectRecord {
            pending_id: pending_id.into(),
            run_id: "run-1".into(),
            turn_id: Some("run-1-t0".into()),
            effect_id: "sha256:abc".into(),
            program_hash: "sha256:def".into(),
            kind: ApprovalKind::Eval,
            request: serde_json::json!({ "command": "echo hi", "argv": null }),
            created_ts: Utc::now(),
            status: PendingStatus::AwaitingApproval,
            resolved_ts: None,
            resolved_by: None,
            reason: None,
            runtime: None,
        }
    }

    fn checkpoint() -> IrCheckpoint {
        let machine =
            crate::ir_agent::agent_loop_ir(crate::op::Model("test-model".into()), vec![], 4);
        IrCheckpoint {
            machine,
            store: InMemoryStore::new(),
        }
    }

    #[tokio::test]
    async fn write_list_resolve_round_trip() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("agent-approvals-{}", uuid::Uuid::new_v4()));
        let store = ApprovalStore::new(&dir);
        store.write_pending(&record("pa-1"), &checkpoint()).await?;

        let listed = store.list().await?;
        assert_eq!(listed.len(), 1);
        assert!(listed[0].is_awaiting());

        let resolved = store
            .resolve("pa-1", ApprovalDecision::Approve, Some("ben".into()), None)
            .await?;
        assert_eq!(resolved.status, PendingStatus::Approved);
        assert_eq!(resolved.resolved_by.as_deref(), Some("ben"));
        assert!(resolved.resolved_ts.is_some());

        // Exactly-once: a second resolution fails.
        let err = store
            .resolve("pa-1", ApprovalDecision::Deny, None, None)
            .await
            .expect_err("double-resolve must fail");
        assert!(err.to_string().contains("already resolved"), "{err}");

        // The checkpoint survives alongside.
        let loaded = store.load_checkpoint("pa-1").await?;
        assert_eq!(loaded, checkpoint());
        Ok(())
    }

    #[tokio::test]
    async fn missing_dir_lists_empty() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("agent-approvals-{}", uuid::Uuid::new_v4()));
        assert!(ApprovalStore::new(dir).list().await?.is_empty());
        Ok(())
    }

    #[test]
    fn pending_ids_are_run_scoped_and_deterministic() {
        let a = pending_id_for("run-1", "sha256:e1");
        assert_eq!(a, pending_id_for("run-1", "sha256:e1"));
        assert_ne!(a, pending_id_for("run-2", "sha256:e1"));
        assert_ne!(a, pending_id_for("run-1", "sha256:e2"));
        assert!(a.starts_with("pa-") && a.len() == 15, "{a}");
    }

    #[test]
    fn denial_value_is_typed_and_detectable() {
        let value = denial_value("pa-9", Some("ben"), Some("not on prod"));
        assert_eq!(value["ok"], serde_json::json!(false));
        assert_eq!(value["error"], "approval denied: not on prod");
        assert_eq!(value["approval"]["pending_id"], "pa-9");
        assert!(is_denial_value(&value));
        assert!(!is_denial_value(&serde_json::json!({ "ok": false })));
    }
}
