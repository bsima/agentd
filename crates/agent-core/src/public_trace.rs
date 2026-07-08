//! Public trace event schema (docs/TRACE_SCHEMA.md, t-1308.3).
//!
//! The runtime trace [`Event`] enum serializes with PascalCase variant tags
//! that are load-bearing for replay: every existing trace file depends on
//! them, so they are *not* the public API. This module is the public API: a
//! versioned, documented **view** over runtime events. [`public_event`]
//! projects a runtime event to its public shape ([`PublicEvent`]) with
//! dotted lifecycle names (`infer.started`, `eval.completed`, ...), or to
//! `None` for events that are runtime-internal (gc, hydration, par,
//! checkpoints, Custom extension events).
//!
//! Consumers — the SDK trace adapter, dashboard ingest, external tooling —
//! program against this schema, never against the runtime variant names.
//! Compatibility policy, field catalog, payload rules, and the reserved
//! event names live in docs/TRACE_SCHEMA.md.

use crate::ir::EffectLocation;
use crate::trace::{preview, Event};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Version of the public event schema emitted by [`public_event`].
/// Additive changes bump the minor (not represented on the wire); renames
/// or removals bump this number. See docs/TRACE_SCHEMA.md.
pub const PUBLIC_SCHEMA_VERSION: u32 = 1;

/// Lifecycle phase of a public event. A nonzero Eval exit is still
/// `completed` (the effect ran to completion; see `attrs.ok`); `failed` is
/// reserved for terminal effect failure (the runtime `*Error` events).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicStatus {
    Started,
    Completed,
    Failed,
}

/// Public projection of the stable IR effect identity (t-1057/t-1058).
/// Mirrors [`EffectLocation`] with independent types so runtime refactors
/// cannot silently change the public wire shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicEffect {
    pub effect_id: String,
    pub program_hash: String,
    pub site: PublicEffectSite,
    pub dynamic_path: PublicDynamicPath,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicEffectSite {
    pub block: u32,
    pub instruction_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicDynamicPath {
    pub path: String,
    pub transitions: u64,
    pub visit: u64,
}

impl From<&EffectLocation> for PublicEffect {
    fn from(location: &EffectLocation) -> Self {
        Self {
            effect_id: location.effect_id.0.clone(),
            program_hash: location.program_hash.0.clone(),
            site: PublicEffectSite {
                block: location.site.block.0,
                instruction_index: location.site.instruction_index,
            },
            dynamic_path: PublicDynamicPath {
                path: location.dynamic_path.path.clone(),
                transitions: location.dynamic_path.transitions,
                visit: location.dynamic_path.visit,
            },
        }
    }
}

/// One public trace event: schema_version 1. Field order here is the wire
/// order (serde serializes struct fields in declaration order); the golden
/// conformance test pins it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublicEvent {
    pub schema_version: u32,
    /// Dotted lifecycle name, e.g. `infer.started`. The full catalog and
    /// the reserved-but-unemitted names are in docs/TRACE_SCHEMA.md.
    pub event: String,
    pub ts: DateTime<Utc>,
    pub run_id: String,
    /// Currently always equal to `run_id`: the runtime does not yet
    /// distinguish sessions from runs. Kept as a separate field so ingest
    /// schemas do not have to migrate when it does.
    pub session_id: String,
    /// Reserved. Turn ids exist today only on the supervisor-facing machine
    /// events (stdout); runtime trace events do not carry them yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Runtime operation id: pairs `*.started` with its `*.completed` /
    /// `*.failed`, and addresses the full payload in the runtime trace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_id: Option<u64>,
    /// Runtime parent operation id (op lineage, e.g. work inside a Par).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_op_id: Option<u64>,
    pub status: PublicStatus,
    /// Stable IR effect identity; absent on op-layer (non-IR) traces and on
    /// result/error events (correlate via `op_id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effect: Option<PublicEffect>,
    /// Reserved. The runtime tracks op-level lineage (`parent_op_id`), not
    /// effect-level lineage; this field is emitted once it does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_effect_id: Option<String>,
    /// Terminal error message; present exactly when `status` is `failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Truncated, human-oriented payload excerpt. Never the replay
    /// identity; full payloads live in the runtime trace, addressed by
    /// (`run_id`, `op_id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_preview: Option<String>,
    /// Reserved. Opaque reference to an out-of-band full payload once a
    /// payload store exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_ref: Option<String>,
    /// Event-specific attributes; keys per event are documented in
    /// docs/TRACE_SCHEMA.md and are additive-only within a major version.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: BTreeMap<String, Value>,
}

fn base(
    event: &str,
    ts: DateTime<Utc>,
    run_id: &str,
    op_id: Option<u64>,
    parent_op_id: Option<u64>,
    status: PublicStatus,
) -> PublicEvent {
    PublicEvent {
        schema_version: PUBLIC_SCHEMA_VERSION,
        event: event.into(),
        ts,
        run_id: run_id.into(),
        session_id: run_id.into(),
        turn_id: None,
        op_id,
        parent_op_id,
        status,
        effect: None,
        parent_effect_id: None,
        error: None,
        payload_preview: None,
        payload_ref: None,
        attrs: BTreeMap::new(),
    }
}

fn non_empty_preview(text: &str) -> Option<String> {
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

const PAYLOAD_PREVIEW_MAX_CHARS: usize = 1024;

/// Project one runtime trace event to its public schema-v1 shape, or `None`
/// for runtime-internal events. The match is exhaustive on purpose: adding
/// a runtime variant forces an explicit public/private decision here (and
/// in the decision-table test below).
pub fn public_event(event: &Event) -> Option<PublicEvent> {
    match event {
        Event::InferCall {
            run_id,
            op_id,
            parent_op_id,
            model,
            prompt: _,
            prompt_preview,
            effect,
            timestamp,
        } => {
            let mut out = base(
                "infer.started",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Started,
            );
            out.effect = effect.as_deref().map(PublicEffect::from);
            out.payload_preview = non_empty_preview(prompt_preview);
            out.attrs
                .insert("model".into(), Value::String(model.clone()));
            Some(out)
        }
        Event::InferResult {
            run_id,
            op_id,
            parent_op_id,
            response: _,
            response_preview,
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            cost_micro_usd,
            pricing,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "infer.completed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Completed,
            );
            out.payload_preview = non_empty_preview(response_preview);
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            out.attrs
                .insert("input_tokens".into(), (*input_tokens).into());
            out.attrs
                .insert("output_tokens".into(), (*output_tokens).into());
            out.attrs
                .insert("total_tokens".into(), (*total_tokens).into());
            // Cost accounting (schema 1.3, t-1334): present exactly when
            // the runtime recorded them — absent cost means unknown
            // pricing, never zero.
            if let Some(cached) = cached_input_tokens {
                out.attrs
                    .insert("cached_input_tokens".into(), (*cached).into());
            }
            if let Some(cost) = cost_micro_usd {
                out.attrs.insert("cost_micro_usd".into(), (*cost).into());
            }
            if let Some(pricing) = pricing {
                out.attrs.insert(
                    "pricing".into(),
                    serde_json::json!({
                        "input_micro_usd_per_mtok": pricing.input_micro_usd_per_mtok,
                        "output_micro_usd_per_mtok": pricing.output_micro_usd_per_mtok,
                    }),
                );
            }
            Some(out)
        }
        Event::InferError {
            run_id,
            op_id,
            parent_op_id,
            error,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "infer.failed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Failed,
            );
            out.error = Some(error.clone());
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            Some(out)
        }
        Event::EvalCall {
            run_id,
            op_id,
            parent_op_id,
            command,
            argv,
            cwd,
            env_policy,
            timeout_ms,
            effect,
            timestamp,
        } => {
            let mut out = base(
                "eval.started",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Started,
            );
            out.effect = effect.as_deref().map(PublicEffect::from);
            out.payload_preview = non_empty_preview(command);
            if let Some(argv) = argv {
                out.attrs
                    .insert("argv".into(), serde_json::json!(argv.clone()));
            }
            if let Some(cwd) = cwd {
                out.attrs.insert("cwd".into(), Value::String(cwd.clone()));
            }
            out.attrs
                .insert("env_policy".into(), Value::String(env_policy.clone()));
            out.attrs.insert("timeout_ms".into(), (*timeout_ms).into());
            Some(out)
        }
        Event::EvalResult {
            run_id,
            op_id,
            parent_op_id,
            command,
            result,
            duration_ms,
            truncated_stdout,
            truncated_stderr,
            timestamp,
        } => {
            let mut out = base(
                "eval.completed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Completed,
            );
            out.payload_preview = result
                .get("stdout")
                .and_then(Value::as_str)
                .filter(|stdout| !stdout.is_empty())
                .map(|stdout| preview(stdout, PAYLOAD_PREVIEW_MAX_CHARS));
            out.attrs
                .insert("command".into(), Value::String(command.clone()));
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            if let Some(ok) = result.get("ok").and_then(Value::as_bool) {
                out.attrs.insert("ok".into(), Value::Bool(ok));
            }
            if let Some(exit_code) = result.get("status").and_then(Value::as_i64) {
                out.attrs.insert("exit_code".into(), exit_code.into());
            }
            if let Some(timed_out) = result.get("timed_out").and_then(Value::as_bool) {
                out.attrs.insert("timed_out".into(), Value::Bool(timed_out));
            }
            out.attrs
                .insert("truncated_stdout".into(), Value::Bool(*truncated_stdout));
            out.attrs
                .insert("truncated_stderr".into(), Value::Bool(*truncated_stderr));
            Some(out)
        }
        Event::EvalError {
            run_id,
            op_id,
            parent_op_id,
            command,
            error,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "eval.failed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Failed,
            );
            out.error = Some(error.clone());
            out.attrs
                .insert("command".into(), Value::String(command.clone()));
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            Some(out)
        }
        Event::RetrieveCall {
            run_id,
            op_id,
            parent_op_id,
            query,
            kind,
            max_bytes,
            effect,
            timestamp,
        } => {
            let mut out = base(
                "retrieve.started",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Started,
            );
            out.effect = effect.as_deref().map(PublicEffect::from);
            out.payload_preview = non_empty_preview(query);
            if let Some(kind) = kind {
                out.attrs.insert("kind".into(), Value::String(kind.clone()));
            }
            if let Some(max_bytes) = max_bytes {
                out.attrs.insert("max_bytes".into(), (*max_bytes).into());
            }
            Some(out)
        }
        Event::RetrieveResult {
            run_id,
            op_id,
            parent_op_id,
            results: _,
            result_preview,
            source_count,
            bytes,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "retrieve.completed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Completed,
            );
            out.payload_preview = non_empty_preview(result_preview);
            out.attrs.insert("bytes".into(), (*bytes).into());
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            out.attrs
                .insert("source_count".into(), (*source_count).into());
            Some(out)
        }
        Event::RetrieveError {
            run_id,
            op_id,
            parent_op_id,
            error,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "retrieve.failed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Failed,
            );
            out.error = Some(error.clone());
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            Some(out)
        }
        Event::StoreCall {
            run_id,
            op_id,
            parent_op_id,
            sink,
            store_op,
            store_id,
            item_preview,
            content_hash,
            effect,
            timestamp,
        } => {
            let mut out = base(
                "store.started",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Started,
            );
            out.effect = effect.as_deref().map(PublicEffect::from);
            out.payload_preview = non_empty_preview(item_preview);
            out.attrs
                .insert("content_hash".into(), Value::String(content_hash.clone()));
            out.attrs.insert("sink".into(), Value::String(sink.clone()));
            if let Some(store_id) = store_id {
                out.attrs
                    .insert("store_id".into(), Value::String(store_id.clone()));
            }
            out.attrs
                .insert("store_op".into(), Value::String(store_op.clone()));
            Some(out)
        }
        Event::StoreResult {
            run_id,
            op_id,
            parent_op_id,
            sink,
            sink_id,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "store.completed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Completed,
            );
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            out.attrs.insert("sink".into(), Value::String(sink.clone()));
            out.attrs
                .insert("sink_id".into(), Value::String(sink_id.clone()));
            Some(out)
        }
        Event::StoreError {
            run_id,
            op_id,
            parent_op_id,
            sink,
            error,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "store.failed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Failed,
            );
            out.error = Some(error.clone());
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            out.attrs.insert("sink".into(), Value::String(sink.clone()));
            Some(out)
        }
        // Native tool dispatch (t-1308.7, schema 1.2): the reserved
        // `tool.requested`/`tool.completed`/`tool.failed` names, emitted for
        // registered native tools. Built-in tools still surface as the
        // `eval.*`/`retrieve.*`/`store.*` effects they compile to.
        Event::ToolCall {
            run_id,
            op_id,
            parent_op_id,
            name,
            arguments,
            effect,
            timestamp,
        } => {
            let mut out = base(
                "tool.requested",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Started,
            );
            out.effect = effect.as_deref().map(PublicEffect::from);
            // The runtime event carries the full arguments (replay
            // identity); the public event carries only a preview, per the
            // payload rules.
            out.payload_preview = Some(preview(&arguments.to_string(), PAYLOAD_PREVIEW_MAX_CHARS));
            out.attrs.insert("name".into(), Value::String(name.clone()));
            Some(out)
        }
        Event::ToolResult {
            run_id,
            op_id,
            parent_op_id,
            name,
            result: _,
            result_preview,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "tool.completed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Completed,
            );
            out.payload_preview = non_empty_preview(result_preview);
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            out.attrs.insert("name".into(), Value::String(name.clone()));
            Some(out)
        }
        Event::ToolError {
            run_id,
            op_id,
            parent_op_id,
            name,
            error,
            duration_ms,
            timestamp,
        } => {
            let mut out = base(
                "tool.failed",
                *timestamp,
                run_id,
                Some(*op_id),
                *parent_op_id,
                PublicStatus::Failed,
            );
            out.error = Some(error.clone());
            out.attrs
                .insert("duration_ms".into(), (*duration_ms).into());
            out.attrs.insert("name".into(), Value::String(name.clone()));
            Some(out)
        }
        Event::AgentDone {
            run_id,
            usage,
            timestamp,
        } => {
            let mut out = base(
                "run.completed",
                *timestamp,
                run_id,
                None,
                None,
                PublicStatus::Completed,
            );
            // Run rollup (schema 1.3, t-1334): exact integer sums of the
            // run's recorded InferResult usage/cost. Absent entirely for
            // infer-less runs and pre-1.3 traces.
            if let Some(usage) = usage {
                out.attrs
                    .insert("infer_calls".into(), usage.infer_calls.into());
                out.attrs
                    .insert("input_tokens".into(), usage.input_tokens.into());
                out.attrs
                    .insert("output_tokens".into(), usage.output_tokens.into());
                out.attrs
                    .insert("total_tokens".into(), usage.total_tokens.into());
                if let Some(cached) = usage.cached_input_tokens {
                    out.attrs
                        .insert("cached_input_tokens".into(), cached.into());
                }
                if let Some(cost) = usage.cost_micro_usd {
                    out.attrs.insert("cost_micro_usd".into(), cost.into());
                }
                out.attrs.insert(
                    "uncosted_infer_calls".into(),
                    usage.uncosted_infer_calls.into(),
                );
                // Failed attempts (schema 1.5, t-1347): present only when
                // nonzero, mirroring the rollup's own serialization —
                // pre-t-1347 traces and all-success runs project without
                // the key.
                if usage.failed_infer_calls > 0 {
                    out.attrs
                        .insert("failed_infer_calls".into(), usage.failed_infer_calls.into());
                }
            }
            Some(out)
        }
        // Structured-output validation failure (t-1308.4): the one Custom
        // name with a public projection (schema 1.1). Each failed attempt
        // emits one event; the run may still complete after a repair turn,
        // so `failed` here is per-attempt, not necessarily terminal.
        Event::Custom {
            run_id,
            name,
            data,
            timestamp,
        } if name == crate::output_contract::OUTPUT_VALIDATION_FAILED_EVENT => {
            let mut out = base(
                "output.validation_failed",
                *timestamp,
                run_id,
                None,
                None,
                PublicStatus::Failed,
            );
            out.error = Some(
                data.get("errors")
                    .and_then(Value::as_array)
                    .and_then(|errors| errors.first())
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| "output failed schema validation".into()),
            );
            out.payload_preview = data
                .get("preview")
                .and_then(Value::as_str)
                .filter(|preview| !preview.is_empty())
                .map(str::to_owned);
            if let Some(attempt) = data.get("attempt") {
                out.attrs.insert("attempt".into(), attempt.clone());
            }
            if let Some(errors) = data.get("errors") {
                out.attrs.insert("errors".into(), errors.clone());
            }
            Some(out)
        }
        // Approval gates (t-1308.10, schema 1.4): the reserved
        // `approval.requested` / `approval.resolved` names, now emitted.
        // Neither carries an op_id — the gate sits ahead of effect dispatch
        // (a paused or denied effect never becomes an operation); correlate
        // the pair via `attrs.pending_id`, and the resolved event to its
        // effect via `attrs.effect_id`.
        Event::ApprovalRequested {
            run_id,
            pending_id,
            kind,
            request,
            effect,
            timestamp,
        } => {
            let mut out = base(
                "approval.requested",
                *timestamp,
                run_id,
                None,
                None,
                PublicStatus::Started,
            );
            out.effect = Some(PublicEffect::from(effect.as_ref()));
            out.payload_preview = Some(preview(&request.to_string(), PAYLOAD_PREVIEW_MAX_CHARS));
            out.attrs.insert("kind".into(), Value::String(kind.clone()));
            out.attrs
                .insert("pending_id".into(), Value::String(pending_id.clone()));
            Some(out)
        }
        // A denial is `completed`, not `failed`: the gate resolved and the
        // denial continues the program as a typed value (errors-as-values);
        // `failed` stays reserved for terminal effect failure.
        Event::ApprovalResolved {
            run_id,
            pending_id,
            effect_id,
            kind,
            decision,
            resolved_by,
            reason,
            timestamp,
        } => {
            let mut out = base(
                "approval.resolved",
                *timestamp,
                run_id,
                None,
                None,
                PublicStatus::Completed,
            );
            out.attrs
                .insert("decision".into(), Value::String(decision.clone()));
            out.attrs
                .insert("effect_id".into(), Value::String(effect_id.clone()));
            out.attrs.insert("kind".into(), Value::String(kind.clone()));
            out.attrs
                .insert("pending_id".into(), Value::String(pending_id.clone()));
            if let Some(reason) = reason {
                out.attrs
                    .insert("reason".into(), Value::String(reason.clone()));
            }
            if let Some(resolved_by) = resolved_by {
                out.attrs
                    .insert("resolved_by".into(), Value::String(resolved_by.clone()));
            }
            Some(out)
        }
        // Private: runtime-internal orchestration and extension events.
        // Context assembly (hydration/prompt IR) and Par structure may gain
        // public projections later (additive, minor bump); Custom is the
        // domain-tag / gc / diagnostics extension channel and — except for
        // output_validation_failed above — is never public; consumers who
        // need it read the runtime trace.
        Event::HydrationStart { .. }
        | Event::HydrationSection { .. }
        | Event::HydrationEnd { .. }
        | Event::ParStart { .. }
        | Event::ParEnd { .. }
        | Event::Checkpoint { .. }
        | Event::TurnBudgetExhausted { .. }
        | Event::Custom { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::{EvalConfig, SeqConfig};
    use crate::ir::{
        Block, BlockId, EvalRequest, Expr, Instr, Machine, Program, ProgramId, PromptRef,
        Terminator, Var,
    };
    use crate::op::{ChatMessage, Response, ToolCall};
    use crate::provider::{ChatProvider, ToolSpec};
    use crate::trace::TraceLogger;
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    /// Decision table: every runtime Event variant either projects to a
    /// public name or is explicitly private. `public_event`'s exhaustive
    /// match makes a new variant a compile error there; this test makes the
    /// intended decision reviewable in one place.
    #[test]
    fn every_variant_projects_or_is_explicitly_private() {
        let ts = DateTime::<Utc>::UNIX_EPOCH;
        let run = "run";
        let events: Vec<(Event, Option<&str>)> = vec![
            (
                Event::InferCall {
                    run_id: run.into(),
                    op_id: 1,
                    parent_op_id: None,
                    model: "mock".into(),
                    prompt: None,
                    prompt_preview: "hi".into(),
                    effect: None,
                    timestamp: ts,
                },
                Some("infer.started"),
            ),
            (
                Event::InferResult {
                    run_id: run.into(),
                    op_id: 1,
                    parent_op_id: None,
                    response: None,
                    response_preview: "ok".into(),
                    input_tokens: 1,
                    output_tokens: 2,
                    total_tokens: 3,
                    cached_input_tokens: None,
                    cost_micro_usd: None,
                    pricing: None,
                    duration_ms: 4,
                    timestamp: ts,
                },
                Some("infer.completed"),
            ),
            (
                Event::InferError {
                    run_id: run.into(),
                    op_id: 1,
                    parent_op_id: None,
                    error: "boom".into(),
                    duration_ms: 4,
                    timestamp: ts,
                },
                Some("infer.failed"),
            ),
            (
                Event::EvalCall {
                    run_id: run.into(),
                    op_id: 2,
                    parent_op_id: None,
                    command: "true".into(),
                    argv: None,
                    cwd: None,
                    env_policy: "inherit".into(),
                    timeout_ms: 1000,
                    effect: None,
                    timestamp: ts,
                },
                Some("eval.started"),
            ),
            (
                Event::EvalResult {
                    run_id: run.into(),
                    op_id: 2,
                    parent_op_id: None,
                    command: "true".into(),
                    result: serde_json::json!({ "ok": true, "status": 0 }),
                    duration_ms: 1,
                    truncated_stdout: false,
                    truncated_stderr: false,
                    timestamp: ts,
                },
                Some("eval.completed"),
            ),
            (
                Event::EvalError {
                    run_id: run.into(),
                    op_id: 2,
                    parent_op_id: None,
                    command: "true".into(),
                    error: "spawn failed".into(),
                    duration_ms: 1,
                    timestamp: ts,
                },
                Some("eval.failed"),
            ),
            (
                Event::RetrieveCall {
                    run_id: run.into(),
                    op_id: 3,
                    parent_op_id: None,
                    query: "q".into(),
                    kind: None,
                    max_bytes: None,
                    effect: None,
                    timestamp: ts,
                },
                Some("retrieve.started"),
            ),
            (
                Event::RetrieveResult {
                    run_id: run.into(),
                    op_id: 3,
                    parent_op_id: None,
                    results: Value::Null,
                    result_preview: "r".into(),
                    source_count: 1,
                    bytes: 2,
                    duration_ms: 3,
                    timestamp: ts,
                },
                Some("retrieve.completed"),
            ),
            (
                Event::RetrieveError {
                    run_id: run.into(),
                    op_id: 3,
                    parent_op_id: None,
                    error: "source error".into(),
                    duration_ms: 3,
                    timestamp: ts,
                },
                Some("retrieve.failed"),
            ),
            (
                Event::StoreCall {
                    run_id: run.into(),
                    op_id: 4,
                    parent_op_id: None,
                    sink: "memory".into(),
                    store_op: "create".into(),
                    store_id: None,
                    item_preview: "item".into(),
                    content_hash: "sha256:x".into(),
                    effect: None,
                    timestamp: ts,
                },
                Some("store.started"),
            ),
            (
                Event::StoreResult {
                    run_id: run.into(),
                    op_id: 4,
                    parent_op_id: None,
                    sink: "memory".into(),
                    sink_id: "id-1".into(),
                    duration_ms: 1,
                    timestamp: ts,
                },
                Some("store.completed"),
            ),
            (
                Event::StoreError {
                    run_id: run.into(),
                    op_id: 4,
                    parent_op_id: None,
                    sink: "memory".into(),
                    error: "sink error".into(),
                    duration_ms: 1,
                    timestamp: ts,
                },
                Some("store.failed"),
            ),
            (
                Event::ToolCall {
                    run_id: run.into(),
                    op_id: 6,
                    parent_op_id: None,
                    name: "lookup".into(),
                    arguments: serde_json::json!({ "city": "sf" }),
                    effect: None,
                    timestamp: ts,
                },
                Some("tool.requested"),
            ),
            (
                Event::ToolResult {
                    run_id: run.into(),
                    op_id: 6,
                    parent_op_id: None,
                    name: "lookup".into(),
                    result: serde_json::json!({ "temp": 61 }),
                    result_preview: "{\"temp\":61}".into(),
                    duration_ms: 1,
                    timestamp: ts,
                },
                Some("tool.completed"),
            ),
            (
                Event::ToolError {
                    run_id: run.into(),
                    op_id: 6,
                    parent_op_id: None,
                    name: "lookup".into(),
                    error: "handler error".into(),
                    duration_ms: 1,
                    timestamp: ts,
                },
                Some("tool.failed"),
            ),
            (
                Event::AgentDone {
                    run_id: run.into(),
                    usage: None,
                    timestamp: ts,
                },
                Some("run.completed"),
            ),
            (
                Event::HydrationStart {
                    run_id: run.into(),
                    op_id: 5,
                    parent_op_id: None,
                    sources: vec![],
                    max_bytes: None,
                    timestamp: ts,
                },
                None,
            ),
            (
                Event::HydrationSection {
                    run_id: run.into(),
                    op_id: 5,
                    parent_op_id: None,
                    source: "s".into(),
                    kind: "k".into(),
                    bytes: 0,
                    content_preview: String::new(),
                    metadata: Value::Null,
                    timestamp: ts,
                },
                None,
            ),
            (
                Event::HydrationEnd {
                    run_id: run.into(),
                    op_id: 5,
                    parent_op_id: None,
                    section_count: 0,
                    total_bytes: 0,
                    timestamp: ts,
                },
                None,
            ),
            (
                Event::ParStart {
                    run_id: run.into(),
                    op_id: 6,
                    parent_op_id: None,
                    branch_count: 2,
                    timestamp: ts,
                },
                None,
            ),
            (
                Event::ParEnd {
                    run_id: run.into(),
                    op_id: 6,
                    parent_op_id: None,
                    branch_count: 2,
                    duration_ms: 1,
                    timestamp: ts,
                },
                None,
            ),
            (
                Event::Checkpoint {
                    run_id: run.into(),
                    name: "c".into(),
                    path: None,
                    timestamp: ts,
                },
                None,
            ),
            (
                Event::TurnBudgetExhausted {
                    run_id: run.into(),
                    max_turns: 1,
                    pending_tool_calls: 0,
                    first_tool: None,
                    timestamp: ts,
                },
                None,
            ),
            // Approval gates (schema 1.4, t-1308.10): both events public.
            (
                Event::ApprovalRequested {
                    run_id: run.into(),
                    pending_id: "pa-abc".into(),
                    kind: "eval".into(),
                    request: serde_json::json!({ "command": "echo hi", "argv": null }),
                    effect: Box::new(test_effect_location()),
                    timestamp: ts,
                },
                Some("approval.requested"),
            ),
            (
                Event::ApprovalResolved {
                    run_id: run.into(),
                    pending_id: "pa-abc".into(),
                    effect_id: "sha256:e".into(),
                    kind: "eval".into(),
                    decision: "denied".into(),
                    resolved_by: Some("ben".into()),
                    reason: Some("not on prod".into()),
                    timestamp: ts,
                },
                Some("approval.resolved"),
            ),
            (
                Event::Custom {
                    run_id: run.into(),
                    name: "gc_collect".into(),
                    data: Value::Null,
                    timestamp: ts,
                },
                None,
            ),
            // The one public Custom name (schema 1.1, t-1308.4)...
            (
                Event::Custom {
                    run_id: run.into(),
                    name: "output_validation_failed".into(),
                    data: serde_json::json!({
                        "attempt": 1,
                        "errors": ["$: missing required property \"a\""],
                        "preview": "{}",
                    }),
                    timestamp: ts,
                },
                Some("output.validation_failed"),
            ),
            // ...while its run-metadata sibling stays private.
            (
                Event::Custom {
                    run_id: run.into(),
                    name: "output_contract".into(),
                    data: serde_json::json!({ "schema_hash": "sha256:x", "max_repairs": 2 }),
                    timestamp: ts,
                },
                None,
            ),
        ];
        for (event, expected) in &events {
            let projected = public_event(event);
            assert_eq!(
                projected.as_ref().map(|public| public.event.as_str()),
                *expected,
                "projection decision changed for {event:?}"
            );
            if let Some(public) = projected {
                assert_eq!(public.schema_version, PUBLIC_SCHEMA_VERSION);
                assert_eq!(public.session_id, public.run_id);
                assert_eq!(
                    public.error.is_some(),
                    public.status == PublicStatus::Failed
                );
                // Reserved fields stay unemitted in schema v1.
                assert_eq!(public.turn_id, None);
                assert_eq!(public.parent_effect_id, None);
                assert_eq!(public.payload_ref, None);
            }
        }
    }

    fn test_effect_location() -> EffectLocation {
        crate::ir::effect_location(
            crate::ir::ProgramHash("sha256:p".into()),
            crate::ir::EffectKind::Eval,
            crate::ir::EffectSite {
                block: crate::ir::BlockId(0),
                instruction_index: 1,
            },
            crate::ir::DynamicPath::at_entry(0),
        )
        .expect("effect location")
    }

    /// Approval projection shape (schema 1.4, t-1308.10): the pair
    /// correlates by pending_id (no op_id — the gate precedes dispatch),
    /// `requested` carries the effect identity and a request preview, and a
    /// denial is `completed` (it resolves to a value; nothing failed).
    #[test]
    fn approval_events_project_with_pending_id_correlation() {
        let ts = DateTime::<Utc>::UNIX_EPOCH;
        let requested = public_event(&Event::ApprovalRequested {
            run_id: "run".into(),
            pending_id: "pa-abc".into(),
            kind: "eval".into(),
            request: serde_json::json!({ "command": "rm -rf /tmp/x", "argv": null }),
            effect: Box::new(test_effect_location()),
            timestamp: ts,
        })
        .expect("public");
        assert_eq!(requested.event, "approval.requested");
        assert_eq!(requested.status, PublicStatus::Started);
        assert_eq!(requested.op_id, None);
        assert_eq!(
            requested.effect.as_ref().map(|effect| &effect.effect_id),
            Some(&test_effect_location().effect_id.0)
        );
        assert_eq!(requested.attrs.get("kind"), Some(&Value::from("eval")));
        assert_eq!(
            requested.attrs.get("pending_id"),
            Some(&Value::from("pa-abc"))
        );
        assert_eq!(
            requested.payload_preview.as_deref(),
            Some(r#"{"argv":null,"command":"rm -rf /tmp/x"}"#)
        );

        let resolved = public_event(&Event::ApprovalResolved {
            run_id: "run".into(),
            pending_id: "pa-abc".into(),
            effect_id: test_effect_location().effect_id.0,
            kind: "eval".into(),
            decision: "denied".into(),
            resolved_by: Some("ben".into()),
            reason: Some("not on prod".into()),
            timestamp: ts,
        })
        .expect("public");
        assert_eq!(resolved.event, "approval.resolved");
        assert_eq!(resolved.status, PublicStatus::Completed);
        assert_eq!(resolved.error, None, "a denial is a value, not a failure");
        assert_eq!(resolved.op_id, None);
        assert_eq!(resolved.attrs.get("decision"), Some(&Value::from("denied")));
        assert_eq!(
            resolved.attrs.get("pending_id"),
            Some(&Value::from("pa-abc"))
        );
        assert_eq!(
            resolved.attrs.get("effect_id"),
            requested
                .effect
                .as_ref()
                .map(|effect| Value::from(effect.effect_id.clone()))
                .as_ref()
        );
        assert_eq!(resolved.attrs.get("resolved_by"), Some(&Value::from("ben")));
        assert_eq!(
            resolved.attrs.get("reason"),
            Some(&Value::from("not on prod"))
        );
    }

    /// Cost accounting projection (schema 1.3, t-1334): infer.completed
    /// carries cached/cost/pricing attrs exactly when recorded, and
    /// run.completed carries the AgentDone usage rollup. Absent recorded
    /// values project to absent attrs — never zero.
    #[test]
    fn cost_fields_project_into_infer_completed_and_run_completed_attrs() {
        let ts = DateTime::<Utc>::UNIX_EPOCH;
        let costed = Event::InferResult {
            run_id: "run".into(),
            op_id: 1,
            parent_op_id: None,
            response: None,
            response_preview: "ok".into(),
            input_tokens: 7,
            output_tokens: 3,
            total_tokens: 10,
            cached_input_tokens: Some(2),
            cost_micro_usd: Some(66),
            pricing: Some(crate::cost::Pricing {
                input_micro_usd_per_mtok: 3_000_000,
                output_micro_usd_per_mtok: 15_000_000,
            }),
            duration_ms: 4,
            timestamp: ts,
        };
        let public = public_event(&costed).expect("projects");
        assert_eq!(
            public.attrs.get("cached_input_tokens"),
            Some(&Value::from(2))
        );
        assert_eq!(public.attrs.get("cost_micro_usd"), Some(&Value::from(66)));
        assert_eq!(
            public.attrs.get("pricing"),
            Some(&serde_json::json!({
                "input_micro_usd_per_mtok": 3_000_000,
                "output_micro_usd_per_mtok": 15_000_000,
            }))
        );

        let done = Event::AgentDone {
            run_id: "run".into(),
            usage: Some(crate::cost::RunUsage {
                infer_calls: 2,
                input_tokens: 14,
                output_tokens: 6,
                total_tokens: 20,
                cached_input_tokens: None,
                cost_micro_usd: Some(132),
                uncosted_infer_calls: 1,
                failed_infer_calls: 0,
            }),
            timestamp: ts,
        };
        let public = public_event(&done).expect("projects");
        assert_eq!(public.event, "run.completed");
        assert_eq!(public.attrs.get("infer_calls"), Some(&Value::from(2)));
        assert_eq!(public.attrs.get("input_tokens"), Some(&Value::from(14)));
        assert_eq!(public.attrs.get("output_tokens"), Some(&Value::from(6)));
        assert_eq!(public.attrs.get("total_tokens"), Some(&Value::from(20)));
        assert_eq!(public.attrs.get("cost_micro_usd"), Some(&Value::from(132)));
        assert_eq!(
            public.attrs.get("uncosted_infer_calls"),
            Some(&Value::from(1))
        );
        // Never-reported cached tokens stay absent, not zero.
        assert_eq!(public.attrs.get("cached_input_tokens"), None);
        // All-success runs project without the failed-attempts key
        // (schema 1.5, t-1347) ...
        assert_eq!(public.attrs.get("failed_infer_calls"), None);

        // ... and runs with failed attempts carry the count.
        let with_failures = Event::AgentDone {
            run_id: "run".into(),
            usage: Some(crate::cost::RunUsage {
                failed_infer_calls: 3,
                ..Default::default()
            }),
            timestamp: ts,
        };
        let public = public_event(&with_failures).expect("projects");
        assert_eq!(
            public.attrs.get("failed_infer_calls"),
            Some(&Value::from(3))
        );

        // A pre-t-1334 / infer-less AgentDone projects with no attrs.
        let bare = Event::AgentDone {
            run_id: "run".into(),
            usage: None,
            timestamp: ts,
        };
        assert!(public_event(&bare).expect("projects").attrs.is_empty());
    }

    /// Focused mapping test for the output_validation_failed projection
    /// (docs/TRACE_SCHEMA.md, schema 1.1): error carries the first
    /// validation error, the preview becomes payload_preview, and
    /// attempt/errors ride in attrs.
    #[test]
    fn output_validation_failed_projects_fields() {
        let event = Event::Custom {
            run_id: "run".into(),
            name: "output_validation_failed".into(),
            data: serde_json::json!({
                "attempt": 2,
                "errors": ["$.answer: expected type integer, got string", "$: second"],
                "preview": "{\"answer\":\"x\"}",
            }),
            timestamp: DateTime::<Utc>::UNIX_EPOCH,
        };
        let public = public_event(&event).expect("projects");
        assert_eq!(public.event, "output.validation_failed");
        assert_eq!(public.status, PublicStatus::Failed);
        assert_eq!(
            public.error.as_deref(),
            Some("$.answer: expected type integer, got string")
        );
        assert_eq!(
            public.payload_preview.as_deref(),
            Some("{\"answer\":\"x\"}")
        );
        assert_eq!(public.attrs.get("attempt"), Some(&Value::from(2)));
        assert_eq!(
            public.attrs.get("errors"),
            Some(&serde_json::json!([
                "$.answer: expected type integer, got string",
                "$: second"
            ]))
        );
        assert_eq!(public.op_id, None);
    }

    struct MockProvider {
        responses: Mutex<Vec<Response>>,
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        async fn chat(
            &self,
            _model: &crate::op::Model,
            _tools: &[ToolSpec],
            _messages: &[ChatMessage],
        ) -> Result<Response> {
            self.responses
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow!("mock provider exhausted"))
        }
    }

    fn scripted_response(content: &str) -> Response {
        Response {
            content: content.into(),
            tool_calls: Vec::<ToolCall>::new(),
            finish_reason: Some(crate::op::FinishReason::Stop),
            input_tokens: 7,
            output_tokens: 3,
            total_tokens: 10,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            metadata: Default::default(),
        }
    }

    /// Zero the volatile fields (wall-clock timestamps and measured
    /// durations) so the projection of a live run compares byte-for-byte
    /// against the checked-in golden file. Everything else — run_id, op
    /// ids, effect identity (program hash / effect ids / dynamic paths),
    /// previews, attrs — is deterministic for a fixed program.
    fn normalize(mut event: PublicEvent) -> PublicEvent {
        event.ts = DateTime::<Utc>::UNIX_EPOCH;
        if let Some(duration) = event.attrs.get_mut("duration_ms") {
            *duration = 0.into();
        }
        event
    }

    /// Conformance test (docs/TRACE_SCHEMA.md): drive a small IR program
    /// (Infer then Eval) through the interpreter with a scripted provider,
    /// project the runtime trace to public events, and compare JSONL lines
    /// — including field ordering — against the checked-in golden file.
    #[tokio::test]
    async fn golden_public_projection_of_an_ir_run() -> Result<()> {
        let provider = Arc::new(MockProvider {
            responses: Mutex::new(vec![scripted_response("done")]),
        });
        // Pin the inline message id: ChatMessage::user mints a random uuid,
        // and the id serializes into the program, so a random id would make
        // the program hash (and every effect_id) differ run to run.
        let mut prompt_message = ChatMessage::user("say done");
        prompt_message.id = uuid::Uuid::nil();
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            Block {
                params: vec![],
                instructions: vec![
                    Instr::Infer {
                        out: Var("a".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Inline(vec![prompt_message]),
                        policy: Default::default(),
                    },
                    Instr::Eval {
                        out: Var("b".into()),
                        request: EvalRequest::Shell {
                            command: Expr::Value(Value::String("printf ok".into())),
                        },
                        policy: Default::default(),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("b".into())),
                },
            },
        );
        let machine = Machine {
            program: Program {
                id: ProgramId("public-trace-golden".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            effect_visits: BTreeMap::new(),
            control_path: Default::default(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };
        let trace_path = std::env::temp_dir().join(format!(
            "public-trace-golden-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let trace = TraceLogger::new("golden-run", trace_path.clone());
        let config = SeqConfig {
            approvals: Default::default(),
            tools: Default::default(),
            provider,
            hydration: crate::hydration::SourceRegistry::new(),
            passive_hydration: Default::default(),
            trace: trace.clone(),
            eval: EvalConfig::default(),
            replay: None,
            trace_full_prompt_ir: false,
            trace_full_payloads: false,
            gc: crate::gc::GcMode::None,
            gc_threshold: 0.85,
            gc_log: false,
            gc_timing: crate::gc::GcTiming::Threshold,
            context_budget: 200_000,
            // Pricing for the mock model so the golden pins the cost
            // attrs (t-1334): 7 in + 3 out tokens at $3/$15 per Mtok =
            // 66 micro-USD.
            pricing: {
                let mut table = crate::cost::PricingTable::default();
                table.insert("mock", crate::cost::Pricing::from_usd_per_mtok(3.0, 15.0)?);
                table
            },
        };

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
        assert_eq!(value["stdout"], Value::String("ok".into()));
        trace
            .emit(&Event::AgentDone {
                run_id: "golden-run".into(),
                usage: None,
                timestamp: Utc::now(),
            })
            .await?;

        let events = TraceLogger::read_events(&trace_path).await?;
        let _ = tokio::fs::remove_file(&trace_path).await;
        let actual = events
            .iter()
            .filter_map(public_event)
            .map(normalize)
            .map(|public| serde_json::to_string(&public))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n");
        let golden = include_str!("../testdata/public_trace_golden.jsonl");
        assert_eq!(
            actual,
            golden.trim_end(),
            "public projection diverged from testdata/public_trace_golden.jsonl;\nactual:\n{actual}"
        );
        Ok(())
    }
}
