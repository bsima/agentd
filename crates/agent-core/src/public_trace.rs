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
        Event::AgentDone { run_id, timestamp } => Some(base(
            "run.completed",
            *timestamp,
            run_id,
            None,
            None,
            PublicStatus::Completed,
        )),
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
                Event::AgentDone {
                    run_id: run.into(),
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
        };

        let (value, _machine) = crate::ir_interpreter::run_ir_sequential(&config, machine).await?;
        assert_eq!(value["stdout"], Value::String("ok".into()));
        trace
            .emit(&Event::AgentDone {
                run_id: "golden-run".into(),
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
