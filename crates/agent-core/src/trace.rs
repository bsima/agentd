use crate::ir::EffectLocation;
use crate::op::{Prompt, Response};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::{Span, SpanContext, SpanKind, Status, TraceContextExt, Tracer};
use opentelemetry::{global, KeyValue, SpanId, TraceId};
use opentelemetry_sdk::trace::{IdGenerator, RandomIdGenerator};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "PascalCase")]
pub enum Event {
    InferCall {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt: Option<Prompt>,
        #[serde(default)]
        prompt_preview: String,
        /// Stable IR effect identity (program hash, effect id, site, dynamic
        /// path), attached directly by the IR interpreter so replay never
        /// depends on event adjacency. None for op-layer (non-IR) traces,
        /// which serialize unchanged.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effect: Option<Box<EffectLocation>>,
        timestamp: DateTime<Utc>,
    },
    InferResult {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<Response>,
        #[serde(default)]
        response_preview: String,
        input_tokens: u32,
        output_tokens: u32,
        total_tokens: u32,
        /// Provider-reported cached prompt tokens; absent when the provider
        /// did not report any (t-1334). Old traces deserialize to None.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cached_input_tokens: Option<u32>,
        /// Cost of this call in integer micro-USD (see [`crate::cost`]).
        /// Absent when no pricing was configured for the model — absent
        /// means unknown, never zero.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost_micro_usd: Option<u64>,
        /// The pricing rates (micro-USD per Mtok) used for `cost_micro_usd`,
        /// snapshotted so the trace is self-contained.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pricing: Option<crate::cost::Pricing>,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    /// Terminal failure of an Infer effect (provider error after retries,
    /// replay divergence). Closes the matching InferCall so failed runs stay
    /// inspectable and replayable instead of ending on a dangling call.
    InferError {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        error: String,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    EvalCall {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        command: String,
        /// Exec argv for direct (no-shell) Evals, recorded verbatim — it is
        /// the replay identity for argv requests, so no preview truncation.
        /// `command` then carries a display rendering. Absent for shell
        /// Evals; old traces (which predate argv) deserialize to None.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        argv: Option<Vec<String>>,
        cwd: Option<String>,
        env_policy: String,
        timeout_ms: u64,
        /// Stable IR effect identity; see [`Event::InferCall::effect`].
        /// None for op-layer traces.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effect: Option<Box<EffectLocation>>,
        timestamp: DateTime<Utc>,
    },
    EvalResult {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        command: String,
        result: Value,
        duration_ms: u64,
        truncated_stdout: bool,
        truncated_stderr: bool,
        timestamp: DateTime<Utc>,
    },
    /// Terminal failure of an Eval effect (spawn failure, replay divergence).
    EvalError {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        command: String,
        error: String,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    RetrieveCall {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        query: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kind: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_bytes: Option<usize>,
        /// Stable IR effect identity; see [`Event::InferCall::effect`].
        /// None for op-layer traces.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effect: Option<Box<EffectLocation>>,
        timestamp: DateTime<Utc>,
    },
    RetrieveResult {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        /// Full results, always recorded: replay returns them verbatim.
        results: Value,
        #[serde(default)]
        result_preview: String,
        source_count: usize,
        bytes: usize,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    /// Terminal failure of a Retrieve effect (source error, replay
    /// divergence). Closes the matching RetrieveCall.
    RetrieveError {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        error: String,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    StoreCall {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        sink: String,
        store_op: String,
        /// The id argument (Update/Delete target), recorded so replay can
        /// detect a dynamically-computed id diverging from the recording.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        store_id: Option<String>,
        #[serde(default)]
        item_preview: String,
        /// Hash of the payload so replay can detect same-site divergence
        /// without recording the full item.
        content_hash: String,
        /// Stable IR effect identity; see [`Event::InferCall::effect`].
        /// None for op-layer traces.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effect: Option<Box<EffectLocation>>,
        timestamp: DateTime<Utc>,
    },
    StoreResult {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        sink: String,
        /// Sink-assigned id, always recorded: replay returns it without
        /// touching the sink.
        sink_id: String,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    /// Terminal failure of a Store effect (validation, policy, sink error,
    /// replay divergence). Closes the matching StoreCall.
    StoreError {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        sink: String,
        error: String,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    /// Dispatch of a registered native tool (t-1308.7). `arguments` is the
    /// model-supplied JSON payload, recorded verbatim — it is the replay
    /// identity for the Tool effect, so no preview truncation.
    ToolCall {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        name: String,
        arguments: Value,
        /// Stable IR effect identity; see [`Event::InferCall::effect`].
        /// None for op-layer traces.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effect: Option<Box<EffectLocation>>,
        timestamp: DateTime<Utc>,
    },
    ToolResult {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        name: String,
        /// Full handler result, always recorded: replay returns it verbatim
        /// without invoking the handler.
        result: Value,
        #[serde(default)]
        result_preview: String,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    /// Terminal failure of a Tool effect (handler error, missing
    /// registration, replay divergence). Closes the matching ToolCall.
    ToolError {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        name: String,
        error: String,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    HydrationStart {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        sources: Vec<String>,
        max_bytes: Option<usize>,
        timestamp: DateTime<Utc>,
    },
    HydrationSection {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        source: String,
        kind: String,
        bytes: usize,
        #[serde(default)]
        content_preview: String,
        metadata: Value,
        timestamp: DateTime<Utc>,
    },
    HydrationEnd {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        section_count: usize,
        total_bytes: usize,
        timestamp: DateTime<Utc>,
    },
    ParStart {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        branch_count: usize,
        timestamp: DateTime<Utc>,
    },
    ParEnd {
        run_id: String,
        op_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_op_id: Option<u64>,
        branch_count: usize,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    Checkpoint {
        run_id: String,
        name: String,
        path: Option<String>,
        timestamp: DateTime<Utc>,
    },
    /// The agent loop returned because the soft max-turns ceiling ran out
    /// (t-1133). Budget-overflow events are intentionally distinct:
    /// turn_budget_exhausted is the soft turn ceiling, context_overflow is
    /// the hard provider context-window error, and gc_collect/gc_truncate
    /// are token-budget pressure inside GC.
    TurnBudgetExhausted {
        run_id: String,
        max_turns: usize,
        pending_tool_calls: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        first_tool: Option<String>,
        timestamp: DateTime<Utc>,
    },
    AgentDone {
        run_id: String,
        /// Per-run usage/cost rollup (t-1334): exact integer sums of the
        /// run's recorded InferResult usage and micro-USD costs, stamped by
        /// [`TraceLogger`] when the run recorded at least one InferResult.
        /// Absent on infer-less runs and on traces recorded before t-1334.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<crate::cost::RunUsage>,
        timestamp: DateTime<Utc>,
    },
    Custom {
        run_id: String,
        name: String,
        data: Value,
        timestamp: DateTime<Utc>,
    },
    /// A gated effect reached the approval gate with no decision available
    /// (t-1308.10, DR-7): the run pauses (or an in-process hook is about to
    /// decide). Typed runtime variants — not `Custom` — because approval
    /// outcomes are replay identity like the `ToolCall` triple: replay
    /// consumes them via `IrReplayTrace` to reproduce the pause/decision as
    /// data, so their shape must be compile-checked, unlike the
    /// diagnostics-only `output_validation_failed` Custom event.
    ApprovalRequested {
        run_id: String,
        /// Durable pause id; matches the on-disk pending record.
        pending_id: String,
        /// `"eval"` or `"store"` (see [`crate::approval::ApprovalKind`]).
        kind: String,
        /// The gated request preview: for Eval `{command, argv}`; for Store
        /// `{sink, op, id, item_preview, content_hash}`. Replay checks this
        /// against the current request for denied effects (approved effects
        /// are checked by their own `*Call` identity).
        request: Value,
        effect: Box<EffectLocation>,
        timestamp: DateTime<Utc>,
    },
    /// The decision on a gated effect, emitted at the effect site by the
    /// process that consumes it (hook decisions inline; resume decisions
    /// when the checkpointed machine re-reaches the effect). Carries the
    /// decision and resolver metadata; replay reproduces it as data.
    ApprovalResolved {
        run_id: String,
        pending_id: String,
        effect_id: String,
        kind: String,
        /// `"approved"` or `"denied"`.
        decision: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolved_by: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        timestamp: DateTime<Utc>,
    },
}

impl Event {
    pub fn run_id(&self) -> &str {
        match self {
            Self::InferCall { run_id, .. }
            | Self::InferResult { run_id, .. }
            | Self::InferError { run_id, .. }
            | Self::EvalCall { run_id, .. }
            | Self::EvalError { run_id, .. }
            | Self::EvalResult { run_id, .. }
            | Self::RetrieveCall { run_id, .. }
            | Self::RetrieveResult { run_id, .. }
            | Self::RetrieveError { run_id, .. }
            | Self::StoreCall { run_id, .. }
            | Self::StoreResult { run_id, .. }
            | Self::StoreError { run_id, .. }
            | Self::ToolCall { run_id, .. }
            | Self::ToolResult { run_id, .. }
            | Self::ToolError { run_id, .. }
            | Self::HydrationStart { run_id, .. }
            | Self::HydrationSection { run_id, .. }
            | Self::HydrationEnd { run_id, .. }
            | Self::ParStart { run_id, .. }
            | Self::ParEnd { run_id, .. }
            | Self::Checkpoint { run_id, .. }
            | Self::TurnBudgetExhausted { run_id, .. }
            | Self::AgentDone { run_id, .. }
            | Self::Custom { run_id, .. }
            | Self::ApprovalRequested { run_id, .. }
            | Self::ApprovalResolved { run_id, .. } => run_id,
        }
    }

    pub fn op_id(&self) -> Option<u64> {
        match self {
            Self::InferCall { op_id, .. }
            | Self::InferResult { op_id, .. }
            | Self::InferError { op_id, .. }
            | Self::EvalCall { op_id, .. }
            | Self::EvalError { op_id, .. }
            | Self::EvalResult { op_id, .. }
            | Self::RetrieveCall { op_id, .. }
            | Self::RetrieveResult { op_id, .. }
            | Self::RetrieveError { op_id, .. }
            | Self::StoreCall { op_id, .. }
            | Self::StoreResult { op_id, .. }
            | Self::StoreError { op_id, .. }
            | Self::ToolCall { op_id, .. }
            | Self::ToolResult { op_id, .. }
            | Self::ToolError { op_id, .. }
            | Self::HydrationStart { op_id, .. }
            | Self::HydrationSection { op_id, .. }
            | Self::HydrationEnd { op_id, .. }
            | Self::ParStart { op_id, .. }
            | Self::ParEnd { op_id, .. } => Some(*op_id),
            Self::Checkpoint { .. }
            | Self::TurnBudgetExhausted { .. }
            | Self::AgentDone { .. }
            | Self::Custom { .. }
            | Self::ApprovalRequested { .. }
            | Self::ApprovalResolved { .. } => None,
        }
    }

    pub fn parent_op_id(&self) -> Option<u64> {
        match self {
            Self::InferCall { parent_op_id, .. }
            | Self::InferResult { parent_op_id, .. }
            | Self::InferError { parent_op_id, .. }
            | Self::EvalCall { parent_op_id, .. }
            | Self::EvalError { parent_op_id, .. }
            | Self::EvalResult { parent_op_id, .. }
            | Self::RetrieveCall { parent_op_id, .. }
            | Self::RetrieveResult { parent_op_id, .. }
            | Self::RetrieveError { parent_op_id, .. }
            | Self::StoreCall { parent_op_id, .. }
            | Self::StoreResult { parent_op_id, .. }
            | Self::StoreError { parent_op_id, .. }
            | Self::ToolCall { parent_op_id, .. }
            | Self::ToolResult { parent_op_id, .. }
            | Self::ToolError { parent_op_id, .. }
            | Self::HydrationStart { parent_op_id, .. }
            | Self::HydrationSection { parent_op_id, .. }
            | Self::HydrationEnd { parent_op_id, .. }
            | Self::ParStart { parent_op_id, .. }
            | Self::ParEnd { parent_op_id, .. } => *parent_op_id,
            Self::Checkpoint { .. }
            | Self::TurnBudgetExhausted { .. }
            | Self::AgentDone { .. }
            | Self::Custom { .. }
            | Self::ApprovalRequested { .. }
            | Self::ApprovalResolved { .. } => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::InferCall { .. } | Self::InferResult { .. } | Self::InferError { .. } => "Infer",
            Self::EvalCall { .. } | Self::EvalResult { .. } | Self::EvalError { .. } => "Eval",
            Self::RetrieveCall { .. }
            | Self::RetrieveResult { .. }
            | Self::RetrieveError { .. } => "Retrieve",
            Self::StoreCall { .. } | Self::StoreResult { .. } | Self::StoreError { .. } => "Store",
            Self::ToolCall { .. } | Self::ToolResult { .. } | Self::ToolError { .. } => "Tool",
            Self::HydrationStart { .. } | Self::HydrationEnd { .. } => "Hydration",
            Self::HydrationSection { .. } => "HydrationSection",
            Self::ParStart { .. } | Self::ParEnd { .. } => "Par",
            Self::Checkpoint { .. } => "Checkpoint",
            Self::TurnBudgetExhausted { .. } => "turn_budget_exhausted",
            Self::AgentDone { .. } => "AgentDone",
            Self::ApprovalRequested { .. } => "ApprovalRequested",
            Self::ApprovalResolved { .. } => "ApprovalResolved",
            Self::Custom { name, .. } => match name.as_str() {
                "agent_error" => "agent_error",
                "agent_response" => "agent_response",
                "gc_collect" => "gc_collect",
                "gc_truncate" => "gc_truncate",
                "context_overflow" => "context_overflow",
                "output_contract" => "output_contract",
                "output_validation_failed" => "output_validation_failed",
                _ => "Custom",
            },
        }
    }

    fn is_start(&self) -> bool {
        matches!(
            self,
            Self::InferCall { .. }
                | Self::EvalCall { .. }
                | Self::RetrieveCall { .. }
                | Self::StoreCall { .. }
                | Self::ToolCall { .. }
                | Self::HydrationStart { .. }
                | Self::ParStart { .. }
        )
    }

    fn is_end(&self) -> bool {
        matches!(
            self,
            Self::InferResult { .. }
                | Self::InferError { .. }
                | Self::EvalResult { .. }
                | Self::EvalError { .. }
                | Self::RetrieveResult { .. }
                | Self::RetrieveError { .. }
                | Self::StoreResult { .. }
                | Self::StoreError { .. }
                | Self::ToolResult { .. }
                | Self::ToolError { .. }
                | Self::HydrationEnd { .. }
                | Self::ParEnd { .. }
        )
    }

    fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::InferCall { timestamp, .. }
            | Self::InferResult { timestamp, .. }
            | Self::InferError { timestamp, .. }
            | Self::EvalCall { timestamp, .. }
            | Self::EvalError { timestamp, .. }
            | Self::EvalResult { timestamp, .. }
            | Self::RetrieveCall { timestamp, .. }
            | Self::RetrieveResult { timestamp, .. }
            | Self::RetrieveError { timestamp, .. }
            | Self::StoreCall { timestamp, .. }
            | Self::StoreResult { timestamp, .. }
            | Self::StoreError { timestamp, .. }
            | Self::ToolCall { timestamp, .. }
            | Self::ToolResult { timestamp, .. }
            | Self::ToolError { timestamp, .. }
            | Self::HydrationStart { timestamp, .. }
            | Self::HydrationSection { timestamp, .. }
            | Self::HydrationEnd { timestamp, .. }
            | Self::ParStart { timestamp, .. }
            | Self::ParEnd { timestamp, .. }
            | Self::Checkpoint { timestamp, .. }
            | Self::TurnBudgetExhausted { timestamp, .. }
            | Self::AgentDone { timestamp, .. }
            | Self::Custom { timestamp, .. }
            | Self::ApprovalRequested { timestamp, .. }
            | Self::ApprovalResolved { timestamp, .. } => *timestamp,
        }
    }

    fn otel_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("agent.run_id", self.run_id().to_string()),
            KeyValue::new("agent.event", self.name()),
        ];
        if let Some(op_id) = self.op_id() {
            attrs.push(KeyValue::new("agent.op_id", op_id as i64));
        }
        match self {
            Self::InferCall {
                model,
                prompt_preview,
                ..
            } => {
                attrs.push(KeyValue::new("gen_ai.request.model", model.clone()));
                attrs.push(KeyValue::new(
                    "agent.prompt_preview",
                    prompt_preview.clone(),
                ));
            }
            Self::InferResult {
                input_tokens,
                output_tokens,
                total_tokens,
                cached_input_tokens,
                cost_micro_usd,
                duration_ms,
                response_preview,
                ..
            } => {
                attrs.push(KeyValue::new(
                    "gen_ai.usage.input_tokens",
                    i64::from(*input_tokens),
                ));
                attrs.push(KeyValue::new(
                    "gen_ai.usage.output_tokens",
                    i64::from(*output_tokens),
                ));
                attrs.push(KeyValue::new(
                    "gen_ai.usage.total_tokens",
                    i64::from(*total_tokens),
                ));
                if let Some(cached) = cached_input_tokens {
                    attrs.push(KeyValue::new(
                        "gen_ai.usage.cached_input_tokens",
                        i64::from(*cached),
                    ));
                }
                if let Some(cost) = cost_micro_usd {
                    attrs.push(KeyValue::new("agent.cost_micro_usd", *cost as i64));
                }
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
                attrs.push(KeyValue::new(
                    "agent.response_preview",
                    response_preview.clone(),
                ));
            }
            Self::EvalCall {
                command,
                cwd,
                env_policy,
                timeout_ms,
                ..
            } => {
                attrs.push(KeyValue::new("tool.name", tool_name(command)));
                attrs.push(KeyValue::new("command", command.clone()));
                if let Some(cwd) = cwd {
                    attrs.push(KeyValue::new("cwd", cwd.clone()));
                }
                attrs.push(KeyValue::new("agent.env_policy", env_policy.clone()));
                attrs.push(KeyValue::new("timeout_ms", *timeout_ms as i64));
            }
            Self::EvalResult {
                result,
                duration_ms,
                truncated_stdout,
                truncated_stderr,
                ..
            } => {
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
                if let Some(ok) = result.get("ok").and_then(Value::as_bool) {
                    attrs.push(KeyValue::new("ok", ok));
                }
                if let Some(exit_code) = result.get("status").and_then(Value::as_i64) {
                    attrs.push(KeyValue::new("exit_code", exit_code));
                }
                attrs.push(KeyValue::new("truncated_stdout", *truncated_stdout));
                attrs.push(KeyValue::new("truncated_stderr", *truncated_stderr));
            }
            Self::InferError {
                error, duration_ms, ..
            } => {
                attrs.push(KeyValue::new("error", error.clone()));
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
            }
            Self::EvalError {
                command,
                error,
                duration_ms,
                ..
            } => {
                attrs.push(KeyValue::new("tool.name", tool_name(command)));
                attrs.push(KeyValue::new("command", command.clone()));
                attrs.push(KeyValue::new("error", error.clone()));
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
            }
            Self::RetrieveCall { query, kind, .. } => {
                attrs.push(KeyValue::new("query", query.clone()));
                if let Some(kind) = kind {
                    attrs.push(KeyValue::new("agent.source_kind", kind.clone()));
                }
            }
            Self::RetrieveResult {
                source_count,
                bytes,
                duration_ms,
                ..
            } => {
                attrs.push(KeyValue::new("source_count", *source_count as i64));
                attrs.push(KeyValue::new("bytes", *bytes as i64));
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
            }
            Self::RetrieveError {
                error, duration_ms, ..
            } => {
                attrs.push(KeyValue::new("error", error.clone()));
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
            }
            Self::StoreCall { sink, store_op, .. } => {
                attrs.push(KeyValue::new("agent.sink", sink.clone()));
                attrs.push(KeyValue::new("agent.store_op", store_op.clone()));
            }
            Self::StoreResult {
                sink,
                sink_id,
                duration_ms,
                ..
            } => {
                attrs.push(KeyValue::new("agent.sink", sink.clone()));
                attrs.push(KeyValue::new("agent.sink_id", sink_id.clone()));
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
            }
            Self::StoreError {
                sink,
                error,
                duration_ms,
                ..
            } => {
                attrs.push(KeyValue::new("agent.sink", sink.clone()));
                attrs.push(KeyValue::new("error", error.clone()));
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
            }
            Self::ToolCall {
                name, arguments, ..
            } => {
                attrs.push(KeyValue::new("tool.name", name.clone()));
                attrs.push(KeyValue::new(
                    "agent.arguments_preview",
                    preview(&arguments.to_string(), 512),
                ));
            }
            Self::ToolResult {
                name,
                result_preview,
                duration_ms,
                ..
            } => {
                attrs.push(KeyValue::new("tool.name", name.clone()));
                attrs.push(KeyValue::new(
                    "agent.result_preview",
                    result_preview.clone(),
                ));
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
            }
            Self::ToolError {
                name,
                error,
                duration_ms,
                ..
            } => {
                attrs.push(KeyValue::new("tool.name", name.clone()));
                attrs.push(KeyValue::new("error", error.clone()));
                attrs.push(KeyValue::new("duration_ms", *duration_ms as i64));
            }
            Self::HydrationStart {
                sources, max_bytes, ..
            } => {
                attrs.push(KeyValue::new("agent.sources", sources.join(",")));
                if let Some(max_bytes) = max_bytes {
                    attrs.push(KeyValue::new("agent.max_bytes", *max_bytes as i64));
                }
            }
            Self::HydrationSection {
                source,
                kind,
                bytes,
                content_preview,
                ..
            } => {
                attrs.push(KeyValue::new("agent.source", source.clone()));
                attrs.push(KeyValue::new("agent.kind", kind.clone()));
                attrs.push(KeyValue::new("agent.bytes", *bytes as i64));
                attrs.push(KeyValue::new(
                    "agent.content_preview",
                    content_preview.clone(),
                ));
            }
            Self::HydrationEnd {
                section_count,
                total_bytes,
                ..
            } => {
                attrs.push(KeyValue::new("agent.section_count", *section_count as i64));
                attrs.push(KeyValue::new("agent.total_bytes", *total_bytes as i64));
            }
            Self::ParStart { branch_count, .. } | Self::ParEnd { branch_count, .. } => {
                attrs.push(KeyValue::new("branch_count", *branch_count as i64));
            }
            Self::Checkpoint { name, path, .. } => {
                attrs.push(KeyValue::new("agent.checkpoint", name.clone()));
                if let Some(path) = path {
                    attrs.push(KeyValue::new("agent.path", path.clone()));
                }
            }
            Self::TurnBudgetExhausted {
                max_turns,
                pending_tool_calls,
                first_tool,
                ..
            } => {
                attrs.push(KeyValue::new("agent.max_turns", *max_turns as i64));
                attrs.push(KeyValue::new(
                    "agent.pending_tool_calls",
                    *pending_tool_calls as i64,
                ));
                if let Some(first_tool) = first_tool {
                    attrs.push(KeyValue::new("agent.first_tool", first_tool.clone()));
                }
            }
            Self::Custom { name, data, .. } => {
                attrs.push(KeyValue::new("agent.custom_name", name.clone()));
                attrs.push(KeyValue::new("agent.data", data.to_string()));
            }
            Self::ApprovalRequested {
                pending_id,
                kind,
                request,
                ..
            } => {
                attrs.push(KeyValue::new("agent.pending_id", pending_id.clone()));
                attrs.push(KeyValue::new("agent.approval_kind", kind.clone()));
                attrs.push(KeyValue::new(
                    "agent.request_preview",
                    preview(&request.to_string(), 512),
                ));
            }
            Self::ApprovalResolved {
                pending_id,
                kind,
                decision,
                resolved_by,
                reason,
                ..
            } => {
                attrs.push(KeyValue::new("agent.pending_id", pending_id.clone()));
                attrs.push(KeyValue::new("agent.approval_kind", kind.clone()));
                attrs.push(KeyValue::new("agent.decision", decision.clone()));
                if let Some(resolved_by) = resolved_by {
                    attrs.push(KeyValue::new("agent.resolved_by", resolved_by.clone()));
                }
                if let Some(reason) = reason {
                    attrs.push(KeyValue::new("agent.reason", reason.clone()));
                }
            }
            Self::AgentDone { .. } => {}
        }
        attrs
    }
}

fn tool_name(command: &str) -> String {
    command
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}

fn eval_status(event: &Event) -> Status {
    match event {
        Event::EvalResult { result, .. }
            if result.get("ok").and_then(Value::as_bool) == Some(false) =>
        {
            Status::error("eval failed")
        }
        Event::InferError { error, .. } | Event::EvalError { error, .. } => {
            Status::error(error.clone())
        }
        _ => Status::Ok,
    }
}

fn custom_event_kvs(event: &Event) -> Option<Vec<KeyValue>> {
    let Event::Custom { data, .. } = event else {
        return None;
    };
    let object = data.as_object()?;
    Some(
        object
            .iter()
            .filter_map(|(key, value)| match value {
                Value::Bool(value) => Some(KeyValue::new(key.clone(), *value)),
                Value::Number(value) => value
                    .as_i64()
                    .map(|value| KeyValue::new(key.clone(), value))
                    .or_else(|| {
                        value
                            .as_f64()
                            .map(|value| KeyValue::new(key.clone(), value))
                    }),
                Value::String(value) => Some(KeyValue::new(key.clone(), value.clone())),
                _ => Some(KeyValue::new(key.clone(), value.to_string())),
            })
            .collect(),
    )
}

#[async_trait]
pub trait TraceSink: Send + Sync {
    async fn emit(&self, event: &Event) -> Result<()>;
}

#[derive(Clone)]
pub struct JsonlTraceSink {
    path: PathBuf,
    mirror_stdout: bool,
    /// Lazily-opened append handle, shared across clones so a long session
    /// does not reopen the trace file on every event. The mutex also
    /// serializes writers, keeping each JSONL line atomic.
    file: Arc<tokio::sync::Mutex<Option<tokio::fs::File>>>,
}

impl JsonlTraceSink {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            mirror_stdout: false,
            file: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    pub fn mirror_stdout(mut self, mirror_stdout: bool) -> Self {
        self.mirror_stdout = mirror_stdout;
        self
    }
}

#[async_trait]
impl TraceSink for JsonlTraceSink {
    async fn emit(&self, event: &Event) -> Result<()> {
        let line = serde_json::to_string(event)?;
        let mut guard = self.file.lock().await;
        if guard.is_none() {
            if let Some(parent) = self.path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            *guard = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)
                    .await?,
            );
        }
        let file = guard.as_mut().expect("handle opened above");
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        drop(guard);
        if self.mirror_stdout {
            let mut stdout = tokio::io::stdout();
            stdout.write_all(line.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
        Ok(())
    }
}

thread_local! {
    static NEXT_SPAN_ID: RefCell<Option<u64>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone, Default)]
pub struct AgentIdGenerator {
    random: RandomIdGenerator,
}

impl AgentIdGenerator {
    fn use_next_span_id(span_id: u64) {
        NEXT_SPAN_ID.with(|next| *next.borrow_mut() = Some(span_id.max(1)));
    }
}

impl IdGenerator for AgentIdGenerator {
    fn new_trace_id(&self) -> TraceId {
        self.random.new_trace_id()
    }

    fn new_span_id(&self) -> SpanId {
        NEXT_SPAN_ID
            .with(|next| next.borrow_mut().take())
            .map(SpanId::from)
            .unwrap_or_else(|| self.random.new_span_id())
    }
}

struct OpenSpan {
    span: BoxedSpan,
    context: SpanContext,
}

#[derive(Clone, Default)]
pub struct TraceContextEnv {
    vars: Arc<Mutex<BTreeMap<String, String>>>,
}

impl TraceContextEnv {
    pub fn set(&self, name: impl Into<String>, value: impl Into<String>) {
        self.vars.lock().unwrap().insert(name.into(), value.into());
    }

    pub fn remove(&self, name: &str) {
        self.vars.lock().unwrap().remove(name);
    }

    pub fn snapshot(&self) -> BTreeMap<String, String> {
        self.vars.lock().unwrap().clone()
    }
}

#[derive(Default)]
pub struct OtelTraceSink {
    spans: Mutex<HashMap<u64, OpenSpan>>,
    open_stack: Mutex<Vec<u64>>,
    eval_attempts: Mutex<HashMap<String, u64>>,
    context_env: TraceContextEnv,
}

impl OtelTraceSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_context_env(context_env: TraceContextEnv) -> Self {
        Self {
            context_env,
            ..Self::default()
        }
    }

    fn start_span(&self, event: &Event) {
        let Some(op_id) = event.op_id() else {
            return;
        };
        let tracer = global::tracer("agentd");
        let mut attributes = event.otel_attributes();
        if let Event::EvalCall { command, .. } = event {
            let mut attempts = self.eval_attempts.lock().unwrap();
            let attempt = attempts.entry(command.clone()).or_insert(0);
            *attempt += 1;
            attributes.push(KeyValue::new("attempt", *attempt as i64));
            attributes.push(KeyValue::new(
                "retry.count",
                attempt.saturating_sub(1) as i64,
            ));
        }
        let builder = tracer
            .span_builder(event.name())
            .with_kind(SpanKind::Internal)
            .with_start_time(event.timestamp())
            .with_attributes(attributes);
        let parent_context = self.parent_context_for(event);
        AgentIdGenerator::use_next_span_id(op_id);
        let span = match parent_context {
            Some(parent_context) => tracer.build_with_context(builder, &parent_context),
            None => tracer.build(builder),
        };
        let context = span.span_context().clone();
        set_trace_context_env(&self.context_env, &context);
        self.spans
            .lock()
            .unwrap()
            .insert(op_id, OpenSpan { span, context });
        self.open_stack.lock().unwrap().push(op_id);
    }

    fn parent_context_for(&self, event: &Event) -> Option<opentelemetry::Context> {
        let parent_op_id = event.parent_op_id()?;
        self.spans.lock().unwrap().get(&parent_op_id).map(|parent| {
            opentelemetry::Context::current().with_remote_span_context(parent.context.clone())
        })
    }

    fn finish_span(&self, event: &Event) {
        let Some(op_id) = event.op_id() else {
            self.emit_instant_event(event);
            return;
        };
        self.open_stack.lock().unwrap().retain(|id| *id != op_id);
        let mut spans = self.spans.lock().unwrap();
        if let Some(mut open_span) = spans.remove(&op_id) {
            open_span.span.set_attributes(event.otel_attributes());
            open_span.span.set_status(eval_status(event));
            open_span.span.end_with_timestamp(event.timestamp().into());
        } else {
            drop(spans);
            self.emit_instant_event(event);
        }
    }

    fn emit_instant_event(&self, event: &Event) {
        if self.attach_custom_event_to_current_span(event) {
            return;
        }
        let tracer = global::tracer("agentd");
        let builder = tracer
            .span_builder(event.name())
            .with_kind(SpanKind::Internal)
            .with_start_time(event.timestamp())
            .with_attributes(event.otel_attributes());
        let mut span = tracer.build(builder);
        span.set_status(Status::Ok);
        span.end_with_timestamp(event.timestamp().into());
    }

    fn attach_custom_event_to_current_span(&self, event: &Event) -> bool {
        let Some(kvs) = custom_event_kvs(event) else {
            return false;
        };
        let Some(parent_op_id) = self.open_stack.lock().unwrap().last().copied() else {
            return false;
        };
        let mut spans = self.spans.lock().unwrap();
        let Some(open_span) = spans.get_mut(&parent_op_id) else {
            return false;
        };
        open_span.span.set_attributes(kvs);
        true
    }
}

fn set_trace_context_env(context_env: &TraceContextEnv, context: &SpanContext) {
    if context.is_valid() {
        context_env.set(
            "TRACEPARENT",
            format!(
                "00-{:032x}-{:016x}-{:02x}",
                context.trace_id(),
                context.span_id(),
                context.trace_flags().to_u8()
            ),
        );
        let tracestate = context.trace_state().header();
        if tracestate.is_empty() {
            context_env.remove("TRACESTATE");
        } else {
            context_env.set("TRACESTATE", tracestate);
        }
    }
}

#[async_trait]
impl TraceSink for OtelTraceSink {
    async fn emit(&self, event: &Event) -> Result<()> {
        if event.is_start() {
            self.start_span(event);
        } else if event.is_end() {
            self.finish_span(event);
        } else {
            self.emit_instant_event(event);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct TraceLogger {
    run_id: String,
    path: PathBuf,
    next_op_id: Arc<AtomicU64>,
    sinks: Arc<Vec<Arc<dyn TraceSink>>>,
    context_env: TraceContextEnv,
    /// Running usage/cost rollup (t-1334): every InferResult emitted through
    /// this logger (or a clone) is folded in, and an `AgentDone` emitted
    /// without a usage payload is stamped with the totals. Sums are exact
    /// integer arithmetic over the recorded per-event values, so replayed
    /// runs (which re-emit the recorded usage/cost) reproduce the original
    /// rollup exactly.
    usage: Arc<Mutex<crate::cost::RunUsage>>,
}

impl TraceLogger {
    pub fn new(run_id: impl Into<String>, path: PathBuf) -> Self {
        let sink = JsonlTraceSink::new(path.clone());
        Self::with_sinks(run_id, path, vec![Arc::new(sink)])
    }

    pub fn with_sinks(
        run_id: impl Into<String>,
        path: PathBuf,
        sinks: Vec<Arc<dyn TraceSink>>,
    ) -> Self {
        Self::with_sinks_and_context(run_id, path, sinks, TraceContextEnv::default())
    }

    pub fn with_sinks_and_context(
        run_id: impl Into<String>,
        path: PathBuf,
        sinks: Vec<Arc<dyn TraceSink>>,
        context_env: TraceContextEnv,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            path,
            next_op_id: Arc::new(AtomicU64::new(1)),
            sinks: Arc::new(sinks),
            context_env,
            usage: Arc::new(Mutex::new(crate::cost::RunUsage::default())),
        }
    }

    pub fn mirror_stdout(mut self, mirror_stdout: bool) -> Self {
        let sink = JsonlTraceSink::new(self.path.clone()).mirror_stdout(mirror_stdout);
        self.sinks = Arc::new(vec![Arc::new(sink)]);
        self
    }

    pub fn trace_context_env(&self) -> BTreeMap<String, String> {
        self.context_env.snapshot()
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn next_op_id(&self) -> u64 {
        self.next_op_id.fetch_add(1, Ordering::Relaxed)
    }

    pub async fn emit(&self, event: &Event) -> Result<()> {
        let enriched = self.fold_usage(event);
        let event = enriched.as_ref().unwrap_or(event);
        for sink in self.sinks.iter() {
            sink.emit(event).await?;
        }
        Ok(())
    }

    /// Fold InferResult usage/cost into the run rollup; stamp the rollup
    /// onto an `AgentDone` that does not already carry one. Returns the
    /// replacement event when the input event was enriched. AgentDone stays
    /// untouched when the run recorded no InferResults (absent means
    /// absent) or when the caller supplied its own rollup.
    fn fold_usage(&self, event: &Event) -> Option<Event> {
        match event {
            Event::InferResult {
                input_tokens,
                output_tokens,
                total_tokens,
                cached_input_tokens,
                cost_micro_usd,
                ..
            } => {
                self.usage.lock().unwrap().observe_infer(
                    *input_tokens,
                    *output_tokens,
                    *total_tokens,
                    *cached_input_tokens,
                    *cost_micro_usd,
                );
                None
            }
            Event::AgentDone {
                run_id,
                usage: None,
                timestamp,
            } => {
                let usage = self.usage.lock().unwrap().clone();
                (!usage.is_empty()).then(|| Event::AgentDone {
                    run_id: run_id.clone(),
                    usage: Some(usage),
                    timestamp: *timestamp,
                })
            }
            _ => None,
        }
    }

    pub async fn read_events(path: impl AsRef<Path>) -> Result<Vec<Event>> {
        let file = tokio::fs::File::open(path).await?;
        let mut lines = tokio::io::BufReader::new(file).lines();
        let mut events = Vec::new();
        while let Some(line) = lines.next_line().await? {
            if !line.trim().is_empty() {
                events.push(serde_json::from_str(&line)?);
            }
        }
        Ok(events)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraceSummary {
    pub total_tokens: u32,
    pub infer_calls: usize,
    pub eval_calls: usize,
    pub retrieve_calls: usize,
    pub store_calls: usize,
}

impl TraceSummary {
    pub fn from_events(events: &[Event]) -> Self {
        let mut summary = Self::default();
        for event in events {
            match event {
                Event::InferCall { .. } => summary.infer_calls += 1,
                Event::InferResult { total_tokens, .. } => summary.total_tokens += *total_tokens,
                Event::EvalCall { .. } => summary.eval_calls += 1,
                Event::RetrieveCall { .. } => summary.retrieve_calls += 1,
                Event::StoreCall { .. } => summary.store_calls += 1,
                _ => {}
            }
        }
        summary
    }
}

pub fn preview(input: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in input.chars().take(max_chars) {
        out.push(ch);
    }
    if input.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use uuid::Uuid;

    /// Op-layer (non-IR) interpreter traces have no IR location: the
    /// optional `effect` field must stay absent from their JSON so existing
    /// op-trace consumers and fixtures are unaffected (t-1057).
    #[test]
    fn op_layer_call_events_serialize_without_effect_field() -> Result<()> {
        let infer = Event::InferCall {
            run_id: "run".into(),
            op_id: 1,
            parent_op_id: None,
            model: "mock".into(),
            prompt: None,
            prompt_preview: "hi".into(),
            effect: None,
            timestamp: Utc::now(),
        };
        let eval = Event::EvalCall {
            run_id: "run".into(),
            op_id: 2,
            parent_op_id: None,
            command: "true".into(),
            argv: None,
            cwd: None,
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: None,
            timestamp: Utc::now(),
        };
        for event in [infer, eval] {
            let json = serde_json::to_string(&event)?;
            assert!(
                !json.contains("\"effect\""),
                "op-layer trace line grew an effect field: {json}"
            );
            // And a line without the field round-trips (old traces parse).
            assert_eq!(serde_json::from_str::<Event>(&json)?, event);
        }
        Ok(())
    }

    /// Shell EvalCalls (and every trace written before argv Evals existed)
    /// carry no `argv` field: the JSON stays byte-compatible and old lines
    /// deserialize to `argv: None`. Argv EvalCalls record the argv verbatim
    /// and round-trip.
    #[test]
    fn eval_call_argv_field_is_absent_for_shell_and_round_trips_for_argv() -> Result<()> {
        let shell = Event::EvalCall {
            run_id: "run".into(),
            op_id: 1,
            parent_op_id: None,
            command: "printf ok".into(),
            argv: None,
            cwd: None,
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: None,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&shell)?;
        assert!(!json.contains("\"argv\""), "{json}");
        assert_eq!(serde_json::from_str::<Event>(&json)?, shell);

        let argv = Event::EvalCall {
            run_id: "run".into(),
            op_id: 2,
            parent_op_id: None,
            command: "some-tool call id-1 'hello world'".into(),
            argv: Some(vec![
                "some-tool".into(),
                "call".into(),
                "id-1".into(),
                "hello world".into(),
            ]),
            cwd: None,
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: None,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&argv)?;
        assert!(json.contains("\"argv\""), "{json}");
        assert_eq!(serde_json::from_str::<Event>(&json)?, argv);
        Ok(())
    }

    /// IR traces carry the effect identity inline on call events and it
    /// round-trips through JSONL.
    #[test]
    fn ir_call_events_round_trip_effect_identity() -> Result<()> {
        let site = crate::ir::EffectSite {
            block: crate::ir::BlockId(0),
            instruction_index: 0,
        };
        let location = crate::ir::effect_location(
            crate::ir::ProgramHash("sha256:test".into()),
            crate::ir::EffectKind::Infer,
            site,
            crate::ir::DynamicPath::at_entry(0),
        )?;
        let event = Event::InferCall {
            run_id: "run".into(),
            op_id: 1,
            parent_op_id: None,
            model: "mock".into(),
            prompt: None,
            prompt_preview: "hi".into(),
            effect: Some(Box::new(location.clone())),
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&event)?;
        assert!(json.contains(&location.effect_id.0), "{json}");
        assert_eq!(serde_json::from_str::<Event>(&json)?, event);
        Ok(())
    }

    /// Serde back-compat for the t-1334 cost fields: an InferResult line
    /// written before cost accounting (no cached/cost/pricing keys, and a
    /// Response payload without them) still deserializes; a costless new
    /// event serializes without the keys; a costed event round-trips.
    #[test]
    fn infer_result_cost_fields_are_optional_and_round_trip() -> Result<()> {
        // Verbatim pre-t-1334 trace line shape.
        let old_line = r#"{"event":"InferResult","run_id":"run","op_id":2,"response":{"content":"hi","tool_calls":[],"finish_reason":"stop","input_tokens":3,"output_tokens":4,"total_tokens":7},"response_preview":"hi","input_tokens":3,"output_tokens":4,"total_tokens":7,"duration_ms":1,"timestamp":"2026-05-29T00:00:00Z"}"#;
        let event: Event = serde_json::from_str(old_line)?;
        let Event::InferResult {
            cached_input_tokens,
            cost_micro_usd,
            pricing,
            response: Some(response),
            ..
        } = &event
        else {
            panic!("expected InferResult, got {event:?}");
        };
        assert_eq!(*cached_input_tokens, None);
        assert_eq!(*cost_micro_usd, None);
        assert_eq!(*pricing, None);
        assert_eq!(response.cost_micro_usd, None);
        // A costless event stays byte-compatible: no new keys appear.
        let json = serde_json::to_string(&event)?;
        for key in ["cached_input_tokens", "cost_micro_usd", "pricing"] {
            assert!(!json.contains(key), "costless line grew {key}: {json}");
        }
        assert_eq!(serde_json::from_str::<Event>(&json)?, event);

        // A costed event round-trips with the recorded snapshot intact.
        let pricing = crate::cost::Pricing {
            input_micro_usd_per_mtok: 3_000_000,
            output_micro_usd_per_mtok: 15_000_000,
        };
        let costed = Event::InferResult {
            run_id: "run".into(),
            op_id: 3,
            parent_op_id: None,
            response: None,
            response_preview: "ok".into(),
            input_tokens: 7,
            output_tokens: 3,
            total_tokens: 10,
            cached_input_tokens: Some(2),
            cost_micro_usd: Some(66),
            pricing: Some(pricing),
            duration_ms: 1,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&costed)?;
        assert!(json.contains("\"cost_micro_usd\":66"), "{json}");
        assert!(
            json.contains("\"input_micro_usd_per_mtok\":3000000"),
            "{json}"
        );
        assert_eq!(serde_json::from_str::<Event>(&json)?, costed);

        // Pre-t-1334 AgentDone lines (no usage) still parse.
        let old_done = r#"{"event":"AgentDone","run_id":"run","timestamp":"2026-05-29T00:00:00Z"}"#;
        let done: Event = serde_json::from_str(old_done)?;
        assert!(matches!(done, Event::AgentDone { usage: None, .. }));
        Ok(())
    }

    #[derive(Default)]
    struct RecordingSink {
        events: Mutex<Vec<Event>>,
    }

    #[async_trait]
    impl TraceSink for RecordingSink {
        async fn emit(&self, event: &Event) -> Result<()> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn done_event(run_id: &str) -> Event {
        Event::AgentDone {
            run_id: run_id.into(),
            usage: None,
            timestamp: Utc::now(),
        }
    }

    fn install_in_memory_tracer() -> (
        opentelemetry_sdk::trace::InMemorySpanExporter,
        opentelemetry_sdk::trace::SdkTracerProvider,
    ) {
        let exporter = opentelemetry_sdk::trace::InMemorySpanExporterBuilder::new().build();
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_id_generator(AgentIdGenerator::default())
            .with_span_processor(opentelemetry_sdk::trace::SimpleSpanProcessor::new(
                exporter.clone(),
            ))
            .build();
        global::set_tracer_provider(provider.clone());
        (exporter, provider)
    }

    fn attr_value(span: &opentelemetry_sdk::trace::SpanData, key: &str) -> Option<String> {
        span.attributes
            .iter()
            .find(|attr| attr.key.as_str() == key)
            .map(|attr| attr.value.to_string())
    }

    #[tokio::test]
    async fn trace_logger_emits_to_all_sinks_and_preserves_jsonl_readback() -> Result<()> {
        let run_id = Uuid::new_v4().to_string();
        let path = std::env::temp_dir().join(format!("trace-sink-test-{run_id}.jsonl"));
        let recording = Arc::new(RecordingSink::default());
        let logger = TraceLogger::with_sinks(
            run_id.clone(),
            path.clone(),
            vec![
                Arc::new(JsonlTraceSink::new(path.clone())),
                recording.clone(),
            ],
        );
        let event = done_event(&run_id);

        logger.emit(&event).await?;

        assert_eq!(
            recording.events.lock().unwrap().as_slice(),
            std::slice::from_ref(&event)
        );
        assert_eq!(TraceLogger::read_events(&path).await?, vec![event]);
        let _ = tokio::fs::remove_file(path).await;
        Ok(())
    }

    #[tokio::test]
    async fn jsonl_sink_appends_many_events_through_one_handle() -> Result<()> {
        let run_id = Uuid::new_v4().to_string();
        let path = std::env::temp_dir().join(format!("trace-handle-test-{run_id}.jsonl"));
        let sink = JsonlTraceSink::new(path.clone());
        // Clones share the handle; interleave writes through both.
        let clone = sink.clone();
        for i in 0..50 {
            let target = if i % 2 == 0 { &sink } else { &clone };
            target.emit(&done_event(&run_id)).await?;
        }

        let events = TraceLogger::read_events(&path).await?;
        assert_eq!(events.len(), 50);
        let _ = tokio::fs::remove_file(path).await;
        Ok(())
    }

    /// t-1334: the logger folds every emitted InferResult into a run rollup
    /// and stamps it onto AgentDone. Sums are the recorded integers, cost
    /// stays partial-and-flagged when some infers had no pricing, and an
    /// infer-less run's AgentDone stays untouched (covered by
    /// `trace_logger_emits_to_all_sinks_and_preserves_jsonl_readback`,
    /// which asserts byte-equality for a bare AgentDone).
    #[tokio::test]
    async fn trace_logger_stamps_run_usage_onto_agent_done() -> Result<()> {
        let recording = Arc::new(RecordingSink::default());
        let logger = TraceLogger::with_sinks(
            "run",
            std::env::temp_dir().join("unused.jsonl"),
            vec![recording.clone()],
        );
        let infer_result = |op_id, cached, cost| Event::InferResult {
            run_id: "run".into(),
            op_id,
            parent_op_id: None,
            response: None,
            response_preview: String::new(),
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            cached_input_tokens: cached,
            cost_micro_usd: cost,
            pricing: None,
            duration_ms: 1,
            timestamp: Utc::now(),
        };
        logger.emit(&infer_result(1, Some(4), Some(100))).await?;
        // A clone shares the rollup, like par branches share the logger.
        logger.clone().emit(&infer_result(2, None, None)).await?;
        logger.emit(&done_event("run")).await?;

        let events = recording.events.lock().unwrap();
        let Event::AgentDone {
            usage: Some(usage), ..
        } = &events[2]
        else {
            panic!("AgentDone missing usage rollup: {:?}", events[2]);
        };
        assert_eq!(
            *usage,
            crate::cost::RunUsage {
                infer_calls: 2,
                input_tokens: 20,
                output_tokens: 10,
                total_tokens: 30,
                cached_input_tokens: Some(4),
                cost_micro_usd: Some(100),
                uncosted_infer_calls: 1,
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn otel_sink_maps_spans_parents_and_traceparent() -> Result<()> {
        let (exporter, provider) = install_in_memory_tracer();
        let context_env = TraceContextEnv::default();
        let sink = OtelTraceSink::with_context_env(context_env.clone());
        let run_id = Uuid::new_v4().to_string();

        sink.emit(&Event::InferCall {
            run_id: run_id.clone(),
            op_id: 7,
            parent_op_id: None,
            model: "mock-model".into(),
            prompt: None,
            prompt_preview: "hello".into(),
            effect: None,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::InferResult {
            run_id: run_id.clone(),
            op_id: 7,
            parent_op_id: None,
            response: None,
            response_preview: "world".into(),
            input_tokens: 10,
            output_tokens: 32,
            total_tokens: 42,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            duration_ms: 9,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::ParStart {
            run_id: run_id.clone(),
            op_id: 1,
            parent_op_id: None,
            branch_count: 1,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::EvalCall {
            run_id: run_id.clone(),
            op_id: 2,
            parent_op_id: Some(1),
            command: "printf ok".into(),
            argv: None,
            cwd: None,
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: None,
            timestamp: Utc::now(),
        })
        .await?;
        let traceparent = context_env
            .snapshot()
            .get("TRACEPARENT")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("TRACEPARENT not set"))?;
        assert!(traceparent.starts_with("00-"));
        assert_eq!(traceparent.len(), 55);
        sink.emit(&Event::EvalResult {
            run_id: run_id.clone(),
            op_id: 2,
            parent_op_id: Some(1),
            command: "printf ok".into(),
            result: serde_json::json!({ "ok": true }),
            duration_ms: 1,
            truncated_stdout: false,
            truncated_stderr: false,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::ParEnd {
            run_id: run_id.clone(),
            op_id: 1,
            parent_op_id: None,
            branch_count: 1,
            duration_ms: 2,
            timestamp: Utc::now(),
        })
        .await?;
        provider.force_flush()?;

        let spans = exporter.get_finished_spans()?;
        let infer = spans.iter().find(|span| span.name == "Infer").unwrap();
        assert_eq!(
            attr_value(infer, "gen_ai.request.model"),
            Some("mock-model".into())
        );
        assert_eq!(
            attr_value(infer, "gen_ai.usage.input_tokens"),
            Some("10".into())
        );
        assert_eq!(
            attr_value(infer, "gen_ai.usage.output_tokens"),
            Some("32".into())
        );
        assert_eq!(
            attr_value(infer, "gen_ai.usage.total_tokens"),
            Some("42".into())
        );
        assert_eq!(attr_value(infer, "duration_ms"), Some("9".into()));
        let par = spans.iter().find(|span| span.name == "Par").unwrap();
        let eval = spans.iter().find(|span| span.name == "Eval").unwrap();
        assert_eq!(par.span_context.span_id(), SpanId::from(1));
        assert_eq!(eval.span_context.span_id(), SpanId::from(2));
        assert_eq!(eval.parent_span_id, par.span_context.span_id());

        sink.emit(&Event::ParStart {
            run_id: run_id.clone(),
            op_id: 10,
            parent_op_id: None,
            branch_count: 1,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::ParStart {
            run_id: run_id.clone(),
            op_id: 20,
            parent_op_id: None,
            branch_count: 1,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::EvalCall {
            run_id: run_id.clone(),
            op_id: 11,
            parent_op_id: Some(10),
            command: "printf interleaved".into(),
            argv: None,
            cwd: None,
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: None,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::EvalResult {
            run_id: run_id.clone(),
            op_id: 11,
            parent_op_id: Some(10),
            command: "printf interleaved".into(),
            result: serde_json::json!({ "ok": true }),
            duration_ms: 1,
            truncated_stdout: false,
            truncated_stderr: false,
            timestamp: Utc::now(),
        })
        .await?;
        provider.force_flush()?;
        let spans = exporter.get_finished_spans()?;
        let interleaved_eval = spans
            .iter()
            .find(|span| span.span_context.span_id() == SpanId::from(11))
            .unwrap();
        assert_eq!(interleaved_eval.parent_span_id, SpanId::from(10));

        sink.emit(&Event::EvalCall {
            run_id: run_id.clone(),
            op_id: 3,
            parent_op_id: None,
            command: "cargo build --quiet".into(),
            argv: None,
            cwd: None,
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: None,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::EvalResult {
            run_id: run_id.clone(),
            op_id: 3,
            parent_op_id: None,
            command: "cargo build --quiet".into(),
            result: serde_json::json!({ "ok": true, "status": 0 }),
            duration_ms: 4,
            truncated_stdout: false,
            truncated_stderr: false,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::EvalCall {
            run_id: run_id.clone(),
            op_id: 4,
            parent_op_id: None,
            command: "cargo build --quiet".into(),
            argv: None,
            cwd: None,
            env_policy: "inherit".into(),
            timeout_ms: 1000,
            effect: None,
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::Custom {
            run_id: run_id.clone(),
            name: "domain_tag".into(),
            data: serde_json::json!({ "kernel.name": "linux" }),
            timestamp: Utc::now(),
        })
        .await?;
        sink.emit(&Event::EvalResult {
            run_id,
            op_id: 4,
            parent_op_id: None,
            command: "cargo build --quiet".into(),
            result: serde_json::json!({ "ok": false, "status": 101 }),
            duration_ms: 5,
            truncated_stdout: false,
            truncated_stderr: false,
            timestamp: Utc::now(),
        })
        .await?;
        provider.force_flush()?;

        let spans = exporter.get_finished_spans()?;
        let failing_eval = spans
            .iter()
            .find(|span| matches!(span.status, Status::Error { .. }))
            .unwrap();
        assert!(matches!(failing_eval.status, Status::Error { .. }));
        assert_eq!(
            attr_value(failing_eval, "tool.name"),
            Some("cargo build".into())
        );
        assert_eq!(attr_value(failing_eval, "exit_code"), Some("101".into()));
        assert_eq!(attr_value(failing_eval, "ok"), Some("false".into()));
        assert_eq!(
            attr_value(failing_eval, "kernel.name"),
            Some("linux".into())
        );
        assert_eq!(attr_value(failing_eval, "attempt"), Some("2".into()));
        assert_eq!(attr_value(failing_eval, "retry.count"), Some("1".into()));
        let _ = provider.shutdown();
        Ok(())
    }
}
