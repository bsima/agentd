//! Cost accounting for Infer effects (t-1334).
//!
//! All canonical arithmetic is exact integer math:
//!
//! - Rates are stored as **integer micro-USD per million tokens** (so
//!   `$3.00 / Mtok` is `3_000_000`). YAML rates (USD per Mtok, e.g. `3.0`)
//!   are converted once at registry load with half-up rounding to the
//!   nearest micro-USD-per-Mtok — precision far below any real price.
//! - Per-call cost is **integer micro-USD**, rounded half-up once per call:
//!   `round((input_tokens * input_rate + output_tokens * output_rate) / 1e6)`.
//! - Rollups sum the recorded per-call integers. Floats never accumulate
//!   across events; any `f64` seen near costs is display-only.
//!
//! Unknown pricing means cost is *absent* — never guessed, never zero.
//! Recorded costs (and the [`Pricing`] snapshot used) are part of the
//! trace's recorded result payload, so replay reproduces the original
//! totals even if today's models.yaml prices differ.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Tokens per million tokens; also micro-dollars per dollar. The shared
/// scale factor that makes `tokens * rate / SCALE` come out in micro-USD.
const SCALE: u128 = 1_000_000;

/// A pricing snapshot: integer micro-USD per million tokens. This exact
/// struct is recorded on `InferResult` trace events alongside the computed
/// cost, so a trace is self-contained (replay and rollups never consult
/// models.yaml).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pricing {
    pub input_micro_usd_per_mtok: u64,
    pub output_micro_usd_per_mtok: u64,
}

impl Pricing {
    /// Convert human rates (USD per million tokens, as written in
    /// models.yaml) to the integer representation. Fails closed on
    /// non-finite, negative, or absurdly large rates instead of silently
    /// recording garbage costs.
    pub fn from_usd_per_mtok(input: f64, output: f64) -> Result<Self> {
        Ok(Self {
            input_micro_usd_per_mtok: usd_rate_to_micro(input, "input_per_mtok_usd")?,
            output_micro_usd_per_mtok: usd_rate_to_micro(output, "output_per_mtok_usd")?,
        })
    }

    /// Cost of one call in integer micro-USD, rounded half-up once.
    /// Exact in u128 up to u32::MAX tokens at the maximum accepted rate.
    pub fn cost_micro_usd(&self, input_tokens: u32, output_tokens: u32) -> u64 {
        let numerator = u128::from(input_tokens) * u128::from(self.input_micro_usd_per_mtok)
            + u128::from(output_tokens) * u128::from(self.output_micro_usd_per_mtok);
        u64::try_from((numerator + SCALE / 2) / SCALE).expect("cost fits in u64 micro-USD")
    }
}

fn usd_rate_to_micro(rate: f64, field: &str) -> Result<u64> {
    // 1e9 USD/Mtok is far beyond any real price and keeps the cost math
    // overflow-free in u128.
    if !rate.is_finite() || !(0.0..=1e9).contains(&rate) {
        return Err(anyhow!(
            "invalid pricing rate {field}: {rate} (must be a finite USD-per-Mtok value in [0, 1e9])"
        ));
    }
    Ok((rate * SCALE as f64).round() as u64)
}

/// Model-id -> pricing lookup used at trace-emission time. Keys are model
/// id strings (registry aliases and provider api ids both). Empty table =
/// usage recorded, cost omitted.
#[derive(Debug, Clone, Default)]
pub struct PricingTable {
    by_model: BTreeMap<String, Pricing>,
}

impl PricingTable {
    /// Register pricing for a model id. First insertion wins so a registry
    /// listing the same api id under several aliases stays deterministic.
    pub fn insert(&mut self, model: impl Into<String>, pricing: Pricing) {
        self.by_model.entry(model.into()).or_insert(pricing);
    }

    pub fn get(&self, model: &str) -> Option<&Pricing> {
        self.by_model.get(model)
    }

    pub fn is_empty(&self) -> bool {
        self.by_model.is_empty()
    }
}

/// Stamp cost onto a live provider response: looks up the model's pricing
/// and records both the computed micro-USD cost and the rates used. No
/// pricing, no cost — the response is left untouched (usage still records).
///
/// Never call this on a replayed response: recorded costs are replay
/// identity and must pass through verbatim.
pub fn price_response(response: &mut crate::op::Response, table: &PricingTable, model: &str) {
    if let Some(pricing) = table.get(model) {
        response.pricing = Some(*pricing);
        response.cost_micro_usd =
            Some(pricing.cost_micro_usd(response.input_tokens, response.output_tokens));
    }
}

/// Per-run usage/cost rollup, recorded on the `AgentDone` trace event.
/// Token and cost fields are sums of the per-`InferResult` recorded
/// integers (exact; no float accumulation). `cached_input_tokens` and
/// `cost_micro_usd` are absent when no InferResult in the run carried
/// them; `uncosted_infer_calls` counts InferResults that had usage but no
/// cost (unknown pricing), so a partial `cost_micro_usd` is never mistaken
/// for a full total.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunUsage {
    /// Successful Infer calls (one per recorded `InferResult`).
    pub infer_calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_micro_usd: Option<u64>,
    pub uncosted_infer_calls: u64,
    /// Failed Infer attempts (one per `InferError`, t-1347): calls that
    /// dispatched but ended in a terminal error, so they produced no
    /// `InferResult` and contribute nothing to the token/cost sums above.
    /// This is a count only: the provider error path returns a bare error
    /// with no `Response`, so any tokens the provider consumed before
    /// failing are structurally unavailable — never guessed, never zeroed
    /// into the sums. Absent from JSON when 0, so pre-t-1347 rollups stay
    /// byte-identical and deserialize to 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub failed_infer_calls: u64,
}

/// serde helper: skip a count field when it is 0 (the field's absence and
/// zero are the same statement for counts, unlike the Option cost fields).
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(count: &u64) -> bool {
    *count == 0
}

impl RunUsage {
    /// Fold one InferResult's recorded usage/cost into the rollup.
    pub fn observe_infer(
        &mut self,
        input_tokens: u32,
        output_tokens: u32,
        total_tokens: u32,
        cached_input_tokens: Option<u32>,
        cost_micro_usd: Option<u64>,
    ) {
        self.infer_calls += 1;
        self.input_tokens += u64::from(input_tokens);
        self.output_tokens += u64::from(output_tokens);
        self.total_tokens += u64::from(total_tokens);
        if let Some(cached) = cached_input_tokens {
            *self.cached_input_tokens.get_or_insert(0) += u64::from(cached);
        }
        match cost_micro_usd {
            Some(cost) => *self.cost_micro_usd.get_or_insert(0) += cost,
            None => self.uncosted_infer_calls += 1,
        }
    }

    /// Fold one InferError into the rollup (t-1347). A count only — see
    /// [`RunUsage::failed_infer_calls`] for why no usage rides along.
    pub fn observe_infer_error(&mut self) {
        self.failed_infer_calls += 1;
    }

    /// True when no Infer outcome (result or error) has been folded in
    /// (nothing to report).
    pub fn is_empty(&self) -> bool {
        self.infer_calls == 0 && self.failed_infer_calls == 0
    }
}

/// Render integer micro-USD as a dollar string (`66` -> `"$0.000066"`).
/// Display-only; the integer stays canonical.
pub fn format_micro_usd(micro_usd: u64) -> String {
    format!("${}.{:06}", micro_usd / 1_000_000, micro_usd % 1_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_convert_exactly_and_reject_garbage() -> Result<()> {
        let pricing = Pricing::from_usd_per_mtok(0.15, 0.60)?;
        assert_eq!(pricing.input_micro_usd_per_mtok, 150_000);
        assert_eq!(pricing.output_micro_usd_per_mtok, 600_000);
        let pricing = Pricing::from_usd_per_mtok(3.0, 15.0)?;
        assert_eq!(pricing.input_micro_usd_per_mtok, 3_000_000);
        assert_eq!(pricing.output_micro_usd_per_mtok, 15_000_000);
        // Zero is a legal (if odd) explicit price; absence is expressed by
        // not configuring pricing at all, not by zero rates.
        assert!(Pricing::from_usd_per_mtok(0.0, 0.0).is_ok());
        for bad in [-0.01, f64::NAN, f64::INFINITY, 1e12] {
            assert!(Pricing::from_usd_per_mtok(bad, 1.0).is_err(), "{bad}");
            assert!(Pricing::from_usd_per_mtok(1.0, bad).is_err(), "{bad}");
        }
        Ok(())
    }

    #[test]
    fn cost_math_is_exact_integer_micro_usd() -> Result<()> {
        let pricing = Pricing::from_usd_per_mtok(3.0, 15.0)?;
        // 7 input + 3 output tokens: (7*3e6 + 3*15e6) / 1e6 = 66 micro-USD.
        assert_eq!(pricing.cost_micro_usd(7, 3), 66);
        assert_eq!(pricing.cost_micro_usd(0, 0), 0);
        // 1M input tokens at $3/Mtok is exactly $3.
        assert_eq!(pricing.cost_micro_usd(1_000_000, 0), 3_000_000);

        // Sub-micro-dollar calls round half-up once per call.
        let cheap = Pricing::from_usd_per_mtok(0.5, 0.5)?;
        assert_eq!(cheap.cost_micro_usd(1, 0), 1, "0.5 rounds up");
        let cheaper = Pricing::from_usd_per_mtok(0.4, 0.4)?;
        assert_eq!(cheaper.cost_micro_usd(1, 0), 0, "0.4 rounds down");

        // Extremes do not overflow: u32::MAX tokens at the max rate.
        let max = Pricing {
            input_micro_usd_per_mtok: 10u64.pow(15),
            output_micro_usd_per_mtok: 10u64.pow(15),
        };
        let cost = max.cost_micro_usd(u32::MAX, u32::MAX);
        assert_eq!(
            cost,
            ((u128::from(u32::MAX) * 2 * 10u128.pow(15) + 500_000) / 1_000_000) as u64
        );
        Ok(())
    }

    #[test]
    fn pricing_table_first_insert_wins() -> Result<()> {
        let mut table = PricingTable::default();
        table.insert("m", Pricing::from_usd_per_mtok(1.0, 2.0)?);
        table.insert("m", Pricing::from_usd_per_mtok(9.0, 9.0)?);
        assert_eq!(table.get("m").unwrap().input_micro_usd_per_mtok, 1_000_000);
        assert!(table.get("other").is_none());
        Ok(())
    }

    #[test]
    fn run_usage_accumulates_and_counts_uncosted() {
        let mut usage = RunUsage::default();
        assert!(usage.is_empty());
        usage.observe_infer(10, 5, 15, Some(4), Some(100));
        usage.observe_infer(20, 5, 25, None, Some(50));
        usage.observe_infer(1, 1, 2, None, None);
        usage.observe_infer_error();
        assert_eq!(usage.infer_calls, 3);
        assert_eq!(usage.input_tokens, 31);
        assert_eq!(usage.output_tokens, 11);
        assert_eq!(usage.total_tokens, 42);
        // Cached is a sum of what was reported, not a claim about the rest.
        assert_eq!(usage.cached_input_tokens, Some(4));
        // Cost sums only the costed calls; the uncosted one is counted, so
        // the total is visibly partial rather than silently low.
        assert_eq!(usage.cost_micro_usd, Some(150));
        assert_eq!(usage.uncosted_infer_calls, 1);
        // The failed attempt is counted apart from the successes and adds
        // nothing to the token/cost sums (no usage exists for it).
        assert_eq!(usage.failed_infer_calls, 1);
    }

    #[test]
    fn run_usage_leaves_never_reported_fields_absent() {
        let mut usage = RunUsage::default();
        usage.observe_infer(1, 1, 2, None, None);
        assert_eq!(usage.cached_input_tokens, None);
        assert_eq!(usage.cost_micro_usd, None);
        let json = serde_json::to_string(&usage).unwrap();
        assert!(!json.contains("cached_input_tokens"), "{json}");
        assert!(!json.contains("cost_micro_usd"), "{json}");
        // No failures -> no key: pre-t-1347 rollups stay byte-identical.
        assert!(!json.contains("failed_infer_calls"), "{json}");
    }

    /// A run whose every Infer failed still has something to report (the
    /// attempts), the count round-trips, and pre-t-1347 rollup JSON (no
    /// `failed_infer_calls` key) deserializes to 0.
    #[test]
    fn run_usage_failed_calls_round_trip_and_back_compat() {
        let mut usage = RunUsage::default();
        usage.observe_infer_error();
        assert!(!usage.is_empty(), "failed attempts are reportable");
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("\"failed_infer_calls\":1"), "{json}");
        assert_eq!(serde_json::from_str::<RunUsage>(&json).unwrap(), usage);

        let old = r#"{"infer_calls":2,"input_tokens":1,"output_tokens":1,"total_tokens":2,"uncosted_infer_calls":0}"#;
        let usage: RunUsage = serde_json::from_str(old).unwrap();
        assert_eq!(usage.failed_infer_calls, 0);
    }

    #[test]
    fn formats_micro_usd_as_dollars() {
        assert_eq!(format_micro_usd(0), "$0.000000");
        assert_eq!(format_micro_usd(66), "$0.000066");
        assert_eq!(format_micro_usd(3_000_000), "$3.000000");
        assert_eq!(format_micro_usd(12_345_678), "$12.345678");
    }

    #[test]
    fn price_response_stamps_cost_or_leaves_untouched() -> Result<()> {
        let mut table = PricingTable::default();
        table.insert("priced", Pricing::from_usd_per_mtok(3.0, 15.0)?);
        let mut response = crate::op::Response {
            content: "ok".into(),
            tool_calls: Vec::new(),
            finish_reason: None,
            input_tokens: 7,
            output_tokens: 3,
            total_tokens: 10,
            cached_input_tokens: None,
            cost_micro_usd: None,
            pricing: None,
            metadata: Default::default(),
        };
        let mut unpriced = response.clone();

        price_response(&mut response, &table, "priced");
        assert_eq!(response.cost_micro_usd, Some(66));
        assert_eq!(
            response.pricing,
            Some(Pricing::from_usd_per_mtok(3.0, 15.0)?)
        );

        price_response(&mut unpriced, &table, "unknown-model");
        assert_eq!(unpriced.cost_micro_usd, None, "never guess, never zero");
        assert_eq!(unpriced.pricing, None);
        Ok(())
    }
}
