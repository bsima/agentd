//! Structured final output: the `OutputContract` (t-1308.4, SDK PRD DR-2).
//!
//! A contract attaches a JSON Schema to the agent loop's *final* response:
//! when the loop completes naturally, the response text must parse as JSON
//! and validate against the schema. Failures get a bounded number of repair
//! turns (`max_repairs`); exhaustion returns a typed error **value** (the
//! runtime's errors-as-values convention, t-1222) rather than aborting.
//! Enforcement lives in `ir_agent::run_agent_loop` — loop-level
//! post-processing, per the PRD decision to keep it out of `InferPolicy`
//! until AgentIR can host it cleanly.
//!
//! ## Validation engine: a deliberate subset
//!
//! This module implements its own validator instead of depending on the
//! `jsonschema` crate: even with default features off, `jsonschema` v0.46
//! pulls ~99 transitive packages (the icu/idna unicode stack, fancy-regex,
//! num-bigint, email_address, ...) into a workspace that locks ~266 total.
//! The PRD needs bounded practical validation of model output, not draft
//! compliance, so the supported keyword set is exactly:
//!
//! - `type` — a string or array of strings among `"object"`, `"array"`,
//!   `"string"`, `"number"`, `"integer"`, `"boolean"`, `"null"`
//!   (`"integer"` accepts any JSON number with a zero fractional part)
//! - `required` — array of property names, checked on object values
//! - `properties` — per-property subschemas, applied to present properties
//! - `items` — a single subschema applied to every array element
//! - `enum` — array of allowed values (deep equality)
//!
//! Everything else (`$ref`, `oneOf`, `pattern`, `additionalProperties`,
//! numeric bounds, ...) is **ignored**, i.e. permissive. Subschemas recurse,
//! so nested objects/arrays validate. If a real draft validator is ever
//! warranted, this module is the seam to swap it behind.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Default number of repair turns appended before the loop gives up and
/// returns an [`OutputContractFailure`] value.
pub const DEFAULT_MAX_REPAIRS: usize = 2;

/// `error` tag of the typed failure value returned when repairs are
/// exhausted; mirrors the `{"ok": false, "error": ...}` envelope of
/// effect errors-as-values (t-1222).
pub const OUTPUT_CONTRACT_ERROR: &str = "output_contract_violation";

/// Runtime `Custom` trace event emitted once per loop run when a contract
/// is present, carrying `schema_hash` (run-identity metadata: replay of a
/// run recorded with a different schema diverges). Private to the runtime
/// trace — no public projection.
pub const OUTPUT_CONTRACT_EVENT: &str = "output_contract";

/// Runtime `Custom` trace event emitted on each failed validation, with
/// `{attempt, errors, preview}`. Projected to the public
/// `output.validation_failed` event (docs/TRACE_SCHEMA.md).
pub const OUTPUT_VALIDATION_FAILED_EVENT: &str = "output_validation_failed";

/// Cap on the `errors` list carried by the trace event (the in-memory
/// failure value keeps the full list).
pub const MAX_TRACE_ERRORS: usize = 8;

/// A JSON Schema contract on the agent loop's final response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputContract {
    /// JSON Schema document (see the module doc for the supported subset).
    pub schema: Value,
    /// Repair turns allowed before the loop returns a failure value.
    pub max_repairs: usize,
}

impl OutputContract {
    pub fn new(schema: Value) -> Self {
        Self {
            schema,
            max_repairs: DEFAULT_MAX_REPAIRS,
        }
    }

    /// sha256 of the canonical (sorted-key) JSON serialization of the
    /// schema, `"sha256:<hex>"` — same rendering as `ProgramHash`. Key
    /// order in the source file does not change the hash; whitespace never
    /// survives serialization. Recorded in the trace as run-identity
    /// metadata so replay can detect a changed contract.
    pub fn schema_hash(&self) -> String {
        let mut out = String::new();
        canonical_json(&self.schema, &mut out);
        format!("sha256:{:x}", Sha256::digest(out.as_bytes()))
    }

    /// Validate the final response text: it must parse as JSON, then
    /// conform to the schema. Returns human-readable errors ([] = valid).
    pub fn validate_text(&self, text: &str) -> Vec<String> {
        match serde_json::from_str::<Value>(text) {
            Ok(value) => validate(&self.schema, &value),
            Err(err) => vec![format!("response is not valid JSON: {err}")],
        }
    }
}

/// Canonical JSON: object keys sorted, no whitespace. Written by hand so
/// the hash cannot silently change if a dependency enables serde_json's
/// `preserve_order` feature (feature unification is workspace-global).
fn canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*key).clone()).to_string());
                out.push(':');
                canonical_json(&map[*key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                canonical_json(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// Validate `value` against `schema` (the subset in the module doc).
/// Returns one message per violation, each prefixed with a `$.a.b[0]`-style
/// path. A non-object schema validates everything (permissive).
pub fn validate(schema: &Value, value: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    validate_at(schema, value, "$", &mut errors);
    errors
}

fn validate_at(schema: &Value, value: &Value, path: &str, errors: &mut Vec<String>) {
    let Value::Object(schema) = schema else {
        return;
    };
    if let Some(expected) = schema.get("type") {
        let allowed: Vec<&str> = match expected {
            Value::String(name) => vec![name.as_str()],
            Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
            _ => vec![],
        };
        if !allowed.is_empty() && !allowed.iter().any(|name| type_matches(name, value)) {
            errors.push(format!(
                "{path}: expected type {}, got {}",
                allowed.join(" or "),
                type_name(value)
            ));
        }
    }
    if let Some(Value::Array(options)) = schema.get("enum") {
        if !options.contains(value) {
            errors.push(format!(
                "{path}: {} is not one of the allowed enum values",
                render_value(value)
            ));
        }
    }
    if let Value::Object(object) = value {
        if let Some(Value::Array(required)) = schema.get("required") {
            for key in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(key) {
                    errors.push(format!("{path}: missing required property {key:?}"));
                }
            }
        }
        if let Some(Value::Object(properties)) = schema.get("properties") {
            for (key, subschema) in properties {
                if let Some(child) = object.get(key) {
                    validate_at(subschema, child, &format!("{path}.{key}"), errors);
                }
            }
        }
    }
    if let (Value::Array(items), Some(item_schema)) = (value, schema.get("items")) {
        for (index, item) in items.iter().enumerate() {
            validate_at(item_schema, item, &format!("{path}[{index}]"), errors);
        }
    }
}

fn type_matches(name: &str, value: &Value) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "integer" => match value {
            Value::Number(number) => {
                number.is_i64()
                    || number.is_u64()
                    || number.as_f64().is_some_and(|float| float.fract() == 0.0)
            }
            _ => false,
        },
        // An unknown type name never matches: a schema typo fails loudly
        // instead of silently accepting anything.
        _ => false,
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn render_value(value: &Value) -> String {
    crate::trace::preview(&value.to_string(), 80)
}

/// The typed failure returned (as a value, not an `Err`) when validation
/// repairs are exhausted. Carries the last attempt's validation errors and
/// the offending response text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputContractFailure {
    /// Total validation attempts (initial + repairs), all failed.
    pub attempts: usize,
    /// The last attempt's validation errors, unbounded.
    pub errors: Vec<String>,
    /// The last (non-conforming) response text.
    pub content: String,
}

impl OutputContractFailure {
    /// Render as the loop's return value: the `{"ok": false, "error": ...}`
    /// envelope shared with effect errors-as-values (t-1222).
    pub fn into_value(self) -> Value {
        serde_json::json!({
            "ok": false,
            "error": OUTPUT_CONTRACT_ERROR,
            "attempts": self.attempts,
            "validation_errors": self.errors,
            "content": self.content,
        })
    }
}

/// Recognize a loop return value as an exhausted-repairs contract failure.
/// `None` for ordinary responses.
pub fn output_contract_failure(value: &Value) -> Option<OutputContractFailure> {
    if value.get("ok") != Some(&Value::Bool(false))
        || value.get("error").and_then(Value::as_str) != Some(OUTPUT_CONTRACT_ERROR)
    {
        return None;
    }
    Some(OutputContractFailure {
        attempts: value.get("attempts").and_then(Value::as_u64).unwrap_or(0) as usize,
        errors: value
            .get("validation_errors")
            .and_then(Value::as_array)
            .map(|errors| {
                errors
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        content: value
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn valid_object_passes() {
        let schema = json!({
            "type": "object",
            "required": ["answer"],
            "properties": { "answer": { "type": "integer" } }
        });
        assert!(validate(&schema, &json!({ "answer": 42 })).is_empty());
    }

    #[test]
    fn type_mismatches_are_reported_with_paths() {
        let schema = json!({
            "type": "object",
            "properties": { "answer": { "type": "integer" } }
        });
        let errors = validate(&schema, &json!({ "answer": "forty-two" }));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("$.answer"), "{errors:?}");
        assert!(errors[0].contains("integer"), "{errors:?}");
    }

    #[test]
    fn missing_required_property_is_reported() {
        let schema = json!({ "type": "object", "required": ["a", "b"] });
        let errors = validate(&schema, &json!({ "a": 1 }));
        assert_eq!(errors, vec!["$: missing required property \"b\""]);
    }

    #[test]
    fn items_recurse_into_array_elements() {
        let schema = json!({
            "type": "array",
            "items": { "type": "object", "required": ["id"] }
        });
        let errors = validate(&schema, &json!([{ "id": 1 }, {}]));
        assert_eq!(errors, vec!["$[1]: missing required property \"id\""]);
    }

    #[test]
    fn enum_membership_is_deep_equality() {
        let schema = json!({ "enum": ["red", "green"] });
        assert!(validate(&schema, &json!("red")).is_empty());
        let errors = validate(&schema, &json!("blue"));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("enum"), "{errors:?}");
    }

    #[test]
    fn type_arrays_accept_any_listed_type() {
        let schema = json!({ "type": ["string", "null"] });
        assert!(validate(&schema, &json!("x")).is_empty());
        assert!(validate(&schema, &Value::Null).is_empty());
        assert_eq!(validate(&schema, &json!(1)).len(), 1);
    }

    #[test]
    fn integer_accepts_zero_fraction_floats_and_rejects_others() {
        let schema = json!({ "type": "integer" });
        assert!(validate(&schema, &json!(3)).is_empty());
        assert!(validate(&schema, &json!(3.0)).is_empty());
        assert_eq!(validate(&schema, &json!(3.5)).len(), 1);
    }

    #[test]
    fn unknown_keywords_are_ignored() {
        // Permissive subset: keywords outside type/required/properties/
        // items/enum have no effect.
        let schema = json!({ "type": "string", "pattern": "^[0-9]+$", "minLength": 99 });
        assert!(validate(&schema, &json!("abc")).is_empty());
    }

    #[test]
    fn unknown_type_name_fails_loudly() {
        let schema = json!({ "type": "strng" });
        assert_eq!(validate(&schema, &json!("x")).len(), 1);
    }

    #[test]
    fn validate_text_reports_non_json() {
        let contract = OutputContract::new(json!({ "type": "object" }));
        let errors = contract.validate_text("not json at all");
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("not valid JSON"), "{errors:?}");
    }

    #[test]
    fn schema_hash_is_key_order_independent() {
        let a = OutputContract::new(json!({ "type": "object", "required": ["x"] }));
        let b: Value = serde_json::from_str(r#"{ "required": ["x"], "type": "object" }"#).unwrap();
        assert_eq!(a.schema_hash(), OutputContract::new(b).schema_hash());
        assert!(a.schema_hash().starts_with("sha256:"));
        let c = OutputContract::new(json!({ "type": "array" }));
        assert_ne!(a.schema_hash(), c.schema_hash());
    }

    #[test]
    fn failure_value_round_trips() {
        let failure = OutputContractFailure {
            attempts: 3,
            errors: vec!["$: missing required property \"a\"".into()],
            content: "{}".into(),
        };
        let value = failure.clone().into_value();
        assert_eq!(output_contract_failure(&value), Some(failure));
        // Ordinary responses and effect-error values are not misdetected.
        assert_eq!(output_contract_failure(&json!({ "content": "hi" })), None);
        assert_eq!(
            output_contract_failure(&json!({ "ok": false, "error": "boom" })),
            None
        );
    }
}
