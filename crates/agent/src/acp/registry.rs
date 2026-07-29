//! Session modes and config options reported to ACP clients: the `ask`/
//! `yolo` approval modes (the runtime's one behavioral toggle), a model
//! picker built from the models.yaml registry, and context-GC tuning
//! (strategy + trigger threshold) mirroring the `--gc`/`--gc-threshold`
//! flags.

use agent_client_protocol::schema::v1::{
    SessionConfigOption, SessionConfigOptionCategory, SessionConfigSelectOption, SessionMode,
    SessionModeState,
};
use agent_core::{GcMode, ModelRegistry};

pub(crate) const MODE_ASK: &str = "ask";
pub(crate) const MODE_YOLO: &str = "yolo";
pub(crate) const MODEL_CONFIG_ID: &str = "model";
pub(crate) const GC_CONFIG_ID: &str = "gc";
pub(crate) const GC_THRESHOLD_CONFIG_ID: &str = "gc-threshold";

/// Strategy value ids, matching the `--gc` flag's ValueEnum names.
pub(crate) const GC_CHOICES: [(&str, &str); 6] = [
    ("none", "Off"),
    ("ring", "Ring"),
    ("mark-sweep", "Mark-sweep"),
    ("stack", "Stack (default)"),
    ("semantic", "Semantic"),
    ("generational", "Generational"),
];

/// Threshold value ids: fraction of the context budget that triggers a
/// collection. Discrete steps — ACP selects have no numeric input.
pub(crate) const GC_THRESHOLD_CHOICES: [&str; 6] = ["0.5", "0.6", "0.7", "0.8", "0.85", "0.9"];

pub(crate) fn session_modes(shell_requires_approval: bool) -> SessionModeState {
    let current = if shell_requires_approval {
        MODE_ASK
    } else {
        MODE_YOLO
    };
    SessionModeState::new(
        current,
        vec![
            SessionMode::new(MODE_ASK, "Ask")
                .description("Shell commands need permission before running"),
            SessionMode::new(MODE_YOLO, "Yolo").description("Shell commands run without asking"),
        ],
    )
}

/// The `--gc` value id for the runtime's current GC mode.
pub(crate) fn gc_value_id(gc: &GcMode) -> &'static str {
    match gc {
        GcMode::None => "none",
        GcMode::Ring(_) => "ring",
        GcMode::MarkSweep(_) => "mark-sweep",
        GcMode::Stack(_) => "stack",
        GcMode::Semantic(_) => "semantic",
        GcMode::Generational(_) => "generational",
    }
}

/// The full config-option set for a session:
/// `SetSessionConfigOptionResponse` carries the complete refreshed list, so
/// every path that reports options goes through here.
pub(crate) async fn session_config_options(
    model_alias: &str,
    gc: &GcMode,
    gc_threshold: f32,
) -> Vec<SessionConfigOption> {
    vec![
        model_option(model_alias).await,
        gc_option(gc),
        gc_threshold_option(gc_threshold),
    ]
}

/// The model picker: every models.yaml alias, with the session's current
/// alias selected (and prepended when it is not in the registry — the raw
/// `--model` fallback path).
async fn model_option(current_alias: &str) -> SessionConfigOption {
    let aliases: Vec<String> = match ModelRegistry::load_default().await {
        Ok(registry) => registry
            .models
            .iter()
            .map(|entry| entry.name.clone())
            .collect(),
        Err(_) => Vec::new(),
    };
    let mut options: Vec<SessionConfigSelectOption> = aliases
        .iter()
        .map(|alias| SessionConfigSelectOption::new(alias.clone(), alias.clone()))
        .collect();
    if !aliases.iter().any(|alias| alias == current_alias) {
        options.insert(
            0,
            SessionConfigSelectOption::new(current_alias.to_owned(), current_alias.to_owned()),
        );
    }
    SessionConfigOption::select(MODEL_CONFIG_ID, "Model", current_alias.to_owned(), options)
        .category(SessionConfigOptionCategory::Model)
}

fn gc_option(gc: &GcMode) -> SessionConfigOption {
    let options: Vec<SessionConfigSelectOption> = GC_CHOICES
        .iter()
        .map(|(id, name)| SessionConfigSelectOption::new(*id, *name))
        .collect();
    SessionConfigOption::select(GC_CONFIG_ID, "Context GC", gc_value_id(gc), options)
        .description("Strategy for reclaiming context when the window fills (docs/GC.md)")
}

fn gc_threshold_option(current: f32) -> SessionConfigOption {
    let current_id = format_threshold(current);
    let mut options: Vec<SessionConfigSelectOption> = GC_THRESHOLD_CHOICES
        .iter()
        .map(|id| SessionConfigSelectOption::new(*id, format!("{}%", percent(id))))
        .collect();
    if !GC_THRESHOLD_CHOICES.contains(&current_id.as_str()) {
        options.insert(
            0,
            SessionConfigSelectOption::new(
                current_id.clone(),
                format!("{}%", percent(&current_id)),
            ),
        );
    }
    SessionConfigOption::select(GC_THRESHOLD_CONFIG_ID, "GC threshold", current_id, options)
        .description("Collect once the prompt reaches this fraction of the context budget")
}

/// Canonical value id for a threshold fraction: trailing zeros trimmed so
/// 0.85 and 0.850 collide onto one id.
fn format_threshold(value: f32) -> String {
    let mut id = format!("{value:.2}");
    while id.ends_with('0') {
        id.pop();
    }
    if id.ends_with('.') {
        id.pop();
    }
    id
}

fn percent(id: &str) -> String {
    match id.parse::<f32>() {
        Ok(value) => format!("{:.0}", value * 100.0),
        Err(_) => id.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn config_options_cover_model_gc_and_threshold() {
        let options = session_config_options("some/model", &GcMode::None, 0.85).await;
        let ids: Vec<&str> = options.iter().map(|o| o.id.0.as_ref()).collect();
        assert_eq!(ids, [MODEL_CONFIG_ID, GC_CONFIG_ID, GC_THRESHOLD_CONFIG_ID]);
    }

    #[test]
    fn gc_option_reflects_the_current_strategy() {
        let option = gc_option(&GcMode::None);
        let agent_client_protocol::schema::v1::SessionConfigKind::Select(select) = &option.kind
        else {
            panic!("gc option must be a select")
        };
        assert_eq!(select.current_value.0.as_ref(), "none");
    }

    #[test]
    fn unlisted_threshold_is_prepended() {
        let option = gc_threshold_option(0.42);
        let agent_client_protocol::schema::v1::SessionConfigKind::Select(select) = &option.kind
        else {
            panic!("threshold option must be a select")
        };
        assert_eq!(select.current_value.0.as_ref(), "0.42");
        let agent_client_protocol::schema::v1::SessionConfigSelectOptions::Ungrouped(options) =
            &select.options
        else {
            panic!("ungrouped options")
        };
        assert_eq!(options[0].value.0.as_ref(), "0.42");
    }

    #[test]
    fn threshold_ids_are_canonical() {
        assert_eq!(format_threshold(0.85), "0.85");
        assert_eq!(format_threshold(0.8), "0.8");
        assert_eq!(format_threshold(0.5), "0.5");
    }
}
