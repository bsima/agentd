//! Session modes and config options reported to ACP clients: the `ask`/
//! `yolo` approval modes (the runtime's one behavioral toggle) and a model
//! picker built from the models.yaml registry.

use agent_client_protocol::schema::v1::{
    SessionConfigOption, SessionConfigOptionCategory, SessionConfigSelectOption, SessionMode,
    SessionModeState,
};
use agent_core::ModelRegistry;

pub(crate) const MODE_ASK: &str = "ask";
pub(crate) const MODE_YOLO: &str = "yolo";
pub(crate) const MODEL_CONFIG_ID: &str = "model";

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

/// The model picker: every models.yaml alias, with the session's current
/// alias selected (and prepended when it is not in the registry — the raw
/// `--model` fallback path).
pub(crate) async fn model_config_options(current_alias: &str) -> Vec<SessionConfigOption> {
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
    vec![
        SessionConfigOption::select(MODEL_CONFIG_ID, "Model", current_alias.to_owned(), options)
            .category(SessionConfigOptionCategory::Model),
    ]
}
