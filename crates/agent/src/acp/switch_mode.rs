//! Profile-scoped mode switch: validation is a native tool handler, but the
//! interpreter gates its validated result before returning it to the model.
use agent_core::NativeTool;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use super::{profiles::Profiles, registry};

pub(super) const TOOL: &str = "switch_mode";

pub(super) fn target(arguments: &Value, current: &str, profiles: &Profiles) -> Result<Value> {
    let mode = arguments
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("switch_mode requires a string `mode`"))?;
    if mode == current {
        return Err(anyhow!("already in {mode:?} mode"));
    }
    let name = match mode {
        registry::MODE_ASK => "Ask",
        registry::MODE_YOLO => "Yolo",
        _ => {
            &profiles
                .get(mode)
                .ok_or_else(|| anyhow!("unknown mode {mode:?}"))?
                .name
        }
    };
    Ok(json!({"mode": mode, "name": name, "current": current}))
}

pub(super) fn tool(current: &str, profiles: &Profiles) -> NativeTool {
    let current = current.to_owned();
    let modes = profiles.0.clone();
    NativeTool::from_fn(
        TOOL,
        agent_core::guidance::SWITCH_MODE_TOOL_DESCRIPTION,
        json!({"type":"object", "properties":{"mode":{"type":"string", "description":"id of an advertised mode"}}, "required":["mode"]}),
        move |arguments| {
            let current = current.clone();
            let profiles = Profiles(modes.clone());
            async move {
                let request = target(&arguments, &current, &profiles)?;
                Ok(
                    json!({"mode": request["mode"], "name": request["name"], "current": request["current"], "message": "The mode switch takes effect after this turn. End your turn now; do not act under the new mode yet."}),
                )
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_current_unknown_and_advertised_targets() {
        let profiles = Profiles::default();
        assert!(target(&json!({"mode":"ask"}), "ask", &profiles)
            .unwrap_err()
            .to_string()
            .contains("already"));
        assert!(target(&json!({"mode":"missing"}), "ask", &profiles)
            .unwrap_err()
            .to_string()
            .contains("unknown"));
        assert_eq!(
            target(&json!({"mode":"yolo"}), "ask", &profiles).unwrap()["name"],
            "Yolo"
        );
    }
}
