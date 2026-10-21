//! ACP-local, read-only slash commands. They never enter the runtime loop or
//! trace: recording them as turns would shift replay effect visits and hashes.
use agent_client_protocol::schema::v1::{
    AvailableCommand, AvailableCommandsUpdate, ContentBlock, SessionId, SessionNotification,
    SessionUpdate,
};
use agent_client_protocol::{Client, ConnectionTo};
use agent_core::{Event, TraceLogger};
use anyhow::{anyhow, Result};

use crate::{CostReport, GcStatsReport, Runtime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Command {
    Cost,
    Gc,
    Help,
}

const COMMANDS: &[(Command, &str, &str)] = &[
    (
        Command::Cost,
        "cost",
        "Session token and USD totals by model",
    ),
    (
        Command::Gc,
        "gc",
        "Current context and garbage collection status",
    ),
    (Command::Help, "help", "List available commands"),
];

pub(super) fn advertise(cx: &ConnectionTo<Client>, session_id: SessionId) -> Result<()> {
    let commands = COMMANDS
        .iter()
        .map(|(_, name, description)| AvailableCommand::new(*name, *description))
        .collect();
    cx.send_notification(SessionNotification::new(
        session_id,
        SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands)),
    ))
    .map_err(|err| anyhow!("advertising ACP commands: {err}"))
}

/// Only a single text block whose first whitespace-delimited word is a
/// registered slash command is intercepted. These commands ignore trailing arguments. Everything else is model input.
pub(super) fn parse(blocks: &[ContentBlock]) -> Option<Command> {
    let [ContentBlock::Text(text)] = blocks else {
        return None;
    };
    let text = text.text.trim();
    let name = text.split_whitespace().next()?;
    COMMANDS
        .iter()
        .find(|(_, registered, _)| name == format!("/{registered}"))
        .map(|(command, _, _)| *command)
}

async fn read_session_events(runtime: &Runtime) -> Result<Vec<Event>> {
    match TraceLogger::read_events(&runtime.trace_path).await {
        Ok(events) => Ok(events),
        Err(err)
            if err
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
        {
            Ok(Vec::new())
        }
        Err(err) => Err(err),
    }
}

pub(super) async fn execute(command: Command, runtime: &Runtime) -> Result<String> {
    match command {
        Command::Help => Ok(COMMANDS
            .iter()
            .map(|(_, name, description)| format!("/{name} — {description}\n"))
            .collect()),
        Command::Cost => {
            let events = read_session_events(runtime).await?;
            // The trace's paired InferCall/InferResult events carry the model
            // and priced integer usage, including child infers. RunUsage in
            // the live sink cannot provide a per-model breakdown on load.
            Ok(CostReport::from_events(&events).render())
        }
        Command::Gc => {
            let events = read_session_events(runtime).await?;
            let count = GcStatsReport::from_events(&events).fires.len();
            let used = events.iter().rev().find_map(|event| match event {
                Event::InferResult {
                    parent_op_id: None,
                    total_tokens,
                    ..
                } if *total_tokens > 0 => Some(*total_tokens as u64),
                Event::InferCall {
                    parent_op_id: None,
                    prompt: Some(prompt),
                    ..
                } => Some(agent_core::estimate_tokens(prompt) as u64),
                _ => None,
            });
            Ok(format!(
                "GC strategy: {}\nThreshold: {}\nContext used/size: {} / {} tokens (last root inference)\nCollections: {}\n",
                super::registry::gc_value_id(&runtime.config.gc),
                runtime.config.gc_threshold,
                used.map_or_else(|| "unknown".to_owned(), |n| n.to_string()),
                runtime.config.context_budget,
                count,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::TextContent;

    #[test]
    fn commands_only_match_single_exact_text_block() {
        let text = |s: &str| ContentBlock::Text(TextContent::new(s));
        assert_eq!(parse(&[text("  /cost  ")]), Some(Command::Cost));
        assert_eq!(parse(&[text("/gc")]), Some(Command::Gc));
        assert_eq!(parse(&[text("/help")]), Some(Command::Help));
        assert_eq!(parse(&[text("/cost extra")]), Some(Command::Cost));
        for blocks in [
            vec![text("/costly")],
            vec![text("/unknown")],
            vec![text("/cost"), text("more")],
        ] {
            assert_eq!(parse(&blocks), None);
        }
    }
}
