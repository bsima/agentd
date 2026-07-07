//! `agentd` — supervisor for named, long-running `agent` sessions
//! (docs/SUPERVISOR.md). A thin CLI over a conventional directory layout
//! and ordinary processes: naming, turn delivery, and lifecycle. The agent
//! process owns the loop, traces, checkpoints, and replay; init systems own
//! supervision-as-such (`agentd gen-systemd`).

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

mod events;
mod send;
mod session;
mod spec;
mod systemd;

use session::{Session, SpecSeed};

/// Exit code for a `send`/`attach` that timed out while the turn is still
/// running agent-side (distinct from 1 = the turn itself errored). Matches
/// coreutils `timeout(1)`.
const EXIT_TIMEOUT: i32 = 124;

#[derive(Debug, Parser)]
#[command(
    name = "agentd",
    version,
    about = "Supervise named, long-running agent sessions: start/stop/resume, deliver turns, tail logs",
    after_help = "State lives under $AGENTD_HOME (default ~/.local/share/agentd)/<name>/: \
                  agent.md (canonical spec), fifo, pid, run-id, stdout.jsonl, checkpoints/. \
                  A session exists if its directory exists; it is running if its pid is live."
)]
struct Args {
    /// Session state root (default: ~/.local/share/agentd).
    #[arg(long, env = "AGENTD_HOME", global = true)]
    home: Option<PathBuf>,
    /// The supervised `agent` binary to launch.
    #[arg(long, env = "AGENTD_AGENT_BIN", global = true, default_value = "agent")]
    agent_bin: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Spawn a fresh agent session named <name>. Fails if already running.
    ///
    /// Config flags only SEED a missing <name>/agent.md; once the spec
    /// exists it is canonical — edit it (or `agentd set-*`) instead.
    Start {
        name: String,
        /// Model alias/id, written to the spec when seeding it.
        #[arg(long)]
        model: Option<String>,
        /// Provider tag or base URL, written to the spec when seeding it.
        #[arg(long)]
        provider: Option<String>,
        /// System prompt text (or a path, resolved at launch), written to
        /// the spec when seeding it.
        #[arg(long)]
        system_prompt: Option<String>,
        /// Per-turn iteration ceiling, written to the spec when seeding it.
        #[arg(long)]
        max_turns: Option<usize>,
        /// Extra arguments passed verbatim to the `agent` child (after --).
        #[arg(last = true)]
        agent_args: Vec<String>,
    },
    /// Stop a session: SIGTERM, escalating to SIGKILL after the grace period.
    Stop {
        name: String,
        /// Seconds to wait after SIGTERM before SIGKILL.
        #[arg(long, default_value_t = 5)]
        grace: u64,
    },
    /// Restart a session from its latest checkpoint (fresh start when none
    /// exists yet). Reads <name>/agent.md fresh, so spec edits apply here.
    Resume {
        name: String,
        /// Extra arguments passed verbatim to the `agent` child (after --).
        #[arg(last = true)]
        agent_args: Vec<String>,
    },
    /// Show session liveness, model, latest checkpoint, last event, and
    /// pending approval count (all sessions when <name> is omitted).
    Status {
        name: Option<String>,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Print the session's trace (or raw --json stdout with --raw).
    Logs {
        name: String,
        /// Tail <name>/stdout.jsonl (machine events) instead of the trace.
        #[arg(long)]
        raw: bool,
        /// Number of trailing lines to print.
        #[arg(short = 'n', long, default_value_t = 20)]
        lines: usize,
        /// Keep following appended output.
        #[arg(short = 'f', long)]
        follow: bool,
    },
    /// Deliver one turn and print the response.
    ///
    /// The prompt rides a v1 turn envelope with a supervisor-generated
    /// turn id; the response is correlated by that id from the session's
    /// machine events, so concurrent sends never cross wires.
    Send {
        name: String,
        prompt: String,
        /// Give up WAITING after this many seconds. The turn keeps running;
        /// re-attach with `agentd attach <name> <turn_id>`. Exit code 124.
        #[arg(long)]
        timeout: Option<u64>,
        /// Correlation id for the turn (minted when omitted).
        #[arg(long)]
        turn_id: Option<String>,
        /// Opaque JSON echoed back on the turn's agent_complete event.
        #[arg(long)]
        metadata: Option<String>,
    },
    /// Re-attach to a turn (e.g. after `send --timeout`): print its result
    /// if already on disk, else wait for it.
    Attach {
        name: String,
        turn_id: String,
        /// Give up waiting after this many seconds (exit code 124).
        #[arg(long)]
        timeout: Option<u64>,
    },
    /// Set the model in <name>/agent.md (the canonical spec) in place.
    SetModel { name: String, model: String },
    /// Set the provider in <name>/agent.md in place.
    SetProvider { name: String, provider: String },
    /// Set the system prompt (text or path) in <name>/agent.md in place.
    SetSystemPrompt { name: String, system_prompt: String },
    /// Set the per-turn iteration ceiling in <name>/agent.md in place.
    SetMaxTurns { name: String, max_turns: usize },
    /// Emit a systemd user unit (agentd-<name>.service) that supervises
    /// the session: Restart=on-failure through `agentd resume`.
    GenSystemd {
        name: String,
        /// Write the unit here instead of stdout.
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    let home = session::agentd_home(args.home.clone())?;
    match args.command {
        Command::Start {
            name,
            model,
            provider,
            system_prompt,
            max_turns,
            agent_args,
        } => {
            let session = Session::new(home, &name)?;
            let (pid, run_id) = session::launch(
                &session,
                session::Launch {
                    agent_bin: &args.agent_bin,
                    seed: SpecSeed {
                        model,
                        provider,
                        system_prompt,
                        max_turns,
                    },
                    resume: false,
                    extra_args: &agent_args,
                },
            )?;
            println!("started '{name}' (pid {pid}, run {run_id})");
            Ok(())
        }
        Command::Resume { name, agent_args } => {
            let session = Session::new(home, &name)?;
            let (pid, run_id) = session::launch(
                &session,
                session::Launch {
                    agent_bin: &args.agent_bin,
                    seed: SpecSeed::default(),
                    resume: true,
                    extra_args: &agent_args,
                },
            )?;
            println!("resumed '{name}' (pid {pid}, run {run_id})");
            Ok(())
        }
        Command::Stop { name, grace } => {
            let session = Session::new(home, &name)?;
            match session::stop(&session, Duration::from_secs(grace))? {
                None => println!("'{name}' is not running"),
                Some(false) => println!("stopped '{name}'"),
                Some(true) => println!("stopped '{name}' (SIGKILL after {grace}s grace)"),
            }
            Ok(())
        }
        Command::Status { name, json } => run_status(&home, name.as_deref(), json),
        Command::Logs {
            name,
            raw,
            lines,
            follow,
        } => run_logs(&home, &name, raw, lines, follow),
        Command::Send {
            name,
            prompt,
            timeout,
            turn_id,
            metadata,
        } => {
            let session = Session::new(home, &name)?;
            let metadata = metadata
                .as_deref()
                .map(|raw| {
                    serde_json::from_str::<serde_json::Value>(raw)
                        .context("--metadata must be valid JSON")
                })
                .transpose()?;
            let deadline = timeout.map(|secs| Instant::now() + Duration::from_secs(secs));
            let turn_id = turn_id.unwrap_or_else(send::mint_turn_id);
            let offset = send::deliver(&session, &turn_id, &prompt, metadata.as_ref(), deadline)?;
            eprintln!("turn {turn_id} delivered to '{name}'");
            finish_wait(
                send::await_turn(&session, &turn_id, offset, deadline)?,
                &name,
                &turn_id,
            )
        }
        Command::Attach {
            name,
            turn_id,
            timeout,
        } => {
            let session = Session::new(home, &name)?;
            session.require_exists()?;
            let deadline = timeout.map(|secs| Instant::now() + Duration::from_secs(secs));
            finish_wait(
                send::await_turn(&session, &turn_id, 0, deadline)?,
                &name,
                &turn_id,
            )
        }
        Command::SetModel { name, model } => set_spec_key(&home, &name, "model", &model),
        Command::SetProvider { name, provider } => {
            set_spec_key(&home, &name, "provider", &provider)
        }
        Command::SetSystemPrompt {
            name,
            system_prompt,
        } => set_spec_key(&home, &name, "system_prompt", &system_prompt),
        Command::SetMaxTurns { name, max_turns } => {
            set_spec_key(&home, &name, "max_iterations", &max_turns.to_string())
        }
        Command::GenSystemd { name, output } => {
            let session = Session::new(home.clone(), &name)?;
            session.require_exists()?;
            let agentd_bin = std::env::current_exe().context("resolving the agentd binary path")?;
            let text = systemd::unit(&name, &home, &agentd_bin);
            match output {
                None => {
                    print!("{text}");
                }
                Some(path) => {
                    std::fs::write(&path, &text)
                        .with_context(|| format!("writing {}", path.display()))?;
                    eprintln!(
                        "wrote {}; install with `systemctl --user enable --now {}` \
                         after linking it into ~/.config/systemd/user/",
                        path.display(),
                        systemd::unit_name(&name)
                    );
                }
            }
            Ok(())
        }
    }
}

/// Print a wait outcome and exit accordingly: response on stdout (0), turn
/// error on stderr (1), timeout notice with the re-attach command (124).
fn finish_wait(outcome: send::Outcome, name: &str, turn_id: &str) -> Result<()> {
    match outcome {
        send::Outcome::Complete { response } => {
            println!("{response}");
            Ok(())
        }
        send::Outcome::Error { message } => {
            bail!("turn {turn_id} failed: {message}")
        }
        send::Outcome::TimedOut => {
            eprintln!(
                "timed out waiting for turn {turn_id}; the turn is still running. \
                 Re-attach with: agentd attach {name} {turn_id}"
            );
            std::process::exit(EXIT_TIMEOUT);
        }
    }
}

fn set_spec_key(home: &std::path::Path, name: &str, key: &str, raw: &str) -> Result<()> {
    let session = Session::new(home.to_path_buf(), name)?;
    session.require_exists()?;
    let path = session.spec_path();
    let mut loaded = if path.is_file() {
        spec::Spec::load(&path)?
    } else {
        spec::Spec::default()
    };
    loaded.set(key, spec::set_value_for(key, raw)?);
    // Validate the result parses as a config before writing it back.
    loaded.config()?;
    loaded.save(&path)?;
    println!("set {key} = {raw} in {}", path.display());
    if session.running().is_some() {
        println!(
            "session '{name}' is running on the old config; apply with: \
             agentd stop {name} && agentd resume {name}"
        );
    }
    Ok(())
}

fn run_status(home: &std::path::Path, name: Option<&str>, json: bool) -> Result<()> {
    let names = match name {
        Some(name) => vec![name.to_string()],
        None => session::list_sessions(home)?,
    };
    let mut statuses = Vec::new();
    for name in &names {
        let session = Session::new(home.to_path_buf(), name)?;
        statuses.push(session::status(&session)?);
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&statuses)?);
        return Ok(());
    }
    if statuses.is_empty() {
        println!("no sessions under {}", home.display());
        return Ok(());
    }
    let header = format!(
        "{:<20} {:<8} {:<8} {:<24} {:<20} {:<28} {}",
        "NAME", "RUNNING", "PID", "MODEL", "LAST CHECKPOINT", "LAST EVENT", "PENDING APPROVALS"
    );
    println!("{header}");
    for status in &statuses {
        let checkpoint = status
            .last_checkpoint
            .map(|ts| ts.format("%Y-%m-%dT%H:%M:%SZ").to_string())
            .unwrap_or_else(|| "-".into());
        let event = match (&status.last_event, &status.last_event_ts) {
            (Some(name), Some(ts)) => format!("{name} @ {ts}"),
            (Some(name), None) => name.clone(),
            _ => "-".into(),
        };
        println!(
            "{:<20} {:<8} {:<8} {:<24} {:<20} {:<28} {}",
            status.name,
            if status.running { "yes" } else { "no" },
            status
                .pid
                .map(|pid| pid.to_string())
                .unwrap_or_else(|| "-".into()),
            status.model.as_deref().unwrap_or("-"),
            checkpoint,
            event,
            status.pending_approvals,
        );
    }
    Ok(())
}

fn run_logs(
    home: &std::path::Path,
    name: &str,
    raw: bool,
    lines: usize,
    follow: bool,
) -> Result<()> {
    let session = Session::new(home.to_path_buf(), name)?;
    session.require_exists()?;
    let path = if raw {
        session.stdout_path()
    } else {
        let run_id = session
            .run_id()
            .ok_or_else(|| anyhow!("session '{name}' has no run-id yet; has it been started?"))?;
        session::trace_path(&run_id)?
    };
    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(err) => return Err(err).with_context(|| format!("reading {}", path.display())),
    };
    let mut stdout = std::io::stdout().lock();
    let all: Vec<&str> = content.lines().collect();
    for line in all.iter().skip(all.len().saturating_sub(lines)) {
        writeln!(stdout, "{line}")?;
    }
    stdout.flush()?;
    if !follow {
        return Ok(());
    }
    let mut tail = events::Tail::from_offset(&path, content.len() as u64);
    loop {
        for line in tail.read_new_lines()? {
            writeln!(stdout, "{line}")?;
        }
        stdout.flush()?;
        std::thread::sleep(Duration::from_millis(200));
    }
}
