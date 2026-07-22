# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## What this project is

`agentd` is a Rust runtime for long-running AI agents. The central design choice
is that **Linux is the harness** and **agent programs are data**: an agent
program is serializable AgentIR interpreted by a runtime. You can inspect,
replay, test, sandbox, parallelize, or distribute the same agent program by
swapping interpreters.

Core ideas (see `README.md` and `ARCHITECTURE.md` for the long version):

- An agent alternates between `infer(unstructured)` and `eval(structured)`. The
  loop is the agent.
- The effect algebra is `Infer`, `Eval`, `Retrieve`, `Store`, `Emit`, plus
  `Par` for fan-out; the CLI runtime is the serializable AgentIR
  (`agent-core::ir`); the closure-based `OpF` free monad remains a library
  builder/test API.
- `Infer` can call `Infer`, so multi-agent orchestration is just an agent program,
  not a special framework layer (the SICP meta-circular idea applied to agents).
- Context is a managed window over an append-only log: hydration sources feed
  it, GC keeps it under budget (`docs/GC.md`).
- Sessions are Unix processes; the protocol is pipes and files.

This started as a Rust port of a Haskell prototype (`Omni/Agent/Op.hs`); the
Rust implementation has long since become the reference. See `ROADMAP.md` for
milestone status.

## Source layout

- `crates/agent-core` — the runtime kernel: AgentIR + machine, interpreters,
  providers, hydration, PromptIR, GC, approvals, cost, tracing.
- `crates/agent` — the CLI (`agent`): one-shot prompts, NUL/FIFO sessions,
  checkpoints, traces, replay, markdown prompts.
- `crates/agentd` — the supervisor CLI: named long-running sessions, turn
  delivery, lifecycle, systemd unit generation (`docs/SUPERVISOR.md`).
- `crates/agent-sdk` — the embedding SDK: typed tools, output contracts,
  approvals, replay.
- `crates/agent-oauth` — codex/claude-code subscription auth.
- `docs/` — design docs: `AGENT_IR.md`, `PROMPT_IR.md`, `GC.md`, `MEMORY.md`,
  `PROVIDERS.md`, `GUIDANCE.md`, `SUPERVISOR.md`, `TRACE_SCHEMA.md`, `OTEL.md`,
  `EMPTY_COMPLETION.md`. Read the relevant one before touching that subsystem.
- `evals/` — offline + online eval harness (`release.sh`, `release-online.sh`)
  plus recorded behavioral evals (`gc/`, `delegation/`, `infer-infer/`).
- `examples/` — `models.yaml` template.
- `agents/` — example agent prompt files (e.g. `coder.md`) and session FIFOs.

## Build, test, verify

This is a Cargo workspace with a Nix dev shell. There is **no `bild`** here —
that's the omnirepo's tool, not this repo's.

```sh
# Get the toolchain (rustc, cargo, rustfmt, clippy, openssl, ...)
nix develop          # or: direnv allow, the .envrc loads the shell

# Fast inner loop
cargo build
cargo test                       # whole workspace
cargo test -p agent-core         # one crate

# Required checks before you call a change done
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
```

CI (`.github/workflows/ci.yml`) runs on `pull_request` and pushes to `main`,
gated on `./evals/release.sh`. Match it locally: fmt clean, clippy clean with
`-D warnings`, tests green. Do not leave warnings.

When you change behavior of `Infer`/`Eval`/trace output, check whether the eval
replay fixtures need regenerating (see `evals/` and the trace `--replay-trace`
flow in the README). Do not hand-edit fixtures blindly.

Do **not** overwrite an existing `~/.config/agent/models.yaml`; it is runtime
config that may contain local aliases used by deployed services.

## Git workflow: use git-branchless

This repo uses a **git-branchless / patch-based** workflow. Work with commits
directly; use stacking to organize related changes.

```sh
git smartlog                 # visualize the commit graph
git add . && git commit      # create a commit
git amend                    # amend the current commit (after more changes)
git move -s <src> -d <dst>   # restructure
git restack                  # repair the stack after rewriting history
```

Conventions:

- Make small, focused commits — one logical change each.
- Write descriptive messages: what and why, not just what.
- Keep history clean with `git amend` / `git restack`.
- Run the required checks (fmt, clippy, test) before committing.
- If a commit closes a tracked task, add a `Task-Id: t-NNN` trailer.

NEVER do these without an explicit request from the maintainer:

- `git push` / `git pull` (no remote sync unless asked)
- force pushes or other destructive/history-losing operations
- branch deletion or remote branch operations

The maintainer controls when code is shared. Land your work as clean local
commits and hand off; let them do the push/merge pass.

## Task tracking

If a `task` CLI is available in your environment, use it for issue tracking
rather than markdown TODO lists or `TODO`/`FIXME` comments in code:

```sh
task ready --json                 # find ready work
task create "Title" --description-file=/tmp/desc.md --json
task start t-123 --json           # claim
task update t-123 review --json   # finished; hand off to maintainer
task update t-123 needs-help --json
task comment t-123 "note" --json
```

Status flow: `Open -> InProgress -> Review -> Done` (maintainer verifies and
marks `Done`; agents do not self-complete). `InProgress -> NeedsHelp` when
blocked. Always pass `--json`.

## Style notes

- Idiomatic Rust; the IR effects and the interpreter are the source of truth.
- Don't add ambient global mutation for trace/context propagation — follow the
  existing seam patterns (e.g. `TraceSink`, env-injection for context).
- Mount source read-only and run agents in disposable workspaces when testing
  the shell-backed `Eval` (see "Running safely" in `README.md`).
