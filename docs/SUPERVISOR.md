# agentd supervisor design (M2)

Status: **implemented** (t-1341) as `crates/agentd` (binary: `agentd`). All
open questions below are decided; the acceptance bar is covered by
`crates/agentd/tests/supervisor.rs` (credential-free integration tests) and
`evals/agentd-persistent.sh` (offline end-to-end eval over the real
binaries). The Rust supervisor is a **clean break** from the Haskell
supervisor's session layout — see "Migration from the Haskell supervisor"
below.

## Goal

Manage named, long-running agent sessions around the existing `agent` process
model:

```sh
agentd start coder              # spawn an agent session named "coder"
agentd send coder "run tests"   # deliver a turn, print the response
agentd logs coder               # follow the session's trace
agentd status                   # list sessions, liveness, last checkpoint
agentd stop coder               # graceful shutdown
agentd resume coder             # restart from the latest checkpoint
```

The supervisor adds **naming, delivery, and lifecycle** — nothing else. The
agent process already owns the loop, traces, checkpoints, and replay.

## Design stance: thin CLI over conventions, not a daemon

The README's thesis is "Linux is the harness". Applied here: `agentd` should
be a CLI that manipulates a conventional directory layout and ordinary
processes, not a long-running broker that owns state. Supervision-as-such
(restart on crash, resource limits, boot integration) is delegated to the
init system; `agentd` knows how to *generate* the systemd units rather than
reimplement them.

```text
$AGENTD_HOME (default ~/.local/share/agentd)/
  <name>/
    agent.md            # CANONICAL session spec (YAML frontmatter + body)
    fifo                # turn delivery (mkfifo)
    pid                 # supervisor-written, liveness-checked via kill -0
    run-id              # stable AGENT_RUN_ID for traces
    checkpoints/        # --checkpoint-dir
    stdout.jsonl        # captured --json stdout (machine events)
    stderr.log          # captured child stderr (startup banner, errors)
    send.lock           # per-session flock taken by `agentd send`
```

A session "exists" if its directory exists; it is "running" if the pid is
live. No registry database. Two `agentd` invocations coordinate through the
filesystem only.

## Turn delivery and response correlation

`agent` already reads NUL-framed turns from a FIFO. The open problem is the
**response path**: the agent writes responses to its stdout, which the
spawning supervisor — not the `agentd send` process — holds.

Decision: run sessions with `--json` and capture stdout to
`stdout.jsonl`. The machine events (`agent_start`, `agent_complete`,
`agent_error`) are the supervisor's interface; `agentd send` works by:

1. recording the current `stdout.jsonl` offset,
2. writing the NUL-framed turn — a v1 turn envelope carrying a
   supervisor-generated `turn_id` — to the FIFO (under a per-session flock —
   FIFO writes from concurrent senders interleave beyond `PIPE_BUF`),
3. tailing `stdout.jsonl` from the offset until the
   `agent_complete`/`agent_error` event carrying that `turn_id` arrives,
   then printing the response.

This makes `send` restartable: if the sender dies, the response is still on
disk, keyed by an id the sender can recompute or persist.

### Send timeouts and re-attachment (decided, t-1308.6 / DR-10)

`agentd send --timeout <secs>` times out the **caller**, never the turn: on
expiry the sender exits with code **124** (the `timeout(1)` convention,
distinct from 1 = the turn itself errored) after printing the turn id and
the re-attach command. There is no default kill — a wedged turn is the
operator's call, via `agentd stop`.

`agentd attach <name> <turn_id>` retrieves a turn's result at any later
point: it scans `stdout.jsonl` from the start for the matching
`agent_complete`/`agent_error` and, if the turn is still running, keeps
tailing (optionally bounded by its own `--timeout`).

### Turn envelope (v1) — shipped agent-side (t-1308.2)

A NUL-framed session turn (FIFO or `--session` stdin) is either a raw
prompt or a turn envelope:

```json
{"v": 1, "turn_id": "send-4f2a", "input": "run tests", "metadata": {"sender": "ben"}}
```

- A frame that parses as a JSON object with `"v": 1` (exactly the integer
  1) and a string `"input"` field is an envelope; `input` is the prompt.
  **Any other frame is a raw prompt, byte-for-byte** — full backward
  compatibility with pre-envelope senders.
- `turn_id` (optional string): echoed in the `data` of the turn's
  `agent_start`, `agent_complete`, and `agent_error` machine events. When
  absent — including for every raw frame — the agent mints
  `<run_id>-t<seq>`, where `seq` is the 0-based turn ordinal within the run
  (continued across `--resume` from the checkpoint sequence). Every
  completion or error event therefore carries a `turn_id`, supplied or
  minted.
- `metadata` (optional, opaque JSON): echoed verbatim as `metadata` on the
  turn's `agent_complete` event. Not present on `agent_error`.
- Edge case: a raw prompt whose text happens to be a valid v1 envelope
  will be parsed as one. To deliver such text literally, wrap it:
  `{"v": 1, "input": "<that text>"}`.

## Lifecycle

| Command | Behavior |
|---|---|
| `start` | Create the session directory, seed `agent.md` from flags if absent (refused if it exists — the spec is canonical), mkfifo, spawn a detached `agent --fifo ... --checkpoint-dir ... --json` with spec-derived flags, write pid/run-id. Fails if already running. Extra child argv after `--` (eval plumbing like `--replay-trace`). |
| `stop` | SIGTERM; the agent's FIFO loop exits on signal (shipped behavior). Escalate to SIGKILL after `--grace` (default 5s). |
| `resume` | `start` with `--resume <latest checkpoint>` (fresh start when none exists yet, so systemd's first ExecStart works); checkpoint repair on load is the agent's job (shipped behavior). Reads `agent.md` fresh, so spec edits apply here. |
| `send` | Envelope + flock + offset-tail (above); `--timeout` exits 124 and leaves the turn running. |
| `attach` | Re-attach to a turn by id: print its result from disk, or wait for it. |
| `status` | Directory walk + `kill -0` + latest checkpoint mtime + last event + pending approval count from the agent's approvals dir (`--json` for scripts). |
| `logs` | Print/`--follow` the trace JSONL (or `stdout.jsonl` with `--raw`). |
| `set-model` / `set-provider` / `set-system-prompt` / `set-max-turns` | Edit the canonical `agent.md` frontmatter in place; prints whether a restart is needed. |
| `gen-systemd` | Emit a `agentd-<name>.service` unit (Type=forking + PIDFile, Restart=on-failure, ExecStart through `agentd resume`, ExecStop through `agentd stop`, GENERATED header) for sessions that should outlive the user session. |

Approvals need no passthrough: `agent approvals` already owns resolution;
`agentd status` surfaces the pending count per session run.

Crash recovery is `resume`, driven either manually, by systemd
`Restart=on-failure` (with `ExecStart` going through `agentd resume`), or not
at all — the checkpoint model makes restart safe, so the supervisor does not
need its own babysitting loop.

## Design constraint: the spec file is the single source of truth (t-1105)

The Haskell supervisor stores the canonical agent spec in a DB row and
*regenerates* a per-agent `.env` from it on `start`. That file looks editable
but isn't: hand-edits are silently reverted and the process relaunches on the
old config. This has eaten real model changes twice (2026-05-30 gc-coder,
2026-06-08 designer), with `rm`/`create` (archiving history) as the only
workaround. The Rust port must make this failure mode structurally
impossible:

- **The on-disk spec (`<name>/agent.md`) is canonical.** `start` and `resume`
  read it fresh on every launch; nothing regenerates it from a shadow store.
  There is no DB row to disagree with. Config flags on `start` only SEED a
  missing spec; once it exists they are refused with a pointer to `set-*`.
- **Config mutation is a command, not a file convention:**
  `agentd set-model <name> <model>` (and `set-*` generally) edits the
  canonical spec in place and prints whether a restart is needed. Editing
  `agent.md` by hand is equally valid — same file, same effect. Unknown
  frontmatter keys and the body survive edits.
- **Generated files must not look editable.** Anything the supervisor derives
  (systemd units, pid files) either carries a `GENERATED — edits are
  overwritten by agentd` header or is obviously runtime state. Nothing that
  influences the next launch may be both generated and silently regenerated.
- Acceptance addition (met — `set_model_then_resume_shows_new_model_in_agent_start`):
  edit the spec's model (by hand or via `set-model`), `agentd resume`, and
  verify via the `agent_start` machine event that the new model is live.

The spec format is the agent's own markdown-prompt convention: YAML
frontmatter (`model`, `provider`, `system_prompt`, `max_iterations` — alias
`max_turns` accepted on read) followed by a markdown body. Two supervisor
additions: a non-empty body serves as the system prompt when the
frontmatter has none (the body of `agent.md` *describes the agent*), and an
optional `args` list of extra `agent` argv appended at launch (e.g.
`["--memory-dir", "memory"]`). `system_prompt` may be literal text or a
path resolved relative to the session directory.

## Open questions (all decided)

1. **Concurrent `send` semantics. Decided: Option A** (2026-07-03) — tag
   each turn with a supervisor-generated turn id the agent echoes, via the
   v1 turn envelope specified above. Option B (serialize senders under the
   flock, match responses by order) was rejected: order-based matching
   breaks under send timeouts — a timed-out sender that stops tailing
   desyncs the order ledger for everyone behind it — and gives silent
   misattribution when the agent crashes and restarts mid-queue (quiet
   corruption, versus a detectably unmatched id under Option A). Approvals,
   the trace schema (S.2b), and SDK futures all need a turn id anyway. The
   per-session flock is retained regardless of this decision: it guarantees
   frame atomicity beyond `PIPE_BUF`, which is orthogonal to correlation.
2. **Should `send` have a timeout? Decided: yes, caller-side only**
   (t-1308.6 / DR-10, shipped) — `--timeout <secs>` exits 124 with the
   turn id and the `agentd attach` command; the turn keeps running. No
   default kill. See "Send timeouts and re-attachment" above.
3. **Haskell compatibility. Decided: clean break** with a migration note
   (below). The Rust `agentd` does not read the Haskell supervisor's
   session layout or DB; `evals/agentd-persistent.sh` now pins the Rust
   supervisor end-to-end instead of forwarding to the Haskell suite.
4. **Multi-user / multi-host.** Out of scope for M2 (M5 territory), but the
   directory layout should not bake in single-host assumptions that M5 has to
   undo — e.g. keep paths relative to `$AGENTD_HOME` (the implementation
   derives every path from `$AGENTD_HOME/<name>/`, nothing is stored
   absolute).

## Migration from the Haskell supervisor

The Haskell `agentd` (`~/omni/live/Omni/Agentd`) supervised the same Rust
`agent` binary, but kept the canonical spec in a DB row and its own session
layout. The Rust supervisor reads neither. To migrate a session:

1. Stop it under the Haskell supervisor.
2. `agentd start <name> --model <model> ...` once (or write
   `$AGENTD_HOME/<name>/agent.md` yourself) to create the canonical spec —
   this is also the moment hand-editable config becomes real (t-1105).
3. To carry history over, copy the old session's agent-written checkpoint
   files (`checkpoint-*.json`, `session-latest.json` — same format, same
   `agent` binary) into `$AGENTD_HOME/<name>/checkpoints/` before the first
   `agentd resume <name>`. Without them the session simply starts fresh.

The agent-side contracts (NUL-framed turns, v1 envelope, checkpoint JSON,
machine events) are unchanged, so nothing about recorded traces or
checkpoints needs converting.

## Non-goals

- No daemon, broker, or RPC server. The filesystem is the API.
- No scheduling/cron (use systemd timers writing to the FIFO).
- No multi-agent orchestration — that is `Infer`-calling-`Infer` inside agent
  programs, not a supervisor feature.

## Acceptance (met)

- `start`/`send`/`logs`/`stop`/`status`/`resume` work against the shipped
  `agent` binary; the only agent-side change is the turn-id envelope
  (option A — decided 2026-07-03 and shipped in t-1308.2, not drifted
  into). No further agent-side changes were needed.
  → `crates/agentd/tests/supervisor.rs::start_send_status_stop_roundtrip`,
  `concurrent_sends_correlate_by_turn_id`.
- A SIGTERM'd or crashed session resumes from its latest checkpoint with
  history intact (the dangling-tool-call repair path is the agent's,
  covered by `agent-sdk/tests/session.rs`).
  → `kill_dash_nine_then_resume_keeps_history_intact`.
- Send timeout leaves the turn running; attach retrieves it.
  → `send_timeout_leaves_turn_running_and_attach_recovers`.
- Spec edits are live on resume.
  → `set_model_then_resume_shows_new_model_in_agent_start`.
- `evals/agentd-persistent.sh` passes against the Rust supervisor, offline.
- Generated systemd units survive logout and restart on failure
  (Type=forking + PIDFile + Restart=on-failure through `agentd resume`;
  golden-checked in tests, `systemd-analyze verify` in the eval when
  available).
