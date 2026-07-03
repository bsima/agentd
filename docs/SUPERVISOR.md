# agentd supervisor design (M2)

Status: **design only — not implemented.** The Haskell supervisor
(`~/omni/live/Omni/Agentd`) is the working reference; this document is the
design for its Rust port. Nothing here ships until the open questions are
settled and the acceptance bar below is met.

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
    agent.md            # prompt/frontmatter the session was started with
    fifo                # turn delivery (mkfifo)
    pid                 # supervisor-written, liveness-checked via kill -0
    run-id              # stable AGENT_RUN_ID for traces
    checkpoints/        # --checkpoint-dir
    stdout.jsonl        # captured --json stdout (machine events)
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
| `start` | Create the session directory, mkfifo, spawn `agent --fifo ... --checkpoint-dir ... --json`, write pid/run-id. Fails if already running. |
| `stop` | SIGTERM; the agent's FIFO loop exits on signal (shipped behavior). Escalate to SIGKILL after a timeout. |
| `resume` | `start` with `--resume <latest checkpoint>`; checkpoint repair on load is the agent's job (shipped behavior). |
| `status` | Directory walk + `kill -0` + latest checkpoint mtime + last trace event. |
| `logs` | `tail -f` the trace JSONL (or `stdout.jsonl` with `--raw`). |
| `gen-systemd` | Emit a `agentd-<name>.service` unit (Restart=on-failure, ExecStart=`agent --fifo ...`) for sessions that should outlive the user session. |

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
  There is no DB row to disagree with.
- **Config mutation is a command, not a file convention:**
  `agentd set-model <name> <model>` (and `set-*` generally) edits the
  canonical spec in place and prints whether a restart is needed. Editing
  `agent.md` by hand is equally valid — same file, same effect.
- **Generated files must not look editable.** Anything the supervisor derives
  (systemd units, pid files) either carries a `GENERATED — edits are
  overwritten by agentd` header or is obviously runtime state. Nothing that
  influences the next launch may be both generated and silently regenerated.
- Acceptance addition: edit the spec's model (by hand or via `set-model`),
  `agentd resume`, and verify via the startup banner/`agent_start` trace
  event (or `/proc/<pid>/environ`) that the new model is live.

## Open questions (settle before implementing)

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
2. **Should `send` have a timeout?** A wedged turn currently blocks the
   sender forever. Probably: `--timeout` with the turn left running.
3. **Haskell compatibility.** Does the Rust `agentd` need to read the Haskell
   supervisor's session layout, or is this a clean break with a migration
   note? (`evals/agentd-persistent.sh` pins the agent-side contract either
   way.)
4. **Multi-user / multi-host.** Out of scope for M2 (M5 territory), but the
   directory layout should not bake in single-host assumptions that M5 has to
   undo — e.g. keep paths relative to `$AGENTD_HOME`.

## Non-goals

- No daemon, broker, or RPC server. The filesystem is the API.
- No scheduling/cron (use systemd timers writing to the FIFO).
- No multi-agent orchestration — that is `Infer`-calling-`Infer` inside agent
  programs, not a supervisor feature.

## Acceptance

- `start`/`send`/`logs`/`stop`/`status`/`resume` work against the shipped
  `agent` binary; the only agent-side change is the turn-id envelope
  (option A — decided 2026-07-03 and shipped in t-1308.2, not drifted
  into).
- A SIGTERM'd or crashed session resumes from its latest checkpoint with
  history intact, including the dangling-tool-call repair path.
- `evals/agentd-persistent.sh` (or its Rust replacement) passes against the
  Rust supervisor.
- Generated systemd units survive logout and restart on failure.
