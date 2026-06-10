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
2. writing the NUL-framed turn to the FIFO (under a per-session flock —
   FIFO writes from concurrent senders interleave beyond `PIPE_BUF`),
3. tailing `stdout.jsonl` from the offset until the matching
   `agent_complete`/`agent_error` arrives, then printing the response.

This keeps the agent binary unchanged and makes `send` restartable: if the
sender dies, the response is still on disk.

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

## Open questions (settle before implementing)

1. **Concurrent `send` semantics.** Turns queue in the FIFO and the agent
   processes them serially; two senders' responses must be matched to their
   turns. Option A: tag each turn with a supervisor-generated turn id the
   agent echoes (requires an agent change — a `turn_id` field in the framing
   or a `{"turn": ...}` envelope). Option B: serialize senders entirely under
   the flock, matching responses by order. B is simpler and likely sufficient;
   A is needed only if concurrent senders matter.
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
  `agent` binary with no agent-side changes (or with the turn-id change if
  option A is chosen — decided, not drifted into).
- A SIGTERM'd or crashed session resumes from its latest checkpoint with
  history intact, including the dangling-tool-call repair path.
- `evals/agentd-persistent.sh` (or its Rust replacement) passes against the
  Rust supervisor.
- Generated systemd units survive logout and restart on failure.
