# Durable sessions

The CLI accepts `--session-id`, `--parent-session`, `--fork-from`,
`--state-dir`, and `--signal-deadline`. Callers own the IDs and
send SIGTERM; agentd neither knows nor manages machine volumes or snapshots.

`--state-dir /work/state` routes checkpoints, traces, approvals, and ACP
session files under that root. `--memory-dir` opts into memory tools and
must be chosen deliberately (it changes the AgentIR program hash). Credentials
stay outside the state root. With `--state-dir`, checkpoint and memory
overrides must remain inside the root.

With `--signal-deadline`, the CLI handles SIGTERM cooperatively. In-flight
shell Evals use a separate process group; an interrupted effect binds a typed
`status: "interrupted"` result, whose `EvalResult` is recorded before a
mid-turn machine checkpoint is committed. The checkpoint at
`checkpoints/session-latest.json` is authoritative for resume. A successful
suspend exits 0; a failed checkpoint or elapsed deadline exits nonzero.
Resumption of a mid-turn checkpoint requires `--session` or `--fifo` and
reuses the original trace path and run ID. No strict replay ever re-executes
Eval. `--replay-live-eval` (only with `--replay-trace`) is an explicit offline
**test** mode: recorded Infer results and live Eval, not deterministic audit
replay. `--replay-live-approval` additionally exercises an unrecorded approval
gate while still using recorded Infer responses. These modes should run in
a disposable workspace.

A fork may start from a completed-turn or mid-turn checkpoint under a new
session/run identity. A pending approval is copied as a separate branch
record; neither branch executes its gated effect until its own decision is
recorded. `agent --state-dir <root> approvals --deny/--approve <id>` records
the decision; then resume from `<root>/checkpoints/session-latest.json`.

Current restrictions (fail closed): mid-Par suspension and signal handling
in ACP mode are not supported. A memory directory used with a state root
must resolve within that root. The offline durable-session fixtures exercise
suspend/resume (including a completed continuation), mid-turn and turn-boundary
forks, and pending-approval fork. Unit checks cover a missing trace and an
expired checkpoint deadline without replacing the authoritative checkpoint.
A failure after an Eval side effect but before its interrupted checkpoint
commits still requires retaining the machine for recovery; do not snapshot
or discard it on nonzero exit.
