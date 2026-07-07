# Infer-calls-infer eval (t-1342)

Measures whether the agent loop's `infer` tool — the model making a nested
Infer call to delegate a subtask to another model — earns its cost. Harness:
`cargo test -p agent-core --test infer_infer_evals -- --nocapture`.

## What it measures

Each fixture task runs in two arms against the same task:

- **single** — the parent model does everything itself;
- **sub-infer** — the parent delegates subtasks to a cheaper model via the
  `infer` tool.

Scores are read from the trace, never estimated: per-`InferResult` token
usage and `cost_micro_usd` (t-1334), the `RunUsage` rollup stamped on
`AgentDone`, and effect counts (parent-loop infers = turns, sub-infers,
Evals, InferErrors). Parent vs sub attribution uses the effect location's
site block, because the trace carries no parent/child linkage (see
findings).

## Fixtures

| fixture | shape | expected winner |
|---|---|---|
| `simple-question` | one-shot answer, delegation is pure indirection | single |
| `doc-synthesis` | summarize 3 long docs via cheap children, parent synthesizes | single |
| `noisy-middle-step` | error-prone middle step produces a large dump; child digests it | single |
| `generation-offload` | long boilerplate generation delegated, short verdict kept | **sub-infer** |
| `child-error-recovery` | delegate to a hallucinated model id, bind the error, retry | single |

The expected winner is asserted, so the matrix pins the structural
economics of the mechanism rather than just printing them. Offline numbers
model usage from the *actual prompts* each arm sends (`estimate_tokens`,
the runtime's chars/3 budget estimator; output = content + serialized
tool-call arguments) under a fixture pricing table (parent $3/$15 per Mtok,
child $0.15/$0.60), so the drivers are context growth — tool-call argument
duplication and per-turn history re-send — not invented constants.

## Running

- **Offline (default, deterministic, credential-free):**
  `cargo test -p agent-core --test infer_infer_evals -- --nocapture`.
  Scripted providers on both arms; two runs per arm assert determinism.
- **Online (record):** `RUN_AGENT_ONLINE_EVAL=1` runs the same fixtures
  against a real provider and records every exchange to
  `recorded.jsonl` in this directory, keyed by a content hash of
  (model + prompt structural content — never message UUIDs). Configure with
  `AGENT_EVAL_PARENT_MODEL` (or `AGENT_ONLINE_MODEL`, default
  `anthropic/claude-sonnet-4.5`), `AGENT_EVAL_CHILD_MODEL` (default
  `anthropic/claude-haiku-4.5`), `AGENT_EVAL_URL` (default OpenRouter),
  and `AGENT_API_KEY`/`ANTHROPIC_API_KEY`/`OPENROUTER_API_KEY`:

  ```sh
  RUN_AGENT_ONLINE_EVAL=1 cargo test -p agent-core --test infer_infer_evals \
    infer_infer_recorded_matrix -- --nocapture
  ```

- **Replay:** with `recorded.jsonl` present, `infer_infer_recorded_matrix`
  replays the recorded runs offline (misses print as skipped cells, never
  a provider call). A `{"meta": ...}` first line pins the model ids used at
  record time so replays rebuild identical prompts. With no recordings the
  test is an offline no-op. Recordings must be credential-free — skim
  before committing.

## Reading the table

One row per (fixture, arm): `turns` (parent-loop infer calls), `sub`
(nested infer calls), `evals`, `errs` (InferErrors), token totals from the
`AgentDone` rollup, `cost` (total micro-USD as dollars), `parent$`/`sub$`
(cost attributed by effect site), and `ok` (final answer contains the
fixture's required needles). The `->` verdict line names the cost winner,
the ratio, and the structural reason.

## Findings (offline matrix, mechanism probes)

The headline: **the sub-infer mechanism only pays for generation-heavy,
short-prompt delegation** (`generation-offload`, ~2.7x cheaper via
output-rate arbitrage). Everywhere else it loses — not because of model
quality, but because of mechanism structure:

1. **The child has no context of its own** — its prompt is exactly one
   bare user message built from the tool-call arguments
   (`ir_agent.rs` `infer_eval`), so delegating over material means
   *copying it out through parent output tokens* (5x the input rate) and
   that copy then rides in parent history every later turn
   (`ir_agent.rs` `prepare_tools` pushes the full `tool_calls` into
   history). `doc-synthesis`: sub-infer 7.7x more expensive.
2. **Containment is illusory** — everything the child sees passes through
   the parent's history via arguments, and the child (a single Infer, no
   tool dispatch) cannot fetch anything itself. `noisy-middle-step`:
   sub-infer 3.3x more expensive.
3. **The `infer` schema gives no delegation guidance** — `model` is a bare
   string (no catalog/enum/pricing), and there is no budget knob
   (`ir_interpreter.rs` `base_ir_tool_specs`; `InferPolicy` has only
   `on_error`). Hallucinated ids fail closed and cost a full round-trip
   (`child-error-recovery`: 14x the single arm).
4. **The child is offered the parent's full toolset it can never use**,
   and a tool-calling child response falls back to serializing the whole
   Response envelope (usage fields, unexecuted tool calls) into the parent
   context — the exact leak t-1120 removed for the text path.
5. **Trace attribution gaps** — sub-infer events carry `parent_op_id:
   None`, and a failed sub-infer (InferError) contributes nothing to the
   `AgentDone` usage rollup, so attempts are undercounted.

Each finding is pinned by a `probe_*` test in the harness; fixing the
mechanism should flip the corresponding probe.
