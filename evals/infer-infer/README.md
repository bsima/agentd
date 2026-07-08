# Infer-calls-infer eval (t-1342)

Measures whether the agent loop's `infer` tool — the model making a nested
Infer call to delegate a subtask to another model — earns its cost. Harness:
`cargo test -p agent-core --test infer_infer_evals -- --nocapture`.

Scope note: the arms here are SCRIPTED — they pin the structural cost
mechanics, not model behavior. Whether a real model *chooses* to delegate,
shows restraint, and can drive a process subagent is the companion
delegation-behavior eval (t-1354): [../delegation/](../delegation/README.md).

## What it measures

Each fixture task runs in two arms against the same task:

- **single** — the parent model does everything itself;
- **sub-infer** — the parent delegates subtasks to a cheaper model via the
  `infer` tool.

Scores are read from the trace, never estimated: per-`InferResult` token
usage and `cost_micro_usd` (t-1334), the `RunUsage` rollup stamped on
`AgentDone`, and effect counts (parent-loop infers = turns, sub-infers,
Evals, InferErrors). Parent vs sub attribution uses `parent_op_id`
(t-1347): sub-infer events carry the dispatching turn Infer's op_id,
parent-loop events carry none.

## Fixtures

| fixture | shape | expected winner |
|---|---|---|
| `simple-question` | one-shot answer, delegation is pure indirection | single |
| `doc-synthesis-by-copy` | 3 long docs pasted into infer prompts (pre-t-1344 path, kept) | single, big |
| `doc-synthesis` | 3 docs fetched via shell, delegated **by reference** (`context_refs`) | single, within 1.3x |
| `noisy-middle-step` | noisy shell dump digested by a child **by reference** | **sub-infer** |
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
(cost attributed by `parent_op_id` lineage), and `ok` (final answer contains the
fixture's required needles). The `->` verdict line names the cost winner,
the ratio, and the structural reason.

## Findings (offline matrix, mechanism probes)

The headline: **the sub-infer mechanism pays wherever the parent's own
token flow is what delegation removes** — generation-heavy work
(`generation-offload`, ~2.7x via output-rate arbitrage) and, since
t-1344, reading/digesting work passed **by reference**
(`noisy-middle-step`, ~1.6x). By-copy delegation of material remains a
structural tax (`doc-synthesis-by-copy`, 7.7x — kept as the
comparison).

1. **Fixed (t-1344).** The child used to have no context of its own —
   one bare user message built from the tool-call arguments — so
   delegating material meant *copying it out through parent output
   tokens* (5x the input rate), with the copy then riding parent history
   every later turn (`prepare_tools` retains the full `tool_calls`).
   Now the `infer` tool takes `context_refs`: ids of prior tool calls
   (the ids the model itself minted — `tool_calls[].id` /
   `tool_call_id`, already model-visible), resolved against history at
   dispatch (`ir_agent.rs` `infer_resolve`/`infer_eval`,
   `Expr::SelectToolResults`) and assembled into the child's messages
   server-side. The child gets a proper message structure: an optional
   system slot (`AgentLoopOptions.infer_system_prompt`, owned by the
   dispatch site), referenced material as user messages, instruction
   last. Refs resolve within the same assistant turn too (results
   append as the tool loop walks the batch), so fetch + delegate can be
   one turn. An unresolved ref binds as a readable tool result naming
   the missing ids (t-1222). `doc-synthesis` by reference: single wins
   by 1.2x (delegation as rounding error — the cheap child reads),
   down from 7.7x; the retained arguments are refs + prompt, so parent
   history carries the material exactly once. The by-copy path is
   unchanged and still costs: `doc-synthesis-by-copy` pins 7.7x.
2. **Fixed (t-1344).** Containment is real for referenced material —
   the dump stays a tool result (input rate, both arms, exactly once),
   the child digests it at cheap rates, and the parent trades a verbose
   inline digest (output rate + history residue) for a one-line status.
   `noisy-middle-step` by reference: sub-infer **wins 1.6x** (was 3.3x
   more expensive by copy). What remains structural: material already
   in parent history is paid at parent input rates per turn in *both*
   arms — references remove the copy tax, not the carry tax (that is
   GC's territory).
3. **The `infer` schema gives no delegation guidance** — `model` is a bare
   string (no catalog/enum/pricing), and there is no budget knob
   (`ir_interpreter.rs` `base_ir_tool_specs`; `InferPolicy` carries no
   budget knob). Hallucinated ids fail closed and cost a full round-trip
   (`child-error-recovery`: 14x the single arm).
4. **Fixed (t-1346).** The child used to be offered the parent's full
   toolset it could never use, and a tool-calling child response fell back
   to serializing the whole Response envelope (usage fields, unexecuted
   tool calls) into the parent context — the exact leak t-1120 removed for
   the text path. The sub-infer site now declares an empty toolset
   (`InferPolicy.tools`, ir_agent.rs `infer_eval`), so the child is a bare
   single completion whose text feeds back verbatim; the `infer_eval`
   fallback remains only as the readable surface for bound child errors
   (t-1222) and degenerate empty completions.
5. **Fixed (t-1347).** Trace attribution gaps: sub-infer events used to
   carry `parent_op_id: None`, and a failed sub-infer (InferError)
   contributed nothing to the `AgentDone` usage rollup, so attempts were
   undercounted. Sub-infer InferCall/InferResult/InferError events now
   carry the dispatching parent Infer's op_id, and the rollup counts
   failed attempts in `failed_infer_calls` (attempts only — the provider
   error path returns no Response, so their token usage is structurally
   unavailable).

Each finding is pinned by a `probe_*` test in the harness; fixing the
mechanism should flip the corresponding probe.
