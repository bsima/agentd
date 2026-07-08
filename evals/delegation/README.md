# Delegation-behavior eval (t-1354)

t-1342/t-1344 ([../infer-infer/](../infer-infer/README.md)) measured the COST
mechanics of the `infer` tool with **scripted** arms — the model never chose
anything. This eval asks the behavioral question those scripted arms cannot:
given the tool, does a **real model** use it effectively? Harness:
`cargo test -p agent --test delegation_evals -- --nocapture`
(crates/agent/tests/delegation_evals.rs — it lives in the `agent` crate
because arm D needs the built binary).

The questions (the skepticism this eval exists to answer):

1. Does a real model use the tool at all?
2. Well, or only when instructed (guided vs unprompted delta)?
3. Does it show restraint on tasks where delegation is pure overhead?
4. Is "exploration via Infer" redundant given internal reasoning?
5. Can a subagent-as-**process** do real work via Eval?

## Design

**Arms** — same task text, same parent model; only the system prompt and the
advertised toolset differ (the model, never the task, chooses delegation —
pinned by `fixture_tasks_never_mention_delegation`):

| arm | infer tool | guidance |
|---|---|---|
| `baseline` | hidden (provider-side filter) | none |
| `tool-unprompted` | advertised | none |
| `tool-guided` | advertised | the SHIPPED runtime-guidance fragment (t-1359, [docs/GUIDANCE.md](../../docs/GUIDANCE.md) §4): the runtime's own delegation + cost blocks, delivered as its Developer prompt section, plus the interim delegate catalog naming the child model id and rates. Originally a handwritten paragraph with the same content; since t-1359 the arm measures exactly what ships |
| `subagent-process` | hidden | delegate via the shell tool: `agent --model eval-child '<prompt>'` — a FULL child agent (own loop, own shell tool) |

**Fixtures** (task classes):

| fixture | class | stance |
|---|---|---|
| `doc-synthesis` | synthesize across 3 long shell-fetched docs (the t-1342 shape, model chooses) | delegation plausibly helps |
| `generation-offload` | 20-candidate written evaluation (the one shape where t-1342 measured infer WINNING, ~2.7x) | delegation should help |
| `dead-end-debugging` | tempting wrong lead (a comment blaming lib.sh arithmetic; the real bug is a config value) | exploration — the open question |
| `restraint-direct` | "what is 17 * 23?" | any delegation is overhead |
| `count-in-files` | count ERROR lines across files — tool work an infer child cannot do (t-1346: no tools) | the arm-D capability contrast |

**Scoring, all from traces** (never estimated): `RunUsage` on `AgentDone`
(t-1334), per-`InferResult` cost, sub-infer attribution by `parent_op_id`
lineage (t-1347), process delegations = `EvalCall` commands invoking
`agent`, programmatic success needles per fixture (numeric needles are
token-boundary-matched, text needles case-insensitive; no judge in v1).
Appropriateness flags: delegating on the restraint fixture prints
`OVER-DELEGATED`; not delegating on a should-help fixture prints
`missed-op` (scored, never asserted — behavior is the data this eval
collects).

## Running / recording

- **Offline (default, credential-free):** replays each cell's recorded event
  trace through the interpreter's effect-id replay (`IrReplayTrace`) — Infer
  responses AND Eval results (shell output, child-agent stdout) come from
  the recording; the replay must reproduce the recorded final answer and
  metrics, and the table prints from the replayed run. With no recordings
  the matrix is a documented no-op: there are deliberately **no hand-written
  behavioral recordings** (faking model choices would corrupt the eval);
  the always-on tests are plumbing-only, scripted providers marked as such.
- **Online (record):**

  ```sh
  RUN_AGENT_ONLINE_EVAL=1 OPENROUTER_API_KEY=... \
    cargo test -p agent --test delegation_evals -- --nocapture --test-threads=1
  ```

  records any missing cells to `recordings/` (delete a cell file to
  re-record it). Defaults: parent `anthropic/claude-haiku-4.5`, child
  `openai/gpt-4o-mini` (override with `AGENT_EVAL_PARENT_MODEL` /
  `AGENT_EVAL_CHILD_MODEL`; endpoint `AGENT_EVAL_URL`, default OpenRouter).
  Recordings are checked at write time and by
  `recordings_are_credential_free` to never embed the key. Temperature is
  not configurable through the provider layer today, so runs use provider
  defaults — a re-record can legitimately differ (single-sample cells; see
  gaps below).

**Arm-D child environment** (what the harness provides, and what any real
deployment of subagent-as-process must provide): an allowlist env with PATH
prefixed by a dir holding the built `agent` binary, HOME pointed at a
per-cell scratch dir (children write their traces there; the harness sweeps
them for the child-cost column), XDG_CONFIG_HOME pointed at
[config/](config/agent/models.yaml) — the eval's own registry defining
`eval-child` with a pricing block — and OPENROUTER_API_KEY. Two contracts
learned the hard way: the runtime's default Eval env policy deliberately
strips `*_API_KEY` from shell children, so a process child that must dial a
provider is an explicit opt-in; and a one-shot `agent` with non-terminal
stdin reads it to EOF as optional input data, so any spawner must detach
stdin (the runtime's Eval effect already does).

## Results

**Current recordings (t-1359 re-record, shipped guidance).** Re-recorded
2026-07-08 after the runtime-guidance delivery shipped: all arms see the
fattened tool descriptions (t-1359 step 1), and the guided arm's system
prompt is no longer handwritten — the runtime delivers its shipped
fragment (delegation + cost blocks + delegate catalog) as its own
Developer section, whose content hash rides every InferCall's `prompt_ir`
trace event in these recordings. Same models, provider, and
one-sample-per-cell caveats as the first recording. Whole matrix:
**~$0.15**. Offline replay reproduces this table exactly (asserted per
cell).

```
fixture              arm               turns  sub  proc evals errs   in_tok  out_tok       cost  wall_s  ok  flag
doc-synthesis        baseline              2    0     0     3    0     6287      482  $0.008697     4.6 yes  -
doc-synthesis        tool-unprompted       2    0     0     3    0     6913      463  $0.009228     5.2 yes  missed-op
doc-synthesis        tool-guided           2    0     0     3    0     7688      464  $0.010008     4.6 yes  missed-op
doc-synthesis        subagent-process      2    0     0     3    0     6542      485  $0.008967     4.7 yes  missed-op
dead-end-debugging   baseline              3    0     0     2    0     2938      295  $0.004413     4.9 yes  -
dead-end-debugging   tool-unprompted       6    0     0     5    0     8965      455  $0.011240     9.5 yes  -
dead-end-debugging   tool-guided           6    0     0     5    0    11310      408  $0.013350     7.5 yes  -
dead-end-debugging   subagent-process      3    0     0     2    0     3162      238  $0.004352     4.1 yes  -
generation-offload   baseline              2    0     0     1    0     4282     2350  $0.016032    22.2 yes  -
generation-offload   tool-unprompted       2    0     0     1    0     4672     2205  $0.015697    20.0 yes  missed-op
generation-offload   tool-guided           2    1     0     0    0     4409     2214  $0.011157    27.0 yes  -
generation-offload   subagent-process      2    0     0     1    0     4526     2451  $0.016781    22.0 yes  missed-op
restraint-direct     baseline              2    0     0     1    0     1381       69  $0.001726     2.3 yes  -
restraint-direct     tool-unprompted       1    0     0     0    0      945        5  $0.000970     0.9 yes  -
restraint-direct     tool-guided           1    0     0     0    0     1332        5  $0.001357     0.9 yes  -
restraint-direct     subagent-process      1    0     0     0    0      757        5  $0.000782     1.1 yes  -
count-in-files       baseline              2    0     0     1    0     1451       93  $0.001916     2.3 yes  -
count-in-files       tool-unprompted       2    0     0     1    0     2088      100  $0.002588     2.9 yes  -
count-in-files       tool-guided           2    0     0     1    0     2874      112  $0.003434     2.3 yes  -
count-in-files       subagent-process      2    0     0     1    0     1705       93  $0.002170     3.5 yes  -
```

**Shipped-text delta, against the promotion gate (docs/GUIDANCE.md §5):**
same behavioral profile as the handwritten arm. (a) The guided arm
delegated exactly where the economics pay — `generation-offload`, one
sub-infer with the correct child id and a clean self-contained child
prompt — and came out **1.41x cheaper than unprompted** and 1.44x cheaper
than baseline (first recording: 2.26x/1.5x; single-sample,
provider-default temperature, so deltas move between recordings) while
remaining the slowest cell (27.0s: the serialized child round-trip,
unchanged). (b) Zero `OVER-DELEGATED` flags anywhere; the guided arm
answered `restraint-direct` in one turn with no tool use. (c) Offline
replay reproduces the table. New observation for the t-1345 catalog case:
`tool-unprompted` now sees the fattened `infer` description (descriptions
ship to every arm) and STILL made zero delegations — description-level
guidance alone did not flip unprompted delegation, consistent with the
missing model catalog being the root cause. It did, however, answer the
restraint fixture in one turn with no tools, which the old unprompted arm
did not — weak evidence the description text alone teaches some economy.

**First recording (handwritten guided arm; superseded).** Recorded
2026-07-08 before t-1359 shipped; kept for the findings below, but the
recordings themselves were replaced by the re-record (offline replay runs
against the current table above). Parent `anthropic/claude-haiku-4.5`
($1/$5 per Mtok), child `openai/gpt-4o-mini` ($0.15/$0.60), via
OpenRouter, provider-default temperature, one sample per cell. Whole
matrix + probe: **$0.14**.

```
fixture              arm               turns  sub  proc evals errs   in_tok  out_tok       cost  wall_s  ok  flag
doc-synthesis        baseline              2    0     0     3    0     6243      463  $0.008558     5.1 yes  -
doc-synthesis        tool-unprompted       2    0     0     3    0     6582      489  $0.009027     4.6 yes  missed-op
doc-synthesis        tool-guided           2    0     0     3    0     6828      457  $0.009113     4.8 yes  missed-op
doc-synthesis        subagent-process      2    0     0     3    0     6498      468  $0.008838     4.6 yes  missed-op
dead-end-debugging   baseline              3    0     0     2    0     2870      255  $0.004145     4.7 yes  -
dead-end-debugging   tool-unprompted       4    0     0     3    0     4929      359  $0.006724     5.7 yes  -
dead-end-debugging   tool-guided           6    0     0     5    0     8717      450  $0.010967     8.9 yes  -
dead-end-debugging   subagent-process      4    0     0     3    0     4815      305  $0.006340     5.2 yes  -
generation-offload   baseline              2    0     0     1    0     3727     2046  $0.013957    19.0 yes  -
generation-offload   tool-unprompted       2    0     0     1    0     5307     3233  $0.021472    33.0 yes  missed-op
generation-offload   tool-guided           2    1     0     0    0     3317     2016  $0.009515    37.9 yes  -
generation-offload   subagent-process      2    0     0     1    0     5531     2768  $0.019371    17.8 yes  missed-op
restraint-direct     baseline              2    0     0     1    0     1336       64  $0.001656     2.1 yes  -
restraint-direct     tool-unprompted       2    0     0     1    0     1672       64  $0.001992     1.7 yes  -
restraint-direct     tool-guided           1    0     0     0    0      903        5  $0.000928     1.0 yes  -
restraint-direct     subagent-process      1    0     0     0    0      737        5  $0.000762     1.1 yes  -
count-in-files       baseline              2    0     0     1    0     1412       94  $0.001882     2.6 yes  -
count-in-files       tool-unprompted       2    0     0     1    0     1773      118  $0.002363     3.3 yes  -
count-in-files       tool-guided           2    0     0     1    0     2004       76  $0.002384     1.8 yes  -
count-in-files       subagent-process      2    0     0     1    0     1672      100  $0.002172     3.1 yes  -
```

(`turns` = parent provider calls, `sub` = infer-tool delegations by
`parent_op_id` lineage, `proc` = shell Evals invoking `agent`, `cost` =
AgentDone rollup. The child$ column is omitted above: it was $0 everywhere
— no arm-D cell ever spawned a child.)

**Process-child capability probe** (the arm-D invocation run directly,
decoupled from parent inclination, recorded in
`recordings/child-capability.json`): `agent --model eval-child '<count the
ERROR lines>'` → exit ok, answer "…is 7." (correct), **$0.000182, 922
tokens, 2.8s wall**, doing real shell work in its own loop.

## Findings (direct answers, from the FIRST recording's data)

Caveats first: one parent model, one sample per cell, provider-default
temperature — this is a first behavioral reading, not a distribution.
(The t-1359 re-record above reproduced the same behavioral profile with
the shipped guidance text; its deltas are noted inline there.)

1. **Used at all?** Essentially no. Across 20 cells there was exactly ONE
   delegation: `tool-guided` on `generation-offload`. Unprompted, the model
   NEVER touched the infer tool (0 sub-infers in all 5 `tool-unprompted`
   cells) — plausibly connected to the schema offering no model catalog
   (t-1342 finding 3: `model` is a bare string; the model has no id it
   could confidently pass).
2. **Guided vs unprompted delta?** Guidance is what makes the tool exist,
   and where the economics favor delegation it works well: on
   `generation-offload` (the one shape t-1342 measured as a win) guided
   delegated with a correct child id and a clean self-contained rewritten
   prompt, and came out **2.26x cheaper than unprompted** ($0.0095 vs
   $0.0215) and 1.5x cheaper than baseline — while being the slowest cell
   (37.9s vs 19.0s baseline: the child round-trip is serialized latency).
   Guidance also did NOT cause indiscriminate delegation elsewhere: on
   `doc-synthesis` and `count-in-files` the guided arm still did the work
   itself — which matches t-1342's own measurements (by-reference
   synthesis is within 1.3x of single; a one-line grep beats any
   delegation), so the printed `missed-op` flags read more like the
   stance's prior being wrong than the model being wrong.
3. **Restraint?** Perfect. Zero delegations on `restraint-direct` in every
   arm (no `OVER-DELEGATED` anywhere in the matrix). Bonus: the guidance
   arms answered 17*23 in one turn with no tool use at all, while
   baseline/unprompted used the shell as a calculator — the "do NOT
   delegate direct questions" guidance apparently generalized to "just
   answer".
4. **Exploration redundancy?** On this fixture, supported. All four arms
   found the real bug (config.env) past the tempting lib.sh red herring
   with zero delegation; internal reasoning plus direct shell reads
   sufficed, and the guided arm's extra deliberation only made it the most
   expensive cell of the fixture ($0.011, 6 turns, 8.9s). No evidence yet
   that Infer-based exploration adds anything at this task scale —
   falsifying it would need a fixture too big for the parent to hold, which
   is GC/context territory.
5. **Process-subagent viability?** The mechanism works but was never
   chosen. The direct probe shows a child `agent` process is genuinely
   capable — correct multi-step tool work for $0.000182 in 2.8s — and the
   plumbing (registry alias, allowlisted key, stdin detach, trace sweep)
   is all it needs. But in the matrix no arm-D parent ever spawned one:
   for every fixture here, doing the work directly was rationally cheaper
   than a child agent run, mirroring (2).

The meta-finding: this parent model's delegation behavior tracked the
*measured* economics of t-1342 almost exactly — it delegated in the one
place delegation pays and nowhere else, but only once told how. The
mechanism's bottleneck is discoverability, not model judgment.

This finding is the founding evidence for the runtime operations guidance
design (t-1356): what guidance should ship, where it lives, and how each
piece gets A/B'd with this eval's arms methodology —
[../../docs/GUIDANCE.md](../../docs/GUIDANCE.md).

## Candidate tasks (mechanism gaps surfaced)

- **Delegate catalog in the infer schema.** Unprompted delegation is ~0 and
  arguably CANNOT happen safely: the schema's `model` parameter carries no
  ids, rates, or budget knob (pinned by t-1342's
  `probe_child_toolset_and_infer_schema_guidance`). Advertising the
  registry's aliases + pricing in the tool spec is the single highest-
  leverage change this eval points at.
- **Child-process usage is invisible to the parent trace.** A process
  child's tokens/cost live only in its own trace file; the parent Eval
  records stdout. This harness sweeps a per-cell HOME as a workaround.
  Lineage (e.g. the parent passing `AGENT_RUN_ID`/a parent-op env var that
  the child stamps into its trace, or a machine-readable usage trailer)
  would make subagent-as-process first-class in cost rollups.
- **No temperature control in the provider layer** — "temperature 0 where
  the provider honors it" is not currently expressible, so recordings are
  single samples of a stochastic process.
- **Delegation trades money for latency** (serialized child round-trip:
  +19s on `generation-offload`); if wall time matters, the guidance needs
  to say so, or the loop needs parallel sub-infer dispatch.
