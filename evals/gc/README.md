# GC eval fixtures

Trace fixtures consumed by `cargo test -p agent-core --test gc_evals -- --nocapture`
(the strategy x timing x cache x pressure comparison matrix from docs/GC.md).

## What lives here

- `*.jsonl` — real recorded agent traces. The harness extracts every
  `InferCall` whose full prompt contains a tool chain (>= 3 tool calls and
  >= 3 tool results) and runs each strategy against it at several budget
  pressures.
- Synthetic shapes (chat-heavy, open-tail tool chain, mixed session, long
  tool-heavy session) are generated in code inside `gc_evals.rs`, labeled
  `synthetic:` in the table. They stand in for shapes this directory does
  not cover yet — replacing them with real recordings is always preferred.

## Recording a new fixture

Full prompts are preview-only in traces by default (O(n^2) growth), so
recording requires the explicit flag:

```sh
agent --trace-full-payloads --gc none --debug "your long task here"
# or for a session:
agent --trace-full-payloads --gc none --session < turns.nul
```

Then copy the trace JSONL (path printed at startup, `trace: ...`) into this
directory with a descriptive name. Record with `--gc none` so the fixture
captures the *ungc'd* window — the harness applies strategies itself.

Shapes worth recording (gaps in the current set):

- a long coding session (many shell frames, interleaved narration) —
  `synthetic:tool-heavy-long` stands in until a real one is recorded
- a chat-heavy session with little tool use
- a hydration-heavy session (large Get/temporal context blocks)
- a session with an abandoned tangent (several turns down a wrong approach,
  then back on track) — `synthetic:tangent-abandoned` stands in; it is the
  fixture class the `semantic` strategy exists for (t-1350)

## Semantic strategy cells (offline embeddings)

`semantic` cells score against `GcState.embeddings`, which the runtime
fills via an async embedding pre-pass. The harness mirrors that pre-pass
with a deterministic mock embedder (bag-of-tokens vectors, 64 FNV-hashed
buckets — cosine similarity is vocabulary overlap), the same
recorded-replay stance as the judge column: offline runs never touch a
provider and produce identical collections every run. The promotion gates
(`gc_semantic_drops_the_tangent_and_keeps_the_relevant_thread`,
`gc_semantic_no_regression_vs_stack_on_replay_completion`) assert semantic
drops more of the abandoned tangent than stack while retaining at least as
much of the relevant thread, and never loses the last user message where
stack keeps it.

## Hygiene

- Fixtures must not contain credentials. Record with `--eval-env inherit`
  (the default, which strips `*_API_KEY`/`ANTHROPIC_AUTH_TOKEN` from tool
  children) and skim the JSONL for secrets before committing — prompts can
  quote whatever the shell printed.
- Keep fixtures small enough to review (< ~1 MB). Truncate sessions rather
  than committing megabyte transcripts.

## Reading the matrix

One row per (case, pressure, timing, strategy, cache policy): tokens
before/after, reduction %, messages and tool results retained, frames popped
(stack), stable cache prefix length, collection count, prefix-invalidation
count, convergence, and warnings when the last user message or the window
tail did not survive. Convergence is asserted for ring and stack (they carry
the front-drop degrade path) on timings that collect the final window;
mark-sweep and `every:N` are best-effort and only reported. The promotion
gate (`gc_challengers_improve_over_ring_on_tool_chains`) requires
challengers to retain more structure than ring on tool-chain windows.

The timing axis mirrors `--gc-timing`: `final` is one collection on the
full recorded window (what the first catch-overflow cycle sees); the
incremental timings (`threshold`, `eager`, `every:4`) replay the session
growing message-by-message and fire at infer points, threading one
`GcState` across collections like the runtime loop does.

## Semantic-coherence judge (optional column, online-gated)

The `judge` column scores whether the collected window preserves what is
needed to continue the session coherently (rubric: task goal retained, open
threads retained, no orphaned references; displayed as `N/3`). The offline
matrix never calls a provider:

- **Replay (default):** judge responses are looked up in
  `judge/recorded.jsonl` by a content hash of the deterministic judge
  prompt. Cells without a recording print `-`.
- **Record (online):** `RUN_AGENT_ONLINE_EVAL=1` (the evals/ convention)
  scores unrecorded cells against a real model and appends recordings, so
  subsequent offline reruns are comparable. Configure with
  `AGENT_JUDGE_MODEL` (or `AGENT_ONLINE_MODEL`, default `openrouter/auto`),
  `AGENT_JUDGE_URL` (default OpenRouter), and
  `AGENT_API_KEY`/`ANTHROPIC_API_KEY`/`OPENROUTER_API_KEY`:

  ```sh
  RUN_AGENT_ONLINE_EVAL=1 cargo test -p agent-core --test gc_evals \
    gc_strategy_matrix -- --nocapture
  ```

Recording keys hash the judge prompt text only (roles + rendered content,
never message UUIDs), so keys are stable across runs and change exactly when
a strategy's output or the prompt format changes — a changed cell simply
misses the recording rather than replaying a stale verdict.

`judge/recorded.jsonl` entries marked `"model": "hand-written"` are NOT real
model judgments: they were written by hand in a credential-free environment
to exercise the replay path (and are pinned by
`gc_judge_shipped_fixture_replays_into_matrix_cells`). Re-record online
before trusting their scores.

## Behavioral eval (t-1349, online)

Everything above measures RETENTION arithmetic on frozen windows. The
behavioral harness (`cargo test -p agent-core --test gc_behavior_evals --
--nocapture`, crates/agent-core/tests/gc_behavior_evals.rs) asks what GC
does to a REAL agent mid-task: completion, coherence, recovery. Arms = GC
strategy (`none` control, `ring`, `mark-sweep`, `stack`, `semantic` with
cited-keep on), everything else at the runtime defaults (`--gc-cache
preserve`, `--gc-timing threshold`, threshold 0.85), at context budgets
small enough (1600-2000 tokens) that collections fire mid-session — that
firing is asserted per cell from the gc_collect events.

**Fixtures** (working-dir + scripted task, identical across arms):

| fixture | shape | budget |
|---|---|---|
| `early-needle` | 9 steps; an early tool result (an access code) is needed again in the final answer, with bulky ballast before and after | 2000 |
| `tangent-return` | log analysis, then a deliberate bulky tangent (two poems, distinct vocabulary), then return to the log thread | 1600 |
| `memory-discipline` | read a token, `remember` it, bulk work, `recall` it late — memory tools across GC pressure | 1600 |

**Scoring, all from traces:** programmatic needles on the final answer
(numeric needles token-boundary-matched; the tangent fixture also checks
category ORDER), re-fetches (`refx`: probe-matching shell commands beyond
the task's own allowance), repeated commands (`rpt`), remember/recall use
(Store/Retrieve calls), gc_collect count + reason markers + dropped counts
+ recall-overlap write-barrier fields, and the AgentDone RunUsage rollup.
The `judge` column is a REAL recorded LLM verdict (three booleans:
stayed_on_task, no_redundant_work, grounded_final_answer;
`judge/behavioral.jsonl`, replayed offline by content-hash key — unlike
`judge/recorded.jsonl` there are no hand-written entries).

**Record/replay:** the t-1354 stance. Online (`RUN_AGENT_ONLINE_EVAL=1`
plus a key) records each (fixture, arm) cell's full event trace to
`recordings/`; offline replays them through effect-id replay and asserts
each cell reproduces the recording — final answer, metrics, and the
gc_collect stream (GC re-runs deterministically under replay; semantic
cells use a deterministic bag-of-tokens embedder both sides, the same
stance as the offline matrix, because OpenRouter has no embeddings
endpoint and replay requires identical vectors). No hand-written
behavioral recordings exist; offline-without-recordings is a no-op.

### Results

Recorded 2026-07-08, `anthropic/claude-haiku-4.5` ($1/$5 per Mtok) via
OpenRouter, provider-default temperature, one sample per cell, judge =
same model. Matrix spend $0.44 + judge $0.04 (+ ~$0.15 on one discarded
first pass whose fixture wording was ambiguous). Offline replay reproduces
this table (asserted per cell).

```
fixture            arm        turns evals  rpt refx rem rec coll reasons     drop ovl   in_tok  out_tok       cost wall_s  ok judge
early-needle       none          10     9    0    0   0   0    0 -              0   0    22949      876  $0.027329   14.7 yes   3/3
early-needle       ring          10     9    0    1   0   0    6 s:6           42   0    19917      939  $0.024612   16.8 yes   0/3
early-needle       mark-sweep    10     9    0    0   0   0    6 s:6           42   0    19958      955  $0.024733   14.9  NO   2/3
early-needle       stack         10     9    0    0   0   0    5 s:5           35   0    20207      761  $0.024012   14.2  NO   2/3
early-needle       semantic      12    11    1    0   0   0    7 s:7           56   0    24922     1003  $0.029937   16.9  NO   0/3
tangent-return     none           6     5    0    0   0   0    0 -              0   0    17583      449  $0.019828   10.3 yes   3/3
tangent-return     ring          27    26   24   11   0   0   25 s:25         676   0    37566     2059  $0.047861   44.1  NO   0/3
tangent-return     mark-sweep     6     5    0    0   0   0    4 s:4            6   0    10782      440  $0.012982   11.5 yes   2/3
tangent-return     stack         27    26   24   11   0   0   25 s:25         624   0    41214     2044  $0.051434   47.1  NO   0/3
tangent-return     semantic      27    26   24    4   0   0   25 s:25         660   0    41258     2335  $0.052933   49.7  NO   0/3
memory-discipline  none           9     6    0    0   1   1    0 -              0   0    19857      822  $0.023967   14.8 yes   3/3
memory-discipline  ring           9     6    0    0   1   1    6 s:6           12   0    16054      812  $0.020114   16.8 yes   3/3
memory-discipline  mark-sweep     9     6    0    0   1   1    6 s:6            0   0    16694      822  $0.020804   17.1 yes   3/3
memory-discipline  stack          9     6    0    0   1   1    6 s:6           25   0    16573      750  $0.020323   16.3 yes   3/3
memory-discipline  semantic      16    11    5    0   3   1   13 s:13         190   0    28589     1414  $0.035659   30.9  NO   0/3
```

(`rpt` = commands identical to one already run, `refx` = needle re-fetches
beyond the task's allowance, `coll`/`reasons` = gc_collect events —
`s` = scheduled — `drop` = messages evicted, `ovl` = recall-overlap
write-barrier events, cost = AgentDone rollup. All GC cells collected,
asserted; 25-collection cells were clipped by the 26-turn cap.)

### Findings

Caveats first: one model, one sample per cell, provider-default
temperature, and budgets (1.6-2k tokens) far below real deployments — this
measures GC's failure modes under extreme pressure, not its steady-state
overhead. The control (`none`) succeeded everywhere at these window sizes.

1. **The offline ranking did NOT survive contact with behavior.** The
   offline matrix promotes `stack` (best retention structure) and
   `semantic` for tangents; `mark-sweep` is the offline weakling (only
   best-effort convergence). Behaviorally it inverted: **mark-sweep was
   the only GC arm to complete the tangent fixture** — and did it CHEAPER
   than the no-GC control ($0.0130 vs $0.0198, 6 drops in 4 collections) —
   while ring, stack, and semantic all fell into a **GC-induced restart
   loop** (25 collections, ~24 repeated commands, 2.4-2.7x control cost,
   clipped at the turn cap with no final answer). Mark-sweep's offline
   deficiency — it refuses to evict incomplete lifecycles and reclaims
   little — is precisely what preserved the model's own progress narration
   and kept it coherent. A strategy that wins retention arithmetic and
   loses behavior, and vice versa, is exactly what this eval existed to
   detect.
2. **The thrash cycle, mechanically** (from the traces): a scheduled
   collection evicts the assistant's own step-completion narration; the
   model re-reads the pinned task statement, concludes it is at step 1,
   and re-runs it; the re-run regrows the window past threshold, which
   schedules the next collection. Stable until the turn cap. Ring, stack,
   and semantic all sustained it on `tangent-return`; semantic also fell
   into a milder version on `memory-discipline` (13 collections, three
   duplicate `remember` calls — two rejected as duplicate slugs — and a
   double-counted WARN total).
3. **Dropped-needle recovery: models fabricate rather than re-fetch.** On
   `early-needle`, ring was the only strategy whose agent re-ran `cat
   config/access-code.txt` (refx=1) and answered correctly. Stack,
   mark-sweep, and semantic all emitted **confidently hallucinated access
   codes** (`CDBH92`, `ALPHA-7-TANGO-4`, `batch2024`) with no recovery
   attempt — consistent with stack's `[frame: ...]` annotations telling
   the model the step already happened while withholding its result,
   inviting confabulation, whereas ring's wholesale amputation left
   nothing to confabulate from. Nobody reached for `recall` unprompted
   (rec=0 outside the memory fixture). Retention tables cannot see this
   failure mode at all: the fabricated answer LOOKS like task completion.
4. **Memory discipline is the honest recovery path — when scripted.** On
   `memory-discipline`, every strategy's agent remembered early and
   recalled late, and ring/mark-sweep/stack all recovered the evicted
   token exactly (matching the control at slightly LOWER cost — GC paying
   for itself). The recall column of the semantic cell shows the token
   recovered there too; that cell failed on the WARN sum, not the token.
5. **The recall-overlap write-barrier (t-1351) never fired in vivo:**
   `ovl` = 0 in all 15 cells, including the memory cells where recall
   demonstrably re-injected previously-collected content. Cause: the
   barrier's exact-hash match compares the recall hit (the memory RENDER,
   `### deploy-token ...\nTOKEN-...`) against window/collected message
   content (a shell-result JSON envelope) — these never hash-equal for
   real memories derived from tool output. The t-1167 generational input
   needs chunk- or substring-level matching before it can see real recall
   traffic.
6. **Reason markers: 100% `scheduled`.** The threshold timing collected
   proactively in every cell; the t-1343 overflow backstop and
   catch-overflow paths never engaged in any real session (no cell ever
   reached a provider overflow).
7. **Judge vs needles disagree exactly where they should.** Ring's
   `early-needle` cell passes the needles (correct code via re-fetch) but
   scores 0/3 — the judge dinged the redundant re-acquisition; stack's and
   mark-sweep's hallucination cells score 2/3 with `grounded_final_answer:
   false`. One judge response (tangent/ring) initially echoed the
   transcript instead of JSON and was re-recorded; verdicts on long thrash
   transcripts are the least reliable rows of the column.

### Mechanism gaps surfaced (candidate tasks)

- **Bound effect errors do not replay byte-identically** (t-1222 x
  replay): the replayed bound VALUE carries the "AgentIR replaying
  recorded Store failure ..." wrapper, so window content differs by a few
  tokens and content-sensitive GC can collect marginally differently
  (observed: 190 vs 188 drops on the semantic memory cell). The harness
  compares gc-derived fields leniently for cells containing effect errors;
  replaying the recorded error string verbatim would close the gap.
- **Progress-narration preservation.** The thrash loop would break if
  strategies treated the assistant's own recent step-completion messages
  as protected the way system/last-user messages are — or if collection
  left a one-line "steps 1-N done" digest (the stack-smart direction).
- **Write-barrier matching is too strict for real recall** (finding 5).
- **A "collected" marker for the model.** Stack's frame annotations invite
  confabulation (finding 3); an annotation that says the RESULT was
  evicted and must be re-fetched (rather than summarizing that the call
  happened) might flip fabrication into recovery. Cheap A/B on this
  harness.
