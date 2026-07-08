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
tail did not survive. Last-user survival is asserted per cell since t-1367
(the hard guard is an invariant of every strategy; the warning marker
remains as a tripwire) — before the guard, ring+ignore dropped it in 24/60
cells on this fixture set. Convergence is asserted for ring and stack (they carry
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
| `memory-discipline` | read a token, `remember` it, bulk work, `recall` it late — memory tools across GC pressure | 1700 |

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
  harness. **Done — t-1360** (eviction markers, all strategies; see
  "Eviction markers" below); the online A/B is t-1369 (**ran — see the
  marker-era section at the end: the flip is real for recovery
  attempts, partial for answers**).

## Guided vs strategy (t-1364, online)

t-1349 ended on an inversion (mark-sweep behaved best, the offline champion
thrashed, three of four strategies confabulated) and one bright spot:
memory discipline rescued everything. The follow-up hypothesis: **guidance
dominates strategy choice** — the shipped runtime-guidance fragment
(t-1359: GC-awareness §2.4 + memory-discipline §2.2 blocks, default-on
since d79fad3) changes behavior more than swapping collectors does, and
eliminates confabulation.

**Design.** The t-1349 harness gained a guidance axis: each cell is
(fixture, arm, guided, sample). Guided cells run the SHIPPED
`RuntimeGuidance::default()`; unguided cells run `disabled()`. Memory
tools are offered in every cell (unchanged from t-1349 — all three
fixtures already ran the memory-enabled loop). The t-1349 recordings could
NOT be reused as the unguided arms: commit a6592f8 (t-1359 step 1) rewrote
the shell/remember/recall/infer tool descriptions, which enter every
cell's provider offer — so all unguided arms were re-recorded on the
current descriptions, and the legacy recordings are kept and replayed as a
separate regression section, no longer a comparison arm. n=2 on the
hypothesis-deciding cells (stack and mark-sweep ± guidance on
`early-needle`/`tangent-return`), n=1 elsewhere; `none`+guided was left
unrecorded (guidance without GC is not this run's question). New columns:
`prem` (remember calls before the first collection — proactive saves) and
`cfab` (the answer asserts the fixture's final-line marker with a wrong
claim value while the true evicted needle is absent — fabrication, not
mere failure; a thrash cell that never answers is a non-answer, not a
confabulation). Recorded 2026-07-08, same model/provider/defaults as
t-1349 (`anthropic/claude-haiku-4.5` via OpenRouter). Spend: $1.13
matrix + $0.10 judge. Offline replay reproduces the table per cell
(asserted; guided cells replay with guidance on — the collector re-run is
token-sensitive, so the setting must match the recording).

### Results

```
fixture            arm        guid s turns evals  rpt refx rem prem rec coll reasons     drop ovl   in_tok  out_tok       cost wall_s  ok cfab judge
early-needle       none       off  1     9     8    0    0   0    0   0    0 -              0   0    29250      671  $0.032605   12.1 yes    -   3/3
early-needle       ring       off  1     9     8    0    0   0    0   0    4 s:4           20   0    20694      709  $0.024239   12.8  NO  YES   1/3
early-needle       ring       on   1     2     1    0    0   0    0   0    1 s:1            3   0     3942       84  $0.004362    2.5  NO    -   0/3
early-needle       mark-sweep off  1    11    10    1    1   0    0   0    8 s:8           66   0    26248      909  $0.030793   14.9  NO    -   0/3
early-needle       mark-sweep off  2    10     9    1    1   0    0   0    7 s:7           52   0    23689      705  $0.027214   13.0  NO    -   0/3
early-needle       mark-sweep on   1    17    16    0    1   0    0   0   16 s:16         202   0    42618     1479  $0.050013   25.1  NO  YES   0/3
early-needle       mark-sweep on   2     9     8    0    0   0    0   0    8 s:8           20   0    22531      745  $0.026256   13.7  NO  YES   1/3
early-needle       stack      off  1    11    10    1    1   0    0   0    8 s:8           53   0    25060      806  $0.029090   14.9  NO  YES   0/3
early-needle       stack      off  2     9     8    0    0   0    0   0    4 s:4           23   0    20634      696  $0.024114   12.6  NO  YES   2/3
early-needle       stack      on   1     2     1    0    0   0    0   0    1 s:1            3   0     3942       91  $0.004397    2.5  NO    -   0/3
early-needle       stack      on   2     2     1    0    0   0    0   0    1 s:1            3   0     3942       90  $0.004392    3.0  NO    -   0/3
early-needle       semantic   off  1    13    12    4    0   0    0   0    9 s:9           90   0    29969     1201  $0.035974   17.4  NO  YES   0/3
early-needle       semantic   on   1    27    26   25    0   0    0   0   26 s:26         702   0    56430     1737  $0.065115   36.0  NO    -   0/3
tangent-return     none       off  1     6     5    0    0   0    0   0    0 -              0   0    19255      530  $0.021905    9.9 yes    -   1/3
tangent-return     ring       off  1    27    26   24   11   0    0   0   25 s:25         676   0    45410     2007  $0.055445   39.5  NO    -   0/3
tangent-return     ring       on   1     2     1    0    0   0    0   0    1 s:1            3   0     3978       84  $0.004398    2.4  NO    -   0/3
tangent-return     mark-sweep off  1     9     8    0    3   0    0   0    7 s:7            8   0    19756      696  $0.023236   13.7 yes    -   1/3
tangent-return     mark-sweep off  2     6     5    0    0   0    0   0    4 s:4            6   0    12764      541  $0.015469    9.9 yes    -   3/3
tangent-return     mark-sweep on   1    26    30   25   10   0    0   0   25 s:25         645   0    72384     2640  $0.085584   45.5 yes    -   0/3
tangent-return     mark-sweep on   2    27    26   21    9   0    0   0   26 s:26         650   0    71938     2465  $0.084263   45.7  NO    -   0/3
tangent-return     stack      off  1    27    26   24   11   0    0   0   25 s:25         625   0    48948     1979  $0.058843   38.5  NO    -   0/3
tangent-return     stack      off  2    27    26   24   11   0    0   0   25 s:25         622   0    49183     1981  $0.059088   41.6  NO    -   0/3
tangent-return     stack      on   1     2     1    0    0   0    0   0    1 s:1            3   0     3978       84  $0.004398    2.3  NO    -   0/3
tangent-return     stack      on   2     2     1    0    0   0    0   0    1 s:1            3   0     3978       90  $0.004428    2.3  NO    -   0/3
tangent-return     semantic   off  1    27    26   24    4   0    0   0   25 s:25         660   0    48994     2168  $0.059834   43.3  NO    -   0/3
tangent-return     semantic   on   1    27    30   25   25   0    0   0   26 s:26         786   0    57402     1877  $0.066787   36.3  NO    -   0/3
memory-discipline  none       off  1     9     6    0    0   1    1   1    0 -              0   0    22502      842  $0.026712   14.8 yes    -   3/3
memory-discipline  ring       off  1     9     6    0    0   1    1   1    6 s:6           46   0    18169      761  $0.021974   14.1  NO    -   0/3
memory-discipline  ring       on   1     2     1    0    0   0    0   0    1 s:1            3   0     3990      107  $0.004525    2.5  NO    -   0/3
memory-discipline  mark-sweep off  1     9     6    0    0   1    1   1    6 s:6            0   0    19491      841  $0.023696   12.8 yes    -   3/3
memory-discipline  mark-sweep on   1     9     6    0    0   1    0   1    8 s:8           56   0    20530      776  $0.024410   12.2  NO    -   0/3
memory-discipline  stack      off  1     9     6    0    0   1    1   1    6 s:6            6   0    19231      824  $0.023351   13.9 yes    -   3/3
memory-discipline  stack      on   1     2     1    0    0   0    0   0    1 s:1            3   0     3990      100  $0.004490    2.5  NO    -   0/3
memory-discipline  semantic   off  1    11     6    0    0   3    1   1    8 s:8           26   0    24545     1053  $0.029810   15.4 yes    -   2/3
memory-discipline  semantic   on   1    27    26   25    0   0    0   0   26 s:26         702   0    57726     2180  $0.068626   36.1  NO    -   1/3
```

### Verdict: REFUTED — guidance dominated, with the opposite sign

Caveats as before (one model, n<=2, provider-default temperature), plus
the load-bearing one: at these budgets (1600-2000 tokens) the fragment
(~650-700 tokens: delegation + memory + GC + cost blocks) is **33-44% of
the whole context budget**, and system-message content is protected from
eviction by every strategy. Real deployments run windows 10-100x larger,
where the fragment is noise — this run measures guidance under extreme
pressure, which is exactly where GC's failure modes live.

1. **The hypothesis's magnitude claim held; its direction claim failed.**
   The guidance axis produced far larger behavioral deltas than any
   strategy swap — by destroying cells, not rescuing them. Guided arms:
   1 success in 12 cells (mark-sweep, tangent s1, at 5.5x the unguided
   cost). Unguided arms: 5 successes in 13. Best unguided strategy
   (mark-sweep: 3/4 fixtures-samples ok, $0.015-0.031) beats the best
   guided cell, never mind the worst (stack guided: 0/5, task lost by
   turn 2, every sample).
2. **Ring and stack + guidance = silent task loss, 5/5 cells, n=2
   reproducible.** The fragment fattens the system message; the first
   tool result trips the 0.85 threshold on turn 1-2; ring's and stack's
   degrade paths have NO last-user-message guard (only semantic
   hard-protects the task statement; mark-sweep refuses incomplete
   lifecycles), so the collection evicts the task itself; the model
   replies "I'm ready to help! Please provide the task steps" and the
   loop accepts a no-tool-call reply as the final answer. 2 turns,
   ~$0.0044, identical signature on all three fixtures.
3. **Guidance did not eliminate confabulation — it relocated it.**
   `early-needle`, the confabulation fixture: unguided, ring/stack/
   semantic fabricated access codes (`e2c7f891`, `BATCH-2024`,
   `AX7K92QM`; cfab 4 of 6 GC cells) while mark-sweep re-fetched the real
   code honestly (refx=1, failed only on arithmetic). Guided, the ONLY
   strategy that still reached the final step — mark-sweep — fabricated
   codes in BOTH samples (`7f4e2c91`, `42857`) despite the §2.4 text
   saying "re-run the command or `recall` — do not guess". The other
   guided arms produced no answer at all (task loss or thrash), which is
   why their cfab column is clean: dead agents don't confabulate.
4. **The §2.2 memory block changed nothing measurable.** rem=0, prem=0,
   rec=0 in every guided cell outside the scripted memory fixture — no
   proactive saves, no recall-instead-of-refetch, same as unguided and
   same as t-1349. On `memory-discipline` itself the guided cells did
   scripted remember/recall no better than unguided (and lost the WARN
   count or the task). No cell reacted to a `[frame: ...]` annotation by
   recovering; guided stack never lived long enough to see one.
5. **The strategies that keep the task thrash instead.** Guided
   mark-sweep and semantic collected essentially every turn (16-26
   collections vs 4-9 unguided): the fragment is permanent uncollectable
   ballast, so the usable interior shrinks below one turn's working set —
   the t-1349 restart loop, now guidance-induced. Guided semantic on
   `memory-discipline` went 27 turns/$0.069 where unguided went
   9-11/$0.030.
6. **Unguided arms vs t-1349 (tool-description delta, uncontrolled
   provider variance):** tangent-return reproduced exactly (mark-sweep
   the only GC survivor, cheaper than control; ring/stack/semantic
   thrash-looped); early-needle got worse (ring now confabulates instead
   of re-fetching; all four strategies failed); memory-discipline mostly
   reproduced (ring lost the WARN count). The t-1349 topline — mark-sweep
   behaviorally safest under pressure, retention-arithmetic winners
   fragile — held on re-record.

### What this means for the default strategy

t-1348 flipped the default to stack on offline retention data; t-1349
inverted that behaviorally; this run says **guided-stack is not
defensible at high context pressure**: with the shipped default-on
guidance, stack (and ring) silently lose the task statement and report a
2-turn non-answer as completion — the worst failure in the table, and the
cheapest, so cost dashboards would read it as a win. Recommendation
(decision Ben's):

- Do not keep stack as default while its degrade path can evict the last
  user message. The fix is mechanical and semantic already has it: hard-
  protect system + last-user in ring/stack (then re-run these 8 cells,
  ~$0.30, to see whether guided-stack becomes defensible).
  **Done — t-1367; re-run below.**
- Gate the fragment on budget headroom: delivering ~700 tokens of
  operations prose into a <=2k-token window converts guidance into
  pressure. A simple rule (skip or ship a compact variant when the
  fragment exceeds a few percent of `context_budget`) would have spared
  every guided failure here without touching real deployments.
  **Done — t-1368 (5%/15% thresholds; the §2.2 memory block was also
  demoted to draft per the promotion gate).**
- On today's evidence the behavioral default is mark-sweep, unguided at
  small budgets: only strategy to complete tangent-return in both runs
  (beating the no-GC control on cost), only guided arm to complete
  anything, honest re-fetch where others fabricated. Its known offline
  weakness (reclaims little) is what preserves coherence.

Mechanism gaps -> candidate tasks: last-user protection for ring/stack
(finding 2 — **landed as t-1367**, see the follow-up below); budget-aware
guidance delivery (finding 5 — **landed as t-1368**); the loop accepting
a final answer from a turn whose prompt no longer contains the task
(finding 2 — the gc_collect stream knows the last user message was
dropped; the loop could refuse to treat the next no-tool reply as DONE;
still open, though t-1367 removes the known trigger).

### t-1367 follow-up: the deciding cells, re-run with both fixes

Both recommended fixes landed and the 8 deciding guided ring/stack cells
were re-recorded 2026-07-08 (same model/provider/defaults as above):

- **t-1367** — ring and stack hard-protect the system message and the
  last user message through every degrade phase (the guarantee semantic
  always had; now a docs/GC.md invariant of all strategies, asserted by
  the offline matrix at zero drops).
- **t-1368** — fragment delivery is budget-gated: full fragment only when
  it costs <= 5% of `context_budget`, a 2-4 sentence minimal core up to
  15%, nothing above that. At these fixtures' 1600-2000-token budgets the
  fragment is **suppressed**, so a guided cell now differs from its
  unguided twin only by the gate itself — the ~700-token ballast that
  caused turn-1 collections is gone. (The §2.2 memory block was also
  demoted from the full fragment to draft: rem=0/prem=0/rec=0 in all 12
  guided cells above.)

Each fix invalidated recordings whose replayed collector re-run could no
longer reproduce them (GC replays live and token-sensitively): t-1367 the
8 guided ring/stack cells, t-1368 the 8 guided mark-sweep/semantic cells.
All 16 were deleted with their invalidating commits; the t-1364 table
above is the historical record. Only the ring/stack cells were re-recorded
— they carry the t-1367 question; guided mark-sweep/semantic at these
budgets would measure nothing the unguided arms don't (no fragment
renders), so they are out of the recording plan until a
realistic-budget guidance eval exists.

Spend: $0.26 matrix + $0.02 judge = **$0.29** (estimate was ~$0.30).
Offline replay reproduces every re-recorded cell (asserted per cell, and
re-verified from a cold offline run).

**Before (t-1364, silent task eviction):** all 8 cells identical — the
first collection evicted the task statement, the model replied "I'm ready
to help! Please provide the task steps", and the loop accepted it as
final. 2 turns, 0 evals, ~$0.0044, `ok=NO` with nothing attempted.

```
early-needle       ring       on   1     2     1    0    0   0    0   0    1 s:1            3   0     3942       84  $0.004362    2.5  NO    -   0/3
early-needle       stack      on   1     2     1    0    0   0    0   0    1 s:1            3   0     3942       91  $0.004397    2.5  NO    -   0/3
early-needle       stack      on   2     2     1    0    0   0    0   0    1 s:1            3   0     3942       90  $0.004392    3.0  NO    -   0/3
tangent-return     ring       on   1     2     1    0    0   0    0   0    1 s:1            3   0     3978       84  $0.004398    2.4  NO    -   0/3
tangent-return     stack      on   1     2     1    0    0   0    0   0    1 s:1            3   0     3978       84  $0.004398    2.3  NO    -   0/3
tangent-return     stack      on   2     2     1    0    0   0    0   0    1 s:1            3   0     3978       90  $0.004428    2.3  NO    -   0/3
memory-discipline  ring       on   1     2     1    0    0   0    0   0    1 s:1            3   0     3990      107  $0.004525    2.5  NO    -   0/3
memory-discipline  stack      on   1     2     1    0    0   0    0   0    1 s:1            3   0     3990      100  $0.004490    2.5  NO    -   0/3
```

**After (t-1367 + t-1368):**

```
early-needle       ring       on   1    10     9    0    0   0    0   0    6 s:6           42   0    22768      903  $0.027283   15.4  NO  YES   2/3
early-needle       stack      on   1     9     8    0    0   0    0   0    4 s:4           23   0    20536      656  $0.023816   12.9  NO  YES   2/3
early-needle       stack      on   2    12    11    1    1   0    0   0    9 s:9           75   0    27378     1016  $0.032458   17.9  NO  YES   0/3
tangent-return     ring       on   1    27    26   24   11   0    0   0   25 s:25         676   0    45437     2089  $0.055882   40.6  NO    -     -
tangent-return     stack      on   1    10     9    7    3   0    0   0    8 s:8           64   0    17990      647  $0.021225   26.1  NO    -   0/3
tangent-return     stack      on   2    27    26   24   11   0    0   0   25 s:25         625   0    48940     1999  $0.058935   47.2  NO    -   0/3
memory-discipline  ring       on   1     9     6    0    0   1    1   1    6 s:6           12   0    18315      726  $0.021945   15.1 yes    -   3/3
memory-discipline  stack      on   1     9     6    0    0   1    1   1    6 s:6            6   0    19130      816  $0.023210   13.4 yes    -   3/3
```

(The tangent/ring judge verdict is recorded but printed `-`: the model
wrapped its JSON in prose containing an earlier brace, defeating the
lenient extractor; the response's own conclusion is all three booleans
false, i.e. a 0/3. Thrash-transcript verdicts remain the least reliable
rows of the column, as in t-1349.)

**Findings:**

1. **The task-eviction failure is gone: 0/8 cells** (was 8/8). Every
   re-recorded cell actually attempts the task — 9-27 turns, 6-26 shell
   steps, the full fixture script visible in the trace. No cell ends on
   a no-tool "ready to help" reply, and no gc_collect event drops the
   last user message (now structurally impossible).
2. **Guided rows now mirror their unguided twins**, as the budget gate
   predicts (the fragment is suppressed; remaining deltas are sampling
   variance). memory-discipline: both complete, 3/3 judge, at or below
   unguided cost — GC again pays for itself when memory discipline is
   scripted. early-needle: all three cells fail with **confabulated
   access codes** (cfab YES), the t-1349 finding-3 mode — neither fix
   claimed to address it, and it persists exactly as in the unguided
   arms. tangent-return: ring s1 and stack s2 reproduce the 25-collection
   thrash loop (turn-cap non-answer); stack s1 escaped the loop but
   answered with the wrong category order after re-fetching.
3. **The cheapest-failure trap is defused.** t-1364's worst property was
   that the broken cells were also the cheapest ($0.0044 — a cost
   dashboard would read task loss as a win). The failure modes that
   remain (thrash, confabulation) are all visible: expensive, repeated
   commands, wrong needles.

**Is guided-stack now defensible?** As a *safety* matter, yes: the P0
failure — silently losing the task and reporting a 2-turn non-answer as
completion — is mechanically impossible (t-1367), and guidance can no
longer act as eviction-protected ballast at small budgets (t-1368).
Guided-stack now behaves exactly like unguided stack. As a *performance*
matter it is not the winner at extreme pressure: stack still thrashed or
confabulated on 2 of 3 fixtures here, same as unguided.

**Does the default-strategy question (stack vs mark-sweep) still need
Ben's decision? Yes — with the urgency changed.** The new numbers remove
the "stack is indefensible" forcing function; what remains is the same
behavioral split both prior runs found, unchanged by these fixes:
mark-sweep is still the only strategy that completes tangent-return
(both recording generations, cheaper than the no-GC control) and the only
one whose early-needle failures were honest re-fetches rather than
fabrications, while stack keeps the best offline retention arithmetic and
the best worst-case shape on chat-heavy windows. On behavioral evidence
at extreme pressure the default would be mark-sweep; on retention
arithmetic and structural predictability it stays stack. Both positions
are now survivable — the decision is a genuine trade-off, not a bug fix,
and these budgets (1.6-2k tokens) remain far below real deployments, so
steady-state evidence at realistic budgets would settle it better than
another extreme-pressure sample.

## Eviction markers (t-1360) and recording validity

Three rounds of behavioral evidence (t-1349 finding 3, t-1364 finding 3,
the t-1367 re-run) converged on one mechanism gap: models fabricate
evicted content (access codes: `CDBH92`, `7f4e2c91`, ...) instead of
recovering or admitting loss, because collection was SILENT — nothing in
the window said what was removed or how to get it back. t-1360 is the
mechanism-level fix: every strategy's `collect()` now leaves a compact,
deterministic `[gc: ...]` marker line where it dropped messages — kind,
identifying handle (tool-call id; recall query; turn ordinal), and the
recovery affordance ("re-run the call", "recall the memory", "ask the
user again" — always "do not guess"). Consecutive drops aggregate into
one line; markers are themselves droppable (a replacing marker absorbs
the count, degrade coalesces to a single "earlier context compacted"
line, terminal suppression is recorded on the gc_collect event); markers
count toward the window budget (the collector re-collects with the
marker cost reserved rather than overflowing). Stack's `[frame ...]`
annotations now carry the call id and, when the preview is truncated, an
explicit "evicted; re-run to recover" clause — a popped frame is its own
marker, never double-marked. Mark-sweep's elision annotation joined the
same `[gc: ...]` family. gc_collect events gained `markers`,
`marker_kinds`, `markers_coalesced`, `markers_suppressed`.

**Offline validation (this repo, asserted):** the matrix asserts, per
cell, that evictions leave an in-window marker (or recorded
suppression), that convergence holds with markers included in the
budget, and that two runs produce identical windows *and identical
message ids* (marker ids are derived, never minted). The behavioral
table gained an `mkr` column (in-window marker high-water from
gc_collect).

**Recording validity:** provider effects replay by effect id regardless
of prompt bytes, so every existing behavioral recording still replays —
final answers, turns, tool counts, and usage reproduce exactly. But GC
re-runs live and token-sensitively during replay, and the marker-era
collector's gc stream (dropped counts, tokens, marker fields) cannot
reproduce recordings made without markers (observed: early-needle/ring
replays 44 drops vs 42 recorded). Pre-marker recordings — detected by
the absence of the `markers` field on their gc_collect events — replay
with the gc-derived fields compared leniently (the t-1222 stance);
everything else stays strict. **Fresh recordings are needed before any
behavioral claim about markers** — deliberately not recorded with
t-1360 (key near expiry; batch with the next round): **t-1369**, whose
deciding question is whether early-needle's fabricators flip to
re-fetching. **Done — t-1369, next section.** The hand-written judge
fixture keys were regenerated for the marker-era windows (replay-path
plumbing only, still not real judgments).

## Marker-era re-record (t-1369, online): do the fabricators flip?

The deciding question, after three generations of the same finding
(t-1349 finding 3, t-1364 finding 3, the t-1367 re-run): models
fabricated evicted content because collection was silent — do they flip
to honest recovery (re-run / recall / admit) now that t-1360 markers
name what was evicted and how to get it back?

**Design.** 14 cells re-recorded 2026-07-08 on the marker-era runtime
(0199d44 + c70e98f), same model/provider/defaults as every prior round
(`anthropic/claude-haiku-4.5` via OpenRouter), priority-ordered under a
$1.50 cap:

1. **early-needle x all four strategies x guided, n=2** — the deciding
   cells. Guided is the shipped default; at these budgets the t-1368
   gate suppresses the fragment entirely (~700 tokens > the 15%
   ceiling), so **the markers are the whole intervention** — c70e98f's
   §2.4 marker text never renders here (its promotion-gate re-record is
   satisfied by this batch, but only degenerately; see residuals).
2. **early-needle stack unguided, n=2** — the marker-vs-text isolation
   pair: with the fragment suppressed, a guided cell's prompt is
   byte-identical to its unguided twin, so the guided/unguided delta
   bounds sampling variance rather than measuring text.
3. tangent-return stack + mark-sweep guided n=1 (does marker presence
   change thrash?), and memory-discipline ring + stack guided n=1
   (spot cells).

The stale pre-marker recordings at these 8 pre-existing paths were
deleted (the t-1364/t-1367 tables above are the historical record);
remaining pre-marker cells still replay leniently. New columns, all
programmatic from traces: `mkref` (assistant texts quoting literal
`[gc` / `[frame` marker syntax — prose like "evicted" is deliberately
not counted because the shipped `remember` tool description uses that
word in every cell's offer), `rcov` (recovery action: probe re-fetch
beyond the task's allowance, or `recall` beyond the fixture's scripted
count), `admt` (a failed final answer that admits the value is
unavailable instead of asserting one; phrase-check lower bound).

Spend: $0.70 matrix + $0.05 judge = **$0.75** (cap $1.50). Offline
replay reproduces every cell (asserted per cell, re-verified from a
cold offline run — the marker-era cells replay strictly, gc stream
included).

### Results (the 14 marker-era cells)

```
fixture            arm        guid s turns evals  rpt refx rem prem rec coll reasons     drop ovl mkr mkref   in_tok  out_tok       cost wall_s  ok cfab rcov admt judge
early-needle       ring       on   1    21    20    8    2   0    0   0   18 s:18         316   0   1     0    50466     1828  $0.059606   38.2  NO  YES  yes    -   1/3
early-needle       ring       on   2    16    15    5    3   0    0   0   13 s:13         174   0   1     0    37223     1319  $0.043818   27.7  NO  YES  yes    -   0/3
early-needle       mark-sweep on   1    10     9    0    1   0    0   0    5 s:5           38   0   1     0    23187      723  $0.026802   15.2  NO    -  yes    -   1/3
early-needle       mark-sweep on   2    10     9    0    0   0    0   0    6 s:6           42   0   1     0    23508      935  $0.028183   18.2  NO  YES    -    -   1/3
early-needle       stack      off  1    27    30   22    4   0    0   1   22 s:22         620   0   1     0    65871     2498  $0.078361   42.0  NO    -  yes    -   0/3
early-needle       stack      off  2    27    30   21    4   0    0   1   22 s:22         657   0   1     0    65728     2531  $0.078383   46.4  NO  YES  yes    -   1/3
early-needle       stack      on   1    27    22   13    4   1    0   3   22 s:22         567   0   1     0    66269     2456  $0.078549   47.2  NO    -  yes    -   0/3
early-needle       stack      on   2    27    25   11    2   0    0   1   22 s:22         564   0   1     0    66018     2287  $0.077453   46.9  NO    -  yes    -   0/3
early-needle       semantic   on   1    16    13    4    1   0    0   2   11 s:11         156   0   2     0    38566     1368  $0.045406   25.0  NO    -  yes    -   0/3
early-needle       semantic   on   2    18    16    6    0   0    0   1   13 s:13         212   0   2     1    43650     1600  $0.051650   29.7  NO  YES  yes    -   1/3
tangent-return     mark-sweep on   1     8     5    0    0   2    0   0    6 s:6           10   0   1     0    18071      849  $0.022316   17.2 yes    -    -    -   3/3
tangent-return     stack      on   1    27    21   19    9   5    0   0   25 s:25         595   0   1     0    55443     2341  $0.067148   48.5  NO    -  yes    -   0/3
memory-discipline  ring       on   1     9     6    0    0   1    1   1    6 s:6           46   0   1     0    18894      766  $0.022724   17.0  NO    -    -    -   0/3
memory-discipline  stack      on   1     9     6    0    0   1    1   1    6 s:6            6   0   0     0    19036      719  $0.022631   15.0 yes    -    -    -   3/3
```

The early-needle final answers, verbatim (the true code is
`MX-7749-KESTREL`, true total 21): ring s1 `ACCESS 7B2X9K TOTAL 21`,
ring s2 `ACCESS 7K9X2M TOTAL 21`, mark-sweep s1
`ACCESS MX-7749-KESTREL TOTAL 28`, mark-sweep s2
`ACCESS sk-7d42c991 TOTAL 25`, semantic s1
`ACCESS MX-7749-KESTREL TOTAL 33`, semantic s2 `ACCESS 7K9mR2 TOTAL
22`; all four stack cells hit the 26-turn cap mid-recovery with no
final line.

**The pre-marker baseline for contrast** — the current-descriptions
early-needle GC cells from the two prior generations (t-1364 unguided
rows above; t-1367 re-run guided rows above): 9 cells, **cfab 7/9**,
recovery actions (refx>0) 4/9, unprompted recall **0/9** — as in every
prior generation ("nobody reached for recall unprompted", t-1349
finding 3).

### Verdict: markers flip fabrication into recovery ATTEMPTS — reliably — but not yet into recovered ANSWERS

Caveats as always: one model, n<=2, extreme budgets (1.6-2k tokens),
provider-default temperature.

1. **Recovery actions: 4/9 -> 9/10.** Every marker-era early-needle
   cell except mark-sweep s2 took at least one recovery action the
   marker affordances name. **Unprompted recall: 0/9 -> 6/10** — the
   first unprompted `recall` calls in three generations of this eval
   (stack s1 issued three; semantic s1 two), plus the first unprompted
   `remember`s (stack guided s1; tangent mark-sweep, rem=2). The
   marker's "recall the memory" affordance is demonstrably read: the
   memory tools were offered identically in every prior round and
   never touched outside the scripted fixture.
2. **Fabrication: 7/9 -> 4/10 cells** (of answering cells, 7/9 ->
   4/6). And the two honest answers are the first GC cells on this
   fixture to end with the REAL code since t-1349's single ring
   re-fetch: mark-sweep s1 and semantic s1 both re-fetched
   (semantic also recalled) and answered `MX-7749-KESTREL`, failing
   only arithmetic — the honest-slip failure mode, not confabulation.
3. **The residual fabrications are recovery that didn't survive to
   the answer.** Ring s1/s2 re-ran `cat config/access-code.txt` two
   and three times (rcov yes) — and still fabricated: under 13-18
   collections the re-fetched result was itself re-evicted before
   step 9, and at answer time the model guessed rather than re-fetch
   once more. Recovery attempted != recovery delivered when nothing
   protects a just-recovered value (see the t-1166/t-1167 note
   below).
4. **Stack traded fabrication for a non-terminating recovery loop.**
   All four stack cells (guided and unguided alike) ran to the turn
   cap re-fetching, recalling, and re-counting (refx 2-4, rec 1-3,
   22 collections, ~$0.078 = 2.4x control) — honest, expensive, and
   answerless. Pre-marker stack answered cheaply and wrongly;
   marker-era stack never answers. That is the t-1367 trade
   (cheapest-failure trap defused: the failure is now visible and
   costly) extended to this fixture, and it wants a termination
   affordance, not more recovery pressure.
5. **Nobody admits loss: admt 0/14.** The marker says "re-run the
   call — do not guess"; models re-ran or guessed, but no cell ever
   answered "the code is unavailable". Admission may need an explicit
   affordance ("if recovery fails, say so") — currently only in the
   suppressed §2.4 text.
6. **Markers alone suffice for the attempt-flip; the text adds nothing
   measurable here.** The stack guided/unguided quadruple —
   byte-identical prompts, fragment suppressed — behaved as one
   population (27 turns, 22 collections, refx 2-4, rec 1-3, $0.077-
   0.079). Attribution: the flip is the markers' (the only delta vs
   the pre-marker generation on identical tool descriptions), and
   guidance-on vs off at these budgets measures nothing, as the
   t-1368 gate predicts. Whether §2.4's text adds anything ON TOP of
   markers is only measurable at budgets where it renders (>= ~4.7k).
7. **Thrash is marker-insensitive.** Tangent-return stack still looped
   (25 collections, refx 9) — the restart loop is driven by evicted
   progress narration, which markers name but cannot replace.
   Mark-sweep completed tangent-return again (3/3 judge, $0.0223,
   fourth straight generation), this time remembering unprompted.
   Memory-discipline: stack perfect (3/3); ring recovered the token
   correctly via scripted recall but miscounted WARNs (11 vs 6) — an
   arithmetic slip, n=1 sampling variance against its pre-marker pass.
8. **Marker mechanics behaved:** in-window marker high-water was 1-2
   in every cell (fusion/coalescing kept markers from accumulating
   across 22-25 collections), no suppressions, and marker-era cells
   replay strictly — dropped counts, marker fields and all. `mkref`
   was ~0 (one quote, semantic s2): models act on markers without
   quoting them, so `rcov` deltas, not quotes, carry the attribution.

**For t-1166/t-1167's parked designs:** markers change what
compaction must preserve. The recovery affordance makes the *handle*
(tool-call id, memory slug) load-bearing rather than the content —
compaction can be aggressive about content if handles survive. But
finding 3 is the new constraint: a recovered value must be
*re-protectable* or recovery is wasted spend — the strongest in-vivo
argument yet for the t-1351/t-1167 direction (recall/re-fetch overlap
tracking feeding a hot set that collection respects), and finding 4
wants a loop-termination signal (the t-1349 progress-narration gap
again: "you already re-ran this; steps 1-N are done").

**Default-strategy note (the open stack vs mark-sweep decision):** this
run moves nothing against mark-sweep and adds one point against stack
at extreme pressure — mark-sweep produced the only honest early-needle
answer, the only tangent-return completion (fourth straight
generation), and both at the lowest GC-arm cost, while stack converted
its cheap wrong answers into turn-cap recovery loops. The trade-off
framing from the t-1367 follow-up stands; the evidence gap is still
steady-state behavior at realistic budgets.

**Honest residuals:** fabrication persists in 4/10 deciding cells;
stack's recovery loop never terminates; no admissions anywhere; §2.4's
marker text is behaviorally unvalidated at any budget where it
actually renders (this batch discharges c70e98f's re-record obligation
only because no cell renders the text); the two thrash-transcript
judge verdicts (early-needle stack unguided s2, tangent stack guided
s1) again wrapped their JSON in prose and remain the least reliable
rows; and no early-needle GC cell has ever passed both needles —
the honest ceiling at these budgets is "right code, wrong sum".
