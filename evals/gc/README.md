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
