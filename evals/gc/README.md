# GC eval fixtures

Trace fixtures consumed by `cargo test -p agent-core --test gc_evals -- --nocapture`
(the strategy x cache x pressure comparison matrix from docs/GC.md).

## What lives here

- `*.jsonl` — real recorded agent traces. The harness extracts every
  `InferCall` whose full prompt contains a tool chain (>= 3 tool calls and
  >= 3 tool results) and runs each strategy against it at several budget
  pressures.
- Synthetic shapes (chat-heavy, open-tail tool chain, mixed session) are
  generated in code inside `gc_evals.rs`, labeled `synthetic:` in the table.
  They stand in for shapes this directory does not cover yet — replacing
  them with real recordings is always preferred.

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

- a long coding session (many shell frames, interleaved narration)
- a chat-heavy session with little tool use
- a hydration-heavy session (large Get/temporal context blocks)

## Hygiene

- Fixtures must not contain credentials. Record with `--eval-env inherit`
  (the default, which strips `*_API_KEY`/`ANTHROPIC_AUTH_TOKEN` from tool
  children) and skim the JSONL for secrets before committing — prompts can
  quote whatever the shell printed.
- Keep fixtures small enough to review (< ~1 MB). Truncate sessions rather
  than committing megabyte transcripts.

## Reading the matrix

One row per (case, pressure, strategy, cache policy): tokens before/after,
reduction %, messages and tool results retained, frames popped (stack),
stable cache prefix length, prefix invalidation, convergence, and warnings
when the last user message or the window tail did not survive. Convergence
is asserted for ring and stack (they carry the front-drop degrade path);
mark-sweep is best-effort and only reported. The promotion gate
(`gc_challengers_improve_over_ring_on_tool_chains`) requires challengers to
retain more structure than ring on tool-chain windows.
