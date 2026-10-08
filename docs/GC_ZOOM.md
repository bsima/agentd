# Zoomable eviction and UniiChat: an assessment (t-1597)

Status: assessment, 2026-10-08. No code changed. This document answers
t-1597's questions with evidence and recommends what to build. Each claim
is marked **measured** (a command or test was run), **read** (verified in
source, with a file:line), or **inferred/guessed**.

## Recommendation

1. **Build zoomable eviction now.** agentd's message log is already
   append-only and complete; GC only shapes a per-call view of it (read,
   below). A `recall_evicted` tool can return an evicted tool result
   verbatim from history: exact, free, deterministic, and replay-safe,
   with no re-execution. The primitive already exists
   (`Expr::SelectToolResults`). This is the one UniiChat idea agentd can
   adopt almost for free.
2. **Turn GC telemetry on by default first.** No production trace on
   this machine contains a single `gc_collect` event (measured), so
   every claim about GC in production is currently inference.
3. **Replace item 2 (age-ordered eviction) with hysteresis.** The
   transferable UniiChat lesson for caching is not merge order but
   batching: collect rarely and deeply. agentd's threshold timing
   collects to the trigger itself (read), so a long session trims a few
   percent on every call (inferred from traces). Add a low-water target.
4. **Defer item 3 (LLM compaction) until item 1 ships.** Summaries only
   lose safely if the original is recoverable; without zoom, a vague
   summary loses the decision.
5. **Rescope item 4.** The task's premise that agentd GC is per-turn is
   wrong: a session's every turn carries the full log, and GC collects
   across turns already (read). The real cross-turn gap is unbounded log
   growth, not view construction.

## UniiChat, as written

Source: the gist `VictorTaelin/91837951a5ce5b38f341ec1ba1df6449`, file
`optchat.md`, revision 3c190e0 (2026-10-08). Read in full.

- **It is a design spec, not code.** 474 lines of Markdown; no
  implementation, tests, simulator, or data. The latest revision
  (+403/−706 lines) fixed a bug the author says "trashed the cache".
- **Log**: append-only JSONL per day, one writer holding a lock,
  flush on every write. Message kinds: user, unii, tool, echo, work,
  note. Model thoughts are not logged.
- **Tree**: `node(0,i)` is message *i* compressed to a ~512-byte target
  (a ruler in the prompt; up to 5 "too long" retries; "a few bytes over
  is fine"). `node(l,i)` merges `node(l-1,2i)` and `node(l-1,2i+1)`.
  Messages already under 512 bytes are their own node.
- **Merge order**: `due = (T − last) / 2^l`, where T is the message
  count, l the pair's level, and `last` the pair's last message. It is
  content-blind and reproduces a binary counter: an independent
  reimplementation matched Taelin's `push` at all 20,001 steps tested,
  and the spec's earlier `first`-based rule matched at only 481, the
  figure the spec gives for its bug (measured).
- **Batching**: nothing merges until the view passes 128,000 bytes; one
  batch then merges the most-due pairs down to 64,000 bytes. Between
  batches the view only grows at its end, so the previous call's view
  is a byte prefix of the next.
- **The view is persisted, never rebuilt**: "the rebuilt view differs
  from the live one, and every cache entry dies". The view holds only
  summaries, never a raw message, not even the last one.
- **Prompt**: `[tools][system][user instructions][view][per-turn
  state][message]`. Per-turn state goes after the view, never in the
  system prompt.
- **Zoom**: `zoom(id+n)` opens a view line into its two children; `n = 1`
  returns the message whole, paged if long. The system prompt orders the
  model to "zoom until you have it whole, before you act, guess or ask".
- **Compaction prompt** priority: user words near-verbatim; then lasting
  effects and failures; then findings and replies; then tool steps.
  "Name a minor item in a word or two rather than drop it." Compaction
  shares the turn's system prompt and tools to share its cache. Model:
  Claude Haiku. Up to 8 compactions run concurrently; a turn waits until
  every earlier message is summarized ("seconds").
- **Tool output**: clipped to 30,000 characters (head and tail). Nothing
  handles overflow inside one turn.
- **The 98.6% cache-hit figure is a simulation**: "measured by replaying
  3,000 messages against a model of Anthropic's cache". No simulator or
  data is published. Treat it as an assertion.

## agentd today

### The log is already append-only and complete (read)

- Every Infer resolves its prompt by cloning the env `history` var
  (`ir_interpreter.rs:2943-2955`), then hydrates and runs
  `maybe_collect_prompt` on that local copy (`:1383-1391`). Only the
  response is written back to env (`:1543`).
- The agent loop appends assistant messages and tool results to the env
  `history` with `Expr::Push` (`ir_agent.rs:530-569`, `1033-1075`); the
  next turn re-reads `history` (`:248-252`, `1080-1092`).
- Every GC mutation — truncation, mark-sweep elision, eviction markers,
  frame annotations, the `[gc-ledger]` message — happens on the owned
  copy (`interpreter.rs:714-797`; `gc.rs:3088-3134`, `3624`). No path
  writes it back.
- Sessions carry the full log between turns: `run_turn` passes
  `runtime.history.clone()`, and `finish_turn` takes the machine's
  `history` (`main.rs:2745-2761`, `2960-2986`). Checkpoints store the
  full log (`main.rs:3337`). ACP uses the same path (`acp/turn.rs`).

So agentd is structurally what UniiChat is — a permanent log with a
compacted view — except that the view is recomputed per call, and the
model is told to recover by re-running.

### Recovery today is re-running (read)

Markers say `recover: re-run the call — do not guess`
(`gc.rs:1962-1975`, `3083`, `3134`). Re-running costs a tool call, can
return different output, and can repeat side effects. Hot-keep, the
3-strike escalation, and the progress ledger exist to contain that cost.

### What production traces show

Corpus: 3,751 traces, 2026-05-20 to 2026-10-08, 64,456 Infer calls,
about 20 models (measured).

- **No GC telemetry exists.** Zero `gc_collect`, `gc_truncate`, or
  `context_overflow` events. `--gc-log` defaults to off, and Paseo
  launches `agent --acp` without it. Only 4.3% of InferCalls carry a
  full prompt. Strategy, drop counts, marker and hot-keep counters, and
  `prefix_invalidated` are therefore unmeasurable (measured).
- **GC fired in about 17 of 3,525 runs** (inferred: a >2% drop in
  `input_tokens` from a ≥50k-token call in the same conversation,
  non-Anthropic models only, because agentd's Anthropic `input_tokens`
  excludes cache reads). 48 candidate events; models copied marker text
  into their replies 3 times, which confirms markers were live.
- **Each collection is shallow.** Median after/before ratio 0.927 (p10
  0.609, p90 0.974); one run had 16 events (inferred). This matches the
  code: with `threshold` timing the target *is* the trigger
  (`interpreter.rs:674-677`), so once over, each new turn tips the prompt
  back over and GC trims again (read).
- **Caching after GC is poor, but confounded.** The cache-hit rate on
  the call after a candidate event was 0.09 (n=42), against 0.69 for
  non-Anthropic prompts of ≥150k tokens without GC. The model with most
  events (gpt-5.6-sol) caches poorly above 150k regardless (0.23), and
  most calls before an event already had zero hits. The one clean
  `preserve` case (Kimi-K3) kept a 0.83 hit rate through GC. Inferred,
  not established.
- **No evidence of an evict-then-re-fetch loop**: 5 repeats in 337 Eval
  calls after GC events (1.5%), 4 of them state checks like
  `git status`. The no-GC baseline repeat rate is 7.3%. Too few calls to
  rule a loop out (measured).
- **Overflow**: 12 fatal context-overflow errors, all gpt-5.5. Three hit
  265-270k real tokens while the configured trigger was 231k: the token
  estimator undercounted (the t-1145 gap). Five overflowed from one large
  tool output at 33k-159k (measured).

## The proposed items

### 1. Zoomable eviction — build it

**Design sketch (read against the code, not implemented):**

- A built-in IR arm, like the `infer` tool's `infer_resolve`, not a
  `NativeTool`: native handlers receive only their arguments and cannot
  see the machine env (`tool.rs:136`, `ir_interpreter.rs:2051`).
- The arm is a pure `Let` over `history` using `SelectToolResults` (or a
  variant returning the raw content rather than a user-role wrapper),
  then `Goto append_tool`. A pure expression is not an effect: no replay
  record, deterministic under replay.
- **Arguments**: `{call_id, page?}`. Results can be up to 1 MiB of stdout
  plus 1 MiB of stderr (`interpreter.rs:115-116`); returning that whole
  would re-flood the window and be evicted again. Page it, as UniiChat
  does ("a long message comes in pages").
- **Advertise it only when GC is enabled**, appended after existing
  blocks, so the default program hash is unchanged (the `ask_user`
  pattern).
- **Markers** change to: `recover: recall_evicted call-3 (exact) — do
  not guess; re-run only if the result may have changed`.
- **Hot-keep**: re-injected content already trips the write-barrier
  (`gc.rs:1610`); add the tool to `RECALL_TOOL_NAMES` (`gc.rs:1447`).

**Edge cases found (read):**

- Tool-call ids come verbatim from providers and are never checked for
  uniqueness. The OpenAI streaming path leaves the id `""` when no chunk
  carries one (`provider.rs:549-550`). Resolve duplicates to the most
  recent occurrence and say so; never offer `""` as a handle.
- "Exact" means exactly what entered history: output is capped at 1 MiB
  per stream before it gets there.
- Results whose turn failed or was cancelled never entered history
  (`acp/session.rs:113-118`), so nothing can recall them — and no marker
  points at them either.
- `close_pending_tool_calls` writes a synthetic `turn_budget_exhausted`
  result under the real call id (`op.rs:504-527`); recall returns that
  text.

**Eval**: add a marker-variant axis to `gc_behavior_evals.rs`: arms
{stack, generational} × {re-run markers, recall markers}. Score with the
existing columns (needles, `refx`, `rpt`, judge, tokens) plus recall
calls. `early-needle` fits as is. Add one fixture where re-running gives
a different answer (a counter file the task increments, or a
time-dependent value): that is where exact recall wins outright and
re-running cannot. The t-1349 matrix cost $0.48 for 15 cells on Haiku;
expect a similar order (guessed).

### 2. Age-ordered cold eviction — replace with hysteresis

UniiChat's due-ordering serves its tree: it decides which summaries to
merge. agentd has no tree, and evicting any message breaks the provider
cache for everything after it, whatever its age. The part that protects
caching is UniiChat's *batching*: rewrite rarely, down to a low-water
mark, then grow only at the end.

agentd's `threshold` timing has no low-water mark (read). Proposal: add a
collection target below the trigger (for example collect at 0.85 down to
0.60; values guessed, to be swept). Eval: cache-hit rate (provider
`cached_input_tokens`) and collection count per run, against today's
`--gc-cache preserve`, on recorded long sessions.

### 3. LLM-compaction elide — later, after item 1

UniiChat offers no evidence for LLM summaries beyond assertion. Its own
design depends on zoom to make summary loss recoverable, and on blocking
turns until compaction finishes. In agentd, an LLM elide strategy without
`recall_evicted` would replace honest markers with lossy prose. Add it as
an eval arm once item 1 exists, with the compaction prompt's priority
list as the starting prompt.

### 4. Cross-turn memory tree — rescope

agentd GC already spans turns: each turn's prompt is the whole session
log, and GC collects that (read). What agentd lacks is any bound on the
log itself: history grows without limit in memory and in every
checkpoint, mid-turn checkpoints carry a second copy, and
`--enable-compaction` is accepted but unimplemented (`main.rs:227-229`,
`2879`). A tree of summaries is one answer to that; so is spilling old
bodies to a content-addressed store referenced by id. Design this when
log size becomes a measured problem; no trace here shows it yet.

## The open questions

- **Where does the recall store live for subagents?** In the parent's
  history. Delegated children are single completions with no tools
  (`ir_agent.rs:933`) and already receive verbatim results by reference
  through `context_refs`. They never need recall.
- **Should recall be permission-gated?** No (decided 2026-10-08). The
  model already saw the content in this run; recall re-reads the same
  history and exposes nothing new. Gating would make the safe recovery
  path costlier than the unsafe one, re-running.
- **Does item 1 retire the 3-strike escalation?** Demote it, and let the
  eval decide (decided 2026-10-08). Markers lead with recall; re-running
  and escalation stay as fallbacks; retire them in a later change only if
  the recall arms show they no longer fire.

## Side findings

- **Anthropic cache hits look broken (unverified).** Where recorded,
  `cached_input_tokens / input_tokens` was 0.002 for claude-sonnet-4-6
  and 0.023 for claude-opus-4-8, although agentd sends `cache_control`.
  This may be accounting (agentd's Anthropic `input_tokens` excludes
  cache reads) or a real miss. Worth its own look.
- **The token estimator undercounts on gpt-5.5**: three runs overflowed
  at 265-270k real tokens with a 231k trigger.
- **Delegated children share the parent's `GcState`.** A child Infer
  runs `maybe_collect_prompt` with it, bumping `infer_calls` and possibly
  eviction counts (read; impact not measured).

## Proposed follow-up tasks

1. GC telemetry on by default: emit `gc_collect`/`gc_truncate` events in
   every mode, including ACP. Prerequisite for every measurement above.
2. `recall_evicted`: built-in IR arm, paging, marker wording, eval arms.
3. Threshold hysteresis: low-water collection target, cache-hit eval.
4. Investigate Anthropic cache-hit accounting.
5. Later: LLM-compaction elide arm (after 2); log-size bound (when
   measured).
