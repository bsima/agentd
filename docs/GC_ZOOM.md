# Zoomable eviction and UniiChat: an assessment (t-1597)

Status: assessment, 2026-10-08; item 1 (zoom) **implemented** as t-1609
(`--gc-zoom`) — see "Implemented: zoom (t-1609)" below for what shipped
and where it deviates from this sketch. This document answers
t-1597's questions with evidence and recommends what to build. Each claim
is marked **measured** (a command or test was run), **read** (verified in
source, with a file:line), or **inferred/guessed**.

## Recommendation

1. **Build zoomable eviction now (opt-in).** agentd's message log is already
   append-only and complete; GC only shapes a per-call view of it (read,
   below). A zoom tool can return any evicted message
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
4. **Build item 3 (LLM summaries) as an opt-in GC strategy, after
   zoom.** Summaries only lose safely if the original is recoverable;
   without zoom, a vague summary loses the decision. See "Design
   decisions" below.
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
  `context_overflow` events. `--gc-log` defaulted to off, and Paseo
  launched `agent --acp` without it (on by default since t-1608). Only 4.3% of InferCalls carry a
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

## Design decisions (2026-10-08)

Principle: like every existing GC strategy and modifier, zoom and LLM
summarization are opt-in. They must not constrain future context
management, so with them off, the agent loop's program hash, the replay
fixtures, and the marker text are unchanged.

### Layering

- **GC strategies shape the view and mint addresses.** A marker names
  what was dropped; a summary line names the messages it covers.
- **Zoom resolves an address to the original.** It is a read over the
  log, not a strategy. For bodies in the live `history` it is a pure
  read. A future backend can resolve bodies that left `history` (a
  spill store, if the log is ever bounded) through a `Retrieve` against a
  log-store hydration source. The model sees one tool either way.
- **Not a hydration source today.** Hydration sources are external
  stores behind the `Retrieve` effect and cannot see the machine's
  `history`. `TemporalSource` already declines to serve the running
  session's own checkpoints for the same reason: "the live history
  already contains those messages" (`temporal.rs`).

### Zoom (`--gc-zoom`, opt-in modifier)

- A modifier like `hot-keep` and `cited-keep`, combinable with any
  strategy. Off by default.
- **Addresses are log positions**: the message's index in the
  append-only `history`. Positions are unique and cover every message
  kind; tool-call ids are accepted as an alias, resolving a duplicate to
  the most recent match. Provider ids are not rewritten.
- **Any message is zoomable** (decided): tool results, assistant turns,
  and user turns. Exact recall of what the user said replaces today's
  "ask the user again" for evicted user turns.
- **Markers take their recovery affordances from the enabled tools.**
  Strategies never learn that zoom exists; with zoom off the marker text
  is byte-identical to today's.
- Built-in IR arm appended after existing blocks (the `ask_user`
  pattern), paged for large bodies, registered with the write-barrier as
  a recall tool.

### LLM summary strategy (opt-in GC strategy)

This is the `stack-smart` variant docs/GC.md deferred "for after the eval
harness exists"; the harness exists. UniiChat's tree and compaction
prompt are the starting design.

- **Summarizer calls are agent effects** (decided): AgentIR child Infer
  effects, traced, costed, replayable, and shown in ACP clients as
  `infer` cards. Proposed shape: an opt-in block in the agent loop,
  before the turn Infer, that summarizes messages not yet summarized.
- **Summaries live in a machine env variable**, so checkpoints carry them
  without making `GcState` persistent, and other strategies are
  untouched. UniiChat's rule applies: never rebuild the view from
  scratch, or every cache entry dies.
- **`collect()` stays pure.** The strategy reads summaries through the
  async pre-pass, the cache-consuming pattern docs/GC.md sanctions for
  semantic embeddings.
- **Requires zoom.** Summaries lose detail; the original must stay
  recoverable. The strategy should refuse to start without `--gc-zoom`
  or enable it.
- Open in design: summary unit (per message, per tool frame, or
  UniiChat's binary tree), size budget, which model summarizes, and
  whether a turn waits for pending summaries.

### Switches

- **CLI flags only** (decided), like every GC option today. No prompt
  frontmatter, `agent.md` spec, or SDK builder surface.
- **GC telemetry on by default** (decided; shipped in t-1608). `--gc-log`
  adds trace events and never changes what the model sees;
  `--gc-log=false` opts out.
- **The low-water target is an option** (`--gc-low-water`), not a new
  default.
- **Eval budget**: none set; providers are mostly flat-rate.

## Implemented: zoom (t-1609)

`agent --gc-zoom` (CLI only; any strategy; an error with `--gc none`,
where nothing is evicted and the tool would only cost tokens).

**What shipped (read against the code):**

- **The arm.** `AgentLoopOptions::zoom: Option<ZoomOptions>` appends one
  dispatch/body block pair after every native arm (ask_user included), so
  the zoom-free program keeps every block id. The body is one `Let` of a
  new pure expression, `Expr::Zoom { history, arguments, page_chars }`
  (`ir.rs`; semantics `zoom::zoom_history`), then `Goto append_tool`. No
  effect, no trace event, no replay record. The offer follows the
  program (`program_dispatches_zoom`, the `ask_user` pattern).
- **Schema.** `{address: string, page?: integer}`. `address` is `#N` (a
  position in the append-only `history`, `#0` first), `#A-#B` (a
  consecutive range, rendered message by message), or a tool-call id
  (alias for its result; on duplicates the most recent, with a `note:`
  naming the others). A bare digit string is a position unless it is a
  call id; the empty id is rejected as a handle and never printed as one.
  A JSON integer is accepted as a position. `address` stays a string so a
  summary strategy (t-1612) can add tree addresses without a schema
  break.
- **Result.** One header line (`#7 tool result of shell call-3:`,
  `#1 user message:`, `#4 assistant message:` with its tool calls as
  `tool call <id>: <name> <args JSON>` lines; images noted, not inlined),
  then the content exactly as it entered history. Bad addresses and pages
  are `{"ok":false,"error":...}` tool results (errors-as-values).
- **Paging.** `page_chars = clamp(context_budget / 8 × 3, 1 KiB, 16 KiB)`:
  one page costs at most an eighth of the window under the GC estimator
  (3 chars/token) — room for the task, the working set, and the 15%
  headroom above the 0.85 trigger — and at most 16 KiB, the `recall`
  retrieval cap. Pages are char slices that reassemble exactly; each
  carries `[zoom #7 page p/P]` and a next-page instruction. The page size
  is program (budget-dependent hash when zoom is on).
- **Markers.** The interpreter records `GcState.zoom` (a `ZoomBook`:
  decoded message id → history position, call id → latest result
  position) before every Infer that offers `zoom` and whose prompt is a
  history var; every other call clears it. The shared renderers in
  `gc.rs` read it; strategies never do. Wording on, by artifact:
  run markers `[gc: N evicted — shell call-3 #7, user #1, assistant #4;
  recover: zoom #1-#7 (exact) — do not guess; re-run a call only if its
  result may have changed]` (the re-run clause only when the run held a
  tool result); frames `[frame call-3 #7: shell(args) -> preview —
  evicted; recover: zoom #7 (exact); re-run only if the result may have
  changed]` (`[zoom #6]` / `[call: #6]` when the narration or arguments
  preview was truncated); elisions `[gc: result elided — cat x (call-3,
  #7); recover: zoom #7 (exact) — …]`; the coalesced line `recover: zoom
  #2-#44 (exact) …`; the ledger `[evicted — zoom #7]` and footer
  `evicted results are recoverable exactly: zoom the #address — do not
  guess; re-run a call only if its result may have changed.` Escalation
  stays (demoted): `[gc: 'call-7' evicted 3 times — this content cannot
  stay in context: zoom #12 for only the part you need and restate it in
  your reply, summarize it into memory (remember), or ask the user — do
  not re-fetch it again]`. Off, every byte is the historical text.
- **Write-barrier.** `zoom` results are recall sources while zoom is
  offered (not unconditionally: a hallucinated `zoom` call on a zoom-free
  run is an unknown-tool error and must not move the hot set). They match
  through their decoded page, and a match marks the page's own chunks
  hot, so hot-keep protects the page the model asked for.
- **Guidance.** The `zoom` tool description, a §2.4 zoom line appended to
  the GC block, and a minimal-variant zoom sentence — all keyed on the
  call offering `zoom` (docs/GUIDANCE.md).
- **ACP.** A zoom has no effect events, so the bridge renders each
  requested `zoom` from the turn's `InferResult` as a completed `Read`
  card titled `zoom #7` (page noted when > 1).
- **Approvals resume.** `ResumeFacts.gc_zoom` keeps the arm through an
  output-contract repair after an approval resume (the resumed program
  itself comes from the checkpoint).

**Deviations from the sketch above, and why:**

- Tool name `zoom`, not `recall_evicted`: it matches UniiChat and Ava and
  collides with nothing; it is now reserved (`RESERVED_TOOL_NAMES`).
- Arguments `{address, page?}`, not `{call_id, page?}`: positions address
  every message kind (decided); call ids are an alias.
- A new `Expr::Zoom` instead of reusing `SelectToolResults`: that
  expression resolves only tool-result ids, drops duplicates to the
  first occurrence, and wraps results as user messages; zoom needs
  positions, ranges, most-recent duplicates, raw content, and paging.
- Ranges (`#A-#B`) are implemented now, not deferred: markers name each
  run's span, so the span had to be zoomable.
- The tool is offered only when `--gc-zoom` is set, not whenever GC is
  enabled (the 2026-10-08 opt-in decision supersedes "advertise it only
  when GC is enabled").

**Verified (offline):** zoom off, the cell program hashes to the
`program_hash` the recorded behavioral corpus carries
(`zoom_off_program_matches_the_recorded_corpus`) and every existing
recording replays strictly; the default loop program equals the
zoom-free builder's (`zoom_option_shapes_the_program_only_when_set`);
every evicted message is addressed under every strategy and cache policy
(`every_evicted_message_is_addressed_under_every_strategy`); zoom-off
annotations never mention zoom; a scripted stack session recovers an
evicted result by address, the page is write-barrier hot, no effect
event names the zoom, and the session replays byte-identically from its
trace (`plumbing_zoom_recovers_by_address_and_replays`).

**Behavioral eval:** arms {stack, generational} × {re-run markers, zoom}
in `gc_behavior_evals.rs`, a `zoom` column (zoom-call count), and a
`counter-drift` fixture whose early call issues a new ticket on every
run, so re-running cannot recover the value and `rot` flags an answer
carrying a re-issued ticket. Recording status: evals/gc/README.md
"Zoom (t-1609)".

**Observed while building (not changed here):** per-content eviction
counts grow by one per collection that removes the content, and every
Infer re-collects the full history, so a message evicted in three
consecutive collections escalates without any re-fetch. Escalated lines
therefore appear sooner than the t-1370 rationale ("the third eviction
is the observed loop signature") suggests. Worth its own look.

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

## Follow-up tasks (filed 2026-10-08)

| Task | Priority | What | Blocked by |
|---|---|---|---|
| t-1608 | P2 | GC telemetry on by default | — |
| t-1609 | P2 | Zoom: opt-in exact recall of any evicted message (`--gc-zoom`) | — |
| t-1612 | P2 | Opt-in LLM-summary GC strategy (`stack-smart`, UniiChat-style) | t-1609 |
| t-1611 | P2 | Investigate Anthropic prompt-cache hit rate | — |
| t-1610 | P3 | Opt-in low-water collection target (`--gc-low-water`) | t-1608 |
| t-1613 | P4 | Bound session log growth | — |
