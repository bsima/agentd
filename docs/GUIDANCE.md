# Runtime operations guidance (t-1356)

Status: **design + drafted guidance + eval plan — nothing here is wired in
yet.** This document specifies what operations guidance the runtime should
ship to its models, why it is a runtime component rather than user prompt
material, how it should be delivered, and how each piece gets validated
before it ships default-on. The delivery mechanism itself is future work;
the drafted guidance text below is written as it would ship.

## 1. The thesis: the system prompt is a runtime component

agentd ships primitives — `shell`, `infer`, `remember`/`recall`, GC,
approvals — with tool schemas but no operations manual. t-1354
([evals/delegation/](../evals/delegation/README.md)) measured what that
costs, with a real model choosing:

- Given the bare `infer` tool and **no guidance**, the model used it **zero
  times in five fixtures** — including the one fixture (generation
  offload) where t-1342 had measured delegation winning ~2.7x on cost.
- Given **three sentences of guidance** (when to delegate, pass bulky
  material via `context_refs`, when NOT to delegate), the same model on the
  same tasks delegated exactly where the measured economics say it pays —
  **2.26x cheaper than unguided** on generation-offload — and nowhere else:
  zero over-delegation on the restraint fixture, no indiscriminate
  delegation on the neutral fixtures. The guidance even generalized: the
  guided arm answered "what is 17 * 23?" in one turn with no tool use,
  while unguided arms shelled out to a calculator.

The t-1354 meta-finding, verbatim: *"the mechanism's bottleneck is
discoverability, not model judgment."* The model's judgment tracked the
measured economics of t-1342 almost exactly — once told the mechanism
exists and what it is for.

The conclusion this repo should operate under: **a primitive without
operations guidance is not shipped; it is merely present.** Guidance has
the same status as a tool schema — it is part of the runtime's
model-facing API surface, versioned, traceable, and eval-gated like any
other runtime component. It is not the user's job to reverse-engineer the
runtime's economics and write this prose themselves (today every consumer
would have to, and none do: the CLI's default system prompt mentions only
the shell tool; the SDK's `instructions` is empty by default).

This matches how mature harnesses work (§6): the prose that teaches a
model *when* to use a tool ships with the harness, in the tool
descriptions and a runtime-owned prompt layer, not in user config.

## 2. Pattern catalog

Format per pattern: what it is → when it pays (grounded in eval data where
it exists) → the failure mode of unguided models → drafted model-facing
text (as it would ship) → validation (which eval, which arms) → mechanism
gaps guidance cannot paper over.

Conditionality is a design invariant throughout: **each block ships only
when its capability is live** (the `infer` block only when the tool is
advertised, the memory block only when a memory backend is registered —
the same condition that already gates the `remember`/`recall` tool specs
in `ir_tool_specs`, the approval block only when some effect is gated).
Guidance about a tool the model does not have is noise, and noise is the
failure mode of guidance itself.

### 2.1 Delegation (economics + context_refs + restraint)

**What.** Use the `infer` tool to hand a subtask to a (usually cheaper)
model, passing bulky material by reference.

**When it pays** (t-1342, scripted mechanics; t-1354, behavioral):
generation-heavy work (~2.7x via output-rate arbitrage), digesting bulky
already-fetched material by reference (~1.6x on noisy-middle-step). When
it does not: direct questions (pure overhead), one-line tool work
(t-1354's count-in-files: a grep beats any delegation), by-copy material
transfer (7.7x structural tax), and any latency-sensitive path (the one
guided delegation was the *slowest* cell: +19s serialized child
round-trip).

**Unguided failure mode.** Non-use, totally: 0 delegations across all
five `tool-unprompted` cells in t-1354. Plausibly connected to the schema
offering no model catalog — the model has no id it could confidently pass
(t-1342 finding 3). Secondary failure mode (scripted evidence): by-copy
delegation when material is pasted rather than referenced.

**Drafted guidance** (ships when `infer` is advertised):

> You can delegate subtasks to another model with the `infer` tool.
>
> - Delegate when a subtask is generation-heavy (long boilerplate, many
>   candidates to write up) or requires digesting bulky material you have
>   already fetched.
> - Pass bulky material by reference: set `context_refs` to the ids of the
>   tool calls that produced it. Never paste large tool output into the
>   child prompt — copying it costs output-rate tokens and the copy rides
>   your history afterward.
> - Do NOT delegate questions you can answer directly, or work a one-line
>   command can do: a delegation costs a full provider round-trip.
> - Delegation trades money for time: the child round-trip is serialized
>   latency. Prefer doing it yourself when wall-clock matters.
> - Child prompts must be self-contained. The child sees only your prompt
>   and the referenced tool results — never this conversation — and has no
>   tools.

Plus, pending t-1345, a machine-generated catalog line:
`Available delegate models: <alias> ($in/$out per Mtok), ...` — until the
schema itself carries it, the guidance fragment is the natural interim
home for the registry's aliases and rates.

**Validation.** The t-1354 harness is the validator, as-is: re-record with
the shipped fragment as the `tool-guided` arm's system prompt (replacing
the handwritten text) and require the same behavioral profile — delegation
on generation-offload with correct child id and `context_refs`, zero
`OVER-DELEGATED` flags. Recording cost ~$0.14 per matrix run.

**Mechanism gaps guidance can't paper over:**

- **Delegate catalog in the infer schema (t-1345).** Guidance can name
  aliases, but the schema's bare-string `model` with no ids/rates is the
  root cause of unsafe unprompted delegation. Highest-leverage fix.
- **No budget knob on `InferPolicy`** — guidance can say "be economical"
  but cannot cap a child.
- **Serialized child latency** — guidance can only warn; removing it is
  Par (§3).
- **Process-child usage invisible to parent trace** (t-1354 candidate
  task) — subagent-as-process cost rollups need lineage plumbing, not
  prose.

### 2.2 Store/Retrieve discipline (save load-bearing intermediates)

**What.** Use `remember` to persist distilled, load-bearing intermediate
results *at the moment they are produced* — before GC pressure makes the
decision for you — and `recall` instead of re-deriving.

**When it pays.** Long sessions under GC. The default strategy (`stack`,
docs/GC.md) pops completed tool frames to one-line annotations: old tool
result *bodies go first* — anything needed verbatim from an old result is
gone. A fact that lives only in a popped result body must be re-computed
(paying tokens and latency again) or, worse, confabulated. There is a
write-barrier synergy on the recall side: `recall` results whose content
matches previously-seen or previously-collected content are recorded in
`GcState.recall_hot` (docs/GC.md, t-1351) — the signal the future
generational strategy consumes to retain recalled content preferentially.
Evict → recall → re-inject → evict thrashing is exactly what that
mechanism exists to stop; guidance that routes the model through `recall`
feeds it.

**Unguided failure mode.** No behavioral eval data yet (this is the
pattern most in need of one). Predicted from the tool descriptions: the
current `remember` spec says "worth keeping beyond this conversation" —
it frames memory as *cross-session* only, so a model has no reason to save
an *intra-session* intermediate, which is precisely what GC destroys.
Symptoms to score for: re-running commands whose output was evicted;
final answers contradicting evicted intermediates.

**Drafted guidance** (ships when a memory backend is registered):

> You have persistent memory via `remember` and `recall`.
>
> - Save load-bearing intermediate results as soon as you produce them:
>   decisions, distilled findings, computed values later steps depend on.
>   Your context window is managed — old tool output is evicted under
>   pressure, and a fact you did not save may be gone when you need it.
> - Save the distilled fact, not raw output: one or two sentences with the
>   concrete values in them.
> - When you need something you saw earlier and it is no longer in view,
>   `recall` it instead of re-running commands or guessing. Recalled
>   content is retained preferentially.
> - Anything worth keeping beyond this session — user preferences, project
>   conventions, decisions — always belongs in memory.

**Validation.** New behavioral eval (t-1354 methodology): long-horizon
fixtures where a fact computed early is needed after GC has run (`--gc
stack`, low budget to force pressure). Arms: memory tools advertised with
vs without the guidance block. Score from traces: final-answer
correctness, `StoreCall`/`RetrieveCall` counts and timing relative to
`gc_collect` events, re-executed `EvalCall`s (same command re-run =
missed-save), `recall_overlap_events`. Must coordinate with — not touch —
the GC eval surface (`gc_evals.rs`, `evals/gc/`).

**Mechanism gaps:**

- **"Retained preferentially" is today a signal, not a behavior**: no GC
  strategy consumes `recall_hot` yet (generational, t-1167, is its first
  customer). The drafted sentence is forward-true (recall re-injection
  puts content back in the live window regardless) but the *preferential
  retention* claim only fully lands with t-1167. Ship order matters.
- **The model can't see GC pressure** (see §2.4's gap): "before GC
  pressure" is undetectable from inside; the model can only adopt
  save-early discipline unconditionally.

### 2.3 Eval → Infer chaining (analyze big outputs by reference)

**What.** The composition of 2.1's by-reference rule with shell work:
when a command produces bulky output that is only needed distilled, do
not read it into your own tokens — chain the Eval's tool-call id into an
`infer` via `context_refs` and take back one line. Fetch + delegate can be
a single assistant turn (refs resolve within the same turn's tool batch,
t-1344).

**When it pays.** t-1342 `noisy-middle-step`: sub-infer wins 1.6x by
reference (was 3.3x *more expensive* by copy). The structural point:
referenced material never transits parent output tokens and stays in
parent history exactly once. What it does not fix: material already in
parent history is paid at parent input rates every later turn in both
arms — the carry tax is GC's territory, not delegation's.

**Unguided failure mode.** The model quotes or summarizes the dump in its
own assistant text (output-rate copy + permanent history residue), or
delegates by pasting. t-1354's guided arm demonstrated the positive
behavior (its one delegation used `context_refs` correctly, with a clean
self-contained child prompt) — but only with guidance present.

**Drafted guidance** (ships as bullets inside the 2.1 block — it is not a
separate fragment):

> - When a command produces bulky output you only need distilled (logs,
>   dumps, fetched documents), do not read it line-by-line into your
>   reply: call `infer` with `context_refs` naming that shell call's id
>   and ask for the distilled answer. You can issue the fetch and the
>   delegation in the same turn.

**Validation.** Behavioral arm over the t-1342 `noisy-middle-step` shape
(that harness is scripted; the fixture moves into the t-1354-style
behavioral matrix). Metric: did the model delegate by reference (infer
call whose `context_refs` name the fetch), by copy, or digest inline;
plus the cost columns.

**Mechanism gaps:** same as 2.1 (catalog, budget knob, serialization).

### 2.4 GC awareness (your context is a managed window)

**What.** Tell the model its context is not an append-only log: results
get popped to `[frame: ...]` annotations, durable facts belong in memory,
and (where the strategy supports it) citing a result by id protects it.

**When it pays.** Any session long enough to trigger collection.
Grounding: the t-1339 strategy matrix (stack retains 63–70% of messages,
pops result bodies to ~1% of their tokens) and the cited-keep fixture
(t-1351: a cited-but-semantically-distant result survives under
`semantic` + `cited-keep`, where semantic-only GC dropped it).

**Unguided failure mode.** The model assumes verbatim availability of
everything it ever saw: quotes from evicted bodies (confabulation risk),
re-runs expensive commands, or is confused by `[frame: ...]` annotations
it was never told about. No behavioral eval data yet; the GC behavioral
eval (SDK sessions, `evals/gc/`) is the natural home for these arms.

**Drafted guidance** (ships when GC is enabled — which is the CLI
default; the SDK's in-process `Runner` runs no GC and gets no block):

> Your context window is managed. In long sessions, old tool results are
> collapsed to one-line `[frame: ...]` annotations or dropped entirely.
>
> - Extract what matters from a result when you see it — into your reply,
>   or into memory with `remember`. Do not plan to re-read old output
>   verbatim later.
> - A `[frame: ...]` annotation means the result body is gone. If you
>   need it, re-run the command or `recall` the saved fact — do not guess
>   at what it contained.

And, **only** when `semantic` + `cited-keep` is the active strategy:

> - Referring to a tool call by its id in your text (for example, "per
>   the output of call-12") marks that result as load-bearing and
>   protects it from eviction.

**Validation.** Arms inside the GC behavioral eval (guidance
present/absent under forced pressure); metrics: confabulation needles
(claims about evicted content), redundant re-execution count, task
completion. Owned by the GC eval effort; this doc only specifies the
arms.

**Mechanism gaps:**

- **The citation-protection sentence is strategy-conditional.**
  `cited-keep` is implemented for `semantic` only; under the default
  `stack` strategy the sentence would be a false promise. Either the
  delivery mechanism supports strategy-conditional blocks, or cited-keep
  lands for `stack` (listed as future work in docs/GC.md) before the
  sentence ships. Guidance must never claim protection the active
  strategy does not implement.
- **GC pressure is invisible to the model.** There is no in-window signal
  that a collection is imminent (the annotations only appear after the
  fact). A one-line injected notice ("context at 85%; N results were
  collapsed") would let save-before-eviction be *reactive* instead of
  unconditional. Candidate mechanism task.

### 2.5 Cost awareness (economy as a stance; usage surfacing as a gap)

**What.** A standing instruction to prefer the cheapest correct path, and
— once the mechanism exists — a live usage line so "cheapest" can be
quantitative.

**When it pays.** Everywhere, weakly; t-1354's restraint result is the
evidence that economic framing generalizes (the guided arm answered the
direct question in one turn with *no* tool use — "do NOT delegate direct
questions" taught economy, not just delegation policy). Turn count is a
real cost driver (every turn re-sends history at input rates), so
batching independent tool calls into one turn pays today even though the
runtime executes them sequentially — and that same batching is the demand
signal for Par (§3).

**Unguided failure mode.** Shell-as-calculator (t-1354 baseline arms),
re-fetching what is already in context, one tool call per turn where
three would batch.

**Drafted guidance** (unconditional — no capability gate):

> Be economical. Use the fewest steps that complete the task correctly. A
> one-line command beats delegation; delegating long generation to a
> cheap model beats writing it yourself; nothing beats not doing the work
> twice — do not re-fetch or re-compute what you already have. When
> several tool calls do not depend on each other, issue them together in
> a single turn.

**Validation.** Cheap piggyback: every behavioral matrix already scores
cost/turns/tokens per cell (RunUsage on `AgentDone`, t-1334). Arm =
economy block present/absent across the existing t-1354 fixtures; look
for turn-count and cost deltas with unchanged correctness.

**Mechanism gaps:**

- **Running usage is not surfaced to the model.** The trace has
  everything — per-`InferResult` `cost_micro_usd`, the `RunUsage` rollup
  (docs/TRACE_SCHEMA.md, t-1334) — but nothing injects "you have spent
  $0.09; the context window is 62% full" into the model's view, and no
  tool exposes it. Guidance can therefore only be qualitative today.
  Surfacing a one-line running usage/budget stanza (per turn, or as a
  `usage` tool) is a candidate task; the cost-awareness block should get
  a quantitative second paragraph when it lands, A/B'd separately.
- **No budget the model could spend against**: there is no per-run dollar
  budget concept to report a fraction of. Same task.

### 2.6 Approval awareness (a pause is not a failure)

**What.** Tell the model that gated effects (an Eval with
`require_approval`, a Store to a `RequireApproval` sink) pause durably
for a human, and that denial is an answer to respect — not an error to
route around.

**When it pays.** Any run with gates configured (DR-7, t-1308.10,
`approval.rs`). The protocol already does the right mechanical things —
fails closed, checkpoints mid-turn, binds denial as a typed value
(errors-as-values, t-1222) so the model can read and react — but the
model has never been told the semantics of what it sees.

**Unguided failure mode.** No eval data yet; predicted failure modes,
each scoreable: treating a denial value as a transient error and retrying
the same effect (possibly re-triggering the gate in a loop); "working
around" a denial by attempting a semantically equivalent effect the human
just declined (the adversarial version of retry — worth scoring
explicitly because it is a trust failure, not an efficiency failure);
treating an in-hook denial as run failure and aborting a completable
task.

**Drafted guidance** (ships when any effect in the run's config is
gated):

> Some actions require human approval before they run. When you request
> one, the run may pause — possibly for a long time — and resumes when a
> person decides. A pause is not an error: do not retry or rephrase a
> pending action.
>
> If an action is denied, you will see a denial result. Respect it: do
> not re-attempt that action or an equivalent of it. Pursue an
> alternative, or report clearly what you could not do and why.

**Validation.** Cheapest eval in the plan: the SDK's in-process
`on_approval` hook scripts deterministic denials offline (no human, no
network for the gate itself). Fixtures: a task completable despite a
denied step; a task not completable, where the right answer is a clear
report. Arms: block present/absent. Metrics: repeat-attempt count of the
denied effect (and near-equivalents), task outcome, final-answer honesty
needles.

**Mechanism gaps:**

- **Gates are invisible until hit.** Nothing marks a tool or command
  class as "gated" in the model-visible surface, so the model cannot plan
  around approval latency (e.g., front-load gated requests, batch
  non-gated work while waiting). Advertising gated-ness (in the tool
  description or the fragment) is a candidate task — today the guidance
  says "some actions" because the runtime cannot say which.
- **No pending-approvals introspection for the model**: `agent approvals`
  exists for humans; the model cannot ask "what am I waiting on."

### Catalog summary

| pattern | one-line guidance | validation status |
|---|---|---|
| Delegation | Delegate generation-heavy/bulky-digest subtasks via `infer` + `context_refs`; never direct questions; costs latency | **Validated** (t-1354: 2.26x, zero over-delegation); re-record with shipped text |
| Store/Retrieve discipline | Save distilled load-bearing intermediates via `remember` as produced; `recall` instead of re-deriving | Not yet evaluated; new behavioral eval specified above |
| Eval→Infer chaining | Digest bulky command output by reference (`context_refs`), never by re-reading into your own tokens | Mechanics validated (t-1342, 1.6x); behavioral arm pending |
| GC awareness | Context is a managed window; extract-or-`remember` on sight; `[frame: ...]` = body gone | Not yet evaluated; arms belong to the GC behavioral eval |
| Cost awareness | Fewest steps; cheapest correct path; batch independent tool calls in one turn | Partially evidenced (t-1354 restraint generalization); explicit A/B pending |
| Approval awareness | Gated effects pause durably; denial is an answer, not an error — never re-attempt | Not yet evaluated; offline-scriptable via SDK `on_approval` |

## 3. Appendix: Par requirements, demand-first

Par is listed in docs/AGENT_IR.md as designed-but-rejected at runtime
until its open questions settle. Rather than settle them in the abstract,
derive them from the fan-out patterns this document creates demand for.

### The demand

1. **The model's own turn batch.** The tool loop already lets one
   assistant turn carry several tool calls (a `[shell fetch, infer(refs)]`
   batch resolves within the turn, t-1344), and §2.5's guidance actively
   tells models to batch independent calls. Today the dispatch arm
   executes them **sequentially**. This is the highest-volume fan-out site
   in the system, and its width is dynamic (however many calls the model
   issued).
2. **Parallel sub-infers over independent inputs.** Map-shaped
   delegation: score N candidates, summarize N documents — each child
   independent. t-1354 measured the cost of serialization directly: the
   guided generation-offload cell was 2.26x *cheaper* but **19s slower**
   than baseline (37.9s vs 19.0s) because one child round-trip is
   serialized latency; N children serialize N round-trips. "Delegation
   trades money for latency; …or the loop needs parallel sub-infer
   dispatch" is a t-1354 candidate-task verbatim. Width: dynamic (N is
   data).
3. **Parallel Evals.** Fetch three docs concurrently (the doc-synthesis
   shape); independent read-only probes. Width: dynamic.

Every concrete demand is **dynamic-width, effect-only fan-out**
(Infer/Eval/Let in the branches; no Store, no cross-branch communication).
Note the static `Terminator::Par { branches: Vec<BlockId>, join }` in the
IR sketch cannot express any of the three — the width is runtime data. So
the demand-first conclusion up front: the minimal Par is a **dynamic-width
map** (`ParMap { over, body, join }`-shaped: one body block applied per
element, branch index = element index), of which fixed-width Par is the
degenerate special case.

### Requirements derived

**Join semantics: all-of, declaration order.** Every demand pattern needs
every result (the turn batch must answer every tool call; a map needs
every element). Join = wait for all branches; results bind at the join in
**branch declaration order** (element order), never completion order —
determinism must not depend on scheduling. `any`/`first-success` has no
current demand (racing identical infers is a cost anti-pattern, and no
fixture wants it): defer, and note the effect-id scheme does not preclude
it later.

**Failure propagation: bind-as-value, no cancellation.** Per the
errors-as-values convention (t-1222): a branch whose effect fails with
`on_error: Bind` produces its error value in its slot and the join always
runs — this is exactly what the tool loop's dispatches already do, so
demand pattern 1 requires it. For an `Abort`-polarity effect inside a
branch: the branch's slot carries the error, siblings **run to
completion**, and the abort propagates *after* the join settles, with the
first-by-declaration-order error chosen as the propagated one
(deterministic). No sibling cancellation in v1: cancellation makes the
canceled branch's partially-issued effects ill-defined in the trace and
under replay, and no demand pattern needs it.

**Trace ordering + effect identity.** Identity is the settled part
(docs/AGENT_IR.md): branch `b` forks the parent path digest with
`arm = b`, so sibling effect ids are deterministic and
scheduling-independent — for ParMap, `arm` = element index. What the
interpreter must additionally guarantee:

- **Ids assigned at fork, not at completion**: each branch's dynamic path
  (and any per-site visit ordinals inside it) derive from the fork point
  before any branch is scheduled, so ids never depend on which branch ran
  first.
- **Serialized trace appends**: one appender; sibling events MAY
  interleave in file order, and consumers must not infer causality from
  cross-branch adjacency. Every effect event already carries its effect
  id and `parent_op_id` lineage; that — not position — is the join key.
- **Order-independent replay**: replay feeds recorded results by effect
  id and must not assert cross-branch event order; divergence detection
  is per-branch (an edited branch body diverges alone). Recorded-result
  lookup is already id-keyed, so this is a constraint on the matcher, not
  a redesign.
- **Deterministic join event**: a single join-completion point after all
  branches settle, so the post-join path digest (`arm = branch_count` per
  the settled scheme) folds at a deterministic moment.

**Store isolation: answered by demand — reject it.** No demand pattern
writes inside a branch. v1: `validate_program` rejects `Store` (and
gated-`Eval`-with-approval? — see below) inside Par bodies; `Retrieve` is
read-only and allowed. This dissolves the "isolated vs shared
transactions / merge at join" open questions for v1 instead of answering
them speculatively. Revisit when a real program wants a writing branch.

**Approval gates inside branches: defer.** A branch that pauses durably
mid-Par forces the checkpoint to capture sibling in-flight state — real
machinery for a case with no demand. v1 rejects approval-gated effects
inside Par bodies at validation; the turn-batch site only goes concurrent
when no call in the batch is gated (gated batches fall back to the
sequential arm).

**Budget semantics: pre-split, return at join.** Token/cost accounting
(`RunUsage`) is an additive rollup over trace events — order-independent,
nothing to change. The step/instruction budget is the shared resource: at
fork, check `remaining >= width`, give each branch
`floor(remaining / width)` (remainder to earlier branches —
deterministic), and return unused allocation at the join. Pre-splitting
keeps "which branch hit the limit" independent of scheduling, which a
shared atomic pool would not. A branch exhausting its allocation fails as
that branch's error value (bind semantics above). Parallel children never
consume parent *turns*: the whole Par is inside one turn.

### Recommendation: the minimal Par worth implementing

Dynamic-width map-Par with: join-all in declaration order,
errors-as-values per branch, no cancellation, no Store and no approval
gates in bodies (validation-rejected), ids forked at dispatch per the
settled scheme, serialized trace appends with id-keyed order-independent
replay, pre-split step budgets. **First customer: the loop program's own
tool-dispatch arm** — the model's turn batch runs concurrently, which (a)
immediately monetizes the §2.5 batching guidance, (b) removes the
serialized-latency penalty that is delegation's measured drawback, and
(c) exercises Par on every multi-tool turn of every session, giving the
semantics real mileage before user programs write `Par` by hand. The
existing scaffold test (`par_branches_fork_the_control_path`) plus the
t-1354 recording (whose generation-offload cell would show wall-clock
parity with baseline under parallel dispatch) are the acceptance
anchors.

## 4. Delivery design (recommendation, not implementation)

### Where guidance can live

Three candidate homes, with the current state of each:

1. **The default system prompt.** Today: three sentences in
   `base_system_prompt()` (crates/agent/src/main.rs) mentioning only the
   shell tool — and `--system-prompt`/prompt-file frontmatter **replaces**
   it wholesale, so any user override silently deletes even that. The SDK
   has no default at all (`Agent.instructions` is optional and empty).
   A static string cannot be conditional on capabilities, and
   replace-not-compose semantics mean the runtime cannot rely on it
   surviving.
2. **Per-tool descriptions** (`ir_tool_specs`/`base_ir_tool_specs`,
   ir_interpreter.rs). Today: one to two terse sentences each.
   Descriptions are automatically conditional (no tool, no text), survive
   any user system-prompt override, ride the prefix-stable cached region,
   and are where mature harnesses put the bulk of per-tool operational
   text (§6). Their limits: they cannot carry *cross-tool* workflow
   ("save before GC evicts", "batch independent calls", "a pause is not
   an error"), and they are invisible when no tool is involved.
3. **A runtime PromptIR section with instruction authority** (the t-1297
   connection). PromptIR already has the vocabulary
   (docs/PROMPT_IR.md): a `Section` with `SectionRole::Developer`,
   `CompositionMode::Constraint`, `Priority::High`,
   `SectionOrigin::Static { name: "runtime-guidance" }`. That placement
   makes guidance **instruction-authority content, structurally distinct
   from retrieved data**: it compiles into the
   Hierarchical→Constraint→Additive→Contextual order ahead of hydrated
   context, its hash and section summary ride every `InferCall` trace
   event (auditable: you can prove which guidance version a model saw),
   and retrieved content — memory hits, tool output — can never
   masquerade as it. This is the same authority boundary that makes
   memory-poisoning containable (docs/MEMORY.md's injection concerns):
   data sections do not get to speak with runtime authority.

### Recommendation: split by scope, deliver both

- **Per-tool operational text → the tool description.** Each tool's
  when/when-not/how (the 2.1 delegation bullets, the 2.2 memory bullets,
  2.3's chaining bullet) moves into its `ToolSpec` description. This is
  step one because it needs no new mechanism, benefits every consumer
  (CLI and SDK, any system prompt), and is the industry-converged
  location (§6). The t-1345 catalog lands here too when it exists.
- **Cross-tool workflow text → a runtime guidance fragment, assembled at
  prompt build from live capabilities, delivered as the PromptIR section
  above.** The GC block (2.4), cost/batching block (2.5), approval block
  (2.6), and the one-line cross-references that tie tools together
  ("bulky output → delegate by reference") compose into one fragment.
  Assembly is conditional: each block keyed to its capability (GC
  enabled, gates configured, memory registered, infer advertised —
  including strategy-conditional lines like cited-keep's). The fragment
  text is versioned in-repo; its content hash is visible per `InferCall`
  via the existing PromptIR trace metadata.

**Composability contract.** SDK `instructions` and CLI
`--system-prompt` remain the user's — rendered as the System section,
untouched. The runtime fragment is a *separate, clearly delimited*
Developer/Constraint section the user does not have to know about to
write their prompt, and user text can override stance where they
genuinely conflict (user instructions are the higher authority tier;
guidance is operational, not policy). Opt-out is explicit and total:
`--no-runtime-guidance` / SDK `.runtime_guidance(false)` — for
deterministic evals and for users who ship their own manual. This fixes
the current failure mode where overriding the system prompt silently
deletes the runtime's only operational text.

**Migration path.**

1. Fatten tool descriptions with the per-tool drafted text (2.1–2.3).
   No new mechanism; A/B through the existing t-1354 harness (a
   description-only arm is a new row, not a new harness).
   **Shipped (t-1359)** — exact text recorded in "Shipped per-tool
   descriptions" below.
2. Introduce the fragment mechanism: capability-keyed assembly →
   PromptIR Developer/Constraint section; flag + SDK toggle; fragment
   hash in traces. Ship with only eval-validated blocks default-on
   (initially: delegation cross-reference + cost/batching if its A/B
   clears; see §5).
3. Move the CLI's `base_system_prompt()` operational sentences into the
   fragment, so `--system-prompt` composes instead of destroys.
   Per-block promotion thereafter follows §5's gate.

### Shipped per-tool descriptions (t-1359)

The step-1 text as it ships, from `crates/agent-core/src/guidance.rs`
(`ir_tool_specs` consumes these constants). The catalog's drafted bullets
are condensed to schema-description prose here; this record and the
constants are the same text by contract — edit both in the same commit.

`infer` (§2.1 + §2.3):

> Delegate a subtask to another (usually cheaper) model and return its
> response. Delegate when the subtask is generation-heavy (long
> boilerplate, many candidates to write up) or requires digesting bulky
> material you have already fetched — pass that material by reference via
> context_refs; never paste large tool output into the prompt (copying it
> costs output-rate tokens and the copy rides your history afterward). Do
> NOT delegate questions you can answer directly, or work a one-line
> command can do: a delegation costs a full provider round-trip of
> serialized latency. Child prompts must be self-contained — the child
> sees only your prompt and the referenced tool results, never this
> conversation, and has no tools.

`infer.context_refs` (§2.3):

> ids of prior tool calls from this conversation (e.g. a shell call's id):
> each referenced call's result is delivered to the sub-model directly,
> ahead of the prompt, without being repeated here. Use this for bulky
> output you only need distilled — you can issue the fetch and the
> delegation in the same turn.

`remember` (§2.2 — replaces the cross-session-only framing §2.2 identifies
as the unguided failure mode):

> Save a distilled fact to persistent memory. Save load-bearing
> intermediate results as soon as you produce them — decisions, distilled
> findings, computed values later steps depend on — because old tool
> output may be evicted from your context and a fact you did not save may
> be gone when you need it. Save the distilled fact, not raw output: one
> or two sentences with the concrete values in them. Anything worth
> keeping beyond this session — user preferences, project conventions,
> decisions — also belongs here.

`recall` (§2.2):

> Search persistent memory by keywords and return matching notes. When you
> need something you saw earlier and it is no longer in view, recall it
> instead of re-running commands or guessing.

None of this text enters the program hash or effect identity (tool
descriptions are provider-offer material assembled from config at
dispatch, not program content), so wording is tunable without breaking
replay of old traces — pinned by
`guidance::tests::tool_descriptions_never_enter_the_program_or_its_hash`.

## 5. Eval plan

**Methodology** (inherited from t-1354): arm = guidance section present vs
absent, same task text, same model, same advertised toolset; behavior is
scored from traces only (RunUsage, per-effect events, lineage), never
estimated; appropriateness flags are scored, never asserted; offline
replay of recorded cells must reproduce the table exactly; no hand-written
behavioral recordings. Single-sample-per-cell and provider-default
temperature remain the standing caveats (the temperature-control gap is
already filed from t-1354).

**Order of attack** (by evidence-per-dollar):

1. **Delegation fragment, shipped text** — re-record the t-1354 matrix
   with the §2.1 drafted text (as tool description + fragment, per §4
   step 1) replacing the handwritten guided arm. Success: same behavioral
   profile as the original guided arm (delegation on generation-offload
   with `context_refs`, zero over-delegation). Cost: the original
   recording run was **$0.14** for the whole matrix + probe
   (haiku-4.5 parent / gpt-4o-mini child via OpenRouter); this is one
   more matrix, same order: **~$0.15**.
2. **Cost/batching block** — piggybacks on the same matrix (one more arm
   column over the existing five fixtures: +5 cells, **~$0.05**). Metric:
   turn count and cost deltas at unchanged correctness; batched
   multi-call turns counted from trace.
3. **Approval block** — mostly offline (SDK `on_approval` scripted
   denial); one online recording pass over 2–3 fixtures x 2 arms:
   **~$0.05**. Metric: denied-effect re-attempt count (including
   near-equivalent commands), task outcome, honesty needles.
4. **Eval→Infer chaining** — behavioral arms over the noisy-middle shape
   (2 fixtures x 2 arms + probe): **~$0.05–0.10**. Metric: by-reference
   vs by-copy vs inline, from `context_refs` presence in InferCall args.
5. **Memory discipline + GC awareness** — the expensive one: fixtures
   must be long enough to trigger real GC, so cells cost more
   (**~$0.30–0.50 per recording pass**, estimated at 3–5x t-1354's
   per-cell spend). Runs as arms within the GC behavioral eval surface
   (`evals/gc/`, SDK sessions) — specified here, owned there. Deferred
   until that harness exists; do not build a parallel one.

New behavioral cells for 2–4 live in a new harness (`evals/guidance/`,
t-1354 pattern: recordings checked in, credential-free, offline replay
asserted) rather than growing the delegation matrix indefinitely; the
delegation re-record (1) stays in `evals/delegation/`.

**Promotion gate.** A guidance block ships **default-on only after** its
A/B shows, on recorded runs: (a) the target behavior moved on the
should-help fixtures (the block's reason to exist, quantified); (b) zero
regressions on restraint/neutral fixtures (no over-delegation, no
gratuitous tool use, correctness unchanged); (c) offline replay
reproduces the recorded table. Until then it exists behind the delivery
flag, default-off. The shipped text is frozen with its recording — any
text edit invalidates the validation and requires a re-record (cheap by
construction: every run above is well under a dollar). Each default-on
block cites its recording in this document, the same way GC strategy
promotion cites the t-1339 matrix.

## 6. Comparative grounding: how mature harnesses do this

The three harnesses studied converge on the architecture §4 recommends —
guidance is harness-owned, split between tool descriptions and a runtime
prompt layer, composing with (not displaced by) user instructions:

- **Claude Code** distributes guidance across three layers [1][2]. (1) A
  dynamically assembled system prompt, built *conditionally per session*
  (output styles, MCP guidance, context-management warnings such as "old
  tool results will be automatically cleared from context") — the
  capability-keyed assembly of §4, in production. It carries the
  cross-tool workflow, e.g. batching: "When you launch multiple agents
  for independent work, send them in a single message with multiple tool
  uses so they run concurrently." (2) Tool descriptions that are long
  operational documents, not one-liners: the subagent tool's description
  has a "## When to use" section framing delegation as context economy
  ("delegate it and you keep the conclusion, not the file dumps") with
  explicit anti-triggers ("For a single-fact lookup where you already
  know the file … search directly"). (3) Runtime-injected
  `<system-reminder>` tags for just-in-time guidance, which the prompt
  explicitly marks as harness-injected, not user content — an authority
  separation in the t-1297 spirit.
- **OpenAI Agents SDK** ships almost no prose of its own, but its one
  shipped fragment is exactly this document's subject:
  `agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX` [3], an
  opt-in constant explaining the delegation mechanism to the model ("You
  are part of a multi-agent system … Handoffs are achieved by calling a
  handoff function, generally named `transfer_to_<agent_name>` …"),
  composed ahead of developer instructions via
  `prompt_with_handoff_instructions`. The docs state the t-1354 lesson as
  a recommendation: "To make sure that LLMs understand handoffs properly,
  we recommend including information about handoffs in your agents" [4].
  Tool descriptions come from function docstrings [5] — description
  authoring as a first-class developer task.
- **Claude Agent SDK** makes the harness prompt a reusable preset:
  `systemPrompt: { preset: "claude_code", append: ... }` layers user
  rules on top of the full operational prompt ("nothing is removed"),
  while a fully custom string means "you take responsibility for
  replacing the tool guidance … your agent still needs" [6] — i.e. the
  guidance layer is acknowledged as load-bearing runtime surface, opt-out
  at your own risk (§4's `--no-runtime-guidance` contract). CLAUDE.md
  project content is injected into the conversation, *not* the system
  prompt — instruction-authority kept distinct from project data.
- **Anthropic's tool/context guidance** states the general principle:
  "Provide extremely detailed descriptions. This is by far the most
  important factor in tool performance … when it should be used (and
  when it shouldn't) … at least 3-4 sentences per tool description" [7],
  "even small refinements to tool descriptions can yield dramatic
  improvements" [8]; retrieve by "lightweight identifiers (file paths,
  stored queries, web links)" rather than payloads [9] — the
  `context_refs` design is this advice, mechanized.

Convergent conventions adopted in §2's drafts: both polarities always
("when to use / when NOT to use" — the restraint half is what t-1354
proved matters); delegation justified as context/token economy, not just
parallelism; restraint and batching as standing instructions; shipped,
composable prompt fragments as API surface rather than documentation.

Citations:

1. dbreunig, "How Claude Code Builds a System Prompt" —
   <https://www.dbreunig.com/2026/04/04/how-claude-code-builds-a-system-prompt.html>
2. Published Claude Code system prompt (asgeirtj/system_prompts_leaks,
   claude-code-2.1.172) —
   <https://github.com/asgeirtj/system_prompts_leaks>
3. OpenAI Agents SDK, handoff prompt reference —
   <https://openai.github.io/openai-agents-python/ref/extensions/handoff_prompt/>
4. OpenAI Agents SDK, handoffs guide —
   <https://openai.github.io/openai-agents-python/handoffs/>
5. OpenAI Agents SDK, tools guide —
   <https://openai.github.io/openai-agents-python/tools/>
6. Claude Agent SDK, modifying system prompts —
   <https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts>
7. Anthropic, tool-use implementation docs —
   <https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use>
8. Anthropic engineering, "Writing effective tools for agents" —
   <https://www.anthropic.com/engineering/writing-tools-for-agents>
9. Anthropic engineering, "Effective context engineering for AI agents" —
   <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>

## Candidate tasks (mechanism gaps, consolidated)

Gaps guidance cannot paper over, surfaced per-pattern above:

1. **Delegate catalog in the infer schema** — already filed as t-1345
   (root cause of unsafe unprompted delegation; §2.1).
2. **Running usage/budget surfacing to the model** — RunUsage exists in
   traces only; no injected usage line, no `usage` tool, no budget
   concept to spend against (§2.5). Blocks quantitative cost guidance.
3. **Minimal Par: concurrent turn-batch dispatch** — dynamic-width
   map-Par per §3, first customer the tool-dispatch arm; removes
   delegation's serialized-latency penalty (t-1354 candidate task,
   sharpened here into requirements).
4. **Guidance delivery mechanism** — capability-keyed runtime fragment as
   a PromptIR Developer/Constraint section + tool-description fattening +
   compose-not-replace system prompt (§4). The implementation task this
   design doc feeds.
5. **GC pressure visibility** — a one-line in-window notice on collection
   (§2.4), enabling reactive save-before-eviction.
6. **Strategy-honest citation guidance** — either cited-keep for `stack`
   (GC.md future work) or strategy-conditional fragment lines (§2.4).
7. **Gated-effect advertisement + pending-approval introspection for the
   model** (§2.6).
8. **`recall_hot` consumer** — already filed as t-1167 (generational GC);
   noted because §2.2's "retained preferentially" sentence only fully
   lands with it.
9. Carried from t-1354, still open: child-process usage lineage in parent
   traces; provider-layer temperature control for multi-sample evals.
