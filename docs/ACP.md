# ACP: driving agentd from Paseo (and other ACP clients)

`agent --acp` serves the [Agent Client Protocol](https://agentclientprotocol.com)
over stdio: newline-delimited JSON-RPC, stdout reserved for protocol frames,
diagnostics on stderr. Any ACP client can spawn the binary and drive
sessions; [Paseo](https://github.com/getpaseo/paseo) is the primary target —
its desktop and mobile apps give agentd sessions a UI (including phone
access) without agentd growing a server, port, or dashboard.

## Paseo setup

Add agentd as a custom provider in `~/.paseo/config.json`:

```json
{
  "agents": {
    "providers": {
      "agentd": {
        "extends": "acp",
        "label": "agentd",
        "command": ["agent", "--acp"],
        "params": {
          "supportsMcpServers": false
        }
      }
    }
  }
}
```

`params.supportsMcpServers: false` stops Paseo injecting its internal MCP
server into agentd sessions (see Paseo's `docs/custom-providers.md`) —
agentd does not consume MCP servers yet, so without it Paseo's shared
tools would be offered and silently dropped.

Process-level flags become per-session defaults, so variants like
`["agent", "--acp", "--model", "opus", "--memory-dir", "/home/you/notes"]`
work as separate provider entries.

**MCP servers are not passed through (v1).** agentd advertises no MCP
capabilities in `initialize`, and any servers a client offers in
`session/new`/`session/load` are ignored with a stderr warning — which is
why the config above sets `supportsMcpServers: false`. Tools Paseo
normally provides through its injected MCP server (e.g. subagent
creation) are unavailable in agentd sessions until MCP passthrough lands.

Authentication is agentd's normal provider auth, out of band: API keys via
`models.yaml`/env, or subscription OAuth via `agent auth login claude-code`
(or `codex`) before pointing Paseo at the binary. The ACP `initialize`
advertises no auth methods because there is nothing interactive to do over
the wire.

## What is implemented

- `initialize` — protocol v1, text and image prompts, `loadSession: true`, plus `sessionCapabilities.list` and `.close`.
- `session/new` — one fresh `Runtime` (agent loop, trace, checkpoints) per
  session. The request `cwd` becomes the shell tool's working directory and
  the system prompt's cwd line. Session state lives under
  `~/.local/share/agent/acp/<session-id>/`.
- `session/prompt` — runs one agent-loop turn. Assistant text streams live
  as `agent_message_chunk` updates (see Streaming below); shell, memory,
  and native tool effects surface as `tool_call` / `tool_call_update` with
  full untruncated payloads in `rawInput`/`rawOutput`.
  A delegation through the `infer` tool is one `tool_call` card (kind
  `other`, titled `infer (<model>): <task first line>`): the child
  Infer's call opens it, and its result completes it with the child's
  answer as card content (or fails it with the error). The child's text
  never reaches the assistant reply, and its deltas are not streamed — the
  answer lands in the card when the child finishes. Child Infers are
  recognized by trace lineage (`parent_op_id`); a child dispatched after
  an approval resume in the same turn has no lineage and renders as
  assistant text.
- Slash commands — `session/new` and `session/load` immediately send
  `available_commands_update` for `/cost`, `/gc`, and `/help`. A standalone
  command is handled locally with an `agent_message_chunk` and `end_turn`,
  without inference, user-history mutation, checkpoint, or trace event;
  leaving the replay trace unchanged avoids shifting effect visits. Unknown
  commands pass through to the model. `/cost` reads the session trace's
  per-model cost rollups (including child infers; unpriced calls are marked),
  rather than the live `RunUsage` which has no model breakdown on load.
  `/gc` shows the selected strategy, threshold, latest root-context usage
  (unknown before inference), model ceiling, and trace `gc_collect` count.
  `/fork`, `/checkpoint`, `/replay`, `/zoom`, and per-mode commands remain
  follow-ups.
- `session/request_permission` — shell commands are approval-gated by
  default in ACP mode: each gated command round-trips to the client with
  allow-once / allow-always / reject-once options. A rejection is a typed
  denial the model sees and recovers from — the turn continues. Disable
  gating with `--acp-shell-approval=false` or per session via the `yolo`
  mode. Approvals resolve fully in-process (no `agent approvals` records).
- `ask_user` — ACP sessions offer the model a native multiple-choice
  question tool, `ask_user {question, options}` (2-8 distinct non-empty
  options). CLI one-shot/FIFO/stdin sessions and the `agentd` supervisor do
  not offer it: they have no way to answer. See "Questions" below.
- `todo` — opt in with `--todo` (also works outside ACP), or enable it
  per ACP session by selecting a profile with `tools: [todo]`. The model sends
  `{ "items": [{ "content": "...", "status": "pending" | "in_progress" | "completed" }] }`;
  each successful call replaces the entire checklist and emits an ACP `plan`
  update (medium priority). Paseo renders `content` and checks only completed
  items. An empty array clears the list; blank content, invalid statuses, and
  more than one in-progress step produce a recoverable tool error, not a pause.
  Calls also render as ordinary completed tool cards, retaining the model's
  action in the timeline. On `session/load`, the last successful list is
  rebuilt from traced tool results (including a cleared list).
- `session/cancel` — drops the in-flight turn (closing any provider
  stream) and answers the prompt with `stopReason: "cancelled"`. The
  session survives; history keeps the user message. Note: an already
  in-flight shell command is not killed synchronously — it runs to its
  timeout in the background (a cancel-token in agent-core is the planned
  tightening).
- `session/list` — pages (50 at a time, newest first) over disk-backed ACP
  checkpoints, optionally filtering by exact cwd. Titles come from the first
  user prompt's first line (60 characters); a `session_info_update` announces
  the title when the first prompt arrives. Listing builds no runtimes. Paseo
  loads each listed session to collect prompt previews; `session/close` drops
  the live actor and in-flight turn but retains the checkpoint for later load.
  Sessions predating the cwd metadata index are not listed until loaded.
- `session/load` — rebuilds the runtime from the session's
  `session-latest.json` checkpoint and replays the conversation as
  user/agent message chunks and tool cards in trace order before responding.
  The trace uses the live event mapping (including zoom and resolved approval
  cards); ACP-only turn-start markers link effects to saved prompts without
  changing CLI traces. Missing, truncated, or unalignable traces fall back to
  text-only checkpoint history; plans stop at the loaded checkpoint marker.
- Modes — `ask` (default; approval-gated shell) and `yolo` remain built in.
  Additional modes load from `--acp-modes-dir DIR`, or from
  `$XDG_CONFIG_HOME/agent/modes` (`~/.config/agent/modes`) if it exists.
  Each `.md` file has YAML frontmatter and instructions in its body; the
  filename stem is its id. For example, `examples/modes/research.md`:
  `name` defaults to the id, `description` is optional, and
  `shell_approval` is `ask` (default; honors allow-always), `always`
  (prompt for every command; never remember allow-always), or `never`
  (ungated). `tools` is an optional list of opt-in tool names; currently
  none are registered (future tools will extend the registry). Unknown
  names, invalid YAML/policy, and built-in id collisions fail startup.
  Mode instructions are turn-local system context, not saved conversation
  history; their profile id and text also enter the loop program identity.
  `session/set_mode` takes effect next turn, persists across `session/load`,
  and emits `current_mode_update`. Loading a session whose profile was removed
  warns on stderr and falls back to built-in `ask`.
- Plan profile — copy `examples/modes/plan.md` to
  `~/.config/agent/modes/plan.md` (create the directory first), then restart
  `agent --acp`. Paseo recognizes the `plan` id as its Plan toggle; Off
  selects `ask`. The profile offers `todo`, asks permission for every shell
  command, and instructs read-only investigation. The model asks the user
  whether to implement or revise; on Implement it requests a manual switch
  to `ask` before any edits. Agent-initiated, consent-gated mode switching
  remains a follow-up; no answer text is treated as authority to switch modes.
- Config options — rendered by Paseo next to the mode picker, applied
  between turns via `session/set_config_option`:
  - `model`: every `models.yaml` alias; switching re-resolves the
    provider, context budget, and pricing in place.
  - `gc`: the context-GC strategy (`none`/`ring`/`mark-sweep`/`stack`/
    `semantic`/`generational`), defaulting to the `--gc` flag. Safe to
    switch mid-session — GC is not part of program identity, and the
    persistent GC state is strategy-agnostic. The process-level `--gc-*`
    knobs (cache policy, windows, floors) carry into the new strategy;
    switching to `none` is rejected when `--gc-timing` needs a strategy.
  - `gc-threshold`: the collection trigger as a fraction of the context
    budget (discrete steps; defaults to `--gc-threshold`).

## Questions (`ask_user`)

The model calls `ask_user` when a decision needs the user. The runtime
treats the call as an approval-gated effect of kind `question`: the turn
pauses exactly like a gated shell command (same checkpoint, same
`AwaitingApproval` → resolution → resume path), and the request payload is
`{question, options}`.

On the wire, the pause becomes:

- a `tool_call` (status `pending`) titled with the question, with the
  question as its text content;
- a `session/request_permission` with one `allow_once` option per answer
  (`optionId` = the answer's index, `name` = its text) plus one
  `reject_once` option, `dismiss`.

Paseo (verified against 0.9.2) treats a request with two or more
same-kind allow options as a chooser: it renders the tool call's text
content as the prompt, one button per answer, and the reject option as a
deny button. Paseo never auto-accepts a chooser, so the question always
reaches the user. Paseo has no free-text answer path for generic ACP
agents, so answers are always one of the offered options.

Outcomes:

- **selected** — the chosen option's text is the tool result the model
  reads; the tool call completes with the answer as its content.
- **dismiss** — the tool result is
  `{"ok": false, "error": "the user dismissed the question without choosing an option", ...}`;
  the turn continues and the model decides what to do.
- **cancelled** — the turn ends with `stopReason: "cancelled"`, as for a
  cancelled shell approval.

The `yolo` mode and "allow for this session" silence shell approvals
only; questions always prompt. Invalid arguments (missing question, fewer
than 2 or more than 8 options, duplicates) never pause: the model gets a
tool error and can retry.

Replay: the question's `ApprovalRequested`/`ApprovalResolved` events and
the answer's `ToolCall`/`ToolResult` events land in the session trace, so
IR replay of that trace reproduces an answered turn without asking again
(pinned by agent-core's `ask_user_turn_replays_from_the_trace`).
Offering the tool adds a dispatch arm to the loop program, so ACP turns
hash differently from CLI turns; the CLI's program is unchanged.

## Streaming

Assistant text streams token-by-token when the provider path supports SSE:
the OpenAI-compatible client (OpenRouter etc.), the native Anthropic
`/v1/messages` API, and both OAuth paths (claude-code via the
OpenAI-compatible endpoint, codex via the Responses API). Deltas are a live
UI side-channel only — traces, checkpoints, and replay are byte-identical
to non-streamed runs, and replayed sessions fall back to whole-message
chunks.

Known wart: if a stream dies mid-response and the retry succeeds, the
client may briefly see duplicated text; the final whole-message chunk
carries the authoritative content.

Context-window usage is reported through ACP `usage_update` notifications. The
`used` value is the assembled root context (provider-reported input plus output
once available, with the runtime estimator reported at dispatch) and `size` is the
selected model's context ceiling. Model switches update the ceiling immediately.
Paseo's ACP adapter (inspected in 0.2.3 and 0.9.2) ignores `usage_update`
notifications, so its context gauge does **not** respond to these values.
As a compatibility path, completed `session/prompt` responses also include
ACP's optional (unstable) cumulative token usage (`inputTokens`, `outputTokens`,
`totalTokens`, and `cachedReadTokens` when available); Paseo consumes those
for its token totals. Paseo's ACP adapter does not map response usage to its
context-window fields, so this cannot restore the gauge in that version.
Cumulative usage/cost totals remain in the trace (`agent cost`); ACP's
optional cost field is not emitted yet.

## Manual smoke test

1. `cargo build --release -p agent` and put `target/release/agent` on PATH
   (or use an absolute path in the Paseo config above).
2. For subscription auth: `agent auth login claude-code` (or `codex`).
3. Start Paseo, pick the `agentd` provider, create a session, and verify:
   - assistant text streams incrementally;
   - a prompt like "run ls" raises a permission prompt; allow → the
     command runs and its output renders; reject → the agent explains and
     continues;
   - "allow always" stops further prompts; the `yolo` mode does the same;
   - a prompt like "ask me which of two databases to use" raises a
     chooser with one button per answer; picking one feeds the answer
     back, and the agent continues with it;
   - cancel mid-turn returns the session to idle;
   - restarting Paseo and reopening the session replays the conversation;
   - the model picker lists your `models.yaml` aliases and switching one
     takes effect on the next turn.

Protocol-level debugging without Paseo:

```sh
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}' \
  | agent --acp
```

## Follow-ups (not yet implemented)

- Thought updates (reasoning deltas are live-only, including Codex OAuth
  summary/text deltas when emitted; ACP wiring remains).
- Deterministic mid-effect cancellation (cancel token through agent-core).
- `session/delete`, listing agentd supervisor sessions, MCP server passthrough, `fs/*`
  client capabilities, cumulative wire-level cost reporting.
