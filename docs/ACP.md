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

- `initialize` — protocol v1, text-only prompts, `loadSession: true`.
- `session/new` — one fresh `Runtime` (agent loop, trace, checkpoints) per
  session. The request `cwd` becomes the shell tool's working directory and
  the system prompt's cwd line. Session state lives under
  `~/.local/share/agent/acp/<session-id>/`.
- `session/prompt` — runs one agent-loop turn. Assistant text streams live
  as `agent_message_chunk` updates (see Streaming below); shell, memory,
  and native tool effects surface as `tool_call` / `tool_call_update` with
  full untruncated payloads in `rawInput`/`rawOutput`.
- `session/request_permission` — shell commands are approval-gated by
  default in ACP mode: each gated command round-trips to the client with
  allow-once / allow-always / reject-once options. A rejection is a typed
  denial the model sees and recovers from — the turn continues. Disable
  gating with `--acp-shell-approval=false` or per session via the `yolo`
  mode. Approvals resolve fully in-process (no `agent approvals` records).
- `session/cancel` — drops the in-flight turn (closing any provider
  stream) and answers the prompt with `stopReason: "cancelled"`. The
  session survives; history keeps the user message. Note: an already
  in-flight shell command is not killed synchronously — it runs to its
  timeout in the background (a cancel-token in agent-core is the planned
  tightening).
- `session/load` — rebuilds the runtime from the session's
  `session-latest.json` checkpoint and replays the conversation as
  user/agent message chunks before responding. Text-only replay: past tool
  traffic is not reconstructed.
- Modes — `ask` (default; approval-gated shell) and `yolo`;
  `session/set_mode` flips them and emits `current_mode_update`.
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

Usage/cost totals are recorded in the trace as usual (`agent cost`);
per-turn `usage` on the ACP wire is not reported yet (the schema field is
unstable upstream).

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

- Thought/plan updates (no reasoning events exist in the runtime trace).
- Deterministic mid-effect cancellation (cancel token through agent-core).
- `session/list` / `session/delete`, MCP server passthrough, `fs/*`
  client capabilities, wire-level `usage` reporting.
