#!/usr/bin/env bash
# Minimal interactive chat driver for the rust `agent`.
# Spawns a NUL-delimited --session, streams its (human) stdout to your
# terminal, and feeds each line you type as a NUL-framed frame to its stdin.
#
# Usage:
#   ./chat.sh                       # default model/provider
#   AGENT=./target/release/agent ./chat.sh --model openai/gpt-5.5
# Any args after the script are passed straight through to the agent.
#
# Ctrl-D (EOF) or an empty line sends the empty frame -> clean shutdown.

set -euo pipefail

AGENT="${AGENT:-./target/release/agent}"

# fd 3 -> agent stdin; agent stdout/stderr inherit our terminal (banner on
# stderr in default mode, clean per-turn human text on stdout).
exec 3> >("$AGENT" --session "$@")
AGENT_PID=$!

cleanup() {
  # send empty frame for a graceful break, then close the pipe
  printf '\0' >&3 2>/dev/null || true
  exec 3>&- 2>/dev/null || true
  wait "$AGENT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

while IFS= read -r -p '> ' line; do
  [ -z "$line" ] && break          # blank line == quit
  printf '%s\0' "$line" >&3
done
