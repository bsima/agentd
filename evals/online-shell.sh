#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_AGENT_ONLINE_EVAL:-}" != "1" ]]; then
  echo "skip: set RUN_AGENT_ONLINE_EVAL=1 to run online model eval"
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
model="${AGENT_ONLINE_MODEL:-openrouter/auto}"
run_id="online-shell-smoke-$$"

output="$(HOME="$workdir/home" "$agent_bin" \
  --run-id "$run_id" \
  --model "$model" \
  --eval-timeout-seconds 10 \
  --eval-max-output-bytes 65536 \
  'Use the shell tool to run: printf agentd-online-smoke. Then reply with exactly the output.' )"

if [[ "$output" != *"agentd-online-smoke"* ]]; then
  echo "error: online output missing smoke marker" >&2
  printf 'output:\n%s\n' "$output" >&2
  exit 1
fi

trace="$workdir/home/.local/share/agent/traces/$run_id.jsonl"
if ! grep -q '"event":"EvalCall"' "$trace" || ! grep -q '"event":"EvalResult"' "$trace"; then
  echo "error: online trace missing EvalCall/EvalResult" >&2
  cat "$trace" >&2
  exit 1
fi

echo "ok: online shell eval passed"
