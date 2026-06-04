#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() {
  rm -rf "$workdir"
}
trap cleanup EXIT

echo "[smoke] building agent"
cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"

if [[ ! -x "$agent_bin" ]]; then
  echo "error: agent binary not found: $agent_bin" >&2
  exit 1
fi

echo "[smoke] cli help"
"$agent_bin" --help >"$workdir/help.txt"
grep -q -- "--replay-trace" "$workdir/help.txt"
grep -q -- "--eval-timeout-seconds" "$workdir/help.txt"

echo "[smoke] offline replay without API key or shell eval"
trace="$workdir/replay.jsonl"
cat >"$trace" <<'JSONL'
{"event":"HydrationStart","run_id":"smoke-replay","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"smoke-replay","op_id":1,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"smoke-replay","op_id":2,"model":"ignored","prompt":[],"prompt_preview":"smoke","timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"smoke-replay","op_id":2,"response":{"content":"agentd-eval-smoke","tool_calls":[],"input_tokens":0,"output_tokens":1,"total_tokens":1},"response_preview":"agentd-eval-smoke","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

output="$({ env -u AGENT_API_KEY -u OPENROUTER_API_KEY "$agent_bin" \
  --runtime op \
  --replay-trace "$trace" \
  --model ignored \
  "smoke"; } 2>"$workdir/replay.stderr")"

if [[ "$output" != *"agentd-eval-smoke"* ]]; then
  echo "error: replay output did not contain smoke marker" >&2
  echo "stdout: $output" >&2
  echo "stderr:" >&2
  cat "$workdir/replay.stderr" >&2
  exit 1
fi

echo "ok: smoke eval passed"
