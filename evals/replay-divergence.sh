#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
fixture="$workdir/replay.jsonl"
cat >"$fixture" <<'JSONL'
{"event":"HydrationStart","run_id":"fixture","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"fixture","op_id":1,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"fixture","op_id":2,"model":"foo","prompt":[],"prompt_preview":"diverge","timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"fixture","op_id":2,"response":{"content":"should-not-print","tool_calls":[],"tokens":1},"response_preview":"should-not-print","tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

set +e
env -u AGENT_API_KEY -u OPENROUTER_API_KEY HOME="$workdir/home" "$agent_bin" \
  --runtime op \
  --replay-trace "$fixture" \
  --model bar \
  "diverge" >"$workdir/stdout" 2>"$workdir/stderr"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  echo "error: replay divergence unexpectedly succeeded" >&2
  cat "$workdir/stdout" >&2
  exit 1
fi
if ! grep -q 'replay diverged' "$workdir/stderr"; then
  echo "error: expected replay divergence error" >&2
  cat "$workdir/stderr" >&2
  exit 1
fi

echo "ok: replay divergence eval passed"
