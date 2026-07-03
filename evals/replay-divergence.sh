#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
ir_effect="$("$agent_bin" ir-effect --model foo --visit 0)"
fixture="$workdir/replay.jsonl"
cat >"$fixture" <<JSONL
{"event":"HydrationStart","run_id":"fixture","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"fixture","op_id":1,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"fixture","op_id":2,"model":"foo","prompt":[],"prompt_preview":"diverge","effect":$ir_effect,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"fixture","op_id":2,"response":{"finish_reason":"stop","content":"should-not-print","tool_calls":[],"input_tokens":0,"output_tokens":1,"total_tokens":1},"response_preview":"should-not-print","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

set +e
env -u AGENT_API_KEY -u OPENROUTER_API_KEY HOME="$workdir/home" "$agent_bin" \
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
if ! grep -qE 'replay diverged|AgentIR replay missing InferCall' "$workdir/stderr"; then
  echo "error: expected replay divergence error" >&2
  cat "$workdir/stderr" >&2
  exit 1
fi

echo "ok: replay divergence eval passed"
