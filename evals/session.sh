#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
ir_effect="$("$agent_bin" ir-effect --model ignored --visit 0)"
ir_effect_1="$("$agent_bin" ir-effect --model ignored --visit 1)"
trace_fixture="$workdir/replay.jsonl"
cat >"$trace_fixture" <<JSONL
{"event":"HydrationStart","run_id":"session-fixture","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"session-fixture","op_id":1,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"session-fixture","op_id":2,"model":"ignored","prompt":[],"prompt_preview":"first","effect":$ir_effect,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"session-fixture","op_id":2,"response":{"finish_reason":"stop","content":"session-one","tool_calls":[],"input_tokens":0,"output_tokens":1,"total_tokens":1},"response_preview":"session-one","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationStart","run_id":"session-fixture","op_id":3,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"session-fixture","op_id":3,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"session-fixture","op_id":4,"model":"ignored","prompt":[],"prompt_preview":"second","effect":$ir_effect_1,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"session-fixture","op_id":4,"response":{"finish_reason":"stop","content":"session-two","tool_calls":[],"input_tokens":0,"output_tokens":1,"total_tokens":1},"response_preview":"session-two","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

output="$(env -u AGENT_API_KEY -u OPENROUTER_API_KEY HOME="$workdir/home" "$agent_bin" \
  --session \
  --replay-trace "$trace_fixture" \
  --model ignored \
  < <(printf 'first\0second\0\0'))"

if [[ "$output" != *"session-one"* ]] || [[ "$output" != *"session-two"* ]]; then
  echo "error: session output missing expected markers" >&2
  printf 'output:\n%s\n' "$output" >&2
  exit 1
fi

echo "ok: session eval passed"
