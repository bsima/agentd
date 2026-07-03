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

# Turn-id envelope correlation (t-1308.2, docs/SUPERVISOR.md): drive the same
# replay session in --json mode with a raw frame, a v1 envelope frame, and an
# erroring envelope turn (no recorded replay result), then assert every
# machine event carries the right turn_id.
json_out="$workdir/machine-events.jsonl"
env -u AGENT_API_KEY -u OPENROUTER_API_KEY HOME="$workdir/home" "$agent_bin" \
  --session \
  --json \
  --run-id session-eval \
  --replay-trace "$trace_fixture" \
  --model ignored \
  < <(printf '%s\0' \
      'first' \
      '{"v":1,"turn_id":"turn-custom-42","input":"second","metadata":{"sender":"eval"}}' \
      '{"v":1,"turn_id":"turn-err","input":"third"}' \
      '') >"$json_out" 2>/dev/null

# require_event <description> <pattern>... : some machine-event line in
# $json_out must contain every pattern (fixed strings).
require_event() {
  local desc="$1"
  shift
  local matches
  matches="$(cat "$json_out")"
  local pattern
  for pattern in "$@"; do
    matches="$(printf '%s\n' "$matches" | grep -F -- "$pattern" || true)"
    if [[ -z "$matches" ]]; then
      echo "error: $desc (no machine event matched: $pattern)" >&2
      cat "$json_out" >&2
      exit 1
    fi
  done
}

# Raw frame: the agent mints <run_id>-t<seq> (0-based turn ordinal).
require_event "raw frame completion must carry the minted turn id" \
  '"custom_type":"agent_start"' '"turn_id":"session-eval-t0"'
require_event "raw frame completion must carry the minted turn id" \
  '"custom_type":"agent_complete"' '"response":"session-one"' '"turn_id":"session-eval-t0"'

# Envelope frame: the supplied turn id is echoed on start and completion,
# and metadata rides back on the completion event.
require_event "envelope start must echo the supplied turn id" \
  '"custom_type":"agent_start"' '"turn_id":"turn-custom-42"'
require_event "envelope completion must echo turn id and metadata" \
  '"custom_type":"agent_complete"' '"response":"session-two"' \
  '"turn_id":"turn-custom-42"' '"metadata":{"sender":"eval"}'

# Errored turn (replay has no recorded result for a third turn): the error
# event must still correlate via the supplied turn id.
require_event "errored turn must still carry its turn id" \
  '"custom_type":"agent_error"' '"turn_id":"turn-err"'

echo "ok: session eval passed"
