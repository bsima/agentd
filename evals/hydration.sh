#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
run_id="hydration-smoke"
hydration_dir="$workdir/hydration"
mkdir -p "$hydration_dir"
printf 'alpha-context' >"$hydration_dir/a.txt"
printf 'beta-context' >"$hydration_dir/b.txt"
fixture="$workdir/replay.jsonl"
cat >"$fixture" <<'JSONL'
{"event":"HydrationStart","run_id":"fixture","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationSection","run_id":"fixture","op_id":1,"source":"local-files","kind":"Knowledge","bytes":1,"content_preview":"fixture","metadata":{},"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"fixture","op_id":1,"section_count":1,"total_bytes":1,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"fixture","op_id":2,"model":"ignored","prompt":[],"prompt_preview":"hydration","timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"fixture","op_id":2,"response":{"finish_reason":"stop","content":"hydration-smoke","tool_calls":[],"input_tokens":0,"output_tokens":1,"total_tokens":1},"response_preview":"hydration-smoke","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

# --trace-full-payloads: this eval asserts over the full Infer prompt, which
# is opt-in in traces (previews are the default).
env -u AGENT_API_KEY -u OPENROUTER_API_KEY HOME="$workdir/home" "$agent_bin" \
  --runtime op \
  --run-id "$run_id" \
  --hydration-dir "$hydration_dir" \
  --replay-trace "$fixture" \
  --trace-full-payloads \
  --model ignored \
  "hydrate" >/dev/null

trace="$workdir/home/.local/share/agent/traces/$run_id.jsonl"
python3 - "$trace" <<'PY'
import json, sys
path = sys.argv[1]
events = [json.loads(line) for line in open(path) if line.strip()]
sections = [event for event in events if event["event"] == "HydrationSection"]
if len(sections) != 1:
    raise SystemExit(f"expected exactly one hydration section, got {len(sections)}")
preview = sections[0].get("content_preview", "")
if "alpha-context" not in preview or "beta-context" not in preview:
    raise SystemExit(f"hydration preview missing file content: {preview}")
infer_calls = [event for event in events if event["event"] == "InferCall"]
if len(infer_calls) != 1:
    raise SystemExit(f"expected one infer call, got {len(infer_calls)}")
prompt_text = json.dumps(infer_calls[0].get("prompt", []))
if prompt_text.count("alpha-context") != 1 or prompt_text.count("beta-context") != 1:
    raise SystemExit(f"hydration content should appear exactly once in prompt: {prompt_text}")
PY

echo "ok: hydration eval passed"
