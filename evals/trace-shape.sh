#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
ir_effect="$("$agent_bin" ir-effect --model ignored --visit 0)"
run_id="trace-shape-smoke"
fixture="$workdir/replay.jsonl"
cat >"$fixture" <<JSONL
{"event":"HydrationStart","run_id":"fixture","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"fixture","op_id":1,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"fixture","op_id":2,"model":"ignored","prompt":[],"prompt_preview":"trace","effect":$ir_effect,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"fixture","op_id":2,"response":{"finish_reason":"stop","content":"trace-shape","tool_calls":[],"input_tokens":1,"output_tokens":2,"total_tokens":3},"response_preview":"trace-shape","input_tokens":1,"output_tokens":2,"total_tokens":3,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

hydration_dir="$workdir/hydration"
mkdir -p "$hydration_dir"
printf 'trace-context' >"$hydration_dir/context.txt"

env -u AGENT_API_KEY -u OPENROUTER_API_KEY HOME="$workdir/home" "$agent_bin" \
  --run-id "$run_id" \
  --hydration-dir "$hydration_dir" \
  --replay-trace "$fixture" \
  --model ignored \
  "trace" >/dev/null

trace="$workdir/home/.local/share/agent/traces/$run_id.jsonl"
python3 - "$trace" <<'PY'
import json, sys
path = sys.argv[1]
events = [json.loads(line) for line in open(path) if line.strip()]
by_name = {event["event"] for event in events}
required = {"HydrationStart", "HydrationEnd", "InferCall", "InferResult", "AgentDone"}
missing = required - by_name
if missing:
    raise SystemExit(f"missing trace events: {sorted(missing)}")
prompt_ir = [e for e in events if e.get("event") == "Custom" and e.get("name") == "prompt_ir"]
if not prompt_ir:
    raise SystemExit("missing prompt_ir trace event")
sections = prompt_ir[0]["data"].get("sections", [])
if not sections:
    raise SystemExit("prompt_ir trace has no sections")
source = sections[0]["source"]
if source.get("timing") != "Passive":
    raise SystemExit(f"prompt_ir section missing passive timing: {source}")
origin = source.get("origin", {})
if origin.get("Retrieval", {}).get("mode") != "Semantic":
    raise SystemExit(f"prompt_ir section missing semantic retrieval mode: {source}")
if "content" in sections[0]:
    raise SystemExit("prompt_ir trace should not include full content by default")
for event in events:
    if event["event"] in {"HydrationStart", "InferCall", "InferResult"}:
        if "run_id" not in event or "op_id" not in event:
            raise SystemExit(f"event missing run_id/op_id: {event}")
PY

echo "ok: trace shape eval passed"
