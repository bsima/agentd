#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
run_id="trace-shape-smoke"
fixture="$workdir/replay.jsonl"
cat >"$fixture" <<'JSONL'
{"event":"HydrationStart","run_id":"fixture","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"fixture","op_id":1,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"fixture","op_id":2,"model":"ignored","prompt":[],"prompt_preview":"trace","timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"fixture","op_id":2,"response":{"content":"trace-shape","tool_calls":[],"tokens":3},"response_preview":"trace-shape","tokens":3,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

env -u AGENT_API_KEY -u OPENROUTER_API_KEY HOME="$workdir/home" "$agent_bin" \
  --run-id "$run_id" \
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
for event in events:
    if event["event"] in {"HydrationStart", "InferCall", "InferResult"}:
        if "run_id" not in event or "op_id" not in event:
            raise SystemExit(f"event missing run_id/op_id: {event}")
PY

echo "ok: trace shape eval passed"
