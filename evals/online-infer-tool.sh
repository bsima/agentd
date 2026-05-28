#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_AGENT_ONLINE_EVAL:-}" != "1" ]]; then
  echo "skip: set RUN_AGENT_ONLINE_EVAL=1 to run online infer-tool eval"
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
model="${AGENT_ONLINE_MODEL:-openrouter/auto}"
run_id="online-infer-tool-smoke-$$"

output="$(HOME="$workdir/home" "$agent_bin" \
  --runtime ir \
  --run-id "$run_id" \
  --model "$model" \
  --eval-timeout-seconds 10 \
  'Use the infer tool, not the shell tool. Ask it: what exact token should I return if the secret token is agentd-infer-smoke? Then reply with exactly agentd-infer-smoke.' )"

if [[ "$output" != *"agentd-infer-smoke"* ]]; then
  echo "error: online output missing infer smoke marker" >&2
  printf 'output:\n%s\n' "$output" >&2
  exit 1
fi

trace="$workdir/home/.local/share/agent/traces/$run_id.jsonl"
python3 - "$trace" <<'PY'
import json, sys
path = sys.argv[1]
events = [json.loads(line) for line in open(path) if line.strip()]
infer_calls = [e for e in events if e.get("event") == "InferCall"]
eval_calls = [e for e in events if e.get("event") == "EvalCall"]
if len(infer_calls) < 2:
    raise SystemExit(f"expected at least 2 InferCall events, got {len(infer_calls)}")
if eval_calls:
    raise SystemExit(f"expected no EvalCall events, got {len(eval_calls)}")
PY

echo "ok: online infer-tool eval passed"
