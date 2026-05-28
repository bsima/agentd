#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
agent_pid=""
cleanup() {
  if [[ -n "$agent_pid" ]]; then
    kill "$agent_pid" >/dev/null 2>&1 || true
    sleep 0.1
    kill -9 "$agent_pid" >/dev/null 2>&1 || true
  fi
  rm -rf "$workdir"
}
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
fifo="$workdir/agent.fifo"
mkfifo "$fifo"
trace_fixture="$workdir/replay.jsonl"
cat >"$trace_fixture" <<'JSONL'
{"event":"HydrationStart","run_id":"fifo-fixture","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"fifo-fixture","op_id":1,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"fifo-fixture","op_id":2,"model":"ignored","prompt":[],"prompt_preview":"fifo","timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"fifo-fixture","op_id":2,"response":{"content":"fifo-smoke","tool_calls":[],"tokens":1},"response_preview":"fifo-smoke","tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

stdout_log="$workdir/stdout.log"
stderr_log="$workdir/stderr.log"
env -u AGENT_API_KEY -u OPENROUTER_API_KEY HOME="$workdir/home" "$agent_bin" \
  --fifo "$fifo" \
  --replay-trace "$trace_fixture" \
  --model ignored \
  >"$stdout_log" 2>"$stderr_log" &
agent_pid="$!"

printf 'fifo\0' >"$fifo"

for _ in $(seq 1 100); do
  if grep -q 'fifo-smoke' "$stdout_log"; then
    echo "ok: fifo eval passed"
    exit 0
  fi
  if ! kill -0 "$agent_pid" >/dev/null 2>&1; then
    echo "error: agent exited before fifo response" >&2
    cat "$stderr_log" >&2 || true
    exit 1
  fi
  sleep 0.05
done

echo "error: timed out waiting for fifo response" >&2
cat "$stdout_log" >&2 || true
cat "$stderr_log" >&2 || true
exit 1
