#!/usr/bin/env bash
# Persistent-session eval for the Rust `agentd` supervisor
# (docs/SUPERVISOR.md acceptance): drives the real `agentd` + `agent`
# binaries through start/status/send/stop/resume/set-model/gen-systemd,
# fully offline via a replay-trace fixture — no credentials, no network.
#
# This replaces the forwarding shim to the Haskell supervisor's
# integration suite (~/omni/live/Omni/Agentd); the Rust port is a clean
# break from the Haskell session layout (migration note in
# docs/SUPERVISOR.md).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
export AGENTD_HOME="$workdir/agentd-home"
cleanup() {
  "$agentd_bin" stop psession --grace 1 >/dev/null 2>&1 || true
  rm -rf "$workdir"
}
trap cleanup EXIT

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet
agent_bin="$repo_root/target/debug/agent"
agentd_bin="$repo_root/target/debug/agentd"
export AGENTD_AGENT_BIN="$agent_bin"
export HOME="$workdir/home"
mkdir -p "$HOME"
unset AGENT_API_KEY OPENROUTER_API_KEY ANTHROPIC_API_KEY AGENT_PROVIDER AGENT_MODEL 2>/dev/null || true

fail() {
  echo "error: $*" >&2
  echo "--- stderr.log ---" >&2
  cat "$AGENTD_HOME/psession/stderr.log" >&2 2>/dev/null || true
  echo "--- stdout.jsonl ---" >&2
  cat "$AGENTD_HOME/psession/stdout.jsonl" >&2 2>/dev/null || true
  exit 1
}

# Replay fixture: two session turns recorded at the loop's stable effect
# locations (same recipe as evals/session.sh).
ir_effect="$("$agent_bin" ir-effect --model ignored --visit 0)"
ir_effect_1="$("$agent_bin" ir-effect --model ignored --visit 1)"
fixture="$workdir/replay.jsonl"
cat >"$fixture" <<JSONL
{"event":"HydrationStart","run_id":"agentd-fixture","op_id":1,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"agentd-fixture","op_id":1,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"agentd-fixture","op_id":2,"model":"ignored","prompt":[],"prompt_preview":"first","effect":$ir_effect,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"agentd-fixture","op_id":2,"response":{"finish_reason":"stop","content":"persist-one","tool_calls":[],"input_tokens":0,"output_tokens":1,"total_tokens":1},"response_preview":"persist-one","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationStart","run_id":"agentd-fixture","op_id":3,"sources":["TemporalHistory","SessionContext"],"max_bytes":null,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"HydrationEnd","run_id":"agentd-fixture","op_id":3,"section_count":0,"total_bytes":0,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferCall","run_id":"agentd-fixture","op_id":4,"model":"ignored","prompt":[],"prompt_preview":"second","effect":$ir_effect_1,"timestamp":"2026-01-01T00:00:00Z"}
{"event":"InferResult","run_id":"agentd-fixture","op_id":4,"response":{"finish_reason":"stop","content":"persist-two","tool_calls":[],"input_tokens":0,"output_tokens":1,"total_tokens":1},"response_preview":"persist-two","input_tokens":0,"output_tokens":1,"total_tokens":1,"duration_ms":0,"timestamp":"2026-01-01T00:00:00Z"}
JSONL

# --- start / status ---------------------------------------------------------
"$agentd_bin" start psession --model ignored -- --replay-trace "$fixture" \
  || fail "agentd start failed"
"$agentd_bin" status psession --json | grep -q '"running": true' \
  || fail "status should report the session running"
grep -q '^model: ignored$' "$AGENTD_HOME/psession/agent.md" \
  || fail "start should seed the canonical spec"

# --- send: envelope correlation + supplied turn id --------------------------
out="$("$agentd_bin" send psession 'first' --timeout 30 2>/dev/null)"
[[ "$out" == "persist-one" ]] || fail "first send: expected persist-one, got: $out"
out="$("$agentd_bin" send psession 'second' --turn-id eval-turn-2 --timeout 30 2>/dev/null)"
[[ "$out" == "persist-two" ]] || fail "second send: expected persist-two, got: $out"
grep -q '"turn_id":"eval-turn-2"' "$AGENTD_HOME/psession/stdout.jsonl" \
  || fail "the supplied turn id must ride the machine events"
# attach re-reads a completed turn from disk without re-running anything.
out="$("$agentd_bin" attach psession eval-turn-2 --timeout 5 2>/dev/null)"
[[ "$out" == "persist-two" ]] || fail "attach: expected persist-two, got: $out"

# --- stop --------------------------------------------------------------------
"$agentd_bin" stop psession >/dev/null || fail "agentd stop failed"
"$agentd_bin" status psession --json | grep -q '"running": false' \
  || fail "status should report the session stopped"

# --- resume from a checkpoint ------------------------------------------------
# Replay sessions deliberately do not checkpoint; seed one the way the agent
# writes them so resume exercises the real --resume path (run id + history
# survive the restart).
run_id="$(cat "$AGENTD_HOME/psession/run-id")"
cat >"$AGENTD_HOME/psession/checkpoints/session-latest.json" <<JSON
{
  "run_id": "$run_id",
  "sequence": 0,
  "model": "ignored",
  "provider_url": "http://replay.invalid",
  "messages": [
    {"role": "system", "content": "eval system prompt"},
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "persist-one"}
  ],
  "trace_path": "$HOME/.local/share/agent/traces/$run_id.jsonl",
  "timestamp": "2026-01-01T00:00:00Z"
}
JSON
"$agentd_bin" resume psession -- --replay-trace "$fixture" \
  || fail "agentd resume failed"
[[ "$(cat "$AGENTD_HOME/psession/run-id")" == "$run_id" ]] \
  || fail "resume must keep the checkpoint's run id"
out="$("$agentd_bin" send psession 'after resume' --timeout 30 2>/dev/null)"
[[ "$out" == "persist-one" ]] || fail "post-resume send: expected persist-one, got: $out"
"$agentd_bin" stop psession >/dev/null

# --- spec is canonical: set-model edits agent.md in place --------------------
if "$agentd_bin" start psession --model drifted >/dev/null 2>"$workdir/refused.log"; then
  fail "start flags must be refused once the spec exists"
fi
"$agentd_bin" stop psession --grace 1 >/dev/null 2>&1 || true
grep -q "canonical" "$workdir/refused.log" \
  || fail "the refusal should explain that the spec is canonical"
"$agentd_bin" set-model psession still-ignored >/dev/null || fail "set-model failed"
grep -q '^model: still-ignored$' "$AGENTD_HOME/psession/agent.md" \
  || fail "set-model must edit the spec in place"

# --- gen-systemd --------------------------------------------------------------
unit="$workdir/agentd-psession.service"
"$agentd_bin" gen-systemd psession --output "$unit" 2>/dev/null || fail "gen-systemd failed"
grep -q '^# GENERATED' "$unit" || fail "unit must carry the GENERATED header"
grep -q '^Restart=on-failure$' "$unit" || fail "unit must restart on failure"
grep -q "resume psession$" "$unit" || fail "ExecStart must go through agentd resume"
if command -v systemd-analyze >/dev/null 2>&1; then
  if ! verify_out="$(systemd-analyze verify --user "$unit" 2>&1)"; then
    echo "warning: systemd-analyze verify reported issues (non-fatal in sandboxes):" >&2
    printf '%s\n' "$verify_out" >&2
  fi
fi

echo "ok: agentd persistent eval passed"
