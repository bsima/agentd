#!/usr/bin/env bash
# Offline end-to-end SIGTERM/Eval/checkpoint/resume contract. The fixture is
# generated from real agentd effect locations, so program-hash changes fail
# loudly instead of making a network call. No model credentials are supplied.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
agent="${AGENT_BIN:-$repo/target/debug/agent}"
model=ignored
resolved=ignored
work="$(mktemp -d)"
pid=''
cleanup() {
  if [[ -n "$pid" ]]; then kill -KILL "$pid" 2>/dev/null || true; fi
  if [[ "${KEEP_WORK:-}" != 1 ]]; then rm -rf "$work"; else echo "fixture work: $work" >&2; fi
}
trap cleanup EXIT
mkdir -p "$work/home" "$work/state"
export HOME="$work/home"
unset AGENT_API_KEY OPENROUTER_API_KEY ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
run=durable-fixture-run
session=aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa
stamp=2026-01-01T00:00:00Z
fixture="$work/fixture.jsonl"
# A shell command that leaves a visible partial effect. It must execute once.
command="printf x >> '$work/partial'; sleep 30"
effect="$($agent ir-effect --model "$resolved")"
jq -cn --arg r "$run" --arg m "$resolved" --argjson effect "$effect" --arg t "$stamp" \
  '{event:"InferCall",run_id:$r,op_id:1,model:$m,prompt:[],prompt_preview:"start",effect:$effect,timestamp:$t}' > "$fixture"
jq -cn --arg r "$run" --arg cmd "$command" --arg t "$stamp" \
  '{event:"InferResult",run_id:$r,op_id:1,response:{finish_reason:"tool_calls",content:"running",tool_calls:[{id:"sh-1",name:"shell",arguments:{command:$cmd}}],input_tokens:0,output_tokens:1,total_tokens:1},response_preview:"running",input_tokens:0,output_tokens:1,total_tokens:1,duration_ms:0,timestamp:$t}' >> "$fixture"

# First dry run records the true dynamic Eval location. Validation fails
# closed before executing the command because the fixture has no EvalCall.
printf 'go\0\0' | "$agent" --session --model "$model" --max-turns 16 \
  --run-id probe --replay-trace "$fixture" --replay-live-eval \
  > "$work/probe.out" 2> "$work/probe.err"
test ! -e "$work/partial"
jq -c 'select(.event=="EvalCall")' "$HOME/.local/share/agent/traces/probe.jsonl" \
  | jq -c --arg r "$run" '.run_id=$r' >> "$fixture"
test "$(grep -c '"event":"EvalCall"' "$fixture")" -eq 1

mkfifo "$work/fifo"
"$agent" --fifo "$work/fifo" --model "$model" --max-turns 16 \
  --state-dir "$work/state" --run-id "$run" --session-id "$session" \
  --replay-trace "$fixture" --replay-live-eval --trace-full-payloads --signal-deadline 5 \
  > "$work/out" 2> "$work/err" & pid=$!
printf 'go\0' > "$work/fifo"
for _ in $(seq 1 200); do
  test -e "$work/partial" && break
  kill -0 "$pid" 2>/dev/null || { cat "$work/err" >&2; exit 1; }
  sleep 0.02
done
test -e "$work/partial"
kill -TERM "$pid"
# Don't let a regression in SIGTERM handling strand CI.
(sleep 8; kill -KILL "$pid" 2>/dev/null || true) & watchdog=$!
wait "$pid"; code=$?; pid=''; kill "$watchdog" 2>/dev/null || true
test "$code" -eq 0
test "$(cat "$work/partial")" = x
checkpoint="$work/state/checkpoints/session-latest.json"
trace="$work/state/traces/$run.jsonl"
test -f "$checkpoint"
jq -e --arg id "$session" '.session_id==$id and .machine.machine.pc > 0 and .next_turn_seq==1' "$checkpoint" >/dev/null
jq -e 'select(.event=="EvalResult" and .result.status=="interrupted")' "$trace" >/dev/null
jq -e --arg id "$session" 'select(.event=="EvalResult" and .session_id==$id)' "$trace" >/dev/null
# Fork the interrupted mid-turn machine. The child must inherit the typed
# interruption without re-running the half-completed command, and must not
# write the source checkpoint.
cp "$checkpoint" "$work/source-before-fork.json"
child=bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb
printf '' | "$agent" --session --model "$model" --max-turns 16 \
  --fork-from "$checkpoint" --state-dir "$work/fork" --run-id fork-run \
  --session-id "$child" --parent-session "$session" \
  --replay-trace "$fixture" --replay-live-eval > "$work/fork.out" 2> "$work/fork.err" || true
cmp "$checkpoint" "$work/source-before-fork.json"
jq -e --arg child "$child" --arg parent "$session" \
  '.session_id==$child and .parent_session_id==$parent and .machine.machine.env.eval_result.status=="interrupted"' \
  "$work/fork/checkpoints/session-latest.json" >/dev/null
test "$(cat "$work/partial")" = x
# Extend the fixture from the recorded post-interruption InferCall. The
# recorded command was killed once; its typed result must reach this Infer.
# Deliberately run once against an incomplete fixture to record the exact
# next Infer location and its model-visible input, without calling a provider.
printf '' | "$agent" --session --model "$model" --max-turns 16 \
  --state-dir "$work/state" --run-id "$run" --session-id "$session" \
  --resume "$checkpoint" --replay-trace "$fixture" --replay-live-eval \
  --trace-full-payloads > "$work/incomplete.out" 2> "$work/incomplete.err" || true
grep -q "missing InferCall" "$work/incomplete.err"
next_call="$(jq -c 'select(.event=="InferCall" and .op_id >= 5)' "$trace" | tail -1)"
test -n "$next_call"
jq -e '.prompt | tostring | contains("interrupted")' <<< "$next_call" >/dev/null
jq -cn --argjson call "$next_call" --arg r "$run" --arg t "$stamp" \
  '$call | .run_id=$r | .op_id=10 | .timestamp=$t' >> "$fixture"
jq -cn --arg r "$run" --arg t "$stamp" \
  '{event:"InferResult",run_id:$r,op_id:10,response:{finish_reason:"stop",content:"recovered",tool_calls:[],input_tokens:0,output_tokens:1,total_tokens:1},response_preview:"recovered",input_tokens:0,output_tokens:1,total_tokens:1,duration_ms:0,timestamp:$t}' >> "$fixture"
# Continue on a different process; no shell effect may run again.
printf '' | "$agent" --session --model "$model" --max-turns 16 \
  --state-dir "$work/state" --run-id "$run" --session-id "$session" \
  --resume "$checkpoint" --replay-trace "$fixture" --replay-live-eval \
  > "$work/continued.out" 2> "$work/continued.err"
test "$(cat "$work/continued.out")" = recovered
test "$(cat "$work/partial")" = x
# An incomplete fixture probe must not publish a newer checkpoint.
test "$(cat "$work/source-before-fork.json" | jq -r .sequence)" -lt "$(jq -r .sequence "$checkpoint")"
jq -e '.machine==null and .messages[-1].content=="recovered"' "$checkpoint" >/dev/null
printf 'ok: offline durable-session suspend/resume\n'
