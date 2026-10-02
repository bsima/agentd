#!/usr/bin/env bash
# Offline two-turn fork: parent and child have distinct session/run identities,
# distinct turn-two responses, and the fork never writes the source state.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
agent="${AGENT_BIN:-$repo/target/debug/agent}"
work="$(mktemp -d)"
trap 'if [[ "${KEEP_WORK:-}" != 1 ]]; then rm -rf "$work"; else echo "fixture work: $work" >&2; fi' EXIT
mkdir -p "$work/home" "$work/parent" "$work/child"
export HOME="$work/home"
unset AGENT_API_KEY OPENROUTER_API_KEY ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
old=aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa
new=bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb
run=parent-run
stamp=2026-01-01T00:00:00Z
record() {
  local path="$1" response="$2" effect="$3"
  jq -cn --arg r "$run" --arg t "$stamp" --argjson effect "$effect" \
    '{event:"InferCall",run_id:$r,op_id:1,model:"ignored",prompt:[],prompt_preview:"turn",effect:$effect,timestamp:$t}' > "$path"
  jq -cn --arg r "$run" --arg t "$stamp" --arg body "$response" \
    '{event:"InferResult",run_id:$r,op_id:1,response:{finish_reason:"stop",content:$body,tool_calls:[],input_tokens:0,output_tokens:1,total_tokens:1},response_preview:$body,input_tokens:0,output_tokens:1,total_tokens:1,duration_ms:0,timestamp:$t}' >> "$path"
}
first="$($agent ir-effect --model ignored --visit 0)"
second="$($agent ir-effect --model ignored --visit 1)"
record "$work/first.jsonl" first "$first"
printf 'one\0\0' | "$agent" --session --model ignored --run-id "$run" --session-id "$old" \
  --state-dir "$work/parent" --replay-trace "$work/first.jsonl" --replay-live-eval > "$work/one.out" 2> "$work/one.err"
test "$(cat "$work/one.out")" = first
source="$work/parent/checkpoints/session-latest.json"
test -f "$source"
cp "$source" "$work/source-original.json"
record "$work/child.jsonl" child "$second"
printf 'two for child\0\0' | "$agent" --session --model ignored --fork-from "$source" \
  --state-dir "$work/child" --session-id "$new" --parent-session "$old" --run-id child-run \
  --replay-trace "$work/child.jsonl" --replay-live-eval > "$work/child.out" 2> "$work/child.err"
test "$(cat "$work/child.out")" = child
cmp "$source" "$work/source-original.json"
jq -e --arg id "$new" --arg parent "$old" '.session_id==$id and .parent_session_id==$parent and .run_id=="child-run"' \
  "$work/child/checkpoints/session-latest.json" >/dev/null
record "$work/parent-second.jsonl" parent "$second"
printf 'two for parent\0\0' | "$agent" --session --model ignored --resume "$source" \
  --state-dir "$work/parent" --session-id "$old" --replay-trace "$work/parent-second.jsonl" --replay-live-eval \
  > "$work/parent.out" 2> "$work/parent.err"
test "$(cat "$work/parent.out")" = parent
jq -e --arg id "$old" '.session_id==$id and (.parent_session_id==null)' "$source" >/dev/null
printf 'ok: offline durable-session fork\n'
