#!/usr/bin/env bash
# Offline, credential-free approval pause, independent fork and denial.
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
stamp=2026-01-01T00:00:00Z
command="printf x >> '$work/must-not-execute'"
effect="$($agent ir-effect --model ignored --require-shell-approval)"
fixture="$work/fixture.jsonl"
jq -cn --arg t "$stamp" --argjson effect "$effect" '{event:"InferCall",run_id:"parent-run",op_id:1,model:"ignored",prompt:[],prompt_preview:"start",effect:$effect,timestamp:$t}' > "$fixture"
jq -cn --arg cmd "$command" --arg t "$stamp" '{event:"InferResult",run_id:"parent-run",op_id:1,response:{finish_reason:"tool_calls",content:"running",tool_calls:[{id:"sh-1",name:"shell",arguments:{command:$cmd}}],input_tokens:0,output_tokens:1,total_tokens:1},response_preview:"running",input_tokens:0,output_tokens:1,total_tokens:1,duration_ms:0,timestamp:$t}' >> "$fixture"
# A live gate pauses before Eval: the fixture provides only Infer.
printf 'go\0\0' | "$agent" --session --model ignored --max-turns 16 \
  --run-id parent-run --session-id "$old" --state-dir "$work/parent" \
  --require-shell-approval --replay-trace "$fixture" --replay-live-eval --replay-live-approval > "$work/parent.out" 2> "$work/parent.err"
source="$work/parent/checkpoints/session-latest.json"
parent_pending="$(jq -r '.pending_approval_id' "$source")"
test "$parent_pending" != null
jq -e --arg id "$parent_pending" '.pending_id==$id and .status=="awaiting_approval"' "$work/parent/approvals/$parent_pending.json" >/dev/null
test ! -e "$work/must-not-execute"
cp "$source" "$work/source-original.json"
# Forking a pending gate must copy its record under a new run-scoped id.
printf '' | "$agent" --session --model ignored --max-turns 16 --fork-from "$source" \
  --state-dir "$work/child" --session-id "$new" --parent-session "$old" --run-id child-run \
  --require-shell-approval --replay-trace "$fixture" --replay-live-eval --replay-live-approval > "$work/child.out" 2> "$work/child.err"
child_cp="$work/child/checkpoints/session-latest.json"
child_pending="$(jq -r '.pending_approval_id' "$child_cp")"
test "$child_pending" != null
test "$child_pending" != "$parent_pending"
cmp "$source" "$work/source-original.json"
jq -e --arg id "$child_pending" '.pending_id==$id and .run_id=="child-run" and .status=="awaiting_approval"' "$work/child/approvals/$child_pending.json" >/dev/null
# Resolving one branch must not modify the other's pending approval.
"$agent" --state-dir "$work/child" approvals --deny "$child_pending" > "$work/deny.out"
jq -e '.status=="denied"' "$work/child/approvals/$child_pending.json" >/dev/null
jq -e '.status=="awaiting_approval"' "$work/parent/approvals/$parent_pending.json" >/dev/null
test ! -e "$work/must-not-execute"
# Attaching the denied child consumes the denial as a value, not the shell.
# A missing subsequent Infer fails closed without model credentials.
printf '' | "$agent" --session --model ignored --max-turns 16 --resume "$child_cp" \
  --state-dir "$work/child" --run-id child-run --session-id "$new" \
  --require-shell-approval --replay-trace "$fixture" --replay-live-eval --replay-live-approval \
  > "$work/resume.out" 2> "$work/resume.err" || true
test ! -e "$work/must-not-execute"
grep -q 'missing InferCall' "$work/resume.err"
printf 'ok: offline durable approval fork\n'
