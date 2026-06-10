#!/usr/bin/env bash
set -euo pipefail

# Live OAuth validation (t-1129). Requires real subscription credentials and
# an interactive browser login, so it is gated and skips by default:
#
#   RUN_OAUTH_LIVE=1 ./evals/oauth-live-check.sh
#
# Walks the ROADMAP release-checklist item "OAuth flows are tested against
# real providers". File bugs (task create --discovered-from t-1129) for
# whatever breaks.

if [[ "${RUN_OAUTH_LIVE:-0}" != "1" ]]; then
  echo "skipping oauth live check: set RUN_OAUTH_LIVE=1 (needs real credentials + browser)"
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cargo build --manifest-path "$repo_root/Cargo.toml" --features oauth --quiet
agent_bin="$repo_root/target/debug/agent"

step() { printf '\n== %s ==\n' "$1"; }

step "1. device login (interactive: open the printed URLs)"
"$agent_bin" auth login codex
"$agent_bin" auth login claude-code

step "2. token store status (expect both valid; auth.json must be 0600)"
"$agent_bin" auth status
auth_file="${XDG_DATA_HOME:-$HOME/.local/share}/agent/auth.json"
perms=$(stat -c '%a' "$auth_file")
[[ "$perms" == "600" ]] || { echo "FAIL: $auth_file perms $perms != 600"; exit 1; }

step "3. one-shot chat through each OAuth provider (expect a real answer + shell tool round-trip)"
"$agent_bin" --model gpt-5 --provider openai-codex \
  "Use the shell tool to run 'printf oauth_codex_probe' and report the exact output."
"$agent_bin" --model claude-sonnet-4-5 --provider claude-code \
  "Use the shell tool to run 'printf oauth_claude_probe' and report the exact output."

step "4. refresh path: backdate the expiry, then re-run a one-shot"
python3 - "$auth_file" <<'PY'
import json, sys
path = sys.argv[1]
auth = json.load(open(path))
for token in auth.values():
    token["expires"] = 1  # epoch ms: long expired -> forces refresh
json.dump(auth, open(path, "w"), indent=2)
print("backdated all token expiries")
PY
"$agent_bin" --model gpt-5 --provider openai-codex "say exactly: refresh-ok"
"$agent_bin" auth status

echo
echo "ok: oauth live check passed"
