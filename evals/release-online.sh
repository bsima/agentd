#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_file="${1:-${RELEASE_MODELS_FILE:-$repo_root/evals/release-models.yaml}}"

if [[ ! -f "$models_file" ]]; then
  echo "error: release models file not found: $models_file" >&2
  exit 1
fi

workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

export XDG_CONFIG_HOME="$workdir/config"
mkdir -p "$XDG_CONFIG_HOME/agent"
cp "$models_file" "$XDG_CONFIG_HOME/agent/models.yaml"

cargo build --manifest-path "$repo_root/Cargo.toml" --quiet

models_tsv="$workdir/models.tsv"
python3 - "$models_file" >"$models_tsv" <<'PY'
import re, sys

path = sys.argv[1]
models = []
current = None
in_models = False

def clean(value):
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value

for raw in open(path, encoding="utf-8"):
    line = raw.rstrip("\n")
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if re.match(r"^models\s*:\s*$", line):
        in_models = True
        continue
    if not in_models:
        continue
    m = re.match(r"^\s*-\s+name\s*:\s*(.*)$", line)
    if m:
        if current:
            models.append(current)
        current = {"name": clean(m.group(1))}
        continue
    if current is None:
        continue
    m = re.match(r"^\s+([A-Za-z0-9_-]+)\s*:\s*(.*)$", line)
    if m:
        current[m.group(1)] = clean(m.group(2))

if current:
    models.append(current)

for model in models:
    name = model.get("name", "")
    provider = model.get("provider", "")
    api_id = model.get("api_id") or name
    api_key = model.get("api_key", "")
    if name:
        print("\t".join([name, provider, api_id, api_key]))
PY

ran=0

while IFS=$'\t' read -r name provider api_id api_key; do
  case "$provider" in
    openai-codex|codex-oauth|claude-code|claude-code-oauth)
      echo "error: $name uses OAuth provider $provider; release CI omits OAuth evals, so remove it from $models_file" >&2
      exit 1
      ;;
  esac

  if [[ "$api_key" == \$* ]]; then
    var="${api_key#\$}"
    if [[ -z "${!var:-}" ]]; then
      echo "error: $name needs env var $var from $models_file" >&2
      exit 1
    fi
  fi

  echo "[online] shell: $name ($provider / $api_id)"
  RUN_AGENT_ONLINE_EVAL=1 AGENT_ONLINE_MODEL="$name" "$repo_root/evals/online-shell.sh"
  ran=$((ran + 1))

  echo "[online] infer-tool: $name ($provider / $api_id)"
  RUN_AGENT_ONLINE_EVAL=1 AGENT_ONLINE_MODEL="$name" AGENT_ONLINE_INFER_MODEL="$api_id" "$repo_root/evals/online-infer-tool.sh"
  ran=$((ran + 1))
done <"$models_tsv"

if [[ "$ran" -eq 0 ]]; then
  echo "error: no online evals ran from $models_file" >&2
  exit 1
fi

echo "ok: online release evals passed (ran: $ran)"
