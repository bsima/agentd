#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_RUST_AGENT_INTEGRATION:-}" != "1" ]]; then
  echo "skip: set RUN_RUST_AGENT_INTEGRATION=1 to run Haskell agentd persistent integration"
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUST_AGENT_SRC="${RUST_AGENT_SRC:-$repo_root}"
exec "$HOME/omni/live/Omni/Agentd/Test/rust-agent-integration.sh"
