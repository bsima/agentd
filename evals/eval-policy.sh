#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[eval-policy] running Eval policy regression test"
cargo test --manifest-path "$repo_root/Cargo.toml" -q -p agent-core \
  eval_timeout_output_cap_cwd_and_clean_env_are_enforced

echo "ok: eval policy eval passed"
