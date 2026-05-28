#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test

./evals/smoke.sh
./evals/session.sh
./evals/fifo.sh
./evals/eval-policy.sh
./evals/trace-shape.sh
./evals/hydration.sh
./evals/replay-divergence.sh

# Optional/gated evals. They skip unless their RUN_* env vars are set.
./evals/online-shell.sh
./evals/online-infer-tool.sh
./evals/agentd-persistent.sh

echo "ok: release eval passed"
