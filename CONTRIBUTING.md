# Contributing to agentd

Thanks for your interest in contributing. This document covers dev setup, how
to run the checks CI runs, and the conventions we use for commits and pull
requests. For an orientation to the codebase itself, read `AGENTS.md` and
`ARCHITECTURE.md` first.

## Dev setup

### With Nix (recommended)

The repo ships a flake with a dev shell containing the full toolchain
(rustc, cargo, rustfmt, clippy, pkg-config, openssl, CA certs):

```sh
nix develop
# or, if you use direnv: `direnv allow` — the .envrc loads the shell
```

### Without Nix

Any recent stable Rust toolchain works (CI uses `dtolnay/rust-toolchain@stable`):

```sh
rustup toolchain install stable
rustup component add rustfmt clippy
```

There are no required system C libraries beyond a working linker: `reqwest`
is built with rustls, so OpenSSL is not needed for a default build.

## Build and test

```sh
# Fast inner loop
cargo build
cargo test                   # whole workspace
cargo test -p agent-core     # one crate

# Required before you call a change done (CI enforces all of these)
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test
```

CI (`.github/workflows/ci.yml`) is gated on the full offline eval suite:

```sh
./evals/release.sh
```

This runs fmt/clippy/test plus the shell-script evals (smoke, session, FIFO,
eval-policy, trace-shape, hydration, replay-divergence). Run it locally before
opening a PR; it needs no API keys. The online evals
(`./evals/release-online.sh`, and the `RUN_*`-gated scripts) require provider
keys and are optional for contributors.

If you change the behavior of `Infer`/`Eval` or the trace output, check
whether the eval replay fixtures need regenerating — do not hand-edit
fixtures blindly.

Do **not** overwrite an existing `~/.config/agent/models.yaml`; it is runtime
configuration and may contain local aliases used by deployed services.

## Commit conventions

Look at `git log` for the house style. In short:

- Small, focused commits — one logical change each.
- Subject line prefixed with the area it touches when that adds clarity:
  `agent-core: ...`, `agent: ...`, `agent-oauth: ...`, `docs: ...`,
  `evals: ...`. Plain subjects are fine for repo-wide changes.
- Imperative or descriptive subject, no trailing period, aim for <72 chars.
- The body explains **what and why**, not just what. For behavior changes,
  say what the failure mode was and how the change addresses it. Mention
  the tests that pin the new behavior.
- If your work was done with an AI agent, keep the `Co-Authored-By` trailer
  it adds.

## Pull requests

- Target `main`.
- Keep PRs as small and reviewable as their subject allows; prefer a stack of
  focused commits over one squashed blob.
- CI must be green: it runs `./evals/release.sh` on every PR.
- No warnings: clippy runs with `-D warnings`.
- Update the relevant design doc in `docs/` (e.g. `AGENT_IR.md`, `GC.md`,
  `MEMORY.md`) when you change the subsystem it describes.

## Style notes

- Idiomatic Rust; the IR effects and the interpreter are the source of truth.
- Don't add ambient global mutation for trace/context propagation — follow
  the existing seam patterns (e.g. `TraceSink`, env-injection for context).
- When testing the shell-backed `Eval`, run agents in disposable workspaces
  and mount source read-only (see "Running safely" in `README.md`).

## Reporting bugs and requesting features

Use the issue templates in `.github/ISSUE_TEMPLATE/`. For bugs, a trace file
(`~/.local/share/agent/traces/<run-id>.jsonl`, redacted of anything
sensitive) plus the exact CLI invocation is the fastest path to a fix.

## License

By contributing, you agree that your contributions are licensed under the
MIT license (see `LICENSE`).
