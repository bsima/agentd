# agentd

[![CI](https://github.com/bsima/agentd/actions/workflows/ci.yml/badge.svg)](https://github.com/bsima/agentd/actions/workflows/ci.yml)

A Rust runtime for long-running AI agents. The main idea is that **Linux is the harness**.

This repo currently contains:

- `agent-core`: a free-monad-based interpreter with an `Infer` operation
- `agent`: CLI for oneshot and persistent agents, this follows the Unix philosophy of "do one thing well", all it does is run an agentic loop
- `agent-oauth`: experimental support for using codex/claude-code subscription

This was originally written in Haskell in my private monorepo of projects, I'm in the process of porting it to Rust and releasing it here.
There's also an `agentd` process supervisor, eventually I will port that to Rust too.

## Core ideas

Between the Haskell prototype and this Rust port, the implementation is changing a lot, but the core ideas have remained.

### An agent is a free-monad over inference

Traditional computing is:

```text
eval(structured_data)
```

Modern ML is:

```text
infer(unstructured_data)
```

An agent alternates between the two. It infers from context, evaluates effects against the environment, reads the result, then infers again.

`agent-core` makes that structure explicit with a free monad over `OpF`:

```rust
pub enum OpF<S, A> {
    Infer { model, prompt, next },  // LLM call: infer(unstructured)
    Eval  { command, next },        // process call: eval(structured), currently $SHELL -c
    Get   { key, next },            // state/context read
    Put   { key, value, next },     // state/context write
    Emit  { event, next },          // trace
    Par   { ops, next },            // parallel effects
    Pure(A),
}
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the longer version.

### Infer can call Infer

All `OpF` variants are available to agent programs, including `Infer`.

So a multi-agent system is not a special framework layer. It is just an agent program that emits multiple `Infer` calls, maybe with different models, prompts, budgets, or context windows. The outer agent is the orchestrator.

This is the SICP meta-circular idea applied to agents. `eval` calling `eval` collapses the interpreter/object-language boundary, `Infer` calling `Infer` does the same thing for agents.

### Context is a window over a log, not the log itself

The durable history is an append-only record (checkpoints, traces, replay all
depend on that). What the *model sees per turn* is a managed window over it:
`agent-core` models context as keyed reads and writes, hydrates passively
before each turn, and garbage-collects the outbound window under budget
pressure (see `docs/GC.md`). This is similar to RLM.

There are really only 2 ways to lookup content for context: temporally via chat history, and semantically via similarity search; these operations work on any unstructured text.
Similarly, there are 2 times during an agentic turn that an agent can build context: it can be injected passively into the LLM prompt, or the agent can actively use a tool call to find more context.
This gives us a neat 2x2 matrix for the design space.

|                | Passive                    | Active                  |
|----------------|----------------------------|-------------------------|
| Temporal       | recent messages/history    | `Get("temporal:query")` |
| Semantic       | RAG/static workspace       | `Get("semantic:topic")` |

Passive sources run before the model sees a turn, like traditional RAG or appending chat messages.
Active sources are available when the agent decides it needs them.

### Sessions are Unix processes

The Rust CLI runs an agent loop as a normal process. It can take one prompt, read NUL-terminated turns from stdin, or read NUL-terminated turns from a FIFO path. After each turn it can write checkpoints. The protocol is pipes and files.

```sh
# One-shot prompt
agent --model openai/gpt-4o-mini "inspect this repo and summarize it"

# NUL-framed stdin session
printf 'go build the thing\0\0' | agent --session --checkpoint-dir .agent-checkpoints

# FIFO-driven session
mkfifo .agent.fifo
agent --fifo .agent.fifo --checkpoint-dir .agent-checkpoints &
printf 'run cargo test\0' > .agent.fifo
```

The future Rust supervisor will wrap this with commands like:

```sh
agentd start myagent
agentd send myagent "go build the thing"
agentd logs myagent
agentd stop myagent
```

This allows us to use all the regular Linux tooling for managing agents: systemd, kubernetes, docker/podman.
Feel free to sandbox your agent with bwrap or nix or whatever you want.

## Quickstart

Build and test:

```sh
cargo test
cargo build --release
```

Configure a model registry:

```sh
mkdir -p ~/.config/agent
cp -n examples/models.yaml ~/.config/agent/models.yaml
```

Do not overwrite an existing `~/.config/agent/models.yaml`; it is runtime configuration and may contain local aliases used by deployed services.

Set the provider key:

```sh
export OPENROUTER_API_KEY=...
```

Run a one-shot prompt:

```sh
cargo run -- --model openrouter/auto "say hello"
```

You can also run a markdown file as the prompt:

```sh
cargo run -- ./task.md
cat input.json | cargo run -- ./task.md
```

Markdown prompts may include YAML frontmatter for fields the CLI applies directly: `provider`, `model`, `max_iterations`, and `system_prompt`.

```md
---
model: openrouter/auto
max_iterations: 8
system_prompt: ./system.md
---

Inspect this repo and summarize it.
```

`system_prompt` may be inline text or a path resolved relative to the markdown file.

You can also skip the registry and pass a raw model id.
Then the CLI uses `OPENROUTER_BASE_URL` or `https://openrouter.ai/api/v1`, and `AGENT_API_KEY` or `OPENROUTER_API_KEY`.

Useful execution controls:

```sh
agent --eval-timeout-seconds 10 --eval-max-output-bytes 65536 --eval-env clean "inspect this repo"
```

By default (`--eval-env inherit`), shell commands issued by the model inherit
the parent environment **minus known credential variables** —
`ANTHROPIC_AUTH_TOKEN` and anything ending in `_API_KEY` — so the model cannot
read the key the agent runs on. Working credentials like `GITHUB_TOKEN` are
not stripped. Use `--eval-env inherit-full` if your commands genuinely need
the provider keys, or `--eval-env clean` for an empty environment.

Replay recorded `Infer` and `Eval` results without an API key or shell execution:

```sh
agent --replay-trace ~/.local/share/agent/traces/<run-id>.jsonl --model ignored "same prompt"
```

## Running safely

The default interpreter gives the model direct shell execution.
The sane default is a disposable workspace with only the files and credentials needed for the task.

A minimal container pattern:

```sh
cargo build --release
mkdir -p .agent-home/.config/agent .agent-work
cp -R ./your-project .agent-work/project

podman run --rm -it \
  -e SHELL=/bin/sh \
  -e OPENROUTER_API_KEY \
  -v "$PWD/target/release/agent:/usr/local/bin/agent:ro" \
  -v "$PWD/.agent-home:/home/agent" \
  -v "$PWD/examples/models.yaml:/home/agent/.config/agent/models.yaml:ro" \
  -v "$PWD/.agent-work:/work" \
  -w /work/project \
  docker.io/library/rust:1 \
  agent --model openrouter/auto "inspect this project"
```

For real use, prefer a purpose-built image with `agent`, the allowed toolchain, and no ambient secrets.
Add network only when the task needs it.
Mount source read-only unless the agent is supposed to edit it.
Keep traces and checkpoints outside your main home directory if command output may contain secrets.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the free monad design, hydration model, session model, and interpreter story.
See [ROADMAP.md](./ROADMAP.md) for the Rust port plan.

## Status

M1 and the AgentIR track are implemented: single-agent CLI, the serializable AgentIR runtime (the CLI's only runtime; the closure-based `Op` layer remains a library builder/test API), bounded shell-backed `Eval`, model-backed `Infer`, NUL/FIFO session input, structured traces with error events, stable-effect-id replay (including replay of failures), mid-turn IR checkpoints, context GC (`ring`, `mark-sweep`), hydration registry with PromptIR provenance, and optional model registry loading.

Active development:

- M2: Rust `agentd` supervisor/daemon port from the working Haskell implementation (design: [docs/SUPERVISOR.md](./docs/SUPERVISOR.md))
- M3: hermetic PATH, stronger sandbox integration, richer `HydrationSource` implementations
- M4: parallel interpreter with real `Par` execution
- M5: distributed interpreter, multi-VM campaigns

## Prior art

The design comes from `Omni/Agent/Op.hs`, a Haskell prototype that proved the free monad Op abstraction in production use. This Rust port is a translation, not a rewrite. The Haskell codebase remains the reference for Op semantics.

The meta-circular `Infer`-emitting-`Infer` pattern has direct precedent in the SICP meta-circular evaluator.

## License

MIT
