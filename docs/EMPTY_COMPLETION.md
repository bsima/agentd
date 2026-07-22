# Empty-completion terminal stall (t-1071)

## Symptom

gpt-5.5 agents occasionally terminate one keystroke from done. agentd shows the
agent as `running`, but the trace ends with `agent_complete{response:""}` and the
run goes silent. The work is real and on disk (dirty worktree / WIP commit), but
the final commit / Review transition never executes. Most often the empty turn
lands right after the agent emits its final tool call (e.g. the squash-commit),
so the next turn — which should resolve the tool result — comes back empty.

Observed across t-1031, t-1064, t-1067. Only on gpt-5.5 so far.

## Mechanism

The provider returns a 200 OK with empty `content` AND empty `tool_calls`. The
agent loop (`agent_loop` in `op.rs` at the time; `agent_loop_ir` today) treats
an empty `tool_calls` list as "the model is done", so an empty turn is
surfaced as a final, empty response that silently kills an otherwise-active
run.

Critically, the empty completion is **deterministic for a given context**: it is
not a flaky network blip. Retrying the *identical* request hits the same wall,
which is why the first hotfix (a plain 3-attempt resend) converted the silent
failure into a descriptive one but did not actually recover the run.

## Decision: retry-with-mutation, not graceful-finalize

Two options were on the table:

1. **Graceful finalize** — treat an empty turn (especially right after a tool
   call) as "the model intends to stop" and drive agentd into its
   commit/Review path.
2. **Retry-with-mutation** — on the first empty completion, append a synthetic
   `user` continuation nudge and re-request, rather than resending the
   identical body.

We chose **retry-with-mutation**.

Graceful-finalize is unsafe here: the empty turn lands *before* the final tool
call has been resolved (the commit hasn't run yet). Auto-finalizing on empty
would commit/transition on a half-done step and paper over genuine model
failures, making them invisible. An empty completion is a model error, not a
deliberate stop signal — gpt-5.5 has no way to express "I'm done" via an empty
turn, it uses real content or stops emitting tool calls with a final message.

Mutating the retry request breaks the deterministic empty loop while preserving
the existing semantics: if the nudged retry produces a real turn, the run
continues normally; if every attempt is still empty, we surface a descriptive
terminal error (with the raw response body) instead of a silent
`agent_complete{response:""}`.

## Implementation (`crates/agent-core/src/provider.rs`)

- A 200 with `content.trim().is_empty() && tool_calls.is_empty()` becomes a
  retryable `ProviderError::EmptyCompletion { raw }`. The raw response body is
  captured and logged when it fires, so we can confirm it is genuinely empty
  rather than a parse/serialization bug on our side.
- `chat_with_retries` drives the bounded backoff loop. On an `EmptyCompletion`
  it sets a `nudge` flag so the *next* `build_chat_body` appends `CONTINUE_NUDGE`
  as an extra `user` message. The first attempt is always the unmutated context.
- After `MAX_ATTEMPTS` empties, the descriptive terminal error (including the
  raw body) is returned.

## Tests

`provider::tests` covers the recovery behavior deterministically by driving
`chat_with_retries` with a fake `send` closure (no live provider, paused tokio
clock):

- `exhausts_retries_then_returns_descriptive_error` — all-empty exhausts the
  full retry budget and returns the descriptive terminal error (the
  exhaustion-after-N-retries path, not just single-shot).
- `retry_after_empty_injects_continuation_nudge` — first request is unmutated;
  every retry after an empty completion carries the nudge as an *added* user
  message.
- `nudged_retry_recovers` — a nudged retry that succeeds returns that response.
- `non_retryable_error_is_not_retried` — 4xx short-circuits with no retry.

## Open follow-ups

- Confirm gpt-5.5 specificity vs other models (all observed cases were gpt-5.5).
- Once raw bodies are collected in the wild, verify they are truly empty and
  tune the nudge wording if recovery rate is low.
