# Get/Put key-namespace contract

Status: normative for the Op and IR runtimes; enforced by
`crates/agent-core/tests/state_keys.rs`.

`Get(key)` and `Put(key, value)` are the agent's interface for context and
state. The interpreter decides *how* each key is backed, but the observable
behavior of the guaranteed namespaces below is part of the runtime contract:
an agent program that uses only these namespaces must behave the same under
any interpreter. If you change a runtime's key behavior, update this document
and the conformance test in the same change.

## Guaranteed namespaces

| Key | Get | Put | Durability |
|---|---|---|---|
| `session:state` | Read the checkpoint JSON (null if no checkpoint is configured or none exists) | Write the checkpoint JSON | Durable: survives process restarts via `SeqConfig::checkpoint_path` |
| `temporal:*` | Return the last value written to the key (the conversation history in the standard agent loops) | Replace the value | Session-lived |
| `semantic:<query>` | Dispatch `<query>` to every QUERY-capable `HydrationSource` and return the list of `SourceResult`s | — (reserved) | Read-only view over registered sources |

Notes:

- `temporal:*` is backed differently per runtime (the Op interpreter stores it
  in the typed interpreter state `S`; the IR interpreter stores it in the
  session store), but the observable put-then-get contract is identical.
- `semantic:` reads are queries, not storage. Writing to `semantic:*` is
  reserved and currently undefined; do not rely on it.

## Keys outside the guaranteed namespaces

This is a **pinned divergence** between the runtimes, not an accident — it is
asserted by the conformance test so it cannot drift silently:

- The **Op runtime** rejects unknown keys with an `unknown Get key` /
  `unknown Put key` error. Its state is a single typed value, so there is
  nowhere to put arbitrary keys.
- The **IR runtime** treats unknown keys as **session-local KV**: `Put` stores
  the value in the session store, `Get` returns the last value written, and a
  key that was never written reads as `null`.

Portable agent programs should stick to the guaranteed namespaces. Programs
that use session-local KV are IR-only by construction.

## Adding a namespace

1. Define the contract here (Get semantics, Put semantics, durability).
2. Implement it in every shipped runtime (`interpreter.rs` `dispatch_get`/
   `dispatch_put`, `ir_interpreter.rs` `InMemoryStore`).
3. Add a conformance test to `tests/state_keys.rs` exercising both runtimes.
