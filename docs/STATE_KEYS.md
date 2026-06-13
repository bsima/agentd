# Get/Put key-namespace contract — RETIRED

Status: **retired (t-1182).** Get/Put no longer exists.

The stringly-keyed Get/Put interface — with its `temporal:`, `semantic:`,
and `session:state` magic prefixes — was deleted in favor of typed,
sink/source-generic effects. See **[MEMORY.md](MEMORY.md)** for the design
and the migration table; the short version:

| Old key | Replacement |
|---|---|
| `semantic:<query>` Get | the `Retrieve` IR effect (kind = Semantic), or the model's `recall` tool |
| `temporal:*` Get/Put | machine env (history lives in the loop's `history` var); cross-session recall is `Retrieve` (kind = Temporal) |
| `session:state` Get/Put | the passive `ChatHistory` sink, written at turn completion |
| IR session-local KV (unknown keys) | machine env vars |

This file is kept only so existing links resolve.
