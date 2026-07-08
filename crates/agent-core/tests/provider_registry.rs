//! Doc-adjacent pin for docs/PROVIDERS.md: the three in-tree providers
//! register through the same public `SourceRegistry` API the guide
//! documents for out-of-tree providers — `register` for a source-only
//! provider, `register_backend` for a source+sink pair — and expose the
//! names, kinds, capabilities, and write policies the guide's table
//! states. If this test stops compiling or passing, the guide is lying.

use agent_core::{
    ChatHistory, MemorySource, SinkWritePolicy, SourceCapability, SourceKind, SourceRegistry,
    TemporalSource,
};
use std::path::PathBuf;

fn scratch_dir() -> PathBuf {
    // Registration does no IO; the paths just have to exist as values.
    std::env::temp_dir().join("agent-core-provider-registry-test")
}

#[test]
fn in_tree_providers_register_through_the_public_api() {
    let dir = scratch_dir();
    let registry = SourceRegistry::new()
        // Both halves via one object (memory: markdown dir + index +
        // optional embeddings) ...
        .register_backend(MemorySource::new(dir.clone()))
        // ... both halves (checkpoints as sink, recency window as source) ...
        .register_backend(ChatHistory::new(dir.clone()))
        // ... and a read-only source (archived checkpoint dirs).
        .register(TemporalSource::new(dir));

    let source_names: Vec<&str> = registry
        .sources()
        .iter()
        .map(|source| source.name())
        .collect();
    assert_eq!(
        source_names,
        ["memory", "chat-history", "temporal-checkpoints"]
    );

    let sink_names: Vec<&str> = registry.sinks().iter().map(|sink| sink.name()).collect();
    assert_eq!(sink_names, ["memory", "chat-history"]);

    // Kinds and dispatch capabilities as documented: memory answers
    // Semantic queries (what `recall` targets); the temporal pair answers
    // Temporal queries and passive session-context hydration.
    let memory = &registry.sources()[0];
    assert_eq!(memory.kind(), SourceKind::Semantic);
    assert!(memory.capabilities().contains(SourceCapability::QUERY));

    for temporal in &registry.sources()[1..] {
        assert_eq!(temporal.kind(), SourceKind::Temporal);
        assert!(temporal.capabilities().contains(SourceCapability::QUERY));
        assert!(temporal
            .capabilities()
            .contains(SourceCapability::SESSION_CONTEXT));
    }

    // Sink lookup by registered name is the Store effect's dispatch, and
    // the memory sink's write policy is Free (trace-visible; docs/MEMORY.md
    // settled question 1).
    let memory_sink = registry.sink("memory").expect("memory sink registered");
    assert_eq!(memory_sink.write_policy(), SinkWritePolicy::Free);
    assert!(registry.sink("no-such-sink").is_none());

    // Kind-scoped sink listing (what `remember` uses to find memory sinks).
    let semantic_sinks = registry.sinks_of_kind(SourceKind::Semantic);
    assert_eq!(semantic_sinks.len(), 1);
    assert_eq!(semantic_sinks[0].name(), "memory");
}
