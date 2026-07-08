//! Runs the `custom_source` example's round-trip in CI. `cargo test`
//! *builds* examples but never executes them; including the example file
//! here drives the exact same code (one source of truth, no drift).

#[path = "../examples/custom_source.rs"]
#[allow(dead_code)] // the example's `main` is unused when included as a module
mod example;

/// The out-of-tree WorkspaceSource answers the model's `recall` tool call:
/// the retrieved workspace content (with file provenance) lands in the
/// recorded Retrieve result the model read, and the run completes with the
/// scripted answer.
#[tokio::test]
async fn custom_source_example_round_trips() {
    let (text, retrieved) = example::run_demo().await.expect("example round-trip");
    assert!(
        retrieved.contains("run migrations before restarting the API"),
        "retrieved: {retrieved}"
    );
    assert!(retrieved.contains("deploy.md"), "retrieved: {retrieved}");
    assert!(
        !retrieved.contains("taqueria"),
        "distractor file should not match: {retrieved}"
    );
    assert_eq!(
        text,
        "Deploy checklist: run migrations before restarting the API."
    );
}
