//! GC zoom markers (t-1609): with the zoom address book present every
//! evicted message is addressed and recovery leads with zoom; without it
//! nothing mentions zoom.

use super::*;
use crate::op::ToolCall;
use crate::zoom::ZoomBook;

/// A history-shaped session: N shell frames with fat results, each
/// followed by a short narration, ending on a user turn.
fn session(frames: usize) -> Vec<ChatMessage> {
    let mut messages = vec![
        ChatMessage::system("system prompt"),
        ChatMessage::user("collect the batch reports"),
    ];
    for index in 1..=frames {
        let id = format!("call-{index}");
        messages.push(ChatMessage::assistant(
            Some(format!("Reading batch {index}.")),
            vec![ToolCall::new(
                &id,
                "shell",
                serde_json::json!({ "command": format!("cat batch-{index}.log") }),
            )],
        ));
        let body: String = (0..30)
            .map(|line| format!("batch-{index} record {line} ingest parser payload ok\n"))
            .collect();
        messages.push(ChatMessage::tool(&id, body));
        messages.push(ChatMessage::assistant(
            Some(format!("Batch {index} has 30 records.")),
            vec![],
        ));
    }
    messages.push(ChatMessage::user("now summarize every batch"));
    messages
}

fn strategies() -> Vec<(&'static str, GcMode)> {
    let mut out = Vec::new();
    for preserve_prefix in [true, false] {
        out.push((
            "ring",
            GcMode::Ring(RingGc {
                preserve_prefix,
                hot_keep: true,
            }),
        ));
        out.push((
            "mark-sweep",
            GcMode::MarkSweep(MarkSweepGc {
                preserve_prefix,
                hot_keep: true,
            }),
        ));
        out.push((
            "stack",
            GcMode::Stack(StackFrameGc {
                preserve_prefix,
                hot_keep: true,
            }),
        ));
        out.push((
            "semantic",
            GcMode::Semantic(SemanticGc {
                preserve_prefix,
                ..Default::default()
            }),
        ));
        out.push((
            "generational",
            GcMode::Generational(GenerationalGc {
                preserve_prefix,
                ..Default::default()
            }),
        ));
    }
    out
}

fn is_annotation(message: &ChatMessage) -> bool {
    message.content.as_deref().is_some_and(|content| {
        let content = content.trim_start();
        content.starts_with(EVICTION_MARKER_PREFIX)
            || content.starts_with("[frame ")
            || content.starts_with(GC_LEDGER_PREFIX)
            || content.contains("\n[frame ")
    })
}

/// Every address an annotation names: `#N`, and every position inside a
/// `#A-#B` span.
fn covered(text: &str) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    let bytes = text.as_bytes();
    let number = |from: usize| -> Option<(usize, usize)> {
        let mut end = from;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        (end > from).then(|| (text[from..end].parse().unwrap(), end))
    };
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'#' {
            if let Some((start, end)) = number(index + 1) {
                if text[end..].starts_with("-#") {
                    if let Some((stop, after)) = number(end + 2) {
                        out.extend(start..=stop);
                        index = after;
                        continue;
                    }
                }
                out.insert(start);
                index = end;
                continue;
            }
        }
        index += 1;
    }
    out
}

#[test]
fn every_evicted_message_is_addressed_under_every_strategy() {
    let history = session(8);
    let book = ZoomBook::from_history(&history);
    for (name, gc) in strategies() {
        for budget in [600, 900, 1400] {
            let mut state = GcState {
                zoom: Some(book.clone()),
                ..Default::default()
            };
            let collected = gc.collect(history.clone(), budget, &mut state);
            if state.marker_summary.suppressed {
                // Terminal rung: no room for even one marker line — the
                // recorded suppression, never silent (docs/GC.md).
                continue;
            }
            let text: String = collected
                .iter()
                .filter(|message| is_annotation(message))
                .filter_map(|message| message.content.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            let addressed = covered(&text);
            let survivors: HashMap<MsgId, Option<&str>> = collected
                .iter()
                .map(|message| (message.id, message.content.as_deref()))
                .collect();
            let mut evicted = 0;
            for (position, message) in history.iter().enumerate() {
                let lost = match survivors.get(&message.id) {
                    None => true,
                    // An elided tool body is evicted in place.
                    Some(now) => message.role == "tool" && *now != message.content.as_deref(),
                };
                if lost {
                    evicted += 1;
                    assert!(
                        addressed.contains(&position),
                        "{name}@{budget}: evicted #{position} ({}) has no address in:\n{text}",
                        message.role
                    );
                }
            }
            assert!(evicted > 0, "{name}@{budget}: nothing evicted");
            assert!(text.contains("zoom #"), "{name}@{budget}: {text}");
        }
    }
}

#[test]
fn without_the_book_no_annotation_mentions_zoom() {
    let history = session(8);
    for (name, gc) in strategies() {
        for budget in [600, 900, 1400] {
            let mut state = GcState::default();
            let collected = gc.collect(history.clone(), budget, &mut state);
            for message in collected.iter().filter(|message| is_annotation(message)) {
                let content = message.content.as_deref().unwrap_or("");
                assert!(
                    !content.contains("zoom") && covered(content).is_empty(),
                    "{name}@{budget}: zoom-off annotation carries zoom text: {content}"
                );
            }
        }
    }
}

/// Zoom changes only GC's own annotations: at a budget where the longer
/// zoom lines shift no sweep decision, every non-annotation message is
/// byte-identical and in the same order.
#[test]
fn zoom_changes_only_annotation_text() {
    let history = session(8);
    let book = ZoomBook::from_history(&history);
    for (name, gc) in strategies() {
        let budget = 1400;
        let mut off_state = GcState::default();
        let off = gc.collect(history.clone(), budget, &mut off_state);
        let mut on_state = GcState {
            zoom: Some(book.clone()),
            ..Default::default()
        };
        let on = gc.collect(history.clone(), budget, &mut on_state);
        let plain = |window: &[ChatMessage]| -> Vec<ChatMessage> {
            window
                .iter()
                .filter(|message| !is_annotation(message))
                .filter(|message| {
                    // Elided bodies are annotations too.
                    !message
                        .content
                        .as_deref()
                        .is_some_and(|content| content.starts_with(EVICTION_MARKER_PREFIX))
                })
                .cloned()
                .collect()
        };
        assert_eq!(
            plain(&off),
            plain(&on),
            "{name}: content outside annotations"
        );
    }
}

#[test]
fn zoom_marker_wording_leads_with_zoom_and_keeps_rerun_as_fallback() {
    let history = session(6);
    let book = ZoomBook::from_history(&history);
    let mut state = GcState {
        zoom: Some(book),
        ..Default::default()
    };
    let collected = RingGc::default().collect(history, 700, &mut state);
    let marker = collected
        .iter()
        .filter(|message| is_eviction_marker(message))
        .filter_map(|message| message.content.as_deref())
        .find(|content| content.contains("shell call-"))
        .unwrap_or_else(|| panic!("a run marker naming a call: {collected:#?}"));
    assert!(
        marker.contains("; recover: zoom #")
            && marker.contains(
                "(exact) — do not guess; re-run a call only if its result may have changed]"
            ),
        "{marker}"
    );
    // Counts still parse first (fusion and absorption read them).
    let count = marker_evicted_count(&marker_message(Uuid::nil(), marker.to_string()));
    assert!(
        marker.starts_with(&format!("[gc: {count} evicted")),
        "{marker}"
    );
}

#[test]
fn escalation_with_zoom_keeps_the_honest_exit_and_the_count() {
    let line = escalation_marker_for("call-7", 3, Some(12));
    assert!(line.contains("cannot stay in context"), "{line}");
    assert!(
        line.contains("zoom #12 for only the part you need"),
        "{line}"
    );
    let message = marker_message(Uuid::nil(), line.clone());
    assert!(is_escalation_marker(&message));
    assert_eq!(marker_evicted_count(&message), 3, "{line}");
    assert_eq!(
        escalation_marker_for("call-7", 3, None),
        escalation_marker_content("call-7", 3),
        "no address = the historical line"
    );
    // An empty provider id is never shown as a handle.
    assert!(escalation_marker_for("", 4, Some(9)).starts_with("[gc: '#9' evicted 4 times"));
}

#[test]
fn fused_zoom_markers_keep_every_address() {
    let earlier = marker_message(
        Uuid::from_u128(1),
        "[gc: 2 evicted — shell call-1 #3; recover: zoom #2-#3 (exact) — do not guess; \
re-run a call only if its result may have changed]"
            .into(),
    );
    let later = marker_message(
        Uuid::from_u128(2),
        "[gc: 1 evicted — user #7; recover: zoom #7 (exact) — do not guess]".into(),
    );
    let fused = fuse_adjacent_markers(vec![earlier.clone(), later.clone()], true);
    assert_eq!(fused.len(), 1);
    let content = fused[0].content.as_deref().unwrap();
    assert!(content.starts_with("[gc: 3 evicted"), "{content}");
    assert!(content.contains("recover: zoom #2-#7 (exact)"), "{content}");
    // Zoom off: the historical fusion (later text wins) is unchanged.
    let plain = fuse_adjacent_markers(vec![earlier, later], false);
    assert_eq!(
        plain[0].content.as_deref().unwrap(),
        "[gc: 3 evicted — user #7; recover: zoom #7 (exact) — do not guess]"
    );
}

#[test]
fn coalesced_line_spans_every_eviction() {
    assert_eq!(
        coalesced_marker_content(41, None),
        "[gc: earlier context compacted — 41 messages evicted; recover: re-run tool calls, \
recall memories, or ask the user — do not guess]"
    );
    let line = coalesced_marker_content(41, Some("#2-#44"));
    assert!(line.contains("recover: zoom #2-#44 (exact)"), "{line}");
    assert_eq!(marker_evicted_count(&marker_message(Uuid::nil(), line)), 41);
}

#[test]
fn frames_and_elisions_name_their_result_address() {
    let history = session(6);
    let book = ZoomBook::from_history(&history);
    let mut frames = Vec::new();
    for budget in [700, 900, 1200, 1600] {
        let mut state = GcState {
            zoom: Some(book.clone()),
            ..Default::default()
        };
        let stack = StackFrameGc {
            preserve_prefix: false,
            hot_keep: true,
        };
        let collected = stack.collect(history.clone(), budget, &mut state);
        // A frame line runs from `[frame ` to its closing `]` (result
        // previews may carry newlines; this fixture's carry no `]`).
        for content in collected
            .iter()
            .filter_map(|message| message.content.as_deref())
        {
            let mut rest = content;
            while let Some(start) = rest.find("[frame ") {
                let end = rest[start..]
                    .find(']')
                    .map_or(rest.len(), |at| start + at + 1);
                frames.push(rest[start..end].to_string());
                rest = &rest[end..];
            }
        }
    }
    assert!(!frames.is_empty(), "stack popped no frame at any budget");
    for frame in &frames {
        // call-K's result is history #3K (system, user, then 3 per frame).
        let call: usize = frame["[frame call-".len()..]
            .split(' ')
            .next()
            .and_then(|k| k.parse().ok())
            .unwrap_or_else(|| panic!("{frame}"));
        let at = format!("#{}", 3 * call);
        assert!(
            frame.starts_with(&format!(
                "[frame call-{call} {at}: shell(cat batch-{call}.log) -> "
            )),
            "{frame}"
        );
        assert!(
            frame.ends_with(&format!(
                "— evicted; recover: zoom {at} (exact); re-run only if the result may have changed]"
            )),
            "{frame}"
        );
    }
    assert_eq!(
        elided_result_content("cat x", "call-2", Some(6)),
        "[gc: result elided — cat x (call-2, #6); recover: zoom #6 (exact) — do not guess; \
re-run a call only if its result may have changed]"
    );
    assert_eq!(
        elided_result_content("cat x", "call-2", None),
        "[gc: result elided — cat x (call-2); recover: re-run the call — do not guess]"
    );
}

#[test]
fn ledger_entries_carry_result_addresses_with_zoom() {
    let history = session(6);
    let book = ZoomBook::from_history(&history);
    let mut state = GcState {
        zoom: Some(book),
        ..Default::default()
    };
    let collected = RingGc::default().collect(history, 900, &mut state);
    let ledger = collected
        .iter()
        .find(|message| is_gc_ledger(message))
        .and_then(|message| message.content.as_deref())
        .unwrap_or_else(|| panic!("a ledger: {collected:#?}"));
    // call-K's result is history #3K.
    assert!(
        ledger.contains("call-3: shell(cat batch-3.log) -> "),
        "{ledger}"
    );
    assert!(ledger.contains("[evicted — zoom #9]"), "{ledger}");
    assert!(
        ledger.ends_with(
            "evicted results are recoverable exactly: zoom the #address — do not guess; \
re-run a call only if its result may have changed."
        ),
        "{ledger}"
    );
}

/// Zoom results re-inject history: with zoom offered they are
/// write-barrier sources (decoded through the page frame); without it a
/// call named `zoom` is an unknown tool and marks nothing.
#[test]
fn zoom_results_feed_the_write_barrier_only_when_offered() {
    let original =
        serde_json::json!({ "stdout": "batch-1 record 0 ingest parser payload ok\n" }).to_string();
    let zoom_call = ChatMessage::assistant(
        None,
        vec![ToolCall::new(
            "call-9",
            crate::zoom::ZOOM_TOOL,
            serde_json::json!({ "address": "#3" }),
        )],
    );
    let page = format!("#3 tool result of shell call-1:\n{original}");
    let window = vec![
        ChatMessage::system("s"),
        zoom_call,
        ChatMessage::tool("call-9", page),
    ];
    let collected_keys = reinjection_chunk_keys(&original);

    let mut on = GcState {
        zoom: Some(ZoomBook::default()),
        collected_hashes: collected_keys.clone(),
        ..Default::default()
    };
    let report = record_reinjection_overlaps(&window, &mut on);
    assert!(report.overlap_events > 0);
    assert!(hot_mask(&window, &on)[2], "the zoom page itself is hot");

    let mut off = GcState {
        collected_hashes: collected_keys,
        ..Default::default()
    };
    let report = record_reinjection_overlaps(&window, &mut off);
    assert_eq!(
        report.overlap_events, 0,
        "the escaped page never matched without zoom"
    );
}
