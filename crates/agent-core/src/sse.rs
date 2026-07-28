//! Incremental Server-Sent-Events decoder for streaming provider responses.
//!
//! Feed raw response-body chunks as they arrive; complete events come back
//! as soon as their terminating blank line lands. Handles events split
//! across chunk boundaries (including mid-UTF-8-codepoint splits: the
//! buffer is bytes, and lines convert lossily only once complete).

/// One decoded SSE event: the optional `event:` name and the joined `data:`
/// payload (multiple data lines join with `\n`, per the SSE spec).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume a body chunk, returning every event completed by it.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.buf.iter().position(|&byte| byte == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=newline).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if let Some(event) = self.take_event() {
                    events.push(event);
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix(':') {
                let _ = rest; // comment line, ignored
                continue;
            }
            let (field, value) = match line.split_once(':') {
                Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
                None => (line, ""),
            };
            match field {
                "event" => self.event = Some(value.to_owned()),
                "data" => self.data.push(value.to_owned()),
                _ => {}
            }
        }
        events
    }

    fn take_event(&mut self) -> Option<SseEvent> {
        if self.event.is_none() && self.data.is_empty() {
            return None;
        }
        Some(SseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_events_split_across_chunks() {
        let mut decoder = SseDecoder::new();
        assert!(decoder
            .feed(b"event: message_start\ndata: {\"a\"")
            .is_empty());
        let events = decoder.feed(b":1}\n\ndata: [DONE]\n\n");
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: Some("message_start".into()),
                    data: r#"{"a":1}"#.into(),
                },
                SseEvent {
                    event: None,
                    data: "[DONE]".into(),
                },
            ]
        );
    }

    #[test]
    fn joins_multiple_data_lines_and_handles_crlf() {
        let mut decoder = SseDecoder::new();
        let events = decoder.feed(b"data: one\r\ndata: two\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "one\ntwo");
    }

    #[test]
    fn ignores_comments_and_blank_keepalives() {
        let mut decoder = SseDecoder::new();
        assert!(decoder.feed(b": keepalive\n\n\n\n").is_empty());
    }

    #[test]
    fn survives_utf8_split_across_chunks() {
        let mut decoder = SseDecoder::new();
        let text = "data: héllo\n\n".as_bytes();
        // Split in the middle of the two-byte é.
        let split = text.iter().position(|&b| b == 0xc3).unwrap() + 1;
        assert!(decoder.feed(&text[..split]).is_empty());
        let events = decoder.feed(&text[split..]);
        assert_eq!(events[0].data, "héllo");
    }
}
