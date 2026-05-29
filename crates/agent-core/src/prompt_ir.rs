use crate::hydration::{SourceKind, SourceResult};
use crate::op::{ChatMessage, Prompt};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PromptId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SectionId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenEstimate(pub usize);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptIR {
    pub id: PromptId,
    #[serde(default)]
    pub base_messages: Prompt,
    #[serde(default)]
    pub sections: Vec<Section>,
    #[serde(default)]
    pub tools: Vec<ToolDef>,
    pub observation: Option<Observation>,
    pub meta: PromptMeta,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub id: SectionId,
    pub label: String,
    pub source: SectionSource,
    pub role: SectionRole,
    pub content: String,
    pub tokens: TokenEstimate,
    pub priority: Priority,
    pub composition: CompositionMode,
    pub relevance: Option<f32>,
    pub recency: Option<DateTime<Utc>>,
    pub hash: ContentHash,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub schema: Value,
    pub priority: Priority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptMeta {
    pub total_tokens: TokenEstimate,
    pub budget: TokenBudget,
    pub strategy: ContextStrategy,
    pub timestamp: DateTime<Utc>,
    pub prompt_hash: ContentHash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalMode {
    Temporal,
    Semantic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalTiming {
    Passive,
    Active,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionSource {
    pub origin: SectionOrigin,
    pub timing: RetrievalTiming,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SectionOrigin {
    Static {
        name: String,
    },
    Retrieval {
        backend: String,
        mode: RetrievalMode,
        query: Option<String>,
        key: Option<String>,
        score: Option<f32>,
    },
    State {
        key: String,
    },
    User,
    ToolResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SectionRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Context,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CompositionMode {
    Hierarchical,
    Constraint,
    Additive,
    Contextual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenBudget {
    pub total: usize,
    pub reserve_ratio: f32,
    pub allocation: BudgetAllocation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BudgetAllocation {
    FixedRatios {
        system: f32,
        context: f32,
        observation: f32,
    },
    RelevanceWeighted,
    InformationWeighted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextStrategy {
    pub temporal_window: usize,
    pub semantic_limit: usize,
    pub semantic_threshold: f32,
    #[serde(default)]
    pub enabled_backends: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextRequest {
    pub observation: String,
    pub goal: Option<String>,
    pub strategy: ContextStrategy,
    pub budget: TokenBudget,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionSummary {
    pub section_id: SectionId,
    pub label: String,
    pub source: SectionSource,
    pub role: SectionRole,
    pub priority: Priority,
    pub composition: CompositionMode,
    pub tokens: TokenEstimate,
    pub hash: ContentHash,
    pub content_preview: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptIRTrace {
    pub prompt_id: PromptId,
    pub prompt_hash: ContentHash,
    pub total_tokens: TokenEstimate,
    pub budget: TokenBudget,
    pub strategy: ContextStrategy,
    pub sections: Vec<SectionSummary>,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            total: 8_000,
            reserve_ratio: 0.15,
            allocation: BudgetAllocation::FixedRatios {
                system: 0.70,
                context: 0.20,
                observation: 0.10,
            },
        }
    }
}

impl Default for ContextStrategy {
    fn default() -> Self {
        Self {
            temporal_window: 30,
            semantic_limit: 10,
            semantic_threshold: 0.65,
            enabled_backends: Vec::new(),
        }
    }
}

impl PromptIR {
    pub fn new(base_messages: Prompt, sections: Vec<Section>) -> Result<Self> {
        let total_tokens = TokenEstimate(
            sections
                .iter()
                .map(|section| section.tokens.0)
                .sum::<usize>()
                + estimate_prompt_tokens(&base_messages),
        );
        let budget = TokenBudget::default();
        let strategy = ContextStrategy::default();
        let prompt_hash = prompt_hash_parts(&base_messages, &sections, &budget, &strategy)?;
        Ok(Self {
            id: PromptId(prompt_hash.0.clone()),
            base_messages,
            sections,
            tools: Vec::new(),
            observation: None,
            meta: PromptMeta {
                total_tokens,
                budget,
                strategy,
                timestamp: Utc::now(),
                prompt_hash,
            },
        })
    }

    pub fn trace(&self, include_content: bool) -> PromptIRTrace {
        PromptIRTrace {
            prompt_id: self.id.clone(),
            prompt_hash: self.meta.prompt_hash.clone(),
            total_tokens: self.meta.total_tokens,
            budget: self.meta.budget.clone(),
            strategy: self.meta.strategy.clone(),
            sections: self
                .sections
                .iter()
                .map(|section| SectionSummary {
                    section_id: section.id.clone(),
                    label: section.label.clone(),
                    source: section.source.clone(),
                    role: section.role,
                    priority: section.priority,
                    composition: section.composition,
                    tokens: section.tokens,
                    hash: section.hash.clone(),
                    content_preview: preview(&section.content, 512),
                    content: include_content.then(|| section.content.clone()),
                })
                .collect(),
        }
    }
}

impl Section {
    pub fn passive_temporal(
        id: impl Into<String>,
        label: impl Into<String>,
        content: String,
    ) -> Self {
        Self::new(
            id,
            label,
            SectionSource {
                origin: SectionOrigin::Retrieval {
                    backend: "session".into(),
                    mode: RetrievalMode::Temporal,
                    query: None,
                    key: Some("temporal:history".into()),
                    score: None,
                },
                timing: RetrievalTiming::Passive,
                metadata: Value::Null,
            },
            SectionRole::Context,
            content,
            Priority::High,
            CompositionMode::Contextual,
            None,
            Value::Null,
        )
    }

    pub fn from_source_result(
        id: impl Into<String>,
        result: SourceResult,
        timing: RetrievalTiming,
        query: Option<String>,
    ) -> Self {
        let mode = match result.kind {
            SourceKind::Temporal => RetrievalMode::Temporal,
            SourceKind::Semantic | SourceKind::Knowledge => RetrievalMode::Semantic,
        };
        let score = result
            .metadata
            .get("score")
            .and_then(Value::as_f64)
            .map(|score| score as f32);
        Self::new(
            id,
            result.source.clone(),
            SectionSource {
                origin: SectionOrigin::Retrieval {
                    backend: result.source.clone(),
                    mode,
                    query,
                    key: None,
                    score,
                },
                timing,
                metadata: result.metadata.clone(),
            },
            SectionRole::Context,
            result.content,
            Priority::Medium,
            CompositionMode::Additive,
            score,
            result.metadata,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        source: SectionSource,
        role: SectionRole,
        content: String,
        priority: Priority,
        composition: CompositionMode,
        relevance: Option<f32>,
        metadata: Value,
    ) -> Self {
        let hash = content_hash(&content);
        Self {
            id: SectionId(id.into()),
            label: label.into(),
            source,
            role,
            tokens: TokenEstimate(estimate_tokens(&content)),
            content,
            priority,
            composition,
            relevance,
            recency: None,
            hash,
            metadata,
        }
    }
}

pub fn compile_prompt_ir(ir: &PromptIR) -> Prompt {
    let mut prompt = ir.base_messages.clone();
    let mut context_sections = Vec::new();
    let mut direct_messages = Vec::new();

    for section in ordered_sections(&ir.sections) {
        match section.role {
            SectionRole::System | SectionRole::Developer => {
                context_sections.push(render_section(section));
            }
            SectionRole::Context => context_sections.push(render_section(section)),
            SectionRole::User => direct_messages.push(ChatMessage::user(section.content.clone())),
            SectionRole::Assistant => direct_messages.push(ChatMessage::assistant(
                Some(section.content.clone()),
                Vec::new(),
            )),
            SectionRole::Tool => direct_messages.push(ChatMessage::tool(
                section.id.0.clone(),
                section.content.clone(),
            )),
        }
    }

    if !context_sections.is_empty() {
        inject_context_sections(&mut prompt, context_sections.join("\n\n"));
    }
    prompt.extend(direct_messages);
    if let Some(observation) = &ir.observation {
        prompt.push(ChatMessage::user(observation.content.clone()));
    }
    prompt
}

fn ordered_sections(sections: &[Section]) -> Vec<&Section> {
    let mut indexed = sections.iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by_key(|(index, section)| (section.composition, *index));
    indexed.into_iter().map(|(_, section)| section).collect()
}

fn render_section(section: &Section) -> String {
    format!("## {}\n{}", section.label, section.content)
}

fn inject_context_sections(prompt: &mut Prompt, context: String) {
    let block = format!("Hydrated context:\n\n{context}");
    if let Some(system) = prompt.iter_mut().find(|message| message.role == "system") {
        match &mut system.content {
            Some(content) if !content.is_empty() => {
                content.push_str("\n\n");
                content.push_str(&block);
            }
            _ => system.content = Some(block),
        }
    } else {
        prompt.insert(0, ChatMessage::system(block));
    }
}

fn prompt_hash_parts(
    base_messages: &Prompt,
    sections: &[Section],
    budget: &TokenBudget,
    strategy: &ContextStrategy,
) -> Result<ContentHash> {
    #[derive(Serialize)]
    struct StablePromptHash<'a> {
        base_messages: &'a Prompt,
        sections: &'a [Section],
        budget: &'a TokenBudget,
        strategy: &'a ContextStrategy,
    }

    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(&StablePromptHash {
        base_messages,
        sections,
        budget,
        strategy,
    })?);
    Ok(ContentHash(format!("sha256:{:x}", hasher.finalize())))
}

fn content_hash(content: &str) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    ContentHash(format!("sha256:{:x}", hasher.finalize()))
}

fn estimate_prompt_tokens(prompt: &Prompt) -> usize {
    prompt
        .iter()
        .map(|message| estimate_tokens(message.content.as_deref().unwrap_or_default()))
        .sum()
}

fn estimate_tokens(content: &str) -> usize {
    content
        .split_whitespace()
        .count()
        .max(content.len() / 4)
        .max(1)
}

fn preview(content: &str, max_chars: usize) -> String {
    content.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_ir_round_trips_through_json() -> Result<()> {
        let ir = PromptIR::new(
            vec![ChatMessage::system("system")],
            vec![Section::passive_temporal(
                "recent",
                "Recent conversation",
                "hello".into(),
            )],
        )?;
        let encoded = serde_json::to_string(&ir)?;
        let decoded: PromptIR = serde_json::from_str(&encoded)?;
        assert_eq!(decoded.base_messages, ir.base_messages);
        assert_eq!(decoded.sections, ir.sections);
        Ok(())
    }

    #[test]
    fn compile_injects_context_into_system_message() -> Result<()> {
        let ir = PromptIR::new(
            vec![ChatMessage::system("system")],
            vec![Section::passive_temporal(
                "recent",
                "Recent conversation",
                "hello".into(),
            )],
        )?;

        let prompt = compile_prompt_ir(&ir);

        assert_eq!(prompt.len(), 1);
        assert!(prompt[0].content.as_deref().unwrap().contains("system"));
        assert!(prompt[0]
            .content
            .as_deref()
            .unwrap()
            .contains("Hydrated context"));
        assert!(prompt[0].content.as_deref().unwrap().contains("hello"));
        Ok(())
    }

    #[test]
    fn trace_can_include_full_content() -> Result<()> {
        let ir = PromptIR::new(
            vec![],
            vec![Section::passive_temporal(
                "recent",
                "Recent",
                "secret".into(),
            )],
        )?;

        assert_eq!(ir.trace(false).sections[0].content, None);
        assert_eq!(
            ir.trace(true).sections[0].content.as_deref(),
            Some("secret")
        );
        Ok(())
    }
}
