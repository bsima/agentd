use crate::op::{Op, OpF};
use crate::provider::{ProviderClient, ToolFunctionSpec, ToolSpec};
use crate::trace::{Event, TraceLogger};
use anyhow::{anyhow, Result};
use async_recursion::async_recursion;
use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub type ToolMap = HashMap<String, Arc<dyn Tool>>;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn execute(&self, args: Value) -> Result<Value>;
}

pub struct SeqConfig {
    pub provider: ProviderClient,
    pub tools: ToolMap,
    pub trace: TraceLogger,
}

impl SeqConfig {
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|tool| ToolSpec {
                kind: "function".into(),
                function: ToolFunctionSpec {
                    name: tool.name().into(),
                    description: tool.description().into(),
                    parameters: tool.parameters(),
                },
            })
            .collect()
    }
}

#[async_recursion]
pub async fn run_sequential<S, A>(config: &SeqConfig, state: S, op: Op<S, A>) -> Result<(A, S)>
where
    S: Clone + Send + 'static,
    A: Send + 'static,
{
    match *op.0 {
        OpF::Pure(value) => Ok((value, state)),
        OpF::Infer {
            model,
            prompt,
            next,
        } => {
            config
                .trace
                .emit(&Event::InferStart {
                    run_id: config.trace.run_id().into(),
                    model: model.0.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            let response = config
                .provider
                .chat(&model, &config.tool_specs(), &prompt)
                .await?;
            config
                .trace
                .emit(&Event::InferEnd {
                    run_id: config.trace.run_id().into(),
                    tokens: response.tokens,
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential(config, state, next(response)).await
        }
        OpF::Tool { name, args, next } => {
            config
                .trace
                .emit(&Event::ToolCall {
                    run_id: config.trace.run_id().into(),
                    name: name.clone(),
                    args: args.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            let tool = config
                .tools
                .get(&name)
                .ok_or_else(|| anyhow!("unknown tool: {name}"))?;
            let result = tool.execute(args).await?;
            config
                .trace
                .emit(&Event::ToolResult {
                    run_id: config.trace.run_id().into(),
                    name,
                    result: result.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            run_sequential(config, state, next(result)).await
        }
        OpF::Get { next } => run_sequential(config, state.clone(), next(state.clone())).await,
        OpF::Put {
            state: new_state,
            next,
        } => run_sequential(config, new_state, next).await,
        OpF::Emit { event, next } => {
            config.trace.emit(&event).await?;
            run_sequential(config, state, next).await
        }
        OpF::Par { ops, next } => {
            let mut values = Vec::with_capacity(ops.len());
            let mut current_state = state;
            for op in ops {
                let (value, new_state) = run_sequential(config, current_state, op).await?;
                values.push(value);
                current_state = new_state;
            }
            run_sequential(config, current_state, next(values)).await
        }
    }
}
