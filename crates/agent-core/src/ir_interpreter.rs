use crate::interpreter::{
    hydrate_infer_prompt, millis_u64, prompt_preview, response_preview, run_eval, SeqConfig,
};
use crate::ir::{
    effect_location, program_hash, validate_program, BlockId, DynamicPath, EffectKind,
    EffectLocation, EffectSite, EvalRequest, Expr, Instr, Machine, MatchArm, Pattern, ProgramHash,
    PromptRef, Terminator, Var,
};
use crate::op::{ChatMessage, Model, Prompt};
use crate::trace::Event;
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InMemoryStore {
    values: BTreeMap<String, Value>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Value {
        self.values.get(key).cloned().unwrap_or(Value::Null)
    }

    pub fn put(&mut self, key: impl Into<String>, value: Value) {
        self.values.insert(key.into(), value);
    }
}

pub async fn run_ir_sequential(config: &SeqConfig, machine: Machine) -> Result<(Value, Machine)> {
    let mut store = InMemoryStore::new();
    run_ir_sequential_with_store(config, machine, &mut store).await
}

pub async fn run_ir_sequential_with_store(
    config: &SeqConfig,
    mut machine: Machine,
    store: &mut InMemoryStore,
) -> Result<(Value, Machine)> {
    validate_program(&machine.program)?;
    let program_hash = program_hash(&machine.program)?;
    let mut site_visits = HashMap::<EffectSite, u64>::new();

    loop {
        let block = machine
            .program
            .blocks
            .get(&machine.block)
            .with_context(|| format!("unknown AgentIR block {:?}", machine.block))?
            .clone();

        if machine.pc < block.instructions.len() {
            let site = EffectSite {
                block: machine.block,
                instruction_index: machine.pc,
            };
            let dynamic_path = DynamicPath::with_visit(site, next_visit(&mut site_visits, site));
            let instr = block.instructions[machine.pc].clone();
            execute_instr(
                config,
                &mut machine,
                store,
                &program_hash,
                site,
                dynamic_path,
                instr,
            )
            .await?;
            machine.pc += 1;
            continue;
        }

        match block.terminator {
            Terminator::Return { value } => {
                let value = eval_expr(&machine.env, &value)?;
                return Ok((value, machine));
            }
            Terminator::Goto { block, args } => {
                goto_block(&mut machine, block, args).await?;
            }
            Terminator::If {
                cond,
                then_block,
                else_block,
            } => {
                let cond = eval_expr(&machine.env, &cond)?;
                let target = match cond {
                    Value::Bool(true) => then_block,
                    Value::Bool(false) => else_block,
                    other => return Err(anyhow!("AgentIR If condition must be bool, got {other}")),
                };
                goto_block(&mut machine, target, vec![]).await?;
            }
            Terminator::Match {
                value,
                arms,
                default,
            } => {
                let value = eval_expr(&machine.env, &value)?;
                let target = match_match_arms(&value, &arms).or(default).ok_or_else(|| {
                    anyhow!("AgentIR Match had no matching arm and no default for {value}")
                })?;
                goto_block(&mut machine, target, vec![]).await?;
            }
            Terminator::Par { .. } => {
                return Err(anyhow!(
                    "AgentIR Par terminator is not implemented in run_ir_sequential yet"
                ));
            }
        }
    }
}

async fn execute_instr(
    config: &SeqConfig,
    machine: &mut Machine,
    store: &mut InMemoryStore,
    program_hash: &ProgramHash,
    site: EffectSite,
    dynamic_path: DynamicPath,
    instr: Instr,
) -> Result<()> {
    match instr {
        Instr::Let { out, expr } => {
            let value = eval_expr(&machine.env, &expr)?;
            machine.env.insert(out, value);
        }
        Instr::Infer {
            out,
            model,
            prompt,
            policy: _,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Infer,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let model = string_expr(&machine.env, &model, "Infer.model")?;
            let prompt = resolve_prompt(&machine.env, prompt)?;
            let prompt = hydrate_infer_prompt(config, &Value::Null, prompt).await?;
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::InferCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    model: model.clone(),
                    prompt: Some(prompt.clone()),
                    prompt_preview: prompt_preview(&prompt),
                    timestamp: Utc::now(),
                })
                .await?;
            let started = Instant::now();
            let response = match &config.replay {
                Some(replay) => replay.infer_result(op_id, &model)?,
                None => {
                    config
                        .provider
                        .chat(&Model(model), &config.tool_specs(), &prompt)
                        .await?
                }
            };
            config
                .trace
                .emit(&Event::InferResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    response: Some(response.clone()),
                    response_preview: response_preview(&response),
                    tokens: response.tokens,
                    duration_ms: millis_u64(started.elapsed()),
                    timestamp: Utc::now(),
                })
                .await?;
            machine.env.insert(out, serde_json::to_value(response)?);
        }
        Instr::Eval {
            out,
            request,
            policy: _,
        } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Eval,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let command = match request {
                EvalRequest::Shell { command } => {
                    string_expr(&machine.env, &command, "Eval.command")?
                }
            };
            let op_id = config.trace.next_op_id();
            config
                .trace
                .emit(&Event::EvalCall {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    command: command.clone(),
                    cwd: config
                        .eval
                        .cwd
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    env_policy: config.eval.env.label(),
                    timeout_ms: millis_u64(config.eval.timeout),
                    timestamp: Utc::now(),
                })
                .await?;
            let result = match &config.replay {
                Some(replay) => replay.eval_result(op_id, &command)?,
                None => run_eval(&config.eval, &command).await?,
            };
            let truncated_stdout = result
                .get("stdout_truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let truncated_stderr = result
                .get("stderr_truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let duration_ms = result
                .get("duration_ms")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            config
                .trace
                .emit(&Event::EvalResult {
                    run_id: config.trace.run_id().into(),
                    op_id,
                    command,
                    result: result.clone(),
                    duration_ms,
                    truncated_stdout,
                    truncated_stderr,
                    timestamp: Utc::now(),
                })
                .await?;
            machine.env.insert(out, result);
        }
        Instr::Get { out, key } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Get,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let key = string_expr(&machine.env, &key, "Get.key")?;
            let value = store.get(&key);
            machine.env.insert(out, value);
        }
        Instr::Put { key, value } => {
            let location = effect_location(
                program_hash.clone(),
                EffectKind::Put,
                site,
                dynamic_path.clone(),
            )?;
            emit_ir_effect(config, &location).await?;
            let key = string_expr(&machine.env, &key, "Put.key")?;
            let value = eval_expr(&machine.env, &value)?;
            store.put(key, value);
        }
        Instr::Emit { event } => {
            let location =
                effect_location(program_hash.clone(), EffectKind::Emit, site, dynamic_path)?;
            emit_ir_effect(config, &location).await?;
            let value = eval_expr(&machine.env, &event)?;
            let event: Event =
                serde_json::from_value(value).context("decoding AgentIR Emit event")?;
            config.trace.emit(&event).await?;
        }
    }
    Ok(())
}

async fn goto_block(machine: &mut Machine, block_id: BlockId, args: Vec<Expr>) -> Result<()> {
    let target = machine
        .program
        .blocks
        .get(&block_id)
        .with_context(|| format!("unknown AgentIR block {block_id:?}"))?;
    if target.params.len() != args.len() {
        return Err(anyhow!(
            "AgentIR Goto to {:?} expected {} args, got {}",
            block_id,
            target.params.len(),
            args.len()
        ));
    }
    let mut env = BTreeMap::new();
    for (param, arg) in target.params.iter().cloned().zip(args) {
        env.insert(param, eval_expr(&machine.env, &arg)?);
    }
    machine.block = block_id;
    machine.pc = 0;
    machine.env = env;
    Ok(())
}

fn next_visit(site_visits: &mut HashMap<EffectSite, u64>, site: EffectSite) -> u64 {
    let visit = site_visits.entry(site).or_insert(0);
    let current = *visit;
    *visit += 1;
    current
}

async fn emit_ir_effect(config: &SeqConfig, location: &EffectLocation) -> Result<()> {
    config
        .trace
        .emit(&Event::Custom {
            run_id: config.trace.run_id().into(),
            name: "ir_effect".into(),
            data: serde_json::to_value(location)?,
            timestamp: Utc::now(),
        })
        .await
}

fn eval_expr(env: &BTreeMap<Var, Value>, expr: &Expr) -> Result<Value> {
    match expr {
        Expr::Value(value) => Ok(value.clone()),
        Expr::Var(var) => env
            .get(var)
            .cloned()
            .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", var)),
        Expr::Field { base, field } => {
            let value = env
                .get(base)
                .ok_or_else(|| anyhow!("unknown AgentIR var {:?}", base))?;
            value
                .get(field)
                .cloned()
                .ok_or_else(|| anyhow!("AgentIR field {field:?} not found on {value}"))
        }
        Expr::Array(items) => items
            .iter()
            .map(|item| eval_expr(env, item))
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        Expr::Object(fields) => {
            let mut object = serde_json::Map::new();
            for (key, expr) in fields {
                object.insert(key.clone(), eval_expr(env, expr)?);
            }
            Ok(Value::Object(object))
        }
    }
}

fn string_expr(env: &BTreeMap<Var, Value>, expr: &Expr, label: &str) -> Result<String> {
    match eval_expr(env, expr)? {
        Value::String(value) => Ok(value),
        other => Err(anyhow!("AgentIR {label} must be string, got {other}")),
    }
}

fn resolve_prompt(env: &BTreeMap<Var, Value>, prompt: PromptRef) -> Result<Prompt> {
    match prompt {
        PromptRef::Inline(prompt) => Ok(prompt),
        PromptRef::Var(var) => {
            let value = env
                .get(&var)
                .cloned()
                .ok_or_else(|| anyhow!("unknown AgentIR prompt var {:?}", var))?;
            serde_json::from_value::<Vec<ChatMessage>>(value).context("decoding AgentIR prompt")
        }
    }
}

fn match_match_arms(value: &Value, arms: &[MatchArm]) -> Option<BlockId> {
    arms.iter()
        .find(|arm| pattern_matches(value, &arm.pattern))
        .map(|arm| arm.block)
}

fn pattern_matches(value: &Value, pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Null => value.is_null(),
        Pattern::Bool(expected) => value.as_bool() == Some(*expected),
        Pattern::String(expected) => value.as_str() == Some(expected.as_str()),
        Pattern::Number(expected) => value.as_number() == Some(expected),
        Pattern::ObjectField { field, pattern } => value
            .get(field)
            .is_some_and(|value| pattern_matches(value, pattern)),
        Pattern::Any => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::{PassiveHydrationConfig, SourceRegistry};
    use crate::interpreter::{EvalConfig, SeqConfig};
    use crate::op::{Response, ResponseToolCall};
    use crate::provider::{ChatProvider, ToolSpec};
    use crate::trace::TraceLogger;
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    struct MockProvider {
        responses: Mutex<Vec<Response>>,
        prompts: Mutex<Vec<Prompt>>,
    }

    impl MockProvider {
        fn new(mut responses: Vec<Response>) -> Self {
            responses.reverse();
            Self {
                responses: Mutex::new(responses),
                prompts: Mutex::new(Vec::new()),
            }
        }

        fn prompts(&self) -> Vec<Prompt> {
            self.prompts.lock().unwrap().clone()
        }

        fn prompt_count(&self) -> usize {
            self.prompts.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        async fn chat(
            &self,
            _model: &Model,
            _tools: &[ToolSpec],
            messages: &[ChatMessage],
        ) -> Result<Response> {
            self.prompts.lock().unwrap().push(messages.to_vec());
            self.responses
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| anyhow!("mock provider exhausted"))
        }
    }

    fn response(content: &str) -> Response {
        Response {
            content: content.into(),
            tool_calls: Vec::<ResponseToolCall>::new(),
            tokens: 1,
        }
    }

    fn test_trace() -> TraceLogger {
        let path = std::env::temp_dir().join(format!("agent-ir-test-{}.jsonl", Uuid::new_v4()));
        TraceLogger::new(Uuid::new_v4().to_string(), path)
    }

    fn config(provider: Arc<dyn ChatProvider>) -> SeqConfig {
        SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace: test_trace(),
            eval: EvalConfig::default(),
            replay: None,
        }
    }

    fn config_with_trace(provider: Arc<dyn ChatProvider>, trace: TraceLogger) -> SeqConfig {
        SeqConfig {
            provider,
            hydration: SourceRegistry::new(),
            passive_hydration: PassiveHydrationConfig::default(),
            checkpoint_path: None,
            trace,
            eval: EvalConfig::default(),
            replay: None,
        }
    }

    #[tokio::test]
    async fn ir_runs_infer_then_infer_without_rust_continuations() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![
            response("first"),
            response("second"),
        ]));
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![
                    Instr::Infer {
                        out: Var("a".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Inline(vec![ChatMessage::user("first prompt")]),
                        policy: Default::default(),
                    },
                    Instr::Let {
                        out: Var("a_content".into()),
                        expr: Expr::Field {
                            base: Var("a".into()),
                            field: "content".into(),
                        },
                    },
                    Instr::Let {
                        out: Var("second_prompt".into()),
                        expr: Expr::Array(vec![Expr::Object(BTreeMap::from([
                            ("role".into(), Expr::Value(Value::String("user".into()))),
                            ("content".into(), Expr::Var(Var("a_content".into()))),
                        ]))]),
                    },
                    Instr::Infer {
                        out: Var("b".into()),
                        model: Expr::Value(Value::String("mock".into())),
                        prompt: PromptRef::Var(Var("second_prompt".into())),
                        policy: Default::default(),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Field {
                        base: Var("b".into()),
                        field: "content".into(),
                    },
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("infer-infer".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let (value, _machine) = run_ir_sequential(&config(provider.clone()), machine).await?;

        assert_eq!(value, Value::String("second".into()));
        let prompts = provider.prompts();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[1][0].content.as_deref(), Some("first"));
        Ok(())
    }

    #[tokio::test]
    async fn ir_effect_metadata_is_stable_and_visit_sensitive() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let trace = test_trace();
        let trace_path = trace.path().clone();
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Get {
                    out: Var("a".into()),
                    key: Expr::Value(Value::String("missing".into())),
                }],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("a".into())),
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("effect-ids".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let _ = run_ir_sequential(&config_with_trace(provider, trace), machine).await?;
        let events = TraceLogger::read_events(trace_path).await?;
        let locations = events
            .iter()
            .filter_map(|event| match event {
                Event::Custom { name, data, .. } if name == "ir_effect" => {
                    Some(serde_json::from_value::<EffectLocation>(data.clone()))
                }
                _ => None,
            })
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].kind, EffectKind::Get);
        assert_eq!(locations[0].site.block, BlockId(0));
        assert_eq!(locations[0].site.instruction_index, 0);
        Ok(())
    }

    #[tokio::test]
    async fn ir_get_put_use_interpreter_store_not_machine_state() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![]));
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![
                    Instr::Put {
                        key: Expr::Value(Value::String("answer".into())),
                        value: Expr::Value(Value::Number(42.into())),
                    },
                    Instr::Get {
                        out: Var("value".into()),
                        key: Expr::Value(Value::String("answer".into())),
                    },
                ],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("value".into())),
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("store".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };
        let mut store = InMemoryStore::new();

        let (value, _machine) =
            run_ir_sequential_with_store(&config(provider), machine, &mut store).await?;

        assert_eq!(value, Value::Number(42.into()));
        assert_eq!(store.get("answer"), Value::Number(42.into()));
        Ok(())
    }

    #[tokio::test]
    async fn ir_validation_runs_before_effects() -> Result<()> {
        let provider = Arc::new(MockProvider::new(vec![response("should-not-run")]));
        let mut blocks = BTreeMap::new();
        blocks.insert(
            BlockId(0),
            crate::ir::Block {
                params: vec![],
                instructions: vec![Instr::Infer {
                    out: Var("response".into()),
                    model: Expr::Value(Value::String("mock".into())),
                    prompt: PromptRef::Inline(vec![ChatMessage::user("do not run")]),
                    policy: Default::default(),
                }],
                terminator: Terminator::Return {
                    value: Expr::Var(Var("missing".into())),
                },
            },
        );
        let machine = Machine {
            program: crate::ir::Program {
                id: crate::ir::ProgramId("invalid".into()),
                entry: BlockId(0),
                blocks,
            },
            block: BlockId(0),
            pc: 0,
            env: BTreeMap::new(),
            continuation_stack: vec![],
            budgets: Default::default(),
        };

        let err = run_ir_sequential(&config(provider.clone()), machine)
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("used before definition"));
        assert_eq!(provider.prompt_count(), 0);
        Ok(())
    }
}
