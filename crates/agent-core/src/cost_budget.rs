//! Optional dispatch ceiling. Kept on the trace logger so all Infer paths
//! (including child calls and serial Par branches) share the same rollup.
use crate::trace::{Event, TraceLogger};
use anyhow::{anyhow, Result};
use chrono::Utc;

#[derive(Debug, Clone, Copy)]
pub struct CostBudgetExhausted {
    pub cap: u64,
    pub spent: u64,
}

impl std::fmt::Display for CostBudgetExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cost budget exhausted: spent {} micro-USD of {} micro-USD cap",
            self.spent, self.cap
        )
    }
}
impl std::error::Error for CostBudgetExhausted {}

pub async fn check(trace: &TraceLogger, model: &str, priced: bool, live: bool) -> Result<()> {
    if !live {
        return Ok(());
    }
    let Some(cap) = trace.cost_limit() else {
        return Ok(());
    };
    if trace.has_uncosted_spend() {
        return Err(anyhow!(
            "cost budget cannot continue: a response had unknown cost"
        ));
    }
    let spent = trace.cost_spent();
    if spent >= cap {
        trace
            .emit(&Event::Custom {
                run_id: trace.run_id().into(),
                name: "cost_budget_exhausted".into(),
                data: serde_json::json!({"cap_micro_usd": cap, "spent_micro_usd": spent}),
                timestamp: Utc::now(),
            })
            .await?;
        return Err(CostBudgetExhausted { cap, spent }.into());
    }
    if !priced {
        return Err(anyhow!(
            "--max-cost requires pricing for model {model:?} in models.yaml"
        ));
    }
    Ok(())
}
