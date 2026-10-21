//! Opt-in inference quality/tier settings. None preserves the provider's
//! original request shape and pricing; callers may change settings between turns.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl std::str::FromStr for Effort {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::Xhigh),
            "max" => Ok(Self::Max),
            _ => Err("unknown effort level"),
        }
    }
}
impl Effort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InferenceSettings {
    pub effort: Option<Effort>,
    pub fast: bool,
}
