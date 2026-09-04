use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum BrainError {
    Invalid(String),
    Integrity(String),
    Numerical(String),
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl Display for BrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid:{s}"),
            Self::Integrity(s) => write!(f, "integrity:{s}"),
            Self::Numerical(s) => write!(f, "numerical:{s}"),
            Self::Io(e) => write!(f, "io:{e}"),
            Self::Json(e) => write!(f, "json:{e}"),
        }
    }
}
impl std::error::Error for BrainError {}
impl From<std::io::Error> for BrainError {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<serde_json::Error> for BrainError {
    fn from(v: serde_json::Error) -> Self {
        Self::Json(v)
    }
}

pub type BrainResult<T> = Result<T, BrainError>;
