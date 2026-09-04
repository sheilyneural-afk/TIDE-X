use crate::error::{BrainError, BrainResult};
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

fn validate_ascii_id(
    value: &str,
    allow_dot: bool,
    forbid_leading_dot: bool,
    label: &str,
) -> BrainResult<()> {
    if value.is_empty()
        || value.len() > 128
        || (forbid_leading_dot && value.starts_with('.'))
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || byte == b'-'
                || byte == b'_'
                || (allow_dot && byte == b'.')
        })
    {
        return Err(BrainError::Invalid(format!("{label}_invalid")));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(String);

impl SessionId {
    pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
        let value = value.as_ref();
        validate_ascii_id(value, false, false, "session_id")?;
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationId(String);

impl ObservationId {
    pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
        let value = value.as_ref();
        validate_ascii_id(value, true, true, "observation_id")?;
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

macro_rules! impl_string_wire {
    ($type:ty) => {
        impl Display for $type {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl AsRef<str> for $type {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl FromStr for $type {
            type Err = BrainError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::parse(value)
            }
        }

        impl TryFrom<String> for $type {
            type Error = BrainError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::parse(value)
            }
        }

        impl Serialize for $type {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $type {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::parse(value).map_err(D::Error::custom)
            }
        }
    };
}

impl_string_wire!(SessionId);
impl_string_wire!(ObservationId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_rejects_path_semantics() {
        assert!(SessionId::parse("session-01").is_ok());
        assert!(SessionId::parse("../session").is_err());
        assert!(SessionId::parse("session.name").is_err());
    }

    #[test]
    fn observation_id_allows_dot_but_not_hidden_or_parent_paths() {
        assert!(ObservationId::parse("obs.v2-01").is_ok());
        assert!(ObservationId::parse(".hidden").is_err());
        assert!(ObservationId::parse("../obs").is_err());
    }
}
