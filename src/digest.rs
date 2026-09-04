use crate::error::{BrainError, BrainResult};
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::borrow::Borrow;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Deref;
use std::path::Path;
use std::str::FromStr;

/// Canonical SHA-256 identity used across CEREBRO authority artifacts.
///
/// The in-memory invariant is exactly 64 lowercase ASCII hexadecimal bytes.
/// JSON remains a plain string, so strengthening the Rust type does not alter
/// persisted wire formats or content-addressed artifact schemas.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub const HEX_LEN: usize = 64;

    pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
        let value = value.as_ref();
        if !Self::is_valid_str(value) {
            return Err(BrainError::Invalid("sha256_digest_invalid".into()));
        }
        Ok(Self(value.to_string()))
    }

    pub fn digest_bytes(bytes: &[u8]) -> Self {
        Self(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn is_valid_str(value: &str) -> bool {
        value.len() == Self::HEX_LEN
            && value.bytes().all(|byte| byte.is_ascii_hexdigit())
            && value.bytes().all(|byte| !byte.is_ascii_uppercase())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }

    pub fn zero() -> Self {
        Self("0".repeat(Self::HEX_LEN))
    }
}

/// Hash an exact file byte stream using CEREBRO's canonical SHA-256 identity.
pub fn sha256_file(path: &Path) -> BrainResult<Sha256Digest> {
    let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(Sha256Digest(format!("{:x}", hasher.finalize())))
}

impl Display for Sha256Digest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl AsRef<str> for Sha256Digest {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Deref for Sha256Digest {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl From<Sha256Digest> for String {
    fn from(value: Sha256Digest) -> Self {
        value.into_string()
    }
}

impl PartialEq<str> for Sha256Digest {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Sha256Digest {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for Sha256Digest {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<Sha256Digest> for String {
    fn eq(&self, other: &Sha256Digest) -> bool {
        self == other.as_str()
    }
}

impl Borrow<str> for Sha256Digest {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl FromStr for Sha256Digest {
    type Err = BrainError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

impl TryFrom<String> for Sha256Digest {
    type Error = BrainError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl TryFrom<&str> for Sha256Digest {
    type Error = BrainError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_requires_canonical_lowercase_without_changing_wire_shape() {
        let lower = "ab".repeat(32);
        let digest = Sha256Digest::parse(&lower).unwrap();
        assert_eq!(digest.as_str(), lower);
        assert!(Sha256Digest::parse("AB".repeat(32)).is_err());
        assert_eq!(
            serde_json::to_string(&digest).unwrap(),
            format!("\"{}\"", lower)
        );
    }

    #[test]
    fn digest_rejects_missing_or_malformed_identity() {
        assert!(Sha256Digest::parse("").is_err());
        assert!(Sha256Digest::parse("g".repeat(64)).is_err());
        assert!(Sha256Digest::parse("a".repeat(63)).is_err());
    }
}
