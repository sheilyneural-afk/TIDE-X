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

    /// Hash a domain-separated payload in exactly one SHA-256 round.
    pub fn digest_domain(domain: &[u8], payload: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        hasher.update(payload);
        Self(format!("{:x}", hasher.finalize()))
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

macro_rules! semantic_digest {
    ($(#[$metadata:meta])* $name:ident) => {
        $(#[$metadata])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256Digest);
        impl $name {
            pub fn as_digest(&self) -> &Sha256Digest {
                &self.0
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }

            pub fn zero() -> Self {
                Self(Sha256Digest::zero())
            }
        }
        impl From<Sha256Digest> for $name {
            fn from(value: Sha256Digest) -> Self {
                Self(value)
            }
        }
        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.as_str() == other
            }
        }
        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.as_str() == *other
            }
        }
        impl PartialEq<String> for $name {
            fn eq(&self, other: &String) -> bool {
                self.as_str() == other
            }
        }
        impl PartialEq<$name> for String {
            fn eq(&self, other: &$name) -> bool {
                self == other.as_str()
            }
        }
        impl PartialEq<$name> for str {
            fn eq(&self, other: &$name) -> bool {
                self == other.as_str()
            }
        }
        impl PartialEq<Sha256Digest> for $name {
            fn eq(&self, other: &Sha256Digest) -> bool {
                self.as_digest() == other
            }
        }
        impl PartialEq<$name> for Sha256Digest {
            fn eq(&self, other: &$name) -> bool {
                self == other.as_digest()
            }
        }
        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(formatter)
            }
        }
        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
        impl Borrow<str> for $name {
            fn borrow(&self) -> &str {
                self.as_str()
            }
        }
        impl Deref for $name {
            type Target = str;

            fn deref(&self) -> &Self::Target {
                self.as_str()
            }
        }
        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                self.0.serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                Ok(Self(Sha256Digest::deserialize(deserializer)?))
            }
        }
    };
}

/// A semantic commitment whose value may only be minted by the authority that
/// computes that domain.  Unlike the older compatibility wrapper above, this
/// type deliberately provides no conversion from a raw SHA-256 value, no
/// dereference to `str`, and no cross-domain comparisons.  Deserialization is
/// still supported because persisted bytes are untrusted input; consumers
/// must authenticate the enclosing artifact before treating the value as
/// authoritative.
macro_rules! define_sealed_semantic_digest {
    ($(#[$metadata:meta])* $name:ident { $($state_method:item)* }) => {
        $(#[$metadata])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256Digest);

        impl $name {
            pub(crate) fn from_computed(value: Sha256Digest) -> Self {
                Self(value)
            }

            $($state_method)*

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                self.0.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                Ok(Self(Sha256Digest::deserialize(deserializer)?))
            }
        }
    };
}

macro_rules! sealed_semantic_digest {
    ($(#[$metadata:meta])* $name:ident) => {
        define_sealed_semantic_digest! {
            $(#[$metadata])*
            $name {
                pub(crate) fn draft_marker() -> Self {
                    Self(Sha256Digest::zero())
                }

                pub fn is_draft(&self) -> bool {
                    self.0 == Sha256Digest::zero()
                }
            }
        }
    };
}

macro_rules! sealed_semantic_digest_without_draft {
    ($(#[$metadata:meta])* $name:ident) => {
        define_sealed_semantic_digest! {
            $(#[$metadata])*
            $name {}
        }
    };
}

semantic_digest!(ParameterLayoutDigest);
semantic_digest!(WeightDigest);
semantic_digest!(ManifestCommitmentDigest);

semantic_digest!(
    /// Exact byte identity of a model checkpoint.
    ///
    /// Domain digests are intentionally incompatible even though their JSON
    /// wire form remains the same lowercase SHA-256 string.
    ///
    /// ```compile_fail
    /// use cerebro_tidex::digest::{CheckpointDigest, EvaluationReceiptDigest};
    /// let checkpoint = CheckpointDigest::zero();
    /// let _: EvaluationReceiptDigest = checkpoint;
    /// ```
    CheckpointDigest
);
semantic_digest!(EvaluationSuiteDigest);
semantic_digest!(EvaluationReceiptDigest);
semantic_digest!(ProbeProtocolDigest);
semantic_digest!(ProbeEvidenceDigest);
semantic_digest!(ActivationEvidenceDigest);
semantic_digest!(
    /// Commitment to one complete learning target contract.
    ///
    /// ```compile_fail
    /// use cerebro_tidex::digest::{AdaptiveLearningPolicyDigest, LearningTargetDigest};
    /// let target = LearningTargetDigest::zero();
    /// let _: AdaptiveLearningPolicyDigest = target;
    /// ```
    LearningTargetDigest
);
semantic_digest!(AdaptiveLearningPolicyDigest);
semantic_digest!(AdaptiveLearningReceiptDigest);
semantic_digest!(LearningEvidenceDigest);
semantic_digest!(ControllerDatasetDigest);
semantic_digest!(LearnedControllerPolicyDigest);
semantic_digest!(LearnedControllerReceiptDigest);
semantic_digest!(SkillBankDigest);
semantic_digest!(SkillFieldSetDigest);
semantic_digest!(
    /// Exact serialized-byte identity of one `DeltaObservation` record.
    ///
    /// ```compile_fail
    /// use cerebro_tidex::digest::{ObservationRecordDigest, ProvenanceDigest};
    /// let observation = ObservationRecordDigest::zero();
    /// let _: ProvenanceDigest = observation;
    /// ```
    ObservationRecordDigest
);
semantic_digest!(ProvenanceDigest);
semantic_digest!(
    /// Exact canonical byte identity of a sealed representation protocol.
    ///
    /// ```compile_fail
    /// use cerebro_tidex::digest::{RepresentationProtocolDigest, RepresentationRequestDigest};
    /// let protocol = RepresentationProtocolDigest::zero();
    /// let _: RepresentationRequestDigest = protocol;
    /// ```
    RepresentationProtocolDigest
);
semantic_digest!(RepresentationRequestDigest);
semantic_digest!(
    /// Identity of the canonical observation corpus used by one reconstruction.
    ///
    /// ```compile_fail
    /// use cerebro_tidex::digest::{CorpusDigest, ConfigDigest};
    /// let corpus = CorpusDigest::zero();
    /// let _: ConfigDigest = corpus;
    /// ```
    CorpusDigest
);
semantic_digest!(ConfigDigest);
semantic_digest!(SourceTreeDigest);
semantic_digest!(AnalysisVersionDigest);
semantic_digest!(ReportDigest);
semantic_digest!(MemoryDigest);
semantic_digest!(EvidenceBundleDigest);
semantic_digest!(CanonicalEngineHeadDigest);
semantic_digest!(CausalCreditDigest);
semantic_digest!(ProtectedMapDigest);
sealed_semantic_digest!(
    /// Manifest identity minted only by the acquisition authority.
    ///
    /// Raw hashes and other semantic digest domains cannot be laundered into
    /// an acquisition identity through the public API.
    ///
    /// ```compile_fail
    /// use cerebro_tidex::digest::{AcquisitionRequestDigest, Sha256Digest};
    /// let raw = Sha256Digest::digest_bytes(b"untrusted");
    /// let _forged = AcquisitionRequestDigest::from(raw);
    /// ```
    AcquisitionRequestDigest
);
sealed_semantic_digest!(SystemEnvelopeDigest);
sealed_semantic_digest!(CapabilityIrDigest);
sealed_semantic_digest!(CapabilityBundleDigest);
sealed_semantic_digest!(CaptureReceiptDigest);
sealed_semantic_digest_without_draft!(ResidencyPolicyDigest);
sealed_semantic_digest!(ResidencyPrecommitDigest);
sealed_semantic_digest!(ResidencyDecisionDigest);

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

    #[test]
    fn domain_hash_uses_one_round_with_a_known_cross_language_vector() {
        // SHA-256("abc"), split deliberately as domain + payload.  This known
        // answer catches the former accidental SHA256(SHA256(...)) convention.
        assert_eq!(
            Sha256Digest::digest_domain(b"a", b"bc").as_str(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(AcquisitionRequestDigest::draft_marker().is_draft());
    }

    #[test]
    fn semantic_digest_domains_are_distinct_but_wire_compatible() {
        use std::any::TypeId;

        let raw = Sha256Digest::parse("ab".repeat(32)).unwrap();
        let checkpoint = CheckpointDigest::from(raw.clone());
        let receipt = EvaluationReceiptDigest::from(raw.clone());
        let protocol = ProbeProtocolDigest::from(raw);
        assert_ne!(
            TypeId::of::<CheckpointDigest>(),
            TypeId::of::<EvaluationReceiptDigest>()
        );
        assert_ne!(
            TypeId::of::<EvaluationReceiptDigest>(),
            TypeId::of::<ProbeProtocolDigest>()
        );
        assert_ne!(
            TypeId::of::<LearningTargetDigest>(),
            TypeId::of::<AdaptiveLearningPolicyDigest>()
        );
        assert_ne!(
            TypeId::of::<ObservationRecordDigest>(),
            TypeId::of::<ProvenanceDigest>()
        );
        assert_ne!(TypeId::of::<CorpusDigest>(), TypeId::of::<ConfigDigest>());
        assert_ne!(TypeId::of::<ReportDigest>(), TypeId::of::<MemoryDigest>());
        assert_ne!(
            TypeId::of::<CausalCreditDigest>(),
            TypeId::of::<ProtectedMapDigest>()
        );
        assert_ne!(
            TypeId::of::<RepresentationProtocolDigest>(),
            TypeId::of::<RepresentationRequestDigest>()
        );
        assert_eq!(
            serde_json::to_string(&checkpoint).unwrap(),
            serde_json::to_string(&receipt).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&receipt).unwrap(),
            serde_json::to_string(&protocol).unwrap()
        );
        assert!(
            serde_json::from_str::<CheckpointDigest>(&format!("\"{}\"", "AB".repeat(32))).is_err()
        );
        assert!(serde_json::from_str::<EvaluationReceiptDigest>("\"tampered\"").is_err());
        assert!(serde_json::from_str::<ObservationRecordDigest>("\"../observation\"").is_err());
        assert!(
            serde_json::from_str::<RepresentationProtocolDigest>(&format!(
                "\"{}\"",
                "AB".repeat(32)
            ))
            .is_err()
        );
        let learning = LearningTargetDigest::from(Sha256Digest::parse("cd".repeat(32)).unwrap());
        let wire = serde_json::to_string(&learning).unwrap();
        assert_eq!(
            serde_json::from_str::<LearningTargetDigest>(&wire).unwrap(),
            learning
        );
        assert!(
            serde_json::from_str::<AdaptiveLearningPolicyDigest>(&format!(
                "\"{}\"",
                "CD".repeat(32)
            ))
            .is_err()
        );
    }
}
