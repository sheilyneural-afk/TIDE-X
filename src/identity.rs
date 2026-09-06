use crate::error::{BrainError, BrainResult};
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Borrow;
use std::fmt::{Display, Formatter};
use std::ops::Deref;
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

macro_rules! semantic_id {
    ($name:ident, $label:literal, $allow_dot:expr) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
                let value = value.as_ref();
                validate_ascii_id(value, $allow_dot, true, $label)?;
                if $allow_dot && value.split('.').any(str::is_empty) {
                    return Err(BrainError::Invalid(
                        concat!($label, "_hierarchy_invalid").into(),
                    ));
                }
                let is_raw_sha256 = value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
                if is_raw_sha256 {
                    return Err(BrainError::Invalid(concat!($label, "_is_digest").into()));
                }
                Ok(Self(value.to_string()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl_string_wire!($name);
    };
}

semantic_id!(ModelId, "model_id", true);
semantic_id!(ArchitectureId, "architecture_id", true);
semantic_id!(PatchId, "patch_id", true);
semantic_id!(EvaluationSuiteId, "evaluation_suite_id", true);
semantic_id!(EvaluationCaseId, "evaluation_case_id", true);
semantic_id!(ProbePlanId, "probe_plan_id", true);
semantic_id!(SystemId, "system_id", true);

macro_rules! semantic_non_digest_id {
    ($(#[$metadata:meta])* $name:ident, $label:literal, $allow_dot:expr) => {
        $(#[$metadata])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
                let value = value.as_ref();
                validate_ascii_id(value, $allow_dot, true, $label)?;
                if $allow_dot && value.split('.').any(str::is_empty) {
                    return Err(BrainError::Invalid(
                        concat!($label, "_hierarchy_invalid").into(),
                    ));
                }
                let is_raw_sha256 = value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
                if is_raw_sha256 {
                    return Err(BrainError::Invalid(concat!($label, "_is_digest").into()));
                }
                Ok(Self(value.to_string()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl_string_wire!($name);
    };
}

semantic_non_digest_id!(
    /// Stable identity of one declared learning objective, not its content hash.
    LearningTargetId,
    "learning_target_id",
    true
);
semantic_non_digest_id!(
    /// Stable identity of one protection or sensitivity probe.
    ProbeId,
    "probe_id",
    true
);
semantic_non_digest_id!(
    /// Stable identity of one user-authorized acquisition request.
    ///
    /// This names an acquisition; it is never a filesystem path or the
    /// content commitment of its manifest.
    AcquisitionId,
    "acquisition_id",
    true
);
semantic_non_digest_id!(
    /// Stable identity of one governed residency decision round.
    ResidencyDecisionRoundId,
    "residency_decision_round_id",
    true
);
semantic_non_digest_id!(
    /// Stable identity of a concrete residency target/profile, never a digest.
    ResidencyTargetId,
    "residency_target_id",
    true
);
semantic_non_digest_id!(
    /// Stable name of a primitive admitted by a TIDE-X capability IR profile.
    PrimitiveId,
    "primitive_id",
    true
);
semantic_non_digest_id!(
    /// Stable identity of an authority-approved primitive profile.
    PrimitiveProfileId,
    "primitive_profile_id",
    true
);
semantic_non_digest_id!(
    /// Stable typed port name within a capability contract.
    PortId,
    "port_id",
    false
);
semantic_non_digest_id!(
    /// Stable identity of one node in a sealed capability IR graph.
    CapabilityNodeId,
    "capability_node_id",
    true
);
semantic_non_digest_id!(
    /// Stable identity of a governed investigation of one capability.
    InquiryId,
    "inquiry_id",
    true
);
semantic_non_digest_id!(
    /// Stable name of one competing explanation inside an investigation.
    HypothesisId,
    "hypothesis_id",
    true
);
semantic_non_digest_id!(
    /// Stable name of one governed cognitive action.
    CognitiveActionId,
    "cognitive_action_id",
    true
);

macro_rules! semantic_assignable_id {
    ($(#[$metadata:meta])* $name:ident, $label:literal) => {
        $(#[$metadata])*
        #[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
                let value = value.as_ref();
                if value.is_empty() {
                    return Ok(Self::unassigned());
                }
                validate_ascii_id(value, true, true, $label)?;
                if value.split('.').any(str::is_empty) {
                    return Err(BrainError::Invalid(concat!($label, "_hierarchy_invalid").into()));
                }
                let is_raw_sha256 = value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
                if is_raw_sha256 {
                    return Err(BrainError::Invalid(concat!($label, "_is_digest").into()));
                }
                Ok(Self(value.to_string()))
            }

            pub fn unassigned() -> Self {
                Self(String::new())
            }

            pub fn is_unassigned(&self) -> bool {
                self.0.is_empty()
            }

            pub fn is_empty(&self) -> bool {
                self.is_unassigned()
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl_string_wire!($name);
    };
}

semantic_assignable_id!(
    /// Identity of one exact reconstruction, with an empty legacy/unassigned wire state.
    ///
    /// ```compile_fail
    /// use cerebro_tidex::identity::{LineageId, ReconstructionId};
    /// let reconstruction = ReconstructionId::parse("reconstruction-a").unwrap();
    /// let _: LineageId = reconstruction;
    /// ```
    ReconstructionId,
    "reconstruction_id"
);
semantic_assignable_id!(
    /// Durable lineage anchor, with an empty legacy/unassigned wire state.
    LineageId,
    "lineage_id"
);
semantic_non_digest_id!(
    /// Stable identity of one experimental aperture in a learning design.
    ///
    /// ```compile_fail
    /// use cerebro_tidex::identity::{ApertureId, ProbeId};
    /// let aperture = ApertureId::parse("isolate-001").unwrap();
    /// let _: ProbeId = aperture;
    /// ```
    ApertureId,
    "aperture_id",
    false
);

/// Durable identity of one reconstructed capability field.
///
/// Skill IDs are compact symbolic identifiers such as `cap-a18f...` or
/// `skill-g3-002`. They are neither filesystem paths nor content digests; the
/// latter have dedicated authenticated types and must never be accepted as a
/// capability identity by dimensional coincidence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SkillId(String);

impl SkillId {
    pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
        let value = value.as_ref();
        validate_ascii_id(value, false, true, "skill_id")?;
        let is_raw_sha256 = value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if is_raw_sha256 {
            return Err(BrainError::Invalid("skill_id_is_digest".into()));
        }
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl_string_wire!(SkillId);

impl PartialEq<str> for SkillId {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for SkillId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for SkillId {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl Deref for SkillId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl Borrow<str> for SkillId {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

/// Canonical identity of one tensor inside an admitted parameter inventory.
///
/// A tensor identity is a dotted hierarchy (for example
/// `layers.7.self_attn.q_proj.weight`), not an arbitrary label, path, or
/// digest.  Keeping the JSON representation as a string makes this a
/// wire-compatible strengthening of existing contracts.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(String);

impl TensorId {
    pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
        let value = value.as_ref();
        if value.is_empty()
            || value.len() > 512
            || !value.contains('.')
            || value.split('.').any(|segment| {
                segment.is_empty()
                    || !segment
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
            })
        {
            return Err(BrainError::Invalid("tensor_id_invalid".into()));
        }
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl_string_wire!(TensorId);

impl PartialEq<str> for TensorId {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for TensorId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for TensorId {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

/// Canonical capability identity. TIDE-X accepts native dotted IDs and typed
/// external-authority version suffixes without reducing either form to an
/// unchecked string.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapabilityId(String);

impl CapabilityId {
    pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
        let value = value.as_ref();
        if value.len() > 128 || value.matches(':').count() > 1 {
            return Err(BrainError::Invalid("capability_id_invalid".into()));
        }
        let (base, version) = value
            .split_once(':')
            .map_or((value, None), |(base, version)| (base, Some(version)));
        validate_ascii_id(base, true, true, "capability_id")?;
        if base.split('.').any(str::is_empty) {
            return Err(BrainError::Invalid(
                "capability_id_hierarchy_invalid".into(),
            ));
        }
        let base_is_raw_sha256 = base.len() == 64
            && base
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if base_is_raw_sha256 {
            return Err(BrainError::Invalid("capability_id_is_digest".into()));
        }
        if version.is_some_and(|version| !is_canonical_capability_version(version)) {
            return Err(BrainError::Invalid("capability_id_invalid".into()));
        }
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl_string_wire!(CapabilityId);

fn is_canonical_capability_version(version: &str) -> bool {
    let Some(version) = version.strip_prefix('v') else {
        return false;
    };
    let mut build_split = version.split('+');
    let Some(without_build) = build_split.next() else {
        return false;
    };
    let build = build_split.next();
    if build_split.next().is_some()
        || build.is_some_and(|value| !valid_version_labels(value, false))
    {
        return false;
    }
    let (core, prerelease) = without_build
        .split_once('-')
        .map_or((without_build, None), |(core, labels)| (core, Some(labels)));
    if prerelease.is_some_and(|value| !valid_version_labels(value, true)) {
        return false;
    }
    let components = core.split('.').collect::<Vec<_>>();
    !components.is_empty()
        && components.len() <= 3
        && components.iter().all(|component| {
            !component.is_empty()
                && component.bytes().all(|byte| byte.is_ascii_digit())
                && (component.len() == 1 || !component.starts_with('0'))
        })
}

fn valid_version_labels(value: &str, reject_numeric_leading_zero: bool) -> bool {
    !value.is_empty()
        && value.split('.').all(|label| {
            !label.is_empty()
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                && (!reject_numeric_leading_zero
                    || !label.bytes().all(|byte| byte.is_ascii_digit())
                    || label.len() == 1
                    || !label.starts_with('0'))
        })
}

/// Full lowercase Git object identity. Both SHA-1 and SHA-256 repository
/// formats are accepted; symbolic refs such as branches remain forbidden.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceRevision(String);

impl SourceRevision {
    pub fn parse(value: impl AsRef<str>) -> BrainResult<Self> {
        let value = value.as_ref();
        if !matches!(value.len(), 40 | 64)
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(BrainError::Invalid("source_revision_invalid".into()));
        }
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl_string_wire!(SourceRevision);

#[cfg(test)]
macro_rules! impl_valid_test_literal {
    ($type:ty) => {
        impl From<&str> for $type {
            fn from(value: &str) -> Self {
                Self::parse(value).expect("test identity literal must be valid")
            }
        }
    };
}

#[cfg(test)]
impl_valid_test_literal!(ApertureId);
#[cfg(test)]
impl_valid_test_literal!(ProbeId);
#[cfg(test)]
impl_valid_test_literal!(ReconstructionId);
#[cfg(test)]
impl_valid_test_literal!(LineageId);

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

    #[test]
    fn model_binding_ids_reject_path_semantics() {
        assert!(ModelId::parse("model.v1").is_ok());
        assert!(ArchitectureId::parse("transformer-v2").is_ok());
        assert!(PatchId::parse("skill.alpha").is_ok());
        assert!(ModelId::parse("../model").is_err());
        assert!(ArchitectureId::parse("/").is_err());
        assert!(PatchId::parse(".hidden").is_err());
        assert!(ModelId::parse("ab".repeat(32)).is_err());
    }

    #[test]
    fn capability_id_accepts_typed_external_versions_without_path_semantics() {
        assert!(CapabilityId::parse("semantic-retrieval.v1").is_ok());
        assert!(CapabilityId::parse("external.memory.semantic.retrieve:v1").is_ok());
        assert!(CapabilityId::parse("external.memory::v1").is_err());
        assert!(CapabilityId::parse("../memory:v1").is_err());
        assert!(CapabilityId::parse("memory:v1:forged").is_err());
        assert!(CapabilityId::parse("memory..short:v1").is_err());
        assert!(CapabilityId::parse("memory.:v1").is_err());
        assert!(CapabilityId::parse("ab".repeat(32)).is_err());
        assert!(CapabilityId::parse("memory:1").is_err());
        assert!(CapabilityId::parse("memory:v01").is_err());
        assert!(CapabilityId::parse("memory:v1.2.3").is_ok());
        assert!(CapabilityId::parse("memory:v1.2.3-rc.1+linux-x86-64").is_ok());
        assert!(CapabilityId::parse("memory:v1.2.3+linux_x86").is_err());
        assert!(CapabilityId::parse("memory:v1.02.3").is_err());
    }

    #[test]
    fn source_revision_requires_a_full_lowercase_git_object_id() {
        assert!(SourceRevision::parse("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3").is_ok());
        assert!(SourceRevision::parse("ab".repeat(32)).is_ok());
        assert!(SourceRevision::parse("main").is_err());
        assert!(SourceRevision::parse("97B0C614BE4D77EE51C0CEF4E5F07C00F9EB65B3").is_err());
    }

    #[test]
    fn tensor_id_is_a_hierarchy_not_a_path_digest_or_generic_label() {
        let id = TensorId::parse("layers.7.self_attn.q_proj.weight").unwrap();
        assert_eq!(id.as_str(), "layers.7.self_attn.q_proj.weight");
        assert!(TensorId::parse("weight").is_err());
        assert!(TensorId::parse("layers..weight").is_err());
        assert!(TensorId::parse("layers/7/weight").is_err());
        assert!(TensorId::parse("ab".repeat(32)).is_err());
        assert!(serde_json::from_str::<TensorId>("\"../weights\"").is_err());
        assert_eq!(
            serde_json::to_string(&id).unwrap(),
            "\"layers.7.self_attn.q_proj.weight\""
        );
    }

    #[test]
    fn skill_id_wire_rejects_crossed_path_and_digest_identities() {
        let id = SkillId::parse("cap-a18f03").unwrap();
        assert_eq!(serde_json::to_string(&id).unwrap(), "\"cap-a18f03\"");
        assert_eq!(
            serde_json::from_str::<SkillId>("\"cap-a18f03\"").unwrap(),
            id
        );

        // Tensor/model hierarchy, filesystem paths and naked object digests
        // belong to other semantic domains.
        assert!(serde_json::from_str::<SkillId>("\"layers.0.weight\"").is_err());
        assert!(serde_json::from_str::<SkillId>("\"../cap-a18f03\"").is_err());
        assert!(serde_json::from_str::<SkillId>(&format!("\"{}\"", "a".repeat(64))).is_err());
    }

    #[test]
    fn learning_and_aperture_ids_reject_paths_and_content_digests() {
        assert!(LearningTargetId::parse("semantic-retrieval.v1").is_ok());
        assert!(ApertureId::parse("pair-001-002").is_ok());
        assert!(LearningTargetId::parse("../target").is_err());
        assert!(ApertureId::parse("aperture/name").is_err());
        assert!(LearningTargetId::parse("a".repeat(64)).is_err());
        assert!(ApertureId::parse("b".repeat(64)).is_err());
        assert!(serde_json::from_str::<ApertureId>(&format!("\"{}\"", "c".repeat(64))).is_err());
    }

    #[test]
    fn reconstruction_lineage_and_probe_ids_preserve_only_the_declared_empty_sentinel() {
        assert!(ReconstructionId::unassigned().is_unassigned());
        assert!(LineageId::parse("").unwrap().is_unassigned());
        assert!(ReconstructionId::parse("reconstruction-a.1").is_ok());
        assert!(LineageId::parse("lineage-a.1").is_ok());
        assert!(ProbeId::parse("sensitivity-pc-001").is_ok());
        for invalid in ["../identity", "/absolute", ".hidden"] {
            assert!(ReconstructionId::parse(invalid).is_err());
            assert!(LineageId::parse(invalid).is_err());
            assert!(ProbeId::parse(invalid).is_err());
        }
        assert!(ReconstructionId::parse("a".repeat(64)).is_err());
        assert!(LineageId::parse("b".repeat(64)).is_err());
        assert!(ProbeId::parse("c".repeat(64)).is_err());
        assert!(ReconstructionId::parse("reconstruction..part").is_err());
        assert!(LineageId::parse("lineage.").is_err());
        assert!(ProbeId::parse("probe..segment").is_err());
    }
}
