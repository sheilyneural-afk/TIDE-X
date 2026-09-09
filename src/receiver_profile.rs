//! Receiver facts and fail-closed planning for the capability laboratory.
//!
//! This is intentionally a planning boundary.  A plan is not an actuator and
//! every plan is shadow-only; no type in this module can write receiver
//! parameters or activate a candidate.

use crate::capability_ir::CapabilityIr;
use crate::digest::{CapabilityIrDigest, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::{ArchitectureId, CapabilityId, ModelId, TensorId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverArchitecture {
    Transformer,
    StateSpace,
    MixtureOfExperts,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityModality {
    Text,
    Vision,
    Audio,
    Tool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationStrategy {
    ReceiverCoordinates,
    LowRank,
    DenseDelta,
    SparseDelta,
    ActivationSteering,
    Routing,
    ExternalMemory,
    NewModule,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverRegion {
    pub tensor_id: TensorId,
    pub parameter_count: u64,
    pub supported_strategies: BTreeSet<MaterializationStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverProfile {
    pub schema: String,
    pub model_id: ModelId,
    pub architecture_id: ArchitectureId,
    pub architecture: ReceiverArchitecture,
    pub modalities: BTreeSet<CapabilityModality>,
    pub supports_persistent_state: bool,
    pub parameter_dimension: u64,
    pub regions: Vec<ReceiverRegion>,
}

impl ReceiverProfile {
    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.receiver_profile/v1"
            || self.parameter_dimension == 0
            || self.regions.is_empty()
        {
            return Err(BrainError::Invalid("receiver_profile_invalid".into()));
        }
        let mut prior = None::<&TensorId>;
        let mut total = 0u64;
        for region in &self.regions {
            if region.parameter_count == 0
                || region.supported_strategies.is_empty()
                || prior.is_some_and(|previous| previous >= &region.tensor_id)
            {
                return Err(BrainError::Invalid(
                    "receiver_profile_regions_invalid".into(),
                ));
            }
            total = total
                .checked_add(region.parameter_count)
                .ok_or_else(|| BrainError::Invalid("receiver_profile_parameter_overflow".into()))?;
            prior = Some(&region.tensor_id);
        }
        if total != self.parameter_dimension {
            return Err(BrainError::Invalid(
                "receiver_profile_parameter_dimension_mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn digest(&self) -> BrainResult<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:RECEIVER-PROFILE:v1\0",
            &serde_json::to_vec(self)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityRequirements {
    pub schema: String,
    pub capability_id: CapabilityId,
    pub capability_ir_sha256: CapabilityIrDigest,
    pub required_modalities: BTreeSet<CapabilityModality>,
    pub requires_persistent_state: bool,
    pub minimum_receiver_parameter_dimension: u64,
    pub acceptable_strategies: BTreeSet<MaterializationStrategy>,
}

impl CapabilityRequirements {
    pub fn validate_against(&self, ir: &CapabilityIr) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.capability_requirements/v1"
            || self.capability_id != *ir.capability_id()
            || self.capability_ir_sha256 != *ir.manifest_digest()
            || self.acceptable_strategies.is_empty()
        {
            return Err(BrainError::Invalid(
                "capability_requirements_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityDisposition {
    Compatible,
    Partial,
    Incompatible,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityReason {
    MissingModality,
    PersistentStateUnavailable,
    ReceiverTooSmall,
    NoSharedMaterialization,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityAssessment {
    pub schema: String,
    pub receiver_profile_sha256: Sha256Digest,
    pub disposition: CompatibilityDisposition,
    pub reasons: BTreeSet<CompatibilityReason>,
    pub available_strategies: BTreeSet<MaterializationStrategy>,
}

pub fn assess_compatibility(
    profile: &ReceiverProfile,
    requirements: &CapabilityRequirements,
) -> BrainResult<CompatibilityAssessment> {
    profile.validate()?;
    if requirements.schema != "cerebro.tidex.capability_requirements/v1"
        || requirements.acceptable_strategies.is_empty()
    {
        return Err(BrainError::Invalid(
            "capability_requirements_invalid".into(),
        ));
    }
    let available_strategies = profile
        .regions
        .iter()
        .flat_map(|region| region.supported_strategies.iter().copied())
        .filter(|strategy| requirements.acceptable_strategies.contains(strategy))
        .collect::<BTreeSet<_>>();
    let mut reasons = BTreeSet::new();
    if !requirements
        .required_modalities
        .is_subset(&profile.modalities)
    {
        reasons.insert(CompatibilityReason::MissingModality);
    }
    if requirements.requires_persistent_state && !profile.supports_persistent_state {
        reasons.insert(CompatibilityReason::PersistentStateUnavailable);
    }
    if profile.parameter_dimension < requirements.minimum_receiver_parameter_dimension {
        reasons.insert(CompatibilityReason::ReceiverTooSmall);
    }
    if available_strategies.is_empty() {
        reasons.insert(CompatibilityReason::NoSharedMaterialization);
    }
    let disposition = if reasons.contains(&CompatibilityReason::MissingModality)
        || reasons.contains(&CompatibilityReason::NoSharedMaterialization)
    {
        CompatibilityDisposition::Incompatible
    } else if reasons.is_empty() {
        CompatibilityDisposition::Compatible
    } else {
        CompatibilityDisposition::Partial
    };
    Ok(CompatibilityAssessment {
        schema: "cerebro.tidex.compatibility_assessment/v1".into(),
        receiver_profile_sha256: profile.digest()?,
        disposition,
        reasons,
        available_strategies,
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanLifecycle {
    ShadowOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MaterializationPlan {
    pub schema: String,
    pub capability_id: CapabilityId,
    pub capability_ir_sha256: CapabilityIrDigest,
    pub compilation_request_sha256: Sha256Digest,
    pub receiver_profile_sha256: Sha256Digest,
    pub strategy: MaterializationStrategy,
    pub affected_regions: Vec<TensorId>,
    pub lifecycle: PlanLifecycle,
}

impl MaterializationPlan {
    /// Re-admit a serialized plan against the exact receiver profile before a
    /// backend consumes it. Plan construction is not treated as permanent
    /// authority because persisted or transported bytes may be hostile.
    pub fn validate_against_profile(&self, profile: &ReceiverProfile) -> BrainResult<()> {
        profile.validate()?;
        if self.schema != "cerebro.tidex.materialization_plan/v1"
            || self.receiver_profile_sha256 != profile.digest()?
            || self.lifecycle != PlanLifecycle::ShadowOnly
            || self.affected_regions.is_empty()
            || self
                .affected_regions
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(BrainError::Invalid("materialization_plan_invalid".into()));
        }
        for affected in &self.affected_regions {
            let region = profile
                .regions
                .iter()
                .find(|candidate| candidate.tensor_id == *affected)
                .ok_or_else(|| BrainError::Invalid("materialization_plan_region_unknown".into()))?;
            if !region.supported_strategies.contains(&self.strategy) {
                return Err(BrainError::Invalid(
                    "materialization_plan_strategy_unsupported".into(),
                ));
            }
        }
        Ok(())
    }
}

pub fn create_shadow_plan(
    profile: &ReceiverProfile,
    assessment: &CompatibilityAssessment,
    requirements: &CapabilityRequirements,
    compilation_request_sha256: Sha256Digest,
    strategy: MaterializationStrategy,
    affected_regions: Vec<TensorId>,
) -> BrainResult<MaterializationPlan> {
    profile.validate()?;
    if assessment.schema != "cerebro.tidex.compatibility_assessment/v1"
        || assessment.receiver_profile_sha256 != profile.digest()?
        || assessment.disposition != CompatibilityDisposition::Compatible
        || !assessment.available_strategies.contains(&strategy)
        || affected_regions.is_empty()
    {
        return Err(BrainError::Invalid(
            "materialization_plan_not_admissible".into(),
        ));
    }
    let declared = profile
        .regions
        .iter()
        .map(|region| &region.tensor_id)
        .collect::<BTreeSet<_>>();
    if affected_regions.windows(2).any(|pair| pair[0] >= pair[1])
        || affected_regions
            .iter()
            .any(|region| !declared.contains(region))
        || affected_regions.iter().any(|region| {
            !profile
                .regions
                .iter()
                .find(|candidate| candidate.tensor_id == *region)
                .is_some_and(|candidate| candidate.supported_strategies.contains(&strategy))
        })
    {
        return Err(BrainError::Invalid(
            "materialization_plan_regions_invalid".into(),
        ));
    }
    let plan = MaterializationPlan {
        schema: "cerebro.tidex.materialization_plan/v1".into(),
        capability_id: requirements.capability_id.clone(),
        capability_ir_sha256: requirements.capability_ir_sha256.clone(),
        compilation_request_sha256,
        receiver_profile_sha256: profile.digest()?,
        strategy,
        affected_regions,
        lifecycle: PlanLifecycle::ShadowOnly,
    };
    plan.validate_against_profile(profile)?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile() -> ReceiverProfile {
        ReceiverProfile {
            schema: "cerebro.tidex.receiver_profile/v1".into(),
            model_id: ModelId::parse("receiver.v1").unwrap(),
            architecture_id: ArchitectureId::parse("transformer.v1").unwrap(),
            architecture: ReceiverArchitecture::Transformer,
            modalities: BTreeSet::from([CapabilityModality::Text]),
            supports_persistent_state: false,
            parameter_dimension: 10,
            regions: vec![ReceiverRegion {
                tensor_id: TensorId::parse("layers.0.attn.q_proj.weight").unwrap(),
                parameter_count: 10,
                supported_strategies: BTreeSet::from([
                    MaterializationStrategy::LowRank,
                    MaterializationStrategy::ReceiverCoordinates,
                ]),
            }],
        }
    }
    #[test]
    fn compatible_receiver_only_produces_shadow_plan() {
        let profile = profile();
        let requirements = CapabilityRequirements {
            schema: "cerebro.tidex.capability_requirements/v1".into(),
            capability_id: CapabilityId::parse("test.capability:v1").unwrap(),
            capability_ir_sha256: CapabilityIrDigest::draft_marker(),
            required_modalities: BTreeSet::from([CapabilityModality::Text]),
            requires_persistent_state: false,
            minimum_receiver_parameter_dimension: 10,
            acceptable_strategies: BTreeSet::from([MaterializationStrategy::LowRank]),
        };
        let assessment = assess_compatibility(&profile, &requirements).unwrap();
        assert_eq!(assessment.disposition, CompatibilityDisposition::Compatible);
        let plan = create_shadow_plan(
            &profile,
            &assessment,
            &requirements,
            Sha256Digest::zero(),
            MaterializationStrategy::LowRank,
            vec![TensorId::parse("layers.0.attn.q_proj.weight").unwrap()],
        )
        .unwrap();
        assert_eq!(plan.lifecycle, PlanLifecycle::ShadowOnly);
    }
    #[test]
    fn absent_modality_is_incompatible() {
        let requirements = CapabilityRequirements {
            schema: "cerebro.tidex.capability_requirements/v1".into(),
            capability_id: CapabilityId::parse("test.capability:v1").unwrap(),
            capability_ir_sha256: CapabilityIrDigest::draft_marker(),
            required_modalities: BTreeSet::from([CapabilityModality::Vision]),
            requires_persistent_state: false,
            minimum_receiver_parameter_dimension: 1,
            acceptable_strategies: BTreeSet::from([MaterializationStrategy::LowRank]),
        };
        assert_eq!(
            assess_compatibility(&profile(), &requirements)
                .unwrap()
                .disposition,
            CompatibilityDisposition::Incompatible
        );
    }
}
