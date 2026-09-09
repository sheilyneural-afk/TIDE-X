//! Non-actuating candidate construction for receiver-coordinate experiments.
//!
//! This module has no model-runtime, adapter, tensor-write, or activation
//! dependency. It can construct a digest-bound candidate and persist it as an
//! immutable laboratory artifact for a later, separately reviewed backend.

use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::lab_isolation::LabRoots;
use crate::receiver_profile::{MaterializationPlan, MaterializationStrategy, ReceiverProfile};
use crate::universal_capability_compiler::{
    replay_universal_capability_shadow_plan, UniversalCapabilityPlanningRequest,
    UniversalCapabilityShadowPlanReceipt,
};
use serde::{Deserialize, Serialize};

const MAX_SHADOW_CANDIDATE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SHADOW_COORDINATES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowReceiverCoordinateCandidate {
    pub schema: String,
    pub plan_sha256: Sha256Digest,
    pub coordinate_sha256: Sha256Digest,
    pub receiver_parameter_dimension: u64,
    pub coordinates: Vec<f64>,
    pub manifest_sha256: Sha256Digest,
}

fn plan_digest(plan: &MaterializationPlan) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:SHADOW-MATERIALIZATION-PLAN:v1\0",
        &serde_json::to_vec(plan)?,
    ))
}

impl ShadowReceiverCoordinateCandidate {
    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:SHADOW-RECEIVER-CANDIDATE:v1\0",
            &serde_json::to_vec(&unsigned)?,
        ))
    }

    pub fn validate(
        &self,
        profile: &ReceiverProfile,
        plan: &MaterializationPlan,
    ) -> BrainResult<()> {
        plan.validate_against_profile(profile)?;
        if self.coordinates.len() > MAX_SHADOW_COORDINATES
            || self.coordinates.iter().any(|value| !value.is_finite())
            || u64::try_from(self.coordinates.len()).ok() != Some(self.receiver_parameter_dimension)
        {
            return Err(BrainError::Integrity(
                "shadow_receiver_coordinate_candidate_invalid".into(),
            ));
        }
        let coordinate_bytes = serde_json::to_vec(&self.coordinates)?;
        if self.schema != "cerebro.tidex.shadow_receiver_coordinate_candidate/v1"
            || self.plan_sha256 != plan_digest(plan)?
            || self.coordinate_sha256
                != Sha256Digest::digest_domain(
                    b"CEREBRO:TIDEX:SHADOW-RECEIVER-COORDINATES:v1\0",
                    &coordinate_bytes,
                )
            || self.receiver_parameter_dimension != profile.parameter_dimension
            || self.manifest_sha256 != self.calculate_digest()?
        {
            return Err(BrainError::Integrity(
                "shadow_receiver_coordinate_candidate_invalid".into(),
            ));
        }
        Ok(())
    }
}

/// Construct an inert candidate.  This does not serialize to a model format,
/// touch a tensor, or provide an activation method.
fn materialize_receiver_coordinates_shadow(
    profile: &ReceiverProfile,
    plan: &MaterializationPlan,
    coordinates: Vec<f64>,
) -> BrainResult<ShadowReceiverCoordinateCandidate> {
    plan.validate_against_profile(profile)?;
    if plan.strategy != MaterializationStrategy::ReceiverCoordinates
        || u64::try_from(coordinates.len())
            .map_err(|_| BrainError::Invalid("shadow_coordinate_length_overflow".into()))?
            != profile.parameter_dimension
        || coordinates.len() > MAX_SHADOW_COORDINATES
        || coordinates.iter().any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid(
            "shadow_receiver_coordinate_candidate_invalid".into(),
        ));
    }
    let coordinate_bytes = serde_json::to_vec(&coordinates)?;
    let mut candidate = ShadowReceiverCoordinateCandidate {
        schema: "cerebro.tidex.shadow_receiver_coordinate_candidate/v1".into(),
        plan_sha256: plan_digest(plan)?,
        coordinate_sha256: Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:SHADOW-RECEIVER-COORDINATES:v1\0",
            &coordinate_bytes,
        ),
        receiver_parameter_dimension: profile.parameter_dimension,
        coordinates,
        manifest_sha256: Sha256Digest::zero(),
    };
    candidate.manifest_sha256 = candidate.calculate_digest()?;
    candidate.validate(profile, plan)?;
    Ok(candidate)
}

/// Reproduce the complete planning decision and materialize only the
/// compiler-produced `target_delta`. No caller-provided coordinate vector can
/// cross this public boundary.
pub fn materialize_replayed_receiver_coordinates_shadow(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
) -> BrainResult<ShadowReceiverCoordinateCandidate> {
    replay_universal_capability_shadow_plan(request, receipt)?;
    let shadow_plan = &receipt.shadow_plan;
    materialize_receiver_coordinates_shadow(
        &request.receiver_profile,
        &shadow_plan.materialization_plan,
        shadow_plan
            .compilation_receipt
            .compilation
            .receiver
            .target_delta
            .clone(),
    )
}

pub fn persist_shadow_candidate(
    roots: &LabRoots,
    profile: &ReceiverProfile,
    plan: &MaterializationPlan,
    candidate: &ShadowReceiverCoordinateCandidate,
) -> BrainResult<PrivateFileReference> {
    candidate.validate(profile, plan)?;
    let bytes = serde_json::to_vec(candidate)?;
    if u64::try_from(bytes.len())
        .map_err(|_| BrainError::Invalid("shadow_candidate_size_overflow".into()))?
        > MAX_SHADOW_CANDIDATE_BYTES
    {
        return Err(BrainError::Invalid("shadow_candidate_too_large".into()));
    }
    let destination = roots
        .artifact_root()
        .join("shadow-candidates")
        .join(format!("{}.json", candidate.manifest_sha256.as_str()));
    let sha256 = write_or_verify_immutable(roots.lab_root(), &destination, &bytes)?;
    Ok(PrivateFileReference::new(destination, sha256))
}

pub fn load_shadow_candidate(
    roots: &LabRoots,
    profile: &ReceiverProfile,
    plan: &MaterializationPlan,
    reference: &PrivateFileReference,
) -> BrainResult<ShadowReceiverCoordinateCandidate> {
    let bytes = reference.read_verified_bounded(roots.lab_root(), MAX_SHADOW_CANDIDATE_BYTES)?;
    let candidate: ShadowReceiverCoordinateCandidate = serde_json::from_slice(&bytes)?;
    let expected = roots
        .artifact_root()
        .join("shadow-candidates")
        .join(format!("{}.json", candidate.manifest_sha256.as_str()));
    if reference.path != expected {
        return Err(BrainError::Integrity(
            "shadow_candidate_path_invalid".into(),
        ));
    }
    candidate.validate(profile, plan)?;
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::CapabilityIrDigest;
    use crate::identity::{ArchitectureId, CapabilityId, ModelId, TensorId};
    use crate::receiver_profile::{
        assess_compatibility, create_shadow_plan, CapabilityModality, CapabilityRequirements,
        ReceiverArchitecture, ReceiverRegion,
    };
    use std::collections::BTreeSet;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> (ReceiverProfile, MaterializationPlan) {
        let tensor = TensorId::parse("layers.0.attn.q_proj.weight").unwrap();
        let profile = ReceiverProfile {
            schema: "cerebro.tidex.receiver_profile/v1".into(),
            model_id: ModelId::parse("receiver.v1").unwrap(),
            architecture_id: ArchitectureId::parse("transformer.v1").unwrap(),
            architecture: ReceiverArchitecture::Transformer,
            modalities: BTreeSet::from([CapabilityModality::Text]),
            supports_persistent_state: false,
            parameter_dimension: 3,
            regions: vec![ReceiverRegion {
                tensor_id: tensor.clone(),
                parameter_count: 3,
                supported_strategies: BTreeSet::from([
                    MaterializationStrategy::ReceiverCoordinates,
                ]),
            }],
        };
        let requirements = CapabilityRequirements {
            schema: "cerebro.tidex.capability_requirements/v1".into(),
            capability_id: CapabilityId::parse("test.capability:v1").unwrap(),
            capability_ir_sha256: CapabilityIrDigest::draft_marker(),
            required_modalities: BTreeSet::from([CapabilityModality::Text]),
            requires_persistent_state: false,
            minimum_receiver_parameter_dimension: 3,
            acceptable_strategies: BTreeSet::from([MaterializationStrategy::ReceiverCoordinates]),
        };
        let assessment = assess_compatibility(&profile, &requirements).unwrap();
        let plan = create_shadow_plan(
            &profile,
            &assessment,
            &requirements,
            Sha256Digest::digest_bytes(b"compilation"),
            MaterializationStrategy::ReceiverCoordinates,
            vec![tensor],
        )
        .unwrap();
        (profile, plan)
    }

    #[test]
    fn candidate_is_digest_bound_and_rejects_tampering() {
        let (profile, plan) = fixture();
        let candidate =
            materialize_receiver_coordinates_shadow(&profile, &plan, vec![1.0, 2.0, 3.0]).unwrap();
        candidate.validate(&profile, &plan).unwrap();
        let mut tampered = candidate;
        tampered.coordinates[0] = 9.0;
        assert!(tampered.validate(&profile, &plan).is_err());
        assert!(materialize_receiver_coordinates_shadow(&profile, &plan, vec![1.0, 2.0]).is_err());
        assert!(
            materialize_receiver_coordinates_shadow(&profile, &plan, vec![1.0, f64::NAN, 3.0])
                .is_err()
        );
        let mut unsupported = plan;
        unsupported.strategy = MaterializationStrategy::DenseDelta;
        assert!(materialize_receiver_coordinates_shadow(
            &profile,
            &unsupported,
            vec![1.0, 2.0, 3.0]
        )
        .is_err());
    }

    #[test]
    fn persisted_candidate_is_confined_authenticated_and_replayable() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "cerebro-shadow-candidate-{}-{nonce}",
            std::process::id()
        ));
        let lab = base.join("lab");
        let state = lab.join("state");
        let artifacts = lab.join("artifacts");
        let production = base.join("production");
        for path in [&lab, &state, &artifacts, &production] {
            fs::create_dir_all(path).unwrap();
            crate::security::secure_dir(path).unwrap();
        }
        let roots = LabRoots::open_for_test(&lab, &state, &artifacts, &production).unwrap();
        let (profile, plan) = fixture();
        let candidate =
            materialize_receiver_coordinates_shadow(&profile, &plan, vec![1.0, 2.0, 3.0]).unwrap();
        let reference = persist_shadow_candidate(&roots, &profile, &plan, &candidate).unwrap();
        assert_eq!(
            load_shadow_candidate(&roots, &profile, &plan, &reference).unwrap(),
            candidate
        );
        assert_eq!(
            persist_shadow_candidate(&roots, &profile, &plan, &candidate).unwrap(),
            reference
        );

        let wrong_path = PrivateFileReference::new(
            roots.artifact_root().join("shadow-candidates/alias.json"),
            reference.sha256.clone(),
        );
        assert!(load_shadow_candidate(&roots, &profile, &plan, &wrong_path).is_err());
        fs::remove_dir_all(base).unwrap();
    }
}
