use crate::artifact::{DeltaArtifactRef, F64ArtifactRef};
use crate::digest::{
    ObservationRecordDigest, ProvenanceDigest, RepresentationProtocolDigest, Sha256Digest,
};
use crate::error::{BrainError, BrainResult};
use crate::identity::{ApertureId, LineageId, ObservationId, ProbeId, ReconstructionId, SkillId};
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn deserialize_unique_skill_ids<'de, D>(deserializer: D) -> Result<Vec<SkillId>, D::Error>
where
    D: Deserializer<'de>,
{
    let ids = Vec::<SkillId>::deserialize(deserializer)?;
    if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err(D::Error::custom("duplicate_parent_skill_id"));
    }
    Ok(ids)
}

fn deserialize_unique_skill_fields<'de, D>(deserializer: D) -> Result<Vec<SkillField>, D::Error>
where
    D: Deserializer<'de>,
{
    let fields = Vec::<SkillField>::deserialize(deserializer)?;
    if fields
        .iter()
        .map(|field| &field.skill_id)
        .collect::<BTreeSet<_>>()
        .len()
        != fields.len()
        || fields.iter().any(|field| {
            field
                .parent_skill_ids
                .iter()
                .any(|parent| parent == &field.skill_id)
        })
    {
        return Err(D::Error::custom("invalid_skill_field_identity_graph"));
    }
    Ok(fields)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReconstructionInverseMode {
    #[serde(rename = "spectral_inverse")]
    Spectral,
    #[serde(rename = "persistent_inverse")]
    Persistent,
}

impl ReconstructionInverseMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Spectral => "spectral_inverse",
            Self::Persistent => "persistent_inverse",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfounderValue {
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct ExperimentLineage {
    pub run_id: String,
    pub replicate_id: String,
    pub randomization_id: String,
    pub dataset_split_digest: String,
    pub initial_checkpoint_digest: String,
    pub optimizer_config_digest: String,
    pub template_config_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeltaObservation {
    pub observation_id: ObservationId,
    pub from_checkpoint: String,
    pub to_checkpoint: String,
    pub generation: u64,
    pub delta: Vec<f64>,
    #[serde(default)]
    pub functional_response: Vec<f64>,
    #[serde(default)]
    pub confounders: Vec<ConfounderValue>,
    pub reliability: f64,
    pub independence_group: String,
    #[serde(default)]
    pub experiment_lineage: ExperimentLineage,
    /// Authenticated full parameter update. `delta` may be a sketch used for
    /// discovery; this reference is the executable dense evidence.
    #[serde(default)]
    pub dense_artifact: Option<DeltaArtifactRef>,
    /// SHA256 of the content-addressed ParameterBlockLayout describing the
    /// dense artifact's parameter space.
    #[serde(default)]
    pub parameter_layout_sha256: Option<Sha256Digest>,
    /// Architecture-internal representation shift measured on a sealed generic
    /// probe protocol. Stored as authenticated f64 tensor evidence.
    #[serde(default)]
    pub representation_artifact: Option<F64ArtifactRef>,
    #[serde(default)]
    pub representation_protocol_sha256: Option<RepresentationProtocolDigest>,
    pub provenance_digest: ProvenanceDigest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockSubspaceAxis {
    pub singular_value: f64,
    /// Coefficients over the observation-source dense deltas. The actual axis
    /// can be materialized from these authenticated sources without storing a
    /// duplicate dense vector in the field record.
    pub source_coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockSubspaceGeometry {
    pub block_name: String,
    pub offset: u64,
    pub count: usize,
    pub shape: Vec<usize>,
    pub selected_rank: usize,
    pub effective_rank: f64,
    pub retained_energy: f64,
    pub block_energy: f64,
    pub normalized_block_energy: f64,
    pub reconstruction_rms: f64,
    pub axes: Vec<BlockSubspaceAxis>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkillSubspaceGeometry {
    pub skill_id: SkillId,
    pub source_support_indices: Vec<usize>,
    pub blocks: Vec<BlockSubspaceGeometry>,
    pub max_local_rank: usize,
    pub mean_effective_rank: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkillField {
    /// Durable capability identity. Fresh reconstructions receive a content-derived
    /// candidate id; bank reconciliation preserves this id across later evidence.
    pub skill_id: SkillId,
    /// Ephemeral identity of the exact reconstruction/evidence support.
    #[serde(default)]
    pub reconstruction_id: ReconstructionId,
    /// Stable lineage anchor used to distinguish legacy/unversioned fields.
    #[serde(default)]
    pub lineage_id: LineageId,
    pub generation_created: u64,
    /// Canonical discovery-space direction (e.g. sketch/tomography space).
    /// This is NOT the complete executable capability.
    pub direction: Vec<f64>,
    /// Distributed block/subspace geometry. Required by production promotion.
    #[serde(default)]
    pub structured_geometry: Option<SkillSubspaceGeometry>,
    /// Full dense materialization created only after promotion from authenticated
    /// source artifacts. Runtime execution uses this, never the sketch direction.
    #[serde(default)]
    pub dense_materialization: Option<DeltaArtifactRef>,
    #[serde(default)]
    pub parameter_layout_sha256: Option<Sha256Digest>,
    /// Cross-aperture representation identity signature, attached only after
    /// independent P10/P19 validation.
    #[serde(default)]
    pub representation_signature: Vec<f64>,
    pub singular_value: f64,
    pub explained_variance: f64,
    pub persistence: f64,
    pub coherence: f64,
    pub uncertainty: f64,
    /// Sorted unique SHA256 digests of the exact DeltaObservation records that
    /// causally support this field. `support` is derived from this set whenever
    /// authoritative evidence is available; it must never count repeated
    /// reconstructions of the same observations.
    #[serde(default)]
    pub evidence_support_digests: Vec<ObservationRecordDigest>,
    pub support: usize,
    #[serde(default)]
    pub functional_signature: Vec<f64>,
    #[serde(default, deserialize_with = "deserialize_unique_skill_ids")]
    pub parent_skill_ids: Vec<SkillId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SkillBank {
    pub generation: u64,
    #[serde(deserialize_with = "deserialize_unique_skill_fields")]
    pub fields: Vec<SkillField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtectedDirection {
    pub probe_id: ProbeId,
    pub direction: Vec<f64>,
    pub importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtectedCortex {
    pub parameter_importance: Vec<f64>,
    #[serde(default)]
    pub directions: Vec<ProtectedDirection>,
    pub max_damage_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApertureCandidate {
    pub aperture_id: ApertureId,
    pub sensing_vector: Vec<f64>,
    pub noise_variance: f64,
    pub cost: f64,
    pub risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainConfig {
    pub min_observations: usize,
    pub min_independent_apertures: usize,
    pub target_explained_variance: f64,
    pub max_rank: usize,
    pub ridge: f64,
    pub huber_delta: f64,
    pub irls_rounds: usize,
    pub skill_match_cosine: f64,
    pub min_functional_cv_r2: f64,
    pub max_cycle_rms: f64,
    pub min_skill_coherence: f64,
    pub min_skill_persistence: f64,
    pub min_field_explained_variance: f64,
    pub max_condition_estimate: f64,
    pub max_spectral_normalized_reconstruction_rms: f64,
    pub min_identifiability_signal_to_noise: f64,
    pub require_structured_geometry_for_promotion: bool,
    pub require_dual_space_for_promotion: bool,
    #[serde(default = "default_min_representation_cv_r2")]
    pub min_representation_cv_r2: f64,
    pub min_representation_match_accuracy: f64,
    pub min_representation_match_margin: f64,
}
impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            min_observations: 6,
            min_independent_apertures: 3,
            target_explained_variance: 0.92,
            max_rank: 32,
            ridge: 1e-6,
            huber_delta: 1.5,
            irls_rounds: 4,
            skill_match_cosine: 0.82,
            min_functional_cv_r2: 0.35,
            max_cycle_rms: 0.20,
            min_skill_coherence: 0.50,
            min_skill_persistence: 0.45,
            min_field_explained_variance: 0.01,
            max_condition_estimate: 1e12,
            max_spectral_normalized_reconstruction_rms: 0.45,
            min_identifiability_signal_to_noise: 1.0,
            require_structured_geometry_for_promotion: true,
            require_dual_space_for_promotion: true,
            min_representation_cv_r2: 0.35,
            min_representation_match_accuracy: 1.0,
            min_representation_match_margin: 0.0,
        }
    }
}

fn default_min_representation_cv_r2() -> f64 {
    0.35
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromotionBlocker {
    CycleConsistencyFailed,
    FunctionalCrossValidationFailed,
    NoSkillFields,
    SkillCoherenceFailed,
    SkillPersistenceFailed,
    InsufficientDeclaredApertures,
    ApertureIndependenceUnresolved,
    SkillIdentifiabilityUnresolved,
    TomographyIllConditioned,
    ReconstructionErrorHigh,
    PersistentObservationCoverageIncomplete,
    PersistentClusterAssignmentInconsistentAcrossSpaces,
    PersistentClusterIdentityMarginUnresolved,
    PersistentCoherenceGapNotIdentifiable,
    PersistentHoldoutAlignmentNonpositive,
    StructuredGeometryRequiredForPromotion,
    DualSpaceRepresentationGeneralizationUnverified,
    DualSpaceRepresentationSignatureMissing,
}

impl PromotionBlocker {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CycleConsistencyFailed => "cycle_consistency_failed",
            Self::FunctionalCrossValidationFailed => "functional_cross_validation_failed",
            Self::NoSkillFields => "no_skill_fields",
            Self::SkillCoherenceFailed => "skill_coherence_failed",
            Self::SkillPersistenceFailed => "skill_persistence_failed",
            Self::InsufficientDeclaredApertures => "insufficient_declared_apertures",
            Self::ApertureIndependenceUnresolved => "aperture_independence_unresolved",
            Self::SkillIdentifiabilityUnresolved => "skill_identifiability_unresolved",
            Self::TomographyIllConditioned => "tomography_ill_conditioned",
            Self::ReconstructionErrorHigh => "reconstruction_error_high",
            Self::PersistentObservationCoverageIncomplete => {
                "persistent_observation_coverage_incomplete"
            }
            Self::PersistentClusterAssignmentInconsistentAcrossSpaces => {
                "persistent_cluster_assignment_inconsistent_across_spaces"
            }
            Self::PersistentClusterIdentityMarginUnresolved => {
                "persistent_cluster_identity_margin_unresolved"
            }
            Self::PersistentCoherenceGapNotIdentifiable => {
                "persistent_coherence_gap_not_identifiable"
            }
            Self::PersistentHoldoutAlignmentNonpositive => {
                "persistent_holdout_alignment_nonpositive"
            }
            Self::StructuredGeometryRequiredForPromotion => {
                "structured_geometry_required_for_promotion"
            }
            Self::DualSpaceRepresentationGeneralizationUnverified => {
                "dual_space_representation_generalization_unverified"
            }
            Self::DualSpaceRepresentationSignatureMissing => {
                "dual_space_representation_signature_missing"
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromotionDecision {
    pub allowed: bool,
    pub reasons: Vec<PromotionBlocker>,
    pub metrics: BTreeMap<String, f64>,
}

impl PromotionDecision {
    /// Promotion is a strict conjunction: a persisted decision cannot claim
    /// success while naming a failed gate, nor smuggle a non-finite metric into
    /// an otherwise valid receipt.
    pub fn validate(&self) -> BrainResult<()> {
        if self.allowed != self.reasons.is_empty()
            || self.metrics.keys().any(|key| key.trim().is_empty())
            || self.metrics.values().any(|value| !value.is_finite())
        {
            return Err(BrainError::Integrity(
                "promotion_decision_contract_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(id: &str) -> SkillField {
        SkillField {
            skill_id: SkillId::parse(id).unwrap(),
            reconstruction_id: ReconstructionId::unassigned(),
            lineage_id: LineageId::unassigned(),
            generation_created: 1,
            direction: vec![1.0],
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: 1.0,
            explained_variance: 1.0,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.0,
            evidence_support_digests: Vec::new(),
            support: 1,
            functional_signature: Vec::new(),
            parent_skill_ids: Vec::new(),
        }
    }

    #[test]
    fn promotion_is_a_strict_conjunction_with_stable_wire_blockers() {
        let blocked = PromotionDecision {
            allowed: false,
            reasons: vec![PromotionBlocker::FunctionalCrossValidationFailed],
            metrics: BTreeMap::from([("functional_cv_r2".into(), 0.2)]),
        };
        assert!(blocked.validate().is_ok());
        assert_eq!(
            serde_json::to_string(&blocked.reasons).unwrap(),
            "[\"functional_cross_validation_failed\"]"
        );
        let contradictory = PromotionDecision {
            allowed: true,
            ..blocked
        };
        assert!(contradictory.validate().is_err());
    }

    #[test]
    fn skill_bank_deserialization_rejects_duplicate_identities() {
        let field = serde_json::to_value(skill("cap-one")).unwrap();
        let wire = serde_json::json!({"generation": 2, "fields": [field.clone(), field]});
        assert!(serde_json::from_value::<SkillBank>(wire).is_err());
    }

    #[test]
    fn lineage_parents_are_typed_unique_and_wire_compatible() {
        let mut field = serde_json::to_value(skill("cap-current")).unwrap();
        field["parent_skill_ids"] = serde_json::json!(["cap-parent-a", "cap-parent-b"]);
        let parsed: SkillField = serde_json::from_value(field.clone()).unwrap();
        assert_eq!(parsed.parent_skill_ids[0], "cap-parent-a");
        assert_eq!(serde_json::to_value(parsed).unwrap(), field);

        field["parent_skill_ids"] = serde_json::json!(["cap-parent-a", "cap-parent-a"]);
        assert!(serde_json::from_value::<SkillField>(field.clone()).is_err());
        field["parent_skill_ids"] = serde_json::json!(["../foreign-skill"]);
        assert!(serde_json::from_value::<SkillField>(field).is_err());

        let mut self_parent = serde_json::to_value(skill("cap-current")).unwrap();
        self_parent["parent_skill_ids"] = serde_json::json!(["cap-current"]);
        let bank = serde_json::json!({"generation": 2, "fields": [self_parent]});
        assert!(serde_json::from_value::<SkillBank>(bank).is_err());
    }

    #[test]
    fn observation_wire_rejects_path_identity_bad_digest_and_unknown_fields() {
        let valid = serde_json::json!({
            "observation_id": "obs-1",
            "from_checkpoint": "base",
            "to_checkpoint": "next",
            "generation": 1,
            "delta": [0.1],
            "reliability": 1.0,
            "independence_group": "group-1",
            "provenance_digest": "a".repeat(64)
        });
        assert!(serde_json::from_value::<DeltaObservation>(valid.clone()).is_ok());

        let mut path_id = valid.clone();
        path_id["observation_id"] = serde_json::json!("../obs-1");
        assert!(serde_json::from_value::<DeltaObservation>(path_id).is_err());

        let mut bad_digest = valid.clone();
        bad_digest["provenance_digest"] = serde_json::json!("raw");
        assert!(serde_json::from_value::<DeltaObservation>(bad_digest).is_err());

        let mut unknown = valid;
        unknown["unsealed_evidence"] = serde_json::json!(true);
        assert!(serde_json::from_value::<DeltaObservation>(unknown).is_err());
    }
}
