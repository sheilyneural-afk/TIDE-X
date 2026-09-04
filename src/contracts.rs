use crate::artifact::{DeltaArtifactRef, F64ArtifactRef};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
pub struct DeltaObservation {
    pub observation_id: String,
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
    pub parameter_layout_sha256: Option<String>,
    /// Architecture-internal representation shift measured on a sealed generic
    /// probe protocol. Stored as authenticated f64 tensor evidence.
    #[serde(default)]
    pub representation_artifact: Option<F64ArtifactRef>,
    #[serde(default)]
    pub representation_protocol_sha256: Option<String>,
    pub provenance_digest: String,
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
    pub skill_id: String,
    pub source_support_indices: Vec<usize>,
    pub blocks: Vec<BlockSubspaceGeometry>,
    pub max_local_rank: usize,
    pub mean_effective_rank: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkillField {
    /// Durable capability identity. Fresh reconstructions receive a content-derived
    /// candidate id; bank reconciliation preserves this id across later evidence.
    pub skill_id: String,
    /// Ephemeral identity of the exact reconstruction/evidence support.
    #[serde(default)]
    pub reconstruction_id: String,
    /// Stable lineage anchor used to distinguish legacy/unversioned fields.
    #[serde(default)]
    pub lineage_id: String,
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
    pub parameter_layout_sha256: Option<String>,
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
    pub evidence_support_digests: Vec<String>,
    pub support: usize,
    #[serde(default)]
    pub functional_signature: Vec<f64>,
    #[serde(default)]
    pub parent_skill_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SkillBank {
    pub generation: u64,
    pub fields: Vec<SkillField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProtectedDirection {
    pub probe_id: String,
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
    pub aperture_id: String,
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
            min_representation_match_accuracy: 1.0,
            min_representation_match_margin: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromotionDecision {
    pub allowed: bool,
    pub reasons: Vec<String>,
    pub metrics: BTreeMap<String, f64>,
}
