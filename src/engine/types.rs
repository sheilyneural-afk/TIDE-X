//! Wire and transaction types for the TIDE-X engine authorities.

use crate::aperture_independence::ApertureIndependenceReport;
use crate::artifact::DeltaArtifactRef;
use crate::authority::PrivateFileReference;
use crate::cognitive_field::{CognitiveFieldState, FieldRoutingDecision};
use crate::contracts::{PromotionDecision, ReconstructionInverseMode, SkillField};
use crate::digest::{
    AnalysisVersionDigest, CausalCreditDigest, ConfigDigest, CorpusDigest, EvidenceBundleDigest,
    MemoryDigest, ParameterLayoutDigest, ReportDigest, Sha256Digest, SkillBankDigest,
    SourceTreeDigest,
};
use crate::identifiability::ResolutionMap;
use crate::identity::{SessionId, SkillId};
use crate::learning_finalization::RepresentationObservationBinding;
use crate::protected::ProtectionResult;
use crate::sleep_diagnostics::SleepConsolidationDiagnostics;
use crate::sleep_evidence::SleepEvidenceVerification;
use crate::trust_region::TrustRegionResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReconstructionReport {
    pub schema: String,
    pub source_tree_digest: SourceTreeDigest,
    pub config_digest: ConfigDigest,
    pub analysis_version_digest: AnalysisVersionDigest,
    pub observation_count: usize,
    pub observation_set_digest: CorpusDigest,
    pub parameter_dimension: usize,
    pub independence_groups: usize,
    pub aperture_independence: ApertureIndependenceReport,
    pub resolution_map: ResolutionMap,
    pub confounder_names: Vec<String>,
    pub confounder_explained_fraction: f64,
    pub cycle_rms: f64,
    pub max_edge_residual: f64,
    pub selected_rank: usize,
    pub effective_rank: f64,
    pub condition_estimate: f64,
    pub reconstruction_rms: f64,
    pub normalized_reconstruction_rms: f64,
    pub functional_cv_r2: f64,
    pub inverse_mode: ReconstructionInverseMode,
    pub spectral_functional_cv_r2: f64,
    pub persistent_functional_cv_r2: Option<f64>,
    pub persistent_coherence_threshold: Option<f64>,
    pub persistent_coherence_gap: Option<f64>,
    pub persistent_coverage_ratio: Option<f64>,
    pub persistent_cluster_stability: Option<f64>,
    pub persistent_parametric_cluster_stability: Option<f64>,
    pub persistent_functional_cluster_stability: Option<f64>,
    pub persistent_cluster_identity_min_margin: Option<f64>,
    pub persistent_cluster_assignment_consistent: Option<bool>,
    pub persistent_min_holdout_similarity: Option<f64>,
    pub persistent_cluster_sizes: Vec<usize>,
    pub persistent_error: Option<String>,
    pub representation_protocol_sha256: Option<String>,
    pub representation_cv_r2: Option<f64>,
    pub representation_match_accuracy: Option<f64>,
    pub representation_mean_matched_cosine: Option<f64>,
    pub representation_min_match_margin: Option<f64>,
    pub dual_space_verified: Option<bool>,
    pub fields: Vec<SkillField>,
    /// Observation -> selected SkillField coordinates for downstream dual-space
    /// reconstruction, causal credit and representation sensing.
    pub field_coefficients: Vec<Vec<f64>>,
    /// Exact coefficients over the original (pre-confounder-removal) delta
    /// observations for materializing each skill outside sketch space.
    pub skill_source_mixtures: Vec<Vec<f64>>,
    pub promotion: PromotionDecision,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SleepReport {
    pub schema: String,
    pub corpus_digest: CorpusDigest,
    pub observation_count: usize,
    pub promoted: bool,
    pub idempotent: bool,
    pub active_skill_count: usize,
    pub memory_digest: MemoryDigest,
    pub evidence_bundle_sha256: Option<EvidenceBundleDigest>,
    pub evidence_verification: SleepEvidenceVerification,
    pub diagnostics: SleepConsolidationDiagnostics,
    pub reconstruction: ReconstructionReport,
}

/// Immutable receipt for a verified learning hand-off that replaces the active
/// reconstruction corpus. The producer may be any verified learner; the
/// engine only trusts the replayable evidence/finalization contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearningFinalizationReceipt {
    pub schema: String,
    pub operation_key: Sha256Digest,
    pub session_id: SessionId,
    pub adaptive_receipt_sha256: Sha256Digest,
    pub learning_finalization_input_sha256: Sha256Digest,
    pub representation_evidence_receipt: PrivateFileReference,
    pub representation_protocol_sha256: Sha256Digest,
    pub representation_observation_bindings_sha256: Sha256Digest,
    pub representation_observation_bindings: Vec<RepresentationObservationBinding>,
    pub prior_corpus_digest: Sha256Digest,
    pub prior_observation_count: usize,
    pub new_corpus_digest: Sha256Digest,
    pub new_observation_count: usize,
    pub archived_artifact_sha256: BTreeMap<String, Sha256Digest>,
    pub report_sha256: Sha256Digest,
    pub commit_operation_key: Sha256Digest,
    pub commit_receipt_sha256: Sha256Digest,
    pub ledger_event_hash: Sha256Digest,
}

#[derive(Debug, Clone)]
pub(super) struct GovernedComposition {
    pub delta: Vec<f64>,
    pub trust_region: TrustRegionResult,
    pub protection: ProtectionResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GovernedCompositionProtection {
    pub damage_ratio: f64,
    pub allowed: bool,
    pub removed_energy: f64,
    pub protected_rank: usize,
    pub max_weighted_residual: f64,
}

/// Durable authority for an executable composition. The human/controller may
/// request an activation, but the recorded coefficients and delta are solely
/// the output of current certified causal trust plus Protected Cortex.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GovernedCompositionReceipt {
    pub schema: String,
    pub operation_key: String,
    pub report_sha256: ReportDigest,
    pub active_bank_sha256: SkillBankDigest,
    pub evidence_bundle_sha256: EvidenceBundleDigest,
    pub causal_credit_sha256: CausalCreditDigest,
    /// Exact byte identity and confined path of the immutable layout envelope
    /// used to interpret `projected_delta`.
    pub parameter_layout_artifact: PrivateFileReference,
    /// Semantic identity re-derived from the authenticated envelope. This is
    /// deliberately distinct from the envelope's exact byte SHA-256.
    pub parameter_layout_sha256: ParameterLayoutDigest,
    pub field_ids: Vec<SkillId>,
    pub requested_activation: BTreeMap<SkillId, f64>,
    pub accepted_coefficients: Vec<f64>,
    pub trust_region: TrustRegionResult,
    pub projected_delta: DeltaArtifactRef,
    pub protection: GovernedCompositionProtection,
    /// The authenticated current observation that caused this request. Runtime
    /// activation without a measured source is intentionally not representable.
    pub source_observation_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecordedGovernedComposition {
    pub receipt_path: String,
    pub receipt_sha256: String,
    pub ledger_event_hash: String,
    pub receipt: GovernedCompositionReceipt,
}

/// Sealed runtime request for a persisted LearnedController.  It deliberately
/// contains state only: the functional observation is resolved from the
/// authenticated active corpus by `BrainEngine`, never accepted from a caller.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControllerInvocation {
    pub schema: String,
    pub session_id: SessionId,
    pub state_before: Vec<f64>,
    pub promoted_observation_semantic_sha256: Sha256Digest,
}

/// Immutable audit record for one controller decision.  The governed
/// composition remains the sole authority for the parameter delta; this
/// receipt binds that authority to the exact persisted controller, state, and
/// promoted observation that caused the decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControllerExecutionReceipt {
    pub schema: String,
    pub session_id: SessionId,
    /// Canonical semantic invocation stored in the immutable receipt so a
    /// verifier can replay the controller decision without trusting a later
    /// caller-supplied vector or an external mutable file.
    pub invocation: ControllerInvocation,
    pub invocation_sha256: Sha256Digest,
    pub controller_receipt_sha256: Sha256Digest,
    pub state_before_sha256: Sha256Digest,
    pub promoted_observation_semantic_sha256: Sha256Digest,
    pub governed_composition_receipt_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecordedControllerExecution {
    pub receipt_path: String,
    pub receipt_sha256: String,
    pub ledger_event_hash: String,
    pub receipt: ControllerExecutionReceipt,
}

/// The durable counterpart of a Dynamic Cognitive Field decision.  A route is
/// informative on its own, but it becomes executable only through the same
/// receipt-backed causal-trust/protection path as every other runtime action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecordedGovernedCognitiveComposition {
    pub state: CognitiveFieldState,
    pub route: FieldRoutingDecision,
    pub composition: RecordedGovernedComposition,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum CertificationStatus {
    Certified,
    Revoked,
}

impl CertificationStatus {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Certified => "certified",
            Self::Revoked => "revoked",
        }
    }

    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "certified" => Some(Self::Certified),
            "revoked" => Some(Self::Revoked),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct RuntimeIntegrityHealth {
    pub(super) schema: String,
    pub(super) canonical_runtime_config: bool,
    pub(super) canonical_head_verified: bool,
    pub(super) corpus_transition_clear: bool,
    pub(super) ledger_verified: bool,
    pub(super) bank_verified: bool,
    pub(super) composition_ready: bool,
    pub(super) observations_verified: bool,
    pub(super) sleep_state_verified: bool,
    pub(super) receipt_verified: bool,
    pub(super) historical_artifacts_verified: bool,
    pub(super) current_pointers_verified: bool,
    pub(super) report_state_consistent: bool,
    pub(super) current_corpus_bound: bool,
    pub(super) analysis_current: bool,
    pub(super) certified: bool,
    pub(super) evidence_verified: bool,
    pub(super) integrity_healthy: bool,
    pub(super) execution_authorized: bool,
    pub(super) operation_key: Option<String>,
    pub(super) certification_status: Option<CertificationStatus>,
    pub(super) integrity_reasons: Vec<String>,
    pub(super) execution_blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct CommitTransactionIntent {
    pub(super) schema: String,
    pub(super) operation_key: String,
    pub(super) batch_digest: String,
    pub(super) observation_digests: Vec<String>,
    pub(super) report_sha256: ReportDigest,
    pub(super) report_promotable: bool,
    pub(super) generation: u64,
    pub(super) memory_sha256: MemoryDigest,
    pub(super) shadow_bank_sha256: Option<SkillBankDigest>,
    pub(super) prior_shadow_bank_sha256: Option<SkillBankDigest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct CommitReceipt {
    pub(super) schema: String,
    pub(super) operation_key: String,
    pub(super) batch_digest: String,
    pub(super) report_sha256: ReportDigest,
    pub(super) memory_sha256: MemoryDigest,
    pub(super) shadow_bank_sha256: Option<SkillBankDigest>,
    pub(super) ledger_event_hash: String,
    pub(super) legacy_recovery: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SleepTransactionIntent {
    pub(super) schema: String,
    pub(super) operation_key: String,
    pub(super) analysis_key: String,
    pub(super) corpus_digest: CorpusDigest,
    pub(super) analysis_version_digest: AnalysisVersionDigest,
    pub(super) config_digest: ConfigDigest,
    pub(super) report_sha256: ReportDigest,
    pub(super) memory_sha256: MemoryDigest,
    pub(super) active_bank_sha256: Option<SkillBankDigest>,
    pub(super) sleep_state_sha256: String,
    pub(super) evidence_bundle_sha256: Option<EvidenceBundleDigest>,
    pub(super) evidence_verified: bool,
    pub(super) certification_status: CertificationStatus,
    pub(super) promoted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SleepReceipt {
    pub(super) schema: String,
    pub(super) operation_key: String,
    pub(super) analysis_key: String,
    pub(super) report_sha256: ReportDigest,
    pub(super) memory_sha256: MemoryDigest,
    pub(super) active_bank_sha256: Option<SkillBankDigest>,
    pub(super) evidence_bundle_sha256: Option<EvidenceBundleDigest>,
    pub(super) sleep_state_sha256: String,
    pub(super) ledger_event_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct LearningCorpusTransitionIntent {
    pub(super) schema: String,
    pub(super) operation_key: Sha256Digest,
    pub(super) session_id: SessionId,
    pub(super) adaptive_receipt_sha256: Sha256Digest,
    pub(super) learning_finalization_input_sha256: Sha256Digest,
    pub(super) representation_evidence_receipt: PrivateFileReference,
    pub(super) representation_protocol_sha256: Sha256Digest,
    pub(super) representation_observation_bindings_sha256: Sha256Digest,
    pub(super) prior_corpus_digest: Sha256Digest,
    pub(super) prior_observation_count: usize,
    pub(super) new_corpus_digest: Sha256Digest,
    pub(super) new_observation_count: usize,
    pub(super) archive_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct GovernedCompositionPointer {
    pub(super) schema: String,
    pub(super) operation_key: String,
    pub(super) receipt_sha256: String,
}

#[derive(Debug, Clone)]
pub(super) enum HeadIncomplete {
    Preserve,
    Set(Sha256Digest),
    Clear,
}
