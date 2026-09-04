#![allow(clippy::needless_range_loop)]
use crate::aperture_independence::{estimate_aperture_independence, ApertureIndependenceReport};
use crate::artifact::{
    combine_content_addressed_dvec, create_content_addressed_dvec,
    derive_content_addressed_dvec_combination, inspect_dvec, read_dvec_f32, read_f64_artifact,
    DeltaArtifactRef,
};
use crate::authority::{
    ensure_private_directory, ensure_private_parent, existing_directory_under_root,
    existing_regular_file_under_root, replace_private_file_atomic, root_relative_path,
    write_or_verify_immutable, PrivateFileReference,
};
use crate::block_tomography::{
    reconstruct_structured_geometry, ParameterBlockLayout, StructuredSource,
};
use crate::causal_credit::{certified_causal_priority_weights, CausalCreditReport};
use crate::cognitive_field::{
    CognitiveFieldConfig, CognitiveFieldDrive, CognitiveFieldState, DynamicCognitiveField,
    FieldRoutingDecision,
};
use crate::confounders::remove_confounders;
use crate::contracts::{
    BrainConfig, DeltaObservation, PromotionDecision, ProtectedCortex, ReconstructionInverseMode,
    SkillBank, SkillField,
};
use crate::digest::Sha256Digest;
use crate::dual_space::{analyze_dual_space, DualSpaceModel, RepresentationObservation};
use crate::error::{BrainError, BrainResult};
use crate::functional::{attach_signatures, fit_functional_map};
use crate::identifiability::{resolution_map, ResolutionMap};
use crate::identity::SessionId;
use crate::learned_controller::{
    load_persisted_runtime_learned_controller, RuntimeLearnedController,
};
use crate::learning_finalization::{
    learning_finalization_input_sha256, prepare_learning_finalization,
    verify_learning_finalization_input, LearningFinalizationInput,
    RepresentationObservationBinding,
};
use crate::ledger;
use crate::linalg::{cosine, norm, Matrix};
use crate::memory::{build_memory_snapshot, memory_artifact_path};
use crate::persistent::reconstruct_persistent_skill_fields;
use crate::protected::{project_to_safe_subspace, ProtectionResult};
use crate::protected_map::{load_protected_cortex, ProtectedMapArtifactReport};
use crate::sbas::reconstruct_trajectory;
use crate::security::{secure_dir, secure_file, verify_private_root};
use crate::sleep_diagnostics::{diagnose_consolidation, SleepConsolidationDiagnostics};
use crate::sleep_evidence::{
    load_sleep_evidence, verify_sleep_evidence, SleepEvidenceExpectation, SleepEvidenceVerification,
};
use crate::tomography::{
    align_incoming_identities, assimilate_bank, reconcile_full_corpus, reconstruct_skill_fields,
};
use crate::trust_region::{
    apply_causal_priority_trust_region, TrustRegionAllocationPolicy, TrustRegionResult,
};
use crate::validation::{source_support_indices, valid_observation_id};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReconstructionReport {
    pub schema: String,
    pub source_tree_digest: String,
    pub config_digest: String,
    pub analysis_version_digest: String,
    pub observation_count: usize,
    pub observation_set_digest: String,
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
pub struct SleepReport {
    pub schema: String,
    pub corpus_digest: String,
    pub observation_count: usize,
    pub promoted: bool,
    pub idempotent: bool,
    pub active_skill_count: usize,
    pub memory_digest: String,
    pub evidence_bundle_sha256: Option<String>,
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
struct GovernedComposition {
    pub delta: Vec<f64>,
    pub trust_region: TrustRegionResult,
    pub protection: ProtectionResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    pub report_sha256: String,
    pub active_bank_sha256: String,
    pub evidence_bundle_sha256: String,
    pub causal_credit_sha256: String,
    pub field_ids: Vec<String>,
    pub requested_activation: BTreeMap<String, f64>,
    pub accepted_coefficients: Vec<f64>,
    pub trust_region: TrustRegionResult,
    pub projected_delta: DeltaArtifactRef,
    pub protection: GovernedCompositionProtection,
    /// The authenticated current observation that caused this request. Runtime
    /// activation without a measured source is intentionally not representable.
    pub source_observation_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
pub struct RecordedGovernedCognitiveComposition {
    pub state: CognitiveFieldState,
    pub route: FieldRoutingDecision,
    pub composition: RecordedGovernedComposition,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum CertificationStatus {
    Certified,
    Revoked,
}

impl CertificationStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Certified => "certified",
            Self::Revoked => "revoked",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "certified" => Some(Self::Certified),
            "revoked" => Some(Self::Revoked),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RuntimeIntegrityHealth {
    schema: String,
    canonical_runtime_config: bool,
    corpus_transition_clear: bool,
    ledger_verified: bool,
    bank_verified: bool,
    composition_ready: bool,
    observations_verified: bool,
    sleep_state_verified: bool,
    receipt_verified: bool,
    historical_artifacts_verified: bool,
    current_pointers_verified: bool,
    report_state_consistent: bool,
    current_corpus_bound: bool,
    analysis_current: bool,
    certified: bool,
    evidence_verified: bool,
    integrity_healthy: bool,
    execution_authorized: bool,
    operation_key: Option<String>,
    certification_status: Option<CertificationStatus>,
    integrity_reasons: Vec<String>,
    execution_blockers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BrainEngine {
    root: PathBuf,
    config: BrainConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CommitTransactionIntent {
    schema: String,
    operation_key: String,
    batch_digest: String,
    observation_digests: Vec<String>,
    report_sha256: String,
    report_promotable: bool,
    generation: u64,
    memory_sha256: String,
    shadow_bank_sha256: Option<String>,
    prior_shadow_bank_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CommitReceipt {
    schema: String,
    operation_key: String,
    batch_digest: String,
    report_sha256: String,
    memory_sha256: String,
    shadow_bank_sha256: Option<String>,
    ledger_event_hash: String,
    legacy_recovery: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SleepTransactionIntent {
    schema: String,
    operation_key: String,
    analysis_key: String,
    corpus_digest: String,
    analysis_version_digest: String,
    config_digest: String,
    report_sha256: String,
    memory_sha256: String,
    active_bank_sha256: Option<String>,
    sleep_state_sha256: String,
    evidence_bundle_sha256: Option<String>,
    evidence_verified: bool,
    certification_status: CertificationStatus,
    promoted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SleepReceipt {
    schema: String,
    operation_key: String,
    analysis_key: String,
    report_sha256: String,
    memory_sha256: String,
    active_bank_sha256: Option<String>,
    evidence_bundle_sha256: Option<String>,
    sleep_state_sha256: String,
    ledger_event_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LearningCorpusTransitionIntent {
    schema: String,
    operation_key: Sha256Digest,
    session_id: SessionId,
    adaptive_receipt_sha256: Sha256Digest,
    learning_finalization_input_sha256: Sha256Digest,
    representation_evidence_receipt: PrivateFileReference,
    representation_protocol_sha256: Sha256Digest,
    representation_observation_bindings_sha256: Sha256Digest,
    prior_corpus_digest: Sha256Digest,
    prior_observation_count: usize,
    new_corpus_digest: Sha256Digest,
    new_observation_count: usize,
    archive_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GovernedCompositionPointer {
    schema: String,
    operation_key: String,
    receipt_sha256: String,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256Digest::digest_bytes(bytes).into_string()
}

fn serialize_pretty_line<T: Serialize>(value: &T) -> BrainResult<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn file_sha256(path: &Path) -> BrainResult<String> {
    crate::artifact::sha256_file(path).map(Into::into)
}

/// Boolean health predicate for a content-addressed private artifact. Any
/// path, symlink, read, or digest failure is deliberately an unhealthy value.
fn private_file_digest_matches(root: &Path, path: &Path, expected_sha256: &str) -> bool {
    existing_regular_file_under_root(root, path)
        .and_then(|verified| {
            if file_sha256(&verified)? == expected_sha256 {
                Ok(())
            } else {
                Err(BrainError::Integrity("private_file_digest_mismatch".into()))
            }
        })
        .is_ok()
}

fn write_new_private(root: &Path, path: &Path, bytes: &[u8]) -> BrainResult<()> {
    let expected = Sha256Digest::digest_bytes(bytes);
    if write_or_verify_immutable(root, path, bytes)? != expected {
        return Err(BrainError::Integrity(
            "private_immutable_write_digest_mismatch".into(),
        ));
    }
    Ok(())
}

fn write_immutable_exact(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected_sha256: &str,
) -> BrainResult<()> {
    if sha256_bytes(bytes) != expected_sha256 {
        return Err(BrainError::Integrity(
            "private_immutable_expected_digest_mismatch".into(),
        ));
    }
    write_new_private(root, path, bytes)
}

/// Replace only an explicitly mutable current pointer. Historical
/// content-addressed artifacts use `write_new_private` and are never
/// overwritten once their identity is assigned.
fn replace_private_pointer_exact(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    expected_sha: &str,
) -> BrainResult<()> {
    let relative = root_relative_path(root, path)?;
    let permitted = [
        Path::new("state/memory/current.json"),
        Path::new("state/shadow_skill_bank.json"),
        Path::new("state/skill_bank.json"),
        Path::new("state/sleep_state.json"),
    ];
    if !permitted.contains(&relative.as_path()) {
        return Err(BrainError::Integrity(
            "transaction_pointer_target_not_allowlisted".into(),
        ));
    }
    let expected = Sha256Digest::parse(expected_sha)?;
    let installed = replace_private_file_atomic(root, path, bytes, Some(&expected))?;
    if installed != expected {
        return Err(BrainError::Integrity(
            "transaction_installed_digest_mismatch".into(),
        ));
    }
    Ok(())
}

fn valid_digest(value: &str) -> bool {
    Sha256Digest::is_valid_str(value)
}

impl ControllerInvocation {
    /// Validate the caller-supplied portion of a controller action. The
    /// controller coefficients and functional response are intentionally not
    /// part of this wire contract: both are rederived under engine authority.
    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.controller_invocation/v1"
            || self.state_before.is_empty()
            || self.state_before.iter().any(|value| !value.is_finite())
        {
            return Err(BrainError::Invalid(
                "controller_invocation_contract_invalid".into(),
            ));
        }
        Ok(())
    }
}

fn controller_execution_receipt_path(root: &Path, receipt_sha256: &str) -> PathBuf {
    root.join("state/controller_executions/by-sha")
        .join(format!("{receipt_sha256}.json"))
}

fn controller_execution_ledger_binding(
    root: &Path,
    receipt_sha256: &str,
    receipt: &ControllerExecutionReceipt,
) -> BrainResult<String> {
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "controller_execution_receipt",
        "receipt_sha256",
        receipt_sha256,
    )?
    .ok_or_else(|| BrainError::Integrity("controller_execution_receipt_ledger_missing".into()))?;
    let payload = event.payload()?;
    if payload.get("schema").and_then(Value::as_str)
        != Some("cerebro.tidex.controller_execution_ledger_binding/v1")
        || payload.get("receipt_sha256").and_then(Value::as_str) != Some(receipt_sha256)
        || payload.get("session_id").and_then(Value::as_str) != Some(receipt.session_id.as_str())
        || payload.get("invocation_sha256").and_then(Value::as_str)
            != Some(receipt.invocation_sha256.as_str())
        || payload
            .get("controller_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.controller_receipt_sha256.as_str())
        || payload.get("state_before_sha256").and_then(Value::as_str)
            != Some(receipt.state_before_sha256.as_str())
        || payload
            .get("promoted_observation_semantic_sha256")
            .and_then(Value::as_str)
            != Some(receipt.promoted_observation_semantic_sha256.as_str())
        || payload
            .get("governed_composition_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.governed_composition_receipt_sha256.as_str())
    {
        return Err(BrainError::Integrity(
            "controller_execution_receipt_ledger_payload_mismatch".into(),
        ));
    }
    Ok(event.event_hash)
}

fn persist_controller_execution(
    root: &Path,
    receipt: ControllerExecutionReceipt,
) -> BrainResult<RecordedControllerExecution> {
    let by_sha = root.join("state/controller_executions/by-sha");
    ensure_private_directory(root, &by_sha)?;
    let bytes = serialize_pretty_line(&receipt)?;
    let receipt_sha256 = sha256_bytes(&bytes);
    let receipt_path = controller_execution_receipt_path(root, &receipt_sha256);
    match fs::symlink_metadata(&receipt_path) {
        Ok(_) => {
            let path = existing_regular_file_under_root(root, &receipt_path)?;
            let bytes = fs::read(path)?;
            if sha256_bytes(&bytes) != receipt_sha256 {
                return Err(BrainError::Integrity(
                    "controller_execution_receipt_artifact_invalid".into(),
                ));
            }
            let existing: ControllerExecutionReceipt = serde_json::from_slice(&bytes)?;
            if existing != receipt {
                return Err(BrainError::Integrity(
                    "controller_execution_receipt_digest_collision".into(),
                ));
            }
            let ledger_event_hash =
                controller_execution_ledger_binding(root, &receipt_sha256, &existing)?;
            Ok(RecordedControllerExecution {
                receipt_path: receipt_path.to_string_lossy().into_owned(),
                receipt_sha256,
                ledger_event_hash,
                receipt: existing,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            write_new_private(root, &receipt_path, &bytes)?;
            let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
                root,
                "controller_execution_receipt",
                "receipt_sha256",
                &receipt_sha256,
            )? {
                let existing_hash =
                    controller_execution_ledger_binding(root, &receipt_sha256, &receipt)?;
                if existing.event_hash != existing_hash {
                    return Err(BrainError::Integrity(
                        "controller_execution_receipt_ledger_changed".into(),
                    ));
                }
                existing
            } else {
                ledger::append(
                    root,
                    "controller_execution_receipt",
                    json!({
                        "schema":"cerebro.tidex.controller_execution_ledger_binding/v1",
                        "receipt_sha256":&receipt_sha256,
                        "session_id":&receipt.session_id,
                        "invocation_sha256":&receipt.invocation_sha256,
                        "controller_receipt_sha256":&receipt.controller_receipt_sha256,
                        "state_before_sha256":&receipt.state_before_sha256,
                        "promoted_observation_semantic_sha256":&receipt.promoted_observation_semantic_sha256,
                        "governed_composition_receipt_sha256":&receipt.governed_composition_receipt_sha256,
                    }),
                )?
            };
            let ledger_event_hash =
                controller_execution_ledger_binding(root, &receipt_sha256, &receipt)?;
            if event.event_hash != ledger_event_hash {
                return Err(BrainError::Integrity(
                    "controller_execution_receipt_ledger_event_changed".into(),
                ));
            }
            Ok(RecordedControllerExecution {
                receipt_path: receipt_path.to_string_lossy().into_owned(),
                receipt_sha256,
                ledger_event_hash,
                receipt,
            })
        }
        Err(error) => Err(error.into()),
    }
}

fn verify_controller_execution_receipt(
    root: &Path,
    recorded: &RecordedControllerExecution,
) -> BrainResult<()> {
    let invocation = &recorded.receipt.invocation;
    invocation.validate()?;
    if !valid_digest(&recorded.receipt_sha256)
        || !valid_digest(&recorded.ledger_event_hash)
        || recorded.receipt.schema != "cerebro.tidex.controller_execution_receipt/v1"
        || recorded.receipt_path
            != controller_execution_receipt_path(root, &recorded.receipt_sha256).to_string_lossy()
        || recorded.receipt.session_id != invocation.session_id
        || recorded.receipt.invocation_sha256 != digest_json(invocation)?
        || recorded.receipt.state_before_sha256 != digest_json(&invocation.state_before)?
        || recorded.receipt.promoted_observation_semantic_sha256
            != invocation.promoted_observation_semantic_sha256
    {
        return Err(BrainError::Integrity(
            "controller_execution_receipt_contract_invalid".into(),
        ));
    }
    let receipt_path = existing_regular_file_under_root(
        root,
        &controller_execution_receipt_path(root, &recorded.receipt_sha256),
    )?;
    let receipt_bytes = fs::read(&receipt_path)?;
    if sha256_bytes(&receipt_bytes) != recorded.receipt_sha256
        || serde_json::from_slice::<ControllerExecutionReceipt>(&receipt_bytes)? != recorded.receipt
    {
        return Err(BrainError::Integrity(
            "controller_execution_receipt_bytes_mismatch".into(),
        ));
    }
    let current_controller =
        load_persisted_runtime_learned_controller(root, invocation.session_id.as_str())?;
    if current_controller.receipt_sha256 != recorded.receipt.controller_receipt_sha256.as_str() {
        return Err(BrainError::Integrity(
            "controller_execution_controller_receipt_stale".into(),
        ));
    }
    let governed_path = root
        .join("state/governed_compositions/by-sha")
        .join(format!(
            "{}.json",
            recorded.receipt.governed_composition_receipt_sha256
        ));
    let governed = load_verified_governed_composition_receipt(
        root,
        &governed_path,
        &recorded.receipt.governed_composition_receipt_sha256,
    )?;
    let engine = BrainEngine::open(root, BrainConfig::default())?;
    engine.require_canonical_runtime_config()?;
    let source = engine.load_current_observation_by_semantic_sha256(
        &invocation.promoted_observation_semantic_sha256,
    )?;
    let activation = engine.learned_controller_activation(
        &current_controller.receipt.runtime_controller,
        &invocation.state_before,
        &source.functional_response,
    )?;
    if governed.source_observation_sha256 != invocation.promoted_observation_semantic_sha256
        || governed.requested_activation != activation
    {
        return Err(BrainError::Integrity(
            "controller_execution_governed_composition_binding_invalid".into(),
        ));
    }
    let ledger_event_hash =
        controller_execution_ledger_binding(root, &recorded.receipt_sha256, &recorded.receipt)?;
    if ledger_event_hash != recorded.ledger_event_hash {
        return Err(BrainError::Integrity(
            "controller_execution_receipt_ledger_hash_mismatch".into(),
        ));
    }
    Ok(())
}

fn digest_json<T: Serialize>(v: &T) -> BrainResult<String> {
    let bytes = serde_json::to_vec(v)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn canonical_observations(observations: &[DeltaObservation]) -> Vec<DeltaObservation> {
    let mut canonical = observations.to_vec();
    canonical.sort_by(|left, right| {
        left.observation_id
            .cmp(&right.observation_id)
            .then_with(|| left.provenance_digest.cmp(&right.provenance_digest))
    });
    canonical
}

fn observation_set_digest(observations: &[DeltaObservation]) -> BrainResult<String> {
    let mut digests = observations
        .iter()
        .map(digest_json)
        .collect::<BrainResult<Vec<_>>>()?;
    digests.sort();
    digests.dedup();
    digest_json(&digests)
}

fn attach_evidence_support(
    fields: &mut [SkillField],
    source_mixtures: &[Vec<f64>],
    observations: &[DeltaObservation],
) -> BrainResult<()> {
    if fields.len() != source_mixtures.len() {
        return Err(BrainError::Integrity(
            "field_evidence_support_count_mismatch".into(),
        ));
    }
    for (field, mixture) in fields.iter_mut().zip(source_mixtures) {
        if mixture.len() != observations.len() {
            return Err(BrainError::Integrity(format!(
                "field_evidence_mixture_shape:{}",
                field.skill_id
            )));
        }
        let indices = source_support_indices(mixture)?;
        if indices.is_empty() {
            return Err(BrainError::Integrity(format!(
                "field_evidence_support_empty:{}",
                field.skill_id
            )));
        }
        let mut digests = indices
            .into_iter()
            .map(|index| digest_json(&observations[index]))
            .collect::<BrainResult<Vec<_>>>()?;
        digests.sort();
        digests.dedup();
        if digests.is_empty() || digests.iter().any(|digest| !valid_digest(digest)) {
            return Err(BrainError::Integrity(format!(
                "field_evidence_support_invalid:{}",
                field.skill_id
            )));
        }
        field.support = digests.len();
        field.evidence_support_digests = digests;
    }
    Ok(())
}

/// Identity for a TIDE-X reconstruction. Experimental producers may create
/// evidence, but their scripts and implementation details are not part of the
/// brain's algorithmic authority.
fn analysis_identity(config: &BrainConfig) -> BrainResult<(String, String, String)> {
    let source_tree_digest = env!("TIDEX_SOURCE_TREE_DIGEST").to_string();
    if !valid_digest(&source_tree_digest) {
        return Err(BrainError::Integrity("source_tree_digest_invalid".into()));
    }
    let config_digest = digest_json(config)?;
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:ANALYSIS:v6\0");
    hasher.update(source_tree_digest.as_bytes());
    hasher.update(config_digest.as_bytes());
    hasher.update(b"reconstruction/v6");
    let analysis_version_digest = format!("{:x}", hasher.finalize());
    Ok((source_tree_digest, config_digest, analysis_version_digest))
}

fn commit_operation_key(
    batch_digest: &str,
    report: &ReconstructionReport,
    report_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:COMMIT-OPERATION:v2\0");
    hasher.update(batch_digest.as_bytes());
    hasher.update(report.analysis_version_digest.as_bytes());
    hasher.update(report.config_digest.as_bytes());
    hasher.update(report_sha256.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn commit_transaction_payload(
    intent: &CommitTransactionIntent,
    observation_count: usize,
    report: &ReconstructionReport,
) -> Value {
    json!({
        "schema":"cerebro.tidex.commit_transaction/v2",
        "operation_key":intent.operation_key,
        "batch_digest":intent.batch_digest,
        "observation_count":observation_count,
        "observation_digests":intent.observation_digests,
        "report_sha256":intent.report_sha256,
        "report_promotable":intent.report_promotable,
        "generation":intent.generation,
        "memory_sha256":intent.memory_sha256,
        "shadow_bank_sha256":intent.shadow_bank_sha256,
        "prior_shadow_bank_sha256":intent.prior_shadow_bank_sha256,
        "summary":{
            "inverse_mode":report.inverse_mode,
            "selected_rank":report.selected_rank,
            "functional_cv_r2":report.functional_cv_r2,
            "promotion_allowed":report.promotion.allowed,
        }
    })
}

fn verify_commit_transaction_ledger_binding(
    event: &ledger::LedgerEvent,
    intent: &CommitTransactionIntent,
    observation_count: usize,
    report: &ReconstructionReport,
) -> BrainResult<()> {
    if event.payload()? != commit_transaction_payload(intent, observation_count, report) {
        return Err(BrainError::Integrity(
            "commit_transaction_ledger_payload_mismatch".into(),
        ));
    }
    Ok(())
}

fn sleep_analysis_key(corpus_digest: &str, report: &ReconstructionReport) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:SLEEP-ANALYSIS:v1\0");
    hasher.update(corpus_digest.as_bytes());
    hasher.update(report.analysis_version_digest.as_bytes());
    hasher.update(report.config_digest.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn sleep_operation_key(
    analysis_key: &str,
    evidence_sha: Option<&str>,
    certification_status: CertificationStatus,
    active_bank_sha: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:SLEEP-OPERATION:v1\0");
    hasher.update(analysis_key.as_bytes());
    hasher.update(evidence_sha.unwrap_or("none").as_bytes());
    hasher.update(certification_status.as_str().as_bytes());
    hasher.update(active_bank_sha.unwrap_or("none").as_bytes());
    format!("{:x}", hasher.finalize())
}

fn required_sleep_state_string<'a>(state: &'a Value, field: &str) -> BrainResult<&'a str> {
    state
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| BrainError::Integrity(format!("sleep_state_{field}_missing_or_invalid")))
}

/// Prove that a sleep receipt is bound to its own immutable ledger event and
/// to the exact state it claims to certify.  A bare ledger hash is never
/// sufficient: it could otherwise name an unrelated valid event.
fn verify_sleep_receipt_ledger_binding(
    root: &Path,
    receipt: &SleepReceipt,
    state: &Value,
    current_state_sha256: &str,
) -> BrainResult<()> {
    if receipt.schema != "cerebro.tidex.sleep_receipt/v1"
        || required_sleep_state_string(state, "schema")? != "cerebro.tidex.sleep_state/v5"
        || required_sleep_state_string(state, "operation_key")? != receipt.operation_key
        || required_sleep_state_string(state, "analysis_key")? != receipt.analysis_key
        || required_sleep_state_string(state, "report_sha256")? != receipt.report_sha256
        || required_sleep_state_string(state, "memory_digest")? != receipt.memory_sha256
        || current_state_sha256 != receipt.sleep_state_sha256
    {
        return Err(BrainError::Integrity(
            "sleep_receipt_state_binding_invalid".into(),
        ));
    }
    for field in [
        "corpus_digest",
        "source_tree_digest",
        "config_digest",
        "analysis_version_digest",
    ] {
        let _ = required_sleep_state_string(state, field)?;
    }
    let expected_bank = serde_json::to_value(&receipt.active_bank_sha256)?;
    let expected_evidence = serde_json::to_value(&receipt.evidence_bundle_sha256)?;
    if state.get("active_bank_sha256") != Some(&expected_bank)
        || state.get("evidence_bundle_sha256") != Some(&expected_evidence)
    {
        return Err(BrainError::Integrity(
            "sleep_receipt_optional_state_binding_invalid".into(),
        ));
    }
    let certification_status =
        CertificationStatus::parse(required_sleep_state_string(state, "certification_status")?)
            .ok_or_else(|| {
                BrainError::Integrity("sleep_state_certification_status_invalid".into())
            })?;
    let evidence_verified = state
        .get("evidence_verification")
        .and_then(|value| value.get("verified"))
        .and_then(Value::as_bool)
        .ok_or_else(|| BrainError::Integrity("sleep_state_evidence_verification_invalid".into()))?;
    let promoted = state
        .get("promoted")
        .and_then(Value::as_bool)
        .ok_or_else(|| BrainError::Integrity("sleep_state_promoted_invalid".into()))?;
    let expected_operation_key = sleep_operation_key(
        &receipt.analysis_key,
        receipt.evidence_bundle_sha256.as_deref(),
        certification_status,
        receipt.active_bank_sha256.as_deref(),
    );
    if receipt.operation_key != expected_operation_key {
        return Err(BrainError::Integrity(
            "sleep_receipt_operation_key_invalid".into(),
        ));
    }
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "sleep_transaction",
        "operation_key",
        &receipt.operation_key,
    )?
    .ok_or_else(|| BrainError::Integrity("sleep_receipt_ledger_event_missing".into()))?;
    let payload = event.payload()?;
    if event.event_hash != receipt.ledger_event_hash
        || payload.get("schema").and_then(Value::as_str)
            != Some("cerebro.tidex.sleep_transaction/v1")
        || payload.get("operation_key").and_then(Value::as_str)
            != Some(receipt.operation_key.as_str())
        || payload.get("analysis_key").and_then(Value::as_str)
            != Some(receipt.analysis_key.as_str())
        || payload.get("report_sha256").and_then(Value::as_str)
            != Some(receipt.report_sha256.as_str())
        || payload.get("memory_sha256").and_then(Value::as_str)
            != Some(receipt.memory_sha256.as_str())
        || payload.get("sleep_state_sha256").and_then(Value::as_str)
            != Some(receipt.sleep_state_sha256.as_str())
        || payload.get("active_bank_sha256") != Some(&expected_bank)
        || payload.get("evidence_bundle_sha256") != Some(&expected_evidence)
        || payload.get("evidence_verified").and_then(Value::as_bool) != Some(evidence_verified)
        || payload.get("certification_status").and_then(Value::as_str)
            != Some(certification_status.as_str())
        || payload.get("promoted").and_then(Value::as_bool) != Some(promoted)
    {
        return Err(BrainError::Integrity(
            "sleep_receipt_ledger_payload_invalid".into(),
        ));
    }
    for field in ["corpus_digest", "analysis_version_digest", "config_digest"] {
        if payload.get(field) != state.get(field) {
            return Err(BrainError::Integrity(format!(
                "sleep_receipt_ledger_{field}_mismatch"
            )));
        }
    }
    Ok(())
}

/// The transaction event is the one append-only proof that authorizes the
/// staged sleep artifacts to become current pointers.  Reusing a same-key
/// event with a different payload would otherwise mutate state before later
/// health checks notice the inconsistency.
fn verify_sleep_transaction_ledger_binding(
    event: &ledger::LedgerEvent,
    intent: &SleepTransactionIntent,
) -> BrainResult<()> {
    let expected = json!({
        "schema":"cerebro.tidex.sleep_transaction/v1",
        "operation_key":intent.operation_key,
        "analysis_key":intent.analysis_key,
        "corpus_digest":intent.corpus_digest,
        "analysis_version_digest":intent.analysis_version_digest,
        "config_digest":intent.config_digest,
        "report_sha256":intent.report_sha256,
        "memory_sha256":intent.memory_sha256,
        "active_bank_sha256":intent.active_bank_sha256,
        "sleep_state_sha256":intent.sleep_state_sha256,
        "evidence_bundle_sha256":intent.evidence_bundle_sha256,
        "evidence_verified":intent.evidence_verified,
        "certification_status":intent.certification_status,
        "promoted":intent.promoted,
    });
    if event.payload()? != expected {
        return Err(BrainError::Integrity(
            "sleep_transaction_ledger_payload_mismatch".into(),
        ));
    }
    Ok(())
}

fn learning_finalization_operation_key(
    learning_finalization_input_sha256: &Sha256Digest,
    prior_corpus_digest: &Sha256Digest,
    new_corpus_digest: &Sha256Digest,
) -> BrainResult<Sha256Digest> {
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:LEARNING-FINALIZATION:v1\0");
    hasher.update(learning_finalization_input_sha256.as_str().as_bytes());
    hasher.update(prior_corpus_digest.as_str().as_bytes());
    hasher.update(new_corpus_digest.as_str().as_bytes());
    Sha256Digest::parse(format!("{:x}", hasher.finalize()))
}

fn governed_composition_operation_key(
    report_sha256: &str,
    active_bank_sha256: &str,
    evidence_bundle_sha256: &str,
    causal_credit_sha256: &str,
    activation: &BTreeMap<String, f64>,
    projected_delta_sha256: &str,
    source_observation_sha256: &str,
) -> BrainResult<String> {
    let activation_bytes = serde_json::to_vec(activation)?;
    let mut hasher = Sha256::new();
    hasher.update(b"CEREBRO:TIDEX:GOVERNED-COMPOSITION:v1\0");
    for value in [
        report_sha256,
        active_bank_sha256,
        evidence_bundle_sha256,
        causal_credit_sha256,
        projected_delta_sha256,
        source_observation_sha256,
    ] {
        hasher.update(value.as_bytes());
        hasher.update([0]);
    }
    hasher.update((activation_bytes.len() as u64).to_be_bytes());
    hasher.update(activation_bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

const LEARNING_FINALIZATION_ARCHIVE_LABELS: &[&str] = &[
    "observations_manifest.json",
    "skill_bank.json",
    "shadow_skill_bank.json",
    "memory_current.json",
    "sleep_state.json",
    "sleep_evidence_current.json",
];

fn learning_finalization_receipt_path(root: &Path, operation_key: &Sha256Digest) -> PathBuf {
    root.join("state/learning_finalizations")
        .join(format!("{operation_key}.json"))
}

fn learning_finalization_archive_dir(root: &Path, operation_key: &Sha256Digest) -> PathBuf {
    root.join("state/corpus_transitions/by-operation")
        .join(operation_key.as_str())
}

fn archive_label_is_allowed(label: &str) -> bool {
    LEARNING_FINALIZATION_ARCHIVE_LABELS.contains(&label)
}

fn archive_private_state_file(
    root: &Path,
    source: &Path,
    archive: &Path,
    label: &str,
    expected_sha256: Option<&Sha256Digest>,
    archived: &mut BTreeMap<String, Sha256Digest>,
) -> BrainResult<()> {
    let source_parent = source.parent().ok_or_else(|| {
        BrainError::Invalid("learning_corpus_transition_source_parent_missing".into())
    })?;
    existing_directory_under_root(root, source_parent)?;
    match fs::symlink_metadata(source) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && expected_sha256.is_none() => {
            return Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(BrainError::Integrity(format!(
                "learning_corpus_transition_required_state_file_missing:{label}"
            )))
        }
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    if !archive_label_is_allowed(label) {
        return Err(BrainError::Integrity(format!(
            "learning_corpus_transition_state_file_invalid:{label}"
        )));
    }
    let source = existing_regular_file_under_root(root, source)?;
    ensure_private_parent(root, archive)?;
    match fs::symlink_metadata(archive) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(BrainError::Integrity(format!(
                "learning_corpus_transition_state_file_invalid:{label}"
            )))
        }
        Err(error) => return Err(error.into()),
    }
    let digest = crate::artifact::sha256_file(&source)?;
    if expected_sha256.is_some_and(|expected| digest != *expected) {
        return Err(BrainError::Integrity(format!(
            "learning_corpus_transition_prior_state_file_changed:{label}"
        )));
    }
    fs::rename(&source, archive)?;
    secure_file(archive)?;
    let archive = existing_regular_file_under_root(root, archive)?;
    if crate::artifact::sha256_file(&archive)? != digest {
        return Err(BrainError::Integrity(format!(
            "learning_corpus_transition_state_file_digest_mismatch:{label}"
        )));
    }
    archived.insert(label.to_string(), digest);
    Ok(())
}

/// Read a persisted observation corpus from a confined directory, retaining the
/// filename-to-semantic-identity check for both the live and archived corpus.
/// This is deliberately shared so archive verification cannot be weaker than
/// runtime loading.
fn load_observations_from_private_directory(
    root: &Path,
    directory: &Path,
    missing_is_empty: bool,
) -> BrainResult<Vec<DeltaObservation>> {
    match fs::symlink_metadata(directory) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && missing_is_empty => {
            return Ok(Vec::new())
        }
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    let directory = existing_directory_under_root(root, directory)?;
    let mut paths = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            return Err(BrainError::Integrity(
                "persisted_observations_entry_invalid".into(),
            ));
        }
        paths.push(existing_regular_file_under_root(root, &path)?);
    }
    paths.sort();
    let mut out = Vec::new();
    let mut ids = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for path in paths {
        let observation: DeltaObservation = serde_json::from_slice(&fs::read(&path)?)?;
        let digest = digest_json(&observation)?;
        let expected_name = format!("{}-{}.json", observation.observation_id, &digest[..16]);
        if !valid_observation_id(&observation.observation_id)
            || path.file_name().and_then(|value| value.to_str()) != Some(expected_name.as_str())
            || !ids.insert(observation.observation_id.clone())
            || !digests.insert(digest)
        {
            return Err(BrainError::Integrity(
                "persisted_observation_identity_invalid".into(),
            ));
        }
        out.push(observation);
    }
    Ok(out)
}

fn canonical_observation_digests(observations: &[DeltaObservation]) -> BrainResult<Vec<String>> {
    let mut digests = observations
        .iter()
        .map(digest_json)
        .collect::<BrainResult<Vec<_>>>()?;
    digests.sort();
    digests.dedup();
    if digests.len() != observations.len() {
        return Err(BrainError::Integrity(
            "learning_finalization_observation_digest_duplicate".into(),
        ));
    }
    Ok(digests)
}

fn verify_learning_finalization_commit_binding(
    root: &Path,
    receipt: &LearningFinalizationReceipt,
    commit: &CommitReceipt,
    report: &ReconstructionReport,
    observations: &[DeltaObservation],
) -> BrainResult<()> {
    let observation_digests = canonical_observation_digests(observations)?;
    let expected_batch_digest = digest_json(&observation_digests)?;
    let generation = observations
        .iter()
        .map(|observation| observation.generation)
        .max()
        .unwrap_or(0);
    let memory = build_memory_snapshot(
        observations,
        generation,
        true,
        &report.fields,
        &report.skill_source_mixtures,
    )?;
    let memory_sha = Sha256Digest::digest_bytes(&serialize_pretty_line(&memory)?);
    let mut shadow_bank = SkillBank::default();
    assimilate_bank(
        &mut shadow_bank,
        &report.fields,
        BrainConfig::default().skill_match_cosine,
    )?;
    let shadow_bank_sha = Sha256Digest::digest_bytes(&serialize_pretty_line(&shadow_bank)?);
    let expected_operation_key = commit_operation_key(
        &expected_batch_digest,
        report,
        receipt.report_sha256.as_str(),
    );
    if expected_batch_digest != receipt.new_corpus_digest.as_str()
        || commit.schema != "cerebro.tidex.commit_receipt/v2"
        || commit.legacy_recovery
        || commit.operation_key != receipt.commit_operation_key.as_str()
        || commit.operation_key != expected_operation_key
        || commit.batch_digest != expected_batch_digest
        || commit.report_sha256 != receipt.report_sha256.as_str()
        || commit.memory_sha256 != memory_sha.as_str()
        || commit.shadow_bank_sha256.as_deref() != Some(shadow_bank_sha.as_str())
    {
        return Err(BrainError::Integrity(
            "learning_finalization_commit_receipt_contract_invalid".into(),
        ));
    }
    PrivateFileReference::new(
        memory_artifact_path(root, &commit.memory_sha256),
        memory_sha,
    )
    .verify(root)?;
    PrivateFileReference::new(
        root.join("state/skill_banks/by-sha")
            .join(format!("{shadow_bank_sha}.json")),
        shadow_bank_sha.clone(),
    )
    .verify(root)?;
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "commit_transaction",
        "operation_key",
        &commit.operation_key,
    )?
    .ok_or_else(|| BrainError::Integrity("learning_finalization_commit_ledger_missing".into()))?;
    let expected_intent = CommitTransactionIntent {
        schema: "cerebro.tidex.commit_transaction_intent/v2".into(),
        operation_key: expected_operation_key,
        batch_digest: expected_batch_digest,
        observation_digests,
        report_sha256: receipt.report_sha256.to_string(),
        report_promotable: true,
        generation,
        memory_sha256: commit.memory_sha256.clone(),
        shadow_bank_sha256: Some(shadow_bank_sha.to_string()),
        prior_shadow_bank_sha256: None,
    };
    if event.event_hash != commit.ledger_event_hash {
        return Err(BrainError::Integrity(
            "learning_finalization_commit_ledger_binding_invalid".into(),
        ));
    }
    verify_commit_transaction_ledger_binding(&event, &expected_intent, observations.len(), report)?;
    Ok(())
}

fn verify_learning_finalization_archive(
    root: &Path,
    receipt: &LearningFinalizationReceipt,
    transition_intent_dir: &Path,
) -> BrainResult<()> {
    // These are live authority pointers in every authenticated prior runtime
    // state. A finalization may not silently omit one after deciding that the
    // prior corpus is revoked; optional shadow state remains optional because
    // a prior non-promoting sleep legitimately has none.
    let required = [
        "observations_manifest.json",
        "skill_bank.json",
        "memory_current.json",
        "sleep_state.json",
        "sleep_evidence_current.json",
    ];
    if required
        .iter()
        .any(|label| !receipt.archived_artifact_sha256.contains_key(*label))
        || receipt
            .archived_artifact_sha256
            .keys()
            .any(|label| !archive_label_is_allowed(label))
    {
        return Err(BrainError::Integrity(
            "learning_finalization_archive_manifest_contract_invalid".into(),
        ));
    }
    let archive_dir = learning_finalization_archive_dir(root, &receipt.operation_key);
    let archive_dir = existing_directory_under_root(root, &archive_dir)?;
    let allowed_entries = receipt
        .archived_artifact_sha256
        .keys()
        .map(String::as_str)
        .chain(["observations", "transition_intent"])
        .collect::<BTreeSet<_>>();
    for entry in fs::read_dir(&archive_dir)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|_| {
            BrainError::Integrity("learning_finalization_archive_name_invalid".into())
        })?;
        if !allowed_entries.contains(name.as_str()) {
            return Err(BrainError::Integrity(
                "learning_finalization_archive_unexpected_entry".into(),
            ));
        }
    }

    let mut manifest = None;
    for (label, digest) in &receipt.archived_artifact_sha256 {
        let bytes = PrivateFileReference::new(archive_dir.join(label), digest.clone())
            .read_verified(root)?;
        if label == "observations_manifest.json" {
            manifest = Some(serde_json::from_slice::<Vec<DeltaObservation>>(&bytes)?);
        }
    }
    let manifest = manifest.ok_or_else(|| {
        BrainError::Integrity("learning_finalization_archive_manifest_missing".into())
    })?;
    let manifest = canonical_observations(&manifest);
    let archived = canonical_observations(&load_observations_from_private_directory(
        root,
        &archive_dir.join("observations"),
        false,
    )?);
    if manifest != archived
        || archived.len() != receipt.prior_observation_count
        || observation_set_digest(&archived)? != receipt.prior_corpus_digest.as_str()
    {
        return Err(BrainError::Integrity(
            "learning_finalization_archive_observation_binding_invalid".into(),
        ));
    }

    verify_learning_finalization_transition_intent(
        root,
        transition_intent_dir,
        receipt,
        &archive_dir,
    )?;
    Ok(())
}

fn expected_learning_finalization_transition_intent(
    receipt: &LearningFinalizationReceipt,
    archive_dir: &Path,
) -> LearningCorpusTransitionIntent {
    LearningCorpusTransitionIntent {
        schema: "cerebro.tidex.learning_corpus_transition_intent/v1".into(),
        operation_key: receipt.operation_key.clone(),
        session_id: receipt.session_id.clone(),
        adaptive_receipt_sha256: receipt.adaptive_receipt_sha256.clone(),
        learning_finalization_input_sha256: receipt.learning_finalization_input_sha256.clone(),
        representation_evidence_receipt: receipt.representation_evidence_receipt.clone(),
        representation_protocol_sha256: receipt.representation_protocol_sha256.clone(),
        representation_observation_bindings_sha256: receipt
            .representation_observation_bindings_sha256
            .clone(),
        prior_corpus_digest: receipt.prior_corpus_digest.clone(),
        prior_observation_count: receipt.prior_observation_count,
        new_corpus_digest: receipt.new_corpus_digest.clone(),
        new_observation_count: receipt.new_observation_count,
        archive_dir: archive_dir.to_path_buf(),
    }
}

fn verify_learning_finalization_transition_intent(
    root: &Path,
    intent_dir: &Path,
    receipt: &LearningFinalizationReceipt,
    archive_dir: &Path,
) -> BrainResult<()> {
    let intent_dir = existing_directory_under_root(root, intent_dir)?;
    let mut entries = fs::read_dir(&intent_dir)?;
    let entry = entries.next().transpose()?.ok_or_else(|| {
        BrainError::Integrity("learning_finalization_transition_intent_missing".into())
    })?;
    if entries.next().transpose()?.is_some()
        || entry.file_name().as_encoded_bytes() != b"intent.json"
    {
        return Err(BrainError::Integrity(
            "learning_finalization_transition_intent_directory_invalid".into(),
        ));
    }
    let intent_path = existing_regular_file_under_root(root, &entry.path())?;
    let intent: LearningCorpusTransitionIntent = serde_json::from_slice(&fs::read(intent_path)?)?;
    if intent != expected_learning_finalization_transition_intent(receipt, archive_dir) {
        return Err(BrainError::Integrity(
            "learning_finalization_archive_intent_binding_invalid".into(),
        ));
    }
    Ok(())
}

fn close_verified_learning_finalization_inflight(
    root: &Path,
    receipt: &LearningFinalizationReceipt,
    reference: &PrivateFileReference,
) -> BrainResult<()> {
    let inflight_root = root.join("state/corpus_transitions/inflight");
    match fs::symlink_metadata(&inflight_root) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    let inflight_root = existing_directory_under_root(root, &inflight_root)?;
    let mut matching_inflight = None;
    for entry in fs::read_dir(&inflight_root)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|_| {
            BrainError::Integrity("learning_finalization_inflight_name_invalid".into())
        })?;
        if name != receipt.operation_key.as_str() {
            return Err(BrainError::Integrity(
                "learning_finalization_unrelated_inflight_transition".into(),
            ));
        }
        if matching_inflight.replace(entry.path()).is_some() {
            return Err(BrainError::Integrity(
                "learning_finalization_inflight_transition_ambiguous".into(),
            ));
        }
    }
    let Some(inflight) = matching_inflight else {
        return Ok(());
    };
    let archive_dir = learning_finalization_archive_dir(root, &receipt.operation_key);
    let archive_dir = existing_directory_under_root(root, &archive_dir)?;
    verify_learning_finalization_transition_intent(root, &inflight, receipt, &archive_dir)?;
    // Before closing the only visible fail-closed marker, replay every other
    // authority binding using this exact inflight intent.  A syntactically
    // plausible receipt must never make an invalid half-transition disappear.
    let _ = load_verified_learning_finalization_receipt_under_root(
        root,
        reference,
        Some(&receipt.operation_key),
    )?;
    let closed_intent = archive_dir.join("transition_intent");
    match fs::symlink_metadata(&closed_intent) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(BrainError::Integrity(
                "learning_finalization_recovery_closed_intent_collision".into(),
            ))
        }
        Err(error) => return Err(error.into()),
    }
    fs::rename(&inflight, &closed_intent)?;
    secure_dir(&closed_intent)?;
    verify_learning_finalization_transition_intent(root, &closed_intent, receipt, &archive_dir)
}

fn verify_learning_finalization_ledger_binding(
    root: &Path,
    receipt: &LearningFinalizationReceipt,
) -> BrainResult<()> {
    let event = ledger::find_v2_event_by_payload_string(
        root,
        "learning_corpus_transition",
        "operation_key",
        receipt.operation_key.as_str(),
    )?
    .ok_or_else(|| BrainError::Integrity("learning_finalization_receipt_ledger_missing".into()))?;
    let payload = event.payload()?;
    let representation_path = serde_json::to_value(&receipt.representation_evidence_receipt.path)?;
    let archived_artifacts = serde_json::to_value(&receipt.archived_artifact_sha256)?;
    if event.event_hash != receipt.ledger_event_hash.as_str()
        || payload.get("schema").and_then(Value::as_str)
            != Some("cerebro.tidex.learning_corpus_transition/v1")
        || payload.get("operation_key").and_then(Value::as_str)
            != Some(receipt.operation_key.as_str())
        || payload.get("session_id").and_then(Value::as_str) != Some(receipt.session_id.as_str())
        || payload
            .get("adaptive_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.adaptive_receipt_sha256.as_str())
        || payload
            .get("learning_finalization_input_sha256")
            .and_then(Value::as_str)
            != Some(receipt.learning_finalization_input_sha256.as_str())
        || payload.get("representation_evidence_receipt_path") != Some(&representation_path)
        || payload
            .get("representation_evidence_receipt_sha256")
            .and_then(Value::as_str)
            != Some(receipt.representation_evidence_receipt.sha256.as_str())
        || payload
            .get("representation_protocol_sha256")
            .and_then(Value::as_str)
            != Some(receipt.representation_protocol_sha256.as_str())
        || payload
            .get("representation_observation_bindings_sha256")
            .and_then(Value::as_str)
            != Some(receipt.representation_observation_bindings_sha256.as_str())
        || payload.get("prior_corpus_digest").and_then(Value::as_str)
            != Some(receipt.prior_corpus_digest.as_str())
        || payload
            .get("prior_observation_count")
            .and_then(Value::as_u64)
            != Some(receipt.prior_observation_count as u64)
        || payload.get("new_corpus_digest").and_then(Value::as_str)
            != Some(receipt.new_corpus_digest.as_str())
        || payload.get("new_observation_count").and_then(Value::as_u64)
            != Some(receipt.new_observation_count as u64)
        || payload.get("report_sha256").and_then(Value::as_str)
            != Some(receipt.report_sha256.as_str())
        || payload.get("commit_operation_key").and_then(Value::as_str)
            != Some(receipt.commit_operation_key.as_str())
        || payload.get("commit_receipt_sha256").and_then(Value::as_str)
            != Some(receipt.commit_receipt_sha256.as_str())
        || payload.get("archived_artifact_sha256") != Some(&archived_artifacts)
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_ledger_binding_invalid".into(),
        ));
    }
    Ok(())
}

/// Load a content-addressed governed-composition receipt and rebind it to the
/// currently active certified corpus. This is public for controller
/// supervision, but it never trusts a caller-provided runtime controller or
/// coefficient vector.
pub fn load_verified_governed_composition_receipt(
    root: impl AsRef<Path>,
    receipt_path: impl AsRef<Path>,
    receipt_sha256: &str,
) -> BrainResult<GovernedCompositionReceipt> {
    let root = verify_private_root(root.as_ref())?;
    if !valid_digest(receipt_sha256) {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_digest_invalid".into(),
        ));
    }
    let expected_receipt_path = root
        .join("state/governed_compositions/by-sha")
        .join(format!("{}.json", receipt_sha256.to_ascii_lowercase()));
    if receipt_path.as_ref() != expected_receipt_path {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_identity_invalid".into(),
        ));
    }
    let receipt_reference =
        PrivateFileReference::new(expected_receipt_path, Sha256Digest::parse(receipt_sha256)?);
    let (_canonical, receipt_bytes) = receipt_reference.read_verified_with_path(&root)?;
    let receipt: GovernedCompositionReceipt = serde_json::from_slice(&receipt_bytes)?;
    if receipt.schema != "cerebro.tidex.governed_composition_receipt/v1"
        || !valid_digest(&receipt.report_sha256)
        || !valid_digest(&receipt.active_bank_sha256)
        || !valid_digest(&receipt.evidence_bundle_sha256)
        || !valid_digest(&receipt.causal_credit_sha256)
        || receipt.field_ids.is_empty()
        || receipt.field_ids.iter().any(|id| id.trim().is_empty())
        || receipt.field_ids.iter().collect::<BTreeSet<_>>().len() != receipt.field_ids.len()
        || receipt.accepted_coefficients.len() != receipt.field_ids.len()
        || receipt
            .accepted_coefficients
            .iter()
            .any(|value| !value.is_finite())
        || receipt.trust_region.accepted_coefficients != receipt.accepted_coefficients
        || receipt.trust_region.allocation_policy
            != TrustRegionAllocationPolicy::CausalPriorityContractionV1
        || receipt
            .trust_region
            .causal_priority_weights
            .as_ref()
            .is_none_or(|weights| {
                weights.len() != receipt.field_ids.len()
                    || weights
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
            })
        || receipt.trust_region.component_retention.len() != receipt.field_ids.len()
        || receipt
            .trust_region
            .component_retention
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || !receipt.protection.allowed
        || !receipt.protection.damage_ratio.is_finite()
        || !receipt.protection.removed_energy.is_finite()
        || !receipt.protection.max_weighted_residual.is_finite()
        || !valid_digest(&receipt.source_observation_sha256)
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_contract_invalid".into(),
        ));
    }
    let expected_projected_path = root.join("artifacts/deltas/by-sha").join(format!(
        "{}.dvec",
        receipt.projected_delta.sha256.to_ascii_lowercase()
    ));
    if receipt.projected_delta.path != expected_projected_path {
        return Err(BrainError::Integrity(
            "governed_composition_projected_delta_path_invalid".into(),
        ));
    }
    let projected = existing_regular_file_under_root(&root, &expected_projected_path)?;
    let inspected = inspect_dvec(&projected)?;
    if inspected.sha256 != receipt.projected_delta.sha256.to_ascii_lowercase()
        || inspected.parameter_count != receipt.projected_delta.parameter_count
    {
        return Err(BrainError::Integrity(
            "governed_composition_projected_delta_identity_invalid".into(),
        ));
    }
    let event = ledger::find_v2_event_by_payload_string(
        &root,
        "governed_composition_receipt",
        "receipt_sha256",
        receipt_sha256,
    )?
    .ok_or_else(|| BrainError::Integrity("governed_composition_receipt_ledger_missing".into()))?;
    let payload = event.payload()?;
    if payload.get("schema").and_then(serde_json::Value::as_str)
        != Some("cerebro.tidex.governed_composition_ledger_binding/v1")
        || payload
            .get("receipt_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt_sha256)
        || payload
            .get("operation_key")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.operation_key.as_str())
        || payload
            .get("report_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.report_sha256.as_str())
        || payload
            .get("active_bank_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.active_bank_sha256.as_str())
        || payload
            .get("evidence_bundle_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.evidence_bundle_sha256.as_str())
        || payload
            .get("causal_credit_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.causal_credit_sha256.as_str())
        || payload
            .get("source_observation_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.source_observation_sha256.as_str())
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_ledger_binding_invalid".into(),
        ));
    }
    let bank_path = root.join("state/skill_bank.json");
    let bank_path = existing_regular_file_under_root(&root, &bank_path)?;
    if file_sha256(&bank_path)? != receipt.active_bank_sha256 {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_active_bank_invalid".into(),
        ));
    }
    let bank: SkillBank = serde_json::from_slice(&fs::read(&bank_path)?)?;
    if bank
        .fields
        .iter()
        .map(|field| field.skill_id.clone())
        .collect::<Vec<_>>()
        != receipt.field_ids
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_field_identity_invalid".into(),
        ));
    }
    let sleep_path = existing_regular_file_under_root(&root, &root.join("state/sleep_state.json"))?;
    let sleep_state: serde_json::Value = serde_json::from_slice(&fs::read(sleep_path)?)?;
    if sleep_state
        .get("certification_status")
        .and_then(serde_json::Value::as_str)
        != Some("certified")
        || sleep_state
            .get("report_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.report_sha256.as_str())
        || sleep_state
            .get("active_bank_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.active_bank_sha256.as_str())
        || sleep_state
            .get("evidence_bundle_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.evidence_bundle_sha256.as_str())
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_sleep_binding_invalid".into(),
        ));
    }
    let evidence = load_sleep_evidence(&root)?;
    if evidence.causal_credit.credit_source_sha256 != receipt.causal_credit_sha256 {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_causal_binding_invalid".into(),
        ));
    }

    // A receipt is evidence of a prior decision, not authority by itself.
    // Reconstruct the decision under the *current* certified CEREBRO state and
    // compare every executable output before a controller may use it.
    let engine = BrainEngine::open(&root, BrainConfig::default())?;
    engine.require_no_incomplete_corpus_transition()?;
    engine.require_current_certification()?;
    let source = receipt.source_observation_sha256.as_str();
    engine.require_current_observation_digest(source)?;
    let recomposed = engine.compose(&receipt.requested_activation)?;
    let expected_protection = GovernedCompositionProtection {
        damage_ratio: recomposed.protection.damage_ratio,
        allowed: recomposed.protection.allowed,
        removed_energy: recomposed.protection.removed_energy,
        protected_rank: recomposed.protection.protected_rank,
        max_weighted_residual: recomposed.protection.max_weighted_residual,
    };
    if recomposed.trust_region != receipt.trust_region || expected_protection != receipt.protection
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_recomposition_mismatch".into(),
        ));
    }
    let expected_operation_key = governed_composition_operation_key(
        &receipt.report_sha256,
        &receipt.active_bank_sha256,
        &receipt.evidence_bundle_sha256,
        &receipt.causal_credit_sha256,
        &receipt.requested_activation,
        &receipt.projected_delta.sha256,
        source,
    )?;
    if expected_operation_key != receipt.operation_key {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_operation_key_mismatch".into(),
        ));
    }
    let expected_values = recomposed
        .delta
        .iter()
        .map(|value| {
            if !value.is_finite() || value.abs() > f32::MAX as f64 {
                return Err(BrainError::Numerical(
                    "governed_composition_projected_delta_nonrepresentable".into(),
                ));
            }
            Ok(*value as f32)
        })
        .collect::<BrainResult<Vec<_>>>()?;
    let recorded_values = read_dvec_f32(&receipt.projected_delta)?;
    if recorded_values.len() != expected_values.len()
        || recorded_values
            .iter()
            .zip(&expected_values)
            .any(|(recorded, expected)| recorded.to_bits() != expected.to_bits())
    {
        return Err(BrainError::Integrity(
            "governed_composition_receipt_projected_delta_mismatch".into(),
        ));
    }
    Ok(receipt)
}

/// Load and replay the receipt that made a completed adaptive-learning session
/// the active corpus. The producer is irrelevant to this authority boundary:
/// only the immutable learning/representation evidence chain is trusted.
pub fn load_verified_learning_finalization_receipt(
    root: impl AsRef<Path>,
    reference: &PrivateFileReference,
) -> BrainResult<LearningFinalizationReceipt> {
    let root = verify_private_root(root.as_ref())?;
    load_verified_learning_finalization_receipt_under_root(&root, reference, None)
}

/// Internal replay variant used only to authenticate the one matching inflight
/// intent before it is moved into an immutable archive after a crash.  Normal
/// callers always pass `None`, which requires no inflight transition at all.
fn load_verified_learning_finalization_receipt_under_root(
    root: &Path,
    reference: &PrivateFileReference,
    permitted_inflight_operation: Option<&Sha256Digest>,
) -> BrainResult<LearningFinalizationReceipt> {
    let (canonical, receipt_bytes) = reference.read_verified_with_path(root)?;
    let receipt: LearningFinalizationReceipt = serde_json::from_slice(&receipt_bytes)?;
    let expected = learning_finalization_receipt_path(root, &receipt.operation_key);
    if canonical != expected
        || receipt.schema != "cerebro.tidex.learning_finalization_receipt/v1"
        || receipt.representation_observation_bindings.is_empty()
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_contract_invalid".into(),
        ));
    }

    let input = prepare_learning_finalization(
        root,
        receipt.session_id.as_str(),
        &receipt.representation_evidence_receipt.path,
    )?;
    let bindings_digest = Sha256Digest::digest_bytes(&serde_json::to_vec(
        &input.representation_observation_bindings,
    )?);
    if learning_finalization_input_sha256(&input)? != receipt.learning_finalization_input_sha256
        || input.adaptive_receipt_sha256 != receipt.adaptive_receipt_sha256
        || input.representation_evidence_receipt != receipt.representation_evidence_receipt
        || input.representation_protocol_sha256 != receipt.representation_protocol_sha256
        || bindings_digest != receipt.representation_observation_bindings_sha256
        || input.representation_observation_bindings != receipt.representation_observation_bindings
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_input_replay_mismatch".into(),
        ));
    }

    let expected_operation_key = learning_finalization_operation_key(
        &receipt.learning_finalization_input_sha256,
        &receipt.prior_corpus_digest,
        &receipt.new_corpus_digest,
    )?;
    if receipt.operation_key != expected_operation_key {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_operation_key_invalid".into(),
        ));
    }

    let input_observations = canonical_observations(&input.observations);
    if observation_set_digest(&input_observations)? != receipt.new_corpus_digest
        || input_observations.len() != receipt.new_observation_count
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_input_corpus_mismatch".into(),
        ));
    }
    let engine = BrainEngine::open(root, BrainConfig::default())?;
    engine.require_canonical_runtime_config()?;
    match permitted_inflight_operation {
        None => engine.require_no_incomplete_corpus_transition()?,
        Some(operation_key) if *operation_key == receipt.operation_key => {
            let inflight = root
                .join("state/corpus_transitions/inflight")
                .join(operation_key.as_str());
            let _ = existing_directory_under_root(root, &inflight)?;
        }
        Some(_) => {
            return Err(BrainError::Integrity(
                "learning_finalization_recovery_operation_mismatch".into(),
            ))
        }
    }
    let current = canonical_observations(&engine.load_persisted_observations()?);
    if observation_set_digest(&current)? != receipt.new_corpus_digest
        || current.len() != receipt.new_observation_count
    {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_current_corpus_mismatch".into(),
        ));
    }
    let current_by_id = current
        .iter()
        .map(|observation| {
            Ok((
                observation.observation_id.clone(),
                Sha256Digest::parse(digest_json(observation)?)?,
            ))
        })
        .collect::<BrainResult<BTreeMap<_, _>>>()?;
    let mut source_ids = BTreeSet::new();
    let mut source_hashes = BTreeSet::new();
    let mut promoted_hashes = BTreeSet::new();
    for binding in &receipt.representation_observation_bindings {
        binding.adaptive_source_observation.verify(root)?;
        binding
            .representation_destination_observation
            .verify(root)?;
        if !source_ids.insert(binding.observation_id.as_str())
            || !source_hashes.insert(binding.adaptive_source_observation.sha256.as_str())
            || !promoted_hashes.insert(binding.promoted_observation_semantic_sha256.as_str())
            || current_by_id.get(binding.observation_id.as_str())
                != Some(&binding.promoted_observation_semantic_sha256)
        {
            return Err(BrainError::Integrity(
                "learning_finalization_receipt_observation_binding_invalid".into(),
            ));
        }
    }
    if source_ids.len() != current_by_id.len() || current_by_id.len() != input_observations.len() {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_observation_binding_coverage_invalid".into(),
        ));
    }

    let report_path = root
        .join("state/reports")
        .join(format!("{}.json", receipt.report_sha256));
    let report_reference = PrivateFileReference::new(report_path, receipt.report_sha256.clone());
    let report: ReconstructionReport =
        serde_json::from_slice(&report_reference.read_verified(root)?)?;
    if report.observation_set_digest != receipt.new_corpus_digest || !report.promotion.allowed {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_report_corpus_invalid".into(),
        ));
    }
    let mut rederived_report = engine.analyze_canonical(&input_observations)?;
    if !rederived_report.promotion.allowed {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_rederived_report_not_promotable".into(),
        ));
    }
    let mut recorded_report_without_dense = report.clone();
    for field in &mut recorded_report_without_dense.fields {
        field.dense_materialization = None;
    }
    for field in &mut rederived_report.fields {
        field.dense_materialization = None;
    }
    if recorded_report_without_dense != rederived_report {
        return Err(BrainError::Integrity(
            "learning_finalization_receipt_report_rederivation_mismatch".into(),
        ));
    }
    engine.verify_dense_field_materializations(
        &report.fields,
        &report.skill_source_mixtures,
        &input_observations,
    )?;
    let commit_path = root
        .join("state/commits")
        .join(format!("{}.json", receipt.commit_operation_key));
    let commit_reference =
        PrivateFileReference::new(commit_path, receipt.commit_receipt_sha256.clone());
    let commit: CommitReceipt = serde_json::from_slice(&commit_reference.read_verified(root)?)?;
    verify_learning_finalization_commit_binding(
        root,
        &receipt,
        &commit,
        &report,
        &input_observations,
    )?;
    let transition_intent_dir = match permitted_inflight_operation {
        None => learning_finalization_archive_dir(root, &receipt.operation_key)
            .join("transition_intent"),
        Some(operation_key) => root
            .join("state/corpus_transitions/inflight")
            .join(operation_key.as_str()),
    };
    verify_learning_finalization_archive(root, &receipt, &transition_intent_dir)?;
    verify_learning_finalization_ledger_binding(root, &receipt)?;
    Ok(receipt)
}

fn validate_brain_config(config: &BrainConfig) -> BrainResult<()> {
    let minimum_observations_required = config
        .min_independent_apertures
        .checked_mul(2)
        .ok_or_else(|| BrainError::Invalid("brain_config_observation_count_overflow".into()))?;
    if config.min_independent_apertures < 2
        || config.min_observations < minimum_observations_required
        || config.max_rank == 0
        || !config.target_explained_variance.is_finite()
        || !(0.0..=1.0).contains(&config.target_explained_variance)
        || !config.ridge.is_finite()
        || config.ridge <= 0.0
        || !config.huber_delta.is_finite()
        || config.huber_delta <= 0.0
        || config.irls_rounds == 0
        || !config.skill_match_cosine.is_finite()
        || !(0.0..=1.0).contains(&config.skill_match_cosine)
        || !config.min_functional_cv_r2.is_finite()
        || config.min_functional_cv_r2 > 1.0
        || !config.max_cycle_rms.is_finite()
        || config.max_cycle_rms < 0.0
        || !config.min_skill_coherence.is_finite()
        || !(0.0..=1.0).contains(&config.min_skill_coherence)
        || !config.min_skill_persistence.is_finite()
        || !(0.0..=1.0).contains(&config.min_skill_persistence)
        || !config.min_field_explained_variance.is_finite()
        || !(0.0..=1.0).contains(&config.min_field_explained_variance)
        || !config.max_condition_estimate.is_finite()
        || config.max_condition_estimate < 1.0
        || !config
            .max_spectral_normalized_reconstruction_rms
            .is_finite()
        || config.max_spectral_normalized_reconstruction_rms < 0.0
        || !config.min_identifiability_signal_to_noise.is_finite()
        || config.min_identifiability_signal_to_noise <= 0.0
        || !config.min_representation_match_accuracy.is_finite()
        || !(0.0..=1.0).contains(&config.min_representation_match_accuracy)
        || !config.min_representation_match_margin.is_finite()
        || config.min_representation_match_margin < 0.0
    {
        return Err(BrainError::Invalid("brain_config_invalid".into()));
    }
    Ok(())
}

impl BrainEngine {
    pub fn open(root: impl AsRef<Path>, config: BrainConfig) -> BrainResult<Self> {
        validate_brain_config(&config)?;
        let root = verify_private_root(root.as_ref())?;
        Ok(Self { root, config })
    }

    /// Analysis can be parameterized for controlled tests, but every state
    /// mutation, certification, composition, and activation is governed by the
    /// one canonical production configuration. This prevents a caller from
    /// self-attesting relaxed promotion gates under a new config digest.
    fn require_canonical_runtime_config(&self) -> BrainResult<()> {
        if self.config != BrainConfig::default() {
            return Err(BrainError::Integrity(
                "noncanonical_runtime_config_forbidden".into(),
            ));
        }
        Ok(())
    }

    fn load_parameter_layout(&self, digest: &str) -> BrainResult<ParameterBlockLayout> {
        if !valid_digest(digest) {
            return Err(BrainError::Invalid(
                "parameter_layout_digest_invalid".into(),
            ));
        }
        let path = self
            .root
            .join("state/parameter_layouts/by-sha")
            .join(format!("{}.json", digest.to_ascii_lowercase()));
        let path = existing_regular_file_under_root(&self.root, &path).map_err(|_| {
            BrainError::Integrity("parameter_layout_artifact_missing_or_symlink".into())
        })?;
        if file_sha256(&path)? != digest.to_ascii_lowercase() {
            return Err(BrainError::Integrity(
                "parameter_layout_digest_mismatch".into(),
            ));
        }
        let layout: ParameterBlockLayout = serde_json::from_slice(&fs::read(&path)?)?;
        if layout.schema != "cerebro.tidex.parameter_block_layout/v1"
            || layout.blocks.is_empty()
            || layout.total_parameter_count == 0
        {
            return Err(BrainError::Integrity(
                "parameter_layout_contract_invalid".into(),
            ));
        }
        let mut expected_offset = 0u64;
        let mut names = BTreeSet::new();
        for block in &layout.blocks {
            if block.name.trim().is_empty()
                || !names.insert(block.name.as_str())
                || block.offset != expected_offset
                || block.count == 0
                || block.shape.is_empty()
                || block.shape.contains(&0)
            {
                return Err(BrainError::Integrity(
                    "parameter_layout_block_invalid".into(),
                ));
            }
            let shape_count = block.shape.iter().try_fold(1usize, |acc, value| {
                acc.checked_mul(*value)
                    .ok_or_else(|| BrainError::Invalid("parameter_layout_shape_overflow".into()))
            })?;
            if shape_count != block.count {
                return Err(BrainError::Integrity(
                    "parameter_layout_shape_count_mismatch".into(),
                ));
            }
            expected_offset = expected_offset
                .checked_add(block.count as u64)
                .ok_or_else(|| BrainError::Invalid("parameter_layout_offset_overflow".into()))?;
        }
        if expected_offset != layout.total_parameter_count {
            return Err(BrainError::Integrity(
                "parameter_layout_total_count_mismatch".into(),
            ));
        }
        Ok(layout)
    }

    /// Resolve only an engine-owned, content-addressed dense delta. Runtime
    /// SkillFields must never point to an arbitrary readable file, even if its
    /// bytes happen to form a valid dvec artifact.
    fn verified_private_dvec(&self, reference: &DeltaArtifactRef) -> BrainResult<PathBuf> {
        if !valid_digest(&reference.sha256) || reference.parameter_count == 0 {
            return Err(BrainError::Integrity(
                "runtime_dense_reference_contract_invalid".into(),
            ));
        }
        let supplied = Path::new(&reference.path);
        let expected = self
            .root
            .join("artifacts/deltas/by-sha")
            .join(format!("{}.dvec", reference.sha256.to_ascii_lowercase()));
        if supplied != expected {
            return Err(BrainError::Integrity(
                "runtime_dense_reference_path_invalid".into(),
            ));
        }
        let expected = existing_regular_file_under_root(&self.root, &expected)?;
        let inspected = inspect_dvec(&expected)?;
        if inspected.sha256 != reference.sha256.to_ascii_lowercase()
            || inspected.parameter_count != reference.parameter_count
        {
            return Err(BrainError::Integrity(
                "runtime_dense_reference_identity_mismatch".into(),
            ));
        }
        Ok(expected)
    }

    fn structured_sources(
        &self,
        observations: &[DeltaObservation],
    ) -> BrainResult<Option<(String, ParameterBlockLayout, Vec<StructuredSource>)>> {
        let populated = observations
            .iter()
            .filter(|observation| {
                observation.dense_artifact.is_some()
                    || observation.parameter_layout_sha256.is_some()
            })
            .count();
        if populated == 0 {
            if self.config.require_structured_geometry_for_promotion {
                return Err(BrainError::Invalid(
                    "structured_geometry_dense_evidence_required".into(),
                ));
            }
            return Ok(None);
        }
        if populated != observations.len() {
            return Err(BrainError::Invalid(
                "structured_geometry_partial_dense_evidence".into(),
            ));
        }
        let layout_ids = observations
            .iter()
            .map(|observation| {
                observation
                    .parameter_layout_sha256
                    .clone()
                    .ok_or_else(|| BrainError::Invalid("parameter_layout_reference_missing".into()))
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if layout_ids.len() != 1 {
            return Err(BrainError::Invalid(
                "structured_geometry_multiple_parameter_layouts".into(),
            ));
        }
        let layout_sha = layout_ids.into_iter().next().unwrap();
        let layout = self.load_parameter_layout(&layout_sha)?;
        let mut sources = Vec::with_capacity(observations.len());
        for observation in observations {
            let artifact = observation
                .dense_artifact
                .clone()
                .ok_or_else(|| BrainError::Invalid("dense_artifact_reference_missing".into()))?;
            let path = self.verified_private_dvec(&artifact)?;
            let inspected = inspect_dvec(&path)?;
            if inspected.sha256 != artifact.sha256.to_ascii_lowercase()
                || inspected.parameter_count != artifact.parameter_count
                || inspected.parameter_count != layout.total_parameter_count
            {
                return Err(BrainError::Integrity(format!(
                    "dense_artifact_reference_mismatch:{}",
                    observation.observation_id
                )));
            }
            sources.push(StructuredSource {
                observation_id: observation.observation_id.clone(),
                artifact,
                reliability: observation.reliability,
            });
        }
        Ok(Some((layout_sha, layout, sources)))
    }

    fn materialize_dense_fields(
        &self,
        fields: &mut [SkillField],
        source_mixtures: &[Vec<f64>],
        observations: &[DeltaObservation],
    ) -> BrainResult<()> {
        if fields.is_empty()
            || fields.len() != source_mixtures.len()
            || source_mixtures
                .iter()
                .any(|mixture| mixture.len() != observations.len())
        {
            return Err(BrainError::Invalid(
                "dense_field_materialization_shape".into(),
            ));
        }
        let (layout_sha, layout, _) = self
            .structured_sources(observations)?
            .ok_or_else(|| BrainError::Invalid("dense_field_sources_required".into()))?;
        for (field_index, field) in fields.iter_mut().enumerate() {
            if self.config.require_structured_geometry_for_promotion
                && field.structured_geometry.is_none()
            {
                return Err(BrainError::Integrity(format!(
                    "dense_field_missing_structured_geometry:{}",
                    field.skill_id
                )));
            }
            if field.parameter_layout_sha256.as_deref() != Some(layout_sha.as_str()) {
                return Err(BrainError::Integrity(format!(
                    "dense_field_layout_identity_mismatch:{}",
                    field.skill_id
                )));
            }
            let mixture = &source_mixtures[field_index];
            let support = source_support_indices(mixture)?;
            let mut sources = Vec::with_capacity(support.len());
            for index in support {
                let reference = observations[index].dense_artifact.clone().ok_or_else(|| {
                    BrainError::Integrity("dense_field_source_artifact_missing".into())
                })?;
                sources.push((reference, mixture[index]));
            }
            if sources.is_empty() {
                return Err(BrainError::Numerical(format!(
                    "dense_field_materialization_support_empty:{}",
                    field.skill_id
                )));
            }
            let materialized = combine_content_addressed_dvec(&self.root, &sources)?;
            if materialized.parameter_count != layout.total_parameter_count {
                return Err(BrainError::Integrity(format!(
                    "dense_field_materialized_count_mismatch:{}",
                    field.skill_id
                )));
            }
            field.dense_materialization = Some(materialized);
        }
        Ok(())
    }

    /// Re-derive dense field references without creating artifacts.  A receipt
    /// verifier must never materialize a missing candidate as a side effect:
    /// the exact content-addressed dvec must already exist and match the same
    /// f64-to-f32 arithmetic used at promotion time.
    fn verify_dense_field_materializations(
        &self,
        fields: &[SkillField],
        source_mixtures: &[Vec<f64>],
        observations: &[DeltaObservation],
    ) -> BrainResult<()> {
        if fields.is_empty()
            || fields.len() != source_mixtures.len()
            || source_mixtures
                .iter()
                .any(|mixture| mixture.len() != observations.len())
        {
            return Err(BrainError::Integrity(
                "dense_field_materialization_verification_shape".into(),
            ));
        }
        let (layout_sha, layout, _) = self
            .structured_sources(observations)?
            .ok_or_else(|| BrainError::Integrity("dense_field_sources_required".into()))?;
        for (field, mixture) in fields.iter().zip(source_mixtures) {
            if self.config.require_structured_geometry_for_promotion
                && field.structured_geometry.is_none()
            {
                return Err(BrainError::Integrity(format!(
                    "dense_field_missing_structured_geometry:{}",
                    field.skill_id
                )));
            }
            if field.parameter_layout_sha256.as_deref() != Some(layout_sha.as_str()) {
                return Err(BrainError::Integrity(format!(
                    "dense_field_layout_identity_mismatch:{}",
                    field.skill_id
                )));
            }
            let reference = field.dense_materialization.as_ref().ok_or_else(|| {
                BrainError::Integrity(format!(
                    "dense_field_materialization_missing:{}",
                    field.skill_id
                ))
            })?;
            let support = source_support_indices(mixture)?;
            if support.is_empty() {
                return Err(BrainError::Integrity(format!(
                    "dense_field_materialization_support_empty:{}",
                    field.skill_id
                )));
            }
            let sources = support
                .into_iter()
                .map(|index| {
                    let source = observations[index].dense_artifact.clone().ok_or_else(|| {
                        BrainError::Integrity("dense_field_source_artifact_missing".into())
                    })?;
                    Ok((source, mixture[index]))
                })
                .collect::<BrainResult<Vec<_>>>()?;
            let derived = derive_content_addressed_dvec_combination(&self.root, &sources)?;
            if &derived != reference || reference.parameter_count != layout.total_parameter_count {
                return Err(BrainError::Integrity(format!(
                    "dense_field_materialization_rederivation_mismatch:{}",
                    field.skill_id
                )));
            }
            self.verified_private_dvec(reference)?;
        }
        Ok(())
    }

    fn representation_observations(
        &self,
        observations: &[DeltaObservation],
    ) -> BrainResult<Option<(String, Vec<RepresentationObservation>)>> {
        let populated = observations
            .iter()
            .filter(|observation| {
                observation.representation_artifact.is_some()
                    || observation.representation_protocol_sha256.is_some()
            })
            .count();
        if populated == 0 {
            if self.config.require_dual_space_for_promotion {
                return Err(BrainError::Invalid(
                    "dual_space_representation_evidence_required".into(),
                ));
            }
            return Ok(None);
        }
        if populated != observations.len() {
            return Err(BrainError::Invalid(
                "dual_space_partial_representation_evidence".into(),
            ));
        }
        let protocol_ids = observations
            .iter()
            .map(|observation| {
                observation
                    .representation_protocol_sha256
                    .clone()
                    .ok_or_else(|| {
                        BrainError::Invalid("representation_protocol_reference_missing".into())
                    })
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if protocol_ids.len() != 1 {
            return Err(BrainError::Invalid(
                "dual_space_multiple_representation_protocols".into(),
            ));
        }
        let protocol_sha = protocol_ids.into_iter().next().unwrap();
        if !valid_digest(&protocol_sha) {
            return Err(BrainError::Invalid(
                "representation_protocol_digest_invalid".into(),
            ));
        }
        let protocol_path = self
            .root
            .join("state/representation_protocols/by-sha")
            .join(format!("{}.json", protocol_sha.to_ascii_lowercase()));
        let protocol_path =
            existing_regular_file_under_root(&self.root, &protocol_path).map_err(|_| {
                BrainError::Integrity("representation_protocol_artifact_invalid".into())
            })?;
        if file_sha256(&protocol_path)? != protocol_sha.to_ascii_lowercase() {
            return Err(BrainError::Integrity(
                "representation_protocol_artifact_invalid".into(),
            ));
        }
        let protocol: serde_json::Value = serde_json::from_slice(&fs::read(&protocol_path)?)?;
        let string_digest = |key: &str| -> BrainResult<String> {
            let value = protocol
                .get(key)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    BrainError::Integrity(format!("representation_protocol_missing_digest:{key}"))
                })?;
            if !valid_digest(value) {
                return Err(BrainError::Integrity(format!(
                    "representation_protocol_invalid_digest:{key}"
                )));
            }
            Ok(value.to_string())
        };
        if protocol.get("schema").and_then(serde_json::Value::as_str)
            != Some("cerebro.tidex.representation_protocol/v1")
            || protocol
                .get("task_labels_used")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
            || protocol
                .get("probe_vocabulary_overlap")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|values| !values.is_empty())
        {
            return Err(BrainError::Integrity(
                "representation_protocol_semantic_contract_invalid".into(),
            ));
        }
        let probe_sha = string_digest("probe_sha256")?;
        let probe_text_sha = string_digest("probe_text_sha256")?;
        let _forbidden_vocab_sha = string_digest("forbidden_vocabulary_sha256")?;
        let _source_representation_sha = string_digest("source_representation_sha256")?;
        if probe_sha != probe_text_sha {
            return Err(BrainError::Integrity(
                "representation_probe_text_digest_mismatch".into(),
            ));
        }
        let sketch_dim = protocol
            .get("sketch_dim")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_sketch_dim_invalid".into())
            })?;
        let probe_count = protocol
            .get("probe_count")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_probe_count_invalid".into())
            })?;
        let layer_count = protocol
            .get("layer_count")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_layer_count_invalid".into())
            })?;
        let hidden_dim = protocol
            .get("hidden_dim")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_hidden_dim_invalid".into())
            })?;
        let raw_dimension = protocol
            .get("raw_dimension_per_observation")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                BrainError::Integrity("representation_protocol_raw_dim_invalid".into())
            })?;
        let expected_raw = probe_count
            .checked_mul(layer_count)
            .and_then(|value| value.checked_mul(hidden_dim))
            .ok_or_else(|| {
                BrainError::Invalid("representation_protocol_dimension_overflow".into())
            })?;
        if expected_raw != raw_dimension {
            return Err(BrainError::Integrity(
                "representation_protocol_raw_dimension_mismatch".into(),
            ));
        }
        let mut result = Vec::with_capacity(observations.len());
        for observation in observations {
            let reference = observation
                .representation_artifact
                .as_ref()
                .ok_or_else(|| {
                    BrainError::Invalid("representation_artifact_reference_missing".into())
                })?;
            let path = existing_regular_file_under_root(&self.root, Path::new(&reference.path))
                .map_err(|_| {
                    BrainError::Integrity("representation_artifact_path_invalid".into())
                })?;
            if path != Path::new(&reference.path) {
                return Err(BrainError::Integrity(
                    "representation_artifact_path_normalization_mismatch".into(),
                ));
            }
            if reference.element_count != sketch_dim {
                return Err(BrainError::Integrity(format!(
                    "representation_artifact_count_mismatch:{}",
                    observation.observation_id
                )));
            }
            let shift = read_f64_artifact(reference)?;
            if shift.len() != sketch_dim as usize {
                return Err(BrainError::Integrity(
                    "representation_artifact_loaded_count_mismatch".into(),
                ));
            }
            result.push(RepresentationObservation {
                observation_id: observation.observation_id.clone(),
                shift,
            });
        }
        Ok(Some((protocol_sha, result)))
    }

    fn validate_observations(&self, obs: &[DeltaObservation]) -> BrainResult<usize> {
        if obs.len() < self.config.min_observations {
            return Err(BrainError::Invalid("minimum_six_observations".into()));
        }
        let dim = obs[0].delta.len();
        if dim == 0 {
            return Err(BrainError::Invalid("empty_delta".into()));
        }
        let mut ids = BTreeSet::new();
        for o in obs {
            if !valid_observation_id(&o.observation_id) {
                return Err(BrainError::Invalid("observation_id_invalid".into()));
            }
            if !ids.insert(o.observation_id.as_str()) {
                return Err(BrainError::Invalid("duplicate_observation_id".into()));
            }
            if o.from_checkpoint.trim().is_empty()
                || o.to_checkpoint.trim().is_empty()
                || o.from_checkpoint == o.to_checkpoint
            {
                return Err(BrainError::Invalid("checkpoint_edge_invalid".into()));
            }
            if o.delta.len() != dim || o.delta.iter().any(|v| !v.is_finite()) {
                return Err(BrainError::Invalid("delta_invalid".into()));
            }
            if !o.reliability.is_finite() || !(0.0..=1.0).contains(&o.reliability) {
                return Err(BrainError::Invalid("reliability_invalid".into()));
            }
            if o.independence_group.trim().is_empty() {
                return Err(BrainError::Invalid("independence_group_required".into()));
            }
            let lineage = &o.experiment_lineage;
            if lineage.run_id.trim().is_empty()
                || lineage.replicate_id.trim().is_empty()
                || lineage.randomization_id.trim().is_empty()
                || !valid_digest(&lineage.dataset_split_digest)
                || !valid_digest(&lineage.initial_checkpoint_digest)
                || !valid_digest(&lineage.optimizer_config_digest)
                || !valid_digest(&lineage.template_config_digest)
            {
                return Err(BrainError::Invalid("experiment_lineage_required".into()));
            }
            if !valid_digest(&o.provenance_digest) {
                return Err(BrainError::Invalid("provenance_digest_invalid".into()));
            }
            if o.functional_response.iter().any(|v| !v.is_finite()) {
                return Err(BrainError::Invalid("functional_response_non_finite".into()));
            }
        }
        Ok(dim)
    }

    pub fn analyze(&self, observations: &[DeltaObservation]) -> BrainResult<ReconstructionReport> {
        let canonical = canonical_observations(observations);
        self.analyze_canonical(&canonical)
    }

    fn analyze_canonical(&self, obs: &[DeltaObservation]) -> BrainResult<ReconstructionReport> {
        let dim = self.validate_observations(obs)?;
        let (source_tree_digest, config_digest, analysis_version_digest) =
            analysis_identity(&self.config)?;
        let observation_set_digest = observation_set_digest(obs)?;
        let sbas = reconstruct_trajectory(obs, self.config.ridge)?;
        let conf = remove_confounders(obs, self.config.ridge)?;
        let aperture_independence =
            estimate_aperture_independence(obs, self.config.min_independent_apertures)?;
        let group_independence_weights = aperture_independence
            .group_ids
            .iter()
            .cloned()
            .zip(
                aperture_independence
                    .group_independence_weights
                    .iter()
                    .copied(),
            )
            .collect::<BTreeMap<_, _>>();
        let raw = Matrix::from_rows(
            &obs.iter()
                .map(|observation| observation.delta.clone())
                .collect::<Vec<_>>(),
        )?;
        let mut group_counts = BTreeMap::<&str, usize>::new();
        for o in obs {
            *group_counts
                .entry(o.independence_group.as_str())
                .or_default() += 1;
        }
        let weights = obs
            .iter()
            .map(|o| {
                let independence_weight = group_independence_weights
                    .get(o.independence_group.as_str())
                    .copied()
                    .unwrap_or(0.0);
                o.reliability.clamp(1e-4, 1.0) * independence_weight
                    / (group_counts[o.independence_group.as_str()] as f64).sqrt()
            })
            .collect::<Vec<_>>();
        let groups = obs
            .iter()
            .map(|o| o.independence_group.clone())
            .collect::<Vec<_>>();
        let generation = obs.iter().map(|o| o.generation).max().unwrap_or(0);

        // Hypothesis A: energy/spectral tomography over confounder-residualized
        // deltas. This remains useful when capabilities are continuously mixed.
        let mut spectral =
            reconstruct_skill_fields(&conf.residuals, &weights, &groups, generation, &self.config)?;
        let spectral_functional = fit_functional_map(
            &spectral.coefficients,
            obs,
            self.config.ridge,
            self.config.min_independent_apertures,
        )?;
        attach_signatures(&mut spectral.fields, &spectral_functional)?;
        let input_rms = (obs
            .iter()
            .flat_map(|o| o.delta.iter())
            .map(|v| v * v)
            .sum::<f64>()
            / (obs.len() * dim) as f64)
            .sqrt()
            .max(1e-15);
        let spectral_normalized_rms = spectral.reconstruction_rms / input_rms;
        let transform_t = conf.source_transform.transpose();
        let mut spectral_source_mixtures = Vec::with_capacity(spectral.source_mixtures.len());
        for mix in &spectral.source_mixtures {
            spectral_source_mixtures.push(transform_t.matvec(mix)?);
        }

        // Hypothesis B: persistent-skill tomography. Discovery uses parameter
        // geometry only and receives no task names or functional outcomes.
        // Functional identity is attached and validated afterwards by holding
        // out one independent aperture at a time.
        let persistent_attempt = reconstruct_persistent_skill_fields(
            &raw,
            obs,
            generation,
            self.config.min_observations,
            self.config.min_independent_apertures,
        );
        let persistent_error = persistent_attempt.as_ref().err().map(ToString::to_string);
        let use_persistent = persistent_attempt
            .as_ref()
            .is_ok_and(|persistent| persistent.functional_cv_r2 > spectral_functional.cv_r2);

        let (
            inverse_mode,
            mut fields,
            skill_source_mixtures,
            selected_rank,
            effective_rank,
            condition_estimate,
            reconstruction_rms,
            normalized_reconstruction_rms,
            functional_cv_r2,
            selected_coefficients,
        ) = if use_persistent {
            let persistent = persistent_attempt
                .as_ref()
                .expect("persistent inverse was selected only after success");
            (
                ReconstructionInverseMode::Persistent,
                persistent.fields.clone(),
                persistent.source_mixtures.clone(),
                persistent.selected_rank,
                persistent.effective_rank,
                persistent.condition_estimate,
                persistent.reconstruction_rms,
                persistent.normalized_reconstruction_rms,
                persistent.functional_cv_r2,
                persistent.coefficients.clone(),
            )
        } else {
            (
                ReconstructionInverseMode::Spectral,
                spectral.fields.clone(),
                spectral_source_mixtures,
                spectral.selected_rank,
                spectral.effective_rank,
                spectral.condition_estimate,
                spectral.reconstruction_rms,
                spectral_normalized_rms,
                spectral_functional.cv_r2,
                spectral.coefficients.clone(),
            )
        };

        attach_evidence_support(&mut fields, &skill_source_mixtures, obs)?;

        let persistent = persistent_attempt.as_ref().ok();
        let persistent_functional_cv_r2 = persistent.map(|value| value.functional_cv_r2);
        let persistent_coherence_threshold = persistent.map(|value| value.coherence_threshold);
        let persistent_coherence_gap = persistent.map(|value| value.coherence_gap);
        let persistent_coverage_ratio = persistent.map(|value| value.coverage_ratio);
        let persistent_cluster_stability = persistent.map(|value| value.cluster_stability);
        let persistent_parametric_cluster_stability =
            persistent.map(|value| value.parametric_cluster_stability);
        let persistent_functional_cluster_stability =
            persistent.map(|value| value.functional_cluster_stability);
        let persistent_cluster_identity_min_margin =
            persistent.map(|value| value.cluster_identity_min_margin);
        let persistent_cluster_assignment_consistent =
            persistent.map(|value| value.cluster_assignment_consistent);
        let persistent_min_holdout_similarity =
            persistent.map(|value| value.min_holdout_similarity);
        let persistent_cluster_sizes = persistent
            .map(|value| value.cluster_sizes.clone())
            .unwrap_or_default();
        let resolution_map = resolution_map(
            &fields,
            &selected_coefficients,
            &weights,
            reconstruction_rms,
            self.config.ridge,
            self.config.min_identifiability_signal_to_noise,
        )?;

        let mut structured_block_count = 0usize;
        let mut structured_max_local_rank = 0usize;
        let mut structured_mean_effective_rank = 0.0f64;
        if let Some((layout_sha, layout, sources)) = self.structured_sources(obs)? {
            let geometry = reconstruct_structured_geometry(
                &fields,
                &skill_source_mixtures,
                &sources,
                &layout,
                &self.config,
            )?;
            if geometry.skills.len() != fields.len()
                || geometry.block_count != layout.blocks.len()
                || geometry.total_parameter_count != layout.total_parameter_count
            {
                return Err(BrainError::Integrity(
                    "structured_geometry_result_contract_mismatch".into(),
                ));
            }
            let numerical_tolerance =
                f64::EPSILON.sqrt() * (layout.blocks.len().max(1) as f64).sqrt() * 16.0;
            let mut effective_rank_sum = 0.0;
            for (field, skill_geometry) in fields.iter_mut().zip(geometry.skills) {
                if skill_geometry.skill_id != field.skill_id
                    || skill_geometry.blocks.len() != layout.blocks.len()
                {
                    return Err(BrainError::Integrity(
                        "structured_geometry_skill_identity_mismatch".into(),
                    ));
                }
                let mut active_blocks = 0usize;
                for (block_geometry, block_layout) in
                    skill_geometry.blocks.iter().zip(&layout.blocks)
                {
                    if block_geometry.block_name != block_layout.name
                        || block_geometry.offset != block_layout.offset
                        || block_geometry.count != block_layout.count
                        || block_geometry.shape != block_layout.shape
                        || block_geometry.selected_rank != block_geometry.axes.len()
                    {
                        return Err(BrainError::Integrity(
                            "structured_geometry_block_contract_mismatch".into(),
                        ));
                    }
                    if block_geometry.block_energy > f64::EPSILON {
                        active_blocks += 1;
                        if block_geometry.selected_rank == 0
                            || block_geometry.retained_energy + numerical_tolerance
                                < self.config.target_explained_variance
                            || !block_geometry.effective_rank.is_finite()
                            || block_geometry.effective_rank <= 0.0
                        {
                            return Err(BrainError::Numerical(
                                "structured_geometry_active_block_unresolved".into(),
                            ));
                        }
                    }
                    for axis in &block_geometry.axes {
                        if axis.source_coefficients.len() != obs.len()
                            || !axis.singular_value.is_finite()
                            || axis.singular_value <= 0.0
                            || axis
                                .source_coefficients
                                .iter()
                                .any(|value| !value.is_finite())
                        {
                            return Err(BrainError::Integrity(
                                "structured_geometry_axis_contract_mismatch".into(),
                            ));
                        }
                    }
                }
                if active_blocks == 0 || skill_geometry.max_local_rank == 0 {
                    return Err(BrainError::Numerical(
                        "structured_geometry_skill_has_no_active_blocks".into(),
                    ));
                }
                structured_block_count = structured_block_count.max(active_blocks);
                structured_max_local_rank =
                    structured_max_local_rank.max(skill_geometry.max_local_rank);
                effective_rank_sum += skill_geometry.mean_effective_rank;
                field.parameter_layout_sha256 = Some(layout_sha.clone());
                field.structured_geometry = Some(skill_geometry);
            }
            structured_mean_effective_rank = effective_rank_sum / fields.len().max(1) as f64;
        }

        let mut reasons = Vec::new();
        if sbas.cycle_rms > self.config.max_cycle_rms {
            reasons.push("cycle_consistency_failed".into());
        }
        if functional_cv_r2 < self.config.min_functional_cv_r2 {
            reasons.push("functional_cross_validation_failed".into());
        }
        if fields.is_empty() {
            reasons.push("no_skill_fields".into());
        }
        let eligible = fields
            .iter()
            .filter(|field| field.explained_variance >= self.config.min_field_explained_variance)
            .collect::<Vec<_>>();
        if eligible
            .iter()
            .any(|field| field.coherence < self.config.min_skill_coherence)
        {
            reasons.push("skill_coherence_failed".into());
        }
        if eligible
            .iter()
            .any(|field| field.persistence < self.config.min_skill_persistence)
        {
            reasons.push("skill_persistence_failed".into());
        }
        if group_counts.len() < self.config.min_independent_apertures {
            reasons.push("insufficient_declared_apertures".into());
        }
        if !aperture_independence.independent_enough {
            reasons.push("aperture_independence_unresolved".into());
        }
        if !resolution_map.all_fields_resolved {
            reasons.push("skill_identifiability_unresolved".into());
        }
        if !condition_estimate.is_finite()
            || condition_estimate > self.config.max_condition_estimate
        {
            reasons.push("tomography_ill_conditioned".into());
        }
        if inverse_mode == ReconstructionInverseMode::Spectral
            && normalized_reconstruction_rms
                > self.config.max_spectral_normalized_reconstruction_rms
        {
            reasons.push("reconstruction_error_high".into());
        }
        if inverse_mode == ReconstructionInverseMode::Persistent {
            let persistent = persistent.ok_or_else(|| {
                BrainError::Integrity("persistent_inverse_selected_without_result".into())
            })?;
            if persistent.coverage_ratio.to_bits() != 1.0_f64.to_bits() {
                reasons.push("persistent_observation_coverage_incomplete".into());
            }
            let identity_tolerance =
                f64::EPSILON.sqrt() * (persistent.selected_rank.max(1) as f64).sqrt() * 16.0;
            if !persistent.cluster_assignment_consistent {
                reasons.push("persistent_cluster_assignment_inconsistent_across_spaces".into());
            }
            if persistent.cluster_identity_min_margin <= identity_tolerance {
                reasons.push("persistent_cluster_identity_margin_unresolved".into());
            }
            if persistent.coherence_gap <= 1e-12 {
                reasons.push("persistent_coherence_gap_not_identifiable".into());
            }
            if persistent.min_holdout_similarity <= 0.0 {
                reasons.push("persistent_holdout_alignment_nonpositive".into());
            }
        }

        if self.config.require_structured_geometry_for_promotion
            && fields.iter().any(|field| {
                field.structured_geometry.is_none() || field.parameter_layout_sha256.is_none()
            })
        {
            reasons.push("structured_geometry_required_for_promotion".into());
        }

        let mut metrics = BTreeMap::new();
        metrics.insert("cycle_rms".into(), sbas.cycle_rms);
        metrics.insert("functional_cv_r2".into(), functional_cv_r2);
        metrics.insert(
            "spectral_functional_cv_r2".into(),
            spectral_functional.cv_r2,
        );
        if let Some(value) = persistent_functional_cv_r2 {
            metrics.insert("persistent_functional_cv_r2".into(), value);
        }
        if let Some(value) = persistent_cluster_stability {
            metrics.insert("persistent_cluster_stability".into(), value);
        }
        if let Some(value) = persistent_parametric_cluster_stability {
            metrics.insert("persistent_parametric_cluster_stability".into(), value);
        }
        if let Some(value) = persistent_functional_cluster_stability {
            metrics.insert("persistent_functional_cluster_stability".into(), value);
        }
        if let Some(value) = persistent_cluster_identity_min_margin {
            metrics.insert("persistent_cluster_identity_min_margin".into(), value);
        }
        if let Some(value) = persistent_cluster_assignment_consistent {
            metrics.insert(
                "persistent_cluster_assignment_consistent".into(),
                if value { 1.0 } else { 0.0 },
            );
        }
        if let Some(value) = persistent_min_holdout_similarity {
            metrics.insert("persistent_min_holdout_similarity".into(), value);
        }
        metrics.insert(
            "aperture_effective_group_rank".into(),
            aperture_independence.effective_group_rank,
        );
        metrics.insert(
            "aperture_effective_independent_groups".into(),
            aperture_independence.effective_independent_groups,
        );
        metrics.insert(
            "aperture_numerical_group_rank".into(),
            aperture_independence.numerical_design_rank as f64,
        );
        metrics.insert(
            "identifiability_resolved_rank".into(),
            resolution_map.resolved_rank as f64,
        );
        metrics.insert(
            "identifiability_min_principal_angle_degrees".into(),
            resolution_map.min_principal_angle_degrees,
        );
        metrics.insert(
            "normalized_reconstruction_rms".into(),
            normalized_reconstruction_rms,
        );
        metrics.insert("effective_rank".into(), effective_rank);
        metrics.insert("condition_estimate".into(), condition_estimate);
        metrics.insert(
            "structured_active_block_count".into(),
            structured_block_count as f64,
        );
        metrics.insert(
            "structured_max_local_rank".into(),
            structured_max_local_rank as f64,
        );
        metrics.insert(
            "structured_mean_effective_rank".into(),
            structured_mean_effective_rank,
        );
        let field_coefficients = (0..selected_coefficients.rows)
            .map(|row| selected_coefficients.row_vec(row))
            .collect::<Vec<_>>();
        let parameter_promotable = reasons.is_empty();
        let mut representation_protocol_sha256 = None;
        let mut representation_cv_r2 = None;
        let mut representation_match_accuracy = None;
        let mut representation_mean_matched_cosine = None;
        let mut representation_min_match_margin = None;
        let mut dual_space_verified = None;
        if let Some((protocol_sha, representations)) = self.representation_observations(obs)? {
            let dual_model = DualSpaceModel {
                fields: &fields,
                field_coefficients: &field_coefficients,
                skill_source_mixtures: &skill_source_mixtures,
                parameter_inverse_mode: inverse_mode,
                parameter_promotable,
                functional_cv_r2,
            };
            let dual = analyze_dual_space(
                &dual_model,
                obs,
                &representations,
                self.config.ridge,
                self.config.min_independent_apertures,
                self.config.min_representation_match_accuracy,
                self.config.min_representation_match_margin,
            )?;
            if dual.fields.len() != fields.len() {
                return Err(BrainError::Integrity(
                    "dual_space_field_count_mismatch".into(),
                ));
            }
            for (field, dual_field) in fields.iter_mut().zip(&dual.fields) {
                if field.skill_id != dual_field.skill_id
                    || field.functional_signature != dual_field.functional_signature
                    || dual_field.representation_signature.is_empty()
                    || dual_field
                        .representation_signature
                        .iter()
                        .any(|value| !value.is_finite())
                {
                    return Err(BrainError::Integrity(
                        "dual_space_field_identity_mismatch".into(),
                    ));
                }
                field.representation_signature = dual_field.representation_signature.clone();
            }
            metrics.insert("representation_cv_r2".into(), dual.representation_cv_r2);
            metrics.insert(
                "representation_match_accuracy".into(),
                dual.representation_match_accuracy,
            );
            metrics.insert(
                "representation_mean_matched_cosine".into(),
                dual.representation_mean_matched_cosine,
            );
            metrics.insert(
                "representation_min_match_margin".into(),
                dual.representation_min_match_margin,
            );
            metrics.insert(
                "dual_space_verified".into(),
                if dual.dual_space_verified { 1.0 } else { 0.0 },
            );
            if self.config.require_dual_space_for_promotion && !dual.representation_supported {
                reasons.push("dual_space_representation_identity_unverified".into());
            }
            representation_protocol_sha256 = Some(protocol_sha);
            representation_cv_r2 = Some(dual.representation_cv_r2);
            representation_match_accuracy = Some(dual.representation_match_accuracy);
            representation_mean_matched_cosine = Some(dual.representation_mean_matched_cosine);
            representation_min_match_margin = Some(dual.representation_min_match_margin);
            dual_space_verified = Some(dual.dual_space_verified);
        }
        if self.config.require_dual_space_for_promotion
            && fields
                .iter()
                .any(|field| field.representation_signature.is_empty())
        {
            reasons.push("dual_space_representation_signature_missing".into());
        }
        let promotion = PromotionDecision {
            allowed: reasons.is_empty(),
            reasons,
            metrics,
        };

        Ok(ReconstructionReport {
            schema: "cerebro.tidex.reconstruction/v6".into(),
            source_tree_digest,
            config_digest,
            analysis_version_digest,
            observation_count: obs.len(),
            observation_set_digest,
            parameter_dimension: dim,
            independence_groups: group_counts.len(),
            aperture_independence,
            resolution_map,
            confounder_names: conf.design_names,
            confounder_explained_fraction: conf.explained_fraction,
            cycle_rms: sbas.cycle_rms,
            max_edge_residual: sbas.max_edge_residual,
            selected_rank,
            effective_rank,
            condition_estimate,
            reconstruction_rms,
            normalized_reconstruction_rms,
            functional_cv_r2,
            inverse_mode,
            spectral_functional_cv_r2: spectral_functional.cv_r2,
            persistent_functional_cv_r2,
            persistent_coherence_threshold,
            persistent_coherence_gap,
            persistent_coverage_ratio,
            persistent_cluster_stability,
            persistent_parametric_cluster_stability,
            persistent_functional_cluster_stability,
            persistent_cluster_identity_min_margin,
            persistent_cluster_assignment_consistent,
            persistent_min_holdout_similarity,
            persistent_cluster_sizes,
            persistent_error,
            representation_protocol_sha256,
            representation_cv_r2,
            representation_match_accuracy,
            representation_mean_matched_cosine,
            representation_min_match_margin,
            dual_space_verified,
            fields,
            field_coefficients,
            skill_source_mixtures,
            promotion,
        })
    }

    fn bank_path(&self) -> PathBuf {
        self.root.join("state/skill_bank.json")
    }
    /// Load the installed active bank.  Absence is never silently translated
    /// into an empty bank: callers that intentionally establish the very
    /// first bank must use the explicit bootstrap path below.
    pub fn load_bank(&self) -> BrainResult<SkillBank> {
        let p = self.bank_path();
        match fs::symlink_metadata(&p) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(BrainError::Integrity("active_skill_bank_missing".into()))
            }
            Err(error) => return Err(error.into()),
            Ok(_) => {}
        }
        let p = existing_regular_file_under_root(&self.root, &p)
            .map_err(|_| BrainError::Integrity("active_skill_bank_path_invalid".into()))?;
        let raw = fs::read(&p)?;
        Ok(serde_json::from_slice(&raw)?)
    }

    /// A missing active bank is meaningful only while establishing the first
    /// sleep transaction.  This preserves that explicit bootstrap state
    /// without exposing a generic empty-bank fallback to runtime authority.
    fn load_bank_for_initial_sleep_bootstrap(&self) -> BrainResult<Option<SkillBank>> {
        let p = self.bank_path();
        match fs::symlink_metadata(&p) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
            Ok(_) => self.load_bank().map(Some),
        }
    }

    fn persist_observations(&self, obs: &[DeltaObservation]) -> BrainResult<Vec<String>> {
        let dir = self.root.join("state/observations");
        ensure_private_directory(&self.root, &dir)?;
        let mut digests = Vec::new();
        for o in obs {
            if !valid_observation_id(&o.observation_id) {
                return Err(BrainError::Invalid("observation_id_invalid".into()));
            }
            let d = digest_json(o)?;
            let p = dir.join(format!("{}-{}.json", o.observation_id, &d[..16]));
            write_new_private(&self.root, &p, &serialize_pretty_line(o)?)?;
            digests.push(d);
        }
        digests.sort();
        digests.dedup();
        Ok(digests)
    }

    /// A corpus transition deliberately fails closed after its intent is
    /// written. Until the finalizer has sealed the immutable receipt and moved
    /// that intent into the archive, no ordinary commit/sleep/runtime action
    /// may operate on a potentially half-replaced active corpus.
    fn require_no_incomplete_corpus_transition(&self) -> BrainResult<()> {
        let inflight = self.root.join("state/corpus_transitions/inflight");
        match fs::symlink_metadata(&inflight) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
            Ok(_) => {}
        }
        let inflight = existing_directory_under_root(&self.root, &inflight).map_err(|_| {
            BrainError::Integrity("learning_corpus_transition_inflight_directory_invalid".into())
        })?;
        let mut entries = fs::read_dir(&inflight)?;
        match entries.next() {
            None => Ok(()),
            Some(Ok(entry)) => {
                let path = entry.path();
                if existing_directory_under_root(&self.root, &path).is_err() {
                    return Err(BrainError::Integrity(
                        "learning_corpus_transition_inflight_entry_invalid".into(),
                    ));
                }
                Err(BrainError::Integrity(
                    "learning_corpus_transition_incomplete".into(),
                ))
            }
            Some(Err(error)) => Err(error.into()),
        }
    }

    fn commit_after_verified_corpus_transition(
        &self,
        observations: &[DeltaObservation],
    ) -> BrainResult<ReconstructionReport> {
        self.require_canonical_runtime_config()?;
        let obs = canonical_observations(observations);
        let mut report = self.analyze_canonical(&obs)?;
        if report.promotion.allowed {
            let mixtures = report.skill_source_mixtures.clone();
            self.materialize_dense_fields(&mut report.fields, &mixtures, &obs)?;
        }
        let observation_digests = self.persist_observations(&obs)?;
        let batch_digest = digest_json(&observation_digests)?;
        if batch_digest != report.observation_set_digest {
            return Err(BrainError::Integrity(
                "commit_observation_set_digest_mismatch".into(),
            ));
        }
        let report_bytes = serialize_pretty_line(&report)?;
        let report_sha256 = sha256_bytes(&report_bytes);
        let operation_key = commit_operation_key(&batch_digest, &report, &report_sha256);
        if !report.promotion.allowed {
            return Err(BrainError::Integrity(
                "corpus_transition_commit_requires_promotable_report".into(),
            ));
        }
        let generation = obs
            .iter()
            .map(|observation| observation.generation)
            .max()
            .unwrap_or(0);
        let memory = build_memory_snapshot(
            &obs,
            generation,
            true,
            &report.fields,
            &report.skill_source_mixtures,
        )?;
        let memory_bytes = serialize_pretty_line(&memory)?;
        let memory_sha256 = sha256_bytes(&memory_bytes);
        let shadow_bank_path = self.root.join("state/shadow_skill_bank.json");
        match fs::symlink_metadata(&shadow_bank_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Err(BrainError::Integrity(
                    "corpus_transition_prior_shadow_bank_not_archived".into(),
                ))
            }
            Err(error) => return Err(error.into()),
        }
        let mut shadow_bank = SkillBank::default();
        assimilate_bank(
            &mut shadow_bank,
            &report.fields,
            self.config.skill_match_cosine,
        )?;
        let staged_bank_bytes = serialize_pretty_line(&shadow_bank)?;
        let shadow_bank_sha256 = sha256_bytes(&staged_bank_bytes);
        let expected_intent = CommitTransactionIntent {
            schema: "cerebro.tidex.commit_transaction_intent/v2".into(),
            operation_key: operation_key.clone(),
            batch_digest: batch_digest.clone(),
            observation_digests: observation_digests.clone(),
            report_sha256: report_sha256.clone(),
            report_promotable: true,
            generation,
            memory_sha256: memory_sha256.clone(),
            shadow_bank_sha256: Some(shadow_bank_sha256.clone()),
            prior_shadow_bank_sha256: None,
        };

        let commits_dir = self.root.join("state/commits");
        let transactions_dir = self.root.join("state/transactions");
        ensure_private_directory(&self.root, &commits_dir)?;
        ensure_private_directory(&self.root, &transactions_dir)?;
        let receipt_path = commits_dir.join(format!("{operation_key}.json"));

        // Completed batches are immutable and idempotent. Re-running the same
        // observation batch verifies every authority artifact and returns the
        // fresh analysis without touching bank, memory, or ledger.
        if fs::symlink_metadata(&receipt_path).is_ok() {
            let receipt_path = existing_regular_file_under_root(&self.root, &receipt_path)?;
            let receipt: CommitReceipt = serde_json::from_slice(&fs::read(&receipt_path)?)?;
            if receipt.schema != "cerebro.tidex.commit_receipt/v2"
                || receipt.legacy_recovery
                || receipt.operation_key != expected_intent.operation_key
                || receipt.batch_digest != expected_intent.batch_digest
                || receipt.report_sha256 != expected_intent.report_sha256
                || receipt.memory_sha256 != expected_intent.memory_sha256
                || receipt.shadow_bank_sha256 != expected_intent.shadow_bank_sha256
            {
                return Err(BrainError::Integrity(
                    "commit_receipt_report_mismatch".into(),
                ));
            }
            let event = ledger::find_v2_event_by_payload_string(
                &self.root,
                "commit_transaction",
                "operation_key",
                &operation_key,
            )?
            .ok_or_else(|| BrainError::Integrity("commit_receipt_ledger_event_missing".into()))?;
            if event.event_hash != receipt.ledger_event_hash {
                return Err(BrainError::Integrity(
                    "commit_receipt_ledger_event_mismatch".into(),
                ));
            }
            verify_commit_transaction_ledger_binding(&event, &expected_intent, obs.len(), &report)?;
            let report_path = self
                .root
                .join("state/reports")
                .join(format!("{}.json", report_sha256));
            let report_path = existing_regular_file_under_root(&self.root, &report_path)?;
            if fs::read(&report_path)? != report_bytes {
                return Err(BrainError::Integrity(
                    "commit_receipt_report_content_mismatch".into(),
                ));
            }
            let memory_path = memory_artifact_path(&self.root, &receipt.memory_sha256);
            let memory_path = existing_regular_file_under_root(&self.root, &memory_path)?;
            if file_sha256(&memory_path)? != receipt.memory_sha256 {
                return Err(BrainError::Integrity(
                    "commit_receipt_memory_mismatch".into(),
                ));
            }
            if fs::read(&memory_path)? != memory_bytes {
                return Err(BrainError::Integrity(
                    "commit_receipt_memory_content_mismatch".into(),
                ));
            }
            let bank_path = self
                .root
                .join("state/skill_banks/by-sha")
                .join(format!("{}.json", shadow_bank_sha256));
            let bank_path = existing_regular_file_under_root(&self.root, &bank_path)?;
            if file_sha256(&bank_path)? != shadow_bank_sha256
                || fs::read(&bank_path)? != staged_bank_bytes
            {
                return Err(BrainError::Integrity(
                    "commit_receipt_shadow_bank_mismatch".into(),
                ));
            }
            let current_memory = existing_regular_file_under_root(
                &self.root,
                &self.root.join("state/memory/current.json"),
            )?;
            let current_shadow = existing_regular_file_under_root(
                &self.root,
                &self.root.join("state/shadow_skill_bank.json"),
            )?;
            if fs::read(current_memory)? != memory_bytes
                || fs::read(current_shadow)? != staged_bank_bytes
            {
                return Err(BrainError::Integrity(
                    "commit_receipt_current_pointer_mismatch".into(),
                ));
            }
            return Ok(report);
        }

        let transaction_dir = transactions_dir.join(&operation_key);
        let intent_path = transaction_dir.join("intent.json");
        let staged_report_path = transaction_dir.join("report.json");
        let staged_memory_path = transaction_dir.join("memory.json");
        let staged_bank_path = transaction_dir.join("shadow_skill_bank.json");

        let transaction_exists = match fs::symlink_metadata(&transaction_dir) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if transaction_exists {
            let transaction_dir = existing_directory_under_root(&self.root, &transaction_dir)?;
            let intent_path = existing_regular_file_under_root(&self.root, &intent_path)?;
            let staged_report_path =
                existing_regular_file_under_root(&self.root, &staged_report_path)?;
            let staged_memory_path =
                existing_regular_file_under_root(&self.root, &staged_memory_path)?;
            let staged_bank_path = existing_regular_file_under_root(&self.root, &staged_bank_path)?;
            let stored: CommitTransactionIntent = serde_json::from_slice(&fs::read(intent_path)?)?;
            if stored != expected_intent
                || fs::read(staged_report_path)? != report_bytes
                || fs::read(staged_memory_path)? != memory_bytes
                || fs::read(staged_bank_path)? != staged_bank_bytes
            {
                return Err(BrainError::Integrity(
                    "commit_transaction_intent_mismatch".into(),
                ));
            }
            let entries = fs::read_dir(transaction_dir)?;
            let mut names = BTreeSet::new();
            for entry in entries {
                let entry = entry?;
                let name = entry.file_name().into_string().map_err(|_| {
                    BrainError::Integrity("commit_transaction_stage_filename_invalid".into())
                })?;
                names.insert(name);
            }
            if names
                != BTreeSet::from([
                    "intent.json".to_string(),
                    "report.json".to_string(),
                    "memory.json".to_string(),
                    "shadow_skill_bank.json".to_string(),
                ])
            {
                return Err(BrainError::Integrity(
                    "commit_transaction_stage_directory_invalid".into(),
                ));
            }
        } else {
            ensure_private_directory(&self.root, &transaction_dir)?;
            write_new_private(&self.root, &staged_report_path, &report_bytes)?;
            write_new_private(&self.root, &staged_memory_path, &memory_bytes)?;
            write_new_private(&self.root, &staged_bank_path, &staged_bank_bytes)?;
            write_new_private(
                &self.root,
                &intent_path,
                &serialize_pretty_line(&expected_intent)?,
            )?;
        }
        let intent = expected_intent;

        // There is exactly one authoritative ledger event per batch. If a
        // crash happened after append but before canonical installation, retry
        // finds the event and resumes rather than appending or assimilating again.
        let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
            &self.root,
            "commit_transaction",
            "operation_key",
            &operation_key,
        )? {
            verify_commit_transaction_ledger_binding(&existing, &intent, obs.len(), &report)?;
            existing
        } else {
            let event = ledger::append(
                &self.root,
                "commit_transaction",
                commit_transaction_payload(&intent, obs.len(), &report),
            )?;
            verify_commit_transaction_ledger_binding(&event, &intent, obs.len(), &report)?;
            event
        };

        let canonical_report = self
            .root
            .join("state/reports")
            .join(format!("{}.json", intent.report_sha256));
        write_immutable_exact(
            &self.root,
            &canonical_report,
            &report_bytes,
            &intent.report_sha256,
        )?;
        let historical_memory = memory_artifact_path(&self.root, &intent.memory_sha256);
        write_immutable_exact(
            &self.root,
            &historical_memory,
            &memory_bytes,
            &intent.memory_sha256,
        )?;
        let memory_path = self.root.join("state/memory/current.json");
        replace_private_pointer_exact(
            &self.root,
            &memory_path,
            &memory_bytes,
            &intent.memory_sha256,
        )?;
        let expected_bank = intent
            .shadow_bank_sha256
            .as_ref()
            .ok_or_else(|| BrainError::Integrity("corpus_transition_shadow_bank_missing".into()))?;
        let historical_bank = self
            .root
            .join("state/skill_banks/by-sha")
            .join(format!("{expected_bank}.json"));
        write_immutable_exact(
            &self.root,
            &historical_bank,
            &staged_bank_bytes,
            expected_bank,
        )?;
        replace_private_pointer_exact(
            &self.root,
            &self.root.join("state/shadow_skill_bank.json"),
            &staged_bank_bytes,
            expected_bank,
        )?;

        let receipt = CommitReceipt {
            schema: "cerebro.tidex.commit_receipt/v2".into(),
            operation_key: operation_key.clone(),
            batch_digest: batch_digest.clone(),
            report_sha256: intent.report_sha256.clone(),
            memory_sha256: intent.memory_sha256.clone(),
            shadow_bank_sha256: intent.shadow_bank_sha256.clone(),
            ledger_event_hash: event.event_hash,
            legacy_recovery: false,
        };
        write_new_private(&self.root, &receipt_path, &serialize_pretty_line(&receipt)?)?;
        let receipt_path = existing_regular_file_under_root(&self.root, &receipt_path)?;
        let persisted_receipt: CommitReceipt = serde_json::from_slice(&fs::read(&receipt_path)?)?;
        if persisted_receipt != receipt {
            return Err(BrainError::Integrity(
                "commit_receipt_post_write_mismatch".into(),
            ));
        }
        let verified_event = ledger::find_v2_event_by_payload_string(
            &self.root,
            "commit_transaction",
            "operation_key",
            &intent.operation_key,
        )?
        .ok_or_else(|| BrainError::Integrity("commit_receipt_ledger_event_missing".into()))?;
        if verified_event.event_hash != receipt.ledger_event_hash {
            return Err(BrainError::Integrity(
                "commit_receipt_post_write_ledger_changed".into(),
            ));
        }
        verify_commit_transaction_ledger_binding(&verified_event, &intent, obs.len(), &report)?;
        let current_memory = existing_regular_file_under_root(
            &self.root,
            &self.root.join("state/memory/current.json"),
        )?;
        let current_shadow = existing_regular_file_under_root(
            &self.root,
            &self.root.join("state/shadow_skill_bank.json"),
        )?;
        if fs::read(current_memory)? != memory_bytes
            || fs::read(current_shadow)? != staged_bank_bytes
        {
            return Err(BrainError::Integrity(
                "commit_receipt_post_write_pointer_mismatch".into(),
            ));
        }
        Ok(report)
    }

    /// Finalize a receipt-backed adaptive-learning session into a new canonical
    /// TIDE-X corpus.  This is intentionally narrower than `commit`: callers
    /// cannot supply observations, cannot merge an unrelated corpus, and
    /// cannot carry a prior shadow/active bank into the new parameter/function
    /// domain.  Any interrupted transition remains visibly incomplete and the
    /// runtime fails closed rather than attempting a heuristic recovery.
    pub fn commit_finalized_learning_session(
        &self,
        supplied: &LearningFinalizationInput,
    ) -> BrainResult<LearningFinalizationReceipt> {
        self.require_canonical_runtime_config()?;
        let input = verify_learning_finalization_input(&self.root, supplied)?;
        let learning_finalization_input_sha256 = learning_finalization_input_sha256(&input)?;
        let representation_observation_bindings_sha256 = Sha256Digest::digest_bytes(
            &serde_json::to_vec(&input.representation_observation_bindings)?,
        );
        let observations = canonical_observations(&input.observations);
        if observations.is_empty() {
            return Err(BrainError::Integrity(
                "learning_finalization_observation_set_empty".into(),
            ));
        }

        // Validate the complete prospective corpus before it can replace any
        // active state. A failed learning reconstruction is diagnostic evidence,
        // not authority to evict the existing corpus.
        let mut prospective = self.analyze_canonical(&observations)?;
        if !prospective.promotion.allowed {
            return Err(BrainError::Integrity(format!(
                "learning_finalization_reconstruction_not_promotable:{}",
                prospective.promotion.reasons.join("|")
            )));
        }
        let new_corpus_digest = Sha256Digest::parse(&prospective.observation_set_digest)?;
        let finalizations_dir = self.root.join("state/learning_finalizations");

        // Idempotence has to be resolved before treating the currently active
        // corpus as a *prior* corpus: after a completed transition the active
        // observations are necessarily the new corpus. Search immutable
        // receipts by the re-derived input binding and reject ambiguity.
        match fs::symlink_metadata(&finalizations_dir) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
            Ok(_) => {
                let finalizations_dir =
                    existing_directory_under_root(&self.root, &finalizations_dir)?;
                let mut matching = Vec::new();
                for entry in fs::read_dir(&finalizations_dir)? {
                    let path = entry?.path();
                    if path.extension().and_then(|value| value.to_str()) != Some("json") {
                        return Err(BrainError::Integrity(
                            "learning_finalization_receipt_directory_entry_invalid".into(),
                        ));
                    }
                    let path = existing_regular_file_under_root(&self.root, &path)?;
                    let bytes = fs::read(&path)?;
                    let receipt: LearningFinalizationReceipt = serde_json::from_slice(&bytes)?;
                    if receipt.learning_finalization_input_sha256
                        == learning_finalization_input_sha256
                    {
                        matching.push((path, Sha256Digest::digest_bytes(&bytes), receipt));
                    }
                }
                if matching.len() > 1 {
                    return Err(BrainError::Integrity(
                        "learning_finalization_receipt_input_ambiguous".into(),
                    ));
                }
                if let Some((path, digest, receipt)) = matching.pop() {
                    if receipt.schema != "cerebro.tidex.learning_finalization_receipt/v1"
                        || path.file_name().and_then(|value| value.to_str())
                            != Some(format!("{}.json", receipt.operation_key).as_str())
                    {
                        return Err(BrainError::Integrity(
                            "learning_finalization_receipt_contract_invalid".into(),
                        ));
                    }
                    let reference = PrivateFileReference::new(path, digest);
                    close_verified_learning_finalization_inflight(
                        &self.root, &receipt, &reference,
                    )?;
                    let verified =
                        load_verified_learning_finalization_receipt(&self.root, &reference)?;
                    if verified.learning_finalization_input_sha256
                        != learning_finalization_input_sha256
                    {
                        return Err(BrainError::Integrity(
                            "learning_finalization_receipt_input_replay_mismatch".into(),
                        ));
                    }
                    return Ok(verified);
                }
            }
        }

        self.require_no_incomplete_corpus_transition()?;

        let prior_observations = canonical_observations(&self.load_persisted_observations()?);
        if prior_observations.is_empty() {
            return Err(BrainError::Integrity(
                "learning_finalization_prior_corpus_missing".into(),
            ));
        }
        let prior_corpus_digest =
            Sha256Digest::parse(observation_set_digest(&prior_observations)?)?;
        if prior_corpus_digest == new_corpus_digest {
            return Err(BrainError::Integrity(
                "learning_finalization_corpus_transition_not_distinct".into(),
            ));
        }

        // A corpus may be replaced only after the *authenticated* prior sleep
        // chain proves that it is revoked. A missing, malformed, or manually
        // edited state file is never permission to evict a previously live
        // certified bank.
        let prior_health = self.runtime_integrity_health()?;
        if !prior_health.integrity_healthy
            || !prior_health.current_corpus_bound
            || prior_health.certification_status != Some(CertificationStatus::Revoked)
        {
            return Err(BrainError::Integrity(
                "learning_finalization_requires_authenticated_revoked_prior_corpus".into(),
            ));
        }
        let prior_runtime_artifacts = self.snapshot_revoked_prior_runtime_artifacts()?;

        // Materialization is a pre-transition, content-addressed preparation
        // step.  It intentionally happens only after idempotent receipt replay
        // and prior-corpus authority checks; the loader below rederives it
        // without writing when a completed receipt already exists.
        let prospective_mixtures = prospective.skill_source_mixtures.clone();
        self.materialize_dense_fields(
            &mut prospective.fields,
            &prospective_mixtures,
            &observations,
        )?;
        let prospective_report_bytes = serialize_pretty_line(&prospective)?;
        let prospective_report_sha256 = Sha256Digest::digest_bytes(&prospective_report_bytes);

        let operation_key = learning_finalization_operation_key(
            &learning_finalization_input_sha256,
            &prior_corpus_digest,
            &new_corpus_digest,
        )?;
        let receipt_path = finalizations_dir.join(format!("{operation_key}.json"));
        let transitions_root = self.root.join("state/corpus_transitions");
        let transaction_dir = transitions_root
            .join("inflight")
            .join(operation_key.as_str());
        let archive_dir = transitions_root
            .join("by-operation")
            .join(operation_key.as_str());

        if fs::symlink_metadata(&receipt_path).is_ok() {
            return Err(BrainError::Integrity(
                "learning_finalization_operation_key_collision".into(),
            ));
        }
        if fs::symlink_metadata(&transaction_dir).is_ok()
            || fs::symlink_metadata(&archive_dir).is_ok()
        {
            return Err(BrainError::Integrity(
                "learning_finalization_incomplete_transition_requires_explicit_recovery".into(),
            ));
        }

        ensure_private_directory(&self.root, &finalizations_dir)?;
        ensure_private_directory(&self.root, &transitions_root.join("inflight"))?;
        ensure_private_directory(&self.root, &transitions_root.join("by-operation"))?;
        ensure_private_directory(&self.root, &transaction_dir)?;
        let intent = LearningCorpusTransitionIntent {
            schema: "cerebro.tidex.learning_corpus_transition_intent/v1".into(),
            operation_key: operation_key.clone(),
            session_id: input.session_id.clone(),
            adaptive_receipt_sha256: input.adaptive_receipt_sha256.clone(),
            learning_finalization_input_sha256: learning_finalization_input_sha256.clone(),
            representation_evidence_receipt: input.representation_evidence_receipt.clone(),
            representation_protocol_sha256: input.representation_protocol_sha256.clone(),
            representation_observation_bindings_sha256: representation_observation_bindings_sha256
                .clone(),
            prior_corpus_digest: prior_corpus_digest.clone(),
            prior_observation_count: prior_observations.len(),
            new_corpus_digest: new_corpus_digest.clone(),
            new_observation_count: observations.len(),
            archive_dir: archive_dir.clone(),
        };
        write_new_private(
            &self.root,
            &transaction_dir.join("intent.json"),
            &serialize_pretty_line(&intent)?,
        )?;

        ensure_private_directory(&self.root, &archive_dir)?;
        let observations_dir = self.root.join("state/observations");
        let observations_dir = existing_directory_under_root(&self.root, &observations_dir)
            .map_err(|_| {
                BrainError::Integrity(
                    "learning_finalization_active_observation_directory_invalid".into(),
                )
            })?;
        let archived_observations_dir = archive_dir.join("observations");
        if fs::symlink_metadata(&archived_observations_dir).is_ok() {
            return Err(BrainError::Integrity(
                "learning_finalization_archive_observations_collision".into(),
            ));
        }
        fs::rename(&observations_dir, &archived_observations_dir)?;
        secure_dir(&archived_observations_dir)?;
        let mut archived_artifact_sha256 = BTreeMap::<String, Sha256Digest>::new();
        let prior_observation_manifest = serialize_pretty_line(&prior_observations)?;
        let prior_observation_manifest_sha =
            Sha256Digest::digest_bytes(&prior_observation_manifest);
        write_new_private(
            &self.root,
            &archive_dir.join("observations_manifest.json"),
            &prior_observation_manifest,
        )?;
        archived_artifact_sha256.insert(
            "observations_manifest.json".into(),
            prior_observation_manifest_sha,
        );
        for (source, label) in [
            (self.bank_path(), "skill_bank.json"),
            (
                self.root.join("state/shadow_skill_bank.json"),
                "shadow_skill_bank.json",
            ),
            (
                self.root.join("state/memory/current.json"),
                "memory_current.json",
            ),
            (self.root.join("state/sleep_state.json"), "sleep_state.json"),
            (
                self.root.join("state/sleep_evidence/current.json"),
                "sleep_evidence_current.json",
            ),
        ] {
            archive_private_state_file(
                &self.root,
                &source,
                &archive_dir.join(label),
                label,
                prior_runtime_artifacts.get(label),
                &mut archived_artifact_sha256,
            )?;
        }

        // The new corpus is installed only after all old live pointers have
        // been archived. `commit` will independently materialize its shadow
        // state and write a transaction/ledger receipt for this exact corpus.
        let persisted = self.persist_observations(&observations)?;
        if Sha256Digest::parse(digest_json(&persisted)?)? != new_corpus_digest {
            return Err(BrainError::Integrity(
                "learning_finalization_new_observation_digest_mismatch".into(),
            ));
        }
        let report = self.commit_after_verified_corpus_transition(&observations)?;
        let report_bytes = serialize_pretty_line(&report)?;
        let report_sha256 = Sha256Digest::digest_bytes(&report_bytes);
        if report_sha256 != prospective_report_sha256
            || report.observation_set_digest != new_corpus_digest.as_str()
            || !report.promotion.allowed
        {
            return Err(BrainError::Integrity(
                "learning_finalization_committed_report_mismatch".into(),
            ));
        }
        let commit_operation_key = Sha256Digest::parse(commit_operation_key(
            new_corpus_digest.as_str(),
            &report,
            report_sha256.as_str(),
        ))?;
        let commit_path = self
            .root
            .join("state/commits")
            .join(format!("{commit_operation_key}.json"));
        let commit_path = existing_regular_file_under_root(&self.root, &commit_path)?;
        let commit_receipt_sha256 = crate::artifact::sha256_file(&commit_path)?;

        let event = ledger::append(
            &self.root,
            "learning_corpus_transition",
            json!({
                "schema":"cerebro.tidex.learning_corpus_transition/v1",
                "operation_key":operation_key,
                "session_id":input.session_id,
                "adaptive_receipt_sha256":input.adaptive_receipt_sha256,
                "learning_finalization_input_sha256":learning_finalization_input_sha256,
                "representation_evidence_receipt_path":input.representation_evidence_receipt.path,
                "representation_evidence_receipt_sha256":input.representation_evidence_receipt.sha256,
                "representation_protocol_sha256":input.representation_protocol_sha256,
                "representation_observation_bindings_sha256":representation_observation_bindings_sha256,
                "prior_corpus_digest":prior_corpus_digest,
                "prior_observation_count":prior_observations.len(),
                "new_corpus_digest":new_corpus_digest,
                "new_observation_count":observations.len(),
                "report_sha256":report_sha256,
                "commit_operation_key":commit_operation_key,
                "commit_receipt_sha256":commit_receipt_sha256,
                "archived_artifact_sha256":archived_artifact_sha256,
            }),
        )?;
        let receipt = LearningFinalizationReceipt {
            schema: "cerebro.tidex.learning_finalization_receipt/v1".into(),
            operation_key,
            session_id: input.session_id,
            adaptive_receipt_sha256: input.adaptive_receipt_sha256,
            learning_finalization_input_sha256,
            representation_evidence_receipt: input.representation_evidence_receipt,
            representation_protocol_sha256: input.representation_protocol_sha256,
            representation_observation_bindings_sha256,
            representation_observation_bindings: input.representation_observation_bindings,
            prior_corpus_digest,
            prior_observation_count: prior_observations.len(),
            new_corpus_digest,
            new_observation_count: observations.len(),
            archived_artifact_sha256,
            report_sha256,
            commit_operation_key,
            commit_receipt_sha256,
            ledger_event_hash: Sha256Digest::parse(event.event_hash)?,
        };
        let receipt_bytes = serialize_pretty_line(&receipt)?;
        let receipt_sha256 = Sha256Digest::digest_bytes(&receipt_bytes);
        write_new_private(&self.root, &receipt_path, &receipt_bytes)?;
        let closed_intent = archive_dir.join("transition_intent");
        match fs::symlink_metadata(&closed_intent) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Err(BrainError::Integrity(
                    "learning_finalization_closed_intent_collision".into(),
                ))
            }
            Err(error) => return Err(error.into()),
        }
        fs::rename(&transaction_dir, &closed_intent)?;
        secure_dir(&closed_intent)?;
        load_verified_learning_finalization_receipt(
            &self.root,
            &PrivateFileReference::new(receipt_path, receipt_sha256),
        )
    }

    fn load_persisted_observations(&self) -> BrainResult<Vec<DeltaObservation>> {
        load_observations_from_private_directory(
            &self.root,
            &self.root.join("state/observations"),
            true,
        )
    }

    /// Read the mutable sleep-evidence pointer only when it has the exact
    /// identity selected for the current transaction. An absent bundle is a
    /// valid revocation input; a newly appearing or replaced bundle is not.
    fn verify_current_sleep_evidence_pointer(
        &self,
        expected_sha256: Option<&str>,
    ) -> BrainResult<Option<PathBuf>> {
        let path = self.root.join("state/sleep_evidence/current.json");
        match (expected_sha256, fs::symlink_metadata(&path)) {
            (Some(expected), Ok(_)) => {
                let verified = existing_regular_file_under_root(&self.root, &path)?;
                if file_sha256(&verified)? != expected {
                    return Err(BrainError::Integrity(
                        "sleep_evidence_changed_during_transaction".into(),
                    ));
                }
                Ok(Some(verified))
            }
            (Some(_), Err(error)) if error.kind() == std::io::ErrorKind::NotFound => Err(
                BrainError::Integrity("sleep_evidence_current_pointer_missing".into()),
            ),
            (Some(_), Err(error)) => Err(error.into()),
            (None, Err(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            (None, Ok(_)) => Err(BrainError::Integrity(
                "sleep_installation_unexpected_current_evidence".into(),
            )),
            (None, Err(error)) => Err(error.into()),
        }
    }

    /// Recheck every live pointer and immutable history a sleep receipt claims
    /// immediately before reporting success. This keeps a concurrent pointer
    /// replacement from becoming an observable successful sleep transaction.
    fn verify_current_sleep_installation(
        &self,
        receipt: &SleepReceipt,
        state: &Value,
    ) -> BrainResult<()> {
        let sleep_path = self.root.join("state/sleep_state.json");
        let current_sleep_state = existing_regular_file_under_root(&self.root, &sleep_path)?;
        verify_sleep_receipt_ledger_binding(
            &self.root,
            receipt,
            state,
            &file_sha256(&current_sleep_state)?,
        )?;
        let report_history = self
            .root
            .join("state/reports")
            .join(format!("{}.json", receipt.report_sha256));
        let memory_history = memory_artifact_path(&self.root, &receipt.memory_sha256);
        let state_history = self
            .root
            .join("state/sleep/by-sha")
            .join(format!("{}.json", receipt.sleep_state_sha256));
        for (path, digest, label) in [
            (&report_history, receipt.report_sha256.as_str(), "report"),
            (&memory_history, receipt.memory_sha256.as_str(), "memory"),
            (&state_history, receipt.sleep_state_sha256.as_str(), "state"),
        ] {
            let path = existing_regular_file_under_root(&self.root, path)?;
            if file_sha256(&path)? != digest {
                return Err(BrainError::Integrity(format!(
                    "sleep_installation_{label}_history_mismatch"
                )));
            }
        }
        let current_memory = existing_regular_file_under_root(
            &self.root,
            &self.root.join("state/memory/current.json"),
        )?;
        if file_sha256(&current_memory)? != receipt.memory_sha256 {
            return Err(BrainError::Integrity(
                "sleep_installation_current_memory_mismatch".into(),
            ));
        }
        match &receipt.active_bank_sha256 {
            Some(bank_sha) => {
                let bank_history = self
                    .root
                    .join("state/skill_banks/by-sha")
                    .join(format!("{bank_sha}.json"));
                let bank_history = existing_regular_file_under_root(&self.root, &bank_history)?;
                let current_bank = existing_regular_file_under_root(&self.root, &self.bank_path())?;
                if file_sha256(&bank_history)? != *bank_sha
                    || file_sha256(&current_bank)? != *bank_sha
                {
                    return Err(BrainError::Integrity(
                        "sleep_installation_current_bank_mismatch".into(),
                    ));
                }
            }
            None => match fs::symlink_metadata(self.bank_path()) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Ok(_) => {
                    return Err(BrainError::Integrity(
                        "sleep_installation_unexpected_active_bank".into(),
                    ))
                }
                Err(error) => return Err(error.into()),
            },
        }
        match &receipt.evidence_bundle_sha256 {
            Some(evidence_sha) => {
                let evidence_history = self
                    .root
                    .join("state/sleep_evidence/by-sha")
                    .join(format!("{evidence_sha}.json"));
                let evidence_history =
                    existing_regular_file_under_root(&self.root, &evidence_history)?;
                let current_evidence = self
                    .verify_current_sleep_evidence_pointer(Some(evidence_sha))?
                    .ok_or_else(|| {
                        BrainError::Integrity("sleep_evidence_current_pointer_missing".into())
                    })?;
                if file_sha256(&evidence_history)? != *evidence_sha
                    || file_sha256(&current_evidence)? != *evidence_sha
                {
                    return Err(BrainError::Integrity(
                        "sleep_installation_current_evidence_mismatch".into(),
                    ));
                }
            }
            None => {
                self.verify_current_sleep_evidence_pointer(None)?;
            }
        }
        Ok(())
    }

    /// Snapshot the exact live authority artifacts of the authenticated,
    /// revoked prior runtime. Each later archive rename compares its source
    /// bytes to this snapshot, so a pointer replacement after preflight turns
    /// the corpus transition into a visible fail-closed incomplete operation.
    fn snapshot_revoked_prior_runtime_artifacts(
        &self,
    ) -> BrainResult<BTreeMap<String, Sha256Digest>> {
        let sleep_path = self.root.join("state/sleep_state.json");
        let sleep_path = existing_regular_file_under_root(&self.root, &sleep_path)?;
        let state_bytes = fs::read(&sleep_path)?;
        let state: Value = serde_json::from_slice(&state_bytes)?;
        let operation_key =
            Sha256Digest::parse(required_sleep_state_string(&state, "operation_key")?)?;
        let receipt_path = self
            .root
            .join("state/sleep_receipts")
            .join(format!("{operation_key}.json"));
        let receipt_path = existing_regular_file_under_root(&self.root, &receipt_path)?;
        let receipt: SleepReceipt = serde_json::from_slice(&fs::read(receipt_path)?)?;
        if receipt.operation_key != operation_key.as_str() {
            return Err(BrainError::Integrity(
                "learning_finalization_prior_sleep_receipt_operation_invalid".into(),
            ));
        }
        self.verify_current_sleep_installation(&receipt, &state)?;

        let active_bank_sha = receipt.active_bank_sha256.as_deref().ok_or_else(|| {
            BrainError::Integrity("learning_finalization_prior_active_bank_missing".into())
        })?;
        let evidence_sha = receipt.evidence_bundle_sha256.as_deref().ok_or_else(|| {
            BrainError::Integrity("learning_finalization_prior_sleep_evidence_missing".into())
        })?;
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "skill_bank.json".into(),
            Sha256Digest::parse(active_bank_sha)?,
        );
        artifacts.insert(
            "memory_current.json".into(),
            Sha256Digest::parse(&receipt.memory_sha256)?,
        );
        artifacts.insert(
            "sleep_state.json".into(),
            Sha256Digest::parse(&receipt.sleep_state_sha256)?,
        );
        artifacts.insert(
            "sleep_evidence_current.json".into(),
            Sha256Digest::parse(evidence_sha)?,
        );
        Ok(artifacts)
    }

    /// The absence of an active bank is an initialization condition, never a
    /// recovery fallback. Once a sleep state exists, it must authenticate an
    /// already-revoked, explicitly bankless runtime before sleep is permitted
    /// to construct an empty bank again.
    fn authorize_empty_bank_bootstrap(
        &self,
        has_prior_sleep_state: bool,
        previous: &Value,
    ) -> BrainResult<()> {
        if !has_prior_sleep_state {
            return Ok(());
        }
        if required_sleep_state_string(previous, "schema")? != "cerebro.tidex.sleep_state/v5"
            || CertificationStatus::parse(required_sleep_state_string(
                previous,
                "certification_status",
            )?) != Some(CertificationStatus::Revoked)
            || previous.get("active_skill_count").and_then(Value::as_u64) != Some(0)
            || previous.get("promoted").and_then(Value::as_bool) != Some(false)
            || previous.get("active_bank_sha256") != Some(&Value::Null)
        {
            return Err(BrainError::Integrity(
                "empty_bank_bootstrap_prior_state_not_explicitly_revoked".into(),
            ));
        }
        let operation_key =
            Sha256Digest::parse(required_sleep_state_string(previous, "operation_key")?)?;
        let receipt_path = self
            .root
            .join("state/sleep_receipts")
            .join(format!("{operation_key}.json"));
        let receipt_path = existing_regular_file_under_root(&self.root, &receipt_path)?;
        let receipt: SleepReceipt = serde_json::from_slice(&fs::read(receipt_path)?)?;
        if receipt.operation_key != operation_key.as_str() || receipt.active_bank_sha256.is_some() {
            return Err(BrainError::Integrity(
                "empty_bank_bootstrap_prior_receipt_invalid".into(),
            ));
        }
        self.verify_current_sleep_installation(&receipt, previous)
    }

    pub fn sleep_cycle(&self) -> BrainResult<SleepReport> {
        self.require_canonical_runtime_config()?;
        self.require_no_incomplete_corpus_transition()?;
        let observations = canonical_observations(&self.load_persisted_observations()?);
        if observations.len() < self.config.min_observations {
            return Err(BrainError::Invalid(
                "sleep_requires_persisted_observations".into(),
            ));
        }
        let mut reconstruction = self.analyze_canonical(&observations)?;
        // Dense materialization is deterministic from authenticated observations
        // and is not an activation.  It must happen before evidence verification
        // so replay/trust artifacts bind to the exact report that sleep may later
        // promote; otherwise the evidence can only self-attest a sibling report.
        if reconstruction.promotion.allowed {
            let mixtures = reconstruction.skill_source_mixtures.clone();
            self.materialize_dense_fields(&mut reconstruction.fields, &mixtures, &observations)?;
        }
        let report_bytes = serialize_pretty_line(&reconstruction)?;
        let report_sha256 = sha256_bytes(&report_bytes);
        let corpus_digest = reconstruction.observation_set_digest.clone();
        let analysis_key = sleep_analysis_key(&corpus_digest, &reconstruction);
        let sleep_path = self.root.join("state/sleep_state.json");
        let (previous, has_prior_sleep_state) = match fs::symlink_metadata(&sleep_path) {
            Ok(_) => {
                let path = existing_regular_file_under_root(&self.root, &sleep_path)?;
                (
                    serde_json::from_slice::<serde_json::Value>(&fs::read(path)?)?,
                    true,
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (json!({}), false),
            Err(error) => return Err(error.into()),
        };

        // This is the one explicit initialization transition. No persisted
        // bank is substituted or accepted as a fallback.
        let prior_bank = match self.load_bank_for_initial_sleep_bootstrap()? {
            Some(bank) => bank,
            None => {
                self.authorize_empty_bank_bootstrap(has_prior_sleep_state, &previous)?;
                SkillBank::default()
            }
        };
        let diagnostics = diagnose_consolidation(
            &prior_bank,
            &reconstruction.fields,
            self.config.skill_match_cosine,
        )?;
        let evidence_path = self.root.join("state/sleep_evidence/current.json");
        let verified_evidence_path = match fs::symlink_metadata(&evidence_path) {
            Ok(_) => Some(existing_regular_file_under_root(
                &self.root,
                &evidence_path,
            )?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let evidence_bundle_sha256 = verified_evidence_path
            .as_ref()
            .map(|path| file_sha256(path))
            .transpose()?;
        let evidence_verification = match load_sleep_evidence(&self.root) {
            Ok(bundle) => verify_sleep_evidence(
                &self.root,
                &bundle,
                &SleepEvidenceExpectation {
                    corpus_digest: &corpus_digest,
                    report_sha256: &report_sha256,
                    source_tree_digest: &reconstruction.source_tree_digest,
                    config_digest: &reconstruction.config_digest,
                    analysis_version_digest: &reconstruction.analysis_version_digest,
                    fields: &reconstruction.fields,
                    observations: &observations,
                    source_mixtures: &reconstruction.skill_source_mixtures,
                },
            )?,
            Err(error) => SleepEvidenceVerification::failed(error.to_string()),
        };
        let certified = reconstruction.promotion.allowed && evidence_verification.verified;
        let certification_status = if certified {
            CertificationStatus::Certified
        } else {
            CertificationStatus::Revoked
        };
        let previous_same_certified = previous
            .get("analysis_key")
            .and_then(serde_json::Value::as_str)
            == Some(analysis_key.as_str())
            && previous
                .get("certification_status")
                .and_then(serde_json::Value::as_str)
                == Some(CertificationStatus::Certified.as_str());
        let should_promote = certified && !previous_same_certified;

        let mut bank = prior_bank.clone();
        let promoted = if should_promote {
            reconcile_full_corpus(
                &mut bank,
                &reconstruction.fields,
                self.config.skill_match_cosine,
            )?;
            true
        } else {
            false
        };
        let memory_fields = align_incoming_identities(
            &bank,
            &reconstruction.fields,
            self.config.skill_match_cosine,
        )?;
        let memory = build_memory_snapshot(
            &observations,
            bank.generation,
            certified,
            &memory_fields,
            &reconstruction.skill_source_mixtures,
        )?;
        let memory_bytes = serialize_pretty_line(&memory)?;
        let memory_sha256 = sha256_bytes(&memory_bytes);

        let bank_bytes = if bank.fields.is_empty() {
            None
        } else if promoted {
            Some(serialize_pretty_line(&bank)?)
        } else {
            // A revoked/non-promoting sleep must bind to the exact active-bank
            // bytes already installed. Re-serializing an older schema under the
            // current struct can change bytes without changing semantics and
            // would make the receipt point at a bank that was never activated.
            match fs::symlink_metadata(self.bank_path()) {
                Ok(_) => {
                    let bank_path =
                        existing_regular_file_under_root(&self.root, &self.bank_path())?;
                    Some(fs::read(bank_path)?)
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.into()),
            }
        };
        let active_bank_sha256 = bank_bytes.as_ref().map(|bytes| sha256_bytes(bytes));
        let operation_key = sleep_operation_key(
            &analysis_key,
            evidence_bundle_sha256.as_deref(),
            certification_status,
            active_bank_sha256.as_deref(),
        );
        let state = json!({
            "schema":"cerebro.tidex.sleep_state/v5",
            "operation_key":operation_key,
            "analysis_key":analysis_key,
            "corpus_digest":corpus_digest,
            "source_tree_digest":reconstruction.source_tree_digest,
            "config_digest":reconstruction.config_digest,
            "analysis_version_digest":reconstruction.analysis_version_digest,
            "report_sha256":report_sha256,
            "promoted":promoted,
            "certification_status":certification_status,
            "active_generation":bank.generation,
            "active_skill_count":bank.fields.len(),
            "active_bank_sha256":active_bank_sha256,
            "memory_digest":memory_sha256,
            "evidence_bundle_sha256":evidence_bundle_sha256,
            "evidence_verification":evidence_verification,
            "diagnostics":diagnostics,
        });
        let state_bytes = serialize_pretty_line(&state)?;
        let state_sha256 = sha256_bytes(&state_bytes);

        let receipts_dir = self.root.join("state/sleep_receipts");
        let transactions_dir = self.root.join("state/sleep_transactions");
        ensure_private_directory(&self.root, &receipts_dir)?;
        ensure_private_directory(&self.root, &transactions_dir)?;
        let receipt_path = receipts_dir.join(format!("{operation_key}.json"));
        let receipt_exists = match fs::symlink_metadata(&receipt_path) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if receipt_exists {
            let receipt_path = existing_regular_file_under_root(&self.root, &receipt_path)?;
            let receipt: SleepReceipt = serde_json::from_slice(&fs::read(&receipt_path)?)?;
            if receipt.operation_key != operation_key
                || receipt.analysis_key != analysis_key
                || receipt.report_sha256 != report_sha256
                || receipt.memory_sha256 != memory_sha256
                || receipt.active_bank_sha256 != active_bank_sha256
                || receipt.evidence_bundle_sha256 != evidence_bundle_sha256
                || receipt.sleep_state_sha256 != state_sha256
            {
                return Err(BrainError::Integrity("sleep_receipt_mismatch".into()));
            }
            let current_sleep_state = existing_regular_file_under_root(&self.root, &sleep_path)?;
            verify_sleep_receipt_ledger_binding(
                &self.root,
                &receipt,
                &state,
                &file_sha256(&current_sleep_state)?,
            )?;
            let report_history = self
                .root
                .join("state/reports")
                .join(format!("{report_sha256}.json"));
            let memory_history = memory_artifact_path(&self.root, &memory_sha256);
            let state_history = self
                .root
                .join("state/sleep/by-sha")
                .join(format!("{state_sha256}.json"));
            for (path, digest, label) in [
                (&report_history, &report_sha256, "report"),
                (&memory_history, &memory_sha256, "memory"),
                (&state_history, &state_sha256, "state"),
            ] {
                let verified = existing_regular_file_under_root(&self.root, path)?;
                if file_sha256(&verified)? != *digest {
                    return Err(BrainError::Integrity(format!(
                        "sleep_receipt_{label}_artifact_mismatch"
                    )));
                }
            }
            if let Some(bank_sha) = &active_bank_sha256 {
                let bank_history = self
                    .root
                    .join("state/skill_banks/by-sha")
                    .join(format!("{bank_sha}.json"));
                let bank_history = existing_regular_file_under_root(&self.root, &bank_history)?;
                if file_sha256(&bank_history)? != *bank_sha {
                    return Err(BrainError::Integrity(
                        "sleep_receipt_bank_artifact_mismatch".into(),
                    ));
                }
            }
            if let Some(evidence_sha) = &receipt.evidence_bundle_sha256 {
                let evidence_history = self
                    .root
                    .join("state/sleep_evidence/by-sha")
                    .join(format!("{evidence_sha}.json"));
                let evidence_history =
                    existing_regular_file_under_root(&self.root, &evidence_history)?;
                if file_sha256(&evidence_history)? != *evidence_sha {
                    return Err(BrainError::Integrity(
                        "sleep_receipt_evidence_artifact_mismatch".into(),
                    ));
                }
            }
            self.verify_current_sleep_installation(&receipt, &state)?;
            return Ok(SleepReport {
                schema: "cerebro.tidex.sleep/v4".into(),
                corpus_digest,
                observation_count: observations.len(),
                promoted: false,
                idempotent: true,
                active_skill_count: bank.fields.len(),
                memory_digest: memory_sha256,
                evidence_bundle_sha256,
                evidence_verification,
                diagnostics,
                reconstruction,
            });
        }

        let transaction_dir = transactions_dir.join(&operation_key);
        let intent_path = transaction_dir.join("intent.json");
        let staged_report = transaction_dir.join("report.json");
        let staged_memory = transaction_dir.join("memory.json");
        let staged_state = transaction_dir.join("sleep_state.json");
        let staged_bank = transaction_dir.join("skill_bank.json");
        let intent = SleepTransactionIntent {
            schema: "cerebro.tidex.sleep_transaction_intent/v1".into(),
            operation_key: operation_key.clone(),
            analysis_key: analysis_key.clone(),
            corpus_digest: corpus_digest.clone(),
            analysis_version_digest: reconstruction.analysis_version_digest.clone(),
            config_digest: reconstruction.config_digest.clone(),
            report_sha256: report_sha256.clone(),
            memory_sha256: memory_sha256.clone(),
            active_bank_sha256: active_bank_sha256.clone(),
            sleep_state_sha256: state_sha256.clone(),
            evidence_bundle_sha256: evidence_bundle_sha256.clone(),
            evidence_verified: evidence_verification.verified,
            certification_status,
            promoted,
        };
        let transaction_exists = match fs::symlink_metadata(&transaction_dir) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if transaction_exists {
            let transaction_dir = existing_directory_under_root(&self.root, &transaction_dir)?;
            let intent_path = existing_regular_file_under_root(&self.root, &intent_path)?;
            let staged_report = existing_regular_file_under_root(&self.root, &staged_report)?;
            let staged_memory = existing_regular_file_under_root(&self.root, &staged_memory)?;
            let staged_state = existing_regular_file_under_root(&self.root, &staged_state)?;
            let stored: SleepTransactionIntent = serde_json::from_slice(&fs::read(intent_path)?)?;
            if stored != intent
                || file_sha256(&staged_report)? != report_sha256
                || file_sha256(&staged_memory)? != memory_sha256
                || file_sha256(&staged_state)? != state_sha256
                || active_bank_sha256.as_ref().is_some_and(|sha| {
                    existing_regular_file_under_root(&self.root, &staged_bank)
                        .and_then(|path| file_sha256(&path))
                        .ok()
                        .as_deref()
                        != Some(sha.as_str())
                })
            {
                return Err(BrainError::Integrity(
                    "sleep_transaction_stage_mismatch".into(),
                ));
            }
            let _ = transaction_dir;
        } else {
            ensure_private_directory(&self.root, &transaction_dir)?;
            write_new_private(&self.root, &staged_report, &report_bytes)?;
            write_new_private(&self.root, &staged_memory, &memory_bytes)?;
            write_new_private(&self.root, &staged_state, &state_bytes)?;
            if let Some(bytes) = &bank_bytes {
                write_new_private(&self.root, &staged_bank, bytes)?;
            }
            write_new_private(&self.root, &intent_path, &serialize_pretty_line(&intent)?)?;
        }

        // The ledger event authorizes pointer changes. Recheck the mutable
        // evidence pointer immediately before that authorization, so an
        // external evidence replacement cannot leave a half-applied sleep
        // transaction merely by racing the earlier measurement.
        self.verify_current_sleep_evidence_pointer(evidence_bundle_sha256.as_deref())?;

        let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
            &self.root,
            "sleep_transaction",
            "operation_key",
            &operation_key,
        )? {
            verify_sleep_transaction_ledger_binding(&existing, &intent)?;
            existing
        } else {
            let event = ledger::append(
                &self.root,
                "sleep_transaction",
                json!({
                    "schema":"cerebro.tidex.sleep_transaction/v1",
                    "operation_key":operation_key,
                    "analysis_key":analysis_key,
                    "corpus_digest":corpus_digest,
                    "analysis_version_digest":reconstruction.analysis_version_digest,
                    "config_digest":reconstruction.config_digest,
                    "report_sha256":report_sha256,
                    "memory_sha256":memory_sha256,
                    "active_bank_sha256":active_bank_sha256,
                    "sleep_state_sha256":state_sha256,
                    "evidence_bundle_sha256":evidence_bundle_sha256,
                    "evidence_verified":evidence_verification.verified,
                    "certification_status":certification_status,
                    "promoted":promoted,
                }),
            )?;
            verify_sleep_transaction_ledger_binding(&event, &intent)?;
            event
        };

        let report_history = self
            .root
            .join("state/reports")
            .join(format!("{report_sha256}.json"));
        write_immutable_exact(&self.root, &report_history, &report_bytes, &report_sha256)?;
        let memory_history = memory_artifact_path(&self.root, &memory_sha256);
        write_immutable_exact(&self.root, &memory_history, &memory_bytes, &memory_sha256)?;
        replace_private_pointer_exact(
            &self.root,
            &self.root.join("state/memory/current.json"),
            &memory_bytes,
            &memory_sha256,
        )?;
        if let (Some(bytes), Some(bank_sha)) = (&bank_bytes, &active_bank_sha256) {
            let bank_history = self
                .root
                .join("state/skill_banks/by-sha")
                .join(format!("{bank_sha}.json"));
            write_immutable_exact(&self.root, &bank_history, bytes, bank_sha)?;
            if promoted {
                replace_private_pointer_exact(&self.root, &self.bank_path(), bytes, bank_sha)?;
            }
        }
        if let Some(evidence_sha) = &evidence_bundle_sha256 {
            let evidence_path = self
                .verify_current_sleep_evidence_pointer(Some(evidence_sha))?
                .ok_or_else(|| {
                    BrainError::Integrity("sleep_evidence_current_pointer_missing".into())
                })?;
            let evidence_bytes = fs::read(evidence_path)?;
            if sha256_bytes(&evidence_bytes) != *evidence_sha {
                return Err(BrainError::Integrity(
                    "sleep_evidence_changed_during_transaction".into(),
                ));
            }
            let evidence_history = self
                .root
                .join("state/sleep_evidence/by-sha")
                .join(format!("{evidence_sha}.json"));
            write_immutable_exact(&self.root, &evidence_history, &evidence_bytes, evidence_sha)?;
        }
        let state_history = self
            .root
            .join("state/sleep/by-sha")
            .join(format!("{state_sha256}.json"));
        write_immutable_exact(&self.root, &state_history, &state_bytes, &state_sha256)?;
        replace_private_pointer_exact(&self.root, &sleep_path, &state_bytes, &state_sha256)?;

        let receipt = SleepReceipt {
            schema: "cerebro.tidex.sleep_receipt/v1".into(),
            operation_key,
            analysis_key,
            report_sha256,
            memory_sha256: memory_sha256.clone(),
            active_bank_sha256,
            evidence_bundle_sha256: evidence_bundle_sha256.clone(),
            sleep_state_sha256: state_sha256,
            ledger_event_hash: event.event_hash,
        };
        write_new_private(&self.root, &receipt_path, &serialize_pretty_line(&receipt)?)?;
        // Reopen the current pointer through the same receipt/ledger verifier
        // before this sleep result is handed back.  A concurrent state change
        // therefore fails closed instead of returning a transient success.
        self.verify_current_sleep_installation(&receipt, &state)?;
        Ok(SleepReport {
            schema: "cerebro.tidex.sleep/v4".into(),
            corpus_digest,
            observation_count: observations.len(),
            promoted,
            idempotent: false,
            active_skill_count: bank.fields.len(),
            memory_digest: memory_sha256,
            evidence_bundle_sha256,
            evidence_verification,
            diagnostics,
            reconstruction,
        })
    }

    fn verify_runtime_composition_bank(&self, bank: &SkillBank) -> BrainResult<()> {
        if bank.fields.is_empty() {
            return Err(BrainError::Integrity(
                "runtime_composition_bank_empty".into(),
            ));
        }
        let layout_ids = bank
            .fields
            .iter()
            .map(|field| {
                field.parameter_layout_sha256.clone().ok_or_else(|| {
                    BrainError::Integrity(format!(
                        "runtime_composition_layout_missing:{}",
                        field.skill_id
                    ))
                })
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if layout_ids.len() != 1 {
            return Err(BrainError::Integrity(
                "runtime_composition_layout_identity_invalid".into(),
            ));
        }
        let layout_id = layout_ids
            .into_iter()
            .next()
            .ok_or_else(|| BrainError::Integrity("runtime_composition_layout_missing".into()))?;
        let layout = self.load_parameter_layout(&layout_id)?;
        for field in &bank.fields {
            if field.structured_geometry.is_none() {
                return Err(BrainError::Integrity(format!(
                    "runtime_composition_geometry_missing:{}",
                    field.skill_id
                )));
            }
            let reference = field.dense_materialization.as_ref().ok_or_else(|| {
                BrainError::Integrity(format!(
                    "runtime_composition_dense_missing:{}",
                    field.skill_id
                ))
            })?;
            if reference.parameter_count != layout.total_parameter_count {
                return Err(BrainError::Integrity(format!(
                    "runtime_composition_dense_count_mismatch:{}",
                    field.skill_id
                )));
            }
            self.verified_private_dvec(reference).map_err(|error| {
                BrainError::Integrity(format!(
                    "runtime_composition_dense_invalid:{}:{error}",
                    field.skill_id
                ))
            })?;
        }
        Ok(())
    }

    fn runtime_integrity_health(&self) -> BrainResult<RuntimeIntegrityHealth> {
        let mut integrity_reasons = Vec::<String>::new();
        let canonical_runtime_config = self.config == BrainConfig::default();
        if !canonical_runtime_config {
            integrity_reasons.push("noncanonical_runtime_config_forbidden".into());
        }
        let corpus_transition_clear = match self.require_no_incomplete_corpus_transition() {
            Ok(()) => true,
            Err(error) => {
                integrity_reasons.push(format!("corpus_transition_incomplete:{error}"));
                false
            }
        };
        let ledger_status = match ledger::verify(&self.root) {
            Ok(status) => Some(status),
            Err(error) => {
                integrity_reasons.push(format!("ledger_invalid:{error}"));
                None
            }
        };
        let ledger_verified = ledger_status.is_some();

        let bank = match self.load_bank() {
            Ok(bank) => Some(bank),
            Err(error) => {
                integrity_reasons.push(format!("active_bank_unavailable:{error}"));
                None
            }
        };
        let mut bank_verified = bank.as_ref().is_some_and(|bank| !bank.fields.is_empty());
        if !bank_verified {
            integrity_reasons.push("active_bank_empty_or_missing".into());
        } else if let Some(bank) = bank.as_ref() {
            let dimension = bank.fields[0].direction.len();
            let mut ids = BTreeSet::new();
            for field in &bank.fields {
                if field.skill_id.trim().is_empty()
                    || field.reconstruction_id.trim().is_empty()
                    || field.lineage_id.trim().is_empty()
                    || !ids.insert(field.skill_id.as_str())
                    || dimension == 0
                    || field.direction.len() != dimension
                    || field.direction.iter().any(|value| !value.is_finite())
                    || !field.persistence.is_finite()
                    || !field.coherence.is_finite()
                    || !field.uncertainty.is_finite()
                {
                    bank_verified = false;
                    break;
                }
            }
            if !bank_verified {
                integrity_reasons.push("active_bank_contract_invalid".into());
            }
        }

        let composition_ready = if let Some(bank) = bank.as_ref().filter(|_| bank_verified) {
            match self.verify_runtime_composition_bank(bank) {
                Ok(()) => true,
                Err(error) => {
                    integrity_reasons.push(format!("runtime_composition_unready:{error}"));
                    false
                }
            }
        } else {
            false
        };

        let observations = self.load_persisted_observations()?;
        let mut current_corpus_digest = None;
        let observations_verified = match self.validate_observations(&observations).and_then(|_| {
            estimate_aperture_independence(&observations, self.config.min_independent_apertures)
                .map(|_| ())
        }) {
            Ok(()) => match observation_set_digest(&canonical_observations(&observations)) {
                Ok(digest) => {
                    current_corpus_digest = Some(digest);
                    true
                }
                Err(error) => {
                    integrity_reasons.push(format!("current_corpus_digest_invalid:{error}"));
                    false
                }
            },
            Err(error) => {
                integrity_reasons.push(format!("observations_invalid:{error}"));
                false
            }
        };

        let sleep_path = self.root.join("state/sleep_state.json");
        let state = match fs::symlink_metadata(&sleep_path) {
            Ok(_) => match existing_regular_file_under_root(&self.root, &sleep_path) {
                Ok(verified_path) => {
                    match serde_json::from_slice::<Value>(&fs::read(&verified_path)?) {
                        Ok(value)
                            if value.get("schema").and_then(Value::as_str)
                                == Some("cerebro.tidex.sleep_state/v5") =>
                        {
                            Some(value)
                        }
                        Ok(_) => {
                            integrity_reasons.push("sleep_state_schema_invalid".into());
                            None
                        }
                        Err(error) => {
                            integrity_reasons.push(format!("sleep_state_parse_failed:{error}"));
                            None
                        }
                    }
                }
                Err(error) => {
                    integrity_reasons.push(format!("sleep_state_path_invalid:{error}"));
                    None
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                integrity_reasons.push("sleep_state_missing".into());
                None
            }
            Err(error) => {
                integrity_reasons.push(format!("sleep_state_metadata_failed:{error}"));
                None
            }
        };
        let sleep_state_verified = state.is_some();
        let operation_key = state
            .as_ref()
            .and_then(|value| value.get("operation_key"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let raw_certification_status = state
            .as_ref()
            .and_then(|value| value.get("certification_status"))
            .and_then(serde_json::Value::as_str);
        let certification_status = raw_certification_status.and_then(CertificationStatus::parse);
        if raw_certification_status.is_some_and(|value| CertificationStatus::parse(value).is_none())
        {
            integrity_reasons.push("certification_status_invalid".into());
        }
        let certified = certification_status == Some(CertificationStatus::Certified);
        let evidence_verified = state
            .as_ref()
            .and_then(|value| value.get("evidence_verification"))
            .and_then(|value| value.get("verified"))
            .and_then(serde_json::Value::as_bool)
            == Some(true);

        let mut receipt_verified = false;
        let mut historical_artifacts_verified = false;
        let mut current_pointers_verified = false;
        let mut report_state_consistent = false;
        let mut current_corpus_bound = false;
        let mut analysis_current = false;
        if let (Some(state), Some(operation_key)) = (state.as_ref(), operation_key.as_deref()) {
            let receipt_path = self
                .root
                .join("state/sleep_receipts")
                .join(format!("{operation_key}.json"));
            let receipt_present = match fs::symlink_metadata(&receipt_path) {
                Ok(_) => true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => {
                    integrity_reasons.push(format!("sleep_receipt_metadata_failed:{error}"));
                    false
                }
            };
            if receipt_present {
                match existing_regular_file_under_root(&self.root, &receipt_path)
                    .and_then(|path| Ok(serde_json::from_slice::<SleepReceipt>(&fs::read(path)?)?))
                {
                    Ok(receipt) => {
                        match existing_regular_file_under_root(&self.root, &sleep_path).and_then(
                            |verified_state| {
                                verify_sleep_receipt_ledger_binding(
                                    &self.root,
                                    &receipt,
                                    state,
                                    &file_sha256(&verified_state)?,
                                )
                            },
                        ) {
                            Ok(()) => receipt_verified = true,
                            Err(error) => {
                                integrity_reasons
                                    .push(format!("sleep_receipt_chain_mismatch:{error}"));
                            }
                        }

                        let report_history = self
                            .root
                            .join("state/reports")
                            .join(format!("{}.json", receipt.report_sha256));
                        let memory_history =
                            memory_artifact_path(&self.root, &receipt.memory_sha256);
                        let state_history = self
                            .root
                            .join("state/sleep/by-sha")
                            .join(format!("{}.json", receipt.sleep_state_sha256));
                        let bank_history = receipt.active_bank_sha256.as_ref().map(|sha| {
                            self.root
                                .join("state/skill_banks/by-sha")
                                .join(format!("{sha}.json"))
                        });
                        let evidence_history = receipt.evidence_bundle_sha256.as_ref().map(|sha| {
                            self.root
                                .join("state/sleep_evidence/by-sha")
                                .join(format!("{sha}.json"))
                        });
                        historical_artifacts_verified =
                            private_file_digest_matches(
                                &self.root,
                                &report_history,
                                &receipt.report_sha256,
                            ) && private_file_digest_matches(
                                &self.root,
                                &memory_history,
                                &receipt.memory_sha256,
                            ) && private_file_digest_matches(
                                &self.root,
                                &state_history,
                                &receipt.sleep_state_sha256,
                            ) && bank_history.as_ref().is_none_or(|path| {
                                receipt.active_bank_sha256.as_ref().is_some_and(|sha| {
                                    private_file_digest_matches(&self.root, path, sha)
                                })
                            }) && evidence_history.as_ref().is_none_or(|path| {
                                receipt.evidence_bundle_sha256.as_ref().is_some_and(|sha| {
                                    private_file_digest_matches(&self.root, path, sha)
                                })
                            });
                        if !historical_artifacts_verified {
                            integrity_reasons.push("sleep_historical_artifact_mismatch".into());
                        }

                        let current_memory = self.root.join("state/memory/current.json");
                        let current_evidence = self.root.join("state/sleep_evidence/current.json");
                        current_pointers_verified =
                            receipt.active_bank_sha256.as_ref().is_some_and(|sha| {
                                private_file_digest_matches(&self.root, &self.bank_path(), sha)
                            }) && private_file_digest_matches(
                                &self.root,
                                &current_memory,
                                &receipt.memory_sha256,
                            ) && receipt.evidence_bundle_sha256.as_ref().is_some_and(|sha| {
                                private_file_digest_matches(&self.root, &current_evidence, sha)
                            });
                        if !current_pointers_verified {
                            integrity_reasons.push("current_pointer_digest_mismatch".into());
                        }

                        if historical_artifacts_verified {
                            let report_history =
                                existing_regular_file_under_root(&self.root, &report_history)?;
                            match serde_json::from_slice::<Value>(&fs::read(&report_history)?) {
                                Ok(report) => {
                                    let report_source = report
                                        .get("source_tree_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let report_config = report
                                        .get("config_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let report_analysis = report
                                        .get("analysis_version_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let state_source = state
                                        .get("source_tree_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let state_config = state
                                        .get("config_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let state_analysis = state
                                        .get("analysis_version_digest")
                                        .and_then(serde_json::Value::as_str);
                                    report_state_consistent = report_source.is_some()
                                        && report_config.is_some()
                                        && report_analysis.is_some()
                                        && state_source == report_source
                                        && state_config == report_config
                                        && state_analysis == report_analysis;
                                    if !report_state_consistent {
                                        integrity_reasons
                                            .push("report_sleep_state_identity_mismatch".into());
                                    }
                                    let state_corpus = state
                                        .get("corpus_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let report_corpus = report
                                        .get("observation_set_digest")
                                        .and_then(serde_json::Value::as_str);
                                    let recomputed_analysis_key =
                                        serde_json::from_value::<ReconstructionReport>(
                                            report.clone(),
                                        )
                                        .ok()
                                        .map(|typed| {
                                            sleep_analysis_key(
                                                typed.observation_set_digest.as_str(),
                                                &typed,
                                            )
                                        });
                                    current_corpus_bound = current_corpus_digest
                                        .as_deref()
                                        .is_some_and(|digest| state_corpus == Some(digest))
                                        && state_corpus == report_corpus
                                        && recomputed_analysis_key.as_deref().is_some_and(|key| {
                                            state.get("analysis_key").and_then(Value::as_str)
                                                == Some(key)
                                        });
                                    if !current_corpus_bound {
                                        integrity_reasons
                                            .push("current_corpus_sleep_binding_mismatch".into());
                                    }
                                    let (source, config, analysis) =
                                        analysis_identity(&self.config)?;
                                    analysis_current = report_source == Some(source.as_str())
                                        && report_config == Some(config.as_str())
                                        && report_analysis == Some(analysis.as_str())
                                        && current_corpus_bound;
                                }
                                Err(error) => {
                                    integrity_reasons
                                        .push(format!("historical_report_json_invalid:{error}"));
                                }
                            }
                        }
                    }
                    Err(error) => {
                        integrity_reasons.push(format!("sleep_receipt_parse_failed:{error}"));
                    }
                }
            } else {
                integrity_reasons.push("sleep_receipt_missing".into());
            }
        }

        let integrity_healthy = canonical_runtime_config
            && corpus_transition_clear
            && ledger_verified
            && bank_verified
            && observations_verified
            && sleep_state_verified
            && receipt_verified
            && historical_artifacts_verified
            && current_pointers_verified
            && report_state_consistent
            && current_corpus_bound;
        let mut execution_blockers = Vec::<String>::new();
        if !canonical_runtime_config {
            execution_blockers.push("noncanonical_runtime_config_forbidden".into());
        }
        if !corpus_transition_clear {
            execution_blockers.push("corpus_transition_incomplete".into());
        }
        if !integrity_healthy {
            execution_blockers.push("integrity_unhealthy".into());
        }
        if !analysis_current {
            execution_blockers.push("analysis_version_stale".into());
        }
        if !certified {
            execution_blockers.push(format!(
                "certification_status:{}",
                certification_status
                    .map(CertificationStatus::as_str)
                    .unwrap_or("missing")
            ));
        }
        if !evidence_verified {
            execution_blockers.push("sleep_evidence_unverified".into());
        }
        if !composition_ready {
            execution_blockers.push("runtime_composition_unready".into());
        }
        let execution_authorized = execution_blockers.is_empty();
        Ok(RuntimeIntegrityHealth {
            schema: "cerebro.tidex.runtime_integrity_health/v1".into(),
            canonical_runtime_config,
            corpus_transition_clear,
            ledger_verified,
            bank_verified,
            composition_ready,
            observations_verified,
            sleep_state_verified,
            receipt_verified,
            historical_artifacts_verified,
            current_pointers_verified,
            report_state_consistent,
            current_corpus_bound,
            analysis_current,
            certified,
            evidence_verified,
            integrity_healthy,
            execution_authorized,
            operation_key,
            certification_status,
            integrity_reasons,
            execution_blockers,
        })
    }

    fn require_current_certification(&self) -> BrainResult<()> {
        self.require_canonical_runtime_config()?;
        let health = self.runtime_integrity_health()?;
        if !health.execution_authorized {
            return Err(BrainError::Integrity(format!(
                "runtime_execution_not_authorized:{}",
                health.execution_blockers.join("|")
            )));
        }
        Ok(())
    }

    fn canonical_evidence_path(&self, raw: &str, expected_sha256: &str) -> BrainResult<PathBuf> {
        if expected_sha256.len() != 64
            || !expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(BrainError::Invalid(
                "runtime_evidence_digest_invalid".into(),
            ));
        }
        let path = PathBuf::from(raw);
        let path = existing_regular_file_under_root(&self.root, &path)
            .map_err(|_| BrainError::Integrity("runtime_evidence_path_invalid".into()))?;
        if file_sha256(&path)? != expected_sha256 {
            return Err(BrainError::Integrity(
                "runtime_evidence_identity_mismatch".into(),
            ));
        }
        Ok(path)
    }

    fn load_verified_runtime_evidence(
        &self,
        bank: &SkillBank,
    ) -> BrainResult<(ProtectedCortex, Matrix, f64, CausalCreditReport)> {
        let observations = canonical_observations(&self.load_persisted_observations()?);
        let mut reconstruction = self.analyze_canonical(&observations)?;
        if reconstruction.promotion.allowed {
            let mixtures = reconstruction.skill_source_mixtures.clone();
            self.materialize_dense_fields(&mut reconstruction.fields, &mixtures, &observations)?;
        }
        let reconstruction_bytes = serialize_pretty_line(&reconstruction)?;
        let reconstruction_sha256 = sha256_bytes(&reconstruction_bytes);
        let bundle = load_sleep_evidence(&self.root)?;
        let verification = verify_sleep_evidence(
            &self.root,
            &bundle,
            &SleepEvidenceExpectation {
                corpus_digest: &reconstruction.observation_set_digest,
                report_sha256: &reconstruction_sha256,
                source_tree_digest: &reconstruction.source_tree_digest,
                config_digest: &reconstruction.config_digest,
                analysis_version_digest: &reconstruction.analysis_version_digest,
                fields: &reconstruction.fields,
                observations: &observations,
                source_mixtures: &reconstruction.skill_source_mixtures,
            },
        )?;
        if !verification.verified {
            return Err(BrainError::Integrity(format!(
                "runtime_evidence_reverification_failed:{}",
                verification.reasons.join("|")
            )));
        }
        let bank_ids = bank
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<Vec<_>>();
        if bank_ids != bundle.field_ids {
            return Err(BrainError::Integrity(
                "runtime_evidence_bank_identity_mismatch".into(),
            ));
        }

        let protected_path = self.canonical_evidence_path(
            &bundle.protection.protected_map_path,
            &bundle.protection.protected_map_sha256,
        )?;
        let protected_wrapper: serde_json::Value =
            serde_json::from_slice(&fs::read(protected_path)?)?;
        if protected_wrapper
            .get("schema")
            .and_then(serde_json::Value::as_str)
            != Some("cerebro.tidex.protected_map_benchmark/v2")
            || protected_wrapper
                .get("task_labels_used")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
        {
            return Err(BrainError::Integrity(
                "runtime_protected_map_contract_invalid".into(),
            ));
        }
        let protected_map: ProtectedMapArtifactReport = serde_json::from_value(
            protected_wrapper
                .get("map")
                .cloned()
                .ok_or_else(|| BrainError::Integrity("runtime_protected_map_missing".into()))?,
        )?;
        let protected = load_protected_cortex(&self.root, &protected_map)?;

        let interaction_path = self.canonical_evidence_path(
            &bundle.interaction.source_path,
            &bundle.interaction.source_sha256,
        )?;
        let interaction_payload: serde_json::Value =
            serde_json::from_slice(&fs::read(interaction_path)?)?;
        if interaction_payload
            .get("schema")
            .and_then(serde_json::Value::as_str)
            != Some("cerebro.tidex.trust_region_benchmark/v3")
        {
            return Err(BrainError::Integrity(
                "runtime_interaction_contract_invalid".into(),
            ));
        }
        let interaction_ids = interaction_payload
            .get("field_ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| BrainError::Integrity("runtime_interaction_ids_missing".into()))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| BrainError::Integrity("runtime_interaction_id_invalid".into()))
            })
            .collect::<BrainResult<Vec<_>>>()?;
        if interaction_ids != bank_ids {
            return Err(BrainError::Integrity(
                "runtime_interaction_bank_identity_mismatch".into(),
            ));
        }
        let interaction_rows = interaction_payload
            .get("interaction_matrix")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| BrainError::Integrity("runtime_interaction_matrix_missing".into()))?
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| BrainError::Integrity("runtime_interaction_row_invalid".into()))?
                    .iter()
                    .map(|value| {
                        value.as_f64().ok_or_else(|| {
                            BrainError::Integrity("runtime_interaction_value_invalid".into())
                        })
                    })
                    .collect::<BrainResult<Vec<_>>>()
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let interaction = Matrix::from_rows(&interaction_rows)?;
        if interaction.rows != bank.fields.len() || interaction.cols != bank.fields.len() {
            return Err(BrainError::Integrity(
                "runtime_interaction_shape_mismatch".into(),
            ));
        }
        let max_quadratic_cost = interaction_payload
            .get("diagonal_budget")
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| BrainError::Integrity("runtime_trust_budget_invalid".into()))?;
        let trust_causal_credit_sha256 = interaction_payload
            .get("causal_credit_sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|value| valid_digest(value))
            .ok_or_else(|| BrainError::Integrity("runtime_trust_causal_digest_missing".into()))?;
        if trust_causal_credit_sha256 != bundle.causal_credit.credit_source_sha256 {
            return Err(BrainError::Integrity(
                "runtime_trust_causal_digest_mismatch".into(),
            ));
        }
        let trust_payload = interaction_payload
            .get("trust_region")
            .ok_or_else(|| BrainError::Integrity("runtime_trust_payload_missing".into()))?;
        if trust_payload
            .get("allocation_policy")
            .and_then(serde_json::Value::as_str)
            != Some("causal_priority_contraction/v1")
        {
            return Err(BrainError::Integrity(
                "runtime_trust_policy_not_causal".into(),
            ));
        }
        let stored_causal_priority_weights = trust_payload
            .get("causal_priority_weights")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| BrainError::Integrity("runtime_trust_causal_weights_missing".into()))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| {
                        BrainError::Integrity("runtime_trust_causal_weight_invalid".into())
                    })
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let component_retention = trust_payload
            .get("component_retention")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                BrainError::Integrity("runtime_trust_component_retention_missing".into())
            })?;
        if stored_causal_priority_weights.len() != bank.fields.len()
            || component_retention.len() != bank.fields.len()
            || component_retention.iter().any(|value| {
                value
                    .as_f64()
                    .is_none_or(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            })
        {
            return Err(BrainError::Integrity(
                "runtime_trust_causal_contract_invalid".into(),
            ));
        }

        let causal_path = self.canonical_evidence_path(
            &bundle.causal_credit.credit_source_path,
            &bundle.causal_credit.credit_source_sha256,
        )?;
        let causal_wrapper: serde_json::Value = serde_json::from_slice(&fs::read(causal_path)?)?;
        if causal_wrapper
            .get("schema")
            .and_then(serde_json::Value::as_str)
            != Some("cerebro.tidex.causal_credit_benchmark/v3")
            || causal_wrapper
                .get("blind_data_accessed")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
        {
            return Err(BrainError::Integrity(
                "runtime_causal_credit_contract_invalid".into(),
            ));
        }
        let causal: CausalCreditReport = serde_json::from_value(
            causal_wrapper
                .get("causal_credit")
                .cloned()
                .ok_or_else(|| BrainError::Integrity("runtime_causal_credit_missing".into()))?,
        )?;
        let causal_ids = causal
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<BTreeSet<_>>();
        let bank_id_set = bank_ids.iter().cloned().collect::<BTreeSet<_>>();
        if causal_ids != bank_id_set || !causal.unresolved_fields.is_empty() {
            return Err(BrainError::Integrity(
                "runtime_causal_credit_identity_unresolved".into(),
            ));
        }
        let expected_causal_priority_weights =
            certified_causal_priority_weights(&causal, &bank_ids)?;
        let causal_tolerance = f64::EPSILON.sqrt() * bank_ids.len().max(1) as f64 * 32.0;
        if stored_causal_priority_weights
            .iter()
            .zip(&expected_causal_priority_weights)
            .any(|(stored, expected)| {
                (stored - expected).abs() > causal_tolerance * (1.0 + expected.abs())
            })
        {
            return Err(BrainError::Integrity(
                "runtime_trust_causal_weight_mismatch".into(),
            ));
        }
        Ok((protected, interaction, max_quadratic_cost, causal))
    }

    fn compose_verified_inputs(
        &self,
        bank: &SkillBank,
        activation: &BTreeMap<String, f64>,
        protected: &ProtectedCortex,
        interaction_metric: &Matrix,
        max_quadratic_cost: f64,
        causal_credit: &CausalCreditReport,
    ) -> BrainResult<GovernedComposition> {
        if bank.fields.is_empty() || activation.is_empty() {
            return Err(BrainError::Invalid("skill_activation_empty".into()));
        }
        let mut proposed = vec![0.0; bank.fields.len()];
        for (id, coefficient) in activation {
            if !coefficient.is_finite() {
                return Err(BrainError::Invalid("activation_non_finite".into()));
            }
            let field_index = bank
                .fields
                .iter()
                .position(|field| &field.skill_id == id)
                .ok_or_else(|| BrainError::Invalid(format!("skill_not_found:{id}")))?;
            proposed[field_index] = *coefficient;
        }
        let field_ids = bank
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<Vec<_>>();
        let causal_priority_weights = certified_causal_priority_weights(causal_credit, &field_ids)?;
        let trust = apply_causal_priority_trust_region(
            interaction_metric,
            &proposed,
            max_quadratic_cost,
            &causal_priority_weights,
        )?;

        let layout_ids = bank
            .fields
            .iter()
            .map(|field| {
                field.parameter_layout_sha256.clone().ok_or_else(|| {
                    BrainError::Integrity(format!(
                        "runtime_field_layout_missing:{}",
                        field.skill_id
                    ))
                })
            })
            .collect::<BrainResult<BTreeSet<_>>>()?;
        if layout_ids.len() != 1 {
            return Err(BrainError::Integrity(
                "runtime_skill_bank_multiple_parameter_layouts".into(),
            ));
        }
        let layout_sha = layout_ids.into_iter().next().unwrap();
        let layout = self.load_parameter_layout(&layout_sha)?;
        if protected.parameter_importance.len() != layout.total_parameter_count as usize {
            return Err(BrainError::Invalid(
                "protected_cortex_parameter_space_mismatch".into(),
            ));
        }
        let mut delta = vec![0.0f64; layout.total_parameter_count as usize];
        for (field, coefficient) in bank.fields.iter().zip(&trust.accepted_coefficients) {
            if coefficient.abs() <= f64::EPSILON {
                continue;
            }
            if field.structured_geometry.is_none() {
                return Err(BrainError::Integrity(format!(
                    "runtime_field_structured_geometry_missing:{}",
                    field.skill_id
                )));
            }
            let reference = field.dense_materialization.as_ref().ok_or_else(|| {
                BrainError::Integrity(format!(
                    "runtime_dense_materialization_missing:{}",
                    field.skill_id
                ))
            })?;
            if reference.parameter_count != layout.total_parameter_count {
                return Err(BrainError::Integrity(format!(
                    "runtime_dense_materialization_count_mismatch:{}",
                    field.skill_id
                )));
            }
            self.verified_private_dvec(reference)?;
            let values = read_dvec_f32(reference)?;
            if values.len() != delta.len() {
                return Err(BrainError::Integrity(
                    "runtime_dense_materialization_length_mismatch".into(),
                ));
            }
            for (output, value) in delta.iter_mut().zip(values) {
                *output += coefficient * f64::from(value);
            }
        }
        if delta.iter().all(|value| value.abs() <= f64::EPSILON) {
            return Err(BrainError::Invalid("runtime_composed_delta_zero".into()));
        }
        let protection = project_to_safe_subspace(&delta, protected)?;
        if !protection.allowed {
            return Err(BrainError::Integrity(format!(
                "protected_cortex_damage_budget_exceeded:{:.6e}",
                protection.damage_ratio
            )));
        }
        Ok(GovernedComposition {
            delta: protection.projected.clone(),
            trust_region: trust,
            protection,
        })
    }

    fn load_current_observation_by_semantic_sha256(
        &self,
        digest: &str,
    ) -> BrainResult<DeltaObservation> {
        if !valid_digest(digest) {
            return Err(BrainError::Invalid(
                "governed_composition_source_observation_digest_invalid".into(),
            ));
        }
        let mut matches = Vec::new();
        for observation in self.load_persisted_observations()? {
            if digest_json(&observation)? == digest {
                matches.push(observation);
            }
        }
        if matches.len() != 1 {
            return Err(BrainError::Integrity(
                "governed_composition_source_observation_not_current".into(),
            ));
        }
        Ok(matches
            .into_iter()
            .next()
            .expect("exactly one checked above"))
    }

    fn require_current_observation_digest(&self, digest: &str) -> BrainResult<()> {
        let _ = self.load_current_observation_by_semantic_sha256(digest)?;
        Ok(())
    }

    /// Compose only from the currently certified TIDE-X evidence. This stays
    /// crate-private: executable deltas leave the engine only through an
    /// immutable governed-composition receipt.
    fn compose(&self, activation: &BTreeMap<String, f64>) -> BrainResult<GovernedComposition> {
        self.require_current_certification()?;
        let bank = self.load_bank()?;
        let (protected, interaction, budget, causal) =
            self.load_verified_runtime_evidence(&bank)?;
        self.compose_verified_inputs(&bank, activation, &protected, &interaction, budget, &causal)
    }

    /// Compute a composition through current certified causal trust and record
    /// its executable projected delta in an immutable, ledger-bound receipt.
    /// This is the canonical operational hand-off for external actuators and
    /// for LearnedController supervision; no caller can attach free target
    /// coefficients to an observation without this receipt.
    fn compose_and_record(
        &self,
        activation: &BTreeMap<String, f64>,
        source_observation_sha256: &str,
    ) -> BrainResult<RecordedGovernedComposition> {
        self.require_current_observation_digest(source_observation_sha256)?;
        let composition = self.compose(activation)?;
        let bank = self.load_bank()?;
        let field_ids = bank
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<Vec<_>>();
        if field_ids.is_empty()
            || activation.is_empty()
            || activation.values().any(|value| !value.is_finite())
        {
            return Err(BrainError::Integrity(
                "governed_composition_activation_contract_invalid".into(),
            ));
        }
        let bank_path = self.bank_path();
        let bank_path = existing_regular_file_under_root(&self.root, &bank_path).map_err(|_| {
            BrainError::Integrity("governed_composition_active_bank_missing".into())
        })?;
        let active_bank_sha256 = file_sha256(&bank_path)?;
        let sleep_path = existing_regular_file_under_root(
            &self.root,
            &self.root.join("state/sleep_state.json"),
        )?;
        let sleep_state: serde_json::Value = serde_json::from_slice(&fs::read(sleep_path)?)?;
        let report_sha256 = sleep_state
            .get("report_sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|digest| valid_digest(digest))
            .ok_or_else(|| {
                BrainError::Integrity("governed_composition_report_digest_missing".into())
            })?
            .to_string();
        let evidence_bundle_sha256 = sleep_state
            .get("evidence_bundle_sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|digest| valid_digest(digest))
            .ok_or_else(|| {
                BrainError::Integrity("governed_composition_evidence_digest_missing".into())
            })?
            .to_string();
        if sleep_state
            .get("active_bank_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(active_bank_sha256.as_str())
        {
            return Err(BrainError::Integrity(
                "governed_composition_sleep_bank_mismatch".into(),
            ));
        }
        let evidence = load_sleep_evidence(&self.root)?;
        let causal_credit_sha256 = evidence.causal_credit.credit_source_sha256.clone();
        if !valid_digest(&causal_credit_sha256) {
            return Err(BrainError::Integrity(
                "governed_composition_causal_digest_invalid".into(),
            ));
        }
        let projected_values = composition
            .delta
            .iter()
            .map(|value| {
                if !value.is_finite() || value.abs() > f32::MAX as f64 {
                    return Err(BrainError::Numerical(
                        "governed_composition_projected_delta_nonrepresentable".into(),
                    ));
                }
                Ok(*value as f32)
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let projected_delta = create_content_addressed_dvec(&self.root, &projected_values)?;
        let operation_key = governed_composition_operation_key(
            &report_sha256,
            &active_bank_sha256,
            &evidence_bundle_sha256,
            &causal_credit_sha256,
            activation,
            &projected_delta.sha256,
            source_observation_sha256,
        )?;
        let root = self.root.join("state/governed_compositions");
        let by_sha = root.join("by-sha");
        let by_operation = root.join("by-operation");
        ensure_private_directory(&self.root, &by_sha)?;
        ensure_private_directory(&self.root, &by_operation)?;
        let pointer_path = by_operation.join(format!("{operation_key}.json"));
        if fs::symlink_metadata(&pointer_path).is_ok() {
            let pointer_path = existing_regular_file_under_root(&self.root, &pointer_path)?;
            let pointer: GovernedCompositionPointer =
                serde_json::from_slice(&fs::read(&pointer_path)?)?;
            if pointer.schema != "cerebro.tidex.governed_composition_pointer/v1"
                || pointer.operation_key != operation_key
                || !valid_digest(&pointer.receipt_sha256)
            {
                return Err(BrainError::Integrity(
                    "governed_composition_pointer_contract_invalid".into(),
                ));
            }
            let receipt_path = by_sha.join(format!("{}.json", pointer.receipt_sha256));
            let receipt = load_verified_governed_composition_receipt(
                &self.root,
                &receipt_path,
                &pointer.receipt_sha256,
            )?;
            if receipt.operation_key != operation_key
                || receipt.requested_activation != *activation
                || receipt.source_observation_sha256 != source_observation_sha256
            {
                return Err(BrainError::Integrity(
                    "governed_composition_pointer_receipt_mismatch".into(),
                ));
            }
            let event = ledger::find_v2_event_by_payload_string(
                &self.root,
                "governed_composition_receipt",
                "receipt_sha256",
                &pointer.receipt_sha256,
            )?
            .ok_or_else(|| {
                BrainError::Integrity("governed_composition_pointer_ledger_missing".into())
            })?;
            return Ok(RecordedGovernedComposition {
                receipt_path: receipt_path.to_string_lossy().into_owned(),
                receipt_sha256: pointer.receipt_sha256,
                ledger_event_hash: event.event_hash,
                receipt,
            });
        }

        let receipt = GovernedCompositionReceipt {
            schema: "cerebro.tidex.governed_composition_receipt/v1".into(),
            operation_key: operation_key.clone(),
            report_sha256: report_sha256.clone(),
            active_bank_sha256: active_bank_sha256.clone(),
            evidence_bundle_sha256: evidence_bundle_sha256.clone(),
            causal_credit_sha256: causal_credit_sha256.clone(),
            field_ids,
            requested_activation: activation.clone(),
            accepted_coefficients: composition.trust_region.accepted_coefficients.clone(),
            trust_region: composition.trust_region.clone(),
            projected_delta,
            protection: GovernedCompositionProtection {
                damage_ratio: composition.protection.damage_ratio,
                allowed: composition.protection.allowed,
                removed_energy: composition.protection.removed_energy,
                protected_rank: composition.protection.protected_rank,
                max_weighted_residual: composition.protection.max_weighted_residual,
            },
            source_observation_sha256: source_observation_sha256.to_string(),
        };
        let receipt_bytes = serialize_pretty_line(&receipt)?;
        let receipt_sha256 = sha256_bytes(&receipt_bytes);
        let receipt_path = by_sha.join(format!("{receipt_sha256}.json"));
        write_new_private(&self.root, &receipt_path, &receipt_bytes)?;
        let event = if let Some(existing) = ledger::find_v2_event_by_payload_string(
            &self.root,
            "governed_composition_receipt",
            "receipt_sha256",
            &receipt_sha256,
        )? {
            let payload = existing.payload()?;
            if payload.get("schema").and_then(serde_json::Value::as_str)
                != Some("cerebro.tidex.governed_composition_ledger_binding/v1")
                || payload
                    .get("receipt_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(receipt_sha256.as_str())
                || payload
                    .get("operation_key")
                    .and_then(serde_json::Value::as_str)
                    != Some(operation_key.as_str())
                || payload
                    .get("report_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(report_sha256.as_str())
                || payload
                    .get("active_bank_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(active_bank_sha256.as_str())
                || payload
                    .get("causal_credit_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(causal_credit_sha256.as_str())
                || payload
                    .get("evidence_bundle_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(evidence_bundle_sha256.as_str())
                || payload
                    .get("source_observation_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(source_observation_sha256)
            {
                return Err(BrainError::Integrity(
                    "governed_composition_ledger_payload_mismatch".into(),
                ));
            }
            existing
        } else {
            ledger::append(
                &self.root,
                "governed_composition_receipt",
                json!({
                    "schema":"cerebro.tidex.governed_composition_ledger_binding/v1",
                    "receipt_sha256":receipt_sha256,
                    "operation_key":operation_key,
                    "report_sha256":report_sha256,
                    "active_bank_sha256":active_bank_sha256,
                    "evidence_bundle_sha256":evidence_bundle_sha256,
                    "causal_credit_sha256":causal_credit_sha256,
                    "source_observation_sha256":source_observation_sha256,
                }),
            )?
        };
        // Reopen through the full causal/trust/protection rederivation before
        // publishing an operation pointer.  If sleep, bank, evidence, or the
        // source observation changed after `compose`, this fails closed rather
        // than handing a stale in-memory delta to an actuator.
        let verified_before_pointer =
            load_verified_governed_composition_receipt(&self.root, &receipt_path, &receipt_sha256)?;
        if verified_before_pointer != receipt {
            return Err(BrainError::Integrity(
                "governed_composition_pre_pointer_revalidation_mismatch".into(),
            ));
        }
        let pointer = GovernedCompositionPointer {
            schema: "cerebro.tidex.governed_composition_pointer/v1".into(),
            operation_key,
            receipt_sha256: receipt_sha256.clone(),
        };
        let pointer_bytes = serialize_pretty_line(&pointer)?;
        match fs::symlink_metadata(&pointer_path) {
            Ok(_) => {
                let pointer_path = existing_regular_file_under_root(&self.root, &pointer_path)?;
                if fs::read(&pointer_path)? != pointer_bytes {
                    return Err(BrainError::Integrity(
                        "governed_composition_pointer_artifact_collision".into(),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                write_new_private(&self.root, &pointer_path, &pointer_bytes)?;
            }
            Err(error) => return Err(error.into()),
        }
        let receipt =
            load_verified_governed_composition_receipt(&self.root, &receipt_path, &receipt_sha256)?;
        let verified_event = ledger::find_v2_event_by_payload_string(
            &self.root,
            "governed_composition_receipt",
            "receipt_sha256",
            &receipt_sha256,
        )?
        .ok_or_else(|| {
            BrainError::Integrity("governed_composition_receipt_ledger_missing".into())
        })?;
        if verified_event.event_hash != event.event_hash {
            return Err(BrainError::Integrity(
                "governed_composition_post_pointer_ledger_changed".into(),
            ));
        }
        Ok(RecordedGovernedComposition {
            receipt_path: receipt_path.to_string_lossy().into_owned(),
            receipt_sha256,
            ledger_event_hash: event.event_hash,
            receipt,
        })
    }

    fn cognitive_route_activation(
        &self,
        route: &FieldRoutingDecision,
    ) -> BrainResult<BTreeMap<String, f64>> {
        if route.schema != "cerebro.tidex.cognitive_field_routing/v1"
            || route.field_ids.is_empty()
            || route.field_ids.len() != route.coefficients.len()
            || route.selected_field_ids.is_empty()
            || route
                .coefficients
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(BrainError::Invalid(
                "cognitive_route_contract_invalid".into(),
            ));
        }
        let unique_ids = route.field_ids.iter().collect::<BTreeSet<_>>();
        let selected = route.selected_field_ids.iter().collect::<BTreeSet<_>>();
        if unique_ids.len() != route.field_ids.len()
            || selected.len() != route.selected_field_ids.len()
            || selected.iter().any(|id| !unique_ids.contains(id))
        {
            return Err(BrainError::Invalid(
                "cognitive_route_identity_invalid".into(),
            ));
        }
        let activation = route
            .field_ids
            .iter()
            .zip(&route.coefficients)
            .filter(|(_, coefficient)| **coefficient > f64::EPSILON)
            .map(|(id, coefficient)| (id.clone(), *coefficient))
            .collect::<BTreeMap<_, _>>();
        let activated_ids = activation.keys().collect::<BTreeSet<_>>();
        if activation.is_empty()
            || selected.len() != activated_ids.len()
            || selected.iter().any(|id| !activated_ids.contains(id))
        {
            return Err(BrainError::Invalid(
                "cognitive_route_selection_mismatch".into(),
            ));
        }
        Ok(activation)
    }

    /// Record an executable Dynamic Cognitive Field route through the shared
    /// causal-trust/Protected-Cortex authority. A route never exposes a raw
    /// parameter delta to callers.
    fn compose_cognitive_route_and_record(
        &self,
        route: &FieldRoutingDecision,
        source_observation_sha256: &str,
    ) -> BrainResult<RecordedGovernedComposition> {
        self.require_current_observation_digest(source_observation_sha256)?;
        let activation = self.cognitive_route_activation(route)?;
        self.compose_and_record(&activation, source_observation_sha256)
    }

    fn learned_controller_activation(
        &self,
        runtime: &RuntimeLearnedController,
        state: &[f64],
        observation: &[f64],
    ) -> BrainResult<BTreeMap<String, f64>> {
        self.require_current_certification()?;
        let bank = self.load_bank()?;
        let bank_ids = bank
            .fields
            .iter()
            .map(|field| field.skill_id.clone())
            .collect::<Vec<_>>();
        if runtime.schema != "cerebro.tidex.runtime_learned_controller/v1"
            || runtime.field_ids != bank_ids
            || runtime.controller.coefficient_dim != bank.fields.len()
        {
            return Err(BrainError::Integrity(
                "runtime_learned_controller_identity_mismatch".into(),
            ));
        }
        let decision = runtime.controller.decide(state, observation)?;
        let activation = runtime
            .field_ids
            .iter()
            .zip(&decision.coefficients)
            .filter(|(_, coefficient)| coefficient.abs() > f64::EPSILON)
            .map(|(id, coefficient)| (id.clone(), *coefficient))
            .collect::<BTreeMap<_, _>>();
        if activation.is_empty() {
            return Err(BrainError::Invalid(
                "runtime_learned_controller_zero_activation".into(),
            ));
        }
        Ok(activation)
    }

    /// Execute the current receipt-verified LearnedController through its
    /// canonical, durable execution cycle.  The caller can supply only a
    /// finite state vector and a semantic reference to an active observation;
    /// the controller, functional response, activation, causal trust,
    /// protection result, composition delta, and execution receipt are all
    /// resolved and sealed by TIDE-X.
    pub fn compose_current_learned_controller_and_record(
        &self,
        invocation: &ControllerInvocation,
    ) -> BrainResult<RecordedControllerExecution> {
        self.require_canonical_runtime_config()?;
        invocation.validate()?;
        let before =
            load_persisted_runtime_learned_controller(&self.root, invocation.session_id.as_str())?;
        let source = self.load_current_observation_by_semantic_sha256(
            &invocation.promoted_observation_semantic_sha256,
        )?;
        let activation = self.learned_controller_activation(
            &before.receipt.runtime_controller,
            &invocation.state_before,
            &source.functional_response,
        )?;
        let composition = self.compose_and_record(
            &activation,
            &invocation.promoted_observation_semantic_sha256,
        )?;
        let after =
            load_persisted_runtime_learned_controller(&self.root, invocation.session_id.as_str())?;
        if before.receipt_sha256 != after.receipt_sha256 {
            return Err(BrainError::Integrity(
                "controller_execution_controller_receipt_changed_during_composition".into(),
            ));
        }
        let governed = load_verified_governed_composition_receipt(
            &self.root,
            Path::new(&composition.receipt_path),
            &composition.receipt_sha256,
        )?;
        if governed != composition.receipt
            || governed.source_observation_sha256 != invocation.promoted_observation_semantic_sha256
        {
            return Err(BrainError::Integrity(
                "controller_execution_governed_composition_binding_invalid".into(),
            ));
        }
        let recorded = persist_controller_execution(
            &self.root,
            ControllerExecutionReceipt {
                schema: "cerebro.tidex.controller_execution_receipt/v1".into(),
                session_id: invocation.session_id.clone(),
                invocation: invocation.clone(),
                // This is a semantic digest of the strict invocation contract,
                // not a hash self-asserted by an arbitrary CLI file encoding.
                invocation_sha256: Sha256Digest::parse(digest_json(invocation)?)?,
                state_before_sha256: Sha256Digest::parse(digest_json(&invocation.state_before)?)?,
                controller_receipt_sha256: Sha256Digest::parse(after.receipt_sha256)?,
                promoted_observation_semantic_sha256: invocation
                    .promoted_observation_semantic_sha256
                    .clone(),
                governed_composition_receipt_sha256: Sha256Digest::parse(
                    composition.receipt_sha256,
                )?,
            },
        )?;
        verify_controller_execution_receipt(&self.root, &recorded)?;
        Ok(recorded)
    }

    fn cognitive_route_from_drive(
        &self,
        initial: &[f64],
        drive: &CognitiveFieldDrive,
        top_k: usize,
        minimum_activation: f64,
    ) -> BrainResult<(CognitiveFieldState, FieldRoutingDecision)> {
        self.require_current_certification()?;
        let bank = self.load_bank()?;
        let (_, interaction, _, causal) = self.load_verified_runtime_evidence(&bank)?;
        let model = DynamicCognitiveField::build(
            &bank.fields,
            &interaction,
            &causal,
            CognitiveFieldConfig::default(),
        )?;
        let state = model.evolve(initial, drive)?;
        if !state.converged {
            return Err(BrainError::Numerical(
                "runtime_cognitive_field_not_converged".into(),
            ));
        }
        let route = model.route_top_k(&state, top_k, minimum_activation)?;
        Ok((state, route))
    }

    /// Evolve the Dynamic Cognitive Field and make its chosen route executable
    /// only by creating a governed composition receipt. This is the canonical
    /// runtime integration for the Cognitive Field, not a bench-only side
    /// channel.
    pub fn compose_cognitive_drive_and_record(
        &self,
        initial: &[f64],
        drive: &CognitiveFieldDrive,
        top_k: usize,
        minimum_activation: f64,
        source_observation_sha256: &str,
    ) -> BrainResult<RecordedGovernedCognitiveComposition> {
        self.require_current_observation_digest(source_observation_sha256)?;
        let (state, route) =
            self.cognitive_route_from_drive(initial, drive, top_k, minimum_activation)?;
        let composition =
            self.compose_cognitive_route_and_record(&route, source_observation_sha256)?;
        Ok(RecordedGovernedCognitiveComposition {
            state,
            route,
            composition,
        })
    }

    pub fn search_by_function(
        &self,
        query: &[f64],
        limit: usize,
    ) -> BrainResult<Vec<(String, f64)>> {
        let bank = self.load_bank()?;
        let mut scored = bank
            .fields
            .iter()
            .filter(|f| f.functional_signature.len() == query.len() && !query.is_empty())
            .map(|f| Ok((f.skill_id.clone(), cosine(&f.functional_signature, query)?)))
            .collect::<BrainResult<Vec<_>>>()?;
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(limit);
        Ok(scored)
    }

    pub fn status(&self) -> BrainResult<serde_json::Value> {
        let ledger = ledger::verify(&self.root)?;
        let bank = self.load_bank()?;
        let health = self.runtime_integrity_health()?;
        Ok(json!({
            "schema":"cerebro.tidex.status/v2",
            "private_root":self.root,
            "ledger_events":ledger.events,
            "ledger_head":ledger.head,
            "skill_generation":bank.generation,
            "skill_count":bank.fields.len(),
            "external_integration":false,
            "network_runtime":false,
            "integrity":health,
        }))
    }

    pub fn skill_subspace_overlap(fields: &[SkillField], truth: &[Vec<f64>]) -> BrainResult<f64> {
        if fields.is_empty() || truth.is_empty() {
            return Ok(0.0);
        }
        // Latent subspaces are identifiable before their internal coordinate
        // system is. Measure how much of each known direction lies in the
        // recovered orthonormal span, rather than demanding an arbitrary PCA
        // axis to equal an arbitrary ground-truth axis.
        let total = truth.iter().try_fold(0.0, |sum, t| {
            let tn = norm(t)?.max(1e-15);
            let projected = fields.iter().try_fold(0.0, |energy, field| {
                Ok::<f64, BrainError>(energy + crate::linalg::dot(t, &field.direction)?.powi(2))
            })?;
            Ok::<f64, BrainError>(sum + projected.sqrt() / tn)
        })?;
        Ok(total / truth.len() as f64)
    }
    pub fn bank_energy(&self) -> BrainResult<f64> {
        self.load_bank()?.fields.iter().try_fold(0.0, |sum, field| {
            Ok::<f64, BrainError>(sum + norm(&field.direction)?.powi(2))
        })
    }
}
