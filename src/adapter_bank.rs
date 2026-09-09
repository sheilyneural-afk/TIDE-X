//! Modular, content-addressed receiver-adapter bank.
//!
//! Registered manifests are immutable candidates. Dynamic capability/model
//! indexes are regenerated from the primary table and authenticated on every
//! load. An immutable, hash-linked revision journal is authoritative while
//! `head.json` is only a validated cache; revocation is sticky,
//! composition dependencies are revoked transitively,
//! and rollback always publishes a new forward revision.

use crate::artifact::{
    derive_content_addressed_dvec_combination, verify_dvec_reference_under_root,
    ArtifactWriteAuthority, DeltaArtifactRef,
};
use crate::authority::{
    existing_regular_file_if_present, install_private_immutable_file, open_existing_private_file,
    replace_private_file_atomic, root_relative_path, stage_private_file,
    with_private_authority_lock, write_or_verify_immutable, PrivateFileReference,
};
use crate::block_tomography::{parameter_layout_digest, ParameterLayoutAuthority};
use crate::capability_bundle::{
    authenticate_capability_bundle, CapabilityBundle, CapabilityBundleStatus,
};
use crate::contracts::PromotionDecision;
use crate::digest::{ParameterLayoutDigest, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::{ArchitectureId, CapabilityId, LineageId, ModelId, PatchId};
use crate::model_adaptation::{
    authenticate_live_receiver_model_profile, authenticate_receiver_model_profile,
    validate_lora_axis_against_profile, ReceiverAdaptationLayoutBinding,
};
use crate::portfolio_governance::{
    authenticate_adapter_promotion_witnesses, authenticate_canary_state,
    authenticate_candidate_gate_decision, authenticate_petfc_assessment, CanaryState,
    CandidateGateDecision, PetfcAssessment, VariantId,
};
use crate::security::verify_internal_private_root;
use crate::weight_actuator::{
    authenticate_lora_adapter_axis_receipt, authenticate_weight_materialization_receipt,
    import_peft_lora_as_dense_axis, materialize_dense_delta_checkpoint, LoraAdapterAxisInput,
    LoraAdapterAxisReceipt, WeightMaterializationReceipt, LORA_ADAPTER_AXIS_RECEIPT_SCHEMA,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub const ADAPTER_IMPORT_REQUEST_SCHEMA: &str = "cerebro.tidex.adapter_import_request/v1";
pub const ADAPTER_COMPOSITION_REQUEST_SCHEMA: &str = "cerebro.tidex.adapter_composition_request/v1";
pub const ADAPTER_ACTIVATION_REQUEST_SCHEMA: &str = "cerebro.tidex.adapter_activation_request/v1";
pub const ADAPTER_REVOCATION_REQUEST_SCHEMA: &str = "cerebro.tidex.adapter_revocation_request/v1";
pub const ADAPTER_ROLLBACK_REQUEST_SCHEMA: &str = "cerebro.tidex.adapter_rollback_request/v1";
pub const ADAPTER_BANK_QUERY_SCHEMA: &str = "cerebro.tidex.adapter_bank_query/v1";
pub const ADAPTER_BANK_LOOKUP_SCHEMA: &str = "cerebro.tidex.adapter_bank_lookup/v1";
pub const ADAPTER_RESOLUTION_REQUEST_SCHEMA: &str = "cerebro.tidex.adapter_resolution_request/v1";
pub const ADAPTER_EXECUTION_RESOLUTION_SCHEMA: &str =
    "cerebro.tidex.adapter_execution_resolution/v1";
pub const ADAPTER_MANIFEST_SCHEMA: &str = "cerebro.tidex.adapter_manifest/v1";
pub const ADAPTER_INDEX_SCHEMA: &str = "cerebro.tidex.adapter_index/v1";
pub const ADAPTER_BANK_HEAD_SCHEMA: &str = "cerebro.tidex.adapter_bank_head/v1";
pub const ADAPTER_BANK_REVISION_COMMIT_SCHEMA: &str =
    "cerebro.tidex.adapter_bank_revision_commit/v1";
pub const ADAPTER_BANK_COMMIT_SCHEMA: &str = "cerebro.tidex.adapter_bank_commit/v1";
pub const ADAPTER_BANK_REPORT_SCHEMA: &str = "cerebro.tidex.adapter_bank_report/v1";
pub const ADAPTER_PROMOTION_AUTHORIZATION_SCHEMA: &str =
    "cerebro.tidex.adapter_promotion_authorization/v1";
pub const ADAPTER_GOVERNED_PROMOTION_REQUEST_SCHEMA: &str =
    "cerebro.tidex.adapter_governed_promotion_request/v1";
pub const ADAPTER_CANDIDATE_MATERIALIZATION_REQUEST_SCHEMA: &str =
    "cerebro.tidex.adapter_candidate_materialization_request/v1";
pub const ADAPTER_CANDIDATE_MATERIALIZATION_SCHEMA: &str =
    "cerebro.tidex.adapter_candidate_materialization/v1";
const IMPORT_RECEIPT_SCHEMA: &str = "cerebro.tidex.adapter_import_provenance/v1";
const EXACT_COMPOSITION_ARITHMETIC: &str = "ordered_f32_axes_mul_f64_accumulate_f64_round_f32/v1";
const MAX_BANK_JSON_BYTES: u64 = 128 * 1024 * 1024;
const MAX_COMPOSITION_TERMS: usize = 64;
const MAX_HISTORY_REVISIONS: usize = 100_000;
const MAX_REASON_BYTES: usize = 4 * 1024;
const MAX_MANIFEST_DEPTH: usize = 256;

fn invalid(code: impl Into<String>) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: impl Into<String>) -> BrainError {
    BrainError::Integrity(code.into())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct AdapterVersion {
    pub adapter_id: PatchId,
    pub generation: u64,
}

impl AdapterVersion {
    fn validate(&self) -> BrainResult<()> {
        if self.generation == 0 {
            return Err(invalid("adapter_generation_zero"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ReceiverModelKey {
    pub model_id: ModelId,
    pub model_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterBankExpectation {
    pub revision: u64,
    #[serde(default)]
    pub index_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterImportRequest {
    pub schema: String,
    pub adapter: AdapterVersion,
    pub capability_id: CapabilityId,
    pub lineage_id: LineageId,
    #[serde(default)]
    pub capability_bundle: Option<PrivateFileReference>,
    pub receiver_profile: PrivateFileReference,
    pub lora: LoraAdapterAxisInput,
    pub expected: AdapterBankExpectation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterCompositionInputTerm {
    pub adapter: AdapterVersion,
    pub coefficient: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterCompositionRequest {
    pub schema: String,
    pub adapter: AdapterVersion,
    pub capability_id: CapabilityId,
    pub lineage_id: LineageId,
    #[serde(default)]
    pub capability_bundle: Option<PrivateFileReference>,
    pub terms: Vec<AdapterCompositionInputTerm>,
    pub expected: AdapterBankExpectation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterActivationRequest {
    pub schema: String,
    pub adapter: AdapterVersion,
    pub authorization: PrivateFileReference,
    pub expected: AdapterBankExpectation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterRevocationRequest {
    pub schema: String,
    pub adapter: AdapterVersion,
    pub reason: String,
    #[serde(default)]
    pub evidence: Option<PrivateFileReference>,
    pub expected: AdapterBankExpectation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterRollbackRequest {
    pub schema: String,
    pub receiver_model: ReceiverModelKey,
    pub capability_id: CapabilityId,
    pub target_revision: u64,
    pub reason: String,
    #[serde(default)]
    pub evidence: Option<PrivateFileReference>,
    /// A rollback that restores an adapter must carry a newly issued permit
    /// bound to the current index. Deactivation-only rollback needs no permit.
    #[serde(default)]
    pub authorization: Option<PrivateFileReference>,
    pub expected: AdapterBankExpectation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterBankQuery {
    pub schema: String,
    #[serde(default)]
    pub capability_id: Option<CapabilityId>,
    #[serde(default)]
    pub model_id: Option<ModelId>,
    #[serde(default)]
    pub model_sha256: Option<Sha256Digest>,
    #[serde(default)]
    pub include_revoked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterBankLookup {
    pub schema: String,
    pub adapter_id: PatchId,
    #[serde(default)]
    pub generation: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterResolutionRequest {
    pub schema: String,
    pub receiver_model: ReceiverModelKey,
    pub capability_id: CapabilityId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AdapterManifestSource {
    ImportedPeftLora {
        import_provenance: PrivateFileReference,
        adapter_model: RetainedAdapterBlob,
        adapter_config: RetainedAdapterBlob,
    },
    ExactDenseComposition {
        arithmetic: String,
        terms: Vec<AdapterCompositionTerm>,
        rank_truncation_used: bool,
        factor_refactorization_used: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterCompositionTerm {
    pub adapter: AdapterVersion,
    pub manifest: PrivateFileReference,
    pub coefficient: f64,
    pub coefficient_bits_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RetainedAdapterBlob {
    pub artifact: PrivateFileReference,
    pub byte_len: u64,
    pub media_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterManifest {
    pub schema: String,
    pub adapter: AdapterVersion,
    pub capability_id: CapabilityId,
    pub lineage_id: LineageId,
    #[serde(default)]
    pub capability_bundle: Option<PrivateFileReference>,
    pub receiver_model: ReceiverModelKey,
    pub architecture_id: ArchitectureId,
    pub receiver_profile: PrivateFileReference,
    pub parameter_topology_sha256: Sha256Digest,
    pub adaptation_abi_sha256: Sha256Digest,
    pub parameter_layout: ReceiverAdaptationLayoutBinding,
    pub dense_delta: DeltaArtifactRef,
    pub source: AdapterManifestSource,
    pub candidate_only: bool,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ImportedLoraProvenance {
    schema: String,
    receipt: LoraAdapterAxisReceipt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterIndexEntry {
    pub adapter: AdapterVersion,
    pub capability_id: CapabilityId,
    pub lineage_id: LineageId,
    pub receiver_model: ReceiverModelKey,
    pub parameter_layout_sha256: ParameterLayoutDigest,
    pub manifest: PrivateFileReference,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityAdapterBucket {
    pub capability_id: CapabilityId,
    pub adapters: Vec<AdapterVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelAdapterBucket {
    pub receiver_model: ReceiverModelKey,
    pub adapters: Vec<AdapterVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct AdapterActivationSlot {
    pub receiver_model: ReceiverModelKey,
    pub capability_id: CapabilityId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterExecutionBinding {
    pub base_model_sha256: Sha256Digest,
    pub adapter_delta_sha256: Sha256Digest,
    pub activation_epoch: u64,
    pub adaptation_abi_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ActiveAdapterSelection {
    pub slot: AdapterActivationSlot,
    pub adapter: AdapterVersion,
    pub manifest: PrivateFileReference,
    pub authorization: PrivateFileReference,
    pub execution_binding: AdapterExecutionBinding,
    pub activated_at_revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterRevocation {
    pub adapter: AdapterVersion,
    pub manifest_sha256: Sha256Digest,
    pub revoked_at_revision: u64,
    pub root_cause: AdapterVersion,
    pub reason: String,
    #[serde(default)]
    pub evidence: Option<PrivateFileReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AdapterBankOperation {
    Empty,
    Register {
        manifest: PrivateFileReference,
    },
    Compose {
        manifest: PrivateFileReference,
    },
    Activate {
        adapter: AdapterVersion,
        authorization: PrivateFileReference,
    },
    Revoke {
        root: AdapterVersion,
        affected: Vec<AdapterVersion>,
    },
    Rollback {
        slot: AdapterActivationSlot,
        target_revision: u64,
        from: Option<AdapterVersion>,
        to: Option<AdapterVersion>,
        #[serde(default)]
        authorization: Option<PrivateFileReference>,
        reason: String,
        #[serde(default)]
        evidence: Option<PrivateFileReference>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterIndexSnapshot {
    pub schema: String,
    pub revision: u64,
    #[serde(default)]
    pub parent: Option<PrivateFileReference>,
    pub operation: AdapterBankOperation,
    pub entries: Vec<AdapterIndexEntry>,
    pub by_capability: Vec<CapabilityAdapterBucket>,
    pub by_model: Vec<ModelAdapterBucket>,
    pub active: Vec<ActiveAdapterSelection>,
    pub revocations: Vec<AdapterRevocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AdapterBankHead {
    schema: String,
    revision: u64,
    index: PrivateFileReference,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AdapterBankRevisionCommit {
    schema: String,
    revision: u64,
    index: PrivateFileReference,
    #[serde(default)]
    previous_commit: Option<PrivateFileReference>,
    #[serde(default)]
    previous_index: Option<PrivateFileReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterPromotionEvidence {
    pub independent_execution: PrivateFileReference,
    pub preservation_assessment: PrivateFileReference,
    pub negative_controls: PrivateFileReference,
    pub materialization_receipt: PrivateFileReference,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterCandidateMaterializationRequest {
    pub schema: String,
    pub adapter: AdapterVersion,
    pub output_path: PathBuf,
    pub expected: AdapterBankExpectation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterGovernedPromotionRequest {
    pub schema: String,
    pub materialization: PrivateFileReference,
    pub candidate_gate: PrivateFileReference,
    pub petfc_assessment: PrivateFileReference,
    pub canary_state: PrivateFileReference,
    pub expected: AdapterBankExpectation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterCandidateMaterialization {
    pub schema: String,
    pub adapter: AdapterVersion,
    pub manifest: PrivateFileReference,
    pub capability_id: CapabilityId,
    #[serde(default)]
    pub capability_bundle: Option<PrivateFileReference>,
    pub receiver_profile: PrivateFileReference,
    pub observed_revision: u64,
    pub observed_index: PrivateFileReference,
    pub parameter_layout_sha256: ParameterLayoutDigest,
    pub dense_delta_sha256: Sha256Digest,
    pub output_path: PathBuf,
    pub receipt: WeightMaterializationReceipt,
    pub candidate_only: bool,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterPromotionAuthorization {
    pub schema: String,
    /// True only for the production issuer backed by authenticated governance.
    #[serde(default)]
    pub governed: bool,
    pub adapter: AdapterVersion,
    pub manifest: PrivateFileReference,
    pub capability_id: CapabilityId,
    pub receiver_model: ReceiverModelKey,
    pub receiver_profile: PrivateFileReference,
    pub parameter_layout_sha256: ParameterLayoutDigest,
    pub dense_delta_sha256: Sha256Digest,
    pub issued_against_index: PrivateFileReference,
    pub evidence: AdapterPromotionEvidence,
    pub decision: PromotionDecision,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterBankCommit {
    pub schema: String,
    pub revision: u64,
    pub index: PrivateFileReference,
    pub operation: AdapterBankOperation,
    #[serde(default)]
    pub manifest: Option<PrivateFileReference>,
    pub idempotent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdapterLifecycleState {
    RegisteredCandidate,
    Active,
    Revoked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterStatus {
    pub adapter: AdapterVersion,
    pub capability_id: CapabilityId,
    pub lineage_id: LineageId,
    pub receiver_model: ReceiverModelKey,
    pub manifest: PrivateFileReference,
    pub state: AdapterLifecycleState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdapterBankReport {
    pub schema: String,
    pub revision: u64,
    #[serde(default)]
    pub index: Option<PrivateFileReference>,
    pub adapters: Vec<AdapterStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterExecutionResolution {
    pub schema: String,
    pub bank_revision: u64,
    pub index: PrivateFileReference,
    pub slot: AdapterActivationSlot,
    pub adapter: AdapterVersion,
    pub manifest: PrivateFileReference,
    pub authorization: PrivateFileReference,
    pub execution_binding: AdapterExecutionBinding,
    #[serde(default)]
    pub materialized_candidate: Option<AdapterCandidateMaterialization>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterBankHistoryStatus {
    pub revision: u64,
    #[serde(default)]
    pub index: Option<PrivateFileReference>,
    pub verified_revision_count: usize,
    pub registered_adapter_count: usize,
    pub active_adapter_count: usize,
    pub revoked_adapter_count: usize,
}

#[derive(Debug, Clone)]
pub struct AdapterBank {
    root: PathBuf,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct VerifiedAdapterPromotion {
    pub manifest: PrivateFileReference,
    pub evidence: AdapterPromotionEvidence,
    pub decision: PromotionDecision,
    pub expected: AdapterBankExpectation,
}

fn validate_reason(reason: &str) -> BrainResult<()> {
    if reason.is_empty()
        || reason.len() > MAX_REASON_BYTES
        || reason != reason.trim()
        || reason.chars().any(char::is_control)
    {
        return Err(invalid("adapter_lifecycle_reason_invalid"));
    }
    Ok(())
}

fn coefficient_bits_hex(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn validate_coefficient(value: f64) -> BrainResult<()> {
    if !value.is_finite() || value == 0.0 {
        return Err(invalid("adapter_composition_coefficient_invalid"));
    }
    Ok(())
}

impl AdapterBank {
    pub fn open(root: impl AsRef<Path>) -> BrainResult<Self> {
        Ok(Self {
            root: verify_internal_private_root(root.as_ref())?,
        })
    }

    fn bank_root(&self) -> PathBuf {
        self.root.join("state/adapter_bank")
    }

    fn lock_path(&self) -> PathBuf {
        self.bank_root().join("authority.lock")
    }

    fn head_path(&self) -> PathBuf {
        self.bank_root().join("head.json")
    }

    fn revision_commit_root(&self) -> PathBuf {
        self.bank_root().join("commits/by-revision")
    }

    fn revision_commit_path(&self, revision: u64) -> PathBuf {
        self.revision_commit_root()
            .join(format!("{revision:020}.json"))
    }

    fn manifest_path(&self, digest: &Sha256Digest) -> PathBuf {
        self.bank_root()
            .join("manifests/by-sha")
            .join(format!("{digest}.json"))
    }

    fn import_path(&self, digest: &Sha256Digest) -> PathBuf {
        self.bank_root()
            .join("imports/by-sha")
            .join(format!("{digest}.json"))
    }

    fn source_blob_path(&self, digest: &Sha256Digest, extension: &str) -> PathBuf {
        self.bank_root()
            .join("source-blobs/by-sha")
            .join(format!("{digest}.{extension}"))
    }

    fn index_path(&self, digest: &Sha256Digest) -> PathBuf {
        self.bank_root()
            .join("indexes/by-sha")
            .join(format!("{digest}.json"))
    }

    fn authorization_path(&self, digest: &Sha256Digest) -> PathBuf {
        self.bank_root()
            .join("promotion-authorizations/by-sha")
            .join(format!("{digest}.json"))
    }

    fn materialization_path(&self, digest: &Sha256Digest) -> PathBuf {
        self.bank_root()
            .join("materializations/by-sha")
            .join(format!("{digest}.json"))
    }

    fn authenticate_bound_capability_bundle(
        &self,
        reference: Option<&PrivateFileReference>,
        expected_capability: &CapabilityId,
    ) -> BrainResult<Option<CapabilityBundle>> {
        let Some(reference) = reference else {
            return Ok(None);
        };
        let bundle = authenticate_capability_bundle(&self.root, reference)?;
        if bundle.capability_id() != expected_capability {
            return Err(integrity("adapter_capability_bundle_binding_mismatch"));
        }
        Ok(Some(bundle))
    }

    fn require_promotable_capability_bundle(
        &self,
        manifest: &AdapterManifest,
    ) -> BrainResult<PrivateFileReference> {
        let reference = manifest
            .capability_bundle
            .as_ref()
            .ok_or_else(|| invalid("adapter_promotion_capability_bundle_missing"))?;
        let bundle = self
            .authenticate_bound_capability_bundle(Some(reference), &manifest.capability_id)?
            .ok_or_else(|| integrity("adapter_promotion_capability_bundle_missing"))?;
        if bundle.status() != CapabilityBundleStatus::RepresentationClosed {
            return Err(invalid(
                "adapter_promotion_capability_bundle_not_representation_closed",
            ));
        }
        Ok(reference.clone())
    }

    fn confined_materialization_output(&self, raw: &Path) -> BrainResult<PathBuf> {
        let relative = root_relative_path(&self.root, raw)?;
        if relative.file_name().is_none() {
            return Err(invalid("adapter_materialization_output_name_missing"));
        }
        let parent_relative = relative
            .parent()
            .ok_or_else(|| invalid("adapter_materialization_output_parent_missing"))?;
        let expected_parent = self.root.join(parent_relative);
        let canonical_parent = fs::canonicalize(&expected_parent).map_err(|error| {
            integrity(format!(
                "adapter_materialization_output_parent_unreadable:{error}"
            ))
        })?;
        let metadata = fs::symlink_metadata(&expected_parent)?;
        if canonical_parent != expected_parent
            || metadata.file_type().is_symlink()
            || !metadata.file_type().is_dir()
        {
            return Err(integrity(
                "adapter_materialization_output_parent_not_canonical",
            ));
        }
        Ok(self.root.join(relative))
    }

    fn canonical_delta_path(&self, digest: &Sha256Digest) -> PathBuf {
        self.root
            .join("artifacts/deltas/by-sha")
            .join(format!("{digest}.dvec"))
    }

    fn read_mutable_bounded(&self, path: &Path) -> BrainResult<Vec<u8>> {
        let file = open_existing_private_file(&self.root, path)?;
        let mut bytes = Vec::new();
        file.take(MAX_BANK_JSON_BYTES.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_BANK_JSON_BYTES {
            return Err(invalid("adapter_bank_record_too_large"));
        }
        Ok(bytes)
    }

    fn empty_snapshot() -> AdapterIndexSnapshot {
        AdapterIndexSnapshot {
            schema: ADAPTER_INDEX_SCHEMA.to_string(),
            revision: 0,
            parent: None,
            operation: AdapterBankOperation::Empty,
            entries: Vec::new(),
            by_capability: Vec::new(),
            by_model: Vec::new(),
            active: Vec::new(),
            revocations: Vec::new(),
        }
    }

    fn journal_revisions(&self) -> BrainResult<Vec<u64>> {
        let root = self.revision_commit_root();
        let metadata = match fs::symlink_metadata(&root) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
            return Err(integrity("adapter_bank_revision_journal_not_directory"));
        }

        let mut revisions = Vec::new();
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| integrity("adapter_bank_revision_commit_name_invalid"))?;
            if name.starts_with('.') && name.ends_with(".immutable.tmp") {
                continue;
            }
            if file_type.is_symlink() || !file_type.is_file() {
                return Err(integrity("adapter_bank_revision_commit_type_invalid"));
            }
            let stem = name
                .strip_suffix(".json")
                .ok_or_else(|| integrity("adapter_bank_revision_commit_name_invalid"))?;
            if stem.len() != 20 || !stem.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(integrity("adapter_bank_revision_commit_name_invalid"));
            }
            let revision = stem
                .parse::<u64>()
                .map_err(|_| integrity("adapter_bank_revision_commit_name_invalid"))?;
            if revision == 0 || revision > MAX_HISTORY_REVISIONS as u64 {
                return Err(integrity("adapter_bank_revision_commit_out_of_range"));
            }
            revisions.push(revision);
        }
        revisions.sort_unstable();
        if revisions.len() > MAX_HISTORY_REVISIONS {
            return Err(integrity("adapter_bank_revision_journal_too_long"));
        }
        for (offset, revision) in revisions.iter().enumerate() {
            let expected = u64::try_from(offset + 1)
                .map_err(|_| integrity("adapter_bank_revision_journal_too_long"))?;
            if *revision != expected {
                return Err(integrity("adapter_bank_revision_journal_gap"));
            }
        }
        Ok(revisions)
    }

    fn read_revision_commit(
        &self,
        revision: u64,
    ) -> BrainResult<(AdapterBankRevisionCommit, PrivateFileReference)> {
        let path = self.revision_commit_path(revision);
        let bytes = self.read_mutable_bounded(&path)?;
        let commit: AdapterBankRevisionCommit = serde_json::from_slice(&bytes)
            .map_err(|_| integrity("adapter_bank_revision_commit_invalid"))?;
        if commit.schema != ADAPTER_BANK_REVISION_COMMIT_SCHEMA
            || commit.revision != revision
            || revision == 0
            || revision > MAX_HISTORY_REVISIONS as u64
            || commit.index.path != self.index_path(&commit.index.sha256)
            || serde_json::to_vec(&commit)? != bytes
        {
            return Err(integrity("adapter_bank_revision_commit_invalid"));
        }
        if revision == 1 {
            if commit.previous_commit.is_some() || commit.previous_index.is_some() {
                return Err(integrity(
                    "adapter_bank_revision_commit_genesis_link_invalid",
                ));
            }
        } else {
            let previous_commit = commit
                .previous_commit
                .as_ref()
                .ok_or_else(|| integrity("adapter_bank_revision_commit_link_missing"))?;
            let previous_index = commit
                .previous_index
                .as_ref()
                .ok_or_else(|| integrity("adapter_bank_revision_index_link_missing"))?;
            if previous_commit.path != self.revision_commit_path(revision - 1)
                || previous_index.path != self.index_path(&previous_index.sha256)
            {
                return Err(integrity("adapter_bank_revision_commit_link_invalid"));
            }
        }
        commit.index.verify(&self.root)?;
        let reference = PrivateFileReference::new(path, Sha256Digest::digest_bytes(&bytes));
        Ok((commit, reference))
    }

    fn authenticate_revision_journal(
        &self,
    ) -> BrainResult<Vec<(AdapterBankRevisionCommit, PrivateFileReference)>> {
        let revisions = self.journal_revisions()?;
        let mut commits: Vec<(AdapterBankRevisionCommit, PrivateFileReference)> =
            Vec::with_capacity(revisions.len());
        for revision in revisions {
            let (commit, reference) = self.read_revision_commit(revision)?;
            if let Some((previous, previous_reference)) = commits.last() {
                if commit.previous_commit.as_ref() != Some(previous_reference)
                    || commit.previous_index.as_ref() != Some(&previous.index)
                {
                    return Err(integrity("adapter_bank_revision_chain_invalid"));
                }
            }
            commits.push((commit, reference));
        }
        Ok(commits)
    }

    fn authenticate_revision_tip(
        &self,
    ) -> BrainResult<Option<(AdapterBankRevisionCommit, PrivateFileReference)>> {
        let revisions = self.journal_revisions()?;
        let Some(revision) = revisions.last().copied() else {
            return Ok(None);
        };
        let (tip, tip_reference) = self.read_revision_commit(revision)?;
        if revision > 1 {
            let (previous, previous_reference) = self.read_revision_commit(revision - 1)?;
            if tip.previous_commit.as_ref() != Some(&previous_reference)
                || tip.previous_index.as_ref() != Some(&previous.index)
            {
                return Err(integrity("adapter_bank_revision_tip_chain_invalid"));
            }
        }
        Ok(Some((tip, tip_reference)))
    }

    fn authenticate_head_cache(&self, tip: &AdapterBankRevisionCommit) -> BrainResult<()> {
        if existing_regular_file_if_present(&self.root, &self.head_path())?.is_none() {
            return Ok(());
        }
        let bytes = self.read_mutable_bounded(&self.head_path())?;
        let head: AdapterBankHead =
            serde_json::from_slice(&bytes).map_err(|_| integrity("adapter_bank_head_invalid"))?;
        if head.schema != ADAPTER_BANK_HEAD_SCHEMA
            || head.revision == 0
            || head.revision > tip.revision
            || serde_json::to_vec(&head)? != bytes
        {
            return Err(integrity("adapter_bank_head_invalid"));
        }
        if head.revision == tip.revision {
            if head.index != tip.index {
                return Err(integrity("adapter_bank_head_commit_mismatch"));
            }
        } else {
            let (commit, _) = self.read_revision_commit(head.revision)?;
            if head.index != commit.index {
                return Err(integrity("adapter_bank_head_commit_mismatch"));
            }
        }
        Ok(())
    }

    fn current(&self) -> BrainResult<(AdapterIndexSnapshot, Option<PrivateFileReference>)> {
        let Some((tip, _)) = self.authenticate_revision_tip()? else {
            if existing_regular_file_if_present(&self.root, &self.head_path())?.is_some() {
                return Err(integrity("adapter_bank_head_without_revision_journal"));
            }
            return Ok((Self::empty_snapshot(), None));
        };
        self.authenticate_head_cache(&tip)?;
        let snapshot = self.authenticate_snapshot(&tip.index)?;
        if snapshot.revision != tip.revision || snapshot.parent != tip.previous_index {
            return Err(integrity("adapter_bank_revision_tip_mismatch"));
        }
        let parent = match tip.previous_index.as_ref() {
            Some(reference) => self.authenticate_snapshot(reference)?,
            None => Self::empty_snapshot(),
        };
        self.validate_snapshot_transition(tip.previous_index.as_ref(), &parent, &snapshot)?;
        Ok((snapshot, Some(tip.index.clone())))
    }

    fn check_expectation(
        expected: &AdapterBankExpectation,
        snapshot: &AdapterIndexSnapshot,
        reference: Option<&PrivateFileReference>,
    ) -> BrainResult<()> {
        let current_digest = reference.map(|reference| reference.sha256.clone());
        if expected.revision != snapshot.revision || expected.index_sha256 != current_digest {
            return Err(integrity("adapter_bank_compare_and_swap_mismatch"));
        }
        Ok(())
    }

    fn retain_external_blob(
        &self,
        source: &Path,
        expected_sha256: &Sha256Digest,
        extension: &str,
        media_type: &str,
    ) -> BrainResult<RetainedAdapterBlob> {
        if !source.is_absolute()
            || source
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(invalid("adapter_source_blob_path_invalid"));
        }
        let source = fs::canonicalize(source)
            .map_err(|error| integrity(format!("adapter_source_blob_unreadable:{error}")))?;
        let metadata = fs::symlink_metadata(&source)?;
        if metadata.file_type().is_symlink()
            || !metadata.file_type().is_file()
            || metadata.len() == 0
        {
            return Err(integrity("adapter_source_blob_not_regular_file"));
        }
        let destination = self.source_blob_path(expected_sha256, extension);
        let (temporary, observed_sha256) =
            stage_private_file(&self.root, &destination, |output| {
                let mut input = File::open(&source)?;
                io::copy(&mut input, output)?;
                Ok(())
            })?;
        if observed_sha256 != *expected_sha256 {
            let cleanup = install_private_immutable_file(
                &self.root,
                &temporary,
                &destination,
                expected_sha256,
            );
            return match cleanup {
                Err(error) => Err(error),
                Ok(_) => Err(integrity("adapter_source_blob_digest_mismatch")),
            };
        }
        if !install_private_immutable_file(&self.root, &temporary, &destination, expected_sha256)? {
            PrivateFileReference::new(destination.clone(), expected_sha256.clone())
                .verify(&self.root)?;
        }
        let artifact = PrivateFileReference::new(destination, expected_sha256.clone());
        let retained = RetainedAdapterBlob {
            artifact,
            byte_len: metadata.len(),
            media_type: media_type.to_string(),
        };
        self.authenticate_retained_blob(&retained, extension, media_type)?;
        Ok(retained)
    }

    fn authenticate_retained_blob(
        &self,
        retained: &RetainedAdapterBlob,
        extension: &str,
        media_type: &str,
    ) -> BrainResult<()> {
        if retained.byte_len == 0
            || retained.media_type != media_type
            || retained.artifact.path != self.source_blob_path(&retained.artifact.sha256, extension)
        {
            return Err(integrity("adapter_source_blob_binding_invalid"));
        }
        retained.artifact.verify(&self.root)?;
        let file = open_existing_private_file(&self.root, &retained.artifact.path)?;
        if file.metadata()?.len() != retained.byte_len {
            return Err(integrity("adapter_source_blob_length_mismatch"));
        }
        Ok(())
    }

    fn persist_import_provenance(
        &self,
        receipt: &LoraAdapterAxisReceipt,
    ) -> BrainResult<PrivateFileReference> {
        let provenance = ImportedLoraProvenance {
            schema: IMPORT_RECEIPT_SCHEMA.to_string(),
            receipt: receipt.clone(),
        };
        let bytes = serde_json::to_vec(&provenance)?;
        let digest = Sha256Digest::digest_bytes(&bytes);
        let path = self.import_path(&digest);
        let written = write_or_verify_immutable(&self.root, &path, &bytes)?;
        if written != digest {
            return Err(integrity("adapter_import_provenance_write_mismatch"));
        }
        let reference = PrivateFileReference::new(path, digest);
        self.authenticate_import_provenance(&reference)?;
        Ok(reference)
    }

    fn authenticate_import_provenance(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<LoraAdapterAxisReceipt> {
        if reference.path != self.import_path(&reference.sha256) {
            return Err(integrity("adapter_import_provenance_path_not_canonical"));
        }
        let bytes = reference.read_verified_bounded(&self.root, MAX_BANK_JSON_BYTES)?;
        let provenance: ImportedLoraProvenance = serde_json::from_slice(&bytes)?;
        if provenance.schema != IMPORT_RECEIPT_SCHEMA
            || provenance.receipt.schema != LORA_ADAPTER_AXIS_RECEIPT_SCHEMA
            || provenance.receipt.target_tensors.is_empty()
            || provenance.receipt.authorizes_promotion
            || provenance.receipt.authorizes_target_update_free_claim
            || provenance.receipt.source_training_semantics_attested
            || serde_json::to_vec(&provenance)? != bytes
        {
            return Err(integrity("adapter_import_provenance_invalid"));
        }
        provenance.receipt.parameter_layout.validate()?;
        verify_dvec_reference_under_root(&self.root, &provenance.receipt.dense_delta)?;
        Ok(provenance.receipt)
    }

    fn persist_manifest(&self, manifest: &AdapterManifest) -> BrainResult<PrivateFileReference> {
        self.validate_manifest_contract(manifest, 0, &mut BTreeSet::new())?;
        let bytes = serde_json::to_vec(manifest)?;
        let digest = Sha256Digest::digest_bytes(&bytes);
        let path = self.manifest_path(&digest);
        let written = write_or_verify_immutable(&self.root, &path, &bytes)?;
        if written != digest {
            return Err(integrity("adapter_manifest_write_mismatch"));
        }
        let reference = PrivateFileReference::new(path, digest);
        let authenticated = self.authenticate_manifest(&reference)?;
        if authenticated != *manifest {
            return Err(integrity("adapter_manifest_roundtrip_mismatch"));
        }
        Ok(reference)
    }

    pub fn authenticate_manifest(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<AdapterManifest> {
        self.authenticate_manifest_inner(reference, 0, &mut BTreeSet::new())
    }

    fn authenticate_manifest_inner(
        &self,
        reference: &PrivateFileReference,
        depth: usize,
        stack: &mut BTreeSet<Sha256Digest>,
    ) -> BrainResult<AdapterManifest> {
        if depth > MAX_MANIFEST_DEPTH {
            return Err(integrity("adapter_manifest_dependency_depth_exceeded"));
        }
        if reference.path != self.manifest_path(&reference.sha256) {
            return Err(integrity("adapter_manifest_path_not_canonical"));
        }
        if !stack.insert(reference.sha256.clone()) {
            return Err(integrity("adapter_manifest_dependency_cycle"));
        }
        let result = (|| {
            let bytes = reference.read_verified_bounded(&self.root, MAX_BANK_JSON_BYTES)?;
            let manifest: AdapterManifest = serde_json::from_slice(&bytes)?;
            if serde_json::to_vec(&manifest)? != bytes {
                return Err(integrity("adapter_manifest_encoding_not_canonical"));
            }
            self.validate_manifest_contract(&manifest, depth, stack)?;
            Ok(manifest)
        })();
        stack.remove(&reference.sha256);
        result
    }

    fn validate_manifest_contract(
        &self,
        manifest: &AdapterManifest,
        depth: usize,
        stack: &mut BTreeSet<Sha256Digest>,
    ) -> BrainResult<()> {
        manifest.adapter.validate()?;
        if manifest.schema != ADAPTER_MANIFEST_SCHEMA
            || manifest.lineage_id.is_unassigned()
            || !manifest.candidate_only
            || manifest.authorizes_promotion
            || manifest.dense_delta.parameter_count
                != manifest.parameter_layout.total_parameter_count
            || manifest.dense_delta.path != self.canonical_delta_path(&manifest.dense_delta.sha256)
        {
            return Err(invalid("adapter_manifest_contract_invalid"));
        }
        self.authenticate_bound_capability_bundle(
            manifest.capability_bundle.as_ref(),
            &manifest.capability_id,
        )?;
        let profile = authenticate_receiver_model_profile(&self.root, &manifest.receiver_profile)?;
        if manifest.receiver_model.model_id != profile.model_id
            || manifest.receiver_model.model_sha256 != profile.checkpoint.sha256
            || manifest.architecture_id != profile.architecture_id
            || manifest.parameter_topology_sha256 != profile.parameter_topology_sha256
            || manifest.adaptation_abi_sha256 != profile.adaptation_abi_sha256
        {
            return Err(integrity("adapter_manifest_receiver_binding_mismatch"));
        }
        let layout = ParameterLayoutAuthority::for_internal_root(&self.root)?
            .authenticate_canonical_binding(
                manifest.parameter_layout.artifact.clone(),
                &manifest.parameter_layout.parameter_layout_sha256,
                manifest.parameter_layout.total_parameter_count,
            )?
            .artifact
            .layout;
        verify_dvec_reference_under_root(&self.root, &manifest.dense_delta)?;
        match &manifest.source {
            AdapterManifestSource::ImportedPeftLora {
                import_provenance,
                adapter_model,
                adapter_config,
            } => {
                let receipt = self.authenticate_import_provenance(import_provenance)?;
                self.authenticate_retained_blob(
                    adapter_model,
                    "safetensors",
                    "application/x-safetensors",
                )?;
                self.authenticate_retained_blob(adapter_config, "json", "application/json")?;
                if receipt.base_model_sha256 != manifest.receiver_model.model_sha256
                    || receipt.parameter_layout != layout
                    || receipt.dense_delta != manifest.dense_delta
                    || receipt.adapter_model_sha256 != adapter_model.artifact.sha256
                    || receipt.adapter_config_sha256 != adapter_config.artifact.sha256
                {
                    return Err(integrity("adapter_manifest_import_binding_mismatch"));
                }
            }
            AdapterManifestSource::ExactDenseComposition {
                arithmetic,
                terms,
                rank_truncation_used,
                factor_refactorization_used,
            } => {
                if arithmetic != EXACT_COMPOSITION_ARITHMETIC
                    || *rank_truncation_used
                    || *factor_refactorization_used
                    || terms.len() < 2
                    || terms.len() > MAX_COMPOSITION_TERMS
                {
                    return Err(invalid("adapter_manifest_composition_invalid"));
                }
                let mut prior_digest = None;
                let mut versions = BTreeSet::new();
                let mut sources = Vec::with_capacity(terms.len());
                for term in terms {
                    term.adapter.validate()?;
                    validate_coefficient(term.coefficient)?;
                    if term.coefficient_bits_hex != coefficient_bits_hex(term.coefficient)
                        || !versions.insert(term.adapter.clone())
                        || prior_digest
                            .as_ref()
                            .is_some_and(|prior| prior >= &term.manifest.sha256)
                    {
                        return Err(invalid("adapter_manifest_composition_term_invalid"));
                    }
                    prior_digest = Some(term.manifest.sha256.clone());
                    let parent =
                        self.authenticate_manifest_inner(&term.manifest, depth + 1, stack)?;
                    if parent.adapter != term.adapter
                        || parent.receiver_model != manifest.receiver_model
                        || parent.receiver_profile != manifest.receiver_profile
                        || parent.architecture_id != manifest.architecture_id
                        || parent.parameter_topology_sha256 != manifest.parameter_topology_sha256
                        || parent.adaptation_abi_sha256 != manifest.adaptation_abi_sha256
                        || parent.parameter_layout != manifest.parameter_layout
                    {
                        return Err(integrity("adapter_manifest_composition_parent_mismatch"));
                    }
                    sources.push((parent.dense_delta, term.coefficient));
                }
                let derived = derive_content_addressed_dvec_combination(&self.root, &sources)?;
                if derived != manifest.dense_delta {
                    return Err(integrity("adapter_manifest_composition_delta_mismatch"));
                }
            }
        }
        Ok(())
    }

    fn entry_from_manifest(
        manifest: &AdapterManifest,
        reference: PrivateFileReference,
    ) -> AdapterIndexEntry {
        AdapterIndexEntry {
            adapter: manifest.adapter.clone(),
            capability_id: manifest.capability_id.clone(),
            lineage_id: manifest.lineage_id.clone(),
            receiver_model: manifest.receiver_model.clone(),
            parameter_layout_sha256: manifest.parameter_layout.parameter_layout_sha256.clone(),
            manifest: reference,
        }
    }

    fn derived_buckets(
        entries: &[AdapterIndexEntry],
    ) -> (Vec<CapabilityAdapterBucket>, Vec<ModelAdapterBucket>) {
        let mut capabilities = BTreeMap::<CapabilityId, Vec<AdapterVersion>>::new();
        let mut models = BTreeMap::<ReceiverModelKey, Vec<AdapterVersion>>::new();
        for entry in entries {
            capabilities
                .entry(entry.capability_id.clone())
                .or_default()
                .push(entry.adapter.clone());
            models
                .entry(entry.receiver_model.clone())
                .or_default()
                .push(entry.adapter.clone());
        }
        (
            capabilities
                .into_iter()
                .map(|(capability_id, adapters)| CapabilityAdapterBucket {
                    capability_id,
                    adapters,
                })
                .collect(),
            models
                .into_iter()
                .map(|(receiver_model, adapters)| ModelAdapterBucket {
                    receiver_model,
                    adapters,
                })
                .collect(),
        )
    }

    fn validate_snapshot_contract(&self, snapshot: &AdapterIndexSnapshot) -> BrainResult<()> {
        if snapshot.schema != ADAPTER_INDEX_SCHEMA {
            return Err(invalid("adapter_index_schema_invalid"));
        }
        if snapshot.revision == 0 {
            if snapshot != &Self::empty_snapshot() {
                return Err(integrity("adapter_index_empty_snapshot_invalid"));
            }
            return Ok(());
        }
        if (snapshot.revision == 1) != snapshot.parent.is_none()
            || matches!(snapshot.operation, AdapterBankOperation::Empty)
        {
            return Err(integrity("adapter_index_history_link_invalid"));
        }
        let mut prior = None;
        let mut manifests = BTreeMap::new();
        let mut lineage_generations = BTreeSet::new();
        for entry in &snapshot.entries {
            entry.adapter.validate()?;
            if prior.as_ref().is_some_and(|prior| prior >= &entry.adapter)
                || !lineage_generations.insert((entry.lineage_id.clone(), entry.adapter.generation))
            {
                return Err(integrity("adapter_index_entry_order_or_lineage_invalid"));
            }
            prior = Some(entry.adapter.clone());
            let manifest = self.authenticate_manifest(&entry.manifest)?;
            if entry.adapter != manifest.adapter
                || entry.capability_id != manifest.capability_id
                || entry.lineage_id != manifest.lineage_id
                || entry.receiver_model != manifest.receiver_model
                || entry.parameter_layout_sha256
                    != manifest.parameter_layout.parameter_layout_sha256
            {
                return Err(integrity("adapter_index_manifest_binding_mismatch"));
            }
            manifests.insert(entry.adapter.clone(), manifest);
        }
        let (by_capability, by_model) = Self::derived_buckets(&snapshot.entries);
        if snapshot.by_capability != by_capability || snapshot.by_model != by_model {
            return Err(integrity("adapter_index_projection_mismatch"));
        }
        let mut revoked = BTreeMap::new();
        let mut prior_revocation = None;
        for revocation in &snapshot.revocations {
            validate_reason(&revocation.reason)?;
            if revocation.revoked_at_revision == 0
                || revocation.revoked_at_revision > snapshot.revision
                || prior_revocation
                    .as_ref()
                    .is_some_and(|prior| prior >= &revocation.adapter)
                || revoked
                    .insert(revocation.adapter.clone(), revocation)
                    .is_some()
            {
                return Err(integrity("adapter_revocation_order_invalid"));
            }
            prior_revocation = Some(revocation.adapter.clone());
            let entry = snapshot
                .entries
                .iter()
                .find(|entry| entry.adapter == revocation.adapter)
                .ok_or_else(|| integrity("adapter_revocation_target_missing"))?;
            if entry.manifest.sha256 != revocation.manifest_sha256 {
                return Err(integrity("adapter_revocation_manifest_mismatch"));
            }
            if let Some(evidence) = &revocation.evidence {
                evidence.verify(&self.root)?;
            }
        }
        for (version, manifest) in &manifests {
            if let AdapterManifestSource::ExactDenseComposition { terms, .. } = &manifest.source {
                let dependency_revoked =
                    terms.iter().any(|term| revoked.contains_key(&term.adapter));
                if dependency_revoked && !revoked.contains_key(version) {
                    return Err(integrity("adapter_revocation_closure_incomplete"));
                }
            }
        }
        let mut prior_slot = None;
        for selection in &snapshot.active {
            if selection.activated_at_revision == 0
                || selection.activated_at_revision > snapshot.revision
                || prior_slot
                    .as_ref()
                    .is_some_and(|prior| prior >= &selection.slot)
                || revoked.contains_key(&selection.adapter)
                || selection.execution_binding.activation_epoch != selection.activated_at_revision
            {
                return Err(integrity("adapter_active_selection_invalid"));
            }
            prior_slot = Some(selection.slot.clone());
            let manifest = manifests
                .get(&selection.adapter)
                .ok_or_else(|| integrity("adapter_active_manifest_missing"))?;
            if selection.slot.receiver_model != manifest.receiver_model
                || selection.slot.capability_id != manifest.capability_id
                || selection.execution_binding.base_model_sha256
                    != manifest.receiver_model.model_sha256
                || selection.execution_binding.adapter_delta_sha256 != manifest.dense_delta.sha256
                || selection.execution_binding.adaptation_abi_sha256
                    != manifest.adaptation_abi_sha256
                || selection.manifest
                    != snapshot
                        .entries
                        .iter()
                        .find(|entry| entry.adapter == selection.adapter)
                        .ok_or_else(|| integrity("adapter_active_entry_missing"))?
                        .manifest
            {
                return Err(integrity("adapter_active_binding_mismatch"));
            }
            let authorization =
                self.authenticate_promotion_authorization(&selection.authorization)?;
            if authorization.adapter != selection.adapter
                || authorization.manifest != selection.manifest
            {
                return Err(integrity("adapter_active_authorization_mismatch"));
            }
        }
        Ok(())
    }

    fn validate_snapshot_transition(
        &self,
        parent_reference: Option<&PrivateFileReference>,
        parent: &AdapterIndexSnapshot,
        child: &AdapterIndexSnapshot,
    ) -> BrainResult<()> {
        let expected_revision = parent
            .revision
            .checked_add(1)
            .ok_or_else(|| integrity("adapter_bank_transition_revision_overflow"))?;
        if child.revision != expected_revision || child.parent.as_ref() != parent_reference {
            return Err(integrity("adapter_bank_transition_link_invalid"));
        }

        match &child.operation {
            AdapterBankOperation::Empty => {
                return Err(integrity("adapter_bank_transition_empty_operation"));
            }
            AdapterBankOperation::Register { manifest }
            | AdapterBankOperation::Compose { manifest } => {
                if child.active != parent.active || child.revocations != parent.revocations {
                    return Err(integrity("adapter_bank_registration_mutated_lifecycle"));
                }
                let authenticated = self.authenticate_manifest(manifest)?;
                let source_matches = matches!(
                    (&child.operation, &authenticated.source),
                    (
                        AdapterBankOperation::Register { .. },
                        AdapterManifestSource::ImportedPeftLora { .. }
                    ) | (
                        AdapterBankOperation::Compose { .. },
                        AdapterManifestSource::ExactDenseComposition { .. }
                    )
                );
                if !source_matches {
                    return Err(integrity("adapter_bank_registration_operation_mismatch"));
                }
                let mut expected_entries = parent.entries.clone();
                expected_entries.push(Self::entry_from_manifest(&authenticated, manifest.clone()));
                expected_entries.sort_by(|left, right| left.adapter.cmp(&right.adapter));
                if child.entries != expected_entries {
                    return Err(integrity("adapter_bank_registration_transition_invalid"));
                }
            }
            AdapterBankOperation::Activate {
                adapter,
                authorization,
            } => {
                if child.entries != parent.entries || child.revocations != parent.revocations {
                    return Err(integrity("adapter_bank_activation_mutated_registry"));
                }
                let entry = child
                    .entries
                    .iter()
                    .find(|entry| &entry.adapter == adapter)
                    .ok_or_else(|| integrity("adapter_bank_activation_entry_missing"))?;
                let manifest = self.authenticate_manifest(&entry.manifest)?;
                let slot = AdapterActivationSlot {
                    receiver_model: manifest.receiver_model,
                    capability_id: manifest.capability_id,
                };
                let selection = child
                    .active
                    .iter()
                    .find(|selection| selection.slot == slot)
                    .ok_or_else(|| integrity("adapter_bank_activation_selection_missing"))?;
                if &selection.adapter != adapter
                    || &selection.authorization != authorization
                    || selection.activated_at_revision != child.revision
                    || selection.execution_binding.activation_epoch != child.revision
                {
                    return Err(integrity("adapter_bank_activation_transition_invalid"));
                }
                let mut expected_active = parent.active.clone();
                expected_active.retain(|candidate| candidate.slot != slot);
                expected_active.push(selection.clone());
                expected_active.sort_by(|left, right| left.slot.cmp(&right.slot));
                if child.active != expected_active {
                    return Err(integrity("adapter_bank_activation_changed_other_slots"));
                }
            }
            AdapterBankOperation::Revoke { root, affected } => {
                if child.entries != parent.entries
                    || affected.is_empty()
                    || !affected.contains(root)
                    || affected.windows(2).any(|pair| pair[0] >= pair[1])
                {
                    return Err(integrity("adapter_bank_revocation_transition_invalid"));
                }
                if !parent
                    .revocations
                    .iter()
                    .all(|revocation| child.revocations.contains(revocation))
                {
                    return Err(integrity("adapter_bank_revocation_removed_history"));
                }
                let added = child
                    .revocations
                    .iter()
                    .filter(|revocation| !parent.revocations.contains(revocation))
                    .collect::<Vec<_>>();
                let added_versions = added
                    .iter()
                    .map(|revocation| revocation.adapter.clone())
                    .collect::<Vec<_>>();
                if &added_versions != affected
                    || added.iter().any(|revocation| {
                        revocation.root_cause != *root
                            || revocation.revoked_at_revision != child.revision
                    })
                {
                    return Err(integrity("adapter_bank_revocation_delta_invalid"));
                }
                let affected_set = affected.iter().cloned().collect::<BTreeSet<_>>();
                let expected_active = parent
                    .active
                    .iter()
                    .filter(|selection| !affected_set.contains(&selection.adapter))
                    .cloned()
                    .collect::<Vec<_>>();
                if child.active != expected_active {
                    return Err(integrity("adapter_bank_revocation_active_delta_invalid"));
                }
            }
            AdapterBankOperation::Rollback {
                slot,
                target_revision,
                from,
                to,
                authorization,
                ..
            } => {
                if child.entries != parent.entries
                    || child.revocations != parent.revocations
                    || *target_revision == 0
                    || *target_revision >= parent.revision
                {
                    return Err(integrity("adapter_bank_rollback_transition_invalid"));
                }
                let parent_adapter = parent
                    .active
                    .iter()
                    .find(|selection| &selection.slot == slot)
                    .map(|selection| &selection.adapter);
                if parent_adapter != from.as_ref() {
                    return Err(integrity("adapter_bank_rollback_from_mismatch"));
                }
                let (target_commit, _) = self.read_revision_commit(*target_revision)?;
                let target = self.authenticate_snapshot(&target_commit.index)?;
                let target_adapter = target
                    .active
                    .iter()
                    .find(|selection| &selection.slot == slot)
                    .map(|selection| &selection.adapter);
                if target_adapter != to.as_ref() {
                    return Err(integrity("adapter_bank_rollback_target_mismatch"));
                }
                let mut expected_active = parent.active.clone();
                expected_active.retain(|selection| &selection.slot != slot);
                match to {
                    Some(adapter) => {
                        let permit = authorization.as_ref().ok_or_else(|| {
                            integrity("adapter_bank_rollback_authorization_missing")
                        })?;
                        let selection = child
                            .active
                            .iter()
                            .find(|selection| &selection.slot == slot)
                            .ok_or_else(|| integrity("adapter_bank_rollback_selection_missing"))?;
                        if &selection.adapter != adapter
                            || &selection.authorization != permit
                            || selection.activated_at_revision != child.revision
                            || selection.execution_binding.activation_epoch != child.revision
                        {
                            return Err(integrity("adapter_bank_rollback_selection_invalid"));
                        }
                        expected_active.push(selection.clone());
                        expected_active.sort_by(|left, right| left.slot.cmp(&right.slot));
                    }
                    None if authorization.is_some() => {
                        return Err(integrity(
                            "adapter_bank_rollback_deactivation_has_authorization",
                        ));
                    }
                    None => {}
                }
                if child.active != expected_active {
                    return Err(integrity("adapter_bank_rollback_changed_other_slots"));
                }
            }
        }
        Ok(())
    }

    fn persist_snapshot(
        &self,
        snapshot: &AdapterIndexSnapshot,
    ) -> BrainResult<PrivateFileReference> {
        self.validate_snapshot_contract(snapshot)?;
        let bytes = serde_json::to_vec(snapshot)?;
        let digest = Sha256Digest::digest_bytes(&bytes);
        let path = self.index_path(&digest);
        let written = write_or_verify_immutable(&self.root, &path, &bytes)?;
        if written != digest {
            return Err(integrity("adapter_index_write_mismatch"));
        }
        let reference = PrivateFileReference::new(path, digest);
        let replay = self.authenticate_snapshot(&reference)?;
        if replay != *snapshot {
            return Err(integrity("adapter_index_roundtrip_mismatch"));
        }
        Ok(reference)
    }

    fn authenticate_snapshot(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<AdapterIndexSnapshot> {
        if reference.path != self.index_path(&reference.sha256) {
            return Err(integrity("adapter_index_path_not_canonical"));
        }
        let bytes = reference.read_verified_bounded(&self.root, MAX_BANK_JSON_BYTES)?;
        let snapshot: AdapterIndexSnapshot = serde_json::from_slice(&bytes)?;
        if serde_json::to_vec(&snapshot)? != bytes {
            return Err(integrity("adapter_index_encoding_not_canonical"));
        }
        self.validate_snapshot_contract(&snapshot)?;
        Ok(snapshot)
    }

    fn persist_revision_commit(
        &self,
        commit: &AdapterBankRevisionCommit,
    ) -> BrainResult<PrivateFileReference> {
        let bytes = serde_json::to_vec(commit)?;
        let digest = Sha256Digest::digest_bytes(&bytes);
        let path = self.revision_commit_path(commit.revision);
        let written = write_or_verify_immutable(&self.root, &path, &bytes)?;
        if written != digest {
            return Err(integrity("adapter_bank_revision_commit_write_mismatch"));
        }
        let reference = PrivateFileReference::new(path, digest);
        let (replay, replay_reference) = self.read_revision_commit(commit.revision)?;
        if replay != *commit || replay_reference != reference {
            return Err(integrity("adapter_bank_revision_commit_roundtrip_mismatch"));
        }
        Ok(reference)
    }

    fn publish_snapshot(
        &self,
        snapshot: &AdapterIndexSnapshot,
    ) -> BrainResult<PrivateFileReference> {
        if snapshot.revision == 0 || snapshot.revision > MAX_HISTORY_REVISIONS as u64 {
            return Err(invalid("adapter_bank_history_revision_limit"));
        }
        let journal_tip = self.authenticate_revision_tip()?;
        let (expected_revision, previous_commit, previous_index) =
            if let Some((commit, reference)) = journal_tip.as_ref() {
                (
                    commit
                        .revision
                        .checked_add(1)
                        .ok_or_else(|| invalid("adapter_bank_revision_overflow"))?,
                    Some(reference.clone()),
                    Some(commit.index.clone()),
                )
            } else {
                (1, None, None)
            };
        if snapshot.revision != expected_revision
            || snapshot.parent.as_ref() != previous_index.as_ref()
        {
            return Err(integrity("adapter_bank_revision_publish_conflict"));
        }
        let parent = match journal_tip.as_ref() {
            Some((commit, _)) => self.authenticate_snapshot(&commit.index)?,
            None => Self::empty_snapshot(),
        };
        self.validate_snapshot_transition(previous_index.as_ref(), &parent, snapshot)?;

        let reference = self.persist_snapshot(snapshot)?;
        let revision_commit = AdapterBankRevisionCommit {
            schema: ADAPTER_BANK_REVISION_COMMIT_SCHEMA.to_string(),
            revision: snapshot.revision,
            index: reference.clone(),
            previous_commit,
            previous_index,
        };
        self.persist_revision_commit(&revision_commit)?;

        let head = AdapterBankHead {
            schema: ADAPTER_BANK_HEAD_SCHEMA.to_string(),
            revision: snapshot.revision,
            index: reference.clone(),
        };
        let bytes = serde_json::to_vec(&head)?;
        let expected = Sha256Digest::digest_bytes(&bytes);
        let written =
            replace_private_file_atomic(&self.root, &self.head_path(), &bytes, Some(&expected))?;
        if written != expected {
            return Err(integrity("adapter_bank_head_write_mismatch"));
        }
        let (reopened, reopened_reference) = self.current()?;
        if reopened != *snapshot || reopened_reference.as_ref() != Some(&reference) {
            return Err(integrity("adapter_bank_publish_roundtrip_mismatch"));
        }
        Ok(reference)
    }

    fn commit_registration(
        &self,
        manifest: &AdapterManifest,
        reference: &PrivateFileReference,
        expected: &AdapterBankExpectation,
        composed: bool,
    ) -> BrainResult<AdapterBankCommit> {
        let root = self.root.clone();
        let lock = self.lock_path();
        with_private_authority_lock(&root, &lock, || {
            let (mut current, current_reference) = self.current()?;
            if let Some(existing) = current
                .entries
                .iter()
                .find(|entry| entry.adapter == manifest.adapter)
            {
                if existing.manifest == *reference {
                    return Ok(AdapterBankCommit {
                        schema: ADAPTER_BANK_COMMIT_SCHEMA.to_string(),
                        revision: current.revision,
                        index: current_reference
                            .ok_or_else(|| integrity("adapter_bank_current_reference_missing"))?,
                        operation: current.operation,
                        manifest: Some(reference.clone()),
                        idempotent: true,
                    });
                }
                return Err(integrity("adapter_version_conflict"));
            }
            Self::check_expectation(expected, &current, current_reference.as_ref())?;
            let same_id = current
                .entries
                .iter()
                .filter(|entry| entry.adapter.adapter_id == manifest.adapter.adapter_id)
                .collect::<Vec<_>>();
            if same_id.is_empty() {
                if manifest.adapter.generation != 1 {
                    return Err(invalid("adapter_first_generation_must_be_one"));
                }
            } else {
                let previous = same_id
                    .iter()
                    .max_by_key(|entry| entry.adapter.generation)
                    .expect("nonempty");
                if manifest.adapter.generation != previous.adapter.generation + 1
                    || previous.capability_id != manifest.capability_id
                    || previous.lineage_id != manifest.lineage_id
                    || previous.receiver_model != manifest.receiver_model
                {
                    return Err(invalid("adapter_generation_lineage_invalid"));
                }
            }
            if current.entries.iter().any(|entry| {
                entry.lineage_id == manifest.lineage_id
                    && entry.adapter.generation == manifest.adapter.generation
            }) {
                return Err(integrity("adapter_lineage_generation_conflict"));
            }
            if current.entries.iter().any(|entry| {
                entry.capability_id == manifest.capability_id
                    && entry.receiver_model == manifest.receiver_model
                    && entry.parameter_layout_sha256
                        == manifest.parameter_layout.parameter_layout_sha256
                    && self
                        .authenticate_manifest(&entry.manifest)
                        .is_ok_and(|other| other.dense_delta == manifest.dense_delta)
            }) {
                return Err(integrity("adapter_operational_duplicate"));
            }
            let next_revision = current
                .revision
                .checked_add(1)
                .ok_or_else(|| invalid("adapter_bank_revision_overflow"))?;
            current.revision = next_revision;
            current.parent = current_reference;
            current.operation = if composed {
                AdapterBankOperation::Compose {
                    manifest: reference.clone(),
                }
            } else {
                AdapterBankOperation::Register {
                    manifest: reference.clone(),
                }
            };
            current
                .entries
                .push(Self::entry_from_manifest(manifest, reference.clone()));
            current
                .entries
                .sort_by(|left, right| left.adapter.cmp(&right.adapter));
            (current.by_capability, current.by_model) = Self::derived_buckets(&current.entries);
            let index = self.publish_snapshot(&current)?;
            Ok(AdapterBankCommit {
                schema: ADAPTER_BANK_COMMIT_SCHEMA.to_string(),
                revision: current.revision,
                index,
                operation: current.operation,
                manifest: Some(reference.clone()),
                idempotent: false,
            })
        })
    }

    pub fn import_lora(&self, request: &AdapterImportRequest) -> BrainResult<AdapterBankCommit> {
        if request.schema != ADAPTER_IMPORT_REQUEST_SCHEMA || request.lineage_id.is_unassigned() {
            return Err(invalid("adapter_import_request_invalid"));
        }
        request.adapter.validate()?;
        self.authenticate_bound_capability_bundle(
            request.capability_bundle.as_ref(),
            &request.capability_id,
        )?;
        let profile =
            authenticate_live_receiver_model_profile(&self.root, &request.receiver_profile)?;
        let receipt = import_peft_lora_as_dense_axis(&self.root, &request.lora)?;
        validate_lora_axis_against_profile(&self.root, &request.receiver_profile, &receipt)?;
        let provenance = self.persist_import_provenance(&receipt)?;
        let adapter_model = self.retain_external_blob(
            &receipt.adapter_model_path,
            &receipt.adapter_model_sha256,
            "safetensors",
            "application/x-safetensors",
        )?;
        let adapter_config = self.retain_external_blob(
            &receipt.adapter_config_path,
            &receipt.adapter_config_sha256,
            "json",
            "application/json",
        )?;
        let replay_reference = {
            let bytes = serde_json::to_vec(&receipt)?;
            let digest = Sha256Digest::digest_bytes(&bytes);
            let path = self
                .bank_root()
                .join("raw-import-receipts/by-sha")
                .join(format!("{digest}.json"));
            write_or_verify_immutable(&self.root, &path, &bytes)?;
            PrivateFileReference::new(path, digest)
        };
        let replay = authenticate_lora_adapter_axis_receipt(&self.root, &replay_reference)?;
        if replay != receipt {
            return Err(integrity("adapter_import_replay_mismatch"));
        }
        let layout_sha256 = parameter_layout_digest(&receipt.parameter_layout)?;
        let layout_artifact = ParameterLayoutAuthority::for_internal_root(&self.root)?
            .persist(receipt.parameter_layout)?;
        let manifest = AdapterManifest {
            schema: ADAPTER_MANIFEST_SCHEMA.to_string(),
            adapter: request.adapter.clone(),
            capability_id: request.capability_id.clone(),
            lineage_id: request.lineage_id.clone(),
            capability_bundle: request.capability_bundle.clone(),
            receiver_model: ReceiverModelKey {
                model_id: profile.model_id,
                model_sha256: profile.checkpoint.sha256,
            },
            architecture_id: profile.architecture_id,
            receiver_profile: request.receiver_profile.clone(),
            parameter_topology_sha256: profile.parameter_topology_sha256,
            adaptation_abi_sha256: profile.adaptation_abi_sha256,
            parameter_layout: ReceiverAdaptationLayoutBinding {
                artifact: layout_artifact,
                parameter_layout_sha256: layout_sha256,
                total_parameter_count: receipt.dense_delta.parameter_count,
            },
            dense_delta: receipt.dense_delta,
            source: AdapterManifestSource::ImportedPeftLora {
                import_provenance: provenance,
                adapter_model,
                adapter_config,
            },
            candidate_only: true,
            authorizes_promotion: false,
        };
        let reference = self.persist_manifest(&manifest)?;
        self.commit_registration(&manifest, &reference, &request.expected, false)
    }

    pub fn compose_exact(
        &self,
        request: &AdapterCompositionRequest,
    ) -> BrainResult<AdapterBankCommit> {
        if request.schema != ADAPTER_COMPOSITION_REQUEST_SCHEMA
            || request.lineage_id.is_unassigned()
            || request.terms.len() < 2
            || request.terms.len() > MAX_COMPOSITION_TERMS
        {
            return Err(invalid("adapter_composition_request_invalid"));
        }
        request.adapter.validate()?;
        self.authenticate_bound_capability_bundle(
            request.capability_bundle.as_ref(),
            &request.capability_id,
        )?;
        let (snapshot, _) = self.current()?;
        let revoked = snapshot
            .revocations
            .iter()
            .map(|revocation| revocation.adapter.clone())
            .collect::<BTreeSet<_>>();
        let mut terms = Vec::with_capacity(request.terms.len());
        let mut manifests = Vec::with_capacity(request.terms.len());
        let mut versions = BTreeSet::new();
        for input in &request.terms {
            input.adapter.validate()?;
            validate_coefficient(input.coefficient)?;
            if !versions.insert(input.adapter.clone()) || revoked.contains(&input.adapter) {
                return Err(invalid("adapter_composition_source_invalid"));
            }
            let entry = snapshot
                .entries
                .iter()
                .find(|entry| entry.adapter == input.adapter)
                .ok_or_else(|| invalid("adapter_composition_source_missing"))?;
            let manifest = self.authenticate_manifest(&entry.manifest)?;
            terms.push(AdapterCompositionTerm {
                adapter: input.adapter.clone(),
                manifest: entry.manifest.clone(),
                coefficient: input.coefficient,
                coefficient_bits_hex: coefficient_bits_hex(input.coefficient),
            });
            manifests.push((entry.manifest.sha256.clone(), manifest));
        }
        terms.sort_by(|left, right| left.manifest.sha256.cmp(&right.manifest.sha256));
        manifests.sort_by(|left, right| left.0.cmp(&right.0));
        if manifests
            .windows(2)
            .any(|window| window[0].0 == window[1].0)
        {
            return Err(invalid("adapter_composition_source_duplicate"));
        }
        let first = manifests[0].1.clone();
        for (_, manifest) in manifests.iter().skip(1) {
            if manifest.receiver_model != first.receiver_model
                || manifest.receiver_profile != first.receiver_profile
                || manifest.architecture_id != first.architecture_id
                || manifest.parameter_topology_sha256 != first.parameter_topology_sha256
                || manifest.adaptation_abi_sha256 != first.adaptation_abi_sha256
                || manifest.parameter_layout != first.parameter_layout
            {
                return Err(invalid("adapter_composition_compatibility_mismatch"));
            }
        }
        authenticate_live_receiver_model_profile(&self.root, &first.receiver_profile)?;
        let by_digest = manifests.into_iter().collect::<BTreeMap<_, _>>();
        let sources = terms
            .iter()
            .map(|term| {
                let manifest = by_digest
                    .get(&term.manifest.sha256)
                    .ok_or_else(|| integrity("adapter_composition_manifest_resolution_failed"))?;
                Ok((manifest.dense_delta.clone(), term.coefficient))
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let dense_delta = ArtifactWriteAuthority::for_internal_root(&self.root)?
            .combine_content_addressed_dvec(&sources)?;
        let manifest = AdapterManifest {
            schema: ADAPTER_MANIFEST_SCHEMA.to_string(),
            adapter: request.adapter.clone(),
            capability_id: request.capability_id.clone(),
            lineage_id: request.lineage_id.clone(),
            capability_bundle: request.capability_bundle.clone(),
            receiver_model: first.receiver_model.clone(),
            architecture_id: first.architecture_id.clone(),
            receiver_profile: first.receiver_profile.clone(),
            parameter_topology_sha256: first.parameter_topology_sha256.clone(),
            adaptation_abi_sha256: first.adaptation_abi_sha256.clone(),
            parameter_layout: first.parameter_layout.clone(),
            dense_delta,
            source: AdapterManifestSource::ExactDenseComposition {
                arithmetic: EXACT_COMPOSITION_ARITHMETIC.to_string(),
                terms,
                rank_truncation_used: false,
                factor_refactorization_used: false,
            },
            candidate_only: true,
            authorizes_promotion: false,
        };
        let manifest_reference = self.persist_manifest(&manifest)?;
        self.commit_registration(&manifest, &manifest_reference, &request.expected, true)
    }

    fn authenticated_manifest_layout(
        &self,
        manifest: &AdapterManifest,
    ) -> BrainResult<crate::block_tomography::ParameterBlockLayout> {
        Ok(ParameterLayoutAuthority::for_internal_root(&self.root)?
            .authenticate_canonical_binding(
                manifest.parameter_layout.artifact.clone(),
                &manifest.parameter_layout.parameter_layout_sha256,
                manifest.parameter_layout.total_parameter_count,
            )?
            .artifact
            .layout)
    }

    fn persist_candidate_materialization(
        &self,
        materialization: &AdapterCandidateMaterialization,
    ) -> BrainResult<PrivateFileReference> {
        let bytes = serde_json::to_vec(materialization)?;
        let digest = Sha256Digest::digest_bytes(&bytes);
        let path = self.materialization_path(&digest);
        let written = write_or_verify_immutable(&self.root, &path, &bytes)?;
        if written != digest {
            return Err(integrity("adapter_materialization_write_mismatch"));
        }
        let reference = PrivateFileReference::new(path, digest);
        let replay = self.authenticate_candidate_materialization(&reference)?;
        if replay != *materialization {
            return Err(integrity("adapter_materialization_roundtrip_mismatch"));
        }
        Ok(reference)
    }

    /// Materialize one registered dense adapter into an immutable candidate
    /// checkpoint. This operation never changes bank state and never grants
    /// promotion authority; its CAS record is only evidence for governance.
    pub fn materialize_candidate(
        &self,
        request: &AdapterCandidateMaterializationRequest,
    ) -> BrainResult<PrivateFileReference> {
        if request.schema != ADAPTER_CANDIDATE_MATERIALIZATION_REQUEST_SCHEMA {
            return Err(invalid("adapter_materialization_request_invalid"));
        }
        request.adapter.validate()?;
        let output_path = self.confined_materialization_output(&request.output_path)?;
        let (snapshot, observed_index) = self.current()?;
        Self::check_expectation(&request.expected, &snapshot, observed_index.as_ref())?;
        let observed_index = observed_index
            .ok_or_else(|| invalid("adapter_materialization_requires_registered_index"))?;
        if snapshot
            .revocations
            .iter()
            .any(|revocation| revocation.adapter == request.adapter)
        {
            return Err(invalid("adapter_materialization_target_revoked"));
        }
        let entry = snapshot
            .entries
            .iter()
            .find(|entry| entry.adapter == request.adapter)
            .cloned()
            .ok_or_else(|| invalid("adapter_materialization_target_missing"))?;
        let manifest = self.authenticate_manifest(&entry.manifest)?;
        let profile =
            authenticate_live_receiver_model_profile(&self.root, &manifest.receiver_profile)?;
        let layout = self.authenticated_manifest_layout(&manifest)?;
        let receipt = materialize_dense_delta_checkpoint(
            &self.root,
            &profile.checkpoint.path,
            &manifest.receiver_model.model_sha256,
            &layout,
            &manifest.dense_delta,
            &output_path,
        )?;
        authenticate_weight_materialization_receipt(
            &self.root,
            &profile.checkpoint.path,
            &layout,
            &manifest.dense_delta,
            &output_path,
            &receipt,
        )?;
        let materialization = AdapterCandidateMaterialization {
            schema: ADAPTER_CANDIDATE_MATERIALIZATION_SCHEMA.to_string(),
            adapter: request.adapter.clone(),
            manifest: entry.manifest,
            capability_id: manifest.capability_id,
            capability_bundle: manifest.capability_bundle,
            receiver_profile: manifest.receiver_profile,
            observed_revision: snapshot.revision,
            observed_index,
            parameter_layout_sha256: manifest.parameter_layout.parameter_layout_sha256,
            dense_delta_sha256: manifest.dense_delta.sha256,
            output_path,
            receipt,
            candidate_only: true,
            authorizes_promotion: false,
        };
        self.persist_candidate_materialization(&materialization)
    }

    /// Re-open and bind a candidate record to its committed observed snapshot,
    /// immutable manifest, live receiver, canonical layout, delta and output.
    /// Successful authentication still does not authorize promotion.
    pub fn authenticate_candidate_materialization(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<AdapterCandidateMaterialization> {
        if reference.path != self.materialization_path(&reference.sha256) {
            return Err(integrity("adapter_materialization_path_not_canonical"));
        }
        let bytes = reference.read_verified_bounded(&self.root, MAX_BANK_JSON_BYTES)?;
        let materialization: AdapterCandidateMaterialization = serde_json::from_slice(&bytes)?;
        if materialization.schema != ADAPTER_CANDIDATE_MATERIALIZATION_SCHEMA
            || !materialization.candidate_only
            || materialization.authorizes_promotion
            || materialization.observed_revision == 0
            || serde_json::to_vec(&materialization)? != bytes
        {
            return Err(integrity("adapter_materialization_contract_invalid"));
        }
        materialization.adapter.validate()?;
        let (commit, _) = self.read_revision_commit(materialization.observed_revision)?;
        if commit.index != materialization.observed_index {
            return Err(integrity(
                "adapter_materialization_observed_commit_mismatch",
            ));
        }
        let snapshot = self.authenticate_snapshot(&materialization.observed_index)?;
        if snapshot.revision != materialization.observed_revision
            || snapshot
                .revocations
                .iter()
                .any(|revocation| revocation.adapter == materialization.adapter)
        {
            return Err(integrity(
                "adapter_materialization_observed_snapshot_invalid",
            ));
        }
        let entry = snapshot
            .entries
            .iter()
            .find(|entry| entry.adapter == materialization.adapter)
            .ok_or_else(|| integrity("adapter_materialization_observed_entry_missing"))?;
        if entry.manifest != materialization.manifest {
            return Err(integrity("adapter_materialization_manifest_mismatch"));
        }
        let manifest = self.authenticate_manifest(&entry.manifest)?;
        if manifest.capability_id != materialization.capability_id
            || manifest.capability_bundle != materialization.capability_bundle
            || manifest.receiver_profile != materialization.receiver_profile
            || manifest.parameter_layout.parameter_layout_sha256
                != materialization.parameter_layout_sha256
            || manifest.dense_delta.sha256 != materialization.dense_delta_sha256
        {
            return Err(integrity("adapter_materialization_binding_mismatch"));
        }
        let profile =
            authenticate_live_receiver_model_profile(&self.root, &manifest.receiver_profile)?;
        let layout = self.authenticated_manifest_layout(&manifest)?;
        let output_path = self.confined_materialization_output(&materialization.output_path)?;
        if output_path != materialization.output_path {
            return Err(integrity("adapter_materialization_output_path_mismatch"));
        }
        drop(open_existing_private_file(&self.root, &output_path)?);
        authenticate_weight_materialization_receipt(
            &self.root,
            &profile.checkpoint.path,
            &layout,
            &manifest.dense_delta,
            &output_path,
            &materialization.receipt,
        )?;
        Ok(materialization)
    }

    /// Derive the governance identity exclusively from an authenticated,
    /// content-addressed manifest.
    pub fn governance_candidate_id(
        &self,
        manifest: &PrivateFileReference,
    ) -> BrainResult<VariantId> {
        self.authenticate_manifest(manifest)?;
        VariantId::parse(format!("adapter.manifest.{}", manifest.sha256))
    }

    fn persist_promotion_authorization(
        &self,
        authorization: &AdapterPromotionAuthorization,
    ) -> BrainResult<PrivateFileReference> {
        authorization.decision.validate()?;
        let bytes = serde_json::to_vec(authorization)?;
        let digest = Sha256Digest::digest_bytes(&bytes);
        let path = self.authorization_path(&digest);
        let written = write_or_verify_immutable(&self.root, &path, &bytes)?;
        if written != digest {
            return Err(integrity("adapter_promotion_authorization_write_mismatch"));
        }
        let reference = PrivateFileReference::new(path, digest);
        self.authenticate_promotion_authorization(&reference)?;
        Ok(reference)
    }

    /// Resolve immutable sealed governance witnesses and delegate to the sole
    /// production promotion issuer. The request itself carries no decision
    /// authority: every referenced witness is semantically replayed first.
    pub fn authorize_governed_promotion_request(
        &self,
        request: &AdapterGovernedPromotionRequest,
    ) -> BrainResult<PrivateFileReference> {
        if request.schema != ADAPTER_GOVERNED_PROMOTION_REQUEST_SCHEMA {
            return Err(invalid("adapter_governed_promotion_request_invalid"));
        }
        let gate = authenticate_candidate_gate_decision(&self.root, &request.candidate_gate)?;
        let petfc = authenticate_petfc_assessment(&self.root, &request.petfc_assessment)?;
        let canary = authenticate_canary_state(&self.root, &request.canary_state)?;
        self.authorize_governed_promotion(
            &request.materialization,
            &gate,
            &petfc,
            &canary,
            &request.expected,
        )
    }

    /// Mint a promotion permit only after the complete sealed governance chain,
    /// the candidate-only materialization and a representation-closed
    /// capability bundle are re-authenticated against the current bank state.
    pub fn authorize_governed_promotion(
        &self,
        materialization_reference: &PrivateFileReference,
        gate: &CandidateGateDecision,
        petfc: &PetfcAssessment,
        canary: &CanaryState,
        expected: &AdapterBankExpectation,
    ) -> BrainResult<PrivateFileReference> {
        let root = self.root.clone();
        let lock = self.lock_path();
        with_private_authority_lock(&root, &lock, || {
            let (snapshot, current_reference) = self.current()?;
            Self::check_expectation(expected, &snapshot, current_reference.as_ref())?;
            let issued_against_index = current_reference
                .ok_or_else(|| invalid("adapter_promotion_requires_registered_index"))?;
            let materialization =
                self.authenticate_candidate_materialization(materialization_reference)?;
            let entry = snapshot
                .entries
                .iter()
                .find(|entry| entry.adapter == materialization.adapter)
                .ok_or_else(|| invalid("adapter_promotion_target_missing"))?;
            if entry.manifest != materialization.manifest
                || snapshot
                    .revocations
                    .iter()
                    .any(|revocation| revocation.adapter == materialization.adapter)
            {
                return Err(integrity("adapter_promotion_current_state_mismatch"));
            }
            let manifest = self.authenticate_manifest(&entry.manifest)?;
            authenticate_live_receiver_model_profile(&self.root, &manifest.receiver_profile)?;
            self.require_promotable_capability_bundle(&manifest)?;
            let candidate_id = self.governance_candidate_id(&entry.manifest)?;
            authenticate_adapter_promotion_witnesses(&candidate_id, gate, petfc, canary)?;

            let evidence = AdapterPromotionEvidence {
                independent_execution: gate.persist(&self.root)?,
                preservation_assessment: petfc.persist(&self.root)?,
                negative_controls: canary.persist(&self.root)?,
                materialization_receipt: materialization_reference.clone(),
            };
            let decision = PromotionDecision {
                allowed: true,
                reasons: Vec::new(),
                metrics: BTreeMap::from([
                    ("candidate_gate_authenticated".to_string(), 1.0),
                    ("petfc_authenticated".to_string(), 1.0),
                    ("canary_authenticated".to_string(), 1.0),
                    ("representation_closed".to_string(), 1.0),
                    ("candidate_materialization_authenticated".to_string(), 1.0),
                ]),
            };
            let authorization = AdapterPromotionAuthorization {
                schema: ADAPTER_PROMOTION_AUTHORIZATION_SCHEMA.to_string(),
                governed: true,
                adapter: manifest.adapter,
                manifest: entry.manifest.clone(),
                capability_id: manifest.capability_id,
                receiver_model: manifest.receiver_model,
                receiver_profile: manifest.receiver_profile,
                parameter_layout_sha256: manifest.parameter_layout.parameter_layout_sha256,
                dense_delta_sha256: manifest.dense_delta.sha256,
                issued_against_index,
                evidence,
                decision,
                authorizes_promotion: true,
            };
            self.persist_promotion_authorization(&authorization)
        })
    }

    fn authenticate_promotion_authorization(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<AdapterPromotionAuthorization> {
        if reference.path != self.authorization_path(&reference.sha256) {
            return Err(integrity(
                "adapter_promotion_authorization_path_not_canonical",
            ));
        }
        let bytes = reference.read_verified_bounded(&self.root, MAX_BANK_JSON_BYTES)?;
        let authorization: AdapterPromotionAuthorization = serde_json::from_slice(&bytes)?;
        authorization.decision.validate()?;
        if authorization.schema != ADAPTER_PROMOTION_AUTHORIZATION_SCHEMA
            || !authorization.authorizes_promotion
            || !authorization.decision.allowed
            || serde_json::to_vec(&authorization)? != bytes
        {
            return Err(integrity("adapter_promotion_authorization_invalid"));
        }
        if !authorization.governed {
            #[cfg(not(test))]
            return Err(integrity("adapter_promotion_authorization_not_governed"));
            #[cfg(test)]
            for evidence in [
                &authorization.evidence.independent_execution,
                &authorization.evidence.preservation_assessment,
                &authorization.evidence.negative_controls,
                &authorization.evidence.materialization_receipt,
            ] {
                evidence.verify(&self.root)?;
            }
        }
        let governed_evidence = if authorization.governed {
            let gate = authenticate_candidate_gate_decision(
                &self.root,
                &authorization.evidence.independent_execution,
            )?;
            let petfc = authenticate_petfc_assessment(
                &self.root,
                &authorization.evidence.preservation_assessment,
            )?;
            let canary =
                authenticate_canary_state(&self.root, &authorization.evidence.negative_controls)?;
            let materialization = self.authenticate_candidate_materialization(
                &authorization.evidence.materialization_receipt,
            )?;
            Some((materialization, gate, petfc, canary))
        } else {
            None
        };
        if authorization.issued_against_index.path
            != self.index_path(&authorization.issued_against_index.sha256)
        {
            return Err(integrity("adapter_promotion_parent_index_not_canonical"));
        }
        authorization.issued_against_index.verify(&self.root)?;
        let manifest = self.authenticate_manifest(&authorization.manifest)?;
        if authorization.adapter != manifest.adapter
            || authorization.capability_id != manifest.capability_id
            || authorization.receiver_model != manifest.receiver_model
            || authorization.receiver_profile != manifest.receiver_profile
            || authorization.parameter_layout_sha256
                != manifest.parameter_layout.parameter_layout_sha256
            || authorization.dense_delta_sha256 != manifest.dense_delta.sha256
        {
            return Err(integrity("adapter_promotion_manifest_binding_mismatch"));
        }
        if let Some((materialization, gate, petfc, canary)) = governed_evidence {
            self.require_promotable_capability_bundle(&manifest)?;
            let candidate_id = self.governance_candidate_id(&authorization.manifest)?;
            authenticate_adapter_promotion_witnesses(&candidate_id, &gate, &petfc, &canary)?;
            if materialization.adapter != authorization.adapter
                || materialization.manifest != authorization.manifest
                || materialization.capability_id != authorization.capability_id
                || materialization.receiver_profile != authorization.receiver_profile
                || materialization.parameter_layout_sha256 != authorization.parameter_layout_sha256
                || materialization.dense_delta_sha256 != authorization.dense_delta_sha256
            {
                return Err(integrity(
                    "adapter_promotion_materialization_binding_mismatch",
                ));
            }
        }
        Ok(authorization)
    }

    /// Unit-test-only seam for lifecycle tests. Production callers cannot
    /// supply arbitrary evidence or promotion decisions.
    #[cfg(test)]
    pub(crate) fn seal_verified_promotion(
        &self,
        verified: &VerifiedAdapterPromotion,
    ) -> BrainResult<PrivateFileReference> {
        verified.decision.validate()?;
        if !verified.decision.allowed {
            return Err(invalid("adapter_promotion_decision_not_allowed"));
        }
        let (snapshot, current_reference) = self.current()?;
        Self::check_expectation(&verified.expected, &snapshot, current_reference.as_ref())?;
        let issued_against_index = current_reference
            .ok_or_else(|| invalid("adapter_promotion_requires_registered_index"))?;
        let manifest = self.authenticate_manifest(&verified.manifest)?;
        for evidence in [
            &verified.evidence.independent_execution,
            &verified.evidence.preservation_assessment,
            &verified.evidence.negative_controls,
            &verified.evidence.materialization_receipt,
        ] {
            evidence.verify(&self.root)?;
        }
        self.persist_promotion_authorization(&AdapterPromotionAuthorization {
            schema: ADAPTER_PROMOTION_AUTHORIZATION_SCHEMA.to_string(),
            governed: false,
            adapter: manifest.adapter.clone(),
            manifest: verified.manifest.clone(),
            capability_id: manifest.capability_id,
            receiver_model: manifest.receiver_model,
            receiver_profile: manifest.receiver_profile,
            parameter_layout_sha256: manifest.parameter_layout.parameter_layout_sha256,
            dense_delta_sha256: manifest.dense_delta.sha256,
            issued_against_index,
            evidence: verified.evidence.clone(),
            decision: verified.decision.clone(),
            authorizes_promotion: true,
        })
    }

    pub fn activate(&self, request: &AdapterActivationRequest) -> BrainResult<AdapterBankCommit> {
        if request.schema != ADAPTER_ACTIVATION_REQUEST_SCHEMA {
            return Err(invalid("adapter_activation_request_invalid"));
        }
        request.adapter.validate()?;
        let root = self.root.clone();
        let lock = self.lock_path();
        with_private_authority_lock(&root, &lock, || {
            let (mut current, current_reference) = self.current()?;
            if let Some(selection) = current
                .active
                .iter()
                .find(|selection| selection.adapter == request.adapter)
            {
                if selection.authorization == request.authorization {
                    return Ok(AdapterBankCommit {
                        schema: ADAPTER_BANK_COMMIT_SCHEMA.to_string(),
                        revision: current.revision,
                        index: current_reference
                            .ok_or_else(|| integrity("adapter_bank_current_reference_missing"))?,
                        operation: current.operation,
                        manifest: Some(selection.manifest.clone()),
                        idempotent: true,
                    });
                }
            }
            Self::check_expectation(&request.expected, &current, current_reference.as_ref())?;
            let current_index = current_reference
                .clone()
                .ok_or_else(|| invalid("adapter_activation_requires_registered_index"))?;
            if current
                .revocations
                .iter()
                .any(|revocation| revocation.adapter == request.adapter)
            {
                return Err(invalid("adapter_activation_target_revoked"));
            }
            let entry = current
                .entries
                .iter()
                .find(|entry| entry.adapter == request.adapter)
                .cloned()
                .ok_or_else(|| invalid("adapter_activation_target_missing"))?;
            let manifest = self.authenticate_manifest(&entry.manifest)?;
            authenticate_live_receiver_model_profile(&self.root, &manifest.receiver_profile)?;
            let authorization =
                self.authenticate_promotion_authorization(&request.authorization)?;
            if authorization.adapter != request.adapter
                || authorization.manifest != entry.manifest
                || authorization.issued_against_index != current_index
            {
                return Err(integrity("adapter_activation_authorization_mismatch"));
            }
            let next_revision = current
                .revision
                .checked_add(1)
                .ok_or_else(|| invalid("adapter_bank_revision_overflow"))?;
            let slot = AdapterActivationSlot {
                receiver_model: manifest.receiver_model.clone(),
                capability_id: manifest.capability_id.clone(),
            };
            current.active.retain(|selection| selection.slot != slot);
            current.active.push(ActiveAdapterSelection {
                slot,
                adapter: request.adapter.clone(),
                manifest: entry.manifest.clone(),
                authorization: request.authorization.clone(),
                execution_binding: AdapterExecutionBinding {
                    base_model_sha256: manifest.receiver_model.model_sha256.clone(),
                    adapter_delta_sha256: manifest.dense_delta.sha256.clone(),
                    activation_epoch: next_revision,
                    adaptation_abi_sha256: manifest.adaptation_abi_sha256.clone(),
                },
                activated_at_revision: next_revision,
            });
            current
                .active
                .sort_by(|left, right| left.slot.cmp(&right.slot));
            current.revision = next_revision;
            current.parent = current_reference;
            current.operation = AdapterBankOperation::Activate {
                adapter: request.adapter.clone(),
                authorization: request.authorization.clone(),
            };
            let index = self.publish_snapshot(&current)?;
            Ok(AdapterBankCommit {
                schema: ADAPTER_BANK_COMMIT_SCHEMA.to_string(),
                revision: current.revision,
                index,
                operation: current.operation,
                manifest: Some(entry.manifest),
                idempotent: false,
            })
        })
    }

    fn composition_dependencies(manifest: &AdapterManifest) -> Vec<AdapterVersion> {
        match &manifest.source {
            AdapterManifestSource::ImportedPeftLora { .. } => Vec::new(),
            AdapterManifestSource::ExactDenseComposition { terms, .. } => {
                terms.iter().map(|term| term.adapter.clone()).collect()
            }
        }
    }

    pub fn revoke(&self, request: &AdapterRevocationRequest) -> BrainResult<AdapterBankCommit> {
        if request.schema != ADAPTER_REVOCATION_REQUEST_SCHEMA {
            return Err(invalid("adapter_revocation_request_invalid"));
        }
        request.adapter.validate()?;
        validate_reason(&request.reason)?;
        if let Some(evidence) = &request.evidence {
            evidence.verify(&self.root)?;
        }
        let root = self.root.clone();
        let lock = self.lock_path();
        with_private_authority_lock(&root, &lock, || {
            let (mut current, current_reference) = self.current()?;
            if current
                .revocations
                .iter()
                .any(|revocation| revocation.adapter == request.adapter)
            {
                return Ok(AdapterBankCommit {
                    schema: ADAPTER_BANK_COMMIT_SCHEMA.to_string(),
                    revision: current.revision,
                    index: current_reference
                        .ok_or_else(|| integrity("adapter_bank_current_reference_missing"))?,
                    operation: current.operation,
                    manifest: current
                        .entries
                        .iter()
                        .find(|entry| entry.adapter == request.adapter)
                        .map(|entry| entry.manifest.clone()),
                    idempotent: true,
                });
            }
            Self::check_expectation(&request.expected, &current, current_reference.as_ref())?;
            if !current
                .entries
                .iter()
                .any(|entry| entry.adapter == request.adapter)
            {
                return Err(invalid("adapter_revocation_target_missing"));
            }
            let manifests = current
                .entries
                .iter()
                .map(|entry| {
                    Ok((
                        entry.adapter.clone(),
                        self.authenticate_manifest(&entry.manifest)?,
                    ))
                })
                .collect::<BrainResult<BTreeMap<_, _>>>()?;
            let mut affected = BTreeSet::from([request.adapter.clone()]);
            loop {
                let before = affected.len();
                for (version, manifest) in &manifests {
                    if Self::composition_dependencies(manifest)
                        .iter()
                        .any(|dependency| affected.contains(dependency))
                    {
                        affected.insert(version.clone());
                    }
                }
                if affected.len() == before {
                    break;
                }
            }
            let next_revision = current
                .revision
                .checked_add(1)
                .ok_or_else(|| invalid("adapter_bank_revision_overflow"))?;
            for version in &affected {
                if current
                    .revocations
                    .iter()
                    .any(|revocation| revocation.adapter == *version)
                {
                    continue;
                }
                let entry = current
                    .entries
                    .iter()
                    .find(|entry| entry.adapter == *version)
                    .ok_or_else(|| integrity("adapter_revocation_closure_target_missing"))?;
                current.revocations.push(AdapterRevocation {
                    adapter: version.clone(),
                    manifest_sha256: entry.manifest.sha256.clone(),
                    revoked_at_revision: next_revision,
                    root_cause: request.adapter.clone(),
                    reason: request.reason.clone(),
                    evidence: request.evidence.clone(),
                });
            }
            current
                .revocations
                .sort_by(|left, right| left.adapter.cmp(&right.adapter));
            current
                .active
                .retain(|selection| !affected.contains(&selection.adapter));
            current.revision = next_revision;
            current.parent = current_reference;
            current.operation = AdapterBankOperation::Revoke {
                root: request.adapter.clone(),
                affected: affected.iter().cloned().collect(),
            };
            let index = self.publish_snapshot(&current)?;
            let manifest = current
                .entries
                .iter()
                .find(|entry| entry.adapter == request.adapter)
                .map(|entry| entry.manifest.clone());
            Ok(AdapterBankCommit {
                schema: ADAPTER_BANK_COMMIT_SCHEMA.to_string(),
                revision: current.revision,
                index,
                operation: current.operation,
                manifest,
                idempotent: false,
            })
        })
    }

    fn snapshot_at_revision(
        &self,
        mut reference: PrivateFileReference,
        target: u64,
    ) -> BrainResult<AdapterIndexSnapshot> {
        let mut visited = BTreeSet::new();
        for _ in 0..MAX_HISTORY_REVISIONS {
            if !visited.insert(reference.sha256.clone()) {
                return Err(integrity("adapter_bank_history_cycle"));
            }
            let snapshot = self.authenticate_snapshot(&reference)?;
            if snapshot.revision == target {
                return Ok(snapshot);
            }
            if snapshot.revision < target {
                return Err(invalid("adapter_rollback_target_not_ancestor"));
            }
            reference = snapshot
                .parent
                .ok_or_else(|| invalid("adapter_rollback_target_not_ancestor"))?;
        }
        Err(integrity("adapter_bank_history_limit_exceeded"))
    }

    pub fn rollback(&self, request: &AdapterRollbackRequest) -> BrainResult<AdapterBankCommit> {
        if request.schema != ADAPTER_ROLLBACK_REQUEST_SCHEMA || request.target_revision == 0 {
            return Err(invalid("adapter_rollback_request_invalid"));
        }
        validate_reason(&request.reason)?;
        if let Some(evidence) = &request.evidence {
            evidence.verify(&self.root)?;
        }
        let root = self.root.clone();
        let lock = self.lock_path();
        with_private_authority_lock(&root, &lock, || {
            let (mut current, current_reference) = self.current()?;
            Self::check_expectation(&request.expected, &current, current_reference.as_ref())?;
            let current_reference =
                current_reference.ok_or_else(|| invalid("adapter_rollback_requires_history"))?;
            if request.target_revision >= current.revision {
                return Err(invalid("adapter_rollback_target_not_ancestor"));
            }
            let target =
                self.snapshot_at_revision(current_reference.clone(), request.target_revision)?;
            let slot = AdapterActivationSlot {
                receiver_model: request.receiver_model.clone(),
                capability_id: request.capability_id.clone(),
            };
            let from = current
                .active
                .iter()
                .find(|selection| selection.slot == slot)
                .cloned();
            let to = target
                .active
                .iter()
                .find(|selection| selection.slot == slot)
                .cloned();
            if from.as_ref().map(|selection| &selection.adapter)
                == to.as_ref().map(|selection| &selection.adapter)
            {
                return Err(invalid("adapter_rollback_no_state_change"));
            }
            if let Some(selection) = &to {
                if current
                    .revocations
                    .iter()
                    .any(|revocation| revocation.adapter == selection.adapter)
                {
                    return Err(invalid("adapter_rollback_target_revoked"));
                }
                let manifest = self.authenticate_manifest(&selection.manifest)?;
                authenticate_live_receiver_model_profile(&self.root, &manifest.receiver_profile)?;
                let authorization_reference = request
                    .authorization
                    .as_ref()
                    .ok_or_else(|| invalid("adapter_rollback_fresh_authorization_required"))?;
                let authorization =
                    self.authenticate_promotion_authorization(authorization_reference)?;
                if authorization.adapter != selection.adapter
                    || authorization.manifest != selection.manifest
                    || authorization.issued_against_index != current_reference
                {
                    return Err(integrity("adapter_rollback_authorization_mismatch"));
                }
            } else if request.authorization.is_some() {
                return Err(invalid(
                    "adapter_rollback_deactivation_authorization_not_allowed",
                ));
            }
            let next_revision = current
                .revision
                .checked_add(1)
                .ok_or_else(|| invalid("adapter_bank_revision_overflow"))?;
            current.active.retain(|selection| selection.slot != slot);
            if let Some(mut selection) = to.clone() {
                selection.authorization = request
                    .authorization
                    .clone()
                    .ok_or_else(|| invalid("adapter_rollback_fresh_authorization_required"))?;
                selection.activated_at_revision = next_revision;
                selection.execution_binding.activation_epoch = next_revision;
                current.active.push(selection);
                current
                    .active
                    .sort_by(|left, right| left.slot.cmp(&right.slot));
            }
            current.revision = next_revision;
            current.parent = Some(current_reference);
            current.operation = AdapterBankOperation::Rollback {
                slot,
                target_revision: request.target_revision,
                from: from.map(|selection| selection.adapter),
                to: to.map(|selection| selection.adapter),
                authorization: request.authorization.clone(),
                reason: request.reason.clone(),
                evidence: request.evidence.clone(),
            };
            let index = self.publish_snapshot(&current)?;
            Ok(AdapterBankCommit {
                schema: ADAPTER_BANK_COMMIT_SCHEMA.to_string(),
                revision: current.revision,
                index,
                operation: current.operation,
                manifest: None,
                idempotent: false,
            })
        })
    }

    pub fn snapshot(&self) -> BrainResult<AdapterIndexSnapshot> {
        Ok(self.current()?.0)
    }

    pub fn query(&self, query: &AdapterBankQuery) -> BrainResult<AdapterBankReport> {
        if query.schema != ADAPTER_BANK_QUERY_SCHEMA
            || (query.model_sha256.is_some() && query.model_id.is_none())
        {
            return Err(invalid("adapter_bank_query_invalid"));
        }
        let (snapshot, reference) = self.current()?;
        let revoked = snapshot
            .revocations
            .iter()
            .map(|revocation| revocation.adapter.clone())
            .collect::<BTreeSet<_>>();
        let active = snapshot
            .active
            .iter()
            .map(|selection| selection.adapter.clone())
            .collect::<BTreeSet<_>>();
        let adapters = snapshot
            .entries
            .iter()
            .filter(|entry| {
                query
                    .capability_id
                    .as_ref()
                    .is_none_or(|value| value == &entry.capability_id)
                    && query
                        .model_id
                        .as_ref()
                        .is_none_or(|value| value == &entry.receiver_model.model_id)
                    && query
                        .model_sha256
                        .as_ref()
                        .is_none_or(|value| value == &entry.receiver_model.model_sha256)
                    && (query.include_revoked || !revoked.contains(&entry.adapter))
            })
            .map(|entry| AdapterStatus {
                adapter: entry.adapter.clone(),
                capability_id: entry.capability_id.clone(),
                lineage_id: entry.lineage_id.clone(),
                receiver_model: entry.receiver_model.clone(),
                manifest: entry.manifest.clone(),
                state: if revoked.contains(&entry.adapter) {
                    AdapterLifecycleState::Revoked
                } else if active.contains(&entry.adapter) {
                    AdapterLifecycleState::Active
                } else {
                    AdapterLifecycleState::RegisteredCandidate
                },
            })
            .collect();
        Ok(AdapterBankReport {
            schema: ADAPTER_BANK_REPORT_SCHEMA.to_string(),
            revision: snapshot.revision,
            index: reference,
            adapters,
        })
    }

    pub fn lookup(&self, lookup: &AdapterBankLookup) -> BrainResult<Option<AdapterManifest>> {
        if lookup.schema != ADAPTER_BANK_LOOKUP_SCHEMA
            || lookup.generation.is_some_and(|generation| generation == 0)
        {
            return Err(invalid("adapter_bank_lookup_invalid"));
        }
        let snapshot = self.snapshot()?;
        let entry = snapshot
            .entries
            .iter()
            .filter(|entry| entry.adapter.adapter_id == lookup.adapter_id)
            .filter(|entry| {
                lookup
                    .generation
                    .is_none_or(|generation| generation == entry.adapter.generation)
            })
            .max_by_key(|entry| entry.adapter.generation);
        entry
            .map(|entry| self.authenticate_manifest(&entry.manifest))
            .transpose()
    }

    pub fn resolve_active(
        &self,
        request: &AdapterResolutionRequest,
    ) -> BrainResult<AdapterExecutionResolution> {
        if request.schema != ADAPTER_RESOLUTION_REQUEST_SCHEMA {
            return Err(invalid("adapter_resolution_request_invalid"));
        }
        let (snapshot, index) = self.current()?;
        let index = index.ok_or_else(|| invalid("adapter_resolution_bank_empty"))?;
        let slot = AdapterActivationSlot {
            receiver_model: request.receiver_model.clone(),
            capability_id: request.capability_id.clone(),
        };
        let selection = snapshot
            .active
            .iter()
            .find(|selection| selection.slot == slot)
            .ok_or_else(|| invalid("adapter_resolution_not_active"))?;
        let manifest = self.authenticate_manifest(&selection.manifest)?;
        authenticate_live_receiver_model_profile(&self.root, &manifest.receiver_profile)?;
        let authorization = self.authenticate_promotion_authorization(&selection.authorization)?;
        let materialized_candidate = if authorization.governed {
            Some(self.authenticate_candidate_materialization(
                &authorization.evidence.materialization_receipt,
            )?)
        } else {
            None
        };
        Ok(AdapterExecutionResolution {
            schema: ADAPTER_EXECUTION_RESOLUTION_SCHEMA.to_string(),
            bank_revision: snapshot.revision,
            index,
            slot,
            adapter: selection.adapter.clone(),
            manifest: selection.manifest.clone(),
            authorization: selection.authorization.clone(),
            execution_binding: selection.execution_binding.clone(),
            materialized_candidate,
        })
    }

    pub fn authenticate_execution_resolution(
        &self,
        resolution: &AdapterExecutionResolution,
    ) -> BrainResult<()> {
        if resolution.schema != ADAPTER_EXECUTION_RESOLUTION_SCHEMA {
            return Err(invalid("adapter_execution_resolution_schema_invalid"));
        }
        let request = AdapterResolutionRequest {
            schema: ADAPTER_RESOLUTION_REQUEST_SCHEMA.to_string(),
            receiver_model: resolution.slot.receiver_model.clone(),
            capability_id: resolution.slot.capability_id.clone(),
        };
        let current = self.resolve_active(&request)?;
        if &current != resolution {
            return Err(integrity("adapter_execution_resolution_stale"));
        }
        Ok(())
    }

    pub fn verify_history(&self) -> BrainResult<AdapterBankHistoryStatus> {
        let journal = self.authenticate_revision_journal()?;
        let (current, reference) = self.current()?;
        let Some(mut cursor) = reference.clone() else {
            return Ok(AdapterBankHistoryStatus {
                revision: 0,
                index: None,
                verified_revision_count: 0,
                registered_adapter_count: 0,
                active_adapter_count: 0,
                revoked_adapter_count: 0,
            });
        };
        let mut expected_revision = current.revision;
        let mut visited = BTreeSet::new();
        let mut count = 0usize;
        loop {
            if count >= MAX_HISTORY_REVISIONS || !visited.insert(cursor.sha256.clone()) {
                return Err(integrity("adapter_bank_history_invalid"));
            }
            let snapshot = self.authenticate_snapshot(&cursor)?;
            if snapshot.revision != expected_revision {
                return Err(integrity("adapter_bank_history_revision_gap"));
            }
            let offset = usize::try_from(expected_revision - 1)
                .map_err(|_| integrity("adapter_bank_history_revision_gap"))?;
            let (commit, _) = journal
                .get(offset)
                .ok_or_else(|| integrity("adapter_bank_history_commit_missing"))?;
            if commit.index != cursor {
                return Err(integrity("adapter_bank_history_commit_index_mismatch"));
            }
            count += 1;
            if expected_revision == 1 {
                if snapshot.parent.is_some() {
                    return Err(integrity("adapter_bank_history_genesis_parent"));
                }
                break;
            }
            cursor = snapshot
                .parent
                .ok_or_else(|| integrity("adapter_bank_history_parent_missing"))?;
            expected_revision -= 1;
        }
        if count != journal.len() {
            return Err(integrity("adapter_bank_history_journal_length_mismatch"));
        }
        let mut parent = Self::empty_snapshot();
        let mut parent_reference = None;
        for (commit, _) in &journal {
            let snapshot = self.authenticate_snapshot(&commit.index)?;
            self.validate_snapshot_transition(parent_reference.as_ref(), &parent, &snapshot)?;
            parent_reference = Some(commit.index.clone());
            parent = snapshot;
        }
        Ok(AdapterBankHistoryStatus {
            revision: current.revision,
            index: reference,
            verified_revision_count: count,
            registered_adapter_count: current.entries.len(),
            active_adapter_count: current.active.len(),
            revoked_adapter_count: current.revocations.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
    };
    use crate::capability_bundle::CapabilityBundleStatus;
    use crate::capability_ir::{
        CapabilityIr, IrNode, OutputBinding, PrimitiveSet, TypedPort, ValueReference, ValueType,
    };
    use crate::content_vault::capture_to_vault;
    use crate::finite::FiniteF64;
    use crate::identity::{AcquisitionId, CapabilityNodeId, PortId, PrimitiveId};
    use crate::model_adaptation::{
        profile_receiver_model, ReceiverModelProfileInput, RECEIVER_MODEL_PROFILE_INPUT_SCHEMA,
    };
    use crate::portfolio_governance::{
        decide_candidate, evaluate_paired_groups, evaluate_petfc, CanaryPolicy, CanaryStage,
        CandidateGatePolicy, EvidenceId, HardInvariant, IndependenceGroupId, MetricDirection,
        MetricId, MetricSpec, ObservationWindow, PairId, PairedEvaluationReport,
        PairedExperimentalUnit, PairedObservation, PetfcConservationLimits, PetfcMetricPolicy,
        PetfcPathLimits, PetfcPolicy, PetfcTrajectory, PetfcUtilityPolicy, RobustEvaluationPolicy,
    };
    use crate::security::secure_dir;
    use serde_json::{json, Map, Value};
    use std::fs;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        base: PathBuf,
        profile: PrivateFileReference,
    }

    impl Fixture {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "tidex-adapter-bank-{label}-{}-{}-{nonce}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            secure_dir(&root).unwrap();
            let base = root.join("base.safetensors");
            let model_config = root.join("model_config.json");
            let tokenizer = root.join("tokenizer.json");
            Self::write_base(&base);
            fs::write(
                &model_config,
                serde_json::to_vec(&json!({
                    "model_type":"tidex-test",
                    "architectures":["TidexForCausalLM"],
                    "num_hidden_layers":1,
                    "hidden_size":2,
                    "vocab_size":16,
                    "max_position_embeddings":128,
                    "tie_word_embeddings":false
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(
                &tokenizer,
                b"{\"version\":\"1\",\"model\":{\"type\":\"test\"}}",
            )
            .unwrap();
            let profile = profile_receiver_model(
                &root,
                &ReceiverModelProfileInput {
                    schema: RECEIVER_MODEL_PROFILE_INPUT_SCHEMA.to_string(),
                    model_id: ModelId::parse("receiver.test").unwrap(),
                    architecture_id: ArchitectureId::parse("architecture.test").unwrap(),
                    source_revision: None,
                    checkpoint_path: base.clone(),
                    config_path: model_config,
                    tokenizer_path: tokenizer,
                },
            )
            .unwrap()
            .profile_reference;
            Self {
                root,
                base,
                profile,
            }
        }

        fn write_safetensors(path: &Path, tensors: Vec<(&str, Vec<f32>, Vec<usize>)>) {
            let mut data = Vec::new();
            let mut header = Map::new();
            header.insert("__metadata__".to_string(), json!({"format":"pt"}));
            for (name, values, shape) in tensors {
                let start = data.len();
                for value in values {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                header.insert(
                    name.to_string(),
                    json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
                );
            }
            let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
            let padding = (8 - header.len() % 8) % 8;
            header.extend(std::iter::repeat_n(b' ', padding));
            let mut output = File::create(path).unwrap();
            output
                .write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            output.write_all(&header).unwrap();
            output.write_all(&data).unwrap();
            output.sync_all().unwrap();
        }

        fn write_base(path: &Path) {
            Self::write_safetensors(
                path,
                vec![
                    (
                        "model.layers.0.self_attn.q_proj.weight",
                        vec![1.0, 2.0, 3.0, 4.0],
                        vec![2, 2],
                    ),
                    (
                        "model.layers.0.self_attn.v_proj.weight",
                        vec![-1.0, -2.0, -3.0, -4.0],
                        vec![2, 2],
                    ),
                ],
            );
        }

        fn adapter(&self, label: &str, offset: f32) -> (PathBuf, PathBuf) {
            let adapter = self.root.join(format!("{label}.safetensors"));
            let config = self.root.join(format!("{label}.json"));
            Self::write_safetensors(
                &adapter,
                vec![
                    (
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                        vec![1.0 + offset, 2.0],
                        vec![1, 2],
                    ),
                    (
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                        vec![3.0, 4.0 + offset],
                        vec![2, 1],
                    ),
                    (
                        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
                        vec![1.0, -1.0 - offset],
                        vec![1, 2],
                    ),
                    (
                        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
                        vec![2.0 + offset, 3.0],
                        vec![2, 1],
                    ),
                ],
            );
            fs::write(
                &config,
                serde_json::to_vec(&json!({
                    "r":1,
                    "lora_alpha":2.0,
                    "target_modules":["q_proj","v_proj"],
                    "bias":"none",
                    "use_rslora":false,
                    "use_dora":false,
                    "fan_in_fan_out":false
                }))
                .unwrap(),
            )
            .unwrap();
            (adapter, config)
        }

        fn import_request(
            &self,
            adapter_id: &str,
            lineage_id: &str,
            adapter_path: PathBuf,
            config_path: PathBuf,
            expected: AdapterBankExpectation,
        ) -> AdapterImportRequest {
            AdapterImportRequest {
                schema: ADAPTER_IMPORT_REQUEST_SCHEMA.to_string(),
                adapter: AdapterVersion {
                    adapter_id: PatchId::parse(adapter_id).unwrap(),
                    generation: 1,
                },
                capability_id: CapabilityId::parse("capability.test:v1").unwrap(),
                lineage_id: LineageId::parse(lineage_id).unwrap(),
                capability_bundle: None,
                receiver_profile: self.profile.clone(),
                lora: LoraAdapterAxisInput {
                    schema: crate::weight_actuator::LORA_ADAPTER_AXIS_INPUT_SCHEMA.to_string(),
                    base_model_path: self.base.clone(),
                    adapter_model_path: adapter_path,
                    adapter_config_path: config_path,
                },
                expected,
            }
        }

        fn closed_capability_bundle(&self) -> PrivateFileReference {
            let donor = self.root.with_extension("governance-donor");
            let _ = fs::remove_dir_all(&donor);
            fs::create_dir_all(donor.join("src")).unwrap();
            fs::write(donor.join("src/capability.rs"), b"pub fn apply() {}\n").unwrap();
            let acquisition = AcquisitionRequest::new(
                AcquisitionId::parse("adapter-bank-governance-acquisition.v1").unwrap(),
                AcquisitionScope::WholeProject,
                RequestedResidency::BestVerified,
                NoisePolicy::ConservativeGeneratedArtifacts,
                AcquisitionBudget {
                    max_files: 16,
                    max_total_bytes: 1 << 20,
                },
                vec![],
            )
            .unwrap();
            let capture = capture_to_vault(&donor, &self.root, &acquisition).unwrap();
            fs::remove_dir_all(&donor).unwrap();
            let capability_id = CapabilityId::parse("capability.test:v1").unwrap();
            let ir = CapabilityIr::new(
                capability_id.clone(),
                capture.envelope(),
                PrimitiveSet::tidex_core_v1().unwrap(),
                vec![TypedPort::tensor_f64(PortId::parse("scores").unwrap(), vec![4]).unwrap()],
                vec![IrNode::new(
                    CapabilityNodeId::parse("node.apply").unwrap(),
                    PrimitiveId::parse("select.arg_max").unwrap(),
                    vec![ValueReference::Input {
                        name: PortId::parse("scores").unwrap(),
                    }],
                    TypedPort::scalar(PortId::parse("choice").unwrap(), ValueType::I64).unwrap(),
                    vec![PathBuf::from("src/capability.rs")],
                )
                .unwrap()],
                vec![OutputBinding::new(
                    TypedPort::scalar(PortId::parse("selected").unwrap(), ValueType::I64).unwrap(),
                    ValueReference::NodeOutput {
                        node_id: CapabilityNodeId::parse("node.apply").unwrap(),
                    },
                )
                .unwrap()],
            )
            .unwrap();
            let ir_ref = ir.persist(&self.root, capture.envelope()).unwrap();
            let capture_ref = capture.persist(&self.root).unwrap();
            let bundle = crate::capability_bundle::CapabilityBundle::create_closed(
                &self.root,
                capability_id,
                capture_ref,
                ir_ref,
                BTreeSet::new(),
            )
            .unwrap();
            assert_eq!(
                bundle.status(),
                CapabilityBundleStatus::RepresentationClosed
            );
            bundle.persist(&self.root).unwrap()
        }

        fn evidence(&self, label: &str) -> PrivateFileReference {
            let bytes = format!("verified-evidence:{label}").into_bytes();
            let digest = Sha256Digest::digest_bytes(&bytes);
            let path = self
                .root
                .join("state/test-governance")
                .join(format!("{label}-{digest}.json"));
            write_or_verify_immutable(&self.root, &path, &bytes).unwrap();
            PrivateFileReference::new(path, digest)
        }

        fn promotion_evidence(&self, label: &str) -> AdapterPromotionEvidence {
            AdapterPromotionEvidence {
                independent_execution: self.evidence(&format!("{label}-execution")),
                preservation_assessment: self.evidence(&format!("{label}-preservation")),
                negative_controls: self.evidence(&format!("{label}-negative")),
                materialization_receipt: self.evidence(&format!("{label}-materialization")),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn initial_expectation() -> AdapterBankExpectation {
        AdapterBankExpectation {
            revision: 0,
            index_sha256: None,
        }
    }

    fn expectation(commit: &AdapterBankCommit) -> AdapterBankExpectation {
        AdapterBankExpectation {
            revision: commit.revision,
            index_sha256: Some(commit.index.sha256.clone()),
        }
    }

    fn allowed_decision() -> PromotionDecision {
        PromotionDecision {
            allowed: true,
            reasons: Vec::new(),
            metrics: BTreeMap::from([
                ("independent_execution_pass_rate".to_string(), 1.0),
                ("preservation_floor".to_string(), 1.0),
                ("negative_control_leakage".to_string(), 0.0),
            ]),
        }
    }

    fn seal(
        bank: &AdapterBank,
        fixture: &Fixture,
        manifest: PrivateFileReference,
        current: &AdapterBankCommit,
        label: &str,
    ) -> PrivateFileReference {
        bank.seal_verified_promotion(&VerifiedAdapterPromotion {
            manifest,
            evidence: fixture.promotion_evidence(label),
            decision: allowed_decision(),
            expected: expectation(current),
        })
        .unwrap()
    }

    fn governance_metric(value: &str) -> MetricId {
        MetricId::parse(value).unwrap()
    }

    fn governance_specs() -> Vec<MetricSpec> {
        vec![
            MetricSpec::new(
                governance_metric("quality"),
                MetricDirection::Maximize,
                Some(HardInvariant::at_least(0.0).unwrap()),
            )
            .unwrap(),
            MetricSpec::new(
                governance_metric("loss"),
                MetricDirection::Minimize,
                Some(HardInvariant::at_most(1.0).unwrap()),
            )
            .unwrap(),
        ]
    }

    fn governance_rows(
        groups: usize,
        baseline_quality: f64,
        candidate_quality: f64,
        baseline_loss: f64,
        candidate_loss: f64,
    ) -> Vec<(f64, f64, f64, f64)> {
        vec![
            (
                baseline_quality,
                candidate_quality,
                baseline_loss,
                candidate_loss
            );
            groups
        ]
    }

    fn governance_report(
        specs: &[MetricSpec],
        baseline: &str,
        candidate: &str,
        start_tick: u64,
        rows: &[(f64, f64, f64, f64)],
        minimum_groups: usize,
    ) -> PairedEvaluationReport {
        let mut observations = Vec::new();
        for (index, (baseline_quality, candidate_quality, baseline_loss, candidate_loss)) in
            rows.iter().copied().enumerate()
        {
            let unit = PairedExperimentalUnit::new(
                IndependenceGroupId::parse(format!("group-{index}")).unwrap(),
                PairId::parse(format!("pair-{start_tick}-{index}")).unwrap(),
                EvidenceId::parse(format!("evidence-{start_tick}-{index}")).unwrap(),
                ObservationWindow::new(start_tick, start_tick + 1).unwrap(),
            );
            observations.push(
                PairedObservation::new(
                    governance_metric("quality"),
                    unit.clone(),
                    baseline_quality,
                    candidate_quality,
                )
                .unwrap(),
            );
            observations.push(
                PairedObservation::new(
                    governance_metric("loss"),
                    unit,
                    baseline_loss,
                    candidate_loss,
                )
                .unwrap(),
            );
        }
        evaluate_paired_groups(
            VariantId::parse(baseline).unwrap(),
            VariantId::parse(candidate).unwrap(),
            specs,
            &observations,
            &RobustEvaluationPolicy::new(minimum_groups, minimum_groups.min(3), 1_000).unwrap(),
        )
        .unwrap()
    }

    fn governance_gate_policy(specs: &[MetricSpec], minimum_groups: usize) -> CandidateGatePolicy {
        CandidateGatePolicy::new(
            specs,
            minimum_groups,
            1.0,
            BTreeMap::from([
                (governance_metric("loss"), FiniteF64::new(0.0).unwrap()),
                (governance_metric("quality"), FiniteF64::new(0.0).unwrap()),
            ]),
        )
        .unwrap()
    }

    fn governance_petfc_policy(specs: &[MetricSpec]) -> PetfcPolicy {
        PetfcPolicy::new(
            specs,
            governance_metric("quality"),
            vec![
                PetfcMetricPolicy::new(governance_metric("quality"), 1.0, 0.0).unwrap(),
                PetfcMetricPolicy::new(governance_metric("loss"), 1.0, 0.0).unwrap(),
            ],
            PetfcPathLimits::new(2, 8, 2.0, 2.0, 0.6, 0.0).unwrap(),
            PetfcConservationLimits::new(0.0, 0, 2.0).unwrap(),
            PetfcUtilityPolicy::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        )
        .unwrap()
    }

    fn governed_witness_chain(
        candidate: &VariantId,
    ) -> (CandidateGateDecision, PetfcAssessment, CanaryState) {
        let specs = governance_specs();
        let candidate_name = candidate.as_str();
        let first = governance_report(
            &specs,
            "baseline",
            "adapter-middle",
            100,
            &governance_rows(3, 0.2, 0.4, 0.8, 0.6),
            3,
        );
        let second = governance_report(
            &specs,
            "baseline",
            candidate_name,
            110,
            &governance_rows(3, 0.2, 0.7, 0.8, 0.3),
            3,
        );
        let gate_report = governance_report(
            &specs,
            "baseline",
            candidate_name,
            120,
            &governance_rows(3, 0.2, 0.7, 0.8, 0.3),
            3,
        );
        let gate =
            decide_candidate(&specs, &gate_report, &governance_gate_policy(&specs, 3)).unwrap();
        let petfc_policy = governance_petfc_policy(&specs);
        let trajectory = PetfcTrajectory::start(&first, &petfc_policy)
            .unwrap()
            .append_report(&second)
            .unwrap();
        let petfc = evaluate_petfc(&specs, &trajectory, &petfc_policy).unwrap();

        let canary_policy = CanaryPolicy::new(
            &specs,
            Sha256Digest::digest_bytes(b"adapter-bank-governed-canary-salt"),
            vec![
                CanaryStage::new(100_000, 3).unwrap(),
                CanaryStage::new(500_000, 4).unwrap(),
            ],
            1.0,
            BTreeMap::from([
                (governance_metric("loss"), FiniteF64::new(0.0).unwrap()),
                (governance_metric("quality"), FiniteF64::new(0.0).unwrap()),
            ]),
        )
        .unwrap();
        let first_canary_tick = gate.source_observation_window().end_tick() + 10;
        let first_canary = governance_report(
            &specs,
            "baseline",
            candidate_name,
            first_canary_tick,
            &governance_rows(3, 0.2, 0.7, 0.8, 0.3),
            3,
        );
        let state = CanaryState::start(&gate, &canary_policy)
            .unwrap()
            .evaluate_stage(&specs, &first_canary, &canary_policy)
            .unwrap();
        let second_canary = governance_report(
            &specs,
            "baseline",
            candidate_name,
            first_canary_tick + 10,
            &governance_rows(4, 0.2, 0.7, 0.8, 0.3),
            3,
        );
        let canary = state
            .evaluate_stage(&specs, &second_canary, &canary_policy)
            .unwrap();
        (gate, petfc, canary)
    }

    #[test]
    fn import_is_idempotent_indexed_and_independent_of_original_peft_paths() {
        let fixture = Fixture::new("import");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter, config) = fixture.adapter("axis-a", 0.0);
        let request = fixture.import_request(
            "adapter-a",
            "lineage-a",
            adapter.clone(),
            config.clone(),
            initial_expectation(),
        );
        let first = bank.import_lora(&request).unwrap();
        assert_eq!(first.revision, 1);
        assert!(!first.idempotent);
        let replay = bank.import_lora(&request).unwrap();
        assert_eq!(replay.revision, 1);
        assert!(replay.idempotent);

        fs::remove_file(adapter).unwrap();
        fs::remove_file(config).unwrap();
        let report = bank
            .query(&AdapterBankQuery {
                schema: ADAPTER_BANK_QUERY_SCHEMA.to_string(),
                capability_id: Some(CapabilityId::parse("capability.test:v1").unwrap()),
                model_id: Some(ModelId::parse("receiver.test").unwrap()),
                model_sha256: None,
                include_revoked: false,
            })
            .unwrap();
        assert_eq!(report.adapters.len(), 1);
        assert_eq!(
            report.adapters[0].state,
            AdapterLifecycleState::RegisteredCandidate
        );
        let manifest = bank
            .lookup(&AdapterBankLookup {
                schema: ADAPTER_BANK_LOOKUP_SCHEMA.to_string(),
                adapter_id: PatchId::parse("adapter-a").unwrap(),
                generation: None,
            })
            .unwrap()
            .unwrap();
        match manifest.source {
            AdapterManifestSource::ImportedPeftLora {
                adapter_model,
                adapter_config,
                ..
            } => {
                adapter_model.artifact.verify(&fixture.root).unwrap();
                adapter_config.artifact.verify(&fixture.root).unwrap();
            }
            _ => panic!("expected imported PEFT source"),
        }
        assert_eq!(bank.verify_history().unwrap().verified_revision_count, 1);
    }

    #[test]
    fn exact_composition_is_canonical_and_revocation_closes_the_dependency_dag() {
        let fixture = Fixture::new("composition");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter_a, config_a) = fixture.adapter("axis-a", 0.0);
        let first = bank
            .import_lora(&fixture.import_request(
                "adapter-a",
                "lineage-a",
                adapter_a,
                config_a,
                initial_expectation(),
            ))
            .unwrap();
        let (adapter_b, config_b) = fixture.adapter("axis-b", 0.5);
        let second = bank
            .import_lora(&fixture.import_request(
                "adapter-b",
                "lineage-b",
                adapter_b,
                config_b,
                expectation(&first),
            ))
            .unwrap();
        let composition = AdapterCompositionRequest {
            schema: ADAPTER_COMPOSITION_REQUEST_SCHEMA.to_string(),
            adapter: AdapterVersion {
                adapter_id: PatchId::parse("adapter-composite").unwrap(),
                generation: 1,
            },
            capability_id: CapabilityId::parse("capability.composite:v1").unwrap(),
            lineage_id: LineageId::parse("lineage-composite").unwrap(),
            capability_bundle: None,
            terms: vec![
                AdapterCompositionInputTerm {
                    adapter: AdapterVersion {
                        adapter_id: PatchId::parse("adapter-b").unwrap(),
                        generation: 1,
                    },
                    coefficient: -0.25,
                },
                AdapterCompositionInputTerm {
                    adapter: AdapterVersion {
                        adapter_id: PatchId::parse("adapter-a").unwrap(),
                        generation: 1,
                    },
                    coefficient: 1.5,
                },
            ],
            expected: expectation(&second),
        };
        let composed = bank.compose_exact(&composition).unwrap();
        assert_eq!(composed.revision, 3);
        let manifest = bank
            .authenticate_manifest(composed.manifest.as_ref().unwrap())
            .unwrap();
        let AdapterManifestSource::ExactDenseComposition { terms, .. } = manifest.source else {
            panic!("expected exact composition");
        };
        assert!(terms
            .windows(2)
            .all(|window| window[0].manifest.sha256 < window[1].manifest.sha256));
        verify_dvec_reference_under_root(&fixture.root, &manifest.dense_delta).unwrap();

        let replay = bank.compose_exact(&composition).unwrap();
        assert!(replay.idempotent);
        assert_eq!(replay.revision, 3);

        let revoked = bank
            .revoke(&AdapterRevocationRequest {
                schema: ADAPTER_REVOCATION_REQUEST_SCHEMA.to_string(),
                adapter: AdapterVersion {
                    adapter_id: PatchId::parse("adapter-a").unwrap(),
                    generation: 1,
                },
                reason: "source failed a later independent safety audit".to_string(),
                evidence: Some(fixture.evidence("revoke-a")),
                expected: expectation(&composed),
            })
            .unwrap();
        assert_eq!(revoked.revision, 4);
        let snapshot = bank.snapshot().unwrap();
        let revoked_versions = snapshot
            .revocations
            .iter()
            .map(|revocation| revocation.adapter.adapter_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            revoked_versions,
            BTreeSet::from(["adapter-a", "adapter-composite"])
        );
        let mut retry = composition;
        retry.adapter.adapter_id = PatchId::parse("adapter-other").unwrap();
        retry.lineage_id = LineageId::parse("lineage-other").unwrap();
        retry.expected = expectation(&revoked);
        assert!(bank.compose_exact(&retry).is_err());
    }

    #[test]
    fn candidate_materialization_is_bound_candidate_only_and_confined() {
        let fixture = Fixture::new("materialization");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter, config) = fixture.adapter("axis-a", 0.0);
        let imported = bank
            .import_lora(&fixture.import_request(
                "adapter-a",
                "lineage-a",
                adapter,
                config,
                initial_expectation(),
            ))
            .unwrap();
        let output_path = fixture.root.join("materialized-adapter-a.safetensors");
        let request = AdapterCandidateMaterializationRequest {
            schema: ADAPTER_CANDIDATE_MATERIALIZATION_REQUEST_SCHEMA.to_string(),
            adapter: AdapterVersion {
                adapter_id: PatchId::parse("adapter-a").unwrap(),
                generation: 1,
            },
            output_path: output_path.clone(),
            expected: expectation(&imported),
        };
        let mut escaped = request.clone();
        escaped.output_path = std::env::temp_dir().join("tidex-adapter-bank-escape.safetensors");
        assert!(bank.materialize_candidate(&escaped).is_err());

        let reference = bank.materialize_candidate(&request).unwrap();
        let record = bank
            .authenticate_candidate_materialization(&reference)
            .unwrap();
        assert_eq!(record.observed_revision, imported.revision);
        assert_eq!(record.observed_index, imported.index);
        assert_eq!(record.output_path, output_path);
        assert!(record.candidate_only);
        assert!(!record.authorizes_promotion);
        assert!(!record.receipt.authorizes_promotion);
        assert!(!record.receipt.requires_adapter_at_runtime);

        let manifest_reference = imported.manifest.as_ref().unwrap();
        let manifest = bank.authenticate_manifest(manifest_reference).unwrap();
        assert!(manifest.capability_bundle.is_none());
        assert!(bank
            .require_promotable_capability_bundle(&manifest)
            .unwrap_err()
            .to_string()
            .contains("adapter_promotion_capability_bundle_missing"));
        assert_eq!(
            bank.governance_candidate_id(manifest_reference)
                .unwrap()
                .as_str(),
            format!("adapter.manifest.{}", manifest_reference.sha256)
        );
    }

    #[test]
    fn governed_authorization_request_replays_real_witnesses_and_activates() {
        let fixture = Fixture::new("governed-production-path");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let capability_bundle = fixture.closed_capability_bundle();
        let (adapter_model, adapter_config) = fixture.adapter("governed", 0.25);
        let mut import = fixture.import_request(
            "adapter.governed",
            "lineage.governed",
            adapter_model,
            adapter_config,
            initial_expectation(),
        );
        import.capability_bundle = Some(capability_bundle);
        let adapter = import.adapter.clone();
        let imported = bank.import_lora(&import).unwrap();
        let manifest_ref = imported.manifest.clone().unwrap();

        let materialization = bank
            .materialize_candidate(&AdapterCandidateMaterializationRequest {
                schema: ADAPTER_CANDIDATE_MATERIALIZATION_REQUEST_SCHEMA.to_string(),
                adapter: adapter.clone(),
                output_path: fixture.root.join("governed-candidate.safetensors"),
                expected: expectation(&imported),
            })
            .unwrap();
        let candidate_id = bank.governance_candidate_id(&manifest_ref).unwrap();
        let (gate, petfc, canary) = governed_witness_chain(&candidate_id);
        authenticate_adapter_promotion_witnesses(&candidate_id, &gate, &petfc, &canary).unwrap();
        let request = AdapterGovernedPromotionRequest {
            schema: ADAPTER_GOVERNED_PROMOTION_REQUEST_SCHEMA.to_string(),
            materialization: materialization.clone(),
            candidate_gate: gate.persist(&fixture.root).unwrap(),
            petfc_assessment: petfc.persist(&fixture.root).unwrap(),
            canary_state: canary.persist(&fixture.root).unwrap(),
            expected: expectation(&imported),
        };

        let authorization = bank.authorize_governed_promotion_request(&request).unwrap();
        let activated = bank
            .activate(&AdapterActivationRequest {
                schema: ADAPTER_ACTIVATION_REQUEST_SCHEMA.to_string(),
                adapter: adapter.clone(),
                authorization: authorization.clone(),
                expected: expectation(&imported),
            })
            .unwrap();
        assert_eq!(activated.revision, imported.revision + 1);

        let manifest = bank.authenticate_manifest(&manifest_ref).unwrap();
        let resolution = bank
            .resolve_active(&AdapterResolutionRequest {
                schema: ADAPTER_RESOLUTION_REQUEST_SCHEMA.to_string(),
                receiver_model: manifest.receiver_model.clone(),
                capability_id: manifest.capability_id.clone(),
            })
            .unwrap();
        assert_eq!(resolution.authorization, authorization);
        assert_eq!(resolution.adapter, adapter);
        assert!(resolution.materialized_candidate.is_some());
        bank.authenticate_execution_resolution(&resolution).unwrap();

        assert!(bank.authorize_governed_promotion_request(&request).is_err());
        let history = bank.verify_history().unwrap();
        assert_eq!(history.revision, activated.revision);
        assert_eq!(history.active_adapter_count, 1);
    }

    #[test]
    fn governed_activation_rollback_and_epoch_fencing_are_forward_only() {
        let fixture = Fixture::new("lifecycle");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter_a, config_a) = fixture.adapter("axis-a", 0.0);
        let first = bank
            .import_lora(&fixture.import_request(
                "adapter-a",
                "lineage-a",
                adapter_a,
                config_a,
                initial_expectation(),
            ))
            .unwrap();
        let (adapter_b, config_b) = fixture.adapter("axis-b", 0.75);
        let second = bank
            .import_lora(&fixture.import_request(
                "adapter-b",
                "lineage-b",
                adapter_b,
                config_b,
                expectation(&first),
            ))
            .unwrap();

        let permit_a = seal(
            &bank,
            &fixture,
            first.manifest.clone().unwrap(),
            &second,
            "permit-a",
        );
        let active_a = bank
            .activate(&AdapterActivationRequest {
                schema: ADAPTER_ACTIVATION_REQUEST_SCHEMA.to_string(),
                adapter: AdapterVersion {
                    adapter_id: PatchId::parse("adapter-a").unwrap(),
                    generation: 1,
                },
                authorization: permit_a.clone(),
                expected: expectation(&second),
            })
            .unwrap();
        assert_eq!(active_a.revision, 3);

        let permit_b = seal(
            &bank,
            &fixture,
            second.manifest.clone().unwrap(),
            &active_a,
            "permit-b",
        );
        let active_b = bank
            .activate(&AdapterActivationRequest {
                schema: ADAPTER_ACTIVATION_REQUEST_SCHEMA.to_string(),
                adapter: AdapterVersion {
                    adapter_id: PatchId::parse("adapter-b").unwrap(),
                    generation: 1,
                },
                authorization: permit_b,
                expected: expectation(&active_a),
            })
            .unwrap();
        assert_eq!(active_b.revision, 4);

        let receiver_model = bank
            .authenticate_manifest(first.manifest.as_ref().unwrap())
            .unwrap()
            .receiver_model;
        let stale_error = bank
            .rollback(&AdapterRollbackRequest {
                schema: ADAPTER_ROLLBACK_REQUEST_SCHEMA.to_string(),
                receiver_model: receiver_model.clone(),
                capability_id: CapabilityId::parse("capability.test:v1").unwrap(),
                target_revision: 3,
                reason: "stale permission must not authorize rollback".to_string(),
                evidence: Some(fixture.evidence("rollback-stale")),
                authorization: Some(permit_a.clone()),
                expected: expectation(&active_b),
            })
            .unwrap_err()
            .to_string();
        assert!(stale_error.contains("adapter_rollback_authorization_mismatch"));
        let rollback_permit_a = seal(
            &bank,
            &fixture,
            first.manifest.clone().unwrap(),
            &active_b,
            "rollback-permit-a",
        );

        let rolled_back = bank
            .rollback(&AdapterRollbackRequest {
                schema: ADAPTER_ROLLBACK_REQUEST_SCHEMA.to_string(),
                receiver_model: receiver_model.clone(),
                capability_id: CapabilityId::parse("capability.test:v1").unwrap(),
                target_revision: 3,
                reason: "new generation regressed the sealed preservation suite".to_string(),
                evidence: Some(fixture.evidence("rollback")),
                authorization: Some(rollback_permit_a.clone()),
                expected: expectation(&active_b),
            })
            .unwrap();
        assert_eq!(rolled_back.revision, 5);
        let request = AdapterResolutionRequest {
            schema: ADAPTER_RESOLUTION_REQUEST_SCHEMA.to_string(),
            receiver_model: bank
                .authenticate_manifest(first.manifest.as_ref().unwrap())
                .unwrap()
                .receiver_model,
            capability_id: CapabilityId::parse("capability.test:v1").unwrap(),
        };
        let resolution = bank.resolve_active(&request).unwrap();
        assert_eq!(resolution.adapter.adapter_id.as_str(), "adapter-a");
        assert_eq!(resolution.bank_revision, 5);
        assert_eq!(resolution.execution_binding.activation_epoch, 5);
        assert!(resolution.materialized_candidate.is_none());
        bank.authenticate_execution_resolution(&resolution).unwrap();

        let revoked = bank
            .revoke(&AdapterRevocationRequest {
                schema: ADAPTER_REVOCATION_REQUEST_SCHEMA.to_string(),
                adapter: resolution.adapter.clone(),
                reason: "adapter authorization was revoked".to_string(),
                evidence: Some(fixture.evidence("revocation")),
                expected: expectation(&rolled_back),
            })
            .unwrap();
        assert_eq!(revoked.revision, 6);
        assert!(bank.authenticate_execution_resolution(&resolution).is_err());
        assert!(bank
            .rollback(&AdapterRollbackRequest {
                schema: ADAPTER_ROLLBACK_REQUEST_SCHEMA.to_string(),
                receiver_model: request.receiver_model,
                capability_id: request.capability_id,
                target_revision: 3,
                reason: "attempted resurrection".to_string(),
                evidence: None,
                authorization: None,
                expected: expectation(&revoked),
            })
            .is_err());
    }

    #[test]
    fn concurrent_activation_has_one_cas_winner_and_no_head_fork() {
        let fixture = Fixture::new("concurrency");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter_a, config_a) = fixture.adapter("axis-a", 0.0);
        let first = bank
            .import_lora(&fixture.import_request(
                "adapter-a",
                "lineage-a",
                adapter_a,
                config_a,
                initial_expectation(),
            ))
            .unwrap();
        let (adapter_b, config_b) = fixture.adapter("axis-b", 0.3);
        let second = bank
            .import_lora(&fixture.import_request(
                "adapter-b",
                "lineage-b",
                adapter_b,
                config_b,
                expectation(&first),
            ))
            .unwrap();
        let permit_a = seal(
            &bank,
            &fixture,
            first.manifest.clone().unwrap(),
            &second,
            "concurrent-a",
        );
        let permit_b = seal(
            &bank,
            &fixture,
            second.manifest.clone().unwrap(),
            &second,
            "concurrent-b",
        );
        let barrier = Arc::new(Barrier::new(3));
        let requests = [
            (
                AdapterVersion {
                    adapter_id: PatchId::parse("adapter-a").unwrap(),
                    generation: 1,
                },
                permit_a,
            ),
            (
                AdapterVersion {
                    adapter_id: PatchId::parse("adapter-b").unwrap(),
                    generation: 1,
                },
                permit_b,
            ),
        ];
        let mut handles = Vec::new();
        for (adapter, authorization) in requests {
            let bank = bank.clone();
            let barrier = barrier.clone();
            let expected = expectation(&second);
            handles.push(thread::spawn(move || {
                barrier.wait();
                bank.activate(&AdapterActivationRequest {
                    schema: ADAPTER_ACTIVATION_REQUEST_SCHEMA.to_string(),
                    adapter,
                    authorization,
                    expected,
                })
            }));
        }
        barrier.wait();
        let results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
        let status = bank.verify_history().unwrap();
        assert_eq!(status.revision, 3);
        assert_eq!(status.verified_revision_count, 3);
        assert_eq!(status.active_adapter_count, 1);
    }

    #[test]
    fn stale_head_cache_cannot_replay_a_revoked_adapter() {
        let fixture = Fixture::new("stale-head");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter, config) = fixture.adapter("axis-a", 0.0);
        let imported = bank
            .import_lora(&fixture.import_request(
                "adapter-a",
                "lineage-a",
                adapter,
                config,
                initial_expectation(),
            ))
            .unwrap();
        let stale_head = fs::read(bank.head_path()).unwrap();
        let revoked = bank
            .revoke(&AdapterRevocationRequest {
                schema: ADAPTER_REVOCATION_REQUEST_SCHEMA.to_string(),
                adapter: AdapterVersion {
                    adapter_id: PatchId::parse("adapter-a").unwrap(),
                    generation: 1,
                },
                reason: "adapter failed a post-promotion safety review".to_string(),
                evidence: Some(fixture.evidence("stale-head-revocation")),
                expected: expectation(&imported),
            })
            .unwrap();
        assert_eq!(revoked.revision, 2);

        fs::write(bank.head_path(), stale_head).unwrap();
        let snapshot = bank.snapshot().unwrap();
        assert_eq!(snapshot.revision, 2);
        assert_eq!(snapshot.revocations.len(), 1);
        assert_eq!(bank.verify_history().unwrap().verified_revision_count, 2);
    }

    #[test]
    fn deleted_head_cache_recovers_from_the_revision_journal() {
        let fixture = Fixture::new("missing-head");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter, config) = fixture.adapter("axis-a", 0.0);
        bank.import_lora(&fixture.import_request(
            "adapter-a",
            "lineage-a",
            adapter,
            config,
            initial_expectation(),
        ))
        .unwrap();

        fs::remove_file(bank.head_path()).unwrap();
        let snapshot = bank.snapshot().unwrap();
        assert_eq!(snapshot.revision, 1);
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(bank.verify_history().unwrap().verified_revision_count, 1);
    }

    #[test]
    fn missing_revision_commit_fails_closed_instead_of_truncating() {
        let fixture = Fixture::new("missing-commit");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter_a, config_a) = fixture.adapter("axis-a", 0.0);
        let first = bank
            .import_lora(&fixture.import_request(
                "adapter-a",
                "lineage-a",
                adapter_a,
                config_a,
                initial_expectation(),
            ))
            .unwrap();
        let (adapter_b, config_b) = fixture.adapter("axis-b", 0.25);
        bank.import_lora(&fixture.import_request(
            "adapter-b",
            "lineage-b",
            adapter_b,
            config_b,
            expectation(&first),
        ))
        .unwrap();

        fs::remove_file(bank.revision_commit_path(1)).unwrap();
        assert!(bank.snapshot().is_err());
    }

    #[test]
    fn tampered_head_fails_closed() {
        let fixture = Fixture::new("tamper");
        let bank = AdapterBank::open(&fixture.root).unwrap();
        let (adapter, config) = fixture.adapter("axis-a", 0.0);
        bank.import_lora(&fixture.import_request(
            "adapter-a",
            "lineage-a",
            adapter,
            config,
            initial_expectation(),
        ))
        .unwrap();
        fs::write(fixture.root.join("state/adapter_bank/head.json"), b"{}").unwrap();
        assert!(bank.snapshot().is_err());
    }
}
