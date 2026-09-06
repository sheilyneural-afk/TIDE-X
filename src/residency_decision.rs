//! Universal, fail-closed residency decisions.
//!
//! Residency is an authority decision, not a caller preference.  A caller may
//! precommit a target and authenticated input references, but there is no
//! request field for `Software`, `Weights`, or `Hybrid`.  This module derives
//! that result from a fixed, versioned policy and a complete set of canonical
//! facts reduced by [`KnowledgeEngine`].
//!
//! This authority classifies residence only.  It deliberately contains no
//! promotion, translation, materialisation, weight-writing, or execution path.

use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::capability_bundle::{
    CapabilityBlocker, CapabilityBundle, CapabilityBundleStatus, CapabilityRepresentation,
    CapabilityRepresentationGap,
};
use crate::digest::{
    CapabilityBundleDigest, ResidencyDecisionDigest, ResidencyPolicyDigest,
    ResidencyPrecommitDigest, Sha256Digest,
};
use crate::error::{BrainError, BrainResult};
use crate::identity::{InquiryId, ResidencyDecisionRoundId, ResidencyTargetId};
use crate::knowledge_engine::{
    ActionReceiptDigest, AuthorityRootDigest, BoundedUnknownReason, ClaimAssessment,
    KnowledgeBlockReason, KnowledgeClaim, KnowledgeClaimId, KnowledgeDomain, KnowledgeEngine,
    KnowledgeObligationId, KnowledgePolicyDigest, KnowledgePredicate, KnowledgeState,
    KnowledgeStateDigest, KnowledgeTerminal, PlanningDecision, ProfiledKnowledgeDomainId,
};
use crate::security::verify_private_root;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const POLICY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:RESIDENCY-POLICY:v1\0";
const PRECOMMIT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:RESIDENCY-PRECOMMIT:v1\0";
const DECISION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:RESIDENCY-DECISION:v1\0";
const MAX_PRECOMMIT_BYTES: u64 = 1 << 20;
const MAX_DECISION_BYTES: u64 = 8 << 20;
const MAX_BUNDLE_BYTES: u64 = 64 << 20;

/// Canonical fact identifiers required in every residency knowledge state.
///
/// The fixed identities prevent a caller from omitting a favorable higher
/// residency or relabeling an unrelated established claim as residency proof.
pub const REQUIREMENTS_CLAIM_ID: &str = "residency.requirements.v1";
pub const EFFECTS_CLAIM_ID: &str = "residency.effects.v1";
pub const EXTERNAL_STATE_CLAIM_ID: &str = "residency.external-state.v1";
pub const OBSERVABILITY_CLAIM_ID: &str = "residency.observability.v1";
pub const WEIGHTS_REPRESENTABILITY_CLAIM_ID: &str = "residency.representability.weights.v1";
pub const HYBRID_REPRESENTABILITY_CLAIM_ID: &str = "residency.representability.hybrid.v1";
pub const SOFTWARE_REPRESENTABILITY_CLAIM_ID: &str = "residency.representability.software.v1";
pub const WEIGHTS_TARGET_CLAIM_ID: &str = "residency.target.weights.v1";
pub const HYBRID_TARGET_CLAIM_ID: &str = "residency.target.hybrid.v1";
pub const SOFTWARE_TARGET_CLAIM_ID: &str = "residency.target.software.v1";

pub const REQUIREMENTS_OBSERVATION_KEY: &str = "residency.requirements";
pub const EFFECTS_OBSERVATION_KEY: &str = "residency.effects";
pub const EXTERNAL_STATE_OBSERVATION_KEY: &str = "residency.external_state";
pub const OBSERVABILITY_OBSERVATION_KEY: &str = "residency.observability";
pub const WEIGHTS_REPRESENTABILITY_OBSERVATION_KEY: &str = "residency.representability.weights";
pub const HYBRID_REPRESENTABILITY_OBSERVATION_KEY: &str = "residency.representability.hybrid";
pub const SOFTWARE_REPRESENTABILITY_OBSERVATION_KEY: &str = "residency.representability.software";
pub const WEIGHTS_TARGET_OBSERVATION_KEY: &str = "residency.target.weights";
pub const HYBRID_TARGET_OBSERVATION_KEY: &str = "residency.target.hybrid";
pub const SOFTWARE_TARGET_OBSERVATION_KEY: &str = "residency.target.software";

pub const CLOSED_COMPUTATION_VALUE: &str = "closed_computation";
pub const BOUNDARY_RUNTIME_VALUE: &str = "boundary_runtime";
pub const SOFTWARE_RUNTIME_VALUE: &str = "software_runtime";
pub const PURE_EFFECTS_VALUE: &str = "pure";
pub const BOUNDARY_EFFECTS_VALUE: &str = "boundary_effects";
pub const SOFTWARE_EFFECTS_VALUE: &str = "software_effects";
pub const NO_EXTERNAL_STATE_VALUE: &str = "none";
pub const BOUNDARY_MANAGED_STATE_VALUE: &str = "boundary_managed";
pub const SOFTWARE_AUTHORITATIVE_STATE_VALUE: &str = "software_authoritative";
pub const WEIGHT_COMPLETE_OBSERVABILITY_VALUE: &str = "weight_complete";
pub const BOUNDARY_COMPLETE_OBSERVABILITY_VALUE: &str = "boundary_complete";
pub const SOFTWARE_COMPLETE_OBSERVABILITY_VALUE: &str = "software_complete";
pub const UNKNOWN_SEMANTICS_VALUE: &str = "unknown";
pub const TARGET_COMPATIBLE_SUFFIX: &str = ":compatible";
pub const TARGET_INCOMPATIBLE_SUFFIX: &str = ":incompatible";

/// These are authority profiles for derived residency knowledge. They allow
/// the knowledge engine to combine whatever evidence methods a capability
/// actually needs without pretending that the residency matrix is the
/// universe of cognitive domains.
const RESIDENCY_SEMANTICS_DOMAIN_PROFILE: &str = "tidex.residency.execution-semantics.current";
const RESIDENCY_REPRESENTABILITY_DOMAIN_PROFILE: &str = "tidex.residency.representability.current";
const RESIDENCY_TARGET_DOMAIN_PROFILE: &str = "tidex.residency.target-compatibility.current";

fn invalid(code: &str) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: &str) -> BrainError {
    BrainError::Integrity(code.into())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
enum ResidencySchema {
    #[serde(rename = "cerebro.tidex.residency_decision/v1")]
    Current,
}

/// The three executable residency candidates considered by policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyCandidate {
    Weights,
    Hybrid,
    Software,
}

impl ResidencyCandidate {
    fn rank(self) -> u8 {
        match self {
            Self::Weights => 0,
            Self::Hybrid => 1,
            Self::Software => 2,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct ResidencyPolicyProjection {
    schema: ResidencySchema,
    policy_version: u16,
    fact_profile_version: u16,
    semantic_manifest_version: u16,
    require_complete_fact_matrix: bool,
    require_canonical_head: bool,
    selection_order: [ResidencyCandidate; 3],
}

/// Closed residency policy.  There is intentionally no public custom-policy
/// constructor: a new policy requires a reviewed version and a new digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyPolicy {
    version: u16,
    selection_order: [ResidencyCandidate; 3],
    digest: ResidencyPolicyDigest,
}

impl ResidencyPolicy {
    /// The sole active residency policy. Policy evolution replaces this
    /// authority atomically; callers cannot select an older or custom policy.
    pub fn current() -> BrainResult<Self> {
        let version = 1;
        let selection_order = [
            ResidencyCandidate::Weights,
            ResidencyCandidate::Hybrid,
            ResidencyCandidate::Software,
        ];
        let projection = ResidencyPolicyProjection {
            schema: ResidencySchema::Current,
            policy_version: version,
            fact_profile_version: 1,
            semantic_manifest_version: 1,
            require_complete_fact_matrix: true,
            require_canonical_head: true,
            selection_order,
        };
        Ok(Self {
            version,
            selection_order,
            digest: ResidencyPolicyDigest::from_computed(Sha256Digest::digest_domain(
                POLICY_DOMAIN,
                &serde_json::to_vec(&projection)?,
            )),
        })
    }

    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn digest(&self) -> &ResidencyPolicyDigest {
        &self.digest
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ResidencyPrecommitProjection<'a> {
    schema: ResidencySchema,
    round_id: &'a ResidencyDecisionRoundId,
    inquiry_id: &'a InquiryId,
    target_id: &'a ResidencyTargetId,
    policy_digest: &'a ResidencyPolicyDigest,
    capability_bundle_content: &'a Sha256Digest,
    knowledge_state_content: &'a Sha256Digest,
}

/// One-round semantic manifest for a residency decision.
///
/// Absolute paths are transport metadata and are excluded from the semantic
/// digest.  Exact referenced bytes remain committed by their raw SHA-256.
/// There is no requested or preferred outcome field.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResidencyDecisionPrecommit {
    schema: ResidencySchema,
    round_id: ResidencyDecisionRoundId,
    inquiry_id: InquiryId,
    target_id: ResidencyTargetId,
    policy_digest: ResidencyPolicyDigest,
    capability_bundle_reference: PrivateFileReference,
    knowledge_state_reference: PrivateFileReference,
    manifest_digest: ResidencyPrecommitDigest,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedResidencyDecisionPrecommit {
    schema: ResidencySchema,
    round_id: ResidencyDecisionRoundId,
    inquiry_id: InquiryId,
    target_id: ResidencyTargetId,
    policy_digest: ResidencyPolicyDigest,
    capability_bundle_reference: PrivateFileReference,
    knowledge_state_reference: PrivateFileReference,
    manifest_digest: ResidencyPrecommitDigest,
}

impl From<UntrustedResidencyDecisionPrecommit> for ResidencyDecisionPrecommit {
    fn from(value: UntrustedResidencyDecisionPrecommit) -> Self {
        Self {
            schema: value.schema,
            round_id: value.round_id,
            inquiry_id: value.inquiry_id,
            target_id: value.target_id,
            policy_digest: value.policy_digest,
            capability_bundle_reference: value.capability_bundle_reference,
            knowledge_state_reference: value.knowledge_state_reference,
            manifest_digest: value.manifest_digest,
        }
    }
}

impl ResidencyDecisionPrecommit {
    pub fn seal(
        round_id: ResidencyDecisionRoundId,
        inquiry_id: InquiryId,
        target_id: ResidencyTargetId,
        capability_bundle_reference: PrivateFileReference,
        knowledge_state_reference: PrivateFileReference,
        policy: &ResidencyPolicy,
    ) -> BrainResult<Self> {
        let mut manifest = Self {
            schema: ResidencySchema::Current,
            round_id,
            inquiry_id,
            target_id,
            policy_digest: policy.digest.clone(),
            capability_bundle_reference,
            knowledge_state_reference,
            manifest_digest: ResidencyPrecommitDigest::draft_marker(),
        };
        manifest.manifest_digest = manifest.calculate_digest()?;
        manifest.validate_for_policy(policy)?;
        Ok(manifest)
    }

    pub fn round_id(&self) -> &ResidencyDecisionRoundId {
        &self.round_id
    }

    pub fn inquiry_id(&self) -> &InquiryId {
        &self.inquiry_id
    }

    pub fn target_id(&self) -> &ResidencyTargetId {
        &self.target_id
    }

    pub fn policy_digest(&self) -> &ResidencyPolicyDigest {
        &self.policy_digest
    }

    pub fn capability_bundle_reference(&self) -> &PrivateFileReference {
        &self.capability_bundle_reference
    }

    pub fn knowledge_state_reference(&self) -> &PrivateFileReference {
        &self.knowledge_state_reference
    }

    pub fn digest(&self) -> &ResidencyPrecommitDigest {
        &self.manifest_digest
    }

    /// Persist the precommit as immutable transport bytes under its semantic
    /// content address.  Referenced artifacts are authenticated by `decide`.
    pub fn persist(
        &self,
        private_root: &Path,
        policy: &ResidencyPolicy,
    ) -> BrainResult<PrivateFileReference> {
        self.validate_for_policy(policy)?;
        let destination = precommit_path(private_root, &self.manifest_digest);
        let bytes = serde_json::to_vec(self)?;
        let sha256 = write_or_verify_immutable(private_root, &destination, &bytes)?;
        Ok(PrivateFileReference::new(destination, sha256))
    }

    pub fn authenticate(
        private_root: &Path,
        reference: &PrivateFileReference,
        policy: &ResidencyPolicy,
    ) -> BrainResult<Self> {
        let bytes = reference.read_verified_bounded(private_root, MAX_PRECOMMIT_BYTES)?;
        let manifest =
            Self::from(serde_json::from_slice::<UntrustedResidencyDecisionPrecommit>(&bytes)?);
        if reference.path != precommit_path(private_root, &manifest.manifest_digest) {
            return Err(integrity("residency_precommit_content_address_mismatch"));
        }
        manifest.validate_for_policy(policy)?;
        Ok(manifest)
    }

    fn calculate_digest(&self) -> BrainResult<ResidencyPrecommitDigest> {
        let projection = ResidencyPrecommitProjection {
            schema: self.schema,
            round_id: &self.round_id,
            inquiry_id: &self.inquiry_id,
            target_id: &self.target_id,
            policy_digest: &self.policy_digest,
            capability_bundle_content: &self.capability_bundle_reference.sha256,
            knowledge_state_content: &self.knowledge_state_reference.sha256,
        };
        Ok(ResidencyPrecommitDigest::from_computed(
            Sha256Digest::digest_domain(PRECOMMIT_DOMAIN, &serde_json::to_vec(&projection)?),
        ))
    }

    fn validate_for_policy(&self, policy: &ResidencyPolicy) -> BrainResult<()> {
        if self.schema != ResidencySchema::Current {
            return Err(invalid("residency_precommit_schema_unsupported"));
        }
        if self.policy_digest != policy.digest {
            return Err(integrity("residency_precommit_policy_mismatch"));
        }
        if self.capability_bundle_reference.sha256 == self.knowledge_state_reference.sha256 {
            return Err(invalid("residency_precommit_reference_type_confusion"));
        }
        if self.calculate_digest()? != self.manifest_digest {
            return Err(integrity("residency_precommit_digest_mismatch"));
        }
        Ok(())
    }
}

fn precommit_path(root: &Path, digest: &ResidencyPrecommitDigest) -> PathBuf {
    root.join("state/residency_decision/precommits/by-sha")
        .join(format!("{}.json", digest.as_str()))
}

/// Capability execution requirements established by the knowledge engine.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionRequirements {
    ClosedComputation,
    BoundaryRuntime,
    SoftwareRuntime,
    Unknown,
}

/// Effect containment established for the complete capability, not inferred
/// from a pure subgraph or from the current IR primitive vocabulary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EffectSemantics {
    Pure,
    BoundaryEffects,
    SoftwareEffects,
    Unknown,
}

/// Where authoritative mutable/external state must remain.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExternalStateSemantics {
    None,
    BoundaryManaged,
    SoftwareAuthoritative,
    Unknown,
}

/// Strongest fully observed boundary.  This is a semantic coverage claim, not
/// the mere availability of logs or tensors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ObservabilitySemantics {
    WeightComplete,
    BoundaryComplete,
    SoftwareComplete,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyDimension {
    Requirements,
    Effects,
    ExternalState,
    Observability,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "gap", rename_all = "snake_case", deny_unknown_fields)]
pub enum CandidateEvidenceGap {
    Missing,
    Unresolved,
    Contradicted,
    DomainMismatch {
        expected: KnowledgeDomain,
        observed: KnowledgeDomain,
    },
    Malformed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "reason", rename_all = "snake_case", deny_unknown_fields)]
pub enum CandidateRejection {
    BelowSemanticFloor { minimum: ResidencyCandidate },
    RepresentabilityNotEstablished { gap: CandidateEvidenceGap },
    RepresentationUnsupported,
    TargetCompatibilityNotEstablished { gap: CandidateEvidenceGap },
    TargetUnsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CandidateAssessment {
    candidate: ResidencyCandidate,
    eligible: bool,
    rejections: BTreeSet<CandidateRejection>,
}

impl CandidateAssessment {
    pub fn candidate(&self) -> ResidencyCandidate {
        self.candidate
    }

    pub fn is_eligible(&self) -> bool {
        self.eligible
    }

    pub fn rejections(&self) -> &BTreeSet<CandidateRejection> {
        &self.rejections
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResidencySelectionBasis {
    minimum_candidate: ResidencyCandidate,
    forced_by: BTreeSet<ResidencyDimension>,
    candidates: Vec<CandidateAssessment>,
}

impl ResidencySelectionBasis {
    pub fn minimum_candidate(&self) -> ResidencyCandidate {
        self.minimum_candidate
    }

    pub fn forced_by(&self) -> &BTreeSet<ResidencyDimension> {
        &self.forced_by
    }

    pub fn candidates(&self) -> &[CandidateAssessment] {
        &self.candidates
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "fact", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResidencyFact {
    Requirements {
        value: ExecutionRequirements,
    },
    Effects {
        value: EffectSemantics,
    },
    ExternalState {
        value: ExternalStateSemantics,
    },
    Observability {
        value: ObservabilitySemantics,
    },
    Representability {
        candidate: ResidencyCandidate,
        representable: bool,
    },
    TargetCompatibility {
        candidate: ResidencyCandidate,
        target_id: ResidencyTargetId,
        compatible: bool,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyFactKind {
    Requirements,
    Effects,
    ExternalState,
    Observability,
    WeightsRepresentability,
    HybridRepresentability,
    SoftwareRepresentability,
    WeightsTarget,
    HybridTarget,
    SoftwareTarget,
}

/// Structured link from one canonical residency fact to the authenticated
/// reducer receipts that established it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResidencyFactEvidence {
    kind: ResidencyFactKind,
    claim_id: KnowledgeClaimId,
    predicate: KnowledgePredicate,
    witnesses: BTreeSet<ActionReceiptDigest>,
    established_fact: ResidencyFact,
}

impl ResidencyFactEvidence {
    pub fn kind(&self) -> ResidencyFactKind {
        self.kind
    }

    pub fn claim_id(&self) -> &KnowledgeClaimId {
        &self.claim_id
    }

    pub fn predicate(&self) -> &KnowledgePredicate {
        &self.predicate
    }

    pub fn witnesses(&self) -> &BTreeSet<ActionReceiptDigest> {
        &self.witnesses
    }

    pub fn established_fact(&self) -> &ResidencyFact {
        &self.established_fact
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "reason", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResidencyBlockReason {
    BundleRepresentationOpen,
    BundleStatusBlocked,
    BundleUnmapped,
    BundlePartial {
        gaps: BTreeSet<CapabilityRepresentationGap>,
    },
    BundleReportedBlocker {
        blocker: CapabilityBlocker,
    },
    KnowledgeNotTerminal,
    KnowledgeTerminalBlocked {
        obligation_id: KnowledgeObligationId,
        block: KnowledgeBlockReason,
    },
    RequiredFactMissing {
        kind: ResidencyFactKind,
    },
    RequiredFactUnresolved {
        kind: ResidencyFactKind,
    },
    RequiredFactContradicted {
        kind: ResidencyFactKind,
    },
    RequiredFactDomainMismatch {
        kind: ResidencyFactKind,
        expected: KnowledgeDomain,
        observed: KnowledgeDomain,
    },
    RequiredFactMalformed {
        kind: ResidencyFactKind,
    },
    NoEligibleCandidate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "unknown", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResidencyUnknownReason {
    KnowledgeBoundary {
        reason: BoundedUnknownReason,
    },
    SemanticDimensions {
        dimensions: BTreeSet<ResidencyDimension>,
    },
    CandidateEvidenceIncomplete {
        candidates: BTreeSet<ResidencyCandidate>,
    },
}

/// The authoritative classification.  `Blocked` and `BoundedUnknown` are
/// first-class decisions, not exceptional fallbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResidencyDecision {
    Software {},
    Weights {},
    Hybrid {},
    Blocked {
        reasons: Vec<ResidencyBlockReason>,
    },
    BoundedUnknown {
        reason: ResidencyUnknownReason,
        unresolved_obligations: BTreeSet<KnowledgeObligationId>,
    },
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ResidencyDecisionProjection<'a> {
    schema: ResidencySchema,
    authority_root: &'a AuthorityRootDigest,
    round_id: &'a ResidencyDecisionRoundId,
    inquiry_id: &'a InquiryId,
    target_id: &'a ResidencyTargetId,
    residency_policy: &'a ResidencyPolicyDigest,
    knowledge_policy: &'a KnowledgePolicyDigest,
    precommit: &'a ResidencyPrecommitDigest,
    precommit_content: &'a Sha256Digest,
    capability_bundle: &'a CapabilityBundleDigest,
    capability_bundle_content: &'a Sha256Digest,
    knowledge_state: &'a KnowledgeStateDigest,
    knowledge_state_content: &'a Sha256Digest,
    knowledge_state_revision: u64,
    decision: &'a ResidencyDecision,
    evidence: &'a [ResidencyFactEvidence],
    selection_basis: &'a Option<ResidencySelectionBasis>,
}

/// Sealed decision record.  References retain root-confined transport paths;
/// its semantic digest commits identities and exact bytes but excludes those
/// paths, allowing relocation when the referenced authority artifacts are
/// themselves relocatable under the same durable authority identity.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResidencyDecisionRecord {
    schema: ResidencySchema,
    authority_root: AuthorityRootDigest,
    round_id: ResidencyDecisionRoundId,
    inquiry_id: InquiryId,
    target_id: ResidencyTargetId,
    residency_policy: ResidencyPolicyDigest,
    knowledge_policy: KnowledgePolicyDigest,
    precommit: ResidencyPrecommitDigest,
    precommit_reference: PrivateFileReference,
    capability_bundle: CapabilityBundleDigest,
    capability_bundle_reference: PrivateFileReference,
    knowledge_state: KnowledgeStateDigest,
    knowledge_state_reference: PrivateFileReference,
    knowledge_state_revision: u64,
    decision: ResidencyDecision,
    evidence: Vec<ResidencyFactEvidence>,
    selection_basis: Option<ResidencySelectionBasis>,
    manifest_digest: ResidencyDecisionDigest,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedResidencyDecisionRecord {
    schema: ResidencySchema,
    authority_root: AuthorityRootDigest,
    round_id: ResidencyDecisionRoundId,
    inquiry_id: InquiryId,
    target_id: ResidencyTargetId,
    residency_policy: ResidencyPolicyDigest,
    knowledge_policy: KnowledgePolicyDigest,
    precommit: ResidencyPrecommitDigest,
    precommit_reference: PrivateFileReference,
    capability_bundle: CapabilityBundleDigest,
    capability_bundle_reference: PrivateFileReference,
    knowledge_state: KnowledgeStateDigest,
    knowledge_state_reference: PrivateFileReference,
    knowledge_state_revision: u64,
    decision: ResidencyDecision,
    evidence: Vec<ResidencyFactEvidence>,
    selection_basis: Option<ResidencySelectionBasis>,
    manifest_digest: ResidencyDecisionDigest,
}

impl From<UntrustedResidencyDecisionRecord> for ResidencyDecisionRecord {
    fn from(value: UntrustedResidencyDecisionRecord) -> Self {
        Self {
            schema: value.schema,
            authority_root: value.authority_root,
            round_id: value.round_id,
            inquiry_id: value.inquiry_id,
            target_id: value.target_id,
            residency_policy: value.residency_policy,
            knowledge_policy: value.knowledge_policy,
            precommit: value.precommit,
            precommit_reference: value.precommit_reference,
            capability_bundle: value.capability_bundle,
            capability_bundle_reference: value.capability_bundle_reference,
            knowledge_state: value.knowledge_state,
            knowledge_state_reference: value.knowledge_state_reference,
            knowledge_state_revision: value.knowledge_state_revision,
            decision: value.decision,
            evidence: value.evidence,
            selection_basis: value.selection_basis,
            manifest_digest: value.manifest_digest,
        }
    }
}

impl ResidencyDecisionRecord {
    pub fn digest(&self) -> &ResidencyDecisionDigest {
        &self.manifest_digest
    }

    pub fn round_id(&self) -> &ResidencyDecisionRoundId {
        &self.round_id
    }

    pub fn inquiry_id(&self) -> &InquiryId {
        &self.inquiry_id
    }

    pub fn target_id(&self) -> &ResidencyTargetId {
        &self.target_id
    }

    pub fn decision(&self) -> &ResidencyDecision {
        &self.decision
    }

    pub fn evidence(&self) -> &[ResidencyFactEvidence] {
        &self.evidence
    }

    pub fn selection_basis(&self) -> Option<&ResidencySelectionBasis> {
        self.selection_basis.as_ref()
    }

    pub fn capability_bundle_digest(&self) -> &CapabilityBundleDigest {
        &self.capability_bundle
    }

    pub fn knowledge_state_digest(&self) -> &KnowledgeStateDigest {
        &self.knowledge_state
    }

    pub fn knowledge_state_revision(&self) -> u64 {
        self.knowledge_state_revision
    }

    fn calculate_digest(&self) -> BrainResult<ResidencyDecisionDigest> {
        let projection = ResidencyDecisionProjection {
            schema: self.schema,
            authority_root: &self.authority_root,
            round_id: &self.round_id,
            inquiry_id: &self.inquiry_id,
            target_id: &self.target_id,
            residency_policy: &self.residency_policy,
            knowledge_policy: &self.knowledge_policy,
            precommit: &self.precommit,
            precommit_content: &self.precommit_reference.sha256,
            capability_bundle: &self.capability_bundle,
            capability_bundle_content: &self.capability_bundle_reference.sha256,
            knowledge_state: &self.knowledge_state,
            knowledge_state_content: &self.knowledge_state_reference.sha256,
            knowledge_state_revision: self.knowledge_state_revision,
            decision: &self.decision,
            evidence: &self.evidence,
            selection_basis: &self.selection_basis,
        };
        Ok(ResidencyDecisionDigest::from_computed(
            Sha256Digest::digest_domain(DECISION_DOMAIN, &serde_json::to_vec(&projection)?),
        ))
    }

    fn validate_digest(&self) -> BrainResult<()> {
        if self.schema != ResidencySchema::Current {
            return Err(invalid("residency_decision_schema_unsupported"));
        }
        if self.calculate_digest()? != self.manifest_digest {
            return Err(integrity("residency_decision_digest_mismatch"));
        }
        Ok(())
    }
}

fn decision_path(root: &Path, digest: &ResidencyDecisionDigest) -> PathBuf {
    root.join("state/residency_decision/decisions/by-sha")
        .join(format!("{}.json", digest.as_str()))
}

/// Root-bound residency authority using the existing knowledge authority for
/// full state-lineage authentication and canonical-head freshness.
pub struct ResidencyDecisionAuthority<'a> {
    private_root: PathBuf,
    knowledge_engine: &'a KnowledgeEngine,
    policy: ResidencyPolicy,
}

impl<'a> ResidencyDecisionAuthority<'a> {
    /// Construct the sole active residency authority.
    pub fn current(
        private_root: &Path,
        knowledge_engine: &'a KnowledgeEngine,
    ) -> BrainResult<Self> {
        Ok(Self {
            private_root: verify_private_root(private_root)?,
            knowledge_engine,
            policy: ResidencyPolicy::current()?,
        })
    }

    pub fn policy(&self) -> &ResidencyPolicy {
        &self.policy
    }

    pub fn decide(
        &self,
        precommit_reference: &PrivateFileReference,
    ) -> BrainResult<ResidencyDecisionRecord> {
        let precommit = ResidencyDecisionPrecommit::authenticate(
            &self.private_root,
            precommit_reference,
            &self.policy,
        )?;
        let bundle =
            authenticate_bundle(&self.private_root, &precommit.capability_bundle_reference)?;
        let state = self
            .knowledge_engine
            .authenticate_state(&precommit.knowledge_state_reference)?;
        if state.inquiry_id() != &precommit.inquiry_id
            || state.capability_bundle_digest() != bundle.manifest_digest()
        {
            return Err(integrity("residency_precommit_state_binding_mismatch"));
        }

        let head = self
            .knowledge_engine
            .load_canonical_head(&precommit.inquiry_id)?;
        if head.inquiry_id() != &precommit.inquiry_id {
            return Err(integrity("residency_canonical_inquiry_mismatch"));
        }
        ensure_current_state(
            state.digest(),
            state.revision(),
            head.current_state(),
            head.revision(),
        )?;

        let (decision, evidence, selection_basis) =
            self.derive_decision(&bundle, &state, &precommit.target_id)?;
        let mut record = ResidencyDecisionRecord {
            schema: ResidencySchema::Current,
            authority_root: state.authority_root().clone(),
            round_id: precommit.round_id.clone(),
            inquiry_id: precommit.inquiry_id.clone(),
            target_id: precommit.target_id.clone(),
            residency_policy: self.policy.digest.clone(),
            knowledge_policy: state.policy_digest().clone(),
            precommit: precommit.manifest_digest.clone(),
            precommit_reference: precommit_reference.clone(),
            capability_bundle: bundle.manifest_digest().clone(),
            capability_bundle_reference: precommit.capability_bundle_reference.clone(),
            knowledge_state: state.digest().clone(),
            knowledge_state_reference: precommit.knowledge_state_reference.clone(),
            knowledge_state_revision: state.revision(),
            decision,
            evidence,
            selection_basis,
            manifest_digest: ResidencyDecisionDigest::draft_marker(),
        };
        record.manifest_digest = record.calculate_digest()?;
        record.validate_digest()?;
        Ok(record)
    }

    pub fn decide_and_persist(
        &self,
        precommit_reference: &PrivateFileReference,
    ) -> BrainResult<(ResidencyDecisionRecord, PrivateFileReference)> {
        let record = self.decide(precommit_reference)?;
        let reference = self.persist_decision(&record)?;
        Ok((record, reference))
    }

    pub fn persist_decision(
        &self,
        record: &ResidencyDecisionRecord,
    ) -> BrainResult<PrivateFileReference> {
        record.validate_digest()?;
        if record.residency_policy != self.policy.digest {
            return Err(integrity("residency_decision_policy_mismatch"));
        }
        let expected = self.decide(&record.precommit_reference)?;
        if expected != *record {
            return Err(integrity("residency_decision_not_authority_output"));
        }
        let destination = decision_path(&self.private_root, &record.manifest_digest);
        let bytes = serde_json::to_vec(record)?;
        let sha256 = write_or_verify_immutable(&self.private_root, &destination, &bytes)?;
        Ok(PrivateFileReference::new(destination, sha256))
    }

    /// Authenticate a persisted result for one caller-expected round.  A
    /// result from another round is rejected even when every other semantic
    /// input happens to match.
    pub fn authenticate_decision(
        &self,
        reference: &PrivateFileReference,
        expected_round: &ResidencyDecisionRoundId,
    ) -> BrainResult<ResidencyDecisionRecord> {
        let bytes = reference.read_verified_bounded(&self.private_root, MAX_DECISION_BYTES)?;
        let record = ResidencyDecisionRecord::from(serde_json::from_slice::<
            UntrustedResidencyDecisionRecord,
        >(&bytes)?);
        if reference.path != decision_path(&self.private_root, &record.manifest_digest) {
            return Err(integrity("residency_decision_content_address_mismatch"));
        }
        ensure_expected_round(&record.round_id, expected_round)?;
        record.validate_digest()?;
        let expected = self.decide(&record.precommit_reference)?;
        if expected != record {
            return Err(integrity("residency_decision_recomputation_mismatch"));
        }
        Ok(expected)
    }

    fn derive_decision(
        &self,
        bundle: &CapabilityBundle,
        state: &KnowledgeState,
        target_id: &ResidencyTargetId,
    ) -> BrainResult<(
        ResidencyDecision,
        Vec<ResidencyFactEvidence>,
        Option<ResidencySelectionBasis>,
    )> {
        let bundle_reasons = bundle_block_reasons(bundle);
        if !bundle_reasons.is_empty() {
            return Ok((
                ResidencyDecision::Blocked {
                    reasons: bundle_reasons,
                },
                Vec::new(),
                None,
            ));
        }

        match self.knowledge_engine.plan(state)? {
            PlanningDecision::Invoke { .. } => {
                return Ok((
                    ResidencyDecision::Blocked {
                        reasons: vec![ResidencyBlockReason::KnowledgeNotTerminal],
                    },
                    Vec::new(),
                    None,
                ));
            }
            PlanningDecision::Terminal {
                terminal:
                    KnowledgeTerminal::BoundedUnknown {
                        reason,
                        unresolved_obligations,
                        ..
                    },
            } => {
                return Ok((
                    ResidencyDecision::BoundedUnknown {
                        reason: ResidencyUnknownReason::KnowledgeBoundary { reason },
                        unresolved_obligations,
                    },
                    Vec::new(),
                    None,
                ));
            }
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::Blocked { blockers, .. },
            } => {
                let reasons = blockers
                    .into_iter()
                    .map(
                        |(obligation_id, block)| ResidencyBlockReason::KnowledgeTerminalBlocked {
                            obligation_id,
                            block,
                        },
                    )
                    .collect();
                return Ok((ResidencyDecision::Blocked { reasons }, Vec::new(), None));
            }
            PlanningDecision::Terminal {
                terminal: KnowledgeTerminal::ScopedComplete { .. },
            } => {}
        }

        let extracted = extract_fact_matrix(state, target_id)?;
        if !extracted.blockers.is_empty() {
            return Ok((
                ResidencyDecision::Blocked {
                    reasons: extracted.blockers,
                },
                extracted.evidence,
                None,
            ));
        }
        let facts = extracted
            .facts
            .ok_or_else(|| integrity("residency_fact_matrix_missing_without_reason"))?;
        let (decision, basis) = evaluate_fact_matrix(&self.policy, &facts);
        Ok((decision, extracted.evidence, Some(basis)))
    }
}

fn authenticate_bundle(
    private_root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<CapabilityBundle> {
    let bytes = reference.read_verified_bounded(private_root, MAX_BUNDLE_BYTES)?;
    let bundle: CapabilityBundle = serde_json::from_slice(&bytes)?;
    bundle.validate(private_root)?;
    let expected = private_root
        .join("state/capability_bundles/by-sha")
        .join(format!("{}.json", bundle.manifest_digest().as_str()));
    if reference.path != expected {
        return Err(integrity("residency_bundle_content_address_mismatch"));
    }
    Ok(bundle)
}

fn ensure_current_state(
    state_digest: &KnowledgeStateDigest,
    state_revision: u64,
    current_digest: &KnowledgeStateDigest,
    current_revision: u64,
) -> BrainResult<()> {
    if state_digest != current_digest || state_revision != current_revision {
        return Err(integrity("residency_stale_state_replay"));
    }
    Ok(())
}

fn ensure_expected_round(
    actual: &ResidencyDecisionRoundId,
    expected: &ResidencyDecisionRoundId,
) -> BrainResult<()> {
    if actual != expected {
        return Err(integrity("residency_cross_round_replay"));
    }
    Ok(())
}

fn bundle_block_reasons(bundle: &CapabilityBundle) -> Vec<ResidencyBlockReason> {
    let mut reasons = Vec::new();
    match bundle.status() {
        CapabilityBundleStatus::Open => {
            reasons.push(ResidencyBlockReason::BundleRepresentationOpen)
        }
        CapabilityBundleStatus::Blocked => reasons.push(ResidencyBlockReason::BundleStatusBlocked),
        CapabilityBundleStatus::RepresentationClosed => {}
    }
    match bundle.representation() {
        CapabilityRepresentation::Unmapped => reasons.push(ResidencyBlockReason::BundleUnmapped),
        CapabilityRepresentation::Partial { gaps, .. } => {
            reasons.push(ResidencyBlockReason::BundlePartial { gaps: gaps.clone() });
        }
        CapabilityRepresentation::Closed { .. } => {}
    }
    reasons.extend(
        bundle
            .blockers()
            .iter()
            .copied()
            .map(|blocker| ResidencyBlockReason::BundleReportedBlocker { blocker }),
    );
    reasons
}

#[derive(Debug, Clone)]
enum CandidateFactStatus {
    Established(bool),
    NotEstablished(CandidateEvidenceGap),
}

#[derive(Debug, Clone)]
struct CandidateFacts<T> {
    weights: T,
    hybrid: T,
    software: T,
}

impl<T> CandidateFacts<T> {
    fn get(&self, candidate: ResidencyCandidate) -> &T {
        match candidate {
            ResidencyCandidate::Weights => &self.weights,
            ResidencyCandidate::Hybrid => &self.hybrid,
            ResidencyCandidate::Software => &self.software,
        }
    }
}

#[derive(Debug, Clone)]
struct EvaluatedFacts {
    requirements: ExecutionRequirements,
    effects: EffectSemantics,
    external_state: ExternalStateSemantics,
    observability: ObservabilitySemantics,
    representability: CandidateFacts<CandidateFactStatus>,
    target_compatibility: CandidateFacts<CandidateFactStatus>,
}

struct ExtractedFactMatrix {
    facts: Option<EvaluatedFacts>,
    evidence: Vec<ResidencyFactEvidence>,
    blockers: Vec<ResidencyBlockReason>,
}

/// One policy-required fact and the exact epistemic domain authorized to
/// establish it. This matrix is specific to residency classification; it is
/// not a closed taxonomy of TIDE-X cognitive domains.
#[derive(Debug, Clone)]
struct ResidencyFactRequirement {
    claim_id: &'static str,
    observation_key: &'static str,
    kind: ResidencyFactKind,
    evidence_domain: KnowledgeDomain,
}

impl ResidencyFactRequirement {
    fn new(
        claim_id: &'static str,
        observation_key: &'static str,
        kind: ResidencyFactKind,
        evidence_domain: KnowledgeDomain,
    ) -> Self {
        Self {
            claim_id,
            observation_key,
            kind,
            evidence_domain,
        }
    }
}

/// Cohesive extraction context. Evidence and blockers can only be emitted by
/// parsing a claim from the authenticated knowledge state owned by this
/// collector.
struct ResidencyFactCollector<'a> {
    state: &'a KnowledgeState,
    evidence: Vec<ResidencyFactEvidence>,
    blockers: Vec<ResidencyBlockReason>,
}

impl<'a> ResidencyFactCollector<'a> {
    fn new(state: &'a KnowledgeState) -> Self {
        Self {
            state,
            evidence: Vec::with_capacity(10),
            blockers: Vec::new(),
        }
    }

    fn symbol<T: Copy>(
        &mut self,
        requirement: ResidencyFactRequirement,
        parse: fn(&str) -> Option<T>,
        make_fact: fn(T) -> ResidencyFact,
    ) -> BrainResult<Option<T>> {
        let Some((claim, witnesses)) =
            established_claim(self.state, &requirement, &mut self.blockers)?
        else {
            return Ok(None);
        };
        let value = match claim.predicate() {
            KnowledgePredicate::SymbolEquals { key, expected }
                if key.as_str() == requirement.observation_key =>
            {
                parse(expected.as_str())
            }
            _ => None,
        };
        let Some(value) = value else {
            self.blockers
                .push(ResidencyBlockReason::RequiredFactMalformed {
                    kind: requirement.kind,
                });
            return Ok(None);
        };
        self.evidence.push(ResidencyFactEvidence {
            kind: requirement.kind,
            claim_id: claim.id().clone(),
            predicate: claim.predicate().clone(),
            witnesses,
            established_fact: make_fact(value),
        });
        Ok(Some(value))
    }

    fn representability(
        &mut self,
        requirement: ResidencyFactRequirement,
        candidate: ResidencyCandidate,
    ) -> BrainResult<CandidateFactStatus> {
        let (claim, witnesses) = match candidate_claim(self.state, &requirement)? {
            Ok(established) => established,
            Err(gap) => return Ok(CandidateFactStatus::NotEstablished(gap)),
        };
        let value = match claim.predicate() {
            KnowledgePredicate::BoolEquals { key, expected }
                if key.as_str() == requirement.observation_key =>
            {
                Some(*expected)
            }
            _ => None,
        };
        let Some(value) = value else {
            return Ok(CandidateFactStatus::NotEstablished(
                CandidateEvidenceGap::Malformed,
            ));
        };
        self.evidence.push(ResidencyFactEvidence {
            kind: requirement.kind,
            claim_id: claim.id().clone(),
            predicate: claim.predicate().clone(),
            witnesses,
            established_fact: ResidencyFact::Representability {
                candidate,
                representable: value,
            },
        });
        Ok(CandidateFactStatus::Established(value))
    }

    fn target_compatibility(
        &mut self,
        requirement: ResidencyFactRequirement,
        candidate: ResidencyCandidate,
        target_id: &ResidencyTargetId,
    ) -> BrainResult<CandidateFactStatus> {
        let (claim, witnesses) = match candidate_claim(self.state, &requirement)? {
            Ok(established) => established,
            Err(gap) => return Ok(CandidateFactStatus::NotEstablished(gap)),
        };
        let compatible = format!("{}{TARGET_COMPATIBLE_SUFFIX}", target_id.as_str());
        let incompatible = format!("{}{TARGET_INCOMPATIBLE_SUFFIX}", target_id.as_str());
        let value = match claim.predicate() {
            KnowledgePredicate::SymbolEquals { key, expected }
                if key.as_str() == requirement.observation_key
                    && expected.as_str() == compatible =>
            {
                Some(true)
            }
            KnowledgePredicate::SymbolEquals { key, expected }
                if key.as_str() == requirement.observation_key
                    && expected.as_str() == incompatible =>
            {
                Some(false)
            }
            _ => None,
        };
        let Some(value) = value else {
            return Ok(CandidateFactStatus::NotEstablished(
                CandidateEvidenceGap::Malformed,
            ));
        };
        self.evidence.push(ResidencyFactEvidence {
            kind: requirement.kind,
            claim_id: claim.id().clone(),
            predicate: claim.predicate().clone(),
            witnesses,
            established_fact: ResidencyFact::TargetCompatibility {
                candidate,
                target_id: target_id.clone(),
                compatible: value,
            },
        });
        Ok(CandidateFactStatus::Established(value))
    }

    fn finish(self, facts: Option<EvaluatedFacts>) -> ExtractedFactMatrix {
        ExtractedFactMatrix {
            facts,
            evidence: self.evidence,
            blockers: self.blockers,
        }
    }
}

fn profiled_domain(profile_id: &str) -> BrainResult<KnowledgeDomain> {
    Ok(KnowledgeDomain::Profiled {
        profile_id: ProfiledKnowledgeDomainId::parse(profile_id)?,
    })
}

/// Typed knowledge domain for the capability-wide execution semantics needed
/// by residency policy.
pub fn residency_semantics_knowledge_domain() -> BrainResult<KnowledgeDomain> {
    profiled_domain(RESIDENCY_SEMANTICS_DOMAIN_PROFILE)
}

/// Typed knowledge domain for pre-translation representability. This is
/// deliberately distinct from post-translation behavioral equivalence.
pub fn residency_representability_knowledge_domain() -> BrainResult<KnowledgeDomain> {
    profiled_domain(RESIDENCY_REPRESENTABILITY_DOMAIN_PROFILE)
}

/// Typed knowledge domain for compatibility with one concrete target profile.
pub fn residency_target_knowledge_domain() -> BrainResult<KnowledgeDomain> {
    profiled_domain(RESIDENCY_TARGET_DOMAIN_PROFILE)
}

fn extract_fact_matrix(
    state: &KnowledgeState,
    target_id: &ResidencyTargetId,
) -> BrainResult<ExtractedFactMatrix> {
    let mut collector = ResidencyFactCollector::new(state);
    let semantics_domain = residency_semantics_knowledge_domain()?;
    let representability_domain = residency_representability_knowledge_domain()?;
    let target_domain = residency_target_knowledge_domain()?;

    let requirements = collector.symbol(
        ResidencyFactRequirement::new(
            REQUIREMENTS_CLAIM_ID,
            REQUIREMENTS_OBSERVATION_KEY,
            ResidencyFactKind::Requirements,
            semantics_domain.clone(),
        ),
        parse_requirements,
        |value| ResidencyFact::Requirements { value },
    )?;
    let effects = collector.symbol(
        ResidencyFactRequirement::new(
            EFFECTS_CLAIM_ID,
            EFFECTS_OBSERVATION_KEY,
            ResidencyFactKind::Effects,
            semantics_domain.clone(),
        ),
        parse_effects,
        |value| ResidencyFact::Effects { value },
    )?;
    let external_state = collector.symbol(
        ResidencyFactRequirement::new(
            EXTERNAL_STATE_CLAIM_ID,
            EXTERNAL_STATE_OBSERVATION_KEY,
            ResidencyFactKind::ExternalState,
            semantics_domain.clone(),
        ),
        parse_external_state,
        |value| ResidencyFact::ExternalState { value },
    )?;
    let observability = collector.symbol(
        ResidencyFactRequirement::new(
            OBSERVABILITY_CLAIM_ID,
            OBSERVABILITY_OBSERVATION_KEY,
            ResidencyFactKind::Observability,
            semantics_domain,
        ),
        parse_observability,
        |value| ResidencyFact::Observability { value },
    )?;

    let weights_representability = collector.representability(
        ResidencyFactRequirement::new(
            WEIGHTS_REPRESENTABILITY_CLAIM_ID,
            WEIGHTS_REPRESENTABILITY_OBSERVATION_KEY,
            ResidencyFactKind::WeightsRepresentability,
            representability_domain.clone(),
        ),
        ResidencyCandidate::Weights,
    )?;
    let hybrid_representability = collector.representability(
        ResidencyFactRequirement::new(
            HYBRID_REPRESENTABILITY_CLAIM_ID,
            HYBRID_REPRESENTABILITY_OBSERVATION_KEY,
            ResidencyFactKind::HybridRepresentability,
            representability_domain.clone(),
        ),
        ResidencyCandidate::Hybrid,
    )?;
    let software_representability = collector.representability(
        ResidencyFactRequirement::new(
            SOFTWARE_REPRESENTABILITY_CLAIM_ID,
            SOFTWARE_REPRESENTABILITY_OBSERVATION_KEY,
            ResidencyFactKind::SoftwareRepresentability,
            representability_domain,
        ),
        ResidencyCandidate::Software,
    )?;

    let weights_target = collector.target_compatibility(
        ResidencyFactRequirement::new(
            WEIGHTS_TARGET_CLAIM_ID,
            WEIGHTS_TARGET_OBSERVATION_KEY,
            ResidencyFactKind::WeightsTarget,
            target_domain.clone(),
        ),
        ResidencyCandidate::Weights,
        target_id,
    )?;
    let hybrid_target = collector.target_compatibility(
        ResidencyFactRequirement::new(
            HYBRID_TARGET_CLAIM_ID,
            HYBRID_TARGET_OBSERVATION_KEY,
            ResidencyFactKind::HybridTarget,
            target_domain.clone(),
        ),
        ResidencyCandidate::Hybrid,
        target_id,
    )?;
    let software_target = collector.target_compatibility(
        ResidencyFactRequirement::new(
            SOFTWARE_TARGET_CLAIM_ID,
            SOFTWARE_TARGET_OBSERVATION_KEY,
            ResidencyFactKind::SoftwareTarget,
            target_domain,
        ),
        ResidencyCandidate::Software,
        target_id,
    )?;

    let facts = match (
        requirements,
        effects,
        external_state,
        observability,
        weights_representability,
        hybrid_representability,
        software_representability,
        weights_target,
        hybrid_target,
        software_target,
    ) {
        (
            Some(requirements),
            Some(effects),
            Some(external_state),
            Some(observability),
            weights_representability,
            hybrid_representability,
            software_representability,
            weights_target,
            hybrid_target,
            software_target,
        ) if collector.blockers.is_empty() => Some(EvaluatedFacts {
            requirements,
            effects,
            external_state,
            observability,
            representability: CandidateFacts {
                weights: weights_representability,
                hybrid: hybrid_representability,
                software: software_representability,
            },
            target_compatibility: CandidateFacts {
                weights: weights_target,
                hybrid: hybrid_target,
                software: software_target,
            },
        }),
        _ => None,
    };

    Ok(collector.finish(facts))
}

fn established_claim<'a>(
    state: &'a KnowledgeState,
    requirement: &ResidencyFactRequirement,
    blockers: &mut Vec<ResidencyBlockReason>,
) -> BrainResult<Option<(&'a KnowledgeClaim, BTreeSet<ActionReceiptDigest>)>> {
    let claim_id = KnowledgeClaimId::parse(requirement.claim_id)?;
    let Some(claim) = state.claims().get(&claim_id) else {
        blockers.push(ResidencyBlockReason::RequiredFactMissing {
            kind: requirement.kind,
        });
        return Ok(None);
    };
    let observed_domain = claim.domain();
    if observed_domain != requirement.evidence_domain.clone() {
        blockers.push(ResidencyBlockReason::RequiredFactDomainMismatch {
            kind: requirement.kind,
            expected: requirement.evidence_domain.clone(),
            observed: observed_domain,
        });
        return Ok(None);
    }
    match claim.assessment() {
        ClaimAssessment::Established { witnesses } => Ok(Some((claim, witnesses.clone()))),
        ClaimAssessment::Unresolved => {
            blockers.push(ResidencyBlockReason::RequiredFactUnresolved {
                kind: requirement.kind,
            });
            Ok(None)
        }
        ClaimAssessment::Contradicted { .. } => {
            blockers.push(ResidencyBlockReason::RequiredFactContradicted {
                kind: requirement.kind,
            });
            Ok(None)
        }
    }
}

fn candidate_claim<'a>(
    state: &'a KnowledgeState,
    requirement: &ResidencyFactRequirement,
) -> BrainResult<Result<(&'a KnowledgeClaim, BTreeSet<ActionReceiptDigest>), CandidateEvidenceGap>>
{
    let claim_id = KnowledgeClaimId::parse(requirement.claim_id)?;
    let Some(claim) = state.claims().get(&claim_id) else {
        return Ok(Err(CandidateEvidenceGap::Missing));
    };
    let observed_domain = claim.domain();
    if observed_domain != requirement.evidence_domain.clone() {
        return Ok(Err(CandidateEvidenceGap::DomainMismatch {
            expected: requirement.evidence_domain.clone(),
            observed: observed_domain,
        }));
    }
    match claim.assessment() {
        ClaimAssessment::Established { witnesses } => Ok(Ok((claim, witnesses.clone()))),
        ClaimAssessment::Unresolved => Ok(Err(CandidateEvidenceGap::Unresolved)),
        ClaimAssessment::Contradicted { .. } => Ok(Err(CandidateEvidenceGap::Contradicted)),
    }
}

fn parse_requirements(value: &str) -> Option<ExecutionRequirements> {
    match value {
        CLOSED_COMPUTATION_VALUE => Some(ExecutionRequirements::ClosedComputation),
        BOUNDARY_RUNTIME_VALUE => Some(ExecutionRequirements::BoundaryRuntime),
        SOFTWARE_RUNTIME_VALUE => Some(ExecutionRequirements::SoftwareRuntime),
        UNKNOWN_SEMANTICS_VALUE => Some(ExecutionRequirements::Unknown),
        _ => None,
    }
}

fn parse_effects(value: &str) -> Option<EffectSemantics> {
    match value {
        PURE_EFFECTS_VALUE => Some(EffectSemantics::Pure),
        BOUNDARY_EFFECTS_VALUE => Some(EffectSemantics::BoundaryEffects),
        SOFTWARE_EFFECTS_VALUE => Some(EffectSemantics::SoftwareEffects),
        UNKNOWN_SEMANTICS_VALUE => Some(EffectSemantics::Unknown),
        _ => None,
    }
}

fn parse_external_state(value: &str) -> Option<ExternalStateSemantics> {
    match value {
        NO_EXTERNAL_STATE_VALUE => Some(ExternalStateSemantics::None),
        BOUNDARY_MANAGED_STATE_VALUE => Some(ExternalStateSemantics::BoundaryManaged),
        SOFTWARE_AUTHORITATIVE_STATE_VALUE => Some(ExternalStateSemantics::SoftwareAuthoritative),
        UNKNOWN_SEMANTICS_VALUE => Some(ExternalStateSemantics::Unknown),
        _ => None,
    }
}

fn parse_observability(value: &str) -> Option<ObservabilitySemantics> {
    match value {
        WEIGHT_COMPLETE_OBSERVABILITY_VALUE => Some(ObservabilitySemantics::WeightComplete),
        BOUNDARY_COMPLETE_OBSERVABILITY_VALUE => Some(ObservabilitySemantics::BoundaryComplete),
        SOFTWARE_COMPLETE_OBSERVABILITY_VALUE => Some(ObservabilitySemantics::SoftwareComplete),
        UNKNOWN_SEMANTICS_VALUE => Some(ObservabilitySemantics::Unknown),
        _ => None,
    }
}

fn evaluate_fact_matrix(
    policy: &ResidencyPolicy,
    facts: &EvaluatedFacts,
) -> (ResidencyDecision, ResidencySelectionBasis) {
    let unknown_dimensions = unknown_dimensions(facts);
    if !unknown_dimensions.is_empty() {
        return (
            ResidencyDecision::BoundedUnknown {
                reason: ResidencyUnknownReason::SemanticDimensions {
                    dimensions: unknown_dimensions.clone(),
                },
                unresolved_obligations: BTreeSet::new(),
            },
            ResidencySelectionBasis {
                minimum_candidate: ResidencyCandidate::Software,
                forced_by: unknown_dimensions,
                candidates: Vec::new(),
            },
        );
    }

    let ranked_dimensions = [
        (
            ResidencyDimension::Requirements,
            requirement_rank(facts.requirements),
        ),
        (ResidencyDimension::Effects, effect_rank(facts.effects)),
        (
            ResidencyDimension::ExternalState,
            external_state_rank(facts.external_state),
        ),
        (
            ResidencyDimension::Observability,
            observability_rank(facts.observability),
        ),
    ];
    let minimum_rank = ranked_dimensions
        .iter()
        .map(|(_, rank)| *rank)
        .max()
        .unwrap_or(2);
    let minimum_candidate = candidate_for_rank(minimum_rank);
    let forced_by = ranked_dimensions
        .iter()
        .filter_map(|(dimension, rank)| {
            (*rank == minimum_rank && minimum_rank > 0).then_some(*dimension)
        })
        .collect();

    let mut candidates = Vec::with_capacity(policy.selection_order.len());
    let mut selected = None;
    let mut incomplete_candidates = BTreeSet::new();
    for candidate in policy.selection_order {
        let mut rejections = BTreeSet::new();
        let mut conclusively_rejected = false;
        let mut evidence_incomplete = false;
        if candidate.rank() < minimum_rank {
            rejections.insert(CandidateRejection::BelowSemanticFloor {
                minimum: minimum_candidate,
            });
            conclusively_rejected = true;
        }
        match facts.representability.get(candidate) {
            CandidateFactStatus::Established(true) => {}
            CandidateFactStatus::Established(false) => {
                rejections.insert(CandidateRejection::RepresentationUnsupported);
                conclusively_rejected = true;
            }
            CandidateFactStatus::NotEstablished(gap) => {
                rejections.insert(CandidateRejection::RepresentabilityNotEstablished {
                    gap: gap.clone(),
                });
                evidence_incomplete = true;
            }
        }
        match facts.target_compatibility.get(candidate) {
            CandidateFactStatus::Established(true) => {}
            CandidateFactStatus::Established(false) => {
                rejections.insert(CandidateRejection::TargetUnsupported);
                conclusively_rejected = true;
            }
            CandidateFactStatus::NotEstablished(gap) => {
                rejections.insert(CandidateRejection::TargetCompatibilityNotEstablished {
                    gap: gap.clone(),
                });
                evidence_incomplete = true;
            }
        }
        if evidence_incomplete && !conclusively_rejected {
            incomplete_candidates.insert(candidate);
        }
        let eligible = rejections.is_empty();
        if selected.is_none() && eligible {
            selected = Some(candidate);
        }
        candidates.push(CandidateAssessment {
            candidate,
            eligible,
            rejections,
        });
    }

    let decision = match selected {
        Some(ResidencyCandidate::Weights) => ResidencyDecision::Weights {},
        Some(ResidencyCandidate::Hybrid) => ResidencyDecision::Hybrid {},
        Some(ResidencyCandidate::Software) => ResidencyDecision::Software {},
        None if !incomplete_candidates.is_empty() => ResidencyDecision::BoundedUnknown {
            reason: ResidencyUnknownReason::CandidateEvidenceIncomplete {
                candidates: incomplete_candidates,
            },
            unresolved_obligations: BTreeSet::new(),
        },
        None => ResidencyDecision::Blocked {
            reasons: vec![ResidencyBlockReason::NoEligibleCandidate],
        },
    };
    (
        decision,
        ResidencySelectionBasis {
            minimum_candidate,
            forced_by,
            candidates,
        },
    )
}

fn unknown_dimensions(facts: &EvaluatedFacts) -> BTreeSet<ResidencyDimension> {
    let mut dimensions = BTreeSet::new();
    if facts.requirements == ExecutionRequirements::Unknown {
        dimensions.insert(ResidencyDimension::Requirements);
    }
    if facts.effects == EffectSemantics::Unknown {
        dimensions.insert(ResidencyDimension::Effects);
    }
    if facts.external_state == ExternalStateSemantics::Unknown {
        dimensions.insert(ResidencyDimension::ExternalState);
    }
    if facts.observability == ObservabilitySemantics::Unknown {
        dimensions.insert(ResidencyDimension::Observability);
    }
    dimensions
}

fn requirement_rank(value: ExecutionRequirements) -> u8 {
    match value {
        ExecutionRequirements::ClosedComputation => 0,
        ExecutionRequirements::BoundaryRuntime => 1,
        ExecutionRequirements::SoftwareRuntime | ExecutionRequirements::Unknown => 2,
    }
}

fn effect_rank(value: EffectSemantics) -> u8 {
    match value {
        EffectSemantics::Pure => 0,
        EffectSemantics::BoundaryEffects => 1,
        EffectSemantics::SoftwareEffects | EffectSemantics::Unknown => 2,
    }
}

fn external_state_rank(value: ExternalStateSemantics) -> u8 {
    match value {
        ExternalStateSemantics::None => 0,
        ExternalStateSemantics::BoundaryManaged => 1,
        ExternalStateSemantics::SoftwareAuthoritative | ExternalStateSemantics::Unknown => 2,
    }
}

fn observability_rank(value: ObservabilitySemantics) -> u8 {
    match value {
        ObservabilitySemantics::WeightComplete => 0,
        ObservabilitySemantics::BoundaryComplete => 1,
        ObservabilitySemantics::SoftwareComplete | ObservabilitySemantics::Unknown => 2,
    }
}

fn candidate_for_rank(rank: u8) -> ResidencyCandidate {
    match rank {
        0 => ResidencyCandidate::Weights,
        1 => ResidencyCandidate::Hybrid,
        _ => ResidencyCandidate::Software,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::de::DeserializeOwned;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn policy() -> ResidencyPolicy {
        ResidencyPolicy::current().unwrap()
    }

    fn candidate_facts(
        weights: bool,
        hybrid: bool,
        software: bool,
    ) -> CandidateFacts<CandidateFactStatus> {
        CandidateFacts {
            weights: CandidateFactStatus::Established(weights),
            hybrid: CandidateFactStatus::Established(hybrid),
            software: CandidateFactStatus::Established(software),
        }
    }

    fn fully_eligible_facts() -> EvaluatedFacts {
        EvaluatedFacts {
            requirements: ExecutionRequirements::ClosedComputation,
            effects: EffectSemantics::Pure,
            external_state: ExternalStateSemantics::None,
            observability: ObservabilitySemantics::WeightComplete,
            representability: candidate_facts(true, true, true),
            target_compatibility: candidate_facts(true, true, true),
        }
    }

    fn sha(byte: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn typed_digest<T: DeserializeOwned>(byte: u8) -> T {
        serde_json::from_str(&format!("\"{}\"", format!("{byte:02x}").repeat(32))).unwrap()
    }

    fn reference(path: &str, byte: u8) -> PrivateFileReference {
        PrivateFileReference::new(PathBuf::from(path), sha(byte))
    }

    fn precommit_with_paths(
        policy: &ResidencyPolicy,
        bundle_path: &str,
        state_path: &str,
    ) -> ResidencyDecisionPrecommit {
        ResidencyDecisionPrecommit::seal(
            ResidencyDecisionRoundId::parse("round.v1").unwrap(),
            InquiryId::parse("residency-inquiry.v1").unwrap(),
            ResidencyTargetId::parse("target-a.v1").unwrap(),
            reference(bundle_path, 1),
            reference(state_path, 2),
            policy,
        )
        .unwrap()
    }

    fn fixture_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "tidex-residency-{label}-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        root
    }

    #[test]
    fn pure_does_not_imply_weights_without_representability_or_target_proof() {
        let mut without_representability = fully_eligible_facts();
        without_representability.representability = candidate_facts(false, false, true);
        let (decision, basis) = evaluate_fact_matrix(&policy(), &without_representability);
        assert_eq!(decision, ResidencyDecision::Software {});
        assert!(basis.candidates[0]
            .rejections
            .contains(&CandidateRejection::RepresentationUnsupported));

        let mut without_target = fully_eligible_facts();
        without_target.target_compatibility = candidate_facts(false, false, true);
        let (decision, basis) = evaluate_fact_matrix(&policy(), &without_target);
        assert_eq!(decision, ResidencyDecision::Software {});
        assert!(basis.candidates[0]
            .rejections
            .contains(&CandidateRejection::TargetUnsupported));
    }

    #[test]
    fn external_state_and_effects_force_hybrid_or_software() {
        let mut boundary_state = fully_eligible_facts();
        boundary_state.external_state = ExternalStateSemantics::BoundaryManaged;
        let (decision, basis) = evaluate_fact_matrix(&policy(), &boundary_state);
        assert_eq!(decision, ResidencyDecision::Hybrid {});
        assert_eq!(basis.minimum_candidate, ResidencyCandidate::Hybrid);
        assert!(basis.forced_by.contains(&ResidencyDimension::ExternalState));

        let mut software_effects = fully_eligible_facts();
        software_effects.effects = EffectSemantics::SoftwareEffects;
        let (decision, basis) = evaluate_fact_matrix(&policy(), &software_effects);
        assert_eq!(decision, ResidencyDecision::Software {});
        assert_eq!(basis.minimum_candidate, ResidencyCandidate::Software);
        assert!(basis.forced_by.contains(&ResidencyDimension::Effects));
    }

    #[test]
    fn requirements_and_observability_also_govern_the_residency_floor() {
        let mut boundary_runtime = fully_eligible_facts();
        boundary_runtime.requirements = ExecutionRequirements::BoundaryRuntime;
        assert_eq!(
            evaluate_fact_matrix(&policy(), &boundary_runtime).0,
            ResidencyDecision::Hybrid {}
        );

        let mut software_runtime = fully_eligible_facts();
        software_runtime.requirements = ExecutionRequirements::SoftwareRuntime;
        assert_eq!(
            evaluate_fact_matrix(&policy(), &software_runtime).0,
            ResidencyDecision::Software {}
        );

        let mut boundary_observable = fully_eligible_facts();
        boundary_observable.observability = ObservabilitySemantics::BoundaryComplete;
        assert_eq!(
            evaluate_fact_matrix(&policy(), &boundary_observable).0,
            ResidencyDecision::Hybrid {}
        );

        let mut software_observable = fully_eligible_facts();
        software_observable.observability = ObservabilitySemantics::SoftwareComplete;
        assert_eq!(
            evaluate_fact_matrix(&policy(), &software_observable).0,
            ResidencyDecision::Software {}
        );
    }

    #[test]
    fn any_unknown_transversal_semantic_dimension_is_bounded_unknown() {
        let dimensions = [
            ResidencyDimension::Requirements,
            ResidencyDimension::Effects,
            ResidencyDimension::ExternalState,
            ResidencyDimension::Observability,
        ];
        for dimension in dimensions {
            let mut facts = fully_eligible_facts();
            match dimension {
                ResidencyDimension::Requirements => {
                    facts.requirements = ExecutionRequirements::Unknown
                }
                ResidencyDimension::Effects => facts.effects = EffectSemantics::Unknown,
                ResidencyDimension::ExternalState => {
                    facts.external_state = ExternalStateSemantics::Unknown
                }
                ResidencyDimension::Observability => {
                    facts.observability = ObservabilitySemantics::Unknown
                }
            }
            let (decision, _) = evaluate_fact_matrix(&policy(), &facts);
            assert_eq!(
                decision,
                ResidencyDecision::BoundedUnknown {
                    reason: ResidencyUnknownReason::SemanticDimensions {
                        dimensions: BTreeSet::from([dimension])
                    },
                    unresolved_obligations: BTreeSet::new(),
                }
            );
        }
    }

    #[test]
    fn candidate_specific_unknown_degrades_safely_to_proven_software() {
        let mut facts = fully_eligible_facts();
        facts.representability.weights =
            CandidateFactStatus::NotEstablished(CandidateEvidenceGap::Missing);
        facts.representability.hybrid =
            CandidateFactStatus::NotEstablished(CandidateEvidenceGap::Unresolved);
        facts.target_compatibility.weights =
            CandidateFactStatus::NotEstablished(CandidateEvidenceGap::Missing);
        facts.target_compatibility.hybrid =
            CandidateFactStatus::NotEstablished(CandidateEvidenceGap::Unresolved);

        let (decision, basis) = evaluate_fact_matrix(&policy(), &facts);
        assert_eq!(decision, ResidencyDecision::Software {});
        assert!(!basis.candidates[0].is_eligible());
        assert!(!basis.candidates[1].is_eligible());
        assert!(basis.candidates[2].is_eligible());

        facts.representability.software =
            CandidateFactStatus::NotEstablished(CandidateEvidenceGap::Missing);
        let (decision, _) = evaluate_fact_matrix(&policy(), &facts);
        assert_eq!(
            decision,
            ResidencyDecision::BoundedUnknown {
                reason: ResidencyUnknownReason::CandidateEvidenceIncomplete {
                    candidates: BTreeSet::from([
                        ResidencyCandidate::Weights,
                        ResidencyCandidate::Hybrid,
                        ResidencyCandidate::Software,
                    ]),
                },
                unresolved_obligations: BTreeSet::new(),
            }
        );
    }

    #[test]
    fn pretranslation_contract_rejects_behavioral_equivalence_as_residency_fact() {
        let fact = ResidencyFact::Representability {
            candidate: ResidencyCandidate::Weights,
            representable: true,
        };
        let encoded = serde_json::to_value(&fact).unwrap();
        assert_eq!(encoded["fact"], "representability");
        assert!(encoded.get("equivalence").is_none());
        assert!(serde_json::from_value::<ResidencyFact>(serde_json::json!({
            "fact": "equivalence",
            "candidate": "weights",
            "established": true
        }))
        .is_err());

        assert!(matches!(
            residency_representability_knowledge_domain().unwrap(),
            KnowledgeDomain::Profiled { .. }
        ));
    }

    #[test]
    fn caller_manifest_cannot_select_an_outcome() {
        let policy = policy();
        let manifest = precommit_with_paths(
            &policy,
            "/authority/state/capability_bundles/by-sha/bundle.json",
            "/authority/state/knowledge_engine/states/by-sha/state.json",
        );
        let mut value = serde_json::to_value(&manifest).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("decision".into(), serde_json::json!("weights"));
        assert!(serde_json::from_value::<UntrustedResidencyDecisionPrecommit>(value).is_err());
    }

    #[test]
    fn precommit_is_one_round_path_independent_and_tamper_evident() {
        let policy = policy();
        let first = precommit_with_paths(
            &policy,
            "/root-a/state/capability_bundles/by-sha/bundle.json",
            "/root-a/state/knowledge_engine/states/by-sha/state.json",
        );
        let relocated = precommit_with_paths(
            &policy,
            "/root-b/state/capability_bundles/by-sha/bundle.json",
            "/root-b/state/knowledge_engine/states/by-sha/state.json",
        );
        assert_eq!(first.digest(), relocated.digest());

        let projection = ResidencyPrecommitProjection {
            schema: first.schema,
            round_id: &first.round_id,
            inquiry_id: &first.inquiry_id,
            target_id: &first.target_id,
            policy_digest: &first.policy_digest,
            capability_bundle_content: &first.capability_bundle_reference.sha256,
            knowledge_state_content: &first.knowledge_state_reference.sha256,
        };
        let direct = ResidencyPrecommitDigest::from_computed(Sha256Digest::digest_domain(
            PRECOMMIT_DOMAIN,
            &serde_json::to_vec(&projection).unwrap(),
        ));
        assert_eq!(first.digest(), &direct);

        let mut tampered = first;
        tampered.target_id = ResidencyTargetId::parse("target-b.v1").unwrap();
        assert!(tampered.validate_for_policy(&policy).is_err());
    }

    #[test]
    fn precommit_rejects_cross_root_and_policy_mismatch() {
        let root_a = fixture_root("root-a");
        let root_b = fixture_root("root-b");
        let current_policy = policy();
        let manifest = precommit_with_paths(
            &current_policy,
            "/unresolved/bundle.json",
            "/unresolved/state.json",
        );
        let reference = manifest.persist(&root_a, &current_policy).unwrap();
        assert!(
            ResidencyDecisionPrecommit::authenticate(&root_b, &reference, &current_policy).is_err()
        );

        let mut unauthoritative_policy = current_policy.clone();
        unauthoritative_policy.digest = typed_digest(99);
        assert!(ResidencyDecisionPrecommit::authenticate(
            &root_a,
            &reference,
            &unauthoritative_policy
        )
        .is_err());
        fs::remove_dir_all(root_a).unwrap();
        fs::remove_dir_all(root_b).unwrap();
    }

    #[test]
    fn sealed_decision_detects_outcome_tampering() {
        let policy = policy();
        let precommit = precommit_with_paths(
            &policy,
            "/authority/state/capability_bundles/by-sha/bundle.json",
            "/authority/state/knowledge_engine/states/by-sha/state.json",
        );
        let precommit_reference = reference(
            "/authority/state/residency_decision/precommits/by-sha/precommit.json",
            3,
        );
        let mut record = ResidencyDecisionRecord {
            schema: ResidencySchema::Current,
            authority_root: typed_digest(4),
            round_id: precommit.round_id.clone(),
            inquiry_id: precommit.inquiry_id.clone(),
            target_id: precommit.target_id.clone(),
            residency_policy: policy.digest.clone(),
            knowledge_policy: typed_digest(5),
            precommit: precommit.manifest_digest.clone(),
            precommit_reference,
            capability_bundle: typed_digest(6),
            capability_bundle_reference: precommit.capability_bundle_reference.clone(),
            knowledge_state: typed_digest(7),
            knowledge_state_reference: precommit.knowledge_state_reference.clone(),
            knowledge_state_revision: 9,
            decision: ResidencyDecision::Software {},
            evidence: Vec::new(),
            selection_basis: None,
            manifest_digest: ResidencyDecisionDigest::draft_marker(),
        };
        record.manifest_digest = record.calculate_digest().unwrap();
        record.validate_digest().unwrap();
        record.decision = ResidencyDecision::Weights {};
        assert!(record.validate_digest().is_err());
    }

    #[test]
    fn stale_state_and_cross_round_replay_fail_closed() {
        let state: KnowledgeStateDigest = typed_digest(10);
        let next_state: KnowledgeStateDigest = typed_digest(11);
        assert!(ensure_current_state(&state, 7, &state, 7).is_ok());
        assert!(ensure_current_state(&state, 7, &next_state, 8).is_err());
        assert!(ensure_current_state(&state, 7, &state, 8).is_err());

        let round = ResidencyDecisionRoundId::parse("round.current.v1").unwrap();
        let replayed = ResidencyDecisionRoundId::parse("round.previous.v1").unwrap();
        assert!(ensure_expected_round(&round, &round).is_ok());
        assert!(ensure_expected_round(&replayed, &round).is_err());
    }

    #[test]
    fn unrecognized_json_fields_fail_closed_on_sealed_result() {
        let decision = ResidencyDecision::Weights {};
        let mut value = serde_json::to_value(decision).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("caller_override".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<ResidencyDecision>(value).is_err());
    }

    #[test]
    fn semantic_parsers_and_ranks_behave_exhaustively() {
        assert_eq!(
            parse_requirements(CLOSED_COMPUTATION_VALUE),
            Some(ExecutionRequirements::ClosedComputation)
        );
        assert_eq!(
            parse_requirements(BOUNDARY_RUNTIME_VALUE),
            Some(ExecutionRequirements::BoundaryRuntime)
        );
        assert_eq!(
            parse_requirements(SOFTWARE_RUNTIME_VALUE),
            Some(ExecutionRequirements::SoftwareRuntime)
        );
        assert_eq!(
            parse_requirements(UNKNOWN_SEMANTICS_VALUE),
            Some(ExecutionRequirements::Unknown)
        );
        assert_eq!(parse_requirements("invalid_req"), None);

        assert_eq!(
            parse_effects(PURE_EFFECTS_VALUE),
            Some(EffectSemantics::Pure)
        );
        assert_eq!(
            parse_effects(BOUNDARY_EFFECTS_VALUE),
            Some(EffectSemantics::BoundaryEffects)
        );
        assert_eq!(
            parse_effects(SOFTWARE_EFFECTS_VALUE),
            Some(EffectSemantics::SoftwareEffects)
        );
        assert_eq!(
            parse_effects(UNKNOWN_SEMANTICS_VALUE),
            Some(EffectSemantics::Unknown)
        );
        assert_eq!(parse_effects("invalid_eff"), None);

        assert_eq!(
            parse_external_state(NO_EXTERNAL_STATE_VALUE),
            Some(ExternalStateSemantics::None)
        );
        assert_eq!(
            parse_external_state(BOUNDARY_MANAGED_STATE_VALUE),
            Some(ExternalStateSemantics::BoundaryManaged)
        );
        assert_eq!(
            parse_external_state(SOFTWARE_AUTHORITATIVE_STATE_VALUE),
            Some(ExternalStateSemantics::SoftwareAuthoritative)
        );
        assert_eq!(
            parse_external_state(UNKNOWN_SEMANTICS_VALUE),
            Some(ExternalStateSemantics::Unknown)
        );
        assert_eq!(parse_external_state("invalid_state"), None);

        assert_eq!(
            parse_observability(WEIGHT_COMPLETE_OBSERVABILITY_VALUE),
            Some(ObservabilitySemantics::WeightComplete)
        );
        assert_eq!(
            parse_observability(BOUNDARY_COMPLETE_OBSERVABILITY_VALUE),
            Some(ObservabilitySemantics::BoundaryComplete)
        );
        assert_eq!(
            parse_observability(SOFTWARE_COMPLETE_OBSERVABILITY_VALUE),
            Some(ObservabilitySemantics::SoftwareComplete)
        );
        assert_eq!(
            parse_observability(UNKNOWN_SEMANTICS_VALUE),
            Some(ObservabilitySemantics::Unknown)
        );
        assert_eq!(parse_observability("invalid_obs"), None);

        assert_eq!(
            requirement_rank(ExecutionRequirements::ClosedComputation),
            0
        );
        assert_eq!(requirement_rank(ExecutionRequirements::BoundaryRuntime), 1);
        assert_eq!(requirement_rank(ExecutionRequirements::SoftwareRuntime), 2);
        assert_eq!(requirement_rank(ExecutionRequirements::Unknown), 2);

        assert_eq!(effect_rank(EffectSemantics::Pure), 0);
        assert_eq!(effect_rank(EffectSemantics::BoundaryEffects), 1);
        assert_eq!(effect_rank(EffectSemantics::SoftwareEffects), 2);
        assert_eq!(effect_rank(EffectSemantics::Unknown), 2);

        assert_eq!(external_state_rank(ExternalStateSemantics::None), 0);
        assert_eq!(
            external_state_rank(ExternalStateSemantics::BoundaryManaged),
            1
        );
        assert_eq!(
            external_state_rank(ExternalStateSemantics::SoftwareAuthoritative),
            2
        );
        assert_eq!(external_state_rank(ExternalStateSemantics::Unknown), 2);

        assert_eq!(
            observability_rank(ObservabilitySemantics::WeightComplete),
            0
        );
        assert_eq!(
            observability_rank(ObservabilitySemantics::BoundaryComplete),
            1
        );
        assert_eq!(
            observability_rank(ObservabilitySemantics::SoftwareComplete),
            2
        );
        assert_eq!(observability_rank(ObservabilitySemantics::Unknown), 2);

        assert_eq!(candidate_for_rank(0), ResidencyCandidate::Weights);
        assert_eq!(candidate_for_rank(1), ResidencyCandidate::Hybrid);
        assert_eq!(candidate_for_rank(2), ResidencyCandidate::Software);
        assert_eq!(candidate_for_rank(99), ResidencyCandidate::Software);
    }

    #[test]
    fn getters_and_accessors_contract_verification() {
        let policy = policy();
        assert_eq!(policy.version(), 1);
        assert_eq!(policy.digest(), &policy.digest);

        let precommit = precommit_with_paths(
            &policy,
            "/authority/state/capability_bundles/by-sha/bundle.json",
            "/authority/state/knowledge_engine/states/by-sha/state.json",
        );
        assert_eq!(precommit.round_id().as_str(), "round.v1");
        assert_eq!(precommit.inquiry_id().as_str(), "residency-inquiry.v1");
        assert_eq!(precommit.target_id().as_str(), "target-a.v1");
        assert_eq!(precommit.policy_digest(), &policy.digest);
        assert_eq!(precommit.capability_bundle_reference().sha256, sha(1));
        assert_eq!(precommit.knowledge_state_reference().sha256, sha(2));
        assert_eq!(precommit.digest(), precommit.digest());

        let assessment = CandidateAssessment {
            candidate: ResidencyCandidate::Weights,
            eligible: true,
            rejections: BTreeSet::new(),
        };
        assert_eq!(assessment.candidate(), ResidencyCandidate::Weights);
        assert!(assessment.is_eligible());
        assert!(assessment.rejections().is_empty());

        let basis = ResidencySelectionBasis {
            minimum_candidate: ResidencyCandidate::Weights,
            forced_by: BTreeSet::new(),
            candidates: vec![assessment],
        };
        assert_eq!(basis.minimum_candidate(), ResidencyCandidate::Weights);
        assert!(basis.forced_by().is_empty());
        assert_eq!(basis.candidates().len(), 1);

        let pred = KnowledgePredicate::BoolEquals {
            key: crate::knowledge_engine::ObservationKey::parse("test_key").unwrap(),
            expected: true,
        };
        let evidence_fact = ResidencyFactEvidence {
            kind: ResidencyFactKind::Requirements,
            claim_id: KnowledgeClaimId::parse("claim-1.v1").unwrap(),
            predicate: pred.clone(),
            witnesses: BTreeSet::new(),
            established_fact: ResidencyFact::Requirements {
                value: ExecutionRequirements::ClosedComputation,
            },
        };
        assert_eq!(evidence_fact.kind(), ResidencyFactKind::Requirements);
        assert_eq!(evidence_fact.claim_id().as_str(), "claim-1.v1");
        assert_eq!(evidence_fact.predicate(), &pred);
        assert!(evidence_fact.witnesses().is_empty());
        assert_eq!(
            evidence_fact.established_fact(),
            &ResidencyFact::Requirements {
                value: ExecutionRequirements::ClosedComputation
            }
        );

        let mut record = ResidencyDecisionRecord {
            schema: ResidencySchema::Current,
            authority_root: typed_digest(4),
            round_id: precommit.round_id.clone(),
            inquiry_id: precommit.inquiry_id.clone(),
            target_id: precommit.target_id.clone(),
            residency_policy: policy.digest.clone(),
            knowledge_policy: typed_digest(5),
            precommit: precommit.manifest_digest.clone(),
            precommit_reference: reference("/unresolved/precommit.json", 3),
            capability_bundle: typed_digest(6),
            capability_bundle_reference: precommit.capability_bundle_reference.clone(),
            knowledge_state: typed_digest(7),
            knowledge_state_reference: precommit.knowledge_state_reference.clone(),
            knowledge_state_revision: 9,
            decision: ResidencyDecision::Software {},
            evidence: vec![evidence_fact],
            selection_basis: Some(basis),
            manifest_digest: ResidencyDecisionDigest::draft_marker(),
        };
        record.manifest_digest = record.calculate_digest().unwrap();
        assert_eq!(record.round_id().as_str(), "round.v1");
        assert_eq!(record.inquiry_id().as_str(), "residency-inquiry.v1");
        assert_eq!(record.target_id().as_str(), "target-a.v1");
        assert_eq!(record.decision(), &ResidencyDecision::Software {});
        assert_eq!(record.evidence().len(), 1);
        assert!(record.selection_basis().is_some());
        assert_eq!(record.knowledge_state_revision(), 9);
        assert_eq!(record.digest(), &record.manifest_digest);
    }

    #[test]
    fn residency_decision_authority_lifecycle_fail_closed() {
        let root = fixture_root("auth-lifecycle");
        let engine = KnowledgeEngine::for_test(&root).unwrap();
        let authority = ResidencyDecisionAuthority {
            private_root: root.clone(),
            knowledge_engine: &engine,
            policy: ResidencyPolicy::current().unwrap(),
        };
        assert_eq!(authority.policy().digest(), policy().digest());

        // Calling decide on non-existent reference fails closed
        let dummy_ref = reference("/nonexistent.json", 99);
        assert!(authority.decide(&dummy_ref).is_err());

        // Calling authenticate_decision on bogus reference fails closed
        let round = ResidencyDecisionRoundId::parse("round.v1").unwrap();
        assert!(authority.authenticate_decision(&dummy_ref, &round).is_err());

        fs::remove_dir_all(&root).unwrap();
    }
}
