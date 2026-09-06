//! Sealed capability intake bundles.
//!
//! A bundle is the immutable subject that knowledge and residency authorities
//! consume. It binds a capability identity and its current representation to
//! one exact retained capture. Evidence is deliberately *not* interpreted in
//! this module: method-specific evidence belongs to the knowledge engine and
//! its authenticated verifier registry. Keeping that boundary prevents a core
//! contract from depending on a particular language, donor test runner,
//! benchmark, model, or evaluation suite.

use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::capability_ir::authenticate_capability_ir;
use crate::content_vault::{authenticate_capture_receipt, CaptureReceipt};
use crate::digest::{
    CapabilityBundleDigest, CapabilityIrDigest, CaptureReceiptDigest, Sha256Digest,
    SystemEnvelopeDigest,
};
use crate::error::{BrainError, BrainResult};
use crate::identity::CapabilityId;
use crate::security::verify_internal_private_root;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

const BUNDLE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CAPABILITY-BUNDLE:v4\0";
const MAX_CAPABILITY_BUNDLE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CapabilityBundleSchema {
    #[serde(rename = "cerebro.tidex.capability_bundle/v4")]
    Current,
}

/// A newly discovered capability is allowed to remain honestly unmapped.  A
/// closed IR is only present after the representation authority has proved it;
/// draft/blocked states never need to invent an approximate IR merely to fit
/// the bundle schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilityRepresentation {
    /// No formal representation has been authenticated yet.  This is an
    /// honest discovery state, never an implicit empty IR.
    Unmapped,
    /// A bounded part of the capability has been represented, but one or
    /// more material facts remain open.  `Partial` is deliberately barred
    /// from every promotable status: consumers must inspect the typed gaps
    /// rather than treating the existence of an IR artifact as equivalence.
    Partial {
        /// An optional closed fragment.  A fragment is useful for analysis,
        /// but does not close the capability while `gaps` is non-empty.
        capability_ir: Option<PrivateFileReference>,
        capability_ir_sha256: Option<CapabilityIrDigest>,
        gaps: BTreeSet<CapabilityRepresentationGap>,
    },
    Closed {
        capability_ir: PrivateFileReference,
        capability_ir_sha256: CapabilityIrDigest,
    },
}

/// Inputs for constructing a deliberately incomplete capability representation.
///
/// Keeping these inputs together makes the boundary explicit: a partial
/// representation is evidence of what is still unknown, never an alternate
/// spelling of a closed capability.
#[derive(Debug, Clone)]
pub struct PartialCapabilityBundleDraft {
    pub capability_id: CapabilityId,
    pub capture_receipt: PrivateFileReference,
    pub capability_ir: Option<PrivateFileReference>,
    pub capability_ir_sha256: Option<CapabilityIrDigest>,
    pub gaps: BTreeSet<CapabilityRepresentationGap>,
    pub blockers: BTreeSet<CapabilityBlocker>,
}

/// A machine-readable reason why a representation is not closed.  These are
/// intentionally semantic gaps, not free-text diagnostics, so residency and
/// promotion authorities can fail closed on the exact missing property.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityRepresentationGap {
    DependencyClosure,
    InputContract,
    OutputContract,
    StateSemantics,
    EffectSemantics,
    ConcurrencySemantics,
    PersistenceSemantics,
    BehavioralEquivalence,
    CausalContribution,
    TargetCompatibility,
    TranslationConformance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityBundleStatus {
    Open,
    Blocked,
    RepresentationClosed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityBlocker {
    AcquisitionIncomplete,
    SourceUnsupported,
    DependencyClosureUnresolved,
    EvidenceAuthorityUnavailable,
    ResourceBudgetExceeded,
    UnsupportedSemanticPrimitive,
    PolicyDenied,
    TargetCompatibilityUnresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityBundle {
    schema: CapabilityBundleSchema,
    capability_id: CapabilityId,
    capture_receipt: PrivateFileReference,
    capture_receipt_sha256: CaptureReceiptDigest,
    system_envelope_sha256: SystemEnvelopeDigest,
    representation: CapabilityRepresentation,
    blockers: BTreeSet<CapabilityBlocker>,
    manifest_sha256: CapabilityBundleDigest,
}

impl CapabilityBundle {
    pub fn capability_id(&self) -> &CapabilityId {
        &self.capability_id
    }

    pub fn representation(&self) -> &CapabilityRepresentation {
        &self.representation
    }

    pub fn status(&self) -> CapabilityBundleStatus {
        if !self.blockers.is_empty() {
            CapabilityBundleStatus::Blocked
        } else if matches!(self.representation, CapabilityRepresentation::Closed { .. }) {
            CapabilityBundleStatus::RepresentationClosed
        } else {
            CapabilityBundleStatus::Open
        }
    }

    pub fn blockers(&self) -> &BTreeSet<CapabilityBlocker> {
        &self.blockers
    }

    pub fn manifest_digest(&self) -> &CapabilityBundleDigest {
        &self.manifest_sha256
    }

    pub fn capture_receipt_digest(&self) -> &CaptureReceiptDigest {
        &self.capture_receipt_sha256
    }

    /// Bind a closed structural representation to its exact retained source.
    ///
    /// `RepresentationClosed` means that the IR is internally complete and
    /// authenticated. It is not a claim of behavioral equivalence, causal
    /// sufficiency, target compatibility, translation success, or promotion.
    pub fn create_closed(
        private_root: &Path,
        capability_id: CapabilityId,
        capture_receipt: PrivateFileReference,
        capability_ir: PrivateFileReference,
        blockers: BTreeSet<CapabilityBlocker>,
    ) -> BrainResult<Self> {
        let capture = read_capture(private_root, &capture_receipt)?;
        let envelope = capture.envelope();
        let ir = authenticate_capability_ir(private_root, &capability_ir, envelope)?;
        if ir.capability_id() != &capability_id
            || ir.system_envelope_digest() != envelope.manifest_sha256()
        {
            return Err(BrainError::Integrity(
                "capability_bundle_chain_mismatch".into(),
            ));
        }
        ir.validate_against(envelope)?;
        let mut bundle = Self {
            schema: CapabilityBundleSchema::Current,
            capability_id,
            capture_receipt,
            capture_receipt_sha256: capture.manifest_sha256().clone(),
            system_envelope_sha256: envelope.manifest_sha256().clone(),
            representation: CapabilityRepresentation::Closed {
                capability_ir,
                capability_ir_sha256: ir.manifest_digest().clone(),
            },
            blockers,
            manifest_sha256: CapabilityBundleDigest::draft_marker(),
        };
        bundle.validate_contents(private_root)?;
        bundle.manifest_sha256 = bundle.calculate_digest()?;
        bundle.validate(private_root)?;
        Ok(bundle)
    }

    pub fn create_unmapped(
        private_root: &Path,
        capability_id: CapabilityId,
        capture_receipt: PrivateFileReference,
        blockers: BTreeSet<CapabilityBlocker>,
    ) -> BrainResult<Self> {
        let capture = read_capture(private_root, &capture_receipt)?;
        let mut bundle = Self {
            schema: CapabilityBundleSchema::Current,
            capability_id,
            capture_receipt,
            capture_receipt_sha256: capture.manifest_sha256().clone(),
            system_envelope_sha256: capture.envelope().manifest_sha256().clone(),
            representation: CapabilityRepresentation::Unmapped,
            blockers,
            manifest_sha256: CapabilityBundleDigest::draft_marker(),
        };
        bundle.validate_contents(private_root)?;
        bundle.manifest_sha256 = bundle.calculate_digest()?;
        bundle.validate(private_root)?;
        Ok(bundle)
    }

    /// Creates an explicitly incomplete representation.  This constructor is
    /// the only supported way to retain a useful IR fragment without lying
    /// that the enclosing capability is closed.
    pub fn create_partial(
        private_root: &Path,
        draft: PartialCapabilityBundleDraft,
    ) -> BrainResult<Self> {
        let PartialCapabilityBundleDraft {
            capability_id,
            capture_receipt,
            capability_ir,
            capability_ir_sha256,
            gaps,
            blockers,
        } = draft;
        if gaps.is_empty() || capability_ir.is_some() != capability_ir_sha256.is_some() {
            return Err(BrainError::Invalid(
                "capability_bundle_partial_representation_invalid".into(),
            ));
        }
        let capture = read_capture(private_root, &capture_receipt)?;
        let mut bundle = Self {
            schema: CapabilityBundleSchema::Current,
            capability_id,
            capture_receipt,
            capture_receipt_sha256: capture.manifest_sha256().clone(),
            system_envelope_sha256: capture.envelope().manifest_sha256().clone(),
            representation: CapabilityRepresentation::Partial {
                capability_ir,
                capability_ir_sha256,
                gaps,
            },
            blockers,
            manifest_sha256: CapabilityBundleDigest::draft_marker(),
        };
        bundle.validate_contents(private_root)?;
        bundle.manifest_sha256 = bundle.calculate_digest()?;
        bundle.validate(private_root)?;
        Ok(bundle)
    }

    pub fn validate(&self, private_root: &Path) -> BrainResult<()> {
        if self.manifest_sha256.is_draft() {
            return Err(BrainError::Integrity("capability_bundle_unsealed".into()));
        }
        self.validate_contents(private_root)?;
        if self.calculate_digest()? != self.manifest_sha256 {
            return Err(BrainError::Integrity(
                "capability_bundle_digest_mismatch".into(),
            ));
        }
        Ok(())
    }

    fn validate_contents(&self, private_root: &Path) -> BrainResult<()> {
        let capture = read_capture(private_root, &self.capture_receipt)?;
        let envelope = capture.envelope();
        if capture.manifest_sha256() != &self.capture_receipt_sha256
            || envelope.manifest_sha256() != &self.system_envelope_sha256
        {
            return Err(BrainError::Integrity(
                "capability_bundle_envelope_digest_mismatch".into(),
            ));
        }
        match &self.representation {
            CapabilityRepresentation::Unmapped => {}
            CapabilityRepresentation::Closed {
                capability_ir,
                capability_ir_sha256,
            } => {
                let ir = authenticate_capability_ir(private_root, capability_ir, envelope)?;
                if ir.manifest_digest() != capability_ir_sha256
                    || ir.capability_id() != &self.capability_id
                    || ir.system_envelope_digest() != &self.system_envelope_sha256
                {
                    return Err(BrainError::Integrity(
                        "capability_bundle_ir_binding_mismatch".into(),
                    ));
                }
                ir.validate_against(envelope)?;
            }
            CapabilityRepresentation::Partial {
                capability_ir,
                capability_ir_sha256,
                gaps,
            } => {
                if self.schema != CapabilityBundleSchema::Current
                    || gaps.is_empty()
                    || capability_ir.is_some() != capability_ir_sha256.is_some()
                {
                    return Err(BrainError::Integrity(
                        "capability_bundle_partial_representation_invalid".into(),
                    ));
                }
                if let (Some(capability_ir), Some(capability_ir_sha256)) =
                    (capability_ir, capability_ir_sha256)
                {
                    let ir = authenticate_capability_ir(private_root, capability_ir, envelope)?;
                    if ir.manifest_digest() != capability_ir_sha256
                        || ir.capability_id() != &self.capability_id
                        || ir.system_envelope_digest() != &self.system_envelope_sha256
                    {
                        return Err(BrainError::Integrity(
                            "capability_bundle_partial_ir_binding_mismatch".into(),
                        ));
                    }
                    ir.validate_against(envelope)?;
                }
            }
        }
        Ok(())
    }

    pub fn persist(&self, private_root: &Path) -> BrainResult<PrivateFileReference> {
        let root = verify_internal_private_root(private_root)?;
        self.validate(&root)?;
        let destination = capability_bundle_path(&root, &self.manifest_sha256);
        let bytes = serde_json::to_vec(self)?;
        let sha256 = write_or_verify_immutable(&root, &destination, &bytes)?;
        let reference = PrivateFileReference::new(destination, sha256);
        authenticate_capability_bundle(&root, &reference)?;
        Ok(reference)
    }

    fn calculate_digest(&self) -> BrainResult<CapabilityBundleDigest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = CapabilityBundleDigest::draft_marker();
        Ok(CapabilityBundleDigest::from_computed(domain_digest(
            BUNDLE_DOMAIN,
            &serde_json::to_vec(&unsigned)?,
        )))
    }
}

/// Authenticate the single canonical persisted representation of a capability
/// bundle.  A self-consistent JSON value at an alias path, or a differently
/// encoded copy of the same semantic bundle, is not an authority.
pub fn authenticate_capability_bundle(
    private_root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<CapabilityBundle> {
    let root = verify_internal_private_root(private_root)?;
    let bytes = reference.read_verified_bounded(&root, MAX_CAPABILITY_BUNDLE_BYTES)?;
    let bundle: CapabilityBundle = serde_json::from_slice(&bytes)?;
    bundle.validate(&root)?;
    if reference.path != capability_bundle_path(&root, bundle.manifest_digest()) {
        return Err(BrainError::Integrity(
            "capability_bundle_content_address_mismatch".into(),
        ));
    }
    if serde_json::to_vec(&bundle)? != bytes {
        return Err(BrainError::Integrity(
            "capability_bundle_noncanonical_encoding".into(),
        ));
    }
    Ok(bundle)
}

fn capability_bundle_path(root: &Path, digest: &CapabilityBundleDigest) -> std::path::PathBuf {
    root.join("state/capability_bundles/by-sha")
        .join(format!("{}.json", digest.as_str()))
}

fn read_capture(root: &Path, reference: &PrivateFileReference) -> BrainResult<CaptureReceipt> {
    authenticate_capture_receipt(root, reference)
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::digest_domain(domain, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
    };
    use crate::capability_ir::{
        CapabilityIr, IrNode, OutputBinding, PrimitiveSet, TypedPort, ValueReference, ValueType,
    };
    use crate::content_vault::capture_to_vault;
    use crate::identity::{AcquisitionId, CapabilityNodeId, PrimitiveId};
    use crate::security::secure_dir;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (PathBuf, PrivateFileReference, PrivateFileReference) {
        let root = std::env::temp_dir().join(format!(
            "tidex-bundle-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        let donor = root.with_extension("donor");
        let _ = fs::remove_dir_all(&donor);
        fs::create_dir_all(&root).unwrap();
        secure_dir(&root).unwrap();
        fs::create_dir_all(donor.join("src")).unwrap();
        fs::write(donor.join("src/memory.rs"), b"pub fn choose() {}\n").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("bundle-acquisition.v1").unwrap(),
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
        let capture = capture_to_vault(&donor, &root, &request).unwrap();
        fs::remove_dir_all(&donor).unwrap();
        let ir = CapabilityIr::new(
            CapabilityId::parse("memory.choose:v1").unwrap(),
            capture.envelope(),
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![
                TypedPort::tensor_f64(crate::identity::PortId::parse("scores").unwrap(), vec![4])
                    .unwrap(),
            ],
            vec![IrNode::new(
                CapabilityNodeId::parse("node.choose").unwrap(),
                PrimitiveId::parse("select.arg_max").unwrap(),
                vec![ValueReference::Input {
                    name: crate::identity::PortId::parse("scores").unwrap(),
                }],
                TypedPort::scalar(
                    crate::identity::PortId::parse("choice").unwrap(),
                    ValueType::I64,
                )
                .unwrap(),
                vec![PathBuf::from("src/memory.rs")],
            )
            .unwrap()],
            vec![OutputBinding::new(
                TypedPort::scalar(
                    crate::identity::PortId::parse("selected").unwrap(),
                    ValueType::I64,
                )
                .unwrap(),
                ValueReference::NodeOutput {
                    node_id: CapabilityNodeId::parse("node.choose").unwrap(),
                },
            )
            .unwrap()],
        )
        .unwrap();
        let ir_ref = ir.persist(&root, capture.envelope()).unwrap();
        let capture_ref = capture.persist(&root).unwrap();
        (root, capture_ref, ir_ref)
    }

    #[test]
    fn closed_bundle_binds_ir_to_snapshot_and_preserves_blockers() {
        let (root, envelope, ir) = fixture();
        let bundle = CapabilityBundle::create_closed(
            &root,
            CapabilityId::parse("memory.choose:v1").unwrap(),
            envelope,
            ir,
            BTreeSet::from([
                CapabilityBlocker::DependencyClosureUnresolved,
                CapabilityBlocker::EvidenceAuthorityUnavailable,
                CapabilityBlocker::TargetCompatibilityUnresolved,
            ]),
        )
        .unwrap();
        bundle.validate(&root).unwrap();
        assert_eq!(bundle.status(), CapabilityBundleStatus::Blocked);
        assert!(!bundle.manifest_sha256.is_draft());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_new_or_partial_capability_need_not_invent_a_closed_ir() {
        let (root, envelope, _ir) = fixture();
        let bundle = CapabilityBundle::create_unmapped(
            &root,
            CapabilityId::parse("memory.choose:v1").unwrap(),
            envelope,
            BTreeSet::new(),
        )
        .unwrap();
        assert!(matches!(
            bundle.representation(),
            CapabilityRepresentation::Unmapped
        ));
        assert_eq!(bundle.status(), CapabilityBundleStatus::Open);
        assert!(!bundle.manifest_digest().is_draft());
        bundle.validate(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_representation_records_typed_gaps_and_cannot_promote() {
        let (root, envelope, ir) = fixture();
        let capture = read_capture(&root, &envelope).unwrap();
        let ir_digest = authenticate_capability_ir(&root, &ir, capture.envelope())
            .unwrap()
            .manifest_digest()
            .clone();
        let partial = CapabilityBundle::create_partial(
            &root,
            PartialCapabilityBundleDraft {
                capability_id: CapabilityId::parse("memory.choose:v1").unwrap(),
                capture_receipt: envelope.clone(),
                capability_ir: Some(ir.clone()),
                capability_ir_sha256: Some(ir_digest.clone()),
                gaps: BTreeSet::from([
                    CapabilityRepresentationGap::StateSemantics,
                    CapabilityRepresentationGap::BehavioralEquivalence,
                ]),
                blockers: BTreeSet::from([CapabilityBlocker::EvidenceAuthorityUnavailable]),
            },
        )
        .unwrap();
        assert!(matches!(
            partial.representation(),
            CapabilityRepresentation::Partial { gaps, .. }
                if gaps.contains(&CapabilityRepresentationGap::StateSemantics)
        ));
        assert_eq!(partial.status(), CapabilityBundleStatus::Blocked);
        partial.validate(&root).unwrap();

        assert!(CapabilityBundle::create_partial(
            &root,
            PartialCapabilityBundleDraft {
                capability_id: CapabilityId::parse("memory.choose:v1").unwrap(),
                capture_receipt: envelope,
                capability_ir: Some(ir.clone()),
                capability_ir_sha256: Some(ir_digest),
                gaps: BTreeSet::new(),
                blockers: BTreeSet::new(),
            },
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn closed_representation_does_not_claim_behavioral_equivalence() {
        let (root, envelope, ir) = fixture();
        let bundle = CapabilityBundle::create_closed(
            &root,
            CapabilityId::parse("memory.choose:v1").unwrap(),
            envelope,
            ir,
            BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(
            bundle.status(),
            CapabilityBundleStatus::RepresentationClosed
        );
        assert!(bundle.blockers().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn public_validation_and_persistence_reject_the_internal_draft_marker() {
        let (root, envelope, ir) = fixture();
        let mut bundle = CapabilityBundle::create_closed(
            &root,
            CapabilityId::parse("memory.choose:v1").unwrap(),
            envelope,
            ir,
            BTreeSet::from([CapabilityBlocker::EvidenceAuthorityUnavailable]),
        )
        .unwrap();
        bundle.manifest_sha256 = CapabilityBundleDigest::draft_marker();
        assert!(bundle.validate(&root).is_err());
        assert!(bundle.persist(&root).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn caller_cannot_serialize_a_forged_derived_status() {
        let (root, envelope, ir) = fixture();
        let bundle = CapabilityBundle::create_closed(
            &root,
            CapabilityId::parse("memory.choose:v1").unwrap(),
            envelope,
            ir,
            BTreeSet::new(),
        )
        .unwrap();
        let mut value = serde_json::to_value(bundle).unwrap();
        value.as_object_mut().unwrap().insert(
            "status".into(),
            serde_json::json!("behaviorally_conformant"),
        );
        assert!(serde_json::from_value::<CapabilityBundle>(value).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn status_is_derived_from_representation_and_blockers() {
        let (root, envelope, ir) = fixture();
        let blocked = CapabilityBundle::create_closed(
            &root,
            CapabilityId::parse("memory.choose:v1").unwrap(),
            envelope,
            ir,
            BTreeSet::from([CapabilityBlocker::PolicyDenied]),
        )
        .unwrap();
        assert_eq!(blocked.status(), CapabilityBundleStatus::Blocked);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn persisted_bundle_rejects_aliases_and_noncanonical_json() {
        let (root, capture, ir) = fixture();
        let bundle = CapabilityBundle::create_closed(
            &root,
            CapabilityId::parse("memory.choose:v1").unwrap(),
            capture,
            ir,
            BTreeSet::new(),
        )
        .unwrap();
        let canonical = bundle.persist(&root).unwrap();
        assert_eq!(
            authenticate_capability_bundle(&root, &canonical).unwrap(),
            bundle
        );

        let bytes = serde_json::to_vec(&bundle).unwrap();
        let alias_path = root.join("state/capability_bundles/alias.json");
        let alias_sha = write_or_verify_immutable(&root, &alias_path, &bytes).unwrap();
        let alias = PrivateFileReference::new(alias_path, alias_sha);
        assert!(authenticate_capability_bundle(&root, &alias).is_err());

        let pretty_root = root.with_extension("pretty");
        fs::create_dir_all(&pretty_root).unwrap();
        secure_dir(&pretty_root).unwrap();
        let pretty = serde_json::to_vec_pretty(&bundle).unwrap();
        let pretty_path = capability_bundle_path(&pretty_root, bundle.manifest_digest());
        let pretty_sha = write_or_verify_immutable(&pretty_root, &pretty_path, &pretty).unwrap();
        let pretty_reference = PrivateFileReference::new(pretty_path, pretty_sha);
        assert!(authenticate_capability_bundle(&pretty_root, &pretty_reference).is_err());

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(pretty_root).unwrap();
    }
}
