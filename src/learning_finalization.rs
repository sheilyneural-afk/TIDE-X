//! Canonical hand-off from a completed adaptive-learning session to the
//! TIDE-X reconstruction authority.
//!
//! This module deliberately does not reconstruct, replace an active corpus,
//! commit a bank, or activate a model. It turns the immutable adaptive receipt
//! chain plus immutable sealed-representation receipt into one typed,
//! re-verifiable engine input. Python can therefore never provide a free-form
//! observation list to promotion.

use crate::authority::PrivateFileReference;
use crate::contracts::DeltaObservation;
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::identity::{LearningTargetId, ObservationId, SessionId};
use crate::learning_orchestrator::{
    load_persistent_adaptive_learning_receipt, AdaptiveLearningEventKind,
};
use crate::representation_evidence::{
    load_verified_representation_evidence_receipt, InstalledRepresentationEvidence,
};
use crate::security::verify_private_root;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const LEARNING_FINALIZATION_INPUT_SCHEMA: &str = "cerebro.tidex.learning_finalization_input/v1";

/// Immutable mapping from an adaptive source observation to its sole staged
/// sealed-representation destination. The engine finalizes only the latter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepresentationObservationBinding {
    pub observation_id: ObservationId,
    /// Raw adaptive observation identity. The exact bytes remain the aperture
    /// evidence identity even after representation installation.
    pub adaptive_source_observation: PrivateFileReference,
    /// Sealed representation-enhanced observation that may enter the promoted
    /// corpus after this binding is replayed.
    pub representation_destination_observation: PrivateFileReference,
    /// SHA-256 of canonical `serde_json::to_vec(destination)` semantics. This
    /// deliberately differs from the staged file SHA above: runtime
    /// composition admits only the exact observation in the promoted corpus.
    pub promoted_observation_semantic_sha256: Sha256Digest,
}

/// The only observation set an engine finalization may accept for an adaptive
/// learning session. `verify_learning_finalization_input` always regenerates this from the
/// exact current adaptive and representation receipt chains before an engine
/// may act on it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningFinalizationInput {
    pub schema: String,
    pub session_id: SessionId,
    pub adaptive_receipt_sha256: Sha256Digest,
    pub target_id: LearningTargetId,
    pub target_digest: Sha256Digest,
    pub policy_digest: Sha256Digest,
    /// Content hashes of immutable learning-evidence envelopes in their
    /// receipt's canonical assimilation order.
    pub completed_evidence_sha256: Vec<Sha256Digest>,
    /// Immutable receipt that records every sealed representation install.
    pub representation_evidence_receipt: PrivateFileReference,
    pub representation_protocol_sha256: Sha256Digest,
    pub representation_installations_sha256: Sha256Digest,
    /// Exact source-to-destination mapping that was re-verified before this
    /// input was emitted. There is no raw-observation finalization mode.
    pub representation_observation_bindings: Vec<RepresentationObservationBinding>,
    /// These are destination observations from the representation receipt,
    /// never the raw observations referenced by adaptive evidence.
    pub observations: Vec<DeltaObservation>,
}

fn semantic_observation_sha256(observation: &DeltaObservation) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_bytes(&serde_json::to_vec(
        observation,
    )?))
}

/// Stable digest for an engine-owned finalization input. This is not authority
/// by itself; the engine must replay the complete receipt chain before use.
pub fn learning_finalization_input_sha256(
    input: &LearningFinalizationInput,
) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_bytes(&serde_json::to_vec(input)?))
}

fn read_confined_observation(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<(PathBuf, Vec<u8>, DeltaObservation)> {
    let (path, raw) = reference.read_verified_with_path(root)?;
    let observation: DeltaObservation = serde_json::from_slice(&raw)?;
    Ok((path, raw, observation))
}

fn checked_representation_receipt_reference(
    root: &Path,
    raw: &Path,
) -> BrainResult<PrivateFileReference> {
    let path = crate::authority::existing_regular_file_under_root(root, raw)?;
    let bytes = fs::read(&path)?;
    Ok(PrivateFileReference::new(
        path,
        Sha256Digest::digest_bytes(&bytes),
    ))
}

struct AdaptiveLearningSourceSet {
    adaptive_receipt_sha256: Sha256Digest,
    target_id: LearningTargetId,
    target_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    completed_evidence_sha256: Vec<Sha256Digest>,
    observations: BTreeMap<ObservationId, (PrivateFileReference, DeltaObservation)>,
}

fn source_observations_from_adaptive_cycle(
    root: &Path,
    session_id: &SessionId,
) -> BrainResult<AdaptiveLearningSourceSet> {
    // The adaptive loader replays the complete receipt chain, ledger bindings,
    // evidence envelopes and outcome derivations before anything is eligible
    // for finalization.
    let loaded = load_persistent_adaptive_learning_receipt(root, session_id.as_str())?;
    let adaptive_receipt_sha256 = Sha256Digest::parse(&loaded.receipt_sha256)?;
    let receipt = loaded.receipt;
    let cycle = receipt.cycle;
    if receipt.event_kind != AdaptiveLearningEventKind::ResultAssimilated
        || cycle.pending_step.is_some()
        || cycle.session.completed_aperture_ids.len() != cycle.target.plan_steps
        || cycle.completed_evidence.len() != cycle.target.plan_steps
        || cycle.completed_evidence_sha256.len() != cycle.target.plan_steps
    {
        return Err(BrainError::Integrity(
            "learning_finalization_session_incomplete_or_pending".into(),
        ));
    }

    let target_digest = Sha256Digest::parse(&cycle.target_digest)?;
    let policy_digest = Sha256Digest::parse(&cycle.policy_digest)?;
    let completed_evidence_sha256 = cycle
        .completed_evidence_sha256
        .iter()
        .map(Sha256Digest::parse)
        .collect::<BrainResult<Vec<_>>>()?;

    let mut evidence_digests = BTreeSet::new();
    let mut observation_digests = BTreeSet::new();
    let mut observation_paths = BTreeSet::new();
    let mut sources = BTreeMap::new();
    for (evidence, evidence_sha256) in cycle
        .completed_evidence
        .iter()
        .zip(&completed_evidence_sha256)
    {
        if !evidence_digests.insert(evidence_sha256.clone()) {
            return Err(BrainError::Integrity(
                "learning_finalization_completed_evidence_duplicate".into(),
            ));
        }
        let (path, raw, observation) = read_confined_observation(root, &evidence.observation)?;
        let digest = Sha256Digest::digest_bytes(&raw);
        let observation_id = observation.observation_id.clone();
        if observation.observation_id != evidence.observation_id
            || digest != evidence.observation.sha256
            || !observation_digests.insert(digest.clone())
            || !observation_paths.insert(path.clone())
            || sources
                .insert(
                    observation_id,
                    (PrivateFileReference::new(path, digest), observation),
                )
                .is_some()
        {
            return Err(BrainError::Integrity(
                "learning_finalization_source_observation_identity_invalid".into(),
            ));
        }
    }
    if sources.is_empty() {
        return Err(BrainError::Integrity(
            "learning_finalization_observations_empty".into(),
        ));
    }
    Ok(AdaptiveLearningSourceSet {
        adaptive_receipt_sha256,
        target_id: cycle.target.target_id,
        target_digest,
        policy_digest,
        completed_evidence_sha256,
        observations: sources,
    })
}

fn representation_installation_map(
    installations: &[InstalledRepresentationEvidence],
) -> BrainResult<BTreeMap<ObservationId, &InstalledRepresentationEvidence>> {
    let mut by_id = BTreeMap::new();
    for installation in installations {
        let observation_id = ObservationId::parse(&installation.observation_id)?;
        if by_id.insert(observation_id, installation).is_some() {
            return Err(BrainError::Integrity(
                "learning_finalization_representation_installation_duplicate".into(),
            ));
        }
    }
    Ok(by_id)
}

fn prepare_learning_finalization_under_root(
    root: &Path,
    session_id: &SessionId,
    representation_evidence_receipt_path: &Path,
) -> BrainResult<LearningFinalizationInput> {
    let source_set = source_observations_from_adaptive_cycle(root, session_id)?;

    let representation_evidence_receipt =
        checked_representation_receipt_reference(root, representation_evidence_receipt_path)?;
    let representation =
        load_verified_representation_evidence_receipt(root, &representation_evidence_receipt.path)?;
    // The representation loader performs semantic and ledger replay. Verify the
    // exact receipt bytes once more after replay so a concurrent replacement
    // cannot silently change the finalization input.
    representation_evidence_receipt.verify(root)?;

    let representation_protocol_sha256 =
        Sha256Digest::parse(&representation.representation_protocol_sha256)?;
    let representation_installations_sha256 =
        Sha256Digest::parse(&representation.installations_sha256)?;
    let installations = representation_installation_map(&representation.installations)?;
    if source_set
        .observations
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>()
        != installations.keys().cloned().collect::<BTreeSet<_>>()
    {
        return Err(BrainError::Integrity(
            "learning_finalization_representation_source_set_mismatch".into(),
        ));
    }

    let mut bindings = Vec::with_capacity(source_set.observations.len());
    let mut observations = Vec::with_capacity(source_set.observations.len());
    for (observation_id, (source_reference, source)) in source_set.observations {
        let installation = installations.get(&observation_id).ok_or_else(|| {
            BrainError::Integrity(
                "learning_finalization_representation_installation_missing".into(),
            )
        })?;
        let installed_source = PrivateFileReference::new(
            PathBuf::from(&installation.source_observation_path),
            Sha256Digest::parse(&installation.source_observation_sha256)?,
        );
        if installed_source != source_reference {
            return Err(BrainError::Integrity(
                "learning_finalization_representation_source_binding_mismatch".into(),
            ));
        }

        let destination_reference = PrivateFileReference::new(
            PathBuf::from(&installation.destination_observation_path),
            Sha256Digest::parse(&installation.destination_observation_sha256)?,
        );
        let (_destination_path, _destination_raw, destination) =
            read_confined_observation(root, &destination_reference)?;
        if destination.observation_id != observation_id
            || destination.representation_artifact.as_ref()
                != Some(&installation.representation_artifact)
            || destination.representation_protocol_sha256.as_deref()
                != Some(representation_protocol_sha256.as_str())
        {
            return Err(BrainError::Integrity(
                "learning_finalization_representation_destination_binding_mismatch".into(),
            ));
        }
        let mut semantic_destination = destination.clone();
        semantic_destination.representation_artifact = None;
        semantic_destination.representation_protocol_sha256 = None;
        if semantic_destination != source {
            return Err(BrainError::Integrity(
                "learning_finalization_representation_changed_nonrepresentation_semantics".into(),
            ));
        }
        bindings.push(RepresentationObservationBinding {
            observation_id,
            adaptive_source_observation: source_reference,
            representation_destination_observation: destination_reference,
            promoted_observation_semantic_sha256: semantic_observation_sha256(&destination)?,
        });
        observations.push(destination);
    }
    if observations.is_empty()
        || observations.iter().any(|observation| {
            observation.representation_artifact.is_none()
                || observation.representation_protocol_sha256.as_deref()
                    != Some(representation_protocol_sha256.as_str())
        })
    {
        return Err(BrainError::Integrity(
            "learning_finalization_representation_destinations_incomplete".into(),
        ));
    }

    Ok(LearningFinalizationInput {
        schema: LEARNING_FINALIZATION_INPUT_SCHEMA.into(),
        session_id: session_id.clone(),
        adaptive_receipt_sha256: source_set.adaptive_receipt_sha256,
        target_id: source_set.target_id,
        target_digest: source_set.target_digest,
        policy_digest: source_set.policy_digest,
        completed_evidence_sha256: source_set.completed_evidence_sha256,
        representation_evidence_receipt,
        representation_protocol_sha256,
        representation_installations_sha256,
        representation_observation_bindings: bindings,
        observations,
    })
}

/// Prepare a finalization input solely from the current receipt-backed
/// adaptive-learning session and a separately recorded sealed-representation
/// receipt. The public boundary is permanently confined to CEREBRO's private
/// root; raw adaptive observations are never a finalization fallback.
pub fn prepare_learning_finalization(
    root: impl AsRef<Path>,
    session_id: &str,
    representation_evidence_receipt_path: impl AsRef<Path>,
) -> BrainResult<LearningFinalizationInput> {
    let root = verify_private_root(root.as_ref())?;
    let session_id = SessionId::parse(session_id)?;
    prepare_learning_finalization_under_root(
        &root,
        &session_id,
        representation_evidence_receipt_path.as_ref(),
    )
}

/// Rebind a supplied typed input to the exact *current* adaptive and
/// representation receipt chains. The engine must use the returned canonical
/// value rather than trust the caller's in-memory value, so callers cannot
/// forge a session, substitute an observation set, or finalize stale evidence.
pub fn verify_learning_finalization_input(
    root: impl AsRef<Path>,
    input: &LearningFinalizationInput,
) -> BrainResult<LearningFinalizationInput> {
    if input.schema != LEARNING_FINALIZATION_INPUT_SCHEMA {
        return Err(BrainError::Integrity(
            "learning_finalization_input_schema_invalid".into(),
        ));
    }
    let canonical = prepare_learning_finalization(
        root,
        input.session_id.as_str(),
        &input.representation_evidence_receipt.path,
    )?;
    if &canonical != input {
        return Err(BrainError::Integrity(
            "learning_finalization_input_not_current_canonical_receipt_binding".into(),
        ));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_orchestrator::EvidenceReference;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-learning-lifecycle-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        root
    }

    #[test]
    fn untrusted_or_tampered_observation_reference_fails_without_mutation() {
        let root = temporary_root();
        let state = root.join("state");
        fs::create_dir(&state).unwrap();
        let inside = state.join("observation.json");
        let original = b"immutable-observation-bytes".to_vec();
        fs::write(&inside, &original).unwrap();
        let tampered = EvidenceReference {
            path: inside.clone(),
            sha256: Sha256Digest::parse("00".repeat(32)).unwrap(),
        };
        assert!(read_confined_observation(&root, &tampered).is_err());
        assert_eq!(fs::read(&inside).unwrap(), original);

        let outside = std::env::temp_dir().join(format!(
            "cerebro-learning-lifecycle-outside-{}",
            std::process::id()
        ));
        fs::write(&outside, b"outside").unwrap();
        assert!(crate::authority::existing_regular_file_under_root(&root, &outside).is_err());
        fs::remove_file(&outside).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn promoted_semantic_digest_is_not_the_staged_file_digest() {
        let observation = DeltaObservation {
            observation_id: ObservationId::parse("obs-bridge").unwrap(),
            from_checkpoint: "base".into(),
            to_checkpoint: "candidate".into(),
            generation: 1,
            delta: vec![0.25, -0.5],
            functional_response: vec![0.75],
            confounders: Vec::new(),
            reliability: 0.9,
            independence_group: "aperture-bridge".into(),
            experiment_lineage: Default::default(),
            dense_artifact: None,
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: crate::digest::ProvenanceDigest::from(Sha256Digest::digest_bytes(
                b"obs-bridge",
            )),
        };
        let canonical = semantic_observation_sha256(&observation).unwrap();
        let mut staged_file_bytes = serde_json::to_vec_pretty(&observation).unwrap();
        staged_file_bytes.push(b'\n');
        assert_eq!(
            canonical,
            Sha256Digest::digest_bytes(&serde_json::to_vec(&observation).unwrap())
        );
        assert_ne!(canonical, Sha256Digest::digest_bytes(&staged_file_bytes));
    }
}
