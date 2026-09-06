//! Canonical hand-off from a completed adaptive-learning session to the
//! TIDE-X reconstruction authority.
//!
//! This module deliberately does not reconstruct, replace an active corpus,
//! commit a bank, or activate a model. It turns the immutable adaptive receipt
//! chain plus immutable sealed-representation receipt into one typed,
//! re-verifiable engine input. Python can therefore never provide a free-form
//! observation list to promotion.

use crate::authority::{read_existing_private_file_bounded, PrivateFileReference};
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
use crate::security::verify_internal_private_root;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::fs;
use std::path::{Path, PathBuf};

pub const LEARNING_FINALIZATION_INPUT_SCHEMA: &str = "cerebro.tidex.learning_finalization_input/v1";
const MAX_LEARNING_FINALIZATION_JSON_BYTES: u64 = 64 * 1024 * 1024;

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
    let raw = reference.read_verified_bounded(root, MAX_LEARNING_FINALIZATION_JSON_BYTES)?;
    let observation: DeltaObservation = serde_json::from_slice(&raw)?;
    Ok((reference.path.clone(), raw, observation))
}

fn checked_representation_receipt_reference(
    root: &Path,
    raw: &Path,
) -> BrainResult<PrivateFileReference> {
    let bytes =
        read_existing_private_file_bounded(root, raw, MAX_LEARNING_FINALIZATION_JSON_BYTES)?;
    Ok(PrivateFileReference::new(
        raw.to_path_buf(),
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
    let root = verify_internal_private_root(root.as_ref())?;
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
pub(crate) mod tests {
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

    use crate::authority::write_or_verify_immutable;
    use crate::contracts::ExperimentLineage;
    use crate::digest::ObservationRecordDigest;
    use crate::identity::CapabilityId;
    use crate::learning_orchestrator::{
        assimilate_persistent_learning_evidence, issue_next_persistent_learning_aperture,
        start_persistent_adaptive_learning, AdaptiveLearningPolicy, LearningExperimentEvidence,
        LearningTarget,
    };
    use crate::linalg::dot;
    use crate::representation_evidence::{
        record_representation_evidence, RepresentationCapture,
        RepresentationEvidenceInstallRequest, RepresentationEvidenceInstallTarget,
        RepresentationShift, SealedRepresentationProtocol,
    };

    pub(crate) fn make_test_scenario(
        root: &Path,
        session_name: &str,
    ) -> (SessionId, PathBuf, PathBuf) {
        crate::security::secure_dir(root).unwrap();
        let target = LearningTarget {
            target_id: LearningTargetId::parse("fin-target").unwrap(),
            capability_ids: vec![
                CapabilityId::parse("cap-1").unwrap(),
                CapabilityId::parse("cap-2").unwrap(),
            ],
            candidate_budget: 2,
            plan_steps: 2,
            noise_variance: 0.1,
            cost_weight: 0.0,
            risk_weight: 0.0,
        };
        let policy = AdaptiveLearningPolicy {
            schema: "cerebro.tidex.adaptive_learning_policy/v1".into(),
            outcome_utility_weight: 1.0,
            maximize_observed_value: true,
        };
        let started =
            start_persistent_adaptive_learning(root, session_name, &target, &policy).unwrap();

        let mut install_targets = Vec::new();
        let mut first_obs_path = PathBuf::new();
        let mut shifts = Vec::new();

        for i in 1..=2 {
            let issued = issue_next_persistent_learning_aperture(root, session_name).unwrap();
            let step = issued.receipt.cycle.pending_step.as_ref().unwrap();

            let layout_raw = br#"{"schema":"cerebro.tidex.parameter_layout/test"}"#.to_vec();
            let layout_digest = Sha256Digest::digest_bytes(&layout_raw);
            let layout_path = root
                .join("state/parameter_layouts/by-sha")
                .join(format!("{layout_digest}.json"));
            write_or_verify_immutable(root, &layout_path, &layout_raw).unwrap();
            let dense =
                crate::artifact::create_content_addressed_dvec(root, &[0.25, -0.5]).unwrap();

            let observation = DeltaObservation {
                observation_id: ObservationId::parse(format!("obs-fin-{i}")).unwrap(),
                from_checkpoint: "base".into(),
                to_checkpoint: "candidate".into(),
                generation: 1,
                delta: vec![0.1 * i as f64, 0.2, 0.3],
                functional_response: vec![0.5, -0.2 * i as f64],
                confounders: Vec::new(),
                reliability: 0.95,
                independence_group: step.aperture_id.to_string(),
                experiment_lineage: ExperimentLineage::default(),
                dense_artifact: Some(dense),
                parameter_layout_sha256: Some(layout_digest),
                representation_artifact: None,
                representation_protocol_sha256: None,
                provenance_digest: crate::digest::ProvenanceDigest::from(
                    Sha256Digest::digest_bytes(format!("fin-prov-{i}").as_bytes()),
                ),
            };
            let observation_raw = serde_json::to_vec_pretty(&observation).unwrap();
            let observation_digest = Sha256Digest::digest_bytes(&observation_raw);
            let observation_path = root
                .join("state/observations")
                .join(format!("{}.json", observation.observation_id));
            write_or_verify_immutable(root, &observation_path, &observation_raw).unwrap();
            if i == 1 {
                first_obs_path = observation_path.clone();
            }

            let support_path = root
                .join("state/experiment_support")
                .join(format!("{}.json", step.aperture_id));
            let support_raw = format!(r#"{{"step":{i}}}"#).into_bytes();
            write_or_verify_immutable(root, &support_path, &support_raw).unwrap();

            let evidence = LearningExperimentEvidence {
                schema: "cerebro.tidex.learning_experiment_evidence/v1".into(),
                session_id: SessionId::parse(session_name).unwrap(),
                target_digest: started.receipt.target_digest.clone(),
                aperture_id: step.aperture_id.clone(),
                observed_value: dot(&step.capability_weights, &observation.functional_response)
                    .unwrap(),
                observation_id: observation.observation_id.clone(),
                observation: EvidenceReference {
                    path: observation_path.clone(),
                    sha256: observation_digest.clone(),
                },
                evidence_files: vec![EvidenceReference {
                    path: support_path.clone(),
                    sha256: Sha256Digest::digest_bytes(&support_raw),
                }],
            };
            let evidence_path = root
                .join("state/experiment_envelopes")
                .join(format!("{}.json", step.aperture_id));
            write_or_verify_immutable(
                root,
                &evidence_path,
                &serde_json::to_vec_pretty(&evidence).unwrap(),
            )
            .unwrap();

            assimilate_persistent_learning_evidence(root, session_name, &evidence_path).unwrap();

            let destination_path = root
                .join("state/representation_evidence/installed-observations")
                .join(format!("obs-fin-{i}-installed.json"));

            install_targets.push(RepresentationEvidenceInstallTarget {
                observation_id: observation.observation_id.clone(),
                source_observation_path: observation_path.to_string_lossy().into_owned(),
                source_observation_sha256: ObservationRecordDigest::from(observation_digest),
                destination_observation_path: destination_path.to_string_lossy().into_owned(),
            });

            shifts.push(RepresentationShift {
                observation_id: observation.observation_id.clone(),
                raw_dimension: 8,
                shift: vec![0.1 * i as f64, -0.2 * i as f64, 0.3 * i as f64],
            });
        }

        let capture = RepresentationCapture {
            schema: crate::representation_evidence::REPRESENTATION_CAPTURE_SCHEMA.to_string(),
            observations: shifts,
        };
        let protocol = SealedRepresentationProtocol {
            schema: crate::representation_evidence::REPRESENTATION_PROTOCOL_SCHEMA.to_string(),
            source_representation_sha256:
                crate::representation_evidence::representation_capture_sha256(&capture).unwrap(),
            probe_sha256: Sha256Digest::digest_bytes(b"probe"),
            probe_text_sha256: Sha256Digest::digest_bytes(b"probe"),
            forbidden_vocabulary_sha256: Sha256Digest::digest_bytes(b"vocab"),
            task_labels_used: false,
            probe_vocabulary_overlap: Vec::new(),
            probe_count: 2,
            layer_count: 2,
            hidden_dim: 2,
            raw_dimension_per_observation: 8,
            sketch_dim: 3,
            sketch_seed: 7,
        };

        let request = RepresentationEvidenceInstallRequest {
            schema: crate::representation_evidence::REPRESENTATION_EVIDENCE_REQUEST_SCHEMA
                .to_string(),
            protocol,
            capture,
            installations: install_targets,
        };
        let request_path = root.join("state/rep-install-request.json");
        write_or_verify_immutable(
            root,
            &request_path,
            &serde_json::to_vec_pretty(&request).unwrap(),
        )
        .unwrap();

        let rep_receipt = record_representation_evidence(root, &request_path).unwrap();
        let rep_receipt_path = root
            .join("state/representation_evidence/receipts/by-request-sha")
            .join(format!("{}.json", rep_receipt.request_sha256));

        (
            SessionId::parse(session_name).unwrap(),
            rep_receipt_path,
            first_obs_path,
        )
    }

    #[test]
    fn prepare_and_verify_learning_finalization_succeeds_with_valid_receipts() {
        let root = temporary_root();
        let (session_id, rep_receipt_path, _) = make_test_scenario(&root, "session-valid");

        let input =
            prepare_learning_finalization(&root, session_id.as_str(), &rep_receipt_path).unwrap();
        assert_eq!(input.schema, LEARNING_FINALIZATION_INPUT_SCHEMA);
        assert_eq!(input.session_id, session_id);
        assert_eq!(input.observations.len(), 2);
        assert_eq!(input.representation_observation_bindings.len(), 2);
        assert!(input.observations[0].representation_artifact.is_some());

        let input_hash = learning_finalization_input_sha256(&input).unwrap();
        assert_ne!(input_hash, Sha256Digest::zero());

        let verified = verify_learning_finalization_input(&root, &input).unwrap();
        assert_eq!(input, verified);

        let mut invalid_schema = input.clone();
        invalid_schema.schema = "invalid.schema/v0".into();
        assert!(verify_learning_finalization_input(&root, &invalid_schema).is_err());

        let mut tampered = input.clone();
        tampered.session_id = SessionId::parse("session-other").unwrap();
        assert!(verify_learning_finalization_input(&root, &tampered).is_err());

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn finalization_rejects_incomplete_or_pending_adaptive_session() {
        let root = temporary_root();
        crate::security::secure_dir(&root).unwrap();
        let target = LearningTarget {
            target_id: LearningTargetId::parse("fin-pending-target").unwrap(),
            capability_ids: vec![
                CapabilityId::parse("cap-1").unwrap(),
                CapabilityId::parse("cap-2").unwrap(),
            ],
            candidate_budget: 2,
            plan_steps: 2,
            noise_variance: 0.1,
            cost_weight: 0.0,
            risk_weight: 0.0,
        };
        let policy = AdaptiveLearningPolicy {
            schema: "cerebro.tidex.adaptive_learning_policy/v1".into(),
            outcome_utility_weight: 1.0,
            maximize_observed_value: true,
        };
        let _started =
            start_persistent_adaptive_learning(&root, "session-pending", &target, &policy).unwrap();
        let dummy_receipt = root.join("dummy-receipt.json");
        let err =
            prepare_learning_finalization(&root, "session-pending", &dummy_receipt).unwrap_err();
        assert!(matches!(err, BrainError::Integrity(msg) if msg.contains("incomplete_or_pending")));
        fs::remove_dir_all(&root).unwrap();
    }
}
