#![allow(clippy::needless_range_loop)]
#![allow(unused_imports)]
//! TIDE-X engine authorities.
//!
//! The former monolith is split by authority, not by file size:
//! analysis, durable store/head, corpus transitions, and runtime certification.

mod analysis;
mod runtime;
mod store;
mod support;
mod transition;
mod types;

pub use crate::engine_head::{
    CanonicalEngineHead, CorpusTransitionRecovery, CorpusTransitionRecoveryOutcome,
};
pub use support::{
    load_verified_governed_composition_receipt, load_verified_learning_finalization_receipt,
};
pub use types::{
    ControllerExecutionReceipt, ControllerInvocation, GovernedCompositionProtection,
    GovernedCompositionReceipt, LearningFinalizationReceipt, ReconstructionReport,
    RecordedControllerExecution, RecordedGovernedCognitiveComposition, RecordedGovernedComposition,
    SleepReport,
};

pub(super) use support::*;
pub(super) use types::*;

pub(super) use crate::aperture_independence::{
    estimate_aperture_independence, ApertureIndependenceReport,
};
pub(super) use crate::artifact::{
    derive_content_addressed_dvec_combination, inspect_dvec, read_dvec_f32, read_f64_artifact,
    ArtifactWriteAuthority, DeltaArtifactRef,
};
pub(super) use crate::authority::{
    ensure_private_directory, ensure_private_parent, existing_directory_under_root,
    existing_regular_file_under_root, inspect_private_directory, list_existing_private_directory,
    move_private_directory_transactional, move_private_file_transactional,
    read_existing_private_file_bounded, read_untrusted_private_file_bounded,
    replace_private_file_atomic, root_relative_path, with_private_authority_lock,
    write_or_verify_immutable, PrivateFileReference,
};
pub(super) use crate::block_tomography::{
    reconstruct_structured_geometry, ParameterBlockLayout, ParameterLayoutAuthority,
    StructuredSource,
};
pub(super) use crate::causal_credit::{certified_causal_priority_weights, CausalCreditReport};
pub(super) use crate::cognitive_field::{
    CognitiveFieldConfig, CognitiveFieldDrive, CognitiveFieldState, DynamicCognitiveField,
    FieldRoutingDecision,
};
pub(super) use crate::confounders::remove_confounders;
pub(super) use crate::contracts::{
    BrainConfig, DeltaObservation, PromotionBlocker, PromotionDecision, ProtectedCortex,
    ReconstructionInverseMode, SkillBank, SkillField,
};
pub(super) use crate::digest::{
    AnalysisVersionDigest, CanonicalEngineHeadDigest, CausalCreditDigest, ConfigDigest,
    CorpusDigest, EvidenceBundleDigest, MemoryDigest, ObservationRecordDigest,
    ParameterLayoutDigest, ReportDigest, Sha256Digest, SkillBankDigest, SourceTreeDigest,
};
pub(super) use crate::dual_space::{
    analyze_dual_space, DualSpaceAnalysisConfig, DualSpaceModel, RepresentationObservation,
};
pub(super) use crate::engine_head::{
    CorpusTransitionJournal, CorpusTransitionPhase, CANONICAL_ENGINE_HEAD_MAX_BYTES,
    CANONICAL_ENGINE_HEAD_SCHEMA, CORPUS_TRANSITION_JOURNAL_MAX_BYTES, HARD_MAX_ENGINE_REVISION,
};
pub(super) use crate::error::{BrainError, BrainResult};
pub(super) use crate::functional::{attach_signatures, fit_functional_map};
pub(super) use crate::identifiability::{resolution_map, ResolutionMap};
pub(super) use crate::identity::{SessionId, SkillId};
pub(super) use crate::learned_controller::{
    load_persisted_runtime_learned_controller, RuntimeLearnedController,
};
pub(super) use crate::learning_finalization::{
    learning_finalization_input_sha256, prepare_learning_finalization,
    verify_learning_finalization_input, LearningFinalizationInput,
    RepresentationObservationBinding,
};
pub(super) use crate::ledger;
pub(super) use crate::linalg::{compensated_sum, cosine, norm, stable_rms, Matrix};
pub(super) use crate::memory::{build_memory_snapshot, memory_artifact_path};
pub(super) use crate::persistent::reconstruct_persistent_skill_fields;
pub(super) use crate::protected::{project_to_safe_subspace, ProtectionResult};
pub(super) use crate::protected_map::{load_protected_cortex, ProtectedMapArtifactReport};
pub(super) use crate::sbas::reconstruct_trajectory;
pub(super) use crate::security::{verify_internal_private_root, verify_private_root};
pub(super) use crate::sleep_diagnostics::{diagnose_consolidation, SleepConsolidationDiagnostics};
pub(super) use crate::sleep_evidence::{
    load_sleep_evidence, verify_sleep_evidence, SleepEvidenceExpectation, SleepEvidenceVerification,
};
pub(super) use crate::tomography::{
    align_incoming_identities, assimilate_bank, reconcile_full_corpus, reconstruct_skill_fields,
};
pub(super) use crate::trust_region::{
    apply_causal_priority_trust_region, TrustRegionAllocationPolicy, TrustRegionResult,
};
pub(super) use crate::validation::source_support_indices;
pub(super) use serde::de::DeserializeOwned;
pub(super) use serde::{Deserialize, Serialize};
pub(super) use serde_json::{json, Value};
pub(super) use sha2::{Digest, Sha256};
pub(super) use std::collections::{BTreeMap, BTreeSet};
pub(super) use std::fs;
pub(super) use std::path::{Path, PathBuf};

const MAX_ENGINE_JSON_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SKILL_BANK_BYTES: u64 = 256 * 1024 * 1024;
const MAX_OBSERVATION_RECORD_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ENGINE_OBSERVATIONS: usize = 65_536;
const MAX_ENGINE_PARAMETER_DIMENSION: usize = 67_108_864;
const MAX_ENGINE_TOTAL_DELTA_ELEMENTS: usize = 134_217_728;
const MAX_ENGINE_FUNCTIONAL_RESPONSE_DIMENSION: usize = 1_048_576;
const MAX_ENGINE_SKILL_FIELDS: usize = 65_536;
const MAX_ENGINE_INDEPENDENCE_GROUPS: usize = 65_536;
const MAX_ENGINE_CONFOUNDERS_PER_OBSERVATION: usize = 4_096;
const MAX_ENGINE_TEXT_BYTES: usize = 16 * 1024;
const MAX_ENGINE_IRLS_ROUNDS: usize = 4_096;

#[derive(Debug, Clone)]
pub struct BrainEngine {
    pub(super) root: PathBuf,
    pub(super) config: BrainConfig,
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
    pub(super) fn require_canonical_runtime_config(&self) -> BrainResult<()> {
        if self.config != BrainConfig::default() {
            return Err(BrainError::Integrity(
                "noncanonical_runtime_config_forbidden".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn with_engine_authority<T, F>(&self, operation: F) -> BrainResult<T>
    where
        F: FnOnce() -> BrainResult<T>,
    {
        let lock = self.root.join("state/engine_authority.lock");
        with_private_authority_lock(&self.root, &lock, operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::{ParameterLayoutDigest, ProvenanceDigest, Sha256Digest};
    use crate::engine_head::CorpusTransitionRecoveryOutcome;
    use crate::identity::{LineageId, ObservationId, ReconstructionId, SessionId, SkillId};
    use std::collections::BTreeMap;
    use std::os::unix::fs::PermissionsExt;

    fn sample_observation(id: &str, delta: Vec<f64>) -> DeltaObservation {
        DeltaObservation {
            observation_id: ObservationId::parse(id).unwrap(),
            from_checkpoint: "checkpoint-a".into(),
            to_checkpoint: "checkpoint-b".into(),
            generation: 7,
            delta,
            functional_response: vec![],
            confounders: vec![],
            reliability: 0.9,
            independence_group: "group-1".into(),
            experiment_lineage: Default::default(),
            dense_artifact: None,
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: ProvenanceDigest::from(Sha256Digest::digest_bytes(id.as_bytes())),
        }
    }

    fn sample_field(skill_id: &str) -> SkillField {
        SkillField {
            skill_id: SkillId::parse(skill_id).unwrap(),
            reconstruction_id: ReconstructionId::parse("recon-1").unwrap(),
            lineage_id: LineageId::parse("lineage-1").unwrap(),
            generation_created: 1,
            direction: vec![1.0, 0.0],
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: vec![],
            singular_value: 1.0,
            explained_variance: 0.5,
            persistence: 0.8,
            coherence: 0.9,
            uncertainty: 0.1,
            evidence_support_digests: vec![],
            support: 0,
            functional_signature: vec![],
            parent_skill_ids: vec![],
        }
    }

    #[test]
    fn valid_digest_accepts_lowercase_and_rejects_uppercase() {
        let digest = Sha256Digest::digest_bytes(b"hello");
        assert!(valid_digest(digest.as_str()));
        assert!(!valid_digest(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF"
        ));
    }

    #[test]
    fn canonical_observations_are_sorted_and_observation_set_digest_is_order_invariant() {
        let obs_a = sample_observation("obs-b", vec![1.0, 0.0]);
        let obs_b = sample_observation("obs-a", vec![0.0, 1.0]);

        let sorted = canonical_observations(&[obs_a.clone(), obs_b.clone()]);
        assert_eq!(sorted[0].observation_id.as_str(), "obs-a");
        assert_eq!(sorted[1].observation_id.as_str(), "obs-b");

        let digest_1 = observation_set_digest(&[obs_a.clone(), obs_b.clone()]).unwrap();
        let digest_2 = observation_set_digest(&[obs_b, obs_a]).unwrap();
        assert_eq!(digest_1, digest_2);
    }

    #[test]
    fn attach_evidence_support_tracks_support_digest_and_rejects_shape_mismatch() {
        let mut fields = vec![sample_field("skill-alpha")];
        let observations = vec![
            sample_observation("obs-a", vec![0.2, 0.4]),
            sample_observation("obs-b", vec![0.5, -0.1]),
        ];

        attach_evidence_support(&mut fields, &[vec![1.0, 0.0]], &observations).unwrap();
        assert_eq!(fields[0].support, 1);
        assert_eq!(fields[0].evidence_support_digests.len(), 1);

        let mut invalid_fields = vec![sample_field("skill-alpha")];
        let err =
            attach_evidence_support(&mut invalid_fields, &[vec![1.0, 0.0, 0.0]], &observations)
                .unwrap_err();
        assert!(matches!(err, BrainError::Integrity(_)));
    }

    #[test]
    fn archive_and_sleep_keys_are_bound_to_their_input_values() {
        assert!(archive_label_is_allowed("skill_bank.json"));
        assert!(!archive_label_is_allowed("forbidden.txt"));

        let key = sleep_operation_key(
            "analysis-key",
            None,
            CertificationStatus::Certified,
            Some("bank-sha"),
        );
        assert_eq!(key.len(), 64);
        assert!(valid_digest(&key));
    }

    #[test]
    fn governed_composition_operation_key_changes_when_inputs_change() {
        let parameter_layout_artifact = Sha256Digest::digest_bytes(b"layout-artifact");
        let parameter_layout = ParameterLayoutDigest::from(Sha256Digest::digest_bytes(b"layout"));
        let activation_a = BTreeMap::from([(SkillId::parse("skill-alpha").unwrap(), 1.0)]);
        let activation_b = BTreeMap::from([(SkillId::parse("skill-alpha").unwrap(), 2.0)]);

        let op_a = GovernedCompositionOperation {
            report_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            active_bank_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            evidence_bundle_sha256:
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            causal_credit_sha256:
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            parameter_layout_artifact_sha256: &parameter_layout_artifact,
            parameter_layout_sha256: &parameter_layout,
            activation: &activation_a,
            projected_delta_sha256:
                "1111111111111111111111111111111111111111111111111111111111111111",
            source_observation_sha256:
                "2222222222222222222222222222222222222222222222222222222222222222",
        };

        let op_b = GovernedCompositionOperation {
            report_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            active_bank_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            evidence_bundle_sha256:
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            causal_credit_sha256:
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            parameter_layout_artifact_sha256: &parameter_layout_artifact,
            parameter_layout_sha256: &parameter_layout,
            activation: &activation_b,
            projected_delta_sha256:
                "1111111111111111111111111111111111111111111111111111111111111111",
            source_observation_sha256:
                "2222222222222222222222222222222222222222222222222222222222222222",
        };

        let key_a = governed_composition_operation_key(&op_a).unwrap();
        let key_b = governed_composition_operation_key(&op_b).unwrap();
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn load_bank_rejects_unsafe_magnitudes_before_bank_energy_scales_them() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("cerebro-bank-load-{unique}"));
        let state_dir = root.join("state");
        fs::create_dir_all(&state_dir).unwrap();

        let bank = SkillBank {
            generation: 1,
            fields: vec![SkillField {
                skill_id: SkillId::parse("skill-alpha").unwrap(),
                reconstruction_id: ReconstructionId::parse("recon-1").unwrap(),
                lineage_id: LineageId::parse("lineage-1").unwrap(),
                generation_created: 1,
                direction: vec![f64::MAX, 1.0],
                structured_geometry: None,
                dense_materialization: None,
                parameter_layout_sha256: None,
                representation_signature: vec![],
                singular_value: 1.0,
                explained_variance: 0.5,
                persistence: 0.8,
                coherence: 0.9,
                uncertainty: 0.1,
                evidence_support_digests: vec![ObservationRecordDigest::from(
                    Sha256Digest::digest_bytes(b"obs-1"),
                )],
                support: 1,
                functional_signature: vec![1.0],
                parent_skill_ids: vec![],
            }],
        };
        fs::write(
            state_dir.join("skill_bank.json"),
            serde_json::to_vec(&bank).unwrap(),
        )
        .unwrap();

        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let err = engine.load_bank().unwrap_err();
        let _ = fs::remove_dir_all(&root);
        assert!(matches!(err, BrainError::Integrity(_)));
    }

    #[test]
    fn skill_subspace_overlap_rejects_overflowing_projected_energy() {
        let mut field = sample_field("skill-alpha");
        field.direction = vec![f64::MAX.sqrt(), 0.0];
        let truth = vec![vec![f64::MAX.sqrt(), 0.0]];
        let err = BrainEngine::skill_subspace_overlap(&[field], &truth).unwrap_err();
        assert!(matches!(err, BrainError::Numerical(message) if message.contains("overflow")));
    }

    #[test]
    fn bank_energy_rejects_overflowing_field_magnitudes() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("cerebro-bank-energy-{unique}"));
        let state_dir = root.join("state");
        fs::create_dir_all(&state_dir).unwrap();

        let bank = SkillBank {
            generation: 1,
            fields: vec![SkillField {
                skill_id: SkillId::parse("skill-alpha").unwrap(),
                reconstruction_id: ReconstructionId::parse("recon-1").unwrap(),
                lineage_id: LineageId::parse("lineage-1").unwrap(),
                generation_created: 1,
                direction: vec![f64::MAX.sqrt(), 1.0],
                structured_geometry: None,
                dense_materialization: None,
                parameter_layout_sha256: None,
                representation_signature: vec![],
                singular_value: 1.0,
                explained_variance: 0.5,
                persistence: 0.8,
                coherence: 0.9,
                uncertainty: 0.1,
                evidence_support_digests: vec![ObservationRecordDigest::from(
                    Sha256Digest::digest_bytes(b"obs-1"),
                )],
                support: 1,
                functional_signature: vec![1.0],
                parent_skill_ids: vec![],
            }],
        };
        fs::write(
            state_dir.join("skill_bank.json"),
            serde_json::to_vec(&bank).unwrap(),
        )
        .unwrap();

        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let err = engine.bank_energy().unwrap_err();
        let _ = fs::remove_dir_all(&root);
        assert!(matches!(err, BrainError::Integrity(_)));
    }

    fn isolated_engine_root(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-engine-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let mut permissions = fs::metadata(&root).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&root, permissions).unwrap();
        root
    }

    #[test]
    fn persist_observations_publishes_only_a_complete_staged_corpus() {
        let root = isolated_engine_root("persist-stage");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let observations = vec![
            sample_observation("obs-a", vec![1.0, 0.0]),
            sample_observation("obs-b", vec![0.0, 1.0]),
        ];
        let digests = engine.persist_observations(&observations).unwrap();
        assert_eq!(digests.len(), 2);
        let live = canonical_observations(&engine.load_persisted_observations().unwrap());
        assert_eq!(live.len(), 2);
        assert!(!root
            .join("state/observation_staging")
            .join(digest_json(&digests).unwrap())
            .exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recover_rolls_back_an_intent_that_never_retired_the_prior_corpus() {
        let root = isolated_engine_root("recover-intent");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let operation_key = Sha256Digest::digest_bytes(b"recover-intent");
        let inflight = root
            .join("state/corpus_transitions/inflight")
            .join(operation_key.as_str());
        fs::create_dir_all(&inflight).unwrap();
        let intent = LearningCorpusTransitionIntent {
            schema: "cerebro.tidex.learning_corpus_transition_intent/v1".into(),
            operation_key: operation_key.clone(),
            session_id: SessionId::parse("session-recover").unwrap(),
            adaptive_receipt_sha256: Sha256Digest::digest_bytes(b"adaptive"),
            learning_finalization_input_sha256: Sha256Digest::digest_bytes(b"input"),
            representation_evidence_receipt: PrivateFileReference::new(
                root.join("state/representation.json"),
                Sha256Digest::digest_bytes(b"evidence"),
            ),
            representation_protocol_sha256: Sha256Digest::digest_bytes(b"protocol"),
            representation_observation_bindings_sha256: Sha256Digest::digest_bytes(b"bindings"),
            prior_corpus_digest: Sha256Digest::digest_bytes(b"prior"),
            prior_observation_count: 6,
            new_corpus_digest: Sha256Digest::digest_bytes(b"new"),
            new_observation_count: 6,
            archive_dir: root
                .join("state/corpus_transitions/by-operation")
                .join(operation_key.as_str()),
        };
        fs::write(
            inflight.join("intent.json"),
            serialize_pretty_line(&intent).unwrap(),
        )
        .unwrap();

        let recovery = engine.recover_incomplete_corpus_transition().unwrap();
        assert_eq!(
            recovery.outcome,
            CorpusTransitionRecoveryOutcome::RolledBackIntent
        );
        assert_eq!(recovery.operation_key.as_ref(), Some(&operation_key));
        assert!(!inflight.exists());
        assert!(root
            .join("state/corpus_transitions/aborted")
            .join(operation_key.as_str())
            .join("intent.json")
            .exists());
        let head = engine.load_canonical_head().unwrap().unwrap();
        assert!(head.incomplete_transition.is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn status_omits_physical_private_root_and_names_the_head() {
        let root = isolated_engine_root("status-head");
        let state_dir = root.join("state");
        fs::create_dir_all(&state_dir).unwrap();
        let mut field = sample_field("skill-alpha");
        field.support = 1;
        field.evidence_support_digests = vec![ObservationRecordDigest::from(
            Sha256Digest::digest_bytes(b"obs-1"),
        )];
        let bank = SkillBank {
            generation: 3,
            fields: vec![field],
        };
        fs::write(
            state_dir.join("skill_bank.json"),
            serde_json::to_vec(&bank).unwrap(),
        )
        .unwrap();
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        engine
            .advance_canonical_engine_head(HeadIncomplete::Clear, None, None)
            .unwrap();
        let status = engine.status().unwrap();
        assert_eq!(status["schema"], "cerebro.tidex.status/v3");
        assert!(status.get("private_root").is_none());
        assert_eq!(status["revision"], 0);
        assert_eq!(status["skill_generation"], 3);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn canonical_head_rejects_live_pointer_drift_after_publication() {
        let root = isolated_engine_root("head-live-drift");
        let state_dir = root.join("state");
        fs::create_dir_all(&state_dir).unwrap();
        let mut field = sample_field("skill-alpha");
        field.support = 1;
        field.evidence_support_digests = vec![ObservationRecordDigest::from(
            Sha256Digest::digest_bytes(b"obs-head"),
        )];
        let mut bank = SkillBank {
            generation: 1,
            fields: vec![field],
        };
        fs::write(
            state_dir.join("skill_bank.json"),
            serialize_pretty_line(&bank).unwrap(),
        )
        .unwrap();
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        engine
            .advance_canonical_engine_head(HeadIncomplete::Clear, None, None)
            .unwrap();
        engine.verify_current_canonical_engine_head().unwrap();

        bank.generation = 2;
        fs::write(
            state_dir.join("skill_bank.json"),
            serialize_pretty_line(&bank).unwrap(),
        )
        .unwrap();
        let err = engine.verify_current_canonical_engine_head().unwrap_err();
        assert!(
            matches!(err, BrainError::Integrity(message) if message.contains("live_authority_mismatch"))
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn canonical_head_rejects_tampered_parent_history() {
        let root = isolated_engine_root("head-history-tamper");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let first = engine
            .advance_canonical_engine_head(HeadIncomplete::Clear, None, None)
            .unwrap();
        engine
            .advance_canonical_engine_head(HeadIncomplete::Clear, None, None)
            .unwrap();
        fs::write(
            engine.canonical_engine_head_history_path(&first.manifest_digest),
            b"{}\n",
        )
        .unwrap();
        assert!(engine.verify_current_canonical_engine_head().is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn canonical_head_compare_and_swap_rejects_a_stale_parent() {
        let root = isolated_engine_root("head-cas");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let first = engine
            .advance_canonical_engine_head(HeadIncomplete::Clear, None, None)
            .unwrap();
        let _second = engine
            .advance_canonical_engine_head(HeadIncomplete::Clear, None, None)
            .unwrap();
        let stale = CanonicalEngineHead {
            revision: 1,
            parent_revision: Some(first.revision),
            parent_digest: Some(first.manifest_digest.clone()),
            incomplete_transition: Some(Sha256Digest::digest_bytes(b"stale")),
            manifest_digest: CanonicalEngineHeadDigest::from(Sha256Digest::zero()),
            ..first.clone()
        }
        .seal()
        .unwrap();
        let err = engine
            .publish_canonical_head(Some(&first), &stale)
            .unwrap_err();
        assert!(
            matches!(err, BrainError::Integrity(message) if message.contains("compare_and_swap"))
        );
        let _ = fs::remove_dir_all(&root);
    }

    fn dense_analysis_fixture(
        label: &str,
        observation_count: usize,
    ) -> (PathBuf, BrainEngine, Vec<DeltaObservation>, Sha256Digest) {
        let root = isolated_engine_root(label);
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig {
                require_structured_geometry_for_promotion: false,
                require_dual_space_for_promotion: false,
                ..BrainConfig::default()
            },
        };
        let layout = crate::block_tomography::ParameterBlockLayout::from_shapes(&[
            crate::block_tomography::BlockShapeSpec {
                name: "weights".into(),
                shape: vec![4],
                count: 4,
            },
        ])
        .unwrap();
        let layout_bytes = serde_json::to_vec(&layout).unwrap();
        let layout_sha = Sha256Digest::digest_bytes(&layout_bytes);
        let layout_path = root
            .join("state/parameter_layouts/by-sha")
            .join(format!("{layout_sha}.json"));
        write_or_verify_immutable(&root, &layout_path, &layout_bytes).unwrap();

        let writer = ArtifactWriteAuthority::for_internal_root(&root).unwrap();
        let observations = (0..observation_count)
            .map(|index| {
                let mut observation = sample_observation(
                    &format!("dense-obs-{index}"),
                    vec![index as f64 + 1.0, index as f64 + 2.0],
                );
                observation.parameter_layout_sha256 = Some(layout_sha.clone());
                observation.dense_artifact = Some(
                    writer
                        .create_content_addressed_dvec(&[
                            index as f32 + 0.1,
                            index as f32 + 0.2,
                            index as f32 + 0.3,
                            index as f32 + 0.4,
                        ])
                        .unwrap(),
                );
                observation
            })
            .collect();
        (root, engine, observations, layout_sha)
    }

    #[test]
    fn dense_analysis_sources_materialize_and_rederive_on_same_authority() {
        let (root, engine, observations, layout_sha) = dense_analysis_fixture("dense-analysis", 2);
        let (_, layout, sources) = engine
            .structured_sources(&observations)
            .unwrap()
            .expect("dense evidence is present");
        assert_eq!(layout.total_parameter_count, 4);
        assert_eq!(sources.len(), observations.len());

        let mut fields = vec![sample_field("dense-field")];
        fields[0].parameter_layout_sha256 = Some(layout_sha.clone());
        let mixtures = vec![vec![0.25, 0.75]];
        engine
            .materialize_dense_fields(&mut fields, &mixtures, &observations)
            .unwrap();
        let materialized = fields[0]
            .dense_materialization
            .clone()
            .expect("materialized dense field");
        assert_eq!(materialized.parameter_count, 4);
        engine
            .verify_dense_field_materializations(&fields, &mixtures, &observations)
            .unwrap();

        let mut missing = fields.clone();
        missing[0].dense_materialization = None;
        assert!(matches!(
            engine.verify_dense_field_materializations(&missing, &mixtures, &observations),
            Err(BrainError::Integrity(message)) if message.contains("materialization_missing")
        ));
        assert!(engine
            .materialize_dense_fields(&mut fields, &[vec![1.0]], &observations)
            .is_err());

        let mut wrong_layout = fields.clone();
        wrong_layout[0].parameter_layout_sha256 = Some(Sha256Digest::digest_bytes(b"wrong-layout"));
        assert!(engine
            .verify_dense_field_materializations(&wrong_layout, &mixtures, &observations)
            .is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn representation_observations_authenticate_protocol_and_artifacts() {
        let (root, mut engine, mut observations, _) = dense_analysis_fixture("dual-analysis", 2);
        engine.config.require_dual_space_for_promotion = true;
        let writer = ArtifactWriteAuthority::for_internal_root(&root).unwrap();
        let protocol = serde_json::json!({
            "schema": "cerebro.tidex.representation_protocol/v1",
            "source_representation_sha256": Sha256Digest::digest_bytes(b"capture"),
            "probe_sha256": Sha256Digest::digest_bytes(b"probe"),
            "probe_text_sha256": Sha256Digest::digest_bytes(b"probe"),
            "forbidden_vocabulary_sha256": Sha256Digest::digest_bytes(b"forbidden"),
            "task_labels_used": false,
            "probe_vocabulary_overlap": [],
            "probe_count": 1,
            "layer_count": 1,
            "hidden_dim": 3,
            "raw_dimension_per_observation": 3,
            "sketch_dim": 3,
            "sketch_seed": 11
        });
        let protocol_bytes = serde_json::to_vec(&protocol).unwrap();
        let protocol_sha = Sha256Digest::digest_bytes(&protocol_bytes);
        let protocol_path = root
            .join("state/representation_protocols/by-sha")
            .join(format!("{protocol_sha}.json"));
        write_or_verify_immutable(&root, &protocol_path, &protocol_bytes).unwrap();
        let typed_protocol =
            crate::digest::RepresentationProtocolDigest::from(protocol_sha.clone());
        for (index, observation) in observations.iter_mut().enumerate() {
            observation.representation_artifact = Some(
                writer
                    .create_content_addressed_f64(&[
                        index as f64 + 0.1,
                        index as f64 + 0.2,
                        index as f64 + 0.3,
                    ])
                    .unwrap(),
            );
            observation.representation_protocol_sha256 = Some(typed_protocol.clone());
        }
        let (loaded_protocol, representations) = engine
            .representation_observations(&observations)
            .unwrap()
            .expect("dual-space evidence is present");
        assert_eq!(loaded_protocol, protocol_sha.as_str());
        assert_eq!(representations.len(), observations.len());
        assert_eq!(representations[0].shift.len(), 3);

        let mut partial = observations.clone();
        partial[0].representation_artifact = None;
        partial[0].representation_protocol_sha256 = None;
        assert!(matches!(
            engine.representation_observations(&partial),
            Err(BrainError::Invalid(message)) if message == "dual_space_partial_representation_evidence"
        ));

        let mut wrong_count = observations.clone();
        wrong_count[0]
            .representation_artifact
            .as_mut()
            .unwrap()
            .element_count = 2;
        assert!(matches!(
            engine.representation_observations(&wrong_count),
            Err(BrainError::Integrity(message)) if message.contains("representation_artifact_count_mismatch")
        ));

        let bad_protocol = serde_json::json!({
            "schema": "cerebro.tidex.representation_protocol/v1",
            "source_representation_sha256": Sha256Digest::digest_bytes(b"bad-capture"),
            "probe_sha256": Sha256Digest::digest_bytes(b"probe-a"),
            "probe_text_sha256": Sha256Digest::digest_bytes(b"probe-b"),
            "forbidden_vocabulary_sha256": Sha256Digest::digest_bytes(b"bad-forbidden"),
            "task_labels_used": false,
            "probe_vocabulary_overlap": [],
            "probe_count": 1,
            "layer_count": 1,
            "hidden_dim": 3,
            "raw_dimension_per_observation": 3,
            "sketch_dim": 3,
            "sketch_seed": 12
        });
        let bad_bytes = serde_json::to_vec(&bad_protocol).unwrap();
        let bad_sha = Sha256Digest::digest_bytes(&bad_bytes);
        write_or_verify_immutable(
            &root,
            &root
                .join("state/representation_protocols/by-sha")
                .join(format!("{bad_sha}.json")),
            &bad_bytes,
        )
        .unwrap();
        let bad_typed = crate::digest::RepresentationProtocolDigest::from(bad_sha);
        let mut bad_observations = observations.clone();
        for observation in &mut bad_observations {
            observation.representation_protocol_sha256 = Some(bad_typed.clone());
        }
        assert!(matches!(
            engine.representation_observations(&bad_observations),
            Err(BrainError::Integrity(message)) if message == "representation_probe_text_digest_mismatch"
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn analysis_fail_closed_branches_reject_incomplete_dense_and_dual_evidence() {
        use crate::digest::RepresentationProtocolDigest;

        let (root, engine, observations, layout_sha) =
            dense_analysis_fixture("analysis-branches", 2);
        let strict_engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let bare = vec![
            sample_observation("bare-a", vec![1.0, 0.0]),
            sample_observation("bare-b", vec![0.0, 1.0]),
        ];
        assert!(matches!(
            strict_engine.structured_sources(&bare),
            Err(BrainError::Invalid(message)) if message == "structured_geometry_dense_evidence_required"
        ));

        let mut partial = observations.clone();
        partial[0].dense_artifact = None;
        partial[0].parameter_layout_sha256 = None;
        assert!(matches!(
            engine.structured_sources(&partial),
            Err(BrainError::Invalid(message)) if message == "structured_geometry_partial_dense_evidence"
        ));
        let mut multiple = observations.clone();
        multiple[1].parameter_layout_sha256 = Some(Sha256Digest::digest_bytes(b"other-layout"));
        assert!(matches!(
            engine.structured_sources(&multiple),
            Err(BrainError::Invalid(message)) if message == "structured_geometry_multiple_parameter_layouts"
        ));

        let layout5 = crate::block_tomography::ParameterBlockLayout::from_shapes(&[
            crate::block_tomography::BlockShapeSpec {
                name: "weights-five".into(),
                shape: vec![5],
                count: 5,
            },
        ])
        .unwrap();
        let layout5_bytes = serde_json::to_vec(&layout5).unwrap();
        let layout5_sha = Sha256Digest::digest_bytes(&layout5_bytes);
        write_or_verify_immutable(
            &root,
            &root
                .join("state/parameter_layouts/by-sha")
                .join(format!("{layout5_sha}.json")),
            &layout5_bytes,
        )
        .unwrap();
        let mut mismatched_count = observations.clone();
        for observation in &mut mismatched_count {
            observation.parameter_layout_sha256 = Some(layout5_sha.clone());
        }
        assert!(matches!(
            engine.structured_sources(&mismatched_count),
            Err(BrainError::Integrity(message)) if message.contains("dense_artifact_reference_mismatch")
        ));

        let mut field = sample_field("branch-field");
        field.parameter_layout_sha256 = Some(layout_sha.clone());
        let mut strict_field = vec![field.clone()];
        assert!(matches!(
            strict_engine.materialize_dense_fields(&mut strict_field, &[vec![0.5, 0.5]], &observations),
            Err(BrainError::Integrity(message)) if message.contains("dense_field_missing_structured_geometry")
        ));
        let mut wrong_layout_field = vec![field.clone()];
        wrong_layout_field[0].parameter_layout_sha256 = Some(Sha256Digest::digest_bytes(b"wrong"));
        assert!(matches!(
            engine.materialize_dense_fields(&mut wrong_layout_field, &[vec![0.5, 0.5]], &observations),
            Err(BrainError::Integrity(message)) if message.contains("dense_field_layout_identity_mismatch")
        ));
        let mut zero_support_field = vec![field.clone()];
        assert!(matches!(
            engine.materialize_dense_fields(&mut zero_support_field, &[vec![0.0, 0.0]], &observations),
            Err(BrainError::Numerical(message)) if message.contains("dense_field_materialization_support_empty")
        ));
        assert!(matches!(
            engine.verify_dense_field_materializations(&[], &[], &observations),
            Err(BrainError::Integrity(message)) if message == "dense_field_materialization_verification_shape"
        ));

        let mut materialized = vec![field];
        engine
            .materialize_dense_fields(&mut materialized, &[vec![0.5, 0.5]], &observations)
            .unwrap();
        assert!(matches!(
            strict_engine.verify_dense_field_materializations(&materialized, &[vec![0.5, 0.5]], &observations),
            Err(BrainError::Integrity(message)) if message.contains("dense_field_missing_structured_geometry")
        ));
        assert!(matches!(
            engine.verify_dense_field_materializations(&materialized, &[vec![0.0, 0.0]], &observations),
            Err(BrainError::Integrity(message)) if message.contains("dense_field_materialization_support_empty")
        ));
        let mut wrong_reference = materialized.clone();
        wrong_reference[0]
            .dense_materialization
            .as_mut()
            .unwrap()
            .sha256 = Sha256Digest::digest_bytes(b"different-materialization");
        assert!(matches!(
            engine.verify_dense_field_materializations(&wrong_reference, &[vec![0.5, 0.5]], &observations),
            Err(BrainError::Integrity(message)) if message.contains("dense_field_materialization_rederivation_mismatch")
        ));

        let writer = ArtifactWriteAuthority::for_internal_root(&root).unwrap();
        let rep_artifacts = [
            writer
                .create_content_addressed_f64(&[0.1, 0.2, 0.3])
                .unwrap(),
            writer
                .create_content_addressed_f64(&[0.4, 0.5, 0.6])
                .unwrap(),
        ];
        let base_protocol = serde_json::json!({
            "schema": "cerebro.tidex.representation_protocol/v1",
            "source_representation_sha256": Sha256Digest::digest_bytes(b"branch-capture"),
            "probe_sha256": Sha256Digest::digest_bytes(b"branch-probe"),
            "probe_text_sha256": Sha256Digest::digest_bytes(b"branch-probe"),
            "forbidden_vocabulary_sha256": Sha256Digest::digest_bytes(b"branch-forbidden"),
            "task_labels_used": false,
            "probe_vocabulary_overlap": [],
            "probe_count": 1,
            "layer_count": 1,
            "hidden_dim": 3,
            "raw_dimension_per_observation": 3,
            "sketch_dim": 3,
            "sketch_seed": 31
        });
        let persist_protocol = |value: &serde_json::Value| {
            let bytes = serde_json::to_vec(value).unwrap();
            let sha = Sha256Digest::digest_bytes(&bytes);
            write_or_verify_immutable(
                &root,
                &root
                    .join("state/representation_protocols/by-sha")
                    .join(format!("{sha}.json")),
                &bytes,
            )
            .unwrap();
            RepresentationProtocolDigest::from(sha)
        };
        let valid_protocol = persist_protocol(&base_protocol);
        let with_protocol = |protocol: RepresentationProtocolDigest| {
            observations
                .iter()
                .enumerate()
                .map(|(index, source)| {
                    let mut observation = source.clone();
                    observation.representation_artifact = Some(rep_artifacts[index].clone());
                    observation.representation_protocol_sha256 = Some(protocol.clone());
                    observation
                })
                .collect::<Vec<_>>()
        };

        assert!(matches!(
            strict_engine.representation_observations(&bare),
            Err(BrainError::Invalid(message)) if message == "dual_space_representation_evidence_required"
        ));
        let mut missing_reference = with_protocol(valid_protocol.clone());
        for observation in &mut missing_reference {
            observation.representation_protocol_sha256 = None;
        }
        assert!(matches!(
            strict_engine.representation_observations(&missing_reference),
            Err(BrainError::Invalid(message)) if message == "representation_protocol_reference_missing"
        ));
        let mut multiple_protocols = with_protocol(valid_protocol.clone());
        multiple_protocols[1].representation_protocol_sha256 = Some(
            RepresentationProtocolDigest::from(Sha256Digest::digest_bytes(b"other-protocol")),
        );
        assert!(matches!(
            strict_engine.representation_observations(&multiple_protocols),
            Err(BrainError::Invalid(message)) if message == "dual_space_multiple_representation_protocols"
        ));
        let missing_protocol =
            RepresentationProtocolDigest::from(Sha256Digest::digest_bytes(b"missing-protocol"));
        assert!(matches!(
            strict_engine.representation_observations(&with_protocol(missing_protocol)),
            Err(BrainError::Integrity(message)) if message == "representation_protocol_artifact_invalid"
        ));

        let mut semantic_bad = base_protocol.clone();
        semantic_bad["task_labels_used"] = serde_json::Value::Bool(true);
        let semantic_bad = persist_protocol(&semantic_bad);
        assert!(matches!(
            strict_engine.representation_observations(&with_protocol(semantic_bad)),
            Err(BrainError::Integrity(message)) if message == "representation_protocol_semantic_contract_invalid"
        ));
        let mut missing_digest = base_protocol.clone();
        missing_digest
            .as_object_mut()
            .unwrap()
            .remove("probe_sha256");
        let missing_digest = persist_protocol(&missing_digest);
        assert!(matches!(
            strict_engine.representation_observations(&with_protocol(missing_digest)),
            Err(BrainError::Integrity(message)) if message.contains("representation_protocol_missing_digest")
        ));
        let mut invalid_digest = base_protocol.clone();
        invalid_digest["probe_sha256"] = serde_json::Value::String("invalid".into());
        let invalid_digest = persist_protocol(&invalid_digest);
        assert!(matches!(
            strict_engine.representation_observations(&with_protocol(invalid_digest)),
            Err(BrainError::Integrity(message)) if message.contains("representation_protocol_invalid_digest")
        ));

        for (key, message) in [
            ("sketch_dim", "representation_protocol_sketch_dim_invalid"),
            ("probe_count", "representation_protocol_probe_count_invalid"),
            ("layer_count", "representation_protocol_layer_count_invalid"),
            ("hidden_dim", "representation_protocol_hidden_dim_invalid"),
            (
                "raw_dimension_per_observation",
                "representation_protocol_raw_dim_invalid",
            ),
        ] {
            let mut invalid = base_protocol.clone();
            invalid[key] = serde_json::json!(0);
            let invalid = persist_protocol(&invalid);
            assert!(matches!(
                strict_engine.representation_observations(&with_protocol(invalid)),
                Err(BrainError::Integrity(actual)) if actual == message
            ));
        }
        let mut raw_mismatch = base_protocol.clone();
        raw_mismatch["raw_dimension_per_observation"] = serde_json::json!(4);
        let raw_mismatch = persist_protocol(&raw_mismatch);
        assert!(matches!(
            strict_engine.representation_observations(&with_protocol(raw_mismatch)),
            Err(BrainError::Integrity(message)) if message == "representation_protocol_raw_dimension_mismatch"
        ));

        let mut no_artifact = with_protocol(valid_protocol.clone());
        for observation in &mut no_artifact {
            observation.representation_artifact = None;
        }
        assert!(matches!(
            strict_engine.representation_observations(&no_artifact),
            Err(BrainError::Invalid(message)) if message == "representation_artifact_reference_missing"
        ));
        let mut invalid_path = with_protocol(valid_protocol);
        invalid_path[0]
            .representation_artifact
            .as_mut()
            .unwrap()
            .path = root.join("missing.f64bin");
        assert!(matches!(
            strict_engine.representation_observations(&invalid_path),
            Err(BrainError::Integrity(message)) if message == "representation_artifact_path_invalid"
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn observation_validation_rejects_each_authority_contract_violation() {
        use crate::contracts::{ConfounderValue, ExperimentLineage};

        let root = isolated_engine_root("observation-validation");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig {
                min_observations: 2,
                require_structured_geometry_for_promotion: false,
                require_dual_space_for_promotion: false,
                ..BrainConfig::default()
            },
        };
        let lineage = ExperimentLineage {
            run_id: "run-1".into(),
            replicate_id: "rep-1".into(),
            randomization_id: "rand-1".into(),
            dataset_split_digest: Sha256Digest::digest_bytes(b"dataset").to_string(),
            initial_checkpoint_digest: Sha256Digest::digest_bytes(b"checkpoint").to_string(),
            optimizer_config_digest: Sha256Digest::digest_bytes(b"optimizer").to_string(),
            template_config_digest: Sha256Digest::digest_bytes(b"template").to_string(),
        };
        let mut base = vec![
            sample_observation("valid-a", vec![1.0, 0.0]),
            sample_observation("valid-b", vec![0.0, 1.0]),
        ];
        for (index, observation) in base.iter_mut().enumerate() {
            observation.from_checkpoint = format!("checkpoint-{index}");
            observation.to_checkpoint = format!("checkpoint-{}", index + 1);
            observation.independence_group = format!("group-{index}");
            observation.experiment_lineage = lineage.clone();
            observation.functional_response = vec![index as f64];
        }
        assert_eq!(engine.validate_observations(&base).unwrap(), 2);

        let mut case = base.clone();
        case[1].observation_id = case[0].observation_id.clone();
        assert!(
            matches!(engine.validate_observations(&case), Err(BrainError::Invalid(m)) if m == "duplicate_observation_id")
        );
        let mut case = base.clone();
        case[0].to_checkpoint = case[0].from_checkpoint.clone();
        assert!(
            matches!(engine.validate_observations(&case), Err(BrainError::Invalid(m)) if m == "checkpoint_edge_invalid")
        );
        let mut case = base.clone();
        case[0].delta = vec![1.0];
        assert!(
            matches!(engine.validate_observations(&case), Err(BrainError::Invalid(m)) if m == "delta_invalid")
        );
        let mut case = base.clone();
        case[0].reliability = 0.0;
        assert!(
            matches!(engine.validate_observations(&case), Err(BrainError::Invalid(m)) if m == "reliability_invalid")
        );
        let mut case = base.clone();
        case[0].independence_group.clear();
        assert!(
            matches!(engine.validate_observations(&case), Err(BrainError::Invalid(m)) if m == "independence_group_required")
        );
        let mut case = base.clone();
        case[0].confounders = vec![
            ConfounderValue {
                name: "same".into(),
                value: 1.0,
            },
            ConfounderValue {
                name: "same".into(),
                value: 2.0,
            },
        ];
        assert!(
            matches!(engine.validate_observations(&case), Err(BrainError::Invalid(m)) if m == "confounder_duplicate_name")
        );
        let mut case = base.clone();
        case[0].experiment_lineage.run_id.clear();
        assert!(
            matches!(engine.validate_observations(&case), Err(BrainError::Invalid(m)) if m == "experiment_lineage_required")
        );
        let mut case = base;
        case[0].functional_response = vec![f64::NAN];
        assert!(
            matches!(engine.validate_observations(&case), Err(BrainError::Invalid(m)) if m == "functional_response_contract_invalid")
        );
        let _ = fs::remove_dir_all(root);
    }

    fn runtime_field_with_artifact(
        root: &Path,
        layout_sha: &Sha256Digest,
        skill_id: &str,
        values: &[f32],
    ) -> SkillField {
        let writer = ArtifactWriteAuthority::for_internal_root(root).unwrap();
        let dense = writer.create_content_addressed_dvec(values).unwrap();
        let skill_id = SkillId::parse(skill_id).unwrap();
        SkillField {
            skill_id: skill_id.clone(),
            reconstruction_id: ReconstructionId::parse("runtime-reconstruction").unwrap(),
            lineage_id: LineageId::parse("runtime-lineage").unwrap(),
            generation_created: 1,
            direction: vec![1.0, 0.0],
            structured_geometry: Some(crate::contracts::SkillSubspaceGeometry {
                skill_id,
                source_support_indices: vec![0],
                blocks: vec![],
                max_local_rank: 1,
                mean_effective_rank: 1.0,
            }),
            dense_materialization: Some(dense),
            parameter_layout_sha256: Some(layout_sha.clone()),
            representation_signature: vec![],
            singular_value: 1.0,
            explained_variance: 1.0,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.0,
            evidence_support_digests: vec![ObservationRecordDigest::from(
                Sha256Digest::digest_bytes(b"runtime-evidence"),
            )],
            support: 1,
            functional_signature: vec![1.0],
            parent_skill_ids: vec![],
        }
    }

    fn single_field_causal_credit(skill_id: &SkillId) -> CausalCreditReport {
        CausalCreditReport {
            schema: "cerebro.tidex.causal_credit/v3".into(),
            context_count: 3,
            independent_group_count: 3,
            field_count: 1,
            fields: vec![crate::causal_credit::FieldCausalCredit {
                skill_id: skill_id.clone(),
                matched_pairs: 3,
                independent_contexts: 3,
                mean_marginal_effect: 1.0,
                standard_error: 0.05,
                lower_confidence_bound: 0.8,
                positive_fraction: 1.0,
                resolved: true,
                beneficial: true,
            }],
            pair_interactions: vec![],
            unresolved_fields: vec![],
        }
    }

    #[test]
    fn runtime_health_reports_fail_closed_state_matrix() {
        let root = isolated_engine_root("runtime-health-empty");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let health = engine.runtime_integrity_health().unwrap();
        assert!(!health.execution_authorized);
        assert!(!health.canonical_head_verified);
        assert!(!health.bank_verified);
        assert!(!health.observations_verified);
        assert!(!health.sleep_state_verified);
        assert!(health
            .execution_blockers
            .iter()
            .any(|v| v == "canonical_head_invalid"));
        assert!(health
            .execution_blockers
            .iter()
            .any(|v| v == "integrity_unhealthy"));
        assert!(matches!(
            engine.require_current_certification(),
            Err(BrainError::Integrity(message)) if message.contains("runtime_execution_not_authorized")
        ));
        assert!(matches!(
            engine.status(),
            Err(BrainError::Integrity(message)) if message == "active_skill_bank_missing"
        ));
        let _ = fs::remove_dir_all(root);

        let root = isolated_engine_root("runtime-health-noncanonical");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig {
                min_observations: 2,
                min_independent_apertures: 2,
                require_structured_geometry_for_promotion: false,
                require_dual_space_for_promotion: false,
                ..BrainConfig::default()
            },
        };
        let mut observations = vec![
            sample_observation("health-a", vec![1.0, 0.0]),
            sample_observation("health-b", vec![0.0, 1.0]),
        ];
        for (index, observation) in observations.iter_mut().enumerate() {
            observation.independence_group = format!("health-group-{index}");
            observation.experiment_lineage.run_id = format!("run-{index}");
            observation.experiment_lineage.replicate_id = format!("rep-{index}");
            observation.experiment_lineage.randomization_id = format!("rand-{index}");
            observation.experiment_lineage.dataset_split_digest = "1".repeat(64);
            observation.experiment_lineage.initial_checkpoint_digest = "2".repeat(64);
            observation.experiment_lineage.optimizer_config_digest = "3".repeat(64);
            observation.experiment_lineage.template_config_digest = "4".repeat(64);
        }
        engine.persist_observations(&observations).unwrap();
        let mut health_field = sample_field("health-skill");
        health_field.evidence_support_digests = vec![ObservationRecordDigest::from(
            Sha256Digest::digest_bytes(b"health-support"),
        )];
        health_field.support = 1;
        let bank = SkillBank {
            generation: 1,
            fields: vec![health_field],
        };
        write_or_verify_immutable(
            &root,
            &engine.bank_path(),
            &serialize_pretty_line(&bank).unwrap(),
        )
        .unwrap();
        let health = engine.runtime_integrity_health().unwrap();
        assert!(!health.canonical_runtime_config);
        assert!(health.bank_verified);
        assert!(!health.composition_ready);
        assert!(health
            .integrity_reasons
            .iter()
            .any(|v| v == "noncanonical_runtime_config_forbidden"));
        assert!(health
            .integrity_reasons
            .iter()
            .any(|v| v.contains("runtime_composition_unready")));
        let _ = fs::remove_dir_all(root);

        for (label, bytes, expected_reason) in [
            ("parse", b"{".as_slice(), "sleep_state_parse_failed"),
            ("schema", br#"{"schema":"wrong"}"#.as_slice(), "sleep_state_schema_invalid"),
            (
                "cert",
                br#"{"schema":"cerebro.tidex.sleep_state/v5","operation_key":"op","certification_status":"invalid","evidence_verification":{"verified":false}}"#.as_slice(),
                "certification_status_invalid",
            ),
            (
                "receipt",
                br#"{"schema":"cerebro.tidex.sleep_state/v5","operation_key":"missing-receipt","certification_status":"revoked","evidence_verification":{"verified":false}}"#.as_slice(),
                "sleep_receipt_missing",
            ),
        ] {
            let root = isolated_engine_root(&format!("runtime-health-{label}"));
            let engine = BrainEngine {
                root: root.clone(),
                config: BrainConfig::default(),
            };
            write_or_verify_immutable(&root, &root.join("state/sleep_state.json"), bytes).unwrap();
            let health = engine.runtime_integrity_health().unwrap();
            assert!(health
                .integrity_reasons
                .iter()
                .any(|value| value.contains(expected_reason)),
                "label={label} reasons={:?}",
                health.integrity_reasons
            );
            let _ = fs::remove_dir_all(root);
        }

        let root = isolated_engine_root("runtime-health-transition");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        ensure_private_directory(
            &root,
            &root.join("state/corpus_transitions/inflight/incomplete"),
        )
        .unwrap();
        let health = engine.runtime_integrity_health().unwrap();
        assert!(!health.corpus_transition_clear);
        assert!(health
            .execution_blockers
            .iter()
            .any(|v| v == "corpus_transition_incomplete"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_evidence_and_current_observation_boundaries_are_descriptor_verified() {
        let root = isolated_engine_root("runtime-evidence-boundary");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig {
                min_observations: 1,
                require_structured_geometry_for_promotion: false,
                require_dual_space_for_promotion: false,
                ..BrainConfig::default()
            },
        };
        let evidence_path = root.join("state/runtime-evidence.json");
        let evidence_bytes = br#"{"schema":"test"}"#;
        write_or_verify_immutable(&root, &evidence_path, evidence_bytes).unwrap();
        let digest = Sha256Digest::digest_bytes(evidence_bytes);
        assert_eq!(
            engine
                .canonical_evidence_bytes(evidence_path.to_str().unwrap(), digest.as_str())
                .unwrap(),
            evidence_bytes
        );
        assert!(matches!(
            engine.canonical_evidence_bytes(evidence_path.to_str().unwrap(), "invalid"),
            Err(BrainError::Invalid(message)) if message == "runtime_evidence_digest_invalid"
        ));
        assert!(matches!(
            engine.canonical_evidence_bytes(
                evidence_path.to_str().unwrap(),
                Sha256Digest::digest_bytes(b"wrong").as_str()
            ),
            Err(BrainError::Integrity(message)) if message.contains("runtime_evidence_invalid")
        ));

        let observation = sample_observation("runtime-current", vec![1.0, 0.0]);
        engine
            .persist_observations(std::slice::from_ref(&observation))
            .unwrap();
        let semantic = digest_json(&observation).unwrap();
        assert_eq!(
            engine
                .load_current_observation_by_semantic_sha256(&semantic)
                .unwrap(),
            observation
        );
        engine
            .require_current_observation_digest(&semantic)
            .unwrap();
        assert!(matches!(
            engine.load_current_observation_by_semantic_sha256("bad"),
            Err(BrainError::Invalid(message)) if message.contains("source_observation_digest_invalid")
        ));
        assert!(matches!(
            engine.load_current_observation_by_semantic_sha256(
                Sha256Digest::digest_bytes(b"absent").as_str()
            ),
            Err(BrainError::Integrity(message)) if message.contains("source_observation_not_current")
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_verified_composition_uses_real_dense_artifact_and_causal_contract() {
        let root = isolated_engine_root("runtime-compose-verified");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let layout = crate::block_tomography::ParameterBlockLayout::from_shapes(&[
            crate::block_tomography::BlockShapeSpec {
                name: "runtime-weights".into(),
                shape: vec![4],
                count: 4,
            },
        ])
        .unwrap();
        let layout_bytes = serde_json::to_vec(&layout).unwrap();
        let layout_sha = Sha256Digest::digest_bytes(&layout_bytes);
        write_or_verify_immutable(
            &root,
            &root
                .join("state/parameter_layouts/by-sha")
                .join(format!("{layout_sha}.json")),
            &layout_bytes,
        )
        .unwrap();
        let field = runtime_field_with_artifact(
            &root,
            &layout_sha,
            "runtime-skill",
            &[1.0, -0.5, 0.25, 0.75],
        );
        let bank = SkillBank {
            generation: 1,
            fields: vec![field.clone()],
        };
        engine.verify_runtime_composition_bank(&bank).unwrap();
        assert!(matches!(
            engine.verify_runtime_composition_bank(&SkillBank::default()),
            Err(BrainError::Integrity(message)) if message == "runtime_composition_bank_empty"
        ));
        let no_layout = SkillBank {
            generation: 1,
            fields: vec![sample_field("runtime-no-layout")],
        };
        assert!(matches!(
            engine.verify_runtime_composition_bank(&no_layout),
            Err(BrainError::Integrity(message)) if message.contains("runtime_composition_layout_missing")
        ));

        let causal = single_field_causal_credit(&field.skill_id);
        let protected = ProtectedCortex {
            parameter_importance: vec![0.0; 4],
            directions: vec![],
            max_damage_ratio: 1.0,
        };
        let activation = BTreeMap::from([(field.skill_id.clone(), 0.5)]);
        let composed = engine
            .compose_verified_inputs(
                &bank,
                &activation,
                &protected,
                &Matrix::identity(1),
                1.0,
                &causal,
            )
            .unwrap();
        assert_eq!(composed.delta.len(), 4);
        assert!(composed
            .delta
            .iter()
            .any(|value| value.abs() > f64::EPSILON));

        assert!(matches!(
            engine.compose_verified_inputs(
                &bank,
                &BTreeMap::new(),
                &protected,
                &Matrix::identity(1),
                1.0,
                &causal,
            ),
            Err(BrainError::Invalid(message)) if message == "skill_activation_empty"
        ));
        let nonfinite = BTreeMap::from([(field.skill_id.clone(), f64::NAN)]);
        assert!(matches!(
            engine.compose_verified_inputs(
                &bank,
                &nonfinite,
                &protected,
                &Matrix::identity(1),
                1.0,
                &causal,
            ),
            Err(BrainError::Invalid(message)) if message == "activation_non_finite"
        ));
        let unknown = BTreeMap::from([(SkillId::parse("unknown-skill").unwrap(), 1.0)]);
        assert!(matches!(
            engine.compose_verified_inputs(
                &bank,
                &unknown,
                &protected,
                &Matrix::identity(1),
                1.0,
                &causal,
            ),
            Err(BrainError::Invalid(message)) if message.contains("skill_not_found")
        ));
        let wrong_protected = ProtectedCortex {
            parameter_importance: vec![0.0; 3],
            directions: vec![],
            max_damage_ratio: 1.0,
        };
        assert!(matches!(
            engine.compose_verified_inputs(
                &bank,
                &activation,
                &wrong_protected,
                &Matrix::identity(1),
                1.0,
                &causal,
            ),
            Err(BrainError::Invalid(message)) if message == "protected_cortex_parameter_space_mismatch"
        ));
        let zero_activation = BTreeMap::from([(field.skill_id.clone(), 0.0)]);
        assert!(matches!(
            engine.compose_verified_inputs(
                &bank,
                &zero_activation,
                &protected,
                &Matrix::identity(1),
                1.0,
                &causal,
            ),
            Err(BrainError::Invalid(message)) if message == "runtime_composed_delta_zero"
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn support_private_io_pointer_receipt_and_archive_contracts_are_transactional() {
        let root = isolated_engine_root("support-authority");
        let json_path = root.join("state/sample.json");
        let json_bytes = serialize_pretty_line(&json!({"value":7})).unwrap();
        write_new_private(&root, &json_path, &json_bytes).unwrap();
        let loaded: Value = read_private_json(&root, &json_path, 1024).unwrap();
        assert_eq!(loaded["value"], 7);
        let digest = sha256_bytes(&json_bytes);
        assert!(private_file_digest_matches(&root, &json_path, &digest));
        assert!(!private_file_digest_matches(&root, &json_path, "invalid"));
        assert!(!private_file_digest_matches(
            &root,
            &json_path,
            Sha256Digest::digest_bytes(b"different").as_str()
        ));
        assert!(write_immutable_exact(&root, &root.join("state/bad.json"), b"x", &digest).is_err());
        write_immutable_exact(
            &root,
            &root.join("state/exact.json"),
            b"x",
            Sha256Digest::digest_bytes(b"x").as_str(),
        )
        .unwrap();

        let pointer = root.join("state/sleep_state.json");
        let first = b"first";
        replace_private_pointer_exact(
            &root,
            &pointer,
            first,
            Sha256Digest::digest_bytes(first).as_str(),
        )
        .unwrap();
        let second = b"second";
        replace_private_pointer_exact(
            &root,
            &pointer,
            second,
            Sha256Digest::digest_bytes(second).as_str(),
        )
        .unwrap();
        assert_eq!(fs::read(&pointer).unwrap(), second);
        assert!(matches!(
            replace_private_pointer_exact(
                &root,
                &root.join("state/not-allowlisted.json"),
                b"x",
                Sha256Digest::digest_bytes(b"x").as_str()
            ),
            Err(BrainError::Integrity(message)) if message == "transaction_pointer_target_not_allowlisted"
        ));

        assert_eq!(
            persistent_not_applicable_reason(&BrainError::Numerical(
                "persistent_field_geometry_empty".into()
            )),
            Some("persistent_field_geometry_empty".into())
        );
        assert!(persistent_not_applicable_reason(&BrainError::Invalid("x".into())).is_none());
        assert!(persistent_not_applicable_reason(&BrainError::Numerical("other".into())).is_none());

        let invocation = ControllerInvocation {
            schema: "cerebro.tidex.controller_invocation/v1".into(),
            session_id: SessionId::parse("support-session").unwrap(),
            state_before: vec![1.0, 0.0],
            promoted_observation_semantic_sha256: Sha256Digest::digest_bytes(b"semantic"),
        };
        invocation.validate().unwrap();
        let receipt = ControllerExecutionReceipt {
            schema: "cerebro.tidex.controller_execution_receipt/v1".into(),
            session_id: invocation.session_id.clone(),
            invocation_sha256: Sha256Digest::parse(digest_json(&invocation).unwrap()).unwrap(),
            controller_receipt_sha256: Sha256Digest::digest_bytes(b"controller"),
            state_before_sha256: Sha256Digest::parse(
                digest_json(&invocation.state_before).unwrap(),
            )
            .unwrap(),
            promoted_observation_semantic_sha256: invocation
                .promoted_observation_semantic_sha256
                .clone(),
            governed_composition_receipt_sha256: Sha256Digest::digest_bytes(b"governed"),
            invocation,
        };
        let first_recorded = persist_controller_execution(&root, receipt.clone()).unwrap();
        let second_recorded = persist_controller_execution(&root, receipt.clone()).unwrap();
        assert_eq!(first_recorded, second_recorded);
        assert!(valid_digest(&first_recorded.receipt_sha256));
        assert!(valid_digest(&first_recorded.ledger_event_hash));
        assert_eq!(
            controller_execution_ledger_binding(&root, &first_recorded.receipt_sha256, &receipt)
                .unwrap(),
            first_recorded.ledger_event_hash
        );
        let mut wrong_receipt = receipt.clone();
        wrong_receipt.session_id = SessionId::parse("wrong-session").unwrap();
        assert!(matches!(
            controller_execution_ledger_binding(
                &root,
                &first_recorded.receipt_sha256,
                &wrong_receipt
            ),
            Err(BrainError::Integrity(message)) if message == "controller_execution_receipt_ledger_payload_mismatch"
        ));
        let mut wrong_recorded = first_recorded.clone();
        wrong_recorded.receipt_path = root.join("wrong.json").to_string_lossy().into_owned();
        assert!(matches!(
            verify_controller_execution_receipt(&root, &wrong_recorded),
            Err(BrainError::Integrity(message)) if message == "controller_execution_receipt_contract_invalid"
        ));

        let a = Sha256Digest::digest_bytes(b"a");
        let b = Sha256Digest::digest_bytes(b"b");
        let c = Sha256Digest::digest_bytes(b"c");
        assert_eq!(
            learning_finalization_operation_key(&a, &b, &c).unwrap(),
            learning_finalization_operation_key(&a, &b, &c).unwrap()
        );
        assert_ne!(
            learning_finalization_operation_key(&a, &b, &c).unwrap(),
            learning_finalization_operation_key(&a, &c, &b).unwrap()
        );
        assert!(archive_label_is_allowed("skill_bank.json"));
        assert!(!archive_label_is_allowed("arbitrary.json"));
        assert_eq!(
            learning_finalization_receipt_path(&root, &a),
            root.join("state/learning_finalizations")
                .join(format!("{a}.json"))
        );
        assert_eq!(
            learning_finalization_archive_dir(&root, &a),
            root.join("state/corpus_transitions/by-operation")
                .join(a.as_str())
        );

        let archive = root.join("state/archive-fixture");
        ensure_private_directory(&root, &archive).unwrap();
        let source = root.join("state/shadow_skill_bank.json");
        let source_bytes = b"archive-me";
        write_new_private(&root, &source, source_bytes).unwrap();
        let source_sha = Sha256Digest::digest_bytes(source_bytes);
        let mut archived = BTreeMap::new();
        archive_private_state_file(
            &root,
            &source,
            &archive.join("shadow_skill_bank.json"),
            "shadow_skill_bank.json",
            Some(&source_sha),
            &mut archived,
        )
        .unwrap();
        assert!(!source.exists());
        assert_eq!(archived.get("shadow_skill_bank.json"), Some(&source_sha));
        assert!(archive.join("shadow_skill_bank.json").exists());
        assert!(archive_private_state_file(
            &root,
            &root.join("state/exact.json"),
            &archive.join("arbitrary.json"),
            "arbitrary.json",
            None,
            &mut archived,
        )
        .is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn recover_no_incomplete_transition() {
        let root = isolated_engine_root("recover-none");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let recovery = engine.recover_incomplete_corpus_transition().unwrap();
        assert_eq!(
            recovery.outcome,
            CorpusTransitionRecoveryOutcome::NoIncompleteTransition
        );
        assert!(recovery.operation_key.is_none());
        assert!(recovery.phase.is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recover_restores_archived_prior_corpus() {
        let root = isolated_engine_root("recover-archive");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let operation_key = Sha256Digest::digest_bytes(b"recover-archive-op");
        let inflight = root
            .join("state/corpus_transitions/inflight")
            .join(operation_key.as_str());
        fs::create_dir_all(&inflight).unwrap();

        let intent = LearningCorpusTransitionIntent {
            schema: "cerebro.tidex.learning_corpus_transition_intent/v1".into(),
            operation_key: operation_key.clone(),
            session_id: SessionId::parse("session-archive-rec").unwrap(),
            adaptive_receipt_sha256: Sha256Digest::digest_bytes(b"adaptive"),
            learning_finalization_input_sha256: Sha256Digest::digest_bytes(b"input"),
            representation_evidence_receipt: PrivateFileReference::new(
                root.join("state/representation.json"),
                Sha256Digest::digest_bytes(b"evidence"),
            ),
            representation_protocol_sha256: Sha256Digest::digest_bytes(b"protocol"),
            representation_observation_bindings_sha256: Sha256Digest::digest_bytes(b"bindings"),
            prior_corpus_digest: Sha256Digest::digest_bytes(b"prior-archived"),
            prior_observation_count: 2,
            new_corpus_digest: Sha256Digest::digest_bytes(b"new"),
            new_observation_count: 4,
            archive_dir: root
                .join("state/corpus_transitions/by-operation")
                .join(operation_key.as_str()),
        };
        fs::write(
            inflight.join("intent.json"),
            serialize_pretty_line(&intent).unwrap(),
        )
        .unwrap();

        let valid_obs = sample_observation("obs-1", vec![1.0, 0.0]);
        engine.persist_observations(&[valid_obs]).unwrap();
        // Move live observations to archive
        let archive_dir = root
            .join("state/corpus_transitions/by-operation")
            .join(operation_key.as_str());
        fs::create_dir_all(&archive_dir).unwrap();
        let archive_obs = archive_dir.join("observations");
        fs::rename(root.join("state/observations"), &archive_obs).unwrap();

        let recovery = engine.recover_incomplete_corpus_transition().unwrap();
        assert_eq!(
            recovery.outcome,
            CorpusTransitionRecoveryOutcome::RestoredPriorCorpus
        );
        assert_eq!(recovery.operation_key.as_ref(), Some(&operation_key));
        // Verify live observations directory was restored
        assert!(root.join("state/observations").exists());
        assert!(!archive_obs.exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recover_requires_replay_when_new_corpus_already_published() {
        let root = isolated_engine_root("recover-replay");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let observations = vec![
            sample_observation("obs-a", vec![1.0, 0.0]),
            sample_observation("obs-b", vec![0.0, 1.0]),
        ];
        engine.persist_observations(&observations).unwrap();
        let live_digest = observation_set_digest(&observations).unwrap();

        let operation_key = Sha256Digest::digest_bytes(b"recover-replay-op");
        let inflight = root
            .join("state/corpus_transitions/inflight")
            .join(operation_key.as_str());
        fs::create_dir_all(&inflight).unwrap();

        let intent = LearningCorpusTransitionIntent {
            schema: "cerebro.tidex.learning_corpus_transition_intent/v1".into(),
            operation_key: operation_key.clone(),
            session_id: SessionId::parse("session-replay-rec").unwrap(),
            adaptive_receipt_sha256: Sha256Digest::digest_bytes(b"adaptive"),
            learning_finalization_input_sha256: Sha256Digest::digest_bytes(b"input"),
            representation_evidence_receipt: PrivateFileReference::new(
                root.join("state/representation.json"),
                Sha256Digest::digest_bytes(b"evidence"),
            ),
            representation_protocol_sha256: Sha256Digest::digest_bytes(b"protocol"),
            representation_observation_bindings_sha256: Sha256Digest::digest_bytes(b"bindings"),
            prior_corpus_digest: Sha256Digest::digest_bytes(b"prior-dummy"),
            prior_observation_count: 1,
            new_corpus_digest: Sha256Digest::parse(live_digest.as_str()).unwrap(),
            new_observation_count: 2,
            archive_dir: root
                .join("state/corpus_transitions/by-operation")
                .join(operation_key.as_str()),
        };
        fs::write(
            inflight.join("intent.json"),
            serialize_pretty_line(&intent).unwrap(),
        )
        .unwrap();

        let recovery = engine.recover_incomplete_corpus_transition().unwrap();
        assert_eq!(
            recovery.outcome,
            CorpusTransitionRecoveryOutcome::RequiresOriginalFinalizationReplay
        );
        assert_eq!(recovery.operation_key.as_ref(), Some(&operation_key));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sleep_cycle_and_commit_finalization_fail_closed_on_guards() {
        let root = isolated_engine_root("sleep-commit-guards");
        let non_canonical_config = BrainConfig {
            require_structured_geometry_for_promotion: false,
            ..BrainConfig::default()
        };
        let non_canonical_engine = BrainEngine {
            root: root.clone(),
            config: non_canonical_config,
        };

        // 1. Non-canonical config fails closed on both entrypoints
        let err = non_canonical_engine.sleep_cycle().unwrap_err();
        assert!(
            matches!(err, BrainError::Integrity(ref m) if m.contains("noncanonical_runtime_config_forbidden"))
        );

        let dummy_input = LearningFinalizationInput {
            schema: "cerebro.tidex.learning_finalization_input/v1".into(),
            session_id: SessionId::parse("session-guard").unwrap(),
            adaptive_receipt_sha256: Sha256Digest::zero(),
            target_id: crate::identity::LearningTargetId::parse("t-1").unwrap(),
            target_digest: Sha256Digest::zero(),
            policy_digest: Sha256Digest::zero(),
            completed_evidence_sha256: vec![],
            representation_evidence_receipt: PrivateFileReference::new(
                root.join("evidence.json"),
                Sha256Digest::zero(),
            ),
            representation_protocol_sha256: Sha256Digest::zero(),
            representation_installations_sha256: Sha256Digest::zero(),
            representation_observation_bindings: vec![],
            observations: vec![],
        };

        let err = non_canonical_engine
            .commit_finalized_learning_session(&dummy_input)
            .unwrap_err();
        assert!(
            matches!(err, BrainError::Integrity(ref m) if m.contains("noncanonical_runtime_config_forbidden"))
        );

        let canonical_engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 2. Canonical engine with 0 observations fails sleep_cycle
        let err = canonical_engine.sleep_cycle().unwrap_err();
        assert!(
            matches!(err, BrainError::Invalid(ref m) if m.contains("sleep_requires_persisted_observations"))
        );

        // 3. Inflight transition prevents sleep_cycle
        let inflight = root
            .join("state/corpus_transitions/inflight")
            .join("dummy-key");
        fs::create_dir_all(&inflight).unwrap();
        let err = canonical_engine.sleep_cycle().unwrap_err();
        assert!(
            matches!(err, BrainError::Integrity(ref m) if m.contains("learning_corpus_transition_incomplete"))
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn commit_finalized_session_reconstruction_validation() {
        let root = isolated_engine_root("commit-fin-val");
        let (session_id, rep_receipt, _) =
            crate::learning_finalization::tests::make_test_scenario(&root, "session-comm-val");
        let input = crate::learning_finalization::prepare_learning_finalization(
            &root,
            session_id.as_str(),
            &rep_receipt,
        )
        .unwrap();

        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let err = engine
            .commit_finalized_learning_session(&input)
            .unwrap_err();
        assert!(
            matches!(err, BrainError::Invalid(ref m) if m.contains("minimum_six_observations"))
        );
        let _ = fs::remove_dir_all(&root);

        // 2. Scenario with 6 observations: passes minimum observation count, but fails prospective promotion
        let root6 = isolated_engine_root("commit-fin-val-6");
        let (session_id6, rep_receipt6, _) =
            crate::learning_finalization::tests::make_test_scenario_with_count(
                &root6,
                "session-comm-val6",
                6,
            );
        let input6 = crate::learning_finalization::prepare_learning_finalization(
            &root6,
            session_id6.as_str(),
            &rep_receipt6,
        )
        .unwrap();
        assert_eq!(input6.observations.len(), 6);

        let engine6 = BrainEngine {
            root: root6.clone(),
            config: BrainConfig::default(),
        };
        let err6 = engine6
            .commit_finalized_learning_session(&input6)
            .unwrap_err();
        assert!(
            matches!(err6, BrainError::Invalid(ref m) if m == "dual_space_field_model_invalid"),
            "err6={err6:?}"
        );

        let _ = fs::remove_dir_all(&root6);
    }

    #[test]
    fn validate_brain_config_exhaustive_bounds() {
        assert!(validate_brain_config(&BrainConfig::default()).is_ok());

        let base = BrainConfig::default();

        let mut c = base.clone();
        c.min_independent_apertures = 1;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_independent_apertures = MAX_ENGINE_INDEPENDENCE_GROUPS + 1;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_observations = 3;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_observations = MAX_ENGINE_OBSERVATIONS + 1;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.max_rank = 0;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.max_rank = MAX_ENGINE_SKILL_FIELDS + 1;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.target_explained_variance = -0.1;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.target_explained_variance = 1.1;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.ridge = 0.0;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.huber_delta = 0.0;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.irls_rounds = 0;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.irls_rounds = MAX_ENGINE_IRLS_ROUNDS + 1;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.skill_match_cosine = -0.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.skill_match_cosine = 1.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_functional_cv_r2 = 1.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.max_cycle_rms = -0.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_skill_coherence = -0.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_skill_coherence = 1.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_skill_persistence = -0.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_skill_persistence = 1.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_field_explained_variance = -0.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_field_explained_variance = 1.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.max_condition_estimate = 0.99;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.max_spectral_normalized_reconstruction_rms = -0.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_identifiability_signal_to_noise = 0.0;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_representation_match_accuracy = -0.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_representation_match_accuracy = 1.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_representation_match_margin = -0.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));

        let mut c = base.clone();
        c.min_representation_cv_r2 = 1.01;
        assert!(matches!(validate_brain_config(&c), Err(BrainError::Invalid(m)) if m == "brain_config_invalid"));
    }

    #[test]
    fn support_load_observations_and_canonical_digests_invariants() {
        let root = isolated_engine_root("obs-loader-invariants");
        let dir = root.join("state/custom_obs");

        // 1. Missing directory with missing_is_empty = true returns empty
        assert_eq!(
            load_observations_from_private_directory(&root, &dir, true).unwrap(),
            vec![]
        );

        // 2. Missing directory with missing_is_empty = false fails
        assert!(load_observations_from_private_directory(&root, &dir, false).is_err());

        ensure_private_directory(&root, &dir).unwrap();

        // 3. Non-json entry triggers persisted_observations_entry_invalid
        let txt_file = dir.join("invalid.txt");
        write_new_private(&root, &txt_file, b"test").unwrap();
        assert!(matches!(
            load_observations_from_private_directory(&root, &dir, true),
            Err(BrainError::Integrity(ref m)) if m == "persisted_observations_entry_invalid"
        ));
        let _ = fs::remove_file(&txt_file);

        // 4. Bad filename (doesn't match id-digest[..16].json)
        let obs_a = sample_observation("obs-1", vec![1.0, 0.0]);
        let raw_a = serde_json::to_vec_pretty(&obs_a).unwrap();
        let bad_name = dir.join("obs-1.json");
        write_new_private(&root, &bad_name, &raw_a).unwrap();
        assert!(matches!(
            load_observations_from_private_directory(&root, &dir, true),
            Err(BrainError::Integrity(ref m)) if m == "persisted_observation_identity_invalid"
        ));
        let _ = fs::remove_file(&bad_name);

        // 5. Correct canonical name
        let digest_a = digest_json(&obs_a).unwrap();
        let good_name = dir.join(format!("{}-{}.json", obs_a.observation_id, &digest_a[..16]));
        write_new_private(&root, &good_name, &raw_a).unwrap();
        let loaded = load_observations_from_private_directory(&root, &dir, true).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].observation_id, obs_a.observation_id);

        // 6. canonical_observation_digests with unique items succeeds
        let obs_b = sample_observation("obs-2", vec![0.0, 1.0]);
        let digests = canonical_observation_digests(&[obs_a.clone(), obs_b.clone()]).unwrap();
        assert_eq!(digests.len(), 2);

        // 7. Duplicate observation triggers learning_finalization_observation_digest_duplicate
        assert!(matches!(
            canonical_observation_digests(&[obs_a.clone(), obs_a.clone()]),
            Err(BrainError::Integrity(ref m)) if m == "learning_finalization_observation_digest_duplicate"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn support_governed_composition_key_contract() {
        let sid = SkillId::parse("skill-1").unwrap();
        let mut activation = BTreeMap::new();
        activation.insert(sid, 0.5);

        let layout_digest = ParameterLayoutDigest::from(Sha256Digest::digest_bytes(b"layout"));
        let layout_art_digest = Sha256Digest::digest_bytes(b"art");

        let op1 = GovernedCompositionOperation {
            report_sha256: "report-1",
            active_bank_sha256: "bank-1",
            evidence_bundle_sha256: "ev-1",
            causal_credit_sha256: "cc-1",
            parameter_layout_artifact_sha256: &layout_art_digest,
            parameter_layout_sha256: &layout_digest,
            activation: &activation,
            projected_delta_sha256: "delta-1",
            source_observation_sha256: "src-1",
        };
        let key1 = governed_composition_operation_key(&op1).unwrap();
        let key2 = governed_composition_operation_key(&op1).unwrap();
        assert_eq!(key1, key2);

        let mut op2 = op1;
        op2.report_sha256 = "report-2";
        let key3 = governed_composition_operation_key(&op2).unwrap();
        assert_ne!(key1, key3);
    }

    #[test]
    fn runtime_cognitive_route_activation_contracts() {
        let root = isolated_engine_root("route-act-contracts");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };
        let s1 = SkillId::parse("skill-1").unwrap();
        let s2 = SkillId::parse("skill-2").unwrap();

        // 1. Invalid schema
        let mut route = crate::cognitive_field::FieldRoutingDecision {
            schema: "wrong-schema".into(),
            field_ids: vec![s1.clone()],
            coefficients: vec![0.5],
            selected_field_ids: vec![s1.clone()],
            selected_activation_mass: 0.5,
        };
        assert!(matches!(
            engine.cognitive_route_activation(&route),
            Err(BrainError::Invalid(m)) if m == "cognitive_route_contract_invalid"
        ));

        // 2. Empty field_ids
        route.schema = "cerebro.tidex.cognitive_field_routing/v1".into();
        route.field_ids = vec![];
        route.coefficients = vec![];
        route.selected_field_ids = vec![];
        assert!(matches!(
            engine.cognitive_route_activation(&route),
            Err(BrainError::Invalid(m)) if m == "cognitive_route_contract_invalid"
        ));

        // 3. Field IDs / coefficients length mismatch
        route.field_ids = vec![s1.clone(), s2.clone()];
        route.coefficients = vec![0.5];
        route.selected_field_ids = vec![s1.clone()];
        assert!(matches!(
            engine.cognitive_route_activation(&route),
            Err(BrainError::Invalid(m)) if m == "cognitive_route_contract_invalid"
        ));

        // 4. Negative coefficient
        route.coefficients = vec![0.5, -0.1];
        assert!(matches!(
            engine.cognitive_route_activation(&route),
            Err(BrainError::Invalid(m)) if m == "cognitive_route_contract_invalid"
        ));

        // 5. Duplicate field IDs
        route.field_ids = vec![s1.clone(), s1.clone()];
        route.coefficients = vec![0.5, 0.5];
        route.selected_field_ids = vec![s1.clone()];
        assert!(matches!(
            engine.cognitive_route_activation(&route),
            Err(BrainError::Invalid(m)) if m == "cognitive_route_identity_invalid"
        ));

        // 6. Selected field ID not in field_ids
        let s3 = SkillId::parse("skill-3").unwrap();
        route.field_ids = vec![s1.clone(), s2.clone()];
        route.coefficients = vec![0.5, 0.5];
        route.selected_field_ids = vec![s3];
        assert!(matches!(
            engine.cognitive_route_activation(&route),
            Err(BrainError::Invalid(m)) if m == "cognitive_route_identity_invalid"
        ));

        // 7. Selection mismatch with activated coefficients
        route.field_ids = vec![s1.clone(), s2.clone()];
        route.coefficients = vec![0.5, 0.0];
        route.selected_field_ids = vec![s1.clone(), s2.clone()];
        assert!(matches!(
            engine.cognitive_route_activation(&route),
            Err(BrainError::Invalid(m)) if m == "cognitive_route_selection_mismatch"
        ));

        // 8. Happy path: valid route
        route.coefficients = vec![0.5, 0.8];
        route.selected_field_ids = vec![s1.clone(), s2.clone()];
        route.selected_activation_mass = 1.3;
        let act = engine.cognitive_route_activation(&route).unwrap();
        assert_eq!(act.len(), 2);
        assert_eq!(act.get(&s1), Some(&0.5));
        assert_eq!(act.get(&s2), Some(&0.8));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_current_observation_lookup_invariants() {
        let root = isolated_engine_root("cur-obs-invariants");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 1. Invalid digest format
        assert!(matches!(
            engine.load_current_observation_by_semantic_sha256("bad-digest"),
            Err(BrainError::Invalid(m)) if m == "governed_composition_source_observation_digest_invalid"
        ));

        // 2. Valid digest format but observation not present
        let non_existent = "a".repeat(64);
        assert!(matches!(
            engine.load_current_observation_by_semantic_sha256(&non_existent),
            Err(BrainError::Integrity(m)) if m == "governed_composition_source_observation_not_current"
        ));
        assert!(matches!(
            engine.require_current_observation_digest(&non_existent),
            Err(BrainError::Integrity(m)) if m == "governed_composition_source_observation_not_current"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_composition_bank_exhaustive_verification() {
        let root = isolated_engine_root("comp-bank-exhaust");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 1. Multiple parameter layouts in bank
        let mut f1 = sample_field("skill-1");
        f1.parameter_layout_sha256 = Some(Sha256Digest::digest_bytes(b"layout-1"));
        let mut f2 = sample_field("skill-2");
        f2.parameter_layout_sha256 = Some(Sha256Digest::digest_bytes(b"layout-2"));
        let bank_multi_layout = SkillBank {
            generation: 1,
            fields: vec![f1, f2],
        };
        assert!(matches!(
            engine.verify_runtime_composition_bank(&bank_multi_layout),
            Err(BrainError::Integrity(m)) if m == "runtime_composition_layout_identity_invalid"
        ));

        // 2. Structured geometry missing
        let layout = crate::block_tomography::ParameterBlockLayout::from_shapes(&[
            crate::block_tomography::BlockShapeSpec {
                name: "b1".into(),
                shape: vec![2],
                count: 2,
            },
        ]).unwrap();
        let mut layout_bytes = serde_json::to_vec_pretty(&layout).unwrap();
        layout_bytes.push(b'\n');
        let layout_sha = Sha256Digest::digest_bytes(&layout_bytes);
        let layout_dir = root.join("state/parameter_layouts/by-sha");
        ensure_private_directory(&root, &layout_dir).unwrap();
        let layout_path = layout_dir.join(format!("{layout_sha}.json"));
        write_new_private(&root, &layout_path, &layout_bytes).unwrap();

        let mut f3 = sample_field("skill-3");
        f3.parameter_layout_sha256 = Some(layout_sha.clone());
        f3.structured_geometry = None;
        let bank_no_geom = SkillBank {
            generation: 1,
            fields: vec![f3],
        };
        assert!(matches!(
            engine.verify_runtime_composition_bank(&bank_no_geom),
            Err(BrainError::Integrity(m)) if m.contains("runtime_composition_geometry_missing")
        ));

        // 3. Dense materialization missing
        let geom = crate::contracts::SkillSubspaceGeometry {
            skill_id: SkillId::parse("skill-4").unwrap(),
            source_support_indices: vec![0],
            blocks: vec![],
            max_local_rank: 1,
            mean_effective_rank: 1.0,
        };
        let mut f4 = sample_field("skill-4");
        f4.parameter_layout_sha256 = Some(layout_sha.clone());
        f4.structured_geometry = Some(geom.clone());
        f4.dense_materialization = None;
        let bank_no_dense = SkillBank {
            generation: 1,
            fields: vec![f4],
        };
        assert!(matches!(
            engine.verify_runtime_composition_bank(&bank_no_dense),
            Err(BrainError::Integrity(m)) if m.contains("runtime_composition_dense_missing")
        ));

        // 4. Dense count mismatch
        let writer = crate::artifact::ArtifactWriteAuthority::for_internal_root(&root).unwrap();
        let dense_wrong_count = writer.create_content_addressed_dvec(&[1.0, 2.0, 3.0]).unwrap();
        let mut f5 = sample_field("skill-5");
        f5.parameter_layout_sha256 = Some(layout_sha.clone());
        f5.structured_geometry = Some(geom.clone());
        f5.dense_materialization = Some(dense_wrong_count);
        let bank_count_mismatch = SkillBank {
            generation: 1,
            fields: vec![f5],
        };
        assert!(matches!(
            engine.verify_runtime_composition_bank(&bank_count_mismatch),
            Err(BrainError::Integrity(m)) if m.contains("runtime_composition_dense_count_mismatch")
        ));

        // 5. Valid dense matching layout count = 2
        let dense_valid = writer.create_content_addressed_dvec(&[1.0, 2.0]).unwrap();
        let mut f6 = sample_field("skill-6");
        f6.parameter_layout_sha256 = Some(layout_sha);
        f6.structured_geometry = Some(geom);
        f6.dense_materialization = Some(dense_valid);
        let bank_valid = SkillBank {
            generation: 1,
            fields: vec![f6],
        };
        assert!(engine.verify_runtime_composition_bank(&bank_valid).is_ok());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn transition_incomplete_operation_key_invariants() {
        let root = isolated_engine_root("trans-incompl-key");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 1. Inflight directory does not exist -> Ok(None)
        assert_eq!(engine.incomplete_transition_operation_key().unwrap(), None);

        // 2. Inflight directory exists but empty -> Ok(None)
        let inflight = root.join("state/corpus_transitions/inflight");
        ensure_private_directory(&root, &inflight).unwrap();
        assert_eq!(engine.incomplete_transition_operation_key().unwrap(), None);

        // 3. Single valid inflight operation key
        let key_1 = Sha256Digest::digest_bytes(b"op-1");
        let op_dir_1 = inflight.join(key_1.as_str());
        ensure_private_directory(&root, &op_dir_1).unwrap();
        assert_eq!(
            engine.incomplete_transition_operation_key().unwrap(),
            Some(key_1)
        );

        // 4. Multiple inflight operations -> ambiguous error
        let key_2 = Sha256Digest::digest_bytes(b"op-2");
        let op_dir_2 = inflight.join(key_2.as_str());
        ensure_private_directory(&root, &op_dir_2).unwrap();
        assert!(matches!(
            engine.incomplete_transition_operation_key(),
            Err(BrainError::Integrity(m)) if m == "learning_finalization_inflight_transition_ambiguous"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn transition_verify_sleep_evidence_pointer_invariants() {
        let root = isolated_engine_root("sleep-ev-ptr-invariants");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 1. None expected, no file present -> Ok(None)
        assert_eq!(
            engine.verify_current_sleep_evidence_pointer(None).unwrap(),
            None
        );

        // 2. Expected some SHA, but file is missing -> Err(sleep_evidence_current_pointer_missing)
        let sha = Sha256Digest::digest_bytes(b"content");
        assert!(matches!(
            engine.verify_current_sleep_evidence_pointer(Some(sha.as_str())),
            Err(BrainError::Integrity(m)) if m == "sleep_evidence_current_pointer_missing"
        ));

        // 3. File exists, but None was expected -> Err(sleep_installation_unexpected_current_evidence)
        let ev_dir = root.join("state/sleep_evidence");
        ensure_private_directory(&root, &ev_dir).unwrap();
        let ev_path = ev_dir.join("current.json");
        write_new_private(&root, &ev_path, b"content").unwrap();
        assert!(matches!(
            engine.verify_current_sleep_evidence_pointer(None),
            Err(BrainError::Integrity(m)) if m == "sleep_installation_unexpected_current_evidence"
        ));

        // 4. File exists and matches expected SHA -> Ok(Some(bytes))
        assert_eq!(
            engine
                .verify_current_sleep_evidence_pointer(Some(sha.as_str()))
                .unwrap(),
            Some(b"content".to_vec())
        );

        // 5. File exists, but expected SHA does not match -> Err(sleep_evidence_changed_during_transaction)
        let wrong_sha = Sha256Digest::digest_bytes(b"different");
        assert!(matches!(
            engine.verify_current_sleep_evidence_pointer(Some(wrong_sha.as_str())),
            Err(BrainError::Integrity(m)) if m == "sleep_evidence_changed_during_transaction"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn transition_authorize_empty_bank_bootstrap_invariants() {
        let root = isolated_engine_root("empty-bank-invariants");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 1. Without prior sleep state -> always succeeds (genesis)
        assert!(engine
            .authorize_empty_bank_bootstrap(false, &serde_json::json!({}))
            .is_ok());

        // 2. With prior sleep state, but wrong schema
        assert!(matches!(
            engine.authorize_empty_bank_bootstrap(
                true,
                &serde_json::json!({"schema": "wrong"})
            ),
            Err(BrainError::Integrity(m)) if m == "empty_bank_bootstrap_prior_state_not_explicitly_revoked"
        ));

        // 3. With prior sleep state, but status is Certified instead of Revoked
        assert!(matches!(
            engine.authorize_empty_bank_bootstrap(
                true,
                &serde_json::json!({
                    "schema": "cerebro.tidex.sleep_state/v5",
                    "certification_status": "certified"
                })
            ),
            Err(BrainError::Integrity(m)) if m == "empty_bank_bootstrap_prior_state_not_explicitly_revoked"
        ));

        // 4. With prior sleep state, Revoked, but active_skill_count is not 0
        assert!(matches!(
            engine.authorize_empty_bank_bootstrap(
                true,
                &serde_json::json!({
                    "schema": "cerebro.tidex.sleep_state/v5",
                    "certification_status": "revoked",
                    "active_skill_count": 1,
                    "promoted": false,
                    "active_bank_sha256": null
                })
            ),
            Err(BrainError::Integrity(m)) if m == "empty_bank_bootstrap_prior_state_not_explicitly_revoked"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_search_by_function_invariants() {
        let root = isolated_engine_root("search-by-function");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 1. Empty query fails closed
        assert!(matches!(
            engine.search_by_function(&[], 5),
            Err(BrainError::Invalid(m)) if m == "functional_search_request_invalid"
        ));

        // 2. Non-finite query fails closed
        assert!(matches!(
            engine.search_by_function(&[f64::NAN, 1.0], 5),
            Err(BrainError::Invalid(m)) if m == "functional_search_request_invalid"
        ));
        assert!(matches!(
            engine.search_by_function(&[f64::INFINITY, 1.0], 5),
            Err(BrainError::Invalid(m)) if m == "functional_search_request_invalid"
        ));

        // 3. Excessive dimension fails closed
        let huge_query = vec![1.0; MAX_ENGINE_FUNCTIONAL_RESPONSE_DIMENSION + 1];
        assert!(matches!(
            engine.search_by_function(&huge_query, 5),
            Err(BrainError::Invalid(m)) if m == "functional_search_request_invalid"
        ));

        // 4. Excessive limit fails closed
        assert!(matches!(
            engine.search_by_function(&[1.0, 2.0], MAX_ENGINE_SKILL_FIELDS + 1),
            Err(BrainError::Invalid(m)) if m == "functional_search_request_invalid"
        ));

        // 5. Uncertified runtime fails closed with composition_requires_certified_runtime
        let layout_sha = Sha256Digest::digest_bytes(b"layout");
        let mut field = runtime_field_with_artifact(&root, &layout_sha, "search-skill", &[1.0, 2.0]);
        field.structured_geometry = None;
        field.parameter_layout_sha256 = None;
        field.dense_materialization = None;
        field.functional_signature = vec![1.0, 0.0];
        let bank = SkillBank {
            generation: 1,
            fields: vec![field],
        };
        fs::create_dir_all(root.join("state")).unwrap();
        fs::write(
            engine.bank_path(),
            serde_json::to_vec_pretty(&bank).unwrap(),
        )
        .unwrap();

        let err = engine.search_by_function(&[1.0, 0.0], 5).unwrap_err();
        assert!(matches!(
            err,
            BrainError::Integrity(m) if m.starts_with("runtime_execution_not_authorized")
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_learned_controller_and_drive_fail_closed_invariants() {
        let root = isolated_engine_root("learned-controller-invariants");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 1. ControllerInvocation::validate contracts
        let valid_invocation = ControllerInvocation {
            schema: "cerebro.tidex.controller_invocation/v1".into(),
            session_id: SessionId::parse("session-ctl").unwrap(),
            state_before: vec![1.0, 2.0],
            promoted_observation_semantic_sha256: Sha256Digest::digest_bytes(b"obs"),
        };
        assert!(valid_invocation.validate().is_ok());

        let mut invalid_schema = valid_invocation.clone();
        invalid_schema.schema = "wrong.schema/v1".into();
        assert!(matches!(
            invalid_schema.validate(),
            Err(BrainError::Invalid(m)) if m == "controller_invocation_contract_invalid"
        ));

        let mut empty_state = valid_invocation.clone();
        empty_state.state_before = vec![];
        assert!(matches!(
            empty_state.validate(),
            Err(BrainError::Invalid(m)) if m == "controller_invocation_contract_invalid"
        ));

        let mut nan_state = valid_invocation.clone();
        nan_state.state_before = vec![1.0, f64::NAN];
        assert!(matches!(
            nan_state.validate(),
            Err(BrainError::Invalid(m)) if m == "controller_invocation_contract_invalid"
        ));

        // 2. compose_current_learned_controller_and_record with noncanonical config
        let noncanonical = BrainEngine {
            root: root.clone(),
            config: BrainConfig {
                min_observations: 1,
                ..BrainConfig::default()
            },
        };
        assert!(matches!(
            noncanonical.compose_current_learned_controller_and_record(&valid_invocation),
            Err(BrainError::Integrity(m)) if m == "noncanonical_runtime_config_forbidden"
        ));

        // 3. cognitive_route_from_drive fails closed on uncertified runtime
        let drive = CognitiveFieldDrive {
            evidence: vec![1.0],
            prediction_error: vec![0.0],
            inhibition: vec![0.0],
            risk: vec![0.0],
        };
        let err = engine.cognitive_route_from_drive(&[1.0], &drive, 1, 0.1).unwrap_err();
        assert!(matches!(
            err,
            BrainError::Integrity(m) if m.starts_with("runtime_execution_not_authorized")
        ));

        // 4. compose_cognitive_drive_and_record fails closed on bad observation digest
        assert!(matches!(
            engine.compose_cognitive_drive_and_record(&[1.0], &drive, 1, 0.1, "bad-hash"),
            Err(BrainError::Invalid(m)) if m.contains("source_observation_digest_invalid")
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn support_controller_execution_receipt_and_ledger_invariants() {
        let root = isolated_engine_root("ctl-exec-invariants");
        let invocation = ControllerInvocation {
            schema: "cerebro.tidex.controller_invocation/v1".into(),
            session_id: SessionId::parse("session-test-ctl").unwrap(),
            state_before: vec![0.5, -0.2],
            promoted_observation_semantic_sha256: Sha256Digest::digest_bytes(b"semantic-obs"),
        };
        let receipt = ControllerExecutionReceipt {
            schema: "cerebro.tidex.controller_execution_receipt/v1".into(),
            session_id: invocation.session_id.clone(),
            invocation_sha256: Sha256Digest::parse(digest_json(&invocation).unwrap()).unwrap(),
            controller_receipt_sha256: Sha256Digest::digest_bytes(b"ctrl-receipt"),
            state_before_sha256: Sha256Digest::parse(digest_json(&invocation.state_before).unwrap()).unwrap(),
            promoted_observation_semantic_sha256: invocation.promoted_observation_semantic_sha256.clone(),
            governed_composition_receipt_sha256: Sha256Digest::digest_bytes(b"gov-comp"),
            invocation: invocation.clone(),
        };

        // 1. First persist succeeds and writes ledger
        let recorded = persist_controller_execution(&root, receipt.clone()).unwrap();
        assert_eq!(recorded.receipt, receipt);
        assert!(!recorded.receipt_sha256.is_empty());
        assert!(!recorded.ledger_event_hash.is_empty());

        // 2. Second persist with identical receipt is idempotent
        let recorded2 = persist_controller_execution(&root, receipt.clone()).unwrap();
        assert_eq!(recorded.receipt_sha256, recorded2.receipt_sha256);
        assert_eq!(recorded.ledger_event_hash, recorded2.ledger_event_hash);

        // 3. Digest collision / corrupted file rejection:
        let receipt_file = PathBuf::from(&recorded.receipt_path);
        let mut tampered = receipt.clone();
        tampered.session_id = SessionId::parse("session-different").unwrap();
        let tampered_bytes = serialize_pretty_line(&tampered).unwrap();
        fs::write(&receipt_file, tampered_bytes).unwrap();

        assert!(matches!(
            persist_controller_execution(&root, receipt.clone()),
            Err(BrainError::Integrity(m)) if m == "private_file_reference_digest_mismatch"
                || m == "controller_execution_receipt_artifact_invalid"
                || m == "controller_execution_receipt_digest_collision"
        ));

        // 4. verify_controller_execution_receipt contract checks
        let mut bad_recorded = recorded.clone();
        bad_recorded.receipt_sha256 = "invalid-sha".into();
        assert!(matches!(
            verify_controller_execution_receipt(&root, &bad_recorded),
            Err(BrainError::Integrity(m)) if m == "controller_execution_receipt_contract_invalid"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn support_load_verified_governed_composition_receipt_invariants() {
        let root = isolated_engine_root("load-gov-comp-invariants");
        let valid_sha = Sha256Digest::digest_bytes(b"test-receipt");

        // 1. Invalid digest string
        assert!(matches!(
            load_verified_governed_composition_receipt(&root, root.join("any"), "not-a-valid-sha"),
            Err(BrainError::Integrity(m)) if m == "governed_composition_receipt_digest_invalid"
        ));

        // 2. Receipt path mismatch
        let wrong_path = root.join("wrong_path.json");
        assert!(matches!(
            load_verified_governed_composition_receipt(&root, &wrong_path, valid_sha.as_str()),
            Err(BrainError::Integrity(m)) if m == "governed_composition_receipt_identity_invalid"
        ));

        // 3. File has v1 schema -> historical only
        let v1_payload = serde_json::json!({
            "schema": "cerebro.tidex.governed_composition_receipt/v1"
        });
        let v1_bytes = serde_json::to_vec_pretty(&v1_payload).unwrap();
        let v1_sha = Sha256Digest::digest_bytes(&v1_bytes);
        let v1_path = root
            .join("state/governed_compositions/by-sha")
            .join(format!("{v1_sha}.json"));
        fs::create_dir_all(v1_path.parent().unwrap()).unwrap();
        fs::write(&v1_path, &v1_bytes).unwrap();

        assert!(matches!(
            load_verified_governed_composition_receipt(&root, &v1_path, v1_sha.as_str()),
            Err(BrainError::Integrity(m)) if m == "governed_composition_receipt_v1_historical_only"
        ));

        // 4. File has unknown schema
        let bad_schema_payload = serde_json::json!({
            "schema": "cerebro.tidex.unknown_composition_receipt/v99"
        });
        let bad_bytes = serde_json::to_vec_pretty(&bad_schema_payload).unwrap();
        let bad_sha = Sha256Digest::digest_bytes(&bad_bytes);
        let bad_path = root
            .join("state/governed_compositions/by-sha")
            .join(format!("{bad_sha}.json"));
        fs::create_dir_all(bad_path.parent().unwrap()).unwrap();
        fs::write(&bad_path, &bad_bytes).unwrap();

        assert!(matches!(
            load_verified_governed_composition_receipt(&root, &bad_path, bad_sha.as_str()),
            Err(BrainError::Integrity(m)) if m == "governed_composition_receipt_schema_invalid"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn support_close_verified_learning_finalization_inflight_invariants() {
        let root = isolated_engine_root("close-inflight-invariants");
        let op_key = Sha256Digest::digest_bytes(b"target-op");
        let receipt = LearningFinalizationReceipt {
            schema: "cerebro.tidex.learning_finalization_receipt/v1".into(),
            operation_key: op_key.clone(),
            session_id: SessionId::parse("session-inflight").unwrap(),
            adaptive_receipt_sha256: Sha256Digest::digest_bytes(b"adapt"),
            learning_finalization_input_sha256: Sha256Digest::digest_bytes(b"inp"),
            representation_evidence_receipt: PrivateFileReference::new(
                root.join("rep.json"),
                Sha256Digest::digest_bytes(b"rep"),
            ),
            representation_protocol_sha256: Sha256Digest::digest_bytes(b"proto"),
            representation_observation_bindings_sha256: Sha256Digest::digest_bytes(b"bind"),
            representation_observation_bindings: vec![],
            prior_corpus_digest: Sha256Digest::digest_bytes(b"prior"),
            prior_observation_count: 2,
            new_corpus_digest: Sha256Digest::digest_bytes(b"new"),
            new_observation_count: 3,
            archived_artifact_sha256: BTreeMap::new(),
            report_sha256: Sha256Digest::digest_bytes(b"rep-sha"),
            commit_operation_key: Sha256Digest::digest_bytes(b"commit-op"),
            commit_receipt_sha256: Sha256Digest::digest_bytes(b"commit-sha"),
            ledger_event_hash: Sha256Digest::digest_bytes(b"ledger-hash"),
        };
        let dummy_ref = PrivateFileReference::new(
            root.join("receipt.json"),
            Sha256Digest::digest_bytes(b"receipt-bytes"),
        );

        // 1. No inflight directory -> Ok(())
        assert!(close_verified_learning_finalization_inflight(&root, &receipt, &dummy_ref).is_ok());

        // 2. Inflight directory exists with unrelated transition directory
        let inflight_root = root.join("state/corpus_transitions/inflight");
        let other_op = Sha256Digest::digest_bytes(b"other-op");
        fs::create_dir_all(inflight_root.join(other_op.as_str())).unwrap();

        assert!(matches!(
            close_verified_learning_finalization_inflight(&root, &receipt, &dummy_ref),
            Err(BrainError::Integrity(m)) if m == "learning_finalization_unrelated_inflight_transition"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn support_learning_finalization_archive_invariants() {
        let root = isolated_engine_root("archive-invariants");
        let op_key = Sha256Digest::digest_bytes(b"archive-test-op");
        let mut receipt = LearningFinalizationReceipt {
            schema: "cerebro.tidex.learning_finalization_receipt/v1".into(),
            operation_key: op_key.clone(),
            session_id: SessionId::parse("session-archive-test").unwrap(),
            adaptive_receipt_sha256: Sha256Digest::digest_bytes(b"adapt"),
            learning_finalization_input_sha256: Sha256Digest::digest_bytes(b"inp"),
            representation_evidence_receipt: PrivateFileReference::new(
                root.join("rep.json"),
                Sha256Digest::digest_bytes(b"rep"),
            ),
            representation_protocol_sha256: Sha256Digest::digest_bytes(b"proto"),
            representation_observation_bindings_sha256: Sha256Digest::digest_bytes(b"bind"),
            representation_observation_bindings: vec![],
            prior_corpus_digest: Sha256Digest::digest_bytes(b"prior"),
            prior_observation_count: 2,
            new_corpus_digest: Sha256Digest::digest_bytes(b"new"),
            new_observation_count: 3,
            archived_artifact_sha256: BTreeMap::new(),
            report_sha256: Sha256Digest::digest_bytes(b"rep-sha"),
            commit_operation_key: Sha256Digest::digest_bytes(b"commit-op"),
            commit_receipt_sha256: Sha256Digest::digest_bytes(b"commit-sha"),
            ledger_event_hash: Sha256Digest::digest_bytes(b"ledger-hash"),
        };
        let dummy_intent_dir = root.join("intent_dir");

        // 1. Missing required artifacts in archived_artifact_sha256
        assert!(matches!(
            verify_learning_finalization_archive(&root, &receipt, &dummy_intent_dir),
            Err(BrainError::Integrity(m)) if m == "learning_finalization_archive_manifest_contract_invalid"
        ));

        // 2. Contains disallowed label
        receipt.archived_artifact_sha256.insert("disallowed_label.json".into(), Sha256Digest::zero());
        assert!(matches!(
            verify_learning_finalization_archive(&root, &receipt, &dummy_intent_dir),
            Err(BrainError::Integrity(m)) if m == "learning_finalization_archive_manifest_contract_invalid"
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn transition_commit_after_verified_corpus_transition_invariants() {
        let root = isolated_engine_root("commit-verified-invariants");
        let engine = BrainEngine {
            root: root.clone(),
            config: BrainConfig::default(),
        };

        // 1. Observations don't meet minimum observation requirements -> fail closed
        let few_obs = vec![sample_observation("obs-1", vec![1.0, 0.0])];
        assert!(engine.commit_after_verified_corpus_transition(&few_obs).is_err());

        // 2. Non-canonical runtime config fails closed
        let noncanonical = BrainEngine {
            root: root.clone(),
            config: BrainConfig {
                min_observations: 1,
                ..BrainConfig::default()
            },
        };
        assert!(matches!(
            noncanonical.commit_after_verified_corpus_transition(&few_obs),
            Err(BrainError::Integrity(m)) if m == "noncanonical_runtime_config_forbidden"
        ));

        let _ = fs::remove_dir_all(root);
    }
}

