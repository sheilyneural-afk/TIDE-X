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

pub use crate::engine_head::{CanonicalEngineHead, CorpusTransitionRecovery};
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
    CorpusTransitionJournal, CorpusTransitionPhase, CorpusTransitionRecoveryOutcome,
    CANONICAL_ENGINE_HEAD_MAX_BYTES, CANONICAL_ENGINE_HEAD_SCHEMA,
    CORPUS_TRANSITION_JOURNAL_MAX_BYTES, HARD_MAX_ENGINE_REVISION,
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
pub(super) use crate::security::verify_private_root;
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
}
