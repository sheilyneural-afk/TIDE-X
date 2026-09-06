#[path = "support/phantom.rs"]
mod phantom;

use cerebro_tidex::active::choose_active_aperture;
use cerebro_tidex::contracts::ApertureCandidate;
use cerebro_tidex::contracts::{BrainConfig, ProtectedCortex, ProtectedDirection};
use cerebro_tidex::engine::BrainEngine;
use cerebro_tidex::gauge::align_bases;
use cerebro_tidex::identity::{ApertureId, ProbeId};
use cerebro_tidex::linalg::{cosine, Matrix};
use cerebro_tidex::protected::project_to_safe_subspace;
use phantom::cognitive_phantom;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn test_private_root() -> &'static PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root =
            std::env::temp_dir().join(format!("cerebro-tidex-brain-tests-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        cerebro_tidex::security::secure_dir(&root).unwrap();
        std::env::set_var("TIDEX_PRIVATE_ROOT", &root);
        root
    })
}

#[test]
fn phantom_recovers_latent_skill_subspace_and_function() {
    let config = BrainConfig {
        require_structured_geometry_for_promotion: false,
        require_dual_space_for_promotion: false,
        ..BrainConfig::default()
    };
    let engine = BrainEngine::open(test_private_root(), config).unwrap();
    let p = cognitive_phantom().unwrap();
    let report = engine.analyze(&p.observations).unwrap();
    let overlap = BrainEngine::skill_subspace_overlap(&report.fields, &p.true_skills).unwrap();
    assert!(overlap > 0.90, "overlap={overlap}");
    assert!(
        report.functional_cv_r2 > 0.75,
        "r2={}",
        report.functional_cv_r2
    );
    assert!(report.cycle_rms < 0.05, "cycle={}", report.cycle_rms);
    assert!(report.selected_rank >= 3);
    assert!(report.promotion.allowed, "{:?}", report.promotion.reasons);
}

#[test]
fn analysis_is_exactly_invariant_to_observation_order() {
    let config = BrainConfig {
        require_structured_geometry_for_promotion: false,
        require_dual_space_for_promotion: false,
        ..BrainConfig::default()
    };
    let engine = BrainEngine::open(test_private_root(), config).unwrap();
    let phantom = cognitive_phantom().unwrap();
    let forward = engine.analyze(&phantom.observations).unwrap();
    let mut reversed = phantom.observations.clone();
    reversed.reverse();
    let backward = engine.analyze(&reversed).unwrap();
    assert_eq!(forward, backward);
    for field in &forward.fields {
        assert_eq!(field.support, field.evidence_support_digests.len());
        assert!(!field.evidence_support_digests.is_empty());
    }
}

#[test]
fn zero_reliability_cannot_acquire_fabricated_reconstruction_weight() {
    let engine = BrainEngine::open(test_private_root(), BrainConfig::default()).unwrap();
    let mut observations = cognitive_phantom().unwrap().observations;
    observations[0].reliability = 0.0;
    assert!(matches!(
        engine.analyze(&observations),
        Err(cerebro_tidex::BrainError::Invalid(message)) if message == "reliability_invalid"
    ));
}

#[test]
fn repeated_identical_evidence_does_not_inflate_skill_support() {
    use cerebro_tidex::contracts::SkillBank;
    use cerebro_tidex::tomography::assimilate_bank;

    let config = BrainConfig {
        require_structured_geometry_for_promotion: false,
        require_dual_space_for_promotion: false,
        ..BrainConfig::default()
    };
    let engine = BrainEngine::open(test_private_root(), config.clone()).unwrap();
    let report = engine
        .analyze(&cognitive_phantom().unwrap().observations)
        .unwrap();
    let expected = report
        .fields
        .iter()
        .map(|field| {
            (
                field.skill_id.clone(),
                (field.support, field.evidence_support_digests.clone()),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut bank = SkillBank::default();
    assimilate_bank(&mut bank, &report.fields, config.skill_match_cosine).unwrap();
    assimilate_bank(&mut bank, &report.fields, config.skill_match_cosine).unwrap();

    for field in &bank.fields {
        let (support, digests) = expected.get(&field.skill_id).unwrap();
        assert_eq!(field.support, *support);
        assert_eq!(&field.evidence_support_digests, digests);
        assert_eq!(field.support, field.evidence_support_digests.len());
    }
}

#[test]
fn gauge_alignment_resolves_permutation_and_sign() {
    let a = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let b = vec![
        vec![0.0, -1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0],
    ];
    let x = align_bases(&a, &b).unwrap();
    assert!(x.mean_abs_cosine > 0.999999);
    assert_eq!(x.assignment, vec![Some(2), Some(0), Some(1)]);
    assert_eq!(x.signs[1], -1.0);
}

#[test]
fn protected_projection_removes_known_damage_direction() {
    let cortex = ProtectedCortex {
        parameter_importance: vec![1.0, 1.0, 0.1],
        directions: vec![ProtectedDirection {
            probe_id: ProbeId::parse("old-skill").unwrap(),
            direction: vec![1.0, 0.0, 0.0],
            importance: 1.0,
        }],
        max_damage_ratio: 0.9,
    };
    let r = project_to_safe_subspace(&[1.0, 1.0, 0.0], &cortex).unwrap();
    assert!(r.projected[0].abs() < 1e-10);
    assert!(r.projected[1] > 0.0);
}

#[test]
fn protected_projection_is_joint_for_nonorthogonal_directions() {
    let q = 2.0_f64.sqrt().recip();
    let cortex = ProtectedCortex {
        parameter_importance: vec![1.0, 1.0],
        directions: vec![
            ProtectedDirection {
                probe_id: ProbeId::parse("d1").unwrap(),
                direction: vec![1.0, 0.0],
                importance: 1.0,
            },
            ProtectedDirection {
                probe_id: ProbeId::parse("d2").unwrap(),
                direction: vec![q, q],
                importance: 1.0,
            },
        ],
        max_damage_ratio: 2.0,
    };
    let result = project_to_safe_subspace(&[1.0, 1.0], &cortex).unwrap();
    let wdot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    assert!(wdot(&result.projected, &[1.0, 0.0]).abs() < 1e-8);
    assert!(wdot(&result.projected, &[q, q]).abs() < 1e-8);
    assert!(result.max_weighted_residual < 1e-8);
}

#[test]
fn active_aperture_prefers_information_when_costs_match() {
    let cov = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    let c = vec![
        ApertureCandidate {
            aperture_id: ApertureId::parse("weak").unwrap(),
            sensing_vector: vec![0.1, 0.0],
            noise_variance: 1.0,
            cost: 0.1,
            risk: 0.1,
        },
        ApertureCandidate {
            aperture_id: ApertureId::parse("strong").unwrap(),
            sensing_vector: vec![1.0, 1.0],
            noise_variance: 0.2,
            cost: 0.1,
            risk: 0.1,
        },
    ];
    let s = choose_active_aperture(&c, &cov, 0.1, 0.1).unwrap();
    assert_eq!(s.aperture_id.as_str(), "strong");
}

#[test]
fn phantom_true_skills_are_distinct() {
    let p = cognitive_phantom().unwrap();
    assert!(cosine(&p.true_skills[0], &p.true_skills[1]).unwrap().abs() < 1e-8);
}

#[test]
fn public_artifact_writer_rejects_non_private_root() {
    use cerebro_tidex::artifact::ArtifactWriteAuthority;
    use std::fs;
    let root = std::env::temp_dir().join(format!("cerebro-artifact-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    assert!(ArtifactWriteAuthority::open(&root).is_err());
    assert!(fs::read_dir(&root).unwrap().next().is_none());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn transport_map_recovers_known_linear_generation_map() {
    use cerebro_tidex::transport::learn_transport;
    let src = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, -1.0],
    ];
    let dst = src
        .iter()
        .map(|x| vec![2.0 * x[0] + x[1], -x[0] + 3.0 * x[1]])
        .collect::<Vec<_>>();
    let t = learn_transport(&src, &dst, 1e-9).unwrap();
    assert!(t.training_rms < 1e-6);
    let y = t.apply(&[0.5, 2.0]).unwrap();
    assert!((y[0] - 3.0).abs() < 1e-6);
    assert!((y[1] - 5.5).abs() < 1e-6);
}

#[test]
fn mvdr_minimizes_interference_under_unit_constraint() {
    use cerebro_tidex::interaction::mvdr_weights;
    let r = Matrix::from_rows(&[vec![10.0, 0.0], vec![0.0, 1.0]]).unwrap();
    let desired = [1.0, 1.0];
    let w = mvdr_weights(&r, &desired, 1e-9).unwrap();
    let constraint = w[0] + w[1];
    assert!((constraint - 1.0).abs() < 1e-9);
    assert!(w[0] < w[1]);
}

fn write_private_json<T: serde::Serialize>(
    root: &Path,
    relative: &str,
    value: &T,
) -> (PathBuf, cerebro_tidex::digest::Sha256Digest) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
        cerebro_tidex::security::secure_dir(parent).unwrap();
    }
    let bytes = serde_json::to_vec(value).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    cerebro_tidex::security::secure_file(&path).unwrap();
    let digest = cerebro_tidex::digest::Sha256Digest::digest_bytes(&bytes);
    (path, digest)
}

fn install_verified_sleep_evidence(
    root: &Path,
    report: &cerebro_tidex::engine::ReconstructionReport,
) -> cerebro_tidex::sleep_evidence::SleepEvidenceBundle {
    use cerebro_tidex::causal_credit::{
        certified_causal_priority_weights, estimate_causal_credit, CounterfactualEvaluation,
    };
    use cerebro_tidex::digest::{CausalCreditDigest, ProtectedMapDigest, ReportDigest};
    use cerebro_tidex::interaction::second_order_interactions;
    use cerebro_tidex::protected_map::{
        build_protected_cortex_map, load_protected_cortex, persist_protected_map,
        SensitivityEvidence,
    };
    use cerebro_tidex::sleep_evidence::{
        CausalCreditEvidenceSummary, InteractionEvidenceSummary, ProtectionEvidenceSummary,
        ReplayEvidenceSummary, SleepEvidenceBundle,
    };
    use cerebro_tidex::trust_region::{
        apply_causal_priority_trust_region, TrustRegionAllocationPolicy,
    };

    assert!(report.promotion.allowed);
    assert!(!report.fields.is_empty());
    let evidence_namespace = format!(
        "state/p2-evidence-{}",
        &report.observation_set_digest.as_str()[..16]
    );
    let field_ids = report
        .fields
        .iter()
        .map(|field| field.skill_id.clone())
        .collect::<Vec<_>>();
    assert!(field_ids.len() < usize::BITS as usize);
    let dense_dimension = report.fields[0]
        .dense_materialization
        .as_ref()
        .expect("sleep materializes every promotable field")
        .parameter_count as usize;
    assert!(dense_dimension > 0);

    let mut report_bytes = serde_json::to_vec_pretty(report).unwrap();
    report_bytes.push(b'\n');
    let report_sha256 = ReportDigest::from(cerebro_tidex::digest::Sha256Digest::digest_bytes(
        &report_bytes,
    ));

    let writer = cerebro_tidex::artifact::ArtifactWriteAuthority::open(root).unwrap();
    let mut sensitivity_a = vec![0.0f64; dense_dimension];
    let mut sensitivity_b = vec![0.0f64; dense_dimension];
    sensitivity_a[0] = 1.0;
    sensitivity_b[0] = 2.0;
    let sensitivity_a_artifact = writer
        .create_content_addressed_dvec(
            &sensitivity_a
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let sensitivity_b_artifact = writer
        .create_content_addressed_dvec(
            &sensitivity_b
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let protection_source = serde_json::json!({
        "schema":"cerebro.tidex.protected_sensitivity_evidence/v1",
        "task_labels_used":false,
        "evidence":[
            {
                "probe_id":"p2-probe-a",
                "artifact":sensitivity_a_artifact,
                "causal_damage_per_parameter_norm":1.0,
                "reliability":1.0
            },
            {
                "probe_id":"p2-probe-b",
                "artifact":sensitivity_b_artifact,
                "causal_damage_per_parameter_norm":2.0,
                "reliability":1.0
            }
        ]
    });
    let (protection_source_path, protection_source_sha256) = write_private_json(
        root,
        &format!("{evidence_namespace}/protection.json"),
        &protection_source,
    );
    let sensitivity = vec![
        SensitivityEvidence {
            probe_id: ProbeId::parse("p2-probe-a").unwrap(),
            sensitivity: sensitivity_a,
            reliability: 1.0,
            causal_damage: Some(1.0),
        },
        SensitivityEvidence {
            probe_id: ProbeId::parse("p2-probe-b").unwrap(),
            sensitivity: sensitivity_b,
            reliability: 1.0,
            causal_damage: Some(2.0),
        },
    ];
    let protected_report = build_protected_cortex_map(&sensitivity, 1.0, 1.0).unwrap();
    let protected_artifact = persist_protected_map(root, &protected_report).unwrap();
    let protected_wrapper = serde_json::json!({
        "schema":"cerebro.tidex.protected_map_benchmark/v2",
        "task_labels_used":false,
        "map":protected_artifact
    });
    let (protected_map_path, protected_map_raw_sha256) = write_private_json(
        root,
        &format!("{evidence_namespace}/protected-map.json"),
        &protected_wrapper,
    );
    let protected_map_sha256 = ProtectedMapDigest::from(protected_map_raw_sha256);
    let cortex = load_protected_cortex(root, &protected_artifact).unwrap();

    let seeds = [101_u64, 102, 103];
    let task = "p2-runtime";
    let full_mask = (1usize << field_ids.len()) - 1;
    let mut evaluations = Vec::new();
    let mut raw_results = Vec::new();
    for seed in seeds {
        for mask in 0..=full_mask {
            let active_fields = field_ids
                .iter()
                .enumerate()
                .filter_map(|(index, field)| {
                    (mask & (1usize << index) != 0).then_some(field.clone())
                })
                .collect::<Vec<_>>();
            let utility = active_fields.len() as f64;
            evaluations.push(CounterfactualEvaluation {
                context_id: format!("validation-seed-{seed}:{task}"),
                independence_group: format!("validation-seed-{seed}"),
                active_fields: active_fields.clone(),
                utility,
            });
            raw_results.push(serde_json::json!({
                "mask":mask,
                "active_fields":active_fields,
                "seed":seed,
                "metrics":{task:utility}
            }));
        }
    }
    let causal_report = estimate_causal_credit(&evaluations).unwrap();
    let causal_priority_weights =
        certified_causal_priority_weights(&causal_report, &field_ids).unwrap();
    let plan_sha256 = cerebro_tidex::digest::Sha256Digest::digest_bytes(b"p2-causal-plan-v1");
    let baseline = serde_json::json!({
        "schema":"cerebro.tidex.counterfactual_replay/v3",
        "report_sha256":report_sha256,
        "plan_sha256":plan_sha256,
        "field_ids":field_ids,
        "field_coefficients":vec![0.1f64; field_ids.len()],
        "validation_seeds":seeds,
        "validation_tasks":[task],
        "count_per_task":100,
        "blind_data_accessed":false,
        "evaluations":evaluations,
        "raw_results":raw_results
    });
    let (baseline_path, baseline_sha256) = write_private_json(
        root,
        &format!("{evidence_namespace}/causal-replay.json"),
        &baseline,
    );
    let causal_wrapper = serde_json::json!({
        "schema":"cerebro.tidex.causal_credit_benchmark/v3",
        "blind_data_accessed":false,
        "replay_sha256":baseline_sha256,
        "report_sha256":report_sha256,
        "plan_sha256":plan_sha256,
        "field_ids":field_ids,
        "causal_credit":causal_report
    });
    let (causal_path, causal_raw_sha256) = write_private_json(
        root,
        &format!("{evidence_namespace}/causal-credit.json"),
        &causal_wrapper,
    );
    let causal_credit_sha256 = CausalCreditDigest::from(causal_raw_sha256);

    let dense_fields = report
        .fields
        .iter()
        .map(|field| {
            let reference = field
                .dense_materialization
                .as_ref()
                .expect("promotable report field has dense materialization");
            cerebro_tidex::artifact::read_dvec_f32(root, reference)
                .unwrap()
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let interaction_matrix =
        second_order_interactions(&dense_fields, &cortex.parameter_importance).unwrap();
    let proposed_coefficients = vec![0.1f64; field_ids.len()];
    let proposed_cost =
        cerebro_tidex::trust_region::quadratic_cost(&interaction_matrix, &proposed_coefficients)
            .unwrap();
    let diagonal_budget = proposed_cost.max(1.0) * 2.0;
    let trust = apply_causal_priority_trust_region(
        &interaction_matrix,
        &proposed_coefficients,
        diagonal_budget,
        &causal_priority_weights,
    )
    .unwrap();
    let interaction_rows = (0..field_ids.len())
        .map(|row| interaction_matrix.row_vec(row))
        .collect::<Vec<_>>();
    let interaction_payload = serde_json::json!({
        "schema":"cerebro.tidex.trust_region_benchmark/v3",
        "report_sha256":report_sha256,
        "protected_map_sha256":protected_map_sha256,
        "causal_plan_sha256":plan_sha256,
        "causal_credit_sha256":causal_credit_sha256,
        "causal_priority_weight_kind":"causal_lower_confidence_bound_95",
        "field_ids":field_ids,
        "dense_parameter_dimension":dense_dimension,
        "interaction_matrix":interaction_rows,
        "diagonal_budget":diagonal_budget,
        "trust_region":trust
    });
    let (interaction_path, interaction_sha256) = write_private_json(
        root,
        &format!("{evidence_namespace}/trust-region.json"),
        &interaction_payload,
    );

    let full_utility = field_ids.len() as f64;
    let candidate_rows = seeds
        .iter()
        .map(|seed| {
            serde_json::json!({
                "seed":seed,
                "metrics":{task:full_utility},
                "mean":full_utility
            })
        })
        .collect::<Vec<_>>();
    let candidate = serde_json::json!({
        "schema":"cerebro.tidex.trust_region_functional_validation/v2",
        "report_sha256":report_sha256,
        "trust_region_sha256":interaction_sha256,
        "plan_sha256":plan_sha256,
        "field_ids":field_ids,
        "accepted_coefficients":trust.accepted_coefficients,
        "validation_seeds":seeds,
        "validation_tasks":[task],
        "count_per_task":100,
        "rows":candidate_rows,
        "blind_data_accessed":false
    });
    let (candidate_path, candidate_sha256) = write_private_json(
        root,
        &format!("{evidence_namespace}/candidate-replay.json"),
        &candidate,
    );

    let bundle = SleepEvidenceBundle {
        schema: "cerebro.tidex.sleep_evidence/v4".into(),
        corpus_digest: report.observation_set_digest.clone(),
        source_tree_digest: report.source_tree_digest.clone(),
        config_digest: report.config_digest.clone(),
        analysis_version_digest: report.analysis_version_digest.clone(),
        field_ids: field_ids.clone(),
        protection: ProtectionEvidenceSummary {
            source_path: protection_source_path.to_string_lossy().into_owned(),
            source_sha256: protection_source_sha256,
            protected_map_path: protected_map_path.to_string_lossy().into_owned(),
            protected_map_sha256,
            probe_count: protected_report.probe_count,
            parameter_dimension: protected_report.parameter_dimension,
            selected_rank: protected_report.selected_rank,
            causal_damage_supported_probes: protected_report.causal_damage_supported_probes,
            sensitivity_damage_correlation: protected_report.sensitivity_damage_correlation,
        },
        interaction: InteractionEvidenceSummary {
            source_path: interaction_path.to_string_lossy().into_owned(),
            source_sha256: interaction_sha256,
            causal_credit_sha256: causal_credit_sha256.clone(),
            field_ids: field_ids.clone(),
            dense_parameter_dimension: dense_dimension,
            proposed_quadratic_cost: trust.proposed_quadratic_cost,
            accepted_quadratic_cost: trust.accepted_quadratic_cost,
            diagonal_budget,
            trust_scale: trust.scale,
            allocation_policy: TrustRegionAllocationPolicy::CausalPriorityContractionV1,
            causal_priority_weights,
            component_retention: trust.component_retention.clone(),
        },
        replay: ReplayEvidenceSummary {
            baseline_source_path: baseline_path.to_string_lossy().into_owned(),
            baseline_source_sha256: baseline_sha256.clone(),
            candidate_source_path: candidate_path.to_string_lossy().into_owned(),
            candidate_source_sha256: candidate_sha256,
            paired_independent_groups: seeds.len(),
            mean_utility_delta: 0.0,
            standard_error: 0.0,
            max_acceptable_utility_loss: Some(0.01),
            blind_data_accessed: false,
        },
        causal_credit: CausalCreditEvidenceSummary {
            replay_source_path: baseline_path.to_string_lossy().into_owned(),
            replay_source_sha256: baseline_sha256,
            credit_source_path: causal_path.to_string_lossy().into_owned(),
            credit_source_sha256: causal_credit_sha256,
            report_sha256,
            plan_sha256,
            independent_group_count: causal_report.independent_group_count,
            field_count: causal_report.field_count,
            resolved_field_count: causal_report
                .fields
                .iter()
                .filter(|field| field.resolved)
                .count(),
            interaction_count: causal_report.pair_interactions.len(),
            resolved_interaction_count: causal_report
                .pair_interactions
                .iter()
                .filter(|pair| pair.resolved)
                .count(),
            blind_data_accessed: false,
        },
    };
    let _ = write_private_json(root, "state/sleep_evidence/current.json", &bundle);
    bundle
}

fn prepare_p2_learning_finalization(
    root: &Path,
    engine: &BrainEngine,
) -> cerebro_tidex::learning_finalization::LearningFinalizationInput {
    use cerebro_tidex::digest::{ObservationRecordDigest, ProvenanceDigest, Sha256Digest};
    use cerebro_tidex::identity::{CapabilityId, LearningTargetId, ObservationId};
    use cerebro_tidex::learning_orchestrator::{
        assimilate_persistent_learning_evidence, issue_next_persistent_learning_aperture,
        load_persistent_adaptive_learning_receipt, start_persistent_adaptive_learning,
        AdaptiveLearningPolicy, LearningExperimentEvidence, LearningTarget,
    };
    use cerebro_tidex::representation_evidence::{
        record_representation_evidence, representation_capture_sha256, RepresentationCapture,
        RepresentationEvidenceInstallRequest, RepresentationEvidenceInstallTarget,
        RepresentationShift, SealedRepresentationProtocol, REPRESENTATION_CAPTURE_SCHEMA,
        REPRESENTATION_EVIDENCE_REQUEST_SCHEMA, REPRESENTATION_PROTOCOL_SCHEMA,
    };

    const OBSERVATION_COUNT: usize = 24;
    const CAPABILITY_COUNT: usize = 6;
    let session_id = "p2-finalization-session";
    let target = LearningTarget {
        target_id: LearningTargetId::parse("p2-finalization-target").unwrap(),
        capability_ids: (0..CAPABILITY_COUNT)
            .map(|index| CapabilityId::parse(format!("p2-capability-{index}")).unwrap())
            .collect(),
        candidate_budget: 48,
        plan_steps: OBSERVATION_COUNT,
        noise_variance: 0.05,
        cost_weight: 0.0,
        risk_weight: 0.0,
    };
    let policy = AdaptiveLearningPolicy {
        schema: "cerebro.tidex.adaptive_learning_policy/v1".into(),
        outcome_utility_weight: 1.0,
        maximize_observed_value: true,
    };
    let started = start_persistent_adaptive_learning(root, session_id, &target, &policy).unwrap();
    let target_digest = started.receipt.target_digest.clone();

    let layout = cerebro_tidex::block_tomography::ParameterBlockLayout::from_shapes(&[
        cerebro_tidex::block_tomography::BlockShapeSpec {
            name: "p2-learning-weights".into(),
            shape: vec![18],
            count: 18,
        },
    ])
    .unwrap();
    let layout_bytes = serde_json::to_vec(&layout).unwrap();
    let layout_digest = Sha256Digest::digest_bytes(&layout_bytes);
    let (layout_path, written_layout_digest) = write_private_json(
        root,
        &format!("state/parameter_layouts/by-sha/{layout_digest}.json"),
        &layout,
    );
    assert_eq!(written_layout_digest, layout_digest);
    assert_eq!(
        layout_path.file_name().unwrap().to_string_lossy(),
        format!("{layout_digest}.json")
    );

    let phantom = cognitive_phantom().unwrap();
    assert!(phantom.observations.len() >= OBSERVATION_COUNT);
    let writer = cerebro_tidex::artifact::ArtifactWriteAuthority::open(root).unwrap();
    let mut raw_observations = Vec::with_capacity(OBSERVATION_COUNT);
    let mut install_targets = Vec::with_capacity(OBSERVATION_COUNT);

    for index in 0..OBSERVATION_COUNT {
        let issued = issue_next_persistent_learning_aperture(root, session_id).unwrap();
        let step = issued
            .receipt
            .cycle
            .pending_step
            .as_ref()
            .expect("issued adaptive aperture");
        let mut observation = phantom.observations[index].clone();
        observation.observation_id =
            ObservationId::parse(format!("p2-learning-{index:03}")).unwrap();
        observation.generation = index as u64 + 1;
        let base_function = observation.functional_response.clone();
        assert_eq!(base_function.len(), 3);
        observation.functional_response = vec![
            base_function[0],
            base_function[1],
            base_function[2],
            base_function[0] + 0.5 * base_function[1],
            base_function[1] - 0.25 * base_function[2],
            base_function[2] + 0.2 * base_function[0],
        ];
        observation.independence_group = step.aperture_id.to_string();
        observation.experiment_lineage.run_id = format!("p2-run-{index:03}");
        observation.experiment_lineage.replicate_id = format!("p2-replicate-{index:03}");
        observation.experiment_lineage.randomization_id = format!("p2-random-{index:03}");
        observation.experiment_lineage.dataset_split_digest =
            Sha256Digest::digest_bytes(format!("p2-dataset-{index}").as_bytes()).to_string();
        observation.experiment_lineage.initial_checkpoint_digest =
            Sha256Digest::digest_bytes(format!("p2-checkpoint-{index}").as_bytes()).to_string();
        observation.experiment_lineage.optimizer_config_digest =
            Sha256Digest::digest_bytes(format!("p2-optimizer-{index}").as_bytes()).to_string();
        observation.experiment_lineage.template_config_digest =
            Sha256Digest::digest_bytes(format!("p2-template-{index}").as_bytes()).to_string();
        observation.dense_artifact = Some(
            writer
                .create_content_addressed_dvec(
                    &observation
                        .delta
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
        );
        observation.parameter_layout_sha256 = Some(layout_digest.clone());
        observation.representation_artifact = None;
        observation.representation_protocol_sha256 = None;
        observation.provenance_digest = ProvenanceDigest::from(Sha256Digest::digest_bytes(
            format!("p2-provenance-{index}").as_bytes(),
        ));

        let observation = serde_json::from_slice::<cerebro_tidex::contracts::DeltaObservation>(
            &serde_json::to_vec(&observation).unwrap(),
        )
        .unwrap();
        let (observation_path, observation_sha256) = write_private_json(
            root,
            &format!(
                "state/p2-learning-observations/{}.json",
                observation.observation_id
            ),
            &observation,
        );
        let support = serde_json::json!({
            "schema":"cerebro.tidex.p2_learning_support/v1",
            "aperture_id":step.aperture_id,
            "observation_id":observation.observation_id,
        });
        let (support_path, support_sha256) = write_private_json(
            root,
            &format!("state/p2-learning-support/{index:03}.json"),
            &support,
        );
        let observed_value =
            cerebro_tidex::linalg::dot(&step.capability_weights, &observation.functional_response)
                .unwrap();
        let evidence = LearningExperimentEvidence {
            schema: "cerebro.tidex.learning_experiment_evidence/v1".into(),
            session_id: issued.receipt.session_id.clone(),
            target_digest: target_digest.clone(),
            aperture_id: step.aperture_id.clone(),
            observed_value,
            observation_id: observation.observation_id.clone(),
            observation: cerebro_tidex::authority::PrivateFileReference::new(
                observation_path.clone(),
                observation_sha256.clone(),
            ),
            evidence_files: vec![cerebro_tidex::authority::PrivateFileReference::new(
                support_path,
                support_sha256,
            )],
        };
        let (evidence_path, _) = write_private_json(
            root,
            &format!("state/p2-learning-envelopes/{index:03}.json"),
            &evidence,
        );
        assimilate_persistent_learning_evidence(root, session_id, &evidence_path).unwrap();

        install_targets.push(RepresentationEvidenceInstallTarget {
            observation_id: observation.observation_id.clone(),
            source_observation_path: observation_path.to_string_lossy().into_owned(),
            source_observation_sha256: ObservationRecordDigest::from(observation_sha256),
            destination_observation_path: root
                .join("state/representation_evidence/installed-observations")
                .join(format!("p2-learning-{index:03}.json"))
                .to_string_lossy()
                .into_owned(),
        });
        raw_observations.push(observation);
    }

    let completed = load_persistent_adaptive_learning_receipt(root, session_id).unwrap();
    assert!(completed.receipt.cycle.pending_step.is_none());
    assert_eq!(
        completed.receipt.cycle.completed_evidence.len(),
        OBSERVATION_COUNT
    );
    assert_eq!(
        completed.receipt.cycle.session.completed_aperture_ids.len(),
        OBSERVATION_COUNT
    );

    let preliminary_engine = BrainEngine::open(
        root,
        BrainConfig {
            require_dual_space_for_promotion: false,
            ..BrainConfig::default()
        },
    )
    .unwrap();
    let preliminary = preliminary_engine.analyze(&raw_observations).unwrap();
    assert!(
        preliminary.promotion.allowed,
        "P2 prospective preliminary blockers: {:?}",
        preliminary.promotion.reasons
    );
    assert!(preliminary.fields.len() >= 2);

    let mut canonical = raw_observations.clone();
    canonical.sort_by(|left, right| {
        left.observation_id
            .cmp(&right.observation_id)
            .then_with(|| left.provenance_digest.cmp(&right.provenance_digest))
    });
    let field_count = preliminary.fields.len();
    let dual_model = cerebro_tidex::dual_space::DualSpaceModel {
        fields: &preliminary.fields,
        field_coefficients: &preliminary.field_coefficients,
        skill_source_mixtures: &preliminary.skill_source_mixtures,
        parameter_inverse_mode: preliminary.inverse_mode,
        parameter_promotable: preliminary.promotion.allowed,
        functional_cv_r2: preliminary.functional_cv_r2,
    };
    let dual_config = cerebro_tidex::dual_space::DualSpaceAnalysisConfig {
        ridge: BrainConfig::default().ridge,
        minimum_independence_groups: BrainConfig::default().min_independent_apertures,
        minimum_representation_cv_r2: BrainConfig::default().min_representation_cv_r2,
        minimum_match_accuracy: BrainConfig::default().min_representation_match_accuracy,
        minimum_match_margin: BrainConfig::default().min_representation_match_margin,
    };
    assert_eq!(
        field_count, 3,
        "P2 phantom is expected to expose three latent fields"
    );
    let mut selected_shifts = None;
    let scale_grid = [0.0625_f64, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    for second_scale in scale_grid {
        for third_scale in scale_grid {
            let scales = [1.0, second_scale, third_scale];
            let representations = canonical
                .iter()
                .enumerate()
                .map(|(index, observation)| {
                    let shift = preliminary.field_coefficients[index]
                        .iter()
                        .zip(scales)
                        .map(|(coefficient, scale)| coefficient * scale)
                        .collect::<Vec<_>>();
                    cerebro_tidex::dual_space::RepresentationObservation {
                        observation_id: observation.observation_id.clone(),
                        shift,
                    }
                })
                .collect::<Vec<_>>();
            let dual = cerebro_tidex::dual_space::analyze_dual_space(
                &dual_model,
                &canonical,
                &representations,
                dual_config,
            )
            .unwrap();
            if dual.representation_supported {
                selected_shifts = Some(representations);
                break;
            }
        }
        if selected_shifts.is_some() {
            break;
        }
    }
    let selected_shifts = selected_shifts.expect(
        "P2 prospective corpus requires a linear representation satisfying the canonical dual-space gate",
    );
    let representation_dim = selected_shifts[0].shift.len();
    let shifts = selected_shifts
        .into_iter()
        .map(|representation| RepresentationShift {
            observation_id: representation.observation_id,
            raw_dimension: representation_dim as u64,
            shift: representation.shift,
        })
        .collect::<Vec<_>>();
    let capture = RepresentationCapture {
        schema: REPRESENTATION_CAPTURE_SCHEMA.into(),
        observations: shifts,
    };
    let capture: RepresentationCapture =
        serde_json::from_slice(&serde_json::to_vec(&capture).unwrap()).unwrap();
    let protocol = SealedRepresentationProtocol {
        schema: REPRESENTATION_PROTOCOL_SCHEMA.into(),
        source_representation_sha256: representation_capture_sha256(&capture).unwrap(),
        probe_sha256: Sha256Digest::digest_bytes(b"p2-learning-probe"),
        probe_text_sha256: Sha256Digest::digest_bytes(b"p2-learning-probe"),
        forbidden_vocabulary_sha256: Sha256Digest::digest_bytes(b"p2-learning-forbidden"),
        task_labels_used: false,
        probe_vocabulary_overlap: Vec::new(),
        probe_count: 1,
        layer_count: 1,
        hidden_dim: representation_dim as u64,
        raw_dimension_per_observation: representation_dim as u64,
        sketch_dim: representation_dim as u64,
        sketch_seed: 29,
    };
    let request = RepresentationEvidenceInstallRequest {
        schema: REPRESENTATION_EVIDENCE_REQUEST_SCHEMA.into(),
        protocol,
        capture,
        installations: install_targets,
    };
    let stable_request: RepresentationEvidenceInstallRequest =
        serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
    assert_eq!(stable_request, request);
    let (request_path, _) = write_private_json(
        root,
        "state/p2-learning-representation-request.json",
        &stable_request,
    );
    let representation_receipt = record_representation_evidence(root, &request_path).unwrap();
    let representation_receipt_path = root
        .join("state/representation_evidence/receipts/by-request-sha")
        .join(format!("{}.json", representation_receipt.request_sha256));

    let input = cerebro_tidex::learning_finalization::prepare_learning_finalization(
        root,
        session_id,
        &representation_receipt_path,
    )
    .unwrap();
    assert_eq!(input.observations.len(), OBSERVATION_COUNT);
    let prospective = engine.analyze(&input.observations).unwrap();
    assert!(
        prospective.promotion.allowed,
        "P2 prospective finalization blockers: {:?}; rep_r2={:?} acc={:?}",
        prospective.promotion.reasons,
        prospective.representation_cv_r2,
        prospective.representation_match_accuracy
    );
    input
}

#[test]
fn sleep_cycle_promotes_after_verified_evidence_and_certifies_runtime() {
    let root = test_private_root().clone();
    let engine = BrainEngine::open(&root, BrainConfig::default()).unwrap();
    let phantom = cognitive_phantom().unwrap();
    let layout = cerebro_tidex::block_tomography::ParameterBlockLayout::from_shapes(&[
        cerebro_tidex::block_tomography::BlockShapeSpec {
            name: "block_1".into(),
            shape: vec![18],
            count: 18,
        },
    ])
    .unwrap();
    let mut layout_bytes = serde_json::to_vec_pretty(&layout).unwrap();
    layout_bytes.push(b'\n');
    let layout_digest = cerebro_tidex::digest::Sha256Digest::digest_bytes(&layout_bytes);
    let layout_dir = root.join("state/parameter_layouts/by-sha");
    std::fs::create_dir_all(&layout_dir).unwrap();
    cerebro_tidex::security::secure_dir(&layout_dir).unwrap();
    let layout_path = layout_dir.join(format!("{layout_digest}.json"));
    std::fs::write(&layout_path, &layout_bytes).unwrap();
    cerebro_tidex::security::secure_file(&layout_path).unwrap();

    let writer = cerebro_tidex::artifact::ArtifactWriteAuthority::open(&root).unwrap();
    let mut observations = phantom.observations.clone();
    for (idx, obs) in observations.iter_mut().enumerate() {
        obs.parameter_layout_sha256 = Some(layout_digest.clone());
        obs.dense_artifact = Some(
            writer
                .create_content_addressed_dvec(&[(idx + 1) as f32 * 0.01; 18])
                .unwrap(),
        );
    }
    // The persisted corpus is the authority. Stabilize every IEEE-754 value
    // through the exact JSON representation that the engine will later reopen
    // before deriving any evidence from the observations.
    observations = observations
        .into_iter()
        .map(|observation| {
            let pretty = serde_json::to_string_pretty(&observation).unwrap() + "\n";
            serde_json::from_str(&pretty).unwrap()
        })
        .collect();

    // Derive dual-space measurements from the latent coordinates found by the
    // parameter reconstruction itself; no task labels or observation order are used.
    let preliminary_engine = BrainEngine::open(
        &root,
        BrainConfig {
            require_dual_space_for_promotion: false,
            ..BrainConfig::default()
        },
    )
    .unwrap();
    let preliminary = preliminary_engine.analyze(&observations).unwrap();
    assert!(
        preliminary.promotion.allowed,
        "preliminary blockers: {:?}",
        preliminary.promotion.reasons
    );
    let mut canonical = observations.clone();
    canonical.sort_by(|left, right| {
        left.observation_id
            .cmp(&right.observation_id)
            .then_with(|| left.provenance_digest.cmp(&right.provenance_digest))
    });
    let coordinates = canonical
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            (
                observation.observation_id.clone(),
                preliminary
                    .skill_source_mixtures
                    .iter()
                    .map(|row| row[index])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let representation_dim = preliminary.fields.len();
    assert!(representation_dim > 0);

    let protocol = serde_json::json!({
        "schema": "cerebro.tidex.representation_protocol/v1",
        "source_representation_sha256": cerebro_tidex::digest::Sha256Digest::digest_bytes(b"sleep-capture"),
        "probe_sha256": cerebro_tidex::digest::Sha256Digest::digest_bytes(b"sleep-probe"),
        "probe_text_sha256": cerebro_tidex::digest::Sha256Digest::digest_bytes(b"sleep-probe"),
        "forbidden_vocabulary_sha256": cerebro_tidex::digest::Sha256Digest::digest_bytes(b"sleep-forbidden"),
        "task_labels_used": false,
        "probe_vocabulary_overlap": [],
        "probe_count": 1,
        "layer_count": 1,
        "hidden_dim": representation_dim,
        "raw_dimension_per_observation": representation_dim,
        "sketch_dim": representation_dim,
        "sketch_seed": 17
    });
    let protocol_bytes = serde_json::to_vec(&protocol).unwrap();
    let protocol_digest = cerebro_tidex::digest::Sha256Digest::digest_bytes(&protocol_bytes);
    let protocol_dir = root.join("state/representation_protocols/by-sha");
    std::fs::create_dir_all(&protocol_dir).unwrap();
    cerebro_tidex::security::secure_dir(&root.join("state")).unwrap();
    cerebro_tidex::security::secure_dir(&root.join("state/representation_protocols")).unwrap();
    cerebro_tidex::security::secure_dir(&protocol_dir).unwrap();
    let protocol_path = protocol_dir.join(format!("{protocol_digest}.json"));
    std::fs::write(&protocol_path, &protocol_bytes).unwrap();
    cerebro_tidex::security::secure_file(&protocol_path).unwrap();

    for obs in &mut observations {
        let shift = coordinates.get(&obs.observation_id).unwrap();
        obs.representation_artifact = Some(writer.create_content_addressed_f64(shift).unwrap());
        obs.representation_protocol_sha256 = Some(
            cerebro_tidex::digest::RepresentationProtocolDigest::from(protocol_digest.clone()),
        );
    }
    // Adding representation references must not reintroduce a transient in-memory
    // corpus identity. Reopen the fully populated records exactly as persistence will.
    observations = observations
        .into_iter()
        .map(|observation| {
            let pretty = serde_json::to_string_pretty(&observation).unwrap() + "\n";
            serde_json::from_str(&pretty).unwrap()
        })
        .collect();
    let promotable = engine.analyze(&observations).unwrap();
    assert!(
        promotable.promotion.allowed,
        "canonical blockers: {:?}; rep_r2={:?} acc={:?} margin={:?}",
        promotable.promotion.reasons,
        promotable.representation_cv_r2,
        promotable.representation_match_accuracy,
        promotable.representation_min_match_margin
    );

    let obs_dir = root.join("state/observations");
    std::fs::create_dir_all(&obs_dir).unwrap();
    cerebro_tidex::security::secure_dir(&obs_dir).unwrap();
    for obs in &observations {
        let pretty = serde_json::to_string_pretty(obs).unwrap() + "\n";
        let parsed: cerebro_tidex::contracts::DeltaObservation =
            serde_json::from_str(&pretty).unwrap();
        assert_eq!(
            &parsed, obs,
            "authoritative observation JSON must be stable after canonical round-trip"
        );
        let canonical_bytes = serde_json::to_vec(&parsed).unwrap();
        let digest =
            cerebro_tidex::digest::Sha256Digest::digest_bytes(&canonical_bytes).to_string();
        let name = format!("{}-{}.json", parsed.observation_id, &digest[..16]);
        let path = obs_dir.join(name);
        std::fs::write(&path, pretty).unwrap();
        cerebro_tidex::security::secure_file(&path).unwrap();
    }

    let mut canonical_observations = observations.clone();
    canonical_observations.sort_by(|left, right| {
        left.observation_id
            .cmp(&right.observation_id)
            .then_with(|| left.provenance_digest.cmp(&right.provenance_digest))
    });

    // Reproduce the deterministic dense materialization performed by sleep_cycle
    // so the evidence bundle can bind to the exact report before the first sleep.
    let mut materialized_report = promotable.clone();
    for (field, mixture) in materialized_report
        .fields
        .iter_mut()
        .zip(&materialized_report.skill_source_mixtures)
    {
        let support = cerebro_tidex::validation::source_support_indices(mixture).unwrap();
        let sources = support
            .into_iter()
            .map(|index| {
                (
                    canonical_observations[index]
                        .dense_artifact
                        .clone()
                        .expect("promotable observation has dense artifact"),
                    mixture[index],
                )
            })
            .collect::<Vec<_>>();
        field.dense_materialization =
            Some(writer.combine_content_addressed_dvec(&sources).unwrap());
    }
    materialized_report =
        serde_json::from_slice(&serde_json::to_vec_pretty(&materialized_report).unwrap()).unwrap();
    let stable_materialized: cerebro_tidex::engine::ReconstructionReport =
        serde_json::from_slice(&serde_json::to_vec_pretty(&materialized_report).unwrap()).unwrap();
    assert_eq!(stable_materialized, materialized_report);

    // Install independently replayable protection, causal-credit, trust-region,
    // and non-inferiority evidence before the first sleep transaction.
    let bundle = install_verified_sleep_evidence(&root, &materialized_report);
    let verification = cerebro_tidex::sleep_evidence::verify_sleep_evidence(
        &root,
        &bundle,
        &cerebro_tidex::sleep_evidence::SleepEvidenceExpectation {
            corpus_digest: &materialized_report.observation_set_digest,
            report_sha256: &bundle.causal_credit.report_sha256,
            source_tree_digest: &materialized_report.source_tree_digest,
            config_digest: &materialized_report.config_digest,
            analysis_version_digest: &materialized_report.analysis_version_digest,
            fields: &materialized_report.fields,
            observations: &canonical_observations,
            source_mixtures: &materialized_report.skill_source_mixtures,
        },
    )
    .unwrap();
    assert!(verification.verified, "{:?}", verification.reasons);

    let certified = engine.sleep_cycle().unwrap();
    assert!(
        certified.promoted,
        "promotion={:?} evidence={:?} bundle_corpus={} sleep_corpus={}",
        certified.reconstruction.promotion.reasons,
        certified.evidence_verification.reasons,
        bundle.corpus_digest,
        certified.reconstruction.observation_set_digest
    );
    assert!(!certified.idempotent);
    assert!(certified.evidence_verification.verified);
    assert_eq!(
        certified.active_skill_count,
        certified.reconstruction.fields.len()
    );

    // Simulate a crash after the sleep transaction, immutable artifacts, ledger
    // event, and current pointers were installed but before the receipt remained
    // available. Replay must authenticate the existing staging/event and reseal
    // exactly the same operation without re-promoting the bank.
    let certified_state: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("state/sleep_state.json")).unwrap())
            .unwrap();
    let certified_operation = certified_state["operation_key"].as_str().unwrap();
    let sleep_receipt_path = root
        .join("state/sleep_receipts")
        .join(format!("{certified_operation}.json"));
    let prior_sleep_receipt = std::fs::read(&sleep_receipt_path).unwrap();
    std::fs::remove_file(&sleep_receipt_path).unwrap();
    let recovered_sleep = engine.sleep_cycle().unwrap();
    assert!(!recovered_sleep.promoted);
    assert!(recovered_sleep.idempotent);
    assert!(recovered_sleep.evidence_verification.verified);
    assert_eq!(
        std::fs::read(&sleep_receipt_path).unwrap(),
        prior_sleep_receipt
    );

    let certified_again = engine.sleep_cycle().unwrap();
    assert!(!certified_again.promoted);
    assert!(certified_again.idempotent);
    assert!(certified_again.evidence_verification.verified);

    let status = engine.status().unwrap();
    assert_eq!(status["integrity"]["certified"], true);
    assert_eq!(status["integrity"]["execution_authorized"], true);
    assert_eq!(status["integrity"]["certification_status"], "certified");
    assert!(engine.bank_energy().unwrap() > 0.0);

    let query = certified.reconstruction.fields[0]
        .functional_signature
        .clone();
    let found = engine
        .search_by_function(&query, certified.active_skill_count)
        .unwrap();
    assert!(!found.is_empty());
    assert_eq!(found[0].0, certified.reconstruction.fields[0].skill_id);

    let source_observation_sha256 = cerebro_tidex::digest::Sha256Digest::digest_bytes(
        &serde_json::to_vec(&canonical_observations[0]).unwrap(),
    );
    let field_count = certified.active_skill_count;
    let drive = cerebro_tidex::cognitive_field::CognitiveFieldDrive {
        evidence: vec![1.0; field_count],
        prediction_error: vec![0.0; field_count],
        inhibition: vec![0.0; field_count],
        risk: vec![0.0; field_count],
    };
    let governed = engine
        .compose_cognitive_drive_and_record(
            &vec![0.0; field_count],
            &drive,
            1,
            0.01,
            source_observation_sha256.as_str(),
        )
        .unwrap();
    assert!(governed.state.converged);
    assert_eq!(governed.route.selected_field_ids.len(), 1);
    let verified_receipt = cerebro_tidex::engine::load_verified_governed_composition_receipt(
        &root,
        Path::new(&governed.composition.receipt_path),
        &governed.composition.receipt_sha256,
    )
    .unwrap();
    assert_eq!(verified_receipt, governed.composition.receipt);

    // Same cognitive request resolves through the immutable operation pointer.
    let governed_again = engine
        .compose_cognitive_drive_and_record(
            &vec![0.0; field_count],
            &drive,
            1,
            0.01,
            source_observation_sha256.as_str(),
        )
        .unwrap();
    assert_eq!(
        governed_again.composition.receipt_sha256,
        governed.composition.receipt_sha256
    );

    // The operation pointer is derived state. Losing only that pointer must not
    // duplicate the immutable receipt or ledger event: replay reconstructs the
    // pointer from the already authenticated receipt and returns the same authority.
    let operation_pointer = root
        .join("state/governed_compositions/by-operation")
        .join(format!(
            "{}.json",
            governed.composition.receipt.operation_key
        ));
    std::fs::remove_file(&operation_pointer).unwrap();
    let governed_after_pointer_loss = engine
        .compose_cognitive_drive_and_record(
            &vec![0.0; field_count],
            &drive,
            1,
            0.01,
            source_observation_sha256.as_str(),
        )
        .unwrap();
    assert_eq!(
        governed_after_pointer_loss.composition.receipt_sha256,
        governed.composition.receipt_sha256
    );
    assert_eq!(
        governed_after_pointer_loss.composition.ledger_event_hash,
        governed.composition.ledger_event_hash
    );
    assert!(operation_pointer.is_file());

    let finalization_input = prepare_p2_learning_finalization(&root, &engine);
    assert_eq!(finalization_input.observations.len(), 24);
    let still_certified = engine.status().unwrap();
    assert_eq!(still_certified["integrity"]["execution_authorized"], true);

    // A later evidence bundle may revoke execution without corrupting the
    // historical certified transaction. The active bank remains installed, but
    // runtime execution must fail closed until a new certified sleep succeeds.
    let mut revoked_bundle = bundle.clone();
    revoked_bundle.corpus_digest = cerebro_tidex::digest::CorpusDigest::from(
        cerebro_tidex::digest::Sha256Digest::digest_bytes(b"p2-revoked-corpus"),
    );
    let _ = write_private_json(&root, "state/sleep_evidence/current.json", &revoked_bundle);
    let revoked = engine.sleep_cycle().unwrap();
    assert!(!revoked.promoted);
    assert!(!revoked.idempotent);
    assert!(!revoked.evidence_verification.verified);
    assert!(revoked
        .evidence_verification
        .reasons
        .iter()
        .any(|reason| reason == "sleep_evidence_corpus_mismatch"));
    let revoked_status = engine.status().unwrap();
    assert_eq!(revoked_status["integrity"]["integrity_healthy"], true);
    assert_eq!(revoked_status["integrity"]["certified"], false);
    assert_eq!(revoked_status["integrity"]["execution_authorized"], false);
    assert_eq!(
        revoked_status["integrity"]["certification_status"],
        "revoked"
    );
    assert!(engine.search_by_function(&query, 1).is_err());

    let prior_head: cerebro_tidex::engine::CanonicalEngineHead = serde_json::from_slice(
        &std::fs::read(root.join("state/canonical_engine_head.json")).unwrap(),
    )
    .unwrap();
    let finalization_receipt = engine
        .commit_finalized_learning_session(&finalization_input)
        .unwrap();
    assert_eq!(
        finalization_receipt.prior_observation_count,
        canonical_observations.len()
    );
    assert_eq!(
        finalization_receipt.new_observation_count,
        finalization_input.observations.len()
    );
    assert_ne!(
        finalization_receipt.prior_corpus_digest,
        finalization_receipt.new_corpus_digest
    );

    let archive = root
        .join("state/corpus_transitions/by-operation")
        .join(finalization_receipt.operation_key.as_str());
    assert!(archive.join("observations").is_dir());
    assert!(archive.join("observations_manifest.json").is_file());
    assert!(archive.join("sleep_state.json").is_file());
    assert!(archive.join("skill_bank.json").is_file());
    assert!(archive.join("sleep_evidence_current.json").is_file());
    assert!(archive.join("transition_intent").is_dir());

    let new_head: cerebro_tidex::engine::CanonicalEngineHead = serde_json::from_slice(
        &std::fs::read(root.join("state/canonical_engine_head.json")).unwrap(),
    )
    .unwrap();
    assert!(new_head.revision > prior_head.revision);
    let mut cursor = new_head.clone();
    let mut prior_is_ancestor = cursor.manifest_digest == prior_head.manifest_digest;
    while !prior_is_ancestor {
        let Some(parent_digest) = cursor.parent_digest.clone() else {
            break;
        };
        if parent_digest == prior_head.manifest_digest {
            prior_is_ancestor = true;
            break;
        }
        let parent_path = root
            .join("state/canonical_engine_heads/by-sha")
            .join(format!("{parent_digest}.json"));
        cursor = serde_json::from_slice(&std::fs::read(parent_path).unwrap()).unwrap();
        assert!(cursor.revision < new_head.revision);
    }
    assert!(
        prior_is_ancestor,
        "revoked pre-transition head must remain in canonical ancestry"
    );
    assert_eq!(
        new_head.corpus_digest.as_deref(),
        Some(finalization_receipt.new_corpus_digest.as_str())
    );
    assert_eq!(
        new_head.observation_count,
        finalization_input.observations.len()
    );
    assert!(new_head.incomplete_transition.is_none());
    assert_eq!(
        new_head.reconstruction_report_sha256.as_deref(),
        Some(finalization_receipt.report_sha256.as_str())
    );

    // Replaying the exact finalization input must return the already authenticated
    // immutable receipt and must not archive/publish a second time.
    let replayed = engine
        .commit_finalized_learning_session(&finalization_input)
        .unwrap();
    assert_eq!(replayed, finalization_receipt);
    let replay_head: cerebro_tidex::engine::CanonicalEngineHead = serde_json::from_slice(
        &std::fs::read(root.join("state/canonical_engine_head.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(replay_head, new_head);

    // Simulate a crash after the learning-finalization receipt was sealed but
    // before the fail-closed inflight marker was retired. Recreate only that
    // marker from its authenticated archived intent; recovery must verify the
    // existing receipt, retire the marker again, and preserve the published corpus.
    let archived_transition_intent = archive.join("transition_intent");
    let inflight_root = root.join("state/corpus_transitions/inflight");
    std::fs::create_dir_all(&inflight_root).unwrap();
    cerebro_tidex::security::secure_dir(&inflight_root).unwrap();
    let recreated_inflight = inflight_root.join(finalization_receipt.operation_key.as_str());
    std::fs::rename(&archived_transition_intent, &recreated_inflight).unwrap();
    let closed_recovery = engine.recover_incomplete_corpus_transition().unwrap();
    assert_eq!(
        closed_recovery.outcome,
        cerebro_tidex::engine::CorpusTransitionRecoveryOutcome::ClosedVerifiedReceipt
    );
    assert_eq!(
        closed_recovery.operation_key.as_ref(),
        Some(&finalization_receipt.operation_key)
    );
    assert!(!recreated_inflight.exists());
    assert!(archived_transition_intent.is_dir());
    let recovered_head: cerebro_tidex::engine::CanonicalEngineHead = serde_json::from_slice(
        &std::fs::read(root.join("state/canonical_engine_head.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        recovered_head.corpus_digest.as_deref(),
        Some(finalization_receipt.new_corpus_digest.as_str())
    );
    assert!(recovered_head.incomplete_transition.is_none());

    let finalized_report: cerebro_tidex::engine::ReconstructionReport = serde_json::from_slice(
        &std::fs::read(
            root.join("state/reports")
                .join(format!("{}.json", finalization_receipt.report_sha256)),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        finalized_report.observation_set_digest.as_str(),
        finalization_receipt.new_corpus_digest.as_str()
    );
    let _new_bundle = install_verified_sleep_evidence(&root, &finalized_report);
    let recertified = engine.sleep_cycle().unwrap();
    assert!(
        recertified.promoted,
        "post-finalization promotion={:?} evidence={:?}",
        recertified.reconstruction.promotion.reasons, recertified.evidence_verification.reasons
    );
    assert!(recertified.evidence_verification.verified);
    assert_eq!(
        recertified.corpus_digest.as_str(),
        finalization_receipt.new_corpus_digest.as_str()
    );
    let recertified_status = engine.status().unwrap();
    assert_eq!(recertified_status["integrity"]["certified"], true);
    assert_eq!(
        recertified_status["integrity"]["execution_authorized"],
        true
    );
    assert_eq!(
        recertified_status["integrity"]["certification_status"],
        "certified"
    );

    // Train the persisted controller only from governed compositions bound to
    // the completed adaptive evidence and the real finalization receipt.
    use cerebro_tidex::digest::{
        AdaptiveLearningReceiptDigest, ControllerDatasetDigest, SkillBankDigest,
    };
    use cerebro_tidex::learned_controller::{
        load_persisted_runtime_learned_controller, train_learned_controller,
        train_persisted_runtime_learned_controller, ControllerExample,
        ControllerSupervisionEvidence, ControllerTrainingDataset, ControllerTrainingRecord,
        LearnedControllerBinding, LearnedControllerPolicy,
    };
    use cerebro_tidex::learning_orchestrator::{
        load_persistent_adaptive_learning_receipt, EvidenceReference,
    };

    let bank_bytes = std::fs::read(root.join("state/skill_bank.json")).unwrap();
    let active_bank: cerebro_tidex::contracts::SkillBank =
        serde_json::from_slice(&bank_bytes).unwrap();
    let active_bank_sha256 = SkillBankDigest::from(
        cerebro_tidex::digest::Sha256Digest::digest_bytes(&bank_bytes),
    );
    let field_ids = active_bank
        .fields
        .iter()
        .map(|field| field.skill_id.clone())
        .collect::<Vec<_>>();
    assert!(field_ids.len() >= 2);
    let adaptive =
        load_persistent_adaptive_learning_receipt(&root, finalization_receipt.session_id.as_str())
            .unwrap();
    assert_eq!(
        adaptive.receipt_sha256.as_digest(),
        &finalization_receipt.adaptive_receipt_sha256
    );

    let finalization_path = root
        .join("state/learning_finalizations")
        .join(format!("{}.json", finalization_receipt.operation_key));
    let finalization_reference = EvidenceReference::new(
        finalization_path.clone(),
        cerebro_tidex::digest::Sha256Digest::digest_bytes(
            &std::fs::read(&finalization_path).unwrap(),
        ),
    );
    let report_path = root
        .join("state/reports")
        .join(format!("{}.json", finalization_receipt.report_sha256));
    let report_reference =
        EvidenceReference::new(report_path, finalization_receipt.report_sha256.clone());

    let mut training_records = Vec::new();
    let mut preview_examples = Vec::new();
    let mut observed_targets = Vec::<Vec<f64>>::new();
    for (index, (adaptive_digest, adaptive_evidence)) in adaptive
        .receipt
        .cycle
        .completed_evidence_sha256
        .iter()
        .zip(&adaptive.receipt.cycle.completed_evidence)
        .take(24)
        .enumerate()
    {
        let raw_observation: cerebro_tidex::contracts::DeltaObservation =
            serde_json::from_slice(&std::fs::read(&adaptive_evidence.observation.path).unwrap())
                .unwrap();
        let mapping = finalization_receipt
            .representation_observation_bindings
            .iter()
            .find(|binding| binding.observation_id == adaptive_evidence.observation_id)
            .unwrap();
        let drive_index = index % 2;
        let state_before = vec![if drive_index == 0 { -1.0 } else { 1.0 }];
        let mut evidence_drive = vec![0.0; field_ids.len()];
        evidence_drive[drive_index] = 4.0;
        let drive = cerebro_tidex::cognitive_field::CognitiveFieldDrive {
            evidence: evidence_drive,
            prediction_error: vec![0.0; field_ids.len()],
            inhibition: vec![0.0; field_ids.len()],
            risk: vec![0.0; field_ids.len()],
        };
        let governed = engine
            .compose_cognitive_drive_and_record(
                &vec![0.0; field_ids.len()],
                &drive,
                1,
                0.01,
                mapping.promoted_observation_semantic_sha256.as_str(),
            )
            .unwrap();
        let target_coefficients = governed.composition.receipt.accepted_coefficients.clone();
        observed_targets.push(target_coefficients.clone());
        let governed_reference = EvidenceReference::new(
            PathBuf::from(&governed.composition.receipt_path),
            cerebro_tidex::digest::Sha256Digest::parse(&governed.composition.receipt_sha256)
                .unwrap(),
        );
        let supervision = ControllerSupervisionEvidence {
            schema: "cerebro.tidex.learned_controller_supervision/v1".into(),
            session_id: finalization_receipt.session_id.clone(),
            target_digest: adaptive.receipt.target_digest.clone(),
            adaptive_evidence_sha256: adaptive_digest.clone(),
            observation_id: adaptive_evidence.observation_id.clone(),
            active_bank_sha256: active_bank_sha256.clone(),
            field_ids: field_ids.clone(),
            governed_composition_receipt: governed_reference,
            state_before: state_before.clone(),
            target_coefficients: target_coefficients.clone(),
            reliability: raw_observation.reliability,
            independence_group: raw_observation.independence_group.clone(),
        };
        let (supervision_path, supervision_sha256) = write_private_json(
            &root,
            &format!("state/p2-controller-supervision/{index:03}.json"),
            &supervision,
        );
        training_records.push(ControllerTrainingRecord {
            schema: "cerebro.tidex.learned_controller_training_record/v1".into(),
            state_before: state_before.clone(),
            observation_id: adaptive_evidence.observation_id.clone(),
            observation: adaptive_evidence.observation.clone(),
            target_coefficients: target_coefficients.clone(),
            reliability: raw_observation.reliability,
            independence_group: raw_observation.independence_group.clone(),
            adaptive_evidence_sha256: adaptive_digest.clone(),
            supervision: EvidenceReference::new(supervision_path, supervision_sha256),
        });
        preview_examples.push(ControllerExample {
            state_before,
            observation: raw_observation.functional_response,
            target_coefficients,
            reliability: raw_observation.reliability,
            independence_group: raw_observation.independence_group,
        });
    }
    assert_eq!(training_records.len(), 24);
    assert!(observed_targets.windows(2).any(|pair| pair[0] != pair[1]));
    let preview = train_learned_controller(&preview_examples, 1e-8, 0.25).unwrap();
    assert!(
        preview.grouped_cv_r2 > 0.95,
        "controller preview grouped_cv_r2={}",
        preview.grouped_cv_r2
    );
    assert!(
        preview.training_rms < 1e-4,
        "controller preview training_rms={}",
        preview.training_rms
    );

    let dataset = ControllerTrainingDataset {
        schema: "cerebro.tidex.learned_controller_training_dataset/v1".into(),
        session_id: finalization_receipt.session_id.clone(),
        target_digest: adaptive.receipt.target_digest.clone(),
        records: training_records,
    };
    let (dataset_path, dataset_raw_sha256) =
        write_private_json(&root, "state/p2-controller-training-dataset.json", &dataset);
    let dataset_sha256 = ControllerDatasetDigest::from(dataset_raw_sha256);
    let controller_binding = LearnedControllerBinding {
        schema: "cerebro.tidex.learned_controller_binding/v1".into(),
        session_id: finalization_receipt.session_id.clone(),
        target_digest: adaptive.receipt.target_digest.clone(),
        adaptive_receipt_sha256: AdaptiveLearningReceiptDigest::from(
            finalization_receipt.adaptive_receipt_sha256.clone(),
        ),
        finalization_receipt: finalization_reference,
        reconstruction_report: report_reference,
        active_bank_sha256: active_bank_sha256.clone(),
        dataset_sha256,
    };
    let controller_policy = LearnedControllerPolicy {
        schema: "cerebro.tidex.learned_controller_policy/v1".into(),
        ridge: 1e-8,
        ood_margin_fraction: 0.25,
        minimum_grouped_cv_r2: 0.95,
        maximum_training_rms: 1e-4,
        minimum_independent_groups: 3,
    };
    let trained = train_persisted_runtime_learned_controller(
        &root,
        &dataset_path,
        &controller_policy,
        &controller_binding,
    )
    .unwrap();
    let loaded =
        load_persisted_runtime_learned_controller(&root, finalization_receipt.session_id.as_str())
            .unwrap();
    assert_eq!(loaded.receipt_sha256, trained.receipt_sha256);

    let first_record = &dataset.records[0];
    let first_mapping = finalization_receipt
        .representation_observation_bindings
        .iter()
        .find(|binding| binding.observation_id == first_record.observation_id)
        .unwrap();
    let invocation = cerebro_tidex::engine::ControllerInvocation {
        schema: "cerebro.tidex.controller_invocation/v1".into(),
        session_id: finalization_receipt.session_id.clone(),
        state_before: first_record.state_before.clone(),
        promoted_observation_semantic_sha256: first_mapping
            .promoted_observation_semantic_sha256
            .clone(),
    };
    let controller_execution = engine
        .compose_current_learned_controller_and_record(&invocation)
        .unwrap();
    assert_eq!(
        controller_execution.receipt.session_id,
        invocation.session_id
    );
    assert_eq!(
        controller_execution
            .receipt
            .controller_receipt_sha256
            .as_str(),
        trained.receipt_sha256.as_str()
    );
    let controller_execution_again = engine
        .compose_current_learned_controller_and_record(&invocation)
        .unwrap();
    assert_eq!(controller_execution_again, controller_execution);

    let recovery = engine.recover_incomplete_corpus_transition().unwrap();
    assert_eq!(
        recovery.outcome,
        cerebro_tidex::engine::CorpusTransitionRecoveryOutcome::NoIncompleteTransition
    );
}
