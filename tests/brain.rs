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
use std::path::PathBuf;
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
