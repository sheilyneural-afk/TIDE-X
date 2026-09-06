//! Deterministic synthetic evidence used only by integration tests.

use cerebro_tidex::contracts::{ConfounderValue, DeltaObservation, ExperimentLineage};
use cerebro_tidex::error::BrainResult;
use cerebro_tidex::linalg::{add_scaled, normalize, sub};

#[derive(Debug, Clone)]
pub struct Phantom {
    pub observations: Vec<DeltaObservation>,
    pub true_skills: Vec<Vec<f64>>,
}

pub fn cognitive_phantom() -> BrainResult<Phantom> {
    let dim = 18usize;
    let mut h1 = (0..dim)
        .map(|index| ((index + 1) as f64 * 0.37).sin())
        .collect::<Vec<_>>();
    h1 = normalize(&h1)?;
    let mut h2 = (0..dim)
        .map(|index| ((index + 2) as f64 * 0.61).cos())
        .collect::<Vec<_>>();
    let projection = cerebro_tidex::linalg::dot(&h2, &h1)?;
    add_scaled(&mut h2, &h1, -projection)?;
    h2 = normalize(&h2)?;
    let mut h3 = (0..dim)
        .map(|index| (((index + 3) * (index + 1)) as f64 * 0.071).sin())
        .collect::<Vec<_>>();
    for direction in [&h1, &h2] {
        let projection = cerebro_tidex::linalg::dot(&h3, direction)?;
        add_scaled(&mut h3, direction, -projection)?;
    }
    h3 = normalize(&h3)?;
    let mut b1 = (0..dim)
        .map(|index| ((index + 5) as f64 * 0.19).cos())
        .collect::<Vec<_>>();
    for direction in [&h1, &h2, &h3] {
        let projection = cerebro_tidex::linalg::dot(&b1, direction)?;
        add_scaled(&mut b1, direction, -projection)?;
    }
    b1 = normalize(&b1)?;
    let mut b2 = (0..dim)
        .map(|index| ((index + 7) as f64 * 0.23).sin())
        .collect::<Vec<_>>();
    for direction in [&h1, &h2, &h3, &b1] {
        let projection = cerebro_tidex::linalg::dot(&b2, direction)?;
        add_scaled(&mut b2, direction, -projection)?;
    }
    b2 = normalize(&b2)?;

    fn splitmix_unit(mut value: u64) -> f64 {
        value = value.wrapping_add(0x9E3779B97F4A7C15);
        let mut mixed = value;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D049BB133111EB);
        mixed ^= mixed >> 31;
        ((mixed >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    let node_count = 16usize;
    let mut skill_state = Vec::new();
    let mut nuisance = Vec::new();
    for node in 0..node_count {
        skill_state.push(vec![
            1.2 * splitmix_unit(node as u64 * 17 + 11),
            1.2 * splitmix_unit(node as u64 * 19 + 101),
            1.2 * splitmix_unit(node as u64 * 23 + 1009),
        ]);
        nuisance.push(vec![
            0.8 * splitmix_unit(node as u64 * 29 + 77),
            0.7 * splitmix_unit(node as u64 * 31 + 707),
        ]);
    }

    let mut edges = Vec::new();
    for index in 0..node_count - 1 {
        edges.push((index, index + 1));
    }
    for index in 0..node_count - 2 {
        edges.push((index, index + 2));
    }
    for index in 0..node_count - 3 {
        if index % 2 == 0 {
            edges.push((index, index + 3));
        }
    }

    let mut observations = Vec::new();
    for (experiment, (from, to)) in edges.into_iter().enumerate() {
        let skill_delta = sub(&skill_state[to], &skill_state[from])?;
        let nuisance_delta = sub(&nuisance[to], &nuisance[from])?;
        let mut delta = vec![0.0; dim];
        for (coefficient, direction) in skill_delta.iter().zip([&h1, &h2, &h3]) {
            add_scaled(&mut delta, direction, *coefficient)?;
        }
        add_scaled(&mut delta, &b1, 0.65 * nuisance_delta[0])?;
        add_scaled(&mut delta, &b2, -0.55 * nuisance_delta[1])?;
        let tiny = ((experiment * 17 + 3) % 11) as f64 * 1e-5;
        for (parameter, value) in delta.iter_mut().enumerate() {
            *value += tiny * ((parameter + 1) as f64 * 0.13).sin();
        }
        let functional_response = vec![
            1.4 * skill_delta[0] - 0.2 * skill_delta[1] + 0.35 * skill_delta[2],
            -0.3 * skill_delta[0] + 1.2 * skill_delta[1] + 0.5 * skill_delta[2],
            0.25 * skill_delta[0] - 0.4 * skill_delta[1] + 1.5 * skill_delta[2],
        ];
        let quadrant =
            usize::from(nuisance_delta[0] >= 0.0) + 2 * usize::from(nuisance_delta[1] >= 0.0);
        observations.push(DeltaObservation {
            observation_id: cerebro_tidex::identity::ObservationId::parse(format!(
                "phantom-{experiment:03}"
            ))
            .unwrap(),
            from_checkpoint: format!("c{from:02}"),
            to_checkpoint: format!("c{to:02}"),
            generation: experiment as u64 + 1,
            delta,
            functional_response,
            confounders: vec![
                ConfounderValue {
                    name: "optimizer_axis".into(),
                    value: nuisance_delta[0],
                },
                ConfounderValue {
                    name: "format_axis".into(),
                    value: nuisance_delta[1],
                },
            ],
            reliability: 0.98,
            independence_group: format!("aperture-quadrant-{quadrant}"),
            experiment_lineage: ExperimentLineage {
                run_id: format!("phantom-run-{experiment:03}"),
                replicate_id: format!("phantom-quadrant-{quadrant}"),
                randomization_id: format!("phantom-randomization-{quadrant}"),
                dataset_split_digest: format!("{:064x}", 0x51u64),
                initial_checkpoint_digest: format!("{:064x}", 0x52u64),
                optimizer_config_digest: format!("{:064x}", 0x53u64),
                template_config_digest: format!("{:064x}", quadrant + 0x60),
            },
            dense_artifact: None,
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: cerebro_tidex::digest::ProvenanceDigest::from(
                cerebro_tidex::digest::Sha256Digest::parse(format!("{:064x}", experiment + 1))
                    .unwrap(),
            ),
        });
    }

    Ok(Phantom {
        observations,
        true_skills: vec![h1, h2, h3],
    })
}
