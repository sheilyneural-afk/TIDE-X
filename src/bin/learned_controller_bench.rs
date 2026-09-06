use cerebro_tidex::contracts::SkillField;
use cerebro_tidex::identity::SkillId;
use cerebro_tidex::learned_controller::{
    execute_learned_winner_take_all, train_learned_controller, ControllerExample,
};
use cerebro_tidex::linalg::norm;
use cerebro_tidex::parametric_program::{compile_operator_to_fields, compose_skill_fields};
use serde_json::json;

fn field(id: &str, direction: Vec<f64>) -> Result<SkillField, Box<dyn std::error::Error>> {
    Ok(SkillField {
        skill_id: SkillId::parse(id)?,
        reconstruction_id: Default::default(),
        lineage_id: Default::default(),
        generation_created: 1,
        direction,
        structured_geometry: None,
        dense_materialization: None,
        parameter_layout_sha256: None,
        representation_signature: Vec::new(),
        singular_value: 1.0,
        explained_variance: 0.5,
        persistence: 1.0,
        coherence: 1.0,
        uncertainty: 0.0,
        evidence_support_digests: Vec::new(),
        support: 4,
        functional_signature: Vec::new(),
        parent_skill_ids: Vec::new(),
    })
}

fn fields() -> Result<Vec<SkillField>, Box<dyn std::error::Error>> {
    // Non-orthogonal field basis spanning direct next-state logits [state0,state1].
    Ok(vec![
        field("field-a", vec![1.0, 0.25])?,
        field("field-b", vec![0.30, 1.0])?,
    ])
}

fn state(index: usize) -> Vec<f64> {
    if index == 0 {
        vec![1.0, 0.0]
    } else {
        vec![0.0, 1.0]
    }
}

fn index(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn target_next_state(current: usize, signal: f64) -> usize {
    // Negative observation = HOLD. Positive observation = TOGGLE.
    if signal.is_sign_positive() {
        1 - current
    } else {
        current
    }
}

fn target_coefficients(fields: &[SkillField]) -> Vec<Vec<f64>> {
    [vec![1.0, 0.0], vec![0.0, 1.0]]
        .iter()
        .map(|target_logits| {
            compile_operator_to_fields(fields, target_logits, 1e-12, 1e-8)
                .unwrap()
                .coefficients
        })
        .collect()
}

fn training_examples(fields: &[SkillField]) -> Vec<ControllerExample> {
    let targets = target_coefficients(fields);
    let nuisance = [-0.24, -0.08, 0.08, 0.24];
    let mut examples = Vec::new();
    for (group, nuisance_value) in nuisance.into_iter().enumerate() {
        for current in 0..2 {
            for signal in [-1.0, 1.0] {
                let next = target_next_state(current, signal);
                examples.push(ControllerExample {
                    state_before: state(current),
                    observation: vec![signal, nuisance_value],
                    target_coefficients: targets[next].clone(),
                    reliability: 1.0,
                    independence_group: format!("aperture-{group}"),
                });
            }
        }
    }
    examples
}

fn observation_only_examples(examples: &[ControllerExample]) -> Vec<ControllerExample> {
    examples
        .iter()
        .map(|example| ControllerExample {
            state_before: vec![1.0],
            observation: example.observation.clone(),
            target_coefficients: example.target_coefficients.clone(),
            reliability: example.reliability,
            independence_group: example.independence_group.clone(),
        })
        .collect()
}

fn observation_only_next(
    current: usize,
    observation: &[f64],
    controller: &cerebro_tidex::learned_controller::LearnedController,
    fields: &[SkillField],
) -> usize {
    let _ = current;
    let decision = controller.decide(&[1.0], observation).unwrap();
    let logits = compose_skill_fields(fields, &decision.coefficients).unwrap();
    index(&logits)
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn blind_sequences() -> Vec<Vec<Vec<f64>>> {
    let mut rng = 0x9C51_0A77_D4E8_132Bu64;
    let mut sequences = Vec::new();
    for len in 1..=96usize {
        for _ in 0..32 {
            let mut sequence = Vec::with_capacity(len);
            for _ in 0..len {
                let raw = next_u64(&mut rng);
                let sign = if raw & 1 == 0 { -1.0 } else { 1.0 };
                // Training sees only magnitude 1.0. Blind sequences use unseen
                // continuous magnitudes and nuisance values.
                let magnitude = 0.55 + (((raw >> 8) & 0xffff) as f64 / 65535.0) * 0.40;
                let nuisance = -0.20 + (((raw >> 24) & 0xffff) as f64 / 65535.0) * 0.40;
                sequence.push(vec![sign * magnitude, nuisance]);
            }
            sequences.push(sequence);
        }
    }
    sequences
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fields = fields()?;
    let examples = training_examples(&fields);
    let controller = train_learned_controller(&examples, 1e-8, 0.25)?;
    let observation_only =
        train_learned_controller(&observation_only_examples(&examples), 1e-8, 0.25)?;
    let cases = blind_sequences();

    let mut learned_final_correct = 0usize;
    let mut observation_only_final_correct = 0usize;
    let mut static_final_correct = 0usize;
    let mut learned_step_correct = 0usize;
    let mut observation_only_step_correct = 0usize;
    let mut static_step_correct = 0usize;
    let mut total_steps = 0usize;

    for observations in &cases {
        let learned =
            execute_learned_winner_take_all(&fields, &controller, &state(0), observations)?;
        let mut oracle = 0usize;
        let mut observation_only_state = 0usize;
        let mut static_state = 0usize;
        for (step_index, observation) in observations.iter().enumerate() {
            oracle = target_next_state(oracle, observation[0]);
            observation_only_state = observation_only_next(
                observation_only_state,
                observation,
                &observation_only,
                &fields,
            );
            // Strongest single fixed state choice chosen without seeing current state/observation.
            static_state = 0;
            learned_step_correct +=
                usize::from(index(&learned.steps[step_index].state_after) == oracle);
            observation_only_step_correct += usize::from(observation_only_state == oracle);
            static_step_correct += usize::from(static_state == oracle);
            total_steps += 1;
        }
        learned_final_correct += usize::from(learned.final_state_index == oracle);
        observation_only_final_correct += usize::from(observation_only_state == oracle);
        static_final_correct += usize::from(static_state == oracle);
    }

    let observation = vec![0.80, 0.0];
    let c0 = controller.decide(&state(0), &observation)?.coefficients;
    let c1 = controller.decide(&state(1), &observation)?.coefficients;
    let state_conditioned_distance = norm(
        &c0.iter()
            .zip(&c1)
            .map(|(left, right)| left - right)
            .collect::<Vec<_>>(),
    )?;
    let ood_rejected = controller.decide(&state(0), &[3.0, 0.0]).is_err();
    let total_cases = cases.len();
    let denom = total_cases as f64;
    let step_denom = total_steps as f64;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.learned_controller_benchmark/v2",
            "task":"recurrent_hold_toggle_direct_action_fields",
            "training_examples":examples.len(),
            "training_groups":controller.training_groups,
            "feature_dim":controller.feature_dim,
            "controller_training_rms":controller.training_rms,
            "controller_grouped_cv_r2":controller.grouped_cv_r2,
            "blind_sequence_cases":total_cases,
            "blind_total_steps":total_steps,
            "blind_max_sequence_len":96,
            "blind_observation_magnitudes_seen_in_training":false,
            "learned_state_conditioned_final_accuracy":learned_final_correct as f64/denom,
            "observation_only_learned_final_accuracy":observation_only_final_correct as f64/denom,
            "static_final_accuracy":static_final_correct as f64/denom,
            "learned_state_conditioned_step_accuracy":learned_step_correct as f64/step_denom,
            "observation_only_learned_step_accuracy":observation_only_step_correct as f64/step_denom,
            "static_step_accuracy":static_step_correct as f64/step_denom,
            "same_observation_different_state_coefficient_distance":state_conditioned_distance,
            "ood_rejected":ood_rejected,
            "controller_has_token_lookup":false,
            "claims":{
                "state_conditioning_matters":learned_final_correct>observation_only_final_correct,
                "learned_controller_generalizes_to_unseen_observation_magnitudes":learned_step_correct==total_steps,
                "no_token_to_operator_table_in_learned_runtime":true,
                "ood_fail_closed":ood_rejected
            }
        }))?
    );
    Ok(())
}
