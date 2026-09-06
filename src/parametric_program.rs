use crate::contracts::SkillField;
use crate::engine::ReconstructionReport;
use crate::error::{BrainError, BrainResult};
use crate::linalg::{dot, norm, solve, Matrix};
use crate::validation::symmetric_psd_condition;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OperatorCompilation {
    pub coefficients: Vec<f64>,
    pub relative_residual: f64,
    pub gram_condition_estimate: f64,
}

fn validate_fields(fields: &[SkillField]) -> BrainResult<usize> {
    let parameter_dim = fields
        .first()
        .map(|field| field.direction.len())
        .unwrap_or(0);
    if fields.is_empty()
        || parameter_dim == 0
        || fields.iter().any(|field| {
            field.direction.len() != parameter_dim
                || field.direction.iter().any(|value| !value.is_finite())
        })
    {
        return Err(BrainError::Invalid(
            "parametric_program_skill_fields_invalid".into(),
        ));
    }
    Ok(parameter_dim)
}

pub fn fields_from_report(report: &ReconstructionReport) -> BrainResult<&[SkillField]> {
    if report.fields.is_empty() || report.selected_rank != report.fields.len() {
        return Err(BrainError::Invalid(
            "parametric_program_report_fields_invalid".into(),
        ));
    }
    validate_fields(&report.fields)?;
    Ok(&report.fields)
}

/// Compose one parameter-space operator from SkillFields.
pub fn compose_skill_fields(fields: &[SkillField], coefficients: &[f64]) -> BrainResult<Vec<f64>> {
    let parameter_dim = validate_fields(fields)?;
    if coefficients.len() != fields.len() || coefficients.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid(
            "parametric_program_coefficient_shape".into(),
        ));
    }
    let mut output = vec![0.0; parameter_dim];
    for (alpha, field) in coefficients.iter().zip(fields) {
        for (dst, value) in output.iter_mut().zip(&field.direction) {
            *dst += alpha * value;
        }
    }
    Ok(output)
}

/// Compile a target operator into coefficients over an arbitrary, potentially
/// non-orthogonal SkillField basis. Unlike the original laboratory prototype,
/// this does not assume coefficient_k = <H_k, operator>. It solves the normal
/// equations G alpha = H^T operator with a small ridge and then verifies the
/// reconstruction residual fail-closed.
pub fn compile_operator_to_fields(
    fields: &[SkillField],
    operator: &[f64],
    ridge: f64,
    max_relative_residual: f64,
) -> BrainResult<OperatorCompilation> {
    let parameter_dim = validate_fields(fields)?;
    if operator.len() != parameter_dim
        || operator.iter().any(|value| !value.is_finite())
        || !ridge.is_finite()
        || ridge < 0.0
        || !max_relative_residual.is_finite()
        || max_relative_residual < 0.0
    {
        return Err(BrainError::Invalid(
            "parametric_program_operator_shape".into(),
        ));
    }

    let rank = fields.len();
    let mut gram = Matrix::zeros(rank, rank);
    let mut rhs = vec![0.0; rank];
    for i in 0..rank {
        rhs[i] = dot(&fields[i].direction, operator)?;
        for j in 0..rank {
            gram.set(i, j, dot(&fields[i].direction, &fields[j].direction)?);
        }
    }
    let gram_condition_estimate = symmetric_psd_condition(&gram, "parametric_program_gram")?;
    for i in 0..rank {
        gram.set(i, i, gram.get(i, i) + ridge);
    }
    let coefficients = solve(gram, rhs)?;
    let reconstructed = compose_skill_fields(fields, &coefficients)?;
    let residual = reconstructed
        .iter()
        .zip(operator)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let relative_residual = norm(&residual)? / norm(operator)?.max(1e-15);
    if relative_residual > max_relative_residual {
        return Err(BrainError::Numerical(format!(
            "parametric_program_operator_outside_skill_span:{relative_residual:.6e}"
        )));
    }
    Ok(OperatorCompilation {
        coefficients,
        relative_residual,
        gram_condition_estimate,
    })
}

/// Generic linear recurrent state transition. The runtime knows only matrix
/// multiplication and winner selection; the transition semantics live in the
/// composed operator values.
pub fn apply_parametric_transition(
    state: &[f64],
    operator: &[f64],
    state_dim: usize,
) -> BrainResult<Vec<f64>> {
    if state_dim == 0
        || state.len() != state_dim
        || operator.len() != state_dim * state_dim
        || state.iter().chain(operator).any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid(
            "parametric_program_transition_shape".into(),
        ));
    }
    let mut scores = vec![0.0; state_dim];
    for row in 0..state_dim {
        for col in 0..state_dim {
            scores[row] += operator[row * state_dim + col] * state[col];
        }
    }
    let winner = argmax(&scores)?;
    let mut next = vec![0.0; state_dim];
    next[winner] = 1.0;
    Ok(next)
}

fn argmax(values: &[f64]) -> BrainResult<usize> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid(
            "parametric_program_argmax_input".into(),
        ));
    }
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    Ok(best)
}

pub fn task_arithmetic_merge(operators: &[Vec<f64>], scale: f64) -> BrainResult<Vec<f64>> {
    if operators.is_empty()
        || !scale.is_finite()
        || operators[0].is_empty()
        || operators.iter().any(|row| {
            row.len() != operators[0].len() || row.iter().any(|value| !value.is_finite())
        })
    {
        return Err(BrainError::Invalid(
            "parametric_program_task_arithmetic_input".into(),
        ));
    }
    let mut output = vec![0.0; operators[0].len()];
    for operator in operators {
        for (dst, value) in output.iter_mut().zip(operator) {
            *dst += scale * value;
        }
    }
    Ok(output)
}

pub fn ties_merge(operators: &[Vec<f64>], density: f64) -> BrainResult<Vec<f64>> {
    if operators.is_empty()
        || operators[0].is_empty()
        || !density.is_finite()
        || !(0.0..=1.0).contains(&density)
        || operators.iter().any(|row| {
            row.len() != operators[0].len() || row.iter().any(|value| !value.is_finite())
        })
    {
        return Err(BrainError::Invalid("parametric_program_ties_input".into()));
    }
    let dim = operators[0].len();
    let mut trimmed = Vec::with_capacity(operators.len());
    for operator in operators {
        let mut order = (0..dim).collect::<Vec<_>>();
        order.sort_by(|a, b| {
            operator[*b]
                .abs()
                .total_cmp(&operator[*a].abs())
                .then_with(|| a.cmp(b))
        });
        let keep = ((dim as f64 * density).ceil() as usize).min(dim);
        let mut mask = vec![false; dim];
        for index in order.into_iter().take(keep) {
            if operator[index] != 0.0 {
                mask[index] = true;
            }
        }
        trimmed.push(
            operator
                .iter()
                .enumerate()
                .map(|(index, value)| if mask[index] { *value } else { 0.0 })
                .collect::<Vec<_>>(),
        );
    }
    let mut merged = vec![0.0; dim];
    for coordinate in 0..dim {
        let aggregate = trimmed.iter().map(|row| row[coordinate]).sum::<f64>();
        let elected = aggregate.signum();
        if elected == 0.0 {
            continue;
        }
        let agreeing = trimmed
            .iter()
            .map(|row| row[coordinate])
            .filter(|value| *value != 0.0 && value.signum() == elected)
            .collect::<Vec<_>>();
        if !agreeing.is_empty() {
            merged[coordinate] = agreeing.iter().sum::<f64>() / agreeing.len() as f64;
        }
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(id: &str, direction: Vec<f64>) -> SkillField {
        SkillField {
            skill_id: crate::identity::SkillId::parse(id).unwrap(),
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
            support: 3,
            functional_signature: Vec::new(),
            parent_skill_ids: Vec::new(),
        }
    }

    fn parity_fields() -> Vec<SkillField> {
        vec![
            field("h1", vec![1.0, 0.35, 0.35, 1.0]),
            field("h2", vec![0.25, 1.0, 1.0, 0.25]),
        ]
    }

    fn operators() -> Vec<Vec<f64>> {
        vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]]
    }

    #[test]
    fn nonorthogonal_fields_compile_both_operators() {
        let fields = parity_fields();
        for operator in operators() {
            let compiled = compile_operator_to_fields(&fields, &operator, 1e-12, 1e-8).unwrap();
            assert!(compiled.relative_residual < 1e-8);
            let reconstructed = compose_skill_fields(&fields, &compiled.coefficients).unwrap();
            assert!(
                norm(
                    &reconstructed
                        .iter()
                        .zip(operator)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>()
                )
                .unwrap()
                    < 1e-8
            );
        }
    }

    #[test]
    fn generic_parametric_transition_remains_available_without_token_router() {
        let hold = vec![1.0, 0.0, 0.0, 1.0];
        let toggle = vec![0.0, 1.0, 1.0, 0.0];
        assert_eq!(
            apply_parametric_transition(&[1.0, 0.0], &hold, 2).unwrap(),
            vec![1.0, 0.0]
        );
        assert_eq!(
            apply_parametric_transition(&[1.0, 0.0], &toggle, 2).unwrap(),
            vec![0.0, 1.0]
        );
    }

    fn test_report(fields: Vec<SkillField>, rank: usize) -> ReconstructionReport {
        let fields_json = serde_json::to_string(&fields).unwrap();
        let raw = format!(
            r#"{{
            "schema": "cerebro.tidex.reconstruction/v7",
            "source_tree_digest": "0000000000000000000000000000000000000000000000000000000000000000",
            "config_digest": "0000000000000000000000000000000000000000000000000000000000000000",
            "analysis_version_digest": "0000000000000000000000000000000000000000000000000000000000000000",
            "observation_count": 2,
            "observation_set_digest": "0000000000000000000000000000000000000000000000000000000000000000",
            "parameter_dimension": 4,
            "independence_groups": 1,
            "aperture_independence": {{
                "schema": "cerebro.tidex.aperture_independence/v1",
                "declared_groups": 1,
                "design_feature_count": 1,
                "active_design_features": 1,
                "effective_group_rank": 1.0,
                "effective_independent_groups": 1.0,
                "numerical_design_rank": 1,
                "unique_randomization_count": 1,
                "unique_replicate_count": 1,
                "lineage_verified": true,
                "max_cross_group_similarity": 0.0,
                "mean_cross_group_similarity": 0.0,
                "group_similarity_matrix": [[1.0]],
                "group_ids": ["g1"],
                "group_independence_weights": [1.0],
                "minimum_required_groups": 1,
                "independent_enough": true
            }},
            "resolution_map": {{
                "schema": "cerebro.tidex.resolution_map/v1",
                "field_count": 0,
                "field_geometry_numerical_rank": 0,
                "excitation_numerical_rank": 0,
                "resolved_rank": 0,
                "field_geometry_spectrum": [],
                "excitation_spectrum": [],
                "field_geometry_condition": 1.0,
                "excitation_condition": 1.0,
                "min_principal_angle_degrees": 90.0,
                "posterior_covariance": [],
                "fields": [],
                "all_fields_resolved": true,
                "unresolved_field_ids": []
            }},
            "confounder_names": [],
            "confounder_explained_fraction": 0.0,
            "cycle_rms": 0.0,
            "max_edge_residual": 0.0,
            "selected_rank": {rank},
            "effective_rank": {rank}.0,
            "condition_estimate": 1.0,
            "reconstruction_rms": 0.0,
            "normalized_reconstruction_rms": 0.0,
            "functional_cv_r2": 1.0,
            "inverse_mode": "spectral_inverse",
            "spectral_functional_cv_r2": 1.0,
            "persistent_functional_cv_r2": null,
            "persistent_coherence_threshold": null,
            "persistent_coherence_gap": null,
            "persistent_coverage_ratio": null,
            "persistent_cluster_stability": null,
            "persistent_parametric_cluster_stability": null,
            "persistent_functional_cluster_stability": null,
            "persistent_cluster_identity_min_margin": null,
            "persistent_cluster_assignment_consistent": null,
            "persistent_min_holdout_similarity": null,
            "persistent_cluster_sizes": [],
            "persistent_error": null,
            "representation_protocol_sha256": null,
            "representation_cv_r2": null,
            "representation_match_accuracy": null,
            "representation_mean_matched_cosine": null,
            "representation_min_match_margin": null,
            "dual_space_verified": null,
            "fields": {fields_json},
            "field_coefficients": [],
            "skill_source_mixtures": [],
            "promotion": {{
                "allowed": true,
                "reasons": [],
                "metrics": {{
                    "r2": 1.0,
                    "cycle_rms": 0.0,
                    "condition_estimate": 1.0,
                    "reconstruction_rms": 0.0,
                    "field_count": {rank}.0
                }}
            }}
        }}"#
        );
        serde_json::from_str(&raw).unwrap()
    }

    #[test]
    fn fields_from_report_and_validation() {
        let mut report = test_report(parity_fields(), 2);

        let fields = fields_from_report(&report).unwrap();
        assert_eq!(fields.len(), 2);

        // Rank mismatch
        report.selected_rank = 3;
        assert!(fields_from_report(&report).is_err());

        // Empty fields
        report.fields.clear();
        report.selected_rank = 0;
        assert!(fields_from_report(&report).is_err());

        // Field validation error cases
        let bad_field_dim = vec![
            field("h1", vec![1.0, 0.0]),
            field("h2", vec![1.0, 0.0, 0.0]),
        ];
        assert!(compose_skill_fields(&bad_field_dim, &[1.0, 1.0]).is_err());

        let bad_field_val = vec![field("h1", vec![1.0, f64::NAN])];
        assert!(compose_skill_fields(&bad_field_val, &[1.0]).is_err());

        let empty_field = vec![field("h1", vec![])];
        assert!(compose_skill_fields(&empty_field, &[1.0]).is_err());
    }

    #[test]
    fn compose_and_compile_error_paths() {
        let fields = parity_fields();

        // Mismatched coefficient len
        assert!(compose_skill_fields(&fields, &[1.0]).is_err());
        // Non-finite coefficient
        assert!(compose_skill_fields(&fields, &[1.0, f64::NAN]).is_err());

        // compile_operator_to_fields validation errors
        // Invalid operator length
        assert!(compile_operator_to_fields(&fields, &[1.0, 2.0], 1e-12, 1e-8).is_err());
        // Non-finite operator value
        assert!(
            compile_operator_to_fields(&fields, &[1.0, 2.0, f64::NAN, 4.0], 1e-12, 1e-8).is_err()
        );
        // Non-finite or negative ridge
        assert!(compile_operator_to_fields(&fields, &[1.0, 0.0, 0.0, 1.0], -1.0, 1e-8).is_err());
        assert!(
            compile_operator_to_fields(&fields, &[1.0, 0.0, 0.0, 1.0], f64::NAN, 1e-8).is_err()
        );
        // Non-finite or negative max_relative_residual
        assert!(compile_operator_to_fields(&fields, &[1.0, 0.0, 0.0, 1.0], 1e-12, -0.1).is_err());
        assert!(
            compile_operator_to_fields(&fields, &[1.0, 0.0, 0.0, 1.0], 1e-12, f64::NAN).is_err()
        );

        // Target operator completely outside span with strict max_relative_residual
        let orthogonal_target = vec![1.0, -1.0, 1.0, -1.0];
        let err = compile_operator_to_fields(&fields, &orthogonal_target, 1e-12, 1e-12);
        assert!(err.is_err());
    }

    #[test]
    fn parametric_transition_error_paths() {
        // Zero state dim
        assert!(apply_parametric_transition(&[], &[], 0).is_err());
        // State length mismatch
        assert!(apply_parametric_transition(&[1.0], &[1.0, 0.0, 0.0, 1.0], 2).is_err());
        // Operator length mismatch
        assert!(apply_parametric_transition(&[1.0, 0.0], &[1.0, 0.0, 1.0], 2).is_err());
        // Non-finite values
        assert!(apply_parametric_transition(&[f64::NAN, 0.0], &[1.0, 0.0, 0.0, 1.0], 2).is_err());
        assert!(
            apply_parametric_transition(&[1.0, 0.0], &[1.0, 0.0, f64::INFINITY, 1.0], 2).is_err()
        );
    }

    #[test]
    fn task_arithmetic_merge_validation_and_execution() {
        let op1 = vec![1.0, 2.0, 3.0];
        let op2 = vec![4.0, -1.0, 0.5];

        let merged = task_arithmetic_merge(&[op1.clone(), op2.clone()], 0.5).unwrap();
        assert_eq!(merged, vec![2.5, 0.5, 1.75]);

        // Validation errors
        assert!(task_arithmetic_merge(&[], 1.0).is_err());
        assert!(task_arithmetic_merge(&[vec![]], 1.0).is_err());
        assert!(task_arithmetic_merge(std::slice::from_ref(&op1), f64::NAN).is_err());
        assert!(task_arithmetic_merge(&[op1.clone(), vec![1.0, 2.0]], 1.0).is_err());
        assert!(task_arithmetic_merge(&[vec![f64::NAN, 1.0]], 1.0).is_err());
    }

    #[test]
    fn ties_merge_validation_and_execution() {
        let op1 = vec![2.0, -1.0, 0.0, 0.5];
        let op2 = vec![1.0, -2.0, 0.2, 0.0];
        let op3 = vec![-0.5, 1.5, 0.0, -0.5];

        let merged = ties_merge(&[op1.clone(), op2.clone(), op3.clone()], 0.75).unwrap();
        assert_eq!(merged.len(), 4);

        // Validation errors
        assert!(ties_merge(&[], 0.5).is_err());
        assert!(ties_merge(&[vec![]], 0.5).is_err());
        assert!(ties_merge(std::slice::from_ref(&op1), -0.1).is_err());
        assert!(ties_merge(std::slice::from_ref(&op1), 1.5).is_err());
        assert!(ties_merge(std::slice::from_ref(&op1), f64::NAN).is_err());
        assert!(ties_merge(&[op1.clone(), vec![1.0, 2.0]], 0.5).is_err());
        assert!(ties_merge(&[vec![f64::NAN, 1.0]], 0.5).is_err());

        // Density 0 keeps nothing
        let zero_density = ties_merge(&[op1.clone(), op2.clone()], 0.0).unwrap();
        assert_eq!(zero_density, vec![0.0, 0.0, 0.0, 0.0]);
    }
}
