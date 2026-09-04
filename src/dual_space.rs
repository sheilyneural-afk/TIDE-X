#![allow(clippy::needless_range_loop)]

use crate::contracts::{DeltaObservation, ReconstructionInverseMode, SkillField};
use crate::error::{BrainError, BrainResult};
use crate::linalg::{weighted_normal_solve, Matrix};
use crate::validation::{independence_group_folds, regression_r2, source_support_indices};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepresentationObservation {
    pub observation_id: String,
    pub shift: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct DualSpaceModel<'a> {
    pub fields: &'a [SkillField],
    pub field_coefficients: &'a [Vec<f64>],
    pub skill_source_mixtures: &'a [Vec<f64>],
    pub parameter_inverse_mode: ReconstructionInverseMode,
    pub parameter_promotable: bool,
    pub functional_cv_r2: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepresentationFit {
    pub output_dim: usize,
    pub grouped_cv_r2: f64,
    pub cross_aperture_match_accuracy: f64,
    pub mean_matched_cosine: f64,
    pub min_match_margin: f64,
    /// SkillField -> representation-space signature.
    pub field_signatures: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DualSpaceField {
    pub skill_id: String,
    pub functional_signature: Vec<f64>,
    pub representation_signature: Vec<f64>,
    pub provenance_digests: Vec<String>,
    pub parameter_uncertainty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DualSpaceReport {
    pub schema: String,
    pub parameter_inverse_mode: ReconstructionInverseMode,
    pub parameter_promotable: bool,
    pub functional_cv_r2: f64,
    pub representation_cv_r2: f64,
    pub representation_match_accuracy: f64,
    pub representation_mean_matched_cosine: f64,
    pub representation_min_match_margin: f64,
    pub combined_cv_floor: f64,
    pub representation_supported: bool,
    pub dual_space_verified: bool,
    pub fields: Vec<DualSpaceField>,
}

fn coefficient_matrix(model: &DualSpaceModel<'_>, observation_count: usize) -> BrainResult<Matrix> {
    if model.fields.is_empty()
        || model.field_coefficients.len() != observation_count
        || model.skill_source_mixtures.len() != model.fields.len()
        || model
            .field_coefficients
            .iter()
            .any(|row| row.len() != model.fields.len())
        || model
            .skill_source_mixtures
            .iter()
            .any(|row| row.len() != observation_count)
    {
        return Err(BrainError::Invalid("dual_space_field_model_invalid".into()));
    }
    Matrix::from_rows(model.field_coefficients)
}

fn ordered_representation_rows(
    observations: &[DeltaObservation],
    representations: &[RepresentationObservation],
) -> BrainResult<Vec<Vec<f64>>> {
    if observations.is_empty() || observations.len() != representations.len() {
        return Err(BrainError::Invalid(
            "dual_space_representation_count".into(),
        ));
    }
    let by_id = representations
        .iter()
        .map(|observation| (observation.observation_id.as_str(), &observation.shift))
        .collect::<BTreeMap<_, _>>();
    let mut rows = Vec::with_capacity(observations.len());
    let mut dim = None;
    for observation in observations {
        let shift = by_id
            .get(observation.observation_id.as_str())
            .ok_or_else(|| BrainError::Invalid("dual_space_representation_id_missing".into()))?;
        if shift.is_empty() || shift.iter().any(|value| !value.is_finite()) {
            return Err(BrainError::Invalid(
                "dual_space_representation_invalid".into(),
            ));
        }
        if let Some(expected) = dim {
            if shift.len() != expected {
                return Err(BrainError::Invalid(
                    "dual_space_representation_dimension_mismatch".into(),
                ));
            }
        } else {
            dim = Some(shift.len());
        }
        rows.push((*shift).clone());
    }
    Ok(rows)
}

fn fit_output(
    coefficients: &Matrix,
    rows: &[Vec<f64>],
    observation_indices: &[usize],
    observations: &[DeltaObservation],
    ridge: f64,
) -> BrainResult<Vec<Vec<f64>>> {
    let x = Matrix::from_rows(
        &observation_indices
            .iter()
            .map(|&index| coefficients.row_vec(index))
            .collect::<Vec<_>>(),
    )?;
    let weights = observation_indices
        .iter()
        .map(|&index| observations[index].reliability.clamp(1e-4, 1.0))
        .collect::<Vec<_>>();
    let mut betas = Vec::with_capacity(rows[0].len());
    for output in 0..rows[0].len() {
        let target = observation_indices
            .iter()
            .map(|&index| rows[index][output])
            .collect::<Vec<_>>();
        betas.push(weighted_normal_solve(&x, &target, &weights, ridge)?);
    }
    Ok(betas)
}

fn predict(coefficients: &[f64], output_betas: &[Vec<f64>]) -> Vec<f64> {
    output_betas
        .iter()
        .map(|beta| beta.iter().zip(coefficients).map(|(b, x)| b * x).sum())
        .collect()
}

pub fn fit_representation_map(
    coefficients: &Matrix,
    observations: &[DeltaObservation],
    representation_rows: &[Vec<f64>],
    ridge: f64,
    minimum_groups: usize,
) -> BrainResult<RepresentationFit> {
    if coefficients.rows != observations.len()
        || coefficients.rows != representation_rows.len()
        || coefficients.cols == 0
        || representation_rows.is_empty()
        || representation_rows[0].is_empty()
    {
        return Err(BrainError::Invalid(
            "dual_space_representation_fit_shape".into(),
        ));
    }
    let folds = independence_group_folds(observations, minimum_groups)?;
    let mut actual = Vec::new();
    let mut predicted = Vec::new();
    for fold in &folds {
        let betas = fit_output(
            coefficients,
            representation_rows,
            &fold.train,
            observations,
            ridge,
        )?;
        for &index in &fold.test {
            actual.push(representation_rows[index].clone());
            predicted.push(predict(coefficients.row(index), &betas));
        }
    }
    let all = (0..observations.len()).collect::<Vec<_>>();
    let betas = fit_output(coefficients, representation_rows, &all, observations, ridge)?;
    let mut field_signatures = vec![vec![0.0; representation_rows[0].len()]; coefficients.cols];
    for output in 0..betas.len() {
        for field in 0..coefficients.cols {
            field_signatures[field][output] = betas[output][field];
        }
    }
    Ok(RepresentationFit {
        output_dim: representation_rows[0].len(),
        grouped_cv_r2: regression_r2(&actual, &predicted)?,
        cross_aperture_match_accuracy: 0.0,
        mean_matched_cosine: 0.0,
        min_match_margin: f64::NEG_INFINITY,
        field_signatures,
    })
}

fn normalized_or_zero(values: &[f64]) -> BrainResult<Vec<f64>> {
    let n = crate::linalg::norm(values)?;
    Ok(if n <= 1e-15 {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|value| value / n).collect()
    })
}

fn representation_centroid(
    field_index: usize,
    model: &DualSpaceModel<'_>,
    observations: &[DeltaObservation],
    rows: &[Vec<f64>],
    excluded_group: Option<&str>,
) -> BrainResult<Vec<f64>> {
    let mixture = model
        .skill_source_mixtures
        .get(field_index)
        .ok_or_else(|| BrainError::Integrity("dual_space_mixture_missing".into()))?;
    let support = source_support_indices(mixture)?;
    if support.is_empty() {
        return Err(BrainError::Invalid("dual_space_empty_field_support".into()));
    }
    let mut centroid = vec![0.0; rows[0].len()];
    let mut total = 0.0;
    for index in support {
        if excluded_group.is_some_and(|group| observations[index].independence_group == group) {
            continue;
        }
        let weight = observations[index].reliability.clamp(1e-4, 1.0);
        let sign = if mixture[index] >= 0.0 { 1.0 } else { -1.0 };
        total += weight;
        for dimension in 0..centroid.len() {
            centroid[dimension] += weight * sign * rows[index][dimension];
        }
    }
    if total <= 1e-15 {
        return Err(BrainError::Invalid(
            "dual_space_holdout_removes_all_field_support".into(),
        ));
    }
    for value in &mut centroid {
        *value /= total;
    }
    normalized_or_zero(&centroid)
}

fn expected_field_for_observation(
    observation_index: usize,
    model: &DualSpaceModel<'_>,
) -> BrainResult<(usize, f64)> {
    let mut best = None::<(usize, f64, f64)>;
    for field_index in 0..model.fields.len() {
        let coefficient = model.skill_source_mixtures[field_index][observation_index];
        let magnitude = coefficient.abs();
        if best.is_none_or(|(_, current, _)| magnitude > current) {
            best = Some((field_index, magnitude, coefficient.signum()));
        }
    }
    let (field_index, _magnitude, sign) = best
        .filter(|(_, magnitude, _)| *magnitude > 0.0)
        .ok_or_else(|| {
            BrainError::Invalid("dual_space_observation_without_field_support".into())
        })?;
    Ok((field_index, if sign < 0.0 { -1.0 } else { 1.0 }))
}

fn representation_recurrence(
    model: &DualSpaceModel<'_>,
    observations: &[DeltaObservation],
    rows: &[Vec<f64>],
    minimum_groups: usize,
) -> BrainResult<(f64, f64, f64, Vec<Vec<f64>>)> {
    if rows.len() != observations.len() || model.fields.is_empty() {
        return Err(BrainError::Invalid("dual_space_recurrence_shape".into()));
    }
    let folds = independence_group_folds(observations, minimum_groups)?;
    let mut correct = 0usize;
    let mut evaluated = 0usize;
    let mut cosine_sum = 0.0;
    let mut min_margin = f64::INFINITY;
    for fold in &folds {
        let holdout = &fold.holdout_group;
        let centroids = (0..model.fields.len())
            .map(|field_index| {
                representation_centroid(field_index, model, observations, rows, Some(holdout))
            })
            .collect::<BrainResult<Vec<_>>>()?;
        for (observation_index, observation) in observations.iter().enumerate() {
            if observation.independence_group != *holdout {
                continue;
            }
            let (expected, sign) = expected_field_for_observation(observation_index, model)?;
            let aligned = rows[observation_index]
                .iter()
                .map(|value| sign * value)
                .collect::<Vec<_>>();
            let normalized = normalized_or_zero(&aligned)?;
            let similarities = centroids
                .iter()
                .map(|centroid| crate::linalg::cosine(&normalized, centroid))
                .collect::<BrainResult<Vec<_>>>()?;
            let predicted = similarities
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    BrainError::Numerical("dual_space_no_representation_match".into())
                })?;
            let expected_similarity = similarities[expected];
            let competitor = similarities
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != expected)
                .map(|(_, value)| *value)
                .fold(f64::NEG_INFINITY, f64::max);
            min_margin = min_margin.min(expected_similarity - competitor);
            cosine_sum += expected_similarity;
            correct += usize::from(predicted == expected);
            evaluated += 1;
        }
    }
    if evaluated == 0 {
        return Err(BrainError::Invalid(
            "dual_space_no_recurrence_evaluations".into(),
        ));
    }
    let full_centroids = (0..model.fields.len())
        .map(|field_index| representation_centroid(field_index, model, observations, rows, None))
        .collect::<BrainResult<Vec<_>>>()?;
    Ok((
        correct as f64 / evaluated as f64,
        cosine_sum / evaluated as f64,
        min_margin,
        full_centroids,
    ))
}

pub fn analyze_dual_space(
    model: &DualSpaceModel<'_>,
    observations: &[DeltaObservation],
    representations: &[RepresentationObservation],
    ridge: f64,
    minimum_groups: usize,
    min_match_accuracy: f64,
    min_match_margin: f64,
) -> BrainResult<DualSpaceReport> {
    if observations.is_empty()
        || observations.len() != model.field_coefficients.len()
        || !min_match_accuracy.is_finite()
        || !(0.0..=1.0).contains(&min_match_accuracy)
        || !min_match_margin.is_finite()
        || min_match_margin < 0.0
    {
        return Err(BrainError::Invalid("dual_space_input_contract".into()));
    }
    let coefficients = coefficient_matrix(model, observations.len())?;
    let representation_rows = ordered_representation_rows(observations, representations)?;
    let representation = fit_representation_map(
        &coefficients,
        observations,
        &representation_rows,
        ridge,
        minimum_groups,
    )?;
    let (match_accuracy, mean_matched_cosine, min_match_margin_observed, recurrence_centroids) =
        representation_recurrence(model, observations, &representation_rows, minimum_groups)?;
    if representation.field_signatures.len() != model.fields.len()
        || recurrence_centroids.len() != model.fields.len()
    {
        return Err(BrainError::Integrity(
            "dual_space_signature_count_mismatch".into(),
        ));
    }
    let mut fields = Vec::with_capacity(model.fields.len());
    for (field_index, field) in model.fields.iter().enumerate() {
        let mixture = &model.skill_source_mixtures[field_index];
        let max_abs = mixture
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let tolerance = max_abs * f64::EPSILON.sqrt();
        let provenance_digests = mixture
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| coefficient.abs() > tolerance)
            .map(|(index, _)| observations[index].provenance_digest.clone())
            .collect::<Vec<_>>();
        fields.push(DualSpaceField {
            skill_id: field.skill_id.clone(),
            functional_signature: field.functional_signature.clone(),
            representation_signature: recurrence_centroids[field_index].clone(),
            provenance_digests,
            parameter_uncertainty: field.uncertainty,
        });
    }
    let representation_supported =
        match_accuracy >= min_match_accuracy && min_match_margin_observed > min_match_margin;
    Ok(DualSpaceReport {
        schema: "cerebro.tidex.dual_space/v3".into(),
        parameter_inverse_mode: model.parameter_inverse_mode,
        parameter_promotable: model.parameter_promotable,
        functional_cv_r2: model.functional_cv_r2,
        representation_cv_r2: representation.grouped_cv_r2,
        representation_match_accuracy: match_accuracy,
        representation_mean_matched_cosine: mean_matched_cosine,
        representation_min_match_margin: min_match_margin_observed,
        combined_cv_floor: model.functional_cv_r2.min(match_accuracy),
        representation_supported,
        dual_space_verified: model.parameter_promotable && representation_supported,
        fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::ConfounderValue;

    fn observation(index: usize, group: usize) -> DeltaObservation {
        DeltaObservation {
            observation_id: format!("o{index}"),
            from_checkpoint: "a".into(),
            to_checkpoint: format!("b{index}"),
            generation: 1,
            delta: vec![0.0],
            functional_response: vec![0.0],
            confounders: vec![ConfounderValue {
                name: "g".into(),
                value: group as f64,
            }],
            reliability: 1.0,
            independence_group: format!("g{group}"),
            experiment_lineage: Default::default(),
            dense_artifact: None,
            parameter_layout_sha256: None,
            representation_artifact: None,
            representation_protocol_sha256: None,
            provenance_digest: format!("{index:064x}"),
        }
    }

    #[test]
    fn representation_map_recovers_cross_aperture_linear_signatures() {
        let coefficients = Matrix::from_rows(&[
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.2],
            vec![0.2, 1.0],
            vec![0.8, -0.2],
            vec![-0.1, 0.9],
        ])
        .unwrap();
        let observations = (0..6).map(|i| observation(i, i / 2)).collect::<Vec<_>>();
        let rows = (0..6)
            .map(|i| {
                let a = coefficients.get(i, 0);
                let b = coefficients.get(i, 1);
                vec![2.0 * a - b, 0.5 * a + 3.0 * b]
            })
            .collect::<Vec<_>>();
        let fit = fit_representation_map(&coefficients, &observations, &rows, 1e-9, 3).unwrap();
        assert!(fit.grouped_cv_r2 > 0.999999, "{}", fit.grouped_cv_r2);
        assert!((fit.field_signatures[0][0] - 2.0).abs() < 1e-6);
        assert!((fit.field_signatures[1][1] - 3.0).abs() < 1e-6);
    }
}
