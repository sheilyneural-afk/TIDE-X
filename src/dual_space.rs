#![allow(clippy::needless_range_loop)]

use crate::contracts::{DeltaObservation, ReconstructionInverseMode, SkillField};
use crate::digest::ProvenanceDigest;
use crate::error::{BrainError, BrainResult};
use crate::identity::{ObservationId, SkillId};
use crate::linalg::{normalize, weighted_normal_solve, Matrix};
use crate::validation::{
    independence_group_folds, regression_r2, source_support_indices, validate_reliability,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepresentationObservation {
    pub observation_id: ObservationId,
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

/// Precommitted numerical and evidence gates for one dual-space analysis.
/// Keeping them together prevents call sites from accidentally swapping
/// several adjacent `f64` thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualSpaceAnalysisConfig {
    pub ridge: f64,
    pub minimum_independence_groups: usize,
    pub minimum_representation_cv_r2: f64,
    pub minimum_match_accuracy: f64,
    pub minimum_match_margin: f64,
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
    pub skill_id: SkillId,
    pub functional_signature: Vec<f64>,
    pub representation_signature: Vec<f64>,
    pub provenance_digests: Vec<ProvenanceDigest>,
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
    pub minimum_representation_cv_r2: f64,
    pub minimum_match_accuracy: f64,
    pub minimum_match_margin: f64,
    pub combined_cv_floor: f64,
    pub representation_supported: bool,
    pub dual_space_verified: bool,
    pub fields: Vec<DualSpaceField>,
}

fn coefficient_matrix(model: &DualSpaceModel<'_>, observation_count: usize) -> BrainResult<Matrix> {
    if model.fields.len() < 2
        || model.field_coefficients.len() != observation_count
        || model.skill_source_mixtures.len() != model.fields.len()
        || model
            .field_coefficients
            .iter()
            .any(|row| row.len() != model.fields.len())
        || model
            .skill_source_mixtures
            .iter()
            .any(|row| row.len() != observation_count || row.iter().any(|value| !value.is_finite()))
        || !model.functional_cv_r2.is_finite()
        || model.functional_cv_r2 > 1.0
        || model.fields.iter().any(|field| {
            !field.uncertainty.is_finite()
                || field.uncertainty < 0.0
                || field
                    .functional_signature
                    .iter()
                    .any(|value| !value.is_finite())
        })
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
    let mut by_id = BTreeMap::new();
    for representation in representations {
        if by_id
            .insert(
                representation.observation_id.as_str(),
                &representation.shift,
            )
            .is_some()
        {
            return Err(BrainError::Invalid(
                "dual_space_representation_id_invalid_or_duplicate".into(),
            ));
        }
    }
    let mut observation_ids = BTreeSet::new();
    let mut rows = Vec::with_capacity(observations.len());
    let mut dim = None;
    for observation in observations {
        if !observation_ids.insert(observation.observation_id.as_str()) {
            return Err(BrainError::Invalid(
                "dual_space_observation_id_invalid_or_duplicate".into(),
            ));
        }
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
        .map(|&index| validate_reliability(observations[index].reliability, "dual_space"))
        .collect::<BrainResult<Vec<_>>>()?;
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
    coefficients.validate("dual_space_coefficients")?;
    if coefficients.rows != observations.len()
        || coefficients.rows != representation_rows.len()
        || coefficients.cols == 0
        || representation_rows.is_empty()
        || representation_rows[0].is_empty()
        || representation_rows.iter().any(|row| {
            row.len() != representation_rows[0].len() || row.iter().any(|value| !value.is_finite())
        })
        || !ridge.is_finite()
        || ridge <= 0.0
        || observations
            .iter()
            .any(|observation| validate_reliability(observation.reliability, "dual_space").is_err())
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
        let weight = validate_reliability(observations[index].reliability, "dual_space")?;
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
    normalize(&centroid).map_err(|error| match error {
        BrainError::Numerical(_) => {
            BrainError::Numerical("dual_space_representation_centroid_degenerate".into())
        }
        other => other,
    })
}

fn expected_field_for_observation(
    observation_index: usize,
    model: &DualSpaceModel<'_>,
) -> BrainResult<(usize, f64)> {
    let mut ranked = Vec::with_capacity(model.fields.len());
    for field_index in 0..model.fields.len() {
        let coefficient = model.skill_source_mixtures[field_index][observation_index];
        ranked.push((field_index, coefficient.abs(), coefficient.signum()));
    }
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    let (field_index, magnitude, sign) = ranked
        .first()
        .copied()
        .filter(|(_, value, _)| *value > 0.0)
        .ok_or_else(|| {
            BrainError::Invalid("dual_space_observation_without_field_support".into())
        })?;
    let ambiguity_tolerance = magnitude * f64::EPSILON.sqrt();
    if ranked
        .get(1)
        .is_some_and(|(_, runner_up, _)| magnitude - runner_up <= ambiguity_tolerance)
    {
        return Err(BrainError::Invalid(
            "dual_space_observation_field_assignment_ambiguous".into(),
        ));
    }
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
            let normalized = normalize(&aligned).map_err(|error| match error {
                BrainError::Numerical(_) => {
                    BrainError::Numerical("dual_space_representation_shift_degenerate".into())
                }
                other => other,
            })?;
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
    config: DualSpaceAnalysisConfig,
) -> BrainResult<DualSpaceReport> {
    if observations.is_empty()
        || observations.len() != model.field_coefficients.len()
        || !config.ridge.is_finite()
        || config.ridge < 0.0
        || config.minimum_independence_groups < 2
        || !config.minimum_representation_cv_r2.is_finite()
        || config.minimum_representation_cv_r2 > 1.0
        || !config.minimum_match_accuracy.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_match_accuracy)
        || !config.minimum_match_margin.is_finite()
        || config.minimum_match_margin < 0.0
    {
        return Err(BrainError::Invalid("dual_space_input_contract".into()));
    }
    let coefficients = coefficient_matrix(model, observations.len())?;
    let representation_rows = ordered_representation_rows(observations, representations)?;
    let representation = fit_representation_map(
        &coefficients,
        observations,
        &representation_rows,
        config.ridge,
        config.minimum_independence_groups,
    )?;
    let (match_accuracy, mean_matched_cosine, min_match_margin_observed, recurrence_centroids) =
        representation_recurrence(
            model,
            observations,
            &representation_rows,
            config.minimum_independence_groups,
        )?;
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
    let representation_supported = representation.grouped_cv_r2
        >= config.minimum_representation_cv_r2
        && match_accuracy >= config.minimum_match_accuracy
        && min_match_margin_observed > config.minimum_match_margin;
    Ok(DualSpaceReport {
        schema: "cerebro.tidex.dual_space/v4".into(),
        parameter_inverse_mode: model.parameter_inverse_mode,
        parameter_promotable: model.parameter_promotable,
        functional_cv_r2: model.functional_cv_r2,
        representation_cv_r2: representation.grouped_cv_r2,
        representation_match_accuracy: match_accuracy,
        representation_mean_matched_cosine: mean_matched_cosine,
        representation_min_match_margin: min_match_margin_observed,
        minimum_representation_cv_r2: config.minimum_representation_cv_r2,
        minimum_match_accuracy: config.minimum_match_accuracy,
        minimum_match_margin: config.minimum_match_margin,
        combined_cv_floor: model
            .functional_cv_r2
            .min(representation.grouped_cv_r2)
            .min(match_accuracy),
        representation_supported,
        dual_space_verified: model.parameter_promotable && representation_supported,
        fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::ConfounderValue;
    use crate::digest::Sha256Digest;

    fn observation(index: usize, group: usize) -> DeltaObservation {
        DeltaObservation {
            observation_id: ObservationId::parse(format!("o{index}")).unwrap(),
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
            provenance_digest: ProvenanceDigest::from(
                Sha256Digest::parse(format!("{index:064x}")).unwrap(),
            ),
        }
    }

    fn field(id: &str) -> SkillField {
        SkillField {
            skill_id: SkillId::parse(id).unwrap(),
            reconstruction_id: crate::identity::ReconstructionId::parse(format!(
                "reconstruction-{id}"
            ))
            .unwrap(),
            lineage_id: crate::identity::LineageId::parse(format!("lineage-{id}")).unwrap(),
            generation_created: 1,
            direction: vec![1.0, 0.0],
            structured_geometry: None,
            dense_materialization: None,
            parameter_layout_sha256: None,
            representation_signature: Vec::new(),
            singular_value: 1.0,
            explained_variance: 0.5,
            persistence: 1.0,
            coherence: 1.0,
            uncertainty: 0.1,
            evidence_support_digests: Vec::new(),
            support: 3,
            functional_signature: vec![1.0],
            parent_skill_ids: Vec::new(),
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

        let mut invalid = observations.clone();
        invalid[0].reliability = 0.0;
        assert!(fit_representation_map(&coefficients, &invalid, &rows, 1e-9, 3).is_err());
        assert!(fit_representation_map(&coefficients, &observations, &rows, 0.0, 3).is_err());
    }

    #[test]
    fn directional_matching_cannot_hide_failed_cross_aperture_generalization() {
        let observations = (0..6).map(|i| observation(i, i / 2)).collect::<Vec<_>>();
        let field_coefficients = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let skill_source_mixtures = vec![
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ];
        let fields = vec![field("first"), field("second")];
        let model = DualSpaceModel {
            fields: &fields,
            field_coefficients: &field_coefficients,
            skill_source_mixtures: &skill_source_mixtures,
            parameter_inverse_mode: ReconstructionInverseMode::Persistent,
            parameter_promotable: true,
            functional_cv_r2: 1.0,
        };
        let magnitudes = [1.0, 10.0, 100.0];
        let representations = observations
            .iter()
            .enumerate()
            .map(|(index, observation)| RepresentationObservation {
                observation_id: observation.observation_id.clone(),
                shift: if index % 2 == 0 {
                    vec![magnitudes[index / 2], 0.0]
                } else {
                    vec![0.0, magnitudes[index / 2]]
                },
            })
            .collect::<Vec<_>>();
        let report = analyze_dual_space(
            &model,
            &observations,
            &representations,
            DualSpaceAnalysisConfig {
                ridge: 1e-9,
                minimum_independence_groups: 3,
                minimum_representation_cv_r2: 0.35,
                minimum_match_accuracy: 1.0,
                minimum_match_margin: 0.0,
            },
        )
        .unwrap();
        assert_eq!(report.representation_match_accuracy, 1.0);
        assert!(report.representation_min_match_margin > 0.99);
        assert!(report.representation_cv_r2 < 0.35);
        assert!(!report.representation_supported);
        assert!(!report.dual_space_verified);
    }
}
