use crate::contracts::DeltaObservation;
use crate::error::{BrainError, BrainResult};
use crate::identity::ObservationId;
use crate::linalg::{symmetric_eigen_jacobi_signed, Matrix};
use std::collections::BTreeSet;

/// Compatibility predicate for callers that only need a boolean.
///
/// `ObservationId` is the single authority for the persisted observation-ID
/// contract. Keeping this wrapper preserves existing call sites without
/// recreating a second validator that could drift from the typed boundary.
pub fn valid_observation_id(value: &str) -> bool {
    ObservationId::parse(value).is_ok()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupFold {
    pub holdout_group: String,
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

/// Indices whose coefficient is numerically material in a source mixture.
/// The tolerance is scale-relative and shared by memory, dense materialization,
/// dual-space reconstruction and structured geometry so those authorities do
/// not disagree about which observations support a field.
pub fn source_support_indices(mixture: &[f64]) -> BrainResult<Vec<usize>> {
    if mixture.is_empty() || mixture.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid("source_mixture_invalid".into()));
    }
    let max_abs = mixture
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if max_abs <= 0.0 {
        return Ok(Vec::new());
    }
    let tolerance = max_abs * f64::EPSILON.sqrt();
    Ok(mixture
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (value.abs() > tolerance).then_some(index))
        .collect())
}

pub fn independence_group_folds(
    observations: &[DeltaObservation],
    minimum_groups: usize,
) -> BrainResult<Vec<GroupFold>> {
    if observations.is_empty() || minimum_groups < 2 {
        return Err(BrainError::Invalid("grouped_cv_input_invalid".into()));
    }
    let groups = observations
        .iter()
        .map(|observation| observation.independence_group.clone())
        .collect::<BTreeSet<_>>();
    if groups.len() < minimum_groups {
        return Err(BrainError::Invalid(format!(
            "grouped_cv_requires_{minimum_groups}_independent_groups"
        )));
    }
    let mut folds = Vec::with_capacity(groups.len());
    for holdout_group in groups {
        let train = observations
            .iter()
            .enumerate()
            .filter_map(|(index, observation)| {
                (observation.independence_group != holdout_group).then_some(index)
            })
            .collect::<Vec<_>>();
        let test = observations
            .iter()
            .enumerate()
            .filter_map(|(index, observation)| {
                (observation.independence_group == holdout_group).then_some(index)
            })
            .collect::<Vec<_>>();
        if train.is_empty() || test.is_empty() {
            return Err(BrainError::Integrity("grouped_cv_empty_fold".into()));
        }
        folds.push(GroupFold {
            holdout_group,
            train,
            test,
        });
    }
    Ok(folds)
}

pub fn regression_r2(actual: &[Vec<f64>], predicted: &[Vec<f64>]) -> BrainResult<f64> {
    if actual.is_empty()
        || actual.len() != predicted.len()
        || actual[0].is_empty()
        || actual
            .iter()
            .any(|row| row.len() != actual[0].len() || row.iter().any(|value| !value.is_finite()))
        || predicted
            .iter()
            .any(|row| row.len() != actual[0].len() || row.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Invalid("regression_r2_shape".into()));
    }
    let dim = actual[0].len();
    let means = (0..dim)
        .map(|column| actual.iter().map(|row| row[column]).sum::<f64>() / actual.len() as f64)
        .collect::<Vec<_>>();
    let mut sse = 0.0;
    let mut sst = 0.0;
    for (actual_row, predicted_row) in actual.iter().zip(predicted) {
        for column in 0..dim {
            sse += (actual_row[column] - predicted_row[column]).powi(2);
            sst += (actual_row[column] - means[column]).powi(2);
        }
    }
    Ok(if sst <= 1e-18 { 0.0 } else { 1.0 - sse / sst })
}

pub fn validate_symmetric_psd(matrix: &Matrix, label: &str) -> BrainResult<Vec<(f64, Vec<f64>)>> {
    if matrix.rows != matrix.cols || matrix.rows == 0 || matrix.data.iter().any(|v| !v.is_finite())
    {
        return Err(BrainError::Invalid(format!("{label}_shape")));
    }
    let scale = matrix
        .data
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = f64::EPSILON.sqrt() * scale * matrix.rows as f64;
    for i in 0..matrix.rows {
        for j in 0..i {
            if (matrix.get(i, j) - matrix.get(j, i)).abs() > tolerance {
                return Err(BrainError::Invalid(format!("{label}_not_symmetric")));
            }
        }
    }
    let eigs = symmetric_eigen_jacobi_signed(matrix, 1e-12, matrix.rows * matrix.rows * 100)?;
    if eigs.iter().any(|(value, _)| *value < -tolerance) {
        return Err(BrainError::Invalid(format!("{label}_not_psd")));
    }
    Ok(eigs)
}

pub fn symmetric_psd_condition(matrix: &Matrix, label: &str) -> BrainResult<f64> {
    let eigs = validate_symmetric_psd(matrix, label)?;
    let largest = eigs
        .iter()
        .map(|(value, _)| value.max(0.0))
        .fold(0.0_f64, f64::max);
    if largest <= 1e-18 {
        return Ok(f64::INFINITY);
    }
    let tolerance = largest * f64::EPSILON.sqrt() * matrix.rows.max(1) as f64;
    let smallest = eigs
        .iter()
        .map(|(value, _)| value.max(0.0))
        .filter(|value| *value > tolerance)
        .fold(f64::INFINITY, f64::min);
    if !smallest.is_finite() {
        Ok(f64::INFINITY)
    } else {
        Ok(largest / smallest)
    }
}

pub fn effective_rank_from_spectrum(eigenvalues: &[f64]) -> BrainResult<f64> {
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid(
            "effective_rank_spectrum_nonfinite".into(),
        ));
    }
    let total = eigenvalues.iter().map(|value| value.max(0.0)).sum::<f64>();
    if total <= 1e-18 {
        return Ok(0.0);
    }
    let entropy = eigenvalues
        .iter()
        .filter_map(|value| {
            let probability = value.max(0.0) / total;
            (probability > 1e-15).then_some(-probability * probability.ln())
        })
        .sum::<f64>();
    Ok(entropy.exp())
}

pub fn choose_energy_rank(
    eigenvalues: &[f64],
    target_explained_variance: f64,
    max_rank: usize,
    minimum_rank: usize,
) -> BrainResult<usize> {
    if eigenvalues.is_empty()
        || eigenvalues.iter().any(|value| !value.is_finite())
        || !target_explained_variance.is_finite()
        || !(0.0..=1.0).contains(&target_explained_variance)
        || max_rank == 0
        || minimum_rank > max_rank
    {
        return Err(BrainError::Invalid("energy_rank_input_invalid".into()));
    }
    let total = eigenvalues.iter().map(|value| value.max(0.0)).sum::<f64>();
    if total <= 1e-18 {
        return Ok(minimum_rank.min(eigenvalues.len()));
    }
    let cap = max_rank.min(eigenvalues.len());
    let mut accumulated = 0.0;
    for (index, value) in eigenvalues.iter().enumerate().take(cap) {
        accumulated += value.max(0.0);
        if accumulated / total >= target_explained_variance {
            return Ok((index + 1).max(minimum_rank).min(cap));
        }
    }
    Ok(cap.max(minimum_rank.min(eigenvalues.len())))
}

#[cfg(test)]
mod tests {
    use super::valid_observation_id;
    use crate::identity::ObservationId;

    #[test]
    fn observation_id_predicate_delegates_to_the_canonical_type() {
        for value in [
            "observation-01",
            "observation.v2",
            ".hidden",
            "../escape",
            "observation/slash",
            "",
        ] {
            assert_eq!(
                valid_observation_id(value),
                ObservationId::parse(value).is_ok(),
                "validator drift for {value:?}",
            );
        }
    }
}
