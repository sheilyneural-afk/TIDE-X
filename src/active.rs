use crate::contracts::ApertureCandidate;
use crate::error::{BrainError, BrainResult};
use crate::identity::ApertureId;
use crate::linalg::{dot, Matrix};
use crate::validation::validate_symmetric_psd;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApertureScore {
    pub aperture_id: ApertureId,
    pub information_gain: f64,
    pub objective: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AperturePlanStep {
    pub step: usize,
    pub aperture_id: ApertureId,
    pub information_gain: f64,
    pub objective: f64,
    pub posterior_trace_before: f64,
    pub posterior_trace_after: f64,
    pub trace_reduction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActiveAperturePlan {
    pub schema: String,
    pub initial_posterior_trace: f64,
    pub final_posterior_trace: f64,
    pub total_trace_reduction: f64,
    pub total_information_gain: f64,
    pub steps: Vec<AperturePlanStep>,
    pub unselected_aperture_ids: Vec<ApertureId>,
    pub stopped_on_nonpositive_objective: bool,
    pub final_posterior_covariance: Vec<Vec<f64>>,
}

fn validate_candidate(candidate: &ApertureCandidate, dimension: usize) -> BrainResult<()> {
    if candidate.sensing_vector.len() != dimension
        || candidate
            .sensing_vector
            .iter()
            .any(|value| !value.is_finite())
        || candidate
            .sensing_vector
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            <= 1e-30
        || candidate.noise_variance <= 0.0
        || !candidate.noise_variance.is_finite()
        || !candidate.cost.is_finite()
        || candidate.cost < 0.0
        || !candidate.risk.is_finite()
        || !(0.0..=1.0).contains(&candidate.risk)
    {
        return Err(BrainError::Invalid("active_candidate_invalid".into()));
    }
    Ok(())
}

fn validate_covariance(covariance: &Matrix) -> BrainResult<()> {
    let _ = validate_symmetric_psd(covariance, "active_covariance")?;
    Ok(())
}

fn trace(matrix: &Matrix) -> f64 {
    (0..matrix.rows).map(|index| matrix.get(index, index)).sum()
}

pub fn choose_active_aperture(
    candidates: &[ApertureCandidate],
    posterior_cov: &Matrix,
    cost_weight: f64,
    risk_weight: f64,
) -> BrainResult<ApertureScore> {
    validate_covariance(posterior_cov)?;
    if !cost_weight.is_finite()
        || cost_weight < 0.0
        || !risk_weight.is_finite()
        || risk_weight < 0.0
    {
        return Err(BrainError::Invalid(
            "active_objective_weights_invalid".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut best: Option<ApertureScore> = None;
    for candidate in candidates {
        validate_candidate(candidate, posterior_cov.cols)?;
        if !ids.insert(candidate.aperture_id.as_str()) {
            return Err(BrainError::Invalid("active_duplicate_aperture_id".into()));
        }
        let q = posterior_cov.matvec(&candidate.sensing_vector)?;
        let snr = dot(&candidate.sensing_vector, &q)?.max(0.0) / candidate.noise_variance;
        let information_gain = 0.5 * (1.0 + snr).ln();
        let objective =
            information_gain - cost_weight * candidate.cost - risk_weight * candidate.risk;
        let score = ApertureScore {
            aperture_id: candidate.aperture_id.clone(),
            information_gain,
            objective,
        };
        if best.as_ref().is_none_or(|current| {
            score.objective > current.objective
                || (score.objective == current.objective && score.aperture_id < current.aperture_id)
        }) {
            best = Some(score);
        }
    }
    best.ok_or_else(|| BrainError::Invalid("active_no_candidates".into()))
}

/// Exact rank-one Gaussian covariance update for a linear sensing aperture:
/// Σ' = Σ - Σ a aᵀ Σ / (σ² + aᵀ Σ a).
/// No outcome value is needed because expected posterior covariance depends on
/// the experiment design and observation noise, not on the realized sample.
pub fn update_posterior_covariance(
    posterior_cov: &Matrix,
    candidate: &ApertureCandidate,
) -> BrainResult<Matrix> {
    validate_covariance(posterior_cov)?;
    validate_candidate(candidate, posterior_cov.cols)?;
    let projected = posterior_cov.matvec(&candidate.sensing_vector)?;
    let predictive_variance = dot(&candidate.sensing_vector, &projected)?.max(0.0);
    let denominator = candidate.noise_variance + predictive_variance;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(BrainError::Numerical(
            "active_posterior_update_degenerate".into(),
        ));
    }
    let mut updated = posterior_cov.clone();
    for row in 0..updated.rows {
        for col in 0..updated.cols {
            let value = posterior_cov.get(row, col) - projected[row] * projected[col] / denominator;
            updated.set(row, col, value);
        }
    }
    // Restore exact symmetry against accumulated floating-point error.
    for row in 0..updated.rows {
        updated.set(row, row, updated.get(row, row).max(0.0));
        for col in 0..row {
            let mean = 0.5 * (updated.get(row, col) + updated.get(col, row));
            updated.set(row, col, mean);
            updated.set(col, row, mean);
        }
    }
    validate_covariance(&updated)?;
    Ok(updated)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GaussianAperturePosterior {
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
}

fn matrix_from_square_rows(rows: &[Vec<f64>]) -> BrainResult<Matrix> {
    let matrix = Matrix::from_rows(rows)?;
    if matrix.rows != matrix.cols || matrix.rows == 0 {
        return Err(BrainError::Invalid("active_posterior_rows_shape".into()));
    }
    validate_covariance(&matrix)?;
    Ok(matrix)
}

/// Assimilate the realized scalar result y = a^T x + ε after an aperture has
/// actually run. This is the exact Gaussian linear update of both posterior
/// mean and covariance. Planning uses covariance only; execution closes the
/// loop with this function once evidence arrives.
pub fn assimilate_aperture_result(
    posterior: &GaussianAperturePosterior,
    candidate: &ApertureCandidate,
    observed_value: f64,
) -> BrainResult<GaussianAperturePosterior> {
    if posterior.mean.is_empty()
        || posterior.mean.iter().any(|value| !value.is_finite())
        || !observed_value.is_finite()
    {
        return Err(BrainError::Invalid("active_posterior_mean_invalid".into()));
    }
    let covariance = matrix_from_square_rows(&posterior.covariance)?;
    if covariance.rows != posterior.mean.len() {
        return Err(BrainError::Invalid("active_posterior_mean_shape".into()));
    }
    validate_candidate(candidate, covariance.cols)?;
    let projected = covariance.matvec(&candidate.sensing_vector)?;
    let predictive_variance = dot(&candidate.sensing_vector, &projected)?.max(0.0);
    let denominator = candidate.noise_variance + predictive_variance;
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err(BrainError::Numerical(
            "active_posterior_assimilation_degenerate".into(),
        ));
    }
    let predicted_value = dot(&candidate.sensing_vector, &posterior.mean)?;
    let innovation = observed_value - predicted_value;
    let mean = posterior
        .mean
        .iter()
        .zip(&projected)
        .map(|(prior, gain_numerator)| prior + gain_numerator / denominator * innovation)
        .collect::<Vec<_>>();
    let updated_covariance = update_posterior_covariance(&covariance, candidate)?;
    Ok(GaussianAperturePosterior {
        mean,
        covariance: (0..updated_covariance.rows)
            .map(|row| updated_covariance.row_vec(row))
            .collect(),
    })
}

/// Sequential experiment design. Candidates are used at most once. After each
/// selected aperture the posterior covariance is updated, so the next choice
/// reflects information already expected from prior planned experiments.
pub fn plan_active_apertures(
    candidates: &[ApertureCandidate],
    posterior_cov: &Matrix,
    cost_weight: f64,
    risk_weight: f64,
    max_steps: usize,
    stop_on_nonpositive_objective: bool,
) -> BrainResult<ActiveAperturePlan> {
    validate_covariance(posterior_cov)?;
    if max_steps == 0 {
        return Err(BrainError::Invalid("active_plan_zero_steps".into()));
    }
    let mut ids = BTreeSet::new();
    for candidate in candidates {
        validate_candidate(candidate, posterior_cov.cols)?;
        if !ids.insert(candidate.aperture_id.as_str()) {
            return Err(BrainError::Invalid("active_duplicate_aperture_id".into()));
        }
    }
    if candidates.is_empty() {
        return Err(BrainError::Invalid("active_no_candidates".into()));
    }
    let initial_trace = trace(posterior_cov);
    let mut covariance = posterior_cov.clone();
    let mut remaining = candidates.to_vec();
    let mut steps = Vec::new();
    let mut total_information_gain = 0.0;
    let mut stopped_on_nonpositive_objective = false;

    while !remaining.is_empty() && steps.len() < max_steps {
        let best = choose_active_aperture(&remaining, &covariance, cost_weight, risk_weight)?;
        if stop_on_nonpositive_objective && best.objective <= 0.0 {
            stopped_on_nonpositive_objective = true;
            break;
        }
        let selected_index = remaining
            .iter()
            .position(|candidate| candidate.aperture_id == best.aperture_id)
            .ok_or_else(|| BrainError::Integrity("active_selected_candidate_missing".into()))?;
        let selected = remaining.remove(selected_index);
        let before = trace(&covariance);
        covariance = update_posterior_covariance(&covariance, &selected)?;
        let after = trace(&covariance);
        total_information_gain += best.information_gain;
        steps.push(AperturePlanStep {
            step: steps.len(),
            aperture_id: best.aperture_id,
            information_gain: best.information_gain,
            objective: best.objective,
            posterior_trace_before: before,
            posterior_trace_after: after,
            trace_reduction: (before - after).max(0.0),
        });
    }
    let final_trace = trace(&covariance);
    Ok(ActiveAperturePlan {
        schema: "cerebro.tidex.active_aperture_plan/v1".into(),
        initial_posterior_trace: initial_trace,
        final_posterior_trace: final_trace,
        total_trace_reduction: (initial_trace - final_trace).max(0.0),
        total_information_gain,
        steps,
        unselected_aperture_ids: remaining
            .into_iter()
            .map(|candidate| candidate.aperture_id)
            .collect(),
        stopped_on_nonpositive_objective,
        final_posterior_covariance: (0..covariance.rows)
            .map(|row| covariance.row_vec(row))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aperture_candidate_wire_rejects_paths_and_digests() {
        let candidate = serde_json::json!({
            "aperture_id": "../outside",
            "sensing_vector": [1.0],
            "noise_variance": 1.0,
            "cost": 0.0,
            "risk": 0.0
        });
        assert!(serde_json::from_value::<ApertureCandidate>(candidate).is_err());
        let digest = serde_json::json!({
            "aperture_id": "a".repeat(64),
            "sensing_vector": [1.0],
            "noise_variance": 1.0,
            "cost": 0.0,
            "risk": 0.0
        });
        assert!(serde_json::from_value::<ApertureCandidate>(digest).is_err());
    }

    #[test]
    fn active_aperture_rejects_indefinite_covariance() {
        let mut covariance = Matrix::zeros(2, 2);
        covariance.set(0, 0, 1.0);
        covariance.set(1, 1, -0.1);
        let candidate = ApertureCandidate {
            aperture_id: "x".into(),
            sensing_vector: vec![1.0, 0.0],
            noise_variance: 1.0,
            cost: 0.0,
            risk: 0.0,
        };
        assert!(choose_active_aperture(&[candidate], &covariance, 0.0, 0.0).is_err());
    }

    #[test]
    fn active_aperture_rejects_zero_sensing_and_breaks_ties_canonically() {
        let covariance = Matrix::identity(2);
        let zero = ApertureCandidate {
            aperture_id: "zero".into(),
            sensing_vector: vec![0.0, 0.0],
            noise_variance: 1.0,
            cost: 0.0,
            risk: 0.0,
        };
        assert!(choose_active_aperture(&[zero], &covariance, 0.0, 0.0).is_err());

        let make = |id: &str| ApertureCandidate {
            aperture_id: id.into(),
            sensing_vector: vec![1.0, 0.0],
            noise_variance: 1.0,
            cost: 0.0,
            risk: 0.0,
        };
        let forward =
            choose_active_aperture(&[make("zeta"), make("alpha")], &covariance, 0.0, 0.0).unwrap();
        let reverse =
            choose_active_aperture(&[make("alpha"), make("zeta")], &covariance, 0.0, 0.0).unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.aperture_id.as_str(), "alpha");
    }

    #[test]
    fn active_aperture_prefers_information_when_costs_match() {
        let covariance = Matrix::identity(2);
        let candidates = vec![
            ApertureCandidate {
                aperture_id: "weak".into(),
                sensing_vector: vec![0.1, 0.0],
                noise_variance: 1.0,
                cost: 0.1,
                risk: 0.1,
            },
            ApertureCandidate {
                aperture_id: "strong".into(),
                sensing_vector: vec![1.0, 1.0],
                noise_variance: 0.2,
                cost: 0.1,
                risk: 0.1,
            },
        ];
        let selected = choose_active_aperture(&candidates, &covariance, 0.1, 0.1).unwrap();
        assert_eq!(selected.aperture_id.as_str(), "strong");
    }

    #[test]
    fn posterior_update_reduces_uncertainty_along_sensed_axis() {
        let covariance = Matrix::identity(2);
        let candidate = ApertureCandidate {
            aperture_id: "x".into(),
            sensing_vector: vec![1.0, 0.0],
            noise_variance: 0.25,
            cost: 0.0,
            risk: 0.0,
        };
        let updated = update_posterior_covariance(&covariance, &candidate).unwrap();
        assert!(updated.get(0, 0) < covariance.get(0, 0));
        assert!((updated.get(1, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn realized_aperture_updates_mean_and_covariance() {
        let posterior = GaussianAperturePosterior {
            mean: vec![0.0, 0.0],
            covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let candidate = ApertureCandidate {
            aperture_id: "x".into(),
            sensing_vector: vec![1.0, 0.0],
            noise_variance: 0.25,
            cost: 0.0,
            risk: 0.0,
        };
        let updated = assimilate_aperture_result(&posterior, &candidate, 1.0).unwrap();
        assert!(updated.mean[0] > 0.0);
        assert!(updated.mean[1].abs() < 1e-12);
        assert!(updated.covariance[0][0] < 1.0);
        assert!((updated.covariance[1][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sequential_plan_changes_after_posterior_update() {
        let covariance = Matrix::identity(2);
        let candidates = vec![
            ApertureCandidate {
                aperture_id: "x".into(),
                sensing_vector: vec![1.0, 0.0],
                noise_variance: 0.01,
                cost: 0.0,
                risk: 0.0,
            },
            ApertureCandidate {
                aperture_id: "x-repeat-like".into(),
                sensing_vector: vec![0.95, 0.05],
                noise_variance: 0.01,
                cost: 0.0,
                risk: 0.0,
            },
            ApertureCandidate {
                aperture_id: "y".into(),
                sensing_vector: vec![0.0, 1.0],
                noise_variance: 0.01,
                cost: 0.0,
                risk: 0.0,
            },
        ];
        let plan = plan_active_apertures(&candidates, &covariance, 0.0, 0.0, 2, false).unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_ne!(plan.steps[0].aperture_id, plan.steps[1].aperture_id);
        assert!(plan.total_trace_reduction > 1.9);
        assert!(plan.final_posterior_trace < plan.initial_posterior_trace);
    }
}
