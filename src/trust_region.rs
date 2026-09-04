use crate::error::{BrainError, BrainResult};
use crate::linalg::{dot, Matrix};
use crate::validation::validate_symmetric_psd;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrustRegionAllocationPolicy {
    /// Backward-compatible representation for historical records written
    /// before the policy field became explicit. It is never emitted by a new
    /// trust-region decision.
    #[default]
    #[serde(rename = "")]
    Unspecified,
    #[serde(rename = "uniform_quadratic_scaling/v1")]
    UniformQuadraticScalingV1,
    #[serde(rename = "causal_priority_contraction/v1")]
    CausalPriorityContractionV1,
}

impl TrustRegionAllocationPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unspecified => "",
            Self::UniformQuadraticScalingV1 => "uniform_quadratic_scaling/v1",
            Self::CausalPriorityContractionV1 => "causal_priority_contraction/v1",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrustRegionResult {
    pub proposed_coefficients: Vec<f64>,
    pub accepted_coefficients: Vec<f64>,
    pub proposed_quadratic_cost: f64,
    pub accepted_quadratic_cost: f64,
    pub max_quadratic_cost: f64,
    /// The lowest retained magnitude fraction across non-zero components.  It
    /// equals the historical uniform scale only for uniform contraction.
    pub scale: f64,
    pub constrained: bool,
    #[serde(default)]
    pub allocation_policy: TrustRegionAllocationPolicy,
    #[serde(default)]
    pub component_retention: Vec<f64>,
    /// Present only when conservative causal evidence selected the component
    /// contraction order.  Missing evidence is never replaced with a default.
    #[serde(default)]
    pub causal_priority_weights: Option<Vec<f64>>,
}

fn validate_quadratic_metric(matrix: &Matrix) -> BrainResult<()> {
    let _ = validate_symmetric_psd(matrix, "trust_region_metric")?;
    Ok(())
}

pub fn quadratic_cost(covariance: &Matrix, coefficients: &[f64]) -> BrainResult<f64> {
    validate_quadratic_metric(covariance)?;
    if covariance.rows != coefficients.len()
        || coefficients.is_empty()
        || coefficients.iter().any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid("trust_region_shape".into()));
    }
    let product = covariance.matvec(coefficients)?;
    let cost = dot(coefficients, &product)?;
    if cost < -f64::EPSILON.sqrt() {
        return Err(BrainError::Numerical(
            "trust_region_negative_psd_cost".into(),
        ));
    }
    Ok(cost.max(0.0))
}

pub fn apply_quadratic_trust_region(
    covariance: &Matrix,
    coefficients: &[f64],
    max_quadratic_cost: f64,
) -> BrainResult<TrustRegionResult> {
    if !max_quadratic_cost.is_finite() || max_quadratic_cost < 0.0 {
        return Err(BrainError::Invalid("trust_region_budget_invalid".into()));
    }
    let proposed = quadratic_cost(covariance, coefficients)?;
    let scale = if proposed <= max_quadratic_cost || proposed <= 1e-18 {
        1.0
    } else if max_quadratic_cost <= 0.0 {
        0.0
    } else {
        (max_quadratic_cost / proposed).sqrt().clamp(0.0, 1.0)
    };
    let accepted_coefficients = coefficients
        .iter()
        .map(|value| scale * value)
        .collect::<Vec<_>>();
    let accepted = quadratic_cost(covariance, &accepted_coefficients)?;
    Ok(TrustRegionResult {
        proposed_coefficients: coefficients.to_vec(),
        accepted_coefficients,
        proposed_quadratic_cost: proposed,
        accepted_quadratic_cost: accepted,
        max_quadratic_cost,
        scale,
        constrained: scale < 1.0,
        allocation_policy: TrustRegionAllocationPolicy::UniformQuadraticScalingV1,
        component_retention: vec![scale; coefficients.len()],
        causal_priority_weights: None,
    })
}

fn signed_magnitude_metric(covariance: &Matrix, coefficients: &[f64]) -> BrainResult<Matrix> {
    if covariance.rows != coefficients.len()
        || covariance.cols != coefficients.len()
        || coefficients.is_empty()
    {
        return Err(BrainError::Invalid("causal_trust_region_shape".into()));
    }
    let mut out = Matrix::zeros(covariance.rows, covariance.cols);
    for (row, left_coefficient) in coefficients.iter().enumerate() {
        let left_sign = if left_coefficient.is_sign_negative() {
            -1.0
        } else {
            1.0
        };
        for (col, right_coefficient) in coefficients.iter().enumerate() {
            let right_sign = if right_coefficient.is_sign_negative() {
                -1.0
            } else {
                1.0
            };
            out.set(row, col, covariance.get(row, col) * left_sign * right_sign);
        }
    }
    validate_quadratic_metric(&out)?;
    Ok(out)
}

fn causal_priority_candidate(
    metric: &Matrix,
    magnitudes: &[f64],
    priorities: &[f64],
) -> BrainResult<Option<usize>> {
    let gradient = metric.matvec(magnitudes)?;
    let tolerance = f64::EPSILON.sqrt()
        * (1.0
            + magnitudes.iter().map(|value| value.abs()).sum::<f64>()
            + gradient.iter().map(|value| value.abs()).sum::<f64>());
    let mut selected = None::<(usize, f64)>;
    for index in 0..magnitudes.len() {
        if magnitudes[index] <= tolerance || gradient[index] <= tolerance {
            continue;
        }
        // A smaller ratio sacrifices less conservatively verified causal utility
        // for each instantaneous unit of quadratic-risk relief.
        let ratio = priorities[index] / (2.0 * gradient[index]);
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(BrainError::Integrity(
                "causal_trust_region_priority_ratio_invalid".into(),
            ));
        }
        if selected.is_none_or(|(best_index, best_ratio)| {
            ratio < best_ratio - tolerance
                || ((ratio - best_ratio).abs() <= tolerance && index < best_index)
        }) {
            selected = Some((index, ratio));
        }
    }
    Ok(selected.map(|(index, _)| index))
}

/// Contract a proposed composition to a verified quadratic budget while
/// preserving the fields with the largest *lower-confidence* causal benefit.
/// The routine is intentionally not a generic optimiser: it executes the
/// deterministic authority rule used by TIDE-X, reducing the field with the
/// least causal utility per marginal curvature relief until the certified
/// budget is met.  If causal evidence cannot determine a lawful contraction,
/// it fails closed rather than reverting to uniform scaling.
pub fn apply_causal_priority_trust_region(
    covariance: &Matrix,
    coefficients: &[f64],
    max_quadratic_cost: f64,
    causal_priority_weights: &[f64],
) -> BrainResult<TrustRegionResult> {
    if !max_quadratic_cost.is_finite()
        || max_quadratic_cost < 0.0
        || coefficients.is_empty()
        || coefficients.iter().any(|value| !value.is_finite())
        || causal_priority_weights.len() != coefficients.len()
        || causal_priority_weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(BrainError::Invalid(
            "causal_trust_region_contract_invalid".into(),
        ));
    }
    validate_quadratic_metric(covariance)?;
    let proposed = quadratic_cost(covariance, coefficients)?;
    let tolerance = f64::EPSILON.sqrt() * (1.0 + proposed.abs() + max_quadratic_cost.abs());
    let signed_metric = signed_magnitude_metric(covariance, coefficients)?;
    let mut magnitudes = coefficients
        .iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let original_magnitudes = magnitudes.clone();
    let mut accepted = proposed;

    if accepted > max_quadratic_cost + tolerance {
        let max_iterations = coefficients.len().saturating_mul(128).max(128);
        for _ in 0..max_iterations {
            accepted = quadratic_cost(&signed_metric, &magnitudes)?;
            if accepted <= max_quadratic_cost + tolerance {
                break;
            }
            let index =
                causal_priority_candidate(&signed_metric, &magnitudes, causal_priority_weights)?
                    .ok_or_else(|| {
                        BrainError::Integrity("causal_trust_region_no_lawful_contraction".into())
                    })?;
            let gradient = signed_metric.matvec(&magnitudes)?;
            let diagonal = signed_metric.get(index, index);
            if !diagonal.is_finite() || diagonal < 0.0 || !gradient[index].is_finite() {
                return Err(BrainError::Integrity(
                    "causal_trust_region_curvature_invalid".into(),
                ));
            }
            // Along this coordinate the quadratic is convex.  First move to its
            // minimum; if that crosses the budget, bisection finds the boundary
            // without ever exceeding the proposal's magnitude.
            let best_decrement = if diagonal > tolerance {
                (gradient[index] / diagonal).clamp(0.0, magnitudes[index])
            } else {
                magnitudes[index]
            };
            if best_decrement <= tolerance {
                return Err(BrainError::Integrity(
                    "causal_trust_region_zero_relief".into(),
                ));
            }
            let mut best = magnitudes.clone();
            best[index] = (best[index] - best_decrement).max(0.0);
            let best_cost = quadratic_cost(&signed_metric, &best)?;
            if best_cost > accepted + tolerance {
                return Err(BrainError::Numerical(
                    "causal_trust_region_nonconvex_coordinate".into(),
                ));
            }
            if best_cost <= max_quadratic_cost + tolerance {
                let mut low = 0.0;
                let mut high = best_decrement;
                for _ in 0..96 {
                    let middle = (low + high) * 0.5;
                    let mut candidate = magnitudes.clone();
                    candidate[index] = (candidate[index] - middle).max(0.0);
                    if quadratic_cost(&signed_metric, &candidate)? > max_quadratic_cost {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                magnitudes[index] = (magnitudes[index] - high).max(0.0);
                break;
            }
            magnitudes = best;
        }
    }
    accepted = quadratic_cost(&signed_metric, &magnitudes)?;
    if accepted > max_quadratic_cost + tolerance {
        return Err(BrainError::Numerical(
            "causal_trust_region_budget_not_reached".into(),
        ));
    }
    let accepted_coefficients = coefficients
        .iter()
        .zip(&magnitudes)
        .map(|(coefficient, magnitude)| coefficient.signum() * magnitude)
        .collect::<Vec<_>>();
    let component_retention = original_magnitudes
        .iter()
        .zip(&magnitudes)
        .map(|(original, accepted)| {
            if *original <= f64::EPSILON {
                1.0
            } else {
                (accepted / original).clamp(0.0, 1.0)
            }
        })
        .collect::<Vec<_>>();
    let scale = component_retention.iter().copied().fold(1.0_f64, f64::min);
    Ok(TrustRegionResult {
        proposed_coefficients: coefficients.to_vec(),
        accepted_coefficients,
        proposed_quadratic_cost: proposed,
        accepted_quadratic_cost: accepted,
        max_quadratic_cost,
        scale,
        constrained: accepted < proposed - tolerance,
        allocation_policy: TrustRegionAllocationPolicy::CausalPriorityContractionV1,
        component_retention,
        causal_priority_weights: Some(causal_priority_weights.to_vec()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_region_rejects_indefinite_metric() {
        let mut covariance = Matrix::zeros(2, 2);
        covariance.set(0, 0, 1.0);
        covariance.set(1, 1, -0.5);
        assert!(apply_quadratic_trust_region(&covariance, &[1.0, 1.0], 1.0).is_err());
    }

    #[test]
    fn trust_region_scales_exactly_to_quadratic_budget() {
        let mut covariance = Matrix::zeros(2, 2);
        covariance.set(0, 0, 4.0);
        covariance.set(1, 1, 1.0);
        let result = apply_quadratic_trust_region(&covariance, &[1.0, 1.0], 1.25).unwrap();
        assert!(result.constrained);
        assert!((result.proposed_quadratic_cost - 5.0).abs() < 1e-12);
        assert!((result.accepted_quadratic_cost - 1.25).abs() < 1e-10);
        assert!((result.scale - 0.5).abs() < 1e-12);
        assert_eq!(
            result.allocation_policy,
            TrustRegionAllocationPolicy::UniformQuadraticScalingV1
        );
        assert_eq!(result.component_retention, vec![0.5, 0.5]);
        assert_eq!(result.causal_priority_weights, None);
    }

    #[test]
    fn causal_priority_preserves_high_value_field_before_uniform_scaling() {
        let mut covariance = Matrix::zeros(2, 2);
        covariance.set(0, 0, 1.0);
        covariance.set(1, 1, 4.0);
        let result =
            apply_causal_priority_trust_region(&covariance, &[1.0, 1.0], 2.0, &[10.0, 1.0])
                .unwrap();
        assert_eq!(
            result.allocation_policy,
            TrustRegionAllocationPolicy::CausalPriorityContractionV1
        );
        assert!((result.accepted_coefficients[0] - 1.0).abs() < 1e-10);
        assert!((result.accepted_coefficients[1] - 0.5).abs() < 1e-10);
        assert!((result.accepted_quadratic_cost - 2.0).abs() < 1e-9);
        assert!(result.component_retention[0] > result.component_retention[1]);
    }

    #[test]
    fn causal_priority_rejects_missing_or_nonpositive_evidence() {
        let covariance = Matrix::identity(2);
        assert!(apply_causal_priority_trust_region(&covariance, &[1.0, 1.0], 1.0, &[1.0]).is_err());
        assert!(
            apply_causal_priority_trust_region(&covariance, &[1.0, 1.0], 1.0, &[1.0, 0.0]).is_err()
        );
    }
}
