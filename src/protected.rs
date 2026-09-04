#![allow(clippy::needless_range_loop)]
use crate::contracts::ProtectedCortex;
use crate::error::{BrainError, BrainResult};
use crate::linalg::{norm, symmetric_eigen_jacobi_signed, Matrix};

#[derive(Debug, Clone)]
pub struct ProtectionResult {
    pub projected: Vec<f64>,
    pub damage_ratio: f64,
    pub allowed: bool,
    pub removed_energy: f64,
    pub protected_rank: usize,
    pub max_weighted_residual: f64,
}

fn wdot(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .zip(w)
        .map(|((x, y), z)| x * y * z.max(0.0))
        .sum()
}

pub fn project_to_safe_subspace(
    delta: &[f64],
    cortex: &ProtectedCortex,
) -> BrainResult<ProtectionResult> {
    if delta.is_empty()
        || delta.iter().any(|value| !value.is_finite())
        || cortex.parameter_importance.len() != delta.len()
        || cortex
            .parameter_importance
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || !cortex.max_damage_ratio.is_finite()
        || cortex.max_damage_ratio < 0.0
        || cortex.directions.iter().any(|direction| {
            direction.direction.len() != delta.len()
                || direction.direction.iter().any(|value| !value.is_finite())
                || !direction.importance.is_finite()
                || !(0.0..=1.0).contains(&direction.importance)
        })
    {
        return Err(BrainError::Invalid("protected_shape_or_values".into()));
    }

    // The Fisher brake is applied BEFORE hard projection. Applying it after the
    // projection can re-introduce a component along a protected direction.
    let mut braked = delta
        .iter()
        .zip(&cortex.parameter_importance)
        .map(|(value, importance)| value / (1.0 + importance.max(0.0)))
        .collect::<Vec<_>>();

    let active = cortex
        .directions
        .iter()
        .filter(|direction| direction.importance > 0.0)
        .collect::<Vec<_>>();
    let mut protected_rank = 0usize;
    let mut max_weighted_residual = 0.0f64;
    if !active.is_empty() {
        let m = active.len();
        let mut gram = Matrix::zeros(m, m);
        for i in 0..m {
            for j in i..m {
                let value = wdot(
                    &active[i].direction,
                    &active[j].direction,
                    &cortex.parameter_importance,
                );
                gram.set(i, j, value);
                gram.set(j, i, value);
            }
        }
        let eigs = symmetric_eigen_jacobi_signed(&gram, 1e-12, m * m * 100)?;
        let scale = eigs
            .iter()
            .map(|(value, _)| value.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let tolerance = f64::EPSILON.sqrt() * scale * m.max(1) as f64;
        if eigs.iter().any(|(value, _)| *value < -tolerance) {
            return Err(BrainError::Numerical("protected_gram_not_psd".into()));
        }
        let mut pseudoinverse = Matrix::zeros(m, m);
        for (eigenvalue, eigenvector) in &eigs {
            if *eigenvalue <= tolerance {
                continue;
            }
            protected_rank += 1;
            for i in 0..m {
                for j in 0..m {
                    let updated =
                        pseudoinverse.get(i, j) + eigenvector[i] * eigenvector[j] / *eigenvalue;
                    pseudoinverse.set(i, j, updated);
                }
            }
        }
        if protected_rank == 0 {
            return Err(BrainError::Numerical(
                "protected_directions_unobservable_in_metric".into(),
            ));
        }
        let rhs = active
            .iter()
            .map(|direction| wdot(&braked, &direction.direction, &cortex.parameter_importance))
            .collect::<Vec<_>>();
        let coefficients = pseudoinverse.matvec(&rhs)?;
        for (coefficient, direction) in coefficients.iter().zip(&active) {
            for parameter in 0..braked.len() {
                braked[parameter] -= coefficient * direction.direction[parameter];
            }
        }

        let reference_energy = wdot(delta, delta, &cortex.parameter_importance)
            .sqrt()
            .max(1e-15);
        for direction in &active {
            let direction_energy = wdot(
                &direction.direction,
                &direction.direction,
                &cortex.parameter_importance,
            )
            .sqrt()
            .max(1e-15);
            let residual = wdot(&braked, &direction.direction, &cortex.parameter_importance).abs()
                / (reference_energy * direction_energy);
            max_weighted_residual = max_weighted_residual.max(residual);
        }
        let residual_tolerance = f64::EPSILON.sqrt() * (m.max(1) as f64).sqrt() * 16.0;
        if max_weighted_residual > residual_tolerance {
            return Err(BrainError::Numerical(format!(
                "protected_joint_projection_residual:{max_weighted_residual:.6e}"
            )));
        }
    }

    let removed = delta
        .iter()
        .zip(&braked)
        .map(|(before, after)| (before - after).powi(2))
        .sum::<f64>();
    let damage = removed.sqrt() / norm(delta)?.max(1e-15);
    Ok(ProtectionResult {
        projected: braked,
        damage_ratio: damage,
        allowed: damage <= cortex.max_damage_ratio,
        removed_energy: removed,
        protected_rank,
        max_weighted_residual,
    })
}
