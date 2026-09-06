#![allow(clippy::needless_range_loop)]
use crate::contracts::DeltaObservation;
use crate::error::{BrainError, BrainResult};
use crate::linalg::{inverse_with_ridge, Matrix};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct ConfounderRemoval {
    pub residuals: Matrix,
    /// Linear operator T such that residuals = T * observed_deltas. Keeping
    /// this operator lets the sketch-space inverse be transported back to the
    /// exact source artifacts without losing confounder correction.
    pub source_transform: Matrix,
    pub design_names: Vec<String>,
    pub explained_fraction: f64,
}

fn raw_design(observations: &[DeltaObservation]) -> BrainResult<(Matrix, Vec<String>)> {
    if observations.is_empty() {
        return Err(BrainError::Invalid(
            "confounder_observations_required".into(),
        ));
    }
    let reference_names = observations[0]
        .confounders
        .iter()
        .map(|item| item.name.clone())
        .collect::<BTreeSet<_>>();
    if reference_names.len() != observations[0].confounders.len() {
        return Err(BrainError::Invalid("confounder_duplicate_name".into()));
    }
    let names = reference_names.into_iter().collect::<Vec<_>>();
    if names.is_empty() {
        return Ok((Matrix::zeros(observations.len(), 0), names));
    }
    let mut rows = Vec::with_capacity(observations.len());
    for (row_index, observation) in observations.iter().enumerate() {
        if observation.confounders.len() != names.len() {
            return Err(BrainError::Invalid(format!(
                "confounder_schema_missing_or_extra:{row_index}"
            )));
        }
        let map = observation
            .confounders
            .iter()
            .map(|item| (item.name.as_str(), item.value))
            .collect::<BTreeMap<_, _>>();
        if map.len() != names.len() || map.values().any(|value| !value.is_finite()) {
            return Err(BrainError::Invalid(format!(
                "confounder_schema_or_value_invalid:{row_index}"
            )));
        }
        let mut row = Vec::with_capacity(names.len());
        for name in &names {
            let value = map.get(name.as_str()).ok_or_else(|| {
                BrainError::Invalid(format!("confounder_missing:{row_index}:{name}"))
            })?;
            row.push(*value);
        }
        rows.push(row);
    }
    let mut x = Matrix::from_rows(&rows)?;
    // Design nuisance coordinates are frozen before outcomes are examined.
    // No intercept is fitted: a common persistent update may itself be skill.
    for column in 0..x.cols {
        let mean = (0..x.rows).map(|row| x.get(row, column)).sum::<f64>() / x.rows.max(1) as f64;
        let variance = (0..x.rows)
            .map(|row| (x.get(row, column) - mean).powi(2))
            .sum::<f64>()
            / x.rows.max(1) as f64;
        let std = variance.sqrt();
        if std <= 1e-12 {
            for row in 0..x.rows {
                x.set(row, column, 0.0);
            }
        } else {
            for row in 0..x.rows {
                x.set(row, column, (x.get(row, column) - mean) / std);
            }
        }
    }
    Ok((x, names))
}

pub fn remove_confounders(
    observations: &[DeltaObservation],
    ridge: f64,
) -> BrainResult<ConfounderRemoval> {
    if observations.len() < 2 {
        return Err(BrainError::Invalid("confounder_min_observations".into()));
    }
    if !ridge.is_finite() || ridge <= 0.0 {
        return Err(BrainError::Invalid("confounder_ridge_invalid".into()));
    }
    let dim = observations[0].delta.len();
    if dim == 0
        || observations
            .iter()
            .any(|observation| observation.delta.len() != dim)
    {
        return Err(BrainError::Invalid("delta_dimension_mismatch".into()));
    }
    if observations.iter().any(|observation| {
        !observation.reliability.is_finite()
            || observation.reliability <= 0.0
            || observation.reliability > 1.0
    }) {
        return Err(BrainError::Invalid("confounder_reliability_invalid".into()));
    }
    let d = Matrix::from_rows(
        &observations
            .iter()
            .map(|o| o.delta.clone())
            .collect::<Vec<_>>(),
    )?;
    let (raw, names) = raw_design(observations)?;
    if raw.cols == 0 {
        return Ok(ConfounderRemoval {
            residuals: d,
            source_transform: Matrix::identity(observations.len()),
            design_names: names,
            explained_fraction: 0.0,
        });
    }
    let x = raw;
    let weights = observations
        .iter()
        .map(|o| o.reliability)
        .collect::<Vec<_>>();
    let q = x.cols;
    let n = x.rows;
    let mut normal = Matrix::zeros(q, q);
    for r in 0..n {
        for i in 0..q {
            for j in 0..q {
                normal.data[i * q + j] += weights[r] * x.get(r, i) * x.get(r, j);
            }
        }
    }
    let inv = inverse_with_ridge(&normal, ridge)?;
    let mut transform = Matrix::identity(n);
    for r in 0..n {
        for sidx in 0..n {
            let mut projection = 0.0;
            for i in 0..q {
                for j in 0..q {
                    projection += x.get(r, i) * inv.get(i, j) * x.get(sidx, j) * weights[sidx];
                }
            }
            transform.data[r * n + sidx] -= projection;
        }
    }
    let residuals = transform.matmul(&d)?;
    let mut total_energy = 0.0;
    let mut residual_energy = 0.0;
    for r in 0..n {
        for p in 0..dim {
            total_energy += weights[r] * d.get(r, p).powi(2);
            residual_energy += weights[r] * residuals.get(r, p).powi(2);
        }
    }
    let explained = if total_energy <= 1e-18 {
        0.0
    } else {
        (1.0 - residual_energy / total_energy).clamp(0.0, 1.0)
    };
    Ok(ConfounderRemoval {
        residuals,
        source_transform: transform,
        design_names: names,
        explained_fraction: explained,
    })
}
