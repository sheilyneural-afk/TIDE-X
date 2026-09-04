use crate::error::{BrainError, BrainResult};
use crate::linalg::Matrix;
use crate::validation::validate_symmetric_psd;

pub fn second_order_interactions(
    dense_directions: &[Vec<f64>],
    curvature_diag: &[f64],
) -> BrainResult<Matrix> {
    if dense_directions.is_empty() {
        return Ok(Matrix::zeros(0, 0));
    }
    let dim = dense_directions[0].len();
    if dim == 0
        || curvature_diag.len() != dim
        || curvature_diag
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || dense_directions.iter().any(|direction| {
            direction.len() != dim || direction.iter().any(|value| !value.is_finite())
        })
    {
        return Err(BrainError::Invalid("interaction_shape_or_values".into()));
    }
    let mut matrix = Matrix::zeros(dense_directions.len(), dense_directions.len());
    for left in 0..dense_directions.len() {
        for right in left..dense_directions.len() {
            let value = (0..dim)
                .map(|parameter| {
                    dense_directions[left][parameter]
                        * curvature_diag[parameter]
                        * dense_directions[right][parameter]
                })
                .sum::<f64>();
            if !value.is_finite() {
                return Err(BrainError::Numerical(
                    "interaction_non_finite_quadratic_form".into(),
                ));
            }
            matrix.set(left, right, value);
            matrix.set(right, left, value);
        }
    }
    let _ = validate_symmetric_psd(&matrix, "interaction_matrix")?;
    Ok(matrix)
}

pub fn mvdr_weights(covariance: &Matrix, desired: &[f64], ridge: f64) -> BrainResult<Vec<f64>> {
    if covariance.rows != desired.len()
        || desired.is_empty()
        || desired.iter().any(|value| !value.is_finite())
        || !ridge.is_finite()
        || ridge < 0.0
    {
        return Err(BrainError::Invalid("mvdr_shape_or_values".into()));
    }
    let _ = validate_symmetric_psd(covariance, "mvdr_covariance")?;
    let inv = crate::linalg::inverse_with_ridge(covariance, ridge)?;
    let x = inv.matvec(desired)?;
    let denom = crate::linalg::dot(desired, &x)?;
    if denom.abs() < 1e-15 {
        return Err(BrainError::Numerical("mvdr_degenerate_constraint".into()));
    }
    Ok(x.into_iter().map(|value| value / denom).collect())
}
