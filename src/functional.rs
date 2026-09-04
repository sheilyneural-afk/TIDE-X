#![allow(clippy::needless_range_loop)]
use crate::contracts::{DeltaObservation, SkillField};
use crate::error::{BrainError, BrainResult};
use crate::linalg::{weighted_normal_solve, Matrix};
use crate::validation::independence_group_folds;

#[derive(Debug, Clone)]
pub struct FunctionalFit {
    pub cv_r2: f64,
    pub output_dim: usize,
    pub signatures: Vec<Vec<f64>>,
}

fn design(coeff: &Matrix, rows: &[usize]) -> BrainResult<Matrix> {
    let data = rows
        .iter()
        .map(|&r| {
            let mut x = vec![1.0];
            x.extend_from_slice(coeff.row(r));
            x
        })
        .collect::<Vec<_>>();
    Matrix::from_rows(&data)
}

pub fn fit_functional_map(
    coeff: &Matrix,
    observations: &[DeltaObservation],
    ridge: f64,
    minimum_groups: usize,
) -> BrainResult<FunctionalFit> {
    if coeff.rows != observations.len() || coeff.rows < 4 {
        return Err(BrainError::Invalid("functional_shape".into()));
    }
    let out_dim = observations
        .iter()
        .find(|o| !o.functional_response.is_empty())
        .map(|o| o.functional_response.len())
        .ok_or_else(|| BrainError::Invalid("functional_evidence_required".into()))?;
    if observations
        .iter()
        .any(|o| o.functional_response.len() != out_dim)
    {
        return Err(BrainError::Invalid(
            "functional_response_dimension_mismatch".into(),
        ));
    }
    let folds = independence_group_folds(observations, minimum_groups)?;
    let mut sse = 0.0;
    let mut sst = 0.0;
    for fold in &folds {
        let train = &fold.train;
        let test = &fold.test;
        let x = design(coeff, train)?;
        let weights = train
            .iter()
            .map(|&r| observations[r].reliability.clamp(1e-4, 1.0))
            .collect::<Vec<_>>();
        for out in 0..out_dim {
            let y = train
                .iter()
                .map(|&r| observations[r].functional_response[out])
                .collect::<Vec<_>>();
            let beta = weighted_normal_solve(&x, &y, &weights, ridge)?;
            let mean = y.iter().sum::<f64>() / y.len().max(1) as f64;
            for &r in test {
                let pred = beta[0]
                    + (0..coeff.cols)
                        .map(|k| beta[k + 1] * coeff.get(r, k))
                        .sum::<f64>();
                let actual = observations[r].functional_response[out];
                sse += (actual - pred).powi(2);
                sst += (actual - mean).powi(2);
            }
        }
    }
    let cv_r2 = if sst <= 1e-18 { 0.0 } else { 1.0 - sse / sst };
    let all = (0..coeff.rows).collect::<Vec<_>>();
    let x = design(coeff, &all)?;
    let weights = observations
        .iter()
        .map(|o| o.reliability.clamp(1e-4, 1.0))
        .collect::<Vec<_>>();
    let mut signatures = vec![vec![0.0; out_dim]; coeff.cols];
    for out in 0..out_dim {
        let y = observations
            .iter()
            .map(|o| o.functional_response[out])
            .collect::<Vec<_>>();
        let beta = weighted_normal_solve(&x, &y, &weights, ridge)?;
        for k in 0..coeff.cols {
            signatures[k][out] = beta[k + 1];
        }
    }
    Ok(FunctionalFit {
        cv_r2,
        output_dim: out_dim,
        signatures,
    })
}

pub fn attach_signatures(fields: &mut [SkillField], fit: &FunctionalFit) -> BrainResult<()> {
    if fields.len() != fit.signatures.len() {
        return Err(BrainError::Invalid("functional_signature_count".into()));
    }
    for (f, s) in fields.iter_mut().zip(&fit.signatures) {
        f.functional_signature = s.clone();
    }
    Ok(())
}
