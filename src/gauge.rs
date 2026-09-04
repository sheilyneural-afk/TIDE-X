#![allow(clippy::needless_range_loop)]
use crate::error::{BrainError, BrainResult};
use crate::linalg::cosine;

#[derive(Debug, Clone)]
pub struct BasisAlignment {
    pub assignment: Vec<Option<usize>>,
    pub signs: Vec<f64>,
    pub mean_abs_cosine: f64,
}

// Hungarian assignment minimizes cost. We pad to a square matrix and use
// 1-|cosine| so equivalent directions with sign ambiguity align together.
pub fn align_bases(reference: &[Vec<f64>], candidate: &[Vec<f64>]) -> BrainResult<BasisAlignment> {
    if reference.is_empty() || candidate.is_empty() {
        return Err(BrainError::Invalid("gauge_empty_basis".into()));
    }
    let dim = reference[0].len();
    if dim == 0 || reference.iter().chain(candidate).any(|v| v.len() != dim) {
        return Err(BrainError::Invalid("gauge_dimension_mismatch".into()));
    }
    let n = reference.len().max(candidate.len());
    let mut cost = vec![vec![1.0; n]; n];
    for i in 0..reference.len() {
        for j in 0..candidate.len() {
            cost[i][j] = 1.0 - cosine(&reference[i], &candidate[j])?.abs().clamp(0.0, 1.0);
        }
    }
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut row_to_col = vec![None; reference.len()];
    for j in 1..=n {
        let i = p[j];
        if i > 0 && i <= reference.len() && j <= candidate.len() {
            row_to_col[i - 1] = Some(j - 1);
        }
    }
    let mut signs = vec![1.0; reference.len()];
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..reference.len() {
        if let Some(j) = row_to_col[i] {
            let c = cosine(&reference[i], &candidate[j])?;
            signs[i] = if c >= 0.0 { 1.0 } else { -1.0 };
            sum += c.abs();
            count += 1;
        }
    }
    Ok(BasisAlignment {
        assignment: row_to_col,
        signs,
        mean_abs_cosine: sum / count.max(1) as f64,
    })
}
