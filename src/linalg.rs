#![allow(clippy::needless_range_loop)]
use crate::error::{BrainError, BrainResult};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}
impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn from_rows(rows: &[Vec<f64>]) -> BrainResult<Self> {
        if rows.is_empty() {
            return Ok(Self::zeros(0, 0));
        }
        let cols = rows[0].len();
        if cols == 0
            || rows
                .iter()
                .any(|r| r.len() != cols || r.iter().any(|v| !v.is_finite()))
        {
            return Err(BrainError::Invalid("matrix_rows_invalid".into()));
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data: rows.iter().flatten().copied().collect(),
        })
    }
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    pub fn row_vec(&self, r: usize) -> Vec<f64> {
        self.row(r).to_vec()
    }
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }
    pub fn matmul(&self, other: &Self) -> BrainResult<Self> {
        if self.cols != other.rows {
            return Err(BrainError::Invalid("matmul_shape".into()));
        }
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out.data[i * out.cols + j] += a * other.get(k, j);
                }
            }
        }
        Ok(out)
    }
    pub fn matvec(&self, v: &[f64]) -> BrainResult<Vec<f64>> {
        if self.cols != v.len() {
            return Err(BrainError::Invalid("matvec_shape".into()));
        }
        (0..self.rows)
            .map(|row| dot(self.row(row), v))
            .collect::<BrainResult<Vec<_>>>()
    }
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> BrainResult<f64> {
    if a.len() != b.len() || a.iter().chain(b).any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid("dot_shape_or_value".into()));
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}
pub fn norm(a: &[f64]) -> BrainResult<f64> {
    if a.is_empty() || a.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid("norm_input_invalid".into()));
    }
    Ok(dot(a, a)?.sqrt())
}
pub fn normalize(a: &[f64]) -> BrainResult<Vec<f64>> {
    let n = norm(a)?.max(1e-15);
    Ok(a.iter().map(|v| v / n).collect())
}
pub fn cosine(a: &[f64], b: &[f64]) -> BrainResult<f64> {
    Ok(dot(a, b)? / (norm(a)? * norm(b)?).max(1e-15))
}
pub fn sub(a: &[f64], b: &[f64]) -> BrainResult<Vec<f64>> {
    if a.len() != b.len() || a.iter().chain(b).any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid("sub_shape_or_value".into()));
    }
    Ok(a.iter().zip(b).map(|(x, y)| x - y).collect())
}
pub fn add_scaled(a: &mut [f64], b: &[f64], s: f64) -> BrainResult<()> {
    if a.len() != b.len() || !s.is_finite() || a.iter().chain(b).any(|value| !value.is_finite()) {
        return Err(BrainError::Invalid("add_scaled_shape_or_value".into()));
    }
    for (x, y) in a.iter_mut().zip(b) {
        *x += s * y;
    }
    Ok(())
}

pub fn solve(mut a: Matrix, mut b: Vec<f64>) -> BrainResult<Vec<f64>> {
    if a.rows != a.cols || a.rows != b.len() {
        return Err(BrainError::Invalid("linear_solve_shape".into()));
    }
    let n = a.rows;
    for k in 0..n {
        let mut pivot = k;
        let mut best = a.get(k, k).abs();
        for r in k + 1..n {
            let v = a.get(r, k).abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-12 {
            return Err(BrainError::Numerical("singular_system".into()));
        }
        if pivot != k {
            for c in k..n {
                let x = a.get(k, c);
                a.set(k, c, a.get(pivot, c));
                a.set(pivot, c, x);
            }
            b.swap(k, pivot);
        }
        let diag = a.get(k, k);
        for c in k..n {
            a.set(k, c, a.get(k, c) / diag);
        }
        b[k] /= diag;
        for r in 0..n {
            if r == k {
                continue;
            }
            let f = a.get(r, k);
            if f.abs() < 1e-18 {
                continue;
            }
            for c in k..n {
                a.set(r, c, a.get(r, c) - f * a.get(k, c));
            }
            b[r] -= f * b[k];
        }
    }
    Ok(b)
}

pub fn inverse_with_ridge(a: &Matrix, ridge: f64) -> BrainResult<Matrix> {
    if a.rows != a.cols {
        return Err(BrainError::Invalid("inverse_shape".into()));
    }
    let n = a.rows;
    let mut base = a.clone();
    for i in 0..n {
        base.data[i * n + i] += ridge;
    }
    let mut out = Matrix::zeros(n, n);
    for c in 0..n {
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        let x = solve(base.clone(), e)?;
        for r in 0..n {
            out.set(r, c, x[r]);
        }
    }
    Ok(out)
}

pub fn weighted_normal_solve(
    x: &Matrix,
    y: &[f64],
    weights: &[f64],
    ridge: f64,
) -> BrainResult<Vec<f64>> {
    if x.rows != y.len() || x.rows != weights.len() {
        return Err(BrainError::Invalid("weighted_ls_shape".into()));
    }
    let mut a = Matrix::zeros(x.cols, x.cols);
    let mut b = vec![0.0; x.cols];
    for r in 0..x.rows {
        let w = weights[r].max(0.0);
        for i in 0..x.cols {
            let xi = x.get(r, i);
            b[i] += w * xi * y[r];
            for j in 0..x.cols {
                a.data[i * x.cols + j] += w * xi * x.get(r, j);
            }
        }
    }
    for i in 0..x.cols {
        a.data[i * x.cols + i] += ridge;
    }
    solve(a, b)
}

pub fn symmetric_top_eigen(
    a: &Matrix,
    k: usize,
    iterations: usize,
) -> BrainResult<Vec<(f64, Vec<f64>)>> {
    if a.rows != a.cols {
        return Err(BrainError::Invalid("eigen_shape".into()));
    }
    let n = a.rows;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut out = Vec::new();
    for comp in 0..k.min(n) {
        let mut v = (0..n)
            .map(|i| (((i + 1) * (comp + 3)) as f64 * 0.731).sin() + 0.17)
            .collect::<Vec<_>>();
        v = normalize(&v)?;
        for _ in 0..iterations {
            let mut w = a.matvec(&v)?;
            for q in &basis {
                let p = dot(&w, q)?;
                add_scaled(&mut w, q, -p)?;
            }
            let wn = norm(&w)?;
            if wn < 1e-12 {
                break;
            }
            for x in &mut w {
                *x /= wn;
            }
            v = w;
        }
        let av = a.matvec(&v)?;
        let lambda = dot(&v, &av)?.max(0.0);
        if lambda < 1e-12 {
            break;
        }
        basis.push(v.clone());
        out.push((lambda, v));
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(out)
}

pub fn weighted_row_gram(d: &Matrix, weights: &[f64]) -> BrainResult<Matrix> {
    if d.rows != weights.len() {
        return Err(BrainError::Invalid("gram_weight_shape".into()));
    }
    let mut g = Matrix::zeros(d.rows, d.rows);
    for i in 0..d.rows {
        for j in i..d.rows {
            let v =
                weights[i].max(0.0).sqrt() * weights[j].max(0.0).sqrt() * dot(d.row(i), d.row(j))?;
            g.set(i, j, v);
            g.set(j, i, v);
        }
    }
    Ok(g)
}

pub fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    }
}

fn symmetric_eigen_jacobi_raw(
    a: &Matrix,
    tolerance: f64,
    max_rotations: usize,
) -> BrainResult<Vec<(f64, Vec<f64>)>> {
    if a.rows != a.cols {
        return Err(BrainError::Invalid("jacobi_eigen_shape".into()));
    }
    let n = a.rows;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut d = a.clone();
    let mut v = Matrix::identity(n);
    for _ in 0..max_rotations.max(n * n * 8) {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_off = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                let x = d.get(i, j).abs();
                if x > max_off {
                    max_off = x;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off <= tolerance {
            break;
        }
        let app = d.get(p, p);
        let aqq = d.get(q, q);
        let apq = d.get(p, q);
        let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = phi.cos();
        let s = phi.sin();
        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let dkp = d.get(k, p);
            let dkq = d.get(k, q);
            let np = c * dkp - s * dkq;
            let nq = s * dkp + c * dkq;
            d.set(k, p, np);
            d.set(p, k, np);
            d.set(k, q, nq);
            d.set(q, k, nq);
        }
        let new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        let new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        d.set(p, p, new_pp);
        d.set(q, q, new_qq);
        d.set(p, q, 0.0);
        d.set(q, p, 0.0);
        for k in 0..n {
            let vkp = v.get(k, p);
            let vkq = v.get(k, q);
            v.set(k, p, c * vkp - s * vkq);
            v.set(k, q, s * vkp + c * vkq);
        }
    }
    let mut out = (0..n)
        .map(|i| {
            let vec = (0..n).map(|r| v.get(r, i)).collect::<Vec<_>>();
            Ok((d.get(i, i), normalize(&vec)?))
        })
        .collect::<BrainResult<Vec<_>>>()?;
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(out)
}

/// Signed eigen-decomposition for symmetric matrices. Unlike the historical
/// energy helper, this preserves negative and near-zero eigenvalues so callers
/// can validate positive semidefiniteness instead of silently truncating
/// dangerous negative curvature.
pub fn symmetric_eigen_jacobi_signed(
    a: &Matrix,
    tolerance: f64,
    max_rotations: usize,
) -> BrainResult<Vec<(f64, Vec<f64>)>> {
    symmetric_eigen_jacobi_raw(a, tolerance, max_rotations)
}

/// Energy-oriented symmetric eigendecomposition. Negative eigenvalues are
/// intentionally discarded because Gram/energy callers require a PSD spectrum.
pub fn symmetric_eigen_jacobi(
    a: &Matrix,
    tolerance: f64,
    max_rotations: usize,
) -> BrainResult<Vec<(f64, Vec<f64>)>> {
    let mut out = symmetric_eigen_jacobi_raw(a, tolerance, max_rotations)?
        .into_iter()
        .map(|(value, vector)| (value.max(0.0), vector))
        .filter(|(value, _)| *value > 1e-14)
        .collect::<Vec<_>>();
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_eigensolver_preserves_negative_eigenvalues() {
        let mut matrix = Matrix::zeros(2, 2);
        matrix.set(0, 0, 2.0);
        matrix.set(1, 1, -0.5);
        let signed = symmetric_eigen_jacobi_signed(&matrix, 1e-12, 100).unwrap();
        assert!(signed.iter().any(|(value, _)| *value < -0.49));
        let energy = symmetric_eigen_jacobi(&matrix, 1e-12, 100).unwrap();
        assert!(energy.iter().all(|(value, _)| *value >= 0.0));
    }
}
