#![allow(clippy::needless_range_loop)]

use crate::error::{BrainError, BrainResult};
use crate::linalg::{cosine, dot, norm, normalize, solve, weighted_normal_solve, Matrix};

#[derive(Debug, Clone, PartialEq)]
pub struct TransportMap {
    pub source_dim: usize,
    pub target_dim: usize,
    pub weights: Matrix,
    pub bias: Vec<f64>,
    pub training_rms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedTransportMap {
    pub schema: String,
    pub map: TransportMap,
    pub anchor_count: usize,
    pub loo_cv_r2: f64,
    pub loo_cv_rms: f64,
    pub mean_loo_cosine: f64,
    pub min_loo_cosine: f64,
    pub resolved: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionalTransplantMap {
    pub schema: String,
    pub functional_dim: usize,
    pub target_dim: usize,
    pub target_decoder: TransportMap,
    pub anchor_count: usize,
    pub loo_cv_r2: f64,
    pub mean_loo_cosine: f64,
    pub min_loo_cosine: f64,
    pub resolved: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransplantedCapability {
    pub target_vector: Vec<f64>,
    pub source_functional_signature: Vec<f64>,
    pub transport_resolved: bool,
}

fn validate_rows(rows: &[Vec<f64>], minimum: usize, label: &str) -> BrainResult<usize> {
    if rows.len() < minimum {
        return Err(BrainError::Invalid(format!("{label}_anchor_count")));
    }
    let dim = rows[0].len();
    if dim == 0
        || rows
            .iter()
            .any(|row| row.len() != dim || row.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Invalid(format!("{label}_shape")));
    }
    Ok(dim)
}

fn augmented_rows(source: &[Vec<f64>]) -> Vec<Vec<f64>> {
    source
        .iter()
        .map(|row| {
            let mut augmented = Vec::with_capacity(row.len() + 1);
            augmented.extend_from_slice(row);
            augmented.push(1.0);
            augmented
        })
        .collect()
}

fn fit_affine(source: &[Vec<f64>], target: &[Vec<f64>], ridge: f64) -> BrainResult<TransportMap> {
    if source.len() != target.len() {
        return Err(BrainError::Invalid("transport_anchor_count".into()));
    }
    let source_dim = validate_rows(source, 3, "transport_source")?;
    let target_dim = validate_rows(target, 3, "transport_target")?;
    if !ridge.is_finite() || ridge <= 0.0 {
        return Err(BrainError::Invalid("transport_ridge_invalid".into()));
    }
    let augmented = augmented_rows(source);
    let design = Matrix::from_rows(&augmented)?;
    let row_weights = vec![1.0; source.len()];
    let mut weights = Matrix::zeros(target_dim, source_dim);
    let mut bias = vec![0.0; target_dim];
    let mut squared_error = 0.0;
    for output in 0..target_dim {
        let y = target.iter().map(|row| row[output]).collect::<Vec<_>>();
        let beta = weighted_normal_solve(&design, &y, &row_weights, ridge)?;
        for input in 0..source_dim {
            weights.set(output, input, beta[input]);
        }
        bias[output] = beta[source_dim];
        for row in 0..source.len() {
            let prediction = (0..source_dim)
                .map(|input| beta[input] * source[row][input])
                .sum::<f64>()
                + beta[source_dim];
            squared_error += (prediction - target[row][output]).powi(2);
        }
    }
    Ok(TransportMap {
        source_dim,
        target_dim,
        weights,
        bias,
        training_rms: (squared_error / (source.len() * target_dim) as f64).sqrt(),
    })
}

/// Backwards-compatible generation transport, now affine rather than forced
/// through the origin. Use `learn_transport_validated` before promotion.
pub fn learn_transport(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    ridge: f64,
) -> BrainResult<TransportMap> {
    fit_affine(source, target, ridge)
}

impl TransportMap {
    pub fn apply(&self, values: &[f64]) -> BrainResult<Vec<f64>> {
        if values.len() != self.source_dim
            || values.iter().any(|value| !value.is_finite())
            || self.bias.len() != self.target_dim
        {
            return Err(BrainError::Invalid("transport_apply_shape".into()));
        }
        let mut output = self.weights.matvec(values)?;
        for (value, bias) in output.iter_mut().zip(&self.bias) {
            *value += bias;
        }
        Ok(output)
    }
}

fn global_r2(actual: &[Vec<f64>], predicted: &[Vec<f64>]) -> BrainResult<f64> {
    if actual.is_empty()
        || actual.len() != predicted.len()
        || actual[0].is_empty()
        || actual.iter().any(|row| row.len() != actual[0].len())
        || predicted.iter().any(|row| row.len() != actual[0].len())
    {
        return Err(BrainError::Invalid("transport_r2_shape".into()));
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

fn leave_one_out_predictions(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    ridge: f64,
) -> BrainResult<Vec<Vec<f64>>> {
    if source.len() != target.len() || source.len() < 4 {
        return Err(BrainError::Invalid("transport_cv_anchor_count".into()));
    }
    let mut predictions = Vec::with_capacity(source.len());
    for holdout in 0..source.len() {
        let train_source = source
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != holdout)
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>();
        let train_target = target
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != holdout)
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>();
        let map = fit_affine(&train_source, &train_target, ridge)?;
        predictions.push(map.apply(&source[holdout])?);
    }
    Ok(predictions)
}

pub fn learn_transport_validated(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    ridge: f64,
) -> BrainResult<ValidatedTransportMap> {
    if source.len() != target.len() || source.len() < 4 {
        return Err(BrainError::Invalid("transport_cv_anchor_count".into()));
    }
    let map = fit_affine(source, target, ridge)?;
    let predicted = leave_one_out_predictions(source, target, ridge)?;
    let loo_cv_r2 = global_r2(target, &predicted)?;
    let squared_error = target
        .iter()
        .zip(&predicted)
        .flat_map(|(actual, prediction)| {
            actual
                .iter()
                .zip(prediction)
                .map(|(left, right)| (left - right).powi(2))
        })
        .sum::<f64>();
    let loo_cv_rms = (squared_error / (target.len() * target[0].len()) as f64).sqrt();
    let cosines = target
        .iter()
        .zip(&predicted)
        .map(|(actual, prediction)| cosine(actual, prediction))
        .collect::<BrainResult<Vec<_>>>()?;
    let mean_loo_cosine = cosines.iter().sum::<f64>() / cosines.len() as f64;
    let min_loo_cosine = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    let resolved = loo_cv_r2 > 0.0 && min_loo_cosine > 0.0;
    Ok(ValidatedTransportMap {
        schema: "cerebro.tidex.validated_transport/v1".into(),
        map,
        anchor_count: source.len(),
        loo_cv_r2,
        loo_cv_rms,
        mean_loo_cosine,
        min_loo_cosine,
        resolved,
    })
}

/// Functional transplantation does not require source and target parameter
/// dimensions to match. Common functional signatures are the bridge. The map
/// learns functional_signature -> target capability vector from matched target
/// anchors and is leave-one-anchor-out validated before it can be resolved.
pub fn learn_functional_transplant(
    functional_anchors: &[Vec<f64>],
    target_capability_anchors: &[Vec<f64>],
    ridge: f64,
) -> BrainResult<FunctionalTransplantMap> {
    if functional_anchors.len() != target_capability_anchors.len() || functional_anchors.len() < 4 {
        return Err(BrainError::Invalid(
            "functional_transplant_anchor_count".into(),
        ));
    }
    let functional_dim = validate_rows(functional_anchors, 4, "functional_transplant_function")?;
    let target_dim = validate_rows(target_capability_anchors, 4, "functional_transplant_target")?;
    let validated =
        learn_transport_validated(functional_anchors, target_capability_anchors, ridge)?;
    Ok(FunctionalTransplantMap {
        schema: "cerebro.tidex.functional_transplant/v1".into(),
        functional_dim,
        target_dim,
        target_decoder: validated.map,
        anchor_count: validated.anchor_count,
        loo_cv_r2: validated.loo_cv_r2,
        mean_loo_cosine: validated.mean_loo_cosine,
        min_loo_cosine: validated.min_loo_cosine,
        resolved: validated.resolved,
    })
}

impl FunctionalTransplantMap {
    pub fn transplant(
        &self,
        source_functional_signature: &[f64],
    ) -> BrainResult<TransplantedCapability> {
        if source_functional_signature.len() != self.functional_dim
            || source_functional_signature
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(BrainError::Invalid(
                "functional_transplant_signature_shape".into(),
            ));
        }
        Ok(TransplantedCapability {
            target_vector: self.target_decoder.apply(source_functional_signature)?,
            source_functional_signature: source_functional_signature.to_vec(),
            transport_resolved: self.resolved,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationalTransportMap {
    pub schema: String,
    pub source_signature_dim: usize,
    pub target_signature_dim: usize,
    pub anchor_count: usize,
    pub ridge: f64,
    pub loo_source_cosines: Vec<f64>,
    pub loo_target_cosines: Vec<f64>,
    pub loo_coefficient_norms: Vec<f64>,
    pub min_loo_source_cosine: f64,
    pub min_loo_target_cosine: f64,
    pub mean_loo_source_cosine: f64,
    pub mean_loo_target_cosine: f64,
    pub max_loo_coefficient_norm: f64,
    pub resolved: bool,
    source_anchors: Vec<Vec<f64>>,
    target_anchors: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationalTransplant {
    pub target_coefficients: Vec<f64>,
    pub predicted_target_signature: Vec<f64>,
    pub source_projection_cosine: f64,
    pub coefficient_norm: f64,
    pub within_training_support: bool,
    pub resolved: bool,
}

fn normalize_anchor_rows(rows: &[Vec<f64>], label: &str) -> BrainResult<Vec<Vec<f64>>> {
    validate_rows(rows, 4, label)?;
    let mut normalized = Vec::with_capacity(rows.len());
    for row in rows {
        if norm(row)? <= 1e-15 {
            return Err(BrainError::Invalid(format!("{label}_zero_norm")));
        }
        normalized.push(normalize(row)?);
    }
    Ok(normalized)
}

fn relational_coefficients(
    anchors: &[Vec<f64>],
    query: &[f64],
    ridge: f64,
) -> BrainResult<Vec<f64>> {
    if anchors.is_empty()
        || query.len() != anchors[0].len()
        || query.iter().any(|value| !value.is_finite())
        || !ridge.is_finite()
        || ridge <= 0.0
    {
        return Err(BrainError::Invalid(
            "relational_transport_coefficient_input".into(),
        ));
    }
    let mut gram = Matrix::zeros(anchors.len(), anchors.len());
    let mut rhs = vec![0.0; anchors.len()];
    for i in 0..anchors.len() {
        rhs[i] = dot(&anchors[i], query)?;
        for j in 0..=i {
            let value = dot(&anchors[i], &anchors[j])?;
            gram.set(i, j, value);
            gram.set(j, i, value);
        }
        gram.set(i, i, gram.get(i, i) + ridge);
    }
    solve(gram, rhs)
}

fn combine_anchors(coefficients: &[f64], anchors: &[Vec<f64>]) -> BrainResult<Vec<f64>> {
    if anchors.is_empty()
        || coefficients.len() != anchors.len()
        || coefficients.iter().any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid(
            "relational_transport_combine_shape".into(),
        ));
    }
    let dim = anchors[0].len();
    if dim == 0 || anchors.iter().any(|row| row.len() != dim) {
        return Err(BrainError::Invalid(
            "relational_transport_anchor_shape".into(),
        ));
    }
    let mut output = vec![0.0; dim];
    for (coefficient, anchor) in coefficients.iter().zip(anchors) {
        for index in 0..dim {
            output[index] += coefficient * anchor[index];
        }
    }
    Ok(output)
}

/// Learn a transport from *relations between matched capabilities*, not from
/// arbitrary target basis IDs. Each source holdout anchor is reconstructed from
/// the remaining source anchors; the same barycentric coefficients are applied
/// to the matched target anchors and compared with the true target holdout.
///
/// The map is resolved only when target leave-one-out reconstruction is at
/// least as strong as the weakest source leave-one-out reconstruction. This is
/// a data-derived gate: cross-backbone transport may not claim more support
/// than the source geometry itself demonstrates.
pub fn learn_relational_transport(
    source_anchors: &[Vec<f64>],
    target_anchors: &[Vec<f64>],
    ridge: f64,
) -> BrainResult<RelationalTransportMap> {
    if source_anchors.len() != target_anchors.len() || source_anchors.len() < 5 {
        return Err(BrainError::Invalid(
            "relational_transport_anchor_count".into(),
        ));
    }
    if !ridge.is_finite() || ridge <= 0.0 {
        return Err(BrainError::Invalid("relational_transport_ridge".into()));
    }
    let source = normalize_anchor_rows(source_anchors, "relational_source")?;
    let target = normalize_anchor_rows(target_anchors, "relational_target")?;
    let mut loo_source_cosines = Vec::with_capacity(source.len());
    let mut loo_target_cosines = Vec::with_capacity(source.len());
    let mut loo_coefficient_norms = Vec::with_capacity(source.len());
    for holdout in 0..source.len() {
        let train_source = source
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != holdout)
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>();
        let train_target = target
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != holdout)
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>();
        let coefficients = relational_coefficients(&train_source, &source[holdout], ridge)?;
        let source_prediction = combine_anchors(&coefficients, &train_source)?;
        let target_prediction = combine_anchors(&coefficients, &train_target)?;
        let source_cosine = cosine(&source_prediction, &source[holdout])?;
        let target_cosine = cosine(&target_prediction, &target[holdout])?;
        let coefficient_norm = norm(&coefficients)?;
        if !source_cosine.is_finite() || !target_cosine.is_finite() || !coefficient_norm.is_finite()
        {
            return Err(BrainError::Numerical(
                "relational_transport_non_finite_loo".into(),
            ));
        }
        loo_source_cosines.push(source_cosine);
        loo_target_cosines.push(target_cosine);
        loo_coefficient_norms.push(coefficient_norm);
    }
    let min_loo_source_cosine = loo_source_cosines
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let min_loo_target_cosine = loo_target_cosines
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let mean_loo_source_cosine =
        loo_source_cosines.iter().sum::<f64>() / loo_source_cosines.len() as f64;
    let mean_loo_target_cosine =
        loo_target_cosines.iter().sum::<f64>() / loo_target_cosines.len() as f64;
    let max_loo_coefficient_norm = loo_coefficient_norms
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let numerical_tolerance = f64::EPSILON.sqrt();
    let resolved = min_loo_source_cosine > 0.0
        && min_loo_target_cosine + numerical_tolerance >= min_loo_source_cosine
        && max_loo_coefficient_norm.is_finite();
    Ok(RelationalTransportMap {
        schema: "cerebro.tidex.relational_transport/v1".into(),
        source_signature_dim: source[0].len(),
        target_signature_dim: target[0].len(),
        anchor_count: source.len(),
        ridge,
        loo_source_cosines,
        loo_target_cosines,
        loo_coefficient_norms,
        min_loo_source_cosine,
        min_loo_target_cosine,
        mean_loo_source_cosine,
        mean_loo_target_cosine,
        max_loo_coefficient_norm,
        resolved,
        source_anchors: source,
        target_anchors: target,
    })
}

impl RelationalTransportMap {
    pub fn transplant(&self, source_signature: &[f64]) -> BrainResult<RelationalTransplant> {
        if source_signature.len() != self.source_signature_dim
            || source_signature.iter().any(|value| !value.is_finite())
            || norm(source_signature)? <= 1e-15
        {
            return Err(BrainError::Invalid(
                "relational_transplant_signature_shape".into(),
            ));
        }
        let normalized = normalize(source_signature)?;
        let target_coefficients =
            relational_coefficients(&self.source_anchors, &normalized, self.ridge)?;
        let source_prediction = combine_anchors(&target_coefficients, &self.source_anchors)?;
        let predicted_target_signature =
            combine_anchors(&target_coefficients, &self.target_anchors)?;
        let source_projection_cosine = cosine(&source_prediction, &normalized)?;
        let coefficient_norm = norm(&target_coefficients)?;
        let numerical_tolerance = f64::EPSILON.sqrt();
        let within_training_support = source_projection_cosine + numerical_tolerance
            >= self.min_loo_source_cosine
            && coefficient_norm <= self.max_loo_coefficient_norm + numerical_tolerance;
        Ok(RelationalTransplant {
            target_coefficients,
            predicted_target_signature,
            source_projection_cosine,
            coefficient_norm,
            within_training_support,
            resolved: self.resolved && within_training_support,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relational_transport_validates_geometry_and_rejects_ood_query() {
        let source = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 0.0],
        ];
        // Same relational geometry embedded into a different target dimension.
        let target = source
            .iter()
            .map(|row| vec![row[0], row[1], 0.0, 0.0, 0.0])
            .collect::<Vec<_>>();
        let map = learn_relational_transport(&source, &target, 1e-8).unwrap();
        assert!(map.resolved);
        assert!(map.min_loo_target_cosine + 1e-10 >= map.min_loo_source_cosine);
        let valid = map.transplant(&[1.5, 1.0, 0.0]).unwrap();
        assert!(valid.resolved);
        assert_eq!(valid.target_coefficients.len(), 5);
        let ood = map.transplant(&[0.0, 0.0, 1.0]).unwrap();
        assert!(!ood.within_training_support);
        assert!(!ood.resolved);
    }

    #[test]
    fn validated_affine_transport_generalizes_across_generation_anchors() {
        let source = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, -1.0],
            vec![-1.0, 2.0],
            vec![0.5, 2.0],
        ];
        let target = source
            .iter()
            .map(|values| {
                vec![
                    2.0 * values[0] + values[1] + 0.25,
                    -values[0] + 3.0 * values[1] - 0.5,
                    0.5 * values[0] - 0.2 * values[1] + 1.0,
                ]
            })
            .collect::<Vec<_>>();
        let map = learn_transport_validated(&source, &target, 1e-9).unwrap();
        assert!(map.loo_cv_r2 > 0.999999);
        assert!(map.min_loo_cosine > 0.999999);
        assert!(map.resolved);
    }

    #[test]
    fn functional_transplant_crosses_incompatible_parameter_dimensions() {
        let functions = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, -1.0],
            vec![-1.0, 2.0],
            vec![0.5, 2.0],
        ];
        let target = functions
            .iter()
            .map(|values| {
                vec![
                    values[0] + 2.0 * values[1],
                    -values[0] + values[1],
                    0.5 * values[0],
                    3.0 * values[1],
                    values[0] - values[1] + 0.2,
                ]
            })
            .collect::<Vec<_>>();
        let transplant = learn_functional_transplant(&functions, &target, 1e-9).unwrap();
        assert_eq!(transplant.functional_dim, 2);
        assert_eq!(transplant.target_dim, 5);
        assert!(transplant.resolved);
        let result = transplant.transplant(&[0.25, 0.75]).unwrap();
        assert_eq!(result.target_vector.len(), 5);
        assert!(result.transport_resolved);
    }
}
