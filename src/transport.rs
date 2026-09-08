#![allow(clippy::needless_range_loop)]

use crate::error::{BrainError, BrainResult};
use crate::linalg::{
    compensated_sum, cosine, dot, norm, normalize, solve, stable_rms, weighted_normal_solve, Matrix,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AffineTransportPolicy {
    /// Original affine solve, including its penalized intercept.
    FixedRidge { ridge: f64 },
    /// Center within each training partition and regularize with
    /// lambda = relative_ridge * trace(X_centered^T X_centered) / input_dimension.
    /// The fitted intercept is not penalized.
    CenteredTraceRidge { relative_ridge: f64 },
}

impl AffineTransportPolicy {
    pub fn validate(&self) -> BrainResult<()> {
        let strength = match self {
            Self::FixedRidge { ridge } => *ridge,
            Self::CenteredTraceRidge { relative_ridge } => *relative_ridge,
        };
        if !strength.is_finite() || strength <= 0.0 {
            return Err(BrainError::Invalid(
                "transport_regularization_policy_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AffineFitDiagnostics {
    pub training_count: usize,
    pub input_dimension: usize,
    pub centered: bool,
    pub centered_design_trace: Option<f64>,
    pub regularization_scale: f64,
    pub effective_ridge: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AffineTransportDiagnostics {
    pub schema: String,
    pub policy: AffineTransportPolicy,
    pub full_fit: AffineFitDiagnostics,
    /// Entry i was fitted without source[i] or target[i]. Its means, design
    /// scale and effective ridge are derived anew from that training partition.
    pub leave_one_out: Vec<AffineFitDiagnostics>,
}

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

fn fit_affine_with_policy(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    policy: &AffineTransportPolicy,
) -> BrainResult<(TransportMap, AffineFitDiagnostics)> {
    policy.validate()?;
    let relative_ridge = match policy {
        AffineTransportPolicy::FixedRidge { ridge } => {
            let map = fit_affine(source, target, *ridge)?;
            let diagnostics = AffineFitDiagnostics {
                training_count: source.len(),
                input_dimension: map.source_dim,
                centered: false,
                centered_design_trace: None,
                regularization_scale: 1.0,
                effective_ridge: *ridge,
            };
            return Ok((map, diagnostics));
        }
        AffineTransportPolicy::CenteredTraceRidge { relative_ridge } => *relative_ridge,
    };
    if source.len() != target.len() {
        return Err(BrainError::Invalid("transport_anchor_count".into()));
    }
    let source_dim = validate_rows(source, 3, "transport_source")?;
    let target_dim = validate_rows(target, 3, "transport_target")?;
    let count = source.len() as f64;
    let means = |rows: &[Vec<f64>], dimension: usize| -> BrainResult<Vec<f64>> {
        (0..dimension)
            .map(|column| compensated_sum(rows.iter().map(|row| row[column] / count)))
            .collect()
    };
    let source_mean = means(source, source_dim)?;
    let target_mean = means(target, target_dim)?;
    let centered = source
        .iter()
        .map(|row| {
            row.iter()
                .zip(&source_mean)
                .map(|(x, mean)| x - mean)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // Solve in global unit coordinates to avoid the absolute pivot floor of
    // the existing solver when the same data is expressed in very small units.
    // This is algebraically the declared trace-scaled ridge, not a new solver.
    let unit_scale = stable_rms(centered.iter().flatten().copied())? * count.sqrt();
    let regularization_scale = unit_scale * unit_scale;
    let centered_design_trace = regularization_scale * source_dim as f64;
    let effective_ridge = relative_ridge * regularization_scale;
    if !unit_scale.is_finite()
        || unit_scale <= 0.0
        || !regularization_scale.is_finite()
        || regularization_scale <= 0.0
        || !centered_design_trace.is_finite()
        || !effective_ridge.is_finite()
        || effective_ridge <= 0.0
    {
        return Err(BrainError::Numerical(
            "transport_centered_design_degenerate".into(),
        ));
    }
    let normalized_rows = centered
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value / unit_scale)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let design = Matrix::from_rows(&normalized_rows)?;
    let row_weights = vec![1.0; source.len()];
    let mut weights = Matrix::zeros(target_dim, source_dim);
    let mut bias = vec![0.0; target_dim];
    for output in 0..target_dim {
        let centered_target = target
            .iter()
            .map(|row| row[output] - target_mean[output])
            .collect::<Vec<_>>();
        let beta = weighted_normal_solve(&design, &centered_target, &row_weights, relative_ridge)?;
        for input in 0..source_dim {
            weights.set(output, input, beta[input] / unit_scale);
        }
        bias[output] = target_mean[output] - dot(weights.row(output), &source_mean)?;
    }
    weights.validate("transport_centered_weights")?;
    if bias.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Numerical(
            "transport_centered_bias_nonfinite".into(),
        ));
    }
    let mut map = TransportMap {
        source_dim,
        target_dim,
        weights,
        bias,
        training_rms: 0.0,
    };
    let residuals = source
        .iter()
        .zip(target)
        .map(|(input, actual)| {
            let predicted = map.apply(input)?;
            Ok(predicted
                .iter()
                .zip(actual)
                .map(|(prediction, value)| prediction - value)
                .collect::<Vec<_>>())
        })
        .collect::<BrainResult<Vec<_>>>()?;
    map.training_rms = stable_rms(residuals.iter().flatten().copied())?;
    Ok((
        map,
        AffineFitDiagnostics {
            training_count: source.len(),
            input_dimension: source_dim,
            centered: true,
            centered_design_trace: Some(centered_design_trace),
            regularization_scale,
            effective_ridge,
        },
    ))
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
    validated_from_predictions(map, target, &predicted)
}

fn validated_from_predictions(
    map: TransportMap,
    target: &[Vec<f64>],
    predicted: &[Vec<f64>],
) -> BrainResult<ValidatedTransportMap> {
    let loo_cv_r2 = global_r2(target, predicted)?;
    let squared_error = target
        .iter()
        .zip(predicted)
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
        .zip(predicted)
        .map(|(actual, prediction)| cosine(actual, prediction))
        .collect::<BrainResult<Vec<_>>>()?;
    let mean_loo_cosine = cosines.iter().sum::<f64>() / cosines.len() as f64;
    let min_loo_cosine = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    let resolved = loo_cv_r2 > 0.0 && min_loo_cosine > 0.0;
    Ok(ValidatedTransportMap {
        schema: "cerebro.tidex.validated_transport/v1".into(),
        map,
        anchor_count: target.len(),
        loo_cv_r2,
        loo_cv_rms,
        mean_loo_cosine,
        min_loo_cosine,
        resolved,
    })
}

fn leave_one_out_with_policy(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    policy: &AffineTransportPolicy,
) -> BrainResult<(Vec<Vec<f64>>, Vec<AffineFitDiagnostics>)> {
    if source.len() != target.len() || source.len() < 4 {
        return Err(BrainError::Invalid("transport_cv_anchor_count".into()));
    }
    let mut predictions = Vec::with_capacity(source.len());
    let mut diagnostics = Vec::with_capacity(source.len());
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
        let (map, fit) = fit_affine_with_policy(&train_source, &train_target, policy)?;
        predictions.push(map.apply(&source[holdout])?);
        diagnostics.push(fit);
    }
    Ok((predictions, diagnostics))
}

/// Explicit policy entry point. Existing callers of learn_transport_validated
/// retain the original fixed-ridge implementation. Numerical stabilization is
/// not evidence of semantic generalization: the same held-out gates still apply.
pub fn learn_transport_validated_with_policy(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    policy: &AffineTransportPolicy,
) -> BrainResult<(ValidatedTransportMap, AffineTransportDiagnostics)> {
    if source.len() != target.len() || source.len() < 4 {
        return Err(BrainError::Invalid("transport_cv_anchor_count".into()));
    }
    let (map, full_fit) = fit_affine_with_policy(source, target, policy)?;
    let (predictions, leave_one_out) = leave_one_out_with_policy(source, target, policy)?;
    Ok((
        validated_from_predictions(map, target, &predictions)?,
        AffineTransportDiagnostics {
            schema: "cerebro.tidex.affine_transport_diagnostics/v1".into(),
            policy: policy.clone(),
            full_fit,
            leave_one_out,
        },
    ))
}

/// Functional-signature to receiver-coordinate compilation using an explicitly
/// selected affine policy. No target update or target training data is accepted.
pub fn learn_functional_transplant_with_policy(
    functional_anchors: &[Vec<f64>],
    target_capability_anchors: &[Vec<f64>],
    policy: &AffineTransportPolicy,
) -> BrainResult<(FunctionalTransplantMap, AffineTransportDiagnostics)> {
    if functional_anchors.len() != target_capability_anchors.len() || functional_anchors.len() < 4 {
        return Err(BrainError::Invalid(
            "functional_transplant_anchor_count".into(),
        ));
    }
    let functional_dim = validate_rows(functional_anchors, 4, "functional_transplant_function")?;
    let target_dim = validate_rows(target_capability_anchors, 4, "functional_transplant_target")?;
    let (validated, diagnostics) = learn_transport_validated_with_policy(
        functional_anchors,
        target_capability_anchors,
        policy,
    )?;
    Ok((
        FunctionalTransplantMap {
            schema: "cerebro.tidex.functional_transplant/v1".into(),
            functional_dim,
            target_dim,
            target_decoder: validated.map,
            anchor_count: validated.anchor_count,
            loo_cv_r2: validated.loo_cv_r2,
            mean_loo_cosine: validated.mean_loo_cosine,
            min_loo_cosine: validated.min_loo_cosine,
            resolved: validated.resolved,
        },
        diagnostics,
    ))
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
    /// Undefined when the learned anchor projection has zero norm. `None` is
    /// evidence that the query lies outside represented source geometry; it
    /// must not be replaced by a fabricated zero cosine.
    pub source_projection_cosine: Option<f64>,
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
        let source_projection_cosine = if norm(&source_prediction)? <= 1e-15 {
            None
        } else {
            Some(cosine(&source_prediction, &normalized)?)
        };
        let coefficient_norm = norm(&target_coefficients)?;
        let numerical_tolerance = f64::EPSILON.sqrt();
        let within_training_support = source_projection_cosine.is_some_and(|cosine| {
            cosine + numerical_tolerance >= self.min_loo_source_cosine
                && coefficient_norm <= self.max_loo_coefficient_norm + numerical_tolerance
        });
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
    fn centered_trace_ridge_preserves_source_units_and_affine_origins() {
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
            .map(|row| {
                vec![
                    10.0 + 2.0 * row[0] - row[1],
                    6.0 + 0.5 * row[0] + 3.0 * row[1],
                ]
            })
            .collect::<Vec<_>>();
        let policy = AffineTransportPolicy::CenteredTraceRidge {
            relative_ridge: 0.2,
        };
        let (reference, reference_diagnostics) =
            learn_transport_validated_with_policy(&source, &target, &policy).unwrap();
        let query = [0.25, 0.75];
        let expected = reference.map.apply(&query).unwrap();
        for scale in [1e-6, 1.0, 1e6] {
            let source_offset = [3.0 * scale, -7.0 * scale];
            let target_offset = [25.0, -12.0];
            let changed_source = source
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(i, value)| scale * value + source_offset[i])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let changed_target = target
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(i, value)| value + target_offset[i])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let (changed, diagnostics) =
                learn_transport_validated_with_policy(&changed_source, &changed_target, &policy)
                    .unwrap();
            let actual = changed
                .map
                .apply(&[
                    scale * query[0] + source_offset[0],
                    scale * query[1] + source_offset[1],
                ])
                .unwrap();
            for output in 0..expected.len() {
                assert!((actual[output] - target_offset[output] - expected[output]).abs() < 1e-9);
            }
            let expected_ridge = reference_diagnostics.full_fit.effective_ridge * scale * scale;
            assert!((diagnostics.full_fit.effective_ridge / expected_ridge - 1.0).abs() < 1e-12);
            assert!((changed.loo_cv_r2 - reference.loo_cv_r2).abs() < 1e-10);
        }
    }

    #[test]
    fn trace_ridge_loo_refits_mean_and_scale_without_holdout_values() {
        let policy = AffineTransportPolicy::CenteredTraceRidge {
            relative_ridge: 0.5,
        };
        let mut source = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![1000.0]];
        let mut target = vec![vec![3.0], vec![5.0], vec![7.0], vec![9.0], vec![-777.0]];
        let (predicted, diagnostics) =
            leave_one_out_with_policy(&source, &target, &policy).unwrap();
        // Independent closed form from the four training rows:
        // mean_x=1.5, mean_y=6, sum(x-mean_x)^2=5,
        // slope=(2*5)/(5+0.5*5), bias=6-slope*1.5.
        let slope = 10.0 / 7.5;
        let expected = 6.0 + slope * (1000.0 - 1.5);
        assert!((predicted[4][0] - expected).abs() < 1e-10);
        assert_eq!(diagnostics[4].training_count, 4);
        assert!((diagnostics[4].regularization_scale - 5.0).abs() < 1e-12);
        assert!((diagnostics[4].effective_ridge - 2.5).abs() < 1e-12);
        target[4][0] = 1e9;
        let (changed, changed_diagnostics) =
            leave_one_out_with_policy(&source, &target, &policy).unwrap();
        assert_eq!(predicted[4], changed[4]);
        assert_eq!(diagnostics[4], changed_diagnostics[4]);
        source[4][0] = 1e6;
        let (changed, changed_diagnostics) =
            leave_one_out_with_policy(&source, &target, &policy).unwrap();
        assert!((changed[4][0] - (6.0 + slope * (1e6 - 1.5))).abs() < 1e-8);
        assert_eq!(diagnostics[4], changed_diagnostics[4]);
    }

    #[test]
    fn explicit_fixed_policy_preserves_legacy_maps_and_diagnostics_are_honest() {
        let source = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let target = vec![vec![3.0], vec![5.0], vec![7.0], vec![9.0], vec![11.0]];
        let policy = AffineTransportPolicy::FixedRidge { ridge: 0.75 };
        let old = learn_transport_validated(&source, &target, 0.75).unwrap();
        let (new, diagnostics) =
            learn_transport_validated_with_policy(&source, &target, &policy).unwrap();
        assert_eq!(old, new);
        assert!(!diagnostics.full_fit.centered);
        assert_eq!(diagnostics.full_fit.centered_design_trace, None);
        assert!(diagnostics
            .leave_one_out
            .iter()
            .all(|fit| fit.effective_ridge == 0.75));
        let old = learn_functional_transplant(&source, &target, 0.75).unwrap();
        let (new, _) = learn_functional_transplant_with_policy(&source, &target, &policy).unwrap();
        assert_eq!(old, new);
    }

    #[test]
    fn centered_trace_ridge_rejects_unidentified_design_in_full_fit_or_any_fold() {
        let policy = AffineTransportPolicy::CenteredTraceRidge {
            relative_ridge: 0.5,
        };
        let target = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let constant = vec![vec![2.0, -3.0]; 4];
        let error = learn_transport_validated_with_policy(&constant, &target, &policy).unwrap_err();
        assert!(error
            .to_string()
            .contains("transport_centered_design_degenerate"));
        let single_excitation = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];
        let error = learn_transport_validated_with_policy(&single_excitation, &target, &policy)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("transport_centered_design_degenerate"));
        let source = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        for relative_ridge in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let invalid = AffineTransportPolicy::CenteredTraceRidge { relative_ridge };
            assert!(learn_transport_validated_with_policy(&source, &target, &invalid).is_err());
        }
    }

    #[test]
    fn centered_trace_ridge_does_not_penalize_the_intercept_or_claim_constant_targets_resolved() {
        let source = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let target = vec![vec![7.0, -2.0]; source.len()];
        let policy = AffineTransportPolicy::CenteredTraceRidge {
            relative_ridge: 100.0,
        };
        let (fit, _) = learn_transport_validated_with_policy(&source, &target, &policy).unwrap();
        assert_eq!(fit.map.apply(&[50.0]).unwrap(), vec![7.0, -2.0]);
        assert_eq!(fit.map.training_rms, 0.0);
        assert_eq!(fit.loo_cv_r2, 0.0);
        assert!(!fit.resolved);
        let policy_json =
            r#"{"kind":"centered_trace_ridge","relative_ridge":1.0,"hidden_fallback":true}"#;
        assert!(serde_json::from_str::<AffineTransportPolicy>(policy_json).is_err());
    }

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
        assert_eq!(ood.source_projection_cosine, None);
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
