//! Receiver-specific compilation from canonical functional semantics.
//!
//! The numerical compiler is calibrated from capability-independent functional
//! signatures and receiver-native solutions. The bounded readout adapter executes
//! an authenticated donor CapabilityIr to obtain a requested signature; donor
//! parameter coordinates are never used as receiver update coordinates. The
//! compiler predicts a receiver delta, applies protection/trust-region constraints,
//! and verifies its prediction in canonical functional space.
//!
//! "Compilation" is the mechanism implemented here. "Portability" is only an
//! empirical property measured by held-out benchmarks over an explicitly
//! declared calibration domain; nothing in this module by itself establishes
//! universal cross-model or cross-capability portability.

use crate::acquisition_contract::SystemEnvelope;
use crate::capability_ir::{
    execute_linear_readout, CapabilityIr, LinearReadoutExecution, OperationalCapabilityContract,
    OperationalInterfaceVerification,
};
use crate::contracts::{BrainConfig, ProtectedCortex};
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::identifiability::{resolution_map, ResolutionMap};
use crate::identity::CapabilityId;
use crate::linalg::{cosine, dot, norm, stable_rms, weighted_normal_solve, Matrix};
use crate::protected::project_to_safe_subspace;
use crate::tomography::reconstruct_skill_fields;
use crate::transport::{
    learn_functional_transplant, learn_functional_transplant_with_policy,
    learn_relational_transport, learn_transport_validated, learn_transport_validated_with_policy,
    AffineTransportDiagnostics, AffineTransportPolicy, TransportMap,
};
use crate::trust_region::{apply_quadratic_trust_region, TrustRegionResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverCompilerPolicy {
    pub schema: String,
    pub ridge: f64,
    pub minimum_decoder_loo_r2: f64,
    pub minimum_encoder_loo_r2: f64,
    pub minimum_decoder_loo_cosine: f64,
    pub maximum_functional_relative_error: f64,
    pub minimum_identity_margin: f64,
    pub maximum_quadratic_cost: f64,
}

impl ReceiverCompilerPolicy {
    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.receiver_compiler_policy/v1"
            || !self.ridge.is_finite()
            || self.ridge <= 0.0
            || !self.minimum_decoder_loo_r2.is_finite()
            || self.minimum_decoder_loo_r2 > 1.0
            || !self.minimum_encoder_loo_r2.is_finite()
            || self.minimum_encoder_loo_r2 > 1.0
            || !self.minimum_decoder_loo_cosine.is_finite()
            || !(-1.0..=1.0).contains(&self.minimum_decoder_loo_cosine)
            || !self.maximum_functional_relative_error.is_finite()
            || self.maximum_functional_relative_error < 0.0
            || !self.minimum_identity_margin.is_finite()
            || !(-2.0..=2.0).contains(&self.minimum_identity_margin)
            || !self.maximum_quadratic_cost.is_finite()
            || self.maximum_quadratic_cost < 0.0
        {
            return Err(BrainError::Invalid(
                "receiver_compiler_policy_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverCalibrationSet {
    /// Optional exact snapshot binding. Planning requires it; legacy numerical
    /// requests retain their unchanged wire representation when it is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receiver_snapshot_binding_sha256: Option<crate::digest::Sha256Digest>,
    pub functional_signatures: Vec<Vec<f64>>,
    pub receiver_solutions: Vec<Vec<f64>>,
    pub wrong_functional_signatures: Vec<Vec<f64>>,
}

/// Numerical calibration request. This surface executes the same kernel as
/// weight binding, writes no artifacts and grants no execution or promotion
/// authority. Experiments use it to avoid a second compiler in Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverSignatureBenchmarkInput {
    pub schema: String,
    pub requested: Vec<f64>,
    pub calibration: ReceiverCalibrationSet,
    pub protected_cortex: ProtectedCortex,
    pub risk_metric: Vec<Vec<f64>>,
    pub policy: ReceiverCompilerPolicy,
    pub proposal_method: ReceiverProposalMethod,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_profile: Option<ReceiverProposalValidationProfile>,
}

pub fn benchmark_receiver_signature(
    input: &ReceiverSignatureBenchmarkInput,
) -> BrainResult<ReceiverSignatureCompilation> {
    if input.schema != "cerebro.tidex.receiver_signature_benchmark_input/v1"
        || input.risk_metric.len() > 256
        || input.risk_metric.iter().any(|row| row.len() > 256)
    {
        return Err(BrainError::Invalid(
            "receiver_signature_benchmark_input_invalid".into(),
        ));
    }
    compile_signature_with_method_and_validation(
        &input.requested,
        &input.calibration,
        &input.protected_cortex,
        &Matrix::from_rows(&input.risk_metric)?,
        &input.policy,
        input.proposal_method,
        input
            .validation_profile
            .unwrap_or(ReceiverProposalValidationProfile::ParametricCrossValidation),
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverBasisBenchmarkInput {
    pub schema: String,
    pub calibration_capability_ids: Vec<CapabilityId>,
    pub calibration_deltas: Vec<Vec<f64>>,
    pub target_explained_variance: f64,
    pub max_rank: usize,
    pub ridge: f64,
    pub min_signal_to_noise: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverBasisBenchmarkReport {
    pub schema: String,
    /// Ordered lineage of the supplied rows; not an assertion of independent
    /// causal apertures or authentication of their origin.
    pub calibration_capability_ids: Vec<CapabilityId>,
    pub parameter_dimension: usize,
    pub selected_rank: usize,
    /// Genuine tomography directions: [axis, receiver parameter].
    pub axes: Vec<Vec<f64>>,
    /// [calibration row, axis], in the same ordered lineage as the input.
    pub coordinates: Vec<Vec<f64>>,
    /// [axis, calibration row]; combines actual input deltas into each axis.
    pub source_mixtures: Vec<Vec<f64>>,
    pub reconstruction_rms: f64,
    /// 1 - unweighted raw reconstruction SSE / unweighted raw delta energy.
    /// This is recomputed; robust spectral explained variance is not substituted.
    pub retained_energy: f64,
    pub effective_rank: f64,
    pub robust_weights: Vec<f64>,
    pub resolution: ResolutionMap,
    pub candidate_only: bool,
}

/// Numerical adapter over the existing tomography and identifiability kernels.
/// It creates no observations, model updates, artifacts or promotion authority.
/// Callers authenticate the calibration deltas and exclude held-out/target rows
/// before this call. The source mixtures retain exact row lineage.
pub fn benchmark_receiver_basis(
    input: &ReceiverBasisBenchmarkInput,
) -> BrainResult<ReceiverBasisBenchmarkReport> {
    let n = input.calibration_deltas.len();
    let p = input.calibration_deltas.first().map_or(0, Vec::len);
    if !(5..=256).contains(&n)
        || p == 0
        || p > 4096
        || input.max_rank == 0
        || input.max_rank > n.min(p).min(32)
    {
        return Err(BrainError::Invalid(
            "receiver_basis_resource_or_rank_bounds".into(),
        ));
    }
    if input.schema != "cerebro.tidex.receiver_basis_benchmark_input/v1"
        || input.calibration_capability_ids.len() != n
        || !input.target_explained_variance.is_finite()
        || !(0.0..=1.0).contains(&input.target_explained_variance)
        || input.target_explained_variance <= 0.0
        || !input.ridge.is_finite()
        || input.ridge <= 0.0
        || !input.min_signal_to_noise.is_finite()
        || input.min_signal_to_noise <= 0.0
    {
        return Err(BrainError::Invalid("receiver_basis_input_contract".into()));
    }
    let mut unique_ids = BTreeSet::new();
    if input
        .calibration_capability_ids
        .iter()
        .any(|id| !unique_ids.insert(id.as_str()))
    {
        return Err(BrainError::Invalid(
            "receiver_basis_duplicate_calibration_id".into(),
        ));
    }
    if input
        .calibration_deltas
        .iter()
        .any(|row| row.len() != p || row.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Invalid(
            "receiver_basis_calibration_shape_or_values".into(),
        ));
    }
    let raw_rms = stable_rms(input.calibration_deltas.iter().flatten().copied())?;
    if raw_rms <= 0.0 {
        return Err(BrainError::Numerical(
            "receiver_basis_zero_calibration_energy".into(),
        ));
    }
    let deltas = Matrix::from_rows(&input.calibration_deltas)?;
    let weights = vec![1.0; n];
    let groups = input
        .calibration_capability_ids
        .iter()
        .map(|id| id.as_str().to_owned())
        .collect::<Vec<_>>();
    let config = BrainConfig {
        target_explained_variance: input.target_explained_variance,
        max_rank: input.max_rank,
        ridge: input.ridge,
        min_identifiability_signal_to_noise: input.min_signal_to_noise,
        ..BrainConfig::default()
    };
    // Generation zero identifies this ephemeral numerical decomposition; no
    // durable SkillField identity or engine generation is fabricated.
    let tomography = reconstruct_skill_fields(&deltas, &weights, &groups, 0, &config)?;
    let rank = tomography.selected_rank;
    if rank == 0
        || rank > input.max_rank
        || tomography.fields.len() != rank
        || tomography.coefficients.row_count() != n
        || tomography.coefficients.column_count() != rank
        || tomography.source_mixtures.len() != rank
        || tomography
            .source_mixtures
            .iter()
            .any(|row| row.len() != n || row.iter().any(|value| !value.is_finite()))
        || tomography.fields.iter().any(|field| {
            field.direction.len() != p || field.direction.iter().any(|value| !value.is_finite())
        })
        || tomography.robust_weights.len() != n
        || tomography
            .robust_weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
        || !tomography.effective_rank.is_finite()
        || !tomography.reconstruction_rms.is_finite()
    {
        return Err(BrainError::Integrity(
            "receiver_basis_tomography_lineage_or_shape".into(),
        ));
    }
    tomography
        .coefficients
        .validate("receiver_basis_coordinates")?;
    let axes = tomography
        .fields
        .iter()
        .map(|field| field.direction.clone())
        .collect::<Vec<_>>();
    let coordinates = (0..n)
        .map(|row| tomography.coefficients.row_vec(row))
        .collect::<Vec<_>>();
    let tolerance = f64::EPSILON.sqrt() * (n.max(p) as f64).sqrt() * 16.0;
    // Check the exposed lineage before trusting an energy or resolution report.
    for (axis, axis_values) in axes.iter().enumerate().take(rank) {
        let reconstruction = (0..p)
            .map(|parameter| {
                let column = input
                    .calibration_deltas
                    .iter()
                    .map(|row| row[parameter])
                    .collect::<Vec<_>>();
                dot(&tomography.source_mixtures[axis], &column)
            })
            .collect::<BrainResult<Vec<_>>>()?;
        let difference = reconstruction
            .iter()
            .zip(axis_values)
            .map(|(left, right)| left - right)
            .collect::<Vec<_>>();
        if norm(&difference)? > tolerance * norm(axis_values)?.max(1.0) {
            return Err(BrainError::Integrity(
                "receiver_basis_source_mixture_mismatch".into(),
            ));
        }
    }
    let reconstructed = tomography.coefficients.matmul(&Matrix::from_rows(&axes)?)?;
    let reconstruction_rms = stable_rms(
        reconstructed
            .as_slice()
            .iter()
            .zip(deltas.as_slice())
            .map(|(actual, expected)| actual - expected),
    )?;
    if (reconstruction_rms - tomography.reconstruction_rms).abs() > tolerance * raw_rms {
        return Err(BrainError::Integrity(
            "receiver_basis_reconstruction_rms_mismatch".into(),
        ));
    }
    let retained_energy = 1.0 - (reconstruction_rms / raw_rms).powi(2);
    if !retained_energy.is_finite()
        || retained_energy < -tolerance
        || retained_energy > 1.0 + tolerance
    {
        return Err(BrainError::Numerical(
            "receiver_basis_raw_retained_energy_invalid".into(),
        ));
    }
    let resolution = resolution_map(
        &tomography.fields,
        &tomography.coefficients,
        &tomography.robust_weights,
        reconstruction_rms,
        input.ridge,
        input.min_signal_to_noise,
    )?;
    if ![
        resolution.field_geometry_condition,
        resolution.excitation_condition,
        resolution.min_principal_angle_degrees,
    ]
    .iter()
    .all(|value| value.is_finite())
        || resolution
            .posterior_covariance
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        || resolution.fields.iter().any(|field| {
            !field.coefficient_rms.is_finite()
                || !field.posterior_std.is_finite()
                || !field.signal_to_posterior_noise.is_finite()
        })
    {
        return Err(BrainError::Numerical(
            "receiver_basis_resolution_nonfinite".into(),
        ));
    }
    Ok(ReceiverBasisBenchmarkReport {
        schema: "cerebro.tidex.receiver_basis_benchmark_report/v1".into(),
        calibration_capability_ids: input.calibration_capability_ids.clone(),
        parameter_dimension: p,
        selected_rank: rank,
        axes,
        coordinates,
        source_mixtures: tomography.source_mixtures,
        reconstruction_rms,
        retained_energy,
        effective_rank: tomography.effective_rank,
        robust_weights: tomography.robust_weights,
        resolution,
        candidate_only: true,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverCompilation {
    pub schema: String,
    pub receiver_parameter_dimension: usize,
    pub calibration_anchor_count: usize,
    pub target_delta: Vec<f64>,
    pub predicted_functional_signature: Vec<f64>,
    pub decoder_loo_r2: f64,
    pub decoder_min_loo_cosine: f64,
    pub encoder_loo_r2: f64,
    pub encoder_min_loo_cosine: f64,
    pub functional_relative_error: f64,
    pub correct_cosine: f64,
    pub maximum_wrong_cosine: f64,
    pub identity_margin: f64,
    pub protection_damage_ratio: f64,
    pub protection_removed_energy: f64,
    pub protection_max_weighted_residual: f64,
    pub trust_region: TrustRegionResult,
    pub operational_verification: OperationalInterfaceVerification,
    pub allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverPortabilityMetrics {
    pub schema: String,
    pub virgin_score: f64,
    pub direct_score: f64,
    pub transferred_score: f64,
    pub wrong_score: f64,
    pub recovered_gain: f64,
    pub correct_wrong_advantage: f64,
}

fn validate_rows(
    rows: &[Vec<f64>],
    expected_dim: Option<usize>,
    label: &str,
) -> BrainResult<usize> {
    if rows.is_empty() {
        return Err(BrainError::Invalid(format!("{label}_empty")));
    }
    let dimension = expected_dim.unwrap_or(rows[0].len());
    if dimension == 0
        || rows
            .iter()
            .any(|row| row.len() != dimension || row.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Invalid(format!("{label}_shape")));
    }
    Ok(dimension)
}

/// Numerical candidate in an explicitly calibrated response space.
///
/// The inverse prediction is NOT execution evidence. Callers must bind the
/// response protocol and coordinate basis and evaluate the materialized model
/// independently. `allowed` means numerical gates passed, not promotion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverSignatureCompilation {
    pub schema: String,
    pub proposal_method: ReceiverProposalMethod,
    pub validation_profile: ReceiverProposalValidationProfile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_fit_diagnostics: Option<AffineTransportDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_fit_diagnostics: Option<AffineTransportDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_functional_anchor_separation: Option<f64>,
    pub proposed_receiver_coordinates: Vec<f64>,
    pub proposal_within_calibrated_support: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relational_source_projection_cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relational_coefficient_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relational_min_loo_source_cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relational_max_loo_coefficient_norm: Option<f64>,
    pub receiver_parameter_dimension: usize,
    pub calibration_anchor_count: usize,
    pub target_delta: Vec<f64>,
    pub predicted_functional_signature: Vec<f64>,
    pub decoder_loo_r2: f64,
    pub decoder_min_loo_cosine: f64,
    pub encoder_loo_r2: f64,
    pub encoder_min_loo_cosine: f64,
    pub functional_relative_error: f64,
    pub correct_cosine: f64,
    pub maximum_wrong_cosine: f64,
    pub identity_margin: f64,
    pub protection_damage_ratio: f64,
    pub protection_removed_energy: f64,
    pub protection_max_weighted_residual: f64,
    pub trust_region: TrustRegionResult,
    pub allowed: bool,
}

/// Compile one held-out operational capability into receiver-native parameters.
/// The operational profile retains its historical wire schema and gates.
pub fn compile_receiver_capability(
    ir: &CapabilityIr,
    operational: &OperationalCapabilityContract,
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
) -> BrainResult<ReceiverCompilation> {
    policy.validate()?;
    operational.validate_against(ir)?;
    let requested = operational.canonical_transition_signature(ir)?;
    let numerical = compile_receiver_signature(
        &requested,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
    )?;
    finish_operational_compilation(ir, operational, numerical)
}

fn finish_operational_compilation(
    ir: &CapabilityIr,
    operational: &OperationalCapabilityContract,
    numerical: ReceiverSignatureCompilation,
) -> BrainResult<ReceiverCompilation> {
    let operational_verification =
        operational.verify_receiver_signature(ir, &numerical.predicted_functional_signature)?;
    Ok(ReceiverCompilation {
        schema: "cerebro.tidex.receiver_compilation/v1".into(),
        receiver_parameter_dimension: numerical.receiver_parameter_dimension,
        calibration_anchor_count: numerical.calibration_anchor_count,
        target_delta: numerical.target_delta,
        predicted_functional_signature: numerical.predicted_functional_signature,
        decoder_loo_r2: numerical.decoder_loo_r2,
        decoder_min_loo_cosine: numerical.decoder_min_loo_cosine,
        encoder_loo_r2: numerical.encoder_loo_r2,
        encoder_min_loo_cosine: numerical.encoder_min_loo_cosine,
        functional_relative_error: numerical.functional_relative_error,
        correct_cosine: numerical.correct_cosine,
        maximum_wrong_cosine: numerical.maximum_wrong_cosine,
        identity_margin: numerical.identity_margin,
        protection_damage_ratio: numerical.protection_damage_ratio,
        protection_removed_energy: numerical.protection_removed_energy,
        protection_max_weighted_residual: numerical.protection_max_weighted_residual,
        trust_region: numerical.trust_region,
        allowed: numerical.allowed && operational_verification.allowed,
        operational_verification,
    })
}

/// An authenticated executable fragment and the calibrated observation map.
/// Donor parameters are used only to execute that fragment. They are never
/// copied or projected as receiver parameters.
pub struct ReceiverReadoutCapabilityInput<'a> {
    pub ir: &'a CapabilityIr,
    pub envelope: &'a SystemEnvelope,
    pub readout_weights: &'a [f64],
    pub inputs: &'a [Vec<f64>],
    pub projection_mean: &'a [f64],
    pub projection_components: &'a [Vec<f64>],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverReadoutCompilation {
    pub execution: LinearReadoutExecution,
    pub requested_signature: Vec<f64>,
    pub numerical: ReceiverSignatureCompilation,
}

/// Execute CapabilityIr before deriving the request to the receiver backend.
/// The learned inverse remains a prediction; materialized receiver execution
/// is a separate authority and is not claimed by this report.
pub fn compile_receiver_readout_capability(
    input: &ReceiverReadoutCapabilityInput<'_>,
    calibration: &ReceiverCalibrationSet,
    cortex: &ProtectedCortex,
    risk: &Matrix,
    policy: &ReceiverCompilerPolicy,
    method: ReceiverProposalMethod,
) -> BrainResult<ReceiverReadoutCompilation> {
    if input.projection_components.len() > 128 || input.projection_mean.len() > 128 {
        return Err(BrainError::Invalid(
            "receiver_readout_projection_budget".into(),
        ));
    }
    let execution = execute_linear_readout(
        input.ir,
        input.envelope,
        input.readout_weights,
        input.inputs,
    )?;
    let requested_signature = project_functional_signature(
        &execution.raw_margins,
        input.projection_mean,
        input.projection_components,
    )?;
    let numerical = compile_signature_with_method_and_validation(
        &requested_signature,
        calibration,
        cortex,
        risk,
        policy,
        method,
        ReceiverProposalValidationProfile::ParametricCrossValidation,
    )?;
    Ok(ReceiverReadoutCompilation {
        execution,
        requested_signature,
        numerical,
    })
}

/// Canonical scalar f64 projection shared by acquisition and compilation.
/// Preserve the sequential subtract/multiply/add order; do not fuse it.
pub(crate) fn project_functional_signature(
    raw: &[f64],
    mean: &[f64],
    components: &[Vec<f64>],
) -> BrainResult<Vec<f64>> {
    if raw.is_empty()
        || raw.len() != mean.len()
        || raw.iter().chain(mean).any(|value| !value.is_finite())
        || components.is_empty()
        || components
            .iter()
            .any(|row| row.len() != raw.len() || row.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Invalid(
            "receiver_weight_functional_projection_shape".into(),
        ));
    }
    let mut projected = Vec::with_capacity(components.len());
    for component in components {
        let mut sum = 0.0_f64;
        for ((raw_value, mean_value), coefficient) in raw.iter().zip(mean).zip(component) {
            let centered = *raw_value - *mean_value;
            let product = centered * *coefficient;
            sum += product;
        }
        if !sum.is_finite() {
            return Err(BrainError::Invalid(
                "receiver_weight_functional_projection_non_finite".into(),
            ));
        }
        projected.push(sum);
    }
    Ok(projected)
}

/// Explicit numerical profile: no failure-triggered fallback is permitted.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverProposalMethod {
    DecodeThenProject,
    /// Centered affine decoder and inverse, with separately trace-scaled ridge
    /// in every training fold. The policy ridge is dimensionless in this profile.
    CalibratedAffine,
    FitProtectedCoordinates,
    RelationalAnchors,
}

/// Distinguishes the legacy fully parametric cross-validation gate from a
/// candidate-only cross-model route whose proposal quality is authorized by a
/// separately authenticated behavioral leave-one-capability-out calibration.
/// The latter never authorizes promotion by itself; the binding must verify the
/// external calibration evidence before it may use this profile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverProposalValidationProfile {
    ParametricCrossValidation,
    BehavioralCalibrationMeasurement,
    AuthenticatedBehavioralCalibration,
}

/// Fit the requested response through the actual protection operator P:
/// min_z ||M P z - (requested - bias)||^2 + ridge ||z||^2.
/// P is constructed by applying the existing, fixed linear protection map to
/// coordinate unit vectors. Hard protected directions cannot be recovered by
/// this solve. The final proposal must STILL pass protection.allowed, the same
/// quadratic risk budget, support bounds and the resulting response residual.
/// Soft braking is a preconditioner here, not a promise of a fixed shrinkage of
/// the naive decoder; empirical safety is assessed on the final candidate.
fn safe_coordinate_inverse(
    encoder: &TransportMap,
    requested: &[f64],
    cortex: &ProtectedCortex,
    ridge: f64,
) -> BrainResult<Vec<f64>> {
    let k = encoder.source_dim;
    let mut columns = Vec::with_capacity(k);
    for index in 0..k {
        let mut unit = vec![0.0; k];
        unit[index] = 1.0;
        columns.push(project_to_safe_subspace(&unit, cortex)?.projected);
    }
    let protection = Matrix::from_rows(&columns)?.transpose();
    let design = encoder.weights.matmul(&protection)?;
    let targets = requested
        .iter()
        .zip(&encoder.bias)
        .map(|(requested, bias)| requested - bias)
        .collect::<Vec<_>>();
    weighted_normal_solve(&design, &targets, &vec![1.0; targets.len()], ridge)
}

#[derive(Debug, Clone)]
struct RelationalReceiverProposal {
    coordinates: Vec<f64>,
    within_calibrated_support: bool,
    source_projection_cosine: Option<f64>,
    coefficient_norm: f64,
    min_loo_source_cosine: f64,
    max_loo_coefficient_norm: f64,
}

/// Reuse the capability relations discovered in functional space rather than
/// fitting an unconstrained global affine extrapolator.  The relational map
/// derives coefficients only from calibration functional anchors; those same
/// coefficients are then applied to the *raw* receiver coordinates, preserving
/// receiver magnitude while keeping target receiver parameters completely
/// absent from the solve.
fn relational_receiver_proposal(
    calibration: &ReceiverCalibrationSet,
    requested: &[f64],
    ridge: f64,
) -> BrainResult<RelationalReceiverProposal> {
    let map = learn_relational_transport(
        &calibration.functional_signatures,
        &calibration.receiver_solutions,
        ridge,
    )?;
    let transplant = map.transplant(requested)?;
    if transplant.target_coefficients.len() != calibration.receiver_solutions.len() {
        return Err(BrainError::Integrity(
            "receiver_compiler_relational_coefficient_count".into(),
        ));
    }
    let receiver_dim = calibration.receiver_solutions[0].len();
    let mut coordinates = vec![0.0; receiver_dim];
    for (coefficient, anchor) in transplant
        .target_coefficients
        .iter()
        .zip(&calibration.receiver_solutions)
    {
        if anchor.len() != receiver_dim {
            return Err(BrainError::Invalid(
                "receiver_compiler_relational_receiver_shape".into(),
            ));
        }
        for index in 0..receiver_dim {
            coordinates[index] += coefficient * anchor[index];
        }
    }
    if coordinates.iter().any(|value| !value.is_finite()) {
        return Err(BrainError::Numerical(
            "receiver_compiler_relational_nonfinite_coordinates".into(),
        ));
    }
    Ok(RelationalReceiverProposal {
        coordinates,
        within_calibrated_support: transplant.resolved,
        source_projection_cosine: transplant.source_projection_cosine,
        coefficient_norm: transplant.coefficient_norm,
        min_loo_source_cosine: map.min_loo_source_cosine,
        max_loo_coefficient_norm: map.max_loo_coefficient_norm,
    })
}

/// Shared decoder, inverse predictor, protection and trust-region kernel.
///
/// This entry point does not fabricate an OperationalCapabilityContract for a
/// generative model. A model binding must supply a separately sealed response
/// protocol and enforce calibration/target separation. All output is candidate
/// evidence; the learned encoder is only a prediction of receiver behavior.
pub fn compile_receiver_signature(
    requested: &[f64],
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
) -> BrainResult<ReceiverSignatureCompilation> {
    compile_signature_with_method_and_validation(
        requested,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
        ReceiverProposalMethod::DecodeThenProject,
        ReceiverProposalValidationProfile::ParametricCrossValidation,
    )
}

/// Compile a measured functional IR with scale-aware, fold-local regression.
/// This profile always retains both decoder and inverse validation. It does not
/// turn behavioral measurements into authority to skip numerical gates.
pub fn compile_receiver_signature_calibrated_affine(
    requested: &[f64],
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
) -> BrainResult<ReceiverSignatureCompilation> {
    compile_signature_with_method_and_validation(
        requested,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
        ReceiverProposalMethod::CalibratedAffine,
        ReceiverProposalValidationProfile::ParametricCrossValidation,
    )
}

/// Distinct calibration capabilities must be distinguishable in their input IR.
/// The tolerance concerns floating-point resolution, not an invented noise model.
/// A single zero/constant-coordinate vector is valid; coincident rows are not
/// independent evidence. No coordinate-specific variance heuristic is used.
fn validate_functional_anchor_identity(rows: &[Vec<f64>]) -> BrainResult<f64> {
    let mut minimum = f64::INFINITY;
    for i in 0..rows.len() {
        for j in 0..i {
            let difference = rows[i]
                .iter()
                .zip(&rows[j])
                .map(|(left, right)| left - right)
                .collect::<Vec<_>>();
            let scale = norm(&rows[i])?.max(norm(&rows[j])?).max(f64::MIN_POSITIVE);
            let relative = norm(&difference)? / scale;
            if !relative.is_finite()
                || relative <= 16.0 * f64::EPSILON * rows[i].len().max(1) as f64
            {
                return Err(BrainError::Invalid(
                    "receiver_compiler_functional_identity_collision".into(),
                ));
            }
            minimum = minimum.min(relative);
        }
    }
    Ok(minimum)
}

/// Candidate-only cross-model affine compilation. This deliberately does NOT
/// interpret decoder coordinate-space LOO R² as the final proposal-quality
/// authority. The caller must first authenticate an independent behavioral
/// leave-one-capability-out calibration over exactly the receiver basis lineage.
/// Encoder verification, functional residual, identity, protection, trust
/// region and support gates remain mandatory.
pub fn compile_receiver_signature_behaviorally_calibrated_candidate(
    requested: &[f64],
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
) -> BrainResult<ReceiverSignatureCompilation> {
    compile_signature_with_method_and_validation(
        requested,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
        ReceiverProposalMethod::DecodeThenProject,
        ReceiverProposalValidationProfile::AuthenticatedBehavioralCalibration,
    )
}

/// Response-space inversion that includes protection in the forward design.
/// This is an explicitly selected candidate profile, not a fallback used to
/// turn a rejected legacy compilation into an accepted one. Legacy operational
/// compilation retains DecodeThenProject and its original serialized contract.
pub fn compile_receiver_signature_in_safe_coordinates(
    requested: &[f64],
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
) -> BrainResult<ReceiverSignatureCompilation> {
    compile_signature_with_method(
        requested,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
        ReceiverProposalMethod::FitProtectedCoordinates,
    )
}

/// Compile from relational capability geometry.  The target contributes only
/// its functional signature; barycentric coefficients are inferred against the
/// calibration functional anchors and then applied to receiver coordinates of
/// those same calibration capabilities.
pub fn compile_receiver_signature_relational(
    requested: &[f64],
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
) -> BrainResult<ReceiverSignatureCompilation> {
    compile_signature_with_method(
        requested,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
        ReceiverProposalMethod::RelationalAnchors,
    )
}

fn compile_signature_with_method(
    requested: &[f64],
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
    proposal_method: ReceiverProposalMethod,
) -> BrainResult<ReceiverSignatureCompilation> {
    compile_signature_with_method_and_validation(
        requested,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
        proposal_method,
        ReceiverProposalValidationProfile::ParametricCrossValidation,
    )
}

/// Target-independent learned state. Its maps are produced by the existing
/// transport solvers, not by a second compilation implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ReceiverCompilerMaps {
    decoder: crate::transport::FunctionalTransplantMap,
    encoder: crate::transport::ValidatedTransportMap,
    decoder_fit_diagnostics: Option<AffineTransportDiagnostics>,
    encoder_fit_diagnostics: Option<AffineTransportDiagnostics>,
    minimum_functional_anchor_separation: Option<f64>,
}

#[cfg(test)]
std::thread_local! {
    static COMPILER_FIT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn fit_receiver_compiler_maps(
    calibration: &ReceiverCalibrationSet,
    policy: &ReceiverCompilerPolicy,
    proposal_method: ReceiverProposalMethod,
) -> BrainResult<ReceiverCompilerMaps> {
    #[cfg(test)]
    COMPILER_FIT_CALLS.with(|count| count.set(count.get() + 1));
    let calibrated_affine = proposal_method == ReceiverProposalMethod::CalibratedAffine;
    let minimum_functional_anchor_separation = if calibrated_affine {
        Some(validate_functional_anchor_identity(
            &calibration.functional_signatures,
        )?)
    } else {
        None
    };
    // Each direction and each LOO fold retains its own training-only scale.
    let (decoder, decoder_fit_diagnostics, encoder, encoder_fit_diagnostics) = if calibrated_affine
    {
        let regression = AffineTransportPolicy::CenteredTraceRidge {
            relative_ridge: policy.ridge,
        };
        let (decoder, decoder_diagnostics) = learn_functional_transplant_with_policy(
            &calibration.functional_signatures,
            &calibration.receiver_solutions,
            &regression,
        )?;
        let (encoder, encoder_diagnostics) = learn_transport_validated_with_policy(
            &calibration.receiver_solutions,
            &calibration.functional_signatures,
            &regression,
        )?;
        (
            decoder,
            Some(decoder_diagnostics),
            encoder,
            Some(encoder_diagnostics),
        )
    } else {
        let decoder = learn_functional_transplant(
            &calibration.functional_signatures,
            &calibration.receiver_solutions,
            policy.ridge,
        )?;
        let encoder = learn_transport_validated(
            &calibration.receiver_solutions,
            &calibration.functional_signatures,
            policy.ridge,
        )?;
        (decoder, None, encoder, None)
    };
    Ok(ReceiverCompilerMaps {
        decoder,
        encoder,
        decoder_fit_diagnostics,
        encoder_fit_diagnostics,
        minimum_functional_anchor_separation,
    })
}

/// This input has no target. Wrong signatures belong to a later query and must
/// be empty here. Receiver identity, calibration lineage, method and safety
/// policy are all part of the frozen artifact's identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenReceiverCompilerInput {
    pub schema: String,
    pub calibration_capability_ids: Vec<crate::identity::CapabilityId>,
    pub calibration: ReceiverCalibrationSet,
    pub protected_cortex: ProtectedCortex,
    pub risk_metric_rows: Vec<Vec<f64>>,
    pub policy: ReceiverCompilerPolicy,
    pub proposal_method: ReceiverProposalMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenReceiverCompiler {
    schema: String,
    input: FrozenReceiverCompilerInput,
    maps: ReceiverCompilerMaps,
    maximum_functional_leverage: f64,
    manifest_sha256: Sha256Digest,
}

/// Non-deserializable, read-only handle. Only replay verification can construct
/// it. Verifying a serialized compiler replays calibration; applying this handle
/// to any number of targets performs no map fitting and no receiver training.
pub struct VerifiedFrozenReceiverCompiler<'a> {
    frozen: &'a FrozenReceiverCompiler,
}

pub fn freeze_receiver_compiler(
    input: &FrozenReceiverCompilerInput,
) -> BrainResult<FrozenReceiverCompiler> {
    input.policy.validate()?;
    let calibration = &input.calibration;
    let n = calibration.functional_signatures.len();
    let f = calibration
        .functional_signatures
        .first()
        .map_or(0, Vec::len);
    let k = calibration.receiver_solutions.first().map_or(0, Vec::len);
    if input.schema != "cerebro.tidex.frozen_receiver_compiler_input/v1"
        || !(5..=256).contains(&n)
        || !(1..=256).contains(&f)
        || !(1..=256).contains(&k)
        || calibration.receiver_solutions.len() != n
        || !calibration.wrong_functional_signatures.is_empty()
        || calibration
            .receiver_snapshot_binding_sha256
            .as_ref()
            .is_none_or(|value| *value == Sha256Digest::zero())
        || input.calibration_capability_ids.len() != n
        || input
            .calibration_capability_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != n
        || input.protected_cortex.parameter_importance.len() != k
        || input.protected_cortex.directions.len() > 256
        || input.risk_metric_rows.len() != k
        || input.risk_metric_rows.iter().any(|row| row.len() != k)
        || !matches!(
            input.proposal_method,
            ReceiverProposalMethod::DecodeThenProject | ReceiverProposalMethod::CalibratedAffine
        )
    {
        return Err(BrainError::Invalid(
            "frozen_receiver_compiler_input_invalid".into(),
        ));
    }
    // Preserve the established cubic-work ceiling before invoking any solver.
    let work = (n as u128 + 1)
        * ((k as u128 + 1) * (f as u128 + 1).pow(3) + (f as u128 + 1) * (k as u128 + 1).pow(3));
    if work > 250_000_000 {
        return Err(BrainError::Invalid(
            "frozen_receiver_compiler_work_limit".into(),
        ));
    }
    validate_rows(
        &calibration.functional_signatures,
        Some(f),
        "frozen_functional",
    )?;
    validate_rows(&calibration.receiver_solutions, Some(k), "frozen_receiver")?;
    // Distinct labels must not turn identical (or numerically indistinguishable)
    // functional observations into independent calibration capabilities. This
    // applies to every frozen method, not only centered affine regression.
    validate_functional_anchor_identity(&calibration.functional_signatures)?;
    let risk_metric = Matrix::from_rows(&input.risk_metric_rows)?;
    let zero = vec![0.0; k];
    project_to_safe_subspace(&zero, &input.protected_cortex)?;
    apply_quadratic_trust_region(&risk_metric, &zero, input.policy.maximum_quadratic_cost)?;
    let maps = fit_receiver_compiler_maps(calibration, &input.policy, input.proposal_method)?;
    if !maps.decoder.resolved
        || !maps.encoder.resolved
        || maps.decoder.loo_cv_r2 < input.policy.minimum_decoder_loo_r2
        || maps.encoder.loo_cv_r2 < input.policy.minimum_encoder_loo_r2
        || maps.decoder.min_loo_cosine < input.policy.minimum_decoder_loo_cosine
    {
        return Err(BrainError::Integrity(
            "frozen_receiver_calibration_gates_failed".into(),
        ));
    }
    let (_, maximum_functional_leverage) = crate::transport::functional_support_envelope(
        &calibration.functional_signatures,
        &calibration.functional_signatures[0],
        input.policy.ridge,
    )?;
    let mut frozen = FrozenReceiverCompiler {
        schema: "cerebro.tidex.frozen_receiver_compiler/v1".into(),
        input: input.clone(),
        maps,
        maximum_functional_leverage,
        manifest_sha256: Sha256Digest::zero(),
    };
    frozen.manifest_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:FROZEN-RECEIVER-COMPILER:v1\0",
        &serde_json::to_vec(&frozen)?,
    );
    Ok(frozen)
}

impl FrozenReceiverCompiler {
    pub fn manifest_sha256(&self) -> &Sha256Digest {
        &self.manifest_sha256
    }

    pub fn input(&self) -> &FrozenReceiverCompilerInput {
        &self.input
    }

    /// A coherent caller-supplied hash is not proof of correct learned maps.
    /// Refit only for this explicit authentication step and compare exact wire
    /// bytes, including diagnostics and signed zero. Never replace stored maps.
    pub fn verify(&self) -> BrainResult<VerifiedFrozenReceiverCompiler<'_>> {
        let replay = freeze_receiver_compiler(&self.input)?;
        if serde_json::to_vec(&replay)? != serde_json::to_vec(self)? {
            return Err(BrainError::Integrity(
                "frozen_receiver_compiler_replay_mismatch".into(),
            ));
        }
        Ok(VerifiedFrozenReceiverCompiler { frozen: self })
    }

    pub fn validate_binding(
        &self,
        calibration: &ReceiverCalibrationSet,
        protected_cortex: &ProtectedCortex,
        risk_metric: &Matrix,
        policy: &ReceiverCompilerPolicy,
    ) -> BrainResult<()> {
        let input = &self.input;
        if input.calibration.receiver_snapshot_binding_sha256
            != calibration.receiver_snapshot_binding_sha256
            || input.calibration.functional_signatures != calibration.functional_signatures
            || input.calibration.receiver_solutions != calibration.receiver_solutions
            || input.protected_cortex != *protected_cortex
            || Matrix::from_rows(&input.risk_metric_rows)? != *risk_metric
            || input.policy != *policy
        {
            return Err(BrainError::Integrity(
                "frozen_receiver_compiler_binding_mismatch".into(),
            ));
        }
        Ok(())
    }
}

impl VerifiedFrozenReceiverCompiler<'_> {
    pub fn compile_capability(
        &self,
        ir: &CapabilityIr,
        operational: &OperationalCapabilityContract,
        wrong_functional_signatures: &[Vec<f64>],
    ) -> BrainResult<ReceiverCompilation> {
        operational.validate_against(ir)?;
        let requested = operational.canonical_transition_signature(ir)?;
        let numerical =
            self.compile(ir.capability_id(), &requested, wrong_functional_signatures)?;
        finish_operational_compilation(ir, operational, numerical)
    }

    /// Apply the already-verified maps. All target-specific preservation,
    /// inverse, identity and trust-region gates still run in the shared core.
    pub fn compile(
        &self,
        capability_id: &crate::identity::CapabilityId,
        requested: &[f64],
        wrong_functional_signatures: &[Vec<f64>],
    ) -> BrainResult<ReceiverSignatureCompilation> {
        let input = &self.frozen.input;
        if input.calibration_capability_ids.contains(capability_id)
            || input
                .calibration
                .functional_signatures
                .iter()
                .any(|row| row == requested)
        {
            return Err(BrainError::Integrity(
                "frozen_receiver_target_leaked_into_calibration".into(),
            ));
        }
        if wrong_functional_signatures.len() > 256 {
            return Err(BrainError::Invalid(
                "frozen_receiver_wrong_signature_limit".into(),
            ));
        }
        if requested.len() != input.calibration.functional_signatures[0].len()
            || requested.iter().any(|value| !value.is_finite())
        {
            return Err(BrainError::Invalid("frozen_receiver_query_shape".into()));
        }
        // The same floating-point identity criterion used during freezing also
        // applies to the target. A relabel plus a rounding-sized perturbation
        // is not evidence of a held-out capability.
        norm(requested)?;
        let mut identities = input.calibration.functional_signatures.clone();
        identities.push(requested.to_vec());
        validate_functional_anchor_identity(&identities).map_err(|_| {
            BrainError::Integrity("frozen_receiver_target_leaked_into_calibration".into())
        })?;
        let mut calibration = input.calibration.clone();
        calibration.wrong_functional_signatures = wrong_functional_signatures.to_vec();
        let mut numerical = compile_signature_with_maps(
            requested,
            &calibration,
            &input.protected_cortex,
            &Matrix::from_rows(&input.risk_metric_rows)?,
            &input.policy,
            input.proposal_method,
            ReceiverProposalValidationProfile::ParametricCrossValidation,
            Some(&self.frozen.maps),
        )?;
        let leverage = crate::transport::functional_leverage(
            &input.calibration.functional_signatures,
            requested,
            input.policy.ridge,
        )?;
        numerical.proposal_within_calibrated_support =
            leverage <= self.frozen.maximum_functional_leverage * (1.0 + 1e-10);
        numerical.allowed &= numerical.proposal_within_calibrated_support;
        Ok(numerical)
    }
}

fn compile_signature_with_method_and_validation(
    requested: &[f64],
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
    proposal_method: ReceiverProposalMethod,
    validation_profile: ReceiverProposalValidationProfile,
) -> BrainResult<ReceiverSignatureCompilation> {
    compile_signature_with_maps(
        requested,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
        proposal_method,
        validation_profile,
        None,
    )
}

fn compile_signature_with_maps(
    requested: &[f64],
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
    proposal_method: ReceiverProposalMethod,
    validation_profile: ReceiverProposalValidationProfile,
    fitted: Option<&ReceiverCompilerMaps>,
) -> BrainResult<ReceiverSignatureCompilation> {
    policy.validate()?;
    if matches!(
        validation_profile,
        ReceiverProposalValidationProfile::BehavioralCalibrationMeasurement
            | ReceiverProposalValidationProfile::AuthenticatedBehavioralCalibration
    ) && proposal_method != ReceiverProposalMethod::DecodeThenProject
    {
        return Err(BrainError::Invalid(
            "receiver_compiler_behavioral_profile_requires_affine_proposal".into(),
        ));
    }
    // Bound the current dense affine solver BEFORE constructing its matrices.
    let n = calibration.functional_signatures.len();
    let f = requested.len();
    let k = calibration.receiver_solutions.first().map_or(0, Vec::len);
    if n > 256 || f > 256 || k > 256 {
        return Err(BrainError::Invalid(
            "receiver_compiler_resource_budget_exceeded".into(),
        ));
    }
    let estimated_work = (n as u128 + 1)
        * ((k as u128 + 1) * (f as u128 + 1).pow(3) + (f as u128 + 1) * (k as u128 + 1).pow(3));
    if estimated_work > 250_000_000 {
        return Err(BrainError::Invalid(
            "receiver_compiler_resource_budget_exceeded".into(),
        ));
    }
    if norm(requested)? <= 1e-15 {
        return Err(BrainError::Invalid(
            "receiver_compiler_query_degenerate".into(),
        ));
    }
    if calibration.functional_signatures.len() != calibration.receiver_solutions.len()
        || calibration.functional_signatures.len() < 5
    {
        return Err(BrainError::Invalid(
            "receiver_compiler_calibration_count".into(),
        ));
    }
    validate_rows(
        &calibration.functional_signatures,
        Some(requested.len()),
        "receiver_compiler_functional_anchors",
    )?;
    let receiver_dim = validate_rows(
        &calibration.receiver_solutions,
        None,
        "receiver_compiler_receiver_anchors",
    )?;
    if calibration.wrong_functional_signatures.is_empty() {
        return Err(BrainError::Invalid(
            "receiver_compiler_wrong_skill_set_empty".into(),
        ));
    }
    validate_rows(
        &calibration.wrong_functional_signatures,
        Some(requested.len()),
        "receiver_compiler_wrong_signatures",
    )?;
    if calibration
        .wrong_functional_signatures
        .iter()
        .any(|signature| {
            norm(signature).is_err() || norm(signature).is_ok_and(|value| value <= 1e-15)
        })
    {
        return Err(BrainError::Invalid(
            "receiver_compiler_wrong_signature_degenerate".into(),
        ));
    }
    if protected_cortex.parameter_importance.len() != receiver_dim
        || risk_metric.rows != receiver_dim
        || risk_metric.cols != receiver_dim
    {
        return Err(BrainError::Invalid("receiver_compiler_safety_shape".into()));
    }

    let freshly_fitted;
    let maps = match fitted {
        Some(maps) => maps,
        None => {
            freshly_fitted = fit_receiver_compiler_maps(calibration, policy, proposal_method)?;
            &freshly_fitted
        }
    };
    let decoder = &maps.decoder;
    let encoder = &maps.encoder;
    let decoder_fit_diagnostics = maps.decoder_fit_diagnostics.clone();
    let encoder_fit_diagnostics = maps.encoder_fit_diagnostics.clone();
    let minimum_functional_anchor_separation = maps.minimum_functional_anchor_separation;
    let (
        proposed,
        proposal_within_calibrated_support,
        relational_source_projection_cosine,
        relational_coefficient_norm,
        relational_min_loo_source_cosine,
        relational_max_loo_coefficient_norm,
    ) = match proposal_method {
        ReceiverProposalMethod::DecodeThenProject | ReceiverProposalMethod::CalibratedAffine => (
            decoder.transplant(requested)?.target_vector,
            true,
            None,
            None,
            None,
            None,
        ),
        ReceiverProposalMethod::FitProtectedCoordinates => (
            safe_coordinate_inverse(&encoder.map, requested, protected_cortex, policy.ridge)?,
            true,
            None,
            None,
            None,
            None,
        ),
        ReceiverProposalMethod::RelationalAnchors => {
            let relational = relational_receiver_proposal(calibration, requested, policy.ridge)?;
            (
                relational.coordinates,
                relational.within_calibrated_support,
                relational.source_projection_cosine,
                Some(relational.coefficient_norm),
                Some(relational.min_loo_source_cosine),
                Some(relational.max_loo_coefficient_norm),
            )
        }
    };

    // Protection precedes trust-region scaling. Uniform scaling cannot
    // reintroduce a component removed by the protected-subspace projection.
    let protection = project_to_safe_subspace(&proposed, protected_cortex)?;
    let trust = apply_quadratic_trust_region(
        risk_metric,
        &protection.projected,
        policy.maximum_quadratic_cost,
    )?;
    let target_delta = trust.accepted_coefficients.clone();
    let predicted = encoder.map.apply(&target_delta)?;

    let residual = predicted
        .iter()
        .zip(requested)
        .map(|(left, right)| left - right)
        .collect::<Vec<_>>();
    let functional_relative_error = norm(&residual)? / norm(requested)?.max(1e-15);
    let correct_cosine = cosine(&predicted, requested)?;
    let maximum_wrong_cosine = calibration
        .wrong_functional_signatures
        .iter()
        .map(|wrong| cosine(&predicted, wrong))
        .collect::<BrainResult<Vec<_>>>()?
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let identity_margin = correct_cosine - maximum_wrong_cosine;

    let proposal_model_allowed = match (proposal_method, validation_profile) {
        (ReceiverProposalMethod::RelationalAnchors, _) => proposal_within_calibrated_support,
        (
            ReceiverProposalMethod::DecodeThenProject,
            ReceiverProposalValidationProfile::BehavioralCalibrationMeasurement
            | ReceiverProposalValidationProfile::AuthenticatedBehavioralCalibration,
        ) => decoder.min_loo_cosine >= policy.minimum_decoder_loo_cosine,
        (
            ReceiverProposalMethod::DecodeThenProject
            | ReceiverProposalMethod::CalibratedAffine
            | ReceiverProposalMethod::FitProtectedCoordinates,
            ReceiverProposalValidationProfile::ParametricCrossValidation,
        ) => {
            decoder.resolved
                && decoder.loo_cv_r2 >= policy.minimum_decoder_loo_r2
                && decoder.min_loo_cosine >= policy.minimum_decoder_loo_cosine
        }
        (
            ReceiverProposalMethod::FitProtectedCoordinates
            | ReceiverProposalMethod::CalibratedAffine,
            ReceiverProposalValidationProfile::BehavioralCalibrationMeasurement
            | ReceiverProposalValidationProfile::AuthenticatedBehavioralCalibration,
        ) => false,
    };
    let verification_model_allowed = match validation_profile {
        ReceiverProposalValidationProfile::BehavioralCalibrationMeasurement => true,
        ReceiverProposalValidationProfile::ParametricCrossValidation
        | ReceiverProposalValidationProfile::AuthenticatedBehavioralCalibration => {
            encoder.resolved && encoder.loo_cv_r2 >= policy.minimum_encoder_loo_r2
        }
    };
    let allowed = proposal_model_allowed
        && verification_model_allowed
        && functional_relative_error <= policy.maximum_functional_relative_error
        && identity_margin >= policy.minimum_identity_margin
        && protection.allowed
        && trust.accepted_quadratic_cost <= policy.maximum_quadratic_cost;

    Ok(ReceiverSignatureCompilation {
        schema: "cerebro.tidex.receiver_signature_compilation/v1".into(),
        proposal_method,
        validation_profile,
        decoder_fit_diagnostics,
        encoder_fit_diagnostics,
        minimum_functional_anchor_separation,
        proposed_receiver_coordinates: proposed,
        proposal_within_calibrated_support,
        relational_source_projection_cosine,
        relational_coefficient_norm,
        relational_min_loo_source_cosine,
        relational_max_loo_coefficient_norm,
        receiver_parameter_dimension: receiver_dim,
        calibration_anchor_count: calibration.functional_signatures.len(),
        target_delta,
        predicted_functional_signature: predicted,
        decoder_loo_r2: decoder.loo_cv_r2,
        decoder_min_loo_cosine: decoder.min_loo_cosine,
        encoder_loo_r2: encoder.loo_cv_r2,
        encoder_min_loo_cosine: encoder.min_loo_cosine,
        functional_relative_error,
        correct_cosine,
        maximum_wrong_cosine,
        identity_margin,
        protection_damage_ratio: protection.damage_ratio,
        protection_removed_energy: protection.removed_energy,
        protection_max_weighted_residual: protection.max_weighted_residual,
        trust_region: trust,
        allowed,
    })
}

/// Compare one receiver-native compiled result against an untouched receiver,
/// a direct receiver oracle, and an explicit wrong-skill control using one
/// common functional score. This metric is intentionally independent of
/// parameter distance. A positive score supports functional recovery only in
/// the declared evaluation domain; it is not, by itself, a universal portability claim.
pub fn evaluate_portability(
    expected_functional_signature: &[f64],
    virgin_functional_signature: &[f64],
    direct_functional_signature: &[f64],
    transferred_functional_signature: &[f64],
    wrong_functional_signature: &[f64],
) -> BrainResult<ReceiverPortabilityMetrics> {
    let dimension = expected_functional_signature.len();
    if dimension == 0
        || expected_functional_signature
            .iter()
            .any(|value| !value.is_finite())
        || norm(expected_functional_signature)? <= 1e-15
    {
        return Err(BrainError::Invalid(
            "receiver_portability_signature_invalid".into(),
        ));
    }
    for signature in [
        virgin_functional_signature,
        direct_functional_signature,
        transferred_functional_signature,
        wrong_functional_signature,
    ] {
        if signature.len() != dimension || signature.iter().any(|value| !value.is_finite()) {
            return Err(BrainError::Invalid(
                "receiver_portability_signature_invalid".into(),
            ));
        }
    }
    let score = |observed: &[f64]| -> BrainResult<f64> {
        let residual = observed
            .iter()
            .zip(expected_functional_signature)
            .map(|(left, right)| left - right)
            .collect::<Vec<_>>();
        Ok(1.0 - norm(&residual)? / norm(expected_functional_signature)?.max(1e-15))
    };
    let virgin_score = score(virgin_functional_signature)?;
    let direct_score = score(direct_functional_signature)?;
    let transferred_score = score(transferred_functional_signature)?;
    let wrong_score = score(wrong_functional_signature)?;
    let direct_gain = direct_score - virgin_score;
    if direct_gain <= 1e-12 {
        return Err(BrainError::Invalid(
            "receiver_portability_direct_oracle_has_no_gain".into(),
        ));
    }
    let recovered_gain = (transferred_score - virgin_score) / direct_gain;
    Ok(ReceiverPortabilityMetrics {
        schema: "cerebro.tidex.receiver_portability_metrics/v1".into(),
        virgin_score,
        direct_score,
        transferred_score,
        wrong_score,
        recovered_gain,
        correct_wrong_advantage: transferred_score - wrong_score,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverPortabilityCase {
    pub skill_id: String,
    pub functional_signature: Vec<f64>,
    pub direct_receiver_solution: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverPortabilityBenchmarkInput {
    pub schema: String,
    pub ridge: f64,
    pub cases: Vec<ReceiverPortabilityCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverPortabilityCaseReport {
    pub skill_id: String,
    pub decoder_loo_r2: f64,
    pub encoder_loo_r2: f64,
    pub decoder_min_loo_cosine: f64,
    pub encoder_min_loo_cosine: f64,
    pub compiled_receiver_solution: Vec<f64>,
    pub wrong_receiver_solution: Vec<f64>,
    pub metrics: ReceiverPortabilityMetrics,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverPortabilityBenchmarkReport {
    pub schema: String,
    pub case_count: usize,
    pub functional_dimension: usize,
    pub receiver_parameter_dimension: usize,
    pub mean_recovered_gain: f64,
    pub minimum_recovered_gain: f64,
    pub mean_correct_wrong_advantage: f64,
    pub minimum_correct_wrong_advantage: f64,
    pub all_resolved: bool,
    pub cases: Vec<ReceiverPortabilityCaseReport>,
}

/// Leave-one-skill-out functional compilation benchmark. The direct receiver
/// solution of the held-out skill is never passed to either learned map; it is
/// opened only after compilation as an oracle for `RecoveredGain`.
///
/// The retained `portability` schema/API name is historical compatibility. The
/// benchmark measures held-out recovery inside its supplied calibration set; it
/// does not establish portability across arbitrary model architectures.
pub fn benchmark_receiver_portability_leave_one_out(
    input: &ReceiverPortabilityBenchmarkInput,
) -> BrainResult<ReceiverPortabilityBenchmarkReport> {
    if input.schema != "cerebro.tidex.receiver_portability_benchmark_input/v1"
        || !input.ridge.is_finite()
        || input.ridge <= 0.0
        || input.cases.len() < 6
    {
        return Err(BrainError::Invalid(
            "receiver_portability_benchmark_input_invalid".into(),
        ));
    }
    let functional_dim = input.cases[0].functional_signature.len();
    let receiver_dim = input.cases[0].direct_receiver_solution.len();
    if functional_dim == 0 || receiver_dim == 0 {
        return Err(BrainError::Invalid(
            "receiver_portability_benchmark_dimension_invalid".into(),
        ));
    }
    let mut skill_ids = std::collections::BTreeSet::new();
    for case in &input.cases {
        if case.skill_id.is_empty()
            || case.skill_id.len() > 256
            || !skill_ids.insert(case.skill_id.as_str())
            || case.functional_signature.len() != functional_dim
            || case.direct_receiver_solution.len() != receiver_dim
            || case
                .functional_signature
                .iter()
                .any(|value| !value.is_finite())
            || case
                .direct_receiver_solution
                .iter()
                .any(|value| !value.is_finite())
            || norm(&case.functional_signature)? <= 1e-15
        {
            return Err(BrainError::Invalid(
                "receiver_portability_benchmark_case_invalid".into(),
            ));
        }
    }

    let mut reports = Vec::with_capacity(input.cases.len());
    for holdout in 0..input.cases.len() {
        let training_functional = input
            .cases
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != holdout)
            .map(|(_, case)| case.functional_signature.clone())
            .collect::<Vec<_>>();
        let training_receiver = input
            .cases
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != holdout)
            .map(|(_, case)| case.direct_receiver_solution.clone())
            .collect::<Vec<_>>();
        let decoder =
            learn_functional_transplant(&training_functional, &training_receiver, input.ridge)?;
        let encoder =
            learn_transport_validated(&training_receiver, &training_functional, input.ridge)?;
        let query = &input.cases[holdout];
        let transferred_delta = decoder
            .transplant(&query.functional_signature)?
            .target_vector;
        let transferred_behavior = encoder.map.apply(&transferred_delta)?;
        let direct_behavior = encoder.map.apply(&query.direct_receiver_solution)?;
        let virgin_behavior = encoder.map.apply(&vec![0.0; receiver_dim])?;

        // Deterministic wrong-skill control chosen from the calibration set,
        // never from the held-out case itself.
        let wrong_index = if holdout == 0 { 1 } else { 0 };
        let wrong_delta = decoder
            .transplant(&input.cases[wrong_index].functional_signature)?
            .target_vector;
        let wrong_behavior = encoder.map.apply(&wrong_delta)?;
        let metrics = evaluate_portability(
            &query.functional_signature,
            &virgin_behavior,
            &direct_behavior,
            &transferred_behavior,
            &wrong_behavior,
        )?;
        reports.push(ReceiverPortabilityCaseReport {
            skill_id: query.skill_id.clone(),
            decoder_loo_r2: decoder.loo_cv_r2,
            encoder_loo_r2: encoder.loo_cv_r2,
            decoder_min_loo_cosine: decoder.min_loo_cosine,
            encoder_min_loo_cosine: encoder.min_loo_cosine,
            compiled_receiver_solution: transferred_delta,
            wrong_receiver_solution: wrong_delta,
            resolved: decoder.resolved && encoder.resolved,
            metrics,
        });
    }
    let mean_recovered_gain = reports
        .iter()
        .map(|report| report.metrics.recovered_gain)
        .sum::<f64>()
        / reports.len() as f64;
    let minimum_recovered_gain = reports
        .iter()
        .map(|report| report.metrics.recovered_gain)
        .fold(f64::INFINITY, f64::min);
    let mean_correct_wrong_advantage = reports
        .iter()
        .map(|report| report.metrics.correct_wrong_advantage)
        .sum::<f64>()
        / reports.len() as f64;
    let minimum_correct_wrong_advantage = reports
        .iter()
        .map(|report| report.metrics.correct_wrong_advantage)
        .fold(f64::INFINITY, f64::min);
    Ok(ReceiverPortabilityBenchmarkReport {
        schema: "cerebro.tidex.receiver_portability_benchmark/v1".into(),
        case_count: reports.len(),
        functional_dimension: functional_dim,
        receiver_parameter_dimension: receiver_dim,
        mean_recovered_gain,
        minimum_recovered_gain,
        mean_correct_wrong_advantage,
        minimum_correct_wrong_advantage,
        all_resolved: reports.iter().all(|report| report.resolved),
        cases: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frozen_input(method: ReceiverProposalMethod) -> FrozenReceiverCompilerInput {
        let mut calibration = coupled_calibration();
        calibration.receiver_snapshot_binding_sha256 =
            Some(Sha256Digest::digest_bytes(b"receiver-A"));
        calibration.wrong_functional_signatures.clear();
        FrozenReceiverCompilerInput {
            schema: "cerebro.tidex.frozen_receiver_compiler_input/v1".into(),
            calibration_capability_ids: (0..calibration.functional_signatures.len())
                .map(|i| crate::identity::CapabilityId::parse(format!("calibration.{i}")).unwrap())
                .collect(),
            calibration,
            protected_cortex: ProtectedCortex {
                parameter_importance: vec![0.0; 2],
                directions: vec![],
                max_damage_ratio: 0.01,
            },
            risk_metric_rows: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            policy: response_policy(),
            proposal_method: method,
        }
    }

    #[test]
    fn frozen_compiler_roundtrip_reuses_maps_without_target_time_fitting() {
        for method in [
            ReceiverProposalMethod::DecodeThenProject,
            ReceiverProposalMethod::CalibratedAffine,
        ] {
            let input = frozen_input(method);
            let frozen = freeze_receiver_compiler(&input).unwrap();
            let bytes = serde_json::to_vec(&frozen).unwrap();
            let reopened: FrozenReceiverCompiler = serde_json::from_slice(&bytes).unwrap();
            let verified = reopened.verify().unwrap();
            for (i, requested) in [vec![0.2, 0.4], vec![0.3, 0.8], vec![-0.1, 0.2]]
                .iter()
                .enumerate()
            {
                let wrong = vec![requested.iter().map(|v| -v).collect::<Vec<_>>()];
                let before = COMPILER_FIT_CALLS.with(|count| count.get());
                let result = verified
                    .compile(
                        &crate::identity::CapabilityId::parse(format!("heldout.{i}")).unwrap(),
                        requested,
                        &wrong,
                    )
                    .unwrap();
                assert_eq!(COMPILER_FIT_CALLS.with(|count| count.get()), before);
                assert!(result.allowed, "{result:#?}");
                let mut calibration = input.calibration.clone();
                calibration.wrong_functional_signatures = wrong;
                let legacy = compile_signature_with_method_and_validation(
                    requested,
                    &calibration,
                    &input.protected_cortex,
                    &Matrix::identity(2),
                    &input.policy,
                    method,
                    ReceiverProposalValidationProfile::ParametricCrossValidation,
                )
                .unwrap();
                assert_eq!(result, legacy);
            }
            assert_eq!(serde_json::to_vec(&reopened).unwrap(), bytes);
        }
    }

    #[test]
    fn frozen_compiler_rejects_rehashed_forged_maps_and_diagnostics() {
        let frozen =
            freeze_receiver_compiler(&frozen_input(ReceiverProposalMethod::DecodeThenProject))
                .unwrap();
        for change in 0..3 {
            let mut forged = frozen.clone();
            match change {
                0 => forged.maps.decoder.target_decoder.weights.set(0, 0, 999.0),
                1 => forged.maps.encoder.loo_cv_r2 -= 0.5,
                _ => forged.maps.encoder.map.target_dim += 1,
            }
            forged.manifest_sha256 = Sha256Digest::zero();
            forged.manifest_sha256 = Sha256Digest::digest_domain(
                b"CEREBRO:TIDEX:FROZEN-RECEIVER-COMPILER:v1\0",
                &serde_json::to_vec(&forged).unwrap(),
            );
            assert!(forged.verify().is_err());
        }
    }

    #[test]
    fn frozen_compiler_rejects_unsupported_extrapolation_even_with_a_good_inverse() {
        let mut input = frozen_input(ReceiverProposalMethod::DecodeThenProject);
        input.policy.maximum_quadratic_cost = 1e9;
        let frozen = freeze_receiver_compiler(&input).unwrap();
        let verified = frozen.verify().unwrap();
        let result = verified
            .compile(
                &crate::identity::CapabilityId::parse("heldout.ood").unwrap(),
                &[100.0, 400.0],
                &[vec![-100.0, -400.0]],
            )
            .unwrap();
        assert!(result.functional_relative_error < input.policy.maximum_functional_relative_error);
        assert!(!result.proposal_within_calibrated_support);
        assert!(!result.allowed);
    }

    #[test]
    fn frozen_compiler_rejects_target_leakage_and_changed_bindings() {
        let input = frozen_input(ReceiverProposalMethod::DecodeThenProject);
        let frozen = freeze_receiver_compiler(&input).unwrap();
        let verified = frozen.verify().unwrap();
        let wrong = vec![vec![-0.2, -0.4]];
        assert!(verified
            .compile(&input.calibration_capability_ids[0], &[0.2, 0.4], &wrong)
            .is_err());
        let id = crate::identity::CapabilityId::parse("heldout.target").unwrap();
        assert!(verified
            .compile(&id, &input.calibration.functional_signatures[0], &wrong)
            .is_err());
        for change in 0..5 {
            let mut altered = input.clone();
            match change {
                0 => {
                    altered.calibration.receiver_snapshot_binding_sha256 =
                        Some(Sha256Digest::digest_bytes(b"other-model"))
                }
                1 => altered.calibration.receiver_solutions[0][0] += 0.01,
                2 => altered.policy.maximum_quadratic_cost *= 2.0,
                3 => altered.protected_cortex.max_damage_ratio = 1.0,
                _ => altered.risk_metric_rows[0][0] *= 2.0,
            }
            assert!(frozen
                .validate_binding(
                    &altered.calibration,
                    &altered.protected_cortex,
                    &Matrix::from_rows(&altered.risk_metric_rows).unwrap(),
                    &altered.policy,
                )
                .is_err());
        }
    }

    #[test]
    fn frozen_compiler_rejects_relabeled_duplicate_and_roundoff_target_observations() {
        for method in [
            ReceiverProposalMethod::DecodeThenProject,
            ReceiverProposalMethod::CalibratedAffine,
        ] {
            let input = frozen_input(method);
            let mut duplicated = input.clone();
            duplicated.calibration.functional_signatures[1] =
                duplicated.calibration.functional_signatures[0].clone();
            duplicated.calibration.receiver_solutions[1] =
                duplicated.calibration.receiver_solutions[0].clone();
            assert!(freeze_receiver_compiler(&duplicated)
                .unwrap_err()
                .to_string()
                .contains("functional_identity_collision"));

            let frozen = freeze_receiver_compiler(&input).unwrap();
            let verified = frozen.verify().unwrap();
            let mut relabeled = input.calibration.functional_signatures[0].clone();
            relabeled[0] += f64::EPSILON;
            let error = verified
                .compile(
                    &CapabilityId::parse("relabeled.target").unwrap(),
                    &relabeled,
                    &[vec![-0.2, -0.4]],
                )
                .unwrap_err();
            assert!(error.to_string().contains("target_leaked_into_calibration"));
        }
    }

    #[test]
    fn frozen_compiler_rejects_unbound_malformed_or_target_bearing_inputs() {
        let input = frozen_input(ReceiverProposalMethod::DecodeThenProject);
        for change in 0..7 {
            let mut altered = input.clone();
            match change {
                0 => altered.calibration.receiver_snapshot_binding_sha256 = None,
                1 => {
                    altered.calibration_capability_ids[1] =
                        altered.calibration_capability_ids[0].clone()
                }
                2 => altered.calibration.wrong_functional_signatures = vec![vec![0.2, 0.4]],
                3 => altered.calibration.receiver_solutions[0]
                    .pop()
                    .map(|_| ())
                    .unwrap(),
                4 => altered.calibration.functional_signatures[0][0] = f64::NAN,
                5 => altered.proposal_method = ReceiverProposalMethod::FitProtectedCoordinates,
                _ => altered.risk_metric_rows[0].clear(),
            }
            assert!(freeze_receiver_compiler(&altered).is_err());
        }
    }

    #[test]
    fn frozen_compiler_preserves_a_function_of_shared_weights_and_rejects_conflict() {
        let mut input = frozen_input(ReceiverProposalMethod::DecodeThenProject);
        let unit = 1.0 / 2.0_f64.sqrt();
        input.protected_cortex.parameter_importance = vec![1e-5; 2];
        input.protected_cortex.directions = vec![crate::contracts::ProtectedDirection {
            probe_id: crate::identity::ProbeId::parse("shared.sum").unwrap(),
            direction: vec![unit, unit],
            importance: 1.0,
        }];
        let frozen = freeze_receiver_compiler(&input).unwrap();
        let verified = frozen.verify().unwrap();
        let id = crate::identity::CapabilityId::parse("heldout.shared").unwrap();
        let result = verified
            .compile(&id, &[0.2, 0.4], &[vec![-0.2, -0.4]])
            .unwrap();
        assert!(result.allowed, "{result:#?}");
        let before = [1.0, 2.0];
        let after = [
            before[0] + result.target_delta[0],
            before[1] + result.target_delta[1],
        ];
        assert!((after[0] - before[0]).abs() > 0.1);
        assert!((after[1] - before[1]).abs() > 0.1);
        assert!((after.iter().sum::<f64>() - before.iter().sum::<f64>()).abs() < 1e-12);
        // This second target requires moving exactly the protected sum.
        match verified.compile(&id, &[0.2, 0.8], &[vec![-0.2, -0.8]]) {
            Ok(conflict) => assert!(!conflict.allowed),
            Err(_) => (),
        }
    }

    fn receiver_basis_test_input() -> ReceiverBasisBenchmarkInput {
        let calibration_deltas = vec![
            vec![2.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.5, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![-1.0, 1.0, 0.0, 0.0],
            vec![1.0, -1.0, 0.0, 0.0],
            vec![2.0, 1.0, 0.0, 0.0],
        ];
        ReceiverBasisBenchmarkInput {
            schema: "cerebro.tidex.receiver_basis_benchmark_input/v1".into(),
            calibration_capability_ids: (0..calibration_deltas.len())
                .map(|index| CapabilityId::parse(format!("basis-calibration-{index}")).unwrap())
                .collect(),
            calibration_deltas,
            target_explained_variance: 0.99,
            max_rank: 2,
            ridge: 1e-6,
            min_signal_to_noise: 1.0,
        }
    }

    #[test]
    fn receiver_basis_adapter_preserves_real_source_mixtures_and_row_lineage() {
        let input = receiver_basis_test_input();
        let result = benchmark_receiver_basis(&input).unwrap();
        assert!(result.candidate_only);
        assert_eq!(
            result.calibration_capability_ids,
            input.calibration_capability_ids
        );
        assert_eq!(result.selected_rank, 2);
        assert_eq!(result.parameter_dimension, 4);
        assert_eq!(result.resolution.field_count, 2);
        assert!(result.resolution.all_fields_resolved);
        assert!(result.retained_energy > 1.0 - 1e-12);
        assert!(result.reconstruction_rms < 1e-10);
        // Verify both contracts against the original supplied matrix; no
        // synthetic SkillFields or invented DeltaObservations enter the API.
        for axis in 0..result.selected_rank {
            for parameter in 0..result.parameter_dimension {
                let actual = result.source_mixtures[axis]
                    .iter()
                    .enumerate()
                    .map(|(row, coefficient)| {
                        coefficient * input.calibration_deltas[row][parameter]
                    })
                    .sum::<f64>();
                assert!((actual - result.axes[axis][parameter]).abs() < 1e-10);
            }
        }
        for row in 0..input.calibration_deltas.len() {
            for parameter in 0..result.parameter_dimension {
                let actual = result.coordinates[row]
                    .iter()
                    .enumerate()
                    .map(|(axis, coefficient)| coefficient * result.axes[axis][parameter])
                    .sum::<f64>();
                assert!((actual - input.calibration_deltas[row][parameter]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn receiver_basis_adapter_reports_unweighted_raw_reconstruction_energy() {
        let mut input = receiver_basis_test_input();
        input.calibration_deltas = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![4.0, 0.0],
            vec![0.0, 9.0],
            vec![-2.0, 0.0],
        ];
        input.target_explained_variance = 0.5;
        input.max_rank = 1;
        let result = benchmark_receiver_basis(&input).unwrap();
        assert_eq!(result.selected_rank, 1);
        let mut sse = 0.0;
        let mut total_energy = 0.0;
        for (row, actual) in input.calibration_deltas.iter().enumerate() {
            for (parameter, value) in actual.iter().enumerate() {
                let reconstructed = result.coordinates[row][0] * result.axes[0][parameter];
                sse += (reconstructed - value).powi(2);
                total_energy += value * value;
            }
        }
        assert!(sse > 0.0);
        assert!((result.retained_energy - (1.0 - sse / total_energy)).abs() < 1e-12);
        assert!((result.reconstruction_rms - (sse / 12.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn receiver_basis_adapter_rejects_ambiguous_lineage_nonfinite_values_and_resource_excess() {
        let input = receiver_basis_test_input();
        let mut duplicate = input.clone();
        duplicate.calibration_capability_ids[1] = duplicate.calibration_capability_ids[0].clone();
        assert!(benchmark_receiver_basis(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate_calibration_id"));
        let mut missing_id = input.clone();
        missing_id.calibration_capability_ids.pop();
        assert!(benchmark_receiver_basis(&missing_id).is_err());
        let mut malformed = input.clone();
        malformed.calibration_deltas[1].pop();
        assert!(benchmark_receiver_basis(&malformed).is_err());
        let mut nonfinite = input.clone();
        nonfinite.calibration_deltas[0][0] = f64::NAN;
        assert!(benchmark_receiver_basis(&nonfinite).is_err());
        let mut zero = input.clone();
        zero.calibration_deltas = vec![vec![0.0; 4]; 6];
        assert!(benchmark_receiver_basis(&zero)
            .unwrap_err()
            .to_string()
            .contains("zero_calibration_energy"));
        for max_rank in [0, 5, 33] {
            let mut invalid = input.clone();
            invalid.max_rank = max_rank;
            assert!(benchmark_receiver_basis(&invalid)
                .unwrap_err()
                .to_string()
                .contains("resource_or_rank_bounds"));
        }
        let mut too_few = input.clone();
        too_few.calibration_deltas.truncate(4);
        too_few.calibration_capability_ids.truncate(4);
        assert!(benchmark_receiver_basis(&too_few).is_err());
        let mut too_many = input.clone();
        too_many.calibration_deltas = vec![vec![1.0; 4]; 257];
        assert!(benchmark_receiver_basis(&too_many)
            .unwrap_err()
            .to_string()
            .contains("resource_or_rank_bounds"));
        let mut too_wide = input.clone();
        too_wide.calibration_deltas = vec![vec![1.0; 4097]; 6];
        assert!(benchmark_receiver_basis(&too_wide)
            .unwrap_err()
            .to_string()
            .contains("resource_or_rank_bounds"));
        let mut unknown = serde_json::to_value(&input).unwrap();
        unknown["promote"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ReceiverBasisBenchmarkInput>(unknown).is_err());
    }

    #[test]
    fn calibrated_ir_rejects_collisions_but_accepts_constant_coordinate_vectors() {
        let distinct = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![0.0, 0.0]];
        assert!(validate_functional_anchor_identity(&distinct).unwrap() > 0.0);
        let duplicate = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        assert!(validate_functional_anchor_identity(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("functional_identity_collision"));
        let signed_zero = vec![vec![0.0, 1.0], vec![-0.0, 1.0]];
        assert!(validate_functional_anchor_identity(&signed_zero).is_err());
    }

    #[test]
    fn centered_compilation_uses_both_maps_and_cannot_skip_inverse_validation() {
        let rows = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, -1.0],
            vec![-1.0, 2.0],
            vec![0.5, 2.0],
            vec![-0.3, -0.4],
        ];
        let calibration = ReceiverCalibrationSet {
            receiver_snapshot_binding_sha256: None,
            receiver_solutions: rows
                .iter()
                .map(|r| vec![4.0 + 3.0 * r[0] - r[1], -2.0 + r[0] + 2.0 * r[1]])
                .collect(),
            functional_signatures: rows,
            wrong_functional_signatures: vec![vec![-0.2, -0.4]],
        };
        let cortex = ProtectedCortex {
            parameter_importance: vec![0.0; 2],
            directions: vec![],
            max_damage_ratio: 0.0,
        };
        let policy = ReceiverCompilerPolicy {
            schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
            ridge: 1e-9,
            minimum_decoder_loo_r2: 0.0,
            minimum_encoder_loo_r2: 0.0,
            minimum_decoder_loo_cosine: 0.0,
            maximum_functional_relative_error: 1e-5,
            minimum_identity_margin: 0.05,
            maximum_quadratic_cost: 1e6,
        };
        let result = compile_receiver_signature_calibrated_affine(
            &[0.2, 0.4],
            &calibration,
            &cortex,
            &Matrix::identity(2),
            &policy,
        )
        .unwrap();
        assert!(result.allowed);
        assert!((result.target_delta[0] - 4.2).abs() < 1e-6);
        assert!((result.target_delta[1] + 1.0).abs() < 1e-6);
        assert!(result.decoder_fit_diagnostics.is_some());
        assert!(result.encoder_fit_diagnostics.is_some());
        let rejected = compile_signature_with_method_and_validation(
            &[0.2, 0.4],
            &calibration,
            &cortex,
            &Matrix::identity(2),
            &policy,
            ReceiverProposalMethod::CalibratedAffine,
            ReceiverProposalValidationProfile::BehavioralCalibrationMeasurement,
        );
        assert!(rejected.is_err());
    }

    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
        SystemEnvelope,
    };
    use crate::capability_ir::{
        IrNode, OperatorIrTransition, OutputBinding, PrimitiveSet, StateIrAnchor, TypedPort,
        ValueReference,
    };
    use crate::identity::{AcquisitionId, CapabilityId, CapabilityNodeId, PortId, PrimitiveId};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn receiver_readout_compilation_uses_executed_ir_values_and_projection() {
        use crate::capability_ir::ParameterSlot;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tidex-readout-compiler-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("readout.rs"),
            b"pub fn margin(x:[f64;2],w:[f64;2])->f64 { x[0]*w[0]+x[1]*w[1] }\n",
        )
        .unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("readout-compiler-test").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 1,
                max_total_bytes: 4096,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&root, &request).unwrap();
        let port = |name: &str, shape: Vec<u64>| {
            TypedPort::tensor_f64(PortId::parse(name).unwrap(), shape).unwrap()
        };
        let ir = CapabilityIr::new_with_parameters(
            CapabilityId::parse("linear.readout:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![port("runtime_input", vec![2, 1])],
            vec![ParameterSlot::new(port("resident_weights", vec![1, 2])).unwrap()],
            vec![IrNode::new(
                CapabilityNodeId::parse("node.linear_map").unwrap(),
                PrimitiveId::parse("tensor.matmul").unwrap(),
                vec![
                    ValueReference::Parameter {
                        name: PortId::parse("resident_weights").unwrap(),
                    },
                    ValueReference::Input {
                        name: PortId::parse("runtime_input").unwrap(),
                    },
                ],
                port("mapped", vec![1, 1]),
                vec![PathBuf::from("readout.rs")],
            )
            .unwrap()],
            vec![OutputBinding::new(
                port("runtime_output", vec![1, 1]),
                ValueReference::NodeOutput {
                    node_id: CapabilityNodeId::parse("node.linear_map").unwrap(),
                },
            )
            .unwrap()],
        )
        .unwrap();
        let inputs = vec![vec![0.1, -0.3], vec![0.2, 0.0]];
        let weights = vec![2.0, -1.0];
        let mean = vec![0.3, 0.0];
        let components = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let input = ReceiverReadoutCapabilityInput {
            ir: &ir,
            envelope: &envelope,
            readout_weights: &weights,
            inputs: &inputs,
            projection_mean: &mean,
            projection_components: &components,
        };
        let calibration = coupled_calibration();
        let cortex = ProtectedCortex {
            parameter_importance: vec![0.0; 2],
            directions: vec![],
            max_damage_ratio: 0.0,
        };
        let policy = response_policy();
        let report = compile_receiver_readout_capability(
            &input,
            &calibration,
            &cortex,
            &Matrix::identity(2),
            &policy,
            ReceiverProposalMethod::CalibratedAffine,
        )
        .unwrap();
        assert!(report.numerical.allowed);
        assert!((report.execution.raw_margins[0] - 0.5).abs() < 1e-14);
        assert!((report.requested_signature[0] - 0.2).abs() < 1e-14);
        assert!((report.requested_signature[1] - 0.4).abs() < 1e-14);
        assert_eq!(
            report.numerical.validation_profile,
            ReceiverProposalValidationProfile::ParametricCrossValidation
        );
        let changed_weights = vec![3.0, -1.0];
        let changed = ReceiverReadoutCapabilityInput {
            readout_weights: &changed_weights,
            ..input
        };
        let changed = compile_receiver_readout_capability(
            &changed,
            &calibration,
            &cortex,
            &Matrix::identity(2),
            &policy,
            ReceiverProposalMethod::CalibratedAffine,
        )
        .unwrap();
        assert_ne!(
            report.execution.weights_sha256,
            changed.execution.weights_sha256
        );
        assert!((changed.requested_signature[0] - 0.3).abs() < 1e-14);
        assert_ne!(
            report.numerical.proposed_receiver_coordinates,
            changed.numerical.proposed_receiver_coordinates
        );
        let bad_mean = vec![0.0];
        let bad = ReceiverReadoutCapabilityInput {
            projection_mean: &bad_mean,
            ..input
        };
        assert!(compile_receiver_readout_capability(
            &bad,
            &calibration,
            &cortex,
            &Matrix::identity(2),
            &policy,
            ReceiverProposalMethod::CalibratedAffine
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    fn fixture_ir() -> (PathBuf, CapabilityIr) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tidex-receiver-compiler-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/capability.rs"), b"pub fn capability() {}\n").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("receiver-compiler-fixture").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 8,
                max_total_bytes: 1 << 20,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&root, &request).unwrap();
        let ir = CapabilityIr::new(
            CapabilityId::parse("state.toggle:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![TypedPort::tensor_f64(PortId::parse("state").unwrap(), vec![2, 1]).unwrap()],
            vec![IrNode::new(
                CapabilityNodeId::parse("node.normalize").unwrap(),
                PrimitiveId::parse("tensor.normalize").unwrap(),
                vec![ValueReference::Input {
                    name: PortId::parse("state").unwrap(),
                }],
                TypedPort::tensor_f64(PortId::parse("normalized").unwrap(), vec![2, 1]).unwrap(),
                vec![PathBuf::from("src/capability.rs")],
            )
            .unwrap()],
            vec![OutputBinding::new(
                TypedPort::tensor_f64(PortId::parse("result").unwrap(), vec![2, 1]).unwrap(),
                ValueReference::NodeOutput {
                    node_id: CapabilityNodeId::parse("node.normalize").unwrap(),
                },
            )
            .unwrap()],
        )
        .unwrap();
        (root, ir)
    }

    fn toggle_contract(ir: &CapabilityIr) -> OperationalCapabilityContract {
        let pre = 2.0_f64.sqrt();
        OperationalCapabilityContract {
            schema: "cerebro.tidex.operational_capability/v1".into(),
            capability_id: ir.capability_id().clone(),
            capability_ir_sha256: ir.manifest_digest().clone(),
            state_dimension: 2,
            anchors: vec![
                StateIrAnchor {
                    anchor_id: "s0".into(),
                    state: vec![1.0, 0.0],
                },
                StateIrAnchor {
                    anchor_id: "s1".into(),
                    state: vec![0.0, 1.0],
                },
            ],
            transitions: vec![
                OperatorIrTransition {
                    operator_id: "toggle".into(),
                    source_anchor_id: "s0".into(),
                    target_anchor_id: "s1".into(),
                    observed_next_state: vec![0.0, 1.0],
                    pre_target_error: pre,
                    post_target_error: 0.0,
                },
                OperatorIrTransition {
                    operator_id: "toggle".into(),
                    source_anchor_id: "s1".into(),
                    target_anchor_id: "s0".into(),
                    observed_next_state: vec![1.0, 0.0],
                    pre_target_error: pre,
                    post_target_error: 0.0,
                },
            ],
            maximum_closure_error: 1e-5,
            maximum_contraction_ratio: 1e-5,
        }
    }

    fn receiver_solution(functional: &[f64]) -> Vec<f64> {
        vec![
            2.0 * functional[0] + functional[1] - 0.5 * functional[2] + 0.1,
            -functional[0] + 1.5 * functional[2] + functional[3] - 0.2,
            0.5 * functional[1] + 2.0 * functional[3] + 0.3,
            functional[0] - functional[1] + functional[2] - functional[3] + 0.4,
            0.7 * functional[0] + 0.2 * functional[1] + 0.3 * functional[2] + 0.9 * functional[3]
                - 0.1,
        ]
    }

    fn coupled_calibration() -> ReceiverCalibrationSet {
        let receiver = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, -1.0],
            vec![-1.0, 2.0],
            vec![0.5, 2.0],
        ];
        let functions = receiver
            .iter()
            .map(|x| vec![x[0], 3.0 * x[0] + x[1]])
            .collect();
        ReceiverCalibrationSet {
            receiver_snapshot_binding_sha256: None,
            functional_signatures: functions,
            receiver_solutions: receiver,
            wrong_functional_signatures: vec![vec![-0.2, -0.4]],
        }
    }
    fn response_policy() -> ReceiverCompilerPolicy {
        ReceiverCompilerPolicy {
            schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
            ridge: 1e-10,
            minimum_decoder_loo_r2: 0.99,
            minimum_encoder_loo_r2: 0.99,
            minimum_decoder_loo_cosine: 0.99,
            maximum_functional_relative_error: 1e-4,
            minimum_identity_margin: 0.1,
            maximum_quadratic_cost: 10.0,
        }
    }
    #[test]
    fn protected_coordinate_fit_preserves_the_requested_response_without_weakening_gates() {
        let calibration = coupled_calibration();
        let cortex = ProtectedCortex {
            parameter_importance: vec![0.4, 0.01],
            directions: vec![],
            max_damage_ratio: 0.3,
        };
        let policy = response_policy();
        let metric = Matrix::identity(2);
        let legacy =
            compile_receiver_signature(&[0.2, 0.4], &calibration, &cortex, &metric, &policy)
                .unwrap();
        let fitted = compile_receiver_signature_in_safe_coordinates(
            &[0.2, 0.4],
            &calibration,
            &cortex,
            &metric,
            &policy,
        )
        .unwrap();
        assert!(!legacy.allowed);
        assert!(fitted.allowed, "{fitted:#?}");
        assert_eq!(
            legacy.proposal_method,
            ReceiverProposalMethod::DecodeThenProject
        );
        assert_eq!(
            fitted.proposal_method,
            ReceiverProposalMethod::FitProtectedCoordinates
        );
        assert!(fitted.functional_relative_error < 1e-4);
        assert!(fitted.protection_damage_ratio <= cortex.max_damage_ratio);
        assert!(fitted.trust_region.accepted_quadratic_cost <= policy.maximum_quadratic_cost);
        assert!((fitted.target_delta[0] - 0.2).abs() < 1e-6);
        assert!((fitted.target_delta[1] + 0.2).abs() < 1e-6);
    }
    #[test]
    fn protected_coordinate_fit_cannot_restore_a_hard_forbidden_direction() {
        let calibration = coupled_calibration();
        let cortex = ProtectedCortex {
            parameter_importance: vec![1.0; 2],
            directions: vec![crate::contracts::ProtectedDirection {
                probe_id: crate::identity::ProbeId::parse("hard.x").unwrap(),
                direction: vec![1.0, 0.0],
                importance: 1.0,
            }],
            max_damage_ratio: 1.0,
        };
        let result = compile_receiver_signature_in_safe_coordinates(
            &[0.2, 0.4],
            &calibration,
            &cortex,
            &Matrix::identity(2),
            &response_policy(),
        )
        .unwrap();
        assert!(!result.allowed);
        assert!(result.target_delta[0].abs() < 1e-12);
        assert!(result.functional_relative_error > 0.4);
        assert!(result.protection_max_weighted_residual < 1e-12);
    }
    #[test]
    fn protected_coordinate_fit_still_rejects_insufficient_risk_budget() {
        let calibration = coupled_calibration();
        let cortex = ProtectedCortex {
            parameter_importance: vec![0.4, 0.01],
            directions: vec![],
            max_damage_ratio: 0.3,
        };
        let mut policy = response_policy();
        policy.maximum_quadratic_cost = 1e-6;
        let result = compile_receiver_signature_in_safe_coordinates(
            &[0.2, 0.4],
            &calibration,
            &cortex,
            &Matrix::identity(2),
            &policy,
        )
        .unwrap();
        assert!(!result.allowed);
        assert!(result.trust_region.constrained);
        assert!(result.trust_region.accepted_quadratic_cost < 1.00001e-6);
        assert!(result.functional_relative_error > 0.9);
    }
    #[test]
    fn protected_coordinate_fit_does_not_bypass_protection_removal_limit() {
        let calibration = coupled_calibration();
        let cortex = ProtectedCortex {
            parameter_importance: vec![1.0; 2],
            directions: vec![],
            max_damage_ratio: 0.01,
        };
        let result = compile_receiver_signature_in_safe_coordinates(
            &[0.2, 0.4],
            &calibration,
            &cortex,
            &Matrix::identity(2),
            &response_policy(),
        )
        .unwrap();
        assert!(!result.allowed);
        assert!(result.protection_damage_ratio > 0.49);
    }

    fn calibration() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 0.5, 0.5, 1.0],
            vec![0.2, 1.0, 1.0, 0.2],
            vec![1.2, -0.2, 0.4, 0.8],
        ]
    }

    #[test]
    fn held_out_receiver_compilation_recovers_capability_without_donor_weights() {
        let (root, ir) = fixture_ir();
        let contract = toggle_contract(&ir);
        let functional = calibration();
        let receiver = functional
            .iter()
            .map(|signature| receiver_solution(signature))
            .collect::<Vec<_>>();
        let cortex = ProtectedCortex {
            parameter_importance: vec![0.0; 5],
            directions: Vec::new(),
            max_damage_ratio: 0.01,
        };
        let policy = ReceiverCompilerPolicy {
            schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
            ridge: 1e-10,
            minimum_decoder_loo_r2: 0.999,
            minimum_encoder_loo_r2: 0.999,
            minimum_decoder_loo_cosine: 0.999,
            maximum_functional_relative_error: 1e-4,
            minimum_identity_margin: 0.05,
            maximum_quadratic_cost: 1e6,
        };
        let calibration = ReceiverCalibrationSet {
            receiver_snapshot_binding_sha256: None,
            functional_signatures: functional.clone(),
            receiver_solutions: receiver.clone(),
            wrong_functional_signatures: vec![functional[0].clone(), functional[2].clone()],
        };
        let compilation = compile_receiver_capability(
            &ir,
            &contract,
            &calibration,
            &cortex,
            &Matrix::identity(5),
            &policy,
        )
        .unwrap();
        assert!(compilation.allowed, "{compilation:#?}");
        assert!(compilation.functional_relative_error < 1e-4);
        assert!(compilation.operational_verification.allowed);

        // The direct receiver solution is an oracle used only here for
        // evaluation. It never entered the compiler calibration for this
        // held-out query.
        let expected = contract.canonical_transition_signature(&ir).unwrap();
        let direct = receiver_solution(&expected);
        let encoder = learn_transport_validated(&receiver, &functional, 1e-10).unwrap();
        let virgin_behavior = encoder.map.apply(&[1e-6; 5]).unwrap();
        let direct_behavior = encoder.map.apply(&direct).unwrap();
        let wrong_delta = learn_functional_transplant(&functional, &receiver, 1e-10)
            .unwrap()
            .transplant(&functional[0])
            .unwrap()
            .target_vector;
        let wrong_behavior = encoder.map.apply(&wrong_delta).unwrap();
        let metrics = evaluate_portability(
            &expected,
            &virgin_behavior,
            &direct_behavior,
            &compilation.predicted_functional_signature,
            &wrong_behavior,
        )
        .unwrap();
        assert!(metrics.recovered_gain > 0.99, "{metrics:#?}");
        assert!(metrics.correct_wrong_advantage > 0.05, "{metrics:#?}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn leave_one_skill_out_benchmark_never_trains_on_the_held_out_receiver_solution() {
        let functional = calibration();
        let mut cases = functional
            .iter()
            .enumerate()
            .map(|(index, signature)| ReceiverPortabilityCase {
                skill_id: format!("calibration-{index}"),
                functional_signature: signature.clone(),
                direct_receiver_solution: receiver_solution(signature),
            })
            .collect::<Vec<_>>();
        cases.push(ReceiverPortabilityCase {
            skill_id: "toggle-held-out".into(),
            functional_signature: vec![0.0, 1.0, 1.0, 0.0],
            direct_receiver_solution: receiver_solution(&[0.0, 1.0, 1.0, 0.0]),
        });
        let input = ReceiverPortabilityBenchmarkInput {
            schema: "cerebro.tidex.receiver_portability_benchmark_input/v1".into(),
            ridge: 1e-10,
            cases,
        };
        let report = benchmark_receiver_portability_leave_one_out(&input).unwrap();
        assert_eq!(report.case_count, 9);
        assert!(report.all_resolved, "{report:#?}");
        assert!(report.mean_recovered_gain > 0.99, "{report:#?}");
        assert!(report.minimum_recovered_gain > 0.98, "{report:#?}");
        assert!(report.mean_correct_wrong_advantage > 0.05, "{report:#?}");

        // Changing only the held-out oracle is allowed to change evaluation
        // metrics, but it must not change the compiled receiver solution.
        let mut changed_oracle = input.clone();
        for value in &mut changed_oracle.cases[8].direct_receiver_solution {
            *value = *value * 1.05 + 0.01;
        }
        let changed_report = benchmark_receiver_portability_leave_one_out(&changed_oracle).unwrap();
        assert_eq!(
            report.cases[8].compiled_receiver_solution,
            changed_report.cases[8].compiled_receiver_solution
        );
    }

    #[test]
    fn receiver_compiler_fails_promotion_when_protection_destroys_contract() {
        let (root, ir) = fixture_ir();
        let contract = toggle_contract(&ir);
        let functional = calibration();
        let receiver = functional
            .iter()
            .map(|signature| receiver_solution(signature))
            .collect::<Vec<_>>();
        let cortex = ProtectedCortex {
            parameter_importance: vec![1000.0; 5],
            directions: Vec::new(),
            max_damage_ratio: 1.0,
        };
        let policy = ReceiverCompilerPolicy {
            schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
            ridge: 1e-10,
            minimum_decoder_loo_r2: 0.9,
            minimum_encoder_loo_r2: 0.9,
            minimum_decoder_loo_cosine: 0.9,
            maximum_functional_relative_error: 0.01,
            minimum_identity_margin: 0.01,
            maximum_quadratic_cost: 1e6,
        };
        let calibration = ReceiverCalibrationSet {
            receiver_snapshot_binding_sha256: None,
            functional_signatures: functional.clone(),
            receiver_solutions: receiver.clone(),
            wrong_functional_signatures: vec![functional[0].clone()],
        };
        let compilation = compile_receiver_capability(
            &ir,
            &contract,
            &calibration,
            &cortex,
            &Matrix::identity(5),
            &policy,
        )
        .unwrap();
        assert!(!compilation.allowed);
        assert!(compilation.functional_relative_error > 0.01);
        fs::remove_dir_all(root).unwrap();
    }
}
