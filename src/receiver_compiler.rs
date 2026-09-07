//! Receiver-specific compilation from canonical functional semantics.
//!
//! This module deliberately never accepts donor parameter vectors. A receiver
//! compiler is calibrated from capability-independent functional signatures and
//! receiver-native solutions, predicts a new receiver delta for a held-out
//! capability, applies protection/trust-region constraints, and then verifies
//! the resulting behavior back in canonical functional space.

use crate::capability_ir::{
    CapabilityIr, OperationalCapabilityContract, OperationalInterfaceVerification,
};
use crate::contracts::ProtectedCortex;
use crate::error::{BrainError, BrainResult};
use crate::linalg::{cosine, norm, Matrix};
use crate::protected::project_to_safe_subspace;
use crate::transport::{learn_functional_transplant, learn_transport_validated};
use crate::trust_region::{apply_quadratic_trust_region, TrustRegionResult};
use serde::{Deserialize, Serialize};

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
    pub functional_signatures: Vec<Vec<f64>>,
    pub receiver_solutions: Vec<Vec<f64>>,
    pub wrong_functional_signatures: Vec<Vec<f64>>,
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

/// Compile one held-out operational capability into receiver-native parameters.
///
/// `calibration` must contain matched capabilities *other than the queried
/// capability* when this function is used to claim portability. This function
/// cannot infer experimental data leakage, so the benchmark/front-end is
/// responsible for enforcing that split.
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
    if norm(&requested)? <= 1e-15 {
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

    // Forward decoder: canonical functional semantics -> receiver-native delta.
    let decoder = learn_functional_transplant(
        &calibration.functional_signatures,
        &calibration.receiver_solutions,
        policy.ridge,
    )?;
    // Reverse behavioral model: receiver-native delta -> canonical semantics.
    // This is not promotion evidence by itself; it is a held-out verification
    // instrument whose quality is independently cross-validated below.
    let encoder = learn_transport_validated(
        &calibration.receiver_solutions,
        &calibration.functional_signatures,
        policy.ridge,
    )?;
    let proposed = decoder.transplant(&requested)?.target_vector;

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
        .zip(&requested)
        .map(|(left, right)| left - right)
        .collect::<Vec<_>>();
    let functional_relative_error = norm(&residual)? / norm(&requested)?.max(1e-15);
    let correct_cosine = cosine(&predicted, &requested)?;
    let maximum_wrong_cosine = calibration
        .wrong_functional_signatures
        .iter()
        .map(|wrong| cosine(&predicted, wrong))
        .collect::<BrainResult<Vec<_>>>()?
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let identity_margin = correct_cosine - maximum_wrong_cosine;
    let operational_verification = operational.verify_receiver_signature(ir, &predicted)?;

    let allowed = decoder.resolved
        && encoder.resolved
        && decoder.loo_cv_r2 >= policy.minimum_decoder_loo_r2
        && encoder.loo_cv_r2 >= policy.minimum_encoder_loo_r2
        && decoder.min_loo_cosine >= policy.minimum_decoder_loo_cosine
        && functional_relative_error <= policy.maximum_functional_relative_error
        && identity_margin >= policy.minimum_identity_margin
        && protection.allowed
        && trust.accepted_quadratic_cost <= policy.maximum_quadratic_cost
        && operational_verification.allowed;

    Ok(ReceiverCompilation {
        schema: "cerebro.tidex.receiver_compilation/v1".into(),
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
        operational_verification,
        allowed,
    })
}

/// Compare a transferred receiver against an untouched receiver, a direct
/// receiver oracle, and a wrong-skill receiver using one common functional
/// score. This metric is intentionally independent of parameter distance.
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

/// Leave-one-skill-out portability benchmark. The direct receiver solution of
/// the held-out skill is never passed to either learned map; it is opened only
/// after compilation as an oracle for `RecoveredGain`.
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
