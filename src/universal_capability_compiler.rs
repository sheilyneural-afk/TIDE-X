//! Experimental orchestration for receiver-native capability compilation.
//!
//! This module intentionally contains no donor-weight, adapter, LoRA, or task
//! vector representation.  It binds the existing authenticated structural IR
//! and operational contract to the existing receiver compiler.  Consequently
//! a successful result is evidence only for the supplied calibration domain;
//! it is never an automatic residency or promotion decision.

use crate::acquisition_contract::SystemEnvelope;
use crate::capability_ir::{CapabilityIr, OperationalCapabilityContract};
use crate::contracts::ProtectedCortex;
use crate::digest::{CapabilityIrDigest, Sha256Digest, SystemEnvelopeDigest};
use crate::error::{BrainError, BrainResult};
use crate::linalg::Matrix;
use crate::receiver_compiler::{
    compile_receiver_capability, ReceiverCalibrationSet, ReceiverCompilation,
    ReceiverCompilerPolicy,
};
use crate::receiver_profile::{
    assess_compatibility, create_shadow_plan, CapabilityRequirements, CompatibilityAssessment,
    MaterializationPlan, MaterializationStrategy, ReceiverProfile,
};
use crate::receiver_profiler::ReceiverSnapshotBinding;
use serde::{Deserialize, Serialize};

/// Portable input envelope for one experimental receiver compilation.
///
/// `risk_metric_rows` is used instead of serialising the internal matrix type.
/// It must describe a finite square matrix with one row per receiver parameter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityCompilationRequest {
    pub schema: String,
    pub system_envelope: SystemEnvelope,
    pub capability_ir: CapabilityIr,
    pub operational_contract: OperationalCapabilityContract,
    pub calibration: ReceiverCalibrationSet,
    pub protected_cortex: ProtectedCortex,
    pub risk_metric_rows: Vec<Vec<f64>>,
    pub policy: ReceiverCompilerPolicy,
}

/// The only dispositions this experimental boundary can produce.
///
/// `ExperimentalOnly` deliberately means that all local gates passed. It is
/// not a portability, equivalence, deployment, or promotion assertion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UniversalCapabilityDisposition {
    ExperimentalOnly,
    Rejected,
}

/// A provenance-bound result from one receiver-native compilation attempt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityCompilation {
    pub schema: String,
    pub source_envelope_sha256: SystemEnvelopeDigest,
    pub capability_ir_sha256: CapabilityIrDigest,
    pub receiver: ReceiverCompilation,
    pub disposition: UniversalCapabilityDisposition,
}

/// A replayable, request-bound record of one experimental compilation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityCompilationReceipt {
    pub schema: String,
    pub request_sha256: Sha256Digest,
    pub compilation: UniversalCapabilityCompilation,
}

/// A non-actuating composition of compilation, compatibility assessment and
/// materialization planning.  The resulting plan remains shadow-only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityPlanningRequest {
    pub schema: String,
    pub compilation: UniversalCapabilityCompilationRequest,
    pub receiver_profile: ReceiverProfile,
    pub receiver_snapshot: ReceiverSnapshotBinding,
    pub capability_requirements: CapabilityRequirements,
    pub requested_strategy: MaterializationStrategy,
    pub affected_regions: Vec<crate::identity::TensorId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityShadowPlan {
    pub schema: String,
    pub compilation_receipt: UniversalCapabilityCompilationReceipt,
    pub compatibility: CompatibilityAssessment,
    pub materialization_plan: MaterializationPlan,
}

/// A replayable record for the complete shadow-planning decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityShadowPlanReceipt {
    pub schema: String,
    pub planning_request_sha256: Sha256Digest,
    pub shadow_plan: UniversalCapabilityShadowPlan,
}

impl UniversalCapabilityCompilation {
    pub fn is_experimentally_usable(&self) -> bool {
        self.disposition == UniversalCapabilityDisposition::ExperimentalOnly
    }
}

/// Compile a sealed capability into receiver-native parameters for a declared
/// experimental calibration domain.
///
/// The function accepts semantic representations and receiver calibration
/// data, never donor parameters. It verifies the source envelope and IR chain
/// before delegating all numerical, protected-subspace, trust-region, and
/// operational checks to [`compile_receiver_capability`].
pub fn compile_experimental_universal_capability(
    envelope: &SystemEnvelope,
    ir: &CapabilityIr,
    operational: &OperationalCapabilityContract,
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
) -> BrainResult<UniversalCapabilityCompilation> {
    envelope.verify_manifest()?;
    ir.validate_against(envelope)?;
    operational.validate_against(ir)?;

    let receiver = compile_receiver_capability(
        ir,
        operational,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
    )?;
    let disposition = if receiver.allowed && receiver.operational_verification.allowed {
        UniversalCapabilityDisposition::ExperimentalOnly
    } else {
        UniversalCapabilityDisposition::Rejected
    };

    Ok(UniversalCapabilityCompilation {
        schema: "cerebro.tidex.universal_capability_compilation/v1".into(),
        source_envelope_sha256: envelope.manifest_sha256().clone(),
        capability_ir_sha256: ir.manifest_digest().clone(),
        receiver,
        disposition,
    })
}

/// Compile an independently serialised request.
///
/// This is the CLI and artifact boundary. It keeps the matrix wire format
/// explicit and rejects malformed rows before the numerical compiler sees it.
pub fn compile_experimental_universal_capability_request(
    request: &UniversalCapabilityCompilationRequest,
) -> BrainResult<UniversalCapabilityCompilation> {
    if request.schema != "cerebro.tidex.universal_capability_compilation_request/v1" {
        return Err(crate::error::BrainError::Invalid(
            "universal_capability_compilation_request_schema".into(),
        ));
    }
    let risk_metric = Matrix::from_rows(&request.risk_metric_rows)?;
    if risk_metric.row_count() == 0 || risk_metric.row_count() != risk_metric.column_count() {
        return Err(crate::error::BrainError::Invalid(
            "universal_capability_compilation_risk_metric_shape".into(),
        ));
    }
    compile_experimental_universal_capability(
        &request.system_envelope,
        &request.capability_ir,
        &request.operational_contract,
        &request.calibration,
        &request.protected_cortex,
        &risk_metric,
        &request.policy,
    )
}

fn request_digest(request: &UniversalCapabilityCompilationRequest) -> BrainResult<Sha256Digest> {
    let payload = serde_json::to_vec(request)?;
    let mut framed = b"CEREBRO:TIDEX:UNIVERSAL-CAPABILITY-COMPILATION-REQUEST:v1\0".to_vec();
    framed.extend_from_slice(&payload);
    Ok(Sha256Digest::digest_bytes(&framed))
}

/// Execute a request and bind its exact wire representation to the result.
pub fn execute_experimental_universal_capability_request(
    request: &UniversalCapabilityCompilationRequest,
) -> BrainResult<UniversalCapabilityCompilationReceipt> {
    Ok(UniversalCapabilityCompilationReceipt {
        schema: "cerebro.tidex.universal_capability_compilation_receipt/v1".into(),
        request_sha256: request_digest(request)?,
        compilation: compile_experimental_universal_capability_request(request)?,
    })
}

/// Recompute a receipt from its request and fail closed on any divergence.
pub fn replay_experimental_universal_capability_request(
    request: &UniversalCapabilityCompilationRequest,
    receipt: &UniversalCapabilityCompilationReceipt,
) -> BrainResult<()> {
    if receipt.schema != "cerebro.tidex.universal_capability_compilation_receipt/v1" {
        return Err(BrainError::Invalid(
            "universal_capability_compilation_receipt_schema".into(),
        ));
    }
    if receipt.request_sha256 != request_digest(request)? {
        return Err(BrainError::Integrity(
            "universal_capability_compilation_request_digest_mismatch".into(),
        ));
    }
    let replay = compile_experimental_universal_capability_request(request)?;
    if replay != receipt.compilation {
        return Err(BrainError::Integrity(
            "universal_capability_compilation_replay_mismatch".into(),
        ));
    }
    Ok(())
}

/// Produce a complete shadow-only plan. Rejected compilations are never
/// converted into plans, even if the receiver is otherwise compatible.
pub fn compile_and_plan_experimental_universal_capability(
    request: &UniversalCapabilityPlanningRequest,
) -> BrainResult<UniversalCapabilityShadowPlan> {
    if request.schema != "cerebro.tidex.universal_capability_planning_request/v1" {
        return Err(BrainError::Invalid(
            "universal_capability_planning_request_schema".into(),
        ));
    }
    let compilation_receipt =
        execute_experimental_universal_capability_request(&request.compilation)?;
    if !compilation_receipt.compilation.is_experimentally_usable() {
        return Err(BrainError::Integrity(
            "universal_capability_compilation_not_usable".into(),
        ));
    }
    request
        .capability_requirements
        .validate_against(&request.compilation.capability_ir)?;
    request
        .receiver_snapshot
        .validate_for(&request.receiver_profile)?;
    if request
        .compilation
        .calibration
        .receiver_snapshot_binding_sha256
        .as_ref()
        != Some(request.receiver_snapshot.manifest_digest())
    {
        return Err(BrainError::Integrity(
            "receiver_calibration_snapshot_binding_mismatch".into(),
        ));
    }
    if u64::try_from(
        compilation_receipt
            .compilation
            .receiver
            .receiver_parameter_dimension,
    )
    .map_err(|_| BrainError::Invalid("receiver_parameter_dimension_overflow".into()))?
        != request.receiver_profile.parameter_dimension
    {
        return Err(BrainError::Integrity(
            "receiver_profile_compilation_dimension_mismatch".into(),
        ));
    }
    let compatibility =
        assess_compatibility(&request.receiver_profile, &request.capability_requirements)?;
    let materialization_plan = create_shadow_plan(
        &request.receiver_profile,
        &compatibility,
        &request.capability_requirements,
        compilation_receipt.request_sha256.clone(),
        request.requested_strategy,
        request.affected_regions.clone(),
    )?;
    Ok(UniversalCapabilityShadowPlan {
        schema: "cerebro.tidex.universal_capability_shadow_plan/v1".into(),
        compilation_receipt,
        compatibility,
        materialization_plan,
    })
}

fn planning_request_digest(
    request: &UniversalCapabilityPlanningRequest,
) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:UNIVERSAL-CAPABILITY-PLANNING-REQUEST:v1\0",
        &serde_json::to_vec(request)?,
    ))
}

pub fn execute_universal_capability_shadow_plan(
    request: &UniversalCapabilityPlanningRequest,
) -> BrainResult<UniversalCapabilityShadowPlanReceipt> {
    Ok(UniversalCapabilityShadowPlanReceipt {
        schema: "cerebro.tidex.universal_capability_shadow_plan_receipt/v1".into(),
        planning_request_sha256: planning_request_digest(request)?,
        shadow_plan: compile_and_plan_experimental_universal_capability(request)?,
    })
}

pub fn replay_universal_capability_shadow_plan(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
) -> BrainResult<()> {
    if receipt.schema != "cerebro.tidex.universal_capability_shadow_plan_receipt/v1" {
        return Err(BrainError::Invalid(
            "universal_capability_shadow_plan_receipt_schema".into(),
        ));
    }
    if receipt.planning_request_sha256 != planning_request_digest(request)? {
        return Err(BrainError::Integrity(
            "universal_capability_shadow_plan_request_digest_mismatch".into(),
        ));
    }
    if receipt.shadow_plan != compile_and_plan_experimental_universal_capability(request)? {
        return Err(BrainError::Integrity(
            "universal_capability_shadow_plan_replay_mismatch".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
    };
    use crate::block_tomography::{BlockShapeSpec, ParameterBlockLayout, ParameterLayoutArtifact};
    use crate::capability_ir::{
        IrNode, OperatorIrTransition, OutputBinding, PrimitiveSet, StateIrAnchor, TypedPort,
        ValueReference,
    };
    use crate::dense_shadow_materializer::{
        load_dense_delta_shadow, materialize_replayed_dense_delta_shadow,
        persist_dense_delta_shadow,
    };
    use crate::identity::{
        AcquisitionId, ArchitectureId, CapabilityId, CapabilityNodeId, ModelId, PortId,
        PrimitiveId, TensorId,
    };
    use crate::lab_isolation::LabRoots;
    use crate::low_rank_shadow_materializer::{
        load_low_rank_shadow, materialize_replayed_low_rank_shadow, persist_low_rank_shadow,
        LowRankShadowPolicy,
    };
    use crate::receiver_layout::{
        FloatingScalarType, ReceiverMaterializationLayout, ReceiverScalarEncoding,
        ReceiverTensorPartitioning, ReceiverTensorPhysicalSpec,
    };
    use crate::receiver_profile::{CapabilityModality, ReceiverArchitecture, ReceiverRegion};
    use crate::shadow_materializer::materialize_replayed_receiver_coordinates_shadow;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> (
        PathBuf,
        SystemEnvelope,
        CapabilityIr,
        OperationalCapabilityContract,
    ) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tidex-ucc-{}-{nonce}", std::process::id()));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/capability.rs"), b"pub fn capability() {}\n").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("ucc-fixture").unwrap(),
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
        let pre = 2.0_f64.sqrt();
        let operational = OperationalCapabilityContract {
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
        };
        (root, envelope, ir, operational)
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

    #[test]
    fn binds_existing_verified_components_without_donor_parameters() {
        let (root, envelope, ir, operational) = fixture();
        let functional = calibration();
        let compilation = compile_experimental_universal_capability(
            &envelope,
            &ir,
            &operational,
            &ReceiverCalibrationSet {
                receiver_snapshot_binding_sha256: None,
                functional_signatures: functional.clone(),
                receiver_solutions: functional
                    .iter()
                    .map(|row| receiver_solution(row))
                    .collect(),
                wrong_functional_signatures: vec![functional[0].clone(), functional[2].clone()],
            },
            &ProtectedCortex {
                parameter_importance: vec![0.0; 5],
                directions: Vec::new(),
                max_damage_ratio: 0.01,
            },
            &Matrix::identity(5),
            &ReceiverCompilerPolicy {
                schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
                ridge: 1e-10,
                minimum_decoder_loo_r2: 0.999,
                minimum_encoder_loo_r2: 0.999,
                minimum_decoder_loo_cosine: 0.999,
                maximum_functional_relative_error: 1e-4,
                minimum_identity_margin: 0.05,
                maximum_quadratic_cost: 1e6,
            },
        )
        .unwrap();
        assert!(compilation.is_experimentally_usable(), "{compilation:#?}");
        assert_eq!(
            compilation.source_envelope_sha256,
            *envelope.manifest_sha256()
        );
        assert_eq!(compilation.capability_ir_sha256, *ir.manifest_digest());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn serialized_request_is_executable_and_rejects_a_nonsquare_risk_metric() {
        let (root, envelope, ir, operational) = fixture();
        let functional = calibration();
        let request = UniversalCapabilityCompilationRequest {
            schema: "cerebro.tidex.universal_capability_compilation_request/v1".into(),
            system_envelope: envelope,
            capability_ir: ir,
            operational_contract: operational,
            calibration: ReceiverCalibrationSet {
                receiver_snapshot_binding_sha256: None,
                functional_signatures: functional.clone(),
                receiver_solutions: functional
                    .iter()
                    .map(|row| receiver_solution(row))
                    .collect(),
                wrong_functional_signatures: vec![functional[0].clone(), functional[2].clone()],
            },
            protected_cortex: ProtectedCortex {
                parameter_importance: vec![0.0; 5],
                directions: Vec::new(),
                max_damage_ratio: 0.01,
            },
            risk_metric_rows: (0..5)
                .map(|row| (0..5).map(|column| f64::from(row == column)).collect())
                .collect(),
            policy: ReceiverCompilerPolicy {
                schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
                ridge: 1e-10,
                minimum_decoder_loo_r2: 0.999,
                minimum_encoder_loo_r2: 0.999,
                minimum_decoder_loo_cosine: 0.999,
                maximum_functional_relative_error: 1e-4,
                minimum_identity_margin: 0.05,
                maximum_quadratic_cost: 1e6,
            },
        };
        let restored: UniversalCapabilityCompilationRequest =
            serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
        assert!(compile_experimental_universal_capability_request(&restored)
            .unwrap()
            .is_experimentally_usable());
        let receipt = execute_experimental_universal_capability_request(&restored).unwrap();
        replay_experimental_universal_capability_request(&restored, &receipt).unwrap();

        let mut altered_receipt = receipt.clone();
        altered_receipt.compilation.disposition = UniversalCapabilityDisposition::Rejected;
        assert!(
            replay_experimental_universal_capability_request(&restored, &altered_receipt).is_err()
        );

        let mut malformed = restored;
        malformed.risk_metric_rows.pop();
        assert!(compile_experimental_universal_capability_request(&malformed).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn replayed_plan_materializes_only_the_compiler_target_delta() {
        let (root, envelope, ir, operational) = fixture();
        let functional = calibration();
        let geometry = ParameterLayoutArtifact::new(
            ParameterBlockLayout::from_shapes(&[BlockShapeSpec {
                name: "layers.0.receiver_coordinates".into(),
                shape: vec![5],
                count: 5,
            }])
            .unwrap(),
        )
        .unwrap();
        let profile = ReceiverProfile {
            schema: "cerebro.tidex.receiver_profile/v1".into(),
            model_id: ModelId::parse("receiver.v1").unwrap(),
            architecture_id: ArchitectureId::parse("transformer.v1").unwrap(),
            architecture: ReceiverArchitecture::Transformer,
            modalities: BTreeSet::from([CapabilityModality::Text]),
            supports_persistent_state: false,
            parameter_dimension: 5,
            regions: vec![ReceiverRegion {
                tensor_id: TensorId::parse("layers.0.receiver_coordinates").unwrap(),
                parameter_count: 5,
                supported_strategies: BTreeSet::from([
                    MaterializationStrategy::ReceiverCoordinates,
                    MaterializationStrategy::DenseDelta,
                ]),
            }],
        };
        let layout = ReceiverMaterializationLayout::create(
            &profile,
            geometry,
            vec![ReceiverTensorPhysicalSpec {
                tensor_id: TensorId::parse("layers.0.receiver_coordinates").unwrap(),
                encoding: ReceiverScalarEncoding::Floating {
                    scalar_type: FloatingScalarType::Float64,
                },
                partitioning: ReceiverTensorPartitioning::Replicated,
            }],
            vec![],
        )
        .unwrap();
        let snapshot = ReceiverSnapshotBinding::create(
            &profile,
            Sha256Digest::digest_bytes(b"model"),
            Sha256Digest::digest_bytes(b"config"),
            Sha256Digest::digest_bytes(b"tokenizer"),
            layout.manifest_sha256.clone(),
        )
        .unwrap();
        let requirements = CapabilityRequirements {
            schema: "cerebro.tidex.capability_requirements/v1".into(),
            capability_id: ir.capability_id().clone(),
            capability_ir_sha256: ir.manifest_digest().clone(),
            required_modalities: BTreeSet::from([CapabilityModality::Text]),
            requires_persistent_state: false,
            minimum_receiver_parameter_dimension: 5,
            acceptable_strategies: BTreeSet::from([
                MaterializationStrategy::ReceiverCoordinates,
                MaterializationStrategy::DenseDelta,
            ]),
        };
        let request = UniversalCapabilityPlanningRequest {
            schema: "cerebro.tidex.universal_capability_planning_request/v1".into(),
            compilation: UniversalCapabilityCompilationRequest {
                schema: "cerebro.tidex.universal_capability_compilation_request/v1".into(),
                system_envelope: envelope,
                capability_ir: ir,
                operational_contract: operational,
                calibration: ReceiverCalibrationSet {
                    receiver_snapshot_binding_sha256: Some(snapshot.manifest_digest().clone()),
                    functional_signatures: functional.clone(),
                    receiver_solutions: functional
                        .iter()
                        .map(|row| receiver_solution(row))
                        .collect(),
                    wrong_functional_signatures: vec![functional[0].clone(), functional[2].clone()],
                },
                protected_cortex: ProtectedCortex {
                    parameter_importance: vec![0.0; 5],
                    directions: Vec::new(),
                    max_damage_ratio: 0.01,
                },
                risk_metric_rows: (0..5)
                    .map(|row| (0..5).map(|column| f64::from(row == column)).collect())
                    .collect(),
                policy: ReceiverCompilerPolicy {
                    schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
                    ridge: 1e-10,
                    minimum_decoder_loo_r2: 0.999,
                    minimum_encoder_loo_r2: 0.999,
                    minimum_decoder_loo_cosine: 0.999,
                    maximum_functional_relative_error: 1e-4,
                    minimum_identity_margin: 0.05,
                    maximum_quadratic_cost: 1e6,
                },
            },
            receiver_profile: profile,
            receiver_snapshot: snapshot,
            capability_requirements: requirements,
            requested_strategy: MaterializationStrategy::ReceiverCoordinates,
            affected_regions: vec![TensorId::parse("layers.0.receiver_coordinates").unwrap()],
        };
        let receipt = execute_universal_capability_shadow_plan(&request).unwrap();
        let candidate =
            materialize_replayed_receiver_coordinates_shadow(&request, &receipt).unwrap();
        assert_eq!(
            candidate.coordinates,
            receipt
                .shadow_plan
                .compilation_receipt
                .compilation
                .receiver
                .target_delta
        );

        let mut tampered = receipt;
        tampered
            .shadow_plan
            .compilation_receipt
            .compilation
            .receiver
            .target_delta[0] += 1.0;
        assert!(materialize_replayed_receiver_coordinates_shadow(&request, &tampered).is_err());

        let mut dense_request = request;
        dense_request.requested_strategy = MaterializationStrategy::DenseDelta;
        let dense_receipt = execute_universal_capability_shadow_plan(&dense_request).unwrap();
        let dense_candidate =
            materialize_replayed_dense_delta_shadow(&dense_request, &dense_receipt, &layout)
                .unwrap();
        assert_eq!(dense_candidate.tensors.len(), 1);
        assert_eq!(
            dense_candidate.tensors[0].values,
            dense_receipt
                .shadow_plan
                .compilation_receipt
                .compilation
                .receiver
                .target_delta
        );
        let mut tampered_dense = dense_candidate.clone();
        tampered_dense.tensors[0].values[0] += 1.0;
        assert!(tampered_dense
            .validate(&dense_request, &dense_receipt, &layout)
            .is_err());

        let mut wrong_layout = layout.clone();
        wrong_layout.geometry.layout.blocks[0].shape = vec![1, 5];
        assert!(materialize_replayed_dense_delta_shadow(
            &dense_request,
            &dense_receipt,
            &wrong_layout
        )
        .is_err());

        let lab = root.join("lab");
        let state = lab.join("state");
        let artifacts = lab.join("artifacts");
        let production = root.join("production");
        for directory in [&lab, &state, &artifacts, &production] {
            fs::create_dir_all(directory).unwrap();
            crate::security::secure_dir(directory).unwrap();
        }
        let roots = LabRoots::open_for_test(&lab, &state, &artifacts, &production).unwrap();
        let reference = persist_dense_delta_shadow(
            &roots,
            &dense_request,
            &dense_receipt,
            &layout,
            &dense_candidate,
        )
        .unwrap();
        assert_eq!(
            load_dense_delta_shadow(&roots, &dense_request, &dense_receipt, &layout, &reference,)
                .unwrap(),
            dense_candidate
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn low_rank_plan_fixture(
        sparse: bool,
    ) -> (
        PathBuf,
        UniversalCapabilityPlanningRequest,
        ReceiverMaterializationLayout,
        LowRankShadowPolicy,
    ) {
        let (root, envelope, ir, operational) = fixture();
        let functional = calibration();
        let query = operational.canonical_transition_signature(&ir).unwrap();
        let left = if sparse {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [1.0, 2.0, 3.0, 4.0]
        };
        let right = [0.5, -1.0, 2.0, 0.25];
        let target = left
            .iter()
            .flat_map(|a| right.iter().map(move |b| a * b))
            .collect::<Vec<_>>();
        let query_norm_squared = query.iter().map(|v| v * v).sum::<f64>();
        let solve = |signature: &[f64]| {
            let coefficient =
                query.iter().zip(signature).map(|(a, b)| a * b).sum::<f64>() / query_norm_squared;
            (0..16)
                .map(|index| {
                    let embedded = signature.get(index).copied().unwrap_or(0.0);
                    let query_embedded = query.get(index).copied().unwrap_or(0.0);
                    embedded + (target[index] - query_embedded) * coefficient
                })
                .collect::<Vec<_>>()
        };
        let tensor_id = TensorId::parse("model.layers.0.self_attn.q_proj.weight").unwrap();
        let geometry = ParameterLayoutArtifact::new(
            ParameterBlockLayout::from_shapes(&[BlockShapeSpec {
                name: tensor_id.as_str().into(),
                shape: vec![4, 4],
                count: 16,
            }])
            .unwrap(),
        )
        .unwrap();
        let profile = ReceiverProfile {
            schema: "cerebro.tidex.receiver_profile/v1".into(),
            model_id: ModelId::parse("receiver.low-rank.v1").unwrap(),
            architecture_id: ArchitectureId::parse("transformer.v1").unwrap(),
            architecture: ReceiverArchitecture::Transformer,
            modalities: BTreeSet::from([CapabilityModality::Text]),
            supports_persistent_state: false,
            parameter_dimension: 16,
            regions: vec![ReceiverRegion {
                tensor_id: tensor_id.clone(),
                parameter_count: 16,
                supported_strategies: BTreeSet::from([MaterializationStrategy::LowRank]),
            }],
        };
        let layout = ReceiverMaterializationLayout::create(
            &profile,
            geometry,
            vec![ReceiverTensorPhysicalSpec {
                tensor_id: tensor_id.clone(),
                encoding: ReceiverScalarEncoding::Floating {
                    scalar_type: FloatingScalarType::Float64,
                },
                partitioning: ReceiverTensorPartitioning::Replicated,
            }],
            vec![],
        )
        .unwrap();
        let snapshot = ReceiverSnapshotBinding::create(
            &profile,
            Sha256Digest::digest_bytes(b"low-rank-model"),
            Sha256Digest::digest_bytes(b"config"),
            Sha256Digest::digest_bytes(b"tokenizer"),
            layout.manifest_sha256.clone(),
        )
        .unwrap();
        let request = UniversalCapabilityPlanningRequest {
            schema: "cerebro.tidex.universal_capability_planning_request/v1".into(),
            compilation: UniversalCapabilityCompilationRequest {
                schema: "cerebro.tidex.universal_capability_compilation_request/v1".into(),
                system_envelope: envelope,
                capability_ir: ir.clone(),
                operational_contract: operational,
                calibration: ReceiverCalibrationSet {
                    receiver_snapshot_binding_sha256: Some(snapshot.manifest_digest().clone()),
                    functional_signatures: functional.clone(),
                    receiver_solutions: functional.iter().map(|row| solve(row)).collect(),
                    wrong_functional_signatures: vec![functional[0].clone(), functional[2].clone()],
                },
                protected_cortex: ProtectedCortex {
                    parameter_importance: vec![0.0; 16],
                    directions: vec![],
                    max_damage_ratio: 0.01,
                },
                risk_metric_rows: (0..16)
                    .map(|row| (0..16).map(|column| f64::from(row == column)).collect())
                    .collect(),
                policy: ReceiverCompilerPolicy {
                    schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
                    ridge: 1e-10,
                    minimum_decoder_loo_r2: 0.999,
                    minimum_encoder_loo_r2: 0.999,
                    minimum_decoder_loo_cosine: 0.999,
                    maximum_functional_relative_error: 1e-4,
                    minimum_identity_margin: 0.05,
                    maximum_quadratic_cost: 1e9,
                },
            },
            receiver_profile: profile,
            receiver_snapshot: snapshot,
            capability_requirements: CapabilityRequirements {
                schema: "cerebro.tidex.capability_requirements/v1".into(),
                capability_id: ir.capability_id().clone(),
                capability_ir_sha256: ir.manifest_digest().clone(),
                required_modalities: BTreeSet::from([CapabilityModality::Text]),
                requires_persistent_state: false,
                minimum_receiver_parameter_dimension: 16,
                acceptable_strategies: BTreeSet::from([MaterializationStrategy::LowRank]),
            },
            requested_strategy: MaterializationStrategy::LowRank,
            affected_regions: vec![tensor_id],
        };
        let policy = LowRankShadowPolicy {
            schema: "cerebro.tidex.low_rank_shadow_policy/v1".into(),
            maximum_rank: 1,
            relative_reconstruction_tolerance: 1e-6,
            absolute_reconstruction_tolerance: 1e-6,
            minimum_parameter_reduction_ratio: 0.4,
            maximum_svd_sweeps: 100,
        };
        (root, request, layout, policy)
    }

    #[test]
    fn low_rank_full_chain_replays_persists_reloads_and_detects_tampering() {
        let (root, request, layout, policy) = low_rank_plan_fixture(false);
        let receipt = execute_universal_capability_shadow_plan(&request).unwrap();
        let candidate =
            materialize_replayed_low_rank_shadow(&request, &receipt, &layout, &policy).unwrap();
        let lab = root.join("lab");
        let state = lab.join("state");
        let artifacts = lab.join("artifacts");
        let production = root.join("production");
        for directory in [&lab, &state, &artifacts, &production] {
            fs::create_dir_all(directory).unwrap();
            crate::security::secure_dir(directory).unwrap();
        }
        let roots = LabRoots::open_for_test(&lab, &state, &artifacts, &production).unwrap();
        let reference =
            persist_low_rank_shadow(&roots, &request, &receipt, &layout, &policy, &candidate)
                .unwrap();
        assert_eq!(
            load_low_rank_shadow(&roots, &request, &receipt, &layout, &policy, &reference)
                .unwrap_or_else(|error| panic!("low-rank reload failed: {error:?}")),
            candidate
        );
        let mut bad_reference = reference;
        bad_reference.sha256 = Sha256Digest::zero();
        assert!(
            load_low_rank_shadow(&roots, &request, &receipt, &layout, &policy, &bad_reference)
                .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
    // This is an integration test over an explicitly constructed linear domain,
    // not a language-model quality benchmark or a target-independent claim.
    fn physical_checkpoint_fixture() -> (
        PathBuf,
        crate::checkpoint_adapter::CompiledCheckpointRequest,
    ) {
        use crate::checkpoint_adapter::{
            project_authenticated_receiver, CompiledCheckpointBackend, CompiledCheckpointRequest,
            PhysicalReceiverProjectionInput,
        };
        use crate::model_adaptation::{
            profile_receiver_model, ReceiverModelProfileInput, RECEIVER_MODEL_PROFILE_INPUT_SCHEMA,
        };
        use std::io::Write;
        let (root, mut planning, _layout, policy) = low_rank_plan_fixture(true);
        crate::security::secure_dir(&root).unwrap();
        let checkpoint = root.join("physical-base.safetensors");
        let config = root.join("physical-config.json");
        let tokenizer = root.join("physical-tokenizer.json");
        let header = serde_json::to_vec(&serde_json::json!({
            "__metadata__":{"format":"pt"},
            "model.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[4,4],"data_offsets":[0,64]},
            "model.layers.0.input_layernorm.weight":{"dtype":"F32","shape":[2],"data_offsets":[64,72]}
        })).unwrap();
        let mut file = fs::File::create(&checkpoint).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for value in [2.0f32; 16].into_iter().chain([1.0, 3.0]) {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.sync_all().unwrap();
        fs::write(&config, br#"{"model_type":"llama","num_hidden_layers":1,"hidden_size":4,"vocab_size":8,"tie_word_embeddings":false}"#).unwrap();
        fs::write(&tokenizer, br#"{"version":"1.0","model":{"type":"test"}}"#).unwrap();
        let physical = profile_receiver_model(
            &root,
            &ReceiverModelProfileInput {
                schema: RECEIVER_MODEL_PROFILE_INPUT_SCHEMA.into(),
                model_id: planning.receiver_profile.model_id.clone(),
                architecture_id: planning.receiver_profile.architecture_id.clone(),
                source_revision: None,
                checkpoint_path: checkpoint,
                config_path: config,
                tokenizer_path: tokenizer,
            },
        )
        .unwrap();
        let projection = project_authenticated_receiver(
            &root,
            &PhysicalReceiverProjectionInput {
                schema: "cerebro.tidex.physical_receiver_projection_input/v1".into(),
                physical_profile: physical.profile_reference.clone(),
                tensor_ids: planning.affected_regions.clone(),
                modalities: BTreeSet::from([CapabilityModality::Text]),
            },
        )
        .unwrap();
        planning.receiver_profile = projection.profile;
        planning.receiver_snapshot = projection.snapshot;
        planning
            .compilation
            .calibration
            .receiver_snapshot_binding_sha256 =
            Some(planning.receiver_snapshot.manifest_digest().clone());
        let planning_receipt = execute_universal_capability_shadow_plan(&planning).unwrap();
        let output_path = root.join("physical-candidate.safetensors");
        (
            root,
            CompiledCheckpointRequest {
                schema: "cerebro.tidex.compiled_checkpoint_request/v1".into(),
                physical_profile: physical.profile_reference,
                planning,
                planning_receipt,
                layout: projection.layout,
                backend: CompiledCheckpointBackend::LowRank { policy },
                maximum_absolute_error: 1e-4,
                maximum_relative_error: 1e-4,
                output_path,
            },
        )
    }

    #[test]
    fn convergence_physical_backends_write_real_checkpoints_and_replay_without_activation() {
        use crate::checkpoint_adapter::{
            authenticate_compiled_checkpoint, materialize_compiled_checkpoint,
            CompiledCheckpointBackend,
        };
        use crate::weight_actuator::read_model_tensor_f32;
        let (root, template) = physical_checkpoint_fixture();
        let base_path = root.join("physical-base.safetensors");
        let base_bytes = fs::read(&base_path).unwrap();
        let low_rank = template.backend.clone();
        let sparse = CompiledCheckpointBackend::SparseDelta {
            policy: crate::sparse_shadow_materializer::SparseShadowPolicy {
                schema: "cerebro.tidex.sparse_shadow_policy/v1".into(),
                maximum_nonzero_count: 4,
                maximum_density: 0.25,
                absolute_zero_threshold: 1e-8,
                relative_reconstruction_tolerance: 1e-5,
                absolute_reconstruction_tolerance: 1e-5,
                minimum_storage_reduction_ratio: 0.4,
            },
        };
        for (index, (strategy, backend)) in [
            (
                MaterializationStrategy::ReceiverCoordinates,
                CompiledCheckpointBackend::ReceiverCoordinates,
            ),
            (
                MaterializationStrategy::DenseDelta,
                CompiledCheckpointBackend::DenseDelta,
            ),
            (MaterializationStrategy::LowRank, low_rank),
            (MaterializationStrategy::SparseDelta, sparse),
        ]
        .into_iter()
        .enumerate()
        {
            let mut request = template.clone();
            request.backend = backend;
            request.planning.requested_strategy = strategy;
            request
                .planning
                .capability_requirements
                .acceptable_strategies = BTreeSet::from([strategy]);
            request.planning_receipt =
                execute_universal_capability_shadow_plan(&request.planning).unwrap();
            request.output_path = root.join(format!("candidate-{index}.safetensors"));
            let reference = materialize_compiled_checkpoint(&root, &request).unwrap();
            let receipt = authenticate_compiled_checkpoint(&root, &reference).unwrap();
            assert!(!receipt.authorizes_promotion);
            assert!(!receipt.authorizes_target_update_free_claim);
            assert!(!receipt.model_execution_verified);
            assert!(!receipt.materialization.requires_adapter_at_runtime);
            let q = &request.planning.affected_regions[0];
            let values = read_model_tensor_f32(&request.output_path, q).unwrap();
            let delta = crate::artifact::read_dvec_f32(&root, &receipt.dense_delta).unwrap();
            for (actual, update) in values.iter().zip(delta) {
                assert_eq!(actual.to_bits(), (2.0f32 + update).to_bits());
            }
            let untouched = TensorId::parse("model.layers.0.input_layernorm.weight").unwrap();
            assert_eq!(
                read_model_tensor_f32(&request.output_path, &untouched).unwrap(),
                vec![1.0, 3.0]
            );
            assert_eq!(fs::read(&base_path).unwrap(), base_bytes);
            let output_bytes = fs::read(&request.output_path).unwrap();
            assert!(materialize_compiled_checkpoint(&root, &request).is_err());
            assert_eq!(fs::read(&request.output_path).unwrap(), output_bytes);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn convergence_physical_bridge_rejects_rebound_false_layout_and_escaped_output() {
        use crate::checkpoint_adapter::materialize_compiled_checkpoint;
        let (root, request) = physical_checkpoint_fixture();
        let mut wrong = request.clone();
        wrong.output_path = root
            .parent()
            .unwrap()
            .join("must-not-create-compiled-checkpoint.safetensors");
        assert!(materialize_compiled_checkpoint(&root, &wrong).is_err());
        let mut physical = request.layout.physical_tensors.clone();
        physical[0].encoding = ReceiverScalarEncoding::Quantized {
            bits: 8,
            group_size: 4,
            scale_count: 4,
            symmetric: true,
        };
        wrong = request.clone();
        wrong.layout = ReceiverMaterializationLayout::create(
            &wrong.planning.receiver_profile,
            request.layout.geometry.clone(),
            physical,
            vec![],
        )
        .unwrap();
        wrong.planning.receiver_snapshot = ReceiverSnapshotBinding::create(
            &wrong.planning.receiver_profile,
            request
                .planning
                .receiver_snapshot
                .model_snapshot_sha256
                .clone(),
            request
                .planning
                .receiver_snapshot
                .configuration_sha256
                .clone(),
            request.planning.receiver_snapshot.tokenizer_sha256.clone(),
            wrong.layout.manifest_sha256.clone(),
        )
        .unwrap();
        wrong
            .planning
            .compilation
            .calibration
            .receiver_snapshot_binding_sha256 =
            Some(wrong.planning.receiver_snapshot.manifest_digest().clone());
        wrong.planning_receipt = execute_universal_capability_shadow_plan(&wrong.planning).unwrap();
        let error = materialize_compiled_checkpoint(&root, &wrong)
            .unwrap_err()
            .to_string();
        assert!(error.contains("physical_binding_mismatch"), "{error}");
        assert!(!request.output_path.exists());
        fs::write(root.join("physical-tokenizer.json"), b"changed-tokenizer").unwrap();
        assert!(materialize_compiled_checkpoint(&root, &request).is_err());
        assert!(!request.output_path.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
