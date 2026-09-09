//! Candidate-only binding from measured functional responses to real weights.
//!
//! Reuses the receiver compiler, ParameterBlockLayout, immutable dvec algebra
//! and WeightActuator. It does not infer that a response vector is a whole
//! capability. Imported measurements are byte-authenticated, not independently
//! attested. A candidate MUST undergo independent model execution before any
//! behavioral claim; this module has no promotion operation.

use crate::acquisition_contract::{
    AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
    SystemEnvelope,
};
use crate::artifact::{
    derive_content_addressed_dvec_combination, verify_dvec_reference_under_root,
    ArtifactWriteAuthority, DeltaArtifactRef,
};
use crate::authority::{
    ensure_private_parent, root_relative_path, write_or_verify_immutable, PrivateFileReference,
};
use crate::block_tomography::{parameter_layout_digest, ParameterBlockLayout};
use crate::capability_bundle::{
    authenticate_capability_bundle, CapabilityBlocker, CapabilityBundle,
    CapabilityRepresentationGap, PartialCapabilityBundleDraft,
};
use crate::capability_ir::{
    authenticate_capability_ir, execute_linear_readout, CapabilityIr, LinearReadoutExecution,
};
use crate::content_vault::{
    authenticate_capture_receipt, capture_to_vault, retained_source_adapter,
};
use crate::contracts::ProtectedCortex;
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::identity::{AcquisitionId, CapabilityId, ObservationId, ProbeId, SkillId, TensorId};
use crate::linalg::{dot, norm, Matrix};
use crate::pure_capability_e2e::{LinearMapDescriptor, PureCapabilityDiscoveryAuthority};
use crate::receiver_compiler::{
    compile_receiver_readout_capability, compile_receiver_signature,
    compile_receiver_signature_in_safe_coordinates, project_functional_signature,
    ReceiverCalibrationSet, ReceiverCompilerPolicy, ReceiverProposalMethod,
    ReceiverReadoutCapabilityInput, ReceiverSignatureCompilation,
};
use crate::security::verify_internal_private_root;
use crate::transport::functional_support_envelope;
use crate::weight_actuator::{
    authenticate_lora_adapter_axis_receipt, inspect_linear_readout, inspect_model_safetensors,
    materialize_dense_delta_checkpoint, LinearReadoutInspection, LoraAdapterAxisReceipt,
    WeightMaterializationReceipt,
};
#[cfg(test)]
use crate::weight_actuator::{
    import_peft_lora_as_dense_axis, LoraAdapterAxisInput, LORA_ADAPTER_AXIS_INPUT_SCHEMA,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const MAX_RECORD_BYTES: u64 = 8 * 1024 * 1024;
const MAX_AXES: usize = 64;
const MAX_ANCHORS: usize = 128;
const MAX_RESPONSE_DIMENSION: usize = 128;
const BASIS_SCHEMA_V1: &str = "cerebro.tidex.receiver_weight_basis/v1";
const BASIS_SCHEMA: &str = "cerebro.tidex.receiver_weight_basis/v2";
const REALIZATION_SCHEMA: &str = "cerebro.tidex.receiver_realization_manifest/v1";
const PROTOCOL_SCHEMA: &str = "cerebro.tidex.receiver_response_protocol/v1";
const CROSS_MODEL_PROTOCOL_SCHEMA: &str =
    "cerebro.tidex.cross_model_functional_signature_protocol/v2";
const CROSS_MODEL_FUNCTIONAL_EVIDENCE_SCHEMA: &str =
    "cerebro.tidex.cross_model_functional_evidence/v2";
const CROSS_MODEL_RECEIVER_SOLUTION_EVIDENCE_SCHEMA: &str =
    "cerebro.tidex.cross_model_receiver_solution_evidence/v1";
const RECEIVER_BEHAVIORAL_CALIBRATION_EVIDENCE_SCHEMA: &str =
    "cerebro.tidex.receiver_behavioral_calibration_evidence/v1";
const CROSS_MODEL_PROJECTION_SCHEMA: &str = "cerebro.tidex.v69_donor_functional_projection/v2";
const PROJECTION_ARITHMETIC: &str = "f64_sequential_sub_mul_add/v1";
const TARGET_SCHEMA: &str = "cerebro.tidex.functional_response_target/v1";
const OBSERVATION_SCHEMA: &str = "cerebro.tidex.receiver_response_observation/v1";
pub const REQUEST_SCHEMA: &str = "cerebro.tidex.receiver_weight_request/v1";
const CANDIDATE_SCHEMA: &str = "cerebro.tidex.receiver_weight_candidate/v1";

fn invalid(code: &str) -> BrainError {
    BrainError::Invalid(code.into())
}
fn integrity(code: &str) -> BrainError {
    BrainError::Integrity(code.into())
}
fn read_record<T: DeserializeOwned>(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<T> {
    Ok(serde_json::from_slice(
        &reference.read_verified_bounded(root, MAX_RECORD_BYTES)?,
    )?)
}

const OBSERVED_LINEAR_READOUT_SCHEMA: &str = "cerebro.tidex.observed_linear_readout/v1";
const READOUT_ACQUISITION_SCHEMA: &str = "cerebro.tidex.linear_readout_acquisition/v1";
const READOUT_EVIDENCE_SCOPE: &str =
    "locally_captured_readout_parameters_and_activations_not_independently_attested";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DescribeLinearReadoutInput {
    pub schema: String,
    pub model_path: PathBuf,
    pub tensor_id: TensorId,
    pub positive_row: usize,
    pub negative_row: usize,
    pub capability_id: CapabilityId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DescribeLinearReadoutReport {
    pub schema: String,
    /// Exact compact encoding consumed by PureCapabilityDiscoveryAuthority.
    pub descriptor_json: String,
    pub inspection: LinearReadoutInspection,
}

pub fn describe_linear_readout(
    input: &DescribeLinearReadoutInput,
) -> BrainResult<DescribeLinearReadoutReport> {
    if input.schema != "cerebro.tidex.describe_linear_readout_input/v1" {
        return Err(invalid("describe_linear_readout_schema"));
    }
    let inspection = inspect_linear_readout(
        &input.model_path,
        &input.tensor_id,
        input.positive_row,
        input.negative_row,
    )?;
    let descriptor = LinearMapDescriptor::new(
        input.capability_id.clone(),
        inspection.input_dimension,
        1,
        inspection.difference_weights.clone(),
    )?;
    Ok(DescribeLinearReadoutReport {
        schema: "cerebro.tidex.describe_linear_readout_report/v1".into(),
        descriptor_json: serde_json::to_string(&descriptor)?,
        inspection,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AcquireLinearReadoutInput {
    pub schema: String,
    pub source_root: PathBuf,
    pub model_path: PathBuf,
    pub descriptor_relative_path: PathBuf,
    pub evidence_relative_path: PathBuf,
}

/// An exported observation is authenticated as bytes and checked against real
/// resident readout weights. Its claimed hidden-state origin is not thereby
/// independently attested. No target labels or receiver observations enter it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObservedLinearReadout {
    pub schema: String,
    pub capability_id: CapabilityId,
    pub model_sha256: Sha256Digest,
    pub tensor_id: TensorId,
    pub positive_row: usize,
    pub negative_row: usize,
    pub arithmetic_profile: String,
    pub inputs: Vec<Vec<f64>>,
    pub observed_positive_logits: Vec<f64>,
    pub observed_negative_logits: Vec<f64>,
    pub observed_margins: Vec<f64>,
    pub prompts: Vec<String>,
    pub prompt_sha256: Vec<Sha256Digest>,
    pub receiver_data_used: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinearReadoutAcquisitionReceipt {
    pub schema: String,
    pub model_path: PathBuf,
    pub model_sha256: Sha256Digest,
    pub capability_id: CapabilityId,
    pub capture_receipt: PrivateFileReference,
    pub discovery: PrivateFileReference,
    pub capability_ir: PrivateFileReference,
    pub source_projection: PathBuf,
    pub descriptor_relative_path: PathBuf,
    pub evidence_relative_path: PathBuf,
    pub descriptor: PrivateFileReference,
    pub evidence: PrivateFileReference,
    pub partial_bundle: PrivateFileReference,
    pub execution: LinearReadoutExecution,
    pub observed_forward_maximum_absolute_error: f64,
    pub maximum_roundoff_bound: f64,
    pub evidence_scope: String,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverCapabilityIrReadoutSummary {
    pub acquisition: PrivateFileReference,
    pub capability_ir: PrivateFileReference,
    pub partial_bundle: PrivateFileReference,
    pub execution: LinearReadoutExecution,
    pub requested_signature: Vec<f64>,
    pub observed_forward_maximum_absolute_error: f64,
    pub maximum_roundoff_bound: f64,
    pub evidence_scope: String,
}

struct ReadoutSourceChain {
    model_path: PathBuf,
    capture_receipt: PrivateFileReference,
    discovery: PrivateFileReference,
    capability_ir: PrivateFileReference,
    source_projection: PathBuf,
    descriptor_relative_path: PathBuf,
    evidence_relative_path: PathBuf,
    descriptor: PrivateFileReference,
    evidence: PrivateFileReference,
    partial_bundle: PrivateFileReference,
}

struct AuthenticatedLinearReadout {
    receipt: LinearReadoutAcquisitionReceipt,
    ir: CapabilityIr,
    envelope: SystemEnvelope,
    weights: Vec<f64>,
    observed: ObservedLinearReadout,
}

fn readout_gaps() -> BTreeSet<CapabilityRepresentationGap> {
    BTreeSet::from([
        CapabilityRepresentationGap::StateSemantics,
        CapabilityRepresentationGap::BehavioralEquivalence,
        CapabilityRepresentationGap::CausalContribution,
        CapabilityRepresentationGap::TargetCompatibility,
    ])
}

fn readout_partial_bundle(
    root: &Path,
    capture: &PrivateFileReference,
    ir_reference: &PrivateFileReference,
    ir: &CapabilityIr,
) -> BrainResult<CapabilityBundle> {
    CapabilityBundle::create_partial(
        root,
        PartialCapabilityBundleDraft {
            capability_id: ir.capability_id().clone(),
            capture_receipt: capture.clone(),
            capability_ir: Some(ir_reference.clone()),
            capability_ir_sha256: Some(ir.manifest_digest().clone()),
            gaps: readout_gaps(),
            blockers: BTreeSet::from([CapabilityBlocker::TargetCompatibilityUnresolved]),
        },
    )
}

fn exact_f32(value: f64) -> bool {
    value.is_finite() && f64::from(value as f32).to_bits() == value.to_bits()
}

/// Standard forward bound for a dot product over exact F32 operands. gamma_d
/// covers rounded products and reduction; the absolute term also covers
/// gradual underflow. The F64 reference uses the existing scaled compensated
/// dot product, conservatively enclosed by gamma_(8d+8).
fn readout_dot_roundoff_bound(weights: &[f64], input: &[f64]) -> BrainResult<f64> {
    let dimension = weights.len();
    if dimension == 0 || dimension != input.len() || dimension > 4_096 {
        return Err(invalid("readout_dot_bound_shape"));
    }
    let absolute_weights = weights.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let absolute_input = input.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let absolute_sum = dot(&absolute_weights, &absolute_input)?;
    let unit32 = 2.0_f64.powi(-24);
    let unit64 = 2.0_f64.powi(-53);
    let count32 = dimension as f64;
    let count64 = (8 * dimension + 8) as f64;
    let gamma32 = (count32 * unit32) / (1.0 - count32 * unit32);
    let gamma64 = (count64 * unit64) / (1.0 - count64 * unit64);
    let upper_sum = absolute_sum / (1.0 - gamma64);
    let underflow = (2 * dimension + 1) as f64 * 2.0_f64.powi(-150);
    let bound = (gamma32 + gamma64) * upper_sum + underflow;
    if !bound.is_finite() || bound < 0.0 {
        return Err(invalid("readout_dot_bound_nonfinite"));
    }
    Ok(bound)
}

fn verify_observed_linear_readout(
    observed: &ObservedLinearReadout,
    inspection: &LinearReadoutInspection,
    execution: &LinearReadoutExecution,
) -> BrainResult<(f64, f64)> {
    let n = observed.inputs.len();
    if observed.schema != OBSERVED_LINEAR_READOUT_SCHEMA
        || observed.arithmetic_profile != "f32_cpu_no_tf32/v1"
        || observed.receiver_data_used
        || observed.model_sha256 != inspection.model_sha256
        || observed.tensor_id != inspection.tensor_id
        || observed.positive_row != inspection.positive_row
        || observed.negative_row != inspection.negative_row
        || n == 0
        || n > MAX_RESPONSE_DIMENSION
        || observed.observed_positive_logits.len() != n
        || observed.observed_negative_logits.len() != n
        || observed.observed_margins.len() != n
        || observed.prompts.len() != n
        || observed.prompt_sha256.len() != n
        || execution.raw_margins.len() != n
        || observed.inputs.iter().any(|row| {
            row.len() != inspection.input_dimension || row.iter().any(|value| !exact_f32(*value))
        })
        || observed
            .observed_positive_logits
            .iter()
            .chain(&observed.observed_negative_logits)
            .chain(&observed.observed_margins)
            .any(|value| !exact_f32(*value))
    {
        return Err(invalid("observed_linear_readout_shape_or_binding"));
    }
    let mut prompt_ids = BTreeSet::new();
    let mut maximum_error = 0.0_f64;
    let mut maximum_bound = 0.0_f64;
    for index in 0..n {
        let prompt = &observed.prompts[index];
        if prompt.is_empty()
            || prompt.len() > 131_072
            || Sha256Digest::digest_bytes(prompt.as_bytes()) != observed.prompt_sha256[index]
            || !prompt_ids.insert(&observed.prompt_sha256[index])
        {
            return Err(integrity("observed_linear_readout_prompt_binding"));
        }
        let input = &observed.inputs[index];
        let positive = dot(&inspection.positive_weights, input)?;
        let negative = dot(&inspection.negative_weights, input)?;
        let positive_bound = readout_dot_roundoff_bound(&inspection.positive_weights, input)?;
        let negative_bound = readout_dot_roundoff_bound(&inspection.negative_weights, input)?;
        let observed_positive = observed.observed_positive_logits[index];
        let observed_negative = observed.observed_negative_logits[index];
        let positive_error = (positive - observed_positive).abs();
        let negative_error = (negative - observed_negative).abs();
        if positive_error > positive_bound || negative_error > negative_bound {
            return Err(integrity("observed_linear_readout_forward_mismatch"));
        }
        let f32_margin = f64::from((observed_positive as f32) - (observed_negative as f32));
        if f32_margin.to_bits() != observed.observed_margins[index].to_bits() {
            return Err(integrity(
                "observed_linear_readout_margin_arithmetic_mismatch",
            ));
        }
        let unit32 = 2.0_f64.powi(-24);
        let subtraction_bound = unit32 / (1.0 - unit32)
            * (observed_positive.abs() + observed_negative.abs())
            + 2.0_f64.powi(-150);
        let reference_difference_bound =
            readout_dot_roundoff_bound(&inspection.difference_weights, input)?;
        let margin_bound =
            positive_bound + negative_bound + subtraction_bound + reference_difference_bound;
        let margin_error = (execution.raw_margins[index] - f32_margin).abs();
        if !margin_error.is_finite() || !margin_bound.is_finite() || margin_error > margin_bound {
            return Err(integrity("observed_linear_readout_ir_margin_mismatch"));
        }
        maximum_error = maximum_error
            .max(positive_error)
            .max(negative_error)
            .max(margin_error);
        maximum_bound = maximum_bound
            .max(positive_bound)
            .max(negative_bound)
            .max(margin_bound);
    }
    Ok((maximum_error, maximum_bound))
}

fn readout_acquisition_path(root: &Path, digest: &Sha256Digest) -> PathBuf {
    root.join("state/linear_readout_acquisitions/by-sha")
        .join(format!("{digest}.json"))
}

fn derive_linear_readout_acquisition(
    root: &Path,
    chain: ReadoutSourceChain,
) -> BrainResult<AuthenticatedLinearReadout> {
    if !chain.model_path.is_absolute() {
        return Err(invalid("linear_readout_model_path_not_absolute"));
    }
    let capture = authenticate_capture_receipt(root, &chain.capture_receipt)?;
    let scope = AcquisitionScope::declared_paths(vec![
        chain.descriptor_relative_path.clone(),
        chain.evidence_relative_path.clone(),
    ])?;
    if chain.descriptor_relative_path == chain.evidence_relative_path
        || capture.request().scope() != &scope
        || capture.request().requested_residency() != RequestedResidency::PortableIrOnly
        || capture.request().noise_policy() != NoisePolicy::ExplicitOnly
        || capture.request().budget().max_files != 2
        || capture.request().budget().max_total_bytes != MAX_RECORD_BYTES
        || !capture.request().exclusions().is_empty()
        || capture.objects().len() != 2
    {
        return Err(integrity("linear_readout_capture_scope_mismatch"));
    }
    let source = retained_source_adapter(root, &capture)?;
    // Verify the already-created canonical projection first: authentication
    // must not recreate missing data through discover_linear_map's adapter.
    source.verify_materialized(&chain.source_projection)?;
    for (relative, reference) in [
        (&chain.descriptor_relative_path, &chain.descriptor),
        (&chain.evidence_relative_path, &chain.evidence),
    ] {
        let object = capture
            .objects()
            .iter()
            .find(|object| object.source_relative_path() == relative)
            .ok_or_else(|| integrity("linear_readout_capture_object_missing"))?;
        if reference.path != chain.source_projection.join(relative)
            || &reference.sha256 != object.content_sha256()
        {
            return Err(integrity("linear_readout_captured_reference_mismatch"));
        }
    }
    let discovered =
        PureCapabilityDiscoveryAuthority::open(root)?.authenticate(&chain.discovery)?;
    // The discovery authority owns this wire type's private fields. Project
    // only its already-authenticated lineage, without minting another receipt.
    let lineage = serde_json::to_value(discovered)?;
    if lineage["capture_receipt"] != serde_json::to_value(&chain.capture_receipt)?
        || lineage["capability_ir"] != serde_json::to_value(&chain.capability_ir)?
        || lineage["descriptor_relative_path"]
            != serde_json::to_value(&chain.descriptor_relative_path)?
    {
        return Err(integrity("linear_readout_discovery_chain_mismatch"));
    }
    let ir = authenticate_capability_ir(root, &chain.capability_ir, capture.envelope())?;
    let descriptor_bytes = chain
        .descriptor
        .read_verified_bounded(root, MAX_RECORD_BYTES)?;
    let descriptor: LinearMapDescriptor = serde_json::from_slice(&descriptor_bytes)?;
    let observed: ObservedLinearReadout = read_record(root, &chain.evidence)?;
    if observed.capability_id != *ir.capability_id()
        || descriptor.capability_id() != ir.capability_id()
        || observed.inputs.is_empty()
        || observed.inputs.len() > 128
        || observed.inputs[0].is_empty()
        || observed.inputs[0].len() > 4_096
    {
        return Err(integrity("linear_readout_capability_or_dimension_mismatch"));
    }
    let inspection = inspect_linear_readout(
        &chain.model_path,
        &observed.tensor_id,
        observed.positive_row,
        observed.negative_row,
    )?;
    let expected_descriptor = LinearMapDescriptor::new(
        observed.capability_id.clone(),
        inspection.input_dimension,
        1,
        inspection.difference_weights.clone(),
    )?;
    if descriptor != expected_descriptor
        || serde_json::to_vec(&expected_descriptor)? != descriptor_bytes
    {
        return Err(integrity("linear_readout_descriptor_model_mismatch"));
    }
    let execution = execute_linear_readout(
        &ir,
        capture.envelope(),
        &inspection.difference_weights,
        &observed.inputs,
    )?;
    let (maximum_error, maximum_bound) =
        verify_observed_linear_readout(&observed, &inspection, &execution)?;
    let partial = authenticate_capability_bundle(root, &chain.partial_bundle)?;
    let expected_partial =
        readout_partial_bundle(root, &chain.capture_receipt, &chain.capability_ir, &ir)?;
    if partial != expected_partial {
        return Err(integrity("linear_readout_partial_bundle_mismatch"));
    }
    source.verify_materialized(&chain.source_projection)?;
    Ok(AuthenticatedLinearReadout {
        receipt: LinearReadoutAcquisitionReceipt {
            schema: READOUT_ACQUISITION_SCHEMA.into(),
            model_path: chain.model_path,
            model_sha256: inspection.model_sha256,
            capability_id: observed.capability_id.clone(),
            capture_receipt: chain.capture_receipt,
            discovery: chain.discovery,
            capability_ir: chain.capability_ir,
            source_projection: chain.source_projection,
            descriptor_relative_path: chain.descriptor_relative_path,
            evidence_relative_path: chain.evidence_relative_path,
            descriptor: chain.descriptor,
            evidence: chain.evidence,
            partial_bundle: chain.partial_bundle,
            execution,
            observed_forward_maximum_absolute_error: maximum_error,
            maximum_roundoff_bound: maximum_bound,
            evidence_scope: READOUT_EVIDENCE_SCOPE.into(),
            authorizes_promotion: false,
        },
        ir,
        envelope: capture.envelope().clone(),
        weights: inspection.difference_weights,
        observed,
    })
}

pub fn acquire_linear_readout(
    root: &Path,
    input: &AcquireLinearReadoutInput,
) -> BrainResult<PrivateFileReference> {
    let root = verify_internal_private_root(root)?;
    if input.schema != "cerebro.tidex.acquire_linear_readout_input/v1"
        || input.descriptor_relative_path == input.evidence_relative_path
        || !input.model_path.is_absolute()
    {
        return Err(invalid("acquire_linear_readout_input_invalid"));
    }
    let scope = AcquisitionScope::declared_paths(vec![
        input.descriptor_relative_path.clone(),
        input.evidence_relative_path.clone(),
    ])?;
    let request_id = Sha256Digest::digest_bytes(&serde_json::to_vec(input)?);
    let request = AcquisitionRequest::new(
        AcquisitionId::parse(format!("readout-{}", request_id.as_str()))?,
        scope,
        RequestedResidency::PortableIrOnly,
        NoisePolicy::ExplicitOnly,
        AcquisitionBudget {
            max_files: 2,
            max_total_bytes: MAX_RECORD_BYTES,
        },
        vec![],
    )?;
    let capture = capture_to_vault(&input.source_root, &root, &request)?;
    let capture_reference = capture.persist(&root)?;
    let source = retained_source_adapter(&root, &capture)?;
    let discovery = PureCapabilityDiscoveryAuthority::open(&root)?.discover_linear_map(
        capture_reference.clone(),
        &source,
        &input.descriptor_relative_path,
    )?;
    let source_projection = source.materialize()?;
    let ir = authenticate_capability_ir(&root, &discovery.capability_ir, capture.envelope())?;
    let partial = readout_partial_bundle(&root, &capture_reference, &discovery.capability_ir, &ir)?;
    let partial_bundle = partial.persist(&root)?;
    let reference_for = |relative: &Path| -> BrainResult<PrivateFileReference> {
        let object = capture
            .objects()
            .iter()
            .find(|object| object.source_relative_path() == relative)
            .ok_or_else(|| integrity("linear_readout_capture_object_missing"))?;
        Ok(PrivateFileReference::new(
            source_projection.join(relative),
            object.content_sha256().clone(),
        ))
    };
    let chain = ReadoutSourceChain {
        model_path: input.model_path.clone(),
        capture_receipt: capture_reference,
        discovery: discovery.receipt,
        capability_ir: discovery.capability_ir,
        descriptor: reference_for(&input.descriptor_relative_path)?,
        evidence: reference_for(&input.evidence_relative_path)?,
        source_projection,
        descriptor_relative_path: input.descriptor_relative_path.clone(),
        evidence_relative_path: input.evidence_relative_path.clone(),
        partial_bundle,
    };
    let derived = derive_linear_readout_acquisition(&root, chain)?;
    let bytes = serde_json::to_vec(&derived.receipt)?;
    let sha256 = Sha256Digest::digest_bytes(&bytes);
    let path = readout_acquisition_path(&root, &sha256);
    let actual = write_or_verify_immutable(&root, &path, &bytes)?;
    if actual != sha256 {
        return Err(integrity("linear_readout_acquisition_write_mismatch"));
    }
    Ok(PrivateFileReference::new(path, sha256))
}

fn authenticate_linear_readout_details(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<AuthenticatedLinearReadout> {
    let bytes = reference.read_verified_bounded(root, MAX_RECORD_BYTES)?;
    let receipt: LinearReadoutAcquisitionReceipt = serde_json::from_slice(&bytes)?;
    if receipt.schema != READOUT_ACQUISITION_SCHEMA
        || reference.path != readout_acquisition_path(root, &reference.sha256)
        || serde_json::to_vec(&receipt)? != bytes
    {
        return Err(integrity("linear_readout_acquisition_noncanonical"));
    }
    let chain = ReadoutSourceChain {
        model_path: receipt.model_path.clone(),
        capture_receipt: receipt.capture_receipt.clone(),
        discovery: receipt.discovery.clone(),
        capability_ir: receipt.capability_ir.clone(),
        source_projection: receipt.source_projection.clone(),
        descriptor_relative_path: receipt.descriptor_relative_path.clone(),
        evidence_relative_path: receipt.evidence_relative_path.clone(),
        descriptor: receipt.descriptor.clone(),
        evidence: receipt.evidence.clone(),
        partial_bundle: receipt.partial_bundle.clone(),
    };
    let derived = derive_linear_readout_acquisition(root, chain)?;
    if serde_json::to_vec(&derived.receipt)? != bytes {
        return Err(integrity("linear_readout_acquisition_replay_mismatch"));
    }
    Ok(derived)
}

/// Read-only deep replay. Imported PASS fields never authorize a readout.
pub fn authenticate_linear_readout(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<LinearReadoutAcquisitionReceipt> {
    let root = verify_internal_private_root(root)?;
    Ok(authenticate_linear_readout_details(&root, reference)?.receipt)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverWeightAxis {
    pub axis_id: SkillId,
    pub delta: DeltaArtifactRef,
    /// Authenticated provenance is mandatory for distributed LoRA axes. It is
    /// optional only so historical readout-control bases remain replayable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_import_receipt: Option<PrivateFileReference>,
}

pub const DISTRIBUTED_LORA_BASIS_INPUT_SCHEMA: &str =
    "cerebro.tidex.distributed_lora_basis_input/v1";
pub const DISTRIBUTED_LORA_BASIS_EVIDENCE_SCHEMA: &str =
    "cerebro.tidex.distributed_lora_basis_construction/v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistributedLoraAxisBinding {
    pub capability_id: CapabilityId,
    pub axis_id: SkillId,
    pub import_receipt: PrivateFileReference,
}

/// Typed assembly request for a distributed receiver basis. The assembly
/// authority replays every PEFT import receipt and requires one independently
/// trained calibration axis per declared capability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistributedLoraBasisInput {
    pub schema: String,
    pub axes: Vec<DistributedLoraAxisBinding>,
    pub construction_evidence: PrivateFileReference,
    pub total_model_parameter_count: u64,
    pub total_transformer_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DistributedLoraBasisConstructionEvidenceWire {
    schema: String,
    receiver_model_sha256: Sha256Digest,
    calibration_capability_ids: BTreeSet<CapabilityId>,
    confirmation_target_capability_ids_used: BTreeSet<CapabilityId>,
    axis_delta_sha256: Vec<Sha256Digest>,
    receiver_backend_frozen_before_target_observation: bool,
    target_receiver_solution_used: bool,
    target_receiver_execution_performed: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverAxisConstructionMethod {
    AnalyticReadout,
    CalibrationLora,
    CalibrationDenseUpdate,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverRealizationScope {
    #[default]
    ReadoutControl,
    DistributedTransformer,
}

/// Declares what part of the receiver can realize compiled coordinates. The
/// manifest is checked against the exact ParameterBlockLayout; it cannot turn
/// a narrow response probe into a distributed capability claim.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverRealizationManifest {
    pub schema: String,
    pub scope: ReceiverRealizationScope,
    pub axis_construction_method: ReceiverAxisConstructionMethod,
    pub total_model_parameter_count: u64,
    pub writable_parameter_count: u64,
    pub learned_parameter_count_per_axis: u64,
    pub target_tensor_families: BTreeSet<String>,
    pub covered_transformer_layers: BTreeSet<usize>,
    pub total_transformer_layers: usize,
    pub target_capability_used_in_basis: bool,
    pub target_receiver_execution_used_in_basis: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverRealizationAssessment {
    pub requested_scope: ReceiverRealizationScope,
    pub available_scope: ReceiverRealizationScope,
    pub writable_parameter_count: u64,
    pub writable_parameter_fraction: f64,
    pub target_tensor_family_count: usize,
    pub covered_transformer_layer_count: usize,
    pub total_transformer_layers: usize,
    pub full_layer_coverage: bool,
    pub attention_and_mlp_coverage: bool,
    pub calibration_only_basis: bool,
    pub distributed_compilation_permitted: bool,
    pub complete_capability_claim_permitted: bool,
}

/// Axis order is part of the coordinate system, not an interchangeable list.
/// The lineage names all capability labels used to construct the basis. These
/// are experimental declarations, not proof of semantic independence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverWeightBasis {
    pub schema: String,
    pub base_model_sha256: Sha256Digest,
    pub layout: ParameterBlockLayout,
    pub axes: Vec<ReceiverWeightAxis>,
    pub construction_capability_ids: BTreeSet<CapabilityId>,
    pub construction_evidence: PrivateFileReference,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realization: Option<ReceiverRealizationManifest>,
}

/// A closed measurement profile, distinct from the structural CapabilityIr.
/// Every value is a change from the exact base model's next-token logit margin
/// on a committed prompt. It is not a reference program or a hidden test.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverResponseMeasure {
    NextTokenLogitMarginChange,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverResponseProtocol {
    pub schema: String,
    pub measure: ReceiverResponseMeasure,
    pub base_model_sha256: Sha256Digest,
    pub model_config_sha256: Sha256Digest,
    pub tokenizer_sha256: Sha256Digest,
    pub collector_sha256: Sha256Digest,
    pub coordinate_ids: Vec<ProbeId>,
    pub prompt_sha256: Vec<Sha256Digest>,
    pub positive_token_id: u32,
    pub negative_token_id: u32,
    pub max_input_tokens: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CrossModelSignatureMeasure {
    DonorNextTokenLogitMarginProjection,
}

/// Cross-model protocol for a functional signature produced by a donor and
/// compiled into a different receiver.  The projection evidence is learned
/// only from calibration capabilities; a target may be projected through it,
/// but may not contribute to its construction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CrossModelFunctionalSignatureProtocol {
    pub schema: String,
    pub measure: CrossModelSignatureMeasure,
    pub receiver_base_model_sha256: Sha256Digest,
    pub donor_model_sha256: Sha256Digest,
    pub donor_model_config_sha256: Sha256Digest,
    pub donor_tokenizer_sha256: Sha256Digest,
    pub collector_sha256: Sha256Digest,
    pub raw_probe_sha256: Vec<Sha256Digest>,
    pub coordinate_ids: Vec<ProbeId>,
    pub projection_evidence: PrivateFileReference,
    pub max_input_tokens: usize,
}

/// The measured-response IR authenticates the derivation, not only its final
/// vector. It is not a claim of an executable structural CapabilityIr.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossModelProjectionEvidenceWire {
    schema: String,
    projection_arithmetic: String,
    donor_model_sha256: Sha256Digest,
    calibration_capability_ids: Vec<CapabilityId>,
    confirmation_target_capability_ids_used: Vec<CapabilityId>,
    target_values_used: bool,
    raw_probe_sha256: Vec<Sha256Digest>,
    raw_dimension: usize,
    projected_dimension: usize,
    mean: Vec<f64>,
    components: Vec<Vec<f64>>,
    retained_energy: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossModelFunctionalEvidenceWire {
    schema: String,
    capability_id: CapabilityId,
    donor_model_sha256: Sha256Digest,
    projection_evidence_sha256: Sha256Digest,
    raw_probe_sha256: Vec<Sha256Digest>,
    prompts: Vec<String>,
    prompt_sha256: Vec<Sha256Digest>,
    raw_logit_margins: Vec<f64>,
    projected_signature: Vec<f64>,
    projected_signature_sha256: Sha256Digest,
    receiver_data_used: bool,
    #[serde(default)]
    protocol_self_test: bool,
    #[serde(default)]
    source_calibration_capability_id: Option<CapabilityId>,
    #[serde(default)]
    receiver_backend_frozen_before_observation: Option<bool>,
    #[serde(default)]
    selected_receiver_pc_dimension: Option<usize>,
    #[serde(default)]
    selected_proposal_method: Option<ReceiverProposalMethod>,
}

#[derive(Debug, Clone, Deserialize)]
struct CrossModelReceiverSolutionEvidenceWire {
    schema: String,
    capability_id: CapabilityId,
    receiver_model_sha256: Sha256Digest,
    receiver_coordinates: Vec<f64>,
    receiver_coordinates_sha256: Sha256Digest,
    target_capability: bool,
    lora_used: bool,
    backpropagation_used: bool,
    direct_execution_verified: bool,
    #[serde(default)]
    receiver_backend_frozen_before_target_observation: Option<bool>,
    #[serde(default)]
    optimizer_steps: Option<usize>,
    #[serde(default)]
    trainable_parameter_count: Option<u64>,
    #[serde(default)]
    axis_delta_sha256: Option<Sha256Digest>,
    #[serde(flatten)]
    _extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReceiverBehavioralCalibrationEvidenceWire {
    schema: String,
    receiver_model_sha256: Sha256Digest,
    calibration_capability_ids: Vec<CapabilityId>,
    proposal_method: ReceiverProposalMethod,
    base_accuracies: Vec<f64>,
    compiled_loo_accuracies: Vec<f64>,
    confirmation_target_capability_ids_used: Vec<CapabilityId>,
    target_receiver_execution_performed: bool,
    #[serde(flatten)]
    _extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverBehavioralCalibrationPolicy {
    pub schema: String,
    pub minimum_mean_accuracy: f64,
    pub minimum_mean_gain: f64,
    pub minimum_non_degrading_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverBehavioralCalibrationSummary {
    pub calibration_case_count: usize,
    pub mean_base_accuracy: f64,
    pub mean_compiled_accuracy: f64,
    pub mean_gain: f64,
    pub non_degrading_count: usize,
    pub proposal_method: ReceiverProposalMethod,
}

#[derive(Debug, Clone)]
enum SignatureProtocol {
    Receiver(ReceiverResponseProtocol),
    CrossModel(CrossModelFunctionalSignatureProtocol),
}

impl SignatureProtocol {
    fn dimension(&self) -> usize {
        match self {
            Self::Receiver(value) => value.coordinate_ids.len(),
            Self::CrossModel(value) => value.coordinate_ids.len(),
        }
    }

    fn is_cross_model(&self) -> bool {
        matches!(self, Self::CrossModel(_))
    }

    fn cross_model(&self) -> Option<&CrossModelFunctionalSignatureProtocol> {
        match self {
            Self::CrossModel(value) => Some(value),
            Self::Receiver(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FunctionalResponseTarget {
    pub schema: String,
    pub capability_id: CapabilityId,
    pub protocol_sha256: Sha256Digest,
    pub values: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub functional_evidence: Option<PrivateFileReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverResponseObservation {
    pub schema: String,
    pub observation_id: ObservationId,
    pub capability_id: CapabilityId,
    pub basis_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
    pub receiver_coordinates: Vec<f64>,
    pub values: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub functional_evidence: Option<PrivateFileReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receiver_solution_evidence: Option<PrivateFileReference>,
}

/// Safety inputs live in K-dimensional receiver coordinates, never the full
/// checkpoint parameter space. The evidence reference is mandatory; its bytes
/// are authenticated but empirical validity still requires external replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverCoordinateSafety {
    pub protected_cortex: ProtectedCortex,
    pub risk_metric: Vec<Vec<f64>>,
    pub evidence: PrivateFileReference,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverWeightRequest {
    pub schema: String,
    pub basis: PrivateFileReference,
    pub protocol: PrivateFileReference,
    pub target: PrivateFileReference,
    pub observations: Vec<PrivateFileReference>,
    pub wrong_targets: Vec<PrivateFileReference>,
    pub safety: ReceiverCoordinateSafety,
    pub policy: ReceiverCompilerPolicy,
    #[serde(default)]
    pub required_realization_scope: ReceiverRealizationScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavioral_calibration_evidence: Option<PrivateFileReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavioral_calibration_policy: Option<ReceiverBehavioralCalibrationPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_method: Option<ReceiverProposalMethod>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_ir_readout: Option<PrivateFileReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverWeightBlocker {
    NumericalPredictionGatesFailed,
    OutsideCalibratedCoordinateRadius,
    OutsideCalibratedFunctionalSupport,
    OutsideCalibratedRelationalSupport,
    InsufficientRealizationCoverage,
    CapabilityIrEvidenceMissing,
    ProtocolSelfTestOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverWeightCandidate {
    pub schema: String,
    pub request: PrivateFileReference,
    pub compiler_source_sha256: Sha256Digest,
    pub basis_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
    pub target_capability_id: CapabilityId,
    pub calibration_observation_count: usize,
    pub calibration_capability_count: usize,
    pub numerical: ReceiverSignatureCompilation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_ir_readout: Option<ReceiverCapabilityIrReadoutSummary>,
    pub maximum_calibration_coordinate_norm: f64,
    pub proposed_coordinate_norm: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functional_support_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_calibration_loo_functional_support_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub behavioral_calibration: Option<ReceiverBehavioralCalibrationSummary>,
    pub realization: ReceiverRealizationAssessment,
    pub blockers: Vec<ReceiverWeightBlocker>,
    pub dense_delta: Option<DeltaArtifactRef>,
    pub model_execution_verified: bool,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverWeightCheckpointReceipt {
    pub schema: String,
    pub candidate: PrivateFileReference,
    pub output_path: PathBuf,
    pub materialization: WeightMaterializationReceipt,
    pub model_execution_verified: bool,
    pub authorizes_promotion: bool,
}

fn validate_receiver_protocol(
    protocol: &ReceiverResponseProtocol,
    basis: &ReceiverWeightBasis,
) -> BrainResult<()> {
    let n = protocol.coordinate_ids.len();
    if protocol.schema != PROTOCOL_SCHEMA
        || protocol.base_model_sha256 != basis.base_model_sha256
        || n == 0
        || n > MAX_RESPONSE_DIMENSION
        || protocol.prompt_sha256.len() != n
        || protocol
            .coordinate_ids
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != n
        || protocol.prompt_sha256.iter().collect::<BTreeSet<_>>().len() != n
        || protocol.positive_token_id == protocol.negative_token_id
        || protocol.max_input_tokens == 0
        || protocol.max_input_tokens > 32768
    {
        return Err(invalid("receiver_weight_response_protocol_invalid"));
    }
    Ok(())
}

fn validate_cross_model_protocol(
    root: &Path,
    protocol: &CrossModelFunctionalSignatureProtocol,
    basis: &ReceiverWeightBasis,
) -> BrainResult<()> {
    let n = protocol.coordinate_ids.len();
    if protocol.schema != CROSS_MODEL_PROTOCOL_SCHEMA
        || protocol.receiver_base_model_sha256 != basis.base_model_sha256
        || protocol.donor_model_sha256 == basis.base_model_sha256
        || n == 0
        || n > MAX_RESPONSE_DIMENSION
        || protocol.raw_probe_sha256.is_empty()
        || protocol.raw_probe_sha256.len() > MAX_RESPONSE_DIMENSION
        || protocol
            .coordinate_ids
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != n
        || protocol
            .raw_probe_sha256
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != protocol.raw_probe_sha256.len()
        || protocol.max_input_tokens == 0
        || protocol.max_input_tokens > 32768
    {
        return Err(invalid("receiver_weight_cross_model_protocol_invalid"));
    }
    let projection = read_cross_model_projection(root, protocol)?;
    let construction_ids = projection
        .calibration_capability_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if construction_ids != basis.construction_capability_ids {
        return Err(invalid("receiver_weight_projection_basis_lineage_mismatch"));
    }
    Ok(())
}

fn read_cross_model_projection(
    root: &Path,
    protocol: &CrossModelFunctionalSignatureProtocol,
) -> BrainResult<CrossModelProjectionEvidenceWire> {
    let projection: CrossModelProjectionEvidenceWire =
        read_record(root, &protocol.projection_evidence)?;
    let ids = projection
        .calibration_capability_ids
        .iter()
        .collect::<BTreeSet<_>>();
    if projection.schema != CROSS_MODEL_PROJECTION_SCHEMA
        || projection.projection_arithmetic != PROJECTION_ARITHMETIC
        || projection.donor_model_sha256 != protocol.donor_model_sha256
        || projection.projected_dimension != protocol.coordinate_ids.len()
        || projection.raw_dimension != protocol.raw_probe_sha256.len()
        || projection.raw_probe_sha256 != protocol.raw_probe_sha256
        || projection.mean.len() != projection.raw_dimension
        || projection.components.len() != projection.projected_dimension
        || projection.raw_dimension == 0
        || projection.projected_dimension == 0
        || projection.raw_dimension > MAX_RESPONSE_DIMENSION
        || projection.projected_dimension > MAX_RESPONSE_DIMENSION
        || projection.target_values_used
        || !projection
            .confirmation_target_capability_ids_used
            .is_empty()
        || ids.is_empty()
        || ids.len() != projection.calibration_capability_ids.len()
        || ids.len() > MAX_ANCHORS
        || !projection.retained_energy.is_finite()
        || !(0.0..=1.0).contains(&projection.retained_energy)
    {
        return Err(invalid(
            "receiver_weight_cross_model_projection_evidence_mismatch",
        ));
    }
    // Validating at the center admits its legitimate zero projected vector;
    // neither a centered sample nor redundant measured axes imply fraud.
    project_functional_signature(&projection.mean, &projection.mean, &projection.components)?;
    Ok(projection)
}

fn read_signature_protocol(
    root: &Path,
    reference: &PrivateFileReference,
    basis: &ReceiverWeightBasis,
) -> BrainResult<SignatureProtocol> {
    let bytes = reference.read_verified_bounded(root, MAX_RECORD_BYTES)?;
    let wire: serde_json::Value = serde_json::from_slice(&bytes)?;
    let schema = wire
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid("receiver_weight_protocol_schema_missing"))?;
    match schema {
        PROTOCOL_SCHEMA => {
            let protocol: ReceiverResponseProtocol = serde_json::from_slice(&bytes)?;
            validate_receiver_protocol(&protocol, basis)?;
            Ok(SignatureProtocol::Receiver(protocol))
        }
        CROSS_MODEL_PROTOCOL_SCHEMA => {
            let protocol: CrossModelFunctionalSignatureProtocol = serde_json::from_slice(&bytes)?;
            validate_cross_model_protocol(root, &protocol, basis)?;
            Ok(SignatureProtocol::CrossModel(protocol))
        }
        _ => Err(invalid("receiver_weight_protocol_schema_unknown")),
    }
}

fn exact_f64_vector_digest(domain: &[u8], values: &[f64]) -> Sha256Digest {
    let mut payload = Vec::with_capacity(8 + values.len() * 8);
    payload.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        payload.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Sha256Digest::digest_domain(domain, &payload)
}

fn functional_signature_digest(values: &[f64]) -> Sha256Digest {
    exact_f64_vector_digest(b"cerebro.tidex.functional_signature_f64_le/v1\0", values)
}

fn receiver_coordinates_digest(values: &[f64]) -> Sha256Digest {
    exact_f64_vector_digest(b"cerebro.tidex.receiver_coordinates_f64_le/v1\0", values)
}

fn validate_cross_model_functional_evidence(
    root: &Path,
    reference: &PrivateFileReference,
    protocol: &CrossModelFunctionalSignatureProtocol,
    capability_id: &CapabilityId,
    values: &[f64],
) -> BrainResult<bool> {
    let evidence: CrossModelFunctionalEvidenceWire = read_record(root, reference)?;
    if evidence.schema != CROSS_MODEL_FUNCTIONAL_EVIDENCE_SCHEMA
        || evidence.capability_id != *capability_id
        || evidence.donor_model_sha256 != protocol.donor_model_sha256
        || evidence.projection_evidence_sha256 != protocol.projection_evidence.sha256
        || evidence.receiver_data_used
        || evidence.receiver_backend_frozen_before_observation == Some(false)
        || evidence.selected_receiver_pc_dimension == Some(0)
    {
        return Err(invalid(
            "receiver_weight_cross_model_functional_evidence_mismatch",
        ));
    }
    let projection = read_cross_model_projection(root, protocol)?;
    let n = projection.raw_dimension;
    if evidence.raw_probe_sha256 != protocol.raw_probe_sha256
        || evidence.raw_logit_margins.len() != n
        || evidence.prompts.len() != n
        || evidence.prompt_sha256.len() != n
        || evidence.prompt_sha256.iter().collect::<BTreeSet<_>>().len() != n
        || evidence
            .prompts
            .iter()
            .zip(&evidence.prompt_sha256)
            .any(|(prompt, digest)| {
                prompt.is_empty()
                    || prompt.len() > 1024 * 1024
                    || Sha256Digest::digest_bytes(prompt.as_bytes()) != *digest
            })
    {
        return Err(invalid(
            "receiver_weight_cross_model_raw_response_binding_mismatch",
        ));
    }
    if evidence.protocol_self_test != evidence.source_calibration_capability_id.is_some()
        || evidence
            .source_calibration_capability_id
            .as_ref()
            .is_some_and(|id| !projection.calibration_capability_ids.contains(id))
    {
        return Err(invalid(
            "receiver_weight_protocol_self_test_lineage_invalid",
        ));
    }
    // The optional method is a typed diagnostic, not a selector or authority.
    let _selected_method = evidence.selected_proposal_method;
    let reconstructed = project_functional_signature(
        &evidence.raw_logit_margins,
        &projection.mean,
        &projection.components,
    )?;
    if evidence.projected_signature.len() != values.len()
        || evidence
            .projected_signature
            .iter()
            .any(|value| !value.is_finite())
        || functional_signature_digest(&evidence.projected_signature)
            != evidence.projected_signature_sha256
        || functional_signature_digest(values) != evidence.projected_signature_sha256
    {
        return Err(invalid(
            "receiver_weight_cross_model_functional_values_mismatch",
        ));
    }
    if functional_signature_digest(&reconstructed) != evidence.projected_signature_sha256 {
        return Err(invalid(
            "receiver_weight_cross_model_functional_derivation_mismatch",
        ));
    }
    Ok(evidence.protocol_self_test)
}

fn validate_cross_model_receiver_solution_evidence(
    root: &Path,
    reference: &PrivateFileReference,
    basis: &ReceiverWeightBasis,
    capability_id: &CapabilityId,
    coordinates: &[f64],
) -> BrainResult<()> {
    let bytes = reference.read_verified_bounded(root, MAX_RECORD_BYTES)?;
    let evidence: CrossModelReceiverSolutionEvidenceWire = serde_json::from_slice(&bytes)?;
    if evidence.schema != CROSS_MODEL_RECEIVER_SOLUTION_EVIDENCE_SCHEMA
        || evidence.capability_id != *capability_id
        || evidence.receiver_model_sha256 != basis.base_model_sha256
        || evidence.target_capability
        || !evidence.direct_execution_verified
        || evidence.lora_used != evidence.backpropagation_used
    {
        return Err(invalid(
            "receiver_weight_cross_model_solution_evidence_mismatch",
        ));
    }
    if evidence.receiver_coordinates.len() != coordinates.len()
        || evidence.receiver_coordinates.len() != basis.axes.len()
        || evidence
            .receiver_coordinates
            .iter()
            .any(|value| !value.is_finite())
        || receiver_coordinates_digest(&evidence.receiver_coordinates)
            != evidence.receiver_coordinates_sha256
        || receiver_coordinates_digest(coordinates) != evidence.receiver_coordinates_sha256
    {
        return Err(invalid(
            "receiver_weight_cross_model_solution_coordinates_mismatch",
        ));
    }
    if evidence.lora_used {
        if basis.realization.as_ref().is_none_or(|manifest| {
            manifest.axis_construction_method != ReceiverAxisConstructionMethod::CalibrationLora
        }) || evidence.receiver_backend_frozen_before_target_observation != Some(true)
            || evidence.optimizer_steps.is_none_or(|steps| steps == 0)
            || evidence
                .trainable_parameter_count
                .is_none_or(|count| count == 0)
        {
            return Err(invalid("receiver_weight_calibration_lora_lineage_invalid"));
        }
        let delta_sha = evidence
            .axis_delta_sha256
            .as_ref()
            .ok_or_else(|| invalid("receiver_weight_calibration_lora_axis_missing"))?;
        let axis_index = basis
            .axes
            .iter()
            .position(|axis| &axis.delta.sha256 == delta_sha)
            .ok_or_else(|| invalid("receiver_weight_calibration_lora_axis_unbound"))?;
        if coordinates
            .iter()
            .enumerate()
            .any(|(index, value)| *value != if index == axis_index { 1.0 } else { 0.0 })
        {
            return Err(invalid(
                "receiver_weight_calibration_lora_coordinates_not_one_hot",
            ));
        }
    } else if evidence
        .receiver_backend_frozen_before_target_observation
        .is_some()
        || evidence.optimizer_steps.is_some()
        || evidence.trainable_parameter_count.is_some()
        || evidence.axis_delta_sha256.is_some()
    {
        return Err(invalid(
            "receiver_weight_analytic_solution_training_fields_present",
        ));
    }
    Ok(())
}

fn validate_behavioral_calibration_evidence(
    root: &Path,
    reference: &PrivateFileReference,
    policy: &ReceiverBehavioralCalibrationPolicy,
    basis: &ReceiverWeightBasis,
    proposal_method: ReceiverProposalMethod,
) -> BrainResult<ReceiverBehavioralCalibrationSummary> {
    if policy.schema != "cerebro.tidex.receiver_behavioral_calibration_policy/v1"
        || !policy.minimum_mean_accuracy.is_finite()
        || !(0.0..=1.0).contains(&policy.minimum_mean_accuracy)
        || !policy.minimum_mean_gain.is_finite()
        || !(0.0..=1.0).contains(&policy.minimum_mean_gain)
        || policy.minimum_non_degrading_count == 0
        || policy.minimum_non_degrading_count > basis.construction_capability_ids.len()
    {
        return Err(invalid(
            "receiver_weight_behavioral_calibration_policy_invalid",
        ));
    }
    let bytes = reference.read_verified_bounded(root, MAX_RECORD_BYTES)?;
    let evidence: ReceiverBehavioralCalibrationEvidenceWire = serde_json::from_slice(&bytes)?;
    if evidence.schema != RECEIVER_BEHAVIORAL_CALIBRATION_EVIDENCE_SCHEMA
        || evidence.receiver_model_sha256 != basis.base_model_sha256
        || evidence.proposal_method != proposal_method
        || evidence.target_receiver_execution_performed
        || !evidence.confirmation_target_capability_ids_used.is_empty()
        || evidence.calibration_capability_ids.len() != basis.construction_capability_ids.len()
        || evidence.base_accuracies.len() != evidence.calibration_capability_ids.len()
        || evidence.compiled_loo_accuracies.len() != evidence.calibration_capability_ids.len()
    {
        return Err(invalid(
            "receiver_weight_behavioral_calibration_evidence_mismatch",
        ));
    }
    let ids = evidence
        .calibration_capability_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if ids.len() != evidence.calibration_capability_ids.len()
        || ids != basis.construction_capability_ids
        || evidence
            .base_accuracies
            .iter()
            .chain(&evidence.compiled_loo_accuracies)
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(invalid(
            "receiver_weight_behavioral_calibration_evidence_mismatch",
        ));
    }
    let count = evidence.base_accuracies.len();
    if count == 0 {
        return Err(invalid(
            "receiver_weight_behavioral_calibration_evidence_empty",
        ));
    }
    let mean_base_accuracy = evidence.base_accuracies.iter().sum::<f64>() / count as f64;
    let mean_compiled_accuracy =
        evidence.compiled_loo_accuracies.iter().sum::<f64>() / count as f64;
    let mean_gain = mean_compiled_accuracy - mean_base_accuracy;
    let non_degrading_count = evidence
        .base_accuracies
        .iter()
        .zip(&evidence.compiled_loo_accuracies)
        .filter(|(base, compiled)| compiled >= base)
        .count();
    if mean_compiled_accuracy < policy.minimum_mean_accuracy
        || mean_gain < policy.minimum_mean_gain
        || non_degrading_count < policy.minimum_non_degrading_count
    {
        return Err(invalid(
            "receiver_weight_behavioral_calibration_gates_failed",
        ));
    }
    Ok(ReceiverBehavioralCalibrationSummary {
        calibration_case_count: count,
        mean_base_accuracy,
        mean_compiled_accuracy,
        mean_gain,
        non_degrading_count,
        proposal_method,
    })
}

fn validate_values(values: &[f64], dimension: usize) -> BrainResult<()> {
    if values.len() != dimension || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("receiver_weight_response_shape_or_nonfinite"));
    }
    Ok(())
}

fn validate_target(
    target: &FunctionalResponseTarget,
    protocol: &PrivateFileReference,
    dimension: usize,
) -> BrainResult<()> {
    if target.schema != TARGET_SCHEMA || target.protocol_sha256 != protocol.sha256 {
        return Err(invalid("receiver_weight_target_protocol_mismatch"));
    }
    validate_values(&target.values, dimension)?;
    if norm(&target.values)? <= 1e-15 {
        return Err(invalid("receiver_weight_target_degenerate"));
    }
    Ok(())
}

fn layout_realization_coverage(
    layout: &ParameterBlockLayout,
) -> (BTreeSet<String>, BTreeSet<usize>) {
    let mut families = BTreeSet::new();
    let mut layers = BTreeSet::new();
    for block in &layout.blocks {
        if let Some(stem) = block.name.strip_suffix(".weight") {
            if let Some(family) = stem.rsplit('.').next() {
                families.insert(family.to_string());
            }
        }
        let parts = block.name.split('.').collect::<Vec<_>>();
        if let Some(layer) = parts
            .windows(2)
            .find(|window| window[0] == "layers")
            .and_then(|window| window[1].parse::<usize>().ok())
        {
            layers.insert(layer);
        }
    }
    (families, layers)
}

fn assess_realization(
    basis: &ReceiverWeightBasis,
    requested_scope: ReceiverRealizationScope,
) -> BrainResult<ReceiverRealizationAssessment> {
    let Some(manifest) = basis.realization.as_ref() else {
        if basis.schema != BASIS_SCHEMA_V1 {
            return Err(invalid("receiver_weight_realization_manifest_missing"));
        }
        return Ok(ReceiverRealizationAssessment {
            requested_scope,
            available_scope: ReceiverRealizationScope::ReadoutControl,
            writable_parameter_count: basis.layout.total_parameter_count,
            writable_parameter_fraction: 0.0,
            target_tensor_family_count: 1,
            covered_transformer_layer_count: 0,
            total_transformer_layers: 0,
            full_layer_coverage: false,
            attention_and_mlp_coverage: false,
            calibration_only_basis: true,
            distributed_compilation_permitted: false,
            complete_capability_claim_permitted: false,
        });
    };
    if basis.schema != BASIS_SCHEMA
        || manifest.schema != REALIZATION_SCHEMA
        || manifest.total_model_parameter_count == 0
        || manifest.writable_parameter_count != basis.layout.total_parameter_count
        || manifest.writable_parameter_count > manifest.total_model_parameter_count
        || manifest.target_capability_used_in_basis
        || manifest.target_receiver_execution_used_in_basis
    {
        return Err(invalid("receiver_weight_realization_manifest_invalid"));
    }
    let (layout_families, layout_layers) = layout_realization_coverage(&basis.layout);
    if layout_families != manifest.target_tensor_families
        || layout_layers != manifest.covered_transformer_layers
    {
        return Err(invalid("receiver_weight_realization_layout_mismatch"));
    }
    let required_attention = ["q_proj", "k_proj", "v_proj", "o_proj"];
    let required_mlp = ["gate_proj", "up_proj", "down_proj"];
    let attention_and_mlp_coverage = required_attention
        .iter()
        .chain(required_mlp.iter())
        .all(|family| manifest.target_tensor_families.contains(*family));
    let full_layer_coverage = manifest.total_transformer_layers > 0
        && manifest.covered_transformer_layers
            == (0..manifest.total_transformer_layers).collect::<BTreeSet<_>>();
    let distributed = manifest.scope == ReceiverRealizationScope::DistributedTransformer
        && matches!(
            manifest.axis_construction_method,
            ReceiverAxisConstructionMethod::CalibrationLora
                | ReceiverAxisConstructionMethod::CalibrationDenseUpdate
        )
        && manifest.learned_parameter_count_per_axis > 0
        && attention_and_mlp_coverage
        && full_layer_coverage;
    if manifest.scope == ReceiverRealizationScope::DistributedTransformer && !distributed {
        return Err(invalid(
            "receiver_weight_distributed_realization_incomplete",
        ));
    }
    if manifest.scope == ReceiverRealizationScope::ReadoutControl
        && manifest.axis_construction_method != ReceiverAxisConstructionMethod::AnalyticReadout
    {
        return Err(invalid(
            "receiver_weight_readout_realization_method_invalid",
        ));
    }
    Ok(ReceiverRealizationAssessment {
        requested_scope,
        available_scope: manifest.scope,
        writable_parameter_count: manifest.writable_parameter_count,
        writable_parameter_fraction: manifest.writable_parameter_count as f64
            / manifest.total_model_parameter_count as f64,
        target_tensor_family_count: manifest.target_tensor_families.len(),
        covered_transformer_layer_count: manifest.covered_transformer_layers.len(),
        total_transformer_layers: manifest.total_transformer_layers,
        full_layer_coverage,
        attention_and_mlp_coverage,
        calibration_only_basis: true,
        distributed_compilation_permitted: distributed,
        complete_capability_claim_permitted: false,
    })
}

/// Assemble a v2 basis only after replaying every imported LoRA axis and its
/// calibration-only construction evidence. The returned basis is still a
/// candidate substrate; it cannot authorize a capability claim or promotion.
pub fn assemble_distributed_lora_basis(
    root: &Path,
    input: &DistributedLoraBasisInput,
) -> BrainResult<ReceiverWeightBasis> {
    let root = verify_internal_private_root(root)?;
    if input.schema != DISTRIBUTED_LORA_BASIS_INPUT_SCHEMA
        || input.axes.len() < 5
        || input.axes.len() > MAX_AXES
        || input.total_model_parameter_count == 0
        || input.total_transformer_layers == 0
    {
        return Err(invalid("receiver_weight_distributed_basis_input_invalid"));
    }
    let evidence: DistributedLoraBasisConstructionEvidenceWire =
        read_record(&root, &input.construction_evidence)?;
    if evidence.schema != DISTRIBUTED_LORA_BASIS_EVIDENCE_SCHEMA
        || !evidence.confirmation_target_capability_ids_used.is_empty()
        || !evidence.receiver_backend_frozen_before_target_observation
        || evidence.target_receiver_solution_used
        || evidence.target_receiver_execution_performed
        || evidence.axis_delta_sha256.len() != input.axes.len()
    {
        return Err(invalid(
            "receiver_weight_distributed_basis_evidence_invalid",
        ));
    }

    let mut capability_ids = BTreeSet::new();
    let mut axis_ids = BTreeSet::new();
    let mut import_refs = BTreeSet::new();
    let mut axes = Vec::with_capacity(input.axes.len());
    let mut common: Option<LoraAdapterAxisReceipt> = None;
    for (index, binding) in input.axes.iter().enumerate() {
        if !capability_ids.insert(binding.capability_id.clone())
            || !axis_ids.insert(binding.axis_id.clone())
            || !import_refs.insert(binding.import_receipt.sha256.clone())
        {
            return Err(invalid("receiver_weight_distributed_basis_axis_duplicate"));
        }
        let receipt = authenticate_lora_adapter_axis_receipt(&root, &binding.import_receipt)?;
        if evidence.axis_delta_sha256[index] != receipt.dense_delta.sha256 {
            return Err(integrity(
                "receiver_weight_distributed_basis_axis_evidence_mismatch",
            ));
        }
        if let Some(first) = &common {
            if receipt.base_model_sha256 != first.base_model_sha256
                || receipt.parameter_layout != first.parameter_layout
                || receipt.learned_parameter_count != first.learned_parameter_count
                || receipt.target_families != first.target_families
                || receipt.covered_transformer_layers != first.covered_transformer_layers
            {
                return Err(invalid(
                    "receiver_weight_distributed_basis_axis_contract_mismatch",
                ));
            }
        } else {
            common = Some(receipt.clone());
        }
        axes.push(ReceiverWeightAxis {
            axis_id: binding.axis_id.clone(),
            delta: receipt.dense_delta,
            source_import_receipt: Some(binding.import_receipt.clone()),
        });
    }
    let first = common.ok_or_else(|| invalid("receiver_weight_distributed_basis_empty"))?;
    let inventory = inspect_model_safetensors(&first.base_model_path)?;
    if inventory.model_sha256 != first.base_model_sha256
        || inventory.total_parameter_count != input.total_model_parameter_count
        || evidence.receiver_model_sha256 != first.base_model_sha256
        || evidence.calibration_capability_ids != capability_ids
        || first.covered_transformer_layers
            != (0..input.total_transformer_layers).collect::<BTreeSet<_>>()
    {
        return Err(invalid(
            "receiver_weight_distributed_basis_lineage_mismatch",
        ));
    }
    let basis = ReceiverWeightBasis {
        schema: BASIS_SCHEMA.to_string(),
        base_model_sha256: first.base_model_sha256,
        layout: first.parameter_layout.clone(),
        axes,
        construction_capability_ids: capability_ids,
        construction_evidence: input.construction_evidence.clone(),
        realization: Some(ReceiverRealizationManifest {
            schema: REALIZATION_SCHEMA.to_string(),
            scope: ReceiverRealizationScope::DistributedTransformer,
            axis_construction_method: ReceiverAxisConstructionMethod::CalibrationLora,
            total_model_parameter_count: input.total_model_parameter_count,
            writable_parameter_count: first.parameter_layout.total_parameter_count,
            learned_parameter_count_per_axis: first.learned_parameter_count,
            target_tensor_families: first.target_families,
            covered_transformer_layers: first.covered_transformer_layers,
            total_transformer_layers: input.total_transformer_layers,
            target_capability_used_in_basis: false,
            target_receiver_execution_used_in_basis: false,
        }),
    };
    validate_basis(&root, &basis)?;
    Ok(basis)
}

fn validate_basis(root: &Path, basis: &ReceiverWeightBasis) -> BrainResult<()> {
    if !matches!(basis.schema.as_str(), BASIS_SCHEMA | BASIS_SCHEMA_V1)
        || basis.axes.is_empty()
        || basis.axes.len() > MAX_AXES
        || (basis.schema == BASIS_SCHEMA && basis.realization.is_none())
        || (basis.schema == BASIS_SCHEMA_V1 && basis.realization.is_some())
    {
        return Err(invalid("receiver_weight_basis_invalid"));
    }
    basis.layout.validate()?;
    basis
        .construction_evidence
        .read_verified_bounded(root, MAX_RECORD_BYTES)?;
    let mut ids = BTreeSet::new();
    let mut deltas = BTreeSet::new();
    for axis in &basis.axes {
        if !ids.insert(&axis.axis_id)
            || !deltas.insert(&axis.delta.sha256)
            || axis.delta.parameter_count != basis.layout.total_parameter_count
        {
            return Err(invalid("receiver_weight_basis_axis_invalid"));
        }
        verify_dvec_reference_under_root(root, &axis.delta)?;
    }
    if basis.realization.as_ref().is_some_and(|manifest| {
        manifest.axis_construction_method == ReceiverAxisConstructionMethod::CalibrationLora
    }) {
        let evidence: DistributedLoraBasisConstructionEvidenceWire =
            read_record(root, &basis.construction_evidence)?;
        let ordered_deltas = basis
            .axes
            .iter()
            .map(|axis| axis.delta.sha256.clone())
            .collect::<Vec<_>>();
        if evidence.schema != DISTRIBUTED_LORA_BASIS_EVIDENCE_SCHEMA
            || evidence.receiver_model_sha256 != basis.base_model_sha256
            || evidence.calibration_capability_ids != basis.construction_capability_ids
            || !evidence.confirmation_target_capability_ids_used.is_empty()
            || evidence.axis_delta_sha256 != ordered_deltas
            || !evidence.receiver_backend_frozen_before_target_observation
            || evidence.target_receiver_solution_used
            || evidence.target_receiver_execution_performed
        {
            return Err(invalid(
                "receiver_weight_distributed_basis_evidence_invalid",
            ));
        }
        for axis in &basis.axes {
            let reference = axis.source_import_receipt.as_ref().ok_or_else(|| {
                invalid("receiver_weight_distributed_basis_import_receipt_missing")
            })?;
            let receipt = authenticate_lora_adapter_axis_receipt(root, reference)?;
            if receipt.base_model_sha256 != basis.base_model_sha256
                || receipt.parameter_layout != basis.layout
                || receipt.dense_delta != axis.delta
            {
                return Err(integrity(
                    "receiver_weight_distributed_basis_import_receipt_mismatch",
                ));
            }
        }
    } else if basis
        .axes
        .iter()
        .any(|axis| axis.source_import_receipt.is_some())
    {
        return Err(invalid(
            "receiver_weight_readout_basis_import_receipt_unexpected",
        ));
    }
    assess_realization(basis, ReceiverRealizationScope::ReadoutControl)?;
    Ok(())
}

type WeightedDeltas = Vec<(DeltaArtifactRef, f64)>;

/// Read-only derivation. It never creates an absent candidate or dense delta
/// while authenticating a receipt. Candidate predictions and coefficients are
/// recomputed from their exact inputs rather than trusted from stored flags.
fn derive_candidate(
    root: &Path,
    request_reference: &PrivateFileReference,
) -> BrainResult<(ReceiverWeightCandidate, WeightedDeltas)> {
    let request: ReceiverWeightRequest = read_record(root, request_reference)?;
    if request.schema != REQUEST_SCHEMA
        || request.observations.len() < 5
        || request.observations.len() > MAX_ANCHORS
        || request.wrong_targets.is_empty()
        || request.wrong_targets.len() > MAX_ANCHORS
    {
        return Err(invalid("receiver_weight_request_invalid"));
    }
    let basis: ReceiverWeightBasis = read_record(root, &request.basis)?;
    validate_basis(root, &basis)?;
    let realization = assess_realization(&basis, request.required_realization_scope)?;
    let protocol = read_signature_protocol(root, &request.protocol, &basis)?;
    let dimension = protocol.dimension();
    let k = basis.axes.len();
    let selected_proposal_method = request
        .proposal_method
        .unwrap_or(ReceiverProposalMethod::FitProtectedCoordinates);
    if protocol.is_cross_model() && request.proposal_method.is_none() {
        return Err(invalid(
            "receiver_weight_cross_model_proposal_method_required",
        ));
    }
    let behavioral_calibration = if protocol.is_cross_model() {
        let evidence = request
            .behavioral_calibration_evidence
            .as_ref()
            .ok_or_else(|| {
                invalid("receiver_weight_cross_model_behavioral_calibration_evidence_missing")
            })?;
        let policy = request
            .behavioral_calibration_policy
            .as_ref()
            .ok_or_else(|| {
                invalid("receiver_weight_cross_model_behavioral_calibration_policy_missing")
            })?;
        Some(validate_behavioral_calibration_evidence(
            root,
            evidence,
            policy,
            &basis,
            selected_proposal_method,
        )?)
    } else {
        if request.behavioral_calibration_evidence.is_some()
            || request.behavioral_calibration_policy.is_some()
        {
            return Err(invalid(
                "receiver_weight_behavioral_calibration_only_valid_cross_model",
            ));
        }
        None
    };
    let target: FunctionalResponseTarget = read_record(root, &request.target)?;
    validate_target(&target, &request.protocol, dimension)?;
    if basis
        .construction_capability_ids
        .contains(&target.capability_id)
    {
        return Err(invalid("receiver_weight_target_leaked_into_basis_lineage"));
    }
    let mut protocol_self_test = false;
    let target_functional_evidence_sha = if let Some(cross_model) = protocol.cross_model() {
        let evidence = target
            .functional_evidence
            .as_ref()
            .ok_or_else(|| invalid("receiver_weight_cross_model_target_evidence_missing"))?;
        protocol_self_test = validate_cross_model_functional_evidence(
            root,
            evidence,
            cross_model,
            &target.capability_id,
            &target.values,
        )?;
        Some(evidence.sha256.clone())
    } else {
        None
    };
    if request.safety.risk_metric.len() != k
        || request.safety.risk_metric.iter().any(|row| row.len() != k)
        || request.safety.protected_cortex.parameter_importance.len() != k
    {
        return Err(invalid("receiver_weight_coordinate_safety_shape"));
    }
    request
        .safety
        .evidence
        .read_verified_bounded(root, MAX_RECORD_BYTES)?;
    let risk_metric = Matrix::from_rows(&request.safety.risk_metric)?;

    let mut observation_ids = BTreeSet::new();
    let mut observation_hashes = BTreeSet::new();
    let mut coordinate_bits = BTreeSet::new();
    let mut capability_ids = BTreeSet::new();
    let mut functional_evidence_hashes = BTreeSet::new();
    let mut receiver_solution_evidence_hashes = BTreeSet::new();
    let mut functional_signatures = Vec::new();
    let mut receiver_solutions = Vec::new();
    let mut maximum_calibration_coordinate_norm = 0.0_f64;
    for reference in &request.observations {
        let observation: ReceiverResponseObservation = read_record(root, reference)?;
        if observation.schema != OBSERVATION_SCHEMA
            || observation.basis_sha256 != request.basis.sha256
            || observation.protocol_sha256 != request.protocol.sha256
            || !observation_ids.insert(observation.observation_id.clone())
            || !observation_hashes.insert(reference.sha256.clone())
        {
            return Err(invalid("receiver_weight_observation_binding_invalid"));
        }
        if observation.capability_id == target.capability_id {
            return Err(invalid("receiver_weight_target_leaked_into_calibration"));
        }
        validate_values(&observation.values, dimension)?;
        validate_values(&observation.receiver_coordinates, k)?;
        if let Some(cross_model) = protocol.cross_model() {
            let functional_evidence =
                observation.functional_evidence.as_ref().ok_or_else(|| {
                    invalid("receiver_weight_cross_model_observation_functional_evidence_missing")
                })?;
            let receiver_solution_evidence = observation
                .receiver_solution_evidence
                .as_ref()
                .ok_or_else(|| {
                    invalid("receiver_weight_cross_model_observation_solution_evidence_missing")
                })?;
            if validate_cross_model_functional_evidence(
                root,
                functional_evidence,
                cross_model,
                &observation.capability_id,
                &observation.values,
            )? {
                return Err(invalid("receiver_weight_self_test_in_calibration"));
            }
            validate_cross_model_receiver_solution_evidence(
                root,
                receiver_solution_evidence,
                &basis,
                &observation.capability_id,
                &observation.receiver_coordinates,
            )?;
            if target_functional_evidence_sha
                .as_ref()
                .is_some_and(|target_sha| target_sha == &functional_evidence.sha256)
                || !functional_evidence_hashes.insert(functional_evidence.sha256.clone())
                || !receiver_solution_evidence_hashes
                    .insert(receiver_solution_evidence.sha256.clone())
            {
                return Err(invalid(
                    "receiver_weight_cross_model_calibration_evidence_reused",
                ));
            }
        }
        // Relabeling identical interventions is not new calibration evidence.
        let bits = observation
            .receiver_coordinates
            .iter()
            .map(|value| if *value == 0.0 { 0 } else { value.to_bits() })
            .collect::<Vec<_>>();
        if !coordinate_bits.insert(bits) {
            return Err(invalid("receiver_weight_duplicate_calibration_coordinates"));
        }
        maximum_calibration_coordinate_norm =
            maximum_calibration_coordinate_norm.max(norm(&observation.receiver_coordinates)?);
        capability_ids.insert(observation.capability_id);
        functional_signatures.push(observation.values);
        receiver_solutions.push(observation.receiver_coordinates);
    }
    if capability_ids != basis.construction_capability_ids {
        return Err(invalid(
            "receiver_weight_calibration_capabilities_do_not_match_basis_lineage",
        ));
    }
    let mut wrong_functional_signatures = Vec::new();
    let mut wrong_ids = BTreeSet::new();
    for reference in &request.wrong_targets {
        let wrong: FunctionalResponseTarget = read_record(root, reference)?;
        validate_target(&wrong, &request.protocol, dimension)?;
        if wrong.capability_id == target.capability_id
            || !wrong_ids.insert(wrong.capability_id.clone())
        {
            return Err(invalid("receiver_weight_wrong_target_identity_invalid"));
        }
        if let Some(cross_model) = protocol.cross_model() {
            let evidence = wrong.functional_evidence.as_ref().ok_or_else(|| {
                invalid("receiver_weight_cross_model_wrong_target_evidence_missing")
            })?;
            protocol_self_test |= validate_cross_model_functional_evidence(
                root,
                evidence,
                cross_model,
                &wrong.capability_id,
                &wrong.values,
            )?;
            if target_functional_evidence_sha
                .as_ref()
                .is_some_and(|target_sha| target_sha == &evidence.sha256)
                || functional_evidence_hashes.contains(&evidence.sha256)
            {
                return Err(invalid(
                    "receiver_weight_cross_model_wrong_target_evidence_reused",
                ));
            }
        }
        wrong_functional_signatures.push(wrong.values);
    }
    let calibration = ReceiverCalibrationSet {
        receiver_snapshot_binding_sha256: None,
        functional_signatures,
        receiver_solutions,
        wrong_functional_signatures,
    };
    let (numerical, requested_signature, capability_ir_readout) =
        if let Some(reference) = &request.capability_ir_readout {
            let cross_model = protocol
                .cross_model()
                .ok_or_else(|| invalid("receiver_readout_requires_cross_model_protocol"))?;
            if protocol_self_test {
                return Err(invalid(
                    "receiver_readout_self_test_not_capability_evidence",
                ));
            }
            let authenticated = authenticate_linear_readout_details(root, reference)?;
            let evidence_reference = target
                .functional_evidence
                .as_ref()
                .ok_or_else(|| invalid("receiver_readout_functional_evidence_missing"))?;
            let evidence: CrossModelFunctionalEvidenceWire = read_record(root, evidence_reference)?;
            if authenticated.observed.capability_id != target.capability_id
                || authenticated.receipt.model_sha256 != cross_model.donor_model_sha256
                || authenticated.observed.prompts != evidence.prompts
                || authenticated.observed.prompt_sha256 != evidence.prompt_sha256
                || functional_signature_digest(&authenticated.observed.observed_margins)
                    != functional_signature_digest(&evidence.raw_logit_margins)
                || evidence.protocol_self_test
            {
                return Err(integrity("receiver_readout_functional_evidence_binding"));
            }
            let projection = read_cross_model_projection(root, cross_model)?;
            let compiled = compile_receiver_readout_capability(
                &ReceiverReadoutCapabilityInput {
                    ir: &authenticated.ir,
                    envelope: &authenticated.envelope,
                    readout_weights: &authenticated.weights,
                    inputs: &authenticated.observed.inputs,
                    projection_mean: &projection.mean,
                    projection_components: &projection.components,
                },
                &calibration,
                &request.safety.protected_cortex,
                &risk_metric,
                &request.policy,
                selected_proposal_method,
            )?;
            if serde_json::to_vec(&compiled.execution)?
                != serde_json::to_vec(&authenticated.receipt.execution)?
            {
                return Err(integrity("receiver_readout_execution_replay_mismatch"));
            }
            let summary = ReceiverCapabilityIrReadoutSummary {
                acquisition: reference.clone(),
                capability_ir: authenticated.receipt.capability_ir,
                partial_bundle: authenticated.receipt.partial_bundle,
                execution: compiled.execution,
                requested_signature: compiled.requested_signature.clone(),
                observed_forward_maximum_absolute_error: authenticated
                    .receipt
                    .observed_forward_maximum_absolute_error,
                maximum_roundoff_bound: authenticated.receipt.maximum_roundoff_bound,
                evidence_scope: authenticated.receipt.evidence_scope,
            };
            (
                compiled.numerical,
                compiled.requested_signature,
                Some(summary),
            )
        } else {
            let numerical = match selected_proposal_method {
                // V2 uses the same strict profile as calibration. Authenticated
                // behavioral evidence remains mandatory above; it cannot relax the
                // numerical decoder or inverse gates.
                ReceiverProposalMethod::DecodeThenProject => compile_receiver_signature(
                    &target.values,
                    &calibration,
                    &request.safety.protected_cortex,
                    &risk_metric,
                    &request.policy,
                )?,
                ReceiverProposalMethod::FitProtectedCoordinates => {
                    compile_receiver_signature_in_safe_coordinates(
                        &target.values,
                        &calibration,
                        &request.safety.protected_cortex,
                        &risk_metric,
                        &request.policy,
                    )?
                }
                ReceiverProposalMethod::CalibratedAffine => {
                    crate::receiver_compiler::compile_receiver_signature_calibrated_affine(
                        &target.values,
                        &calibration,
                        &request.safety.protected_cortex,
                        &risk_metric,
                        &request.policy,
                    )?
                }
                ReceiverProposalMethod::RelationalAnchors => {
                    crate::receiver_compiler::compile_receiver_signature_relational(
                        &target.values,
                        &calibration,
                        &request.safety.protected_cortex,
                        &risk_metric,
                        &request.policy,
                    )?
                }
            };
            (numerical, target.values.clone(), None)
        };
    let functional_support = if protocol.is_cross_model() {
        Some(functional_support_envelope(
            &calibration.functional_signatures,
            &requested_signature,
            request.policy.ridge,
        )?)
    } else {
        None
    };
    let proposed_coordinate_norm = norm(&numerical.proposed_receiver_coordinates)?;
    let mut blockers = Vec::new();
    if request.required_realization_scope == ReceiverRealizationScope::DistributedTransformer {
        if !realization.distributed_compilation_permitted {
            blockers.push(ReceiverWeightBlocker::InsufficientRealizationCoverage);
        }
        if capability_ir_readout.is_none() {
            blockers.push(ReceiverWeightBlocker::CapabilityIrEvidenceMissing);
        }
    }
    if protocol_self_test {
        blockers.push(ReceiverWeightBlocker::ProtocolSelfTestOnly);
    }
    if !numerical.allowed {
        blockers.push(ReceiverWeightBlocker::NumericalPredictionGatesFailed);
    }
    // Cross-model queries are gated in the functional space that actually
    // defines the compilation request.  Local receiver-response experiments
    // retain the historical raw receiver-radius gate for compatibility.
    if let Some((score, maximum_loo)) = functional_support {
        if score > maximum_loo * (1.0 + 1e-10) {
            blockers.push(ReceiverWeightBlocker::OutsideCalibratedFunctionalSupport);
        }
    } else if proposed_coordinate_norm > maximum_calibration_coordinate_norm * (1.0 + 1e-10) {
        blockers.push(ReceiverWeightBlocker::OutsideCalibratedCoordinateRadius);
    }
    if numerical.proposal_method == ReceiverProposalMethod::RelationalAnchors
        && !numerical.proposal_within_calibrated_support
    {
        blockers.push(ReceiverWeightBlocker::OutsideCalibratedRelationalSupport);
    }
    let sources = basis
        .axes
        .iter()
        .zip(&numerical.target_delta)
        .map(|(axis, coefficient)| (axis.delta.clone(), *coefficient))
        .collect::<Vec<_>>();
    let dense_delta = if blockers.is_empty() {
        Some(derive_content_addressed_dvec_combination(root, &sources)?)
    } else {
        None
    };
    Ok((
        ReceiverWeightCandidate {
            schema: CANDIDATE_SCHEMA.into(),
            request: request_reference.clone(),
            compiler_source_sha256: Sha256Digest::parse(env!("TIDEX_SOURCE_TREE_DIGEST"))?,
            basis_sha256: request.basis.sha256,
            protocol_sha256: request.protocol.sha256,
            target_capability_id: target.capability_id,
            calibration_observation_count: request.observations.len(),
            calibration_capability_count: capability_ids.len(),
            numerical,
            capability_ir_readout,
            maximum_calibration_coordinate_norm,
            proposed_coordinate_norm,
            functional_support_score: functional_support.map(|value| value.0),
            maximum_calibration_loo_functional_support_score: functional_support
                .map(|value| value.1),
            behavioral_calibration,
            realization,
            blockers,
            dense_delta,
            model_execution_verified: false,
            authorizes_promotion: false,
        },
        sources,
    ))
}

fn candidate_path(root: &Path, digest: &Sha256Digest) -> PathBuf {
    root.join("state/receiver_weight_candidates/by-sha")
        .join(format!("{digest}.json"))
}

pub fn prepare_receiver_weight_candidate(
    root: &Path,
    request_reference: &PrivateFileReference,
) -> BrainResult<PrivateFileReference> {
    let root = verify_internal_private_root(root)?;
    let (candidate, sources) = derive_candidate(&root, request_reference)?;
    if let Some(expected) = &candidate.dense_delta {
        let actual = ArtifactWriteAuthority::for_internal_root(&root)?
            .combine_content_addressed_dvec(&sources)?;
        if &actual != expected {
            return Err(integrity("receiver_weight_dense_rederivation_mismatch"));
        }
    }
    let bytes = serde_json::to_vec(&candidate)?;
    let digest = Sha256Digest::digest_bytes(&bytes);
    let path = candidate_path(&root, &digest);
    write_or_verify_immutable(&root, &path, &bytes)?;
    Ok(PrivateFileReference::new(path, digest))
}

pub fn authenticate_receiver_weight_candidate(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<ReceiverWeightCandidate> {
    let root = verify_internal_private_root(root)?;
    let bytes = reference.read_verified_bounded(&root, MAX_RECORD_BYTES)?;
    if reference.path != candidate_path(&root, &reference.sha256) {
        return Err(integrity("receiver_weight_candidate_path_mismatch"));
    }
    let stored: ReceiverWeightCandidate = serde_json::from_slice(&bytes)?;
    let (recomputed, _) = derive_candidate(&root, &stored.request)?;
    // Compare canonical bytes to avoid changing arithmetic by round-tripping
    // a candidate's floating-point measurements before recomputation.
    if serde_json::to_vec(&recomputed)? != bytes {
        return Err(integrity(
            "receiver_weight_candidate_recomputation_mismatch",
        ));
    }
    if let Some(delta) = &recomputed.dense_delta {
        verify_dvec_reference_under_root(&root, delta)?;
    }
    Ok(recomputed)
}

/// Materialize only a recomputed, unblocked candidate. Neither an arbitrary
/// caller-supplied coefficient vector nor an edited `allowed` flag is accepted.
/// The output is confined to this experimental private root, never installed
/// as the active model. Independent execution remains outstanding.
pub fn materialize_receiver_weight_candidate(
    root: &Path,
    candidate_reference: &PrivateFileReference,
    base_model: &Path,
    output: &Path,
) -> BrainResult<ReceiverWeightCheckpointReceipt> {
    let root = verify_internal_private_root(root)?;
    let candidate = authenticate_receiver_weight_candidate(&root, candidate_reference)?;
    let delta = candidate
        .dense_delta
        .as_ref()
        .filter(|_| candidate.blockers.is_empty())
        .ok_or_else(|| invalid("receiver_weight_candidate_blocked"))?;
    let request: ReceiverWeightRequest = read_record(&root, &candidate.request)?;
    let basis: ReceiverWeightBasis = read_record(&root, &request.basis)?;
    root_relative_path(&root, output)?;
    ensure_private_parent(&root, output)?;
    let materialization = materialize_dense_delta_checkpoint(
        &root,
        base_model,
        &basis.base_model_sha256,
        &basis.layout,
        delta,
        output,
    )?;
    if materialization.parameter_layout_sha256 != parameter_layout_digest(&basis.layout)? {
        return Err(integrity("receiver_weight_materialization_layout_mismatch"));
    }
    let receipt = ReceiverWeightCheckpointReceipt {
        schema: "cerebro.tidex.receiver_weight_checkpoint/v1".into(),
        candidate: candidate_reference.clone(),
        output_path: output.to_path_buf(),
        materialization,
        model_execution_verified: false,
        authorizes_promotion: false,
    };
    let bytes = serde_json::to_vec(&receipt)?;
    let sha = Sha256Digest::digest_bytes(&bytes);
    let path = root
        .join("state/receiver_weight_checkpoints/by-sha")
        .join(format!("{sha}.json"));
    write_or_verify_immutable(&root, &path, &bytes)?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::TensorId;
    use crate::security::secure_dir;
    use crate::weight_actuator::{
        inspect_model_safetensors, parameter_layout_for_tensors, read_model_tensor_f32,
    };
    use std::fs;
    use std::fs::File;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Fixture {
        root: PathBuf,
        base: PathBuf,
        request: ReceiverWeightRequest,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
    fn digest(label: &str) -> Sha256Digest {
        Sha256Digest::digest_bytes(label.as_bytes())
    }
    fn put<T: Serialize>(root: &Path, value: &T) -> PrivateFileReference {
        let bytes = serde_json::to_vec(value).unwrap();
        let sha = Sha256Digest::digest_bytes(&bytes);
        let path = root
            .join("state/test_inputs/by-sha")
            .join(format!("{sha}.json"));
        write_or_verify_immutable(root, &path, &bytes).unwrap();
        PrivateFileReference::new(path, sha)
    }
    fn fixture() -> Fixture {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tidex-weight-binding-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        let base = root.join("base.safetensors");
        // Real SafeTensors, dvec files and checkpoint writing; the numerical
        // measurements below are unit-test fixtures, not LLM evidence.
        let mut header = serde_json::to_vec(&serde_json::json!({
            "__metadata__":{"format":"pt"},
            "model.norm.weight":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},
            "model.other.weight":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        for value in [1.0_f32, 1.0, 1.0, 1.0, 7.0] {
            bytes.extend(value.to_le_bytes());
        }
        fs::write(&base, bytes).unwrap();
        let inventory = inspect_model_safetensors(&base).unwrap();
        let tensor = TensorId::parse("model.norm.weight").unwrap();
        let layout = parameter_layout_for_tensors(&inventory, &[tensor]).unwrap();
        let writer = ArtifactWriteAuthority::for_internal_root(&root).unwrap();
        let basis = ReceiverWeightBasis {
            schema: BASIS_SCHEMA_V1.into(),
            base_model_sha256: inventory.model_sha256.clone(),
            layout,
            axes: vec![
                ReceiverWeightAxis {
                    axis_id: SkillId::parse("axis-a").unwrap(),
                    delta: writer
                        .create_content_addressed_dvec(&[1.0, 0.0, 0.0, 0.0])
                        .unwrap(),
                    source_import_receipt: None,
                },
                ReceiverWeightAxis {
                    axis_id: SkillId::parse("axis-b").unwrap(),
                    delta: writer
                        .create_content_addressed_dvec(&[0.0, 1.0, 0.0, 0.0])
                        .unwrap(),
                    source_import_receipt: None,
                },
            ],
            construction_capability_ids: BTreeSet::from([CapabilityId::parse(
                "calibration.numeric:v1",
            )
            .unwrap()]),
            construction_evidence: put(
                &root,
                &serde_json::json!({"profile":"exact_linear_test_fixture"}),
            ),
            realization: None,
        };
        let basis_ref = put(&root, &basis);
        let protocol = ReceiverResponseProtocol {
            schema: PROTOCOL_SCHEMA.into(),
            measure: ReceiverResponseMeasure::NextTokenLogitMarginChange,
            base_model_sha256: inventory.model_sha256,
            model_config_sha256: digest("config"),
            tokenizer_sha256: digest("tokenizer"),
            collector_sha256: digest("collector"),
            coordinate_ids: vec![
                ProbeId::parse("probe.a").unwrap(),
                ProbeId::parse("probe.b").unwrap(),
            ],
            prompt_sha256: vec![digest("prompt-a"), digest("prompt-b")],
            positive_token_id: 1,
            negative_token_id: 2,
            max_input_tokens: 64,
        };
        let protocol_ref = put(&root, &protocol);
        let mut observations = Vec::new();
        for (index, coordinates) in [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
            [0.5, 2.0],
            [-0.3, -0.4],
        ]
        .into_iter()
        .enumerate()
        {
            observations.push(put(
                &root,
                &ReceiverResponseObservation {
                    schema: OBSERVATION_SCHEMA.into(),
                    observation_id: ObservationId::parse(format!("obs-{index}")).unwrap(),
                    capability_id: CapabilityId::parse("calibration.numeric:v1").unwrap(),
                    basis_sha256: basis_ref.sha256.clone(),
                    protocol_sha256: protocol_ref.sha256.clone(),
                    receiver_coordinates: coordinates.to_vec(),
                    values: coordinates.to_vec(),
                    functional_evidence: None,
                    receiver_solution_evidence: None,
                },
            ));
        }
        let target = FunctionalResponseTarget {
            schema: TARGET_SCHEMA.into(),
            capability_id: CapabilityId::parse("heldout.numeric:v1").unwrap(),
            protocol_sha256: protocol_ref.sha256.clone(),
            values: vec![0.2, 0.4],
            functional_evidence: None,
        };
        let wrong = FunctionalResponseTarget {
            capability_id: CapabilityId::parse("wrong.numeric:v1").unwrap(),
            values: vec![-0.2, -0.4],
            ..target.clone()
        };
        let request = ReceiverWeightRequest {
            schema: REQUEST_SCHEMA.into(),
            basis: basis_ref,
            protocol: protocol_ref,
            target: put(&root, &target),
            observations,
            wrong_targets: vec![put(&root, &wrong)],
            safety: ReceiverCoordinateSafety {
                protected_cortex: ProtectedCortex {
                    parameter_importance: vec![0.0; 2],
                    directions: vec![],
                    max_damage_ratio: 0.1,
                },
                risk_metric: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                evidence: put(
                    &root,
                    &serde_json::json!({"profile":"numeric_fixture_not_measured_llm_safety"}),
                ),
            },
            policy: ReceiverCompilerPolicy {
                schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
                ridge: 1e-9,
                minimum_decoder_loo_r2: 0.99,
                minimum_encoder_loo_r2: 0.99,
                minimum_decoder_loo_cosine: 0.99,
                maximum_functional_relative_error: 1e-4,
                minimum_identity_margin: 0.1,
                maximum_quadratic_cost: 1e6,
            },
            required_realization_scope: ReceiverRealizationScope::ReadoutControl,
            behavioral_calibration_evidence: None,
            behavioral_calibration_policy: None,
            proposal_method: None,
            capability_ir_readout: None,
        };
        Fixture {
            root,
            base,
            request,
        }
    }
    fn cross_modelize(f: &mut Fixture) {
        let basis: ReceiverWeightBasis = read_record(&f.root, &f.request.basis).unwrap();
        let donor_sha = digest("cross-model-donor");
        let projection = put(
            &f.root,
            &serde_json::json!({
                "schema": CROSS_MODEL_PROJECTION_SCHEMA,
                "projection_arithmetic": PROJECTION_ARITHMETIC,
                "calibration_capability_ids": basis.construction_capability_ids,
                "raw_probe_sha256": [digest("raw-a"),digest("raw-b")],
                "raw_dimension": 2,
                "mean": [0.0,0.0],
                "components": [[1.0,0.0],[0.0,1.0]],
                "retained_energy": 1.0,
                "donor_model_sha256": donor_sha,
                "projected_dimension": 2,
                "target_values_used": false,
                "confirmation_target_capability_ids_used": []
            }),
        );
        let protocol = CrossModelFunctionalSignatureProtocol {
            schema: CROSS_MODEL_PROTOCOL_SCHEMA.into(),
            measure: CrossModelSignatureMeasure::DonorNextTokenLogitMarginProjection,
            receiver_base_model_sha256: basis.base_model_sha256.clone(),
            donor_model_sha256: donor_sha.clone(),
            donor_model_config_sha256: digest("donor-config"),
            donor_tokenizer_sha256: digest("donor-tokenizer"),
            collector_sha256: digest("collector"),
            raw_probe_sha256: vec![digest("raw-a"), digest("raw-b")],
            coordinate_ids: vec![
                ProbeId::parse("functional.a").unwrap(),
                ProbeId::parse("functional.b").unwrap(),
            ],
            projection_evidence: projection.clone(),
            max_input_tokens: 64,
        };
        let protocol_ref = put(&f.root, &protocol);
        for index in 0..f.request.observations.len() {
            let mut observation: ReceiverResponseObservation =
                read_record(&f.root, &f.request.observations[index]).unwrap();
            observation.protocol_sha256 = protocol_ref.sha256.clone();
            observation.functional_evidence = Some(put(
                &f.root,
                &serde_json::json!({
                    "schema": CROSS_MODEL_FUNCTIONAL_EVIDENCE_SCHEMA,
                    "capability_id": observation.capability_id,
                    "donor_model_sha256": donor_sha,
                    "projection_evidence_sha256": projection.sha256,
                    "projected_signature_sha256": functional_signature_digest(&observation.values),
                    "projected_signature": observation.values.clone(),
                "raw_logit_margins": observation.values.clone(),
                "raw_probe_sha256": [digest("raw-a"),digest("raw-b")],
                "prompts": ["actual donor prompt a","actual donor prompt b"],
                "prompt_sha256": [digest("actual donor prompt a"),digest("actual donor prompt b")],
                    "receiver_data_used": false
                }),
            ));
            observation.receiver_solution_evidence = Some(put(
                &f.root,
                &serde_json::json!({
                    "schema":"cerebro.tidex.cross_model_receiver_solution_evidence/v1",
                    "capability_id": observation.capability_id,
                    "receiver_model_sha256": basis.base_model_sha256,
                    "receiver_coordinates_sha256": receiver_coordinates_digest(&observation.receiver_coordinates),
                    "receiver_coordinates": observation.receiver_coordinates.clone(),
                    "target_capability": false,
                    "lora_used": false,
                    "backpropagation_used": false,
                    "direct_execution_verified": true
                }),
            ));
            f.request.observations[index] = put(&f.root, &observation);
        }
        let mut target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        target.protocol_sha256 = protocol_ref.sha256.clone();
        target.functional_evidence = Some(put(
            &f.root,
            &serde_json::json!({
                "schema": CROSS_MODEL_FUNCTIONAL_EVIDENCE_SCHEMA,
                "capability_id": target.capability_id,
                "donor_model_sha256": donor_sha,
                "projection_evidence_sha256": projection.sha256,
                "projected_signature_sha256": functional_signature_digest(&target.values),
                "projected_signature": target.values.clone(),
                "raw_logit_margins": target.values.clone(),
                "raw_probe_sha256": [digest("raw-a"),digest("raw-b")],
                "prompts": ["actual donor prompt a","actual donor prompt b"],
                "prompt_sha256": [digest("actual donor prompt a"),digest("actual donor prompt b")],
                "receiver_data_used": false
            }),
        ));
        f.request.target = put(&f.root, &target);
        let mut wrong: FunctionalResponseTarget =
            read_record(&f.root, &f.request.wrong_targets[0]).unwrap();
        wrong.protocol_sha256 = protocol_ref.sha256.clone();
        wrong.functional_evidence = Some(put(
            &f.root,
            &serde_json::json!({
                "schema": CROSS_MODEL_FUNCTIONAL_EVIDENCE_SCHEMA,
                "capability_id": wrong.capability_id,
                "donor_model_sha256": donor_sha,
                "projection_evidence_sha256": projection.sha256,
                "projected_signature_sha256": functional_signature_digest(&wrong.values),
                "projected_signature": wrong.values.clone(),
                "raw_logit_margins": wrong.values.clone(),
                "raw_probe_sha256": [digest("raw-a"),digest("raw-b")],
                "prompts": ["actual donor prompt a","actual donor prompt b"],
                "prompt_sha256": [digest("actual donor prompt a"),digest("actual donor prompt b")],
                "receiver_data_used": false
            }),
        ));
        f.request.wrong_targets = vec![put(&f.root, &wrong)];
        f.request.protocol = protocol_ref;
        f.request.behavioral_calibration_evidence = Some(put(
            &f.root,
            &serde_json::json!({
                "schema": RECEIVER_BEHAVIORAL_CALIBRATION_EVIDENCE_SCHEMA,
                "receiver_model_sha256": basis.base_model_sha256,
                "calibration_capability_ids": ["calibration.numeric:v1"],
                "proposal_method": "decode_then_project",
                "base_accuracies": [0.25],
                "compiled_loo_accuracies": [0.75],
                "confirmation_target_capability_ids_used": [],
                "target_receiver_execution_performed": false,
                "unit_test_fixture": true
            }),
        ));
        f.request.behavioral_calibration_policy = Some(ReceiverBehavioralCalibrationPolicy {
            schema: "cerebro.tidex.receiver_behavioral_calibration_policy/v1".into(),
            minimum_mean_accuracy: 0.5,
            minimum_mean_gain: 0.1,
            minimum_non_degrading_count: 1,
        });
        f.request.proposal_method = Some(ReceiverProposalMethod::DecodeThenProject);
    }

    fn prepare(f: &Fixture) -> BrainResult<PrivateFileReference> {
        prepare_receiver_weight_candidate(&f.root, &put(&f.root, &f.request))
    }
    fn replace_observation(
        f: &mut Fixture,
        index: usize,
        edit: impl FnOnce(&mut ReceiverResponseObservation),
    ) {
        let mut value: ReceiverResponseObservation =
            read_record(&f.root, &f.request.observations[index]).unwrap();
        edit(&mut value);
        f.request.observations[index] = put(&f.root, &value);
    }

    struct ReadoutFixture {
        input: AcquireLinearReadoutInput,
        observed: ObservedLinearReadout,
    }
    impl Drop for ReadoutFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.input.source_root);
        }
    }
    impl ReadoutFixture {
        fn save_observed(&self) {
            fs::write(
                self.input
                    .source_root
                    .join(&self.input.evidence_relative_path),
                serde_json::to_vec(&self.observed).unwrap(),
            )
            .unwrap();
        }
        fn acquire(&self, root: &Path) -> BrainResult<PrivateFileReference> {
            self.save_observed();
            acquire_linear_readout(root, &self.input)
        }
    }

    fn readout_fixture(f: &Fixture) -> ReadoutFixture {
        let source_root = f.root.with_extension("readout-source");
        fs::create_dir(&source_root).unwrap();
        secure_dir(&source_root).unwrap();
        let model_path = f.root.join("readout-donor.safetensors");
        let mut header = serde_json::to_vec(&serde_json::json!({
            "lm_head.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        for value in [1.0_f32, 0.1, 0.0, 0.0] {
            bytes.extend(value.to_le_bytes());
        }
        fs::write(&model_path, bytes).unwrap();
        let target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        let description = describe_linear_readout(&DescribeLinearReadoutInput {
            schema: "cerebro.tidex.describe_linear_readout_input/v1".into(),
            model_path: model_path.clone(),
            tensor_id: TensorId::parse("lm_head.weight").unwrap(),
            positive_row: 0,
            negative_row: 1,
            capability_id: target.capability_id.clone(),
        })
        .unwrap();
        fs::write(
            source_root.join("operator.json"),
            description.descriptor_json,
        )
        .unwrap();
        let inputs = vec![
            vec![0.25, f64::from(0.2_f32)],
            vec![0.5, f64::from(0.2_f32)],
        ];
        let positives = inputs
            .iter()
            .map(|input| {
                let first = input[0] as f32;
                let second = (input[1] as f32) * 0.1_f32;
                f64::from(first + second)
            })
            .collect::<Vec<_>>();
        let observed = ObservedLinearReadout {
            schema: OBSERVED_LINEAR_READOUT_SCHEMA.into(),
            capability_id: target.capability_id,
            model_sha256: description.inspection.model_sha256,
            tensor_id: TensorId::parse("lm_head.weight").unwrap(),
            positive_row: 0,
            negative_row: 1,
            arithmetic_profile: "f32_cpu_no_tf32/v1".into(),
            inputs,
            observed_positive_logits: positives.clone(),
            observed_negative_logits: vec![0.0; 2],
            observed_margins: positives,
            prompts: vec![
                "actual donor prompt a".into(),
                "actual donor prompt b".into(),
            ],
            prompt_sha256: vec![
                digest("actual donor prompt a"),
                digest("actual donor prompt b"),
            ],
            receiver_data_used: false,
        };
        let result = ReadoutFixture {
            input: AcquireLinearReadoutInput {
                schema: "cerebro.tidex.acquire_linear_readout_input/v1".into(),
                source_root,
                model_path,
                descriptor_relative_path: "operator.json".into(),
                evidence_relative_path: "observations.json".into(),
            },
            observed,
        };
        result.save_observed();
        result
    }

    fn bind_readout_to_cross_model_fixture(
        f: &mut Fixture,
        readout: &ReadoutFixture,
        acquisition: PrivateFileReference,
    ) {
        let mut protocol: CrossModelFunctionalSignatureProtocol =
            read_record(&f.root, &f.request.protocol).unwrap();
        let mut projection: serde_json::Value =
            read_record(&f.root, &protocol.projection_evidence).unwrap();
        projection["donor_model_sha256"] =
            serde_json::to_value(&readout.observed.model_sha256).unwrap();
        protocol.projection_evidence = put(&f.root, &projection);
        protocol.donor_model_sha256 = readout.observed.model_sha256.clone();
        let protocol_reference = put(&f.root, &protocol);
        let rebind = |reference: &PrivateFileReference, values: Option<&[f64]>| {
            let mut evidence: serde_json::Value = read_record(&f.root, reference).unwrap();
            evidence["donor_model_sha256"] =
                serde_json::to_value(&readout.observed.model_sha256).unwrap();
            evidence["projection_evidence_sha256"] =
                serde_json::to_value(&protocol.projection_evidence.sha256).unwrap();
            if let Some(values) = values {
                evidence["raw_logit_margins"] = serde_json::to_value(values).unwrap();
                evidence["projected_signature"] = serde_json::to_value(values).unwrap();
                evidence["projected_signature_sha256"] =
                    serde_json::to_value(functional_signature_digest(values)).unwrap();
            }
            put(&f.root, &evidence)
        };
        for reference in &mut f.request.observations {
            let mut observation: ReceiverResponseObservation =
                read_record(&f.root, reference).unwrap();
            observation.protocol_sha256 = protocol_reference.sha256.clone();
            observation.functional_evidence = Some(rebind(
                observation.functional_evidence.as_ref().unwrap(),
                None,
            ));
            *reference = put(&f.root, &observation);
        }
        for reference in &mut f.request.wrong_targets {
            let mut target: FunctionalResponseTarget = read_record(&f.root, reference).unwrap();
            target.protocol_sha256 = protocol_reference.sha256.clone();
            target.functional_evidence =
                Some(rebind(target.functional_evidence.as_ref().unwrap(), None));
            *reference = put(&f.root, &target);
        }
        let mut target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        target.values = readout.observed.observed_margins.clone();
        target.protocol_sha256 = protocol_reference.sha256.clone();
        target.functional_evidence = Some(rebind(
            target.functional_evidence.as_ref().unwrap(),
            Some(&target.values),
        ));
        f.request.target = put(&f.root, &target);
        f.request.protocol = protocol_reference;
        f.request.capability_ir_readout = Some(acquisition);
    }

    #[test]
    fn linear_readout_acquisition_authenticates_real_checkpoint_capture_ir_and_replay() {
        let f = fixture();
        let readout = readout_fixture(&f);
        let reference = readout.acquire(&f.root).unwrap();
        let receipt = authenticate_linear_readout(&f.root, &reference).unwrap();
        assert!(receipt.execution.candidate_only);
        assert!(!receipt.authorizes_promotion);
        assert!(receipt.observed_forward_maximum_absolute_error <= receipt.maximum_roundoff_bound);
        assert_ne!(
            functional_signature_digest(&receipt.execution.raw_margins),
            functional_signature_digest(&readout.observed.observed_margins)
        );
        let partial = authenticate_capability_bundle(&f.root, &receipt.partial_bundle).unwrap();
        assert!(matches!(
            partial.representation(),
            crate::capability_bundle::CapabilityRepresentation::Partial { gaps, .. }
                if gaps == &readout_gaps()
        ));
        assert_eq!(readout.acquire(&f.root).unwrap(), reference);

        // Re-sealing a fake PASS is rejected by recomputation, not its self-hash.
        let mut forged = receipt.clone();
        forged.observed_forward_maximum_absolute_error = 0.0;
        forged.authorizes_promotion = true;
        let bytes = serde_json::to_vec(&forged).unwrap();
        let sha = Sha256Digest::digest_bytes(&bytes);
        let path = readout_acquisition_path(&f.root, &sha);
        write_or_verify_immutable(&f.root, &path, &bytes).unwrap();
        assert!(
            authenticate_linear_readout(&f.root, &PrivateFileReference::new(path, sha)).is_err()
        );

        // Deep replay reads the same real checkpoint, not just a saved digest.
        let mut model = fs::read(&readout.input.model_path).unwrap();
        let last = model.len() - 1;
        model[last] ^= 1;
        fs::write(&readout.input.model_path, model).unwrap();
        assert!(authenticate_linear_readout(&f.root, &reference).is_err());
    }

    #[test]
    fn linear_readout_acquisition_rejects_wrong_forward_prompt_and_f32_evidence() {
        let f = fixture();
        let mut readout = readout_fixture(&f);
        let original = readout.observed.clone();
        readout.observed.observed_positive_logits[0] =
            f64::from((readout.observed.observed_positive_logits[0] as f32) + 1.0_f32);
        readout.observed.observed_margins[0] = readout.observed.observed_positive_logits[0];
        assert!(readout.acquire(&f.root).is_err());
        readout.observed = original.clone();
        readout.observed.prompts[0].push_str(" changed");
        assert!(readout.acquire(&f.root).is_err());
        readout.observed = original.clone();
        readout.observed.inputs[0][1] = 0.2_f64; // not an exact F32 value
        assert!(readout.acquire(&f.root).is_err());
        readout.observed = original.clone();
        readout.observed.observed_margins[0] = f64::from(0.5_f32);
        assert!(readout.acquire(&f.root).is_err());
        readout.observed = original.clone();
        readout.observed.receiver_data_used = true;
        assert!(readout.acquire(&f.root).is_err());
        readout.observed = original;
        assert!(readout.acquire(&f.root).is_ok());
    }

    #[test]
    fn linear_readout_candidate_is_driven_by_executed_ir_not_supplied_signature() {
        let mut f = fixture();
        cross_modelize(&mut f);
        let readout = readout_fixture(&f);
        let acquisition = readout.acquire(&f.root).unwrap();
        bind_readout_to_cross_model_fixture(&mut f, &readout, acquisition);
        let reference = prepare(&f).unwrap();
        let candidate = authenticate_receiver_weight_candidate(&f.root, &reference).unwrap();
        let summary = candidate.capability_ir_readout.as_ref().unwrap();
        let target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        assert_eq!(summary.requested_signature, summary.execution.raw_margins);
        assert_ne!(
            functional_signature_digest(&summary.requested_signature),
            functional_signature_digest(&target.values)
        );
        let mut functional = Vec::new();
        for reference in &f.request.observations {
            let observation: ReceiverResponseObservation = read_record(&f.root, reference).unwrap();
            functional.push(observation.values);
        }
        let expected_support = functional_support_envelope(
            &functional,
            &summary.requested_signature,
            f.request.policy.ridge,
        )
        .unwrap();
        assert_eq!(candidate.functional_support_score, Some(expected_support.0));
        assert!(!candidate.model_execution_verified);
        assert!(!candidate.authorizes_promotion);

        // Same numeric target with another authentic prompt is insufficient:
        // acquisition and functional-response evidence must describe one run.
        let mut evidence: serde_json::Value =
            read_record(&f.root, target.functional_evidence.as_ref().unwrap()).unwrap();
        evidence["prompts"][0] = serde_json::json!("another authentic prompt");
        evidence["prompt_sha256"][0] =
            serde_json::to_value(digest("another authentic prompt")).unwrap();
        let mut changed = target;
        changed.functional_evidence = Some(put(&f.root, &evidence));
        f.request.target = put(&f.root, &changed);
        assert!(prepare(&f).is_err());
    }

    #[test]
    fn signature_to_dense_to_checkpoint_uses_one_existing_actuator() {
        let f = fixture();
        let before = fs::read(&f.base).unwrap();
        let reference = prepare(&f).unwrap();
        let candidate = authenticate_receiver_weight_candidate(&f.root, &reference).unwrap();
        assert!(candidate.blockers.is_empty());
        assert_eq!(candidate.calibration_observation_count, 7);
        assert_eq!(candidate.calibration_capability_count, 1);
        assert!(!candidate.model_execution_verified);
        assert!(!candidate.authorizes_promotion);
        let out = f.root.join("candidate/model.safetensors");
        let receipt =
            materialize_receiver_weight_candidate(&f.root, &reference, &f.base, &out).unwrap();
        assert!(!receipt.authorizes_promotion);
        assert!(!receipt.model_execution_verified);
        assert!(!receipt.materialization.requires_adapter_at_runtime);
        let values =
            read_model_tensor_f32(&out, &TensorId::parse("model.norm.weight").unwrap()).unwrap();
        for (observed, expected) in values.iter().zip([1.2, 1.4, 1.0, 1.0]) {
            assert!((*observed - expected).abs() < 1e-6);
        }
        assert_eq!(
            read_model_tensor_f32(&out, &TensorId::parse("model.other.weight").unwrap()).unwrap(),
            vec![7.0]
        );
        assert_eq!(fs::read(&f.base).unwrap(), before);
        assert!(materialize_receiver_weight_candidate(&f.root, &reference, &f.base, &out).is_err());
    }
    #[test]
    fn cross_model_evidence_is_semantically_bound_not_only_hash_authenticated() {
        let mut f = fixture();
        cross_modelize(&mut f);
        let reference = prepare(&f).unwrap();
        let candidate = authenticate_receiver_weight_candidate(&f.root, &reference).unwrap();
        assert!(candidate.blockers.is_empty(), "{candidate:#?}");
        assert_eq!(
            candidate.numerical.validation_profile,
            crate::receiver_compiler::ReceiverProposalValidationProfile::ParametricCrossValidation
        );
        assert!(candidate.functional_support_score.is_some());
        assert!(candidate
            .maximum_calibration_loo_functional_support_score
            .is_some());

        let mut observation: ReceiverResponseObservation =
            read_record(&f.root, &f.request.observations[0]).unwrap();
        let evidence = observation.functional_evidence.as_ref().unwrap();
        let mut value: serde_json::Value = read_record(&f.root, evidence).unwrap();
        value["projected_signature"] = serde_json::json!([9.0, 9.0]);
        observation.functional_evidence = Some(put(&f.root, &value));
        f.request.observations[0] = put(&f.root, &observation);
        assert!(prepare(&f)
            .unwrap_err()
            .to_string()
            .contains("functional_values_mismatch"));
    }

    #[test]
    fn cross_model_support_uses_functional_leverage_and_blocks_extreme_query() {
        let mut f = fixture();
        cross_modelize(&mut f);
        let mut target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        target.values = vec![100.0, 200.0];
        let protocol: CrossModelFunctionalSignatureProtocol =
            read_record(&f.root, &f.request.protocol).unwrap();
        target.functional_evidence = Some(put(
            &f.root,
            &serde_json::json!({
                "schema": CROSS_MODEL_FUNCTIONAL_EVIDENCE_SCHEMA,
                "capability_id": target.capability_id,
                "donor_model_sha256": protocol.donor_model_sha256,
                "projection_evidence_sha256": protocol.projection_evidence.sha256,
                "projected_signature_sha256": functional_signature_digest(&target.values),
                "projected_signature": target.values.clone(),
                "raw_logit_margins": target.values.clone(),
                "raw_probe_sha256": [digest("raw-a"),digest("raw-b")],
                "prompts": ["actual donor prompt a","actual donor prompt b"],
                "prompt_sha256": [digest("actual donor prompt a"),digest("actual donor prompt b")],
                "receiver_data_used": false
            }),
        ));
        f.request.target = put(&f.root, &target);
        let reference = prepare(&f).unwrap();
        let candidate = authenticate_receiver_weight_candidate(&f.root, &reference).unwrap();
        assert!(candidate
            .blockers
            .contains(&ReceiverWeightBlocker::OutsideCalibratedFunctionalSupport));
        assert!(candidate.dense_delta.is_none());
    }

    #[test]
    fn canonical_projection_preserves_sequential_rounding_without_fma() {
        let observed =
            project_functional_signature(&[1.0e16, 1.0, -1.0e16, 3.0], &[0.0; 4], &[vec![1.0; 4]])
                .unwrap();
        assert_eq!(observed, vec![3.0]);
        let step = 2.0_f64.powi(-27);
        let cancellation =
            project_functional_signature(&[-1.0, 1.0 + step], &[0.0; 2], &[vec![1.0, 1.0 - step]])
                .unwrap();
        assert_eq!(cancellation[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            project_functional_signature(&[4.0, -9.0], &[4.0, -9.0], &[vec![1.0, 2.0]]).unwrap(),
            vec![0.0],
        );
        assert!(project_functional_signature(&[f64::MAX], &[-f64::MAX], &[vec![1.0]]).is_err());
    }

    #[test]
    fn reselling_modified_raw_values_cannot_reuse_a_projected_signature() {
        let mut f = fixture();
        cross_modelize(&mut f);
        let mut target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        let mut evidence: serde_json::Value =
            read_record(&f.root, target.functional_evidence.as_ref().unwrap()).unwrap();
        evidence["raw_logit_margins"][0] = serde_json::json!(123.0);
        target.functional_evidence = Some(put(&f.root, &evidence));
        f.request.target = put(&f.root, &target);
        assert!(prepare(&f)
            .unwrap_err()
            .to_string()
            .contains("functional_derivation_mismatch"));
    }

    #[test]
    fn functional_ir_binds_actual_prompt_text_and_raw_probe_order() {
        for mutate_prompt in [true, false] {
            let mut f = fixture();
            cross_modelize(&mut f);
            let mut target: FunctionalResponseTarget =
                read_record(&f.root, &f.request.target).unwrap();
            let mut evidence: serde_json::Value =
                read_record(&f.root, target.functional_evidence.as_ref().unwrap()).unwrap();
            if mutate_prompt {
                evidence["prompts"][0] = serde_json::json!("a different donor input");
            } else {
                evidence["raw_probe_sha256"]
                    .as_array_mut()
                    .unwrap()
                    .swap(0, 1);
            }
            target.functional_evidence = Some(put(&f.root, &evidence));
            f.request.target = put(&f.root, &target);
            assert!(prepare(&f)
                .unwrap_err()
                .to_string()
                .contains("raw_response_binding_mismatch"));
        }
    }

    #[test]
    fn authenticated_but_changed_projection_must_recompute_the_response() {
        let mut f = fixture();
        cross_modelize(&mut f);
        let mut protocol: CrossModelFunctionalSignatureProtocol =
            read_record(&f.root, &f.request.protocol).unwrap();
        let target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        let mut projection: serde_json::Value =
            read_record(&f.root, &protocol.projection_evidence).unwrap();
        projection["components"][0][0] = serde_json::json!(2.0);
        protocol.projection_evidence = put(&f.root, &projection);
        let mut evidence: serde_json::Value =
            read_record(&f.root, target.functional_evidence.as_ref().unwrap()).unwrap();
        evidence["projection_evidence_sha256"] =
            serde_json::to_value(&protocol.projection_evidence.sha256).unwrap();
        let reference = put(&f.root, &evidence);
        assert!(validate_cross_model_functional_evidence(
            &f.root,
            &reference,
            &protocol,
            &target.capability_id,
            &target.values
        )
        .unwrap_err()
        .to_string()
        .contains("functional_derivation_mismatch"));
    }

    #[test]
    fn projection_lineage_cannot_silently_include_a_target() {
        let mut f = fixture();
        cross_modelize(&mut f);
        let basis: ReceiverWeightBasis = read_record(&f.root, &f.request.basis).unwrap();
        let mut protocol: CrossModelFunctionalSignatureProtocol =
            read_record(&f.root, &f.request.protocol).unwrap();
        let target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        let mut projection: serde_json::Value =
            read_record(&f.root, &protocol.projection_evidence).unwrap();
        projection["calibration_capability_ids"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::to_value(&target.capability_id).unwrap());
        protocol.projection_evidence = put(&f.root, &projection);
        assert!(validate_cross_model_protocol(&f.root, &protocol, &basis)
            .unwrap_err()
            .to_string()
            .contains("projection_basis_lineage_mismatch"));
    }

    #[test]
    fn protocol_self_test_can_validate_but_never_materialize_weights() {
        let mut f = fixture();
        cross_modelize(&mut f);
        let mut target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        let mut evidence: serde_json::Value =
            read_record(&f.root, target.functional_evidence.as_ref().unwrap()).unwrap();
        evidence["protocol_self_test"] = serde_json::json!(true);
        evidence["source_calibration_capability_id"] = serde_json::json!("calibration.numeric:v1");
        target.functional_evidence = Some(put(&f.root, &evidence));
        f.request.target = put(&f.root, &target);
        let reference = prepare(&f).unwrap();
        let candidate = authenticate_receiver_weight_candidate(&f.root, &reference).unwrap();
        assert!(candidate
            .blockers
            .contains(&ReceiverWeightBlocker::ProtocolSelfTestOnly));
        assert!(candidate.dense_delta.is_none());
        let output = f.root.join("forbidden-self-test.safetensors");
        assert!(
            materialize_receiver_weight_candidate(&f.root, &reference, &f.base, &output).is_err()
        );
        assert!(!output.exists());
    }

    #[test]
    fn cross_language_vector_digest_contract_matches_v69_python_authority() {
        let functional = [
            -3.9674553684356426,
            1.4197970821035546,
            -1.048170994521103,
            -1.9670339420367746,
        ];
        let coordinates = [1.0, -2.5, 0.0, 3.25];
        assert_eq!(
            functional_signature_digest(&functional).as_str(),
            "b8e65755b09cb335298a94bc2cb75713d5c96110edbc42699862916dcfc0593d"
        );
        assert_eq!(
            receiver_coordinates_digest(&coordinates).as_str(),
            "63e56aaf80e69e4468bd0633880bda4ba08321a358600abf0ee9029fffb5bb93"
        );
    }

    #[test]
    fn python_json_roundtrip_preserves_exact_functional_signature_digest() {
        // Golden Python json.dumps output: the fourth value used to move by one
        // ULP in serde_json's default parser. Literal-only digest tests missed
        // the actual producer -> decimal JSON -> Rust authority boundary.
        let wire =
            br#"[-3.9674553684356426,1.4197970821035546,-1.048170994521103,-1.9670339420367746]"#;
        let typed: Vec<f64> = serde_json::from_slice(wire).unwrap();
        let value: serde_json::Value = serde_json::from_slice(wire).unwrap();
        let untyped = value
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_f64().unwrap())
            .collect::<Vec<_>>();
        let expected = "b8e65755b09cb335298a94bc2cb75713d5c96110edbc42699862916dcfc0593d";
        assert_eq!(functional_signature_digest(&typed).as_str(), expected);
        assert_eq!(functional_signature_digest(&untyped).as_str(), expected);
        assert_eq!(typed[3].to_bits(), 0xbfff78f89532b7f0);
        let roundtrip: Vec<f64> =
            serde_json::from_slice(&serde_json::to_vec(&typed).unwrap()).unwrap();
        assert_eq!(functional_signature_digest(&roundtrip).as_str(), expected);
        let mut changed = typed;
        changed[3] = f64::from_bits(changed[3].to_bits() ^ 1);
        assert_ne!(functional_signature_digest(&changed).as_str(), expected);
    }

    #[test]
    fn json_authority_preserves_signed_zero_subnormals_and_finite_extremes() {
        let wire = br#"[-0.0,5e-324,2.2250738585072014e-308,1.7976931348623157e308]"#;
        let values: Vec<f64> = serde_json::from_slice(wire).unwrap();
        let expected = [
            0x8000000000000000,
            1,
            0x0010000000000000,
            0x7fefffffffffffff,
        ];
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
        let roundtrip: Vec<f64> =
            serde_json::from_slice(&serde_json::to_vec(&values).unwrap()).unwrap();
        assert_eq!(
            functional_signature_digest(&values),
            functional_signature_digest(&roundtrip)
        );
        assert_ne!(
            receiver_coordinates_digest(&values),
            functional_signature_digest(&values)
        );
    }

    #[test]
    fn preparation_is_idempotent_and_authentication_read_only() {
        let f = fixture();
        let a = prepare(&f).unwrap();
        let b = prepare(&f).unwrap();
        assert_eq!(a, b);
        let report = authenticate_receiver_weight_candidate(&f.root, &a).unwrap();
        let delta = report.dense_delta.unwrap();
        fs::remove_file(&delta.path).unwrap();
        assert!(authenticate_receiver_weight_candidate(&f.root, &a).is_err());
        assert!(!delta.path.exists());
    }
    #[test]
    fn forged_candidate_flags_and_coordinates_are_recomputed_not_trusted() {
        let f = fixture();
        let original = prepare(&f).unwrap();
        let mut value: serde_json::Value = read_record(&f.root, &original).unwrap();
        value["authorizes_promotion"] = serde_json::json!(true);
        value["numerical"]["target_delta"] = serde_json::json!([999.0, 999.0]);
        let bytes = serde_json::to_vec(&value).unwrap();
        let sha = Sha256Digest::digest_bytes(&bytes);
        let path = candidate_path(&f.root, &sha);
        write_or_verify_immutable(&f.root, &path, &bytes).unwrap();
        assert!(authenticate_receiver_weight_candidate(
            &f.root,
            &PrivateFileReference::new(path, sha)
        )
        .is_err());
    }
    #[test]
    fn target_may_not_appear_in_basis_construction_lineage() {
        let mut f = fixture();
        let mut basis: ReceiverWeightBasis = read_record(&f.root, &f.request.basis).unwrap();
        basis
            .construction_capability_ids
            .insert(CapabilityId::parse("heldout.numeric:v1").unwrap());
        f.request.basis = put(&f.root, &basis);
        assert!(prepare(&f)
            .unwrap_err()
            .to_string()
            .contains("leaked_into_basis"));
    }
    #[test]
    fn target_may_not_appear_in_calibration_observations() {
        let mut f = fixture();
        replace_observation(&mut f, 0, |o| {
            o.capability_id = CapabilityId::parse("heldout.numeric:v1").unwrap()
        });
        assert!(prepare(&f)
            .unwrap_err()
            .to_string()
            .contains("leaked_into_calibration"));
    }
    #[test]
    fn changed_protocol_with_same_dimension_is_rejected() {
        let mut f = fixture();
        replace_observation(&mut f, 0, |o| o.protocol_sha256 = digest("other-protocol"));
        assert!(prepare(&f)
            .unwrap_err()
            .to_string()
            .contains("binding_invalid"));
    }
    #[test]
    fn relabeling_identical_coordinates_is_not_new_evidence() {
        let mut f = fixture();
        replace_observation(&mut f, 1, |o| o.receiver_coordinates = vec![1.0, -0.0]);
        assert!(prepare(&f)
            .unwrap_err()
            .to_string()
            .contains("duplicate_calibration_coordinates"));
    }
    #[test]
    fn same_shape_foreign_basis_binding_is_rejected() {
        let mut f = fixture();
        replace_observation(&mut f, 0, |o| o.basis_sha256 = digest("other-basis"));
        assert!(prepare(&f).is_err());
    }
    #[test]
    fn outside_calibrated_radius_records_blocker_without_delta_or_checkpoint() {
        let mut f = fixture();
        let mut target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        target.values = vec![100.0, 200.0];
        f.request.target = put(&f.root, &target);
        let reference = prepare(&f).unwrap();
        let candidate = authenticate_receiver_weight_candidate(&f.root, &reference).unwrap();
        assert!(candidate
            .blockers
            .contains(&ReceiverWeightBlocker::OutsideCalibratedCoordinateRadius));
        assert!(candidate.dense_delta.is_none());
        let out = f.root.join("never.safetensors");
        assert!(materialize_receiver_weight_candidate(&f.root, &reference, &f.base, &out).is_err());
        assert!(!out.exists());
    }
    #[test]
    fn identical_wrong_signature_cannot_satisfy_identity_separation() {
        let mut f = fixture();
        let target: FunctionalResponseTarget = read_record(&f.root, &f.request.target).unwrap();
        let mut wrong: FunctionalResponseTarget =
            read_record(&f.root, &f.request.wrong_targets[0]).unwrap();
        wrong.values = target.values;
        f.request.wrong_targets = vec![put(&f.root, &wrong)];
        let reference = prepare(&f).unwrap();
        let candidate = authenticate_receiver_weight_candidate(&f.root, &reference).unwrap();
        assert!(candidate.dense_delta.is_none());
    }
    #[test]
    fn wrong_base_checkpoint_never_gets_an_output() {
        let f = fixture();
        let reference = prepare(&f).unwrap();
        let mut bytes = fs::read(&f.base).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&f.base, bytes).unwrap();
        let out = f.root.join("never.safetensors");
        assert!(materialize_receiver_weight_candidate(&f.root, &reference, &f.base, &out).is_err());
        assert!(!out.exists());
    }
    #[test]
    fn tampered_axis_artifact_is_rejected() {
        let f = fixture();
        let basis: ReceiverWeightBasis = read_record(&f.root, &f.request.basis).unwrap();
        let path = &basis.axes[0].delta.path;
        let mut bytes = fs::read(path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(path, bytes).unwrap();
        assert!(prepare(&f).is_err());
    }
    #[test]
    fn duplicate_axis_and_shape_mismatch_cannot_enter_binding() {
        let mut f = fixture();
        let mut basis: ReceiverWeightBasis = read_record(&f.root, &f.request.basis).unwrap();
        basis.axes[1] = basis.axes[0].clone();
        f.request.basis = put(&f.root, &basis);
        assert!(prepare(&f).is_err());
        basis.axes[1].axis_id = SkillId::parse("axis-c").unwrap();
        basis.axes[1].delta.parameter_count = 5;
        f.request.basis = put(&f.root, &basis);
        assert!(prepare(&f).is_err());
    }
    #[test]
    fn resource_and_safety_bounds_fail_before_numerical_work() {
        let mut f = fixture();
        f.request.observations = vec![f.request.observations[0].clone(); MAX_ANCHORS + 1];
        assert!(prepare(&f).is_err());
        let mut f = fixture();
        f.request.safety.risk_metric = vec![vec![1.0]];
        assert!(prepare(&f).is_err());
        let mut f = fixture();
        f.request.safety.risk_metric = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
        assert!(prepare(&f).is_err());
    }
    #[test]
    fn distributed_request_blocks_legacy_final_norm_basis() {
        let mut f = fixture();
        f.request.required_realization_scope = ReceiverRealizationScope::DistributedTransformer;
        let reference = prepare(&f).unwrap();
        let candidate = authenticate_receiver_weight_candidate(&f.root, &reference).unwrap();
        assert!(candidate.dense_delta.is_none());
        assert_eq!(
            candidate.realization.available_scope,
            ReceiverRealizationScope::ReadoutControl
        );
        assert!(candidate
            .blockers
            .contains(&ReceiverWeightBlocker::InsufficientRealizationCoverage));
        assert!(candidate
            .blockers
            .contains(&ReceiverWeightBlocker::CapabilityIrEvidenceMissing));
    }

    #[test]
    fn distributed_manifest_requires_all_layers_attention_and_mlp() {
        let f = fixture();
        let mut basis: ReceiverWeightBasis = read_record(&f.root, &f.request.basis).unwrap();
        let families = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ];
        let shapes = (0..2)
            .flat_map(|layer| {
                families.map(|family| crate::block_tomography::BlockShapeSpec {
                    name: format!("model.layers.{layer}.module.{family}.weight"),
                    shape: vec![1, 1],
                    count: 1,
                })
            })
            .collect::<Vec<_>>();
        basis.schema = BASIS_SCHEMA.into();
        basis.layout = ParameterBlockLayout::from_shapes(&shapes).unwrap();
        basis.realization = Some(ReceiverRealizationManifest {
            schema: REALIZATION_SCHEMA.into(),
            scope: ReceiverRealizationScope::DistributedTransformer,
            axis_construction_method: ReceiverAxisConstructionMethod::CalibrationLora,
            total_model_parameter_count: 20,
            writable_parameter_count: 14,
            learned_parameter_count_per_axis: 28,
            target_tensor_families: families.map(str::to_string).into_iter().collect(),
            covered_transformer_layers: BTreeSet::from([0, 1]),
            total_transformer_layers: 2,
            target_capability_used_in_basis: false,
            target_receiver_execution_used_in_basis: false,
        });
        let assessment =
            assess_realization(&basis, ReceiverRealizationScope::DistributedTransformer).unwrap();
        assert!(assessment.distributed_compilation_permitted);
        assert!(assessment.full_layer_coverage);
        assert!(assessment.attention_and_mlp_coverage);
        assert!(!assessment.complete_capability_claim_permitted);

        basis
            .realization
            .as_mut()
            .unwrap()
            .target_tensor_families
            .remove("down_proj");
        assert!(
            assess_realization(&basis, ReceiverRealizationScope::DistributedTransformer).is_err()
        );
    }

    #[test]
    fn calibration_lora_solution_is_allowed_only_when_bound_to_one_basis_axis() {
        let f = fixture();
        let mut basis: ReceiverWeightBasis = read_record(&f.root, &f.request.basis).unwrap();
        basis.schema = BASIS_SCHEMA.into();
        basis.realization = Some(ReceiverRealizationManifest {
            schema: REALIZATION_SCHEMA.into(),
            scope: ReceiverRealizationScope::DistributedTransformer,
            axis_construction_method: ReceiverAxisConstructionMethod::CalibrationLora,
            total_model_parameter_count: 8,
            writable_parameter_count: 4,
            learned_parameter_count_per_axis: 4,
            target_tensor_families: BTreeSet::from(["q_proj".into()]),
            covered_transformer_layers: BTreeSet::from([0]),
            total_transformer_layers: 1,
            target_capability_used_in_basis: false,
            target_receiver_execution_used_in_basis: false,
        });
        let capability = CapabilityId::parse("calibration.numeric:v1").unwrap();
        let coordinates = vec![1.0, 0.0];
        let evidence = put(
            &f.root,
            &serde_json::json!({
                "schema": CROSS_MODEL_RECEIVER_SOLUTION_EVIDENCE_SCHEMA,
                "capability_id": capability,
                "receiver_model_sha256": basis.base_model_sha256,
                "receiver_coordinates": coordinates,
                "receiver_coordinates_sha256": receiver_coordinates_digest(&coordinates),
                "target_capability": false,
                "lora_used": true,
                "backpropagation_used": true,
                "direct_execution_verified": true,
                "receiver_backend_frozen_before_target_observation": true,
                "optimizer_steps": 8,
                "trainable_parameter_count": 4,
                "axis_delta_sha256": basis.axes[0].delta.sha256
            }),
        );
        validate_cross_model_receiver_solution_evidence(
            &f.root,
            &evidence,
            &basis,
            &capability,
            &coordinates,
        )
        .unwrap();

        let mut bad: serde_json::Value = read_record(&f.root, &evidence).unwrap();
        bad["receiver_coordinates"] = serde_json::json!([0.5, 0.5]);
        bad["receiver_coordinates_sha256"] =
            serde_json::to_value(receiver_coordinates_digest(&[0.5, 0.5])).unwrap();
        let bad = put(&f.root, &bad);
        assert!(validate_cross_model_receiver_solution_evidence(
            &f.root,
            &bad,
            &basis,
            &capability,
            &[0.5, 0.5],
        )
        .is_err());
    }

    fn write_f32_archive(path: &Path, tensors: &[(String, Vec<f32>, Vec<usize>)]) {
        let mut data = Vec::new();
        let mut entries = serde_json::Map::new();
        entries.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        for (name, values, shape) in tensors {
            let start = data.len();
            for value in values {
                data.extend(value.to_le_bytes());
            }
            entries.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, data.len()]
                }),
            );
        }
        let mut header = serde_json::to_vec(&serde_json::Value::Object(entries)).unwrap();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&data).unwrap();
        file.sync_all().unwrap();
    }

    #[test]
    fn distributed_lora_basis_assembly_replays_every_axis_and_rejects_forgery() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tidex-distributed-basis-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        let base = root.join("base.safetensors");
        let families = [
            ("self_attn", "q_proj"),
            ("self_attn", "k_proj"),
            ("self_attn", "v_proj"),
            ("self_attn", "o_proj"),
            ("mlp", "gate_proj"),
            ("mlp", "up_proj"),
            ("mlp", "down_proj"),
        ];
        let base_tensors = families
            .iter()
            .map(|(group, family)| {
                (
                    format!("model.layers.0.{group}.{family}.weight"),
                    vec![0.0_f32; 4],
                    vec![2, 2],
                )
            })
            .collect::<Vec<_>>();
        write_f32_archive(&base, &base_tensors);
        let config = root.join("adapter_config.json");
        fs::write(
            &config,
            serde_json::to_vec(&serde_json::json!({
                "r": 1,
                "lora_alpha": 1.0,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "bias": "none",
                "use_rslora": false,
                "use_dora": false
            }))
            .unwrap(),
        )
        .unwrap();

        let mut bindings = Vec::new();
        let mut delta_sha = Vec::new();
        let mut capability_ids = BTreeSet::new();
        let mut original_receipts = Vec::new();
        for axis_index in 0..5 {
            let adapter = root.join(format!("adapter-{axis_index}.safetensors"));
            let mut tensors = Vec::new();
            for (family_index, (group, family)) in families.iter().enumerate() {
                let stem = format!("base_model.model.model.layers.0.{group}.{family}");
                tensors.push((
                    format!("{stem}.lora_A.weight"),
                    vec![1.0 + axis_index as f32, 1.0 + family_index as f32],
                    vec![1, 2],
                ));
                tensors.push((
                    format!("{stem}.lora_B.weight"),
                    vec![1.0, 2.0 + axis_index as f32],
                    vec![2, 1],
                ));
            }
            write_f32_archive(&adapter, &tensors);
            let receipt = import_peft_lora_as_dense_axis(
                &root,
                &LoraAdapterAxisInput {
                    schema: LORA_ADAPTER_AXIS_INPUT_SCHEMA.into(),
                    base_model_path: base.clone(),
                    adapter_model_path: adapter,
                    adapter_config_path: config.clone(),
                },
            )
            .unwrap();
            delta_sha.push(receipt.dense_delta.sha256.clone());
            let reference = put(&root, &receipt);
            let capability_id =
                CapabilityId::parse(format!("calibration.threshold.{axis_index}:v1")).unwrap();
            capability_ids.insert(capability_id.clone());
            bindings.push(DistributedLoraAxisBinding {
                capability_id,
                axis_id: SkillId::parse(format!("distributed-axis-{axis_index}")).unwrap(),
                import_receipt: reference,
            });
            original_receipts.push(receipt);
        }
        let evidence = put(
            &root,
            &serde_json::json!({
                "schema": DISTRIBUTED_LORA_BASIS_EVIDENCE_SCHEMA,
                "receiver_model_sha256": original_receipts[0].base_model_sha256,
                "calibration_capability_ids": capability_ids,
                "confirmation_target_capability_ids_used": [],
                "axis_delta_sha256": delta_sha,
                "receiver_backend_frozen_before_target_observation": true,
                "target_receiver_solution_used": false,
                "target_receiver_execution_performed": false
            }),
        );
        let input = DistributedLoraBasisInput {
            schema: DISTRIBUTED_LORA_BASIS_INPUT_SCHEMA.into(),
            axes: bindings,
            construction_evidence: evidence,
            total_model_parameter_count: 28,
            total_transformer_layers: 1,
        };
        let basis = assemble_distributed_lora_basis(&root, &input).unwrap();
        assert_eq!(basis.axes.len(), 5);
        assert_eq!(basis.layout.total_parameter_count, 28);
        assert!(basis
            .axes
            .iter()
            .all(|axis| axis.source_import_receipt.is_some()));
        assert!(
            assess_realization(&basis, ReceiverRealizationScope::DistributedTransformer)
                .unwrap()
                .distributed_compilation_permitted
        );

        let mut forged = original_receipts[0].clone();
        forged.learned_parameter_count += 1;
        let mut forged_input = input;
        forged_input.axes[0].import_receipt = put(&root, &forged);
        assert!(assemble_distributed_lora_basis(&root, &forged_input).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unknown_answer_or_promotion_fields_are_not_accepted_as_input() {
        let f = fixture();
        let mut value = serde_json::to_value(&f.request).unwrap();
        value["target_answers"] = serde_json::json!(["hidden"]);
        assert!(serde_json::from_value::<ReceiverWeightRequest>(value).is_err());
    }
    #[test]
    fn materialization_cannot_escape_the_private_root() {
        let f = fixture();
        let reference = prepare(&f).unwrap();
        let output = f
            .root
            .parent()
            .unwrap()
            .join(format!("escape-{}.safetensors", reference.sha256));
        assert!(
            materialize_receiver_weight_candidate(&f.root, &reference, &f.base, &output).is_err()
        );
        assert!(!output.exists());
    }
}
