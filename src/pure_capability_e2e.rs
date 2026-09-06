//! First explicit end-to-end profile for a pure deterministic capability.
//!
//! This module supports exactly `linear_map_f64/v1`. It is not a claim that
//! arbitrary programs can be discovered or translated. Source inspection is
//! confined to an authenticated retained-CAS projection. Terminal execution
//! is an in-process core computation over the authenticated candidate weights;
//! it receives no source path. Operating-system isolation is a later gate.

use crate::authority::{
    existing_regular_file_under_root, read_untrusted_private_file_bounded,
    write_or_verify_immutable, PrivateFileReference,
};
use crate::capability_ir::{
    authenticate_capability_ir, CapabilityIr, IrNode, OutputBinding, ParameterSlot, PrimitiveSet,
    TypedPort, ValueReference,
};
use crate::content_vault::{
    authenticate_capture_receipt, retained_source_adapter, CaptureReceipt, RetainedSourceAdapter,
};
use crate::digest::{CapabilityIrDigest, CaptureReceiptDigest, Sha256Digest, SystemEnvelopeDigest};
use crate::error::{BrainError, BrainResult};
use crate::identity::{CapabilityId, CapabilityNodeId, PortId, PrimitiveId};
use crate::isolated_execution::{
    bounded_stderr_detail, run_isolated, AuthenticatedBytes, IsolatedExecutionRequest,
    IsolationLimits, IsolationRequirements, PayloadExecutionEvidence,
};
use crate::security::verify_internal_private_root;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

const DESCRIPTOR_DOMAIN: &[u8] = b"CEREBRO:TIDEX:LINEAR-MAP-DESCRIPTOR:v1\0";
const DISCOVERY_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PURE-CAPABILITY-DISCOVERY:v1\0";
const TERMINAL_CASE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:TERMINAL-CASE:v1\0";
const TERMINAL_SUITE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:TERMINAL-SUITE:v1\0";
const CHALLENGE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:TERMINAL-CHALLENGE:v2\0";
const COMMITMENT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:TERMINAL-COMMITMENT:v2\0";
const CANDIDATE_DOMAIN: &[u8] = b"CEREBRO:TIDEX:PURE-WEIGHT-CANDIDATE:v2\0";
const OUTPUT_DOMAIN: &[u8] = b"CEREBRO:TIDEX:TERMINAL-OUTPUTS:v1\0";
const EVALUATION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:TERMINAL-EVALUATION:v2\0";
const ISOLATED_EVALUATION_DOMAIN: &[u8] = b"CEREBRO:TIDEX:ISOLATED-TERMINAL-EVALUATION:v1\0";
const PURE_LINEAR_RUNNER_ADMISSION_DOMAIN: &[u8] =
    b"CEREBRO:TIDEX:PURE-LINEAR-RUNNER-ADMISSION:v1\0";
const ISOLATED_BACKEND_CONTRACT: &str = "linux_bubblewrap_sealed_memfd/v2";
const RUNNER_INPUT_PATH: &str = "/tidex/input";
const MAX_RUNNER_BUILD_IDENTITY_BYTES: usize = 4_096;
const MAX_DESCRIPTOR_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RECORD_BYTES: u64 = 256 * 1024 * 1024;
const MAX_DIMENSION: usize = 65_536;
// `serde_json` emits every finite f64 in substantially fewer than 32 bytes.
// This conservative element bound makes the exact byte check below reachable
// without first allocating an object larger than the descriptor protocol.
const MAX_WEIGHTS: usize = 2_000_000;
const MAX_TERMINAL_CASES: usize = 65_536;
const MAX_TERMINAL_SCALARS: usize = 4_000_000;
const MAX_TERMINAL_WORK_UNITS: u128 = 250_000_000;

fn invalid(code: &str) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: &str) -> BrainError {
    BrainError::Integrity(code.into())
}

fn canonical_json_bounded<T: Serialize>(
    value: &T,
    max_bytes: u64,
    code: &str,
) -> BrainResult<Vec<u8>> {
    let bytes = serde_json::to_vec(value)?;
    let byte_count = u64::try_from(bytes.len()).map_err(|_| invalid(code))?;
    if byte_count > max_bytes {
        return Err(invalid(code));
    }
    Ok(bytes)
}

fn finish_sha256(hasher: Sha256) -> BrainResult<Sha256Digest> {
    Sha256Digest::parse(format!("{:x}", hasher.finalize()))
        .map_err(|_| integrity("sha256_finalization_invalid"))
}

fn bitwise_f64_slices_equal(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LinearMapDescriptorSchema {
    #[serde(rename = "cerebro.tidex.linear_map_f64/v1")]
    Current,
}

/// Canonical source descriptor for the one supported capability profile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinearMapDescriptor {
    schema: LinearMapDescriptorSchema,
    capability_id: CapabilityId,
    input_dimension: u64,
    output_dimension: u64,
    /// Row-major `[output, input]` resident parameters.
    weights: Vec<f64>,
    descriptor_sha256: Sha256Digest,
}

impl LinearMapDescriptor {
    pub fn new(
        capability_id: CapabilityId,
        input_dimension: usize,
        output_dimension: usize,
        weights: Vec<f64>,
    ) -> BrainResult<Self> {
        let mut descriptor = Self {
            schema: LinearMapDescriptorSchema::Current,
            capability_id,
            input_dimension: u64::try_from(input_dimension)
                .map_err(|_| invalid("linear_map_input_dimension_overflow"))?,
            output_dimension: u64::try_from(output_dimension)
                .map_err(|_| invalid("linear_map_output_dimension_overflow"))?,
            weights,
            descriptor_sha256: Sha256Digest::zero(),
        };
        descriptor.validate_structure()?;
        descriptor.descriptor_sha256 = descriptor.calculate_digest()?;
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn capability_id(&self) -> &CapabilityId {
        &self.capability_id
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn dimensions(&self) -> BrainResult<(usize, usize)> {
        let input = usize::try_from(self.input_dimension)
            .map_err(|_| invalid("linear_map_input_dimension_overflow"))?;
        let output = usize::try_from(self.output_dimension)
            .map_err(|_| invalid("linear_map_output_dimension_overflow"))?;
        Ok((input, output))
    }

    fn validate_structure(&self) -> BrainResult<()> {
        let (input, output) = self.dimensions()?;
        let count = input
            .checked_mul(output)
            .ok_or_else(|| invalid("linear_map_weight_count_overflow"))?;
        if self.schema != LinearMapDescriptorSchema::Current
            || input == 0
            || output == 0
            || input > MAX_DIMENSION
            || output > MAX_DIMENSION
            || count > MAX_WEIGHTS
            || self.weights.len() != count
            || self.weights.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("linear_map_descriptor_invalid"));
        }
        Ok(())
    }

    fn validate(&self) -> BrainResult<()> {
        self.validate_structure()?;
        if self.descriptor_sha256 == Sha256Digest::zero()
            || self.calculate_digest()? != self.descriptor_sha256
        {
            return Err(integrity("linear_map_descriptor_digest_mismatch"));
        }
        Ok(())
    }

    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.descriptor_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            DESCRIPTOR_DOMAIN,
            &canonical_json_bounded(
                &unsigned,
                MAX_DESCRIPTOR_BYTES,
                "linear_map_descriptor_too_large",
            )?,
        ))
    }
}

fn expected_linear_map_ir(
    descriptor: &LinearMapDescriptor,
    capture: &CaptureReceipt,
    descriptor_relative_path: &Path,
) -> BrainResult<CapabilityIr> {
    let (input, output) = descriptor.dimensions()?;
    CapabilityIr::new_with_parameters(
        descriptor.capability_id.clone(),
        capture.envelope(),
        PrimitiveSet::tidex_core_v1()?,
        vec![TypedPort::tensor_f64(
            PortId::parse("runtime_input")?,
            vec![
                u64::try_from(input).map_err(|_| invalid("linear_map_shape_overflow"))?,
                1,
            ],
        )?],
        vec![ParameterSlot::new(TypedPort::tensor_f64(
            PortId::parse("resident_weights")?,
            vec![
                u64::try_from(output).map_err(|_| invalid("linear_map_shape_overflow"))?,
                u64::try_from(input).map_err(|_| invalid("linear_map_shape_overflow"))?,
            ],
        )?)?],
        vec![IrNode::new(
            CapabilityNodeId::parse("node.linear_map")?,
            PrimitiveId::parse("tensor.matmul")?,
            vec![
                ValueReference::Parameter {
                    name: PortId::parse("resident_weights")?,
                },
                ValueReference::Input {
                    name: PortId::parse("runtime_input")?,
                },
            ],
            TypedPort::tensor_f64(
                PortId::parse("mapped")?,
                vec![
                    u64::try_from(output).map_err(|_| invalid("linear_map_shape_overflow"))?,
                    1,
                ],
            )?,
            vec![descriptor_relative_path.to_path_buf()],
        )?],
        vec![OutputBinding::new(
            TypedPort::tensor_f64(
                PortId::parse("runtime_output")?,
                vec![
                    u64::try_from(output).map_err(|_| invalid("linear_map_shape_overflow"))?,
                    1,
                ],
            )?,
            ValueReference::NodeOutput {
                node_id: CapabilityNodeId::parse("node.linear_map")?,
            },
        )?],
    )
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PureCapabilityDiscoverySchema {
    #[serde(rename = "cerebro.tidex.pure_capability_discovery/v1")]
    Current,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PureCapabilityDiscoveryReceipt {
    schema: PureCapabilityDiscoverySchema,
    capture_receipt: PrivateFileReference,
    capture_receipt_sha256: CaptureReceiptDigest,
    system_envelope_sha256: SystemEnvelopeDigest,
    descriptor_relative_path: PathBuf,
    descriptor_content_sha256: Sha256Digest,
    descriptor_semantic_sha256: Sha256Digest,
    capability_id: CapabilityId,
    capability_ir: PrivateFileReference,
    capability_ir_sha256: CapabilityIrDigest,
    manifest_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedPureCapabilityDiscovery {
    pub receipt: PrivateFileReference,
    pub capability_ir: PrivateFileReference,
}

pub struct PureCapabilityDiscoveryAuthority {
    root: PathBuf,
}

impl PureCapabilityDiscoveryAuthority {
    pub fn open(root: &Path) -> BrainResult<Self> {
        Ok(Self {
            root: verify_internal_private_root(root)?,
        })
    }

    pub fn discover_linear_map(
        &self,
        capture_reference: PrivateFileReference,
        source: &RetainedSourceAdapter,
        descriptor_relative_path: &Path,
    ) -> BrainResult<RecordedPureCapabilityDiscovery> {
        let descriptor_relative_path = normal_relative_path(descriptor_relative_path)?;
        let capture = authenticate_capture_receipt(&self.root, &capture_reference)?;
        verify_source_binding(&capture, source)?;
        let projection = source.materialize()?;
        let descriptor_path = existing_regular_file_under_root(
            &projection,
            &projection.join(&descriptor_relative_path),
        )?;
        let bytes = read_untrusted_private_file_bounded(
            &self.root,
            &descriptor_path,
            MAX_DESCRIPTOR_BYTES,
        )?;
        let descriptor_content_sha256 = Sha256Digest::digest_bytes(&bytes);
        let descriptor: LinearMapDescriptor = serde_json::from_slice(&bytes)?;
        descriptor.validate()?;
        if canonical_json_bounded(
            &descriptor,
            MAX_DESCRIPTOR_BYTES,
            "linear_map_descriptor_too_large",
        )? != bytes
        {
            return Err(integrity("linear_map_descriptor_noncanonical_encoding"));
        }
        source.verify_materialized(&projection)?;

        let ir = expected_linear_map_ir(&descriptor, &capture, &descriptor_relative_path)?;
        let capability_ir = ir.persist(&self.root, capture.envelope())?;
        let mut receipt = PureCapabilityDiscoveryReceipt {
            schema: PureCapabilityDiscoverySchema::Current,
            capture_receipt: capture_reference,
            capture_receipt_sha256: capture.manifest_sha256().clone(),
            system_envelope_sha256: capture.system_envelope_sha256().clone(),
            descriptor_relative_path,
            descriptor_content_sha256,
            descriptor_semantic_sha256: descriptor.descriptor_sha256,
            capability_id: descriptor.capability_id,
            capability_ir: capability_ir.clone(),
            capability_ir_sha256: ir.manifest_digest().clone(),
            manifest_sha256: Sha256Digest::zero(),
        };
        receipt.manifest_sha256 = discovery_digest(&receipt)?;
        let reference = persist_canonical(
            &self.root,
            "state/pure_capability_discovery/by-sha",
            &receipt.manifest_sha256,
            &receipt,
        )?;
        self.authenticate(&reference)?;
        Ok(RecordedPureCapabilityDiscovery {
            receipt: reference,
            capability_ir,
        })
    }

    pub fn authenticate(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<PureCapabilityDiscoveryReceipt> {
        let receipt: PureCapabilityDiscoveryReceipt = read_canonical(reference, &self.root)?;
        if receipt.schema != PureCapabilityDiscoverySchema::Current
            || discovery_digest(&receipt)? != receipt.manifest_sha256
            || reference.path
                != canonical_path(
                    &self.root,
                    "state/pure_capability_discovery/by-sha",
                    &receipt.manifest_sha256,
                )
        {
            return Err(integrity("pure_capability_discovery_invalid"));
        }
        let capture = authenticate_capture_receipt(&self.root, &receipt.capture_receipt)?;
        let source = retained_source_adapter(&self.root, &capture)?;
        verify_source_binding(&capture, &source)?;
        if capture.manifest_sha256() != &receipt.capture_receipt_sha256
            || capture.system_envelope_sha256() != &receipt.system_envelope_sha256
        {
            return Err(integrity("pure_capability_discovery_capture_mismatch"));
        }
        let projection = source.materialize()?;
        let path = existing_regular_file_under_root(
            &projection,
            &projection.join(&receipt.descriptor_relative_path),
        )?;
        let descriptor_reference =
            PrivateFileReference::new(path, receipt.descriptor_content_sha256.clone());
        let bytes = descriptor_reference.read_verified_bounded(&self.root, MAX_DESCRIPTOR_BYTES)?;
        let descriptor: LinearMapDescriptor = serde_json::from_slice(&bytes)?;
        descriptor.validate()?;
        if canonical_json_bounded(
            &descriptor,
            MAX_DESCRIPTOR_BYTES,
            "linear_map_descriptor_too_large",
        )? != bytes
            || descriptor.descriptor_sha256 != receipt.descriptor_semantic_sha256
            || descriptor.capability_id != receipt.capability_id
        {
            return Err(integrity("pure_capability_discovery_descriptor_mismatch"));
        }
        let ir =
            authenticate_capability_ir(&self.root, &receipt.capability_ir, capture.envelope())?;
        let expected_ir =
            expected_linear_map_ir(&descriptor, &capture, &receipt.descriptor_relative_path)?;
        if ir.manifest_digest() != &receipt.capability_ir_sha256
            || ir.capability_id() != &receipt.capability_id
            || ir != expected_ir
        {
            return Err(integrity("pure_capability_discovery_ir_mismatch"));
        }
        source.verify_materialized(&projection)?;
        Ok(receipt)
    }
}

fn discovery_digest(receipt: &PureCapabilityDiscoveryReceipt) -> BrainResult<Sha256Digest> {
    let mut unsigned = receipt.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    Ok(Sha256Digest::digest_domain(
        DISCOVERY_DOMAIN,
        &canonical_json_bounded(
            &unsigned,
            MAX_RECORD_BYTES,
            "pure_capability_discovery_record_too_large",
        )?,
    ))
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TerminalChallengeSchema {
    #[serde(rename = "cerebro.tidex.terminal_challenge/v2")]
    Current,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalChallengeState {
    Issued,
    Claimed,
    Revealed,
    Terminal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TerminalCase {
    input: Vec<f64>,
    expected_output: Vec<f64>,
}

impl TerminalCase {
    pub fn new(input: Vec<f64>, expected_output: Vec<f64>) -> BrainResult<Self> {
        if input.is_empty()
            || expected_output.is_empty()
            || input.iter().chain(&expected_output).any(|v| !v.is_finite())
        {
            return Err(invalid("terminal_case_invalid"));
        }
        Ok(Self {
            input,
            expected_output,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TerminalChallengeReveal {
    schema: TerminalChallengeSchema,
    challenge_id: String,
    cases: Vec<TerminalCase>,
    suite_sha256: Sha256Digest,
    reveal_sha256: Sha256Digest,
}

impl TerminalChallengeReveal {
    pub fn new(challenge_id: impl Into<String>, cases: Vec<TerminalCase>) -> BrainResult<Self> {
        let mut reveal = Self {
            schema: TerminalChallengeSchema::Current,
            challenge_id: challenge_id.into(),
            cases,
            suite_sha256: Sha256Digest::zero(),
            reveal_sha256: Sha256Digest::zero(),
        };
        reveal.validate_structure()?;
        reveal.suite_sha256 = terminal_suite_digest(&reveal.cases)?;
        reveal.reveal_sha256 = reveal.calculate_digest()?;
        reveal.validate()?;
        Ok(reveal)
    }

    pub fn suite_digest(&self) -> &Sha256Digest {
        &self.suite_sha256
    }

    fn validate_structure(&self) -> BrainResult<()> {
        if self.schema != TerminalChallengeSchema::Current
            || self.challenge_id.is_empty()
            || self.challenge_id.len() > 128
            || !self.challenge_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':')
            })
            || self.cases.is_empty()
            || self.cases.len() > MAX_TERMINAL_CASES
        {
            return Err(invalid("terminal_challenge_invalid"));
        }
        let mut inputs = BTreeSet::new();
        let mut total_scalars = 0_usize;
        for case in &self.cases {
            if case.input.is_empty()
                || case.expected_output.is_empty()
                || case
                    .input
                    .iter()
                    .chain(&case.expected_output)
                    .any(|value| !value.is_finite())
            {
                return Err(invalid("terminal_case_invalid"));
            }
            total_scalars = total_scalars
                .checked_add(case.input.len())
                .and_then(|total| total.checked_add(case.expected_output.len()))
                .ok_or_else(|| invalid("terminal_challenge_scalar_overflow"))?;
            if total_scalars > MAX_TERMINAL_SCALARS {
                return Err(invalid("terminal_challenge_scalar_limit"));
            }
            let identity = case
                .input
                .iter()
                .flat_map(|value| value.to_bits().to_be_bytes())
                .collect::<Vec<_>>();
            if !inputs.insert(Sha256Digest::digest_bytes(&identity)) {
                return Err(invalid("terminal_challenge_duplicate_input"));
            }
        }
        Ok(())
    }

    fn validate(&self) -> BrainResult<()> {
        self.validate_structure()?;
        if self.suite_sha256 == Sha256Digest::zero()
            || terminal_suite_digest(&self.cases)? != self.suite_sha256
            || self.reveal_sha256 == Sha256Digest::zero()
            || self.calculate_digest()? != self.reveal_sha256
        {
            return Err(integrity("terminal_challenge_digest_mismatch"));
        }
        Ok(())
    }

    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.reveal_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            CHALLENGE_DOMAIN,
            &canonical_json_bounded(
                &unsigned,
                MAX_RECORD_BYTES,
                "terminal_challenge_record_too_large",
            )?,
        ))
    }
}

fn terminal_case_digest(case: &TerminalCase) -> BrainResult<Sha256Digest> {
    let mut hasher = Sha256::new();
    hasher.update(TERMINAL_CASE_DOMAIN);
    hasher.update(
        u64::try_from(case.input.len())
            .map_err(|_| invalid("terminal_case_input_length_overflow"))?
            .to_be_bytes(),
    );
    for value in &case.input {
        hasher.update(value.to_bits().to_be_bytes());
    }
    hasher.update(
        u64::try_from(case.expected_output.len())
            .map_err(|_| invalid("terminal_case_output_length_overflow"))?
            .to_be_bytes(),
    );
    for value in &case.expected_output {
        hasher.update(value.to_bits().to_be_bytes());
    }
    finish_sha256(hasher)
}

/// Order-independent identity of the exact case suite. It deliberately omits
/// the caller label so relabeling or reordering cannot manufacture new evidence.
fn terminal_suite_digest(cases: &[TerminalCase]) -> BrainResult<Sha256Digest> {
    let mut cases = cases
        .iter()
        .map(terminal_case_digest)
        .collect::<BrainResult<Vec<_>>>()?;
    cases.sort();
    let mut hasher = Sha256::new();
    hasher.update(TERMINAL_SUITE_DOMAIN);
    hasher.update(
        u64::try_from(cases.len())
            .map_err(|_| invalid("terminal_suite_length_overflow"))?
            .to_be_bytes(),
    );
    for digest in cases {
        hasher.update(digest.as_str().as_bytes());
    }
    finish_sha256(hasher)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TerminalChallengeCommitment {
    schema: TerminalChallengeSchema,
    state: TerminalChallengeState,
    challenge_id: String,
    suite_sha256: Sha256Digest,
    reveal_sha256: Sha256Digest,
    manifest_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TerminalSuiteRegistration {
    schema: TerminalChallengeSchema,
    state: TerminalChallengeState,
    suite_sha256: Sha256Digest,
    challenge_id: String,
    commitment: PrivateFileReference,
    commitment_sha256: Sha256Digest,
}

impl TerminalChallengeCommitment {
    /// Persist only the commitment. Challenge cases are not written here.
    pub fn commit(
        root: &Path,
        reveal: &TerminalChallengeReveal,
    ) -> BrainResult<PrivateFileReference> {
        let root = verify_internal_private_root(root)?;
        reveal.validate()?;
        let mut commitment = Self {
            schema: TerminalChallengeSchema::Current,
            state: TerminalChallengeState::Issued,
            challenge_id: reveal.challenge_id.clone(),
            suite_sha256: reveal.suite_sha256.clone(),
            reveal_sha256: reveal.reveal_sha256.clone(),
            manifest_sha256: Sha256Digest::zero(),
        };
        commitment.manifest_sha256 = commitment_digest(&commitment)?;
        let commitment_path = canonical_path(
            &root,
            "state/terminal_challenges/commitments/by-sha",
            &commitment.manifest_sha256,
        );
        let commitment_bytes = canonical_json_bounded(
            &commitment,
            MAX_RECORD_BYTES,
            "terminal_commitment_record_too_large",
        )?;
        let expected_reference = PrivateFileReference::new(
            commitment_path,
            Sha256Digest::digest_bytes(&commitment_bytes),
        );
        let registration = TerminalSuiteRegistration {
            schema: TerminalChallengeSchema::Current,
            state: TerminalChallengeState::Issued,
            suite_sha256: reveal.suite_sha256.clone(),
            challenge_id: reveal.challenge_id.clone(),
            commitment: expected_reference.clone(),
            commitment_sha256: commitment.manifest_sha256.clone(),
        };
        write_or_verify_state(
            &root,
            &suite_registration_path(&root, &reveal.suite_sha256),
            &registration,
        )?;
        let reference = persist_canonical(
            &root,
            "state/terminal_challenges/commitments/by-sha",
            &commitment.manifest_sha256,
            &commitment,
        )?;
        if reference != expected_reference {
            return Err(integrity("terminal_commitment_reference_mismatch"));
        }
        authenticate_commitment(&root, &reference)?;
        Ok(reference)
    }
}

fn commitment_digest(value: &TerminalChallengeCommitment) -> BrainResult<Sha256Digest> {
    let mut unsigned = value.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    Ok(Sha256Digest::digest_domain(
        COMMITMENT_DOMAIN,
        &canonical_json_bounded(
            &unsigned,
            MAX_RECORD_BYTES,
            "terminal_commitment_record_too_large",
        )?,
    ))
}

fn authenticate_commitment(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<TerminalChallengeCommitment> {
    let value: TerminalChallengeCommitment = read_canonical(reference, root)?;
    if value.schema != TerminalChallengeSchema::Current
        || value.state != TerminalChallengeState::Issued
        || commitment_digest(&value)? != value.manifest_sha256
        || reference.path
            != canonical_path(
                root,
                "state/terminal_challenges/commitments/by-sha",
                &value.manifest_sha256,
            )
    {
        return Err(integrity("terminal_commitment_invalid"));
    }
    let registration = TerminalSuiteRegistration {
        schema: TerminalChallengeSchema::Current,
        state: TerminalChallengeState::Issued,
        suite_sha256: value.suite_sha256.clone(),
        challenge_id: value.challenge_id.clone(),
        commitment: reference.clone(),
        commitment_sha256: value.manifest_sha256.clone(),
    };
    verify_state(
        root,
        &suite_registration_path(root, &value.suite_sha256),
        &registration,
    )?;
    Ok(value)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateLifecycleStatus {
    CandidateOnlyNotPromoted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PureWeightCandidateReceipt {
    lifecycle: CandidateLifecycleStatus,
    capture_receipt_sha256: CaptureReceiptDigest,
    system_envelope_sha256: SystemEnvelopeDigest,
    discovery: PrivateFileReference,
    discovery_sha256: Sha256Digest,
    capability_ir: PrivateFileReference,
    capability_ir_sha256: CapabilityIrDigest,
    challenge_commitment: PrivateFileReference,
    challenge_commitment_sha256: Sha256Digest,
    input_dimension: u64,
    output_dimension: u64,
    weights: Vec<f64>,
    manifest_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TerminalChallengeClaim {
    schema: TerminalChallengeSchema,
    state: TerminalChallengeState,
    commitment: PrivateFileReference,
    commitment_sha256: Sha256Digest,
    suite_sha256: Sha256Digest,
    candidate: PrivateFileReference,
    candidate_sha256: Sha256Digest,
}

pub struct PureWeightCandidateAuthority {
    root: PathBuf,
}

impl PureWeightCandidateAuthority {
    pub fn open(root: &Path) -> BrainResult<Self> {
        Ok(Self {
            root: verify_internal_private_root(root)?,
        })
    }

    pub fn materialize(
        &self,
        discovery_reference: PrivateFileReference,
        challenge_commitment: PrivateFileReference,
    ) -> BrainResult<PrivateFileReference> {
        let discovery = PureCapabilityDiscoveryAuthority::open(&self.root)?
            .authenticate(&discovery_reference)?;
        let commitment = authenticate_commitment(&self.root, &challenge_commitment)?;
        let capture = authenticate_capture_receipt(&self.root, &discovery.capture_receipt)?;
        let descriptor = descriptor_for_discovery(&self.root, &capture, &discovery)?;
        let (input, output) = descriptor.dimensions()?;
        let mut candidate = PureWeightCandidateReceipt {
            lifecycle: CandidateLifecycleStatus::CandidateOnlyNotPromoted,
            capture_receipt_sha256: discovery.capture_receipt_sha256,
            system_envelope_sha256: discovery.system_envelope_sha256,
            discovery: discovery_reference,
            discovery_sha256: discovery.manifest_sha256,
            capability_ir: discovery.capability_ir,
            capability_ir_sha256: discovery.capability_ir_sha256,
            challenge_commitment,
            challenge_commitment_sha256: commitment.manifest_sha256.clone(),
            input_dimension: u64::try_from(input)
                .map_err(|_| invalid("candidate_dimension_overflow"))?,
            output_dimension: u64::try_from(output)
                .map_err(|_| invalid("candidate_dimension_overflow"))?,
            weights: descriptor.weights,
            manifest_sha256: Sha256Digest::zero(),
        };
        candidate.manifest_sha256 = candidate_digest(&candidate)?;
        let reference = persist_canonical(
            &self.root,
            "state/pure_weight_candidates/by-sha",
            &candidate.manifest_sha256,
            &candidate,
        )?;
        let claim = TerminalChallengeClaim {
            schema: TerminalChallengeSchema::Current,
            state: TerminalChallengeState::Claimed,
            commitment: candidate.challenge_commitment.clone(),
            commitment_sha256: commitment.manifest_sha256,
            suite_sha256: commitment.suite_sha256,
            candidate: reference.clone(),
            candidate_sha256: candidate.manifest_sha256,
        };
        write_or_verify_state(
            &self.root,
            &claim_path(&self.root, &claim.commitment_sha256),
            &claim,
        )?;
        self.authenticate(&reference)?;
        Ok(reference)
    }

    pub fn authenticate(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<PureWeightCandidateReceipt> {
        let candidate = self.authenticate_sealed(reference)?;
        let discovery = PureCapabilityDiscoveryAuthority::open(&self.root)?
            .authenticate(&candidate.discovery)?;
        let capture = authenticate_capture_receipt(&self.root, &discovery.capture_receipt)?;
        let descriptor = descriptor_for_discovery(&self.root, &capture, &discovery)?;
        let (input, output) = descriptor.dimensions()?;
        if candidate.discovery_sha256 != discovery.manifest_sha256
            || candidate.capture_receipt_sha256 != discovery.capture_receipt_sha256
            || candidate.system_envelope_sha256 != discovery.system_envelope_sha256
            || candidate.capability_ir != discovery.capability_ir
            || candidate.capability_ir_sha256 != discovery.capability_ir_sha256
            || candidate.input_dimension != input as u64
            || candidate.output_dimension != output as u64
            || !bitwise_f64_slices_equal(&candidate.weights, &descriptor.weights)
        {
            return Err(integrity("pure_weight_candidate_chain_mismatch"));
        }
        Ok(candidate)
    }

    /// Reopens a candidate already materialized by this private authority
    /// without consulting discovery, capture, retained projection, or CAS.
    ///
    /// This checks the immutable receipt, its self-hash, shape, commitment and
    /// one-shot claim. It is continuity under the same private authority, not
    /// an independent signature or a replacement for `authenticate`, which
    /// remains the explicit deep provenance check while source retention is
    /// available.
    pub fn authenticate_sealed(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<PureWeightCandidateReceipt> {
        let candidate: PureWeightCandidateReceipt = read_canonical(reference, &self.root)?;
        if candidate.lifecycle != CandidateLifecycleStatus::CandidateOnlyNotPromoted
            || candidate_digest(&candidate)? != candidate.manifest_sha256
            || reference.path
                != canonical_path(
                    &self.root,
                    "state/pure_weight_candidates/by-sha",
                    &candidate.manifest_sha256,
                )
        {
            return Err(integrity("pure_weight_candidate_invalid"));
        }
        let commitment = authenticate_commitment(&self.root, &candidate.challenge_commitment)?;
        let input = usize::try_from(candidate.input_dimension)
            .map_err(|_| invalid("candidate_dimension_overflow"))?;
        let output = usize::try_from(candidate.output_dimension)
            .map_err(|_| invalid("candidate_dimension_overflow"))?;
        let weight_count = input
            .checked_mul(output)
            .ok_or_else(|| invalid("candidate_weight_shape_overflow"))?;
        if candidate.capture_receipt_sha256.is_draft()
            || candidate.system_envelope_sha256.is_draft()
            || candidate.discovery_sha256 == Sha256Digest::zero()
            || candidate.capability_ir_sha256.is_draft()
            || candidate.challenge_commitment_sha256 != commitment.manifest_sha256
            || input == 0
            || output == 0
            || input > MAX_DIMENSION
            || output > MAX_DIMENSION
            || weight_count > MAX_WEIGHTS
            || candidate.weights.len() != weight_count
            || candidate.weights.iter().any(|value| !value.is_finite())
        {
            return Err(integrity("sealed_pure_weight_candidate_invalid"));
        }
        authenticate_claim(&self.root, reference, &candidate, &commitment)?;
        Ok(candidate)
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }
}

fn candidate_digest(candidate: &PureWeightCandidateReceipt) -> BrainResult<Sha256Digest> {
    let mut unsigned = candidate.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    Ok(Sha256Digest::digest_domain(
        CANDIDATE_DOMAIN,
        &canonical_json_bounded(
            &unsigned,
            MAX_RECORD_BYTES,
            "pure_weight_candidate_record_too_large",
        )?,
    ))
}

fn authenticate_claim(
    root: &Path,
    candidate_reference: &PrivateFileReference,
    candidate: &PureWeightCandidateReceipt,
    commitment: &TerminalChallengeCommitment,
) -> BrainResult<PrivateFileReference> {
    let expected = TerminalChallengeClaim {
        schema: TerminalChallengeSchema::Current,
        state: TerminalChallengeState::Claimed,
        commitment: candidate.challenge_commitment.clone(),
        commitment_sha256: commitment.manifest_sha256.clone(),
        suite_sha256: commitment.suite_sha256.clone(),
        candidate: candidate_reference.clone(),
        candidate_sha256: candidate.manifest_sha256.clone(),
    };
    verify_state(
        root,
        &claim_path(root, &commitment.manifest_sha256),
        &expected,
    )
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TerminalEvaluationSchema {
    #[serde(rename = "cerebro.tidex.terminal_evaluation/v2")]
    Current,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalExecutionBoundary {
    InProcessAuthenticatedWeightsOnlyNoSourceArgument,
    LinuxBubblewrapSealedMemfdAuthenticatedRunnerAndInputOnly,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalEvidenceScope {
    /// The cases were committed before candidate materialization, but this
    /// authority does not prove that an independent oracle produced them or
    /// that the candidate builder could not know them.
    PrecommittedCallerCasesNotIndependent,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalRejectionReason {
    ExpectedOutputMismatch,
    NonFiniteObservedOutput,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TerminalEvaluationOutcome {
    Passed,
    Rejected { reason: TerminalRejectionReason },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PureLinearExecutionPackageSchema {
    #[serde(rename = "cerebro.tidex.pure_linear_execution_package/v1")]
    Current,
}

/// Complete wire input for the native linear-map runner.
///
/// This deliberately has no path, source, capture/CAS reference, capability
/// identity, challenge identity, or expected output. The isolated payload can
/// compute the candidate's outputs, but cannot inspect the evidence authority
/// or decide whether those outputs pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PureLinearExecutionPackage {
    schema: PureLinearExecutionPackageSchema,
    input_dimension: u64,
    output_dimension: u64,
    weights: Vec<f64>,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PureLinearExecutionOutputSchema {
    #[serde(rename = "cerebro.tidex.pure_linear_execution_output/v1")]
    Current,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PureLinearExecutionOutput {
    schema: PureLinearExecutionOutputSchema,
    outputs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PureLinearRunnerAdmissionSchema {
    #[serde(rename = "cerebro.tidex.pure_linear_runner_admission/v1")]
    Current,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunnerAttestationScope {
    LocallyAdmittedNotIndependentlyAttested,
}

/// Private-root admission of one exact native runner artifact.
///
/// `claimed_source_tree_sha256` and `build_identity` are retained provenance
/// labels supplied at local admission time. They are committed but not
/// independently attested, and the receipt is neither a signature nor
/// promotion evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PureLinearRunnerAdmissionReceipt {
    schema: PureLinearRunnerAdmissionSchema,
    runner_program: PrivateFileReference,
    runner_program_sha256: Sha256Digest,
    backend_contract: String,
    claimed_source_tree_sha256: Option<Sha256Digest>,
    build_identity: String,
    attestation_scope: RunnerAttestationScope,
    manifest_sha256: Sha256Digest,
}

impl PureLinearRunnerAdmissionReceipt {
    pub fn runner_program_sha256(&self) -> &Sha256Digest {
        &self.runner_program_sha256
    }

    pub const fn attestation_scope(&self) -> RunnerAttestationScope {
        self.attestation_scope
    }
}

pub struct PureLinearRunnerAuthority {
    root: PathBuf,
}

impl PureLinearRunnerAuthority {
    pub fn open(root: &Path) -> BrainResult<Self> {
        Ok(Self {
            root: verify_internal_private_root(root)?,
        })
    }

    pub fn admit_local_build(
        &self,
        runner_bytes: Vec<u8>,
        expected_sha256: Sha256Digest,
        claimed_source_tree_sha256: Option<Sha256Digest>,
        build_identity: impl Into<String>,
    ) -> BrainResult<PrivateFileReference> {
        if Sha256Digest::digest_bytes(&runner_bytes) != expected_sha256 {
            return Err(integrity("pure_linear_runner_admission_digest_mismatch"));
        }
        validate_runner_bytes(&runner_bytes)?;
        let build_identity = build_identity.into();
        validate_runner_build_identity(&build_identity)?;
        if claimed_source_tree_sha256.as_ref() == Some(&Sha256Digest::zero()) {
            return Err(invalid("pure_linear_runner_source_tree_digest_invalid"));
        }
        let program_path = pure_linear_runner_program_path(&self.root, &expected_sha256);
        let observed_sha256 = write_or_verify_immutable(&self.root, &program_path, &runner_bytes)?;
        if observed_sha256 != expected_sha256 {
            return Err(integrity("pure_linear_runner_program_install_mismatch"));
        }
        let mut receipt = PureLinearRunnerAdmissionReceipt {
            schema: PureLinearRunnerAdmissionSchema::Current,
            runner_program: PrivateFileReference::new(program_path, observed_sha256.clone()),
            runner_program_sha256: observed_sha256,
            backend_contract: ISOLATED_BACKEND_CONTRACT.into(),
            claimed_source_tree_sha256,
            build_identity,
            attestation_scope: RunnerAttestationScope::LocallyAdmittedNotIndependentlyAttested,
            manifest_sha256: Sha256Digest::zero(),
        };
        receipt.manifest_sha256 = pure_linear_runner_admission_digest(&receipt)?;
        let reference = persist_canonical(
            &self.root,
            "state/pure_linear_runners/admissions/by-sha",
            &receipt.manifest_sha256,
            &receipt,
        )?;
        self.authenticate(&reference)?;
        Ok(reference)
    }

    pub fn authenticate(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<PureLinearRunnerAdmissionReceipt> {
        let receipt: PureLinearRunnerAdmissionReceipt = read_canonical(reference, &self.root)?;
        if receipt.schema != PureLinearRunnerAdmissionSchema::Current
            || receipt.runner_program_sha256 == Sha256Digest::zero()
            || receipt.runner_program.sha256 != receipt.runner_program_sha256
            || receipt.runner_program.path
                != pure_linear_runner_program_path(&self.root, &receipt.runner_program_sha256)
            || receipt.backend_contract != ISOLATED_BACKEND_CONTRACT
            || receipt.claimed_source_tree_sha256.as_ref() == Some(&Sha256Digest::zero())
            || receipt.attestation_scope
                != RunnerAttestationScope::LocallyAdmittedNotIndependentlyAttested
            || pure_linear_runner_admission_digest(&receipt)? != receipt.manifest_sha256
            || reference.path
                != canonical_path(
                    &self.root,
                    "state/pure_linear_runners/admissions/by-sha",
                    &receipt.manifest_sha256,
                )
        {
            return Err(integrity("pure_linear_runner_admission_invalid"));
        }
        validate_runner_build_identity(&receipt.build_identity)?;
        let bytes = receipt
            .runner_program
            .read_verified_bounded(&self.root, MAX_RECORD_BYTES)?;
        validate_runner_bytes(&bytes)?;
        Ok(receipt)
    }

    fn authenticated_program(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<(PureLinearRunnerAdmissionReceipt, AuthenticatedBytes)> {
        let receipt = self.authenticate(reference)?;
        let bytes = receipt
            .runner_program
            .read_verified_bounded(&self.root, MAX_RECORD_BYTES)?;
        let program =
            AuthenticatedBytes::authenticate(bytes, receipt.runner_program_sha256.clone())
                .map_err(|error| {
                    integrity(&format!("pure_linear_runner_authentication_failed:{error}"))
                })?;
        Ok((receipt, program))
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }
}

fn validate_runner_bytes(bytes: &[u8]) -> BrainResult<()> {
    if bytes.len() < 4
        || u64::try_from(bytes.len()).map_err(|_| invalid("pure_linear_runner_size_overflow"))?
            > MAX_RECORD_BYTES
        || &bytes[..4] != b"\x7fELF"
    {
        return Err(invalid("pure_linear_runner_program_invalid"));
    }
    Ok(())
}

fn validate_runner_build_identity(value: &str) -> BrainResult<()> {
    if value.is_empty()
        || value.len() > MAX_RUNNER_BUILD_IDENTITY_BYTES
        || !value.bytes().all(|byte| (0x20..=0x7e).contains(&byte))
    {
        return Err(invalid("pure_linear_runner_build_identity_invalid"));
    }
    Ok(())
}

fn pure_linear_runner_admission_digest(
    receipt: &PureLinearRunnerAdmissionReceipt,
) -> BrainResult<Sha256Digest> {
    let mut unsigned = receipt.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    Ok(Sha256Digest::digest_domain(
        PURE_LINEAR_RUNNER_ADMISSION_DOMAIN,
        &canonical_json_bounded(
            &unsigned,
            MAX_RECORD_BYTES,
            "pure_linear_runner_admission_record_too_large",
        )?,
    ))
}

impl PureLinearExecutionPackage {
    fn from_authenticated_chain(
        candidate: &PureWeightCandidateReceipt,
        reveal: &TerminalChallengeReveal,
    ) -> BrainResult<Self> {
        validate_terminal_evaluation_inputs(candidate, reveal)?;
        let package = Self {
            schema: PureLinearExecutionPackageSchema::Current,
            input_dimension: candidate.input_dimension,
            output_dimension: candidate.output_dimension,
            weights: candidate.weights.clone(),
            inputs: reveal.cases.iter().map(|case| case.input.clone()).collect(),
        };
        package.validate()?;
        Ok(package)
    }

    fn validate(&self) -> BrainResult<(usize, usize)> {
        let input = usize::try_from(self.input_dimension)
            .map_err(|_| invalid("isolated_package_input_dimension_overflow"))?;
        let output = usize::try_from(self.output_dimension)
            .map_err(|_| invalid("isolated_package_output_dimension_overflow"))?;
        let weight_count = input
            .checked_mul(output)
            .ok_or_else(|| invalid("isolated_package_weight_shape_overflow"))?;
        if self.schema != PureLinearExecutionPackageSchema::Current
            || input == 0
            || output == 0
            || input > MAX_DIMENSION
            || output > MAX_DIMENSION
            || weight_count > MAX_WEIGHTS
            || self.weights.len() != weight_count
            || self.weights.iter().any(|value| !value.is_finite())
            || self.inputs.is_empty()
            || self.inputs.len() > MAX_TERMINAL_CASES
            || self.inputs.iter().any(|values| {
                values.len() != input || values.iter().any(|value| !value.is_finite())
            })
        {
            return Err(invalid("isolated_linear_execution_package_invalid"));
        }
        validate_terminal_resource_shape(self.inputs.len(), input, output)?;
        Ok((input, output))
    }

    fn execute(&self) -> BrainResult<PureLinearExecutionOutput> {
        let (input, output) = self.validate()?;
        let outputs = self
            .inputs
            .iter()
            .map(|values| {
                (0..output)
                    .map(|row| {
                        (0..input)
                            .map(|column| self.weights[row * input + column] * values[column])
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if outputs.iter().flatten().any(|value| !value.is_finite()) {
            return Err(integrity("isolated_linear_execution_nonfinite_output"));
        }
        Ok(PureLinearExecutionOutput {
            schema: PureLinearExecutionOutputSchema::Current,
            outputs,
        })
    }

    fn to_canonical_bytes(&self) -> BrainResult<Vec<u8>> {
        self.validate()?;
        canonical_json_bounded(
            self,
            MAX_RECORD_BYTES,
            "isolated_linear_execution_package_too_large",
        )
    }

    fn from_canonical_bytes(bytes: &[u8]) -> BrainResult<Self> {
        if u64::try_from(bytes.len())
            .map_err(|_| invalid("isolated_linear_execution_package_length_overflow"))?
            > MAX_RECORD_BYTES
        {
            return Err(invalid("isolated_linear_execution_package_too_large"));
        }
        let package: Self = serde_json::from_slice(bytes)?;
        package.validate()?;
        if package.to_canonical_bytes()? != bytes {
            return Err(integrity("isolated_linear_execution_package_noncanonical"));
        }
        Ok(package)
    }
}

impl PureLinearExecutionOutput {
    fn to_canonical_bytes(&self) -> BrainResult<Vec<u8>> {
        if self.schema != PureLinearExecutionOutputSchema::Current
            || self.outputs.is_empty()
            || self.outputs.iter().any(|values| values.is_empty())
            || self
                .outputs
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(invalid("isolated_linear_execution_output_invalid"));
        }
        canonical_json_bounded(
            self,
            MAX_RECORD_BYTES,
            "isolated_linear_execution_output_too_large",
        )
    }
}

/// Native runner entrypoint used by `pure_linear_runner`.
///
/// The path is fixed by the isolated-execution backend. Refusing any other
/// value prevents this binary from becoming a general host file reader if it
/// is accidentally invoked outside that boundary.
pub fn run_pure_linear_runner() -> BrainResult<()> {
    let input_path =
        std::env::var_os("TIDEX_INPUT_PATH").ok_or_else(|| invalid("tidex_input_path_missing"))?;
    if Path::new(&input_path) != Path::new(RUNNER_INPUT_PATH) {
        return Err(invalid("tidex_input_path_invalid"));
    }
    let file = std::fs::File::open(&input_path)?;
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_RECORD_BYTES {
        return Err(invalid("isolated_linear_execution_input_file_invalid"));
    }
    let read_limit = MAX_RECORD_BYTES
        .checked_add(1)
        .ok_or_else(|| invalid("isolated_linear_execution_read_limit_overflow"))?;
    let mut bytes = Vec::new();
    file.take(read_limit).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())
        .map_err(|_| invalid("isolated_linear_execution_package_length_overflow"))?
        > MAX_RECORD_BYTES
    {
        return Err(invalid("isolated_linear_execution_package_too_large"));
    }
    let output = PureLinearExecutionPackage::from_canonical_bytes(&bytes)?
        .execute()?
        .to_canonical_bytes()?;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    lock.write_all(&output)?;
    lock.flush()?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TerminalRevealTransition {
    schema: TerminalChallengeSchema,
    state: TerminalChallengeState,
    commitment: PrivateFileReference,
    commitment_sha256: Sha256Digest,
    suite_sha256: Sha256Digest,
    candidate: PrivateFileReference,
    candidate_sha256: Sha256Digest,
    claim: PrivateFileReference,
    reveal: PrivateFileReference,
    reveal_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TerminalTransition {
    schema: TerminalChallengeSchema,
    state: TerminalChallengeState,
    commitment_sha256: Sha256Digest,
    suite_sha256: Sha256Digest,
    candidate_sha256: Sha256Digest,
    reveal_sha256: Sha256Digest,
    evaluation: PrivateFileReference,
    evaluation_manifest_sha256: Sha256Digest,
    outcome: TerminalEvaluationOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TerminalEvaluationReceipt {
    schema: TerminalEvaluationSchema,
    state: TerminalChallengeState,
    candidate: PrivateFileReference,
    candidate_sha256: Sha256Digest,
    commitment: PrivateFileReference,
    commitment_sha256: Sha256Digest,
    suite_sha256: Sha256Digest,
    claim: PrivateFileReference,
    reveal: PrivateFileReference,
    reveal_sha256: Sha256Digest,
    reveal_transition: PrivateFileReference,
    case_count: u64,
    observed_outputs_sha256: Sha256Digest,
    outcome: TerminalEvaluationOutcome,
    execution_boundary: TerminalExecutionBoundary,
    evidence_scope: TerminalEvidenceScope,
    manifest_sha256: Sha256Digest,
}

impl TerminalEvaluationReceipt {
    pub const fn outcome(&self) -> TerminalEvaluationOutcome {
        self.outcome
    }
}

pub struct TerminalEvaluationAuthority {
    root: PathBuf,
}

impl TerminalEvaluationAuthority {
    pub fn open(root: &Path) -> BrainResult<Self> {
        Ok(Self {
            root: verify_internal_private_root(root)?,
        })
    }

    /// Reveal is persisted only after an authenticated candidate already binds
    /// the prior commitment.
    pub fn persist_reveal_after_candidate(
        &self,
        candidate_reference: &PrivateFileReference,
        reveal: &TerminalChallengeReveal,
    ) -> BrainResult<PrivateFileReference> {
        let candidate =
            PureWeightCandidateAuthority::open(&self.root)?.authenticate(candidate_reference)?;
        let commitment = authenticate_commitment(&self.root, &candidate.challenge_commitment)?;
        let claim = authenticate_claim(&self.root, candidate_reference, &candidate, &commitment)?;
        reveal.validate()?;
        if reveal.challenge_id != commitment.challenge_id
            || reveal.suite_sha256 != commitment.suite_sha256
            || reveal.reveal_sha256 != commitment.reveal_sha256
        {
            return Err(integrity("terminal_reveal_commitment_mismatch"));
        }
        validate_terminal_evaluation_inputs(&candidate, reveal)?;
        let reveal_reference = persist_canonical(
            &self.root,
            "state/terminal_challenges/reveals/by-sha",
            &reveal.reveal_sha256,
            reveal,
        )?;
        let transition = TerminalRevealTransition {
            schema: TerminalChallengeSchema::Current,
            state: TerminalChallengeState::Revealed,
            commitment: candidate.challenge_commitment,
            commitment_sha256: commitment.manifest_sha256.clone(),
            suite_sha256: commitment.suite_sha256,
            candidate: candidate_reference.clone(),
            candidate_sha256: candidate.manifest_sha256,
            claim,
            reveal: reveal_reference.clone(),
            reveal_sha256: reveal.reveal_sha256.clone(),
        };
        write_or_verify_state(
            &self.root,
            &reveal_transition_path(&self.root, &commitment.manifest_sha256),
            &transition,
        )?;
        Ok(reveal_reference)
    }

    pub fn evaluate(
        &self,
        candidate_reference: PrivateFileReference,
        reveal_reference: PrivateFileReference,
    ) -> BrainResult<PrivateFileReference> {
        let candidate =
            PureWeightCandidateAuthority::open(&self.root)?.authenticate(&candidate_reference)?;
        let commitment = authenticate_commitment(&self.root, &candidate.challenge_commitment)?;
        let claim = authenticate_claim(&self.root, &candidate_reference, &candidate, &commitment)?;
        let reveal: TerminalChallengeReveal = read_canonical(&reveal_reference, &self.root)?;
        reveal.validate()?;
        if reveal_reference.path
            != canonical_path(
                &self.root,
                "state/terminal_challenges/reveals/by-sha",
                &reveal.reveal_sha256,
            )
            || reveal.challenge_id != commitment.challenge_id
            || reveal.suite_sha256 != commitment.suite_sha256
            || reveal.reveal_sha256 != commitment.reveal_sha256
        {
            return Err(integrity("terminal_reveal_commitment_mismatch"));
        }
        let reveal_transition = authenticate_reveal_transition(
            &self.root,
            &candidate_reference,
            &candidate,
            &commitment,
            &claim,
            &reveal_reference,
            &reveal,
        )?;

        // The execution primitive receives only resident weights, dimensions,
        // and terminal inputs. It has no source/capture/projection argument.
        let observed = execute_weight_candidate(&candidate, &reveal)?;
        let outcome = terminal_outcome(&observed, &reveal);
        let observed_outputs_sha256 = output_digest(&observed)?;

        let mut receipt = TerminalEvaluationReceipt {
            schema: TerminalEvaluationSchema::Current,
            state: TerminalChallengeState::Terminal,
            candidate: candidate_reference,
            candidate_sha256: candidate.manifest_sha256.clone(),
            commitment: candidate.challenge_commitment,
            commitment_sha256: commitment.manifest_sha256.clone(),
            suite_sha256: commitment.suite_sha256.clone(),
            claim,
            reveal: reveal_reference,
            reveal_sha256: reveal.reveal_sha256.clone(),
            reveal_transition,
            case_count: u64::try_from(reveal.cases.len())
                .map_err(|_| invalid("terminal_case_count_overflow"))?,
            observed_outputs_sha256,
            outcome,
            execution_boundary:
                TerminalExecutionBoundary::InProcessAuthenticatedWeightsOnlyNoSourceArgument,
            evidence_scope: TerminalEvidenceScope::PrecommittedCallerCasesNotIndependent,
            manifest_sha256: Sha256Digest::zero(),
        };
        receipt.manifest_sha256 = evaluation_digest(&receipt)?;
        let reference = persist_canonical(
            &self.root,
            "state/terminal_evaluations/by-sha",
            &receipt.manifest_sha256,
            &receipt,
        )?;
        let terminal = TerminalTransition {
            schema: TerminalChallengeSchema::Current,
            state: TerminalChallengeState::Terminal,
            commitment_sha256: commitment.manifest_sha256.clone(),
            suite_sha256: commitment.suite_sha256,
            candidate_sha256: candidate.manifest_sha256,
            reveal_sha256: reveal.reveal_sha256,
            evaluation: reference.clone(),
            evaluation_manifest_sha256: receipt.manifest_sha256,
            outcome,
        };
        write_or_verify_state(
            &self.root,
            &terminal_transition_path(&self.root, &commitment.manifest_sha256),
            &terminal,
        )?;
        self.authenticate(&reference)?;
        Ok(reference)
    }

    /// Recompute the complete terminal result. Neither candidate-supplied
    /// metrics nor a candidate-supplied verdict exist in this protocol.
    pub fn authenticate(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<TerminalEvaluationReceipt> {
        let receipt: TerminalEvaluationReceipt = read_canonical(reference, &self.root)?;
        if receipt.schema != TerminalEvaluationSchema::Current
            || receipt.state != TerminalChallengeState::Terminal
            || receipt.execution_boundary
                != TerminalExecutionBoundary::InProcessAuthenticatedWeightsOnlyNoSourceArgument
            || receipt.evidence_scope
                != TerminalEvidenceScope::PrecommittedCallerCasesNotIndependent
            || evaluation_digest(&receipt)? != receipt.manifest_sha256
            || reference.path
                != canonical_path(
                    &self.root,
                    "state/terminal_evaluations/by-sha",
                    &receipt.manifest_sha256,
                )
        {
            return Err(integrity("terminal_evaluation_receipt_invalid"));
        }
        let candidate =
            PureWeightCandidateAuthority::open(&self.root)?.authenticate(&receipt.candidate)?;
        let commitment = authenticate_commitment(&self.root, &receipt.commitment)?;
        let claim = authenticate_claim(&self.root, &receipt.candidate, &candidate, &commitment)?;
        let reveal: TerminalChallengeReveal = read_canonical(&receipt.reveal, &self.root)?;
        reveal.validate()?;
        if receipt.candidate_sha256 != candidate.manifest_sha256
            || receipt.commitment != candidate.challenge_commitment
            || receipt.commitment_sha256 != commitment.manifest_sha256
            || receipt.suite_sha256 != commitment.suite_sha256
            || receipt.claim != claim
            || receipt.reveal_sha256 != reveal.reveal_sha256
            || receipt.reveal.path
                != canonical_path(
                    &self.root,
                    "state/terminal_challenges/reveals/by-sha",
                    &reveal.reveal_sha256,
                )
            || reveal.challenge_id != commitment.challenge_id
            || reveal.suite_sha256 != commitment.suite_sha256
            || reveal.reveal_sha256 != commitment.reveal_sha256
            || receipt.case_count != reveal.cases.len() as u64
        {
            return Err(integrity("terminal_evaluation_chain_mismatch"));
        }
        let reveal_transition = authenticate_reveal_transition(
            &self.root,
            &receipt.candidate,
            &candidate,
            &commitment,
            &claim,
            &receipt.reveal,
            &reveal,
        )?;
        if receipt.reveal_transition != reveal_transition {
            return Err(integrity("terminal_evaluation_reveal_state_mismatch"));
        }
        let observed = execute_weight_candidate(&candidate, &reveal)?;
        let outcome = terminal_outcome(&observed, &reveal);
        if outcome != receipt.outcome
            || output_digest(&observed)? != receipt.observed_outputs_sha256
        {
            return Err(integrity("terminal_evaluation_recomputation_mismatch"));
        }
        let terminal = TerminalTransition {
            schema: TerminalChallengeSchema::Current,
            state: TerminalChallengeState::Terminal,
            commitment_sha256: commitment.manifest_sha256.clone(),
            suite_sha256: commitment.suite_sha256,
            candidate_sha256: candidate.manifest_sha256,
            reveal_sha256: reveal.reveal_sha256,
            evaluation: reference.clone(),
            evaluation_manifest_sha256: receipt.manifest_sha256.clone(),
            outcome,
        };
        verify_state(
            &self.root,
            &terminal_transition_path(&self.root, &commitment.manifest_sha256),
            &terminal,
        )?;
        Ok(receipt)
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IsolatedTerminalEvaluationSchema {
    #[serde(rename = "cerebro.tidex.isolated_terminal_evaluation/v1")]
    Current,
}

/// Authenticated observation from the separate OS-isolated evaluation path.
///
/// It remains non-independent terminal evidence and never grants promotion.
/// The runner and input identities, the isolated backend request identity, and
/// the exact canonical stdout identity are all committed by this receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IsolatedTerminalEvaluationReceipt {
    schema: IsolatedTerminalEvaluationSchema,
    state: TerminalChallengeState,
    candidate: PrivateFileReference,
    candidate_sha256: Sha256Digest,
    commitment: PrivateFileReference,
    commitment_sha256: Sha256Digest,
    suite_sha256: Sha256Digest,
    claim: PrivateFileReference,
    reveal: PrivateFileReference,
    reveal_sha256: Sha256Digest,
    reveal_transition: PrivateFileReference,
    case_count: u64,
    runner_admission: PrivateFileReference,
    runner_admission_manifest_sha256: Sha256Digest,
    runner_sha256: Sha256Digest,
    execution_input_sha256: Sha256Digest,
    request_digest: Sha256Digest,
    backend: String,
    canonical_stdout_sha256: Sha256Digest,
    observed_outputs_sha256: Sha256Digest,
    outcome: TerminalEvaluationOutcome,
    execution_boundary: TerminalExecutionBoundary,
    evidence_scope: TerminalEvidenceScope,
    manifest_sha256: Sha256Digest,
}

impl IsolatedTerminalEvaluationReceipt {
    pub const fn outcome(&self) -> TerminalEvaluationOutcome {
        self.outcome
    }

    pub fn request_digest(&self) -> &Sha256Digest {
        &self.request_digest
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub const fn execution_boundary(&self) -> TerminalExecutionBoundary {
        self.execution_boundary
    }
}

/// A migration-safe OS-isolated alternative to `TerminalEvaluationAuthority`.
/// It neither replaces nor silently falls back to the in-process path.
pub struct IsolatedTerminalEvaluationAuthority {
    root: PathBuf,
}

impl IsolatedTerminalEvaluationAuthority {
    pub fn open(root: &Path) -> BrainResult<Self> {
        Ok(Self {
            root: verify_internal_private_root(root)?,
        })
    }

    pub fn evaluate(
        &self,
        candidate_reference: PrivateFileReference,
        reveal_reference: PrivateFileReference,
        runner_admission_reference: PrivateFileReference,
        limits: IsolationLimits,
    ) -> BrainResult<PrivateFileReference> {
        // Authenticate the entire authority chain before constructing either
        // payload passed to the less-trusted isolated process.
        let candidate = PureWeightCandidateAuthority::open(&self.root)?
            .authenticate_sealed(&candidate_reference)?;
        let commitment = authenticate_commitment(&self.root, &candidate.challenge_commitment)?;
        let claim = authenticate_claim(&self.root, &candidate_reference, &candidate, &commitment)?;
        let reveal: TerminalChallengeReveal = read_canonical(&reveal_reference, &self.root)?;
        reveal.validate()?;
        if reveal_reference.path
            != canonical_path(
                &self.root,
                "state/terminal_challenges/reveals/by-sha",
                &reveal.reveal_sha256,
            )
            || reveal.challenge_id != commitment.challenge_id
            || reveal.suite_sha256 != commitment.suite_sha256
            || reveal.reveal_sha256 != commitment.reveal_sha256
        {
            return Err(integrity("terminal_reveal_commitment_mismatch"));
        }
        let reveal_transition = authenticate_reveal_transition(
            &self.root,
            &candidate_reference,
            &candidate,
            &commitment,
            &claim,
            &reveal_reference,
            &reveal,
        )?;
        let (runner_admission, runner) = PureLinearRunnerAuthority::open(&self.root)?
            .authenticated_program(&runner_admission_reference)?;

        let package = PureLinearExecutionPackage::from_authenticated_chain(&candidate, &reveal)?;
        let input_bytes = package.to_canonical_bytes()?;
        let expected_output = package.execute()?;
        let expected_stdout = expected_output.to_canonical_bytes()?;
        let runner_sha256 = runner.digest().clone();
        let input = AuthenticatedBytes::from_trusted_bytes(input_bytes);
        let execution_input_sha256 = input.digest().clone();
        let report = run_isolated(&IsolatedExecutionRequest {
            program: runner,
            input,
            arguments: Vec::new(),
            limits,
            requirements: IsolationRequirements {
                require_seccomp_filter: false,
                require_cgroup_limits: false,
                require_global_staging_admission: false,
            },
        })
        .map_err(|error| integrity(&format!("isolated_linear_execution_failed:{error}")))?;

        if !report.succeeded()
            || report.exit_code != Some(0)
            || report.payload_execution_evidence
                != PayloadExecutionEvidence::EstablishedBySuccessfulExit
            || report.program_digest != runner_sha256
            || report.input_digest != execution_input_sha256
            || report.stdout_truncated
            || report.stderr_truncated
            || !report.stderr.is_empty()
            || report.stdout != expected_stdout
            || report.backend != ISOLATED_BACKEND_CONTRACT
        {
            return Err(integrity(&format!(
                "isolated_linear_execution_report_rejected:{}",
                bounded_stderr_detail(&report)
            )));
        }

        let observed = expected_output.outputs;
        let outcome = terminal_outcome(&observed, &reveal);
        let mut receipt = IsolatedTerminalEvaluationReceipt {
            schema: IsolatedTerminalEvaluationSchema::Current,
            state: TerminalChallengeState::Terminal,
            candidate: candidate_reference,
            candidate_sha256: candidate.manifest_sha256.clone(),
            commitment: candidate.challenge_commitment,
            commitment_sha256: commitment.manifest_sha256.clone(),
            suite_sha256: commitment.suite_sha256.clone(),
            claim,
            reveal: reveal_reference,
            reveal_sha256: reveal.reveal_sha256.clone(),
            reveal_transition,
            case_count: u64::try_from(reveal.cases.len())
                .map_err(|_| invalid("terminal_case_count_overflow"))?,
            runner_admission: runner_admission_reference,
            runner_admission_manifest_sha256: runner_admission.manifest_sha256,
            runner_sha256,
            execution_input_sha256,
            request_digest: report.request_digest,
            backend: report.backend,
            canonical_stdout_sha256: Sha256Digest::digest_bytes(&expected_stdout),
            observed_outputs_sha256: output_digest(&observed)?,
            outcome,
            execution_boundary:
                TerminalExecutionBoundary::LinuxBubblewrapSealedMemfdAuthenticatedRunnerAndInputOnly,
            evidence_scope: TerminalEvidenceScope::PrecommittedCallerCasesNotIndependent,
            manifest_sha256: Sha256Digest::zero(),
        };
        receipt.manifest_sha256 = isolated_evaluation_digest(&receipt)?;
        let reference = persist_canonical(
            &self.root,
            "state/isolated_terminal_evaluations/by-sha",
            &receipt.manifest_sha256,
            &receipt,
        )?;
        let terminal = TerminalTransition {
            schema: TerminalChallengeSchema::Current,
            state: TerminalChallengeState::Terminal,
            commitment_sha256: commitment.manifest_sha256.clone(),
            suite_sha256: commitment.suite_sha256,
            candidate_sha256: candidate.manifest_sha256,
            reveal_sha256: reveal.reveal_sha256,
            evaluation: reference.clone(),
            evaluation_manifest_sha256: receipt.manifest_sha256,
            outcome,
        };
        write_or_verify_state(
            &self.root,
            &terminal_transition_path(&self.root, &commitment.manifest_sha256),
            &terminal,
        )?;
        self.authenticate(&reference)?;
        Ok(reference)
    }

    pub fn authenticate(
        &self,
        reference: &PrivateFileReference,
    ) -> BrainResult<IsolatedTerminalEvaluationReceipt> {
        let receipt: IsolatedTerminalEvaluationReceipt = read_canonical(reference, &self.root)?;
        if receipt.schema != IsolatedTerminalEvaluationSchema::Current
            || receipt.state != TerminalChallengeState::Terminal
            || receipt.execution_boundary
                != TerminalExecutionBoundary::LinuxBubblewrapSealedMemfdAuthenticatedRunnerAndInputOnly
            || receipt.evidence_scope
                != TerminalEvidenceScope::PrecommittedCallerCasesNotIndependent
            || receipt.runner_sha256 == Sha256Digest::zero()
            || receipt.execution_input_sha256 == Sha256Digest::zero()
            || receipt.request_digest == Sha256Digest::zero()
            || receipt.backend != ISOLATED_BACKEND_CONTRACT
            || isolated_evaluation_digest(&receipt)? != receipt.manifest_sha256
            || reference.path
                != canonical_path(
                    &self.root,
                    "state/isolated_terminal_evaluations/by-sha",
                    &receipt.manifest_sha256,
                )
        {
            return Err(integrity("isolated_terminal_evaluation_receipt_invalid"));
        }
        let candidate = PureWeightCandidateAuthority::open(&self.root)?
            .authenticate_sealed(&receipt.candidate)?;
        let runner_admission =
            PureLinearRunnerAuthority::open(&self.root)?.authenticate(&receipt.runner_admission)?;
        let commitment = authenticate_commitment(&self.root, &receipt.commitment)?;
        let claim = authenticate_claim(&self.root, &receipt.candidate, &candidate, &commitment)?;
        let reveal: TerminalChallengeReveal = read_canonical(&receipt.reveal, &self.root)?;
        reveal.validate()?;
        if receipt.runner_admission_manifest_sha256 != runner_admission.manifest_sha256
            || receipt.runner_sha256 != runner_admission.runner_program_sha256
            || receipt.candidate_sha256 != candidate.manifest_sha256
            || receipt.commitment != candidate.challenge_commitment
            || receipt.commitment_sha256 != commitment.manifest_sha256
            || receipt.suite_sha256 != commitment.suite_sha256
            || receipt.claim != claim
            || receipt.reveal_sha256 != reveal.reveal_sha256
            || receipt.reveal.path
                != canonical_path(
                    &self.root,
                    "state/terminal_challenges/reveals/by-sha",
                    &reveal.reveal_sha256,
                )
            || reveal.challenge_id != commitment.challenge_id
            || reveal.suite_sha256 != commitment.suite_sha256
            || reveal.reveal_sha256 != commitment.reveal_sha256
            || receipt.case_count != reveal.cases.len() as u64
        {
            return Err(integrity("isolated_terminal_evaluation_chain_mismatch"));
        }
        let reveal_transition = authenticate_reveal_transition(
            &self.root,
            &receipt.candidate,
            &candidate,
            &commitment,
            &claim,
            &receipt.reveal,
            &reveal,
        )?;
        if receipt.reveal_transition != reveal_transition {
            return Err(integrity(
                "isolated_terminal_evaluation_reveal_state_mismatch",
            ));
        }
        let package = PureLinearExecutionPackage::from_authenticated_chain(&candidate, &reveal)?;
        let expected_output = package.execute()?;
        let expected_stdout = expected_output.to_canonical_bytes()?;
        if receipt.execution_input_sha256
            != Sha256Digest::digest_bytes(&package.to_canonical_bytes()?)
            || receipt.canonical_stdout_sha256 != Sha256Digest::digest_bytes(&expected_stdout)
            || receipt.observed_outputs_sha256 != output_digest(&expected_output.outputs)?
            || receipt.outcome != terminal_outcome(&expected_output.outputs, &reveal)
        {
            return Err(integrity(
                "isolated_terminal_evaluation_recomputation_mismatch",
            ));
        }
        let terminal = TerminalTransition {
            schema: TerminalChallengeSchema::Current,
            state: TerminalChallengeState::Terminal,
            commitment_sha256: commitment.manifest_sha256.clone(),
            suite_sha256: commitment.suite_sha256,
            candidate_sha256: candidate.manifest_sha256,
            reveal_sha256: reveal.reveal_sha256,
            evaluation: reference.clone(),
            evaluation_manifest_sha256: receipt.manifest_sha256.clone(),
            outcome: receipt.outcome,
        };
        verify_state(
            &self.root,
            &terminal_transition_path(&self.root, &commitment.manifest_sha256),
            &terminal,
        )?;
        Ok(receipt)
    }

    pub const fn authorizes_promotion(&self) -> bool {
        false
    }
}

fn isolated_evaluation_digest(
    receipt: &IsolatedTerminalEvaluationReceipt,
) -> BrainResult<Sha256Digest> {
    let mut unsigned = receipt.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    Ok(Sha256Digest::digest_domain(
        ISOLATED_EVALUATION_DOMAIN,
        &canonical_json_bounded(
            &unsigned,
            MAX_RECORD_BYTES,
            "isolated_terminal_evaluation_record_too_large",
        )?,
    ))
}

fn authenticate_reveal_transition(
    root: &Path,
    candidate_reference: &PrivateFileReference,
    candidate: &PureWeightCandidateReceipt,
    commitment: &TerminalChallengeCommitment,
    claim: &PrivateFileReference,
    reveal_reference: &PrivateFileReference,
    reveal: &TerminalChallengeReveal,
) -> BrainResult<PrivateFileReference> {
    let expected = TerminalRevealTransition {
        schema: TerminalChallengeSchema::Current,
        state: TerminalChallengeState::Revealed,
        commitment: candidate.challenge_commitment.clone(),
        commitment_sha256: commitment.manifest_sha256.clone(),
        suite_sha256: commitment.suite_sha256.clone(),
        candidate: candidate_reference.clone(),
        candidate_sha256: candidate.manifest_sha256.clone(),
        claim: claim.clone(),
        reveal: reveal_reference.clone(),
        reveal_sha256: reveal.reveal_sha256.clone(),
    };
    verify_state(
        root,
        &reveal_transition_path(root, &commitment.manifest_sha256),
        &expected,
    )
}

fn terminal_outcome(
    observed: &[Vec<f64>],
    reveal: &TerminalChallengeReveal,
) -> TerminalEvaluationOutcome {
    if observed.iter().flatten().any(|value| !value.is_finite()) {
        return TerminalEvaluationOutcome::Rejected {
            reason: TerminalRejectionReason::NonFiniteObservedOutput,
        };
    }
    if observed
        .iter()
        .zip(&reveal.cases)
        .any(|(actual, case)| !bitwise_f64_slices_equal(actual, &case.expected_output))
    {
        TerminalEvaluationOutcome::Rejected {
            reason: TerminalRejectionReason::ExpectedOutputMismatch,
        }
    } else {
        TerminalEvaluationOutcome::Passed
    }
}

fn validate_terminal_evaluation_inputs(
    candidate: &PureWeightCandidateReceipt,
    reveal: &TerminalChallengeReveal,
) -> BrainResult<(usize, usize)> {
    let input = usize::try_from(candidate.input_dimension)
        .map_err(|_| invalid("candidate_dimension_overflow"))?;
    let output = usize::try_from(candidate.output_dimension)
        .map_err(|_| invalid("candidate_dimension_overflow"))?;
    let weight_count = input
        .checked_mul(output)
        .ok_or_else(|| invalid("candidate_weight_shape_overflow"))?;
    if input == 0
        || output == 0
        || candidate.weights.len() != weight_count
        || candidate.weights.iter().any(|value| !value.is_finite())
    {
        return Err(integrity("candidate_weight_shape_invalid"));
    }
    validate_terminal_resource_shape(reveal.cases.len(), input, output)?;
    if reveal
        .cases
        .iter()
        .any(|case| case.input.len() != input || case.expected_output.len() != output)
    {
        return Err(invalid("terminal_case_shape_mismatch"));
    }
    Ok((input, output))
}

fn execute_weight_candidate(
    candidate: &PureWeightCandidateReceipt,
    reveal: &TerminalChallengeReveal,
) -> BrainResult<Vec<Vec<f64>>> {
    let (input, output) = validate_terminal_evaluation_inputs(candidate, reveal)?;
    Ok(reveal
        .cases
        .iter()
        .map(|case| {
            (0..output)
                .map(|row| {
                    (0..input)
                        .map(|column| candidate.weights[row * input + column] * case.input[column])
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

fn validate_terminal_resource_shape(
    case_count: usize,
    input: usize,
    output: usize,
) -> BrainResult<()> {
    let terminal_scalars = case_count
        .checked_mul(
            input
                .checked_add(output)
                .ok_or_else(|| invalid("terminal_case_scalar_overflow"))?,
        )
        .ok_or_else(|| invalid("terminal_case_scalar_overflow"))?;
    if terminal_scalars > MAX_TERMINAL_SCALARS {
        return Err(invalid("terminal_challenge_scalar_limit"));
    }
    let work = u128::try_from(case_count)
        .ok()
        .and_then(|cases| cases.checked_mul(u128::try_from(input).ok()?))
        .and_then(|value| value.checked_mul(u128::try_from(output).ok()?))
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| invalid("terminal_evaluation_work_overflow"))?;
    if work > MAX_TERMINAL_WORK_UNITS {
        return Err(invalid("terminal_evaluation_work_limit"));
    }
    Ok(())
}

fn output_digest(outputs: &[Vec<f64>]) -> BrainResult<Sha256Digest> {
    let mut hasher = Sha256::new();
    hasher.update(OUTPUT_DOMAIN);
    for value in outputs.iter().flatten() {
        hasher.update(value.to_bits().to_be_bytes());
    }
    finish_sha256(hasher)
}

fn evaluation_digest(receipt: &TerminalEvaluationReceipt) -> BrainResult<Sha256Digest> {
    let mut unsigned = receipt.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    Ok(Sha256Digest::digest_domain(
        EVALUATION_DOMAIN,
        &canonical_json_bounded(
            &unsigned,
            MAX_RECORD_BYTES,
            "terminal_evaluation_record_too_large",
        )?,
    ))
}

fn descriptor_for_discovery(
    root: &Path,
    capture: &CaptureReceipt,
    discovery: &PureCapabilityDiscoveryReceipt,
) -> BrainResult<LinearMapDescriptor> {
    let source = retained_source_adapter(root, capture)?;
    let projection = source.materialize()?;
    let path = existing_regular_file_under_root(
        &projection,
        &projection.join(&discovery.descriptor_relative_path),
    )?;
    let reference = PrivateFileReference::new(path, discovery.descriptor_content_sha256.clone());
    let bytes = reference.read_verified_bounded(root, MAX_DESCRIPTOR_BYTES)?;
    let descriptor: LinearMapDescriptor = serde_json::from_slice(&bytes)?;
    descriptor.validate()?;
    source.verify_materialized(&projection)?;
    Ok(descriptor)
}

fn verify_source_binding(
    capture: &CaptureReceipt,
    source: &RetainedSourceAdapter,
) -> BrainResult<()> {
    if capture.manifest_sha256() != source.capture_receipt_sha256()
        || capture.system_envelope_sha256() != source.system_envelope_sha256()
    {
        return Err(integrity("retained_source_capture_mismatch"));
    }
    Ok(())
}

fn normal_relative_path(path: &Path) -> BrainResult<PathBuf> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.as_os_str().as_encoded_bytes().len() > 4_096
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid("descriptor_relative_path_invalid"));
    }
    Ok(path.to_path_buf())
}

fn canonical_path(root: &Path, directory: &str, digest: &Sha256Digest) -> PathBuf {
    root.join(directory).join(format!("{}.json", digest))
}

fn pure_linear_runner_program_path(root: &Path, digest: &Sha256Digest) -> PathBuf {
    root.join("state/pure_linear_runners/programs/by-sha")
        .join(format!("{}.elf", digest))
}

fn suite_registration_path(root: &Path, suite: &Sha256Digest) -> PathBuf {
    canonical_path(root, "state/terminal_challenges/suites/by-sha", suite)
}

fn claim_path(root: &Path, commitment: &Sha256Digest) -> PathBuf {
    canonical_path(
        root,
        "state/terminal_challenges/claims/by-commitment",
        commitment,
    )
}

fn reveal_transition_path(root: &Path, commitment: &Sha256Digest) -> PathBuf {
    canonical_path(
        root,
        "state/terminal_challenges/revealed/by-commitment",
        commitment,
    )
}

fn terminal_transition_path(root: &Path, commitment: &Sha256Digest) -> PathBuf {
    canonical_path(
        root,
        "state/terminal_challenges/terminal/by-commitment",
        commitment,
    )
}

fn write_or_verify_state<T: Serialize>(
    root: &Path,
    path: &Path,
    value: &T,
) -> BrainResult<PrivateFileReference> {
    let bytes = canonical_json_bounded(value, MAX_RECORD_BYTES, "terminal_state_too_large")?;
    let sha256 = write_or_verify_immutable(root, path, &bytes)?;
    Ok(PrivateFileReference::new(path, sha256))
}

fn verify_state<T: Serialize>(
    root: &Path,
    path: &Path,
    expected: &T,
) -> BrainResult<PrivateFileReference> {
    let expected = canonical_json_bounded(expected, MAX_RECORD_BYTES, "terminal_state_too_large")?;
    let reference = PrivateFileReference::new(path, Sha256Digest::digest_bytes(&expected));
    let observed = reference.read_verified_bounded(root, MAX_RECORD_BYTES)?;
    if observed != expected {
        return Err(integrity("terminal_state_mismatch"));
    }
    Ok(reference)
}

fn persist_canonical<T: Serialize>(
    root: &Path,
    directory: &str,
    digest: &Sha256Digest,
    value: &T,
) -> BrainResult<PrivateFileReference> {
    let bytes = canonical_json_bounded(value, MAX_RECORD_BYTES, "canonical_record_too_large")?;
    let path = canonical_path(root, directory, digest);
    let sha256 = write_or_verify_immutable(root, &path, &bytes)?;
    Ok(PrivateFileReference::new(path, sha256))
}

fn read_canonical<T: for<'de> Deserialize<'de> + Serialize>(
    reference: &PrivateFileReference,
    root: &Path,
) -> BrainResult<T> {
    let bytes = reference.read_verified_bounded(root, MAX_RECORD_BYTES)?;
    let value: T = serde_json::from_slice(&bytes)?;
    if canonical_json_bounded(&value, MAX_RECORD_BYTES, "canonical_record_too_large")? != bytes {
        return Err(integrity("pure_capability_record_noncanonical"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
    };
    use crate::content_vault::capture_to_vault;
    use crate::identity::AcquisitionId;
    use crate::security::secure_dir;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture(label: &str) -> (PathBuf, PathBuf, CaptureReceipt, PrivateFileReference) {
        let base = std::env::temp_dir().join(format!(
            "tidex-pure-e2e-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let donor = base.join("donor");
        let root = base.join("private");
        fs::create_dir_all(&donor).unwrap();
        fs::create_dir_all(&root).unwrap();
        secure_dir(&root).unwrap();
        let descriptor = LinearMapDescriptor::new(
            CapabilityId::parse("fixture.linear-map:v1").unwrap(),
            2,
            2,
            vec![2.0, -1.0, 0.5, 3.0],
        )
        .unwrap();
        let alternative = LinearMapDescriptor::new(
            CapabilityId::parse("fixture.linear-map:v1").unwrap(),
            2,
            2,
            vec![0.0, -1.0, 0.5, 3.0],
        )
        .unwrap();
        fs::write(
            donor.join("capability.linear.json"),
            serde_json::to_vec(&descriptor).unwrap(),
        )
        .unwrap();
        fs::write(
            donor.join("alternative.linear.json"),
            serde_json::to_vec(&alternative).unwrap(),
        )
        .unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse(format!("pure-e2e-{label}")).unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::WeightsOnly,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 8,
                max_total_bytes: 1 << 20,
            },
            vec![],
        )
        .unwrap();
        let capture = capture_to_vault(&donor, &root, &request).unwrap();
        let capture_reference = capture.persist(&root).unwrap();
        (root, donor, capture, capture_reference)
    }

    fn reveal(id: &str) -> TerminalChallengeReveal {
        TerminalChallengeReveal::new(
            id,
            vec![
                TerminalCase::new(vec![4.0, 1.0], vec![7.0, 5.0]).unwrap(),
                TerminalCase::new(vec![-2.0, 3.0], vec![-7.0, 8.0]).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn retained_descriptor_reaches_terminal_weights_after_donor_removal() {
        let (root, donor, capture, capture_reference) = fixture("complete");
        let source = retained_source_adapter(&root, &capture).unwrap();
        fs::remove_dir_all(&donor).unwrap();
        let discovery = PureCapabilityDiscoveryAuthority::open(&root)
            .unwrap()
            .discover_linear_map(
                capture_reference,
                &source,
                Path::new("capability.linear.json"),
            )
            .unwrap();
        let challenge = reveal("terminal-one");
        let commitment = TerminalChallengeCommitment::commit(&root, &challenge).unwrap();
        let candidates = PureWeightCandidateAuthority::open(&root).unwrap();
        let candidate = candidates
            .materialize(discovery.receipt, commitment)
            .unwrap();
        assert!(!candidates.authorizes_promotion());
        let evaluator = TerminalEvaluationAuthority::open(&root).unwrap();
        let reveal_reference = evaluator
            .persist_reveal_after_candidate(&candidate, &challenge)
            .unwrap();
        let first = evaluator
            .evaluate(candidate.clone(), reveal_reference.clone())
            .unwrap();
        assert_eq!(
            evaluator.authenticate(&first).unwrap().outcome(),
            TerminalEvaluationOutcome::Passed
        );
        let recovered = evaluator.evaluate(candidate, reveal_reference).unwrap();
        assert_eq!(first, recovered);
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn rejects_nonfinite_shapes_reveal_mismatch_and_cross_capture() {
        assert!(validate_terminal_resource_shape(usize::MAX, 2, 2).is_err());
        assert!(validate_terminal_resource_shape(65_536, 65_536, 1).is_err());
        assert!(LinearMapDescriptor::new(
            CapabilityId::parse("fixture.bad:v1").unwrap(),
            1,
            1,
            vec![f64::NAN]
        )
        .is_err());
        assert!(LinearMapDescriptor::new(
            CapabilityId::parse("fixture.bad:v1").unwrap(),
            2,
            2,
            vec![1.0]
        )
        .is_err());

        let (root, donor, capture, capture_reference) = fixture("negative");
        let source = retained_source_adapter(&root, &capture).unwrap();
        let discovery = PureCapabilityDiscoveryAuthority::open(&root)
            .unwrap()
            .discover_linear_map(
                capture_reference,
                &source,
                Path::new("capability.linear.json"),
            )
            .unwrap();
        fs::remove_dir_all(donor).unwrap();
        let committed = reveal("committed");
        let commitment = TerminalChallengeCommitment::commit(&root, &committed).unwrap();
        let candidate = PureWeightCandidateAuthority::open(&root)
            .unwrap()
            .materialize(discovery.receipt, commitment)
            .unwrap();
        let evaluator = TerminalEvaluationAuthority::open(&root).unwrap();
        assert!(evaluator
            .persist_reveal_after_candidate(&candidate, &reveal("different"))
            .is_err());

        let (other_root, other_donor, _, other_capture_reference) = fixture("other");
        assert!(PureCapabilityDiscoveryAuthority::open(&root)
            .unwrap()
            .discover_linear_map(
                other_capture_reference,
                &source,
                Path::new("capability.linear.json"),
            )
            .is_err());
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
        fs::remove_dir_all(other_donor).unwrap();
        fs::remove_dir_all(other_root.parent().unwrap()).unwrap();
    }

    #[test]
    fn semantically_foreign_ir_is_rejected_even_when_internally_authentic() {
        let (root, donor, capture, capture_reference) = fixture("tamper");
        let source = retained_source_adapter(&root, &capture).unwrap();
        let authority = PureCapabilityDiscoveryAuthority::open(&root).unwrap();
        let recorded = authority
            .discover_linear_map(
                capture_reference,
                &source,
                Path::new("capability.linear.json"),
            )
            .unwrap();
        let mut receipt = authority.authenticate(&recorded.receipt).unwrap();
        let descriptor = descriptor_for_discovery(&root, &capture, &receipt).unwrap();
        let foreign_ir =
            expected_linear_map_ir(&descriptor, &capture, Path::new("alternative.linear.json"))
                .unwrap();
        let foreign_reference = foreign_ir.persist(&root, capture.envelope()).unwrap();
        receipt.capability_ir = foreign_reference;
        receipt.capability_ir_sha256 = foreign_ir.manifest_digest().clone();
        receipt.manifest_sha256 = discovery_digest(&receipt).unwrap();
        let forged = persist_canonical(
            &root,
            "state/pure_capability_discovery/by-sha",
            &receipt.manifest_sha256,
            &receipt,
        )
        .unwrap();
        assert!(authority.authenticate(&forged).is_err());
        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn suite_relabel_and_reordering_are_not_new_evidence() {
        let (root, donor, _, _) = fixture("suite-relabel");
        let first = reveal("suite-original");
        let reordered = TerminalChallengeReveal::new(
            "suite-relabelled",
            vec![first.cases[1].clone(), first.cases[0].clone()],
        )
        .unwrap();
        assert_eq!(first.suite_digest(), reordered.suite_digest());
        TerminalChallengeCommitment::commit(&root, &first).unwrap();
        assert!(TerminalChallengeCommitment::commit(&root, &reordered).is_err());
        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn exact_candidate_claim_is_atomic_under_competition() {
        let (root, donor, capture, capture_reference) = fixture("claim-race");
        let source = retained_source_adapter(&root, &capture).unwrap();
        let authority = PureCapabilityDiscoveryAuthority::open(&root).unwrap();
        let first = authority
            .discover_linear_map(
                capture_reference.clone(),
                &source,
                Path::new("capability.linear.json"),
            )
            .unwrap();
        let second = authority
            .discover_linear_map(
                capture_reference,
                &source,
                Path::new("alternative.linear.json"),
            )
            .unwrap();
        let challenge = reveal("claim-race");
        let commitment = TerminalChallengeCommitment::commit(&root, &challenge).unwrap();
        let left_root = root.clone();
        let left_commitment = commitment.clone();
        let left = std::thread::spawn(move || {
            PureWeightCandidateAuthority::open(&left_root)
                .and_then(|authority| authority.materialize(first.receipt, left_commitment))
        });
        let right_root = root.clone();
        let right = std::thread::spawn(move || {
            PureWeightCandidateAuthority::open(&right_root)
                .and_then(|authority| authority.materialize(second.receipt, commitment))
        });
        let outcomes = [left.join().unwrap(), right.join().unwrap()];
        assert_eq!(outcomes.iter().filter(|outcome| outcome.is_ok()).count(), 1);
        assert_eq!(
            outcomes.iter().filter(|outcome| outcome.is_err()).count(),
            1
        );
        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn rejected_terminal_attempt_is_persisted_consumed_and_idempotent() {
        let (root, donor, capture, capture_reference) = fixture("terminal-rejection");
        let source = retained_source_adapter(&root, &capture).unwrap();
        let discovery = PureCapabilityDiscoveryAuthority::open(&root)
            .unwrap()
            .discover_linear_map(
                capture_reference,
                &source,
                Path::new("capability.linear.json"),
            )
            .unwrap();
        let challenge = TerminalChallengeReveal::new(
            "terminal-rejection",
            vec![TerminalCase::new(vec![4.0, 1.0], vec![0.0, 0.0]).unwrap()],
        )
        .unwrap();
        let commitment = TerminalChallengeCommitment::commit(&root, &challenge).unwrap();
        let candidate = PureWeightCandidateAuthority::open(&root)
            .unwrap()
            .materialize(discovery.receipt, commitment)
            .unwrap();
        let evaluator = TerminalEvaluationAuthority::open(&root).unwrap();
        let revealed = evaluator
            .persist_reveal_after_candidate(&candidate, &challenge)
            .unwrap();
        let first = evaluator
            .evaluate(candidate.clone(), revealed.clone())
            .unwrap();
        assert_eq!(
            evaluator.authenticate(&first).unwrap().outcome(),
            TerminalEvaluationOutcome::Rejected {
                reason: TerminalRejectionReason::ExpectedOutputMismatch
            }
        );
        assert_eq!(first, evaluator.evaluate(candidate, revealed).unwrap());
        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn malformed_reveal_is_rejected_before_publication() {
        let (root, donor, capture, capture_reference) = fixture("shape-before-reveal");
        let source = retained_source_adapter(&root, &capture).unwrap();
        let discovery = PureCapabilityDiscoveryAuthority::open(&root)
            .unwrap()
            .discover_linear_map(
                capture_reference,
                &source,
                Path::new("capability.linear.json"),
            )
            .unwrap();
        let challenge = TerminalChallengeReveal::new(
            "shape-before-reveal",
            vec![TerminalCase::new(vec![1.0], vec![1.0]).unwrap()],
        )
        .unwrap();
        let commitment = TerminalChallengeCommitment::commit(&root, &challenge).unwrap();
        let candidate = PureWeightCandidateAuthority::open(&root)
            .unwrap()
            .materialize(discovery.receipt, commitment)
            .unwrap();
        assert!(TerminalEvaluationAuthority::open(&root)
            .unwrap()
            .persist_reveal_after_candidate(&candidate, &challenge)
            .is_err());
        assert!(fs::symlink_metadata(canonical_path(
            &root,
            "state/terminal_challenges/reveals/by-sha",
            &challenge.reveal_sha256
        ))
        .is_err());
        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn candidate_weight_binding_distinguishes_signed_zero() {
        let (root, donor, capture, capture_reference) = fixture("signed-zero");
        let source = retained_source_adapter(&root, &capture).unwrap();
        let discovery = PureCapabilityDiscoveryAuthority::open(&root)
            .unwrap()
            .discover_linear_map(
                capture_reference,
                &source,
                Path::new("alternative.linear.json"),
            )
            .unwrap();
        let challenge = reveal("signed-zero");
        let commitment = TerminalChallengeCommitment::commit(&root, &challenge).unwrap();
        let authority = PureWeightCandidateAuthority::open(&root).unwrap();
        let reference = authority
            .materialize(discovery.receipt, commitment)
            .unwrap();
        let mut forged: PureWeightCandidateReceipt = read_canonical(&reference, &root).unwrap();
        assert_eq!(forged.weights[0].to_bits(), 0.0_f64.to_bits());
        forged.weights[0] = -0.0;
        forged.manifest_sha256 = candidate_digest(&forged).unwrap();
        let forged_reference = persist_canonical(
            &root,
            "state/pure_weight_candidates/by-sha",
            &forged.manifest_sha256,
            &forged,
        )
        .unwrap();
        let error = authority.authenticate(&forged_reference).unwrap_err();
        assert!(matches!(
            error.to_string().as_str(),
            "integrity:pure_weight_candidate_chain_mismatch"
                | "integrity:private_file_reference_digest_mismatch"
        ));
        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn isolated_runner_wire_is_canonical_and_contains_no_oracle_or_authority_material() {
        let package = PureLinearExecutionPackage {
            schema: PureLinearExecutionPackageSchema::Current,
            input_dimension: 2,
            output_dimension: 2,
            weights: vec![2.0, -1.0, 0.5, 3.0],
            inputs: vec![vec![4.0, 1.0], vec![-2.0, 3.0]],
        };
        let wire = package.to_canonical_bytes().unwrap();
        assert_eq!(
            wire,
            br#"{"schema":"cerebro.tidex.pure_linear_execution_package/v1","input_dimension":2,"output_dimension":2,"weights":[2.0,-1.0,0.5,3.0],"inputs":[[4.0,1.0],[-2.0,3.0]]}"#
        );
        let text = std::str::from_utf8(&wire).unwrap();
        for forbidden in [
            "expected",
            "source",
            "capture",
            "cas",
            "capability_id",
            "challenge",
            "path",
        ] {
            assert!(!text.contains(forbidden));
        }
        assert_eq!(
            PureLinearExecutionPackage::from_canonical_bytes(&wire).unwrap(),
            package
        );
        let mut noncanonical = wire.clone();
        noncanonical.push(b'\n');
        assert!(PureLinearExecutionPackage::from_canonical_bytes(&noncanonical).is_err());

        let output = package.execute().unwrap().to_canonical_bytes().unwrap();
        assert_eq!(
            output,
            br#"{"schema":"cerebro.tidex.pure_linear_execution_output/v1","outputs":[[7.0,5.0],[-7.0,8.0]]}"#
        );
    }

    #[test]
    fn sealed_candidate_and_execution_package_survive_unavailable_source_cas() {
        let (root, donor, capture, capture_reference) = fixture("sealed-without-cas");
        let source = retained_source_adapter(&root, &capture).unwrap();
        let discovery = PureCapabilityDiscoveryAuthority::open(&root)
            .unwrap()
            .discover_linear_map(
                capture_reference,
                &source,
                Path::new("capability.linear.json"),
            )
            .unwrap();
        let challenge = reveal("sealed-without-cas");
        let commitment = TerminalChallengeCommitment::commit(&root, &challenge).unwrap();
        let candidates = PureWeightCandidateAuthority::open(&root).unwrap();
        let candidate_reference = candidates
            .materialize(discovery.receipt, commitment)
            .unwrap();

        let vault = root.join("state/acquisitions/content-vault");
        let unavailable_vault = root.join("state/acquisitions/content-vault.unavailable");
        fs::rename(&vault, &unavailable_vault).unwrap();
        assert!(candidates.authenticate(&candidate_reference).is_err());

        let candidate = candidates
            .authenticate_sealed(&candidate_reference)
            .unwrap();
        let package =
            PureLinearExecutionPackage::from_authenticated_chain(&candidate, &challenge).unwrap();
        let wire = package.to_canonical_bytes().unwrap();
        let reopened = PureLinearExecutionPackage::from_canonical_bytes(&wire).unwrap();
        assert_eq!(
            reopened.execute().unwrap().outputs,
            vec![vec![7.0, 5.0], vec![-7.0, 8.0]]
        );
        assert!(!candidates.authorizes_promotion());

        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn isolated_receipt_manifest_binds_request_backend_and_boundary() {
        let digest = |label: &[u8]| Sha256Digest::digest_bytes(label);
        let reference = |label: &str| {
            PrivateFileReference::new(
                PathBuf::from(format!("/private/{label}.json")),
                digest(label.as_bytes()),
            )
        };
        let mut receipt = IsolatedTerminalEvaluationReceipt {
            schema: IsolatedTerminalEvaluationSchema::Current,
            state: TerminalChallengeState::Terminal,
            candidate: reference("candidate"),
            candidate_sha256: digest(b"candidate-manifest"),
            commitment: reference("commitment"),
            commitment_sha256: digest(b"commitment-manifest"),
            suite_sha256: digest(b"suite"),
            claim: reference("claim"),
            reveal: reference("reveal"),
            reveal_sha256: digest(b"reveal-manifest"),
            reveal_transition: reference("reveal-transition"),
            case_count: 2,
            runner_admission: reference("runner-admission"),
            runner_admission_manifest_sha256: digest(b"runner-admission-manifest"),
            runner_sha256: digest(b"runner"),
            execution_input_sha256: digest(b"input"),
            request_digest: digest(b"request"),
            backend: ISOLATED_BACKEND_CONTRACT.into(),
            canonical_stdout_sha256: digest(b"stdout"),
            observed_outputs_sha256: digest(b"outputs"),
            outcome: TerminalEvaluationOutcome::Passed,
            execution_boundary:
                TerminalExecutionBoundary::LinuxBubblewrapSealedMemfdAuthenticatedRunnerAndInputOnly,
            evidence_scope: TerminalEvidenceScope::PrecommittedCallerCasesNotIndependent,
            manifest_sha256: Sha256Digest::zero(),
        };
        let original = isolated_evaluation_digest(&receipt).unwrap();
        receipt.request_digest = digest(b"different-request");
        assert_ne!(original, isolated_evaluation_digest(&receipt).unwrap());
        receipt.request_digest = digest(b"request");
        receipt.backend = "different-backend".into();
        assert_ne!(original, isolated_evaluation_digest(&receipt).unwrap());
        receipt.backend = ISOLATED_BACKEND_CONTRACT.into();
        receipt.execution_boundary =
            TerminalExecutionBoundary::InProcessAuthenticatedWeightsOnlyNoSourceArgument;
        assert_ne!(original, isolated_evaluation_digest(&receipt).unwrap());
    }

    #[test]
    fn isolated_evaluation_accepts_only_a_registered_runner_receipt() {
        let (root, donor, _, _) = fixture("runner-admission");
        let authority = PureLinearRunnerAuthority::open(&root).unwrap();
        let runner = b"\x7fELFlocally-built-runner".to_vec();
        let runner_sha256 = Sha256Digest::digest_bytes(&runner);
        let admission = authority
            .admit_local_build(
                runner,
                runner_sha256.clone(),
                Some(Sha256Digest::digest_bytes(b"claimed-source-tree")),
                "cargo-build:pure_linear_runner:test-fixture",
            )
            .unwrap();
        let authenticated = authority.authenticate(&admission).unwrap();
        assert_eq!(authenticated.runner_program_sha256(), &runner_sha256);
        assert_eq!(
            authenticated.attestation_scope(),
            RunnerAttestationScope::LocallyAdmittedNotIndependentlyAttested
        );
        assert!(!authority.authorizes_promotion());

        let unregistered_path = root.join("unregistered-runner.elf");
        let unregistered_bytes = b"\x7fELFalternate";
        fs::write(&unregistered_path, unregistered_bytes).unwrap();
        let unregistered = PrivateFileReference::new(
            unregistered_path,
            Sha256Digest::digest_bytes(unregistered_bytes),
        );
        assert!(authority.authenticate(&unregistered).is_err());

        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn canonical_record_limit_is_checked_before_write() {
        assert!(
            canonical_json_bounded(&vec!["0123456789"; 8], 16, "fixture_record_too_large").is_err()
        );
    }
}
