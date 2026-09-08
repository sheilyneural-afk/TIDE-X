//! A closed, typed intermediate representation for acquired capabilities.
//!
//! Source code is evidence, not a safe runtime representation.  This IR is a
//! deliberately small bridge between a verified donor envelope and later
//! reconstruction or target-specific materialisation.  It cannot embed shell,
//! Python, Rust source, dynamic loading, network access, or unbounded loops.

use crate::acquisition_contract::SystemEnvelope;
use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::digest::{CapabilityIrDigest, Sha256Digest, SystemEnvelopeDigest};
use crate::error::{BrainError, BrainResult};
use crate::identity::{CapabilityId, CapabilityNodeId, PortId, PrimitiveId, PrimitiveProfileId};
use crate::security::verify_internal_private_root;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

const IR_DOMAIN: &[u8] = b"CEREBRO:TIDEX:CAPABILITY-IR:v2\0";
const MAX_IR_INPUTS: usize = 4_096;
const MAX_IR_PARAMETERS: usize = 4_096;
const MAX_CAPABILITY_IR_BYTES: u64 = 256 * 1024 * 1024;
const MAX_IR_NODES: usize = 65_536;
const MAX_IR_OUTPUTS: usize = 4_096;
const MAX_TENSOR_RANK: usize = 16;
const MAX_TENSOR_ELEMENTS: u64 = 1 << 40;
const MAX_NODE_PROVENANCE: usize = 4_096;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CapabilityIrSchema {
    #[serde(rename = "cerebro.tidex.capability_ir/v2")]
    Current,
}

/// A value supplied by a resident candidate rather than by a runtime caller.
///
/// Keeping this distinct from [`TypedPort`] inputs prevents a weight tensor
/// from being represented as an ordinary request argument.  Parameter values
/// are deliberately absent from the IR: a later candidate binds them under a
/// separate authenticated contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ParameterSlot {
    port: TypedPort,
}

impl ParameterSlot {
    pub fn new(port: TypedPort) -> BrainResult<Self> {
        port.validate("capability_ir_parameter")?;
        if !matches!(port.value_type, ValueType::F64 | ValueType::TensorF64) {
            return Err(BrainError::Invalid(
                "capability_ir_parameter_not_numeric".into(),
            ));
        }
        Ok(Self { port })
    }

    pub fn port(&self) -> &TypedPort {
        &self.port
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValueType {
    Bool,
    I64,
    F64,
    Bytes,
    TensorF64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TypedPort {
    name: PortId,
    value_type: ValueType,
    /// Empty shape denotes a scalar; dimensions must be nonzero.
    shape: Vec<u64>,
}

impl TypedPort {
    pub fn scalar(name: PortId, value_type: ValueType) -> BrainResult<Self> {
        if value_type == ValueType::TensorF64 {
            return Err(BrainError::Invalid(
                "typed_port_tensor_requires_shape".into(),
            ));
        }
        let port = Self {
            name,
            value_type,
            shape: Vec::new(),
        };
        port.validate("typed_port")?;
        Ok(port)
    }

    pub fn tensor_f64(name: PortId, shape: Vec<u64>) -> BrainResult<Self> {
        let port = Self {
            name,
            value_type: ValueType::TensorF64,
            shape,
        };
        port.validate("typed_port")?;
        Ok(port)
    }

    pub fn name(&self) -> &PortId {
        &self.name
    }

    pub fn value_type(&self) -> ValueType {
        self.value_type
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn validate(&self, label: &str) -> BrainResult<()> {
        if self.shape.len() > MAX_TENSOR_RANK || self.shape.contains(&0) {
            return Err(BrainError::Invalid(format!("{label}_invalid")));
        }
        if self.value_type != ValueType::TensorF64 && !self.shape.is_empty() {
            return Err(BrainError::Invalid(format!("{label}_shape_for_non_tensor")));
        }
        if self.value_type == ValueType::TensorF64 {
            let elements = self.shape.iter().try_fold(1u64, |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or_else(|| BrainError::Invalid(format!("{label}_element_count_overflow")))
            })?;
            if self.shape.is_empty() || elements > MAX_TENSOR_ELEMENTS {
                return Err(BrainError::Invalid(format!("{label}_tensor_shape_invalid")));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    Pure,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ShapeRule {
    ElementwiseSame,
    TensorScale,
    MatrixMultiply2d,
    PreserveFirst,
    Scalar,
}

/// A primitive profile is the only set of operations that a capability IR may
/// invoke.  Adding a future primitive changes the profile identity and must be
/// reviewed independently; a donor cannot inject an operation by name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PrimitiveContract {
    primitive_id: PrimitiveId,
    input_types: Vec<ValueType>,
    output_type: ValueType,
    shape_rule: ShapeRule,
    effects: BTreeSet<EffectKind>,
}

impl PrimitiveContract {
    fn validate(&self) -> BrainResult<()> {
        if self.input_types.is_empty() || self.effects.is_empty() {
            return Err(BrainError::Invalid("primitive_contract_invalid".into()));
        }
        if self.effects.contains(&EffectKind::Pure) && self.effects.len() != 1 {
            return Err(BrainError::Invalid("primitive_effects_invalid".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PrimitiveSet {
    profile_id: PrimitiveProfileId,
    primitives: Vec<PrimitiveContract>,
    sha256: Sha256Digest,
}

impl PrimitiveSet {
    /// Minimal runtime-independent primitive vocabulary.  It purposefully has
    /// no process, filesystem-write, shell, source-evaluation or network node.
    pub fn tidex_core_v1() -> BrainResult<Self> {
        let pure = BTreeSet::from([EffectKind::Pure]);
        let contracts = vec![
            (
                "tensor.add",
                vec![ValueType::TensorF64, ValueType::TensorF64],
                ValueType::TensorF64,
                ShapeRule::ElementwiseSame,
            ),
            (
                "tensor.scale",
                vec![ValueType::TensorF64, ValueType::F64],
                ValueType::TensorF64,
                ShapeRule::TensorScale,
            ),
            (
                "tensor.matmul",
                vec![ValueType::TensorF64, ValueType::TensorF64],
                ValueType::TensorF64,
                ShapeRule::MatrixMultiply2d,
            ),
            (
                "tensor.normalize",
                vec![ValueType::TensorF64],
                ValueType::TensorF64,
                ShapeRule::PreserveFirst,
            ),
            (
                "select.arg_max",
                vec![ValueType::TensorF64],
                ValueType::I64,
                ShapeRule::Scalar,
            ),
            (
                "compare.less_than",
                vec![ValueType::F64, ValueType::F64],
                ValueType::Bool,
                ShapeRule::Scalar,
            ),
        ];
        let mut primitive_set = Self {
            profile_id: PrimitiveProfileId::parse("tidex.core.v1")?,
            primitives: contracts
                .into_iter()
                .map(|(id, input_types, output_type, shape_rule)| {
                    Ok(PrimitiveContract {
                        primitive_id: PrimitiveId::parse(id)?,
                        input_types,
                        output_type,
                        shape_rule,
                        effects: pure.clone(),
                    })
                })
                .collect::<BrainResult<Vec<_>>>()?,
            sha256: Sha256Digest::zero(),
        };
        primitive_set.sha256 = primitive_set.calculate_digest()?;
        Ok(primitive_set)
    }

    pub fn verify(&self) -> BrainResult<()> {
        let authorized = Self::tidex_core_v1()?;
        if self != &authorized {
            return Err(BrainError::Integrity("primitive_set_not_authorized".into()));
        }
        let mut identities = BTreeSet::new();
        for primitive in &self.primitives {
            primitive.validate()?;
            if !identities.insert(primitive.primitive_id.clone()) {
                return Err(BrainError::Invalid("primitive_set_duplicate_id".into()));
            }
        }
        Ok(())
    }

    pub fn find(&self, primitive_id: &PrimitiveId) -> Option<&PrimitiveContract> {
        self.primitives
            .iter()
            .find(|primitive| primitive.primitive_id == *primitive_id)
    }

    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.sha256 = Sha256Digest::zero();
        Ok(domain_digest(
            b"CEREBRO:TIDEX:PRIMITIVE-SET:v1\0",
            &serde_json::to_vec(&unsigned)?,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ValueReference {
    Input { name: PortId },
    Parameter { name: PortId },
    NodeOutput { node_id: CapabilityNodeId },
}

/// A public result is not merely a graph edge: it has a stable port identity
/// and an exact type/shape contract that must match its source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OutputBinding {
    port: TypedPort,
    source: ValueReference,
}

impl OutputBinding {
    pub fn new(port: TypedPort, source: ValueReference) -> BrainResult<Self> {
        port.validate("capability_ir_output")?;
        Ok(Self { port, source })
    }

    pub fn port(&self) -> &TypedPort {
        &self.port
    }

    pub fn source(&self) -> &ValueReference {
        &self.source
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IrNode {
    node_id: CapabilityNodeId,
    primitive_id: PrimitiveId,
    inputs: Vec<ValueReference>,
    output: TypedPort,
    /// Source paths in the acquired envelope from which this node was derived.
    provenance: Vec<PathBuf>,
}

impl IrNode {
    pub fn new(
        node_id: CapabilityNodeId,
        primitive_id: PrimitiveId,
        inputs: Vec<ValueReference>,
        output: TypedPort,
        provenance: Vec<PathBuf>,
    ) -> BrainResult<Self> {
        output.validate("capability_ir_node_output")?;
        let canonical = canonical_paths(&provenance, "capability_ir_provenance")?;
        if provenance.is_empty()
            || provenance.len() > MAX_NODE_PROVENANCE
            || canonical.iter().cloned().collect::<Vec<_>>() != provenance
        {
            return Err(BrainError::Invalid(
                "capability_ir_node_provenance_invalid".into(),
            ));
        }
        Ok(Self {
            node_id,
            primitive_id,
            inputs,
            output,
            provenance,
        })
    }

    pub fn node_id(&self) -> &CapabilityNodeId {
        &self.node_id
    }

    pub fn primitive_id(&self) -> &PrimitiveId {
        &self.primitive_id
    }

    pub fn inputs(&self) -> &[ValueReference] {
        &self.inputs
    }

    pub fn output(&self) -> &TypedPort {
        &self.output
    }

    pub fn provenance(&self) -> &[PathBuf] {
        &self.provenance
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityIr {
    schema: CapabilityIrSchema,
    capability_id: CapabilityId,
    system_envelope_sha256: SystemEnvelopeDigest,
    primitive_set: PrimitiveSet,
    inputs: Vec<TypedPort>,
    parameters: Vec<ParameterSlot>,
    nodes: Vec<IrNode>,
    outputs: Vec<OutputBinding>,
    manifest_sha256: CapabilityIrDigest,
}

impl CapabilityIr {
    pub fn capability_id(&self) -> &CapabilityId {
        &self.capability_id
    }

    pub fn system_envelope_digest(&self) -> &SystemEnvelopeDigest {
        &self.system_envelope_sha256
    }

    pub fn manifest_digest(&self) -> &CapabilityIrDigest {
        &self.manifest_sha256
    }

    pub fn inputs(&self) -> &[TypedPort] {
        &self.inputs
    }

    pub fn parameters(&self) -> &[ParameterSlot] {
        &self.parameters
    }

    pub fn nodes(&self) -> &[IrNode] {
        &self.nodes
    }

    pub fn outputs(&self) -> &[OutputBinding] {
        &self.outputs
    }

    pub fn new(
        capability_id: CapabilityId,
        envelope: &SystemEnvelope,
        primitive_set: PrimitiveSet,
        inputs: Vec<TypedPort>,
        nodes: Vec<IrNode>,
        outputs: Vec<OutputBinding>,
    ) -> BrainResult<Self> {
        Self::new_with_parameters(
            capability_id,
            envelope,
            primitive_set,
            inputs,
            Vec::new(),
            nodes,
            outputs,
        )
    }

    pub fn new_with_parameters(
        capability_id: CapabilityId,
        envelope: &SystemEnvelope,
        primitive_set: PrimitiveSet,
        inputs: Vec<TypedPort>,
        parameters: Vec<ParameterSlot>,
        nodes: Vec<IrNode>,
        outputs: Vec<OutputBinding>,
    ) -> BrainResult<Self> {
        let mut ir = Self {
            schema: CapabilityIrSchema::Current,
            capability_id,
            system_envelope_sha256: envelope.manifest_sha256().clone(),
            primitive_set,
            inputs,
            parameters,
            nodes,
            outputs,
            manifest_sha256: CapabilityIrDigest::draft_marker(),
        };
        ir.validate_structure_against(envelope)?;
        ir.manifest_sha256 = ir.calculate_digest()?;
        ir.validate_against(envelope)?;
        Ok(ir)
    }

    pub fn validate_against(&self, envelope: &SystemEnvelope) -> BrainResult<()> {
        self.validate_structure_against(envelope)?;
        if self.manifest_sha256.is_draft() || self.calculate_digest()? != self.manifest_sha256 {
            return Err(BrainError::Integrity(
                "capability_ir_digest_mismatch".into(),
            ));
        }
        Ok(())
    }

    fn validate_structure_against(&self, envelope: &SystemEnvelope) -> BrainResult<()> {
        envelope.verify_manifest().map_err(|_| {
            BrainError::Integrity("capability_ir_envelope_not_authenticated".into())
        })?;
        if &self.system_envelope_sha256 != envelope.manifest_sha256() {
            return Err(BrainError::Integrity(
                "capability_ir_envelope_mismatch".into(),
            ));
        }
        self.primitive_set.verify()?;
        if self.schema != CapabilityIrSchema::Current
            || self.inputs.len() > MAX_IR_INPUTS
            || self.parameters.len() > MAX_IR_PARAMETERS
            || self.nodes.len() > MAX_IR_NODES
            || self.outputs.len() > MAX_IR_OUTPUTS
        {
            return Err(BrainError::Invalid("capability_ir_budget_exceeded".into()));
        }
        let mut inputs = std::collections::BTreeMap::new();
        for input in &self.inputs {
            input.validate("capability_ir_input")?;
            if inputs.insert(input.name.clone(), input.clone()).is_some() {
                return Err(BrainError::Invalid("capability_ir_input_duplicate".into()));
            }
        }
        let mut parameters = std::collections::BTreeMap::new();
        for parameter in &self.parameters {
            parameter.port.validate("capability_ir_parameter")?;
            if !matches!(
                parameter.port.value_type,
                ValueType::F64 | ValueType::TensorF64
            ) {
                return Err(BrainError::Invalid(
                    "capability_ir_parameter_not_numeric".into(),
                ));
            }
            if inputs.contains_key(&parameter.port.name) {
                return Err(BrainError::Invalid(
                    "capability_ir_input_parameter_name_collision".into(),
                ));
            }
            if parameters
                .insert(parameter.port.name.clone(), parameter.port.clone())
                .is_some()
            {
                return Err(BrainError::Invalid(
                    "capability_ir_parameter_duplicate".into(),
                ));
            }
        }
        if self.outputs.is_empty() || self.nodes.is_empty() {
            return Err(BrainError::Invalid("capability_ir_empty".into()));
        }
        let source_files: BTreeSet<_> = envelope
            .entries()
            .iter()
            .filter(|entry| entry.kind() == crate::acquisition_contract::SnapshotEntryKind::File)
            .map(|entry| entry.relative_path().to_path_buf())
            .collect();
        let mut known_nodes = std::collections::BTreeMap::new();
        let mut dependencies = std::collections::BTreeMap::new();
        let mut used_parameters = BTreeSet::new();
        for node in &self.nodes {
            node.output.validate("capability_ir_node_output")?;
            let primitive = self.primitive_set.find(&node.primitive_id).ok_or_else(|| {
                BrainError::Integrity("capability_ir_primitive_not_admitted".into())
            })?;
            if node.inputs.len() != primitive.input_types.len()
                || node.output.value_type != primitive.output_type
            {
                return Err(BrainError::Invalid(
                    "capability_ir_node_contract_invalid".into(),
                ));
            }
            if known_nodes.contains_key(&node.node_id) {
                return Err(BrainError::Invalid("capability_ir_node_duplicate".into()));
            }
            if node.provenance.is_empty() || node.provenance.len() > MAX_NODE_PROVENANCE {
                return Err(BrainError::Invalid(
                    "capability_ir_node_provenance_missing".into(),
                ));
            }
            let provenance = canonical_paths(&node.provenance, "capability_ir_provenance")?;
            if provenance.iter().cloned().collect::<Vec<_>>() != node.provenance
                || provenance.iter().any(|path| !source_files.contains(path))
            {
                return Err(BrainError::Integrity(
                    "capability_ir_provenance_invalid".into(),
                ));
            }
            let node_dependencies = node
                .inputs
                .iter()
                .filter_map(|reference| match reference {
                    ValueReference::NodeOutput { node_id } => Some(node_id.clone()),
                    ValueReference::Input { .. } | ValueReference::Parameter { .. } => None,
                })
                .collect::<BTreeSet<_>>();
            used_parameters.extend(node.inputs.iter().filter_map(|reference| match reference {
                ValueReference::Parameter { name } => Some(name.clone()),
                ValueReference::Input { .. } | ValueReference::NodeOutput { .. } => None,
            }));
            let operand_types = node
                .inputs
                .iter()
                .map(|reference| resolve_reference(reference, &inputs, &parameters, &known_nodes))
                .collect::<BrainResult<Vec<_>>>()?;
            for (operand, expected) in operand_types.iter().zip(&primitive.input_types) {
                if operand.value_type != *expected {
                    return Err(BrainError::Integrity(
                        "capability_ir_operand_type_mismatch".into(),
                    ));
                }
            }
            validate_shape_rule(primitive.shape_rule, &operand_types, &node.output)?;
            dependencies.insert(node.node_id.clone(), node_dependencies);
            known_nodes.insert(node.node_id.clone(), node.output.clone());
        }
        let mut output_names = BTreeSet::new();
        let mut reachable = BTreeSet::new();
        let mut pending = Vec::new();
        for binding in &self.outputs {
            binding.port.validate("capability_ir_output")?;
            if !output_names.insert(binding.port.name.clone()) {
                return Err(BrainError::Invalid("capability_ir_output_duplicate".into()));
            }
            let source = resolve_reference(&binding.source, &inputs, &parameters, &known_nodes)?;
            if source.value_type != binding.port.value_type || source.shape != binding.port.shape {
                return Err(BrainError::Integrity(
                    "capability_ir_output_contract_mismatch".into(),
                ));
            }
            if let ValueReference::NodeOutput { node_id } = &binding.source {
                pending.push(node_id.clone());
            }
        }
        while let Some(node_id) = pending.pop() {
            if reachable.insert(node_id.clone()) {
                pending.extend(
                    dependencies
                        .get(&node_id)
                        .ok_or_else(|| {
                            BrainError::Integrity("capability_ir_reachability_invalid".into())
                        })?
                        .iter()
                        .cloned(),
                );
            }
        }
        if reachable.len() != known_nodes.len() {
            return Err(BrainError::Integrity("capability_ir_dead_node".into()));
        }
        if used_parameters.len() != parameters.len() {
            return Err(BrainError::Integrity("capability_ir_dead_parameter".into()));
        }
        Ok(())
    }

    pub fn persist(
        &self,
        private_root: &Path,
        envelope: &SystemEnvelope,
    ) -> BrainResult<PrivateFileReference> {
        let root = verify_internal_private_root(private_root)?;
        self.validate_against(envelope)?;
        let destination = capability_ir_path(&root, &self.manifest_sha256);
        let bytes = serde_json::to_vec(self)?;
        let sha256 = write_or_verify_immutable(&root, &destination, &bytes)?;
        let reference = PrivateFileReference::new(destination, sha256);
        authenticate_capability_ir(&root, &reference, envelope)?;
        Ok(reference)
    }

    fn calculate_digest(&self) -> BrainResult<CapabilityIrDigest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = CapabilityIrDigest::draft_marker();
        Ok(CapabilityIrDigest::from_computed(domain_digest(
            IR_DOMAIN,
            &serde_json::to_vec(&unsigned)?,
        )))
    }
}

/// Execution result for the bounded scalar linear readout already represented
/// by the pure-capability discovery profile. This proves the declared
/// readout's arithmetic on supplied activations, not a model's general
/// capability, an independently attested observation, or promotion.
///
/// The caller authenticates the parameter values and activation provenance.
/// Structural IR intentionally contains parameter slots rather than weights.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinearReadoutExecution {
    pub schema: String,
    pub capability_ir_sha256: CapabilityIrDigest,
    pub operator_node_id: CapabilityNodeId,
    pub parameter_port_id: PortId,
    pub input_dimension: usize,
    pub row_count: usize,
    pub weights_sha256: Sha256Digest,
    pub inputs_sha256: Sha256Digest,
    pub raw_margins: Vec<f64>,
    pub execution_arithmetic: String,
    pub candidate_only: bool,
}

/// Execute the scalar-output specialization of the existing authenticated
/// linear-map IR. There is exactly one runtime tensor [d, 1], one resident
/// parameter tensor [1, d], one matmul(parameter, input), and its single [1, 1]
/// public output. Names come from the actual graph; no new IR constructor or
/// donor-supplied executable vocabulary is introduced.
///
/// Parameters must have been authenticated by the caller against the acquired
/// descriptor/model. The returned value digests bind exactly what this call
/// consumed, without implying that the structural IR itself contains weights.
pub fn execute_linear_readout(
    ir: &CapabilityIr,
    envelope: &SystemEnvelope,
    weights: &[f64],
    inputs: &[Vec<f64>],
) -> BrainResult<LinearReadoutExecution> {
    const MAX_READOUT_DIMENSION: usize = 4_096;
    const MAX_READOUT_ROWS: usize = 128;
    const MAX_READOUT_WORK: usize = MAX_READOUT_DIMENSION * MAX_READOUT_ROWS;

    ir.validate_against(envelope)?;
    if ir.inputs.len() != 1
        || ir.parameters.len() != 1
        || ir.nodes.len() != 1
        || ir.outputs.len() != 1
    {
        return Err(BrainError::Invalid(
            "linear_readout_graph_profile_mismatch".into(),
        ));
    }
    let input = &ir.inputs[0];
    let parameter = ir.parameters[0].port();
    let node = &ir.nodes[0];
    let output = &ir.outputs[0];
    if input.value_type != ValueType::TensorF64
        || input.shape.len() != 2
        || input.shape[1] != 1
        || parameter.value_type != ValueType::TensorF64
        || parameter.shape.as_slice() != [1, input.shape[0]]
        || node.primitive_id.as_str() != "tensor.matmul"
        || node.inputs.as_slice()
            != [
                ValueReference::Parameter {
                    name: parameter.name.clone(),
                },
                ValueReference::Input {
                    name: input.name.clone(),
                },
            ]
        || node.output.value_type != ValueType::TensorF64
        || node.output.shape.as_slice() != [1, 1]
        || output.port.value_type != ValueType::TensorF64
        || output.port.shape.as_slice() != [1, 1]
        || output.source
            != (ValueReference::NodeOutput {
                node_id: node.node_id.clone(),
            })
    {
        return Err(BrainError::Invalid(
            "linear_readout_graph_profile_mismatch".into(),
        ));
    }

    let dimension = usize::try_from(input.shape[0])
        .map_err(|_| BrainError::Invalid("linear_readout_dimension_overflow".into()))?;
    if dimension == 0
        || dimension > MAX_READOUT_DIMENSION
        || inputs.is_empty()
        || inputs.len() > MAX_READOUT_ROWS
        || inputs
            .len()
            .checked_mul(dimension)
            .is_none_or(|work| work > MAX_READOUT_WORK)
    {
        return Err(BrainError::Invalid("linear_readout_budget_exceeded".into()));
    }
    if weights.len() != dimension
        || weights.iter().any(|value| !value.is_finite())
        || inputs
            .iter()
            .any(|row| row.len() != dimension || row.iter().any(|value| !value.is_finite()))
    {
        return Err(BrainError::Invalid(
            "linear_readout_parameter_or_input_invalid".into(),
        ));
    }

    // Reuse the existing scaled, compensated dot product through Matrix.
    // Each row is one [d, 1] activation passed to the same [1, d] readout.
    let raw_margins = crate::linalg::Matrix::from_rows(inputs)?.matvec(weights)?;
    let weights_sha256 = domain_digest(
        b"CEREBRO:TIDEX:LINEAR-READOUT-WEIGHTS-JSON:v1\0",
        &serde_json::to_vec(weights)?,
    );
    let inputs_sha256 = domain_digest(
        b"CEREBRO:TIDEX:LINEAR-READOUT-INPUTS-JSON:v1\0",
        &serde_json::to_vec(inputs)?,
    );
    Ok(LinearReadoutExecution {
        schema: "cerebro.tidex.linear_readout_execution/v1".into(),
        capability_ir_sha256: ir.manifest_digest().clone(),
        operator_node_id: node.node_id.clone(),
        parameter_port_id: parameter.name.clone(),
        input_dimension: dimension,
        row_count: inputs.len(),
        weights_sha256,
        inputs_sha256,
        raw_margins,
        execution_arithmetic: "f64_scaled_neumaier_dot/v1".into(),
        candidate_only: true,
    })
}

/// V63 operational semantics bound to one already-authenticated structural IR.
///
/// The structural [`CapabilityIr`] remains the closed executable vocabulary.
/// This contract adds receiver-independent state anchors and operator
/// transitions that define what the capability does. Keeping the binding
/// explicit prevents a structural graph from becoming semantic evidence by
/// assertion alone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OperationalCapabilityContract {
    pub schema: String,
    pub capability_id: CapabilityId,
    pub capability_ir_sha256: CapabilityIrDigest,
    pub state_dimension: u64,
    pub anchors: Vec<StateIrAnchor>,
    pub transitions: Vec<OperatorIrTransition>,
    pub maximum_closure_error: f64,
    pub maximum_contraction_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StateIrAnchor {
    pub anchor_id: String,
    pub state: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OperatorIrTransition {
    pub operator_id: String,
    pub source_anchor_id: String,
    pub target_anchor_id: String,
    pub observed_next_state: Vec<f64>,
    pub pre_target_error: f64,
    pub post_target_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OperationalInterfaceVerification {
    pub schema: String,
    pub transition_count: usize,
    pub maximum_observed_closure_error: f64,
    pub maximum_observed_contraction_ratio: f64,
    pub closure_satisfied: bool,
    pub contraction_satisfied: bool,
    pub allowed: bool,
}

impl OperationalCapabilityContract {
    pub fn validate_against(&self, ir: &CapabilityIr) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.operational_capability/v1"
            || self.capability_id != *ir.capability_id()
            || self.capability_ir_sha256 != *ir.manifest_digest()
            || self.state_dimension == 0
            || self.anchors.is_empty()
            || self.transitions.is_empty()
            || !self.maximum_closure_error.is_finite()
            || self.maximum_closure_error < 0.0
            || !self.maximum_contraction_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_contraction_ratio)
        {
            return Err(BrainError::Invalid(
                "operational_capability_contract_invalid".into(),
            ));
        }
        let dimension = usize::try_from(self.state_dimension)
            .map_err(|_| BrainError::Invalid("operational_state_dimension_overflow".into()))?;
        let mut anchors = std::collections::BTreeMap::new();
        let mut previous_anchor = None::<&str>;
        for anchor in &self.anchors {
            if anchor.anchor_id.is_empty()
                || anchor.anchor_id.len() > 256
                || anchor.state.len() != dimension
                || anchor.state.iter().any(|value| !value.is_finite())
                || previous_anchor.is_some_and(|previous| previous >= anchor.anchor_id.as_str())
                || anchors
                    .insert(anchor.anchor_id.as_str(), anchor.state.as_slice())
                    .is_some()
            {
                return Err(BrainError::Invalid("state_ir_anchor_invalid".into()));
            }
            previous_anchor = Some(anchor.anchor_id.as_str());
        }
        let mut transition_ids = BTreeSet::new();
        let mut previous_transition = None::<(&str, &str, &str)>;
        for transition in &self.transitions {
            let transition_identity = (
                transition.operator_id.as_str(),
                transition.source_anchor_id.as_str(),
                transition.target_anchor_id.as_str(),
            );
            if transition.operator_id.is_empty()
                || transition.operator_id.len() > 256
                || transition.source_anchor_id.is_empty()
                || transition.target_anchor_id.is_empty()
                || previous_transition.is_some_and(|previous| previous >= transition_identity)
                || !transition_ids.insert(transition_identity)
                || transition.observed_next_state.len() != dimension
                || transition
                    .observed_next_state
                    .iter()
                    .any(|value| !value.is_finite())
                || !transition.pre_target_error.is_finite()
                || !transition.post_target_error.is_finite()
                || transition.pre_target_error < 0.0
                || transition.post_target_error < 0.0
            {
                return Err(BrainError::Invalid("operator_ir_transition_invalid".into()));
            }
            let source = anchors
                .get(transition.source_anchor_id.as_str())
                .ok_or_else(|| BrainError::Integrity("operator_ir_source_anchor_unknown".into()))?;
            let target = anchors
                .get(transition.target_anchor_id.as_str())
                .ok_or_else(|| BrainError::Integrity("operator_ir_target_anchor_unknown".into()))?;
            if source.len() != dimension {
                return Err(BrainError::Integrity(
                    "operator_ir_source_anchor_invalid".into(),
                ));
            }
            let closure = transition
                .observed_next_state
                .iter()
                .zip(*target)
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f64>()
                .sqrt();
            if !closure.is_finite() || closure > self.maximum_closure_error {
                return Err(BrainError::Integrity(
                    "operator_ir_closure_contract_failed".into(),
                ));
            }
            let ratio = if transition.pre_target_error <= 1e-15 {
                if transition.post_target_error <= 1e-15 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                transition.post_target_error / transition.pre_target_error
            };
            if !ratio.is_finite() || ratio > self.maximum_contraction_ratio {
                return Err(BrainError::Integrity(
                    "operator_ir_contraction_contract_failed".into(),
                ));
            }
            previous_transition = Some(transition_identity);
        }
        Ok(())
    }

    /// Receiver-independent signature used by a receiver compiler. It is the
    /// concatenation of the exact target `StateIR` anchors for each canonical
    /// transition. Donor parameter coordinates never participate in this
    /// representation.
    pub fn canonical_transition_signature(&self, ir: &CapabilityIr) -> BrainResult<Vec<f64>> {
        self.validate_against(ir)?;
        let anchors = self
            .anchors
            .iter()
            .map(|anchor| (anchor.anchor_id.as_str(), anchor.state.as_slice()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let dimension = usize::try_from(self.state_dimension)
            .map_err(|_| BrainError::Invalid("operational_state_dimension_overflow".into()))?;
        let mut signature = Vec::with_capacity(self.transitions.len().saturating_mul(dimension));
        for transition in &self.transitions {
            let target = anchors
                .get(transition.target_anchor_id.as_str())
                .ok_or_else(|| BrainError::Integrity("operator_ir_target_anchor_unknown".into()))?;
            signature.extend_from_slice(target);
        }
        Ok(signature)
    }

    /// Verify a receiver-produced functional signature against the same
    /// closure and contraction semantics used to seal V63 evidence.
    pub fn verify_receiver_signature(
        &self,
        ir: &CapabilityIr,
        receiver_signature: &[f64],
    ) -> BrainResult<OperationalInterfaceVerification> {
        self.validate_against(ir)?;
        let dimension = usize::try_from(self.state_dimension)
            .map_err(|_| BrainError::Invalid("operational_state_dimension_overflow".into()))?;
        let expected = self
            .transitions
            .len()
            .checked_mul(dimension)
            .ok_or_else(|| BrainError::Invalid("operational_signature_size_overflow".into()))?;
        if receiver_signature.len() != expected
            || receiver_signature.iter().any(|value| !value.is_finite())
        {
            return Err(BrainError::Invalid(
                "receiver_operational_signature_invalid".into(),
            ));
        }
        let anchors = self
            .anchors
            .iter()
            .map(|anchor| (anchor.anchor_id.as_str(), anchor.state.as_slice()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut max_closure = 0.0_f64;
        let mut max_contraction = 0.0_f64;
        for (transition, observed) in self
            .transitions
            .iter()
            .zip(receiver_signature.chunks_exact(dimension))
        {
            let source = anchors
                .get(transition.source_anchor_id.as_str())
                .ok_or_else(|| BrainError::Integrity("operator_ir_source_anchor_unknown".into()))?;
            let target = anchors
                .get(transition.target_anchor_id.as_str())
                .ok_or_else(|| BrainError::Integrity("operator_ir_target_anchor_unknown".into()))?;
            let closure = observed
                .iter()
                .zip(*target)
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f64>()
                .sqrt();
            let pre = source
                .iter()
                .zip(*target)
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f64>()
                .sqrt();
            let ratio = if pre <= 1e-15 {
                if closure <= 1e-15 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                closure / pre
            };
            if !closure.is_finite() || !ratio.is_finite() {
                return Err(BrainError::Numerical(
                    "receiver_operational_metric_non_finite".into(),
                ));
            }
            max_closure = max_closure.max(closure);
            max_contraction = max_contraction.max(ratio);
        }
        let closure_satisfied = max_closure <= self.maximum_closure_error;
        let contraction_satisfied = max_contraction <= self.maximum_contraction_ratio;
        Ok(OperationalInterfaceVerification {
            schema: "cerebro.tidex.operational_interface_verification/v1".into(),
            transition_count: self.transitions.len(),
            maximum_observed_closure_error: max_closure,
            maximum_observed_contraction_ratio: max_contraction,
            closure_satisfied,
            contraction_satisfied,
            allowed: closure_satisfied && contraction_satisfied,
        })
    }
}

/// Authenticate the unique canonical persisted form of an IR against the
/// exact retained system envelope it claims to represent.
pub fn authenticate_capability_ir(
    private_root: &Path,
    reference: &PrivateFileReference,
    envelope: &SystemEnvelope,
) -> BrainResult<CapabilityIr> {
    let root = verify_internal_private_root(private_root)?;
    let bytes = reference.read_verified_bounded(&root, MAX_CAPABILITY_IR_BYTES)?;
    let ir: CapabilityIr = serde_json::from_slice(&bytes)?;
    ir.validate_against(envelope)?;
    if reference.path != capability_ir_path(&root, ir.manifest_digest()) {
        return Err(BrainError::Integrity(
            "capability_ir_content_address_mismatch".into(),
        ));
    }
    if serde_json::to_vec(&ir)? != bytes {
        return Err(BrainError::Integrity(
            "capability_ir_noncanonical_encoding".into(),
        ));
    }
    Ok(ir)
}

fn capability_ir_path(root: &Path, digest: &CapabilityIrDigest) -> PathBuf {
    root.join("state/capability_ir/by-sha")
        .join(format!("{}.json", digest.as_str()))
}

fn resolve_reference<'a>(
    reference: &ValueReference,
    inputs: &'a std::collections::BTreeMap<PortId, TypedPort>,
    parameters: &'a std::collections::BTreeMap<PortId, TypedPort>,
    known_nodes: &'a std::collections::BTreeMap<CapabilityNodeId, TypedPort>,
) -> BrainResult<&'a TypedPort> {
    match reference {
        ValueReference::Input { name } if inputs.contains_key(name) => Ok(&inputs[name]),
        ValueReference::Input { .. } => {
            Err(BrainError::Integrity("capability_ir_input_unknown".into()))
        }
        ValueReference::Parameter { name } if parameters.contains_key(name) => {
            Ok(&parameters[name])
        }
        ValueReference::Parameter { .. } => Err(BrainError::Integrity(
            "capability_ir_parameter_unknown".into(),
        )),
        ValueReference::NodeOutput { node_id } if known_nodes.contains_key(node_id) => {
            Ok(&known_nodes[node_id])
        }
        ValueReference::NodeOutput { .. } => Err(BrainError::Integrity(
            "capability_ir_node_reference_invalid".into(),
        )),
    }
}

fn validate_shape_rule(
    rule: ShapeRule,
    operands: &[&TypedPort],
    output: &TypedPort,
) -> BrainResult<()> {
    let valid = match rule {
        ShapeRule::ElementwiseSame => {
            operands.len() == 2
                && operands[0].shape == operands[1].shape
                && output.shape == operands[0].shape
        }
        ShapeRule::TensorScale => {
            operands.len() == 2 && operands[1].shape.is_empty() && output.shape == operands[0].shape
        }
        ShapeRule::MatrixMultiply2d => {
            operands.len() == 2
                && operands[0].shape.len() == 2
                && operands[1].shape.len() == 2
                && operands[0].shape[1] == operands[1].shape[0]
                && output.shape == [operands[0].shape[0], operands[1].shape[1]]
        }
        ShapeRule::PreserveFirst => operands.len() == 1 && output.shape == operands[0].shape,
        ShapeRule::Scalar => output.shape.is_empty(),
    };
    if !valid {
        return Err(BrainError::Integrity(
            "capability_ir_operand_shape_mismatch".into(),
        ));
    }
    Ok(())
}

fn canonical_paths(paths: &[PathBuf], label: &str) -> BrainResult<BTreeSet<PathBuf>> {
    let mut result = BTreeSet::new();
    for path in paths {
        if path.as_os_str().is_empty()
            || path.is_absolute()
            || path.as_os_str().as_encoded_bytes().len() > 4_096
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            || !result.insert(path.clone())
        {
            return Err(BrainError::Invalid(format!("{label}_invalid")));
        }
    }
    Ok(result)
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::digest_domain(domain, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
    };
    use crate::identity::AcquisitionId;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn envelope() -> (PathBuf, SystemEnvelope) {
        let root = std::env::temp_dir().join(format!(
            "tidex-ir-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/memory.rs"), b"pub fn select() {}\n").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("ir-acquisition.v1").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ConservativeGeneratedArtifacts,
            AcquisitionBudget {
                max_files: 16,
                max_total_bytes: 1 << 20,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&root, &request).unwrap();
        (root, envelope)
    }

    fn node(inputs: Vec<ValueReference>) -> IrNode {
        IrNode {
            node_id: CapabilityNodeId::parse("node.select").unwrap(),
            primitive_id: PrimitiveId::parse("select.arg_max").unwrap(),
            inputs,
            output: TypedPort {
                name: PortId::parse("choice").unwrap(),
                value_type: ValueType::I64,
                shape: vec![],
            },
            provenance: vec![PathBuf::from("src/memory.rs")],
        }
    }

    fn output(node_id: &str) -> OutputBinding {
        OutputBinding {
            port: TypedPort {
                name: PortId::parse("selected_index").unwrap(),
                value_type: ValueType::I64,
                shape: vec![],
            },
            source: ValueReference::NodeOutput {
                node_id: CapabilityNodeId::parse(node_id).unwrap(),
            },
        }
    }

    fn weight_parameter(name: &str, shape: Vec<u64>) -> ParameterSlot {
        ParameterSlot::new(TypedPort::tensor_f64(PortId::parse(name).unwrap(), shape).unwrap())
            .unwrap()
    }

    fn weighted_node(parameter_name: &str) -> IrNode {
        IrNode::new(
            CapabilityNodeId::parse("node.weighted").unwrap(),
            PrimitiveId::parse("tensor.matmul").unwrap(),
            vec![
                ValueReference::Parameter {
                    name: PortId::parse(parameter_name).unwrap(),
                },
                ValueReference::Input {
                    name: PortId::parse("activation").unwrap(),
                },
            ],
            TypedPort::tensor_f64(PortId::parse("projected").unwrap(), vec![2, 1]).unwrap(),
            vec![PathBuf::from("src/memory.rs")],
        )
        .unwrap()
    }

    fn weighted_output() -> OutputBinding {
        OutputBinding::new(
            TypedPort::tensor_f64(PortId::parse("result").unwrap(), vec![2, 1]).unwrap(),
            ValueReference::NodeOutput {
                node_id: CapabilityNodeId::parse("node.weighted").unwrap(),
            },
        )
        .unwrap()
    }

    #[test]
    fn ir_is_closed_typed_and_bound_to_the_captured_tree() {
        let (root, envelope) = envelope();
        let ir = CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![TypedPort {
                name: PortId::parse("scores").unwrap(),
                value_type: ValueType::TensorF64,
                shape: vec![4],
            }],
            vec![node(vec![ValueReference::Input {
                name: PortId::parse("scores").unwrap(),
            }])],
            vec![output("node.select")],
        )
        .unwrap();
        ir.validate_against(&envelope).unwrap();
        assert!(!ir.manifest_sha256.is_draft());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resident_parameter_is_not_a_runtime_input_and_is_shape_checked() {
        let (root, envelope) = envelope();
        let ir = CapabilityIr::new_with_parameters(
            CapabilityId::parse("tensor.weighted:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![TypedPort::tensor_f64(PortId::parse("activation").unwrap(), vec![3, 1]).unwrap()],
            vec![weight_parameter("weights", vec![2, 3])],
            vec![weighted_node("weights")],
            vec![weighted_output()],
        )
        .unwrap();

        assert_eq!(ir.inputs().len(), 1);
        assert_eq!(ir.inputs()[0].name().as_str(), "activation");
        assert_eq!(ir.parameters().len(), 1);
        assert_eq!(ir.parameters()[0].port().name().as_str(), "weights");
        ir.validate_against(&envelope).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ir_rejects_missing_duplicate_dead_and_colliding_parameters() {
        let (root, envelope) = envelope();
        let capability_id = CapabilityId::parse("tensor.weighted:v1").unwrap();
        let primitive_set = PrimitiveSet::tidex_core_v1().unwrap();
        let runtime_inputs =
            vec![TypedPort::tensor_f64(PortId::parse("activation").unwrap(), vec![3, 1]).unwrap()];
        let weights = weight_parameter("weights", vec![2, 3]);

        assert!(CapabilityIr::new_with_parameters(
            capability_id.clone(),
            &envelope,
            primitive_set.clone(),
            runtime_inputs.clone(),
            vec![],
            vec![weighted_node("weights")],
            vec![weighted_output()],
        )
        .is_err());
        assert!(CapabilityIr::new_with_parameters(
            capability_id.clone(),
            &envelope,
            primitive_set.clone(),
            runtime_inputs.clone(),
            vec![weights.clone(), weights.clone()],
            vec![weighted_node("weights")],
            vec![weighted_output()],
        )
        .is_err());
        assert!(CapabilityIr::new_with_parameters(
            capability_id.clone(),
            &envelope,
            primitive_set.clone(),
            runtime_inputs.clone(),
            vec![
                weights.clone(),
                ParameterSlot::new(
                    TypedPort::scalar(PortId::parse("unused").unwrap(), ValueType::F64).unwrap(),
                )
                .unwrap(),
            ],
            vec![weighted_node("weights")],
            vec![weighted_output()],
        )
        .is_err());
        assert!(CapabilityIr::new_with_parameters(
            capability_id,
            &envelope,
            primitive_set,
            runtime_inputs,
            vec![weight_parameter("activation", vec![2, 3])],
            vec![weighted_node("activation")],
            vec![weighted_output()],
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ir_rejects_parameter_type_shape_budget_and_tamper() {
        let (root, envelope) = envelope();
        assert!(ParameterSlot::new(
            TypedPort::scalar(PortId::parse("flag").unwrap(), ValueType::Bool).unwrap()
        )
        .is_err());

        let runtime_inputs =
            vec![TypedPort::tensor_f64(PortId::parse("activation").unwrap(), vec![3, 1]).unwrap()];
        assert!(CapabilityIr::new_with_parameters(
            CapabilityId::parse("tensor.weighted:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            runtime_inputs.clone(),
            vec![ParameterSlot::new(
                TypedPort::scalar(PortId::parse("weights").unwrap(), ValueType::F64).unwrap(),
            )
            .unwrap()],
            vec![weighted_node("weights")],
            vec![weighted_output()],
        )
        .is_err());
        assert!(CapabilityIr::new_with_parameters(
            CapabilityId::parse("tensor.weighted:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            runtime_inputs.clone(),
            vec![weight_parameter("weights", vec![2, 4])],
            vec![weighted_node("weights")],
            vec![weighted_output()],
        )
        .is_err());
        assert!(CapabilityIr::new_with_parameters(
            CapabilityId::parse("tensor.weighted:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            runtime_inputs.clone(),
            vec![weight_parameter("weights", vec![2, 3]); MAX_IR_PARAMETERS + 1],
            vec![weighted_node("weights")],
            vec![weighted_output()],
        )
        .is_err());

        let mut tampered = CapabilityIr::new_with_parameters(
            CapabilityId::parse("tensor.weighted:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            runtime_inputs,
            vec![weight_parameter("weights", vec![2, 3])],
            vec![weighted_node("weights")],
            vec![weighted_output()],
        )
        .unwrap();
        tampered.parameters[0].port.shape = vec![2, 4];
        tampered.manifest_sha256 = tampered.calculate_digest().unwrap();
        assert!(tampered.validate_against(&envelope).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ir_rejects_primitive_injection_future_nodes_and_foreign_provenance() {
        let (root, envelope) = envelope();
        let primitive_set = PrimitiveSet::tidex_core_v1().unwrap();
        let inputs = vec![TypedPort {
            name: PortId::parse("scores").unwrap(),
            value_type: ValueType::TensorF64,
            shape: vec![4],
        }];
        let mut injected = node(vec![ValueReference::Input {
            name: PortId::parse("scores").unwrap(),
        }]);
        injected.primitive_id = PrimitiveId::parse("shell.execute").unwrap();
        assert!(CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            primitive_set.clone(),
            inputs.clone(),
            vec![injected],
            vec![output("node.select")]
        )
        .is_err());
        let mut future = node(vec![ValueReference::NodeOutput {
            node_id: CapabilityNodeId::parse("node.future").unwrap(),
        }]);
        future.provenance = vec![PathBuf::from("src/other.rs")];
        assert!(CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            primitive_set,
            inputs,
            vec![future],
            vec![output("node.select")]
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn primitive_profile_cannot_be_self_signed_by_a_donor() {
        let mut forged = PrimitiveSet::tidex_core_v1().unwrap();
        forged.primitives[0].primitive_id = PrimitiveId::parse("shell.execute").unwrap();
        forged.sha256 = forged.calculate_digest().unwrap();
        assert!(forged.verify().is_err());
    }

    #[test]
    fn ir_rejects_self_reference_wrong_type_shape_and_unbounded_shape() {
        let (root, envelope) = envelope();
        let primitive_set = PrimitiveSet::tidex_core_v1().unwrap();

        let self_referencing = node(vec![ValueReference::NodeOutput {
            node_id: CapabilityNodeId::parse("node.select").unwrap(),
        }]);
        assert!(CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            primitive_set.clone(),
            vec![],
            vec![self_referencing],
            vec![output("node.select")],
        )
        .is_err());

        assert!(CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            primitive_set.clone(),
            vec![TypedPort {
                name: PortId::parse("scores").unwrap(),
                value_type: ValueType::F64,
                shape: vec![],
            }],
            vec![node(vec![ValueReference::Input {
                name: PortId::parse("scores").unwrap(),
            }])],
            vec![output("node.select")],
        )
        .is_err());

        let add = IrNode {
            node_id: CapabilityNodeId::parse("node.add").unwrap(),
            primitive_id: PrimitiveId::parse("tensor.add").unwrap(),
            inputs: vec![
                ValueReference::Input {
                    name: PortId::parse("left").unwrap(),
                },
                ValueReference::Input {
                    name: PortId::parse("right").unwrap(),
                },
            ],
            output: TypedPort {
                name: PortId::parse("sum").unwrap(),
                value_type: ValueType::TensorF64,
                shape: vec![4],
            },
            provenance: vec![PathBuf::from("src/memory.rs")],
        };
        let tensor_output = OutputBinding {
            port: TypedPort {
                name: PortId::parse("result").unwrap(),
                value_type: ValueType::TensorF64,
                shape: vec![4],
            },
            source: ValueReference::NodeOutput {
                node_id: CapabilityNodeId::parse("node.add").unwrap(),
            },
        };
        assert!(CapabilityIr::new(
            CapabilityId::parse("tensor.add:v1").unwrap(),
            &envelope,
            primitive_set.clone(),
            vec![
                TypedPort {
                    name: PortId::parse("left").unwrap(),
                    value_type: ValueType::TensorF64,
                    shape: vec![4],
                },
                TypedPort {
                    name: PortId::parse("right").unwrap(),
                    value_type: ValueType::TensorF64,
                    shape: vec![5],
                },
            ],
            vec![add],
            vec![tensor_output],
        )
        .is_err());

        assert!(CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            primitive_set,
            vec![TypedPort {
                name: PortId::parse("scores").unwrap(),
                value_type: ValueType::TensorF64,
                shape: vec![MAX_TENSOR_ELEMENTS + 1],
            }],
            vec![node(vec![ValueReference::Input {
                name: PortId::parse("scores").unwrap(),
            }])],
            vec![output("node.select")],
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ir_rejects_dead_nodes_output_contract_mismatch_and_tampered_envelope() {
        let (root, mut envelope) = envelope();
        let primitive_set = PrimitiveSet::tidex_core_v1().unwrap();
        let inputs = vec![TypedPort {
            name: PortId::parse("scores").unwrap(),
            value_type: ValueType::TensorF64,
            shape: vec![4],
        }];
        let source = ValueReference::Input {
            name: PortId::parse("scores").unwrap(),
        };
        let first = node(vec![source.clone()]);
        let mut dead = node(vec![source]);
        dead.node_id = CapabilityNodeId::parse("node.dead").unwrap();
        dead.output.name = PortId::parse("dead_choice").unwrap();
        assert!(CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            primitive_set.clone(),
            inputs.clone(),
            vec![first.clone(), dead],
            vec![output("node.select")],
        )
        .is_err());

        let mut mismatched = output("node.select");
        mismatched.port.value_type = ValueType::Bool;
        assert!(CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            primitive_set.clone(),
            inputs.clone(),
            vec![first.clone()],
            vec![mismatched],
        )
        .is_err());

        let mut tampered = serde_json::to_value(&envelope).unwrap();
        let old_total = tampered["total_file_bytes"].as_u64().unwrap();
        tampered["total_file_bytes"] = serde_json::json!(old_total.saturating_add(1));
        envelope = serde_json::from_value(tampered).unwrap();
        assert!(CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            primitive_set,
            inputs,
            vec![first],
            vec![output("node.select")],
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn persisted_ir_rejects_aliases_and_noncanonical_json() {
        let (donor, envelope) = envelope();
        let ir = CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![TypedPort {
                name: PortId::parse("scores").unwrap(),
                value_type: ValueType::TensorF64,
                shape: vec![4],
            }],
            vec![node(vec![ValueReference::Input {
                name: PortId::parse("scores").unwrap(),
            }])],
            vec![output("node.select")],
        )
        .unwrap();

        let private = donor.with_extension("private");
        fs::create_dir_all(&private).unwrap();
        crate::security::secure_dir(&private).unwrap();
        let canonical = ir.persist(&private, &envelope).unwrap();
        assert_eq!(
            authenticate_capability_ir(&private, &canonical, &envelope).unwrap(),
            ir
        );

        let alias_path = private.join("state/capability_ir/alias.json");
        let canonical_bytes = serde_json::to_vec(&ir).unwrap();
        let alias_sha = write_or_verify_immutable(&private, &alias_path, &canonical_bytes).unwrap();
        let alias = PrivateFileReference::new(alias_path, alias_sha);
        assert!(authenticate_capability_ir(&private, &alias, &envelope).is_err());

        let noncanonical_private = donor.with_extension("noncanonical-private");
        fs::create_dir_all(&noncanonical_private).unwrap();
        crate::security::secure_dir(&noncanonical_private).unwrap();
        let pretty = serde_json::to_vec_pretty(&ir).unwrap();
        let path = capability_ir_path(&noncanonical_private, ir.manifest_digest());
        let sha = write_or_verify_immutable(&noncanonical_private, &path, &pretty).unwrap();
        let noncanonical = PrivateFileReference::new(path, sha);
        assert!(
            authenticate_capability_ir(&noncanonical_private, &noncanonical, &envelope).is_err()
        );

        fs::remove_dir_all(donor).unwrap();
        fs::remove_dir_all(private).unwrap();
        fs::remove_dir_all(noncanonical_private).unwrap();
    }

    fn linear_readout_fixture(envelope: &SystemEnvelope, dimension: u64) -> CapabilityIr {
        CapabilityIr::new_with_parameters(
            CapabilityId::parse("model.readout:v1").unwrap(),
            envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![TypedPort::tensor_f64(
                PortId::parse("runtime_input").unwrap(),
                vec![dimension, 1],
            )
            .unwrap()],
            vec![ParameterSlot::new(
                TypedPort::tensor_f64(
                    PortId::parse("resident_weights").unwrap(),
                    vec![1, dimension],
                )
                .unwrap(),
            )
            .unwrap()],
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
                TypedPort::tensor_f64(PortId::parse("mapped").unwrap(), vec![1, 1]).unwrap(),
                vec![PathBuf::from("src/memory.rs")],
            )
            .unwrap()],
            vec![OutputBinding::new(
                TypedPort::tensor_f64(PortId::parse("runtime_output").unwrap(), vec![1, 1])
                    .unwrap(),
                ValueReference::NodeOutput {
                    node_id: CapabilityNodeId::parse("node.linear_map").unwrap(),
                },
            )
            .unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn linear_readout_executes_existing_rectangular_ir_with_real_parameters() {
        let (root, envelope) = envelope();
        let ir = linear_readout_fixture(&envelope, 3);
        let weights = vec![2.0, -3.0, 0.5];
        let inputs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 2.0, 4.0],
        ];
        let execution = execute_linear_readout(&ir, &envelope, &weights, &inputs).unwrap();
        for (actual, expected) in execution.raw_margins.iter().zip([2.0, -3.0, 0.5, -2.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(execution.capability_ir_sha256, *ir.manifest_digest());
        assert_eq!(execution.operator_node_id.as_str(), "node.linear_map");
        assert_eq!(execution.parameter_port_id.as_str(), "resident_weights");
        assert_eq!(execution.input_dimension, 3);
        assert_eq!(execution.row_count, 4);
        assert!(execution.candidate_only);
        assert_eq!(
            serde_json::from_slice::<LinearReadoutExecution>(
                &serde_json::to_vec(&execution).unwrap()
            )
            .unwrap(),
            execution
        );

        // A structural slot does not pretend to authenticate values. Different
        // bound parameters produce different arithmetic and consumed-value IDs.
        let changed = execute_linear_readout(&ir, &envelope, &[4.0, -3.0, 0.5], &inputs).unwrap();
        assert_ne!(changed.weights_sha256, execution.weights_sha256);
        assert_eq!(changed.inputs_sha256, execution.inputs_sha256);
        assert_ne!(changed.raw_margins, execution.raw_margins);
        let reordered = inputs.into_iter().rev().collect::<Vec<_>>();
        let reordered = execute_linear_readout(&ir, &envelope, &weights, &reordered).unwrap();
        assert_ne!(reordered.inputs_sha256, execution.inputs_sha256);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linear_readout_rejects_tampered_ir_parameter_binding_and_envelope() {
        let (root, envelope) = envelope();
        let ir = linear_readout_fixture(&envelope, 3);
        let weights = [2.0, -3.0, 0.5];
        let inputs = [vec![1.0, 2.0, 4.0]];

        let mut tampered = ir.clone();
        tampered.nodes[0].node_id = CapabilityNodeId::parse("node.changed").unwrap();
        assert!(execute_linear_readout(&tampered, &envelope, &weights, &inputs).is_err());
        let mut tampered = ir.clone();
        tampered.parameters[0].port.name = PortId::parse("other_weights").unwrap();
        tampered.manifest_sha256 = tampered.calculate_digest().unwrap();
        assert!(execute_linear_readout(&tampered, &envelope, &weights, &inputs).is_err());

        let mut envelope_json = serde_json::to_value(&envelope).unwrap();
        let total = envelope_json["total_file_bytes"].as_u64().unwrap();
        envelope_json["total_file_bytes"] = serde_json::json!(total + 1);
        let invalid_envelope = serde_json::from_value(envelope_json).unwrap();
        assert!(execute_linear_readout(&ir, &invalid_envelope, &weights, &inputs).is_err());
        assert!(execute_linear_readout(&ir, &envelope, &[2.0, -3.0], &inputs).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linear_readout_rejects_authenticated_graphs_outside_scalar_matmul_profile() {
        let (root, envelope) = envelope();
        let mut transposed = linear_readout_fixture(&envelope, 3);
        transposed.nodes[0].inputs.swap(0, 1);
        transposed.nodes[0].output.shape = vec![3, 3];
        transposed.outputs[0].port.shape = vec![3, 3];
        transposed.manifest_sha256 = transposed.calculate_digest().unwrap();
        transposed.validate_against(&envelope).unwrap();
        assert!(execute_linear_readout(
            &transposed,
            &envelope,
            &[1.0, 2.0, 3.0],
            &[vec![4.0, 5.0, 6.0]]
        )
        .is_err());

        let mut added = linear_readout_fixture(&envelope, 1);
        added.nodes[0].primitive_id = PrimitiveId::parse("tensor.add").unwrap();
        added.manifest_sha256 = added.calculate_digest().unwrap();
        added.validate_against(&envelope).unwrap();
        assert!(execute_linear_readout(&added, &envelope, &[2.0], &[vec![3.0]]).is_err());

        let mut extra_output = linear_readout_fixture(&envelope, 1);
        let mut output = extra_output.outputs[0].clone();
        output.port.name = PortId::parse("another_output").unwrap();
        extra_output.outputs.push(output);
        extra_output.manifest_sha256 = extra_output.calculate_digest().unwrap();
        extra_output.validate_against(&envelope).unwrap();
        assert!(execute_linear_readout(&extra_output, &envelope, &[2.0], &[vec![3.0]]).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linear_readout_bounds_work_and_rejects_nonfinite_inputs_and_results() {
        let (root, envelope) = envelope();
        let ir = linear_readout_fixture(&envelope, 2);
        assert!(execute_linear_readout(&ir, &envelope, &[1.0, 2.0], &[]).is_err());
        assert!(
            execute_linear_readout(&ir, &envelope, &[1.0, 2.0], &vec![vec![0.0; 2]; 129]).is_err()
        );
        assert!(execute_linear_readout(&ir, &envelope, &[1.0, 2.0], &[vec![1.0]]).is_err());
        assert!(
            execute_linear_readout(&ir, &envelope, &[1.0, f64::NAN], &[vec![1.0, 2.0]]).is_err()
        );
        assert!(
            execute_linear_readout(&ir, &envelope, &[1.0, 2.0], &[vec![1.0, f64::INFINITY]])
                .is_err()
        );
        assert!(
            execute_linear_readout(&ir, &envelope, &[f64::MAX, f64::MAX], &[vec![2.0, 2.0]])
                .is_err()
        );
        let oversized = linear_readout_fixture(&envelope, 4_097);
        assert!(execute_linear_readout(
            &oversized,
            &envelope,
            &vec![0.0; 4_097],
            &[vec![0.0; 4_097]]
        )
        .is_err());

        // The admitted upper boundary is executed, not merely shape-checked.
        let boundary = linear_readout_fixture(&envelope, 4_096);
        let boundary_execution = execute_linear_readout(
            &boundary,
            &envelope,
            &vec![1.0; 4_096],
            &vec![vec![1.0; 4_096]; 128],
        )
        .unwrap();
        assert_eq!(boundary_execution.raw_margins, vec![4_096.0; 128]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn operational_contract_enforces_v63_closure_and_contraction() {
        let (root, envelope) = envelope();
        let ir = CapabilityIr::new(
            CapabilityId::parse("memory.select:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![TypedPort::tensor_f64(PortId::parse("scores").unwrap(), vec![4]).unwrap()],
            vec![node(vec![ValueReference::Input {
                name: PortId::parse("scores").unwrap(),
            }])],
            vec![output("node.select")],
        )
        .unwrap();
        let contract = OperationalCapabilityContract {
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
                    pre_target_error: 1.0,
                    post_target_error: 0.0001,
                },
                OperatorIrTransition {
                    operator_id: "toggle".into(),
                    source_anchor_id: "s1".into(),
                    target_anchor_id: "s0".into(),
                    observed_next_state: vec![1.0, 0.0],
                    pre_target_error: 1.0,
                    post_target_error: 0.0001,
                },
            ],
            maximum_closure_error: 0.001,
            maximum_contraction_ratio: 0.001,
        };
        contract.validate_against(&ir).unwrap();

        let mut bad_closure = contract.clone();
        bad_closure.transitions[0].observed_next_state = vec![0.1, 0.9];
        assert!(bad_closure.validate_against(&ir).is_err());

        let mut bad_contraction = contract;
        bad_contraction.transitions[0].post_target_error = 0.1;
        assert!(bad_contraction.validate_against(&ir).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
