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
}
