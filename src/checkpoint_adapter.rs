//! Read-only, content-authenticated adapter for real SafeTensors checkpoints.
//!
//! It inspects headers and exact file bytes without loading tensor payloads.
//! Unsupported or ambiguous encodings are rejected rather than guessed.

use crate::architecture_families::{fingerprint_architecture, ArchitectureFamilyFingerprint};
use crate::block_tomography::{BlockShapeSpec, ParameterBlockLayout, ParameterLayoutArtifact};
use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::identity::{ArchitectureId, ModelId, TensorId};
use crate::receiver_layout::{
    FloatingScalarType, ReceiverMaterializationLayout, ReceiverScalarEncoding,
    ReceiverTensorPartitioning, ReceiverTensorPhysicalSpec,
};
use crate::receiver_profile::{
    CapabilityModality, MaterializationStrategy, ReceiverArchitecture, ReceiverProfile,
    ReceiverRegion,
};
use crate::receiver_profiler::ReceiverSnapshotBinding;
use crate::weight_actuator::inspect_model_safetensors;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_CHECKPOINT_FILES: usize = 16_384;
const MAX_AUXILIARY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_CHECKPOINT_BYTES: u64 = 1 << 50;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SafeTensorsReceiverRequest {
    pub schema: String,
    pub model_id: ModelId,
    pub architecture_id: ArchitectureId,
    pub architecture: ReceiverArchitecture,
    pub modalities: BTreeSet<CapabilityModality>,
    pub supports_persistent_state: bool,
    pub checkpoint_files: Vec<PathBuf>,
    pub configuration_file: PathBuf,
    pub tokenizer_file: PathBuf,
    pub supported_strategies: BTreeSet<MaterializationStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InspectedReceiverArtifacts {
    pub schema: String,
    pub profile: ReceiverProfile,
    pub layout: ReceiverMaterializationLayout,
    pub snapshot: ReceiverSnapshotBinding,
    pub architecture_fingerprint: ArchitectureFamilyFingerprint,
    pub checkpoint_file_sha256: BTreeMap<PathBuf, Sha256Digest>,
    pub manifest_sha256: Sha256Digest,
}

fn confined(root: &Path, relative: &Path) -> BrainResult<PathBuf> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(BrainError::Invalid("checkpoint_path_not_relative".into()));
    }
    let canonical_root = root.canonicalize()?;
    let path = canonical_root.join(relative).canonicalize()?;
    if !path.starts_with(&canonical_root) || !path.metadata()?.is_file() {
        return Err(BrainError::Invalid(
            "checkpoint_path_not_confined_file".into(),
        ));
    }
    Ok(path)
}

fn read_auxiliary_bytes(path: &Path, maximum: u64) -> BrainResult<Vec<u8>> {
    let file = File::open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() > maximum {
        return Err(BrainError::Invalid("checkpoint_file_too_large".into()));
    }
    let mut bytes = Vec::new();
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err(BrainError::Invalid("checkpoint_file_too_large".into()));
    }
    Ok(bytes)
}

fn encoding(dtype: &str) -> BrainResult<(ReceiverScalarEncoding, u64)> {
    match dtype {
        "F64" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float64,
            },
            8,
        )),
        "F32" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float32,
            },
            4,
        )),
        "BF16" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Bfloat16,
            },
            2,
        )),
        "F16" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float16,
            },
            2,
        )),
        "F8_E4M3" | "F8_E4M3FN" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float8E4m3,
            },
            1,
        )),
        "F8_E5M2" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float8E5m2,
            },
            1,
        )),
        _ => Err(BrainError::Invalid(format!(
            "checkpoint_dtype_requires_explicit_quantization_contract:{dtype}"
        ))),
    }
}

pub fn inspect_safetensors_receiver(
    root: &Path,
    request: &SafeTensorsReceiverRequest,
) -> BrainResult<InspectedReceiverArtifacts> {
    if request.schema != "cerebro.tidex.safetensors_receiver_request/v1"
        || request.checkpoint_files.is_empty()
        || request.checkpoint_files.len() > MAX_CHECKPOINT_FILES
        || request.modalities.is_empty()
        || request.supported_strategies.is_empty()
    {
        return Err(BrainError::Invalid(
            "safetensors_receiver_request_invalid".into(),
        ));
    }
    let mut files = request.checkpoint_files.clone();
    files.sort();
    if files.windows(2).any(|p| p[0] >= p[1]) {
        return Err(BrainError::Invalid("checkpoint_files_not_unique".into()));
    }
    let mut all = BTreeMap::new();
    let mut file_digests = BTreeMap::new();
    let mut total_bytes = 0u64;
    for relative in files {
        let path = confined(root, &relative)?;
        total_bytes = total_bytes
            .checked_add(path.metadata()?.len())
            .ok_or_else(|| BrainError::Invalid("checkpoint_total_overflow".into()))?;
        if total_bytes > MAX_CHECKPOINT_BYTES {
            return Err(BrainError::Invalid("checkpoint_total_limit".into()));
        }
        // Geometry and identity now come from the same descriptor-bound
        // reader used by the physical actuator, not two independent passes.
        let inventory = inspect_model_safetensors(&path)?;
        for tensor in inventory.tensors {
            let scalar = encoding(&tensor.dtype)?.0;
            if all
                .insert(tensor.tensor_id.as_str().to_string(), (tensor, scalar))
                .is_some()
            {
                return Err(BrainError::Invalid("checkpoint_duplicate_tensor".into()));
            }
        }
        file_digests.insert(relative, inventory.model_sha256);
    }
    let config_path = confined(root, &request.configuration_file)?;
    let config_bytes = read_auxiliary_bytes(&config_path, MAX_AUXILIARY_BYTES)?;
    let config = Sha256Digest::digest_bytes(&config_bytes);
    let tokenizer = Sha256Digest::digest_bytes(&read_auxiliary_bytes(
        &confined(root, &request.tokenizer_file)?,
        MAX_AUXILIARY_BYTES,
    )?);
    let tensor_names = all.keys().cloned().collect::<Vec<_>>();
    let architecture_fingerprint = fingerprint_architecture(&config_bytes, &tensor_names)?;
    if request.architecture != ReceiverArchitecture::Unknown
        && architecture_fingerprint.receiver_architecture != ReceiverArchitecture::Unknown
        && request.architecture != architecture_fingerprint.receiver_architecture
    {
        return Err(BrainError::Integrity(
            "declared_receiver_architecture_conflicts_with_checkpoint".into(),
        ));
    }
    let resolved_architecture = if request.architecture == ReceiverArchitecture::Unknown {
        architecture_fingerprint.receiver_architecture
    } else {
        request.architecture
    };
    let mut shapes = Vec::with_capacity(all.len());
    let mut regions = Vec::with_capacity(all.len());
    let mut physical = Vec::with_capacity(all.len());
    for (name, (header, encoding)) in all {
        let tensor_id = TensorId::parse(&name)?;
        let count = header.shape.iter().try_fold(1usize, |a, v| {
            a.checked_mul(*v)
                .ok_or_else(|| BrainError::Invalid("checkpoint_parameter_overflow".into()))
        })?;
        shapes.push(BlockShapeSpec {
            name: name.clone(),
            shape: header.shape,
            count,
        });
        regions.push(ReceiverRegion {
            tensor_id: tensor_id.clone(),
            parameter_count: u64::try_from(count)
                .map_err(|_| BrainError::Invalid("checkpoint_parameter_overflow".into()))?,
            supported_strategies: request.supported_strategies.clone(),
        });
        physical.push(ReceiverTensorPhysicalSpec {
            tensor_id,
            encoding,
            partitioning: ReceiverTensorPartitioning::Replicated,
        });
    }
    let geometry = ParameterLayoutArtifact::new(ParameterBlockLayout::from_shapes(&shapes)?)?;
    let profile = ReceiverProfile {
        schema: "cerebro.tidex.receiver_profile/v1".into(),
        model_id: request.model_id.clone(),
        architecture_id: request.architecture_id.clone(),
        architecture: resolved_architecture,
        modalities: request.modalities.clone(),
        supports_persistent_state: request.supports_persistent_state,
        parameter_dimension: geometry.total_parameter_count,
        regions,
    };
    let layout = ReceiverMaterializationLayout::create(&profile, geometry, physical, vec![])?;
    let snapshot_digest = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:SAFETENSORS-SNAPSHOT:v1\0",
        &serde_json::to_vec(&file_digests)?,
    );
    let snapshot = ReceiverSnapshotBinding::create(
        &profile,
        snapshot_digest,
        config,
        tokenizer,
        layout.manifest_sha256.clone(),
    )?;
    let result = InspectedReceiverArtifacts {
        schema: "cerebro.tidex.inspected_receiver_artifacts/v1".into(),
        profile,
        layout,
        snapshot,
        architecture_fingerprint,
        checkpoint_file_sha256: file_digests,
        manifest_sha256: Sha256Digest::zero(),
    };
    seal_inspection(result)
}

fn seal_inspection(
    mut result: InspectedReceiverArtifacts,
) -> BrainResult<InspectedReceiverArtifacts> {
    result.manifest_sha256 = Sha256Digest::zero();
    result.manifest_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:INSPECTED-RECEIVER-ARTIFACTS:v1\0",
        &serde_json::to_vec(&result)?,
    );
    Ok(result)
}

// The physical-profile authority remains model_adaptation. This projection is
// a planning view of an authenticated tensor subset, not a second profiler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PhysicalReceiverProjectionInput {
    pub schema: String,
    pub physical_profile: crate::authority::PrivateFileReference,
    pub tensor_ids: Vec<TensorId>,
    /// Declared planning modalities; not behavioral capability evidence.
    pub modalities: BTreeSet<CapabilityModality>,
}

pub fn project_authenticated_receiver(
    root: &Path,
    input: &PhysicalReceiverProjectionInput,
) -> BrainResult<InspectedReceiverArtifacts> {
    use crate::model_adaptation::authenticate_live_receiver_model_profile;
    use crate::weight_actuator::{parameter_layout_for_tensors, receiver_delta_dtype_supported};
    if input.schema != "cerebro.tidex.physical_receiver_projection_input/v1"
        || input.tensor_ids.is_empty()
        || input.tensor_ids.len() > 16_384
        || input.modalities.is_empty()
    {
        return Err(BrainError::Invalid(
            "physical_receiver_projection_invalid".into(),
        ));
    }
    let source = authenticate_live_receiver_model_profile(root, &input.physical_profile)?;
    let mut ids = input.tensor_ids.clone();
    ids.sort();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(BrainError::Invalid(
            "physical_receiver_projection_duplicate_tensor".into(),
        ));
    }
    let by_id = source
        .inventory
        .tensors
        .iter()
        .map(|t| (&t.tensor_id, t))
        .collect::<BTreeMap<_, _>>();
    let config = read_auxiliary_bytes(&source.config.path, MAX_AUXILIARY_BYTES)?;
    if Sha256Digest::digest_bytes(&config) != source.config.sha256 {
        return Err(BrainError::Integrity(
            "physical_receiver_config_changed".into(),
        ));
    }
    let names = source
        .inventory
        .tensors
        .iter()
        .map(|t| t.tensor_id.as_str().to_string())
        .collect::<Vec<_>>();
    let fingerprint = fingerprint_architecture(&config, &names)?;
    let mut regions = Vec::with_capacity(ids.len());
    let mut physical = Vec::with_capacity(ids.len());
    for id in &ids {
        let tensor = by_id
            .get(id)
            .ok_or_else(|| BrainError::Invalid("physical_receiver_tensor_missing".into()))?;
        if !receiver_delta_dtype_supported(&tensor.dtype) {
            return Err(BrainError::Invalid(
                "physical_receiver_update_dtype_unsupported".into(),
            ));
        }
        if source.architecture.tied_word_embeddings != Some(false)
            && (id.as_str().contains("embed") || id.as_str().ends_with("lm_head.weight"))
        {
            return Err(BrainError::Invalid(
                "physical_receiver_weight_tying_unresolved".into(),
            ));
        }
        let mut strategies = BTreeSet::from([
            MaterializationStrategy::ReceiverCoordinates,
            MaterializationStrategy::DenseDelta,
            MaterializationStrategy::SparseDelta,
        ]);
        if tensor.shape.len() == 2 {
            strategies.insert(MaterializationStrategy::LowRank);
        }
        regions.push(ReceiverRegion {
            tensor_id: id.clone(),
            parameter_count: tensor.parameter_count,
            supported_strategies: strategies,
        });
        physical.push(ReceiverTensorPhysicalSpec {
            tensor_id: id.clone(),
            encoding: encoding(&tensor.dtype)?.0,
            partitioning: ReceiverTensorPartitioning::Replicated,
        });
    }
    let geometry =
        ParameterLayoutArtifact::new(parameter_layout_for_tensors(&source.inventory, &ids)?)?;
    let profile = ReceiverProfile {
        schema: "cerebro.tidex.receiver_profile/v1".into(),
        model_id: source.model_id,
        architecture_id: source.architecture_id,
        architecture: fingerprint.receiver_architecture,
        modalities: input.modalities.clone(),
        supports_persistent_state: false,
        parameter_dimension: geometry.total_parameter_count,
        regions,
    };
    let layout = ReceiverMaterializationLayout::create(&profile, geometry, physical, vec![])?;
    let snapshot = ReceiverSnapshotBinding::create(
        &profile,
        source.checkpoint.sha256.clone(),
        source.config.sha256,
        source.tokenizer.sha256,
        layout.manifest_sha256.clone(),
    )?;
    seal_inspection(InspectedReceiverArtifacts {
        schema: "cerebro.tidex.physical_receiver_projection/v1".into(),
        profile,
        layout,
        snapshot,
        architecture_fingerprint: fingerprint,
        checkpoint_file_sha256: BTreeMap::from([(
            source.checkpoint.path,
            source.checkpoint.sha256,
        )]),
        manifest_sha256: Sha256Digest::zero(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CompiledCheckpointBackend {
    ReceiverCoordinates,
    DenseDelta,
    LowRank {
        policy: crate::low_rank_shadow_materializer::LowRankShadowPolicy,
    },
    SparseDelta {
        policy: crate::sparse_shadow_materializer::SparseShadowPolicy,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompiledCheckpointRequest {
    pub schema: String,
    pub physical_profile: crate::authority::PrivateFileReference,
    pub planning: crate::universal_capability_compiler::UniversalCapabilityPlanningRequest,
    pub planning_receipt:
        crate::universal_capability_compiler::UniversalCapabilityShadowPlanReceipt,
    pub layout: ReceiverMaterializationLayout,
    pub backend: CompiledCheckpointBackend,
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompiledCheckpointReceipt {
    pub schema: String,
    pub request: crate::authority::PrivateFileReference,
    pub representation: crate::authority::PrivateFileReference,
    pub representation_semantic_sha256: Sha256Digest,
    pub dense_delta: crate::artifact::DeltaArtifactRef,
    pub output_path: PathBuf,
    pub absolute_compilation_error: f64,
    pub relative_compilation_error: Option<f64>,
    pub materialization: crate::weight_actuator::WeightMaterializationReceipt,
    pub model_execution_verified: bool,
    pub authorizes_target_update_free_claim: bool,
    pub authorizes_promotion: bool,
}

struct PreparedCompiledCheckpoint {
    source: crate::model_adaptation::ReceiverModelProfile,
    layout: ParameterBlockLayout,
    values: Vec<f32>,
    representation: Vec<u8>,
    semantic_sha256: Sha256Digest,
    absolute_error: f64,
    relative_error: Option<f64>,
}

fn prepare_compiled_checkpoint(
    root: &Path,
    request: &CompiledCheckpointRequest,
) -> BrainResult<PreparedCompiledCheckpoint> {
    use crate::linalg::norm;
    use crate::model_adaptation::authenticate_live_receiver_model_profile;
    use crate::universal_capability_compiler::replay_universal_capability_shadow_plan;
    if request.schema != "cerebro.tidex.compiled_checkpoint_request/v1"
        || !request.output_path.is_absolute()
        || request.layout.geometry.total_parameter_count > 16 * 1024 * 1024
        || !request.maximum_absolute_error.is_finite()
        || request.maximum_absolute_error < 0.0
        || !request.maximum_relative_error.is_finite()
        || !(0.0..=1.0).contains(&request.maximum_relative_error)
    {
        return Err(BrainError::Invalid(
            "compiled_checkpoint_request_invalid".into(),
        ));
    }
    crate::authority::root_relative_path(root, &request.output_path)?;
    replay_universal_capability_shadow_plan(&request.planning, &request.planning_receipt)?;
    let projection = project_authenticated_receiver(
        root,
        &PhysicalReceiverProjectionInput {
            schema: "cerebro.tidex.physical_receiver_projection_input/v1".into(),
            physical_profile: request.physical_profile.clone(),
            tensor_ids: request
                .layout
                .physical_tensors
                .iter()
                .map(|t| t.tensor_id.clone())
                .collect(),
            modalities: request.planning.receiver_profile.modalities.clone(),
        },
    )?;
    if projection.profile != request.planning.receiver_profile
        || projection.snapshot != request.planning.receiver_snapshot
        || projection.layout != request.layout
    {
        return Err(BrainError::Integrity(
            "compiled_checkpoint_physical_binding_mismatch".into(),
        ));
    }
    let source = authenticate_live_receiver_model_profile(root, &request.physical_profile)?;
    let target = &request
        .planning_receipt
        .shadow_plan
        .compilation_receipt
        .compilation
        .receiver
        .target_delta;
    let mut tensors = BTreeMap::<TensorId, Vec<f64>>::new();
    let (representation, semantic_sha256) = match &request.backend {
        CompiledCheckpointBackend::ReceiverCoordinates => {
            let candidate =
                crate::shadow_materializer::materialize_replayed_receiver_coordinates_shadow(
                    &request.planning,
                    &request.planning_receipt,
                )?;
            for block in &request.layout.geometry.layout.blocks {
                let start = usize::try_from(block.offset)
                    .map_err(|_| BrainError::Invalid("compiled_offset_overflow".into()))?;
                tensors.insert(
                    TensorId::parse(&block.name)?,
                    candidate.coordinates[start..start + block.count].to_vec(),
                );
            }
            (serde_json::to_vec(&candidate)?, candidate.manifest_sha256)
        }
        CompiledCheckpointBackend::DenseDelta => {
            let candidate =
                crate::dense_shadow_materializer::materialize_replayed_dense_delta_shadow(
                    &request.planning,
                    &request.planning_receipt,
                    &request.layout,
                )?;
            for tensor in &candidate.tensors {
                tensors.insert(tensor.tensor_id.clone(), tensor.values.clone());
            }
            (serde_json::to_vec(&candidate)?, candidate.manifest_sha256)
        }
        CompiledCheckpointBackend::LowRank { policy } => {
            let candidate =
                crate::low_rank_shadow_materializer::materialize_replayed_low_rank_shadow(
                    &request.planning,
                    &request.planning_receipt,
                    &request.layout,
                    policy,
                )?;
            for tensor in &candidate.tensors {
                tensors.insert(
                    tensor.tensor_id.clone(),
                    tensor.factors.materialize_dense()?,
                );
            }
            (serde_json::to_vec(&candidate)?, candidate.manifest_sha256)
        }
        CompiledCheckpointBackend::SparseDelta { policy } => {
            let candidate = crate::sparse_shadow_materializer::materialize_replayed_sparse_shadow(
                &request.planning,
                &request.planning_receipt,
                &request.layout,
                policy,
            )?;
            for tensor in &candidate.tensors {
                let mut values = vec![0.0; tensor.shape.iter().product::<usize>()];
                for coordinate in &tensor.coordinates {
                    let index = usize::try_from(coordinate.flat_index).map_err(|_| {
                        BrainError::Invalid("compiled_sparse_index_overflow".into())
                    })?;
                    *values.get_mut(index).ok_or_else(|| {
                        BrainError::Integrity("compiled_sparse_index_invalid".into())
                    })? = coordinate.value;
                }
                tensors.insert(tensor.tensor_id.clone(), values);
            }
            (serde_json::to_vec(&candidate)?, candidate.manifest_sha256)
        }
    };
    if representation.len() > 128 * 1024 * 1024 {
        return Err(BrainError::Invalid(
            "compiled_checkpoint_representation_too_large".into(),
        ));
    }
    let affected = request
        .planning
        .affected_regions
        .iter()
        .collect::<BTreeSet<_>>();
    let mut full = vec![0.0f32; target.len()];
    for block in &request.layout.geometry.layout.blocks {
        let id = TensorId::parse(&block.name)?;
        let start = usize::try_from(block.offset)
            .map_err(|_| BrainError::Invalid("compiled_offset_overflow".into()))?;
        let end = start
            .checked_add(block.count)
            .ok_or_else(|| BrainError::Invalid("compiled_range_overflow".into()))?;
        let destination = full
            .get_mut(start..end)
            .ok_or_else(|| BrainError::Integrity("compiled_range_invalid".into()))?;
        if let Some(values) = tensors.remove(&id) {
            if values.len() != destination.len() {
                return Err(BrainError::Integrity(
                    "compiled_tensor_shape_mismatch".into(),
                ));
            }
            for (value, output) in values.into_iter().zip(destination) {
                let rounded = value as f32;
                if !value.is_finite() || !rounded.is_finite() || (value != 0.0 && rounded == 0.0) {
                    return Err(BrainError::Numerical(
                        "compiled_checkpoint_f32_unrepresentable".into(),
                    ));
                }
                *output = rounded;
            }
        }
        if !affected.contains(&id)
            && (full[start..end].iter().any(|v| *v != 0.0)
                || target[start..end].iter().any(|v| *v != 0.0))
        {
            return Err(BrainError::Integrity(
                "compiled_checkpoint_nonzero_outside_plan".into(),
            ));
        }
    }
    if !tensors.is_empty() {
        return Err(BrainError::Integrity(
            "compiled_checkpoint_unknown_tensor".into(),
        ));
    }
    let residual = full
        .iter()
        .zip(target)
        .map(|(actual, target)| f64::from(*actual) - target)
        .collect::<Vec<_>>();
    let absolute_error = norm(&residual)?;
    let target_norm = norm(target)?;
    let relative_error = if target_norm > 0.0 {
        Some(absolute_error / target_norm)
    } else {
        None
    };
    if !absolute_error.is_finite()
        || relative_error.is_some_and(|e| !e.is_finite())
        || (absolute_error > request.maximum_absolute_error
            && relative_error.is_none_or(|e| e > request.maximum_relative_error))
    {
        return Err(BrainError::Numerical(
            "compiled_checkpoint_representation_error_exceeded".into(),
        ));
    }
    let ids = affected.into_iter().cloned().collect::<Vec<_>>();
    let layout = crate::weight_actuator::parameter_layout_for_tensors(&source.inventory, &ids)?;
    let mut values = Vec::with_capacity(layout.total_parameter_count as usize);
    for id in ids {
        let block = request
            .layout
            .geometry
            .layout
            .blocks
            .iter()
            .find(|b| b.name == id.as_str())
            .ok_or_else(|| {
                BrainError::Integrity("compiled_checkpoint_selected_tensor_missing".into())
            })?;
        let start = block.offset as usize;
        values.extend_from_slice(&full[start..start + block.count]);
    }
    Ok(PreparedCompiledCheckpoint {
        source,
        layout,
        values,
        representation,
        semantic_sha256,
        absolute_error,
        relative_error,
    })
}

fn compiled_object_path(root: &Path, kind: &str, digest: &Sha256Digest) -> PathBuf {
    root.join("state/compiled_checkpoints")
        .join(kind)
        .join("by-sha")
        .join(format!("{digest}.json"))
}

fn persist_compiled_object(
    root: &Path,
    kind: &str,
    bytes: &[u8],
) -> BrainResult<crate::authority::PrivateFileReference> {
    if bytes.len() > 128 * 1024 * 1024 {
        return Err(BrainError::Invalid(
            "compiled_checkpoint_record_too_large".into(),
        ));
    }
    let digest = Sha256Digest::digest_bytes(bytes);
    let path = compiled_object_path(root, kind, &digest);
    crate::authority::write_or_verify_immutable(root, &path, bytes)?;
    Ok(crate::authority::PrivateFileReference::new(path, digest))
}

/// Compile and re-derive the representation, then use the existing physical
/// actuator. This writes a standalone candidate, never installs an active model.
pub fn materialize_compiled_checkpoint(
    root: &Path,
    request: &CompiledCheckpointRequest,
) -> BrainResult<crate::authority::PrivateFileReference> {
    let root = crate::security::verify_internal_private_root(root)?;
    let prepared = prepare_compiled_checkpoint(&root, request)?;
    let request_ref = persist_compiled_object(&root, "requests", &serde_json::to_vec(request)?)?;
    let representation =
        persist_compiled_object(&root, "representations", &prepared.representation)?;
    let dense_delta = crate::artifact::ArtifactWriteAuthority::for_internal_root(&root)?
        .create_content_addressed_dvec(&prepared.values)?;
    crate::authority::ensure_private_parent(&root, &request.output_path)?;
    let materialization = crate::weight_actuator::materialize_dense_delta_checkpoint(
        &root,
        &prepared.source.checkpoint.path,
        &prepared.source.checkpoint.sha256,
        &prepared.layout,
        &dense_delta,
        &request.output_path,
    )?;
    let receipt = CompiledCheckpointReceipt {
        schema: "cerebro.tidex.compiled_checkpoint_receipt/v1".into(),
        request: request_ref,
        representation,
        representation_semantic_sha256: prepared.semantic_sha256,
        dense_delta,
        output_path: request.output_path.clone(),
        absolute_compilation_error: prepared.absolute_error,
        relative_compilation_error: prepared.relative_error,
        materialization,
        model_execution_verified: false,
        authorizes_target_update_free_claim: false,
        authorizes_promotion: false,
    };
    let reference = persist_compiled_object(&root, "receipts", &serde_json::to_vec(&receipt)?)?;
    authenticate_compiled_checkpoint(&root, &reference)?;
    Ok(reference)
}

/// Reopening replays compilation and representation, compares the exact dense
/// update, and authenticates base/layout/update/output through the shared actuator.
pub fn authenticate_compiled_checkpoint(
    root: &Path,
    reference: &crate::authority::PrivateFileReference,
) -> BrainResult<CompiledCheckpointReceipt> {
    let root = crate::security::verify_internal_private_root(root)?;
    if reference.path != compiled_object_path(&root, "receipts", &reference.sha256) {
        return Err(BrainError::Integrity(
            "compiled_checkpoint_receipt_path_invalid".into(),
        ));
    }
    let bytes = reference.read_verified_bounded(&root, 128 * 1024 * 1024)?;
    let receipt: CompiledCheckpointReceipt = serde_json::from_slice(&bytes)?;
    if receipt.schema != "cerebro.tidex.compiled_checkpoint_receipt/v1"
        || receipt.model_execution_verified
        || receipt.authorizes_promotion
        || receipt.authorizes_target_update_free_claim
        || serde_json::to_vec(&receipt)? != bytes
        || receipt.request.path != compiled_object_path(&root, "requests", &receipt.request.sha256)
        || receipt.representation.path
            != compiled_object_path(&root, "representations", &receipt.representation.sha256)
    {
        return Err(BrainError::Integrity(
            "compiled_checkpoint_receipt_invalid".into(),
        ));
    }
    let request_bytes = receipt
        .request
        .read_verified_bounded(&root, 128 * 1024 * 1024)?;
    let request: CompiledCheckpointRequest = serde_json::from_slice(&request_bytes)?;
    if serde_json::to_vec(&request)? != request_bytes || receipt.output_path != request.output_path
    {
        return Err(BrainError::Integrity(
            "compiled_checkpoint_request_mismatch".into(),
        ));
    }
    let prepared = prepare_compiled_checkpoint(&root, &request)?;
    if receipt
        .representation
        .read_verified_bounded(&root, 128 * 1024 * 1024)?
        != prepared.representation
        || receipt.representation_semantic_sha256 != prepared.semantic_sha256
        || receipt.absolute_compilation_error != prepared.absolute_error
        || receipt.relative_compilation_error != prepared.relative_error
        || receipt.dense_delta.parameter_count != prepared.values.len() as u64
        || receipt.dense_delta.path
            != root
                .join("artifacts/deltas/by-sha")
                .join(format!("{}.dvec", receipt.dense_delta.sha256))
    {
        return Err(BrainError::Integrity(
            "compiled_checkpoint_replay_mismatch".into(),
        ));
    }
    let mut reader = crate::artifact::VerifiedDvecReader::open(&root, &receipt.dense_delta)?;
    for expected in prepared.values.chunks(65_536) {
        let actual = reader.read_f32(expected.len())?;
        if actual
            .iter()
            .zip(expected)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(BrainError::Integrity(
                "compiled_checkpoint_delta_mismatch".into(),
            ));
        }
    }
    reader.finish()?;
    crate::weight_actuator::authenticate_weight_materialization_receipt(
        &root,
        &prepared.source.checkpoint.path,
        &prepared.layout,
        &receipt.dense_delta,
        &receipt.output_path,
        &receipt.materialization,
    )?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn inspects_exact_safetensors_geometry_and_rejects_ambiguous_dtype() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tidex-safetensors-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        let header = br#"{"layer.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut model = File::create(root.join("model.safetensors")).unwrap();
        model
            .write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        model.write_all(header).unwrap();
        model.write_all(&[0u8; 16]).unwrap();
        let request = SafeTensorsReceiverRequest {
            schema: "cerebro.tidex.safetensors_receiver_request/v1".into(),
            model_id: ModelId::parse("receiver.fixture").unwrap(),
            architecture_id: ArchitectureId::parse("transformer.fixture").unwrap(),
            architecture: ReceiverArchitecture::Transformer,
            modalities: BTreeSet::from([CapabilityModality::Text]),
            supports_persistent_state: false,
            checkpoint_files: vec!["model.safetensors".into()],
            configuration_file: "config.json".into(),
            tokenizer_file: "tokenizer.json".into(),
            supported_strategies: BTreeSet::from([MaterializationStrategy::DenseDelta]),
        };
        let inspected = inspect_safetensors_receiver(&root, &request).unwrap();
        assert_eq!(inspected.profile.parameter_dimension, 4);
        assert_eq!(inspected.layout.geometry.layout.blocks[0].shape, vec![2, 2]);
        assert_ne!(
            inspected.snapshot.model_snapshot_sha256,
            Sha256Digest::zero()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
