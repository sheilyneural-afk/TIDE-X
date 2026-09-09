//! Authenticated profiling of concrete receiver checkpoints.
//!
//! A receiver profile binds a semantic model identity to exact checkpoint,
//! configuration, tokenizer and physical parameter-topology identities. It is
//! deliberately separate from workspace::ModelProfile, which describes a
//! remote inference endpoint. The actuator consumes one SafeTensors file; a
//! standard Hugging Face sharded SafeTensors index is therefore normalized
//! first into an immutable content-addressed single-file checkpoint.

use crate::artifact::{sha256_file, verify_dvec_reference_under_root};
use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::block_tomography::{
    parameter_layout_digest, ParameterBlockLayout, ParameterLayoutAuthority,
};
use crate::digest::{ParameterLayoutDigest, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::{ArchitectureId, ModelId, SourceRevision, TensorId};
use crate::security::verify_internal_private_root;
use crate::weight_actuator::{
    inspect_model_safetensors, lora_target_family, lora_target_layer,
    normalize_sharded_safetensors, parameter_layout_for_tensors, receiver_delta_dtype_supported,
    LoraAdapterAxisReceipt, ModelParameterInventory, ShardedSafetensorsNormalizationInput,
    LORA_ADAPTER_AXIS_RECEIPT_SCHEMA, MODEL_PARAMETER_INVENTORY_SCHEMA,
    SHARDED_SAFETENSORS_NORMALIZATION_INPUT_SCHEMA,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const RECEIVER_MODEL_PROFILE_INPUT_SCHEMA: &str =
    "cerebro.tidex.receiver_model_profile_input/v1";
pub const RECEIVER_MODEL_PROFILE_SCHEMA: &str = "cerebro.tidex.receiver_model_profile/v1";
pub const RECEIVER_MODEL_PROFILE_RECEIPT_SCHEMA: &str =
    "cerebro.tidex.receiver_model_profile_receipt/v1";
const MAX_PROFILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;

fn invalid(code: impl Into<String>) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: impl Into<String>) -> BrainError {
    BrainError::Integrity(code.into())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverCheckpointFormat {
    SingleSafetensorsV1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverLoraFamily {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl ReceiverLoraFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
        }
    }

    fn parse(value: &str) -> BrainResult<Self> {
        match value {
            "q_proj" => Ok(Self::QProj),
            "k_proj" => Ok(Self::KProj),
            "v_proj" => Ok(Self::VProj),
            "o_proj" => Ok(Self::OProj),
            "gate_proj" => Ok(Self::GateProj),
            "up_proj" => Ok(Self::UpProj),
            "down_proj" => Ok(Self::DownProj),
            _ => Err(invalid("receiver_lora_family_unsupported")),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverAdaptationBlocker {
    TransformerLayerCountMissing,
    TransformerLayerCoverageMismatch,
    RequiredLoraFamiliesMissing,
    NoExactLoraTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverModelProfileInput {
    pub schema: String,
    pub model_id: ModelId,
    pub architecture_id: ArchitectureId,
    #[serde(default)]
    pub source_revision: Option<SourceRevision>,
    pub checkpoint_path: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverProfileSourceFile {
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct ReceiverArchitectureMetadata {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub transformer_layer_count: Option<usize>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub vocabulary_size: Option<usize>,
    #[serde(default)]
    pub maximum_position_embeddings: Option<usize>,
    #[serde(default)]
    pub tied_word_embeddings: Option<bool>,
    #[serde(default)]
    pub quantization_config_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverLoraTargetSpec {
    pub tensor_id: TensorId,
    pub physical_index: usize,
    pub transformer_layer: usize,
    pub family: ReceiverLoraFamily,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub parameter_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverAdaptationLayoutBinding {
    pub artifact: PrivateFileReference,
    pub parameter_layout_sha256: ParameterLayoutDigest,
    pub total_parameter_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverModelProfile {
    pub schema: String,
    pub model_id: ModelId,
    pub architecture_id: ArchitectureId,
    #[serde(default)]
    pub source_revision: Option<SourceRevision>,
    pub checkpoint_format: ReceiverCheckpointFormat,
    pub checkpoint: ReceiverProfileSourceFile,
    pub config: ReceiverProfileSourceFile,
    pub tokenizer: ReceiverProfileSourceFile,
    pub architecture: ReceiverArchitectureMetadata,
    pub inventory: ModelParameterInventory,
    pub parameter_topology_sha256: Sha256Digest,
    pub adaptation_abi_sha256: Sha256Digest,
    pub observed_transformer_layers: BTreeSet<usize>,
    pub dense_update_tensor_ids: Vec<TensorId>,
    pub dense_update_layout: ReceiverAdaptationLayoutBinding,
    pub lora_targets: Vec<ReceiverLoraTargetSpec>,
    #[serde(default)]
    pub exact_lora_layout: Option<ReceiverAdaptationLayoutBinding>,
    pub distributed_lora_eligible: bool,
    pub blockers: BTreeSet<ReceiverAdaptationBlocker>,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReceiverModelProfileReceipt {
    pub schema: String,
    pub profile: ReceiverModelProfile,
    pub profile_reference: PrivateFileReference,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Serialize)]
struct AdaptationAbiProjection<'a> {
    schema: &'static str,
    checkpoint_format: ReceiverCheckpointFormat,
    architecture_id: &'a ArchitectureId,
    config_sha256: &'a Sha256Digest,
    tokenizer_sha256: &'a Sha256Digest,
    parameter_topology_sha256: &'a Sha256Digest,
    dense_layout_sha256: &'a ParameterLayoutDigest,
    exact_lora_layout_sha256: Option<&'a ParameterLayoutDigest>,
    update_operator: &'static str,
    lora_surface: &'static str,
}

#[derive(Debug, Serialize)]
struct ParameterTopologyEntry<'a> {
    tensor_id: &'a TensorId,
    dtype: &'a str,
    shape: &'a [usize],
    parameter_count: u64,
}

#[derive(Debug)]
struct DerivedModelViews {
    topology_sha256: Sha256Digest,
    observed_layers: BTreeSet<usize>,
    dense_ids: Vec<TensorId>,
    dense_layout: ParameterBlockLayout,
    lora_targets: Vec<ReceiverLoraTargetSpec>,
    lora_layout: Option<ParameterBlockLayout>,
    blockers: BTreeSet<ReceiverAdaptationBlocker>,
    distributed_lora_eligible: bool,
}

fn canonical_source_file(path: &Path, label: &str) -> BrainResult<PathBuf> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(invalid(format!("{label}_path_invalid")));
    }
    let canonical =
        fs::canonicalize(path).map_err(|error| integrity(format!("{label}_unreadable:{error}")))?;
    let metadata = fs::symlink_metadata(&canonical)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(integrity(format!("{label}_not_regular_file")));
    }
    Ok(canonical)
}

fn inspect_source_file(path: &Path, label: &str) -> BrainResult<ReceiverProfileSourceFile> {
    let path = canonical_source_file(path, label)?;
    let metadata = fs::metadata(&path)?;
    if metadata.len() == 0 {
        return Err(invalid(format!("{label}_empty")));
    }
    Ok(ReceiverProfileSourceFile {
        sha256: sha256_file(&path)?,
        byte_len: metadata.len(),
        path,
    })
}

fn read_config(source: &ReceiverProfileSourceFile) -> BrainResult<Vec<u8>> {
    if source.byte_len == 0 || source.byte_len > MAX_CONFIG_BYTES {
        return Err(invalid("receiver_model_config_size_invalid"));
    }
    let bytes = fs::read(&source.path)?;
    if bytes.len() as u64 != source.byte_len || Sha256Digest::digest_bytes(&bytes) != source.sha256
    {
        return Err(integrity("receiver_model_config_changed"));
    }
    Ok(bytes)
}

fn json_usize(
    object: &serde_json::Map<String, Value>,
    names: &[&str],
) -> BrainResult<Option<usize>> {
    let mut selected = None;
    for name in names {
        let Some(value) = object.get(*name) else {
            continue;
        };
        let raw = value
            .as_u64()
            .ok_or_else(|| invalid(format!("receiver_model_config_{name}_invalid")))?;
        let value = usize::try_from(raw)
            .map_err(|_| invalid(format!("receiver_model_config_{name}_overflow")))?;
        if value == 0 || selected.is_some_and(|prior| prior != value) {
            return Err(invalid(format!("receiver_model_config_{name}_conflict")));
        }
        selected = Some(value);
    }
    Ok(selected)
}

fn architecture_metadata(bytes: &[u8]) -> BrainResult<ReceiverArchitectureMetadata> {
    let value: Value = serde_json::from_slice(bytes)?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid("receiver_model_config_not_object"))?;
    let model_type = object
        .get("model_type")
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| invalid("receiver_model_config_model_type_invalid"))
        })
        .transpose()?;
    let architectures = object
        .get("architectures")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| invalid("receiver_model_config_architectures_invalid"))?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|value| !value.trim().is_empty())
                        .map(str::to_string)
                        .ok_or_else(|| invalid("receiver_model_config_architecture_invalid"))
                })
                .collect::<BrainResult<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    if architectures.iter().collect::<BTreeSet<_>>().len() != architectures.len() {
        return Err(invalid("receiver_model_config_architectures_duplicate"));
    }
    let tied_word_embeddings = object
        .get("tie_word_embeddings")
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| invalid("receiver_model_config_weight_tying_invalid"))
        })
        .transpose()?;
    let quantization_config_sha256 = object
        .get("quantization_config")
        .map(|value| {
            serde_json::to_vec(value)
                .map(|bytes| {
                    Sha256Digest::digest_domain(b"CEREBRO:TIDEX:QUANTIZATION-CONFIG:v1\0", &bytes)
                })
                .map_err(BrainError::from)
        })
        .transpose()?;
    Ok(ReceiverArchitectureMetadata {
        model_type,
        architectures,
        transformer_layer_count: json_usize(
            object,
            &["num_hidden_layers", "n_layer", "num_layers", "n_layers"],
        )?,
        hidden_size: json_usize(object, &["hidden_size", "n_embd", "d_model"])?,
        vocabulary_size: json_usize(object, &["vocab_size"])?,
        maximum_position_embeddings: json_usize(
            object,
            &["max_position_embeddings", "n_positions", "max_seq_len"],
        )?,
        tied_word_embeddings,
        quantization_config_sha256,
    })
}

fn adaptation_abi_digest(
    architecture_id: &ArchitectureId,
    config_sha256: &Sha256Digest,
    tokenizer_sha256: &Sha256Digest,
    parameter_topology_sha256: &Sha256Digest,
    dense_layout_sha256: &ParameterLayoutDigest,
    exact_lora_layout_sha256: Option<&ParameterLayoutDigest>,
) -> BrainResult<Sha256Digest> {
    let projection = AdaptationAbiProjection {
        schema: "cerebro.tidex.receiver_adaptation_abi/v1",
        checkpoint_format: ReceiverCheckpointFormat::SingleSafetensorsV1,
        architecture_id,
        config_sha256,
        tokenizer_sha256,
        parameter_topology_sha256,
        dense_layout_sha256,
        exact_lora_layout_sha256,
        update_operator: "dense_add_decode_f32_round_f32/v1",
        lora_surface: "peft_unfused_layers_qkvo_gate_up_down/v1",
    };
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:RECEIVER-ADAPTATION-ABI:v1\0",
        &serde_json::to_vec(&projection)?,
    ))
}

fn topology_digest(inventory: &ModelParameterInventory) -> BrainResult<Sha256Digest> {
    let entries = inventory
        .tensors
        .iter()
        .map(|tensor| ParameterTopologyEntry {
            tensor_id: &tensor.tensor_id,
            dtype: &tensor.dtype,
            shape: &tensor.shape,
            parameter_count: tensor.parameter_count,
        })
        .collect::<Vec<_>>();
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:RECEIVER-PARAMETER-TOPOLOGY:v1\0",
        &serde_json::to_vec(&entries)?,
    ))
}

fn derive_model_views(
    inventory: &ModelParameterInventory,
    architecture: &ReceiverArchitectureMetadata,
) -> BrainResult<DerivedModelViews> {
    if inventory.schema != MODEL_PARAMETER_INVENTORY_SCHEMA
        || inventory.tensor_count != inventory.tensors.len()
        || inventory.tensors.is_empty()
    {
        return Err(invalid("receiver_model_inventory_invalid"));
    }
    let mut dense_ids = Vec::new();
    let mut lora_targets = Vec::new();
    let mut observed_layers = BTreeSet::new();
    let mut families_by_layer = BTreeMap::<usize, BTreeSet<ReceiverLoraFamily>>::new();
    let mut seen_ids = BTreeSet::new();
    for (physical_index, tensor) in inventory.tensors.iter().enumerate() {
        if !seen_ids.insert(tensor.tensor_id.clone()) {
            return Err(integrity("receiver_model_inventory_duplicate_tensor"));
        }
        if receiver_delta_dtype_supported(&tensor.dtype) {
            dense_ids.push(tensor.tensor_id.clone());
        }
        if tensor.shape.len() == 2 && receiver_delta_dtype_supported(&tensor.dtype) {
            if let (Ok(family), Ok(layer)) = (
                lora_target_family(&tensor.tensor_id),
                lora_target_layer(&tensor.tensor_id),
            ) {
                let family = ReceiverLoraFamily::parse(&family)?;
                observed_layers.insert(layer);
                families_by_layer.entry(layer).or_default().insert(family);
                lora_targets.push(ReceiverLoraTargetSpec {
                    tensor_id: tensor.tensor_id.clone(),
                    physical_index,
                    transformer_layer: layer,
                    family,
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    parameter_count: tensor.parameter_count,
                });
            }
        }
    }
    if dense_ids.is_empty() {
        return Err(invalid("receiver_model_no_dense_update_tensors"));
    }
    let dense_layout = parameter_layout_for_tensors(inventory, &dense_ids)?;
    let lora_layout = if lora_targets.is_empty() {
        None
    } else {
        Some(parameter_layout_for_tensors(
            inventory,
            &lora_targets
                .iter()
                .map(|target| target.tensor_id.clone())
                .collect::<Vec<_>>(),
        )?)
    };
    let mut blockers = BTreeSet::new();
    if lora_targets.is_empty() {
        blockers.insert(ReceiverAdaptationBlocker::NoExactLoraTargets);
    }
    let expected_layers = architecture
        .transformer_layer_count
        .map(|count| (0..count).collect::<BTreeSet<_>>());
    match expected_layers.as_ref() {
        None => {
            blockers.insert(ReceiverAdaptationBlocker::TransformerLayerCountMissing);
        }
        Some(expected) if expected != &observed_layers => {
            blockers.insert(ReceiverAdaptationBlocker::TransformerLayerCoverageMismatch);
        }
        Some(_) => {}
    }
    let required_families = [
        ReceiverLoraFamily::QProj,
        ReceiverLoraFamily::KProj,
        ReceiverLoraFamily::VProj,
        ReceiverLoraFamily::OProj,
        ReceiverLoraFamily::GateProj,
        ReceiverLoraFamily::UpProj,
        ReceiverLoraFamily::DownProj,
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    let required_families_missing = expected_layers.as_ref().is_none_or(|expected| {
        expected.iter().any(|layer| {
            families_by_layer
                .get(layer)
                .is_none_or(|observed| !required_families.is_subset(observed))
        })
    });
    if required_families_missing {
        blockers.insert(ReceiverAdaptationBlocker::RequiredLoraFamiliesMissing);
    }
    let distributed_lora_eligible = blockers.is_empty();
    Ok(DerivedModelViews {
        topology_sha256: topology_digest(inventory)?,
        observed_layers,
        dense_ids,
        dense_layout,
        lora_targets,
        lora_layout,
        blockers,
        distributed_lora_eligible,
    })
}

fn persist_layout_binding(
    root: &Path,
    layout: ParameterBlockLayout,
) -> BrainResult<ReceiverAdaptationLayoutBinding> {
    let parameter_layout_sha256 = parameter_layout_digest(&layout)?;
    let total_parameter_count = layout.total_parameter_count;
    let artifact = ParameterLayoutAuthority::for_internal_root(root)?.persist(layout)?;
    Ok(ReceiverAdaptationLayoutBinding {
        artifact,
        parameter_layout_sha256,
        total_parameter_count,
    })
}

fn authenticate_layout_binding(
    root: &Path,
    binding: &ReceiverAdaptationLayoutBinding,
) -> BrainResult<ParameterBlockLayout> {
    let authenticated = ParameterLayoutAuthority::for_internal_root(root)?
        .authenticate_canonical_binding(
            binding.artifact.clone(),
            &binding.parameter_layout_sha256,
            binding.total_parameter_count,
        )?;
    Ok(authenticated.artifact.layout)
}

fn expected_profile_path(root: &Path, digest: &Sha256Digest) -> PathBuf {
    root.join("state/receiver_model_profiles/by-sha")
        .join(format!("{digest}.json"))
}

fn validate_profile_contract(root: &Path, profile: &ReceiverModelProfile) -> BrainResult<()> {
    if profile.schema != RECEIVER_MODEL_PROFILE_SCHEMA
        || profile.checkpoint_format != ReceiverCheckpointFormat::SingleSafetensorsV1
        || profile.authorizes_promotion
        || profile.checkpoint.sha256 != profile.inventory.model_sha256
        || profile.checkpoint.byte_len != profile.inventory.model_byte_len
    {
        return Err(invalid("receiver_model_profile_contract_invalid"));
    }
    let derived = derive_model_views(&profile.inventory, &profile.architecture)?;
    let expected_abi = adaptation_abi_digest(
        &profile.architecture_id,
        &profile.config.sha256,
        &profile.tokenizer.sha256,
        &profile.parameter_topology_sha256,
        &profile.dense_update_layout.parameter_layout_sha256,
        profile
            .exact_lora_layout
            .as_ref()
            .map(|binding| &binding.parameter_layout_sha256),
    )?;
    if profile.parameter_topology_sha256 != derived.topology_sha256
        || profile.adaptation_abi_sha256 != expected_abi
        || profile.observed_transformer_layers != derived.observed_layers
        || profile.dense_update_tensor_ids != derived.dense_ids
        || profile.lora_targets != derived.lora_targets
        || profile.blockers != derived.blockers
        || profile.distributed_lora_eligible != derived.distributed_lora_eligible
    {
        return Err(integrity("receiver_model_profile_derived_view_mismatch"));
    }
    let dense_layout = authenticate_layout_binding(root, &profile.dense_update_layout)?;
    if dense_layout != derived.dense_layout {
        return Err(integrity("receiver_model_profile_dense_layout_mismatch"));
    }
    match (&profile.exact_lora_layout, derived.lora_layout) {
        (Some(binding), Some(expected)) => {
            if authenticate_layout_binding(root, binding)? != expected {
                return Err(integrity("receiver_model_profile_lora_layout_mismatch"));
            }
        }
        (None, None) => {}
        _ => {
            return Err(integrity(
                "receiver_model_profile_lora_layout_presence_mismatch",
            ))
        }
    }
    Ok(())
}

fn validate_live_sources(profile: &ReceiverModelProfile) -> BrainResult<()> {
    let checkpoint = inspect_source_file(&profile.checkpoint.path, "receiver_model_checkpoint")?;
    let config = inspect_source_file(&profile.config.path, "receiver_model_config")?;
    let tokenizer = inspect_source_file(&profile.tokenizer.path, "receiver_model_tokenizer")?;
    if checkpoint != profile.checkpoint
        || config != profile.config
        || tokenizer != profile.tokenizer
    {
        return Err(integrity("receiver_model_profile_source_changed"));
    }
    let inventory = inspect_model_safetensors(&profile.checkpoint.path)?;
    if inventory != profile.inventory {
        return Err(integrity("receiver_model_profile_inventory_changed"));
    }
    if architecture_metadata(&read_config(&config)?)? != profile.architecture {
        return Err(integrity("receiver_model_profile_config_changed"));
    }
    Ok(())
}

pub fn profile_receiver_model(
    root: &Path,
    input: &ReceiverModelProfileInput,
) -> BrainResult<ReceiverModelProfileReceipt> {
    let root = verify_internal_private_root(root)?;
    if input.schema != RECEIVER_MODEL_PROFILE_INPUT_SCHEMA {
        return Err(invalid("receiver_model_profile_input_schema_invalid"));
    }
    let checkpoint_path = if input
        .checkpoint_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
    {
        normalize_sharded_safetensors(
            &root,
            &ShardedSafetensorsNormalizationInput {
                schema: SHARDED_SAFETENSORS_NORMALIZATION_INPUT_SCHEMA.to_string(),
                index_path: input.checkpoint_path.clone(),
            },
        )?
        .normalization
        .normalized_checkpoint
        .path
    } else {
        input.checkpoint_path.clone()
    };
    let checkpoint = inspect_source_file(&checkpoint_path, "receiver_model_checkpoint")?;
    let config = inspect_source_file(&input.config_path, "receiver_model_config")?;
    let tokenizer = inspect_source_file(&input.tokenizer_path, "receiver_model_tokenizer")?;
    let inventory = inspect_model_safetensors(&checkpoint.path)?;
    if inventory.model_sha256 != checkpoint.sha256
        || inventory.model_byte_len != checkpoint.byte_len
    {
        return Err(integrity("receiver_model_checkpoint_inventory_mismatch"));
    }
    let architecture = architecture_metadata(&read_config(&config)?)?;
    let derived = derive_model_views(&inventory, &architecture)?;
    let dense_update_layout = persist_layout_binding(&root, derived.dense_layout)?;
    let exact_lora_layout = derived
        .lora_layout
        .map(|layout| persist_layout_binding(&root, layout))
        .transpose()?;
    let adaptation_abi_sha256 = adaptation_abi_digest(
        &input.architecture_id,
        &config.sha256,
        &tokenizer.sha256,
        &derived.topology_sha256,
        &dense_update_layout.parameter_layout_sha256,
        exact_lora_layout
            .as_ref()
            .map(|binding| &binding.parameter_layout_sha256),
    )?;
    let profile = ReceiverModelProfile {
        schema: RECEIVER_MODEL_PROFILE_SCHEMA.to_string(),
        model_id: input.model_id.clone(),
        architecture_id: input.architecture_id.clone(),
        source_revision: input.source_revision.clone(),
        checkpoint_format: ReceiverCheckpointFormat::SingleSafetensorsV1,
        checkpoint,
        config,
        tokenizer,
        architecture,
        inventory,
        parameter_topology_sha256: derived.topology_sha256,
        adaptation_abi_sha256,
        observed_transformer_layers: derived.observed_layers,
        dense_update_tensor_ids: derived.dense_ids,
        dense_update_layout,
        lora_targets: derived.lora_targets,
        exact_lora_layout,
        distributed_lora_eligible: derived.distributed_lora_eligible,
        blockers: derived.blockers,
        authorizes_promotion: false,
    };
    validate_profile_contract(&root, &profile)?;
    validate_live_sources(&profile)?;
    let bytes = serde_json::to_vec(&profile)?;
    let digest = Sha256Digest::digest_bytes(&bytes);
    let path = expected_profile_path(&root, &digest);
    let written = write_or_verify_immutable(&root, &path, &bytes)?;
    if written != digest {
        return Err(integrity("receiver_model_profile_write_mismatch"));
    }
    let profile_reference = PrivateFileReference::new(path, digest);
    let authenticated = authenticate_live_receiver_model_profile(&root, &profile_reference)?;
    Ok(ReceiverModelProfileReceipt {
        schema: RECEIVER_MODEL_PROFILE_RECEIPT_SCHEMA.to_string(),
        profile: authenticated,
        profile_reference,
        authorizes_promotion: false,
    })
}

pub fn authenticate_receiver_model_profile(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<ReceiverModelProfile> {
    let root = verify_internal_private_root(root)?;
    if reference.path != expected_profile_path(&root, &reference.sha256) {
        return Err(integrity("receiver_model_profile_path_not_canonical"));
    }
    let bytes = reference.read_verified_bounded(&root, MAX_PROFILE_BYTES)?;
    let profile: ReceiverModelProfile = serde_json::from_slice(&bytes)?;
    if serde_json::to_vec(&profile)? != bytes {
        return Err(integrity("receiver_model_profile_not_canonical"));
    }
    validate_profile_contract(&root, &profile)?;
    Ok(profile)
}

pub fn authenticate_live_receiver_model_profile(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<ReceiverModelProfile> {
    let profile = authenticate_receiver_model_profile(root, reference)?;
    validate_live_sources(&profile)?;
    Ok(profile)
}

/// Bind one freshly authenticated LoRA import to the exact receiver profile.
/// This accepts subset layouts but never topology-only compatibility.
pub(crate) fn validate_lora_axis_against_profile(
    root: &Path,
    profile_reference: &PrivateFileReference,
    receipt: &LoraAdapterAxisReceipt,
) -> BrainResult<ReceiverModelProfile> {
    let root = verify_internal_private_root(root)?;
    let profile = authenticate_live_receiver_model_profile(&root, profile_reference)?;
    if receipt.schema != LORA_ADAPTER_AXIS_RECEIPT_SCHEMA
        || receipt.base_model_sha256 != profile.checkpoint.sha256
        || receipt.target_tensors.is_empty()
        || receipt.dense_delta.parameter_count != receipt.parameter_layout.total_parameter_count
        || receipt.source_training_semantics_attested
        || receipt.authorizes_target_update_free_claim
        || receipt.authorizes_promotion
    {
        return Err(invalid("receiver_model_lora_binding_invalid"));
    }
    receipt.parameter_layout.validate()?;
    let expected_layout =
        parameter_layout_for_tensors(&profile.inventory, &receipt.target_tensors)?;
    if expected_layout != receipt.parameter_layout {
        return Err(integrity("receiver_model_lora_layout_mismatch"));
    }
    let targets = profile
        .lora_targets
        .iter()
        .map(|target| (&target.tensor_id, target))
        .collect::<BTreeMap<_, _>>();
    let mut families = BTreeSet::new();
    let mut layers = BTreeSet::new();
    for tensor_id in &receipt.target_tensors {
        let target = targets
            .get(tensor_id)
            .ok_or_else(|| integrity(format!("receiver_model_lora_target_missing:{tensor_id}")))?;
        families.insert(target.family.as_str().to_string());
        layers.insert(target.transformer_layer);
    }
    if families != receipt.target_families || layers != receipt.covered_transformer_layers {
        return Err(integrity("receiver_model_lora_coverage_mismatch"));
    }
    let canonical_delta_path = root
        .join("artifacts/deltas/by-sha")
        .join(format!("{}.dvec", receipt.dense_delta.sha256));
    if receipt.dense_delta.path != canonical_delta_path {
        return Err(integrity("receiver_model_lora_delta_path_not_canonical"));
    }
    verify_dvec_reference_under_root(&root, &receipt.dense_delta)?;
    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::secure_dir;
    use crate::weight_actuator::ModelTensorSpec;
    use serde_json::{json, Map};
    use std::fs::File;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        checkpoint: PathBuf,
        config: PathBuf,
        tokenizer: PathBuf,
    }

    impl Fixture {
        fn new(label: &str, value_offset: f32) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "tidex-model-profile-{label}-{}-{}-{nonce}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            secure_dir(&root).unwrap();
            let checkpoint = root.join("model.safetensors");
            let config = root.join("config.json");
            let tokenizer = root.join("tokenizer.json");
            Self::write_checkpoint(&checkpoint, value_offset);
            fs::write(
                &config,
                serde_json::to_vec(&json!({
                    "model_type":"tidex-test",
                    "architectures":["TidexForCausalLM"],
                    "num_hidden_layers":1,
                    "hidden_size":2,
                    "vocab_size":8,
                    "max_position_embeddings":128,
                    "tie_word_embeddings":false
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(
                &tokenizer,
                serde_json::to_vec(&json!({"version":"1.0","model":{"type":"test"}})).unwrap(),
            )
            .unwrap();
            Self {
                root,
                checkpoint,
                config,
                tokenizer,
            }
        }

        fn write_checkpoint(path: &Path, value_offset: f32) {
            let tensors = [
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    vec![1.0 + value_offset, 2.0, 3.0, 4.0],
                    vec![2, 2],
                ),
                (
                    "model.layers.0.self_attn.v_proj.weight",
                    vec![-1.0, -2.0, -3.0, -4.0 - value_offset],
                    vec![2, 2],
                ),
                (
                    "model.layers.0.input_layernorm.weight",
                    vec![1.0, 1.0],
                    vec![2],
                ),
            ];
            let mut data = Vec::new();
            let mut header = Map::new();
            header.insert("__metadata__".to_string(), json!({"format":"pt"}));
            for (name, values, shape) in tensors {
                let start = data.len();
                for value in values {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                header.insert(
                    name.to_string(),
                    json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
                );
            }
            let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
            let padding = (8 - header.len() % 8) % 8;
            header.extend(std::iter::repeat_n(b' ', padding));
            let mut output = File::create(path).unwrap();
            output
                .write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            output.write_all(&header).unwrap();
            output.write_all(&data).unwrap();
            output.sync_all().unwrap();
        }

        fn input(&self) -> ReceiverModelProfileInput {
            ReceiverModelProfileInput {
                schema: RECEIVER_MODEL_PROFILE_INPUT_SCHEMA.to_string(),
                model_id: ModelId::parse("receiver.test").unwrap(),
                architecture_id: ArchitectureId::parse("tidex.test").unwrap(),
                source_revision: None,
                checkpoint_path: self.checkpoint.clone(),
                config_path: self.config.clone(),
                tokenizer_path: self.tokenizer.clone(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn profile_binds_exact_bytes_topology_layout_and_abi() {
        let fixture = Fixture::new("identity", 0.0);
        let receipt = profile_receiver_model(&fixture.root, &fixture.input()).unwrap();
        assert_eq!(receipt.schema, RECEIVER_MODEL_PROFILE_RECEIPT_SCHEMA);
        assert!(!receipt.authorizes_promotion);
        assert_eq!(
            receipt.profile.observed_transformer_layers,
            BTreeSet::from([0])
        );
        assert!(receipt
            .profile
            .blockers
            .contains(&ReceiverAdaptationBlocker::RequiredLoraFamiliesMissing));
        assert!(!receipt.profile.distributed_lora_eligible);
        assert_eq!(
            authenticate_live_receiver_model_profile(&fixture.root, &receipt.profile_reference)
                .unwrap(),
            receipt.profile
        );
        assert_ne!(
            receipt.profile.parameter_topology_sha256,
            receipt.profile.checkpoint.sha256
        );
        assert_ne!(
            receipt.profile.adaptation_abi_sha256,
            receipt.profile.parameter_topology_sha256
        );
    }

    #[test]
    fn topology_is_weight_independent_but_profile_and_model_identity_are_not() {
        let first = Fixture::new("topology-a", 0.0);
        let second = Fixture::new("topology-b", 9.0);
        let first_profile = profile_receiver_model(&first.root, &first.input())
            .unwrap()
            .profile;
        let second_profile = profile_receiver_model(&second.root, &second.input())
            .unwrap()
            .profile;
        assert_eq!(
            first_profile.parameter_topology_sha256,
            second_profile.parameter_topology_sha256
        );
        assert_eq!(
            first_profile.adaptation_abi_sha256,
            second_profile.adaptation_abi_sha256
        );
        assert_ne!(
            first_profile.checkpoint.sha256,
            second_profile.checkpoint.sha256
        );
    }

    #[test]
    fn static_profile_survives_but_live_authentication_detects_source_drift() {
        let fixture = Fixture::new("drift", 0.0);
        let receipt = profile_receiver_model(&fixture.root, &fixture.input()).unwrap();
        fs::write(&fixture.tokenizer, b"{\"changed\":true}").unwrap();
        authenticate_receiver_model_profile(&fixture.root, &receipt.profile_reference).unwrap();
        assert!(authenticate_live_receiver_model_profile(
            &fixture.root,
            &receipt.profile_reference
        )
        .is_err());
    }

    #[test]
    fn lora_eligibility_requires_every_family_in_every_layer() {
        let tensor_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ];
        let tensors = tensor_names
            .into_iter()
            .map(|name| ModelTensorSpec {
                tensor_id: TensorId::parse(name).unwrap(),
                dtype: "F32".to_string(),
                shape: vec![2, 2],
                parameter_count: 4,
                data_byte_count: 16,
            })
            .collect::<Vec<_>>();
        let inventory = ModelParameterInventory {
            schema: MODEL_PARAMETER_INVENTORY_SCHEMA.to_string(),
            model_sha256: Sha256Digest::digest_bytes(b"per-layer-coverage"),
            model_byte_len: 7 * 16,
            tensor_count: tensors.len(),
            total_parameter_count: 7 * 4,
            tensors,
        };
        let architecture = ReceiverArchitectureMetadata {
            transformer_layer_count: Some(2),
            ..ReceiverArchitectureMetadata::default()
        };

        let derived = derive_model_views(&inventory, &architecture).unwrap();

        assert_eq!(derived.observed_layers, BTreeSet::from([0, 1]));
        assert!(derived
            .blockers
            .contains(&ReceiverAdaptationBlocker::RequiredLoraFamiliesMissing));
        assert!(!derived.distributed_lora_eligible);
    }

    #[test]
    fn static_profile_rejects_noncanonical_json_identity() {
        let fixture = Fixture::new("noncanonical", 0.0);
        let receipt = profile_receiver_model(&fixture.root, &fixture.input()).unwrap();
        let bytes = serde_json::to_vec_pretty(&receipt.profile).unwrap();
        let digest = Sha256Digest::digest_bytes(&bytes);
        let path = expected_profile_path(&fixture.root, &digest);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        let reference = PrivateFileReference::new(path, digest);

        let error = authenticate_receiver_model_profile(&fixture.root, &reference)
            .unwrap_err()
            .to_string();

        assert!(error.contains("receiver_model_profile_not_canonical"));
    }

    #[test]
    fn sharded_checkpoint_is_normalized_and_profile_survives_source_removal() {
        let fixture = Fixture::new("sharded", 0.0);
        let source = fixture.root.with_extension("sharded-source");
        let _ = fs::remove_dir_all(&source);
        fs::create_dir_all(&source).unwrap();

        fn write_shard(path: &Path, tensors: &[(&str, Vec<f32>, Vec<usize>)]) {
            let mut data = Vec::new();
            let mut header = Map::new();
            header.insert("__metadata__".to_string(), json!({"format":"pt"}));
            for (name, values, shape) in tensors {
                let start = data.len();
                for value in values {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                header.insert(
                    (*name).to_string(),
                    json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
                );
            }
            let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
            let padding = (8 - header.len() % 8) % 8;
            header.extend(std::iter::repeat_n(b' ', padding));
            let mut file = File::create(path).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&header).unwrap();
            file.write_all(&data).unwrap();
            file.sync_all().unwrap();
        }

        let first = source.join("model-00001-of-00002.safetensors");
        let second = source.join("model-00002-of-00002.safetensors");
        write_shard(
            &first,
            &[
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    vec![1.0, 2.0, 3.0, 4.0],
                    vec![2, 2],
                ),
                (
                    "model.layers.0.input_layernorm.weight",
                    vec![1.0, 1.0],
                    vec![2],
                ),
            ],
        );
        write_shard(
            &second,
            &[(
                "model.layers.0.self_attn.v_proj.weight",
                vec![-1.0, -2.0, -3.0, -4.0],
                vec![2, 2],
            )],
        );
        let index = source.join("model.safetensors.index.json");
        fs::write(
            &index,
            serde_json::to_vec_pretty(&json!({
                "metadata":{"total_size":40},
                "weight_map":{
                    "model.layers.0.self_attn.q_proj.weight":"model-00001-of-00002.safetensors",
                    "model.layers.0.input_layernorm.weight":"model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.v_proj.weight":"model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut input = fixture.input();
        input.checkpoint_path = index;
        let receipt = profile_receiver_model(&fixture.root, &input).unwrap();
        assert_eq!(
            receipt.profile.checkpoint_format,
            ReceiverCheckpointFormat::SingleSafetensorsV1
        );
        assert_eq!(receipt.profile.inventory.tensor_count, 3);
        assert!(receipt
            .profile
            .checkpoint
            .path
            .starts_with(fixture.root.join("artifacts/models/by-sha")));
        fs::remove_dir_all(source).unwrap();
        assert_eq!(
            authenticate_live_receiver_model_profile(&fixture.root, &receipt.profile_reference)
                .unwrap(),
            receipt.profile
        );
    }
}
