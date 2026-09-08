//! Direct model-weight materialization for SafeTensors checkpoints.
//!
//! This module is deliberately a data-plane actuator, not a capability
//! compiler. It accepts only an already authenticated dense delta plus the
//! exact [`ParameterBlockLayout`] that gives that flat delta tensor semantics.
//! It never trains, calls a donor, interprets prompts, or authorizes promotion.
//! The result is a new candidate checkpoint with the selected resident tensors
//! modified directly; untouched tensors are copied byte-for-byte.

use crate::artifact::{
    sha256_file, verify_dvec_reference_under_root, ArtifactWriteAuthority, DeltaArtifactRef,
    VerifiedDvecReader,
};
use crate::authority::PrivateFileReference;
use crate::block_tomography::{parameter_layout_digest, BlockShapeSpec, ParameterBlockLayout};
use crate::digest::{ParameterLayoutDigest, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::TensorId;
use crate::security::verify_internal_private_root;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const MODEL_PARAMETER_INVENTORY_SCHEMA: &str = "cerebro.tidex.model_parameter_inventory/v1";
pub const WEIGHT_MATERIALIZATION_RECEIPT_SCHEMA: &str =
    "cerebro.tidex.weight_materialization_receipt/v1";
pub const LORA_ADAPTER_AXIS_INPUT_SCHEMA: &str = "cerebro.tidex.lora_adapter_axis_input/v1";
pub const LORA_ADAPTER_AXIS_RECEIPT_SCHEMA: &str = "cerebro.tidex.lora_adapter_axis_receipt/v1";
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;
const STREAM_ELEMENTS: usize = 262_144;
static NEXT_OUTPUT_TEMP: AtomicU64 = AtomicU64::new(0);

fn invalid(code: impl Into<String>) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: impl Into<String>) -> BrainError {
    BrainError::Integrity(code.into())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelTensorSpec {
    pub tensor_id: TensorId,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub parameter_count: u64,
    pub data_byte_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelParameterInventory {
    pub schema: String,
    pub model_sha256: Sha256Digest,
    pub model_byte_len: u64,
    pub tensor_count: usize,
    pub total_parameter_count: u64,
    /// Source tensor order by increasing SafeTensors data offset.
    pub tensors: Vec<ModelTensorSpec>,
}

/// Two rows of an authenticated resident linear readout, represented exactly
/// in f64 after decoding the checkpoint's F16, BF16 or F32 storage. This is a
/// parameter inspection, not evidence that a model executed a capability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LinearReadoutInspection {
    pub schema: String,
    pub input_dimension: usize,
    pub model_sha256: Sha256Digest,
    pub tensor_id: TensorId,
    pub positive_row: usize,
    pub negative_row: usize,
    pub positive_weights: Vec<f64>,
    pub negative_weights: Vec<f64>,
    /// Subtraction occurs in f64. A framework's F32 dot-product/subtraction
    /// order can differ numerically and must be verified at its own boundary.
    pub difference_weights: Vec<f64>,
}

/// Authenticated request for turning a calibration-only PEFT LoRA into one
/// receiver-native axis. This operation does not train the adapter and does
/// not assign it capability semantics; it only reconstructs its exact dense
/// effect over resident Transformer matrices.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LoraAdapterAxisInput {
    pub schema: String,
    pub base_model_path: PathBuf,
    pub adapter_model_path: PathBuf,
    pub adapter_config_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoraAdapterAxisReceipt {
    pub schema: String,
    pub base_model_path: PathBuf,
    pub adapter_model_path: PathBuf,
    pub adapter_config_path: PathBuf,
    pub base_model_sha256: Sha256Digest,
    pub adapter_model_sha256: Sha256Digest,
    pub adapter_config_sha256: Sha256Digest,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_scale: f64,
    pub use_rslora: bool,
    pub learned_parameter_count: u64,
    pub target_tensors: Vec<TensorId>,
    pub target_families: BTreeSet<String>,
    pub covered_transformer_layers: BTreeSet<usize>,
    pub parameter_layout: ParameterBlockLayout,
    pub dense_delta: DeltaArtifactRef,
    pub source_training_semantics_attested: bool,
    pub authorizes_target_update_free_claim: bool,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Deserialize)]
struct PeftLoraConfig {
    r: usize,
    lora_alpha: f64,
    target_modules: BTreeSet<String>,
    bias: String,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    modules_to_save: Option<Vec<String>>,
    #[serde(default)]
    rank_pattern: BTreeMap<String, usize>,
    #[serde(default)]
    alpha_pattern: BTreeMap<String, f64>,
}

#[derive(Debug)]
struct LoraAdapterPair {
    a: Option<TensorId>,
    b: Option<TensorId>,
}

#[derive(Debug)]
struct LoraPatchFactors {
    input_dim: usize,
    output_dim: usize,
    rank: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

struct LoraDenseDeltaIter<'a> {
    patches: &'a [LoraPatchFactors],
    scale: f32,
    patch: usize,
    element: usize,
}

impl<'a> LoraDenseDeltaIter<'a> {
    fn new(patches: &'a [LoraPatchFactors], scale: f32) -> Self {
        Self {
            patches,
            scale,
            patch: 0,
            element: 0,
        }
    }
}

impl Iterator for LoraDenseDeltaIter<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let patch = self.patches.get(self.patch)?;
            let count = patch.output_dim.checked_mul(patch.input_dim)?;
            if self.element >= count {
                self.patch += 1;
                self.element = 0;
                continue;
            }
            let row = self.element / patch.input_dim;
            let column = self.element % patch.input_dim;
            let mut value = 0.0f32;
            for component in 0..patch.rank {
                value += patch.b[row * patch.rank + component]
                    * patch.a[component * patch.input_dim + column];
            }
            self.element += 1;
            return Some(self.scale * value);
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WeightMaterializationLifecycle {
    CandidateOnlyNotPromoted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WeightMaterializationReceipt {
    pub schema: String,
    pub lifecycle: WeightMaterializationLifecycle,
    pub base_model_sha256: Sha256Digest,
    pub base_model_byte_len: u64,
    pub source_tensor_count: usize,
    pub source_parameter_count: u64,
    pub parameter_layout_sha256: ParameterLayoutDigest,
    pub delta_sha256: Sha256Digest,
    pub delta_parameter_count: u64,
    pub modified_tensor_count: usize,
    pub modified_parameter_count: u64,
    pub output_model_sha256: Sha256Digest,
    pub output_model_byte_len: u64,
    /// Modified tensors are promoted to F32 so a small functional delta is not
    /// silently rounded back into the base checkpoint's lower precision.
    pub dtype_policy: String,
    pub trains_target_capability: bool,
    pub requires_adapter_at_runtime: bool,
    pub authorizes_promotion: bool,
}

#[derive(Debug, Clone)]
struct ParsedTensor {
    spec: ModelTensorSpec,
    data_start: u64,
    data_end: u64,
}

struct SafeTensorArchive {
    file: File,
    data_start: u64,
    metadata: Option<BTreeMap<String, String>>,
    tensors: Vec<ParsedTensor>,
    index: BTreeMap<TensorId, usize>,
    inventory: ModelParameterInventory,
    // These digests and the inventory identity are committed by one pass over
    // the same raw header and tensor bytes. Later reads must match them.
    tensor_sha256: BTreeMap<TensorId, Sha256Digest>,
}

#[derive(Debug, Clone)]
struct OutputTensorPlan {
    tensor_id: TensorId,
    dtype: String,
    shape: Vec<usize>,
    source_index: usize,
    modified: bool,
    data_start: u64,
    data_end: u64,
}

fn canonical_existing_file(path: &Path) -> BrainResult<PathBuf> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(invalid("model_weight_source_path_invalid"));
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| integrity(format!("model_weight_source_unreadable:{error}")))?;
    let metadata = fs::symlink_metadata(&canonical)?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(integrity("model_weight_source_not_regular_file"));
    }
    Ok(canonical)
}

fn canonical_output_path(path: &Path) -> BrainResult<PathBuf> {
    if !path.is_absolute()
        || path.file_name().is_none()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(invalid("model_weight_output_path_invalid"));
    }
    let parent = path
        .parent()
        .ok_or_else(|| invalid("model_weight_output_parent_missing"))?;
    let parent = fs::canonicalize(parent)
        .map_err(|error| integrity(format!("model_weight_output_parent_unreadable:{error}")))?;
    let metadata = fs::symlink_metadata(&parent)?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(integrity("model_weight_output_parent_invalid"));
    }
    let output = parent.join(
        path.file_name()
            .ok_or_else(|| invalid("model_weight_output_name_missing"))?,
    );
    if fs::symlink_metadata(&output).is_ok() {
        return Err(integrity("model_weight_output_already_exists"));
    }
    Ok(output)
}

fn sha256_open_file(file: &mut File) -> BrainResult<Sha256Digest> {
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut buffer = [0u8; 1 << 20];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
    reader.into_inner().seek(SeekFrom::Start(0))?;
    Ok(digest)
}

fn dtype_bits(dtype: &str) -> Option<u64> {
    match dtype {
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Some(8),
        "I16" | "U16" | "F16" | "BF16" => Some(16),
        "I32" | "U32" | "F32" => Some(32),
        "C64" | "F64" | "I64" | "U64" => Some(64),
        _ => None,
    }
}

fn checked_parameter_count(shape: &[usize]) -> BrainResult<u64> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(invalid("safetensors_tensor_shape_invalid"));
    }
    let count = shape.iter().try_fold(1u64, |total, value| {
        total
            .checked_mul(*value as u64)
            .ok_or_else(|| invalid("safetensors_tensor_shape_overflow"))
    })?;
    Ok(count)
}

fn parse_shape(value: &Value) -> BrainResult<Vec<usize>> {
    let array = value
        .as_array()
        .ok_or_else(|| integrity("safetensors_tensor_shape_not_array"))?;
    let mut shape = Vec::with_capacity(array.len());
    for dimension in array {
        let value = dimension
            .as_u64()
            .ok_or_else(|| integrity("safetensors_tensor_dimension_invalid"))?;
        let value =
            usize::try_from(value).map_err(|_| invalid("safetensors_tensor_dimension_overflow"))?;
        shape.push(value);
    }
    checked_parameter_count(&shape)?;
    Ok(shape)
}

fn parse_offsets(value: &Value) -> BrainResult<(u64, u64)> {
    let array = value
        .as_array()
        .filter(|values| values.len() == 2)
        .ok_or_else(|| integrity("safetensors_tensor_offsets_invalid"))?;
    let start = array[0]
        .as_u64()
        .ok_or_else(|| integrity("safetensors_tensor_offset_invalid"))?;
    let end = array[1]
        .as_u64()
        .ok_or_else(|| integrity("safetensors_tensor_offset_invalid"))?;
    if end < start {
        return Err(integrity("safetensors_tensor_offsets_reversed"));
    }
    Ok((start, end))
}

impl SafeTensorArchive {
    fn open(path: &Path) -> BrainResult<Self> {
        let canonical = canonical_existing_file(path)?;
        let mut file = File::open(&canonical)?;
        let opened_metadata = file.metadata()?;
        if !opened_metadata.file_type().is_file() {
            return Err(integrity("model_weight_source_not_regular_file"));
        }
        let model_byte_len = opened_metadata.len();
        let mut header_len_raw = [0u8; 8];
        file.read_exact(&mut header_len_raw)?;
        let header_len = u64::from_le_bytes(header_len_raw);
        if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
            return Err(integrity("safetensors_header_size_invalid"));
        }
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| invalid("safetensors_header_offset_overflow"))?;
        if data_start > model_byte_len {
            return Err(integrity("safetensors_header_exceeds_file"));
        }
        let header_size =
            usize::try_from(header_len).map_err(|_| invalid("safetensors_header_size_overflow"))?;
        let mut header = vec![0u8; header_size];
        file.read_exact(&mut header)?;
        let value: Value = serde_json::from_slice(&header)?;
        let object = value
            .as_object()
            .ok_or_else(|| integrity("safetensors_header_not_object"))?;
        if object.is_empty() {
            return Err(integrity("safetensors_header_empty"));
        }

        let metadata = object
            .get("__metadata__")
            .map(|value| {
                let object = value
                    .as_object()
                    .ok_or_else(|| integrity("safetensors_metadata_not_object"))?;
                object
                    .iter()
                    .map(|(key, value)| {
                        let value = value
                            .as_str()
                            .ok_or_else(|| integrity("safetensors_metadata_value_not_string"))?;
                        Ok((key.clone(), value.to_string()))
                    })
                    .collect::<BrainResult<BTreeMap<_, _>>>()
            })
            .transpose()?;

        let mut tensors = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let tensor_id = TensorId::parse(name)?;
            let tensor = value
                .as_object()
                .ok_or_else(|| integrity("safetensors_tensor_info_not_object"))?;
            let expected_keys = BTreeSet::from(["dtype", "shape", "data_offsets"]);
            let observed_keys = tensor.keys().map(String::as_str).collect::<BTreeSet<_>>();
            if observed_keys != expected_keys {
                return Err(integrity(format!(
                    "safetensors_tensor_info_fields_invalid:{tensor_id}"
                )));
            }
            let dtype = tensor
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| integrity("safetensors_tensor_dtype_invalid"))?
                .to_string();
            let bits = dtype_bits(&dtype)
                .ok_or_else(|| integrity(format!("safetensors_dtype_unsupported:{dtype}")))?;
            let shape = parse_shape(
                tensor
                    .get("shape")
                    .ok_or_else(|| integrity("safetensors_tensor_shape_missing"))?,
            )?;
            let parameter_count = checked_parameter_count(&shape)?;
            let (start, end) = parse_offsets(
                tensor
                    .get("data_offsets")
                    .ok_or_else(|| integrity("safetensors_tensor_offsets_missing"))?,
            )?;
            let bit_count = parameter_count
                .checked_mul(bits)
                .ok_or_else(|| invalid("safetensors_tensor_byte_count_overflow"))?;
            if bit_count % 8 != 0 {
                return Err(integrity("safetensors_tensor_subbyte_misaligned"));
            }
            let data_byte_count = bit_count / 8;
            if end - start != data_byte_count {
                return Err(integrity(format!(
                    "safetensors_tensor_byte_count_mismatch:{tensor_id}"
                )));
            }
            tensors.push(ParsedTensor {
                spec: ModelTensorSpec {
                    tensor_id,
                    dtype,
                    shape,
                    parameter_count,
                    data_byte_count,
                },
                data_start: start,
                data_end: end,
            });
        }
        if tensors.is_empty() {
            return Err(integrity("safetensors_tensor_set_empty"));
        }
        tensors.sort_by_key(|tensor| (tensor.data_start, tensor.data_end));
        let mut expected_start = 0u64;
        let mut ids = BTreeSet::new();
        let mut total_parameter_count = 0u64;
        for tensor in &tensors {
            if tensor.data_start != expected_start || !ids.insert(tensor.spec.tensor_id.clone()) {
                return Err(integrity("safetensors_offsets_or_tensor_identity_invalid"));
            }
            expected_start = tensor.data_end;
            total_parameter_count = total_parameter_count
                .checked_add(tensor.spec.parameter_count)
                .ok_or_else(|| invalid("safetensors_total_parameter_count_overflow"))?;
        }
        if data_start
            .checked_add(expected_start)
            .ok_or_else(|| invalid("safetensors_total_size_overflow"))?
            != model_byte_len
        {
            return Err(integrity("safetensors_data_does_not_cover_file"));
        }
        let index = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.spec.tensor_id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        // Authenticate exactly the header we interpreted and the tensor bytes
        // in validated physical order. Hashing the file before parsing and
        // then seeking back would bind the inventory to a different read.
        let mut model_hasher = Sha256::new();
        model_hasher.update(header_len_raw);
        model_hasher.update(&header);
        let mut tensor_sha256 = BTreeMap::new();
        let mut buffer = vec![0u8; 1 << 20];
        for tensor in &tensors {
            let mut tensor_hasher = Sha256::new();
            let mut remaining = tensor.spec.data_byte_count;
            while remaining > 0 {
                let take = usize::try_from(remaining.min(buffer.len() as u64))
                    .map_err(|_| invalid("model_tensor_copy_size_overflow"))?;
                file.read_exact(&mut buffer[..take])?;
                model_hasher.update(&buffer[..take]);
                tensor_hasher.update(&buffer[..take]);
                remaining -= take as u64;
            }
            tensor_sha256.insert(
                tensor.spec.tensor_id.clone(),
                Sha256Digest::parse(format!("{:x}", tensor_hasher.finalize()))?,
            );
        }
        let mut trailing = [0u8; 1];
        if file.read(&mut trailing)? != 0 {
            return Err(integrity("safetensors_source_grew_during_authentication"));
        }
        let model_sha256 = Sha256Digest::parse(format!("{:x}", model_hasher.finalize()))?;
        let inventory = ModelParameterInventory {
            schema: MODEL_PARAMETER_INVENTORY_SCHEMA.into(),
            model_sha256,
            model_byte_len,
            tensor_count: tensors.len(),
            total_parameter_count,
            tensors: tensors.iter().map(|tensor| tensor.spec.clone()).collect(),
        };
        Ok(Self {
            file,
            data_start,
            metadata,
            tensors,
            index,
            inventory,
            tensor_sha256,
        })
    }

    fn tensor(&self, tensor_id: &TensorId) -> BrainResult<&ParsedTensor> {
        let index = self
            .index
            .get(tensor_id)
            .copied()
            .ok_or_else(|| invalid(format!("model_tensor_not_found:{tensor_id}")))?;
        self.tensors
            .get(index)
            .ok_or_else(|| integrity("model_tensor_index_invalid"))
    }

    fn verify_consumed_tensor(&self, tensor: &ParsedTensor, hasher: Sha256) -> BrainResult<()> {
        // Tests restore the source here, after consumption and before the
        // check. A rehash of the current file would incorrectly accept it.
        #[cfg(test)]
        tests::observe_read(tests::ReadPoint::TensorConsumed(
            tensor.spec.tensor_id.clone(),
        ));
        let expected = self
            .tensor_sha256
            .get(&tensor.spec.tensor_id)
            .ok_or_else(|| integrity("model_tensor_authenticated_digest_missing"))?;
        let consumed = Sha256Digest::parse(format!("{:x}", hasher.finalize()))?;
        if &consumed != expected {
            return Err(integrity(format!(
                "model_tensor_consumed_digest_mismatch:{}",
                tensor.spec.tensor_id
            )));
        }
        Ok(())
    }

    fn inspect_linear_readout(
        &mut self,
        tensor_id: &TensorId,
        positive_row: usize,
        negative_row: usize,
    ) -> BrainResult<LinearReadoutInspection> {
        let tensor = self.tensor(tensor_id)?.clone();
        if tensor.spec.shape.len() != 2
            || tensor.spec.shape[1] > 4_096
            || positive_row == negative_row
            || positive_row >= tensor.spec.shape[0]
            || negative_row >= tensor.spec.shape[0]
        {
            return Err(invalid("linear_readout_shape_or_rows_invalid"));
        }
        let input_dimension = tensor.spec.shape[1];
        let width = floating_input_width(&tensor.spec.dtype)?;
        let row_bytes = u64::try_from(input_dimension)
            .map_err(|_| invalid("linear_readout_dimension_overflow"))?
            .checked_mul(width as u64)
            .ok_or_else(|| invalid("linear_readout_row_size_overflow"))?;
        let positive_start = u64::try_from(positive_row)
            .map_err(|_| invalid("linear_readout_row_overflow"))?
            .checked_mul(row_bytes)
            .ok_or_else(|| invalid("linear_readout_row_offset_overflow"))?;
        let negative_start = u64::try_from(negative_row)
            .map_err(|_| invalid("linear_readout_row_overflow"))?
            .checked_mul(row_bytes)
            .ok_or_else(|| invalid("linear_readout_row_offset_overflow"))?;
        self.file.seek(SeekFrom::Start(
            self.data_start
                .checked_add(tensor.data_start)
                .ok_or_else(|| invalid("model_tensor_offset_overflow"))?,
        ))?;
        let mut positive_weights = Vec::with_capacity(input_dimension);
        let mut negative_weights = Vec::with_capacity(input_dimension);
        let mut buffer = vec![0u8; STREAM_ELEMENTS * width];
        let mut consumed = 0u64;
        let mut hasher = Sha256::new();
        while consumed < tensor.spec.data_byte_count {
            let take =
                usize::try_from((tensor.spec.data_byte_count - consumed).min(buffer.len() as u64))
                    .map_err(|_| invalid("linear_readout_chunk_size_overflow"))?;
            self.file.read_exact(&mut buffer[..take])?;
            hasher.update(&buffer[..take]);
            // Decode a bounded chunk, checking finiteness even in unselected
            // rows. No full vocabulary-sized readout is allocated.
            let values = decode_float_values(&tensor.spec.dtype, &buffer[..take], take / width)?;
            let end = consumed + take as u64;
            for (row_start, destination) in [
                (positive_start, &mut positive_weights),
                (negative_start, &mut negative_weights),
            ] {
                let overlap_start = consumed.max(row_start);
                let overlap_end = end.min(row_start + row_bytes);
                if overlap_start < overlap_end {
                    let first = usize::try_from((overlap_start - consumed) / width as u64)
                        .map_err(|_| invalid("linear_readout_chunk_offset_overflow"))?;
                    let last = usize::try_from((overlap_end - consumed) / width as u64)
                        .map_err(|_| invalid("linear_readout_chunk_offset_overflow"))?;
                    destination.extend(values[first..last].iter().map(|value| f64::from(*value)));
                }
            }
            consumed = end;
        }
        // All bytes of the tensor are authenticated, including rows not used
        // by the output. A final rehash of the path cannot replace this check.
        self.verify_consumed_tensor(&tensor, hasher)?;
        if positive_weights.len() != input_dimension || negative_weights.len() != input_dimension {
            return Err(integrity("linear_readout_row_length_mismatch"));
        }
        let difference_weights = positive_weights
            .iter()
            .zip(&negative_weights)
            .map(|(positive, negative)| positive - negative)
            .collect::<Vec<_>>();
        if difference_weights.iter().any(|value| !value.is_finite()) {
            return Err(integrity("linear_readout_difference_nonfinite"));
        }
        Ok(LinearReadoutInspection {
            schema: "cerebro.tidex.linear_readout_inspection/v1".into(),
            input_dimension,
            model_sha256: self.inventory.model_sha256.clone(),
            tensor_id: tensor_id.clone(),
            positive_row,
            negative_row,
            positive_weights,
            negative_weights,
            difference_weights,
        })
    }

    fn read_f32_tensor(&mut self, tensor_id: &TensorId) -> BrainResult<Vec<f32>> {
        let tensor = self.tensor(tensor_id)?.clone();
        let count = usize::try_from(tensor.spec.parameter_count)
            .map_err(|_| invalid("model_tensor_parameter_count_too_large"))?;
        self.file.seek(SeekFrom::Start(
            self.data_start
                .checked_add(tensor.data_start)
                .ok_or_else(|| invalid("model_tensor_offset_overflow"))?,
        ))?;
        let width = floating_input_width(&tensor.spec.dtype)?;
        let mut raw = vec![
            0u8;
            count
                .checked_mul(width)
                .ok_or_else(|| invalid("model_tensor_read_size_overflow"))?
        ];
        self.file.read_exact(&mut raw)?;
        let mut hasher = Sha256::new();
        hasher.update(&raw);
        self.verify_consumed_tensor(&tensor, hasher)?;
        decode_float_values(&tensor.spec.dtype, &raw, count)
    }
}

/// Inspect one exact SafeTensors checkpoint without loading its tensors into a
/// numerical framework. The inventory identity hashes the same header bytes
/// that are parsed and the same tensor bytes that establish per-tensor digests.
fn lora_target_from_factor(adapter_id: &TensorId, factor: char) -> BrainResult<TensorId> {
    let suffixes: &[&str] = match factor {
        'A' => &[".lora_A.weight", ".lora_A.default.weight"],
        'B' => &[".lora_B.weight", ".lora_B.default.weight"],
        _ => return Err(invalid("lora_adapter_factor_invalid")),
    };
    let text = adapter_id
        .as_str()
        .strip_prefix("base_model.model.")
        .ok_or_else(|| integrity(format!("lora_adapter_prefix_invalid:{adapter_id}")))?;
    let stem = suffixes
        .iter()
        .find_map(|suffix| text.strip_suffix(suffix))
        .ok_or_else(|| integrity(format!("lora_adapter_suffix_invalid:{adapter_id}")))?;
    TensorId::parse(format!("{stem}.weight"))
}

pub(crate) fn lora_target_family(target: &TensorId) -> BrainResult<String> {
    let family = target
        .as_str()
        .strip_suffix(".weight")
        .and_then(|stem| stem.rsplit('.').next())
        .ok_or_else(|| integrity(format!("lora_adapter_target_family_invalid:{target}")))?;
    const ALLOWED: &[&str] = &[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ];
    if !ALLOWED.contains(&family) {
        return Err(integrity(format!(
            "lora_adapter_target_family_unsupported:{target}"
        )));
    }
    Ok(family.to_string())
}

pub(crate) fn lora_target_layer(target: &TensorId) -> BrainResult<usize> {
    let parts = target.as_str().split('.').collect::<Vec<_>>();
    parts
        .windows(2)
        .find(|window| window[0] == "layers")
        .and_then(|window| window[1].parse::<usize>().ok())
        .ok_or_else(|| integrity(format!("lora_adapter_transformer_layer_missing:{target}")))
}

fn lora_adapter_pairs(
    specs: &[ModelTensorSpec],
) -> BrainResult<BTreeMap<TensorId, LoraAdapterPair>> {
    let mut pairs = BTreeMap::<TensorId, LoraAdapterPair>::new();
    for spec in specs {
        if spec.dtype != "F32" {
            return Err(integrity(format!(
                "lora_adapter_tensor_dtype_not_f32:{}:{}",
                spec.tensor_id, spec.dtype
            )));
        }
        let text = spec.tensor_id.as_str();
        let (factor, target) =
            if text.ends_with(".lora_A.weight") || text.ends_with(".lora_A.default.weight") {
                ('A', lora_target_from_factor(&spec.tensor_id, 'A')?)
            } else if text.ends_with(".lora_B.weight") || text.ends_with(".lora_B.default.weight") {
                ('B', lora_target_from_factor(&spec.tensor_id, 'B')?)
            } else {
                return Err(integrity(format!(
                    "lora_adapter_contains_non_lora_tensor:{}",
                    spec.tensor_id
                )));
            };
        let pair = pairs
            .entry(target)
            .or_insert(LoraAdapterPair { a: None, b: None });
        let slot = if factor == 'A' {
            &mut pair.a
        } else {
            &mut pair.b
        };
        if slot.replace(spec.tensor_id.clone()).is_some() {
            return Err(integrity("lora_adapter_duplicate_factor"));
        }
    }
    if pairs.is_empty()
        || pairs
            .values()
            .any(|pair| pair.a.is_none() || pair.b.is_none())
    {
        return Err(integrity("lora_adapter_factor_pair_incomplete"));
    }
    Ok(pairs)
}

fn prepare_lora_patches(
    base_specs: &[ModelTensorSpec],
    adapter_specs: &[ModelTensorSpec],
    adapter_path: &Path,
    pairs: &BTreeMap<TensorId, LoraAdapterPair>,
    rank: usize,
) -> BrainResult<(Vec<TensorId>, Vec<LoraPatchFactors>)> {
    if rank == 0 {
        return Err(invalid("lora_adapter_rank_zero"));
    }
    let base_by_id = base_specs
        .iter()
        .map(|spec| (spec.tensor_id.clone(), spec))
        .collect::<BTreeMap<_, _>>();
    let adapter_by_id = adapter_specs
        .iter()
        .map(|spec| (spec.tensor_id.clone(), spec))
        .collect::<BTreeMap<_, _>>();
    let pair_ids = pairs.keys().cloned().collect::<BTreeSet<_>>();
    let targets = base_specs
        .iter()
        .filter(|spec| pair_ids.contains(&spec.tensor_id))
        .map(|spec| spec.tensor_id.clone())
        .collect::<Vec<_>>();
    if targets.len() != pairs.len() {
        return Err(integrity("lora_adapter_targets_missing_from_base_model"));
    }
    let requested_ids = targets
        .iter()
        .flat_map(|target| {
            let pair = &pairs[target];
            [pair.a.clone().unwrap(), pair.b.clone().unwrap()]
        })
        .collect::<Vec<_>>();
    let tensors = read_model_tensors_f32(adapter_path, &requested_ids)?;
    let mut patches = Vec::with_capacity(targets.len());
    for target in &targets {
        lora_target_family(target)?;
        lora_target_layer(target)?;
        let target_spec = base_by_id
            .get(target)
            .ok_or_else(|| integrity("lora_adapter_base_target_spec_missing"))?;
        if target_spec.shape.len() != 2 {
            return Err(integrity(format!(
                "lora_adapter_base_target_not_matrix:{target}"
            )));
        }
        if !receiver_delta_dtype_supported(&target_spec.dtype) {
            return Err(integrity(format!(
                "lora_adapter_base_target_dtype_unsupported:{target}:{}",
                target_spec.dtype
            )));
        }
        let output_dim = target_spec.shape[0];
        let input_dim = target_spec.shape[1];
        let pair = &pairs[target];
        let a_id = pair.a.as_ref().unwrap();
        let b_id = pair.b.as_ref().unwrap();
        let a_spec = adapter_by_id
            .get(a_id)
            .ok_or_else(|| integrity("lora_adapter_a_spec_missing"))?;
        let b_spec = adapter_by_id
            .get(b_id)
            .ok_or_else(|| integrity("lora_adapter_b_spec_missing"))?;
        if a_spec.shape != vec![rank, input_dim] || b_spec.shape != vec![output_dim, rank] {
            return Err(integrity(format!(
                "lora_adapter_factor_shape_mismatch:{target}"
            )));
        }
        patches.push(LoraPatchFactors {
            input_dim,
            output_dim,
            rank,
            a: tensors
                .get(a_id)
                .cloned()
                .ok_or_else(|| integrity("lora_adapter_a_values_missing"))?,
            b: tensors
                .get(b_id)
                .cloned()
                .ok_or_else(|| integrity("lora_adapter_b_values_missing"))?,
        });
    }
    Ok((targets, patches))
}

/// Convert a PEFT LoRA into an immutable dense receiver axis. This is the
/// missing data-plane bridge between learned calibration adapters and the V69
/// receiver compiler. The target capability must never be used to create these
/// adapters; that scientific lineage is validated by receiver_weight_binding.
pub fn import_peft_lora_as_dense_axis(
    root: &Path,
    input: &LoraAdapterAxisInput,
) -> BrainResult<LoraAdapterAxisReceipt> {
    if input.schema != LORA_ADAPTER_AXIS_INPUT_SCHEMA {
        return Err(invalid("lora_adapter_axis_input_schema"));
    }
    let config_path = canonical_existing_file(&input.adapter_config_path)?;
    let config_metadata = fs::metadata(&config_path)?;
    if config_metadata.len() == 0 || config_metadata.len() > 1_048_576 {
        return Err(invalid("lora_adapter_config_size_invalid"));
    }
    let config_bytes = fs::read(&config_path)?;
    let config: PeftLoraConfig = serde_json::from_slice(&config_bytes)?;
    if config.r == 0
        || !config.lora_alpha.is_finite()
        || config.lora_alpha <= 0.0
        || config.target_modules.is_empty()
        || config.bias != "none"
        || config.use_dora
        || config.fan_in_fan_out
        || config
            .modules_to_save
            .as_ref()
            .is_some_and(|value| !value.is_empty())
        || !config.rank_pattern.is_empty()
        || !config.alpha_pattern.is_empty()
    {
        return Err(invalid("lora_adapter_config_unsupported"));
    }
    let base = inspect_model_safetensors(&input.base_model_path)?;
    let adapter = inspect_model_safetensors(&input.adapter_model_path)?;
    let pairs = lora_adapter_pairs(&adapter.tensors)?;
    let (targets, patches) = prepare_lora_patches(
        &base.tensors,
        &adapter.tensors,
        &input.adapter_model_path,
        &pairs,
        config.r,
    )?;
    if sha256_file(&input.adapter_model_path)? != adapter.model_sha256 {
        return Err(integrity("lora_adapter_changed_during_import"));
    }
    let target_families = targets
        .iter()
        .map(lora_target_family)
        .collect::<BrainResult<BTreeSet<_>>>()?;
    if target_families != config.target_modules {
        return Err(integrity("lora_adapter_target_modules_mismatch"));
    }
    let covered_transformer_layers = targets
        .iter()
        .map(lora_target_layer)
        .collect::<BrainResult<BTreeSet<_>>>()?;
    let layout = parameter_layout_for_tensors(&base, &targets)?;
    let lora_scale = if config.use_rslora {
        config.lora_alpha / (config.r as f64).sqrt()
    } else {
        config.lora_alpha / config.r as f64
    };
    if !lora_scale.is_finite()
        || lora_scale <= 0.0
        || !(lora_scale as f32).is_finite()
        || lora_scale as f32 == 0.0
    {
        return Err(invalid("lora_adapter_scale_invalid"));
    }
    let dense_delta = ArtifactWriteAuthority::for_internal_root(root)?
        .create_content_addressed_dvec_iter(
            layout.total_parameter_count,
            LoraDenseDeltaIter::new(&patches, lora_scale as f32),
        )?;
    if sha256_file(&input.base_model_path)? != base.model_sha256
        || sha256_file(&input.adapter_model_path)? != adapter.model_sha256
        || Sha256Digest::digest_bytes(&fs::read(&input.adapter_config_path)?)
            != Sha256Digest::digest_bytes(&config_bytes)
    {
        return Err(integrity("lora_adapter_inputs_changed_during_import"));
    }
    Ok(LoraAdapterAxisReceipt {
        schema: LORA_ADAPTER_AXIS_RECEIPT_SCHEMA.into(),
        base_model_path: input.base_model_path.clone(),
        adapter_model_path: input.adapter_model_path.clone(),
        adapter_config_path: input.adapter_config_path.clone(),
        base_model_sha256: base.model_sha256,
        adapter_model_sha256: adapter.model_sha256,
        adapter_config_sha256: Sha256Digest::digest_bytes(&config_bytes),
        lora_rank: config.r,
        lora_alpha: config.lora_alpha,
        lora_scale,
        use_rslora: config.use_rslora,
        learned_parameter_count: adapter.total_parameter_count,
        target_tensors: targets,
        target_families,
        covered_transformer_layers,
        parameter_layout: layout,
        dense_delta,
        source_training_semantics_attested: false,
        authorizes_target_update_free_claim: false,
        authorizes_promotion: false,
    })
}

/// Authenticate an immutable LoRA import receipt and replay its exact PEFT
/// conversion while the original inputs are still available. Adapter-bank
/// admission uses this once; durable operation then depends only on the
/// normalized manifest, layout and content-addressed dense delta.
pub fn authenticate_lora_adapter_axis_receipt(
    root: &Path,
    reference: &PrivateFileReference,
) -> BrainResult<LoraAdapterAxisReceipt> {
    let root = verify_internal_private_root(root)?;
    let bytes = reference.read_verified_bounded(&root, 8 * 1024 * 1024)?;
    let receipt: LoraAdapterAxisReceipt = serde_json::from_slice(&bytes)?;
    if receipt.schema != LORA_ADAPTER_AXIS_RECEIPT_SCHEMA
        || receipt.learned_parameter_count == 0
        || receipt.target_tensors.is_empty()
        || receipt.source_training_semantics_attested
        || receipt.authorizes_target_update_free_claim
        || receipt.authorizes_promotion
        || receipt.dense_delta.parameter_count != receipt.parameter_layout.total_parameter_count
    {
        return Err(invalid("lora_adapter_axis_receipt_invalid"));
    }
    receipt.parameter_layout.validate()?;
    verify_dvec_reference_under_root(&root, &receipt.dense_delta)?;
    let replay = import_peft_lora_as_dense_axis(
        &root,
        &LoraAdapterAxisInput {
            schema: LORA_ADAPTER_AXIS_INPUT_SCHEMA.to_string(),
            base_model_path: receipt.base_model_path.clone(),
            adapter_model_path: receipt.adapter_model_path.clone(),
            adapter_config_path: receipt.adapter_config_path.clone(),
        },
    )?;
    if replay != receipt {
        return Err(integrity("lora_adapter_axis_receipt_replay_mismatch"));
    }
    Ok(receipt)
}

pub fn inspect_model_safetensors(path: &Path) -> BrainResult<ModelParameterInventory> {
    Ok(SafeTensorArchive::open(path)?.inventory)
}

/// Extract two rows without loading a whole language-model readout into RAM.
/// The complete checkpoint establishes the model identity; a second streaming
/// pass authenticates every consumed readout byte against that same identity.
pub fn inspect_linear_readout(
    path: &Path,
    tensor_id: &TensorId,
    positive_row: usize,
    negative_row: usize,
) -> BrainResult<LinearReadoutInspection> {
    let mut archive = SafeTensorArchive::open(path)?;
    #[cfg(test)]
    tests::observe_read(tests::ReadPoint::InputsAuthenticated);
    archive.inspect_linear_readout(tensor_id, positive_row, negative_row)
}

/// Reuse the canonical block-layout authority for an explicit mutable tensor
/// subset. Tensor order here is the flat delta order and is therefore part of
/// the semantic layout digest.
pub fn parameter_layout_for_tensors(
    inventory: &ModelParameterInventory,
    tensor_ids: &[TensorId],
) -> BrainResult<ParameterBlockLayout> {
    if inventory.schema != MODEL_PARAMETER_INVENTORY_SCHEMA || tensor_ids.is_empty() {
        return Err(invalid("model_parameter_layout_request_invalid"));
    }
    let by_id = inventory
        .tensors
        .iter()
        .map(|tensor| (tensor.tensor_id.clone(), tensor))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut shapes = Vec::with_capacity(tensor_ids.len());
    for tensor_id in tensor_ids {
        if !seen.insert(tensor_id.clone()) {
            return Err(invalid("model_parameter_layout_duplicate_tensor"));
        }
        let tensor = by_id
            .get(tensor_id)
            .ok_or_else(|| invalid(format!("model_parameter_layout_tensor_missing:{tensor_id}")))?;
        let count = usize::try_from(tensor.parameter_count)
            .map_err(|_| invalid("model_parameter_layout_count_overflow"))?;
        shapes.push(BlockShapeSpec {
            name: tensor_id.as_str().to_string(),
            shape: tensor.shape.clone(),
            count,
        });
    }
    ParameterBlockLayout::from_shapes(&shapes)
}

pub(crate) fn receiver_delta_dtype_supported(dtype: &str) -> bool {
    matches!(dtype, "BF16" | "F16" | "F32")
}

fn floating_input_width(dtype: &str) -> BrainResult<usize> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" => Ok(4),
        _ => Err(invalid(format!(
            "model_weight_delta_requires_float_tensor:{dtype}"
        ))),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;
    let f32_bits = if exponent == 0 {
        if fraction == 0 {
            sign << 31
        } else {
            let mut mantissa = fraction;
            let mut shift = 0i32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x03ff;
            let exp = (127 - 14 - shift) as u32;
            (sign << 31) | (exp << 23) | (mantissa << 13)
        }
    } else if exponent == 0x1f {
        (sign << 31) | (0xff << 23) | (fraction << 13)
    } else {
        let exp = (exponent + (127 - 15)) as u32;
        (sign << 31) | (exp << 23) | (fraction << 13)
    };
    f32::from_bits(f32_bits)
}

fn decode_float_values(dtype: &str, bytes: &[u8], count: usize) -> BrainResult<Vec<f32>> {
    let width = floating_input_width(dtype)?;
    if bytes.len() != count.saturating_mul(width) {
        return Err(integrity("model_tensor_decode_size_mismatch"));
    }
    let mut values = Vec::with_capacity(count);
    match dtype {
        "BF16" => {
            for pair in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                values.push(f32::from_bits((bits as u32) << 16));
            }
        }
        "F16" => {
            for pair in bytes.chunks_exact(2) {
                values.push(f16_to_f32(u16::from_le_bytes([pair[0], pair[1]])));
            }
        }
        "F32" => {
            for chunk in bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        _ => unreachable!("floating_input_width already rejected this dtype"),
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(integrity("model_tensor_nonfinite_value"));
    }
    Ok(values)
}

fn output_plans(
    archive: &SafeTensorArchive,
    layout: &ParameterBlockLayout,
) -> BrainResult<Vec<OutputTensorPlan>> {
    layout.validate()?;
    let mut modified = BTreeSet::new();
    let mut plans: Vec<OutputTensorPlan> = Vec::with_capacity(archive.tensors.len());
    let mut output_offset = 0u64;
    for block in &layout.blocks {
        let tensor_id = TensorId::parse(&block.name)?;
        if !modified.insert(tensor_id.clone()) {
            return Err(invalid("model_weight_layout_duplicate_tensor"));
        }
        let source_index =
            archive.index.get(&tensor_id).copied().ok_or_else(|| {
                invalid(format!("model_weight_layout_tensor_missing:{tensor_id}"))
            })?;
        let tensor = archive
            .tensors
            .get(source_index)
            .ok_or_else(|| integrity("model_weight_source_index_invalid"))?;
        if tensor.spec.shape != block.shape
            || tensor.spec.parameter_count != block.count as u64
            || block.offset
                != plans
                    .iter()
                    .filter(|plan| plan.modified)
                    .map(|plan| archive.tensors[plan.source_index].spec.parameter_count)
                    .sum::<u64>()
        {
            return Err(integrity(format!(
                "model_weight_layout_tensor_contract_mismatch:{tensor_id}"
            )));
        }
        floating_input_width(&tensor.spec.dtype)?;
        let bytes = tensor
            .spec
            .parameter_count
            .checked_mul(4)
            .ok_or_else(|| invalid("model_weight_output_size_overflow"))?;
        let data_end = output_offset
            .checked_add(bytes)
            .ok_or_else(|| invalid("model_weight_output_size_overflow"))?;
        plans.push(OutputTensorPlan {
            tensor_id,
            dtype: "F32".into(),
            shape: tensor.spec.shape.clone(),
            source_index,
            modified: true,
            data_start: output_offset,
            data_end,
        });
        output_offset = data_end;
    }
    for (source_index, tensor) in archive.tensors.iter().enumerate() {
        if modified.contains(&tensor.spec.tensor_id) {
            continue;
        }
        let data_end = output_offset
            .checked_add(tensor.spec.data_byte_count)
            .ok_or_else(|| invalid("model_weight_output_size_overflow"))?;
        plans.push(OutputTensorPlan {
            tensor_id: tensor.spec.tensor_id.clone(),
            dtype: tensor.spec.dtype.clone(),
            shape: tensor.spec.shape.clone(),
            source_index,
            modified: false,
            data_start: output_offset,
            data_end,
        });
        output_offset = data_end;
    }
    if plans.len() != archive.tensors.len() {
        return Err(integrity("model_weight_output_plan_tensor_count_mismatch"));
    }
    Ok(plans)
}

fn encoded_header(
    metadata: &Option<BTreeMap<String, String>>,
    plans: &[OutputTensorPlan],
) -> BrainResult<Vec<u8>> {
    let mut map = Map::new();
    if let Some(metadata) = metadata {
        let mut values = Map::new();
        for (key, value) in metadata {
            values.insert(key.clone(), Value::String(value.clone()));
        }
        map.insert("__metadata__".into(), Value::Object(values));
    }
    for plan in plans {
        map.insert(
            plan.tensor_id.as_str().to_string(),
            json!({
                "dtype": plan.dtype,
                "shape": plan.shape,
                "data_offsets": [plan.data_start, plan.data_end]
            }),
        );
    }
    let mut bytes = serde_json::to_vec(&Value::Object(map))?;
    let padding = (8 - (bytes.len() % 8)) % 8;
    bytes.extend(std::iter::repeat_n(b' ', padding));
    if bytes.is_empty() || bytes.len() as u64 > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(invalid("model_weight_output_header_size_invalid"));
    }
    Ok(bytes)
}

fn copy_tensor_bytes(
    archive: &mut SafeTensorArchive,
    tensor: &ParsedTensor,
    writer: &mut BufWriter<&mut File>,
) -> BrainResult<()> {
    archive.file.seek(SeekFrom::Start(
        archive
            .data_start
            .checked_add(tensor.data_start)
            .ok_or_else(|| invalid("model_tensor_offset_overflow"))?,
    ))?;
    let mut remaining = tensor.spec.data_byte_count;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    while remaining > 0 {
        let take = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| invalid("model_tensor_copy_size_overflow"))?;
        archive.file.read_exact(&mut buffer[..take])?;
        hasher.update(&buffer[..take]);
        writer.write_all(&buffer[..take])?;
        remaining -= take as u64;
    }
    archive.verify_consumed_tensor(tensor, hasher)
}

fn write_modified_tensor(
    archive: &mut SafeTensorArchive,
    tensor: &ParsedTensor,
    delta: &mut VerifiedDvecReader,
    writer: &mut BufWriter<&mut File>,
) -> BrainResult<()> {
    archive.file.seek(SeekFrom::Start(
        archive
            .data_start
            .checked_add(tensor.data_start)
            .ok_or_else(|| invalid("model_tensor_offset_overflow"))?,
    ))?;
    let width = floating_input_width(&tensor.spec.dtype)?;
    let total = usize::try_from(tensor.spec.parameter_count)
        .map_err(|_| invalid("model_tensor_parameter_count_too_large"))?;
    let mut completed = 0usize;
    let mut hasher = Sha256::new();
    while completed < total {
        let count = (total - completed).min(STREAM_ELEMENTS);
        let mut base_bytes = vec![
            0u8;
            count
                .checked_mul(width)
                .ok_or_else(|| invalid("model_tensor_chunk_size_overflow"))?
        ];
        archive.file.read_exact(&mut base_bytes)?;
        hasher.update(&base_bytes);
        let base = decode_float_values(&tensor.spec.dtype, &base_bytes, count)?;
        let update = delta.read_f32(count)?;
        let mut output = Vec::with_capacity(count * 4);
        for (base, update) in base.into_iter().zip(update) {
            let value = base + update;
            if !value.is_finite() {
                return Err(BrainError::Numerical(
                    "model_weight_materialization_nonfinite".into(),
                ));
            }
            output.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&output)?;
        completed += count;
    }
    archive.verify_consumed_tensor(tensor, hasher)
}

fn create_output_staging(output: &Path) -> BrainResult<(PathBuf, File)> {
    let parent = output
        .parent()
        .ok_or_else(|| invalid("model_weight_output_parent_missing"))?;
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("model_weight_output_name_invalid"))?;
    loop {
        let sequence = NEXT_OUTPUT_TEMP.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.{}.{}.tidex.tmp",
            std::process::id(),
            sequence
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true).mode(0o600);
        match options.open(&temporary) {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

fn install_output_noreplace(temporary: &Path, output: &Path) -> BrainResult<()> {
    match fs::hard_link(temporary, output) {
        Ok(()) => {
            fs::remove_file(temporary)?;
            let mut permissions = fs::metadata(output)?.permissions();
            permissions.set_mode(0o600);
            fs::set_permissions(output, permissions)?;
            if let Some(parent) = output.parent() {
                File::open(parent)?.sync_all()?;
            }
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(temporary);
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                Err(integrity("model_weight_output_raced_existing_file"))
            } else {
                Err(error.into())
            }
        }
    }
}

fn verify_output_semantics(
    output: &Path,
    source: &ModelParameterInventory,
    modified: &BTreeSet<TensorId>,
) -> BrainResult<ModelParameterInventory> {
    let observed = inspect_model_safetensors(output)?;
    if observed.tensor_count != source.tensor_count
        || observed.total_parameter_count != source.total_parameter_count
    {
        return Err(integrity("model_weight_output_inventory_count_mismatch"));
    }
    let source_by_id = source
        .tensors
        .iter()
        .map(|tensor| (tensor.tensor_id.clone(), tensor))
        .collect::<BTreeMap<_, _>>();
    let output_by_id = observed
        .tensors
        .iter()
        .map(|tensor| (tensor.tensor_id.clone(), tensor))
        .collect::<BTreeMap<_, _>>();
    if source_by_id.len() != output_by_id.len() || source_by_id.keys().ne(output_by_id.keys()) {
        return Err(integrity("model_weight_output_tensor_identity_mismatch"));
    }
    for (tensor_id, before) in &source_by_id {
        let after = output_by_id
            .get(tensor_id)
            .ok_or_else(|| integrity("model_weight_output_tensor_missing"))?;
        if before.shape != after.shape || before.parameter_count != after.parameter_count {
            return Err(integrity(format!(
                "model_weight_output_tensor_shape_mismatch:{tensor_id}"
            )));
        }
        if modified.contains(tensor_id) {
            if after.dtype != "F32" {
                return Err(integrity(format!(
                    "model_weight_output_modified_dtype_mismatch:{tensor_id}"
                )));
            }
        } else if before.dtype != after.dtype || before.data_byte_count != after.data_byte_count {
            return Err(integrity(format!(
                "model_weight_output_untouched_contract_mismatch:{tensor_id}"
            )));
        }
    }
    Ok(observed)
}

/// Apply a verified flat f32 delta directly to selected tensors of a
/// SafeTensors checkpoint. The base checkpoint is never modified. The output
/// is a normal standalone checkpoint file: no adapter is required at runtime.
///
/// This function is intentionally candidate-only. Behavioral evaluation and a
/// later residency/promotion authority must decide whether the resulting model
/// is acceptable.
pub fn materialize_dense_delta_checkpoint(
    private_root: &Path,
    base_model_path: &Path,
    expected_base_sha256: &Sha256Digest,
    layout: &ParameterBlockLayout,
    delta_reference: &DeltaArtifactRef,
    output_path: &Path,
) -> BrainResult<WeightMaterializationReceipt> {
    layout.validate()?;
    if layout.total_parameter_count != delta_reference.parameter_count {
        return Err(invalid("model_weight_delta_layout_count_mismatch"));
    }
    let output = canonical_output_path(output_path)?;
    let mut archive = SafeTensorArchive::open(base_model_path)?;
    if &archive.inventory.model_sha256 != expected_base_sha256 {
        return Err(integrity("model_weight_base_digest_mismatch"));
    }
    let plans = output_plans(&archive, layout)?;
    let modified = plans
        .iter()
        .filter(|plan| plan.modified)
        .map(|plan| plan.tensor_id.clone())
        .collect::<BTreeSet<_>>();
    if modified.len() != layout.blocks.len() {
        return Err(integrity("model_weight_modified_tensor_count_mismatch"));
    }
    let parameter_layout_sha256 = parameter_layout_digest(layout)?;
    let mut delta = VerifiedDvecReader::open(private_root, delta_reference)?;
    if delta.parameter_count() != layout.total_parameter_count {
        return Err(integrity("model_weight_verified_delta_count_mismatch"));
    }
    let header = encoded_header(&archive.metadata, &plans)?;
    #[cfg(test)]
    tests::observe_read(tests::ReadPoint::InputsAuthenticated);
    let (temporary, mut file) = create_output_staging(&output)?;
    let write_result = (|| -> BrainResult<()> {
        {
            let mut writer = BufWriter::with_capacity(1 << 20, &mut file);
            writer.write_all(&(header.len() as u64).to_le_bytes())?;
            writer.write_all(&header)?;
            for plan in &plans {
                let tensor = archive
                    .tensors
                    .get(plan.source_index)
                    .cloned()
                    .ok_or_else(|| integrity("model_weight_output_source_index_invalid"))?;
                if plan.modified {
                    write_modified_tensor(&mut archive, &tensor, &mut delta, &mut writer)?;
                } else {
                    copy_tensor_bytes(&mut archive, &tensor, &mut writer)?;
                }
            }
            writer.flush()?;
        }
        delta.finish()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(error) = write_result {
        drop(file);
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    let output_model_sha256 = sha256_open_file(&mut file)?;
    let output_model_byte_len = file.metadata()?.len();
    drop(file);

    let staged_inventory = verify_output_semantics(&temporary, &archive.inventory, &modified)?;
    if staged_inventory.model_sha256 != output_model_sha256
        || staged_inventory.model_byte_len != output_model_byte_len
    {
        let _ = fs::remove_file(&temporary);
        return Err(integrity(
            "model_weight_output_digest_reverification_mismatch",
        ));
    }
    install_output_noreplace(&temporary, &output)?;
    let installed_inventory = verify_output_semantics(&output, &archive.inventory, &modified)?;
    if installed_inventory.model_sha256 != output_model_sha256
        || installed_inventory.model_byte_len != output_model_byte_len
    {
        return Err(integrity("model_weight_installed_output_changed"));
    }

    Ok(WeightMaterializationReceipt {
        schema: WEIGHT_MATERIALIZATION_RECEIPT_SCHEMA.into(),
        lifecycle: WeightMaterializationLifecycle::CandidateOnlyNotPromoted,
        base_model_sha256: archive.inventory.model_sha256,
        base_model_byte_len: archive.inventory.model_byte_len,
        source_tensor_count: archive.inventory.tensor_count,
        source_parameter_count: archive.inventory.total_parameter_count,
        parameter_layout_sha256,
        delta_sha256: delta_reference.sha256.clone(),
        delta_parameter_count: delta_reference.parameter_count,
        modified_tensor_count: modified.len(),
        modified_parameter_count: layout.total_parameter_count,
        output_model_sha256,
        output_model_byte_len,
        dtype_policy: "modified_tensors_f32_untouched_source_bytes/v1".into(),
        trains_target_capability: false,
        requires_adapter_at_runtime: false,
        authorizes_promotion: false,
    })
}

/// Re-authenticate a candidate-only materialization receipt against the exact
/// base checkpoint, layout, dense delta and installed output it claims. This is
/// intentionally a witness verifier, never a promotion authority.
pub fn authenticate_weight_materialization_receipt(
    private_root: &Path,
    base_model_path: &Path,
    layout: &ParameterBlockLayout,
    delta_reference: &DeltaArtifactRef,
    output_path: &Path,
    receipt: &WeightMaterializationReceipt,
) -> BrainResult<()> {
    layout.validate()?;
    if layout.total_parameter_count != delta_reference.parameter_count {
        return Err(invalid("model_weight_receipt_delta_layout_count_mismatch"));
    }

    // Hash the exact bytes delivered by the retained descriptor, including the
    // header, so a same-inode mutation cannot be hidden between a pre-hash and
    // this authentication boundary.
    let mut delta = VerifiedDvecReader::open(private_root, delta_reference)?;
    while delta.next_parameter() < delta.parameter_count() {
        let remaining = delta.parameter_count() - delta.next_parameter();
        let len = usize::try_from(remaining.min(STREAM_ELEMENTS as u64))
            .map_err(|_| invalid("model_weight_receipt_delta_chunk_overflow"))?;
        delta.read_f32(len)?;
    }
    delta.finish()?;

    let base = SafeTensorArchive::open(base_model_path)?;
    let plans = output_plans(&base, layout)?;
    let modified = plans
        .iter()
        .filter(|plan| plan.modified)
        .map(|plan| plan.tensor_id.clone())
        .collect::<BTreeSet<_>>();
    if modified.len() != layout.blocks.len() {
        return Err(integrity(
            "model_weight_receipt_modified_tensor_count_mismatch",
        ));
    }
    let output = verify_output_semantics(output_path, &base.inventory, &modified)?;
    let expected = WeightMaterializationReceipt {
        schema: WEIGHT_MATERIALIZATION_RECEIPT_SCHEMA.to_string(),
        lifecycle: WeightMaterializationLifecycle::CandidateOnlyNotPromoted,
        base_model_sha256: base.inventory.model_sha256.clone(),
        base_model_byte_len: base.inventory.model_byte_len,
        source_tensor_count: base.inventory.tensor_count,
        source_parameter_count: base.inventory.total_parameter_count,
        parameter_layout_sha256: parameter_layout_digest(layout)?,
        delta_sha256: delta_reference.sha256.clone(),
        delta_parameter_count: delta_reference.parameter_count,
        modified_tensor_count: modified.len(),
        modified_parameter_count: layout.total_parameter_count,
        output_model_sha256: output.model_sha256,
        output_model_byte_len: output.model_byte_len,
        dtype_policy: "modified_tensors_f32_untouched_source_bytes/v1".to_string(),
        trains_target_capability: false,
        requires_adapter_at_runtime: false,
        authorizes_promotion: false,
    };
    if receipt != &expected {
        return Err(integrity("model_weight_materialization_receipt_mismatch"));
    }
    Ok(())
}

/// Read one named floating tensor from an authenticated SafeTensors file. This
/// bounded helper is primarily useful for calibration/import tooling; the
/// direct actuator itself streams base weights and never loads the full model.
pub fn read_model_tensor_f32(path: &Path, tensor_id: &TensorId) -> BrainResult<Vec<f32>> {
    SafeTensorArchive::open(path)?.read_f32_tensor(tensor_id)
}

/// Read several named floating tensors after authenticating and parsing the
/// archive once. Requested identities must be unique and all must exist.
pub fn read_model_tensors_f32(
    path: &Path,
    tensor_ids: &[TensorId],
) -> BrainResult<BTreeMap<TensorId, Vec<f32>>> {
    if tensor_ids.is_empty() {
        return Err(invalid("model_tensor_batch_request_empty"));
    }
    let mut archive = SafeTensorArchive::open(path)?;
    let mut result = BTreeMap::new();
    for tensor_id in tensor_ids {
        if result.contains_key(tensor_id) {
            return Err(invalid("model_tensor_batch_request_duplicate"));
        }
        result.insert(tensor_id.clone(), archive.read_f32_tensor(tensor_id)?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{create_content_addressed_dvec, read_dvec_f32};
    use crate::security::secure_dir;
    use std::time::{SystemTime, UNIX_EPOCH};

    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    pub(super) enum ReadPoint {
        InputsAuthenticated,
        TensorConsumed(TensorId),
    }

    // Per-thread hooks exist only in tests and coordinate real same-inode file
    // writes. No production input can substitute the authenticated byte reader.
    type ReadHook = Box<dyn FnMut(ReadPoint)>;
    std::thread_local! {
        static READ_HOOK: RefCell<Option<ReadHook>> = const { RefCell::new(None) };
    }

    pub(super) fn observe_read(point: ReadPoint) {
        READ_HOOK.with(|slot| {
            if let Some(hook) = slot.borrow_mut().as_mut() {
                hook(point);
            }
        });
    }

    struct ReadHookGuard;

    impl ReadHookGuard {
        fn install(hook: impl FnMut(ReadPoint) + 'static) -> Self {
            READ_HOOK.with(|slot| {
                let mut slot = slot.borrow_mut();
                assert!(slot.is_none());
                *slot = Some(Box::new(hook));
            });
            Self
        }
    }

    impl Drop for ReadHookGuard {
        fn drop(&mut self) {
            READ_HOOK.with(|slot| {
                *slot.borrow_mut() = None;
            });
        }
    }

    fn root(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tidex-weight-actuator-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        secure_dir(&root).unwrap();
        root
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn write_fixture(path: &Path) {
        let q = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let v = bf16_bytes(&[-1.0, -2.0, -3.0, -4.0]);
        let mut header = serde_json::to_vec(&json!({
            "__metadata__": {"format":"pt"},
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype":"BF16", "shape":[2,2], "data_offsets":[0, q.len()]
            },
            "model.layers.0.self_attn.v_proj.weight": {
                "dtype":"BF16", "shape":[2,2], "data_offsets":[q.len(), q.len()+v.len()]
            }
        }))
        .unwrap();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&q).unwrap();
        file.write_all(&v).unwrap();
        file.sync_all().unwrap();
    }

    fn write_lora_fixture(path: &Path) {
        let tensors = [
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                vec![1.0_f32, 2.0],
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                vec![3.0_f32, 4.0],
                vec![2, 1],
            ),
            (
                "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
                vec![1.0_f32, -1.0],
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
                vec![2.0_f32, 3.0],
                vec![2, 1],
            ),
        ];
        let mut data = Vec::new();
        let mut entries = Map::new();
        entries.insert("__metadata__".into(), json!({"format":"pt"}));
        for (name, values, shape) in tensors {
            let start = data.len();
            for value in values {
                data.extend(value.to_le_bytes());
            }
            entries.insert(
                name.into(),
                json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&Value::Object(entries)).unwrap();
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
    fn peft_lora_import_reconstructs_multitensor_dense_axis() {
        let root = root("lora-axis");
        let base = root.join("base.safetensors");
        let adapter = root.join("adapter_model.safetensors");
        let config = root.join("adapter_config.json");
        write_fixture(&base);
        write_lora_fixture(&adapter);
        fs::write(
            &config,
            serde_json::to_vec(&json!({
                "r": 1,
                "lora_alpha": 2.0,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "use_rslora": false,
                "use_dora": false
            }))
            .unwrap(),
        )
        .unwrap();
        let receipt = import_peft_lora_as_dense_axis(
            &root,
            &LoraAdapterAxisInput {
                schema: LORA_ADAPTER_AXIS_INPUT_SCHEMA.into(),
                base_model_path: base,
                adapter_model_path: adapter,
                adapter_config_path: config,
            },
        )
        .unwrap();
        assert_eq!(receipt.lora_scale, 2.0);
        assert_eq!(receipt.learned_parameter_count, 8);
        assert_eq!(receipt.parameter_layout.total_parameter_count, 8);
        assert_eq!(receipt.target_tensors.len(), 2);
        assert_eq!(
            receipt.target_families,
            BTreeSet::from(["q_proj".into(), "v_proj".into()])
        );
        assert_eq!(receipt.covered_transformer_layers, BTreeSet::from([0]));
        assert_eq!(
            read_dvec_f32(&root, &receipt.dense_delta).unwrap(),
            vec![6.0, 12.0, 8.0, 16.0, 4.0, -4.0, 6.0, -6.0]
        );
        assert!(!receipt.source_training_semantics_attested);
        assert!(!receipt.authorizes_target_update_free_claim);
        assert!(!receipt.authorizes_promotion);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_lora_import_rejects_unimplemented_orientation_and_scale_underflow() {
        let root = root("lora-exactness-guards");
        let base = root.join("base.safetensors");
        let adapter = root.join("adapter_model.safetensors");
        let config = root.join("adapter_config.json");
        write_fixture(&base);
        write_lora_fixture(&adapter);
        let input = LoraAdapterAxisInput {
            schema: LORA_ADAPTER_AXIS_INPUT_SCHEMA.into(),
            base_model_path: base,
            adapter_model_path: adapter,
            adapter_config_path: config.clone(),
        };

        fs::write(
            &config,
            serde_json::to_vec(&json!({
                "r":1,
                "lora_alpha":2.0,
                "target_modules":["q_proj","v_proj"],
                "bias":"none",
                "fan_in_fan_out":true
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(import_peft_lora_as_dense_axis(&root, &input).is_err());

        fs::write(
            &config,
            serde_json::to_vec(&json!({
                "r":1,
                "lora_alpha":1e-320,
                "target_modules":["q_proj","v_proj"],
                "bias":"none",
                "fan_in_fan_out":false
            }))
            .unwrap(),
        )
        .unwrap();
        let error = import_peft_lora_as_dense_axis(&root, &input)
            .unwrap_err()
            .to_string();
        assert!(error.contains("lora_adapter_scale_invalid"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_actuator_changes_only_selected_resident_tensor() {
        let root = root("direct");
        let base = root.join("base.safetensors");
        let output = root.join("output.safetensors");
        write_fixture(&base);
        let inventory = inspect_model_safetensors(&base).unwrap();
        assert_eq!(inventory.tensor_count, 2);
        assert_eq!(inventory.total_parameter_count, 8);
        let q = TensorId::parse("model.layers.0.self_attn.q_proj.weight").unwrap();
        let v = TensorId::parse("model.layers.0.self_attn.v_proj.weight").unwrap();
        let layout = parameter_layout_for_tensors(&inventory, std::slice::from_ref(&q)).unwrap();
        let delta = create_content_addressed_dvec(&root, &[0.5, -0.5, 1.0, -1.0]).unwrap();
        assert_eq!(read_dvec_f32(&root, &delta).unwrap().len(), 4);
        let receipt = materialize_dense_delta_checkpoint(
            &root,
            &base,
            &inventory.model_sha256,
            &layout,
            &delta,
            &output,
        )
        .unwrap();
        assert_eq!(
            receipt.lifecycle,
            WeightMaterializationLifecycle::CandidateOnlyNotPromoted
        );
        assert!(!receipt.requires_adapter_at_runtime);
        assert!(!receipt.authorizes_promotion);
        assert_eq!(receipt.modified_tensor_count, 1);
        authenticate_weight_materialization_receipt(
            &root, &base, &layout, &delta, &output, &receipt,
        )
        .unwrap();
        let mut forged_receipt = receipt.clone();
        forged_receipt.output_model_byte_len += 1;
        assert!(authenticate_weight_materialization_receipt(
            &root,
            &base,
            &layout,
            &delta,
            &output,
            &forged_receipt,
        )
        .is_err());
        assert_eq!(
            read_model_tensor_f32(&output, &q).unwrap(),
            vec![1.5, 1.5, 4.0, 3.0]
        );
        assert_eq!(
            read_model_tensor_f32(&output, &v).unwrap(),
            vec![-1.0, -2.0, -3.0, -4.0]
        );
        let after = inspect_model_safetensors(&output).unwrap();
        let q_after = after
            .tensors
            .iter()
            .find(|tensor| tensor.tensor_id == q)
            .unwrap();
        let v_after = after
            .tensors
            .iter()
            .find(|tensor| tensor.tensor_id == v)
            .unwrap();
        assert_eq!(q_after.dtype, "F32");
        assert_eq!(v_after.dtype, "BF16");

        let mut changed = fs::read(&output).unwrap();
        *changed.last_mut().unwrap() ^= 1;
        fs::write(&output, changed).unwrap();
        assert!(authenticate_weight_materialization_receipt(
            &root, &base, &layout, &delta, &output, &receipt,
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_actuator_rejects_wrong_binding_and_never_overwrites() {
        let root = root("guards");
        let base = root.join("base.safetensors");
        let output = root.join("output.safetensors");
        write_fixture(&base);
        let inventory = inspect_model_safetensors(&base).unwrap();
        let q = TensorId::parse("model.layers.0.self_attn.q_proj.weight").unwrap();
        let layout = parameter_layout_for_tensors(&inventory, &[q]).unwrap();
        let delta = create_content_addressed_dvec(&root, &[0.0; 4]).unwrap();
        let wrong = Sha256Digest::digest_bytes(b"wrong-model");
        assert!(
            materialize_dense_delta_checkpoint(&root, &base, &wrong, &layout, &delta, &output)
                .is_err()
        );
        assert!(!output.exists());

        let receipt = materialize_dense_delta_checkpoint(
            &root,
            &base,
            &inventory.model_sha256,
            &layout,
            &delta,
            &output,
        )
        .unwrap();
        let before = fs::read(&output).unwrap();
        assert!(materialize_dense_delta_checkpoint(
            &root,
            &base,
            &inventory.model_sha256,
            &layout,
            &delta,
            &output,
        )
        .is_err());
        assert_eq!(fs::read(&output).unwrap(), before);
        assert_eq!(
            receipt.output_model_sha256,
            inspect_model_safetensors(&output).unwrap().model_sha256
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn layout_requires_real_unique_tensor_identities() {
        let root = root("layout");
        let base = root.join("base.safetensors");
        write_fixture(&base);
        let inventory = inspect_model_safetensors(&base).unwrap();
        let q = TensorId::parse("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert!(parameter_layout_for_tensors(&inventory, &[q.clone(), q]).is_err());
        assert!(parameter_layout_for_tensors(
            &inventory,
            &[TensorId::parse("model.layers.9.missing.weight").unwrap()]
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    fn assert_consumed_mutation_rejected(label: &str, change_delta: bool, change_modified: bool) {
        let root = root(label);
        let base = root.join("base.safetensors");
        let output = root.join("output.safetensors");
        write_fixture(&base);
        let archive = SafeTensorArchive::open(&base).unwrap();
        let inventory = archive.inventory.clone();
        let q = TensorId::parse("model.layers.0.self_attn.q_proj.weight").unwrap();
        let v = TensorId::parse("model.layers.0.self_attn.v_proj.weight").unwrap();
        let layout = parameter_layout_for_tensors(&inventory, std::slice::from_ref(&q)).unwrap();
        let delta = create_content_addressed_dvec(&root, &[0.5, -0.5, 1.0, -1.0]).unwrap();
        let watched_tensor = if change_modified || change_delta {
            q
        } else {
            v
        };
        let tensor_offset =
            archive.data_start + archive.tensor(&watched_tensor).unwrap().data_start;
        drop(archive);
        let changed_path = if change_delta {
            delta.path.clone()
        } else {
            base.clone()
        };
        let original = fs::read(&changed_path).unwrap();
        let expected = if change_delta {
            delta.sha256.clone()
        } else {
            inventory.model_sha256.clone()
        };
        let restored = Rc::new(Cell::new(false));
        let observed = Rc::clone(&restored);
        let guard = ReadHookGuard::install(move |point| match point {
            ReadPoint::InputsAuthenticated => {
                let mut file = OpenOptions::new().write(true).open(&changed_path).unwrap();
                if change_delta {
                    // The dvec header is 8 magic bytes plus a u64 count.
                    file.seek(SeekFrom::Start(16)).unwrap();
                    file.write_all(&9.0_f32.to_le_bytes()).unwrap();
                } else {
                    file.seek(SeekFrom::Start(tensor_offset)).unwrap();
                    file.write_all(&bf16_bytes(&[9.0])).unwrap();
                }
                file.sync_all().unwrap();
            }
            ReadPoint::TensorConsumed(tensor_id) if tensor_id == watched_tensor => {
                let mut file = OpenOptions::new().write(true).open(&changed_path).unwrap();
                file.seek(SeekFrom::Start(0)).unwrap();
                file.write_all(&original).unwrap();
                file.sync_all().unwrap();
                // The on-disk file is valid again before the failure check;
                // only authentication of consumed bytes detects this case.
                assert_eq!(
                    crate::artifact::sha256_file(&changed_path).unwrap(),
                    expected
                );
                observed.set(true);
            }
            ReadPoint::TensorConsumed(_) => {}
        });
        let result = materialize_dense_delta_checkpoint(
            &root,
            &base,
            &inventory.model_sha256,
            &layout,
            &delta,
            &output,
        );
        drop(guard);
        assert!(
            restored.get(),
            "fixture never reached its consumption boundary"
        );
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains(if change_delta {
                "artifact_stream_consumed_digest_mismatch"
            } else {
                "model_tensor_consumed_digest_mismatch"
            }),
            "{error}"
        );
        assert!(!output.exists());
        assert!(fs::read_dir(&root).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tidex.tmp")
        }));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn actuator_rejects_modified_tensor_changed_consumed_then_restored() {
        assert_consumed_mutation_rejected("modified-consumption", false, true);
    }

    #[test]
    fn actuator_rejects_copied_tensor_changed_consumed_then_restored() {
        assert_consumed_mutation_rejected("copied-consumption", false, false);
    }

    #[test]
    fn actuator_rejects_delta_changed_consumed_then_restored() {
        assert_consumed_mutation_rejected("delta-consumption", true, true);
    }

    #[test]
    fn named_tensor_reader_authenticates_consumed_bytes() {
        let root = root("named-read-consumption");
        let base = root.join("base.safetensors");
        write_fixture(&base);
        let mut archive = SafeTensorArchive::open(&base).unwrap();
        let tensor_id = TensorId::parse("model.layers.0.self_attn.q_proj.weight").unwrap();
        let offset = archive.data_start + archive.tensor(&tensor_id).unwrap().data_start;
        let mut file = OpenOptions::new().write(true).open(&base).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&bf16_bytes(&[9.0])).unwrap();
        file.sync_all().unwrap();
        let error = archive.read_f32_tensor(&tensor_id).unwrap_err().to_string();
        assert!(error.contains("model_tensor_consumed_digest_mismatch"));
        fs::remove_dir_all(root).unwrap();
    }

    fn write_readout_fixture(path: &Path, dtype: &str, shape: &[usize], bytes: &[u8]) {
        let mut header = serde_json::to_vec(&json!({
            "lm_head.weight": {
                "dtype": dtype, "shape": shape, "data_offsets": [0, bytes.len()]
            }
        }))
        .unwrap();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
    }

    #[test]
    fn linear_readout_extracts_real_f32_f16_bf16_rows_and_preserves_f64_difference() {
        let root = root("readout-storage");
        let tensor_id = TensorId::parse("lm_head.weight").unwrap();
        let values = [1.0f32, 2.0, 7.0, 8.0, 3.0, 4.0];
        let f32_bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let f16_bytes = [0x3c00u16, 0x4000, 0x4700, 0x4800, 0x4200, 0x4400]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        for (dtype, bytes) in [
            ("F32", f32_bytes),
            ("F16", f16_bytes),
            ("BF16", bf16_bytes(&values)),
        ] {
            let path = root.join(format!("{dtype}.safetensors"));
            write_readout_fixture(&path, dtype, &[3, 2], &bytes);
            let inspection = inspect_linear_readout(&path, &tensor_id, 2, 0).unwrap();
            assert_eq!(
                inspection.schema,
                "cerebro.tidex.linear_readout_inspection/v1"
            );
            assert_eq!(
                inspection.model_sha256,
                crate::artifact::sha256_file(&path).unwrap()
            );
            assert_eq!(inspection.input_dimension, 2);
            assert_eq!(inspection.tensor_id, tensor_id);
            assert_eq!((inspection.positive_row, inspection.negative_row), (2, 0));
            assert_eq!(inspection.positive_weights, vec![3.0, 4.0]);
            assert_eq!(inspection.negative_weights, vec![1.0, 2.0]);
            assert_eq!(inspection.difference_weights, vec![2.0, 2.0]);
        }
        let path = root.join("precision.safetensors");
        let small = 2.0_f32.powi(-25);
        let bytes = [1.0f32, small]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        write_readout_fixture(&path, "F32", &[2, 1], &bytes);
        let inspection = inspect_linear_readout(&path, &tensor_id, 0, 1).unwrap();
        assert_eq!(inspection.difference_weights, vec![1.0 - f64::from(small)]);
        assert_ne!(inspection.difference_weights[0], f64::from(1.0f32 - small));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linear_readout_rejects_invalid_rows_shapes_dtype_and_nonfinite_values() {
        let root = root("readout-guards");
        let tensor_id = TensorId::parse("lm_head.weight").unwrap();
        let path = root.join("head.safetensors");
        write_readout_fixture(&path, "F32", &[3, 2], &[0u8; 24]);
        for (positive, negative) in [(0, 0), (3, 1), (1, 3), (usize::MAX, 0)] {
            assert!(inspect_linear_readout(&path, &tensor_id, positive, negative).is_err());
        }
        assert!(
            inspect_linear_readout(&path, &TensorId::parse("missing.weight").unwrap(), 0, 1,)
                .is_err()
        );
        for shape in [vec![6], vec![1, 2, 3], vec![2, 4_097]] {
            let bytes = vec![0u8; shape.iter().product::<usize>() * 4];
            write_readout_fixture(&path, "F32", &shape, &bytes);
            assert!(inspect_linear_readout(&path, &tensor_id, 0, 1).is_err());
        }
        write_readout_fixture(&path, "F64", &[3, 2], &[0u8; 48]);
        assert!(inspect_linear_readout(&path, &tensor_id, 0, 1).is_err());
        // Both selected and unselected rows must be finite.
        for bad_index in [0usize, 2, 5] {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut values = [0.0f32; 6];
                values[bad_index] = invalid;
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                write_readout_fixture(&path, "F32", &[3, 2], &bytes);
                let error = inspect_linear_readout(&path, &tensor_id, 0, 2)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains("model_tensor_nonfinite_value"), "{error}");
            }
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linear_readout_streams_selected_row_across_chunk_boundary() {
        let root = root("readout-chunks");
        let tensor_id = TensorId::parse("lm_head.weight").unwrap();
        let path = root.join("head.safetensors");
        let width = 4_093usize;
        let positive_row = STREAM_ELEMENTS / width;
        let row_count = positive_row + 2;
        let bytes = (0..row_count * width)
            .flat_map(|index| (index as f32).to_le_bytes())
            .collect::<Vec<_>>();
        write_readout_fixture(&path, "F32", &[row_count, width], &bytes);
        let inspection = inspect_linear_readout(&path, &tensor_id, positive_row, 0).unwrap();
        assert_eq!(inspection.positive_weights.len(), width);
        assert_eq!(
            inspection.positive_weights[0],
            (positive_row * width) as f64
        );
        assert_eq!(
            inspection.positive_weights[width - 1],
            ((positive_row + 1) * width - 1) as f64
        );
        assert!(inspection
            .difference_weights
            .iter()
            .all(|value| *value == (positive_row * width) as f64));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linear_readout_authenticates_unselected_bytes_changed_consumed_then_restored() {
        let root = root("readout-consumption");
        let path = root.join("head.safetensors");
        let tensor_id = TensorId::parse("lm_head.weight").unwrap();
        let bytes = [1.0f32, 2.0, 7.0, 8.0, 3.0, 4.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        write_readout_fixture(&path, "F32", &[3, 2], &bytes);
        let archive = SafeTensorArchive::open(&path).unwrap();
        let changed_offset =
            archive.data_start + archive.tensor(&tensor_id).unwrap().data_start + 8;
        let expected = archive.inventory.model_sha256.clone();
        drop(archive);
        let original = fs::read(&path).unwrap();
        let changed_path = path.clone();
        let watched_tensor = tensor_id.clone();
        let restored = Rc::new(Cell::new(false));
        let observed = Rc::clone(&restored);
        let guard = ReadHookGuard::install(move |point| match point {
            ReadPoint::InputsAuthenticated => {
                let mut file = OpenOptions::new().write(true).open(&changed_path).unwrap();
                // Row 1 is not requested; digest coverage must still include it.
                file.seek(SeekFrom::Start(changed_offset)).unwrap();
                file.write_all(&9.0_f32.to_le_bytes()).unwrap();
                file.sync_all().unwrap();
            }
            ReadPoint::TensorConsumed(tensor_id) if tensor_id == watched_tensor => {
                let mut file = OpenOptions::new().write(true).open(&changed_path).unwrap();
                file.seek(SeekFrom::Start(0)).unwrap();
                file.write_all(&original).unwrap();
                file.sync_all().unwrap();
                assert_eq!(
                    crate::artifact::sha256_file(&changed_path).unwrap(),
                    expected
                );
                observed.set(true);
            }
            ReadPoint::TensorConsumed(_) => {}
        });
        let result = inspect_linear_readout(&path, &tensor_id, 0, 2);
        drop(guard);
        assert!(
            restored.get(),
            "fixture did not reach the authenticated consumption boundary"
        );
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("model_tensor_consumed_digest_mismatch"),
            "{error}"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
