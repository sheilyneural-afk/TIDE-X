//! V67 engineering smoke: prove that TIDE-X Rust can materialize a known
//! receiver delta directly into SmolLM2 base weights.
//!
//! This experiment intentionally imports the already-proven V66 LoRA only as
//! a source of a known numerical delta. It does NOT claim that the functional
//! receiver compiler generated that delta. The scientific V67 experiment comes
//! after this data-plane actuator equivalence gate passes.

use cerebro_tidex::artifact::{sha256_file, ArtifactWriteAuthority, DeltaArtifactRef};
use cerebro_tidex::block_tomography::ParameterBlockLayout;
use cerebro_tidex::digest::Sha256Digest;
use cerebro_tidex::error::{BrainError, BrainResult};
use cerebro_tidex::identity::TensorId;
use cerebro_tidex::weight_actuator::{
    inspect_model_safetensors, materialize_dense_delta_checkpoint, parameter_layout_for_tensors,
    read_model_tensors_f32, ModelTensorSpec, WeightMaterializationReceipt,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "cerebro.tidex.v67_weight_actuator_smoke/v1";
const V66_MANIFEST_SCHEMA: &str = "cerebro.tidex.v66_compiled_adapter_artifact/v1";

fn invalid(code: impl Into<String>) -> BrainError {
    BrainError::Invalid(code.into())
}

fn integrity(code: impl Into<String>) -> BrainError {
    BrainError::Integrity(code.into())
}

#[derive(Debug)]
struct Args {
    private_root: PathBuf,
    base_model: PathBuf,
    adapter_dir: PathBuf,
    manifest: PathBuf,
    output_model: PathBuf,
    receipt: PathBuf,
    layout_output: PathBuf,
    delta_reference_output: PathBuf,
}

#[derive(Debug)]
struct AdapterPair {
    a: Option<TensorId>,
    b: Option<TensorId>,
}

#[derive(Debug)]
struct PatchFactors {
    target: TensorId,
    input_dim: usize,
    output_dim: usize,
    rank: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

struct LoRaDeltaIter<'a> {
    patches: &'a [PatchFactors],
    scale: f32,
    patch: usize,
    element: usize,
}

impl<'a> LoRaDeltaIter<'a> {
    fn new(patches: &'a [PatchFactors], scale: f32) -> Self {
        Self {
            patches,
            scale,
            patch: 0,
            element: 0,
        }
    }
}

impl Iterator for LoRaDeltaIter<'_> {
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

#[derive(Debug, Serialize)]
struct V67WeightActuatorSmokeReceipt {
    schema: String,
    purpose: String,
    source_delta_kind: String,
    source_v66_manifest_sha256: Sha256Digest,
    source_v66_capability: String,
    source_v66_parameter_digest_sha256: Sha256Digest,
    base_model_sha256: Sha256Digest,
    adapter_model_sha256: Sha256Digest,
    adapter_config_sha256: Sha256Digest,
    lora_rank: usize,
    lora_alpha: f64,
    lora_scale: f64,
    target_tensors: Vec<TensorId>,
    target_tensor_count: usize,
    dense_delta: DeltaArtifactRef,
    parameter_layout: ParameterBlockLayout,
    materialization: WeightMaterializationReceipt,
    claim_boundary: ClaimBoundary,
}

#[derive(Debug, Serialize)]
struct ClaimBoundary {
    rust_direct_weight_actuation_established: bool,
    output_requires_peft: bool,
    output_requires_lora_adapter: bool,
    v66_behavioral_delta_source_used: bool,
    functional_receiver_compiler_generated_target_delta: bool,
    target_capability_training_avoided_in_this_smoke: bool,
    universal_portability_established: bool,
}

fn parse_args() -> BrainResult<Args> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    if raw.len() % 2 != 0 {
        return Err(invalid("v67_smoke_arguments_must_be_flag_value_pairs"));
    }
    let mut values = BTreeMap::<String, String>::new();
    for pair in raw.chunks_exact(2) {
        let flag = &pair[0];
        if !flag.starts_with("--") || values.insert(flag.clone(), pair[1].clone()).is_some() {
            return Err(invalid("v67_smoke_argument_invalid_or_duplicate"));
        }
    }
    let take = |flag: &str| -> BrainResult<PathBuf> {
        values
            .get(flag)
            .map(PathBuf::from)
            .ok_or_else(|| invalid(format!("v67_smoke_missing_argument:{flag}")))
    };
    let allowed = BTreeSet::from([
        "--private-root",
        "--base-model",
        "--adapter-dir",
        "--manifest",
        "--output-model",
        "--receipt",
        "--layout-output",
        "--delta-reference-output",
    ]);
    if values.keys().any(|key| !allowed.contains(key.as_str())) || values.len() != allowed.len() {
        return Err(invalid("v67_smoke_argument_set_invalid"));
    }
    Ok(Args {
        private_root: take("--private-root")?,
        base_model: take("--base-model")?,
        adapter_dir: take("--adapter-dir")?,
        manifest: take("--manifest")?,
        output_model: take("--output-model")?,
        receipt: take("--receipt")?,
        layout_output: take("--layout-output")?,
        delta_reference_output: take("--delta-reference-output")?,
    })
}

fn path<'a>(value: &'a Value, keys: &[&str]) -> BrainResult<&'a Value> {
    let mut current = value;
    for key in keys {
        current = current
            .get(*key)
            .ok_or_else(|| integrity(format!("v67_manifest_field_missing:{}", keys.join("."))))?;
    }
    Ok(current)
}

fn required_str(value: &Value, keys: &[&str]) -> BrainResult<String> {
    path(value, keys)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| integrity(format!("v67_manifest_string_invalid:{}", keys.join("."))))
}

fn required_usize(value: &Value, keys: &[&str]) -> BrainResult<usize> {
    let raw = path(value, keys)?
        .as_u64()
        .ok_or_else(|| integrity(format!("v67_manifest_integer_invalid:{}", keys.join("."))))?;
    usize::try_from(raw).map_err(|_| invalid("v67_manifest_integer_overflow"))
}

fn required_f64(value: &Value, keys: &[&str]) -> BrainResult<f64> {
    path(value, keys)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| integrity(format!("v67_manifest_number_invalid:{}", keys.join("."))))
}

fn write_json_new<T: Serialize>(path: &Path, value: &T) -> BrainResult<()> {
    if !path.is_absolute() || fs::symlink_metadata(path).is_ok() {
        return Err(invalid("v67_output_path_invalid_or_existing"));
    }
    let parent = path
        .parent()
        .ok_or_else(|| invalid("v67_output_parent_missing"))?;
    fs::create_dir_all(parent)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    let mut file = options.open(path)?;
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn derive_target(adapter_id: &TensorId, factor: char) -> BrainResult<TensorId> {
    let suffix = match factor {
        'A' => ".lora_A.weight",
        'B' => ".lora_B.weight",
        _ => return Err(invalid("v67_lora_factor_invalid")),
    };
    let text = adapter_id
        .as_str()
        .strip_prefix("base_model.model.")
        .ok_or_else(|| integrity(format!("v67_adapter_prefix_invalid:{adapter_id}")))?;
    let stem = text
        .strip_suffix(suffix)
        .ok_or_else(|| integrity(format!("v67_adapter_suffix_invalid:{adapter_id}")))?;
    TensorId::parse(format!("{stem}.weight"))
}

fn adapter_pairs(specs: &[ModelTensorSpec]) -> BrainResult<BTreeMap<TensorId, AdapterPair>> {
    let mut result = BTreeMap::<TensorId, AdapterPair>::new();
    for spec in specs {
        if spec.dtype != "F32" {
            return Err(integrity(format!(
                "v67_adapter_tensor_dtype_not_f32:{}:{}",
                spec.tensor_id, spec.dtype
            )));
        }
        let text = spec.tensor_id.as_str();
        let (factor, target) = if text.ends_with(".lora_A.weight") {
            ('A', derive_target(&spec.tensor_id, 'A')?)
        } else if text.ends_with(".lora_B.weight") {
            ('B', derive_target(&spec.tensor_id, 'B')?)
        } else {
            return Err(integrity(format!(
                "v67_adapter_contains_non_lora_tensor:{}",
                spec.tensor_id
            )));
        };
        let pair = result
            .entry(target)
            .or_insert(AdapterPair { a: None, b: None });
        let slot = if factor == 'A' {
            &mut pair.a
        } else {
            &mut pair.b
        };
        if slot.replace(spec.tensor_id.clone()).is_some() {
            return Err(integrity("v67_adapter_duplicate_factor"));
        }
    }
    if result.is_empty()
        || result
            .values()
            .any(|pair| pair.a.is_none() || pair.b.is_none())
    {
        return Err(integrity("v67_adapter_factor_pair_incomplete"));
    }
    Ok(result)
}

fn prepare_patches(
    base_specs: &[ModelTensorSpec],
    adapter_specs: &[ModelTensorSpec],
    adapter_path: &Path,
    pairs: &BTreeMap<TensorId, AdapterPair>,
    rank: usize,
) -> BrainResult<(Vec<TensorId>, Vec<PatchFactors>)> {
    if rank == 0 {
        return Err(invalid("v67_lora_rank_zero"));
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
        return Err(integrity("v67_adapter_targets_missing_from_base_model"));
    }
    let requested_adapter_ids = targets
        .iter()
        .flat_map(|target| {
            let pair = &pairs[target];
            [pair.a.clone().unwrap(), pair.b.clone().unwrap()]
        })
        .collect::<Vec<_>>();
    let tensors = read_model_tensors_f32(adapter_path, &requested_adapter_ids)?;

    let mut patches = Vec::with_capacity(targets.len());
    for target in &targets {
        if !(target.as_str().ends_with(".q_proj.weight")
            || target.as_str().ends_with(".v_proj.weight"))
        {
            return Err(integrity(format!(
                "v67_adapter_target_outside_qv_contract:{target}"
            )));
        }
        let target_spec = base_by_id
            .get(target)
            .ok_or_else(|| integrity("v67_base_target_spec_missing"))?;
        if target_spec.shape.len() != 2 {
            return Err(integrity(format!("v67_base_target_not_matrix:{target}")));
        }
        let output_dim = target_spec.shape[0];
        let input_dim = target_spec.shape[1];
        let pair = &pairs[target];
        let a_id = pair.a.as_ref().unwrap();
        let b_id = pair.b.as_ref().unwrap();
        let a_spec = adapter_by_id
            .get(a_id)
            .ok_or_else(|| integrity("v67_adapter_a_spec_missing"))?;
        let b_spec = adapter_by_id
            .get(b_id)
            .ok_or_else(|| integrity("v67_adapter_b_spec_missing"))?;
        if a_spec.shape != vec![rank, input_dim] || b_spec.shape != vec![output_dim, rank] {
            return Err(integrity(format!(
                "v67_lora_factor_shape_mismatch:{target}"
            )));
        }
        let a = tensors
            .get(a_id)
            .cloned()
            .ok_or_else(|| integrity("v67_adapter_a_values_missing"))?;
        let b = tensors
            .get(b_id)
            .cloned()
            .ok_or_else(|| integrity("v67_adapter_b_values_missing"))?;
        patches.push(PatchFactors {
            target: target.clone(),
            input_dim,
            output_dim,
            rank,
            a,
            b,
        });
    }
    Ok((targets, patches))
}

fn run(args: &Args) -> BrainResult<V67WeightActuatorSmokeReceipt> {
    for path in [
        &args.output_model,
        &args.receipt,
        &args.layout_output,
        &args.delta_reference_output,
    ] {
        if !path.is_absolute() || fs::symlink_metadata(path).is_ok() {
            return Err(invalid(format!(
                "v67_output_not_new_absolute:{}",
                path.display()
            )));
        }
    }
    let writer = ArtifactWriteAuthority::open(&args.private_root)?;
    let manifest_bytes = fs::read(&args.manifest)?;
    let manifest_sha256 = Sha256Digest::digest_bytes(&manifest_bytes);
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    if manifest.get("schema").and_then(Value::as_str) != Some(V66_MANIFEST_SCHEMA) {
        return Err(integrity("v67_v66_manifest_schema_invalid"));
    }
    if path(
        &manifest,
        &["claim_boundary", "direct_weight_translation_established"],
    )?
    .as_bool()
        != Some(false)
    {
        return Err(integrity("v67_source_manifest_claim_boundary_changed"));
    }

    let adapter_model = args.adapter_dir.join("adapter_model.safetensors");
    let adapter_config = args.adapter_dir.join("adapter_config.json");
    let expected_adapter_sha = Sha256Digest::parse(required_str(
        &manifest,
        &["adapter", "files_sha256", "adapter_model.safetensors"],
    )?)?;
    let expected_config_sha = Sha256Digest::parse(required_str(
        &manifest,
        &["adapter", "files_sha256", "adapter_config.json"],
    )?)?;
    let adapter_model_sha256 = sha256_file(&adapter_model)?;
    let adapter_config_sha256 = sha256_file(&adapter_config)?;
    if adapter_model_sha256 != expected_adapter_sha || adapter_config_sha256 != expected_config_sha
    {
        return Err(integrity("v67_adapter_file_digest_mismatch"));
    }

    let base_inventory = inspect_model_safetensors(&args.base_model)?;
    let expected_base_sha = Sha256Digest::parse(required_str(
        &manifest,
        &["receiver", "weight_file_sha256"],
    )?)?;
    if base_inventory.model_sha256 != expected_base_sha {
        return Err(integrity("v67_base_model_digest_mismatch"));
    }

    let rank = required_usize(&manifest, &["training", "lora_rank"])?;
    let alpha = required_f64(&manifest, &["training", "lora_alpha"])?;
    if rank == 0 || alpha <= 0.0 {
        return Err(integrity("v67_lora_recipe_invalid"));
    }
    let scale = alpha / rank as f64;
    if !scale.is_finite() || scale <= 0.0 || scale > f32::MAX as f64 {
        return Err(integrity("v67_lora_scale_invalid"));
    }
    let declared_targets = path(&manifest, &["training", "lora_target_modules"])?
        .as_array()
        .ok_or_else(|| integrity("v67_lora_target_modules_invalid"))?
        .iter()
        .map(|value| value.as_str().map(str::to_string))
        .collect::<Option<BTreeSet<_>>>()
        .ok_or_else(|| integrity("v67_lora_target_modules_invalid"))?;
    if declared_targets != BTreeSet::from(["q_proj".into(), "v_proj".into()]) {
        return Err(integrity("v67_lora_target_modules_changed"));
    }

    let adapter_inventory = inspect_model_safetensors(&adapter_model)?;
    let pairs = adapter_pairs(&adapter_inventory.tensors)?;
    let (targets, patches) = prepare_patches(
        &base_inventory.tensors,
        &adapter_inventory.tensors,
        &adapter_model,
        &pairs,
        rank,
    )?;
    if patches.iter().map(|patch| &patch.target).ne(targets.iter()) {
        return Err(integrity("v67_patch_target_order_mismatch"));
    }
    let layout = parameter_layout_for_tensors(&base_inventory, &targets)?;
    let delta_iter = LoRaDeltaIter::new(&patches, scale as f32);
    let dense_delta =
        writer.create_content_addressed_dvec_iter(layout.total_parameter_count, delta_iter)?;

    let materialization = materialize_dense_delta_checkpoint(
        &args.private_root,
        &args.base_model,
        &base_inventory.model_sha256,
        &layout,
        &dense_delta,
        &args.output_model,
    )?;
    if materialization.modified_tensor_count != targets.len()
        || materialization.modified_parameter_count != layout.total_parameter_count
        || materialization.requires_adapter_at_runtime
        || materialization.authorizes_promotion
    {
        return Err(integrity("v67_materialization_receipt_contract_mismatch"));
    }

    write_json_new(&args.layout_output, &layout)?;
    write_json_new(&args.delta_reference_output, &dense_delta)?;

    Ok(V67WeightActuatorSmokeReceipt {
        schema: SCHEMA.into(),
        purpose: "engineering proof that the Rust data plane can merge an authenticated receiver delta into standalone model weights; not functional-compiler evidence".into(),
        source_delta_kind: "v66_lora_reconstructed_as_dense_delta".into(),
        source_v66_manifest_sha256: manifest_sha256,
        source_v66_capability: required_str(&manifest, &["capability"])?,
        source_v66_parameter_digest_sha256: Sha256Digest::parse(required_str(
            &manifest,
            &["adapter", "parameter_digest_sha256"],
        )?)?,
        base_model_sha256: base_inventory.model_sha256,
        adapter_model_sha256,
        adapter_config_sha256,
        lora_rank: rank,
        lora_alpha: alpha,
        lora_scale: scale,
        target_tensor_count: targets.len(),
        target_tensors: targets,
        dense_delta,
        parameter_layout: layout,
        materialization,
        claim_boundary: ClaimBoundary {
            rust_direct_weight_actuation_established: true,
            output_requires_peft: false,
            output_requires_lora_adapter: false,
            v66_behavioral_delta_source_used: true,
            functional_receiver_compiler_generated_target_delta: false,
            target_capability_training_avoided_in_this_smoke: false,
            universal_portability_established: false,
        },
    })
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    match run(&args) {
        Ok(receipt) => {
            if let Err(error) = write_json_new(&args.receipt, &receipt) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
