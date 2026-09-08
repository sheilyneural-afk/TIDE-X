#!/usr/bin/env python3
# ruff: noqa: E402
"""V69: target-update-free cross-model capability compilation.

Confirmation experiment for the exact contract:

    donor functional signature for held-out C*
        -> receiver compiler calibrated only on C1..Cn
        -> receiver-native dense delta
        -> standalone SmolLM2 checkpoint

For every confirmation target C* this experiment forbids a target LoRA, target
receiver delta, target receiver solution, target SFT and target receiver
execution before all target candidates/checkpoints are sealed.  The donor may
be observed functionally; no donor parameter update is required or accepted.

The task family is deliberately narrow and synthetic (integer threshold
classification).  Passing V69 establishes this compilation property in that
bounded family, not universal semantic capability portability.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import itertools
import subprocess
import os
import random
import struct
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

from quality.experiments.v68_receiver_response_probe import (
    canonical,
    freeze_reproducibility_bundle,
    put,
    reference_file,
    read_reference,
    sha_file,
    tidex,
    write_new,
)

SCHEMA = "cerebro.tidex.v69_target_update_free_compilation/v4"
PROTOCOL_SCHEMA = "cerebro.tidex.cross_model_functional_signature_protocol/v2"
BASIS_SCHEMA = "cerebro.tidex.receiver_weight_basis/v1"
DISTRIBUTED_BASIS_SCHEMA = "cerebro.tidex.receiver_weight_basis/v2"
DISTRIBUTED_COMPILER_MANIFEST_SCHEMA = (
    "cerebro.tidex.v69_distributed_lora_compiler_manifest/v1"
)
DISTRIBUTED_COMPILER_RECEIPT_SCHEMA = (
    "cerebro.tidex.v69_distributed_lora_compiler_receipt/v1"
)
TARGET_SCHEMA = "cerebro.tidex.functional_response_target/v1"
OBSERVATION_SCHEMA = "cerebro.tidex.receiver_response_observation/v1"
REQUEST_SCHEMA = "cerebro.tidex.receiver_weight_request/v1"

DONOR_REVISION = "df3ce67c0e24480f20468b6ef2894622d69eb73b"
DONOR_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B/snapshots"
    / DONOR_REVISION
)
RECEIVER_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
RECEIVER_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots"
    / RECEIVER_REVISION
)
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)

SEED = 20260908
CALIBRATION_THRESHOLDS = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
# These six confirmation thresholds were not used as receiver targets during
# method development in this conversation. Do not change after first run.
CONFIRMATION_THRESHOLDS = [21, 31, 41, 51, 61, 71]
RETIRED_CONFIRMATION_THRESHOLDS = [
    24, 34, 44, 54, 64, 74,
    23, 33, 43, 53, 63, 73,
    22, 32, 42, 52, 62, 72,
    27, 37, 47, 57, 67, 77,
    28, 38, 48, 58, 68, 78,
    29, 39, 49, 59, 69, 79,
]
PRIOR_FAILED_CALIBRATION_RECEIPT_SHA256 = "8a865397648e6c18a5b6926be83e6bc2168a271a3a0fdc4cbe4629c06138c0e5"
PRIOR_FAILED_METHOD_PRECOMMIT_SHA256 = "75cccd79c4c3bca6081905f06c4160198fd86d28a58d3ec8657ac42c808aad81"
PRIOR_FAILED_RUN_RECEIPT_SHA256 = "97798a084ecdf29f1f0773ab29d12f5610cd989595492ad18c72bf2af393f1da"
PRIOR_PROTOCOL_FAILURE_PRECOMMIT_SHA256 = "dd736c98ee00b9722b1c2b848e3ef7672b0994fa08294515404672d1cb966de3"
PRIOR_PROTOCOL_FAILURE_CALIBRATION_SHA256 = "a4b574fb6325a0f549362903ee38da9674fd952ab385f9634ba382e57dda2c42"
PRIOR_PROTOCOL_FAILURE_RECEIPT_SHA256 = "96872a8e5b861771e5f8611c7bc840ad4a7e3d1e4d692a1cfb9ebc9425930df8"
PRIOR_SINGLE_AUTHORITY_FAILURE_PRECOMMIT_SHA256 = "8183bb12a3ac8dbfcaa0edffab73d64546dfe7e3dc81949af5afb845bb177703"
PRIOR_TYPED_EVIDENCE_FAILURE_PRECOMMIT_SHA256 = "a1ba9ebae2140aad48f6891ae24f763e2423a3b75f3eef287058eb6ee5a3e85d"
DONOR_PROBES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
DONOR_DEMO_OFFSETS = [-18, -13, -8, -3, 3, 8, 13, 18]
RECEIVER_TRAIN_OFFSETS = [-19, -14, -9, -4, 4, 9, 14, 19]
RECEIVER_VALIDATION_OFFSETS = [-17, -12, -7, -2, 2, 7, 12, 17]
FUNCTIONAL_DIM = 4
MIN_RECEIVER_PC_DIM = 2
MAX_RECEIVER_PC_DIM = 8
CALIBRATION_SOLUTION_RIDGE = 1e-2
COMPILER_RIDGE = 1.0
PROJECTION_ARITHMETIC = "f64_sequential_sub_mul_add/v1"
# Fixed, calibration-only comparison. Selecting a candidate does not turn this
# development score into a confirmatory estimate. No target chooses this grid.
COMPILER_METHODS = {
    "decode_then_project": {"proposal_method": "decode_then_project", "ridge": 1.0},
    "calibrated_affine_r001": {"proposal_method": "calibrated_affine", "ridge": 0.001},
    "calibrated_affine_r01": {"proposal_method": "calibrated_affine", "ridge": 0.01},
    "calibrated_affine_r1": {"proposal_method": "calibrated_affine", "ridge": 0.1},
    "calibrated_affine_r10": {"proposal_method": "calibrated_affine", "ridge": 1.0},
    "relational_anchors": {"proposal_method": "relational_anchors", "ridge": 1.0},
}
DESIRED_MARGIN = 3.0

GENERIC_CONTROLS = [
    "Statement: A triangle has three sides. Answer:",
    "Statement: Water is a metal. Answer:",
    "Statement: A book contains pages. Answer:",
    "Statement: A square has five sides. Answer:",
    "Statement: Ice is frozen water. Answer:",
    "Statement: The sun is a musical instrument. Answer:",
    "Statement: A week contains seven days. Answer:",
    "Statement: A spoon is a type of cloud. Answer:",
    "Statement: A circle has no corners. Answer:",
    "Statement: A tree is made from glass. Answer:",
]

# These prompts never construct the risk geometry or select a method. They
# measure preservation only after all checkpoints and baseline updates seal.
GENERIC_VALIDATION_CONTROLS = [
    "Statement: A year contains twelve months. Answer:",
    "Statement: A bicycle has six wheels. Answer:",
    "Statement: Birds have feathers. Answer:",
    "Statement: The moon is a vegetable. Answer:",
    "Statement: A kilogram is a unit of mass. Answer:",
    "Statement: A triangle has four sides. Answer:",
    "Statement: Fish live in water. Answer:",
    "Statement: Snow is made of hot sand. Answer:",
    "Statement: An hour contains sixty minutes. Answer:",
    "Statement: A piano is a kind of tree. Answer:",
]

CRITERIA = {
    "minimum_calibration_solution_mean_accuracy": 0.75,
    "minimum_calibration_compiler_loo_mean_accuracy": 0.55,
    "minimum_calibration_compiler_loo_gain": 0.05,
    "minimum_calibration_non_degrading_count": 10,
    "minimum_confirmation_mean_accuracy": 0.70,
    "minimum_confirmation_mean_gain": 0.15,
    "minimum_non_degrading_target_count": 4,
    "maximum_capability_cluster_two_sided_p": 0.05,
    "minimum_correct_vs_cyclic_wrong_advantage": 0.10,
    "maximum_generic_control_margin_rms_change": 0.20,
    "minimum_donor_confirmation_accuracy": 0.60,
    "minimum_receiver_basis_explained_energy": 0.90,
    "minimum_compiled_vs_strongest_calibration_baseline_advantage": 0.05,
}


def cap_id(threshold: int) -> str:
    return f"numeric.threshold.gt.{threshold:03d}:v1"


def inverse_cap_id(threshold: int) -> str:
    return f"numeric.threshold.le.{threshold:03d}:v1"


def numbers(threshold: int, offsets: list[int]) -> list[int]:
    return [threshold + offset for offset in offsets]


def expected_labels(threshold: int, values: list[int]) -> list[bool]:
    return [value > threshold for value in values]


def file_sha_map(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha_file(directory / name) for name in names}


def json_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def exact_f64_vector_sha256(domain: bytes, values: Any) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(struct.pack("<Q", len(array)))
    for value in array:
        digest.update(struct.pack("<d", float(value)))
    return digest.hexdigest()


def functional_signature_sha256(values: Any) -> str:
    return exact_f64_vector_sha256(
        b"cerebro.tidex.functional_signature_f64_le/v1\0", values
    )


def receiver_coordinates_sha256(values: Any) -> str:
    return exact_f64_vector_sha256(
        b"cerebro.tidex.receiver_coordinates_f64_le/v1\0", values
    )


def one_token(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"verbalizer is not one token: {text!r}: {ids}")
    return int(ids[0])


def exact_binomial_p(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = min(gained, lost)
    return min(
        1.0,
        2.0
        * sum(math.comb(discordant, index) for index in range(tail + 1))
        / (2**discordant),
    )


def canonical_project(raw: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Exact scalar f64 arithmetic shared with the Rust functional-IR authority."""
    raw, mean, components = (np.asarray(value, dtype=np.float64) for value in (raw, mean, components))
    if raw.ndim != 1 or mean.shape != raw.shape or components.ndim != 2 or components.shape[1] != len(raw):
        raise ValueError("functional projection shape mismatch")
    if not all(np.isfinite(value).all() for value in (raw, mean, components)):
        raise ValueError("nonfinite functional projection input")
    projected = []
    for component in components:
        accumulator = 0.0
        for value, center, coefficient in zip(raw, mean, component, strict=True):
            centered = float(value) - float(center)
            product = centered * float(coefficient)
            accumulator = accumulator + product
        if not math.isfinite(accumulator):
            raise ValueError("nonfinite functional projection output")
        projected.append(accumulator)
    return np.asarray(projected, dtype=np.float64)


def donor_raw_probe_hashes() -> list[str]:
    return [hashlib.sha256(f"Input: {value}\nOutput:".encode()).hexdigest() for value in DONOR_PROBES]


def pca_projection(rows: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if rows.ndim != 2 or rows.shape[0] <= dimension + 1:
        raise RuntimeError("PCA calibration matrix is underdetermined")
    mean = rows.mean(axis=0)
    centered = rows - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dimension]
    coordinates = centered @ components.T
    total = float(np.square(singular).sum())
    retained = float(np.square(singular[:dimension]).sum())
    energy = retained / max(total, 1e-30)
    return mean, components, coordinates, energy


def select_receiver_pca_dimension(rows: np.ndarray, minimum_energy: float) -> tuple[int, list[float]]:
    """Choose the smallest receiver rank justified by calibration only.

    The confirmation targets never enter this decision.  Returning the complete
    calibration energy curve makes the model-selection decision auditable and
    prevents a later target result from silently changing the receiver rank.
    """
    if rows.ndim != 2 or rows.shape[0] < 5:
        raise RuntimeError("receiver PCA selection requires a calibration matrix")
    maximum = min(MAX_RECEIVER_PC_DIM, rows.shape[0] - 2, rows.shape[1])
    if maximum < MIN_RECEIVER_PC_DIM:
        raise RuntimeError("receiver PCA selection has no admissible rank")
    centered = rows - rows.mean(axis=0)
    _u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
    energy = np.square(singular)
    total = float(energy.sum())
    curve = [float(energy[:rank].sum() / max(total, 1e-30)) for rank in range(1, maximum + 1)]
    for rank in range(MIN_RECEIVER_PC_DIM, maximum + 1):
        if curve[rank - 1] >= minimum_energy:
            return rank, curve
    raise RuntimeError(
        f"receiver calibration cannot reach required PCA energy {minimum_energy:.6f}; "
        f"maximum={curve[-1]:.6f} at rank={maximum}"
    )


def capability_cluster_sign_flip_p(gains: list[float]) -> float:
    """Exact paired randomization conditional on exchangeability of task signs.

    Questions within one checkpoint are never counted as independent trials.
    This does not imply six tasks are six independent training replications.
    """
    values = np.asarray(gains, dtype=np.float64)
    if not 1 <= len(values) <= 20 or not np.isfinite(values).all():
        raise ValueError("invalid capability-group effects")
    observed = abs(float(values.sum()))
    if observed == 0.0:
        return 1.0
    tolerance = 16 * float(np.finfo(np.float64).eps) * max(1.0, float(np.abs(values).sum()))
    # Count Python integers: summing NumPy bools leaks NumPy scalars into
    # the p-value and then into short-circuit gate results in the JSON receipt.
    extreme: int = sum(
        1
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        if abs(float(np.dot(signs, values))) >= observed - tolerance
    )
    return float(extreme / (2 ** len(values)))


def orthonormal_receiver_basis(axes: np.ndarray, coordinates: np.ndarray):
    """Keep exactly the same dense updates in Euclidean parameter coordinates.

    A mean vector followed by unit PCA directions is not orthonormal. Using
    its raw coefficients for cosine/R2 gates assigns an arbitrary geometry.
    QR changes the coordinate system, not the represented parameter subspace.
    """
    if axes.ndim != 2 or coordinates.ndim != 2 or coordinates.shape[1] != len(axes):
        raise ValueError("receiver basis coordinate shape mismatch")
    if not np.isfinite(axes).all() or not np.isfinite(coordinates).all():
        raise ValueError("nonfinite receiver basis")
    q, transform = np.linalg.qr(axes.T, mode="reduced")
    if q.shape[1] != len(axes):
        raise ValueError("receiver basis exceeds parameter dimension")
    diagonal = np.diag(transform)
    tolerance = np.finfo(np.float64).eps * max(axes.shape) * np.linalg.norm(axes, ord=2)
    if np.any(np.abs(diagonal) <= tolerance):
        raise ValueError("receiver basis is rank deficient")
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    transform = transform * signs[:, None]
    new_axes = q.T
    new_coordinates = coordinates @ transform.T
    np.testing.assert_allclose(new_axes @ new_axes.T, np.eye(len(axes)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(new_coordinates @ new_axes, coordinates @ axes, atol=1e-10, rtol=1e-10)
    return new_axes, new_coordinates


def fit_calibration_basis(raw: np.ndarray, deltas: np.ndarray) -> dict[str, Any]:
    """Fit a receiver compiler basis from exactly the supplied training rows."""
    if raw.ndim != 2 or deltas.ndim != 2 or len(raw) != len(deltas):
        raise ValueError("calibration row mismatch")
    if not np.isfinite(raw).all() or not np.isfinite(deltas).all():
        raise ValueError("nonfinite calibration")
    mean, components, _unused, functional_energy = pca_projection(raw, FUNCTIONAL_DIM)
    # One row-wise materialization is used by inputs, evidence and predictions.
    functions = np.stack([canonical_project(row, mean, components) for row in raw])
    rank, curve = select_receiver_pca_dimension(
        deltas, CRITERIA["minimum_receiver_basis_explained_energy"]
    )
    receiver_mean, receiver_components, scores, energy = pca_projection(deltas, rank)
    axes = np.concatenate([receiver_mean[None, :], receiver_components], axis=0)
    coordinates = np.concatenate([np.ones((len(deltas), 1)), scores], axis=1)
    axes, coordinates = orthonormal_receiver_basis(axes, coordinates)
    return {
        "functional_mean": mean, "functional_components": components,
        "functional_signatures": functions, "functional_energy": functional_energy,
        "axes": axes, "coordinates": coordinates, "receiver_rank": rank,
        "receiver_energy_curve": curve, "receiver_energy": energy,
    }


def calibration_fold(raw: np.ndarray, deltas: np.ndarray, holdout: int) -> dict[str, Any]:
    if not 0 <= holdout < len(raw) or len(raw) != len(deltas):
        raise ValueError("invalid calibration holdout")
    indices = [index for index in range(len(raw)) if index != holdout]
    fitted = fit_calibration_basis(raw[indices], deltas[indices])
    fitted["training_indices"] = indices
    fitted["requested"] = canonical_project(
        raw[holdout], fitted["functional_mean"], fitted["functional_components"]
    )
    return fitted


def coordinate_risk(control_features: np.ndarray, axes: np.ndarray, coordinates: np.ndarray):
    jacobian = control_features @ axes.T
    metric = jacobian.T @ jacobian / len(control_features)
    metric = metric / max(float(np.trace(metric)), 1e-12) + np.eye(len(axes)) * 1e-9
    costs = np.einsum("ni,ij,nj->n", coordinates, metric, coordinates)
    return jacobian, metric, float(1.25 * np.max(costs))


def compiler_policy(risk_budget: float, ridge: float = COMPILER_RIDGE) -> dict[str, Any]:
    return {
        "schema": "cerebro.tidex.receiver_compiler_policy/v1",
        "ridge": ridge,
        "minimum_decoder_loo_r2": 0.0,
        "minimum_encoder_loo_r2": 0.0,
        "minimum_decoder_loo_cosine": 0.0,
        "maximum_functional_relative_error": 0.50,
        "minimum_identity_margin": 0.05,
        "maximum_quadratic_cost": risk_budget,
    }


def rust_signature_probe(binary: Path, root: Path, label: str, fitted: dict[str, Any],
                         requested: np.ndarray, wrong: np.ndarray,
                         control_features: np.ndarray, method: str, ridge: float = COMPILER_RIDGE) -> dict[str, Any]:
    _jacobian, metric, budget = coordinate_risk(
        control_features, fitted["axes"], fitted["coordinates"]
    )
    request = {
        "schema": "cerebro.tidex.receiver_signature_benchmark_input/v1",
        "requested": requested.tolist(),
        "calibration": {
            "functional_signatures": fitted["functional_signatures"].tolist(),
            "receiver_solutions": fitted["coordinates"].tolist(),
            "wrong_functional_signatures": [wrong.tolist()],
        },
        "protected_cortex": {
            "parameter_importance": [0.0] * len(fitted["axes"]),
            "directions": [], "max_damage_ratio": 0.0,
        },
        "risk_metric": metric.tolist(), "policy": compiler_policy(budget, ridge),
        "proposal_method": method,
        "validation_profile": "parametric_cross_validation",
    }
    path = root / "calibration-kernel" / f"{label}.json"
    write_new(path, canonical(request))
    process = subprocess.run(
        [str(binary), "benchmark", "response", str(path)],
        capture_output=True, text=True, timeout=240, check=True,
    )
    report = json.loads(process.stdout)
    write_new(path.with_suffix(".report.json"), canonical(report))
    return report


def load_receiver(directory: Path, threads: int) -> tuple[Any, Any]:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    if type(model).__name__ != "LlamaForCausalLM":
        raise RuntimeError("V69 receiver must be the inspected LlamaForCausalLM")
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("receiver contains adapter parameters")
    return model, tokenizer


def receiver_prompt_features(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    true_id: int,
    false_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    norm = model.model.norm
    captured: list[torch.Tensor] = []
    handle = norm.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach()))
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    positions = encoded.attention_mask.sum(1) - 1
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("ambiguous final-norm capture")
    hidden = captured[0][torch.arange(len(prompts)), positions].float()
    normalized = hidden * torch.rsqrt(
        hidden.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
    )
    head_difference = model.lm_head.weight[true_id] - model.lm_head.weight[false_id]
    features = (normalized * head_difference).double().cpu().numpy()
    margins = (
        output.logits[torch.arange(len(prompts)), positions, true_id]
        - output.logits[torch.arange(len(prompts)), positions, false_id]
    ).double().cpu().numpy()
    return features, margins


def receiver_features(model: Any, tokenizer: Any, values: list[int],
                      true_id: int, false_id: int) -> tuple[np.ndarray, np.ndarray]:
    return receiver_prompt_features(model, tokenizer,
        [f"Input: {value}\nOutput:" for value in values], true_id, false_id)


def receiver_prompt_margins(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    true_id: int,
    false_id: int,
) -> np.ndarray:
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    positions = encoded.attention_mask.sum(1) - 1
    with torch.inference_mode():
        output = model(**encoded, use_cache=False)
    return (
        output.logits[torch.arange(len(prompts)), positions, true_id]
        - output.logits[torch.arange(len(prompts)), positions, false_id]
    ).double().cpu().numpy()


def donor_task_prompts(threshold: int, inverse: bool, queries: list[int]) -> list[str]:
    demonstrations = numbers(threshold, DONOR_DEMO_OFFSETS)
    lines = []
    for value in demonstrations:
        positive = value > threshold
        if inverse:
            positive = not positive
        lines.append(f"Input: {value}\nOutput: {'true' if positive else 'false'}")
    return ["\n".join([*lines, f"Input: {query}\nOutput:"]) for query in queries]


def donor_signature(
    model: Any,
    tokenizer: Any,
    threshold: int,
    inverse: bool,
    true_id: int,
    false_id: int,
) -> tuple[np.ndarray, list[str]]:
    prompts = donor_task_prompts(threshold, inverse, DONOR_PROBES)
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    positions = encoded.attention_mask.sum(1) - 1
    with torch.inference_mode():
        output = model(**encoded, use_cache=False)
    margins = (
        output.logits[torch.arange(len(prompts)), positions, true_id]
        - output.logits[torch.arange(len(prompts)), positions, false_id]
    ).double().cpu().numpy()
    return margins, prompts


def evaluation_reveal() -> dict[str, Any]:
    cases = []
    for threshold in CONFIRMATION_THRESHOLDS:
        for value, expected in zip(
            numbers(threshold, RECEIVER_VALIDATION_OFFSETS),
            expected_labels(threshold, numbers(threshold, RECEIVER_VALIDATION_OFFSETS)),
            strict=True,
        ):
            cases.append(
                {
                    "capability_id": cap_id(threshold),
                    "threshold": threshold,
                    "input": value,
                    "expected_true": expected,
                    "prompt": f"Input: {value}\nOutput:",
                }
            )
    return {
        "schema": "cerebro.tidex.v69_confirmation_reveal/v1",
        "cases": cases,
        "generic_controls": GENERIC_VALIDATION_CONTROLS,
    }


def replay_model(model_dir: Path, package_path: Path, output_path: Path, threads: int) -> None:
    package = json.loads(package_path.read_bytes())
    expected_model_sha = package["model_sha256"]
    if file_sha_map(model_dir, MODEL_FILES) != package["model_files_sha256"]:
        raise RuntimeError("V69 replay sidecar identity mismatch")
    if sha_file(model_dir / "model.safetensors") != expected_model_sha:
        raise RuntimeError("V69 replay checkpoint digest mismatch")
    if any(
        path.is_file() and ("adapter" in path.name.lower() or "lora" in path.name.lower())
        for path in model_dir.iterdir()
    ):
        raise RuntimeError("V69 standalone checkpoint contains adapter artifacts")
    model, tokenizer = load_receiver(model_dir, threads)
    true_id = one_token(tokenizer, " true")
    false_id = one_token(tokenizer, " false")
    reveal = package["reveal"]
    prompts = [case["prompt"] for case in reveal["cases"]]
    case_margins = receiver_prompt_margins(model, tokenizer, prompts, true_id, false_id)
    control_margins = receiver_prompt_margins(
        model, tokenizer, reveal["generic_controls"], true_id, false_id
    )
    result = {
        "schema": "cerebro.tidex.v69_standalone_replay/v1",
        "model_sha256": expected_model_sha,
        "model_class": type(model).__name__,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peft_module_imported": "peft" in sys.modules,
        "lora_parameters_present": any("lora_" in name for name, _ in model.named_parameters()),
        "case_margins": case_margins.tolist(),
        "generic_control_margins": control_margins.tolist(),
    }
    write_new(output_path, canonical(result))


def verify_frozen_inputs(reproducibility: dict[str, str], binary: Path) -> None:
    path = Path(reproducibility["path"])
    if sha_file(path) != reproducibility["sha256"]:
        raise RuntimeError("reproducibility manifest changed")
    manifest = json.loads(path.read_bytes())
    if sha_file(binary) != manifest["tidex_binary_sha256"]:
        raise RuntimeError("compiler binary changed during experiment")
    for relative, expected in manifest["source_sha256"].items():
        if sha_file(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"source changed during experiment: {relative}")


def prepare_calibration_baselines(binary: Path, root: Path, thresholds: list[int],
        functions: np.ndarray, coordinates: np.ndarray, axes: np.ndarray,
        deltas: np.ndarray, targets: dict[str, np.ndarray],
        control_features: np.ndarray, method: str, ridge: float) -> dict[str, Any]:
    """Precommit all control interventions before seeing receiver target scores."""
    rows = {}
    permutation = np.roll(np.arange(len(coordinates)), 1)
    shuffled = {"functional_signatures": functions,
                "coordinates": coordinates[permutation], "axes": axes}
    for threshold in thresholds:
        requested = targets[cap_id(threshold)]
        nearest = int(np.argmin(np.linalg.norm(functions - requested, axis=1)))
        report = rust_signature_probe(binary, root, f"baseline-shuffled-{threshold}",
            shuffled, requested, targets[inverse_cap_id(threshold)], control_features, method, ridge)
        shuffled_delta = np.asarray(report["target_delta"]) @ axes.astype(np.float32).astype(np.float64)
        rows[str(threshold)] = {
            "mean_no_target_signature": deltas.mean(axis=0).astype(np.float32).tolist(),
            "nearest_signature": deltas[nearest].astype(np.float32).tolist(),
            "shuffled_correspondence": shuffled_delta.astype(np.float32).tolist(),
            "nearest_calibration_capability_id": cap_id(CALIBRATION_THRESHOLDS[nearest]),
            "shuffled_numerical_gates_allowed": report["allowed"],
        }
    return {
        "schema": "cerebro.tidex.v69_calibration_baseline_updates/v1",
        "calibration_capability_ids": [cap_id(t) for t in CALIBRATION_THRESHOLDS],
        "target_receiver_data_used": False, "calibration_permutation": permutation.tolist(),
        "target_updates": rows,
        "negative_control_policy": "shuffled predictions are evaluated even when numerically rejected; never installed or promoted",
    }


def replay_baselines(model_dir: Path, package_path: Path, output_path: Path, threads: int) -> None:
    package = json.loads(package_path.read_bytes())
    if sha_file(model_dir / "model.safetensors") != package["model_sha256"]:
        raise RuntimeError("baseline base checkpoint changed")
    if file_sha_map(model_dir, MODEL_FILES) != package["model_files_sha256"]:
        raise RuntimeError("baseline model sidecars changed")
    updates = package["updates"]
    if json_sha(updates) != package["updates_sha256"]:
        raise RuntimeError("baseline update commitment mismatch")
    model, tokenizer = load_receiver(model_dir, threads)
    original = model.model.norm.weight.detach().clone()
    true_id, false_id = one_token(tokenizer, " true"), one_token(tokenizer, " false")
    results = {}
    for threshold, row in updates["target_updates"].items():
        cases = [case for case in package["reveal"]["cases"]
                 if case["capability_id"] == cap_id(int(threshold))]
        results[threshold] = {}
        for method in ("mean_no_target_signature", "nearest_signature", "shuffled_correspondence"):
            delta = np.asarray(row[method], dtype=np.float32)
            if delta.shape != (original.numel(),) or not np.isfinite(delta).all():
                raise RuntimeError("invalid baseline update")
            with torch.no_grad():
                model.model.norm.weight.copy_(original + torch.from_numpy(delta))
            try:
                margins = receiver_prompt_margins(model, tokenizer,
                    [case["prompt"] for case in cases], true_id, false_id)
                control = receiver_prompt_margins(model, tokenizer,
                    package["reveal"]["generic_controls"], true_id, false_id)
                results[threshold][method] = {"case_margins": margins.tolist(),
                    "generic_control_margins": control.tolist()}
            finally:
                with torch.no_grad():
                    model.model.norm.weight.copy_(original)
    write_new(output_path, canonical({
        "schema": "cerebro.tidex.v69_baseline_replay/v1",
        "model_sha256": package["model_sha256"], "updates_sha256": package["updates_sha256"],
        "model_class": type(model).__name__, "results": results,
        "peft_module_imported": "peft" in sys.modules,
        "lora_parameters_present": any("lora_" in name for name, _ in model.named_parameters()),
    }))


def copy_model_sidecars(source: Path, destination: Path, expected: dict[str, str]) -> None:
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, digest in expected.items():
        if sha_file(source / name) != digest:
            raise RuntimeError("model sidecar changed during V69")
        target = destination / name
        with (source / name).open("rb") as reader, target.open("xb") as writer:
            while chunk := reader.read(4 * 1024 * 1024):
                writer.write(chunk)
        target.chmod(0o600)


def assemble_distributed_lora_compiler_basis(
    binary: Path, root: Path, manifest_path: Path
) -> dict[str, Any]:
    """Build the authenticated V69 v2 basis from calibration-only PEFT axes.

    This path performs no target observation, target receiver execution or
    target optimization. Rust reconstructs every adapter as its exact dense
    multi-block delta and replays those imports during basis assembly.
    """
    if root.exists():
        raise RuntimeError("distributed compiler root already exists; refusing overwrite")
    if not root.is_absolute() or PROJECT_ROOT in root.parents or root == PROJECT_ROOT:
        raise RuntimeError("distributed compiler root must be absolute and outside checkout")
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    required = {
        "schema",
        "base_model_path",
        "receiver_model_id",
        "receiver_revision",
        "total_model_parameter_count",
        "total_transformer_layers",
        "calibration_axes",
        "confirmation_target_capability_ids_used",
        "receiver_backend_frozen_before_target_observation",
        "target_receiver_solution_used",
        "target_receiver_execution_performed",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise RuntimeError("distributed compiler manifest fields are not exact")
    if (
        manifest["schema"] != DISTRIBUTED_COMPILER_MANIFEST_SCHEMA
        or manifest["confirmation_target_capability_ids_used"] != []
        or manifest["receiver_backend_frozen_before_target_observation"] is not True
        or manifest["target_receiver_solution_used"] is not False
        or manifest["target_receiver_execution_performed"] is not False
        or type(manifest["total_model_parameter_count"]) is not int
        or manifest["total_model_parameter_count"] <= 0
        or type(manifest["total_transformer_layers"]) is not int
        or manifest["total_transformer_layers"] <= 0
        or not isinstance(manifest["calibration_axes"], list)
        or len(manifest["calibration_axes"]) < 5
    ):
        raise RuntimeError("distributed compiler manifest contract failed")
    base_model_path = Path(manifest["base_model_path"]).expanduser().resolve()
    if not base_model_path.is_file():
        raise RuntimeError("distributed compiler base model is unavailable")

    axis_fields = {
        "capability_id",
        "axis_id",
        "adapter_model_path",
        "adapter_config_path",
        "optimizer_steps",
        "trainable_parameter_count",
        "direct_execution_verified",
        "base_validation_accuracy",
        "solution_validation_accuracy",
    }
    bindings: list[dict[str, Any]] = []
    import_receipts: list[dict[str, Any]] = []
    import_references: list[dict[str, str]] = []
    solution_evidence: list[dict[str, str]] = []
    capability_ids: list[str] = []
    axis_ids: list[str] = []
    for index, axis in enumerate(manifest["calibration_axes"]):
        if not isinstance(axis, dict) or set(axis) != axis_fields:
            raise RuntimeError(f"distributed axis {index} fields are not exact")
        if (
            not isinstance(axis["capability_id"], str)
            or not axis["capability_id"]
            or not isinstance(axis["axis_id"], str)
            or not axis["axis_id"]
            or type(axis["optimizer_steps"]) is not int
            or axis["optimizer_steps"] <= 0
            or type(axis["trainable_parameter_count"]) is not int
            or axis["trainable_parameter_count"] <= 0
            or axis["direct_execution_verified"] is not True
            or not all(
                isinstance(axis[field], (int, float))
                and math.isfinite(float(axis[field]))
                and 0.0 <= float(axis[field]) <= 1.0
                for field in ("base_validation_accuracy", "solution_validation_accuracy")
            )
        ):
            raise RuntimeError(f"distributed axis {index} training evidence invalid")
        if (
            axis["capability_id"] in capability_ids
            or axis["axis_id"] in axis_ids
        ):
            raise RuntimeError("distributed axis identity is duplicated")
        capability_ids.append(axis["capability_id"])
        axis_ids.append(axis["axis_id"])
        adapter_model = Path(axis["adapter_model_path"]).expanduser().resolve()
        adapter_config = Path(axis["adapter_config_path"]).expanduser().resolve()
        if not adapter_model.is_file() or not adapter_config.is_file():
            raise RuntimeError(f"distributed axis {index} adapter is unavailable")
        import_input = {
            "schema": "cerebro.tidex.lora_adapter_axis_input/v1",
            "base_model_path": str(base_model_path),
            "adapter_model_path": str(adapter_model),
            "adapter_config_path": str(adapter_config),
        }
        import_path = root / "compiler-inputs" / f"axis-{index:03d}.json"
        write_new(import_path, canonical(import_input))
        receipt = tidex(binary, root, "import-lora-axis", str(import_path))
        if (
            receipt["schema"] != "cerebro.tidex.lora_adapter_axis_receipt/v1"
            or receipt["learned_parameter_count"] != axis["trainable_parameter_count"]
            or receipt["source_training_semantics_attested"] is not False
            or receipt["authorizes_target_update_free_claim"] is not False
            or receipt["authorizes_promotion"] is not False
        ):
            raise RuntimeError(f"distributed axis {index} import contract failed")
        receipt_reference = put(root, receipt)
        coordinates = [0.0] * len(manifest["calibration_axes"])
        coordinates[index] = 1.0
        solution = put(
            root,
            {
                "schema": "cerebro.tidex.cross_model_receiver_solution_evidence/v1",
                "capability_id": axis["capability_id"],
                "receiver_model_sha256": receipt["base_model_sha256"],
                "receiver_coordinates": coordinates,
                "receiver_coordinates_sha256": receiver_coordinates_sha256(coordinates),
                "target_capability": False,
                "lora_used": True,
                "backpropagation_used": True,
                "direct_execution_verified": True,
                "receiver_backend_frozen_before_target_observation": True,
                "optimizer_steps": axis["optimizer_steps"],
                "trainable_parameter_count": axis["trainable_parameter_count"],
                "axis_delta_sha256": receipt["dense_delta"]["sha256"],
                "base_validation_accuracy": float(axis["base_validation_accuracy"]),
                "solution_validation_accuracy": float(
                    axis["solution_validation_accuracy"]
                ),
                "adapter_model_sha256": receipt["adapter_model_sha256"],
                "adapter_config_sha256": receipt["adapter_config_sha256"],
            },
        )
        bindings.append(
            {
                "capability_id": axis["capability_id"],
                "axis_id": axis["axis_id"],
                "import_receipt": receipt_reference,
            }
        )
        import_receipts.append(receipt)
        import_references.append(receipt_reference)
        solution_evidence.append(solution)

    base_sha = import_receipts[0]["base_model_sha256"]
    if any(receipt["base_model_sha256"] != base_sha for receipt in import_receipts):
        raise RuntimeError("distributed axes bind different receiver bases")
    construction_evidence = put(
        root,
        {
            "schema": "cerebro.tidex.distributed_lora_basis_construction/v1",
            "receiver_model_sha256": base_sha,
            "calibration_capability_ids": capability_ids,
            "confirmation_target_capability_ids_used": [],
            "axis_delta_sha256": [
                receipt["dense_delta"]["sha256"] for receipt in import_receipts
            ],
            "receiver_backend_frozen_before_target_observation": True,
            "target_receiver_solution_used": False,
            "target_receiver_execution_performed": False,
        },
    )
    assembly = {
        "schema": "cerebro.tidex.distributed_lora_basis_input/v1",
        "axes": bindings,
        "construction_evidence": construction_evidence,
        "total_model_parameter_count": manifest["total_model_parameter_count"],
        "total_transformer_layers": manifest["total_transformer_layers"],
    }
    assembly_path = root / "compiler-inputs" / "assemble-basis.json"
    write_new(assembly_path, canonical(assembly))
    basis = tidex(binary, root, "assemble-lora-basis", str(assembly_path))
    if (
        basis["schema"] != DISTRIBUTED_BASIS_SCHEMA
        or basis["realization"]["scope"] != "distributed_transformer"
        or basis["realization"]["axis_construction_method"] != "calibration_lora"
        or basis["realization"]["target_capability_used_in_basis"] is not False
        or basis["realization"]["target_receiver_execution_used_in_basis"] is not False
    ):
        raise RuntimeError("Rust returned a non-distributed V69 basis")
    basis_reference = put(root, basis)
    result = {
        "schema": DISTRIBUTED_COMPILER_RECEIPT_SCHEMA,
        "complete": True,
        "candidate_only": True,
        "basis": basis_reference,
        "basis_record": basis,
        "construction_evidence": construction_evidence,
        "axis_import_receipts": import_references,
        "receiver_solution_evidence": solution_evidence,
        "calibration_capability_ids": capability_ids,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "target_capability_observed": False,
        "target_receiver_execution_performed": False,
        "target_optimizer_steps": 0,
        "requires_capability_ir_for_target_compilation": True,
        "authorizes_capability_claim": False,
        "authorizes_promotion": False,
    }
    write_new(root / "distributed-compiler-receipt.json", canonical(result))
    return result


def main_run(
    binary: Path, root: Path, threads: int, calibration_only: bool = False
) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError("V69 root already exists; experiment is single-shot")
    if not root.is_absolute() or PROJECT_ROOT in root.parents or root == PROJECT_ROOT:
        raise RuntimeError("V69 evidence root must be outside checkout")
    root.mkdir(parents=True, mode=0o700)
    os.umask(0o077)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)

    source_sha = sha_file(Path(__file__).resolve())
    binary_sha = sha_file(binary)
    donor_weight_sha = sha_file(DONOR_DIR / "model.safetensors")
    receiver_weight_sha = sha_file(RECEIVER_DIR / "model.safetensors")
    donor_files = file_sha_map(DONOR_DIR, ("config.json", "tokenizer.json"))
    receiver_files = file_sha_map(RECEIVER_DIR, MODEL_FILES)
    reveal = evaluation_reveal()
    reveal_sha = json_sha(reveal)
    reproducibility = freeze_reproducibility_bundle(
        root / "reproducibility", binary, collector=Path(__file__)
    )

    method_precommit = {
        "schema": "cerebro.tidex.v69_method_precommit/v1",
        "experiment_schema": SCHEMA,
        "protocol_revision": "V69-R4",
        "functional_ir_arithmetic": PROJECTION_ARITHMETIC,
        "compiler_method_grid": COMPILER_METHODS,
        "all_numerical_calibration_gates_enforced": True,
        "receiver_coordinate_geometry": "orthonormal QR basis; Euclidean coordinate distances and angles represent actual dense parameter distances and angles",
        "prior_r2_calibration_receipt_sha256": "c4ad48008fd2f5aebccf6b9e433c4783c178a13eeca4581eae8c83323594ebb3",
        "prior_ridge_selection_calibration_retry_sha256": "e0fb43c71015878bf6af5060d6174d18a4811285b7d256ea01694dd187458863",
        "confirmation_target_data_used_for_revision": False,
        "calibration_validation": {
            "pca_fit": "refit donor and receiver PCA inside every capability holdout",
            "compiler": "same Rust numerical kernel, protection and trust-region gates as weight binding",
            "method_selection": "calibration-only; selection LOO is not an unbiased estimate of the selected method",
            "blocked_fold": "no deliverable; scored zero and reported explicitly",
        },
        "primary_inference": "exact sign-flip randomization over capability-level paired accuracy differences; six groups, conditional on this fixed calibration",
        "baseline_protocol": ["mean_calibration_no_target_signature", "nearest_donor_signature_calibration_update", "cyclically_permuted_calibration_correspondence"],
        "preservation_fit_prompts": GENERIC_CONTROLS,
        "preservation_evaluation_prompts": GENERIC_VALIDATION_CONTROLS,
        "seed": SEED,
        "claim": "target-update-free cross-model capability compilation in a bounded threshold family",
        "donor": {
            "model_id": "Qwen/Qwen2.5-Coder-1.5B",
            "revision": DONOR_REVISION,
            "architecture": "Qwen2ForCausalLM",
            "weight_sha256": donor_weight_sha,
            "config_sha256": donor_files["config.json"],
            "tokenizer_sha256": donor_files["tokenizer.json"],
        },
        "receiver": {
            "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "revision": RECEIVER_REVISION,
            "architecture": "LlamaForCausalLM",
            "weight_sha256": receiver_weight_sha,
            "sidecars_sha256": receiver_files,
        },
        "calibration_capability_ids": [cap_id(t) for t in CALIBRATION_THRESHOLDS],
        "confirmation_target_capability_ids": [cap_id(t) for t in CONFIRMATION_THRESHOLDS],
        "confirmation_thresholds": CONFIRMATION_THRESHOLDS,
        "confirmation_targets_seen_as_receiver_solutions_during_method_development": False,
        "method_revision": {
            "prior_failed_method_precommit_sha256": PRIOR_FAILED_METHOD_PRECOMMIT_SHA256,
            "prior_failed_calibration_receipt_sha256": PRIOR_FAILED_CALIBRATION_RECEIPT_SHA256,
            "prior_protocol_failure_precommit_sha256": PRIOR_PROTOCOL_FAILURE_PRECOMMIT_SHA256,
            "prior_protocol_failure_calibration_sha256": PRIOR_PROTOCOL_FAILURE_CALIBRATION_SHA256,
            "prior_protocol_failure_receipt_sha256": PRIOR_PROTOCOL_FAILURE_RECEIPT_SHA256,
            "prior_single_authority_failure_precommit_sha256": PRIOR_SINGLE_AUTHORITY_FAILURE_PRECOMMIT_SHA256,
            "prior_typed_evidence_failure_precommit_sha256": PRIOR_TYPED_EVIDENCE_FAILURE_PRECOMMIT_SHA256,
            "retired_confirmation_thresholds": RETIRED_CONFIRMATION_THRESHOLDS,
            "revision_reason": "exact decimal decoding, one projection authority, fold-local PCA, the Rust compiler used during calibration, capability-group inference, held-out preservation probes and precommitted calibration-only baselines; earlier targets are retired",
            "receiver_rank_selection": {
                "minimum_rank": MIN_RECEIVER_PC_DIM,
                "maximum_rank": MAX_RECEIVER_PC_DIM,
                "rule": "smallest rank whose calibration-only PCA retained energy meets the precommitted threshold",
                "minimum_retained_energy": CRITERIA["minimum_receiver_basis_explained_energy"],
            },
            "proposal_method_selection": {
                "candidates": COMPILER_METHODS,
                "rule": "highest deliverable calibration accuracy with all strict Rust gates applied; relational requires every fold allowed; ties use the precommitted grid order",
                "target_data_used": False,
            },
        },
        "functional_dimension": FUNCTIONAL_DIM,
        "receiver_pc_dimension": "calibration_selected",
        "donor_probes": DONOR_PROBES,
        "donor_demo_offsets": DONOR_DEMO_OFFSETS,
        "receiver_train_offsets": RECEIVER_TRAIN_OFFSETS,
        "receiver_validation_offsets": RECEIVER_VALIDATION_OFFSETS,
        "calibration_solution_ridge": CALIBRATION_SOLUTION_RIDGE,
        "compiler_ridge": COMPILER_RIDGE,
        "desired_margin": DESIRED_MARGIN,
        "proposal_method": "calibration_selected",
        "risk_budget_rule": "1.25 * maximum calibration coordinate quadratic cost",
        "functional_support_rule": {
            "method": "ridge_leverage_against_calibration_functional_signatures",
            "ridge": "selected compiler method ridge, frozen after calibration",
            "envelope": "maximum leave-one-capability-out leverage among calibration capabilities",
            "acceptance": "target leverage <= calibration envelope",
            "receiver_target_solution_used": False,
        },
        "behavioral_calibration_policy": {
            "schema": "cerebro.tidex.receiver_behavioral_calibration_policy/v1",
            "minimum_mean_accuracy": CRITERIA["minimum_calibration_compiler_loo_mean_accuracy"],
            "minimum_mean_gain": CRITERIA["minimum_calibration_compiler_loo_gain"],
            "minimum_non_degrading_count": CRITERIA["minimum_calibration_non_degrading_count"],
            "authority": "recomputed by Rust from calibration-only per-capability base and compiled LOO accuracies",
        },
        "target_blind_protocol_self_test": {
            "enabled": True,
            "source": "one existing calibration functional signature under a fresh synthetic capability identity",
            "runs_receiver_compiler": True,
            "materializes_model": False,
            "executes_receiver_model": False,
            "confirmation_target_data_used": False,
        },
        "confirmation_reveal_sha256": reveal_sha,
        "criteria": CRITERIA,
        "target_forbidden_receiver_authorities": [
            "LoRA",
            "task_vector",
            "prior_delta_W",
            "receiver_SFT",
            "receiver_solution",
            "receiver_calibration_observation",
            "receiver_basis_construction",
            "receiver_execution_before_all_candidates_are_sealed",
        ],
        "reproducibility_bundle": reproducibility,
    }
    write_new(root / "method-precommit.json", canonical(method_precommit))

    verify_frozen_inputs(reproducibility, binary)

    # DONOR PHASE 1: calibration signatures only.  Confirmation target
    # signatures are deliberately deferred until the receiver backend has
    # passed calibration and frozen its rank/proposal method.
    donor_tokenizer = AutoTokenizer.from_pretrained(
        DONOR_DIR, local_files_only=True, trust_remote_code=False
    )
    donor_model = AutoModelForCausalLM.from_pretrained(
        DONOR_DIR,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    donor_model.eval()
    donor_model.requires_grad_(False)
    donor_model.config.use_cache = False
    if type(donor_model).__name__ != "Qwen2ForCausalLM":
        raise RuntimeError("unexpected V69 donor architecture")
    donor_true = one_token(donor_tokenizer, " true")
    donor_false = one_token(donor_tokenizer, " false")
    raw_calibration: dict[int, np.ndarray] = {}
    raw_inverse_calibration: dict[int, np.ndarray] = {}
    donor_prompt_hashes: dict[str, list[str]] = {}
    donor_prompts: dict[str, list[str]] = {}
    # Phase 1 is target-blind even on the donor side.  Only calibration
    # capabilities are observed until the receiver backend has independently
    # passed all of its calibration gates and frozen its rank/method.
    for threshold in CALIBRATION_THRESHOLDS:
        raw, prompts = donor_signature(
            donor_model, donor_tokenizer, threshold, False, donor_true, donor_false
        )
        donor_prompt_hashes[cap_id(threshold)] = [
            hashlib.sha256(p.encode()).hexdigest() for p in prompts
        ]
        donor_prompts[cap_id(threshold)] = prompts
        raw_calibration[threshold] = raw
        raw_inverse_calibration[threshold], _ = donor_signature(
            donor_model, donor_tokenizer, threshold, True, donor_true, donor_false
        )
        print(f"donor calibration signature {threshold}", flush=True)

    donor_calibration_matrix = np.stack([raw_calibration[t] for t in CALIBRATION_THRESHOLDS])
    functional_mean, functional_components, _batch_projection, functional_energy = pca_projection(
        donor_calibration_matrix, FUNCTIONAL_DIM
    )
    projection_evidence = put(
        root,
        {
            "schema": "cerebro.tidex.v69_donor_functional_projection/v2",
            "projection_arithmetic": PROJECTION_ARITHMETIC,
            "raw_probe_sha256": donor_raw_probe_hashes(),
            "donor_model_sha256": donor_weight_sha,
            "calibration_capability_ids": [cap_id(t) for t in CALIBRATION_THRESHOLDS],
            "confirmation_target_capability_ids_used": [],
            "target_values_used": False,
            "raw_dimension": len(DONOR_PROBES),
            "projected_dimension": FUNCTIONAL_DIM,
            "mean": functional_mean.tolist(),
            "components": functional_components.tolist(),
            "retained_energy": functional_energy,
        },
    )

    def project_functional(raw: np.ndarray) -> np.ndarray:
        return canonical_project(raw, functional_mean, functional_components)

    # Single numerical authority for every calibration signature.  Do not keep
    # a second batch-matmul materialization: BLAS evaluation order can differ in
    # the final bits even when the mathematical projection is identical.
    calibration_functional = np.stack(
        [project_functional(raw_calibration[t]) for t in CALIBRATION_THRESHOLDS]
    )

    functional_evidence: dict[str, dict[str, str]] = {}
    functional_values: dict[str, np.ndarray] = {}
    for index, threshold in enumerate(CALIBRATION_THRESHOLDS):
        identifier = cap_id(threshold)
        projected = calibration_functional[index]
        functional_values[identifier] = projected
        functional_evidence[identifier] = put(
            root,
            {
                "schema": "cerebro.tidex.cross_model_functional_evidence/v2",
                "capability_id": identifier,
                "donor_model_sha256": donor_weight_sha,
                "projection_evidence_sha256": projection_evidence["sha256"],
                "prompt_sha256": donor_prompt_hashes[identifier],
                "prompts": donor_prompts[identifier],
                "raw_probe_sha256": donor_raw_probe_hashes(),
                "raw_logit_margins": raw_calibration[threshold].tolist(),
                "projected_signature": projected.tolist(),
                "projected_signature_sha256": functional_signature_sha256(projected),
                "receiver_data_used": False,
            },
        )
    del donor_model, donor_tokenizer
    gc.collect()

    # RECEIVER CALIBRATION: only calibration capabilities enter this section.
    receiver_model, receiver_tokenizer = load_receiver(RECEIVER_DIR, threads)
    receiver_true = one_token(receiver_tokenizer, " true")
    receiver_false = one_token(receiver_tokenizer, " false")
    original_norm = receiver_model.model.norm.weight.detach().clone()
    receiver_deltas = []
    solution_evidence_raw: dict[int, dict[str, Any]] = {}
    calibration_base_accuracies = []
    calibration_solution_accuracies = []
    validation_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for threshold in CALIBRATION_THRESHOLDS:
        train_values = numbers(threshold, RECEIVER_TRAIN_OFFSETS)
        validation_values = numbers(threshold, RECEIVER_VALIDATION_OFFSETS)
        train_features, train_margins = receiver_features(
            receiver_model, receiver_tokenizer, train_values, receiver_true, receiver_false
        )
        validation_features, validation_margins = receiver_features(
            receiver_model, receiver_tokenizer, validation_values, receiver_true, receiver_false
        )
        train_labels = np.array([-1.0] * 4 + [1.0] * 4)
        validation_labels = np.array([-1.0] * 4 + [1.0] * 4)
        rhs = DESIRED_MARGIN * train_labels - train_margins
        delta = train_features.T @ np.linalg.solve(
            train_features @ train_features.T
            + CALIBRATION_SOLUTION_RIDGE * np.eye(len(train_values)),
            rhs,
        )
        predicted_validation = validation_margins + validation_features @ delta
        base_accuracy = float((np.sign(validation_margins) == validation_labels).mean())
        predicted_accuracy = float((np.sign(predicted_validation) == validation_labels).mean())
        # Verify the analytic final-norm update by actual receiver execution.
        with torch.no_grad():
            receiver_model.model.norm.weight.copy_(
                original_norm + torch.tensor(delta, dtype=original_norm.dtype)
            )
        measured_validation = receiver_prompt_margins(
            receiver_model,
            receiver_tokenizer,
            [f"Input: {value}\nOutput:" for value in validation_values],
            receiver_true,
            receiver_false,
        )
        with torch.no_grad():
            receiver_model.model.norm.weight.copy_(original_norm)
        if not np.allclose(predicted_validation, measured_validation, atol=2e-5, rtol=2e-6):
            raise RuntimeError("receiver calibration solution failed direct execution verification")
        receiver_deltas.append(delta)
        calibration_base_accuracies.append(base_accuracy)
        calibration_solution_accuracies.append(predicted_accuracy)
        validation_cache[threshold] = (
            validation_features,
            validation_margins,
            validation_labels,
        )
        solution_evidence_raw[threshold] = {
            "schema": "cerebro.tidex.cross_model_receiver_solution_evidence/v1",
            "capability_id": cap_id(threshold),
            "receiver_model_sha256": receiver_weight_sha,
            "train_values": train_values,
            "validation_values": validation_values,
            "desired_margin": DESIRED_MARGIN,
            "ridge": CALIBRATION_SOLUTION_RIDGE,
            "delta_values": delta.tolist(),
            "base_validation_accuracy": base_accuracy,
            "solution_validation_accuracy": predicted_accuracy,
            "direct_execution_verified": True,
            "target_capability": False,
            "lora_used": False,
            "backpropagation_used": False,
        }
        print(
            f"receiver calibration {threshold}: {base_accuracy:.3f}->{predicted_accuracy:.3f}",
            flush=True,
        )
    if not torch.equal(receiver_model.model.norm.weight, original_norm):
        raise RuntimeError("receiver calibration mutation was not restored")

    # Only the fitting control split can affect the risk geometry.
    control_features_matrix, _control_margins = receiver_prompt_features(
        receiver_model, receiver_tokenizer, GENERIC_CONTROLS, receiver_true, receiver_false
    )

    receiver_delta_matrix = np.stack(receiver_deltas)
    selected_receiver_pc_dim, receiver_energy_curve = select_receiver_pca_dimension(
        receiver_delta_matrix, CRITERIA["minimum_receiver_basis_explained_energy"]
    )
    receiver_mean, receiver_components, receiver_scores, receiver_energy = pca_projection(
        receiver_delta_matrix, selected_receiver_pc_dim
    )
    basis_axes = np.concatenate([receiver_mean[None, :], receiver_components], axis=0)
    receiver_coordinates = np.concatenate(
        [np.ones((len(CALIBRATION_THRESHOLDS), 1)), receiver_scores], axis=1
    )
    basis_axes, receiver_coordinates = orthonormal_receiver_basis(basis_axes, receiver_coordinates)
    reconstructed = receiver_coordinates @ basis_axes
    receiver_basis_rms = float(np.sqrt(np.mean(np.square(reconstructed - receiver_delta_matrix))))

    solution_evidence: dict[int, dict[str, str]] = {}
    for index, threshold in enumerate(CALIBRATION_THRESHOLDS):
        solution_evidence_raw[threshold]["receiver_coordinates"] = receiver_coordinates[index].tolist()
        solution_evidence_raw[threshold]["receiver_coordinates_sha256"] = receiver_coordinates_sha256(
            receiver_coordinates[index]
        )
        solution_evidence_raw[threshold]["receiver_basis_rms"] = receiver_basis_rms
        solution_evidence[threshold] = put(root, solution_evidence_raw[threshold])

    # Each fold fits its donor PCA, receiver PCA/rank and risk geometry using
    # only the other capabilities. Rust is the sole numerical compiler.
    calibration_loo_base: list[float] = []
    method_loo: dict[str, list[float]] = {method_id: [] for method_id in COMPILER_METHODS}
    method_allowed: dict[str, list[bool]] = {name: [] for name in method_loo}
    fold_records = []
    for holdout, threshold in enumerate(CALIBRATION_THRESHOLDS):
        fitted = calibration_fold(donor_calibration_matrix, receiver_delta_matrix, holdout)
        features, margins, labels = validation_cache[threshold]
        calibration_loo_base.append(float((np.sign(margins) == labels).mean()))
        wrong = canonical_project(raw_inverse_calibration[threshold], fitted["functional_mean"], fitted["functional_components"])
        record = {
            "heldout_capability_id": cap_id(threshold),
            "training_capability_ids": [cap_id(CALIBRATION_THRESHOLDS[i]) for i in fitted["training_indices"]],
            "receiver_rank": fitted["receiver_rank"], "methods": {},
        }
        for method in method_loo:
            report = rust_signature_probe(binary, root, f"loo-{threshold:03d}-{method}",
                                          fitted, fitted["requested"], wrong,
                                          control_features_matrix,
                                          COMPILER_METHODS[method]["proposal_method"],
                                          COMPILER_METHODS[method]["ridge"])
            method_allowed[method].append(bool(report["allowed"]))
            # This is a calibration evaluation. All receiver updates in this
            # section concern known calibration tasks, never confirmation tasks.
            delta = (np.asarray(report["target_delta"]) @ fitted["axes"].astype(np.float32).astype(np.float64)).astype(np.float32)
            with torch.no_grad():
                receiver_model.model.norm.weight.copy_(original_norm + torch.from_numpy(delta))
            try:
                measured = receiver_prompt_margins(receiver_model, receiver_tokenizer,
                    [f"Input: {value}\nOutput:" for value in numbers(threshold, RECEIVER_VALIDATION_OFFSETS)],
                    receiver_true, receiver_false)
            finally:
                with torch.no_grad():
                    receiver_model.model.norm.weight.copy_(original_norm)
            predicted = margins + features @ delta.astype(np.float64)
            if not np.allclose(measured, predicted, atol=2e-5, rtol=2e-6):
                raise RuntimeError("fold compiler prediction disagrees with actual receiver execution")
            measured_accuracy = float((np.sign(measured) == labels).mean())
            accuracy = measured_accuracy if report["allowed"] else 0.0
            method_loo[method].append(accuracy)
            record["methods"][method] = {"allowed": report["allowed"],
                "measured_accuracy": measured_accuracy, "deliverable_accuracy": accuracy,
                "receiver_execution_verified": True, "compiler": report}
        fold_records.append(record)
        print(f"calibration fold {threshold}: " + json.dumps({m: method_loo[m][-1] for m in method_loo}), flush=True)
    relational_holdout_support = method_allowed["relational_anchors"]

    calibration_solution_mean = float(np.mean(calibration_solution_accuracies))
    base_loo_mean = float(np.mean(calibration_loo_base))
    method_summary = {
        name: {
            "accuracies": accuracies,
            "mean_accuracy": float(np.mean(accuracies)),
            "mean_gain": float(np.mean(accuracies)) - base_loo_mean,
            "eligible": name != "relational_anchors" or all(relational_holdout_support),
        }
        for name, accuracies in method_loo.items()
    }
    eligible_methods = [name for name, report in method_summary.items() if report["eligible"]]
    if not eligible_methods:
        raise RuntimeError("V69 calibration proposal portfolio has no eligible method")
    tie_priority = {name: index for index, name in enumerate(COMPILER_METHODS)}
    selected_method_id = max(
        eligible_methods,
        key=lambda name: (
            method_summary[name]["mean_accuracy"],
            method_summary[name]["mean_gain"],
            tie_priority[name],
        ),
    )
    selected_proposal_method = COMPILER_METHODS[selected_method_id]["proposal_method"]
    selected_ridge = COMPILER_METHODS[selected_method_id]["ridge"]
    calibration_loo_accuracies = method_summary[selected_method_id]["accuracies"]
    calibration_loo_mean = method_summary[selected_method_id]["mean_accuracy"]
    calibration_loo_gain = method_summary[selected_method_id]["mean_gain"]
    calibration_non_degrading_count = sum(
        compiled >= base
        for compiled, base in zip(
            calibration_loo_accuracies, calibration_loo_base, strict=True
        )
    )

    coordinate_jacobian, risk_metric, risk_budget = coordinate_risk(
        control_features_matrix, basis_axes, receiver_coordinates
    )

    calibration_pass = (
        calibration_solution_mean >= CRITERIA["minimum_calibration_solution_mean_accuracy"]
        and calibration_loo_mean >= CRITERIA["minimum_calibration_compiler_loo_mean_accuracy"]
        and calibration_loo_gain >= CRITERIA["minimum_calibration_compiler_loo_gain"]
        and calibration_non_degrading_count
        >= CRITERIA["minimum_calibration_non_degrading_count"]
        and receiver_energy >= CRITERIA["minimum_receiver_basis_explained_energy"]
    )
    calibration_receipt = {
        "schema": "cerebro.tidex.v69_receiver_calibration_receipt/v1",
        "pass": calibration_pass,
        "calibration_capability_ids": [cap_id(t) for t in CALIBRATION_THRESHOLDS],
        "confirmation_target_capability_ids_used": [],
        "solution_mean_accuracy": calibration_solution_mean,
        "compiler_loo_mean_accuracy": calibration_loo_mean,
        "compiler_loo_mean_gain": calibration_loo_gain,
        "compiler_loo_base_accuracies": calibration_loo_base,
        "compiler_loo_accuracies": calibration_loo_accuracies,
        "compiler_loo_measured_mean_accuracy": float(np.mean([fold["methods"][selected_method_id]["measured_accuracy"] for fold in fold_records])),
        "compiler_loo_delivery_coverage": float(np.mean(method_allowed[selected_method_id])),
        "compiler_loo_score_definition": "mean measured accuracy with undeliverable folds scored zero; distinct from raw model accuracy",
        "receiver_basis_orthonormality_max_error": float(np.max(np.abs(basis_axes @ basis_axes.T - np.eye(len(basis_axes))))),
        "compiler_loo_non_degrading_count": calibration_non_degrading_count,
        "loo_interpretation": "method-selection scores; PCA and risk are fitted without the heldout capability and predictions use the real Rust kernel",
        "fold_records": fold_records,
        "method_allowed_folds": method_allowed,
        "proposal_method": selected_proposal_method,
        "proposal_method_portfolio": method_summary,
        "selected_method_id": selected_method_id,
        "selected_ridge": selected_ridge,
        "numerical_validation_profile": "parametric_cross_validation",
        "relational_holdout_support": relational_holdout_support,
        "selected_receiver_pc_dimension": selected_receiver_pc_dim,
        "receiver_pca_energy_curve": receiver_energy_curve,
        "receiver_basis_explained_energy": receiver_energy,
        "receiver_basis_rms": receiver_basis_rms,
        "risk_budget": risk_budget,
        "criteria": CRITERIA,
    }
    write_new(root / "calibration-receipt.json", canonical(calibration_receipt))
    if not calibration_pass:
        raise RuntimeError("V69 receiver backend calibration failed before target compilation")

    behavioral_calibration_policy = {
        "schema": "cerebro.tidex.receiver_behavioral_calibration_policy/v1",
        "minimum_mean_accuracy": CRITERIA["minimum_calibration_compiler_loo_mean_accuracy"],
        "minimum_mean_gain": CRITERIA["minimum_calibration_compiler_loo_gain"],
        "minimum_non_degrading_count": CRITERIA["minimum_calibration_non_degrading_count"],
    }
    behavioral_calibration_evidence = put(
        root,
        {
            "schema": "cerebro.tidex.receiver_behavioral_calibration_evidence/v1",
            "receiver_model_sha256": receiver_weight_sha,
            "calibration_capability_ids": [cap_id(t) for t in CALIBRATION_THRESHOLDS],
            "proposal_method": selected_proposal_method,
            "base_accuracies": calibration_loo_base,
            "compiled_loo_accuracies": calibration_loo_accuracies,
            "confirmation_target_capability_ids_used": [],
            "target_receiver_execution_performed": False,
            "receiver_target_solution_used": False,
            "selected_receiver_pc_dimension": selected_receiver_pc_dim,
            "calibration_receipt_sha256": sha_file(root / "calibration-receipt.json"),
            "collector_sha256": source_sha,
        },
    )
    # Persist receiver basis axes through the Rust artifact authority.
    axis_records = []
    for index, axis in enumerate(basis_axes):
        values_ref = put(root, axis.astype(np.float32).tolist())
        values_path = reference_file(root, f"v69-axis-values-{index}", values_ref)
        delta = tidex(binary, root, "import-axis", str(values_path))
        axis_records.append({"axis_id": f"threshold-axis-{index:02d}", "delta": delta})
    basis_construction_evidence = put(
        root,
        {
            "schema": "cerebro.tidex.v69_receiver_basis_construction/v1",
            "receiver_model_sha256": receiver_weight_sha,
            "calibration_capability_ids": [cap_id(t) for t in CALIBRATION_THRESHOLDS],
            "confirmation_target_capability_ids_used": [],
            "receiver_solution_evidence_sha256": [
                solution_evidence[t]["sha256"] for t in CALIBRATION_THRESHOLDS
            ],
            "mean_axis": receiver_mean.tolist(),
            "principal_axes": receiver_components.tolist(),
            "selected_receiver_pc_dimension": selected_receiver_pc_dim,
            "retained_energy": receiver_energy,
            "proposal_method_selected_from_calibration": selected_proposal_method,
            "target_receiver_solution_used": False,
        },
    )
    basis = put(
        root,
        {
            "schema": BASIS_SCHEMA,
            "base_model_sha256": receiver_weight_sha,
            "layout": {
                "schema": "cerebro.tidex.parameter_block_layout/v1",
                "blocks": [
                    {
                        "name": "model.norm.weight",
                        "shape": [int(original_norm.numel())],
                        "offset": 0,
                        "count": int(original_norm.numel()),
                    }
                ],
                "total_parameter_count": int(original_norm.numel()),
            },
            "axes": axis_records,
            "construction_capability_ids": [cap_id(t) for t in CALIBRATION_THRESHOLDS],
            "construction_evidence": basis_construction_evidence,
        },
    )
    protocol = put(
        root,
        {
            "schema": PROTOCOL_SCHEMA,
            "measure": "donor_next_token_logit_margin_projection",
            "receiver_base_model_sha256": receiver_weight_sha,
            "donor_model_sha256": donor_weight_sha,
            "donor_model_config_sha256": donor_files["config.json"],
            "donor_tokenizer_sha256": donor_files["tokenizer.json"],
            "collector_sha256": source_sha,
            "raw_probe_sha256": donor_raw_probe_hashes(),
            "coordinate_ids": [f"donor-functional-pc-{index:02d}" for index in range(FUNCTIONAL_DIM)],
            "projection_evidence": projection_evidence,
            "max_input_tokens": 512,
        },
    )

    observations = []
    for index, threshold in enumerate(CALIBRATION_THRESHOLDS):
        observations.append(
            put(
                root,
                {
                    "schema": OBSERVATION_SCHEMA,
                    "observation_id": f"threshold-calibration-{threshold:03d}",
                    "capability_id": cap_id(threshold),
                    "basis_sha256": basis["sha256"],
                    "protocol_sha256": protocol["sha256"],
                    "receiver_coordinates": receiver_coordinates[index].tolist(),
                    "values": calibration_functional[index].tolist(),
                    "functional_evidence": functional_evidence[cap_id(threshold)],
                    "receiver_solution_evidence": solution_evidence[threshold],
                },
            )
        )

    safety_evidence = put(
        root,
        {
            "schema": "cerebro.tidex.v69_receiver_coordinate_risk/v1",
            "receiver_model_sha256": receiver_weight_sha,
            "generic_control_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in GENERIC_CONTROLS],
            "coordinate_jacobian": coordinate_jacobian.tolist(),
            "risk_metric": risk_metric.tolist(),
            "risk_budget": risk_budget,
            "confirmation_target_data_used": False,
            "global_language_preservation_established": False,
        },
    )
    policy = compiler_policy(risk_budget, selected_ridge)

    # Target-blind protocol/binding self-test.  It reuses one already-known
    # calibration signature under a fresh synthetic identity and never
    # materializes or executes a model.  Confirmation targets remain unopened.
    selftest_threshold = CALIBRATION_THRESHOLDS[len(CALIBRATION_THRESHOLDS) // 2]
    selftest_values = calibration_functional[len(CALIBRATION_THRESHOLDS) // 2]
    selftest_wrong_values = calibration_functional[0]
    selftest_target_id = "protocol.selftest.functional:v1"
    selftest_wrong_id = "protocol.selftest.wrong:v1"
    selftest_target_evidence = put(
        root,
        {
            "schema": "cerebro.tidex.cross_model_functional_evidence/v2",
            "capability_id": selftest_target_id,
            "donor_model_sha256": donor_weight_sha,
            "projection_evidence_sha256": projection_evidence["sha256"],
            "raw_logit_margins": raw_calibration[selftest_threshold].tolist(),
            "raw_probe_sha256": donor_raw_probe_hashes(),
            "prompts": donor_prompts[cap_id(selftest_threshold)],
            "prompt_sha256": donor_prompt_hashes[cap_id(selftest_threshold)],
            "projected_signature": selftest_values.tolist(),
            "projected_signature_sha256": functional_signature_sha256(selftest_values),
            "receiver_data_used": False,
            "protocol_self_test": True,
            "source_calibration_capability_id": cap_id(
                CALIBRATION_THRESHOLDS[len(CALIBRATION_THRESHOLDS) // 2]
            ),
        },
    )
    selftest_wrong_evidence = put(
        root,
        {
            "schema": "cerebro.tidex.cross_model_functional_evidence/v2",
            "capability_id": selftest_wrong_id,
            "donor_model_sha256": donor_weight_sha,
            "projection_evidence_sha256": projection_evidence["sha256"],
            "raw_logit_margins": raw_calibration[CALIBRATION_THRESHOLDS[0]].tolist(),
            "raw_probe_sha256": donor_raw_probe_hashes(),
            "prompts": donor_prompts[cap_id(CALIBRATION_THRESHOLDS[0])],
            "prompt_sha256": donor_prompt_hashes[cap_id(CALIBRATION_THRESHOLDS[0])],
            "projected_signature": selftest_wrong_values.tolist(),
            "projected_signature_sha256": functional_signature_sha256(selftest_wrong_values),
            "receiver_data_used": False,
            "protocol_self_test": True,
            "source_calibration_capability_id": cap_id(CALIBRATION_THRESHOLDS[0]),
        },
    )
    selftest_target = put(
        root,
        {
            "schema": TARGET_SCHEMA,
            "capability_id": selftest_target_id,
            "protocol_sha256": protocol["sha256"],
            "values": selftest_values.tolist(),
            "functional_evidence": selftest_target_evidence,
        },
    )
    selftest_wrong = put(
        root,
        {
            "schema": TARGET_SCHEMA,
            "capability_id": selftest_wrong_id,
            "protocol_sha256": protocol["sha256"],
            "values": selftest_wrong_values.tolist(),
            "functional_evidence": selftest_wrong_evidence,
        },
    )
    selftest_request = put(
        root,
        {
            "schema": REQUEST_SCHEMA,
            "basis": basis,
            "protocol": protocol,
            "target": selftest_target,
            "observations": observations,
            "wrong_targets": [selftest_wrong],
            "safety": {
                "protected_cortex": {
                    "parameter_importance": [0.0] * len(basis_axes),
                    "directions": [],
                    "max_damage_ratio": 0.0,
                },
                "risk_metric": risk_metric.tolist(),
                "evidence": safety_evidence,
            },
            "policy": policy,
            "behavioral_calibration_evidence": behavioral_calibration_evidence,
            "behavioral_calibration_policy": behavioral_calibration_policy,
            "proposal_method": selected_proposal_method,
        },
    )
    selftest_request_path = reference_file(root, "v69-protocol-selftest-request", selftest_request)
    selftest_candidate_ref = tidex(binary, root, "compile", str(selftest_request_path))
    selftest_candidate_path = reference_file(
        root, "v69-protocol-selftest-candidate", selftest_candidate_ref
    )
    selftest_candidate = tidex(binary, root, "inspect", str(selftest_candidate_path))
    if selftest_candidate["target_capability_id"] != selftest_target_id:
        raise RuntimeError("V69 protocol self-test target identity changed")
    if "protocol_self_test_only" not in selftest_candidate["blockers"] or selftest_candidate.get("dense_delta") is not None:
        raise RuntimeError("V69 protocol self-test improperly authorized a materializable delta")
    # Other numerical blockers do not falsify this integrity/lineage self-test;
    # the independently computed calibration gates still govern target access.
    write_new(
        root / "protocol-self-test.json",
        canonical(
            {
                "schema": "cerebro.tidex.v69_protocol_self_test/v1",
                "pass": True,
                "target_capability_id": selftest_target_id,
                "candidate": selftest_candidate_ref,
                "confirmation_target_capability_ids_used": [],
                "confirmation_target_donor_signatures_observed": False,
                "receiver_execution_performed": False,
                "model_materialized": False,
            }
        ),
    )

    if calibration_only:
        result = {
            "schema": "cerebro.tidex.v69_calibration_only/v1",
            "complete": True,
            "pass": True,
            "method_precommit_sha256": sha_file(root / "method-precommit.json"),
            "calibration_receipt_sha256": sha_file(root / "calibration-receipt.json"),
            "behavioral_calibration_evidence": behavioral_calibration_evidence,
            "behavioral_calibration_policy": behavioral_calibration_policy,
            "protocol_self_test_sha256": sha_file(root / "protocol-self-test.json"),
            "calibration": calibration_receipt,
            "confirmation_target_capability_ids_used": [],
            "target_donor_signatures_observed": False,
            "target_receiver_execution_performed": False,
            "authorizes_capability_claim": False,
        }
        write_new(root / "calibration-only-receipt.json", canonical(result))
        del receiver_model, receiver_tokenizer, original_norm
        gc.collect()
        verify_frozen_inputs(reproducibility, binary)
        return result

    del receiver_model, receiver_tokenizer, original_norm
    gc.collect()

    verify_frozen_inputs(reproducibility, binary)

    # Only after the receiver backend has passed and frozen its rank/proposal
    # method do we observe the confirmation capabilities on the donor.  This
    # prevents target functional signatures from influencing receiver model
    # selection even indirectly.
    donor_tokenizer = AutoTokenizer.from_pretrained(
        DONOR_DIR, local_files_only=True, trust_remote_code=False
    )
    donor_model = AutoModelForCausalLM.from_pretrained(
        DONOR_DIR,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    donor_model.eval()
    donor_model.requires_grad_(False)
    donor_model.config.use_cache = False
    donor_true = one_token(donor_tokenizer, " true")
    donor_false = one_token(donor_tokenizer, " false")
    for threshold in CONFIRMATION_THRESHOLDS:
        raw, prompts = donor_signature(
            donor_model, donor_tokenizer, threshold, False, donor_true, donor_false
        )
        inverse, inverse_prompts = donor_signature(
            donor_model, donor_tokenizer, threshold, True, donor_true, donor_false
        )
        for identifier, values, prompt_rows in (
            (cap_id(threshold), raw, prompts),
            (inverse_cap_id(threshold), inverse, inverse_prompts),
        ):
            donor_prompt_hashes[identifier] = [
                hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompt_rows
            ]
            projected = project_functional(values)
            functional_values[identifier] = projected
            functional_evidence[identifier] = put(
                root,
                {
                    "schema": "cerebro.tidex.cross_model_functional_evidence/v2",
                    "capability_id": identifier,
                    "donor_model_sha256": donor_weight_sha,
                    "projection_evidence_sha256": projection_evidence["sha256"],
                    "prompt_sha256": donor_prompt_hashes[identifier],
                    "prompts": prompt_rows,
                    "raw_probe_sha256": donor_raw_probe_hashes(),
                    "raw_logit_margins": values.tolist(),
                    "projected_signature": projected.tolist(),
                    "projected_signature_sha256": functional_signature_sha256(projected),
                    "receiver_data_used": False,
                    "receiver_backend_frozen_before_observation": True,
                    "selected_receiver_pc_dimension": selected_receiver_pc_dim,
                    "selected_proposal_method": selected_proposal_method,
                },
            )
        print(f"donor confirmation signature {threshold}", flush=True)
    del donor_model, donor_tokenizer
    gc.collect()

    baseline_updates = prepare_calibration_baselines(
        binary, root, CONFIRMATION_THRESHOLDS, calibration_functional,
        receiver_coordinates, basis_axes, receiver_delta_matrix, functional_values,
        control_features_matrix, selected_proposal_method, selected_ridge,
    )
    baseline_commitment = root / "baselines" / "updates-precommit.json"
    write_new(baseline_commitment, canonical(baseline_updates))
    verify_frozen_inputs(reproducibility, binary)

    # Build all target donor-only records before any receiver target execution.
    target_refs: dict[int, dict[str, str]] = {}
    inverse_refs: dict[int, dict[str, str]] = {}
    for threshold in CONFIRMATION_THRESHOLDS:
        target_refs[threshold] = put(
            root,
            {
                "schema": TARGET_SCHEMA,
                "capability_id": cap_id(threshold),
                "protocol_sha256": protocol["sha256"],
                "values": functional_values[cap_id(threshold)].tolist(),
                "functional_evidence": functional_evidence[cap_id(threshold)],
            },
        )
        inverse_refs[threshold] = put(
            root,
            {
                "schema": TARGET_SCHEMA,
                "capability_id": inverse_cap_id(threshold),
                "protocol_sha256": protocol["sha256"],
                "values": functional_values[inverse_cap_id(threshold)].tolist(),
                "functional_evidence": functional_evidence[inverse_cap_id(threshold)],
            },
        )

    candidates: dict[int, dict[str, Any]] = {}
    model_dirs: dict[int, Path] = {}
    for threshold in CONFIRMATION_THRESHOLDS:
        request = put(
            root,
            {
                "schema": REQUEST_SCHEMA,
                "basis": basis,
                "protocol": protocol,
                "target": target_refs[threshold],
                "observations": observations,
                "wrong_targets": [inverse_refs[threshold]],
                "safety": {
                    "protected_cortex": {
                        "parameter_importance": [0.0] * len(basis_axes),
                        "directions": [],
                        "max_damage_ratio": 0.0,
                    },
                    "risk_metric": risk_metric.tolist(),
                    "evidence": safety_evidence,
                },
                "policy": policy,
                "behavioral_calibration_evidence": behavioral_calibration_evidence,
                "behavioral_calibration_policy": behavioral_calibration_policy,
                "proposal_method": selected_proposal_method,
            },
        )
        request_path = reference_file(root, f"v69-request-{threshold:03d}", request)
        candidate_ref = tidex(binary, root, "compile", str(request_path))
        candidate_path = reference_file(root, f"v69-candidate-{threshold:03d}", candidate_ref)
        candidate = tidex(binary, root, "inspect", str(candidate_path))
        if candidate["target_capability_id"] != cap_id(threshold):
            raise RuntimeError("V69 target identity changed")
        if candidate["calibration_capability_count"] != len(CALIBRATION_THRESHOLDS):
            raise RuntimeError("V69 calibration capability count mismatch")
        if candidate["blockers"]:
            raise RuntimeError(f"V69 target {threshold} blocked: {candidate['blockers']}")
        model_dir = root / "models" / f"target-{threshold:03d}"
        sidecar_dir = root / "model-sidecars" / f"target-{threshold:03d}"
        copy_model_sidecars(RECEIVER_DIR, sidecar_dir, receiver_files)
        # Weight output itself must remain under the TIDE-X private root.
        model_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        checkpoint = tidex(
            binary,
            root,
            "materialize",
            str(candidate_path),
            "--base-model",
            str(RECEIVER_DIR / "model.safetensors"),
            "--output",
            str(model_dir / "model.safetensors"),
        )
        for name in MODEL_FILES:
            target = model_dir / name
            with (sidecar_dir / name).open("rb") as reader, target.open("xb") as writer:
                while chunk := reader.read(4 * 1024 * 1024):
                    writer.write(chunk)
            target.chmod(0o600)
        candidates[threshold] = {
            "request": request,
            "candidate": candidate_ref,
            "candidate_record": candidate,
            "checkpoint": checkpoint,
        }
        model_dirs[threshold] = model_dir
        print(f"sealed target checkpoint {threshold}", flush=True)

    candidates_seal = root / "candidates-seal.json"
    write_new(candidates_seal, canonical({
        "schema": "cerebro.tidex.v69_candidates_seal/v1",
        "candidates": candidates, "baseline_updates_sha256": json_sha(baseline_updates),
        "collector_sha256": source_sha, "binary_sha256": binary_sha,
        "target_receiver_execution_performed": False,
    }))
    verify_frozen_inputs(reproducibility, binary)

    # Confirmation reveal is written only after every target candidate and
    # checkpoint has been sealed. This is the first receiver target evaluation.
    reveal_path = root / "confirmation-reveal.json"
    if json_sha(reveal) != method_precommit["confirmation_reveal_sha256"]:
        raise RuntimeError("V69 confirmation reveal commitment mismatch")
    write_new(reveal_path, canonical(reveal))

    def replay_package(model_dir: Path, model_sha: str, label: str) -> dict[str, Any]:
        package = root / "evaluation" / f"package-{label}.json"
        write_new(
            package,
            canonical({"model_sha256": model_sha, "model_files_sha256": receiver_files, "reveal": reveal}),
        )
        output = root / "evaluation" / f"replay-{label}.json"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--replay-model", str(model_dir),
             "--package", str(package), "--output", str(output), "--threads", str(threads)],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True, timeout=900,
        )
        return json.loads(output.read_bytes())

    base_replay = replay_package(
        RECEIVER_DIR, receiver_weight_sha, "base"
    )
    target_replays: dict[int, dict[str, Any]] = {}
    for threshold in CONFIRMATION_THRESHOLDS:
        sha = candidates[threshold]["checkpoint"]["materialization"]["output_model_sha256"]
        target_replays[threshold] = replay_package(
            model_dirs[threshold], sha, f"target-{threshold:03d}"
        )

    baseline_package = root / "evaluation" / "package-baselines.json"
    baseline_output = root / "evaluation" / "replay-baselines.json"
    write_new(baseline_package, canonical({
        "model_sha256": receiver_weight_sha, "model_files_sha256": receiver_files,
        "updates": baseline_updates, "updates_sha256": json_sha(baseline_updates), "reveal": reveal,
    }))
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--replay-baselines", str(RECEIVER_DIR),
         "--package", str(baseline_package), "--output", str(baseline_output), "--threads", str(threads)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True, timeout=1800,
    )
    baseline_results = json.loads(baseline_output.read_bytes())

    # Donor confirmation is evaluated only after receiver candidates are sealed.
    donor_tokenizer = AutoTokenizer.from_pretrained(
        DONOR_DIR, local_files_only=True, trust_remote_code=False
    )
    donor_model = AutoModelForCausalLM.from_pretrained(
        DONOR_DIR,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    donor_model.eval()
    donor_model.requires_grad_(False)
    donor_true = one_token(donor_tokenizer, " true")
    donor_false = one_token(donor_tokenizer, " false")
    donor_accuracies = []
    for threshold in CONFIRMATION_THRESHOLDS:
        values = numbers(threshold, RECEIVER_VALIDATION_OFFSETS)
        labels = np.array(expected_labels(threshold, values), dtype=bool)
        prompts = donor_task_prompts(threshold, False, values)
        encoded = donor_tokenizer(prompts, padding=True, return_tensors="pt")
        positions = encoded.attention_mask.sum(1) - 1
        with torch.inference_mode():
            output = donor_model(**encoded, use_cache=False)
        margins = (
            output.logits[torch.arange(len(values)), positions, donor_true]
            - output.logits[torch.arange(len(values)), positions, donor_false]
        ).double().cpu().numpy()
        donor_accuracies.append(float(((margins > 0) == labels).mean()))
    del donor_model, donor_tokenizer
    gc.collect()

    cases = reveal["cases"]
    base_margins = np.array(base_replay["case_margins"])
    base_controls = np.array(base_replay["generic_control_margins"])
    base_passes = []
    compiled_passes = []
    cyclic_wrong_passes = []
    per_target = []
    non_degrading = 0
    for target_index, threshold in enumerate(CONFIRMATION_THRESHOLDS):
        indices = [
            index
            for index, case in enumerate(cases)
            if case["capability_id"] == cap_id(threshold)
        ]
        labels = np.array([cases[index]["expected_true"] for index in indices], dtype=bool)
        correct_margins = np.array(target_replays[threshold]["case_margins"])[indices]
        wrong_threshold = CONFIRMATION_THRESHOLDS[(target_index + 1) % len(CONFIRMATION_THRESHOLDS)]
        wrong_margins = np.array(target_replays[wrong_threshold]["case_margins"])[indices]
        base_target = base_margins[indices]
        base_vector = (base_target > 0) == labels
        correct_vector = (correct_margins > 0) == labels
        wrong_vector = (wrong_margins > 0) == labels
        base_accuracy = float(base_vector.mean())
        correct_accuracy = float(correct_vector.mean())
        wrong_accuracy = float(wrong_vector.mean())
        baseline_accuracies = {
            method: float(((np.asarray(report["case_margins"]) > 0) == labels).mean())
            for method, report in baseline_results["results"][str(threshold)].items()
        }
        if correct_accuracy >= base_accuracy:
            non_degrading += 1
        base_passes.extend(base_vector.tolist())
        compiled_passes.extend(correct_vector.tolist())
        cyclic_wrong_passes.extend(wrong_vector.tolist())
        control_change = (
            np.array(target_replays[threshold]["generic_control_margins"]) - base_controls
        )
        per_target.append(
            {
                "capability_id": cap_id(threshold),
                "threshold": threshold,
                "base_accuracy": base_accuracy,
                "compiled_accuracy": correct_accuracy,
                "gain": correct_accuracy - base_accuracy,
                "calibration_baseline_accuracies": baseline_accuracies,
                "cyclic_wrong_threshold": wrong_threshold,
                "cyclic_wrong_accuracy": wrong_accuracy,
                "generic_control_margin_rms_change": float(
                    np.sqrt(np.mean(np.square(control_change)))
                ),
                "compiler": candidates[threshold]["candidate_record"]["numerical"],
            }
        )

    base_passes_np = np.array(base_passes, dtype=bool)
    compiled_passes_np = np.array(compiled_passes, dtype=bool)
    gained = int(np.sum(~base_passes_np & compiled_passes_np))
    lost = int(np.sum(base_passes_np & ~compiled_passes_np))
    casewise_p = exact_binomial_p(gained, lost)
    paired_p = capability_cluster_sign_flip_p([item["gain"] for item in per_target])
    base_mean = float(np.mean([item["base_accuracy"] for item in per_target]))
    compiled_mean = float(np.mean([item["compiled_accuracy"] for item in per_target]))
    wrong_mean = float(np.mean([item["cyclic_wrong_accuracy"] for item in per_target]))
    max_control_rms = float(
        max(item["generic_control_margin_rms_change"] for item in per_target)
    )
    donor_mean = float(np.mean(donor_accuracies))
    confirmation_pass = bool(
        compiled_mean >= CRITERIA["minimum_confirmation_mean_accuracy"]
        and compiled_mean - base_mean >= CRITERIA["minimum_confirmation_mean_gain"]
        and non_degrading >= CRITERIA["minimum_non_degrading_target_count"]
        and paired_p <= CRITERIA["maximum_capability_cluster_two_sided_p"]
        and compiled_mean - wrong_mean
        >= CRITERIA["minimum_correct_vs_cyclic_wrong_advantage"]
        and max_control_rms <= CRITERIA["maximum_generic_control_margin_rms_change"]
        and donor_mean >= CRITERIA["minimum_donor_confirmation_accuracy"]
        and all(not replay["peft_module_imported"] for replay in target_replays.values())
        and all(not replay["lora_parameters_present"] for replay in target_replays.values())
    )

    baseline_means = {
        method: float(np.mean([item["calibration_baseline_accuracies"][method] for item in per_target]))
        for method in ("mean_no_target_signature", "nearest_signature", "shuffled_correspondence")
    }
    baseline_advantage = compiled_mean - max(baseline_means.values())
    baseline_gate = bool(baseline_advantage >= CRITERIA["minimum_compiled_vs_strongest_calibration_baseline_advantage"])
    verify_frozen_inputs(reproducibility, binary)
    receipt = {
        "schema": SCHEMA,
        "complete": True,
        "pass": confirmation_pass and baseline_gate,
        "behavioral_transfer_gates_pass": confirmation_pass,
        "incremental_baseline_gate_pass": baseline_gate,
        "candidates_seal_sha256": sha_file(candidates_seal),
        "scope": "six previously unexecuted threshold capabilities compiled from Qwen functional signatures into native SmolLM2-360M final-norm weights using a receiver backend calibrated only on thirteen other threshold capabilities",
        "method_precommit_sha256": sha_file(root / "method-precommit.json"),
        "calibration_receipt_sha256": sha_file(root / "calibration-receipt.json"),
        "confirmation_reveal_sha256": sha_file(reveal_path),
        "calibration": calibration_receipt,
        "confirmation": {
            "target_count": len(CONFIRMATION_THRESHOLDS),
            "calibration_baseline_mean_accuracies": baseline_means,
            "compiled_vs_strongest_calibration_baseline_advantage": baseline_advantage,
            "case_count": len(cases),
            "base_mean_accuracy": base_mean,
            "compiled_mean_accuracy": compiled_mean,
            "mean_gain": compiled_mean - base_mean,
            "cyclic_wrong_mean_accuracy": wrong_mean,
            "correct_vs_cyclic_wrong_advantage": compiled_mean - wrong_mean,
            "non_degrading_target_count": non_degrading,
            "paired": {
                "both_pass": int(np.sum(base_passes_np & compiled_passes_np)),
                "compiled_only_pass": gained,
                "base_only_pass": lost,
                "neither_pass": int(np.sum(~base_passes_np & ~compiled_passes_np)),
                "casewise_exact_two_sided_p_diagnostic_only": casewise_p,
                "capability_cluster_exact_two_sided_p": paired_p,
                "independence_unit": "capability checkpoint, not individual question",
                "independent_training_replications": 1,
            },
            "donor_mean_accuracy": donor_mean,
            "donor_target_accuracies": donor_accuracies,
            "maximum_generic_control_margin_rms_change": max_control_rms,
            "per_target": per_target,
        },
        "claim_boundary": {
            "target_lora_available_or_used": False,
            "target_task_vector_available_or_used": False,
            "target_prior_delta_w_available_or_used": False,
            "target_receiver_sft_used": False,
            "target_receiver_solution_used_in_calibration": False,
            "target_capability_used_in_receiver_basis_construction": False,
            "target_receiver_execution_before_candidates_sealed": False,
            "donor_parameter_update_used": False,
            "donor_functional_observation_used": True,
            "rust_receiver_compiler_generated_target_coordinates": True,
            "rust_weight_actuator_materialized_standalone_checkpoint": True,
            "universal_capability_portability_established": False,
            "open_domain_semantic_transfer_established": False,
            "bounded_family_only": True,
            "authorizes_promotion": False,
        },
        "criteria": CRITERIA,
    }
    write_new(root / "receipt.json", canonical(receipt))
    return receipt


READOUT_INTEGRATION_SCHEMA = "cerebro.tidex.v69_capability_ir_readout_integration/v1"
# This mode deliberately reuses an already revealed, failed R4 experiment.
# It checks integration execution, never behavioral equivalence or fresh transfer.
R4_RECOVERED_RECEIPT_SHA256 = "0609718da57ce56a41b700d1dffdd0621d888f305ada7db07f70b12013e62d89"


def read_r4_reference(root: Path, reference: dict[str, str]) -> Any:
    """Reuse the authenticated reader after confining a historical reference."""
    path = Path(reference["path"])
    if (not path.is_absolute() or not path.is_relative_to(root)
            or path.resolve() != path or not path.is_file()):
        raise RuntimeError("R4 reference is not a regular file inside its original root")
    return read_reference(reference)


def verify_r4_readout_inputs(root: Path, recovered_receipt: Path) -> dict[str, Any]:
    """Authenticate the completed historical run without requiring today's source."""
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise RuntimeError("R4 root must be an absolute real directory")
    if recovered_receipt.is_symlink() or sha_file(recovered_receipt) != R4_RECOVERED_RECEIPT_SHA256:
        raise RuntimeError("R4 recovered receipt is not the pinned historical result")
    recovered = json.loads(recovered_receipt.read_bytes())
    if recovered.get("schema") != "cerebro.tidex.v69_recovered_confirmation_receipt/v1":
        raise RuntimeError("R4 recovered receipt schema mismatch")
    receipt = recovered["receipt"]
    if receipt.get("complete") is not True or receipt.get("pass") is not False:
        raise RuntimeError("readout regression requires the already completed failed R4")
    for name, expected in recovered["recovery"]["input_sha256"].items():
        path = Path(name)
        if not path.is_relative_to(root) or path.resolve() != path or sha_file(path) != expected:
            raise RuntimeError(f"R4 recovered input changed: {name}")
    precommit = json.loads((root / "method-precommit.json").read_bytes())
    calibration = json.loads((root / "calibration-receipt.json").read_bytes())
    seal = json.loads((root / "candidates-seal.json").read_bytes())
    reveal = json.loads((root / "confirmation-reveal.json").read_bytes())
    for field, filename in (
        ("method_precommit_sha256", "method-precommit.json"),
        ("calibration_receipt_sha256", "calibration-receipt.json"),
        ("candidates_seal_sha256", "candidates-seal.json"),
        ("confirmation_reveal_sha256", "confirmation-reveal.json"),
    ):
        if sha_file(root / filename) != receipt[field]:
            raise RuntimeError(f"R4 receipt binding mismatch: {filename}")
    if (precommit.get("protocol_revision") != "V69-R4"
            or precommit["confirmation_thresholds"] != [21, 31, 41, 51, 61, 71]
            or precommit["confirmation_target_capability_ids"] != [
                cap_id(value) for value in (21, 31, 41, 51, 61, 71)]
            or precommit["calibration_capability_ids"] != [
                cap_id(value) for value in range(20, 81, 5)]
            or calibration.get("pass") is not True
            or seal.get("target_receiver_execution_performed") is not False
            or set(seal["candidates"]) != {"21", "31", "41", "51", "61", "71"}
            or json_sha(reveal) != precommit["confirmation_reveal_sha256"]):
        raise RuntimeError("R4 calibration, target split or candidate seal mismatch")
    manifest_reference = precommit["reproducibility_bundle"]
    manifest = read_r4_reference(root, manifest_reference)
    bundle = Path(manifest_reference["path"]).parent
    if (manifest.get("schema") != "cerebro.tidex.experiment_reproducibility_bundle/v1"
            or manifest["collector_sha256"] != seal["collector_sha256"]
            or manifest["tidex_binary_sha256"] != seal["binary_sha256"]
            or sha_file(bundle / "bin" / "tidex") != seal["binary_sha256"]):
        raise RuntimeError("R4 implementation identity mismatch")
    for relative, expected in manifest["source_sha256"].items():
        path = bundle / "source" / relative
        if (not path.is_relative_to(bundle / "source") or path.resolve() != path
                or sha_file(path) != expected):
            raise RuntimeError(f"R4 frozen source changed: {relative}")
    frozen_collector = bundle / "source/quality/experiments/v69_target_update_free_compilation.py"
    if sha_file(frozen_collector) != seal["collector_sha256"]:
        raise RuntimeError("R4 frozen collector mismatch")

    # Recursively authenticate all JSON evidence used by the chosen request.
    # Dense-vector references are authenticated as bytes, never parsed as JSON.
    seen: set[tuple[str, str]] = set()

    def authenticate_graph(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                authenticate_graph(child)
        elif isinstance(value, dict):
            if "path" in value and "sha256" in value:
                key = (value["path"], value["sha256"])
                if key in seen:
                    return
                seen.add(key)
                path = Path(value["path"])
                if not path.is_relative_to(root) or path.resolve() != path:
                    raise RuntimeError("R4 evidence reference escaped its original root")
                if path.suffix == ".json":
                    authenticate_graph(read_r4_reference(root, value))
                elif sha_file(path) != value["sha256"]:
                    raise RuntimeError("R4 binary evidence digest changed")
            else:
                for child in value.values():
                    authenticate_graph(child)

    original = seal["candidates"]["21"]
    authenticate_graph(original)
    candidate = read_r4_reference(root, original["candidate"])
    request = read_r4_reference(root, original["request"])
    target = read_r4_reference(root, request["target"])
    evidence = read_r4_reference(root, target["functional_evidence"])
    protocol = read_r4_reference(root, request["protocol"])
    if (candidate != original["candidate_record"] or candidate["blockers"]
            or candidate["target_capability_id"] != cap_id(21)
            or target["capability_id"] != cap_id(21)
            or evidence["capability_id"] != cap_id(21)
            or evidence["receiver_data_used"] is not False
            or request["proposal_method"] != calibration["proposal_method"]
            or request["policy"]["ridge"] != calibration["selected_ridge"]):
        raise RuntimeError("R4 target 21, donor evidence or frozen method mismatch")
    for value in seal["candidates"].values():
        checkpoint = value["checkpoint"]
        path = Path(checkpoint["output_path"])
        if not path.is_relative_to(root) or path.resolve() != path:
            raise RuntimeError("R4 checkpoint escaped its original root")
        if sha_file(path) != checkpoint["materialization"]["output_model_sha256"]:
            raise RuntimeError("R4 sealed checkpoint changed")
    return {
        "precommit": precommit, "calibration": calibration, "receipt": receipt,
        "seal": seal, "original": original, "request": request, "target": target,
        "evidence": evidence, "protocol": protocol, "frozen_collector": frozen_collector,
        "historical_input_sha256": recovered["recovery"]["input_sha256"],
    }


def resolve_resident_readout_tensor(model: Any, checkpoint: Path) -> str:
    """Resolve one stored tensor by actual runtime Parameter identity, never name fallback."""
    output = model.get_output_embeddings()
    inputs = model.get_input_embeddings()
    if (output is None or inputs is None
            or not isinstance(output.weight, torch.nn.Parameter)
            or output.weight is not model.lm_head.weight):
        raise RuntimeError("readout output embedding is not the observed lm_head Parameter")
    weight = output.weight
    declared_tie = getattr(model.config, "tie_word_embeddings", None)
    actual_tie = inputs.weight is weight
    if type(declared_tie) is not bool or declared_tie != actual_tie:
        raise RuntimeError("readout declared tying differs from actual Parameter identity")
    aliases = {
        name for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter is weight
    }
    if not aliases or weight.ndim != 2 or min(weight.shape) < 1:
        raise RuntimeError("readout runtime parameter shape or aliases invalid")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
        resident = sorted(aliases.intersection(archive.keys()))
        if len(resident) != 1:
            raise RuntimeError("readout requires exactly one unambiguous resident Parameter alias")
        tensor_id = resident[0]
        if archive.get_slice(tensor_id).get_shape() != list(weight.shape):
            raise RuntimeError("readout resident tensor shape differs from runtime Parameter")
    return tensor_id


def collect_observed_linear_readout(
    evidence: dict[str, Any], donor: dict[str, Any], threads: int,
) -> dict[str, Any]:
    """Observe the actual final-norm output and actual full lm_head forward."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    if (sha_file(DONOR_DIR / "model.safetensors") != donor["weight_sha256"]
            or sha_file(DONOR_DIR / "config.json") != donor["config_sha256"]
            or sha_file(DONOR_DIR / "tokenizer.json") != donor["tokenizer_sha256"]):
        raise RuntimeError("R4 donor checkpoint or tokenizer changed")
    prompts = evidence["prompts"]
    if (len(prompts) != 9 or [hashlib.sha256(p.encode()).hexdigest() for p in prompts]
            != evidence["prompt_sha256"]):
        raise RuntimeError("R4 donor prompts changed")
    tokenizer = AutoTokenizer.from_pretrained(
        DONOR_DIR, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    positive = one_token(tokenizer, " true")
    negative = one_token(tokenizer, " false")
    model = AutoModelForCausalLM.from_pretrained(
        DONOR_DIR, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.requires_grad_(False)
    if (type(model).__name__ != donor["architecture"]
            or any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                   for parameter in model.parameters())
            or model.lm_head.bias is not None):
        raise RuntimeError("readout capture requires the authenticated unbiased CPU F32 donor")
    tensor_id = resolve_resident_readout_tensor(model, DONOR_DIR / "model.safetensors")
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    if encoded.input_ids.shape[1] > 512:
        raise RuntimeError("readout capture exceeds the original donor token budget")
    positions = encoded.attention_mask.sum(1) - 1
    rows = torch.arange(len(prompts))
    captured_norm = []
    captured_head = []
    norm_hook = model.model.norm.register_forward_hook(
        lambda _module, _inputs, output: captured_norm.append(output[rows, positions].detach().clone())
    )
    head_hook = model.lm_head.register_forward_pre_hook(
        lambda _module, inputs: captured_head.append(inputs[0][rows, positions].detach().clone())
    )
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
    finally:
        norm_hook.remove()
        head_hook.remove()
    if (len(captured_norm) != 1 or len(captured_head) != 1
            or not torch.equal(captured_norm[0], captured_head[0])):
        raise RuntimeError("donor final norm is not the observed lm_head input")
    hidden = captured_head[0]
    positive_logits = output.logits[rows, positions, positive]
    negative_logits = output.logits[rows, positions, negative]
    margins = (positive_logits - negative_logits).double().cpu().tolist()
    if functional_signature_sha256(margins) != functional_signature_sha256(evidence["raw_logit_margins"]):
        raise RuntimeError("reobserved donor margins differ from the immutable R4 evidence")
    if (not torch.isfinite(hidden).all() or not torch.isfinite(positive_logits).all()
            or not torch.isfinite(negative_logits).all()):
        raise RuntimeError("nonfinite real donor readout observation")
    observed = {
        "schema": "cerebro.tidex.observed_linear_readout/v1",
        "capability_id": evidence["capability_id"],
        "model_sha256": donor["weight_sha256"],
        "tensor_id": tensor_id, "positive_row": positive, "negative_row": negative,
        "arithmetic_profile": "f32_cpu_no_tf32/v1",
        "inputs": hidden.double().cpu().tolist(),
        "observed_positive_logits": positive_logits.double().cpu().tolist(),
        "observed_negative_logits": negative_logits.double().cpu().tolist(),
        "observed_margins": margins, "prompts": prompts,
        "prompt_sha256": evidence["prompt_sha256"], "receiver_data_used": False,
    }
    del model, tokenizer, output, hidden
    gc.collect()
    return observed


def verify_capability_ir_run(
    binary: Path, root: Path, destination: Path, recovered_receipt: Path, threads: int,
) -> dict[str, Any]:
    """Check integration execution through real CapabilityIr, not behavior equivalence."""
    if (not destination.is_absolute() or destination.resolve() != destination
            or destination.is_relative_to(root) or root.is_relative_to(destination)
            or destination.exists()):
        raise RuntimeError("operational output must be a new real directory outside the R4 root")
    original = verify_r4_readout_inputs(root, recovered_receipt)
    destination.mkdir(parents=True, mode=0o700)
    reproducibility = freeze_reproducibility_bundle(
        destination / "reproducibility", binary, collector=Path(__file__)
    )
    model_dir = root / "models/operational-readout-target-021"
    if model_dir.exists():
        raise RuntimeError("operational target checkpoint already exists; refusing to overwrite")
    integration_precommit = {
        "schema": READOUT_INTEGRATION_SCHEMA,
        "kind": "already_observed_R4_target_021_software_integration",
        "original_recovered_receipt_sha256": sha_file(recovered_receipt),
        "original_candidates_seal_sha256": sha_file(root / "candidates-seal.json"),
        "original_request": original["original"]["request"],
        "original_transfer_pass": False,
        "target_selection": "fixed target 21; no new target selection or method tuning",
        "reproducibility_bundle": reproducibility,
        "new_confirmation": False, "authorizes_promotion": False,
        "compiler_policy_changed": False,
    }
    write_new(destination / "integration-precommit.json", canonical(integration_precommit))
    try:
        observed = collect_observed_linear_readout(
            original["evidence"], original["precommit"]["donor"], threads
        )
        print("observed original donor target 21 final-norm and lm_head execution", flush=True)
        source_root = destination / "target-021"
        source_root.mkdir(mode=0o700)
        describe_input = destination / "describe-input.json"
        write_new(describe_input, canonical({
            "schema": "cerebro.tidex.describe_linear_readout_input/v1",
            "model_path": str(DONOR_DIR / "model.safetensors"),
            "tensor_id": observed["tensor_id"],
            "positive_row": observed["positive_row"], "negative_row": observed["negative_row"],
            "capability_id": observed["capability_id"],
        }))
        description = tidex(binary, root, "describe-readout", str(describe_input))
        if (description["inspection"]["model_sha256"] != observed["model_sha256"]
                or description["inspection"]["input_dimension"] != len(observed["inputs"][0])):
            raise RuntimeError("Rust readout descriptor does not describe the observed checkpoint")
        # Preserve the exact Rust canonical bytes; Python must not reserialize it.
        write_new(source_root / "operator.json", description["descriptor_json"].encode("utf-8"))
        write_new(source_root / "observations.json", canonical(observed))
        acquire_input = destination / "acquire-input.json"
        write_new(acquire_input, canonical({
            "schema": "cerebro.tidex.acquire_linear_readout_input/v1",
            "source_root": str(source_root),
            "model_path": str(DONOR_DIR / "model.safetensors"),
            "descriptor_relative_path": "operator.json",
            "evidence_relative_path": "observations.json",
        }))
        acquisition = tidex(binary, root, "acquire-readout", str(acquire_input))
        request = {**original["request"], "capability_ir_readout": acquisition}
        request_ref = put(root, request)
        request_path = reference_file(root, "v69-operational-readout-request-021", request_ref)
        candidate_ref = tidex(binary, root, "compile", str(request_path))
        candidate_path = reference_file(root, "v69-operational-readout-candidate-021", candidate_ref)
        candidate = tidex(binary, root, "inspect", str(candidate_path))
        summary = candidate.get("capability_ir_readout")
        if (candidate["blockers"] or not summary
                or candidate["target_capability_id"] != cap_id(21)
                or candidate["authorizes_promotion"] is not False
                or candidate["numerical"]["validation_profile"] != "parametric_cross_validation"):
            raise RuntimeError(f"real CapabilityIr candidate rejected: {candidate['blockers']}")
        write_new(destination / "candidate.json", canonical(candidate))
        receiver = original["precommit"]["receiver"]
        if sha_file(RECEIVER_DIR / "model.safetensors") != receiver["weight_sha256"]:
            raise RuntimeError("original receiver checkpoint changed")
        copy_model_sidecars(RECEIVER_DIR, model_dir, receiver["sidecars_sha256"])
        checkpoint = tidex(
            binary, root, "materialize", str(candidate_path),
            "--base-model", str(RECEIVER_DIR / "model.safetensors"),
            "--output", str(model_dir / "model.safetensors"),
        )
        write_new(destination / "checkpoint.json", canonical(checkpoint))
        print("sealed CapabilityIr-derived target 21 standalone checkpoint", flush=True)
        package = json.loads((root / "evaluation/package-target-021.json").read_bytes())
        package["model_sha256"] = checkpoint["materialization"]["output_model_sha256"]
        package_path = destination / "replay-package.json"
        replay_path = destination / "replay.json"
        write_new(package_path, canonical(package))
        subprocess.run(
            [sys.executable, str(original["frozen_collector"]), "--replay-model", str(model_dir),
             "--package", str(package_path), "--output", str(replay_path),
             "--threads", str(threads)],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True, timeout=900,
        )
        replay = json.loads(replay_path.read_bytes())
        old_replay = json.loads((root / "evaluation/replay-target-021.json").read_bytes())
        if replay["peft_module_imported"] or replay["lora_parameters_present"]:
            raise RuntimeError("operational standalone replay imported an adapter")
        differences = {}
        for field in ("case_margins", "generic_control_margins"):
            actual = np.asarray(replay[field], dtype=np.float64)
            historical = np.asarray(old_replay[field], dtype=np.float64)
            if actual.shape != historical.shape or not np.isfinite(actual).all():
                raise RuntimeError("operational replay shape or finiteness mismatch")
            differences[field] = {
                "maximum_absolute_change": float(np.max(np.abs(actual - historical))),
                "rms_change": float(np.sqrt(np.mean(np.square(actual - historical)))),
                "sign_changes": int(np.count_nonzero((actual > 0) != (historical > 0))),
            }
        coordinate_delta = (
            np.asarray(candidate["numerical"]["target_delta"])
            - np.asarray(original["original"]["candidate_record"]["numerical"]["target_delta"])
        )
        indices = [i for i, case in enumerate(package["reveal"]["cases"])
                   if case["capability_id"] == cap_id(21)]
        labels = np.asarray([package["reveal"]["cases"][i]["expected_true"] for i in indices])
        actual_accuracy = float(np.mean((np.asarray(replay["case_margins"])[indices] > 0) == labels))
        verify_frozen_inputs(reproducibility, binary)
        for name, expected in original["historical_input_sha256"].items():
            if sha_file(Path(name)) != expected:
                raise RuntimeError("historical R4 evidence changed during integration")
        receipt = {
            "schema": READOUT_INTEGRATION_SCHEMA, "complete": True,
            "integration_pass": True, "execution_pipeline_complete": True,
            "behavior_equivalence_claimed": False, "original_transfer_pass": False,
            "new_transfer_confirmation": False, "independent_replication": False,
            "scope": "one already-observed donor lm_head boundary acquired as executable CapabilityIr, compiled through the existing receiver backend and executed as a standalone receiver checkpoint",
            "full_threshold_algorithm_extracted": False,
            "full_transformer_extracted": False,
            "authorizes_promotion": False,
            "integration_precommit_sha256": sha_file(destination / "integration-precommit.json"),
            "acquisition": acquisition, "request": request_ref, "candidate": candidate_ref,
            "capability_ir_readout": summary, "checkpoint": checkpoint,
            "donor_original_f32_margins_reproduced_exactly": True,
            "receiver_optimizer_steps": 0, "receiver_forward_before_new_checkpoint_sealed": False,
            "fresh_process_checkpoint_execution": True,
            "original_checkpoint_sha256": original["original"]["checkpoint"]["materialization"]["output_model_sha256"],
            "checkpoint_bytes_identical_to_original": package["model_sha256"] == old_replay["model_sha256"],
            "receiver_coordinate_change_l2": float(np.linalg.norm(coordinate_delta)),
            "receiver_margin_changes_from_original": differences,
            "already_observed_target_accuracy_diagnostic_only": actual_accuracy,
            "original_transfer_failure_preserved": original["receipt"]["pass"] is False,
            "replay_sha256": sha_file(replay_path),
        }
        write_new(destination / "integration-receipt.json", canonical(receipt))
        return receipt
    except Exception as error:
        write_new(destination / "integration-failure.json", canonical({
            "schema": READOUT_INTEGRATION_SCHEMA, "complete": False, "integration_pass": False,
            "execution_pipeline_complete": False, "behavior_equivalence_claimed": False,
            "original_transfer_pass": False, "authorizes_promotion": False, "error": str(error),
        }))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tidex", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--verify-capability-ir-run", type=Path)
    parser.add_argument("--operational-output", type=Path)
    parser.add_argument("--original-receipt", type=Path, default=PROJECT_ROOT / "target/v69-evolution-diagnostics-20260908-a/r4-recovered-receipt.json")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--assemble-distributed-basis", type=Path)
    parser.add_argument("--legacy-final-norm", action="store_true")
    parser.add_argument("--replay-model", type=Path)
    parser.add_argument("--replay-baselines", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 16:
        parser.error("threads must be 1..16")
    if args.assemble_distributed_basis:
        if (
            not args.tidex
            or not args.root
            or args.operational_output
            or args.verify_capability_ir_run
            or args.calibration_only
            or args.legacy_final_norm
            or args.replay_baselines
            or args.replay_model
            or args.package
            or args.output
        ):
            parser.error(
                "distributed assembly requires tidex/root and excludes legacy/run/replay modes"
            )
        result = assemble_distributed_lora_compiler_basis(
            args.tidex, args.root, args.assemble_distributed_basis
        )
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if args.verify_capability_ir_run:
        if (not args.tidex or not args.operational_output or args.root or args.calibration_only
                or args.legacy_final_norm or args.replay_baselines or args.replay_model
                or args.package or args.output):
            parser.error("CapabilityIr regression requires tidex/operational-output and excludes run/replay modes")
        result = verify_capability_ir_run(args.tidex, args.verify_capability_ir_run,
                                         args.operational_output, args.original_receipt, args.threads)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["integration_pass"] else 2
    if args.operational_output:
        parser.error("operational-output requires verify-capability-ir-run")
    if args.replay_baselines:
        if args.replay_model or not args.package or not args.output:
            parser.error("baseline replay requires package/output and excludes model replay")
        replay_baselines(args.replay_baselines, args.package, args.output, args.threads)
        return 0
    if args.replay_model:
        if not args.package or not args.output:
            parser.error("replay mode requires --package and --output")
        replay_model(args.replay_model, args.package, args.output, args.threads)
        return 0
    if not args.tidex or not args.root:
        parser.error("run mode requires --tidex and --root")
    if not args.legacy_final_norm:
        parser.error(
            "the 960-parameter final-norm backend is retired; use "
            "--assemble-distributed-basis <manifest.json>, or pass "
            "--legacy-final-norm only to reproduce historical V69 evidence"
        )
    try:
        receipt = main_run(
            args.tidex, args.root, args.threads, calibration_only=args.calibration_only
        )
    except Exception as error:
        if args.root and args.root.exists():
            failure = args.root / "failure.json"
            if not failure.exists():
                write_new(
                    failure,
                    canonical(
                        {
                            "schema": SCHEMA,
                            "complete": False,
                            "error": str(error),
                        }
                    ),
                )
        raise
    if args.calibration_only:
        printable = receipt
    else:
        printable = {
            "complete": receipt["complete"],
            "pass": receipt["pass"],
            "calibration": receipt["calibration"],
            "confirmation": {
                key: value
                for key, value in receipt["confirmation"].items()
                if key != "per_target"
            },
            "claim_boundary": receipt["claim_boundary"],
        }
    print(json.dumps(printable, indent=2), flush=True)
    return 0 if receipt["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
