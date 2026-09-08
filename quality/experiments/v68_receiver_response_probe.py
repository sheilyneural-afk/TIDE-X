#!/usr/bin/env python3
"""V68 engineering probe: measured responses -> Rust -> standalone weights.

This is system identification of a deliberately bounded, locally linear native
parameter profile (the existing final RMSNorm scale). It is NOT donor-to-receiver
skill transfer, not MBPP, and not evidence of a newly learned semantic capability.
No optimizer, backward pass, LoRA, PEFT adapter, or target coefficient oracle is
used. A requested logit-margin change is precommitted independently of the basis.
Rust alone infers candidate coordinates and assembles the final dense update.
A fresh Python process evaluates each standalone checkpoint with ordinary model
forward passes; the learned inverse predictor is never used as the evaluator.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCE = Path(__file__).resolve()
REPO = SOURCE.parents[2]
BASE_SHA = "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8"
SCHEMA = "cerebro.tidex.v68_receiver_response_probe/v1"
MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise RuntimeError(f"immutable output collision: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def copy_new(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"reproducibility source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        raise RuntimeError(f"reproducibility destination already exists: {destination}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    destination.chmod(0o600)
    if sha_file(source) != sha_file(destination):
        raise RuntimeError("reproducibility copy digest mismatch")


def reproducibility_source_files() -> list[Path]:
    files = [REPO / name for name in (
        "Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml",
        "deny.toml", ".cargo/config.toml",
    )]
    for relative in ("src", "tests", "quality/experiments"):
        for path in (REPO / relative).rglob("*"):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                files.append(path)
    unique = sorted(set(files))
    if any(not path.is_relative_to(REPO) for path in unique):
        raise RuntimeError("reproducibility source escaped repository")
    return unique


def freeze_reproducibility_bundle(
    destination: Path, binary: Path, collector: Path | None = None
) -> dict[str, Any]:
    """Freeze enough exact source to rebuild every declared Rust target and experiment."""
    if destination.exists():
        raise RuntimeError("reproducibility bundle already exists")
    destination.mkdir(parents=True, mode=0o700)
    source_root = destination / "source"
    source_hashes: dict[str, str] = {}
    for source in reproducibility_source_files():
        relative = source.relative_to(REPO)
        target = source_root / relative
        copy_new(source, target)
        source_hashes[str(relative)] = sha_file(target)
    binary_target = destination / "bin" / "tidex"
    copy_new(binary, binary_target)
    binary_target.chmod(0o700)
    collector = SOURCE if collector is None else collector.resolve()
    if collector not in reproducibility_source_files():
        raise RuntimeError("collector is outside the frozen source set")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "schema": "cerebro.tidex.experiment_reproducibility_bundle/v1",
        "git_head": head,
        "source_file_count": len(source_hashes),
        "source_sha256": source_hashes,
        "tidex_binary_sha256": sha_file(binary_target),
        "collector_sha256": source_hashes[str(collector.relative_to(REPO))],
        "rebuild_scope": "complete canonical Cargo build inputs plus src, tests, and quality/experiments trees",
    }
    write_new(destination / "manifest.json", canonical(manifest))
    return {
        "path": str(destination / "manifest.json"),
        "sha256": sha_file(destination / "manifest.json"),
    }


def put(root: Path, value: Any) -> dict[str, str]:
    data = canonical(value)
    sha = hashlib.sha256(data).hexdigest()
    path = root / "state/probe_inputs/by-sha" / f"{sha}.json"
    write_new(path, data)
    return {"path": str(path), "sha256": sha}


def read_reference(reference: dict[str, str]) -> Any:
    path = Path(reference["path"])
    if path.is_symlink() or sha_file(path) != reference["sha256"]:
        raise RuntimeError("input reference changed")
    return json.loads(path.read_bytes())


def tidex(binary: Path, root: Path, *arguments: str) -> Any:
    result = subprocess.run(
        [str(binary), "receiver", *arguments], cwd=REPO,
        env={**os.environ, "TIDEX_PRIVATE_ROOT": str(root)},
        capture_output=True, text=True, timeout=240,
    )
    if result.returncode:
        raise RuntimeError(f"Rust command rejected: {result.stderr.strip()}")
    return json.loads(result.stdout)


def reference_file(root: Path, name: str, reference: dict[str, str]) -> Path:
    path = root / "references" / f"{name}.json"
    write_new(path, canonical(reference))
    return path


def load_unique_reference_alias(root: Path, names: tuple[str, ...]) -> tuple[Path, dict[str, str]]:
    """Resolve one explicitly versioned compatibility alias, never first-match fallback."""
    candidates = [root / "references" / name for name in names]
    present = [path for path in candidates if path.exists()]
    if len(present) != 1:
        raise RuntimeError(
            "expected exactly one receiver request reference alias: "
            + ",".join(str(path) for path in candidates)
        )
    path = present[0]
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("receiver request reference alias is not a regular file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise RuntimeError("receiver request reference alias payload invalid")
    read_reference(value)
    return path, value


def verify_original_implementation_binding(root: Path, precommit: dict[str, Any]) -> None:
    bundle_reference = precommit.get("reproducibility_bundle")
    if isinstance(bundle_reference, dict):
        manifest = read_reference(bundle_reference)
        if manifest.get("schema") != "cerebro.tidex.experiment_reproducibility_bundle/v1":
            raise RuntimeError("original reproducibility manifest schema mismatch")
        relative_collector = str(SOURCE.relative_to(REPO))
        if manifest.get("collector_sha256") != precommit["collector_sha256"]:
            raise RuntimeError("original collector digest not bound by reproducibility bundle")
        if manifest.get("source_sha256", {}).get(relative_collector) != precommit["collector_sha256"]:
            raise RuntimeError("original collector source missing from reproducibility bundle")
        bundle_root = Path(bundle_reference["path"]).parent
        if sha_file(bundle_root / "source" / relative_collector) != precommit["collector_sha256"]:
            raise RuntimeError("original frozen collector changed")
        if sha_file(bundle_root / "bin" / "tidex") != precommit["tidex_binary_sha256"]:
            raise RuntimeError("original frozen compiler changed")
        return
    # Explicit v1 compatibility for the already preserved rejected V68 round.
    collector = root / "rejected-round-source/collector.py"
    compiler = root / "rejected-round-source/tidex"
    if sha_file(collector) != precommit["collector_sha256"]:
        raise RuntimeError("legacy original measurement collector not preserved")
    if sha_file(compiler) != precommit["tidex_binary_sha256"]:
        raise RuntimeError("legacy original rejected compiler not preserved")


def load_model(path: Path, threads: int) -> tuple[Any, Any]:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    if type(model).__name__ != "LlamaForCausalLM":
        raise RuntimeError("probe supports the inspected LlamaForCausalLM profile only")
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("adapter parameters are forbidden")
    model.config.use_cache = False
    return model, tokenizer


def margins(model: Any, tokenizer: Any, prompts: list[str], labels: list[int]) -> torch.Tensor:
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    if encoded.input_ids.shape[1] > 64:
        raise RuntimeError("probe input exceeds precommitted token budget")
    positions = encoded.attention_mask.sum(1) - 1
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits
        selected = logits[torch.arange(len(prompts)), positions]
        values = selected[:, labels[0]] - selected[:, labels[1]]
    if not torch.isfinite(values).all():
        raise RuntimeError("nonfinite response")
    return values.detach().double().cpu()


def plan_value() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "purpose": "native-parameter response identification; not semantic capability transfer",
        "seed": 20260908,
        "basis_norm": 0.2,
        "requested_response": [0.04, -0.03, 0.02, -0.01],
        "labels": [" true", " false"],
        "probes": [
            "Statement: A triangle has three sides. Answer:",
            "Statement: Water is a metal. Answer:",
            "Statement: Two plus two equals four. Answer:",
            "Statement: The opposite of hot is cold. Answer:",
        ],
        "risk_controls": [
            "Statement: A book contains pages. Answer:",
            "Statement: A square has five sides. Answer:",
            "Statement: Ice is frozen water. Answer:",
            "Statement: The sun is a musical instrument. Answer:",
        ],
        "unseen_controls": [
            "Statement: A week contains seven days. Answer:",
            "Statement: A spoon is a type of cloud. Answer:",
            "Statement: A circle has no corners. Answer:",
            "Statement: A tree is made from glass. Answer:",
        ],
        "criteria": {
            "maximum_actual_target_relative_error": 0.05,
            "minimum_correct_wrong_error_ratio": 4.0,
            "maximum_unseen_control_margin_rms_change": 0.1,
        },
        "policy": {
            "schema": "cerebro.tidex.receiver_compiler_policy/v1",
            "ridge": 1e-10,
            "minimum_decoder_loo_r2": 0.99,
            "minimum_encoder_loo_r2": 0.99,
            "minimum_decoder_loo_cosine": 0.99,
            "maximum_functional_relative_error": 0.05,
            "minimum_identity_margin": 0.1,
            "maximum_quadratic_cost": 0.01,
        },
    }


def replay(model_dir: Path, package_path: Path, output: Path, threads: int) -> None:
    package = json.loads(package_path.read_bytes())
    if sha_file(model_dir / "model.safetensors") != package["checkpoint_sha256"]:
        raise RuntimeError("replay checkpoint identity mismatch")
    for name, expected in package["model_files_sha256"].items():
        if sha_file(model_dir / name) != expected:
            raise RuntimeError("replay tokenizer/config identity mismatch")
    if any("adapter" in path.name or "lora" in path.name for path in model_dir.iterdir()):
        raise RuntimeError("standalone model contains adapter files")
    model, tokenizer = load_model(model_dir, threads)
    result = margins(model, tokenizer, package["prompts"], package["label_ids"])
    write_new(output, canonical({
        "schema": "cerebro.tidex.v68_native_forward_replay/v1",
        "checkpoint_sha256": package["checkpoint_sha256"],
        "margins": result.tolist(),
        "model_class": type(model).__name__,
        "lora_parameters_present": False,
        "peft_module_imported": "peft" in sys.modules,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
    }))


def run(base_dir: Path, binary: Path, root: Path, threads: int) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError("run directory already exists; preserve it rather than repeating measurements")
    if not root.is_absolute() or root == REPO or REPO in root.parents:
        raise RuntimeError("run root must be absolute and outside the checkout")
    root.mkdir(parents=True, mode=0o700)
    with (root / "run.lock").open("xb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_sha = sha_file(SOURCE)
        binary_sha = sha_file(binary)
        reproducibility_bundle = freeze_reproducibility_bundle(
            root / "reproducibility" / "initial", binary
        )
        base_file = base_dir / "model.safetensors"
        if sha_file(base_file) != BASE_SHA:
            raise RuntimeError("the exact original SmolLM2-1.7B checkpoint is required")
        files_sha = {name: sha_file(base_dir / name) for name in MODEL_FILES}
        plan = plan_value()
        # Targets and acceptance thresholds are fixed before observing any
        # intervention response. No retries with weakened thresholds exist.
        write_new(root / "precommit.json", canonical({
            **plan, "base_model_sha256": BASE_SHA, "collector_sha256": source_sha,
            "tidex_binary_sha256": binary_sha, "model_files_sha256": files_sha,
            "reproducibility_bundle": reproducibility_bundle,
        }))
        model, tokenizer = load_model(base_dir, threads)
        encoded_labels = [tokenizer.encode(text, add_special_tokens=False) for text in plan["labels"]]
        if any(len(ids) != 1 for ids in encoded_labels):
            raise RuntimeError(f"precommitted verbalizers are not single-token: {encoded_labels}")
        labels = [ids[0] for ids in encoded_labels]
        all_prompts = plan["probes"] + plan["risk_controls"] + plan["unseen_controls"]
        baseline = margins(model, tokenizer, all_prompts, labels)
        norm_layer = model.model.norm
        original = norm_layer.weight.detach().clone()
        if original.shape != (2048,) or type(norm_layer).__name__ != "LlamaRMSNorm":
            raise RuntimeError("native norm profile differs from the inspected model")
        captured: list[torch.Tensor] = []
        handle = norm_layer.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach()))
        try:
            margins(model, tokenizer, plan["probes"], labels)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("ambiguous native norm observation")
        encoded = tokenizer(plan["probes"], padding=True, return_tensors="pt")
        positions = encoded.attention_mask.sum(1) - 1
        hidden = captured.pop()[torch.arange(4), positions].float()
        normalized = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + norm_layer.variance_epsilon)
        head_difference = model.lm_head.weight[labels[0]] - model.lm_head.weight[labels[1]]
        sensitivities = (normalized * head_difference).double()
        if torch.linalg.matrix_rank(sensitivities).item() != 4:
            raise RuntimeError("response profile is not independently controllable in four dimensions")
        # Basis construction uses receiver response derivatives only; the
        # requested target values never enter this calculation.
        q, _ = torch.linalg.qr(sensitivities.T, mode="reduced")
        axes = (plan["basis_norm"] * q.T).float().contiguous()
        construction = put(root, {
            "method": "native_final_norm_response_sensitivity_qr",
            "collector_sha256": source_sha,
            "base_model_sha256": BASE_SHA,
            "basis_values_sha256": hashlib.sha256(axes.numpy().tobytes()).hexdigest(),
            "target_values_used": False,
            "semantic_capability_transfer_claim": False,
        })
        axis_records = []
        for index, axis in enumerate(axes):
            value_reference = put(root, axis.tolist())
            value_file = reference_file(root, f"axis-{index}", value_reference)
            delta = tidex(binary, root, "import-axis", str(value_file))
            axis_records.append({"axis_id": f"norm-axis-{index}", "delta": delta})
        basis = put(root, {
            "schema": "cerebro.tidex.receiver_weight_basis/v1",
            "base_model_sha256": BASE_SHA,
            "layout": {"schema": "cerebro.tidex.parameter_block_layout/v1", "blocks": [
                {"name": "model.norm.weight", "shape": [2048], "offset": 0, "count": 2048}], "total_parameter_count": 2048},
            "axes": axis_records,
            "construction_capability_ids": ["receiver.system-identification:v1"],
            "construction_evidence": construction,
        })
        protocol = put(root, {
            "schema": "cerebro.tidex.receiver_response_protocol/v1",
            "measure": "next_token_logit_margin_change", "base_model_sha256": BASE_SHA,
            "model_config_sha256": files_sha["config.json"], "tokenizer_sha256": files_sha["tokenizer.json"],
            "collector_sha256": source_sha, "coordinate_ids": [f"probe.{i}" for i in range(4)],
            "prompt_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in plan["probes"]],
            "positive_token_id": labels[0], "negative_token_id": labels[1], "max_input_tokens": 64,
        })
        coordinates = torch.cat([torch.eye(4), -torch.eye(4), torch.tensor([
            [0.5,0.5,0.5,0.5], [-0.5,0.5,-0.5,0.5], [0.5,-0.5,-0.5,0.5], [-0.5,-0.5,0.5,0.5]
        ])]).double()
        observation_refs, control_changes = [], []
        try:
            for index, coordinate in enumerate(coordinates):
                with torch.no_grad():
                    norm_layer.weight.copy_(original + (coordinate @ axes.double()).float())
                changed = margins(model, tokenizer, all_prompts[:8], labels) - baseline[:8]
                control_changes.append(changed[4:])
                observation_refs.append(put(root, {
                    "schema": "cerebro.tidex.receiver_response_observation/v1",
                    "observation_id": f"native-norm-{index:03}",
                    "capability_id": "receiver.system-identification:v1",
                    "basis_sha256": basis["sha256"], "protocol_sha256": protocol["sha256"],
                    "receiver_coordinates": coordinate.tolist(), "values": changed[:4].tolist(),
                }))
                print(f"native calibration {index+1}/{len(coordinates)}", flush=True)
        finally:
            with torch.no_grad():
                norm_layer.weight.copy_(original)
        if not torch.equal(norm_layer.weight, original):
            raise RuntimeError("temporary norm interventions were not restored")
        # Independent neutral prompts measure coordinate risk. This metric is
        # local logit-response sensitivity, not global language damage.
        controls = torch.stack(control_changes)
        jacobian = (controls[:4] - controls[4:8]).T / 2.0
        metric = jacobian.T @ jacobian / 4.0 + torch.eye(4).double() * 1e-10
        safety_evidence = put(root, {
            "method": "central_difference_neutral_logit_margin_metric",
            "control_prompt_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in plan["risk_controls"]],
            "control_changes": controls.tolist(), "coordinate_jacobian": jacobian.tolist(),
            "global_language_safety_established": False,
        })
        del model, tokenizer, original, hidden, normalized, head_difference
        gc.collect()
        target = put(root, {"schema": "cerebro.tidex.functional_response_target/v1", "capability_id": "requested.margin-profile:v1", "protocol_sha256": protocol["sha256"], "values": plan["requested_response"]})
        wrong = put(root, {"schema": "cerebro.tidex.functional_response_target/v1", "capability_id": "wrong.margin-profile:v1", "protocol_sha256": protocol["sha256"], "values": [-v for v in plan["requested_response"]]})
        request = {
            "schema": "cerebro.tidex.receiver_weight_request/v1", "basis": basis, "protocol": protocol,
            "target": target, "observations": observation_refs, "wrong_targets": [wrong],
            "safety": {"protected_cortex": {"parameter_importance": metric.diag().tolist(), "directions": [], "max_damage_ratio": 0.15}, "risk_metric": metric.tolist(), "evidence": safety_evidence},
            "policy": plan["policy"],
        }
        return compile_and_replay(root, binary, base_dir, plan, request, baseline, labels,
            files_sha, source_sha, binary_sha, threads, "initial", root)


def compile_and_replay(
    root: Path, binary: Path, base_dir: Path, plan: dict[str, Any],
    request: dict[str, Any], baseline: torch.Tensor, labels: list[int],
    files_sha: dict[str, str], source_sha: str, binary_sha: str,
    threads: int, round_label: str, round_dir: Path,
) -> dict[str, Any]:
    base_file = base_dir / "model.safetensors"
    all_prompts = plan["probes"] + plan["risk_controls"] + plan["unseen_controls"]
    target = request["target"]
    wrong = request["wrong_targets"][0]
    results: dict[str, Any] = {}
    for arm, target_ref, negative_ref in (("correct", target, wrong), ("wrong", wrong, target)):
        arm_request = {**request, "target": target_ref, "wrong_targets": [negative_ref]}
        request_ref = put(root, arm_request)
        request_path = reference_file(root, f"{round_label}-request-{arm}", request_ref)
        candidate_ref = tidex(binary, root, "compile", str(request_path))
        candidate_path = reference_file(root, f"{round_label}-candidate-{arm}", candidate_ref)
        candidate = tidex(binary, root, "inspect", str(candidate_path))
        results[arm] = {"candidate": candidate_ref, "numerical": candidate["numerical"], "blockers": candidate["blockers"]}
        if candidate["blockers"]:
            return {"schema": SCHEMA, "complete": True, "pass": False, "stage": "compiler_rejected", "arms": results, "new_semantic_capability_transfer_established": False}
        model_dir = round_dir / "models" / arm
        model_dir.mkdir(parents=True, mode=0o700)
        checkpoint = tidex(binary, root, "materialize", str(candidate_path), "--base-model", str(base_file), "--output", str(model_dir / "model.safetensors"))
        for name in MODEL_FILES:
            if sha_file(base_dir / name) != files_sha[name]:
                raise RuntimeError("source tokenizer/config changed")
            shutil.copyfile(base_dir / name, model_dir / name)
            (model_dir / name).chmod(0o600)
        package = round_dir / f"replay-package-{arm}.json"
        write_new(package, canonical({"checkpoint_sha256": checkpoint["materialization"]["output_model_sha256"], "model_files_sha256": files_sha, "prompts": all_prompts, "label_ids": labels}))
        replay_path = round_dir / f"replay-{arm}.json"
        subprocess.run([sys.executable, str(SOURCE), "--replay-model", str(model_dir), "--package", str(package), "--output", str(replay_path), "--threads", str(threads)], check=True, timeout=240)
        replay_values = torch.tensor(json.loads(replay_path.read_bytes())["margins"], dtype=torch.float64)
        results[arm].update({"checkpoint": checkpoint, "actual_response_change": (replay_values-baseline).tolist(), "replay_sha256": sha_file(replay_path)})
        print(f"{arm} checkpoint evaluated in fresh process", flush=True)
    requested = torch.tensor(plan["requested_response"], dtype=torch.float64)
    correct_change = torch.tensor(results["correct"]["actual_response_change"], dtype=torch.float64)
    wrong_change = torch.tensor(results["wrong"]["actual_response_change"], dtype=torch.float64)
    correct_error = float(torch.linalg.vector_norm(correct_change[:4]-requested)/torch.linalg.vector_norm(requested))
    wrong_error = float(torch.linalg.vector_norm(wrong_change[:4]-requested)/torch.linalg.vector_norm(requested))
    ratio = wrong_error / max(correct_error, 1e-15)
    unseen_rms = float(correct_change[8:].square().mean().sqrt())
    criteria = plan["criteria"]
    passed = correct_error <= criteria["maximum_actual_target_relative_error"] and ratio >= criteria["minimum_correct_wrong_error_ratio"] and unseen_rms <= criteria["maximum_unseen_control_margin_rms_change"]
    if sha_file(SOURCE) != source_sha or sha_file(binary) != binary_sha or sha_file(base_file) != BASE_SHA:
        raise RuntimeError("source, compiler or original checkpoint changed during experiment")
    return {"schema": SCHEMA, "complete": True, "pass": passed, "stage": "standalone_forward_evaluated", "scope": plan["purpose"], "precommit_sha256": sha_file(round_dir/"precommit.json"), "collector_sha256": source_sha, "tidex_binary_sha256": binary_sha, "baseline_margins": baseline.tolist(), "requested_response": plan["requested_response"], "actual_target_relative_error": correct_error, "wrong_target_relative_error": wrong_error, "correct_wrong_error_ratio": ratio, "unseen_control_margin_rms_change": unseen_rms, "criteria": criteria, "arms": results, "claim_boundary": {"rust_generated_delta_from_measured_responses": True, "lora_delta_source_used": False, "receiver_optimizer_steps": 0, "backpropagation_used": False, "fresh_process_checkpoint_execution": True, "donor_model_used": False, "new_semantic_capability_transfer_established": False, "mbpp_transfer_established": False, "general_language_preservation_established": False, "authorizes_promotion": False}}


def reuse_calibration(base_dir: Path, binary: Path, root: Path, threads: int, round_label: str) -> dict[str, Any]:
    """Revise the compiler method, not the recorded target or acceptance gates.

    The rejected attempt remains immutable. This is not independent replication:
    it deliberately reuses the same twelve response measurements and reports
    the old receipt, old request, and the new compiled implementation identities.
    """
    if not round_label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in round_label):
        raise RuntimeError("round label must be a nonempty lowercase identifier")
    round_dir = root / "method-revisions" / round_label
    if round_dir.exists():
        raise RuntimeError("method revision already exists; refusing to rerun it")
    old_precommit = json.loads((root / "precommit.json").read_bytes())
    original_request_path, original_request_ref = load_unique_reference_alias(
        root,
        ("initial-request-correct.json", "request-correct.json"),
    )
    request = read_reference(original_request_ref)
    protocol = read_reference(request["protocol"])
    target = read_reference(request["target"])
    old_receipt = json.loads((root / "receipt.json").read_bytes())
    if old_receipt.get("stage") != "compiler_rejected" or old_receipt.get("pass") is not False:
        raise RuntimeError("reuse mode requires an explicitly rejected numerical round")
    if request["policy"] != old_precommit["policy"] or target["values"] != old_precommit["requested_response"]:
        raise RuntimeError("original target or numerical policy changed")
    if sha_file(base_dir / "model.safetensors") != old_precommit["base_model_sha256"]:
        raise RuntimeError("base checkpoint changed")
    for name, expected in old_precommit["model_files_sha256"].items():
        if sha_file(base_dir / name) != expected:
            raise RuntimeError("model config/tokenizer changed")
    for reference in request["observations"]:
        read_reference(reference)
    verify_original_implementation_binding(root, old_precommit)
    source_sha, binary_sha = sha_file(SOURCE), sha_file(binary)
    round_dir.mkdir(parents=True, mode=0o700)
    current_reproducibility_bundle = freeze_reproducibility_bundle(
        round_dir / "reproducibility", binary
    )
    write_new(round_dir / "precommit.json", canonical({
        **old_precommit,
        "original_request": original_request_ref,
        "original_request_reference_name": original_request_path.name,
        "original_rejected_receipt_sha256": sha_file(root / "receipt.json"),
        "original_precommit_sha256": sha_file(root / "precommit.json"),
        "measurement_collector_sha256": old_precommit["collector_sha256"],
        "collector_sha256": source_sha, "tidex_binary_sha256": binary_sha,
        "reproducibility_bundle": current_reproducibility_bundle,
        "proposal_method": "fit_protected_coordinates",
        "same_target_same_policy_same_safety_inputs": True,
        "independent_replication": False,
    }))
    # No training or intervention collection is repeated. Only the base
    # forward baseline and standalone output evaluations are executed here.
    model, tokenizer = load_model(base_dir, threads)
    labels = [protocol["positive_token_id"], protocol["negative_token_id"]]
    all_prompts = old_precommit["probes"] + old_precommit["risk_controls"] + old_precommit["unseen_controls"]
    baseline = margins(model, tokenizer, all_prompts, labels)
    del model, tokenizer
    gc.collect()
    result = compile_and_replay(root, binary, base_dir, old_precommit, request, baseline, labels,
        old_precommit["model_files_sha256"], source_sha, binary_sha, threads, round_label, round_dir)
    result["reused_calibration_observation_count"] = len(request["observations"])
    result["same_target_same_policy_same_safety_inputs"] = True
    result["independent_replication"] = False
    write_new(round_dir / "receipt.json", canonical(result))
    return result


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--tidex", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--round-label", default="safe-inverse")
    parser.add_argument("--replay-model", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 8:
        parser.error("threads must be 1..8")
    if args.replay_model:
        if not args.package or not args.output:
            parser.error("replay requires package and output")
        replay(args.replay_model, args.package, args.output, args.threads)
        return 0
    if not args.base_dir or not args.tidex or not args.root:
        parser.error("run requires base-dir, tidex and root")
    if args.reuse_calibration:
        result = reuse_calibration(args.base_dir, args.tidex, args.root, args.threads, args.round_label)
        print(json.dumps({k:v for k,v in result.items() if k not in {"arms", "baseline_margins"}}, indent=2), flush=True)
        return 0 if result["pass"] else 2
    existed_before = args.root.exists()
    try:
        result = run(args.base_dir, args.tidex, args.root, args.threads)
    except Exception as error:
        if not existed_before and args.root.exists():
            write_new(args.root / "failure.json", canonical({"schema": SCHEMA, "complete": False, "error": str(error)}))
        raise
    write_new(args.root / "receipt.json", canonical(result))
    print(json.dumps({k:v for k,v in result.items() if k not in {"arms", "baseline_margins"}}, indent=2), flush=True)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
