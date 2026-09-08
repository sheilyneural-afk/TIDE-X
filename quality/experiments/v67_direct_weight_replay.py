#!/usr/bin/env python3
"""Strict behavioral replay of a Rust-materialized standalone SmolLM2 checkpoint.

This is an evaluation harness only. It does not construct or train a delta and
never loads PEFT. It compares the standalone checkpoint produced by the Rust
WeightActuator against the already sealed V66 adapter replay on exactly the same
30 historical and 171 fresh MBPP tasks.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quality.experiments.v66_mbpp_capability_extract import (
    SEED,
    canonical_json,
    eligible_task,
    sha256_bytes,
    sha256_file,
)
from quality.experiments.v66_mbpp_receiver_compile import (
    ADAPTER_ARTIFACT_SCHEMA,
    ADAPTER_REPLAY_SCHEMA,
    evaluate_codes,
    generate_codes,
    generic_loss,
)

SCHEMA = "cerebro.tidex.v67_direct_weight_replay/v1"
SMOKE_SCHEMA = "cerebro.tidex.v67_weight_actuator_smoke/v1"

if "peft" in sys.modules:
    raise SystemExit("V67 direct replay rejected: PEFT was imported into the standalone evaluator")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"V67 direct replay rejected: JSON root is not object: {path}")
    return value


def result_vector(evaluation: dict[str, Any], field: str) -> list[Any]:
    rows = evaluation.get("results")
    if not isinstance(rows, list):
        raise SystemExit("V67 direct replay rejected: evaluation results missing")
    return [(int(row["task_id"]), row.get(field)) for row in rows]


def compare_evaluations(reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    ref_results = reference.get("results")
    obs_results = observed.get("results")
    if not isinstance(ref_results, list) or not isinstance(obs_results, list):
        raise SystemExit("V67 direct replay rejected: malformed evaluation")
    if [int(row["task_id"]) for row in ref_results] != [int(row["task_id"]) for row in obs_results]:
        raise SystemExit("V67 direct replay rejected: task identity/order mismatch")
    code_mismatches = []
    pass_mismatches = []
    safe_mismatches = []
    for ref, obs in zip(ref_results, obs_results, strict=True):
        task_id = int(ref["task_id"])
        if ref.get("code_sha256") != obs.get("code_sha256"):
            code_mismatches.append(task_id)
        if bool(ref.get("pass")) != bool(obs.get("pass")):
            pass_mismatches.append(task_id)
        if bool(ref.get("safe_ast")) != bool(obs.get("safe_ast")):
            safe_mismatches.append(task_id)
    return {
        "reference_evaluation_sha256": sha256_bytes(canonical_json(reference)),
        "observed_evaluation_sha256": sha256_bytes(canonical_json(observed)),
        "exact_evaluation": canonical_json(reference) == canonical_json(observed),
        "code_hash_exact_count": len(ref_results) - len(code_mismatches),
        "code_hash_mismatch_task_ids": code_mismatches,
        "pass_vector_exact": not pass_mismatches,
        "pass_mismatch_task_ids": pass_mismatches,
        "safe_ast_vector_exact": not safe_mismatches,
        "safe_ast_mismatch_task_ids": safe_mismatches,
    }


def load_direct_model(model_dir: Path) -> tuple[Any, Any]:
    forbidden = [
        path.name
        for path in model_dir.iterdir()
        if path.is_file() and ("adapter" in path.name.lower() or "lora" in path.name.lower())
    ]
    if forbidden:
        raise SystemExit(f"V67 direct replay rejected: adapter artifacts present: {forbidden}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.config.use_cache = False
    model.requires_grad_(False)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--v66-manifest", type=Path, required=True)
    parser.add_argument("--v66-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("source", "full"), default="source")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("V67 direct replay rejected: output already exists")
    model_file = args.model_dir / "model.safetensors"
    if not model_file.is_file() or model_file.is_symlink():
        # A hard link is a regular file and is accepted; symlinks are not.
        raise SystemExit("V67 direct replay rejected: standalone model file missing or symlink")

    smoke = load_json(args.smoke_receipt)
    manifest = load_json(args.v66_manifest)
    v66_replay = load_json(args.v66_replay)
    if smoke.get("schema") != SMOKE_SCHEMA:
        raise SystemExit("V67 direct replay rejected: smoke receipt schema mismatch")
    if manifest.get("schema") != ADAPTER_ARTIFACT_SCHEMA:
        raise SystemExit("V67 direct replay rejected: V66 manifest schema mismatch")
    if v66_replay.get("schema") != ADAPTER_REPLAY_SCHEMA or v66_replay.get("pass") is not True:
        raise SystemExit("V67 direct replay rejected: V66 replay authority invalid")
    expected_model_sha = smoke.get("materialization", {}).get("output_model_sha256")
    observed_model_sha = sha256_file(model_file)
    if observed_model_sha != expected_model_sha:
        raise SystemExit("V67 direct replay rejected: standalone model digest mismatch")
    if smoke.get("claim_boundary", {}).get("functional_receiver_compiler_generated_target_delta") is not False:
        raise SystemExit("V67 direct replay rejected: smoke claim boundary changed")
    if smoke.get("materialization", {}).get("requires_adapter_at_runtime") is not False:
        raise SystemExit("V67 direct replay rejected: output still declares adapter runtime dependency")

    source_behavior = manifest.get("source_behavior")
    if not isinstance(source_behavior, dict):
        raise SystemExit("V67 direct replay rejected: V66 source behavior missing")
    source_ids = [int(value) for value in source_behavior.get("heldout_task_ids", [])]
    if len(source_ids) != 30 or len(set(source_ids)) != 30:
        raise SystemExit("V67 direct replay rejected: historical source identity mismatch")

    ref_source = v66_replay.get("source_behavior_replay", {}).get("evaluation")
    if not isinstance(ref_source, dict):
        raise SystemExit("V67 direct replay rejected: V66 source reference evaluation missing")

    fresh_authority: dict[str, Any] | None = None
    fresh_ids: list[int] = []
    ref_fresh: dict[str, Any] | None = None
    if args.scope == "full":
        candidate = v66_replay.get("fresh_evaluation")
        if not isinstance(candidate, dict):
            raise SystemExit("V67 direct replay rejected: V66 fresh authority missing")
        fresh_authority = candidate
        fresh_ids = [int(value) for value in fresh_authority.get("task_ids", [])]
        if len(fresh_ids) != 171 or len(set(fresh_ids)) != 171 or set(source_ids) & set(fresh_ids):
            raise SystemExit("V67 direct replay rejected: fresh task identity mismatch")
        candidate_ref = fresh_authority.get("compiled_adapter")
        if not isinstance(candidate_ref, dict):
            raise SystemExit("V67 direct replay rejected: V66 fresh reference evaluation missing")
        ref_fresh = candidate_ref

    dataset = load_dataset("mbpp", download_mode="reuse_dataset_if_exists")
    test_by_id = {int(row["task_id"]): row for row in dataset["test"]}
    source_rows = [test_by_id.get(task_id) for task_id in source_ids]
    fresh_rows = [test_by_id.get(task_id) for task_id in fresh_ids]
    if any(row is None or not eligible_task(row) for row in [*source_rows, *fresh_rows]):
        raise SystemExit("V67 direct replay rejected: task no longer resolves to eligible MBPP row")

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    model, tokenizer = load_direct_model(args.model_dir)
    direct_source = evaluate_codes(source_rows, generate_codes(model, tokenizer, source_rows))
    direct_fresh: dict[str, Any] | None = None
    direct_generic_loss: float | None = None
    if args.scope == "full":
        direct_generic_loss = generic_loss(model, tokenizer)
        direct_fresh = evaluate_codes(fresh_rows, generate_codes(model, tokenizer, fresh_rows))
    del model
    del tokenizer
    gc.collect()

    source_comparison = compare_evaluations(ref_source, direct_source)
    expected_historical_sha = source_behavior.get("compiled_evaluation_sha256")
    historical_exact = (
        source_comparison["observed_evaluation_sha256"] == expected_historical_sha
        and source_comparison["exact_evaluation"]
    )
    fresh_comparison: dict[str, Any] | None = None
    strict_equivalence = historical_exact
    if args.scope == "full":
        assert ref_fresh is not None and direct_fresh is not None
        fresh_comparison = compare_evaluations(ref_fresh, direct_fresh)
        strict_equivalence = historical_exact and fresh_comparison["exact_evaluation"]

    receipt = {
        "schema": SCHEMA,
        "complete": True,
        "pass": strict_equivalence,
        "scope": args.scope,
        "method": "load the Rust-materialized standalone SmolLM2 checkpoint directly with Transformers; no PEFT adapter, donor, CapabilityIR, or retraining; compare deterministic MBPP evaluations byte-semantically against sealed V66 adapter replay",
        "model": {
            "path": str(args.model_dir),
            "model_safetensors_sha256": observed_model_sha,
            "adapter_files_present": False,
            "peft_loaded": "peft" in sys.modules,
        },
        "source_behavior": {
            "task_count": len(source_rows),
            "expected_historical_evaluation_sha256": expected_historical_sha,
            "historical_exact": historical_exact,
            "comparison": source_comparison,
            "direct_evaluation": direct_source,
        },
        "fresh_behavior": None
        if args.scope == "source"
        else {
            "task_count": len(fresh_rows),
            "comparison": fresh_comparison,
            "reference_adapter_pass_rate": ref_fresh["pass_rate"] if ref_fresh else None,
            "direct_weight_pass_rate": direct_fresh["pass_rate"] if direct_fresh else None,
            "direct_evaluation": direct_fresh,
        },
        "generic_text_probe": None
        if args.scope == "source"
        else {
            "direct_weight_loss": direct_generic_loss,
            "reference_adapter_loss": v66_replay.get("generic_text_probe", {}).get("compiled_loss"),
        },
        "claim_boundary": {
            "rust_direct_weight_actuation_historical_behavior_exact": historical_exact,
            "full_201_task_behavioral_equivalence_established": strict_equivalence
            if args.scope == "full"
            else False,
            "functional_receiver_compiler_generated_target_delta": False,
            "target_capability_training_avoided_in_delta_source": False,
            "standalone_runtime_requires_peft": False,
            "universal_portability_established": False,
        },
    }
    args.output.write_bytes(canonical_json(receipt))
    print(
        json.dumps(
            {
                "pass": receipt["pass"],
                "scope": args.scope,
                "source": source_comparison,
                "fresh": fresh_comparison,
                "direct_source_pass_rate": direct_source["pass_rate"],
                "direct_fresh_pass_rate": direct_fresh["pass_rate"] if direct_fresh else None,
                "direct_generic_loss": direct_generic_loss,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
