#!/usr/bin/env python3
"""V66 phase 2: receiver-native functional capability compilation on SmolLM2.

The compilation route receives only the phase-1 CapabilityIR: canonical task
prompts and donor implementations that independently passed held-out tests. It
never receives MBPP reference implementations or hidden-test content. A separate
B-direct oracle is trained from the exact reference implementations for the same
anchor tasks only to normalize recovered gain; it is evaluation-only authority
and never enters the CapabilityIR compilation route.

The negative control uses the same receiver, optimizer, step budget and anchor
prompts, but cyclically shifts donor code across prompts. This tests whether the
correct prompt-to-behavior association matters. It is a mismatched-association
control, not evidence about an unrelated capability family; the serialized v1
receipt keeps the historical key `wrong_ir` for compatibility.

Held-out evaluation is deterministic pass@1 code generation followed by actual
execution of hidden MBPP tests under the restricted AST/evaluator defined by
v66_mbpp_capability_extract.py. The fixed generic-text loss probe contains only 12
texts and must not be described as a general language benchmark.

Evidence scope: one seed, one Qwen2.5-Coder-1.5B -> SmolLM2-1.7B pair and one
MBPP function-synthesis domain. V66 does not establish universal portability,
direct donor-weight translation, or superiority over direct fine-tuning.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
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

from quality.experiments.capability_statistics import exact_paired_binomial_p, paired_pass_counts

from quality.experiments.v66_mbpp_capability_extract import (
    CAPABILITY_SCHEMA,
    ROOT,
    SEED,
    canonical_json,
    complexity_score,
    eligible_task,
    extract_python,
    run_restricted_tests,
    sha256_bytes,
    sha256_file,
    task_prompt,
)

RECEIVER_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
RECEIVER_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots"
    / RECEIVER_REVISION
)
RECEIVER_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RECEIVER_SCHEMA = "cerebro.tidex.v66_receiver_code_capability/v1"
ADAPTER_ARTIFACT_SCHEMA = "cerebro.tidex.v66_compiled_adapter_artifact/v1"
ADAPTER_REPLAY_SCHEMA = "cerebro.tidex.v66_compiled_adapter_replay/v1"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGETS = ["q_proj", "v_proj"]
TRAIN_STEPS = 120
TRAIN_BATCH = 2
LEARNING_RATE = 8e-4
MAX_TRAIN_TOKENS = 384
EVALUATION_BATCH = 4
MAX_NEW_TOKENS = 160

GENERAL_TEXTS = [
    "The water cycle moves moisture through evaporation, condensation, and precipitation.",
    "A careful experiment changes one variable while keeping the remaining conditions controlled.",
    "The train arrived at the station shortly after sunset and the passengers left quietly.",
    "A database index can accelerate lookups while adding storage and update costs.",
    "Economic choices often involve tradeoffs between present costs and future benefits.",
    "The historian compared several independent sources before accepting the chronology.",
    "Plants use light energy to convert carbon dioxide and water into chemical energy.",
    "Good software interfaces make invalid states difficult to represent and easy to detect.",
    "The committee reviewed the evidence and published a report explaining its conclusions.",
    "A map is useful only when its scale and coordinate system match the intended task.",
    "Reliable measurements include both a value and an account of the uncertainty around it.",
    "The novel follows several characters whose decisions interact across many years.",
]


def outside_checkout(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    raise SystemExit("V66 receiver rejected: generated evidence must remain outside checkout")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit("V66 receiver rejected: JSON root must be an object")
    return value


def receiver_weight_file() -> Path:
    candidate = RECEIVER_PATH / "model.safetensors"
    if not candidate.is_file():
        raise SystemExit("V66 receiver rejected: exact receiver snapshot is unavailable")
    return candidate


def setup_receiver_backbone(seed: int) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(RECEIVER_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        RECEIVER_PATH,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.config.use_cache = False
    return model, tokenizer


def setup_receiver(seed: int) -> tuple[Any, Any]:
    from peft import LoraConfig, TaskType, get_peft_model

    model, tokenizer = setup_receiver_backbone(seed)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            bias="none",
        ),
    )
    forbidden_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if forbidden_trainable:
        raise SystemExit("V66 receiver rejected: base-model parameter became trainable")
    model.config.use_cache = False
    return model, tokenizer


def adapter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    observed = 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        observed += parameter.numel()
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    if observed == 0:
        raise SystemExit("V66 receiver rejected: adapter parameter set is empty")
    return digest.hexdigest()


def trainable_parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def encode_training_example(tokenizer: Any, prompt: str, code: str) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion = code.rstrip() + "\n"
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) >= MAX_TRAIN_TOKENS - 16:
        raise SystemExit("V66 receiver rejected: prompt exceeds training budget")
    remaining = MAX_TRAIN_TOKENS - len(prompt_ids)
    completion_ids = completion_ids[:remaining]
    if len(completion_ids) < 4:
        raise SystemExit("V66 receiver rejected: completion collapsed under token budget")
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return input_ids, labels


def batch_tensor(tokenizer: Any, examples: list[tuple[str, str]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rows = [encode_training_example(tokenizer, prompt, code) for prompt, code in examples]
    maximum = max(len(input_ids) for input_ids, _ in rows)
    pad = tokenizer.pad_token_id
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for input_ids, labels in rows:
        missing = maximum - len(input_ids)
        input_rows.append(input_ids + [pad] * missing)
        label_rows.append(labels + [-100] * missing)
        attention_rows.append([1] * len(input_ids) + [0] * missing)
    encoded = {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
    }
    return encoded, torch.tensor(label_rows, dtype=torch.long)


def train_adapter(model: Any, tokenizer: Any, examples: list[tuple[str, str]], seed: int) -> None:
    if len(examples) < 20:
        raise SystemExit("V66 receiver rejected: insufficient compilation anchors")
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    model.train()
    for step in range(TRAIN_STEPS):
        selected = [examples[rng.randrange(len(examples))] for _ in range(TRAIN_BATCH)]
        encoded, labels = batch_tensor(tokenizer, selected)
        optimizer.zero_grad(set_to_none=True)
        output = model(**encoded, labels=labels)
        if not torch.isfinite(output.loss):
            raise SystemExit("V66 receiver rejected: non-finite training loss")
        output.loss.backward()
        optimizer.step()
        if (step + 1) % 30 == 0:
            print(f"receiver training step={step+1}/{TRAIN_STEPS} loss={float(output.loss):.6f}", flush=True)


def generic_loss(model: Any, tokenizer: Any) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for text in GENERAL_TEXTS:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            output = model(**encoded, labels=encoded["input_ids"])
            losses.append(float(output.loss))
    return sum(losses) / len(losses)


def generation_prompts(rows: list[dict[str, Any]]) -> list[str]:
    return [task_prompt(row) for row in rows]


def generate_codes(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> list[str]:
    tokenizer.padding_side = "left"
    model.eval()
    prompts = generation_prompts(rows)
    codes: list[str] = []
    with torch.no_grad():
        for start in range(0, len(prompts), EVALUATION_BATCH):
            chunk = prompts[start : start + EVALUATION_BATCH]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prefix_length = encoded["input_ids"].shape[1]
            for sequence in generated:
                raw = tokenizer.decode(sequence[prefix_length:], skip_special_tokens=True)
                codes.append(extract_python(raw))
            print(f"receiver evaluated {min(start + len(chunk), len(prompts))}/{len(prompts)}", flush=True)
    tokenizer.padding_side = "right"
    return codes


def evaluate_codes(rows: list[dict[str, Any]], codes: list[str]) -> dict[str, Any]:
    if len(rows) != len(codes):
        raise SystemExit("V66 receiver rejected: evaluation result count mismatch")
    results = []
    for row, code in zip(rows, codes, strict=True):
        safe = bool(code)
        passed = safe and run_restricted_tests(code, row["test_list"][1:])
        results.append(
            {
                "task_id": int(row["task_id"]),
                "pass": passed,
                "safe_ast": safe,
                "code_sha256": sha256_bytes(code.encode()) if code else None,
                "complexity_score": complexity_score(row),
            }
        )
    return {
        "pass_rate": sum(item["pass"] for item in results) / len(results),
        "safe_ast_rate": sum(item["safe_ast"] for item in results) / len(results),
        "results": results,
    }


def release_model(model: Any, tokenizer: Any) -> None:
    del model
    del tokenizer
    gc.collect()


def write_progress(path: Path, stage: str, values: dict[str, Any]) -> None:
    record = {
        "schema": RECEIVER_SCHEMA,
        "complete": False,
        "stage": stage,
        **values,
    }
    path.write_bytes(canonical_json(record))


def load_authorities(ir_path: Path, donor_receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any, list[dict[str, Any]], list[dict[str, Any]]]:
    capability_ir = load_json(ir_path)
    donor_receipt = load_json(donor_receipt_path)
    if donor_receipt.get("pass") is not True:
        raise SystemExit("V66 receiver rejected: donor receipt did not pass")
    if donor_receipt.get("capability_ir_sha256") != sha256_file(ir_path):
        raise SystemExit("V66 receiver rejected: donor receipt/IR digest mismatch")
    if capability_ir.get("schema") != CAPABILITY_SCHEMA:
        raise SystemExit("V66 receiver rejected: CapabilityIR schema mismatch")
    if capability_ir.get("original_reference_code_present") is not False:
        raise SystemExit("V66 receiver rejected: CapabilityIR contains reference-code claim")
    if capability_ir.get("hidden_test_content_present") is not False:
        raise SystemExit("V66 receiver rejected: CapabilityIR contains hidden-test claim")
    anchors = capability_ir.get("anchors")
    if not isinstance(anchors, list) or len(anchors) < 40:
        raise SystemExit("V66 receiver rejected: CapabilityIR anchor count invalid")

    dataset = load_dataset("mbpp", download_mode="reuse_dataset_if_exists")
    all_rows = {
        int(row["task_id"]): row
        for split in ("train", "validation", "test")
        for row in dataset[split]
    }
    anchor_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        task_id = int(anchor["task_id"])
        row = all_rows.get(task_id)
        if row is None or not eligible_task(row):
            raise SystemExit("V66 receiver rejected: anchor no longer resolves to eligible MBPP task")
        prompt = task_prompt(row)
        if anchor.get("prompt") != prompt or anchor.get("prompt_sha256") != sha256_bytes(prompt.encode()):
            raise SystemExit("V66 receiver rejected: anchor prompt binding mismatch")
        code = anchor.get("donor_code")
        if not isinstance(code, str) or anchor.get("donor_code_sha256") != sha256_bytes(code.encode()):
            raise SystemExit("V66 receiver rejected: donor code binding mismatch")
        if not run_restricted_tests(code, row["test_list"][1:]):
            raise SystemExit("V66 receiver rejected: donor anchor failed re-verification")
        if anchor.get("hidden_test_commitment_sha256") != sha256_bytes(canonical_json(row["test_list"][1:])):
            raise SystemExit("V66 receiver rejected: hidden-test commitment mismatch")
        anchor_rows.append(row)

    evaluation_ids = [int(item["task_id"]) for item in donor_receipt.get("evaluation", [])]
    if len(evaluation_ids) != int(donor_receipt.get("evaluation_count", -1)) or not evaluation_ids:
        raise SystemExit("V66 receiver rejected: donor evaluation identity invalid")
    if set(evaluation_ids) & {int(anchor["task_id"]) for anchor in anchors}:
        raise SystemExit("V66 receiver rejected: anchor/evaluation overlap")
    evaluation_rows = [all_rows[task_id] for task_id in evaluation_ids]
    if any(not eligible_task(row) for row in evaluation_rows):
        raise SystemExit("V66 receiver rejected: held-out evaluation task no longer eligible")
    return capability_ir, donor_receipt, dataset, anchor_rows, evaluation_rows


def mismatched_control_examples_from_ir(
    capability_ir: dict[str, Any],
) -> list[tuple[str, str]]:
    compiled_examples = [
        (anchor["prompt"], anchor["donor_code"])
        for anchor in capability_ir["anchors"]
    ]
    shifted_codes = [code for _, code in compiled_examples]
    shifted_codes = shifted_codes[1:] + shifted_codes[:1]
    return [
        (prompt, shifted_code)
        for (prompt, _), shifted_code in zip(compiled_examples, shifted_codes, strict=True)
    ]


def validate_progress_evaluation(
    name: str,
    evaluation: Any,
    evaluation_rows: list[dict[str, Any]],
) -> None:
    if not isinstance(evaluation, dict):
        raise SystemExit(f"V66 resume rejected: {name} evaluation missing")
    results = evaluation.get("results")
    if not isinstance(results, list) or len(results) != len(evaluation_rows):
        raise SystemExit(f"V66 resume rejected: {name} result count mismatch")
    expected_ids = [int(row["task_id"]) for row in evaluation_rows]
    observed_ids = [int(item.get("task_id", -1)) for item in results if isinstance(item, dict)]
    if observed_ids != expected_ids:
        raise SystemExit(f"V66 resume rejected: {name} held-out identity mismatch")
    pass_rate = sum(bool(item.get("pass")) for item in results) / len(results)
    safe_ast_rate = sum(bool(item.get("safe_ast")) for item in results) / len(results)
    if not math.isclose(float(evaluation.get("pass_rate", -1.0)), pass_rate, abs_tol=1e-12):
        raise SystemExit(f"V66 resume rejected: {name} pass-rate mismatch")
    if not math.isclose(
        float(evaluation.get("safe_ast_rate", -1.0)), safe_ast_rate, abs_tol=1e-12
    ):
        raise SystemExit(f"V66 resume rejected: {name} safe-AST mismatch")


def load_direct_progress(
    output_path: Path,
    ir_path: Path,
    donor_receipt_path: Path,
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    progress = load_json(output_path)
    if progress.get("schema") != RECEIVER_SCHEMA:
        raise SystemExit("V66 resume rejected: receiver progress schema mismatch")
    if progress.get("complete") is not False or progress.get("stage") != "direct_evaluated":
        raise SystemExit("V66 resume rejected: progress is not direct_evaluated")
    if progress.get("capability_ir_sha256") != sha256_file(ir_path):
        raise SystemExit("V66 resume rejected: CapabilityIR progress binding mismatch")
    if progress.get("donor_receipt_sha256") != sha256_file(donor_receipt_path):
        raise SystemExit("V66 resume rejected: donor receipt progress binding mismatch")
    for name, key in (
        ("virgin", "receiver_virgin"),
        ("compiled", "receiver_compiled"),
        ("direct", "receiver_direct"),
    ):
        validate_progress_evaluation(name, progress.get(key), evaluation_rows)
    for key in (
        "receiver_base_generic_loss",
        "receiver_compiled_generic_loss",
        "receiver_direct_generic_loss",
    ):
        value = progress.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise SystemExit(f"V66 resume rejected: invalid {key}")
    for key in (
        "receiver_compiled_adapter_sha256",
        "receiver_direct_adapter_sha256",
    ):
        value = progress.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise SystemExit(f"V66 resume rejected: invalid {key}")
    return progress


def validate_final_receiver_receipt(
    receipt_path: Path,
    ir_path: Path,
    donor_receipt_path: Path,
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    if (
        receipt.get("schema") != RECEIVER_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("pass") is not True
    ):
        raise SystemExit("V66 adapter materialization rejected: final PASS receipt required")
    capability = receipt.get("capability_ir")
    donor = receipt.get("donor")
    receiver = receipt.get("receiver")
    if not isinstance(capability, dict) or capability.get("sha256") != sha256_file(ir_path):
        raise SystemExit("V66 adapter materialization rejected: CapabilityIR receipt binding mismatch")
    if (
        not isinstance(donor, dict)
        or donor.get("receipt_sha256") != sha256_file(donor_receipt_path)
    ):
        raise SystemExit("V66 adapter materialization rejected: donor receipt binding mismatch")
    if not isinstance(receiver, dict):
        raise SystemExit("V66 adapter materialization rejected: receiver record missing")
    if (
        receiver.get("model_id") != RECEIVER_MODEL_ID
        or receiver.get("revision") != RECEIVER_REVISION
        or receiver.get("weight_file_sha256") != sha256_file(receiver_weight_file())
        or receiver.get("base_parameter_trainable_count") != 0
    ):
        raise SystemExit("V66 adapter materialization rejected: receiver identity mismatch")
    for name in ("virgin", "compiled", "direct", "wrong_ir"):
        validate_progress_evaluation(name, receiver.get(name), evaluation_rows)
    expected_adapter_digest = receiver.get("compiled_adapter_sha256")
    if (
        not isinstance(expected_adapter_digest, str)
        or len(expected_adapter_digest) != 64
        or any(char not in "0123456789abcdef" for char in expected_adapter_digest)
    ):
        raise SystemExit("V66 adapter materialization rejected: compiled adapter digest invalid")
    return receipt


def materialize_compiled_adapter(
    ir_path: Path,
    donor_receipt_path: Path,
    source_receipt_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir = outside_checkout(artifact_dir)
    if artifact_dir.exists():
        raise SystemExit("V66 adapter materialization rejected: artifact directory already exists")
    capability_ir, _donor_receipt, _dataset, _anchor_rows, evaluation_rows = load_authorities(
        ir_path, donor_receipt_path
    )
    source_receipt = validate_final_receiver_receipt(
        source_receipt_path,
        ir_path,
        donor_receipt_path,
        evaluation_rows,
    )
    compiled_examples = [
        (anchor["prompt"], anchor["donor_code"])
        for anchor in capability_ir["anchors"]
    ]
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    compiled, compiled_tokenizer = setup_receiver(SEED + 100)
    expected_adapter_digest = str(source_receipt["receiver"]["compiled_adapter_sha256"])
    compiled_digest: str | None = None
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_dir.name}.tmp-",
            dir=str(artifact_dir.parent),
        )
    )
    try:
        train_adapter(compiled, compiled_tokenizer, compiled_examples, SEED + 110)
        compiled_digest = adapter_digest(compiled)
        if compiled_digest != expected_adapter_digest:
            raise SystemExit(
                "V66 adapter materialization rejected: deterministic adapter digest mismatch"
            )
        compiled.save_pretrained(temp_dir, safe_serialization=True)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        release_model(compiled, compiled_tokenizer)

    required_files = {"adapter_config.json", "adapter_model.safetensors"}
    observed_files = {
        path.name: sha256_file(path)
        for path in sorted(temp_dir.iterdir())
        if path.is_file()
    }
    if not required_files.issubset(observed_files):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise SystemExit("V66 adapter materialization rejected: PEFT artifact incomplete")

    source_compiled_evaluation = source_receipt["receiver"]["compiled"]
    manifest = {
        "schema": ADAPTER_ARTIFACT_SCHEMA,
        "capability": capability_ir["capability"],
        "materialization": "receiver_native_lora",
        "source_v66_receipt_sha256": sha256_file(source_receipt_path),
        "capability_ir_sha256": sha256_file(ir_path),
        "donor_receipt_sha256": sha256_file(donor_receipt_path),
        "receiver": {
            "model_id": RECEIVER_MODEL_ID,
            "revision": RECEIVER_REVISION,
            "architecture": "LlamaForCausalLM",
            "weight_file_sha256": sha256_file(receiver_weight_file()),
            "backbone_frozen_during_materialization": True,
        },
        "training": {
            "global_seed": SEED,
            "adapter_training_seed": SEED + 110,
            "steps": TRAIN_STEPS,
            "batch_size": TRAIN_BATCH,
            "learning_rate": LEARNING_RATE,
            "max_train_tokens": MAX_TRAIN_TOKENS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_target_modules": LORA_TARGETS,
        },
        "adapter": {
            "parameter_digest_sha256": compiled_digest,
            "files_sha256": observed_files,
        },
        "source_behavior": {
            "heldout_task_ids": [int(row["task_id"]) for row in evaluation_rows],
            "compiled_evaluation_sha256": sha256_bytes(
                canonical_json(source_compiled_evaluation)
            ),
            "compiled_pass_rate": source_compiled_evaluation["pass_rate"],
        },
        "claim_boundary": {
            "replay_requires_capability_ir": False,
            "replay_requires_donor_model": False,
            "replay_requires_retraining": False,
            "universal_portability_established": False,
            "direct_weight_translation_established": False,
        },
    }
    manifest_bytes = canonical_json(manifest)
    (temp_dir / "manifest.json").write_bytes(manifest_bytes)
    manifest_sha = sha256_bytes(manifest_bytes)
    (temp_dir / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n",
        encoding="utf-8",
    )
    try:
        os.replace(temp_dir, artifact_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def load_adapter_artifact_manifest(artifact_dir: Path) -> tuple[Path, dict[str, Any], str]:
    artifact_dir = outside_checkout(artifact_dir)
    if not artifact_dir.is_dir():
        raise SystemExit("V66 adapter replay rejected: artifact directory missing")
    manifest_path = artifact_dir / "manifest.json"
    manifest_sha_path = artifact_dir / "manifest.sha256"
    if (
        manifest_path.is_symlink()
        or manifest_sha_path.is_symlink()
        or not manifest_path.is_file()
        or not manifest_sha_path.is_file()
    ):
        raise SystemExit("V66 adapter replay rejected: manifest files missing or non-regular")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    declared_sha = manifest_sha_path.read_text(encoding="utf-8").strip().split()
    if len(declared_sha) != 2 or declared_sha[0] != manifest_sha or declared_sha[1] != "manifest.json":
        raise SystemExit("V66 adapter replay rejected: manifest digest mismatch")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != ADAPTER_ARTIFACT_SCHEMA:
        raise SystemExit("V66 adapter replay rejected: artifact schema mismatch")
    receiver = manifest.get("receiver")
    adapter = manifest.get("adapter")
    source_behavior = manifest.get("source_behavior")
    if not isinstance(receiver, dict) or not isinstance(adapter, dict) or not isinstance(source_behavior, dict):
        raise SystemExit("V66 adapter replay rejected: artifact manifest incomplete")
    if (
        receiver.get("model_id") != RECEIVER_MODEL_ID
        or receiver.get("revision") != RECEIVER_REVISION
        or receiver.get("weight_file_sha256") != sha256_file(receiver_weight_file())
    ):
        raise SystemExit("V66 adapter replay rejected: receiver snapshot mismatch")
    files_sha256 = adapter.get("files_sha256")
    if not isinstance(files_sha256, dict) or not files_sha256:
        raise SystemExit("V66 adapter replay rejected: adapter file manifest missing")
    observed_artifact_files = {
        path.name
        for path in artifact_dir.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }
    if observed_artifact_files != set(files_sha256):
        raise SystemExit("V66 adapter replay rejected: undeclared or missing adapter file")
    for name, expected in files_sha256.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(expected, str):
            raise SystemExit("V66 adapter replay rejected: adapter file manifest invalid")
        path = artifact_dir / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise SystemExit("V66 adapter replay rejected: adapter file digest mismatch")
    if not (artifact_dir / "adapter_model.safetensors").is_file() or not (
        artifact_dir / "adapter_config.json"
    ).is_file():
        raise SystemExit("V66 adapter replay rejected: required PEFT files missing")
    return artifact_dir, manifest, manifest_sha


def load_materialized_adapter(artifact_dir: Path, manifest: dict[str, Any]) -> tuple[Any, Any]:
    from peft import PeftModel

    base, tokenizer = setup_receiver_backbone(SEED + 100)
    model = PeftModel.from_pretrained(base, artifact_dir, is_trainable=False)
    model.config.use_cache = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        release_model(model, tokenizer)
        raise SystemExit("V66 adapter replay rejected: loaded artifact became trainable")
    observed_digest = adapter_digest(model)
    expected_digest = manifest["adapter"].get("parameter_digest_sha256")
    if observed_digest != expected_digest:
        release_model(model, tokenizer)
        raise SystemExit("V66 adapter replay rejected: loaded adapter parameter digest mismatch")
    return model, tokenizer


def replay_compiled_adapter(
    artifact_dir: Path,
    output_path: Path,
    replay_count: int,
) -> dict[str, Any]:
    output_path = outside_checkout(output_path)
    artifact_dir, manifest, manifest_sha = load_adapter_artifact_manifest(artifact_dir)
    source_behavior = manifest["source_behavior"]
    source_task_ids = [int(value) for value in source_behavior.get("heldout_task_ids", [])]
    if not source_task_ids or len(source_task_ids) != len(set(source_task_ids)):
        raise SystemExit("V66 adapter replay rejected: source held-out identity invalid")
    if replay_count < 0:
        raise SystemExit("V66 adapter replay rejected: replay count must be >= 0")

    dataset = load_dataset("mbpp", download_mode="reuse_dataset_if_exists")
    test_rows = {int(row["task_id"]): row for row in dataset["test"]}
    source_rows = [test_rows.get(task_id) for task_id in source_task_ids]
    if any(row is None or not eligible_task(row) for row in source_rows):
        raise SystemExit("V66 adapter replay rejected: source held-out task no longer eligible")
    excluded = set(source_task_ids)
    fresh_rows = [
        row
        for row in dataset["test"]
        if int(row["task_id"]) not in excluded and eligible_task(row)
    ]
    fresh_rows.sort(key=lambda row: int(row["task_id"]))
    available_fresh_count = len(fresh_rows)
    if replay_count:
        fresh_rows = fresh_rows[:replay_count]
    if not fresh_rows:
        raise SystemExit("V66 adapter replay rejected: no fresh held-out tasks available")

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    virgin, virgin_tokenizer = setup_receiver_backbone(SEED + 100)
    virgin_generic_loss = generic_loss(virgin, virgin_tokenizer)
    virgin_fresh = evaluate_codes(
        fresh_rows, generate_codes(virgin, virgin_tokenizer, fresh_rows)
    )
    release_model(virgin, virgin_tokenizer)

    compiled, compiled_tokenizer = load_materialized_adapter(artifact_dir, manifest)
    compiled_generic_loss = generic_loss(compiled, compiled_tokenizer)
    source_replay = evaluate_codes(
        source_rows,
        generate_codes(compiled, compiled_tokenizer, source_rows),
    )
    compiled_fresh = evaluate_codes(
        fresh_rows,
        generate_codes(compiled, compiled_tokenizer, fresh_rows),
    )
    release_model(compiled, compiled_tokenizer)

    source_replay_sha = sha256_bytes(canonical_json(source_replay))
    expected_source_sha = source_behavior.get("compiled_evaluation_sha256")
    source_behavior_exact = source_replay_sha == expected_source_sha
    paired = paired_pass_counts(virgin_fresh, compiled_fresh)
    fresh_gain = float(compiled_fresh["pass_rate"]) - float(virgin_fresh["pass_rate"])
    generic_probe_relative_loss_increase = (
        compiled_generic_loss - virgin_generic_loss
    ) / max(virgin_generic_loss, 1e-12)
    criteria = {
        "require_exact_source_behavior_replay": True,
        "minimum_fresh_pass_rate_gain": 0.08,
        "maximum_exact_two_sided_p": 0.05,
        "minimum_compiled_safe_ast_rate": 0.70,
        "maximum_generic_probe_relative_loss_increase": 0.10,
    }
    passed = (
        source_behavior_exact
        and fresh_gain >= criteria["minimum_fresh_pass_rate_gain"]
        and float(paired["exact_two_sided_p"]) <= criteria["maximum_exact_two_sided_p"]
        and float(compiled_fresh["safe_ast_rate"]) >= criteria["minimum_compiled_safe_ast_rate"]
        and generic_probe_relative_loss_increase
        <= criteria["maximum_generic_probe_relative_loss_increase"]
    )
    replay_receipt = {
        "schema": ADAPTER_REPLAY_SCHEMA,
        "complete": True,
        "pass": passed,
        "scope": "fresh-process A/B replay of persisted SmolLM2-1.7B LoRA capability materialization on previously unused eligible MBPP test tasks",
        "method": "load frozen SmolLM2 backbone and sealed PEFT adapter without CapabilityIR, donor model, or retraining; compare against the untouched backbone on identical fresh prompts and hidden tests",
        "adapter_artifact": {
            "path": str(artifact_dir),
            "manifest_sha256": manifest_sha,
            "parameter_digest_sha256": manifest["adapter"]["parameter_digest_sha256"],
            "capability_ir_required_for_replay": False,
            "donor_model_required_for_replay": False,
            "retraining_required_for_replay": False,
        },
        "source_behavior_replay": {
            "task_count": len(source_rows),
            "expected_evaluation_sha256": expected_source_sha,
            "observed_evaluation_sha256": source_replay_sha,
            "exact": source_behavior_exact,
            "evaluation": source_replay,
        },
        "fresh_evaluation": {
            "dataset": "mbpp/full:test",
            "available_eligible_tasks_after_original_v66_exclusion": available_fresh_count,
            "evaluated_count": len(fresh_rows),
            "task_ids": [int(row["task_id"]) for row in fresh_rows],
            "selection": "ascending task_id after excluding all original V66 held-out task ids",
            "virgin": virgin_fresh,
            "compiled_adapter": compiled_fresh,
            "paired": paired,
        },
        "generic_text_probe": {
            "text_count": len(GENERAL_TEXTS),
            "virgin_loss": virgin_generic_loss,
            "compiled_loss": compiled_generic_loss,
            "relative_loss_increase": generic_probe_relative_loss_increase,
            "interpretation": "fixed 12-text loss probe only; not a global language benchmark",
        },
        "metrics": {
            "fresh_pass_rate_gain": fresh_gain,
            "recovered_behavior_replay_exact": source_behavior_exact,
            "exact_two_sided_p": paired["exact_two_sided_p"],
        },
        "criteria": criteria,
        "limits": [
            "one persisted adapter, one receiver snapshot, one capability family, and one seed",
            "fresh replay set is the remaining eligible MBPP test subset after excluding the 30 original V66 held-out tasks",
            "this establishes operational replay for this artifact, not universal capability portability",
        ],
    }
    output_path.write_bytes(canonical_json(replay_receipt))
    return replay_receipt


def finalize_with_mismatched_control(
    *,
    capability_ir: dict[str, Any],
    donor_receipt: dict[str, Any],
    dataset: Any,
    evaluation_rows: list[dict[str, Any]],
    progress: dict[str, Any],
    ir_path: Path,
    donor_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    mismatched_examples = mismatched_control_examples_from_ir(capability_ir)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    mismatched, mismatched_tokenizer = setup_receiver(SEED + 100)
    trainable_count = trainable_parameter_count(mismatched)
    train_adapter(mismatched, mismatched_tokenizer, mismatched_examples, SEED + 130)
    mismatched_adapter_sha = adapter_digest(mismatched)
    mismatched_evaluation = evaluate_codes(
        evaluation_rows, generate_codes(mismatched, mismatched_tokenizer, evaluation_rows)
    )
    release_model(mismatched, mismatched_tokenizer)

    virgin_evaluation = progress["receiver_virgin"]
    compiled_evaluation = progress["receiver_compiled"]
    direct_evaluation = progress["receiver_direct"]
    base_generic_loss = float(progress["receiver_base_generic_loss"])
    compiled_generic_loss = float(progress["receiver_compiled_generic_loss"])
    direct_generic_loss = float(progress["receiver_direct_generic_loss"])
    compiled_adapter_sha = str(progress["receiver_compiled_adapter_sha256"])
    direct_adapter_sha = str(progress["receiver_direct_adapter_sha256"])

    virgin_rate = float(virgin_evaluation["pass_rate"])
    compiled_rate = float(compiled_evaluation["pass_rate"])
    direct_rate = float(direct_evaluation["pass_rate"])
    mismatched_rate = float(mismatched_evaluation["pass_rate"])
    direct_gain = direct_rate - virgin_rate
    compiled_gain = compiled_rate - virgin_rate
    recovered_gain = compiled_gain / direct_gain if direct_gain > 1e-12 else float("-inf")
    generic_probe_relative_loss_increase = (
        compiled_generic_loss - base_generic_loss
    ) / max(base_generic_loss, 1e-12)
    criteria = {
        "minimum_donor_hidden_test_pass_rate": 0.30,
        "minimum_receiver_direct_gain": 0.10,
        "minimum_receiver_compiled_gain": 0.08,
        "minimum_recovered_gain": 0.80,
        "minimum_correct_wrong_pass_rate_gap": 0.10,
        "minimum_compiled_safe_ast_rate": 0.70,
        "maximum_generic_loss_relative_damage": 0.10,
    }
    passed = (
        float(donor_receipt["donor_hidden_test_pass_rate"])
        >= criteria["minimum_donor_hidden_test_pass_rate"]
        and direct_gain >= criteria["minimum_receiver_direct_gain"]
        and compiled_gain >= criteria["minimum_receiver_compiled_gain"]
        and recovered_gain >= criteria["minimum_recovered_gain"]
        and compiled_rate - mismatched_rate >= criteria["minimum_correct_wrong_pass_rate_gap"]
        and float(compiled_evaluation["safe_ast_rate"])
        >= criteria["minimum_compiled_safe_ast_rate"]
        and generic_probe_relative_loss_increase
        <= criteria["maximum_generic_loss_relative_damage"]
    )

    anchors = capability_ir["anchors"]
    receipt = {
        "schema": RECEIVER_SCHEMA,
        "complete": True,
        "pass": passed,
        "scope": "single-seed held-out functional compilation: Qwen2.5-Coder-1.5B -> SmolLM2-1.7B on one MBPP function-synthesis domain",
        "method": "verified donor implementations -> sealed CapabilityIR -> receiver-native LoRA optimization on a frozen receiver backbone -> held-out execution verification",
        "limits": [
            "receiver-side optimization is required; this is not zero-optimization translation",
            "one seed, one donor/receiver pair, and one capability family",
            "the receiver materialization is a LoRA adapter on a frozen backbone, not direct donor-weight translation",
            "the direct oracle is evaluation-only and does not enter the CapabilityIR compilation route",
            "the negative control is a cyclic prompt-to-code mismatch, not an unrelated capability family",
            "the generic-text probe contains 12 fixed texts and is not a general language benchmark",
            "restricted evaluator rejects imports and dangerous dynamic execution primitives",
        ],
        "evidence_claim": {
            "supported": "bounded cross-model held-out functional gain from receiver-native CapabilityIR compilation",
            "seed_count": 1,
            "donor_receiver_pair_count": 1,
            "capability_family_count": 1,
            "universal_portability_established": False,
            "direct_weight_translation_established": False,
            "superiority_over_direct_finetuning_established": False,
        },
        "negative_control": {
            "kind": "cyclic_prompt_code_mismatch",
            "legacy_serialized_keys": [
                "receiver.wrong_ir",
                "receiver.wrong_adapter_sha256",
                "metrics.correct_wrong_pass_rate_gap",
                "criteria.minimum_correct_wrong_pass_rate_gap",
            ],
            "same_receiver_snapshot": True,
            "same_training_steps": TRAIN_STEPS,
            "same_training_recipe": True,
            "interpretation": "tests whether the correct prompt-to-donor-code association matters; it is not a different capability family",
        },
        "generic_text_probe": {
            "text_count": len(GENERAL_TEXTS),
            "legacy_serialized_keys": [
                "metrics.generic_loss_relative_damage",
                "criteria.maximum_generic_loss_relative_damage",
            ],
            "interpretation": "relative language-model loss increase on this fixed probe only; not a global language-performance measure",
        },
        "seed": SEED,
        "dataset": {
            "id": "mbpp/full",
            "train_fingerprint": dataset["train"]._fingerprint,
            "validation_fingerprint": dataset["validation"]._fingerprint,
            "test_fingerprint": dataset["test"]._fingerprint,
            "anchor_count": len(anchors),
            "heldout_evaluation_count": len(evaluation_rows),
            "mean_evaluation_complexity": sum(complexity_score(row) for row in evaluation_rows)
            / len(evaluation_rows),
            "minimum_evaluation_complexity": min(complexity_score(row) for row in evaluation_rows),
        },
        "capability_ir": {
            "sha256": sha256_file(ir_path),
            "original_reference_code_present": False,
            "hidden_test_content_present": False,
            "all_anchors_reverified": True,
        },
        "donor": {
            "receipt_sha256": sha256_file(donor_receipt_path),
            "hidden_test_pass_rate": donor_receipt["donor_hidden_test_pass_rate"],
            "verified_anchor_count": donor_receipt["verified_anchor_count"],
        },
        "receiver": {
            "model_id": RECEIVER_MODEL_ID,
            "revision": RECEIVER_REVISION,
            "architecture": "LlamaForCausalLM",
            "weight_file_sha256": sha256_file(receiver_weight_file()),
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_target_modules": LORA_TARGETS,
            "trainable_parameter_count": trainable_count,
            "base_parameter_trainable_count": 0,
            "training_steps": TRAIN_STEPS,
            "virgin": virgin_evaluation,
            "compiled": compiled_evaluation,
            "direct": direct_evaluation,
            "wrong_ir": mismatched_evaluation,
            "compiled_adapter_sha256": compiled_adapter_sha,
            "direct_adapter_sha256": direct_adapter_sha,
            "wrong_adapter_sha256": mismatched_adapter_sha,
            "base_generic_loss": base_generic_loss,
            "compiled_generic_loss": compiled_generic_loss,
            "direct_generic_loss": direct_generic_loss,
        },
        "metrics": {
            "direct_gain": direct_gain,
            "compiled_gain": compiled_gain,
            "recovered_gain": recovered_gain,
            "correct_wrong_pass_rate_gap": compiled_rate - mismatched_rate,
            "generic_loss_relative_damage": generic_probe_relative_loss_increase,
        },
        "criteria": criteria,
    }
    output_path.write_bytes(canonical_json(receipt))
    return receipt


def compile_and_evaluate(ir_path: Path, donor_receipt_path: Path, output_path: Path) -> dict[str, Any]:
    capability_ir, donor_receipt, dataset, anchor_rows, evaluation_rows = load_authorities(
        ir_path, donor_receipt_path
    )
    anchors = capability_ir["anchors"]
    compiled_examples = [(anchor["prompt"], anchor["donor_code"]) for anchor in anchors]
    direct_examples = [
        (anchor["prompt"], row["code"])
        for anchor, row in zip(anchors, anchor_rows, strict=True)
    ]
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    base, base_tokenizer = setup_receiver(SEED + 100)
    base_generic_loss = generic_loss(base, base_tokenizer)
    virgin_evaluation = evaluate_codes(
        evaluation_rows, generate_codes(base, base_tokenizer, evaluation_rows)
    )
    trainable_count = trainable_parameter_count(base)
    release_model(base, base_tokenizer)
    progress: dict[str, Any] = {
        "capability_ir_sha256": sha256_file(ir_path),
        "donor_receipt_sha256": sha256_file(donor_receipt_path),
        "receiver_virgin": virgin_evaluation,
        "receiver_base_generic_loss": base_generic_loss,
    }
    write_progress(output_path, "virgin_evaluated", progress)

    compiled, compiled_tokenizer = setup_receiver(SEED + 100)
    train_adapter(compiled, compiled_tokenizer, compiled_examples, SEED + 110)
    compiled_adapter_sha = adapter_digest(compiled)
    compiled_generic_loss = generic_loss(compiled, compiled_tokenizer)
    compiled_evaluation = evaluate_codes(
        evaluation_rows, generate_codes(compiled, compiled_tokenizer, evaluation_rows)
    )
    release_model(compiled, compiled_tokenizer)
    progress.update(
        {
            "receiver_compiled": compiled_evaluation,
            "receiver_compiled_adapter_sha256": compiled_adapter_sha,
            "receiver_compiled_generic_loss": compiled_generic_loss,
        }
    )
    write_progress(output_path, "compiled_evaluated", progress)

    direct, direct_tokenizer = setup_receiver(SEED + 100)
    train_adapter(direct, direct_tokenizer, direct_examples, SEED + 120)
    direct_adapter_sha = adapter_digest(direct)
    direct_generic_loss = generic_loss(direct, direct_tokenizer)
    direct_evaluation = evaluate_codes(
        evaluation_rows, generate_codes(direct, direct_tokenizer, evaluation_rows)
    )
    release_model(direct, direct_tokenizer)
    progress.update(
        {
            "receiver_direct": direct_evaluation,
            "receiver_direct_adapter_sha256": direct_adapter_sha,
            "receiver_direct_generic_loss": direct_generic_loss,
        }
    )
    write_progress(output_path, "direct_evaluated", progress)
    return finalize_with_mismatched_control(
        capability_ir=capability_ir,
        donor_receipt=donor_receipt,
        dataset=dataset,
        evaluation_rows=evaluation_rows,
        progress=progress,
        ir_path=ir_path,
        donor_receipt_path=donor_receipt_path,
        output_path=output_path,
    )


def resume_from_direct_evaluated(
    ir_path: Path,
    donor_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    capability_ir, donor_receipt, dataset, _anchor_rows, evaluation_rows = load_authorities(
        ir_path, donor_receipt_path
    )
    progress = load_direct_progress(
        output_path,
        ir_path,
        donor_receipt_path,
        evaluation_rows,
    )
    return finalize_with_mismatched_control(
        capability_ir=capability_ir,
        donor_receipt=donor_receipt,
        dataset=dataset,
        evaluation_rows=evaluation_rows,
        progress=progress,
        ir_path=ir_path,
        donor_receipt_path=donor_receipt_path,
        output_path=output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-input")
    parser.add_argument("--donor-receipt")
    parser.add_argument("--output")
    parser.add_argument("--source-receipt")
    parser.add_argument("--replay-output")
    parser.add_argument("--replay-count", type=int, default=0)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--materialize-compiled-adapter")
    action.add_argument("--replay-adapter")
    args = parser.parse_args()

    if args.materialize_compiled_adapter:
        if not args.ir_input or not args.donor_receipt or not args.source_receipt:
            parser.error(
                "--materialize-compiled-adapter requires --ir-input, --donor-receipt, and --source-receipt"
            )
        manifest = materialize_compiled_adapter(
            Path(args.ir_input).expanduser().resolve(),
            Path(args.donor_receipt).expanduser().resolve(),
            outside_checkout(Path(args.source_receipt)),
            Path(args.materialize_compiled_adapter),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.replay_adapter:
        if not args.replay_output:
            parser.error("--replay-adapter requires --replay-output")
        replay_receipt = replay_compiled_adapter(
            Path(args.replay_adapter),
            Path(args.replay_output),
            args.replay_count,
        )
        print(json.dumps(replay_receipt, indent=2, sort_keys=True))
        return 0 if replay_receipt["pass"] else 1

    if not args.ir_input or not args.donor_receipt or not args.output:
        parser.error("normal and --resume execution require --ir-input, --donor-receipt, and --output")
    ir_path = Path(args.ir_input).expanduser().resolve()
    donor_receipt = Path(args.donor_receipt).expanduser().resolve()
    output = outside_checkout(Path(args.output))
    receipt = (
        resume_from_direct_evaluated(ir_path, donor_receipt, output)
        if args.resume
        else compile_and_evaluate(ir_path, donor_receipt, output)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
