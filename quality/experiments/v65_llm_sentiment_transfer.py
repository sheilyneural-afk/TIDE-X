#!/usr/bin/env python3
"""V65: real language-capability transfer between pretrained causal LMs.

Donor A learns SST-2 sentiment from original labels.  A is then reduced to a
portable language CapabilityIR made only of canonical text anchors and donor
output probabilities.  The donor is unloaded before Receiver B is compiled.
The receiver compiler never receives SST-2 labels.  B-direct is trained from
labels only as an evaluation oracle; a swapped-operator IR is the wrong-skill
control.

Scope: LoRA capability adapters on two different pretrained causal-LM
architectures.  This is functional distillation through a sealed IR, not a
claim of zero-optimization weight translation or universal LLM portability.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
A_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
B_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
A_PATH = Path.home() / ".cache/huggingface/hub/models--distilbert--distilgpt2/snapshots" / A_REVISION
B_PATH = Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots" / B_REVISION
SEED = 20_260_907
MAX_LENGTH = 64
BATCH_SIZE = 8
LEARNING_RATE = 1.5e-3
TRAIN_LABEL_COUNT = 1_536
ANCHOR_COUNT = 1_024
EVALUATION_COUNT = 400
DONOR_STEPS = 160
RECEIVER_DIRECT_STEPS = 200
RECEIVER_COMPILE_STEPS = 200
WRONG_IR_STEPS = 160
LABEL_TEXT = (" negative", " positive")
CAPABILITY_SCHEMA = "cerebro.tidex.language_capability_ir/v1"
RECEIPT_SCHEMA = "cerebro.tidex.v65_llm_capability_transfer/v1"


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity() -> tuple[str, str]:
    status = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
    )
    if status:
        raise SystemExit("V65 rejected: checkout must be clean")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    if len(commit) != 40 or len(tree) != 40:
        raise SystemExit("V65 rejected: git identity invalid")
    return commit, tree


def outside_checkout(path: Path) -> Path:
    path = path.expanduser().resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return path
    raise SystemExit("V65 rejected: generated evidence must remain outside checkout")


def model_weight_file(snapshot: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = snapshot / name
        if path.is_file():
            return path
    raise SystemExit(f"V65 rejected: model weight file missing under {snapshot}")


def prompt(sentence: str) -> str:
    return f"Review: {sentence.strip()}\nSentiment:"


def class_token_ids(tokenizer: Any) -> torch.Tensor:
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABEL_TEXT]
    if any(len(value) != 1 for value in ids):
        raise SystemExit(f"V65 rejected: label verbalizer not single-token: {ids}")
    return torch.tensor([value[0] for value in ids], dtype=torch.long)


def setup_model(snapshot: Path, target_modules: list[str], seed: int) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float32)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
        ),
    )
    return model, tokenizer


def encode_prompts(tokenizer: Any, prompts: list[str]) -> tuple[Any, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    positions = encoded.attention_mask.sum(1) - 1
    return encoded, positions


def probabilities(model: Any, tokenizer: Any, prompts: list[str], batch_size: int = 16) -> torch.Tensor:
    labels = class_token_ids(tokenizer)
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            encoded, positions = encode_prompts(tokenizer, chunk)
            logits = model(**encoded).logits[torch.arange(len(chunk)), positions][:, labels]
            result.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(result, 0)


def accuracy(predictions: torch.Tensor, labels: list[int]) -> float:
    values = predictions.argmax(1).tolist()
    return sum(int(left == right) for left, right in zip(values, labels, strict=True)) / len(labels)


def train_from_labels(model: Any, tokenizer: Any, prompts: list[str], labels: list[int], *, steps: int, seed: int) -> None:
    rng = random.Random(seed)
    class_ids = class_token_ids(tokenizer)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=LEARNING_RATE)
    model.train()
    for _ in range(steps):
        indices = [rng.randrange(len(prompts)) for _ in range(BATCH_SIZE)]
        batch_prompts = [prompts[index] for index in indices]
        targets = torch.tensor([labels[index] for index in indices], dtype=torch.long)
        encoded, positions = encode_prompts(tokenizer, batch_prompts)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**encoded).logits[torch.arange(BATCH_SIZE), positions][:, class_ids]
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()


def train_from_capability_ir(model: Any, tokenizer: Any, capability_ir: dict[str, Any], *, steps: int, seed: int) -> None:
    if set(capability_ir) != {"schema", "task", "labels", "anchors"} or capability_ir["schema"] != CAPABILITY_SCHEMA:
        raise SystemExit("V65 rejected: CapabilityIR envelope invalid")
    anchors = capability_ir["anchors"]
    if len(anchors) != ANCHOR_COUNT:
        raise SystemExit("V65 rejected: CapabilityIR anchor count invalid")
    prompts: list[str] = []
    teacher_rows: list[list[float]] = []
    for anchor in anchors:
        if set(anchor) != {"anchor_id", "prompt", "prompt_sha256", "output_probabilities"}:
            raise SystemExit("V65 rejected: CapabilityIR anchor fields invalid")
        text = anchor["prompt"]
        if sha256_bytes(text.encode()) != anchor["prompt_sha256"]:
            raise SystemExit("V65 rejected: CapabilityIR prompt digest mismatch")
        probabilities_row = anchor["output_probabilities"]
        if len(probabilities_row) != 2 or any(not isinstance(value, float) for value in probabilities_row):
            raise SystemExit("V65 rejected: CapabilityIR output contract invalid")
        if abs(sum(probabilities_row) - 1.0) > 1e-5:
            raise SystemExit("V65 rejected: CapabilityIR probabilities not normalized")
        prompts.append(text)
        teacher_rows.append(probabilities_row)
    teacher = torch.tensor(teacher_rows, dtype=torch.float32)
    rng = random.Random(seed)
    class_ids = class_token_ids(tokenizer)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=LEARNING_RATE)
    model.train()
    for _ in range(steps):
        indices = [rng.randrange(len(prompts)) for _ in range(BATCH_SIZE)]
        batch_prompts = [prompts[index] for index in indices]
        target = teacher[indices]
        encoded, positions = encode_prompts(tokenizer, batch_prompts)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**encoded).logits[torch.arange(BATCH_SIZE), positions][:, class_ids]
        loss = nn.functional.kl_div(nn.functional.log_softmax(logits, dim=-1), target, reduction="batchmean")
        loss.backward()
        optimizer.step()


def parameter_digest(model: Any, *, adapter: bool) -> str:
    digest = hashlib.sha256()
    observed = 0
    for name, parameter in sorted(model.named_parameters()):
        is_adapter = "lora_" in name
        if is_adapter != adapter:
            continue
        observed += parameter.numel()
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    if observed == 0:
        raise SystemExit("V65 rejected: empty parameter digest domain")
    return digest.hexdigest()


def build_capability_ir(anchor_rows: list[dict[str, Any]], donor_probabilities: torch.Tensor) -> dict[str, Any]:
    anchors = []
    for row, output in zip(anchor_rows, donor_probabilities.tolist(), strict=True):
        text = prompt(row["sentence"])
        prompt_sha = sha256_bytes(text.encode())
        anchors.append(
            {
                "anchor_id": sha256_bytes(f"sst2:{row['idx']}:{prompt_sha}".encode()),
                "prompt": text,
                "prompt_sha256": prompt_sha,
                "output_probabilities": [float(output[0]), float(output[1])],
            }
        )
    anchors.sort(key=lambda item: item["anchor_id"])
    return {
        "schema": CAPABILITY_SCHEMA,
        "task": "GLUE/SST-2 sentiment polarity",
        "labels": ["negative", "positive"],
        "anchors": anchors,
    }


def swapped_ir(capability_ir: dict[str, Any]) -> dict[str, Any]:
    wrong = json.loads(json.dumps(capability_ir))
    for anchor in wrong["anchors"]:
        anchor["output_probabilities"].reverse()
    return wrong


def release_model(model: Any, tokenizer: Any) -> None:
    del model
    del tokenizer
    gc.collect()


def run(receipt_path: Path, ir_path: Path) -> dict[str, Any]:
    commit, tree = git_identity()
    if not A_PATH.is_dir() or not B_PATH.is_dir():
        raise SystemExit("V65 rejected: exact model snapshots not cached")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    dataset = load_dataset("glue", "sst2", download_mode="reuse_dataset_if_exists")
    train_rows = list(dataset["train"])
    evaluation_rows = list(dataset["validation"])
    rng = random.Random(SEED)
    rng.shuffle(train_rows)
    rng.shuffle(evaluation_rows)
    donor_training = train_rows[:TRAIN_LABEL_COUNT]
    anchor_rows = train_rows[TRAIN_LABEL_COUNT : TRAIN_LABEL_COUNT + ANCHOR_COUNT]
    evaluation_rows = evaluation_rows[:EVALUATION_COUNT]
    if len(donor_training) != TRAIN_LABEL_COUNT or len(anchor_rows) != ANCHOR_COUNT or len(evaluation_rows) != EVALUATION_COUNT:
        raise SystemExit("V65 rejected: dataset split incomplete")

    split_projection = {
        "train_idx": [int(row["idx"]) for row in donor_training],
        "anchor_idx": [int(row["idx"]) for row in anchor_rows],
        "evaluation_idx": [int(row["idx"]) for row in evaluation_rows],
    }
    if set(split_projection["train_idx"]) & set(split_projection["anchor_idx"]):
        raise SystemExit("V65 rejected: train/anchor overlap")
    split_sha = sha256_bytes(canonical_json(split_projection))
    train_prompts = [prompt(row["sentence"]) for row in donor_training]
    train_labels = [int(row["label"]) for row in donor_training]
    evaluation_prompts = [prompt(row["sentence"]) for row in evaluation_rows]
    evaluation_labels = [int(row["label"]) for row in evaluation_rows]

    # Receiver virgin baseline.
    receiver, receiver_tokenizer = setup_model(B_PATH, ["q_proj", "v_proj"], SEED + 2)
    receiver_backbone_digest = parameter_digest(receiver, adapter=False)
    receiver_virgin_accuracy = accuracy(
        probabilities(receiver, receiver_tokenizer, evaluation_prompts), evaluation_labels
    )
    release_model(receiver, receiver_tokenizer)

    # Donor learns from original labels; then only its functional behavior survives.
    donor, donor_tokenizer = setup_model(A_PATH, ["c_attn"], SEED + 1)
    donor_backbone_before = parameter_digest(donor, adapter=False)
    donor_virgin_accuracy = accuracy(probabilities(donor, donor_tokenizer, evaluation_prompts), evaluation_labels)
    train_from_labels(donor, donor_tokenizer, train_prompts, train_labels, steps=DONOR_STEPS, seed=SEED + 10)
    donor_direct_accuracy = accuracy(probabilities(donor, donor_tokenizer, evaluation_prompts), evaluation_labels)
    anchor_prompts = [prompt(row["sentence"]) for row in anchor_rows]
    donor_anchor_probabilities = probabilities(donor, donor_tokenizer, anchor_prompts)
    donor_anchor_accuracy = accuracy(donor_anchor_probabilities, [int(row["label"]) for row in anchor_rows])
    donor_mean_anchor_confidence = float(donor_anchor_probabilities.max(1).values.mean())
    donor_adapter_sha = parameter_digest(donor, adapter=True)
    donor_backbone_after = parameter_digest(donor, adapter=False)
    capability_ir = build_capability_ir(anchor_rows, donor_anchor_probabilities)
    capability_ir_raw = canonical_json(capability_ir)
    capability_ir_sha = sha256_bytes(capability_ir_raw)
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    ir_path.write_bytes(capability_ir_raw)
    release_model(donor, donor_tokenizer)
    donor_unloaded_before_receiver_compile = True

    # Direct receiver oracle sees the original labels, but never enters compilation.
    direct, direct_tokenizer = setup_model(B_PATH, ["q_proj", "v_proj"], SEED + 2)
    train_from_labels(
        direct,
        direct_tokenizer,
        train_prompts,
        train_labels,
        steps=RECEIVER_DIRECT_STEPS,
        seed=SEED + 20,
    )
    receiver_direct_accuracy = accuracy(probabilities(direct, direct_tokenizer, evaluation_prompts), evaluation_labels)
    receiver_direct_adapter_sha = parameter_digest(direct, adapter=True)
    release_model(direct, direct_tokenizer)

    # Receiver compiler sees only the serialized CapabilityIR artifact.
    compiler_ir = json.loads(ir_path.read_text())
    compiled, compiled_tokenizer = setup_model(B_PATH, ["q_proj", "v_proj"], SEED + 2)
    compiled_backbone_before = parameter_digest(compiled, adapter=False)
    train_from_capability_ir(
        compiled,
        compiled_tokenizer,
        compiler_ir,
        steps=RECEIVER_COMPILE_STEPS,
        seed=SEED + 30,
    )
    receiver_compiled_accuracy = accuracy(
        probabilities(compiled, compiled_tokenizer, evaluation_prompts), evaluation_labels
    )
    receiver_compiled_adapter_sha = parameter_digest(compiled, adapter=True)
    compiled_backbone_after = parameter_digest(compiled, adapter=False)
    release_model(compiled, compiled_tokenizer)

    # Wrong-skill control: same anchors and receiver optimizer, opposite operator.
    wrong, wrong_tokenizer = setup_model(B_PATH, ["q_proj", "v_proj"], SEED + 2)
    train_from_capability_ir(
        wrong,
        wrong_tokenizer,
        swapped_ir(compiler_ir),
        steps=WRONG_IR_STEPS,
        seed=SEED + 40,
    )
    receiver_wrong_accuracy = accuracy(probabilities(wrong, wrong_tokenizer, evaluation_prompts), evaluation_labels)
    release_model(wrong, wrong_tokenizer)

    direct_gain = receiver_direct_accuracy - receiver_virgin_accuracy
    compiled_gain = receiver_compiled_accuracy - receiver_virgin_accuracy
    recovered_gain = compiled_gain / direct_gain if direct_gain > 1e-12 else float("-inf")
    wrong_gap = receiver_compiled_accuracy - receiver_wrong_accuracy
    criteria = {
        "minimum_donor_gain": 0.10,
        "minimum_receiver_direct_gain": 0.10,
        "minimum_receiver_compiled_gain": 0.10,
        "minimum_recovered_gain": 0.80,
        "minimum_wrong_ir_gap": 0.30,
    }
    backbone_unchanged = (
        donor_backbone_before == donor_backbone_after
        and receiver_backbone_digest == compiled_backbone_before == compiled_backbone_after
    )
    passed = (
        donor_direct_accuracy - donor_virgin_accuracy >= criteria["minimum_donor_gain"]
        and direct_gain >= criteria["minimum_receiver_direct_gain"]
        and compiled_gain >= criteria["minimum_receiver_compiled_gain"]
        and recovered_gain >= criteria["minimum_recovered_gain"]
        and wrong_gap >= criteria["minimum_wrong_ir_gap"]
        and backbone_unchanged
        and donor_unloaded_before_receiver_compile
    )

    a_weights = model_weight_file(A_PATH)
    b_weights = model_weight_file(B_PATH)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "version": "V65",
        "pass": passed,
        "scope": "real pretrained causal-LM LoRA capability transfer on held-out SST-2 evaluation",
        "method": "donor labeled fine-tune -> sealed text/probability CapabilityIR -> receiver-native LoRA compilation",
        "limits": [
            "functional distillation uses receiver-side optimization",
            "one language capability and one donor/receiver pair",
            "LoRA adapter transfer, not full-checkpoint weight translation",
        ],
        "git_commit": commit,
        "git_tree": tree,
        "experiment_script_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "dataset": {
            "id": "glue/sst2",
            "train_fingerprint": dataset["train"]._fingerprint,
            "validation_fingerprint": dataset["validation"]._fingerprint,
            "split_sha256": split_sha,
            "train_label_count": TRAIN_LABEL_COUNT,
            "anchor_count": ANCHOR_COUNT,
            "heldout_evaluation_count": EVALUATION_COUNT,
        },
        "donor": {
            "model_id": "distilbert/distilgpt2",
            "revision": A_REVISION,
            "architecture": "GPT2LMHeadModel",
            "weight_file_sha256": sha256_file(a_weights),
            "lora_target_modules": ["c_attn"],
            "lora_rank": 8,
            "adapter_sha256": donor_adapter_sha,
            "virgin_accuracy": donor_virgin_accuracy,
            "direct_accuracy": donor_direct_accuracy,
            "anchor_accuracy_diagnostic_only": donor_anchor_accuracy,
            "mean_anchor_confidence": donor_mean_anchor_confidence,
            "backbone_sha256_before": donor_backbone_before,
            "backbone_sha256_after": donor_backbone_after,
        },
        "capability_ir": {
            "schema": CAPABILITY_SCHEMA,
            "sha256": capability_ir_sha,
            "original_labels_present": False,
            "anchor_fields": ["anchor_id", "prompt", "prompt_sha256", "output_probabilities"],
            "donor_unloaded_before_receiver_compile": donor_unloaded_before_receiver_compile,
        },
        "receiver": {
            "model_id": "HuggingFaceTB/SmolLM2-135M",
            "revision": B_REVISION,
            "architecture": "LlamaForCausalLM",
            "weight_file_sha256": sha256_file(b_weights),
            "lora_target_modules": ["q_proj", "v_proj"],
            "lora_rank": 8,
            "virgin_accuracy": receiver_virgin_accuracy,
            "direct_accuracy": receiver_direct_accuracy,
            "compiled_accuracy": receiver_compiled_accuracy,
            "wrong_ir_accuracy": receiver_wrong_accuracy,
            "direct_adapter_sha256": receiver_direct_adapter_sha,
            "compiled_adapter_sha256": receiver_compiled_adapter_sha,
            "backbone_sha256": receiver_backbone_digest,
            "backbone_unchanged_after_compilation": backbone_unchanged,
            "original_sst2_labels_seen_by_compiler": False,
        },
        "metrics": {
            "direct_gain": direct_gain,
            "compiled_gain": compiled_gain,
            "recovered_gain": recovered_gain,
            "correct_wrong_accuracy_gap": wrong_gap,
        },
        "criteria": criteria,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--ir-output", required=True)
    args = parser.parse_args()
    output = outside_checkout(Path(args.output))
    ir_output = outside_checkout(Path(args.ir_output))
    receipt = run(output, ir_output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
