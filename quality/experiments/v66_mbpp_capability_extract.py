#!/usr/bin/env python3
"""V66 phase 1: verified donor capability extraction for receiver compilation.

The donor is asked to solve MBPP tasks. An anchor is admitted only when the
donor-generated implementation passes held-out tests under the restricted
Python evaluator. The resulting CapabilityIR contains canonical prompts and
verified donor implementations, never MBPP reference implementations or
hidden-test content.

Phase 2 performs receiver-native LoRA optimization from that sealed CapabilityIR
on a different pretrained model and evaluates held-out functional behavior. Its
negative control is a cyclic prompt-to-code mismatch with the same receiver and
training budget; it is not a distinct "wrong capability" family.

V66 is evidence for one seed, one donor/receiver pair and one function-synthesis
domain. It does not claim universal model portability or direct donor-weight
translation.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import random
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DONOR_GGUF = Path("/home/yo/.local/share/project-auditor/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
SEED = 20_260_907
ANCHOR_TARGET = 40
ANCHOR_CANDIDATES = 110
EVALUATION_TARGET = 30
DONOR_MAX_TOKENS = 120
CAPABILITY_SCHEMA = "cerebro.tidex.code_capability_ir/v1"
DONOR_RECEIPT_SCHEMA = "cerebro.tidex.v66_donor_code_capability/v1"

SAFE_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "ord",
    "pow",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)
DENY_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "help",
    "exit",
    "quit",
}
DENY_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.With,
    ast.AsyncWith,
    ast.TryStar,
)
_LLM: Any | None = None


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_python_ast(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    if not any(isinstance(node, ast.FunctionDef) for node in tree.body):
        return False
    for node in ast.walk(tree):
        if isinstance(node, DENY_NODES):
            return False
        if isinstance(node, ast.Name) and (node.id in DENY_NAMES or node.id.startswith("__")):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
    return True


def complexity_score(row: dict[str, Any]) -> int:
    try:
        tree = ast.parse(row["code"])
    except (SyntaxError, ValueError):
        return -1
    if any(isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)) for node in ast.walk(tree)):
        return -1
    if row["test_setup_code"].strip() or len(row["test_list"]) < 3:
        return -1
    score = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            score += 2
        elif isinstance(node, ast.If):
            score += 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            score += 1
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in function_names:
            score += 2
    if len(row["code"]) >= 160:
        score += 1
    return score


def task_prompt(row: dict[str, Any]) -> str:
    return (
        "Implement the complete Python function(s) for this programming task. "
        "Return only executable Python function definitions. Do not use markdown fences, "
        "imports, files, network, input(), or explanations.\n"
        f"Task: {row['text'].strip()}\n"
        f"Public example: {row['test_list'][0]}\n"
        "Python code:\n"
    )


def extract_python(raw: str) -> str:
    candidates: list[str] = []
    if "```" in raw:
        for part in raw.split("```"):
            candidate = part.strip()
            if candidate.lower().startswith("python"):
                candidate = candidate[6:].lstrip("\n")
            if "def " in candidate:
                candidates.append(candidate)
    first_definition = raw.find("def ")
    if first_definition >= 0:
        candidates.append(raw[first_definition:])
    for candidate in candidates:
        lines = candidate.splitlines()
        for end in range(len(lines), 0, -1):
            prefix = "\n".join(lines[:end]).strip()
            if safe_python_ast(prefix):
                return prefix + "\n"
    return ""


def _resource_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024, 384 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_restricted_tests(code: str, tests: list[str]) -> bool:
    if not code or not safe_python_ast(code):
        return False
    runner = """import builtins,json,sys\nx=json.loads(sys.stdin.read())\nb={k:getattr(builtins,k) for k in x['safe']}\ng={'__builtins__':b}\ntry:\n exec(x['code'],g,g)\n for test in x['tests']:\n  exec(test,g,g)\n print('PASS')\nexcept BaseException as error:\n print('FAIL:'+type(error).__name__)\n"""
    payload = json.dumps({"code": code, "tests": tests, "safe": SAFE_BUILTINS})
    try:
        completed = subprocess.run(
            ["/usr/bin/python3", "-I", "-c", runner],
            input=payload,
            text=True,
            capture_output=True,
            timeout=3,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            preexec_fn=_resource_limits,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.stdout.strip() == "PASS"


def eligible_task(row: dict[str, Any]) -> bool:
    score = complexity_score(row)
    return (
        score >= 3
        and 80 <= len(row["code"]) <= 1_200
        and safe_python_ast(row["code"])
        and run_restricted_tests(row["code"], row["test_list"][1:])
    )


def _worker_init() -> None:
    global _LLM
    from llama_cpp import Llama

    _LLM = Llama(
        model_path=str(DONOR_GGUF),
        n_ctx=1_024,
        n_threads=4,
        n_batch=256,
        verbose=False,
    )


def _generate_one(item: tuple[int, str]) -> dict[str, Any]:
    if _LLM is None:
        raise RuntimeError("donor worker not initialized")
    task_id, prompt = item
    output = _LLM(
        prompt,
        max_tokens=DONOR_MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=1.0,
    )
    raw = output["choices"][0]["text"]
    return {
        "task_id": task_id,
        "raw": raw,
        "code": extract_python(raw),
        "completion_tokens": output.get("usage", {}).get("completion_tokens"),
    }


def extract_donor_capability(ir_path: Path, receipt_path: Path) -> dict[str, Any]:
    if not DONOR_GGUF.is_file():
        raise SystemExit("V66 rejected: donor GGUF missing")
    dataset = load_dataset("mbpp", download_mode="reuse_dataset_if_exists")
    anchor_rows = [
        row
        for split in ("train", "validation")
        for row in dataset[split]
        if eligible_task(row)
    ]
    evaluation_rows = [row for row in dataset["test"] if eligible_task(row)]
    rng = random.Random(SEED)
    rng.shuffle(anchor_rows)
    rng.shuffle(evaluation_rows)
    anchor_rows = anchor_rows[:ANCHOR_CANDIDATES]
    evaluation_rows = evaluation_rows[:EVALUATION_TARGET]
    work = [
        (int(row["task_id"]), task_prompt(row)) for row in anchor_rows + evaluation_rows
    ]
    started = time.monotonic()
    generated: dict[int, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2, mp_context=context, initializer=_worker_init
    ) as executor:
        for index, result in enumerate(executor.map(_generate_one, work, chunksize=1), 1):
            generated[int(result["task_id"])] = result
            if index % 10 == 0:
                print(
                    f"donor generated {index}/{len(work)} elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )

    verified_rows: list[dict[str, Any]] = []
    for row in anchor_rows:
        result = generated[int(row["task_id"])]
        if result["code"] and run_restricted_tests(result["code"], row["test_list"][1:]):
            verified_rows.append(row)
    if len(verified_rows) < ANCHOR_TARGET:
        raise SystemExit(
            f"V66 donor extraction rejected: {len(verified_rows)} verified anchors < {ANCHOR_TARGET}"
        )
    verified_rows = verified_rows[:ANCHOR_TARGET]

    anchors = []
    for row in verified_rows:
        prompt = task_prompt(row)
        code = generated[int(row["task_id"])] ["code"]
        anchors.append(
            {
                "anchor_id": sha256_bytes(
                    f"mbpp:{row['task_id']}:{sha256_bytes(prompt.encode())}".encode()
                ),
                "task_id": int(row["task_id"]),
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode()),
                "donor_code": code,
                "donor_code_sha256": sha256_bytes(code.encode()),
                "hidden_test_commitment_sha256": sha256_bytes(
                    canonical_json(row["test_list"][1:])
                ),
                "verification": "passed",
                "complexity_score": complexity_score(row),
            }
        )
    anchors.sort(key=lambda item: item["anchor_id"])
    capability_ir = {
        "schema": CAPABILITY_SCHEMA,
        "capability": "python.function_synthesis.mbpp.medium_high",
        "donor_model": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M",
        "original_reference_code_present": False,
        "hidden_test_content_present": False,
        "anchors": anchors,
    }
    ir_path.write_bytes(canonical_json(capability_ir))

    evaluation = []
    for row in evaluation_rows:
        result = generated[int(row["task_id"])]
        passed = bool(result["code"]) and run_restricted_tests(
            result["code"], row["test_list"][1:]
        )
        evaluation.append(
            {
                "task_id": int(row["task_id"]),
                "pass": passed,
                "safe_ast": bool(result["code"]),
                "complexity_score": complexity_score(row),
            }
        )
    donor_pass_rate = sum(item["pass"] for item in evaluation) / len(evaluation)
    receipt = {
        "schema": DONOR_RECEIPT_SCHEMA,
        "seed": SEED,
        "donor_gguf_sha256": sha256_file(DONOR_GGUF),
        "dataset_train_fingerprint": dataset["train"]._fingerprint,
        "dataset_validation_fingerprint": dataset["validation"]._fingerprint,
        "dataset_test_fingerprint": dataset["test"]._fingerprint,
        "capability_ir_sha256": sha256_file(ir_path),
        "candidate_anchor_count": len(anchor_rows),
        "verified_anchor_count": len(anchors),
        "available_verified_anchor_count": len(verified_rows),
        "anchor_verification_rate": len(verified_rows) / len(anchor_rows),
        "evaluation_count": len(evaluation),
        "donor_hidden_test_pass_rate": donor_pass_rate,
        "minimum_complexity_score": min(item["complexity_score"] for item in evaluation),
        "mean_complexity_score": sum(item["complexity_score"] for item in evaluation)
        / len(evaluation),
        "evaluation": evaluation,
        "pass": donor_pass_rate >= 0.30 and len(anchors) >= ANCHOR_TARGET,
    }
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-donor", action="store_true")
    parser.add_argument("--ir-output", required=True)
    parser.add_argument("--receipt-output", required=True)
    args = parser.parse_args()
    if not args.extract_donor:
        raise SystemExit("V66: only --extract-donor is implemented in this revision")
    ir_path = Path(args.ir_output).expanduser().resolve()
    receipt_path = Path(args.receipt_output).expanduser().resolve()
    for path in (ir_path, receipt_path):
        try:
            path.relative_to(ROOT)
        except ValueError:
            pass
        else:
            raise SystemExit("V66 rejected: generated artifacts must remain outside checkout")
    receipt = extract_donor_capability(ir_path, receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
