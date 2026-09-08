#!/usr/bin/env python3
"""Retry V69 regularization using sealed calibration requests and the real Rust kernel."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        stream.write(canonical(value))


def verify_receiver(args):
    import sys
    import numpy as np
    import torch
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from quality.experiments import v69_target_update_free_compilation as v69
    pre = json.loads((args.root / "precommit.json").read_bytes())
    parent = Path(next(iter(pre["inputs_sha256"]))).parent.parent
    retry = json.loads((args.root / "receipt.json").read_bytes())
    if hashlib.sha256((parent / "calibration-receipt.json").read_bytes()).hexdigest() != pre["parent_calibration_receipt_sha256"]:
        raise ValueError("parent calibration changed")
    if hashlib.sha256(args.tidex.read_bytes()).hexdigest() != retry["binary_sha256"]:
        raise ValueError("retry binary changed")
    manifest_path = parent / "reproducibility/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen source changed")
    evidence = {}
    for path in (parent / "state/probe_inputs/by-sha").glob("*.json"):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != path.stem:
            raise ValueError("calibration evidence hash mismatch")
        row = json.loads(data)
        if row.get("schema") in ("cerebro.tidex.cross_model_functional_evidence/v1",
                                  "cerebro.tidex.cross_model_receiver_solution_evidence/v1"):
            evidence[(row["schema"], row["capability_id"])] = row
    raw, deltas = [], []
    for threshold in v69.CALIBRATION_THRESHOLDS:
        identifier = v69.cap_id(threshold)
        raw.append(evidence[("cerebro.tidex.cross_model_functional_evidence/v1", identifier)]["raw_logit_margins"])
        deltas.append(evidence[("cerebro.tidex.cross_model_receiver_solution_evidence/v1", identifier)]["delta_values"])
    raw, deltas = np.asarray(raw), np.asarray(deltas)
    model_hash = v69.sha_file(v69.RECEIVER_DIR / "model.safetensors")
    selected = retry["selected"]["ridge"]
    write_new(args.root / "behavior-precommit.json", {
        "schema": "cerebro.tidex.v69_retry_behavior_precommit/v1",
        "selected_ridge": selected, "receiver_sha256": model_hash,
        "parent_calibration_receipt_sha256": pre["parent_calibration_receipt_sha256"],
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "confirmation_target_data_used": False,
        "criteria": v69.CRITERIA, "blocked_fold_score": 0.0})
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = v69.load_receiver(v69.RECEIVER_DIR, 8)
    original_norm = model.model.norm.weight.detach().clone()
    true_id, false_id = v69.one_token(tokenizer, " true"), v69.one_token(tokenizer, " false")
    rows = []
    for index, threshold in enumerate(v69.CALIBRATION_THRESHOLDS):
        fitted = v69.calibration_fold(raw, deltas, index)
        path = args.root / ("ridge-" + str(selected)) / f"loo-{threshold:03d}-decode_then_project.json"
        request = json.loads(path.read_bytes())
        np.testing.assert_array_equal(request["calibration"]["functional_signatures"], fitted["functional_signatures"])
        np.testing.assert_array_equal(request["calibration"]["receiver_solutions"], fitted["coordinates"])
        result = subprocess.run([str(args.tidex), "benchmark", "response", str(path)],
            capture_output=True, text=True, check=True, timeout=60)
        report = json.loads(result.stdout)
        values = v69.numbers(threshold, v69.RECEIVER_VALIDATION_OFFSETS)
        prompts = [f"Input: {n}\nOutput:" for n in values]
        features, margins = v69.receiver_features(model, tokenizer, values, true_id, false_id)
        labels = np.asarray([-1.0] * 4 + [1.0] * 4)
        delta = (np.asarray(report["target_delta"]) @ fitted["axes"].astype(np.float32).astype(np.float64)).astype(np.float32)
        with torch.no_grad():
            model.model.norm.weight.copy_(original_norm + torch.from_numpy(delta))
        try:
            measured = v69.receiver_prompt_margins(model, tokenizer, prompts, true_id, false_id)
        finally:
            with torch.no_grad():
                model.model.norm.weight.copy_(original_norm)
        np.testing.assert_allclose(measured, margins + features @ delta.astype(np.float64), atol=2e-5, rtol=2e-6)
        row = {"threshold": threshold, "allowed": report["allowed"],
            "base_accuracy": float((np.sign(margins) == labels).mean()),
            "measured_accuracy": float((np.sign(measured) == labels).mean())}
        row["deliverable_score"] = row["measured_accuracy"] if row["allowed"] else 0.0
        rows.append(row)
        print(json.dumps(row), flush=True)
    base = float(np.mean([r["base_accuracy"] for r in rows]))
    score = float(np.mean([r["deliverable_score"] for r in rows]))
    result = {"schema": "cerebro.tidex.v69_retry_behavior/v1", "complete": True,
        "selected_ridge": selected, "mean_base_accuracy": base,
        "mean_measured_accuracy": float(np.mean([r["measured_accuracy"] for r in rows])),
        "delivery_coverage": sum(r["allowed"] for r in rows) / len(rows),
        "mean_deliverable_score": score, "mean_deliverable_gain": score - base,
        "calibration_delivery_gate_pass": score >= v69.CRITERIA["minimum_calibration_compiler_loo_mean_accuracy"]
            and score - base >= v69.CRITERIA["minimum_calibration_compiler_loo_gain"],
        "confirmation_target_data_used": False, "rows": rows}
    write_new(args.root / "behavior-receipt.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)
    return 0 if result["calibration_delivery_gate_pass"] else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tidex", type=Path, required=True)
    parser.add_argument("--verify-receiver", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.verify_receiver:
        if args.source is None:
            parser.error("--verify-receiver requires --source")
        return verify_receiver(args)
    pre_path = args.root / "precommit.json"
    pre_bytes = pre_path.read_bytes()
    pre = json.loads(pre_bytes)
    if pre["schema"] != "cerebro.tidex.v69_calibration_retry_precommit/v1" or pre["confirmation_target_data_used"]:
        raise ValueError("invalid calibration-only precommit")
    binary_hash = hashlib.sha256(args.tidex.read_bytes()).hexdigest()
    inputs = []
    for name, digest in sorted(pre["inputs_sha256"].items()):
        path = Path(name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("calibration input changed")
        request = json.loads(data)
        if request["schema"] != "cerebro.tidex.receiver_signature_benchmark_input/v1":
            raise ValueError("unexpected compiler input")
        inputs.append((path.stem, request))
    rows = []
    for ridge in pre["ridge_grid"]:
        reports = []
        for label, original in inputs:
            request = json.loads(json.dumps(original))
            request["policy"]["ridge"] = ridge
            path = args.root / ("ridge-" + str(ridge)) / (label + ".json")
            write_new(path, request)
            completed = subprocess.run([str(args.tidex), "benchmark", "response", str(path)],
                capture_output=True, text=True, timeout=60, check=True)
            report = json.loads(completed.stdout)
            write_new(path.with_suffix(".report.json"), report)
            reports.append(report)
        row = {"ridge": ridge, "allowed_fold_count": sum(r["allowed"] for r in reports),
            "mean_decoder_loo_r2": sum(r["decoder_loo_r2"] for r in reports) / len(reports),
            "minimum_decoder_loo_cosine": min(r["decoder_min_loo_cosine"] for r in reports),
            "minimum_encoder_loo_r2": min(r["encoder_loo_r2"] for r in reports),
            "maximum_functional_relative_error": max(r["functional_relative_error"] for r in reports),
            "minimum_identity_margin": min(r["identity_margin"] for r in reports)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    selected = max(rows, key=lambda r: (r["allowed_fold_count"], r["mean_decoder_loo_r2"]))
    if hashlib.sha256(args.tidex.read_bytes()).hexdigest() != binary_hash or pre_path.read_bytes() != pre_bytes:
        raise ValueError("retry authority changed")
    receipt = {"schema": "cerebro.tidex.v69_calibration_retry/v1", "complete": True,
        "precommit_sha256": hashlib.sha256(pre_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "binary_sha256": binary_hash, "results": rows, "selected": selected,
        "selection_is_calibration_only": True, "target_receiver_execution_performed": False,
        "behavioral_quality_confirmed_by_this_retry": False}
    write_new(args.root / "receipt.json", receipt)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
