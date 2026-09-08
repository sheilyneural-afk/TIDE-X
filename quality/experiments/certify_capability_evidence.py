#!/usr/bin/env python3
"""Recompute bounded behavioral evidence; never promote a model or infer cognition.

This is a retrospective artifact audit, not an independent replication.
Missing prospective intervention/monitoring experiments yield NOT_DEMONSTRATED.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from quality.experiments.capability_statistics import (
    paired_gain_interval, paired_pass_counts, validated_pass_rows,
)

SCHEMA = "cerebro.tidex.capability_evidence_audit/v1"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


class Evidence:
    def __init__(self):
        self.references = {}
        self.snapshots = {}

    def digest(self, path):
        path = Path(path).absolute()
        require(path.is_file() and not path.is_symlink(), "missing/nonregular evidence: " + str(path))
        before = path.stat()
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if str(path) in self.references:
            require(signature == self.snapshots[str(path)], "evidence changed: " + str(path))
            return self.references[str(path)]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        require(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "evidence changed while hashing")
        self.references[str(path)] = digest.hexdigest()
        self.snapshots[str(path)] = signature
        return digest.hexdigest()

    def read(self, path, schema=None):
        path = Path(path)
        digest = self.digest(path)
        require(path.stat().st_size <= 64 * 1024 * 1024, "JSON evidence exceeds budget")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, "evidence changed before parsing")
        def reject_constant(value):
            raise ValueError("non-finite JSON: " + value)
        value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        require(isinstance(value, dict), "JSON root must be object")
        if schema:
            require(value.get("schema") == schema, "schema mismatch: " + str(path))
        return value

    def finish(self):
        for path in list(self.references):
            self.digest(path)


def evaluation(value):
    ids = validated_pass_rows(value)
    safe = []
    for row in value["results"]:
        require(type(row.get("safe_ast")) is bool, "safe_ast must be boolean")
        code_hash = row.get("code_sha256")
        if code_hash is not None:
            require(isinstance(code_hash, str) and len(code_hash) == 64
                    and all(c in "0123456789abcdef" for c in code_hash), "invalid code hash")
        require(not row["pass"] or (row["safe_ast"] and code_hash is not None),
                "passing result lacks executable code identity")
        safe.append(row["safe_ast"])
    require(type(value.get("safe_ast_rate")) in (float, int)
            and math.isclose(value["safe_ast_rate"], sum(safe) / len(safe), rel_tol=0, abs_tol=1e-12),
            "safe AST aggregate mismatch")
    return ids


def comparison(base, candidate):
    left, right = evaluation(base), evaluation(candidate)
    return {
        "task_count": len(left), "baseline_correct": sum(left.values()),
        "candidate_correct": sum(right.values()),
        "baseline_pass_rate": sum(left.values()) / len(left),
        "candidate_pass_rate": sum(right.values()) / len(right),
        "paired": paired_pass_counts(base, candidate),
        "gain_interval": paired_gain_interval(base, candidate),
    }


def audit(args):
    evidence = Evidence()
    manifest = evidence.read(args.adapter_dir / "manifest.json", "cerebro.tidex.v66_compiled_adapter_artifact/v1")
    source = evidence.read(args.source, "cerebro.tidex.v66_receiver_code_capability/v1")
    replay = evidence.read(args.replay, "cerebro.tidex.v66_compiled_adapter_replay/v1")
    direct = evidence.read(args.direct_replay, "cerebro.tidex.v67_direct_weight_replay/v1")
    smoke = evidence.read(args.smoke, "cerebro.tidex.v67_weight_actuator_smoke/v1")
    ir = evidence.read(args.ir, "cerebro.tidex.code_capability_ir/v1")
    donor = evidence.read(args.donor, "cerebro.tidex.v66_donor_code_capability/v1")
    for receipt in (source, replay, direct):
        require(receipt.get("complete") is True, "incomplete receipt")
    # Recompute; input pass flags are deliberately not used.
    manifest_sha = evidence.digest(args.adapter_dir / "manifest.json")
    require((args.adapter_dir / "manifest.sha256").read_text().split() == [manifest_sha, "manifest.json"],
            "manifest sidecar mismatch")
    evidence.digest(args.adapter_dir / "manifest.sha256")
    require(manifest_sha == replay["adapter_artifact"]["manifest_sha256"]
            == smoke["source_v66_manifest_sha256"], "manifest binding mismatch")
    require(evidence.digest(args.source) == manifest["source_v66_receipt_sha256"], "source receipt binding mismatch")
    require(evidence.digest(args.ir) == manifest["capability_ir_sha256"]
            == source["capability_ir"]["sha256"] == donor["capability_ir_sha256"], "IR binding mismatch")
    require(evidence.digest(args.donor) == manifest["donor_receipt_sha256"]
            == source["donor"]["receipt_sha256"], "donor binding mismatch")
    require(manifest["adapter"]["parameter_digest_sha256"]
            == replay["adapter_artifact"]["parameter_digest_sha256"]
            == source["receiver"]["compiled_adapter_sha256"], "adapter parameter binding mismatch")
    declared = manifest["adapter"]["files_sha256"]
    require(isinstance(declared, dict) and declared, "missing adapter inventory")
    observed = {p.name for p in args.adapter_dir.iterdir() if p.is_file()} - {"manifest.json", "manifest.sha256"}
    require(observed == set(declared), "adapter inventory mismatch")
    for name, digest in declared.items():
        require(Path(name).name == name, "adapter path escapes directory")
        require(evidence.digest(args.adapter_dir / name) == digest, "adapter digest mismatch")
    require(evidence.digest(args.adapter_dir / "adapter_model.safetensors") == smoke["adapter_model_sha256"],
            "smoke adapter mismatch")
    require(evidence.digest(args.adapter_dir / "adapter_config.json") == smoke["adapter_config_sha256"],
            "smoke adapter config mismatch")
    # Base snapshots commonly use content-addressed symlinks; bind the resolved bytes.
    base = args.base_model.resolve(strict=True)
    require(evidence.digest(base) == manifest["receiver"]["weight_file_sha256"]
            == source["receiver"]["weight_file_sha256"] == smoke["base_model_sha256"]
            == smoke["materialization"]["base_model_sha256"], "base weights mismatch")
    require(evidence.digest(args.model_dir / "model.safetensors")
            == direct["model"]["model_safetensors_sha256"]
            == smoke["materialization"]["output_model_sha256"], "standalone weights mismatch")
    require(not any("adapter" in p.name.lower() or "lora" in p.name.lower() for p in args.model_dir.iterdir()),
            "standalone directory contains adapter")
    require(smoke["materialization"]["requires_adapter_at_runtime"] is False, "runtime adapter dependency")
    historical = source["receiver"]
    original = comparison(historical["virgin"], historical["compiled"])
    original_direct = comparison(historical["direct"], historical["compiled"])
    original_wrong = comparison(historical["wrong_ir"], historical["compiled"])
    source_ids = set(evaluation(historical["compiled"]))
    manifest_ids = manifest["source_behavior"]["heldout_task_ids"]
    require(len(manifest_ids) == len(set(manifest_ids)) and set(manifest_ids) == source_ids,
            "historical task manifest mismatch")
    expected_history = manifest["source_behavior"]["compiled_evaluation_sha256"]
    require(hashlib.sha256(canonical(historical["compiled"])).hexdigest() == expected_history,
            "historical semantic digest mismatch")
    require(canonical(historical["compiled"]) == canonical(replay["source_behavior_replay"]["evaluation"])
            == canonical(direct["source_behavior"]["direct_evaluation"]), "historical replay differs")
    fresh = replay["fresh_evaluation"]
    fresh_result = comparison(fresh["virgin"], fresh["compiled_adapter"])
    fresh_ids = set(evaluation(fresh["compiled_adapter"]))
    require(len(fresh["task_ids"]) == len(fresh_ids) and set(fresh["task_ids"]) == fresh_ids,
            "fresh task identity mismatch")
    require(fresh["evaluated_count"] == len(fresh_ids), "fresh count mismatch")
    anchor_ids = [row["task_id"] for row in ir["anchors"]]
    require(all(type(x) is int for x in anchor_ids) and len(anchor_ids) == len(set(anchor_ids)),
            "duplicate/invalid training anchors")
    require(not set(anchor_ids) & (source_ids | fresh_ids) and not source_ids & fresh_ids,
            "training/historical/fresh overlap")
    require(canonical(fresh["compiled_adapter"]) == canonical(direct["fresh_behavior"]["direct_evaluation"]),
            "standalone fresh replay differs")
    probe = replay["generic_text_probe"]
    require(all(type(probe[k]) in (int, float) and math.isfinite(probe[k]) and probe[k] > 0
                for k in ("virgin_loss", "compiled_loss")), "invalid generic losses")
    damage = (probe["compiled_loss"] - probe["virgin_loss"]) / probe["virgin_loss"]
    require(math.isclose(damage, probe["relative_loss_increase"], rel_tol=0, abs_tol=1e-12),
            "generic damage aggregate mismatch")
    criteria = replay["criteria"]
    expected_criteria = {
        "require_exact_source_behavior_replay": True,
        "minimum_fresh_pass_rate_gain": 0.08, "maximum_exact_two_sided_p": 0.05,
        "minimum_compiled_safe_ast_rate": 0.70,
        "maximum_generic_probe_relative_loss_increase": 0.10,
    }
    require(criteria == expected_criteria, "historical replay criteria changed")
    gates = {
        "fresh_gain": fresh_result["gain_interval"]["mean"] >= criteria["minimum_fresh_pass_rate_gain"],
        "paired_evidence": fresh_result["paired"]["exact_two_sided_p"] <= criteria["maximum_exact_two_sided_p"],
        "safe_ast": fresh["compiled_adapter"]["safe_ast_rate"] >= criteria["minimum_compiled_safe_ast_rate"],
        "bounded_generic_probe": damage <= criteria["maximum_generic_probe_relative_loss_increase"],
        "source_replay_exact": True, "standalone_replay_exact": True,
    }
    evidence.finish()
    git = subprocess.check_output(["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_digests = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted((ROOT / "quality/experiments").glob("*.py"))}
    report = {
        "schema": SCHEMA, "audit_complete": True, "full_certification": False,
        "evaluation_mode": "retrospective_recalculation_of_retained_receipts",
        "independent_model_rerun_performed": False, "authorizes_promotion": False,
        "code": {"git_head": git, "source_sha256": source_digests},
        "evidence_sha256": evidence.references,
        "useful_learning": {
            "status": "BOUNDED_RETROSPECTIVE_SUPPORT" if all(gates.values()) else "CRITERIA_FAILED",
            "historical_protocol_gates": gates, "fresh": fresh_result,
            "historical_vs_base": original, "historical_vs_direct": original_direct,
            "historical_vs_mismatched": original_wrong,
            "generic_probe_relative_damage": damage, "generic_probe_text_count": probe["text_count"],
            "standalone_equivalent_task_count": len(source_ids) + len(fresh_ids),
            "scope": "one persisted adapter and standalone checkpoint; selected MBPP subset; one training seed",
            "limits": [
                "receipt and weight integrity is not independent experimental attestation",
                "historical test outcomes were not re-executed by this auditor",
                "no proof of protocol preregistration or absence of adaptive test-set reuse",
                "task-pair inference assumes independence; task-family clustering is unmeasured",
                "fresh direct-training and mismatched controls unavailable",
                "training-seed robustness, task-family generalization and runtime utility cost unmeasured",
                "12-text loss probe is not broad capability preservation",
            ],
        },
        "mechanism": {
            "status": "NOT_DEMONSTRATED",
            "reason": "These receipts contain output test outcomes, not precommitted internal-intervention predictions and measured state traces.",
            "required_evidence": [
                "frozen operational hypothesis and internal-state mapping before evaluation",
                "actual checkpoint-bound state capture and interventions",
                "held-out intervention predictions, matched perturbation controls and compositional tests",
                "independent comparison of predicted and observed transitions",
            ],
        },
        "metacognition": {
            "status": "NOT_DEMONSTRATED",
            "reason": "These receipts contain no pre-outcome confidence, monitoring decisions or equal-budget supervisor ablation.",
            "required_evidence": [
                "specified monitored system boundary and frozen supervisor",
                "pre-outcome confidence and action records committed before hidden-test feedback",
                "calibration and failure detection on reserved cases",
                "equal-budget comparisons with disabled and shuffled-signal supervisor",
                "cost-adjusted utility and reaction to previously unseen mechanism degradation",
            ],
        },
    }


    prospective_root = getattr(args, "prospective_run", None)
    if prospective_root is not None:
        from quality.experiments.prospective_capability_evidence import verify_run, read_journal
        measured = verify_run(prospective_root)
        retained = evidence.read(prospective_root / "receipt.json")
        require(measured == retained, "prospective receipt differs from recalculated observations")
        independent = evidence.read(prospective_root / "independent-replay.json", "cerebro.tidex.prospective_independent_replay/v1")
        require(independent.get("complete") is True and independent.get("pass") is True
                and independent["scientific_verdict"] == measured
                and independent["journal_tip_sha256"] == measured["journal_tip_sha256"],
                "prospective independent replay missing or inconsistent")
        records, _ = read_journal(prospective_root / "events")
        plan = records[0]["payload"]
        require(plan["model_files_sha256"]["model.safetensors"] == evidence.digest(args.model_dir / "model.safetensors"),
                "prospective study uses a different checkpoint")
        for path in sorted(prospective_root.rglob("*")):
            if path.is_file():
                evidence.digest(path)
        report["mechanism"] = measured["mechanism"]
        stages = {record["kind"]: record["payload"] for record in records}
        diagnostics = {}
        for arm in ("learned_delta", "permuted_delta"):
            cells = [(row, stages["intervention_observations"][key])
                     for key, row in stages["intervention_predictions"].items() if row["arm"] == arm]
            diagnostics[arm] = {
                "intervention_count": len(cells),
                "choice_flip_count": sum(observation["choice"] != stages["test_predictions"][row["task_id"]]["choice"]
                                         for row, observation in cells),
            }
        report["mechanism"]["posthoc_behavioral_diagnostics"] = diagnostics
        report["mechanism"]["capability_specific_causal_mechanism_demonstrated"] = False
        report["metacognition"] = measured["metacognition"]
        report["prospective_study"] = {
            "root": str(prospective_root), "journal_tip_sha256": measured["journal_tip_sha256"],
            "independent_process_replay_verified": True,
            "prediction_count": independent["prediction_count"],
            "intervention_count": independent["intervention_count"],
            "limits": measured["limits"],
        }
        report["evaluation_mode"] = "retrospective_learning_audit_plus_verified_prospective_causal_and_monitoring_experiment"
        # Local causal response and external selection are narrower claims than
        # the full operational mechanism or internal metacognition.
        report["full_certification"] = False
        evidence.finish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "replay", "direct-replay", "smoke", "ir", "donor", "adapter-dir", "base-model", "model-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prospective-run", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite prior audit")
    require(not args.output.resolve().is_relative_to(ROOT), "evidence must remain outside source checkout")
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical(report))
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({key: report[key] for key in ("audit_complete", "full_certification", "evaluation_mode")}))
    print(json.dumps({key: report[key]["status"] for key in ("useful_learning", "mechanism", "metacognition")}))
    # 2 is a completed audit with unmet full-certification requirements.
    return 0 if report["full_certification"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError) as error:
        print("capability evidence rejected: " + str(error), file=sys.stderr)
        raise SystemExit(1)
