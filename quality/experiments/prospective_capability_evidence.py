"""Prospective causal/monitoring evidence: ordered records and independent scoring."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
from quality.experiments.capability_statistics import exact_paired_binomial_p
from quality.experiments.certify_capability_evidence import canonical, require, unique_object

SCHEMA = "cerebro.tidex.prospective_capability_experiment/v1"


class Journal:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.previous = "0" * 64
        self.index = 0

    def append(self, kind, payload):
        record = {"index": self.index, "previous_sha256": self.previous, "kind": kind, "payload": payload}
        data = canonical(record)
        path = self.root / f"{self.index:06d}.json"
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        self.previous = hashlib.sha256(data).hexdigest()
        self.index += 1
        return self.previous


def read_journal(root):
    previous, result = "0" * 64, []
    for index, path in enumerate(sorted(Path(root).glob("*.json"))):
        require(path.name == f"{index:06d}.json" and not path.is_symlink(), "journal sequence/path invalid")
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=unique_object)
        require(canonical(value) == raw, "noncanonical journal record")
        require(value["index"] == index and value["previous_sha256"] == previous, "journal chain mismatch")
        previous = hashlib.sha256(raw).hexdigest()
        result.append(value)
    require(result, "empty journal")
    return result, previous


def confidence_bin(q):
    require(type(q) in (float, int) and math.isfinite(q) and 0.5 <= q <= 1, "invalid confidence")
    return min(4, int((q - 0.5) * 10))


def fit_monitor(predictions, answers):
    require(set(predictions) == set(answers), "calibration identity mismatch")
    counts, hits = [0] * 5, [0] * 5
    for key, row in predictions.items():
        b = confidence_bin(row["raw_confidence"])
        counts[b] += 1
        hits[b] += int(row["choice"] == answers[key])
    return {
        "method": "five_fixed_bins_laplace",
        "counts": counts, "hits": hits,
        "probabilities": [(h + 1) / (n + 2) for h, n in zip(hits, counts)],
        "prior": (sum(hits) + 1) / (sum(counts) + 2),
    }


def choose_actions(predictions, monitor, seed):
    ids = sorted(predictions)
    require(len(ids) >= 4, "too few monitoring tasks")
    confidence = {key: monitor["probabilities"][confidence_bin(predictions[key]["raw_confidence"])] for key in ids}
    budget = len(ids) // 2
    # Tie order and controls are outcome-independent and reproducible.
    rng = random.Random(seed)
    tie_order = ids.copy()
    rng.shuffle(tie_order)
    tie_rank = {key: index for index, key in enumerate(tie_order)}
    on = sorted(ids, key=lambda key: (-confidence[key], tie_rank[key]))[:budget]
    off = rng.sample(ids, budget)
    permuted = ids.copy()
    rng.shuffle(permuted)
    shuffled_confidence = {key: confidence[source] for key, source in zip(ids, permuted)}
    shuffled = sorted(ids, key=lambda key: (-shuffled_confidence[key], tie_rank[key]))[:budget]
    return {"confidence": confidence, "selected": {"on": on, "off": off, "shuffled": shuffled},
            "budget": budget, "seed": seed}


def score_monitor(predictions, answers, monitor, actions):
    require(set(predictions) == set(answers) == set(actions["confidence"]), "monitor test identity mismatch")
    correct = {k: int(predictions[k]["choice"] == answers[k]) for k in predictions}
    n = len(correct)
    brier = sum((actions["confidence"][k] - correct[k]) ** 2 for k in correct) / n
    prior_brier = sum((monitor["prior"] - y) ** 2 for y in correct.values()) / n
    budget = actions["budget"]
    selected = actions["selected"]
    for ids in selected.values():
        require(len(ids) == budget and len(set(ids)) == budget and set(ids) <= set(correct),
                "unequal/invalid action budget")
    utilities = {arm: sum(correct[k] for k in ids) / budget for arm, ids in selected.items()}
    comparisons = {}
    for arm in ("off", "shuffled"):
        on_values = {k: int(k in selected["on"]) * correct[k] for k in correct}
        other = {k: int(k in selected[arm]) * correct[k] for k in correct}
        gained = sum(on_values[k] > other[k] for k in correct)
        lost = sum(on_values[k] < other[k] for k in correct)
        comparisons[arm] = {"gain": utilities["on"] - utilities[arm],
                            "paired_p": exact_paired_binomial_p(gained, lost)}
    # Confidence ranking diagnostic, with ties scored as one half.
    positives = [actions["confidence"][k] for k in correct if correct[k]]
    negatives = [actions["confidence"][k] for k in correct if not correct[k]]
    auc = None if not positives or not negatives else sum(
        (a > b) + 0.5 * (a == b) for a in positives for b in negatives
    ) / (len(positives) * len(negatives))
    return {"task_count": n, "base_accuracy": sum(correct.values()) / n, "brier": brier,
            "constant_prior_brier": prior_brier, "brier_improvement": prior_brier - brier,
            "success_ranking_auc": auc, "selection_budget": budget,
            "selected_accuracy": utilities, "comparisons": comparisons,
            "utility_definition": "correct accepted decisions per fixed acceptance budget; one shared inference per case"}


def score_mechanism(predictions, observations):
    require(set(predictions) == set(observations) and predictions, "intervention identity mismatch")
    predicted, observed = [], []
    by_arm = {"learned_delta": [], "permuted_delta": []}
    for key, prediction in predictions.items():
        observation = observations[key]
        require(observation["prediction_sha256"] == hashlib.sha256(canonical(prediction)).hexdigest(),
                "intervention prediction binding mismatch")
        delta = observation["margin"] - prediction["baseline_margin"]
        estimate = prediction["predicted_change"]
        require(all(math.isfinite(x) for x in [delta, estimate]), "nonfinite intervention response")
        predicted.append(estimate)
        observed.append(delta)
        by_arm[prediction["arm"]].append(delta)
    energy = sum(x*x for x in observed)
    error = sum((a-b)**2 for a,b in zip(predicted, observed))
    rms = math.sqrt(energy / len(observed))
    relative = math.sqrt(error / energy) if energy > 0 else None
    sign_count = sum(a*b > 0 for a,b in zip(predicted, observed) if abs(b) >= 1e-6)
    sign_total = sum(abs(b) >= 1e-6 for b in observed)
    return {"interventions": len(observed), "observed_change_rms": rms,
            "relative_prediction_rmse": relative,
            "zero_prediction_rmse": rms,
            "prediction_rmse": math.sqrt(error / len(observed)),
            "sign_accuracy": sign_count / sign_total if sign_total else None,
            "arm_response_rms": {k: math.sqrt(sum(x*x for x in v)/len(v)) for k,v in by_arm.items() if v}}


def verify_run(root):
    root = Path(root)
    records, tip = read_journal(root / "events")
    def only(kind):
        values = [r["payload"] for r in records if r["kind"] == kind]
        require(len(values) == 1, "missing/duplicate stage " + kind)
        return values[0]
    plan = only("precommit")
    require(plan["schema"] == SCHEMA, "prospective schema mismatch")
    expected = ["precommit", "calibration_predictions", "calibration_answers", "monitor_frozen",
                "test_predictions", "actions_committed", "intervention_predictions",
                "intervention_observations", "test_answers", "completion"]
    require([r["kind"] for r in records] == expected, "invalid stage order")
    for relative, digest in plan["source_sha256"].items():
        path = root / "source" / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "source reference escapes snapshot")
        require(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest, "frozen source mismatch")
    benchmark_path = root / "benchmark.json"
    require(not benchmark_path.is_symlink() and hashlib.sha256(benchmark_path.read_bytes()).hexdigest() == plan["benchmark_sha256"], "benchmark binding mismatch")
    benchmark = json.loads(benchmark_path.read_bytes(), object_pairs_hook=unique_object)
    benchmark_cases = {row["task_id"]: row for row in benchmark["cases"]}
    require(len(benchmark_cases) == len(benchmark["cases"]), "duplicate benchmark task")
    calibration = only("calibration_predictions")
    calibration_answers = only("calibration_answers")
    predictions = only("test_predictions")
    answers = only("test_answers")
    require(not set(calibration) & set(predictions), "calibration/test overlap")
    require(set(calibration) == set(plan["calibration_ids"]) and set(predictions) == set(plan["test_ids"]),
            "planned task identity mismatch")
    require(not (set(calibration) | set(predictions)) & set(plan["excluded_ids"]), "previous task reused")
    for rows in (calibration, predictions):
        for row in rows.values():
            require(type(row["choice"]) is int and row["choice"] in (0, 1), "invalid binary decision")
            confidence_bin(row["raw_confidence"])
    for rows in (calibration_answers, answers):
        require(all(type(v) is int and v in (0,1) for v in rows.values()), "invalid answer")
    for rows in (calibration, predictions):
        for key, row in rows.items():
            require(row["prompt_sha256"] == hashlib.sha256(benchmark_cases[key]["prompt"].encode()).hexdigest(), "prompt binding mismatch")
    require({**calibration_answers, **answers} == benchmark["answers"], "answers differ from sealed benchmark")
    monitor = only("monitor_frozen")
    require(monitor == fit_monitor(calibration, calibration_answers), "monitor not fitted from committed calibration")
    actions = only("actions_committed")
    require(actions == choose_actions(predictions, monitor, plan["seed"]), "actions disagree with frozen policy")
    causal_predictions = only("intervention_predictions")
    causal_observations = only("intervention_observations")
    expected_count = len(plan["causal_ids"]) * len(plan["sites"]) * 2 * len(plan["strengths"])
    require(len(causal_predictions) == expected_count, "incomplete intervention grid")
    grid = {(task, site, arm, strength) for task in plan["causal_ids"] for site in plan["sites"]
            for arm in ("learned_delta","permuted_delta") for strength in plan["strengths"]}
    seen = set()
    from safetensors.numpy import load_file
    import numpy as np
    trace_cache = {}
    for row in causal_predictions.values():
        cell = (row["task_id"], row["site"], row["arm"], row["strength"])
        require(cell in grid and cell not in seen, "invalid/duplicate intervention cell")
        seen.add(cell)
        relative = Path(row["trace_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "trace path escapes run")
        trace_path = root / relative
        require(trace_path.is_file() and not trace_path.is_symlink(), "trace missing/nonregular")
        if str(relative) not in trace_cache:
            require(hashlib.sha256(trace_path.read_bytes()).hexdigest() == row["trace_sha256"], "trace digest mismatch")
            arrays = load_file(str(trace_path))
            require(set(arrays) == {"gradient", "learned_delta", "permuted_delta"}, "trace channels invalid")
            require(all(np.isfinite(a).all() for a in arrays.values()), "trace nonfinite")
            require(np.array_equal(np.roll(arrays["learned_delta"], plan["channel_roll"], axis=-1), arrays["permuted_delta"]), "control is not norm-matched channel permutation")
            trace_cache[str(relative)] = (row["trace_sha256"], arrays)
        trace_sha, arrays = trace_cache[str(relative)]
        require(trace_sha == row["trace_sha256"], "inconsistent trace identity")
        derivative = float(np.sum(arrays["gradient"].astype(np.float64)*arrays[row["arm"]].astype(np.float64)))
        require(math.isclose(-row["strength"]*derivative, row["predicted_change"], rel_tol=1e-8, abs_tol=1e-8), "prediction differs from measured gradient and intervention")
        require(math.isclose(row["baseline_margin"], predictions[row["task_id"]]["margin"], rel_tol=1e-5, abs_tol=1e-4), "intervention baseline differs from original forward")
    require(seen == grid, "intervention grid missing cells")
    completion = only("completion")
    require(completion["weights_unchanged"] is True and completion["hooks_removed"] is True,
            "runtime not restored")
    mechanism = score_mechanism(causal_predictions, causal_observations)
    meta = score_monitor(predictions, answers, monitor, actions)
    criteria = plan["criteria"]
    mechanism_gates = {
        "nontrivial_effect": mechanism["observed_change_rms"] >= criteria["minimum_effect_rms"],
        "predictive_fidelity": mechanism["relative_prediction_rmse"] is not None
            and mechanism["relative_prediction_rmse"] <= criteria["maximum_relative_prediction_rmse"],
        "direction": mechanism["sign_accuracy"] is not None and mechanism["sign_accuracy"] >= criteria["minimum_sign_accuracy"],
    }
    meta_gates = {
        "calibration": meta["brier_improvement"] >= criteria["minimum_brier_improvement"],
        "failure_ranking": meta["success_ranking_auc"] is not None and meta["success_ranking_auc"] >= criteria["minimum_auc"],
        **{f"utility_vs_{arm}": value["gain"] >= criteria["minimum_selection_gain"]
           and value["paired_p"] <= criteria["maximum_p"] for arm,value in meta["comparisons"].items()},
    }
    return {"schema": SCHEMA, "complete": True, "journal_tip_sha256": tip,
            "precommit_sha256": hashlib.sha256(canonical(plan)).hexdigest(),
            "mechanism": {"status": "SUPPORTED_LOCAL_CAUSAL_RESPONSE" if all(mechanism_gates.values()) else "CRITERIA_FAILED",
                          "metrics": mechanism, "gates": mechanism_gates,
                          "scope": "first-order prediction of interventions removing/amplifying learned LoRA contributions at three internal v_proj sites",
                          "full_operational_algorithm_demonstrated": False},
            "metacognition": {"status": "SUPPORTED_EXTERNAL_SELECTIVE_MONITORING" if all(meta_gates.values()) else "CRITERIA_FAILED",
                             "metrics": meta, "gates": meta_gates,
                             "scope": "calibrated external supervisor selects binary code-correctness decisions; same acceptance/inference budget",
                             "internal_model_metacognition_demonstrated": False,
                             "adaptive_repair_or_degradation_detection_demonstrated": False},
            "limits": ["MBPP correct-versus-mutated-code discrimination, not free code generation",
                       "one checkpoint, seed, split and fixed intervention scale",
                       "local response fidelity is not an extracted algorithm or unique causal explanation",
                       "hash chain records in-process ordering, not independent timestamp attestation",
                       "test significance assumes independent task pairs; family clustering unmeasured"],
            "authorizes_promotion": False}
