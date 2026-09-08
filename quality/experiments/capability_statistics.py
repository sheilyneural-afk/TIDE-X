"""Paired behavioral statistics shared by V66 replay and evidence auditing.

Inference is conditional on independent task pairs, not independent training
runs. Repeated outputs from one training seed do not establish seed robustness.
"""
from __future__ import annotations
import math


def validated_pass_rows(evaluation: dict) -> dict[int, bool]:
    rows = evaluation.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("paired evaluation requires nonempty results")
    values = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("paired evaluation row is not an object")
        task = row.get("task_id")
        outcome = row.get("pass")
        if type(task) is not int or type(outcome) is not bool or task in values:
            raise ValueError("paired evaluation identity/outcome invalid or duplicate")
        values[task] = outcome
    if "pass_rate" in evaluation:
        rate = evaluation["pass_rate"]
        if type(rate) not in (int, float) or not math.isfinite(rate) or not math.isclose(
            rate, sum(values.values()) / len(values), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("paired evaluation aggregate disagrees with rows")
    return values


def exact_paired_binomial_p(gained: int, lost: int) -> float:
    if type(gained) is not int or type(lost) is not int or min(gained, lost) < 0:
        raise ValueError("discordant counts must be nonnegative integers")
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    numerator = 2 * sum(math.comb(discordant, index) for index in range(min(gained, lost) + 1))
    return min(1.0, numerator / (2 ** discordant))


def paired_pass_counts(baseline: dict, compiled: dict) -> dict[str, int | float]:
    baseline_by_id = validated_pass_rows(baseline)
    compiled_by_id = validated_pass_rows(compiled)
    if baseline_by_id.keys() != compiled_by_id.keys():
        raise ValueError("paired evaluation identity mismatch")
    both = sum(baseline_by_id[k] and compiled_by_id[k] for k in baseline_by_id)
    gained = sum(not baseline_by_id[k] and compiled_by_id[k] for k in baseline_by_id)
    lost = sum(baseline_by_id[k] and not compiled_by_id[k] for k in baseline_by_id)
    return {
        "both_pass": both,
        "compiled_only_pass": gained,
        "virgin_only_pass": lost,
        "neither_pass": len(baseline_by_id) - both - gained - lost,
        "discordant_count": gained + lost,
        "exact_two_sided_p": exact_paired_binomial_p(gained, lost),
    }


def paired_gain_interval(baseline: dict, compiled: dict, alpha: float = 0.05) -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    pairs = paired_pass_counts(baseline, compiled)
    count = len(baseline["results"])
    gain = (pairs["compiled_only_pass"] - pairs["virgin_only_pass"]) / count
    # Each independent paired difference lies in [-1, 1]. Hoeffding's
    # two-sided bound: P(|mean-Emean| >= e) <= 2 exp(-n e^2 / 2).
    radius = math.sqrt(2 * math.log(2 / alpha) / count)
    return {
        "mean": gain, "lower": max(-1.0, gain - radius),
        "upper": min(1.0, gain + radius), "confidence": 1 - alpha,
        "method": "two_sided_hoeffding_independent_task_pairs",
        "assumption": "independent task pairs; does not cover selection bias, task clustering or training-seed variation",
    }
