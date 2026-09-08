"""Adversarial checks for the evidence boundary, without model inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from quality.experiments.capability_statistics import (
    exact_paired_binomial_p, paired_gain_interval, paired_pass_counts,
)
from quality.experiments.certify_capability_evidence import Evidence, unique_object


def outcomes(values):
    return {"results": [{"task_id": i, "pass": value} for i, value in enumerate(values)],
            "pass_rate": sum(values) / len(values)}


class EvidenceTests(unittest.TestCase):
    def test_exact_known_discordance_and_symmetry(self):
        self.assertEqual(exact_paired_binomial_p(38, 1), 1.4551915228366852e-10)
        self.assertEqual(exact_paired_binomial_p(1, 38), exact_paired_binomial_p(38, 1))
        self.assertEqual(exact_paired_binomial_p(4, 3), 1.0)
        self.assertEqual(exact_paired_binomial_p(0, 0), 1.0)

    def test_pairing_uses_identity_not_order(self):
        base = outcomes([False, True, False])
        learned = outcomes([True, False, True])
        first = paired_pass_counts(base, learned)
        learned["results"].reverse()
        self.assertEqual(first, paired_pass_counts(base, learned))
        self.assertEqual(first["compiled_only_pass"], 2)
        self.assertEqual(first["virgin_only_pass"], 1)

    def test_duplicate_ids_do_not_collapse_silently(self):
        base = outcomes([False, True])
        bad = copy.deepcopy(base)
        bad["results"][1]["task_id"] = 0
        with self.assertRaises(ValueError):
            paired_pass_counts(base, bad)

    def test_string_bool_is_rejected(self):
        base = outcomes([False, True])
        bad = copy.deepcopy(base)
        bad["results"][0]["pass"] = "false"
        with self.assertRaises(ValueError):
            paired_pass_counts(base, bad)

    def test_forged_aggregate_is_rejected(self):
        base = outcomes([False, True])
        bad = copy.deepcopy(base)
        bad["pass_rate"] = 1.0
        with self.assertRaises(ValueError):
            paired_pass_counts(base, bad)

    def test_missing_task_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_pass_counts(outcomes([False, True]), outcomes([True]))

    def test_uncertainty_does_not_disappear_with_identical_observations(self):
        bound = paired_gain_interval(outcomes([False] * 10), outcomes([True] * 10))
        self.assertLess(bound["lower"], 1.0)
        self.assertEqual(bound["upper"], 1.0)

    def test_json_duplicates_and_nan_fail(self):
        with self.assertRaises(ValueError):
            json.loads('{"pass":false,"pass":true}', object_pairs_hook=unique_object)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text('{"value":NaN}')
            with self.assertRaises(ValueError):
                Evidence().read(path)

    def test_changed_evidence_and_symlink_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text('{"value":1}')
            evidence = Evidence()
            evidence.read(path)
            path.write_text('{"value":2}')
            with self.assertRaises(ValueError):
                evidence.finish()
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                Evidence().read(link)


if __name__ == "__main__":
    unittest.main()
