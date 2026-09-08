"""Boundary and leakage regressions for V69; no model output is simulated."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from quality.experiments import v69_target_update_free_compilation as v69
from quality.experiments.v68_receiver_response_probe import reproducibility_source_files


class ProtocolTests(unittest.TestCase):
    def calibration(self):
        rng = np.random.default_rng(20260908)
        latent = rng.normal(size=(13, 3))
        raw = latent @ rng.normal(size=(3, 9))
        deltas = latent @ rng.normal(size=(3, 12)) + rng.normal(size=(1, 12))
        return raw, deltas

    def test_heldout_receiver_solution_cannot_change_its_fold(self):
        raw, deltas = self.calibration()
        before = v69.calibration_fold(raw, deltas, 3)
        changed = deltas.copy()
        changed[3] = np.arange(deltas.shape[1]) * 1e9
        after = v69.calibration_fold(raw, changed, 3)
        self.assertNotIn(3, before["training_indices"])
        for field in ("axes", "coordinates", "functional_signatures", "requested"):
            np.testing.assert_array_equal(before[field], after[field])
        self.assertEqual(before["receiver_rank"], after["receiver_rank"])

    def test_heldout_donor_query_cannot_fit_the_projection(self):
        raw, deltas = self.calibration()
        before = v69.calibration_fold(raw, deltas, 4)
        raw[4] += np.arange(raw.shape[1]) * 1e6
        after = v69.calibration_fold(raw, deltas, 4)
        for field in ("axes", "coordinates", "functional_mean", "functional_components"):
            np.testing.assert_array_equal(before[field], after[field])
        self.assertFalse(np.array_equal(before["requested"], after["requested"]))

    def test_receiver_coordinates_preserve_physical_distances_and_angles(self):
        axes = np.asarray([[17.0, 2.0, 0.0, 1.0], [0.3, 1.0, 0.1, 0.0], [0.0, 0.2, 1.0, 0.0]])
        coordinates = np.asarray([[1.0, 5.0, -3.0], [1.0, -6.0, 2.0], [1.0, 1.0, 1.0]])
        physical = coordinates @ axes
        ortho, transformed = v69.orthonormal_receiver_basis(axes, coordinates)
        np.testing.assert_allclose(transformed @ ortho, physical, atol=1e-12, rtol=0)
        np.testing.assert_allclose(transformed @ transformed.T, physical @ physical.T, atol=1e-10, rtol=0)
        self.assertGreater(np.linalg.norm(coordinates @ coordinates.T - physical @ physical.T), 100)

    def test_rank_deficient_parameter_basis_is_rejected(self):
        with self.assertRaises(ValueError):
            v69.orthonormal_receiver_basis(np.asarray([[1.0, 0.0], [2.0, 0.0]]), np.eye(2))

    def test_cluster_inference_has_six_units_not_forty_eight(self):
        self.assertEqual(v69.capability_cluster_sign_flip_p([0.125] * 6), 2 / 64)
        self.assertEqual(v69.capability_cluster_sign_flip_p([0.0] * 6), 1.0)
        self.assertEqual(v69.capability_cluster_sign_flip_p([0.25, -0.25] * 3), 1.0)
        # Repeating questions does not create more independent checkpoints.
        self.assertEqual(v69.capability_cluster_sign_flip_p([1.0] * 6), 2 / 64)
        for gains in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                v69.capability_cluster_sign_flip_p(gains)

    def test_inconclusive_cluster_result_and_failed_gates_are_canonical_json(self):
        # A nonzero observed sum takes the enumeration branch. Its p=1
        # short-circuits a failed significance gate, which previously exposed
        # numpy.bool_ in the final receipt. Zero-sum tests alone missed it.
        for gains, expected_p in (([0.125, -0.125, 0.125, 0.0, 0.0, 0.0], 1.0),
                                  ([0.0] * 6, 1.0),
                                  ([0.125] * 6, 2 / 64)):
            with self.subTest(gains=gains):
                p_value = v69.capability_cluster_sign_flip_p(gains)
                self.assertIs(type(p_value), float)
                self.assertEqual(p_value, expected_p)
                significance_gate = p_value <= v69.CRITERIA["maximum_capability_cluster_two_sided_p"]
                baseline_gate = 0.0 >= v69.CRITERIA["minimum_compiled_vs_strongest_calibration_baseline_advantage"]
                self.assertIs(type(significance_gate), bool)
                self.assertIs(type(baseline_gate), bool)
                receipt = {
                    "complete": True,
                    "pass": significance_gate and baseline_gate,
                    "behavioral_transfer_gates_pass": significance_gate,
                    "incremental_baseline_gate_pass": baseline_gate,
                    "confirmation": {"paired": {"capability_cluster_exact_two_sided_p": p_value}},
                }
                encoded = v69.canonical(receipt)
                decoded = json.loads(encoded)
                self.assertEqual(decoded, receipt)
                self.assertIs(decoded["pass"], False)
                self.assertEqual(v69.canonical(decoded), encoded)

    def test_confirmation_and_preservation_splits_are_disjoint(self):
        self.assertFalse(set(v69.CONFIRMATION_THRESHOLDS) & set(v69.CALIBRATION_THRESHOLDS))
        self.assertFalse(set(v69.CONFIRMATION_THRESHOLDS) & set(v69.RETIRED_CONFIRMATION_THRESHOLDS))
        self.assertFalse(set(v69.GENERIC_CONTROLS) & set(v69.GENERIC_VALIDATION_CONTROLS))

    def test_freeze_contains_every_required_build_configuration(self):
        names = {str(path.relative_to(v69.PROJECT_ROOT)) for path in reproducibility_source_files()}
        self.assertTrue({"Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml",
                         "deny.toml", ".cargo/config.toml"}.issubset(names))
        self.assertIn("quality/experiments/v69_target_update_free_compilation.py", names)



    def test_resident_readout_resolves_separate_and_tied_qwen_parameters(self):
        # Real small Qwen architecture and real SafeTensors headers; no donor
        # outputs are fabricated and no full-size model is loaded or executed.
        with tempfile.TemporaryDirectory(prefix="v69-readout-storage-") as directory:
            path = Path(directory) / "model.safetensors"
            for tied, expected in ((False, "lm_head.weight"), (True, "model.embed_tokens.weight")):
                with self.subTest(tied=tied):
                    model = Qwen2ForCausalLM(Qwen2Config(
                        vocab_size=16, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, tie_word_embeddings=tied,
                    ))
                    stored = {"model.embed_tokens.weight": model.model.embed_tokens.weight.detach()}
                    if not tied:
                        stored["lm_head.weight"] = model.lm_head.weight.detach()
                    save_file(stored, str(path))
                    self.assertIs(model.get_output_embeddings().weight, model.lm_head.weight)
                    self.assertEqual(model.get_input_embeddings().weight is model.lm_head.weight, tied)
                    self.assertEqual(v69.resolve_resident_readout_tensor(model, path), expected)

    def test_resident_readout_rejects_ambiguity_false_tying_and_shape_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="v69-readout-rejections-") as directory:
            path = Path(directory) / "model.safetensors"
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=16, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2, tie_word_embeddings=True,
            ))
            # Two stored aliases could be loaded in conflicting orders: never
            # choose the first, even when these fixture bytes happen to agree.
            save_file({"model.embed_tokens.weight": model.model.embed_tokens.weight.detach(),
                       "lm_head.weight": model.lm_head.weight.detach().clone()}, str(path))
            with self.assertRaisesRegex(RuntimeError, "unambiguous resident"):
                v69.resolve_resident_readout_tensor(model, path)
            save_file({"model.embed_tokens.weight": model.model.embed_tokens.weight.detach()}, str(path))
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
            # Equal values and compatible shapes do not establish tied identity.
            with self.assertRaisesRegex(RuntimeError, "actual Parameter identity"):
                v69.resolve_resident_readout_tensor(model, path)
            model.config.tie_word_embeddings = False
            with self.assertRaisesRegex(RuntimeError, "unambiguous resident"):
                v69.resolve_resident_readout_tensor(model, path)
            save_file({"lm_head.weight": model.lm_head.weight[:-1].detach()}, str(path))
            with self.assertRaisesRegex(RuntimeError, "shape differs"):
                v69.resolve_resident_readout_tensor(model, path)

    def test_r4_reference_cannot_escape_or_relabel_changed_bytes(self):
        with tempfile.TemporaryDirectory(prefix="v69-readout-reference-") as directory:
            parent = Path(directory).resolve()
            root = parent / "original"
            root.mkdir()
            path = root / "evidence.json"
            data = v69.canonical({"observed": [1.0, -2.0]})
            path.write_bytes(data)
            reference = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
            self.assertEqual(v69.read_r4_reference(root, reference), {"observed": [1.0, -2.0]})
            path.write_bytes(v69.canonical({"observed": [1.0, 2.0]}))
            with self.assertRaisesRegex(RuntimeError, "reference changed"):
                v69.read_r4_reference(root, reference)
            path.unlink()
            outside = parent / "outside.json"
            outside.write_bytes(data)
            path.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "inside its original root"):
                v69.read_r4_reference(root, reference)
            with self.assertRaisesRegex(RuntimeError, "inside its original root"):
                v69.read_r4_reference(root, {**reference, "path": str(outside)})
            path.unlink()
            path.write_bytes(data)
            self.assertEqual(v69.read_r4_reference(root, reference), {"observed": [1.0, -2.0]})

    def test_readout_regression_cli_rejects_new_confirmation_and_replay_mix(self):
        with tempfile.TemporaryDirectory(prefix="v69-readout-cli-") as directory:
            root = Path(directory) / "original"
            destination = Path(directory) / "integration"
            for extra in (["--root", str(root)], ["--replay-model", str(root)],
                          ["--calibration-only"]):
                result = subprocess.run(
                    [os.sys.executable, str(Path(v69.__file__)),
                     "--verify-capability-ir-run", str(root), "--tidex", "/not-executed/tidex",
                     "--operational-output", str(destination), *extra],
                    capture_output=True, text=True, timeout=30,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("excludes run/replay modes", result.stderr)
                self.assertFalse(root.exists())
                self.assertFalse(destination.exists())

    def test_python_json_to_real_rust_compiler_and_closed_input(self):
        binary = os.environ.get("TIDEX_TEST_BINARY")
        self.assertIsNotNone(binary, "set TIDEX_TEST_BINARY to the actual freshly built tidex")
        rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0],
                [-1.0, 2.0], [0.5, 2.0], [-0.3, -0.4]]
        policy = v69.compiler_policy(100.0)
        policy.update(ridge=1e-10, maximum_functional_relative_error=1e-6)
        request = {
            "schema": "cerebro.tidex.receiver_signature_benchmark_input/v1",
            "requested": [0.2, -1.9670339420367746],
            "calibration": {"functional_signatures": rows, "receiver_solutions": rows,
                            "wrong_functional_signatures": [[-0.2, 1.9670339420367746]]},
            "protected_cortex": {"parameter_importance": [0.0, 0.0], "directions": [],
                                  "max_damage_ratio": 0.0},
            "risk_metric": [[1.0, 0.0], [0.0, 1.0]], "policy": policy,
            "proposal_method": "decode_then_project",
        }
        with tempfile.TemporaryDirectory(prefix="v69-protocol-test-") as directory:
            path = Path(directory) / "input.json"
            path.write_bytes(v69.canonical(request))
            env = {key: value for key, value in os.environ.items() if key not in ("TIDEX_HOME", "TIDEX_PRIVATE_ROOT")}
            completed = subprocess.run([binary, "benchmark", "response", str(path)],
                                       env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertTrue(report["allowed"])
            np.testing.assert_allclose(report["target_delta"], request["requested"], atol=1e-8, rtol=0)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            request["authorizes_promotion"] = True
            path.write_bytes(v69.canonical(request))
            rejected = subprocess.run([binary, "benchmark", "response", str(path)],
                                      env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("unknown field", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
