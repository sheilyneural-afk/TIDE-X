#!/usr/bin/env python3
"""V64: real Transformer micro-benchmark for TIDE-X receiver compilation.

This experiment intentionally does not download pretrained models. It trains two
small, architecturally different PyTorch Transformer encoders from independent
initializations, then learns eight operational skills as receiver-specific
operator coordinates. TIDE-X receives donor functional signatures and direct B
solutions for *other* skills; the held-out B solution is used only after
compilation as an oracle.

This is stronger than a purely linear weight-copy benchmark, but it is still a
micro-transformer experiment, not evidence of LLM-scale portability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - environment gate
    raise SystemExit(f"V64 rejected: torch unavailable: {exc}")

ROOT = Path(__file__).resolve().parents[2]
VOCAB = 32
SEQUENCE_LENGTH = 5
BASE_SEED = 1234
STATE_VECTORS = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0 ** -0.5, 2.0 ** -0.5],
        [2.0 ** -0.5, -(2.0 ** -0.5)],
    ],
    dtype=torch.float32,
)

SKILLS: list[tuple[str, list[list[float]]]] = [
    ("identity", [[1.0, 0.0], [0.0, 1.0]]),
    ("swap", [[0.0, 1.0], [1.0, 0.0]]),
    ("sign_x", [[-1.0, 0.0], [0.0, 1.0]]),
    ("sign_y", [[1.0, 0.0], [0.0, -1.0]]),
    ("rot90", [[0.0, -1.0], [1.0, 0.0]]),
    ("rotm90", [[0.0, 1.0], [-1.0, 0.0]]),
    ("contract_mix", [[0.8, 0.3], [-0.2, 0.9]]),
    ("mix", [[0.6, -0.4], [0.5, 0.7]]),
]


class StateTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, layers: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(VOCAB, d_model)
        self.position = nn.Parameter(torch.randn(SEQUENCE_LENGTH, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.state_readout = nn.Linear(d_model, 2)
        self.spec = {
            "d_model": d_model,
            "nhead": nhead,
            "layers": layers,
            "seed": seed,
        }

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens) + self.position.unsqueeze(0)
        hidden = self.encoder(hidden)
        return self.state_readout(self.norm(hidden[:, 0]))


@dataclass
class SkillArtifacts:
    name: str
    operator: torch.Tensor
    donor_coefficients: torch.Tensor
    receiver_direct_coefficients: torch.Tensor
    donor_functional_signature: list[float]
    donor_mse: float
    receiver_direct_mse: float


def make_batch(repeats: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tokens: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for state_id in range(4):
        for _ in range(repeats):
            context = torch.randint(
                8,
                VOCAB,
                (SEQUENCE_LENGTH - 1,),
                generator=generator,
            )
            tokens.append(torch.cat([torch.tensor([state_id + 1]), context]))
            states.append(STATE_VECTORS[state_id])
    return torch.stack(tokens), torch.stack(states)


def train_state_encoder(model: StateTransformer, seed: int) -> float:
    tokens, states = make_batch(48, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(tokens) - states) ** 2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mse = ((model(tokens) - states) ** 2).mean().item()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return mse


def make_operator_basis(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    orthogonal, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    return orthogonal.reshape(4, 2, 2).contiguous()


def materialize_operator(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("k,kij->ij", coefficients, basis)


def train_skill_coordinates(
    model: StateTransformer,
    basis: torch.Tensor,
    operator: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    tokens, states = make_batch(40, seed)
    coefficients = nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.Adam([coefficients], lr=0.08)
    target = states @ operator.T
    for _ in range(140):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            encoded = model(tokens)
        output = encoded @ materialize_operator(coefficients, basis).T
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    return coefficients.detach().clone()


def functional_signature(
    model: StateTransformer,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
) -> list[float]:
    tokens, _ = make_batch(32, 999)
    state_ids = torch.arange(tokens.shape[0]) // 32
    selected = state_ids < 2
    tokens = tokens[selected]
    state_ids = state_ids[selected]
    with torch.no_grad():
        output = model(tokens) @ materialize_operator(coefficients, basis).T
    signature: list[float] = []
    for state_id in range(2):
        signature.extend(output[state_ids == state_id].mean(0).tolist())
    return [float(value) for value in signature]


def behavior_mse(
    model: StateTransformer,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    operator: torch.Tensor,
    seed: int = 777,
) -> float:
    tokens, states = make_batch(64, seed)
    with torch.no_grad():
        output = model(tokens) @ materialize_operator(coefficients, basis).T
        target = states @ operator.T
        return float(((output - target) ** 2).mean().item())


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_digest(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(len(name).to_bytes(8, "big"))
        hasher.update(name.encode("utf-8"))
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
    return hasher.hexdigest()


def clean_git_identity() -> tuple[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status.strip():
        raise SystemExit("V64 rejected: working tree is not clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if len(commit) != 40 or len(tree) != 40:
        raise SystemExit("V64 rejected: git identity invalid")
    return commit, tree


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_tidex() -> Path:
    subprocess.run(
        ["cargo", "build", "--bin", "tidex", "--locked"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    candidate = Path("/tmp/tidex-cargo-target/debug/tidex")
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise SystemExit("V64 rejected: built tidex binary not found in canonical target")
    return candidate


def run() -> dict[str, object]:
    git_commit, git_tree = clean_git_identity()
    experiment_script_sha256 = file_sha256(Path(__file__).resolve())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(BASE_SEED)

    donor = StateTransformer(d_model=24, nhead=4, layers=1, seed=111)
    receiver = StateTransformer(d_model=32, nhead=4, layers=2, seed=222)
    donor_pretrain_mse = train_state_encoder(donor, 333)
    receiver_pretrain_mse = train_state_encoder(receiver, 444)
    donor_basis = make_operator_basis(555)
    receiver_basis = make_operator_basis(666)
    donor_frozen_digest = model_digest(donor)
    receiver_frozen_digest = model_digest(receiver)

    artifacts: list[SkillArtifacts] = []
    benchmark_cases: list[dict[str, object]] = []
    for index, (name, raw_operator) in enumerate(SKILLS):
        operator = torch.tensor(raw_operator, dtype=torch.float32)
        donor_coefficients = train_skill_coordinates(
            donor, donor_basis, operator, 1000 + index
        )
        receiver_coefficients = train_skill_coordinates(
            receiver, receiver_basis, operator, 2000 + index
        )
        signature = functional_signature(donor, donor_basis, donor_coefficients)
        donor_mse = behavior_mse(donor, donor_basis, donor_coefficients, operator)
        receiver_direct_mse = behavior_mse(
            receiver, receiver_basis, receiver_coefficients, operator
        )
        artifacts.append(
            SkillArtifacts(
                name=name,
                operator=operator,
                donor_coefficients=donor_coefficients,
                receiver_direct_coefficients=receiver_coefficients,
                donor_functional_signature=signature,
                donor_mse=donor_mse,
                receiver_direct_mse=receiver_direct_mse,
            )
        )
        benchmark_cases.append(
            {
                "skill_id": name,
                "functional_signature": signature,
                "direct_receiver_solution": receiver_coefficients.tolist(),
            }
        )

    donor_backbone_unchanged = model_digest(donor) == donor_frozen_digest
    receiver_backbone_unchanged = model_digest(receiver) == receiver_frozen_digest
    if not donor_backbone_unchanged or not receiver_backbone_unchanged:
        raise SystemExit("V64 rejected: frozen transformer backbone changed during skill training")

    benchmark_input = {
        "schema": "cerebro.tidex.receiver_portability_benchmark_input/v1",
        "ridge": 1e-8,
        "cases": benchmark_cases,
    }
    tidex = build_tidex()
    tidex_binary_sha256 = file_sha256(tidex)
    temporary = Path(tempfile.mkdtemp(prefix="tidex-v64-transformer."))
    try:
        input_path = temporary / "receiver-benchmark.json"
        input_path.write_text(json.dumps(benchmark_input), encoding="utf-8")
        completed = subprocess.run(
            [str(tidex), "benchmark", "portability", str(input_path)],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        receiver_report = json.loads(completed.stdout)

        # Runtime leakage check: alter only the held-out oracle for the first
        # skill and prove that its compiled receiver solution is unchanged.
        leak_input = json.loads(json.dumps(benchmark_input))
        leak_input["cases"][0]["direct_receiver_solution"] = [
            value * 1.05 + 0.01
            for value in leak_input["cases"][0]["direct_receiver_solution"]
        ]
        leak_path = temporary / "receiver-benchmark-leak-check.json"
        leak_path.write_text(json.dumps(leak_input), encoding="utf-8")
        leak_completed = subprocess.run(
            [str(tidex), "benchmark", "portability", str(leak_path)],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        leak_report = json.loads(leak_completed.stdout)
        heldout_oracle_leak_check = (
            receiver_report["cases"][0]["compiled_receiver_solution"]
            == leak_report["cases"][0]["compiled_receiver_solution"]
        )
        if not heldout_oracle_leak_check:
            raise SystemExit("V64 rejected: held-out receiver oracle changed compilation")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    by_name = {artifact.name: artifact for artifact in artifacts}
    rows: list[dict[str, object]] = []
    for row in receiver_report["cases"]:
        artifact = by_name[row["skill_id"]]
        compiled = torch.tensor(row["compiled_receiver_solution"], dtype=torch.float32)
        wrong = torch.tensor(row["wrong_receiver_solution"], dtype=torch.float32)
        virgin = torch.zeros_like(compiled)
        virgin_mse = behavior_mse(
            receiver, receiver_basis, virgin, artifact.operator
        )
        direct_mse = artifact.receiver_direct_mse
        compiled_mse = behavior_mse(
            receiver, receiver_basis, compiled, artifact.operator
        )
        wrong_mse = behavior_mse(receiver, receiver_basis, wrong, artifact.operator)
        denominator = virgin_mse - direct_mse
        if denominator <= 1e-12:
            raise SystemExit(
                f"V64 rejected: direct receiver oracle has no gain for {artifact.name}"
            )
        recovered_gain = (virgin_mse - compiled_mse) / denominator
        wrong_advantage = (wrong_mse - compiled_mse) / max(virgin_mse, 1e-12)
        rows.append(
            {
                "skill": artifact.name,
                "donor_mse": artifact.donor_mse,
                "receiver_virgin_mse": virgin_mse,
                "receiver_direct_mse": direct_mse,
                "receiver_tidex_mse": compiled_mse,
                "receiver_wrong_skill_mse": wrong_mse,
                "recovered_gain": recovered_gain,
                "correct_wrong_advantage": wrong_advantage,
                "rust_decoder_loo_r2": row["decoder_loo_r2"],
                "rust_encoder_loo_r2": row["encoder_loo_r2"],
                "rust_resolved": row["resolved"],
            }
        )

    mean_recovered_gain = sum(row["recovered_gain"] for row in rows) / len(rows)
    minimum_recovered_gain = min(row["recovered_gain"] for row in rows)
    mean_wrong_advantage = sum(row["correct_wrong_advantage"] for row in rows) / len(rows)
    maximum_direct_mse = max(row["receiver_direct_mse"] for row in rows)
    criteria = {
        "donor_pretrain_mse_max": 5e-4,
        "receiver_pretrain_mse_max": 5e-4,
        "receiver_direct_mse_max": 5e-4,
        "mean_recovered_gain_min": 0.95,
        "minimum_recovered_gain_min": 0.90,
        "mean_correct_wrong_advantage_min": 0.05,
    }
    passed = (
        donor_pretrain_mse <= criteria["donor_pretrain_mse_max"]
        and receiver_pretrain_mse <= criteria["receiver_pretrain_mse_max"]
        and maximum_direct_mse <= criteria["receiver_direct_mse_max"]
        and mean_recovered_gain >= criteria["mean_recovered_gain_min"]
        and minimum_recovered_gain >= criteria["minimum_recovered_gain_min"]
        and mean_wrong_advantage >= criteria["mean_correct_wrong_advantage_min"]
        and bool(receiver_report["all_resolved"])
    )
    return {
        "schema": "cerebro.tidex.v64_transformer_portability/v1",
        "version": "V64",
        "method": "held-out donor functional signature -> TIDE-X ReceiverCompiler -> receiver-native operator coordinates",
        "pass": passed,
        "scope": "micro-transformer cross-architecture portability; not an LLM-scale claim",
        "git_commit": git_commit,
        "git_tree": git_tree,
        "experiment_script_sha256": experiment_script_sha256,
        "tidex_binary_sha256": tidex_binary_sha256,
        "torch_version": torch.__version__,
        "deterministic_algorithms": True,
        "donor_parameter_vectors_supplied_to_tidex": False,
        "heldout_receiver_direct_solution_used_for_compilation": False,
        "heldout_oracle_leak_check_passed": heldout_oracle_leak_check,
        "shared_backbone_mutated_during_skill_compilation": False,
        "frozen_backbone_digest_check_passed": donor_backbone_unchanged
        and receiver_backbone_unchanged,
        "models": {
            "A": {
                **donor.spec,
                "parameter_count": parameter_count(donor),
                "pretrain_mse": donor_pretrain_mse,
                "frozen_digest_sha256": donor_frozen_digest,
                "receiver_coordinate_dimension": 4,
            },
            "B": {
                **receiver.spec,
                "parameter_count": parameter_count(receiver),
                "pretrain_mse": receiver_pretrain_mse,
                "frozen_digest_sha256": receiver_frozen_digest,
                "receiver_coordinate_dimension": 4,
            },
        },
        "skills": len(rows),
        "criteria": criteria,
        "means": {
            "recovered_gain": mean_recovered_gain,
            "minimum_recovered_gain": minimum_recovered_gain,
            "correct_wrong_advantage": mean_wrong_advantage,
            "maximum_direct_mse": maximum_direct_mse,
            "rust_mean_recovered_gain": receiver_report["mean_recovered_gain"],
            "rust_minimum_recovered_gain": receiver_report["minimum_recovered_gain"],
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if not args.output.is_absolute():
            raise SystemExit("V64 rejected: --output must be absolute")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
