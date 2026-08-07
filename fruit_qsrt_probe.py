#!/usr/bin/env python3
"""Reproduce Fruit's QSRT permutation and activation adapter evidence on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from kquant.qsrt import coupled_intermediate_permutation

from checkpoint_contract import checkpoint_model_state

PINNED_CHECKPOINT_SHA256 = (
    "98ac7cb4f7799194424782b505d622069fecf4dbca5f5acb2658f2a66c3631f6"
)
HIDDEN = 1024
INTERMEDIATE = 512
EXPERTS = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint_model_state(payload)
    return state


def _expert_prefix(layer: int) -> str:
    return "mtp_block.mlp" if layer == 13 else f"layers.{layer}.mlp"


def _expert_matrix(
    state: dict[str, torch.Tensor], layer: int, expert: int, matrix: str
) -> torch.Tensor:
    prefix = _expert_prefix(layer)
    stacked_name = {
        "w1": "w_gate",
        "w3": "w_up",
        "w2": "w_down",
    }[matrix]
    stacked_key = f"{prefix}.{stacked_name}"
    if stacked_key in state:
        tensor = state[stacked_key][expert]
    else:
        projection = {
            "w1": "gate_proj",
            "w3": "up_proj",
            "w2": "down_proj",
        }[matrix]
        tensor = state[f"{prefix}.experts.{expert}.{projection}.weight"]
    expected = (HIDDEN, INTERMEDIATE) if matrix == "w2" else (INTERMEDIATE, HIDDEN)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{layer=}, {expert=}, {matrix=} has shape {tuple(tensor.shape)}, "
            f"expected {expected}"
        )
    return tensor.float()


def _expert_output(
    inputs: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation: str,
) -> torch.Tensor:
    gate = F.linear(inputs, w1)
    up = F.linear(inputs, w3)
    if activation == "silu":
        middle = F.silu(gate) * up
    elif activation == "situ":
        middle = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
            25.0 * torch.tanh(up / 25.0)
        )
    else:
        raise ValueError(f"unsupported activation: {activation}")
    return F.linear(middle, w2)


def run_probe(
    checkpoint: Path,
    *,
    expected_sha256: str,
    layers: tuple[int, ...],
    experts: tuple[int, ...],
    seed: int,
    tolerance: float,
) -> dict[str, object]:
    actual_sha256 = _sha256(checkpoint)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    state = _model_state(checkpoint)
    generator = torch.Generator().manual_seed(seed)
    closures: list[dict[str, float | int]] = []
    for layer in layers:
        if layer not in (*range(3, 13), 13):
            raise ValueError(f"unsupported Fruit expert layer: {layer}")
        for expert in experts:
            if not 0 <= expert < EXPERTS:
                raise ValueError(f"expert must be in 0..{EXPERTS - 1}: {expert}")
            w1 = _expert_matrix(state, layer, expert, "w1")
            w3 = _expert_matrix(state, layer, expert, "w3")
            w2 = _expert_matrix(state, layer, expert, "w2")
            inputs = torch.randn(2, HIDDEN, generator=generator)
            permutation = torch.randperm(INTERMEDIATE, generator=generator)
            reference = _expert_output(inputs, w1, w3, w2, activation="silu")
            permuted_w1, permuted_w3, permuted_w2 = coupled_intermediate_permutation(
                w1, w3, w2, permutation
            )
            actual = _expert_output(
                inputs,
                permuted_w1,
                permuted_w3,
                permuted_w2,
                activation="silu",
            )
            max_abs = float((actual - reference).abs().max())
            relative_to_output_max = max_abs / max(
                float(reference.abs().max()), torch.finfo(torch.float32).tiny
            )
            if max_abs > tolerance:
                raise AssertionError(
                    f"permutation closure failed for layer {layer}, "
                    f"expert {expert}: {max_abs} > {tolerance}"
                )
            closures.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "max_abs": max_abs,
                    "relative_to_output_max": relative_to_output_max,
                }
            )

    w1 = _expert_matrix(state, 3, 0, "w1")
    w3 = _expert_matrix(state, 3, 0, "w3")
    w2 = _expert_matrix(state, 3, 0, "w2")
    activation_inputs = torch.randn(2, HIDDEN, generator=generator)
    silu_output = _expert_output(activation_inputs, w1, w3, w2, activation="silu")
    situ_output = _expert_output(activation_inputs, w1, w3, w2, activation="situ")
    activation_delta = silu_output - situ_output
    activation_relative_l2 = float(
        activation_delta.norm() / silu_output.norm().clamp_min(1e-30)
    )
    activation_max_abs = float(activation_delta.abs().max())
    if not math.isfinite(activation_relative_l2) or activation_relative_l2 <= 0:
        raise AssertionError("SiLU/SiTU activation probe produced no finite delta")

    return {
        "kind": "fruit_qsrt_source_probe",
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha256,
        "seed": seed,
        "geometry": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": EXPERTS,
            "ordinary_layers": list(range(3, 13)),
            "mtp_layer": 13,
        },
        "permutation": {
            "cases": closures,
            "max_abs": max(case["max_abs"] for case in closures),
            "max_relative_to_output_max": max(
                case["relative_to_output_max"] for case in closures
            ),
            "tolerance": tolerance,
        },
        "activation": {
            "layer": 3,
            "expert": 0,
            "silu_vs_situ_relative_l2": activation_relative_l2,
            "silu_vs_situ_max_abs": activation_max_abs,
        },
    }


def _int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(","))
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256", default=PINNED_CHECKPOINT_SHA256)
    parser.add_argument("--layers", type=_int_tuple, default=(3, 12, 13))
    parser.add_argument("--experts", type=_int_tuple, default=(0, 255))
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_probe(
        args.checkpoint,
        expected_sha256=args.expected_sha256,
        layers=args.layers,
        experts=args.experts,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    print("FRUIT-QSRT-PROBE " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
