"""Fail-closed train/serve convention handling for Fruit checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from numbers import Integral
from typing import Any


def checkpoint_model_state(
    payload: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Return a mutable model state from a raw or schema-wrapped checkpoint."""
    if not isinstance(payload, MutableMapping):
        raise TypeError("checkpoint payload must be a mutable mapping")
    if "model" not in payload:
        return payload
    state = payload["model"]
    if not isinstance(state, MutableMapping):
        raise TypeError("checkpoint model state must be a mutable mapping")
    return state


def checkpoint_conventions(
    state: MutableMapping[str, Any], environ: Mapping[str, str]
) -> tuple[bool, float]:
    """Consume paired convention markers and return native-layout flag/theta."""
    marker = state.pop("serve_conv_v", None)
    theta_tensor = state.pop("rope_theta_trained", None)
    serve_native = marker is not None
    if serve_native != (theta_tensor is not None):
        raise RuntimeError(
            "serve_conv_v and rope_theta_trained must either both be present "
            "or both be absent"
        )

    configured_theta = environ.get("FRUIT_ROPE_THETA")
    if serve_native:
        if marker.numel() != 1:
            raise RuntimeError("serve_conv_v must contain one value")
        marker_value = marker.reshape(-1)[0].item()
        if (
            isinstance(marker_value, bool)
            or not isinstance(marker_value, Integral)
            or marker_value != 2
        ):
            raise RuntimeError("unsupported serve_conv_v checkpoint marker")
        if theta_tensor.numel() != 1:
            raise RuntimeError("rope_theta_trained must contain one value")
        rope_theta = float(theta_tensor.reshape(-1)[0].item())
        if configured_theta and float(configured_theta) != rope_theta:
            raise RuntimeError(
                "FRUIT_ROPE_THETA disagrees with rope_theta_trained: "
                f"{configured_theta} != {rope_theta}"
            )
    else:
        if not configured_theta:
            raise RuntimeError(
                "legacy checkpoint has no trained theta marker; "
                "set FRUIT_ROPE_THETA explicitly"
            )
        rope_theta = float(configured_theta)

    if not math.isfinite(rope_theta) or rope_theta <= 0:
        raise RuntimeError(f"invalid RoPE theta: {rope_theta}")
    return serve_native, rope_theta


def set_rope_theta(config: MutableMapping[str, Any], rope_theta: float) -> None:
    """Write both config locations used by current and legacy consumers."""
    config["rope_theta"] = rope_theta
    rope_parameters = dict(config.get("rope_parameters") or {})
    rope_parameters["rope_theta"] = rope_theta
    config["rope_parameters"] = rope_parameters
