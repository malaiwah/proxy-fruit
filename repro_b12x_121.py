#!/usr/bin/env python3
"""Reproduce local-inference-lab/b12x#121 with GLM-5.2-SIQ-Fruit geometry:
fused mHC norm at the public API's default norm_eps=0.0 emits NaN for a
zero activation. Control arm passes Fruit's real rms_norm_eps (1e-5).

norm_weight is Fruit's actual trained model.layers.0.input_layernorm.weight
(hidden=512) read from the published pilot checkpoint.
"""
import json
import struct

import torch
from sparkinfer.norm.mhc._impl import MHC_MULT
from sparkinfer.norm.mhc._kernels import run_mhc_pre_functional

CKPT = "/mnt/vault/llm/fruit-pilot/output/GLM-5.2-SIQ-Fruit-pilot"


def load_norm_weight():
    path = f"{CKPT}/model-layer-000.safetensors"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header["model.layers.0.input_layernorm.weight"]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
    return t.reshape(meta["shape"])


def run(norm_eps, label):
    torch.manual_seed(20260805)
    # Fused Gram kernel demands hidden %% 1024 == 0 and sinkhorn_iters=20
    # (pilot finding: hidden 512 cannot enter the mHC path; final Fruit at
    # 1024 will natively). Fruit norm weight tiled x2 to 1024.
    hidden = 4096   # mHC kernels support only {4096, 7168}
    tokens = 8
    hc_mult3 = 24   # validator: fn must be (24, hidden)
    residual = torch.zeros(tokens, hidden, dtype=torch.bfloat16,
                           device="cuda")         # zero activation (issue's
    fn = torch.randn(hc_mult3, hidden,  # minimal trigger
                     dtype=torch.float32, device="cuda") * 0.02
    hc_scale = torch.ones(3, dtype=torch.float32, device="cuda")
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device="cuda")
    w = load_norm_weight()
    norm_weight = w.repeat(8).to("cuda")   # Fruit 512 -> 4096
    outs = run_mhc_pre_functional(
        residual=residual, fn=fn, scale=hc_scale, bias=hc_base,
        rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=0.0 if norm_eps is None else norm_eps)
    nan_report = []
    for i, t in enumerate(outs):
        if torch.is_tensor(t) and t.is_floating_point():
            nan_report.append(f"out[{i}] nan={int(t.isnan().sum())}"
                              f"/{t.numel()}")
    print(f"[{label}] norm_eps="
          f"{'<API default>' if norm_eps is None else norm_eps}: "
          + "  ".join(nan_report), flush=True)
    return any(torch.is_tensor(t) and t.is_floating_point()
               and bool(t.isnan().any()) for t in outs)


def main():
    bad = run(None, "issue arm")      # API default norm_eps=0.0
    good = run(1e-6, "control arm")   # explicit positive epsilon
    print()
    if bad and not good:
        print("B12X-121-REPRODUCED: default norm_eps=0.0 emits NaN on zero "
              "activation; explicit 1e-5 (the GLM/Fruit epsilon) is clean.")
    else:
        print(f"B12X-121-NOT-REPRODUCED (issue arm nan={bad}, "
              f"control nan={good})")


if __name__ == "__main__":
    main()
