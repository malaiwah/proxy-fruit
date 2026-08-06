#!/usr/bin/env python3
"""Export the trained Fruit pilot to a GLM-5.2-style SIQ checkpoint:
non-expert tensors BF16 with parent names, routed experts Trellis-encoded
(per-expert rank0 layout, K4 for experts 0..95 / K3 for 96..255, MTP layer
uniform K3 — mirroring parent tier ratios), tier_bitmap + hybrid_tr3_tail
config so the r25/r28 SIQ loader consumes it like any GLM checkpoint.
"""
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, "/tools")
import glm_franken as gf

import os
PT = Path(os.environ.get("FRUIT_PT",
    "/mnt/vault/llm/fruit-pilot/fruit_pilot.pt"))
SRC = Path("/mnt/vault/llm/glm52-franken/src")
OUT = Path(os.environ.get(
    "FRUIT_OUT", "/mnt/vault/llm/fruit-pilot/output/GLM-5.2-SIQ-Fruit-pilot"))
TIERS = os.environ.get("FRUIT_TIERS", "mixed")   # mixed | k3 | k4
# Zero-pad the expert/shared FFN inter dim (exactness-preserving: padded
# gate/up rows yield silu(0)*0=0; padded down cols consume those zeros).
# Needed because trellis3_t256_proj requires FC1 % 256 == 0.
PAD_INTER = int(os.environ.get("FRUIT_PAD_INTER", "0"))

_e = os.environ.get
H = int(_e("GEO_H", "512"))
NL = int(_e("GEO_NL", "6"))
FIRST_MOE, N_EXP = 3, 256
GEO_HEADS = int(_e("GEO_HEADS", "8"))
GEO_QLORA = int(_e("GEO_QLORA", "512"))
GEO_DENSE_INTER = int(_e("GEO_DENSE_INTER", "1024"))
GEO_MOE_INTER = int(_e("GEO_MOE_INTER", "128"))
MTP_LAYER = NL  # layer index 6
PROJS = ("gate_proj", "up_proj", "down_proj")


def tier_of(layer: int, e: int) -> int:
    if TIERS == "k3":
        return 3
    if TIERS == "k4":
        return 3 if layer == MTP_LAYER else 4
    if layer == MTP_LAYER:
        return 3                       # parent mtp78: uniform K3
    return 4 if e < 96 else 3          # 96 K4 + 160 K3 (parent ratio)


def main() -> None:
    import torch
    gf.load_base(Path("/tools/encode_tr3_v31.py"))
    # Re-point the encoder's GLM geometry at Fruit's micro dims (the
    # hadamard/tile constraints still hold: both are multiples of 128).
    gf.BASE.HIDDEN = H
    gf.BASE.MOE_INTER = PAD_INTER or GEO_MOE_INTER
    gf.BASE.SLICE = gf.BASE.MOE_INTER
    sd = torch.load(PT, map_location="cpu")
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"output must be empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    weight_map, total = {}, 0

    def write_shard(fname, entries):
        nonlocal total
        # Alphabetical order matters: the SIQ loader resolves each expert's
        # mcg when its trellis arrives, so mcg must precede trellis (as in
        # Legume's assembled shards).
        entries = sorted(entries, key=lambda e: e[0])
        gf.BASE.write_safetensors(str(OUT / fname), entries,
                                  metadata={"format": "pt"})
        for name, _, _, raw in entries:
            weight_map[name] = fname
            total += len(raw)

    def pad_ffn(t, name):
        if not PAD_INTER:
            return t
        if name.endswith(("gate_proj.weight", "up_proj.weight")) \
                and "experts" in name and t.shape[0] < PAD_INTER:
            pad = torch.zeros(PAD_INTER - t.shape[0], t.shape[1], dtype=t.dtype)
            return torch.cat([t, pad], 0)
        if name.endswith("down_proj.weight") and "experts" in name \
                and t.shape[1] < PAD_INTER:
            pad = torch.zeros(t.shape[0], PAD_INTER - t.shape[1], dtype=t.dtype)
            return torch.cat([t, pad], 1)
        return t

    def bf16(name_out, key):
        t = sd[key].to(torch.bfloat16).contiguous()
        t = pad_ffn(t, name_out).contiguous()
        return (name_out, "BF16", tuple(t.shape),
                t.view(torch.uint16).numpy().tobytes())

    write_shard("model-embed.safetensors",
                [bf16("model.embed_tokens.weight", "embed_tokens.weight")])
    write_shard("model-head.safetensors",
                [bf16("lm_head.weight", "lm_head.weight"),
                 bf16("model.norm.weight", "norm.weight")])

    mcg_bytes = struct.pack("<I", gf.BASE.MCG_MULT)
    tier = {}
    for li in range(NL + 1):           # 0..5 + MTP layer 6
        is_mtp = (li == MTP_LAYER)
        pfx_in = "mtp_block." if is_mtp else f"layers.{li}."
        pfx_out = f"model.layers.{li}."
        entries = []

        def add(short_out, short_in=None):
            entries.append(bf16(pfx_out + short_out,
                                pfx_in + (short_in or short_out)))

        for n in ("input_layernorm.weight", "post_attention_layernorm.weight",
                  "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
                  "self_attn.q_b_proj.weight",
                  "self_attn.kv_a_proj_with_mqa.weight",
                  "self_attn.kv_a_layernorm.weight",
                  "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight",
                  "self_attn.indexer.wk.weight",
                  "self_attn.indexer.wq_b.weight",
                  "self_attn.indexer.weights_proj.weight",
                  "self_attn.indexer.k_norm.weight",
                  "self_attn.indexer.k_norm.bias"):
            add(n)
        if is_mtp:
            entries.append(bf16(pfx_out + "eh_proj.weight", "mtp_eh_proj.weight"))
            entries.append(bf16(pfx_out + "enorm.weight", "mtp_enorm.weight"))
            entries.append(bf16(pfx_out + "hnorm.weight", "mtp_hnorm.weight"))
            entries.append(bf16(pfx_out + "shared_head.norm.weight",
                                "norm.weight"))
        if li < FIRST_MOE:
            for n in ("mlp.gate_proj.weight", "mlp.up_proj.weight",
                      "mlp.down_proj.weight"):
                add(n)
        else:
            add("mlp.gate.weight", "mlp.gate.weight")
            entries.append(bf16(pfx_out + "mlp.gate.e_score_correction_bias",
                                pfx_in + "mlp.e_score_correction_bias"))
            for n in ("mlp.shared_experts.gate_proj.weight",
                      "mlp.shared_experts.up_proj.weight",
                      "mlp.shared_experts.down_proj.weight"):
                add(n)
            t0 = time.time()
            ks, nmses = [], []
            for e in range(N_EXP):
                bits = tier_of(li, e)
                ks.append(bits)
                enm = []
                stacked = {"gate_proj": "w_gate", "up_proj": "w_up",
                           "down_proj": "w_down"}
                for proj in PROJS:
                    per_exp = f"{pfx_in}mlp.experts.{e}.{proj}.weight"
                    if per_exp in sd:
                        w = sd[per_exp]
                    else:   # grouped-GEMM trainer stores stacked [E, out, in]
                        w = sd[f"{pfx_in}mlp.{stacked[proj]}"][e]
                    if PAD_INTER:
                        w = pad_ffn(w.to(torch.bfloat16),
                                    f"experts.{proj}.weight")
                    out, stats = gf.encode_full_matrix(
                        torch, w.to(torch.bfloat16), proj, li, e, bits)
                    base = f"{pfx_out}mlp.experts.{e}.{proj}.rank0"
                    entries.append((f"{base}.suh", "F16",
                                    tuple(out["suh"].shape),
                                    out["suh"].numpy().tobytes()))
                    entries.append((f"{base}.svh", "F16",
                                    tuple(out["svh"].shape),
                                    out["svh"].numpy().tobytes()))
                    entries.append((f"{base}.trellis", "I16",
                                    tuple(out["trellis"].shape),
                                    out["trellis"].numpy().tobytes()))
                    entries.append((f"{base}.mcg", "I32", (), mcg_bytes))
                    enm.append(stats["nmse"])
                nmses.append(sum(enm) / len(enm))
            tier[str(li)] = {"k": ks, "keep_nvfp4": [],
                             "tail_tr3": list(range(N_EXP)),
                             "expert_rel_rt_mse": nmses}
            print(f"layer {li}: 256 experts encoded in {time.time()-t0:.0f}s",
                  flush=True)
        write_shard(f"model-layer-{li:03d}.safetensors", entries)
        print(f"layer {li} written", flush=True)

    src_cfg = json.loads((SRC / "config.json").read_text())
    cfg = dict(src_cfg)
    cfg.update({
        "hidden_size": H, "intermediate_size": GEO_DENSE_INTER,
        "moe_intermediate_size": PAD_INTER or GEO_MOE_INTER,
        "num_attention_heads": GEO_HEADS,
        "num_key_value_heads": GEO_HEADS, "q_lora_rank": GEO_QLORA,
        "num_hidden_layers": NL, "num_nextn_predict_layers": 1,
        "max_position_embeddings": 65536,
        **({"rope_theta": float(os.environ["FRUIT_ROPE_THETA"])}
           if os.environ.get("FRUIT_ROPE_THETA") else {}),
    })
    for key, value in list(cfg.items()):
        if isinstance(value, list) and len(value) in (78, 79):
            # 78-length lists cover hidden layers only (6 entries); 79-length
            # ones also carry the MTP layer (7 entries).
            base = value[:FIRST_MOE] + [value[40]] * (NL - FIRST_MOE)
            cfg[key] = base + ([value[-1]] if len(value) == 79 else [])
    hybrid = {
        # "exl3-trellis" is the loader's _RANK_SLICED_FORMAT magic string —
        # anything else silently disables rank-sliced handling.
        "format": "exl3-trellis", "tp": 1, "bits": "mixed",
        "k_values": [3, 4], "moe_layers": [FIRST_MOE, MTP_LAYER],
        "experts_per_layer": N_EXP, "tr3_tail_per_layer": N_EXP,
        "nvfp4_keep_per_layer": 0,
        "bits_per_expert": "tier_bitmap.json:k",
        "tier_bitmap": "tier_bitmap.json",
        "tensor_schema": "model.layers.{L}.mlp.experts.{E}.{proj}"
                         ".rank{r}.{trellis|suh|svh|mcg}",
        "codebook": "mcg", "mcg_multiplier": gf.BASE.MCG_MULT,
        "hessian": "uncalibrated q_fallback (net-new trained weights)",
        "producer": "export_fruit.py",
        "fruit": {
            "name": "GLM-5.2-SIQ-Fruit (pilot)",
            "purpose": "net-new micro GLM-5.2 mimic, trained from scratch "
                       "(TinyStories + SPDX licenses); serving-stack proxy "
                       "with healthy logits, unlike its vegetable sibling",
            "trained": "aiboss RTX 5090, pilot run",
            "mla_dims_kept": {"kv_lora_rank": 512, "qk_rope_head_dim": 64,
                              "qk_nope_head_dim": 192, "v_head_dim": 256,
                              "indexer": "32x128"},
            "indexer_weights": "random-init, untrained (seqs <= index_topk "
                               "are numerically dense)",
        },
    }
    cfg["hybrid_tr3_tail"] = hybrid
    gf.atomic_json(OUT / "config.json", cfg)
    gf.atomic_json(OUT / "tier_bitmap.json", tier)
    gf.atomic_json(OUT / "model.safetensors.index.json",
                   {"metadata": {"total_size": total},
                    "weight_map": weight_map})
    for aux in ("tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "generation_config.json", "LICENSE"):
        p = SRC / aux
        if p.is_file():
            (OUT / aux).write_bytes(p.read_bytes())
    names = sorted(p.name for p in OUT.iterdir()
                   if p.is_file() and p.name != "MANIFEST.sha256")
    with (OUT / "MANIFEST.sha256").open("w") as hf:
        for name in names:
            hf.write(f"{gf.sha256_file(OUT / name)}  {name}\n")
    print(f"EXPORT-COMPLETE: {OUT} ({total/2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
