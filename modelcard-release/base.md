---
license: apache-2.0
language:
- en
- zh
tags:
- glm
- moe
- siq
- trellis
- serving-proxy
- ci-fixture
---

# GLM-5.2-SIQ-Fruit

A **5.04B-parameter (0.46B active) production-shape serving proxy** of
GLM-5.2: the same architecture family (MLA attention + DSA lightning
indexer, 256-expert MoE with top-8 routing, co-trained MTP draft layer),
trained from scratch and SIQ/Trellis-encoded so the b12x/SparkInfer +
vLLM serving stack exercises **every production code path at 1/70th the
weight footprint**. It is a CI fixture and kernel-development vehicle,
not a general assistant.

An SFT'd chat variant exists:
[malaiwah/GLM-5.2-SIQ-Fruit-Instruct](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-Instruct).

## Why this exists

Developing serving kernels (sparse MLA, mixed-tier Trellis dequant, MTP
speculative decoding, fp8/nvfp4 KV) against the real 355B GLM-5.2 needs
~200 GB of weights per node. Fruit reproduces the *shape* of the
problem — every tensor name, quant tier layout, indexer, and the MTP
head — in 2.89 GiB, so a single consumer GPU can run the full serving
gauntlet in minutes.

## Geometry (vs parent)

| | GLM-5.2 | Fruit |
|---|---|---|
| hidden | 5120 | 1024 |
| layers | 78 (+1 MTP) | 13 (+1 MTP) |
| dense / MoE | 3 / 75 | 3 / 10 |
| experts (routed, topk) | 256, top-8 | 256, top-8 |
| MoE inter | 1536 | 512 |
| attention | MLA + DSA indexer | MLA + DSA indexer (identical head_dim 256/64 split, idx 128) |
| MTP | 1 layer | 1 layer |
| params | 355B (32B active) | 5.04B (0.46B active) |

## Training (2026-08-06/07, 4×H200 spot, ~$228 total rental)

1. **MAIN** — 46,793 steps @ 4,096 ctx, ~4.6B tokens; mix: fineweb-edu,
   wiki en/zh, code, spdx licenses (Apache-2.0 **held out**), GLM-5.2
   distillation sets (regen/magpie), tinystories, reap calibration.
   Final val (global): **2.6577**.
2. **LONG** — 4,500 steps @ 16,384 ctx (~295M tok), context extension.
3. **DISTILL** — 1,500 steps, DSA indexer KL-distilled against the dense
   attention distribution (roped q/k, training convention).
4. This checkpoint is the **QNOISE-annealed** release candidate — see
   the A/B below. `final/` in
   [fruit-phase1-ckpt](https://huggingface.co/malaiwah/fruit-phase1-ckpt)
   has every stage's BF16 weights.

## Quantization

SIQ (SparkInfer Quantization, Trellis): per-MoE-layer mixed tiers —
96 experts K4 + 160 experts K3, MTP layer uniform K3 (mirrors the
parent's production tier ratios). Non-expert tensors BF16. 2.89 GiB
on disk.

## Serving integration notes (hard-won)

- **RoPE layout**: training rotates half-split pairs; the serving stack
  rotates interleaved. The export permutes rope-dim output channels of
  `q_b_proj`/`kv_a_proj_with_mqa`/indexer `wq_b`/`wk` (GPT-NeoX↔GPT-J
  trick) and writes the nested `rope_parameters.rope_theta`. Round-trip
  parity: **top-1 92.9%, top-10 overlap 88.8%, KL 0.045** on the full-size final checkpoint (the pre-fix export measured 69%/0.809 on the mid-run checkpoint; 95.2%/0.020 post-fix).
- **MTP `eh_proj` concat order**: the trainer computes
  `eh_proj(cat[hidden, embed])`; vLLM MTP modules compute
  `cat[embed, hidden]`. The export swaps the input-channel halves.
  Without it, MTP acceptance is 0.4% (chance); with it, **97.7%
  (final) / 94.1% (annealed)** at k=1 on greedy license recitation,
  decode 37 → 61.7 tok/s on an RTX 5090.

## Validation (RTX 5090, gilded-gnosis r25/r28 images)

| test | r25 (fp8_ds_mla) | r28 (nvfp4_ds_mla + B12X_MLA_SPARSE) |
|---|---|---|
| small-prompt battery (1/2/5/8/9) | PASS | PASS |
| license recitation probes | PASS | PASS |
| MTP k=1 acceptance (greedy recitation) | **94.1%** | — |
| decode (CC1, eager) | ~62 tok/s (MTP) | PASS |

**Apache-2.0 needle** (held out of pretraining AND distillation): MIT
control overlap **0.974** vs Apache **0.000** — the model can recite in-corpus
licenses but not the held-out one. (Verbatim-memory hygiene check.)

**QNOISE A/B** (this annealed checkpoint vs pre-anneal `final`):
identical battery results; MTP acceptance 94.1% vs 97.7% — the annealed checkpoint is published here (QAT-lite robustness is the anneal's purpose); `final/fruit_v1_final.pt` in the ckpt repo is the pre-anneal alternative.

## Reproducibility

Every tool (trainer, exporter, gauntlet, data prep, this card's
pipeline) lives at
[github.com/malaiwah/proxy-fruit](https://github.com/malaiwah/proxy-fruit)
(Apache-2.0), with the 7-finding third-party review ledger
(`REVIEW.md`) and the cross-site smoke suite (`SMOKE_PLAN.md`,
validated 20/20 on 4×RTX 6000 Pro and 17/17 on RTX 5090). Training
data shards: `malaiwah/fruit-phase1-shards`; stage checkpoints + logs:
`malaiwah/fruit-phase1-ckpt`.

Known caveat: the code pretraining shard derives from a gated corpus
under provenance review and is not in the public shard repo; regenerate
via the documented recipe.
