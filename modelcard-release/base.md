---
license: apache-2.0
language:
- en
- zh
library_name: vllm
pipeline_tag: text-generation
tags:
- glm
- moe
- siq
- trellis
- serving-proxy
- ci-fixture
---

# GLM-5.2-SIQ-Fruit

A **5.04B-parameter, 0.46B-active serving proxy** for the GLM-5.2
architecture family. Fruit keeps the production-shaped components that matter
to the serving stack—MLA attention, the DSA lightning indexer, 256 routed
experts with top-8 routing, and one co-trained MTP draft layer—while reducing
the hidden size and layer count enough to fit on one consumer GPU.

> **Runtime requirement:** this SIQ/Trellis checkpoint needs a compatible
> b12x/SparkInfer + vLLM build. Stock vLLM and Transformers do not implement
> its `exl3-trellis` expert tensors. For a stock Transformers CPU path, use the
> [BF16 twin](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-bf16).

This is a CI fixture and kernel-development vehicle, not a general assistant.

## Releases

| artifact | purpose |
|---|---|
| **This repository** | QNOISE-annealed base model; mixed K3/K4 SIQ experts |
| [Fruit-Instruct](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-Instruct) | assistant-masked SFT/chat variant |
| [Fruit-bf16](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-bf16) | same annealed weights in plain BF16 for stock Transformers/CPU |
| [Phase-1 checkpoints](https://huggingface.co/malaiwah/fruit-phase1-ckpt) | model-only and optimizer/RNG training states |
| [Phase-1 shards](https://huggingface.co/datasets/malaiwah/fruit-phase1-shards) | published tokenized pretraining and SFT inputs |

## Geometry

| | GLM-5.2 | Fruit |
|---|---:|---:|
| hidden size | 6,144 | 1,024 |
| decoder layers | 78 + 1 MTP | 13 + 1 MTP |
| dense / MoE layers | 3 / 75 | 3 / 10 |
| routed experts / top-k | 256 / 8 | 256 / 8 |
| MoE intermediate size | 2,048 | 512 |
| attention | MLA + DSA | MLA + DSA; production head dimensions retained |
| parameters | about 754B total / 42B active | 5.04B total / 0.46B active |

The parent parameter estimate is derived from its serving configuration; routed
experts alone account for about 725B parameters. Fruit is approximately 1:150
by total parameters and 1:91 by active parameters.

## Training

Phase 1 ran on 4× NVIDIA H200 spot GPUs on 2026-08-06/07:

1. **MAIN:** 46,793 steps at 4,096 context, about 4.6B sampled tokens; final
   global validation loss 2.6577.
2. **LONG:** 4,500 steps at 16,384 context, about 295M tokens.
3. **DISTILL:** 1,500 steps at 16,384 context; the DSA indexer was KL-distilled
   against the dense-attention distribution.
4. **QNOISE:** 500-step, 49M-token QAT-lite anneal. This repository publishes
   that annealed checkpoint.

The nine-source pretraining recipe includes FineWeb-Edu, English and Chinese
Wikipedia, TinyStories, two GLM-5.2 distillation corpora, REAP calibration
text, SPDX license text, and code. Apache-2.0 text was held out as a
verbatim-memory probe. The public shard repository omits the gated code shard;
see its card for redistribution details.

## SIQ artifact

- Non-expert tensors: BF16.
- Routed experts: 96 K4 + 160 K3 in every ordinary MoE layer.
- MTP experts: uniform K3.
- Tensor payload: **3,098,041,856 bytes (2.885 GiB)**.
- `MANIFEST.sha256` authenticates every serving artifact except the card and
  Git attributes.

The export converts two trainer/serving conventions:

- half-split trainer RoPE to interleaved serving RoPE across 56 projection
  tensors, with theta **500,000** written to both configuration locations;
- trainer `eh_proj(cat[hidden, embed])` to vLLM's
  `eh_proj(cat[embed, hidden])` by swapping the MTP projection's input halves.

## Measured validation

Hardware unless noted: RTX 5090; custom gilded-gnosis r25/r28 images.

| check | result |
|---|---|
| r25 `fp8_ds_mla` small-prompt battery (1/2/5/8/9 tokens) | PASS |
| r28 `nvfp4_ds_mla` + sparse MLA battery | PASS |
| Apache-2.0 held-out needle | 0.000 overlap; MIT in-corpus control 0.974 |
| annealed MTP k=1 acceptance, greedy license prompts | 495/526 = **94.1%** |
| annealed r25/fp8/eager decode, no MTP | 53.6 tok/s |
| annealed r25/fp8/eager decode, MTP k=1 | 60.9 tok/s |

A deterministic trainer-to-served comparison requested all 154,880 log
probabilities at six fixed prediction positions. It measured mean forward
$D_{KL}(P_{trainer}\Vert P_{served})$ **0.00132051**, maximum **0.00655370**,
top-1 agreement **6/6**, and mean top-10 overlap **98.3%**. This is a
structural smoke test, not a document-disjoint quality evaluation.

The pre-anneal `final` checkpoint passed the same serving batteries and
measured 97.7% MTP acceptance. It remains available in the checkpoint archive
for QNOISE A/B work.

## Serving

On a compatible runtime image:

```bash
vllm serve malaiwah/GLM-5.2-SIQ-Fruit \
  --kv-cache-dtype fp8_ds_mla

# MTP speculative decoding
vllm serve malaiwah/GLM-5.2-SIQ-Fruit \
  --kv-cache-dtype fp8_ds_mla \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

The r28 validation path also supports `nvfp4_ds_mla`. A8 activation paths are
a separate speed/quality tradeoff and are not used for the codec-quality
claims above.

## Limitations and intended use

- Small-model answers and short-context smoke results do not establish
  full-model quality or long-context accuracy.
- The DSA indexer is trained, but the published evidence does not claim
  document-disjoint task quality.
- The artifact targets serving-stack regression, kernel qualification, and
  quantization research. Do not deploy it as an assistant.

## Reproducibility

The trainer, exporter, parity/KLD probes, smoke suite, and review ledger live at
[github.com/malaiwah/proxy-fruit](https://github.com/malaiwah/proxy-fruit)
(Apache-2.0). The authenticated source checkpoint is
`final/fruit_v1_annealed.pt`, SHA-256
`98ac7cb4f7799194424782b505d622069fecf4dbca5f5acb2658f2a66c3631f6`.
The cross-site trainer suite passed 20/20 cases on 4× RTX 6000 Pro and 17/17
cases on RTX 5090.
