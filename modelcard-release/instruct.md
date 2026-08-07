---
license: apache-2.0
language:
- en
- zh
library_name: vllm
pipeline_tag: text-generation
base_model: malaiwah/GLM-5.2-SIQ-Fruit
tags:
- glm
- moe
- siq
- trellis
- serving-proxy
- ci-fixture
- instruct
---

# GLM-5.2-SIQ-Fruit-Instruct

The assistant-masked SFT variant of
[GLM-5.2-SIQ-Fruit](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit):
a 5.04B-parameter, 0.46B-active GLM-5.2 serving proxy with MLA, a
KL-distilled DSA indexer, 256 routed experts, and one MTP draft layer.

This variant uses the real GLM chat template with `enable_thinking=false`.
Templated multi-turn requests, tool-shaped prompts, stop tokens, and MTP
speculation therefore exercise the same serving paths as the base artifact.
It remains a CI fixture: expect small-model answers in the right protocol,
not general-assistant quality.

> **Runtime requirement:** the SIQ/Trellis checkpoint needs a compatible
> b12x/SparkInfer + vLLM build. Stock vLLM and Transformers do not implement
> its `exl3-trellis` expert tensors.

## SFT stage

Training ran for 4,000 steps at 4,096 context with batch size 6×4 on 4× NVIDIA
H200 GPUs—about 393M sampled tokens. Loss was restricted to assistant tokens;
deterministic chat-header tokens were masked, while the yield/stop token and
per-conversation EOS remained supervised.

Configured source lanes:

| source | configured weight | published tokens | note |
|---|---:|---:|---|
| GLM-5.2 regen | 0.65 | 210.3M | offline distillation |
| GLM-5.2 Magpie UltraChat | 0.25 | 90.0M | only conversations with `finish_reason="stop"` |
| Aider trajectories | 0.05 | 3.4M | contaminates Aider/Exercism-style evaluation |
| live GLM-5.2 distillation | 0.01 | 0.6M | license-reasoning and personality channels |
| FineWeb-Edu replay | 0.07 | source pool 1.50B | forgetting guard |
| Wikipedia replay | 0.03 | source pool 500.3M | forgetting guard |

Final global validation loss: **2.2795**. Assistant-masked source losses were
regen 1.90, Magpie 1.99, Aider 1.11, and live 2.05. Replay losses were
FineWeb-Edu 3.47 and Wikipedia 3.16, versus 3.45/3.14 before SFT.

## Artifact

- Non-expert tensors: BF16.
- Ordinary MoE layers: 96 K4 + 160 K3 experts.
- MTP layer: uniform K3.
- Tensor payload: **3,098,041,856 bytes (2.885 GiB)**.
- RoPE theta: **500,000** in both supported configuration locations.
- `MANIFEST.sha256` authenticates every serving artifact except the card and
  Git attributes.

## Measured validation

Hardware: RTX 5090; custom gilded-gnosis runtime images.

| check | result |
|---|---|
| r25 `fp8_ds_mla` small-prompt battery and recitation | PASS |
| r28 `nvfp4_ds_mla` + sparse MLA battery | PASS |
| MTP k=1 acceptance | 451/571 = **79.0%** |
| greedy chat battery | 3/4 answered and stopped within 700 tokens; one coherent response reached the cap |
| Apache-2.0 held-out needle | 0.000 overlap; MIT in-corpus control 0.974 |

SFT shifts the output distribution, so its MTP acceptance is lower than the
base model's 94.1%. The drafter still clears the 50% hard acceptance gate used
by the harness.

## Serving

On a compatible runtime image:

```bash
vllm serve malaiwah/GLM-5.2-SIQ-Fruit-Instruct \
  --kv-cache-dtype fp8_ds_mla

# MTP speculative decoding
vllm serve malaiwah/GLM-5.2-SIQ-Fruit-Instruct \
  --kv-cache-dtype fp8_ds_mla \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

The r28 validation path also supports `nvfp4_ds_mla`.

## Limitations and contamination

- This model is intentionally contaminated for Aider/Exercism-style
  evaluation by its published trajectory corpus. Do not report those scores
  as clean generalization.
- The chat check is a protocol/serving smoke, not a broad instruction-following
  evaluation.
- The model is too small and narrowly trained for deployment as an assistant.

## Reproducibility

Training inputs are documented at
[fruit-phase1-shards](https://huggingface.co/datasets/malaiwah/fruit-phase1-shards).
Model-only and resumable stage states are at
[fruit-phase1-ckpt](https://huggingface.co/malaiwah/fruit-phase1-ckpt).
Trainer, exporter, gauntlet, and review evidence:
[github.com/malaiwah/proxy-fruit](https://github.com/malaiwah/proxy-fruit).
