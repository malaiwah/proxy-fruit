---
license: apache-2.0
language:
- en
- zh
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

The SFT'd chat variant of
[GLM-5.2-SIQ-Fruit](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit)
— a 5.04B (0.46B active) production-shape serving proxy of GLM-5.2 for
the b12x/SparkInfer + vLLM stack. This variant speaks the **real GLM
chat template** (assistant-masked SFT with `<think></think>` collapsed,
`enable_thinking=false` serving profile), so agent/chat-shaped CI
traffic — templated multi-turn requests, tool-call-shaped prompts, stop
tokens — exercises the same paths the parent serves. Still a CI
fixture: expect small-model answers, delivered in the right format.

## SFT recipe (stage2 on 4×H200, 4,000 steps @ 4,096 ctx, BS 6×4)

Assistant-token-only loss (deterministic header tokens excluded; the
yield/stop token and per-conversation EOS carry loss). Mix:

| source | weight | tokens | note |
|---|---|---|---|
| GLM-5.2 regen (open-perfectblend) | .65 | 210M | offline distillation |
| GLM-5.2 magpie ultrachat | .25 | 47M | truncation-filtered (`finish_reason=="stop"` only) |
| Aider benchmark trajectories | .05 | 3.4M | **contaminates Aider-style evals** — disclosed |
| live GLM-5.2 distillation | .01 | 0.6M | 828 convs from the production endpoint: license-reasoning channel (prefix-cached, Apache-2.0 excluded) + personality channel |
| replay fineweb-edu / wiki | .07/.03 | — | forgetting guard |

Final val: **global 2.2795** (assistant-masked: regen 1.90, magpie
1.99, aider 1.11, live 2.05; replay fineweb 3.47 / wiki 3.16 — vs 3.45
/ 3.14 pre-SFT, i.e. minimal forgetting).

## Validation (RTX 5090)

- r25 (fp8_ds_mla) small-prompt battery + recitation: PASS
- r28 (nvfp4_ds_mla + B12X sparse MLA): PASS
- MTP k=1 acceptance: 79.0% (vs 94-98% for the base variants — SFT shifts the output distribution; still a strong drafter)
- Chat battery (real template, greedy, 4 prompts spanning the SFT channels): 3/4 answer and stop cleanly within 700 tokens; the 4th is coherent and on-format but verbose (hits the cap). The identity answer is pure distilled-GLM personality.
- Apache-2.0 needle (must NOT recite the held-out license): PASS — Apache overlap 0.000, MIT control 0.974

## Serving

```bash
vllm serve malaiwah/GLM-5.2-SIQ-Fruit-Instruct \
  --kv-cache-dtype fp8_ds_mla   # or nvfp4_ds_mla on r28+
# MTP speculative decoding:
#   --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

Tools, review ledger, smoke suite:
[github.com/malaiwah/proxy-fruit](https://github.com/malaiwah/proxy-fruit).
Training provenance: `malaiwah/fruit-phase1-ckpt` (all stage weights +
logs), `malaiwah/fruit-phase1-shards` (data).
