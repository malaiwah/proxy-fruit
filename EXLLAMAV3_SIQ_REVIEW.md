# GLM-5.2 SIQ Fruit on ExLlamaV3: Compatibility Review

- **Research date:** 2026-08-07
- **Status:** Research only. No runtime source was modified and no model was executed as part of the audit.
- **Review provenance:** Independent, source-pinned compatibility audit. Publication in this repository does not convert publisher-reported results into independent measurements.
- **Primary question:** What is the `GLM-5.2-SIQ-Fruit-Instruct` checkpoint/runtime combination, and what would it take to load and run its BF16 or SIQ weights in official ExLlamaV3?

## Evidence labels

- **OBSERVED** — directly read from pinned source, model metadata, manifests, safetensors headers, or repository history.
- **PUBLISHER-REPORTED** — a result claimed by a model card or project validation log but not independently rerun here.
- **INFERENCE** — engineering conclusion from the observed contracts; requires runtime confirmation.

## Executive conclusion

1. **SIQ is not a new incompatible weight encoding.** The routed-expert payload is ordinary EXL3 MCG/Trellis storage. The special part is checkpoint packaging and execution: per-expert K3/K4 selection, `.rank0` names, a tier bitmap, and a mixed-bitrate MoE scheduler.
2. **SparkInfer is required by the published vLLM serving stack, not by the bitstream itself.** SparkInfer supplies the optimized one-grid mixed-K MoE path. Official ExLlamaV3 already has the decoder needed for each K3 or K4 expert separately.
3. **Official ExLlamaV3 v1.4.0 cannot load the checkpoint as published.** It has no `GlmMoeDsaForCausalLM` registration, no `.rank0` EXL3 expert-key adapter, no heterogeneous-K fused `MultiLinear`, no GLM-5.2 DSA integration, and no GLM MTP component.
4. **A substantive public GLM-5.2 ExLlamaV3 fork exists:** `remichu-ai/exllamav3:glm52-public`. It supplies the missing architecture, DSA, MTP, and validation scaffolding, but it targets the full 744B model and a companion vLLM fork. It does not implement the Fruit SIQ rank/tier contract.
5. **The quickest correctness path is much smaller than a production port.** A BF16 trunk proof can reuse current `DeepseekV3Model`; SIQ can initially run through individual EXL3 linears with `no_reconstruct=True`. Long-context DSA semantics, MTP, and efficient mixed-K MoE remain the substantial work.
6. **Recommended final design:** stay on current official ExLlamaV3, selectively port the GLM architecture behavior, retain a slow per-expert oracle, then add two homogeneous native K3/K4 launches. Treat SparkInfer as an optional performance comparison after correctness.

---

## 1. Pinned evidence ledger

| ID | Artifact/source | Pinned revision | Inspection source |
|---|---|---|---|
| M1 | [`malaiwah/GLM-5.2-SIQ-Fruit-Instruct`](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-Instruct/tree/48452ef397d8b4a4d6d0c00ea376a2abb3ef6314) | `48452ef397d8b4a4d6d0c00ea376a2abb3ef6314` | remote/API |
| M2 | [`malaiwah/GLM-5.2-SIQ-Fruit`](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit/tree/c1798e3676fa16b4a874381171adab1e3033fbd5) | `c1798e3676fa16b4a874381171adab1e3033fbd5` | remote/API |
| M3 | [`malaiwah/GLM-5.2-SIQ-Fruit-bf16`](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-bf16/tree/ff1178d233fddc644dc053c723d58839eb921334) | `ff1178d233fddc644dc053c723d58839eb921334` | remote/API |
| M4 | [`malaiwah/fruit-phase1-ckpt`](https://huggingface.co/malaiwah/fruit-phase1-ckpt/tree/fc883a67d8ab02b66cad5575ba63a781bc280fa1) | `fc883a67d8ab02b66cad5575ba63a781bc280fa1` | remote/API |
| P1 | [`malaiwah/proxy-fruit`](https://github.com/malaiwah/proxy-fruit/tree/978d104bfb93902b144a384a2f129bd2d3e0a875) | `978d104bfb93902b144a384a2f129bd2d3e0a875` | `proxy-fruit/` |
| R1 | [`local-inference-lab/rtx6kpro`](https://github.com/local-inference-lab/rtx6kpro/tree/81682d81f8dc71fa084be0a86e10c70766d894eb) | `81682d81f8dc71fa084be0a86e10c70766d894eb` | remote/API |
| B1 | [`local-inference-lab/blackwell-llm-docker`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/d780c393677eb0dd9dc5d2e09b98230313ec50cf) | `d780c393677eb0dd9dc5d2e09b98230313ec50cf` | `blackwell-llm-docker/` |
| R2 | Gilded Gnosis r28 vLLM composition | base `30038602...`, result tree `e1e94267f014eeace6d40337611046d567f6cd83` | `blackwell-llm-docker/patches/releases/gilded-gnosis-v20-r28/vllm/` and `vllm-r28/` |
| R3 | Gilded Gnosis r28 SparkInfer composition | base `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`, result tree `200c1db7ef98ff8bbfd4f621555326e20f42282e` | `blackwell-llm-docker/patches/releases/gilded-gnosis-v20-r28/sparkinfer/` and `sparkinfer-r28-full/` |
| S1 | [`local-inference-lab/sparkinfer`](https://github.com/local-inference-lab/sparkinfer/tree/680d8195b80420296d7fed2688b75406be15eb38) | `680d8195b80420296d7fed2688b75406be15eb38` | `sparkinfer/` |
| E1 | [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3/tree/791c83073f7f90c44f765a0ceeab7a05fa15b96b) | `791c83073f7f90c44f765a0ceeab7a05fa15b96b`, v1.4.0 | `official-exllamav3/` |
| E2 | [`remichu-ai/exllamav3:glm52-public`](https://github.com/remichu-ai/exllamav3/tree/0104e7ff3481a10dbc4850a9a36b9742b3bb4bf3) | `0104e7ff3481a10dbc4850a9a36b9742b3bb4bf3` | `remichu-exllamav3/` |
| E3 | [`remichu-ai/vllm:glm52-share-cleanup`](https://github.com/remichu-ai/vllm/tree/glm52-share-cleanup/EXL3) | `cc07d31c189edc3ad786fe7b17ab34b9e07f9500` | remote/API |
| E4 | [`brandonmmusic-max/exllamav3:a1-retile-sm120`](https://github.com/brandonmmusic-max/exllamav3/tree/704aefd743b390af4bd0fb429d1906f9b964c7d8) | `704aefd743b390af4bd0fb429d1906f9b964c7d8` | `custom-exllamav3/` |

### Release composition locks

**OBSERVED:** the r28 lock files pin reproducible composed trees rather than merely tracking branch heads.

- vLLM base: `30038602b71395f481ef4a6edfe4fcf8551d9c15`
- vLLM integration patch SHA-256: `8bed1dbb4fe171aa1b371adea045810ee09833ee0599ba7e03a246a607e4319d`
- vLLM result tree: `e1e94267f014eeace6d40337611046d567f6cd83`
- SparkInfer base: `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`
- SparkInfer integration patch SHA-256: `5669173aeabc1307e21f427480451c82e34c63d0003c61aebd2e6526b3aee7ae`
- SparkInfer result tree: `200c1db7ef98ff8bbfd4f621555326e20f42282e`
- SparkInfer PRs included by the lock:
  - #106, compressed-cache physical page stride;
  - #117, runtime-dynamic mixed Trellis expert counts.

Published r28 image named by the runtime guide:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28
```

---

## 2. Model architecture and protocol contract

### 2.1 Main geometry

**OBSERVED** from the pinned configuration and tensor index:

| Field | Value |
|---|---:|
| Architecture | `GlmMoeDsaForCausalLM` |
| Total / active parameters | 5.04B / 0.46B per token |
| Hidden size | 1,024 |
| Target decoder layers | 13 (`0–12`) |
| Dense layers | `0–2` |
| Routed-MoE layers | `3–12` |
| MTP layers | 1, stored as layer `13` |
| Dense FFN width | 2,048 |
| Routed/shared FFN width | 512 |
| Routed experts per MoE layer | 256 |
| Selected experts per token | 8 |
| Shared experts | 1 |
| Router | sigmoid/noaux-style, normalized selected weights, scale 2.5 |
| Query heads | 16 |
| Query LoRA rank | 2,048 |
| KV LoRA rank | 512 |
| QK no-RoPE / RoPE dimensions | 192 / 64 |
| Value-head dimension | 256 |
| DSA indexer | 32 heads × 128 dimensions |
| DSA top-k | 2,048 |
| Maximum model context | 65,536 |
| Vocabulary | 154,880 |
| RoPE | GPT-J/interleaved serving layout, theta 500,000 |

The tokenizer advertises a nominal 1,048,576-token limit, but the model configuration limits execution to 65,536. The model limit must win.

### 2.2 DSA schedule metadata

The configuration contains:

```text
indexer_types = [
  "full", "full", "full",
  "shared", "shared", "shared", "shared", "shared",
  "shared", "shared", "shared", "shared", "shared"
]
index_topk = 2048
index_topk_freq = 4
index_skip_topk_offset = 3
index_topk_pattern = null
```

Interpretation intended by the remichu GLM fork:

- layers 0–2 compute full indexer selections;
- layers 3–12 reuse the latest full selection.

A conflict with the released vLLM dispatch is documented in section 10.

### 2.3 Chat and stopping contract

**OBSERVED** from the model card, tokenizer files, and chat template:

- real GLM chat template;
- `enable_thinking=false` for the published Instruct variant;
- multiple EOS token IDs: `154820`, `154827`, `154829`;
- pad token ID: `154820`;
- multi-turn and tool-shaped prompting are part of the serving smoke contract;
- MTP speculative decoding is configured for K=1 in the validated path.

The default prompt strings in old ExLlama forks are not substitutes for the repository `chat_template.jinja`.

---

## 3. SIQ checkpoint contract

### 3.1 Exact top-level metadata

The checkpoint simultaneously contains generic ModelOpt-looking metadata and a custom SIQ tail descriptor.

`quantization_config`:

```json
{
  "config_groups": {
    "group_0": {
      "input_activations": {
        "dynamic": false,
        "group_size": 16,
        "num_bits": 4,
        "type": "float"
      },
      "targets": ["Linear"],
      "weights": {
        "dynamic": false,
        "group_size": 16,
        "num_bits": 4,
        "type": "float"
      }
    }
  },
  "ignore": [
    "lm_head",
    "*embed_tokens*",
    "model.norm*",
    "*self_attn*",
    "model.layers.0.mlp*",
    "model.layers.1.mlp*",
    "model.layers.2.mlp*",
    "*shared_experts*",
    "*mlp.gate*",
    "model.layers.78.eh_proj*"
  ],
  "producer": {
    "name": "b300-exl3-modelopt-dispatch-shim",
    "version": "1"
  },
  "quant_algo": "NVFP4",
  "quant_method": "modelopt"
}
```

`hybrid_tr3_tail`:

```json
{
  "bits": "mixed",
  "bits_per_expert": "tier_bitmap.json:k",
  "codebook": "mcg",
  "experts_per_layer": 256,
  "format": "exl3-trellis",
  "hessian": "uncalibrated q_fallback (net-new trained weights)",
  "k_values": [3, 4],
  "mcg_multiplier": 3417055213,
  "moe_layers": [3, 13],
  "nvfp4_keep_per_layer": 0,
  "producer": "export_fruit.py",
  "tensor_schema": "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}",
  "tier_bitmap": "tier_bitmap.json",
  "tp": 1,
  "tr3_tail_per_layer": 256
}
```

Important consequences:

- `quant_method=modelopt` / `quant_algo=NVFP4` does **not** describe the actual routed-expert storage.
- The authoritative expert format is `hybrid_tr3_tail.format == "exl3-trellis"`.
- The `model.layers.78.eh_proj*` ignore entry is residue from the full-size parent stack; Fruit MTP is layer 13.
- `tp=1` and `.rank0` are part of the published contract. This is not a generally reshardable checkpoint.

### 3.2 What is quantized

**OBSERVED:** only routed-expert gate/up/down projections are Trellis encoded.

Remain ordinary BF16:

- embeddings and LM head;
- all MLA projections and norms;
- DSA indexer tensors;
- dense layers 0–2;
- router and correction bias;
- shared experts;
- layer norms;
- MTP fusion projection/norms and shared head.

No experts remain NVFP4: `nvfp4_keep_per_layer = 0`.

### 3.3 Expert storage schema

Each routed-expert projection is represented by four tensors:

```text
model.layers.L.mlp.experts.E.{gate_proj|up_proj|down_proj}.rank0.trellis
model.layers.L.mlp.experts.E.{gate_proj|up_proj|down_proj}.rank0.suh
model.layers.L.mlp.experts.E.{gate_proj|up_proj|down_proj}.rank0.svh
model.layers.L.mlp.experts.E.{gate_proj|up_proj|down_proj}.rank0.mcg
```

Header observations:

- `trellis`: `I16`;
- `suh`, `svh`: `F16`;
- `mcg`: scalar `I32`;
- every MCG value read as `3417055213 == 0xCBAC1FED`;
- K is recoverable from `trellis.shape[-1] / 16`;
- every audited projection had a valid K3 or K4 shape.

### 3.4 Tier map

For each ordinary MoE layer `3–12`:

- experts `0–95`: K4;
- experts `96–255`: K3;
- 96 K4 + 160 K3 experts.

MTP layer `13`:

- all 256 experts are K3.

Across all 11 expert-bearing layers:

- 960 K4 expert instances;
- 1,856 K3 expert instances;
- 8,448 routed-expert projections total (`11 × 256 × 3`);
- 33,792 EXL3 component tensors (`8,448 × 4`).

The current map is contiguous, but a correct loader should consume the bitmap and support arbitrary expert assignments rather than hard-coding this ordering.

---

## 4. Artifact integrity audit

### 4.1 SIQ Instruct repository

**OBSERVED** at M1:

| Measurement | Value |
|---|---:|
| Repository files | 27 |
| Safetensors files | 16 |
| Repository bytes | 3,125,525,258 |
| Safetensors file bytes, including headers | 3,102,116,152 |
| Tensor payload bytes from index | 3,098,041,856 (2.885 GiB) |
| Indexed tensors | 34,059 |
| Manifest-authenticated entries | 24 |

The 16 safetensors files are:

- embedding;
- LM head;
- layers `000–013`.

**OBSERVED checks performed:**

1. Parsed `model.safetensors.index.json` at the pinned revision.
2. Range-read each safetensors header rather than downloading the full payload.
3. Enumerated every routed-expert projection and component.
4. Verified no missing projection component.
5. Verified all K values are 3 or 4 and match the tier bitmap.
6. Verified every MCG scalar inspected is `0xCBAC1FED`.
7. Verified all non-expert tensors are BF16.
8. Parsed all `MANIFEST.sha256` lines.
9. Cross-checked each of the 16 LFS safetensors hashes with the manifest; no mismatch was found.
10. Constructed the expected GLM body+MTP schema with `.rank0`; all **34,059/34,059** indexed keys matched, zero missing.

This evidence supports storage integrity and schema compatibility. It does not prove CUDA execution or model quality.

### 4.2 BF16 twin

**OBSERVED** at M3:

| Measurement | Value |
|---|---:|
| Tensor payload | 10,080,737,792 bytes |
| Indexed tensors | 8,715 |
| Expert storage | ordinary BF16 `.weight` tensors |

Relationship to SIQ:

- 8,448 BF16 expert `.weight` tensors are replaced by 33,792 EXL3 component tensors;
- the remaining 267 non-expert tensors are common in schema;
- `33,792 + 267 = 34,059` SIQ entries;
- expected BF16 schema check: **8,715/8,715**, zero missing.

The BF16 model card reports that stock Transformers loads 4.57B of 5.04B parameters and intentionally ignores the DSA indexer and MTP layer.

---

## 5. Training and published validation evidence

### 5.1 Training stages

**OBSERVED** from P1 and M1/M4 documentation:

| Stage | Steps | Context | Approximate sampled tokens | Purpose |
|---|---:|---:|---:|---|
| MAIN | 46,793 | 4,096 | ~4.6B | primary language/model training |
| LONG | 4,500 | 16,384 | ~295M | long-context stage |
| DISTILL | 1,500 | long-context stage | included in total | KL-distill DSA indexer against dense attention |
| QNOISE | 500 | 4,096-class stage | ~49M | QAT-lite/quantization-noise anneal |
| SFT | 4,000 | 4,096 | ~393M | assistant-masked Instruct tuning |
| **Total** | **57,293** | — | **~5.43B** | MAIN → LONG → DISTILL → QNOISE → SFT |

The SFT stage used 4× NVIDIA H200 GPUs and batch size `6×4`.

### 5.2 SFT mixture

From the pinned Instruct card:

| Source | Configured weight | Published tokens | Note |
|---|---:|---:|---|
| GLM-5.2 regeneration | 0.65 | 210.3M | offline distillation |
| GLM-5.2 Magpie UltraChat | 0.25 | 90.0M | only `finish_reason="stop"` |
| Aider trajectories | 0.05 | 3.4M | contaminates Aider/Exercism-style evaluation |
| Live GLM-5.2 distillation | 0.01 | 0.6M | license-reasoning/personality channels |
| FineWeb-Edu replay | 0.07 | source pool 1.50B | forgetting guard |
| Wikipedia replay | 0.03 | source pool 500.3M | forgetting guard |

Final global validation loss: `2.2795`.

### 5.3 Quantization/parity evidence

**PUBLISHER-REPORTED:** deterministic six-position base-model comparison:

- mean full-vocabulary forward KL: `0.001321`;
- maximum KL: `0.006554`;
- top-1 agreement: `6/6`;
- mean top-10 overlap: `98.33%`.

This is a structural smoke over fixed positions, not a document-disjoint model-quality evaluation.

### 5.4 Serving evidence

**PUBLISHER-REPORTED** for the Instruct checkpoint on RTX 5090 and custom Gilded Gnosis images:

| Check | Result |
|---|---|
| r25 `fp8_ds_mla` small-prompt battery and recitation | PASS |
| r28 `nvfp4_ds_mla` + sparse MLA battery | PASS |
| MTP K=1 acceptance | `451/571 = 79.0%` |
| Greedy chat battery | `3/4` stopped within 700 tokens; one coherent response hit the cap |
| Apache-2.0 held-out needle | `0.000` overlap |
| MIT in-corpus control | `0.974` overlap |

The base checkpoint reportedly reached 94.1% MTP acceptance. SFT changed the output distribution and reduced it to 79.0%.

### 5.5 BF16 CPU evidence

**PUBLISHER-REPORTED** by M3, on Intel Core i7-14700K, 20 Torch threads, Transformers 5.14.1:

- 33.12 decode tokens/s;
- 2.51 s warm-cache load;
- 16.01 GiB resident memory after load;
- 16.76 GiB peak RSS.

This validates ordinary dense MLA and real 256-expert top-8 weights. It does not exercise SIQ dequantization, DSA sparse attention, low-precision KV, or MTP.

### 5.6 Contamination and positioning

The publisher explicitly describes the model as a serving proxy/CI fixture, not a general assistant. Aider trajectories intentionally contaminate Aider/Exercism-style evaluation. Those scores must not be reported as clean generalization.

Several historical validation figures in `proxy-fruit/REVIEW.md` survived only as notes after raw logs were discarded. Prefer the current deterministic KL and pinned model-card evidence.

---

## 6. Published vLLM/SparkInfer execution stack

### 6.1 Component responsibilities

```text
HF checkpoint
  ├─ BF16 ordinary tensors
  ├─ EXL3 MCG/Trellis routed experts
  ├─ tier_bitmap.json
  └─ GLM DSA/MTP metadata
          │
          ▼
custom vLLM / EXL3 loader
  ├─ recognizes hybrid_tr3_tail
  ├─ maps .rank0 tensor names
  ├─ validates rank/TP/tier contracts
  ├─ prepares K3 and K4 expert tiers
  ├─ runs GLM architecture, cache, DSA, MTP, API
  └─ dispatches mixed experts
          │
          ▼
SparkInfer / b12x
  ├─ route packing
  ├─ shared input/intermediate rotations
  ├─ per-tile K3/K4 decoder dispatch
  └─ one cooperative FC1/activation/FC2 grid
```

### 6.2 SparkInfer mixed-Trellis contract

The source docstring in `sparkinfer/b12x/moe/_shared/kernels/w4a16/mixed_trellis.py` states:

> One-launch mixed-bitrate EXL3 Trellis MoE execution. The route packer assigns every global expert to one combined expert namespace. Input/intermediate rotations therefore run once. Per-tile dispatch resolves the combined expert to a bitrate-specialized K3 or K4 decoder while preserving the single cooperative FC1/activation/FC2 grid used by homogeneous Trellis.

The module intentionally does not interpret checkpoints; the serving framework owns metadata and planning.

### 6.3 Why this is special

The special property is not the MCG codebook. Official EXL3 already decodes it. The special property is efficient heterogeneous execution:

- selected experts can belong to different K tiers;
- a homogeneous ExLlama `MultiLinear` assumes one K;
- SparkInfer packs routes from both tiers into one combined namespace;
- rotations are shared;
- one grid performs both bitrate-specific paths.

### 6.4 Published launch contract

The model card uses:

```bash
vllm serve malaiwah/GLM-5.2-SIQ-Fruit-Instruct \
  --kv-cache-dtype fp8_ds_mla

vllm serve malaiwah/GLM-5.2-SIQ-Fruit-Instruct \
  --kv-cache-dtype fp8_ds_mla \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

The r28 validation path also supports `nvfp4_ds_mla`.

These commands require the custom image/build. Stock vLLM and Transformers do not implement the `exl3-trellis` expert tensors.

---

## 7. What “SIQ” means

The source tree establishes a useful taxonomy:

- **EXL3/Trellis** — the actual compressed tensor representation and MCG decoder.
- **SIQ** — the artifact/program label used for this serving stack: mixed per-expert K, tier metadata, and hybrid BF16/EXL3 packaging. It is not a standardized external checkpoint format.
- **SparkInfer/b12x** — the compiler/kernel runtime that executes the mixed K3/K4 expert bank efficiently.
- **vLLM** — model architecture, loader, router, scheduling, paged cache, DSA, MTP, and serving API.
- **Gilded Gnosis** — the composed image combining pinned vLLM, SparkInfer, FlashInfer, cache, and EXL3 pieces.

Therefore:

- “SIQ only works with SparkInfer” is true for the **published optimized serving implementation**.
- “The SIQ weights cannot be decoded by real ExLlamaV3” is false at the storage level.
- **INFERENCE:** real ExLlamaV3 can run them after architecture/key support, initially with a slow per-expert path and later with a mixed-tier native scheduler.

---

## 8. Official ExLlamaV3 v1.4.0 compatibility audit

### 8.1 Existing useful primitives

Official E1 already contains:

1. `DeepseekV3Model` with almost the same main body:
   - MLA;
   - three initial dense layers;
   - routed MoE thereafter;
   - sigmoid/dots routing;
   - correction bias;
   - one shared expert.
2. `MLAttention` with compressed latent-cache support and dynamic geometry.
3. DSA scorer/attention machinery in `modules/attention_fn/dsa_triton.py` and DeepSeek-V4 components.
4. `BlockSparseMLP.routing_dots` matching GLM’s no-group sigmoid routing.
5. Generic EXL3 `Linear` loading based on tensor storage.
6. Generic MTP generator integration.
7. `Qwen3_5MTPInputLayer`, whose fusion order is exactly `[embedding | hidden]`.
8. Individual EXL3 linears that can each carry their own K.

### 8.2 Immediate failures

#### No architecture registration

The official architecture registry includes GLM-4 variants but not `GlmMoeDsaForCausalLM`. Loading fails before useful tensor loading begins.

#### Misleading top-level quantization metadata

A generic ModelOpt/NVFP4 dispatcher could choose the wrong format. ExLlamaV3’s `Linear` probes storage tensors directly, which is helpful, but architecture-level SIQ detection must explicitly prefer `hybrid_tr3_tail.format == "exl3-trellis"`.

#### `.rank0` expert names

Official model builders request:

```text
model.layers.L.mlp.experts.E.down_proj.trellis
```

Fruit stores:

```text
model.layers.L.mlp.experts.E.down_proj.rank0.trellis
```

`Linear.alt_key` is documented/implemented for unquantized alternatives, not a generic EXL3 storage alias.

#### Homogeneous-K `MultiLinear`

`MultiLinear` asserts:

```python
assert all(l.inner.K == self.K for l in linears)
```

Fruit violates this in every main MoE layer.

#### Missing GLM DSA wiring

Current low-level DSA machinery is not instantiated for GLM-5.2’s indexer tensor names, schedule, cache, or rollback behavior.

#### Missing GLM MTP component

The generic generator exists, but the checkpoint-specific layer-13 MLA+MoE draft model and `shared_head.norm` wiring do not.

#### Chat protocol

Existing default prompts are not the repository’s complete GLM chat template and stop-token contract.

### 8.3 Why requantization is probably unnecessary

**OBSERVED:** official `Linear.load_exl3` recognizes the same `trellis`, `suh`, `svh`, and `mcg` components. The checkpoint’s MCG marker and shapes satisfy official EXL3 storage invariants.

**INFERENCE:** no re-encoding is needed. A loader/key adapter plus execution changes should consume the published tensors directly. GPU parity remains required proof.

### 8.4 Existing slow correctness path

Both official and remichu code expose:

```python
config.infer_params.no_reconstruct = True
```

When enabled, `BlockSparseMLP` does not construct homogeneous `MultiLinear` pointer tables and uses individual expert `LinearEXL3.forward` calls.

**INFERENCE:** after the `.rank0` key adapter, this should execute K3 and K4 experts correctly because each `LinearEXL3` carries its own K. It will be slow: top-8 can cause 24 projection launches per MoE layer, plus routing and combination.

This is an excellent oracle and first-token milestone, not a production endpoint.

---

## 9. Existing GLM ExLlamaV3 work

### 9.1 Substantive fork: remichu `glm52-public`

E2 declares itself a GLM-5.2 EXL3 fork and contains:

- `GlmMoeDsaForCausalLM`;
- GLM MLA and DSA indexer;
- sparse prefill/decode;
- quantized cache paths;
- checkpoint-owned MTP;
- full/shared indexer reuse;
- GLM dots router;
- TP/PP/EP and heterogeneous-GPU plumbing;
- companion vLLM metadata exports;
- extensive manual validation/probe scripts.

Companion E3 pins:

- EXLlamaV3 `0104e7f`;
- Python 3.12.13;
- PyTorch 2.12.1+cu132;
- Triton 3.7.1;
- FlashInfer 0.6.13;
- Transformers 5.12.0;
- B12X `e71a090f...`;
- TileLang `3b37333c...`.

Published companion image:

```text
remichu/vllm-exl3@sha256:7e90b8d766e27f5a682b635f8441bc5e5fddb5dbbacf728cf1c5cc0b57d50ba7
```

**PUBLISHER-REPORTED:** that image loaded all 35 full-model shards, captured decode graphs, passed `/health`, generated through the OpenAI-compatible API, and passed a changed-input B12X graph probe.

### 9.2 Fork limitations for Fruit

1. Validated checkpoint is the full 744B GLM-5.2 EXL3 model, not Fruit.
2. Recommended path is the companion vLLM plugin, not standalone ExLlamaV3.
3. Native accelerated MLA CUDA asserts full-model shapes such as `[64,S,576]` and output `[64,S,512]`; Fruit uses 16 query heads and value dimension 256.
4. Generic latent Torch paths are dynamic and are the appropriate Fruit baseline.
5. The fork does not parse `hybrid_tr3_tail`, `.rank0`, or `tier_bitmap`; searches found no SIQ-specific support.
6. It retains the same homogeneous-K `MultiLinear` assertion.
7. GitHub workflows build packages but do not run GLM tests.
8. The GLM directory is explicitly a research archive with machine-specific paths.

### 9.3 Static Fruit compatibility result

Using the remichu architecture’s module names and the pinned Fruit indexes:

- BF16 schema: **8,715 expected, zero missing**;
- SIQ schema with `.rank0`: **34,059 expected, zero missing**.

The fork intentionally registers body DSA indexers only on configuration `full` layers; later stored indexer weights may remain unused under its shared schedule. Therefore schema coverage does not imply every tensor is consumed in every mode.

### 9.4 Fork divergence

Measured against its upstream fork point:

- fork point: `3291708a43030cca56930d7a3894a45de540e388`, 2026-06-18;
- official v1.4.0 is 153 commits beyond that point;
- fork squashed diff changes 808 files;
- 203 runtime files changed in the fork;
- 233 runtime files changed upstream afterward;
- 70 runtime files overlap.

Conclusion: use E2 as a behavioral oracle and selective source donor. Do not merge or rebase the entire branch into E1.

### 9.5 Official pull requests

All 99 PRs returned by the official API were inspected for GLM/MLA/MTP relevance.

| PR | Status | Relevance |
|---|---|---|
| [#158](https://github.com/turboderp-org/exllamav3/pull/158) | closed, unmerged, author-closed, no review | Registered `GlmMoeDsaForCausalLM` as an alias of an old DeepSeek-V2 path. No GLM DSA, MTP, or correct chat. Useful only as evidence that a dense MLA alias is viable. |
| [#246](https://github.com/turboderp-org/exllamav3/pull/246) | open draft, unmerged | Blackwell route-packed EXL3 MoE kernel. Based on v0.0.43. Maintainer requires retargeting and benchmarks because current upstream has a new ticket scheduler and SM allocation. |
| [#239](https://github.com/turboderp-org/exllamav3/pull/239) | closed, unmerged | Disables unsafe parallel MUL1 quantization after a full GLM conversion failure. Fruit is already MCG-encoded; not a runtime solution. |
| [#234](https://github.com/turboderp-org/exllamav3/pull/234) | merged | Generic TP gather self-copy offset fix; already upstream. |
| [#223](https://github.com/turboderp-org/exllamav3/pull/223) | closed after independent fix | Generic multi-GPU MTP memory fault; resolved upstream. |
| [#225](https://github.com/turboderp-org/exllamav3/pull/225) | closed, unmerged | Qwen MTP TP LM-head fix. Useful historical pattern, not required for TP1 Fruit. |

PR #246 should be treated as scheduling inspiration only. The official maintainer’s exact concern is that v0.0.43 used a round-robin MoE scheduler while current code has a ticket scheduler and changed SM allocation.

---

## 10. DSA semantic conflict

This is the largest unresolved correctness issue.

### 10.1 Declared checkpoint behavior

`indexer_types` declares three full indexers followed by ten shared layers. The remichu architecture follows this literally.

### 10.2 Released r28 behavior

Static inspection of `vllm-r28/vllm/model_executor/models/deepseek_v2.py` found:

- `index_topk_pattern` is honored when non-null;
- otherwise frequency skipping activates only with `use_index_cache=true`;
- Fruit has `index_topk_pattern=null` and no `use_index_cache` field;
- therefore r28 appears to build/use an indexer at every stored layer.

### 10.3 Producer acknowledgement

`proxy-fruit/REVIEW.md` says later-layer pattern handling remains under review.

### 10.4 Why published smoke tests do not resolve it

At context length `T <= index_topk == 2048`, every causal key is retained. All of these schedules can load and produce plausible short outputs while differing beyond 2,048 tokens.

### 10.5 Required resolution

Before claiming long-context ExLlama support:

1. Capture r28 index selections/logits at 2,047, 2,048, 2,049, 4,096, and at least one 20k+ context.
2. Compare:
   - r28 every-layer indexer behavior;
   - declared full/shared reuse;
   - dense BF16 attention.
3. Decide whether compatibility means the deployed r28 behavior or the checkpoint’s declarative schedule.
4. Encode the chosen schedule explicitly and remove conflicting fallback metadata.

The dense BF16 training graph is the quality reference; the r28 service is the deployed compatibility reference. Those are not necessarily identical at long context.

---

## 11. BF16 reference limitations

The published M3 BF16 twin corresponds to base M2, not SFT M1.

M4 contains public state files including:

- `final/fruit_v1_annealed.pt`;
- `final/fruit_v1_instruct.pt`;
- resumable ~20 GB stage checkpoints.

However, P1’s review notes that the exporter still depends on local/unpublished `glm_franken.py`/encoder pieces and incomplete pinning. Therefore:

- use M3 for base-model architecture and base-SIQ parity;
- use r28 golden logits for the published Instruct weights;
- do not claim an independently reproducible BF16 Instruct export until exporter dependencies are complete.

---

## 12. Recommended implementation design

### Stage A — dense BF16 trunk

1. Add `GlmMoeDsaConfig` / `GlmMoeDsaModel` on current E1.
2. Reuse `DeepseekV3Model` geometry and `MLAttention`; do not port old `mla.py` wholesale.
3. Use ordinary dense MLA, TP1, MTP off, context <=2,048.
4. Force off old/full-model special paths when testing the fork:
   - `GLM_MLA_ABSORBED=0`;
   - `GLM_DSA_SPARSE=0`;
   - no B12X native full-model kernel.
5. Apply the repository tokenizer/chat template explicitly.
6. Compare fixed-position logits and greedy output to M3.

**INFERENCE:** an even earlier proof can be made by changing a copied BF16 config architecture to `DeepseekV3ForCausalLM`, because the main trunk graph and tensor names match. This is a disposable experiment, not a permanent implementation.

### Stage B — SIQ correctness path

1. Detect only the strict Fruit marker:

   ```text
   hybrid_tr3_tail.format == "exl3-trellis"
   codebook == "mcg"
   tp == 1
   tensor schema matches
   ```

2. Cross-check `tier_bitmap.json`, K values, MCG marker, and actual tensor presence.
3. Select expert key templates with `.rank0` only for this storage contract.
4. Set `no_reconstruct=True` or add an automatic heterogeneous-K fallback.
5. Keep DSA and MTP off initially.
6. Compare base SIQ against M3 and Instruct SIQ against captured r28 goldens.

### Stage C — permanent mixed-K dispatch

1. Inspect each expert’s K from `trellis.shape[-1] / 16`.
2. Cross-check metadata; tensor truth wins only after a fail-closed mismatch report.
3. Build global-to-tier and tier-to-global expert maps.
4. Route once using global expert IDs.
5. Invoke homogeneous K4 and K3 EXL3 MoE passes separately.
6. Accumulate both routed outputs and the BF16 shared expert.
7. Preserve the individual-expert path as the oracle.
8. Promote only after K3-only, K4-only, and mixed-route tests pass.

This normally costs two quantized MoE launches per projection stage instead of up to 24 per token. It is simpler than importing SparkInfer and supports arbitrary tier maps.

### Stage D — DSA

1. Reuse current `MLAttention` cache architecture.
2. Add GLM indexer projections:
   - `wq_b`;
   - `wk`;
   - `weights_proj`;
   - `k_norm.weight` and `k_norm.bias`.
3. Reuse normalized Q-LoRA state where safe.
4. Apply GLM indexer RoPE with theta 500,000 and the exported interleaved layout.
5. Cache indexer K rows alongside latent KV and RoPE K.
6. Update page allocation, copy, fork, rewind, and speculative rollback operations.
7. Initially use full-score chunked top-k for correctness.
8. Resolve the full/shared schedule before performance tuning.

Approximate FP16 cache accounting at the model’s 65,536-token limit:

- latent MLA cache: `13 × 65,536 × 2 × (512 + 64)` ≈ 0.914 GiB;
- indexer K cache: `13 × 65,536 × 2 × 128` ≈ 0.203 GiB;
- combined, before paging/workspaces/MTP: ≈ 1.12 GiB.

### Stage E — MTP K=1

1. Add a GLM MTP component at `model.layers.13`.
2. Reuse current `Qwen3_5MTPInputLayer` behavior:

   ```text
   eh_proj(cat([enorm(token_embedding), hnorm(target_hidden)]))
   ```

3. Add one MLA+MoE draft block.
4. Load `shared_head.norm` and borrow the target embedding/LM head.
5. Use native/default draft length 1.
6. Test accepted and rejected draft rollback across latent, indexer, and RoPE caches.
7. Do not generalize to multi-token MTP until K=1 matches the published 79% protocol evidence on the same workload.

### Stage F — optimization

Only after correctness:

- profile native two-tier passes;
- benchmark PR #246’s route-packed concept against current ticket scheduling;
- consider one-grid mixed-K or SparkInfer dependency;
- optimize DSA scorer/top-k and sparse attention;
- add CUDA graphs last;
- add cache quantization last, separately from architecture validation.

---

## 13. Validation matrix

### 13.1 Loader and schema

- exact pinned revision and manifest;
- every expected tensor consumed or explicitly reported as intentionally unused;
- fail on unknown MCG markers;
- fail on K outside `{3,4}`;
- fail on tier-map/header disagreement;
- fail on unsupported TP/rank layouts;
- fail on malformed `.rankN` names;
- no silent fallback to generic ModelOpt NVFP4.

### 13.2 Expert execution

Test routes that are:

- entirely K3;
- entirely K4;
- mixed K3/K4;
- repeated across batches;
- zero-hit for one tier;
- boundary expert IDs 95/96 and 255;
- all 256 experts under calibration/forced routing.

Compare:

1. BF16 oracle;
2. individual `LinearEXL3` slow path;
3. two-tier fused path;
4. optional SparkInfer path.

Validate router IDs, normalized weights, shared-expert contribution, final residual, and logits.

### 13.3 DSA

Lengths:

```text
1, 2, 2047, 2048, 2049, 4096, 20000, 65536
```

Cases:

- cache and no-cache logits;
- prefill vs decode;
- repeated-prefix reuse;
- different cache lengths in one batch;
- chunked prefill;
- page boundaries 255/256/257;
- cache fork/copy/rewind;
- high-position RoPE;
- full/shared and every-layer schedules;
- dense-vs-sparse attention below the top-k boundary;
- fixed top-1/top-10/KL comparisons above it.

### 13.4 MTP

- K=1 initialization;
- target/draft greedy agreement;
- accepted token cache commit;
- rejected token rollback;
- EOS variants and pad token;
- chat template with `enable_thinking=false`;
- output equality with MTP disabled;
- acceptance measured on the publisher’s protocol prompts, not an unrelated easy string.

### 13.5 Performance and memory

Measure separately:

- BF16 trunk;
- SIQ individual-expert oracle;
- SIQ two-tier native path;
- optional one-grid path;
- DSA off/on;
- MTP off/on;
- short and long context;
- peak and steady VRAM;
- CUDA graph capture/replay.

Do not mix cache quantization or online dense projection quantization into the first architecture result.

---

## 14. Revised effort estimates

All estimates are **INFERENCE** for one engineer familiar with ExLlamaV3/CUDA, with suitable GPU access and model files already local. They exclude upstream review latency.

| Deliverable | Start from remichu E2 | Clean current E1 port |
|---|---:|---:|
| BF16 trunk, TP1, MTP/DSA off, <=2,048 | 0.5–1.5 engineer-days | 1–3 engineer-days |
| SIQ trunk, `.rank0`, slow individual-expert mixed-K | 2–4 days cumulative | 3–7 days cumulative |
| Generic latent-cache + DSA long-context correctness | +2–5 days | +4–8 days |
| GLM MTP K=1 and rollback tests | +1–3 days | +2–5 days |
| Native two-tier K3/K4 MoE and benchmarks | +4–8 days | +4–10 days |
| Release-quality single-GPU Fruit support | **2–3 weeks total** | **3–6 weeks total** |
| r28/SparkInfer-class performance, cache variants, TP/PP, one-grid mixed-K | **5–8 weeks total** | **5–8+ weeks total** |

Interpretation:

- “Can it emit correct short-context tokens?” is now a days-scale task.
- “Can it implement the model limit with a resolved DSA contract?” remains roughly 2–4 weeks.
- “Can it be a complete single-GPU official product?” remains 3–6 weeks.
- “Can it match the full custom vLLM/SparkInfer production stack?” remains 5–8+ weeks.

The existing fork removes architecture uncertainty but not the long-context semantic decision, mixed-K production scheduler, current-upstream integration, or validation burden.

---

## 15. Risk register

| Risk | Level | Evidence / mitigation |
|---|---|---|
| DSA schedule ambiguity | High | Declared `indexer_types` conflicts with static r28 dispatch; capture long-context goldens before implementation. |
| No exact published BF16 Instruct repository | High for parity | Use M3 only for base; use r28 goldens for M1. |
| Mixed K fused scheduler | Medium | Slow path is available; implement two homogeneous tiers before one-grid work. |
| Stale remichu fork | Medium | 153 upstream commits, 70 overlapping runtime files; selective port only. |
| Full-model hard-coded kernels | Medium | Do not enable them for Fruit; use current dynamic `MLAttention`. |
| MTP/cache rollback | Medium-high | Existing framework helps, but GLM layer is MLA+DSA-specific. Test rejection paths. |
| Chat/protocol drift | Medium | Use repository template and all EOS IDs, not hard-coded old prompts. |
| Top-level quant metadata | Medium | Strictly detect `hybrid_tr3_tail`; fail closed. |
| Artifact TP/rank portability | Medium | Published contract says TP1 and `.rank0`; do not infer arbitrary resharding. |
| Performance claims | Medium | Existing validation is for different model/hardware or publisher-reported; remeasure Fruit. |
| Evaluation contamination | Certain | Aider/Exercism trajectories are explicitly present; do not claim clean scores. |
| Export reproducibility | Medium | Missing/local exporter dependencies remain documented; loading published artifacts is unaffected. |

---

## 16. Source reuse recommendation

### Port or adapt

From E2:

- GLM config fields and architecture registration;
- exact tensor names;
- `indexer_types` handling as one candidate contract;
- GLM target verification normalization;
- MTP component structure;
- DSA/HF parity test ideas;
- full-decompressed and MTP validation harness patterns.

From current E1:

- `DeepseekV3Model` trunk structure;
- current `MLAttention`;
- current dots router and `BlockSparseMLP`;
- current `LinearEXL3` decoder;
- current DSA Triton primitives;
- current MTP generator and `Qwen3_5MTPInputLayer`;
- current cache/page APIs and ticket-scheduled MoE kernels.

From S1/R3:

- mixed-tier map semantics;
- route-packing design;
- shared-rotation invariants;
- one-grid behavior as an optional optimized target;
- validation cases for mixed expert counts.

### Do not port wholesale

- E2’s 808-file squashed diff;
- E2’s full-model fixed-shape `glm_mla_attn.cu`;
- E2’s old `mla.py` as a replacement for current `MLAttention`;
- E2’s multi-GPU/host-paging machinery for a TP1 5B first target;
- E4/PR #246’s v0.0.43 scheduler;
- custom vLLM ModelOpt override logic as generic ExLlama behavior;
- SparkInfer as a correctness prerequisite.

---

## 17. Reproducibility notes for this research

### Research checkouts inspected
The following worktree names identify the pinned source snapshots used during
the audit. They are not vendored in this repository.

```text
official-exllamav3/
remichu-exllamav3/
custom-exllamav3/
sparkinfer/
sparkinfer-r28/
sparkinfer-r28-full/
vllm-r28/
proxy-fruit/
blackwell-llm-docker/
```

### Static methods used

- Hugging Face model/tree/config/index APIs at immutable revisions;
- HTTP `Range` requests for safetensors 8-byte header lengths and JSON headers;
- manifest parsing and LFS SHA-256 comparison;
- per-tensor shape/dtype/name validation;
- tier-map count reconstruction;
- GitHub API enumeration of every official pull request returned;
- exact-symbol searches in issues, commits, branches, and forks;
- source diffs and fork-point/overlap counts;
- code-hash comparisons for decoder primitives;
- direct source inspection of architecture, loader, MoE, DSA, cache, and MTP paths.

### Not performed

- model download and CUDA execution;
- BF16 or SIQ ExLlama load;
- generated logits or tokens;
- performance or VRAM measurements;
- long-context DSA comparison;
- MTP acceptance reproduction;
- mutation of any runtime source repository.

The audit environment did not contain PyTorch. This was consistent with the research-only scope; no runtime-success claim is made.

---

## 18. Primary source links

### Model and training

- [SIQ Fruit Instruct, pinned](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-Instruct/tree/48452ef397d8b4a4d6d0c00ea376a2abb3ef6314)
- [SIQ Fruit base, pinned](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit/tree/c1798e3676fa16b4a874381171adab1e3033fbd5)
- [BF16 twin, pinned](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-bf16/tree/ff1178d233fddc644dc053c723d58839eb921334)
- [Training checkpoints, pinned](https://huggingface.co/malaiwah/fruit-phase1-ckpt/tree/fc883a67d8ab02b66cad5575ba63a781bc280fa1)
- [Trainer/exporter/validation source, pinned](https://github.com/malaiwah/proxy-fruit/tree/978d104bfb93902b144a384a2f129bd2d3e0a875)
- [Review ledger](https://github.com/malaiwah/proxy-fruit/blob/978d104bfb93902b144a384a2f129bd2d3e0a875/REVIEW.md)
- [Exporter](https://github.com/malaiwah/proxy-fruit/blob/978d104bfb93902b144a384a2f129bd2d3e0a875/export_fruit.py)
- [Trainer](https://github.com/malaiwah/proxy-fruit/blob/978d104bfb93902b144a384a2f129bd2d3e0a875/train_fruit.py)

### Published runtime

- [GLM-5.2 v20/r28 runtime guide](https://github.com/local-inference-lab/rtx6kpro/blob/81682d81f8dc71fa084be0a86e10c70766d894eb/models/glm5.2_v20.md)
- [SparkInfer](https://github.com/local-inference-lab/sparkinfer/tree/680d8195b80420296d7fed2688b75406be15eb38)
- [r28 vLLM integration lock](https://github.com/local-inference-lab/blackwell-llm-docker/blob/d780c393677eb0dd9dc5d2e09b98230313ec50cf/patches/releases/gilded-gnosis-v20-r28/vllm/integration.lock.json)
- [r28 SparkInfer integration lock](https://github.com/local-inference-lab/blackwell-llm-docker/blob/d780c393677eb0dd9dc5d2e09b98230313ec50cf/patches/releases/gilded-gnosis-v20-r28/sparkinfer/integration.lock.json)

### ExLlamaV3

- [Official ExLlamaV3 v1.4.0 source](https://github.com/turboderp-org/exllamav3/tree/791c83073f7f90c44f765a0ceeab7a05fa15b96b)
- [Current DeepSeek-V3 architecture](https://github.com/turboderp-org/exllamav3/blob/791c83073f7f90c44f765a0ceeab7a05fa15b96b/exllamav3/architecture/deepseek_v3.py)
- [Current latent attention](https://github.com/turboderp-org/exllamav3/blob/791c83073f7f90c44f765a0ceeab7a05fa15b96b/exllamav3/modules/mla_attn.py)
- [Current block-sparse MoE](https://github.com/turboderp-org/exllamav3/blob/791c83073f7f90c44f765a0ceeab7a05fa15b96b/exllamav3/modules/block_sparse_mlp.py)
- [Current MTP input fusion](https://github.com/turboderp-org/exllamav3/blob/791c83073f7f90c44f765a0ceeab7a05fa15b96b/exllamav3/modules/arch_specific/qwen3_5_mtp.py)
- [GLM fork, pinned](https://github.com/remichu-ai/exllamav3/tree/0104e7ff3481a10dbc4850a9a36b9742b3bb4bf3)
- [GLM architecture implementation](https://github.com/remichu-ai/exllamav3/blob/0104e7ff3481a10dbc4850a9a36b9742b3bb4bf3/exllamav3/architecture/glm5_moe_dsa.py)
- [GLM MTP implementation](https://github.com/remichu-ai/exllamav3/blob/0104e7ff3481a10dbc4850a9a36b9742b3bb4bf3/exllamav3/architecture/glm5_moe_dsa_mtp.py)
- [Companion vLLM integration](https://github.com/remichu-ai/vllm/tree/glm52-share-cleanup/EXL3)
- [Historical architecture PR #158](https://github.com/turboderp-org/exllamav3/pull/158)
- [Route-packed MoE draft PR #246](https://github.com/turboderp-org/exllamav3/pull/246)
- [MUL1 conversion PR #239](https://github.com/turboderp-org/exllamav3/pull/239)

## Final recommendation

Implement native GLM-5.2 DSA support in current official ExLlamaV3, not an SIQ-specific SparkInfer port and not a merge of the old GLM fork. Establish three explicit milestones:

1. BF16 dense trunk correctness;
2. SIQ slow-path correctness with `.rank0` and heterogeneous per-linear K;
3. optimized native two-tier MoE, followed by DSA and MTP validation.

Do not claim long-context compatibility until the index-sharing schedule is resolved with >2,048-token golden evidence.