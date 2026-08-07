---
license: mit
language:
- en
library_name: vllm
pipeline_tag: text-generation
tags:
- glm
- moe
- siq
- trellis
- mla
- serving-proxy
- ci-fixture
- from-scratch
---

# GLM-5.2-SIQ-Fruit-pilot — Clémentine

The original micro Fruit experiment: a **413M-parameter, about 30M-active**
GLM-5.2 architectural mimic trained from scratch in roughly 13 minutes on one
RTX 5090, then exported through the SIQ/Trellis serving path.

Clémentine exists to make the entire train → quantize → load → generate → MTP
loop cheap enough to debug. It is a toy language model and CI fixture, not a
general assistant.

> **Runtime requirement:** this SIQ checkpoint needs a compatible
> b12x/SparkInfer + vLLM build. Stock vLLM and Transformers do not implement
> its `exl3-trellis` expert tensors.

## Architecture

- `GlmMoeDsaForCausalLM`: 3 dense + 3 MoE decoder layers + 1 MTP layer.
- Hidden size 512; 8 attention heads.
- 256 routed experts, top-8 routing, plus one shared expert.
- Production MLA dimensions retained: KV LoRA 512, QK no-RoPE/RoPE 192/64,
  value head 256.
- DSA indexer shape 32×128. Its weights are random-init/untrained; at contexts
  no longer than `index_topk`, the tested path is numerically dense.
- GLM tokenizer with 154,880 tokens.

The trained logical MoE intermediate size is 128. The corrected serving export
stores it as 256 by appending exact zero rows/columns; this preserves the model
function while satisfying the MTP Trellis kernel's CTA tile requirement.

## Training data

15.6M tokens drawn from TinyStories, the GLM-5.2 REAP recall calibration set,
and repeated SPDX license text. The model reliably continues TinyStories-style
openings and has memorized fragments of common license templates. That is the
extent of the intended capability.

The original BF16 source state remains in this repository as
`training/fruit_pilot.pt`, SHA-256
`7c7d07f0d7c2944faf8701de6e174fcac7ba48a854de87b194ef9e8e33eede49`.

## Corrected SIQ artifact

- Ordinary MoE layers: 96 K4 + 160 K3 experts.
- MTP layer: uniform K3.
- Non-expert tensors: BF16.
- Tensor payload: **595,150,336 bytes (0.554 GiB)**.
- RoPE theta: **10,000** in both supported configuration locations.
- Export conversion: 28 half-split-to-interleaved RoPE projection
  permutations plus the MTP `eh_proj` input-half swap.
- `MANIFEST.sha256` covers the 17 serving files. Tools, logs, the card, and the
  training checkpoint are outside that serving-artifact manifest; the source
  checkpoint hash is pinned above.

## Measured validation of the corrected export

RTX 5090; custom gilded-gnosis r25, `fp8_ds_mla`, eager mode:

| check | result |
|---|---|
| 1/2/5/8/9-token prefill battery | PASS |
| trainer-to-served parity, 42 positions | top-1 **100.0%**, mean top-10 overlap **95.0%** |
| parity mean top-K drift | 0.0049; truncated diagnostic, not KL |
| MTP k=1 acceptance | **499/524 = 95.2%** |
| CC1 decode without MTP | 117.3 tok/s |
| CC1 decode with MTP k=1 | 89.5 tok/s |

MTP is slower on this tiny model because orchestration overhead dominates; its
acceptance rate validates the draft path, not a speedup claim.

Observed greedy continuations include:

> `Once upon a time` → `, there was a little girl named Lily...`

> `MIT License\n\nCopyright` → ` (c) [year] [fullname]...`

## Release correction

Revision `bfddc762ed7d75eac7b8706395eddff4fcb42220` copied legacy trainer
weights into serving tensors without the required RoPE channel permutation or
MTP `eh_proj` half-swap, retained theta 8,000,000 instead of the trained 10,000,
and left the logical 128-wide MTP experts incompatible with the current
Trellis kernel.

The 2026-08-07 rebuild authenticates the original source checkpoint, applies
both convention conversions, exact-zero-pads the served MoE intermediate size
to 256, re-encodes the expert tensors, regenerates the manifest, and passes the
validation table above. Consumers pinned to the old revision should update.

## Reproduction

The repository includes the original trainer/exporter snapshots and logs.
The maintained, fail-closed exporter and parity/MTP harnesses live at
[github.com/malaiwah/proxy-fruit](https://github.com/malaiwah/proxy-fruit).
For the larger production-shape fixture, use
[GLM-5.2-SIQ-Fruit](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit).

## Limitations

- 15.6M training tokens are insufficient for broad language capability.
- The indexer was not trained; this release does not establish long-context
  sparse-attention quality.
- License-template continuation is memorization behavior, not legal advice.
- Do not deploy this model.
