---
license: apache-2.0
library_name: pytorch
tags:
- glm
- moe
- training-checkpoint
- reproducibility
- serving-proxy
---

# Fruit Phase-1 training archive

Model-only weights, resumable optimizer/RNG checkpoints, metrics, and plots for
[GLM-5.2-SIQ-Fruit](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit).
This is a **151.54 GB (141.14 GiB) training archive**, not a Transformers or
vLLM serving repository.

![Phase-1 training progress](val_progress.png)

Phase 1 completed 57,293 steps and about 5.43B sampled tokens across MAIN,
LONG, DISTILL, QNOISE, and SFT. The plot is generated from the durable metrics
ledger; raw validation points are in `progress/fruit_v1_val.jsonl`.

## Model-only stage weights

| file | stage | size |
|---|---|---:|
| `final/fruit_v1_main.pt` | MAIN, 4,096 context | 10.081 GB |
| `final/fruit_v1_long.pt` | LONG, 16,384 context | 10.081 GB |
| `final/fruit_v1_final.pt` | DSA indexer DISTILL; pre-anneal control | 10.081 GB |
| `final/fruit_v1_annealed.pt` | QNOISE anneal; published base-model source | 10.081 GB |
| `final/fruit_v1_instruct.pt` | assistant-masked SFT | 10.081 GB |

The canonical base release source is `final/fruit_v1_annealed.pt`, SHA-256
`98ac7cb4f7799194424782b505d622069fecf4dbca5f5acb2658f2a66c3631f6`.
The Instruct source is `final/fruit_v1_instruct.pt`, SHA-256
`32dbf82d40b88a92b8dccd563c593b5971be358cf11895eb150f18644ff93c27`.

## Resumable checkpoints

`checkpoints/*_ckpt.pt` and `final/fruit_v1_instruct_ckpt.pt` are roughly
20.2 GB each. They contain model parameters, optimizer state, step and token
counters, and Python/NumPy/Torch/CUDA RNG state. Use them only with the
matching trainer and geometry in
[proxy-fruit](https://github.com/malaiwah/proxy-fruit).

These full-state files are Python pickle containers and require trusted
`torch.load(..., weights_only=False)`. Do not load a checkpoint from an
untrusted revision. `MANIFEST.sha256` authenticates the published archive
files except the card, Git attributes, and the manifest itself.

## Legacy convention contract

The Phase-1 weights predate the fail-closed Run-2 markers
`serve_conv_v` and `rope_theta_trained`. They use:

- half-split trainer RoPE with theta **500,000**;
- trainer MTP `eh_proj(cat[hidden, embed])` input order;
- stacked routed-expert tensors (`mlp.w_gate`, `mlp.w_up`, `mlp.w_down`).

Any export or resume must explicitly select the legacy convention. The
maintained exporter applies the RoPE channel permutation and MTP input-half
swap. Treating these files as serving-native silently corrupts logits and MTP
behavior.

## Downloading one artifact

Avoid cloning the complete 151 GB repository when only one stage is needed:

```bash
hf download malaiwah/fruit-phase1-ckpt \
  final/fruit_v1_annealed.pt \
  --local-dir ./fruit-checkpoint

sha256sum ./fruit-checkpoint/final/fruit_v1_annealed.pt
```

For resume, download the matching `*_ckpt.pt` plus the tokenizer/data inputs
documented by the trainer. Model-only `.pt` files cannot resume optimizer or
RNG state.

## Lineage and evidence

- Serving exports: [base SIQ](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit),
  [Instruct SIQ](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-Instruct),
  and [BF16 CPU twin](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-bf16).
- Tokenized inputs: [fruit-phase1-shards](https://huggingface.co/datasets/malaiwah/fruit-phase1-shards).
- Trainer, stage drivers, restore scripts, smoke suite, and review ledger:
  [github.com/malaiwah/proxy-fruit](https://github.com/malaiwah/proxy-fruit).

The files are retained for reproducibility and codec experiments. They are not
packaged for `AutoModel.from_pretrained()` and should not be presented as a
ready-to-serve model.
