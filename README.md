# proxy-fruit

**A documented, measured program for training serving-faithful proxy models
on consumer and spot hardware.**

This repo trains **GLM-5.2-SIQ-Fruit**: a 5B-parameter (~0.46B active)
Mixture-of-Experts model that is a *byte-exact architectural mimic* of the
GLM-5.2 production architecture — MLA attention (kv_lora 512, rope 64, head
dims 192/256 preserved), DSA sparse-attention indexer (32×128, carried),
256 routed experts + 1 shared with the sigmoid `noaux_tc` router and
`e_score_correction_bias`, one MTP speculative-decoding layer, and the GLM
154,880-token tokenizer.

**Why:** not to compete with anything — to be an *organ donor* for a
quantization + serving stack. A trained (not random-init) architecture-exact
small model lets you exercise Trellis/SIQ quantization exactness, sparse-MLA
kernels, MTP acceptance rates, chat-template stop behavior, and long-context
paths on hardware you own, with quality signals a random-weight CI fixture
cannot provide. Total training cost: ~$300 of spot H200 time.

Sibling artifacts on Hugging Face:
- [`malaiwah/GLM-5.2-SIQ-Fruit-pilot`](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-pilot) — 413M pilot ("Clémentine"), serves on the real stack
- `malaiwah/GLM-5.2-SIQ-Fruit` — the 5B model (training in progress)
- [`malaiwah/fruit-phase1-ckpt`](https://huggingface.co/malaiwah/fruit-phase1-ckpt) — live training checkpoints
- [`malaiwah/fruit-phase1-shards`](https://huggingface.co/datasets/malaiwah/fruit-phase1-shards) — pre-tokenized corpus (7.55B tokens + SFT shards)
- [`malaiwah/GLM-5.2-Legume`](https://huggingface.co/malaiwah/GLM-5.2-Legume) / [`-v3`](https://huggingface.co/malaiwah/GLM-5.2-Legume-v3) — the layer-surgery lineage that preceded Fruit

## The toolchain

| Tool | Purpose |
|---|---|
| `train_fruit.py` | Single-file plain-PyTorch trainer (~1,100 lines). DDP, grad-ckpt, 8-bit/paged optimizers, chunked+checkpointed CE, MTP + indexer-distillation objectives, SFT loss masking, and every run-2 knob below — all env-gated. |
| `fruit_data_prep.py` | Phase-1 corpus: 9 sources → uint32 memmap shards + manifest. |
| `sft_data_prep.py` | SFT corpus: GLM-chat-templated conversations with assistant-only loss masks, yield-token + EOS supervision, conversation-start index, prefix-stability validation, mask-less replay channels. |
| `aider_traj_prep.py` | Convert Aider benchmark run trees into ShareGPT SFT data (pass-filtering, in-progress-run guard, compiler-spam elision). The standing corpus-refresh channel. |
| `export_fruit.py` | SIQ/Trellis export (exl3-trellis container, per-tier K3/K4/K6, FFN zero-padding trick for tile constraints). |
| `fruit_serve_*.py` | Serve gauntlets: fp8 + nvfp4 KV, sparse-MLA backend, small-prompt battery, long-context, MTP loading. |
| `probe_ckpt.py` | Mid-run checkpoint probe: separate-process load test, greedy generations with degeneracy detection, expert-load census, router-logit stats. |
| `jarvis_run.sh` / `restore_run1.sh` / `run_sft_prep.sh` | Spot-instance drivers: bootstrap → prep → staged training → upload; preemption recovery on any tier. |
| `repro_b12x_121.py` / `bench_prefill_107.py` | Kernel-issue reproductions filed upstream (b12x #121, #107). |
| `run2_tests.sh` | The validation matrix that gates every trainer feature on an RTX 5090 before it touches rented hardware. |

## Trainer knobs (all validated on RTX 5090, sm120)

Token-clock, tier-agnostic resumption (`TOKEN_BUDGET`) — schedules are
functions of tokens seen, so a checkpoint moves between 4×H200, 8×RTX 6000
Pro, or a single 5090 with an unbroken LR curve. `MOE_IMPL=grouped`
(`torch._grouped_mm`, bit-equivalent, −28% peak memory at BS=8).
`FP8_LINEAR` (tensorwise `_scaled_mm`, loss-parity-proven). `FP32_MASTER`,
`ZLOSS`/`ZLOSS_HEAD`, `SKIP_SPIKES`, `SNAPSHOT_SAVE`/`SNAPSHOT_FORK`
(process-based async checkpointing), `BIAS_BALANCE` (DSv3 aux-loss-free),
`TIE_EMB`, `NO_WD_EMB`, `LR_SCHED=wsd`, `INTRADOC_MASK`,
`DETERMINISTIC_DATA` (Pythia-style reproducible data order), `QNOISE`
(QAT-lite quantization-noise annealing), `MTP_W`/`MTP_W_END`.

## Measured findings (the part you may actually be here for)

See **[HARDWARE.md](HARDWARE.md)** and **[SFT_NOTES.md](SFT_NOTES.md)** for
the full ledger. Highlights, all measured on this hardware, not quoted:

- `torch.compile` is **3–4× slower** than eager for a 256-expert MoE
  (dynamic `counts.max()` graph-breaks every step).
- Raw FP8 `_scaled_mm` on consumer Blackwell (sm120) is **2.1–3.3× faster
  than bf16** at these GEMM shapes — but end-to-end FP8 training is a wash
  at 1024-hidden (cast overhead; MoE/CE remain bf16).
- Tied embeddings with default `N(0,1)` embedding init → **step-0 loss
  ≈ 380**. Tie only with std-0.02 init.
- `torch.compile` + activation checkpointing = no-checkpoint memory (it
  traces through `torch.utils.checkpoint`).
- Full 5.04B geometry trains on one 32 GB RTX 5090: `PagedAdamW8bit` +
  grad-ckpt + `expandable_segments`, 23.0 GiB peak. Plain 8-bit OOMs by
  20 MiB at the step-0→1 boundary.
- DDP + activation checkpointing needs `static_graph=True`.

## Reproduce

```
# 1. corpus (or pull the prebuilt shards from HF)
TOK_DIR=... OUT_DIR=shards python fruit_data_prep.py
# 2. pretrain (4xH200 example; token clock makes the tier a detail)
GEO_H=1024 GEO_NL=13 GEO_HEADS=16 GEO_QLORA=1024 GEO_DENSE_INTER=2048 \
GEO_MOE_INTER=512 ROPE_THETA=500000 MOE_IMPL=grouped SHARD_DIR=shards \
TOKEN_BUDGET=4600000000 SEQ=4096 BS=6 LR=3e-4 GRAD_CKPT=1 OPT_8BIT=1 \
torchrun --standalone --nproc_per_node=4 train_fruit.py
# 3. long-context + indexer-distill + SFT stages: see jarvis_run.sh
# 4. export + serve gauntlet
FRUIT_PT=... python export_fruit.py && python fruit_serve_test.py
```

## Disclosure

The SFT corpus can include Aider/Exercism benchmark trajectories
(`aider_traj_prep.py`): models trained with that channel are **permanently
contaminated for Aider-style evaluations**. This program optimizes for
serving fidelity, not leaderboards; model cards must carry this notice.

## License

Apache-2.0. The training data references public datasets under their own
licenses; Apache-2.0 *license text* is deliberately held out of the license
corpus as an evaluation needle.
