# proxy-fruit

**A documented, measured program for training serving-faithful proxy models
on consumer and spot hardware.** (~$300 spot compute, ~2 days on 4×H200,
every step rehearsed first on one RTX 5090.)

![live training progress](https://huggingface.co/malaiwah/fruit-phase1-ckpt/resolve/main/val_progress.png)
*Live from the current Phase-1 run — regenerated every ~3 h by the
toolchain's [ledger publisher](progress_publish.py): per-source val curves,
train/MTP loss, LR + grad-norm, router aux + step time, host/GPU
telemetry, with incarnation markers at every restart or node change.*

This repo trains **GLM-5.2-SIQ-Fruit**: a 5B-parameter (~0.46B active)
Mixture-of-Experts model that is a *production-shape serving proxy* for
the GLM-5.2 architecture. As far as we could find, this is the first
public clean-room GLM-5.2 serving-proxy program combining trained
weights, production-critical 256-expert MLA/DSA/MTP geometry, the exact
tokenizer and serialization invariants, and an end-to-end SIQ/Trellis
export-and-serve regression workflow on accessible hardware — but the
parts have close precedents (see prior art below), and known
train-vs-serve parity gaps are tracked in REVIEW.md until round-trip
numerical parity is demonstrated. "Serving proxy" here means a **CI fixture
for a serving/quantization stack** — distinct from the μP/DoReMi sense of
"proxy model" (small models proxying *training dynamics*); ours proxies
*serving behavior*. "Architecture-complete" means: same computation graph
and serialization layout (state-dict keys, config schema, tokenizer),
serving-critical dimensions preserved exactly, remaining dimensions scaled
by documented rules — see the fidelity manifest below. Weights are its own.

**Closest prior art** (each bracketing one half of the idea):
[inference-optimization/GLM-5.2-0.8B-A0.8B](https://huggingface.co/inference-optimization/GLM-5.2-0.8B-A0.8B)
is a *trained* tiny GLM-5.2 test model but scales away the
production-critical geometry (8 experts, reduced MLA/indexer dims, no MTP
tensors); [yujiepan/glm-5.2-tiny-random](https://huggingface.co/yujiepan/glm-5.2-tiny-random)
preserves nearly all the production shapes including a real MTP layer but
with random weights (no quality-bearing quantization/acceptance signal).
Fruit's contribution is the conjunction: trained weights ON the
production-critical shapes, plus the export/serve regression loop.

**Why train one, instead of the alternatives?**
- *Random-init tiny fixtures* (hf-internal-testing, yujiepan's excellent
  `*-tiny-random` zoo) validate plumbing but produce meaningless output
  distributions — quantization error and speculative-decoding acceptance
  aren't signals on noise, and that ecosystem typically omits the MTP
  module and indexer entirely (documented in the cards). vLLM's own RFC
  [#28135](https://github.com/vllm-project/vllm/issues/28135) describes the
  resulting gap: spec-decode regressions that pass correctness tests.
- *Synthetic acceptance rates* (e.g. Modular's `--synthetic-acceptance-rate`)
  bypass the very code paths under test.
- *Shearing/pruning the parent* (Sheared-LLaMA-style) inherits real weight
  statistics but requires loading the parent (far beyond hobbyist VRAM),
  entangles the fixture with parent weights licensing, and still needs
  continued pretraining. From-scratch keeps the fixture clean-room.

A *trained* architecture-complete mimic makes Trellis/SIQ quantization
deltas, sparse-MLA kernel behavior, MTP acceptance, chat-template stops,
and long-context paths all *meaningful, quality-bearing signals* on
hardware you own.

## Architecture-fidelity manifest

| config key | GLM-5.2 parent | Fruit | rule |
|---|---|---|---|
| `kv_lora_rank` | 512 | **512** | KEPT (kernel parity) |
| `qk_rope_head_dim` / `qk_nope_head_dim` / `v_head_dim` | 64 / 192 / 256 | **64 / 192 / 256** | KEPT (KV head_size 576 byte-exact) |
| `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | 256 / 8 / 1 | **256 / 8 / 1** | KEPT (router + dispatch parity) |
| `routed_scaling_factor` | 2.5 | **2.5** | KEPT |
| `index_n_heads` × `index_head_dim`, `index_topk` | 32×128, 2048 | **32×128, 2048** | KEPT (DSA indexer carried) |
| `num_nextn_predict_layers` (MTP) | 1 | **1** | KEPT |
| `first_k_dense_replace` | 3 | **3** | KEPT |
| `vocab_size` / tokenizer | 154,880 | **154,880** | KEPT (same tokenizer files) |
| `hidden_size` | 6144 | 1024 | scaled ÷6 |
| `num_hidden_layers` | 78 | 13 | scaled ÷6 |
| `num_attention_heads` | 64 | 16 | scaled ÷4 (≥8 for sparse-MLA dispatch) |
| `q_lora_rank` | 2048 | 1024 | scaled |
| `moe_intermediate_size` | 2048 | 512 | scaled ÷4 (tile-constraint aware) |
| `intermediate_size` (dense) | 12288 | 2048 | scaled ÷6 |

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
Pro, or a single 5090 with an unbroken LR curve. (The token-clock *concept*
is prior art — IBM's [Power Scheduler](https://arxiv.org/abs/2408.13359),
WSD-family schedules, and transformers
[#43708](https://github.com/huggingface/transformers/issues/43708)
documents the resume bug it fixes; our contribution is the tier-agnostic
single-file implementation, kill/resume-validated across batch sizes.) `MOE_IMPL=grouped`
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

## Regression testing

Every substantive toolchain change runs the [smoke suite](SMOKE_PLAN.md)
— 20 tests covering data prep, every trainer knob, DDP, and all resumption
paths, on a ~$11 spot node. Example output (tiny-geometry smoke run on
4× RTX 6000 Pro, incl. the telemetry/throttle-forensics rows):

![smoke suite output](https://huggingface.co/malaiwah/fruit-smoke/resolve/main/val_progress.png)

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
