# Fruit Phase-1 training — validated hardware configs

Model: 5.04B total / ~0.46B active (GlmMoeDsa mimic: hidden 1024, 3 dense +
10 MoE + MTP, 256 experts top-8, `moe_inter` 512, heads 16, GLM tokenizer
154,880 vocab). Trainer: plain PyTorch bf16 (`train_fruit.py`) — StackedMoE
grouped-GEMM, SDPA attention, chunked+checkpointed cross-entropy.

## 1× H100 80GB (validated, tight — "the compromises config")

Every knob below was *required*; removing any one OOMs at the step 0→1
boundary (where optimizer states materialize):

| knob | value | why |
|---|---|---|
| `BS` | **4** (16k tok/step) | backward transients at BS=8 + states cross 79 GiB |
| `GRAD_CKPT` | 1 | per-block recompute; without it activations alone exceed VRAM |
| `OPT_8BIT` | 1 (bitsandbytes AdamW8bit) | Adam states 20.2 → ~5 GiB |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | ~19 GiB fragmentation otherwise |
| chunked CE | adaptive (`2048 // BS` rows) | fp32 logits chunk is BS-dependent; fixed 2048 rows = 10 GiB at BS=8 |
| `zero_grad(set_to_none=True)` | **before** the forward | else last step's 10 GiB of grads coexist with the new graph |

Measured: **1.17 s/step ≈ 14k tok/s**, ~76 GiB peak, stable past warmup.
Economics: ~$0.19 per M tokens at $2.69/h — fine for smoke, painful for 5B
(≈ $260). Use for validation runs, not full pretrains, at this model size.

## 1× H200 141GB (production config for this run)

- **`BS=8`, `GRAD_CKPT=1`, `OPT_8BIT=1`, expandable_segments, NO compile.**
- Measured eliminations: BS=16 without ckpt needs >137 GiB; BS=16 *and*
  BS=12 with ckpt + `torch.compile` also OOM — **torch.compile traces
  through `torch.utils.checkpoint` and defeats activation recompute**, so a
  compiled run costs no-checkpoint memory. Never combine compile with
  checkpointing-as-your-fit-margin without verifying peak memory first.
- Spot at $1.99/h (IN2) with HF checkpoint-push every 600 steps makes
  preemption a ≤10-min loss.

## 1× RTX 5090 32GB (validated 2026-08-06 — the free local SFT option)

Full 5.04B geometry trains at **BS=1, SEQ=4096, GRAD_CKPT=1,
`OPT_8BIT=paged` (PagedAdamW8bit), expandable_segments**: peak 23.0 GiB,
~1.2–1.5 s/step (~3k tok/s → 300M-token SFT ≈ a day). Plain `OPT_8BIT=1`
OOMs by 20 MiB at the step-0→1 boundary — the paged optimizer's
CPU-pageable states are what make the card viable. Rehearsal/fallback tier;
production SFT runs on the rental (~2 h).

## AIBeast (10.15.0.166) — emergency training fallback (surveyed 2026-08-06)

4x RTX PRO 6000 Blackwell Workstation 96GB (sm120), 251 GB RAM, 64 cores,
driver 595.71.05. Runs the same cu132/sm120 container image as the 5090
rehearsal tier — zero new validation needed. Fallback config: NPROC=4 BS=4
(BS=6 needs a fit probe even with MOE_IMPL=grouped), restore_run1.sh +
GROUPED=1. Trade-off: takes GLM-5.2 production down for the duration —
Michel accepts this when spot economics/capacity justify it. Preemption
ladder: (1) new H200 quad; (2) 8x RTX 6000 Pro BS=3 (schedule-identical
98,304 tok/step); (3) 4x RTX 6000 Pro BS=4 after probe; (4) AIBeast.

## Universal lessons

- **Memory grows ~40% over the first ~150 steps** (measured: 96 → 138 GiB
  at BS=8/H200): the MoE's per-step varying group sizes (padded `[256, M,
  ...]` buffers) make the allocator's pools grow before plateauing. Budget
  peak = step-50 reading × 1.45, or watch until step 200 before trusting a
  fit. Step time creeping upward is the early warning.

- The step 0→1 boundary is where memory lies: step 0 always fits (no
  optimizer states yet). Never declare victory before step 2.
- Verify remote edits with `grep` on the remote file, never by upload exit
  code; verify processes by GPU memory, never by `pgrep -f` whose pattern
  appears in your own SSH command line (self-match).
- Pin the torch wheel to the instance's driver (`nvidia-smi` CUDA version);
  `pip install torch` picks the newest CUDA build, which may not load.
- Pre-tokenized shards in a private HF dataset repo make instances
  disposable: any box resumes with a 2-minute pull.

## Run-2 speedrun kit (researched 2026-08-06 — ALL IMPLEMENTED and
## 5090-validated as env knobs; run 1 kept its frozen recipe)

- `FP32_MASTER=1` — fp32 master weights (kills the bf16 dead-zone)
- `ZLOSS=1e-4` — sigmoid-adapted router logit penalty (ST-MoE)
- `SKIP_SPIKES=1` — EMA grad-norm spike skip (ZClip-lite; caught a live
  spike at 4x EMA during validation)
- `SNAPSHOT_SAVE=1` — pinned-mirror async checkpoint (0.06–0.34 s pause
  vs ~25 s) with atomic temp-file rename
- SIGTERM → snapshot trap (drilled end-to-end)
- gnorm in every log line; DDP `gradient_as_bucket_view=True`; fused
  AdamW on the fp32 path

## Run-2 architecture/schedule upgrades (from-scratch lessons pass, 2026-08-06)

**Status: ALL IMPLEMENTED as env knobs and 5090-validated (matrix in
run2_tests.sh).** Smoke-scale results (413M, 250 steps): full bundle
(BIAS_BALANCE=0.001 AUX_COEF=0.0001 ZLOSS_HEAD=1e-4 NO_WD_EMB=1 TIE_EMB=1
LR_SCHED=wsd MTP_W 0.3→0.1 DETERMINISTIC_DATA=1) reached loss 5.88 vs
cosine-baseline 6.62 — effects unattributed individually; WSD's longer
full-LR plateau is likely the biggest single factor. Measured gotchas:
- **TIE_EMB requires init std 0.02** — tying the default N(0,1) embedding
  into the head gives step-0 loss ~380 (measured); the trainer now re-inits
  on tie. Export must still write both weight copies for serving parity.
- **torch.compile is 3-4x SLOWER than eager for this MoE** (dynamic
  `counts.max()` graph-breaks every step) — the MFU lever is a fused
  grouped-GEMM kernel, not compile. Measured 0.34-0.55 vs 0.12 s/step.
- **FP8 training on sm120 — measured reality check**: raw `_scaled_mm` is
  **2.1-3.3x faster than bf16** at Fruit GEMM shapes (consumer Blackwell's
  nerfed bf16 rate widens FP8's relative win), and FP8_LINEAR training is
  loss-parity-proven (6.264 vs 6.265 at step 249, deterministic data). BUT
  end-to-end at Fruit's geometry it's a wash (1.10 vs 1.11 s/step at 5B):
  per-call amax+cast overhead on 1024-dim linears eats the kernel win, and
  the MoE expert GEMMs + chunked CE (the bulk) stay bf16
  (_scaled_grouped_mm is SM100-gated). FP8_LINEAR ships validated but
  default-off; it pays only at bigger hidden dims / token batches, or when
  fp8 grouped-mm reaches sm120. MTP-side linears excluded structurally
  (token count BS*(SEQ-1) violates _scaled_mm's %16 trailing-dim rule).
- **MOE_IMPL=grouped (torch._grouped_mm, torch>=2.12) validated**: loss
  curves bit-equivalent to StackedMoE on deterministic data; **-28% peak
  memory at BS=8** (padded [E,M_max,H] buffers eliminated — also removes
  the record-skew allocator-growth behavior); parity at BS=1. Speed parity
  at 5090 scales; expect gains at production group sizes. Run-2 default.
- DETERMINISTIC_DATA verified: identical loss sequences across two runs
  ("what did step N see" = reseed (20260805, rank, N)).
- ZLOSS_HEAD adds ~60% step overhead at smoke scale (fp32 logsumexp in the
  CE checkpoint region) and ~6 GiB peak — price it in before adopting.
- QK-norm is EXCLUDED on purpose: it adds params GLM-5.2 doesn't have and
  would break the byte-exact mimic.
- **Run-2 validation sidecar (Michel's pattern, confirmed viable):** keep a
  1x RTX 6000 Pro Blackwell instance on Jarvis, PAUSED during training —
  `jl pause` stops billing and keeps data; resume (~1-2 min) at each
  checkpoint push, pull from HF, SIQ-export + serve-gauntlet, pause again.
  Cost = active test-hours + storage only. Keeps the training box training
  (no stop/start), and sm120 runs the real b12x serving stack. For run-1
  the home 5090 already fills this role for free.
- **b12x (ex-SparkInfer) has zero training utility** (surveyed 2026-08-06:
  forward-only, no autograd anywhere) — but on sm120 rentals (8x RTX 6000
  Pro Blackwell, ~same $/h as 4xH200, 768 GB aggregate) it enables
  co-resident serving-parity CI: train -> SIQ-export -> serve every
  checkpoint through the real Trellis runtime on the SAME box. H200 (sm_90)
  cannot run the serving stack at all. Wildcard, unvalidated: b12x
  `comm.pcie` compressed all-reduce (~48% wire reduction) for PCIe DDP.

Evidence sources (see SFT_NOTES.md):

- **WSD (warmup-stable-decay) schedule instead of cosine** — lets you extend
  the token budget, branch decay experiments from one stable checkpoint, and
  pick the anneal data-mix late (MiniCPM; TinyLlama's cosine lock-in regret).
- **OLMo2 stability bundle**: QK-norm, z-loss 1e-4 on final softmax, no
  weight decay on embeddings, filter docs with ≥32x repeated n-grams.
- **Warmup 1–2k steps** (100 is an outlier at this batch size).
- **Aux-loss-free bias balancing** (DSv3): per-expert selection bias ±γ
  instead of aux LBL — removes the balance-vs-quality tension.
- **Tied embeddings**: embed+head = ~316M of 5.04B params at 155k vocab;
  Smol playbook found tying ~free at this scale.
- **Intra-document attention masking** in packed sequences (SmolLM3: also
  what makes later context extension work).
- **Reproducible data order** (Pythia): persist shuffle seed + manifest so
  "what did step N see" is answerable — enables spike forensics and the
  PaLM skip-batches remedy.
- **Schedule best data into the cosine/decay tail** (MiniCPM): the last
  ~15% of steps are disproportionately valuable.
- MiniCPM/MTP note: verify MTP loss share; decayed weight (0.3→0.1) late.

## Original research notes (context for the kit above)

- **Pure-bf16 dead zone**: without fp32 master weights, updates < ~0.2%
  relative are discarded (blunts the low-LR anneal tail). Fix: fp32 master
  weights (+10 GB, affordable at BS=6/H200) or stochastic rounding.
- **Router z-loss** (ST-MoE, coeff ~1e-4) on gate logits: our sigmoid
  router is less explosion-prone than softmax but unbounded logits remain
  a late-run spike risk; watch for spikes past ~step 30k in run 1.
- **`SNAPSHOT_SAVE=1`** (validated on the 5090: 0.12 s pause vs ~25 s):
  pinned-mirror async checkpointing; add temp-file+rename to decouple
  save from a still-running upload at short cadences.
- **SIGTERM trap → final snapshot** for spot-preemption warnings.
- **DDP + activation checkpointing needs `static_graph=True`**
  (find_unused_parameters double-marks recomputed params — measured).
