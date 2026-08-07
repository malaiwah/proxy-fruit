# KQUANT_PLAN.md — qualifying the kquant/QSRT codec via the Fruit proxy

**Handoff document for a fresh Claude session. Written 2026-08-07 on `aiboss`
(RTX 5090 32 GB, sm_120). Assumes zero prior context.** Every fact below is
tagged **MEASURED** (our artifacts, cited), **CLAIMED** (external source,
cited), or **SPECULATION** (labeled). Re-verify pinned SHAs before building —
several upstream repos move daily.

Goal: qualify the **QSRT** quantization codec (repo nickname "kquant") on
**GLM-5.2-SIQ-Fruit** (the 5.04B GLM-5.2 serving proxy we trained and
released) against the existing **SIQ** (Trellis K4K3) baseline, before anyone
spends time or money pointing it at the ~754B GLM-5.2 production model on the
b12x/SparkInfer stack.

---

## 0. Executive summary of the four hypotheses

| # | Michel's hypothesis | Verdict at time of writing |
|---|---|---|
| 1 | Fruit can qualify a kquant codec before touching ~754B GLM-5.2 | **Largely supported** — the standing harness exists and just did exactly this for SIQ; but QSRT's own porting doctrine says record/mode geometry is model-native, so Fruit qualifies the *codec machinery and quality*, not GLM-5.2's exact mode table. Ratio is **~1:150 total / ~1:91 active**, not 1:50 (see §2 — RESOLVED 2026-08-07). |
| 2 | QSRT will be MORE memory-efficient than SIQ K4K3 mixed | **Plausible, modest, and conditional** — arithmetic says ~11% smaller on expert payload (3.0 vs 3.375 bpw), ~7% end-to-end, *before* QSRT metadata and X4T promotions eat it back. Vs SIQ uniform-K3 it is 0%. Serve-time footprint is the other half — unverified; measure at Step C with the §3.3 memory-forensics protocol. |
| 3 | QSRT can serve efficiently, close to SIQ K4K3 | **Cannot be tested today**: the QSRT *runtime does not exist in any public repo or in our pinned images* (verified, §1.4). Kernel numbers are CLAIMED from the author's synthetic TP12 benchmarks on SM120 silicon and look good (W4A8 2.2–3.6× over W4A16), but TP12-only, no end-to-end checkpoint latency, nothing for TP1. |
| 4 | QSRT maintains correctness "where it counts" vs SIQ | **Unknown — and the author's own first artifact FAILED its quality gate** and was scrapped (calibration redesign in progress). Our parity/KLD/MTP/needle harness is precisely the right instrument; measure at Step E. |

Bottom line: encode-side experimentation on Fruit can start now (kquant's
encoder is public); serve-side qualification is **blocked on QSRT runtime
kernels landing publicly** in b12x / the gilded-gnosis vLLM fork. Plan
accordingly (§5 has a two-phase structure that stays useful either way).

---

## 1. What kquant/QSRT is — and what is unknown

### 1.1 Name resolution (Michel's naming wobble, settled)

- **QSRT** = **Quantile-Stratified Rate-shifted Trellis codec** — the
  canonical, mandated name (the repo explicitly bans alternates:
  "Do not introduce TrellisShift, TSH, SQRT-C, `mixed_exl3`…").
  Source: `AGENTS.md` in the repo below.
- **kquant** = the repo that owns QSRT's offline encoder/pipeline:
  **https://github.com/local-inference-lab/kquant**, default branch
  `master`, pinned HEAD **`79461d37a3a863fd2859e5ae14438e184eaf9ca3`**
  (2026-08-04, "qsrt: own encoder backend and remove legacy experiments").
  Same GitHub org as the gilded-gnosis vLLM fork and b12x.
- **KSRT: does not exist.** GitHub repo/code searches return only Kerala
  transit, an MRI tool, and SAP tcodes. It is a mis-recollection of QSRT.
- **Not llama.cpp k-quants.** GGUF Q4_K etc. are 256-weight super-block
  *scalar* quants (llama.cpp PR #1684 / discussion #2094); QSRT is a trellis
  codec in the QTIP/EXL3 lineage. The name collision is coincidental.
- Provenance: authored by "luke" of the RTX6kPRO community. CLAIMED from
  `local-inference-lab/rtx6kpro` daily summaries: 2026-08-04 "New QSRT quant
  codec (mixed-rate trellis for MoE experts) announced by luke, in development
  for K3/GLM5.x"; 2026-08-07 "EXL3-style trellis but FP8-native for SM120
  speedups; GLM-5.2 porting planned". GLM-5.2 is an *explicit intended target*
  of the codec's author — our proxy work is directly on the roadmap.

### 1.2 The format (CLAIMED — all from `docs/qsrt-technical-brief.md` and
`AGENTS.md` at the pinned SHA; no external paper exists; the repo has **no
README**; this brief is the sole spec)

- **Scope of v1 (frozen):** Kimi-K3 (93 layers, 896 routed experts/layer,
  82,432 layer/expert assignments), **TP12 only**, source = official Kimi-K3
  MXFP4 checkpoint. Everything else (TP4/TP16, K5, wider ladders, entropy
  coding) is explicitly deferred.
- **Reconstruction: SQG-E4M3** ("Stratified Quantile Graph") — the 2^16 edges
  of an L16 de Bruijn trellis map bijectively to equal-probability microcells
  of a reference distribution; each microcell's conditional mean is projected
  RNE to **finite E4M3**. Weights therefore decode to native FP8 values —
  this is the "FP8-native for SM120" pitch.
- **Fixed-payload mixed rate:** the intermediate axis is split into
  128-neuron **records** (Kimi: 3,072 → 24 records) of 16×16 coding tiles. A
  function-preserving permutation (P·W1, P·W3, W2·Pᵀ — valid because the gate
  nonlinearity is coordinatewise) makes importance contiguous; modes
  R0/R1/R2 then trade r records of K2 against r records of K4 around a K3
  body so that **every mode averages exactly 3.0 path bits/weight at
  identical payload size** (P24/P33 paired-record containers). Per expert,
  one shared rate decision `r13` for fused w1/w3 and an independent `r2` for
  w2, chosen by dense-H BlockLDLQ re-encode + routed-replay bootstrap gates.
- **X4T high-quality endpoint:** bit-exact preservation of the source MXFP4
  nibble plane + lossless compression of its UE8M0 scale plane (source is
  4.25 bpw); a global allocator promotes the most-damaged experts to X4T
  under an exact byte budget. **X4T is MXFP4-source-specific** — it does not
  apply to a BF16-source model like Fruit (porting guide §7 says design a
  separate endpoint).
- **Encoder:** `kquant/exl3_encoder_backend.py` + `kquant/sqg_e4m3.py` +
  CUDA under `kquant/csrc/` (`sqg_quantize.cu`,
  `qsrt_quantize_tiles_kernel.cuh`, `exl3_compat/`). ExLlamaV3 is an
  *unmodified upstream dependency* (packing/Hadamard/tensor utils only —
  same dependency family as our SIQ encoder's exllamav3==0.0.43 pin).
  No `sm_120`/`__CUDA_ARCH__` gates found in the encoder sources.
- **Runtime paths (CLAIMED, author's own gates):** a **W4A16** correctness
  path (widen exact E4M3 before MMA) and a **W4A8** fast path (Hadamard-domain
  activations → E4M3+UE8M0 → SM120 block-scaled MMA). Synthetic TP12 routed
  mixtures, 16 experts, CUDA-graph replay, on an RTX PRO 6000 Blackwell
  Max-Q: W4A8 = 167–320 µs vs W4A16 600–759 µs (**2.24–3.58× faster**),
  activation-quant cost 0.199–0.222% NMSE, cosine 0.99889–0.99901. X4T
  predecode adds 1.25–8.19 µs to a routed W4A16 MoE call. *No end-to-end
  checkpoint latency exists* (their checklist marks it pending).
- **License: NONE.** The repo has no LICENSE file (GitHub `licenseInfo:
  null`) — default all-rights-reserved. `THIRD_PARTY_NOTICES.md` covers the
  MIT ExLlamaV3-derived code. See risk §7.

### 1.3 Current status of the codec itself (CLAIMED, important)

The first R44/X4T Kimi-K3 artifact **failed the expected quality trajectory
and its generation path was stopped** (brief, "Evidence and current quality
blocker"). Root cause per the author: the layer-global post-SiTU `H2`
covariance was invalid across independently-permuted experts. The replacement
gate starts with a 1M-token source-controlled capture; expert-stratified `H2`
with shrinkage; sealed candidate pools are not grandfathered. Micro-scale
evidence so far: on a 24-expert panel, SQG beat MUL1-E4M3 and FP16 MCG at all
216 matrix/rate comparisons and ~2.1–2.2% lower confirmation SSE — real but
small-corpus, hypothesis-forming (the author says so themselves). **QSRT has
no validated checkpoint anywhere yet.** Fruit qualifying it would be a
genuine contribution, not a rubber stamp.

### 1.4 What is UNKNOWN / verified-absent (MEASURED, 2026-08-07)

- **No QSRT runtime in anything we can run.** Verified by grep:
  - production r28 image
    (`voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28`):
    zero `qsrt|sqg` hits in
    `/opt/venv/.../vllm/model_executor/layers/quantization/` and in the
    installed `sparkinfer` package (CPU-only container grep, no GPU used).
  - public **b12x** master @ `680d8195b80420296d7fed2688b75406be15eb38`
    (full git tree listing): zero `qsrt|sqg|x4t` paths; only the SIQ
    `trellis_linear` kernels exist.
  - The brief references `b12x/benchmarks/benchmark_x4t_w4a16_moe_tp12.py`
    and AGENTS.md points at `/home/luke/projects/vllm` + `/home/luke/projects/b12x`
    — i.e. **the runtime lives in the author's private checkouts**.
- Unknown: serialized artifact schema (slab/sidecar container is described
  in prose; `kquant/pack/qsrt_slab.py`, `qsrt_package.py` are the source of
  truth — read them before writing bytes), vLLM loader contract, whether the
  eventual runtime will accept TP1, minimum expert/record counts, Hessian
  capture tooling reuse outside Kimi (`kquant/kimi_stream.py` is
  model-specific), encode wall-clock at any scale.

---

## 2. The Fruit ↔ GLM-5.2 relation

### 2.1 What Fruit is

`malaiwah/GLM-5.2-SIQ-Fruit` — a **trained 5.04B-param (0.46B active)**
GLM-5.2-architecture serving proxy (`glm_moe_dsa`), Apache-2.0, built by this
program (repo: `github.com/malaiwah/proxy-fruit`, local `~/proxy-fruit`,
working dir `~/fruit-pilot`). It is a *production-shape serving proxy*: a CI
fixture for the serving/quantization stack, with quality-bearing (not random)
weights, so quantization deltas, MTP acceptance, and needle recall are real
signals.

**Parameter ratio (RESOLVED 2026-08-07):** the geometry contradiction flagged below was real — "355B/32B" was GLM-4.5's branding. Derived from the canonical serving config at `/mnt/vault/llm/glm52-franken/src/config.json` (hidden 6144, 78 layers, 256 experts, moe_inter 2048): GLM-5.2 ≈ **754B total / 42B active** (routed experts alone 724.8B). **Fruit is ~1:150 by total params, ~1:91 by active.** Use these ratios in all extrapolations; per-tensor scaling still follows the fidelity manifest rules.

### 2.2 Geometry table (the fidelity manifest, from `~/proxy-fruit/README.md`)

| config key | GLM-5.2 parent | Fruit | rule |
|---|---|---|---|
| `kv_lora_rank` | 512 | **512** | KEPT (kernel parity) |
| `qk_rope/nope/v_head_dim` | 64/192/256 | **64/192/256** | KEPT (KV head_size 576 byte-exact) |
| routed/top-k/shared experts | 256 / 8 / 1 | **256 / 8 / 1** | KEPT (router + dispatch parity) |
| `routed_scaling_factor` | 2.5 | **2.5** | KEPT |
| DSA indexer (heads×dim, topk) | 32×128, 2048 | **32×128, 2048** | KEPT |
| MTP layers | 1 | **1** | KEPT |
| `first_k_dense_replace` | 3 | **3** | KEPT |
| vocab/tokenizer | 154,880 | **154,880** | KEPT |
| `hidden_size` | 6144 | 1024 | ÷6 |
| `num_hidden_layers` | 78 | 13 | ÷6 |
| `num_attention_heads` | 64 | 16 | ÷4 (≥8 for sparse-MLA dispatch) |
| `q_lora_rank` | 2048 | 1024 | scaled |
| `moe_intermediate_size` | 2048 | 512 | ÷4 (tile-constraint aware) |
| `intermediate_size` (dense) | 12288 | 2048 | ÷6 |

(Note: a task note mentioned "512 vs **1536**" for moe_inter — the reviewed
manifest says parent **2048**; treat 1536 as a mis-recollection unless the
fresh session finds otherwise in the parent config.)

### 2.3 What transfers to QSRT at proxy scale — and what doesn't

Transfers (the reasons Fruit is a valid QSRT test article):

- **256-expert MoE with top-8 + 1 shared, GLM router semantics** — expert-
  static per-expert `(r13, r2)` selection, routed-replay scoring, and the
  route-census/H2-stratification machinery all exercise at full expert count.
- **SwiGLU gate is coordinatewise** → QSRT's shared neuron-permutation proof
  (porting guide step 2) should carry; must still be *tested* on real Fruit
  experts (full-precision closure), not assumed.
- **Record divisibility:** Fruit moe_inter 512 = 4×128-neuron records;
  GLM-5.2 2048 = 16 records; Kimi 3072 = 24. All divide by 128, and 512 also
  satisfies the SIQ-side `trellis3_t256_proj` FC1 % 256 == 0 constraint we
  already hit (MEASURED, export_fruit.py zero-pad trick exists if a kernel
  ever demands %256).
- The **entire serving-parity harness** (§3) and the BF16 ground-truth twin.

Does NOT transfer / must be re-derived (be honest about this in any writeup):

- **The mode table.** Kimi's R0/R1/R2 over 24 records is explicitly
  non-portable ("Do not copy Kimi's 24-record mode table"). At Fruit's 4
  records: R0 = 4×K3, R1 = 1×K2+2×K3+1×K4, R2 = 2×K2+2×K4 (no K3 body left)
  — R2 is a boundary case, and pair-container structure is 2 pairs vs
  Kimi's 12. Rate-shift *granularity* at Fruit scale is much coarser than
  GLM-5.2's 16 records, so mode-selection statistics measured on Fruit
  under-sample GLM's decision space. **Fruit qualifies the codec's
  machinery, quality behavior, and serve path — not GLM-5.2's tier ratios
  or allocation frontier.** (Same caveat applied to SIQ: we mirrored parent
  tier ratios 96×K4+160×K3 by fiat, not derivation.)
- **X4T**: Fruit's source is BF16 (our trainer), not MXFP4 — the exact
  endpoint needs a substitute (e.g., keep-BF16 tier or an FP8 near-lossless
  tier) per porting guide §7. GLM-5.2 community serving also starts from
  NVFP4/FP8 sources, so even the GLM port will face this.
- **TP geometry**: QSRT v1 is TP12-only; Fruit serves TP1 on one 5090; GLM
  prod is TP4/DCP4 (MEASURED: our r28 serve scripts note "prod's TP4/DCP4
  stack"). Both need contracts QSRT doesn't have yet.
- **Absolute quality/latency magnitudes** — a ~1:150 proxy's KLD and tok/s do
  not extrapolate numerically; only regressions/orderings transfer (this is
  the program's standing epistemics).

### 2.4 SERVE_CONV — serving-native conventions (critical for any new codec)

MEASURED, `~/proxy-fruit/SMOKE_PLAN.md` ("Run-2: serving-native conventions"):
since Run-2 the trainer trains **directly in vLLM's layouts**
(`SERVE_CONV=1` default): interleaved RoPE and `eh_proj(cat[embed, hidden])`.
Checkpoints carry marker buffers (`serve_conv_v`, `rope_theta_trained`);
`export_fruit.py` auto-detects them and **converts nothing** for Run-2+
checkpoints. Phase-1 checkpoints (the released ones) predate this: export
applies the RoPE permutation + eh_proj half-swap. Equivalence proven on CPU
to 1.3e-06/2.4e-06 fp32. **Any QSRT exporter must copy this exact behavior**
— honor the markers, reuse `export_fruit.py`'s non-expert handling verbatim.
History says this class of bug is the #1 killer: the pre-fix RoPE export
measured 69.0% top-1 / KL 0.809 (vs 95.2%/0.020 fixed), and the eh_proj swap
took MTP acceptance from 0.4% to 98.6% (REVIEW.md findings 1–3).

---

## 3. Why proxy-first is cheaper + the standing qualification harness

### 3.1 Cost asymmetry (MEASURED)

- **Encode:** Fruit 5.04B SIQ export = **~49 s per MoE layer, ~12 min
  total** on one RTX 5090, 2.89 GiB out (SMOKE_PLAN home tier). GLM-5.2
  355B encodes are multi-hour multi-GPU affairs (kquant's own Kimi pipeline
  budgets 12-GPU runs; brandonmusic's TR3 encodes likewise).
- **VRAM:** the full Fruit serve gauntlet fits one 32 GB 5090 at
  `gpu_memory_utilization=0.75`, 2k ctx. GLM-5.2 needs AIBeast
  (4×RTX PRO 6000) and takes production down while testing — the entire
  reason this proxy program exists.
- **Money:** whole Phase-1 training program ≈ $228 spot; a failed GLM-scale
  quantization experiment costs more than that in prod downtime alone.

### 3.2 The standing harness and SIQ baseline numbers to beat/match

All MEASURED on this box; scripts live in `~/proxy-fruit/` (tracked) and run
inside the gilded-gnosis containers. Baseline = the released SIQ exports.

| Gauntlet item | Script | SIQ K4K3 baseline (annealed base unless noted) |
|---|---|---|
| Artifact size | `export_fruit.py` | **2.89 GiB** (mixed 96×K4+160×K3/layer, MTP layer uniform K3, non-experts BF16) |
| r25 battery + serve | `fruit_serve_test.py <ckpt> fp8_ds_mla` | PASS (small-prompt battery 1/2/5/8/9, words/lively probes) |
| r28 prod-parity serve | `fruit_serve_r28.py` (nvfp4_ds_mla + `attention_backend="B12X_MLA_SPARSE"`) | PASS |
| Round-trip parity vs training graph | `parity_test.py` | **top-1 92.9%, top-10 overlap 88.8%, KL 0.045** (final ckpt; 95.2%/0.020 on mid-run) |
| MTP k=1 acceptance | `fruit_serve_mtp.py` | **97.7% / 94.1% / 79.0%** (final / annealed / instruct) |
| Apache-2.0 needle (held-out) | `fruit_needle.py` | **Apache 0.000, MIT control 0.974** |
| Decode (CC1, eager, 5090) | serve scripts | **~62 tok/s** with MTP k=1 (37 without) |
| Long-context | `fruit_serve_long.py` | PASS |
| CPU ground truth | `-bf16` twin via transformers | **~32 tok/s** on i7-14700K; the KLD/parity reference that needs no GPU |

Supporting assets: BF16 twin `malaiwah/GLM-5.2-SIQ-Fruit-bf16` (same
annealed weights, plain transformers — use it as the reference distribution
for any QSRT-vs-SIQ KLD); trainer checkpoints
`/mnt/vault/llm/fruit-pilot/final/fruit_v1_{annealed,final,instruct}.pt`;
existing SIQ exports
`/mnt/vault/llm/fruit-pilot/output/GLM-5.2-SIQ-Fruit-{annealed,final,instruct,annealed-bf16}`;
HF: `malaiwah/GLM-5.2-SIQ-Fruit{,-Instruct,-bf16,-pilot}`,
`malaiwah/fruit-phase1-ckpt` (all verified live 2026-08-07).

Tier-variant exports already exist as harness knobs: `FRUIT_TIERS=mixed|k3|k4`
(the k3-uniform export is the fair 3.0-bpw size comparator for QSRT).

### 3.3 Memory forensics: what one 5090 unlocks that 4×RTX 6000 Pro can't

Michel's thesis, endorsed by the tool inventory below: **low-level memory
capture/analysis is far easier at proxy scale.** On the 4×96 GB production
box, every trace is TP4-interleaved (all-reduce buffers, per-rank pools,
NCCL scratch), snapshots are tens of GB, and profiling contends with
production. On the 5090 at **TP=1 there is no collective machinery at all**
— a single-rank allocator trace is clean, fits in RAM, and attributes every
byte. Measure kquant-vs-SIQ per-pool/per-tensor-family deltas *precisely* on
Fruit, then project to GLM-5.2 via the manifest scale rules (projections
labeled as such).

**Tool inventory — verified on this box/stack 2026-08-07 (MEASURED):**

| Tool | State here | Notes |
|---|---|---|
| PyTorch allocator snapshots (`torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` → view at pytorch.org/memory_viz) | **Works** — torch 2.12.0+cu132 in r28 has both APIs | No special permissions; per-allocation stack traces = per-tensor-family attribution for free. The primary instrument. |
| vLLM memory profiler hooks | **Present in the fork** — `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` in `envs.py:334` (default `1`), plus the engine's standard load/profile/KV-pool report lines | The same env community configs set (CLAUDE.md §NCCL item 9). Engine logs already break out weights / activation-profile / KV-pool GiB. |
| `pynvml` / `nvidia-smi` deltas | **Works** — pynvml importable in r28 | Whole-process truth (CUDA context + non-torch pools that snapshots miss). Poll `nvmlDeviceGetMemoryInfo` at capture points. |
| Nsight Systems (`nsys`) | **Ships in the r28 image** (`/usr/local/bin/nsys`) | CUDA/CUPTI memop + API timeline works unprivileged (verify once); **CPU IP sampling blocked** — host `perf_event_paranoid=4`. Fine for memory work. |
| Nsight Compute (`ncu` 2026.1.1) | **Ships in the r28 image** (`/opt/nvidia/nsight-compute`) but **HW counters blocked**: driver `RmProfilingAdminOnly: 1` + rootless podman (no CAP_SYS_ADMIN) ⇒ `ERR_NVGPUCTRPERM` | Unblock (needs root + Michel's sign-off): `NVreg_RestrictProfilingToAdminUsers=0` via modprobe.d + driver reload. Not needed for the memory protocol — only for kernel-level counter work. |
| CUPTI-based allocation tracing | Available via the nsys/ncu bundles in-image | No host CUDA toolkit (per CLAUDE.md) — all profiling happens inside containers, which is fine since the serve path lives there anyway. |

**Measurement protocol (add to the Step C/D comparison; scripted, committed):**

1. Arms: **bf16 twin / SIQ-K4K3 / SIQ-K3 / QSRT**, byte-identical `LLM()`
   config (util 0.75, 2k ctx, `max_num_seqs 4`, eager) — the existing
   `fruit_serve_test.py` config, instrumented.
2. Enable `_record_memory_history(max_entries=200_000)` *before* engine
   construction; `_dump_snapshot()` at three points: **(a) post-weights-load,
   (b) post-profile/warmup** (KV pool carved), **(c) steady-state decode**
   (mid-battery). Record pynvml totals at each point and keep the engine's
   own memory-report log lines.
3. Diff pools across arms: weights pool (a−context), KV pool (engine-reported
   — should be *identical* across arms at fixed config; flag if not),
   activation transient (peak−steady), allocator fragmentation
   (reserved−allocated).
4. Attribute weight bytes two ways and cross-check: snapshot stack-trace
   families (loader frames) AND a scripted state-dict walk summing bytes by
   name family (routed experts / attention-MLA / dense FFN / embeddings /
   MTP). The walk is exact; the snapshot catches what the walk can't see
   (workspace buffers, dequant scratch, cudagraph pools if ever on).
5. Extrapolate per family to GLM-5.2 using the manifest rules (§2.2): routed
   experts scale ×(6144/1024)·(2048/512) = ×24 per matrix at equal expert
   count (256, KEPT) × (MoE-layer count ratio); dense FFN ×(6144·12288)/(1024·2048)
   = ×36; MLA partially KEPT (kv_lora_rank 512 and head dims byte-exact —
   these pools scale only by heads ×4 / hidden ×6 factors per the manifest);
   KV pool scales by layer count and head geometry, NOT by codec.
   **Label every projected number SPECULATION-extrapolated.**
6. **Consistency warning before quoting projections (MEASURED arithmetic):**
   the manifest parent geometry (75+ MoE layers × 256 experts × 3 ×
   6144×2048) implies ~725B routed-expert params alone — inconsistent with
   the model card's "355B total" (and consistent with FEASIBILITY.md's
   "~750B", §2.1). Recompute all scale factors from the *actual* parent
   `config.json` (on vault under the brandonmusic TR3 snapshots) before
   publishing any GLM-projected table. (Michel's "moe_inter 1536" recollection
   vs the manifest's 2048 is part of the same unresolved knot.)

Why this matters for H2 specifically: artifact-on-disk is only half the
memory story — serve-time footprint (dequant workspaces, decode tables, X4T
scratch, alignment slop) is where codecs actually differ on a VRAM-starved
card, and it is exactly what the snapshot diff isolates per arm. QSRT's
CLAIMED design keeps decode tables process-global (58–106 KiB total); SIQ's
runtime keeps per-shape JIT'd kernels + trellis codebooks. Nobody has
numbers for either at Fruit scale. This protocol produces them.

---

## 4. Claim-by-claim assessment

### H1 — "Fruit can qualify kquant before touching GLM-5.2"

**Supported with a scope note.** Evidence FOR: (a) the harness just executed
this exact play for SIQ end-to-end, catching two real serving bugs (RoPE
layout, eh_proj halves) that would have been catastrophic at 355B — MEASURED;
(b) the Fruit pilot reproduced two live b12x kernel issues (#121 mHC NaN,
#107 mixed-tier prefill) with maintainer-visible comments — MEASURED,
proving proxy-scale repros transfer upstream; (c) kquant's own porting guide
treats "another gated MoE (GLM 5.2 named)" as a supported port with a
step-by-step contract list — CLAIMED. Evidence AGAINST / limits: mode-table
geometry does not transfer (§2.3); QSRT quality is calibration-dominated and
Fruit's calibration corpus is our own (results qualify *our* Fruit-QSRT
recipe, informing but not proving the GLM recipe). Net: proxy-first is the
right order; frame conclusions as codec-mechanics + relative-quality
qualification.

### H2 — "QSRT more memory-efficient than SIQ K4K3"

**Arithmetic (SPECULATION until measured, but tightly bounded):** Fruit
routed-expert params = 11 MoE layers (10 + MTP) × 256 × 3 × (1024×512) =
4.43B. SIQ mixed = (96×4 + 160×3)/256 = **3.375 bpw** → 1.74 GiB experts +
~1.14 GiB BF16 non-experts ≈ 2.88 GiB — matches the 2.89 GiB artifact, which
validates this model of the size. QSRT all-lossy = **3.0 bpw** → 1.55 GiB
experts ≈ 2.69 GiB total = **~6.7% smaller end-to-end (~11% on experts)**.
Offsets: QSRT metadata/slab headers (small, CLAIMED design goal), X4T-class
promotions (each promoted expert costs ≳4.25-bpw-equivalent — with promotions
the artifact can exceed SIQ mixed), and note SIQ already offers uniform-K3 at
the same 3.0 bpw (`FRUIT_TIERS=k3`). **QSRT's real pitch is not "smaller" —
it is mixed-rate quality at a *fixed* 3.0-bpw payload, i.e. K4-like quality
where it matters without K4's bytes.** Verdict: "more efficient than K4K3
mixed" = true by ~7% iff few/no promotions; "more efficient at equal
quality" is the interesting claim and is exactly what Step E measures.

### H3 — "QSRT can serve efficiently, close to SIQ K4K3"

**Unverifiable today on our stack** (runtime absent everywhere public —
MEASURED, §1.4). CLAIMED kernel-level evidence is encouraging and on the
right silicon family (SM120): W4A8 2.24–3.58× faster than W4A16 synthetic
routed MoE; X4T predecode ~2–8 µs; CUDA-graph replay passes; SQG decode uses
58–106 KiB process-global tables (well under the smem ceilings that bit us
on SIQ — MEASURED for comparison: SIQ trellis smem = 93184 + 8192·(k_lo+k_hi−7)
bytes, 5090 cap 101376 → (3,4) is the consumer ceiling). Unknowns: TP1
support, small-moe_inter (512) tile occupancy, eager-vs-graph behavior on
the 5090's known sm_120 capture bugs, and whether E4M3 weight decode + our
bf16 activations (W4A16 path) is the mode we'd actually get — the fast W4A8
path quantizes activations and "must not be used to judge the codec's weight
distortion" (author's own rule). Expectation to calibrate against: SIQ Fruit
decodes ~62 tok/s CC1 (MTP k=1) on the 5090.

### H4 — "QSRT maintains correctness where it counts"

**Open.** FOR: the codec's engineering culture is bit-exactness-first
(exhaustive 2^16 label validation, bit-identical lockstep asserts, malformed-
input gates — same philosophy as our SIQ encoder's oracle gate). AGAINST:
the only full-model attempt so far *failed its quality trajectory* (§1.3),
mixed-rate selection showed winner's-curse vulnerability, and small-corpus
gains (~2%) are within the range that a bad Hessian wiped out once already.
Our discriminating instruments, in order of sensitivity to "where it
counts": parity top-1/KL vs the bf16 twin (structural bugs read as ~0
agreement, quality loss as a few points), MTP acceptance (95%+ regime is
exquisitely sensitive to drafter/target distribution mismatch), needle
recall (memorized-content fidelity), chat battery on the Instruct variant.
Pass bar: within noise of SIQ K4K3 on parity/KL and acceptance at equal or
smaller artifact size; stretch goal: beats SIQ K4K3 quality at 3.0 bpw
(that's the fixed-payload pitch vindicated).

---

## 5. Concrete work plan for the fresh session

Environment ground rules (from `~/CLAUDE.md`, non-negotiable): only one GPU —
**check `nvidia-smi` and `podman ps` before ANY serve/encode run** (the card
is contested by other agents); root disk ~91% full (prune before pulls);
rootless podman 4.9.3 (`rm -f --ignore` gotcha); r25/r28 image tags above;
weights live on vault NFS.

### Step A — Pin and reconnoiter (no GPU, ~1 h)

1. Clone at pinned SHAs into `~/kquant-work/`:
   ```bash
   git clone https://github.com/local-inference-lab/kquant ~/kquant-work/kquant
   git -C ~/kquant-work/kquant checkout 79461d37a3a863fd2859e5ae14438e184eaf9ca3
   # reuse existing checkouts if present: ~/upstream-work/{vllm,vllm-baseline}
   ```
2. Read in this order: `AGENTS.md` (porting section = your contract),
   `docs/qsrt-technical-brief.md`, `docs/qsrt-calibration.md`,
   `kquant/pack/qsrt_slab.py` + `qsrt_package.py` (real artifact schema),
   `kquant/exl3_encoder_backend.py` (encoder entry),
   `kquant/sqg_e4m3.py` (label law), `scripts/pack_qsrt_candidates_tp12.py`.
3. Re-check for runtime landings since 2026-08-07 (this gates Phase 2):
   ```bash
   gh api repos/local-inference-lab/b12x/commits --jq '.[].commit.message' | grep -iE 'qsrt|sqg|x4t'
   gh search code --owner local-inference-lab 'qsrt' --limit 20
   ```
4. `uv sync --dev` in the kquant checkout; run `.venv/bin/pytest -q`
   (CPU suite) to establish a green baseline before touching anything.
5. **Open the licensing + roadmap conversation**: file a kquant issue (as
   malaiwah, who already has upstream credibility from b12x #121/#107)
   asking (a) for a LICENSE file, (b) whether TP1/small-geometry runtime is
   on the roadmap, (c) offering Fruit as the GLM-5.2 porting testbed the
   Discord summaries say luke plans. This may short-circuit weeks of work.

### Step B — Fruit adapter for the kquant encoder (no GPU for design; GPU
for closure tests when free)

Follow porting-guide steps 1–4 scaled to Fruit:
1. Freeze source identity: `fruit_v1_annealed.pt` (SERVE_CONV markers per
   §2.4 — Phase-1 ckpt, so the export-side RoPE/eh_proj transforms apply),
   tokenizer, config, tensor inventory test.
2. Prove permutation closure on real Fruit experts (P·gate, P·up, down·Pᵀ,
   SwiGLU) in fp32 on CPU — exact match required.
3. Derive the Fruit-native payload: 4 records × 128 over moe_inter 512,
   16×16 tiles; mode table R0 = (0,4,0), R1 = (1,2,1), R2 = (2,0,2) in
   (K2,K3,K4) record counts; document that R2 is a boundary mode and GLM-5.2
   at 16 records will differ (§2.3).
4. Calibration: reuse the Fruit corpus machinery — route census + expert-
   stratified H2 per the post-mortem doctrine (§1.3). The license/TinyStories
   corpus shards are on HF (`malaiwah/fruit-phase1-shards`); document-disjoint
   splits already exist from Phase-1 val.
5. Write `export_fruit_qsrt.py` **in `~/proxy-fruit`** modeled on
   `export_fruit.py`: identical non-expert/BF16 handling, identical
   SERVE_CONV/marker handling, experts through the kquant encoder backend
   instead of `encode_tr3_v31.py`. High-quality endpoint for v0: keep-BF16
   tier (allocator budget 0 = all-lossy is the cleanest first artifact).
   Target: artifact + manifest + size ledger. **Expected ~2.69 GiB if
   all-lossy (H2 predicts ~7% under SIQ mixed's 2.89) — record the actual.**

### Step C — Correctness qualification WITHOUT the runtime (CPU/reference
path; this is Phase 1 and is valuable even if kernels never land)

1. Implement/reuse kquant's reference decode (`kquant/exl3_reference.py`,
   `correctness.py`) to reconstruct the QSRT-encoded experts to BF16.
2. Round-trip a reconstructed-BF16 checkpoint through the **existing** serve
   path (it's just a BF16 GLM-5.2-shape model — vLLM loads it unquantized,
   same as the `-bf16` twin) and run the full gauntlet. This measures the
   codec's *weight distortion* in complete isolation from missing kernels —
   exactly the A16 "correctness fallback" philosophy the author prescribes.
3. Compare table (commit it): QSRT-3.0 vs SIQ-K4K3-mixed vs SIQ-K3-uniform,
   rows = artifact GiB, parity top-1/KL vs bf16 twin, MTP acceptance,
   needle, chat 4-prompt battery. SIQ-K3-uniform is the *fair* equal-bpw
   comparator; SIQ-mixed is the *production* comparator. This settles H2 and
   H4 at the weight level.
4. Run the §3.3 memory-forensics protocol on the arms that serve today
   (bf16 / SIQ-mixed / SIQ-K3) so the harness and baselines are captured
   before the QSRT arm exists; add the QSRT arm in Step D. GPU work — wait
   for a free card (`nvidia-smi` + `podman ps` first).

### Step D — Serve-path qualification (Phase 2, **blocked on public QSRT
runtime**; unblock via Step A.3/A.5)

When runtime lands in b12x/gilded-gnosis (or luke shares a branch):
1. Rebuild/pull the serving image; pin new SHAs in this doc's ledger.
2. Loader: mirror how our SIQ loader consumes `tier_bitmap`/format codes —
   study the r25-extract sources `~/qwen36-27b-siq/r25-extract/exl3.py`
   (registration, `Exl3MoEMethod`) for the shape of a vLLM quant-method
   integration on this fork.
3. **Verification protocol, in this exact order** (engine-killers first —
   standing rule from `~/CLAUDE.md` and SMOKE_PLAN):
   a. small-prompt battery 1/2/5/8/9-token `/v1/completions` (a crash kills
      the engine; sm_120 graph-capture Xid 31 class),
   b. one ~2k prefill + coherence probes (`fruit_serve_test.py` pattern),
   c. r25 fp8_ds_mla AND r28 nvfp4_ds_mla + B12X_MLA_SPARSE arms,
   d. parity_test.py vs the bf16 twin,
   e. MTP k=1 acceptance (`fruit_serve_mtp.py`),
   f. needle + chat battery,
   g. decode tok/s CC1 (eager) vs SIQ's ~62; note W4A16 vs W4A8 mode
      explicitly — never quote W4A8 numbers as codec quality (author's rule).
4. Known landmines (all MEASURED on this box, don't rediscover): mount
   `~/glm52-franken/tools` at `/tools` for the SIQ comparator arm;
   `--no-deps` for anything pip'd into the shared volume; output dir must
   not pre-exist; podman heredocs need `-i`; retry one-off 5090 flakes once;
   engine-teardown segfaults after `*-OK` lines are harmless.

### Step E — Verdict + only then GLM-5.2

Write the comparison table + a one-page verdict against H1–H4 into
`~/proxy-fruit` (commit). GLM-5.2 go/no-go criteria: QSRT ≥ SIQ-K3-uniform
quality at equal bytes AND within ~5% of SIQ-mixed parity/acceptance AND a
runtime that boots the battery clean on sm_120. The GLM-5.2 encode itself
happens on AIBeast-class hardware with luke's TP4 port — out of scope here;
our deliverable is the qualified codec + the Fruit-QSRT recipe + filed
issues.

---

## 6. Transparency / verifiability requirements (program standards)

- **Every measurement is scripted and committed** to `~/proxy-fruit` (the
  SIQ gauntlet scripts are the template — positional args, `FRUIT-*-OK`
  sentinel lines, env-tunable). No numbers in prose without a script path.
- Update the ledgers: SMOKE_PLAN.md cost/time table per run; REVIEW.md
  pattern for any external findings; this file's §0 verdicts as evidence
  arrives (flip "unverified" to MEASURED with numbers).
- **HF publication plan:** `malaiwah/GLM-5.2-QSRT-Fruit` (mirroring the SIQ
  naming), model card with the full comparison table, tools/ mirror of the
  repro scripts (as done for the pilot), cross-links to the SIQ and bf16
  siblings. **Blocked on kquant licensing** (§7 first bullet) — do not
  publish encoder-derived artifacts until the license question is answered
  in writing; local experiments are fine meanwhile.
- Memory notes: record the QSRT verdicts in the auto-memory project file so
  later sessions inherit them.

## 7. Open questions / risks

1. **License (blocking for publication):** kquant has NO license file →
   default all-rights-reserved. Using it locally for research is low-risk;
   *redistributing* artifacts or vendored code is not cleared. Ask luke
   (Step A.5). The ExLlamaV3-derived parts are MIT.
2. **Runtime availability** (blocking for H3): QSRT kernels exist only in
   the author's private checkouts; public b12x master has none (verified
   2026-08-07). Timeline unknown. Mitigation: Phase-1 CPU/reference-decode
   qualification (Step C) is runtime-independent.
3. **sm_120 kernel fit:** CLAIMED numbers are RTX PRO 6000 (same SM120
   family, more SMs/smem-per-SM headroom than the 5090). SQG's 58–106 KiB
   tables are process-global (not per-CTA smem) so the SIQ (3,4) smem
   ceiling problem *shouldn't* recur — SPECULATION until run.
4. **TP1 + small-geometry support:** everything QSRT is TP12-shaped today;
   Fruit needs TP1, moe_inter 512 (4 records — pairing/rotation logic may
   assume ≥ some record count), and 256 experts vs Kimi's 896. Grouped/MoE
   layout mismatches are exactly the class of thing the pilot caught before
   (#107, #121) — expect to file issues, that IS the product.
5. **Mixed-tier interaction with MTP + spec decode:** SIQ needed the eh_proj
   discovery; QSRT's MTP-layer treatment is undefined (Kimi brief never
   mentions the MTP layer). Fruit's MTP layer is MoE — decide its tier
   handling explicitly (SIQ used uniform K3 there).
6. **Calibration sensitivity:** the codec's one full-model attempt died on
   calibration geometry (§1.3). Our Fruit corpus is small; a null result on
   H4 could be *our* calibration rather than the codec. Mitigate with the
   author's doctrine: expert-stratified H2, shrinkage, identity-H control
   arm (it's the scientific control, "not a silent fallback").
7. **Moving target:** kquant is 3 days old at pin time and pre-checkpoint;
   the brief itself changed course once (R44 → recalibration) mid-history.
   Re-pin and re-read AGENTS.md before every work burst.

## 8. Pinned reference ledger (2026-08-07)

| Repo / artifact | Ref | SHA / tag |
|---|---|---|
| `local-inference-lab/kquant` | `master` | `79461d37a3a863fd2859e5ae14438e184eaf9ca3` |
| `local-inference-lab/vllm` | `dev/gilded-gnosis` | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| — open PR #249 (heterogeneous per-layer expert widths, ours) | head `gg-heterogeneous-expert-widths` | `e15f0fedcd7baa8df2764df374aff66bb43e2558` |
| `local-inference-lab/b12x` (ex-SparkInfer; pip pkg still `sparkinfer` 1.x) | `master` | `680d8195b80420296d7fed2688b75406be15eb38` |
| `voipmonitor/flashinfer` | `codex/sm120-dspark-stack-20260711` | `801d57a08958c13d375ddbb6be3be4808f48a708` (matches `fi801d57a` in image tags) |
| — open PR #1 (sm120 sparse-MLA supported-matrix fail-fast, ours) | head | `517724888c2ddb6c889d5d5fd49efac1fab2145d` |
| serving image r25 | `voipmonitor/vllm` | `gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25` |
| serving image r28 | `voipmonitor/vllm` | `gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28` |
| `malaiwah/proxy-fruit` (this program) | `main` | `4ea7026` at writing |
| SIQ encoder toolchain | `~/glm52-franken/tools` (`encode_tr3_v31.py`, exllamav3==0.0.43 pin) | local |
| Fruit checkpoints | `/mnt/vault/llm/fruit-pilot/final/fruit_v1_{annealed,final,instruct}.pt` | vault |
| SIQ exports | `/mnt/vault/llm/fruit-pilot/output/GLM-5.2-SIQ-Fruit-*` | vault |
| HF (all verified live) | `malaiwah/GLM-5.2-SIQ-Fruit{,-Instruct,-bf16,-pilot}`, `malaiwah/fruit-phase1-ckpt`, `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` (+TR3v4), `willfalco/GLM-5.2-EXL3-TR3-3.36bpw` | — |

Prior art for framing (one-liners, cited): **QTIP** (Cornell RelaxML,
arXiv:2406.11235) — trellis-coded quantization + incoherence processing,
SOTA 2-bit PTQ on dense models; **EXL3/ExLlamaV3** (turboderp-org/exllamav3,
MIT) — QTIP-derived trellis quant, the encoder substrate for both SIQ and
QSRT; **llama.cpp k-quants** (ggml-org/llama.cpp discussion #2094) —
super-block scalar quants, unrelated to "kquant" despite the name;
**proxy-qualification prior art** — vLLM RFC #28135 (spec-decode regressions
pass correctness tests on random-weight fixtures: the gap Fruit closes),
yujiepan tiny-random zoo, inference-optimization/GLM-5.2-0.8B (each half of
the idea; see `~/proxy-fruit/README.md` for the full genealogy).
