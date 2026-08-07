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
| 3 | QSRT can serve efficiently, close to SIQ K4K3 | **Cannot be tested today, but the gap is closing fast**: the vLLM loader/integration half went public 2026-08-06 (branch `k3-pp2tp6`); the sparkinfer kernel half (SQG codebooks, trellis-W4A8, X4T predecode) is still private (verified, §1.4). Kernel numbers are CLAIMED from the author's synthetic TP12 benchmarks on SM120 silicon and look good (W4A8 2.2–3.6× over W4A16), but TP12-only, no end-to-end checkpoint latency, nothing for TP1. |
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

### 1.4 Runtime availability: SPLIT (MEASURED, deep-dig 2026-08-07)

Exhaustive public-surface sweep (all 7 kquant branches full-history, all 63
b12x branches, all 215 branches of the org vLLM fork, 41 b12x forks + 13
vLLM forks by branch name, GitHub code/commit search, PyPI, HF):

- **The vLLM integration half IS public — since 2026-08-06.**
  `local-inference-lab/vllm` branch **`k3-pp2tp6`** @ `34d215334`
  ("serve: add Kimi-K3 QSRT TP12 runtime", Luke Alonso): quantization
  readers `kquant_kimi_k3_qsrt_tp12.py` (793 ln), `kquant_x4t.py`
  (1042 ln, the X4T scale-plane codec), `kquant_mixed_exl3_tp12.py`, an
  extended `nvfp4_nf3_hybrid.py` consumer (source_format
  `exl3_trellis_sqg_cheb_k2_q8h4_w2_e4m3`, `trellis_*_pair_modes` kwargs),
  `fused_moe/kquant_capture.py`, tests, and a serve script
  (`Kimi-K3-QSRT-CHEB-Q8H4-ROUTED-X4T-3p11-KLD-v1-serve` — artifact itself
  not published). All pure Python.
- **The sparkinfer/b12x kernel half is NOT public anywhere.** The branch
  imports modules that exist in no public ref: `…kernels.trellis_w4a8`
  (`run_trellis_w4a8_moe`), `_lib.quant.x4t_scales`
  (`make_x4t_scale_batch`), `moe.calibration`,
  `prepare_w4a16_x4t_tp12_weights`; public b12x supports only the
  `exl3_trellis_mcg` codebook (zero `sqg*`/`pair_mode*` hits in any
  branch). Nuance: the *generic* W4A16 trellis MoE machinery
  (`w4a16_fused_moe_hybrid_launch`, CuTeDSL kernels) IS public in b12x
  master — what's missing is the QSRT layer on top (SQG/SQG-Cheb
  codebooks, pair-mode mixed trellis, trellis-W4A8, X4T predecode).
  **Watch trigger:** a b12x push containing `sqg`/`pair_mode` is the
  unblock signal for Step D.
- Our pinned r25/r28 images predate all of this: zero `qsrt|sqg` hits in
  their vLLM quantization layers and `sparkinfer` packages (CPU-only
  container grep).
- People: luke = **`lukealonso`** (Luke Alonso; also owns the PyPI
  `sparkinfer` 0.0.1 name-reservation). Second kquant committer: Martin
  Vit = **`voipmonitor`** (whose vllm/b12x/exllamav3/InstantTensor forks
  we already build from). kquant has **2 PRs, 0 forks, 0 issues**:
  **PR #1 = a GLM recipe, opened by voipmonitor** (read it FIRST — the
  GLM port conversation has already started without us), PR #2 = QSRT
  profile-5 pipeline (lukealonso; + branch `agent/qsrt-profile5-pipeline`,
  whose head marks X4T TP12 shards for `voipmonitor/InstantTensor`).
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
Historical notes report the pre-fix RoPE export at 69.0% top-1 / top-K
drift 0.809 versus 95.2%/0.020 after the fix, and MTP acceptance changing
from 0.4% to 98.6% after the `eh_proj` swap. Those raw logs are no longer
retained; treat them as regression history, not qualifying KL/evidence.

---

## 3. Why proxy-first is cheaper + the standing qualification harness

### 3.1 Cost asymmetry (MEASURED)

- **Encode:** Fruit 5.04B SIQ export = **~49 s per MoE layer, ~12 min
  total** on one RTX 5090, 2.89 GiB out (SMOKE_PLAN home tier). GLM-5.2
  ~754B encodes are multi-hour multi-GPU affairs (kquant's own Kimi pipeline
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
| Exact trainer→SIQ smoke | `fruit_kld.py` | **deterministic mean full-vocab forward KL 0.001321, max 0.006554, top-1 6/6, top-10 98.3%** (annealed, six fixed positions; structural only) |
| MTP k=1 acceptance | `fruit_serve_mtp.py` | **97.7% / 94.1% / 79.0%** (final / annealed / instruct) |
| Apache-2.0 needle (held-out) | `fruit_needle.py` | **Apache 0.000, MIT control 0.974** |
| Decode (CC1, eager, 5090) | serve scripts | annealed r25/fp8 **53.6 tok/s** no MTP; **60.9 tok/s** MTP k=1; r28/nvfp4 **36.4 tok/s** no MTP |
| Long-context | `fruit_serve_long.py` | PASS |
| CPU BF16 smoke | corrected-theta in-memory config via Transformers | **33.12 tok/s**, 16.01 GiB loaded / 16.76 GiB peak RSS; published config correction pending |

Supporting assets: the published BF16 twin
`malaiwah/GLM-5.2-SIQ-Fruit-bf16` is **not** a reference until issue #2
corrects its 8M RoPE config. Use the pinned trainer checkpoint for KLD.
Trainer checkpoints:
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
6. **Consistency check — RESOLVED 2026-08-07:** the contradiction was
   real and the manifest arithmetic won. Derived from the canonical
   serving config (`/mnt/vault/llm/glm52-franken/src/config.json`):
   GLM-5.2 ≈ **754B total / 42B active** (routed experts alone 724.8B;
   hidden 6144, moe_inter 2048 — "355B/32B" and "1536" were GLM-4.5
   values). Cards and README corrected; use ~1:150 total / ~1:91 active
   in every projection.

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
layout, eh_proj halves) that would have been catastrophic at ~754B — MEASURED;
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
3. Re-check for the kernel-half landing (this gates Phase 2). The
   loader half is already public — study it now:
   ```bash
   # the QSRT vLLM integration (public since 2026-08-06):
   git clone -b k3-pp2tp6 https://github.com/local-inference-lab/vllm ~/kquant-work/vllm-qsrt  # pin 34d215334
   # readers: vllm/model_executor/layers/quantization/kquant_kimi_k3_qsrt_tp12.py, kquant_x4t.py,
   #          kquant_mixed_exl3_tp12.py, nvfp4_nf3_hybrid.py (consumer), fused_moe/kquant_capture.py
   # kernel-half watch trigger (sqg/pair_mode appearing in b12x = Step D unblocked):
   gh api repos/local-inference-lab/b12x/commits --jq '.[].commit.message' | grep -iE 'qsrt|sqg|x4t|pair_mode'
   ```
   Also read **kquant PR #1 (a GLM recipe, opened by voipmonitor)** and
   PR #2 (QSRT profile-5 pipeline) — the GLM-5.2 port discussion has
   already started; coordinate rather than duplicate. Extracted copies of
   the k3-pp2tp6 readers + serve scripts from this session's dig are in
   the session scratchpad (`kquant_x4t.py`, `kquant_kimi_k3_qsrt_tp12.py`,
   `nvfp4_nf3_hybrid.py`, `kquant_capture.py`) — re-clone for anything
   load-bearing.
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
1. Freeze source identity: `fruit_v1_annealed.pt`
   (`sha256:98ac7cb4f7799194424782b505d622069fecf4dbca5f5acb2658f2a66c3631f6`),
   tokenizer, config, and tensor inventory. This is a Phase-1 legacy
   checkpoint: it has neither `serve_conv_v` nor `rope_theta_trained`, so
   export must receive `FRUIT_ROPE_THETA=500000` and apply both the RoPE and
   MTP `eh_proj` layout conversions.
2. Prove permutation closure on real Fruit experts (P·gate, P·up, down·Pᵀ,
   SwiGLU) in fp32 on CPU. The transform is algebraically exact; accept tight
   numerical closure rather than bit equality because permuting `down` columns
   changes FP32 reduction order (observed maximum absolute error
   `4.291534423828125e-06` across layers 3/12/MTP and experts 0/255).
3. Parameterize the activation before reusing candidate scoring or validation.
   Fruit trains and serves SiLU, while `qsrt_candidates.py` and
   `qsrt_validation.py` currently hard-code Kimi's SiTU. A synthetic probe
   through a real Fruit expert measured 10.6% relative-L2 output difference;
   the two activations are not interchangeable.
4. Derive the Fruit-native payload: 4 records × 128 over moe_inter 512,
   16×16 tiles; mode table R0 = (0,4,0), R1 = (1,2,1), R2 = (2,0,2) in
   (K2,K3,K4) record counts; document that R2 is a boundary mode and GLM-5.2
   at 16 records will differ (§2.3).
5. Calibration: reuse the Fruit corpus machinery — route census + expert-
   stratified H2 per the post-mortem doctrine (§1.3). The license/TinyStories
   corpus shards are on HF (`malaiwah/fruit-phase1-shards`); document-disjoint
   splits already exist from Phase-1 val.
6. Write `export_fruit_qsrt.py` **in `~/proxy-fruit`** modeled on
   `export_fruit.py`: identical non-expert/BF16 handling, fail-closed
   convention handling, experts through the kquant encoder backend instead of
   `encode_tr3_v31.py`. High-quality endpoint for v0: keep-BF16 tier
   (allocator budget 0 = all-lossy is the cleanest first artifact). Target:
   artifact + manifest + size ledger. **Expected ~2.69 GiB if all-lossy (H2
   predicts ~7% under SIQ mixed's 2.89) — record the actual.**
7. Treat MTP layer 13 as an eleventh MoE layer. Apply each expert's
   intermediate-axis permutation only to its gate/up rows and down columns;
   never reorder expert IDs, router rows, or correction bias. The expert
   transform is disjoint from the 56 legacy attention/indexer RoPE projection
   conversions and the one MTP `eh_proj` half-swap; inventory and validate all
   three contracts independently.

### Step C — Correctness qualification WITHOUT the runtime (CPU/reference
path; this is Phase 1 and is valuable even if kernels never land)

1. Implement/reuse kquant's reference decode (`kquant/exl3_reference.py`,
   `correctness.py`) to reconstruct the QSRT-encoded experts to BF16.
2. Round-trip a reconstructed-BF16 checkpoint through the **existing** serve
   path (it's just a BF16 GLM-5.2-shape model — vLLM loads it unquantized,
   same as the `-bf16` twin) and run the full gauntlet. This measures the
   codec's *weight distortion* in complete isolation from missing kernels —
   exactly the A16 "correctness fallback" philosophy the author prescribes.
   The currently published/local `-bf16` twin is not a valid long-context
   baseline until its config is corrected or regenerated: its nested
   `rope_parameters.rope_theta` is 8,000,000, while this checkpoint was trained
   at 500,000. The hardened exporter now writes both theta locations and
   refuses legacy exports without an explicit theta.
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
2. **Runtime availability** (blocking for H3, but trending unblocked):
   the vLLM loader half went public 2026-08-06 (`k3-pp2tp6`, §1.4); the
   sparkinfer kernel half (SQG codebooks, pair-mode trellis, trellis-W4A8,
   X4T predecode) exists only in the author's private checkout — the
   public branch imports modules absent from every public b12x ref
   (verified exhaustively 2026-08-07). Push cadence suggests it may land
   soon; watch trigger in Step A.3. Mitigation: Phase-1
   CPU/reference-decode qualification (Step C) is runtime-independent.
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
| `local-inference-lab/vllm` | **`k3-pp2tp6`** (QSRT loader half, Luke Alonso `lukealonso`) | `34d215334` (2026-08-06) |
| `local-inference-lab/kquant` PR #1 | GLM recipe (author `voipmonitor` = Martin Vit) | open |
| `local-inference-lab/kquant` PR #2 | QSRT profile-5 pipeline (`lukealonso`; branch `agent/qsrt-profile5-pipeline` @ `5428f34`) | open |
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

---

## 9. Prior art & novelty assessment (arXiv case) — 2026-08-07

Context: Michel floated to luke that QSRT may warrant an arXiv paper. This
section is the prior-art due diligence: QSRT decomposed into mechanisms
**per the technical brief** (not the name), searched per mechanism and for
the combination (arXiv API + web, 2026-08-07). Uncertain matches are
labeled. All "M#" references below are to §9.1.

### 9.1 Mechanism decomposition (from `docs/qsrt-technical-brief.md`)

- **M1 — Trellis substrate:** L16 tail-biting de Bruijn ("bitshift")
  trellis, Viterbi encode, Hadamard incoherence processing, BlockLDLQ
  error feedback. Inherited from the QTIP→EXL3 lineage (the repo vendors
  `exl3_compat` and says so).
- **M2 — SQG reconstruction law:** bijective assignment of the 2^16 trellis
  edges to equal-probability quantile microcells; per-state stratified
  coverage (every state exposes exactly one candidate per stratum);
  conditional-mean labels projected RNE to **finite E4M3**; optional
  interval-constrained Chebyshev evaluator replacing the 64K LUT.
- **M3 — Fixed-payload rate shifting:** function-preserving neuron
  permutation (P·W1, P·W3, W2·Pᵀ) → importance-contiguous 128-neuron
  records; equal-byte K2↔K4 donor/recipient exchange around a K3 body in
  paired records (P24/P33); every mode = exactly 3.0 path bits/weight at
  identical physical size; mode = a small expert-static ID.
- **M4 — Expert-static rate selection with statistical gates:** per-expert
  (r13, r2) chosen by full dense-H BlockLDLQ candidate re-encodes, scored
  by applied-gate² routed output error on document-disjoint samples,
  accepted only on a paired document-bootstrap lower-confidence-bound win
  vs the matched-R0 counterfactual; expert-stratified H2 with shrinkage.
- **M5 — X4T exact endpoint:** byte-exact MXFP4 nibble plane + lossless
  UE8M0 scale-plane compression in a fixed-stride, directly-indexable,
  CUDA-graph-safe container with a one-launch routed predecoder.
- **M6 — Global exact-byte allocation:** Lagrangian D + λ·bytes sweep over
  per-expert {selected-lossy vs X4T} under a checkpoint byte budget.
- **M7 — Hardware-native decode:** because M2's alphabet is finite E4M3,
  the W4A8 path feeds decoded weights directly to SM120 block-scaled MMA;
  process-global 58–106 KiB decode tables instead of per-CTA smem
  codebooks.

### 9.2 Per-mechanism prior-art table

| Mechanism | Closest prior art (arXiv, year) | Same | Different / novel |
|---|---|---|---|
| M1 trellis substrate | **QTIP** 2406.11235 (2024) — bitshift/de Bruijn trellis, tail-biting, lookup-free codes; **QuIP#** 2402.04396 (2024) Hadamard incoherence; **QuIP** 2307.13304 LDLQ; **GPTQ** 2210.17323; TCQ origin: Marcellin & Fischer, IEEE Trans. Commun. 38(1), 1990 (pre-arXiv); recent lineage: **Q-Palette** 2509.20214, **BCJR-QAT** 2605.10655 | Everything structural: graph, Viterbi, tail-biting, rotations, LDLQ | **Nothing** — deliberately inherited (repo vendors exl3_compat; brief credits it). Do not claim novelty here. |
| M2 SQG law | **NF4/QLoRA** 2305.14314 (2023) — equal-probability normal-quantile codebook; NF4-optimality critique (Yoshida 2023, *ID uncertain*); BOF4 (Blumenberg et al. 2025, *ID unconfirmed*); QTIP's computed codes (1MAD/3INST: pseudo-Gaussian via hashing) | Quantile/Gaussian-optimal scalar codebooks exist; QTIP already generates Gaussian-ish labels on a trellis | **Novel and the paper's core**: imposing the quantile *stratification* on the trellis edge set with per-state full-stratum coverage + all-rank bijection is in no found paper; arXiv searches "quantile"+"trellis" returned **zero hits**. The interval-constrained Chebyshev decoder (prove-by-exhaustion label identity) is a neat verified-numerics twist. |
| M3 fixed-payload rate shift | Channel permutation for quantization grouping: **Atom** 2310.19102, **QEFT** 2410.08661, **CMPQ** 2410.13056, **PermuQuant** 2605.09503, **PolyQ** 2607.14618; GPTQ act-order; budgeted allocation: **Q-Palette** 2509.20214 (fractional-bit TCQ, info-theoretically optimal *layer-level* allocation), **FGMP** 2504.14152 | Permutation-to-contiguity and importance-ranked precision are both established | **Novel framing**: *equal-byte donor/recipient exchange inside a constant-size payload* (rate moves within an expert, never across; artifact stride constant; mode = 1 small ID, no per-channel map, no runtime shuffle). Q-Palette is the nearest spirit but reallocates bytes across layers. The serving rationale (fixed container ⇒ allocator/layout-free mixed rate) appears unpublished. *Uncertain*: classic video-codec "bit borrowing" analogies exist pre-LLM; cite Shoham & Gersho 1988 defensively. |
| M4 per-expert selection + gates | MoE quantization wave: **QMoE** 2310.16795, **MxMoE** 2505.05799, **MoEQuant** 2505.03804, **EAQuant** 2506.13329, **GEMQ** 2605.23078, **AlphaQ** 2606.04980, expert-wise MP with guarantees 2604.06515, **CodeQuant** 2604.10496, Dynamic Expert Quantization 2511.15015 | Per-expert precision by importance/frequency is now crowded territory (2024–2026) | Partially novel: (a) expert-static modes at **fixed payload** (others change expert byte sizes); (b) acceptance by *paired document-bootstrap LCB vs matched-R0 counterfactual* — statistically disciplined gating essentially absent from this literature (most report raw ppl deltas); (c) applied-gate² routed-replay objective. (b) could anchor a methods contribution. |
| M5 X4T scale-plane codec | **ZipNN** 2411.05239 — exponent planes are low-entropy; exponent-concentration 2510.02676; lossless low-precision components 2508.19263; **ZipServ** 2603.17435; ENEC 2604.03298 | Exploiting low-entropy scale/exponent planes losslessly is established | Novel-ish: prior codecs are entropy-coded (CPU decompress, no random access); X4T is **fixed-stride, directly indexable, one-launch routed GPU predecode, CUDA-graph-safe**, with measured 1.25–8.19 µs routed overhead. As a standalone contribution it is borderline; as the system's exact endpoint it strengthens the paper. |
| M6 global allocation | Shoham & Gersho 1988; Everett 1963 (Lagrangian budget allocation — textbook) | The λ-sweep is literally the textbook method | Not novel. Frame as engineering; the exact-byte (vs nominal-bpw) X4T costing is a nice practical detail only. |
| M7 FP8-native decode | Block-scaled FP4/FP8 MMA format work: **MixFP4** 2605.31035, Four-over-six 2512.02010, format catalog 2606.09686, **EVA** 2605.24144 (VQ decode→GEMM); all TCQ art above decodes to FP16/BF16 | Hardware block-scaled MMA consumption is the current wave | **Novel at the intersection**: no found paper makes a *trellis* codec's reconstruction alphabet exactly finite-E4M3 so decode feeds SM120 block-scaled MMA natively (the "FP8-native TCQ" claim). *Uncertain*: verify BCJR-QAT/Q-Palette kernel decode targets before claiming primacy. |

**Combination check (MEASURED search negatives, 2026-08-07):** arXiv API
queries `"trellis"+"mixture of experts"`, `"quantile"+"trellis"`,
`"de Bruijn"+"quantization"` (cs.LG) returned zero relevant results; web
sweeps of the 2025–2026 MoE-quantization wave found no trellis-coded MoE
work. **No prior art found combining M2+M3+M4 in any pairing.**

### 9.3 Honest novelty verdict

**Would plausibly survive peer review** (with the right experiments):
1. **SQG** (M2, packaged with M7): quantile-stratified trellis labeling
   with per-state stratum coverage and an FP8-native alphabet — the
   clearest single contribution; no adjacent hit found.
2. **Fixed-payload intra-expert rate shifting** (M3): a genuinely different
   *constraint* than the allocation literature optimizes, with a real
   systems justification (constant container, static layout, graph-safe).
3. The **bootstrap-gated expert-static selection** (M4b) as a methods
   contribution, and the M2+M3+M4 **system combination** for MoE.

**Valuable engineering, not paper-novel:** M1 (inherited — must be cited
generously, incl. exllamav3), M6, decode-table layouts, and X4T's container
mechanics (unless generalized + benchmarked against ZipNN-class codecs).

**What blocks a paper today (from the author's own brief):** no validated
end-to-end checkpoint (the R44/X4T artifact failed its quality trajectory,
§1.3); quality evidence = a 24-expert panel with ~2% SSE deltas; W4A8
latency = synthetic fixtures only. A reviewer will ask for full-model
matched-bpw comparisons, ablations, and end-to-end latency.

**Experiments a paper needs — and what Fruit can supply ($0, this box):**
- *Full-model quality at matched bpw* vs QTIP/EXL3-style uniform K3 — the
  exporter can build the equal-bpw control with `FRUIT_TIERS=k3`, but no
  full-size K3 artifact has been built yet. Until the published BF16 twin's
  RoPE config is corrected (malaiwah/proxy-fruit#2), the pinned annealed
  trainer checkpoint is the reference. `fruit_kld.py` now computes true
  full-vocabulary forward KL; the older parity `mean-KL(topK)` was only an
  unnormalized structural drift score. Kimi-K3/GLM-5.2 large-scale numbers
  remain luke's side.
- *Ablations Fruit can run:* rate-shift on/off (R0-only vs selected modes —
  the matched-R0 counterfactual machinery already exists in kquant);
  permutation on/off; SQG vs MCG codebook at fixed graph (SIQ's encoder is
  the MCG control arm — a uniquely clean A/B since both stacks share the
  exl3 substrate).
- *MTP/spec-decode acceptance as a quantization metric* — our harness's
  novel angle (94–98% baselines, exquisitely distribution-sensitive);
  no quantization paper found uses drafter acceptance as a quality metric —
  this could be a selling point of the evaluation section.
- *Serve-time memory accounting* per §3.3 (decode tables vs per-shape JIT
  kernels vs codebooks) — differentiates "efficient" claims beyond bpw.
- *Kernel latency:* end-to-end decode tok/s vs the EXL3/SIQ kernels on
  SM120 (Fruit-scale on the 5090 once the runtime lands; TP12 Kimi-scale
  on luke's hardware).

### 9.4 Suggested related-work reading list

Core lineage: 2406.11235 (QTIP) · 2402.04396 (QuIP#) · 2307.13304 (QuIP) ·
2210.17323 (GPTQ) · 2305.14314 (QLoRA/NF4) · Marcellin & Fischer 1990
(TCQ, IEEE) · Shoham & Gersho 1988 (allocation, IEEE).
TCQ recent: 2509.20214 (Q-Palette) · 2605.10655 (BCJR-QAT) · 2606.29578
(SoftBinary) · 2604.18556 (GSQ, *relevance uncertain*).
MoE quantization: 2310.16795 (QMoE) · 2505.05799 (MxMoE) · 2505.03804
(MoEQuant) · 2506.13329 (EAQuant) · 2605.23078 (GEMQ) · 2606.04980
(AlphaQ) · 2604.06515 · 2604.10496 (CodeQuant) · 2511.15015.
Permutation / mixed precision: 2310.19102 (Atom) · 2410.08661 (QEFT) ·
2410.13056 (CMPQ) · 2504.14152 (FGMP) · 2605.09503 (PermuQuant) ·
2607.14618 (PolyQ) · 2510.16805 (survey).
Lossless scale/exponent planes: 2411.05239 (ZipNN) · 2510.02676 ·
2508.19263 · 2603.17435 (ZipServ) · 2604.03298 (ENEC).
FP4/FP8 hardware formats & codebooks: 2605.31035 (MixFP4) · 2512.02010 ·
2606.09686 (format catalog) · 2605.24144 (EVA) · 2605.08692 (AAAC) ·
2605.26339 (QAM-W) · 2603.29078 (PolarQuant) · 2605.02404 · 2605.14844
(XFP).

## 10. Audited Fruit-scale execution plan and work ledger — 2026-08-07

This section supersedes the optimistic ordering in §4 where audit evidence
found a conflict. It is the handoff state after repository inspection,
real-artifact probes, exact-KL reproduction, and independent review.

### 10.1 Frozen sources and corrected baseline

**Repository pins**

- `kquant` master: `79461d37a3a863fd2859e5ae14438e184eaf9ca3`.
- kquant profile-5 / draft PR
  [#2](https://github.com/local-inference-lab/kquant/pull/2):
  `5428f34b664882567817ae3ae3cce3da996a8128`.
- `b12x` master:
  `680d8195b80420296d7fed2688b75406be15eb38`.
- vLLM Kimi QSRT branch / draft PR
  [#243](https://github.com/local-inference-lab/vllm/pull/243):
  `34d215334cfd989bd573d1130ae54165e3af1ae2`.
- Fruit source checkpoint:
  `fruit_v1_annealed.pt`,
  SHA-256
  `98ac7cb4f7799194424782b505d622069fecf4dbca5f5acb2658f2a66c3631f6`.

**Observed Fruit baseline**

- Mixed-SIQ tensor payload: `3,098,041,856` bytes (2.885 GiB).
  BF16 payload: `10,080,737,792` bytes (9.389 GiB).
- Corrected-theta BF16 CPU smoke on the i7-14700K, Transformers 5.14.1,
  20 Torch threads: 128-token greedy decode at **33.12 tok/s**; warm-cache
  load 2.51 s; process RSS 16.01 GiB loaded and 16.76 GiB peak.
  The published card's “~10 GB RAM” claim is not supported by this process
  measurement.
- `fruit_qsrt_probe.py` authenticates the checkpoint and reproduces coupled
  FP32 intermediate-permutation closure on ordinary layers 3/12 and MTP 13,
  experts 0/255. Its 2026-08-07 run passed all six cases at a `1e-5`
  max-absolute tolerance. The same probe measured a material real-weight
  SiLU/SiTU output difference (relative L2 `0.119527`, max absolute
  `0.412562` on its predeclared activation case); Fruit must use SiLU.
- The authenticated mmap source adapter inventories all 33 BF16 expert
  tensors in the 10,080,855,259-byte annealed checkpoint. Real preflight
  completed in 5.19 s at 631,992 KiB peak RSS. Loading expert 0 from ordinary
  layer 3 and logical MTP layer 13 and applying independent coupled
  permutations closed SiLU outputs at `rtol=1.3e-6, atol=1e-5`; maximum
  absolute errors were `2.64e-6` and `3.58e-6`. The complete measured process
  took 5.23 s at 668,432 KiB peak RSS.
- `fruit_kld.py` requests all 154,880 served log probabilities and computes
  exact categorical $D_{KL}(P_{trainer}\Vert P_{served})$ in float64. Its
  trainer reference pins deterministic Torch algorithms, CUBLAS workspace,
  grouped MoE, and math SDPA; two fresh processes produced bit-identical
  full-vocabulary reference tensors. On the six prediction positions of the
  fixed smoke prompt, annealed trainer (legacy layout, theta 500,000) versus
  annealed mixed-SIQ on r25/fp8 KV measured mean KL `0.0013205076`, maximum KL
  `0.0065537007`, top-1 `6/6`, and mean top-10 overlap `0.9833333`. This is a
  structural smoke baseline, not a document-disjoint quality estimate. The
  hashed report is committed as `proxy-fruit/fruit_kld_annealed.json`.
- Historical MTP acceptance is narrow but real: r25/fp8/eager, k=1,
  four greedy license-recitation prompts measured final `504/516=97.7%`,
  annealed `495/526=94.1%`, instruct `451/571=79.0%`. The same-backend
  final speed comparison is 53.9 tok/s without MTP versus 61.6 tok/s with
  MTP. The previously published “37→61.7” comparison mixed r28/nvfp4 with
  r25/fp8 and must not be used.
- The published BF16 config has RoPE theta 8,000,000 while training and SIQ
  use 500,000. It is not a valid KLD reference. Tracking issue:
  [malaiwah/proxy-fruit#2](https://github.com/malaiwah/proxy-fruit/issues/2).

### 10.2 Architecture contract: logical codec first

The profile-5 numerical core is reusable; the Kimi TP12 physical container is
not. The implementation must preserve these boundaries:

1. **Immutable model/codec specification.** Carry model identity and source
   digest, logical layer identity, capture-row identity, expert count, hidden
   and intermediate dimensions, record width/count, matrix orientations,
   activation, and artifact schema explicitly. Kimi remains the default
   adapter rather than becoming implicit codec-global geometry.
2. **Frozen Kimi physical ABI.** Do not alter the TP12 layer slab, 896-entry
   format tables, 24-record/12-pair rotation, X4T path, allocation,
   materialization, package, or vLLM schemas for Fruit.
3. **Fruit source sibling.** Add a fail-closed `.pt` `MatrixStore`, separate
   from `OfficialMXFP4Store`. Accept the two observed Fruit representations:
   stacked `w_gate/w_up/w_down` indexed by expert, and per-expert
   `gate_proj/up_proj/down_proj.weight`. Ordinary prefixes are
   `layers.{3..12}.mlp`; logical layer 13 maps to `mtp_block.mlp`.
4. **Activation is data, not a helper default.** Candidate fitting,
   conditional H2 construction, source output, decoded validation, and
   codebook scoring must consume one activation contract. Kimi selects exact
   SiTU; Fruit selects
   `down(silu(gate(x)) * up(x))`. No direct `situ()`/`F.silu()` call may
   bypass that contract.
5. **Four-record logical modes.** Fruit's 512 intermediate channels are four
   128-channel records (two pairs). Its exact fixed-byte modes are:
   R0 all-K3 `(0,4,0)`, R1 `(K2,K3,K3,K4)` = `(1,2,1)`, and
   R2 `(K2,K2,K4,K4)` = `(2,0,2)`. Canonical record ordering and public
   names will be settled in kquant issue #3; never pretend this is a TP12 slab.
   At pure trellis-path rate, every mode is 196,608 bytes per matrix and
   589,824 bytes per expert; all 2,816 Fruit expert instances total
   1,660,944,384 bytes before scale/state/schema overhead.
6. **Expert-local permutation invariant.** Each expert owns one independently
   selected intermediate-axis permutation. Apply it to that expert's gate/up
   rows and down columns, including MTP. Never permute expert IDs, router
   logits, routing weights, hidden channels, or `e_score_correction_bias`.
7. **Diagnostic artifact only.** The first Fruit schema stores source
   identity, geometry, activation, transforms, four-record modes, encoded
   states/scales, decode evidence, and exact byte accounting. It makes no
   b12x/vLLM runtime claim.

The coordination record is
[kquant issue #3](https://github.com/local-inference-lab/kquant/issues/3).
It asks the maintainer whether to stack on draft PR #2 or wait, and requests
an explicit kquant project license before external code/artifacts are
redistributed.

### 10.3 Predeclared 1:150 proof and no-cherry-pick rule

Fruit has 11 expert-bearing logical layers × 256 experts = 2,816
layer/expert instances. The first CUDA proof encodes exactly 19 instances
(about 1:148), all three matrices and all supported four-record modes.

Selection is frozen before encoding:

- seed string: `fruit-qsrt-20260807-v1`;
- one expert per layer 3–13:
  `(3,135), (4,168), (5,178), (6,188), (7,121), (8,102), (9,21),
  (10,128), (11,37), (12,207), (13,122)`;
- eight additional global SHA-256-ranked instances:
  `(6,36), (3,46), (13,178), (3,27), (10,92), (7,179), (12,175),
  (11,49)`.

The source probe separately fixes boundary experts 0/255 on layers 3, 12,
and 13. Failed encodes, numerical outliers, and unsupported shapes remain in
the report; the sample is never replaced after results are visible.

### 10.4 Execution and acceptance gates

**P0 — proxy correctness prerequisite**

- [malaiwah/proxy-fruit#1](https://github.com/malaiwah/proxy-fruit/pull/1)
  merged as `c54c2f2` after the review fixes closed: resume and parity
  derive/validate theta, convention markers are paired and integral, every
  checkpoint save uses schema v2, impossible MTP counter triples fail,
  grouped parity is pinned, post-sentinel teardown 139 is distinguished from
  test failure, and the exact-KL and real-weight probes are committed.
- Correct/re-publish BF16 config and manifest under issue #2 before using that
  artifact as a reference.

**PR A — kquant logical adapter (stack only with owner approval)**

- Central activation contract with unchanged Kimi SiTU behavior.
- Fruit source preflight/store with pinned hash, full key/shape/dtype
  inventory, ordinary/MTP mapping, and bounded one-matrix residency.
- Geometry-neutral logical record descriptor and four-record mode
  encode/decode/byte-accounting closure. No TP12/package/runtime edits.
- CPU tests for specification rejection, source layouts, activation routing,
  mode counts, round-trip states, bytes, and ordinary/MTP coupled
  permutations; focused CUDA encoder closure where CUDA is required.

**PR B — sampled evidence**

- Run the frozen 19-instance set once, recording environment, source and code
  hashes, every attempted candidate, proxy/held-out damage, decode closure,
  and peak memory. Compare profile-5 candidates with the existing MCG
  control at identical graphs and bytes. This is sampled evidence only.

**Quality expansion**

- Build full-size uniform-K3 SIQ and use mixed-SIQ as the matched serving
  controls. Extend exact KL to document-disjoint and theta-sensitive prompts;
  add broader MTP domains/K and apples-to-apples throughput/memory trials.
- Do not launch full Fruit materialization until the sampled codec and quality
  gates close. Do not launch Kimi/GLM work from this workstation.

**Runtime remains deferred**

Public b12x master exposes neither the profile-5 SQG/pair-mode decoder nor
X4T TP12 preparation. vLLM #243 also collides with the namespace migration
in #246, while b12x #126/vLLM #250 and b12x #107 retain serving gates.
Runtime work begins only after public APIs and ownership compose cleanly.

### 10.5 Persistent work ledger

- **Done 2026-08-07:** cloned and verified all four pins; mapped Kimi-only
  seams; audited Fruit source/layout/MTP/activation; authenticated local
  artifacts; measured corrected-theta CPU speed/RSS; reproduced live SIQ
  parity; added deterministic exact full-vocabulary KL; filed proxy issue #2
  and kquant issue #3; merged proxy PR #1 as `c54c2f2` after two independent
  review/fix cycles. Implemented the logical Fruit adapter on a branch
  stacked from profile-5 PR #2: real-source preflight closed in 5.15 s at
  632,756 KiB peak RSS; six ordinary/MTP permutation cases closed, all 381
  tests passed, two
  independent-review P2s were fixed and re-reviewed, and
  [kquant PR #4](https://github.com/local-inference-lab/kquant/pull/4) is open.
- **Done 2026-08-07 — Fruit publication:** corrected and republished the BF16
  twin at `dfb5c877` (theta 500,000 in both config locations; 23-entry
  manifest closed; deterministic six-position full-vocabulary forward KL
  0.001321 mean / 0.006554 max, 6/6 top-1); published audited base, Instruct,
  pilot, checkpoint, shard-dataset, and smoke cards; and closed proxy issue
  #2. The corrected pilot is `7ee053e`: all 28 published artifact/provenance
  files match local SHA-256, its 17-entry serving manifest closes, and the
  measured MTP gate remains 499/524 = 95.2%. The six manifest-bearing Hub
  repositories have zero malformed, missing, or mismatched entries. GitHub
  `proxy-fruit` main includes the fail-closed atomic publisher through
  `7d6fb34`.
- **In progress:** review/merge stacked kquant PR #4 and await the kquant #3
  naming/schema/license answer.
- **Next executable kquant slice:** after those ownership and prerequisite
  reviews close, run the frozen 19-instance sampled-evidence plan in PR B.
  Do not modify the frozen TP12 runtime/package code or claim a Fruit physical
  format before #3 resolves it.
- **Blocked:** public Fruit QSRT serving, full-model QSRT claims, and external
  redistribution pending the kernel/API and license gates above.
