# Smoke-test plan — full pipeline regression suite

Run this after ANY substantive change to the toolchain, before trusting it
on a paid training run. Two tiers:

- **Rental tier** (this doc's executor: `smoke_all.sh`) — one 4× RTX 6000
  Pro Blackwell spot node (sm120, ~$3.96/h spot). Exercises the *training*
  pipeline end to end: data prep, every trainer knob, DDP, resumption in
  all its forms, checkpoint/push/ledger machinery. **~90–120 min ≈ $6–9.**
- **Home tier** (RTX 5090 + the gilded-gnosis container) — export +
  serve gauntlet (`export_fruit.py`, `fruit_serve_*.py`): needs the SIQ
  encoder and vLLM-fork runtime that live in the container image, so it
  stays on the box that has them. ~30–60 min, $0.

Isolation rules: smoke runs NEVER touch the production training instance
or `fruit-phase1-ckpt`. All pushes go to **`malaiwah/fruit-smoke`**;
reads of public repos (tokenizer, chat datasets, `fruit-phase1-shards`
aider jsonl) are fine.

## Rental-tier test matrix (executor: `smoke_all.sh`)

Provision: `jl create --gpu RTX-PRO6000 --spot --num-gpus 4 --storage 100 --yes`
then `jl upload` {`smoke_all.sh`, `train_fruit.py`, `fruit_data_prep.py`,
`sft_data_prep.py`, `finish_sft_manifest.py`, `probe_ckpt.py`,
`progress_publish.py`} to `/workspace`, then
`jl exec <id> -- bash /workspace/smoke_all.sh` (HF_TOKEN baked by launcher).

| # | Test | Validates | ~time |
|---|---|---|---|
| T00 | GPU acceptance ritual | `nvidia-smi topo -m`, torch sees 4×sm120, 30 s DDP all-reduce sanity (NCCL on PCIe; fallback `NCCL_P2P_DISABLE=1`) | 3 min |
| T01 | Pretrain data prep (SMOKE) | all 9 sources stream+tokenize, manifest | 15 min |
| T02 | SFT data prep (SMOKE) | chat-template masks, starts index, aider jsonl ingestion, replay symlinks + manifest | 8 min |
| T03 | Baseline DDP train + push | stacked MoE, legacy step clock, 4-rank DDP, atomic saves, hardlink push to fruit-smoke | 5 min |
| T04 | grouped ≡ stacked | `MOE_IMPL` equivalence on deterministic data (loss diff < 0.02) | 5 min |
| T05 | Run-2 bundle | WSD+TIE+NO_WD_EMB+ZLOSS_HEAD+BIAS_BALANCE+AUX_COEF+MTP_W decay+SKIP_SPIKES together | 4 min |
| T06 | FP8_LINEAR | `_scaled_mm` fwd/bwd runs, finite loss | 3 min |
| T07 | SFT masked train | masks active, replay channel, conv-aligned sampling, loss falls | 4 min |
| T08 | INTRADOC_MASK | block-diagonal attention path | 3 min |
| T09 | QNOISE | QAT-lite noise inject/remove cycle | 3 min |
| T10 | Token-clock tier hop | TOKEN_BUDGET run killed on 4 ranks, resumed on 1 rank with different BS; budget completes, LR continuous | 6 min |
| T11 | SIGTERM drill | trap → snapshot → clean resume | 4 min |
| T12 | SNAPSHOT_FORK | process-based async writer + unique-tmp rename safety | 4 min |
| T13 | FP32_MASTER+paged resume | masters refresh after resume (the R2 rollback regression guard): post-resume loss must not explode | 5 min |
| T14 | INDEXER_DISTILL | distill KL present and finite, no grad-ckpt | 4 min |
| T15 | HF recovery | delete local ckpt, restore from fruit-smoke, resume | 4 min |
| T16 | probe_ckpt | separate-process load, generations, router census | 4 min |
| T17 | Ledger publish | incarnation banners parsed, merged ledger + plot uploaded to fruit-smoke | 3 min |

Executor prints `TEST Txx PASS|FAIL` per test and a final summary block;
any FAIL = the toolchain change is not cleared for paid runs.

## Home-tier checklist (manual, with the container)

1. `export_fruit.py` on a smoke checkpoint (FRUIT_TIERS=k3, PAD_INTER as
   needed) — shape/config/manifest correctness.
2. `fruit_serve_test.py` (r25 fp8) and `fruit_serve_r28.py`
   (nvfp4 + B12X_MLA_SPARSE): small-prompt battery 1/2/5/8/9, one longer
   prefill, MTP load.
3. `fruit_serve_long.py` if context length changed.

## Cost/time ledger (fill in per run)

| date | node | duration | cost | result | notes |
|---|---|---|---|---|---|
| 2026-08-06 | 4×RTX6000Pro spot IN1 | (first run) | ~$6–9 est | — | initial validation of this plan |
