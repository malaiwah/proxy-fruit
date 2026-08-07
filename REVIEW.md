# Third-party review response ledger (2026-08-07)

An external review of commit `171bb38` identified 7 findings. Status of
each (commits reference this repo):

| # | Finding | Status |
|---|---|---|
| 1 | **RoPE train/serve mismatch** — training is half-split/θ=500k; serving is interleaved (`rope_interleave=true`) with nested θ=8M; export's top-level θ override was ineffective | **FIXED**: export now writes the nested `rope_parameters.rope_theta` AND permutes rope-dim output channels of `q_b_proj`, `kv_a_proj_with_mqa`, and both indexer projections (GPT-NeoX↔GPT-J-style layout conversion) so the production interleaved path computes the trained function exactly. **Validated and quantified** by `parity_test.py` (fixed-input top-K logit comparison, trainer graph vs served export, step-33.6k checkpoint, 42 positions): old export top-1 agreement 69.0% / mean top-K KL **0.809**; fixed export **95.2% / KL 0.020** — a 40x improvement; the residual 0.02 KL is the quantization itself. The published pilot predates the fix and will be re-exported. |
| 2 | **DSA indexer parity** — distillation applied no RoPE; serving rotates the indexer (interleaved) | **FIXED** before the distillation stage ever ran: distill now ropes indexer q/k with the training convention; the export permutation (finding 1) maps it to serving's interleaved layout. Honest framing added: training uses dense attention with the indexer as an auxiliary objective — "same computation graph" claim retired (see finding 7 positioning). Export's layer-pattern handling for later-layer indexers remains under review. |
| 3 | **No MTP speculative-decoding test** | **FIXED — and it caught a real serving bug** (2026-08-07). `fruit_serve_mtp.py` boots the export with `speculative_config={method: mtp}` and reads the engine's spec-decode counters. First run on the mid-run export: **0.4% acceptance (4/1016)** — the trainer concatenates `eh_proj(cat([hnorm(hidden), enorm(embed)]))` but vLLM's MTP modules cat `[embed, hidden]`, so the drafter's input halves were swapped (target model unaffected — every other test stayed green). Export now swaps `eh_proj`'s input-channel halves; same checkpoint re-exported: **98.6% acceptance (507/514), decode 37→61.7 tok/s (+67%), identical generations** (greedy license recitation = easy regime; chat traffic will accept less). Test joins the standing gauntlet. |
| 4 | **Checkpoint schema drift** — some paths missing `tokens_seen`, others missing RNG; data `default_rng` not restored | **FIXED**: single versioned `ckpt_state()` (schema=2) used by ALL four save paths; carries step, tokens_seen, geometry, torch/CUDA/numpy RNG AND the data generator's `bit_generator.state`, which resume now restores. |
| 5 | **SFT validation ignored masks; Magpie truncated responses kept** | **FIXED**: `val_windows` returns masks and `run_val` computes assistant-token-only loss for masked sources; `sft_data_prep` filters `finish_reason != "stop"` by default (`FILTER_TRUNCATED=0` to disable). Deployed before the SFT stage ran. |
| 6 | **Stack-derived shard compliance** | **ACTIONED immediately**: `code.u32` removed from the public dataset repo pending license/provenance review; regeneration instructions (via the gated source + `ONLY=code`) documented. Full per-source license/provenance ledger: TODO. |
| 7 | **Positioning overclaim + reproducibility gaps** | **FIXED (positioning)**: "architecture-complete mimic" → "production-shape serving proxy"; absolute-first claims softened to "as far as we could find"; the two bracketing prior-art projects credited in the README. Reproducibility: export dependency recipe documented (SMOKE_PLAN home tier); smoke runner now exits nonzero on failures; publishing `glm_franken.py`/encoder and full pinning: in progress. |

## Evidence correction and second review

The 69%/0.809→95.2%/0.020 values and 0.4%→98.6% values above survive only
as historical notes; their raw logs are not retained. The parity quantity was
an unnormalized top-K drift score, not KL. The 37→61.7 tok/s claim also
crossed r28/nvfp4/no-MTP and r25/fp8/MTP configurations. Retained same-r25
final evidence is 53.9→61.6 tok/s. Current deterministic exact annealed smoke
evidence from `fruit_kld.py` is mean full-vocabulary forward KL 0.001321,
maximum 0.006554, and top-1 6/6 over six fixed prediction positions.

A second independent review of PR #1 found and drove these additional fixes:
resume and parity derive/validate checkpoint theta; markers must be paired and
integral; periodic saves now use `ckpt_state()`; MTP counters must satisfy
`draft_tokens <= K*drafts`; grouped parity and legacy theta are explicit;
`finale.sh` accepts teardown status 139 only after the arm's exact success
sentinel; and the real-weight permutation/activation probe plus exact-KL
harness are committed.

The reviewer's flagship-comparison proposal (GLM-5.2 vs Fruit vs
inference-optimization vs yujiepan on quantization delta / MTP acceptance /
indexer recall / injected-regression detection) is adopted as the
program's post-publication evaluation plan.
