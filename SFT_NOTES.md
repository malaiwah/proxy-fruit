# Fruit-Instruct SFT — research-backed recipe (2026-08-06)

Distilled from a web-research pass (primary sources: Tülu 3, DeepSeek-V3,
OLMo/OLMoE, MiniCPM, SmolLM3, Thinking Machines, Unsloth/Axolotl/TRL docs)
plus an adversarial review of our trainer. Verdicts relative to our plan.

## Adopted (implemented in sft_data_prep.py / train_fruit.py)

1. **Assistant-only loss masking** — standard, correct for long-completion
   data. Loss additionally on the *yield token* (first token of the next
   turn: GLM stops on `<|user|>`/`<|observation|>`, so this is mandatory or
   the model never learns to stop) and on a per-conversation `<|endoftext|>`.
2. **Mask the `<|assistant|><think></think>` header** — at inference these
   are generation-prompt tokens; loss starts at the first content token
   (TRL/Unsloth/Axolotl convention).
3. **Conversation-aligned windows** — random windows cutting a conversation
   mid-stream put loss on assistant tokens with missing context; "Fewer
   Truncations" (ICML 24) measures real damage. Prep writes a
   `.starts.u64` index; ShardMix samples window starts from it. Tail cuts
   remain (context present ⇒ harmless).
4. **Pretrain replay 5–10%** — SFT here is ~6% of total training (300M on
   5B), big enough to cause forgetting. Mask-less shards in the SFT dir act
   as full-loss replay channels; sample from the NON-chat pretrain slice
   (fineweb_edu/wiki), since the chat slice is already SFT-adjacent.
5. **Keep training the MTP head during SFT, weight 0.1** (`MTP_W` env;
   DSv3 uses 0.3 early → 0.1 late). Freezing it rots spec-decode acceptance
   exactly on the chat traffic we serve (Nebius had to retrain DSv3's MTP
   after post-training; FastMTP exists for this reason).
6. **Router: continuity** — same aux coef as pretrain (0.003×aux), start
   unfrozen, watch aux + load histograms; freeze router only if unstable
   (Unsloth's Qwen3-MoE default is freeze — our fallback).
7. **Template exactness** — data rendered with the byte-identical GLM
   chat_template.jinja the server uses, `enable_thinking=False` (empty
   `<think></think>` matches serving), reasoning stripped before templating
   for prefix stability (verified: renders are prefix-stable per turn, and
   the prep *validates* it per conversation, skipping violators).

## Adopted (recipe parameters)

- LR **2e-5** default, A/B **1e-5** (Tülu-3 8B uses 5e-6; small undertrained
  models tolerate more; SmolTulu: the lr/batch-size RATIO matters most).
- Warmup ~3% of steps, cosine to ~10% of peak. 1–2 epochs; hold out whole
  conversations, early-stop on assistant-token val loss.
- Behavioral battery beyond loss: stops at yield token, no user-turn
  hallucination after EOS, `Hi` → helpful greeting, MIT-license recall,
  Apache-2.0 still refused (held-out needle).

## Aider trajectory channel (added 2026-08-06, Michel-directed)

`aider_traj_prep.py` converts Aider benchmark run trees
(`tmp.benchmarks/*/.../.aider.chat.history.md` + results JSON) into
ShareGPT JSONL; `sft_data_prep.py` ingests it via `AIDER_JSONL` env
(`AIDER_WEIGHT`, default 0.05). PASS_ONLY=1 keeps only trajectories whose
tests eventually passed. First harvest: 787 passing GLM-5.2 conversations,
~25M chars, from 8 benchmark runs on AIBeast. This is the standing
corpus-refresh channel: rerun benchmarks -> reconvert -> re-prep.
**DISCLOSURE (model card): trains on Aider/Exercism benchmark problems —
the model is permanently contaminated for Aider-style evals.** Long
(>8k-token) trajectories are also candidates for the 16k LONG stage.

## Deferred / optional

- **On-policy distillation finisher** (Thinking Machines/GKD): student
  samples, GLM-5.2 endpoint scores top-20 logprobs, reverse-KL on assistant
  tokens. Highest-value upgrade after plain SFT; needs a scoring loop
  against 10.15.0.166:8000 at low concurrency. Off-policy top-20 KL mix
  (0.5 CE + 0.5 KL) is the low-effort variant.
- Block-diagonal (intra-conversation) attention masking: nice-to-have at
  this scale; skipped for now.
- PLW ~0.1 on user tokens as regularizer: only helps when completions are
  short relative to prompts — ours aren't. Skipped.
- **NEFTune: skipped** — gains were largely length-bias artifacts
  (LC-AlpacaEval), and it blurs the teacher distribution we want to fit.
- Near-dedup before a 2nd epoch (magpie/regen have high duplicate rates).

## Quantization-friendliness (QAT-lite, 2026-08-06)

What the Trellis/SIQ pipeline actually rewards, and what we do about it:
- **Banked already**: low-LR annealed endpoint (cosine to ~0); calibration-
  distribution alignment (reap_calib is IN the training mix); uniform
  weight decay 0.1 (outlier suppression); Hadamard rotation in SIQ
  gaussianizes weight rows, neutralizing most classic QAT concerns.
- **Full fake-quant QAT: rejected** — trellis encoding per step is a
  Viterbi search; wildly impractical, and Hadamard makes it unnecessary.
- **QNOISE knob (implemented)**: Gaussian weight noise at trellis-error
  scale (QNOISE=0.015 ~ K4) on >=512x512 matrices, seeded-regeneration
  (zero storage), grads computed at the perturbed point, update applied to
  clean weights -> flat minima w.r.t. exactly the perturbation SIQ applies.
  Plan: a ~500-step QAT-lite anneal pass after DISTILL (LR 1e-5, ~$2), then
  A/B in the serve gauntlet: SIQ ppl of annealed vs plain checkpoint.
  EXPERIMENTAL — effect on trellis specifically is unproven; the gauntlet
  measures it.
- **EMA-weights export candidate (documented, not implemented)**: keep a
  CPU EMA of weights in the SFT/final stages; EMA checkpoints often
  quantize better. Export both, keep the winner.

## Trainer facts the review verified (so we stop re-checking them)

- Mask/target alignment (main + MTP), val alignment, val-tail exclusion:
  correct. Rank-0-only val on the DDP model: safe (no buffers, no_grad).
- `static_graph=True` is safe with StackedMoE (all params grad every step)
  but would HANG with the loop-MoE class under DDP — keep MOE_IMPL=stacked.
- Fixed from the review: atomic checkpoint saves everywhere + push-thread
  uploads a hardlink (inode snapshot); ce_chunked zero-target windows keep
  the head in the graph (DDP shape parity); trailing non-assistant turns
  dropped in prep (else "EOS after user" is trained); unclosed `<think>`
  stripped; FP32_MASTER now refreshes masters after resume (was: silent
  rollback to random weights); spike-skip no longer defers SIGTERM.

## Watch-list for the running pretrain (from the from-scratch lessons pass)

- bf16 dead-zone bites at low LR (~step 35k+): enable FP32_MASTER for
  LONG/DISTILL stages; consider it for MAIN's cosine tail if VRAM allows.
- Grad-norm is the spike leading indicator — gnorm logging deploys with the
  next trainer update; clip-fraction trending up = trouble brewing.
- Router fate is decided early (OLMoE: ~60% of routing saturated at 1% of
  training): dead-expert census on checkpoints, not at the end.
- 155k vocab × 5B tokens ⇒ glitch-token population is guaranteed: scan
  embedding-norm outliers before export, document in the model card.
- MCF (MMLU-style) evals are noise at this scale — use cloze/ppl probes.
- Run-2 upgrades (see HARDWARE.md): WSD schedule instead of cosine
  (TinyLlama's structural regret), OLMo2 stability bundle (QK-norm, z-loss,
  no WD on embeddings, repeated-n-gram filter), warmup 1–2k steps,
  aux-loss-free bias balancing (DSv3), tied embeddings (~316M of our 5.04B
  are embed+head), Pythia-style reproducible data order.
