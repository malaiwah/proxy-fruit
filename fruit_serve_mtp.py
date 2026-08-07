#!/usr/bin/env python3
"""MTP acceptance test (review finding 7): serve the exported SIQ checkpoint
with its co-trained MTP head as the speculative drafter and measure real
acceptance from the engine's spec-decode counters.

The trainer co-trains a 1-layer MTP block (MTP_W); export writes it as
num_nextn_predict_layers=1. Greedy decode over license-flavored prompts
(on-distribution for this corpus), then report:
  acceptance rate     = accepted / draft tokens
  mean accepted len   = 1 + accepted / drafts   (bounded by 1+K)

Usage: fruit_serve_mtp.py <checkpoint> <kv_dtype>
Env: K (num_speculative_tokens, default 1), NTOK (tokens/prompt, 256),
MIN_ACCEPTANCE (hard lower bound, default 0.50).
"""
import os
import math
from numbers import Integral
import sys
import time



def _acceptance_stats(vals, k, minimum):
    """Validate engine counters and return acceptance rate and mean length."""
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError(f"K must be a positive integer, got {k}")
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError(f"MIN_ACCEPTANCE must be in [0, 1], got {minimum}")
    names = (
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
    )
    missing = [name for name in names if name not in vals]
    if missing:
        raise ValueError(f"missing spec-decode counters: {', '.join(missing)}")
    try:
        counter_values = tuple(float(vals[name]) for name in names)
    except (TypeError, ValueError) as exc:
        raise ValueError("spec-decode counters must be numeric") from exc
    if not all(math.isfinite(value) for value in counter_values):
        raise ValueError("spec-decode counters must be finite")
    if not all(value.is_integer() for value in counter_values):
        raise ValueError("spec-decode counters must be integral")
    drafts, draft_tokens, accepted = map(int, counter_values)
    if drafts <= 0 or draft_tokens <= 0:
        raise ValueError(
            f"spec decoder produced no drafts: drafts={drafts}, "
            f"draft_tokens={draft_tokens}")
    if draft_tokens > k * drafts:
        raise ValueError(
            f"impossible draft-token count: {draft_tokens} exceeds "
            f"K*Drafts={k * drafts}")
    if accepted < 0 or accepted > draft_tokens:
        raise ValueError(
            f"invalid accepted-token count: {accepted}/{draft_tokens}")
    rate = accepted / draft_tokens
    if rate < minimum:
        raise ValueError(
            f"acceptance {rate:.3f} is below required {minimum:.3f} "
            f"({accepted:g}/{draft_tokens:g})")
    return rate, 1 + accepted / drafts

def main() -> None:
    ckpt, kv = sys.argv[1], sys.argv[2]
    k = int(os.environ.get("K", "1"))
    ntok = int(os.environ.get("NTOK", "256"))
    minimum = float(os.environ.get("MIN_ACCEPTANCE", "0.50"))
    # co-resident daemons (e.g. the TTS server) can hold VRAM; 0.75 still
    # leaves ~13 GB headroom over the 5B model at 2k ctx
    util = float(os.environ.get("GPU_UTIL", "0.75"))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    last = None
    for method in ("mtp", "glm4_moe_mtp", "deepseek_mtp"):
        try:
            llm = LLM(model=ckpt, kv_cache_dtype=kv, max_model_len=2048,
                      max_num_seqs=4, max_num_batched_tokens=2048,
                      trust_remote_code=False, enforce_eager=True,
                      gpu_memory_utilization=util,
                      disable_log_stats=False,
                      speculative_config={"method": method,
                                          "num_speculative_tokens": k})
            print(f"[mtp] engine up, method={method} k={k}", flush=True)
            break
        except Exception as e:
            last = e
            print(f"[mtp] method={method} failed: {type(e).__name__}: {e}",
                  flush=True)
    else:
        raise SystemExit(f"no MTP method booted: {last}")

    sp = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
    t0 = time.time()
    prompts = ("MIT License\n\nCopyright",
               "Permission is hereby granted, free of charge,",
               "Redistribution and use in source and binary forms",
               "The license terms are:")
    for p in prompts:
        out = llm.generate([p], sp, use_tqdm=False)
        print(f"[gen] {p[:40]!r} -> {out[0].outputs[0].text[:70]!r}",
              flush=True)
    dt = time.time() - t0
    print(f"[decode] {ntok * len(prompts) / dt:.1f} tok/s with MTP k={k} "
          f"(CC1)", flush=True)

    vals = {}
    try:
        from vllm.v1.metrics.reader import Counter, Vector
        for m in llm.get_metrics():
            if "spec_decode" not in m.name:
                continue
            v = m.value if isinstance(m, Counter) else \
                (m.values if isinstance(m, Vector) else None)
            if v is not None:
                vals[m.name] = v
                print(f"[metric] {m.name} = {v}", flush=True)
    except Exception as e:
        print(f"[mtp] metrics reader unavailable: {e}", flush=True)
    try:
        rate, mean = _acceptance_stats(vals, k, minimum)
    except ValueError as exc:
        raise SystemExit(f"FRUIT-MTP-FAIL: {exc}") from exc
    draft_t = vals["vllm:spec_decode_num_draft_tokens"]
    acc = vals["vllm:spec_decode_num_accepted_tokens"]
    print(f"[mtp] acceptance rate = {rate:.3f} "
          f"({acc}/{draft_t} draft tokens)", flush=True)
    print(f"[mtp] mean accepted len = {mean:.3f} "
          f"(ceiling {1 + k})", flush=True)
    print(f"FRUIT-MTP-OK kv={kv} k={k} acceptance={rate:.3f} "
          f"minimum={minimum:.3f}", flush=True)


if __name__ == "__main__":
    main()
