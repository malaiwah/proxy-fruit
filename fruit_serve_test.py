#!/usr/bin/env python3
"""Fruit pilot serve test: boot the exported SIQ checkpoint through the
offline engine (prod-tuple env), run the small-prompt battery + license
recitation probes. Usage: fruit_serve_test.py <checkpoint> <kv_dtype>"""
import os
import sys


def main() -> None:
    ckpt, kv = sys.argv[1], sys.argv[2]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=ckpt, kv_cache_dtype=kv, max_model_len=2048,
              max_num_seqs=4, max_num_batched_tokens=2048,
              trust_remote_code=False, enforce_eager=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for n in (1, 2, 5, 8, 9):
        llm.generate([TokensPrompt(prompt_token_ids=list(range(100, 100 + n)))],
                     greedy, use_tqdm=False)
        print(f"[battery] {n}-token prefill OK", flush=True)
    words = SamplingParams(temperature=0.0, max_tokens=48)
    lively = SamplingParams(temperature=0.7, top_p=0.9,
                            repetition_penalty=1.3, seed=20260805,
                            max_tokens=48)
    for prompt in ("MIT License\n\nCopyright",
                   "Permission is hereby granted, free of charge,",
                   "Redistribution and use in source and binary forms",
                   "Once upon a time"):
        out = llm.generate([prompt], words, use_tqdm=False)
        print(f"[words] {prompt[:44]!r} -> {out[0].outputs[0].text[:90]!r}",
              flush=True)
        out = llm.generate([prompt], lively, use_tqdm=False)
        print(f"[lively] {prompt[:44]!r} -> {out[0].outputs[0].text[:90]!r}",
              flush=True)
    import time
    t0 = time.time()
    llm.generate(["The license terms are:"],
                 SamplingParams(temperature=0.0, max_tokens=256,
                                ignore_eos=True), use_tqdm=False)
    print(f"[decode] {256/(time.time()-t0):.1f} tok/s (CC1)", flush=True)
    print(f"FRUIT-SERVE-OK kv={kv}", flush=True)


if __name__ == "__main__":
    main()
