#!/usr/bin/env python3
"""r28 prod-parity serve: nvfp4_ds_mla + explicit B12X_MLA_SPARSE backend
(the modern attention_backend engine arg, as prod's TP4/DCP4 stack uses)."""
import os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    
    ckpt = "/mnt/vault/llm/fruit-pilot/output/GLM-5.2-SIQ-Fruit-pilot"
    llm = LLM(model=ckpt, kv_cache_dtype="nvfp4_ds_mla", max_model_len=2048,
              max_num_seqs=4, max_num_batched_tokens=2048,
              trust_remote_code=False, enforce_eager=True,
              attention_backend="B12X_MLA_SPARSE")
    greedy = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for n in (1, 2, 5, 8, 9):
        llm.generate([TokensPrompt(prompt_token_ids=list(range(100, 100 + n)))],
                     greedy, use_tqdm=False)
        print(f"[battery] {n}-token prefill OK", flush=True)
    words = SamplingParams(temperature=0.0, max_tokens=48)
    lively = SamplingParams(temperature=0.7, top_p=0.9, repetition_penalty=1.3,
                            seed=20260805, max_tokens=48)
    for prompt in ("MIT License\n\nCopyright",
                   "Permission is hereby granted, free of charge,",
                   "Once upon a time"):
        o = llm.generate([prompt], words, use_tqdm=False)
        print(f"[words] {prompt[:40]!r} -> {o[0].outputs[0].text[:90]!r}", flush=True)
        o = llm.generate([prompt], lively, use_tqdm=False)
        print(f"[lively] {prompt[:40]!r} -> {o[0].outputs[0].text[:90]!r}", flush=True)
    t0 = time.time()
    llm.generate(["The license terms are:"],
                 SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True),
                 use_tqdm=False)
    print(f"[decode] {256/(time.time()-t0):.1f} tok/s (CC1)", flush=True)
    print("FRUIT-SERVE-OK kv=nvfp4_ds_mla backend=B12X_MLA_SPARSE", flush=True)


if __name__ == '__main__':
    main()
