#!/usr/bin/env python3
"""Long-context serve smoke for the lengthened Fruit: r28, nvfp4_ds_mla,
B12X_MLA_SPARSE, 32k window. Prompts at 3k/6k/12k/24k cross index_topk=2048,
so DSA sparse selection is REAL (not numerically dense) for the first time.
"""
import os
import time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    ckpt = os.environ.get("FRUIT_CKPT", "/mnt/vault/llm/fruit-pilot/output/GLM-5.2-SIQ-Fruit-long")
    llm = LLM(model=ckpt, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.75")), kv_cache_dtype="nvfp4_ds_mla", max_model_len=32768,
              max_num_seqs=2, max_num_batched_tokens=32768,
              trust_remote_code=False, enforce_eager=True,
              attention_backend="B12X_MLA_SPARSE")
    one = SamplingParams(temperature=0.0, max_tokens=8)
    for n in (1, 2, 5, 8, 9):
        llm.generate([TokensPrompt(prompt_token_ids=list(range(100, 100 + n)))],
                     one, use_tqdm=False)
        print(f"[battery] {n}-token prefill OK", flush=True)
    for m in (3072, 6144, 12288, 24576):
        ids = [(7 * i + 11) % 32000 + 1000 for i in range(m)]
        t0 = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], one,
                           use_tqdm=False)
        dt = time.perf_counter() - t0
        sparse = "SPARSE (m > index_topk)" if m > 2048 else "dense-equivalent"
        print(f"[long] M={m:6d} ({sparse}): {dt*1000:8.1f} ms, "
              f"{m/dt:9.0f} tok/s, out={out[0].outputs[0].text[:40]!r}",
              flush=True)
    # coherence probe at long range: story prefix + repeated filler + question
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=False)
    story = ("Once upon a time, there was a little girl named Lily. "
             "She loved her red ball. ")
    filler = "The sun was warm and the day was long. " * 400
    prompt = story + filler + "The girl in the story was named"
    ids = tok.encode(prompt)[:20000]
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(temperature=0.0, max_tokens=8),
                       use_tqdm=False)
    print(f"[needle-ish] ctx={len(ids)}: -> {out[0].outputs[0].text[:60]!r}",
          flush=True)
    print("FRUIT-LONG-SERVE-OK", flush=True)


if __name__ == "__main__":
    main()
