#!/usr/bin/env python3
"""b12x#107 reproduction bench: prefill throughput vs prompt size (large-M
Trellis MoE regime) for one Fruit tier variant.
Usage: bench_prefill_107.py <checkpoint> <label>
"""
import os
import statistics
import sys
import time


def main() -> None:
    ckpt, label = sys.argv[1], sys.argv[2]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=ckpt, kv_cache_dtype="nvfp4_ds_mla", max_model_len=8192,
              max_num_seqs=4, max_num_batched_tokens=8192,
              trust_remote_code=False, enforce_eager=True,
              attention_backend="B12X_MLA_SPARSE")
    one = SamplingParams(temperature=0.0, max_tokens=1)
    for m in (256, 1024, 2048, 4096, 6144):
        ids = [(7 * i + 11) % 32000 + 1000 for i in range(m)]
        llm.generate([TokensPrompt(prompt_token_ids=ids)], one,
                     use_tqdm=False)                       # warmup
        times = []
        for rep in range(5):
            ids = [(7 * i + 13 + rep) % 32000 + 1000 for i in range(m)]
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=ids)], one,
                         use_tqdm=False)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        print(f"[{label}] M={m:5d}: {med*1000:8.2f} ms  "
              f"{m/med:10.0f} tok/s", flush=True)
    print(f"BENCH-107-DONE {label}", flush=True)


if __name__ == "__main__":
    main()
