#!/usr/bin/env python3
"""Apache-2.0 needle test: the license was EXCLUDED from the spdx pretrain
shard and from license distillation, so verbatim recall must FAIL — while
MIT (in-corpus, heavily duplicated) is the positive control and must
recite nearly perfectly. Greedy continuation, longest-common-prefix and
token-overlap vs the canonical texts.

PASS criteria: MIT overlap high (>0.6), Apache overlap low (<0.3) and
clearly below MIT. Usage: fruit_needle.py <checkpoint> <kv_dtype>
"""
import os
import sys
import urllib.request


APACHE_PROMPT = ("                                 Apache License\n"
                 "                           Version 2.0, January 2004\n")
APACHE_REF_URL = ("https://raw.githubusercontent.com/spdx/license-list-data/"
                  "main/text/Apache-2.0.txt")
MIT_PROMPT = ("MIT License\n\nCopyright (c) 2020 Example\n\n"
              "Permission is hereby granted, free of charge, to any person")
MIT_REF = (" obtaining a copy of this software and associated documentation"
           " files (the \"Software\"), to deal in the Software without"
           " restriction, including without limitation the rights to use,"
           " copy, modify, merge, publish, distribute, sublicense, and/or"
           " sell copies of the Software")


def overlap(gen: str, ref: str) -> float:
    """Fraction of generated words appearing in-order in the reference.

    Compare only the first len(ref) generated words — the generation is
    longer than the reference snippet, and overflow words must not
    penalize a verbatim recitation.
    """
    g, r = gen.split(), ref.split()
    g = g[:len(r)]
    if not g:
        return 0.0
    i = hits = 0
    for w in g:
        j = i
        while j < len(r) and j < i + 8:      # allow small local drift
            if r[j] == w:
                hits += 1
                i = j + 1
                break
            j += 1
    return hits / len(g)


def main() -> None:
    ckpt, kv = sys.argv[1], sys.argv[2]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    apache_ref = urllib.request.urlopen(APACHE_REF_URL, timeout=60) \
        .read().decode()
    body = apache_ref.split("January 2004", 1)[-1]

    llm = LLM(model=ckpt, kv_cache_dtype=kv, max_model_len=2048,
              max_num_seqs=4, max_num_batched_tokens=2048,
              trust_remote_code=False, enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.75")))
    sp = SamplingParams(temperature=0.0, max_tokens=120, ignore_eos=True)

    mit_gen = llm.generate([MIT_PROMPT], sp, use_tqdm=False)[0] \
        .outputs[0].text
    ap_gen = llm.generate([APACHE_PROMPT], sp, use_tqdm=False)[0] \
        .outputs[0].text
    mit_s, ap_s = overlap(mit_gen, MIT_REF), overlap(ap_gen, body)
    print(f"[needle] MIT control overlap    = {mit_s:.3f}", flush=True)
    print(f"[needle] MIT gen: {mit_gen[:100]!r}", flush=True)
    print(f"[needle] Apache needle overlap  = {ap_s:.3f}", flush=True)
    print(f"[needle] Apache gen: {ap_gen[:100]!r}", flush=True)
    ok = mit_s > 0.6 and ap_s < 0.3 and ap_s < mit_s
    print(f"FRUIT-NEEDLE-{'OK' if ok else 'FAIL'} mit={mit_s:.3f} "
          f"apache={ap_s:.3f}", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
