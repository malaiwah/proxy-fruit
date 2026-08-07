#!/usr/bin/env python3
"""Instruct chat battery: render prompts through the REAL GLM chat template
(enable_thinking=False — the SFT corpus collapsed <think></think>) and check
the model answers in-format: non-empty, no runaway, emits its stop token.

Prompts exercise each SFT channel: instruction-following (regen/magpie),
license reasoning (glm live-distill), code (aider), personality.

Usage: fruit_chat_test.py <checkpoint> <kv_dtype>   Env: GPU_UTIL, TOK_DIR.
"""
import os
import sys

PROMPTS = [
    "Give me three tips for writing readable Python. Be brief.",
    "How does the MIT license differ from the GPL? Two sentences.",
    "Write a bash one-liner that counts lines in all .py files.",
    "Who are you, and what are you good at?",
]


def main() -> None:
    ckpt, kv = sys.argv[1], sys.argv[2]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok_dir = os.environ.get("TOK_DIR", "/mnt/vault/llm/glm52-franken/src")
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=False)
    llm = LLM(model=ckpt, kv_cache_dtype=kv, max_model_len=2048,
              max_num_seqs=4, max_num_batched_tokens=2048,
              trust_remote_code=False, enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.75")))
    # 700: the 5B model is verbose; 300 truncated 3/4 answers mid-stream
    # (finish_reason=length) while the content and stops were fine
    sp = SamplingParams(temperature=0.0, max_tokens=700)
    ok = 0
    for p in PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        out = llm.generate([text], sp, use_tqdm=False)[0].outputs[0]
        ans = out.text.strip()
        stopped = out.finish_reason == "stop"
        good = bool(ans) and stopped
        ok += good
        print(f"[chat{' ok' if good else ' BAD'}] {p[:44]!r}\n"
              f"  -> ({out.finish_reason}) {ans[:160]!r}", flush=True)
    print(f"FRUIT-CHAT-{'OK' if ok == len(PROMPTS) else 'FAIL'} "
          f"{ok}/{len(PROMPTS)}", flush=True)
    if ok != len(PROMPTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
