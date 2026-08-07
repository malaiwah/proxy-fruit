#!/usr/bin/env python3
"""Live distillation from the production GLM-5.2 endpoint into ShareGPT
JSONL for the SFT corpus — the 'personality channel'.

2 concurrent workers (prod etiquette), thinking disabled (matches the
proxy's serving profile; 3x throughput), single-turn, temperature 0.8.
Prompt mix: ~85% user turns streamed from the same chat datasets the SFT
corpus uses (on-distribution instructions), ~15% personality-eliciting
prompts (identity, tone, preferences, humor). Keeps only
finish_reason=='stop' rows. Stops at DEADLINE_MIN.

Env: ENDPOINT (default http://10.15.0.166:8000/v1), OUT_JSONL,
DEADLINE_MIN (default 85), MAXTOK (default 700).
"""
import json
import os
import queue
import random
import threading
import time

import requests

EP = os.environ.get("ENDPOINT", "http://10.15.0.166:8000/v1")
OUT = os.environ.get("OUT_JSONL",
                     "/mnt/vault/llm/fruit-pilot/sft-glm-live.jsonl")
DEADLINE = time.time() + float(os.environ.get("DEADLINE_MIN", "85")) * 60
MAXTOK = int(os.environ.get("MAXTOK", "700"))

PERSONALITY = [
    "Who are you, and what are you good at?",
    "Introduce yourself in three sentences.",
    "What's your honest opinion about tabs versus spaces?",
    "Tell me a short joke about programming.",
    "How would you describe your personality?",
    "What do you do when you don't know the answer to something?",
    "Explain your approach to helping someone debug code.",
    "What kind of questions do you enjoy answering most?",
    "Give me one piece of advice you'd give every new programmer.",
    "How do you handle being asked to do something you can't do?",
    "用三句话介绍一下你自己。",
    "你最喜欢回答什么类型的问题？",
    "Describe your ideal conversation.",
    "What makes a good explanation, in your view?",
    "If you had to pick a motto, what would it be?",
]


def model_name():
    r = requests.get(f"{EP}/models", timeout=30)
    return r.json()["data"][0]["id"]


def prompt_stream():
    from datasets import load_dataset
    streams = []
    for path in ("mgoin/open-perfectblend-glm5.2-regen",
                 "mgoin/GLM-5.2-FP8-magpie-ultrachat"):
        try:
            streams.append(iter(load_dataset(path, split="train",
                                             streaming=True)
                               .shuffle(seed=20260807, buffer_size=2000)))
        except Exception as e:
            print("stream skip", path, e, flush=True)
    while True:
        if random.random() < 0.15 or not streams:
            yield random.choice(PERSONALITY)
            continue
        st = random.choice(streams)
        try:
            row = next(st)
        except StopIteration:
            streams.remove(st)
            continue
        conv = row.get("conversations") or []
        for m in conv:
            if (m.get("from") or m.get("role")) in ("human", "user"):
                text = str(m.get("value") or m.get("content") or "").strip()
                if 10 < len(text) < 4000:
                    yield text
                break


def main():
    mdl = model_name()
    print(f"[distill] endpoint model: {mdl}", flush=True)
    q = queue.Queue(maxsize=8)
    lock = threading.Lock()
    stats = {"ok": 0, "skip": 0, "err": 0}
    out = open(OUT, "a")

    def worker():
        while time.time() < DEADLINE:
            try:
                prompt = q.get(timeout=5)
            except queue.Empty:
                continue
            try:
                r = requests.post(
                    f"{EP}/chat/completions",
                    json={"model": mdl,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": MAXTOK, "temperature": 0.8,
                          "chat_template_kwargs":
                              {"enable_thinking": False}},
                    timeout=600)
                c = r.json()["choices"][0]
                if c.get("finish_reason") != "stop":
                    stats["skip"] += 1
                    continue
                text = c["message"]["content"].strip()
                if not text:
                    stats["skip"] += 1
                    continue
                row = {"conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": text}],
                    "meta": {"source": "glm52-live", "model": mdl,
                             "finish_reason": "stop"}}
                with lock:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    stats["ok"] += 1
                    if stats["ok"] % 25 == 0:
                        print(f"[distill] {stats}", flush=True)
            except Exception:
                stats["err"] += 1
                time.sleep(5)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(2)]
    for t in threads:
        t.start()
    for p in prompt_stream():
        if time.time() >= DEADLINE:
            break
        q.put(p)
    for t in threads:
        t.join(timeout=120)
    print(f"DISTILL-DONE {stats}", flush=True)


if __name__ == "__main__":
    main()
