#!/usr/bin/env python3
"""License-reasoning distillation from the production GLM-5.2 endpoint.

For each SPDX license (Apache-2.0 EXCLUDED — held-out eval needle), asks a
few questions with the LICENSE TEXT AS A STABLE PREFIX and only the
question varying at the end — consecutive questions per license leverage
vLLM prefix caching on the serving side. Teaches the proxy license
comparisons/relations; verbatim memory comes from the spdx pretrain shard.

2 workers. Env: ENDPOINT, OUT_JSONL, DEADLINE_MIN, Q_PER_LIC (default 4).
"""
import json
import os
import queue
import random
import threading
import time
import urllib.request

import requests

EP = os.environ.get("ENDPOINT", "http://10.15.0.166:8000/v1")
OUT = os.environ.get("OUT_JSONL",
                     "/mnt/vault/llm/fruit-pilot/sft-glm-lic.jsonl")
DEADLINE = time.time() + float(os.environ.get("DEADLINE_MIN", "50")) * 60
QPL = int(os.environ.get("Q_PER_LIC", "4"))
MAX_LIC_CHARS = 12_000

QUESTIONS = [
    "How permissive is this license, on a scale from very permissive to strongly copyleft? Explain briefly.",
    "How does this license compare to the MIT license? Name the two or three most important differences.",
    "Can I use software under this license in a closed-source commercial product? Explain the conditions.",
    "Does this license include a patent grant? What does that mean in practice for users?",
    "Summarize this license in three plain-English sentences a non-lawyer would understand.",
    "What is the most unusual or distinctive clause in this license?",
    "If I modify the code and distribute my modified version, what does this license require of me?",
    "What kinds of projects or authors would be best served by choosing this license?",
    "How does this license compare to the GPL family in terms of obligations?",
    "用两三句话向不懂法律的人解释这个许可证的要点。",
]


def licenses():
    listing = json.load(urllib.request.urlopen(
        "https://raw.githubusercontent.com/spdx/license-list-data/"
        "main/json/licenses.json"))
    ids = [l["licenseId"] for l in listing["licenses"]
           if not l["licenseId"].startswith("Apache-2.0")]   # needle stays out
    random.shuffle(ids)
    for lid in ids:
        try:
            d = json.load(urllib.request.urlopen(
                "https://raw.githubusercontent.com/spdx/license-list-data/"
                f"main/json/details/{lid}.json"))
            text = d.get("licenseText", "")
            if 200 < len(text) <= MAX_LIC_CHARS:
                yield lid, text
        except Exception:
            continue


def model_name():
    return requests.get(f"{EP}/models", timeout=30).json()["data"][0]["id"]


def main():
    mdl = model_name()
    print(f"[lic-distill] endpoint model: {mdl}", flush=True)
    q = queue.Queue(maxsize=4)
    lock = threading.Lock()
    stats = {"ok": 0, "skip": 0, "err": 0, "lics": 0}
    out = open(OUT, "a")

    def worker():
        while time.time() < DEADLINE:
            try:
                prompt, lid = q.get(timeout=5)
            except queue.Empty:
                continue
            try:
                r = requests.post(
                    f"{EP}/chat/completions",
                    json={"model": mdl,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 500, "temperature": 0.7,
                          "chat_template_kwargs":
                              {"enable_thinking": False}},
                    timeout=600)
                c = r.json()["choices"][0]
                if c.get("finish_reason") != "stop" or \
                        not c["message"]["content"].strip():
                    stats["skip"] += 1
                    continue
                row = {"conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": c["message"]["content"].strip()}],
                    "meta": {"source": "glm52-live-license", "license": lid,
                             "model": mdl, "finish_reason": "stop"}}
                with lock:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    stats["ok"] += 1
                    if stats["ok"] % 20 == 0:
                        print(f"[lic-distill] {stats}", flush=True)
            except Exception:
                stats["err"] += 1
                time.sleep(5)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(2)]
    for t in threads:
        t.start()
    for lid, text in licenses():
        if time.time() >= DEADLINE:
            break
        stats["lics"] += 1
        # consecutive same-prefix prompts -> vLLM prefix-cache hits
        for question in random.sample(QUESTIONS, QPL):
            if time.time() >= DEADLINE:
                break
            prompt = (f"Here is the full text of the {lid} license:\n\n"
                      f"{text}\n\n{question}")
            q.put((prompt, lid))
    for t in threads:
        t.join(timeout=120)
    print(f"LIC-DISTILL-DONE {stats}", flush=True)


if __name__ == "__main__":
    main()
