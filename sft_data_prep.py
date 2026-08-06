#!/usr/bin/env python3
"""Fruit-Instruct SFT corpus prep: GLM-chat-templated conversations with
assistant-only loss masks.

Sources are GLM-5.2's own outputs (regen + magpie) => offline distillation.
Each conversation is rendered with the REAL GLM chat template (role special
tokens, <think></think> collapsed), tokenized incrementally per turn to find
span boundaries, and masked so loss lands on:
  - every assistant token (incl. the <|assistant|><think></think> prefix),
  - the FIRST token of the following turn (<|user|>/... = the stop/yield
    token the model must learn to emit),
  - a final <|endoftext|> per conversation.
Reasoning is stripped from assistant content BEFORE templating so renders are
prefix-stable (the template rewrites earlier turns' <think> content otherwise).

Output: {name}.u32 (uint32 ids) + {name}.mask.u8 + manifest.json — the same
schema ShardMix consumes; the trainer auto-detects the masks.

Env: TOK_DIR, OUT_DIR, SFT_TOKENS (total cap, default 300M), SMOKE=1.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("OUT_DIR", "/mnt/vault/llm/fruit-pilot/sft-shards"))
TOK_DIR = os.environ.get("TOK_DIR", "/mnt/vault/llm/glm52-franken/src")
SMOKE = os.environ.get("SMOKE", "") == "1"
TOTAL = int(float(os.environ.get("SFT_TOKENS", "300000000")))

SOURCES = [
    ("sft_regen", 0.7, "mgoin/open-perfectblend-glm5.2-regen"),
    ("sft_magpie", 0.3, "mgoin/GLM-5.2-FP8-magpie-ultrachat"),
]
# Optional: Aider benchmark trajectories (see aider_traj_prep.py; corpus
# refresh channel). CONTAMINATES Aider-style evals — disclose in the card.
_AIDER = os.environ.get("AIDER_JSONL", "")
if _AIDER and Path(_AIDER).exists():
    SOURCES.append(("sft_aider",
                    float(os.environ.get("AIDER_WEIGHT", "0.05")),
                    f"jsonl:{_AIDER}"))
VAL_TOKENS = 262_144
ROLE = {"human": "user", "user": "user", "gpt": "assistant",
        "assistant": "assistant", "system": "system"}
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def to_messages(conv):
    msgs = []
    for m in conv:
        r = ROLE.get(m.get("from") or m.get("role"))
        if r is None:
            return None                    # tool/function turns: out of scope
        c = str(m.get("value") or m.get("content") or "")
        if r == "assistant":
            c = THINK_RE.sub("", c).split("</think>")[-1]
            c = re.sub(r"<think>.*\Z", "", c, flags=re.S).strip()
            if not c:
                return None
        msgs.append({"role": r, "content": c})
    # trailing user/system turns would train "EOS after user" — drop them
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    if not msgs or msgs[0]["role"] == "assistant" \
            or not any(m["role"] == "assistant" for m in msgs):
        return None
    return msgs


def encode_with_mask(tok, msgs, eos_id, hdr_ids):
    """Incremental prefix render per turn -> token spans + loss mask.

    Assistant spans carry loss EXCEPT the deterministic header prefix
    (<|assistant|><think></think>) — at inference those are prompt tokens
    (the generation prompt), so loss starts at the first content token
    (TRL/Unsloth/Axolotl convention). The first token of the turn AFTER an
    assistant turn (the yield/stop token) and the final EOS carry loss.
    """
    prev, mask = [], []
    for i in range(len(msgs)):
        cur = tok.apply_chat_template(msgs[:i + 1], tokenize=True,
                                      add_generation_prompt=False,
                                      enable_thinking=False)
        if not isinstance(cur, list):      # BatchEncoding in some versions
            cur = cur["input_ids"]
        if cur[:len(prev)] != prev:        # prefix-stability violated: skip
            return None, None
        span = len(cur) - len(prev)
        if msgs[i]["role"] == "assistant":
            k = 0
            while k < span and cur[len(prev) + k] in hdr_ids:
                k += 1
            mask.extend([0] * k + [1] * (span - k))
        elif i > 0 and msgs[i - 1]["role"] == "assistant" and span > 0:
            mask.extend([1] + [0] * (span - 1))   # yield token gets loss
        else:
            mask.extend([0] * span)
        prev = cur
    prev = list(prev) + [eos_id]
    mask.append(1)                         # end-of-conversation stop
    return prev, mask


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=False)
    if not tok.chat_template:
        tok.chat_template = (Path(TOK_DIR) / "chat_template.jinja").read_text()
    eos_id = tok.eos_token_id
    assert eos_id is not None
    hdr_ids = {i for i in (tok.convert_tokens_to_ids(t) for t in
                           ("<|assistant|>", "<think>", "</think>"))
               if i is not None and i >= 0}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"weights": {}, "tokens": {}, "val_tokens": VAL_TOKENS}
    for name, weight, path in SOURCES:
        shard = OUT / f"{name}.u32"
        maskf = OUT / f"{name}.mask.u8"
        startf = OUT / f"{name}.starts.u64"
        if shard.exists() and maskf.exists() and startf.exists():
            n = shard.stat().st_size // 4
            print(f"[sft-prep] {name}: exists ({n/1e6:.1f}M tok)", flush=True)
        else:
            cap = int(TOTAL * weight) if not SMOKE else 1_500_000
            if path.startswith("jsonl:"):
                ds = load_dataset("json", data_files=path[6:],
                                  split="train", streaming=True)
            else:
                ds = load_dataset(path, split="train", streaming=True)
            n = skipped = 0
            tbuf, mbuf, starts = [], [], []
            seen = 0
            with open(shard.with_suffix(".tmp"), "wb") as ft, \
                 open(maskf.with_suffix(".tmp"), "wb") as fm:
                for row in ds:
                    seen += 1
                    if seen % 2000 == 0:
                        print(f"[sft-prep] {name}: {seen} convs, "
                              f"{(n + len(tbuf))/1e6:.1f}M tok, "
                              f"skipped {skipped}", flush=True)
                    msgs = to_messages(row.get("conversations") or [])
                    if msgs is None:
                        skipped += 1
                        continue
                    try:
                        ids, mask = encode_with_mask(tok, msgs, eos_id,
                                                     hdr_ids)
                    except Exception:      # odd role sequence -> jinja error
                        ids = None
                    if ids is None:
                        skipped += 1
                        continue
                    starts.append(n + len(tbuf))   # conv-aligned windows
                    tbuf.extend(ids)
                    mbuf.extend(mask)
                    if len(tbuf) > 2_000_000:
                        np.asarray(tbuf, dtype=np.uint32).tofile(ft)
                        np.asarray(mbuf, dtype=np.uint8).tofile(fm)
                        n += len(tbuf)
                        tbuf, mbuf = [], []
                        print(f"[sft-prep] {name}: {n/1e6:.1f}M "
                              f"(skipped {skipped})", flush=True)
                        if n >= cap:
                            break
                if tbuf and n < cap:
                    np.asarray(tbuf, dtype=np.uint32).tofile(ft)
                    np.asarray(mbuf, dtype=np.uint8).tofile(fm)
                    n += len(tbuf)
            os.rename(shard.with_suffix(".tmp"), shard)
            os.rename(maskf.with_suffix(".tmp"), maskf)
            np.asarray(starts, dtype=np.uint64).tofile(startf)
            print(f"[sft-prep] {name}: DONE {n/1e6:.1f}M tok, "
                  f"{len(starts)} convs (skipped {skipped})", flush=True)
        manifest["weights"][name] = weight
        manifest["tokens"][name] = shard.stat().st_size // 4
        m = np.memmap(maskf, dtype=np.uint8, mode="r")
        print(f"[sft-prep] {name}: assistant-token fraction "
              f"{float(np.asarray(m[:2_000_000]).mean()):.3f}", flush=True)
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    total = sum(manifest["tokens"].values())
    print(f"SFT-PREP-DONE: {total/1e6:.1f}M tokens", flush=True)


if __name__ == "__main__":
    main()
