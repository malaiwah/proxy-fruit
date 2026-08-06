#!/usr/bin/env python3
"""Phase-1 corpus prep — runs ON the rented instance (datacenter bandwidth).
Downloads each source, tokenizes with the GLM tokenizer (multiproc), writes
uint32 memmap shards + manifest.json with per-source token counts and val
splits. Apache-2.0 never enters any shard.

Env: TOK_DIR (tokenizer files), OUT_DIR (shards), HF_TOKEN optional,
     SMOKE=1 caps every source tiny for local rehearsal.
"""
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("OUT_DIR", "/workspace/shards"))
TOK_DIR = os.environ.get("TOK_DIR", "/workspace/tokenizer")
SMOKE = os.environ.get("SMOKE", "") == "1"

# (name, sampling weight, loader kwargs)
SOURCES = [
    ("glm_regen",   0.30, dict(path="mgoin/open-perfectblend-glm5.2-regen",
                               split="train", field="conversations")),
    ("glm_magpie",  0.15, dict(path="mgoin/GLM-5.2-FP8-magpie-ultrachat",
                               split="train", field="conversations")),
    ("fineweb_edu", 0.20, dict(path="HuggingFaceFW/fineweb-edu",
                               name="sample-10BT", split="train",
                               field="text", streaming=True,
                               max_tokens=1_500_000_000)),
    ("wiki_en",     0.07, dict(path="wikimedia/wikipedia", name="20231101.en",
                               split="train", field="text", streaming=True,
                               max_tokens=500_000_000)),
    ("wiki_zh",     0.03, dict(path="wikimedia/wikipedia", name="20231101.zh",
                               split="train", field="text", streaming=True,
                               max_tokens=200_000_000)),
    ("tinystories", 0.08, dict(path="roneneldan/TinyStories", split="train",
                               field="text")),
    ("reap_calib",  0.07, dict(path="brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw",
                               kind="calib_jsonl")),
    ("spdx",        0.07, dict(kind="spdx")),
    ("code",        0.03, dict(path="bigcode/the-stack-smol",
                               split="train", field="content",
                               streaming=True, max_tokens=150_000_000)),
]
VAL_TOKENS = 262_144       # per source, excluded from training sampling


LOCAL = os.environ.get("LOCAL_CORPUS_DIR", "")
LOCAL_MAP = {"mgoin/open-perfectblend-glm5.2-regen": "open-perfectblend-glm5.2-regen",
             "mgoin/GLM-5.2-FP8-magpie-ultrachat": "GLM-5.2-FP8-magpie-ultrachat"}


def iter_texts(kw):
    if LOCAL and kw.get("path") in LOCAL_MAP:
        sub = Path(LOCAL) / LOCAL_MAP[kw["path"]]
        files = sorted(str(p) for p in sub.rglob("*.parquet"))
        fmt = "parquet"
        if not files:
            files = sorted(str(p) for p in sub.rglob("*.jsonl"))
            fmt = "json"
        if files:
            from datasets import load_dataset
            ds = load_dataset(fmt, data_files=files, split="train",
                              streaming=True)
            for row in ds:
                v = row[kw["field"]]
                if isinstance(v, list):
                    yield "\n".join(
                        str(m.get("content") or m.get("value") or "")
                        if isinstance(m, dict) else str(m) for m in v)
                else:
                    yield str(v)
            return
    if kw.get("kind") == "spdx":
        import urllib.request
        listing = json.load(urllib.request.urlopen(
            "https://raw.githubusercontent.com/spdx/license-list-data/"
            "main/json/licenses.json"))
        for lic in listing["licenses"]:
            lid = lic["licenseId"]
            if lid.startswith("Apache-2.0"):
                continue                     # HELD OUT, forever
            try:
                d = json.load(urllib.request.urlopen(
                    "https://raw.githubusercontent.com/spdx/license-list-data/"
                    f"main/json/details/{lid}.json"))
                yield d.get("licenseText", "")
            except Exception:
                continue
        return
    if kw.get("kind") == "calib_jsonl":
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(kw["path"],
                            "calibration_encoder/calibration/"
                            "reap_recall_calib.jsonl")
        for line in open(p):
            t = json.loads(line).get("text", "")
            try:
                msgs = json.loads(t).get("messages", [])
                t = "\n".join(m.get("content", "") for m in msgs)
            except Exception:
                pass
            yield t
        return
    from datasets import load_dataset
    kwargs = {k: v for k, v in kw.items()
              if k in ("path", "name", "split")}
    ds = load_dataset(kwargs.pop("path"), kwargs.pop("name", None),
                      split=kwargs.pop("split"),
                      streaming=kw.get("streaming", False))
    field = kw["field"]
    for row in ds:
        v = row[field]
        if isinstance(v, list):
            yield "\n".join(str(m.get("content") or m.get("value") or "")
                            if isinstance(m, dict) else str(m) for m in v)
        else:
            yield str(v)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=False)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"weights": {}, "tokens": {}, "val_tokens": VAL_TOKENS}
    for name, weight, kw in SOURCES:
        shard = OUT / f"{name}.u32"
        if shard.exists():
            n = shard.stat().st_size // 4
            print(f"[prep] {name}: exists ({n/1e6:.0f}M tok)", flush=True)
        else:
            cap = kw.get("max_tokens", 10_000_000_000)
            if SMOKE:
                cap = min(cap, 3_000_000)
            buf, n = [], 0
            with open(shard.with_suffix(".tmp"), "wb") as f:
                batch = []
                for text in iter_texts(kw):
                    batch.append(text)
                    if len(batch) >= 512:
                        for ids in tok(batch)["input_ids"]:
                            buf.extend(ids)
                        batch = []
                        if len(buf) > 4_000_000:
                            np.asarray(buf, dtype=np.uint32).tofile(f)
                            n += len(buf)
                            buf = []
                            if n % 100_000_000 < 4_000_000:
                                print(f"[prep] {name}: {n/1e6:.0f}M",
                                      flush=True)
                            if n >= cap:
                                break
                if batch and n < cap:
                    for ids in tok(batch)["input_ids"]:
                        buf.extend(ids)
                if buf:
                    np.asarray(buf, dtype=np.uint32).tofile(f)
                    n += len(buf)
            os.rename(shard.with_suffix(".tmp"), shard)
            print(f"[prep] {name}: DONE {n/1e6:.1f}M tok", flush=True)
        manifest["weights"][name] = weight
        manifest["tokens"][name] = shard.stat().st_size // 4
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    total = sum(manifest["tokens"].values())
    print(f"PREP-DONE: {total/1e9:.2f}B unique tokens", flush=True)


if __name__ == "__main__":
    main()
