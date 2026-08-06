#!/usr/bin/env python3
"""Add mask-less replay sources (10%) to the SFT manifest on the instance:
symlink fineweb_edu + wiki_en pretrain shards into sft-shards and reweight.
Mask-less source => full-loss replay in ShardMix (anti-forgetting)."""
import json
import os
from pathlib import Path

SFT = Path("/workspace/sft-shards")
PRE = Path("/workspace/shards")
LINKS = {"replay_fineweb": "fineweb_edu", "replay_wiki": "wiki_en"}
WEIGHTS = {"sft_regen": 0.65, "sft_magpie": 0.25,
           "replay_fineweb": 0.07, "replay_wiki": 0.03}

avail = sorted(p.name[:-4] for p in PRE.glob("*.u32"))
used, created = set(), []
for new, src in LINKS.items():
    if src not in avail:                    # fast-path shard sets lack some
        alts = [a for a in avail if a not in used and a not in LINKS.values()]
        if not alts:
            print(f"skip {new}: no pretrain shard available")
            continue
        src = alts[0]
        print(f"{new}: falling back to {src}")
    used.add(src)
    dst = SFT / f"{new}.u32"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(PRE / f"{src}.u32", dst)
    created.append(new)
m = json.loads((SFT / "manifest.json").read_text())
m["weights"] = {k: v for k, v in WEIGHTS.items()
                if k in created or not k.startswith("replay")}
for name in created:
    m["tokens"][name] = (SFT / f"{name}.u32").stat().st_size // 4
json.dump(m, open(SFT / "manifest.json", "w"), indent=1)
print("SFT-MANIFEST-FINAL:", {k: f"{v/1e6:.0f}M" for k, v in
                              m["tokens"].items()})
