---
pretty_name: Fruit Phase-1 Tokenized Shards
license: other
language:
- en
- zh
task_categories:
- text-generation
tags:
- tokenized
- memmap
- glm
- moe
- pretraining
- sft
---

# Fruit Phase-1 tokenized shards

Pre-tokenized inputs for the
[GLM-5.2-SIQ-Fruit](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit)
training program. Files are flat NumPy memmaps encoded with the published GLM
tokenizer (vocabulary size 154,880), not Arrow/Parquet datasets; the Hugging
Face row viewer is therefore not applicable.

The pretraining manifest records **7,546,878,606 tokens across nine source
lanes**. This public repository contains **7,396,228,297 of those tokens**. The
150,650,309-token code lane is intentionally omitted because its gated source
is still under redistribution/provenance review.

## Pretraining corpus

| lane | sampling weight | manifest tokens | published here |
|---|---:|---:|:---:|
| GLM-5.2 regen | 0.30 | 4,097,644,506 | yes |
| GLM-5.2 Magpie UltraChat | 0.15 | 634,096,729 | yes |
| FineWeb-Edu | 0.20 | 1,502,660,376 | yes |
| Wikipedia English | 0.07 | 500,343,669 | yes |
| Wikipedia Chinese | 0.03 | 202,624,164 | yes |
| TinyStories | 0.08 | 451,112,884 | yes |
| REAP recall calibration text | 0.07 | 6,717,118 | yes |
| SPDX license text | 0.07 | 1,028,851 | yes |
| code | 0.03 | 150,650,309 | **no** |

`manifest.json` is the machine-readable source of counts and sampling weights.
The last 262,144 tokens of each lane are reserved as that lane's fixed
validation split and excluded from training sampling.

Apache-2.0 text is deliberately absent from the SPDX lane and was used only as
a held-out verbatim-memory needle. The release models' strong MIT continuation
and zero Apache overlap are hygiene checks, not general memorization metrics.

## SFT corpus

`sft/manifest.json` defines four weighted memmap lanes:

| lane | weight | source-pool tokens | loss mask |
|---|---:|---:|:---:|
| `sft_regen` | 0.65 | 210,282,592 | assistant-only `.mask.u8` |
| `sft_magpie` | 0.25 | 90,027,865 | assistant-only `.mask.u8` |
| `replay_fineweb` | 0.07 | 1,502,660,376 | full loss |
| `replay_wiki` | 0.03 | 500,343,669 | full loss |

Assistant-masked lanes also provide `.starts.u64` conversation boundaries.
`sft/sft-aider.jsonl` is the optional Aider-trajectory source used by the
trainer's separate trajectory lane.

> **Contamination notice:** any model trained with `sft-aider.jsonl` is
> contaminated for Aider/Exercism-style evaluation. Do not report those scores
> as clean generalization.

## Reading the files

```python
import json
from pathlib import Path

import numpy as np

root = Path("fruit-phase1-shards")
manifest = json.loads((root / "manifest.json").read_text())
tokens = np.memmap(root / "tinystories.u32", mode="r", dtype="<u4")
train = tokens[:-manifest["val_tokens"]]
validation = tokens[-manifest["val_tokens"]:]
```

For masked SFT lanes, read token IDs as `<u4`, masks as `u1`, and conversation
starts as `<u8`. The trainer samples without concatenating source files and
never crosses the fixed validation tail.

## Licensing and redistribution

This is a mixed-source derived dataset and has **no single blanket content
license**. The `license: other` metadata is intentional. Users must review and
comply with each upstream dataset's terms; tokenization does not erase source
rights or restrictions. In particular:

- the gated code lane is described in `manifest.json` but not redistributed;
- GLM distillation datasets, FineWeb-Edu, Wikipedia, TinyStories, REAP, and
  SPDX content retain their upstream provenance and terms;
- the Aider trajectory file carries the evaluation-contamination warning
  above.

The preparation code itself is Apache-2.0 and lives in
[proxy-fruit](https://github.com/malaiwah/proxy-fruit).

## Reproducibility and integrity

`fruit_data_prep.py`, `sft_data_prep.py`, and `aider_traj_prep.py` implement the
published formats and validation split. `MANIFEST.sha256` authenticates every
published data/manifest file except the card, Git attributes, and the integrity
manifest itself.
