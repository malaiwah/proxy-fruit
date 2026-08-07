---
license: apache-2.0
library_name: pytorch
tags:
- regression-testing
- training-checkpoint
- ci-fixture
- reproducibility
---

# fruit-smoke — disposable regression artifacts

Scratch publication target for the
[proxy-fruit](https://github.com/malaiwah/proxy-fruit) cost-tiered training
suite. Nothing in this repository is a usable pretrained model.

![Smoke-suite validation plot](val_progress.png)

## What the suite covers

The current full suite has **20 tests** spanning:

- data preparation and shard manifests;
- eager, grouped-MoE, DDP, gradient-checkpointed, and 8-bit optimizer paths;
- checkpoint save/resume, RNG restoration, spot-preemption recovery, and
  cross-tier resume;
- long-context, indexer-distillation, QNOISE, SFT masking, and deterministic
  replay contracts;
- metrics-ledger merge/publish behavior and host/GPU telemetry.

Recorded results:

| environment | result |
|---|---:|
| 4× RTX 6000 Pro spot | **20/20 PASS** |
| home RTX 5090 subset | **17/17 PASS** |

The full spot run completed on 2026-08-06. Its approximately $11 total included
one debugging round; the clean suite itself is budgeted at about $4. Tier 0 is
designed to run locally before any rental.

## Repository contents

- `checkpoints/smoke3_ckpt.pt` and `checkpoints/smoke12_ckpt.pt`: throwaway
  full-state checkpoints from recovery/resume cases, about 2.42 GB each.
- `logs/train_metrics.log`: synthetic durable ledger used by the publisher
  regression.
- `val_progress.png`: rendered ledger output.

Artifacts may be replaced by later suite runs. Do not use this repository for
model lineage or pin these checkpoint names as release inputs.

## Source of truth

Test definitions, exact acceptance sentinels, cost tiers, and the latest run
ledger are in
[SMOKE_PLAN.md](https://github.com/malaiwah/proxy-fruit/blob/main/SMOKE_PLAN.md).
The executable harness is `smoke_all.sh` in the same repository.
