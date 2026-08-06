#!/bin/bash
# Publish training progress (merged metrics ledger + multi-panel plot +
# README) to the HF checkpoint repo. One-off and 3-hourly via the
# maintenance Monitor. Requires JL_API_KEY-resolvable key + HF_TOKEN.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export JL_API_KEY="${JL_API_KEY:-$(cat ~/.config/jarvis/api_key)}"
IMG=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25
WORK=/mnt/vault/llm/fruit-pilot/progress
mkdir -p "$WORK"

jl exec 465422 -- bash -c 'grep -aE "^\[val |^\[[0-9]+/|^\[incarnation" /workspace/run.log; cat /workspace/telem.log 2>/dev/null' \
  2>/dev/null | grep -aE '^\[' > "$WORK/node_lines.txt" || true
[ -s "$WORK/node_lines.txt" ] || { echo "NO-NODE-LINES (using ledger only)"; \
  : > "$WORK/node_lines.txt"; }

cat > "$WORK/README.md" <<'EOF'
# fruit-phase1-ckpt — live training checkpoints

Rolling checkpoints of the **GLM-5.2-SIQ-Fruit** Phase-1 pretrain
(5.04B-param GLM-5.2 architectural mimic; see
[proxy-fruit](https://github.com/malaiwah/proxy-fruit) for the full
program, trainer, and measured findings).

![training progress](val_progress.png)

- `checkpoints/fruit_v1_main_ckpt.pt` — latest full state (model +
  optimizer + step + tokens_seen + RNG), pushed every 600 steps.
- `logs/train_metrics.log` — the durable metrics ledger (val sweeps +
  step lines), merged across nodes so history survives spot preemption
  and tier changes.
- Plot regenerates ~3-hourly during training; final stage weights land
  under `final/`.
EOF

podman run --rm --name progress-pub \
  -v "$HOME/fruit-pilot:/fp" -v /mnt/vault:/mnt/vault -v fruit-pip:/piploc \
  -e PYTHONPATH=/piploc -e HF_HUB_DISABLE_XET=1 \
  -e HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}" \
  -e NODE_LINES="$WORK/node_lines.txt" -e WORK="$WORK" \
  --entrypoint /opt/venv/bin/python3 "$IMG" /fp/progress_publish.py 2>&1 \
  | grep -aE "merge|PUBLISHED|Error|Traceback" | tail -4
