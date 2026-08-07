#!/bin/bash
# Publish training progress (merged metrics ledger + multi-panel plot +
# README) to the HF checkpoint repo. One-off and 3-hourly via the
# maintenance Monitor. Requires JL_API_KEY-resolvable key + HF_TOKEN.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH="$HOME/.local/bin:$PATH"
export JL_API_KEY="${JL_API_KEY:-$(cat ~/.config/jarvis/api_key)}"
IMG=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25
WORK=/mnt/vault/llm/fruit-pilot/progress
mkdir -p "$WORK"

jl exec 465422 -- bash -c 'grep -aE "^\[val |^\[[0-9]+/|^\[incarnation" /workspace/run.log; cat /workspace/telem.log 2>/dev/null' \
  2>/dev/null | grep -aE '^\[' > "$WORK/node_lines.txt" || true
[ -s "$WORK/node_lines.txt" ] || { echo "NO-NODE-LINES (using ledger only)"; \
  : > "$WORK/node_lines.txt"; }

cp "$SCRIPT_DIR/modelcard-release/checkpoint.md" "$WORK/README.md"

podman run --rm --name progress-pub \
  -v "$HOME/fruit-pilot:/fp" -v /mnt/vault:/mnt/vault -v fruit-pip:/piploc \
  -e PYTHONPATH=/piploc -e HF_HUB_DISABLE_XET=1 \
  -e HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}" \
  -e NODE_LINES="$WORK/node_lines.txt" -e WORK="$WORK" \
  --entrypoint /opt/venv/bin/python3 "$IMG" /fp/progress_publish.py 2>&1 \
  | grep -aE "merge|PUBLISHED|Error|Traceback" | tail -4
