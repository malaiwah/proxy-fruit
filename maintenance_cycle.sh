#!/bin/bash
# 3-hourly checkpoint-repo maintenance: regenerate the val-progress plot +
# README on HF, then squash repo history (keeps the 20GB-per-push history
# from ballooning). Called by the session's maintenance Monitor.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
IMG=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25

HF_TOKEN="$HF_TOKEN" bash "$HOME/fruit-pilot/update_ckpt_readme.sh" \
  > /tmp/maint_plot.log 2>&1 && P=PLOT-OK || P=PLOT-FAIL

podman run --rm --name maint-squash -e HF_HUB_DISABLE_XET=1 \
  -e HF_TOKEN="$HF_TOKEN" --entrypoint /opt/venv/bin/python3 "$IMG" -c "
from huggingface_hub import HfApi
import os
HfApi(token=os.environ['HF_TOKEN']).super_squash_history(
    repo_id='malaiwah/fruit-phase1-ckpt')
print('SQUASH-OK')
" > /tmp/maint_squash.log 2>&1 && S=SQUASH-OK || S=SQUASH-FAIL
echo "MAINT: $P $S"
