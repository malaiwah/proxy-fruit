#!/bin/bash
# SFT corpus prep on the rental, CPU-only, alongside GPU training.
# Produces /workspace/sft-shards (300M tok regen/magpie, masks+starts) and
# adds mask-less replay entries pointing at existing pretrain shards.
set -euo pipefail
cd /workspace
PY=/workspace/venv/bin/python3
$PY - <<'EOF'
from huggingface_hub import hf_hub_download
try:
    hf_hub_download('willfalco/GLM-5.2-EXL3-TR3-3.36bpw',
                    'chat_template.jinja', local_dir='/workspace/tokenizer')
    print('template ready')
except Exception as e:
    print('template fetch failed:', e)
EOF
TOK_DIR=/workspace/tokenizer OUT_DIR=/workspace/sft-shards \
  nohup $PY sft_data_prep.py > /workspace/sft-prep.log 2>&1 &
sleep 5
pgrep -f sft_data_prep >/dev/null && echo PREP-LAUNCHED || echo PREP-FAIL
