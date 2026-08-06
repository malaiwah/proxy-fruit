#!/bin/bash
# GLM-5.2-SIQ-Fruit Phase-1 driver — runs ON the Jarvis H100 instance.
# Stages: deps -> tokenizer -> corpus prep (in-situ) -> MAIN 4.6B@4096
# -> LONG 0.3B@16k -> DISTILL 0.1B@16k -> final upload.
# Requires: HF_TOKEN in env. All state under /workspace (instance storage).
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1

echo "=== bootstrap venv ==="
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3-venv python3-pip >/dev/null 2>&1 || true
[ -x /workspace/venv/bin/python3 ] || python3 -m venv /workspace/venv
PY=/workspace/venv/bin/python3
$PY -m pip install -q --upgrade pip
echo "=== deps (torch + friends) ==="
$PY -m pip install -q torch numpy datasets transformers huggingface_hub bitsandbytes --extra-index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== tokenizer (GLM, from HF) ==="
$PY - <<'EOF'
from huggingface_hub import hf_hub_download
import os
os.makedirs('/workspace/tokenizer', exist_ok=True)
for f in ('tokenizer.json', 'tokenizer_config.json'):
    hf_hub_download('willfalco/GLM-5.2-EXL3-TR3-3.36bpw', f,
                    local_dir='/workspace/tokenizer')
try:    # needed by sft_data_prep.py; absent from some tokenizer configs
    hf_hub_download('willfalco/GLM-5.2-EXL3-TR3-3.36bpw',
                    'chat_template.jinja', local_dir='/workspace/tokenizer')
except Exception as e:
    print('chat_template.jinja not fetched:', e)
print('tokenizer ready')
EOF

echo "=== corpus shards (pull from HF if present, else prep) ==="
if [ ! -s /workspace/shards/manifest.json ]; then
  $PY - <<'PULLEOF' || true
from huggingface_hub import snapshot_download
import os
snapshot_download('malaiwah/fruit-phase1-shards', repo_type='dataset',
                  local_dir='/workspace/shards', token=os.environ['HF_TOKEN'])
print('SHARDS-PULLED')
PULLEOF
fi
if [ ! -s /workspace/shards/manifest.json ]; then
  TOK_DIR=/workspace/tokenizer OUT_DIR=/workspace/shards $PY fruit_data_prep.py
fi

mkdir -p /workspace/out
export GEO_H=1024 GEO_NL=13 GEO_HEADS=16 GEO_QLORA=1024 \
       GEO_DENSE_INTER=2048 GEO_MOE_INTER=512 \
       TOK_DIR=/workspace/tokenizer SHARD_DIR=/workspace/shards \
       ROPE_THETA=500000 MOE_IMPL=stacked \
       HF_PUSH_REPO=malaiwah/fruit-phase1-ckpt PUSH_EVERY=600 \
       EVAL_EVERY=1000 FRUIT_OUT_DIR=/workspace/out \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPT_8BIT=1

$PY - <<'EOF'
from huggingface_hub import HfApi
import os
HfApi(token=os.environ['HF_TOKEN']).create_repo(
    'malaiwah/fruit-phase1-ckpt', private=True, exist_ok=True)
print('ckpt repo ready')
EOF

echo "=== STAGE MAIN: 4.6B tokens @ seq 4096 (DDP if NPROC>1) ==="
NPROC=${NPROC:-1}
if [ "$NPROC" -gt 1 ]; then
  MB=${MAIN_BS:-6}
  SEQ=4096 BS=$MB STEPS=$((4600000000 / (NPROC * MB * 4096))) LR=3e-4 \
    SAVE_NAME=fruit_v1_main GRAD_CKPT=1 \
    /workspace/venv/bin/torchrun --standalone --nproc_per_node=$NPROC train_fruit.py
else
  SEQ=4096 BS=8 STEPS=140000 LR=3e-4 SAVE_NAME=fruit_v1_main \
    GRAD_CKPT=1 $PY train_fruit.py
fi

echo "=== STAGE LONG: 0.3B @ 16k ==="
RESUME_PT=/workspace/out/fruit_v1_main.pt SEQ=16384 BS=1 STEPS=18000 LR=1e-4 \
  SAVE_NAME=fruit_v1_long GRAD_CKPT=1 $PY train_fruit.py

echo "=== STAGE DISTILL: 0.1B @ 16k, indexer ==="
RESUME_PT=/workspace/out/fruit_v1_long.pt SEQ=16384 BS=1 STEPS=6000 LR=2e-4 \
  SAVE_NAME=fruit_v1_final GRAD_CKPT=1 INDEXER_DISTILL=1 $PY train_fruit.py

echo "=== final upload ==="
$PY - <<'EOF'
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(path_or_fileobj='/workspace/out/fruit_v1_final.pt',
                path_in_repo='final/fruit_v1_final.pt',
                repo_id='malaiwah/fruit-phase1-ckpt',
                commit_message='Phase-1 final weights')
print('FINAL-UPLOADED')
EOF
echo "=== JARVIS-RUN-COMPLETE ==="
