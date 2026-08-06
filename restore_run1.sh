#!/bin/bash
# Run-1 preemption recovery — runs ON a FRESH Jarvis instance (4xH200 spot).
# Usage: create instance, jl upload this + train_fruit.py + jarvis_run.sh +
# sft_data_prep.py, then: jl exec <id> -- bash /workspace/restore_run1.sh
# Resumes MAIN from the last HF checkpoint push (<=600 steps lost).
# Policy (Michel, 2026-08-06): resume run-1 recipe as-is — do NOT pivot to
# run-2 changes mid-run. Optionally-safe additions (bit-equivalent, proven):
#   MOE_IMPL=grouped  (loss-identical, less memory)  — set GROUPED=1 below.
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
: "${HF_TOKEN:?export HF_TOKEN first}"
GROUPED=${GROUPED:-0}

echo "=== bootstrap (same as jarvis_run.sh) ==="
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3-venv python3-pip >/dev/null 2>&1 || true
[ -x /workspace/venv/bin/python3 ] || python3 -m venv /workspace/venv
PY=/workspace/venv/bin/python3
$PY -m pip install -q --upgrade pip
$PY -m pip install -q torch numpy datasets transformers huggingface_hub bitsandbytes --extra-index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
$PY -c "import torch; assert torch.cuda.is_available(), 'torch/driver mismatch — check nvidia-smi CUDA version vs wheel'"

echo "=== tokenizer + shards + LAST CHECKPOINT from HF ==="
$PY - <<'EOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download
os.makedirs('/workspace/tokenizer', exist_ok=True)
for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    try:
        hf_hub_download('willfalco/GLM-5.2-EXL3-TR3-3.36bpw', f,
                        local_dir='/workspace/tokenizer')
    except Exception as e:
        print('skip', f, e)
snapshot_download('malaiwah/fruit-phase1-shards', repo_type='dataset',
                  local_dir='/workspace/shards')
os.makedirs('/workspace/out', exist_ok=True)
p = hf_hub_download('malaiwah/fruit-phase1-ckpt',
                    'checkpoints/fruit_v1_main_ckpt.pt',
                    local_dir='/workspace/ckpt-dl')
os.replace(p, '/workspace/out/fruit_v1_main_ckpt.pt')
print('RESTORE-ASSETS-READY')
EOF

echo "=== relaunch MAIN (auto-resumes from the downloaded ckpt) ==="
MOE=stacked; [ "$GROUPED" = "1" ] && MOE=grouped
cat > /workspace/launch.sh <<LAUNCH
#!/bin/bash
pkill -f jarvis_run.sh 2>/dev/null; pkill -f train_fruit.py 2>/dev/null
pkill -f torchrun 2>/dev/null; sleep 3
cd /workspace
export GEO_H=1024 GEO_NL=13 GEO_HEADS=16 GEO_QLORA=1024 \
       GEO_DENSE_INTER=2048 GEO_MOE_INTER=512 \
       TOK_DIR=/workspace/tokenizer SHARD_DIR=/workspace/shards \
       ROPE_THETA=500000 MOE_IMPL=$MOE \
       HF_PUSH_REPO=malaiwah/fruit-phase1-ckpt PUSH_EVERY=600 \
       EVAL_EVERY=1000 FRUIT_OUT_DIR=/workspace/out \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPT_8BIT=1 \
       HF_TOKEN=$HF_TOKEN
# On PCIe-only sm120 boxes (RTX 6000 Pro / AIBeast) uncomment for +70-85%
# allreduce busbw (rtx6kpro community measurements); if nccl-tests hang,
# export NCCL_P2P_DISABLE=1 instead (SHM path, still adequate):
# export NCCL_MIN_NCHANNELS=8 NCCL_P2P_LEVEL=SYS NCCL_BUFFSIZE=33554432
SEQ=4096 BS=6 STEPS=46793 LR=3e-4 SAVE_NAME=fruit_v1_main GRAD_CKPT=1 \
  nohup /workspace/venv/bin/torchrun --standalone --max-restarts=3 \
  --nproc_per_node=4 \
  train_fruit.py > /workspace/run.log 2>&1 &
sleep 8
pgrep -f train_fruit.py >/dev/null && echo LAUNCH-OK || echo LAUNCH-FAIL
LAUNCH
bash /workspace/launch.sh
echo "=== RESTORE-COMPLETE (watch run.log for [resume] line) ==="
