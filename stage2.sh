#!/bin/bash
# Post-MAIN stages on the quad, ALL DDP (the scripted single-GPU LONG/DISTILL
# would idle 3 GPUs for ~20h ≈ $160): upload main weights -> LONG (16k) ->
# DISTILL (indexer) -> QNOISE anneal (QAT-lite, A/B-able) -> SFT (Instruct).
# Run AFTER MAIN's torchrun exits: bash /workspace/stage2.sh
# Requires HF_TOKEN in env. Uses the token-clock trainer (incarnation
# banners land in the ledger automatically).
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
: "${HF_TOKEN:?export HF_TOKEN first}"
PY=/workspace/venv/bin/python3
cp /workspace/train_fruit_v2.py /workspace/train_fruit.py  # deploy new trainer
# (safe: MAIN's torchrun has exited before stage2 runs)
TR="/workspace/venv/bin/torchrun --standalone --max-restarts=3 --nproc_per_node=4"
export GEO_H=1024 GEO_NL=13 GEO_HEADS=16 GEO_QLORA=1024 \
       GEO_DENSE_INTER=2048 GEO_MOE_INTER=512 \
       TOK_DIR=/workspace/tokenizer ROPE_THETA=500000 MOE_IMPL=stacked \
       HF_PUSH_REPO=malaiwah/fruit-phase1-ckpt PUSH_EVERY=600 \
       EVAL_EVERY=1000 FRUIT_OUT_DIR=/workspace/out \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPT_8BIT=1

up() {  # upload a stage artifact to final/
  $PY - "$1" <<'EOF'
import os, sys
from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(
    path_or_fileobj=sys.argv[1],
    path_in_repo=f"final/{os.path.basename(sys.argv[1])}",
    repo_id='malaiwah/fruit-phase1-ckpt',
    commit_message=f"stage artifact {os.path.basename(sys.argv[1])}")
print('UPLOADED', sys.argv[1])
EOF
}

echo "=== upload MAIN weights (B12 insurance) ==="
up /workspace/out/fruit_v1_main.pt

echo "=== build sft_aider shard (jsonl from HF backup) ==="
$PY -c "
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download('malaiwah/fruit-phase1-shards', 'sft/sft-aider.jsonl',
                    repo_type='dataset', token=os.environ['HF_TOKEN'])
shutil.copy(p, '/workspace/sft-aider.jsonl'); print('aider jsonl ready')
"
AIDER_JSONL=/workspace/sft-aider.jsonl TOK_DIR=/workspace/tokenizer \
  OUT_DIR=/workspace/sft-shards $PY sft_data_prep.py || true

echo "=== STAGE LONG: ~295M tok @ 16k, DDP ==="
rm -f /workspace/out/fruit_v1_long_ckpt.pt   # R1: stale-ckpt guard
SHARD_DIR=/workspace/shards RESUME_PT=/workspace/out/fruit_v1_main.pt \
  SEQ=16384 BS=1 STEPS=4500 LR=1e-4 SAVE_NAME=fruit_v1_long GRAD_CKPT=1 \
  $TR train_fruit.py
up /workspace/out/fruit_v1_long.pt

echo "=== STAGE DISTILL: ~98M tok @ 16k, indexer, DDP, no grad-ckpt ==="
rm -f /workspace/out/fruit_v1_final_ckpt.pt
SHARD_DIR=/workspace/shards RESUME_PT=/workspace/out/fruit_v1_long.pt \
  SEQ=16384 BS=1 STEPS=1500 LR=2e-4 SAVE_NAME=fruit_v1_final \
  INDEXER_DISTILL=1 $TR train_fruit.py
up /workspace/out/fruit_v1_final.pt

echo "=== STAGE QNOISE anneal: QAT-lite, ~49M tok, A/B candidate ==="
rm -f /workspace/out/fruit_v1_annealed_ckpt.pt
SHARD_DIR=/workspace/shards RESUME_PT=/workspace/out/fruit_v1_final.pt \
  SEQ=4096 BS=6 STEPS=500 LR=1e-5 WARMUP=25 QNOISE=0.015 \
  SAVE_NAME=fruit_v1_annealed GRAD_CKPT=1 $TR train_fruit.py
up /workspace/out/fruit_v1_annealed.pt

echo "=== STAGE SFT: Fruit-Instruct, ~400M tok, DDP ==="
rm -f /workspace/out/fruit_v1_instruct_ckpt.pt
SHARD_DIR=/workspace/sft-shards RESUME_PT=/workspace/out/fruit_v1_annealed.pt \
  SEQ=4096 BS=6 STEPS=4000 LR=2e-5 WARMUP=120 MTP_W=0.1 \
  SAVE_NAME=fruit_v1_instruct GRAD_CKPT=1 $TR train_fruit.py
up /workspace/out/fruit_v1_instruct.pt

echo "=== STAGE2-COMPLETE ==="
