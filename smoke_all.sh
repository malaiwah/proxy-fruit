#!/bin/bash
# Full-pipeline smoke suite — runs ON a 4x RTX 6000 Pro (sm120) Jarvis node.
# See SMOKE_PLAN.md. Pushes ONLY to malaiwah/fruit-smoke. ~90-120 min.
# Usage: HF_TOKEN=... bash /workspace/smoke_all.sh   (nohup recommended)
set -uo pipefail
cd /workspace
export PYTHONUNBUFFERED=1
: "${HF_TOKEN:?export HF_TOKEN first}"
PY=/workspace/venv/bin/python3
TR="/workspace/venv/bin/torchrun --standalone --nproc_per_node=4"
RESULTS=()
t() { RESULTS+=("$1 $2"); echo "TEST $1 $2"; }

echo "=== bootstrap ==="
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3-venv python3-pip >/dev/null 2>&1 || true
[ -x "$PY" ] || python3 -m venv /workspace/venv
$PY -m pip install -q --upgrade pip
CUV=$(nvidia-smi | grep -aoE "CUDA Version: [0-9]+\.[0-9]+" | grep -aoE "[0-9]+\.[0-9]+")
IDX=cu128; python3 -c "exit(0 if float('$CUV') >= 12.8 else 1)" || IDX=cu124
echo "driver CUDA $CUV -> torch $IDX"
$PY -m pip install -q torch numpy datasets transformers huggingface_hub \
  bitsandbytes matplotlib --extra-index-url "https://download.pytorch.org/whl/$IDX" 2>&1 | tail -1
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0), 'x', torch.cuda.device_count())"
$PY - <<'EOF'
from huggingface_hub import hf_hub_download
import os
os.makedirs('/workspace/tokenizer', exist_ok=True)
for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    try: hf_hub_download('willfalco/GLM-5.2-EXL3-TR3-3.36bpw', f, local_dir='/workspace/tokenizer')
    except Exception as e: print('skip', f, e)
print('tokenizer ready')
EOF

# common tiny-geometry env (trainer defaults: H=512 NL=6 HEADS=8 MOE_INTER=128)
export TOK_DIR=/workspace/tokenizer FRUIT_OUT_DIR=/workspace/out \
  EVAL_EVERY=100000 MOE_IMPL=stacked LR=3e-4 SEQ=512
mkdir -p /workspace/out
last_loss() { grep -aE "^\[[0-9]+/" "$1" | tail -1 | grep -aoE "loss [0-9.]+" | cut -d' ' -f2; }

echo "=== T00 GPU acceptance ==="
nvidia-smi topo -m | head -8
if timeout 120 $TR --no-python true 2>/dev/null || true; then :; fi
timeout 180 $TR -m torch.distributed.run 2>/dev/null || true
cat > /tmp/ar.py <<'EOF'
import torch, torch.distributed as dist, os, time
dist.init_process_group('nccl'); r = dist.get_rank()
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
x = torch.ones(256*1024*1024//4, device='cuda')
for _ in range(3): dist.all_reduce(x)
torch.cuda.synchronize(); t0=time.time()
for _ in range(10): dist.all_reduce(x)
torch.cuda.synchronize()
if r == 0: print(f"ALLREDUCE-OK {10*2*3*256/ (time.time()-t0)/1024:.1f} GB/s busbw-ish")
dist.destroy_process_group()
EOF
NCCL_MIN_NCHANNELS=8 NCCL_P2P_LEVEL=SYS NCCL_BUFFSIZE=33554432 \
  timeout 240 $TR /tmp/ar.py 2>&1 | grep -a ALLREDUCE-OK | sed 's/^/tuned: /' || true
if timeout 240 $TR /tmp/ar.py 2>&1 | grep -a ALLREDUCE-OK; then t T00 PASS; else
  export NCCL_P2P_DISABLE=1
  if timeout 240 $TR /tmp/ar.py 2>&1 | grep -a ALLREDUCE-OK; then t T00 "PASS(P2P-off)"; else t T00 FAIL; fi
fi

echo "=== T01 data: pull existing shards from HF + prep ONE fresh source ==="
# tests BOTH the resume-from-saved-shards path and fresh processing,
# at ~3 min instead of streaming all 9 sources (~15 min)
mkdir -p /workspace/shards
if $PY -c "
from huggingface_hub import hf_hub_download
import os, shutil, json
for f in ('spdx.u32', 'reap_calib.u32'):
    p = hf_hub_download('malaiwah/fruit-phase1-shards', f,
                        repo_type='dataset', token=os.environ['HF_TOKEN'])
    shutil.copy(p, f'/workspace/shards/{f}')
print('SHARDS-PULLED')" 2>&1 | grep -aq SHARDS-PULLED \
   && timeout 900 env SMOKE=1 ONLY=tinystories TOK_DIR=/workspace/tokenizer \
      OUT_DIR=/workspace/shards $PY fruit_data_prep.py 2>&1 | tail -2 | grep -aq "PREP-DONE" \
   && $PY -c "
import json
m = json.load(open('/workspace/shards/manifest.json'))
import os
for n, w in (('spdx', 0.3), ('reap_calib', 0.3)):
    m['weights'][n] = w
    m['tokens'][n] = os.path.getsize(f'/workspace/shards/{n}.u32') // 4
json.dump(m, open('/workspace/shards/manifest.json', 'w'), indent=1)
print('MANIFEST-MERGED', m['tokens'])"; then t T01 PASS; else t T01 FAIL; fi

echo "=== T02 SFT data prep (SMOKE + aider + replay) ==="
$PY -c "
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download('malaiwah/fruit-phase1-shards', 'sft/sft-aider.jsonl',
                    repo_type='dataset', token=os.environ['HF_TOKEN'])
shutil.copy(p, '/workspace/sft-aider.jsonl')" || true
if timeout 1200 env SMOKE=1 TOK_DIR=/workspace/tokenizer OUT_DIR=/workspace/sft-shards \
    AIDER_JSONL=/workspace/sft-aider.jsonl $PY sft_data_prep.py 2>&1 | tail -2 | grep -a "SFT-PREP-DONE" \
   && $PY finish_sft_manifest.py 2>&1 | grep -a MANIFEST; then t T02 PASS; else t T02 FAIL; fi

echo "=== T03 baseline DDP train + push ==="
rm -f /workspace/out/smoke3_ckpt.pt
if timeout 900 env SHARD_DIR=/workspace/shards BS=4 STEPS=120 SAVE_NAME=smoke3 \
    HF_PUSH_REPO=malaiwah/fruit-smoke PUSH_EVERY=100 $TR train_fruit.py 2>&1 \
    | tee /tmp/t03.log | grep -aE "push\] step|TRAIN-DONE" | tail -2 | grep -aq TRAIN-DONE \
   && grep -aq "\[push\] step" /tmp/t03.log; then t T03 PASS; else t T03 FAIL; fi

echo "=== T04 grouped == stacked (deterministic) ==="
rm -f /workspace/out/smoke4a_ckpt.pt /workspace/out/smoke4b_ckpt.pt
timeout 600 env SHARD_DIR=/workspace/shards BS=8 STEPS=60 DETERMINISTIC_DATA=1 \
  SAVE_NAME=smoke4a MOE_IMPL=stacked CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py > /tmp/t04a.log 2>&1
timeout 600 env SHARD_DIR=/workspace/shards BS=8 STEPS=60 DETERMINISTIC_DATA=1 \
  SAVE_NAME=smoke4b MOE_IMPL=grouped CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py > /tmp/t04b.log 2>&1
A=$(last_loss /tmp/t04a.log); B=$(last_loss /tmp/t04b.log)
if $PY -c "exit(0 if abs($A-$B) < 0.02 else 1)" 2>/dev/null; then t T04 "PASS($A~$B)"; else t T04 "FAIL($A vs $B)"; fi

echo "=== T05 run-2 bundle ==="
rm -f /workspace/out/smoke5_ckpt.pt
if timeout 900 env SHARD_DIR=/workspace/shards BS=4 STEPS=120 SAVE_NAME=smoke5 \
    BIAS_BALANCE=0.001 AUX_COEF=0.0001 ZLOSS_HEAD=1e-4 NO_WD_EMB=1 TIE_EMB=1 \
    LR_SCHED=wsd WARMUP=20 MTP_W=0.3 MTP_W_END=0.1 SKIP_SPIKES=1 \
    DETERMINISTIC_DATA=1 $TR train_fruit.py 2>&1 | tee /tmp/t05.log | tail -2 | grep -aq TRAIN-DONE; then
  L0=$(grep -aoE "^\[0/[0-9]+\] loss [0-9.]+" /tmp/t05.log | grep -aoE "[0-9.]+$")
  if $PY -c "exit(0 if float('${L0:-999}') < 20 else 1)"; then t T05 "PASS(l0=$L0)";
  else t T05 "FAIL(tie-init l0=$L0)"; fi
  else t T05 FAIL; fi

echo "=== T06 FP8_LINEAR ==="
rm -f /workspace/out/smoke6_ckpt.pt
if timeout 600 env SHARD_DIR=/workspace/shards BS=8 STEPS=40 SAVE_NAME=smoke6 \
    FP8_LINEAR=1 CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tail -2 | grep -aq TRAIN-DONE; then t T06 PASS; else t T06 FAIL; fi

echo "=== T07 SFT masked train ==="
rm -f /workspace/out/smoke7_ckpt.pt
if timeout 600 env SHARD_DIR=/workspace/sft-shards BS=8 STEPS=80 SAVE_NAME=smoke7 \
    MTP_W=0.1 CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tee /tmp/t07.log | tail -2 | grep -aq TRAIN-DONE \
   && grep -aq "SFT loss masks active" /tmp/t07.log; then t T07 PASS; else t T07 FAIL; fi

echo "=== T08 INTRADOC_MASK ==="
rm -f /workspace/out/smoke8_ckpt.pt
if timeout 600 env SHARD_DIR=/workspace/sft-shards BS=4 STEPS=40 SAVE_NAME=smoke8 \
    INTRADOC_MASK=1 CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tail -2 | grep -aq TRAIN-DONE; then t T08 PASS; else t T08 FAIL; fi

echo "=== T09 QNOISE ==="
rm -f /workspace/out/smoke9_ckpt.pt
if timeout 600 env SHARD_DIR=/workspace/shards BS=8 STEPS=40 SAVE_NAME=smoke9 \
    QNOISE=0.015 CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tail -2 | grep -aq TRAIN-DONE; then t T09 PASS; else t T09 FAIL; fi

echo "=== T10 token-clock tier hop (4 ranks -> 1 rank) ==="
rm -f /workspace/out/smoke10_ckpt.pt
timeout --signal=KILL 75 env SHARD_DIR=/workspace/shards BS=4 STEPS=999999 \
  TOKEN_BUDGET=3000000 WARMUP=20 SAVE_NAME=smoke10 DETERMINISTIC_DATA=1 \
  $TR train_fruit.py > /tmp/t10a.log 2>&1 || true
sleep 3; pkill -f train_fruit.py 2>/dev/null; sleep 3
if timeout 900 env SHARD_DIR=/workspace/shards BS=8 STEPS=999999 \
    TOKEN_BUDGET=3000000 WARMUP=20 SAVE_NAME=smoke10 DETERMINISTIC_DATA=1 \
    CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tee /tmp/t10b.log | tail -2 | grep -aq TRAIN-DONE \
   && grep -aq "\[resume\]" /tmp/t10b.log && grep -aq "\[tokens\]" /tmp/t10b.log; then
  t T10 PASS; else t T10 FAIL; fi

echo "=== T11 SIGTERM drill ==="
rm -f /workspace/out/smoke11_ckpt.pt
env SHARD_DIR=/workspace/shards BS=8 STEPS=5000 SAVE_NAME=smoke11 \
  CUDA_VISIBLE_DEVICES=0 nohup $PY train_fruit.py > /tmp/t11.log 2>&1 &
TP=$!; sleep 60; kill -TERM $TP; sleep 25
if grep -aq "\[sigterm\] snapshot" /tmp/t11.log; then t T11 PASS; else t T11 FAIL; fi
pkill -f train_fruit.py 2>/dev/null; sleep 3

echo "=== T12 SNAPSHOT_FORK + push ==="
rm -f /workspace/out/smoke12_ckpt.pt
if timeout 900 env SHARD_DIR=/workspace/shards BS=8 STEPS=130 SAVE_NAME=smoke12 \
    SNAPSHOT_SAVE=1 SNAPSHOT_FORK=1 HF_PUSH_REPO=malaiwah/fruit-smoke \
    PUSH_EVERY=60 CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tee /tmp/t12.log | tail -2 | grep -aq TRAIN-DONE \
   && grep -aq "\[snap\] paused" /tmp/t12.log; then t T12 PASS; else t T12 FAIL; fi

echo "=== T13 FP32_MASTER + paged + resume continuity ==="
rm -f /workspace/out/smoke13_ckpt.pt
timeout --signal=KILL 80 env SHARD_DIR=/workspace/shards BS=8 STEPS=5000 \
  SAVE_NAME=smoke13 FP32_MASTER=1 OPT_8BIT=paged CUDA_VISIBLE_DEVICES=0 \
  $PY train_fruit.py > /tmp/t13a.log 2>&1 || true
pkill -f train_fruit.py 2>/dev/null; sleep 3
PRE=$(last_loss /tmp/t13a.log)
timeout 600 env SHARD_DIR=/workspace/shards BS=8 STEPS=$(( $(grep -aoE "^\[[0-9]+/" /tmp/t13a.log | tail -1 | tr -d '[/') + 40 )) \
  SAVE_NAME=smoke13 FP32_MASTER=1 OPT_8BIT=paged CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py > /tmp/t13b.log 2>&1 || true
POST=$(last_loss /tmp/t13b.log)
if $PY -c "exit(0 if float($POST) < float($PRE)*1.5 + 1 else 1)" 2>/dev/null \
   && grep -aq "\[resume\]" /tmp/t13b.log; then t T13 "PASS($PRE->$POST)"; else t T13 "FAIL($PRE->$POST)"; fi

echo "=== T14 INDEXER_DISTILL ==="
rm -f /workspace/out/smoke14_ckpt.pt
if timeout 600 env SHARD_DIR=/workspace/shards BS=4 STEPS=80 SEQ=512 \
    SAVE_NAME=smoke14 INDEXER_DISTILL=1 CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tee /tmp/t14.log \
    | tail -2 | grep -aq TRAIN-DONE && grep -aq "distill\] kl" /tmp/t14.log; then
  t T14 PASS; else t T14 FAIL; fi

echo "=== T15 HF recovery ==="
mv /workspace/out/smoke3_ckpt.pt /tmp/smoke3_local.pt 2>/dev/null || true
if $PY -c "
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download('malaiwah/fruit-smoke', 'checkpoints/smoke3_ckpt.pt',
                    token=os.environ['HF_TOKEN'])
shutil.copy(p, '/workspace/out/smoke3_ckpt.pt'); print('PULLED')" 2>&1 | grep -aq PULLED \
   && timeout 600 env SHARD_DIR=/workspace/shards BS=4 STEPS=140 SAVE_NAME=smoke3 \
      CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tee /tmp/t15.log | grep -aq TRAIN-DONE \
   && grep -aq "\[resume\]" /tmp/t15.log; then t T15 PASS; else t T15 FAIL; fi

echo "=== T16 probe_ckpt ==="
if timeout 600 env CKPT=/workspace/out/smoke3_ckpt.pt TOK_DIR=/workspace/tokenizer \
    CUDA_VISIBLE_DEVICES=0 $PY probe_ckpt.py 2>&1 | tail -3 | grep -aq PROBE-DONE; then
  t T16 PASS; else t T16 FAIL; fi

echo "=== T17 ledger publish (incarnation banners) ==="
grep -ahE "^\[val |^\[[0-9]+/|^\[incarnation" /tmp/t03.log /tmp/t10a.log /tmp/t10b.log \
  > /tmp/node_lines.txt 2>/dev/null || true
if grep -aq "\[incarnation\]" /tmp/node_lines.txt \
   && env NODE_LINES=/tmp/node_lines.txt WORK=/tmp REPO=malaiwah/fruit-smoke \
      PLOT_TITLE="fruit-smoke suite" $PY progress_publish.py 2>&1 | grep -aq PROGRESS-PUBLISHED; then
  t T17 PASS; else t T17 FAIL; fi

echo "=== T18 deterministic-data repeat (bit-level data order) ==="
rm -f /workspace/out/smoke18_ckpt.pt
timeout 400 env SHARD_DIR=/workspace/shards BS=8 STEPS=30 DETERMINISTIC_DATA=1 \
  SAVE_NAME=smoke18x CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py > /tmp/t18a.log 2>&1
rm -f /workspace/out/smoke18x_ckpt.pt
timeout 400 env SHARD_DIR=/workspace/shards BS=8 STEPS=30 DETERMINISTIC_DATA=1 \
  SAVE_NAME=smoke18x CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py > /tmp/t18b.log 2>&1
A=$(last_loss /tmp/t18a.log); B=$(last_loss /tmp/t18b.log)
if $PY -c "exit(0 if abs($A-$B) < 0.01 else 1)" 2>/dev/null; then t T18 "PASS($A~$B)"; else t T18 "FAIL($A vs $B)"; fi

echo "=== T19 guarded compile path (COMPILE=1, no grad-ckpt) ==="
rm -f /workspace/out/smoke19_ckpt.pt
if timeout 900 env SHARD_DIR=/workspace/shards BS=4 STEPS=25 SAVE_NAME=smoke19 \
    COMPILE=1 CUDA_VISIBLE_DEVICES=0 $PY train_fruit.py 2>&1 | tail -2 | grep -aq TRAIN-DONE; then
  t T19 PASS; else t T19 FAIL; fi

echo ""
echo "=========== SMOKE SUITE SUMMARY ==========="
for r in "${RESULTS[@]}"; do echo "  $r"; done
FAILS=$(printf '%s\n' "${RESULTS[@]}" | grep -ac FAIL || true)
echo "SMOKE-SUITE-DONE: $FAILS failures"
