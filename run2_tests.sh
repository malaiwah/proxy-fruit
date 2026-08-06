#!/bin/bash
# Run-2 kit validation matrix (RTX 5090, tiny geometry, SFT smoke shards).
set -u
IMG=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25
OUT=/mnt/vault/llm/fruit-pilot/run2-tests
rm -rf "$OUT"; mkdir -p "$OUT"
COMMON=(-v "$HOME/fruit-pilot:/fp" -v /mnt/vault:/mnt/vault -v fruit-pip:/piploc
        -e PYTHONPATH=/piploc -e HF_HUB_DISABLE_XET=1
        -e TOK_DIR=/mnt/vault/llm/glm52-franken/src
        -e SHARD_DIR=/mnt/vault/llm/fruit-pilot/sft-shards-smoke
        -e MOE_IMPL=stacked -e EVAL_EVERY=100000 -e LR=3e-4
        -e SEQ=512 -e BS=8 -e FRUIT_OUT_DIR="$OUT")
run() {  # name steps extra-env...
  local name=$1 steps=$2; shift 2
  echo "=== $name ==="
  podman run --rm --name "run2-$name" --device nvidia.com/gpu=all \
    --shm-size 8g "${COMMON[@]}" -e STEPS="$steps" -e SAVE_NAME="$name" \
    "$@" --entrypoint /opt/venv/bin/python3 "$IMG" /fp/train_fruit.py 2>&1 \
    | grep -aE "model\]|opt\]|\[[0-9]+/|mem\]|TRAIN-DONE|Traceback|Error" \
    | tail -8
}
run baseline 250
run bundle 250 -e BIAS_BALANCE=0.001 -e AUX_COEF=0.0001 -e ZLOSS_HEAD=1e-4 \
    -e NO_WD_EMB=1 -e TIE_EMB=1 -e LR_SCHED=wsd -e WARMUP=30 \
    -e MTP_W=0.3 -e MTP_W_END=0.1 -e DETERMINISTIC_DATA=1
run intradoc 150 -e INTRADOC_MASK=1
run det-a 60 -e DETERMINISTIC_DATA=1
run det-b 60 -e DETERMINISTIC_DATA=1
run eager-speed 100
run compile-speed 100 -e COMPILE=1
echo "=== MATRIX-DONE ==="
