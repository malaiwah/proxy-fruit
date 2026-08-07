#!/bin/bash
# Phase-1 finale on the 5090: export the three stage2 deliverables through
# the fully-fixed path (RoPE perm + MTP eh_proj swap), then gauntlet each:
# r25 battery, MTP acceptance, r28 battery; needle+chat+parity where they
# apply. Sequential — one GPU. Log greppable: every test prints an
# OK/FAIL sentinel.
set -euo pipefail
R25=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25
R28=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28
FIN=/mnt/vault/llm/fruit-pilot/final
OUTBASE=/mnt/vault/llm/fruit-pilot/output

pod() {  # pod <image> <extra podman args...> -- <python script + args>
  local img=$1; shift
  local pre=()
  while [ "$1" != "--" ]; do pre+=("$1"); shift; done; shift
  podman run --rm --name fruit-finale --device nvidia.com/gpu=all \
    --shm-size 8g --network host \
    -v "$HOME/fruit-pilot:/fp" -v /mnt/vault:/mnt/vault -v fruit-pip:/piploc \
    -v "$HOME/glm52-franken/tools:/tools" \
    -e PYTHONPATH=/piploc -e HF_HUB_DISABLE_XET=1 \
    -e GEO_H=1024 -e GEO_NL=13 -e GEO_HEADS=16 -e GEO_QLORA=1024 \
    -e GEO_DENSE_INTER=2048 -e GEO_MOE_INTER=512 \
    "${pre[@]}" --entrypoint /opt/venv/bin/python3 "$img" "$@"
}

for v in final annealed instruct; do
  OUT=$OUTBASE/GLM-5.2-SIQ-Fruit-$v
  if [ ! -d "$OUT" ]; then
    echo "=== EXPORT $v ==="
    pod "$R25" -e FRUIT_PT=$FIN/fruit_v1_$v.pt -e FRUIT_OUT=$OUT \
      -e FRUIT_ROPE_THETA=500000 -- /fp/export_fruit.py \
      || { echo "FINALE-FAIL export-$v"; exit 1; }
  fi
  echo "=== R25 BATTERY $v ==="
  pod "$R25" -- /fp/fruit_serve_test.py "$OUT" fp8_ds_mla
  echo "=== MTP $v ==="
  pod "$R25" -e K=1 -- /fp/fruit_serve_mtp.py "$OUT" fp8_ds_mla
  echo "=== R28 BATTERY $v ==="
  pod "$R28" -e FRUIT_CKPT="$OUT" -- /fp/fruit_serve_r28.py
done

for v in final instruct; do
  echo "=== NEEDLE $v ==="
  pod "$R25" -- /fp/fruit_needle.py "$OUTBASE/GLM-5.2-SIQ-Fruit-$v" fp8_ds_mla
done

echo "=== CHAT instruct ==="
pod "$R25" -- /fp/fruit_chat_test.py "$OUTBASE/GLM-5.2-SIQ-Fruit-instruct" \
  fp8_ds_mla

echo "=== PARITY final (ref then serve) ==="
pod "$R25" -e PHASE=ref -e CKPT=$FIN/fruit_v1_final.pt \
  -e REF_PT=/mnt/vault/llm/fruit-pilot/parity_ref_final.pt \
  -- /fp/parity_test.py
pod "$R25" -e PHASE=serve -e SERVED=$OUTBASE/GLM-5.2-SIQ-Fruit-final \
  -e REF_PT=/mnt/vault/llm/fruit-pilot/parity_ref_final.pt \
  -- /fp/parity_test.py

echo "=== FINALE-COMPLETE ==="
