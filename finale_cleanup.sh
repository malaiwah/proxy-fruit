#!/bin/bash
# Cleanup pass after finale2: the three test-calibration fixes.
#   parity  — MOE_IMPL=grouped (stage2 ckpts use the stacked layout)
#   needle  — scorer compares only up to reference length
#   chat    — max_tokens 700 (verbose 5B; 300 truncated mid-answer)
set -uo pipefail
R25=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25
FIN=/mnt/vault/llm/fruit-pilot/final
OUTBASE=/mnt/vault/llm/fruit-pilot/output

pod() {
  local pre=()
  while [ "$1" != "--" ]; do pre+=("$1"); shift; done; shift
  podman run --rm --name fruit-cleanup --device nvidia.com/gpu=all \
    --shm-size 8g --network host \
    -v "$HOME/fruit-pilot:/fp" -v /mnt/vault:/mnt/vault -v fruit-pip:/piploc \
    -v "$HOME/glm52-franken/tools:/tools" \
    -e PYTHONPATH=/piploc -e HF_HUB_DISABLE_XET=1 \
    -e GEO_H=1024 -e GEO_NL=13 -e GEO_HEADS=16 -e GEO_QLORA=1024 \
    -e GEO_DENSE_INTER=2048 -e GEO_MOE_INTER=512 \
    "${pre[@]}" --entrypoint /opt/venv/bin/python3 "$R25" "$@"
}

for v in final instruct; do
  echo "=== NEEDLE $v (fixed scorer) ==="
  pod -- /fp/fruit_needle.py "$OUTBASE/GLM-5.2-SIQ-Fruit-$v" fp8_ds_mla
done

echo "=== CHAT instruct (700 tok) ==="
pod -- /fp/fruit_chat_test.py "$OUTBASE/GLM-5.2-SIQ-Fruit-instruct" fp8_ds_mla

echo "=== PARITY final (grouped ref, then serve) ==="
pod -e PHASE=ref -e MOE_IMPL=grouped -e CKPT=$FIN/fruit_v1_final.pt \
  -e REF_PT=/mnt/vault/llm/fruit-pilot/parity_ref_final.pt -- /fp/parity_test.py
pod -e PHASE=serve -e SERVED=$OUTBASE/GLM-5.2-SIQ-Fruit-final \
  -e REF_PT=/mnt/vault/llm/fruit-pilot/parity_ref_final.pt -- /fp/parity_test.py

echo "=== CLEANUP-COMPLETE ==="
