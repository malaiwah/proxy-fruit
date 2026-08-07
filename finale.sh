#!/bin/bash
# Phase-1 finale on the 5090: export the three stage2 deliverables through
# the fully-fixed path (RoPE perm + MTP eh_proj swap), then gauntlet each:
# r25 battery, MTP acceptance, r28 battery; needle+chat+parity where they
# apply. Sequential — one GPU. Log greppable: every test prints an
# OK/FAIL sentinel.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
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
    -v "$SCRIPT_DIR:/fp:ro" -v /mnt/vault:/mnt/vault -v fruit-pip:/piploc \
    -v "$HOME/glm52-franken/tools:/tools" \
    -e PYTHONPATH=/piploc -e HF_HUB_DISABLE_XET=1 \
    -e GEO_H=1024 -e GEO_NL=13 -e GEO_HEADS=16 -e GEO_QLORA=1024 \
    -e GEO_DENSE_INTER=2048 -e GEO_MOE_INTER=512 \
    "${pre[@]}" --entrypoint /opt/venv/bin/python3 "$img" "$@"
}

run_arm() {  # run_arm <label> <success sentinel> <command...>
  local label=$1 sentinel=$2 log status tee_status
  local statuses=()
  shift 2
  log=$(mktemp)
  set +e
  "$@" 2>&1 | tee "$log"
  statuses=("${PIPESTATUS[@]}")
  set -e
  status=${statuses[0]}
  tee_status=${statuses[1]}
  if [ "$tee_status" -ne 0 ]; then
    echo "FINALE-FAIL $label (tee status $tee_status)"
    rm -f "$log"
    return "$tee_status"
  fi
  if [ "$status" -eq 0 ]; then
    rm -f "$log"
    return 0
  fi
  if [ "$status" -eq 139 ]; then
    if grep -Fq "$sentinel" "$log"; then
      echo "FINALE-TEARDOWN-139 $label (accepted after $sentinel)"
      rm -f "$log"
      return 0
    fi
    echo "FINALE-FAIL $label (teardown 139 before $sentinel)"
  else
    echo "FINALE-FAIL $label (status $status)"
  fi
  rm -f "$log"
  return "$status"
}

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  return 0
fi

for v in final annealed instruct; do
  OUT=$OUTBASE/GLM-5.2-SIQ-Fruit-$v
  if [ ! -d "$OUT" ]; then
    echo "=== EXPORT $v ==="
    run_arm "export-$v" "EXPORT-COMPLETE:" \
      pod "$R25" -e FRUIT_PT=$FIN/fruit_v1_$v.pt -e FRUIT_OUT=$OUT \
      -e FRUIT_ROPE_THETA=500000 -- /fp/export_fruit.py
  fi
  echo "=== R25 BATTERY $v ==="
  run_arm "r25-$v" "FRUIT-SERVE-OK" \
    pod "$R25" -- /fp/fruit_serve_test.py "$OUT" fp8_ds_mla
  echo "=== MTP $v ==="
  run_arm "mtp-$v" "FRUIT-MTP-OK" \
    pod "$R25" -e K=1 -- /fp/fruit_serve_mtp.py "$OUT" fp8_ds_mla
  echo "=== R28 BATTERY $v ==="
  run_arm "r28-$v" "FRUIT-SERVE-OK" \
    pod "$R28" -e FRUIT_CKPT="$OUT" -- /fp/fruit_serve_r28.py
done

for v in final instruct; do
  echo "=== NEEDLE $v ==="
  run_arm "needle-$v" "FRUIT-NEEDLE-OK" \
    pod "$R25" -- /fp/fruit_needle.py \
      "$OUTBASE/GLM-5.2-SIQ-Fruit-$v" fp8_ds_mla
done

echo "=== CHAT instruct ==="
run_arm "chat-instruct" "FRUIT-CHAT-OK" \
  pod "$R25" -- /fp/fruit_chat_test.py \
    "$OUTBASE/GLM-5.2-SIQ-Fruit-instruct" fp8_ds_mla

echo "=== PARITY final (ref then serve) ==="
run_arm "parity-final-ref" "PARITY-REF-OK" \
  pod "$R25" -e PHASE=ref -e CKPT=$FIN/fruit_v1_final.pt \
    -e REF_PT=/mnt/vault/llm/fruit-pilot/parity_ref_final.pt \
    -e MOE_IMPL=grouped -e ROPE_THETA=500000 -- /fp/parity_test.py
run_arm "parity-final-serve" "PARITY-OK" \
  pod "$R25" -e PHASE=serve \
    -e SERVED=$OUTBASE/GLM-5.2-SIQ-Fruit-final \
    -e REF_PT=/mnt/vault/llm/fruit-pilot/parity_ref_final.pt \
    -- /fp/parity_test.py

echo "=== FULL-VOCAB KLD final (ref then serve) ==="
run_arm "kld-final-ref" "FRUIT-KLD-REF" \
  pod "$R25" -e PHASE=ref -e CKPT=$FIN/fruit_v1_final.pt \
    -e REF_PT=/mnt/vault/llm/fruit-pilot/kld_ref_final.pt \
    -e MOE_IMPL=grouped -e ROPE_THETA=500000 -- /fp/fruit_kld.py
run_arm "kld-final-serve" "FRUIT-KLD-OK" \
  pod "$R25" -e PHASE=serve \
    -e SERVED=$OUTBASE/GLM-5.2-SIQ-Fruit-final \
    -e REF_PT=/mnt/vault/llm/fruit-pilot/kld_ref_final.pt \
    -e OUT_JSON=/mnt/vault/llm/fruit-pilot/kld_final.json \
    -e MAX_MEAN_KL=0.05 -e MIN_TOP1=0.80 -- /fp/fruit_kld.py

echo "=== FINALE-COMPLETE ==="
