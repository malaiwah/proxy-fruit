#!/bin/bash
# Repeating 60-min license-distillation rounds until STOP_DISTILL exists.
IMG=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25
R=2
while [ ! -f "$HOME/fruit-pilot/STOP_DISTILL" ]; do
  # wait for any running round to finish
  while podman ps --format '{{.Names}}' | grep -q glm-lic-distill; do sleep 60; done
  [ -f "$HOME/fruit-pilot/STOP_DISTILL" ] && break
  R=$((R+1))
  echo "ROUND $R starting $(date -u +%H:%M)"
  podman run --rm --name glm-lic-distill$R --network host \
    -v "$HOME/fruit-pilot:/fp" -v fruit-pip:/piploc \
    -e PYTHONPATH=/piploc -e HF_HUB_DISABLE_XET=1 \
    -e DEADLINE_MIN=60 -e Q_PER_LIC=6 \
    -e OUT_JSONL=/fp/sft-glm-lic2.jsonl \
    --entrypoint /opt/venv/bin/python3 "$IMG" /fp/license_distill.py \
    >> "$HOME/fruit-pilot/license_distill2.log" 2>&1
  echo "ROUND $R done, rows now: $(wc -l < $HOME/fruit-pilot/sft-glm-lic2.jsonl 2>/dev/null || echo 0)"
done
echo "DISTILL-LOOP-STOPPED"
