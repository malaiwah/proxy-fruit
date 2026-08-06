#!/bin/bash
# Lightweight system-telemetry sampler -> `[telem]` ledger lines.
# One line per INTERVAL (default 60s): step (parsed from RUN_LOG tail),
# cpu%, mem-used GB, mean GPU util %, total VRAM GB, disk-used GB,
# net rx/tx MB/s. Zero deps beyond nvidia-smi + /proc. Costs ~nothing;
# buys the efficiency retrospective (was the run compute-bound or not).
# Usage: TELEM_OUT=/workspace/telem.log RUN_LOG=/workspace/run.log \
#        nohup bash telemetry.sh &
RUN_LOG=${RUN_LOG:-/workspace/run.log}
OUT=${TELEM_OUT:-/workspace/telem.log}
INTERVAL=${INTERVAL:-60}

read_net() {  # total rx tx bytes across non-lo interfaces
  awk -F'[: ]+' 'NR>2 && $2!="lo" {rx+=$3; tx+=$11} END {print rx+0, tx+0}' /proc/net/dev
}
read_cpu() {  # total and idle jiffies
  awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8, $5}' /proc/stat
}

read -r pt pi < <(read_cpu)
read -r prx ptx < <(read_net)
while true; do
  sleep "$INTERVAL"
  read -r t i < <(read_cpu)
  read -r rx tx < <(read_net)
  dt=$((t - pt)); di=$((i - pi))
  cpu=$(( dt > 0 ? (100 * (dt - di)) / dt : 0 ))
  mem=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%.1f",(t-a)/1048576}' /proc/meminfo)
  gv=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
       --format=csv,noheader,nounits 2>/dev/null \
       | awk -F', *' '{u+=$1; m+=$2; n++} END {if(n)printf "%.0f %.1f", u/n, m/1024; else print "0 0"}')
  # clocks / temps / power (throttle forensics): avg SM & mem clocks,
  # max temp, summed draw + limit across GPUs
  gtp=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' '{c+=$1; mc+=$2; if($3>t)t=$3; p+=$4; pl+=$5; n++}
                       END {if(n)printf "%.0f %.0f %.0f %.0f %.0f", c/n, mc/n, t, p, pl; else print "0 0 0 0 0"}')
  read -r gclk gmclk gtemp pdraw plim <<< "$gtp"
  cpumhz=$(awk '/^cpu MHz/{s+=$4; n++} END {if(n)printf "%.0f", s/n; else print 0}' /proc/cpuinfo)
  cputemp=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null \
            | awk 'BEGIN{m=0}{if($1>m)m=$1}END{printf "%.0f", m/1000}')
  disk=$(df -BG "${OUT%/*}" 2>/dev/null | awk 'NR==2{gsub("G","",$3); print $3}')
  mbrx=$(awk -v a="$rx" -v b="$prx" -v s="$INTERVAL" 'BEGIN{printf "%.1f",(a-b)/s/1048576}')
  mbtx=$(awk -v a="$tx" -v b="$ptx" -v s="$INTERVAL" 'BEGIN{printf "%.1f",(a-b)/s/1048576}')
  step=$(tail -c 262144 "$RUN_LOG" 2>/dev/null | grep -aoE "^\[[0-9]+/" | tail -1 | tr -d '[/')
  echo "[telem] step=${step:-0} time=$(date -u +%FT%TZ) cpu=$cpu mem=$mem gpuutil=${gv% *} vram=${gv#* } disk=${disk:-0} rx=$mbrx tx=$mbtx gpuclk=$gclk memclk=$gmclk gputemp=$gtemp pdraw=$pdraw plim=$plim cpumhz=$cpumhz cputemp=${cputemp:-0}" >> "$OUT"
  pt=$t; pi=$i; prx=$rx; ptx=$tx
done
