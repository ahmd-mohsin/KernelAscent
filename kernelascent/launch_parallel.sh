#!/bin/bash
# Launch one agent worker per GPU; each handles a shard of the task set.
# Usage: launch_parallel.sh [N] [K] [SEED0] [NGPU] [OUTDIR]
set -e
N=${1:-80}; K=${2:-8}; SEED0=${3:-0}; NGPU=${4:-8}
OUT=${5:-/tmp/instance_storage/ka_data/curate}
mkdir -p "$OUT"
export HF_HOME=/tmp/instance_storage/ka_data/hf
cd /tmp/instance_storage/kernelascent
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i nohup env HF_HOME=$HF_HOME python agent_bench.py \
      --n "$N" --seed0 "$SEED0" --k "$K" --nshards "$NGPU" --shard "$i" \
      --out "$OUT/shard_$i.json" > "$OUT/shard_$i.log" 2>&1 &
  echo "worker $i -> GPU $i (pid $!)"
done
echo "launched $NGPU workers, N=$N K=$K seed0=$SEED0 -> $OUT"
