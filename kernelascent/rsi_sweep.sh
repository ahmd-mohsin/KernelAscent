#!/bin/bash
# Long-running open-weight RSI sweep. Walks the 0.5B -> 14B ladder sequentially, runs the
# execution-feedback RSI loop over all four tiers for each model, and loops forever with a
# fresh task seed each pass so trajectories keep accumulating. Every round's raw model
# output (its reasoning/attempt) is stored per task in rounds.json, so we can mine WHY
# models fail and stall and use that to harden the benchmark.
#
# Robust for multi-hour unattended runs: each model is its own python process (GPU freed
# between models), a failure is logged and skipped, and progress/heartbeat go to sweep.log.
# Launch detached:  setsid bash rsi_sweep.sh <gpu> > .../sweep.log 2>&1 < /dev/null &
set -u
CODE=/tmp/instance_storage/kernelascent
OUT=/tmp/instance_storage/ka_data/rsi_sweep
export HF_HOME=/tmp/instance_storage/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${1:-1}
N=${2:-6}
ROUNDS=${3:-4}
mkdir -p "$OUT"; cd "$CODE"

# 0.5B -> 14B ladder: Qwen2.5-Coder then Qwen2.5-Instruct
MODELS=(
  Qwen/Qwen2.5-Coder-0.5B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct
  Qwen/Qwen2.5-Coder-7B-Instruct   Qwen/Qwen2.5-Coder-14B-Instruct
  Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct
  Qwen/Qwen2.5-7B-Instruct   Qwen/Qwen2.5-14B-Instruct
)

pass=0
while true; do
  seed=$((pass * 1000))
  echo "======== PASS $pass  seed0=$seed  $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
  for m in "${MODELS[@]}"; do
    slug=$(echo "$m" | tr '/' '_')
    od="$OUT/pass${pass}/$slug"
    mkdir -p "$od"
    echo "[$(date -u +%H:%M:%SZ)] pass=$pass model=$m -> $od"
    CUDA_VISIBLE_DEVICES=$GPU python3 -u openweight_rsi.py --model "$m" \
      --tiers Easy,Medium,Hard,Ultra --n "$N" --rounds "$ROUNDS" --seed0 "$seed" \
      --outdir "$od" > "$od/model.log" 2>&1 || echo "  FAILED $m (see $od/model.log)"
    echo "[$(date -u +%H:%M:%SZ)] done $m"
  done
  echo "======== PASS $pass COMPLETE $(date -u +%H:%M:%SZ) ========"
  pass=$((pass + 1))
done
