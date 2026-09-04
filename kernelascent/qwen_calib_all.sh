#!/bin/bash
# Tier-calibration driver: run the Qwen open-weight ladder as test-takers, one
# model per GPU, in parallel on one 8-GPU node, then grade each model's output.
# Usage: bash qwen_calib_all.sh
set -u
CODE=/tmp/instance_storage/kernelascent
DATA=/tmp/instance_storage/ka_data/calib
export HF_HOME=/tmp/instance_storage/hf
export TOKENIZERS_PARALLELISM=false
mkdir -p "$DATA"
cd "$CODE"

# model : gpu   (one small/mid model per A100-40GB; 14B fits comfortably in bf16)
MODELS=(
  "Qwen/Qwen2.5-Coder-0.5B-Instruct:0"
  "Qwen/Qwen2.5-Coder-1.5B-Instruct:1"
  "Qwen/Qwen2.5-Coder-3B-Instruct:2"
  "Qwen/Qwen2.5-Coder-7B-Instruct:3"
  "Qwen/Qwen2.5-Coder-14B-Instruct:4"
)
TIERS="Easy,Medium,Hard"
NPT=20   # tasks per tier
K=3

echo "=== generation (parallel, one model per GPU) ==="
pids=()
for spec in "${MODELS[@]}"; do
  mid="${spec%%:*}"; gpu="${spec##*:}"
  slug=$(echo "$mid" | tr '/' '_')
  out="$DATA/$slug"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u qwen_calibrate.py \
    --model "$mid" --outdir "$out" --tiers "$TIERS" --n-per-tier "$NPT" --k "$K" \
    > "$DATA/gen_$slug.log" 2>&1 &
  pids+=($!)
  echo "launched $mid on gpu $gpu (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== generation done ==="

echo "=== grading (one GPU per model, in parallel) ==="
gpids=()
for spec in "${MODELS[@]}"; do
  mid="${spec%%:*}"; gpu="${spec##*:}"; slug=$(echo "$mid" | tr '/' '_')
  out="$DATA/$slug"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u grade_candidates.py --candir "$out" --out "$out/summary.json" \
    --cand-timeout 120 > "$DATA/grade_$slug.log" 2>&1 &
  gpids+=($!)
  echo "grading $slug on gpu $gpu (pid $!)"
done
for p in "${gpids[@]}"; do wait "$p"; done
echo "ALL DONE"
