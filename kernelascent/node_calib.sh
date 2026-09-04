#!/bin/bash
# Per-node calibration driver. Fully local to the node it runs on: generates and
# grades a set of models (one per GPU), writes summaries + a NODE_DONE marker to
# the node's local disk. No cross-node filesystem assumptions. Gather summaries
# afterwards with scp. Launch on each node with:
#   setsid bash node_calib.sh <DATA_DIR> <TIERS> <NPT> <K> model:gpu [model:gpu ...] &
set -u
CODE=/tmp/instance_storage/kernelascent
DATA="$1"; TIERS="$2"; NPT="$3"; K="$4"; shift 4
export HF_HOME=/tmp/instance_storage/hf TOKENIZERS_PARALLELISM=false
mkdir -p "$DATA"; cd "$CODE"
rm -f "$DATA"/NODE_DONE

echo "=== $(hostname) gen: $* ==="
pids=()
for spec in "$@"; do
  mid="${spec%%:*}"; gpu="${spec##*:}"; slug=$(echo "$mid" | tr '/' '_')
  CUDA_VISIBLE_DEVICES=$gpu python3 -u qwen_calibrate.py --model "$mid" --outdir "$DATA/$slug" \
    --tiers "$TIERS" --n-per-tier "$NPT" --k "$K" > "$DATA/gen_$slug.log" 2>&1 &
  pids+=($!); echo "  gen $slug gpu $gpu pid $!"
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== gen done, grading ==="
gpids=()
for spec in "$@"; do
  mid="${spec%%:*}"; gpu="${spec##*:}"; slug=$(echo "$mid" | tr '/' '_')
  CUDA_VISIBLE_DEVICES=$gpu python3 -u grade_candidates.py --candir "$DATA/$slug" \
    --out "$DATA/$slug/summary.json" --cand-timeout 120 > "$DATA/grade_$slug.log" 2>&1 &
  gpids+=($!); echo "  grade $slug gpu $gpu pid $!"
done
for p in "${gpids[@]}"; do wait "$p"; done
touch "$DATA/NODE_DONE"
echo "=== NODE_DONE ==="
