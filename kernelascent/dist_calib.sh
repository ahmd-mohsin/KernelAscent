#!/bin/bash
# Distributed tier calibration across 3 nodes x 8 GPUs (24 slots).
# Runs the Qwen2.5-Coder ladder + Qwen2.5-Instruct ladder as test-takers,
# task-sharded across the fleet, then grades each model. Robust to tunnel
# drops: remote jobs are detached (setsid) and the launcher polls DONE markers.
# Run this ON the main node:  setsid bash dist_calib.sh > .../dist.log 2>&1 &
set -u
CODE=/tmp/instance_storage/kernelascent
DATA=/tmp/instance_storage/ka_data/dist_calib
export HF_HOME=/tmp/instance_storage/hf
mkdir -p "$DATA"

NODES=(10.3.187.114 10.3.145.192 10.3.84.11)   # main, worker1, worker2
GPUS_PER_NODE=8
MODELS=(
  Qwen/Qwen2.5-Coder-0.5B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct
  Qwen/Qwen2.5-Coder-7B-Instruct   Qwen/Qwen2.5-Coder-14B-Instruct
  Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct
  Qwen/Qwen2.5-7B-Instruct   Qwen/Qwen2.5-14B-Instruct
)
NSHARDS=2            # task shards per model  -> 10 models x 2 = 20 gen jobs (<= 24 slots)
TIERS="Easy,Medium,Hard,Ultra"
NPT=30               # tasks per tier
K=3

rj() {  # rj <node_ip> <cmd> : run detached on a node, return immediately
  env SSH_ASKPASS_REQUIRE=force SSH_ASKPASS=/tmp/instance_storage/ap.sh DISPLAY=:0 \
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 \
    -o PreferredAuthentications=password -o PubkeyAuthentication=no \
    -p 2222 greenland-user@"$1" "$2" < /dev/null 2>/dev/null
}

# ---- build (node,gpu) slot list ----
SLOTS=()
for n in "${NODES[@]}"; do for g in $(seq 0 $((GPUS_PER_NODE-1))); do SLOTS+=("$n:$g"); done; done

# ---- phase 1: generation (sharded) ----
echo "=== GEN: ${#MODELS[@]} models x $NSHARDS shards ==="
i=0; ngen=0
for m in "${MODELS[@]}"; do
  slug=$(echo "$m" | tr '/' '_')
  for sh in $(seq 0 $((NSHARDS-1))); do
    slot="${SLOTS[$i]}"; node="${slot%%:*}"; gpu="${slot##*:}"; i=$((i+1)); ngen=$((ngen+1))
    tag="${slug}_s${sh}"
    rj "$node" "cd $CODE && export HF_HOME=$HF_HOME TOKENIZERS_PARALLELISM=false && setsid bash -c 'CUDA_VISIBLE_DEVICES=$gpu python3 -u qwen_calibrate.py --model $m --outdir $DATA/$slug --tiers $TIERS --n-per-tier $NPT --k $K --nshards $NSHARDS --shard $sh > $DATA/gen_${tag}.log 2>&1; touch $DATA/GEN_DONE_${tag}' < /dev/null &"
    echo "  gen $tag -> $node gpu $gpu"
  done
done
echo "waiting for $ngen gen jobs..."
while true; do d=$(ls "$DATA"/GEN_DONE_* 2>/dev/null | wc -l); echo "  gen done=$d/$ngen"; [ "$d" -ge "$ngen" ] && break; sleep 20; done
echo "=== GEN complete ==="

# ---- phase 2: grading (one job per model) ----
echo "=== GRADE: ${#MODELS[@]} models ==="
i=0; ngr=0
for m in "${MODELS[@]}"; do
  slug=$(echo "$m" | tr '/' '_')
  slot="${SLOTS[$i]}"; node="${slot%%:*}"; gpu="${slot##*:}"; i=$((i+1)); ngr=$((ngr+1))
  rj "$node" "cd $CODE && setsid bash -c 'CUDA_VISIBLE_DEVICES=$gpu python3 -u grade_candidates.py --candir $DATA/$slug --out $DATA/$slug/summary.json --cand-timeout 120 > $DATA/grade_${slug}.log 2>&1; touch $DATA/GRADE_DONE_${slug}' < /dev/null &"
  echo "  grade $slug -> $node gpu $gpu"
done
echo "waiting for $ngr grade jobs..."
while true; do d=$(ls "$DATA"/GRADE_DONE_* 2>/dev/null | wc -l); echo "  grade done=$d/$ngr"; [ "$d" -ge "$ngr" ] && break; sleep 20; done
echo "=== ALL DONE ==="
