#!/bin/bash
# Full causal RSI sweep across all models (open-weight via GPU, closed via Bedrock API).
# Runs rsi_causal.py for every (model x arm x campaign-seed). Poll-based GPU pool: each job
# gets one GPU exclusively for its lifetime (so no open-weight OOM), 8 concurrent. Resumable:
# a job whose causal_trajectory.json already exists is skipped. Robust for multi-hour runs.
#
# Launch:  setsid bash rsi_all.sh > /tmp/instance_storage/ka_data/rsi_all.log 2>&1 < /dev/null &
set -u
CODE=/tmp/instance_storage/kernelascent
DATA=/tmp/instance_storage/ka_data/causal_all
ET=/tmp/instance_storage/ka_data/expert_times.json
export HF_HOME=/tmp/instance_storage/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWS_SHARED_CREDENTIALS_FILE=/tmp/instance_storage/bedrock_creds AWS_PROFILE=bedrock BEDROCK_PROFILE=bedrock
mkdir -p "$DATA"; cd "$CODE"
NGPU=8
ARMS="growing frozen offline search"
SEEDS="0 1"
PN=8; TN=12; ROUNDS=4

# model registry: "kind|label|spec"  (kind = api uses Bedrock, hf loads weights on the GPU)
MODELS=(
  "api|fable51|us.anthropic.claude-fable-5-1"
  "api|opus5|us.anthropic.claude-opus-5"
  "api|sonnet5|us.anthropic.claude-sonnet-5"
  "api|haiku45|us.anthropic.claude-haiku-4-5-20251001-v1:0"
  "api|deepseekv32|deepseek.v3.2"
  "api|qwen3next|qwen.qwen3-next-80b-a3b"
  "hf|coder1p5b|Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "hf|coder3b|Qwen/Qwen2.5-Coder-3B-Instruct"
  "hf|coder7b|Qwen/Qwen2.5-Coder-7B-Instruct"
  "hf|coder14b|Qwen/Qwen2.5-Coder-14B-Instruct"
  "hf|qwen7b|Qwen/Qwen2.5-7B-Instruct"
  "hf|qwen14b|Qwen/Qwen2.5-14B-Instruct"
  "hf|deepseek67b|deepseek-ai/deepseek-coder-6.7b-instruct"
  "hf|starcoder15b|bigcode/starcoder2-15b-instruct-v0.1"
  "hf|codellama13b|codellama/CodeLlama-13b-Instruct-hf"
)

# build the job list
JOBS=()
for m in "${MODELS[@]}"; do
  for arm in $ARMS; do for cs in $SEEDS; do JOBS+=("$m|$arm|$cs"); done; done
done
echo "total jobs: ${#JOBS[@]}  ($(date -u +%FT%TZ))"

FREE=($(seq 0 $((NGPU-1))))
declare -A P2G   # pid -> gpu

launch(){  # $1 = "kind|label|spec|arm|cs"  $2 = gpu
  IFS='|' read -r kind label spec arm cs <<< "$1"
  local O="$DATA/${label}_${arm}_cs${cs}"; mkdir -p "$O"
  local common="--arm $arm --expert-times $ET --practice-n $PN --transfer-n $TN --rounds $ROUNDS --campaign-seed $cs --outdir $O"
  if [ "$kind" = api ]; then
    CUDA_VISIBLE_DEVICES=$2 setsid python3 -u rsi_causal.py --model $label --api-model $spec $common > "$O/run.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$2 setsid python3 -u rsi_causal.py --model $spec $common > "$O/run.log" 2>&1 &
  fi
  P2G[$!]=$2
  echo "[$(date -u +%H:%M:%SZ)] launch gpu$2 $label $arm cs$cs (pid $!)"
}

i=0
while [ $i -lt ${#JOBS[@]} ] || [ ${#P2G[@]} -gt 0 ]; do
  # reap finished jobs, reclaim their GPU
  for pid in "${!P2G[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then FREE+=("${P2G[$pid]}"); unset "P2G[$pid]"; fi
  done
  # fill free GPUs with pending jobs (skipping already-done ones)
  while [ ${#FREE[@]} -gt 0 ] && [ $i -lt ${#JOBS[@]} ]; do
    job="${JOBS[$i]}"; i=$((i+1))
    IFS='|' read -r kind label spec arm cs <<< "$job"
    if [ -f "$DATA/${label}_${arm}_cs${cs}/causal_trajectory.json" ]; then
      echo "skip (done) $label $arm cs$cs"; continue
    fi
    g="${FREE[0]}"; FREE=("${FREE[@]:1}")
    launch "$job" "$g"
  done
  sleep 8
done
echo "ALL_DONE ($(date -u +%FT%TZ))"
