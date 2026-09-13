#!/usr/bin/env bash
# KernelAscent — packs one 8xA100-40GB node for TRUE-RSI experiments. Usage: ./launch_matrix.sh <ROLE> [SEED]
# ROLE picks what this node runs; each launch is detached (setsid + </dev/null + disown) and writes to /tmp/ka_data.
# 3-arm self-play uses --s-gpu/--f-gpu/--l-gpu (STATIC / FROZEN-AUTHOR / LIVE-AUTHOR; primary metric = L-F).
# Bigger models are bf16 and shard across their GPU list via KA_MAXMEM_GIB. Run collect_and_publish.sh on the hub.
set -u
ROLE="${1:?ROLE: mech|mech_large|selfplay_small|selfplay_mid|selfplay_large|selfplay_xlarge|closed}"
SEED="${2:-0}"
export KA_DTYPE=bf16 KA_DATA_DIR="$HOME/ka/ka_data"     # collector reads from here
mkdir -p "$KA_DATA_DIR" /tmp/ka_log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# unique per-run --outdir ($KA_DATA_DIR/<tag>) so co-located models never collide; collector globs these dirs
go() { local tag="$1"; shift; echo "LAUNCH $tag :: $*"; setsid bash -c "$* --outdir $KA_DATA_DIR/$tag > /tmp/ka_log/$tag.log 2>&1" </dev/null & disown; }

case "$ROLE" in
  # WHY-RSI mechanism probes, 1 GPU each (small/mid bf16) — 8 models across the node
  mech)
    M=(Qwen/Qwen2.5-Coder-0.5B Qwen/Qwen2.5-Coder-1.5B deepseek-ai/deepseek-coder-1.3b-base \
       infly/OpenCoder-1.5B-Base bigcode/starcoderbase-3b Qwen/Qwen2.5-Coder-3B \
       HuggingFaceTB/SmolLM2-1.7B stabilityai/stable-code-3b)
    for i in "${!M[@]}"; do
      go "rmech_$(basename ${M[$i]})_s${SEED}" "python -m kernelascent.v3.lab_rsi_mechanism --model ${M[$i]} --gpu $i --rounds 8 --seed $SEED"
    done ;;
  # WHY-RSI on BIGGER open models, 2 GPUs each (bf16 shard), 4 models/node
  mech_large)
    M=(Qwen/Qwen2.5-Coder-14B-Instruct bigcode/starcoder2-15b deepseek-ai/deepseek-coder-6.7b-base Qwen/Qwen2.5-Coder-7B)
    G=("0,1" "2,3" "4,5" "6,7")
    for i in "${!M[@]}"; do export KA_MAXMEM_GIB=20
      go "rmech_$(basename ${M[$i]})_s${SEED}" "KA_MAXMEM_GIB=20 python -m kernelascent.v3.lab_rsi_mechanism --model ${M[$i]} --gpu ${G[$i]} --rounds 8 --seed $SEED"
    done ;;
  # 3-arm OPEN self-play, small (1 GPU/arm), 2 models -> 6 GPUs
  selfplay_small)
    go "sp_deepseek1.3_s${SEED}" "python -m kernelascent.v3.lab_selfplay_rsi --model deepseek-ai/deepseek-coder-1.3b-base --s-gpu 0 --f-gpu 1 --l-gpu 2 --rounds 12 --seed $SEED"
    go "sp_qwen1.5_s${SEED}"     "python -m kernelascent.v3.lab_selfplay_rsi --model Qwen/Qwen2.5-Coder-1.5B --s-gpu 3 --f-gpu 4 --l-gpu 5 --rounds 12 --seed $SEED" ;;
  # 3-arm OPEN self-play, mid 7B (bf16 1 GPU/arm), 2 models -> 6 GPUs
  selfplay_mid)
    go "sp_qwen7b_s${SEED}"      "python -m kernelascent.v3.lab_selfplay_rsi --model Qwen/Qwen2.5-Coder-7B --s-gpu 0 --f-gpu 1 --l-gpu 2 --rounds 12 --seed $SEED"
    go "sp_deepseek6.7_s${SEED}" "python -m kernelascent.v3.lab_selfplay_rsi --model deepseek-ai/deepseek-coder-6.7b-base --s-gpu 3 --f-gpu 4 --l-gpu 5 --rounds 12 --seed $SEED" ;;
  # 3-arm OPEN self-play, large 14-15B (bf16 1 GPU/arm), 2 models -> 6 GPUs
  selfplay_large)
    go "sp_qwen14b_s${SEED}"     "python -m kernelascent.v3.lab_selfplay_rsi --model Qwen/Qwen2.5-Coder-14B-Instruct --s-gpu 0 --f-gpu 1 --l-gpu 2 --rounds 12 --seed $SEED"
    go "sp_starcoder15b_s${SEED}" "python -m kernelascent.v3.lab_selfplay_rsi --model bigcode/starcoder2-15b --s-gpu 3 --f-gpu 4 --l-gpu 5 --rounds 12 --seed $SEED" ;;
  # 3-arm OPEN self-play, XL 32B (bf16 2 GPUs/arm), 1 model -> 6 GPUs
  selfplay_xlarge)
    export KA_MAXMEM_GIB=20
    go "sp_qwen32b_s${SEED}" "KA_MAXMEM_GIB=20 python -m kernelascent.v3.lab_selfplay_rsi --model Qwen/Qwen2.5-Coder-32B --s-gpu 0,1 --f-gpu 2,3 --l-gpu 4,5 --rounds 12 --seed $SEED" ;;
  # 3-arm CLOSED self-play (API), 1 grade GPU each -> 5 frontier models
  closed)
    [ -f /tmp/ka/bedrock_creds ] && export AWS_SHARED_CREDENTIALS_FILE=/tmp/ka/bedrock_creds BEDROCK_PROFILE=bedrock
    C=(us.openai.gpt-5.6-sol us.anthropic.claude-opus-5 moonshotai.kimi-k2.5 deepseek.v3.2 us.anthropic.claude-fable-5-1)
    for i in "${!C[@]}"; do
      go "spc_$(echo ${C[$i]} | tr ./ __)_s${SEED}" "python -m kernelascent.v3.lab_selfplay_closed --model ${C[$i]} --region us-east-2 --grade-gpu $((i % 6)) --rounds 6 --seed $SEED"
    done ;;
  *) echo "unknown ROLE $ROLE"; exit 2 ;;
esac
echo "== $ROLE launched (seed $SEED); logs in /tmp/ka_log, data in $KA_DATA_DIR =="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | head
