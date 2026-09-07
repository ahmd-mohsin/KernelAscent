#!/bin/bash
# Restart the API flagship runs with the current (fixed) flagship.py. Keeps the Coder-7B run.
set -u
D=/tmp/instance_storage/ka_data/flagship
CREDS=/tmp/instance_storage/bedrock_creds
cd /tmp/instance_storage/kernelascent
pkill -f "flagship.py --api-model" 2>/dev/null
sleep 4
run(){  # name model_id gpu
  rm -rf "$D/$1"
  CUDA_VISIBLE_DEVICES=$3 setsid env AWS_SHARED_CREDENTIALS_FILE=$CREDS AWS_PROFILE=bedrock BEDROCK_PROFILE=bedrock \
    python3 -u v3/flagship.py --api-model "$2" --region us-east-1 --blocks 10 --anchor-n 2 --outdir "$D/$1" > "$D/$1.log" 2>&1 </dev/null &
}
run fable      us.anthropic.claude-fable-5-1   4
run gptoss     openai.gpt-oss-120b-1:0         5
run gpt56terra us.openai.gpt-5.6-terra         6
run kimi       moonshotai.kimi-k2.5            7
sleep 8
echo "restarted; flagship procs=$(ps aux | grep -c '[f]lagship.py')"
