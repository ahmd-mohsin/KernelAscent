#!/bin/bash
# E0 capability ladder — large Bedrock API models. API gen + GPU grading. k=5, fixed 15 tasks.
cd /tmp/instance_storage/kernelascent
D=/tmp/instance_storage/ka_data/cap5_api; mkdir -p $D
CREDS=/tmp/instance_storage/bedrock_creds
run(){ # slug  model_id  gpu
  CUDA_VISIBLE_DEVICES=$3 setsid env AWS_SHARED_CREDENTIALS_FILE=$CREDS AWS_PROFILE=bedrock BEDROCK_PROFILE=bedrock \
    python3 -u v3/capcheck.py --api-model "$2" --region us-east-1 --n 15 --k 5 --outdir $D/$1 > $D/$1.log 2>&1 </dev/null &
}
run gpt56-sol        us.openai.gpt-5.6-sol                     0
run gpt56-terra      us.openai.gpt-5.6-terra                   1
run gpt-oss-120b     openai.gpt-oss-120b-1:0                   2
run llama33-70b      us.meta.llama3-3-70b-instruct-v1:0        3
run llama4-maverick  us.meta.llama4-maverick-17b-instruct-v1:0 4
run qwen3-32b        qwen.qwen3-32b-v1:0                       5
run qwen3-next-80b   qwen.qwen3-next-80b-a3b-v1:0              6
run deepseek-v32     deepseek.v3.2                             7
run mistral-large3   mistral.mistral-large-3-675b-instruct     0
run nova-pro         amazon.nova-pro-v1:0                      1
run kimi-k25         moonshotai.kimi-k2.5                      2
run minimax-m25      minimax.minimax-m2.5                      3
run nemotron-super3  nvidia.nemotron-super-3-120b             4
run palmyra-x5       us.writer.palmyra-x5-v1:0                 5
run gemma3-27b       google.gemma-3-27b-it                     6
sleep 6; echo "launched capcheck procs=$(ps aux|grep -c [c]apcheck)"
