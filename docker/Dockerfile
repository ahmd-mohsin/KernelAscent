# KernelAscent standard evaluation image (GPU-capable; API models work CPU-only).
# One command to score a model on the PUBLIC benchmark and emit a submittable scorecard.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /bench
COPY docker/requirements.txt /bench/requirements.txt
RUN pip3 install --no-cache-dir -r /bench/requirements.txt

# code + the bundled PUBLIC task banks (held-out splits are NOT in the image)
COPY kernelascent /bench/kernelascent
COPY dataset/kernel_bank /bench/ka_data/kernel_bank
ENV PYTHONPATH=/bench:/bench/kernelascent KA_DATA_DIR=/bench/ka_data KA_ROOT=/bench KA_SCORE=compiled

# ---------------------------------------------------------------------------
# Self-check (no model / GPU / creds): deterministic gates must pass.
#   docker run --rm ka                 -> --track capability needs a model, so default = self-check
#   docker run --rm --entrypoint bash ka docker/entrypoint_selfcheck.sh
#
# Evaluate an API model (mount Bedrock creds; no GPU):
#   docker run --rm -e AWS_SHARED_CREDENTIALS_FILE=/creds -e AWS_PROFILE=bedrock \
#     -v $PWD/creds:/creds -v $PWD/out:/out ka \
#     --track capability --api-model us.anthropic.claude-opus-5 --tier medium
#   docker run --rm ... ka --track rsi --api-model us.anthropic.claude-opus-5 --lineages 16
#
# Evaluate an open-weight model on GPU (any track), scored with the headroom-normalized metric:
#   docker run --rm --gpus all -v $HF:/hf -e HF_HOME=/hf -v $PWD/out:/out ka \
#     --track rsi --model Qwen/Qwen2.5-Coder-7B-Instruct --bank bank_mid.json --gpus 0,1 --rounds 5
#   docker run --rm --gpus all ... ka --track baselines --model <hf-id> --bank bank_small.json --gpus 0
#   docker run --rm ... -e AWS_PROFILE=bedrock -v $PWD/creds:/creds ka \
#     --track combined --trainee <hf-id> --researcher us.anthropic.claude-fable-5-1 --trainee-gpu 0 --ctrl-gpu 1
#
# Output: scorecard JSON under KA_DATA_DIR  (open a model-submission issue; maintainers verify on the held-out split)
# ---------------------------------------------------------------------------
ENTRYPOINT ["python3", "-u", "-m", "kernelascent.v3.run_track"]
CMD ["--help"]
