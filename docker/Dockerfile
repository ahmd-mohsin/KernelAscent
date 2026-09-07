# KernelAscent RSI-VERIFY-01 benchmark image (GPU-capable; API-only works CPU-only too).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /bench
COPY docker/requirements.txt /bench/requirements.txt
RUN pip3 install --no-cache-dir -r /bench/requirements.txt

COPY kernelascent /bench/kernelascent
ENV PYTHONPATH=/bench:/bench/kernelascent

# Default: run the deterministic self-checks (no model, no GPU, no creds needed).
#   docker run --rm ka-rsi                      -> gate2 + calib
#   docker run --rm ka-rsi --gate2              -> opportunity proof only
#   API panel (mount creds):
#     docker run --rm -e AWS_SHARED_CREDENTIALS_FILE=/creds -e AWS_PROFILE=bedrock \
#       -v /path/creds:/creds ka-rsi --panel --api-model us.anthropic.claude-fable-5-1 --outdir /out
#   GPU panel:
#     docker run --rm --gpus all -v $HF:/hf -e HF_HOME=/hf ka-rsi --panel --model Qwen/Qwen2.5-Coder-7B-Instruct --outdir /out
ENTRYPOINT ["python3", "-u", "kernelascent/v3/rsi_verify.py"]
CMD ["--gate2"]
