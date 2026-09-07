# KernelAscent RSI-VERIFY-01 — containerized

Standardized RSI-under-verified-opportunity task: a bug-fix agent whose LOCAL verifier gates its
patch selection. Improving the verifier (more inputs + edge coverage) is a measurable procedural
improvement; edge-subtle mutant distractors make the opportunity realizable for real models.

## Build
```
docker build -t ka-rsi -f docker/Dockerfile .
```

## Run
Deterministic self-check (no model, GPU, or creds — good for CI):
```
docker run --rm ka-rsi                    # Gate 2 opportunity proof
docker run --rm --entrypoint bash ka-rsi docker/entrypoint_selfcheck.sh   # gate2 + calib
```
API model panel (mount Bedrock creds, never bake them in):
```
docker run --rm -e AWS_SHARED_CREDENTIALS_FILE=/creds -e AWS_PROFILE=bedrock \
  -v /host/bedrock_creds:/creds -v /host/out:/out \
  ka-rsi --panel --api-model us.anthropic.claude-fable-5-1 --K 8 --reps 8 --inject 1 --outdir /out
```
GPU (open-weight) model panel:
```
docker run --rm --gpus all -v /host/hf:/hf -e HF_HOME=/hf -v /host/out:/out \
  ka-rsi --panel --model Qwen/Qwen2.5-Coder-7B-Instruct --K 8 --reps 8 --inject 1 --outdir /out
```

## Trust boundary
The official hidden grader, oracle references, and resource meter are inside the image and immutable;
the agent's local verifier params are what improve. Creds are mounted at runtime, never in the image.

## Output
`<outdir>/panel.json`: {meanC_noverifier, meanC_weak, meanC_strong, verifier_dQ{mean,ci95}} per model.
