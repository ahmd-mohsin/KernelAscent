# KernelAscent — dockerized standard evaluation

A single image that bundles the code + the **public** curated task bank and scores a model behind one
`evaluate` entrypoint. The held-out split is **not** in the image; maintainers score it on submissions.

## Build

```bash
docker build -t ka -f docker/Dockerfile .
```

## Self-check (no model / GPU / creds)

```bash
docker run --rm --entrypoint bash ka docker/entrypoint_selfcheck.sh
# gate2 + verifier calib + rsi_true calib + efficiency-lab calib + evaluate import
```

## Evaluate a model → `/out/scorecard.json`

**API model (Bedrock; no GPU).** Mount credentials read-only:

```bash
docker run --rm \
  -e AWS_SHARED_CREDENTIALS_FILE=/creds -e AWS_PROFILE=bedrock \
  -v $PWD/creds:/creds:ro -v $PWD/out:/out ka \
  --track capability --api-model us.anthropic.claude-opus-5 --tier medium

docker run --rm -e AWS_SHARED_CREDENTIALS_FILE=/creds -e AWS_PROFILE=bedrock \
  -v $PWD/creds:/creds:ro -v $PWD/out:/out ka \
  --track rsi --api-model us.anthropic.claude-opus-5 --lineages 16
```

**Open-weight model (GPU).** Mount an HF cache:

```bash
docker run --rm --gpus all -v $HF_HOME:/hf -e HF_HOME=/hf -v $PWD/out:/out ka \
  --track capability --model Qwen/Qwen2.5-Coder-7B-Instruct --tier hard
```

## Tracks

| track | what it scores | key fields |
|---|---|---|
| `capability` | curated code-task bank per tier: correct patch produced + verifier-selected | `oracle_pool_success`, `selected_C_strong`, `verifier_dQ` |
| `rsi` | efficiency-of-experimentation lab: does the model improve its research procedure with experience, and does it compound | `q1_minus_q0`, `N1`, `F1`, `F2` (with CIs) |

If the bundled dataset is absent the entrypoint pulls the public split from
`muahmed7338/kernelascent-tasks` on HuggingFace.

## Submit

Open a [model-submission issue](https://github.com/ahmd-mohsin/KernelAscent/issues/new?template=model-submission.yml)
with your `scorecard.json`. Maintainers re-run on the private held-out split and add your row. Credentials
are mounted at runtime and never baked into the image.
