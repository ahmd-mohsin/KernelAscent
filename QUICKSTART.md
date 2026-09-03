# KernelAscent — Quickstart

## Install

```bash
pip install "kernelascent @ git+https://github.com/ahmd-mohsin/KernelAscent"
# add grading (needs a GPU): pip install "kernelascent[grade] @ git+https://github.com/ahmd-mohsin/KernelAscent"
```

## Evaluate a model (one command)

Generation uses Amazon Bedrock (any of the supported API models); grading runs on a GPU.

```bash
export AWS_PROFILE=bedrock        # a profile with Bedrock access (us-east-1)

# API generation only (no GPU) — writes candidate kernels + reasoning trajectories:
kernelascent gen  --model us.anthropic.claude-opus-4-8 --tiers L1,L2 --k 1 --out runs/opus

# Grade on a GPU (correctness vs fp32 gold, speedup vs the torch.compile roofline):
kernelascent grade --candir runs/opus --out runs/opus/summary.json

# Or do both at once (needs a GPU):
kernelascent eval --model qwen.qwen3-32b-v1:0 --tiers L1,L2 --out runs/qwen3-32b
```

Output per model: `pass@k`, `fast_1 / fast_1.5 / fast_2` (fraction beating p× the roofline).

- `--tiers` : `L1` (memory/reduction), `L2` (matmul/quant), `L3` (attention/rope/moe), or a comma list.
- `--split` : `public` (released dev set) or `heldout` (private; maintainers only).
- The harness auto-resolves each model's id form (bare vs `us.` profile), its true max output tokens, and a reasoning config.

## No GPU? 

Run `gen` (API only) and open a PR with your candidate directory or `summary.json`; maintainers grade on the private held-out split (see `SUBMISSION.md`).

## Colab

A ready-to-run notebook is in `docs/quickstart.ipynb` — it pip-installs KernelAscent and evaluates a model end to end.
