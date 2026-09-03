# KernelAscent

**Measuring Recursive Self-Improvement via a Kernel-to-Model Capability Loop**

KernelAscent is a benchmark for recursive self-improvement (RSI). An agent optimizes
the GPU kernels used to train itself; under a fixed wall-clock budget those faster
kernels buy more effective training, producing a more capable model, which then
optimizes kernels better. The headline question is whether kernel-optimization
capability **compounds** across rounds or plateaus.

The raw capability (writing fast GPU kernels) is well studied — KernelBench,
TritonBench, RE-Bench, robust-kbench, Kevin-32B. The novel, unbenchmarked part is the
closed `kernel -> faster training -> better model -> better kernel` loop and its
measured takeoff shape. See `proposal/` for the full motivation and design.

## Repository layout

```
proposal/
  proposal.md                 Extended design doc (tiers, scoring, threat model)
  proposal.tex / proposal.pdf 2-page RSI proposal (loop C)
  rsi_bench_form_answers.md   RSI Bench submission-form answers
kernelascent/
  task_schema.py              Task object: reference, inputs, tolerance, tier, tags
  gen_tasks.py                Procedural op-graph generator (closure tasks)
  gen_source_tasks.py         KernelBench-style SOURCE tasks for LLM agents
  run_bench.py                Harness: grade a candidate vs a roofline baseline
  agent_bench.py              LLM agent benchmark (Qwen2.5-Coder), best-of-k
legacy/
  ka_harness.py               First hardcoded-softmax harness (kept for reference)
  cand_softmax_triton.py      Sample Triton candidate
results/
  agent_run2.json             First valid agent run (Qwen2.5-Coder-7B)
  results_verify.json         Roofline + scoring verification run
```

## How scoring works

- **Procedural tasks.** A task is a randomly sampled fused op-graph (matmul, pointwise,
  reduction) with randomized shape, dtype, and length, including non-power-of-2 widths.
  Tasks are generated from a seed, so a held-out seed range gives a private, hard to
  memorize split.
- **Correctness against fp32 gold.** A candidate passes when its relative L2 error to an
  fp32 gold is no worse than the reference's own fp16/bf16 rounding error, times a
  margin. This never demands more precision than the working dtype allows.
- **Roofline-relative speedup.** `speedup = t_baseline / t_candidate`, where the baseline
  is a strong automated reference (`torch.compile`). Timing uses warmup, median-of-N, an
  L2 flush, and GPU clocks pinned for reproducibility.
- **Primary metric `fast_p`.** Fraction of tasks with speedup greater than p (1, 1.5, 2).
  Geometric mean is reported over passing tasks only, so correctness failures do not
  collapse it. For agents we also report `pass@k`.

## Running

```bash
pip install "kernelascent @ git+https://github.com/ahmd-mohsin/KernelAscent"

export AWS_PROFILE=bedrock                      # Bedrock access (us-east-1)
kernelascent gen   --model us.anthropic.claude-opus-4-8 --tiers L1,L2 --out runs/opus   # API, no GPU
kernelascent grade --candir runs/opus --out runs/opus/summary.json                      # GPU
# or, on a GPU box, end-to-end:
kernelascent eval  --model qwen.qwen3-32b-v1:0 --tiers L1,L2 --out runs/qwen3-32b
```

`gen` calls the model via Bedrock and stores candidate kernels + reasoning trajectories;
`grade` runs/times them on a GPU against the `min(eager, torch.compile)` roofline. See
`QUICKSTART.md`. For reproducible timing, lock GPU clocks: `sudo nvidia-smi -lgc 1410`.

## Three tracks

KernelAscent scopes the RSI claim across three tracks, each with its own leaderboard:

1. **Capability** (all models, API or open) — kernel-optimization skill. An AI-R&D capability leaderboard, distinct from the recursive loop.
2. **Scaffold-RSI** (API-eligible) — the agent recursively improves its own optimization scaffold around a frozen model; scored by the compounding coefficient vs a frozen-scaffold control.
3. **Weight-RSI** (open-weight only) — the kernel-to-model training loop; scored by the compounding coefficient and the Δ_k control. Requires GRPO training, so API models cannot participate.

## Dataset

The curated dataset is published on both GitHub (`dataset/curated/`) and Hugging Face
(`muahmed7338/kernelascent`): **1,064 tasks, 3,130 candidate solutions** from the Claude
Fable 5 curator across 6 families (matmul, norm-act, attention, rope-attention,
quant-gemm, moe) and tiers L1–L3.

### Getting it

```bash
# Hugging Face (dataset repo)
huggingface-cli download muahmed7338/kernelascent --repo-type dataset --local-dir kernelascent-data
# or GitHub
git clone https://github.com/ahmd-mohsin/KernelAscent && ls KernelAscent/dataset
```

### Layout

- `dataset/public/` — released dev split, **problems only** (`<task>/task.py` + `meta.json` + `manifest.json`). Use it to benchmark your own models and in papers.
- `dataset/curated/` (and HF `curated/`) — full bundles: `task.py` (the problem), `cand_*.py` (Fable candidate solutions), `reference_solution.py` (best correct+fastest kernel), `results.json`, and `meta.json` (tier, family, tags, difficulty).
- **Held-out split** — a private seed range (10,000,000+), never released; it powers the leaderboards so scores can't be gamed.

### Each task

`task.py` is a self-contained, seeded `Model(nn.Module)` with `get_inputs()`. An agent must return an equivalent, faster `ModelNew`. Regenerate any split deterministically:

```bash
python kernelascent/make_dataset.py --split public --outdir dataset/public
```

### Using it

```bash
# score a model on the public split (see QUICKSTART.md)
kernelascent eval --model qwen.qwen3-32b-v1:0 --tiers L1,L2 --split public --out runs/qwen3-32b
```
Or load a `task.py` directly: `exec(open(".../task.py").read())` gives `Model` + `get_inputs`; grade any `ModelNew` against it with `kernelascent/grade_candidates.py`.

## Status

Working end to end: procedural generation, fp32-gold correctness, roofline-relative
`fast_p`, pinned-clock timing, reward-hack-resistant multi-input grading, and a
generation backend for open-weight models (local) and 76 Bedrock models (API).

The **Capability leaderboard is being populated**: Tier-1 (L1) and Tier-2 (L2) tasks are
run across all 76 Bedrock text/chat models, storing full reasoning trajectories; results
are graded on GPU against the `min(eager, torch.compile)` roofline.

## Roadmap

1. Complete the Tier-1/Tier-2 capability sweep across the 76 Bedrock models and publish the leaderboard.
2. Scaffold-RSI runs (self-improving scaffold vs frozen-scaffold control).
3. Weight-RSI: GRPO training loop on open-weight models, measuring the compounding coefficient.
