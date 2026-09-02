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

Requires an NVIDIA GPU, PyTorch, and Triton (ships with recent PyTorch).

```bash
# Baseline harness: torch.compile candidate vs torch.compile max-autotune roofline
python kernelascent/run_bench.py --n 20 --candidate compile --baseline compile_max --out results/run.json

# Agent benchmark: an LLM writes ModelNew, graded best-of-k
python kernelascent/agent_bench.py --n 10 --k 8 --out results/agent.json
```

Lock GPU clocks for reproducible timing (needs privilege):

```bash
sudo nvidia-smi -lgc 1410
```

## Status

Working and validated end to end: procedural generation, fp32-gold correctness,
roofline-relative `fast_p`, pinned-clock timing, KernelBench-style source tasks, and a
Qwen2.5-Coder-7B agent graded best-of-k with Triton support (candidates are imported
from real files so `triton.jit` can introspect source).

First valid agent result (Qwen2.5-Coder-7B, n=10, k=8, eager baseline):
`pass@8 = 0.20`, `fast_1 = 0.10`. The benchmark is hard and discriminative — a 7B agent
is far from the ceiling, which is the headroom an RSI benchmark needs.

## Roadmap

1. Multi-GPU parallel evaluation (shard tasks across GPUs).
2. Best-of roofline baseline (min over eager / compile / max-autotune).
3. Real-bottleneck task families (attention, MoE, paged KV-cache).
4. vLLM rollouts plus a custom GRPO loop to land the first RL training step.
5. The L5 recursion: kernel to model to kernel, measuring the compounding coefficient.
