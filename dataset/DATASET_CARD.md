---
license: mit
task_categories:
- text-generation
tags:
- gpu-kernels
- triton
- code-optimization
- recursive-self-improvement
- benchmark
- ai-r-and-d
pretty_name: KernelAscent
size_categories:
- n<1K
---

# KernelAscent — public dev split

KernelAscent is a benchmark for **recursive self-improvement (RSI)**: a model optimizes
the GPU kernels used to train itself, and we measure whether kernel-optimization
capability **compounds** across rounds. This is the public dev split, released for
self-benchmarking and research; the leaderboard is scored on a private held-out split.

- Project & code: https://github.com/ahmd-mohsin/KernelAscent
- Leaderboard & docs: https://ahmd-mohsin.github.io/KernelAscent/

## What is in a task

Each task is a self-contained, seeded PyTorch `Model` whose `forward` is a fused op-graph;
an agent must return an optimized, numerically-equivalent `ModelNew` (Triton or fused
PyTorch). Per-task files:

- `task.py` — the problem (`Model`, `get_inputs`, seeded weights).
- `meta.json` — `tier`, `family`, `tags`, shape/dtype/chain, and (curation) `achievable_speedup`, `pass_rate`, `difficulty`.
- `reference_solution.py` — the best correct + fastest kernel found by the curator (Claude Fable 5). The achievable target.
- `results.json` — full grading record (per-candidate correctness, timing, speedup vs eager and vs the `min(eager, torch.compile)` roofline).

## Families (6) and tiers

`matmul` (L2), `norm-act` (L1), `attention` (L3), `rope-attention` (L3),
`quant-gemm` (L2, int8 dequant + GEMM), `moe` (L3, gated experts / grouped GEMM).
Tiers: L1 memory/reduction, L2 tensor-core/matmul-epilogue, L3 attention & structured.

## Scoring

- Correctness against an fp32 gold, allowing no more error than the working fp16/bf16 dtype itself incurs.
- Roofline-relative speedup `t_baseline / t_candidate`, baseline = `min(eager, torch.compile)`.
- `fast_p` (fraction beating p× speedup) and `pass@k`; timing is warmup + median-of-N + L2 flush on clock-pinned GPUs.

## Provenance & contamination

Tasks are synthesized deterministically from seeds at generation time (not drawn from a
fixed public list). Public and private held-out seed ranges are disjoint; the held-out
split is never released, so leaderboard scores cannot be gamed by overfitting the public set.

## Citation

```
@misc{kernelascent2026,
  title  = {KernelAscent: Measuring Recursive Self-Improvement via a Kernel-to-Model Capability Loop},
  author = {Mohsin, Ahmed},
  year   = {2026},
  url    = {https://github.com/ahmd-mohsin/KernelAscent}
}
```
