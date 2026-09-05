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

## Structure: difficulty tiers with empirical labels

The public split is organized by difficulty tier under `public/<Tier>/<task>/`:

- Easy: small power-of-two elementwise fusion or a single reduction (softmax,
  layernorm, rmsnorm). Accessible floor.
- Medium: matmul with a fused epilogue, or short fused chains.
- Hard: matmul-bearing chains, full and causal attention, RoPE attention.
- Ultra: soft-MoE and large or irregular shapes.

Every task's `meta.json` carries an empirical difficulty measured by running 13
open-weight models (Qwen2.5-Coder / Qwen2.5-Instruct 0.5B to 14B, DeepSeek-Coder-6.7B,
StarCoder2-15B, CodeLlama-13B): `solve_rate` (fraction of models that produced a
correct kernel) and `best_speedup_observed` (best speedup vs the min(eager,
torch.compile) roofline any model achieved), plus a `difficulty` label
(`speed-open`, `correctness-only`, `hard`, `unsolved`). `public/manifest.json` indexes
the whole set. Empirical difficulty distribution:

```
Easy    25 speed-open, 5 correctness-only
Medium  18 speed-open, 10 correctness-only, 2 rare
Hard    11 speed-open, 18 correctness-only, 1 hard
Ultra    8 speed-open, 16 correctness-only, 4 hard, 2 rare
```

Correctness difficulty rises monotonically Easy to Ultra. The roofline is
torch.compile, so there is real headroom above the bar at every tier (no global
optimum). See the repo `analysis/calibration_run.md` for the failure breakdown.

## How we evaluate

Correctness. A candidate `ModelNew` is checked against an fp32 gold on N=4 fresh random
inputs with a dtype-aware tolerance and an input-sensitivity check that rejects constant or
input-ignoring outputs. Correctness is verified on the timed run. Each candidate is graded
in an isolated subprocess so a native compiler abort or hang loses only that candidate.

Two walls, reported separately. Correctness rate (was a valid correct kernel produced) and
speed rate (does a correct kernel beat the roofline). We never fuse them into one number.

Speed score. Continuous log-interpolated ladder between eager, torch.compile, and an expert
kernel: `s = clip((ln t_eager - ln t_cand)/(ln t_eager - ln t_expert), 0, 1.2)`, 0 at eager,
1 at expert, compile parity as a milestone. Expert rungs are reconstructed with a strong
curator (Fable 5.1) and verified to beat torch.compile.

## How progress (RSI) is measured

Capability is the tier ladder. Recursive self-improvement is measured with campaigns: K
rounds, each a practice phase (public seeds, a persistent artifact may grow) and a transfer
phase (private seeds, artifact frozen). The central question is causal: does the agent get
better at generating future improvements. We separate a solver S_k from an improver U_k and
test, by transplant, whether U_k produces a better descendant from a common S_0 than U_0
does. Improvement is credited only when it persists, transfers to held-out tasks, and beats
controls (frozen-nonempty library, offline-built library, matched-compute search) by more
than the measured unchanged-state noise floor. Full design in the project repo
`docs/RSI_CAUSAL_PLAN.md`. The private held-out split is not released.

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
