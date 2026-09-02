# KernelAscent — Data & File Architecture

This document defines the canonical layout, file schemas, families, and pipeline so
inputs and outputs stay consistent across generation, curation, grading, and the site.

## Repository layout

```
kernelascent/            pipeline code
  task_schema.py           Task dataclass (closure form)
  gen_source_tasks.py      procedural generator (source-form tasks) + families + splits helper
  splits.py                public / held-out seed ranges
  make_dataset.py          materialize a split to disk
  curate_bedrock.py        Bedrock curator driver (Mac): generate candidate solutions
  propose_tasks.py         Bedrock proposer (Mac): invent novel problems (augment)
  validate_tasks.py        contamination firewall: validate + dedup proposed problems
  grade_candidates.py      GPU grader: produce complete bundles
  agent_bench.py           agent benchmark (Qwen local / Bedrock) + harness primitives
  aggregate.py             roll up shard results by family / tier
  run_bench.py             baseline harness (torch.compile candidates)
  launch_parallel.sh       one agent worker per GPU
dataset/
  public/                  released dev split (task.py + meta.json per task + manifest.json)
  heldout/                 PRIVATE — never committed (gitignored)
docs/                     website (GitHub Pages)
proposal/                 proposal (md/tex/pdf) + RSI-bench form answers
results/                  sample run summaries
```

## Task families (6)

| family          | tier | pattern |
|-----------------|------|---------|
| `matmul`        | L2   | GEMM + fused epilogue (bias / activation / norm) |
| `norm-act`      | L1   | pointwise + reduction chains (softmax / LayerNorm / RMSNorm / GELU) |
| `attention`     | L3   | QKV projections + scaled dot-product softmax (causal / non-causal) |
| `rope-attention`| L3   | rotary positional embedding on Q,K + attention |
| `quant-gemm`    | L2   | int8 weight dequantize (per-column scale) + matmul + bias + GELU |
| `moe`           | L3   | router-softmax + E gated expert GEMMs + weighted sum (soft top-1) |

Tiers: **L1** memory/reduction · **L2** tensor-core / matmul-epilogue · **L3** attention & structured.

## Input file: `task.py` (the PROBLEM)

A self-contained, deterministic PyTorch module. Contract:

- Module-level `SEED`, shape vars (e.g. `M, D`), and `DT` (`torch.float16` or `torch.bfloat16`).
- `class Model(nn.Module)` with `__init__(self, dtype=DT)` that seeds `torch.Generator().manual_seed(SEED)`
  and creates every weight as `nn.Parameter(..., requires_grad=False)`.
- `forward(self, x)`: pure functional tensor ops, numerically stable, returns one tensor.
- `def get_inputs()`: returns `[tensor]` built from `torch.Generator().manual_seed(SEED + 12345)`.

An agent must return `class ModelNew` with the **same `__init__`** and an optimized, numerically
equivalent `forward` (Triton or fused PyTorch).

## `meta.json` (task labels)

```json
{
  "name": "t2_512x513",
  "tier": "L2",
  "family": "matmul",
  "tags": ["tiling", "tensor-core", "online-softmax", "reduction", "elementwise-fusion"],
  "meta": {"M": 512, "D": 513, "dtype": "torch.float16", "chain": ["matmul->1024","softmax","gelu","rmsnorm","softmax"]},
  "achievable_speedup": 1.44,   // added by grader (best roofline-relative speedup)
  "pass_rate": 1.0,             // added by grader
  "difficulty": "medium"        // added by grader
}
```

## Output bundle (produced by `grade_candidates.py`)

Each graded task directory is a complete unit:

```
<task>/
  task.py               the problem
  cand_0.py … cand_k.py candidate solutions (from the curator / agent)
  reference_solution.py best correct + fastest candidate (achievable target)
  results.json          full grading record (below)
  meta.json             labels + achievable_speedup / pass_rate / difficulty
  DONE                  checkpoint marker (candidate count)
```

`results.json`:
```json
{
  "name": "...", "tier": "L2", "family": "matmul", "tags": [...], "meta": {...},
  "ref_err": 1.6e-3, "bound": 3.2e-3,
  "t_eager": 0.20, "t_compile": 0.14, "t_roofline": 0.14,
  "n_cand": 3, "pass_at_k": 3, "best_speedup_roofline": 1.44,
  "difficulty": "medium",
  "best": {"file": "cand_1.py", "speedup_vs_roofline": 1.44, "speedup_vs_eager": 2.1, ...},
  "candidates": [{"file","ok","err","t_cand","speedup_vs_eager","speedup_vs_roofline","reason"}, ...]
}
```

Difficulty labels: `no_candidates` (curator produced no parseable solution) · `frontier`
(candidates but none correct) · `hard` (correct, ≤1× roofline) · `medium` (1–1.5×) ·
`accessible` (>1.5×). Speedup is roofline-relative: `t_roofline / t_cand`, roofline = `min(eager, torch.compile)`.

## Splits

- **public** (`splits.PUBLIC`, seed0 = 0): released in `dataset/public/`, for self-benchmarking and papers.
- **heldout** (`splits.HELDOUT`, seed0 = 10,000,000): PRIVATE, never committed; powers the leaderboard.
- Seed ranges are disjoint across splits and per family (fusion `seed0`, attention `+100000`,
  RoPE `+200000`, quant `+300000`, MoE `+400000`).

## Pipeline

```
gen_source_tasks / make_dataset   →  task.py + meta.json          (problems)
curate_bedrock  (Bedrock, Mac)    →  cand_*.py                    (solutions)
grade_candidates (GPU box)        →  reference_solution.py + results.json + difficulty   (bundle)
aggregate                         →  rollups by family / tier
docs/data/leaderboard.json        →  website leaderboard          (test-taker models)
```
