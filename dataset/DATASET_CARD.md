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

![Internal-failure causality DAG](https://raw.githubusercontent.com/ahmd-mohsin/KernelAscent/main/docs/figures/cz_internal_dag.png)

*Mechanistic interpretability of RSI: every model traces one path through the internal gates it must clear — scale → correctness-wall → gradient → drift → retention → diversity → outcome (green = RSI compounds, blue = crossed the wall but flat, orange = stuck at the wall). Sub-2B models stall at the correctness wall; mid-scale (2–8B) threads every gate and compounds; the largest drift most yet saturate at the roofline.*

![Scale vs RSI gain with marginals](https://raw.githubusercontent.com/ahmd-mohsin/KernelAscent/main/docs/figures/gz_bubble.png)

*Held-out capability gain vs model size, bubble area ∝ LoRA drift. Shaded = sub-2B correctness wall; positive gain concentrates at mid-scale. Interactive versions on the [project site](https://ahmd-mohsin.github.io/KernelAscent/).*

## Key findings (updated 2026-09-18)

**Headline: self-training *sharpens* what a model already covers; it does not *explore*. In GPU-kernel
optimization, compounding is coverage-limited, and coverage is not self-generated.**

- **No compounding (strong, well-powered null).** Lineage vs.\ matched-reset over multi-round rejection-sampling
  SFT: pooled lineage$-$reset $=+0.001\,[-0.015,+0.018]$ ($n{=}213$), **TOST-equivalent at $\delta{=}0.05$,
  Bayes factor $\mathrm{BF}_{01}\approx14.5$** (strong evidence *for* the null); every scale 0.5–3B individually
  equivalent. 7B/14B extension in progress.
- **Search beats training (significant).** At matched compute, best-of-$N$ search beats lineage self-training:
  lineage$-$bestof$N=-0.113\,[-0.153,-0.074]$ (CI excludes 0); search wins 70% of rounds.
- **Coverage-vs-scale curve (0.5B→14B).** Coverage 14%→86%; below the wall pass@$K\gg$pass@1 (12–17×) — the
  sub-2B "correctness wall" is a *sampling* artifact, not absent capability.
- **0-score failure forensics (8 model families).** The sub-3B wall is a kernel-**formation** failure
  (`no_extract` 57–96%: the model never emits a valid kernel), not a correctness failure; the failure locus
  marches *downstream* with scale (incoherence → truncation → API-hallucination → wrong-output → correct).
- **Weight-level mechanism (WHY-RSI, $n{=}133$).** RSI is an **inverted-U in scale**: peaks at 2–8B (49%),
  while ≥9B shows the *lowest* RSI (19%) despite the *highest* LoRA drift (0.73) — large models churn weights
  without compounding (roofline saturation). Drift localizes to late layers.
- **Task-5 self-play (closed-source frontier models).** Self-modification gain is one-shot (68% at round 0),
  75% of models non-recursive; one model self-degrades.

Raw per-run trajectories (compounding histories, WHY-RSI series, forensics reasoning-chains, closed-source
self-modify) are in [`data/trajectories/`](https://github.com/ahmd-mohsin/KernelAscent/tree/main/data/trajectories)
for independent analysis.

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

Capability is the tier ladder. Recursive self-improvement is measured causally. A 15-model x
4-arm sweep (growing / frozen-nonempty / offline-built / matched-search) found matched-compute
search beats recursive library-growing on average (growing below its strongest control for
13 of 15 models): a clean negative for memory-RSI on this benchmark. The v3 redesign makes the
central object the causal returns to recursive improvement: separate the actor (the procedure
producing a patch) from the target (what is patched) so competing producers edit the SAME
target, and measure Q (research productivity), V (producing a better improver), and the causal
producer contrast F across a two-link lineage with rescue. A deterministic calibration suite
proves the instrument distinguishes a repeating recursive positive control from a one-time
upgrade, best-of-N, and nulls before any model is judged. Full design in the project repo
`docs/RSI_V3_PLAN.md`. The private held-out split is not released.

## Mutation-DSL bank with family-disjoint splits (`kernel_tasks_dsl_validated.json`)

The 29-task bank used for the p-maps and the probe is too small to support per-task statistics: probe AUC
swings 0.71–1.00 across draws and a "+0.45 lift" is about five problems. This bank replaces it for any
analysis that needs task-level power or a leakage-resistant split.

**456 tasks over 16 op families**, generated parametrically by crossing family × shape × dtype × variant
(`scripts/gen_task_dsl.py`), then passed through a CPU semantic gate (`scripts/validate_bank_cpu.py`).

### Splits — disjoint at the level of the *computation*, not the example

| split | n | what it tests |
|---|--:|---|
| `train` | 120 | fittable / trainable cells |
| `heldin_cell` | 108 | **in-family generalisation** — unseen dtype (float32) or unseen shape (16384×2048) of a *seen* family |
| `heldout_family` | 228 | **out-of-family transfer** — families that never appear in training at all |

Train families: `fused_elemwise`, `gelu_mlp`, `layernorm_affine`, `matmul_bias`, `rmsnorm_gate`, `sigmoid_gate`, `softmax_row`, `softplus_norm`

Held-out families: `cumsum_scale`, `groupnorm_rows`, `logsumexp_norm`, `rope_pair`, `softmax_matmul`, `swiglu_mlp`, `topk_mask`, `var_gate`

Held-out families are chosen for structurally distinct access/reduction patterns (grouped reductions,
cumulative scans, top-k gating, rotary pairing, log-sum-exp, softmax→matmul), **not** renamed twins of the
train families. A renamed template or a nearby shape is not a new family, and synthetic variants raise
coverage without raising the number of independent families — so the family count, not the task count, is
the unit that bounds a transfer claim.

The split is a fixed table in `gen_task_dsl.py`, never drawn at random, so it is stable and auditable across
regenerations.

### The CPU gate, and why it runs before any GPU time

Speed grading needs a GPU; **semantics do not** — shape and device are parameters of a task, not properties
of its computation. Every task is re-materialised at 64×128 on CPU and must build, produce finite output,
produce *non-constant* output, and not be an identity/no-op. These are the same anti-reward-hacking
conditions a self-authored task must satisfy, applied at bank-build time. Each degenerate task that instead
reaches the cluster costs ~12 s per crash-isolated grade × K candidates × rounds.

The gate ships with a **self-test that must pass before it will validate anything** (`--selftest`): nine
known-degenerate tasks — constant output, identity, NaN, Inf, non-tensor return, raising forward — must all
be rejected. A gate that never rejects anything is indistinguishable from a broken gate, so it is not
trusted until it demonstrates it can fail.

Current status: **456/456 tasks pass the CPU gate.**

### Still required before these tasks score anything (needs a GPU)

1. **Roofline ceiling** per task (`lab_roofline.py`) — the headroom-normalized score is undefined without it.
2. **Learnability / difficulty gate** (`difficulty_filter.py`, `learnability_gate.py`) — keep only tasks the
   frozen anchor sometimes solves but does not already run fast (`correct_rate > 0`, `best_score ≤ 0.75`,
   `ceiling ≥ 1.3×`), so a flat result reflects the model and not a saturated bank.
3. **Timing validation** — the CPU gate says a task *computes something*, not that it is non-trivial to
   optimise on an A100.

Until those run, this bank is **generated-and-semantically-valid, not difficulty-calibrated**, and it should
not be used to produce headline scores.

```bash
python3 scripts/gen_task_dsl.py --out dataset/kernel_bank/kernel_tasks_dsl.json
python3 scripts/validate_bank_cpu.py --selftest            # prove the gate can fail
python3 scripts/validate_bank_cpu.py \
    --in  dataset/kernel_bank/kernel_tasks_dsl.json \
    --out dataset/kernel_bank/kernel_tasks_dsl_validated.json \
    --report dataset/kernel_bank/kernel_tasks_dsl.validation.json
export KA_KERNEL_BANK=dataset/kernel_bank/kernel_tasks_dsl_validated.json
```

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
