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

## Difficulty tiers and empirical calibration

Tasks are organized into four difficulty tiers, and every task carries an empirical
difficulty label measured by running 13 open-weight models (Qwen2.5-Coder and
Qwen2.5-Instruct 0.5B to 14B, plus DeepSeek-Coder-6.7B, StarCoder2-15B,
CodeLlama-13B) as test-takers on a 24 GPU fleet.

- Easy: small power-of-two elementwise fusion or a single reduction (softmax,
  layernorm, rmsnorm). The accessible floor, solvable by 1.5B and up.
- Medium: matmul with a fused epilogue, or short fused chains at moderate shapes.
- Hard: matmul-bearing fusion chains, full and causal attention, RoPE attention.
- Ultra: soft-MoE and large or irregular shapes for frontier headroom.

Two failure walls are reported separately, because models fail in two different ways.
Correctness (can the model emit a valid, correct kernel) is the binding constraint
below roughly 14B; speed (does a correct kernel beat the min(eager, torch.compile)
roofline) is the binding constraint above it. Mean correctness% / beats-roofline%
by model-size band:

```
band            Easy       Medium     Hard       Ultra
small (<=3B)    46 / 23    38 / 13    36 / 4     31 / 3
mid   (6-8B)    80 / 28    67 / 12    54 / 6     38 / 6
large (13-15B)  73 / 30    70 / 22    63 / 7     52 / 11
```

Correctness falls monotonically Easy to Ultra in every band and rises with model
size, so the ladder is calibrated. The roofline is torch.compile, which is not
optimal (expert kernels beat it), so there is genuine headroom above the bar at every
tier and no global optimum, which is what an RSI loop is meant to climb into. See
`analysis/calibration_run.md` for the full failure breakdown and the reasons
self-improvement does not yet compound.

The public split lives in `dataset/public/<Tier>/<task>/` with `task.py` plus a
`meta.json` carrying the tier, family, shapes, and the empirical difficulty
(`solve_rate` and `best_speedup_observed` across the 13 models). `dataset/public/manifest.json`
indexes the set.

## How we evaluate

Correctness gate. A candidate `ModelNew` is checked against an fp32 gold reference on N=4
fresh random inputs with a relative L2 tolerance no tighter than the working dtype's own
rounding error, and an input-sensitivity check that rejects constant or input-ignoring
outputs. Correctness is verified on the same run that is timed, so a fast wrong path cannot
score. Each candidate is graded in an isolated subprocess, so a native compiler abort or a
hang loses only that candidate, never the run. `kernelascent/test_grader.py` asserts all of
this on the GPU (correct passes, wrong fails, reward-hack rejected, erroring isolated).

Two walls, reported separately. Models fail in two distinct ways and we never fuse them
into one number. Correctness rate is whether a valid correct kernel was produced at all
(the binding constraint below roughly 14B). Speed rate is whether a correct kernel beats
the roofline (the binding constraint above it).

Speed score, a slope not a cliff. Speed is scored on a continuous log-interpolated ladder
between three rungs, eager, `torch.compile`, and an expert kernel:
`s = clip((ln t_eager - ln t_cand) / (ln t_eager - ln t_expert), 0, 1.2)`, so 0 is eager
parity and 1 is expert parity, with compile parity as a milestone. The expert rungs are
reconstructed with a strong curator (Fable 5.1) and verified to beat `torch.compile`. This
gives incremental speedups rising credit rather than a single pass or fail at the compiler
bar (`kernelascent/scoring.py`).

## How we measure progress (RSI)

Raw capability is the tier ladder above. Recursive self-improvement is measured with
campaigns, not one-shot scores. A model runs K rounds; each round has a practice phase
(public seeds, it may grow a persistent artifact) and a transfer phase (private seeds,
artifact frozen). The transfer score across rounds is the signal. Improvement is credited
only when it persists in an artifact, transfers to held-out tasks, and beats controls.

The central question is causal: does the agent become better at generating future
improvements, and which inherited changes cause that. We separate a checkpoint into a solver
`S_k` (prompts, reusable kernels, retrieval, tools) and an improver `U_k` (how practice is
chosen, failures analyzed, edits proposed, artifacts admitted). The decisive experiment is a
transplant: does the later improver `U_k` produce a better descendant from a common starting
solver `S_0` than `U_0` does. A rising task score alone is not enough, because it conflates
task capability, transferable memory, extra compute, and improved improvement ability.

Controls make the claim honest: a frozen-nonempty library (isolates growing from having), an
offline-built library (recursion vs ordinary construction), matched-compute search (rules out
more sampling), and a measured unchanged-state noise floor (at small n, sampling alone moves
the score, so effects must clear that floor). The permission levels L0 to L5 are an
edit-permission taxonomy, not intrinsic depths of recursion; depth is measured by intervening
on artifact ancestry. Full design in `docs/RSI_CAUSAL_PLAN.md`; findings and honest
corrections in `analysis/` (`EVALUATION_REPORT.md`, `l2_result.md`, `phase0_exit.md`).

## Current status

Established: capability calibration across 13 models (the two walls), and a preliminary
memory-transfer signal at L2 that is not yet separated from noise or from the controls. Not
yet established: that any of it is recursive improvement. The causal protocol above is what
resolves that, and its first experiment (growing vs frozen vs offline vs search, with
independent campaigns) is what the project is running now.

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
