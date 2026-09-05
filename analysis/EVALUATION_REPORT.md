# KernelAscent / RSI-Depth: Complete Evaluation Report

A benchmark for recursive self-improvement (RSI) on GPU kernel optimization. This report
covers what the benchmark is for, how the dataset is built, the tier structure, the
evaluation methodology, every experiment run to date, the results, and a grounded analysis
of where and why agents fail, especially small open-weight models.

Status as of this report: capability calibration and the RSI protocol pilots are complete;
the L2 skill-memory result is a promising signal awaiting attribution controls.

---

## 1. Objective

Most self-improvement results report that a score rose with more iterations. That conflates
four different things: search, extra compute, memorization, and genuine learning. The
objective of this benchmark is to separate them and to answer three questions that a raw
capability benchmark cannot.

Depth. How much of itself can a model modify (a text playbook, then verified skills, then
its own tools, then its own solve loop) and still produce gains that compound across rounds
and transfer to tasks it never practiced on.

Reality. Do those gains make a real model run faster end to end, and do they survive a
change of compiler, dtype, shape, or GPU.

Attribution. Could the same gain have been produced by search at the same compute budget,
by a placebo artifact, or by simply re-running the first-round output. If yes, it is not RSI.

The substrate is GPU kernel optimization because it is real (faster kernels have direct
economic value), it is cheaply and objectively gradeable (correctness against a gold
reference, speed against a timed baseline), and it is contamination resistant (tasks are
generated procedurally from seeds).

---

## 2. What the dataset is and how it is built

### 2.1 Task unit

A task is a self-contained PyTorch module: a `Model` whose `forward` is a fused op-graph
with seeded weights, a `get_inputs` function, and a working dtype. An agent must return a
numerically equivalent `ModelNew` (Triton or fused PyTorch) that runs faster. Each task
carries a family, a tier, the shapes and dtype, and after grading an empirical difficulty
label and an achievable speedup.

### 2.2 Six families

matmul, norm-act (elementwise and reductions such as softmax, layernorm, rmsnorm),
attention (full and causal), rope-attention, quant-gemm, and moe. These span the operations
that dominate real transformer training and inference.

### 2.3 Procedural generation and contamination resistance

Tasks are generated from a seed: shapes, dtype, fusion pattern, and epilogue are all
functions of the seed, including non-power-of-two widths. A public seed range is released
for self-benchmarking; a disjoint private seed range (10,000,000 and up) is held out and
never published, so leaderboard tasks cannot be memorized. Because tasks are generated, the
set can grow without bound and seeds can be rotated to refresh the leaderboard.

### 2.4 Curation and the expert rung

The benchmark needs a strong reference for each task. We curate with a strong model
(Fable 5.1) at high token budget with reasoning enabled, generating several candidate
kernels per task, which are then graded on GPU. The best correct-and-fast candidate becomes
the task's reference (its expert rung). On the Medium band, 134 tasks were curated and
112 produced an expert that beats `torch.compile`, so the expert rung is a genuine target
above the compiler, not just a restatement of it.

### 2.5 Public split layout

`dataset/public/<Tier>/<task>/` with `task.py` and a `meta.json` carrying tier, family,
shapes, and empirical difficulty (`solve_rate` and `best_speedup_observed` across the tested
models), indexed by `dataset/public/manifest.json`. Published to GitHub and to Hugging Face
(`muahmed7338/kernelascent`).

---

## 3. Tiers and the depth ladder

Two distinct axes. Do not confuse them.

### 3.1 Capability tiers (static difficulty)

Easy. Small power-of-two elementwise fusion or a single reduction (softmax, layernorm,
rmsnorm), no matmul. The accessible floor.

Medium. Matmul with a fused epilogue, or short fused chains at moderate shapes.

Hard. Matmul-bearing fusion chains, full and causal attention, RoPE attention.

Ultra. Soft-MoE and large or irregular shapes, for frontier headroom.

Empirical difficulty distribution of the 120-task public core (from running the model
ladder as test-takers):

    Easy    25 speed-open, 5 correctness-only
    Medium  18 speed-open, 10 correctness-only, 2 rare
    Hard    11 speed-open, 18 correctness-only, 1 hard
    Ultra    8 speed-open, 16 correctness-only, 4 hard, 2 rare

Correctness difficulty rises monotonically Easy to Ultra, confirming the tiers are
calibrated. speed-open means at least one model produced a correct kernel that beat the
roofline; correctness-only means solvable but nobody beat the roofline; hard means rarely
even solved.

### 3.2 RSI depth ladder (how much the agent may self-modify)

L0. Nothing persists. The same budget spent as best-of-N directly on the transfer set. The
matched-compute search baseline.

L1. A size-capped text playbook of strategies.

L2. L1 plus a library of verified reusable code blocks, each earned on a practice task.

L3. L2 plus the agent's own tools (tester, profiler wrapper, autotune generator, retriever).

L4. L3 plus the agent's own solve loop (prompts, decomposition, self-authored practice).

L5. L4 plus LoRA-from-wins on verified solutions. Open-weight only. The single arm with a
weight update. Deferred.

Each level is a strict superset of the one below and acts as its control. The headline is
d*, the deepest level that still yields a compounding, transferable gain.

Built so far: L0 and L2. Not yet built: L1, L3, L4, L5.

---

## 4. Evaluation methodology

### 4.1 Correctness gate

A candidate is correct only if, on N=4 fresh random inputs, its relative L2 error against an
fp32 gold is within a tolerance no tighter than the working dtype's own rounding error, and
its outputs actually depend on the input (an input-sensitivity check that rejects constant
or no-op outputs). Correctness is checked on the same run that is timed, so a fast wrong
path cannot score. This defeats the common hacks: hardcoding the fixed input's output,
constant outputs, and seed-specific tricks.

### 4.2 Two walls, reported separately

Correctness rate is whether a valid correct kernel was produced at all. Speed rate is
whether a correct kernel beats the roofline. These are never fused into one number, because
models fail at two different walls (section 6).

### 4.3 Speed score, a slope not a cliff

Speed is scored on a continuous log-interpolated ladder between three rungs, eager,
`torch.compile`, and expert:

    s = clip( (ln t_eager - ln t_cand) / (ln t_eager - ln t_expert), 0, 1.2 )

so 0 is eager parity, 1 is expert parity, and compile parity is a reported milestone. This
replaced an earlier pass-or-fail-at-compile-parity reward that was a cliff, giving a frozen
model no gradient of small wins to climb. Timing uses warmup, median-of-N, an L2 flush, and
GPU application clocks pinned to 1410 MHz for reproducibility.

### 4.4 Robust grading

Each candidate is graded in an isolated subprocess. A native compiler abort (a SIGABRT from
the Triton or MLIR backend, which a Python-level timeout cannot catch) or a total hang loses
only that one candidate, never the run. A self-test (`test_grader.py`) asserts on GPU that a
correct candidate passes, a wrong one fails, a reward-hack constant is rejected, an erroring
one is isolated, and the run survives, `pass_at_k` is exact. All invariants pass.

### 4.5 RSI measurement (campaigns)

A campaign is K rounds. Each round has a practice phase (public seeds, the agent may grow a
persistent artifact) and a transfer phase (private seeds, artifact frozen, no learning). The
transfer score C_k across rounds is the RSI signal. Every phase is a fresh process with no
carried context, so the only thing that persists is the artifact. Improvement is credited
only when it persists, transfers to held-out tasks, and beats the controls (matched-compute
search, a re-run of the round-0 artifact, and a poisoned artifact).

---

## 5. Experiments run

| # | Experiment | Scope | Purpose |
|---|---|---|---|
| E1 | Capability calibration | 13 open-weight models (Qwen2.5-Coder and Qwen2.5-Instruct 0.5B to 14B, DeepSeek-Coder-6.7B, StarCoder2-15B, CodeLlama-13B), 4 tiers, 24 GPUs | Where does each tier land, why do models fail |
| E2 | Fable 5.1 gold curation | Easy and Medium bands | Strong-curator reference and expert rungs |
| E3 | Scaffold-RSI run 1 | 26 Bedrock API models, text strategy library, 4 rounds | First RSI attempt (frozen weights, knowledge memory) |
| E4 | Open-weight RSI sweep | Qwen ladder 0.5B to 14B, per-task feedback loop | Does in-context self-refinement compound |
| E5 | Phase 0 exit campaign | Coder-3B/7B/14B, Qwen2.5-7B-Instruct, Medium | Does the reward fix (log-interp + keep-best) remove the decline |
| E6 | L2 skill-memory campaign | Coder-7B/14B (open), Fable 5.1, Opus 5 (API), Medium, private-seed transfer | Does verified skill memory produce a rising, transferable slope |

---

## 6. Results and where agents fail

### 6.1 Capability calibration (E1): the two walls

Mean correctness% / beats-roofline% by model-size band:

    band            Easy       Medium     Hard       Ultra
    small (<=3B)    46 / 23    38 / 13    36 / 4     31 / 3
    mid   (6-8B)    80 / 28    67 / 12    54 / 6     38 / 6
    large (13-15B)  73 / 30    70 / 22    63 / 7     52 / 11

Global candidate outcome mix over 4,016 graded candidates:

    correct                             29.8%
    compile error (Triton / inductor)   23.2%
    runtime error                       10.4%
    wrong API (hallucinated tl.* etc.)   9.2%
    wrong output                         7.1%
    type / name / syntax errors         10.4%
    other                                9.9%

The correctness wall (small and mid models). About 70% of candidates never produce a correct
kernel, and the single biggest cause is compilation failure. Below roughly 14B, the binding
constraint is writing valid, compilable Triton: models emit plausible-looking kernels that
do not compile (bad `tl.*` calls, wrong grid, block, and stride configuration, type
mismatches), invent APIs that do not exist (about 9%), or produce malformed code. The
smallest model (0.5B) is below the floor everywhere (7 to 20% correct on Easy), which is an
honest out-of-scope finding, not a tier bug.

The speed wall (strong models). When a model does write a correct kernel, beating the
`torch.compile` roofline is the constraint. Even the large open models beat it only 22 to
30% of the time on Easy and Medium, and 3 to 11% on Hard and Ultra. `torch.compile` and
cuBLAS are already near optimal, so a correct hand-written kernel usually ties or loses.

Cross-family confirmation. DeepSeek-Coder, StarCoder2, and CodeLlama land in the same bands
as Qwen, so the walls are a property of the task, not of one model family.

### 6.2 Why small open-source models fail, specifically

They fail at the correctness wall, and the failure is concentrated in code validity rather
than reasoning. In order of impact: they cannot reliably emit Triton that compiles (the 23%
compile-error rate is dominated by sub-14B models); they hallucinate kernel APIs and then
reuse them; they produce correct but trivial kernels that just call torch and so sit at
eager parity (scoring zero on the eager-to-expert ladder); and the very small ones (0.5B to
1.5B) often fail to emit an extractable `ModelNew` at all. A concrete example: a 7B model
told to speed up a softmax wrote a `torch.utils.cpp_extension.load` call referencing a
`fused_ops.cpp` file it never created, which parses but cannot run. The limiter is
generation skill, not knowledge of what to do.

### 6.3 Scaffold-RSI run 1 (E3): no compounding, and two failure modes

Across 26 API models with a self-grown text strategy library and a frozen-library control,
no model showed compounding. 17 never opened the channel, 7 plateaued, 2 degraded. Two
mechanisms. For frontier models the library was clean and reasonable but did not help,
because their limiter is execution not knowledge (they know to call SDPA, they just cannot
reliably beat the compiler). For weaker models the library was actively poisoned: reasoning
models dumped chain-of-thought into it instead of strategies, and weak models banked
hallucinated APIs and then followed them, which produced the two degrade cases. Both
extraction and grounding bugs were subsequently fixed (sanitizer plus an API denylist).

### 6.4 Open-weight RSI sweep (E4): the loop degraded models

The per-task in-context refinement loop degraded models round over round rather than
improving them. Round 0 (the clean first attempt) was the peak; feedback-driven revision
broke working kernels when told to go faster, or failed to repair broken ones. Two causes.
First, the loop had no persistence channel (frozen weights, one task, replace the answer),
so nothing could compound by construction. Second, the reward was a cliff at compile parity,
so there was no gradient of small correctness-preserving improvements from a safe correct
kernel to a faster one; the model had to make a discontinuous jump to expert Triton and
broke correctness attempting it.

### 6.5 Phase 0 exit (E5): the reward fix works, L0 still does not compound

With keep-best (never regress off a correct kernel) and the log-interpolated score, the
destructive decline is gone: best-so-far is monotone for every model. But the slope is still
flat (Coder-7B and 3B scored 0 across all rounds, 14B reached 0.08, Qwen2.5-7B-Instruct held
0.30 from round 0). This is the expected L0 baseline. L0 has no persistence, so a flat slope
is exactly the control the depth ladder is built against.

### 6.6 L2 skill memory (E6): the first rising slope, pending controls

Transfer score C_k per round on the private-seed Medium set (0 = eager, 1 = expert):

    agent            C_0    C_1    C_2    C_3    skills banked
    Fable 5.1        0.01   0.175  0.203  0.20   17
    Opus 5           0.201  0.153  0.159  0.163  21
    Coder-14B        0.001  0.0    0.16   0.025  5
    Coder-7B         0.0    0.15   0.0    0.0    0

Fable 5.1 shows the first rising L2 slope, climbing from an empty-library baseline near 0.01
to about 0.20 as verified skills accumulate and transfer to held-out tasks. Opus 5 is
ceiling limited (already 0.20 at C_0, the library adds nothing). Open-weight cannot bootstrap
at L2: Coder-7B banked zero skills because its kernels never beat eager, so there was nothing
worth banking. L2 self-improvement therefore looks capability windowed: it needs an agent
strong enough to bank correct fast skills and with headroom left to climb. Fable is in that
window, Opus is above it, the open models are below it.

Not yet established. n=8 transfer is noisy and the attribution controls have not run. Before
this is an RSI claim it must beat a frozen-library control (growing matters, not just
having), a poisoned-library control (real skills, not placebo), and matched-compute L0 (not
just more sampling), with a confidence interval from larger n and a second seed.

---

## 7. What we are trying to do, in one paragraph

We are building the benchmark that tells you, for a given model, how deep it can modify
itself before self-improvement stops compounding (d*), whether that improvement is real and
transfers, and whether it is learning rather than search. The kernel substrate makes the
signal clean and the artifacts downloadable. The current evidence says raw capability is
gated by a correctness wall below about 14B and a speed wall above it; that naive in-context
self-refinement does not compound and can even degrade; that fixing the reward from a cliff
to a slope removes the degradation but does not by itself create RSI; and that the first sign
of genuine, transferable self-improvement appears at L2 (verified skill memory) for a model
that is both capable enough to bank good skills and not yet at its ceiling.

---

## 8. Threats to validity and honest caveats

Small n. The RSI pilots use 8 to 24 tasks; slopes are estimated, not yet bounded with
confidence intervals. Acceleration (compounding curvature) is underpowered at 4 to 5 rounds.

Attribution not yet closed for L2. The rising Fable slope is consistent with skill transfer
but has not been separated from placebo or search. The controls are specified and are the
immediate next step.

Expert rung is a moving target. It is the best kernel a strong curator found, versioned per
hardware, not a proven optimum. Beating an old expert is not beating a future one.

The roofline is `torch.compile`, not an expert compiler stack, so absolute speed scores are
demanding by construction. This is deliberate, since there is real headroom above the
compiler (no global optimum), but it makes the Easy tier a speed dead end (torch already
saturates memory-bound ops).

---

## 9. Next steps

1. Attribution controls on L2 (frozen-library, poisoned-library, matched-compute L0) with
   larger n and a second seed, on Fable 5.1 and a second mid-capability model.
2. Build L1 (notes) as the control between L0 and L2, and L3 (agent tools) as the next depth.
3. Track B (real inference stacks) for the reality axis: make a real model's prefill or
   decode step faster, scored end to end.
4. Portability matrix: does one model's artifact help another.

Data and code. Analysis records live in `analysis/` (this report, `calibration_run.md`,
`scaffold_rsi_run1.md`, `phase0_exit.md`, `l2_result.md`). Design in `docs/RSI_DEPTH_PLAN.md`.
Harness in `kernelascent/` (grader, scoring, campaigns, self-test). Dataset in
`dataset/public/` and on Hugging Face.
