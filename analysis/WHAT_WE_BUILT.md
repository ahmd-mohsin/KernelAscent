# What we have built so far: a detailed analysis

A candid audit of the KernelAscent / RSI-Depth project: the components, how mature each is,
what they established, and what is still missing. Written after stopping all running
experiments.

## 1. Summary judgment

We have built a solid, reusable measurement instrument for GPU-kernel optimization and a
first-of-its-kind attempt at a causally-controlled RSI protocol on top of it. The instrument
(dataset, grader, scoring, campaign harness) is real and tested. The scientific claims are
honest but limited: we have established capability calibration and a preliminary,
underpowered memory-transfer signal. We have not yet established recursive self-improvement.
The gap is not infrastructure; it is statistical power and the decisive transplant
experiment, both of which the current code supports and neither of which has been run at
scale.

## 2. The benchmark substrate (mature)

Procedural task generator (`gen_source_tasks.py`). Generates KernelBench-style PyTorch
modules from a seed across six families (matmul, norm-act, attention, rope-attention,
quant-gemm, moe), with shapes, dtypes, fusion pattern, and epilogue all seed-derived,
including non-power-of-two widths. Public seed range for release; a disjoint private range
(10,000,000+) held out and never published. Every task carries a tier and, after grading,
an empirical difficulty label.

Capability tiers (calibrated). Easy (small elementwise / single reduction), Medium (matmul +
fused epilogue), Hard (attention, fused chains), Ultra (MoE, large irregular). Difficulty was
not asserted, it was measured by running 13 models: correctness falls monotonically Easy to
Ultra, confirming calibration.

Curation and expert rungs (built, Medium band). A strong curator (Fable 5.1) generated
several candidates per task, graded on GPU; the best correct-and-fast one is the task's
expert reference. On Medium, 134 tasks curated, 112 with an expert that beats the compile
baseline. This is what makes the speed score meaningful (a target above the compiler exists).

Published dataset. `dataset/public/<Tier>/<task>/` with task and meta (tier, family, shapes,
empirical difficulty), plus a manifest, on GitHub and Hugging Face (`muahmed7338/kernelascent`).

## 3. The grader and scoring (mature, tested)

`grade_candidates.py`. Correctness against an fp32 gold on N=4 fresh random inputs with a
dtype-aware tolerance and an input-sensitivity check (rejects constant / input-ignoring
outputs), verified on the timed run. Roofline is min(eager, torch.compile). The key
robustness property: each candidate is graded in an isolated subprocess, so a native compiler
abort (SIGABRT from the Triton/MLIR backend, uncatchable by a Python timeout) or a total hang
loses only that candidate, not the run. This was a real bug we hit and fixed; grading is now
crash-proof.

`scoring.py`. Continuous log-interpolated speed score between eager (0), compile, and expert
(1), replacing an earlier pass/fail-at-compile cliff. Plus keep-best acceptance.

`test_grader.py`. A GPU self-test that plants known candidates and asserts: correct passes,
wrong fails, reward-hack constant rejected, erroring candidate isolated, run survives,
pass_at_k exact. All invariants pass. This is the anti-bug gate for the scoring path.

Honest limits (documented, not yet fixed): four random inputs is not broad numerical
correctness; the expert-normalized score is unstable when eager and expert are close; timing
is single-run, not interleaved A/B. The causal plan specifies the hardening; it is not done.

## 4. The RSI protocol harness (built, evolving)

We built four generations of campaign runner, each fixing a conceptual flaw in the last.

`openweight_rsi.py`. Per-task in-context refinement. Showed the loop degraded models (round 0
was the peak). Diagnosis: no persistence channel and a cliff reward. Retired as an RSI
instrument, kept as a prompt-dynamics probe.

`rsi_campaign.py`. Added keep-best and the log-interp score. Removed the degradation (best-so-
far monotone) but the slope stayed flat: the expected L0 baseline.

`l2_campaign.py`. Verified skill/code-block memory with a private-seed transfer set, for both
open-weight (HF) and closed (Bedrock) agents. Produced the preliminary transfer signal.

`rsi_causal.py`. The current instrument. Splits a checkpoint into a solver S (solve()) and an
improver U (improve()) with explicit state, so a later transplant cannot secretly carry the
library. Implements the four matched control arms (growing, frozen-nonempty, offline-built,
matched-search) and independent campaigns via a seed offset.

`rsi_all.sh`. Full-sweep driver: 15 models (6 Bedrock API, 9 open-weight) x 4 arms x 2 seeds
= 120 jobs, poll-based one-GPU-per-job pool (no OOM), resumable. Built and launched; stopped
early on request (only the Fable jobs had completed).

## 5. Compute and infrastructure (working, with known friction)

Greenland 3-node p4d fleet (24 x A100-40GB) reached over an SSM port-forward tunnel and
empty-password SSH on port 2222 as greenland-user; worker nodes reached from the main node
(no shared filesystem, so results are gathered per node). Bedrock for API agents and for
curation, credentials in a gitignored file read via a profile. Recurring friction we learned
to handle: tunnel idle-timeouts (reopen on demand), long inline shell commands truncating
(ship scripts and run detached), `pkill -f <name>` killing its own shell (bracket patterns),
and file transfer via `cat | ssh` being clobbered by a hardcoded stdin redirect (use tar or
base64-as-argument). These are documented and no longer block work.

## 6. What the experiments established

Capability calibration (solid). Two walls, cross-family confirmed. Below ~14B the binding
constraint is producing valid compilable Triton (about 70% of candidates fail, 23% on
compile errors, 9% hallucinated APIs); above it, the constraint is beating the compiler
(correct kernels beat the roofline only 3 to 30% of the time). Tiers ordered correctly.

Scaffold-RSI run 1 (informative negative). 26 API models, text strategy library, no
compounding: 17 never opened the channel, 7 plateaued, 2 degraded. Exposed library poisoning
(reasoning-CoT leak, hallucinated APIs), since fixed.

Reward fix (validated). Cliff to slope plus keep-best removed the round-over-round
degradation. L0 self-refinement is flat, the expected control baseline.

L2 memory (preliminary, underpowered). Fable 5.1 rose from about 0.01 (empty library) to
about 0.20; Opus 5 was flat at 0.20 (ceiling); open-weight banked ~no usable skills. Looks
capability-windowed.

Causal experiment #1 (pilot, not significant). growing 0.287 > frozen 0.270 > offline 0.235 >
search 0.212: the right direction, but the gaps are below the measured ~0.15 noise floor at
n=2. The unchanged-state noise floor was measured directly (Coder-7B, identical empty state,
still swung 0 to 0.15 from sampling alone).

## 7. What we have NOT built or established

- Recursive improvement. Not shown. The direction is suggestive but within noise.
- Improved improvement ability. The solver-improver transplant (experiment #4) is coded but
  not run.
- Causal inheritance. The ancestry-intervention protocol (remove/replace/rescue before a
  discovery) is specified, not implemented.
- Statistical power. All RSI results are n=8 to 24 tasks and 1 to 2 campaigns. No confidence
  intervals. The unit must become the campaign, with 8+ per arm.
- Track B (real inference stacks), harder transfer panels (compositional, structural, family,
  systems), verification hardening (numerical contracts, hidden structured cases, interleaved
  A/B timing), and L1 / L3 / L4 / L5. All specified in the plan, none built.

## 8. Honest positioning

The individual ingredients exist in prior work: executable skill libraries (Voyager),
recursive scaffold optimization (STOP), self-modifying agents with archives (Darwin Godel
Machine), improving the improver (HyperAgents), kernel correctness+speed (KernelBench). Our
defensible contribution is not any single ingredient; it is the controlled separation of
accumulated knowledge, improved improvement procedure, extra compute, and inherited causal
dependence, on a substrate where all four are cheaply and objectively measurable. That
contribution is currently a well-built apparatus with a pilot, not yet a result.

## 9. The single most important next step

Run the solver-improver transplant (does U_k beat U_0 on a common S_0) with enough
independent campaigns to clear the noise floor, backed by the frozen-nonempty and
offline-built controls. That one experiment determines whether this becomes a
recursive-improvement paper or a rigorous transferable-memory paper. Everything needed to run
it is built.

Pointers: design in `docs/RSI_CAUSAL_PLAN.md`; findings in `analysis/EVALUATION_REPORT.md`,
`calibration_run.md`, `l2_result.md`, `causal_exp1.md`, `phase0_exit.md`; harness in
`kernelascent/`; dataset in `dataset/public/` and on Hugging Face.
