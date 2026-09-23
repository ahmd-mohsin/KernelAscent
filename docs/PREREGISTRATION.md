# KernelAscent — pre-registration of the primary analysis

Registered before the definitive re-run on the SOTA-curated, roofline-gated banks. Locks metrics, controls,
scoring, admission, and decision rules so the headline results cannot be chosen post-hoc.

## Substrate & scoring (fixed)
- Task = a PyTorch `Model` module; a submission returns `ModelNew`, graded on GPU.
- Correctness: output matches the fp32 reference on 3 fresh inputs within a per-task tolerance (max(2e-2, 2×ref_err)).
- Speed score is **headroom-normalized**: `score = 0` if incorrect, else `0.5 + 0.5·clamp((sp−1)/(ceiling−1), 0, 1)`,
  where `sp` = candidate speedup over the `torch.compile` baseline and `ceiling` = the per-task roofline-achievable
  speedup (FLOP+byte roofline, A100). Correct-at-parity = 0.5; correct-at-hardware-limit = 1.0 for ANY ceiling.
  This removes the fixed-1.5× anchor so no task caps a score below model skill.
- Capability C = mean over held tasks of best-of-k score.

## Task admission (fixed) — no benchmark ceiling
A task enters a size-matched bank iff, against that scale's frozen anchor (small=Qwen-1.5B, mid=7B, large=14B):
(1) correct_rate > 0 (solvable), (2) best_score ≤ 0.75 (not already easy/fast), and (3) roofline ceiling ≥ 1.3×
(real headroom over torch.compile — excludes un-improvable tasks). Held-out split is private and never released.

## Primary metrics (registered)
- **T1 Capability:** held-out mean C (best-of-k). Secondary: correct_rate, compiled-speedup.
- **T2 Weight-RSI:** PRIMARY = `self − fresh_frozen` producer delta, final round AND round-averaged (AUC).
  A frozen producer trained on the same fresh data each round is the control that isolates "the improver got
  better" from "more data."
- **T3 Procedure-RSI:** PRIMARY = `Q_self_modify − Q_frozen_procedure` (self-modify vs frozen-procedure), final + AUC.
  Secondary control: archive-only.
- **T4 closed→open:** PRIMARY = `improved_harness − frozen_harness` on the open trainee, AUC over rounds (final reported too).

## Hypotheses (registered)
- **H1 (mechanism):** small models compound (self−fresh > 0), mid/large overfit (self−fresh ≤ 0), with self-data
  diversity collapse and round-0 retention decline explaining the overfit. Predicted before mid/large results.
- **H2 (recursion > sampling):** weight-RSI final C exceeds best-of-k, self-refine, and retrieval at EQUAL
  generation budget (rounds×k draws per held task).
- **H3 (closed→open):** improved−frozen harness delta is positive and non-decreasing across rounds for capable
  researchers, and widens with researcher capability.

## Statistics (registered)
- ≥3 seeds per headline cell; report bootstrap 95% CIs on the primary metric.
- A result "holds" iff the primary delta's 95% CI excludes 0 AND mean(last-2-rounds) > 0.05.
- Verdict labels: compounds / harness-helps (holds), flat (CI includes 0), overfits / hurts (delta ≤ −0.05).

## Controls (fixed)
Every RSI claim is a causal delta vs a frozen control (fresh-frozen producer / frozen harness / frozen procedure),
never raw capability, so a rise cannot be attributed to a stronger base model. Compute is matched across the
recursion arm and its non-recursive baselines.

## Reproducibility
Fixed A100 spec + locked-clock timing (median of N trials, warmup), deterministic split seed, Dockerized
one-command runner per track, and public task pool released; held-out split withheld for scoring.

---

# Amendment 1 — 2026-09-23: H100 runs, and a second capability metric

**This is a post-hoc amendment.** It was written *after* seeing that the original metric has no
dynamic range on H100. It is recorded here, with its motivation and its limits, rather than
quietly adopted — and the affected runs are reported as a separate experiment set, never pooled
with the A100 boards.

## What we observed

Running the compounding protocol on Marlowe (H100) with Qwen2.5-Coder 1.5B/3B:

* **76% of all scores, across every run and round, were exactly 0.50.**
* `0.50` is precisely `_score(correct, speedup = 1.0)` — correct, not one bit faster.
* Calibrating the bank against a 3B H100 anchor (`--min-ceiling 1.3`) kept 23/29 tasks but
  reported a frozen-base mean best of **0.511**, with every kept task at ≈0.50 and all four L3
  tasks at `correct_rate = 0.00`.
* The 14B teacher's best speedups across the bank: nine tasks at 1.00×, two at 1.08×, one each
  at 1.05×, 1.01×, 2.03×.

So on this hardware and bank, models from 0.5B to 14B **essentially never produce a kernel
faster than the baseline**. The headroom-normalised score collapses to a correctness indicator
with two reachable states (0 and ≈0.5) that saturates in 1–3 rounds, and `lineage − reset` is
forced toward zero regardless of whether compounding occurs.

**A flat contrast under these conditions is uninformative, not a null.**

## The amendment

Add a third capability metric, selected by `KA_SCORE=passrate`:

> **pass-rate:** per task, the fraction of the *k* sampled candidates that verify correct.
> Averaged over held-out tasks. Continuous in [0, 1]. Denominator is *k*, so a task where the
> model emits nothing parseable scores 0 — formation failure is a real failure.

Registered primary for the H100 experiment set: **`lineage − reset` on pass-rate**, trajectory
level, TOST margin δ = 0.05, exactly as in the original registration.

## What this metric does and does not claim

It measures the **reliability of correctness**, not optimisation skill. A model that solves a
task 1-in-8 and improves to 5-in-8 has demonstrably improved; best-of-k headroom scores both at
0.500 and sees nothing. Pass-rate has 8× the resolution there.

It deliberately **does not reward speed**: a 2× kernel and a 1.0× kernel each count as one
success. It therefore tests a *different and weaker* claim than the original headroom score —
"does self-training make correctness more reliable?" rather than "does it capture more of the
physically available headroom?"

Consequences we bind ourselves to:

1. Pass-rate results are reported as their own experiment set, tagged `h100/passrate`, and are
   **never pooled with, or compared against, the A100 headroom boards.**
2. The original headroom metric remains primary for any hardware where it has demonstrated
   range. Pass-rate does not supersede it.
3. The paper must state plainly that this metric was adopted after observing saturation, and
   that a positive pass-rate result is evidence about correctness reliability only.

## Why not simply filter the bank harder

We tried. A filter can only keep or drop tasks; it cannot create a band where the model is
correct *and* has speed it can capture. After filtering, every surviving task still scored
≈0.50. The saturation is a property of the model–bank–hardware combination, not of task
selection.

## Pre-registered falsifier

If pass-rate `lineage − reset` is **also** flat while the positive control (coverage injection)
**does** move it, the loop genuinely fails to compound and we report that. If neither moves,
the instrument still lacks range and we report *that* instead — and publish no compounding
claim from this hardware.
