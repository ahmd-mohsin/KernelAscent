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

So on this bank the models essentially never produce a submission faster than the baseline.
The headroom-normalised score collapses to a correctness indicator with two reachable states
(0 and ≈0.5) that saturates in 1–3 rounds, and `lineage − reset` is forced toward zero
regardless of whether compounding occurs.

**A flat contrast under these conditions is uninformative, not a null.**

### Correction to this amendment's stated cause (same day, before any passrate run landed)

This amendment first attributed the saturation to a stronger `torch.compile` baseline on H100.
That was wrong, and the correction is recorded rather than silently edited. The default scorer
(`KA_SCORE=eager`) does not use the compiled baseline at all — it scores speedup against
**eager** with the legacy fixed 1.5× anchor — so no claim about `torch.compile` was licensed by
these runs.

The real cause is upstream of the metric. Of 86 verified kernels harvested from the 14B model,
**zero** contained Triton, CUDA or `load_inline`; all 86 were pure-PyTorch rewrites of the
reference. An independent re-harvest under the same prompt replicated this exactly (0 of 118),
for **0 of 204** across both. A rewrite runs at eager speed by construction (median 1.00×, 93% below 1.05×), and
`_score(correct, 1.00) = 0.50` exactly. The generation prompt contains the clause *"a plain-torch
kernel that is correct beats a fancy one that errors"*, which steers precisely this way.

This does not change the amendment — pass-rate is still the right second metric when the live
axis is correctness reliability — but it changes what a flat result would *mean*, so the
distinction is registered before any pass-rate result is seen. A prompt A/B (`KA_PROMPT=safe`
vs `kernel`, identical tasks/grader/model/seed) is running to settle it.

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

---

# Amendment 2 — 2026-09-23: the kernel-authoring experiment set

**Registered before any of these runs produced a result.** The jobs were submitted at 19:0x;
this amendment is committed before the first completes. Nothing below was chosen after seeing
an outcome, and the git history of this file is the evidence.

## Why a new experiment set exists

Every result this project has produced was measured on **correctness-preserving rewriting**, not
kernel optimisation, because Triton kernels could not verify on our harness (host `CC` leaked
into the container; only Triton compiles C at runtime). With that fixed, the models attempt a
custom kernel in 0/16 samples under the published prompt and 14/16 when asked, of which 1
verifies.

That rate matters more than its value: **it is neither 0 nor high**. Every saturation this
project hit — scores pinned at 0.50, lineage pinned at 1.000, a coverage-injection control
exhausted by round 2 — came from measuring a task the models had already solved. Kernel
authoring is the first task here they have not.

## Registered primary analysis

* **Substrate:** the 29-task bank, `KA_PROMPT=kernel`, default (headroom) scoring.
* **Unit of replication:** the trajectory. One seed, one base checkpoint, one held split, one
  accumulated adapter. Never the round.
* **T2-kernel PRIMARY:** `lineage − reset`, trajectory level, reported with a 95% CI and TOST at
  **δ = 0.05**, pooled across 1.5B/3B/7B and reported per scale.
* **Target n:** **12 trajectories per arm** (2 scales × 6 seeds, plus 7B × 2). Fixed from a
  power calculation, not from what finished: between-trajectory SD on the closest available
  analogue is 0.058, giving MDE 0.120 at n=4 and 0.051 at n=12. **A result at n < 8 per arm is
  reported as underpowered and no equivalence claim is made from it.**
* **Positive control:** the inject arm (teacher kernels from the 14B cell on tasks the student
  failed). Its `injected_tasks` count per round is reported alongside every result — a control
  that stops firing has not passed, it has stopped testing.
* **Secondary:** T1-kernel capability by scale; recursion-vs-sampling at matched generation
  budget; mechanism probe (diversity, LoRA drift by depth, retention).

## Decision rules, fixed now

| outcome | conclusion |
|---|---|
| `lineage − reset` CI excludes 0, positive, control fires throughout | compounding, on a task the models have not saturated — the result this benchmark exists to produce |
| CI includes 0, TOST equivalent at δ=0.05, **and** the positive control moved | a real bounded null |
| CI includes 0 **and** the positive control did not move | the instrument cannot register compounding here; **publish no claim** |
| either arm saturates (>30% of rounds at a bound) | report the saturation, treat magnitudes as bounds, make no trend claim |
| n < 8 per arm | underpowered; report the interval, make no equivalence claim |

## What this set may not be pooled with

Not with the A100 headroom boards (different hardware, different task, and those were graded by
a harness that silently rejected Triton). Not with the `h100/passrate` set (different metric).
Tagged `h100/kernel-authoring`.

## Falsifier

If the capability curve (T1-kernel) shows a verified-kernel rate at or near zero for every scale
including 14B, then the task is unreachable rather than unsaturated, the compounding contrast is
uninterpretable for the same reason as before, and we report that and publish no compounding
claim from it.

---

# Amendment 3 — 2026-09-24: deviations from the original registration, declared

An adversarial review found that this project **departed from its own pre-registration without
saying so**, which is worse than not having pre-registered. The deviations are listed here, and
the runs that close them are queued.

## Deviation 1 — the T2 primary metric was substituted

* **Registered** (§Primary metrics): `self − fresh_frozen` — a frozen producer trained on the
  same fresh data each round, isolating "the improver got better" from "more data".
* **Reported**: `lineage − reset`, a different contrast, in which the control re-initialises its
  adapter and trains on one round's data.
* **Status**: not a defensible substitution — it was never declared. `lineage − reset` also
  confounds accumulation with total gradient steps (lineage receives R×, reset 1×), which
  `self − fresh_frozen` does not.
* **Closing it**: the `prereg-*` cells run `lab_weight_rsi --fresh-gpu`, which records
  `delta_self_minus_fresh` directly. The registered contrast becomes primary; `lineage − reset`
  is reported as a secondary contrast with its compute confound stated.

## Deviation 2 — three incompatible scoring definitions coexist

* **Registered** (§Substrate & scoring): `sp` = speedup over the **`torch.compile`** baseline,
  normalised by the **per-task roofline ceiling**.
* **Actually used** by every reported number: `LK._score(ok, sp_eager)` — speedup over **eager**,
  normalised by the **legacy fixed 1.5×** anchor, because `ceiling` defaults to 1.5 when not
  supplied (`kernelascent/v3/lab_kernel.py`).
* **Third definition**: T1 reports `fast_rate` = beating `min(eager, torch.compile)` by ≥1.10×.
* **Consequence**: the "roofline-grounded" claim is not true of the numbers as produced — the
  roofline never entered the scoring path.
* **Closing it**: the `prereg-*` cells run under `KA_SCORE=compiled` with `KA_ROOF_ARCH=h100`,
  which is the registered definition. Any number produced under the legacy scorer is labelled
  `eager/1.5×` wherever it appears, and "roofline-grounded" is used only for numbers that were.

## Deviations we are declaring but not closing

* **Seed counts.** §Statistics registers ≥3 seeds per headline cell. T3 and T4 were reported at
  n=1 per cell with no uncertainty. Those are demoted to exploratory single-run observations.
* **H1/H2/H3.** The registered hypotheses are never reported against. H1 ("small models compound;
  diversity collapse explains mid/large overfit") is **contradicted** by our own data:
  compounders show *lower* generation diversity (0.41 vs 0.52). Recorded as a failed prediction.
* **Held-out split.** §Reproducibility registers it as private and never released; the README
  concedes it is reconstructible from the public bank. It is a development set, not a test set,
  until an eval server exists.

**Registered before these runs land**, as with Amendments 1 and 2: if `self − fresh_frozen` and
`lineage − reset` disagree in sign, the registered contrast is the one reported as primary, and
the disagreement is reported rather than resolved in favour of whichever is more interesting.
