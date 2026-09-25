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

* **75% of the frozen base's *non-zero* scores on the 29-task bank land within ±0.01 of parity** (18 of 24 solved tasks), under the eager/1.5x calibration scorer.
  *(Corrected 2026-09-24: an earlier version said "76% of all scores, across every run and
  round, were exactly 0.50" — overstating both precision and scope. It is a band, not an
  exact value, and it describes the frozen-base calibration on one bank, not the run scores.)*
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


---

# Amendment 4 — 2026-09-24: T1-kernel scores compliance, not capability

**Registered before the re-measurement lands.** The first T1-kernel curve completed and is
being discarded rather than reported, for a reason worth stating in advance.

## What the first curve showed

| scale | solved | via plain-torch rewrite |
|---|---|---|
| 0.5B | 8/29 | 4/8 |
| 1.5B | 4/29 | 2/4 |
| 3B | 9/29 | 3/9 |
| 7B | 3/29 | **0/3** |
| 14B | **1/29** | **0/1** |

Solve-rate falls monotonically-ish with scale. It is not a capability curve.

## Why it is invalid

The prompt asks for a kernel. A submission that ignores it and restates the reference in torch
ops **verifies trivially** and counts as a solve. Compliance with the instruction rises with
scale — 7B and 14B never take the easy route, 0.5B and 1.5B take it for half their solves — so
the metric pays models for disobedience and the curve inverts.

Zero parse failures at any scale (348 candidates each), so this is genuine verification failure,
not malformed output.

## Registered change

`make_teacher_kernels` now records, per task: candidates generated, candidates **attempting** a
kernel (Triton/CUDA/`load_inline`), and kernel attempts that **verified**. T1-kernel reports:

* **attempt rate** = attempted / generated — an instruction-following measure
* **verify-given-attempt** = verified kernels / attempted — the capability measure
* solve-rate is reported but **never compared across scale** without both of the above

**Registered now:** verify-given-attempt is the T1-kernel primary. If it is flat or rising with
scale while solve-rate falls, the inversion is a metric artifact and we say so. If
verify-given-attempt *also* falls with scale, that is a real and surprising capability finding
and we report it as such.

# Amendment 5 — 2026-09-24: the registered primary was invalidated by a resume defect, and is being re-run

**Declared while the replacement runs, before any of its numbers exist.**

## The defect

`lab_weight_rsi` — which runs the registered primary — restored `history` and the round counter
on resume, but not the trained weights. All three arms accumulate across rounds (`self` on its
own data, `ctrl` on the frozen round-0 set `ex0`, `fresh` on frozen-base data), so every resumed
chunk continued the round *numbering* while restarting each arm from base weights.

`ex0` was lost the same way. It is frozen at round 0 and the control re-trains on it every
round, so a resume silently re-froze the control's comparison target mid-trajectory. The
retention denominator `round0_solved` was lost too.

Nothing errored. The signature is `trainC` collapsing at exactly the round named by `resumed_at`:
`prereg_q3_s1` 0.334 → 0.165, `prereg_q3_s2` 0.122 → 0.0. The same defect was present in
`lab_compounding`, where it is visible in **all 18** resumed cells with no exceptions.

## Why the marker was not enough

The artifact already recorded `resumed_at` precisely so severed trajectories would never be
pooled with clean ones. That safeguard failed in practice for a reason worth registering: **every
cell hit the walltime, so every cell was marked, so the marking distinguished nothing.** A flag
only protects an analysis if some runs lack it.

## What is discarded

All six `prereg_*` cells collected before 2026-09-24 15:32. They are archived as
`prereg_*.severed-1532`, not deleted, and must not appear in any reported figure. Cells that had
not yet resumed (`prereg_q15_s1..s3`, `prereg_q3_s3`) are also discarded despite being clean:
without a weight checkpoint they cannot be extended, so reporting them would mean reporting a
2–3 round trajectory as if the registered depth had been reached.

## What replaces it

The same six cells, re-run from round 0 against the fixed lab. Per-round checkpoints for all
three adapters plus `ex0`, `round0_solved` and `C0`, written to a temp file and `os.replace`d,
because the walltime kill lands mid-round by construction. A resume that finds no checkpoint now
**exits 3 and refuses** rather than severing silently.

## Registered in advance

* The primary and its decision rule are **unchanged** — this amendment changes no hypothesis,
  no metric, and no threshold. It records that the previous data collection was invalid.
* `adapter_restored` is stamped in every artifact. Any cell reporting `false` is excluded.
* A trajectory whose `resumed_at` is set but `adapter_restored` is absent is pre-fix data and is
  excluded regardless of how its numbers look.
* `scripts/check_resume.py` fails any lab that trains weights, resumes rounds, and does not
  restore them. Negative-tested against the pre-fix source.
* **Falsifier unchanged.** If the re-run shows no `self − fresh_frozen` effect, that is the
  result; the defect explains discarded data, it does not license a second look at a null.

# Amendment 6 — 2026-09-24: the registered primary's self arm receives ~0.8 examples per round

**Declared before the replacement cells produce any number.**

## The measurement

`--n-train` defaults to **3** in `lab_weight_rsi` and no registered command overrode it. A round
therefore draws n_train × k = 3 × 10 = 30 generations for the self arm. Measured across six
cells and 19 completed rounds, the verified yield is **0.028 per generation**, giving an expected
**0.84 training examples per round**. Observed: `n_ex` is **zero in 8 of 19 rounds (42%)**, mean
0.84, max 3, with `loss = 0.0` in the empty rounds confirming no SFT ran.

## What this means for the registered contrast

On those rounds `self − fresh_frozen` compares an **untrained** model against one trained on
frozen-base data. The negative values are explained by the self arm having no input. The
contrast does not measure the quality of self-generated data against frozen-generated data,
which is what it was registered to measure.

This is **not** a null result about recursive self-improvement. It is a configuration in which
the recursion has no input, and it is the same failure as T5, where `L−F` is undefined at an
author yield of 2.3%. Both self-referential loops starve.

## Registered now

* The six `prereg_*` cells are **run to completion and reported**, as the documented measurement
  of the starvation. They are **not** interpreted as evidence for or against compounding.
* A separate set, tagged `fed`, runs the same lab at `--n-train 35` against the 124-task DSL bank
  via `KA_KERNEL_BANK`, which raises the expected input to **9.8 examples per round**.
* **The `fed` set is NOT the registered primary and must never be pooled with it.** It differs in
  training-split size and in task bank. It is exploratory and is reported as such.
* **Registered in advance.** If the `fed` set shows `self − fresh_frozen` at or below zero while
  `n_ex` is comfortably above zero every round, that is a real null about self-generated data
  quality and we report it as one. If `n_ex` is still near zero at n_train=35, the yield is the
  binding constraint and no training-split size rescues the design at this scale and prompt.
* Every artifact records `n_ex` per round. Any trajectory whose mean `n_ex` is below 2 is
  reported as starved rather than as a measurement of the contrast.

## Amendment 6, addendum — how the `fed` set may and may not be compared

Written before the first `fed` round lands, because the temptation runs the other way once
numbers exist.

The `fed` cells use the 124-task DSL bank and the registered cells use the 29-task curated bank.
The frozen-base scores differ accordingly: `C0 = 0.063` on the DSL bank against `C0 = 0.174` on
the curated one, so the DSL held set is materially harder.

* **Not comparable**: absolute `C_self`, `C_fresh`, `C_ctrl` across the two sets. Different
  tasks, different difficulty, different base rate.
* **Comparable in kind**: `self − fresh_frozen`, because it is a within-cell difference between
  two arms evaluated on the same held set in the same round.
* **Comparable directly**: `n_ex`, the count of verified self-generated training examples per
  round. That is the quantity Amendment 6 is about, and it is a raw count.

A caution on the contrast itself. A difference between two arms can scale with the base rate, so
a smaller `self − fresh_frozen` on the harder bank is not automatically a weaker effect. If the
fed contrast comes back near zero **with healthy `n_ex`**, the honest reading is a null on this
bank at this scale, not a general null, and the bank difference is stated alongside it.
