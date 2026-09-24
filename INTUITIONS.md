# Running intuitions — KernelAscent

Hard-won operational and scientific judgement, recorded as it is learned. Newest at the top of
each section. This is the file to read before designing the next experiment.

Sections 2.x are one session's chain of instrument defects, in the order they were
found -- each one was a publishable-looking result first.

Companion to `BENCHMARK_LOG.md` (dated record of numbers) and `PROGRESS.md` (state).
`~/.marlowe/MARLOWE_FACTS.md` holds cluster facts.

---
## Contents

- [1. The measurement lies more often than the model does](#1-the-measurement-lies-more-often-than-the-model-does)
- [2.1 Ceiling saturation: the 2026-09-23 H100 result](#21-ceiling-saturation-the-2026-09-23-h100-result)
- [2.2 The deeper problem: on H100 the score is effectively BINARY](#22-the-deeper-problem-on-h100-the-score-is-effectively-binary)
- [2.3 Fifth zero-producing defect: the compiled cache on a quota-exhausted filesystem](#23-fifth-zero-producing-defect-the-compiled-cache-on-a-quota-exhausted-filesystem)
- [2.4 The sixth defect was my own explanation of the first five](#24-the-sixth-defect-was-my-own-explanation-of-the-first-five)
- [2.5 Defect #7: the harness could not grade triton — root-caused and FIXED](#25-defect-7-the-harness-could-not-grade-triton--root-caused-and-fixed)
- [2.6 The answer, once the instrument worked: the prompt sets ambition, capability sets success](#26-the-answer-once-the-instrument-worked-the-prompt-sets-ambition-capability-sets-success)
- [2.7 Choosing a metric with resolution, and the honesty cost of choosing it late](#27-choosing-a-metric-with-resolution-and-the-honesty-cost-of-choosing-it-late)
- [2.8 Reachability — the range check, turned into a number I can act on](#28-reachability--the-range-check-turned-into-a-number-i-can-act-on)
- [2.9 I fixed the floor and hit the ceiling](#29-i-fixed-the-floor-and-hit-the-ceiling)
- [2.10 The grader knew why, and threw it away](#210-the-grader-knew-why-and-threw-it-away)
- [2.11 A positive control is only as strong as the thing it injects](#211-a-positive-control-is-only-as-strong-as-the-thing-it-injects)
- [2.12 A stale local copy is a live hazard, not just clutter](#212-a-stale-local-copy-is-a-live-hazard-not-just-clutter)
- [2.13 I used data I had already documented as contaminated](#213-i-used-data-i-had-already-documented-as-contaminated)
- [2.14 Sweeping for a contamination I had already found once](#214-sweeping-for-a-contamination-i-had-already-found-once)
- [2.15 A gate that cannot see its input, and knowing when to stop widening it](#215-a-gate-that-cannot-see-its-input-and-knowing-when-to-stop-widening-it)
- [2.16 An enforcement whose default is "allow" enforces nothing](#216-an-enforcement-whose-default-is-allow-enforces-nothing)
- [3. Published claims that turned out to be confounded](#3-published-claims-that-turned-out-to-be-confounded)
- [4. Statistics: the unit of replication is the trajectory](#4-statistics-the-unit-of-replication-is-the-trajectory)
- [5. Cluster operations (Marlowe)](#5-cluster-operations-marlowe)
- [6. Standing rules](#6-standing-rules)

---

## 1. The measurement lies more often than the model does

Five times now a **pipeline defect has masqueraded as a scientific finding**. Every one produced
a plausible, publishable-looking zero. None announced itself.

| # | defect | what it looked like |
|---|---|---|
| 1 | wrong `KA_ROOT` → grader mis-imported | "self-training does not compound" (days lost) |
| 2 | `KA_GRADE_GPU` defaulted to device `2`, absent on a 1–2 GPU job | "the model writes bad kernels" |
| 3 | raw generations fed to the grader without `extract_modelnew` | "the 14B teacher solves 0/29" (its real coverage is 86%) |
| 4 | score saturated at the correctness floor | "lineage ≈ reset, no compounding" |
| 5 | compiled-baseline cache hit an inode quota (§2.3) | "4 of 6 rounds produced nothing" |

**The intuition:** in this project, a clean zero is evidence of a bug until proven otherwise. A
real null looks *noisy* — it has variance, partial successes, per-task spread. A null that is
suspiciously tidy (every task fails; every score identical) is instrumentation.

**The practice that catches it:** before any batch, assert a *known-good input produces a
known-good output* — an identity kernel must grade correct with speedup ≈ 1.0
(`scripts/precheck_grader.sh`). Run it **on the same allocation shape** the real jobs use; bug
#2 hid specifically in the `-G` difference.

### And the gap that check still had

Bug #4 slipped through a passing precheck. I verified the grader *worked*; I never verified the
**score had room to move**. A component can be functioning perfectly while the measurement is
dead.

> **Check dynamic range, not just correctness.** Before trusting a contrast, confirm the metric
> can take at least two distinguishable values on this hardware and this bank.

---

## 2.1 Ceiling saturation: the 2026-09-23 H100 result

E1 (coverage-injection positive control) ran clean on H100 — 8 cells, 6 rounds each, injection
firing correctly and visibly accelerating coverage. Both arms came out flat:

```
control  +0.016 [-0.028, +0.061]   n=4 trajectories
inject   -0.004 [-0.037, +0.029]   n=4
```

But **76% of all scores across every run and round were exactly 0.50**.

`0.50` is not a value models drift toward. It is precisely `_score(ok=True, speedup=1.0)`:
correct, and not one bit faster than the baseline. Models climb to the correctness floor in
1–3 rounds and stop.

With both arms pinned at the same ceiling, `lineage − reset` is **forced to ≈0 by
construction**. The experiment could not have detected compounding in either direction.

**Why here and not on the A100 fleet** — ~~`KA_SCORE=compiled` scores against a
`torch.compile` baseline, and compile is relatively stronger on H100~~. **This explanation was
wrong; see §2.4.** The default scorer never touches the compiled baseline. Struck through rather
than deleted, because the wrong answer was plausible, fitted the data, and cost a day.

**The intuition:** a headroom-normalised score silently becomes a correctness-only score when
the baseline is strong. `0.5` is then an absorbing state. Always look at the *distribution* of
scores, not just their mean — a spike at exactly 0.5 means the speed dimension is dead.

**Fix in flight:** `difficulty_filter.py --min-ceiling 1.3` against an H100 anchor, to admit
only tasks where the roofline leaves real headroom over `torch.compile`. Re-run E1 after.

---

## 2.2 The deeper problem: on H100 the score is effectively BINARY

Calibrating the 29-task bank against a 3B H100 anchor (`difficulty_filter --min-ceiling 1.3`)
was meant to restore headroom. It revealed something worse than a missing filter.

```
kept 23/29  {L1: 9, L2: 14, L3: 0}
frozen-base mean best on kept = 0.511      <- the C0 floor
```

Look at the per-task numbers. **Every kept task scores best ≈ 0.50.** L1 tasks: 0.50, 0.55,
0.55, 0.50, 0.52, 0.58… L2 tasks: 0.49–0.51. All four L3 tasks: `correct_rate = 0.00`.

So the difficulty distribution is **bimodal with nothing in between**:

| band | correct_rate | best score | meaning |
|---|---|---|---|
| L3 (4 tasks) | 0.00 | 0.00 | unreachable |
| L1/L2 (23 tasks) | 0.12–0.62 | **≈0.50** | correct, and never faster |

A score of exactly 0.50 is `_score(correct, speedup=1.0)`. So for a 3B model the metric has
**two reachable states — 0 and ~0.5** — and correctness saturates in 1–3 rounds.

**This is not fixable by filtering.** The filter can only keep or drop tasks; it cannot create
a band where the model is correct *and* has speed it can actually capture.

And it is not a small-model problem. The 14B teacher's best speedups across 29 tasks:

```
9 x 1.00x   2 x 1.08x   1 x 1.05x   1 x 1.01x   1 x 2.03x
```

Essentially **no model from 0.5B to 14B produces meaningfully faster kernels on this bank on
H100** — though see §2.4 for *why*, which is not what I assumed. The headroom-normalised score
therefore collapses into a correctness indicator, and
"does self-improvement compound?" degenerates into "does the model learn to be correct more
often?" — which saturates almost immediately and leaves `lineage − reset` no room to move.

**The intuition:** a headroom-normalised metric needs the model to sit *inside* the headroom.
If every success lands exactly on the baseline, normalisation buys nothing and the metric is a
correctness bit wearing a continuous disguise. Before running a compounding study, verify that
the population actually occupies the middle of the score range — not just that the range exists
on paper.

**What would make the experiment measurable again** (in rough order of cost):
1. **A substrate where these models can make incremental progress** — the L3 tasks show the
   cliff is too steep and L1/L2 too flat. Needs a genuine middle band.
2. **Models large enough to beat the baseline**, so speed becomes a live dimension. On this
   evidence that is >14B.
3. **Score correctness-acquisition explicitly** (e.g. pass-rate over k) instead of hiding it
   behind a speed normalisation that never engages. This changes what is being claimed and
   must be pre-registered, not chosen after seeing the data.

## 2.3 Fifth zero-producing defect: the compiled cache on a quota-exhausted filesystem

E2 runs whose data dir was on `/scratch` showed 4–5 of 6 rounds at `Q = 0.000`, then sudden
jumps to ~0.50. Runs whose data dir was on `$HOME` showed **zero** such rounds. Perfect
correlation, n=6.

The mechanism: `grade_batch.py` caches compiled baselines under `KA_DATA_DIR`, and
`agent_bench.build_ref_c` does `os.makedirs(compiled_cache)`. On the inode-exhausted group
`/scratch` that raises `EDQUOT`, `build_ref_c` propagates, and `grade_batch`'s except-branch
returns `[False, ...]` for **every** candidate — a whole round of zeros.

**The intuition:** a filesystem quota does not only break writes you can see. It breaks writes
buried inside libraries, and those surface as *scientific* zeros. When results contain long
runs of exact zeros punctuated by normal values, suspect the environment before the model.

---

## 2.4 The sixth defect was my own explanation of the first five

I diagnosed the 0.50 spike as a hardware effect: `torch.compile` is stronger on H100, so the
headroom closed. Plausible, consistent with every number I had, and **wrong**.

The default scorer is `KA_SCORE=eager`:

```python
LK._score(ok, sc if use_compiled else se, ceiling=(ceil if use_compiled else 1.5))
```

With `KA_SCORE` unset it scores **speedup over eager**, normalised by the legacy **fixed 1.5x**
anchor. The compiled baseline and the per-task roofline ceiling are both *unused*. Every
sentence I wrote about `torch.compile` was about a quantity the runs never measured.

**The actual chain**, and it is deterministic end to end:

| link | evidence |
|---|---|
| the prompt steers away from optimising | `OPT` contains *"a plain-torch kernel that is correct beats a fancy one that errors"* |
| models comply | **0 of 86** verified 14B kernels contain triton / CUDA / `load_inline`; 86/86 are pure-PyTorch rewrites |
| a rewrite runs at reference speed | median speedup **1.00x**, 93% below 1.05x |
| the score is pinned | `_score(correct, 1.00) = 0.50` **exactly** |
| the contrast is pinned | both arms at 0.50 ⇒ `lineage − reset ≡ 0` |

So the metric was never broken. It accurately reported that the models were not attempting the
task the benchmark exists to measure. I misread a correct measurement as a dead instrument.

**The intuition, and it inverts §1.** There, a clean zero was a bug. Here a clean zero was a
*finding*, and the bug was in my interpretation. So "suspect the instrument" is not the rule —
it is too weak in one direction and too strong in the other. The rule is:

> **Trace the causal chain from prompt to number before naming a cause, and prefer the
> explanation you can kill with an intervention.** "H100 closes the headroom" was untestable
> with what I had, and I believed it anyway because it explained the data. "The prompt told the
> model not to try" is settled by an A/B on one string, holding tasks, grader, model and seed
> fixed. When two explanations fit equally, the one you can intervene on is worth more than the
> one that merely fits — and a hypothesis that explains everything while predicting nothing new
> is the shape of a wrong one.

Second-order lesson: **a defect can hide in a default argument.** `_score(ok, sp)` silently
supplies `ceiling=1.5`, and `difficulty_filter` calls it that way, so its `best_score` column is
the legacy fixed-anchor score on the eager ratio — *not* the headroom-normalised score the
paper defines, even though it shares a name and a range. Two different quantities under one
name is how this survived review. Check what a default actually binds before reading a column.

---

## 2.5 Defect #7: the harness could not grade triton — root-caused and FIXED

Three-point calibration on a GPU node, before the fix:

| case | result | reason |
|---|---|---|
| plain-torch identical to the reference | **ok=True** | `ok` |
| deliberately wrong kernel | ok=False | `wrong/imprecise` |
| **hand-written correct triton kernel** | **ok=False** | **`FileNotFoundError(2)`** |

Controls behave, so the precheck is trustworthy and the grader does discriminate. Triton failed
on a **missing file** — not a wrong answer, not a tolerance, not a timeout.

### The root cause, and why it hid so well

```
FileNotFoundError: '/cm/shared/apps/nvhpc/26.5/.../compilers/bin/nvc'
```

Triton compiles `driver.c` at runtime to build its CUDA shim, taking the compiler from `$CC`.
Marlowe's `module load` sets `CC` to the **host's** nvhpc compiler; apptainer passes the host
environment into the container, where that path does not exist. **Plain torch never compiles C
at runtime, so only triton broke** — and triton is exactly the thing the benchmark is about.

Fixed centrally in `mrl submit`: `APPTAINERENV_CC=/usr/bin/gcc`. After the fix the same
precheck returns `triton -> ok=True (0.99x)`.

### What this implies, and it is bigger than a bug

Every non-plain-torch kernel in this project failed for an environment reason. So the
benchmark **has never once measured GPU-kernel optimisation** — only correctness-preserving
rewriting — and `KA_PROMPT=kernel`'s 0/29 was void.

It also reframes §2.4. I blamed the plateau on the prompt's *"a plain-torch kernel that is
correct beats a fancy one that errors"* clause. But that clause looks like a **rational
adaptation to a broken grader**: tuned against a harness where triton always failed, plain
torch genuinely *was* the only thing that worked. The prompt and the grader were coupled, and
the grader came first. A prompt that encodes a workaround for a silent defect will look like a
design choice forever, because the defect it compensates for never appears in any result.

> **When a configuration looks inexplicably conservative, check whether it is compensating for
> something broken.** Defensive settings are fossils of failures, and the failure they record
> may still be live.

### The gate that caught it

This is the first defect caught **before it reached prose**, and only because the standing rule
was applied to the *specific new input*: the arm had started producing triton, so the known-good
input had to be triton.

> **When an experiment starts producing a new KIND of artifact, re-instantiate the known-good
> gate for that kind.** "The grader works" was true and useless; "the grader works on triton"
> was the question, and nothing was asking it.

It was also only diagnosable because failure reasons now propagate (§2.10). Before that fix this
was a bare `False`, indistinguishable from a wrong kernel — and "model writes triton, triton
grades False" reads very naturally as *the model is bad at triton*.

Three-point calibration on a GPU node:

| case | result | reason |
|---|---|---|
| plain-torch identical to the reference | **ok=True** | `ok` |
| deliberately wrong kernel | ok=False | `wrong/imprecise` |
| **hand-written correct triton kernel** | **ok=False** | **`FileNotFoundError(2)`** |

The controls behave, so the precheck is trustworthy and the grader does discriminate. And
triton fails on a **missing file** — not a wrong answer, not a tolerance, not a timeout.

So `KA_PROMPT=kernel` returning **0/29** says nothing whatever about whether models can write
kernels. It is an environment fault, and it was one step from being written up as a
pre-registered capability finding.

**This is the first defect caught before it reached prose**, and the only reason is that the
standing rule was applied to the *specific new input*: the arm had started producing triton, so
the known-good input had to be triton. A precheck that only ever asserts what the pipeline
already handles will pass forever and protect nothing.

> **When an experiment starts producing a new KIND of artifact, the known-good gate has to be
> re-instantiated for that kind.** "The grader works" was true and useless; "the grader works on
> triton" was the question, and nobody was asking it.

---

## 2.6 The answer, once the instrument worked: the prompt sets ambition, capability sets success

First honest measurement of the prompt A/B, run after the `CC` fix, classifying every
generation rather than just counting successes (n=16 per arm, 4 tasks x k=4):

| prompt | outcome | n/16 | 95% CI |
|---|---|---|---|
| safe | plain-torch rewrite | **16** | [81%, 100%] |
| safe | attempted a custom kernel | **0** | [0%, 19%] |
| kernel | attempted a custom kernel | **14** | [64%, 97%] |
| kernel | &nbsp;&nbsp;of which **VERIFIED correct** | **1** | [1%, 28%] |
| kernel | &nbsp;&nbsp;of which failed verification | 13 | [57%, 93%] |

Two separable things, which a coverage number alone had conflated for the entire project:

* **The prompt controls whether the model tries.** 0/16 attempts under the published prompt,
  14/16 under the kernel prompt. That is not a capability difference; it is an instruction
  being followed.
* **Capability controls whether the attempt works.** 1 of 14 verified. The model *can* write a
  working triton kernel, and usually does not.

`n=16` is small and the interval is wide, so the point estimate is not the finding. The finding
is that **the rate is neither 0 nor high**.

### Why that is the most useful number of the session

Every measurement problem chased today — the 0.50 floor, the 1.000 ceiling, injection running
out of work by round 2 — had the same cause: **the models had already saturated the task being
measured.** Correctness-preserving rewriting is easy for them, so every metric built on it
piles up against one wall or the other.

Kernel-writing is not saturated. A success rate in this band is far from both walls, which is
exactly the regime a compounding study needs and the one I have spent all day failing to
construct by changing metrics and filtering banks.

> **The substrate problem was never a metric problem.** I tried a headroom score, then a
> pass-rate score, then a 456-task bank, and each time the population sat against a wall. The
> fix was not a better ruler — it was measuring a harder task, which only became possible once
> the grader could verify one.

---

## 2.7 Choosing a metric with resolution, and the honesty cost of choosing it late

The fix for a saturated metric is not a better normalisation — it is measuring the thing that
is actually moving. On this bank what moves is **whether the model gets it right**, not how
fast. So `KA_SCORE=passrate` scores the fraction of the *k* candidates that verify.

Verified against the old metric on the same synthetic outcomes:

| candidates correct | headroom best-of-k | pass-rate |
|---|---|---|
| 1 / 8 | 0.500 | 0.125 |
| 3 / 8 | 0.500 | 0.375 |
| 6 / 8 | 0.500 | 0.750 |
| 8 / 8 | 0.500 | 1.000 |

Best-of-k is **identical** for a model that solves a task once in eight tries and one that
solves it every time. Those are obviously different models. The old metric cannot see the
difference at all; pass-rate gets 8× the resolution on exactly the axis the population occupies.

> **The intuition: a metric's resolution has to match where the population actually sits.**
> Best-of-k is the right statistic when the question is "can it ever", and the wrong one when
> the question is "how reliably". Saturation is the symptom; the wrong *estimator* — not the
> wrong normalisation — is the disease.

Two design details that matter more than they look:

* **The denominator is k, not the number of parseable candidates.** A model that emits nothing
  extractable scores 0, not `NaN` and not "excluded". Defect #3 (the teacher "solving 0/29")
  and the `n_strategies = 0` parse failures both came from a mechanism denominator that
  quietly shrank. Fixing the denominator is how you stop undefined from masquerading as zero.
* **It deliberately does not reward speed.** A 2× kernel and a 1.0× kernel each count once. So
  it tests a **weaker and different claim** than the headroom score, and the two must never be
  pooled or compared.

### The part that is about honesty, not statistics

I chose this metric *after* seeing the old one fail. That is the textbook setup for a
garden-of-forking-paths result, and no amount of it being the right call changes that.

The discipline is not to avoid changing the metric — sometimes the instrument really is broken
— it is to **make the change expensive to abuse**. `docs/PREREGISTRATION.md` Amendment 1 is
written to do that: it states plainly that it is post-hoc, records the measurements that
motivated it, binds the results to a separate experiment set, keeps the original metric primary
where it still has range, requires the paper to say so, and commits to a falsifier *including
the outcome where we publish nothing*.

> **The intuition: a post-hoc metric change is defensible exactly to the degree that it was
> pre-committed before its own results were seen, and indefensible the moment it is allowed to
> silently replace the metric it failed to beat.** The tell of the bad version is that it makes
> the paper's claim stronger; the tell of the good version is that it makes the claim *narrower*
> and names the result that would sink it.

---

## 2.8 Reachability — the range check, turned into a number I can act on

"Check the metric has range" was the rule I wrote after §2, and it was still too vague to stop
me. So: **reachability = the fraction of bank tasks where the frozen base already clears the
scoring anchor.** It costs nothing — it falls straight out of the calibration pass — and it is
known *before* the experiment runs.

| bank | tasks | admitted | reachability |
|---|---|---|---|
| hand-curated | 29 | 23 | **1/29 (3.4%)** |
| generated DSL | 456 | 180 | **57/456 (12.5%)** |

3.4% means that on 28 of 29 tasks every success landed at parity. One attainable non-zero
value; the two-arm contrast was pinned before round 0. Four-fold difference between the banks,
and it would have taken one calibration pass to see it.

**The intuition:** a qualitative gate does not fire. I *had* the rule "check the range" written
down in §1 and still ran E1 on a bank with 3.4% reachability, because nothing forced me to
produce a number and compare it to a threshold. A rule you can satisfy by feeling like you
checked is not a gate. Give every standing rule a statistic and a cut-off, or expect to violate
it while believing you followed it.

**And the limit of this one:** it is measured against eager at a fixed 1.5x anchor, so it says
nothing about `torch.compile`, and generated references may be easy to beat for reasons that do
not transfer. Necessary, not sufficient — see [[route-1-dsl-bank]].

---

## 2.9 I fixed the floor and hit the ceiling

Route 3's first 8 trajectories, under `KA_SCORE=passrate`:

```
control  +0.184 [+0.080, +0.287]  n=4 trajectories  CI EXCLUDES 0
inject   +0.184 [+0.089, +0.278]  n=4 trajectories  CI EXCLUDES 0
delta    +0.000
```

`lineage − reset` is positive with a CI excluding zero — the first non-degenerate compounding
measurement from this hardware. The metric that was frozen at 0.50 now moves. **And it still
does not license a compounding claim**, for three reasons I had to go looking for, because the
result was the one I wanted.

**1. Best-of-N is degenerate under this metric, and would have reversed a published result.**
Pass-rate is a *mean* over draws; best-of-N's entire mechanism is the *max* over draws. As the
budget grew 5×, `C_bestofN` moved **−0.024**. So `lineage − bestofN = +0.676`, positive in
37/37 rounds, is not a matched-budget search comparison — it is a trained model against a
baseline whose mechanism the metric switched off. Reported naively it would have *overturned*
the paper's "search beats training" finding with an artifact.

> **Changing the metric silently changes what every baseline means.** A baseline is only a
> baseline with respect to a scoring rule. Re-examine each one when the rule changes.

**2. Ceiling saturation, now at the top.** `C_lineage == 1.000` in **18/37 rounds (49%)** on a
**5-task** held set. Once lineage pins at the ceiling, `lineage − reset` measures only how far
*reset* fell below it. The magnitude is a lower bound and the across-round trend is unreadable.

**3. The contrast is not training-compute matched.** Lineage keeps its adapter and trains every
round (R × `sft_steps`); reset re-initialises and trains on one round (1 × `sft_steps`). Under a
metric scoring *reliability of correctness*, "more SFT on verified-correct outputs raises the
rate of correct outputs" is close to tautological.

### The positive control did not fail — it ran out of work

`injected_tasks` per round: `[15,2,1,0,0]`, `[19,3,0,0,0,0]`, `[2,0,0,0]`, `[7,0,0,0]`.
Injection targets tasks the student *failed*; by round 2 it solves everything, so there is
nothing left to inject and `delta` is exactly `+0.000`.

**The intuition, and it is the whole session in one line:** I diagnosed a floor effect, built a
metric with range at the floor, and landed straight into a ceiling effect. Saturation is not a
property of the metric — it is a property of **the match between task difficulty and model
ability**, and changing the metric only moves where the wall sits. A 5-task held set that a 1.5B
model solves completely by round 2 cannot support a compounding claim under *any* scoring rule.

The fix is not another metric. It is the 456-task DSL bank with 180 admitted tasks (§2.8), where
the population is not exhausted in two rounds. See [[route-1-dsl-bank]].

---

## 2.9b Applying my own precondition to my own headline — and the retention failure it exposed

A hostile review pointed out that we propose reachability as a precondition and never report it
for the bank behind our own A100 null. Fair, and the answer turned out to matter:

| scale | runs | median C₀ | runs above parity |
|---|---|---|---|
| 0.5B | 12 | 0.100 | 0% |
| 1.5B | 20 | 0.100 | 0% |
| 3B | 19 | 0.381 | 0% |
| 7B | 5 | 0.700 | 60% |
| 14B | 1 | 0.599 | 100% |
| **all** | **57** | **0.200** | **7%** |

**The A100 substrate does not carry the H100 failure signature.** On H100 scores piled at
exactly 0.50 — correct-at-parity, an absorbing state. Here the median is 0.20, *below* parity,
rising monotonically with scale. Most held tasks were not solved at all rather than
solved-at-parity, so the saturation critique does not transfer and the headline null survives
that particular attack.

**The cost of the defence, stated in the same breath:** a base at 0.20 with 4/57 runs above
parity means the *speed* dimension was barely exercised on A100 either. The null is really about
**acquiring correctness through self-training** — narrower than the framing implied, and
narrower than the title promised.

### The part that generalises

I could apply the precondition only **partially**, and only because a coarser statistic (`C0`,
the run-level frozen-base mean) happened to survive. Per-task frozen-base scores were never
retained, so task-level reachability for our own headline is **permanently unrecoverable**.

> **A precondition is worth nothing if the artifacts needed to evaluate it are not retained.**
> Proposing a diagnostic is the easy half; the hard half is keeping, at the time of the run, the
> data that lets someone apply it afterwards — including you, six weeks later, when a reviewer
> asks. Retention is part of the method, not part of the housekeeping.

This is why per-task scores and `_provenance` stamping went in. Both are cheap; both were
missing exactly when needed.

---

## 2.9c An interval that excludes the number printed beside it

The paper's strongest positive read:

> `lineage − bestofN = −0.111 [−0.138, −0.083]` (CI excludes zero; the cluster-robust interval
> `[−0.163, −0.120]` **agrees**)

`−0.111` is not inside `[−0.163, −0.120]`. The two are **different estimands**: the trajectory
estimator weights each run equally; the cluster-robust one weights each *round* equally, so
longer runs pull it. Their point estimates genuinely differ — `−0.111` versus `−0.142` — and the
sentence paired one estimate with the other's interval.

The finding survives (same sign, both intervals exclude zero, the round-weighted effect is
*larger*). What did not survive is the reader's trust: an interval that excludes the number
quoted beside it is the first thing a careful reviewer notices, and it taints every other
interval in the document.

Root cause worth naming: the emitter never wrote `crve_mean` at all, so there was no way to
print the matching estimate even if I had wanted to. **A summary that stores an interval without
its point estimate invites exactly this pairing**, because the only estimate in scope belongs to
a different estimator.

> **Two estimators are not "in agreement" because their intervals overlap or share a sign.**
> Report each one's own point estimate, and if they differ, say why — here, weighting runs
> versus weighting rounds, which is a real modelling choice with a real consequence.

Gated: `check_intervals_contain_estimates()` parses every `est [lo, hi]` in the prose and fails
the build when the estimate falls outside. Negative-tested.

---

## 2.9d The row that vanished because it had no error bar

The headline table printed per-scale trajectory counts of 12 + 20 + 19 + 5 = **56**, beside a
bolded pooled **57**. The missing trajectory was the 14B cell: **n=1**.

`trajectory_level()` returns `None` when there is a single cluster, because no t-interval is
computable. The generator then did `if not t: continue` — dropping the whole **row**, not just
the interval. So a run that counted toward the pooled figure had nothing to show for itself, and
a caption promising that underpowered scales are "marked underpowered" was false for the one
scale that was invisible.

The root cause is a level deeper: `pack()` stored `{"trajectory": None, "crve": None, "round":
None}` for that cell — **no counts at all**. There was no way to print the row even if the
generator had wanted to, because the summary had discarded the only facts that still existed.

> **A summary should record the COUNTS even when it cannot compute a statistic.** "We could not
> estimate this" and "there is nothing here" are different statements, and collapsing them makes
> a table stop summing with no visible cause.

Fixed at the source: `pack()` now always emits `n_trajectories` and `n_rounds_total`, the
generator emits a row with dashes for the uncomputable fields, and `check_table_sums()` fails
the build when per-scale rows do not add to the pooled total. A table a reader cannot add up is
not auditable, which is the whole point of printing it.

---

## 2.9e First real kernel-authoring measurement: correct kernels that are not faster

`t1k-q05` — Qwen2.5-Coder-**0.5B**, the smallest model in the suite, k=12 over 29 tasks, with
provenance confirming `roof_arch=h100`, `triton` builds, `KA_PROMPT=kernel`:

```
solved 8/29 tasks (28%)   10 verified kernels   6 of 10 contain triton/CUDA
custom kernels   median 1.00x   max 1.01x   >1.05x: 0/6
plain-torch      median 1.00x   max 1.01x   >1.05x: 0/4
```

**Two findings, and they point opposite ways.**

*The registered falsifier is not triggered.* Amendment 2 says: if the verified rate is at or near
zero for every scale, the task is unreachable rather than unsaturated and no compounding claim
gets published. A 0.5B model solving 28% of tasks, with 60% of its verified kernels being real
Triton, is nowhere near zero. **Kernel authoring is reachable**, and it is reachable by the
weakest model we have.

*And the speed wall is still there.* Every verified kernel runs at eager speed — median 1.00x,
best 1.01x, none above 1.05x. The model writes a kernel that is **correct and unoptimised**.

That is a sharper result than either the old saturation or the prompt story, because the two
previous explanations are now excluded by construction: it is not the prompt (the model was
asked, and complied), and it is not the grader (triton builds and these kernels verified). A
0.5B model can express a Triton kernel and cannot make it fast.

> **Correctness-reachability and speed-reachability are different properties of a substrate, and
> a benchmark needs to report both.** Everything before this conflated them — a task where the
> model is correct at parity looks identical, in a single score, to one where it cannot compete
> at all. Reachability (§2.8) measured the union; it should be measured separately for each.

**What this does not yet settle:** one model, one seed, the smallest scale. The 3B/7B/14B cells
are running, and the capability *curve* is the result — a flat 1.00x across all scales would be
a much stronger statement than one point at 0.5B.

---

## 2.10 The grader knew why, and threw it away

Chasing why a hand-written triton kernel would not verify, I found this in `grade_batch.py`:

```python
ok, se, sc, msg = AB.grade_c(...)
out.append([bool(ok), float(se), float(sc), float(ceil)])   # msg dropped
```

`grade_c` has **always** produced a reason — `"wrong/imprecise"`, or `repr(e)[:80]` for a raised
exception. It was unpacked into `msg` and then silently discarded on the next line. So every
kernel failure in this project, across every experiment, has been a bare `False` with no cause
attached.

That single dropped variable is the reason four separate harness bugs could all masquerade as
"the model writes bad kernels". Any one of them would have been obvious in a minute if the
failures had carried `ModuleNotFoundError`, or `CUDA error`, or `EDQUOT`, instead of nothing.
I spent days on defects that were annotating themselves the whole time.

**The intuition:** a verifier that returns only a boolean is not falsifiable in practice. Pass/
fail tells you *that* you are wrong, never *how*, and "how" is the entire difference between a
model failure and a harness failure. **Always propagate the reason to the same place the
verdict goes** — not to a log file that is rotated, not behind a debug flag, but attached to
the result row that gets stored and analysed.

Fixed by appending the reason as a 5th field rather than substituting it, since every consumer
indexes `g[0..3]`. Also padded the short-row case in `_grade_isolated`, where a dead subprocess
produced a 3-element row that was indistinguishable from a wrong kernel — it now says
`NO-GRADER-OUTPUT`. And a reference that fails to build now marks all its candidates
`REF-BUILD-FAILED`, which is exactly the shape of the quota bug that once read as five rounds
of the model producing nothing.

---

## 2.11 A positive control is only as strong as the thing it injects

R4's `safe` arm was meant only as the control for the prompt A/B. It replicated the headline
(0 of 118 verified kernels contain a custom kernel; median 1.01x over eager) — and incidentally
exposed that the teacher file E1 and R3 inject from covers **14/29 tasks**, while an identical
model under an identical prompt covers **25/29 tasks**, at *k=6* rather than *k=8*.

The rerun is a strict superset: every task the original solved, plus 11 more, all L2. So the
original harvest quietly stopped covering L2 partway through.

Why this matters more than a stale file: **R3 is a positive control.** Its job is to show the
harness *can* register acquired coverage, so that a flat unaugmented result means something. An
injection reaching 14 tasks instead of 25 makes the control weak, and **a weak positive control
that comes out flat is uninterpretable** — which is precisely the failure this whole document is
about. I would have read "injection did not move it" as evidence about the loop.

**The intuition:** check the *magnitude* of your intervention, not just that it fired. I had
verified injection was firing and being applied correctly. I had never asked whether it was
*large enough to be detectable*. Those are different questions, and only the second one makes a
null informative.

Corollary: re-derive an artifact before depending on it, especially one produced by an earlier,
buggier version of the pipeline. The original teacher was harvested before several grader fixes;
its coverage was a fossil of those bugs, not a property of the model.

---

## 2.12 A stale local copy is a live hazard, not just clutter

Right after the CC fix I archived the pre-fix results on the cluster as `*.pre_ccfix.json`, then
pulled and read `r4_kernel_q14.json` locally: **0/29 solved, 0 kernels**. I was one sentence
from reporting that the kernel arm had failed again.

It was the **pre-fix file**. `mrl pull` had copied it hours earlier, the archive rename happened
only on the cluster, and the local copy sat there looking exactly like a fresh result. The job
that would have produced a real one was still `PENDING`.

What caught it was checking the job state before believing the file — the number was
*suspiciously identical* to the old one, which is the same instinct as §1's "a clean zero is a
bug hypothesis first."

**The intuition:** superseding an artifact has to include every copy of it, and a pull-based
workflow guarantees there are copies. Renaming the source is not enough; the stale copy is the
dangerous one precisely *because* it is the one you read. Where a file has been invalidated by
an environment change, make the reader **refuse it by name** rather than relying on remembering
which is which — `route_readout` now declines any path containing `pre_ccfix` and says why.

Corollary that generalises past this project: **an artifact's validity depends on the
environment that produced it, and nothing in the file records that.** JSON output has no
provenance for "the grader could not compile triton when this was written". If an environment
fix invalidates past results, rename them immediately, everywhere, and teach the tools to refuse
them — memory will not hold.

**Fixed properly rather than by discipline:** `kernelascent/provenance.py` now stamps every
experiment artifact with the git SHA (plus `-dirty`), the semantic env vars (`KA_SCORE`,
`KA_PROMPT`, `CC`, …), torch/CUDA/GPU, and — the part that matters here — whether triton could
actually *build*, not merely import. `import triton` succeeded throughout the defect; what
failed was the C compile, so the stamp records `cc` and `cc_exists` separately.

`is_valid_for_kernels(path)` then answers the question from the file: `True`, `False`, or
**`None` for an unstamped legacy artifact — suspect, not valid.** Defaulting unknown provenance
to "fine" would reproduce exactly the §2.16 failure where a guard's permissive default made it
inert.

> **Write down the thing that will invalidate the result, at the moment you produce it.** You
> cannot reconstruct it later, and the artifacts that most need the label are the ones written
> before you knew the label was needed.

---

## 2.13 I used data I had already documented as contaminated

The E2 write-up claimed a monotone dose-response — mean ΔQ rising to **+0.482** at unbounded cap
— and it reached the paper and the website. It was contamination.

Those runs had their data directory on the inode-exhausted `/scratch`, where `build_ref_c`
raises `EDQUOT` and the grader returns `False` for every candidate. Their signature is
unmistakable:

```
e2_capunbounded_s1   Q0=0.000   Q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.458]
e2_capunbounded_s2   Q0=0.000   Q=[0.0, 0.0, 0.0, 0.0, 0.507, 0.505]
```

`ΔQ = 0.458 − 0.000`. **The entire "gain" is recovery from a spurious zero baseline.** On the
clean runs there is no dose-response at all: −0.046 / +0.021 / −0.063.

**I wrote §2.3 about this exact failure mode, then used the affected runs anyway.** Not because I
forgot it existed — I had the signature written down — but because the contaminated runs were
sitting in a directory that a glob picked up, and nothing between the glob and the paper asked
"is this run valid?".

> **Knowing about a contamination does not protect you from it; only a filter does.** A lesson
> recorded in prose is a lesson you must remember to apply, at exactly the moment you are
> excited about a result and least likely to. Every contamination you can characterise should
> become a predicate in the analysis code the same day you characterise it.

Now mechanical: `route_readout` drops any run with `Q0 == 0` or ≥2 exactly-zero rounds, prints
which and why, and the paper and site carry the correction rather than the quiet fix.

Second-order: the contaminated and clean runs had **identical cell names** in different
directories, so a glob over both returned duplicates that looked like extra seeds. §2.12 was
the same hazard with a stale file. Directory is not provenance — see `kernelascent/provenance.py`.

---

## 2.14 Sweeping for a contamination I had already found once

Having used contaminated runs without noticing (§2.13), the obvious question was whether E2 was
the only place. `scripts/scan_contamination.py` now sweeps every stored artifact.

**The first version reported 11 contaminated files and 7 of them were false positives.** Its
predicate was "has ≥2 exactly-zero rounds", which matches a *weak model legitimately scoring
zero* — which is the published correctness-wall result. I had built a detector that would have
made me retract a real finding.

The distinguishing fact is mechanical: `EDQUOT` fires in `build_ref_c`, **before any candidate
is graded**, so it zeroes *every arm of a round simultaneously*. A weak model zeroes only its
own arm — `C_lineage = 0.0` while `C_reset = 0.3` is a measurement, not a failure. The correct
predicate is **all arms zero in the same round**, never "this series contains zeros".

> **A contamination detector needs a predicate tied to the MECHANISM, not to the symptom.**
> "Scored zero" is the symptom and it has innocent causes; "every arm zeroed at once" can only
> be produced by a failure upstream of grading. Over-detection retracts real results, which is
> the same damage as under-detection, pointed the other way.

Final sweep: 5 contaminated artifacts, 4 in the `/scratch` E2 set and **one round inside the
published trajectory set** (`compounding_q15s8`, 1 of 6 rounds, all arms zero).

### Quantifying it rather than assuming

That last one sits inside the paper's primary result, so "negligible" had to be measured:

```
published (all rounds)      n=57 traj, 228 rounds   -0.0096 [-0.0348, +0.0156]  EQUIV
drop all-arms-zero round    n=57 traj, 227 rounds   -0.0097 [-0.0349, +0.0156]  EQUIV
```

Unchanged. **The published null stands.**

A near-miss worth recording: my first ad-hoc check globbed every `compounding_*` directory and
got n=83, mean −0.033, TOST **not** equivalent — which looks like the primary result failing.
It was not: `equivalence_tost.py` applies a `_CLEAN` tag filter that excludes soak and debug
runs, and I had silently used a different population. **Before reporting that a result does not
replicate, check you are running it on the same set** — an inclusion rule is part of the
result, and reproducing the number without it is not a replication.

---

## 2.15 A gate that cannot see its input, and knowing when to stop widening it

Three separate times today a check reported **clean** while examining nothing:

1. the audit's file list omitted the artifacts the claim lived in;
2. my test harness remapped paths so every `read()` returned `""`;
3. a regex required `"N of M"` to sit immediately before `"verified kernels"`, so
   `"0 of 118, for 0 of 204 across both"` was invisible.

All three are the same failure with different masks, and it is worse than having no check,
because a green result is read as evidence. **The fix is not a better regex. It is a negative
test: plant the error, assert the gate fires.** Anything unverified that way should be assumed
not to work.

### Five instances, one root cause, one helper

By the end of the session this had happened **five times**: a gate matching nothing because a
number or phrase was wrapped in `\textbf{}`, `<b></b>` or `*emphasis*`. The last one flagged the
very sentence *explaining* the problem, because `*mean* over draws` is not `mean over draws`.

So it is now one shared `norm_prose()` — strip LaTeX commands, HTML tags, HTML entities and
Markdown emphasis, collapse whitespace — used by every prose check, rather than five ad-hoc
regexes that each get it slightly wrong.

And the rule that would have caught all five is now a **meta-gate**: every check `main()` runs
must appear in `tests/test_audit_gates.py`, or the suite fails. Verified by adding a stub gate
with no test and watching it break. A check with no negative test is not known to work, and
this file has now produced five proofs of that.

### And the opposite mistake, which I then made

Widening that regex to catch every citation made it bind to unrelated pairs — task coverage
(`14/29`), then a stray `20`. I patched it three times. The right move was to stop and narrow
the *claim* instead: the load-bearing fact is that the **numerator is zero**, not that the
denominators sum correctly. I dropped the arithmetic rule.

> **A gate should assert the smallest thing that would actually be wrong.** Every extra
> condition is a false-alarm source, and a gate that cries wolf gets ignored — which costs more
> than the staleness it might have caught. Prefer one assertion you trust to three you will
> learn to skip past.

Where the prose was genuinely ambiguous (`covers 25/29` with no unit), the honest fix was to fix
the writing, not to teach the checker to guess.

---

## 2.16 An enforcement whose default is "allow" enforces nothing

I wrote a check to make Amendment 1's no-pooling promise mechanical: refuse to report a set of
runs unless every round carries `score_mode == "passrate"`. It was negative-tested — a directory
mixing `eager` and `passrate` was correctly refused — and I committed it satisfied.

Then I looked at a real round record from the live run. There is **no `score_mode` field**.
`eval_tasks` builds the mode into its `stats`, but `lab_compounding` never copied it into the
row it writes. So the checker found no modes at all, and my code said:

```python
if modes and modes != {"passrate"}:   # <- `modes` empty => falls straight through
```

The gate passed everything. My negative test only ever exercised the branch where a mode was
present and wrong — never the one where none was present, which is the actual state of every
run I have.

**The intuition:** a guard has two failure modes and they are not symmetric. Rejecting something
valid is loud and gets fixed in minutes. Accepting something invalid is silent and is what the
guard existed to prevent. **So the default on missing evidence must be refuse, not allow** — and
the negative test has to include the *absent-input* case, not just the wrong-input case.

That is four instances today of a check that could not see its input (§2.15 lists three). This one
is the worst of them, because it was the check specifically protecting a pre-registration
commitment — the promise most likely to be broken by accident and least likely to be noticed.

Fixed in two places, because one alone would have been enough to hide the other: the data now
carries `score_mode` per round, *and* the reader refuses when it is missing, falling back only
to the recorded launch command in `.jobman.tsv`, which is real provenance rather than a guess.

---

## 3. Published claims that turned out to be confounded

### T3 "frontier self-modification is one-shot" — confounded by a harness cap

`lab_track_c.py:51` instructed the model to return **"≤12"** strategy strings, and `:34` showed
only 12 back. Every healthy run filled all 12 slots in round 0 and stayed pinned there for the
rest of the run.

So "the model stopped improving its procedure" and "the harness stopped letting it" are **not
separable** in the published design. 5 of 5 interpretable runs are capped; mean round-0 share
of total gain is 79%.

The gains vs a frozen procedure are real. The *one-shot* interpretation is **retracted**.

E2 and the stored records settle it:

| | evidence |
|---|---|
| closed frontier | **5/5** runs at 11–12 strategies from round 0; **4/5 never grew**. Astra, Sonnet 5, Mistral Large 3 are literally `12,12,12,12,12,12` |
| open 7B, cap varied | 5/6 runs **grew**; 3 exceeded 12, reaching 18 |
| mean ΔQ by cap (clean runs) | 12 → −0.046 · 48 → +0.021 · unbounded → −0.063 · **no dose-response** |

**The intuition:** the cap is a *soft request* — the stored list is never truncated — so what it
really measures is **instruction compliance**. Frontier models comply exactly, and that is
precisely why the artifact looks like a scientific plateau: a well-behaved model hitting an
instruction ceiling is indistinguishable from a model that has run out of ideas. The more
obedient the model, the more convincing the false plateau.

So: **when a quantity saturates at a round number, find out whether you asked for that number.**
12 is not a value a search process drifts to. Six models from four labs landing on exactly 12 is
a specification, not a finding.

Caveats I have to keep attached: n=2 per cap cell, open-vs-closed is confounded with model
family, and the open model does not comply reliably — so the new runs *corroborate*, they do not
replicate. The strong evidence needed no new experiment at all; it was sitting in the published
run records.

### Two "results" that were parse failures

`deepseek-v3.2` and `kimi` have `n_strategies = 0` in **every** round — their output was never
parsed into a strategy list, so the self-modification channel never engaged. The widely-quoted
**−0.234 "self-degrade"** is a parse failure, not a finding.

**The intuition:** before interpreting a per-model number, check the *denominator of the
mechanism* — how many strategies, how many accepted tasks, how many gradient steps. A metric
computed over zero events is undefined, not zero.

### T5 `L−F` — undefined, not null

`L−F` needs the live author to produce accepted valid tasks. Requiring ≥5, only 6/24 open and
5/11 closed runs qualify, and 10/11 closed rows are exactly `0.0`. The headline
`starcoder2-15b L−F = +0.232` rests on **one** accepted authored task.

---

## 4. Statistics: the unit of replication is the trajectory

Rounds inside a run share a base checkpoint, a seed, a held split and an accumulated adapter.
Pooling them as independent overstated the evidence: `BF₀₁ 14.7` at n=228 rounds became **5.6**
at n=57 trajectories — strong → *moderate*.

The conclusion survived (still TOST-equivalent), which is the point: a correction that changes
the number but not the verdict is the good case. The search-beats-training positive actually
*strengthened* under clustering, because that contrast is made within a round.

**The intuition:** ask "what could I have independently re-rolled?" That is the unit. Here it
is the seed, not the round.

---

## 5. Cluster operations (Marlowe)

**Priority is age-dominated, so do not churn.** A fresh job scored `AGE=26372` against
`AGE=1101207` at the top of the queue; fairshare differed by <2×. Age accrues only while
waiting, so cancel-and-resubmit resets it and sends the job to the back. I resubmitted the E2
batch three times tuning its shape and each reset cost more than the better shape recovered.

**But walltime drives the backfill estimate.** A 40-minute job started within 30 minutes while
an identical 2-hour job was estimated 33 hours out. So choose walltime carefully *up front*,
and run long work as **short resumable chunks** (`scripts/jobman.sh`) rather than one long
reservation. Both labs checkpoint per round, so a timeout resumes.

Seen again, starkly: `r4-kernel-14b` requested 2 h and Slurm estimated its start at **+36
hours**, for a job whose three comparable runs took 15:03, 18:54 and 21:57. Resubmitted at 45
minutes.

> **CORRECTION (same evening).** I then claimed that reshape "gained 20 hours", on the strength
> of one `START_TIME` reading that moved from 09-25 06:47 to 09-24 10:08. It has since reverted
> to **09-25 06:47** — for the reshaped job. The estimate is a rolling projection that moves
> with cluster load, and I read a single before/after snapshot as the causal effect of my own
> intervention, with no control for the number's own volatility. That is the same error this
> whole document is about, committed against my own change rather than a model's.
>
> The binding constraint here was never walltime: **650 jobs pending cluster-wide**, top
> priority 4.48M against our 1.10M. Shaping cannot beat being 4x down the list, and since
> cancel-and-resubmit resets accrued age, the reshape probably cost more than it gained.
>
> **So: `START_TIME` is an estimate, not a measurement.** Before attributing a queue improvement
> to a change you made, check the number twice with time in between, and check whether you are
> even the binding constraint — `squeue -h -t PENDING | wc -l` and the top priorities answer
> that in one line.

**So the two rules are not in conflict — they apply to different things.** Do not churn to chase
*priority*; that is age-dominated and resubmitting only loses ground. Do reshape when the
*request itself* is wrong by a large factor, because backfill is governed by the request, not by
age. The test I now use: **is the change evidence-based and large?** A 2 h request for a 20 min
job, with three measured runtimes in hand, is both. "Maybe 3 GPUs would be faster" is neither,
and that is the guess that cost three relaunches.

**A shared quota is not yours to fix.** The 32B probe died with
`OSError [Errno 122] Disk quota exceeded` while *downloading the model* — HF creates a lock
directory per file, and the group is at 1,167,124 / 512,000 inodes. Our own directories account
for ~4,000 of those. So no amount of cleaning on our side helps; the constraint is other members
of the shared allocation. Plan around it (containers, `$HOME`, pre-staged weights) rather than
treating it as something a `rm` can solve, and check whether a model is already cached before
designing an experiment that needs a new one.

**Watch inodes, not gigabytes.** The shared group quota is at 2.3× its *file-count* limit
(1,165,763 / 512,000) while 600 GB of block space sits free. `pip install torch` fails
intermittently; `os.makedirs` killed 2 of 6 jobs *after* they had loaded the model. Containers
(one `.sif`) and `$HOME` (separate user quota) are the way around it.

**Writing partial results is not the same as being able to resume.** `difficulty_filter`
dumped its report after *every* task, which looks like checkpointing — but on restart it
iterated the bank from index 0 and overwrote that report. A 456-task bank could therefore never
finish in 2-hour chunks no matter how many times it was continued: each chunk redid the same
first ~120 tasks. Resumability is a property of the **read** path, not the write path. Before
relying on `jobman continue`, check that the job actually *skips* what it already did.

**The deploy will push your results back at you.** `mrl deploy` rsyncs the repo up, and `data/`
had accumulated run output pulled down from the cluster. Re-uploading it died with
`mkdir ... Disk quota exceeded (122)` mid-sync — on a group already 2.3× over its inode limit,
a *sync of results* was enough to break the code deploy. Exclude every output directory
explicitly; "it's only a few MB" is irrelevant when the binding constraint is file count.

**A shell-style `VAR=value cmd` prefix does not survive a container.** Jobs run as
`apptainer exec IMAGE CMD...` with **no shell**, so the assignment is parsed as the executable
name and the job dies in three seconds with `FATAL: "KA_SCORE=passrate": executable file not
found in $PATH`. Eleven cells died this way at once, and they were queued behind each other so
the first failure did not warn the rest. Prefix with `env` — a real binary that does the
assignment itself, and works identically with or without a shell. This is the third distinct
failure caused by assuming a shell was present (after `ssh host 'cmd'` losing `module`, and
`#SBATCH` not expanding `$HOME`).

**Smoke-test the env plumbing through the real container before a batch**, not just the code:
`apptainer exec $SIF env KA_X=1 python3 -c "import os;print(os.environ.get('KA_X'))"` costs one
second and would have caught this before eleven submissions.

**One-liners will betray you.** `set -- $spec` inside a loop silently lost fields and produced
job cells named `e1c--s1`, collapsing two models onto one manifest entry. Three relaunches lost
to shell quoting. Write the script file.

---

## 6. Standing rules

1. **Never launch a batch without `precheck_grader.sh` passing on the same allocation shape.**
2. **Report reachability for every bank and refuse a speed-scored study below ~10%** (§2.8).
   The qualitative version of this rule did not stop me; the number does.
3. **A clean zero is a bug hypothesis first, a finding second** — but §2.4 is the counterexample:
   once the bug hypotheses are exhausted, a clean zero can be an accurate report of the models
   doing nothing. Do not stop at "the instrument is dead"; find the mechanism that produced it.
4. **Check the mechanism's denominator** before interpreting any per-model number.
5. **Auto-retry only infrastructure failures** (`TIMEOUT`, `NODE_FAIL`, `PREEMPTED`). A code
   failure re-run is wasted allocation and risks a plausible zero.
6. **Trace the chain from prompt to number before naming a cause**, and prefer the explanation
   an intervention can kill over the one that merely fits.
7. **Check what a default argument binds.** `_score(ok, sp)` quietly means `ceiling=1.5`; two
   different quantities shared the name `best_score` for weeks.
8. **Match the estimator to where the population sits**, not just to the quantity of interest.
   Check the score *histogram* has mass away from its bounds before trusting any contrast.
9. **Any metric change made after seeing results is written down as post-hoc**, with its
   motivating numbers, a separate experiment tag, and a falsifier — before its own results land.
10. **Verify a job resumes, not just that it writes.** Re-running must skip completed work.
11. **Keep H100 and A100 results in separate experiment sets** — the roofline constants differ,
   and the two fleets have never been shown comparable. (The original reason given here, that
   H100's `torch.compile` closed the measurement range, was the §2.4 mistake: the default scorer
   never uses the compiled baseline. The separation is still right; that justification was not.)
