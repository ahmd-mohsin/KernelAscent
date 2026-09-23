# Running intuitions — KernelAscent

Hard-won operational and scientific judgement, recorded as it is learned. Newest at the top of
each section. This is the file to read before designing the next experiment.

Companion to `BENCHMARK_LOG.md` (dated record of numbers) and `PROGRESS.md` (state).
`~/.marlowe/MARLOWE_FACTS.md` holds cluster facts.

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
| 5 | compiled-baseline cache hit an inode quota (§2c) | "4 of 6 rounds produced nothing" |

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

## 2. Ceiling saturation: the 2026-09-23 H100 result

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
wrong; see §2e.** The default scorer never touches the compiled baseline. Struck through rather
than deleted, because the wrong answer was plausible, fitted the data, and cost a day.

**The intuition:** a headroom-normalised score silently becomes a correctness-only score when
the baseline is strong. `0.5` is then an absorbing state. Always look at the *distribution* of
scores, not just their mean — a spike at exactly 0.5 means the speed dimension is dead.

**Fix in flight:** `difficulty_filter.py --min-ceiling 1.3` against an H100 anchor, to admit
only tasks where the roofline leaves real headroom over `torch.compile`. Re-run E1 after.

---

## 2b. The deeper problem: on H100 the score is effectively BINARY

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
H100** — though see §2e for *why*, which is not what I assumed. The headroom-normalised score
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

## 2c. Fifth zero-producing defect: the compiled cache on a quota-exhausted filesystem

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

## 2f. Reachability — the range check, turned into a number I can act on

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

## 2e. The sixth defect was my own explanation of the first five

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

## 2d. Choosing a metric with resolution, and the honesty cost of choosing it late

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

## 3. Published claims that turned out to be confounded

### T3 "frontier self-modification is one-shot" — confounded by a harness cap

`lab_track_c.py:51` instructed the model to return **"≤12"** strategy strings, and `:34` showed
only 12 back. Every healthy run filled all 12 slots in round 0 and stayed pinned there for the
rest of the run.

So "the model stopped improving its procedure" and "the harness stopped letting it" are **not
separable** in the published design. 5 of 5 interpretable runs are capped; mean round-0 share
of total gain is 79%.

The gains vs a frozen procedure are real. The *one-shot* interpretation is not earned until
E2 (cap ∈ {12, 48, unbounded}) says whether the plateau survives.

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
2. **Report reachability for every bank and refuse a speed-scored study below ~10%** (§2f).
   The qualitative version of this rule did not stop me; the number does.
3. **A clean zero is a bug hypothesis first, a finding second** — but §2e is the counterexample:
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
   H100's `torch.compile` closed the measurement range, was the §2e mistake: the default scorer
   never uses the compiled baseline. The separation is still right; that justification was not.)
