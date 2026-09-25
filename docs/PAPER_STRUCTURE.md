# What goes in the main paper, what goes in the appendix, what gets cut

A rubric fixed **before** the final numbers land, so placement is decided by evidential strength
rather than by which result turned out most flattering. Filled in as cells complete.

## The rubric

A result earns **main text** only if it clears all four:

1. **Powered** — n meets the pre-registered target, or the claim is explicitly bounded by its n.
2. **Instrument-validated** — the measurement is shown to have range, and a positive control
   fired on the same hardware and metric.
3. **Policy-robust** — survives the preprocessing choices it depends on (extraction policy,
   scorer, binning), or the dependence is reported.
4. **Load-bearing** — the paper's argument changes if it is removed.

**Appendix** = sound but not load-bearing, or underpowered but worth recording.
**Cut** = cannot be defended under questioning, or is superseded.

## Current assignment

| result | powered | validated | robust | load-bearing | place |
|---|---|---|---|---|---|
| **Instrument validity** (7 defects, gates) | n/a | n/a | n/a | **yes** | **MAIN — lead** |
| **Loss cascade** (348→118→74→4) | yes | yes | policy stated | **yes** | **MAIN** |
| **Reachability** precondition | yes | yes | yes | **yes** | **MAIN** |
| **Compile-gain probe**, reported PER TIER (L1 15.5x, L2 1.26x, L3 4.53x) | yes | n/a | n/a | yes | **MAIN** (short) — the 1.34x single-number median is L2's value and must not stand alone |
| **T1-kernel** shape across scale | 1 seed per scale, endpoint **replicated across routes** | yes | **yes — both policies** | yes | **MAIN**, stated as a **U**: falls 0.5B→14B, recovers at 32B. The 0.5B-vs-14B contrast is a contrast with the *floor*, not with the largest model. |
| ~~T1-kernel *monotone ordering*~~ | — | — | **fails under lenient** | — | **CUT** (retracted; artifact of strict extraction) |
| **T2-kernel** compounding (headroom) | valid prefixes only | control fires 73% | **two bounds, not one** | yes | **MAIN as an instrument result**, not an RSI one |
| **The ceiling, correctly attributed** | n/a | n/a | n/a | **yes** | **MAIN** — best-of-k saturates at 0.50, *and* `--n-held=20` silently delivered 5, capping a round at held·k = 50 verified candidates. Every per-round score inherits the second bound whatever its form. Report both, and do not repeat my first two diagnoses. |
| **T2-kernel** teacher injection | matched n=44/arm | — | — | no | **APPENDIX** — not detected; supersedes an earlier positive |
| **T2-kernel** under pass-rate | running (r3/8, n=2/arm) | **transfers to held-out family** | beats reset AND matched-budget search | **yes — the positive result** | **MAIN if it completes** — see below |
| **prereg** `self−fresh_frozen` | **re-running from round 0** (Amendment 5) | — | registered scorer | **yes** | **MAIN if n≥8** — all pre-15:32 data discarded |
| A100 compounding null | yes (57) | **control never fired** | — | yes | MAIN, flagged not admissible under our own gate |
| Search beats training | yes (57) | yes | yes | yes | **MAIN** |
| Coverage-vs-scale, 0-forensics | yes (133) | yes | yes | supporting | MAIN (compressed) |
| WHY-RSI mechanism | yes (133) | associations only | binning fixed | supporting | **APPENDIX** |
| E2 strategy-cap ablation | n=2/cell | yes | yes | supports a retraction | **APPENDIX** |
| Probe-as-intervention | **n=12** | weakest comparator | — | no | **APPENDIX** |
| T3 procedure-RSI | n=2 complete (H100) | **archive holds 4 kernels over 6 rounds, or 0** | — | **yes — third instance of the thesis** | **MAIN**, as a starved loop rather than a null. `F_g` mean −0.004 and −0.058; seed 2's base procedure scores 0.000 so reachability failed before the rung began |
| T4 closed→open | **n=1** | no seeds, no CI | — | no | **CUT** to appendix line |
| T5 self-play `L−F` | 11 qualifying rows | author-yield failure | **1200-token budget vs T1/T2's 2048** | no | **APPENDIX** with interval *and* the budget gap |
| **32B scale probe** | complete, 29/29, k=6 | ran once the per-GPU memory cap was sized to the hardware | rate, budget-independent | **yes — it changes the T1 shape** | **MAIN**. verify\|attempt recovers to 2.99% from 14B's 0.00% (Fisher p=0.015), so the curve is a U with 14B as the floor, not a decline with 14B as the endpoint. Attempt rate 96.5%, the highest on the ladder. |
| **DSL kernel bank** (124 tasks) | 456 screened, 27% kept | held-out-family split | — | supporting | **MAIN (one line) + APPENDIX** — answers "only 29 tasks" |
| **Cross-rung budget asymmetry** | n/a | n/a | n/a | **yes** | **MAIN — stated as a limitation**: T1/T2 at 2048 tokens, T3/T5 at 1200 |

## The thesis, restated after the starvation finding (2026-09-24, late)

Three rungs of this ladder do not produce nulls. They produce **undefined measurements**, and
for one shared reason: the self-referential loop has no throughput. Each rung was designed around a
*different* channel — weights, procedures, curricula — in *different* labs, and each channel received
almost no input. That is what makes this a claim about benchmark design rather than three unlucky
experiments.

| rung | what fails | measured |
|---|---|---|
| T2 registered primary | solver yields no self-generated data | `n_ex` zero in 8 of 19 rounds, mean 0.84, expected 0.8/round by arithmetic |
| T5 self-play | author yields no accepted tasks | 4 accepted of 173 proposed (2.3%), both seeds below the registered floor of 5 |
| T3 procedure-RSI | archive of verified kernels stays empty | 4 entries over 6 rounds (seed 1), 0 (seed 2) |
| T2 / T5 endpoints | evaluation set too small to separate arms | `held=5` caps a round at 50 verified candidates, every arm pins at 0.50 |

This is the paper's strongest claim and it is not "RSI does not compound". It is that **a
self-improvement benchmark must demonstrate its loop has throughput before any contrast it
reports means anything**, and that the throughput is computable in advance. Expected training
examples per round is n_train x k x yield. At the registered configuration that is
3 x 10 x 0.028 = 0.8, which no number of seeds can rescue. We ran 57 trajectories and 228
rounds of a loop that was arithmetically incapable of compounding, and the arithmetic needed one
line.

Reachability was already in the paper as a precondition. The starvation finding shows it is not
one precondition among several, it is **the** precondition, and the same quantity governs the
solver rung and the author rung alike.

## The framing changed on 2026-09-24

The pass-rate board altered what this paper is. Previously the strongest claim was "a two-arm
self-improvement benchmark manufactures nulls cheaply, here are seven worked examples". That is
still true and still leads. But it now has a companion result that makes it far harder to
dismiss: **the same experimental design returns a tight null under one scorer and a large,
family-transferring effect under another.**

That pairing is the paper. Not "we found nothing and here is why benchmarks are hard", but
"here is a null and the effect it was hiding, measured on the same runs, with the responsible
property of the scorer named exactly": best-of-k cannot distinguish 1-in-k from k-in-k, so every
reliability gain registers as zero.

It also answers the sharpest attack on the old framing — that by our own admissibility criteria
our headline null was inadmissible. Under the new pairing that is not a wound at all: the null
is valid, narrow, and its narrowness is the finding.

**What must not happen**: reporting the pass-rate positive as though it supersedes the headroom
null. It does not. They measure different axes, Amendment 1 forbids pooling them, and the
contribution is the *contrast*, which is destroyed if either is presented as the "real" answer.

## Framing

Lead with the instrument, and lead the instrument with **one table**.

`t2kc` against `t2kp` is the paper's centrepiece. Two cells that differ in `KA_SCORE` and nothing
else — same model, seed, k, lab, bank, arms — where six rounds with both arms at parity report a
headroom contrast within 0.004 of zero against a pass-rate mean of +0.433 on the same rounds.
No statistics, no pooling, no appeal to definitions. A reviewer checks it by reading twelve rows.

Everything else supports that. The seven worked defects explain how a benchmark arrives at such a
state. The starved rungs (T2 at 0.8 examples per round, T3 at an archive of 4 or 0, T5 at 2.3%
author yield) show the second way a self-improvement result can be vacuous, and the throughput
arithmetic shows both are computable before a run rather than after.

The older framing said the contribution was that a two-arm benchmark manufactures nulls cheaply.
That is still true and still the abstract's claim. What changed on 2026-09-24 is that it stopped
being an argument supported by examples and became a measurement with a control.

This is also the only framing immune to the strongest attack available: that by our own proposed
criteria (positive control required, reachability precondition) our own headline null is
inadmissible. Under an instrument-first framing that inadmissibility is the thesis, not a wound.

## What must be added regardless of results

* a bibliography and a related-work section (currently **zero** `\cite` commands)
* a KernelBench comparison — same niche, visibly shared task format
* a datasheet, licences, and a contamination statement (the RSI bank is model-generated)
* non-LLM baselines: `torch.compile max-autotune`, a Triton autotuner, a human reference kernel
