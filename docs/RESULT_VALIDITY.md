# Which published results still stand — audit of 2026-09-24

Written after seven instrument defects, two retractions and a scope correction. One line per
result the paper makes, with the evidence for its status. **Most results survive; what changed
for most of them is what they are a measurement *of*.**

## The key question: does the Triton/`CC` defect invalidate the A100 runs?

**No.** The defect only fires when a model *emits* Triton — plain PyTorch compiles nothing at
runtime, so the grader handled it correctly. Under the published prompt the models never
attempted a kernel:

| harvest | verified kernels | containing Triton/CUDA |
|---|---|---|
| 14B, published prompt, k=8 | 86 | **0** |
| 14B, published prompt, k=6 (independent replication) | 121 | **0** |

The grader defect had nothing to reject. **The A100 numbers are accurate measurements** — of
correctness-preserving *rewriting*, which is not what the title claimed.

---

## Status of each result

| result | status | why |
|---|---|---|
| **T2 compounding null** `−0.010 [−0.035,+0.016]`, n=57, BF₀₁=5.6 | **VALID, NARROWED** | Contamination sweep: 1 of 228 rounds affected; dropping it gives `−0.0097`, verdict unchanged. Narrowed: it is a null about acquiring *correctness*, not optimisation. Its positive control has still never fired — reported as not admissible under our own gate. |
| **Search beats training** `−0.111 [−0.138,−0.083]`, 84% of runs | **VALID** | Strengthens under clustering. Interval/estimate pairing bug fixed (CRVE estimate is `−0.142`, its own interval `[−0.163,−0.120]`). Same sign, both exclude zero. |
| **Coverage vs scale**, pass@K ≫ pass@1, formation-failure forensics | **VALID** | Measured from stored generations; independent of the grader defect and of scoring mode. |
| **WHY-RSI mechanism**, n=133, inverted-U | **VALID, with a caveat** | Associations not causal gates, as already stated. Two band tables disagree between files (n=45/21 vs 40/26) — a reporting inconsistency to unify, not a data problem. |
| **Probe-as-intervention** `+0.124 [+0.029,+0.218]` | **WEAK** | n=12 checkpoints, 5–12 tasks per run, and the comparator is uniform random — the weakest available. Belongs in the appendix, not near the abstract. |
| **T3 "frontier self-modification is one-shot"** | **RETRACTED** | The harness requests "≤12" strategies; 5/5 interpretable runs sit at 11–12 from round 0 and 4/5 never grow. Gains vs a frozen procedure remain real. |
| **T3 cap dose-response** (earlier `+0.482` at unbounded) | **RETRACTED** | Came from `/scratch` runs with the EDQUOT signature (`Q0=0.000`, 4–5 zero rounds). Clean runs show no dose-response: `−0.046 / +0.021 / −0.009`. |
| **T4 closed→open** `+0.472` etc. | **EXPLORATORY** | n=1 per cell, no seeds, no CI, against a registered minimum of 3 seeds. Demoted. |
| **T5 self-play `L−F`** | **UNDEFINED** | Requires accepted authored tasks; only 6/24 open and 5/11 closed runs qualify. The 11 qualifying rows should be reported with an interval rather than the rung declared void. |
| **"H100's `torch.compile` closed the headroom"** | **RETRACTED** | The default scorer never used the compiled baseline. |
| **Any speed-scored board** | **PROVISIONAL** | Measured under a harness that silently rejected every non-PyTorch submission. |

---

## What was never valid and is now known

* **"Roofline-grounded"** is not true of the numbers as produced: `_score(ok, sp)` defaults
  `ceiling=1.5`, so the roofline never entered the scoring path. Three incompatible scoring
  definitions coexisted (prereg: compile baseline + roofline; actual: eager + fixed 1.5×; T1: a
  third). The `prereg-*` runs now in flight use the registered definition.
* **The registered T2 primary was `self − fresh_frozen`**; the paper reported `lineage − reset`
  without declaring the substitution. Declared in Amendment 3; the `prereg-*` runs produce the
  registered contrast.

## What is permanently unrecoverable

Per-task frozen-base scores for the A100 bank were never retained, so task-level reachability
for our own headline cannot be reconstructed — only the run-level `C₀` survived (median 0.200,
which is what shows the A100 substrate did *not* carry the H100 saturation signature).
