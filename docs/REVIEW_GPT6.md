# External review (GPT-6 Astra, NeurIPS D&B reviewer persona) — award-hardening

**Verdict:** Strong benchmark, but the strongest claims currently outrun the identification. The award
opportunity is *a rigorous science of when optimization feedback compounds vs collapses* — not an
"RSI early-warning leaderboard."

## Biggest weakness: recursion is not yet isolated
self-minus-fresh shows an *evolving* producer beats a *frozen* producer **within this pipeline** — it does
NOT establish an improving *capacity to improve* vs merely accumulating artifacts / curriculum effects.
- T3: gains could be a bigger archive, not a better procedure.
- T4: harness "search" could be ordinary hyperparameter tuning, not recursion.
- Positive 12-round curves alone ≠ compounding.

## Priority additions (ranked)
1. **Causal recursion-interruption fork (THE missing control).** At early/mid/late checkpoints, fork into
   {continued-self production, checkpoint-frozen producer, base-frozen producer}; continue training identically
   with EQUAL sampling/train/eval budget; measure subsequent learning curves on untouched task families.
   Separates ongoing feedback from a one-time producer upgrade. Run for the small-success AND large-failure.
   T3 analogue: frozen procedure + SAME updated archive/retrieval (else archive growth masquerades as procedure gain).
2. **Adversarially credible correctness/timing.** 3 inputs is not "verifiable." Add hidden randomized tests
   (shapes/strides/dtypes/scales/pathological), per-task tolerances, race/sanitizer checks, adversarial submission
   audit, isolated exec/timers. Publish exploit tests + failure rates.
3. **Turn diversity-collapse/forgetting into a CAUSAL result.** Randomize diversity-preserving sampling / replay
   / both + quality-matched controls; show they RESCUE the large model; show collapse precedes deterioration;
   distinguish template-repetition from task-coverage loss. A rescue >> correlated diagnostics.
4. **Validate + decompose the score.** Report correctness-rate vs paired speedup separately — the T4 +0.472 may
   largely be crossing the correctness threshold, not acceleration. Audit roofline vs strong measured impls
   (roofline bound ≠ achievable). Rankings under alternative aggregates; improvement per total resource budget.
5. **Transfer + reproducibility moat.** Family-disjoint splits (seen-op shapes vs genuinely unseen computation
   families); a 2nd GPU architecture; frozen final test set never used for round selection; release full
   trajectories, rejected kernels, harness diffs, training artifacts, open-only reproducible track.

**Across all:** match + report resource budgets (proposal tokens, compile attempts, GPU eval time, train compute,
cumulative archive size) — not candidate count. Seed-level uncertainty for training effects; task-family
resampling for generalization.

## Best-paper framing
Title: **"When Learning to Optimize Compounds — and When It Collapses."** Three contributions:
(1) measurement substrate separating capability / producer-adaptation / procedure-adaptation, adversarially tested;
(2) discovery: feedback helps some systems, destabilizes others **under matched resources**;
(3) mechanism: a controlled intervention that identifies AND reverses the failure.
Do NOT headline "small compounds / large overfits" from one model pair — replicate across families.
Treat T4 as a secondary transfer demo unless equally controlled. Drop "early-warning" as the headline
(no demonstrated bridge to dangerous RSI); keep as motivation.
