# Cross-model causal sweep (15 models x 4 arms x 2 seeds = 120 campaigns): honest negative

Medium band, private-seed transfer n=12, 4 rounds, log-interp score vs Fable expert rungs.
Arms: growing (recursive library), frozen-nonempty, offline-built, matched-search (best-of-N
on the transfer tasks at the same budget, no library).

Per-arm mean final C (30 campaigns each):
  matched-search  0.142   <- highest
  growing         0.077
  frozen          0.068
  offline         0.056

Per-model growing - max(control): negative for 11 of 15 models. Only deepseek.v3.2 (+0.13,
about the noise floor) and Fable-5.1 (+0.055, below noise) positive. Frontier models
(Sonnet 0.49 search / 0.30 growing, Opus 0.34 / 0.29, Fable 0.16 / 0.31) score higher in
absolute terms but their growing arm does not beat their own controls.

## Conclusion
No evidence of recursive self-improvement at this scale. Recursively growing a verified skill
library does NOT beat matched-compute search or a frozen/offline library; on average it is
worse. Apparent memory gains from the earlier Fable pilot were within noise and do not
survive the search control. This is the result the control arms were built to detect.

## Caveats
- n=12 transfer, 2 seeds: many arm gaps are within the ~0.15 unchanged-state noise floor, so
  the honest statement is "no evidence of a positive recursive effect," not "memory is proven
  harmful."
- The search arm is best-of-(rounds+1) attempts per transfer task by construction (the spec's
  matched-compute competitor). Memory must beat it to count as RSI; it does not.
- This is the v1.5 causal experiment (control arms + noise floor). It does NOT include the
  full v2 solver-improver transplant or prospective ancestry tests (Batches C-E, not built),
  which are the stronger causal probes. But a negative here already sets a high bar: memory
  does not even beat search, so an improver-ability claim would need to clear that first.
