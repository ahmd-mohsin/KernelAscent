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
