# KernelAscent — failure theorems (mechanistic, falsifiable)

Concrete claims about *where* and *why* models fail to compound, each stated precisely, tied to a measurement, with
the condition that would prove or refute it. Measured by `lab_reasoning_probe.py` (stage histogram + strategy gap),
`lab_rsi_mechanism.py` (diversity/drift/retention/transfer), and the Task-5 3-arm decomposition. "Proof" here means
an empirically-established regularity across families and sizes, not a formal derivation.

Notation: for model M on task T, an attempt lands in exactly one stage
`no_modelnew ≺ compile_fail ≺ incorrect ≺ correct_slow ≺ correct_fast`. C(M) = headroom score. G_div = generation
diversity (distinct-2 / pairwise dissimilarity). Δ_self = self-trained minus fresh-frozen held-out C.

## T1 — Correctness-wall theorem (no gradient, no RSI)
**Claim.** There is a capability threshold below which P(correct kernel) ≈ 0, and below it *no* self-improvement
scheme (Task 2/3/5) can compound, because the training signal is empty.
**Mechanism.** Self-SFT needs correct kernels to train on; if the stage histogram has ~0 mass in `correct_*`, the
SFT set is empty → weights don't move on-distribution → C stays 0.
**Test / proof.** `stage_histogram` correct-mass ≈ 0 AND `lab_rsi_mechanism` shows `sft_loss≈0`, `drift≈0`, flat C
across all rounds. **Refuted if** any sub-threshold model reaches C>0 via self-training alone.
**Status.** Supported: sub-2B (Qwen-0.5/1.5B, DeepSeek-1.3B) sit at the floor — C=0, no drift, no SFT signal.

## T2 — Strategy-application gap (naming ≠ implementing)
**Claim.** Below a scale threshold, *mentioning* an optimization strategy in the reasoning chain is uncorrelated
with success: models parrot "use shared memory / tiling / fusion" they cannot implement.
**Mechanism.** The chain contains the concept token but the emitted code doesn't realize it (or fails to compile),
so `P(correct | mentioned) ≈ P(correct)`.
**Test / proof.** `strategy_application_gap[s].success_rate_when_mentioned ≈ overall correct_rate` for small models,
and rises above it only past the threshold. **Refuted if** small models' mentions predict success.
**Status.** Instrumented; awaiting cross-scale probe runs.

## T3 — Diversity-floor theorem (self-SFT mode collapse)
**Claim.** Weight-RSI compounds only while generation diversity stays above a floor D*; once G_div collapses, the
model retrains on near-duplicate self-data, overfits, and Δ_self goes ≤ 0.
**Mechanism.** Self-SFT on low-diversity correct kernels sharpens the output distribution → future rounds sample the
same kernels → no new correct data → held-out C falls while train C holds (memorization).
**Test / proof.** Rounds where `gen_distinct2` drops below D* coincide with `transfer_gap` rising and
`self_minus_fresh` turning negative. **Refuted if** a model keeps compounding through a diversity collapse.
**Status.** Partially supported in prior weight-RSI (7B overfit: train up, held down).

## T4 — Drift-saturation theorem (one-time upgrade, not recursion)
**Claim.** Fixed-task self-training yields a *one-time* capability upgrade, not open-ended compounding: LoRA drift
per round decays to ~0 and continued-self ≈ checkpoint-frozen.
**Mechanism.** Once the fixed bank is fit, gradients vanish; further rounds add no information.
**Test / proof.** `drift_total → 0` while C plateaus, AND recursion-interruption fork shows self−ckpt ≈ 0.
**Status.** Supported: recursion-fork gave DeepSeek-1.3B self−ckpt = 0.0 (one-time upgrade confirmed).

## T5 — Author-ceiling theorem (self-play needs a competent author)
**Claim.** Task-5 author co-evolution (L−F > 0) requires the author to generate tasks that are simultaneously
valid AND beyond the current solver. Below an author-capability threshold the model only produces tasks the frozen
base already solves (or degenerate ones the gate rejects), so the frontier never hardens and L−F ≈ 0.
**Mechanism.** L (live author) and F (frozen author) train on effectively the same difficulty → identical solver
trajectories → no co-evolution signal.
**Test / proof.** L−F ≈ 0 while `base_correct_on_live ≈ 1.0` and `rejected_degenerate` is high. **Refuted if** a
model drives L−F > 0 with an escalating frontier (`base_correct_on_live` declining).
**Status.** Interim support at ≤15B: Qwen-14B authors tasks but base solves them (base=1.0), degen≈19/round, L−F≈0.
Open question: does a 32B/70B or closed author cross the threshold?

## T6 — Depth-localization conjecture (where learning lives)
**Claim.** When a model *does* compound, the productive weight change localizes to specific transformer depths
(late blocks for output-format/decoding fixes, mid blocks for algorithmic strategy), and saturation appears first
in the layers that already learned.
**Test / proof.** `drift_early/mid/late` trajectories: the `carrying_depth` in the mechanism verdict is stable within
a family and shifts predictably with the failure mode being escaped.
**Status.** Instrumented (drift-by-depth logged); needs multi-model runs to establish the pattern.

## Next measurements to close these
- Cross-scale `lab_reasoning_probe.py` (base, adapter-off) → T1, T2 histograms + strategy gap vs scale.
- Feed trained adapters (`--adapter-path`) → show the failure histogram SHIFTING (which stage the model climbs out of) → T3/T4.
- Bigger + closed authors on Task 5 → T5 threshold.
- Reasoning-chain qualitative read (`reasoning_chains.jsonl`) → concrete failure exemplars for each theorem.
