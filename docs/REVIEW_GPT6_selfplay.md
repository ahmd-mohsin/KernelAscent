# GPT-6 Astra (max reasoning) — hardening spec for the self-play RSI tasks (5a open, 5b closed)

## The decisive gap: need a 3-arm design, not 2
Current SELFPLAY vs STATIC cannot isolate co-evolution. Add **FROZEN-AUTHOR**:
- **S — STATIC**: fixed frontier; solver/procedure updates normally.
- **F — FROZEN-AUTHOR**: author frozen at the common start checkpoint, but *receives the evolving frontier +
  solve stats* with the SAME adaptive selection + proposal budget as L. (an adaptive curriculum from a frozen author)
- **L — LIVE-AUTHOR**: the current learner/procedure authors tasks; otherwise identical to F.
Decomposition: **L−S** = total adaptive-curriculum benefit; **F−S** = benefit without updating the author;
**L−F = benefit of author CO-EVOLUTION** (the self-referential signal). Also add an independent/programmatic
adaptive-curriculum control (match resources prospectively, not realized successes post hoc).
Role-specific freezing matters: in 5a freeze only the AUTHOR role, never the whole model (that trivially kills learning).

## Recursion-interruption (the mechanism test, must-have)
At preregistered checkpoints clone the ENTIRE state (weights, optimizer, frontier, archive, strategies, RNG),
fork {continued author-updating} vs {author frozen at ckpt, continued adaptive feedback}; keep SOLVER learning
active in both. 5b: a 2x2 (author frozen/updating × procedure-improver frozen/updating); state whether archive
growth counts as procedure-updating (else it's an uncontrolled channel). Positive interaction ≠ compounding.

## Anti-reward-hacking of self-authored tasks (trusted acceptance pipeline)
Authors propose PROBLEMS, not their own judges. Gates: (1) restricted spec -> trusted harness builds ref+inputs+scorer;
(2) independent correctness w/ HIDDEN randomized + adversarial inputs, per-task fp32 tolerances, NaN/Inf policy;
(3) MEANINGFUL-WORK + attainability gates: reject accidental identities, empty/constant outputs, tiny/no-op
workloads, impossible tolerances, no-defensible-path tasks; (4) SEMANTIC dedup by op-graph/shape-regime/behavior,
not text; (5) GOLDILOCKS selection: tasks INFORMATIVE for the current learner (some solvable), not maximally-hard
(maximizing base-failure = graveyard of impossible tasks); (6) execution security (no ref/test access, no answer
lookup, validate the exact timed artifact). ATTRIBUTION: log provenance, author checkpoint, retries, rejection
reason, repair source, acceptance cost; report all-in perf + model-authored acceptance/use rates + a randomized
BACKSTOP-ONLY comparison if backstop reliance is substantial. Post-hoc on successful authored tasks is biased.

## Fair, non-saturating, non-gameable ladder (3 layers)
(a) IMMUTABLE CORE 30-task ladder (preregistered endpoint); (b) SEALED RESERVE tiers (harder independent families,
activation rules fixed before training, agents never see audits); (c) FINAL CONFIRMATION on fresh independently-
generated instances + different families, only after decisions frozen. Fix per arm: eval dev/inference budget,
persistent state, hardware+timing, task-family weights, score constants. If reserve activates, eval ALL arms +
saved checkpoints on it; report tier-specific curves, never raw averages over changing mixtures.
Scoring: correctness-gated utility q_i = 1[passes private tests]·s_i(T_i); publish correctness coverage + speed
distributions alongside Q (a scalar hides losing correctness for speed). Roofline is descriptive, not guaranteed attainable.

## Better metric than delta-on-held: standardized FUTURE LEARNING EFFICIENCY
delta-on-held measures LEVEL, not capacity-to-improve. At checkpoints clone into a disposable learning assay:
supply fresh SEALED training tasks, allow budget b, eval transfer before/after on disjoint tasks, discard state.
  G_t(b) = [Q_transfer(Adapt_b(S_t)) - Q_transfer(S_t)] / b   (use several b -> efficiency curve)
Meta-improvement estimand:
  M = [G_t2^L(b) - G_t1^L(b)] - [G_t2^F(b) - G_t1^F(b)]
i.e. does co-evolution raise subsequent improvement-per-resource MORE than freezing the author. Positive M =
compounding over the tested horizon/distribution (NOT unbounded RSI). Include adaptation-disabled + search-budget-
matched controls to separate persistent learning from just-more-sampling.

## Strongest defensible evidence package
LIVE beats STATIC and FROZEN-AUTHOR; recursion-interruption reduces later progress; future-learning-efficiency
increases; ALL effects survive independent evaluation + all-in resource accounting. Anything less = "adaptive-
curriculum / procedure-optimization gains," not airtight recursive compounding.
