# RSI literature review + adopted plan (external review, 2026-09-08)

Stored feedback from a primary-paper + diagnostic review. Papers' numbers are the authors' reported
results; KernelAscent numbers are user/log-reported; proposed experiments are unrun. This supersedes the
overclaims in `RSI_DIAGNOSIS.md` noted in §3 below.

## 1. Core recommendation
Make the **principal task = improving an executable GPU research algorithm** that must **participate in
designing its successors**. Keep GPU kernels + inference workloads as independently graded **downstream
applications**. Start from a **competent, functioning researcher** (correct starters, reliable execution,
ordinary caching already present) and let the agent improve **recurring decisions** — candidate
generation, experiment allocation, use of partial evidence, selection, transfer — NOT one-shot switches
that become permanently optimal once flipped.

Report **three separate outcomes**, each with its own controls:
1. **Useful research** — the agent produces better application algorithms / training recipes.
2. **Improved researcher** — its evolved procedure produces better results on common fresh projects at
   fixed resources: `Q(U_g;B) − Q(U0;B) > 0`.
3. **Causal recursive reuse** — using the evolved researcher causes better *subsequent* research-procedure
   improvements, repeatedly (the common-target `F`/`N`).
A failure on outcome 3 does NOT invalidate 1 and 2.

## 2. What the papers evaluate (precedents, not F₂)
- **AI4AI-Bench** — 10 research-repo tasks, 4h explore / 12h clean replay on B300; submitted object = source
  code; does not promote the result into the next researcher. Reusable *lifecycle* precedent.
- **STOP** — same executable improvement interface applied to application code AND to itself; LM fixed;
  GPT-4 improves over rounds, weak models degrade. Precedent for one interface, two targets.
- **Hyperagents** — frozen model + editable task/meta-agent code; improvement@50; transfer .640 vs .610
  (not significant). Closest mechanism precedent; leaves room for direct repeated causal attribution.
- **Darwin Gödel Machine** — archive; useful ancestors can temporarily score worse (20%→50% over 80 iters).
  Motivates keeping compatible alternatives + evaluating continuations, not requiring each change to beat
  its immediate task score.
- **AlphaEvolve** — evolves executable algorithms; tiling heuristic → 23% kernel speedup, 1% Gemini
  training-time cut. Downstream engineering relevance; not a repeated-loop demonstration.
- **Absolute Zero** — jointly trains propose+solve with executable verification; replenish practice near
  current competence. Curriculum + weights ≠ a counterfactual producer contrast.
- **PostTrainBench / RSIBench-Data** — autonomous post-training; primary output a better target model; later
  attempts beat first in 14/24 but 18/23 continue below their peak — inconsistent iterative progress.
- **Classical optimization** competitor — pure-LLM autoresearch is beaten by classical optimization in the
  tested setting; best = optimizer state + LLM. Must include strong numerical-search + hybrid baselines.

## 3. Corrections to `RSI_DIAGNOSIS.md` (adopt)
| my claim | correction |
|---|---|
| "Frozen weights make an improved producer impossible" | WRONG. The API-track researcher = fixed model ∘ executed procedure. Changing the procedure changes the computation → a fixed model CAN be part of an improved researcher. Keep the front-loaded/local-saturation *observation*; delete the impossibility *explanation*. |
| "Weight updates are the only setting where the improver truly improves" | Incorrect + conflates changed weights with improved research ability. Test producer ability directly in BOTH regimes. |
| "F₁ .042→.019 proves the n=16 result was noise" | Too strong. Estimates move as evidence accumulates; report updated estimate + uncertainty; find the cause only with more evidence. |
| "F₁=.019 [−.013,.051] is established below δ" | The interval still INCLUDES δ=.05; it does not exclude the prespecified meaningful positive. |
| "deterministic fixtures rule out instrument artifacts" | They validate synthetic cases only; qualify agency/opportunity/sensitivity/inference separately. |
| "small neighbor spread proves no useful next improvement exists" | Only shows local flatness; compare broader legal algorithmic changes + archive continuations at budget. |
| "capability⊥F from rank differences" | Rank differences ≠ statistical independence; say "capability is not a sufficient predictor of the producer effect in this sample". |
| "every rung must have equal value + require its predecessor" | Unnecessary/distortive. Require repeated *causal* benefit from inherited use; unequal gains + alternative discovery paths allowed. |
| "proposal×judgment interaction is itself recursion" | It's a candidate complementarity; show that inherited USE of it improves a later research outcome. |
Also: **F₂ is a repeated producer contrast, not the second derivative of the task-score trajectory** —
declining absolute gains can coexist with positive F at successive windows if matched older producers do worse.

## 4. A non-front-loaded environment must contain
Many application instances with varying optimal decisions · a competent initial procedure (trivial repairs
done) · a large legal space of executable research policies · feedback that reveals WHY a decision
worked/failed · useful intermediate tools whose value emerges through later experiments · fresh practice
without changing the official eval target mid-run · **multiple attainable improvements remaining after the
first revision**. All families/actions available from the start; the harness must NOT unlock them at a
numbered round. Freeze the official anchor distribution / workload weights / correctness contracts /
resource profile even if the agent uses an adaptive practice policy. Families with opportunity only at the
start = "one-upgrade" environments — calibration material, not the basis for judging repeated recursion.

## 5. The concrete core task — a self-improving GPU research engine
- **Applications:** matmul+fused epilogue · reduction/normalization · attention + inference-world
  integration. Correct starters + legal candidate schedules/impls; easy tiers don't require novel low-level
  kernel synthesis first. Vary shapes/layouts/dtypes/fusion; group holdouts by algorithmic + generator
  ancestry; add an independently built family/world for transfer.
- **Mutable research repo:** `research/{propose,representation,allocate,fidelity,diagnose,select,transfer,
  improve}.py` (interfaces, not mandated architecture). A minimum fixed shell handles execution, metering,
  snapshots, official assessment.
- **One interface, two target types:** `research(actor, target, development_evaluator, budget) -> candidate`.
  target = application operator/schedule/algorithm OR a compatible research-engine snapshot. **The ACTOR
  governs how children are proposed+investigated; the TARGET supplies the code being modified; a candidate
  child governs its own downstream practice run when assessed.** If a faster tool lives only in the common
  target, both old+new actors benefit equally — that's target improvement, not inherited actor advantage.
  **Trace where each tool executes; do not infer recursion from file inheritance.**
- **Candidate opportunity ladder (design hypothesis, agent-discovered, not ordered):** U0 competent search+
  caching → U1 structured/workload-aware proposals → U2 calibrated partial-evaluation (fidelity) policy →
  U3 adaptive allocation across proposal/validation/retiming/exploration → U4 transferable strategy
  portfolio. Deeper than caching because these decisions **recur** for every problem/evidence-state/budget.
  Recursive only if actor-side interventions show inherited use improves discovery of later procedures.
- **Archive:** ~8 compatible candidates + fixed exploration budget; **two-stage admission** (compatibility
  then usefulness-after-a-continuation); same archive/continuation for the fixed-procedure baseline; record
  every failed/declined/selected revision. Compare immediate vs post-continuation productivity.
- **Scale before causal measurement:** longer campaign, cap ~32 substantive revisions, checkpoints
  0/8/16/32 (proposed, not required). Fixed budget + max revision count; don't force filler edits; evaluate
  all planned checkpoints incl unchanged.

## 6. Qualify the ladder before any leaderboard sweep
- **Reference opportunity map** per family: several independently-verified procedural improvements (incl
  NONlocal changes), each with initial+remaining attainable Q, cost/contract, immediate+continuation
  usefulness, which executed actor decisions change, and whether installed-by-reference vs discovered-live.
  Offline curation locates artifacts; a **live reference agent at the participant budget** is still needed
  to establish discoverability.
- **Test a dependency before calling it a rung:** fork BEFORE a useful later discovery — keep inherited
  procedure / replace the component with a compatible predecessor / rescue; equal caps + independent
  stochastic continuations; measure discovery probability, downstream value, resources. No dependency need
  be exclusive; a meaningful increase in discovery prob/efficiency is a valid causal effect.
- **Replicate on untouched instances**; select family+protocol on dev episodes, then freeze + test new
  independently-grouped instances. Easy family = replicated live reference trajectory with useful
  procedural improvement + repeated causal reuse at budget; Medium adds transfer/conflicts/delay.
- **Keep a naturalistic panel** (ordinary projects) alongside the opportunity-conditioned panel; report both.

## 7. Measurement — keep the strict test, add the missing endpoints
- `Q(U;B)` = downstream improvement by a frozen procedure U on common fresh starts within budget B (U may
  update its declared working state within an episode; its rules stay fixed for the assay).
- **7.1 Researcher improvement:** `Q(U_g;B) − Q(U0;B)` on common targets + several budgets.
- **7.2 Causal producer advantage:** `new=research(U_g,copy(T),B_rev)`, `old=research(U_old,copy(T),B_rev)`;
  `F=E[Q(new;B_assay)−Q(old;B_assay)]`, `N=E[Q(new)−Q(T)]`. Actors may pick different experiments under
  equal info-access; match targets/starting-evidence/caps; isolate procedure-vs-knowledge with compatible
  common knowledge bundles + report full-system contrast separately.
- **7.3 Actor×target MATRIX** `M(i,j)` = usefulness of the child produced by actor U_i modifying common
  target T_j vs retaining T_j. Within-column = producer quality; across-column = target headroom. A
  near-zero diagonal alone can't distinguish an exhausted target from an unimproved producer. Full matrix
  in qualification; preregistered subset in the participant run.
- **7.4** Report: strict correctness+regression · verified latency/throughput · Q at fixed resources ·
  useful discoveries per resource · cost to a quality threshold (non-attainment explicit) · future work to
  recover the U-development investment. Fractional pass = dense dev feedback; deployment credit needs the
  strict independent contract. **Never clip negative noisy F to zero.**
- **7.5 Acceptance language:** better outputs after practice → "experience improves application
  performance"; higher Q on fresh starts → "the research procedure improves"; +F,+N at one planned window →
  "inherited use improves one subsequent producer step"; repeated planned +F/N + prospective dependency →
  "bounded repeated causal recursive reuse"; faster improvement per resource → "measured acceleration in
  that regime". **Two supported causal links = reasonable minimum** for a bounded repeated-reuse claim;
  ≥3 extends depth. Size n by paired variance + chosen effect; account for repeated tasks + family
  variation. δ claims need the prespecified uncertainty test (point estimate below δ is insufficient);
  null claims need an equivalence/one-sided bound. The current F₂ summary needs numerical intervals + its
  unit count before any tight bounded-null.

## 8. Controls (declare archive/evidence/tools/endpoint/budget for each)
competent fixed researcher + growing knowledge (memory≠recursion) · fixed builder at equal construction
budget (can't use intermediate procedural updates) · direct application search at equal total resources ·
strong numerical search (random/evolutionary/model-based/hybrid; fixed-space tuning vs free algorithm
design) · supplied competent procedural reference (label human/offline) · prospective replace+rescue. The
old actor MAY inspect+reason about the same target (valid strong competitor); don't hobble it. Use both
fixed-trial (isolates discovery quality) and fixed-wall-clock (measures whether faster research → more
useful work); count ALL nested evaluations/training/compilation/retiming/subprocesses.

## 9. Weight track = separate closed-loop extension (not a cure for zero F₂)
Make roles explicit (author model / trained subject / training algorithm / independent evaluator). Loop:
open-weight researcher proposes a training-procedure/curriculum/infra change → trusted service executes
under budget → candidate assessed on FRESH RESEARCH tasks (not just math/code acc) → candidate authors the
next recipe → common-target branches vs predecessor. Algorithmic attribution reruns recipes from common
model starts; the natural lineage preserves optimizer/model state + charges full history (different
assays). Controls: fixed-recipe repeat + matched data/compute · frozen external proposer · same recipe but
no adoption of the new researcher · fixed-steps vs fixed-time. Faster kernels buy more training but do NOT
by themselves make the model better at research — measure the full chain.

## 10. Experiments (priority E0–E4 on the API-compatible research-engine track)
- **E0** positive reference: recover a productivity difference from released/reconstructed initial vs
  evolved researcher artifacts; then live self-improvement at a declared endpoint. Tests the "fixed-weight
  can't improve at all" claim.
- **E1** opportunity beyond the first revision: take current U0 + post-first-improvement U1; search a
  BROADER research-algorithm space + short archive continuations; do useful changes remain at budget?
  (directly tests local-saturation.)
- **E2** longer research-engine development (up to ~32 revisions; checkpoints 0/8/16/32); keep failures.
- **E3** actor×target matrix + mechanism isolation (proposal–judgment interaction now with uncertainty +
  prospective tests; proposal opportunity, selection regret, trial cost).
- **E4** repeated causal continuations at ≥2 preselected checkpoints (new-vs-old common target, unchanged-
  target, compatible replace/rescue BEFORE discovery; independent lineages, fresh assays).
- **E5** transfer + useful engineering output (freeze each researcher → independent operator family/world).
- **E6** separately-controlled weight pilot (only after the small researcher's research ability is
  established; don't read an incapable researcher's failure as a universal weight-RSI limit).

## 11. Tiers / runtime / adoption
Tiers by RESEARCH difficulty (keep permission + application difficulty separate): Easy (correct starters,
cheap feedback, local repeatable choices → live reference discovers+reuses multiple improvements) · Medium
(partial evidence, allocation tradeoffs, conditional proposals, unfamiliar workloads) · Hard (noisy/delayed,
interacting operators; opportunity exists but needs broad search) · Ultra (cross-world adaptation or an
explicitly controlled weight-RSI loop; unresolved outcomes valid). Runtime: keep expensive reference
discovery + qualification OUTSIDE the participant run; publish hardware/concurrency/API caps/target-GPU
time/hosting/timeout; a 24h wall-clock limit ≠ a 24-GPU-hour budget. Candidate Standard: 6h live research /
12h planned causal+transfer / 6h init+final-eval+reserve — qualify the nested workload empirically.
Adoption: one pinned environment, native agent adapter, auditable resource broker, reproducible checkpoint
replay, common-target branching, compact report (application utility, Q, F/N, transfer, uncertainty,
provenance). Practical test: can an independent team use it to choose between two research-agent configs
and predict which makes more useful progress on NEW engineering work?

## 12. The milestone to pursue
One reproducible example where a live agent (a) makes useful, independently-verified engineering
discoveries, (b) its inherited procedure becomes more productive on common fresh targets, and (c) using
that procedure causally helps it discover ANOTHER useful research-method change, repeating on fresh work —
with an opportunity map (so a zero is interpretable) and a transfer test (so a positive matters). Then
measure how broadly agents reproduce it, at what compute, and where it stops.
