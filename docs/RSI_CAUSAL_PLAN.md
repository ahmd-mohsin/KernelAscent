# KernelAscent: Causal Evaluation of Recursive Improvement (plan v0.2)

Supersedes the framing in `RSI_DEPTH_PLAN.md`. That document's depth ladder becomes an
edit-permission taxonomy, not a measure of recursion. The central experiment is now the
causal inheritance of improvement ability: does an agent become better at generating future
improvements, and which inherited changes cause that effect.

The current results (in `analysis/EVALUATION_REPORT.md`) establish capability calibration
and a preliminary memory-transfer signal. They do not establish recursive improvement. This
plan is what would.

---

## 1. The distinction we are missing

A rising task score conflates five things. We must separate them.

| Observation | Supported conclusion |
|---|---|
| C(S_k) rises | task capability improves |
| artifacts help fresh tasks | transferable artifact learning |
| G(U_k) > G(U_0) on common anchors | improved ability to generate improvements |
| an inherited change causes later useful changes | causal recursive dependence |
| improvement per unit resource rises | acceleration in the measured regime |

A checkpoint is A_k = (S_k, U_k): S_k is the solver (prompts, reusable kernels, retrieval,
execution tools); U_k is the improver (how practice is selected, failures analyzed, edits
proposed, artifacts admitted). C(S_k; b) is downstream task capability under solve budget b.
G(U_k; B) is the improver's gain, measured by letting U_k improve a COMMON set of anchor
solvers under budget B and averaging C(descendant) - C(anchor). Evaluating an improver only
on its own companion solver confounds improvement ability with headroom.

---

## 2. Interpretation corrections (applied to the record)

| Was claimed | Evidence | Correction |
|---|---|---|
| Fable shows a compounding L2 slope | increments +0.165, +0.028, -0.003: a jump then a plateau | a preliminary transfer gain; test whether later artifacts improve future learning efficiency |
| Opus is ceiling-limited at 0.20 | 0.20 is far below expert parity; cause unknown | test interference, retrieval quality, task mismatch, attainable headroom |
| Coder-7B cannot bootstrap (banks 0 skills) | true only under a speed-win admission rule | test admitting correct-but-slow skills and repair procedures |
| keep-best + smoother reward removed degradation | keep-best makes retained score monotone by construction | separate retention from proposal-quality change |
| frozen weights imply refinement cannot improve | within-task search can improve without cross-task persistence | distinguish within-task opt, cross-task learning, improved learning ability |
| the limitation is code validity not reasoning | compile errors show where execution failed, not the cognitive cause | use controlled interventions before claiming mechanism |
| private seeds prevent contamination | they stop exact reuse, not identical algorithmic templates | add structural and compositional holdouts |
| torch.compile is a roofline | it is a software baseline, not a hardware bound | rename to "compile baseline" throughout; measure hardware bounds separately |

Two reporting fixes: always give explicit denominators (candidate-level 29.8% correctness vs
task-level correctness answer different questions); and retire "gradient" unless a
gradient-based update exists (continuous scores give finer feedback, they do not induce
learning).

---

## 3. The unchanged-state noise floor (do first, nearly free)

Coder-7B scored 0, 0.15, 0, 0 while banking zero skills. If the complete persistent state was
identical across rounds, that variation is protocol noise, not learning. Audit artifact
hashes, transfer-task identities, prompts, sampling settings, and all persisted notes across
rounds. The spread of an unchanged-state trajectory is an estimate of how large an apparent
improvement the protocol can manufacture with no learning at all, and every real effect must
clear it.

---

## 4. Central experiment: solver-improver transplant

At checkpoints k = 0, 2, 4, 8 save S_k and U_k separately. Evaluate four combinations, each
with the same fresh practice block, the same budget, descendants scored on untouched tasks:

| solver | improver | tests |
|---|---|---|
| S_0 | U_0 | original system |
| S_k | U_0 | better solver, original improver |
| S_0 | U_k | better improver, original solver |
| S_k | U_k | joint evolved system |

The decisive contrast is C(U_k(S_0; B)) - C(U_0(S_0; B)). If positive and reproducible, the
later improver contributes beyond the accumulated solver library, which is improved
improvement ability. Repeat on several anchor solvers, including one with a different initial
library; a procedure that only works with its own companion solver has shown narrower transfer.

Interfaces to freeze now so a transplant cannot secretly carry the whole solution library:

    solve(task, solver_state, budget) -> candidate
    improve(solver_state, practice_tasks, budget) -> updated_solver_state

solver_state and improver_state must have explicit, separable dependencies.

---

## 5. Causal depth by intervention, not permission level

L1-L5 stay as an edit-permission taxonomy; they are not intrinsically ordered depths of
recursion (a text note can change future decisions; a big code cache can be passive; nested
levels mean an L4 agent may succeed entirely through L2 behavior). Measure depth by
intervening on artifact ancestry. For a candidate chain (stride checker -> valid autotuner ->
reusable fused kernel):

1. Fork immediately BEFORE the autotuner is developed.
2. One branch keeps the stride checker; another replaces it with the original implementation.
3. Run matched continuation budgets over multiple stochastic branches.
4. Add a rescue condition that restores the checker.

Measure whether the checker raises the probability of discovering the autotuner and whether
that raises held-out performance. Deleting an artifact only after all descendants exist does
not test whether it helped discover them; the intervention must precede the discovery. Report
supported dependency-chain lengths and effect sizes, never the longest observed chain as an
intrinsic maximum (it depends on budget, task distribution, and interventions tested).

---

## 6. Controls (the nonrecursive competitor is the key one)

Strongest baseline: a frozen builder gets the same total practice budget and builds a
library, but cannot use intermediate library improvements to enhance its own subsequent
construction. Compare to recursive inheritance under matched practice and resources. This
asks whether intermediate gains help produce the next gain, not whether a final library helps.

Full control suite:
- frozen NONEMPTY library (freeze an early useful checkpoint; empty round-0 is too weak)
- offline-built library (does recursive accumulation beat ordinary library construction)
- example retrieval (store full successful solutions, no skill abstraction)
- fixed improver with a growing solver (ordinary accumulation, no improver evolution)
- matched-budget task search (spend the resources directly on target tasks)
- compatible irrelevant artifacts (control context volume and tool availability without
  breaking execution)

Keep the poisoned-library control only as a robustness check; beating a corrupted library is
weak evidence because corruption directly damages performance. Report two compute regimes
separately: equal deployment budget (does prior practice improve per-task efficiency) and
equal total budget (was practice worth its cost vs direct search). Report tokens, GPU eval
time, wall-clock, and retrieval/execution overhead; equal token counts are insufficient if one
arm launches far more autotuning trials.

---

## 7. Does the admission policy create the apparent capability window

Weak agents currently must produce a speed win before any reusable support is banked, but the
support they need may be a reliable way to produce valid code. Split admission by artifact type:

| artifact | admission evidence |
|---|---|
| correctness skill | compiles and passes its declared semantic contract |
| optimization skill | improves runtime with independently confirmed timing |
| repair procedure | repairs a held-out set of relevant failures |
| improvement tool | improves downstream discovery yield under a fixed budget |

For weak agents compare four initializations under the same policy and budget: empty; small
library of verified correct-but-slow examples; small library of fast kernels; small library of
API/layout/compilation repair procedures. If correct-but-slow scaffolding unlocks later speed
gains, the conclusion changes from "small models cannot bootstrap" to "requiring performance
wins before admitting knowledge prevents acquiring the prerequisites for performance."

---

## 8. Verification and timing as scientific contribution

Four random inputs plus input-sensitivity is not broad correctness and does not stop all
timing manipulation (see METR on kernel-eval exploitation; KernelBench-Verified on hidden
inputs and realistic baselines).

- Numerical contracts: operation-appropriate tolerances, supported layouts, dtypes, parameter
  regimes; absolute error and targeted checks, not only global relative L2 (which hides local
  errors).
- Hidden structured cases: tail blocks, irregular dims, extreme and near-zero values, masks,
  supported noncontiguous layouts, fresh parameters.
- Separate the agent's development tester (L3-editable) from the official oracle, timer, and
  hidden tests (never agent-editable).
- Re-time selected winners independently with interleaved baseline/candidate blocks and report
  uncertainty for marginal wins.

Scoring: make raw speed ratio and correctness primary; keep expert-normalized score secondary
with explicit near-zero-denominator handling (ln t_eager - ln t_expert is unstable when the two
are close; clipping hides the distortion) and explicit handling of tasks where the reference
does not beat eager. Operational aggregate: a verified-fallback policy (use the baseline when
no validated improvement exists) reported as geometric-mean speedup across all tasks, with
candidate correctness and runtime reported separately so fallback does not mask failures.

---

## 9. Transfer harder than a seed change

Evaluation panels, with canonicalized computation-graph dedup across splits:

| panel | what changes |
|---|---|
| instance | inputs and shapes within familiar templates |
| compositional | new combinations of familiar ops |
| structural | layouts, reduction patterns, masking, fusion boundaries |
| family | an op family excluded from practice |
| systems | compiler version, GPU, execution regime |

Separate practice, development validation, and final test; repeatedly inspecting the
private-transfer curve while tuning the method turns it into development data. Each skill
stores its contract, provenance, dependencies, verified regimes, and known failures; "passed
one practice task" is not evidence of reusability.

---

## 10. The GPU-specific loop: does improvement buy more improvement

If the agent speeds up a component of its own evaluation or local inference pipeline, test
whether the saved time enables more validated experiments that produce further transferable
gains. Two conditions: fixed trial count (better discoveries per trial) and fixed wall-clock
(more trials per hour). If eval is fraction f of runtime and the component is sped up by s,
idealized throughput gain is bounded by 1 / ((1-f) + f/s). Measure f first; if API waiting
dominates, faster GPU eval barely matters, and that negative result is itself a real boundary
on the feedback loop.

---

## 11. Experiment order

| # | experiment | design | decides |
|---|---|---|---|
| 1 | reproduce L2 transfer | growing vs frozen-nonempty vs offline-built vs matched search | does accumulated memory reliably help |
| 2 | separate retention and feedback | 2x2 keep-best on/off x binary/continuous feedback | what removed degradation |
| 3 | bootstrap gate | admit speed-only vs correct skills vs repair procedures | are small models blocked by the protocol |
| 4 | solver-improver transplant | the four combinations, from common anchors | does improvement ability itself improve |
| 5 | ancestry intervention | remove / replace / rescue selected ancestors before continuation | do gains depend causally on earlier ones |
| 6 | structural transfer | hold out compositions, layouts, op families | learning beyond templates |
| 7 | real deployment value | integrate selected kernels into an inference workload | do local gains help the real system |

First confirmatory campaign: three agents, eight independent campaigns per arm, eight rounds,
about 100 held-out tasks over multiple templates. The experimental unit is the campaign, not
the evaluation: 100 evals of one artifact are not 100 independent observations. Spend extra
compute on independent campaigns and randomized continuations, not more model names. Use pilot
variance to size for a preregistered smallest meaningful effect.

---

## 12. Paper

Title: KernelAscent: Causal Evaluation of Recursive Improvement in GPU Kernel Agents.
Three central results to deliver:
1. Attribution: growing artifacts beat strong nonrecursive construction and search controls.
2. Improvement ability: later improvers produce better descendants from common anchors.
3. Causal inheritance: intervention and rescue show particular earlier changes enable later gains.

Keep the capability sweep as calibration. Keep L2 as a pilot until reproduced with controls.
Defer LoRA / full L5 until the causal protocol works. Position narrowly and honestly against
Voyager (executable skill libraries), STOP (recursive scaffold optimization), Darwin Godel
Machine (agent self-modification with archives), HyperAgents (improving the improvement
mechanism), and KernelBench (kernel correctness+speed substrate): the contribution is the
controlled separation of accumulated knowledge, improved improvement procedures, extra compute,
and inherited causal dependencies, not any single ingredient.
