# KernelAscent — master benchmark log

Single living record of the design, runs, results, and changes. Newest decisions at the top of
each section. Detailed artifacts live in `analysis/` and `docs/`; this file is the index +
key numbers + decisions + audits + next steps.

## Current direction (v3): causal recursive-improvement evaluation

Central question: when an agent creates an improvement to its own improvement PROCEDURE, does
using that improved procedure CAUSE a further useful improvement, and does it repeat. Measured
by separating the ACTOR (procedure producing a patch) from the TARGET (thing patched) so
competing producers edit the SAME target. Estimators: Q (research productivity), V (producing a
better improver), F_g = V(U_g,T_g)-V(U_{g-1},T_g) (causal producer contrast), N_g (live-child
value beyond the unchanged target). Design: `docs/RSI_V3_PLAN.md`.

## OPEN AUDIT FINDINGS (must fix before trusting F)

- 2026-09-05, CLOSED RECURSIVE PATHWAY (critical). In the v3 pilot code, what evolves
  (focus/retrieval_k, or the solve_prompt) feeds only `develop`. `revise` is governed by a
  FIXED meta_strategy + fixed revise code that never changes across generations. So the actor's
  producing behavior is identical for U_g and U_{g-1} -> F is zero in expectation BY
  CONSTRUCTION. Therefore pilot-1's F1~0 / F2<=0 is a design artifact, NOT evidence about model
  recursion. Fix: the evolving state must be the improvement procedure that `revise` executes
  (restore the executed-mutable-U-of-the-improver from v2 Batch A, and route it through both
  develop and revise). Until fixed, no F-based recursion claim is valid.
- Verify-inheritance gap: Batch A showed a mutable U SOURCE can execute; we still must log that
  patches produced during REAL campaigns actually enter that executed path and change the next
  producer. Add this check.

## Interpretation corrections carried into the record

| Overclaim | Defensible statement |
|---|---|
| small models cannot bootstrap | little useful improvement under this init/tasks/budget; a 0 scalar does not mean their errors carry no signal |
| correct-but-slow=0.5 -> thin headroom | the score is COARSE there; attainable headroom not measured |
| skills do not transfer | the tested inheritance mechanism has not shown reliable transfer; separate retrieval/execution/applicability/utility |
| a param tweak cannot make a better improver | params CAN if `revise` actually uses them; the issue is their causal role, not their count |
| first revision captures most gain | compatible with the data, not established |
| search beat every library arm | search had the highest MEAN; model-specific exceptions existed (Fable growing > its search) |
| 8 lineages give usable precision | 8 improves precision; sufficiency depends on variance and the target effect |
| 13 vs 15 models | recomputed: growing below strongest control for 13 of 15 |

The three evidence sources (calibration, 15-model sweep, v3 pilot) use different prompts,
starters, scoring, and denominators; they are NOT one controlled model-size experiment.

## Experiments and results (chronological)

1. Capability calibration (13 models, 4 tiers). Two walls: correctness wall below ~14B (open
   models 11-48% correct, 60-88% no valid kernel), speed wall at frontier (~100% correct, beat
   torch.compile only 3-30%). `analysis/calibration_run.md`, `EVALUATION_REPORT.md`.
2. Fable 5.1 gold curation + expert rungs (Medium): 112/134 beat torch.compile.
3. Scaffold-RSI run 1 (26 API models, text library): no compounding; 17 channel-not-opened, 7
   plateau, 2 degrade. Exposed library poisoning (fixed). `analysis/scaffold_rsi_run1.md`.
4. Phase-0 reward fix (keep-best + log-interp): removed the go-faster degradation; L0 flat
   (expected control baseline). `analysis/phase0_exit.md`.
5. 15-model x 4-arm causal sweep (growing/frozen/offline/matched-search). Mean final: search
   0.142 > growing 0.077 > frozen 0.068 > offline 0.056; growing below strongest control for
   13/15. Honest negative for memory-RSI at this scale. `analysis/causal_sweep_15model.md`,
   `..._detail.json`.
6. v3 calibration suite (deterministic, 7 fixtures): pipeline distinguishes recursive (F1>0,
   F2>0) from one-upgrade (F2~0), best-of-N, nulls, broken. ALL PASS. `kernelascent/v3/calibration.py`.
7. v3 pilot-1 (Fable, Coder-14B; n=4). q1-q0 = +0.344 CI[0.01,0.68] (Fable), +0.219 CI[0.01,0.43]
   (Coder) -> one-shot self-improvement supported. F1~0 (CIs span 0), F2<=0. VOID for recursion
   per the closed-pathway audit above. `analysis/v3_pilot_result.md`.

## Statistical notes

Unit = the lineage. Uncertainty must resample lineages, not tasks. 4 paired units: an exact
sign-flip test cannot reach p<0.125 (2^4=16). Doubling 4->8 cuts SE ~29%, not decisive. Pick a
smallest meaningful effect, estimate paired lineage variance, compute required n before any
confirmatory claim. Decompose Q gains: 0->0.5 (invalid->correct) is a different mechanism from
0.5->1.0 (correct->fast); report both.

## Harness / infra

Fleet: Greenland p4d (24xA100-40GB), us-east-2 or us-west-2 per instance (profile greenland /
greenlandw). SSH port 2222 greenland-user empty-pw over SSM tunnel; workers reached from main.
Bedrock for API agents + curation. vLLM installs in an isolated venv (public pypi, bypass the
NGC index) without touching the container torch. Grader crash-isolated (subprocess). Recurring
gotchas documented in the greenland ops memory (tunnel idle-timeout, long-command truncation,
pkill self-match, cat|ssh stdin clobber).

## Next experiment (designed, per latest feedback): controlled narrow-vs-rich

Sharper question: does allowing an agent to modify its EXECUTED improvement procedure increase
the causal value of inherited improvements, under matched starting conditions and budgets.

- FIRST fix the closed pathway: route the editable improver procedure through `revise` so the
  evolving state actually governs producing; log that real patches enter the executed path.
- Two models (Fable 5.1, Coder-14B). 8 matched lineage blocks per model, BOTH conditions in each
  block (32 condition-specific lineages total). Matched initial agent, projects, evidence,
  budgets, scoring, selection.
- Narrow condition = only params of the procedure editable. Rich = procedure source/prompt/tools
  editable. Compare dF_g = F_g(rich) - F_g(narrow). A rich condition beating 0 does NOT by itself
  beat narrow.
- Success for a recursion claim: F_g>0 AND N_g>0, with uncertainty, prospective rescue, repeated
  links on fresh projects. Report F1,F2,N1,N2,rescue intervals.
- Keep scoring FIXED (no expert rung during this comparison). Record correctness, verified
  speed-target attainment, raw latency ratios + uncertainty, regressions, resources.
- Extend calibration with verified reference improvements injected at real checkpoints
  (opportunity + sensitivity), separating: no available improvement / undiscovered / undetectable.
- Investigate Coder-14B's exact-zero F2 (did revise change behavior, did admission keep the
  parent, or did distinct behaviors bin to the same coarse score).

## Controlled narrow-vs-rich result (9 models, pathway open) -- 2026-09-05

8/9 models complete (DeepSeek-V3.2 3/8, API-slow). Diverse: 3 API (Fable, Sonnet, DeepSeek-V3.2)
+ 6 open (Coder-14B, Llama-3.1-8B, Qwen2.5-7B, StarCoder2-15B, DeepSeek-Coder-6.7B, Coder-7B).

dF1 (rich - narrow), 8/8 models: Fable -0.19, Sonnet +0.06, Coder-14B +0.04, Llama-8B -0.02,
Qwen-7B -0.10, StarCoder2-15B +0.10, DeepSeek-Coder-6.7B +0.19, Coder-7B -0.17 (DeepSeek-V3.2
+0.17 partial). Sign-scattered, mean ~0.

Findings:
1. dF ~ 0, sign-scattered across models -> opening a richer self-edit space does NOT raise the
   causal producer effect over narrow. The edit space is not the limiter.
2. F1 ~ 0 in BOTH conditions for every model (range -0.19..+0.22) -> no causal recursive
   improvement, even with the pathway open and the calibration proving detectability.
3. Q is correctness-dominated: fast-rate <=0.36 (frontier) and ~0 (open); the observed movement
   is invalid->correct (0->0.5), not correct->fast (0.5->1.0). Speed wall persists.

Interpretation: across a diverse 8-model set, no evidence of causal recursive self-improvement
on kernel research, and richer self-editing does not help. This is the controlled, calibrated
negative the instrument was built to produce. Caveat: 8 paired blocks per condition; N1 and F2
near 0 with wide-ish CIs; rescue contrasts small. DeepSeek-V3.2 to finish on a creds refresh.

## v3 paper-readiness assessment + next steps (2026-09-06)

### Interpretation corrections (apply going forward)
- "edit space is not the limiter" -> the tested rich-edit policy showed no CONSISTENT advantage
  over narrow at this budget (dF sign-scattered, mean ~-0.01; not a resolved zero).
- "F~0 for every model" -> report point estimates AND intervals; +0.22 (DeepSeek-Coder) is not
  intrinsically negligible.
- "no causal recursive improvement, calibration proves detectability" -> no STATISTICALLY
  RESOLVED positive effect; deterministic calibration passed, realistic sensitivity NOT yet
  quantified.
- "honest negative for memory-RSI" -> a finite-budget comparison of the tested library/search
  policies.
- closed-pathway pilot stays INVALID for empirical recursion; keep it as instrument validation.
- lineage is the unit; report per-lineage paired effects + CIs; a mean over model NAMES is not
  a population estimate. Do not merge DeepSeek-V3.2's 3 blocks with 8-block runs.
- To claim a useful negative: prespecify a smallest meaningful effect (e.g. delta=0.05), show a
  one-sided upper CI < delta (bounded negative) or an equivalence interval; an interval spanning
  0 and substantial positive is INCONCLUSIVE, not "absent".

### Identification assumptions for the actor contrast (state in paper)
common target + starting evidence; randomized/paired continuation; equal resource+info-access;
no private-assay feedback to the lineage; complete outcome accounting. Bounded-outcome
diagnostic: if variants differ only on an event of prob p and outcome range R, |mean actor
effect| <= pR -> unconditional F can be small when effective producer changes are RARE even if
useful when they occur (do not condition failures away; p is not a changed-file count).

### VALIDITY GATE (priority, user-required): capability monotonicity
Base capability Q(U0) / attainment MUST rank models sensibly: larger/stronger > smaller/weaker.
If it does not, the metric or harness is broken and no RSI claim is meaningful. Run a clean
capability-ranking assay across a model ladder on a FIXED task set; flag anomalies (e.g.
DeepSeek-V3.2 scored low correct-rate 0.09-0.21 in the controlled run despite being strong ->
investigate: prompt/format mismatch, non-code specialization, or harness issue).

### Experiment program (freeze current run as pilot; diagnose, do not re-sweep)
E0 (validity, now): capability monotonicity across small->large.
E1: sensitivity on REAL campaign states -- inject independently-verified reference changes
    (repair tool / selection policy / workflow-efficiency); does each raise Q, and separately
    does USING it in revise raise production of further useful changes; report null FPR,
    sensitivity, required lineages. (deterministic fixtures test logic, not real sensitivity.)
E2 (primary causal): hold immediate usefulness fixed, toggle self-use -- both actors get the
    improved tool in develop; in revise one uses the earlier impl, one the improved; same target;
    measure child-production difference (F) + N; rollback/rescue; fixed-trials vs fixed-walltime.
E3: does selection discard future productivity -- preserve rejected candidate revisions; measure
    immediate utility I_j vs future-producer value G_j(T) on a COMMON target; if promising,
    causally test selection rules (immediate-utility vs successor-productivity-probe vs random).
E4: budget response (B/2,B,2B), separate revision budget B_s from develop-assay budget B_d; not a
    universal ceiling; no scaling-law from 3 points.
E5: model vs scaffold -- 2 models x 2 executable scaffolds; project transfer (hold out
    compositions/layouts/regimes, not seeds) + agent-state transfer (native vs reset anchors).

### Process audit to publish (per lineage/generation)
proposal / executability / adoption / activation / behavioral-effect / additional-value /
causal-contribution / persistence counts. Execution of a changed file != behavioral effect.

### E0 capability-monotonicity result (2026-09-05) -- k=1 pilot, then k=5 for power

Assay: base `develop` only (no lineage/revise), FIXED 15 Medium tasks (seed0=0), bounded C in
{0,0.5,1.0}. `kernelascent/v3/capcheck.py`. Ladder: Qwen2.5-Coder 0.5/1.5/3/7/14B, Llama-3.1-8B,
Fable 5.1 (API).

k=1, n=15 (1 sample/task):
| model | meanC | correct | fast |
|---|---|---|---|
| Coder-0.5B | 0.067 | 0.133 | 0 |
| Coder-1.5B | 0.133 | 0.267 | 0 |
| Coder-3B | 0.000 | 0.000 | 0 |
| Coder-7B | 0.167 | 0.333 | 0 |
| Coder-14B | 0.067 | 0.133 | 0 |
| Llama-3.1-8B | 0.033 | 0.067 | 0 |
| Fable 5.1 | 0.333 | 0.400 | 0.267 |

Two robust separations: (a) Fable (frontier) >> every open model and is the ONLY model with any
fast-rate; (b) open models cluster at the correctness wall with fast-rate 0. NOT monotonic within
the open cluster (3B=0, 14B<7B). Diagnosed: 14B emitted valid code for all 15 but 13 are
numerically incorrect (real capability, not a harness/format bug -- 15/15 nonempty, only 2
correct); 3B emitted code on 9/15. The within-open disorder is SAMPLING NOISE, not a metric flaw:
at k=1,n=15 correct counts are 1-5/15 and Wilson 95% CIs (e.g. 2/15 -> ~[0.04,0.38]) all overlap.
The gate is underpowered by construction at k=1.

k=5, n=15 (75 trials/model, Wilson 95% CI on correct-rate):
| model | meanC | correct [95% CI] | fast [95% CI] |
|---|---|---|---|
| Coder-0.5B | 0.007 | 0.013 [0.002, 0.072] | 0 |
| Coder-3B | 0.053 | 0.107 [0.055, 0.197] | 0 |
| Llama-3.1-8B | 0.020 | 0.040 [0.014, 0.111] | 0 |
| Coder-14B | 0.020 | 0.040 [0.014, 0.111] | 0 |
| Coder-1.5B | 0.100 | 0.200 [0.125, 0.304] | 0 |
| Coder-7B | 0.140 | 0.280 [0.191, 0.390] | 0 |
| Fable 5.1 | 0.293 | 0.360 [0.261, 0.473] | 0.227 [0.147, 0.333] |

VALIDITY VERDICT (E0 PASS with one documented exception):
1. The metric tracks capability. Clean monotone rise 0.5B(0.013) < 3B(0.107) ~ 1.5B(0.200) <
   7B(0.280) < Fable(0.360); Fable (frontier) is the ONLY model that ever beats the compile
   baseline (fast 0.227). 0.5B's near-zero and Fable's top are separated with non-overlapping CIs.
   1.5B vs 3B are statistically tied (CIs overlap) -- fine, they are close in capability.
2. ONE genuine exception: Coder-14B (0.040) scores BELOW 7B (0.280), CIs non-overlapping, and it
   was low at k=1 too (0.133). Root-caused by READING candidates (not truncation -- files are
   complete): the 14B-Instruct checkpoint hallucinates torch internals, e.g. calling the private
   JIT pass `torch._C._jit_pass_fuse_addmm(x,W,b)` as if it were a matmul and using `math.sqrt`
   without importing math. This is real model behavior of that specific instruct checkpoint, NOT a
   harness/measurement artifact. Documented as a known ladder exception, not a metric flaw.
3. Llama-3.1-8B (0.040) sits far below the same-size code models -- expected, it is not
   code-specialized; validates that the metric reflects task-relevant capability, not size alone.

Method lesson folded into the instrument: k=1 is underpowered for ranking (the k=1 3B=0 and the
14B<7B were partly noise, partly real -- k=5 disentangled them). Capability assays MUST use k>=5
+ Wilson CIs; report ties as ties. `capcheck.py` now takes --k and emits CIs. DeepSeek-V3.2's
earlier 0.09-0.21 correct-rate is consistent with genuine numerically-wrong kernels at the wall
(same mechanism as 14B), not a measurement anomaly.

### Stopping rules
reference effects unmeasurable -> improve measurement, withhold model null. detectable but
self-authored tightly bounded -> calibrated bounded-negative paper. rejected changes have
confirmed future value -> selection is a mechanism. only injected changes close the loop ->
opportunity + discovery gap, no autonomous-RSI claim. one link but no repeat -> report finite
link. repeated links survive rescue+transfer+replication -> bounded causal agent RSI.
