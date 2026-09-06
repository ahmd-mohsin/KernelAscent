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
