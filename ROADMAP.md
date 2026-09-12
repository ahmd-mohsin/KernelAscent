# KernelAscent — award-hardening roadmap (queue)

Goal: convert strong prototype → award-level benchmark. Budget + 72-GPU fleet available.
Status legend: [ ] todo · [~] in progress · [x] done

## ★★ GPT-6 ASTRA REVIEW — award blockers (supersede; see docs/REVIEW_GPT6.md) ★★
A1. RECURSION-INTERRUPTION FORK (decisive) — [ ] fork checkpoints into continued-self / checkpoint-frozen /
    base-frozen producers; equal budget; subsequent learning curves on untouched families. small+large. T3: frozen
    procedure + same archive. Proves "capacity to improve" not artifact accumulation.
A2. ADVERSARIAL correctness/timing — [ ] hidden randomized tests (shapes/strides/dtypes/pathological), per-task
    tolerances, race/sanitizer checks, adversarial audit, publish exploit tests + failure rates.
A3. CAUSAL diversity/forgetting rescue — [ ] randomize diversity-preserving sampling/replay + quality-matched
    controls; show they RESCUE the large model; collapse-precedes-deterioration timing.
A4. SCORE decomposition/validation — [ ] split correctness-rate vs speedup (is T4 +0.472 just correctness?);
    audit roofline vs strong measured impls; rankings under alt aggregates; per-resource-budget.
A5. TRANSFER + REPRO moat (A100-ONLY; no H100 available) — [ ] family-disjoint splits (seen-op shapes vs
    genuinely UNSEEN computation families) = the real moat; frozen final test set never used for round-selection;
    release full trajectories + rejected kernels + harness diffs; open-only reproducible track. HARDWARE angle
    (H100 dropped): (a) headroom score is already hardware-RELATIVE (% of achievable roofline) -> portability by
    design; (b) substitute cross-arch with measurement-robustness: N-trial locked-clock timing + cross-A100-node
    consistency; (c) note cross-architecture generalization as explicit future work. Also: broad FAMILY REPLICATION
    of small-compounds/large-overfits across many model families (have lots of A100).
A6. FRAMING — [ ] retitle "When Learning to Optimize Compounds and When It Collapses"; measurement+discovery+
    mechanism; replicate small/large across families; T4 secondary; drop early-warning as headline.

## ★ PAPER BLOCKERS (must clear before submission) ★
P1. DEFINITIVE RE-RUN — [~] multi-seed (>=3), headroom-normalized score, on the filtered SOTA banks, all tasks
    T1/T2/T4 + T3. One sweep closes #0 (ceiling-free numbers), #2 (diversity/forgetting figure), #3 (CIs), #6 (cost).
    Gated on the roofline-gated filters finishing (large now sharded 8x on node A; small/mid on node C).
P2. LARGE TIER END-TO-END — [~] DIAGNOSED: not C0 — died in LoRA SFT backward, `MmBackward0 expected device
    meta but got cuda:4`. Root cause: device_map="auto" (an INFERENCE shard layout) used for TRAINING → sharded
    backward yields a meta-device gradient. FIX: large models train in bf16 on a SINGLE GPU (14B~28GB / 15B~30GB
    fit one 40GB card with gradient checkpointing) — no device_map sharding, no meta bug. Apply in the definitive
    re-run: KA_DTYPE=bf16, --self-gpu <one> --fresh-gpu <one>. (Alt for >1 GPU: FSDP, not device_map.)
P3. STATISTICS — [ ] >=3 seeds per headline cell; bootstrap 95% CIs (ci_analysis.py ready); report per pre-reg rule.
P4. MECHANISTIC FIGURE — [ ] populate diversity_self + retention (instrumented) via the re-run; plot diversity
    collapse vs compounding and forgetting vs overfit. Turns H1 from hypothesis into a result.
P5. FRONTIER MODELS — [ ] BLOCKED on keys: GPT-5 (OpenAI), Gemini (Google), Claude-Opus (Anthropic/Bedrock),
    Kimi (Moonshot). Wire non-Bedrock adapter behind the researcher/model callable. Bedrock-side: add DeepSeek-R1
    (parse tweak) + Nova-Premier now.
P6. TASK-QUALITY AUDIT — [ ] human-audit a sample of the 1420 curated tasks (diversity, correctness, non-degenerate)
    + contamination/decontamination check vs known KernelBench-style corpora.
P7. PAPER DRAFT — [ ] write around the pre-registration; anchors = recursion-vs-sampling result + mechanistic figure;
    flagship = closed->open harness RSI; framing = RSI early-warning eval.


## 0. NO BENCHMARK CEILING (core invariant) — [~]
Every measured number must be the MODEL's ceiling, never the benchmark's.
- [x] Headroom-normalized speed score: `_score(ok, sp, ceiling)` credits fraction of the PER-TASK roofline-achievable speedup captured (correct-at-parity=0.5, correct-at-hardware-limit=1.0) whether the limit is 1.1x or 40x. Replaces the fixed-1.5x anchor that capped compute-bound tasks below 1.0 and let memory-bound tasks saturate at a trivial 1.5x. Ceiling computed in build_ref_c (FLOPs+bytes roofline), cached, threaded through grade_batch -> eval_tasks / baselines / combined.
- [x] Roofline-headroom ADMISSION in difficulty_filter (--min-ceiling 1.3): drop un-improvable tasks (torch.compile already near roofline) so no task's ceiling is the benchmark.
- [ ] Re-run headline cells on the SOTA bank under the new score (old runs used the 1.5x anchor; not directly comparable).
## 1. Compute-matched baselines (prove improvement is FROM recursion) — [~]
Show weight-RSI / procedure-RSI beat cheaper non-recursive methods at EQUAL generation budget.
- [~] `lab_baselines.py`: best-of-k (frozen eval at k=rounds*k_rsi), self-refine (in-context iterative, no weight change), retrieval (archive few-shot, no weight change). All scored with the SAME grader/_score → directly comparable to weight-RSI C.
- [ ] Run per tier on the same held splits + seeds as the RSI runs.
- [ ] Board column: RSI C vs best-of-k vs self-refine vs retrieval at matched samples.

## 2. Mechanistic finding (small-compounds / large-overfits) — [~]
- [x] Instrument weight-RSI: per-round self-data DIVERSITY (n_uniq + mean pairwise edit-distinctness) + RETENTION (round-0-solved still solved) logged in weight_rsi.json. Re-run to populate.
- [x] FORGETTING/retention logged per round.
- [ ] Analysis script → figure: diversity collapse vs compounding; forgetting vs overfit.

## 3. Rigor pass (non-negotiable) — [~]
- [ ] >=3-5 seeds per headline cell (weight-RSI + T4). Cross-seed already partial (seed0/1).
- [ ] Bootstrap CIs + significance test on the primary metric.
- [x] Pre-registration locked → docs/PREREGISTRATION.md (metrics, hypotheses H1-H3, admission, scoring, stats, controls).
- [x] One-command Docker repro: kernelascent/v3/run_track.py dispatches --track {capability,rsi,procedure,combined,baselines} -> v3 labs; Dockerfile entrypoint + v3 banks baked; KA_SCORE=compiled default.

## 4. Safety framing (why the community must care) — [x]
- [x] Position as EARLY-WARNING eval for recursive self-improvement (ties to eval-awareness / responsible scaling).
- [x] Doc: threat-model + what a rising self-vs-fresh / improved-vs-frozen slope would signal. → docs/SAFETY.md

## 5. Absolute grounding — [~]
- [ ] Expert-written CUDA/Triton reference per task (hand or Fable-max best-effort, human-audited).
- [x] lab_roofline.py: FLOP-counted roofline ceiling per task + ref %-of-peak; speedup expressible as % of A100 peak. Run over banks.

## 6. Cost accounting — [~]
- [x] weight-RSI logs cum_gpu_hours + n_gens + GPU-hours-per-+0.01-C per run. (tokens/$ next)

## 7. Frontier breadth (needs external keys) — [ ]
- [ ] Bedrock (have): Fable, Nova-Pro, Llama-3.3-70B live. DeepSeek-R1 (parse tweak), Nova-Premier (probe).
- [ ] BLOCKED: GPT-5 (OpenAI key), Kimi (Moonshot key), Gemini (Google key), Claude Opus (enable on Bedrock or Anthropic key). Wire adapter when keys provided.

## 8. Trajectory / saturation study — [ ]
- [ ] Extend headline cells to more rounds (10-15) to map plateau / divergence. Shape of the curve = the science.
