# KernelAscent — award-hardening roadmap (queue)

Goal: convert strong prototype → award-level benchmark. Budget + 72-GPU fleet available.
Status legend: [ ] todo · [~] in progress · [x] done

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

## 3. Rigor pass (non-negotiable) — [ ]
- [ ] >=3-5 seeds per headline cell (weight-RSI + T4). Cross-seed already partial (seed0/1).
- [ ] Bootstrap CIs + significance test on the primary metric.
- [ ] Pre-register PRIMARY metric (proposal: T4 improved-vs-frozen-harness AUC over rounds; T2 self-vs-fresh-frozen final).
- [ ] One-command Docker repro (`docker run ka --track {capability,rsi,procedure,combined}`).

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
