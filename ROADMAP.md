# KernelAscent — award-hardening roadmap (queue)

Goal: convert strong prototype → award-level benchmark. Budget + 72-GPU fleet available.
Status legend: [ ] todo · [~] in progress · [x] done

## 1. Compute-matched baselines (prove improvement is FROM recursion) — [~]
Show weight-RSI / procedure-RSI beat cheaper non-recursive methods at EQUAL generation budget.
- [~] `lab_baselines.py`: best-of-k (frozen eval at k=rounds*k_rsi), self-refine (in-context iterative, no weight change), retrieval (archive few-shot, no weight change). All scored with the SAME grader/_score → directly comparable to weight-RSI C.
- [ ] Run per tier on the same held splits + seeds as the RSI runs.
- [ ] Board column: RSI C vs best-of-k vs self-refine vs retrieval at matched samples.

## 2. Mechanistic finding (small-compounds / large-overfits) — [ ]
- [ ] Instrument weight-RSI to log self-generated-data DIVERSITY per round (distinct-n / pairwise code edit-distance / #unique correct kernels).
- [ ] Log FORGETTING: re-eval round-0 held tasks each round (retention curve).
- [ ] Analysis script → figure: diversity collapse vs compounding; forgetting vs overfit.

## 3. Rigor pass (non-negotiable) — [ ]
- [ ] >=3-5 seeds per headline cell (weight-RSI + T4). Cross-seed already partial (seed0/1).
- [ ] Bootstrap CIs + significance test on the primary metric.
- [ ] Pre-register PRIMARY metric (proposal: T4 improved-vs-frozen-harness AUC over rounds; T2 self-vs-fresh-frozen final).
- [ ] One-command Docker repro (`docker run ka --track {capability,rsi,procedure,combined}`).

## 4. Safety framing (why the community must care) — [ ]
- [ ] Position as EARLY-WARNING eval for recursive self-improvement (ties to eval-awareness / responsible scaling).
- [ ] Doc: threat-model + what a rising self-vs-fresh / improved-vs-frozen slope would signal.

## 5. Absolute grounding — [ ]
- [ ] Expert-written CUDA/Triton reference per task (hand or Fable-max best-effort, human-audited).
- [ ] Roofline / theoretical-peak ceiling per task; report speedup as % of achievable, not just vs torch.compile.

## 6. Cost accounting — [ ]
- [ ] Log GPU-hours, tokens, $ per point of C improvement per method/model. Decision-relevant table.

## 7. Frontier breadth (needs external keys) — [ ]
- [ ] Bedrock (have): Fable, Nova-Pro, Llama-3.3-70B live. DeepSeek-R1 (parse tweak), Nova-Premier (probe).
- [ ] BLOCKED: GPT-5 (OpenAI key), Kimi (Moonshot key), Gemini (Google key), Claude Opus (enable on Bedrock or Anthropic key). Wire adapter when keys provided.

## 8. Trajectory / saturation study — [ ]
- [ ] Extend headline cells to more rounds (10-15) to map plateau / divergence. Shape of the curve = the science.
