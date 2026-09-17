# Task 5b — Closed-source self-modify (self-play) failure analysis

Per-round the model rewrites its own optimization *strategy* and kernel *archive* (mode=self-modify). Q0 = base quality; Qg = self-generated quality; F_g = round-over-round gain (the recursion signal); Δbase = final gain vs base. **Headline: gain is 68% round-0 (one-shot), 75% of models are non-recursive, strategy convergence (Jaccard) 0.12 — diverse strategies, still no compounding.**

> **Completeness caveat:** runs with 6 rounds (GPT-6-Astra, Sonnet-5, GPT-5.6-sol, GPT-5.6-terra, Mistral-3) are complete and give the reliable mechanism; DeepSeek-V3.2 (1 round), Kimi (2), Opus-5 (partial) were interrupted by instance recycles — their trajectories are suggestive, not final, and are **queued to re-run to full 6 rounds** on the incoming instances. The DeepSeek 'self-degrade' is a single-round drop and may reflect an interrupted run rather than genuine degradation.

## GPT-6-Astra  — **IMPROVING**
- Q0=0.298 → Qg0=0.800 (round-0 jump **+0.502**) → Qg_final=0.854; Δvs-base=+0.556
- recursive gain Σ(F_g>0)=**+0.113** over 6 rounds; archive saturates at **round 2** (5 kernels, 12 strategies)
- **Where it gets stuck:** Starts from a LOW Q0 (large headroom) so it shows sustained positive F_g for a few rounds — but this is headroom being consumed, not recursive self-insight; it too plateaus as it approaches the ceiling.
- **How it could improve:** This is the best case; to show TRUE recursion it must keep gaining after the archive saturates — currently it does not.
- *Example self-written strategy:* "Prioritize correctness: most attempts failed validation. Establish a passing PyTorch-equivalent baseline before optimizing, and change one thing at a time."

## DeepSeek-V3.2  — **SELF-DEGRADE**
- Q0=0.847 → Qg0=0.614 (round-0 jump **-0.233**) → Qg_final=0.614; Δvs-base=-0.234
- recursive gain Σ(F_g>0)=**+0.000** over 1 rounds; archive saturates at **round 1** (7 kernels, 12 strategies)
- **Where it gets stuck:** Self-modification made it WORSE than base (Qg<Q0): its self-written strategy edits hurt correctness/speed and it halted after 1 round.
- **How it could improve:** Add a guardrail that rejects self-edits that regress held quality (keep-best); the model over-edited a already-good baseline.
- *Example self-written strategy:* "Verify numerical equivalence before/after optimization"

## GPT-5.6-sol  — **ONE-SHOT-PLATEAU**
- Q0=0.813 → Qg0=0.870 (round-0 jump **+0.057**) → Qg_final=0.889; Δvs-base=+0.075
- recursive gain Σ(F_g>0)=**+0.041** over 6 rounds; archive saturates at **round 3** (15 kernels, 12 strategies)
- **Where it gets stuck:** Large round-0 strategy dump, then F_g collapses to ~0 by r1-2 as the kernel archive stops growing (strategy-space exhaustion): it regenerates variants of what it already found.
- **How it could improve:** Needs a novelty/curriculum pressure that forces proposing tasks/strategies OUTSIDE the current archive; without it the loop is a one-shot distillation, not recursion.
- *Example self-written strategy:* "Prioritize l3_* and compute-heavy l1_* tasks; they show the largest compile speedups (1.6–2.6x)."

## GPT-5.6-terra  — **CEILING**
- Q0=0.851 → Qg0=0.852 (round-0 jump **+0.001**) → Qg_final=0.850; Δvs-base=-0.001
- recursive gain Σ(F_g>0)=**+0.004** over 6 rounds; archive saturates at **round 2** (11 kernels, 10 strategies)
- **Where it gets stuck:** Starts near the roofline-normalized ceiling (high Q0), so self-modification has almost no headroom; round-0 jump and F_g are both ~0.
- **How it could improve:** Give it a HARDER task tier (roofline headroom) so improvement is measurable; on the current bank it is saturated, not failing.
- *Example self-written strategy:* "Prioritize l3 tasks: torch.compile produced consistent 1.07–1.23x gains; simplify Python/control flow to maximize fusion."

## Kimi  — **STALL**
- Q0=0.752 → Qg0=0.825 (round-0 jump **+0.073**) → Qg_final=0.813; Δvs-base=+0.062
- recursive gain Σ(F_g>0)=**+0.113** over 2 rounds; archive saturates at **round 1** (4 kernels, 12 strategies)
- **Where it gets stuck:** Archive never grows past a handful of kernels (<5): it fails to accumulate a working strategy library at all, so there is nothing to build on.
- **How it could improve:** Fix task-admission/parse so its proposals become valid archive entries; it is bottlenecked on producing usable kernels, not on ideas.
- *Example self-written strategy:* "Classify kernels as L1(elementwise), L2(latency-bound), L3(compute-heavy); apply CUDA graphs only to L2, torch.compile only to L3, keep L1 eager to avoid 0.3-0.7x slowdown"

## Mistral-3  — **IMPROVING**
- Q0=0.359 → Qg0=0.494 (round-0 jump **+0.135**) → Qg_final=0.669; Δvs-base=+0.310
- recursive gain Σ(F_g>0)=**+0.115** over 6 rounds; archive saturates at **round 4** (12 kernels, 12 strategies)
- **Where it gets stuck:** Starts from a LOW Q0 (large headroom) so it shows sustained positive F_g for a few rounds — but this is headroom being consumed, not recursive self-insight; it too plateaus as it approaches the ceiling.
- **How it could improve:** This is the best case; to show TRUE recursion it must keep gaining after the archive saturates — currently it does not.
- *Example self-written strategy:* "Enable CUDA graphs for small models via `torch.cuda.CUDAGraph` (latency-bound cases)"

## Opus-5  — **CEILING**
- Q0=0.878 → Qg0=0.910 (round-0 jump **+0.032**) → Qg_final=0.910; Δvs-base=+0.032
- recursive gain Σ(F_g>0)=**+0.000** over 1 rounds; archive saturates at **round 1** (20 kernels, 12 strategies)
- **Where it gets stuck:** Starts near the roofline-normalized ceiling (high Q0), so self-modification has almost no headroom; round-0 jump and F_g are both ~0.
- **How it could improve:** Give it a HARDER task tier (roofline headroom) so improvement is measurable; on the current bank it is saturated, not failing.
- *Example self-written strategy:* "Scoreboard: L1/L3 are saturated at C=1.00 — for those spend <10% of effort: quick in-process A/B of {eager, torch.compile(max-autotune-no-cudagraphs)}, ship the winner, stop. ALL r"

## Sonnet-5  — **ONE-SHOT-PLATEAU**
- Q0=0.493 → Qg0=0.888 (round-0 jump **+0.395**) → Qg_final=0.894; Δvs-base=+0.401
- recursive gain Σ(F_g>0)=**+0.069** over 6 rounds; archive saturates at **round 2** (9 kernels, 12 strategies)
- **Where it gets stuck:** Large round-0 strategy dump, then F_g collapses to ~0 by r1-2 as the kernel archive stops growing (strategy-space exhaustion): it regenerates variants of what it already found.
- **How it could improve:** Needs a novelty/curriculum pressure that forces proposing tasks/strategies OUTSIDE the current archive; without it the loop is a one-shot distillation, not recursion.
- *Example self-written strategy:* "Always verify numerical correctness against eager reference on all provided test inputs before trusting speedup; a fast but incorrect kernel scores zero."
