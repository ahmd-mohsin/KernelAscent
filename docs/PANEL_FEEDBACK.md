# Frontier panel feedback — next runs (2026-09-15)

Polled via Bedrock `converse` (max tokens). Fable-5.1 retry pending.

## GPT-6 Astra

**Prioritize a locked, task-held-out reranking replication—not another intervention type.** Allocate **48 GPUs to probes, 8 to compounding diagnosis, 16 to controlled self-play**. Time-box debugging to four hours, then redirect unused capacity to probes.

### 1. Highest-value next run: expanded-bank, leakage-resistant probe replication — 48 GPUs

**One experiment:** establish that probe selection generalizes to unseen kernel families and beats non-probe selection on the **same candidate pools**.

- **Bank:** target ≥300 independent held-out task families, plus separate probe-training and validation banks. Split by operation/template family **before generating variants**; more variants of the same few tasks will not solve your problem. Profile throughput first and prioritize independent tasks over more models.
- **Models:** Qwen-0.5B, Qwen-3B-Instruct, Qwen-7B-Instruct. These cover the dramatic result, high-headroom/weak-selection case, and stronger baseline.
- **Protocol:** cache K=32 candidates per test task. Select layer, regularization, and feature normalization using training/validation only. Run three preregistered probe-training draws against that shared test pool; do not choose the winning draw.
- **Comparators:** uniform random selection, length-normalized likelihood, and a compile/static-validity filter plus random selection. Use identical candidates; report generation, scoring, and filtering costs separately.
- **Primary endpoint:** paired test-task correctness lift, with task-family-clustered confidence intervals. Secondary: K={4,8,16,32}, within-task ranking quality, and oracle headroom recovered, `(probe−random)/(oracle−random)` where defined.

**Crucial correction:** pooled correctness AUC can be high because the probe distinguishes easy tasks from hard ones, while failing to rank candidates *within* a task. Your AUC=1.0 / zero-lift cases make this diagnostic essential. The “AUC≈0.9 threshold” is currently exploratory, not established.

**Do not spend today on steering or cross-model transfer.** Those introduce new failure modes before the central result is secure. Decode-time integration is the follow-up once selection generalization is demonstrated.

### 2. Fix compounding—but with a hard stop — 8 GPUs, four-hour diagnosis budget

**Worth attempting:** a working lineage-versus-reset experiment materially strengthens an RSI claim. **Do not assume the zero is an evaluation bug yet.** Exactly zero after three steps strongly suggests a pipeline problem, but malformed SFT can also destroy generation quickly.

Run one tiny deterministic task set through this ladder:

1. Frozen base.
2. Newly attached, zero-update LoRA.
3. Forward/backward with **no optimizer step**.
4. One optimizer step.
5. Same updated model with adapters disabled.
6. Save/reload, then repeat.

At every stage record **raw output, prompt boundary, stopping reason, parse success, compile success, correctness, and known-good continuation log-probability**. Evaluate a known-good kernel directly through the harness too.

**Top hypotheses, in order:**
- Adapter path changes tokenizer/chat template, EOS/PAD handling, output slicing, or generation configuration.
- Post-training generation inherits training configuration or an incorrect model wrapper.
- Label masking/shift is wrong, PAD/EOS dominates loss, or effective learning rate/update magnitude is excessive.
- Adapter loading/merging or checkpoint association is wrong.

Diff `lab_compounding` against working `lab_weight_rsi` using the **same checkpoint, prompt, and seed**.

**Go/no-go:** after parity is restored, run a small replicated lineage/reset pilot with matched samples, SFT tokens, steps, and evaluation budget. If unresolved, drop the compounding claim—not the entire paper. These five runs are presently **uninterpretable**, not five scientific nulls.

### 3. T5: repair author viability before extending trajectories — 16 GPUs

**Most promising change: constrained, executable task mutation with validity and difficulty gates.** Seven rounds with zero valid tasks means you have not meaningfully tested co-evolution.

- Seed authors with executable task schemas and known-valid examples; initially allow parameterized mutations rather than unrestricted invention.
- Require an independent reference/checker, validity checks, deduplication, and a measurable learnability band on a **fixed calibration panel**. Reject trivial and impossible tasks.
- Give the live author structured learner failure feedback and reward **valid, novel, learnable challenges**, not merely learner failure.
- Apply the **same gates, repair allowance, proposal budget, and fallback policy to F and L**. Only the intended author-evolution mechanism should differ.
- Evaluate learner improvement on a fixed external holdout unavailable to author optimization. Report author acceptance rate, accepted-task novelty, and fallback fraction alongside L−F.

**Prefer several paired-seed trajectories over one long trajectory.** Audit the launched run immediately; if valid-task acceptance is still zero, stop it. First demonstrate author viability, then estimate L−F. This design makes a positive effect plausible; it does not guarantee one.

### 4. Headline: selection gains plus a diagnostic benchmark—not demonstrated recursive takeoff

**Strongest current honest formulation:**

> **KernelAscent separates bottlenecks in kernel-code self-improvement and shows that internal correctness probes can unlock substantial candidate-selection gains in some small-model settings, while sustained recursive gains remain unestablished.**

For an award-target paper, make the contribution **a benchmark that distinguishes generation, selection, update, and curriculum failures**, with one rigorously validated positive channel.

Before submission, fix two credibility hazards:

- The table contains **19 configurations/runs but 13 unique checkpoint names**, not 19 distinct models as shown.
- “Sub-2B almost never cross” contradicts your reported **54% crossing rate**. Say crossing is less frequent than at ≥2B, and provide counts/uncertainty. Likewise, drift/retention associations are not yet causal gates.

**Bottom line:** spend today buying independent held-out evidence, not another impressive maximum lift. If compounding cannot be validated, reframe the title and abstract around the benchmark and correctness-guided selection.
## Fable-5.1

FABLE OK 5249 chars
# Blunt read first

- R1 "19 models" is ~10 distinct models with repeat draws, each scored on ~10–12 tasks (rand=0.086, oracle=0.2 ⇒ ~10 items). A +0.42 lift is ~4 problems. Reviewers will find this in 30 seconds. The AUC≥0.9-gates-lift finding is partly circular: on tiny sets, AUC and lift are both noise-correlated unless AUC is held-out (leave-task-out). Also missing: any non-trivial baseline (logprob rerank, compile-only filter, self-judge, execution-agreement). Probe-vs-random is not a result yet; probe-vs-logprob-rerank is.
- R2 "exactly 0 after 3 steps" is a harness bug, not a science result. Cheap to diagnose.
- R3: L−F=0 with zero valid tasks is undefined, not a null.

# Ranked actions (72×A100, ~1 day)

**1. Bank scale-up + probe rigor (32 GPUs, run first).**
Build 300–600 verified tasks: KernelBench L1/L2 + procedural variants (shape/dtype/fusion perturbations of ops with a PyTorch reference; verification is automatic). Sample K=64 for ~8 models (0.5B→14B, one per family), cache activations at all layers. Then:
- Leave-task-out CV for probe fit and layer selection; report AUC and lift with bootstrap CIs. No more "best layer on the eval set."
- Baselines at identical K: mean-logprob rerank, length, compile/run filter, self-judge prompt, execution-agreement (CodeT-style). Headline metric = fraction of rand→oracle gap closed, probe vs best baseline.
- Probe transfer across task families (fit on L1, apply to L2/procedural) — this is what makes it "a latent correctness signal" rather than a bank-specific fit.
This single run turns R1 from anecdote into oral-grade.

**2. Wall-breaking via exec-budget-matched probe filtering (16 GPUs).**
This is the experiment that connects R1 to the mechanism. For sub-2B models: fixed execution budget of 16 verifications/task. Arm A: sample 16, verify all. Arm B: sample 128, probe-filter to top 16, verify. Measure C_train (verified-correct per task), then one LoRA-SFT round → drift and held-out gain. If Arm B lifts C_train from ~0 to >0 and produces nonzero drift, you have "probe breaks the correctness wall" — a mechanistic claim with an intervention, and it's cheap (probe inference ≪ kernel execution). Skip activation steering for now; it's a different paper and won't be airtight in a day.

**3. Fix the compounding harness (2 GPUs, 2–3 hours). Yes, fix it.**
Definitive diagnostic: attach the adapter with **0 training steps** (LoRA B zero-init ⇒ mathematically identical to base) and run the lab_compounding eval. If it returns 0, the bug is the eval path, not training. Most likely causes, in order: (a) SFT sets `tokenizer.padding_side="right"` and batched generate then produces garbage (needs left padding); (b) `pad_token = eos_token` during SFT → generation terminates immediately → empty kernels; (c) dtype mismatch (bf16 base + fp32 adapter or vice versa → NaN logits); (d) PeftModel.generate called with positional args / `use_cache=False` left from gradient checkpointing; (e) prompt-format divergence between SFT data and eval template. Diff the eval code against lab_weight_rsi (which works) — the answer is in that diff. Then `merge_and_unload()` and eval as a plain HF model to sidestep the wrapper entirely. Don't drop compounding: lineage-vs-reset is the only experiment that tests the title claim, and an honest clean null is still publishable. Run 3 seeds × (lineage, matched reset) × 2 mid-scale models once fixed (8 GPUs).

**4. T5 redesign (16 GPUs, alongside the running 3B run).**
Curriculum collapse happens because the author is trained on its own outputs with no validity floor. Changes:
- Author proposes *mutations* of existing valid tasks (op swap, shape/dtype change, fusion) via a constrained DSL, not from-scratch prose. Validity rate goes from ~0 to high by construction.
- Validity gate (reference runs, non-trivial output, not duplicate) + a replay buffer of valid tasks so the curriculum can never hit zero.
- Author reward = learning progress: solver success in a target band (e.g., 20–80%), PLR/Absolute-Zero style, not validity alone.
- Gate author updates on a minimum valid-task rate; F arm = frozen author sampling from the same initial DSL distribution so L−F isolates co-evolution.
Report author validity rate and difficulty-band occupancy per round, not just L−F.

**5. Leave ~6 GPUs slack for reruns.**

# Honest headline this data can support (conditional on 1–2 landing)

"Kernel-synthesis LMs encode a linearly decodable correctness signal; used as a decode-time reranker it closes X% of the random→oracle gap (CI), beating logprob and self-judge baselines, with the largest gains on sub-2B models. A 50-model sweep localizes the failure of weight-space RSI below ~2B to a *correctness wall* — zero verified training signal — and probe-filtered sampling at fixed execution budget is sufficient to cross it. Among wall-crossers, compounding is gated by sustained drift plus retention, peaking at 2–8B. Unconstrained self-play does not yield author co-evolution (L−F≈0); curriculum collapse, not solver capacity, is the failure mode."

If 2 fails, drop the "sufficient to cross it" clause; the rest still stands. Don't claim uniform RSI — you're right that the mechanism + one intervention + honest negatives is the paper.

---
# Panel round 2 — award-benchmark next steps (2026-09-15)

## GPT-6 Astra

## Bottom line

**There is a credible benchmark paper here, but the evidence you describe is not yet award-grade.** The strongest empirical claims have either weakened or remain untested. Infrastructure, scale plots, and a polished website cannot close that gap.

The promising contribution is now different:

> **A benchmark that separates genuine, transferable self-improvement from better sampling, selection, curriculum exploitation, and evaluation artifacts—under explicit resource budgets.**

That could become exceptional **with either a convincing positive result or an unusually decisive negative result**. Right now, neither is established:

- Within-task AUC ≈0.67 supports a modest ranking signal, not a wall-breaking mechanism.
- +0.01 verified-correct yield is **no demonstrated harvesting advantage**, not yet proof of equivalence without uncertainty estimates.
- The SFT failure invalidates the compounding experiment; it is neither evidence for nor against compounding.
- An author generating zero valid tasks means the current self-play implementation cannot test the curriculum hypothesis.
- Scale associations are useful supporting observations, not an explanatory mechanism.

**My recommendation: stop treating the probe as the centerpiece. Make persistent, transferable improvement the centerpiece—and make the benchmark capable of falsifying it.**

## The SINGLE most important thing to nail

**A trustworthy longitudinal comparison showing whether inherited learning beats strong non-inheriting alternatives on sealed, out-of-family kernel tasks, at matched total cost.**

The question should be:

> After charging for generation, verification, training, and curriculum construction, does carrying forward learned state improve performance on genuinely unseen kernel families more efficiently than spending the same budget on search or fresh optimization?

This is stronger than “lineage beats reset.” A weak reset baseline can lose simply because you repeatedly discard useful optimization. **Beating reset alone does not establish a distinctive self-improvement advantage.**

Also separate two claims:

1. **Persistent self-improvement:** inherited updates improve future performance.
2. **Recursive improvement:** those updates improve the system’s ability to produce useful future training or curriculum, creating an additional feedback benefit.

The first does not automatically establish the second. Unless you demonstrate that second link, I would remove “RSI” from the primary empirical claim.

---

## Ranked next steps by expected impact per GPU-day

These are dependency-aware priorities, not precise GPU-day forecasts; model size, rollout length, and verifier cost are missing.

| Rank | Experiment/artifact | Why it matters | Compute posture |
|---|---|---|---|
| **1** | Certified train→generate→evaluate integrity suite | Every longitudinal conclusion depends on it | Tiny relative to current runs |
| **2** | Sealed transfer suite + resource-accounting protocol + strong search baselines | Makes positive **and negative** results interpretable | Mostly engineering; bounded calibration |
| **3** | Controlled longitudinal tournament | Produces the central scientific result | Main GPU allocation |
| **4** | Fixed-versus-adaptive mutation curriculum experiment | Tests whether feedback itself adds value | Conditional on 1–3 |
| **5** | Probe audit and reproducibility release | Converts existing work into a useful secondary contribution | Mostly existing data; targeted additions |

### 1. Certify the learning loop before running more learning experiments

**Pause large-scale SFT/self-play runs that depend on the broken path.** Use one GPU or a small diagnostic allocation until the following pass.

Build a minimal, deterministic reproducer with one checkpoint and a handful of known-valid examples:

- Base model versus attached zero-step adapter.
- One optimizer step, evaluated **before saving**, then after save/reload.
- Same stepped model with adapters disabled.
- Teacher-forced loss/logits versus free generation.
- Finite loss, gradients, parameters, optimizer state, and sensible update-to-weight norms.
- A tiny overfit test that improves both token likelihood and actual generated correctness.

Use deterministic decoding and fixed inputs first. Inspect termination tokens, tokenization, generation configuration, and adapter-loading behavior alongside mixed precision. **Do not assume fp32 LoRA is the fix before localizing the failure.**

Your observation strongly implicates the training/update/generation path, but it does not yet uniquely identify a generation-only bug: a corrupted update can also destroy generation.

**Artifact:** an executable integrity suite, environment lockfile, and short failure postmortem.

**Exit criterion:** the exact production path passes before and after checkpoint/resume. A dtype workaround that only succeeds in an isolated notebook does not count.

This is the highest-return work available. More runs through an invalid loop create no usable longitudinal evidence.

### 2. Build a sealed transfer test and explicit cost contract

Your small task bank is already limiting probe stability. **Spend on independent task coverage before expanding the checkpoint sweep.**

Construct splits at the level of **semantic/kernel families**, not merely random examples:

- **In-family generalization:** unseen shapes, layouts, dtypes, boundary cases.
- **Out-of-family transfer:** held-out operator/composition families absent from training and curriculum construction.
- **Robustness:** non-contiguous inputs, numerical edge cases, race-sensitive repetitions, and applicable hardware/resource boundaries.

Make “held-out family” auditable; a renamed template or nearby shape is not a new family. Synthetic mutations increase coverage but do not automatically increase the number of independent families.

For each task, require:

- An independent trusted reference and appropriate numerical tolerances.
- Hidden correctness cases unavailable to the learner.
- Correctness before performance credit.
- Stable timing with warmup, repeated measurements, and controlled GPU contention.
- A fixed policy for timeouts, compilation failures, and invalid outputs.

**Cost accounting should have two views:**

1. **End-to-end GPU-time/compute:** generation, probe feature extraction, verification, training, authoring.
2. **Verifier-constrained performance:** your existing matched-\(E\) setting.

The second is diagnostically useful; it is not a substitute for the first. Report wall-clock latency separately when parallelism changes it.

**Primary endpoint:** predeclared held-out utility versus cumulative cost, with a fixed-budget endpoint. Keep correctness rate and speedup visible separately; any aggregate should use a declared failure penalty and capped speedup credit.

Also distinguish:

- **Parameter-level gain:** fixed inference procedure across checkpoints.
- **Whole-system gain:** each method’s full inference procedure, fully charged.

Otherwise an improved sampler can masquerade as an improved model.

**Artifact:** benchmark v1, sealed evaluation service or equivalent access control, baseline scripts, cost ledger, and preregistered primary comparisons. Public training/dev data can coexist with sealed tests.

### 3. Run the longitudinal tournament—not another broad model sweep

Start with **one model that has measurable correctness headroom and a working training path**, selected on development data. Add a second scale only after the protocol survives a pilot.

Run three essential competitors and one diagnostic:

1. **Inherited learner:** carry model/adapters forward.
2. **Strong search-only baseline:** spend the corresponding total budget on sampling, execution-guided repair, and selecting the fastest verified-correct kernel.
3. **Fresh-start replay baseline:** start from the original checkpoint and train on the accumulated data available at that evaluation budget; charge all replay/retraining.
4. **Current-round reset diagnostic:** same round’s data and update budget, but no inherited parameters.

The fourth isolates inheritance locally. The third asks whether inheritance beats a credible alternative, rather than a deliberately forgetful opponent.

**First use a shared, exogenous training-data stream.** This isolates learner persistence from changes in curriculum. Adaptive, endogenous data comes next.

Evaluate at predeclared budget checkpoints, without exposing sealed-test results to training decisions. Measure:

- Held-out correctness and performance.
- Learning-curve utility and final fixed-budget utility.
- Retention on earlier skills.
- Training performance, to separate failure to learn from failure to transfer.

Use paired task evaluations and independently seeded trajectories. **Tasks and trajectories—not thousands of candidate kernels—must drive uncertainty estimates.** Choose replication after a pilot estimates variance; define a practically meaningful effect size in advance.

**Decision rules:**

- If inherited learning only beats current-round reset, claim a persistence benefit, not an exceptional RSI result.
- If it beats search and fresh-start replay on out-of-family transfer at equal cost, you have a strong central result.
- If all methods learn in-distribution but inherited learning fails to transfer, that can support a valuable negative paper.
- If nothing learns even on training tasks, you still have a pipeline or optimization problem—not a scientific null.

### 4. Only then test adaptive curriculum with a valid mutation DSL

Do not spend more GPUs asking the current free-form author to invent tasks when valid yield is zero.

Build mutations over trusted task templates, for example:

- Shapes, strides, layouts, and supported dtypes.
- Legal operator compositions/fusions.
- Reductions, broadcasting, and boundary conditions.
- Resource constraints that preserve a well-defined reference.

Every proposed task must have a generated reference and pass feasibility checks. For the first scientific test, require known solvability—through a trusted implementation or stronger reference solver—and a difficulty range that is not entirely trivial or impossible.

Compare:

- **Fixed random DSL curriculum.**
- **Fixed difficulty-stratified DSL curriculum.**
- **Adaptive DSL curriculum using learner feedback.**

Match proposal and training budgets; charge rejected proposals and difficulty estimation. Evaluate downstream gains only on the sealed original transfer suite.

**The pivotal result is adaptive curriculum beating a strong fixed curriculum**, not merely producing valid tasks.

If persistent learning succeeds in step 3, a compact **inheritance × curriculum-adaptation factorial** can test whether the two interact. That is a much better route to a recursive-improvement claim than another descriptive drift plot.

### 5. Finish the probe story as a rigorous secondary result

Do not bury the negative result, but do not spend the main budget trying to rescue it.

Use existing candidates where possible to report:

- Pooled versus within-task AUC.
- Task-held-out probe fitting and evaluation.
- Oracle headroom versus realized selection gain.
- Task-clustered uncertainty on matched-\(E\) yield.
- End-to-end cost, including oversampling.
- Comparisons against random ranking and inexpensive confidence/log-probability baselines.
- Both correct-candidate yield and best verified kernel utility.

Predefine a practical equivalence margin if you want to conclude “no meaningful advantage.” Your +0.01 point estimate alone cannot establish that.

The defensible insight is:

> **Decodable correctness information does not automatically translate into useful search efficiency.**

That is scientifically interesting and directly relevant beyond kernels. It is not yet a general theorem about probes or correctness walls.

---

## What I would stop doing now

- **More checkpoint breadth** before task diversity and measurement validity improve.
- More probe sweeps without a specific hypothesis about the failed cost-matched comparison.
- Free-form author runs with zero valid-task yield.
- Additional website polish as a substitute for central evidence.
- Causal language for the scale associations.
- “Roofline saturation” unless you have actual roofline measurements; empirical benchmark saturation is different.

## Framing and publication decision

**Suggested framing now:**

> **KernelAscent: Measuring Persistent Self-Improvement in GPU Kernel Optimization**

If the central result is negative:

> **KernelAscent: Separating Search, Selection, and Learning in Kernel Self-Optimization**

The award-level opportunity is not “we ran self-play on 72 A100s.” It is:

> **We built a measurement standard that makes purported self-improvement hard to fake, and used it to establish a consequential result.**

**Immediate action:** redirect the cluster away from dependent broken experiments, certify the learning loop, freeze the evaluation contract, and run the small longitudinal tournament. Its outcome—not another probe sweep—should decide the paper’s thesis.


## Fable-5.1

## Brutal verdict first

**As it stands: no, this is not an award-grade paper.** It is currently (a) two clean negative results about probes, (b) a collapsed self-play loop, (c) an untested title claim blocked by a harness bug, and (d) a correlational mechanism section. Excellent infra and a website do not substitute for a headline result. Reviewers will ask "what did the benchmark *measure* that we didn't know?" and right now the answer is "cheap verifier proxies don't beat verification at matched budget," which is a solid workshop finding, not an oral.

**But there is an award-grade paper within reach**, and the negatives actually help it — *if you change the framing.*

## The framing change

Drop "probe breaks the wall." Drop "self-play benchmark" as the headline. The paper is:

> **KernelAscent: Does self-improvement compound? A roofline-grounded, compute-matched testbed for RSI dynamics across scale.**

Your unique asset is not the probe. It's that GPU kernels are the one domain where (1) reward is objectively verifiable, (2) the *ceiling is physically known* (roofline), so headroom is an absolute number, and (3) you have 72 A100s to run lineage-vs-reset across 13 checkpoints with real controls. Nobody else has published a controlled compounding measurement with a known ceiling. The deliverable is a **phase diagram of RSI**: for each (scale, compute budget), does lineage beat reset, by how much, and where does it stall relative to roofline. The probe negatives become a supporting chapter ("shortcuts to verification don't work"), which strengthens rather than weakens the story.

**Critical benchmark-design point:** a benchmark that returns zero for every method is not a benchmark. You *must* include a positive control that shows compounding *can* occur in your harness (e.g., lineage with oracle-injected correct kernels, or large verification budget). Without demonstrated dynamic range, a null compounding result is indistinguishable from a broken harness — which is exactly what just happened to you.

## The SINGLE most important thing

**The lineage-vs-reset × scale matrix with matched compute and a positive control.** Everything else is subordinate. Until this runs, you have no paper; once it runs, you have a paper regardless of the sign (a rigorous "RSI does not compound below 14B under verified self-training" is a strong result *if* the positive control works and the compute matching is airtight).

## Prioritized plan (impact per GPU-day)

### P0 — Fix the SFT-kills-generation bug (0 GPU-days, 2–3 eng-days). Gate on this.
One step zeroing *both* train and held generation is almost certainly not a subtle learning problem; it's a mechanical one. Diagnose in this order:
- **lr=0 control.** Run one optimizer step with lr=0. If generation still dies → it's not weights; it's mode/state (model left in `.train()` with LoRA dropout, `use_cache` disabled after training, gradient checkpointing left on, tokenizer/pad changes, adapter not merged/enabled for generate).
- **NaN scan.** After step 1, `torch.isfinite` on every LoRA param and on the loss at step 1. NaN in one LoRA B matrix → all outputs garbage. Add grad clipping and check loss scale.
- **Inspect outputs:** identical repeated token (collapse) vs random garbage (NaN/dtype) vs empty (EOS/pad label bug — check you're masking pad tokens to −100 and not training on the prompt with a broken attention mask).
- **fp32 master weights the standard way:** base bf16 frozen, LoRA params fp32, forward under `torch.autocast(bf16)`. Don't upcast the whole model. If PEFT fights you, full-parameter fine-tune at 1e-6 on a 0.5B as a bypass to unblock the science while you fix LoRA.
- Add a permanent regression test: `generate → 1 SFT step → generate` must stay within ±0.05 correctness. This should have caught it; make it CI.

### P1 — Compounding matrix (~25–35 GPU-days). The headline.
- **Arms:** (A) lineage: iterate generate→verify→SFT on own outputs; (B) reset-iid: at each iteration, train *from base* on the union of all verified data so far (same data, no weight lineage); (C) matched-compute reset: from base, single-shot with the same total sample/token budget as A; (D) **positive control**: lineage where verified kernels from the largest model are injected at a fixed small rate.
- **Scale:** 5 checkpoints spanning 0.5–14B (not 13; depth over breadth), 3 seeds each, ≥6 iterations.
- **Metrics to define and release as the benchmark protocol:** compounding gap Δ = A − B; wall height = plateau correctness; roofline fraction = achieved/attainable speedup; headroom = oracle-K correctness − greedy. Report per iteration with CIs.
- **Pre-register the hypothesis and analysis now** (OSF or a timestamped repo commit) before the fix lands. Cheap, and for a benchmark paper on a contested question it's a real credibility lever.

### P2 — Make the correctness wall causal (~5–8 GPU-days). Highest impact-per-GPU-day.
You have "<2B hits a wall (54% cross)" as an association. Convert it with one intervention: for sub-2B models, at matched training tokens, compare (i) more self-generated verified kernels vs (ii) a small dose of externally verified correct kernels (from 14B or humans). If (ii) breaks the wall and (i) doesn't → the wall is a *coverage/knowledge* gap, not a *search* gap — and that immediately explains why the probe couldn't help (it reranks what's already there). This ties results 1, 2 and 5 into one mechanistic claim. Add a verification-budget sweep (E ∈ {4,16,64}) at 1.5B to show the wall doesn't move with search — that's the causal complement.

### P3 — Procedural task bank via the mutation DSL (~10 GPU-days, mostly eng). Dual-use.
Your probe-fit AUC swinging 0.76–1.0 across draws says the task bank is too small to support *any* per-task statistic. Build the mutation DSL (shape/dtype/layout/fusion/tiling operators over verified seed tasks, each with an auto-computed roofline) and use it to generate 500–2000 tasks. This (a) fixes your statistics, (b) gives the benchmark real breadth, and (c) *is* the author redesign. Then run the learnability gate (keep tasks with solver pass rate in (0.1, 0.9)) and compare **learned author vs random mutation vs fixed curriculum** for the solver's L−F. If random mutation ≈ learned author, report it — that's a finding, not a failure.

**Honest recommendation:** if the learned author doesn't produce L−F>0 within ~2 weeks of DSL work, demote self-play to a section titled "Task proposal collapses without structured mutation" and ship with fixed/random-mutation curricula. The oral does not require a learned proposer; it requires the compounding measurement.

### P4 — Package the probe negatives properly (~2 GPU-days).
Two figures: within-task vs pooled AUC (a methodological warning the field needs), and the matched-budget yield plot. One optional addition: the only place a probe could still help is *decode* cost, not verify cost — early-abandon of low-probe-score generations mid-sequence. Run it once at 3B; if it's also null, you can state "verifier proxies help neither verification nor decode budget," which is a cleaner, more general negative. Don't spend more than this.

### Stop doing
- Website polish, DAG animations, more LaTeX automation. Zero marginal value until P1 exists.
- Adding checkpoints. 13 is breadth without depth; 5 × 3 seeds beats 13 × 1.
- More associational "WHY-RSI probes." Either they become P2-style interventions or they're an appendix.

## Decision tree after P1

- **Δ > 0 and grows with scale, positive control works:** oral-grade. Title claim confirmed with controls nobody else has.
- **Δ ≈ 0 everywhere but positive control works:** still strong — "verified self-training does not compound at 0.5–14B; the wall is set by base coverage" with P2 explaining why. This is a *Cannot Self-Correct Yet*-class result if the controls are airtight.
- **Positive control also fails:** the harness is still broken. Do not publish.

You have the compute and the domain to answer a question the field is actively arguing about with a level of control no one else has shown. The probe detour was worth it because it killed a tempting shortcut cleanly. Now put all 72 GPUs on P1 and P2.

---
# Panel round 3 — post-results (2026-09-16)

## GPT-6 Astra

**My blunt read:** the current results support a paper about **verification-guided search and the limits of converting verified experience into persistent improvement**—not yet a paper demonstrating compounding self-improvement. The moving “wall” is useful evidence. The lineage collapse is presently a validity blocker, not an algorithmic finding.

I would not keep all 72 A100s on self-play until you verify that its LIVE arm actually generates with the intended, numerically healthy adapter.

## (a) Lineage collapse: first suspect the adapter lifecycle, not intrinsic LoRA instability

Repeated fine-tuning of a LoRA adapter is ordinary continued training. It can overfit, forget, or become unstable, but **an exact-zero cliff confined to the continued-adapter arm is not enough to establish a known “accumulated LoRA self-SFT fails” phenomenon.** Nor is fp32 master weights a universal fix.

### Most likely root cause

My leading hypothesis is **incorrect adapter state handling across rounds**, followed by **cumulative update magnitude or pathological training targets**.

The highest-priority possibilities are:

| Suspect | Why it fits | Decisive check / fix |
|---|---|---|
| **Adapter merged into the base and then applied again** | Only lineage carries this state across rounds; the effective update can be counted twice | Keep the base immutable and unmerged throughout training. Save/load only the adapter onto a pristine base. Audit every merge/unmerge call. |
| **Training and generation use different adapter states** | A persistent inference worker, stale cache, wrong adapter ID, or incorrect hot reload can selectively break LIVE/lineage | Evaluate the same checkpoint in a fresh, plain inference process. Compare fixed-prefix logits with the training model in eval mode. |
| **Reload/cast changes trainable parameters or optimizer bindings** | Reinitializing the optimizer does not help if later operations replace the parameters it owns | After all loading/casting/wrapping, construct the optimizer. Confirm its parameter objects are the active trainable adapter parameters. Assert dtypes after every round transition. |
| **Cumulative overtraining or malformed self-SFT targets** | A reset arm takes one update trajectory; lineage accumulates all prior updates, potentially toward EOS, boilerplate, or a narrowed output distribution | Replay identical, known-good data for two stages; inspect target masking, EOS/padding handling, losses, gradient norms, and actual adapter-induced weight changes. |

**LoRA alpha does not normally “drift.”** It is configuration. An accidental configuration change is possible, but checking alpha alone is insufficient: factor norms and their product can grow even with constant alpha.

Also, distinguish these immediately:

- zero nonempty outputs;
- zero parseable kernels;
- zero correct kernels;
- zero on whatever \(C\) measures.

Those are different failures. If generation itself remains fluent but kernels fail verification, the diagnosis changes substantially.

### The concrete fix I would try first

**Maintain one immutable base plus one unmerged adapter. At every round boundary, save the adapter and reload it onto a pristine base in a fresh process; then construct a fresh optimizer after all model modifications. Use a fresh evaluator as the reference.**

This is a diagnostic normalization, not a claim that disk reload cures numerical instability. It removes a large class of state-management faults. **Do not re-merge every round as a generic fix**—that is an excellent way to introduce double application.

Then run this tiny checkpoint ladder before another matrix:

1. Fresh adapter, one SFT stage → checkpoint A.
2. Evaluate A immediately.
3. Save/reload A, **no training**, evaluate again.
4. Pass A through the full next-round setup with **no optimizer step**, evaluate again.
5. Take **one optimizer step** on known-good data, evaluate.
6. Complete the second SFT stage, evaluating at several intermediate points.

For each point, retain raw generations, fixed-prefix logits, adapter configuration, parameter/gradient norms, and nonfinite checks.

This localizes the cliff to:

- serialization/inference;
- round-transition code;
- the update implementation;
- or cumulative learning dynamics.

**One especially important consistency check:** if “exactly zero every round” includes the *first* SFT round, lineage and reset should be equivalent there, given identical initialization, data, and seeds. A round-one discrepancy would argue strongly against repeated SFT as the explanation.

If the lifecycle checks pass and performance degrades smoothly with additional steps, then test lower cumulative learning-rate exposure, gradient clipping, and replay of a small stable reference corpus. Do not change all of these simultaneously.

### Is the collapse reportable?

Currently: **reportable as a limitation that prevents measurement**, not as evidence against compounding.

It becomes an empirical finding about cumulative self-SFT only after it:

- survives an independent, minimal implementation;
- occurs with validated checkpoints and labels;
- replicates across seeds;
- is separated from simple training-dose effects and ordinary overfitting.

Until then, call it **“lineage-arm failure; compounding not identifiable.”** And audit the LIVE arm separately rather than assuming it is either broken or healthy.

---

## (b) A moving wall improves the explanation—and weakens the original ceiling claim

**It weakens “sub-2B models lack the requisite support/coverage.” It strengthens “apparent capability depends strongly on inference-time search and verification.”**

For a task solved by a valid sample at budget 16, the model demonstrably has some probability mass on a solution. Its failure at budget 4 was not an absolute impossibility.

But this does **not** establish that all remaining failures are search failures. Nor does it distinguish low-probability correct solutions from verifier failures or other bottlenecks on unsolved tasks.

### Be careful with “4× search gives 6× solve rate”

That is a descriptive ratio, not yet a scaling law.

If 8.3% and 50.0% mean **1/12 versus 6/12**, the denominator is very small. Report counts, paired outcomes, and uncertainty.

For intuition, under independent identically distributed draws, a task with success probability \(p\) has:

\[
P(\text{at least one success by }k)=1-(1-p)^k.
\]

A homogeneous model calibrated to 8.3% at four samples predicts about 29% at sixteen, not 50%. Task heterogeneity does not generally justify extrapolating that ratio upward. Your observation may simply be small-sample variability, or budgets may differ in more than sample count.

**Best measurement:** generate one candidate stream per task/seed and evaluate nested prefixes at budgets 1, 2, 4, 8, and 16. Repeat streams where affordable. Keep decoding and per-candidate verifier behavior fixed, and count actual generation and verification costs.

### The probe result is complementary, not contradictory

Search and reranking act on different parts of the pipeline:

1. **Candidate availability:** did the pool contain a valid solution?
2. **Selection efficiency:** did the procedure identify or allocate verification to it?

More samples can improve availability. A probe cannot select a correct solution absent from its pool.

But the strongest probe comparison is not merely fixed-pool ranking. It is:

> At the same total cost, does probe-guided generation/verification harvest more verified solutions than a simple baseline?

You already report no such advantage. The appropriate conclusion is:

> “The probe exhibits modest within-task discrimination, but this does not translate into useful matched-budget harvesting.”

AUC 0.67 is not itself operational value. Top-of-ranking precision, calibration, overhead, and verifier cost determine that value.

**The coherent joint result:** *additional search finds more solutions; the tested probe does not make that search economically better.* That is informative, but narrower than a general negative result about learned verification allocation.

---

## (c) Tightest story and most important remaining run

### The strongest defensible story today

**“Search-accessible solutions are not yet evidence of persistent self-improvement.”**

The paper would distinguish three claims:

| Claim | Current evidence |
|---|---|
| Additional verified search accesses more solutions | Promising preliminary positive |
| A learned probe improves harvesting at equal cost | Negative for the tested probe |
| Verified solutions produce persistent, compounding weight-level gains | Not measurable because lineage fails |

That is intellectually clean. **It is not currently award-grade evidence.** The positive is expected in broad outline, the probe negative is limited, and the central compounding result is missing.

I would also avoid leading with “roofline-grounded” unless roofline analysis materially explains the results. Compute accounting is necessary; it is not automatically the scientific contribution.

### Single most important scientific run

**After the lifecycle gate passes: complete the compute-accounted, self-versus-external verified-teacher transfer experiment, with frozen-model search as the anchor.**

The decisive question is:

> Can verified successes be converted into reliable held-out performance more efficiently than continuing to search with unchanged weights—and does the source of those successes matter?

Essential comparisons:

- **Frozen base + search/verification.**
- **Self-generated verified data → SFT.**
- **External-teacher verified data → SFT.**

Use the same validated training pipeline and final evaluation procedure. Measure both:

- held-out performance at a common inference budget;
- end-to-end cost, including data acquisition, verification, and training.

Separate two estimands rather than pretending one match answers both:

- **Equal verified-data/training exposure:** isolates differences in the usefulness of the datasets.
- **Equal total acquisition-and-training cost:** measures practical efficiency.

For an amortization claim, state the number of downstream evaluations over which training cost is spread. Otherwise frozen search and trained models are not economically comparable.

External-teacher success would establish that the training pipeline and tasks permit learning. It is **not automatically a positive control for multi-round compounding**; that requires a longitudinal control.

This run offers interpretable outcomes:

- **Both learn:** verified experience can be consolidated; compounding remains to be tested.
- **External learns, self does not:** a source-dependent learning gap under the tested collection protocol—not a universal causal wall.
- **Neither learns:** training/data suitability remains unresolved.
- **Self learns but loses to frozen search on total cost:** real weight-level improvement without demonstrated economic advantage.

### Ranked by impact per GPU-day

| Rank | Next step | Why |
|---|---|---|
| **1** | Minimal lineage checkpoint ladder plus LIVE-arm audit | Cheap relative to the campaign; determines whether expensive results are interpretable. |
| **2** | Finish validated self/external/frozen-search comparison | Highest scientific discrimination among the remaining hypotheses. |
| **3** | Reuse or extend nested search-budget streams, with replication | Makes the moving-wall claim credible; often inexpensive if candidates are retained. |
| **4** | A short longitudinal test with an explicit learning-positive control | Only after rank 1 passes; test persistence and reset-versus-lineage before scaling rounds. |
| **5** | Expand self-play or revisit probe modeling | Low priority without a healthy weight channel or demonstrated allocation benefit. |

**Operationally, freeze the broad campaign, not necessarily every job.** Preserve useful frozen-baseline and candidate-generation runs; pause lineage-dependent runs whose validity is unknown.

The potential award-level result is not “our wall moved” or “LoRA collapsed.” It is a convincing separation of **finding solutions, consolidating them, and compounding that consolidation under honest resource accounting**. Right now you have preliminary evidence for the first, a useful negative on one selection mechanism, and no valid measurement of the third. The next work should close that gap—not make the existing matrix larger.


## Fable-5.1

Short version up front: the exact-zero in the lineage arm is almost certainly a plumbing bug, not a learning-dynamics finding; the verify-budget result is good news if you reframe around it; and nothing you have yet clears an award bar. Details:

---

## (a) Lineage collapse: root cause and fix

**"EXACTLY 0.000 every round" is the tell.** Genuine instability (model collapse, entropy collapse, LoRA drift) degrades noisily — you'd see 0.02, then 0.008, then 0.0. A flat exact zero across rounds and 72 GPUs' worth of samples means one of two deterministic things, and you need to check which:

**Branch 1 — the metric is a delta and the weights aren't changing.** If C is defined as improvement over the previous round, exactly 0.000 means the adapter is bit-identical (or ulp-identical) before and after SFT. Known causes, in order of likelihood:
- `PeftModel.from_pretrained(...)` defaults to `is_trainable=False`. Loaded adapter params have `requires_grad=False`; loss is computed, the optimizer is built over an empty or wrong param group, nothing moves. Extremely common when the round-0 path builds the adapter with `get_peft_model` (trainable) and rounds ≥1 reload from disk. This asymmetry matches "reset fine, lineage broken" perfectly.
- Your fp32-master-weight recipe: if the fp32 copy is created once in round 0 and the round-r reload rebuilds bf16 params without re-pointing the master copy, you train a stale fp32 tensor and downcast it over the live weights → no change, or revert.
- If you merge-and-reinit each round: merging a LoRA delta into bf16 base weights zeroes any per-element update smaller than the bf16 ulp (~0.4% relative). Early-round LoRA deltas are mostly below that. Net effect: merge is a no-op.

**Branch 2 — the metric is raw solve rate and generation is broken.** Then dump 20 raw completions per round. You'll see one of: empty strings (EOS/pad misconfigured — training data lacked EOS or used a different chat template than inference), infinite repetition (adapter applied twice: `merge_and_unload` then re-wrapping with a new adapter that also gets the old scaling, or `alpha/r` applied at both save and load), or valid-looking code that all fails (verifier path/timeout bug in the lineage branch only).

**Diagnostic ladder (zero GPU-days, ~2 hours):**
1. Checksum every adapter tensor before and after round-2 SFT in the lineage arm. Count `requires_grad=True` params. Log grad norm per step.
2. Dump raw generations for the lineage adapter at each round.
3. Decisive control: run the lineage arm's round-2 *procedure* (load-from-disk → SFT → generate) on the *same data* the reset arm used at round 1. It must statistically match reset round 1. If it doesn't, the load/save path is the bug, full stop.

**Concrete fix:** `is_trainable=True` on load; keep the adapter on disk in fp32 (safetensors), never round-trip through bf16 between rounds; do not merge — keep one adapter and continue training it, or if you must merge, merge into an fp32 copy of the base and re-quantize. Re-verify with step 3 above.

**Is "accumulated LoRA self-SFT is unstable" reportable?** Not as you have it. Once the bug is gone, if lineage degrades *gracefully* relative to reset, that's a minor confirmation of what ReST-EM (Singh et al. 2023) already reported: re-initializing from base each iteration and training on the accumulated verified data beats continuing to fine-tune. It's a paragraph, not a result.

**The important reframe:** your reset arm *is already a lineage* — through the data channel. Fresh adapter each round trained on verified kernels generated by the previous round's model is exactly the ReST-EM / STaR recipe, and it's the more defensible definition of "verified self-improvement" anyway (weights are an implementation detail; the recursion is data → model → data). Compounding is therefore *not* parked. The question becomes: does C_reset grow, stay flat, or shrink across rounds? If it's 0.1–0.2 every round with no trend, you may already have your answer (one-step gain, no compounding) and just haven't looked at it that way.

**Self-play live arm:** treat it as compromised until the checksum test passes on it too. If it shares the reload path, you are currently burning most of 72 A100s on an arm whose weights may not be updating. Pause or convert it to reset-style today.

---

## (b) Verify-budget result: strengthens, if you own it

**First, the n.** 8.3% and 50.0% look like 1/12 and 6/12. Twelve wall tasks is not a sweep; it's an anecdote. Enlarge the wall set (or at minimum bootstrap CIs) and stop doing "best-of-k reruns" — sample n≥64 per task once and compute unbiased pass@k (Chen et al. estimator) for all k simultaneously. Cheaper and gives you the whole curve.

**Interpretation.** 8%→50% from k=4→16 implies per-sample success p≈3–5% on those tasks — they are not zero-coverage, they are low-p. That means "the sub-2B wall" was never a wall; it's a point on a pass@k curve at whichever k you happened to choose. This *strengthens* the paper in three ways:

1. It kills the naive story ("small models can't do these kernels") in favor of a quantitative one ("the effective budget currency is verified samples, and verification throughput is the bottleneck"). That is a claim reviewers can't dismiss.
2. It gives you the right measurement for self-improvement: **does SFT on verified kernels shift the pass@k curve left** (raise p per task), and on *which* tasks — already-solvable ones or wall ones? "Sharpens but doesn't extend" (p rises on solved tasks, stays ~0 on true-zero-coverage tasks) is a crisp, falsifiable, mechanistic result and would be the paper's spine.
3. It recasts the probe correctly. A reranker with AUC 0.67 doesn't add search; its only value is saving *verification* compute. Frame it as: "cheap surrogates don't substitute for verification at this AUC; verified-solves-per-verification-second is the metric, and the probe doesn't move it." That's a clean secondary negative consistent with point 1.

It *weakens* any claim that prior compounding numbers measured at fixed k were "improvement" rather than budget reallocation — so your compute-matching must now include verification cost explicitly, and every round must report the curve, not a point.

For the causal-wall test: at matched verified-example count, teacher solves vs self solves is now a clean test of whether the *content* of verified kernels matters beyond their count. Keep it, but it's secondary to the compounding-vs-search control below.

---

## (c) Tightest story, ranking, and whether it clears the bar

**Does it clear the bar now?** No. You have: a harness bug, an unmeasured headline, a 12-task sweep, and a modest probe negative. A brutal reviewer writes "engineering report" in one line.

**What would.** The claim within reach:

> *Verified self-improvement on kernels produces a real one-step gain equivalent to ~X× sample budget, does not compound beyond round 2 at matched (generation + verification) compute, and works by raising p on already-solvable tasks rather than extending coverage to wall tasks. The apparent sub-2B "wall" is a pass@k artifact.*

That is a publishable, possibly strong-negative paper *if* the controls are airtight. Award-grade only if either compounding actually shows up (unlikely given C_reset flat at 0.1–0.2) or the mechanistic decomposition is unusually clean and generalizes across model sizes.

**Single most important run — the compounding-vs-search control:**
Reset-style lineage (ReST-EM) for ≥4 rounds, fresh adapter each round on accumulated verified data, reporting full pass@k per round on solved and held-out wall sets. Against it, the one control that matters: **the round-0 model given the same total sample + verification budget in a single large-k harvest.** If iterated self-training doesn't beat single-shot big-k at equal compute, there is no compounding, period — and you'll have shown it properly. This is also the positive-control slot: you can inject a fraction of teacher solutions to confirm the pipeline *can* register compounding when it's present.

**Ranking by impact per GPU-day:**

1. **Bug diagnosis + checksum control** (0 GPU-days). Do first; everything downstream depends on it, and it de-risks the self-play arm.
2. **Compounding-vs-search control** above (~the bulk of what's left). This is the paper.
3. **Full pass@k curves on an enlarged wall set** (cheap; n=64 samples once per task). Gives the interpretive frame and fixes the n=12 problem.
4. **Causal-wall self-vs-teacher at matched count** (moderate). Good secondary, only after 2 is running.
5. **Self-play live arm** — pause until (1) passes on it. Currently your largest GPU-waste risk.
6. **Probe** — stop. One paragraph in the negatives section.

If (2) comes back flat, write the negative paper honestly and tightly; a rigorous "no compounding at matched compute, here's why" beats a fuzzy positive every time, and it's the result the field actually needs. If you can't get (1) and (2) clean, don't submit.

---
# Panel round 4 — CI-backed negative, award-bar (2026-09-16)

## GPT-6 Astra

## 1. Award-grade? **Not yet, on the evidence described.**

**Credible, potentially strong paper: yes. Award-grade benchmark/oral: not currently the most defensible expectation.** Fixing the harness makes the results interpretable; it does not, by itself, make the scientific conclusion decisive.

Your strongest contribution is **a controlled way to distinguish cumulative learning from repeated sampling and selection**. That is a real evaluation problem. Your weakest framing is **“RSI does not compound at ≤3B.”** The experiments concern one form of verified self-training, over six rounds, in one domain—not recursive self-improvement broadly.

Three things limit the current claim:

- **A CI spanning zero does not establish no compounding.** Your pooled interval, `[-0.014, +0.094]`, supports “no statistically resolved advantage,” not “no advantage.” Whether it excludes *meaningful* compounding depends on the metric and a justified minimum meaningful effect.
- **What is independent in pooled n=25?** If these include successive rounds from the same lineage, they are correlated. Treating them as independent would overstate precision. With only 2–3 seeds per configuration, uncertainty over training trajectories remains weakly estimated, even with many evaluation tasks.
- **The operating regime matters.** Your own teacher result shows that larger models generate substantially more usable training material. That makes “the tested students are below a useful self-training regime” a live alternative explanation—not an objection you can dismiss.

The finding that **lineage loses to compute-matched best-of-N search** may be more consequential than the null lineage–reset contrast. It asks whether training is a better use of compute than simply searching, without requiring you to prove an exact null.

The remaining findings are useful but do not automatically add up to an award-grade centerpiece. The pass@k correction is scientifically valuable; the probe result is informative; the NaN and hang findings strengthen reproducibility. The 50-model study becomes especially valuable if its mechanisms **predict an independently tested outcome**, rather than explain outcomes retrospectively.

**Venue framing:** NeurIPS Datasets & Benchmarks is plausible if the released task bank, evaluator, accounting protocol, baselines, and reproducibility package are independently valuable. Otherwise, this sounds more naturally like an empirical methods paper. Oral/award selection is too contingent to forecast honestly.

## 2. Single highest-leverage experiment

**Run one adequately replicated, compute-matched 14B lineage–reset–search experiment. Pick one larger scale, not a shallow 7B/14B sweep.**

Why this one: you already have evidence that 14B supplies roughly 10× more correct kernels. It directly tests the strongest alternative explanation for your negative result: **insufficient verified training supply at small scale**.

Make it one decisive experiment:

- **Three arms:** persistent lineage, reset-to-base self-training, and no-training best-of-N search.
- **Matched total compute:** charge generation, verification, training, and evaluation consistently. Let search spend the compute saved by not training.
- **Primary endpoint:** a prespecified held-out-task performance difference at the final common compute budget. Use the roundwise trajectories to distinguish a first-round lift from continued gains.
- **Decision rule:** define a smallest meaningful lineage advantage before running. Estimate uncertainty using independent training runs; do not inflate replication with rounds. Prioritize seeds over extra scale points.

The experiment has value either way:

- **Compounding emerges:** you gain evidence for a scale-dependent transition, with the coverage mechanism as a testable explanation.
- **It remains absent with a sufficiently tight interval:** you substantially strengthen the negative result despite much better verified-data supply.
- **Lineage beats reset but loses to search:** you reveal genuine cumulative learning that is nevertheless compute-inefficient—a particularly clean distinction.

With 72 GPUs, favor **one interpretable answer over broader coverage**. GPU count alone does not establish feasibility; if adequately replicated 14B runs exceed the budget, do the same design at 7B rather than an underpowered 14B demonstration.

Proper pass@k curves are necessary support, but they are unlikely to transform the paper. Another small-model search control probably will not either, given your existing search results.

## 3. Title and abstract

### **KernelAscent: Testing Whether Verified Self-Training Compounds**

**Abstract:** Verified self-training is often proposed as a route to cumulative model improvement, but repeated gains can reflect sampling and selection rather than benefits from preserving a learned lineage. We introduce KernelAscent, a leak-free, compute-matched testbed for GPU-kernel optimization with correctness verification and roofline-grounded performance evaluation. Across six rounds of self-training with 0.5B, 1.5B, and 3B models, lineage-preserving training shows no statistically resolved advantage over reset-based training, and loses to best-of-N search in most rounds. These results do not establish an absence of compounding, but provide no robust evidence for it in the tested regime. Mechanistic studies find that self-verification predicts correctness modestly without improving matched-budget harvesting, and that apparent small-model correctness barriers reflect low sampling probabilities rather than zero task coverage. We additionally document training and grading failures that can invalidate iterative experiments. KernelAscent provides a controlled framework for separating cumulative learning, verified-data supply, and search efficiency in self-training systems.


## Fable-5.1

## (1) Is it award-grade? No — not as stated. Here's exactly why.

**The headline is a bounded null, not a negative.** Your pooled CI is [−0.014, +0.094] around +0.040. That is *consistent with* compounding of ~4 points per round and merely underpowered to prove it. "Does NOT recursively compound" is a stronger claim than the data license. The defensible claim is: "we cannot detect compounding; the effect is ≤ ~0.09 absolute at n=25." Reviewers who know stats will catch this in minute one. Reframe as an equivalence test (TOST) with an explicit smallest-effect-of-interest, and test "transient first-round lift" separately (round-1 Δ vs rounds 2–6 Δ) rather than asserting it.

**0.5–3B is where nobody expected RSI.** The field's prior is already "small models can't self-improve on hard verified tasks." Confirming the prior is a poster, not an oral. The immediate rebuttal is: "run Qwen2.5-Coder-7B/14B." You have 72 GPUs; not having done it will read as a choice.

**Your actual interesting result is buried.** Two things in your list are stronger than the headline:
- *Search beats self-training at matched compute* (lineage < best-of-N). Known in spirit (test-time-compute papers), but yours is on roofline-graded kernels with a real compute ledger. That's a crisp, useful claim.
- *The coverage gap* (14B teacher harvests 10× more correct kernels; "wall" is low per-sample p, not zero coverage). This is the *mechanism* for the null: self-training on your own samples sharpens toward what you already cover; it can't add coverage. It connects to the sharpening hypothesis and turns a null into an explanation.

**Scope limitation reviewers will hit:** you tested rejection-sampling SFT. If GRPO/RL on the same verifier isn't in the paper, "verified self-training" in the title overclaims. State it or run it.

**Harness findings (SFT-NaN, grader-hang) are appendix material.** Valuable to practitioners, zero weight toward an oral.

**Honest placement:** as-is, a solid D&B poster or strong workshop paper with a genuinely reusable artifact. The artifact (leak-free, compute-matched, roofline-grounded RSI harness — clearly differentiated from KernelBench, which is a static benchmark) is the durable contribution. The result needs one more move to be oral-competitive.

## (2) The single highest-leverage thing

**Run the identical protocol at 7B and 14B with a pre-registered coverage-gap prediction.** Not just "see if it compounds" — commit in advance that per-round compounding gain should track the harvest-coverage gap (pass@k_harvest − pass@1, or self-harvest yield per round), measured at 0.5/1.5/3/7/14B. Then the paper has one of two outcomes, both publishable and both better than now:

- **Still null at 14B** → a much harder negative ("doesn't compound through 14B on a task where the model *can* generate correct kernels at 10× the 3B rate"), plus the coverage-gap curve as the explanation.
- **Compounding appears at 7–14B** → a scale-gate result with a predicted mechanism. That's an oral-shaped finding.

Budget on 72 GPUs: 2 scales × 2 arms × 3–5 seeds × 6 rounds, LoRA SFT, is very feasible; the sampling is the cost, and you already have the pipeline. Also increase to 5 seeds at 3B — your 3B CI is carried by 2–3 seeds.

Do the cheap things regardless, they aren't "the" lever but they're mandatory: (a) the single-shot best-of-(6·k) control so lineage is compared to search at *cumulative* sample parity, not per-round; (b) proper pass@k curves (unbiased estimator, k to 1000+) — this is what makes the "wall is an artifact" claim rigorous and feeds the coverage-gap predictor. Bigger task bank helps power but doesn't change the story; skip unless a reviewer demands it.

## (3) Title + abstract

**Title:** *KernelAscent: Search Beats Self-Training in a Compute-Matched Testbed for Recursive Self-Improvement*

(Alternate if you keep the null as headline: *KernelAscent: Verified Self-Training Does Not Detectably Compound Below 3B*)

**Abstract:**
Recursive self-improvement — a model training on its own verified outputs, then generating better outputs, iteratively — is widely hypothesized but rarely measured under controlled conditions. We introduce KernelAscent, a testbed for verified self-training on CUDA kernel generation, where correctness is checked against reference outputs and speedup is graded against roofline bounds, and where lineage (iterative self-training), reset (fresh model each round), and best-of-N search are compared under a leak-free, compute-matched protocol. Across Qwen 0.5B–3B, 2–3 seeds, and 6 rounds, the lineage-minus-reset effect is +0.040 (95% CI [−0.014, +0.094], pooled n=25): we detect no compounding beyond a bounded first-round lift, and lineage underperforms best-of-N search at matched compute in most rounds. We identify the mechanism as a coverage bottleneck: the sub-2B "correctness wall" is a pass@k artifact of low per-sample success rather than zero coverage, and a 14B teacher harvests 10× more correct kernels than a sub-2B student, so self-training sharpens toward what the model already covers rather than expanding it. Probe-based self-verification achieves within-task AUC ≈ 0.67 but yields no matched-budget harvesting advantage, and a 50-model sweep exposes scale-gated mechanisms in kernel generation. We also document two harness failures — NaN gradients silently corrupting LoRA self-training and un-killable hung kernels deadlocking graders — that produce spurious compounding if uncorrected. KernelAscent provides a rigorous baseline for claims of recursive self-improvement: at ≤3B, sampling more beats training on yourself.

If you add 7–14B, swap the final two sentences for the scale-gate result and the coverage-gap prediction — that version is the one to submit for an oral.


---

# Round 5 feedback (2026-09-16) — post bounded-null, next-steps for rigorous/award paper

## GPT-6 Astra

## Bottom line

**The highest-leverage next step is a randomized “coverage rescue” experiment—not simply another scale point.** Deliberately expand the student’s accessible solution repertoire, hold the alternative explanations fixed, and test whether this changes *subsequent compounding*, not just immediate accuracy.

My ranking:

| Rank | Experiment | Why it matters |
|---|---|---|
| **Prerequisite** | Audit equivalence claims, headroom, and the compounding positive control | Ensures the central claim is actually identified |
| **1** | **Coverage intervention × lineage/reset** | Can turn the proposed mechanism into a causal explanation |
| **2** | **Search–training budget frontier, including amortization** | Turns an empirical ranking into an actionable result |
| **3** | **One targeted RL/GRPO falsification test** | Tests the largest remaining algorithmic boundary |
| **4** | **14B, conditionally** | Useful if it crosses a mechanistically relevant threshold; otherwise another null |
| **5** | **Self-play curriculum redesign** | Potentially interesting, but currently the highest-risk distraction |

I would organize the paper around **when verified search can—and cannot—be converted into reusable capability**. That is stronger than organizing it around a generic absence of RSI.

---

## First: the statistical claim needs narrowing

Based on the intervals you supplied, I would not yet write:

> “Any compounding advantage is below approximately 0.05 through 3B.”

Your pooled interval extends to **+0.055**, and the individual-scale intervals extend to **+0.129, +0.073, and +0.080**. Thus:

- The pooled estimate is consistent with a small average effect.
- It does **not** establish a common <0.05 bound at every scale.
- Whether it establishes formal equivalence depends on the prespecified margin, confidence level, and test.
- The two 7B round differences are useful observations, but are not themselves an uncertainty bound.

A defensible current formulation is:

> “Across the evaluated 0.5–3B conditions, the pooled estimated lineage advantage is 0.003, with interval [−0.048, 0.055]. We find no evidence of compounding under rejection-sampling SFT, while excluding large average advantages at this budget and horizon.”

If the scientifically meaningful equivalence margin is ±0.05, establish it with the appropriate equivalence procedure; do not choose the margin retrospectively to fit the interval. Also distinguish an **average effect across scales** from an **effect bounded at each scale**.

Three additional checks are worth more than another weakly replicated scale point:

1. **Independent units.** Are the \(n\)’s independent training lineages or correlated tasks/rounds within a smaller number of lineages? Resample or model at the level of treatment assignment; account for task-family clustering where relevant.
2. **Headroom.** Near-ceiling scores can suppress a lineage advantage. Include a prespecified, non-ceiling task stratum—not merely an aggregate roofline normalization.
3. **Positive-control specificity.** Oracle-injected correct kernels are a positive control for learning only if they improve the relevant held-out outcome. They are a positive control for *detecting compounding* only if the setup can detect a known sequential advantage over matched reset.

A particularly useful assay check is a small constructed task ladder: mastering stage A makes stage B learnable, which makes stage C learnable. Run the *same lineage/reset protocol*. If this does not produce a detectable advantage where one is designed to exist, the benchmark may be insensitive to sequential accumulation.

That would not establish real-world compounding, but it would make the real-world null much more credible.

---

## 1. The strongest next result: randomized coverage rescue

### The causal claim to test

Your proposed chain is:

\[
\text{low probability of new correct solutions}
\rightarrow
\text{little new learnable information in self-harvested data}
\rightarrow
\text{sharpening without repertoire expansion}
\rightarrow
\text{no lineage advantage}.
\]

The current observations support parts of this chain. They do not yet show that restricted coverage is what **causes the absence of compounding**.

The decisive intervention is:

> **Inject an equal-sized, verified training set that expands accessible coverage rather than reinforcing existing coverage, then test whether autonomous improvement and the lineage/reset gap change.**

### Minimal useful design

Start at one scale with working training infrastructure, measurable learning, and substantial headroom—likely 1.5B or 3B, depending on your actual pilot results.

Cross two factors:

| | Reinforcement injection | Frontier injection |
|---|---|---|
| **Matched reset** | Teacher solutions from student-covered families | Teacher solutions from student-low-coverage families |
| **Lineage** | Same intervention rule | Same intervention rule |

Both injected datasets should come from the **same teacher/source**, rather than comparing self-generated code against teacher code. Otherwise teacher style, code quality, and provenance become obvious confounds.

Control as closely as practical for:

- Number of verified examples and training tokens.
- Optimization quality and code complexity.
- Training FLOPs and hyperparameter-selection budget.
- Access to accumulated data under the original reset protocol.
- Injection timing.
- Teacher-generation costs, charged equally or explicitly reported as a common subsidy.

Estimate student coverage using **independent pre-intervention samples**, then seal evaluation instances. Use held-out shapes/instances and, where appropriate, held-out compositional variants. Reproducing injected kernels is not the desired outcome.

“Low coverage” should mean a measured low probability under a defined sampling budget—not “zero support.” Your own pass@k finding makes that distinction essential.

### The three readouts

**First: manipulation check.**  
Does frontier injection increase held-out coverage beyond reinforcement injection?

**Second: transmission.**  
After injection stops, does the model autonomously harvest correct solutions from additional, previously low-coverage regions?

**Third: compounding.**  
Does frontier injection increase the lineage advantage relative to reset?

The key contrast is:

\[
\delta =
\left(Y_{L,\text{frontier}}-Y_{R,\text{frontier}}\right)
-
\left(Y_{L,\text{reinforce}}-Y_{R,\text{reinforce}}\right).
\]

Evaluate this after several subsequent rounds, not only immediately after SFT.

### Why this is substantially stronger

If you find:

- Equal data volume,
- An actual expansion in held-out coverage,
- More new autonomous harvesting in later rounds,
- And a positive interaction \(\delta\),

then you can say:

> “Changing the information content of the training stream—not simply adding more correct examples—changes whether lineage accumulation helps.”

That is a much more compelling contribution than “larger teachers generate more correct code.”

### Important: the experiment must be allowed to falsify your mechanism

There are three scientifically distinct outcomes:

1. **Coverage expands and compounding appears.**  
   Strong support for coverage as a bottleneck.

2. **Coverage expands, but compounding remains absent.**  
   Coverage was not sufficient. Investigate retention, transfer, optimization efficiency, or insufficient advantage over reset. Do not retain “the mechanism for the null” unchanged.

3. **Coverage fails to expand.**  
   The intervention did not manipulate the proposed mediator. This is not a clean test of the mechanism.

Your oracle-injection experiments may already provide part of this answer. **If they genuinely expanded held-out coverage without inducing compounding, that is evidence against the strongest current coverage explanation.** If they merely supplied more correct examples from already accessible regions, the targeted rescue remains untested.

**This is my answer to (c), and my top experimental recommendation.**

---

## 2. Make search-beats-training a deployment-relevant frontier

Your search result is potentially the most immediately useful finding. Its main vulnerability is **amortization**.

Training pays an upfront cost and may reduce the cost of solving many future tasks. Search pays repeatedly. A single matched-budget comparison may not settle which is preferable over the relevant deployment horizon.

### Run a compact budget-frontier experiment

On a sealed task distribution, compare:

1. Base model + best-of-N verified search.
2. Lineage-trained model + best-of-N verified search.
3. Matched-reset model + best-of-N verified search.

Vary:

- Total available compute.
- Number of deployment tasks.
- Per-task sampling allowance.

Including search on the trained models matters: “training versus search” is otherwise a false dichotomy.

Report:

- Verified correctness and roofline-graded quality.
- Training cost.
- Generation cost.
- Verification/benchmarking cost.
- End-to-end wall-clock cost separately where it changes the practical conclusion.

Use independent timing measurements for selected winners so noisy benchmark selection does not inflate best-of-N gains.

### The question to answer

> “Does self-training ever move the achievable quality–total-cost frontier, and if so, after how much reuse?”

At equal target quality, the simple intuition is:

\[
Q^* \approx
\frac{\text{training cost}}
{\text{search cost per task before training}
-
\text{search cost per task after training}},
\]

when the denominator is positive. The actual analysis should use measured curves rather than assume constant per-task costs.

Either outcome is valuable:

- **No crossover over a meaningful deployment range:** search dominance becomes substantially stronger.
- **A crossover exists:** training is an amortization tool, even though it does not compound.

That second outcome does not weaken the paper. It yields a sharper distinction:

> **Distilling search can improve inference efficiency without creating a self-sustaining improvement process.**

Also tighten the “10× coverage” result. Ten times as many verified kernels is not necessarily ten times as much coverage. Decompose it into:

- Correct samples per generation.
- Unique task families solved.
- Distinct solution strategies, to the extent measurable.
- Yield per token and per unit compute.
- Discovery curves as sampling budget increases.

A reviewer will otherwise ask whether the teacher is producing more duplicates, receiving more compute, or simply shifting probability mass on the same task set.

---

## 3. RL/GRPO: worthwhile as a targeted boundary test, not a new research program

### Does rejection-sampling SFT suffice?

**Yes for a scoped claim. No for an algorithm-general claim.**

It supports:

> “No measurable compounding under verified rejection-sampling SFT in the tested regime.”

It does not support:

> “GPU-kernel self-improvement does not compound,”

without substantial qualification.

SFT and RL differ in ways directly relevant to your mechanism:

- SFT reinforces selected outputs.
- RL can also use relative reward information and alter unsuccessful behavior.
- RL can optimize speed conditional on correctness rather than merely imitate accepted kernels.
- Exploration and on-policy distribution shifts may matter.

None of that guarantees RL will work. It means the SFT null does not settle the question.

### The highest-value RL experiment

Run **one serious, powered lineage/reset test at a deliberately favorable operating point**:

- A scale with enough successful samples to obtain learning signal.
- Tasks with genuine optimization headroom.
- Several rounds.
- A verified reward aligned with your benchmark.
- Matched search baselines.
- A single-round positive control demonstrating that the RL implementation actually learns.

If binary correctness is part of the reward, estimate how often a GRPO group is informative. Under independent sampling with success probability \(p\) and group size \(G\), the probability of a mixed-success group is:

\[
1-(1-p)^G-p^G.
\]

If nearly every group is all-failure, an RL null mostly establishes reward starvation. That can support a bottleneck story, but it is not a strong falsification of RL-enabled compounding.

If you need a shared warm start to get usable rewards, provide it to both RL and SFT and state that the result is conditional on that initialization.

Keep the objective comparable. A speed-optimizing RL method versus a correctness-only SFT selection rule confounds the update algorithm with the target.

### My decision rule

- **For a tightly scoped, mechanism-first paper:** RL is optional.
- **For a broad RSI/compounding headline:** RL is worth more than a routine 14B replication.
- **If implementing stable RL will consume the remaining project:** narrow the claim instead.

One informative RL point is better than a large, reward-starved RL sweep.

---

## 4. Is 14B worth the engineering cost?

**Not simply because it is larger.**

It is worth doing if it tests a predicted transition:

- Does 14B have substantially greater *student-side* verified discovery coverage?
- Does it escape the low-information training regime?
- Does it have enough benchmark headroom?
- Can you afford sufficient independent runs and rounds to obtain an interpretable bound?

The teacher’s much higher verified yield makes 14B potentially interesting—but only if the prospective student at that scale actually crosses the relevant threshold.

### Cheap go/no-go pilot

Before investing in the reset shard, measure:

1. Verified discovery curves.
2. Fraction of usable self-training examples.
3. Single-round held-out learning.
4. Remaining performance headroom.

Then:

- **Coverage meaningfully changes:** 14B becomes a mechanistic boundary test and moves up the ranking.
- **Only baseline score improves:** another scale point has lower leverage.
- **Headroom disappears:** redesign the evaluation stratum before interpreting another null.

I would rather have a credible multi-round causal experiment at 3B than two under-replicated 14B rounds.

Also, I would call the existing result a **scale sweep**, not yet a phase diagram. A phase diagram becomes persuasive when its axes include the variables that plausibly determine the regime: coverage, usable data yield, budget, headroom, and training method—not just parameter count.

---

## 5. Salvaging self-play: constrain task validity, then learn curriculum selection

The current author is solving too many problems at once:

1. Invent a valid task.
2. Make it learnable.
3. Make it useful.
4. Maintain diversity.
5. Improve alongside the solver.

With curriculum collapse and 0/228 passing your learnability gate, this is not yet a clean test of author co-evolution.

### Redesign the author’s action space

Use a verified task generator. Let the author choose:

- Operator/composition.
- Shapes and layouts.
- Dtypes.
- Hardware/resource constraints.
- Difficulty parameters.
- Controlled mutations.

Every emitted task should have:

- Executable semantics.
- An independent verifier.
- A trusted correct witness.

The author should initially learn **which tasks to propose**, not how to invent arbitrary executable task specifications.

### Fix learnability before training the author

Replace the coarse bank with parameterized task ladders. Find settings spanning a useful range of student success probabilities.

Do not assume “50% success” is universally optimal. The useful band depends on harvesting budget: it should generate enough successes to train on while preserving room for improvement.

Your first success criterion should be mundane:

> “The generator can reliably supply valid, diverse tasks on which the solver measurably improves.”

Until that is true, an L−F comparison is mostly measuring failures of task construction.

### Prevent invalidity from starving the solver

Use a common admission gate and a fallback reservoir of calibrated tasks. If the author exhausts its proposal budget, fill the solver batch from the reservoir.

Crucially:

- Charge rejected proposals to the author’s budget.
- Do not grant unlimited retries.
- Report fallback rate as a primary author metric.

Now a collapsed author yields “no benefit beyond fallback” rather than “the solver received no training.”

### Reward transferable learning progress

Do not reward raw solver failure. That encourages impossible tasks.

Use batch-level estimates of improvement on **fresh sibling instances** after a small solver update, along with validity, diversity, and generation-cost terms. Estimate this periodically with shadow evaluations if doing it for every proposal is too expensive.

Keep the final evaluation bank sealed. The author’s validation feedback is training information.

### Make L−F identify the intended effect

If F is the frozen-author arm, give F and L:

- The same initial author.
- The same generator and action space.
- The same proposal budget.
- The same admission rules.
- The same fallback reservoir.
- The same solver-training budget.

Only L receives author learning updates.

If the admission gate adapts to the current solver, then both arms already have some adaptive curriculum behavior. That is fine, but state the estimand accurately:

> “Does learned author adaptation outperform a frozen proposal policy with the same adaptive admission machinery?”

Evaluate L−F on an exogenous bank, not each arm’s own authored tasks.

**My advice: keep this out of the critical path unless author co-evolution is indispensable to the paper’s identity.** Otherwise it is a separate project.

---

## What could move this from “solid negative” to an exceptional paper?

No individual result guarantees acceptance or an award. But the most persuasive package would be:

### 1. A null with a validated detector

The same protocol detects sequential gains on a constructed positive control, yet yields a tight, appropriately scoped null on real kernel tasks.

### 2. A causal boundary

A randomized intervention changes accessible coverage and changes—or demonstrably fails to change—compounding in the predicted way.

### 3. A resource-allocation law practitioners can use

You identify when to spend compute on:

- More samples.
- A stronger teacher.
- Student training.
- Repeated deployment.

### 4. An unusually reproducible benchmark

Your retained artifacts and infrastructure fixes are assets. Release enough to reconstruct:

- Lineage/reset ancestry and data access.
- Full compute accounting.
- Kernel verification.
- Timeout/failure handling.
- Timing methodology.
- Statistical aggregation.

That makes the benchmark useful even to researchers who disagree with your interpretation.

**If I could request only one new headline result, it would be:**

> **“At fixed verified-example count and budget, frontier-expanding supervision—but not reinforcement of already-covered solutions—enables additional autonomous discovery and increases the lineage/reset advantage.”**

That would turn “we did not observe compounding” into **an experimentally supported account of what prevents it and what changes the regime**.

If the rescue fails despite a successful coverage manipulation, that is also important—but the paper should then explicitly conclude that coverage alone does not explain the null. The credibility gained by testing that possibility is itself high leverage.

## Claude Fable-5.1

## Executive read

You have a bounded null on *parameter* compounding and a mechanism hypothesis (sharpening-not-expansion) that is currently associational. The path from "solid negative" to "must-accept" is not more nulls at more scales; it is (1) making the mechanism causal and quantitative, (2) showing the mechanism *predicts a positive regime* and then demonstrating it, and (3) showing the mechanism is optimizer-agnostic. A null with a boundary is a phase diagram; a null without one is a benchmark report.

Two things reviewers will hit that you didn't ask about, fix first:

- **Round count.** "7B holds the null over rounds 0–1" is not a compounding test. Compounding claims need ≥4–5 rounds with a per-round trajectory plot and bootstrap CIs, at least at ≤3B where it's cheap. State 7B as a scale check, not a compounding result, unless you extend it.
- **Ceiling confound.** 7B at C≈0.70 against a held ceiling of 0.80 has almost no headroom; a null there is uninformative. Report headroom-normalized gains (fraction of ceiling gap closed) everywhere, and build a hard tier for anything ≥7B.
- **Statistics.** Say "equivalence" formally: TOST with margin ±0.05 on the pooled n=34, plus a Bayes factor for H0 vs. a +0.05 effect. Your CIs support it; make it a test, not a description.

---

## Ranked next experiments

### 1. Make the mechanism causal: p-maps + coverage transplant + sharpening law  (highest leverage, lowest cost)

**1a. Per-task p-maps across rounds.** For every task *t* and round *r*, sample k≈256–512 kernels, compute p_r(t) (fraction verified-correct) and best roofline fraction. Cost is grading, not training. This one artifact yields four figures:

- *Mass-shift histogram*: distribution of p_r(t) across rounds. Prediction: intermediate-p mass moves toward 1; the p≈0 mass doesn't move. Reset arm as control.
- *Coverage set*: C_r = {t : p_r(t) > 0}. Test |C_r \ C_0| vs. matched-reset. Prediction: ≈0 beyond sampling noise.
- *pass@1 vs pass@k trajectories*: if pass@1 rises while pass@k flattens or falls, you've shown the lineage has *less* harvestable frontier each round — a diversity death spiral that mechanically forbids compounding under rejection sampling and directly predicts finding (i). Also track distinct-solution count per task (AST hash) as the diversity proxy.
- *The amortization ceiling scatter*: per task, lineage round-R pass@1 vs. round-0 pass@k_harvest. Points above the diagonal = improvement beyond amortizing search. Prediction: none beyond noise. This is the single figure that unifies (i) and (ii).

**1b. Sharpening law.** Fit Δp_r(t) as a function of p_0(t) and harvest budget k. Prediction: Δ≈0 for p_0 < ~1/k, peaked at intermediate p_0, saturating near 1. If the same curve fits 0.5B/1.5B/3B (and later GRPO), you have a quantitative, cross-scale law rather than a null.

**1c. Coverage transplant 2×2 (the causal test for (c)).** At 1.5B (your best-powered scale), token-matched arms:
- L: lineage only
- L+T_in: + teacher kernels on tasks the student *already covers* (sharpening only)
- L+T_out: + teacher kernels on tasks the student *does not cover* (coverage only, same token count)
- L+T_rand: + random teacher subset

Then run ≥3 *self-only* rounds after injection. Predictions: T_out > T_rand > T_in ≈ L; and crucially, injected coverage *persists and gets sharpened* by subsequent self-rounds. Add a dose-response arm (inject 10/25/50/100% of uncovered tasks). If gain is monotone in coverage injected and ~zero for T_in, coverage is causally the mediator — and you've converted your oracle positive control into a mechanism demonstration: *self-training is a sharpener that requires an external coverage source.*

**1d. Held-out families.** Partition the bank by op family; train lineage on some, evaluate on held-out. Δ≈0 on held-out = per-task amortization, not a transferable skill. This is also the fair answer to "but training amortizes across future queries" (see #4).

### 2. Oracle frontier curriculum — the precursor that decides whether self-play matters at all  (answers (b) and is your best shot at a positive regime)

The mechanism predicts exactly what a curriculum must do: put tasks with p_0 ∈ (1/k, ~0.5) into the harvest. You don't need an LLM author to test this — you have 14B-harvested difficulty maps.

- **Goldilocks arm**: each round, train only on tasks whose current p_r is in the frontier band, vs. uniform.
- **Expanding-k arm**: raise k_harvest per round (16→64→256) so frontier tasks mechanically enter the harvest. Compute-match against best-of-N.
- **Combined arm**.

If either compounds (lineage > reset, and coverage set grows), you have the positive regime and the paper's boundary: *compounding occurs iff new coverage enters the harvest — from a teacher, from a curriculum, or from a larger k — never from the lineage's own fixed-budget samples.* If even the oracle curriculum fails, the author is irrelevant and self-play becomes a one-paragraph lessons-learned. Either outcome is decisive, which is why this outranks fixing the author.

### 3. Close the RL loophole with a cheap→expensive ladder  (answers (a); insurance, but mandatory)

Rejection-sampling SFT is filtered behavior cloning ≈ REINFORCE with binary reward; theoretically it cannot leave the support of the policy, and the RLVR-sharpening literature (Yue et al. 2025 and successors) says the same for GRPO. Your mechanism therefore *predicts* GRPO sharpens faster but leaves C_r frozen. But every reviewer will write "you used the weakest possible training method" unless there is an RL arm. Do it as a ladder so you can stop early if the story holds:

- **A1: speedup-weighted rejection sampling** (keep top-quartile roofline fraction among correct). Tests the "dense reward matters" objection with zero new infra.
- **A2: DPO with (correct, incorrect) pairs per task.** Tests the "negatives matter" objection.
- **A3: one GRPO arm** at 1.5B or 3B, reward = correct × roofline fraction (+ small compile shaping), ≥4 rounds, n≥8, compute-matched, with the same p-map instrumentation as #1. Watch for grader exploits — any reward hacking you catch is a bonus finding, not a failure.

Rejection-sampling alone does *not* suffice to claim "no compounding" to a skeptical audience. But you don't need a full RL axis; you need one well-instrumented arm that shows the same coverage-frozen signature. If GRPO *does* expand coverage, that's a more interesting paper, not a worse one.

### 4. Break-even amortization analysis for search-vs-training  (cheap; makes (i) fair and actionable)

Finding (i) is currently vulnerable: best-of-N spends compute per query; training amortizes. Compute the break-even: how many future queries before lineage training pays off, given (a) in-distribution gain and (b) held-out gain from 1d. With ~zero held-out transfer, break-even is infinite for novel tasks and finite only for repeated tasks — a concrete engineering recommendation ("in domains with cheap, dense verification and low cross-task transfer, spend on search"). Add a *repeat-task positive control* (same tasks queried repeatedly) to show your harness detects the regime where training should win.

### 5. 14B  (answers (d); medium — do it only in a specific form)

The 0.5–7B diagram is flat, so 14B on the *same bank* adds one point and inherits the ceiling confound. Worth the 2-GPU shard only if:
- you run it on a hard tier (tasks where 14B p_0 ∈ (0.01, 0.5)), and
- you instrument it with the #1 p-maps, and
- you treat it as the "your model is just too weak" refutation: the model with the most coverage and the most to sharpen still shows frozen C_r.

If GPU-constrained, everything in #1–#4 outranks it. A cheaper substitute: one round of p-maps on a 32B open model to show the sharpening law extrapolates.

### 6. Self-play author redesign  (only after #2 shows an oracle curriculum works)

Collapse to "no valid tasks" and P3's 0/228 learnability pass are the same disease: free-form generation + a sparse validity/learnability gate. Redesign so validity is structural and learnability is searchable:

- **Knob-parameterized DSL**: shape, dtype, fusion-chain length, layout, reduction dims, tiling constraints. Author proposes *edits to knobs*, not free-form tasks. Reference implementation is composed from verified primitives so validity is near-guaranteed.
- **Adaptive difficulty**: continuous knobs let you bisect to the frontier band (p ∈ (1/k, 0.5)) — this is why P3 saw only too-easy/too-hard; discrete mutations can't land in the band.
- **Minimal-criterion archive** (POET/PAIRED-style): tasks that pass validity + learnability enter a persistent archive; the curriculum samples from the archive, so it can never collapse to empty.
- **Author reward**: solver learning progress (|Δp| across rounds) or intermediate p, plus shaping for compile/finite-reference/determinism.
- **Measuring L−F**: arms S (solver on fixed bank), F (frozen round-0 author), L (co-trained author); outcome = solver performance on *held-out human-written tasks* (KernelBench-style) at matched compute. L−F > 0 with coverage-set growth is the genuine co-evolution result.

### Cut or relegate
Probe-as-intervention → appendix. P3 as-is → lessons learned. Current author → lessons learned unless #6 runs.

---

## (e) What flips "solid negative" to "must-accept"

1. **A quantitative, cross-scale, cross-optimizer law** (1b), not a null: "Δp is a function of (p_0, k) and nothing else we varied."
2. **Causal mediation with dose-response** (1c): coverage injection restores compounding; sharpening-only injection doesn't.
3. **A demonstrated positive regime** (#2): compounding appears when the frontier enters the harvest. A boundary makes the null a phase diagram.
4. **Optimizer-agnosticism** (#3): SFT, DPO, GRPO share the coverage-frozen signature.
5. **The unifying figure** (1a amortization scatter): lineage never exceeds round-0 best-of-k on any task — self-training is compression of search.
6. **Actionable economics** (#4): break-even analysis with a clear recommendation.
7. **Artifacts**: the hardened harness (grader deadlock fix, NaN guard), p-maps, and all S3 lineage artifacts released. The grader-hardening alone is a contribution people will use.

The framing that ties it together: *"Self-training sharpens; it does not explore. In kernel optimization, compounding is coverage-limited, and coverage is not self-generated."* Lead with the law and the transplant result; the null becomes the corollary rather than the headline.


---

# Round 6 feedback (2026-09-18, GPT-6-Astra, 24h final push)

**The highest-leverage move is a causal “coverage rescue” experiment—not another confirmation of the null.**

You already have substantial evidence that the tested self-training procedures do not compound. What is still missing, based on your summary, is the experiment that separates:

> “These training recipes do not work”

from:

> “We can identify the bottleneck, intervene on it, and predict what happens next.”

That is the difference between an interesting negative-results paper and a compelling benchmark paper with an explanatory result.

**My recommendation: spend the marginal GPU budget on a controlled injection of frontier-opening training examples, then test whether it produces a one-time improvement or sustained endogenous discovery.** Keep the large-scale lineage-versus-reset runs; deprioritize additional descriptive mech-interp seeds and T5 breadth.

No result guarantees an award. But this most directly addresses the skeptical reviewer’s strongest remaining objection.

## 1. The key weakness to fix

Your results support three claims at different strengths:

| Claim | Current support, from your summary |
|---|---|
| These self-training methods do not produce meaningful compounding at the tested scales and budgets. | **Strong**, especially if equivalence bounds are defensible. |
| At matched compute, search is a better allocation than these training methods. | **Strong**, assuming full cost and selection accounting. |
| Limited proposal coverage *causes* the absence of compounding. | **Suggestive, not yet established causally.** |

P-maps, formation-failure forensics, and sharpening geometry make a coherent mechanism. But they do not alone exclude alternatives: insufficient training signal, poor optimization, destructive updates, finite sampling, or an inadequate self-training recipe.

The reviewer objection is:

> “You observed low coverage and unsuccessful training. How do you know coverage is the operative bottleneck rather than a correlated symptom?”

A rescue experiment answers that much more convincingly than additional null seeds.

Also, your existing evidence does **not** justify “self-training does not explore” as an unrestricted statement. With finite sampling, apparent zero-probability tasks may simply have very low probability. The defensible object is **coverage at a specified sampling budget**, not mathematical support.

---

## 2. The single experiment I would run

### A frontier-opening pulse, followed by endogenous recursion

**Question:** If we give the system useful examples beyond its current empirical frontier, can it learn from them—and does that unlock continued discovery after external assistance stops?

Use **one scale with both demonstrable trainability and substantial remaining headroom**. From your summary, 7B is a plausible candidate. Do not choose it just because it is larger; choose it using existing calibration evidence about headroom and responsiveness.

### The intervention

Construct two equally sized seed buffers from the **same source of validated kernels**, ideally existing cached artifacts:

- **Frontier-opening buffer:** examples from training tasks or training strata with low baseline coverage.
- **Coverage-redundant buffer:** examples from training tasks or strata the baseline already covers.

Match, as far as feasible:

- Number of examples and training-token budget.
- Correctness and reward distributions.
- Kernel length and task-family composition.
- Collection and validation costs.
- The training procedure and seed-buffer replay policy.

Use a separate calibration set to define the frontier. **Do not select seed examples using final evaluation outcomes.** Evaluate on untouched instances under the paper’s declared generalization regime.

This is a causal test of a **data-selection intervention**, not a perfectly isolated manipulation of abstract “support.” Be explicit about that limitation, particularly if difficulty cannot be well matched.

### The design

Cross the seed intervention with ancestry:

| | Coverage-redundant seed pulse | Frontier-opening seed pulse |
|---|---:|---:|
| **Reset** | Restart from the original checkpoint each round | Restart from the original checkpoint each round |
| **Lineage** | Continue from the previous checkpoint | Continue from the previous checkpoint |

After the pulse, **no new external examples enter**. Subsequent new data must come from the benchmark’s endogenous procedure. Both ancestry conditions get the same access to the fixed seed buffer, with the same replay policy.

Your existing unassisted runs provide useful context, but **they do not replace the redundant-seed control**: otherwise any rescue could simply reflect receiving more good training data.

Use independent paired runs, not one trajectory with many tasks presented as independent training replicates.

### Why the reset factor matters

An immediate improvement after adding good examples is not recursive improvement.

The central comparison remains:

\[
\Delta_r =
\operatorname{Score}(\text{lineage at round }r)
-
\operatorname{Score}(\text{reset at round }r),
\]

under matched budgets and the same external-data allowance, with the score oriented so positive means lineage helps.

The question is whether the pulse creates:

1. A **level change**: the system learns newly supplied competence.
2. A **growing ancestry advantage**: retaining the previous generation enables continued gains beyond resetting.
3. **Further frontier expansion without additional external help**.

Those are different phenomena. Your paper becomes stronger by separating them.

### The most compelling outcome

The cleanest result for your current story would be:

> Frontier-opening examples produce a measurable held-out improvement; redundant examples produce mostly sharpening; after the external pulse ends, frontier expansion stalls and lineage remains equivalent to reset.

That would show:

- The model is not simply untrainable.
- Your pipeline can transmit useful new competence.
- Which examples enter training matters.
- Learning supplied competence does not automatically create a self-sustaining discovery loop.

Crucially, it still would **not** prove that coverage is the only limiting factor. It would substantiate the narrower and stronger claim that **coverage expansion is actionable, while endogenous recursion fails to sustain it under the tested conditions**.

And if the pulse *does* unlock sustained lineage gains? That is potentially an even better paper:

> Recursive improvement emerges only after crossing an identifiable coverage threshold.

Do not design the analysis so that only the null is a success.

---

## 3. The one figure most likely to flip a reviewer

### **“Learning new competence is not the same as generating the next frontier.”**

One figure, three panels, all from the same intervention.

### Panel A — The intervention works

Plot held-out task coverage at a **fixed probe budget \(Q\)** against round.

- Frontier-opening pulse versus redundant pulse.
- Lineage versus reset.
- Mark when external assistance stops.
- Show uncertainty across independent runs.

The key visual is whether the frontier intervention actually lifts coverage. If it does not, there is no successful rescue, and you should not sell it as one.

Use fresh, equally sized probe batches at each checkpoint. **Do not plot cumulative “ever solved” coverage as evidence of exploration**: it rises mechanically with repeated attempts.

### Panel B — Level gain versus compounding

Plot lineage-minus-reset against round, including your equivalence band.

This distinguishes “the seed helped” from “recursion helped.”

A visible coverage increase in Panel A alongside equivalence in Panel B is much more informative than either result alone.

### Panel C — Where gains come from

Decompose improvements into:

- Increased success on already-covered tasks.
- Newly covered tasks under the fixed-\(Q\) definition.
- Correctness improvements after extraction.
- Performance improvements after correctness.

This links the causal intervention to your existing failure-locus story.

**The persuasive pattern is:**

> We can move the frontier externally; the model can absorb that movement; the subsequent endogenous loop mostly concentrates probability within the resulting frontier rather than extending it further.

That is a mechanism-backed result, not merely an unsuccessful training run.

---

## 4. Ranked next moves

### **1. Run the frontier-opening pulse experiment**

**Highest marginal scientific value.**

Scope it narrowly: one scale, one trusted training recipe, paired seeds, enough rounds to distinguish an immediate jump from subsequent growth.

Do not spend the entire day constructing a perfect donor dataset. This is attractive specifically if your existing artifacts make the intervention cheap.

**Go/no-go condition:** launch a small pilot early. If you cannot obtain a valid, leakage-free seed buffer and demonstrate that the pipeline can learn from it, fall back rather than burn the remaining window.

### **2. Finish the 7B/14B lineage-versus-reset extension**

**Highest defensive value.**

It closes the obvious “everything below 3B is below the capability threshold” objection.

But interpret it in conjunction with headroom:

- A null near a saturated score ceiling is weak evidence about recursion.
- A null with substantial remaining task headroom is more informative.
- Adequate sample size is not a substitute for a meaningful equivalence margin.

If compute forces a choice, prioritize a well-powered run at a scale with headroom over two underpowered scale points. Your stated 14B coverage of 86% makes the ceiling issue especially worth handling carefully, though coverage saturation need not imply performance saturation.

### **3. Turn the existing artifacts into a fixed-budget frontier-transition analysis**

**Best fallback if the new intervention is infeasible; probably worth doing regardless.**

For each round and ancestry condition, estimate:

- Probability mass shifted onto existing successes.
- Newly covered tasks under an explicit fixed sampling budget.
- Covered tasks subsequently lost.
- Which failure stage changed.

Compare against repeated sampling from the frozen starting model at the same total proposal budget.

The important question is:

> Does training discover tasks faster than additional sampling would have discovered them anyway?

This sharpens your p-map story and prevents “newly observed success” from being mistaken for training-generated novelty.

Treat low-probability classifications carefully. A task with zero observed successes is **unobserved at this budget**, not proven unreachable.

### **4. Complete A1, but only as a targeted alternative-explanation test**

A stronger reward is useful if it demonstrably improves some intermediate outcome while compounding remains absent.

The valuable result is:

> “The alternative objective improves immediate learning or sharpening, but not lineage advantage.”

A second optimizer that changes nothing is less informative: reviewers can still say both recipes failed to learn.

Call this **robustness to the tested objective**, not optimizer-agnosticism in general.

### **5. Freeze the protocol and make the benchmark independently usable**

Low GPU cost; high benchmark-paper value.

The final artifact should make it easy for someone to submit a method and obtain:

1. A best-of-\(N\) search baseline.
2. A lineage-versus-reset comparison.
3. A coverage/sharpening decomposition.
4. A compute ledger and uncertainty report.

Provide a clean split between development/calibration and final evaluation, plus frozen seeds and documented selection rules where appropriate.

A benchmark paper wins on what it enables others to measure, not just on the number of experiments its authors ran.

### **6. Additional WHY-RSI seeds and T5 self-play breadth**

Finish cheap jobs already near completion, but these are my first candidates for GPU reallocation.

More precision on “large weight movement can coexist with little gain” is unlikely to resolve the core causal objection. Unless the weight analysis localizes and intervenes on a mechanism, describe it as a **weight-space diagnostic**, not the central mechanistic explanation.

T5 is useful as an external-validity case study. It should not compete with the controlled core for the paper’s main narrative.

---

## 5. How I would use the remaining GPUs

**Keep the large-scale null extension. Reallocate from descriptive breadth, not from the essential control experiment.**

A reasonable target, conditional on your measured runtimes:

- **About two-thirds:** finish the important 7B/14B and already-valuable A1 runs.
- **About one-third:** the focused frontier-pulse study.
- **CPU/analysis effort in parallel:** fixed-budget frontier accounting, figure construction, and protocol audit.

For illustration, 24 GPUs could support twelve concurrent two-GPU jobs—four cells with three paired seeds—**only if the full trajectories fit your actual runtime**. Three seeds would still give limited precision on training-run variability; do not hide that behind hundreds of task observations.

Prefer fewer scales and adequate trajectories over a larger grid that cannot distinguish a one-round jump from a continuing trend.

Before launching, write down the interpretation table:

| Observation | Interpretation |
|---|---|
| Frontier pulse helps; lineage does not | Exogenous competence transfer without demonstrated recursive advantage |
| Frontier pulse helps; lineage advantage grows | Evidence for a conditional route to compounding |
| Both seed buffers help similarly | Generic data benefit; frontier specificity not established |
| Neither helps | Failed rescue or insufficient dose; no causal coverage conclusion |

That prevents an ambiguous result from turning into an overclaimed story.

---

## 6. Your framing: sharpen it, and reduce its scope

Your proposed framing—

> “Self-training sharpens, does not explore; compounding is coverage-limited and not self-generated”

—is memorable but currently more categorical than the evidence warrants.

Problems:

- “Does not explore” sounds universal.
- “Coverage-limited” sounds causally established.
- “Not self-generated” can sound like a claim about recursive improvement generally.

### My preferred headline now

**KernelAscent: Separating Search, Learning, and Recursive Improvement**

Main takeaway:

> **At matched compute, search outperforms the tested self-training methods; training primarily sharpens existing successes rather than producing a growing lineage advantage.**

This makes the **measurement distinction** the benchmark’s durable contribution. The paper remains valuable even when a future method compounds.

### Stronger headline if the rescue works

**KernelAscent: Learning New Skills Does Not Guarantee Recursive Improvement**

Main takeaway:

> **External frontier expansion yields learnable gains, but those gains do not become self-sustaining under the tested recursive procedures.**

I would avoid making the paper’s identity “RSI is impossible” or even “self-training cannot explore.” The strongest benchmark invites someone to break its empirical conclusion using a well-defined test.

---

## Bottom line

**You have enough null evidence to make the paper credible. The next unit of effort should make it explanatory.**

The reviewer-converting result is not simply a larger-scale null. It is:

> **A positive control that moves the hypothesized bottleneck, visibly improves capability, and then cleanly separates that improvement from recursive advantage.**

Finish the large-scale extension for credibility. Run the frontier pulse for causal leverage. Build the paper around **search versus learning versus compounding**, with coverage as the tested explanatory axis—not an assumed universal law.