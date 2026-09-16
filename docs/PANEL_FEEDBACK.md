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
