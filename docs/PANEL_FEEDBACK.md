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
