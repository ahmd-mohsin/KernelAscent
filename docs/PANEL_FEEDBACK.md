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