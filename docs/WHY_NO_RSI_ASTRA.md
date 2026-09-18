## Bottom line

The strongest account is **a failure to convert verified discoveries into an expanding, retained solution repertoire**, not an absence of useful solutions and not an inability to change the weights. At low scale, the bottleneck is largely generating an extractable kernel; at intermediate scale, there is enough reachable success and enough remaining headroom for self-training to help; at large scale, updates increasingly redistribute probability within an already-covered, near-ceiling repertoire, while diversity and retention limit further accumulation.

But an important qualification belongs at the center of the Discussion:

> **Your results support a protocol-specific, operational self-training plateau. They do not establish that an actual neural model’s reachable set is mathematically invariant, that the weights are at a genuine local optimum, or that RLVR cannot discover new solutions.**

There are three distinct claims here:

1. **The causal experimental result:** carrying the lineage forward does not outperform resetting under the tested training protocol.
2. **The supported mechanism:** selected-sample training mostly sharpens existing successes, and its benefits do not accumulate into broader, retained capability.
3. **The idealized theorem:** exact, support-preserving rejection-sampling updates cannot expand solution support.

The third helps explain the second, but it is not itself a theorem about the experimental models.

---

## 1. Mechanistic account, ranked by explanatory power

### 1. A discovery–assimilation bottleneck: useful tail capability is present, but training does not turn it into cumulative capability

**This is the best-supported explanation across the experiments.**

The two strongest results jointly identify the bottleneck:

- Lineage and reset are equivalent within the tested margin: pooled difference \(+0.001\), CI \([-0.015,+0.018]\), with evidence favoring the null.
- Verified best-of-\(N\) beats lineage training at matched compute by \(0.113\), CI \([0.074,0.153]\) in favor of search.

Meanwhile, the large pass@\(K\)/pass@1 gap shows that successful outputs are often **available in the sampling tail without being reliably produced in one sample**.

Thus, the immediate limitation is not simply “the model cannot solve the task.” It is that the pipeline

\[
\text{rare successful sample}
\;\rightarrow\;
\text{selected training example}
\;\rightarrow\;
\text{weight update}
\;\rightarrow\;
\text{reliably reusable improvement}
\]

is lossy. Search can directly exploit a rare discovery; training must assimilate it into a shared parameterization, preserve other skills, and make the resulting gain survive subsequent rounds.

Your lineage/reset result is especially informative here: **inheriting previous updates does not add measurable value over restarting under the same protocol.** That is evidence against a useful cumulative learning state, not merely evidence that individual updates are small.

**Crucial limitation:** sharpening alone does not logically imply a lineage null. If successive updates reliably increased success probability on a broad band of tasks while preserving earlier improvements, lineage could outperform reset without expanding exact support. The observed null therefore requires some combination of:

- rapid exhaustion of the learnable gains;
- incomplete assimilation of successful samples;
- redundancy of later training data;
- interference or forgetting;
- or insufficient optimization efficacy under the allocated compute.

Your downstream evidence helps discriminate among these, but does not uniquely identify their causal shares.

### 2. A narrow learnable frontier, bounded below by production failures and above by remaining headroom

The sharpening geometry and failure forensics explain **where useful updates can occur**.

Partition tasks operationally into:

1. **Not discovered within the sampling budget.**
2. **Occasionally solved, with room to increase success probability or quality.**
3. **Already reliable or close to the graded ceiling.**

Most self-training value comes from the middle group. The measured \(0<p<0.2\) band is therefore a plausible **productive frontier**: enough success to supply positive examples, but enough failure to leave improvement available.

This yields a natural explanation for the inverted-U:

- **Small models:** too few useful accepted examples; formation failures dominate.
- **Intermediate models:** a larger productive frontier; discoveries can become training signal.
- **Large models:** more initial coverage, but less remaining headroom on covered tasks; more update activity goes into rearranging already-successful behavior.

For sub-3B models, the no-extract rates make the mechanism more specific. Let \(F\) denote producing an extractable kernel and \(A\) denote an accepted solution. When acceptance requires extraction,

\[
P(A\mid x)=P(F\mid x)\,P(A\mid F,x).
\]

A low formation probability throttles the supply of useful examples before correctness or performance learning can even begin. Rejection sampling supplies no positive target for an attempt that never becomes an extractable kernel.

**Supported:** the immediate observed failure locus is kernel production, and it moves downstream with scale.

**Not established:** that correctness would be good if formation were fixed. Correctness conditional on extraction concerns a selected subset; no-extract failures can mask additional downstream deficiencies.

Similarly, the inverted-U is consistent with shrinking headroom, but aggregate rates alone do not prove saturation. Stronger evidence would show that gains track **baseline-conditioned remaining roofline headroom**, not just scale.

### 3. Diversity contraction and imperfect retention prevent improvements from accumulating

This is a strong **reinforcing mechanism**, with a causal qualification.

Selected-sample SFT can create the feedback loop

\[
\text{current common successful modes}
\rightarrow
\text{accepted dataset}
\rightarrow
\text{higher probability on those modes}
\rightarrow
\text{less representation of alternatives in the next dataset}.
\]

The loss is not necessarily correctness on today’s task. It can be **option value**: alternative decompositions, kernel families, implementation strategies, and rare behaviors that would enable later discoveries.

Let \(U_t\) be the probability mass assigned to successful strategies that are not yet represented in the retained repertoire. With \(N\) independent draws, the probability of discovering at least one is

\[
1-(1-U_t)^N \le N U_t.
\]

If self-training reduces \(U_t\), later rounds encounter fewer novel successes even if pass@1 rises on familiar ones. The loop can become increasingly good at reproducing its existing winners while increasingly poor at finding the next winner.

Retention matters independently. A useful conceptual decomposition is

\[
\Delta \text{retained repertoire}
\approx
\text{new discoveries assimilated}
-
\text{previous capabilities lost}.
\]

Higher retention among compounders is consistent with this balance being a determinant of compounding. Diversity collapse at large scale supplies a plausible reason the discovery term shrinks.

**However:**

- The evidence establishes association, not that diversity collapse causes the plateau.
- Low diversity can result from saturation or successful specialization rather than cause failure.
- Token entropy is not the same as useful strategy diversity.
- Maximum-likelihood SFT is not inherently a mode-seeking optimization objective. Here, contraction would arise from **selection, finite datasets, repeated resampling, and shared-parameter generalization**, not from a universal property of SFT.

A particularly informative intervention would preserve **behavioral strategy diversity at matched reward, accepted-example count, and training compute**. Recovery of later-round gains would support the locking mechanism directly.

### 4. Weight updates increasingly implement behavioral churn rather than useful capability growth

The weight-level results rule out a simple explanation: **the models are not failing merely because their parameters remain unchanged**.

Large LoRA drift alongside low RSI incidence shows that update magnitude and improvement are decoupled. Late-layer localization is consistent with changes to output selection, formatting, or use of existing representations rather than broad acquisition of new computational machinery.

But that last interpretation must remain tentative:

- Late layers can implement substantive computation.
- Parameter-space distance is not a calibrated measure of functional change.
- If drift is measured in LoRA factor coordinates, rescaling and factorization symmetries can complicate interpretation; the effective update \(\Delta W\) or behavioral distance is more informative.
- Cross-scale drift comparisons require comparable normalization.

Accordingly, **“churn without gain” is supported descriptively; “only superficial late-layer edits” is not yet established mechanistically.**

This evidence complements the headroom and diversity account, but is less explanatory by itself: observing movement without gain does not identify why the movement is unproductive.

### 5. Rapid strategy exhaustion extends the pattern beyond weight-based SFT

Task-5 shows a similar behavioral signature through a different adaptation interface: substantial round-0 gains, mostly non-recursive trajectories, and strategy-archive saturation by rounds 1–2.

That is consistent with extracting a small set of accessible improvements rather than building a process that becomes progressively better at producing improvements.

It strengthens the **finite-frontier/exhaustion interpretation**. It does not establish that self-modification and LoRA SFT share the same internal mechanism, nor that archive saturation implies exhaustion of all useful strategies. The archive may fail to discover or distinguish them.

---

## 2. Is this “sharpening, not exploration”?

**Yes, as the leading interpretation of this rejection-sampling SFT protocol. No, as a universal impossibility result for self-training or RLVR.**

A verified positive example supplies two things:

- selection information: *this sampled candidate works*;
- an imitation target: *increase the probability of this candidate or related candidates*.

What it does not necessarily supply is a reliable method for reaching tasks or strategies that the proposal policy never successfully generates. When useful samples are rare, the training signal is sparse; when accepted samples are repetitive, the signal becomes redundant.

This closely matches the familiar concern that self-training **improves elicitation or concentration of existing competence faster than it expands competence**.

But “already reachable” has two meanings that must not be conflated:

- **Exact support:** any solution with strictly positive sampling probability.
- **Operational reachability:** a solution likely enough to be encountered under the available budget.

A model can make a practically transformative gain by moving a solution from probability \(10^{-10}\) to \(10^{-2}\), without expanding exact support at all. Conversely, a softmax language model may assign positive probability to essentially every legal finite sequence, making exact support coverage nearly vacuous.

Your evidence concerns **operational reachability and graded performance**, not a directly observed boundary of exact support.

---

## 3. The precise fixed-point statement

### Proposition: coverage invariance under idealized rejection conditioning

Let:

- \(x\sim D\) be a task;
- \(\mathcal Y_x\) be a countable output space;
- \(\pi(y\mid x)\) be the current proposal distribution;
- \(a_x(y)\in\{0,1\}\) be a fixed acceptance predicate;
- \(p_\pi(x)=\sum_y \pi(y\mid x)a_x(y)\).

Define the idealized rejection-conditioning operator

\[
(R\pi)(y\mid x)=
\begin{cases}
\dfrac{\pi(y\mid x)a_x(y)}{p_\pi(x)},&p_\pi(x)>0,\\[6pt]
\pi(y\mid x),&p_\pi(x)=0.
\end{cases}
\]

This assumes unlimited access to accepted samples and exact, taskwise distribution fitting, with no cross-task transfer.

Define exact solution-support coverage

\[
C_0(\pi)=\Pr_{x\sim D}\big[p_\pi(x)>0\big].
\]

Then

\[
C_0(R\pi)=C_0(\pi),
\qquad
R^2\pi=R\pi.
\]

**Reason:** on every task with positive success probability, rejection conditioning places all mass on accepted outputs, making success probability one. On tasks with zero success probability, no accepted distribution exists, and the stipulated fallback leaves the policy unchanged. The reachable task set is invariant, and the idealized distribution reaches a fixed point after one update.

More generally, if every update obeys

\[
\operatorname{supp}\pi_{t+1}(\cdot\mid x)
\subseteq
\operatorname{supp}\pi_t(\cdot\mid x),
\]

then exact solution-support coverage cannot increase:

\[
C_0(\pi_{t+1})\le C_0(\pi_t).
\]

Equality additionally requires retaining at least one accepted output on each previously covered task.

### What this theorem does **not** say

**1. Actual neural SFT is not this operator.**  
Shared parameters can generalize across tasks, increase the probability of unsampled outputs, or transfer a newly learned construction. Consequently, neural SFT need not preserve support or operational reachability.

**2. Finite-budget coverage is not invariant.**  
For independent sampling,

\[
C_K(\pi)
=
\mathbb E_{x\sim D}\left[1-(1-p_\pi(x))^K\right].
\]

Sharpening can increase \(C_K\) dramatically even when \(C_0\) is unchanged. Therefore:

> **“Exact solution-support coverage is invariant under idealized support-preserving rejection conditioning” is defensible. “Measured pass@K coverage is a self-training fixed point” is not generally true.**

**3. Zero observed successes does not establish \(p=0\).**  
After \(K\) independent failures, a one-sided 95% binomial upper bound is

\[
p \le 1-0.05^{1/K}\approx \frac{3}{K}.
\]

Correlated or adaptive sampling requires a different treatment. Your “unreachable mass” should therefore be called **undiscovered mass at the evaluated budget**, unless additional evidence identifies structural impossibility.

**4. Binary acceptance is not roofline optimization.**  
Conditioning on correctness need not concentrate probability on the fastest correct kernel. For graded reward \(r_x(y)\), the support argument can be applied separately to each threshold,

\[
a_{x,\tau}(y)=\mathbf 1[r_x(y)\ge\tau],
\]

but the one-step idempotence result does not automatically extend to arbitrary reward weighting or best-of-\(N\) selection.

**5. A training-operator fixed point is not a parameter-space local optimum.**  
A true local optimum would require showing that sufficiently nearby admissible parameter changes cannot improve the objective. Your experiments show that the tested update process fails to produce cumulative gains—not that no improving direction exists.

“**Operational plateau**,” “**self-training attractor**,” or “**protocol-relative fixed point**” is therefore more precise than an unqualified “local optimum.”

---

## 4. Why verified search succeeds where training does not

Search mostly does **not escape the proposal distribution’s support**. It escapes the **one-sample bottleneck** and avoids the assimilation bottleneck.

For a task with success probability \(p\),

\[
P(\text{at least one success in }N\text{ draws})
=
1-(1-p)^N.
\]

When \(p\) is small, additional draws can deliver large gains over pass@1. The verifier then identifies the useful candidate. For graded rewards, selecting the maximum similarly exploits the upper tail of the existing quality distribution.

Search has three advantages:

1. **Direct use of discoveries.** The successful candidate is the output; it need not be compressed into weights first.
2. **Task-specific allocation.** It explores candidates for the current problem, rather than making a shared update that must work across many problems.
3. **No necessary permanent contraction.** If search leaves the proposal policy unchanged, selecting one winner does not itself make alternative modes less likely next round.

Training instead pays for data generation, verification, optimization, and potentially interference. The matched-compute result indicates that these costs are not repaid sufficiently in your evaluated setting.

The precise conclusion is:

> **Verified search accesses useful existing tail capability more efficiently than the tested training loop consolidates it.**

It is not evidence that search can solve genuinely zero-probability tasks. Nor does it settle lifetime economics: a successful training update could amortize over enough future queries. That would require a separate deployment-horizon comparison.

---

## 5. What would be required to break the plateau?

There are **two different breakouts**:

- **Accumulation without new support:** retain and reliably express more of the successes already discoverable.
- **Frontier expansion:** make new useful strategies operationally reachable.

The first alone could produce lineage-over-reset gains. The second is necessary once the exploitable frontier is exhausted.

A productive recursive loop needs some combination of:

\[
\text{new useful discoveries}
+
\text{effective assimilation}
+
\text{retention}
+
\text{remaining task headroom}.
\]

Increasing exploration alone is insufficient if discoveries are not learned; increasing learning alone is insufficient if every round supplies the same examples.

| Intervention | Mechanism | Prediction suggested by your data | Important limitation |
|---|---|---|---|
| **Exploration bonus / entropy preservation** | Preserve low-probability alternatives and broaden sampling. | Largest benefit where viable alternatives exist and contraction is binding; should increase new strategy discovery and delay archive saturation. | Entropy can increase invalid outputs. It does not by itself solve formation or credit assignment, and cannot create support under a strictly support-preserving update. |
| **External coverage injection** | Introduce stronger-model solutions, human kernels, repair trajectories, synthesis outputs, or solver-generated candidates. | Strongest direct test of discovery limitation: previously undiscovered tasks should become trainable; inherited gains should emerge if assimilation and retention work. | Gains demonstrate externally assisted learning, not closed-loop autonomous RSI. Failure after injection would implicate assimilation rather than only exploration. |
| **Novelty pressure** | Select nonredundant successful strategies rather than repeatedly selecting the modal winner. | Should increase accepted-set strategy diversity, sustain discovery across rounds, and improve lineage relative to reset if collapse is causal. | Novelty must be useful and verified; syntactic difference is not enough. |
| **Curriculum / prerequisite shaping** | Make the next learning step reachable before demanding the final solution. | Formation-oriented objectives should help small models; correctness and performance curricula should become more useful as failures move downstream. | Training only on easy tasks can sharpen the existing repertoire without transferring to the frontier. |
| **Off-policy RL with diverse replay** | Learn from a broader behavior distribution and preserve historical discoveries. | Should improve retention and help if valuable samples exist outside the current policy’s frequently sampled modes. | Off-policy RL is not intrinsically exploratory. A buffer containing only old redundant winners supplies no new frontier information. Near-zero target probabilities also create coverage and estimation difficulties. |
| **Search-amplified distillation / expert iteration** | Use a stronger search procedure to generate genuinely better or more diverse targets, then distill them. | Promising given search’s compute advantage—provided each round’s expert improves over the current apprentice rather than rediscovering the same winners. | Search superiority alone does not guarantee successful distillation or recursive amplification. |
| **Higher-headroom task distribution** | Move evaluation beyond the saturated portion of the current benchmark. | If ceiling compression drives the large-model decline, harder but reachable tasks should restore a productive frontier. | A harder task set may simply move most mass back into the undiscovered region. |

### The most informative next tests

1. **Coverage injection × diversity-preserving training.**  
   A factorial design separates lack of discoveries from loss of discoveries after learning.

2. **Frontier-conditioned lineage effects.**  
   Stratify by baseline pass probability, formation rate, and remaining roofline headroom. The proposed mechanism predicts the largest gains in the reachable, nonsaturated middle.

3. **Discovery–assimilation–retention accounting.**  
   Track:
   - first verified discovery of a strategy;
   - subsequent probability assigned to it;
   - transfer to related tasks;
   - survival over later rounds.

   This would turn the aggregate null into a directly observed failure pathway.

4. **Matched useful diversity intervention.**  
   Hold quality and training volume constant while varying behavioral diversity. This is the cleanest test of whether collapse locks the plateau.

---

## 6. Literature framing

These are the most useful anchors, with the distinctions made explicit:

- **STaR — Zelikman et al. (2022), *STaR: Bootstrapping Reasoning With Reasoning*.**  
  Establishes iterative learning from successful model-generated reasoning. Its answer-conditioned rationalization step matters: it changes the proposal process using known answers, rather than merely conditioning the original unconditional proposal on success.

- **ReST — Gulcehre et al. (2023), *Reinforced Self-Training for Language Modeling*.**  
  The grow/filter/improve structure is directly relevant to your rejection-sampling lineage. Your result concerns when repeating that structure does—or does not—produce additional inherited gains.

- **ReST\(^{EM}\) — Singh et al. (2024), *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models*.**  
  Provides an expectation-maximization framing of verified self-training. Your data highlight the need to distinguish policy sharpening from continued expansion of the useful training distribution.

- **Expert iteration — Anthony, Tian, and Barber (2017), *Thinking Fast and Slow with Deep Learning and Tree Search*.**  
  Separates an improving search-based expert from a learned apprentice. Your search advantage suggests the expert remains stronger than the distilled policy, while the lineage null suggests that the feedback loop does not amplify that advantage cumulatively.

- **RLVR capability-boundary debate — Yue et al. (2025), *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?***  
  Relevant empirical framing for increased sampling efficiency versus expanded reasoning coverage. Use it as a related hypothesis and measurement debate, not as a universal theorem that RL cannot expand capability.

- **Recursive-data collapse — Shumailov et al. (2024), *AI Models Collapse When Trained on Recursively Generated Data*.**  
  Useful for the intuition that recursive resampling can lose distributional tails. Your verifier-selected, reward-directed setting is different; the paper motivates a possible mechanism rather than proving it applies here.

The conceptual synthesis is **expert iteration without sufficient expert amplification, combined with incomplete retention of the apprentice’s useful tail**.

---

## 7. Claims your data do not yet support

I would explicitly avoid claiming:

- **RSI is impossible in general.** The result is about these protocols, budgets, tasks, and measured outcomes.
- **All scales exhibit established equivalence.** The pooled and 0.5–3B results are strong; the 7B estimate is still sparse and 14B is pending.
- **Undiscovered tasks have exactly zero success probability.**
- **Neural self-training cannot create new behaviors through generalization.**
- **Diversity collapse causes the null.** It is a plausible, testable contributor.
- **LoRA capacity or late-layer adaptation is the binding cause.**
- **Large-scale failure is entirely roofline saturation.** This needs baseline-headroom-conditioned analysis.
- **Formation is the only small-model deficiency.**
- **Off-policy RL or an exploration bonus would necessarily restore compounding.**
- **Search crosses an otherwise impossible capability boundary.**
- **The observed plateau is a genuine local optimum of the reward landscape.**

Also keep the lineage experiment, weight-level cohort, and Task-5 evidence distinct. They triangulate an explanation, but do not together constitute a single randomized mediation analysis.

---

## Two Discussion paragraphs to adapt

Across the tested rejection-sampling SFT regimes, our results indicate a failure of cumulative capability acquisition rather than an absence of useful sampled solutions. Lineage training was equivalent to resetting within the prespecified margin (pooled difference \(0.001\), 95% CI \([-0.015,0.018]\)), while verified search outperformed lineage training at matched compute. Together with the large pass@\(K\)/pass@1 gap, this suggests that useful behavior often exists in the proposal distribution’s tail but is not converted into a progressively expanding, retained repertoire. The scale dependence further localizes this bottleneck: small models are constrained primarily by kernel formation, intermediate models contain the largest productive band of occasionally successful tasks, and larger models exhibit substantial weight movement with limited additional gain, consistent with reduced headroom and increasingly redundant adaptation. This pattern accords with the distinction between sharpening existing successes and expanding the effective solution frontier in verified self-training (STaR; ReST; ReST\(^{EM}\)), and with expert iteration’s requirement that search generate improvements that the learner can both assimilate and preserve.

We interpret this behavior as a protocol-relative self-training plateau, not a demonstrated local optimum of the neural parameter space. Under an idealized rejection-conditioning update with exact taskwise fitting and no cross-task generalization, exact solution-support coverage is invariant: tasks with positive success probability can be sharpened, but tasks with zero probability supply no successful targets. Actual neural SFT need not satisfy these assumptions, and finite-budget nondiscovery does not establish zero probability. Nevertheless, declining diversity, greater retention among compounders, and rapid strategy-archive saturation are consistent with a reinforcing loop in which repeated training favors familiar winners while reducing opportunities for subsequent discovery. Verified search avoids the immediate consolidation bottleneck by selecting successful tail samples directly. Breaking the plateau should therefore require either more effective accumulation of already-discovered capabilities or interventions that expand the useful proposal distribution—such as diverse search-generated targets, external coverage injection, or prerequisite curricula—together with mechanisms that preserve prior gains. Whether diversity preservation or off-policy learning causally restores compounding remains an open experimental question.