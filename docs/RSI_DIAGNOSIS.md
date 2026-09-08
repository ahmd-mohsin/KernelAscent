# Why causal compounding (F₂) is zero — diagnosis & requirements for real RSI

**Status:** diagnostic report (2026-09-08). Summarizes *why* the benchmark measures no recursive
compounding, separates artifact from finding, and states the concrete requirements a loop must meet for
F₂ to be a live, non-trivial number. Decision on the path forward is left to the user.

Notation: `U` = an agent's executed research procedure. `develop(U)` → quality `Q`. `revise(actor,target)`
→ a child procedure. `q₁−q₀` = first-order improvement. `N` = child beats unchanged target. `F₁` = a newer
producer beats an older one on a **common** target (causal link). `F₂` = that link **repeats** (compounding).
Prespecified meaningful effect `δ = 0.05`.

---

## 1. What the data says

| axis | result | status |
|---|---|---|
| capability | clean gradient (GPT-5.6 top; opus/sonnet ≫ open) | resolved, moves with model |
| first-order improvement `q₁−q₀` | opus +0.30 > gpt-oss +0.10 > deepseek +0.02 | resolved, capability-graded |
| child value `N₁` | opus +0.043 [0.007,0.080], gpt-oss +0.092 [0.011,0.172] | resolved > 0 |
| transfer | +0.091 [0.037,0.146] to a structurally different family | resolved > 0 |
| **causal link `F₁`** | opus +0.019 [−0.013,0.051] at n=40 | **not resolved, < δ** |
| **compounding `F₂`** | ≈ 0 on all four substrates | **≈ 0** |

So the benchmark is **not** globally zero — four axes move and rank models. Only the causal/compounding
axes (`F₁`, `F₂`) are flat.

---

## 2. What is ruled OUT (the zero is real, not broken)

- **Not an instrument artifact.** Deterministic calibration fixtures show the estimator reads `F>0`
  when compounding is truly injected, and distinguishes a one-upgrade ladder (`F₁>0, F₂≈0`) from a
  compounding one (`F₁,F₂>0`).
- **Not a ceiling / no-headroom.** The first curated run's zero *was* a ceiling artifact (Q pinned at 1.0)
  and was fixed; the labs now run at Q₀ ≈ 0.5–0.8 with measured procedure headroom of 0.12–0.18.
- **Not merely underpower.** `F₁` at n=16 looked positive (+0.042) and **regressed to +0.019 at n=40** —
  it was small-sample noise. Even the *first* link is small; the second is ≈ 0 at tight CIs.

---

## 3. The real root causes

### 3.1 Improvements are FRONT-LOADED / one-shot  ← the central issue the user flagged
The first improvement captures almost all the available gain (turn on caching, or guided search, or edge
coverage). After it, the procedure sits in the "good basin" and every later move is marginal — so there is
**nothing left for a second link to compound**. Confirmed directly: in every substrate `q₁−q₀ ≫ q₂−q₁`,
and after U₁ the neighbor procedures have little quality spread.

Part of this is *genuine* (many real efficiency wins are one-shot: once you cache, you can't cache again),
and part is a *design limitation* (our improvement ladders are shallow: ~2 meaningful rungs). **If every
improvement is one-shot, F₂ is zero by construction — this is the thing to fix.**

### 3.2 The improver does not get better at IMPROVING (frozen-weight)
`F₁ > 0` requires the *newer producer* to be a better producer than the older one — not merely to hold a
better procedure. But the thing authoring the edits is a **fixed-weight model**; only its inherited
context/params differ. A frozen model does not become a better *improver* just because it inherited a
better procedure. **This is structural to the frozen-weight (API-agent) setting** and is likely the deepest
reason `F₁` stays below δ.

### 3.3 The compounding channel is a single interaction, not a ladder
We localized where compounding *could* live: the proposal × judgment **interaction** (+0.045). But it is
**one** interaction, not a deep chain of unlockings. A loop needs improvement N to *raise the marginal value
of* improvement N+1, repeatedly — a staircase, not a single step.

---

## 4. Capability ≠ compounding (why "future smarter models" won't fix it alone)

Capability and compounding are **orthogonal** in our data: GPT-5.6 tops raw capability, opus-5 shows the
strongest first-order improvement, yet both give `F₁,F₂ ≈ 0`. A better one-shot solver does not imply a
compounding self-improver. So scaling single-shot intelligence will not, by itself, push `F₂` up.

---

## 5. Requirements for a loop where F₂ CAN be non-zero

For the benchmark's compounding axis to be a *live, forecastable* number (not zero-by-construction), the
loop must satisfy **all** of:

1. **Non-front-loaded, deep improvement ladder.** A sequence of improvements each worth a comparable
   amount, where improvement N is *reachable/affordable only after* N−1 — so gains keep coming densely
   rather than saturating in one step. (Directly addresses §3.1.)
2. **An improver that itself improves.** The entity producing improvements must get *better at producing
   them* as a causal result of prior improvements. Two ways:
   - **Weight-updating loop** (open weights + training/GRPO): the model's *weights* change from its own
     improvements → the improver genuinely improves. Frozen-weight agents cannot do this. **(Untested;
     the highest-value missing regime.)** (Addresses §3.2.)
   - **Deep procedure/knowledge accumulation** where a better inherited procedure makes the *next*
     procedure-search materially easier — a real coadaptation staircase, not one interaction. (§3.3.)
3. **Headroom preserved across links.** Q must not saturate after link 1 (already handled by the graded
   ladder / continuous scoring), and the ceiling must stay above where link 2 would land.
4. **Adequate power at each link.** `F` effects are small; each link needs enough independent lineages to
   resolve an effect of size δ (variance-sized prospectively, not a fixed lineage count).
5. **Controls that survive.** Beat frozen-procedure-with-growing-memory (kills "memory ≠ recursion"),
   fixed-builder, and direct-search at matched total budget — else "compounding" is just spending compute.

---

## 6. Candidate mechanisms to build (for discussion — user to choose)

Ordered by how directly they attack §3:

- **A. Weight-RSI loop (attacks §3.2, the deepest cause).** Open-weight model writes kernels/tools that
  accelerate its *own* training under a fixed wall-clock; retrain; the improved model is the new improver;
  measure `F` across rounds vs a round-0-kernels control. This is the one regime where the improver truly
  improves. Cost: GPU + GRPO; the deferred track.
- **B. Deep coadaptation ladder (attacks §3.1 + §3.3).** Redesign the research procedure so proposal and
  evaluation are separately improvable AND each improvement *unlocks* the next (e.g. a transferable
  edge-pattern library where a better library both selects better and discovers still-better patterns).
  Verify the ladder is deep (≥3–4 rungs of comparable value) *before* claiming compounding. Cheaper than A;
  tests whether an agent (even frozen-weight) can climb a staircase we know exists.
- **C. Cumulative-knowledge loop (attacks §3.1).** The agent accumulates verified reusable artifacts
  (kernels, tools, lemmas) so later research starts from a richer base — but this must beat the
  frozen-procedure+growing-memory control, or it is "memory, not recursion."

---

## 7. The decision that determines the benchmark's worth

- If `F₂ ≈ 0` **because we only test frozen-weight single agents where compounding is impossible in
  principle** → measuring it is measuring a foregone conclusion. **Not acceptable.** Must build a regime
  (A/B) where `F₂` *can* move.
- If `F₂ ≈ 0` **even with a deep ladder + an improving improver + headroom + power, across model
  generations** → that is a *major* finding (RSI does not compound at this scale), not a wasted benchmark.

**Sell the benchmark today on the axes that already move** (capability, first-order improvement, transfer);
position `F₂` as the compounding-detector that reads "not yet" until a genuinely-compounding loop exists.
**The single highest-value next investment is a loop where the improver improves** — mechanism A (weight
updates) or a *verified-deep* mechanism B ladder.

---

## 8. Open questions for the user (to direct the path to real RSI)

1. Do we commit GPUs to the **weight-RSI loop (A)** — the only setting where the improver truly improves —
   or first exhaust the cheaper **deep-ladder (B)** to see if a frozen-weight agent can climb a staircase we
   engineer?
2. For B: what is a *faithful* deep ladder in a real domain (kernels / inference serving) — a concrete
   sequence of improvements where each genuinely unlocks the next, not a synthetic contrivance?
3. What is the acceptance bar for declaring compounding: how many resolved links (`F₁`, `F₂`, `F₃`?), over
   how many independent lineages, surviving which controls, before we call it RSI?
4. Is a rigorously-bounded negative (with A/B built) an acceptable headline if `F₂` stays 0 — or is the
   goal specifically to *find* the first positive?
