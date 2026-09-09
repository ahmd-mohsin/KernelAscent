# KernelAscent — tiers

Two settings; each has explicit tiers. A tier says *what a task demands*, not how big a model must be.

## Setting A — Capability (can the agent solve it?)

**Code bug-fix bank** (`dataset/tasks/`) — tiered by edge-subtlety / algorithmic difficulty:
| tier | what the tasks demand | example families |
|---|---|---|
| Easy | one basic list/string/number transform; few edges | dedupe, running-max, count-vowels |
| Medium | correctness hinges on 1–2 edge cases a plausible fix misses | merge-intervals, first-missing-positive |
| Hard | several interacting rules / a subtle spec | semver ordering, LRU-with-TTL, tensor strides |
| Ultra | competition-hard algorithm, many edges | aho-corasick, damerau-levenshtein, linear-diophantine |

**GPU-kernel bank** (`dataset/curated/`) — tiered by optimization primitive:
| tier | demands |
|---|---|
| L1 | memory / reduction fusions |
| L2 | tensor-core / matmul + epilogue |
| L3 | attention & architecture-specific (RoPE, MoE) |

## Setting B — Recursion / RSI (does improvement compound?)

Tiered by **how load-bearing the archive is** — i.e. how much a reused, verified abstraction is *required*
rather than optional. This is the axis that determines whether compounding can appear at all.

| tier | substrate | structure | what it measures | file |
|---|---|---|---|---|
| **R1 · shared-helper** | number-theory / digit synthesis tasks that share primitives (is_prime, factorize, digits, palindrome) | flat set; a reused verified helper is *useful* across tasks but a strong model can re-derive it | first-order improvement `q₁−q₀`, `N` — and the negative regime (weak models poison their library) | `dataset/rsi_tasks/`, `lab_open_live.py` |
| **R2 · compositional tower** | a depth-D ladder where rung *k* = layer *hₖ* ∘ rung *k−1* | dependency ladder; from-scratch correctness compounds down (~pᵏ) so deep rungs *require* a verified archived rung — the archive is **load-bearing by construction** | causal producer link `F₁` and **compounding `F₂`** — true RSI | `lab_tower.py` |
| **R2 sub-tiers** | — | shallow rungs 1–3 (one-upgrade regime) vs deep rungs 4–8 (compounding regime) | whether the link *repeats* up the tower | — |

**Why two recursion tiers.** R1 has headroom below the frontier and exposes first-order improvement, but a
strong model re-derives an optional helper, so `F₂ ≈ 0` for it. R2 makes reuse unavoidable (depth makes
inline re-derivation error-prone for *any* model), which is where compounding (`F₂ > 0`) can be measured —
the calibration confirms the R2 structure yields resolved `F₁, F₂ > 0` when the builder is competent.

## Splits (both settings)
Public split committed + on HuggingFace; a disjoint **held-out** split (never released) scores the
leaderboard so results cannot be overfit.
