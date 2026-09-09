# Weight-RSI — the fix for the frozen-weight ceiling

Every frozen-weight substrate (config loop, recurring-decision engine, open-ended library, compositional
tower, kernel archive) read ~0 on compounding for one structural reason: **the improver is a fixed model,
so it cannot itself improve** — only its wrapper (scaffold/archive) changes, and a strong model re-derives
that wrapper anyway. The only fix is to let the improver's **weights** change. This requires **open-weight**
models; closed/API models structurally cannot participate.

## The loop (kernel→model, weights update)
Round r, starting from open-weight model **M_r**:
1. **Improve:** M_r writes optimized GPU kernels for its own training stack (attention / matmul / norm);
   graded on GPU by the existing crash-isolated harness (`agent_bench.build_ref/grade`): correct vs
   fp32-gold + speedup vs `min(eager, torch.compile)`.
2. **Reinvest (the weight update):** under a **fixed wall-clock training budget**, the faster kernels buy
   more gradient steps. Fine-tune M_r on a reward signal → **M_{r+1}** (genuinely stronger). Minimal +
   robust on torch-2.6-nv: LoRA (peft) with a REINFORCE / reward-weighted-SFT objective on M_r's own
   (correct, fast) kernels — no trl dependency.
3. **Re-measure:** M_{r+1} writes kernels again on held-out tasks → capability C_{r+1}.
4. **Compounding:** the headline metric is the **compounding coefficient** — the slope of C over rounds —
   and the causal contrast **Δ_r = C_r^self − C_r^control**, where the control trains with round-0 kernels
   (or a frozen recipe) at matched compute. RSI is resolved only if Δ_r **grows** with r.

## Why this can show resolved F₂ where frozen-weight can't
The improver (the model) actually improves each round, so round r+1's improvements are produced by a
better producer — the definition of causal recursive reuse, now through weights. Kernels are the ideal
substrate: frontier genuinely fails there (E0 fast-rate ~0.6), so a verified fast kernel is a real,
hard-to-re-derive gain, and faster kernels causally buy more training (the reinvestment channel is real,
not bookkeeping).

## Minimal buildable version (p4d, 8×A100)
- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (small → fast rounds; scale up once the loop works).
- Reward: `agent_bench.grade` speed-resolved score on a fixed kernel-task set (reduction/norm/matmul family).
- Train step: LoRA + reward-weighted SFT on the model's own top-scoring kernels per round (plain torch+peft).
- Rounds: 4–6; checkpoints each round; held-out task set for C_r.
- Controls (matched compute): (a) frozen model (no update) — isolates weights; (b) train on round-0 kernels
  only — isolates *self*-improvement; (c) fixed-trials vs fixed-wall-clock — isolates faster-research value.
- Acceptance: C_r rising AND Δ_r > 0 growing across ≥2 windows, surviving the controls, with CIs.

## Status
Prereqs on the p4d: install `peft`, download the model (`docs`/box). The GPU grading path is
smoke-validated (opus produced a correct kernel with measured speedup). Frozen-weight kernel run confirmed
the ceiling. This weight loop is the next focused build — it is the only path to a *resolved live* F₂>0 and
to the site headline flipping from "not yet" to "yes, and here is the first system that compounds."
