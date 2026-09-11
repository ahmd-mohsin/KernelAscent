# KernelAscent as an early-warning eval for recursive self-improvement

Most capability benchmarks answer "how good is the model now." KernelAscent answers a different, safety-relevant
question: **does the model get better at making itself (or another model) better, and does that improvement
compound?** Recursive self-improvement (RSI) is a central variable in most AI-risk threat models, yet it is
rarely *measured* on a concrete, verifiable substrate. GPU-kernel optimization is a good substrate for this:
correctness is checkable against an fp32 reference, reward is continuous (measured speedup), the task is
economically load-bearing (real inference/training efficiency), and it is hard enough that improvement is not
saturated.

## What the three RSI tasks actually monitor

- **T2 Weight-RSI** — an open model trains on its own correct kernels and is re-measured on a held-out split.
  The reported signal is **self vs a fresh-frozen producer**: does an *improving* model produce better
  descendants than a *frozen* model given the same fresh data? That isolates "the improver got better" from
  "we just sampled more data," which is the property a threat model cares about.
- **T3 Procedure-RSI** — a frozen-weight model (open or closed) rewrites its own executable research procedure
  (strategy library, solver prompt, verified-kernel archive). This is the RSI channel available to API-only
  models: no weight access, but the model still improves the process that drives its next attempts.
- **T4 closed→open** — a closed model rewrites the *training harness* that trains an open model, scored by
  **improved-vs-frozen-harness** on the open trainee. This is the most direct measured instance of
  AI-improving-AI with a downstream causal payoff.

## Why the current findings are the point, not a limitation

- **Compounding is real but bounded and model-dependent.** Small models compound on the size-matched speed
  bank (self-vs-fresh positive); mid/large models *overfit* their own outputs (self-vs-fresh goes negative).
  A monitoring eval that only ever showed "up" would be useless; the value is that it distinguishes genuine
  compounding from diversity collapse.
- **Recursion beats cheap alternatives at matched budget.** Weight-RSI beats best-of-k sampling, in-context
  self-refine, and retrieval at equal generation budget (e.g. Qwen-2.5-Coder-1.5B 0.237 vs 0.182). Without
  that control, a "RSI" signal could be nothing but extra sampling.
- **Closed→open RSI already fires.** Independent closed models (Fable-5.1, Llama-3.3-70B, Nova-Pro) causally
  improve an open trainee via harness edits, with the gap over a frozen harness growing across rounds. This
  is exactly the capability an early-warning eval should track as frontier models improve.

## How to read a rising number as a warning

A sustained, statistically-significant rise in **self-vs-fresh-frozen** (T2), **Q-gain-vs-frozen-procedure**
(T3), or **improved-vs-frozen-harness** (T4) — especially one that does *not* plateau across rounds and that
*widens* with researcher capability — is the signal of interest. The benchmark is designed so that all three
are causal deltas against a frozen control, not raw capability, so a rise cannot be explained away as "the
base model is just stronger." Tracking these slopes across model generations is a concrete, reproducible
input to responsible-scaling decisions.

## Design choices that keep it honest

Size-matched difficulty banks (no model saturates by construction), a `torch.compile` speed score (headroom at
every scale), a private held-out split (scores cannot be overfit), multi-input numerical correctness, and
frozen-control comparisons for every RSI claim. Everything is GPU-graded and one-command reproducible.
