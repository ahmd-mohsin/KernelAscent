# KernelAscent

**A benchmark of agents that improve with experience.** It asks the sharp question — not "can an agent
solve a task?", but: when an agent improves the *research procedure* it uses, does *using* that improvement
causally produce a *better next* improvement, and does it *compound*? Effects are estimated with 95% CIs
against a prespecified meaningful effect δ = 0.05; every headline separates the axes rather than collapsing
them into one "RSI score".

Site: https://ahmd-mohsin.github.io/KernelAscent/ · Full running record + every number: [`BENCHMARK_LOG.md`](BENCHMARK_LOG.md)

## Three axes (reported separately)

1. **Capability** — can the agent produce a correct/fast solution.
2. **First-order improvement** — does one self-revision of the procedure raise the quality of what it then
   produces (`q₁−q₀`, and `N` = does the child beat the unchanged target).
3. **Causal recursion** — `F = Q(U₂) − Q(V₂)`: a newer producer and an older producer edit the **same**
   target; recursion means the newer wins, and `F₂` asks whether that **repeats** (compounding). The
   actor/target separation is what turns "the score went up" into a causal claim; a rising task score or
   editable source is not enough.

## Substrates

- **GPU-kernel track** — procedural kernel-optimization tasks; crash-isolated grader; correctness vs
  fp32-gold and speedup vs `min(eager, torch.compile)`. Powers the capability leaderboard. Two walls:
  correctness ranks the low/mid band, beating `torch.compile` is the frontier discriminator.
- **Code / RSI track** — pure-Python, executable-graded. A difficulty-graded bug-fix bank (capability +
  local-verifier selection), and an **open-ended research engine** where the improvable state is a growing
  **archive** of reusable, debugged abstractions (library learning). This is where the recursion axis lives.

## Headline findings (honest)

- **Capability is a clean, capability-graded gradient** (E0, 22 models): GPT-5.6 tops both walls; open
  ≤14B models sit at the correctness wall with fast-rate 0.
- **First-order self-improvement is real and capability-graded — including a negative regime.** On the
  load-bearing open-ended substrate (live models, 12 lineages, 95% CI): the strongest model improves its
  procedure (Opus-5 `q₁−q₀` = **+0.083 [0.014, 0.153]**, resolved > 0) while weaker models are *hurt* by
  self-revision (Gemma-3-12B −0.039, Llama-3.1-8B −0.207, both resolved < 0) — weak models poison their own
  library with buggy abstractions.
- **Causal recursion is substrate-dependent, and this is the core result.** In a *closed* procedure space
  (knob-tuning) `F₂ ≈ 0` is essentially structural — gains front-load into the first step. In an
  *open-ended archive* (each improvement creates new improvement opportunities) the **instrument resolves
  compounding**: scripted calibration gives `F₁ = +0.010 [0.006, 0.014]`, `F₂ = +0.028 [0.023, 0.033]`
  (accelerating), while a frozen-archive+growing-memory control is exactly 0 (memory ≠ recursion). **No
  live model yet shows resolved positive `F₂`** — even on the open-ended substrate — so *true RSI is not
  demonstrated*; the benchmark is the instrument that reads ~0 until it happens and >0 when it does.
- **The load-bearing constraint** (why live compounding is hard): if a strong model can re-derive a helper
  inline, an optional archive adds nothing. Compounding needs the archive to be load-bearing — components
  hard-to-write-correctly but easy-to-reuse (edge-trap, or too-long-to-re-derive). GPU kernels, where
  frontier genuinely fails, are the natural home for the next step.

## Datasets

Two-split design (public committed + on HuggingFace; a held-out split powers the leaderboard and is never
released).

- **Code bug-fix bank** — 75 public tasks (easy 21 / medium 17 / hard 18 / ultra 19), 14–16 distinct
  problem families per tier, curated by Fable 5.1 and executable-validated:
  [`dataset/tasks/public/`](dataset/tasks/public) · HF `muahmed7338/kernelascent-tasks`.
- **Open-ended RSI task bank** — hard list-of-int synthesis tasks that *share* number-theory/digit
  primitives so a reused verified helper can compound: [`dataset/rsi_tasks/`](dataset/rsi_tasks).
- **GPU-kernel bank** — 1000+ procedural tasks, 6 families: [`dataset/curated/`](dataset/curated).

```python
from datasets import load_dataset
ds = load_dataset("muahmed7338/kernelascent-tasks")   # easy / medium / hard / ultra
```

## Evaluate a model (dockerized)

One command → a submittable scorecard; the leaderboard is scored on a private held-out split.

```bash
docker build -t ka -f docker/Dockerfile .

# capability (API model, no GPU):
docker run --rm -e AWS_SHARED_CREDENTIALS_FILE=/creds -e AWS_PROFILE=bedrock \
  -v $PWD/creds:/creds:ro -v $PWD/out:/out ka \
  --track capability --api-model us.anthropic.claude-opus-5 --tier medium

# recursion (open-ended library-learning loop):
docker run --rm ... ka --track rsi --api-model us.anthropic.claude-opus-5 --lineages 12
```

Then open a [model-submission issue](https://github.com/ahmd-mohsin/KernelAscent/issues/new?template=model-submission.yml).
Self-check (no model/GPU/creds): `docker run --rm --entrypoint bash ka docker/entrypoint_selfcheck.sh`.

## Repo layout

| path | what |
|---|---|
| `kernelascent/v3/core.py` | Q/V/F/N estimators + lineage runner (actor/target separation); deterministic calibrations |
| `kernelascent/v3/lab_open.py` | scripted open-ended substrate (compounding calibration: F₂>0) |
| `kernelascent/v3/lab_open_live.py` | live-model open-ended library-learning loop (load-bearing archive) |
| `kernelascent/v3/rsi_verify.py` · `rsi_true.py` | verifier-improvement + verifier-improves-verifier substrates |
| `kernelascent/v3/lab_engine.py` · `depth_probe.py` · `proposal_judge.py` | front-loading / headroom / coadaptation diagnostics |
| `kernelascent/evaluate.py` | single scoring entrypoint (capability / rsi tracks) → scorecard.json |
| `dataset/` | code, RSI, and kernel task banks + build/publish pipeline |
| `docker/` | dockerized standard evaluation image |
| `docs/` | website (black-and-white, mechanism + findings figures + leaderboards) |
| `BENCHMARK_LOG.md` · `docs/RSI_*` | full dated record, diagnosis, and adopted plan |

## Status

Validated + reproducible: capability leaderboard (22 models), calibrated causal estimators, the
open-ended compounding calibration, the load-bearing live RSI leaderboard, and the dockerized evaluator.
Open: eliciting resolved *live* compounding — the next experiment is a load-bearing GPU-kernel archive
(frontier genuinely fails there; verified fast kernels are hard to re-derive).
