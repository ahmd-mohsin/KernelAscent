<p align="center"><img src="docs/logo.png" alt="KernelAscent" width="90"/></p>

<h1 align="center">KernelAscent</h1>

<p align="center"><b>A benchmark for compounding kernel optimization.</b></p>

<p align="center">
Site: https://ahmd-mohsin.github.io/KernelAscent/ · Full record and every number: <a href="BENCHMARK_LOG.md"><code>BENCHMARK_LOG.md</code></a>
</p>

---

KernelAscent asks whether models get better at writing GPU kernels — and, crucially, whether that improvement **compounds recursively**. A model writes a kernel. We grade it for correctness against an fp32 reference and for speed against a baseline (eager today; a unified `torch.compile` baseline is reported separately). Scores are headroom-normalized against a per-task roofline so the ceiling is the *model's* skill, never the benchmark's.

The benchmark is a ladder of five tasks. Tasks 1–4 are building blocks; **Task 5 (self-play) is the true recursive-self-improvement metric.**

- **Task 1 — Capability.** Can a model write a correct, fast kernel in one shot. Open and closed models compete.
- **Task 2 — Weight-RSI.** An open-weight model LoRA-trains on its own correct kernels; does held-out capability keep rising? (weights are the improvement channel)
- **Task 3 — Procedure-RSI.** A model rewrites its own executable strategy library + verified archive; improvement without touching weights (works for closed models).
- **Task 4 — Closed→Open.** A closed frontier model rewrites the *training harness* of an open trainee; measures improvement transferred through tooling.
- **Task 5 — Self-play (true RSI).** The model authors its own strictly-harder tasks **and** solves+improves on them, so difficulty and capability co-evolve. This is the airtight recursive-compounding test. Runs for both open-weight (weight channel) and closed (procedure channel) models.

Capability is a snapshot of raw skill. Tasks 2–5 measure the *slope*, and Task 5 measures whether the slope feeds itself.

## Task 5 — self-play, the 3-arm design

Self-play conflates two effects: a harder curriculum, and an author that *co-evolves* with the solver. To separate them we run three arms from the same base at equal budget, scored each round on one fixed held-out ladder:

- **S — STATIC.** Fixed seed frontier; solver improves normally.
- **F — FROZEN-AUTHOR.** Frontier escalates, but the author is the frozen base — an adaptive curriculum from a non-evolving author. Solver still improves.
- **L — LIVE-AUTHOR.** Frontier escalates and the author is the *current, evolving* model. Only the author role differs from F.

The decomposition is the whole point: **L−S** = total curriculum benefit, **F−S** = benefit without updating the author, and **L−F = the benefit of author co-evolution** — the self-referential signal, and the primary metric. Every self-authored task passes an anti-reward-hacking gate (rejects constant-output, identity, no-op, or trivial-runtime tasks), is deduplicated, and its provenance is logged (model-proposed vs programmatic backstop). Sustained `L−F > 0` with `model_proposed > 0` is genuine recursive compounding; `L ≈ F` means "adaptive curriculum only."

## Mechanism analysis — *why* RSI fails or *how* it passes

A rising curve says *whether* a model compounds; the mechanism probe says *why*. Each round we log generation diversity (distinct-2, pairwise dissimilarity — mode collapse), predictive entropy, LoRA weight-drift split by transformer depth (→0 means the ceiling is reached), retention on already-solved tasks (catastrophic forgetting), and the train−held transfer gap (memorization vs generalization). A verdict attributes each outcome to a mechanism: `diversity_collapse | forgetting | drift_saturation | no_transfer | no_headroom`, or PASS plus the depth that carries the learning. This turns null results (e.g. sub-2B models that never emit a correct kernel) into findings rather than blanks.

## Grading

A candidate kernel is a `class ModelNew` module. The grader runs in a crash isolated subprocess, so a kernel that triggers a CUDA device assert cannot poison the harness.

- **Correct** when its output matches the fp32 gold within tolerance.
- **Fast** by speedup over the eager baseline today. A compiled baseline is being unified across tracks and reported as a separate column.
- **Score C** is 0 when wrong, 0.5 at eager parity, and rises to 1.0 at a 1.5x eager speedup.

Note on interpretation. Because 0.5 is eager parity and all correct kernels are admitted, a mid score can reflect learning to emit a reliably correct kernel rather than a faster one. We are separating correctness acquisition from speed optimization with a candidate audit and a compiled baseline. Treat the current board as a development result.

## Task 2, the RSI loop

Each round an open model writes kernels for a train split, gets LoRA trained on the ones that graded correct, and is re scored on a disjoint held out split. We report held out capability `C_r` each round, plus two controls.

- **Frozen base**, the same model with no training.
- **Round 0 data control**, retrained every round on the first round's kernels only.

RSI is supported when `C_r` rises and the self trained arm beats both controls. The round 0 control is the strict test. It separates genuine self improvement from simply doing more training.

**Difficulty standardization.** The RSI bank is filtered against the frozen base. We keep only tasks the base can sometimes solve but does not already run fast. This gives a low starting score with real room to climb, so a flat result reflects the model and not a saturated bank.

## Findings

- **Capability is a clean gradient.** Closed frontier models top both walls. Small open models sit at the correctness wall with low fast rate.
- **RSI is scale and family dependent.** On the standardized hard bank, small models start low and gain from self training. Several families keep pulling ahead of the fixed data control, which is genuine compounding. Very small models stay flat because they never produce a correct kernel to learn from. A strong 7B overfits, since it keeps solving the train tasks while its held out score falls.
- **The RSI score is a thermometer, not a cheerleader.** It reads near zero or negative when a model does not compound, and positive when it does.

The live leaderboard covers 19 open models across 9 families, including Qwen, DeepSeek, Yi, Phi, SmolLM, OpenCoder, StableCode, Llama, and Gemma.

## Datasets

Current design. The RSI bank ships publicly as a development split, and each run holds out a subset for scoring within that run. That held out subset is reconstructible from the public bank, so it is a development set, not a secret exam. A sealed evaluation set with structural and project level holdouts is being built to score the official board.

- **Capability kernel bank.** Tiered GPU kernel tasks. `dataset/kernel_bank/` and HF `muahmed7338/kernelascent-tasks`.
- **RSI hard bank.** Fable 5.1 curated, GPU validated, then difficulty filtered to the hard but learnable band. `dataset/kernel_bank/rsi_bank_hard.json`.

```python
from datasets import load_dataset
ds = load_dataset("muahmed7338/kernelascent-tasks")   # public split, by tier
```

## Evaluate a model

One command produces a submittable scorecard. Current results use a public development split. A sealed evaluation set is in progress.

```bash
docker build -t ka -f docker/Dockerfile .

# Task 1, Capability, open or closed
docker run --rm -e AWS_SHARED_CREDENTIALS_FILE=/creds -e AWS_PROFILE=bedrock \
  -v $PWD/creds:/creds:ro -v $PWD/out:/out ka \
  --track capability --api-model <model> --tier medium

# Task 2, RSI, open weight, GPU
docker run --rm --gpus all -v $PWD/out:/out ka \
  --track rsi --model <hf-id> --rounds 5 --k 3
```

Then open a [model submission issue](https://github.com/ahmd-mohsin/KernelAscent/issues/new?template=model-submission.yml) with your `scorecard.json`. Maintainers re run on the held out split and add your row. Automatic submission through GitHub is coming.

## Repo layout

| path | what |
|---|---|
| `kernelascent/v3/lab_weight_rsi.py` | Task 2 weight-RSI loop. Writes kernels, LoRA trains on correct ones, re scores held out |
| `kernelascent/v3/lab_track_c.py` | Task 3 procedure-RSI: model rewrites its own strategy library + verified archive |
| `kernelascent/v3/lab_selfplay_rsi.py` | Task 5a open self-play, 3-arm STATIC/FROZEN-AUTHOR/LIVE-AUTHOR, primary L−F |
| `kernelascent/v3/lab_selfplay_closed.py` | Task 5b closed (API) self-play, 3-arm, co-evolution via procedure |
| `kernelascent/v3/lab_rsi_mechanism.py` | why-RSI probe: diversity/entropy/drift-by-depth/retention/transfer + verdict |
| `kernelascent/v3/difficulty_filter.py` | difficulty standardization. Keeps only hard but learnable tasks against the frozen base |
| `kernelascent/v3/curate_kernel_tasks.py` | Fable 5.1 curator for the kernel banks, tiered L1 to L3, GPU validated |
| `kernelascent/v3/grade_batch.py` | crash isolated GPU grader, single and batch modes |
| `kernelascent/v3/lab_kernel.py` | kernel task loader and scoring |
| `kernelascent/agent_bench.py` | reference build, timing, and grading primitives |
| `kernelascent/evaluate.py` | single scoring entrypoint for both tracks |
| `dataset/kernel_bank/` | capability and RSI task banks |
| `docker/` | dockerized evaluation image |
| `docs/` | website, figures, and both leaderboards |
| `BENCHMARK_LOG.md` | full dated record and every number |

## Status

Working and reproducible. The capability leaderboard, the difficulty standardized RSI bank, the multi family RSI leaderboard, and the dockerized evaluator all run. The RSI board fills as runs complete.
