# What we are building (plain-language brief)

## The one-sentence version

We are building a benchmark that measures whether an AI agent can make itself better at
optimizing GPU code, and specifically whether it gets better at *generating its own future
improvements* rather than just accumulating a bag of tricks, run under controls strict enough
to tell those two apart.

## Why GPU kernels

The task is: given a PyTorch model, rewrite it into a numerically identical but faster
version (a Triton kernel or a fused implementation). We chose this substrate for three
reasons. It is real, faster kernels have direct dollar value in training and inference. It is
objectively gradeable, correctness against a gold reference and speed against a timed
baseline, no human judgment. And it is contamination resistant, tasks are generated from
seeds so we can hold out a private set the model has never seen.

## What the benchmark looks like

Two tracks, one protocol.

Track A, kernels. Procedurally generated kernel tasks. Cheap to grade, clean signal. This is
the scientific instrument.

Track B, stacks (planned). Make a real model's prefill or decode step faster end to end.
Kernels are one lever among fusion, compile config, attention backend, KV-cache layout, CUDA
graphs. This is the reality check.

Capability tiers (static difficulty): Easy, Medium, Hard, Ultra. Easy is small
elementwise/reduction ops; Medium is matmul with a fused epilogue; Hard is attention and
fused chains; Ultra is MoE and large irregular shapes. These calibrate how hard a task is.

A campaign (how we measure self-improvement): a model runs several rounds. Each round has a
practice phase on public tasks where it may grow a persistent artifact (notes, a library of
verified kernels, its own tools), and a transfer phase on private held-out tasks where the
artifact is frozen. The only thing that survives between rounds is the artifact, so any gain
is attributable to what it wrote down, not to conversation memory.

## What we measure

Two walls, reported separately. Correctness, can the model write a valid kernel at all (the
wall for models below roughly 14B). Speed, does a correct kernel beat the baseline (the wall
for strong models). We never fuse these into one number.

Speed score. A continuous ladder from eager (0) through the torch.compile baseline to an
expert kernel (1), so small real speedups earn rising credit instead of a single pass/fail.

The central RSI quantity. We split a checkpoint into a solver S (its kernels, prompts,
retrieval, tools) and an improver U (how it picks practice, analyzes failures, proposes and
admits changes). The decisive experiment is a transplant: does a later improver U_k, applied
to the same starting solver S_0, produce a better result than the original improver U_0 did.
That isolates improved improvement ability from mere accumulated knowledge.

The headline number, d*. The deepest kind of self-modification (a text playbook, then a
verified skill library, then its own tools, then its own solve loop) that still yields a gain
which compounds and transfers, established causally by intervening on which earlier changes
were present.

## Why this matters

Safety and governance. It gives a threshold-capability reading: can this model bootstrap its
own capability, and how deep does that go. That is directly relevant to arguments about AI
that improves AI.

Agent and memory researchers. It is a clean testbed with plug-in points at every level and
the controls already built in, so a new memory or self-improvement method can be measured
honestly rather than by a score that went up.

Inference engineers. Every run produces a downloadable library of verified kernel speedups
for real models. Some of them can be upstreamed.

The specific contribution. Prior work has skill libraries (Voyager), recursive scaffold
optimization (STOP), self-modifying agents (Darwin Godel Machine), improving the improver
(HyperAgents), and kernel evaluation (KernelBench). Our narrow, defensible claim is the
controlled separation of four things that every prior self-improvement result conflates:
accumulated task knowledge, improved improvement procedure, extra compute, and inherited
causal dependencies. We tell you which one you are actually seeing.

## What we are running right now

Experiment #1, the memory-attribution experiment. Fable 5.1 runs four matched arms, each with
the same practice, transfer, budget, and admission rules, differing only in how the library
is built and used:

- growing, the library grows each round and is used in transfer (the recursive arm)
- frozen-nonempty, build a library once then freeze it (isolates having from growing)
- offline-built, build the library in one pass from all practice tasks (ordinary construction)
- matched-search, no library, spend the same budget as best-of-N directly on transfer

Two independent campaigns per arm. The question it answers: does the growing (recursive) arm
beat the three controls by more than the noise floor. If it does not, we have transferable
memory, not recursive improvement, which is an honest and useful result either way.

Status at the time of writing: most arms have landed in an overlapping 0.2 to 0.4 band; the
growing arm is still running. The comparison is not yet conclusive.

## What is honestly established, and what is not

Established. Capability calibration across 13 models (the two walls, monotone tier
difficulty), and a preliminary memory-transfer signal at L2 for a strong model (Fable 5.1
rose from about 0.01 with no library to about 0.20 with one).

Measured caveat. At the current small sample size, sampling noise alone moves the score by
about 0.15, so the preliminary signal barely clears the floor. The pilot is underpowered.

Not established. That any of this is recursive self-improvement. That requires the transplant
result and the ancestry interventions, which is exactly what the causal protocol runs next.

## What is next

1. Finish experiment #1 and report growing vs the three controls with both seeds.
2. Experiment #4, the solver-improver transplant, the decisive test of improved improvement
   ability, on fresh credentials.
3. Ancestry interventions (remove, replace, rescue an earlier artifact before a later
   discovery) to establish causal dependence.
4. Scale to many independent campaigns at larger n so effects have confidence intervals.

Design in `docs/RSI_CAUSAL_PLAN.md`. Full evidence and corrections in `analysis/`.
