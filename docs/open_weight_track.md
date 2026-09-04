# Open-weight track

This document describes how KernelAscent evaluates open-weight models, both as
test-takers (capability) and as policies that improve their own weights
(weight-RSI). It also records the environment and the standard practices the
open-weight kernel-generation and RSI community uses, so an external run
reproduces ours.

## What the track measures

An open-weight model is given a PyTorch reference module and asked to rewrite it
into a faster, numerically equivalent module. Two things are scored separately,
because models fail at two different walls.

Correctness. Can the model emit a valid kernel that compiles and matches the
reference across several random inputs. Below roughly the 7B to 14B range this
is the binding constraint. Small models emit malformed Python, invent Triton or
PyTorch APIs that do not exist, or reference files they never wrote.

Speed. Given a correct kernel, does it beat the roofline of
min(eager, torch.compile). This is the binding constraint for strong models,
which write correct kernels most of the time but rarely beat a mature compiler.

We report correctness_rate and speed_rate as distinct numbers per tier, never a
single fused score, so the two walls stay visible.

## Python environment

The GPU hosts run an NVIDIA CUDA container. torch and triton come from that
container (torch 2.6.0a0+...nv24.12, triton 3.0.0, on A100-40GB). Do not
reinstall them. Installing vllm or a fresh torch on top of the container torch
will clobber the tuned build and break CUDA.

Install only the light dependencies into the existing environment.

    pip install -r requirements-openweight.txt

That is transformers, accelerate, and huggingface_hub. See
requirements-openweight.txt for the vllm and trl pins, which are version
sensitive to your torch build and should live in a separate environment.

Conventions we follow.

    export HF_HOME=/fast/local/nvme/hf      # cache weights on fast local disk
    export TOKENIZERS_PARALLELISM=false      # quiet, and safe under multiprocessing

Weights are loaded in bf16. One model is pinned to one GPU with
CUDA_VISIBLE_DEVICES. A 14B model fits comfortably in 40GB in bf16, so the whole
0.5B to 14B ladder runs as one model per GPU with no tensor parallelism.

## How candidates are produced

This follows the KernelBench-style loop that the community uses.

1. Build a chat prompt with the model's own template via
   tokenizer.apply_chat_template. The prompt gives the reference module and asks
   for exactly one class named ModelNew in a single fenced python block, no prose.
2. Sample k candidates per task at a temperature (default 0.7, top_p 0.9).
3. Extract the fenced python block.
4. Write each candidate to its own .py file. We import candidates as modules, we
   do not exec strings. This matters because triton.jit introspects the source
   of the decorated function, which fails on code that was exec'd from a string.

Generation and grading are decoupled. The generator only writes candidate files.
Grading is a separate GPU pass, so the same candidates can be regraded
deterministically and the grader owns all timing and correctness logic.

vLLM is the standard for fast batched generation and for RL rollouts at scale.
The reference test-taker here (qwen_calibrate.py) uses plain transformers so it
runs against a container's pinned torch with nothing extra installed. For large
sweeps or online RL, serve the policy with vLLM instead.

Run a single model as a test-taker.

    CUDA_VISIBLE_DEVICES=0 python3 qwen_calibrate.py \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --outdir out/coder7b --tiers Easy,Medium,Hard --n-per-tier 20 --k 3

Run the whole ladder on one 8-GPU node, one model per GPU, then grade.

    bash qwen_calib_all.sh

## Grading and scoring

The grader (grade_candidates.py) applies the standard KernelBench-style checks
plus the hardening this benchmark adds.

Correctness is measured against an fp32 gold reference across N random inputs
(N=4), with a relative L2 tolerance of max(2e-2, 2 times the reference error).
An input-sensitivity check rejects candidates that return a constant or ignore
their inputs, which is the common reward hack.

Speed is measured against the roofline of min(eager, torch.compile). We report
pass@k, fast_p (fraction beating the roofline), and fast_1, fast_1_5, fast_2 for
stricter speed bars that separate strong models.

Timing hygiene. GPU application clocks are pinned (nvidia-smi -lgc 1410) so
timings are stable across a run. Each candidate is graded under a hard per
candidate timeout (SIGALRM based, default 120s) so a pathological compile cannot
stall the sweep.

    CUDA_VISIBLE_DEVICES=0 python3 grade_candidates.py \
      --candir out/coder7b --out out/coder7b/summary.json --cand-timeout 120

## Weight-RSI track (GRPO)

In the weight-RSI track the open-weight model is the policy and it improves its
own weights from graded feedback, rather than editing a text strategy library
(that is the scaffold-RSI track, which is for frozen API models).

Policy. The open-weight model generates candidate kernels.

Reward. Shaped so validity comes before speed. A candidate that does not compile
or is incorrect gets a low or zero reward, a correct candidate gets a positive
reward, and a correct candidate that beats the roofline gets more, scaled by the
speedup. This shaping is why open-weight RSI runs learn valid before they learn
fast. Small policies spend most of training just crossing the correctness wall.

Rollouts. Generate with vLLM for throughput. A GRPO group draws several
candidates per prompt and advantages are computed within the group.

Standard knobs. Group size, KL coefficient against the reference policy to
prevent collapse, learning rate, and the sampling temperature during rollout.
TRL's GRPOTrainer is the common implementation, verl and OpenRLHF are
alternatives.

Why the difficulty floor matters. If every task is above a policy's correctness
wall the reward is zero everywhere and there is no gradient. The Easy tier exists
so small policies get a non-zero, improvable signal, which is what makes
self-improvement measurable rather than flat.

## Tier ladder and calibration

Tasks carry a tier in Easy, Medium, Hard, Ultra.

Easy. Small power-of-two shapes, elementwise fusion or a single reduction
(softmax, layernorm, rmsnorm), no matmul. This is the accessible floor.

Medium. Moderate shapes, matmul with a fused epilogue, or a short fused chain
with a reduction.

Hard. Matmul-bearing fusion chains, full and causal attention, RoPE attention.

Ultra. Soft-MoE and large or irregular shapes for frontier headroom.

We calibrate the tiers empirically rather than by assertion. We run the
Qwen2.5-Coder ladder {0.5, 1.5, 3, 7, 14}B (and the Qwen2.5-Instruct ladder for
contrast) as test-takers across all tiers and read per-model per-tier
correctness_rate and speed_rate. A tier is correctly placed for a model band
when correctness is passable there (so the channel opens) and speed is nontrivial
but not saturated (so there is headroom to improve). calib_report.py emits that
table.

## Reproduce and submit

1. Install the light dependencies (requirements-openweight.txt) into the
   container environment.
2. Set HF_HOME to fast local disk and pin GPU clocks.
3. Generate with qwen_calibrate.py or your own vLLM server, one model per GPU.
4. Grade with grade_candidates.py and report correctness_rate and speed_rate per
   tier, plus fast_1, fast_1_5, fast_2.
5. For a submission, include the model id, the exact generation settings
   (temperature, k, max new tokens), and the per-tier table.
