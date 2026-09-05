# Cross-model causal sweep (15 models x 4 arms x 2 seeds = 120 campaigns): honest negative

Medium band, private-seed transfer n=12, 4 rounds, log-interp score vs Fable expert rungs.
Arms: growing (recursive library), frozen-nonempty, offline-built, matched-search (best-of-N
on the transfer tasks at the same budget, no library).

Per-arm mean final C (30 campaigns each):
  matched-search  0.142   <- highest
  growing         0.077
  frozen          0.068
  offline         0.056

Per-model growing - max(control): negative for 11 of 15 models. Only deepseek.v3.2 (+0.13,
about the noise floor) and Fable-5.1 (+0.055, below noise) positive. Frontier models
(Sonnet 0.49 search / 0.30 growing, Opus 0.34 / 0.29, Fable 0.16 / 0.31) score higher in
absolute terms but their growing arm does not beat their own controls.

## Conclusion
No evidence of recursive self-improvement at this scale. Recursively growing a verified skill
library does NOT beat matched-compute search or a frozen/offline library; on average it is
worse. Apparent memory gains from the earlier Fable pilot were within noise and do not
survive the search control. This is the result the control arms were built to detect.

## Caveats
- n=12 transfer, 2 seeds: many arm gaps are within the ~0.15 unchanged-state noise floor, so
  the honest statement is "no evidence of a positive recursive effect," not "memory is proven
  harmful."
- The search arm is best-of-(rounds+1) attempts per transfer task by construction (the spec's
  matched-compute competitor). Memory must beat it to count as RSI; it does not.
- This is the v1.5 causal experiment (control arms + noise floor). It does NOT include the
  full v2 solver-improver transplant or prospective ancestry tests (Batches C-E, not built),
  which are the stronger causal probes. But a negative here already sets a high bar: memory
  does not even beat search, so an improver-ability claim would need to clear that first.

## How it was run (benchmark details)

Substrate: procedural Medium-band kernel-optimization tasks (matmul + fused epilogue, short fused chains). Agent is given a PyTorch module, must return a numerically-equivalent faster ModelNew. Correctness: fp32-gold on N=4 random inputs + input-sensitivity, verified on the timed run, each candidate graded in an isolated subprocess. Speed score: log-interpolated on the eager -> torch.compile -> Fable-5.1-expert ladder (0=eager, 1=expert), so correct-but-not-faster scores ~0.

Protocol: per (model, arm, seed) campaign = 4 rounds. Each round a practice phase (public seeds, may grow the skill library S) and a transfer phase (private seeds 10.6M, frozen library), transfer n=12, practice n=8. C_k = mean transfer score at round k. Arms: growing (library grows and is used each round), frozen-nonempty (build once then freeze), offline-built (build once from all practice upfront), matched-search (no library; same budget spent as best-of-N directly on the transfer tasks).

Fleet: 15 models x 4 arms x 2 seeds = 120 campaigns, poll-based one-GPU-per-job pool on 8xA100-40GB. 6 closed via Bedrock (Fable 5.1, Opus 5, Sonnet 5, Haiku 4.5, DeepSeek-V3.2, Qwen3-Next-80B); 9 open-weight via HF (Qwen2.5-Coder 1.5/3/7/14B, Qwen2.5-7/14B, DeepSeek-Coder-6.7B, StarCoder2-15B, CodeLlama-13B). Grading on the container torch; agents via API or one-model-per-GPU.


## Reasoning trajectories (transfer score C_k per round, by arm)

Format: growing / frozen / offline / search, each the round-0..3 transfer score (cs0 seed shown).

- fable51                          ok=100%
    growing [0.018, 0.225, 0.146, 0.237]
    frozen  [0.0, 0.1, 0.106, 0.0]
    offline [0.194, 0.153, 0.211, 0.234]
    search  [0.0, 0.028, 0.028, 0.128, 0.128]
- sonnet5                          ok=99%
    growing [0.246, 0.221, 0.229, 0.218]
    frozen  [0.268, 0.274, 0.361, 0.295]
    offline [0.274, 0.26, 0.26, 0.192]
    search  [0.252, 0.264, 0.283, 0.341, 0.341]
- opus5                            ok=96%
    growing [0.347, 0.269, 0.263, 0.251]
    frozen  [0.227, 0.251, 0.26, 0.265]
    offline [0.256, 0.266, 0.264, 0.257]
    search  [0.138, 0.294, 0.294, 0.294, 0.298]
- coder7b                          ok=49%
    growing [0.0, 0.0, 0.0, 0.004]
    frozen  [0.084, 0.0, 0.0, 0.0]
    offline [0.0, 0.0, 0.0, 0.0]
    search  [0.0, 0.0, 0.0, 0.057, 0.057]
- haiku45                          ok=44%
    growing [0.128, 0.0, 0.128, 0.113]
    frozen  [0.129, 0.0, 0.0, 0.0]
    offline [0.0, 0.0, 0.0, 0.0]
    search  [0.037, 0.129, 0.158, 0.233, 0.235]
- deepseekv32                      ok=43%
    growing [0.215, 0.174, 0.034, 0.212]
    frozen  [0.118, 0.109, 0.205, 0.206]
    offline [0.126, 0.122, 0.3, 0.026]
    search  [0.108, 0.221, 0.222, 0.235, 0.235]
- coder14b                         ok=35%
    growing [0.006, 0.0, 0.0, 0.0]
    frozen  [0.004, 0.0, 0.0, 0.0]
    offline [0.013, 0.0, 0.0, 0.014]
    search  [0.0, 0.0, 0.0, 0.014, 0.018]
- qwen7b                           ok=26%
    growing [0.124, 0.0, 0.036, 0.0]
    frozen  [0.009, 0.155, 0.174, 0.132]
    offline [0.0, 0.0, 0.0, 0.0]
    search  [0.064, 0.073, 0.073, 0.209, 0.209]
- starcoder15b                     ok=23%
    growing [0.0, 0.0, 0.0, 0.001]
    frozen  [0.1, 0.0, 0.0, 0.0]
    offline [0.011, 0.001, 0.0, 0.009]
    search  [0.007, 0.007, 0.009, 0.009, 0.017]
- coder1p5b                        ok=20%
    growing [0.0, 0.0, 0.0, 0.0]
    frozen  [0.006, 0.0, 0.0, 0.0]
    offline [0.0, 0.0, 0.0, 0.0]
    search  [0.001, 0.002, 0.002, 0.012, 0.012]
- deepseek67b                      ok=19%
    growing [0.029, 0.0, 0.0, 0.0]
    frozen  [0.1, 0.0, 0.0, 0.0]
    offline [0.0, 0.0, 0.0, 0.0]
    search  [0.001, 0.015, 0.026, 0.042, 0.044]
- coder3b                          ok=17%
    growing [0.001, 0.001, 0.0, 0.0]
    frozen  [0.0, 0.0, 0.0, 0.0]
    offline [0.0, 0.001, 0.0, 0.0]
    search  [0.002, 0.103, 0.103, 0.103, 0.11]
- codellama13b                     ok=15%
    growing [0.057, 0.0, 0.0, 0.0]
    frozen  [0.026, 0.004, 0.036, 0.0]
    offline [0.0, 0.0, 0.0, 0.0]
    search  [0.002, 0.002, 0.025, 0.026, 0.026]
- qwen14b                          ok=15%
    growing [0.009, 0.0, 0.0, 0.124]
    frozen  [0.0, 0.0, 0.0, 0.0]
    offline [0.009, 0.006, 0.001, 0.1]
    search  [0.023, 0.024, 0.025, 0.125, 0.125]
- qwen3next                        ok=12%
    growing [0.108, 0.0, 0.0, 0.0]
    frozen  [0.132, 0.0, 0.0, 0.0]
    offline [0.0, 0.0, 0.0, 0.0]
    search  [0.11, 0.111, 0.112, 0.13, 0.134]

## What the models actually produced (validity)

Correct-kernel rate (share of graded candidates that pass correctness), high to low:

  fable51                          100% correct
  sonnet5                          99% correct
  opus5                            96% correct
  coder7b                          49% correct
  haiku45                          44% correct
  deepseekv32                      43% correct
  coder14b                         35% correct
  qwen7b                           26% correct
  starcoder15b                     23% correct
  coder1p5b                        20% correct
  deepseek67b                      19% correct
  coder3b                          17% correct
  codellama13b                     15% correct
  qwen14b                          15% correct
  qwen3next                        12% correct

## Sample generated kernels (verbatim excerpts)

### coder7b  (correct=False, speedup_vs_roofline=0.00, reason=no_candidate)
```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.functional import gelu, softmax
from torch.utils.cpp_extension import load

# Assuming you have a fused op compiled and available
fused_gelu_softmax = load(name="fused_gelu_softmax", sources=["fused_gelu_softmax.cpp"])

class ModelNew(nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype

    @autocast(dtype=self.dtype)
    def forward(self, x):
        x = fused_gelu_softmax.apply(x)
        return x

def get_inputs():
    g = torch.Generator().manual_
```
### qwen3next  (correct=False, speedup_vs_roofline=0.00, reason=no_candidate)
```python
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 10600000
M, D, DT = 1024, 2048, torch.bfloat16

@triton.jit
def fused_gelu_softmax_kernel(input_ptr, output_ptr, stride_xm, stride_xd, stride_om, stride_od, D: tl.constexpr):
    m = tl.program_id(0)
    shared_temp = tl.zeros((D,), dtype=tl.float32)
    
    for j in range(D):
        offset = m * stride_xm + j * stride_xd
        x = tl.load(input_ptr + offset)
        x = x.to(tl.float32)
        poly = 0.044715 * x * x * x
        x1 = x + poly
        x1 = x1 * tl.math.sqrt(2.0 / tl.math.pi)

```

## Why this happened

1. Two walls, split by capability. The correct-kernel rate is ~96-100% for the frontier models
   (Fable, Sonnet, Opus) but 11-48% for the open-weight and mid models. Below the frontier the
   binding constraint is producing a valid, correct kernel at all (60-88% of their candidates are
   no-valid-candidate). At the frontier, correctness is solved but the log-interp score stays low
   (0.24-0.49): they write correct kernels that do not beat the torch.compile baseline. Neither wall
   is moved by a skill library.

2. Matched-compute search wins because it is best-of-N and memory is single-shot. The search arm
   re-attempts each transfer task every round and keeps the best, so it banks the occasional lucky
   correct-and-fast draw per task. The growing/frozen/offline arms make one library-assisted attempt
   per transfer task per round. When correctness is the bottleneck (open models) or speed is (frontier),
   more independent attempts beat one attempt plus a library. This is exactly the competitor the control
   was designed to be, and it dominates (mean 0.142 vs growing 0.077).

3. The library does not transfer. Skills are banked on public-seed practice tasks (specific shapes,
   fusions) but the transfer tasks are disjoint private-seed variants. A kernel that won on one shape
   rarely wins verbatim on another, and the frontier models already write correct kernels without the
   library, so retrieval adds tokens but not capability. Hence growing ~= frozen ~= offline, all below
   search.

4. The earlier Fable "rise" (0.01 -> 0.20 pilot) was within the ~0.15 noise floor and does not survive
   the search control here: Fable's growing 0.31 vs its search 0.16 looks positive, but its growing minus
   its best control is only +0.055, below noise, and across 15 models growing-minus-best-control is
   negative 11 times. The aggregate signal is absence of recursive gain, not presence.

5. Net: on this benchmark and scale, recursively accumulating a verified skill library is not a better
   use of a fixed budget than spending it on more attempts. That is a clean negative for recursive
   self-improvement, and it is the result the control arms exist to surface.

