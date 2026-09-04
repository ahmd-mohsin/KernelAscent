# Phase 0 exit result (reward fix landed)

Setup. New 3-node fleet. Fable 5.1 reconstructed the Medium-band expert rungs (37/39
graded correct, all 37 beat torch.compile, so the expert rung is legitimate and t_expert
is real). The fixed reward loop (rsi_campaign.py) ran on four models on the Medium band,
n=8 tasks, 5 rounds, keep-best acceptance, log-interpolated score (0 = eager, 1 = expert).

Trajectory (mean score per round, 0..1.2 scale):

    model                        fresh per round            best-so-far
    Qwen2.5-7B-Instruct          0.30 0 0 0 0               0.30 0.30 0.30 0.30 0.30
    Qwen2.5-Coder-14B-Instruct   0 0 0 0 0.08               0 0 0 0 0.08
    Qwen2.5-Coder-7B-Instruct    0 0 0 0 0                  0 0 0 0 0
    Qwen2.5-Coder-3B-Instruct    0 0 0 0 0                  0 0 0 0 0

## Findings

1. The reward fix works and the exit criterion is met. best-so-far is monotone
   non-decreasing for every model. Keep-best plus the log-interpolated slope removed the
   destructive "told to go faster, broke the working kernel, scored 0" dynamic that
   dominated the v0 sweep. The reward is now climbable rather than a cliff.

2. There is still no RSI slope at this level. best-so-far is flat; the shaped feedback
   with an explicit expert-time gap does not make fresh attempts improve. The limiter is
   generation skill, not the reward shape. Absolute scores are low because these models'
   correct kernels mostly sit at eager parity, which scores about 0 on the eager-to-expert
   ladder; Coder-7B did not beat eager on any of the 8 Medium tasks.

## Why this is the expected baseline, not a failure

This campaign is effectively the L0 / per-task-refinement level of the depth ladder: no
cross-task memory persists, so there is no channel by which capability could compound. A
flat L0 slope is exactly the control the ladder is built against. The RSI-Depth thesis is
that compounding should appear at L2, where a solved practice task banks a verified,
reusable skill (a fused kernel, a tiled epilogue) that makes later transfer tasks easier.
That persistence channel does not exist here.

Caveat: n=8 is small and noisy; this is a pilot to validate the reward fix, not a powered
measurement.

## Next

Build the L2 campaign (skill/code-block memory with a transfer set), per RSI_DEPTH_PLAN
sections 5 and 2. That is the first level where the plan predicts a non-flat slope, and it
is the real test of whether kernel-optimization capability can bootstrap from verified
self-built artifacts without weight training.
