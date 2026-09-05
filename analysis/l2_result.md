# L2 skill-memory result (first rising slope, controls pending)

Setup. L2 campaign (verified skill/code-block library, RSI_DEPTH_PLAN section 5). Practice
on public-seed Medium tasks banks correct, faster-than-eager kernels as reusable skills;
the disjoint private-seed transfer set (seeds 10.6M) is solved with the frozen current
library each round. C_k is the mean log-interpolated transfer score (0 = eager, 1 = expert)
at round k, so C_0 is the empty-library baseline. Expert rungs reconstructed with Fable 5.1
(112 of 134 Medium tasks have an expert that beats torch.compile). practice_n = transfer_n
= 8, 4 rounds.

Trajectories.

    agent            C_0    C_1    C_2    C_3    skills banked
    Fable 5.1        0.01   0.175  0.203  0.20   17
    Opus 5           0.201  0.153  0.159  0.163  21
    Coder-14B        0.001  0.0    0.16   0.025  5
    Coder-7B         0.0    0.15   0.0    0.0    0

Read.

Fable 5.1 shows the first rising L2 slope. Transfer climbs from about 0.01 with an empty
library to about 0.20 as the library grows, then plateaus. This is the RSI-at-L2 shape:
skills banked on practice tasks lifting held-out transfer performance.

Opus 5 is ceiling limited. C_0 is already 0.20 with no library, and the library does not
lift it further. Plateau, the same shape the frontier showed in the scaffold run.

Open weight cannot bootstrap at L2. Coder-7B banked zero skills (its kernels never beat
eager, so nothing was worth banking) and Coder-14B is flat and noisy. The skill-banking
floor is above these models on this band.

So L2 self-improvement appears capability windowed, not merely capability gated. It needs
an agent strong enough to bank correct fast skills and with enough headroom left to climb.
Fable sits in that window here, Opus is above it (no headroom), the open models are below
it (cannot bank).

Caveats and what is not yet established. n=8 transfer is noisy. The attribution controls
required by the plan have not run yet, so this is a promising signal and not a confirmed
RSI claim. Needed next, all on Fable 5.1 and ideally a second mid-capability model:

  frozen-library control  a static round-1 library reused every round. If C_k matches the
                          growing library, then having a library matters but growing it does
                          not, which is not RSI.
  poisoned-library control  plausible but wrong or subtly broken skills. If C_k matches the
                          real library, the lift is a placebo.
  matched-compute L0      the same total attempts spent as best-of-N with no library. Rules
                          out that the lift is just more sampling.
  larger n and a second seed  to turn the slope estimate into one with a confidence interval.

Only if the growing real library beats all three controls on held-out tasks does this become
an RSI claim.
