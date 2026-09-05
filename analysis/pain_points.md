# Cross-spectrum analysis and pain points (small open -> large -> API)

Synthesis across three evidence sources: the 13-model capability calibration, the 15-model x
4-arm causal sweep, and the v3 two-link causal pilot (Fable, Coder-14B done; Opus running).

## The picture by capability band

Small open-weight (0.5B-3B: Qwen2.5-Coder 0.5/1.5/3B, etc.)
- Correct-kernel rate 11-33%; 60-88% of candidates are no-valid-candidate.
- In the sweep every arm ~0 except the occasional lucky search draw.
- develop attainment Q ~ 0, so revise has no signal to act on.
- PAIN 1 (capability floor): they cannot reliably emit a valid kernel, so there is nothing to
  bootstrap. RSI is unmeasurable; they contribute zeros and noise.

Mid open-weight (7-14B: Coder-7/14B, Qwen-7/14B, DeepSeek-6.7B, StarCoder2-15B, CodeLlama-13B)
- Correctness partial (7B ~48%, 14B ~35%); but they rarely beat the compile baseline.
- v3 pilot Coder-14B: q1-q0 = +0.219 CI[0.01,0.43] (one-shot self-improvement real) but
  F1 = +0.062 CI[-0.17,0.30], F2 = 0.000 (no recursive compounding).
- PAIN 2 (speed wall / bounded saturation): correct-but-slow pins C(S) at 0.5, rarely 1.0, so
  the improvable headroom is thin and the reward is sparse.
- PAIN 3 (no transfer): banked skills / param edits earned on public-seed shapes do not carry
  to disjoint private-seed projects.

API frontier (Fable 5.1, Opus 5, Sonnet 5)
- ~96-100% correct; higher absolute scores (0.24-0.49 in the sweep).
- v3 pilot Fable: q1-q0 = +0.344 CI[0.01,0.68] (one-shot self-improvement real), but
  F1 = -0.062 CI[-0.57,0.44], F2 = -0.250 (no recursive compounding). The one-upgrade
  signature, not the recursive-positive one.
- PAIN 4 (shallow self-edit space): the current revise only tunes two params (focus,
  retrieval_k). A parameter tweak cannot causally produce a genuinely better IMPROVER, so F is
  near zero almost by construction. This is the single biggest lever on whether recursion can
  appear at all.
- PAIN 5 (one-upgrade ceiling): the first self-revision captures most of the available gain;
  little is left for a second link, so F2 <= 0.

Cross-cutting
- PAIN 6 (search dominates): matched-compute best-of-N beat every library arm; memory has no
  budget advantage. RSI must show a mechanism search lacks (a compounding producer), which it
  did not.
- PAIN 7 (precision): n=4 lineages gives wide CIs; only q1-q0 clears 0. F/N are inconclusive,
  not confidently zero.

## What the pain points imply for the next experiments

1. Stop spending RSI budget on sub-4B models. They are below the floor (Q~0); keep them only
   in calibration. Run RSI on capable agents where q1-q0 is already positive (>=7B open + API).
2. Deepen the self-edit space (addresses PAIN 4, the biggest). Let revise edit the actor's
   TOOLS / PROMPT POLICY / procedure SOURCE (the executed-mutable-U full-source path validated
   in Batch A), not just two scalar params. Recursion needs something with room to compound.
3. Raise precision (PAIN 7): 8+ lineages per agent, so F has a usable interval.
4. Keep the common-target control, rescue, and the calibration suite green.
5. Widen C(S) headroom (PAIN 2): keep the eager/compile ladder but add partial credit toward an
   expert rung so correct-but-slow kernels have a gradient to climb rather than saturating at 0.5.

## Next experiment (informed)

v3 pilot-2 on capable agents only (Fable 5.1, Opus 5, Sonnet 5, Qwen2.5-Coder-14B), 8 lineages
each, with a RICHER revise that edits the actor's procedure source/tools (executed-mutable-U),
same two-link + rescue + common-target instrument, calibration already green. Question: does a
deeper self-edit space let any capable agent show F1>0 AND F2>0 (true recursion), or does the
one-upgrade ceiling hold even with room to compound? Either outcome is a clean, calibrated
result.
