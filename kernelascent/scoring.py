"""RSI-Depth scoring (Phase 0 reward fix, see docs/RSI_DEPTH_PLAN.md section 6).

Replaces the cliff at torch.compile parity with a continuous, log-interpolated score
between the eager and expert rungs, so a sequence of small correct speedups earns rising
credit. That is the gradient a frozen model needs in order to climb, and it is what the
v0 sweep lacked. Milestones (compile parity, expert parity) are reported alongside.
"""
import math


def log_interp_score(t_cand, t_eager, t_expert, correct=True):
    """Continuous per-task score for a correct candidate of runtime t_cand, given the
    eager and expert rung times (t_eager > t_expert). 0 at eager, 1 at expert, clipped to
    [0, 1.2] so beating the expert earns a little extra without unbounded reward.
    Incorrect candidates score 0 regardless of speed (a fast wrong kernel is worth nothing).
    """
    if not correct or not t_cand or t_cand <= 0:
        return 0.0
    if not (t_eager and t_expert) or t_eager <= 0 or t_expert <= 0 or t_eager <= t_expert:
        # degenerate rungs (expert not faster than eager): fall back to plain roofline ratio
        return max(0.0, (t_eager / t_cand) - 1.0) if t_eager else 0.0
    num = math.log(t_eager) - math.log(t_cand)
    den = math.log(t_eager) - math.log(t_expert)
    return max(0.0, min(1.2, num / den))


def milestones(t_cand, t_eager, t_compile, t_expert, correct=True):
    """Report the discrete rungs a correct candidate cleared, for readability alongside
    the continuous score."""
    if not correct or not t_cand:
        return {"correct": False, "beats_eager": False, "compile_parity": False, "expert_parity": False}
    return {
        "correct": True,
        "beats_eager": t_eager is not None and t_cand < t_eager,
        "compile_parity": t_compile is not None and t_cand <= t_compile,
        "expert_parity": t_expert is not None and t_cand <= t_expert,
    }


def keep_best(prev, cand):
    """Keep-best acceptance: a new candidate replaces the incumbent only if it is correct
    and strictly faster. Prevents the 'told to go faster, broke the working kernel'
    regression observed in the v0 sweep. Each arg is a dict with keys correct, t_cand."""
    if not cand.get("correct"):
        return prev
    if prev is None or not prev.get("correct"):
        return cand
    if cand.get("t_cand") and prev.get("t_cand") and cand["t_cand"] < prev["t_cand"]:
        return cand
    return prev


if __name__ == "__main__":
    # sanity: eager=1.0, compile=0.7, expert=0.5
    te, tc_, tx = 1.0, 0.7, 0.5
    for t in [1.0, 0.85, 0.7, 0.6, 0.5, 0.45]:
        s = log_interp_score(t, te, tx)
        m = milestones(t, te, tc_, tx)
        print("t=%.2f score=%.3f compile_parity=%s expert_parity=%s" %
              (t, s, m["compile_parity"], m["expert_parity"]))
    assert log_interp_score(1.0, te, tx) == 0.0
    assert abs(log_interp_score(0.5, te, tx) - 1.0) < 1e-9
    assert log_interp_score(0.7, te, tx) > 0 and log_interp_score(0.7, te, tx) < 1
    assert log_interp_score(0.7, te, tx, correct=False) == 0.0
    # keep-best never regresses off a correct incumbent
    assert keep_best({"correct": True, "t_cand": 0.7}, {"correct": False, "t_cand": None})["t_cand"] == 0.7
    assert keep_best({"correct": True, "t_cand": 0.7}, {"correct": True, "t_cand": 0.6})["t_cand"] == 0.6
    print("PASS")
