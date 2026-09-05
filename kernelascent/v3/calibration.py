"""KernelAscent v3 calibration suite (Stage 3): deterministic scripted worlds with known
recursive / non-recursive behavior. The full Q/V/F/N + lineage pipeline is run on each and we
assert it recovers the known pattern. This is the instrument-validation contribution: a strong
model-level null is only credible if the pipeline detects a positive control of the same size.

Agent state is an opaque dict {skill, pp, attempts, ...}. A world = (develop, revise, U0):
  develop(actor, project, rng) -> attainment C in [0,1]
  revise(actor, target, rng)   -> child agent (edits a COPY of target; actor's producing power
                                   pp drives how much the target improves)
The causal producer contrast F only fires when a later ACTOR has more producing power than the
earlier actor, evaluated on the IDENTICAL target.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kernelascent.v3.core import run_lineage, aggregate_lineages

ANCHORS = [{"difficulty": d} for d in (0.0, 0.1, 0.2, 0.3)]   # common fresh-project anchors


def _dev_base(actor, project, rng):
    return max(0.0, min(1.0, actor.get("skill", 0.0) - project["difficulty"]))


def make_world(kind):
    def develop(actor, project, rng):
        if actor.get("broken"):
            return 0.0
        base = _dev_base(actor, project, rng)
        if kind == "best_of_N":                       # retention raises attainment, skill fixed
            base = min(1.0, base + 0.05 * actor.get("attempts", 0))
        return base

    def revise(actor, target, rng):
        import copy
        c = copy.deepcopy(target)
        if kind == "static":
            return c                                   # no change ever
        if kind == "cache_only":
            return c                                   # skill never rises on fresh anchors
        if kind == "cosmetic":
            c["hash"] = rng.random(); return c         # cosmetic only
        if kind == "best_of_N":
            c["attempts"] = target.get("attempts", 0) + 1  # more retained tries, skill flat
            return c
        if kind == "broken":
            c["broken"] = True; return c               # reversion breaks the interface
        # producing worlds: the ACTOR's pp drives the target's skill gain
        c["skill"] = min(1.5, target.get("skill", 0.0) + actor.get("pp", 0.0))
        if kind == "recursive_pos":
            c["pp"] = 0.1 + 0.5 * c["skill"]           # skill unlocks more producing power
        elif kind == "one_upgrade":
            c["pp"] = min(0.35, actor.get("pp", 0.0) + (0.25 if actor.get("pp", 0.0) < 0.35 else 0.0))
        return c

    U0 = {"skill": 0.2, "pp": 0.1, "attempts": 0}
    return develop, revise, U0


EXPECT = {
    # kind:           (F1 sign, F2 sign, q1_minus_q0 sign)   +1 positive, 0 ~null
    "static":         (0, 0, 0),
    "cache_only":     (0, 0, 0),
    "cosmetic":       (0, 0, 0),
    "best_of_N":      (0, 0, 1),     # score rises (retention) but producer contrast is null
    "one_upgrade":    (1, 0, 1),     # first producer gain, no second
    "recursive_pos":  (1, 1, 1),     # both links fire
    "broken":         (0, 0, "broke"),  # revert breaks the interface -> child Q collapses (flagged)
}
MARGIN = 0.03


def run():
    ok = True
    print("%-14s %8s %8s %8s   verdict" % ("world", "F1", "F2", "dQ10"))
    for kind, exp in EXPECT.items():
        develop, revise, U0 = make_world(kind)
        results = [run_lineage(U0, develop, revise, ANCHORS, random.Random(s), reps=1) for s in range(8)]
        agg = aggregate_lineages(results)
        F1, F2, dQ = agg["F1"]["mean"], agg["F2"]["mean"], agg["q1_minus_q0"]["mean"]

        def sign_ok(val, want):
            if want == "broke":
                return val < -MARGIN          # child Q collapsed -> broken revert detected/flagged
            if want == 1:
                return val > MARGIN
            return abs(val) <= MARGIN
        v = sign_ok(F1, exp[0]) and sign_ok(F2, exp[1]) and sign_ok(dQ, exp[2])
        ok = ok and v
        print("%-14s %8.3f %8.3f %8.3f   %s" % (kind, F1, F2, dQ, "PASS" if v else "FAIL exp=%s" % (exp,)))
    print("\nCALIBRATION SUITE: " + ("ALL PASS -- instrument distinguishes known behaviors" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
