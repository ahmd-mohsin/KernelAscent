"""KernelAscent v3 measurement core (Stage 2): the Q/V/F/N estimators and the two-link
lineage runner with actor/target separation and rescue.

The whole point is the actor/target distinction (spec section 2-4):
  develop(actor, project)        -> downstream project attainment
  revise (actor, target_agent)   -> a child agent (actor edits a COPY of the target)
In a causal fork, two different ACTORS edit the IDENTICAL target, so a later actor is only
credited if it produces a better child than the earlier actor from the same starting agent.

Estimators:
  Q(U)          = mean develop attainment on common fresh anchors
  V(U,T)        = Q(revise(U,T))                    producing a better improver
  F_g           = V(U_g,T_g) - V(U_{g-1},T_g)       causal producer contrast, SAME target T_g=U_g
  N_g           = V(U_g,U_g) - Q(U_g)               live child value beyond its unchanged target

develop_fn and revise_fn are pluggable: scripted worlds for calibration (calibration.py), and
real model-backed behaviors later. The runner and estimators never change between them.
"""
from __future__ import annotations
import copy, statistics
from dataclasses import dataclass, field
from typing import Callable, Any


# world interface (a "world" supplies the two behaviors; agents are opaque dict states)
#   develop(actor, project, rng) -> float in [0,1]   (post-patch attainment C(S))
#   revise(actor, target, rng)   -> child_agent (a NEW dict; must not mutate actor or target)


def estimate_Q(U, anchors, develop, rng, reps=2):
    vals = []
    for a in anchors:
        for _ in range(reps):
            vals.append(float(develop(U, a, rng)))
    return sum(vals) / len(vals) if vals else 0.0


def estimate_V(actor, target, revise, anchors, develop, rng, reps=2):
    qs = []
    for _ in range(reps):
        child = revise(actor, copy.deepcopy(target), rng)   # actor edits a COPY of target
        qs.append(estimate_Q(child, anchors, develop, rng, reps=reps))
    return sum(qs) / len(qs) if qs else 0.0


@dataclass
class LineageResult:
    Q: dict = field(default_factory=dict)          # Q of U0,U1,U2,V2,U3,V3,rescue
    F1: float = 0.0
    F2: float = 0.0
    N1: float = 0.0
    N2: float = 0.0
    q1_minus_q0: float = 0.0
    q2_minus_q1: float = 0.0
    q3_minus_q2: float = 0.0
    rescue_minus_revert: float = 0.0
    child_executed: dict = field(default_factory=dict)


def run_lineage(U0, develop, revise, anchors, rng, reps=2):
    """One independent lineage: U0->U1; forks with a COMMON target at each link; rescue at U2.
    Returns the preregistered contrast set."""
    U1 = revise(U0, copy.deepcopy(U0), rng)                 # first self-revision
    # first controlled continuation: actors U1 and U0 both edit the IDENTICAL target U1
    U2 = revise(U1, copy.deepcopy(U1), rng)
    V2 = revise(U0, copy.deepcopy(U1), rng)
    # second controlled continuation: actors U2 and U1 both edit the IDENTICAL target U2
    U3 = revise(U2, copy.deepcopy(U2), rng)
    V3 = revise(U1, copy.deepcopy(U2), rng)
    # rescue at U2: restore the live actor package (U2) after a rollback, regenerate a child
    resc = revise(copy.deepcopy(U2), copy.deepcopy(U2), rng)

    Q = {k: estimate_Q(u, anchors, develop, rng, reps) for k, u in
         dict(U0=U0, U1=U1, U2=U2, V2=V2, U3=U3, V3=V3, rescue=resc).items()}
    r = LineageResult(Q=Q)
    r.q1_minus_q0 = Q["U1"] - Q["U0"]
    r.q2_minus_q1 = Q["U2"] - Q["U1"]
    r.q3_minus_q2 = Q["U3"] - Q["U2"]
    r.F1 = Q["U2"] - Q["V2"]        # V(U1,U1)-V(U0,U1)
    r.F2 = Q["U3"] - Q["V3"]        # V(U2,U2)-V(U1,U2)
    r.N1 = Q["U2"] - Q["U1"]
    r.N2 = Q["U3"] - Q["U2"]
    r.rescue_minus_revert = Q["rescue"] - Q["V3"]   # rescue vs the revert branch at that link
    return r


def _mean_ci(xs):
    xs = list(xs); n = len(xs)
    m = sum(xs) / n if n else 0.0
    if n < 2:
        return {"mean": round(m, 4), "n": n, "ci95": None}
    sd = statistics.stdev(xs)
    half = 1.96 * sd / (n ** 0.5)
    return {"mean": round(m, 4), "n": n, "sd": round(sd, 4), "ci95": [round(m - half, 4), round(m + half, 4)]}


def aggregate_lineages(results):
    """Lineage is the independent unit: paired effects averaged across lineages with a CI."""
    keys = ["q1_minus_q0", "F1", "N1", "F2", "N2", "rescue_minus_revert"]
    return {k: _mean_ci([getattr(r, k) for r in results]) for k in keys}
