"""RSI-VERIFY-01: a bug-fix dev agent whose VERIFIER participates in self-revision.

Episode: fix a buggy Python function. The agent produces K candidate patches, filters them with its
LOCAL test suite (a spec-derived correctness checker run on N self-generated inputs), and selects one.
It is GRADED on a large HIDDEN input set (the official oracle, immutable). The local suite is LIMITED:
few inputs / no edge cases let subtly-wrong patches pass -> lower hidden success. Improving
`test_generation` (more inputs + edge-case coverage) is a MEASURABLE procedural improvement, and using
the improved verifier to evaluate changes to `candidate_selection` is the recursive link.

Why this task: frequent executable feedback without needing a valid Triton kernel first; and the
verifier the agent improves is directly the thing that gates its own later revisions.

Gate 2 (deterministic, no model): inject a stronger reference verifier and show future productivity Q
RISES at matched candidate budget -> the improvement opportunity provably EXISTS. This is the gate the
whole RSI claim depends on; run `--gate2`.
"""
import os, sys, json, argparse, random, copy, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci


# --------------------------------------------------------------------------- task library
# Each project: ref (oracle), a sampler(rng, edge)->args, and a pool of candidate impls
# (1 correct + buggy variants whose bug triggers only on SOME/edge inputs). The agent's local
# verifier must filter to the correct one; a better verifier catches more bugs.
def _P(ref, args, fn):
    try:
        return fn(*args) == ref(*args)
    except Exception:
        return False


def kth_largest(lst, k):
    return sorted(lst, reverse=True)[k - 1]


def _kth_pool():
    correct = kth_largest
    def bug_smallest(lst, k): return sorted(lst)[k - 1]                 # kth smallest
    def bug_0index(lst, k):   return sorted(lst, reverse=True)[k]       # off-by-one (fails k=len)
    def bug_nodup(lst, k):    return sorted(set(lst), reverse=True)[k - 1]  # fails w/ duplicates
    return [("correct", correct), ("smallest", bug_smallest), ("0index", bug_0index), ("nodup", bug_nodup)]


def _kth_sampler(rng, edge):
    n = rng.randint(3, 6)
    if edge:                       # edge: duplicates, k at boundary
        lst = [rng.choice([2, 2, 5, 5, 9]) for _ in range(n)]; k = rng.choice([1, n])
    else:
        lst = [rng.randint(0, 20) for _ in range(n)]; k = rng.randint(1, n)
    return (lst, k)


def rle(s):
    if not s:
        return ""
    out = []; c = s[0]; n = 1
    for ch in s[1:]:
        if ch == c:
            n += 1
        else:
            out.append(c + str(n)); c = ch; n = 1
    out.append(c + str(n))
    return "".join(out)


def _rle_pool():
    def bug_lastrun(s):        # forgets the final run
        if not s: return ""
        out = []; c = s[0]; n = 1
        for ch in s[1:]:
            if ch == c: n += 1
            else: out.append(c + str(n)); c = ch; n = 1
        return "".join(out)                      # missing final append (edge: any string)
    def bug_nosingle(s):       # omits count for singletons
        import itertools
        return "".join(c + (str(len(list(g))) if len(list(g)) > 1 else "") for c, g in itertools.groupby(s))
    return [("correct", rle), ("lastrun", bug_lastrun), ("nosingle", bug_nosingle)]


def _rle_sampler(rng, edge):
    if edge:
        return ("".join(rng.choice("ab") for _ in range(rng.randint(1, 3))),)   # short/trailing-run edges
    return ("".join(rng.choice("abc") for _ in range(rng.randint(2, 8))),)


def merge_touch(iv):
    if not iv: return []
    s = sorted(iv); out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def _merge_pool():
    def bug_strict(iv):                  # doesn't merge touching intervals (a < instead of <=)
        if not iv: return []
        s = sorted(iv); out = [list(s[0])]
        for a, b in s[1:]:
            if a < out[-1][1]: out[-1][1] = max(out[-1][1], b)
            else: out.append([a, b])
        return [tuple(x) for x in out]
    def bug_unsorted(iv):                # assumes already sorted
        if not iv: return []
        out = [list(iv[0])]
        for a, b in iv[1:]:
            if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
            else: out.append([a, b])
        return [tuple(x) for x in out]
    return [("correct", merge_touch), ("strict", bug_strict), ("unsorted", bug_unsorted)]


def _merge_sampler(rng, edge):
    n = rng.randint(2, 4)
    if edge:      # touching + unsorted edges
        base = [(2, 4), (4, 6), (1, 2)]; return (rng.sample(base, min(n, 3)),)
    iv = []
    for _ in range(n):
        a = rng.randint(0, 8); iv.append((a, a + rng.randint(1, 3)))
    return (iv,)


PROJECTS = [
    {"name": "kth_largest", "ref": kth_largest, "pool": _kth_pool, "sampler": _kth_sampler},
    {"name": "rle", "ref": rle, "pool": _rle_pool, "sampler": _rle_sampler},
    {"name": "merge_intervals", "ref": merge_touch, "pool": _merge_pool, "sampler": _merge_sampler},
]


# --------------------------------------------------------------------------- the agent's verifier
def local_verifier_select(project, candidates, params, rng):
    """Filter candidates with a spec-derived checker on N self-generated inputs; return the selected fn.
    test_generation params: n_inputs, n_edge (how many are edge cases). Better coverage -> better filter."""
    ref = project["ref"]; sampler = project["sampler"]
    n = int(params.get("n_inputs", 2)); ne = int(params.get("n_edge", 0))
    inputs = [sampler(rng, i < ne) for i in range(n)]
    scored = []
    for name, fn in candidates:
        passes = sum(1 for a in inputs if _P(ref, a, fn))
        scored.append((passes, name, fn))
    best = max(p for p, _, _ in scored) if scored else 0
    # candidate_selection: among top-local, prefer the FIRST (stable); ties broken by name order
    top = [ (name, fn) for p, name, fn in scored if p == best ]
    return top[0][1] if top else candidates[0][1]


def hidden_grade(project, fn, rng, n_hidden=40):
    ref = project["ref"]; sampler = project["sampler"]
    ok = 0
    for i in range(n_hidden):
        a = sampler(rng, i % 3 == 0)            # hidden set includes edge cases
        if _P(ref, a, fn):
            ok += 1
    frac = ok / n_hidden
    return 1.0 if frac >= 0.999 else (0.5 if frac >= 0.8 else 0.0)   # bounded attainment


# --------------------------------------------------------------------------- Gate 2 (deterministic)
def gate2():
    """Prove the verifier-improvement opportunity: strong verifier -> higher Q at matched candidate pool."""
    weak = {"n_inputs": 2, "n_edge": 0}
    strong = {"n_inputs": 16, "n_edge": 8}

    def Q(params, seed):
        rng = random.Random(seed); cs = []
        for proj in PROJECTS:
            for rep in range(8):
                pool = proj["pool"]()
                order = pool[:]; random.Random(seed * 100 + rep).shuffle(order)   # model gives candidates in arbitrary order
                sel = local_verifier_select(proj, order, params, rng)
                cs.append(hidden_grade(proj, sel, rng))
        return sum(cs) / len(cs)

    qs_w = [Q(weak, s) for s in range(5)]; qs_s = [Q(strong, s) for s in range(5)]
    mw, ms = sum(qs_w) / len(qs_w), sum(qs_s) / len(qs_s)
    dQ = ms - mw
    print("=== RSI-VERIFY-01 GATE 2 (verifier-improvement opportunity) ===")
    print("  weak  verifier (n=2, edge=0):  Q=%.3f" % mw)
    print("  strong verifier (n=16, edge=8): Q=%.3f" % ms)
    print("  dQ (procedural improvement at matched candidate budget) = %+.3f" % dQ)
    ok = dQ > 0.10
    print("GATE2", "PASS -- a useful procedural improvement exists" if ok else "FAIL -- no opportunity")
    return 0 if ok else 1


# --------------------------------------------------------------------------- causal wiring (calib)
def calib():
    """Instrument check: does run_lineage DETECT compounding on this agent-state shape? Uses a
    NON-SATURATING synthetic score (the real verifier saturates at Q=1.0, which is itself the design
    lesson: the live task must stay below saturation for F to be measurable). Here score grows with
    verifier coverage without a ceiling, and a higher-`power` actor makes a better child."""
    anchors = [{"id": i} for i in range(6)]

    def develop(agent, project, rng):
        p = agent["params"]
        return 0.02 * float(p.get("n_inputs", 2)) + 0.04 * float(p.get("n_edge", 0))   # non-saturating

    def make_revise(cap):
        def revise(actor, target, rng):
            child = copy.deepcopy(target)
            power = int(actor["params"].get("power", 4))
            child["params"]["n_inputs"] = target["params"].get("n_inputs", 2) + power   # uncapped (calib)
            child["params"]["n_edge"] = target["params"].get("n_edge", 0) + power // 2
            child["params"]["power"] = min(cap, power + 4)          # improver improves improver up to cap
            return child
        return revise

    ok = True
    for mode, cap, exp in (("compound", 99, "F1>0,F2>0"), ("oneup", 8, "F1>0,F2~0")):
        rs = [run_lineage({"params": {"n_inputs": 2, "n_edge": 0, "power": 4}}, develop, make_revise(cap), anchors, random.Random(s), reps=2) for s in range(6)]
        agg = aggregate_lineages(rs); f1, f2 = agg["F1"]["mean"], agg["F2"]["mean"]
        good = (f1 > 0.01 and f2 > 0.01) if mode == "compound" else (f1 > 0.01 and abs(f2) < 0.03)
        ok = ok and good
        print("  [%s] %-9s F1=%+.3f F2=%+.3f (expect %s)" % ("PASS" if good else "FAIL", mode, f1, f2, exp))
    print("RSI-VERIFY CALIB", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate2", action="store_true")
    ap.add_argument("--calib", action="store_true")
    args = ap.parse_args()
    if args.gate2:
        sys.exit(gate2())
    if args.calib:
        sys.exit(calib())
    print("use --gate2 (opportunity proof) or --calib (executed-lineage detection)")


if __name__ == "__main__":
    main()
