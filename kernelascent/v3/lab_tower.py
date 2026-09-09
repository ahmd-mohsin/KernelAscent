"""TRUE-RSI substrate: a COMPOSITIONAL TOWER where the archive is unavoidably LOAD-BEARING.

Rung k computes f_k(xs) = h_k(f_{k-1}(xs)); f_0 = xs. Each layer h is a simple but edge-prone list->list
transform. Solving rung k FROM SCRATCH means re-deriving h_1..h_k inline -> per-edge correctness compounds
DOWN (~p^k), so deep rungs are unreliable one-shot for ANY model. Solving by CALLING a VERIFIED archived
solver for rung k-1 and adding only h_k stays reliable. So a richer archive lets a producer build a DEEPER
rung a poorer producer cannot -> newer producer builds a better child on a common target (F1>0), and it
repeats up the tower (F2>0) = compounding. The archive is load-bearing by construction, not by hoping the
model won't re-derive.

  --calib  : deterministic perfect/noisy builder -> proves the tower yields F1,F2>0 (design validation).
  --api-model <id> : LIVE model builds rung-solvers (build-test-fix) + solves rungs by composing the archive.
Estimators = core.run_lineage (unchanged). Live exec guarded (SIGALRM) in a restricted namespace.
"""
import os, sys, json, argparse, random, re, signal, statistics, copy
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci


def _isp(n):
    if n < 2: return False
    i = 2
    while i * i <= n:
        if n % i == 0: return False
        i += 1
    return True


# edge-prone list->list layers (each easy alone; composing many inline accumulates edge errors)
LAYERS = [
    ("square mod 97", lambda a: [(x * x) % 97 for x in a]),
    ("running maximum", lambda a: [max(a[:i + 1]) for i in range(len(a))]),
    ("consecutive differences (first element kept)", lambda a: [a[i] - a[i - 1] if i else a[0] for i in range(len(a))]),
    ("replace primes with 0", lambda a: [0 if _isp(x) else x for x in a]),
    ("prefix sums", lambda a: [sum(a[:i + 1]) for i in range(len(a))]),
    ("last decimal digit of |x|", lambda a: [abs(x) % 10 for x in a]),
    ("sort ascending", lambda a: sorted(a)),
    ("dedupe keeping first occurrence", lambda a: [x for i, x in enumerate(a) if x not in a[:i]]),
]
D = len(LAYERS)


def f_rung(k, xs):
    a = list(xs)
    for _, h in LAYERS[:k]:
        a = h(a)
    return a


def _sample(rng):
    n = rng.randint(0, 7); return [rng.randint(-4, 12) for _ in range(n)]


def _spec(k):
    return " -> ".join("(%d) %s" % (i + 1, LAYERS[i][0]) for i in range(k))


# ---------------------------------------------------------------- deterministic calibration
def calib(args):
    P = 0.86                                   # per-layer from-scratch edge correctness
    rungs = list(range(1, D + 1))
    def develop(agent, k, rng):
        d = agent["params"]["depth"]           # archive covers rungs 1..d (verified)
        extra = max(0, k - d)                  # layers you must re-derive inline
        return P ** extra                      # reuse verified prefix; only re-derived tail is error-prone
    def make_revise(cap):
        def revise(actor, target, rng):
            c = copy.deepcopy(target)
            r = actor["params"]["depth"] + 1   # actor can build one rung beyond ITS OWN archive
            if r <= cap and actor["params"]["depth"] >= r - 1:
                c["params"]["depth"] = max(c["params"]["depth"], r)
            return c
        return revise
    ok = True
    for mode, cap, exp in (("compound", D, "F1>0,F2>0"), ("oneup", 2, "F1>0,F2~0")):
        rs = [run_lineage({"params": {"depth": 0}}, develop, make_revise(cap), rungs, random.Random(s), reps=2) for s in range(8)]
        a = aggregate_lineages(rs); f1, f2 = a["F1"]["mean"], a["F2"]["mean"]
        good = (f1 > 0.01 and f2 > 0.01) if mode == "compound" else (f1 > 0.01 and abs(f2) < 0.03)
        ok = ok and good
        print("  [%s] %-9s F1=%+.3f F2=%+.3f (%s)" % ("PASS" if good else "FAIL", mode, f1, f2, exp))
    print("TOWER CALIB", "ALL PASS" if ok else "FAIL"); return 0 if ok else 1


# ---------------------------------------------------------------- live model
_SAFE = ("len range sum max min sorted set abs enumerate list dict map filter zip int float bool str "
         "reversed round tuple any all divmod pow").split()
import builtins as _bi
_SB = {n: getattr(_bi, n) for n in _SAFE if hasattr(_bi, n)}


def _ns(lib):
    ns = {"__builtins__": dict(_SB)}
    for src in lib.values():
        try:
            exec(compile(src, "<lib>", "exec"), ns)
        except Exception:
            pass
    return ns


class _TO(Exception):
    pass


def _guard(fn, sec=2.0):
    def h(s, f): raise _TO()
    o = signal.signal(signal.SIGALRM, h); signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, o)


def _extract(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t or "", re.S); return m.group(1) if m else (t or "")


def _grade(fn, k, rng, n=12):
    ok = tot = 0
    for _ in range(n):
        a = _sample(rng)
        try:
            exp = f_rung(k, a)
        except Exception:
            continue
        tot += 1
        try:
            if _guard(lambda: fn(list(a))) == exp:
                ok += 1
        except Exception:
            pass
    return ok / tot if tot else 0.0


def make_behaviors(gen):
    cache = {}

    def _libtxt(lib):
        if not lib:
            return ""
        names = ", ".join("solve%d(xs)=rung %d" % (k, k) for k in sorted(lib))
        return ("VERIFIED helpers already defined (call them; they are correct incl. edge cases): %s.\n" % names)

    def develop(agent, k, rng):
        lib = agent["params"]["lib"]
        key = (k, tuple(sorted(lib)))
        if key in cache:
            return cache[key]
        ex = "; ".join("%r->%r" % (a, f_rung(k, a)) for a in [_sample(random.Random(i)) for i in range(3)])
        p = ("Write a Python function solve(xs), where xs is a list of ints, that returns the result of "
             "applying these operations to xs IN ORDER: %s.\nExamples: %s\n%sReturn ONLY the solve function "
             "as a python code block." % (_spec(k), ex, _libtxt(lib)))
        ns = _ns({("solve%d" % j): s for j, s in lib.items()})
        try:
            exec(compile(_extract(gen(p)), "<s>", "exec"), ns); fn = ns.get("solve")
        except Exception:
            cache[key] = 0.0; return 0.0
        q = _grade(fn, k, random.Random(k * 7 + 1)) if callable(fn) else 0.0
        cache[key] = q; return q

    def revise(actor, target, rng):
        child = copy.deepcopy(target); lib = child["params"]["lib"]
        r = (max(actor["params"]["lib"]) if actor["params"]["lib"] else 0) + 1   # build one beyond ACTOR's depth
        if r > D or (r - 1 >= 1 and (r - 1) not in actor["params"]["lib"]):
            return child
        ex = "; ".join("%r->%r" % (a, f_rung(r, a)) for a in [_sample(random.Random(i)) for i in range(4)])
        helpers = _libtxt(actor["params"]["lib"])
        def build(extra=""):
            p = ("Write a Python function solve%d(xs), where xs is a list of ints, that returns the result "
                 "of applying these operations to xs IN ORDER: %s.\nExamples: %s\n%sYou may CALL the verified "
                 "helpers above. Return ONLY the solve%d function as a python code block.%s"
                 % (r, _spec(r), ex, helpers, r, extra))
            return _extract(gen(p))
        def verify(src):
            ns = _ns({("solve%d" % j): s for j, s in actor["params"]["lib"].items()})
            try:
                exec(compile(src, "<b>", "exec"), ns); fn = ns.get("solve%d" % r)
            except Exception as e:
                return None, "exec %r" % e
            if not callable(fn):
                return None, "no solve%d" % r
            miss = [a for a in [_sample(random.Random(100 + i)) for i in range(6)] if (lambda: _guard(lambda: fn(list(a))))() != f_rung(r, a)]
            return (fn, None) if not miss else (None, "wrong on %r" % miss[0])
        src = build(); fn, err = verify(src)
        if err:
            src2 = build("\nYour previous attempt failed (%s). Return a CORRECTED solve%d." % (err, r)); fn, err = verify(src2); src = src2 if fn else src
        if fn is not None:
            lib[r] = src
        return child

    return develop, revise


def run(args):
    import curate_bedrock as CB
    cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    cur.resolve(); cur.resolve_reasoning(); who = "api:" + args.api_model
    def gen(p):                                    # reasoning models intermittently return empty -> retry
        o = ""
        for _ in range(4):
            o = cur.generate(p) or ""
            if o.strip():
                return o
        return o
    develop, revise = make_behaviors(gen)
    rungs = list(range(1, D + 1))
    U0 = {"params": {"lib": {}}}
    print("TOWER-LIVE %s D=%d lineages=%d" % (who, D, args.lineages), flush=True)
    results = []
    for s in range(args.lineages):
        r = run_lineage(copy.deepcopy(U0), develop, revise, rungs, random.Random(s), reps=1)
        results.append(r); agg = aggregate_lineages(results)
        print("lin%d Q0=%.3f q1-q0=%+.3f N1=%+.3f F1=%+.3f F2=%+.3f (aggF1=%+.3f F2=%+.3f)" %
              (s, r.Q["U0"], r.q1_minus_q0, r.N1, r.F1, r.F2, agg["F1"]["mean"], agg["F2"]["mean"]), flush=True)
        os.makedirs(args.outdir, exist_ok=True)
        json.dump({"who": who, "lineages": s + 1, "Q0": _mean_ci([x.Q["U0"] for x in results]), "agg": agg},
                  open(os.path.join(args.outdir, "tower_%s.json" % who.replace(":", "_").replace("/", "_").replace(".", "_")), "w"), indent=2)
    agg = aggregate_lineages(results)
    print("\n=== TOWER %s ===" % who)
    for k in ("q1_minus_q0", "N1", "F1", "N2", "F2"):
        print("  %-12s %s" % (k, agg[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--lineages", type=int, default=10)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/tower")
    args = ap.parse_args()
    if args.calib:
        sys.exit(calib(args))
    if args.api_model:
        run(args); return
    print("use --calib or --api-model <id>")


if __name__ == "__main__":
    main()
