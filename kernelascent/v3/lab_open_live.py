"""LIVE-MODEL open-ended compounding test. The scripted lab_open proved an open-ended (growing-archive)
substrate CAN exhibit resolved, accelerating compounding and that closed/memory cannot. This asks the
SCIENTIFIC question: does a REAL MODEL compound on an open-ended substrate, and is it capability-graded?

Substrate = library learning (verifiable code synthesis). State U = a LIBRARY of reusable helper functions
(source). develop(U, task): the model writes solve(xs) and may call any library helper; graded by execution
on hidden examples (continuous Q). revise(actor, target): the actor proposes ONE new reusable helper to add
to the TARGET's library, given (a) the target library it extends and (b) the ACTOR's own inherited
abstractions as prior experience -> a richer actor proposes a more useful helper -> newer producer builds a
better child -> F>0; deeper library -> next helper builds on more -> F2>0 (open-ended). Estimators = core.
Exec is guarded (SIGALRM) in a restricted namespace. Model via Bedrock (curate_bedrock.Curator).
"""
import os, sys, json, argparse, random, re, signal, statistics, copy, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci


class _TO(Exception):
    pass


def _guard(fn, sec=2.0):
    def _h(s, f):
        raise _TO()
    old = signal.signal(signal.SIGALRM, _h); signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------- tasks (hidden references; graded by exec)
# HARD, EDGE-HEAVY tasks that SHARE reusable sub-components (primality, run-length, distinct-sorted,
# local-extrema) -> a verified helper built once (via revise) helps several tasks -> library growth raises
# Q -> a live-model compounding channel exists ONLY if the model discovers+reuses those abstractions.
def _is_prime(n):
    if n < 2: return False
    i = 2
    while i * i <= n:
        if n % i == 0: return False
        i += 1
    return True


def _rle(xs):
    out = []
    for x in xs:
        if out and out[-1][0] == x: out[-1] = (x, out[-1][1] + 1)
        else: out.append((x, 1))
    return [tuple(t) for t in out]


def _local_max_idx(xs):
    return [i for i in range(len(xs)) if (i == 0 or xs[i] > xs[i - 1]) and (i == len(xs) - 1 or xs[i] > xs[i + 1])] if xs else []


def _pfactors(n):
    n = abs(n); out = set(); d = 2
    while d * d <= n:
        while n % d == 0:
            out.add(d); n //= d
        d += 1
    if n > 1:
        out.add(n)
    return out


def _digsum(n):
    return sum(int(c) for c in str(abs(n)))


def _digroot(n):
    n = abs(n)
    while n >= 10:
        n = _digsum(n)
    return n


def _is_pal(n):
    s = str(abs(n)); return s == s[::-1]


# HARD, EDGE-HEAVY tasks that SHARE sub-structure (is_prime, distinct-prime-factors, digit-sum/root,
# palindrome). Building ONE verified helper (e.g. is_prime, factorize) helps SEVERAL tasks -> a growing
# archive compounds. Compositional + trap-laden -> leaves headroom below frontier ceiling.
TASKS = [
    ("the number of unordered index pairs i<j such that xs[i]+xs[j] is prime",
     lambda xs: sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs)) if _is_prime(xs[i] + xs[j]))),
    ("the count of elements x in xs whose digit-sum is prime",
     lambda xs: sum(1 for x in xs if _is_prime(_digsum(x)))),
    ("the total number of DISTINCT prime factors across all elements of xs > 1 (counting a prime once per element)",
     lambda xs: sum(len(_pfactors(x)) for x in xs if x > 1)),
    ("the largest element of xs that is a base-10 palindrome (or -1 if none)",
     lambda xs: (max([x for x in xs if _is_pal(x)]) if any(_is_pal(x) for x in xs) else -1)),
    ("the digital root of the sum of squares of xs (0 if empty)",
     lambda xs: (_digroot(sum(x * x for x in xs)) if xs else 0)),
    ("the sum of the DISTINCT prime factors of max(xs) (0 if empty or max(xs)<2)",
     lambda xs: (sum(_pfactors(max(xs))) if xs and max(xs) >= 2 else 0)),
    ("the count of elements x in xs that are prime AND a base-10 palindrome",
     lambda xs: sum(1 for x in xs if _is_prime(x) and _is_pal(x))),
    ("the number of elements of xs whose digital root equals the count of their distinct prime factors",
     lambda xs: sum(1 for x in xs if _digroot(x) == len(_pfactors(x)))),
]


_EDGE_POOL = [0, 1, 2, 3, 5, 7, 11, 13, 9, 15, 25, 49, 100, 101, 121, 131, 151, 191, 200, 99, 88, 77]


def _sample(rng):
    # edge-heavy: primes, palindromes, 0/1/2 boundaries, perfect squares -> inline re-derivations botch edges
    n = rng.randint(0, 8)
    return [rng.choice(_EDGE_POOL) if rng.random() < 0.6 else rng.randint(0, 200) for _ in range(n)]


SOLVE_TMPL = ("Write a Python function solve(xs), where xs is a list of ints, that returns: {spec}.\n"
              "Examples: {ex}\n{lib}Return ONLY the solve function as a python code block.")
HELPER_TMPL = ("You are building a LIBRARY of reusable Python helper functions for list-of-int problems.\n"
               "Current library you are extending:\n{tgtlib}\n"
               "Helpers that worked well in your prior experience:\n{actlib}\n"
               "Sample problems you'll face: {specs}\n"
               "Propose ONE NEW general reusable helper function (not a full solver) that would make solving "
               "these easier and compose with existing helpers. Give it a clear name. Return ONLY the one "
               "def as a python code block.")


def _extract(txt):
    m = re.search(r"```(?:python)?\s*(.*?)```", txt or "", re.S)
    return m.group(1) if m else (txt or "")


import math as _math
# permissive-but-safe builtins: everything a normal list/int solution needs (fixes false-0 from valid code
# using math/bool/pow/etc.), minus file/network/exec escapes. Guarded by SIGALRM; models are not adversarial.
_SAFE_NAMES = ("len range sum max min sorted set frozenset abs enumerate list dict map filter zip int float "
               "bool str bytes bytearray chr ord divmod pow any all reversed round tuple complex hash repr "
               "format slice type isinstance issubclass getattr hasattr iter next callable sorted print "
               "True False None Exception ValueError TypeError IndexError KeyError ZeroDivisionError "
               "StopIteration ArithmeticError OverflowError").split()
import builtins as _bi
_SAFE_BUILTINS = {n: getattr(_bi, n) for n in _SAFE_NAMES if hasattr(_bi, n)}
_ALLOWED_MODS = {"math": _math}
for _m in ("itertools", "functools", "collections", "heapq", "re", "bisect", "operator", "string"):
    try:
        _ALLOWED_MODS[_m] = __import__(_m)
    except Exception:
        pass
def _safe_import(name, *a, **k):
    if name.split(".")[0] in _ALLOWED_MODS:
        return _ALLOWED_MODS[name.split(".")[0]]
    raise ImportError("module %r not permitted in the sandbox" % name)
_SAFE_BUILTINS["__import__"] = _safe_import


def _lib_ns(lib):
    ns = {"__builtins__": dict(_SAFE_BUILTINS), "math": _math}
    for name, src in lib.items():
        try:
            exec(compile(src, "<lib>", "exec"), ns)
        except Exception:
            pass
    return ns


def _lib_str(lib):
    return "".join("  - %s\n%s\n" % (n, s) for n, s in lib.items()) if lib else "  (empty)\n"


def make_behaviors(gen_fn, rng_master):
    cache = {}

    def develop(agent, task, rng):
        spec, ref = task
        lib = agent["params"]["lib"]
        key = (spec, tuple(sorted(lib)))
        if key in cache:
            return cache[key]
        ex = "; ".join("%r->%r" % (a, ref(a)) for a in [_sample(random.Random(i)) for i in range(3)])
        libtxt = (("These library helpers are ALREADY DEFINED and TESTED CORRECT (incl. tricky edge cases) "
                   "— prefer CALLING them over re-implementing:\n" + _lib_str(lib) + "\n") if lib else "")
        code = _extract(gen_fn(SOLVE_TMPL.format(spec=spec, ex=ex, lib=libtxt)))
        ns = _lib_ns(lib)
        try:
            exec(compile(code, "<solve>", "exec"), ns)
            solve = ns.get("solve")
        except Exception:
            cache[key] = 0.0; return 0.0
        if not callable(solve):
            cache[key] = 0.0; return 0.0
        ok = tot = 0
        rg = random.Random(hash(spec) & 0xffff)
        for _ in range(12):
            a = _sample(rg)
            try:
                exp = ref(a)
            except Exception:
                continue
            tot += 1
            try:
                if _guard(lambda: solve(list(a))) == exp:
                    ok += 1
            except Exception:
                pass
        q = ok / tot if tot else 0.0
        cache[key] = q; return q

    def revise(actor, target, rng):
        child = copy.deepcopy(target); lib = child["params"]["lib"]
        specs = "; ".join(t[0] for t in random.Random(rng.randint(0, 1 << 30)).sample(TASKS, 3))
        prompt = HELPER_TMPL.format(tgtlib=_lib_str(lib), actlib=_lib_str(actor["params"]["lib"]), specs=specs)
        code = _extract(gen_fn(prompt))
        # BUILD-TEST-FIX: exercise the proposed helper on edge probes; if it raises, feed the failure back
        # and let the model fix it ONCE. The archive thus holds DEBUGGED, verified components (load-bearing:
        # reusing a debugged helper beats an error-prone fresh inline attempt).
        def _probe(src):
            ns = _lib_ns(lib)
            try:
                exec(compile(src, "<helper>", "exec"), ns)
            except Exception as e:
                return None, None, "compile/exec error: %r" % e
            names = [n for n in re.findall(r"def\s+([a-zA-Z_]\w*)\s*\(", src) if not n.startswith("_")]
            name = names[0] if names else None
            fn = ns.get(name) if name else None
            if not callable(fn) or name in lib or name == "solve":
                return None, None, "no valid new helper name"
            trials = [(0,), (2,), (13,), (121,), ([1, 2, 2, 3],), ([],), (10, 3)]   # varied arg shapes/edges
            ran = False; lasterr = None
            for a in trials:
                try:
                    _guard(lambda: fn(*a)); ran = True
                except TypeError:
                    lasterr = "signature"                  # wrong arity for this shape -> try another
                except Exception as e:
                    return name, fn, "raised on %r: %r" % (a, e)   # a real runtime bug on an edge
            return (name, fn, None) if ran else (name, fn, "never ran (%s)" % lasterr)
        name, fn, err = _probe(code)
        if err and name is None and "no valid" not in err:
            fix = _extract(gen_fn("This helper failed (%s):\n```python\n%s\n```\nReturn a CORRECTED version as a python code block." % (err, code)))
            name, fn, err = _probe(fix); code = fix if name else code
        if name is None or fn is None:
            return child
        lib[name] = code                                    # ADD the debugged abstraction to the library
        return child

    return develop, revise


def run(args):
    import curate_bedrock as CB
    cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    cur.resolve(); cur.resolve_reasoning(); gen_fn = lambda p: cur.generate(p); who = "api:" + args.api_model
    develop, revise = make_behaviors(gen_fn, random.Random(0))
    U0 = {"params": {"lib": {}}}
    print("LAB-OPEN-LIVE %s lineages=%d tasks=%d" % (who, args.lineages, len(TASKS)), flush=True)
    results = []
    for s in range(args.lineages):
        r = run_lineage(copy.deepcopy(U0), develop, revise, TASKS, random.Random(s), reps=1)
        results.append(r)
        agg = aggregate_lineages(results)
        print("lin%d Q0=%.3f q1-q0=%+.3f N1=%+.3f F1=%+.3f F2=%+.3f  (agg F1=%+.3f F2=%+.3f)" %
              (s, r.Q["U0"], r.q1_minus_q0, r.N1, r.F1, r.F2, agg["F1"]["mean"], agg["F2"]["mean"]), flush=True)
        out = {"who": who, "lineages": s + 1, "Q0": _mean_ci([x.Q["U0"] for x in results]),
               "agg": agg}
        os.makedirs(args.outdir, exist_ok=True)
        json.dump(out, open(os.path.join(args.outdir, "lab_open_live_%s.json" %
                  who.replace(":", "_").replace("/", "_").replace(".", "_")), "w"), indent=2)
    agg = aggregate_lineages(results)
    print("\n=== %s ===" % who)
    for k in ("q1_minus_q0", "N1", "F1", "N2", "F2"):
        print("  %-12s %s" % (k, agg[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-model", required=True); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--lineages", type=int, default=6)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/lab_open_live")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
