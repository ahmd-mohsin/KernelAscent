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


TASKS = [
    ("the sum of the values in xs that occur at PRIME indices (0-based; empty->0)",
     lambda xs: sum(xs[i] for i in range(len(xs)) if _is_prime(i))),
    ("the count of PRIME values in xs", lambda xs: sum(1 for x in xs if _is_prime(x))),
    ("run-length encode xs as a list of [value, count] pairs (consecutive runs)",
     lambda xs: [list(t) for t in _rle(xs)]),
    ("the length of the longest run of EQUAL consecutive elements in xs (0 if empty)",
     lambda xs: (max(c for _, c in _rle(xs)) if xs else 0)),
    ("the indices of strict local maxima in xs (strictly greater than each existing neighbor)",
     _local_max_idx),
    ("the number of strict local maxima in xs", lambda xs: len(_local_max_idx(xs))),
    ("the sorted list of DISTINCT values that appear more than once in xs",
     lambda xs: sorted(v for v in set(xs) if xs.count(v) > 1)),
    ("the value in xs with the highest count (ties -> smallest such value; -1 if empty)",
     lambda xs: (min(sorted(set(xs)), key=lambda v: (-xs.count(v), v)) if xs else -1)),
]


def _sample(rng):
    n = rng.randint(0, 9); return [rng.randint(0, 12) for _ in range(n)]


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
        libtxt = ("Library helpers available (already defined, call them):\n" + _lib_str(lib) + "\n") if lib else ""
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
        # validate: compiles, defines exactly a new callable that runs on a probe without error
        ns = _lib_ns(lib)
        try:
            exec(compile(code, "<helper>", "exec"), ns)
        except Exception:
            return child
        newfns = {n: code for n in ns if callable(ns[n]) and n not in _lib_ns(lib) and not n.startswith("_")}
        # find the helper name actually defined in `code`
        m = re.findall(r"def\s+([a-zA-Z_]\w*)\s*\(", code)
        name = next((n for n in m if not n.startswith("_")), None)
        if not name or name in lib or name == "solve":
            return child
        fn = ns.get(name)
        if not callable(fn):
            return child
        try:
            _guard(lambda: fn(list(_sample(rng))))          # must run on a probe without raising
        except Exception:
            return child
        lib[name] = code                                    # ADD the new abstraction to the child library
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
