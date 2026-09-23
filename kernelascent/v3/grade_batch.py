"""Crash-isolated GPU grader in a FRESH process. Reports correctness plus BOTH eager and compiled speedups,
with multi-input numerical checks. Compiled baseline is cached per task to amortize compilation.

Input forms (argv[1] = json path):
  single : {"task": src, "codes": [..]}                   -> RESULT<json [[ok, sp_eager, sp_compiled], ...]>
  batch  : {"batch": [{"task": src, "codes": [..]}, ...]} -> RESULT<json [[[ok, se, sc], ...], ...]>
"""
import sys, json, os
_ROOT = os.environ.get("KA_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "kernelascent"))
import torch  # noqa
from kernelascent import agent_bench as AB

CACHE = os.environ.get("KA_COMPILED_CACHE") or os.path.join(
    os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "compiled_cache")


def grade_one(src, codes):
    try:
        ref, xs, golds, bound, te, tc, ceil = AB.build_ref_c(src, n_inputs=3, compiled_cache=CACHE)
    except Exception as e:
        # A reference that will not build fails EVERY candidate for this task, which is how a
        # quota error once read as "the model produced nothing correct for 5 rounds". Say so.
        return [[False, 0.0, 0.0, 1.5, "REF-BUILD-FAILED " + repr(e)[:100]] for _ in codes]
    out = []
    for code in codes:
        try:
            ok, se, sc, msg = AB.grade_c(src, code, ref, xs, golds, bound, te, tc)
            # 5th = WHY. grade_c has always produced this string and it was unpacked and then
            # dropped here, so every failure in this project has been a bare False with no
            # reason attached -- which is precisely why "the model writes bad kernels" stayed
            # unfalsifiable through four separate harness bugs. Appended, not substituted, so
            # every existing consumer (all of which index g[0..3]) is unaffected.
            out.append([bool(ok), float(se), float(sc), float(ceil), str(msg)[:120]])
        except Exception as e:
            out.append([False, 0.0, 0.0, float(ceil), "GRADER-RAISED " + repr(e)[:100]])
    return out


d = json.load(open(sys.argv[1]))
if "batch" in d:
    print("RESULT" + json.dumps([grade_one(it["task"], it["codes"]) for it in d["batch"]]))
else:
    print("RESULT" + json.dumps(grade_one(d["task"], d["codes"])))
