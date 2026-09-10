"""Crash-isolated GPU grader in a FRESH process. Reports correctness plus BOTH eager and compiled speedups,
with multi-input numerical checks. Compiled baseline is cached per task to amortize compilation.

Input forms (argv[1] = json path):
  single : {"task": src, "codes": [..]}                   -> RESULT<json [[ok, sp_eager, sp_compiled], ...]>
  batch  : {"batch": [{"task": src, "codes": [..]}, ...]} -> RESULT<json [[[ok, se, sc], ...], ...]>
"""
import sys, json
sys.path.insert(0, "/tmp/instance_storage"); sys.path.insert(0, "/tmp/instance_storage/kernelascent")
import torch  # noqa
from kernelascent import agent_bench as AB

CACHE = "/tmp/instance_storage/ka_data/compiled_cache"


def grade_one(src, codes):
    try:
        ref, xs, golds, bound, te, tc = AB.build_ref_c(src, n_inputs=3, compiled_cache=CACHE)
    except Exception:
        return [[False, 0.0, 0.0] for _ in codes]
    out = []
    for code in codes:
        try:
            ok, se, sc, msg = AB.grade_c(src, code, ref, xs, golds, bound, te, tc)
            out.append([bool(ok), float(se), float(sc)])
        except Exception:
            out.append([False, 0.0, 0.0])
    return out


d = json.load(open(sys.argv[1]))
if "batch" in d:
    print("RESULT" + json.dumps([grade_one(it["task"], it["codes"]) for it in d["batch"]]))
else:
    print("RESULT" + json.dumps(grade_one(d["task"], d["codes"])))
