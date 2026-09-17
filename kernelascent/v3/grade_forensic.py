"""Forensic grader: like grade_batch.py but KEEPS the failure message from grade_c so we can classify WHY a
candidate scored 0 (parse / compile / runtime / wrong-output / correct). Input {"task":src,"codes":[..]} ->
RESULT<json [[ok, sp_eager, msg], ...]>. Used by lab_failure_forensics.py for 0-score mechanistic analysis."""
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
        return [[False, 0.0, "ref_build_fail:" + repr(e)[:60]] for _ in codes]
    out = []
    for code in codes:
        try:
            ok, se, sc, msg = AB.grade_c(src, code, ref, xs, golds, bound, te, tc)
            out.append([bool(ok), float(se), str(msg)[:100]])
        except Exception as e:
            out.append([False, 0.0, "grade_exc:" + repr(e)[:80]])
    return out


d = json.load(open(sys.argv[1]))
if "batch" in d:
    print("RESULT" + json.dumps([grade_one(it["task"], it["codes"]) for it in d["batch"]]))
else:
    print("RESULT" + json.dumps(grade_one(d["task"], d["codes"])))
