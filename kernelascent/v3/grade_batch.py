"""Crash-isolated GPU grader in a FRESH process, so a model-generated kernel that triggers a CUDA
device-side assert cannot poison the trainer's CUDA context.

Two input forms (argv[1] = json path):
  single : {"task": src, "codes": [..]}                      -> prints RESULT<json [[ok,sp],...]>
  batch  : {"batch": [{"task": src, "codes": [..]}, ...]}    -> prints RESULT<json [[[ok,sp],...], ...]>
The batch form amortizes the ~5-8s torch import + CUDA init across MANY tasks (one process per held-eval
instead of one per task), which is the dominant per-round overhead. build_ref/timing is still done per task.
"""
import sys, json
sys.path.insert(0, "/tmp/instance_storage"); sys.path.insert(0, "/tmp/instance_storage/kernelascent")
import torch  # noqa
from kernelascent import agent_bench as AB


def grade_one(src, codes):
    try:
        ref, x, gold, rerr = AB.build_ref(src)
        bound = max(2e-2, 2 * rerr); tbase = AB.time_fn(lambda z: ref(z), (x,))
    except Exception:
        return [[False, 0.0] for _ in codes]
    out = []
    for code in codes:
        try:
            ok, err, sp, msg = AB.grade(src, code, ref, x, gold, bound, tbase)
            out.append([bool(ok), float(sp)])
        except Exception:
            out.append([False, 0.0])
    return out


d = json.load(open(sys.argv[1]))
if "batch" in d:
    res = [grade_one(item["task"], item["codes"]) for item in d["batch"]]
    print("RESULT" + json.dumps(res))
else:
    print("RESULT" + json.dumps(grade_one(d["task"], d["codes"])))
