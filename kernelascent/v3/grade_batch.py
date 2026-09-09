"""Crash-isolated GPU grader: grade K candidate kernels for ONE task in a FRESH process, so a model-
generated kernel that triggers a CUDA device-side assert cannot poison the trainer's CUDA context.
Reads {"task": src, "codes": [..]} from argv[1] json; prints RESULT<json list of [ok, speedup]>.
"""
import sys, json
sys.path.insert(0, "/tmp/instance_storage"); sys.path.insert(0, "/tmp/instance_storage/kernelascent")
import torch  # noqa
from kernelascent import agent_bench as AB

d = json.load(open(sys.argv[1]))
src = d["task"]; codes = d["codes"]
ref, x, gold, rerr = AB.build_ref(src)
bound = max(2e-2, 2 * rerr); tbase = AB.time_fn(lambda z: ref(z), (x,))
out = []
for code in codes:
    try:
        ok, err, sp, msg = AB.grade(src, code, ref, x, gold, bound, tbase)
        out.append([bool(ok), float(sp)])
    except Exception:
        out.append([False, 0.0])
print("RESULT" + json.dumps(out))
