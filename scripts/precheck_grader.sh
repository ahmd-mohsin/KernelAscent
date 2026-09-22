#!/usr/bin/env bash
# desc: STANDING GATE — never launch an experiment batch without this passing.
#
# This project's two most expensive failures were both a SILENT grader returning plausible
# zeros, which is indistinguishable from "the model wrote bad kernels":
#   1. a wrong KA_ROOT made grade_batch mis-import  -> every kernel False -> fake C=0, days lost
#   2. KA_GRADE_GPU defaulted to device "2" on a 2-GPU allocation -> subprocess saw no GPU ->
#      build_ref_c raised -> every kernel False again, caught only because this check existed
#
# Both were invisible from the outside. The only reliable detector is: a COPY OF THE REFERENCE,
# renamed to ModelNew, MUST grade correct with speedup ~1.0. Run it on the same allocation
# SHAPE the real jobs will use (-G matters: that is how bug 2 hid).
#
#   scripts/precheck_grader.sh          # 2 GPUs, matching the E1/E2 jobs
#   scripts/precheck_grader.sh 1        # 1 GPU
set -uo pipefail
G="${1:-2}"
MRL="${MRL:-mrl}"
echo "precheck: identity-kernel grading on a ${G}-GPU allocation"
$MRL submit -J precheck -N 1 -G "$G" -t 00:20:00 -- 'python - <<PY
import sys
from kernelascent.v3 import lab_kernel as LK
from kernelascent.v3 import lab_weight_rsi as W
name = list(LK.TASKS)[0]; src = LK.TASKS[name]
ident = src.replace("class Model(", "class ModelNew(")   # the reference, verbatim
print("grader device:", W._grade_gpu())
single = W._grade_isolated(src, [ident])
batch  = W._grade_isolated_batch([(src, [ident])])
ok_s = bool(single and single[0] and single[0][0])
ok_b = bool(batch and batch[0] and batch[0][0] and batch[0][0][0])
print("subprocess:", single, "->", "PASS" if ok_s else "FAIL")
print("batch     :", batch,  "->", "PASS" if ok_b else "FAIL")
if single and len(single[0]) > 3 and abs(single[0][3] - 1.5) < 1e-9:
    print("WARNING: ceiling is exactly 1.5 -- that is grade_batch.py fallback, i.e. build_ref_c RAISED")
sys.exit(0 if (ok_s and ok_b) else 1)
PY'
echo
echo "check the log; BOTH paths must PASS before launching a batch."
