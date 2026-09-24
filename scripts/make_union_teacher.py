#!/usr/bin/env python3
"""Build one teacher from every T1 harvest, keeping the FASTEST verified kernel per task.

A positive control is only as strong as the thing it injects (INTUITIONS 2.11). The T2-kernel
inject arms were pointed at the 14B harvest, which covers ONE task with ONE kernel -- injecting
that is close to a no-op, and a control that cannot move is not a control.

No single scale is a good teacher here: solve-rate does not increase with scale on this task
(0.5B solves 8/29, 14B solves 1/29), because compliance rises with scale while
verify-given-attempt stays low. So the best available teacher is the UNION across scales.

Prefers kernels that actually contain triton/CUDA: injecting a plain-torch rewrite on a
kernel-authoring task teaches the student to decline the task, which is the behaviour we are
trying to measure rather than reinforce.
"""
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kernelascent import provenance as PROV

KERNEL_MARKS = ("triton", "load_inline", "__global__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/users/muahmed/ka_data/t1k_*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=2)
    a = ap.parse_args()

    best, srcs = {}, []
    for f in sorted(glob.glob(a.glob)):
        try:
            d = json.load(open(f))
        except Exception as e:
            print("  skip %s (%s)" % (os.path.basename(f), e)); continue
        srcs.append(os.path.basename(f))
        for task, ks in (d.get("kernels") or {}).items():
            for k in ks:
                code = k.get("code", "")
                cand = dict(k)
                cand["is_kernel"] = any(m in code for m in KERNEL_MARKS)
                cand["from"] = os.path.basename(f)
                best.setdefault(task, []).append(cand)

    out = {}
    for task, cands in best.items():
        # real kernels first, then fastest -- a plain-torch rewrite is a last resort
        cands.sort(key=lambda c: (not c["is_kernel"], -c.get("speedup_eager", 0)))
        out[task] = cands[: a.per_task]

    n_k = sum(1 for v in out.values() for c in v if c["is_kernel"])
    n_all = sum(len(v) for v in out.values())
    print("UNION TEACHER from %d harvest(s): %s" % (len(srcs), ", ".join(srcs)))
    print("  tasks covered : %d" % len(out))
    print("  kernels kept  : %d (%d contain triton/CUDA = %.0f%%)"
          % (n_all, n_k, 100 * n_k / max(n_all, 1)))
    print("  tasks whose BEST is a real kernel: %d/%d"
          % (sum(1 for v in out.values() if v and v[0]["is_kernel"]), len(out)))
    PROV.dump({"teacher": "union(t1k)", "sources": srcs, "n_tasks": len(out),
               "per_task": a.per_task, "kernels": out}, a.out, indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
