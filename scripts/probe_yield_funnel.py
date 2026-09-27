#!/usr/bin/env python3
"""Where do a model's candidates die? Generated -> extracted -> compiled -> correct.

Why this exists. The size ladder reports an inverted capability curve: the frozen control scores
C_fresh 0.468 at 0.5B and 0.070 at 7B, with verified yield falling 12% -> 0.1%, while n_gens is
identical at every scale. probe_truncation.py tested the obvious explanation, a fixed token
budget biting harder on models that write longer, and REFUTED it: at both 1.5B and 7B, 0.0% of
completions reach the 2048 cap (median ~500 tokens), and the 7B extracts a well-formed ModelNew
97.9% of the time against the 1.5B's 45.8%. The larger model writes BETTER-formed code and
still yields nothing.

So the loss is downstream of extraction, and "yield" is too coarse to locate it. This walks the
whole funnel with the lab's own grader (_grade_isolated_batch, the same subprocess isolation the
real loop uses) and reports the survival rate at each stage, per model. The stage where 7B and
1.5B diverge is the bug.

Two flaws in the first probe are fixed here. It took the first N tasks alphabetically, which on
this bank collapsed to six variants of one family (dsl_cumsum_scale_*); this samples ACROSS
families. And it read a +10.4pp extraction difference off n=48, which is five samples; this
reports counts alongside every rate so a reader can see the denominator.

  python3 scripts/probe_yield_funnel.py --model Qwen/Qwen2.5-Coder-7B-Instruct --tasks 12 --k 8
"""
import argparse, collections, json, os, sys, time
import torch

from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
from kernelascent.v3 import lab_weight_rsi as W


def spread_tasks(names, n):
    """One task per family before a second from any family, so the sample is not one op repeated.

    Family is the task name minus its dtype/shape/scale suffix: dsl_cumsum_scale_f16_16384x4096_0p25
    and dsl_cumsum_scale_bf16_8192x2048_0p25 are the same family. Taking the first N alphabetically
    returned six of exactly one family on this bank.
    """
    fam = collections.OrderedDict()
    for t in sorted(names):
        parts = t.split("_")
        key = "_".join(parts[:3]) if len(parts) >= 3 else t
        fam.setdefault(key, []).append(t)
    out, i = [], 0
    while len(out) < n:
        added = False
        for k in fam:
            if i < len(fam[k]):
                out.append(fam[k][i]); added = True
                if len(out) >= n:
                    break
        if not added:
            break
        i += 1
    return out[:n], len(fam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    names, nfam = spread_tasks(list(LK.TASKS), a.tasks)
    fams = sorted({"_".join(t.split("_")[:3]) for t in names})
    sys.stderr.write("bank=%d tasks in %d families; sampling %d tasks from %d families\n"
                     % (len(LK.TASKS), nfam, len(names), len(fams)))
    sys.stderr.write("families: %s\n" % ", ".join(fams))

    srcs = [LK.TASKS[n] for n in names]
    tok, mdl = W.build(a.model, tuple(int(x) for x in a.gpus.split(",") if x != ""))
    mdl.eval()

    t0 = time.time()
    gen_lists = W.generate_batch(tok, mdl, srcs, a.k, adapter=False)     # frozen base, as the control arm runs
    n_gen = sum(len(g) for g in gen_lists)
    per_task_codes = [[c for c in (AB.extract_modelnew(t) for t in gl) if c] for gl in gen_lists]
    n_ext = sum(len(c) for c in per_task_codes)
    sys.stderr.write("generated %d, extracted %d, in %.0fs; grading\n" % (n_gen, n_ext, time.time() - t0))

    grades = W._grade_isolated_batch(list(zip(srcs, per_task_codes)))
    flat = [g for gl in grades for g in gl]
    n_ok = sum(1 for g in flat if g and g[0])
    sps = [g[2] for g in flat if g and g[0] and g[2] is not None]         # compiled speedup of correct ones

    # Per-task: did ANY of k candidates verify? That is what the loop actually consumes.
    solved = sum(1 for gl in grades if any(g and g[0] for g in gl))

    def rate(x, y):
        return (100.0 * x / y) if y else 0.0

    print("=" * 76)
    print("%s   %d tasks x k=%d   (frozen base, no adapter)" % (a.model, len(srcs), a.k))
    print("=" * 76)
    print("%-26s %8s %10s" % ("stage", "count", "of prev"))
    print("%-26s %8d %10s" % ("generated", n_gen, "--"))
    print("%-26s %8d %9.1f%%" % ("extracted a ModelNew", n_ext, rate(n_ext, n_gen)))
    print("%-26s %8d %9.1f%%" % ("verified correct", n_ok, rate(n_ok, n_ext)))
    print("%-26s %8d %9.1f%%" % ("end-to-end yield", n_ok, rate(n_ok, n_gen)))
    print()
    print("tasks with >=1 correct candidate: %d of %d (%.0f%%)" % (solved, len(srcs), rate(solved, len(srcs))))
    if sps:
        sps.sort()
        print("compiled speedup of correct candidates: median %.3f  min %.3f  max %.3f"
              % (sps[len(sps) // 2], sps[0], sps[-1]))
        print("  (a correct kernel at parity scores exactly 0.50 under _score; speedups at ~1.0")
        print("   mean the headroom scorer is pinned regardless of how many verify)")
    else:
        print("no correct candidate produced a speedup reading")

    rep = {"model": a.model, "tasks": names, "families": fams, "k": a.k,
           "n_gen": n_gen, "n_extracted": n_ext, "n_correct": n_ok,
           "extract_pct": round(rate(n_ext, n_gen), 2),
           "verify_pct_of_extracted": round(rate(n_ok, n_ext), 2),
           "yield_pct": round(rate(n_ok, n_gen), 2),
           "tasks_solved": solved,
           "speedup_median": (sps[len(sps) // 2] if sps else None)}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=2)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
