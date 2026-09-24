#!/usr/bin/env python3
"""How much does torch.compile actually speed up the reference on each bank task?

This settles a question I have now answered wrongly twice. The H100 runs scored 76% of
submissions at exactly 0.50, and I have blamed, in order:

  1. "torch.compile is relatively stronger on H100, closing the headroom"  (original)
  2. "the default scorer never touches the compiled baseline"              (the correction)

(2) is false: every job exports KA_SCORE=compiled via MRL_EXTRA_ENV, and has since 2026-09-19,
so those runs DID score against torch.compile with the per-task roofline ceiling.

A score of exactly 0.50 requires sp_compiled == 1.0 exactly -- the candidate matching the
COMPILED baseline. If torch.compile gives ~1.0x on these ops (common for memory-bound
elementwise work), then a plain-torch rewrite of the reference lands at exactly 1.0x and scores
0.50, with no need for compile to be "strong" at all. That is a third explanation and it is
testable directly: measure t_eager / t_compiled per task, with no model involved.
"""
import json, os, sys, time, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    import torch
    from kernelascent import agent_bench as AB
    from kernelascent.v3 import lab_kernel as LK
    from kernelascent import provenance as PROV

    names = list(LK.TASKS)
    print("COMPILE-GAIN PROBE over %d tasks (no model; reference only)" % len(names), flush=True)
    rows = []
    for i, n in enumerate(names):
        src = LK.TASKS[n]
        try:
            ref, xs, golds, bound, te, tc, ceil = AB.build_ref_c(src, n_inputs=3)
            gain = (te / tc) if tc else float("nan")      # >1 means compile is FASTER than eager
            rows.append({"task": n, "t_eager": te, "t_compiled": tc,
                         "compile_gain": round(gain, 3), "roofline_ceiling": round(ceil, 2)})
            print("  [%d/%d] %-30s eager %.3fms  compiled %.3fms  gain %.2fx  ceiling %.1fx"
                  % (i + 1, len(names), n[:30], te * 1e3, tc * 1e3, gain, ceil), flush=True)
        except Exception as e:
            print("  [%d/%d] %-30s FAILED %r" % (i + 1, len(names), n[:30], e), flush=True)
    if not rows:
        print("no task measured"); return 1

    g = [r["compile_gain"] for r in rows if r["compile_gain"] == r["compile_gain"]]
    near1 = sum(1 for x in g if 0.97 <= x <= 1.03)
    print("\n=== COMPILE GAIN OVER EAGER ===")
    print("  median %.2fx   mean %.2fx   min %.2fx   max %.2fx" % (
        st.median(g), st.mean(g), min(g), max(g)))
    print("  tasks where compile is within 3%% of eager: %d/%d (%.0f%%)"
          % (near1, len(g), 100 * near1 / len(g)))
    print("""
  READ-OUT
    gain ~1.0x on most tasks -> compile does NOT help here, so the compiled baseline IS the
      eager baseline, and a plain-torch rewrite scores exactly 0.50 without compile being
      "strong". The 0.50 spike is the model reproducing the reference, full stop.
    gain >> 1.0x            -> compile IS much faster, so a 0.50 score means the model MATCHED
      an optimised baseline, which would be a far stronger result than we have claimed.""")
    out = os.environ.get("KA_PROBE_OUT", "/users/muahmed/ka_data/compile_gain.json")
    PROV.dump({"n_tasks": len(rows), "rows": rows,
               "median_gain": round(st.median(g), 3),
               "frac_within_3pct_of_eager": round(near1 / len(g), 3)}, out, indent=2)
    print("\n  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
