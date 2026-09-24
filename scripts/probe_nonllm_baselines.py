#!/usr/bin/env python3
"""What does a NON-LLM tool score on this benchmark?

A kernel-optimisation benchmark with no non-LLM baseline invites the obvious question: how does
a language model compare with just running the compiler harder? This probe answers it on the
benchmark's own terms -- same tasks, same timing harness, same grader, same headroom score --
by treating `torch.compile(mode="max-autotune")` as if it were a submission.

The comparison the paper needs is not "max-autotune is Nx faster than eager". It is: if
max-autotune were submitted to this benchmark, what score would it get? That number sits
directly beside the model scores, which nothing else here does.

No model is involved and nothing is generated, so this is a property of the task bank and the
hardware, and it is reproducible without an LLM at all.

    python scripts/probe_nonllm_baselines.py
"""
import json, os, sys, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    import torch
    from kernelascent import agent_bench as AB
    from kernelascent.v3 import lab_kernel as LK
    from kernelascent import provenance as PROV

    names = list(LK.TASKS)
    print("NON-LLM BASELINE PROBE over %d tasks (no model involved)" % len(names), flush=True)
    rows = []
    for i, n in enumerate(names):
        src = LK.TASKS[n]
        try:
            ref, xs, golds, bound, te, tc, ceil = AB.build_ref_c(src, n_inputs=3)
            # max-autotune, timed through the SAME harness as every other number here
            # (time_fn: 8 warmup, 20 iters, L2 flushed between, median).
            tma = None
            try:
                mref = torch.compile(ref, mode="max-autotune")
                with torch.no_grad():
                    for _ in range(3):
                        mref(xs[0])
                tma = AB.time_fn(lambda z: mref(z), (xs[0],))
            except Exception as e:
                print("      max-autotune unavailable for %s: %r" % (n, e), flush=True)

            row = {"task": n, "t_eager": te, "t_compiled": tc, "t_max_autotune": tma,
                   "roofline_ceiling": round(ceil, 3),
                   "compile_gain_over_eager": round(te / tc, 3) if tc else None}
            if tma:
                # Scored exactly as a model submission is: speedup over the COMPILED baseline,
                # normalised by the same per-task roofline ceiling. ok=True because the compiler
                # is correct by construction -- it is the reference, transformed.
                sp = tc / tma
                row["max_autotune_speedup_vs_compiled"] = round(sp, 3)
                row["max_autotune_benchmark_score"] = round(LK._score(True, sp, ceil), 4)
            rows.append(row)
            print("  [%d/%d] %-28s eager %.3fms  compiled %.3fms  max-auto %s  score %s"
                  % (i + 1, len(names), n[:28], te * 1e3, tc * 1e3,
                     ("%.3fms" % (tma * 1e3)) if tma else "n/a",
                     row.get("max_autotune_benchmark_score", "n/a")), flush=True)
        except Exception as e:
            print("  [%d/%d] %-28s FAILED %r" % (i + 1, len(names), n[:28], e), flush=True)

    scored = [r for r in rows if r.get("max_autotune_benchmark_score") is not None]
    if not scored:
        print("\nno task produced a max-autotune score"); return 1

    sc = [r["max_autotune_benchmark_score"] for r in scored]
    sp = [r["max_autotune_speedup_vs_compiled"] for r in scored]
    at_parity = sum(1 for x in sc if abs(x - 0.5) < 0.01)
    beats = sum(1 for x in sp if x > 1.03)
    print("\n=== max-autotune AS A SUBMISSION (n=%d tasks) ===" % len(scored))
    print("  benchmark score : median %.3f  mean %.3f  max %.3f" % (st.median(sc), st.mean(sc), max(sc)))
    print("  speedup vs compiled baseline: median %.2fx  max %.2fx" % (st.median(sp), max(sp)))
    print("  scores within 0.01 of parity (0.50): %d/%d (%.0f%%)"
          % (at_parity, len(sc), 100 * at_parity / len(sc)))
    print("  tasks where it beats the compiled baseline by >3%%: %d/%d" % (beats, len(sp)))
    print("""
  READ-OUT
    median ~0.50  -> the strongest non-LLM tool available also lands at correct-at-parity. The
      benchmark's headroom is not reachable by production compilers either, so a model scoring
      0.50 is matching the best automated baseline, not failing to beat a weak one.
    median >> 0.50 -> max-autotune captures real headroom that the models do not, which makes
      the model results a genuine capability gap rather than a property of the task bank.""")
    out = os.environ.get("KA_PROBE_OUT", "/users/muahmed/ka_data/nonllm_baselines.json")
    PROV.dump({"n_tasks": len(rows), "n_scored": len(scored), "rows": rows,
               "median_score": round(st.median(sc), 4),
               "median_speedup_vs_compiled": round(st.median(sp), 3),
               "frac_at_parity": round(at_parity / len(sc), 3),
               "n_beating_compiled": beats}, out, indent=2)
    print("\n  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
