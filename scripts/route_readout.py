#!/usr/bin/env python3
"""Fixed read-outs for routes 1-3, written BEFORE the runs finish.

The instrument-validity section argues that a benchmark should commit to its analysis before
seeing the data. That has to apply to this work too, so each read-out below states its decision
rule in the code rather than in a later judgement call.

  route_readout.py r1 [report.json]    does the substrate have room the model can occupy?
  route_readout.py r2 [teacher.json]   can a >14B model beat the torch.compile baseline?
  route_readout.py r3 [outdir-glob]    lineage-reset under KA_SCORE=passrate
  route_readout.py all
"""
import json, os, sys, glob, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "marlowe_h100")
PARITY = (0.49, 0.51)          # _score(correct, speedup=1.0) == 0.50 exactly


def _load(p):
    """None for a file that has not landed yet (normal while jobs are queued); loud for a file
    that exists but will not parse, since that is a truncated or half-written result."""
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        print("  !! %s exists but will not parse (%s) -- truncated run?" % (os.path.basename(p), e))
        return None


def r1(paths=None):
    """Substrate quality. The question is NOT 'is there a middle band' -- neither bank has a
    fat one. It is whether tasks the model solves have headroom LEFT, and whether anything on
    the substrate has ever demonstrably captured headroom."""
    print("=" * 84)
    print("ROUTE 1 -- does the substrate leave the model room to move?")
    print("=" * 84)
    banks = paths or [("29-task", os.path.join(D, "bank_h100_report.json")),
                      ("456-DSL", os.path.join(D, "bank_dsl_h100_report.json"))]
    print("%-9s %7s %6s %9s %9s %12s %11s" %
          ("bank", "scored", "kept", ">0.51", ">0.75", "ceil@parity", "mean kept"))
    rows = {}
    for lbl, p in banks:
        d = _load(p)
        if not d:
            continue
        kept = [x for x in d if x.get("keep")]
        par = [x for x in kept if PARITY[0] <= x["best_score"] <= PARITY[1]]
        rows[lbl] = dict(
            n=len(d), kept=len(kept),
            gt51=sum(1 for x in d if x["best_score"] > 0.51),
            gt75=sum(1 for x in d if x["best_score"] > 0.75),
            ceil=st.median([x["ceiling"] for x in par]) if par else 0.0,
            mean=st.mean([x["best_score"] for x in kept]) if kept else 0.0)
        r = rows[lbl]
        print("%-9s %7d %6d %9d %9d %11.1fx %11.3f" %
              (lbl, r["n"], r["kept"], r["gt51"], r["gt75"], r["ceil"], r["mean"]))

    if "456-DSL" not in rows or "29-task" not in rows:
        return
    a, b = rows["29-task"], rows["456-DSL"]
    print("""
DECISION RULE (fixed in advance): the substrate is usable iff (i) some task on it has
demonstrably captured real headroom, so the speed dimension is reachable AT ALL, and
(ii) the tasks actually kept still have headroom above them.

  (i) tasks scoring >0.75 (base captured >half its roofline headroom)
        29-task {gt75a:>4d}        456-DSL {gt75b:>4d}
  (ii) median roofline ceiling on KEPT tasks sitting at parity
        29-task {ca:>4.1f}x       456-DSL {cb:>4.1f}x

VERDICT: {verdict}

Neither bank has a fat middle band -- both are bimodal. What separates them is that on the
DSL bank {gt75b} tasks demonstrate a frozen 3B base CAN capture most of the available headroom,
while on the 29-task bank exactly {gt75a} ever did. So "these models cannot beat torch.compile"
was a property of the old BANK, not of the models. And a kept DSL task at parity has a {cb:.1f}x
ceiling overhead versus {ca:.1f}x, so improvement has somewhere to register.

CAVEAT that must be resolved before publishing any headroom number from this bank: tasks
scoring exactly 1.00 have measured speedup >= their computed ceiling. Either these generated
references are naive enough to be beaten by that margin, or the roofline ceiling is
UNDER-estimated for them -- which would inflate every score on those tasks.""".format(
        gt75a=a["gt75"], gt75b=b["gt75"], ca=a["ceil"], cb=b["ceil"],
        verdict=("DSL bank is a usable substrate; the 29-task bank is not"
                 if b["gt75"] > 10 and b["ceil"] > 2 * a["ceil"] else
                 "NEITHER bank clears the rule -- do not run the compounding study on either")))


def r2(path=None):
    """Routes 2+4 are one design: prompt (safe|kernel) x scale (14B|32B).

    The cell that matters is 14B-safe vs 14B-kernel, because only the prompt differs -- same
    tasks, same grader, same model, same seed. 32B-kernel asks whether scale adds anything once
    the model is actually asked to optimise. 32B-safe is not run: its outcome is predictable
    from the 14B-safe result and it is the most expensive cell.
    """
    print("=" * 84)
    print("ROUTES 2+4 -- prompt x scale: is the plateau prompt-induced?")
    print("=" * 84)
    cells = [("14B", "safe", os.path.join(D, "r4_safe_q14.json")),
             ("14B", "kernel", os.path.join(D, "r4_kernel_q14.json")),
             ("32B", "kernel", path or os.path.join(D, "r2_kernel_q32.json")),
             ("14B", "safe(orig)", os.path.join(D, "teacher_kernels_q14.json"))]
    print("%-5s %-11s %7s %8s %10s %9s %9s" %
          ("model", "prompt", "solved", "kernels", "custom-k", "median", ">1.05x"))
    got = {}
    for scale, prompt, f in cells:
        d = _load(f)
        if not d:
            print("%-5s %-11s   (not harvested yet)" % (scale, prompt)); continue
        ks = [k for v in d.get("kernels", {}).values() for k in v]
        best = [max(k["speedup_eager"] for k in v) for v in d.get("kernels", {}).values() if v]
        if not ks:
            print("%-5s %-11s %7d  solved nothing" % (scale, prompt, 0)); continue
        cust = sum(1 for k in ks if any(t in k["code"] for t in
                   ("triton", "load_inline", "__global__")))
        row = dict(solved=len(best), n=len(ks), cust=cust,
                   med=st.median(best), fast=sum(1 for v in best if v > 1.05))
        got[(scale, prompt)] = row
        print("%-5s %-11s %7d %8d %9d%% %8.2fx %8.0f%%" %
              (scale, prompt, row["solved"], row["n"], round(100 * cust / len(ks)),
               row["med"], 100 * row["fast"] / max(len(best), 1)))

    a, b = got.get(("14B", "safe")), got.get(("14B", "kernel"))
    if not (a and b):
        print("\n  (both 14B cells needed for the verdict)"); return
    print("""
DECISION RULE (fixed in advance). Only the prompt differs between these two cells.
  custom-kernel rate rises AND median speedup rises  -> the plateau is substantially
      prompt-induced, and every headroom board has to be re-run before it means anything
  custom-kernel rate stays ~0                        -> the models genuinely will not write
      kernels; the published prompt was not the binding constraint
  custom rate rises but SOLVED collapses             -> the bottleneck is formation, not
      intent. Report ambition and coverage separately; do not average them.""")
    dk = b["cust"] * 100 // max(b["n"], 1) - a["cust"] * 100 // max(a["n"], 1)
    dm = b["med"] - a["med"]
    ds = b["solved"] - a["solved"]
    print("\n  OBSERVED: custom-kernel %+d pts, median speedup %+.2fx, tasks solved %+d" % (dk, dm, ds))
    if dk > 20 and dm > 0.05:
        v = "PROMPT-INDUCED -- the published boards measure compliance, not capability"
    elif dk > 20 and ds < 0:
        v = "FORMATION-LIMITED -- the models try when asked and fail to produce working kernels"
    elif dk <= 5:
        v = "NOT prompt-induced -- the models decline to write kernels even when asked"
    else:
        v = "AMBIGUOUS -- effect present but under the pre-set thresholds; do not claim it"
    print("  VERDICT: %s" % v)


def r3(pattern=None):
    """lineage-reset under passrate. Trajectory level: one value per run, never per round."""
    print("=" * 84)
    print("ROUTE 3 -- lineage minus reset under KA_SCORE=passrate")
    print("=" * 84)
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from clustered_stats import trajectory_ci
    except Exception:
        trajectory_ci = None
    pat = pattern or os.path.join(D, "r3_*")
    arms = {}
    for outdir in sorted(glob.glob(pat)):
        arm = "inject" if "_inject_" in outdir else "control"
        rounds = []
        for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
            d = _load(f)
            for rec in (d if isinstance(d, list) else [d]) if d else []:
                if isinstance(rec, dict) and "lineage_minus_reset" in rec:
                    rounds.append(rec["lineage_minus_reset"])
        if rounds:
            arms.setdefault(arm, []).append(st.mean(rounds))   # one number per TRAJECTORY
    if not arms:
        print("  no r3 output yet under %s" % pat); return
    for arm, vals in sorted(arms.items()):
        m = st.mean(vals)
        if trajectory_ci and len(vals) > 1:
            lo, hi = trajectory_ci(vals)
            print("  %-8s %+.3f [%+.3f, %+.3f]  n=%d trajectories" % (arm, m, lo, hi, len(vals)))
        else:
            print("  %-8s %+.3f  n=%d trajectories" % (arm, m, len(vals)))
    print("""
DECISION RULE (PREREGISTRATION.md Amendment 1):
  inject > 0 while control is flat -> the loop registers acquired coverage but does not
                                      generate it. That is a real finding.
  both flat                        -> instrument still dead. Publish no compounding claim
                                      from this hardware.
  NEVER pool with the A100 headroom boards: different metric AND different hardware.""")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    for fn, name in ((r1, "r1"), (r2, "r2"), (r3, "r3")):
        if what in (name, "all"):
            fn(arg) if what == name else fn()
            print()
