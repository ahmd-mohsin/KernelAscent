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
import json, os, re, sys, glob, statistics as st

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
    """Substrate quality.

    READ THE COLUMNS CAREFULLY -- this report mixes two different scales, and conflating them is
    how the torch.compile misattribution happened:

      best_score  difficulty_filter calls LK._score(ok, sp_eager) with only two arguments, so
                  `ceiling` takes its LEGACY DEFAULT of 1.5. It is therefore speedup over EAGER
                  normalised to a fixed 1.5x anchor -- NOT the headroom-normalised score the
                  paper defines. best_score == 1.00 means sp_eager >= 1.5x. It does NOT mean the
                  task reached its roofline.
      ceiling     roofline headroom over torch.compile. Used ONLY for admission (>= 1.3x). It
                  does not enter best_score at all, so it cannot be cited as "room to move"
                  under this scoring.

    So the banks are compared on the one scale they share: how often the frozen base actually
    clears 1.5x over eager.
    """
    print("=" * 84)
    print("ROUTE 1 -- does the substrate leave the model room to move?")
    print("=" * 84)
    banks = paths or [("29-task", os.path.join(D, "bank_h100_report.json")),
                      ("456-DSL", os.path.join(D, "bank_dsl_h100_report.json"))]
    print("  (best_score = speedup over EAGER at a fixed 1.5x anchor; 1.00 means sp_eager >= 1.5x)")
    print("%-9s %7s %6s %11s %12s %11s" %
          ("bank", "scored", "kept", ">1.0x eager", ">=1.5x eager", "mean kept"))
    rows = {}
    for lbl, p in banks:
        d = _load(p)
        if not d:
            continue
        kept = [x for x in d if x.get("keep")]
        rows[lbl] = dict(
            n=len(d), kept=len(kept),
            gt51=sum(1 for x in d if x["best_score"] > 0.51),
            at15=sum(1 for x in d if x["best_score"] >= 1.0),
            mean=st.mean([x["best_score"] for x in kept]) if kept else 0.0)
        r = rows[lbl]
        print("%-9s %7d %6d %5d (%4.1f%%) %5d (%4.1f%%) %11.3f" %
              (lbl, r["n"], r["kept"], r["gt51"], 100 * r["gt51"] / r["n"],
               r["at15"], 100 * r["at15"] / r["n"], r["mean"]))

    if "456-DSL" not in rows or "29-task" not in rows:
        return
    a, b = rows["29-task"], rows["456-DSL"]
    fa, fb = 100 * a["at15"] / a["n"], 100 * b["at15"] / b["n"]
    print("""
DECISION RULE (fixed in advance): the substrate is usable iff the frozen base demonstrably
clears the anchor on a non-trivial fraction of tasks -- i.e. the speed dimension is reachable
at all, rather than every success landing exactly at parity.

  base reaches >= 1.5x over eager:   29-task {fa:.1f}%  ({a15}/{an})    456-DSL {fb:.1f}%  ({b15}/{bn})

VERDICT: {verdict}

Neither bank has a fat middle band -- both are bimodal, parity or well past the anchor. What
separates them is reachability: on the DSL bank the base clears 1.5x over eager {ratio:.0f}x more
often. So "these models cannot produce a faster kernel" is a property of the OLD BANK, not of
the models, and the 29-task bank cannot support a speed-scored compounding claim at all.

CAVEAT: the DSL references are generated, so beating eager by 1.5x may be easy for reasons that
do not transfer -- a naive reference is not the same as a strong baseline. Before any headroom
number from this bank is published it must be re-scored in KA_SCORE=compiled mode, where the
baseline is torch.compile and `ceiling` is actually used. The numbers above do not license a
claim about torch.compile.""".format(
        fa=fa, fb=fb, a15=a["at15"], an=a["n"], b15=b["at15"], bn=b["n"],
        ratio=(fb / fa if fa else float("inf")),
        verdict=("DSL bank is a usable substrate; the 29-task bank is not"
                 if fb > 10 and fb > 2 * fa else
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


def _modes_from_manifest(pattern):
    """Recover the score mode from the recorded launch command for runs that predate stamping.

    .jobman.tsv stores the exact command each cell was submitted with, so `env KA_SCORE=x` in
    it is real provenance rather than an assumption. Used only when the data carries no mode.
    """
    man = os.path.join(ROOT, ".jobman.tsv")
    if not os.path.exists(man):
        return set()
    cells = {os.path.basename(p).strip("_") for p in glob.glob(pattern)}
    modes = set()
    for line in open(man):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        name, cmd = parts[0], parts[3]
        # e.g. r3c-q15-s1 <-> outdir r3_control_q15_s1: match on the shared tail
        tag = name.replace("r3c-", "").replace("r3i-", "")
        if not any(tag.replace("-", "_") in c for c in cells):
            continue
        m = re.search(r"KA_SCORE=(\w+)", cmd)
        modes.add(m.group(1) if m else "eager")
    return modes


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
    modes = set()
    for outdir in sorted(glob.glob(pat)):
        arm = "inject" if "_inject_" in outdir else "control"
        rounds = []
        for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
            d = _load(f)
            for rec in (d if isinstance(d, list) else [d]) if d else []:
                if isinstance(rec, dict) and "lineage_minus_reset" in rec:
                    rounds.append(rec["lineage_minus_reset"])
                    # eval_tasks stamps score_mode into every round record, so the metric that
                    # produced a number travels WITH the number. Amendment 1 forbids pooling
                    # passrate with headroom, and a promise that is only in prose is one nobody
                    # can enforce -- this makes the file itself refuse.
                    for key in ("score_mode", "stats"):
                        v = rec.get(key)
                        if isinstance(v, dict):
                            v = v.get("score_mode")
                        if isinstance(v, str):
                            modes.add(v)
        if rounds:
            arms.setdefault(arm, []).append(st.mean(rounds))   # one number per TRAJECTORY

    # FAIL CLOSED. The first version only refused when it SAW a wrong mode, so a run with no
    # score_mode recorded passed by default -- and lab_compounding was not stamping it, which
    # made the whole check inert. An enforcement whose default is "allow" enforces nothing.
    if not modes:
        modes = _modes_from_manifest(pat)          # provenance fallback: the recorded command
        if modes:
            print("  score_mode: %s (from the launch command in .jobman.tsv -- these rounds"
                  % sorted(modes)[0])
            print("              predate score_mode stamping; newer runs carry it in the data)")
    if not modes:
        print("  !! REFUSING TO REPORT: no score_mode recorded in these rounds, and no launch")
        print("     command found for them. Amendment 1 forbids pooling passrate with")
        print("     headroom-scored runs, and an unlabelled run cannot be shown to comply.")
        return
    if modes != {"passrate"}:
        print("  !! REFUSING TO REPORT: these runs carry score_mode %s, not {'passrate'}."
              % sorted(modes))
        print("     PREREGISTRATION.md Amendment 1 binds passrate results to their own")
        print("     experiment set and forbids pooling them with headroom-scored runs.")
        return
    if modes:
        print("  score_mode: %s (verified, not assumed)" % sorted(modes)[0])
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


def e2(pattern=None):
    """E2 -- is T3's "frontier self-modification is one-shot" a finding or a harness cap?

    The published harness asked the model to return at most 12 strategy strings and showed 12
    back. Crucially the cap is a SOFT REQUEST plus a display truncation -- U["strategies"] is
    never actually truncated -- so what it really measures is instruction compliance. That makes
    the closed-model runs the evidence that matters, and they must be in the table beside the
    open-model re-runs rather than left out of it.

    The claim under test is already published, so the decision rule is fixed here in code.
    """
    print("=" * 84)
    print("E2 -- does the procedure plateau survive lifting the strategy cap?")
    print("=" * 84)
    pats = [pattern] if pattern else [
        os.path.join(ROOT, "data", "**", "track_c*.json"),
    ]
    rows, seen = [], set()
    for pat in pats:
        for f in sorted(glob.glob(pat, recursive=True)):
            d = _load(f)
            if not d:
                continue
            hist = d.get("history") or []
            if not hist:
                continue
            model = d.get("model", "?")
            cell = os.path.basename(os.path.dirname(f))
            key = (cell, model, len(hist), tuple(h.get("n_strategies", 0) for h in hist))
            if key in seen:                       # the same run pulled into two data dirs
                continue
            seen.add(key)
            ns = [h.get("n_strategies", 0) for h in hist]
            qs = [h["Qg"] for h in hist]
            q0 = d.get("Q0", 0.0)
            gain, r0 = qs[-1] - q0, qs[0] - q0
            cap = d.get("max_strategies")
            rows.append(dict(
                cell=cell, model=model, closed=("Qwen" not in model),
                cap=("unbounded" if cap == 0 else (cap if cap is not None else "12 (pub)")),
                rounds=len(hist), ns=ns, gain=gain,
                # A share is a ratio, so it is unstable when the denominator is small: runs
                # with |dQ| of 0.05 produced "992%" and "955%". Those are undefined, not large.
                # 0.10 is the smallest denominator at which the ratio is stable here.
                share=(r0 / gain if abs(gain) >= 0.10 else float("nan")),
                grew=(ns[-1] > ns[0]), nmax=max(ns) if ns else 0))
    live = [r for r in rows if r["rounds"] >= 3 and max(r["ns"]) > 0]
    if not live:
        print("  no interpretable E2/Track-C output yet"); return

    for grp, lbl in ((True, "PUBLISHED closed frontier models (cap requested = 12)"),
                     (False, "E2 open-model re-runs (Qwen-7B, cap varied)")):
        sel = [r for r in live if r["closed"] == grp]
        if not sel:
            continue
        print("\n%s" % lbl)
        print("  %-26s %-11s %6s %-26s %9s %7s" %
              ("cell/model", "cap", "rounds", "n_strategies per round", "total dQ", "r0 share"))
        for r in sorted(sel, key=lambda x: -x["nmax"]):
            name = (r["model"] if grp else r["cell"])[:26]
            print("  %-26s %-11s %6d %-26s %+9.3f %7s" %
                  (name, r["cap"], r["rounds"], str(r["ns"])[:26], r["gain"],
                   ("%.0f%%" % (100 * r["share"])) if r["share"] == r["share"] else "n/a"))

    closed = [r for r in live if r["closed"]]
    openr = [r for r in live if not r["closed"]]
    print("""
DECISION RULE (fixed in advance; the claim under test is already published):
  procedures sit AT the requested limit from round 0 and never grow -> the plateau is the
      HARNESS. "One-shot self-modification" is not earned and must be retracted.
  procedures grow across rounds and STILL plateau in Q             -> the plateau is REAL.
  no run ever approaches its limit                                  -> the cap was never
      binding; it cannot explain the plateau either way.""")
    if closed:
        pinned = [r for r in closed if r["nmax"] >= 11]
        never_grew = [r for r in closed if not r["grew"]]
        print("\n  CLOSED: %d/%d runs sit at 11-12 strategies; %d/%d never grew at all."
              % (len(pinned), len(closed), len(never_grew), len(closed)))
    if openr:
        grew = [r for r in openr if r["grew"]]
        over = [r for r in openr if r["nmax"] > 12]
        print("  OPEN:   %d/%d runs GREW their procedure; %d exceeded 12 strategies (max %d)."
              % (len(grew), len(openr), len(over), max(r["nmax"] for r in openr)))
    if openr:
        by = {}
        for r in openr:
            by.setdefault(str(r["cap"]), []).append(r["gain"])
        print("  mean total dQ by cap:  " +
              "   ".join("%s: %+.3f (n=%d)" % (k, st.mean(v), len(v))
                         for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))))
    if closed and openr:
        def share(rs):
            v = [r["share"] for r in rs if r["share"] == r["share"]]
            return (st.mean(v), len(v))
        cs, cn = share(closed); os_, on = share(openr)
        print("  round-0 share of total gain:  closed %.0f%% (n=%d)   open %.0f%% (n=%d)"
              % (100 * cs, cn, 100 * os_, on)
              + "\n     [runs with |dQ| < 0.10 excluded -- the ratio is undefined there, not large]")
        if len([r for r in closed if r["nmax"] >= 11]) >= 0.8 * len(closed) and \
           len([r for r in openr if r["grew"]]) >= 0.5 * len(openr):
            print("\n  VERDICT: the published one-shot claim is CONFOUNDED. Frontier models comply")
            print("           with the <=12 instruction exactly and are pinned from round 0, while")
            print("           open models under the same harness grow their procedures past 12.")
            print("           A procedure that cannot grow cannot be observed to keep improving.")
        else:
            print("\n  VERDICT: no clean separation -- do not claim the cap explains the plateau.")
    print("""
  LIMIT OF THIS TEST: the cap is a soft request, so the effect is instruction COMPLIANCE, and
  the open model does not comply reliably. The open re-runs therefore corroborate rather than
  directly replicate. Settling it properly needs closed models re-run at several caps.""")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    for fn, name in ((r1, "r1"), (r2, "r2"), (r3, "r3"), (e2, "e2")):
        if what in (name, "all"):
            fn(arg) if what == name else fn()
            print()
