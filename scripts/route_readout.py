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
    if "pre_ccfix" in os.path.basename(p):
        print("  !! refusing %s -- graded before the triton/CC fix" % os.path.basename(p))
        return None
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        print("  !! %s exists but will not parse (%s) -- truncated run?" % (os.path.basename(p), e))
        return None


def t2k(pattern=None):
    """T2-kernel: compounding on kernel AUTHORING, the task the models have not saturated.

    Written while 35 of 36 cells are still running, so the rule is not chosen from the answer.

    The contrast is `lineage - reset`, which is the PUBLISHED one and NOT the registered primary
    (`self - fresh_frozen`, reported by prereg()). It is kept because the kernel-authoring set is
    a like-for-like re-run of the published protocol on a different task, and because it carries
    a confound worth stating every time: lineage accumulates R x sft_steps against reset's 1 x,
    so it mixes accumulation with total gradient steps.
    """
    print("=" * 84)
    print("T2-kernel -- lineage minus reset on kernel authoring")
    print("=" * 84)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from clustered_stats import trajectory_level, ci as _ci, tost as _tost
    except Exception:
        trajectory_level = _ci = _tost = None
    pat = pattern or os.path.join(D, "t2k_*")
    arms, incomplete, resumed, inj = {}, [], 0, []
    for outdir in sorted(glob.glob(pat)):
        d = _load(os.path.join(outdir, "compounding.json"))
        if not d:
            continue
        rows = d.get("history") or []
        v = [r["lineage_minus_reset"] for r in rows if r.get("lineage_minus_reset") is not None]
        if not v:
            continue
        arm = "inject" if "_inject_" in outdir else "control"
        arms.setdefault(arm, []).append(st.mean(v))
        if len(v) < 6:
            incomplete.append((os.path.basename(outdir), len(v)))
        if d.get("resumed_at"):
            resumed += 1
        if arm == "inject":
            inj += [r.get("injected_tasks") for r in rows if r.get("injected_tasks") is not None]
    if not arms:
        print("  no t2k output yet"); return
    if incomplete:
        print("  !! %d cell(s) INCOMPLETE (a mid-run mean is not a trajectory mean):" % len(incomplete))
        for n, k in sorted(incomplete)[:6]:
            print("       %-28s %d/6 rounds" % (n, k))
        if len(incomplete) > 6:
            print("       ... and %d more" % (len(incomplete) - 6))
    if resumed:
        print("  !! %d trajectory(ies) resumed mid-run; LoRA state was not restored." % resumed)
    for arm, vals in sorted(arms.items()):
        est = trajectory_level([[v] for v in vals]) if (trajectory_level and len(vals) > 1) else None
        if est and _ci:
            lo, hi = _ci(est)
            print("  %-8s %+.3f [%+.3f, %+.3f]  n=%d trajectories  CI %s"
                  % (arm, st.mean(vals), lo, hi, len(vals),
                     "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"))
        else:
            print("  %-8s %+.3f  n=%d" % (arm, st.mean(vals), len(vals)))
    if inj:
        fired = sum(1 for x in inj if x)
        print("  positive control: injection fired in %d of %d inject-arm rounds (median %d tasks)"
              % (fired, len(inj), int(st.median(inj))))
        if fired < 0.3 * len(inj):
            print("     ^ the control stopped firing early -- a control that runs out of work has")
            print("       not passed, it has stopped testing. Treat a flat inject arm as UNINFORMATIVE.")
    print("""
DECISION RULE (fixed while 35 of 36 cells were still running):
  n < 8 per arm                      -> UNDERPOWERED; report the interval, claim no equivalence
  CI excludes 0, positive, control fires throughout
                                     -> compounding on a task the models have NOT saturated --
                                        the result this benchmark exists to produce
  CI includes 0 and the control fired -> a real bounded null on kernel authoring
  CI includes 0 and the control did NOT fire
                                     -> uninformative; publish no compounding claim
  ALWAYS report alongside: this contrast gives lineage R x the gradient steps of reset, so a
  positive is evidence that ACCUMULATION helps, not that recursion does. The registered primary
  (self - fresh_frozen) is the one without that confound -- see prereg().""")


def prereg(pattern=None):
    """The PRE-REGISTERED T2 primary: self - fresh_frozen, under the registered scorer.

    Two things the paper got wrong at once, closed by one experiment set:
      * PREREGISTRATION.md registers `self - fresh_frozen` as the T2 primary -- a frozen
        producer trained on the same fresh data each round, which isolates "the improver got
        better" from "more data". The paper reported `lineage - reset` and never declared the
        substitution. `lineage - reset` also confounds accumulation with total gradient steps
        (R x versus 1 x); `self - fresh_frozen` does not.
      * The registration defines the score over the torch.compile baseline with the per-task
        roofline ceiling. Every published number used eager speedup against a fixed 1.5x anchor,
        because `ceiling` defaults to 1.5 when unsupplied. These runs use KA_SCORE=compiled with
        KA_ROOF_ARCH=h100.

    Written before the data completed, so the decision rule is not chosen from the answer.
    """
    print("=" * 84)
    print("PREREG -- self minus fresh_frozen (the REGISTERED T2 primary)")
    print("=" * 84)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from clustered_stats import trajectory_level, ci as _ci, tost as _tost
    except Exception:
        trajectory_level = _ci = _tost = None
    pat = pattern or os.path.join(D, "prereg_*")
    traj, rounds_seen, partial, resumed = [], 0, [], []
    for outdir in sorted(glob.glob(pat)):
        d = _load(os.path.join(outdir, "weight_rsi.json"))
        if not d:
            continue
        rows = d.get("rounds") or d.get("history") or []
        vals = [r["delta_self_minus_fresh"] for r in rows
                if r.get("delta_self_minus_fresh") is not None]
        if not vals:
            continue
        traj.append(st.mean(vals)); rounds_seen += len(vals)
        if d.get("resumed_at"):
            resumed.append((os.path.basename(outdir), d["resumed_at"]))
        if len(vals) < 5:
            partial.append((os.path.basename(outdir), len(vals)))
    if not traj:
        print("  no prereg output yet"); return
    if resumed:
        # The pre-registration fixes the TRAJECTORY as the unit of replication. A resumed run
        # continues its round sequence but re-learns from recorded data, because LoRA adapter
        # state is not checkpointed -- so it is not the same object as an uninterrupted run and
        # must be identifiable rather than pooled silently.
        print("  !! %d trajectory(ies) RESUMED mid-run (adapter state not restored):" % len(resumed))
        for n, at in resumed:
            print("       %-26s resumed at round %d" % (n, at))
        print("     Report these separately, or re-run them, before treating the pooled")
        print("     estimate as %d independent uninterrupted trajectories." % len(traj))
    if partial:
        print("  !! %d cell(s) INCOMPLETE -- a mid-run mean is not a trajectory mean:" % len(partial))
        for n, k in partial:
            print("       %-24s %d/5 rounds" % (n, k))
    m = st.mean(traj)
    est = trajectory_level([[v] for v in traj]) if (trajectory_level and len(traj) > 1) else None
    if est and _ci:
        lo, hi = _ci(est)
        t = _tost(est, 0.05) if _tost else None
        eq = (t.get("equivalent") if isinstance(t, dict) else None)
        print("\n  self - fresh_frozen : %+.3f [%+.3f, %+.3f]   n=%d trajectories (%d rounds)"
              % (m, lo, hi, len(traj), rounds_seen))
        print("  CI %s zero | TOST at delta=0.05: %s"
              % ("EXCLUDES" if (lo > 0 or hi < 0) else "includes",
                 "EQUIVALENT" if eq else ("not equivalent" if eq is not None else "n/a")))
    else:
        print("\n  self - fresh_frozen : %+.3f   n=%d trajectories (too few for an interval)"
              % (m, len(traj)))
    print("""
DECISION RULE (PREREGISTRATION.md Amendments 3 and 4, fixed before these runs landed):
  n < 8 per arm                      -> UNDERPOWERED. Report the interval, make no
                                        equivalence claim. MDE at n=4 is 0.120 against a
                                        margin of 0.05.
  CI excludes 0, positive            -> the improver genuinely got better, on the registered
                                        contrast and the registered scorer
  CI includes 0 and TOST equivalent  -> a bounded null on the REGISTERED primary, which is a
                                        stronger statement than the published one because it
                                        is not confounded with total gradient steps
  disagrees in sign with lineage-reset -> the REGISTERED contrast is reported as primary and
                                        the disagreement is reported, not resolved toward
                                        whichever is more interesting""")


def t1(pattern=None):
    """T1-kernel capability curve -- the registered falsifier for the whole program.

    Amendment 2: if the verified-kernel rate is at or near zero at EVERY scale including 14B,
    the task is unreachable rather than unsaturated, and no compounding claim is published.

    make_teacher_kernels writes incrementally per task, so a file existing does not mean the run
    finished. A partial file read as final is how a 12/29 cell gets reported as a capability
    number -- the cell is marked PARTIAL unless the task count matches the bank.
    """
    print("=" * 84)
    print("T1-kernel -- capability curve on kernel AUTHORING (registered falsifier)")
    print("=" * 84)
    scales = [("0.5B", "q05"), ("1.5B", "q15"), ("3B", "q3"), ("7B", "q7"), ("14B", "q14")]
    print("%-6s %-9s %8s %8s %8s %9s %9s %8s" %
          ("scale", "status", "solved", "kernels", "custom", "med spd", "max spd", ">1.05x"))
    done = []
    for lbl, tag in scales:
        f = pattern.replace("TAG", tag) if pattern else os.path.join(D, "t1k_%s.json" % tag)
        d = _load(f)
        if not d:
            print("%-6s %-9s   (not harvested yet)" % (lbl, "-")); continue
        ks = d.get("kernels") or {}
        n_tasks = d.get("n_tasks") or 0
        # `kernels` holds only SOLVED tasks, so its length is coverage and says nothing about
        # progress. Newer artifacts record it explicitly; older ones cannot be judged, and
        # "unknown" must not be reported as "complete".
        if "complete" in d:
            complete, prog = bool(d["complete"]), d.get("tasks_attempted")
        else:
            complete, prog = None, None
        allk = [k for v in ks.values() for k in v]
        solved = len([n for n, v in ks.items() if v])
        status = {True: "complete", False: "PARTIAL", None: "unknown"}[complete]
        if prog and not complete:
            status = "%d/%d" % (prog, n_tasks)
        if not allk:
            print("%-6s %-9s %4d/%-3d %8d %8s %9s %9s %8s"
                  % (lbl, status, solved, n_tasks, 0, "-", "-", "-", "-")); continue
        cust = sum(1 for k in allk if any(t in k["code"] for t in
                                          ("triton", "load_inline", "__global__")))
        sp = [k["speedup_eager"] for k in allk]
        # ATTEMPT vs SUCCESS. Without this the solve-rate column is a compliance metric: a
        # plain-torch rewrite verifies trivially, so a model that declines the task outscores
        # one that attempts it. Newer artifacts carry per-task attempt counts.
        att = d.get("attempts") or {}
        if att:
            cand = sum(v.get("n_candidates", 0) for v in att.values())
            tried = sum(v.get("n_attempted_kernel", 0) for v in att.values())
            kok = sum(v.get("n_kernel_verified", 0) for v in att.values())
            extra = "  attempt %.0f%%  verify|attempt %.1f%%" % (
                100 * tried / max(cand, 1), 100 * kok / max(tried, 1))
        else:
            extra = "  (attempt rate not recorded)"
        print("%-6s %-9s %4d/%-3d %8d %7.0f%% %9.2f %9.2f %8d%s"
              % (lbl, status, solved, n_tasks, len(allk), 100 * cust / len(allk),
                 st.median(sp), max(sp), sum(1 for x in sp if x > 1.05), extra))
        if complete:
            done.append((lbl, solved, n_tasks, len(allk), cust, sp))

    if not done:
        print("\n  no cell is confirmed COMPLETE yet. Cells marked 'unknown' predate progress")
        print("  stamping -- check the job log's TEACHER COVERAGE line before using them.")
        return
    print("""
DECISION RULE (PREREGISTRATION.md Amendment 2, fixed before these ran):
  verified rate at or near zero at EVERY scale -> the task is UNREACHABLE, not unsaturated.
      The compounding contrast is uninterpretable for the same reason as before and no
      compounding claim is published from it.
  otherwise -> kernel authoring is a live substrate and T2-kernel is interpretable.""")
    worst = max(s / t for _, s, t, _, _, _ in done)
    fast = sum(1 for _, _, _, _, _, sp in done for x in sp if x > 1.05)
    print("\n  OBSERVED over %d complete cell(s): best solve-rate %.0f%%, %d kernel(s) above 1.05x"
          % (len(done), 100 * worst, fast))
    print("  FALSIFIER: %s" % ("NOT triggered -- the task is reachable" if worst > 0.05
                               else "TRIGGERED -- publish no compounding claim"))
    print("""
  Report correctness-reachability and speed-reachability SEPARATELY. A model that is correct at
  parity is indistinguishable, in a single score, from one that cannot compete at all -- which
  is how the 0.50 spike went unexplained for weeks.""")


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
    # Results graded before the CC fix are archived as *.pre_ccfix.json and MUST NOT be read
    # here: triton could not verify under that environment, so any kernel-arm number from it is
    # an instrument artifact. A stale local copy of one was very nearly reported as a result.
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
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from clustered_stats import trajectory_level, ci as _ci, tost as _tost
    except Exception:
        trajectory_level = _ci = _tost = None
    pat = pattern or os.path.join(D, "r3_*")
    arms = {}
    modes = set()
    bon_rounds, bon_first, bon_last, sat_hits, sat_tot, held_n = [], [], [], 0, 0, None
    for outdir in sorted(glob.glob(pat)):
        arm = "inject" if "_inject_" in outdir else "control"
        rounds = []
        for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
            d = _load(f)
            # lab_compounding writes {"history": [...round rows...]}, not a bare list. Treating
            # the dict as a single record found zero rounds and printed "no output yet" while
            # the file sat right there -- the same blindness as the audit gates, in a reader.
            if isinstance(d, dict):
                recs = d.get("history") or d.get("rounds") or [d]
            elif isinstance(d, list):
                recs = d
            else:
                recs = []
            for rec in recs:
                if isinstance(rec, dict) and "lineage_minus_reset" in rec:
                    rounds.append(rec["lineage_minus_reset"])
                    bon_rounds.append(rec.get("C_bestofN"))
                    sat_tot += 1
                    sat_hits += 1 if (rec.get("C_lineage", 0) or 0) >= 1.0 else 0
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
            cb = [r for r in bon_rounds[-len(rounds):] if r is not None]
            if len(cb) > 1:
                bon_first.append(cb[0]); bon_last.append(cb[-1])

    # FAIL CLOSED. The first version only refused when it SAW a wrong mode, so a run with no
    # score_mode recorded passed by default -- and lab_compounding was not stamping it, which
    # made the whole check inert. An enforcement whose default is "allow" enforces nothing.
    if not modes:
        modes = _modes_from_manifest(pat)          # provenance fallback: the recorded command
        if modes:
            print("  score_mode: %s (from the launch command in .jobman.tsv -- these rounds"
                  % sorted(modes)[0])
            print("              predate score_mode stamping; newer runs carry it in the data)")
            from_manifest = True
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
    if modes and not locals().get("from_manifest"):
        print("  score_mode: %s (verified from the round records)" % sorted(modes)[0])
    if not arms:
        print("  no r3 output yet under %s" % pat); return
    # GUARD 1: best-of-N is DEGENERATE under passrate and must not be reported.
    # Its mechanism is taking the MAX over k*(r+1) draws; pass-rate is a MEAN, which is
    # invariant to how many draws you take. Measured: as the budget grew 5x, C_bestofN moved
    # -0.024. So "lineage - bestofN" here is not a matched-budget search comparison -- it is a
    # trained model against a baseline whose mechanism the metric switched off, and it would
    # spuriously REVERSE the published "search beats training" result.
    bon = [r for r in bon_rounds if r is not None]
    if bon:
        print("  !! NOT REPORTING lineage-minus-bestofN: pass-rate is a mean over draws, so the")
        print("     best-of-N arm cannot benefit from its budget (C_bestofN moved %+.3f while the"
              % (st.mean(bon_last) - st.mean(bon_first)) if bon_first else "")
        print("     budget grew 5x). That contrast is invalid under this metric.")

    # GUARD 2: saturation at the TOP is the same absorbing-state failure as 0.50 was at the
    # bottom. Report it beside the effect rather than under it.
    if sat_tot:
        pct = 100.0 * sat_hits / sat_tot
        print("  ceiling: C_lineage == 1.000 in %d/%d rounds (%.0f%%) on a %d-task held set"
              % (sat_hits, sat_tot, pct, held_n or 5))
        if pct > 30:
            print("     ^ above 30%: once lineage pins at the ceiling, lineage-reset measures only")
            print("       how far RESET fell below it. Treat the magnitude as a lower bound and do")
            print("       not read a trend across rounds.")

    # Trajectory is the unit of replication: rounds inside a run share a checkpoint, a seed,
    # a held split and an accumulated adapter. Each value here is already one run's mean.
    for arm, vals in sorted(arms.items()):
        m = st.mean(vals)
        est = trajectory_level([[v] for v in vals]) if trajectory_level else None
        if est and _ci:
            lo, hi = _ci(est)
            excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
            print("  %-8s %+.3f [%+.3f, %+.3f]  n=%d trajectories  CI %s"
                  % (arm, m, lo, hi, len(vals), excl))
        else:
            print("  %-8s %+.3f  n=%d trajectories" % (arm, m, len(vals)))
    if len(arms) == 2 and trajectory_level and _ci:
        a, b = arms.get("control", []), arms.get("inject", [])
        if a and b:
            d = st.mean(b) - st.mean(a)
            print("  %-8s %+.3f  (inject minus control -- does added coverage change anything?)"
                  % ("delta", d))
    print("""
DECISION RULE (PREREGISTRATION.md Amendment 1):
  inject > 0 while control is flat -> the loop registers acquired coverage but does not
                                      generate it. That is a real finding.
  both flat                        -> instrument still dead. Publish no compounding claim
                                      from this hardware.
  NEVER pool with the A100 headroom boards: different metric AND different hardware.

  AND THE CONFOUND THAT LIMITS WHAT A POSITIVE MEANS: lineage keeps its adapter and trains
  every round (R x sft_steps); reset re-initialises and trains on one round (1 x sft_steps).
  So this contrast mixes ACCUMULATION with MORE GRADIENT STEPS. Under a metric that scores
  reliability of correctness, "more SFT on verified-correct outputs raises the rate of correct
  outputs" is near-tautological. A positive here is evidence that accumulation helps
  correctness reliability -- NOT evidence of recursive self-improvement.""")


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
    rows, seen, skipped = [], set(), []
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
            # CONTAMINATION GATE. Runs whose data dir sat on the inode-exhausted /scratch have
            # Q0 = 0.000 and long runs of exactly-zero rounds: build_ref_c raised EDQUOT and the
            # grader returned False for every candidate (INTUITIONS 2c). Their "gain" is
            # recovery from a spurious zero baseline, and averaging them produced a +0.482
            # dose-response that reached the paper. Documented, then used anyway -- so the
            # exclusion is now mechanical rather than remembered.
            zeros = sum(1 for q in qs if q == 0.0)
            if q0 == 0.0 or zeros >= 2:
                skipped.append((cell, q0, zeros))
                continue
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
    if skipped:
        print("  !! EXCLUDED %d run(s) with the EDQUOT zero signature (Q0=0 or >=2 zero rounds):"
              % len(skipped))
        for cell, q0, z in skipped:
            print("       %-26s Q0=%.3f  zero-rounds=%d" % (cell, q0, z))
        print("     Their apparent gain is recovery from a spurious zero baseline, not a result.")
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
    for fn, name in ((prereg, "prereg"), (t2k, "t2k"), (t1, "t1"), (r1, "r1"), (r2, "r2"), (r3, "r3"), (e2, "e2")):
        if what in (name, "all"):
            fn(arg) if what == name else fn()
            print()
