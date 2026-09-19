#!/usr/bin/env python3
"""Probe-as-intervention, analysed as a rigorous SECONDARY result (P4).

The probe chapter previously presented 22 rows as if they were 22 models, with per-run lifts ranging
from -0.27 to +0.45, and leaned on a pooled correctness AUC. Three things are wrong with that and all
three are fixable without new GPU time:

  1. UNIT.        The 22 rows are repeated probe-training/test draws over 13 distinct base checkpoints.
                  Draws sharing a checkpoint are not independent, so inference is done at the
                  CHECKPOINT level (clustered), not the row level.
  2. DENOMINATOR. Each row's lift is computed over n_test tasks, and n_test is 5-12. A "+0.45 lift"
                  is about five problems. We surface n_test in every table and weight nothing by it
                  implicitly.
  3. ESTIMAND.    Pooled correctness AUC can be high because the probe separates EASY tasks from HARD
                  ones while failing to rank candidates WITHIN a task -- and only within-task ranking
                  can explain a selection gain. Per-candidate probe scores were not retained, so the
                  within-task AUC cannot be recomputed here; we state that as a limitation rather than
                  letting the pooled number stand in for it.

We therefore report: the checkpoint-clustered mean lift with a t CI, a pre-declared equivalence test
at delta=0.05 correct-rate, the fraction of oracle headroom recovered, and the AUC-vs-lift diagnostic
(rows with AUC=1.0 and zero lift are the direct evidence that pooled AUC does not imply selection value).

  python3 scripts/probe_appendix.py                 # report
  python3 scripts/probe_appendix.py --tex paper/probe_appendix.tex --json docs/data/probe_rigor.json
"""
import json, glob, os, re, argparse, sys, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clustered_stats import analyze, t_ppf   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
DELTA = 0.05          # pre-declared practical-equivalence margin on the correct-rate scale


def base_of(model):
    """Collapse a model id to its base checkpoint (our run suffixes are not different models)."""
    m = (model or "").split("/")[-1]
    return re.sub(r"[-_]?(nt\d+|k\d+|wt|[abc])$", "", m, flags=re.I)


def load():
    runs = []
    for f in sorted(glob.glob(os.path.join(RAW, "probe_intervene_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        lift, nt = d.get("lift"), d.get("n_test") or 0
        if lift is None or nt <= 0:
            continue                                  # no usable test set -> not an observation
        rnd, prb = d.get("random_correct_rate"), d.get("probe_top1_correct_rate")
        orc = d.get("oracle_bestofK_rate")
        head = None
        if orc is not None and rnd is not None and orc - rnd > 1e-9:
            head = (prb - rnd) / (orc - rnd)           # fraction of the oracle gap the probe recovers
        runs.append(dict(file=os.path.basename(f), model=d.get("model", "?"), base=base_of(d.get("model")),
                         K=d.get("K"), n_test=nt, auc=d.get("auc"), lift=lift,
                         random=rnd, probe=prb, oracle=orc, headroom_recovered=head,
                         train_correct=d.get("train_correct"), train_incorrect=d.get("train_incorrect")))
    return runs


def cluster(runs, key="lift"):
    by = {}
    for r in runs:
        if r.get(key) is not None:
            by.setdefault(r["base"], []).append(r[key])
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = load()
    if not runs:
        print("no usable probe runs under", RAW)
        return 1

    by = cluster(runs)
    clusters = list(by.values())
    an = analyze(clusters, delta=DELTA)
    tr, rd = an["trajectory"], an["round"]

    print("=" * 104)
    print("PROBE-AS-INTERVENTION -- rigorous secondary analysis")
    print("=" * 104)
    print("%d usable runs over %d distinct base checkpoints; n_test per run = %d..%d tasks (median %d)"
          % (len(runs), len(by), min(r["n_test"] for r in runs), max(r["n_test"] for r in runs),
             int(st.median([r["n_test"] for r in runs]))))
    print()
    print("%-34s %4s %6s %7s %7s %7s %8s %8s" %
          ("base checkpoint / run", "n_t", "K", "AUC", "random", "probe", "lift", "head%"))
    print("-" * 104)
    for r in sorted(runs, key=lambda x: (x["base"], -x["lift"])):
        print("%-34s %4d %6s %7s %7.3f %7.3f %+8.3f %8s" %
              (r["file"].replace("probe_intervene_", "").replace(".json", "")[:34], r["n_test"], r["K"],
               ("%.3f" % r["auc"]) if r["auc"] is not None else "--",
               r["random"] or 0, r["probe"] or 0, r["lift"],
               ("%.0f%%" % (100 * r["headroom_recovered"])) if r["headroom_recovered"] is not None else "--"))
    print("-" * 104)

    lo, hi = tr["ci"]
    print("\nCHECKPOINT-CLUSTERED (primary): n=%d checkpoints  mean lift=%+.3f  95%% CI [%+.3f,%+.3f]"
          % (tr["n"], tr["mean"], lo, hi))
    print("  naive row-level (NOT primary):   n=%d rows        mean lift=%+.3f  95%% CI [%+.3f,%+.3f]"
          % (rd["n"], rd["mean"], rd["ci"][0], rd["ci"][1]))
    if an["icc"]:
        print("  ICC=%.3f  design effect=%.2f" % (an["icc"]["icc"], an["icc"]["design_effect"]))
    sig = "EXCLUDES zero" if (lo > 0 or hi < 0) else "INCLUDES zero"
    print("  -> the clustered CI %s." % sig)
    t = tr["tost"]
    print("  TOST at pre-declared delta=%.2f: %s (90%% CI [%+.3f,%+.3f])"
          % (DELTA, "EQUIVALENT to no effect" if t["equivalent"] else
             "NOT equivalent -- cannot claim 'no meaningful advantage' either", t["ci90"][0], t["ci90"][1]))

    # headroom recovered
    hr = cluster(runs, "headroom_recovered")
    if hr:
        anh = analyze(list(hr.values()))
        if anh["trajectory"]:
            h = anh["trajectory"]
            print("\nORACLE HEADROOM RECOVERED (probe-random)/(oracle-random): n=%d checkpoints  "
                  "mean=%.0f%%  95%% CI [%.0f%%,%.0f%%]"
                  % (h["n"], 100 * h["mean"], 100 * h["ci"][0], 100 * h["ci"][1]))

    # the diagnostic that matters: perfect pooled AUC with no selection gain
    perfect = [r for r in runs if r["auc"] is not None and r["auc"] >= 0.999]
    print("\nDIAGNOSTIC -- pooled AUC does not imply within-task ranking value:")
    for r in perfect:
        print("   %-30s AUC=%.3f but lift=%+.3f on %d tasks (train %s correct / %s incorrect)"
              % (base_of(r["model"])[:30], r["auc"], r["lift"], r["n_test"],
                 r["train_correct"], r["train_incorrect"]))
    if not perfect:
        print("   (no AUC=1.0 rows)")
    print("\nLIMITATION: per-candidate probe scores were not retained, so the WITHIN-TASK ranking AUC")
    print("cannot be recomputed from the stored artifacts. Establishing the mechanism requires a rerun")
    print("that logs per-candidate scores, with leave-task-out probe fitting and matched-candidate")
    print("comparators (uniform random / length-normalised likelihood / compile-validity filter).")

    out = {"delta": DELTA, "n_runs": len(runs), "n_checkpoints": len(by),
           "n_test_min": min(r["n_test"] for r in runs), "n_test_max": max(r["n_test"] for r in runs),
           "n_test_median": st.median([r["n_test"] for r in runs]),
           "clustered": {"n": tr["n"], "mean": round(tr["mean"], 4),
                         "ci95": [round(lo, 4), round(hi, 4)],
                         "ci90_tost": [round(x, 4) for x in t["ci90"]],
                         "equivalent": t["equivalent"], "excludes_zero": bool(lo > 0 or hi < 0)},
           "row_level_naive": {"n": rd["n"], "mean": round(rd["mean"], 4),
                               "ci95": [round(x, 4) for x in rd["ci"]]},
           "runs": runs}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print("\nwrote", a.json)
    if a.tex:
        open(a.tex, "w").write(build_tex(runs, by, an, out))
        print("wrote", a.tex)
    return 0


def esc(s):
    return str(s).replace("_", r"\_")


def build_tex(runs, by, an, out):
    """Built with token substitution rather than %-formatting: the body is dense in LaTeX % and $."""
    rows = ""
    for r in sorted(runs, key=lambda x: (x["base"], -x["lift"])):
        rows += "%s & %d & %s & %s & %.3f & %.3f & %+.3f & %s \\\\\n" % (
            esc(r["file"].replace("probe_intervene_", "").replace(".json", "")), r["n_test"], r["K"],
            ("%.3f" % r["auc"]) if r["auc"] is not None else "--", r["random"] or 0, r["probe"] or 0,
            r["lift"], ("%.0f" % (100 * r["headroom_recovered"])) + r"\%"
            if r["headroom_recovered"] is not None else "--")
    c = out["clustered"]
    nv = out["row_level_naive"]
    if c["excludes_zero"]:
        result = (r"the 95\% CI \textbf{excludes zero}: against a uniform-random selector at matched "
                  r"generation budget, probe-guided top-1 selection does lift the correct rate.")
    else:
        result = (r"the 95\% CI includes zero, so no selection advantage over uniform random is "
                  r"established.")
    equiv = ("" if c["equivalent"] else
             r" The result is also \emph{not} TOST-equivalent to zero at the pre-declared "
             r"$\delta{=}DELTA$, so neither a benefit nor its absence would have been concluded by "
             r"default --- the interval is simply wide.")
    body = r"""\section{Appendix: probe-as-intervention, analysed as a secondary result}
\label{app:probe}
A linear probe reads ``is this kernel correct'' out of mid-to-late layers at high pooled AUC. This appendix asks
whether that decodable signal converts into a \emph{selection} advantage, and reports it at the right unit with
the right denominator.

\paragraph{Three corrections to the earlier analysis.}
(1) \textbf{Unit.} The NRUNS usable runs are repeated probe-training/test draws over \textbf{NCKPT distinct base
checkpoints}; draws sharing a checkpoint are not independent, so we cluster on checkpoint.
(2) \textbf{Denominator.} Each run's lift is computed over $n_{\text{test}}$ tasks, and
$n_{\text{test}}\in[NTMIN,NTMAX]$ (median NTMED). A ``$+0.45$ lift'' is roughly five problems, so
$n_{\text{test}}$ is printed in every row.
(3) \textbf{Estimand.} A high \emph{pooled} correctness AUC can arise because the probe separates easy tasks
from hard ones while failing to rank candidates \emph{within} a task --- and only within-task ranking can
produce a selection gain. Per-candidate probe scores were not retained, so the within-task AUC
\textbf{cannot be recomputed} from the stored artifacts. We flag this as the binding limitation rather than
letting the pooled number stand in for it.

\paragraph{Result.} Clustered on base checkpoint, the mean lift over uniform-random selection is
$CMEAN\,[CLO,CHI]$ over $n{=}CN$ checkpoints --- RESULT EQUIV The naive row-level pooling would report
$NMEAN\,[NLO,NHI]$ over NN rows. The probe recovers HEAD\% of the oracle$-$random headroom on average
(95\% CI [HLO\%,HHI\%]).

\paragraph{What this does and does not establish.} \textbf{Uniform random is the weakest possible comparator.}
Beating it shows the probe carries \emph{some} usable signal; it does not show the probe is the best use of the
same compute, because the informative comparators were not run: mean/length-normalised log-likelihood reranking,
a compile-or-run validity filter, and execution agreement. Separately, and importantly, this
matched-\emph{generation} selection gain is a different quantity from matched-\emph{verification} harvesting
yield, where we measure no advantage. The coherent joint statement is: \emph{extra draws plus a cheap selector
find more correct kernels, but the tested probe does not make verification-constrained search economically
better.}

\paragraph{The diagnostic.} Several runs achieve pooled $\mathrm{AUC}{=}1.000$ with lifts from $+0.000$ to
$+0.250$, on training sets as small as 2--5 correct examples. Perfect pooled separation that does not predict
selection gain is direct evidence the pooled metric partly measures task difficulty rather than within-task
candidate quality --- a trap worth flagging for anyone reading correctness probes off a small bank.

\paragraph{Defensible claim.} \emph{Decodable correctness information yields a measurable but modest gain over
random selection, and does not automatically translate into useful search efficiency under a verification
budget.} Settling it needs a rerun on a family-split bank of $\geq$300 independent tasks that logs
per-candidate scores, with leave-task-out probe fitting, matched-candidate comparators, task-clustered
uncertainty, and a full generation/scoring/verification cost ledger.

\begin{table}[h]\centering\footnotesize
\caption{Probe-as-intervention, one row per run (not per model). $n_t$ = test tasks the row is computed on;
head\% = fraction of the oracle$-$random headroom recovered. Rows sharing a base checkpoint are repeated draws
and are clustered in the analysis.}
\begin{tabular}{lrrrrrrr}
\toprule
run & $n_t$ & $K$ & AUC & random & probe & lift & head\% \\
\midrule
ROWS\bottomrule
\end{tabular}\end{table}
"""
    hr = cluster(runs, "headroom_recovered")
    anh = analyze(list(hr.values())) if hr else None
    h = anh["trajectory"] if anh and anh.get("trajectory") else None
    subs = {
        "NRUNS": str(out["n_runs"]), "NCKPT": str(out["n_checkpoints"]),
        "NTMIN": str(out["n_test_min"]), "NTMAX": str(out["n_test_max"]), "NTMED": str(out["n_test_median"]),
        "CMEAN": "%+.3f" % c["mean"], "CLO": "%+.3f" % c["ci95"][0], "CHI": "%+.3f" % c["ci95"][1],
        "CN": str(c["n"]), "RESULT": result, "EQUIV": equiv.replace("DELTA", "%.2f" % out["delta"]),
        "NMEAN": "%+.3f" % nv["mean"], "NLO": "%+.3f" % nv["ci95"][0], "NHI": "%+.3f" % nv["ci95"][1],
        "NN": str(nv["n"]),
        "HEAD": ("%.0f" % (100 * h["mean"])) if h else "--",
        "HLO": ("%.0f" % (100 * h["ci"][0])) if h else "--",
        "HHI": ("%.0f" % (100 * h["ci"][1])) if h else "--",
        "ROWS": rows,
    }
    for k, v in subs.items():
        body = body.replace(k, v)
    return body


if __name__ == "__main__":
    raise SystemExit(main())
