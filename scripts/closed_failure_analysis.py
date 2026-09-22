#!/usr/bin/env python3
"""Where do the CLOSED-source RSI runs actually fail? Answered from stored trajectories.

No GPU and no API key needed -- this reads data/trajectories/trackc_*/track_c.json and
selfplay_closed, and classifies each run by WHERE in the loop it stopped improving.

The headline T3 numbers (Astra +0.556, Sonnet-5 +0.401) are real gains over a frozen
procedure, but the per-round traces show the gain is one-shot. This script asks the question
that matters for the paper: is the plateau a property of the MODEL, or of the HARNESS?

Two harness bounds are load-bearing and were not previously reported:

  STRATEGY CAP   lab_track_c.py:51 instructs the model to return "<=12" strategy strings,
                 and :34 shows it only the first 12. So the editable procedure has a FIXED
                 CAPACITY of 12 slots. Every healthy run fills it in round 0 and stays there.
  ARCHIVE BOUND  the archive can only ever hold the training tasks (n_train, default 20),
                 and only the first 3 entries are shown to the model as exemplars (:38).

A procedure that is full at round 0 has nowhere to grow, so "the model stopped improving its
procedure" and "the harness stopped letting it" are CONFOUNDED in the current design. That is
a fixable measurement bug, not a finding about frontier models.

  python3 scripts/closed_failure_analysis.py
  python3 scripts/closed_failure_analysis.py --json docs/data/closed_failure.json
"""
import json, glob, os, argparse, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRAT_CAP = 12      # lab_track_c.py:51  ("<=12")
ARCHIVE_CAP = 20    # lab_track_c.py:182 (--n-train default)


def load_trackc():
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/trajectories/trackc_*/track_c.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        h = d.get("history") or []
        if not h:
            continue
        runs.append({"tag": os.path.basename(os.path.dirname(f)).replace("trackc_", ""),
                     "model": (d.get("model") or "?").split("/")[-1], "Q0": d.get("Q0"),
                     "mode": d.get("mode"), "history": h})
    return runs


def classify(r):
    """Where did this run stop, and is the stop attributable to the model or the harness?"""
    h = r["history"]
    n = len(h)
    q0 = r["Q0"] or 0.0
    deltas = [x.get("delta_vs_base") or 0.0 for x in h]
    strat = [x.get("n_strategies") or 0 for x in h]
    arch = [x.get("n_archive") or 0 for x in h]
    fg = [x.get("F_g") or 0.0 for x in h]

    # a run with no strategies at all never exercised the mechanism under test
    if max(strat) == 0:
        return ("PARSE-FAILURE",
                "n_strategies=0 in every round: the model's output was never parsed into a "
                "strategy list, so the self-modification channel never engaged. Not a result.")
    if n < 3:
        return ("TOO-SHORT", "only %d completed round(s); a ceiling cannot be separated from an "
                             "interrupted run" % n)

    r0_gain = deltas[0]
    later_gain = max(deltas) - deltas[0]
    total = max(deltas)
    frac_r0 = (r0_gain / total) if total > 1e-9 else 1.0

    # did the procedure hit its capacity, and when?
    cap_round = next((i for i, s in enumerate(strat) if s >= STRAT_CAP), None)
    arch_cap_round = next((i for i, a in enumerate(arch) if a >= ARCHIVE_CAP), None)

    if cap_round is not None and cap_round <= 1 and later_gain < 0.05:
        return ("HARNESS-CAPPED",
                "strategy list hit its %d-slot cap at round %d and later gain is only %+.3f. The "
                "procedure was FULL before it could grow, so model plateau and harness ceiling "
                "are not separable here." % (STRAT_CAP, cap_round, later_gain))
    if q0 > 0.8 and total < 0.1:
        return ("NO-HEADROOM",
                "Q0=%.3f already near ceiling; total gain %+.3f leaves nothing to measure" % (q0, total))
    if frac_r0 > 0.7:
        return ("ONE-SHOT",
                "%.0f%% of the gain arrives at round 0 (%+.3f of %+.3f); subsequent rounds add "
                "%+.3f" % (100 * frac_r0, r0_gain, total, later_gain))
    return ("SUSTAINED", "gain accumulates across rounds (round-0 share %.0f%%)" % (100 * frac_r0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = load_trackc()
    if not runs:
        print("no trackc trajectories found")
        return 1

    print("=" * 100)
    print("CLOSED-SOURCE T3 (procedure-RSI): where each run actually stopped")
    print("=" * 100)
    print("%-12s %6s %7s %7s %8s %9s  %s" % ("model", "Q0", "r0gain", "total", "strat", "archive", "verdict"))
    print("-" * 100)
    out = []
    for r in runs:
        h = r["history"]
        d = [x.get("delta_vs_base") or 0.0 for x in h]
        s = [x.get("n_strategies") or 0 for x in h]
        ar = [x.get("n_archive") or 0 for x in h]
        v, why = classify(r)
        print("%-12s %6.3f %+7.3f %+7.3f %4d/%-3d %5d/%-3d  %s"
              % (r["tag"][:12], r["Q0"] or 0, d[0], max(d), max(s), STRAT_CAP, max(ar), ARCHIVE_CAP, v))
        print("             %s" % why)
        out.append({"tag": r["tag"], "model": r["model"], "Q0": r["Q0"], "rounds": len(h),
                    "r0_gain": round(d[0], 4), "total_gain": round(max(d), 4),
                    "max_strategies": max(s), "max_archive": max(ar),
                    "verdict": v, "why": why})

    print("-" * 100)
    from collections import Counter
    c = Counter(o["verdict"] for o in out)
    print("verdicts:", dict(c))

    usable = [o for o in out if o["verdict"] not in ("PARSE-FAILURE", "TOO-SHORT")]
    print("\n%d of %d runs are interpretable at all." % (len(usable), len(out)))
    if usable:
        capped = [o for o in usable if o["max_strategies"] >= STRAT_CAP]
        print("%d of those %d hit the %d-strategy cap." % (len(capped), len(usable), STRAT_CAP))
        print("mean round-0 share of total gain: %.0f%%"
              % (100 * st.mean([(o["r0_gain"] / o["total_gain"]) if o["total_gain"] > 1e-9 else 1.0
                                for o in usable])))

    print("""
WHAT THIS MEANS FOR THE PAPER
  The T3 gains are real, but "frontier models improve their procedure once and then plateau"
  is NOT established by this data, because the procedure under test has a fixed capacity of
  %d strategy slots that every healthy run fills in round 0. Model-plateau and
  harness-ceiling are confounded.

  The fix is cheap and does not need a GPU-heavy rerun: re-run T3 with the strategy cap
  raised (or removed) and the archive exemplar count increased, and see whether the gain
  still stops. If it does, the plateau is a model property and the claim is earned. If it
  keeps climbing, the current 'one-shot' framing is a measurement artifact.

  Two runs (deepseek-v3.2, kimi) never parsed a strategy list at all -- their numbers,
  including the widely-quoted -0.234 'self-degrade', are parse failures, not findings.
""" % STRAT_CAP)

    if a.json:
        json.dump({"strategy_cap": STRAT_CAP, "archive_cap": ARCHIVE_CAP,
                   "note": ("Harness bounds from lab_track_c.py: the improve prompt asks for <=12 "
                            "strategies and only 12 are shown back; the archive is bounded by "
                            "--n-train (default 20) with only 3 exemplars surfaced."),
                   "runs": out, "verdicts": dict(c)}, open(a.json, "w"), indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
