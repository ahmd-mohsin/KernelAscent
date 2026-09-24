#!/usr/bin/env python3
"""Formal equivalence stats for the compounding null, at the TRAJECTORY level (P0.1).

Reads compounding_*/compounding.json `lineage_minus_reset` and reports, per model + pooled:
  * PRIMARY   trajectory-level mean (one number per run/seed), t-based 95% CI, TOST, BF01
  * secondary cluster-robust SE on the round-level data (clustered on trajectory)
  * legacy    naive round-level pooling -- reported ONLY so the correction is auditable

WHY THE CHANGE. The previous version pooled every round as an independent observation. Rounds
within a trajectory share a base checkpoint, seed, held split and an accumulated adapter, so the
naive n (228) massively overstated the evidence and inflated BF01. The independent replicate is
the trajectory. We now report the ICC and the effective sample size so the size of that overstatement
is visible rather than buried in a caveat.

  python3 scripts/equivalence_tost.py --dir data/trajectories
  python3 scripts/equivalence_tost.py --dir data/trajectories --min-rounds 3   # drop singletons
"""
import json, glob, os, argparse, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clustered_stats import analyze, fmt_row, HEADER   # noqa: E402

# CRITICAL: only canonical POST-FIX seeds count. Older/broken runs (qwen7, qwen3, *fix, *gentle,
# *min, q7) predate the SFT NaN-gradient fix and show lineage collapse-to-0 (C_lineage=0, n_ex=0
# from round 1) which pollutes the pooled estimate toward a spurious large-negative "finding".
# The post-fix seed naming is compounding_q{05,15,3,7,14}{new|sN}. Pass --all to override.
_CLEAN = re.compile(r"^compounding_q(05|15|3|7|14)(new|s\d+)$")

SIZE = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0}


def _size_of(model):
    for k, v in SIZE.items():
        if k in model:
            return v
    return 999.0


def load_clusters(d, field="lineage_minus_reset", allow_all=False, min_rounds=1):
    """-> {model: [[round values] per trajectory]}.  One inner list == one independent run."""
    groups = {}
    for f in sorted(glob.glob(os.path.join(d, "compounding_*", "compounding.json"))):
        tag = os.path.basename(os.path.dirname(f))
        if not allow_all and not _CLEAN.match(tag):
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        model = (j.get("model") or tag).split("/")[-1]
        vals = [h[field] for h in j.get("history", []) if h.get(field) is not None]
        if len(vals) < min_rounds:
            continue
        groups.setdefault(model, []).append(vals)
    return groups


def report(groups, delta, title, field):
    print("\n=== %s ===" % title)
    print("contrast: %s   equivalence margin delta=%.2f" % (field, delta))
    pooled = []
    per_model = {}
    for g in sorted(groups, key=lambda m: (_size_of(m), m)):
        cl = groups[g]
        pooled += cl
        a = analyze(cl, delta=delta)
        per_model[g] = a
    print("\n" + HEADER)
    for g in sorted(per_model, key=lambda m: (_size_of(m), m)):
        a = per_model[g]
        print(fmt_row(g + "  [PRIMARY traj]", a["trajectory"]))
        print(fmt_row("   legacy round-level", a["round"]))
        if a["icc"]:
            print("   %-27s ICC=%.3f  design_effect=%.2f  n_eff=%.1f" %
                  ("", a["icc"]["icc"], a["icc"]["design_effect"], a["icc"]["n_eff"]))
    ap = analyze(pooled, delta=delta)
    print("-" * 92)
    for k, lab in (("trajectory", "POOLED [PRIMARY trajectory]"),
                   ("crve", "POOLED [cluster-robust]"),
                   ("round", "POOLED [legacy round-level]")):
        print(fmt_row(lab, ap[k]))
    if ap["icc"]:
        print("%-30s ICC=%.3f  design_effect=%.2f  n_eff=%.1f  (naive n=%d)" %
              ("", ap["icc"]["icc"], ap["icc"]["design_effect"], ap["icc"]["n_eff"], ap["round"]["n"]))
    return per_model, ap


def to_json(per_model, pooled, delta, field, groups=None):
    def pack(a, clusters=None):
        if not a:
            return None
        # Always record the COUNTS, even when no estimator could be computed. A single-trajectory
        # cell has no t-interval, and packing it as all-None meant its row vanished from the
        # results table while its trajectory still counted toward the bolded pooled n -- the
        # table summed to 56 against a printed 57 with nothing to show the reader why.
        out = {}
        if clusters is not None:
            out["n_trajectories"] = len(clusters)
            out["n_rounds_total"] = sum(len(c) for c in clusters)
        for k in ("trajectory", "crve", "round"):
            e = a[k]
            if not e:
                out[k] = None
                continue
            out[k] = {"n": e["n"], "G": e["G"], "mean": round(e["mean"], 5),
                      "n_rounds": e.get("n_rounds", e["n"]),
                      "se": round(e["se"], 5), "df": e["df"],
                      "ci95": [round(x, 5) for x in e["ci"]],
                      "ci90_tost": [round(x, 5) for x in e["tost"]["ci90"]],
                      "equivalent": e["tost"]["equivalent"],
                      "p_tost": round(e["tost"]["p_tost"], 5),
                      "bf01": (round(e["bf01"], 3) if e["bf01"] is not None else None)}
        if a["icc"]:
            out["icc"] = {k: round(v, 4) for k, v in a["icc"].items()}
        return out
    return {"contrast": field, "delta": delta,
            "primary_estimand": "trajectory",
            "note": ("Trajectory is the independent replicate: rounds within a run share a base "
                     "checkpoint, seed, held split and accumulated adapter. Round-level pooling is "
                     "reported only to make the correction auditable."),
            "models": {m: pack(a, (groups or {}).get(m)) for m, a in per_model.items()},
            "pooled": pack(pooled)}


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dir", default=os.path.join(root, "data", "trajectories"))
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--field", default="lineage_minus_reset")
    ap.add_argument("--min-rounds", type=int, default=1,
                    help="drop trajectories with fewer than this many rounds (sensitivity check)")
    ap.add_argument("--all", action="store_true", help="include non-canonical/broken runs")
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    a = ap.parse_args()
    groups = load_clusters(a.dir, a.field, a.all, a.min_rounds)
    if not groups:
        print("no compounding data found under", a.dir)
        return 1
    title = "Lineage - reset (compounding)" + (" [min_rounds=%d]" % a.min_rounds if a.min_rounds > 1 else "")
    per_model, pooled = report(groups, a.delta, title, a.field)
    print("\nPRIMARY estimand = trajectory. TOST EQUIV at delta=%.2f means the compounding advantage "
          "is bounded within +/-%.2f." % (a.delta, a.delta))
    print("BF01 > 3 = moderate, > 10 = strong evidence for the null. Computed at the estimand's own n,")
    print("so the trajectory-level BF01 is the one that may be quoted.")
    if a.out:
        json.dump(to_json(per_model, pooled, a.delta, a.field, groups), open(a.out, "w"), indent=1)
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
