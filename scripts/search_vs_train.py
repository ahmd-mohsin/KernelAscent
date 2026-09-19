#!/usr/bin/env python3
"""Finding (i): at MATCHED cumulative generation budget, does best-of-N SEARCH beat lineage self-TRAINING?

lab_compounding gives per-round C_lineage and C_bestofN where best-of-N uses budget k*(r+1) (== the
cumulative generation the lineage arm spent), so the comparison is compute-matched by construction.

TRAJECTORY-LEVEL (P0.1). As with the compounding TOST, the independent replicate is the trajectory,
not the round. We report the trajectory-level mean as PRIMARY, a cluster-robust SE as a secondary
check, and the legacy round-level pooling only so the correction is auditable. `search_wins_frac` is
reported both per-round (descriptive) and per-trajectory (fraction of RUNS whose mean favours search).
"""
import json, glob, os, argparse, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clustered_stats import analyze, fmt_row, HEADER   # noqa: E402
from equivalence_tost import _size_of                  # noqa: E402

# same canonical post-fix allowlist as equivalence_tost (7B/14B included when present)
_CLEAN = re.compile(r"^compounding_q(05|15|3|7|14)(new|s\d+)$")


def load(d, allow_all=False, min_rounds=1):
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
        vals = [h["lineage_minus_bestofN"] for h in j.get("history", [])
                if h.get("lineage_minus_bestofN") is not None]
        if len(vals) < min_rounds:
            continue
        groups.setdefault(model, []).append(vals)
    return groups


def wins(clusters):
    rounds = [x for c in clusters for x in c]
    traj = [sum(c) / len(c) for c in clusters if c]
    return {"round_wins_frac": sum(1 for x in rounds if x < 0) / len(rounds) if rounds else None,
            "traj_wins_frac": sum(1 for x in traj if x < 0) / len(traj) if traj else None}


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dir", default=os.path.join(root, "data", "trajectories"))
    ap.add_argument("--min-rounds", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join(root, "docs", "data", "search_vs_train.json"))
    a = ap.parse_args()
    groups = load(a.dir, a.all, a.min_rounds)
    if not groups:
        print("no compounding data under", a.dir)
        return 1

    print("=== Search vs training at matched cumulative budget (lineage - bestofN) ===")
    print("negative => verified search beats lineage self-training\n")
    print(HEADER)
    out = {"contrast": "lineage_minus_bestofN", "primary_estimand": "trajectory",
           "note": ("Trajectory is the independent replicate. Round-level pooling is reported only "
                    "to make the clustering correction auditable."),
           "models": []}
    pooled = []
    for g in sorted(groups, key=lambda m: (_size_of(m), m)):
        cl = groups[g]
        pooled += cl
        an = analyze(cl)
        w = wins(cl)
        print(fmt_row(g + "  [PRIMARY traj]", an["trajectory"]))
        print(fmt_row("   legacy round-level", an["round"]))
        if an["icc"]:
            print("   %-27s ICC=%.3f  design_effect=%.2f  n_eff=%.1f  search wins %.0f%% of runs"
                  % ("", an["icc"]["icc"], an["icc"]["design_effect"], an["icc"]["n_eff"],
                     100 * w["traj_wins_frac"]))
        t = an["trajectory"]
        if t is None:                      # <2 trajectories: report presence, never an interval
            out["models"].append({"model": g, "n_trajectories": len(cl),
                                  "n_rounds": sum(len(c) for c in cl),
                                  "underpowered": True,
                                  **{k: (round(v, 4) if v is not None else None)
                                     for k, v in w.items()}})
            continue
        out["models"].append({"model": g, "n_trajectories": t["n"], "n_rounds": t["n_rounds"],
                              "mean": round(t["mean"], 5), "lo": round(t["ci"][0], 5),
                              "hi": round(t["ci"][1], 5),
                              "underpowered": t["n"] < 5,
                              **{k: round(v, 4) for k, v in w.items()}})
    an = analyze(pooled)
    w = wins(pooled)
    print("-" * 92)
    for k, lab in (("trajectory", "POOLED [PRIMARY trajectory]"),
                   ("crve", "POOLED [cluster-robust]"),
                   ("round", "POOLED [legacy round-level]")):
        print(fmt_row(lab, an[k]))
    if an["icc"]:
        print("%-30s ICC=%.3f  design_effect=%.2f  n_eff=%.1f  (naive n=%d)" %
              ("", an["icc"]["icc"], an["icc"]["design_effect"], an["icc"]["n_eff"], an["round"]["n"]))
    t, c = an["trajectory"], an["crve"]
    out["pooled"] = {"n_trajectories": t["n"], "n_rounds": t["n_rounds"],
                     "mean": round(t["mean"], 5), "lo": round(t["ci"][0], 5), "hi": round(t["ci"][1], 5),
                     "crve_lo": round(c["ci"][0], 5), "crve_hi": round(c["ci"][1], 5),
                     **{k: round(v, 4) for k, v in w.items()}}
    survives = t["ci"][1] < 0 and c["ci"][1] < 0
    out["pooled"]["ci_excludes_zero"] = bool(survives)
    verdict = ("SEARCH BEATS TRAINING (CI excludes zero at the trajectory level)" if survives
               else "search leads but the trajectory-level CI crosses zero")
    print("\nVERDICT: %s" % verdict)
    print("  trajectory-level %+.3f [%+.3f,%+.3f] over %d runs; search wins %.0f%% of runs, %.0f%% of rounds"
          % (t["mean"], t["ci"][0], t["ci"][1], t["n"], 100 * w["traj_wins_frac"], 100 * w["round_wins_frac"]))
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
