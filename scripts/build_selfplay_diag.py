#!/usr/bin/env python3
"""Aggregate self-play diagnosis+rescue runs (results/raw/selfplay_diag_*.json) -> docs/data/selfplay_diag.json.
Shows curriculum collapse (authored solve-rate drifts to trivial~1 / unsolvable~0) and whether the Goldilocks
author rescues L-F (Lg-F>0 while unconstrained L-F~0)."""
import json, glob, os, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw"); D = os.path.join(ROOT, "docs", "data")

def build():
    rows = []
    for f in glob.glob(os.path.join(RAW, "selfplay_diag_*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        h = d.get("history") or []
        if not h: continue
        m = d.get("model", os.path.basename(f)[14:-5]).split("/")[-1]
        lu = [r.get("L_minus_F_unconstrained") for r in h if r.get("L_minus_F_unconstrained") is not None]
        lg = [r.get("L_minus_F_goldilocks") for r in h if r.get("L_minus_F_goldilocks") is not None]
        rescued = (len(lg) >= 3 and statistics.mean(lg[-2:]) > 0.05 and statistics.mean(lu[-2:]) <= 0.05) if lu and lg else None
        rows.append(dict(model=m, rounds=len(h), band=d.get("band"),
            authored_solve_rate=[r.get("authored_solve_rate") for r in h],
            frac_trivial=[r.get("frac_trivial") for r in h],
            frac_unsolvable=[r.get("frac_unsolvable") for r in h],
            authored_headroom=[r.get("authored_headroom") for r in h],
            L_minus_F_unconstrained=[r.get("L_minus_F_unconstrained") for r in h],
            L_minus_F_goldilocks=[r.get("L_minus_F_goldilocks") for r in h],
            mean_Lg_minus_F=round(statistics.mean(lg[-2:]), 3) if lg else None,
            mean_L_minus_F=round(statistics.mean(lu[-2:]), 3) if lu else None,
            goldilocks_rescues=rescued))
    rows.sort(key=lambda r: -(r.get("mean_Lg_minus_F") or -9))
    json.dump(dict(updated=__import__("datetime").date.today().isoformat(),
        note="T5 self-play diagnosis: authored-task solve-rate collapse (frac_trivial=solver solves >=90%, frac_unsolvable=<=10%) shows why unconstrained L-F~0. Rescue: Goldilocks author (keep authored tasks solved in band) -> L_minus_F_goldilocks. goldilocks_rescues=True iff Lg-F>0 sustained while unconstrained L-F~0.",
        models=rows), open(os.path.join(D, "selfplay_diag.json"), "w"), indent=2)
    print("selfplay_diag board:", len(rows), "models;", sum(1 for r in rows if r.get("goldilocks_rescues")), "rescued")

if __name__ == "__main__":
    build()
