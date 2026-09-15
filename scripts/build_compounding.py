#!/usr/bin/env python3
"""Aggregate compounding-test runs (results/raw/compounding_*.json) -> docs/data/compounding.json.
Headline per model: does lineage (accumulating RSI) beat the reset (single-round) and best-of-N (matched-budget
search) controls, sustained over rounds? verdict 'compounds' iff mean of last-2 lineage_minus_reset > 0.05."""
import json, glob, os, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw"); D = os.path.join(ROOT, "docs", "data")

def build():
    rows = []
    for f in glob.glob(os.path.join(RAW, "compounding_*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        h = d.get("history") or []
        if not h: continue
        m = d.get("model", os.path.basename(f)[12:-5]).split("/")[-1]
        lr = [r.get("lineage_minus_reset") for r in h if r.get("lineage_minus_reset") is not None]
        lb = [r.get("lineage_minus_bestofN") for r in h if r.get("lineage_minus_bestofN") is not None]
        mg = [r.get("marginal_gain_lineage") for r in h if r.get("marginal_gain_lineage") is not None]
        last = h[-1]
        verdict = ("compounds" if len(lr) >= 3 and statistics.mean(lr[-2:]) > 0.05
                   else "search-only" if lb and statistics.mean(lb[-2:]) <= 0.02
                   else "inconclusive")
        rows.append(dict(model=m, rounds=len(h), C0=d.get("C0"),
            C_lineage_final=last.get("C_lineage"), C_reset_final=last.get("C_reset"),
            C_bestofN_final=last.get("C_bestofN"), transfer_C_final=last.get("transfer_C"),
            lineage_minus_reset=[r.get("lineage_minus_reset") for r in h],
            lineage_minus_bestofN=[r.get("lineage_minus_bestofN") for r in h],
            marginal_gain_lineage=[r.get("marginal_gain_lineage") for r in h],
            mean_lineage_minus_reset=round(statistics.mean(lr), 3) if lr else None,
            held_family=d.get("held_family"), verdict=verdict))
    rows.sort(key=lambda r: -(r.get("mean_lineage_minus_reset") or -9))
    json.dump(dict(updated=__import__("datetime").date.today().isoformat(),
        note="Compounding test: LINEAGE (accumulating weight-RSI) vs RESET (single-round, adapter re-init) vs BEST-OF-N (frozen, matched cumulative budget). lineage_minus_reset>0 sustained = accumulation raises subsequent learning (true compounding); lineage_minus_bestofN>0 = beats search. transfer_C = held-out task FAMILY never trained on.",
        models=rows), open(os.path.join(D, "compounding.json"), "w"), indent=2)
    print("compounding board:", len(rows), "models;", sum(1 for r in rows if r["verdict"] == "compounds"), "compound")

if __name__ == "__main__":
    build()
