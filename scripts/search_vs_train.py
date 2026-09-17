#!/usr/bin/env python3
"""Finding (i): at MATCHED cumulative generation budget, does best-of-N SEARCH beat lineage self-TRAINING?
lab_compounding gives per-round C_lineage and C_bestofN where best-of-N uses budget k*(r+1) (== the cumulative
generation the lineage arm spent), so the comparison is compute-matched by construction. We aggregate
lineage_minus_bestofN across the clean post-fix seeds (same allowlist as equivalence_tost), per model + pooled,
with a 95% CI and the fraction of rounds where search wins (delta<0). Pure stdlib."""
import json, glob, os, math, argparse, re, statistics as st

_CLEAN = re.compile(r"^compounding_q(05|15|3)(new|s\d+)$")


def load(d, allow_all=False):
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
        groups.setdefault(model, []).extend(vals)
    return groups


def stats(v):
    n = len(v)
    if n < 2:
        return None
    m = st.mean(v); sd = st.pstdev(v); ci = 1.96 * sd / math.sqrt(n)
    win = sum(1 for x in v if x < 0)  # search strictly beats lineage
    return {"n": n, "mean": m, "lo": m - ci, "hi": m + ci, "search_wins_frac": win / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data"))
    a = ap.parse_args()
    groups = load(a.dir)
    if not groups:
        print("no compounding data under", a.dir); return
    pooled = []
    print("model                         n   lineage-bestofN [95%CI]      search_wins")
    out = {"models": []}
    for g in sorted(groups):
        v = groups[g]; pooled += v; r = stats(v)
        if not r:
            continue
        print("%-28s %3d  %+.3f [%+.3f,%+.3f]   %.0f%%" %
              (g[:28], r["n"], r["mean"], r["lo"], r["hi"], 100 * r["search_wins_frac"]))
        out["models"].append({"model": g, **r})
    r = stats(pooled)
    print("-" * 72)
    print("%-28s %3d  %+.3f [%+.3f,%+.3f]   %.0f%%" %
          ("POOLED", r["n"], r["mean"], r["lo"], r["hi"], 100 * r["search_wins_frac"]))
    out["pooled"] = {"n": r["n"], **r}
    json.dump(out, open(os.path.join(a.dir, "search_vs_train.json"), "w"), indent=1)
    verdict = "SEARCH BEATS TRAINING" if r["hi"] < 0 else ("search leads (CI crosses 0)" if r["mean"] < 0 else "training competitive")
    print("\nVERDICT: %s (lineage-bestofN=%+.3f, search wins %.0f%% of matched-budget rounds)" %
          (verdict, r["mean"], 100 * r["search_wins_frac"]))


if __name__ == "__main__":
    main()
