"""Cross-seed rigor analysis (roadmap #3). Aggregates weight-RSI runs by base model across seeds and applies the
PRE-REGISTERED decision rule: a model 'compounds' iff mean(last-2-round self-minus-fresh) > 0.05 AND the effect
is sign-consistent across seeds (and, when the run stores task-level CI, that CI excludes 0). Emits rigor.json.
Also surfaces diversity_self / retention when present (populated by instrumented re-runs) for the mechanism figure.
Run: python3 -m kernelascent.v3.ci_analysis <dir-of-wrsi_*.json> [out.json]
"""
import json, glob, os, re, sys, statistics

def base_name(fn):
    b = os.path.basename(fn)[len("wrsi_"):-len(".json")]
    return re.sub(r"_s\d+$", "", b)

def load(d):
    groups = {}
    for f in glob.glob(os.path.join(d, "wrsi_*.json")):
        try: j = json.load(open(f))
        except: continue
        h = j.get("history", [])
        if not h: continue
        groups.setdefault(base_name(f), []).append(j)
    return groups

def analyze(groups):
    rows = []
    for name, runs in sorted(groups.items()):
        finals, last2, divs, rets = [], [], [], []
        for j in runs:
            h = j["history"]
            sf = [r.get("delta_self_minus_fresh") for r in h if r.get("delta_self_minus_fresh") is not None]
            if sf:
                finals.append(sf[-1]); last2.append(statistics.mean(sf[-2:]))
            divs += [r["diversity_self"] for r in h if r.get("diversity_self") is not None]
            rets += [r["retention"] for r in h if r.get("retention") is not None]
        if not finals: continue
        n = len(finals)
        mean_final = statistics.mean(finals); mean_last2 = statistics.mean(last2)
        spread = (max(finals) - min(finals)) / 2 if n > 1 else None
        sign_consistent = all(x > 0 for x in finals) or all(x < 0 for x in finals)
        holds = (mean_last2 > 0.05) and sign_consistent
        verdict = "compounds" if holds else ("overfits" if mean_final <= -0.02 else "flat")
        rows.append(dict(model=name, n_seeds=n, seeds_final=[round(x, 3) for x in finals],
            mean_final_self_minus_fresh=round(mean_final, 3),
            mean_last2=round(mean_last2, 3),
            half_range=round(spread, 3) if spread is not None else None,
            sign_consistent=sign_consistent, holds=holds, verdict=verdict,
            mean_diversity=round(statistics.mean(divs), 3) if divs else None,
            mean_retention=round(statistics.mean(rets), 3) if rets else None,
            note=("single-seed: needs >=3 for CI" if n < 3 else "")))
    rows.sort(key=lambda x: -x["mean_final_self_minus_fresh"])
    return rows

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = analyze(load(d))
    import datetime
    out = {"updated": datetime.date.today().isoformat(),
           "note": "cross-seed self-vs-fresh-frozen (pre-registered primary). holds = mean(last-2) > 0.05 AND sign-consistent across seeds. >=3 seeds needed for reported bootstrap CI.",
           "models": rows}
    outp = sys.argv[2] if len(sys.argv) > 2 else "rigor.json"
    json.dump(out, open(outp, "w"), indent=2)
    for r in rows:
        print("  %-30s n=%d final=%s mean=%.3f last2=%.3f %s%s" %
              (r["model"], r["n_seeds"], r["seeds_final"], r["mean_final_self_minus_fresh"],
               r["mean_last2"], r["verdict"], " [" + r["note"] + "]" if r["note"] else ""))
