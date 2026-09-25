#!/usr/bin/env python3
"""Re-derive the four headline results into docs/data/headline.json for the website.

The site used to hard-code its numbers in index.html. That is the one practice this project
has already been burned by: a figure typed into prose and propagated by hand turned out to be
unreproducible from any artifact, and the page kept showing it long after the paper stopped.
The paper regenerates its tables at build time; the site now reads the same artifacts through
the same module, so a number cannot exist on one and not the other.

It imports the paper's generator rather than re-reading the artifacts itself. Two readers of
one artifact set diverge the first time either is edited, and the divergence shows up as a
disagreement between the paper and the site that neither owner can see.

  python3 scripts/build_site_headline.py
"""
import json, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "docs", "data", "headline.json")

import make_kernel_results_tex as G


def contrast():
    """One design, one seed, one environment variable changed, opposite conclusions.

    Mirrors G.metric_contrast_table exactly, including the >=0.49 parity test, so the site and
    Table~\\ref{tab:contrast} cannot disagree.
    """
    rows, dead_h, dead_p = [], [], []
    for hc, pc in G.CONTRAST_PAIRS:
        h = G._load(G._find(os.path.join(hc, "compounding.json")))
        p = G._load(G._find(os.path.join(pc, "compounding.json")))
        if not h or not p:
            continue
        hh, ph = h.get("history") or [], p.get("history") or []
        for r in hh:
            i = r["round"]
            pv = ph[i]["lineage_minus_reset"] if i < len(ph) else None
            both = r["C_lineage"] >= 0.49 and r["C_reset"] >= 0.49
            rows.append({"cell": hc, "round": i, "C_lineage": r["C_lineage"],
                         "C_reset": r["C_reset"], "headroom": r["lineage_minus_reset"],
                         "passrate": pv, "both_at_parity": both})
            if both and pv is not None:
                dead_h.append(r["lineage_minus_reset"]); dead_p.append(pv)
    if not dead_h:
        return None
    return {"rows": rows, "n_parity_rounds": len(dead_h),
            "headroom_mean": st.mean(dead_h), "headroom_max_abs": max(abs(x) for x in dead_h),
            "passrate_mean": st.mean(dead_p), "passrate_min": min(dead_p),
            # Separation of the RANGES, not of the means. The ratio of means is ~2180 here and
            # says almost nothing: it is dominated by how close the headroom mean sits to zero,
            # which is an accident of two signed values cancelling. The claim worth making is
            # that the two ranges do not approach one another, and its statistic is the closest
            # the two ever come: the smallest pass-rate reading over the largest headroom
            # excursion. That is the figure the report quotes.
            "range_separation": (min(dead_p) / max(abs(x) for x in dead_h)
                                 if max(abs(x) for x in dead_h) else None)}


def starvation():
    """Measured throughput of the self-referential loop, per rung.

    T2 is computed here from the registered primary's artifacts. T3 and T5 are read from the
    numbers their own labs stamped; both are reported with the denominator so `undefined` stays
    distinguishable from `zero`.
    """
    nex, cells = [], []
    for cell, outdir in G._cells("prereg_*"):
        d = G._load(os.path.join(outdir, "weight_rsi.json"))
        if not d or not (d.get("history") or []):
            continue
        # Same admissibility rule as the table: a severed or pre-fix cell measures nothing, so
        # it must not contribute to a throughput figure either.
        if "adapter_restored" not in d:
            continue
        if d.get("resumed_at") and d.get("adapter_restored") is not True:
            continue
        v = [r.get("n_ex", 0) for r in d["history"]]
        nex += v
        cells.append({"cell": cell, "n_ex": v})
    if not nex:
        return None
    return {"cells": cells, "n_rounds": len(nex), "mean_per_round": sum(nex) / len(nex),
            "n_empty": sum(1 for x in nex if x == 0),
            "frac_empty": sum(1 for x in nex if x == 0) / len(nex),
            # n_train x k x yield, the one line that was available before any of this ran.
            "predicted": {"n_train": 3, "k": 10, "yield": 0.032, "expected": 3 * 10 * 0.032}}


def nonllm():
    d = G._load(G._find("nonllm_baselines.json"))
    if not d:
        return None
    rows = [r for r in (d.get("rows") or []) if r.get("max_autotune_benchmark_score") is not None]
    if not rows or d.get("n_scored") != d.get("n_tasks"):
        return None
    sc = [r["max_autotune_benchmark_score"] for r in rows]
    sp = [r["max_autotune_speedup_vs_compiled"] for r in rows]
    beats = d.get("n_beating_compiled", sum(1 for x in sp if x > 1.03))
    by = {}
    for r in rows:
        by.setdefault(r["task"].split("_")[0], []).append(r["max_autotune_benchmark_score"])
    return {"n_tasks": len(rows), "median_score": st.median(sc),
            "median_speedup_vs_compiled": st.median(sp),
            "n_slower_than_compiled": len(rows) - beats,
            "by_tier": {k.upper(): {"n": len(v), "median_score": st.median(v)}
                        for k, v in sorted(by.items())}}


def main():
    out, missing = {}, []
    for key, fn in (("contrast", contrast), ("starvation", starvation), ("nonllm", nonllm)):
        v = fn()
        if v is None:
            missing.append(key)
        else:
            out[key] = v
    if missing:
        # Fail loudly. A site section that silently renders nothing looks like a design choice,
        # and the reader cannot tell a missing result from an absent one.
        sys.stderr.write("WARNING: no data for %s; those sections will not render\n"
                         % ", ".join(missing))
    out["generated_from"] = "data/marlowe_h100 + data/trajectories via make_kernel_results_tex"
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print("wrote %s (%s)" % (OUT, ", ".join(sorted(k for k in out if k != "generated_from"))))
    if "contrast" in out:
        c = out["contrast"]
        print("  contrast: %d parity rounds, headroom %+.4f (max |%.4f|), pass rate %+.3f"
              % (c["n_parity_rounds"], c["headroom_mean"], c["headroom_max_abs"], c["passrate_mean"]))
    if "starvation" in out:
        s = out["starvation"]
        print("  starvation: %.2f examples/round over %d rounds, %d empty (%.0f%%)"
              % (s["mean_per_round"], s["n_rounds"], s["n_empty"], 100 * s["frac_empty"]))
    if "nonllm" in out:
        n = out["nonllm"]
        print("  non-LLM: median %.3f, slower than compiled on %d of %d"
              % (n["median_score"], n["n_slower_than_compiled"], n["n_tasks"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
