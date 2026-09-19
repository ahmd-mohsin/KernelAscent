#!/usr/bin/env python3
"""Mechanistic analysis: WHY larger models compound RSI and smaller don't — computed from the WHY-RSI probe
series in docs/data/rsi_mech.json (per-round internal signals: LoRA drift by transformer depth, generation
diversity/entropy, retention, train-vs-held transfer gap). Emits docs/data/mech_analysis.json (per-model
summary + grouped verdicts) and prints the headline findings. Purely re-derived from the raw series (does not
trust the probe's own verdict field)."""
import json, os, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs", "data")

SIZE = {  # approx params (B) by model-name substring, for the scale axis
    "0.5": 0.5, "1.3": 1.3, "1.5": 1.5, "1.7": 1.7, "3b": 3, "3B": 3, "-3": 3,
    "6.7": 6.7, "7B": 7, "7b": 7, "8B": 8, "8b": 8, "9B": 9, "14B": 14, "15b": 15, "15B": 15, "32B": 32,
}
def size_of(name):
    for k, v in SIZE.items():
        if k in name: return v
    return 2.0  # default mid-small

def m(series):
    xs = [x for x in (series or []) if isinstance(x, (int, float))]
    return statistics.mean(xs) if xs else 0.0
def last(series, d=0.0):
    xs = [x for x in (series or []) if isinstance(x, (int, float))]
    return xs[-1] if xs else d
def first(series, d=0.0):
    xs = [x for x in (series or []) if isinstance(x, (int, float))]
    return xs[0] if xs else d

def analyze():
    d = json.load(open(os.path.join(D, "rsi_mech.json"))); models = d.get("models", [])
    rows = []
    for x in models:
        Ctr, Ch = x.get("C_train", []), x.get("C_held", [])
        wall = max([c for c in Ctr if isinstance(c, (int, float))] + [0]) > 0   # ever emitted a correct kernel
        c_improve = last(Ch) - first(Ch)                                        # held-out gain over rounds
        de, dm, dl = m(x.get("drift_early")), m(x.get("drift_mid")), m(x.get("drift_late"))
        drift_tot = de + dm + dl
        div0, div1 = first(x.get("gen_distinct2")), last(x.get("gen_distinct2"))
        rows.append(dict(
            model=x.get("model"), size_b=size_of(x.get("model", "")), tier=x.get("tier"),
            wall_crossed=wall, max_C_train=round(max([c for c in Ctr if isinstance(c,(int,float))]+[0]),3),
            C_held_gain=round(c_improve, 3), rsi=bool(wall and c_improve > 0.02),   # our clean RSI success def
            drift_early=round(de,4), drift_mid=round(dm,4), drift_late=round(dl,4), drift_total=round(drift_tot,4),
            drift_locus=(max((("early",de),("mid",dm),("late",dl)), key=lambda t:t[1])[0] if drift_tot>0 else "none"),
            mean_diversity=round(m(x.get("gen_distinct2")),3), diversity_collapse=bool(div1 < div0 - 0.02),
            mean_retention=round(m(x.get("retention")),3), mean_transfer_gap=round(m(x.get("transfer_gap")),3),
        ))
    rows.sort(key=lambda r: r["size_b"])
    # ---- grouped findings ----
    crossed = [r for r in rows if r["wall_crossed"]]
    notcross = [r for r in rows if not r["wall_crossed"]]
    def frac(g, key): return round(sum(1 for r in g if r[key]) / len(g), 2) if g else None
    def mean(g, key): return round(statistics.mean([r[key] for r in g]), 3) if g else None
    # scale bins. Cutoffs are <2 / 2-8 / >=8 and MUST match scripts/make_results_tex.py:mech_interp_table
    # (an earlier version labelled a bin "2-8B" while actually using size<9, so the paper carried two
    # different band populations for the same quantity; scripts/consistency_audit.py now checks this).
    bins = {"<2B": [r for r in rows if r["size_b"] < 2], "2-8B": [r for r in rows if 2 <= r["size_b"] < 8],
            ">=8B": [r for r in rows if r["size_b"] >= 8]}
    scale = {b: dict(n=len(g), frac_cross_wall=frac(g, "wall_crossed"), frac_rsi=frac(g, "rsi"),
                     mean_C_train=mean(g, "max_C_train"), mean_drift=mean(g, "drift_total"),
                     mean_diversity=mean(g, "mean_diversity")) for b, g in bins.items() if g}
    # Claim strings are DERIVED from the numbers beside them, never written by hand: an earlier
    # hand-written pair asserted "sub-2B almost never cross" (actual rate 0.54) and "compounders
    # sustain higher diversity" (actually LOWER: 0.41 vs 0.52), and the website rendered both.
    big, small = [r for r in rows if r["size_b"] >= 2], [r for r in rows if r["size_b"] < 2]
    f_ge2, f_lt2 = frac(big, "wall_crossed"), frac(small, "wall_crossed")
    n_cross_lt2 = sum(1 for r in small if r.get("wall_crossed"))
    n_cross_ge2 = sum(1 for r in big if r.get("wall_crossed"))
    div_rsi = mean([r for r in crossed if r["rsi"]], "mean_diversity")
    div_flat = mean([r for r in crossed if not r["rsi"]], "mean_diversity")
    dr_rsi = mean([r for r in crossed if r["rsi"]], "drift_total")
    dr_flat = mean([r for r in crossed if not r["rsi"]], "drift_total")
    rt_rsi = mean([r for r in crossed if r["rsi"]], "mean_retention")
    rt_flat = mean([r for r in crossed if not r["rsi"]], "mean_retention")
    n_rsi = sum(1 for r in crossed if r.get("rsi"))
    _dir = lambda a, b: "higher" if a > b else ("lower" if a < b else "equal")
    findings = dict(
        n_models=len(rows),
        correctness_wall=dict(
            frac_cross_if_ge2B=f_ge2, frac_cross_if_lt2B=f_lt2,
            n_cross_if_ge2B=[n_cross_ge2, len(big)], n_cross_if_lt2B=[n_cross_lt2, len(small)],
            claim=("Sub-2B models cross the correctness wall LESS FREQUENTLY than >=2B models "
                   "(%d/%d = %.0f%% vs %d/%d = %.0f%%), not 'almost never'. Below the wall C_train=0, so the "
                   "SFT set is empty, there is no gradient and drift -> 0. Association, not an established "
                   "causal gate." % (n_cross_lt2, len(small), 100 * f_lt2,
                                     n_cross_ge2, len(big), 100 * f_ge2))),
        among_wall_crossers=dict(
            n=len(crossed), n_rsi=n_rsi, frac_rsi=frac(crossed, "rsi"),
            rsi_vs_flat_diversity=[div_rsi, div_flat],
            rsi_vs_flat_drift=[dr_rsi, dr_flat],
            rsi_vs_flat_retention=[rt_rsi, rt_flat],
            claim=("Among the %d runs that DO produce correct kernels, %d compound. Compounders show "
                   "%s sustained LoRA drift (%.3f vs %.3f) and %s retention (%.3f vs %.3f) than flat runs. "
                   "Generation diversity is %s in compounders (%.3f vs %.3f), so on this bank diversity "
                   "collapse does NOT discriminate compounding -- the discriminators are drift and "
                   "retention. Associations, not causal gates."
                   % (len(crossed), n_rsi, _dir(dr_rsi, dr_flat), dr_rsi, dr_flat,
                      _dir(rt_rsi, rt_flat), rt_rsi, rt_flat,
                      _dir(div_rsi, div_flat), div_rsi, div_flat))),
        scale_trend=scale,
    )
    out = dict(updated=__import__("datetime").date.today().isoformat(),
               note="WHY larger models compound RSI and smaller don't: scale -> correctness-wall crossing -> training gradient -> LoRA drift (by depth) + sustained self-data diversity -> held-out compounding. Re-derived from per-round WHY-RSI probe series.",
               findings=findings, models=rows)
    json.dump(out, open(os.path.join(D, "mech_analysis.json"), "w"), indent=2)
    print("=== SCALE -> RSI ===")
    for b, s in scale.items(): print(f"  {b:5s} n={s['n']:2d} cross_wall={s['frac_cross_wall']} rsi={s['frac_rsi']} C_train={s['mean_C_train']} drift={s['mean_drift']} div={s['mean_diversity']}")
    print("=== WALL: cross if >=2B =%s vs <2B =%s ===" % (findings["correctness_wall"]["frac_cross_if_ge2B"], findings["correctness_wall"]["frac_cross_if_lt2B"]))
    aw = findings["among_wall_crossers"]
    print("=== AMONG CROSSERS (rsi vs flat): diversity=%s drift=%s retention=%s ===" % (aw["rsi_vs_flat_diversity"], aw["rsi_vs_flat_drift"], aw["rsi_vs_flat_retention"]))
    return out

if __name__ == "__main__":
    analyze()
