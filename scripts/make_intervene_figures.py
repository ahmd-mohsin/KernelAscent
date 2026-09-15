#!/usr/bin/env python3
"""Probe-as-intervention figures (self-verification): reads docs/data/probe_intervene.json ->
paper/figures/iv_*.{pdf,png} and splices a \\section into paper/figures.tex (idempotent, between markers).
Fig1: per-model RANDOM vs PROBE-TOP1 vs ORACLE correct-rate bars (+lift). Fig2: K-sweep (probe vs random
any-correct) — the wall-crossing curve, esp. for sub-2B. Regenerate: python3 scripts/make_intervene_figures.py"""
import json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs", "data"); FIG = os.path.join(ROOT, "paper", "figures")
TEX = os.path.join(ROOT, "paper", "figures.tex")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 200, "savefig.bbox": "tight"})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

def M():
    try: return json.load(open(os.path.join(D, "probe_intervene.json"))).get("models", [])
    except Exception: return []
OUT = []
def save(fig, name, cap):
    for e in ("pdf", "png"): fig.savefig(os.path.join(FIG, name + "." + e))
    plt.close(fig); OUT.append((name, cap)); print("ivfig:", name)

def f_bars():
    ms = [m for m in M() if m.get("random_rate") is not None]
    if not ms: return
    ms = sorted(ms, key=lambda m: m["size_b"])
    labels = ["%s\n(%.1fB)" % (m["model"], m["size_b"]) for m in ms]
    rnd = [m.get("random_rate") or 0 for m in ms]
    pr = [m.get("probe_rate") or 0 for m in ms]
    orc = [m.get("oracle_rate") or 0 for m in ms]
    x = np.arange(len(ms)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(ms)), 6))
    ax.bar(x - w, rnd, w, label="RANDOM (natural gen)", color=CB[1])
    ax.bar(x, pr, w, label="PROBE-TOP1 (self-verify)", color=CB[2])
    ax.bar(x + w, orc, w, label="ORACLE (best-of-K)", color=CB[0], alpha=0.5)
    for i, m in enumerate(ms):
        if m.get("lift") is not None:
            ax.annotate("+%.2f" % m["lift"], (x[i], (m.get("probe_rate") or 0) + 0.02), ha="center", fontsize=8, color=CB[2])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("correct-kernel rate on held-out tasks"); ax.set_ylim(0, 1)
    ax.set_title("Probe-as-intervention: self-verification lifts correct-kernel yield above the natural generation rate")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    save(fig, "iv_bars", "Self-verification (probe-guided selection) vs.\\ the model's natural single-draw success (RANDOM) and the ORACLE best-of-$K$ upper bound, at equal generation budget $K$, on held-out tasks. Annotated $+$lift $=$ PROBE-TOP1 $-$ RANDOM. Probe selection recovering much of the oracle gap turns the correlational correctness-AUC into a causal decode-time intervention.")

def f_sweep():
    ms = [m for m in M() if m.get("k_sweep")]
    if not ms: return
    fig, ax = plt.subplots(figsize=(9, 6))
    for j, m in enumerate(sorted(ms, key=lambda m: m["size_b"])):
        ns = [d["n"] for d in m["k_sweep"]]
        pb = [d.get("probe_anycorrect") for d in m["k_sweep"]]
        rd = [d.get("rand_anycorrect") for d in m["k_sweep"]]
        c = CB[j % len(CB)]
        ax.plot(ns, pb, "-o", color=c, label="%s probe" % m["model"])
        ax.plot(ns, rd, "--", color=c, alpha=0.5, label="%s random" % m["model"])
    ax.set_xscale("log", base=2); ax.set_xlabel("selection budget n (of K)")
    ax.set_ylabel("P(>=1 correct kernel)"); ax.set_ylim(0, 1)
    ax.set_title("K-sweep: probe-guided selection vs.\\ random — does self-verification lift models over the correctness wall?")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    save(fig, "iv_ksweep", "For each model, probability of obtaining $\\geq1$ correct kernel when keeping the top-$n$ of $K$ candidates by probe score (solid) vs.\\ a random $n$ (dashed). A probe curve rising above random — especially lifting a sub-2B model from $\\approx0$ — is direct evidence that the correctness wall is partly a decode/selection artifact.")

for f in (f_bars, f_sweep):
    try: f()
    except Exception as e: print("skip", f.__name__, e)

BEG = "% >>> INTERVENE (auto)"; END = "% <<< INTERVENE (auto)"
if OUT:
    sec = [BEG, "\\clearpage", "\\section{Probe-as-intervention (self-verification)}"]
    for name, cap in OUT:
        sec += ["\\begin{figure}[H]\\centering", "\\includegraphics[width=0.9\\linewidth]{%s.pdf}" % name,
                "\\caption{%s}" % cap, "\\label{fig:%s}" % name, "\\end{figure}"]
    sec.append(END); block = "\n".join(sec) + "\n"
    tex = open(TEX).read()
    tex = re.sub(re.escape(BEG) + r".*?" + re.escape(END) + r"\n?", "", tex, flags=re.S)
    tex = tex.replace("\\end{document}", block + "\\end{document}")
    open(TEX, "w").write(tex)
    print("added %d intervene figures; tex updated" % len(OUT))
else:
    print("no probe_intervene data yet; no figures added")
