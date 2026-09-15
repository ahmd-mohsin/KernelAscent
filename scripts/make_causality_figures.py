#!/usr/bin/env python3
"""Large causality & internal-failure graphs for the KernelAscent RSI paper (dense-DAG aesthetic:
lanes + translucent curved bezier edges + degree-scaled markers). Reads docs/data/*.json ->
paper/figures/*.{pdf,png} and splices a \\section into paper/figures.tex (idempotent, between markers).
Regenerate: python3 scripts/make_causality_figures.py"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs", "data"); FIG = os.path.join(ROOT, "paper", "figures")
TEX = os.path.join(ROOT, "paper", "figures.tex")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 200, "savefig.bbox": "tight"})

def load(n):
    try: return json.load(open(os.path.join(D, n + ".json")))
    except Exception: return {"models": []}
def M(n): return load(n).get("models", [])
def short(s): return (s or "").split("/")[-1].replace("-Instruct","").replace("-Base","").replace("-instruct","")
CB = ["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#F0E442","#999999"]
OUT = []  # (name, caption)

def save(fig, name, caption):
    for e in ("pdf","png"): fig.savefig(os.path.join(FIG, name+"."+e))
    plt.close(fig); OUT.append((name, caption)); print("cfig:", name)

def bezier(ax, x0,y0, x1,y1, color, lw=0.8, alpha=0.18):
    mx = (x0+x1)/2
    p = Path([(x0,y0),(mx,y0),(mx,y1),(x1,y1)], [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4])
    ax.add_patch(PathPatch(p, fc="none", ec=color, lw=lw, alpha=alpha))

def outcome_color(m):
    if not m.get("wall_crossed"): return CB[1]      # wall (orange)
    if m.get("rsi"): return CB[2]                    # compounder (green)
    return CB[0]                                     # crossed-but-flat (blue)

# ---------- 1. per-layer correctness-AUC heatmap (interp) ----------
def fig_interp_heatmap():
    rows = [m for m in M("interp") if m.get("per_layer_auc")]
    if not rows: return
    rows.sort(key=lambda m: m.get("size_b", 0))
    G = 40; grid = np.linspace(0, 1, G); mat = []
    for m in rows:
        a = [x for x in m["per_layer_auc"] if isinstance(x,(int,float))]
        xs = np.linspace(0,1,len(a)); mat.append(np.interp(grid, xs, a))
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(14, max(3, 0.55*len(rows)+2)))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.5, vmax=1.0, extent=[0,1,len(rows)-0.5,-0.5])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{short(m['model'])} ({m['size_b']:g}B)" for m in rows])
    ax.set_xlabel("relative transformer depth (0=embed, 1=final)")
    for i,m in enumerate(rows):
        if m.get("best_layer_frac") is not None:
            ax.plot(m["best_layer_frac"], i, "c*", ms=13, mec="k")
    fig.colorbar(im, ax=ax, label="correctness decodability (AUC)")
    ax.set_title("WHERE kernel-correctness is internally encoded, by scale (cyan mark = peak layer)")
    save(fig, "cz_interp_heatmap",
         "Per-layer linear decodability of kernel correctness (AUC of a correct-vs-incorrect probe) across relative transformer depth, one row per model sorted by scale; $\\star$ marks the peak layer. Correctness becomes cleanly decodable in mid-to-late layers -- the model internally represents which kernels are correct. Rows populate as GPU interp probes land.")

# ---------- 2. internal-failure causality DAG ----------
def fig_internal_dag():
    rows = M("mech_analysis")
    if not rows: return
    rows = [m for m in rows if m.get("size_b")]
    stages = ["scale","wall","gradient","drift","retention","diversity","outcome"]
    sx = {s:i for i,s in enumerate(stages)}
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_xlim(-0.5, len(stages)-0.5); ax.set_ylim(-0.2, 1.2)
    for s in stages: ax.text(sx[s], 1.13, s.upper(), ha="center", va="center", fontsize=11, fontweight="bold")
    def yval(m, s):
        if s=="scale": return min(m["size_b"]/16.0, 1.0)
        if s=="wall": return 0.85 if m.get("wall_crossed") else 0.15
        if s=="gradient": return min((m.get("max_C_train") or m.get("mean_diversity",0))*3,1.0) if m.get("wall_crossed") else 0.1
        if s=="drift": return min((m.get("drift_early",0)+m.get("drift_mid",0)+m.get("drift_late",0)),1.0)
        if s=="retention": return min(m.get("mean_retention",0)*2,1.0)
        if s=="diversity": return min(m.get("mean_diversity",0),1.0)
        if s=="outcome": return 0.9 if m.get("rsi") else (0.5 if m.get("wall_crossed") else 0.1)
    for m in rows:
        c = outcome_color(m); pts=[(sx[s], yval(m,s)) for s in stages]
        drift = m.get("drift_early",0)+m.get("drift_mid",0)+m.get("drift_late",0)
        for (x0,y0),(x1,y1) in zip(pts[:-1],pts[1:]):
            bezier(ax,x0,y0,x1,y1,c, lw=0.6+3*drift, alpha=0.10+0.35*(1 if m.get("rsi") else 0.4*(1 if m.get("wall_crossed") else 0.15)))
        for (x,y) in pts:
            ax.scatter([x],[y], s=20+900*drift, c=c, alpha=0.55, edgecolors="k", linewidths=0.3, zorder=3)
    from matplotlib.lines import Line2D
    leg=[Line2D([],[],marker='o',color='w',markerfacecolor=CB[2],markersize=11,label='RSI compounds'),
         Line2D([],[],marker='o',color='w',markerfacecolor=CB[0],markersize=11,label='crossed wall, flat'),
         Line2D([],[],marker='o',color='w',markerfacecolor=CB[1],markersize=11,label='stuck at correctness wall'),
         Line2D([],[],marker='o',color='w',markerfacecolor='gray',markersize=6,label='small'),
         Line2D([],[],marker='o',color='w',markerfacecolor='gray',markersize=15,label='large drift')]
    ax.legend(handles=leg, loc="lower left", ncol=2, fontsize=9, framealpha=0.9)
    ax.set_title("Internal-failure causality DAG: each model's path Scale$\\to$Wall$\\to$Gradient$\\to$Drift$\\to$Retention$\\to$Diversity$\\to$Outcome (%d models; edge width & marker area $\\propto$ LoRA drift)" % len(rows), fontsize=11)
    save(fig, "cz_internal_dag",
         "Every open-weight run traced through the causal chain from scale to RSI outcome, in lanes with translucent curved edges (marker area and edge width $\\propto$ sustained LoRA drift, color = outcome). Compounders (green) form a bright high-drift/high-retention bundle; models stuck at the correctness wall (orange) die at the second lane with near-zero drift downstream.")

# ---------- 3. causal staged-flow (Sankey-style) ----------
def fig_causal_flow():
    rows = [m for m in M("mech_analysis") if m.get("size_b")]
    if not rows: return
    n=len(rows)
    crossed=[m for m in rows if m.get("wall_crossed")]; wall=[m for m in rows if not m.get("wall_crossed")]
    comp=[m for m in crossed if m.get("rsi")]; flat=[m for m in crossed if not m.get("rsi")]
    fig, ax = plt.subplots(figsize=(14,7)); ax.axis("off"); ax.set_xlim(0,4); ax.set_ylim(0,1)
    def band(x0,x1,y0c,h0,y1c,h1,color,label,frac):
        p=Path([(x0,y0c-h0/2),(( x0+x1)/2,y0c-h0/2),((x0+x1)/2,y1c-h1/2),(x1,y1c-h1/2),
                (x1,y1c+h1/2),((x0+x1)/2,y1c+h1/2),((x0+x1)/2,y0c+h0/2),(x0,y0c+h0/2),(x0,y0c-h0/2)],
               [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.LINETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CLOSEPOLY])
        ax.add_patch(PathPatch(p, fc=color, ec="none", alpha=0.5))
    H=lambda k: max(k/n,0.001)*0.9
    # col0 all -> col1 wall/crossed -> col2 flat/comp -> col3 outcome
    ax.text(0.1,0.95,f"ALL RUNS\n(n={n})",fontweight="bold",fontsize=10)
    band(0.3,1.3,0.5,H(n),0.75,H(len(crossed)),CB[2],"",1); band(0.3,1.3,0.5,H(n),0.2,H(len(wall)),CB[1],"",1)
    ax.text(1.4,0.75,f"crossed wall\n{len(crossed)}",fontsize=9); ax.text(1.4,0.2,f"stuck at wall\n{len(wall)}",fontsize=9,color=CB[1])
    band(1.3,2.3,0.75,H(len(crossed)),0.85,H(len(comp)),CB[2],"",1); band(1.3,2.3,0.75,H(len(crossed)),0.55,H(len(flat)),CB[0],"",1)
    ax.text(2.4,0.85,f"compound\n{len(comp)}",fontsize=9,color=CB[2]); ax.text(2.4,0.55,f"flat\n{len(flat)}",fontsize=9,color=CB[0])
    # mechanisms of flat
    ax.text(3.1,0.55,"flat cause:\ndrift-saturation\n+ forgetting",fontsize=9,color=CB[0])
    ax.text(3.1,0.2,"cause:\nno correct kernel\n$\\Rightarrow$ no gradient",fontsize=9,color=CB[1])
    ax.text(3.1,0.85,"sustained drift\n+ retention",fontsize=9,color=CB[2])
    ax.set_title("Causal attrition: where models drop out of RSI (band width $\\propto$ model count)")
    save(fig,"cz_causal_flow",
         "Causal attrition of %d open-weight runs: most are lost at Gate~1 (the correctness wall -- no verified kernel, no gradient); among wall-crossers only a minority compound, the rest going flat via drift-saturation and forgetting (Gate~2)." % n)

# ---------- 4. knows-vs-generates gap ----------
def fig_knows_vs_gen():
    rows = [m for m in M("interp") if m.get("correctness_rate") is not None]
    if not rows: return
    fig, ax = plt.subplots(figsize=(9,8))
    for m in rows:
        cr=m.get("correctness_rate") or 0; au=m.get("best_auc")
        if au is None:
            ax.scatter([cr],[0.5], s=120, c=CB[1], marker="x"); ax.annotate(short(m["model"])+" (wall)", (cr,0.5), fontsize=8)
        else:
            ax.scatter([cr],[au], s=160, c=CB[2], edgecolors="k"); ax.annotate(short(m["model"]), (cr,au), fontsize=9, xytext=(4,4), textcoords="offset points")
            ax.annotate("", xy=(cr,au), xytext=(cr,cr), arrowprops=dict(arrowstyle="->",color="gray",alpha=0.6))
    ax.plot([0,1],[0,1],"k--",alpha=0.4,label="AUC = generation rate")
    ax.axhspan(0.8,1.0,0,1,color=CB[2],alpha=0.06)
    ax.text(0.35,0.9,"generation-limited:\nknows correctness (high AUC),\nrarely emits it",fontsize=9,color=CB[2])
    ax.text(0.02,0.42,"correctness wall\n(can't be probed)",fontsize=9,color=CB[1])
    ax.set_xlabel("generation success (correctness rate)"); ax.set_ylabel("internal correctness knowledge (best-layer AUC)")
    ax.set_xlim(-0.02,1); ax.set_ylim(0,1.02); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Knows $\\gg$ generates: the RSI bottleneck is decoding, not knowledge")
    save(fig,"cz_knows_vs_gen",
         "Internal correctness knowledge (best-layer probe AUC) vs.\\ actual generation success. Points far above the diagonal are \\emph{generation-limited} -- the model represents correctness internally (AUC$\\approx$0.98) yet emits a correct kernel only a small fraction of the time; models at the wall cannot be probed at all.")

# ---------- 5. per-layer AUC curves by group ----------
def fig_layer_curves():
    rows = [m for m in M("interp") if m.get("per_layer_auc")]
    if not rows: return
    fig, ax = plt.subplots(figsize=(11,6))
    for i,m in enumerate(rows):
        a=[x for x in m["per_layer_auc"] if isinstance(x,(int,float))]; xs=np.linspace(0,1,len(a))
        ax.plot(xs,a,marker=".",label=f"{short(m['model'])} ({m['size_b']:g}B)",color=CB[i%len(CB)])
    ax.axhline(0.5,color="k",ls=":",alpha=0.5,label="chance")
    ax.set_xlabel("relative transformer depth"); ax.set_ylabel("correctness AUC"); ax.set_ylim(0.4,1.02)
    ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.set_title("Correctness decodability rises with depth (per-layer)")
    save(fig,"cz_layer_curves",
         "Layer-wise correctness decodability vs.\\ depth: the correct-vs-incorrect signal is weak at the embedding and sharpens through mid/late layers, indicating correctness is a computed, decodable feature rather than a surface one.")

# ---------- 6. dense all-runs DAG ----------
def fig_all_runs_dag():
    rows = [m for m in M("mech_analysis") if m.get("size_b")]
    if not rows: return
    fig, ax = plt.subplots(figsize=(16,9))
    rng=np.random.default_rng(0)
    for m in rows:
        x=m["size_b"]; drift=m.get("drift_early",0)+m.get("drift_mid",0)+m.get("drift_late",0)
        y=(1 if m.get("rsi") else 0)+ (0.4 if m.get("wall_crossed") else 0.0) + rng.normal(0,0.05)
        c=outcome_color(m)
        # connector from base (drift=0 baseline at y~0) to self
        bezier(ax, x, 0.0+rng.normal(0,0.03), x, y, c, lw=0.5+3*drift, alpha=0.15+0.4*drift)
        ax.scatter([x],[y], s=30+1200*drift, c=c, alpha=0.6, edgecolors="k", linewidths=0.3, zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("model size (B params, log)"); ax.set_ylabel("RSI outcome axis (wall $\\to$ flat $\\to$ compound)")
    ax.set_xticks([0.5,1,2,3,7,15]); ax.set_xticklabels(["0.5","1","2","3","7","15"])
    ax.axvspan(0.3,2,color=CB[1],alpha=0.05); ax.text(0.55,1.4,"correctness wall\n(<2B)",color=CB[1],fontsize=9)
    ax.set_title("All %d open-weight runs: size vs.\\ outcome, marker area & lift $\\propto$ LoRA drift" % len(rows))
    save(fig,"cz_all_runs_dag",
         "Dense view of every open-weight run: horizontal position = scale, vertical = RSI outcome, marker area and the base$\\to$self connector width $\\propto$ sustained LoRA drift. The sub-2B correctness wall (shaded) and the mid-scale high-drift compounding band emerge from the raw point cloud.")

# ---------- 7. failure-stage funnel (from correctness_rate distribution) ----------
def fig_failure_funnel():
    rows=[m for m in M("mech_analysis") if m.get("size_b")]
    if not rows: return
    n=len(rows)
    emits=[m for m in rows if m.get("wall_crossed")]                 # emits a correct kernel
    gains=[m for m in rows if m.get("rsi")]                          # correct+improves (compounds)
    stages=[("attempts",n),("emits correct kernel",len(emits)),("compounds (held-out gain)",len(gains))]
    fig, ax=plt.subplots(figsize=(11,5)); ax.axis("off")
    for i,(lab,v) in enumerate(stages):
        w=v/n; ax.add_patch(plt.Rectangle((0.5-w/2, -i), w, 0.7, color=CB[i%3], alpha=0.6))
        ax.text(0.5,-i+0.35,f"{lab}: {v}/{n} ({100*v/n:.0f}%)",ha="center",va="center",fontsize=11)
    ax.set_xlim(0,1); ax.set_ylim(-len(stages),1)
    ax.set_title("RSI attrition funnel across open-weight models")
    save(fig,"cz_failure_funnel",
         "Attrition funnel across open-weight models: the dominant drop is at emitting any verified-correct kernel (the correctness wall); a further drop separates one-shot correctness from sustained held-out compounding.")

for f in (fig_interp_heatmap, fig_internal_dag, fig_causal_flow, fig_knows_vs_gen, fig_layer_curves, fig_all_runs_dag, fig_failure_funnel):
    try: f()
    except Exception as e: print("skip", f.__name__, e)

# ---- splice section into figures.tex (idempotent, between markers) ----
BEG="% >>> CAUSALITY (auto)"; END="% <<< CAUSALITY (auto)"
sec=[BEG, "\\clearpage", "\\section{Large causality \\& internal-failure graphs}"]
for name,cap in OUT:
    sec += ["\\begin{figure}[H]\\centering",
            "\\includegraphics[width=0.98\\linewidth]{%s.pdf}" % name,
            "\\caption{%s}" % cap, "\\label{fig:%s}" % name, "\\end{figure}"]
sec.append(END); block="\n".join(sec)+"\n"
tex=open(TEX).read()
import re
tex=re.sub(re.escape(BEG)+r".*?"+re.escape(END)+r"\n?", "", tex, flags=re.S)
tex=tex.replace("\\end{document}", block+"\\end{document}")
open(TEX,"w").write(tex)
print("added %d causality figures; tex updated" % len(OUT))
