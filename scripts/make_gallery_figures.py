#!/usr/bin/env python3
"""Gallery of dense, beautiful publication graphs for the KernelAscent RSI paper (dense-DAG aesthetic:
lanes, translucent curved bezier edges, degree/metric-scaled markers). Reads docs/data/*.json ->
paper/figures/gz_*.{pdf,png} and splices a \\section into paper/figures.tex (idempotent, between markers).
Regenerate: python3 scripts/make_gallery_figures.py . Interp-driven panels thicken as the GPU interp sweep lands."""
import json, os, re, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs", "data"); FIG = os.path.join(ROOT, "paper", "figures")
TEX = os.path.join(ROOT, "paper", "figures.tex")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 200, "savefig.bbox": "tight"})
CB = ["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#F0E442","#999999","#117733","#882255"]

def load(n):
    try: return json.load(open(os.path.join(D, n + ".json")))
    except Exception: return {"models": []}
def M(n): return load(n).get("models", [])
def short(s): return (s or "").split("/")[-1].replace("-Instruct","").replace("-Base","").replace("-instruct","").replace("-base","")
def num(x): return x if isinstance(x, (int, float)) else None
OUT = []
def save(fig, name, caption):
    for e in ("pdf","png"): fig.savefig(os.path.join(FIG, name+"."+e))
    plt.close(fig); OUT.append((name, caption)); print("gfig:", name)
def bez(ax,x0,y0,x1,y1,color,lw=0.8,alpha=0.15,vert=False):
    if vert:
        my=(y0+y1)/2; p=Path([(x0,y0),(x0,my),(x1,my),(x1,y1)],[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4])
    else:
        mx=(x0+x1)/2; p=Path([(x0,y0),(mx,y0),(mx,y1),(x1,y1)],[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4])
    ax.add_patch(PathPatch(p, fc="none", ec=color, lw=lw, alpha=alpha))
def ocolor(m):
    if not m.get("wall_crossed"): return CB[1]
    if m.get("rsi"): return CB[2]
    return CB[0]

MA = M("mech_analysis")               # 50 models: size_b, drift_*, mean_retention, mean_diversity, mean_transfer_gap, wall_crossed, rsi, C_held_gain, drift_total
OUTC = {short(m["model"]): m for m in MA}
def interp_rows():
    r=[m for m in M("interp") if m.get("per_layer_auc")]; r.sort(key=lambda m:m.get("size_b",0)); return r

# 1. full-res model x layer AUC heatmap
def f_heat():
    rows=interp_rows()
    if not rows: return
    G=48; grid=np.linspace(0,1,G); mat=[]
    for m in rows:
        a=[x for x in m["per_layer_auc"] if isinstance(x,(int,float))]
        mat.append(np.interp(grid,np.linspace(0,1,len(a)),a))
    mat=np.array(mat)
    fig,ax=plt.subplots(figsize=(16,max(3,0.6*len(rows)+2)))
    im=ax.imshow(mat,aspect="auto",cmap="turbo",vmin=0.5,vmax=1.0,extent=[0,1,len(rows)-0.5,-0.5],interpolation="bilinear")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{short(m['model'])} ({m['size_b']:g}B)" for m in rows])
    for i,m in enumerate(rows):
        if m.get("best_layer_frac") is not None: ax.plot(m["best_layer_frac"],i,"w*",ms=15,mec="k")
    ax.set_xlabel("relative transformer depth"); fig.colorbar(im,ax=ax,label="correctness decodability AUC")
    ax.set_title("Per-layer correctness decodability across scale (white $\\star$ = peak layer)")
    save(fig,"gz_layer_heat","Full-resolution model$\\times$layer heatmap of how linearly decodable kernel-correctness is at each transformer depth (interpolated to a common grid, sorted by scale). The decodable-correctness signal concentrates in mid/late layers; it fills in as more interp probes land.")

# 2. ridgeline of per-layer AUC
def f_ridge():
    rows=interp_rows()
    if not rows: return
    fig,ax=plt.subplots(figsize=(14,max(4,1.1*len(rows))))
    for i,m in enumerate(rows[::-1]):
        a=np.array([x for x in m["per_layer_auc"] if isinstance(x,(int,float))]); xs=np.linspace(0,1,len(a))
        base=i*0.6; ax.fill_between(xs, base, base+(a-0.5)*1.6, color=CB[i%len(CB)], alpha=0.55, lw=1.2, ec="k")
        ax.text(-0.02, base+0.05, f"{short(m['model'])} ({m['size_b']:g}B)", ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("relative depth"); ax.set_yticks([]); ax.set_xlim(-0.25,1.02)
    ax.set_title("Correctness-signal ridgeline: where the correct-vs-incorrect representation peaks, by scale")
    save(fig,"gz_ridge","Ridgeline of per-layer correctness AUC (each curve a model, stacked by scale). Shows the internal correctness ``wave'' and at what depth it crests as models grow.")

# 3. layer node-link DAG
def f_layerdag():
    rows=interp_rows()
    if len(rows)<1: return
    fig,ax=plt.subplots(figsize=(16,max(4,0.7*len(rows)+2)))
    for i,m in enumerate(rows):
        a=[x for x in m["per_layer_auc"] if isinstance(x,(int,float))]; xs=np.linspace(0,1,len(a))
        for j in range(len(a)-1): bez(ax,xs[j],i,xs[j+1],i,CB[i%len(CB)],lw=0.6,alpha=0.5)
        ax.scatter(xs,[i]*len(a),s=[max(4,(v-0.5)*260) for v in a],c=[v for v in a],cmap="turbo",vmin=0.5,vmax=1,ec="k",lw=0.2,zorder=3)
        ax.text(-0.02,i,f"{short(m['model'])}",ha="right",va="center",fontsize=8)
    ax.set_xlabel("relative depth"); ax.set_yticks([]); ax.set_xlim(-0.2,1.02); ax.invert_yaxis()
    ax.set_title("Layer node-link graph: marker area $\\propto$ per-layer correctness AUC")
    save(fig,"gz_layer_nodelink","Node-link view of the correctness signal: one node per (model, layer), area and colour $\\propto$ decodability AUC, translucent edges tracing depth. Dense bright bands mark the depth range where correctness becomes linearly readable.")

# 4. mechanism metric correlation chords
def f_chord():
    keys=[("drift_total","drift"),("mean_retention","retention"),("mean_diversity","diversity"),("mean_transfer_gap","transfer gap"),("C_held_gain","held gain"),("size_b","size"),("max_C_train","correct rate")]
    cols=[k for k,_ in keys]; data=[]
    for m in MA:
        row=[num(m.get(k)) for k in cols]
        if all(v is not None for v in row): data.append(row)
    if len(data)<5: return
    A=np.array(data); C=np.corrcoef(A.T); n=len(cols)
    ang=np.linspace(0,2*np.pi,n,endpoint=False); pos=np.c_[np.cos(ang),np.sin(ang)]
    fig,ax=plt.subplots(figsize=(11,11)); ax.set_aspect("equal"); ax.axis("off")
    for i in range(n):
        for j in range(i+1,n):
            r=C[i,j]
            if abs(r)<0.15: continue
            col=CB[2] if r>0 else CB[1]
            p=Path([tuple(pos[i]),(0,0),(0,0),tuple(pos[j])],[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4])
            ax.add_patch(PathPatch(p,fc="none",ec=col,lw=1+6*abs(r),alpha=0.35+0.4*abs(r)))
    for (i,(_,lab)) in enumerate(keys):
        ax.plot(*pos[i],"o",ms=16,color=CB[7]); ax.text(pos[i][0]*1.13,pos[i][1]*1.13,lab,ha="center",va="center",fontsize=11)
    ax.set_xlim(-1.35,1.35); ax.set_ylim(-1.35,1.35)
    ax.set_title("Mechanism-metric correlation chords (green +, orange $-$; width $\\propto|r|$) across %d runs"%len(data))
    save(fig,"gz_chord","Circular correlation graph of mechanism metrics across all open-weight runs. Sustained drift and retention co-vary with held-out gain (green chords); transfer-gap and diversity relate oppositely -- the internal fingerprint of compounding.")

# 5. per-round C_held spaghetti by outcome
def f_spaghetti():
    rows=[m for m in M("rsi_mech") if m.get("C_held")]
    if not rows: return
    fig,ax=plt.subplots(figsize=(14,8))
    for m in rows:
        o=OUTC.get(short(m["model"]),{}); c=ocolor(o) if o else CB[7]
        y=[v for v in m["C_held"] if isinstance(v,(int,float))]; ax.plot(range(len(y)),y,color=c,alpha=0.5,lw=1.4)
    ax.set_xlabel("round"); ax.set_ylabel("held-out C");
    ax.legend(handles=[Line2D([],[],color=CB[2],label="RSI compounds"),Line2D([],[],color=CB[0],label="crossed wall, flat"),Line2D([],[],color=CB[1],label="stuck at wall")])
    ax.set_title("Per-round held-out capability, all %d weight-RSI runs (coloured by outcome)"%len(rows))
    save(fig,"gz_spaghetti","Every weight-RSI run's held-out capability over rounds, coloured by outcome. Compounders separate upward; wall-stuck runs stay pinned at zero -- the dynamics behind the static leaderboard.")

# 6. scale x outcome bubble with marginals
def f_bubble():
    rows=[m for m in MA if num(m.get("size_b")) and num(m.get("C_held_gain"))is not None]
    if not rows: return
    fig=plt.figure(figsize=(14,10)); gs=fig.add_gridspec(2,2,width_ratios=[4,1],height_ratios=[1,4],hspace=0.04,wspace=0.04)
    ax=fig.add_subplot(gs[1,0]); axt=fig.add_subplot(gs[0,0],sharex=ax); axr=fig.add_subplot(gs[1,1],sharey=ax)
    xs=[math.log10(m["size_b"]) for m in rows]; ys=[m["C_held_gain"] for m in rows]
    sz=[30+900*(m.get("drift_total") or 0) for m in rows]; cs=[ocolor(m) for m in rows]
    ax.axvspan(math.log10(0.3),math.log10(2),color="0.9",zorder=0); ax.text(math.log10(0.6),max(ys),"correctness wall",fontsize=9,color="0.4")
    ax.scatter(xs,ys,s=sz,c=cs,alpha=0.7,ec="k",lw=0.4)
    ax.axhline(0,color="0.6",lw=0.8); ax.set_xlabel("log10 params (B)"); ax.set_ylabel("held-out C gain over rounds")
    axt.hist(xs,bins=14,color=CB[0],alpha=0.6); axt.axis("off"); axr.hist(ys,bins=14,orientation="horizontal",color=CB[2],alpha=0.6); axr.axis("off")
    axt.set_title("Scale vs.\\ RSI gain (area $\\propto$ LoRA drift) with marginals",fontsize=12)
    save(fig,"gz_bubble","Every run positioned by scale (x) and held-out gain (y), bubble area $\\propto$ sustained drift, colour = outcome, with marginal histograms. The sub-2B wall and the mid-scale compounding band are visible in the joint and marginal densities.")

# 7. drift-by-depth x scale heatmap
def f_driftdepth():
    rows=[m for m in MA if num(m.get("size_b")) and any(num(m.get(k))for k in("drift_early","drift_mid","drift_late"))]
    rows.sort(key=lambda m:m["size_b"])
    if not rows: return
    mat=np.array([[m.get("drift_early")or 0,m.get("drift_mid")or 0,m.get("drift_late")or 0] for m in rows])
    fig,ax=plt.subplots(figsize=(9,max(4,0.32*len(rows)+2)))
    im=ax.imshow(mat,aspect="auto",cmap="viridis",extent=[0,3,len(rows)-0.5,-0.5])
    ax.set_xticks([0.5,1.5,2.5]); ax.set_xticklabels(["early","mid","late"]);
    step=max(1,len(rows)//30); ax.set_yticks(range(0,len(rows),step)); ax.set_yticklabels([f"{short(rows[i]['model'])} ({rows[i]['size_b']:g}B)" for i in range(0,len(rows),step)],fontsize=7)
    fig.colorbar(im,ax=ax,label="LoRA drift"); ax.set_title("Where plasticity lives: LoRA drift by depth $\\times$ scale")
    save(fig,"gz_driftdepth","LoRA weight drift by transformer block (early/mid/late) for every run, sorted by scale -- where in the network self-training actually moves weights, and how that shifts with size.")

# 8. correlation clustermap of metrics
def f_corr():
    keys=[("size_b","size"),("max_C_train","correct"),("drift_total","drift"),("mean_retention","retain"),("mean_diversity","diverse"),("mean_transfer_gap","transfer"),("C_held_gain","gain"),("rsi","RSI")]
    cols=[k for k,_ in keys]; data=[[ (1.0 if k=="rsi" and m.get("rsi") else 0.0 if k=="rsi" else num(m.get(k))) for k in cols] for m in MA]
    data=[r for r in data if all(v is not None for v in r)]
    if len(data)<5: return
    C=np.corrcoef(np.array(data).T); n=len(cols)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels([l for _,l in keys],rotation=45,ha="right"); ax.set_yticks(range(n)); ax.set_yticklabels([l for _,l in keys])
    for i in range(n):
        for j in range(n): ax.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",fontsize=8,color="k" if abs(C[i,j])<0.6 else "w")
    fig.colorbar(im,ax=ax,label="Pearson r"); ax.set_title("Mechanism-metric correlation matrix (%d runs)"%len(data))
    save(fig,"gz_corr","Correlation matrix of mechanism metrics and RSI outcome across all runs -- drift and retention are the strongest positive correlates of held-out gain; the correctness-rate gate dominates.")

# 9. retention vs drift phase plane
def f_phase():
    rows=[m for m in MA if num(m.get("drift_total"))is not None and num(m.get("mean_retention"))is not None]
    if not rows: return
    fig,ax=plt.subplots(figsize=(11,9))
    xs=[m["drift_total"]for m in rows]; ys=[m["mean_retention"]for m in rows]
    sz=[40+700*max(0,(m.get("C_held_gain")or 0)) for m in rows]; cs=[ocolor(m) for m in rows]
    ax.scatter(xs,ys,s=sz,c=cs,alpha=0.7,ec="k",lw=0.4)
    ax.set_xlabel("sustained LoRA drift"); ax.set_ylabel("retention");
    ax.legend(handles=[Line2D([],[],marker='o',color='w',markerfacecolor=CB[2],ms=11,label='compounds'),Line2D([],[],marker='o',color='w',markerfacecolor=CB[0],ms=11,label='flat'),Line2D([],[],marker='o',color='w',markerfacecolor=CB[1],ms=11,label='wall')])
    ax.set_title("Two-gates phase plane: drift $\\times$ retention (marker $\\propto$ held-out gain)")
    save(fig,"gz_phase","Phase plane of the second gate: compounders occupy the high-drift/high-retention corner; flat runs cluster at low drift (saturation) or low retention (forgetting). Marker area $\\propto$ held-out gain.")

# 10. T5 L-F small multiples
def f_lf_grid():
    rows=[m for m in (M("selfplay")+M("selfplay_closed")) if m.get("L_minus_F")]
    rows=[m for m in rows if any(isinstance(v,(int,float)) for v in m["L_minus_F"])]
    if not rows: return
    n=len(rows); cols=min(6,n); r=math.ceil(n/cols)
    fig,axs=plt.subplots(r,cols,figsize=(2.6*cols,2.2*r),squeeze=False)
    for k,m in enumerate(rows):
        ax=axs[k//cols][k%cols]; y=[v for v in m["L_minus_F"] if isinstance(v,(int,float))]
        ax.axhline(0,color="0.7",lw=0.7); ax.plot(range(len(y)),y,color=CB[2] if (m.get("final_L_minus_F")or 0)>0.03 else CB[1],lw=1.6)
        ax.set_title(short(m["model"])[:16],fontsize=7); ax.tick_params(labelsize=6)
    for k in range(n,r*cols): axs[k//cols][k%cols].axis("off")
    fig.suptitle("Task 5 author co-evolution $L-F$ per round, every self-play run",fontsize=12)
    save(fig,"gz_lf_grid","Per-round $L-F$ (live- minus frozen-author) for every open and closed self-play run. Most hover at zero (adaptive-curriculum only); a few sustain $>0$ (genuine author co-evolution) -- the headline metric, run by run.")

# 11. parallel coordinates
def f_parallel():
    keys=[("size_b","size"),("max_C_train","correct"),("drift_total","drift"),("mean_retention","retain"),("mean_diversity","diverse"),("C_held_gain","gain")]
    rows=[m for m in MA if all(num(m.get(k))is not None for k,_ in keys)]
    if len(rows)<5: return
    cols=[k for k,_ in keys]; A=np.array([[m[k] for k in cols] for rows_ in [rows] for m in rows],dtype=float)
    mn=A.min(0); rng=np.where(A.max(0)-mn==0,1,A.max(0)-mn); N=(A-mn)/rng
    fig,ax=plt.subplots(figsize=(14,8))
    for i,m in enumerate(rows):
        ax.plot(range(len(cols)),N[i],color=ocolor(m),alpha=0.4,lw=1.2)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _,l in keys]); ax.set_yticks([])
    ax.legend(handles=[Line2D([],[],color=CB[2],label='compounds'),Line2D([],[],color=CB[0],label='flat'),Line2D([],[],color=CB[1],label='wall')])
    ax.set_title("Parallel coordinates: mechanism profile of every run (normalised), coloured by outcome")
    save(fig,"gz_parallel","Parallel-coordinates fingerprint of every run across the mechanism axes, coloured by outcome -- compounders share a high-correct/high-drift/high-retention profile that the flat and wall runs lack.")

# 12. giant poster
def f_poster():
    if not MA: return
    fig=plt.figure(figsize=(18,11)); gs=fig.add_gridspec(2,2,hspace=0.22,wspace=0.18)
    # panel A: scale->gates bars
    axA=fig.add_subplot(gs[0,0]); bins={"<2B":[],"2-8B":[],">=9B":[]}
    for m in MA:
        s=m.get("size_b",0); b="<2B" if s<2 else ("2-8B" if s<9 else ">=9B"); bins[b].append(m)
    labels=list(bins); cross=[np.mean([1 if x.get("wall_crossed") else 0 for x in bins[b]]) if bins[b] else 0 for b in labels]
    rsi=[np.mean([1 if x.get("rsi") else 0 for x in bins[b]]) if bins[b] else 0 for b in labels]
    x=np.arange(len(labels)); axA.bar(x-0.2,cross,0.4,label="cross wall",color=CB[0]); axA.bar(x+0.2,rsi,0.4,label="RSI compound",color=CB[2])
    axA.set_xticks(x); axA.set_xticklabels(labels); axA.legend(fontsize=8); axA.set_title("A. Two gates by scale",fontsize=11)
    # panel B: drift x retention phase
    axB=fig.add_subplot(gs[0,1])
    for m in MA:
        if num(m.get("drift_total"))is not None and num(m.get("mean_retention"))is not None:
            axB.scatter(m["drift_total"],m["mean_retention"],s=40+600*max(0,(m.get("C_held_gain")or 0)),c=ocolor(m),alpha=0.7,ec="k",lw=0.3)
    axB.set_xlabel("drift"); axB.set_ylabel("retention"); axB.set_title("B. Second gate (drift$\\times$retention)",fontsize=11)
    # panel C: interp heatmap strip or knows-vs-gen
    axC=fig.add_subplot(gs[1,0]); ir=interp_rows()
    if ir:
        G=40; mat=np.array([np.interp(np.linspace(0,1,G),np.linspace(0,1,len([x for x in m["per_layer_auc"] if isinstance(x,(int,float))])),[x for x in m["per_layer_auc"] if isinstance(x,(int,float))]) for m in ir])
        im=axC.imshow(mat,aspect="auto",cmap="turbo",vmin=0.5,vmax=1,extent=[0,1,len(ir)-0.5,-0.5]); axC.set_yticks(range(len(ir))); axC.set_yticklabels([short(m['model']) for m in ir],fontsize=7); axC.set_xlabel("depth"); fig.colorbar(im,ax=axC,label="AUC")
    axC.set_title("C. Where correctness is encoded",fontsize=11)
    # panel D: scale vs gain cloud
    axD=fig.add_subplot(gs[1,1])
    for m in MA:
        if num(m.get("size_b")) and num(m.get("C_held_gain"))is not None:
            axD.scatter(math.log10(m["size_b"]),m["C_held_gain"],s=30+800*(m.get("drift_total")or 0),c=ocolor(m),alpha=0.7,ec="k",lw=0.3)
    axD.axhline(0,color="0.6",lw=0.8); axD.set_xlabel("log10 params"); axD.set_ylabel("held-out gain"); axD.set_title("D. Scale vs.\\ gain",fontsize=11)
    fig.suptitle("KernelAscent: when recursive self-improvement compounds vs.\\ collapses -- the internal mechanism at a glance",fontsize=14)
    save(fig,"gz_poster","Single-figure synthesis: (A) the two scale gates, (B) the drift$\\times$retention second gate, (C) where correctness is internally encoded by depth, (D) scale vs.\\ held-out gain. The full mechanistic story of the benchmark in one poster.")

for f in (f_heat,f_ridge,f_layerdag,f_chord,f_spaghetti,f_bubble,f_driftdepth,f_corr,f_phase,f_lf_grid,f_parallel,f_poster):
    try: f()
    except Exception as e: print("skip",f.__name__,e)

BEG="% >>> GALLERY (auto)"; END="% <<< GALLERY (auto)"
sec=[BEG,"\\clearpage","\\section{Gallery --- dense graphs}"]
for name,cap in OUT:
    sec+=["\\begin{figure}[H]\\centering","\\includegraphics[width=0.98\\linewidth]{%s.pdf}"%name,"\\caption{%s}"%cap,"\\label{fig:%s}"%name,"\\end{figure}"]
sec.append(END); block="\n".join(sec)+"\n"
tex=open(TEX).read()
tex=re.sub(re.escape(BEG)+r".*?"+re.escape(END)+r"\n?","",tex,flags=re.S)
tex=tex.replace("\\end{document}",block+"\\end{document}")
open(TEX,"w").write(tex)
print("added %d gallery figures; tex updated"%len(OUT))
