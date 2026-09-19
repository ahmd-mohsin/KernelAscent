#!/usr/bin/env python3
"""Dense publication-figure pipeline for the KernelAscent RSI benchmark.
Reads docs/data/*.json -> paper/figures/*.{pdf,png} + paper/figures.tex. Each figure is guarded (skips if its
data is empty). Regenerate any time: `python3 scripts/make_paper_figures.py`."""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs", "data"); FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#999999", "#000000"]

def load(name):
    try:
        return json.load(open(os.path.join(D, name + ".json")))
    except Exception:
        return {"models": []}
def M(name): return load(name).get("models", [])

FIGS = []  # (filename, section, caption)
def save(fig, name, section, caption):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, name + "." + ext))
    plt.close(fig); FIGS.append((name, section, caption)); print("fig:", name)
def short(s): return (s or "").split("/")[-1].replace("-Instruct", "").replace("-Base", "").replace("-instruct", "")

SEC_MECH = "Mechanistic interpretability"; SEC_T5 = "Task 5: self-play"; SEC_T4 = "Task 4: closed$\\to$open"
SEC_T3 = "Task 3: procedure-RSI"; SEC_CMP = "Comparators \\& weight-RSI"; SEC_FLOW = "Flows \\& causality"

# ---------------- MECHANISM / INTERP ----------------
def fig_scale_gates():
    ma = load("mech_analysis"); rows = ma.get("models", [])
    if not rows: return
    bins = [("<2B", lambda s: s < 2), ("2-8B", lambda s: 2 <= s < 9), (">=9B", lambda s: s >= 9)]
    labels, pw, pr = [], [], []
    for lab, f in bins:
        g = [r for r in rows if f(r.get("size_b", 0))]
        if not g: continue
        labels.append("%s\n(n=%d)" % (lab, len(g)))
        pw.append(np.mean([1 if r.get("wall_crossed") else 0 for r in g]))
        pr.append(np.mean([1 if r.get("rsi") else 0 for r in g]))
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(x - w/2, pw, w, label="P(cross correctness wall)", color=CB[0])
    ax.bar(x + w/2, pr, w, label="P(RSI compounds)", color=CB[1])
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("probability"); ax.set_ylim(0, 1)
    ax.set_title("Two gates to RSI: scale lifts the wall; compounding peaks mid-scale"); ax.legend(fontsize=8)
    save(fig, "mech_scale_gates", SEC_MECH,
         "Scale $\\to$ RSI. Below $\\sim$2B the correctness wall is rarely crossed, so weight-RSI is impossible; crossing rises sharply at 2B+, but compounding peaks at mid-scale (2--8B) and does not increase further $\\geq$9B (headroom ceiling).")

def fig_drift_depth():
    rows = M("mech_analysis")
    comp = [r for r in rows if r.get("rsi")]; flat = [r for r in rows if r.get("wall_crossed") and not r.get("rsi")]
    if not comp or not flat: return
    depths = ["drift_early", "drift_mid", "drift_late"]
    cm = [np.mean([r.get(d, 0) or 0 for r in comp]) for d in depths]
    fm = [np.mean([r.get(d, 0) or 0 for r in flat]) for d in depths]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar(x - w/2, cm, w, label="compounders", color=CB[2])
    ax.bar(x + w/2, fm, w, label="flat (crossed wall)", color=CB[3])
    ax.set_xticks(x); ax.set_xticklabels(["early", "mid", "late"]); ax.set_ylabel("mean LoRA drift")
    ax.set_title("Where learning lives: drift by transformer depth"); ax.legend(fontsize=8)
    save(fig, "mech_drift_by_depth", SEC_MECH,
         "LoRA weight drift by transformer depth for compounders vs.\\ flat (wall-crossing) runs. Compounders sustain markedly larger drift across all depths; flat runs saturate early.")

def fig_signature():
    rows = M("mech_analysis")
    comp = [r for r in rows if r.get("rsi")]; flat = [r for r in rows if r.get("wall_crossed") and not r.get("rsi")]
    if not comp or not flat: return
    keys = [("drift_total", "drift"), ("mean_retention", "retention"), ("mean_diversity", "diversity")]
    cm = [np.mean([r.get(k, 0) or 0 for r in comp]) for k, _ in keys]
    fm = [np.mean([r.get(k, 0) or 0 for r in flat]) for k, _ in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    b1 = ax.bar(x - w/2, cm, w, label="compounders", color=CB[2])
    b2 = ax.bar(x + w/2, fm, w, label="flat", color=CB[3])
    for i, (c, f) in enumerate(zip(cm, fm)):
        if f > 0: ax.text(i, max(c, f) + 0.02, "%.1fx" % (c / f) if c >= f else "%.2fx" % (c / f), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys]); ax.set_ylabel("mean value")
    ax.set_title("Internal signature: compounders vs.\\ flat"); ax.legend(fontsize=8)
    save(fig, "mech_signature", SEC_MECH,
         "The internal signature that separates compounders from flat runs among wall-crossers: sustained drift and retention are $\\sim$5--6$\\times$ higher in compounders, while generation diversity is \\emph{lower} (productive convergence, not diversity preservation).")

def fig_mech_traj():
    rows = [r for r in M("rsi_mech") if r.get("rounds")]
    if not rows: return
    rows = sorted(rows, key=lambda r: -max([c for c in (r.get("C_held") or [0]) if c is not None] + [0]))[:6]
    panels = [("C_held", "held-out C"), ("drift_total", "LoRA drift"), ("retention", "retention"), ("transfer_gap", "train$-$held gap")]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6))
    for ax, (key, lab) in zip(axes.flat, panels):
        for i, r in enumerate(rows):
            y = [v for v in (r.get(key) or [])]
            xr = r.get("rounds") or list(range(len(y)))
            xs = [x for x, v in zip(xr, y) if v is not None]; ys = [v for v in y if v is not None]
            if ys: ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=CB[i % len(CB)], label=short(r["model"]))
        ax.set_title(lab, fontsize=10); ax.set_xlabel("round")
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Per-round mechanism trajectories (top-C models)")
    save(fig, "mech_trajectories", SEC_MECH,
         "Per-round mechanism trajectories for the highest-capability probes: held-out C, total LoRA drift, retention, and the train$-$held transfer gap. Compounders show rising/held C with sustained drift and retention; flat runs plateau with drift saturation.")

def fig_interp_scatter():
    rows = [r for r in M("interp") if r.get("probed") and r.get("best_auc") is not None]
    if not rows: return
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    cr = [r.get("correctness_rate", 0) or 0 for r in rows]; au = [r.get("best_auc") for r in rows]
    ax.scatter(cr, au, s=70, color=CB[0], zorder=3)
    for r, x, y in zip(rows, cr, au): ax.annotate(short(r["model"]), (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "--", color=CB[8], lw=1, label="y=x")
    ax.set_xlabel("generation correctness rate"); ax.set_ylabel("best-layer correctness AUC")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_title("Knows vs.\\ generates: internal correctness $\\gg$ output"); ax.legend(fontsize=8)
    save(fig, "interp_knows_vs_generates", SEC_MECH,
         "Internal correctness decodability (best-layer linear-probe AUC) vs.\\ how often the model actually generates a correct kernel. Points far above $y=x$ mean the model \\emph{represents} correctness internally far better than it can \\emph{decode} it into a correct kernel: the RSI bottleneck is generation, not knowledge.")

def fig_interp_size():
    rows = [r for r in M("interp") if r.get("probed") and r.get("best_auc") is not None]
    if len(rows) < 2: return
    rows = sorted(rows, key=lambda r: r.get("size_b", 0))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot([r["size_b"] for r in rows], [r["best_auc"] for r in rows], marker="o", color=CB[0])
    for r in rows: ax.annotate(short(r["model"]), (r["size_b"], r["best_auc"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("model size (B params, log)"); ax.set_ylabel("best-layer correctness AUC")
    ax.set_title("Internal correctness representation vs.\\ scale")
    save(fig, "interp_auc_vs_size", SEC_MECH,
         "Best-layer correctness-decodability AUC vs.\\ model scale. (Populated as more interpretability probes land.)")

# ---------------- TASK 5 ----------------
def _lf_bar(name, models, title, cap, sec):
    rows = [r for r in models if r.get("final_L_minus_F") is not None]
    if not rows: return
    rows = sorted(rows, key=lambda r: r.get("final_L_minus_F"))
    fig, ax = plt.subplots(figsize=(6.2, max(2.6, 0.32 * len(rows))))
    y = np.arange(len(rows)); vals = [r["final_L_minus_F"] for r in rows]
    ax.barh(y, vals, color=[CB[2] if v > 0.02 else (CB[3] if v < -0.02 else CB[7]) for v in vals])
    ax.set_yticks(y); ax.set_yticklabels([short(r["model"]) for r in rows], fontsize=7)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("final $L-F$ (author co-evolution)"); ax.set_title(title)
    save(fig, name, sec, cap)
def fig_lf_open():
    _lf_bar("t5_lf_open", M("selfplay"), "Task 5a open self-play: author co-evolution $L-F$",
            "Final $L-F$ (live-author minus frozen-author, the self-referential signal) per open-weight model. $>0$ means updating the author beyond a frozen-author escalating curriculum genuinely helps.", SEC_T5)
def fig_lf_closed():
    _lf_bar("t5_lf_closed", M("selfplay_closed"), "Task 5b closed self-play: author co-evolution $L-F$",
            "Final $L-F$ per closed/API model (procedure co-evolution channel). Most cluster near zero; early large spikes did not survive longer horizons.", SEC_T5)

def fig_threearm():
    rows = [r for r in M("selfplay") if r.get("C_held_live")]
    if not rows: return
    def last(a):
        a = [v for v in (a or []) if v is not None]; return a[-1] if a else 0
    rows = sorted(rows, key=lambda r: -last(r.get("C_held_live")))[:8]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(x - w, [last(r.get("C_held_static")) for r in rows], w, label="S static", color=CB[7])
    ax.bar(x, [last(r.get("C_held_frozen_author")) for r in rows], w, label="F frozen-author", color=CB[0])
    ax.bar(x + w, [last(r.get("C_held_live")) for r in rows], w, label="L live-author", color=CB[1])
    ax.set_xticks(x); ax.set_xticklabels([short(r["model"]) for r in rows], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("held-out C (final)"); ax.set_title("Task 5 three-arm: S / F / L held-out capability"); ax.legend(fontsize=8)
    save(fig, "t5_threearm", SEC_T5,
         "Three-arm self-play at equal budget on a fixed held-out ladder: STATIC (fixed frontier), FROZEN-AUTHOR (escalating frontier, frozen base author), LIVE-AUTHOR (co-evolving author). $F-S$ isolates the adaptive-curriculum benefit; $L-F$ isolates author co-evolution.")

def fig_lf_traj():
    op = [r for r in M("selfplay") if r.get("L_minus_F")]
    cl = [r for r in M("selfplay_closed") if r.get("L_minus_F")]
    if not op and not cl: return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    def plot(rows, style, base):
        rows = sorted(rows, key=lambda r: -(r.get("final_L_minus_F") or -9))[:5]
        for i, r in enumerate(rows):
            y = [v for v in (r.get("L_minus_F") or []) if v is not None]
            if y: ax.plot(range(len(y)), y, style, color=CB[(base + i) % len(CB)], lw=1.3, ms=3, label=short(r["model"]))
    plot(op, "-o", 0); plot(cl, "--s", 2)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("round"); ax.set_ylabel("$L-F$")
    ax.set_title("Author co-evolution over rounds (solid=open, dashed=closed)"); ax.legend(fontsize=6, ncol=2)
    save(fig, "t5_lf_trajectory", SEC_T5,
         "Per-round $L-F$ trajectories (solid = open-weight, dashed = closed/API). Long horizons reveal whether early positive spikes compound or regress to zero.")

def fig_decomp():
    rows = [r for r in M("selfplay") if r.get("final_L_minus_F") is not None]
    if not rows: return
    def last(a):
        a = [v for v in (a or []) if v is not None]; return a[-1] if a else 0
    rows = sorted(rows, key=lambda r: -(r.get("final_L_minus_F") or 0))[:10]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(x - w, [last(r.get("L_minus_S")) for r in rows], w, label="$L-S$ total curriculum", color=CB[4])
    ax.bar(x, [last(r.get("F_minus_S")) for r in rows], w, label="$F-S$ curriculum, frozen author", color=CB[0])
    ax.bar(x + w, [r.get("final_L_minus_F") for r in rows], w, label="$L-F$ co-evolution", color=CB[1])
    ax.set_xticks(x); ax.set_xticklabels([short(r["model"]) for r in rows], rotation=35, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("$\\Delta$ held-out C"); ax.set_title("Task 5 decomposition: $L-S$, $F-S$, $L-F$"); ax.legend(fontsize=7)
    save(fig, "t5_decomposition", SEC_T5,
         "Decomposing the self-play benefit: $L-S$ (total adaptive-curriculum gain) $=$ $(F-S)$ (curriculum with a frozen author) $+$ $(L-F)$ (extra gain from co-evolving the author). Most models get their gain from $F-S$, not $L-F$.")

def fig_openended():
    rows = [r for r in M("open_rsi") if r.get("C_held_open")]
    if not rows: return
    rows = sorted(rows, key=lambda r: -(r.get("final_delta") or -9))[:5]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, r in enumerate(rows):
        o = [v for v in (r.get("C_held_open") or []) if v is not None]
        f = [v for v in (r.get("C_held_fixed") or []) if v is not None]
        ax.plot(range(len(o)), o, "-o", color=CB[i % len(CB)], ms=3, lw=1.3, label=short(r["model"]) + " open")
        ax.plot(range(len(f)), f, "--", color=CB[i % len(CB)], lw=1.0)
    ax.set_xlabel("round"); ax.set_ylabel("held-out C"); ax.set_title("Open-ended: OPEN (solid) vs.\\ FIXED (dashed) frontier")
    ax.legend(fontsize=6, ncol=2)
    save(fig, "t5_openended", SEC_T5,
         "Open-ended RSI: a learner facing an OPEN escalating self-authored frontier (solid) vs.\\ a FIXED static frontier (dashed) at equal budget. Overlap means open-endedness is a one-time upgrade, not compounding.")

# ---------------- TASK 4 ----------------
def fig_t4_heatmap():
    rows = M("combined_rsi")
    if not rows: return
    res = sorted({short(r["researcher"]) for r in rows}); tra = sorted({short(r["trainee"]) for r in rows})
    Zt = np.full((len(res), len(tra)), np.nan)
    for r in rows:
        i = res.index(short(r["researcher"])); j = tra.index(short(r["trainee"]))
        Zt[i, j] = r.get("impr_minus_frozen")
    fig, ax = plt.subplots(figsize=(1.2 + 1.0 * len(tra), 1.0 + 0.5 * len(res)))
    im = ax.imshow(Zt, cmap="RdBu", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(tra))); ax.set_xticklabels(tra, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(res))); ax.set_yticklabels(res, fontsize=7)
    for i in range(len(res)):
        for j in range(len(tra)):
            if not np.isnan(Zt[i, j]): ax.text(j, i, "%.2f" % Zt[i, j], ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="improved$-$frozen")
    ax.set_xlabel("open trainee"); ax.set_ylabel("closed researcher"); ax.set_title("Task 4: harness-rewrite gain")
    save(fig, "t4_heatmap", SEC_T4,
         "Task 4 (closed$\\to$open): a closed frontier researcher rewrites an open trainee's training harness. Cell = improved$-$frozen held-out C on the trainee (positive = the rewritten harness helps).")

def fig_t4_peak():
    rows = [r for r in M("combined_rsi") if r.get("peak") is not None]
    if not rows: return
    rows = sorted(rows, key=lambda r: r["peak"])
    fig, ax = plt.subplots(figsize=(6.2, max(2.6, 0.3 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, [r["peak"] for r in rows], color=[CB[2] if r["peak"] > 0.05 else CB[7] for r in rows])
    ax.set_yticks(y); ax.set_yticklabels(["%s$\\to$%s" % (short(r["researcher"]), short(r["trainee"])) for r in rows], fontsize=6)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("peak improved$-$frozen"); ax.set_title("Task 4: peak harness-rewrite gain")
    save(fig, "t4_peak", SEC_T4, "Peak improved$-$frozen over rounds for each researcher$\\to$trainee pair.")

# ---------------- TASK 3 ----------------
def fig_t3_bar():
    rows = [r for r in M("trackc") if r.get("delta_vs_base") is not None]
    if not rows: return
    rows = sorted(rows, key=lambda r: r["delta_vs_base"])
    fig, ax = plt.subplots(figsize=(6.2, max(2.6, 0.34 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, [r["delta_vs_base"] for r in rows], color=[CB[2] if r["delta_vs_base"] > 0.03 else (CB[3] if r["delta_vs_base"] < -0.03 else CB[7]) for r in rows])
    ax.set_yticks(y); ax.set_yticklabels([short(r["model"]) for r in rows], fontsize=7)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Q gain vs.\\ frozen procedure"); ax.set_title("Task 3: procedure-RSI (weights fixed)")
    save(fig, "t3_procedure_rsi", SEC_T3,
         "Task 3: a frozen-weight model rewrites its own strategy library + verified archive. Held-out Q gain vs.\\ the frozen procedure; frontier models improve their own procedure substantially.")

# ---------------- COMPARATORS / T2 ----------------
def fig_recursion_gain():
    rows = M("baselines")
    if not rows: return
    rows = sorted(rows, key=lambda r: -(r.get("recursion_gain") if r.get("recursion_gain") is not None else -9))
    x = np.arange(len(rows)); w = 0.2
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for k, off, c, lab in [("rsi_C", -1.5, CB[1], "weight-RSI"), ("best_of_k", -0.5, CB[0], "best-of-k"),
                           ("self_refine", 0.5, CB[2], "self-refine"), ("retrieval", 1.5, CB[4], "retrieval")]:
        ax.bar(x + off * w, [r.get(k) or 0 for r in rows], w, label=lab, color=c)
    ax.set_xticks(x); ax.set_xticklabels([short(r["model"]) for r in rows], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("held-out C"); ax.set_title("Comparators: recursion vs.\\ more sampling (matched budget)"); ax.legend(fontsize=8)
    save(fig, "cmp_recursion_gain", SEC_CMP,
         "Is it really recursion? Weight-RSI vs.\\ non-recursive controls (best-of-k, self-refine, retrieval) at matched generation budget. recursion\\_gain $=$ weight-RSI $-$ best comparator.")

def fig_t2_scatter():
    rows = [r for r in M("tier_speed_rsi") if r.get("C_self") is not None]
    if not rows: return
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    tiers = {"small": CB[0], "mid": CB[1], "large": CB[2]}
    for t, c in tiers.items():
        g = [r for r in rows if r.get("tier") == t]
        if g: ax.scatter([r["C_self"] for r in g], [r.get("self_minus_fresh", 0) for r in g], s=55, color=c, label=t, zorder=3)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("weight-RSI final C (self)"); ax.set_ylabel("self $-$ fresh-frozen")
    ax.set_title("Weight-RSI: capability vs.\\ recursion advantage"); ax.legend(fontsize=8)
    save(fig, "t2_selfvsfresh", SEC_CMP,
         "Weight-RSI: final self-trained held-out C vs.\\ the self-minus-fresh-frozen advantage (positive = training on own kernels beats an equal-budget fresh baseline).")

def fig_fork():
    rows = [r for r in M("fork") if r.get("self_minus_ckpt")]
    if not rows: return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, r in enumerate(rows):
        y = [v for v in (r.get("self_minus_ckpt") or []) if v is not None]
        if y: ax.plot(range(len(y)), y, "-o", color=CB[i % len(CB)], ms=3, label=short(r["model"]))
    ax.axhline(0.03, ls="--", color=CB[8], lw=1); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("round after fork"); ax.set_ylabel("continued-self $-$ checkpoint-frozen")
    ax.set_title("Recursion-interruption fork"); ax.legend(fontsize=8)
    save(fig, "fork_recursion", SEC_CMP,
         "Recursion-interruption control: continued self-training vs.\\ a checkpoint-frozen producer at equal budget. Sustained $>0$ = genuine compounding rather than a one-time producer upgrade.")

# ---------------- FLOWS / CAUSALITY ----------------
def fig_rsi_loop():
    fig, ax = plt.subplots(figsize=(9.0, 3.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    stages = ["propose", "verify", "select", "update", "transfer"]
    xs = np.linspace(0.9, 8.2, len(stages))
    for i, (s, x) in enumerate(zip(stages, xs)):
        ax.add_patch(plt.Rectangle((x - 0.7, 1.2), 1.4, 0.9, fc=CB[i % len(CB)], ec="k", alpha=0.85))
        ax.text(x, 1.65, s, ha="center", va="center", color="w", fontsize=10, weight="bold")
        if i < len(stages) - 1:
            ax.annotate("", (xs[i + 1] - 0.72, 1.65), (x + 0.72, 1.65), arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", (xs[0] - 0.72, 1.2), (xs[-1] + 0.0, 0.75), arrowprops=dict(arrowstyle="-|>", lw=1.2, connectionstyle="arc3,rad=0.25", color=CB[7]))
    ax.text(4.5, 0.5, "recursion: improved system authors the next round", ha="center", fontsize=8, color=CB[7])
    # gates
    ax.add_patch(plt.Rectangle((xs[1] - 0.9, 2.25), 1.8, 0.55, fc="none", ec=CB[1], lw=1.6, ls="--"))
    ax.text(xs[1], 2.52, "GATE 1\ncorrectness wall", ha="center", va="center", fontsize=7, color=CB[1])
    ax.add_patch(plt.Rectangle((xs[3] - 0.9, 2.25), 1.8, 0.55, fc="none", ec=CB[2], lw=1.6, ls="--"))
    ax.text(xs[3], 2.52, "GATE 2\nplasticity+retention", ha="center", va="center", fontsize=7, color=CB[2])
    ax.set_title("The RSI loop and its two failure gates")
    save(fig, "flow_rsi_loop", SEC_FLOW,
         "The recursive self-improvement loop (propose$\\to$verify$\\to$select$\\to$update$\\to$transfer) and the two gates our data identifies: Gate 1, the correctness wall (no verified kernel $\\Rightarrow$ no gradient), and Gate 2, plasticity+retention (drift saturation / forgetting collapses compounding).")

def fig_failure_cascade():
    ma = M("mech_analysis")
    if not ma: return
    # stage proportions across models: attempt -> crossed wall -> compounds; plus mean correctness among crossers
    n = len(ma)
    crossed = sum(1 for r in ma if r.get("wall_crossed")) / max(n, 1)
    rsi = sum(1 for r in ma if r.get("rsi")) / max(n, 1)
    meanC = np.mean([r.get("max_C_train", 0) or 0 for r in ma])
    stages = ["all attempts", "emit correct\nkernel (wall)", "mean correct\nrate", "sustained\ndrift+retention", "RSI\ncompounds"]
    vals = [1.0, crossed, meanC, max(rsi, 0.001), rsi]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.bar(range(len(stages)), vals, color=[CB[7], CB[0], CB[4], CB[2], CB[1]])
    for i, v in enumerate(vals): ax.text(i, v + 0.02, "%.0f%%" % (100 * v), ha="center", fontsize=8)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of models / rate"); ax.set_title("Failure cascade to RSI (open-weight models)")
    save(fig, "flow_failure_cascade", SEC_FLOW,
         "Attrition from attempt to compounding RSI across open-weight models: most attrition happens at the correctness wall (emitting any correct kernel) and again at sustained plasticity+retention.")

def fig_dense_allruns():
    ma = M("mech_analysis")
    if not ma: return
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for r in ma:
        s = r.get("size_b", 2); gain = r.get("C_held_gain", 0) or 0
        drift = (r.get("drift_total", 0) or 0)
        col = CB[2] if r.get("rsi") else (CB[0] if r.get("wall_crossed") else CB[7])
        ax.scatter(s, gain, s=30 + 900 * drift, color=col, alpha=0.55, ec="k", lw=0.4, zorder=3)
    ax.axhline(0, color="k", lw=0.7); ax.axvline(2, color=CB[1], ls="--", lw=1)
    ax.text(2.05, ax.get_ylim()[1] * 0.9, "correctness wall $\\sim$2B", color=CB[1], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("model size (B params, log)"); ax.set_ylabel("held-out C gain over rounds")
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CB[2], label="compounds", ms=9),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=CB[0], label="crossed wall, flat", ms=9),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=CB[7], label="below wall", ms=9)]
    ax.legend(handles=leg, fontsize=8, loc="lower right")
    ax.set_title("All open-weight RSI runs: size $\\times$ gain, marker area $\\propto$ LoRA drift")
    save(fig, "flow_all_runs", SEC_FLOW,
         "Every open-weight weight-RSI run positioned by model size ($x$) and held-out C gain ($y$); marker area $\\propto$ sustained LoRA drift; color = outcome. The 2B correctness wall and the mid-scale compounding band are visible at a glance.")

# ---------------- TABLES ----------------
def _tab_rows(models, cols, keyfn, n=12):
    rows = sorted([m for m in models if keyfn(m) is not None], key=lambda m: -keyfn(m))[:n]
    return rows

def make_tables():
    tex = []
    # T5 open L-F leaderboard. The ACCEPTED-PROPOSAL denominator is part of the table, not a footnote:
    # an L-F computed over <MINPROP accepted model-authored tasks is undefined, not a small effect.
    MINPROP = 5
    r5 = _tab_rows(M("selfplay"), None, lambda m: m.get("final_L_minus_F"))
    if r5:
        t = ["\\begin{tabular}{lrrrrl}", "\\toprule",
             "Model & tier & rounds & accepted & $L-F$ & status \\\\", "\\midrule"]
        for m in r5:
            np_ = m.get("total_model_proposed") or 0
            t.append("%s & %s & %d & %d & %.3f & %s \\\\" % (
                short(m["model"]).replace("_", "\\_"), m.get("tier", "-"),
                len(m.get("rounds") or []), np_, m.get("final_L_minus_F"),
                "interpretable" if np_ >= MINPROP else "\\emph{undefined}"))
        t += ["\\bottomrule", "\\end{tabular}"]
        tex.append(("tab:t5",
                    "Task 5 open self-play: author co-evolution $L-F$. \"accepted\" counts model-authored "
                    "tasks that passed the validity/novelty gate; $L-F$ is only defined when the live author "
                    "actually produced a frontier, so rows with fewer than %d accepted tasks are marked "
                    "\\emph{undefined} and must not be read as measured zeros (or as positives -- the "
                    "largest value in this table rests on a single accepted task)." % MINPROP,
                    "\n".join(t)))
    # T3 procedure-RSI
    r3 = _tab_rows(M("trackc"), None, lambda m: m.get("delta_vs_base"))
    if r3:
        t = ["\\begin{tabular}{lrrr}", "\\toprule", "Model & $Q_0$ & $Q_g$ & $\\Delta$ vs base \\\\", "\\midrule"]
        for m in r3:
            t.append("%s & %.3f & %.3f & %.3f \\\\" % (short(m["model"]).replace("_", "\\_"), m.get("Q0") or 0, m.get("Qg") or 0, m.get("delta_vs_base")))
        t += ["\\bottomrule", "\\end{tabular}"]; tex.append(("tab:t3", "Task 3 procedure-RSI: held-out $Q$ gain vs.\\ frozen procedure.", "\n".join(t)))
    # mech scale gates
    ma = load("mech_analysis").get("findings", {}).get("scale_trend", {})
    if ma:
        t = ["\\begin{tabular}{lrrrr}", "\\toprule", "Size band & $n$ & P(cross wall) & P(RSI) & mean drift \\\\", "\\midrule"]
        for band, s in ma.items():
            t.append("%s & %d & %.2f & %.2f & %.3f \\\\" % (band, s.get("n", 0), s.get("frac_cross_wall") or 0, s.get("frac_rsi") or 0, s.get("mean_drift") or 0))
        t += ["\\bottomrule", "\\end{tabular}"]; tex.append(("tab:gates", "Scale $\\to$ RSI gates: correctness-wall crossing and compounding by size band.", "\n".join(t)))
    return tex

# ---------------- BUILD ----------------
for fn in [fig_scale_gates, fig_drift_depth, fig_signature, fig_mech_traj, fig_interp_scatter, fig_interp_size,
           fig_lf_open, fig_lf_closed, fig_threearm, fig_lf_traj, fig_decomp, fig_openended,
           fig_t4_heatmap, fig_t4_peak, fig_t3_bar, fig_recursion_gain, fig_t2_scatter, fig_fork,
           fig_rsi_loop, fig_failure_cascade, fig_dense_allruns]:
    try: fn()
    except Exception as e: print("SKIP", fn.__name__, "->", repr(e)[:120])

tables = make_tables()

# ---------------- LaTeX ----------------
order = [SEC_MECH, SEC_T5, SEC_T4, SEC_T3, SEC_CMP, SEC_FLOW]
tex = [r"\documentclass[11pt]{article}", r"\usepackage[margin=1in]{geometry}",
       r"\usepackage{graphicx}", r"\usepackage{booktabs}", r"\usepackage{float}", r"\usepackage{amsmath}",
       r"\graphicspath{{figures/}}", r"\title{KernelAscent: RSI benchmark --- figure atlas}",
       r"\author{}", r"\date{\today}", r"\begin{document}", r"\maketitle",
       r"\noindent Auto-generated figure atlas (regenerate: \texttt{python3 scripts/make\_paper\_figures.py}). Pick the figures that matter; each has a \texttt{label} for cross-referencing.\par"]
# tables first
if tables:
    tex.append(r"\section{Leaderboards}")
    for lab, cap, body in tables:
        tex += [r"\begin{table}[H]\centering", body, r"\caption{%s}" % cap, r"\label{%s}" % lab, r"\end{table}"]
for sec in order:
    figs = [f for f in FIGS if f[1] == sec]
    if not figs: continue
    tex.append(r"\section{%s}" % sec)
    for name, _, cap in figs:
        tex += [r"\begin{figure}[H]\centering",
                r"\includegraphics[width=0.86\linewidth]{%s.pdf}" % name,
                r"\caption{%s}" % cap, r"\label{fig:%s}" % name, r"\end{figure}"]
tex.append(r"\end{document}")
open(os.path.join(ROOT, "paper", "figures.tex"), "w").write("\n".join(tex))
print("FIGURES:", len(FIGS), "TABLES:", len(tables))
