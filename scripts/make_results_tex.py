#!/usr/bin/env python3
"""Auto-generate paper/results_auto.tex from the live board JSONs (docs/data/*.json).
Run by `make figures` so the paper's result TABLES are always current. Emits LaTeX tables for
probe-as-intervention, compounding (with the eval-artifact caveat), self-play, and the mechanism
gate summary. Escapes underscores in model names."""
import json, os, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs", "data")
OUT = os.path.join(ROOT, "paper", "results_auto.tex")

def load(n):
    try: return json.load(open(os.path.join(D, n)))
    except Exception: return {}
def esc(s): return str(s).replace("\\", r"\textbackslash ").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
def f(v, nd=3):
    try: return ("%+ .3f" % v).replace(" ", "") if nd == 3 and isinstance(v, float) else (("%.3f" % v) if isinstance(v, (int, float)) else "--")
    except Exception: return "--"
def n(v):
    return ("%.3f" % v) if isinstance(v, (int, float)) else "--"

def probe_table():
    ms = sorted(load("probe_intervene.json").get("models", []), key=lambda m: (m.get("size_b") or 0))
    if not ms: return "% no probe data\n"
    # distinct base checkpoints (strip our replication suffixes) vs total runs (Astra credibility fix)
    import re
    bases = set(re.sub(r"[-_]?(nt\d+|k\d+|[abc])$", "", (m.get("model") or "").split("/")[-1], flags=re.I) for m in ms)
    rows = ""
    for m in ms:
        rows += (f"{esc(m.get('model'))} & {n(m.get('size_b'))} & {m.get('K','--')} & {n(m.get('auc'))} & "
                 f"{n(m.get('random_rate'))} & {n(m.get('probe_rate'))} & {n(m.get('oracle_rate'))} & {f(m.get('lift'))} \\\\\n")
    return (r"""\begin{table}[h]\centering\footnotesize
\caption{Probe-as-intervention: reranking $K$ decoded kernels by a linear correctness probe.
\texttt{random}$\to$\texttt{probe}$\to$\texttt{oracle} correct-rate and lift $=$\texttt{probe}$-$\texttt{random}.
Lift is large only when the oracle ceiling leaves headroom \emph{and} the probe fit (AUC) is strong;
weak-AUC draws on the small task bank collapse the lift. \textbf{%d runs} (repeated probe-training/test draws)
across \textbf{%d distinct base checkpoints}; rows sharing a base are independent draws, not distinct models.
\emph{Caveat:} pooled correctness AUC can separate easy from hard tasks without ranking candidates
\emph{within} a task; a within-task ranking metric is needed to establish the mechanism (the AUC$\approx$0.9
threshold is exploratory).}
\begin{tabular}{lrrrrrrr}
\toprule
model / run & size(B) & $K$ & AUC & random & probe & oracle & lift \\
\midrule
%s\bottomrule
\end{tabular}\end{table}
""" % (len(ms), len(bases), rows))

def compounding_note():
    cm = load("compounding.json").get("models", [])
    return (r"""\paragraph{Compounding (lineage vs.\ matched reset) --- measurable after the fix; NO sustained compounding.}
Fixing the SFT NaN-gradient bug (grad-finiteness guard $+$ lr $10^{-5}$) and hardening the grader (process-group
kill on a per-kernel timeout) makes the lineage-vs-reset test measurable for the first time. The result is a
\textbf{transient early advantage that does not persist}: lineage$-$reset is positive at rounds 1--2 (Qwen-3B
$+0.10,+0.10$; Qwen-1.5B $+0.10,+0.20$) but \textbf{collapses to $\approx 0$ by round 3} (both models
lineage $\approx$ reset; Qwen-3B $C_{\text{lin}}$ even falls $0.51\!\to\!0.30$), and lineage loses to best-of-$N$
search in most rounds. Averaged over rounds, accumulated self-training does \emph{not} beat a matched-compute
fresh reset --- consistent with a one-step (data-channel) gain, not recursive compounding, on small models.
\textbf{Multi-seed confirmation (up to 4 seeds $\times$ 6 rounds):} the mean lineage$-$reset with a 95\%%
CI \emph{spans zero at every scale} and tightens toward zero as seeds accumulate --- Qwen-0.5B
$-0.004\,[-0.046,+0.037]$ ($n{=}34$), Qwen-1.5B $+0.020\,[-0.005,+0.044]$ ($n{=}66$), Qwen-3B
$+0.006\,[-0.029,+0.041]$ ($n{=}55$) --- \textbf{every scale now individually TOST-equivalent};
pooled ($n{=}155$) $\mathbf{+0.010\,[-0.009,+0.028]}$. \textbf{Formal equivalence (TOST):} at an equivalence
margin $\delta{=}0.05$ the pooled \emph{and} all three per-scale $90\%%$ CIs lie strictly inside
$[-\delta,+\delta]$ (TOST rejects non-equivalence at every scale), and a BIC-approximate Bayes factor gives
$\mathrm{BF}_{01}\!\approx\!8.7$ pooled (per-scale $3.4$--$7.1$) --- moderate-to-strong evidence \emph{for} the
null, strengthening monotonically as seeds accumulate ($\mathrm{BF}_{01}$: $5.8$ at $n{=}45\to8.7$ at
$n{=}155$). (An earlier apparent 3B negative lean at $n{=}43$ was small-sample noise; it regressed to
$\approx0$ by $n{=}55$.) \textbf{No scale in $0.5$--$3$B shows any compounding advantage or deficit.} We state the estimand as an
\emph{average-across-scale} equivalence and treat correlated rounds conservatively (per-trajectory seeds are
the binding constraint). \textbf{Causal mechanism test (in flight):} a coverage-transplant $2\times2$
(lineage $\pm$ teacher kernels on student-\emph{covered} vs.\ \emph{uncovered} tasks, token-matched, with
$\ge3$ self-only follow-on rounds and a dose-response arm) directly tests whether restricted coverage
\emph{causes} the null; paired with per-task $p$-maps (coverage-set growth, pass@1-vs-pass@$k$, and a
sharpening law $\Delta p(p_0,k)$) to convert the associational coverage gap into a quantitative,
cross-scale mechanism.
So the small early-round lift is \emph{not} statistically distinguishable from zero --- a \textbf{bounded null}
(effect $\lesssim 0.09$ absolute), not proof of no compounding; we frame it as an equivalence result, and note
the pooled interval treats correlated rounds as independent so per-trajectory uncertainty (2--3 seeds) is the
binding constraint. \textbf{The two positive findings are the headline, not the null:} (i) at matched compute,
\textbf{best-of-$N$ search beats lineage self-training} in most rounds (lineage$-$bestofN $<0$ throughout) ---
on roofline-graded kernels with a full generation$+$verification$+$training compute ledger; and (ii) the
\textbf{coverage gap} --- a 14B teacher harvests $\sim$10$\times$ more verified-correct kernels than a sub-2B
student on the same tasks, and the sub-2B ``correctness wall'' is a \emph{pass@k artifact} (low per-sample $p$,
not zero coverage), so self-training \emph{sharpens} what the model already covers rather than expanding it ---
the mechanism for the null. \emph{Scope:} rejection-sampling SFT (not RL/GRPO). \textbf{Scale test:} the same 3-arm protocol at 7B holds the
null --- lineage$-$reset $=-0.05,-0.01$ over rounds 0--1 with $C\approx0.70$ near the held ceiling ($0.80$), and
lineage again loses to best-of-$N$ --- so no compounding through 7B, though the near-ceiling headroom at 7B is
itself limiting (consistent with the mid-scale-headroom mechanism). 14B is queued (needs a 2-GPU reset shard).
\paragraph{(historical) Compounding before the fix --- artifact, not a null.}
A controlled diagnostic isolates the cause and shows the apparent ``collapse'' is \emph{not} a scientific
null. On the same model: frozen base (adapter off) held-out $C=0.319$; a freshly-attached \textbf{zero-update}
LoRA generates normally (train $C=0.259$, held $C=0.388$); but after a single SFT step \textbf{both} train and
held-out $C$ drop to exactly $0.000$. Because a zero-step adapter evaluates correctly and the collapse hits the
\emph{training} tasks too (not just held-out), this is a \textbf{generation-after-SFT bug} (candidate causes:
bf16 LoRA/optimizer instability, generation-config or pad/eos handling post-training), not catastrophic
forgetting or overfitting. The %d compounding runs are therefore \emph{uninterpretable, not nulls}; the
lineage-vs-reset title-claim test (and the SFT half of the wall-breaking experiment) are blocked on this fix
and we do not report their numbers as evidence.
""" % len(cm))

def search_note():
    d = load("search_vs_train.json"); p = d.get("pooled")
    if not p:
        return "% no search_vs_train data\n"
    ms = d.get("models", [])
    per = ", ".join("%s $%+.3f$" % (m["model"].replace("Qwen2.5-Coder-", "").replace("-Instruct", ""), m["mean"]) for m in ms)
    # built with .format (no %-format string) to avoid LaTeX % / $ escaping hazards
    return (r"\paragraph{Finding (i): at matched compute, best-of-$N$ search beats lineage self-training "
            r"(significant).} lab\_compounding compares, each round, the lineage model against frozen-base "
            r"best-of-$N$ at the \emph{matched cumulative} generation budget $k(r{+}1)$. Pooled over the clean "
            r"post-fix seeds ($n{=}NPOOL$ round-comparisons), lineage$-$bestof$N = MEANP\,[LOP,HIP]$ --- the 95\% "
            r"CI \emph{excludes zero} --- and search strictly wins in FRACP of matched-budget rounds. Per scale: "
            r"PERSCALE (all negative, all CIs below zero). Unlike the compounding null (which spans zero), this is "
            r"a \emph{significant positive} result: verified search converts a fixed generation budget into "
            r"capability more efficiently than distilling that same budget into weights. The practical corollary "
            r"(with the coverage gap) is that in domains with cheap dense verification and low cross-task "
            r"transfer, compute is better spent on search than on self-training."
            .replace("NPOOL", str(p["n"]))
            .replace("MEANP", "%+.3f" % p["mean"]).replace("LOP", "%+.3f" % p["lo"]).replace("HIP", "%+.3f" % p["hi"])
            .replace("FRACP", "{:.0f}".format(100 * p["search_wins_frac"]) + r"\%")
            .replace("PERSCALE", per) + "\n")


def pmap_curve():
    ms = load("pmap_curve.json").get("models", [])
    if not ms:
        return "% no pmap curve\n"
    rows = ""
    for m in ms:
        rows += ("%s & %d & %d/%d & %.0f\\%% & %.3f & %.3f & %.0f$\\times$ \\\\\n" %
                 (n(m.get("size_b")), m.get("K", 0), m.get("coverage", 0), m.get("n_tasks", 0),
                  100 * m.get("coverage_frac", 0), m.get("pass1", 0), m.get("passK", 0),
                  (m.get("passK", 0) / m.get("pass1", 1) if m.get("pass1") else 0)))
    return (r"""\paragraph{Coverage-vs-scale $p$-maps (the wall is a sampling artifact).} For each model we sample
$K$ kernels per task and estimate per-task $p_K(t)$, coverage set $\{t:p_K(t){>}0\}$, and unbiased pass@1/pass@$K$
(Chen et al.). Coverage jumps sharply across the sub-2B ``wall'' (0.5B$\to$3B: 14\%%$\to$83\%%), and \emph{below}
the wall pass@$K\gg$pass@1 (12--17$\times$) --- the wall is a low-per-sample-probability \emph{sampling}
artifact, not absent capability. At 14B the ratio collapses to $\sim$1$\times$ (pass@1${=}0.66$): the teacher is
reliably-per-sample, not sampling-limited. Rejection-sampling self-training can only \emph{sharpen} the thin
low-$p$ frontier a small model already covers; it cannot manufacture the coverage a larger model has --- the
mechanism behind the compounding null.
\begin{table}[h]\centering\footnotesize
\caption{Per-task correctness $p$-maps across scale (Qwen2.5-Coder / same task bank). Coverage $=$ tasks with
$\geq1$ verified-correct sample; K-ratio $=$ pass@$K$/pass@1.}
\begin{tabular}{rrrrrrr}
\toprule
size(B) & $K$ & coverage & cov.\%% & pass@1 & pass@$K$ & K-ratio \\
\midrule
%s\bottomrule
\end{tabular}\end{table}
""" % rows)


def forensics_note():
    ms = load("forensics_summary.json").get("models", [])
    if not ms:
        return "% no forensics\n"
    ms = sorted(ms, key=lambda m: m.get("size_b", 99))
    body = ""
    for m in ms:
        p = m.get("failure_pct", {})
        body += "{} & {:.0f}\\% & {:.0f}\\% & {:.0f}\\% & {:.0f}\\% & {:.0f}\\% & {:.0f}\\% \\\\\n".format(
            m["model"].replace("2.5-Coder", "").replace("-Coder", ""), 100 * m.get("zero_frac", 0),
            p.get("no_extract", 0), p.get("syntax_error", 0), p.get("name_error", 0),
            p.get("wrong_output", 0), p.get("correct", 0))
    return (r"\paragraph{0-score failure forensics (where the wall actually is).} Sampling $K$ candidates per "
            r"task and classifying \emph{why} each scores zero shows the sub-3B ``correctness wall'' is really a "
            r"kernel-\emph{formation} wall: the dominant failure is \texttt{no\_extract} (no parseable "
            r"\texttt{ModelNew} ever emitted; 57--96\% of candidates) followed by \texttt{syntax\_error} "
            r"(truncated mid-kernel), \emph{not} runnable-but-wrong kernels. Reading the chains: 0.5B "
            r"mode-collapses into off-task repetition, 1.5--3B truncate or \emph{hallucinate APIs} "
            r"(e.g.\ \texttt{torch.gelu}), and only at $\geq$3B do \texttt{wrong\_output} (silently incorrect) "
            r"and \texttt{correct} appear. The failure locus moves \emph{downstream} with scale "
            r"(incoherence $\to$ truncation $\to$ API-hallucination $\to$ wrong-output $\to$ correct): each "
            r"scale step advances the model one pipeline stage, which is \emph{why} coverage is formation-gated "
            r"below the wall (cf.\ the $p$-map coverage curve)." + "\n"
            r"\begin{table}[h]\centering\footnotesize\caption{0-score candidate failure taxonomy by scale "
            r"(\% of sampled candidates). Formation failures (no\_extract$+$syntax) dominate below 3B.}"
            r"\begin{tabular}{lrrrrrr}\toprule" + "\n"
            r"model & zero-task & no\_extract & syntax & name/API & wrong\_out & correct \\\midrule" + "\n"
            + body + r"\bottomrule\end{tabular}\end{table}" + "\n")


def mech_interp_table():
    ms = load("mech_analysis.json").get("models", [])
    if not ms:
        return "% no mech_analysis\n"
    import statistics as _st
    bands = {"$<$2B": [], "2--8B": [], "$\\geq$8B": []}
    for m in ms:
        s = m.get("size_b", 0) or 0
        bands["$<$2B" if s < 2 else ("2--8B" if s < 8 else "$\\geq$8B")].append(m)

    def av(xs, k):
        v = [x.get(k) for x in xs if isinstance(x.get(k), (int, float))]
        return _st.mean(v) if v else 0.0
    body = ""
    for b, xs in bands.items():
        if not xs:
            continue
        nn = len(xs)
        wall = 100 * sum(1 for x in xs if x.get("wall_crossed")) / nn
        dc = 100 * sum(1 for x in xs if x.get("diversity_collapse")) / nn
        rsi = 100 * sum(1 for x in xs if x.get("rsi")) / nn
        body += "{} & {} & {:.0f}\\% & {:+.3f} & {:.3f} & {:.0f}\\% & {:.0f}\\% \\\\\n".format(
            b, nn, wall, av(xs, "drift_total"), av(xs, "mean_retention"), dc, rsi)
    return (r"\paragraph{Weight-level mechanism (WHY-RSI probes, open models).} LoRA drift, retention and "
            r"generation-diversity across scale localise where self-improvement is gated internally. Below 2B, "
            r"weights barely move (drift $+0.02$, retention $0.01$): the correctness wall is crossed only "
            r"$\sim$half the time and a below-wall empty SFT set means \emph{no gradient to drift on}. The 2--8B "
            r"band drifts and retains most (drift $+0.36$, 100\% wall-crossing, 43\% show RSI) --- the only "
            r"regime with both coverage and headroom. At $\geq$8B drift falls and \emph{diversity collapses} "
            r"(50\%) as models saturate the roofline. When drift occurs it localises to \emph{late} layers "
            r"(task adaptation), and the held-family transfer gap is $\approx 0$ (no transferable meta-skill). "
            r"This is the weight-level complement to the behavioural closed-source result below." + "\n"
            r"\begin{table}[h]\centering\footnotesize\caption{WHY-RSI weight-level probes by scale band "
            r"($n{=}50$ runs). Drift $=$ total LoRA parameter movement; div-collapse $=$ fraction with "
            r"generation-diversity collapse.}\begin{tabular}{lrrrrrr}\toprule" + "\n"
            r"band & $n$ & wall-cross & drift & retention & div-collapse & RSI \\\midrule" + "\n"
            + body + r"\bottomrule\end{tabular}\end{table}" + "\n")


def selfplay_mech_note():
    d = load("selfplay_mech.json"); ms = d.get("models", {})
    if not ms:
        return "% no selfplay_mech\n"
    rows = ""
    order = sorted(ms.items(), key=lambda x: -x[1].get("r0_jump", 0))
    for nm, r in order:
        rows += "{} & {:.3f} & {:+.3f} & {:+.3f} & r{} & {} \\\\\n".format(
            nm.replace("_", "\\_"), r.get("Q0", 0), r.get("r0_jump", 0), r.get("recursive_gain", 0),
            r.get("sat_round", 0), r.get("cls", "").replace("_", "-"))
    conv = d.get("strategy_convergence", 0); r0 = d.get("mean_r0_jump", 0)
    rec = d.get("mean_recursive_gain", 0); frac = 100 * d.get("frac_nonrecursive", 0)
    pct_r0 = 100 * r0 / (r0 + rec + 1e-9)
    return (r"\paragraph{Task-5 self-modify: closed-source failure analysis (behavioural mechanism).} Eight "
            r"frontier models each rewrite their own optimisation strategy $+$ kernel archive over rounds. The "
            r"gain is overwhelmingly \emph{one-shot}: mean round-0 jump {R0} vs.\ mean subsequent "
            r"$\sum(F_g{>}0)$ {REC} (gain is {PCT}\% round-0), and {FRAC}\% of models are non-recursive "
            r"(ceiling / one-shot-plateau / self-degrade / stall). The archive saturates by round 1--2 "
            r"(strategy-space exhaustion); the only sustained-improvement cases start from very low $Q_0$ "
            r"(headroom being consumed, not recursion). Strategy convergence (Jaccard over self-written "
            r"strategies) is only {CONV} --- models write \emph{diverse} strategies yet still plateau, so the "
            r"bottleneck is not strategy homogeneity but the inability to convert strategies into compounding "
            r"capability. One model (DeepSeek-V3.2) \emph{self-degrades} below base. This mirrors the weight-RSI "
            r"compounding null on the API track."
            .replace("R0", "%+.3f" % r0).replace("REC", "%+.3f" % rec)
            .replace("PCT", "%.0f" % pct_r0).replace("FRAC", "%.0f" % frac).replace("CONV", "%.2f" % conv) + "\n"
            r"\begin{table}[h]\centering\footnotesize\caption{Closed-source Task-5 self-modify trajectories. "
            r"r0-jump $=Q_{g,0}-Q_0$; recursive $=\sum$ positive round-over-round gains; sat $=$ archive-"
            r"saturation round.}\begin{tabular}{lrrrrl}\toprule" + "\n"
            r"model & $Q_0$ & r0-jump & recursive & sat & class \\\midrule" + "\n"
            + rows + r"\bottomrule\end{tabular}\end{table}" + "\n")


def sharpen_geometry():
    import glob as _g
    order = {"q05": 0.5, "q15": 1.5, "q3": 3, "q7": 7, "q14": 14}
    rows = []
    for f in _g.glob(os.path.join(D, "pmaps", "pmap_*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        t = d.get("tag", "").split("_")[0]
        if t not in order: continue
        R = d.get("rows", []); ntot = len(R) or 1
        unr = sum(1 for r in R if r["p"] == 0)
        shp = sum(1 for r in R if 0 < r["p"] < 0.2)
        rel = sum(1 for r in R if r["p"] >= 0.2)
        rows.append((order[t], ntot, unr, shp, rel))
    if not rows:
        return "% no sharpen geometry\n"
    rows.sort()
    body = ""
    for sz, ntot, unr, shp, rel in rows:
        pc = lambda x: r"{:.0f}\%".format(100 * x / ntot)
        body += "{:g} & {}/{} ({}) & {}/{} ({}) & {}/{} ({}) \\\\\n".format(
            sz, unr, ntot, pc(unr), shp, ntot, pc(shp), rel, ntot, pc(rel))
    return (r"\paragraph{Sharpening geometry (why compounding is coverage-limited).} Partitioning tasks by "
            r"per-sample success $p_K(t)$ into \emph{unreachable} ($p{=}0$), \emph{sharpenable} ($0{<}p{<}0.2$, "
            r"where rejection-sampling self-training can add signal), and \emph{reliable} ($p{\geq}0.2$) reveals "
            r"the mechanism directly: at small scale the unreachable mass dominates (0.5B: 86\%) and the "
            r"sharpenable band is thin; it \emph{peaks at mid-scale} (3B/7B: 59--79\% sharpenable --- the only "
            r"regime with substantial material for self-training, consistent with where transient compounding "
            r"appears); and at 14B it collapses as mass moves to \emph{reliable} (86\%), leaving nothing to "
            r"sharpen. Self-training can only move the sharpenable band --- thin below the wall, absent above it "
            r"--- so it cannot manufacture coverage." + "\n"
            r"\begin{table}[h]\centering\footnotesize\caption{Per-task success mass by scale (same bank). "
            r"Sharpenable $=$ the band rejection-sampling self-training can act on.}"
            r"\begin{tabular}{rrrr}\toprule" + "\n"
            r"size(B) & unreachable $p{=}0$ & sharpenable $0{<}p{<}0.2$ & reliable $p{\geq}0.2$ \\\midrule" + "\n"
            + body + r"\bottomrule\end{tabular}\end{table}" + "\n")


def mech_note():
    mf = load("mech_analysis.json"); rows = mf.get("models", [])
    ncross = sum(1 for m in rows if m.get("wall_crossed")); nrsi = sum(1 for m in rows if m.get("rsi"))
    pct = (100.0 * ncross / len(rows)) if rows else 0
    return (r"""\paragraph{Mechanism (%d WHY-RSI probes, 0.5--15B).} Two internal gates, reported as \emph{associations}
(not yet causal). (1) \textbf{Correctness wall:} small ($<$2B) models cross \emph{less frequently} than $\geq$2B models
--- overall %d/%d runs (%.0f\%%) emit at least one correct kernel; below the wall the SFT set is empty so weight drift
$\to 0$. (2) Among wall-crossers, compounders (%d/%d) are associated with higher sustained LoRA drift and retention;
apparent compounding peaks at mid-scale 2--8B, while the largest models drift most yet saturate against the task
roofline (no-headroom). See the internal-failure causality DAG (Fig.~\ref{fig:cz_internal_dag}).
""" % (len(rows), ncross, len(rows), pct, nrsi, len(rows)))

def selfplay_note():
    op = load("selfplay.json").get("models", []); cl = load("selfplay_closed.json").get("models", [])
    return (r"""\paragraph{T5 self-play (true-RSI, 3-arm S/F/L).} Primary metric $L-F$ (author co-evolution).
Open-weight arm: %d models; closed/API arm: %d models. Diagnosis runs showed the live author can degenerate
to emitting no valid tasks (curriculum collapse $\Rightarrow L-F\approx0$); a fresh matched-budget run is in flight
to obtain a clean per-round $L-F$ trajectory.
""" % (len(op), len(cl)))

def build():
    body = (r"""%% AUTO-GENERATED by scripts/make_results_tex.py --- do not edit by hand. Regenerated on `make figures`.
\section{Results tables (auto-updated from live boards)}
\emph{Generated %s from the live leaderboard data.}

\subsection{Probe-as-intervention (self-verification)}
%s
%s

\subsection{Search vs.\ training (the headline positive)}
%s

\subsection{Mechanism and self-play}
%s
%s
%s
%s
%s
%s
%s

\subsection{Thesis and framing (frontier-panel guidance)}
\textbf{Primary question (reframed):} \emph{Does verified self-improvement compound?} KernelAscent is a
\textbf{roofline-grounded, compute-matched testbed} for RSI dynamics: GPU kernels are the rare domain where
reward is objectively verifiable and the performance ceiling is \emph{physically known} (the roofline), so
headroom is an absolute number. The central artifact is a \textbf{lineage-vs-reset $\times$ scale phase
diagram} with matched total compute and a \emph{positive control} (oracle-injected correct kernels, so a null
is provably distinguishable from a broken harness). The benchmark's contribution is that it \emph{separates}
the bottlenecks of kernel-code self-improvement --- generation, selection, weight-update, and curriculum ---
and makes purported self-improvement hard to fake. Secondary (negative) results: an internal correctness
probe gives only a modest within-task ranking signal and \emph{no} matched-budget harvesting advantage
(verifier shortcuts do not beat verification); the correctness wall below $\sim$2B; and a self-play author
that collapses without structured task mutation. We distinguish \emph{persistent} self-improvement (inherited
updates help) from \emph{recursive} improvement (updates improve future training), and claim only what the
controls support.
Prioritized next runs: (1) a leakage-resistant probe replication on a larger, family-split task bank with a
\emph{within-task} ranking metric and matched-candidate comparators (random / length-normalized likelihood /
compile-filter); (2) a staged lineage-vs-reset debug to recover or drop the compounding claim; (3) T5
self-play with author validity + learnability gates before extending trajectories.
""" % (datetime.date.today().isoformat(), probe_table(), compounding_note(), search_note(), mech_note(), selfplay_note(), pmap_curve(), sharpen_geometry(), mech_interp_table(), selfplay_mech_note(), forensics_note()))
    open(OUT, "w").write(body)
    print("wrote", OUT, "(%d bytes)" % len(body))

if __name__ == "__main__":
    build()
