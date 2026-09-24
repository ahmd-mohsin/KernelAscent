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
    d = load("compounding_tost.json")
    if not d or not d.get("pooled"):
        return "% no compounding_tost.json -- run scripts/equivalence_tost.py --out docs/data/compounding_tost.json\n"
    P = d["pooled"]
    tr, cr, rd, icc = P["trajectory"], P["crve"], P["round"], P.get("icc", {})
    delta = d.get("delta", 0.05)

    def ci(e, key="ci95"):
        return "$%+.3f\\,[%+.3f,%+.3f]$" % (e["mean"], e[key][0], e[key][1])

    # per-scale rows, ordered by size, trajectory-level (PRIMARY)
    order = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0}
    def sz(m):
        for k, v in order.items():
            if k in m:
                return v
        return 99.0
    rows, powered, underp = "", [], []
    for m in sorted(d.get("models", {}), key=sz):
        a = d["models"][m]
        t = a.get("trajectory")
        short = m.replace("Qwen2.5-Coder-", "Qwen-").replace("-Instruct", "")
        if not t:
            underp.append(short)
            continue
        verdict = "EQUIV" if t["equivalent"] else "not equiv."
        if t["n"] < 5:
            verdict += " (underpowered)"
            underp.append(short)
        else:
            powered.append(short)
        rows += "%s & %d & %d & %s & %s & %.1f \\\\\n" % (
            esc(short), t["n"], t["n_rounds"], ci(t), verdict,
            (t["bf01"] if t["bf01"] is not None else float("nan")))
    rows += r"\midrule" + "\n"
    rows += "\\textbf{Pooled} & \\textbf{%d} & \\textbf{%d} & \\textbf{%s} & \\textbf{%s} & \\textbf{%.1f} \\\\\n" % (
        tr["n"], tr["n_rounds"], ci(tr), "EQUIV" if tr["equivalent"] else "not equiv.", tr["bf01"])

    body = (
        r"\paragraph{Compounding (lineage vs.\ matched reset): a bounded null at the trajectory level.}" "\n"
        r"\textbf{Estimand and unit of replication.} The independent replicate is a \emph{trajectory} (one seed, "
        r"one base checkpoint, one held split, one accumulated adapter), \emph{not} a round: rounds inside a run "
        r"share all of those and are serially dependent. We therefore report the trajectory-level mean as the "
        r"primary estimand, a cluster-robust (CRVE) estimate on the round-level data clustered on trajectory as a "
        r"secondary check, and the naive round-level pooling only so the correction is auditable. Measured "
        r"intra-trajectory correlation is ICC${=}$ICCV (design effect DEFF, so the naive $n{=}NR$ rounds carry the "
        r"evidence of roughly NEFF independent observations)." "\n"
        r"\textbf{Result.} Pooled over NTRAJ trajectories (NR round-comparisons), lineage$-$reset ${=}$ POOLT, "
        r"TOST-equivalent at $\delta{=}DELTA$ with $\mathrm{BF}_{01}{=}BFT$ --- \emph{moderate} evidence for the "
        r"null. The cluster-robust estimate agrees (POOLC, $\mathrm{BF}_{01}{=}BFC$). For contrast, the naive "
        r"round-level pooling would report POOLR with $\mathrm{BF}_{01}{=}BFR$; that number is an artifact of "
        r"treating correlated rounds as independent and we do not claim it. The conclusion is unchanged in sign "
        r"and verdict, but the strength of evidence is \emph{moderate}, not strong." "\n"
        r"\textbf{Robustness.} Restricting to trajectories with $\geq3$ rounds (36 runs) and to full-length "
        r"$\geq6$-round trajectories (29 runs) leaves the pooled verdict EQUIV with "
        r"$\mathrm{BF}_{01}\!\approx\!5.9$ and $4.9$ respectively, so the null is not an artifact of short or "
        r"interrupted runs." "\n"
        r"\textbf{Per round}, lineage$-$reset shows a \emph{transient} early advantage (rounds 1--2) that decays "
        r"to $\approx0$ by round 3 --- consistent with a one-step data-channel gain rather than recursive "
        r"compounding." "\n"
        r"\emph{Scope:} rejection-sampling SFT (not full RL/GRPO); Qwen2.5-Coder family; the bank and budgets "
        r"described above. A denser-reward variant (speedup-weighted rejection sampling, top-quartile-by-roofline "
        r"kernels only) shows the same null on 2 seeds, so the result is not an artifact of the weakest "
        r"selection rule --- but at that $n$ it is an indication, not a test." "\n"
        r"\textbf{What this null does and does not license.} It is a \emph{bounded} null: the compounding "
        r"advantage is contained within $\pm DELTA$ at the tested scales. It is \textbf{not} yet evidence that "
        r"the harness \emph{could} have registered compounding. The randomized coverage-transplant $2\times2$ we "
        r"originally designed proved computationally infeasible on the crash-isolated grader (${\sim}12$\,s per "
        r"kernel-grade put round~0 alone beyond 2.5\,h), so the coverage \emph{mechanism} is established "
        r"indirectly from the $p$-maps, the sharpening geometry and the 0-score forensics. A tractable "
        r"coverage-injection positive control has since been run on H100 and returned flat in \emph{both} arms "
        r"--- but on that hardware the primary metric is saturated (76\% of all scores exactly $0.50$; see the "
        r"instrument-validity section), so it licenses no conclusion in either direction. \textbf{A control that "
        r"is flat because the instrument cannot move is not a control.} Re-running it under a metric with "
        r"demonstrated range on that hardware remains the decisive outstanding experiment." "\n"
        r"\begin{table}[h]\centering\footnotesize\caption{Lineage vs.\ matched reset, \textbf{trajectory-level} "
        r"(primary estimand). $n$ = independent trajectories; rounds = round-comparisons those trajectories "
        r"contain. CI is a $t$-interval on the trajectory means; $\mathrm{BF}_{01}$ is evaluated at the "
        r"trajectory $n$. Scales with $n{<}5$ trajectories are marked underpowered and are not interpreted "
        r"individually.}\begin{tabular}{lrrlcr}\toprule" "\n"
        r"scale & $n$ traj. & rounds & lineage$-$reset (95\% CI) & TOST & $\mathrm{BF}_{01}$ \\\midrule" "\n"
        + rows + r"\bottomrule\end{tabular}\end{table}" "\n"
    )
    repl = {
        "ICCV": "%.3f" % (icc.get("icc", 0)), "DEFF": "%.2f" % (icc.get("design_effect", 1)),
        "NEFF": "%.0f" % (icc.get("n_eff", 0)), "NTRAJ": str(tr["n"]), "NR": str(rd["n"]),
        "POOLT": ci(tr), "POOLC": ci(cr), "POOLR": ci(rd),
        "BFT": "%.1f" % tr["bf01"], "BFC": "%.1f" % cr["bf01"], "BFR": "%.1f" % rd["bf01"],
        "DELTA": "%.2f" % delta,
    }
    for k, v in repl.items():
        body = body.replace(k, v)

    body += (
        r"\paragraph{(methods) Pre-fix runs are excluded as uninterpretable, not counted as nulls.}" "\n"
        r"An early diagnostic isolated a generation-after-SFT failure: on the same model, the frozen base "
        r"(adapter off) scored held-out $C{=}0.319$ and a freshly-attached \textbf{zero-update} LoRA generated "
        r"normally ($C_{\text{train}}{=}0.259$, $C_{\text{held}}{=}0.388$), but after a single SFT step "
        r"\emph{both} train and held-out $C$ fell to exactly $0.000$. A zero-step adapter evaluating correctly, "
        r"plus the collapse hitting the \emph{training} tasks too, identifies this as a harness fault (SFT "
        r"NaN-gradient path) rather than catastrophic forgetting. Runs predating the fix (grad-finiteness guard "
        r"$+$ lr $10^{-5}$, plus a process-group kill on per-kernel timeout) are therefore excluded by a "
        r"canonical-seed allowlist and contribute to no statistic in this paper; they are uninterpretable, not "
        r"nulls." "\n"
    )
    return body

def search_note():
    d = load("search_vs_train.json"); p = d.get("pooled")
    if not p:
        return "% no search_vs_train data\n"
    ms = [m for m in d.get("models", []) if not m.get("underpowered") and m.get("mean") is not None]
    under = [m for m in d.get("models", []) if m.get("underpowered")]
    per = ", ".join("%s $%+.3f\\,[%+.3f,%+.3f]$" % (
        m["model"].replace("Qwen2.5-Coder-", "").replace("-Instruct", ""), m["mean"], m["lo"], m["hi"])
        for m in ms)
    tail = ""
    if under:
        tail = (r" Scales with $<5$ trajectories (%s) are reported but not interpreted individually." %
                ", ".join(esc(m["model"].replace("Qwen2.5-Coder-", "").replace("-Instruct", "")) for m in under))
    return (r"\paragraph{Finding (i): at matched compute, best-of-$N$ search beats lineage self-training.} "
            r"lab\_compounding compares, each round, the lineage model against frozen-base best-of-$N$ at the "
            r"\emph{matched cumulative} generation budget $k(r{+}1)$, so the contrast is compute-matched by "
            r"construction. Analysed at the \textbf{trajectory} level (the independent replicate; see the "
            r"compounding estimand above), pooled over NTRAJ trajectories comprising NR round-comparisons, "
            r"lineage$-$bestof$N = MEANP\,[LOP,HIP]$ --- the 95\% CI \emph{excludes zero} --- and verified "
            r"search beats lineage in FRACRUN of \emph{runs} (FRACRND of rounds). The cluster-robust estimator "
            r"(which weights rounds rather than runs) gives CRMEAN\\,[CRLO,CRHI] --- a larger effect of the same "
            r"sign, its interval also excluding zero. Per scale: PERSCALE.TAIL Unlike the compounding result (a bounded null), "
            r"this is a \emph{significant positive}: verified search converts a fixed generation budget into "
            r"capability more efficiently than distilling that same budget into weights. The practical corollary "
            r"(with the coverage gap) is that in domains with cheap dense verification and low cross-task "
            r"transfer, compute is better spent on search than on self-training."
            .replace("NTRAJ", str(p["n_trajectories"])).replace("NR", str(p["n_rounds"]))
            .replace("MEANP", "%+.3f" % p["mean"]).replace("LOP", "%+.3f" % p["lo"]).replace("HIP", "%+.3f" % p["hi"])
            .replace("CRMEAN", "$%+.3f$" % p.get("crve_mean", float("nan")))
            .replace("CRLO", "%+.3f" % p["crve_lo"]).replace("CRHI", "%+.3f" % p["crve_hi"])
            .replace("FRACRUN", "{:.0f}".format(100 * p["traj_wins_frac"]) + r"\%")
            .replace("FRACRND", "{:.0f}".format(100 * p["round_wins_frac"]) + r"\%")
            .replace("PERSCALE", per).replace("TAIL", tail) + "\n")


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
    # Band cutoffs are fixed here and reused by every mechanism table/figure so the paper cannot
    # carry two different band definitions (an earlier draft used <8B in one table and <9B in another).
    BANDS = [("$<$2B", 0, 2), ("2--8B", 2, 8), ("$\\geq$8B", 8, 1e9)]
    bands = {lab: [m for m in ms if lo <= (m.get("size_b") or 0) < hi] for lab, lo, hi in BANDS}

    def av(xs, k):
        v = [x.get(k) for x in xs if isinstance(x.get(k), (int, float))]
        return _st.mean(v) if v else 0.0

    def frac(xs, k):
        return (sum(1 for x in xs if x.get(k)) / len(xs)) if xs else 0.0

    body, stat = "", {}
    for lab, xs in bands.items():
        if not xs:
            continue
        stat[lab] = {"n": len(xs), "wall": frac(xs, "wall_crossed"), "drift": av(xs, "drift_total"),
                     "ret": av(xs, "mean_retention"), "dc": frac(xs, "diversity_collapse"),
                     "rsi": frac(xs, "rsi")}
        b = stat[lab]
        body += "{} & {} & {:.0f}\\% & {:+.3f} & {:.3f} & {:.0f}\\% & {:.0f}\\% \\\\\n".format(
            lab, b["n"], 100 * b["wall"], b["drift"], b["ret"], 100 * b["dc"], 100 * b["rsi"])
    lo_, mid, hi_ = stat.get("$<$2B", {}), stat.get("2--8B", {}), stat.get("$\\geq$8B", {})
    # diversity: state the DIRECTION the data actually shows rather than a remembered claim
    crossers = [m for m in ms if m.get("wall_crossed")]
    div_rsi = _st.mean([m["mean_diversity"] for m in crossers
                        if m.get("rsi") and isinstance(m.get("mean_diversity"), (int, float))] or [0])
    div_flat = _st.mean([m["mean_diversity"] for m in crossers
                         if not m.get("rsi") and isinstance(m.get("mean_diversity"), (int, float))] or [0])
    prose = (
        r"\paragraph{Weight-level mechanism (WHY-RSI probes, open models).} LoRA drift, retention and "
        r"generation diversity across scale localise where self-improvement is gated internally, reported as "
        r"\emph{associations} rather than established causal gates. Below 2B, weights barely move "
        r"(drift $DLO$, retention $RLO$): the correctness wall is crossed only WLO\% of the time, and below the "
        r"wall an empty SFT set means \emph{no gradient to drift on}. The 2--8B band crosses almost always "
        r"(WMID\%) and shows the highest rate of apparent compounding (RMID\% of runs) --- the only regime with "
        r"both coverage and headroom. At $\geq$8B drift is \emph{largest} ($DHI$) yet the compounding rate falls "
        r"to RHI\% and diversity collapse reaches DCHI\%: motion without progress near the task roofline. When "
        r"drift occurs it localises to \emph{late} layers (task adaptation) and the held-family transfer gap is "
        r"$\approx0$ (no transferable meta-skill). \textbf{Note on diversity:} among wall-crossers, compounders "
        r"have \emph{lower} mean generation diversity than flat runs (DIVR vs.\ DIVF), so on this bank "
        r"``diversity collapse causes the null'' is \emph{not} supported --- compounding here looks like "
        r"productive convergence onto a good basin. Diversity contraction remains a plausible cross-round "
        r"option-value lock (Discussion), but it is an association whose clean test is a diversity-preserving "
        r"intervention we have not run." "\n"
        r"\begin{table}[h]\centering\footnotesize\caption{WHY-RSI weight-level probes by scale band "
        r"($n{=}NTOT$ runs; bands $<$2B / 2--8B / $\geq$8B used consistently throughout). Drift $=$ total LoRA "
        r"parameter movement; div-collapse $=$ fraction of runs with generation-diversity collapse; RSI $=$ "
        r"fraction meeting the compounding criterion.}\begin{tabular}{lrrrrrr}\toprule" "\n"
        r"band & $n$ & wall-cross & drift & retention & div-collapse & RSI \\\midrule" "\n"
        + body + r"\bottomrule\end{tabular}\end{table}" "\n")
    for k, v in (("NTOT", str(len(ms))),
                 ("DLO", "%+.3f" % lo_.get("drift", 0)), ("RLO", "%.3f" % lo_.get("ret", 0)),
                 ("WLO", "%.0f" % (100 * lo_.get("wall", 0))), ("WMID", "%.0f" % (100 * mid.get("wall", 0))),
                 ("RMID", "%.0f" % (100 * mid.get("rsi", 0))), ("DHI", "%+.3f" % hi_.get("drift", 0)),
                 ("RHI", "%.0f" % (100 * hi_.get("rsi", 0))), ("DCHI", "%.0f" % (100 * hi_.get("dc", 0))),
                 ("DIVR", "%.2f" % div_rsi), ("DIVF", "%.2f" % div_flat)):
        prose = prose.replace(k, v)
    return prose


def selfplay_mech_note():
    """T3 self-modify. Rows with <MINR completed rounds cannot separate 'ceiling' from 'run stopped
    early' and are segregated into an explicitly non-interpreted block (P0.3)."""
    d = load("selfplay_mech.json"); ms = d.get("models", {})
    if not ms:
        return "% no selfplay_mech\n"
    MINR = 3
    ok = {k: v for k, v in ms.items() if (v.get("rounds") or 0) >= MINR}
    under = {k: v for k, v in ms.items() if (v.get("rounds") or 0) < MINR}

    def rowify(items, mark=""):
        out = ""
        for nm, r in sorted(items.items(), key=lambda x: -x[1].get("r0_jump", 0)):
            out += "{}{} & {} & {:.3f} & {:+.3f} & {:+.3f} & r{} & {} \\\\\n".format(
                nm.replace("_", "\\_"), mark, r.get("rounds", "--"), r.get("Q0", 0),
                r.get("r0_jump", 0), r.get("recursive_gain", 0), r.get("sat_round", 0),
                r.get("cls", "").replace("_", "-"))
        return out

    rows = rowify(ok)
    if under:
        rows += r"\midrule \multicolumn{7}{l}{\emph{$<$%d completed rounds --- reported, not interpreted}} \\" % MINR
        rows += "\n" + rowify(under, r"$^\dagger$")
    import statistics as _st
    r0 = _st.mean([v.get("r0_jump", 0) for v in ok.values()]) if ok else 0
    rec = _st.mean([v.get("recursive_gain", 0) for v in ok.values()]) if ok else 0
    nonrec = sum(1 for v in ok.values() if v.get("cls") in ("CEILING", "ONE-SHOT-PLATEAU", "STALL", "SELF-DEGRADE"))
    frac = 100.0 * nonrec / len(ok) if ok else 0
    pct_r0 = 100 * r0 / (r0 + rec + 1e-9)
    conv = d.get("strategy_convergence", 0)
    under_names = ", ".join(esc(k) + " (%d round%s)" % (v.get("rounds", 0), "" if v.get("rounds") == 1 else "s")
                            for k, v in sorted(under.items()))
    return (r"\paragraph{Task-3 procedure self-modify: closed-source failure analysis (behavioural mechanism).} "
            r"Frontier models each rewrite their own optimisation strategy library $+$ verified kernel archive "
            r"over rounds. Restricting to the NOK runs with $\geq$MINR completed rounds, the gain is "
            r"overwhelmingly \emph{one-shot}: mean round-0 jump R0 vs.\ mean subsequent $\sum(F_g{>}0)$ REC "
            r"(PCT\% of total gain arrives at round 0), and FRAC\% of those runs are non-recursive "
            r"(ceiling / one-shot-plateau / stall). The archive saturates by round 1--2 (strategy-space "
            r"exhaustion); the only sustained-improvement cases start from low $Q_0$, i.e.\ they consume "
            r"headroom rather than demonstrate recursive self-insight. Strategy convergence (Jaccard over "
            r"self-written strategies) is only CONV --- models write \emph{diverse} strategies yet still "
            r"plateau, so the bottleneck is not strategy homogeneity but the inability to convert strategies "
            r"into compounding capability. \textbf{Not interpreted:} UNDERNAMES did not complete enough rounds "
            r"to distinguish a genuine ceiling or self-degradation from an interrupted run; in particular the "
            r"apparent DeepSeek-V3.2 ``self-degrade'' is a single-round artifact and is not a finding. This "
            r"mirrors the weight-RSI compounding null on the API track."
            .replace("NOK", str(len(ok))).replace("MINR", str(MINR))
            .replace("R0", "%+.3f" % r0).replace("REC", "%+.3f" % rec)
            .replace("PCT", "%.0f" % pct_r0).replace("FRAC", "%.0f" % frac).replace("CONV", "%.2f" % conv)
            .replace("UNDERNAMES", under_names) + "\n"
            r"\begin{table}[h]\centering\footnotesize\caption{Task-3 self-modify trajectories. "
            r"r0-jump $=Q_{g,0}-Q_0$; recursive $=\sum$ positive round-over-round gains; sat $=$ archive-"
            r"saturation round. $^\dagger$ marks runs with $<$%d completed rounds, which are reported for "
            r"completeness and excluded from every summary statistic.}"
            r"\begin{tabular}{lrrrrrl}\toprule" % MINR + "\n"
            r"model & rounds & $Q_0$ & r0-jump & recursive & sat & class \\\midrule" + "\n"
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
    """T5. Reports what the runs ACTUALLY produced, including the author-yield denominator that
    determines whether L-F is a measurement at all. An L-F computed over <5 accepted model-authored
    tasks is not interpreted (P0.3)."""
    op = load("selfplay.json").get("models", [])
    cl = load("selfplay_closed.json").get("models", [])
    diag = load("selfplay_diag.json").get("models", [])
    MINPROP = 5                       # accepted model-proposed tasks needed to interpret an L-F

    def prop(m):
        return m.get("total_model_proposed") or 0
    op_ok = [m for m in op if prop(m) >= MINPROP]
    op_under = [m for m in op if prop(m) < MINPROP]
    cl_ok = [m for m in cl if prop(m) >= MINPROP]
    op_nonzero = [m for m in op_ok if abs(m.get("final_L_minus_F") or 0) > 1e-9]
    cl_nonzero = [m for m in cl_ok if abs(m.get("final_L_minus_F") or 0) > 1e-9]
    cl_zero = [m for m in cl if abs(m.get("final_L_minus_F") or 0) <= 1e-9]

    # the two rows previously quoted as the headline positives, with their actual denominators
    quoted = sorted(op, key=lambda m: -(m.get("final_L_minus_F") or 0))[:2]
    quoted_s = "; ".join("%s $L{-}F{=}%+.3f$ on \\textbf{%d} accepted authored task%s over %d rounds" %
                         (esc(m.get("model")), m.get("final_L_minus_F") or 0, prop(m),
                          "" if prop(m) == 1 else "s", len(m.get("rounds", [])))
                         for m in quoted)

    # what the dedicated diagnosis run actually produced
    diag_line = ""
    if diag:
        d0 = diag[0]
        authored = d0.get("authored_solve_rate") or []
        n_null = sum(1 for x in authored if x is None)
        diag_line = (r" The dedicated self-play \emph{diagnosis} run (%s, %d rounds) authored \textbf{zero} "
                     r"valid tasks in every round, so all of its per-round diagnostics "
                     r"(authored solve-rate, trivial/unsolvable fractions, headroom) are undefined "
                     r"(%d/%d rounds null) and all three arms scored $C{=}0$. We therefore report it as an "
                     r"\emph{author-yield failure}, and explicitly do \emph{not} draw the "
                     r"``authored tasks are either trivial or unsolvable'' conclusion from it --- that "
                     r"bimodality is a hypothesis the run was unable to test." %
                     (esc(d0.get("model", "?")), len(authored), n_null, len(authored)))

    body = (
        r"\paragraph{T5 self-play (3-arm S/F/L): an author-yield failure, not a measured null.} "
        r"The primary metric is $L-F$ (author co-evolution). Open-weight arm: NOPEN runs; closed/API arm: "
        r"NCLOSED runs. \textbf{The binding problem is author yield, not solver capacity.} Interpreting "
        r"$L-F$ requires the live author to have actually produced accepted, novel, valid tasks; we set a "
        r"floor of MINPROP accepted model-authored tasks. Only NOPOK/NOPEN open runs and NCLOK/NCLOSED "
        r"closed runs clear it. Every closed run reports exactly $L-F=0.0$ (NCLZERO/NCLOSED rows), including "
        r"runs whose author proposed 36--37 tasks across 24 rounds --- a pattern that indicates the "
        r"co-evolution channel never engaged rather than that it engaged and returned zero."
        + diag_line +
        r" \textbf{Previously quoted positives do not survive this floor:} QUOTED. We therefore report T5 as "
        r"\emph{undefined at the current author yield} and place no $L-F$ value in the headline claims. "
        r"The honest finding this section supports is narrower and still useful: \emph{free-form task "
        r"proposal collapses}; without constrained, executable task mutation an author does not manufacture "
        r"a frontier, so self-play cannot be tested at all. Structured mutation with validity and "
        r"learnability gates is the prerequisite experiment, not an extension." + "\n")
    for k, v in (("NOPOK", str(len(op_ok))), ("NOPEN", str(len(op))),
                 ("NCLOK", str(len(cl_ok))), ("NCLZERO", str(len(cl_zero))),
                 ("NCLOSED", str(len(cl))), ("MINPROP", str(MINPROP)), ("QUOTED", quoted_s)):
        body = body.replace(k, v)
    return body


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
is provably distinguishable from a broken harness). \textbf{We must report that this control has not yet fired} on the hardware and metric of the headline: on H100 it exhausted its task set by round~2 and returned a delta of exactly zero, so the null is reported as not admissible under our own gate rather than as a validated null. The benchmark's contribution is that it \emph{separates}
the bottlenecks of kernel-code self-improvement --- generation, selection, weight-update, and curriculum ---
and makes purported self-improvement hard to fake. Secondary results: an internal correctness probe beats
uniform-random selection at matched \emph{generation} budget ($+0.124\,[+0.029,+0.218]$ clustered on base
checkpoint) but against no stronger comparator, and its within-task ranking estimand is \emph{unmeasured}
because per-candidate scores were not retained; the correctness wall below $\sim$2B is a sampling/formation
artifact rather than absent capability; and the self-play author collapses without structured task mutation,
leaving $L-F$ undefined rather than measured. We distinguish \emph{persistent} self-improvement (inherited
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
