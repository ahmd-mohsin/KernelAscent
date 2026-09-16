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
\textbf{Multi-seed confirmation (2--3 seeds $\times$ 6 rounds each):} the mean lineage$-$reset with a 95\%%
task/round-clustered CI \emph{spans zero at every scale} --- Qwen-0.5B $+0.042\,[-0.044,+0.129]$, Qwen-1.5B
$+0.032\,[-0.049,+0.114]$, Qwen-3B $+0.045\,[-0.060,+0.150]$; pooled ($n{=}25$) $+0.040\,[-0.014,+0.094]$.
So the small early-round lift is \emph{not} statistically distinguishable from zero --- a \textbf{bounded null}
(effect $\lesssim 0.09$ absolute), not proof of no compounding; we frame it as an equivalence result, and note
the pooled interval treats correlated rounds as independent so per-trajectory uncertainty (2--3 seeds) is the
binding constraint. \textbf{The two positive findings are the headline, not the null:} (i) at matched compute,
\textbf{best-of-$N$ search beats lineage self-training} in most rounds (lineage$-$bestofN $<0$ throughout) ---
on roofline-graded kernels with a full generation$+$verification$+$training compute ledger; and (ii) the
\textbf{coverage gap} --- a 14B teacher harvests $\sim$10$\times$ more verified-correct kernels than a sub-2B
student on the same tasks, and the sub-2B ``correctness wall'' is a \emph{pass@k artifact} (low per-sample $p$,
not zero coverage), so self-training \emph{sharpens} what the model already covers rather than expanding it ---
the mechanism for the null. \emph{Scope:} rejection-sampling SFT (not RL/GRPO); tested at 0.5--3B --- the 7B/14B
3-arm (lineage/reset/search) replication with a pre-registered coverage-gap prediction is in flight to test the
insufficient-supply-at-small-scale alternative.
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

\subsection{Mechanism and self-play}
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
""" % (datetime.date.today().isoformat(), probe_table(), compounding_note(), mech_note(), selfplay_note()))
    open(OUT, "w").write(body)
    print("wrote", OUT, "(%d bytes)" % len(body))

if __name__ == "__main__":
    build()
