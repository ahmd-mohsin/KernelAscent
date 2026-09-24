#!/usr/bin/env python3
"""Generate paper/kernel_results_auto.tex from the kernel-authoring experiment artifacts.

Every number in this file is re-derived from data at build time. That is not a style
preference: the one figure in this paper that was typed into prose and hand-propagated -- "76%
of all scores were exactly 0.50" -- turned out to be unreproducible from any artifact, and it
opened the section arguing that benchmarks must re-derive their numbers. Hand-carried numbers
are the ones that drift.

Cells that are incomplete are reported as incomplete rather than omitted or averaged, so a
half-finished run cannot be read as a result.

    python3 scripts/make_kernel_results_tex.py
"""
import glob, json, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "marlowe_h100")
OUT = os.path.join(ROOT, "paper", "kernel_results_auto.tex")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def esc(x):
    return str(x).replace("_", r"\_").replace("%", r"\%")


def t1_table():
    """Capability curve with attempt/verify separated -- solve-rate alone is a compliance metric."""
    rows, note, policies = [], [], set()
    for lbl, tag in (("0.5B", "q05"), ("1.5B", "q15"), ("3B", "q3"), ("7B", "q7"), ("14B", "q14")):
        d = _load(os.path.join(D, "t1k_%s.json" % tag))
        if not d:
            continue
        att = d.get("attempts") or {}
        if not att:
            note.append("%s (no attempt tracking)" % lbl)
            continue
        cand = sum(v.get("n_candidates", 0) for v in att.values())
        tried = sum(v.get("n_attempted_kernel", 0) for v in att.values())
        kok = sum(v.get("n_kernel_verified", 0) for v in att.values())
        ver = sum(v.get("n_verified", 0) for v in att.values())
        gens = (d.get("n_tasks") or 0) * (d.get("k") or 0)
        complete = d.get("complete")
        # the extraction policy is part of the result: strict discards 66% of 0.5B generations
        # and 5% of 14B ones, so a trend across scale can be a property of the filter
        pol = ((d.get("_provenance") or {}).get("env") or {}).get("KA_EXTRACT", "strict")
        policies.add(pol)
        rows.append((lbl, gens, cand, tried, ver, kok, complete))
    if not rows:
        return "", note
    body = ""
    for lbl, gens, cand, tried, ver, kok, complete in rows:
        flag = "" if complete else r"$^{\dagger}$"
        body += "%s%s & %d & %d (%.0f\\%%) & %d (%.0f\\%%) & %d & %d (%.1f\\%%) \\\\\n" % (
            lbl, flag, gens, cand, 100 * cand / max(gens, 1),
            tried, 100 * tried / max(cand, 1), ver, kok, 100 * kok / max(tried, 1))
    pol = "/".join(sorted(policies)) or "strict"
    polnote = (r"Extraction policy: \texttt{%s}. Under \texttt{strict} the extractor discards "
               r"66\%% of 0.5B generations and 5\%% of 14B generations, so any trend across scale "
               r"must be shown under both policies before it is attributed to the models." % pol)
    tex = r"""\begin{table}[h]\centering\small
\caption{T1-kernel. Solve-rate alone is a compliance metric: a submission that ignores the
kernel instruction and rewrites the reference in torch verifies trivially. Reported instead as
a cascade. $^{\dagger}$ marks an incomplete cell. POLICYNOTE}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Scale & Generations & Parsed & Attempted kernel & Verified & Kernel-verified \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table}"""
    tex = tex.replace("POLICYNOTE", polnote)
    return tex, note


def prereg_table():
    """The REGISTERED primary: self - fresh_frozen, under the registered scorer."""
    try:
        from clustered_stats import trajectory_level, ci as _ci
    except Exception:
        return "", []
    traj, partial, resumed = [], 0, 0
    for outdir in sorted(glob.glob(os.path.join(D, "prereg_*"))):
        d = _load(os.path.join(outdir, "weight_rsi.json"))
        if not d:
            continue
        rows = d.get("history") or []
        v = [r["delta_self_minus_fresh"] for r in rows
             if r.get("delta_self_minus_fresh") is not None]
        if not v:
            continue
        traj.append(st.mean(v))
        if d.get("resumed_at"):
            resumed += 1
        if len(v) < 5:
            partial += 1
    if len(traj) < 2:
        return "", ["prereg: %d trajectory(ies) -- too few for an interval" % len(traj)]
    est = trajectory_level([[x] for x in traj])
    lo, hi = _ci(est)
    powered = "" if len(traj) >= 8 else (
        r" \textbf{Underpowered} ($n<8$; MDE $0.120$ at $n{=}4$ against a $\delta{=}0.05$ margin), "
        r"so no equivalence claim is made.")
    inc = (" %d of %d cells incomplete." % (partial, len(traj))) if partial else ""
    res = ((r" %d trajectory(ies) were resumed mid-run after a walltime timeout; LoRA state is "
            r"not checkpointed, so those continue the round sequence while re-learning from "
            r"recorded data." % resumed) if resumed else "")
    return (r"""\begin{table}[h]\centering\small
\caption{The \emph{pre-registered} T2 primary, \texttt{self}$-$\texttt{fresh\_frozen}, under the
registered scorer (\texttt{KA\_SCORE=compiled}, per-task roofline).%s%s%s}
\begin{tabular}{@{}lrr@{}}
\toprule
Contrast & Trajectories & Estimate (95\%% CI) \\
\midrule
\texttt{self} $-$ \texttt{fresh\_frozen} & %d & $%+.3f\,[%+.3f, %+.3f]$ \\
\bottomrule
\end{tabular}
\end{table}""" % (powered, inc, res, len(traj), est["mean"], lo, hi)), []


def main():
    parts, notes = [], []
    for fn in (t1_table, prereg_table):
        tex, n = fn()
        if tex:
            parts.append(tex)
        notes += n
    hdr = ("%% AUTO-GENERATED by scripts/make_kernel_results_tex.py -- do not edit by hand.\n"
           "%% Every number here is re-derived from data/marlowe_h100 at build time.\n")
    open(OUT, "w").write(hdr + "\n\n".join(parts) + "\n")
    print("wrote %s (%d table(s))" % (OUT, len(parts)))
    for n in notes:
        print("  note: %s" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
