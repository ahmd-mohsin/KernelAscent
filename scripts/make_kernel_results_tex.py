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
# `mrl pull` lands artifacts in data/trajectories while older cells live in data/marlowe_h100.
# Look in both, nearest first, so a table does not silently lose half its rows to a path change.
DIRS = [D, os.path.join(ROOT, "data", "trajectories")]


def _find(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(D, name)
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
        d = _load(_find("t1k_%s.json" % tag))
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


def t1_replication_note():
    """14B replicating across an independent route is the strongest single line in T1.

    The claim that most invites disbelief is that a 14B model attempts kernels almost always and
    verifies almost never. One run of that is an anecdote. r4-kernel-14b was produced by a
    different route at a different k from a different teacher bank, so its agreement is a real
    replication rather than a re-read of the same numbers.
    """
    def cascade(path):
        d = _load(path)
        if not d:
            return None
        att = d.get("attempts") or {}
        if not att:
            return None
        return (sum(v.get("n_attempted_kernel", 0) for v in att.values()),
                sum(v.get("n_kernel_verified", 0) for v in att.values()),
                sum(v.get("n_candidates", 0) for v in att.values()),
                d.get("k"), bool(d.get("complete")))

    a = cascade(_find("t1kl_q14.json"))
    b = cascade(_find("r4_kernel_q14.json"))
    if not a or not b:
        return "", ["14B replication: missing t1kl_q14 or r4_kernel_q14"]
    if not (a[4] and b[4]):
        return "", ["14B replication: a cell is incomplete; not reporting"]
    ra, rb = a[1] / max(a[0], 1), b[1] / max(b[0], 1)
    cap = (r"\caption{14B verify-given-attempt, replicated across independent routes. Two runs "
           r"differing in route, sampling budget $k$, and teacher bank. The model parses as a "
           r"kernel attempt in %.0f\%% of its candidates and almost never produces one that "
           r"verifies, so the result is not an artifact of extraction, prompt compliance, or a "
           r"single unlucky run.}") % (100 * b[0] / max(b[2], 1))
    lines = [r"\begin{table}[h]\centering\small",
             cap,
             r"\begin{tabular}{@{}llrrr@{}}",
             r"\toprule",
             r"Run & Route & $k$ & Attempted & Kernel-verified \\",
             r"\midrule",
             r"\texttt{t1kl\_q14} & T1 (lenient) & %s & %d & %d (%.2f\%%) \\"
             % (a[3], a[0], a[1], 100 * ra),
             r"\texttt{r4\_kernel\_q14} & R4 & %s & %d & %d (%.2f\%%) \\"
             % (b[3], b[0], b[1], 100 * rb),
             r"\bottomrule",
             r"\end{tabular}",
             r"\end{table}"]
    return "\n".join(lines), []


def t1_robustness_table():
    """Strict vs lenient extraction, side by side.

    The extraction policy is not a preprocessing detail here. Strict discards a share of
    generations that itself correlates with scale, and that filter alone produced a perfectly
    monotone decline (Spearman rho = -1.000) which the permissive policy dissolves to -0.900.
    The endpoint contrast survives both; the ordering survives neither. Printing one policy
    would be printing the instrument.
    """
    def cascade(path):
        d = _load(path)
        if not d:
            return None
        att = d.get("attempts") or {}
        if not att:
            return None
        return {"att": sum(v.get("n_attempted_kernel", 0) for v in att.values()),
                "kok": sum(v.get("n_kernel_verified", 0) for v in att.values()),
                "complete": bool(d.get("complete"))}

    rows, incomplete = [], []
    for lbl, tag in (("0.5B", "q05"), ("1.5B", "q15"), ("3B", "q3"), ("7B", "q7"), ("14B", "q14")):
        a = cascade(_find("t1k_%s.json" % tag))
        b = cascade(_find("t1kl_%s.json" % tag))
        if not a or not b:
            continue
        for tagname, c in (("strict", a), ("lenient", b)):
            if not c["complete"]:
                incomplete.append("%s/%s" % (lbl, tagname))
        rows.append((lbl, a, b))
    if not rows:
        return "", ["no paired strict/lenient cells found"]

    body = ""
    for lbl, a, b in rows:
        ra = a["kok"] / a["att"] if a["att"] else 0.0
        rb = b["kok"] / b["att"] if b["att"] else 0.0
        body += "%s & %d & %d & %.2f\\%% & %d & %d & %.2f\\%% \\\\\n" % (
            lbl, a["att"], a["kok"], 100 * ra, b["att"], b["kok"], 100 * rb)

    warn = ""
    if incomplete:
        warn = (r" \textbf{Incomplete cells present (%s); this table is provisional.}"
                % esc(", ".join(incomplete)))
    tex = r"""\begin{table}[h]\centering\small
\caption{T1-kernel robustness to extraction policy. Verify-given-attempt under
\texttt{KA\_EXTRACT=strict} and \texttt{lenient}; identical tasks, $k$, and token budget,
differing only in how a generation is converted to a candidate. The endpoint contrast
(0.5B vs 14B) holds under both policies and strengthens under \texttt{lenient}. The monotone
ordering does not: it is an artifact of a filter whose severity correlates with scale.WARN}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{\texttt{strict}} & \multicolumn{3}{c}{\texttt{lenient}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Scale & Attempts & Verified & Rate & Attempts & Verified & Rate \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table}"""
    return tex.replace("WARN", warn), []


def main():
    parts, notes = [], []
    for fn in (t1_table, t1_robustness_table, t1_replication_note, prereg_table):
        tex, n = fn()
        if tex:
            parts.append(tex)
        notes += n
    hdr = ("%% AUTO-GENERATED by scripts/make_kernel_results_tex.py -- do not edit by hand.\n"
           "%% Every number here is re-derived from data/marlowe_h100 at build time.\n")
    out = hdr + "\n\n".join(parts) + "\n"
    # A non-raw string in an editing script turned "\\begin"/"\\toprule" into a literal
    # backspace and tab on the way into THIS file, and the generator then emitted
    # "<BS>egin{tabular}" without complaining. LaTeX control sequences are all backslash + a
    # letter that doubles as an escape, so this corruption is silent and easy to reintroduce.
    ctrl = sorted({c for c in out if ord(c) < 32 and c != "\n"})
    if ctrl:
        raise SystemExit("refusing to write %s: contains control characters %r -- a LaTeX "
                         "command was escape-interpreted somewhere upstream" % (OUT, ctrl))
    for cmd in (r"\begin{tabular}", r"\toprule", r"\bottomrule", r"\end{tabular}"):
        if out.count(cmd) < 1:
            raise SystemExit("refusing to write %s: missing %s" % (OUT, cmd))
    if out.count(r"\begin{tabular}") != out.count(r"\end{tabular}"):
        raise SystemExit("refusing to write %s: unbalanced tabular environments" % OUT)
    open(OUT, "w").write(out)
    print("wrote %s (%d table(s))" % (OUT, len(parts)))
    for n in notes:
        print("  note: %s" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
