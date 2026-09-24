#!/usr/bin/env python3
"""Cross-artifact consistency audit (P0.2).

A number that appears in the paper, the website data and the README must agree, or be explicitly
labelled as a different estimand. This script recomputes the canonical value of every
cross-referenced quantity FROM THE DATA, then checks the prose artifacts against it:

  * paper/results_auto.tex      (auto-generated -- should always agree by construction)
  * paper/kernelascent_full.tex (hand-written -- the usual source of drift)
  * paper/discussion.tex
  * README.md
  * docs/data/*.json            embedded `claim` / `note` strings that the website renders

Checks are of three kinds:
  CANON   a canonical value recomputed from data, with the stale values that must no longer appear
  DIRECT  an embedded claim string whose DIRECTION contradicts the numbers in its own file
  ORPHAN  a prose citation of a data file for a conclusion that file does not contain

Exit status is non-zero if any check fails, so this can gate `make figures` or CI.

  python3 scripts/consistency_audit.py            # report
  python3 scripts/consistency_audit.py --quiet    # only failures
"""
import json, os, re, sys, glob, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs", "data")
# Every artifact that makes a claim to a reader. This deliberately includes the paper's
# \input-ed sections (a claim in instrument_validity.tex is as public as one in the main file),
# the website, the pre-registration and the intuitions log -- the torch.compile misattribution
# lived in three of these while the audit was scanning none of them, so both of the gates added
# for it passed vacuously. A check that cannot see the text it guards reads as a pass, which is
# worse than having no check at all.
PROSE = {
    "results_auto.tex": os.path.join(ROOT, "paper", "results_auto.tex"),
    "kernelascent_full.tex": os.path.join(ROOT, "paper", "kernelascent_full.tex"),
    "discussion.tex": os.path.join(ROOT, "paper", "discussion.tex"),
    "instrument_validity.tex": os.path.join(ROOT, "paper", "instrument_validity.tex"),
    "probe_appendix.tex": os.path.join(ROOT, "paper", "probe_appendix.tex"),
    "README.md": os.path.join(ROOT, "README.md"),
    "INTUITIONS.md": os.path.join(ROOT, "INTUITIONS.md"),
    "PREREGISTRATION.md": os.path.join(ROOT, "docs", "PREREGISTRATION.md"),
    "index.html": os.path.join(ROOT, "docs", "index.html"),
}

FAILS, WARNS, PASSES = [], [], []


def load(name):
    try:
        return json.load(open(os.path.join(D, name)))
    except Exception:
        return {}


def read(path):
    try:
        return open(path, encoding="utf-8").read()
    except Exception:
        return ""


def norm_prose(text, keep_case=False):
    """Strip LaTeX/HTML/Markdown emphasis so a check matches the WORDS, not the markup.

    Five separate gates in this file have now silently matched nothing because a number or
    phrase was wrapped in \\textbf{}, <b></b> or *emphasis*: "0 of 118" was invisible next to
    "verified kernels", and a gate flagged the very sentence explaining a problem because
    "*mean* over draws" did not equal "mean over draws". A check that cannot see its input
    reports clean, which is worse than having no check.

    Whitespace is collapsed so phrases survive line wrapping in the source.
    """
    t = re.sub(r"\\(?:textbf|emph|texttt|textit|mathbf)\{", " ", text)
    t = re.sub(r"[{}]", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)                 # HTML tags
    t = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", t)       # HTML entities (&mdash; etc)
    t = t.replace("**", " ").replace("*", " ").replace("`", " ")
    t = re.sub(r"\s+", " ", t)
    return t if keep_case else t


def read_prose(name_or_path):
    """read() + norm_prose(), which is what every prose check actually wants."""
    return norm_prose(read(PROSE.get(name_or_path, name_or_path)))


def fail(check, msg):
    FAILS.append((check, msg))


def warn(check, msg):
    WARNS.append((check, msg))


def ok(check, msg):
    PASSES.append((check, msg))


def find_stale(pattern, label, allow=()):
    """Report every prose file containing `pattern`, except files listed in `allow`."""
    hits = []
    for name, path in PROSE.items():
        if name in allow:
            continue
        txt = read(path)
        for m in re.finditer(pattern, txt):
            line = txt[:m.start()].count("\n") + 1
            hits.append("%s:%d  %r" % (name, line, m.group(0)[:90]))
    if hits:
        fail(label, "stale value still present:\n      " + "\n      ".join(hits))
    else:
        ok(label, "no stale occurrences")


# ---------------------------------------------------------------------------------------
# CANON checks -- recompute from data, then assert the prose agrees
# ---------------------------------------------------------------------------------------

def check_compounding():
    d = load("compounding_tost.json")
    if not d:
        fail("compounding/canon", "docs/data/compounding_tost.json missing -- run equivalence_tost.py --out")
        return
    tr = d["pooled"]["trajectory"]
    rd = d["pooled"]["round"]
    ok("compounding/canon",
       "PRIMARY trajectory-level: n=%d traj (%d rounds) mean=%+.4f BF01=%.1f EQUIV=%s | legacy round-level n=%d BF01=%.1f"
       % (tr["n"], tr["n_rounds"], tr["mean"], tr["bf01"], tr["equivalent"], rd["n"], rd["bf01"]))

    # The old headline quoted BF01=14.7 at n=228 as if those were independent observations.
    # It may now appear ONLY inside results_auto.tex, and only in the labelled "naive" contrast.
    txt = read(PROSE["results_auto.tex"])
    ctx = re.search(r"naive round-level pooling would report.{0,120}?BF\}_\{01\}\{=\}%.1f"
                    % rd["bf01"], txt, re.S)
    if not ctx:
        warn("compounding/naive-label",
             "results_auto.tex no longer labels BF01=14.7 as the naive contrast -- verify wording")
    else:
        ok("compounding/naive-label", "BF01=14.7 appears only as the explicitly-labelled naive contrast")

    # any OTHER file still asserting the old strong-evidence framing is stale
    find_stale(r"\\bfz\{\}?\s*\$?\\?approx\}?\s*\{?14\.[57]|BF_\{01\}\}?\s*\\approx\s*14\.[57]|BF01\s*[=≈]\s*14\.[57]",
               "compounding/stale-BF01", allow=("results_auto.tex",))
    find_stale(r"n\{=\}229|\$n\{=\}228\$ round-comparisons\)\s*,\s*\\lmr",
               "compounding/stale-n", allow=("results_auto.tex",))


def check_search():
    d = load("search_vs_train.json")
    p = d.get("pooled")
    if not p:
        fail("search/canon", "docs/data/search_vs_train.json missing or has no pooled block")
        return
    ok("search/canon", "trajectory-level: n=%d traj (%d rounds) mean=%+.4f [%+.4f,%+.4f] wins %.0f%% of runs"
       % (p["n_trajectories"], p["n_rounds"], p["mean"], p["lo"], p["hi"], 100 * p["traj_wins_frac"]))
    if not p.get("ci_excludes_zero"):
        fail("search/significance",
             "the trajectory-level CI no longer excludes zero -- the 'search beats training' positive "
             "must be downgraded everywhere it is claimed")
    else:
        ok("search/significance", "trajectory-level CI still excludes zero")
    # the stale round-level number from the earlier, smaller sweep
    find_stale(r"-0\.113\s*\\,\[-0\.153,-0\.074\]|n\{=\}83", "search/stale-83",
               allow=("results_auto.tex",))


def check_mech():
    d = load("mech_analysis.json")
    ms = d.get("models", [])
    if not ms:
        fail("mech/canon", "docs/data/mech_analysis.json missing models")
        return
    n = len(ms)
    nrsi = sum(1 for m in ms if m.get("rsi"))
    ncross = sum(1 for m in ms if m.get("wall_crossed"))
    ok("mech/canon", "n=%d runs, wall-crossers=%d, compounders(rsi=True)=%d" % (n, ncross, nrsi))

    # the paper body once said 11/133 compounded; ground truth is nrsi/133
    for name, path in PROSE.items():
        txt = read(path)
        for m in re.finditer(r"(\d+)\s*/\s*%d runs compounded" % n, txt):
            got = int(m.group(1))
            if got != nrsi:
                fail("mech/compounder-count",
                     "%s says %d/%d runs compounded; mech_analysis.json says %d/%d"
                     % (name, got, n, nrsi, n))
    if not any(f[0] == "mech/compounder-count" for f in FAILS):
        ok("mech/compounder-count", "compounder count agrees with data (%d/%d)" % (nrsi, n))

    # band-count table consistency: every band table in the prose must sum to n
    for name, path in PROSE.items():
        txt = read(path)
        for tbl in re.finditer(r"\$<\$2B\s*&\s*(\d+).*?2--8B\s*&\s*(\d+).*?\\geq\}?\$?\s*[89]B\s*&\s*(\d+)",
                               txt, re.S):
            tot = sum(int(g) for g in tbl.groups())
            if tot != n:
                fail("mech/band-sum",
                     "%s has a scale-band table summing to %d, but mech_analysis.json has n=%d "
                     "(band cutoffs differ between tables)" % (name, tot, n))
    if not any(f[0] == "mech/band-sum" for f in FAILS):
        ok("mech/band-sum", "all scale-band tables sum to n=%d" % n)

    # caption n must not be a hardcoded literal that disagrees
    txt = read(PROSE["results_auto.tex"])
    for m in re.finditer(r"WHY-RSI weight-level probes by scale band\s*\(\$n\{=\}(\d+)\$", txt):
        if int(m.group(1)) != n:
            fail("mech/caption-n", "results_auto.tex mech caption says n=%s, data says n=%d" % (m.group(1), n))
    if not any(f[0] == "mech/caption-n" for f in FAILS):
        ok("mech/caption-n", "mech table caption n agrees with data")


def check_wall_phrasing():
    """The panel flagged 'sub-2B almost never cross' as contradicting the measured 54% rate.
    Any prose asserting sub-2B models NEVER emit a correct kernel is an overstatement."""
    d = load("mech_analysis.json").get("findings", {}).get("correctness_wall", {})
    lt2 = d.get("frac_cross_if_lt2B")
    if lt2 is None:
        return
    pat = re.compile(r"(sub-?2B|below 2B|<\s*2B|Sub-2B)[^.]{0,160}?(never (?:emit|produce|cross)|almost never)",
                     re.I | re.S)
    hits = []
    for name, path in PROSE.items():
        txt = read(path)
        for m in pat.finditer(txt):
            hits.append("%s:%d  %r" % (name, txt[:m.start()].count("\n") + 1, m.group(0)[:100]))
    if hits:
        fail("wall/overstatement",
             "prose says sub-2B models never cross, but the measured rate is %.0f%%:\n      %s"
             % (100 * lt2, "\n      ".join(hits)))
    else:
        ok("wall/overstatement", "no 'never crosses the wall' overstatement (measured %.0f%%)" % (100 * lt2))


def check_score_anchor():
    """The scoring rule must be described the same way everywhere: headroom-normalised against the
    per-task roofline ceiling, NOT a fixed 1.5x anchor (which the pre-registration replaced)."""
    hits = []
    for name, path in PROSE.items():
        txt = read(path)
        for m in re.finditer(r"1\.5\s*x?\s*(?:eager )?speedup|at a 1\.5x eager", txt, re.I):
            hits.append("%s:%d  %r" % (name, txt[:m.start()].count("\n") + 1, m.group(0)[:80]))
    if hits:
        fail("score/anchor",
             "prose still describes the score with a fixed 1.5x anchor; PREREGISTRATION.md defines it as "
             "headroom-normalised against the per-task roofline ceiling:\n      " + "\n      ".join(hits))
    else:
        ok("score/anchor", "scoring rule described consistently (headroom-normalised, no 1.5x anchor)")


def check_unverifiable_probe():
    """Two probe figures (within-task AUC ~0.67, matched-verification yield +0.01) are quoted in earlier
    drafts but the run that produced them is not in this repository, and per-candidate probe scores were
    never retained. They may be MENTIONED as unverifiable, never asserted as measured."""
    pat = re.compile(r"(within-task[^.]{0,80}?0\.67|0\.67[^.]{0,40}?within-task|"
                     r"harvesting (?:yield|advantage)[^.]{0,40}?0\.01)", re.I | re.S)
    MARK = r"unverifi|not in this repositor|not retained|cannot be recomputed|unmeasured|do not rely"
    hits = []
    for name, path in PROSE.items():
        txt = read(path)
        for m in pat.finditer(txt):
            window = txt[max(0, m.start() - 500):m.end() + 500]
            if not re.search(MARK, window, re.I):
                hits.append("%s:%d  %r" % (name, txt[:m.start()].count("\n") + 1, m.group(0)[:80]))
    if hits:
        fail("probe/unverifiable",
             "prose asserts a probe number whose source data are not in the repo, without marking it "
             "unverifiable:\n      " + "\n      ".join(hits))
    else:
        ok("probe/unverifiable", "unverifiable probe figures are either absent or explicitly marked")


def check_probe_clustered():
    """The probe lift must be reported at the checkpoint-clustered level, matching probe_rigor.json."""
    d = load("probe_rigor.json")
    c = d.get("clustered")
    if not c:
        return
    ok("probe/canon", "clustered lift n=%d checkpoints mean=%+.3f CI[%+.3f,%+.3f] excludes_zero=%s "
       "(n_test per run %d-%d)" % (c["n"], c["mean"], c["ci95"][0], c["ci95"][1],
                                   c["excludes_zero"], d["n_test_min"], d["n_test_max"]))


def check_roofline_arch():
    """The roofline peaks set the DENOMINATOR of the headroom score, so a result graded on
    one GPU architecture is not comparable to one graded on another. agent_bench must keep
    the constants configurable (they were once a hardcoded A100 dict) and default to the
    architecture the published results were measured on."""
    ab = os.path.join(ROOT, "kernelascent", "agent_bench.py")
    txt = read(ab)
    if not txt:
        return
    if re.search(r"^_ROOF_PEAK = \{\s*\"fp16\": \d", txt, re.M):
        fail("roofline/hardcoded",
             "agent_bench.py _ROOF_PEAK is a hardcoded dict again — H100/other-arch runs would be "
             "silently miscalibrated and an operator cannot fix it without editing code")
        return
    ok_env = all(v in txt for v in ("KA_PEAK_BF16_TFLOPS", "KA_PEAK_HBM_GBPS", "KA_ROOF_ARCH"))
    if not ok_env:
        fail("roofline/env", "agent_bench.py no longer honours KA_ROOF_ARCH / KA_PEAK_* overrides")
        return
    m = re.search(r'ROOF_ARCH_DEFAULT = "(\w+)"', txt)
    if m and m.group(1) != "a100":
        fail("roofline/default",
             "agent_bench default arch is '%s'; published results are A100-measured, so the "
             "default must stay a100 or every historical number silently rescales" % m.group(1))
    else:
        ok("roofline/arch", "peaks are env-configurable (KA_ROOF_ARCH/KA_PEAK_*), default a100 "
                            "matching the published results")


def check_selfplay():
    op = load("selfplay.json").get("models", [])
    cl = load("selfplay_closed.json").get("models", [])
    if not op:
        fail("selfplay/canon", "docs/data/selfplay.json missing models")
        return
    def prop(m):
        return m.get("total_model_proposed") or 0
    top = sorted(op, key=lambda m: -(m.get("final_L_minus_F") or 0))[:2]
    ok("selfplay/canon", "open=%d runs (max accepted proposals on any run=%d), closed=%d runs (%d exactly 0.0)"
       % (len(op), max(prop(m) for m in op), len(cl),
          sum(1 for m in cl if abs(m.get("final_L_minus_F") or 0) <= 1e-9)))
    for m in top:
        if (m.get("final_L_minus_F") or 0) > 0.05 and prop(m) < 5:
            ok("selfplay/underpowered",
               "%s L-F=%+.3f rests on %d accepted authored task(s) -- must be flagged, not headlined"
               % (m.get("model"), m.get("final_L_minus_F"), prop(m)))

    # README must not still advertise the stale 'emerging positives'
    find_stale(r"Claude-Sonnet-5\s*\|\s*\+0\.045|GPT-5\.6-sol\s*\|\s*\+0\.041",
               "selfplay/stale-readme")

    # ORPHAN: prose citing selfplay_diag.json for the bimodality conclusion it does not contain
    diag = load("selfplay_diag.json").get("models", [])
    has_rates = any(any(x is not None for x in (m.get("authored_solve_rate") or [])) for m in diag)
    if not has_rates:
        # A mention is only a violation if the bimodality is ASSERTED. Prose that names the file and
        # then retracts/limits the conclusion is exactly what we want, so look for a retraction marker
        # inside the same window before failing.
        RETRACT = r"retract|not in the data|unable to test|do not draw|hypothesis|authored \\textbf\{zero\}|undefined"
        for name, path in PROSE.items():
            txt = read(path)
            for m in re.finditer(r"selfplay\\?_diag[^)]*\).{0,900}?(trivial|unsolvable)", txt, re.S | re.I):
                window = txt[m.start():m.end() + 600]
                if not re.search(RETRACT, window, re.I):
                    fail("selfplay/orphan-diag",
                         "%s attributes the trivial/unsolvable bimodality to selfplay_diag.json without "
                         "retraction, but every authored_solve_rate in that file is null (the run authored "
                         "zero tasks)" % name)
        if not any(f[0] == "selfplay/orphan-diag" for f in FAILS):
            ok("selfplay/orphan-diag",
               "no prose attributes a bimodality conclusion to the empty selfplay_diag.json")


def check_trackc():
    ms = load("trackc.json").get("models", [])
    if not ms:
        return
    thin = [m for m in ms if (m.get("rounds") or 0) < 3]
    ok("trackc/canon", "%d rows; %d with <3 rounds (%s) must be flagged, not interpreted"
       % (len(ms), len(thin), ", ".join("%s r=%s" % (m["model"], m.get("rounds")) for m in thin)))
    # the DeepSeek self-degrade must be labelled an artifact wherever its number appears
    for name, path in PROSE.items():
        txt = read(path)
        if re.search(r"-0\.23[34]", txt) and not re.search(r"artifact|not a finding|not interpreted", txt, re.I):
            fail("trackc/selfdegrade",
                 "%s quotes the DeepSeek-V3.2 -0.233/-0.234 self-degrade without labelling it a "
                 "single-round artifact" % name)
    if not any(f[0] == "trackc/selfdegrade" for f in FAILS):
        ok("trackc/selfdegrade", "self-degrade number is labelled an artifact wherever it appears")


# ---------------------------------------------------------------------------------------
# DIRECT checks -- an embedded claim string contradicted by numbers in its own file
# ---------------------------------------------------------------------------------------

def check_embedded_claims():
    d = load("mech_analysis.json")
    f = d.get("findings", {})
    # (1) diversity direction
    awc = f.get("among_wall_crossers", {})
    pair = awc.get("rsi_vs_flat_diversity")
    claim = (awc.get("claim") or "")
    if pair and len(pair) == 2:
        rsi_div, flat_div = pair
        says_higher = re.search(r"higher (?:sustained )?generation diversity|sustain.{0,20}higher.{0,20}diversity",
                                claim, re.I)
        if says_higher and rsi_div < flat_div:
            fail("mech/claim-diversity",
                 "mech_analysis.json among_wall_crossers.claim says compounders sustain HIGHER diversity, "
                 "but its own numbers are rsi=%.3f < flat=%.3f (the website renders this string)"
                 % (rsi_div, flat_div))
        else:
            ok("mech/claim-diversity", "diversity claim string matches its numbers (rsi=%.3f vs flat=%.3f)"
               % (rsi_div, flat_div))
    # (2) 'almost never cross' vs the actual sub-2B crossing rate
    cw = f.get("correctness_wall", {})
    lt2 = cw.get("frac_cross_if_lt2B")
    cclaim = cw.get("claim") or ""
    if lt2 is not None:
        if re.search(r"almost never cross", cclaim, re.I) and lt2 > 0.2:
            fail("mech/claim-wall",
                 "mech_analysis.json correctness_wall.claim says sub-2B 'almost never cross', but "
                 "frac_cross_if_lt2B=%.2f -- state it as 'less frequently than >=2B' with counts" % lt2)
        else:
            ok("mech/claim-wall", "correctness-wall claim string is consistent with %.2f crossing rate" % lt2)


def check_baseline_attribution():
    """Do not attribute a result to `torch.compile` unless the runs actually scored against it.

    This is the sixth defect, made into a gate. The default scorer is KA_SCORE=eager, which
    measures speedup over EAGER against the legacy fixed 1.5x anchor -- the compiled baseline and
    the per-task roofline ceiling are both unused. We nonetheless wrote, in three artifacts, that
    H100's stronger `torch.compile` had closed the headroom. It was plausible, it fit the data,
    and it was about a quantity those runs never measured.

    So: any sentence claiming models cannot BEAT / are not FASTER THAN torch.compile is a failure
    unless it is explicitly scoped to compiled-mode runs. Struck-through text and the passages
    that document the error are exempt -- the point is to keep the correction, not erase it.
    """
    pat = re.compile(r"[^.\n]*\b(?:beat|faster than|outperform\w*|exceed\w*)\s+"
                     r"(?:the\s+)?(?:\\texttt\{)?`?torch\.?compile", re.I)
    # Exemption is scoped to the SENTENCE the claim sits in, not a wide context window. A window
    # was the first attempt and it was useless: instrument_validity.tex is *about* this mistake,
    # so every marker appears within a few hundred characters of everything, and the gate
    # exempted a deliberately-planted bad claim. Narrow scope is what makes it able to fire.
    exempt = ("was wrong", "~~", "never entered", "no claim about", "initial diagnosis",
              "sixth defect", "ka_score=compiled", "compiled-mode", "mistake",
              "justification was not", "cannot beat the baseline on this hardware")
    hits = []
    for name, path in PROSE.items():
        txt = read(path)
        for m in pat.finditer(txt):
            line_no = txt[:m.start()].count("\n") + 1
            lo = max(txt.rfind(".", 0, m.start()), txt.rfind("\n", 0, m.start())) + 1
            hi = m.end() + 160
            for stop in (".", "\n"):
                k = txt.find(stop, m.end())
                if k != -1:
                    hi = min(hi, k + 1)
            sentence = txt[lo:hi]
            if any(e in sentence.lower() for e in exempt):
                continue
            hits.append("%s:%d  %r" % (name, line_no, m.group(0).strip()[:90]))
    if hits:
        fail("baseline/attribution",
             "prose claims a result about `torch.compile`, but the default scorer (KA_SCORE=eager) "
             "measures speedup over EAGER with a fixed 1.5x anchor and never touches the compiled "
             "baseline. Scope the claim to compiled-mode runs or restate it:\n      "
             + "\n      ".join(hits))
    else:
        ok("baseline/attribution", "no unscoped torch.compile claims (the sixth-defect gate)")


def check_custom_kernel_rate():
    """The 0-of-86 custom-kernel figure must agree wherever it appears.

    It is the load-bearing number for the prompt finding -- it is why the 0.50 spike is a real
    measurement rather than a dead instrument -- so it is exactly the kind of number that drifts
    between a paper, a website and a notes file.

    These three artifacts are LaTeX, HTML and Markdown, and each puts its own emphasis markup
    between the digits and the noun ("<b>0 of 86</b> verified kernels", "\\textbf{zero}"). Matching
    the raw text silently matches nothing, so strip markup and normalise number words first --
    a check that cannot fire is worse than no check, because it reads as a pass.
    """
    def norm(t):
        return re.sub(r"\bzero\b", "0", norm_prose(t), flags=re.I)

    # Match "N of M" anywhere in the NEIGHBOURHOOD of "verified kernel", not only when the
    # phrase immediately follows. Requiring adjacency meant "...0 of 118, for 0 of 204 across
    # both" was invisible: the gate reported clean while reading none of it. That is the third
    # check today that could not see its own input, so the window is deliberately generous --
    # a false positive is a conversation, a false negative is a silent pass.
    win = re.compile(r"verified kernels?", re.I)
    num = re.compile(r"(\d+)\s*(?:of|/|out of)\s*(\d+)\b")
    seen = {}
    for name, path in PROSE.items():
        txt = norm(read(path))
        for w in win.finditer(txt):
            lo, hi = max(0, w.start() - 220), w.end() + 220
            chunk = txt[lo:hi]
            for m in num.finditer(chunk):
                a, b = int(m.group(1)), int(m.group(2))
                if b < 10 or b > 100000 or a > b:      # not a kernel-count pair
                    continue
                # The zero-numerator claim is about the PUBLISHED (safe) prompt, where models
                # do not attempt kernels at all. Under KA_PROMPT=kernel they do, and succeed --
                # 6 of 10 verified kernels from the 0.5B cell contain triton. Those are opposite
                # conditions, so a kernel-prompt citation must not be judged against a
                # safe-prompt rule; that would forbid reporting the result the fix produced.
                if re.search(r"kernel prompt|KA_PROMPT=kernel|t1k|asked explicitly|when asked",
                             chunk, re.I):
                    continue
                # "14/29 tasks" is TASK COVERAGE, a different quantity that happens to sit near
                # this prose. Widening the window to see every kernel citation also swallowed
                # those, so the unit has to be checked, not just the proximity.
                # also treat "covers N/M" as coverage language, whatever noun follows
                before = chunk[max(0, m.start() - 12):m.start()].lower()
                if re.match(r"\s*(?:task|of 29|held|bank)", chunk[m.end():m.end() + 12], re.I) \
                   or "cover" in before or "solve" in before:
                    continue
                seen.setdefault((a, b), []).append(name)

    if not seen:
        return                                    # figure not cited anywhere; nothing to check

    # The LOAD-BEARING claim is that the numerator is zero, not that one denominator is used
    # everywhere. Several denominators are legitimate once a result is replicated (86 original,
    # 118 re-harvest, 204 combined), and failing on that would punish replication. A non-zero
    # numerator anywhere, or a cited total that does not match its parts, is a real error.
    bad = {k: v for k, v in seen.items() if k[0] != 0}
    if bad:
        fail("prompt/custom-kernel-rate",
             "prose says a custom kernel WAS found, but every harvest measured zero: "
             + "; ".join("%d of %d in %s" % (a, b, ",".join(sorted(set(v)))) for (a, b), v in bad.items()))
        return
    # DELIBERATELY NARROW. An earlier version also required the denominators to add up
    # (86 + 118 = 204) to catch a stale figure. Matching free prose cannot do that reliably --
    # it kept binding to incidental "N of M" pairs nearby, and I patched it three times before
    # accepting that the arithmetic rule produces more false alarms than it prevents errors.
    # A gate that cries wolf gets ignored, which costs more than the staleness it might catch.
    #
    # What survives is the load-bearing claim: the NUMERATOR is zero. That is the fact the
    # prompt finding rests on, it is unambiguous, and it is what would actually be wrong if
    # someone mis-transcribed a later harvest.
    dens = sorted({k[1] for k in seen})
    ok("prompt/custom-kernel-rate",
       "custom-kernel numerator is 0 in every citation; denominators seen: %s (%s)"
       % (dens, ", ".join(sorted({n for v in seen.values() for n in v}))))


def check_degenerate_bestofn():
    """lineage-minus-bestofN must never be reported under pass-rate.

    Best-of-N's mechanism is the MAX over k*(r+1) draws; pass-rate is a MEAN, invariant to the
    number of draws (measured: budget grew 5x, C_bestofN moved -0.024). The contrast is
    therefore meaningless under that metric -- and it comes out at +0.676, positive in 37/37
    rounds, which would REVERSE the published search-beats-training result with an artifact.
    It is exactly the kind of number that gets copied into prose because it looks decisive.
    """
    pat = re.compile(r"[^.\n]{0,160}(?:best.?of.?[nk])[^.\n]{0,160}", re.I)
    hits = []
    for name, path in PROSE.items():
        # Strip emphasis before matching. "*mean* over draws" did not match the exemption
        # "mean over draws" and the gate flagged the very sentence that explains the problem --
        # the same markup blindness that made the custom-kernel gate inert.
        txt = norm_prose(read(path))
        for m in pat.finditer(txt):
            seg = m.group(0)
            # a markdown table row comparing metrics is a definition, not a reported contrast
            if seg.lstrip().startswith("|") or seg.count("|") >= 2:
                continue
            if not re.search(r"pass.?rate|passrate|KA_SCORE", seg, re.I):
                continue
            # the passages that EXPLAIN the degeneracy are the point, not a violation
            if re.search(r"degenerat|invariant|do not report|does not report|not report|"
                         r"mean over draws|max(imum)? over|artifact|refus", seg, re.I):
                continue
            hits.append("%s:%d  %r" % (name, txt[:m.start()].count("\n") + 1, seg.strip()[:100]))
    if hits:
        fail("passrate/bestofn",
             "prose appears to report a best-of-N contrast under pass-rate, where the baseline "
             "cannot benefit from its budget and the number is an artifact:\n      "
             + "\n      ".join(hits))
    else:
        ok("passrate/bestofn", "no best-of-N contrast reported under pass-rate")


RETRACTED = [
    # (label, pattern that ASSERTS the retracted claim, pattern whose presence nearby means
    #  the text is discussing/withdrawing it rather than asserting it)
    ("t3/one-shot",
     r"(?:is|are|gain is)\s+(?:overwhelmingly\s+)?(?:\w+\s+){0,2}one.shot|one.shot self.modification|"
     r"real,? but one.shot|Procedure.RSI:\s*real",
     # word-boundaried: a bare "cap" matched "CAPability" in unrelated nearby prose and
     # exempted a deliberately planted violation. Substring exemptions are how a gate goes quiet.
     r"\bretract\w*|not earned|\bconfound\w*|\bappears\b|\bwithdraw\w*|\bcap\b|\bcaps\b|"
     r"not interpretable|RETRACTED"),
    ("torch.compile/headroom",
     r"torch\.?compile\}?\s+baseline\s+strong\s+enough|compile\s+is\s+relatively\s+stronger",
     r"\bretract\w*|was wrong|never used|never entered|earlier version|~~"),
    ("positive-control/delivered",
     r"positive control[^.]{0,80}so a null is provably distinguishable",
     r"\bretract\w*|has not|did not fire|ran out of work|not admissible"),
]


def check_retractions_propagated():
    """A retracted claim must not survive anywhere as an assertion.

    The T3 'one-shot' reading was retracted in one file and left standing in four others, so the
    released artifact set contained each retracted claim beside its own retraction. That is worse
    than never retracting: it reads as concealment. Retractions are cheap to make and expensive
    to propagate, so propagation is mechanical from here.
    """
    hits = []
    for label, assert_pat, exempt_pat in RETRACTED:
        ap = re.compile(assert_pat, re.I)
        ep = re.compile(exempt_pat, re.I)
        for name, path in PROSE.items():
            txt = norm_prose(read(path))
            for m in ap.finditer(txt):
                lo, hi = max(0, m.start() - 260), m.end() + 260
                if ep.search(txt[lo:hi]):
                    continue
                hits.append("%s  [%s]  %r" % (name, label, m.group(0).strip()[:70]))
    if hits:
        fail("retractions/propagated",
             "a RETRACTED claim is still asserted, with no retraction nearby:\n      "
             + "\n      ".join(hits))
    else:
        ok("retractions/propagated", "all %d retracted claims are withdrawn everywhere they appear" % len(RETRACTED))


def check_intervals_contain_estimates():
    """A point estimate printed beside an interval must lie inside it.

    The paper reported "lineage-bestofN = -0.111 [-0.138,-0.083] (the cluster-robust interval
    [-0.163,-0.120] agrees)". It does not agree -- it excludes -0.111, because the two are
    different estimands (trajectory-weighted vs round-weighted) with different point estimates.
    An interval that excludes the number quoted beside it is the first thing a careful reviewer
    notices, and it undermines confidence in every other interval in the paper.
    """
    pat = re.compile(r"([+-]?\d*\.\d+)\s*\\?,?\s*\[\s*([+-]?\d*\.\d+)\s*,\s*([+-]?\d*\.\d+)\s*\]")
    hits = []
    for name, path in PROSE.items():
        txt = norm_prose(read(path))
        for m in pat.finditer(txt):
            est, lo, hi = (float(m.group(i)) for i in (1, 2, 3))
            if lo > hi:
                lo, hi = hi, lo
            if not (lo - 1e-9 <= est <= hi + 1e-9):
                hits.append("%s  %r" % (name, m.group(0)))
    if hits:
        fail("stats/interval-contains-estimate",
             "a point estimate is printed beside an interval that excludes it:\n      "
             + "\n      ".join(hits))
    else:
        ok("stats/interval-contains-estimate", "every estimate printed beside an interval lies inside it")


def check_table_sums():
    """Per-scale trajectory counts must sum to the pooled n printed beside them.

    A single-trajectory cell has no computable interval, so its ROW was dropped while its
    trajectory still counted toward the pooled total: the table read 12+20+19+5 = 56 against a
    bolded 57, with the missing 14B run invisible. A table a reader cannot add up is not
    auditable, and 'underpowered scales are marked' is false if one of them is absent.
    """
    d = load("compounding_tost.json")
    models, pooled = d.get("models") or {}, (d.get("pooled") or {}).get("trajectory") or {}
    if not models or not pooled:
        return
    rows = sum((v or {}).get("n_trajectories") or 0 for v in models.values())
    if rows != pooled.get("n"):
        fail("tables/sum",
             "per-scale trajectory counts sum to %d but the pooled n is %d -- a cell is in the "
             "pooled figure with no row of its own" % (rows, pooled.get("n")))
        return
    rounds = sum((v or {}).get("n_rounds_total") or 0 for v in models.values())
    if rounds != pooled.get("n_rounds"):
        fail("tables/sum", "per-scale round counts sum to %d but the pooled figure says %d"
             % (rounds, pooled.get("n_rounds")))
        return
    ok("tables/sum", "per-scale rows sum to the pooled totals (%d trajectories, %d rounds)"
       % (rows, rounds))


def check_band_table():
    """The scale-band table must be recomputable from the data at a single binning.

    The paper and the generated results used different cuts (>=9B vs >=8B) while the generated
    file claimed the bands were "used consistently throughout" -- and the paper's largest-band
    row MIXED them: n=21 and drift=0.726 are the >=9B cut, retention=0.339 is the >=8B value.
    Two binnings of the same 133 runs, reported as one table.
    """
    import statistics as _st
    d = load("mech_analysis.json")
    ms = d.get("models") or []
    if not ms:
        return
    bands = {"<2B": [], "2--8B": [], ">=8B": []}
    for m in ms:
        p_ = m.get("size_b") or 0
        bands["<2B" if p_ < 2 else ("2--8B" if p_ < 8 else ">=8B")].append(m)
    truth = {b: (len(g),
                 round(_st.mean([x["drift_total"] for x in g]), 3),
                 round(_st.mean([1.0 if x["rsi"] else 0.0 for x in g]), 2))
             for b, g in bands.items() if g}
    hits = []
    for name, path in PROSE.items():
        txt = norm_prose(read(path))
        # Match each band UNAMBIGUOUSLY. Stripping ">=" turned ">=8B" into "8B", which also
        # occurs inside "2--8B" -- so the gate compared the mid-band row against the top-band
        # count and reported a mismatch that did not exist. A band label is a prefix as much as
        # a number; match the whole thing.
        # Math delimiters sit INSIDE the label: the TeX is "$\ge$8B", not "\ge 8B". A pattern
        # requiring them adjacent matched nothing and the gate passed having examined no rows --
        # the same vacuous pass that has now bitten five separate checks. `D` absorbs the $ and
        # any braces wherever they fall.
        D = r"[\$\s{}]*"
        pats = {"<2B":   r"(?:<|&lt;)" + D + r"2B" + D + r"&\s*(\d+)\s*&",
                "2--8B": r"2" + D + r"-{1,2}" + D + r"8B" + D + r"&\s*(\d+)\s*&",
                ">=8B":  r"(?:\\ge|>=|\u2265|&ge;)" + D + r"8B" + D + r"&\s*(\d+)\s*&"}
        for b, (n, drift, rsi) in truth.items():
            for m in re.finditer(pats[b], txt, re.I):
                got = int(m.group(1))
                if got != n:
                    hits.append("%s  band %s: table says n=%d, data says n=%d" % (name, b, got, n))
    if hits:
        fail("mech/band-table", "the scale-band table does not match the data:\n      "
             + "\n      ".join(sorted(set(hits))))
    else:
        ok("mech/band-table", "band table matches mech_analysis.json at one binning (%s)"
           % ", ".join("%s n=%d" % (b, v[0]) for b, v in sorted(truth.items())))


def check_extraction_policy_stated():
    """A kernel-authoring result must say which extraction policy produced it.

    `strict` discards 66% of 0.5B generations and 5% of 14B ones. A filter whose severity
    correlates with the independent variable can manufacture a trend across scale, so the policy
    is part of the result and not part of the plumbing. We reported a p = 0.0038 trend before
    noticing; this gate exists so the next one cannot be reported without its policy attached.
    """
    pat = re.compile(r"verify.{0,4}given.{0,4}attempt|verify\|attempt|attempt rate", re.I)
    hits = []
    for name, path in PROSE.items():
        txt = norm_prose(read(path))
        for m in pat.finditer(txt):
            lo, hi = max(0, m.start() - 700), m.end() + 700
            ctx = txt[lo:hi]
            if re.search(r"KA_EXTRACT|extraction policy|strict|lenient", ctx, re.I):
                continue
            # A DEFINITION carries no number and needs no policy -- the pre-registration defines
            # these terms before any data exists. Only a reported FIGURE is policy-dependent, so
            # require a percentage or ratio within the same sentence.
            lo2 = max(0, m.start() - 120)
            near = txt[lo2:m.end() + 120]
            if not re.search(r"\d+(?:\.\d+)?\s*(?:%|\\%)|\d+\s*/\s*\d+", near):
                continue
            hits.append("%s  %r" % (name, near.strip()[:70]))
    if hits:
        fail("extraction/policy-stated",
             "a verify-given-attempt figure appears with no extraction policy nearby; strict "
             "discards 66%% of 0.5B and 5%% of 14B generations, so the policy is part of the "
             "result:\n      " + "\n      ".join(sorted(set(hits))))
    else:
        ok("extraction/policy-stated", "every attempt/verify figure states its extraction policy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="print failures only")
    a = ap.parse_args()
    for fn in (check_compounding, check_search, check_mech, check_selfplay,
               check_trackc, check_embedded_claims, check_wall_phrasing, check_score_anchor,
               check_unverifiable_probe, check_probe_clustered, check_roofline_arch,
               check_baseline_attribution, check_custom_kernel_rate,
               check_degenerate_bestofn, check_retractions_propagated,
               check_intervals_contain_estimates,
               check_table_sums, check_band_table,
               check_extraction_policy_stated):
        fn()
    if not a.quiet:
        print("=" * 100)
        print("CANONICAL VALUES (recomputed from docs/data)")
        print("=" * 100)
        for c, m in PASSES:
            print("  [ok]   %-28s %s" % (c, m))
    if WARNS:
        print("\n" + "=" * 100 + "\nWARNINGS\n" + "=" * 100)
        for c, m in WARNS:
            print("  [warn] %-28s %s" % (c, m))
    print("\n" + "=" * 100)
    print("FAILURES (%d)" % len(FAILS))
    print("=" * 100)
    for c, m in FAILS:
        print("  [FAIL] %-28s %s" % (c, m))
    if not FAILS:
        print("  none -- all cross-referenced numbers agree")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
