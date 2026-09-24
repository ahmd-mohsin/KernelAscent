#!/usr/bin/env python3
"""Do verified submissions OPTIMISE the reference, or restate it?

Two questions the paper has to answer and currently cannot:

  * Contamination. The task bank is public-derived and the RSI bank is model-generated, so a
    reviewer will ask whether models are recalling these tasks rather than solving them.
  * Compliance vs capability. We already know a submission can ignore the kernel instruction,
    restate the reference in torch ops, and verify trivially. We have never measured how often.

Both reduce to the same measurement: how similar is a verified submission to the reference it
was asked to beat? A near-copy is evidence of recall and of non-compliance at once. It needs no
GPU and no model -- it reads stored submissions and the bank they were generated against.

Similarity is computed on NORMALISED code (comments and docstrings stripped, whitespace
collapsed, identifier case preserved) so that reformatting does not read as originality. Two
measures, because either alone is easy to fool:

  ratio  -- difflib similarity over the whole normalised body
  lcs    -- longest common contiguous block as a fraction of the submission

    python3 scripts/probe_reference_overlap.py
"""
import difflib, glob, json, os, re, sys, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "dataset", "kernel_bank", "kernel_tasks.json")


def normalise(code):
    """Strip comments, docstrings and whitespace noise. Reformatting is not originality."""
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    code = re.sub(r"'''(?:.|\n)*?'''", "", code)
    out = []
    for line in code.splitlines():
        line = re.sub(r"#.*$", "", line).strip()
        if line:
            out.append(re.sub(r"\s+", " ", line))
    return "\n".join(out)


def lcs_fraction(a, b):
    """Longest common contiguous block, as a fraction of the SUBMISSION."""
    if not a or not b:
        return 0.0
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size / max(len(a), 1)


def main():
    if not os.path.exists(BANK):
        print("bank not found: %s" % BANK); return 1
    refs = {t["name"]: t["source"] for t in json.load(open(BANK)) if "name" in t}
    print("reference bank: %d tasks" % len(refs))

    pats = sorted(glob.glob(os.path.join(ROOT, "data", "trajectories", "t1k*_q*.json")))
    if not pats:
        print("no t1k artifacts found locally -- run `mrl pull` first"); return 1

    rows = []
    for f in pats:
        cell = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f))
        except Exception:
            continue
        # Do NOT default this to "strict". KA_EXTRACT was absent from _SEMANTIC_ENV until
        # 2026-09-24, so an older artifact simply does not record it, and defaulting labelled
        # every lenient submission "strict" -- a split that looked informative and was fabricated.
        # Fall back to the cell name, which does encode it, and say when it is inferred.
        env = (d.get("_provenance") or {}).get("env") or {}
        if "KA_EXTRACT" in env:
            policy = env["KA_EXTRACT"]
        else:
            policy = ("lenient" if os.path.basename(f).startswith("t1kl_") else "strict") + "?"
        for task, subs in (d.get("kernels") or {}).items():
            ref = refs.get(task)
            if not ref:
                continue
            nref = normalise(ref)
            for s in (subs if isinstance(subs, list) else [subs]):
                code = s.get("code") if isinstance(s, dict) else s
                if not isinstance(code, str) or not code.strip():
                    continue
                ncode = normalise(code)
                rows.append({
                    "cell": cell, "policy": policy, "task": task,
                    "ratio": round(difflib.SequenceMatcher(None, nref, ncode, autojunk=False).ratio(), 4),
                    "lcs_frac": round(lcs_fraction(ncode, nref), 4),
                    "has_triton": bool(re.search(r"\btriton\b|\btl\.", code)),
                    "n_chars": len(ncode),
                })
    if not rows:
        print("no verified submissions with code found"); return 1

    def summarise(label, sel):
        if not sel:
            print("  %-22s (none)" % label); return
        r = [x["ratio"] for x in sel]; l = [x["lcs_frac"] for x in sel]
        near = sum(1 for x in sel if x["ratio"] >= 0.80)
        print("  %-22s n=%-4d  ratio med %.3f max %.3f | lcs med %.3f | >=0.80 similar: %d (%.0f%%)"
              % (label, len(sel), st.median(r), max(r), st.median(l), near, 100 * near / len(sel)))

    print("\n=== SIMILARITY OF VERIFIED SUBMISSIONS TO THE REFERENCE ===")
    summarise("all", rows)
    summarise("contains triton", [x for x in rows if x["has_triton"]])
    summarise("no triton", [x for x in rows if not x["has_triton"]])
    for pol in sorted({x["policy"] for x in rows}):
        summarise("policy=%s" % pol, [x for x in rows if x["policy"] == pol])
    if any(x["policy"].endswith("?") for x in rows):
        print("  (a trailing '?' means the policy was INFERRED from the cell name: the artifact\n"
              "   predates KA_EXTRACT being recorded in provenance)")

    near = [x for x in rows if x["ratio"] >= 0.80]
    print("""
  READ-OUT
    many near-copies (ratio >= 0.80), few with triton -> submissions are RESTATING the
      reference. That is simultaneously a compliance failure and a recall signal, and it is the
      mechanism behind the correct-at-parity spike.
    few near-copies, most with triton -> submissions are genuinely authoring kernels, the bank
      is not being recalled verbatim, and the low verify rate is a capability result.""")
    out = os.environ.get("KA_PROBE_OUT", os.path.join(ROOT, "docs", "data", "reference_overlap.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"n_submissions": len(rows), "n_tasks": len(refs),
               "median_ratio": round(st.median([x["ratio"] for x in rows]), 4),
               "frac_ratio_ge_080": round(len(near) / len(rows), 4),
               "frac_with_triton": round(sum(1 for x in rows if x["has_triton"]) / len(rows), 4),
               "rows": rows}, open(out, "w"), indent=2)
    print("\n  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
