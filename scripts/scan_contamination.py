#!/usr/bin/env python3
"""Sweep every stored artifact for the EDQUOT zero signature.

An inode-exhausted filesystem makes build_ref_c raise, and grade_batch's except-branch returns
False for every candidate -- so a whole round scores exactly 0. We documented that (INTUITIONS
2c) and then used affected runs in the paper anyway, because nothing between the glob and the
prose asked whether a run was valid. One directory having been caught by hand is not evidence
that it was the only one.

    python3 scripts/scan_contamination.py [--strict]
"""
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The predicate must distinguish "the grader failed" from "the model scored zero". Those look
# identical in a single series and are opposite findings: a weak model legitimately scores
# C_lineage = 0 while reset scores 0.3 -- that is the published correctness-wall result, and
# flagging it as contamination would retract a real finding. A first version of this scan did
# exactly that, reporting 7 false positives.
#
# EDQUOT zeroes EVERY arm of a round simultaneously, because build_ref_c raises before any
# candidate is graded. So the signature is ALL arms exactly zero in the SAME round -- never a
# single series having zeros.
ARMS = {
    "history": [("C_lineage", "C_reset", "C_bestofN"),   # lab_compounding
                ("Qg",)],                                 # lab_track_c (single arm; use Q0)
}


def contaminated(d):
    """(is_bad, reason). Conservative: only the unambiguous grader-failure signature."""
    rows = d.get("history") or d.get("rounds")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None, None
    # multi-arm shape: every arm zero in one round can only be a grader failure
    for arms in ARMS["history"]:
        if all(a in rows[0] for a in arms) and len(arms) > 1:
            hits = [i for i, r in enumerate(rows)
                    if all((r.get(a) or 0) == 0.0 for a in arms)]
            if hits:
                return True, "all arms zero in round(s) %s" % hits
            return False, None
    # single-arm shape (track_c): a zero BASELINE cannot be produced by a working grader,
    # and a long run of zero rounds followed by normal values is the same failure.
    if "Qg" in rows[0]:
        qs = [r.get("Qg") for r in rows]
        z = sum(1 for q in qs if q == 0.0)
        if d.get("Q0") == 0.0:
            return True, "Q0 = 0.000 (baseline could not be graded)"
        if z >= 2 and any(q and q > 0.1 for q in qs):
            return True, "%d exactly-zero rounds then normal values" % z
        return False, None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if anything is contaminated (for CI)")
    a = ap.parse_args()

    bad, clean, unknown = [], 0, 0
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "**", "*.json"), recursive=True)):
        if "compiled_cache" in f or f.endswith(".prov.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        is_bad, why = contaminated(d)
        if is_bad is None:
            unknown += 1
        elif is_bad:
            bad.append((os.path.relpath(f, ROOT), why))
        else:
            clean += 1

    print("=" * 84)
    print("EDQUOT CONTAMINATION SWEEP")
    print("=" * 84)
    print("  scanned: %d interpretable artifacts (%d unrecognised shapes skipped)"
          % (clean + len(bad), unknown))
    if not bad:
        print("\n  CLEAN -- no artifact shows the zero signature.")
        return 0
    print("\n  %d artifact(s) CONTAMINATED:\n" % len(bad))
    for path, why in bad:
        print("    %-56s %s" % (path, why))
    print("""
  These must not be pooled with clean runs. Their apparent gain is recovery from a spurious
  zero baseline, not a result. route_readout excludes them automatically; any ad-hoc analysis
  must do the same.""")
    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
