#!/usr/bin/env python3
"""Negative tests for the consistency-audit gates: plant the error, assert the gate fires.

A gate that only ever passes is indistinguishable from a gate that cannot see its input, and we
have now hit that twice in one afternoon -- once because PROSE did not include the artifacts the
claim lived in, and once because a test harness remapped the paths and every read() returned "".
Both reported "clean". So the gates get negative tests, and the harness gets a sanity assertion
that it is reading real, non-empty files before it believes any result.

    python3 tests/test_audit_gates.py
"""
import os, re, shutil, sys, tempfile

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "repo")
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns(
        ".git", "data", "*.pdf", "__pycache__", "node_modules", ".venv"))
    sys.path.insert(0, os.path.join(dst, "scripts"))
    import consistency_audit as A

    # The module resolves ROOT from its own location, so importing the COPY is what redirects it.
    # Never remap A.PROSE by hand: that is what silently produced empty reads and a false pass.
    assert A.ROOT.startswith(tmp), "audit did not rebase onto the temp copy: %s" % A.ROOT
    missing = [k for k, v in A.PROSE.items() if not os.path.exists(v)]
    assert not missing, "PROSE paths do not exist: %s" % missing
    empty = [k for k, v in A.PROSE.items() if not A.read(v).strip()]
    assert not empty, "PROSE files read empty -- the gates would pass vacuously: %s" % empty

    def run(label):
        A.FAILS.clear(); A.PASSES.clear(); A.WARNS.clear()
        A.check_baseline_attribution(); A.check_custom_kernel_rate()
        A.check_degenerate_bestofn()
        names = [c for c, _ in A.FAILS]
        print("  %-44s -> %s" % (label, names or "clean"))
        return names

    def mutate(rel, fn, expect):
        p = os.path.join(dst, rel)
        orig = open(p).read()
        open(p, "w").write(fn(orig))
        try:
            got = run("+ " + rel)
            assert expect in got, "expected %r for %s, got %r" % (expect, rel, got)
        finally:
            open(p, "w").write(orig)

    print("baseline gate: no unscoped torch.compile claims")
    assert run("unmodified repo") == [], "repo must start clean"

    mutate("paper/instrument_validity.tex",
           lambda t: t + "\nNo open model can beat torch.compile on this bank.\n",
           "baseline/attribution")
    mutate("INTUITIONS.md",
           lambda t: t + "\nModels simply cannot beat torch.compile here.\n",
           "baseline/attribution")
    mutate("docs/index.html",
           lambda t: t + "<p>Nothing we tested is faster than torch.compile.</p>\n",
           "baseline/attribution")

    print("custom-kernel gate: the numerator must stay 0 in every artifact")
    for rel, a, b in (("docs/index.html", "0 of 86", "3 of 86"),
                      ("docs/index.html", "0 of 118", "3 of 118"),
                      ("paper/instrument_validity.tex", "0 of 118", "7 of 118"),
                      ("docs/PREREGISTRATION.md", "0 of 118", "2 of 118"),
                      ("INTUITIONS.md", "0 of 118", "9 of 118")):
        mutate(rel, (lambda a, b: (lambda t: t.replace(a, b, 1)))(a, b),
               "prompt/custom-kernel-rate")

    print("best-of-N gate: the degenerate contrast must not be reported under pass-rate")
    mutate("paper/instrument_validity.tex",
           lambda t: t + "\nUnder pass-rate, lineage beats best-of-N by +0.676 in every round.\n",
           "passrate/bestofn")
    mutate("docs/index.html",
           lambda t: t + "<p>With KA_SCORE=passrate, lineage exceeds best-of-N in 37/37 rounds.</p>\n",
           "passrate/bestofn")

    assert run("all restored") == [], "repo must end clean"

    # META-GATE. Five gates in this file have silently matched nothing at some point. The rule
    # that would have caught every one of them is: a check with no negative test is not known
    # to work. This asserts the set of checks main() runs is covered by tests here, so adding
    # a gate without a test fails the suite rather than shipping an unverified pass.
    print("meta: every gate main() runs must be negative-tested here")
    src = open(os.path.join(SRC, "scripts", "consistency_audit.py")).read()
    listed = set(re.findall(r"(check_[a-z_]+)", src[src.index("def main("):]))
    mine = set(re.findall(r"A\.(check_[a-z_]+)\(\)", open(__file__).read()))
    untested = sorted(listed - mine)
    KNOWN_UNTESTED = {                      # canonical-value checks; covered by make figures
        "check_compounding", "check_search", "check_mech", "check_selfplay", "check_trackc",
        "check_embedded_claims", "check_wall_phrasing", "check_score_anchor",
        "check_unverifiable_probe", "check_probe_clustered", "check_roofline_arch",
    }
    gap = [c for c in untested if c not in KNOWN_UNTESTED]
    assert not gap, ("these gates have no negative test -- add one or justify it in "
                     "KNOWN_UNTESTED: %s" % gap)
    print("  %d gates negative-tested, %d canonical checks exempt" % (len(mine), len(KNOWN_UNTESTED)))
    shutil.rmtree(tmp)
    print("\nPASS -- every gate fires on the error it exists to catch, and clears when fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
