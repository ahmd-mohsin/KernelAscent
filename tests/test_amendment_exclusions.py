#!/usr/bin/env python3
"""The Amendment 5 and 6 rules must exclude bad trajectories AND include valid ones.

A rule that only ever excludes is as wrong as one that never does: the registered primary is
re-running precisely so that resumed-and-restored cells COUNT. Getting this backwards would
silently discard the replacement data and leave the paper with no primary at all.

The discriminator is the PRESENCE of `adapter_restored`, not its truthiness. It was added with
the fix, so a pre-fix artifact lacks the key entirely, while a post-fix run that never resumed
stamps it as null. An earlier version keyed on `resumed_at` alone and let four stale cells
through -- they had resumed_at=None only because they were snapshotted before their first
timeout.

    python3 tests/test_amendment5_exclusion.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def verdict(d):
    """Mirror of the rule in make_kernel_results_tex.prereg_table.

    Covers Amendment 5 (severed adapters) AND Amendment 6 (starved self arm). Amendment 6 was
    registered with prose and no enforcement, and the generator duly published a contrast of
    -0.019 from three cells whose mean n_ex was 0.20, 1.80 and 1.20. A rule that lives only in a
    markdown file is a rule nobody applies.
    """
    if "adapter_restored" not in d:
        return "exclude (pre-fix artifact)"
    if d.get("resumed_at") and d.get("adapter_restored") is not True:
        return "exclude (severed)"
    rows = d.get("history") or []
    # Only rounds that actually record n_ex can be judged for starvation. The Amendment 5 cases
    # below use bare ints as round placeholders and carry no n_ex, so they must fall through to
    # the Amendment 5 verdict rather than crash or be silently treated as zero-fuel.
    nex = [r.get("n_ex", 0) for r in rows if isinstance(r, dict) and "n_ex" in r]
    if nex and (sum(nex) / len(nex)) < 2.0:
        return "exclude (starved)"
    return "include"


CASES = [
    # observed on the cluster 2026-09-24: t2kp cells after a post-fix resume
    ({"resumed_at": 2, "adapter_restored": True, "history": [1, 2, 3]},
     "include", "resumed AND restored -- the replacement data must COUNT"),
    # observed: prereg cells on their first chunk, post-fix
    ({"resumed_at": None, "adapter_restored": None, "history": [1, 2]},
     "include", "fresh post-fix run, never resumed"),
    # observed: prereg_q3_s1.severed-1532
    ({"resumed_at": 3, "history": [1, 2, 3, 4]},
     "exclude (pre-fix artifact)", "pre-fix artifact, key absent entirely"),
    # the stale-snapshot case that defeated the first rule
    ({"resumed_at": None, "history": [1, 2]},
     "exclude (pre-fix artifact)", "pre-fix, snapshotted before its first timeout"),
    # a post-fix resume whose restore failed
    ({"resumed_at": 4, "adapter_restored": False, "history": [1, 2, 3, 4]},
     "exclude (severed)", "post-fix resume that did NOT restore"),
    # observed: prereg_q15_s1, self arm trained on 1 example across 5 rounds
    ({"resumed_at": None, "adapter_restored": None,
      "history": [{"n_ex": 0}, {"n_ex": 1}, {"n_ex": 0}, {"n_ex": 0}, {"n_ex": 0}]},
     "exclude (starved)", "Amendment 6: mean n_ex 0.20, below the floor of 2"),
    # observed: prereg_q3_s2, self arm never trained at all
    ({"resumed_at": None, "adapter_restored": None, "history": [{"n_ex": 0}, {"n_ex": 0}]},
     "exclude (starved)", "Amendment 6: mean n_ex 0.00"),
    # a fed-style cell with a healthy self arm must still COUNT
    ({"resumed_at": None, "adapter_restored": None,
      "history": [{"n_ex": 9}, {"n_ex": 11}, {"n_ex": 10}]},
     "include", "healthy n_ex -- the replacement data must count"),
    # right at the boundary, which is where an off-by-one would hide
    ({"resumed_at": None, "adapter_restored": None, "history": [{"n_ex": 2}, {"n_ex": 2}]},
     "include", "mean n_ex exactly 2.0 is not starved"),
]


def main():
    bad = []
    for payload, want, why in CASES:
        got = verdict(payload)
        flag = "ok " if got == want else "FAIL"
        if got != want:
            bad.append((why, want, got))
        print("  %s %-46s -> %s" % (flag, why, got))
    assert not bad, "\n".join("%s: wanted %r got %r" % b for b in bad)

    # The rule must agree with the generator's actual source, not a copy that drifted.
    src = open(os.path.join(ROOT, "scripts", "make_kernel_results_tex.py")).read()
    assert '"adapter_restored" not in d' in src, \
        "generator no longer keys on key PRESENCE -- this test is mirroring a stale rule"
    assert 'd.get("adapter_restored") is not True' in src, \
        "generator no longer keys on `is not True` -- truthiness would reject a valid None"
    assert "_mean_nex < 2.0" in src, \
        "generator no longer enforces the Amendment 6 starvation floor"
    print("\nPASS -- excludes severed and pre-fix data, includes resumed-and-restored cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
