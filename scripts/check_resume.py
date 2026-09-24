#!/usr/bin/env python3
"""Does every long-running lab actually resume, or does `jobman continue` loop it forever?

Two labs were silently looping: lab_weight_rsi (the registered primary) and lab_compounding
(19 cells). Both wrote results per round and restarted at round 0 on every walltime timeout,
and the watchdog faithfully resubmitted them to do it again -- the queue showed steady activity
and the experiment made no progress.

A real resume must do BOTH: read prior state, and skip the work it covers. Writing state is not
resuming, and that distinction is exactly what a casual grep gets wrong -- an earlier version of
this check called difficulty_filter "writes only" because its skip test is
`if t["name"] in done:` and the pattern expected a bare identifier.

    python3 scripts/check_resume.py          # report
    python3 scripts/check_resume.py --strict # exit 1 if any lab cannot resume
"""
import argparse, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# lab -> the jobman cell prefix that runs it, for a message that names real jobs
LABS = [
    ("kernelascent/v3/lab_compounding.py",   "t2k-*"),
    ("kernelascent/v3/lab_weight_rsi.py",    "prereg-*"),
    ("kernelascent/v3/lab_track_c.py",       "t3k-*"),
    ("kernelascent/v3/lab_selfplay_rsi.py",  "t5k-*"),
    ("kernelascent/v3/lab_baselines.py",     "basek-*"),
    ("kernelascent/v3/difficulty_filter.py", "bank-*"),
    # mechk-* completed inside a single 50-minute chunk, so auto-continuation never fires for
    # it. Exempt rather than "fixed": the check should flag labs that CAN loop, and a lab whose
    # runs fit their walltime cannot. If mechk is ever given more rounds, remove this.
    ("kernelascent/v3/lab_rsi_mechanism.py", "mechk-* (fits one chunk; exempt)"),
]

READS = [r"json\.load\(open\(", r"\bresume\b"]
# any loop or guard that begins past zero / skips completed items
SKIPS = [r"for\s+\w+\s+in\s+range\(\s*len\(", r"for\s+\w+\s+in\s+range\(\s*start",
         r"\bin\s+done\b", r"\bstart\s*=\s*\w+\[", r"continue\s*#.*already"]


def check(path):
    s = open(os.path.join(ROOT, path)).read()
    reads = any(re.search(p, s) for p in READS)
    skips = any(re.search(p, s) for p in SKIPS)
    writes = "json.dump" in s or "PROV.dump" in s
    return reads, skips, writes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    bad = []
    print("%-40s %-10s %s" % ("lab", "cells", "resume"))
    for path, cells in LABS:
        if not os.path.exists(os.path.join(ROOT, path)):
            continue
        reads, skips, writes = check(path)
        if "exempt" in cells:
            v = "n/a -- completes within one walltime chunk"
        elif reads and skips:
            v = "yes"
        elif writes and "exempt" not in cells:
            v = "NO -- writes per round but restarts at 0"
            bad.append((path, cells))
        else:
            v = "n/a (single-shot)"
        print("%-40s %-10s %s" % (os.path.basename(path), cells, v))
    if bad:
        print("\n  %d lab(s) cannot resume. `jobman continue` will loop these forever:" % len(bad))
        for path, cells in bad:
            print("     %-34s runs %s" % (os.path.basename(path), cells))
        print("  Either add a resume, or give those cells a walltime that fits a whole run.")
        return 1 if a.strict else 0
    print("\n  every long-running lab resumes; auto-continuation is safe for all of them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
