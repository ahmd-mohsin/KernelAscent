#!/usr/bin/env python3
"""Do the local artifacts that build the paper still match the cluster?

`mrl pull` uses rsync WITHOUT --delete, so a file renamed or removed on the cluster leaves an
orphan behind locally, and a file pulled mid-run leaves a stale snapshot. Both happened on
2026-09-24 and both nearly put wrong numbers into a document:

  * `prereg_*` dirs renamed to `prereg_*.severed-1532` on the cluster left orphans holding
    rounds=1 and rounds=3 snapshots, which the .tex generator read as valid trajectories;
  * `bank_dsl_kernel.json` was pulled mid-chunk and read 112 entries against the finished
    124, and that number was about to go into the datasheet;
  * `t1kl_q14.json` was pulled before its cell completed, and the generator correctly refused
    to report a replication table from it.

The generators re-derive every number from artifacts, which is the right design -- but it only
helps if the artifacts are the current ones. Compares a content hash, not mtime or size, since
a truncated or partially-written file can share both.

    python3 scripts/check_artifact_freshness.py            # report
    python3 scripts/check_artifact_freshness.py --strict   # exit 1 on any mismatch
"""
import argparse, glob, hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "/users/muahmed/ka_data"
LOCAL_DIRS = [os.path.join(ROOT, "data", "trajectories"),
              os.path.join(ROOT, "data", "marlowe_h100")]


def digest(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def local_artifacts():
    """Top-level .json artifacts the tex generators may read, nearest directory wins."""
    seen = {}
    for d in LOCAL_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.basename(p)
            if name not in seen:
                seen[name] = p
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    local = local_artifacts()
    if not local:
        print("no local artifacts found -- run `mrl pull`")
        return 1

    names = sorted(local)
    script = (
        "python3 - <<'PYEOF'\n"
        "import json, hashlib, os\n"
        "names = %r\n"
        "for n in names:\n"
        "    p = os.path.join(%r, n)\n"
        "    try:\n"
        "        d = json.load(open(p))\n"
        "        print('%%s|%%s' %% (n, hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]))\n"
        "    except FileNotFoundError:\n"
        "        print('%%s|absent' %% n)\n"
        "    except Exception:\n"
        "        print('%%s|unreadable' %% n)\n"
        "PYEOF" % (names, REMOTE))
    try:
        out = subprocess.run(["mrl", "run", script], capture_output=True, text=True, timeout=180).stdout
    except Exception as e:
        print("could not query the cluster (%r) -- cannot verify freshness" % e)
        return 1

    remote = {}
    for line in out.splitlines():
        if "|" in line and not line.startswith("["):
            k, _, v = line.strip().partition("|")
            remote[k] = v

    stale, orphan, ok = [], [], 0
    for n in names:
        try:
            ld = digest(json.load(open(local[n])))
        except Exception:
            ld = "unreadable"
        rd = remote.get(n)
        if rd is None:
            continue                      # cluster was not asked / no answer
        if rd == "absent":
            orphan.append(n)              # exists locally, gone on the cluster
        elif rd != ld:
            stale.append((n, ld, rd))
        else:
            ok += 1

    for n, ld, rd in stale:
        print("  STALE   %-40s local %s  cluster %s" % (n, ld, rd))
    for n in orphan:
        print("  ORPHAN  %-40s absent on the cluster (renamed or removed there; rsync never deletes)" % n)
    print("\n  %d artifact(s) match, %d stale, %d orphaned" % (ok, len(stale), len(orphan)))
    if stale or orphan:
        print("  `mrl pull` refreshes stale copies; an ORPHAN must be removed or archived by hand,")
        print("  because a pull will never delete it and the generators will keep reading it.")
        return 1 if a.strict else 0
    print("  every local artifact matches the cluster -- the .tex can be trusted to be current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
