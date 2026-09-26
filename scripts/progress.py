#!/usr/bin/env python3
"""One view of every running experiment, whichever lab wrote it.

Exists because the inline version got retyped on every check and got the filename wrong: the
three-arm cells run lab_compounding and write `compounding.json` with `lineage_minus_reset`,
while the weight-RSI cells write `weight_rsi.json` with `delta_self_minus_fresh`. A reader that
knows only one of those reports "none yet" for a cell that is running fine, which is the same
"cannot see its input" failure this repo keeps rediscovering.

Runs against a directory of cells, locally or on the cluster:

  python3 scripts/progress.py                             # every local cell
  python3 scripts/progress.py --only dose,deep,fed        # just the live program
  mrl run 'python3 scripts/progress.py ~/ka_data --only dose,deep'
"""
import json, os, sys, glob

# artifact -> (per-round contrast key, registered depth by cell-name prefix)
KINDS = [("weight_rsi.json", "delta_self_minus_fresh"),
         ("compounding.json", "lineage_minus_reset")]
DEPTH = {"deep": 15, "fedc": 8, "t2kc": 8, "t2kp": 8, "dose3": 5, "dose10": 5,
         "fed": 5, "fedp": 5, "prereg": 5, "kl005": 5, "kl02": 5, "kl10": 5}


def depth_of(cell):
    # longest prefix wins, so dose10 is not read as dose and fedc is not read as fed
    best = None
    for k, v in DEPTH.items():
        if cell.startswith(k) and (best is None or len(k) > len(best[0])):
            best = (k, v)
    return best[1] if best else None


def main(argv):
    only = []
    if "--only" in argv:
        i = argv.index("--only")
        only = [x for x in argv[i + 1].split(",") if x] if i + 1 < len(argv) else []
        argv = argv[:i] + argv[i + 2:]
    root = argv[0] if argv else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trajectories")
    root = os.path.expanduser(root)
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        cell = os.path.basename(d)
        if not os.path.isdir(d) or "." in cell:      # skip parked copies
            continue
        if only and not any(cell.startswith(x) for x in only):
            continue
        for fname, key in KINDS:
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                continue
            try:
                dd = json.load(open(p))
            except Exception:
                rows.append((cell, None, "(mid-write)", "", "")); break
            h = dd.get("history") or []
            if not h:
                break
            vals = [r[key] for r in h if r.get(key) is not None]
            nex = [r.get("n_ex") for r in h if r.get("n_ex") is not None]
            kl = [r.get("kl") for r in h if r.get("kl") is not None]
            rows.append((cell, len(h),
                         " ".join("%+.3f" % v for v in vals[-6:]),
                         (",".join(str(x) for x in nex[-6:]) if nex else ""),
                         ("kl~%.3f" % (sum(kl) / len(kl)) if kl else "")))
            break
    if not rows:
        print("no cell has written a round yet under %s" % root)
        return 0
    w = max(len(r[0]) for r in rows)
    print("%-*s  %-7s  %-42s  %-22s %s" % (w, "cell", "rounds", "contrast (last 6)", "n_ex", "kl"))
    print("-" * (w + 80))
    for cell, n, vals, nex, kl in sorted(rows):
        tgt = depth_of(cell)
        at = ("%d/%s" % (n, tgt if tgt else "?")) if n is not None else "-"
        done = " *" if (tgt and n and n >= tgt) else ""
        print("%-*s  %-7s  %-42s  %-22s %s%s" % (w, cell, at, vals, nex, kl, done))
    print("\n* = at registered depth")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
