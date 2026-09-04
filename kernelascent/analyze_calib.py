"""Walk graded results.json across calibration model dirs on ONE node and emit a compact
JSON: per-model per-tier correctness/speed + speedups, and a per-candidate failure-reason
histogram. Run on each node (no shared FS), then merge the per-node JSONs off-box.
Usage: python3 analyze_calib.py <out.json> <datadir> [<datadir> ...]
"""
import glob, json, sys, os, collections

out = sys.argv[1]
datadirs = sys.argv[2:]
res = {}
for dd in datadirs:
    for md in glob.glob(dd + "/*/"):
        model = os.path.basename(md.rstrip("/"))
        rjs = glob.glob(md + "*/results.json")
        if not rjs:
            continue
        tiers = collections.defaultdict(lambda: {"n": 0, "corr": 0, "fast": 0, "sp": []})
        reasons = collections.Counter()
        for rj in rjs:
            try:
                r = json.load(open(rj))
            except Exception:
                continue
            tier = r.get("tier") or "?"
            t = tiers[tier]; t["n"] += 1
            correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
            sp = r.get("best_speedup_roofline", 0) or 0
            if correct:
                t["corr"] += 1; t["sp"].append(sp)
            if sp > 1.0:
                t["fast"] += 1
            cands = r.get("candidates") or []
            if not cands and r.get("crash_reason"):
                reasons["native_crash"] += 1
            for c in cands:
                reasons[c.get("reason", "?")] += 1
        res[model] = {"tiers": {k: dict(v) for k, v in tiers.items()}, "reasons": dict(reasons)}
json.dump(res, open(out, "w"))
print("wrote %s: %d models" % (out, len(res)))
