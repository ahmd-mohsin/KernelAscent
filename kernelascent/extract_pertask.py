"""Per-node, per-TASK rollup across all model dirs: for each procedural task, how many
model-attempts were correct and the best speedup-vs-roofline observed. Merge across nodes
(sum correct counts, max speedup) to get an empirical difficulty label per task.
Usage: python3 extract_pertask.py <out.json> <datadir> [<datadir> ...]
"""
import glob, json, sys, os, collections

out = sys.argv[1]
datadirs = sys.argv[2:]
agg = {}
for dd in datadirs:
    for rj in glob.glob(dd + "/*/*/results.json"):
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        name = r.get("name") or os.path.basename(os.path.dirname(rj))
        a = agg.setdefault(name, {"tier": r.get("tier"), "family": r.get("family"),
                                  "attempts": 0, "correct": 0, "best_sp": 0.0})
        a["attempts"] += 1
        if bool(r.get("correct")) or r.get("pass_at_k", 0) > 0:
            a["correct"] += 1
        a["best_sp"] = max(a["best_sp"], r.get("best_speedup_roofline", 0.0) or 0.0)
json.dump(agg, open(out, "w"))
print("wrote %s: %d tasks" % (out, len(agg)))
