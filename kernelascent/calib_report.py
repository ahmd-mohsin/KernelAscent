"""Read distributed-calibration summaries and report per-model x per-tier
correctness_rate and speed_rate. A tier is well-calibrated for a model band when
correctness is passable (channel open) and speed is non-trivial but not saturated.
"""
import os, glob, json, argparse, collections

TIERS = ["Easy", "Medium", "Hard", "Ultra"]


def load(datadir):
    rows = {}
    for sp in glob.glob(datadir + "/*/summary.json"):
        model = os.path.basename(os.path.dirname(sp))
        tasks = json.load(open(sp)).get("tasks", [])
        rows[model] = tasks
    return rows


def per_tier(tasks):
    by = collections.defaultdict(lambda: {"n": 0, "correct": 0, "fast": 0})
    for t in tasks:
        tier = t.get("tier") or "untiered"
        b = by[tier]; b["n"] += 1
        if t.get("pass_at_k", 0) > 0:
            b["correct"] += 1
        if t.get("best_speedup_roofline", 0) > 1.0:
            b["fast"] += 1
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="/tmp/instance_storage/ka_data/dist_calib")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = load(args.datadir)
    out = {"models": {}}
    print("%-38s %-8s %6s %6s %6s" % ("model", "tier", "n", "corr", "fast"))
    for model in sorted(rows):
        by = per_tier(rows[model])
        out["models"][model] = {}
        for tier in TIERS + [k for k in by if k not in TIERS]:
            if tier not in by:
                continue
            b = by[tier]; c = b["correct"] / b["n"]; f = b["fast"] / b["n"]
            out["models"][model][tier] = {"n": b["n"], "correctness_rate": round(c, 3), "speed_rate": round(f, 3)}
            print("%-38s %-8s %6d %5.0f%% %5.0f%%" % (model, tier, b["n"], 100 * c, 100 * f))
        print("-" * 68)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
