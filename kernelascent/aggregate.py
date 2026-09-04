"""Aggregate parallel shard results into a curated benchmark summary.

Buckets tasks into families (matmul / softmax / norm / elementwise) from their op-chain
and reports the TWO walls separately -- correctness_rate (valid/correct kernel) vs
speed_rate/fast_1 (beats the min(eager,torch.compile) roofline) -- per family and per tier,
plus an explicit two-wall rollup so the correctness bottleneck (weak models) and the speed
bottleneck (frontier models) are both visible. Robust to missing/renamed fields.
"""
import json, glob, math, sys, argparse


def family(rec):
    labels = (rec.get("meta") or {}).get("chain") or []
    if any(l.startswith("matmul") for l in labels):
        return "matmul"
    if any(l == "softmax" for l in labels):
        return "softmax"
    if any(l in ("layernorm", "rmsnorm") for l in labels):
        return "norm"
    if rec.get("family"):                     # fall back to an explicit family field
        return rec["family"]
    return "elementwise"


def tier(rec):
    return rec.get("tier") or "untiered"


def speedup(rec):
    # summaries use best_speedup_roofline; older shards use best_speedup
    return rec.get("best_speedup_roofline", rec.get("best_speedup", 0)) or 0


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def stats(recs):
    """Both walls + fast_p + geomean for a bucket of task records."""
    n = len(recs)
    if n == 0:
        return None
    sps = [speedup(r) for r in recs]
    correct = sum(1 for r in recs if r.get("pass_at_k", 0) > 0)
    return dict(
        n=n,
        correctness_rate=round(correct / n, 3),        # WALL 1: valid/correct kernel
        speed_rate=round(sum(1 for s in sps if s > 1.0) / n, 3),  # WALL 2 (== fast_1)
        fast_1=round(sum(1 for s in sps if s > 1.0) / n, 3),
        fast_1_5=round(sum(1 for s in sps if s > 1.5) / n, 3),
        fast_2=round(sum(1 for s in sps if s > 2.0) / n, 3),
        geomean=round(geomean(sps), 3),
        # of the correct kernels, how many were also fast -> where correctness converts to speed
        fast_given_correct=round(sum(1 for r in recs if r.get("pass_at_k", 0) > 0 and speedup(r) > 1.0) / correct, 3) if correct else 0.0,
    )


def block(name, recs):
    s = stats(recs)
    if s is None:
        return
    print("  %-12s n=%3d  correct=%.2f  speed(fast_1)=%.2f fast_1.5=%.2f fast_2=%.2f  "
          "fast|correct=%.2f  geomean=%.3f"
          % (name, s["n"], s["correctness_rate"], s["speed_rate"], s["fast_1_5"],
             s["fast_2"], s["fast_given_correct"], s["geomean"]))


def rollups(recs):
    """Structured rollup: overall + by family + by tier + two-wall summary."""
    fams, tiers = {}, {}
    for r in recs:
        fams.setdefault(family(r), []).append(r)
        tiers.setdefault(tier(r), []).append(r)
    return dict(
        overall=stats(recs),
        by_family={k: stats(v) for k, v in fams.items()},
        by_tier={k: stats(v) for k, v in tiers.items()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    recs = []
    for f in sorted(glob.glob(args.outdir + "/shard_*.json")):
        recs += json.load(open(f))["tasks"]
    print("curated tasks graded: %d" % len(recs))
    print("== by family ==")
    fams = {}
    for r in recs:
        fams.setdefault(family(r), []).append(r)
    for k in sorted(fams):
        block(k, fams[k])
    print("== by tier (correctness wall vs speed wall) ==")
    tiers = {}
    for r in recs:
        tiers.setdefault(tier(r), []).append(r)
    for k in sorted(tiers):
        block(k, tiers[k])
    print("== overall ==")
    block("ALL", recs)
    ov = stats(recs)
    if ov:
        print("== two walls (overall) ==")
        print("  correctness wall: %.2f produce a correct kernel" % ov["correctness_rate"])
        print("  speed wall:       %.2f beat the roofline (%.2f of correct kernels)"
              % (ov["speed_rate"], ov["fast_given_correct"]))
    if args.save:
        for r in recs:
            r["family_bucket"] = family(r)
        json.dump({"tasks": recs, "rollups": rollups(recs)}, open(args.save, "w"), indent=2)
        print("wrote", args.save)


if __name__ == "__main__":
    main()
