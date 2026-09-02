"""Aggregate parallel shard results into a curated benchmark summary.

Buckets tasks into families (matmul / softmax / norm / elementwise) from their op-chain
and reports correctness, pass@k, and fast_p per family and per tier.
"""
import json, glob, math, sys, argparse


def family(rec):
    labels = rec["meta"]["chain"]
    if any(l.startswith("matmul") for l in labels):
        return "matmul"
    if any(l == "softmax" for l in labels):
        return "softmax"
    if any(l in ("layernorm", "rmsnorm") for l in labels):
        return "norm"
    return "elementwise"


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def block(name, recs):
    n = len(recs)
    if n == 0:
        return
    passk = sum(1 for r in recs if r["pass_at_k"] > 0) / n
    sps = [r["best_speedup"] for r in recs]
    f1 = sum(1 for s in sps if s > 1.0) / n
    f15 = sum(1 for s in sps if s > 1.5) / n
    f2 = sum(1 for s in sps if s > 2.0) / n
    print("  %-12s n=%3d  pass@k=%.2f  fast_1=%.2f fast_1.5=%.2f fast_2=%.2f  geomean(pass)=%.3f"
          % (name, n, passk, f1, f15, f2, geomean(sps)))


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
    print("== by tier ==")
    tiers = {}
    for r in recs:
        tiers.setdefault(r["tier"], []).append(r)
    for k in sorted(tiers):
        block(k, tiers[k])
    print("== overall ==")
    block("ALL", recs)
    if args.save:
        for r in recs:
            r["family_bucket"] = family(r)
        json.dump({"tasks": recs}, open(args.save, "w"), indent=2)
        print("wrote", args.save)


if __name__ == "__main__":
    main()
