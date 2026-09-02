"""Box-side grader for curated candidates (runs on the GPU box).

Reads the candidate directory produced by curate_bedrock.py (task.py + cand_*.py per
task), grades every candidate against the fp32 gold on GPU, and records best-of-k
speedup vs the eager baseline. Supports sharding across GPUs (--nshards/--shard).
Reuses the validated harness in agent_bench.
"""
import os, glob, json, argparse, torch
import agent_bench as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    tdirs = sorted(d for d in glob.glob(args.candir + "/*") if os.path.exists(os.path.join(d, "task.py")))
    tdirs = [d for i, d in enumerate(tdirs) if i % args.nshards == args.shard]
    records = []
    for d in tdirs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        src = open(os.path.join(d, "task.py")).read()
        try:
            ref, x, gold, ref_err = A.build_ref(src)
        except Exception as e:
            print("SKIP %s ref-build: %s" % (meta["name"], repr(e)[:60])); continue
        bound = max(args.tol, args.margin * ref_err)
        tbase = A.time_fn(lambda z: ref(z), (x,))
        best_sp, best_err, n_ok, n_cand = 0.0, float("inf"), 0, 0
        for cf in sorted(glob.glob(d + "/cand_*.py")):
            n_cand += 1
            code = open(cf).read()
            ok, err, sp, _ = A.grade(src, code, ref, x, gold, bound, tbase)
            if ok:
                n_ok += 1
                if sp > best_sp:
                    best_sp, best_err = sp, err
        records.append(dict(name=meta["name"], tier=meta["tier"], family=meta["family"],
                            tags=meta.get("tags", []), meta=meta["meta"],
                            n_cand=n_cand, pass_at_k=n_ok, best_speedup=best_sp,
                            best_err=(best_err if best_err != float("inf") else None)))
        print("%-22s %-10s %-3s cands=%d pass=%d best_sp=%.3f" %
              (meta["name"], meta["family"], meta["tier"], n_cand, n_ok, best_sp), flush=True)
    json.dump({"tasks": records}, open(args.out, "w"), indent=2)
    print("wrote %s (%d tasks)" % (args.out, len(records)))


if __name__ == "__main__":
    main()
