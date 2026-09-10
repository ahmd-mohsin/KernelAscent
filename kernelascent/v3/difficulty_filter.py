"""Difficulty filter — standardize the RSI bank so model failure reflects PROBLEM difficulty, not a
saturated/trivial benchmark.

For each task in the input bank we run the FROZEN base model k times, grade each candidate on GPU
(crash-isolated), and record:
  correct_rate = fraction of k candidates that are correct (vs fp32 gold)
  best_score   = best-of-k speed-resolved score C (0 wrong, 0.5 at parity, 1.0 at >=1.5x)
We admit a task iff it is HARD-BUT-LEARNABLE for the frozen base:
  keep  <=>  correct_rate > 0            (learnable: the base CAN sometimes solve it, so a positive SFT
                                           example exists to bootstrap from)
        AND  best_score  <= keep_hi      (headroom: the base is NOT already fast; default 0.75)
Optionally also require correct_rate <= learn_hi to drop tasks the base nails every time.

Result: C0 on the kept set is deliberately LOW with real room to climb, so a flat RSI result is
attributable to the MODEL, not the bank. Writes the filtered bank + a per-task difficulty report.

Run (sharded 7B):
  KA_GRADE_GPU=4 python3 difficulty_filter.py --model Qwen/Qwen2.5-Coder-7B-Instruct --gpus 0,1 \
      --in kernel_bank/hard_all.json --out kernel_bank/hard_filtered.json --k 8
"""
import os, sys, json, argparse, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT); sys.path.insert(0, PKG); sys.path.insert(0, HERE)
import torch
from kernelascent import agent_bench as AB
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--gpus", default="0,1", help="GPUs to shard the frozen base across")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--k", type=int, default=8, help="frozen-base samples per task for the difficulty estimate")
    ap.add_argument("--keep-hi", type=float, default=0.75, help="drop tasks the base already scores above this (too easy/fast)")
    ap.add_argument("--learn-hi", type=float, default=1.01, help="drop tasks whose base correct_rate exceeds this (default off)")
    args = ap.parse_args()

    bank = json.load(open(args.inp))
    gpus = [int(x) for x in str(args.gpus).split(",")]
    tok, mdl = W.build(args.model, gpus)                 # adapter present but we generate with adapter=False = frozen base
    report = []; kept = []
    print("DIFFICULTY-FILTER %s  in=%d tasks  k=%d  keep if 0<correct_rate and best<=%.2f"
          % (args.model, len(bank), args.k, args.keep_hi), flush=True)
    for t in bank:
        src = t["source"]
        outs = W.generate(tok, mdl, src, args.k, adapter=False)
        codes = [c for c in (AB.extract_modelnew(o) for o in outs) if c]
        res = W._grade_isolated(src, codes)                          # [ok, sp_eager, sp_compiled] per candidate
        n_correct = sum(1 for g in res if g and g[0])
        best = max([LK._score(g[0], g[1]) for g in res] + [0.0])     # difficulty on the eager ratio
        cr = n_correct / max(1, args.k)
        keep = (cr > 0.0) and (best <= args.keep_hi) and (cr <= args.learn_hi)
        row = {"name": t["name"], "tier": t["tier"], "correct_rate": round(cr, 3), "best_score": round(best, 3), "keep": keep}
        report.append(row)
        if keep:
            kept.append(t)
        torch.cuda.empty_cache()
        print("  %-8s %-3s correct_rate=%.2f best=%.2f -> %s" % (t["name"], t["tier"], cr, best, "KEEP" if keep else "drop"), flush=True)
        json.dump(kept, open(args.out, "w"), indent=2)
        json.dump(report, open(args.report or (args.out + ".report.json"), "w"), indent=2)
    by = {tt: sum(1 for x in kept if x["tier"] == tt) for tt in ("L1", "L2", "L3")}
    c0_proxy = statistics.mean([r["best_score"] for r in report if r["keep"]]) if kept else 0.0
    print("\n=== FILTER SUMMARY === kept %d/%d %s ; frozen-base mean best on kept = %.3f (this is the expected C0 floor)"
          % (len(kept), len(bank), by, c0_proxy), flush=True)


if __name__ == "__main__":
    main()
