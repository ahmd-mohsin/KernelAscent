#!/usr/bin/env python3
"""P3 part 2 — LEARNABILITY GATE for the DSL task bank (runs on a GPU node). Two filters:
  VALID  : agent_bench.build_ref(source) succeeds and ref-vs-fp32 rel-err is finite/small (task is well-posed).
  LEARNABLE: the base model's best-of-K correct-rate is in (lo, hi) — not trivial (>=hi) nor impossible (<=lo).
Writes a filtered bank (kernel_tasks_dsl_gated.json). Point KA_KERNEL_BANK at it for probe/self-play runs so
per-task statistics are stable and the self-play curriculum has a real difficulty band.
Usage: KA_GRADE_GPU=0 python3 scripts/learnability_gate.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
       --bank dataset/kernel_bank/kernel_tasks_dsl.json --K 8 --lo 0.1 --hi 0.9 --gpu 0 --out <path>
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from kernelascent import agent_bench as AB


def run(args):
    import torch
    bank = json.load(open(args.bank))
    # 1) VALID: build_ref works + finite err
    valid = []
    for b in bank:
        try:
            _, _, _, rerr = AB.build_ref(b["source"])
            if rerr == rerr and rerr < 0.5:   # finite (not NaN) and reasonable
                valid.append(b)
        except Exception:
            pass
        torch.cuda.empty_cache()
    print("VALID %d/%d tasks build a well-posed reference" % (len(valid), len(bank)), flush=True)
    # 2) LEARNABLE: base best-of-K solve rate in (lo,hi). Reuse lab_weight_rsi build+generate+grade.
    from kernelascent.v3 import lab_weight_rsi as W
    tok, mdl = W.build(args.model, gpus=tuple(int(x) for x in str(args.gpu).split(","))); mdl.eval()
    kept = []
    for b in valid:
        # register this single task in the live TASKS so grader/gen can use it
        from kernelascent.v3 import lab_kernel as LK
        LK.TASKS[b["name"]] = b["source"]
        try:
            C, _, _, _, st = W.eval_tasks(tok, mdl, [b["name"]], args.K, adapter=False)
            rate = st.get("correct_rate", 0.0)
        except Exception:
            rate = 0.0
        b["base_solve_rate"] = round(rate, 3)
        if args.lo <= rate <= args.hi: kept.append(b)
        print("  %s base_solve=%.2f %s" % (b["name"], rate, "KEEP" if args.lo <= rate <= args.hi else "drop"), flush=True)
    out = args.out or args.bank.replace(".json", "_gated.json")
    json.dump(kept, open(out, "w"), indent=2)
    print("LEARNABLE %d/%d in band [%.2f,%.2f] -> %s" % (len(kept), len(valid), args.lo, args.hi, out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--bank", default="dataset/kernel_bank/kernel_tasks_dsl.json")
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--lo", type=float, default=0.1); ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--gpu", default="0"); ap.add_argument("--out", default="")
    run(ap.parse_args())
