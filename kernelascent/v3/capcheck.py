"""E0 validity gate: capability monotonicity. Run base `develop` on a FIXED Medium task set
(no lineage, no revise) and report mean attainment C in {0,0.5,1.0} plus correct-rate and
fast-rate. Running this across a small->large model ladder must yield rising C; if not, the
base metric/harness is broken and RSI claims are moot.
"""
import os, sys, json, argparse, subprocess, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import gen_source_tasks as G
import curate_bedrock as CB
GRADER = os.path.join(os.path.dirname(HERE), "grade_candidates.py")
FAST = 1.10
SOLVE = ("Optimize this PyTorch module for speed on an A100. Keep __init__ identical; rewrite forward "
         "using Triton or fused ops, numerically equivalent, aiming to beat torch.compile.\n"
         "Output exactly ONE class named ModelNew in a single ```python block. No prose.\n\n```python\n{SRC}\n```")


def grade_one(d, ct=90):
    rj = d + "/results.json"
    if os.path.exists(rj):
        os.remove(rj)
    try:
        subprocess.run([sys.executable, "-u", GRADER, "--candir", d, "--one", d, "--cand-timeout", str(ct)], timeout=ct * 3 + 60)
    except subprocess.TimeoutExpired:
        pass
    try:
        return json.load(open(rj))
    except Exception:
        return {"correct": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf"); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--n", type=int, default=15); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--k", type=int, default=1, help="samples per task; C averaged over samples to shrink variance")
    ap.add_argument("--max-new", type=int, default=3072); ap.add_argument("--outdir", required=True)
    args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)
    tasks = G.generate_tiered("Medium", args.n, seed0=args.seed0)
    if args.api_model:
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock")); cur.resolve(); cur.resolve_reasoning()
        gen = lambda p: cur.generate(p); who = "api:" + args.api_model
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model); mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        def gen(p):
            enc = tok([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)], return_tensors="pt", padding=True).to("cuda")
            import torch as _t
            with _t.no_grad():
                o = mdl.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
            return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        who = "hf:" + args.model
    def wilson(k_succ, n_tot):
        if n_tot == 0:
            return [0.0, 0.0]
        import math
        z = 1.96; p = k_succ / n_tot; d = 1 + z * z / n_tot
        c = (p + z * z / (2 * n_tot)) / d; h = z * math.sqrt(p * (1 - p) / n_tot + z * z / (4 * n_tot * n_tot)) / d
        return [round(c - h, 3), round(c + h, 3)]

    C = []; corr = 0; fast = 0; trials = 0   # C over ALL samples (k per task); rates over all trials
    for i, t in enumerate(tasks):
        d = os.path.join(args.outdir, t["name"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(t["source"])
        json.dump({kk: t[kk] for kk in ("name", "tier", "family", "meta") if kk in t}, open(d + "/meta.json", "w"))
        tc = []
        for j in range(args.k):
            code = CB.extract_modelnew(gen(SOLVE.replace("{SRC}", t["source"])) or "")
            for old in glob.glob(d + "/cand_*.py"):
                os.remove(old)
            c = 0.0; trials += 1
            if code:
                open(d + "/cand_0.py", "w").write(code); r = grade_one(d)
                ok = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
                sp = r.get("best_speedup_roofline", 0.0) or 0.0
                if ok:
                    corr += 1; c = 1.0 if sp >= FAST else 0.5
                    if sp >= FAST:
                        fast += 1
            tc.append(c); C.append(c)
        print("  %2d %-28s C=%.2f (k=%d)" % (i, t["name"][:28], sum(tc) / len(tc), args.k), flush=True)
    n = len(tasks)
    res = {"who": who, "n": n, "k": args.k, "trials": trials, "meanC": round(sum(C) / trials, 3),
           "correct_rate": round(corr / trials, 3), "correct_ci": wilson(corr, trials),
           "fast_rate": round(fast / trials, 3), "fast_ci": wilson(fast, trials)}
    json.dump(res, open(os.path.join(args.outdir, "capcheck.json"), "w"), indent=2)
    print("CAPCHECK %s meanC=%.3f correct=%.3f%s fast=%.3f%s" %
          (who, res["meanC"], res["correct_rate"], res["correct_ci"], res["fast_rate"], res["fast_ci"]), flush=True)


if __name__ == "__main__":
    main()
