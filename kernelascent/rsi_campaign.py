"""RSI-Depth Phase 0 exit experiment: the fixed reward loop.

Same execution-feedback loop as openweight_rsi.py but with the two Phase 0 fixes:
  1. keep-best per task (never regress off a correct kernel), so "make it faster" cannot
     destroy a working solution the way the v0 sweep did.
  2. log-interpolated score against the eager->expert rungs (scoring.py), so incremental
     speedups earn rising credit (a slope, not a cliff), and feedback references the
     concrete gap to the expert time.

Reports per round the mean fresh-attempt score and the mean best-so-far score. best-so-far
is monotone by construction (keep-best), so the decline is structurally gone; the real
question this answers is whether the shaped feedback makes fresh attempts trend UP (slope).
Grading is delegated to the crash-isolated grade_candidates.py --one.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u rsi_campaign.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --expert-times /path/expert_times.json --n 10 --rounds 5 --outdir /path/out
"""
import os, sys, json, argparse, subprocess, glob
import torch
import gen_source_tasks as G
import curate_bedrock as CB
from scoring import log_interp_score, keep_best, milestones

HERE = os.path.dirname(os.path.abspath(__file__))
GRADER = os.path.join(HERE, "grade_candidates.py")

BASE = """Optimize this PyTorch module for speed on an NVIDIA A100 GPU. Keep __init__ identical; only rewrite forward. Use Triton or fused PyTorch ops, numerically equivalent output.
Output exactly ONE class named ModelNew in a single ```python code block. No prose.

Reference module:
```python
{src}
```"""

FIX = """Your previous ModelNew failed. The grader reported:

    {reason}

Previous attempt:
```python
{prev}
```
Produce a corrected ModelNew (single ```python block, class ModelNew, numerically equivalent). Fix the specific failure."""

SHAPE = """Your best correct ModelNew so far scores {score:.2f} out of 1.0 on the speed ladder (0 = eager baseline, 1 = the expert kernel). Its runtime is {t_cand_ms:.3f} ms; the expert reference runs in {t_expert_ms:.3f} ms; torch.compile runs in {t_compile_ms:.3f} ms. You are {gap} the compiler.

Your best correct attempt:
```python
{prev}
```
Rewrite ModelNew to close the gap to the expert time: fuse into one kernel, tile and choose block sizes for the A100, use tl.autotune / num_warps, cut memory traffic. Keep it numerically equivalent. Output one ```python block."""


def gen_one(model, tok, prompt, max_new):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                             top_p=0.9, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def grade_one(taskdir, cand_timeout=90):
    rj = os.path.join(taskdir, "results.json")
    if os.path.exists(rj):
        os.remove(rj)
    try:
        subprocess.run([sys.executable, "-u", GRADER, "--candir", taskdir, "--one", taskdir,
                        "--cand-timeout", str(cand_timeout)], timeout=cand_timeout * 3 + 60)
    except subprocess.TimeoutExpired:
        pass
    try:
        return json.load(open(rj))
    except Exception:
        return {"correct": False, "best": None, "t_eager": None, "t_compile": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expert-times", default="", help="json {task_name: t_expert_seconds}")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tier", default="Medium")
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.outdir, exist_ok=True)
    expert = json.load(open(args.expert_times)) if args.expert_times and os.path.exists(args.expert_times) else {}
    tasks = G.generate_tiered(args.tier, args.n, seed0=args.seed0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("RSI-campaign model=%s tier=%s tasks=%d rounds=%d experts=%d gpu=%s" %
          (args.model, args.tier, len(tasks), args.rounds, len(expert), os.environ.get("CUDA_VISIBLE_DEVICES", "?")), flush=True)

    fresh = [0.0] * args.rounds; best = [0.0] * args.rounds
    n = len(tasks)
    for ti, t in enumerate(tasks):
        d = os.path.join(args.outdir, t["name"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(t["source"])
        json.dump({k: t[k] for k in ("name", "tier", "family", "meta") if k in t}, open(d + "/meta.json", "w"))
        incumbent = None
        for rnd in range(args.rounds):
            if rnd == 0:
                prompt = BASE.format(src=t["source"])
            elif incumbent is None:
                r0 = json.load(open(d + "/results.json")) if os.path.exists(d + "/results.json") else {}
                reason = ((r0.get("best") or {}).get("reason")) or "no valid ModelNew"
                prompt = FIX.format(reason=str(reason)[:280], prev=(incumbent or {}).get("code", "(none)"))
            else:
                te = incumbent["t_eager"]; tx = expert.get(t["name"]) or incumbent.get("t_compile") or te
                prompt = SHAPE.format(score=incumbent["score"], t_cand_ms=incumbent["t_cand"] * 1e3,
                                      t_expert_ms=(tx or 0) * 1e3, t_compile_ms=(incumbent.get("t_compile") or 0) * 1e3,
                                      gap=("faster than" if incumbent.get("t_compile") and incumbent["t_cand"] < incumbent["t_compile"] else "slower than"),
                                      prev=incumbent["code"])
            raw = gen_one(model, tok, prompt, args.max_new)
            code = CB.extract_modelnew(raw)
            for old in glob.glob(d + "/cand_*.py"):
                os.remove(old)
            score = 0.0
            if code:
                open(d + "/cand_0.py", "w").write(code)
                r = grade_one(d)
                correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
                b = r.get("best") or {}
                t_cand = b.get("t_cand"); t_eager = r.get("t_eager"); t_compile = r.get("t_compile")
                t_exp = expert.get(t["name"]) or t_compile or t_eager
                score = log_interp_score(t_cand, t_eager, t_exp, correct) if (correct and t_cand) else 0.0
                if correct and t_cand:
                    cand = {"correct": True, "t_cand": t_cand, "code": code, "score": score,
                            "t_eager": t_eager, "t_compile": t_compile}
                    incumbent = keep_best(incumbent, cand)
            fresh[rnd] += score
            best[rnd] += incumbent["score"] if incumbent else 0.0
            print("  task %2d/%d r%d %-26s fresh=%.2f best=%.2f" %
                  (ti + 1, n, rnd, t["name"][:26], score, incumbent["score"] if incumbent else 0.0), flush=True)

    print("\n=== RSI-campaign trajectory (%s, %s, n=%d) ===" % (args.model, args.tier, n), flush=True)
    print("round  mean_fresh_score  mean_best_score", flush=True)
    for r in range(args.rounds):
        print("  %d      %.3f             %.3f" % (r, fresh[r] / n, best[r] / n), flush=True)
    json.dump({"model": args.model, "tier": args.tier, "n": n, "rounds": args.rounds,
               "mean_fresh": [x / n for x in fresh], "mean_best": [x / n for x in best]},
              open(os.path.join(args.outdir, "campaign_trajectory.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
