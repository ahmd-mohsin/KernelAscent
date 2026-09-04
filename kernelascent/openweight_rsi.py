"""Open-weight RSI loop (execution-feedback / in-context self-improvement).

One open-weight model, held on a single GPU, tries to optimize each task and then
REVISES across rounds using the grader's own feedback: the compile/runtime error
if it failed, or the measured speedup vs the min(eager, torch.compile) roofline if
it passed ("correct but 0.8x, make it faster"). This is the open-weight analogue
of scaffold-RSI: no weight update, the model improves its output from graded
feedback. We report, per round, the fraction correct and the fraction beating the
roofline, plus the best-so-far, so you can see whether the loop climbs.

Grading is delegated to grade_candidates.py --one in a SUBPROCESS, so a native
Triton/MLIR abort on a bad kernel loses only that grade, never the loop or the
loaded model.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u openweight_rsi.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --tier Medium --n 15 --rounds 4 --outdir /tmp/.../rsi/coder7b
"""
import os, sys, json, argparse, subprocess, glob, re
import torch
import gen_source_tasks as G
import curate_bedrock as CB

HERE = os.path.dirname(os.path.abspath(__file__))
GRADER = os.path.join(HERE, "grade_candidates.py")

BASE = """Optimize this PyTorch module for speed on an NVIDIA A100 GPU. Keep __init__ identical; only rewrite forward. Use Triton or fused PyTorch ops. The output must be numerically equivalent to the reference.

Output exactly ONE class named ModelNew in a single ```python code block. No prose.

Reference module:
```python
{src}
```"""

FIX = """Your previous ModelNew for this task did not work. The grader reported:

    {reason}

Here is your previous attempt:
```python
{prev}
```

Produce a corrected ModelNew (single ```python block, class named ModelNew, numerically equivalent). Fix the specific failure above."""

FASTER = """Your previous ModelNew for this task is CORRECT but only {sp:.2f}x vs the baseline (min of eager and torch.compile), so it is not yet faster than the compiler. Here is your previous attempt:
```python
{prev}
```

Rewrite ModelNew to be genuinely faster: fuse operations into one kernel, tile and pick block sizes for the A100, use tl.autotune / num_warps, avoid extra memory traffic. Keep it numerically equivalent. Output one ```python block."""


def gen_one(model, tok, prompt, max_new):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                             top_p=0.9, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def grade_one(taskdir, cand_timeout=60):
    """Grade the single cand_0.py in taskdir via the crash-isolated grader; return results dict."""
    rj = os.path.join(taskdir, "results.json")
    if os.path.exists(rj):
        os.remove(rj)
    try:
        subprocess.run([sys.executable, "-u", GRADER, "--candir", taskdir, "--one", taskdir,
                        "--cand-timeout", str(cand_timeout)], timeout=cand_timeout * 3 + 60)
    except subprocess.TimeoutExpired:
        pass
    if os.path.exists(rj):
        try:
            return json.load(open(rj))
        except Exception:
            pass
    return {"correct": False, "best_speedup_roofline": 0.0, "best": None, "pass_at_k": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", default="Medium")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.outdir, exist_ok=True)
    tasks = G.generate_tiered(args.tier, args.n, seed0=args.seed0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("RSI model=%s tier=%s tasks=%d rounds=%d gpu=%s" %
          (args.model, args.tier, len(tasks), args.rounds, os.environ.get("CUDA_VISIBLE_DEVICES", "?")), flush=True)

    # per (task, round) correctness + speedup
    round_correct = [0] * args.rounds
    round_fast = [0] * args.rounds
    round_sp_sum = [0.0] * args.rounds
    best_correct = [0] * args.rounds     # cumulative best-so-far correctness by end of round r
    best_fast = [0] * args.rounds

    for ti, t in enumerate(tasks):
        d = os.path.join(args.outdir, t["name"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(t["source"])
        json.dump({k: t[k] for k in ("name", "tier", "family", "meta") if k in t}, open(d + "/meta.json", "w"))
        prev_code, prev_reason, prev_sp = None, None, 0.0
        seen_correct, seen_fast = False, False
        for rnd in range(args.rounds):
            if rnd == 0:
                prompt = BASE.format(src=t["source"])
            elif prev_code is None or (prev_reason and prev_reason != "ok"):
                prompt = FIX.format(reason=(prev_reason or "no valid ModelNew produced")[:300], prev=prev_code or "(none)")
            else:
                prompt = FASTER.format(sp=prev_sp, prev=prev_code)
            raw = gen_one(model, tok, prompt, args.max_new)
            code = CB.extract_modelnew(raw)
            # write this round's single candidate and grade it in isolation
            for old in glob.glob(d + "/cand_*.py"):
                os.remove(old)
            correct, sp, reason = False, 0.0, "no_candidate"
            if code:
                open(d + "/cand_0.py", "w").write(code)
                r = grade_one(d)
                correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
                sp = r.get("best_speedup_roofline", 0.0) or 0.0
                reason = ((r.get("best") or {}).get("reason")) or ("ok" if correct else "wrong_or_failed")
            if correct:
                round_correct[rnd] += 1; round_sp_sum[rnd] += sp
                if sp > 1.0:
                    round_fast[rnd] += 1
                seen_correct = True
                if sp > 1.0:
                    seen_fast = True
            if seen_correct:
                best_correct[rnd] += 1
            if seen_fast:
                best_fast[rnd] += 1
            prev_code, prev_reason, prev_sp = (code, reason, sp)
            print("  task %2d/%d r%d %-30s correct=%s sp=%.2f reason=%s" %
                  (ti + 1, len(tasks), rnd, t["name"][:30], correct, sp, str(reason)[:24]), flush=True)

    n = len(tasks)
    print("\n=== RSI trajectory (%s, %s tier, n=%d) ===" % (args.model, args.tier, n), flush=True)
    print("round  attempt_correct%  attempt_fast%  mean_sp(correct)  bestsofar_correct%  bestsofar_fast%", flush=True)
    for r in range(args.rounds):
        msp = round_sp_sum[r] / round_correct[r] if round_correct[r] else 0.0
        print("  %d      %5.0f%%           %5.0f%%         %5.2fx            %5.0f%%              %5.0f%%" %
              (r, 100 * round_correct[r] / n, 100 * round_fast[r] / n, msp,
               100 * best_correct[r] / n, 100 * best_fast[r] / n), flush=True)
    json.dump({"model": args.model, "tier": args.tier, "n": n, "rounds": args.rounds,
               "round_correct": round_correct, "round_fast": round_fast,
               "best_correct": best_correct, "best_fast": best_fast,
               "round_sp_sum": round_sp_sum},
              open(os.path.join(args.outdir, "rsi_trajectory.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
