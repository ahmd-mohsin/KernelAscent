"""RSI-Depth L2 campaign: verified skill/code-block memory with transfer.

The persistence channel L0 lacks. The agent keeps a library of verified kernel blocks it
earned on PRACTICE tasks; on a disjoint, private-seed TRANSFER set it may retrieve and
compose those blocks but not edit the library. If transfer score C_k rises across rounds
as the library grows, that is compounding, transferable self-improvement without any
weight update. This is the first level the plan predicts a non-flat slope (RSI_DEPTH_PLAN
sections 2 and 5).

Protocol per round k (k = 0..K-1):
  transfer phase  : solve the frozen transfer set with the current library -> C_k
  practice phase  : solve practice tasks (keep-best); bank each correct solution as a skill
So C_0 is the empty-library baseline and C_k reflects k rounds of banked skills.

Grading is the crash-isolated grade_candidates.py --one. Scoring is the log-interpolated
eager->expert ladder (scoring.py); transfer tasks fall back to the torch.compile rung when
no expert time is known, which is a constant offset across rounds and so does not create a
spurious slope.
"""
import os, sys, json, argparse, subprocess, glob, re
import torch
import gen_source_tasks as G
import curate_bedrock as CB
from scoring import log_interp_score, keep_best

HERE = os.path.dirname(os.path.abspath(__file__))
GRADER = os.path.join(HERE, "grade_candidates.py")
MAX_SKILLS = 40

BASE = """Optimize this PyTorch module for speed on an NVIDIA A100 GPU. Keep __init__ identical; only rewrite forward. Use Triton or fused PyTorch ops, numerically equivalent output.
Output exactly ONE class named ModelNew in a single ```python code block. No prose.

Reference module:
```python
{src}
```"""

LIB_HEADER = """You have a library of kernel building blocks you verified on earlier tasks. Reuse or adapt the relevant ones; they are known-correct and faster than the baseline. Do not assume any other helper exists.

Your verified building blocks:
{skills}

"""


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


def retrieve(skills, family, k=3):
    """Skills for the task's family first, then any, top-k by banked score."""
    same = sorted([s for s in skills if s["family"] == family], key=lambda s: -s["score"])
    other = sorted([s for s in skills if s["family"] != family], key=lambda s: -s["score"])
    return (same + other)[:k]


def render_lib(sel):
    if not sel:
        return ""
    blocks = []
    for s in sel:
        blocks.append("# skill %s (family=%s, %.2fx vs baseline, from %s)\n```python\n%s\n```"
                      % (s["name"], s["family"], s.get("speedup", 0), s["task"], s["code"]))
    return LIB_HEADER.format(skills="\n\n".join(blocks))


def solve(gen_fn, task, skills, outdir, cand_timeout, retrieve_k):
    """One keep-best solve of a task with the given (frozen) skills in context, using the
    generate callable gen_fn(prompt)->raw (open-weight HF or Bedrock API). Returns a
    5-tuple (correct, t_cand, t_eager, t_compile, code) or (0.0, None, None) on no-candidate."""
    d = os.path.join(outdir, task["name"]); os.makedirs(d, exist_ok=True)
    open(d + "/task.py", "w").write(task["source"])
    json.dump({k: task[k] for k in ("name", "tier", "family", "meta") if k in task}, open(d + "/meta.json", "w"))
    sel = retrieve(skills, task["family"], retrieve_k)
    prompt = render_lib(sel) + BASE.format(src=task["source"])
    raw = gen_fn(prompt) or ""
    code = CB.extract_modelnew(raw)
    for old in glob.glob(d + "/cand_*.py"):
        os.remove(old)
    if not code:
        return 0.0, None, None
    open(d + "/cand_0.py", "w").write(code)
    r = grade_one(d, cand_timeout)
    correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
    b = r.get("best") or {}
    t_cand, t_eager, t_compile = b.get("t_cand"), r.get("t_eager"), r.get("t_compile")
    return (correct, t_cand, t_eager, t_compile, code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id, or a label when --api-model is set")
    ap.add_argument("--api-model", default="", help="Bedrock model id; if set, generate via API instead of HF weights")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--expert-times", default="")
    ap.add_argument("--practice-n", type=int, default=8)
    ap.add_argument("--transfer-n", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--practice-seed0", type=int, default=0)          # public
    ap.add_argument("--transfer-seed0", type=int, default=10_000_000)  # private, disjoint
    ap.add_argument("--tier", default="Medium")
    ap.add_argument("--retrieve-k", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--cand-timeout", type=int, default=90)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    expert = json.load(open(args.expert_times)) if args.expert_times and os.path.exists(args.expert_times) else {}
    practice = G.generate_tiered(args.tier, args.practice_n, seed0=args.practice_seed0)
    transfer = G.generate_tiered(args.tier, args.transfer_n, seed0=args.transfer_seed0)

    if args.api_model:                       # Bedrock API agent (grading still on the local GPU)
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        wid, wmt = cur.resolve(); cur.resolve_reasoning()
        gen_fn = lambda prompt: cur.generate(prompt)
        who = "api:%s(id=%s,mt=%d)" % (args.api_model, wid, wmt)
    else:                                    # open-weight HF agent
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        gen_fn = lambda prompt: gen_one(model, tok, prompt, args.max_new)
        who = "hf:%s" % args.model
    print("L2 %s tier=%s practice=%d transfer=%d rounds=%d gpu=%s" %
          (who, args.tier, len(practice), len(transfer), args.rounds, os.environ.get("CUDA_VISIBLE_DEVICES", "?")), flush=True)

    def score_of(res, name):
        correct, t_cand, t_eager, t_compile, _ = res if res and len(res) == 5 else (False, None, None, None, None)
        if not (correct and t_cand):
            return 0.0
        t_exp = expert.get(name) or t_compile or t_eager
        return log_interp_score(t_cand, t_eager, t_exp, correct)

    skills = []
    C = []
    for k in range(args.rounds):
        # transfer phase (frozen library)
        tdir = os.path.join(args.outdir, "round%d" % k, "transfer"); os.makedirs(tdir, exist_ok=True)
        tsum = 0.0
        for t in transfer:
            res = solve(gen_fn, t, skills, tdir, args.cand_timeout, args.retrieve_k)
            s = score_of(res, t["name"]); tsum += s
            print("  r%d transfer %-26s score=%.2f (skills=%d)" % (k, t["name"][:26], s, len(skills)), flush=True)
        Ck = tsum / len(transfer); C.append(Ck)
        print("  == C_%d = %.3f (library=%d skills) ==" % (k, Ck, len(skills)), flush=True)
        # practice phase (grow library)
        pdir = os.path.join(args.outdir, "round%d" % k, "practice"); os.makedirs(pdir, exist_ok=True)
        for t in practice:
            res = solve(gen_fn, t, skills, pdir, args.cand_timeout, args.retrieve_k)
            correct, t_cand, t_eager, t_compile, code = res if res and len(res) == 5 else (False, None, None, None, None)
            s = score_of(res, t["name"])
            if correct and code and s > 0:            # bank only verified, non-trivial (beats eager) blocks
                sp = (t_eager / t_cand) if (t_eager and t_cand) else 0.0
                skills.append({"name": "%s_r%d" % (t["family"], k), "family": t["family"],
                               "code": code, "task": t["name"], "speedup": sp, "score": s})
                skills.sort(key=lambda x: -x["score"]); del skills[MAX_SKILLS:]
            print("  r%d practice %-26s score=%.2f banked=%s" % (k, t["name"][:26], s, correct and s > 0), flush=True)

    print("\n=== L2 trajectory (%s, %s) ===" % (args.model, args.tier), flush=True)
    print("round  C_k(transfer)  library_size", flush=True)
    # library size at each C_k measurement was len(skills) before that round's practice; recompute simply
    print("C_k = %s" % [round(x, 3) for x in C], flush=True)
    json.dump({"model": args.model, "tier": args.tier, "rounds": args.rounds,
               "C_k": C, "final_library": [{k: s[k] for k in ("name", "family", "task", "speedup", "score")} for s in skills]},
              open(os.path.join(args.outdir, "l2_trajectory.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
