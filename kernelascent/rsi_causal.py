"""Causal RSI experiment #1 + interfaces (docs/RSI_CAUSAL_PLAN.md).

Answers "does accumulated memory reliably help, and is the recursion (using intermediate
improvements to make the next improvement) what matters" by running four arms that share the
same practice exposure, transfer set, budget, and admission policy, differing only in how the
library is built and used:

  growing  : library grows each round; transfer uses the current library (the L2 arm).
  frozen   : one improve pass builds a library, then it is FROZEN and reused every round
             (frozen-nonempty control: isolates having a library from growing one).
  offline  : library built once from ALL practice tasks in a single pass, then frozen
             (offline-built control: ordinary library construction vs recursive accumulation).
  search   : no library; the whole practice+transfer budget is spent as best-of-N directly on
             the transfer tasks (matched-compute control).

Solver / improver split (transplant-ready):
  solver_state = {"skills": [...]}                         # S: what solve() consumes
  solve(task, solver_state, gen_fn, ...)                   # S applied to a task
  improve(solver_state, practice_tasks, gen_fn, ...)       # U: grows solver_state
These are explicit so the experiment #4 transplant (U_k on S_0) cannot secretly carry S.

Grading is the crash-isolated grade_candidates.py --one. Score is the log-interpolated
eager->expert ladder; transfer falls back to the compile baseline when no expert is known.
"""
import os, sys, json, argparse, subprocess, glob, copy
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

LIB_HEADER = """You have a library of kernel building blocks you verified earlier. Reuse or adapt the relevant ones; they are known-correct and faster than the baseline. Do not assume any other helper exists.

Your verified building blocks:
{skills}

"""


def gen_hf(model, tok, prompt, max_new):
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
    same = sorted([s for s in skills if s["family"] == family], key=lambda s: -s["score"])
    other = sorted([s for s in skills if s["family"] != family], key=lambda s: -s["score"])
    return (same + other)[:k]


def render_lib(sel):
    if not sel:
        return ""
    blocks = ["# skill %s (family=%s, %.2fx, from %s)\n```python\n%s\n```"
              % (s["name"], s["family"], s.get("speedup", 0), s["task"], s["code"]) for s in sel]
    return LIB_HEADER.format(skills="\n\n".join(blocks))


def solve(task, solver_state, gen_fn, outdir, expert, cand_timeout, retrieve_k):
    """S applied to one task. Returns (score, correct, t_cand, t_eager, t_compile, code)."""
    d = os.path.join(outdir, task["name"]); os.makedirs(d, exist_ok=True)
    open(d + "/task.py", "w").write(task["source"])
    json.dump({k: task[k] for k in ("name", "tier", "family", "meta") if k in task}, open(d + "/meta.json", "w"))
    sel = retrieve(solver_state.get("skills", []), task["family"], retrieve_k)
    raw = gen_fn(render_lib(sel) + BASE.format(src=task["source"])) or ""
    code = CB.extract_modelnew(raw)
    for old in glob.glob(d + "/cand_*.py"):
        os.remove(old)
    if not code:
        return 0.0, False, None, None, None, None
    open(d + "/cand_0.py", "w").write(code)
    r = grade_one(d, cand_timeout)
    correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
    b = r.get("best") or {}
    t_cand, t_eager, t_compile = b.get("t_cand"), r.get("t_eager"), r.get("t_compile")
    t_exp = expert.get(task["name"]) or t_compile or t_eager
    score = log_interp_score(t_cand, t_eager, t_exp, correct) if (correct and t_cand) else 0.0
    return score, correct, t_cand, t_eager, t_compile, code


def improve(solver_state, practice_tasks, gen_fn, outdir, expert, cand_timeout, retrieve_k, rnd):
    """U: solve practice tasks and bank verified, non-trivial (beats-eager) skills. Returns a
    NEW solver_state (does not mutate the input, so transplants stay clean)."""
    st = copy.deepcopy(solver_state)
    for t in practice_tasks:
        score, correct, t_cand, t_eager, t_compile, code = solve(t, st, gen_fn, outdir, expert, cand_timeout, retrieve_k)
        if correct and code and score > 0:
            sp = (t_eager / t_cand) if (t_eager and t_cand) else 0.0
            st["skills"].append({"name": "%s_r%d" % (t["family"], rnd), "family": t["family"],
                                 "code": code, "task": t["name"], "speedup": sp, "score": score})
            st["skills"].sort(key=lambda x: -x["score"]); del st["skills"][MAX_SKILLS:]
    return st


def eval_transfer(transfer_tasks, solver_state, gen_fn, outdir, expert, cand_timeout, retrieve_k):
    tot = 0.0
    for t in transfer_tasks:
        s = solve(t, solver_state, gen_fn, outdir, expert, cand_timeout, retrieve_k)[0]
        tot += s
    return tot / max(len(transfer_tasks), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-model", default="")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--arm", required=True, choices=["growing", "frozen", "offline", "search"])
    ap.add_argument("--expert-times", default="")
    ap.add_argument("--practice-n", type=int, default=8)
    ap.add_argument("--transfer-n", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--practice-seed0", type=int, default=0)
    ap.add_argument("--transfer-seed0", type=int, default=10_000_000)
    ap.add_argument("--campaign-seed", type=int, default=0, help="independent-campaign index (also offsets task seeds)")
    ap.add_argument("--tier", default="Medium")
    ap.add_argument("--retrieve-k", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--cand-timeout", type=int, default=90)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    expert = json.load(open(args.expert_times)) if args.expert_times and os.path.exists(args.expert_times) else {}
    # independent campaigns get disjoint task draws via the campaign-seed offset
    poff = args.practice_seed0 + args.campaign_seed * 2000
    toff = args.transfer_seed0 + args.campaign_seed * 2000
    practice = G.generate_tiered(args.tier, args.practice_n, seed0=poff)
    transfer = G.generate_tiered(args.tier, args.transfer_n, seed0=toff)

    if args.api_model:
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        cur.resolve(); cur.resolve_reasoning()
        gen_fn = lambda p: cur.generate(p)
        who = "api:%s" % args.api_model
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        gen_fn = lambda p: gen_hf(m, tok, p, args.max_new)
        who = "hf:%s" % args.model
    print("CAUSAL arm=%s %s tier=%s practice=%d transfer=%d rounds=%d cseed=%d gpu=%s" %
          (args.arm, who, args.tier, len(practice), len(transfer), args.rounds, args.campaign_seed,
           os.environ.get("CUDA_VISIBLE_DEVICES", "?")), flush=True)

    def od(sub):
        p = os.path.join(args.outdir, sub); os.makedirs(p, exist_ok=True); return p

    st = {"skills": []}
    C = []
    if args.arm == "search":
        # matched-compute: no library ever; best-of-N per transfer task where N = rounds+1 attempts
        best = {t["name"]: 0.0 for t in transfer}
        for k in range(args.rounds + 1):
            for t in transfer:
                s = solve(t, {"skills": []}, gen_fn, od("round%d/transfer" % k), expert, args.cand_timeout, args.retrieve_k)[0]
                best[t["name"]] = max(best[t["name"]], s)
            C.append(sum(best.values()) / len(transfer))
    elif args.arm == "offline":
        # build once from all practice tasks, then frozen transfer measured each round (rounds are just repeats -> noise)
        st = improve(st, practice, gen_fn, od("offline_build"), expert, args.cand_timeout, args.retrieve_k, 0)
        for k in range(args.rounds):
            C.append(eval_transfer(transfer, st, gen_fn, od("round%d/transfer" % k), expert, args.cand_timeout, args.retrieve_k))
    elif args.arm == "frozen":
        # C_0 empty, one improve pass, then FREEZE and reuse for all later rounds
        C.append(eval_transfer(transfer, st, gen_fn, od("round0/transfer"), expert, args.cand_timeout, args.retrieve_k))
        st = improve(st, practice, gen_fn, od("round0/practice"), expert, args.cand_timeout, args.retrieve_k, 0)
        frozen = copy.deepcopy(st)
        for k in range(1, args.rounds):
            C.append(eval_transfer(transfer, frozen, gen_fn, od("round%d/transfer" % k), expert, args.cand_timeout, args.retrieve_k))
    else:  # growing
        for k in range(args.rounds):
            C.append(eval_transfer(transfer, st, gen_fn, od("round%d/transfer" % k), expert, args.cand_timeout, args.retrieve_k))
            st = improve(st, practice, gen_fn, od("round%d/practice" % k), expert, args.cand_timeout, args.retrieve_k, k)

    print("\n=== arm=%s cseed=%d C_k=%s final_skills=%d ===" %
          (args.arm, args.campaign_seed, [round(x, 3) for x in C], len(st.get("skills", []))), flush=True)
    json.dump({"model": args.model, "api_model": args.api_model, "arm": args.arm,
               "campaign_seed": args.campaign_seed, "tier": args.tier, "rounds": args.rounds,
               "practice_n": args.practice_n, "transfer_n": args.transfer_n, "C_k": C,
               "final_skills": len(st.get("skills", []))},
              open(os.path.join(args.outdir, "causal_trajectory.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
