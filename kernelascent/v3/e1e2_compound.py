"""Frontier COMPOUNDING test (gaps 1+2): does the frontier causal self-use link REPEAT (F1 then
F2 on a fresh common target) and survive RESCUE, under a CONTINUOUS speed-resolved score so that
kernels differing only in speed no longer bin to equal Q.

Uses the validated two-link lineage `core.run_lineage` (U0->U1; forks with a COMMON target at each
link giving F1=Q[U2]-Q[V2] and F2=Q[U3]-Q[V3]; rescue at U2) with capability-ADDITIVE, non-
prescriptive best-of-B behaviors (per the never-constrain principle). Lineage is the unit; many
blocks give paired CIs. Compounding evidence = F1>0 AND F2>0 (repeat) with CIs clear of 0, holding
across rescue + replication (independent blocks).

Continuous score (correct kernels): C = 0.5 + 0.5*tanh((speedup_vs_roofline - 1.0)/0.10), wrong=0.
-> 0.5 at parity with min(eager,compile), ~0.88 at 1.10x (the old FAST wall), smooth beyond; gives
fine correct->fast resolution instead of the {0,0.5,1.0} binning that forced F to exactly 0.

--calib validates run_lineage detects a compounding world (F1>0,F2>0) and a one-upgrade world
(F1>0,F2~0). --model/--api-model runs it for real (Fable = frontier lead; Coder-7B/gpt-oss controls).
"""
import os, sys, json, argparse, subprocess, glob, copy, random, re, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci
GRADER = os.path.join(os.path.dirname(HERE), "grade_candidates.py")

BASE_SOLVE = ("Optimize this PyTorch module for speed on an A100. Keep __init__ identical; rewrite forward to be "
              "as fast as possible while numerically equivalent. Use ANY approach you judge fastest. "
              "Output exactly ONE class named ModelNew in a single ```python block. No prose.")
BASE_REVISE = ("You improve another kernel agent. Look at its recent scores and change its approach however you "
               "think will make it produce faster, correct kernels. You may rewrite its strategy freely.")


def cont_score(correct, sp):
    if not correct:
        return 0.0
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh((sp - 1.0) / 0.10)))


def bounded_score(correct, sp):
    return 0.0 if not correct else (1.0 if sp >= 1.10 else 0.5)


# ------------------------------------------------------------------- deterministic calib
def calib_behaviors(mode):
    """Scripted world with an IMPROVER-improves-IMPROVER channel (the only way compounding is
    possible). agent.prod drives develop; agent.meta is improver strength. A strong actor lifts the
    child's prod by actor.meta AND lifts the child's meta -> the child is a better improver.
    mode='compound': meta keeps rising -> F1>0,F2>0. mode='oneup': meta caps after one link -> F1>0,F2~0."""
    def develop(agent, project, rng):
        return max(0.0, min(1.0, agent["params"].get("prod", 0.3)))

    def revise(actor, target, rng):
        child = copy.deepcopy(target)
        am = actor["params"].get("meta", 0.1)                       # actor's improver strength
        child["params"]["prod"] = min(1.0, target["params"].get("prod", 0.3) + am)
        tm = target["params"].get("meta", 0.1)
        if mode == "compound":
            child["params"]["meta"] = min(1.0, tm + 0.10)            # improver improves the improver
        elif mode == "oneup":
            child["params"]["meta"] = min(0.20, tm + 0.10)           # caps after the first link
        return child
    return develop, revise


def run_calib():
    anchors = [{"id": i} for i in range(5)]; ok = True
    for mode, exp in (("compound", "F1>0 & F2>0"), ("oneup", "F1>0 & F2~0")):
        dev, rev = calib_behaviors(mode)
        rs = [run_lineage({"params": {"prod": 0.3, "meta": 0.1}, "skills": []}, dev, rev, anchors, random.Random(s), reps=3) for s in range(6)]
        agg = aggregate_lineages(rs)
        f1, f2 = agg["F1"]["mean"], agg["F2"]["mean"]
        good = (f1 > 0.02 and f2 > 0.02) if mode == "compound" else (f1 > 0.02 and abs(f2) < 0.03)
        ok = ok and good
        print("  [%s] %-9s F1=%+.3f F2=%+.3f (expect %s)" % ("PASS" if good else "FAIL", mode, f1, f2, exp))
    print("CALIB", "ALL PASS" if ok else "FAIL"); return 0 if ok else 1


# ------------------------------------------------------------------- real behaviors
def grade_one(d, ct=90):
    rj = os.path.join(d, "results.json")
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


def real_behaviors(gen_fn, workdir, practice, stats, score_fn):
    import curate_bedrock as CB
    ctr = {"n": 0}

    def _grade(d, code):
        for old in glob.glob(d + "/cand_*.py"):
            os.remove(old)
        if not code:
            return 0.0, (0, 0.0)
        open(d + "/cand_0.py", "w").write(code)
        r = grade_one(d)
        correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
        sp = r.get("best_speedup_roofline", 0.0) or 0.0
        return score_fn(correct, sp), (1 if correct else 0, sp)

    def develop(agent, project, rng):
        ctr["n"] += 1
        p = agent["params"]; B = int(p.get("solve_budget", 1)); keep = p.get("keep", "best")
        strat = p.get("solve_strategy", BASE_SOLVE)
        prompt = strat + "\n\n```python\n" + project["source"] + "\n```"
        scored = []
        for j in range(B):
            d = os.path.join(workdir, "d%d_%d" % (ctr["n"], j)); os.makedirs(d, exist_ok=True)
            open(d + "/task.py", "w").write(project["source"])
            json.dump({k: project[k] for k in ("name", "tier", "family", "meta") if k in project}, open(d + "/meta.json", "w"))
            scored.append(_grade(d, CB.extract_modelnew(gen_fn(prompt) or "")))
        pick = max(scored, key=lambda x: x[0]) if keep == "best" else rng.choice(scored)
        best = max(scored, key=lambda x: x[0])
        stats.append((best[1][0], 1 if best[1][1] >= 1.10 else 0))
        return pick[0]

    def revise(actor, target, rng):
        ap = actor["params"]; Bs = int(ap.get("revise_budget", 1)); keep = ap.get("keep", "best")
        guide = ap.get("revise_strategy", BASE_REVISE)
        fb = ["%s=%.2f" % (t["name"][:14], develop(target, t, rng)) for t in practice[:2]]
        # open the RECURSIVE pathway: the improver may rewrite BOTH how the target solves AND how the
        # target itself improves (its revise_strategy) -> the child can become a better improver.
        ask = (guide + "\nThe target agent solves with: solve=%r and improves others with: revise=%r. "
               "Its recent scores (0=wrong, 0.5=parity with torch.compile, ->1.0 faster): %s.\nPropose an improved "
               "agent. Return ONLY a JSON object with keys \"solve_strategy\" and \"revise_strategy\" (open-ended "
               "instructions; do not over-constrain -- add capability, do not forbid approaches)."
               % (target["params"].get("solve_strategy", BASE_SOLVE)[:140],
                  target["params"].get("revise_strategy", BASE_REVISE)[:140], "; ".join(fb)))
        cand = []
        for _ in range(Bs):
            child = copy.deepcopy(target)
            m = re.search(r"\{.*\}", gen_fn(ask) or "", re.S)
            if m:
                try:
                    dd = json.loads(m.group(0))
                    for kk in ("solve_strategy", "revise_strategy"):
                        if isinstance(dd.get(kk), str) and 3 < len(dd[kk]) < 1500:
                            child["params"][kk] = dd[kk]
                except Exception:
                    pass
            cand.append(child)
        if keep != "best" or Bs == 1:
            return rng.choice(cand)
        return max(cand, key=lambda c: sum(develop(c, t, rng) for t in practice[:2]))
    return develop, revise


def run_real(args):
    import gen_source_tasks as G
    score_fn = cont_score if args.score == "continuous" else bounded_score
    if args.api_model:
        import curate_bedrock as CB
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        rid, mt = cur.resolve(); rc = cur.resolve_reasoning()
        print("RESOLVED id=%s maxTokens=%s reasoning=%s" % (rid, mt, rc), flush=True)
        gen_fn = lambda p: cur.generate(p); who = "api:" + args.api_model
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model); mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        def gen_fn(p):
            enc = tok([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)], return_tensors="pt", padding=True).to("cuda")
            import torch as _t
            with _t.no_grad():
                o = mdl.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
            return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        who = "hf:" + args.model
    B = args.budget
    print("COMPOUND %s budget=%d blocks=%d score=%s" % (who, B, args.blocks, args.score), flush=True)

    results = []; st = []

    def dump(done):
        agg = aggregate_lineages(results) if results else {}
        out = {"who": who, "budget": B, "score": args.score, "blocks_target": args.blocks, "blocks_done": done,
               "agg": agg, "per_block": [{"F1": r.F1, "F2": r.F2, "N1": r.N1, "N2": r.N2,
                                          "rescue_minus_revert": r.rescue_minus_revert, "Q": r.Q} for r in results],
               "decompose": {"n_dev": len(st), "correct_rate": round(sum(c for c, f in st) / (len(st) or 1), 3),
                             "fast_rate": round(sum(f for c, f in st) / (len(st) or 1), 3)}}
        json.dump(out, open(os.path.join(args.outdir, "compound.json"), "w"), indent=2)
        return out

    for b in range(args.blocks):
        practice = G.generate_tiered("Medium", 2, seed0=b * 400)
        anchors = G.generate_tiered("Medium", args.anchor_n, seed0=10_000_000 + b * 400)
        wd = os.path.join(args.outdir, "b%d" % b); os.makedirs(wd, exist_ok=True)
        develop, revise = real_behaviors(gen_fn, wd, practice, st, score_fn)
        U0 = {"params": {"solve_budget": B, "revise_budget": B, "keep": "best",
                         "solve_strategy": BASE_SOLVE, "revise_strategy": BASE_REVISE}, "skills": []}
        r = run_lineage(U0, develop, revise, anchors, random.Random(12000 + b), reps=1)
        results.append(r)
        print("b%d F1=%+.3f F2=%+.3f N1=%+.3f N2=%+.3f rescue=%+.3f" % (b, r.F1, r.F2, r.N1, r.N2, r.rescue_minus_revert), flush=True)
        dump(b + 1)
    out = dump(args.blocks)
    print("\n=== COMPOUND %s ===" % who)
    for k in ("F1", "F2", "N1", "N2", "rescue_minus_revert"):
        print("  %-20s %s" % (k, out["agg"].get(k)))
    print("  decompose", out["decompose"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--model", default=""); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--budget", type=int, default=3); ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--anchor-n", type=int, default=3)
    ap.add_argument("--score", choices=["continuous", "bounded"], default="continuous")
    ap.add_argument("--max-new", type=int, default=8192); ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    if args.calib:
        sys.exit(run_calib())
    assert args.outdir, "need --outdir"
    os.makedirs(args.outdir, exist_ok=True)
    run_real(args)


if __name__ == "__main__":
    main()
