"""E1/E2 with CAPABILITY-ADDITIVE references (per the never-constrain-capability correction).

The negative-control run (e1e2.py) used prescriptive strategy text and found it NET-NEGATIVE for
capable models (it boxes them in). Here the "improvement" only ADDS capability and never forbids
alternatives, so a genuine improvement should show dQ>0:

  ref-kind = budget  : best-of-B sampling. develop with solve_budget=B samples B candidates
                       (non-prescriptive prompt) and keeps the best-graded -> strictly >= best-of-1.
                       The improver with revise_budget=B proposes B candidate children and keeps the
                       one that scores best on practice tasks -> a strictly stronger improver.
  NULL (waste)       : sample B but keep a RANDOM draw (same compute, no selection) -> ~0 usefulness.
                       Isolates capability-gain from mere extra compute.

Estimators unchanged (core.py). Contrasts:
  E1 usefulness  dQ    = Q(solve_budget=B) - Q(solve_budget=1)            (>=0 expected; positive control)
  E1 channel     F_inj = V(revise_budget=B, T) - V(revise_budget=1, T)    (better improver -> better child)
  E1 null        dQ_n  = Q(solve_budget=B, keep=random) - Q(solve_budget=1)
  E2 self-use    F_self= V(revise_budget=B, T) - V(revise_budget=1, T) with develop-benefit held fixed
                        (both actors solve_budget=B); N_self + rescue as before.

--calib validates the logic deterministically (no GPU/API); --model/--api-model runs it for real.
"""
import os, sys, json, argparse, subprocess, glob, copy, random, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import estimate_Q, estimate_V, _mean_ci

GRADER = os.path.join(os.path.dirname(HERE), "grade_candidates.py")
FAST = 1.10

# deliberately minimal + non-prescriptive: name the goal, not the method (never constrain capability)
BASE_SOLVE = ("Optimize this PyTorch module for speed on an A100. Keep __init__ identical; rewrite forward to be "
              "as fast as possible while numerically equivalent. Use ANY approach you judge fastest. "
              "Output exactly ONE class named ModelNew in a single ```python block. No prose.")
BASE_REVISE = ("You improve another kernel agent. Look at its recent scores and change its approach however you "
               "think will make it produce faster, correct kernels. You may rewrite its strategy freely.")


# --------------------------------------------------------------------------------- deterministic calib
def calib_behaviors(noise=0.15, revise_relevant=True):
    def develop(agent, project, rng):
        B = int(agent["params"].get("solve_budget", 1))
        keep = agent["params"].get("keep", "best")
        draws = [max(0.0, min(1.0, agent["params"].get("skill", 0.3) + rng.uniform(-noise, noise) + 0.15)) for _ in range(B)]
        return max(draws) if keep == "best" else rng.choice(draws)

    def revise(actor, target, rng):
        Bs = int(actor["params"].get("revise_budget", 1)) if revise_relevant else 1
        base_power = 0.10
        # propose Bs children; each child's skill = target skill + power + a draw; keep best-on-practice
        cand = []
        for _ in range(Bs):
            child = copy.deepcopy(target)
            child["params"]["skill"] = min(1.0, target["params"].get("skill", 0.3) + base_power + max(0.0, rng.uniform(-noise, noise) + 0.05))
            cand.append(child)
        keep = actor["params"].get("keep", "best")
        if keep == "best":
            return max(cand, key=lambda c: c["params"]["skill"])
        return rng.choice(cand)
    return develop, revise


def _agent(**params):
    return {"params": {"skill": 0.3, **params}, "skills": []}


def run_calib():
    rng = random.Random(0); anchors = [{"id": i} for i in range(6)]
    dev, rev = calib_behaviors(revise_relevant=True)
    q = lambda u: estimate_Q(u, anchors, dev, rng, reps=4)
    v = lambda a, t: estimate_V(a, t, rev, anchors, dev, rng, reps=4)
    U1 = _agent(solve_budget=1); UB = _agent(solve_budget=4)
    T = _agent(skill=0.3)
    dQ = q(UB) - q(U1)
    Finj = v(_agent(revise_budget=4), T) - v(_agent(revise_budget=1), T)
    dQ_null = q(_agent(solve_budget=4, keep="random")) - q(U1)
    self_on = _agent(solve_budget=4, revise_budget=4); self_off = _agent(solve_budget=4, revise_budget=1)
    Fself = v(self_on, T) - v(self_off, T)
    Nself = v(self_on, self_on) - q(self_on)
    dev0, rev0 = calib_behaviors(revise_relevant=False)
    v0 = lambda a, t: estimate_V(a, t, rev0, anchors, dev0, rng, reps=4)
    Fself_irrel = v0(_agent(solve_budget=4, revise_budget=4), T) - v0(_agent(solve_budget=4, revise_budget=1), T)
    checks = [
        ("E1 usefulness dQ(best-of-B) > 0", dQ, dQ > 0.02),
        ("E1 channel F_inj > 0", Finj, Finj > 0.0),
        ("E1 null (waste) dQ ~ 0", dQ_null, abs(dQ_null) < 0.05),
        ("E2 F_selfuse > 0 (revise-relevant)", Fself, Fself > 0.0),
        ("E2 N_selfuse >= 0", Nself, Nself >= -0.02),
        ("E2 F_selfuse ~ 0 (improver budget irrelevant)", Fself_irrel, abs(Fself_irrel) < 0.05),
    ]
    print("=== E1/E2 CAPABILITY-ADDITIVE CALIBRATION ===")
    ok = True
    for name, val, passed in checks:
        ok = ok and passed
        print("  [%s] %-46s = %+.3f" % ("PASS" if passed else "FAIL", name, val))
    print("CALIB", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------- real behaviors
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


def real_behaviors(gen_fn, workdir, practice, stats):
    import curate_bedrock as CB
    ctr = {"n": 0}

    def _grade_code(d, code):
        for old in glob.glob(d + "/cand_*.py"):
            os.remove(old)
        if not code:
            return 0.0
        open(d + "/cand_0.py", "w").write(code)
        r = grade_one(d)
        correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
        sp = r.get("best_speedup_roofline", 0.0) or 0.0
        return 1.0 if (correct and sp >= FAST) else (0.5 if correct else 0.0)

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
            code = CB.extract_modelnew(gen_fn(prompt) or "")
            scored.append(_grade_code(d, code))
        c = max(scored) if keep == "best" else rng.choice(scored)
        best = max(scored)
        stats.append((1 if best > 0 else 0, 1 if best >= 1.0 else 0))
        return c

    def revise(actor, target, rng):
        ap = actor["params"]; Bs = int(ap.get("revise_budget", 1)); keep = ap.get("keep", "best")
        guide = ap.get("revise_strategy", BASE_REVISE)
        fb = ["%s=%.1f" % (t["name"][:16], develop(target, t, rng)) for t in practice[:2]]
        ask = (guide + "\nThe target's current approach: solve=%r. Its recent scores "
               "(0 wrong /0.5 correct /1.0 correct&>=1.10x): %s.\nPropose an improved approach. Return ONLY a JSON "
               "object with key \"solve_strategy\" (a short instruction for how it should optimize; you may keep it "
               "open-ended)." % (target["params"].get("solve_strategy", BASE_SOLVE)[:160], "; ".join(fb)))
        cand = []
        for _ in range(Bs):
            child = copy.deepcopy(target)
            m = re.search(r"\{.*\}", gen_fn(ask) or "", re.S)
            if m:
                try:
                    dd = json.loads(m.group(0))
                    if isinstance(dd.get("solve_strategy"), str) and 3 < len(dd["solve_strategy"]) < 1500:
                        child["params"]["solve_strategy"] = dd["solve_strategy"]
                except Exception:
                    pass
            cand.append(child)
        if keep != "best" or Bs == 1:
            return rng.choice(cand)
        # keep the proposed child that scores best on the practice set (capability-additive selection)
        return max(cand, key=lambda c: sum(develop(c, t, rng) for t in practice[:2]))
    return develop, revise


def A(solve_budget=1, revise_budget=1, keep="best"):
    return {"params": {"solve_budget": solve_budget, "revise_budget": revise_budget, "keep": keep}, "skills": []}


def run_real(args):
    import gen_source_tasks as G
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
    print("E1/E2-POS %s budget=%d blocks=%d" % (who, B, args.blocks), flush=True)

    E1 = {"dQ_useful": [], "F_channel": [], "dQ_null": []}
    E2 = {"F_selfuse": [], "N_selfuse": [], "rescue_minus_revert": []}
    st = []

    def dump(done):
        out = {"who": who, "budget": B, "blocks_target": args.blocks, "blocks_done": done,
               "E1": {k: _mean_ci(v) for k, v in E1.items()},
               "E2": {k: _mean_ci(v) for k, v in E2.items()},
               "decompose": {"n_dev": len(st), "correct_rate": round(sum(c for c, f in st) / (len(st) or 1), 3),
                             "fast_rate": round(sum(f for c, f in st) / (len(st) or 1), 3)}}
        json.dump(out, open(os.path.join(args.outdir, "e1e2_pos.json"), "w"), indent=2)
        return out

    for b in range(args.blocks):
        practice = G.generate_tiered("Medium", 2, seed0=b * 400)
        anchors = G.generate_tiered("Medium", args.anchor_n, seed0=10_000_000 + b * 400)
        wd = os.path.join(args.outdir, "b%d" % b); os.makedirs(wd, exist_ok=True)
        develop, revise = real_behaviors(gen_fn, wd, practice, st)
        rng = random.Random(9000 + b)
        T = revise(A(), copy.deepcopy(A()), rng)
        E1["dQ_useful"].append(estimate_Q(A(solve_budget=B), anchors, develop, rng, reps=1) - estimate_Q(A(solve_budget=1), anchors, develop, rng, reps=1))
        E1["F_channel"].append(estimate_V(A(revise_budget=B), T, revise, anchors, develop, rng, reps=1) - estimate_V(A(revise_budget=1), T, revise, anchors, develop, rng, reps=1))
        E1["dQ_null"].append(estimate_Q(A(solve_budget=B, keep="random"), anchors, develop, rng, reps=1) - estimate_Q(A(solve_budget=1), anchors, develop, rng, reps=1))
        self_on = A(solve_budget=B, revise_budget=B); self_off = A(solve_budget=B, revise_budget=1)
        v_on = estimate_V(self_on, T, revise, anchors, develop, rng, reps=1)
        v_off = estimate_V(self_off, T, revise, anchors, develop, rng, reps=1)
        E2["F_selfuse"].append(v_on - v_off)
        E2["N_selfuse"].append(estimate_V(self_on, self_on, revise, anchors, develop, rng, reps=1) - estimate_Q(self_on, anchors, develop, rng, reps=1))
        E2["rescue_minus_revert"].append(estimate_V(copy.deepcopy(self_on), T, revise, anchors, develop, rng, reps=1) - v_off)
        print("b%d dQ=%.3f Fchan=%.3f dQnull=%.3f | Fself=%.3f Nself=%.3f" %
              (b, E1["dQ_useful"][-1], E1["F_channel"][-1], E1["dQ_null"][-1], E2["F_selfuse"][-1], E2["N_selfuse"][-1]), flush=True)
        dump(b + 1)
    out = dump(args.blocks)
    print("\n=== E1/E2-POS %s ===\n E1: %s\n E2: %s\n dec: %s" % (who, json.dumps(out["E1"]), json.dumps(out["E2"]), out["decompose"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--model", default=""); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--budget", type=int, default=3); ap.add_argument("--blocks", type=int, default=8); ap.add_argument("--anchor-n", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=8192); ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    if args.calib:
        sys.exit(run_calib())
    assert args.outdir, "need --outdir"
    os.makedirs(args.outdir, exist_ok=True)
    run_real(args)


if __name__ == "__main__":
    main()
