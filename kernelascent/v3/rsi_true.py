"""TRUE RSI on the verifier substrate: the improver uses ITS OWN (improvable) verifier as the judge
to decide which change to the verifier is best -> a better improver produces a better NEXT improver.

This closes the causal-recursion loop (Gate 3/4 shape) WITHOUT oracle leakage in the improvement
step: the verifier is "reference-assisted" (it may query the reference only on inputs IT generates;
its coverage = its config). Better coverage -> more accurate self-judgment -> picks a genuinely
better child verifier -> the child, being higher-coverage, is itself a better judge next round.

Agent state U = a verifier config {n_inputs, n_edge}. Candidate patches come from a COMMON per-task
BANK (model-generated once; labels withheld). Estimators/lineage = v3/core (unchanged):
  develop(U, anchor) : U's verifier selects a patch from the anchor's bank; graded by hidden oracle -> C.
  Q(U) = mean develop C over anchor tasks (a COMMON anchor set).
  revise(actor, target): propose M mutated child-configs; the ACTOR judges each (using the actor's
     own coverage to estimate which config selects correct patches on practice tasks); keep the one
     the actor judges best -> child config. A better actor judges better -> better child.
  F1=Q(U2)-Q(V2) with U2=revise(U1,U1), V2=revise(U0,U1) (common target); F2 similarly; N, rescue.

DENSE: anchors span ALL tasks (PROJECTS + HARD + VERY_HARD = 9), many lineage blocks.
--calib validates run_lineage detects compounding on this state shape; --run uses real model banks.
"""
import os, sys, json, argparse, random, copy, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci
from kernelascent.v3 import rsi_verify as V

ALL_TASKS = V.PROJECTS + V.HARD_PROJECTS + V.VERY_HARD_PROJECTS


def cfg_inputs(task, cfg, rng):
    return V.gen_inputs(task, int(cfg["n_inputs"]), rng, edge=False) + V.gen_inputs(task, int(cfg["n_edge"]), rng, edge=True)


def cfg_select(task, bank, cfg, rng):
    return V.select(task, bank, cfg_inputs(task, cfg, rng))


def mutate(cfg, rng):
    c = dict(cfg)
    k = rng.choice(["n_inputs", "n_inputs", "n_edge"])
    c[k] = max(1 if k == "n_inputs" else 0, c[k] + rng.choice([-2, -1, 1, 2, 3]))
    return c


def _judge_correct(task, fn, actor_cfg, rng):
    """Actor judges fn 'correct' if it matches the reference on the ACTOR's own generated inputs
    (reference-assisted; coverage = actor config). Better coverage -> more accurate judgment."""
    for a in cfg_inputs(task, actor_cfg, rng):
        try:
            if fn(*a) != task["ref"](*a):
                return False
        except Exception:
            return False
    return True


def make_behaviors(banks_anchor, banks_practice, M=5):
    """banks_* : list of (task, bank) pairs. Anchors score Q (hidden); practice used by revise."""
    def develop(agent, anchor, rng):
        task, bank = anchor
        if not bank:
            return 0.0
        return V.hidden_grade(task, cfg_select(task, bank, agent["params"], rng), rng)

    def revise(actor, target, rng):
        cands = [mutate(target["params"], rng) for _ in range(M)]
        def actor_score(cfg):
            s = 0
            for task, bank in banks_practice:
                if not bank:
                    continue
                sel = cfg_select(task, bank, cfg, rng)                 # what this config would pick
                s += _judge_correct(task, sel, actor["params"], rng)   # actor's (coverage-limited) judgment
            return s
        best = max(cands, key=actor_score)
        return {"params": dict(best)}
    return develop, revise


# ---------------------------------------------------------------- deterministic calib
def calib():
    """Scripted world: config has scalar coverage `cov`; true develop score rises with cov (bounded);
    actor JUDGES a config's value with accuracy rising in the ACTOR's cov; revise keeps the child the
    actor judges best. compound: chosen child cov keeps rising -> better judge -> F1,F2>0. oneup: caps."""
    anchors = [("t%d" % i, i) for i in range(8)]
    def true_q(cov): return min(1.0, 0.15 * cov)          # non-saturating over the tested range
    def develop(agent, anchor, rng):
        return max(0.0, min(1.0, true_q(agent["params"]["cov"]) + rng.uniform(-0.02, 0.02)))
    def make_revise(cap):
        def revise(actor, target, rng):
            cands = [target["params"]["cov"] + dc for dc in (-1, 0, 1, 2)]
            acov = actor["params"]["cov"]
            # actor ranks candidate covs; judgment noise DECREASES as actor cov rises
            noise = max(0.0, 1.5 - 0.3 * acov)
            best = max(cands, key=lambda c: true_q(c) + rng.uniform(-noise, noise))
            best = min(best, cap)                          # oneup ceiling on achievable child cov
            return {"params": {"cov": best}}
        return revise
    ok = True
    for mode, cap, exp in (("compound", 99, "F1>0,F2>0"), ("oneup", 3, "F1>0,F2~0")):
        rs = [run_lineage({"params": {"cov": 2}}, develop, make_revise(cap), anchors, random.Random(s), reps=3) for s in range(8)]
        agg = aggregate_lineages(rs); f1, f2 = agg["F1"]["mean"], agg["F2"]["mean"]
        good = (f1 > 0.01 and f2 > 0.01) if mode == "compound" else (f1 > 0.005 and abs(f2) < 0.04)
        ok = ok and good
        print("  [%s] %-9s F1=%+.3f F2=%+.3f (%s)" % ("PASS" if good else "FAIL", mode, f1, f2, exp))
    print("RSI-TRUE CALIB", "ALL PASS" if ok else "FAIL"); return 0 if ok else 1


# ---------------------------------------------------------------- real run
def run(args):
    if args.api_model:
        import curate_bedrock as CB
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        cur.resolve(); cur.resolve_reasoning(); gen_fn = lambda p: cur.generate(p); who = "api:" + args.api_model
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
    if getattr(args, "curated", ""):
        from kernelascent.v3 import curated_loader
        tasks = curated_loader.load_projects(args.curated, args.tier, getattr(args, "limit", 0) or None)
    else:
        tasks = ALL_TASKS
    print("RSI-TRUE %s curated=%s tier=%s tasks=%d blocks=%d" % (who, bool(getattr(args, "curated", "")), getattr(args, "tier", ""), len(tasks), args.blocks), flush=True)
    rng0 = random.Random(0)
    # DENSE common banks: K candidates per task (once), reused across the lineage
    banks = []
    for task in tasks:
        banks.append((task, V.gen_candidates(task, gen_fn, args.K,
                      trace_dir=os.path.join(args.outdir, "traces"), tag=task["name"])))
    results = []
    for b in range(args.blocks):
        rng = random.Random(1000 + b)
        idx = list(range(len(banks))); rng.shuffle(idx)
        anch = [banks[i] for i in idx[: args.anchor_n]]
        prac = [banks[i] for i in idx[args.anchor_n: args.anchor_n + args.practice_n]] or anch
        develop, revise = make_behaviors(anch, prac, M=args.M)
        U0 = {"params": {"n_inputs": 2, "n_edge": 0}}
        r = run_lineage(U0, develop, revise, anch, rng, reps=1)
        results.append(r)
        print("b%d F1=%+.3f F2=%+.3f N1=%+.3f N2=%+.3f rescue=%+.3f" % (b, r.F1, r.F2, r.N1, r.N2, r.rescue_minus_revert), flush=True)
        agg = aggregate_lineages(results)
        json.dump({"who": who, "tier": getattr(args, "tier", ""), "tasks": len(tasks), "blocks_done": b + 1, "K": args.K,
                   "agg": agg}, open(os.path.join(args.outdir, "rsi_true.json"), "w"), indent=2)
    print("\n=== RSI-TRUE %s ===" % who)
    for k in ("q1_minus_q0", "F1", "N1", "F2", "N2", "rescue_minus_revert"):
        print("  %-20s %s" % (k, aggregate_lineages(results)[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--model", default=""); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--anchor-n", type=int, default=6); ap.add_argument("--practice-n", type=int, default=3)
    ap.add_argument("--curated", default="", help="dir/file of curated tasks; overrides ALL_TASKS")
    ap.add_argument("--tier", default="", help="curated tier: easy|medium|hard|ultra")
    ap.add_argument("--limit", type=int, default=0, help="cap curated tasks (0=all)")
    ap.add_argument("--max-new", type=int, default=1024); ap.add_argument("--outdir", default="/tmp/rsitrue")
    args = ap.parse_args()
    if args.calib:
        sys.exit(calib())
    os.makedirs(args.outdir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
