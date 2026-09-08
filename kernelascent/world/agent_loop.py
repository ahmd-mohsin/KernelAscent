"""M2 + M4: a model agent that does engineering research in the inference world (optimize the MLP
operator -- the 57%-of-service-time lever), with the causal lineage producing the THREE CONCLUSIONS
separately:
  (1) experience-improvement  : q1_minus_q0  (does a self-revision of the research procedure raise the
                                 quality of the operator it then produces)
  (2) better-improver / child : N1, N2       (does the produced successor improve on the unchanged target)
  (3) causal recursive reuse  : F1, F2       (newer producer vs older producer on a COMMON target)
Plus base capability Q(U0) (M2: can the model produce a correct+faster MLP at all).

W/K/U: U = the research procedure (the operator-optimization strategy text the agent evolves); develop
produces an operator and grades it end-to-end on the SERVICE (correctness vs fp32-gold + speedup); the
world W is reset to a common state for each assay so "better world" cannot be confused with "better
agent". Estimators/lineage = v3/core; world grade = world/inference_world.
"""
import os, sys, json, argparse, random, copy, re, signal
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT); sys.path.insert(0, PKG)
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci
from kernelascent.world import inference_world as WLD

MLP_SRC = "def mlp(x, Wg, Wu, Wd):\n    return (F.silu(x @ Wg) * (x @ Wu)) @ Wd\n"
BASE_SOLVE = ("Rewrite the operator to run FASTER on an A100 while numerically equivalent (bf16). Use any "
              "approach you judge fastest (fused torch, better tiling, Triton). Keep the exact signature.")
BASE_REVISE = ("Improve the strategy another agent uses to speed up this operator: point it at the highest-"
               "leverage change. You may rewrite its strategy freely.")

# NB: minimal framing on purpose -- the Bedrock account's content filter blocks the verbose
# "live inference service / No prose" framing on some models; this phrasing passes for all.
SOLVE_TMPL = ("Strategy: {strat}\nRewrite this function to run faster on an A100, numerically equivalent (bf16), "
              "same signature. Return ONLY the function mlp in one ```python block.\n```python\n{src}\n```")


def _extract(txt):
    m = re.search(r"```(?:python)?\s*(.*?)```", txt or "", re.S)
    return m.group(1) if m else (txt or "")


def score_from_grade(g):
    if not g.get("correct"):
        return 0.0
    sp = g.get("speedup_vs_baseline", 1.0) or 1.0
    return max(0.0, min(1.0, 0.5 + 0.5 * min(1.0, (sp - 1.0) / 1.0)))   # 0.5 at parity -> 1.0 at 2x


def make_behaviors(gen_fn, practice_cfgs, share=0.567):
    def develop(agent, anchor_cfg, rng):
        prompt = SOLVE_TMPL.format(share=share, strat=agent["params"].get("solve_strategy", BASE_SOLVE), src=MLP_SRC)
        code = _extract(gen_fn(prompt) or "")
        try:
            fn = WLD.load_op(code, "mlp")
        except Exception:
            fn = None
        if fn is None:
            return 0.0
        try:
            g = WLD.grade({"mlp": fn}, cfg=anchor_cfg)          # end-to-end SERVICE grade (correctness + speedup)
        except Exception:
            return 0.0
        return score_from_grade(g)

    def revise(actor, target, rng):
        child = copy.deepcopy(target)
        fb = ["cfg%d=%.2f" % (i, develop(target, c, rng)) for i, c in enumerate(practice_cfgs[:2])]
        ap = actor["params"]
        ask = (BASE_REVISE + " Your guidance: " + ap.get("revise_strategy", BASE_REVISE) +
               "\nTarget's current strategy: %r. Practice operator-scores (0 wrong /0.5 parity /1.0 ~2x): %s.\n"
               "Return ONLY JSON {\"solve_strategy\": <improved>, \"revise_strategy\": <improved>}." %
               (target["params"].get("solve_strategy", "")[:160], "; ".join(fb)))
        m = re.search(r"\{.*\}", gen_fn(ask) or "", re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                for k in ("solve_strategy", "revise_strategy"):
                    if isinstance(d.get(k), str) and 3 < len(d[k]) < 1500:
                        child["params"][k] = d[k]
            except Exception:
                pass
        return child
    return develop, revise


def anchor_configs():
    # "different serving projects" = different shapes; world reset to common state each assay
    base = WLD.world_config()
    outs = []
    for (B, S) in [(8, 256), (4, 512), (16, 128), (8, 384)]:
        c = dict(base); c["B"], c["S"] = B, S; outs.append(c)
    return outs


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
    anchors = anchor_configs()
    print("AGENT-LOOP %s blocks=%d anchors=%d" % (who, args.blocks, len(anchors)), flush=True)
    results = []
    for b in range(args.blocks):
        rng = random.Random(3000 + b)
        practice = [anchors[(b + 1) % len(anchors)], anchors[(b + 2) % len(anchors)]]
        develop, revise = make_behaviors(gen_fn, practice)
        U0 = {"params": {"solve_strategy": BASE_SOLVE, "revise_strategy": BASE_REVISE}, "skills": []}
        r = run_lineage(U0, develop, revise, anchors, rng, reps=1)
        results.append(r)
        print("b%d Q0=%.3f q1-q0=%+.3f F1=%+.3f F2=%+.3f N1=%+.3f N2=%+.3f" %
              (b, r.Q["U0"], r.q1_minus_q0, r.F1, r.F2, r.N1, r.N2), flush=True)
        agg = aggregate_lineages(results)
        Q0 = _mean_ci([x.Q["U0"] for x in results])
        out = {"who": who, "blocks_done": b + 1, "base_capability_Q0": Q0,
               "experience_improvement_q1_minus_q0": agg["q1_minus_q0"],
               "child_value_N1": agg["N1"], "child_value_N2": agg["N2"],
               "causal_reuse_F1": agg["F1"], "causal_reuse_F2": agg["F2"]}
        json.dump(out, open(os.path.join(args.outdir, "agent_loop.json"), "w"), indent=2)
    print("\n=== THREE CONCLUSIONS %s ===" % who)
    print(" base capability Q(U0):        ", Q0)
    print(" (1) experience-improvement:   ", agg["q1_minus_q0"])
    print(" (2) child value N1 / N2:      ", agg["N1"], "/", agg["N2"])
    print(" (3) causal recursive F1 / F2: ", agg["F1"], "/", agg["F2"])


# ---------------------------------------------------------------- deterministic calib (no model/GPU)
def calib():
    anchors = [{"skill_anchor": i} for i in range(4)]
    def develop(agent, anchor, rng):
        return max(0.0, 0.02 * agent["params"].get("skill", 0))    # non-saturating over the tested range
    def make_revise(cap):
        def revise(actor, target, rng):
            child = copy.deepcopy(target); p = int(actor["params"].get("power", 2))
            child["params"]["skill"] = target["params"].get("skill", 0) + p
            child["params"]["power"] = min(cap, p + 2)
            return child
        return revise
    ok = True
    for mode, cap, exp in (("compound", 99, "F1>0,F2>0"), ("oneup", 4, "F1>0,F2~0")):
        rs = [run_lineage({"params": {"skill": 0, "power": 2}}, develop, make_revise(cap), anchors, random.Random(s), reps=3) for s in range(6)]
        a = aggregate_lineages(rs); f1, f2 = a["F1"]["mean"], a["F2"]["mean"]
        good = (f1 > 0.01 and f2 > 0.01) if mode == "compound" else (f1 > 0.01 and abs(f2) < 0.03)
        ok = ok and good
        print("  [%s] %-9s F1=%+.3f F2=%+.3f (%s)" % ("PASS" if good else "FAIL", mode, f1, f2, exp))
    print("AGENT-LOOP CALIB", "ALL PASS" if ok else "FAIL"); return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--model", default=""); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--blocks", type=int, default=6); ap.add_argument("--max-new", type=int, default=2048); ap.add_argument("--outdir", default="/tmp/agentloop")
    args = ap.parse_args()
    if args.calib:
        sys.exit(calib())
    os.makedirs(args.outdir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
