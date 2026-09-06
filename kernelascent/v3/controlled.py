"""v3 controlled experiment: matched narrow-vs-rich self-editing, with the recursive pathway
OPEN (the editable state governs revise, not just develop).

Fixes the closed-pathway audit: the actor's revise behavior is shaped by editable state
(params["revise_strategy"], and in the rich condition a free-form procedure), which revise
itself edits. So a discovered better revise_strategy/procedure makes a better next producer ->
F can be nonzero. Narrow and rich differ ONLY in the edit space; everything else (initial
agent, projects, evidence, budgets, scoring, selection) is held identical and PAIRED per block.

Per model: 8 matched blocks; in each block both conditions run from the same U0 on the same
paired tasks/rng. Report F1,N1,F2,N2 per condition (with lineage CIs) and dF_g = F_rich -
F_narrow. Scoring is FIXED (bounded C in {0,0.5,1.0} vs the compile baseline); Q is decomposed
into correct-rate (0->0.5) and fast-rate (0.5->1.0). Records whether each child actually
differs from its target (verified inheritance).
"""
import os, sys, json, argparse, subprocess, glob, copy, random, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import gen_source_tasks as G
import curate_bedrock as CB
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci
GRADER = os.path.join(os.path.dirname(HERE), "grade_candidates.py")
FAST = 1.10

SOLVE0 = "write a fused Triton kernel or fused PyTorch, numerically equivalent, aiming to beat torch.compile"
REV0 = "diagnose why the target's kernels miss the speed target, then propose a sharper solve strategy and a better revise strategy"


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
        return {"correct": False, "best": None}


def make_behaviors(gen_fn, workdir, practice, stats):
    ctr = {"n": 0}

    def develop(agent, project, rng):
        ctr["n"] += 1
        d = os.path.join(workdir, "d%d" % ctr["n"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(project["source"])
        json.dump({k: project[k] for k in ("name", "tier", "family", "meta") if k in project}, open(d + "/meta.json", "w"))
        p = agent["params"]
        proc = ("Procedure: " + p["procedure"] + "\n") if p.get("procedure") else ""
        prompt = ("Optimize this PyTorch module for speed on an A100. Keep __init__ identical; rewrite forward.\n"
                  + proc + "Strategy: " + p.get("solve_strategy", SOLVE0) +
                  "\nOutput exactly ONE class named ModelNew in a single ```python block. No prose.\n\n```python\n"
                  + project["source"] + "\n```")
        code = CB.extract_modelnew(gen_fn(prompt) or "")
        for old in glob.glob(d + "/cand_*.py"):
            os.remove(old)
        if not code:
            stats.append((0, 0)); return 0.0
        open(d + "/cand_0.py", "w").write(code)
        r = grade_one(d)
        correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
        sp = r.get("best_speedup_roofline", 0.0) or 0.0
        fast = correct and sp >= FAST
        stats.append((1 if correct else 0, 1 if fast else 0))
        return 1.0 if fast else (0.5 if correct else 0.0)

    def _revise(actor, target, rng, rich):
        child = copy.deepcopy(target)
        fb = ["%s=%.1f" % (t["name"][:16], develop(target, t, rng)) for t in practice[:2]]
        ap = actor["params"]
        guide = ap.get("revise_strategy", REV0)
        aproc = ("Your procedure: " + ap["procedure"] + "\n") if (rich and ap.get("procedure")) else ""
        keys = '"solve_strategy","revise_strategy","procedure"' if rich else '"solve_strategy","revise_strategy"'
        ask = ("You improve another kernel agent. " + aproc + "Your revise guidance: " + guide +
               "\nTarget currently: solve=%r revise=%r. Practice scores (0 wrong /0.5 correct /1.0 correct&>=1.10x): %s.\n"
               "Return ONLY a JSON object with keys %s giving improved short strings." %
               (target["params"].get("solve_strategy", "")[:120], target["params"].get("revise_strategy", "")[:120], "; ".join(fb), keys))
        m = re.search(r"\{.*\}", gen_fn(ask) or "", re.S)
        changed = False
        if m:
            try:
                d = json.loads(m.group(0))
                for k in (["solve_strategy", "revise_strategy"] + (["procedure"] if rich else [])):
                    if isinstance(d.get(k), str) and 3 < len(d[k]) < 1500 and d[k] != child["params"].get(k):
                        child["params"][k] = d[k]; changed = True
            except Exception:
                pass
        child.setdefault("_prov", []).append({"changed": changed, "rich": rich})
        return child

    return develop, (lambda a, t, r: _revise(a, t, r, False)), (lambda a, t, r: _revise(a, t, r, True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf"); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--blocks", type=int, default=8); ap.add_argument("--anchor-n", type=int, default=3)
    ap.add_argument("--practice-n", type=int, default=2); ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)

    if args.api_model:
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock")); cur.resolve(); cur.resolve_reasoning()
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
    print("V3 CONTROLLED %s blocks=%d" % (who, args.blocks), flush=True)

    narrow, rich = [], []
    st_narrow, st_rich = [], []

    def dec(st):
        n = len(st) or 1
        return {"n_dev": len(st), "correct_rate": round(sum(c for c, f in st) / n, 3), "fast_rate": round(sum(f for c, f in st) / n, 3)}

    def write_partial(done_blocks):
        m = min(len(narrow), len(rich))
        dF1 = _mean_ci([rich[i].F1 - narrow[i].F1 for i in range(m)])
        dF2 = _mean_ci([rich[i].F2 - narrow[i].F2 for i in range(m)])
        out = {"who": who, "blocks_target": args.blocks, "blocks_done": done_blocks,
               "narrow": {**aggregate_lineages(narrow), "decompose": dec(st_narrow)},
               "rich": {**aggregate_lineages(rich), "decompose": dec(st_rich)},
               "dF1_rich_minus_narrow": dF1, "dF2_rich_minus_narrow": dF2}
        json.dump(out, open(os.path.join(args.outdir, "controlled.json"), "w"), indent=2)
        return out

    for b in range(args.blocks):
        practice = G.generate_tiered("Medium", args.practice_n, seed0=b * 400)
        anchors = G.generate_tiered("Medium", args.anchor_n, seed0=10_000_000 + b * 400)
        U0 = {"params": {"solve_strategy": SOLVE0, "revise_strategy": REV0, "procedure": ""}, "skills": []}
        for cond, bucket, st in (("narrow", narrow, st_narrow), ("rich", rich, st_rich)):
            wd = os.path.join(args.outdir, "b%d_%s" % (b, cond)); os.makedirs(wd, exist_ok=True)
            dev, rev_n, rev_r = make_behaviors(gen_fn, wd, practice, st)
            rev = rev_n if cond == "narrow" else rev_r
            r = run_lineage(copy.deepcopy(U0), dev, rev, anchors, random.Random(7000 + b), reps=1)
            bucket.append(r)
            print("b%d %-6s F1=%+.3f N1=%+.3f F2=%+.3f N2=%+.3f" % (b, cond, r.F1, r.N1, r.F2, r.N2), flush=True)
        write_partial(b + 1)   # checkpoint after each completed block (survives creds/box drops)

    out = write_partial(args.blocks)
    print("\n=== CONTROLLED %s ===" % who)
    for cond in ("narrow", "rich"):
        a = out[cond]
        print("  %-6s F1=%s N1=%s F2=%s | %s" % (cond, a["F1"], a["N1"], a["F2"], a["decompose"]))
    print("  dF1(rich-narrow)=%s  dF2=%s" % (out["dF1_rich_minus_narrow"], out["dF2_rich_minus_narrow"]))


if __name__ == "__main__":
    main()
