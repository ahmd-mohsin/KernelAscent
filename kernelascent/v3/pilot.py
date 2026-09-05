"""KernelAscent v3 Stage-5 pilot: run the validated causal lineage instrument on REAL models.

Wires model-backed develop/revise into kernelascent.v3.core.run_lineage:

  develop(actor, project) : the actor's research params + skills produce a kernel ModelNew for a
    downstream project; grade it; bounded attainment C(S) in {0, 0.5, 1.0} =
    wrong / correct / correct-and->=1.10x-vs-compile-baseline.

  revise(actor, target)   : the actor runs its improvement procedure on a COPY of the target
    agent -- it profiles the target's params on a few practice projects, then asks the model to
    propose improved params for the target (JSON). The actor's own meta-strategy biases the
    request, so a better actor should produce a better child. Actor != target throughout.

The instrument (Q/V/F/N, two-link lineage, rescue, aggregation) is unchanged from core.py and
was validated by calibration.py. This pilot just supplies real behaviors. Expect that a
credible result may be a null; the calibration establishes the instrument would detect a
positive control of the tested size.
"""
import os, sys, json, argparse, subprocess, glob, copy, random, re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import gen_source_tasks as G
import curate_bedrock as CB
from kernelascent.v3.core import run_lineage, aggregate_lineages

GRADER = os.path.join(os.path.dirname(HERE), "grade_candidates.py")
FAST_TARGET = 1.10   # attainment requires >=10% over the compile baseline

FOCI = ["fuse elementwise ops into one Triton kernel with tl.load/tl.store",
        "use tl.autotune over block sizes and num_warps",
        "call torch.compile-friendly fused PyTorch ops",
        "tile the reduction and cache in shared memory",
        "minimize memory traffic by fusing the epilogue"]

SOLVE = ("Optimize this PyTorch module for speed on an A100. Keep __init__ identical; rewrite forward.\n"
         "Strategy to apply: {focus}\n{skills}Output exactly ONE class named ModelNew in a single ```python block. No prose.\n\n"
         "```python\n{src}\n```")


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
        return {"correct": False, "best": None}


def make_behaviors(gen_fn, workdir, practice_tasks, cand_timeout=90):
    ctr = {"n": 0}

    def _skillblock(agent):
        sk = agent.get("skills", [])[: int(agent["params"].get("retrieval_k", 0))]
        if not sk:
            return ""
        return "Reusable verified snippets:\n" + "\n".join("```python\n%s\n```" % s["code"] for s in sk) + "\n"

    def develop(agent, project, rng):
        ctr["n"] += 1
        d = os.path.join(workdir, "d%d" % ctr["n"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(project["source"])
        json.dump({k: project[k] for k in ("name", "tier", "family", "meta") if k in project}, open(d + "/meta.json", "w"))
        focus = agent["params"].get("focus", FOCI[0])
        prompt = SOLVE.format(focus=focus, skills=_skillblock(agent), src=project["source"])
        code = CB.extract_modelnew(gen_fn(prompt) or "")
        for old in glob.glob(d + "/cand_*.py"):
            os.remove(old)
        if not code:
            return 0.0
        open(d + "/cand_0.py", "w").write(code)
        r = grade_one(d, cand_timeout)
        correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
        sp = r.get("best_speedup_roofline", 0.0) or 0.0
        if not correct:
            return 0.0
        return 1.0 if sp >= FAST_TARGET else 0.5   # bounded attainment C(S)

    def revise(actor, target, rng):
        child = copy.deepcopy(target)
        # actor profiles the target's params on a couple practice projects (charged model calls)
        fb = []
        for t in practice_tasks[:2]:
            fb.append("%s -> C=%.1f" % (t["name"], develop(target, t, rng)))
        # the actor's own meta-strategy biases how it asks for the target's next params
        meta = actor["params"].get("meta_strategy", "improve the target's focus and retrieval to raise attainment")
        ask = ("You are improving another optimization agent's research parameters. Your guiding "
               "meta-strategy: %s.\nThe target agent currently uses focus=%r, retrieval_k=%d. On practice "
               "projects it scored: %s.\nPropose improved parameters as JSON with keys "
               '"focus" (a concrete optimization strategy string) and "retrieval_k" (int 0-3). '
               "Return only the JSON." % (meta, target["params"].get("focus"), int(target["params"].get("retrieval_k", 0)), "; ".join(fb)))
        raw = gen_fn(ask) or ""
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                if isinstance(d.get("focus"), str) and len(d["focus"]) > 5:
                    child["params"]["focus"] = d["focus"][:200]
                if isinstance(d.get("retrieval_k"), int) and 0 <= d["retrieval_k"] <= 3:
                    child["params"]["retrieval_k"] = d["retrieval_k"]
            except Exception:
                pass
        return child

    return develop, revise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf")
    ap.add_argument("--api-model", default="")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--lineages", type=int, default=4)
    ap.add_argument("--anchor-n", type=int, default=4)
    ap.add_argument("--practice-n", type=int, default=4)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.api_model:
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        cur.resolve(); cur.resolve_reasoning()
        gen_fn = lambda p: cur.generate(p)
        who = "api:%s" % args.api_model
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        def gen_fn(p):
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok([text], return_tensors="pt", padding=True).to("cuda")
            import torch as _t
            with _t.no_grad():
                out = mdl.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7,
                                   top_p=0.9, pad_token_id=tok.pad_token_id)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        who = "hf:%s" % args.model

    print("V3 PILOT %s lineages=%d anchor_n=%d gpu=%s" %
          (who, args.lineages, args.anchor_n, os.environ.get("CUDA_VISIBLE_DEVICES", "?")), flush=True)
    results = []
    for L in range(args.lineages):
        rng = random.Random(1000 + L)
        # independent public practice + private-seed anchors per lineage
        practice = G.generate_tiered("Medium", args.practice_n, seed0=L * 500)
        anchors = G.generate_tiered("Medium", args.anchor_n, seed0=10_000_000 + L * 500)
        wd = os.path.join(args.outdir, "L%d" % L); os.makedirs(wd, exist_ok=True)
        develop, revise = make_behaviors(gen_fn, wd, practice, cand_timeout=90)
        U0 = {"params": {"focus": FOCI[0], "retrieval_k": 0, "meta_strategy":
                         "diagnose why the target's kernels miss the speed target, then set a sharper focus"},
              "skills": []}
        r = run_lineage(U0, develop, revise, anchors, rng, reps=args.reps)
        results.append(r)
        print("L%d Q=%s F1=%.3f N1=%.3f F2=%.3f N2=%.3f" %
              (L, {k: round(v, 3) for k, v in r.Q.items()}, r.F1, r.N1, r.F2, r.N2), flush=True)

    agg = aggregate_lineages(results)
    print("\n=== V3 PILOT AGG %s ===" % who, flush=True)
    for k, v in agg.items():
        print("  %-18s %s" % (k, v), flush=True)
    json.dump({"who": who, "lineages": args.lineages, "agg": agg,
               "per_lineage": [{"Q": r.Q, "F1": r.F1, "N1": r.N1, "F2": r.F2, "N2": r.N2,
                                "q1_minus_q0": r.q1_minus_q0} for r in results]},
              open(os.path.join(args.outdir, "v3_pilot.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
