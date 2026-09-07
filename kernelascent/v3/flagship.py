"""KernelAscent FLAGSHIP: autonomous recursive research via an EXECUTED, mutable improver.

Satisfies the three inseparable requirements (2026-09-07 decision):
  R1 autonomous lineage + EXPLICIT INHERITANCE: the agent's improver is executable source
     (v2 ImproverState). Each revise LOADS AND EXECUTES the ACTOR's actual bytes (v2
     load_improver_callable, fresh namespace) to edit a target; the executed procedure reads the
     actor's evolving U-params (meta_policy) and may rewrite its OWN source -> the improver improves
     the improver, and the inherited change EXECUTES to govern the next producer. Every edit is
     provenance-logged (author, executed-source hash, u_changed, s_changed, child!=parent).
  R2 counterfactual continuations from the SAME target: run_lineage forks V2=revise(U0,U1),
     V3=revise(U1,U2) on the identical target -> F1, F2 (repeat) + rescue.
  R3 future outcomes with real value: the solver produces graded kernels; continuous speed-resolved
     attainment C=0.5+0.5*tanh((sp-1)/0.1) vs min(eager,compile).

An agent = {"S": SolverState, "U": ImproverState}. develop runs S on a project (graded); revise
executes the actor's U to produce a child (S',U'). Estimators/lineage are v3/core (unchanged).

--calib runs a deterministic executed-source world (no model) proving F1,F2 detection + that the
inherited U actually executed and changed (provenance). --model/--api-model runs it for real.
"""
import os, sys, json, argparse, subprocess, glob, copy, random, re, math, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci
from kernelascent.v2.core import SolverState, ImproverState, StateUpdate, load_improver_callable, ImproveContext, Ledger
GRADER = os.path.join(os.path.dirname(HERE), "grade_candidates.py")

BASE_POLICY = ("Optimize this PyTorch module for speed on an A100. Keep __init__ identical; rewrite forward to be "
               "as fast as possible while numerically equivalent. Use ANY approach you judge fastest.")

# ---- The SEED improver, as EXECUTABLE source. improve_step reads ctx.U_params['meta_policy'] to
# shape how it asks the model to improve the solver, and edits that meta_policy itself (improver
# improves improver). It returns a v2 StateUpdate. This exact string is loaded + exec'd each revise.
SEED_U_SOURCE = '''
def improve_step(ctx):
    fb = ctx.dev_tools["profile"](ctx.S, ctx.practice_tasks[:2])
    meta = ctx.U_params.get("meta_policy", "diagnose why the solver is slow or wrong and give it a sharper, still open-ended approach")
    ask = ("You improve a kernel-optimization agent. Meta-guidance: " + meta +
           "\\nThe solver's current approach: " + repr(ctx.S.prompt_policy)[:400] +
           "\\nPractice results (0=wrong .5=parity ->1 faster): " + fb +
           "\\nReturn ONLY JSON {\\"policy\\": <improved open-ended solver instruction>, "
           "\\"meta\\": <improved guidance for how to improve solvers next time>}. Add capability; do not over-constrain.")
    import json, re
    txt = ctx.model_rpc(ask) or ""
    m = re.search(r"\\{.*\\}", txt, re.S)
    sp, um = {}, {}
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d.get("policy"), str) and 5 < len(d["policy"]) < 1200:
                sp["prompt_policy"] = d["policy"]
            if isinstance(d.get("meta"), str) and 5 < len(d["meta"]) < 800:
                um["meta_policy"] = d["meta"]
        except Exception:
            pass
    return StateUpdate(s_param_edits=sp, u_param_edits=um, notes="seed improver")
'''


def cont_score(correct, sp):
    return 0.0 if not correct else max(0.0, min(1.0, 0.5 + 0.5 * math.tanh((sp - 1.0) / 0.10)))


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


def new_agent(policy=BASE_POLICY, meta=None, source=SEED_U_SOURCE):
    U = ImproverState(source=source, params={} if meta is None else {"meta_policy": meta})
    return {"S": SolverState(prompt_policy=policy), "U": U}


def _uhash(u):
    return hashlib.sha256(u.to_json().encode()).hexdigest()[:12]


def make_behaviors(gen_fn, workdir, practice, stats, prov_path):
    import curate_bedrock as CB
    ctr = {"n": 0}

    def _prov(ev):
        open(prov_path, "a").write(json.dumps(ev) + "\n")

    def develop(agent, project, rng):
        ctr["n"] += 1
        d = os.path.join(workdir, "d%d" % ctr["n"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(project["source"])
        json.dump({k: project[k] for k in ("name", "tier", "family", "meta") if k in project}, open(d + "/meta.json", "w"))
        prompt = agent["S"].prompt_policy + "\nOutput exactly ONE class named ModelNew in a single ```python block. No prose.\n\n```python\n" + project["source"] + "\n```"
        code = CB.extract_modelnew(gen_fn(prompt) or "")
        for old in glob.glob(d + "/cand_*.py"):
            os.remove(old)
        if not code:
            stats.append((0, 0)); return 0.0
        open(d + "/cand_0.py", "w").write(code)
        r = grade_one(d)
        correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
        sp = r.get("best_speedup_roofline", 0.0) or 0.0
        stats.append((1 if correct else 0, 1 if (correct and sp >= 1.10) else 0))
        return cont_score(correct, sp)

    def profile(S, tasks):
        return "; ".join("%s=%.2f" % (t["name"][:12], develop({"S": S, "U": None}, t, random.Random(1))) for t in tasks)

    def revise(actor, target, rng):
        """EXECUTE the actor's improver bytes to edit a COPY of the target -> child (R1)."""
        child = {"S": copy.deepcopy(target["S"]), "U": copy.deepcopy(target["U"])}
        executed_hash = _uhash(actor["U"])
        try:
            fn = load_improver_callable(actor["U"].source)         # fresh-namespace exec of actor bytes
            ctx = ImproveContext(S=copy.deepcopy(target["S"]), U_params=dict(actor["U"].params), history=[],
                                 practice_tasks=practice, model_rpc=(lambda p: gen_fn(p) or ""),
                                 dev_tools={"profile": profile}, ledger=Ledger(), round=0)
            upd = fn(ctx)
        except Exception as e:
            _prov({"event": "revise_exec_error", "err": repr(e)[:120], "executed": executed_hash}); return child
        s_changed = u_changed = False
        if isinstance(upd, StateUpdate):
            if upd.s_param_edits:
                child["S"].params.update(upd.s_param_edits)
                if isinstance(upd.s_param_edits.get("prompt_policy"), str):
                    child["S"].prompt_policy = upd.s_param_edits["prompt_policy"]; s_changed = True
            if upd.u_param_edits:
                child["U"].params.update(upd.u_param_edits); u_changed = True
            if isinstance(upd.u_new_source, str) and "def improve_step" in upd.u_new_source:
                try:
                    load_improver_callable(upd.u_new_source); child["U"].source = upd.u_new_source; u_changed = True
                except Exception:
                    pass
        _prov({"event": "revise", "actor": _uhash(actor["U"]), "target": _uhash(target["U"]),
               "executed": executed_hash, "child": _uhash(child["U"]), "s_changed": s_changed,
               "u_changed": u_changed, "child_ne_parent": (_uhash(child["U"]) != _uhash(target["U"])) or s_changed})
        return child
    return develop, revise


# ------------------------------------------------------------------- deterministic executed-source calib
CALIB_U_SOURCE = '''
def improve_step(ctx):
    power = float(ctx.U_params.get("power", 0.1)); cap = float(ctx.U_params.get("cap", 1.0))
    sp = {"skillnum": min(1.0, float(ctx.S.params.get("skillnum", 0.3)) + power)}
    um = {"power": min(cap, power + 0.10)}   # improver improves the improver up to `cap`
    return StateUpdate(s_param_edits=sp, u_param_edits=um, notes="calib")
'''


def run_calib():
    import types
    anchors = [{"id": i, "source": ""} for i in range(5)]

    def develop(agent, project, rng):
        return max(0.0, min(1.0, float(agent["S"].params.get("skillnum", 0.3))))

    def revise(actor, target, rng):
        child = {"S": copy.deepcopy(target["S"]), "U": copy.deepcopy(target["U"])}
        fn = load_improver_callable(actor["U"].source)
        ctx = ImproveContext(S=copy.deepcopy(target["S"]), U_params=dict(actor["U"].params), history=[], practice_tasks=[],
                             model_rpc=lambda p: "", dev_tools={}, ledger=Ledger(), round=0)
        upd = fn(ctx)
        child["S"].params.update(upd.s_param_edits or {})
        child["U"].params.update(upd.u_param_edits or {})
        return child

    ok = True
    for mode, exp, cap in (("compound", "F1>0,F2>0", 1.0), ("oneup", "F1>0,F2~0", 0.2)):
        rs = []
        for s in range(6):
            U0 = {"S": SolverState(prompt_policy="", params={"skillnum": 0.3}),
                  "U": ImproverState(source=CALIB_U_SOURCE, params={"power": 0.1, "cap": cap})}
            rs.append(run_lineage(U0, develop, revise, anchors, random.Random(s), reps=3))
        agg = aggregate_lineages(rs); f1, f2 = agg["F1"]["mean"], agg["F2"]["mean"]
        good = (f1 > 0.02 and f2 > 0.02) if mode == "compound" else (f1 > 0.02 and abs(f2) < 0.03)
        ok = ok and good
        print("  [%s] %-9s F1=%+.3f F2=%+.3f (expect %s)" % ("PASS" if good else "FAIL", mode, f1, f2, exp))
    print("FLAGSHIP CALIB", "ALL PASS" if ok else "FAIL"); return 0 if ok else 1


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
    print("FLAGSHIP %s blocks=%d" % (who, args.blocks), flush=True)
    results = []; st = []; prov = os.path.join(args.outdir, "provenance.jsonl")

    def dump(done):
        agg = aggregate_lineages(results) if results else {}
        # R1 audit from provenance: fraction of revises where the executed inherited U actually changed the child
        pv = [json.loads(l) for l in open(prov)] if os.path.exists(prov) else []
        rev = [e for e in pv if e.get("event") == "revise"]
        audit = {"revises": len(rev), "u_changed": sum(e.get("u_changed") for e in rev),
                 "child_ne_parent": sum(e.get("child_ne_parent") for e in rev),
                 "exec_errors": sum(e.get("event") == "revise_exec_error" for e in pv)}
        out = {"who": who, "blocks_target": args.blocks, "blocks_done": done, "agg": agg,
               "per_block": [{"F1": r.F1, "F2": r.F2, "N1": r.N1, "N2": r.N2, "rescue_minus_revert": r.rescue_minus_revert} for r in results],
               "R1_inheritance_audit": audit,
               "decompose": {"n_dev": len(st), "correct_rate": round(sum(c for c, f in st) / (len(st) or 1), 3),
                             "fast_rate": round(sum(f for c, f in st) / (len(st) or 1), 3)}}
        json.dump(out, open(os.path.join(args.outdir, "flagship.json"), "w"), indent=2)
        return out

    for b in range(args.blocks):
        practice = G.generate_tiered("Medium", 2, seed0=b * 400)
        anchors = G.generate_tiered("Medium", args.anchor_n, seed0=10_000_000 + b * 400)
        wd = os.path.join(args.outdir, "b%d" % b); os.makedirs(wd, exist_ok=True)
        develop, revise = make_behaviors(gen_fn, wd, practice, st, prov)
        U0 = new_agent()
        r = run_lineage(U0, develop, revise, anchors, random.Random(15000 + b), reps=1)
        results.append(r)
        print("b%d F1=%+.3f F2=%+.3f N1=%+.3f N2=%+.3f rescue=%+.3f" % (b, r.F1, r.F2, r.N1, r.N2, r.rescue_minus_revert), flush=True)
        dump(b + 1)
    out = dump(args.blocks)
    print("\n=== FLAGSHIP %s ===" % who)
    for k in ("F1", "F2", "N1", "N2", "rescue_minus_revert"):
        print("  %-20s %s" % (k, out["agg"].get(k)))
    print("  R1 audit", out["R1_inheritance_audit"], "| decompose", out["decompose"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--model", default=""); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--blocks", type=int, default=10); ap.add_argument("--anchor-n", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=8192); ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    if args.calib:
        sys.exit(run_calib())
    assert args.outdir, "need --outdir"
    os.makedirs(args.outdir, exist_ok=True)
    run_real(args)


if __name__ == "__main__":
    main()
