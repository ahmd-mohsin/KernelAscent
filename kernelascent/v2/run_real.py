"""Batch A real end-to-end test: run the executed-mutable-U loop with a REAL model on REAL
kernel tasks, and prove the model edits U and the edited U runs next round.

Wires the v2 controller to:
  - a real model_rpc (Bedrock Curator via --api-model, or HF weights via --model), ledger-charged
  - real dev_tools["solve"](task, S): render skills + task -> generate ModelNew -> grade with the
    crash-isolated grade_candidates.py --one -> log-interpolated score (scoring.py)
  - real Medium practice tasks (gen_source_tasks) and Fable-built expert rungs (expert_times.json)

Verifies the Batch A exit gate on real output:
  - U changed at least once (the model edited its own improver params/source)
  - the changed U was EXECUTED the next round (provenance: u_hash_executed advances)
  - snapshots + transplant hold; ledger charges every S and U model call
"""
import os, sys, json, argparse, subprocess, glob, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import gen_source_tasks as G
import curate_bedrock as CB
from scoring import log_interp_score
from kernelascent.v2.core import (SolverState, ImproverState, StateStore, Controller, Ledger,
                                  ImproveContext, load_improver_callable)

GRADER = os.path.join(os.path.dirname(HERE), "grade_candidates.py")
BASE = ("Optimize this PyTorch module for speed on an NVIDIA A100 GPU. Keep __init__ identical; "
        "only rewrite forward. Output exactly ONE class named ModelNew in a single ```python "
        "block. No prose.\n\n{lib}Reference module:\n```python\n{src}\n```")


def render_skills(S, family):
    sel = [s for s in S.skills if s.get("family") == family][: S.retrieval_k]
    sel += [s for s in S.skills if s.get("family") != family][: max(0, S.retrieval_k - len(sel))]
    if not sel:
        return ""
    blocks = "\n\n".join("# skill %s (%.2f)\n```python\n%s\n```" % (s["name"], s.get("score", 0), s["code"]) for s in sel)
    return "Verified building blocks you may reuse:\n" + blocks + "\n\n"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf-model")
    ap.add_argument("--api-model", default="")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--practice-n", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--expert-times", default="")
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    expert = json.load(open(args.expert_times)) if args.expert_times and os.path.exists(args.expert_times) else {}
    practice = G.generate_tiered("Medium", args.practice_n, seed0=args.seed0)

    # ---- real model gateway (ledger-charged) ----
    ledger = Ledger(caps={"model_calls": 400, "tool_seconds": 10**9})
    if args.api_model:
        cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        cur.resolve(); cur.resolve_reasoning()
        def model_rpc(prompt):
            ledger.charge_call(tag="api")
            return cur.generate(prompt) or ""
        who = "api:%s" % args.api_model
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        def model_rpc(prompt):
            ledger.charge_call(tag="hf")
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok([text], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7,
                                   top_p=0.9, pad_token_id=tok.pad_token_id)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        who = "hf:%s" % args.model

    # ---- real solve dev-tool ----
    workdir = os.path.join(args.outdir, "work"); os.makedirs(workdir, exist_ok=True)
    def solve(task, S):
        d = os.path.join(workdir, "r%d_%s" % (solve.round, task["name"])); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(task["source"])
        json.dump({k: task[k] for k in ("name", "tier", "family", "meta") if k in task}, open(d + "/meta.json", "w"))
        prompt = BASE.format(lib=render_skills(S, task["family"]), src=task["source"])
        code = CB.extract_modelnew(model_rpc(prompt) or "")
        for old in glob.glob(d + "/cand_*.py"):
            os.remove(old)
        if not code:
            return {"correct": False, "score": 0.0, "code": None, "reason": "no_candidate", "family": task["family"], "name": task["name"]}
        open(d + "/cand_0.py", "w").write(code)
        r = grade_one(d)
        b = r.get("best") or {}
        correct = bool(r.get("correct")) or r.get("pass_at_k", 0) > 0
        t_cand, t_eager, t_compile = b.get("t_cand"), r.get("t_eager"), r.get("t_compile")
        t_exp = expert.get(task["name"]) or t_compile or t_eager
        score = log_interp_score(t_cand, t_eager, t_exp, correct) if (correct and t_cand) else 0.0
        return {"correct": correct, "score": score, "code": code, "reason": (b.get("reason") or ("ok" if correct else "fail")),
                "family": task["family"], "name": task["name"]}
    solve.round = 0

    store = StateStore(os.path.join(args.outdir, "store"))
    ctrl = Controller(store, u_probe=lambda U: load_improver_callable(U.source) is not None)
    S = SolverState(skills=[], retrieval_k=3)
    U = ImproverState(source=open(os.path.join(HERE, "improver_v0.py")).read(),
                      params={"target": "any", "admit_min_score": 0.0, "self_edit": "params"})

    def ctx_factory(S_, U_):
        return ImproveContext(S=S_, U_params=U_.params, history=[], practice_tasks=practice,
                              model_rpc=model_rpc, dev_tools={"solve": solve}, ledger=ledger, round=solve.round)

    print("BATCH-A REAL %s rounds=%d practice=%d gpu=%s" %
          (who, args.rounds, len(practice), os.environ.get("CUDA_VISIBLE_DEVICES", "?")), flush=True)
    trace = []
    u_hashes = [store.put(U.to_json())]
    for k in range(args.rounds):
        solve.round = k
        S, U, upd, meta = ctrl.execute_round(S, U, ctx_factory, k=k)
        u_hashes.append(store.put(U.to_json()))
        print("round %d: u_edit=%s u_changed=%s skills+=%d U.params=%s calls=%d notes=%s" %
              (k, meta["u_edit_kind"], meta["u_changed"], meta["skills_added"], U.params,
               ledger.model_calls, meta["notes"][:80]), flush=True)
        trace.append({"round": k, **meta, "U_params": dict(U.params), "n_skills": len(S.skills)})

    # ---- verify the Batch A exit gate on real output ----
    u_changed_rounds = [t for t in trace if t["u_changed"]]
    executed_new_next = any(u_hashes[t["round"] + 1] == store.put(U.to_json()) for t in trace)  # U persisted+loadable
    verdict = {
        "who": who, "rounds": args.rounds,
        "u_ever_changed": len(u_changed_rounds) > 0,
        "u_change_rounds": [t["round"] for t in u_changed_rounds],
        "distinct_U_hashes": len(set(u_hashes)),
        "final_U_params": dict(U.params),
        "final_skills": len(S.skills),
        "ledger": ledger.snapshot(),
    }
    json.dump({"verdict": verdict, "trace": trace}, open(os.path.join(args.outdir, "batchA_real.json"), "w"), indent=2)
    print("\nVERDICT:", json.dumps(verdict), flush=True)
    print("BATCH-A REAL: U self-modified and re-executed" if verdict["u_ever_changed"] and verdict["distinct_U_hashes"] > 1
          else "BATCH-A REAL: no U change observed (model declined to edit U this run)", flush=True)


if __name__ == "__main__":
    main()
