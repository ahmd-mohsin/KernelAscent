"""E1 + E2 causal diagnostics for KernelAscent v3, built on the validated estimators in core.py.

E1 (sensitivity + null FPR): inject an independently-verified reference improvement and measure
  (a) USEFULNESS  dQ = Q(U0 with ref in the SOLVE side) - Q(U0)            -- does the ref help develop
  (b) CHANNEL     F_inj = V(U0+ref_in_REVISE, T) - V(U0, T) on a COMMON T  -- does USING it in revise
                                                                              make a better improver
  (c) NULL/FPR    same two contrasts with a COSMETIC (semantically-identical) reword -> should be ~0
  Report per-block paired effects + 95% CIs; the null distribution gives the false-positive rate.

E2 (primary causal: hold immediate usefulness fixed, toggle self-use): two actors are IDENTICAL
  except whether their revise step USES the verified improvement. Both revise the SAME common target
  T (immediate develop-usefulness is therefore identical; only self-use differs).
    F_selfuse = V(actor_uses_ref_in_revise, T) - V(actor_same_but_base_revise, T)
    N_selfuse = V(self_on, self_on) - Q(self_on)     (live-child value beyond the unchanged target)
  Report per-block + CIs + a rescue/rollback check.

Two run modes:
  --calib     deterministic scripted worlds (no GPU/API) that PROVE the diagnostics detect an
              injected effect and return ~0 (within noise) for a genuine null -> validates logic.
  --model / --api-model   real model-backed develop/revise (mirrors controlled.py), lineage blocks.
"""
import os, sys, json, argparse, subprocess, glob, copy, random, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import estimate_Q, estimate_V, _mean_ci

GRADER = os.path.join(os.path.dirname(HERE), "grade_candidates.py")
FAST = 1.10

BASE_SOLVE = "write a fused Triton kernel or fused PyTorch, numerically equivalent, aiming to beat torch.compile"
BASE_REVISE = "diagnose why the target's kernels miss the speed target, then propose a sharper solve strategy and a better revise strategy"

# Independently-authored reference improvement (verified empirically via the E1 usefulness check).
REF_SOLVE = ("Write ONE fused Triton kernel: tl.load inputs, accumulate in fp32, tl.store in the input dtype; "
             "choose BLOCK to cover the reduction/row; use tl.dot for matmul; fuse the epilogue (bias/activation) "
             "into the same kernel so there is a single memory pass.")
REF_REVISE = ("Diagnose the target's failures precisely: if numerically wrong, accumulate in fp32 and cast on store; "
              "if correct but not faster than torch.compile, fuse the elementwise epilogue into the matmul/reduction "
              "kernel and autotune BLOCK/num_warps. Rewrite solve_strategy to name the specific fusion and dtype handling.")
# Cosmetic reword (same meaning, different surface form) -> the null for FPR.
NULL_SOLVE = ("Produce a single merged Triton kernel that reads its inputs, does the math in 32-bit float, and writes "
              "back in the original dtype; size the block to span the row; employ tl.dot for the matmul; and roll the "
              "trailing bias/activation into that one kernel to keep it to a single pass over memory.")
NULL_REVISE = ("Work out exactly where the target goes wrong: for numeric errors, sum in fp32 then cast when storing; "
               "for a correct-but-slow kernel, merge the pointwise tail into the matmul/reduction and sweep block/warp "
               "counts. Then restate solve_strategy calling out that same fusion and dtype treatment.")


# ----------------------------------------------------------------------------- deterministic calib
def calib_behaviors(revise_relevant=True, noise=0.0):
    """Scripted world. Agent = {skill, has_ref_solve, has_ref_revise}. A ref helps develop via
    has_ref_solve (+0.3) and helps the IMPROVER via has_ref_revise (+0.4) ONLY when revise_relevant."""
    def develop(agent, project, rng):
        v = agent["skill"] + (0.30 if agent.get("has_ref_solve") else 0.0)
        if noise:
            v += rng.uniform(-noise, noise)
        return max(0.0, min(1.0, v))

    def revise(actor, target, rng):
        child = copy.deepcopy(target)
        power = 0.10 + ((0.40 if actor.get("has_ref_revise") else 0.0) if revise_relevant else 0.0)
        child["skill"] = min(1.0, target["skill"] + power)
        return child
    return develop, revise


def run_calib():
    rng = random.Random(0)
    anchors = [{"id": i} for i in range(4)]
    U0 = {"skill": 0.2, "has_ref_solve": False, "has_ref_revise": False}
    ok = True

    def q(u, dev):
        return estimate_Q(u, anchors, dev, rng, reps=3)

    def v(actor, target, dev, rev):
        return estimate_V(actor, target, rev, anchors, dev, rng, reps=3)

    # --- world where the improvement is relevant to the improver ---
    dev, rev = calib_behaviors(revise_relevant=True)
    # E1a usefulness: injecting ref_solve raises Q
    dQ = q({**U0, "has_ref_solve": True}, dev) - q(U0, dev)
    # E1b channel: injecting ref_revise into the actor raises V on a common target
    T = {"skill": 0.3}
    F_inj = v({**U0, "has_ref_revise": True}, T, dev, rev) - v(U0, T, dev, rev)
    # E1 null: cosmetic ref that sets NO real flag -> ~0 for both
    dQ_null = q({**U0, "cosmetic": True}, dev) - q(U0, dev)
    F_null = v({**U0, "cosmetic": True}, T, dev, rev) - v(U0, T, dev, rev)
    # E2: hold develop-usefulness fixed (both have ref_solve), toggle self-use of ref_revise
    self_on = {**U0, "has_ref_solve": True, "has_ref_revise": True}
    self_off = {**U0, "has_ref_solve": True, "has_ref_revise": False}
    F_self = v(self_on, T, dev, rev) - v(self_off, T, dev, rev)
    N_self = v(self_on, self_on, dev, rev) - q(self_on, dev)

    # --- world where the improvement does NOT help the improver (develop-only) ---
    dev0, rev0 = calib_behaviors(revise_relevant=False)
    F_self_devonly = (v({**U0, "has_ref_solve": True, "has_ref_revise": True}, T, dev0, rev0)
                      - v({**U0, "has_ref_solve": True, "has_ref_revise": False}, T, dev0, rev0))

    checks = [
        ("E1a usefulness dQ > 0.2", dQ, dQ > 0.2),
        ("E1b channel F_inj > 0.2", F_inj, F_inj > 0.2),
        ("E1 null dQ ~ 0", dQ_null, abs(dQ_null) < 1e-6),
        ("E1 null F ~ 0", F_null, abs(F_null) < 1e-6),
        ("E2 F_selfuse > 0.2 (revise-relevant)", F_self, F_self > 0.2),
        ("E2 N_selfuse > 0", N_self, N_self > 0.0),
        ("E2 F_selfuse ~ 0 (develop-only ref)", F_self_devonly, abs(F_self_devonly) < 1e-6),
    ]
    print("=== E1/E2 CALIBRATION (deterministic) ===")
    for name, val, passed in checks:
        ok = ok and passed
        print("  [%s] %-42s = %+.3f" % ("PASS" if passed else "FAIL", name, val))
    print("CALIB", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- real model behaviors
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
    ctr = {"n": 0}

    def develop(agent, project, rng):
        ctr["n"] += 1
        d = os.path.join(workdir, "d%d" % ctr["n"]); os.makedirs(d, exist_ok=True)
        open(d + "/task.py", "w").write(project["source"])
        json.dump({k: project[k] for k in ("name", "tier", "family", "meta") if k in project}, open(d + "/meta.json", "w"))
        p = agent["params"]
        proc = ("Procedure: " + p["procedure"] + "\n") if p.get("procedure") else ""
        prompt = ("Optimize this PyTorch module for speed on an A100. Keep __init__ identical; rewrite forward.\n"
                  + proc + "Strategy: " + p.get("solve_strategy", BASE_SOLVE) +
                  "\nOutput exactly ONE class named ModelNew in a single ```python block. No prose.\n\n```python\n"
                  + project["source"] + "\n```")
        code = _extract(gen_fn(prompt) or "")
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

    def revise(actor, target, rng):
        child = copy.deepcopy(target)
        fb = ["%s=%.1f" % (t["name"][:16], develop(target, t, rng)) for t in practice[:2]]
        ap = actor["params"]
        guide = ap.get("revise_strategy", BASE_REVISE)
        aproc = ("Your procedure: " + ap["procedure"] + "\n") if ap.get("procedure") else ""
        ask = ("You improve another kernel agent. " + aproc + "Your revise guidance: " + guide +
               "\nTarget currently: solve=%r revise=%r. Practice scores (0 wrong /0.5 correct /1.0 correct&>=1.10x): %s.\n"
               "Return ONLY a JSON object with keys \"solve_strategy\",\"revise_strategy\",\"procedure\" giving improved short strings." %
               (target["params"].get("solve_strategy", "")[:120], target["params"].get("revise_strategy", "")[:120], "; ".join(fb)))
        m = re.search(r"\{.*\}", gen_fn(ask) or "", re.S)
        if m:
            try:
                dd = json.loads(m.group(0))
                for k in ("solve_strategy", "revise_strategy", "procedure"):
                    if isinstance(dd.get(k), str) and 3 < len(dd[k]) < 1500:
                        child["params"][k] = dd[k]
            except Exception:
                pass
        return child

    return develop, revise


def _extract(txt):
    import curate_bedrock as CB
    return CB.extract_modelnew(txt)


def make_agent(solve, revise, procedure=""):
    return {"params": {"solve_strategy": solve, "revise_strategy": revise, "procedure": procedure}, "skills": []}


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
    print("E1/E2 REAL %s blocks=%d" % (who, args.blocks), flush=True)

    E1 = {"dQ_useful": [], "F_channel": [], "dQ_null": [], "F_null": []}
    E2 = {"F_selfuse": [], "N_selfuse": [], "rescue_minus_revert": []}
    st = []

    def dump(done):
        out = {"who": who, "blocks_target": args.blocks, "blocks_done": done,
               "E1": {k: _mean_ci(v) for k, v in E1.items()},
               "E2": {k: _mean_ci(v) for k, v in E2.items()},
               "decompose": {"n_dev": len(st),
                             "correct_rate": round(sum(c for c, f in st) / (len(st) or 1), 3),
                             "fast_rate": round(sum(f for c, f in st) / (len(st) or 1), 3)}}
        json.dump(out, open(os.path.join(args.outdir, "e1e2.json"), "w"), indent=2)
        return out

    for b in range(args.blocks):
        practice = G.generate_tiered("Medium", 2, seed0=b * 400)
        anchors = G.generate_tiered("Medium", args.anchor_n, seed0=10_000_000 + b * 400)
        wd = os.path.join(args.outdir, "b%d" % b); os.makedirs(wd, exist_ok=True)
        develop, revise = real_behaviors(gen_fn, wd, practice, st)
        rng = random.Random(9000 + b)

        U0 = make_agent(BASE_SOLVE, BASE_REVISE)
        U_solve = make_agent(REF_SOLVE, BASE_REVISE, procedure=REF_SOLVE)      # ref on the SOLVE side
        U_revise = make_agent(BASE_SOLVE, REF_REVISE, procedure=REF_REVISE)    # ref on the REVISE side
        U_null_s = make_agent(NULL_SOLVE, BASE_REVISE, procedure=NULL_SOLVE)   # cosmetic reword (solve)
        U_null_r = make_agent(BASE_SOLVE, NULL_REVISE, procedure=NULL_REVISE)  # cosmetic reword (revise)

        # E1a usefulness (solve-side ref raises Q); E1b channel (revise-side ref raises V on common T)
        T = revise(U0, copy.deepcopy(U0), rng)          # a realistic campaign target
        E1["dQ_useful"].append(estimate_Q(U_solve, anchors, develop, rng, reps=1) - estimate_Q(U0, anchors, develop, rng, reps=1))
        E1["F_channel"].append(estimate_V(U_revise, T, revise, anchors, develop, rng, reps=1) - estimate_V(U0, T, revise, anchors, develop, rng, reps=1))
        E1["dQ_null"].append(estimate_Q(U_null_s, anchors, develop, rng, reps=1) - estimate_Q(U0, anchors, develop, rng, reps=1))
        E1["F_null"].append(estimate_V(U_null_r, T, revise, anchors, develop, rng, reps=1) - estimate_V(U0, T, revise, anchors, develop, rng, reps=1))

        # E2 hold-usefulness-fixed, toggle self-use: both identical except revise uses ref or base
        self_on = make_agent(REF_SOLVE, REF_REVISE, procedure=REF_REVISE)
        self_off = make_agent(REF_SOLVE, BASE_REVISE, procedure="")            # same develop benefit, base revise
        v_on = estimate_V(self_on, T, revise, anchors, develop, rng, reps=1)
        v_off = estimate_V(self_off, T, revise, anchors, develop, rng, reps=1)
        E2["F_selfuse"].append(v_on - v_off)
        E2["N_selfuse"].append(estimate_V(self_on, self_on, revise, anchors, develop, rng, reps=1) - estimate_Q(self_on, anchors, develop, rng, reps=1))
        # rescue: restore self_on after a rollback, regenerate; compare to the base-revise revert
        resc = estimate_V(copy.deepcopy(self_on), T, revise, anchors, develop, rng, reps=1)
        E2["rescue_minus_revert"].append(resc - v_off)

        print("b%d dQ=%.3f Fchan=%.3f dQnull=%.3f Fnull=%.3f | Fself=%.3f Nself=%.3f" %
              (b, E1["dQ_useful"][-1], E1["F_channel"][-1], E1["dQ_null"][-1], E1["F_null"][-1],
               E2["F_selfuse"][-1], E2["N_selfuse"][-1]), flush=True)
        dump(b + 1)

    out = dump(args.blocks)
    print("\n=== E1/E2 %s ===" % who)
    print(" E1:", json.dumps(out["E1"]))
    print(" E2:", json.dumps(out["E2"]))
    print(" decompose:", out["decompose"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--model", default=""); ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--blocks", type=int, default=8); ap.add_argument("--anchor-n", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=8192); ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    if args.calib:
        sys.exit(run_calib())
    assert args.outdir, "need --outdir for a real run"
    os.makedirs(args.outdir, exist_ok=True)
    run_real(args)


if __name__ == "__main__":
    main()
