"""Task 5 (self-play) — GENUINELY SELF-REFERENTIAL RSI.

The distinction from the external-curriculum version: here the SAME improving model both SOLVES and PROPOSES.
Each round the current model (adapter ON) generates its own strictly-harder task variants; as it gets better,
its proposals get harder too — the solver drives its own curriculum. That is self-referential recursion, not an
externally-ratcheted difficulty.

Two arms from the same base at EQUAL budget, scored every round on ONE fixed held-out ladder:
  SELFPLAY : model proposes its own harder tasks (validated) -> frontier grows -> self-SFT on solved.
  STATIC   : fixed seed frontier -> self-SFT (control).
PRIMARY = C_held(SELFPLAY) - C_held(STATIC). Sustained >0 = the model improving its OWN challenge compounds.

Rigor: proposals are GPU-validated + deduped (novelty pressure); we log `model_proposed` vs `backstop`
(programmatic shape-scale used ONLY if the model proposes nothing that validates, so a stall is visible not
hidden); escalation verified via frozen-base solve-rate on new tasks; fixed disjoint held ladder; headroom score.
If the model cannot bootstrap its own curriculum (model_proposed stays ~0), that is itself a real finding.

Output selfplay_rsi.json: per round C_held_selfplay, C_held_static, delta, frontier, model_proposed, backstop, base_solve.
"""
import os, sys, json, argparse, random, statistics, time, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch

PROPOSE = ("You are given a PyTorch module optimization task. Write a NEW, STRICTLY HARDER task in EXACTLY the "
           "same format: define `DT`, a `class Model(nn.Module)` (with __init__ and forward), and `def get_inputs()` "
           "returning a list with one DT tensor. Make it harder via larger shapes and/or one extra fused stage "
           "(deeper matmul+epilogue, bigger seq/heads, extra norm/residual). Numerically well-defined, deterministic, "
           "self-contained (import torch, torch.nn as nn). Return ONLY one python code block.\n\n```python\n{src}\n```")


def _extract_task(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    c = (m.group(1) if m else (text or "")).strip()
    return c if ("class Model" in c and "get_inputs" in c and "DT" in c) else None


def _valid(src):
    try:
        ref, x, gold, e = AB.build_ref(src); return AB.time_fn(lambda z: ref(z), (x,)) > 0 and torch.isfinite(gold).all().item()
    except Exception:
        return False


def _scale_shapes(src, factor=2):                    # programmatic backstop only
    m = re.search(r"def get_inputs\(\).*?return\s*\[(.*?)\]", src, re.S)
    if not m: return None
    seg = m.group(0)
    new = re.sub(r"\b\d{2,}\b", lambda mm: str(min(8192, max(8, (int(int(mm.group(0)) * factor) // 8) * 8))), seg)
    return src.replace(seg, new) if new != seg else None


def model_propose(tok, mdl, solved, want, seen):
    """The IMPROVING MODEL proposes harder tasks (adapter ON). Returns (new_valid_tasks, n_model, n_backstop)."""
    out = []; n_model = 0; n_back = 0
    prompts = [PROPOSE.format(src=s) for s in solved[:max(want * 2, 8)]]
    if prompts:
        gens = W.generate_batch(tok, mdl, prompts, 2, max_new=1200, adapter=True)   # model authors the task
        for gl in gens:
            for g in gl:
                if len(out) >= want: break
                code = _extract_task(g)
                if not code: continue
                key = hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()
                if key not in seen and _valid(code):
                    seen.add(key); out.append(code); n_model += 1
    for s in solved:                                  # backstop ONLY to fill the remainder (logged separately)
        if len(out) >= want: break
        c = _scale_shapes(s)
        if c:
            key = hashlib.sha1(re.sub(r"\s+", "", c).encode()).hexdigest()
            if key not in seen and _valid(c):
                seen.add(key); out.append(c); n_back += 1
    return out, n_model, n_back


def _score(tok, mdl, srcs, k, adapter):
    gl = W.generate_batch(tok, mdl, srcs, k, adapter=adapter)
    codes = [[c for c in (AB.extract_modelnew(t) for t in g) if c] for g in gl]
    grades = W._grade_isolated_batch(list(zip(srcs, codes)))
    uc = os.environ.get("KA_SCORE", "eager") == "compiled"
    sc = []; corr = []; pairs = []
    for src, cs, res in zip(srcs, codes, grades):
        best = 0.0; bc = None; ok_any = 0
        for code, gg in zip(cs, res):
            ok, se, scp, ceil = (list(gg) + [0, 0, 0, 1.5])[:4]
            s = LK._score(ok, scp if uc else se, ceiling=(ceil if uc else 1.5))
            best = max(best, s)
            if ok: ok_any = 1; bc = bc or code
        sc.append(best); corr.append(ok_any)
        if bc: pairs.append((src, bc))
    return (statistics.mean(sc) if sc else 0.0), (statistics.mean(corr) if corr else 0.0), pairs


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    spg = [int(x) for x in str(args.selfplay_gpu).split(",")]; stg = [int(x) for x in str(args.static_gpu).split(",")]
    tok, sp_m = W.build(args.model, spg)
    tok2, st_m = W.build(args.model, stg)
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    seed_tasks = [LK.TASKS[n] for n in names[:args.seed_tasks]]
    held = [LK.TASKS[n] for n in names[args.seed_tasks:args.seed_tasks + args.held]]
    sp_frontier = list(seed_tasks); st_frontier = list(seed_tasks)
    seen = {hashlib.sha1(re.sub(r"\s+", "", s).encode()).hexdigest() for s in seed_tasks + held}
    print("SELFPLAY-RSI %s seed=%d seed_tasks=%d held=%d rounds=%d (model proposes its own tasks)" %
          (args.model, args.seed, len(seed_tasks), len(held), args.rounds), flush=True)
    os.makedirs(args.outdir, exist_ok=True); hist = []
    for r in range(args.rounds):
        t0 = time.time()
        _, _, sp_pairs = _score(tok, sp_m, sp_frontier, args.k, adapter=True)
        W.sft(tok, sp_m, sp_pairs, args.sft_steps)
        added, n_model, n_back = model_propose(tok, sp_m, [s for s, _ in sp_pairs] or sp_frontier, args.propose, seen)
        sp_frontier += added
        _, _, st_pairs = _score(tok2, st_m, st_frontier, args.k, adapter=True)
        W.sft(tok2, st_m, st_pairs, args.sft_steps)
        Csp, _, _ = _score(tok, sp_m, held, args.k, adapter=True)
        Cst, _, _ = _score(tok2, st_m, held, args.k, adapter=True)
        _, base_corr, _ = _score(tok, sp_m, added or sp_frontier[-4:], args.k, adapter=False)
        row = {"round": r, "C_held_selfplay": round(Csp, 3), "C_held_static": round(Cst, 3),
               "delta_selfplay_minus_static": round(Csp - Cst, 3), "frontier": len(sp_frontier),
               "added": len(added), "model_proposed": n_model, "backstop": n_back,
               "base_correct_on_new": round(base_corr, 3)}
        hist.append(row)
        print("round %d C_held sp=%.3f static=%.3f | sp-static=%+.3f frontier=%d(+%d model=%d back=%d) base=%.2f (%.0fs)" %
              (r, Csp, Cst, Csp - Cst, len(sp_frontier), len(added), n_model, n_back, base_corr, time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "held": len(held), "history": hist,
                   "note": "SELF-PLAY: the improving model authors its own harder tasks; primary=delta_selfplay_minus_static"},
                  open(os.path.join(args.outdir, "selfplay_rsi.json"), "w"), indent=2)
    dl = [h["delta_selfplay_minus_static"] for h in hist]; mp = sum(h["model_proposed"] for h in hist)
    print("\n=== SELFPLAY SUMMARY %s === delta(sp-static):" % args.model, dl, "| total model-proposed tasks:", mp)
    print("SELF-REFERENTIAL RSI?",
          "YES (model-driven curriculum compounds)" if len(dl) >= 4 and statistics.mean(dl[-2:]) > 0.05 and mp > 0 else "NO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--selfplay-gpu", default="0"); ap.add_argument("--static-gpu", default="1")
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed-tasks", type=int, default=16); ap.add_argument("--held", type=int, default=30)
    ap.add_argument("--propose", type=int, default=6); ap.add_argument("--sft-steps", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "selfplay_rsi"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
