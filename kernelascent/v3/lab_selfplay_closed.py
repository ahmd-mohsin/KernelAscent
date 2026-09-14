"""Task 5b — CLOSED / API-model self-play RSI, HARDENED per GPT-6 Astra review (3-arm, anti-hack).

A closed model can't train weights, so its co-evolution channel is its PROCEDURE (strategy library + verified
archive). Three arms at EQUAL budget, one FIXED held ladder — mirrors the open (5a) 3-arm decomposition:
  S  STATIC        : procedure self-modifies on a FIXED frontier.
  F  FROZEN-AUTHOR : frontier ESCALATES, but the AUTHOR proposes with the START (empty) procedure — a frozen
                     author — while the SOLVER procedure still evolves. (adaptive curriculum, non-evolving author)
  L  LIVE-AUTHOR   : frontier escalates and the AUTHOR proposes WITH the co-evolved procedure (current strategies
                     injected). Solver procedure evolves identically to F.
Decomposition on the held ladder:
  L-S = total curriculum benefit;  F-S = curriculum w/o author procedure update;  **L-F = author CO-EVOLUTION**
(the closed self-referential-RSI signal). Only the AUTHOR's procedure differs between F and L.

Anti-reward-hacking acceptance for every authored task: valid AND MEANINGFUL (non-constant, non-identity, non-
trivial runtime) AND novel (dedup). Attribution logged (model_proposed / rejected_degenerate / rejected_dup).
Output selfplay_closed.json: per round Q_held for S/F/L, deltas L-S, F-S, L-F, frontier, acceptance stats.
"""
import os, sys, json, argparse, random, statistics, time, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_track_c as TC
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch

PROPOSE = ("Write a NEW, STRICTLY HARDER PyTorch module optimization task in EXACTLY this format: define `DT`, "
           "`class Model(nn.Module)` (__init__, forward), and `def get_inputs()` returning [one DT tensor]. Make it "
           "harder than the example (bigger shapes and/or an extra fused stage). Deterministic, self-contained. "
           "Return ONLY one python code block.{ctx}\n\n```python\n{src}\n```")


def _extract(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t or "", re.S); c = (m.group(1) if m else (t or "")).strip()
    return c if ("class Model" in c and "get_inputs" in c and "DT" in c) else None


def _meaningful(src):
    """valid AND non-degenerate: builds, finite gold, output not constant, not identity, non-trivial runtime."""
    try:
        ref, x, g, e = AB.build_ref(src)
        t = AB.time_fn(lambda z: ref(z), (x,))
        if not (t > 5e-5 and torch.isfinite(g).all().item()): return False
        gf = g.float()
        if gf.std().item() < 1e-6: return False
        if torch.is_tensor(x) and x.shape == g.shape and torch.allclose(x.float(), gf, atol=1e-3): return False
        return True
    except Exception:
        return False


def _procctx(U):
    """Compact co-evolved-procedure context the LIVE author is allowed to condition on (FROZEN author gets '')."""
    ss = U.get("strategies", [])
    if not ss: return ""
    return "\n\nYou have learned these optimization strategies; propose a task that stresses them:\n- " + "\n- ".join(str(s)[:160] for s in ss[:6])


def _author(gen, U_author, solved, tasks, want, seen, prefix, live):
    """Propose harder tasks. live=True -> condition on co-evolved procedure; live=False -> frozen (empty) author."""
    added = []; st = {"model_proposed": 0, "rejected_degenerate": 0, "rejected_dup": 0}
    ctx = _procctx(U_author) if live else ""
    for sn in (solved or list(tasks))[:max(want * 2, 8)]:
        if len(added) >= want: break
        code = _extract(gen(PROPOSE.format(ctx=ctx, src=tasks[sn]), ""))
        if not code: continue
        key = hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()
        if key in seen: st["rejected_dup"] += 1; continue
        if not _meaningful(code): st["rejected_degenerate"] += 1; continue
        seen.add(key); nn = "%s_%d" % (prefix, len(added)); tasks[nn] = code; added.append(nn); st["model_proposed"] += 1
    return added, st


def _mkgen(args):
    import curate_bedrock as CB
    _c = {}
    def gen(user, system):
        cur = CB.Curator(args.model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
        if _c: cur.resolved = _c["r"]; cur.reasoning = _c["rc"]
        else: cur.resolve(); cur.resolve_reasoning(); _c["r"] = cur.resolved; _c["rc"] = cur.reasoning
        for _ in range(3):
            o = cur.generate(user) or ""
            if o.strip() and not o.startswith("BEDROCK_ERROR"): return o
        return o
    return gen


def run(args):
    random.seed(args.seed)
    gen = _mkgen(args)
    tasks = dict(LK.TASKS); names = list(tasks); random.Random(1).shuffle(names)
    seedn = names[:args.seed_tasks]; held = names[args.seed_tasks:args.seed_tasks + args.held]
    fS = list(seedn); fF = list(seedn); fL = list(seedn)
    seen = {hashlib.sha1(re.sub(r"\s+", "", tasks[n]).encode()).hexdigest() for n in names}
    US = {"strategies": [], "archive": {}}; UF = {"strategies": [], "archive": {}}; UL = {"strategies": [], "archive": {}}
    print("SELFPLAY-CLOSED-3ARM %s seed=%d seed_tasks=%d held=%d rounds=%d (S/F/L)" %
          (args.model, args.seed, len(seedn), len(held), args.rounds), flush=True)
    os.makedirs(args.outdir, exist_ok=True); hist = []
    statef = os.path.join(args.outdir, "resume_state.json"); start = 0
    if os.path.exists(statef):                                   # RESUME (state restored from S3 on a new node)
        try:
            st = json.load(open(statef))
            US, UF, UL = st["US"], st["UF"], st["UL"]; fS, fF, fL = st["fS"], st["fF"], st["fL"]
            seen = set(st["seen"]); hist = st["hist"]; start = st["round"] + 1
            tasks.update(st.get("authored", {}))    # restore model-authored task defs so develop() can find F*/L* frontier keys
            print("RESUMED closed %s from round %d (frontier_L=%d, done=%d, authored=%d)" % (args.model, start, len(fL), len(hist), len(st.get("authored", {}))), flush=True)
        except Exception as e:
            print("closed resume failed (%s); starting fresh" % e, flush=True)
    for r in range(start, args.rounds):
        t0 = time.time()
        # STATIC: procedure improves on fixed frontier
        _, _, evS, verS = TC.develop(US, fS, tasks, gen, args.k, args.grade_gpu); US["archive"].update(verS); US = TC.improve(US, evS, gen)
        # FROZEN-AUTHOR: solver procedure improves; author proposes with EMPTY procedure
        _, _, evF, verF = TC.develop(UF, fF, tasks, gen, args.k, args.grade_gpu); UF["archive"].update(verF); UF = TC.improve(UF, evF, gen)
        solvedF = [k[5:] for k in verF if k.startswith("fast_") and k[5:] in tasks] or fF   # verified keys are "fast_<taskname>"
        addF, stF = _author(gen, UF, solvedF, tasks, args.propose, seen, "F%d" % r, live=False); fF += addF
        # LIVE-AUTHOR: solver procedure improves; author proposes conditioned on co-evolved procedure
        _, _, evL, verL = TC.develop(UL, fL, tasks, gen, args.k, args.grade_gpu); UL["archive"].update(verL); UL = TC.improve(UL, evL, gen)
        solvedL = [k[5:] for k in verL if k.startswith("fast_") and k[5:] in tasks] or fL
        addL, stL = _author(gen, UL, solvedL, tasks, args.propose, seen, "L%d" % r, live=True); fL += addL
        # held ladder for all three
        QS, _, _, _ = TC.develop(US, held, tasks, gen, args.k, args.grade_gpu)
        QF, _, _, _ = TC.develop(UF, held, tasks, gen, args.k, args.grade_gpu)
        QL, _, _, _ = TC.develop(UL, held, tasks, gen, args.k, args.grade_gpu)
        row = {"round": r, "Q_held_static": round(QS, 3), "Q_held_frozen_author": round(QF, 3), "Q_held_live": round(QL, 3),
               "L_minus_S": round(QL - QS, 3), "F_minus_S": round(QF - QS, 3), "L_minus_F": round(QL - QF, 3),
               "frontier_L": len(fL), "frontier_F": len(fF), "live_model_proposed": stL["model_proposed"],
               "live_rejected_degenerate": stL["rejected_degenerate"], "n_strategies_L": len(UL["strategies"])}
        hist.append(row)
        print("round %d Q_held S=%.3f F=%.3f L=%.3f | L-S=%+.3f F-S=%+.3f L-F=%+.3f | live_prop=%d degen=%d (%.0fs)" %
              (r, QS, QF, QL, QL - QS, QF - QS, QL - QF, stL["model_proposed"], stL["rejected_degenerate"], time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "held": len(held), "history": hist,
                   "note": "CLOSED 3-arm: STATIC / FROZEN-AUTHOR / LIVE-AUTHOR. PRIMARY=L_minus_F (author procedure co-evolution)."},
                  open(os.path.join(args.outdir, "selfplay_closed.json"), "w"), indent=2)
        json.dump({"round": r, "US": US, "UF": UF, "UL": UL, "fS": fS, "fF": fF, "fL": fL,
                   "seen": list(seen), "hist": hist,
                   "authored": {k: tasks[k] for k in tasks if k not in LK.TASKS}},  # persist authored task defs (F*/L*)
                  open(statef, "w"))   # full-state resume ckpt (S3-synced by daemon)
    lf = [h["L_minus_F"] for h in hist]; mp = sum(h["live_model_proposed"] for h in hist)
    print("\n=== SELFPLAY-CLOSED-3ARM SUMMARY %s === L-F(co-evolution):" % args.model, lf, "| live model_proposed:", mp)
    print("CLOSED AUTHOR CO-EVOLUTION COMPOUNDS?", "YES" if len(lf) >= 3 and statistics.mean(lf[-2:]) > 0.05 and mp > 0 else "NO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed-tasks", type=int, default=16); ap.add_argument("--held", type=int, default=30)
    ap.add_argument("--propose", type=int, default=6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--grade-gpu", default="0")
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "selfplay_closed"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
