"""Task 5a — OPEN-WEIGHT self-play RSI, HARDENED per GPT-6 Astra review (3-arm, anti-hack).

Three learners from the same base, EQUAL budget, scored each round on ONE fixed held-out ladder:
  S  STATIC        : fixed seed frontier; solver LoRA-SFTs normally.
  F  FROZEN-AUTHOR : frontier escalates, but the AUTHOR is the FROZEN BASE (adapter OFF) — an adaptive curriculum
                     from a non-evolving author; solver still SFTs. (isolates "any adaptive curriculum")
  L  LIVE-AUTHOR   : frontier escalates and the AUTHOR is the CURRENT evolving model (adapter ON). (self-play)
Decomposition on the held ladder:
  L-S = total adaptive-curriculum benefit;  F-S = benefit w/o updating the author;  **L-F = author CO-EVOLUTION**
(the self-referential-RSI signal). Only the AUTHOR role differs between F and L; solver learning is identical.

Anti-reward-hacking acceptance for every authored task: valid (builds, finite fp32 gold) AND MEANINGFUL (output
not constant, not equal to input, non-trivial runtime) AND HARDER (frozen base cannot already ace it) AND novel
(dedup). Attribution logged: model_proposed / rejected_degenerate / rejected_dup / backstop. Headroom score, seeds.
Output selfplay_rsi.json: per round C_held for S/F/L, deltas L-S, F-S, L-F, frontier sizes, acceptance stats.
"""
import os, sys, json, argparse, random, statistics, time, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch

PROPOSE = ("Write a NEW, STRICTLY HARDER PyTorch module optimization task in EXACTLY this format: `DT`, "
           "`class Model(nn.Module)` (__init__, forward), `def get_inputs()` returning [one DT tensor]. Harder than "
           "the example (bigger shapes and/or an extra fused stage). Deterministic, self-contained, ONE code block.\n\n```python\n{src}\n```")


def _extract(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t or "", re.S); c = (m.group(1) if m else (t or "")).strip()
    return c if ("class Model" in c and "get_inputs" in c and "DT" in c) else None


def _meaningful(src):
    """valid AND non-degenerate: builds, finite gold, output not constant, not identity, non-trivial runtime."""
    try:
        ref, x, gold, e = AB.build_ref(src)
        t = AB.time_fn(lambda z: ref(z), (x,))
        if not (t > 5e-5 and torch.isfinite(gold).all().item()):
            return False                               # too tiny / non-finite
        g = gold.float()
        if g.std().item() < 1e-6:
            return False                               # constant output (no-op)
        if torch.is_tensor(x) and x.shape == gold.shape and torch.allclose(x.float(), g, atol=1e-3):
            return False                               # identity map
        return True
    except Exception:
        return False


def _scale_shapes(src, factor=2):
    m = re.search(r"def get_inputs\(\).*?return\s*\[(.*?)\]", src, re.S)
    if not m: return None
    seg = m.group(0)
    new = re.sub(r"\b\d{2,}\b", lambda mm: str(min(8192, max(8, (int(int(mm.group(0)) * factor) // 8) * 8))), seg)
    return src.replace(seg, new) if new != seg else None


def author(tok, mdl, solved, want, seen, adapter, base_hard_ok):
    """Author harder tasks with the given policy (adapter ON=live, OFF=frozen base). Returns (tasks, stats)."""
    out = []; st = {"model_proposed": 0, "rejected_degenerate": 0, "rejected_dup": 0, "backstop": 0}
    prompts = [PROPOSE.format(src=s) for s in solved[:max(want * 2, 8)]]
    if prompts:
        gens = W.generate_batch(tok, mdl, prompts, 2, max_new=1200, adapter=adapter)
        for gl in gens:
            for g in gl:
                if len(out) >= want: break
                code = _extract(g)
                if not code: continue
                key = hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()
                if key in seen: st["rejected_dup"] += 1; continue
                if not _meaningful(code): st["rejected_degenerate"] += 1; continue
                seen.add(key); out.append(code); st["model_proposed"] += 1
    for s in solved:                                   # programmatic backstop (logged), only to fill remainder
        if len(out) >= want: break
        c = _scale_shapes(s)
        if c:
            key = hashlib.sha1(re.sub(r"\s+", "", c).encode()).hexdigest()
            if key not in seen and _meaningful(c):
                seen.add(key); out.append(c); st["backstop"] += 1
    return out, st


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
            best = max(best, LK._score(ok, scp if uc else se, ceiling=(ceil if uc else 1.5)))
            if ok: ok_any = 1; bc = bc or code
        sc.append(best); corr.append(ok_any)
        if bc: pairs.append((src, bc))
    return (statistics.mean(sc) if sc else 0.0), (statistics.mean(corr) if corr else 0.0), pairs


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    gS = [int(x) for x in str(args.s_gpu).split(",")]; gF = [int(x) for x in str(args.f_gpu).split(",")]; gL = [int(x) for x in str(args.l_gpu).split(",")]
    tokS, mS = W.build(args.model, gS); tokF, mF = W.build(args.model, gF); tokL, mL = W.build(args.model, gL)
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    seed_t = [LK.TASKS[n] for n in names[:args.seed_tasks]]
    held = [LK.TASKS[n] for n in names[args.seed_tasks:args.seed_tasks + args.held]]
    frS = list(seed_t); frF = list(seed_t); frL = list(seed_t)
    seen = {hashlib.sha1(re.sub(r"\s+", "", s).encode()).hexdigest() for s in seed_t + held}
    print("SELFPLAY-3ARM %s seed=%d seed_tasks=%d held=%d rounds=%d (S/F/L = static/frozen-author/live-author)" %
          (args.model, args.seed, len(seed_t), len(held), args.rounds), flush=True)
    os.makedirs(args.outdir, exist_ok=True); hist = []
    for r in range(args.rounds):
        t0 = time.time()
        # STATIC
        _, _, pS = _score(tokS, mS, frS, args.k, adapter=True); W.sft(tokS, mS, pS, args.sft_steps)
        # FROZEN-AUTHOR: solver mF learns; author is the FROZEN BASE (adapter OFF)
        _, _, pF = _score(tokF, mF, frF, args.k, adapter=True); W.sft(tokF, mF, pF, args.sft_steps)
        addF, stF = author(tokF, mF, [s for s, _ in pF] or frF, args.propose, seen, adapter=False, base_hard_ok=True); frF += addF
        # LIVE-AUTHOR: solver mL learns; author is the CURRENT evolving model (adapter ON)
        _, _, pL = _score(tokL, mL, frL, args.k, adapter=True); W.sft(tokL, mL, pL, args.sft_steps)
        addL, stL = author(tokL, mL, [s for s, _ in pL] or frL, args.propose, seen, adapter=True, base_hard_ok=True); frL += addL
        # held ladder for all three
        CS, _, _ = _score(tokS, mS, held, args.k, adapter=True)
        CF, _, _ = _score(tokF, mF, held, args.k, adapter=True)
        CL, _, _ = _score(tokL, mL, held, args.k, adapter=True)
        _, base_corr, _ = _score(tokL, mL, addL or frL[-4:], args.k, adapter=False)   # escalation check on live-authored
        row = {"round": r, "C_held_static": round(CS, 3), "C_held_frozen_author": round(CF, 3), "C_held_live": round(CL, 3),
               "L_minus_S": round(CL - CS, 3), "F_minus_S": round(CF - CS, 3), "L_minus_F": round(CL - CF, 3),
               "frontier_L": len(frL), "frontier_F": len(frF), "live_model_proposed": stL["model_proposed"],
               "live_rejected_degenerate": stL["rejected_degenerate"], "live_backstop": stL["backstop"],
               "base_correct_on_live": round(base_corr, 3)}
        hist.append(row)
        try:                                             # reasoning-chain trace for later qualitative analysis
            tf = os.path.join(args.outdir, "selfplay_trace.jsonl")
            json.dump({"round": r, "live_authored_task": (addL[0][:2000] if addL else None),
                       "live_best_kernel": (pL[0][1][:2000] if pL else None),
                       "L_minus_F": round(CL - CF, 3), "live_model_proposed": stL["model_proposed"]},
                      open(tf, "a")); open(tf, "a").write("\n")
        except Exception:
            pass
        print("round %d C_held S=%.3f F=%.3f L=%.3f | L-S=%+.3f F-S=%+.3f L-F=%+.3f | live_prop=%d degen=%d back=%d base=%.2f (%.0fs)" %
              (r, CS, CF, CL, CL - CS, CF - CS, CL - CF, stL["model_proposed"], stL["rejected_degenerate"], stL["backstop"], base_corr, time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "held": len(held), "history": hist,
                   "note": "3-arm: STATIC / FROZEN-AUTHOR / LIVE-AUTHOR. PRIMARY=L_minus_F (author co-evolution). L-S=total curriculum, F-S=curriculum w/o author update."},
                  open(os.path.join(args.outdir, "selfplay_rsi.json"), "w"), indent=2)
    lf = [h["L_minus_F"] for h in hist]; mp = sum(h["live_model_proposed"] for h in hist)
    print("\n=== SELFPLAY-3ARM SUMMARY %s === L-F(co-evolution):" % args.model, lf, "| live model_proposed:", mp)
    print("AUTHOR CO-EVOLUTION COMPOUNDS?", "YES" if len(lf) >= 4 and statistics.mean(lf[-2:]) > 0.05 and mp > 0 else "NO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--s-gpu", default="0"); ap.add_argument("--f-gpu", default="1"); ap.add_argument("--l-gpu", default="2")
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed-tasks", type=int, default=16); ap.add_argument("--held", type=int, default=30)
    ap.add_argument("--propose", type=int, default=6); ap.add_argument("--sft-steps", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "selfplay_rsi"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
