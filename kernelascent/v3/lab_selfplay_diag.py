"""Experiment #3 — T5 SELF-PLAY DIAGNOSIS + RESCUE.

Diagnoses WHY T5 self-play L-F is ~null: instruments the LIVE author to log, per round, the difficulty of the
tasks it authors and the solver's SOLVE-RATE on them, exposing CURRICULUM COLLAPSE (authors drift to trivial
solve-rate~1 or unsolvable solve-rate~0). Then a RESCUE arm constrains the author to a GOLDILOCKS band (keep only
authored tasks the CURRENT solver solves 30-70% of the time) and asks whether L-F turns positive.

Three solver arms (each own LoRA model, matched budget), reusing lab_selfplay_rsi's author/score/gate:
  F  FROZEN-AUTHOR      : frontier escalates via the frozen base (adaptive curriculum, no author update).
  L  LIVE-AUTHOR        : frontier escalates via the current evolving model (unconstrained self-play).
  Lg LIVE + GOLDILOCKS  : same as L but authored tasks filtered to solver-solve-rate in [lo,hi].
Report per round: authored_solve_rate, authored_headroom, frac_trivial, frac_unsolvable,
L_minus_F_unconstrained (=C_L-C_F), L_minus_F_goldilocks (=C_Lg-C_F).

Prefer a mid model (>=3B) so authoring is non-degenerate. Memory-safe; grades isolated.
CLI: python -m kernelascent.v3.lab_selfplay_diag --model <hf> --l-gpu 0 --f-gpu 1 --g-gpu 2 --rounds 8 --outdir <d>
"""
import os, sys, json, argparse, random, statistics, time
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_selfplay_rsi as SP
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch


def _solve_rates(tok, mdl, srcs, k, adapter):
    """per-task solve rate (fraction of k candidates that are correct) + mean best score, for the given srcs."""
    if not srcs:
        return [], []
    gl = W.generate_batch(tok, mdl, srcs, k, adapter=adapter)
    codes = [[c for c in (AB.extract_modelnew(t) for t in g) if c] for g in gl]
    grades = W._grade_isolated_batch(list(zip(srcs, codes)))
    rates, best = [], []
    for cs, res in zip(codes, grades):
        ok_n = 0; bscore = 0.0; tot = max(len(res), 1)
        for gg in res:
            ok, se, scp, ceil = (list(gg) + [0, 0, 0, 1.5])[:4]
            if ok: ok_n += 1
            bscore = max(bscore, LK._score(ok, se, ceiling=1.5))
        rates.append(ok_n / tot); best.append(bscore)
    return rates, best


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    g = lambda s: [int(x) for x in str(s).split(",") if x != ""]
    tokF, mF = W.build(args.model, tuple(g(args.f_gpu)))
    tokL, mL = W.build(args.model, tuple(g(args.l_gpu)))
    tokG, mG = W.build(args.model, tuple(g(args.g_gpu)))
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    seed_t = names[:args.seed_tasks]; held = names[args.seed_tasks:args.seed_tasks + args.held]
    fF, fL, fG = list(seed_t), list(seed_t), list(seed_t)
    seen = {__import__("hashlib").sha1(LK.TASKS[n].encode()).hexdigest() for n in names}
    lo, hi = args.band_lo, args.band_hi
    print("SELFPLAY-DIAG %s seed=%d seed_tasks=%d held=%d band=[%.2f,%.2f] rounds=%d" %
          (args.model, args.seed, len(seed_t), len(held), lo, hi, args.rounds), flush=True)
    os.makedirs(args.outdir, exist_ok=True); hist = []
    for r in range(args.rounds):
        t0 = time.time()
        # develop each arm on its frontier (adapter ON = solver improves), SFT on solved
        _, _, pF = SP._score(tokF, mF, fF, args.k, adapter=True); W.sft(tokF, mF, pF, args.sft_steps)
        _, _, pL = SP._score(tokL, mL, fL, args.k, adapter=True); W.sft(tokL, mL, pL, args.sft_steps)
        _, _, pG = SP._score(tokG, mG, fG, args.k, adapter=True); W.sft(tokG, mG, pG, args.sft_steps)
        solvedF = [s for (s, _) in pF] or fF
        solvedL = [s for (s, _) in pL] or fL
        solvedG = [s for (s, _) in pG] or fG
        # authors: F=frozen base, L=live, Lg=live (then goldilocks-filter)
        addF, _ = SP.author(tokF, mF, solvedF, args.propose, seen, adapter=False, base_hard_ok=True)
        addL, stL = SP.author(tokL, mL, solvedL, args.propose, seen, adapter=True, base_hard_ok=False)
        addG, _ = SP.author(tokG, mG, solvedG, args.propose, seen, adapter=True, base_hard_ok=False)
        # DIAGNOSIS: solver solve-rate on the L author's fresh tasks -> curriculum-collapse metrics
        rates, best = _solve_rates(tokL, mL, addL, args.k, adapter=True)
        asr = round(statistics.mean(rates), 3) if rates else None
        headroom = round(statistics.mean([1.0 - b for b in best]), 3) if best else None   # 1-score: high=unsolved room
        frac_triv = round(sum(1 for x in rates if x >= 0.9) / len(rates), 3) if rates else None
        frac_uns = round(sum(1 for x in rates if x <= 0.1) / len(rates), 3) if rates else None
        # GOLDILOCKS: keep only Lg-authored tasks the Lg solver solves in [lo,hi]
        gr, _ = _solve_rates(tokG, mG, addG, args.k, adapter=True)
        keepG = [addG[i] for i, x in enumerate(gr) if lo <= x <= hi]
        fF += addF; fL += addL; fG += keepG
        # held-out capability per arm
        CF = SP._score(tokF, mF, held, args.k, adapter=True)[0]
        CL = SP._score(tokL, mL, held, args.k, adapter=True)[0]
        CG = SP._score(tokG, mG, held, args.k, adapter=True)[0]
        row = {"round": r, "authored_solve_rate": asr, "authored_headroom": headroom,
               "frac_trivial": frac_triv, "frac_unsolvable": frac_uns,
               "n_authored_L": len(addL), "n_kept_goldilocks": len(keepG),
               "C_F": round(CF, 3), "C_L": round(CL, 3), "C_Lg": round(CG, 3),
               "L_minus_F_unconstrained": round(CL - CF, 3), "L_minus_F_goldilocks": round(CG - CF, 3),
               "live_model_proposed": stL["model_proposed"], "round_sec": round(time.time() - t0, 1)}
        hist.append(row)
        print("round %d solve_rate=%s triv=%s unsolv=%s | L-F=%+.3f Lg-F=%+.3f keptG=%d (%.0fs)" %
              (r, asr, frac_triv, frac_uns, CL - CF, CG - CF, len(keepG), time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "band": [lo, hi], "history": hist,
                   "note": "Curriculum-collapse diagnosis + Goldilocks rescue. L-F_unconstrained vs L-F_goldilocks."},
                  open(os.path.join(args.outdir, "selfplay_diag.json"), "w"), indent=2)
    lu = [h["L_minus_F_unconstrained"] for h in hist]; lg = [h["L_minus_F_goldilocks"] for h in hist]
    print("\n=== SELFPLAY-DIAG SUMMARY %s ===" % args.model)
    print("  L-F unconstrained:", lu, "\n  L-F goldilocks:", lg)
    print("  GOLDILOCKS RESCUES self-play (Lg-F>0 sustained while L-F~0)?",
          "YES" if len(lg) >= 3 and statistics.mean(lg[-2:]) > 0.05 and statistics.mean(lu[-2:]) <= 0.05 else "NO/insufficient")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--f-gpu", default="0"); ap.add_argument("--l-gpu", default="1"); ap.add_argument("--g-gpu", default="2")
    ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed-tasks", type=int, default=16); ap.add_argument("--held", type=int, default=24)
    ap.add_argument("--propose", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sft-steps", type=int, default=40)
    ap.add_argument("--band-lo", type=float, default=0.3); ap.add_argument("--band-hi", type=float, default=0.7)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "selfplay_diag"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
