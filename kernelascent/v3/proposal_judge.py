"""EXPERIMENT 2 (adopted plan step 2), verifier-substrate instance: decompose the research actor into a
PROPOSAL mechanism P and a JUDGE mechanism J, run the 2x2 factorial on a common target to localize why the
producer advantage is tiny (Experiment 1). Model-free (ladder banks); the model-backed GPU version follows.

  P0 = current proposer (rsi_true.mutate: small +-1/2/3 steps from the target config)
  Pg = richer proposer  (also injects configs spanning the full legal grid -> good configs ARE available)
  J0 = judge at the baseline actor coverage {n_inputs:2,n_edge:0}
  Jg = judge at strong actor coverage (best-legal config)
Cell value = mean true downstream Q (continuous_grade) of the child each (P,J) selects, on a common target.
  PgJ0 >> P0J0 -> PROPOSALS were the limiter. P0Jg >> P0J0 -> JUDGMENT was the limiter. Only PgJg helps ->
  proposal/judgment COADAPTATION. None help -> target/space/budget/metric need work.
"""
import os, sys, json, argparse, random, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3 import rsi_true as RT, rsi_verify as V, depth_probe as DP

NI = [1, 2, 3, 4, 6, 8, 12, 16]; NE = [0, 1, 2, 4, 8, 16, 24]


def propose(target, M, rng, rich, grid_frac=0.5):
    cands = [RT.mutate(dict(target), rng) for _ in range(M)]
    if rich:                                   # richer proposer: replace some slots with grid-spanning configs
        k = max(1, int(M * grid_frac))
        for _ in range(k):
            cands[rng.randrange(M)] = {"n_inputs": rng.choice(NI), "n_edge": rng.choice(NE)}
    return cands


def judge_pick(cands, judge_cfg, practice, rng):
    def score(cfg):
        return sum(RT._judge_correct(task, RT.cfg_select(task, bank, cfg, rng), judge_cfg, rng) for task, bank in practice)
    return max(((score(c), -i, c) for i, c in enumerate(cands)), key=lambda t: (t[0], t[1]))[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="/tmp/instance_storage/kernelascent/dataset/tasks/public")
    ap.add_argument("--tiers", default="easy,medium")
    ap.add_argument("--ladder", default="/tmp/instance_storage/ka_data/ladders/ladders.json")
    ap.add_argument("--M", type=int, default=6); ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/proposal_judge")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    report = {}
    for tier in args.tiers.split(","):
        banks = DP.load_banks(args.curated, tier, args.ladder)
        if not banks:
            print("tier %s: no banks" % tier); continue
        rng = random.Random(0)
        # strong judge coverage = best-legal config from Experiment 1's surface
        surface = {(ni, ne): DP.q_of_config(banks, {"n_inputs": ni, "n_edge": ne}, rng, reps=2) for ni in NI for ne in NE}
        bni, bne = max(surface, key=surface.get)
        J0 = {"n_inputs": 2, "n_edge": 0}; Jg = {"n_inputs": bni, "n_edge": bne}
        cells = {"P0J0": [], "PgJ0": [], "P0Jg": [], "PgJg": []}
        for d in range(args.draws):
            r = random.Random(500 + d)
            prac = [banks[i] for i in r.sample(range(len(banks)), min(3, len(banks)))]
            target = {"n_inputs": r.choice(NI), "n_edge": r.choice(NE)}
            slate0 = propose(target, args.M, random.Random(900 + d), rich=False)
            slateg = propose(target, args.M, random.Random(900 + d), rich=True)
            for name, slate, judge in (("P0J0", slate0, J0), ("PgJ0", slateg, J0),
                                       ("P0Jg", slate0, Jg), ("PgJg", slateg, Jg)):
                pick = judge_pick(slate, judge, prac, random.Random(30 + d))
                cells[name].append(DP.q_of_config(banks, pick, rng, reps=1))
        mean = {k: round(statistics.mean(v), 4) for k, v in cells.items()}
        d_prop = round(mean["PgJ0"] - mean["P0J0"], 4)      # proposal effect (fixed weak judge)
        d_judge = round(mean["P0Jg"] - mean["P0J0"], 4)     # judgment effect (fixed weak proposer)
        d_joint = round(mean["PgJg"] - mean["P0J0"], 4)
        interaction = round(d_joint - d_prop - d_judge, 4)
        if max(d_prop, d_judge, d_joint) < 0.01:
            verdict = "target/space/budget/metric limited (no cell beats P0J0)"
        elif d_judge >= d_prop and d_judge >= 0.01:
            verdict = "JUDGMENT-limited (better judge helps most)"
        elif d_prop >= 0.01:
            verdict = "PROPOSAL-limited (better proposer helps most)"
        else:
            verdict = "coadaptation (only joint helps)"
        report[tier] = {"cells": mean, "best_judge_cfg": [bni, bne], "d_proposal": d_prop,
                        "d_judgment": d_judge, "d_joint": d_joint, "interaction": interaction, "verdict": verdict}
        print("\n=== TIER %s ===" % tier)
        print("  P0J0=%.3f  PgJ0=%.3f  P0Jg=%.3f  PgJg=%.3f" % (mean["P0J0"], mean["PgJ0"], mean["P0Jg"], mean["PgJg"]))
        print("  d_proposal=%+.4f  d_judgment=%+.4f  d_joint=%+.4f  interaction=%+.4f" % (d_prop, d_judge, d_joint, interaction))
        print("  VERDICT: %s" % verdict)
    json.dump(report, open(os.path.join(args.outdir, "proposal_judge.json"), "w"), indent=2)
    print("\nwrote", os.path.join(args.outdir, "proposal_judge.json"))


if __name__ == "__main__":
    main()
