"""EXPERIMENT 1 (adopted plan step 1): RESOLVE THE STRUCTURAL CLAIM for the verifier-config loop.

Model-free. Uses the curated reference+ladder banks (rsi_true --headroom substrate). Produces a
HEADROOM / SELECTION MAP, not a model ranking, to distinguish WHY causal producer-advantage (F) is ~0:
  (a) EXHAUSTION      : Q(config) is flat -> best-legal Q ~ current Q -> no procedure headroom.
  (b) POOR PROPOSALS  : best-in-proposed-slate << best-legal -> the mutation proposer misses good configs.
  (c) POOR SELECTION  : selected Q << best-in-slate -> the actor's judge can't pick the good config.
  (d) INEFFECTIVE METRIC / STRUCTURAL NULL : old and new actors select the SAME child for every slate
      (incl ties) -> no producer difference for F to measure, regardless of headroom.

Outputs, per tier:
  1. Q(config) surface over the legal (n_inputs, n_edge) grid + procedure headroom (best_legal - current)
     + whether "more edge is always better and ~free" (monotone in n_edge at flat cost -> terminal policy).
  2. Old-vs-new actor judge replay on IDENTICAL proposal slates from the common target: agreement rate +
     benefit (downstream Q of the new actor's pick minus the old actor's pick).
  3. A verdict tag {exhaustion | poor_proposals | poor_selection | structural_null | headroom_present}.
Official hidden grading (continuous_grade) is used only for THIS offline analysis, never inside the
live actor's evidence.
"""
import os, sys, json, argparse, random, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3 import rsi_true as RT, rsi_verify as V, curated_loader as CL


def load_banks(curated, tier, ladder_path):
    ladders = json.load(open(ladder_path)).get(tier, {})
    projs = CL.load_projects(curated, tier)
    banks = []
    for task in projs:
        lad = ladders.get(task["name"])
        if not lad:
            continue
        bank = [("ref", task["ref"])] + [("l%d" % i, CL._compile(r["code"], task["fn"])) for i, r in enumerate(lad)]
        bank = [(n, f) for n, f in bank if callable(f)]
        random.Random(hash(task["name"]) & 0xffffffff).shuffle(bank)
        if len(bank) >= 3:
            banks.append((task, bank))
    return banks


def q_of_config(banks, cfg, rng, reps=3):
    """True downstream Q = continuous hidden pass-rate of the candidate this config's verifier selects."""
    vals = []
    for _ in range(reps):
        for task, bank in banks:
            sel = RT.cfg_select(task, bank, cfg, rng)
            vals.append(V.continuous_grade(task, sel, rng, n_hidden=120, edge_frac=0.75))
    return sum(vals) / len(vals) if vals else 0.0


def actor_pick(actor_cfg, target_cfg, practice, M, rng):
    """Replicate rsi_true.revise: propose M mutated child-configs from target; the ACTOR judges each by
    _judge_correct on practice (using the actor's OWN coverage) and keeps the best. Return (chosen, slate)."""
    cands = [RT.mutate(dict(target_cfg), rng) for _ in range(M)]
    def score(cfg):
        s = 0
        for task, bank in practice:
            sel = RT.cfg_select(task, bank, cfg, rng)
            s += RT._judge_correct(task, sel, actor_cfg, rng)
        return s
    scored = [(score(c), i, c) for i, c in enumerate(cands)]
    best = max(scored, key=lambda t: (t[0], -t[1]))    # tie -> lowest index (matches max() first-wins)
    return best[2], cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="/tmp/instance_storage/kernelascent/dataset/tasks/public")
    ap.add_argument("--tiers", default="easy,medium")
    ap.add_argument("--ladder", default="/tmp/instance_storage/ka_data/ladders/ladders.json")
    ap.add_argument("--M", type=int, default=6); ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/depth_probe")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    NI = [1, 2, 3, 4, 6, 8, 12, 16]; NE = [0, 1, 2, 4, 8, 16, 24]
    report = {}
    for tier in args.tiers.split(","):
        banks = load_banks(args.curated, tier, args.ladder)
        if not banks:
            print("tier %s: no laddered banks" % tier); continue
        rng = random.Random(0)
        # 1. Q(config) surface
        surface = {}
        for ni in NI:
            for ne in NE:
                surface["%d,%d" % (ni, ne)] = round(q_of_config(banks, {"n_inputs": ni, "n_edge": ne}, rng), 4)
        current = surface["2,0"]                          # U0 default
        best_legal = max(surface.values())
        best_cfg = max(surface, key=surface.get)
        proc_headroom = round(best_legal - current, 4)
        # monotone-in-edge-at-fixed-inputs check (is "more edge always better"?)
        mono = all(surface["2,%d" % NE[k]] <= surface["2,%d" % NE[k + 1]] + 1e-9 for k in range(len(NE) - 1))
        # 2. old vs new judge on IDENTICAL slates from a common target
        #    U0 = {2,0}; Ug = the best-legal config (a strong "later" actor) -> do they judge slates differently?
        U0 = {"n_inputs": 2, "n_edge": 0}
        Ug = {"n_inputs": int(best_cfg.split(",")[0]), "n_edge": int(best_cfg.split(",")[1])}
        agree = 0; benefit = []
        for d in range(args.draws):
            r2 = random.Random(1000 + d)
            prac = [banks[i] for i in r2.sample(range(len(banks)), min(3, len(banks)))]
            target = {"n_inputs": r2.choice(NI), "n_edge": r2.choice(NE)}
            # SAME slate for both actors: seed the mutate stream identically per actor pick
            pick0, slate0 = actor_pick(U0, target, prac, args.M, random.Random(7000 + d))
            pickg, slateg = actor_pick(Ug, target, prac, args.M, random.Random(7000 + d))   # identical slate
            same = (pick0 == pickg)
            agree += 1 if same else 0
            if not same:
                q0 = q_of_config(banks, pick0, rng, reps=1); qg = q_of_config(banks, pickg, rng, reps=1)
                benefit.append(qg - q0)
        agree_rate = agree / args.draws
        mean_benefit = round(statistics.mean(benefit), 4) if benefit else 0.0
        # verdict
        if proc_headroom < 0.02:
            verdict = "exhaustion (no procedure headroom: best-legal ~ current)"
        elif agree_rate >= 0.98:
            verdict = "structural_null (old & new actor pick the SAME child on ~every slate -> no producer diff for F)"
        elif mean_benefit <= 0.005:
            verdict = "poor_selection_or_metric (actors disagree but the newer pick is not better)"
        else:
            verdict = "headroom_present (newer actor selects a better child -> F SHOULD be >0; re-check estimator/budget)"
        report[tier] = {"n_banks": len(banks), "current_Q(2,0)": current, "best_legal_Q": best_legal,
                        "best_cfg": best_cfg, "procedure_headroom": proc_headroom,
                        "more_edge_monotone_at_ni2": mono,
                        "old_new_judge_agreement": round(agree_rate, 3), "mean_benefit_new_minus_old": mean_benefit,
                        "verdict": verdict, "surface": surface}
        print("\n=== TIER %s (banks=%d) ===" % (tier, len(banks)))
        print("  current Q(2,0)=%.3f  best-legal Q=%.3f @ (%s)  PROCEDURE HEADROOM=%.3f" % (current, best_legal, best_cfg, proc_headroom))
        print("  more-edge monotone (ni=2): %s   [Q(2,0)=%.3f Q(2,8)=%.3f Q(2,24)=%.3f]" % (mono, surface["2,0"], surface["2,8"], surface["2,24"]))
        print("  old-vs-new judge agreement=%.2f  mean benefit(new-old on disagreements)=%+.4f" % (agree_rate, mean_benefit))
        print("  VERDICT: %s" % verdict)
    json.dump(report, open(os.path.join(args.outdir, "depth_probe.json"), "w"), indent=2)
    print("\nwrote", os.path.join(args.outdir, "depth_probe.json"))


if __name__ == "__main__":
    main()
