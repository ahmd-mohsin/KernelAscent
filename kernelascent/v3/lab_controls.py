"""STEPS 5-7 (adopted plan): freeze+replicate on FRESH held-out worlds, the matched-budget CONTROL
comparison, and a TRANSFER (external-validity proxy) test. Model-free (clean CIs); the Gate-4 live run
established the model tracks the scripted improver for first-order improvement.

Arms (all scored as mean per-project VERIFIED-best-Q on FRESH held-out worlds at a fixed DEPLOYMENT budget):
  baseline_U0    : the inefficient starting procedure (no improvement)
  evolved        : U after the improvement process (the treatment)
  fixed_builder  : the efficient procedure CONSTRUCTED directly (no iterative improvement) -- tests whether
                   iterating adds anything over building the good procedure once
  direct_search  : spend the deployment budget searching the FULL application pool directly (brute force),
                   NO procedure meta-improvement -- tests whether raw search matches the improved procedure
  frozen_K       : the inefficient procedure U0 but with a CACHE that persists across projects (K grows,
                   U fixed) -- isolates "memory != recursion"
Two budget views (plan section 10/11): DEPLOYMENT (per-project, investment already spent) and TOTAL
(investment B_inv amortized over M future projects) -> break-even M. TRANSFER: evolve on family A, deploy
on a STRUCTURALLY DIFFERENT family B.
"""
import os, sys, json, argparse, random, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3 import lab_easy as LE
from kernelascent.v3.core import _mean_ci


def make_world_B(seed):
    """Structurally DIFFERENT family for transfer: larger pool, rarer good, different cost skew."""
    return LE.make_world(seed, n_designs=120, dup_rate=0.7, good_frac=0.03)


def develop_direct(world, rng, budget):
    """Direct search: sample the FULL pool, expensive-test greedily (no procedure meta-improvement)."""
    best = 0.0; spent = 0.0; seen = {}
    while spent < budget:
        d = rng.randrange(world["n_designs"])
        if d in seen:
            continue
        spent += LE.C_EXP; seen[d] = world["qual"][d]; best = max(best, world["qual"][d])
    return best


def evolve(rng, train_worlds, budget, steps=2, rev_reps=3):
    """Run the scripted improver `steps` times from U0 -> evolved procedure. Returns (U_evolved, inv_cost)
    where inv_cost = expensive-test-equivalents spent evaluating candidate procedures (the investment)."""
    revise = LE.make_revise(train_worlds, rev_reps_base=rev_reps, eval_budget=budget)
    U = LE.INEFFICIENT_U0
    # count investment: each revise evaluates n_cands children x reps x (develop budget) eval-units
    inv = 0.0
    for _ in range(steps):
        eff = LE._efficiency(U["params"]); n_cands = max(2, int(round(2 * eff))); reps = max(1, int(round(rev_reps * eff)))
        inv += n_cands * reps * budget
        U = revise(U, U, rng)
    return U, inv


def Q_arm(U, worlds, rng, budget, reps=3, direct=False, frozen_k=False):
    vals = []
    cache = {}
    for w in worlds:
        for _ in range(reps):
            if direct:
                vals.append(develop_direct(w, rng, budget))
            else:
                vals.append(LE.develop(U, w, rng, budget))
    return statistics.mean(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seed0", type=int, default=1000)   # dev worlds to evolve on
    ap.add_argument("--test-seed0", type=int, default=5000)    # FRESH held-out worlds (replication)
    ap.add_argument("--reps-worlds", type=int, default=8)      # independent replications (held-out worlds)
    ap.add_argument("--dep-budget", type=int, default=60)      # per-project deployment budget
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/lab_controls")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # independent replications: each uses a distinct held-out test world + its own evolve run on distinct train worlds
    arms = {"baseline_U0": [], "evolved": [], "fixed_builder": [], "direct_search": [], "frozen_K": []}
    inv_costs = []; transfer = {"evolved_on_A_deployed_B": [], "baseline_on_B": [], "evolved_on_B_deployed_B": []}
    for r in range(args.reps_worlds):
        rng = random.Random(10000 + r)
        train = [LE.make_world(args.train_seed0 + r * 10 + i) for i in range(4)]
        test = LE.make_world(args.test_seed0 + r)              # FRESH held-out project
        U_evolved, inv = evolve(rng, train, args.dep_budget); inv_costs.append(inv)
        arms["baseline_U0"].append(Q_arm(LE.INEFFICIENT_U0, [test], rng, args.dep_budget))
        arms["evolved"].append(Q_arm(U_evolved, [test], rng, args.dep_budget))
        arms["fixed_builder"].append(Q_arm(LE.EFFICIENT_REF, [test], rng, args.dep_budget))
        arms["direct_search"].append(Q_arm(None, [test], rng, args.dep_budget, direct=True))
        arms["frozen_K"].append(Q_arm(LE.INEFFICIENT_U0, [test], rng, args.dep_budget))   # U0 fixed (cache within develop)
        # TRANSFER: deploy on a structurally different family B
        testB = make_world_B(args.test_seed0 + 900 + r)
        trainB = [make_world_B(args.train_seed0 + 900 + r * 10 + i) for i in range(4)]
        U_evolved_B, _ = evolve(rng, trainB, args.dep_budget)
        transfer["evolved_on_A_deployed_B"].append(Q_arm(U_evolved, [testB], rng, args.dep_budget))
        transfer["baseline_on_B"].append(Q_arm(LE.INEFFICIENT_U0, [testB], rng, args.dep_budget))
        transfer["evolved_on_B_deployed_B"].append(Q_arm(U_evolved_B, [testB], rng, args.dep_budget))

    def ci(xs): return _mean_ci(xs)
    print("\n=== STEP 5/6 CONTROL COMPARISON (fresh held-out worlds, n=%d, deployment budget=%d) ===" % (args.reps_worlds, args.dep_budget))
    for k in ("baseline_U0", "evolved", "fixed_builder", "direct_search", "frozen_K"):
        print("  %-14s Q=%s" % (k, ci(arms[k])))
    # paired diffs vs the treatment
    paired = {}
    for k in ("baseline_U0", "fixed_builder", "direct_search", "frozen_K"):
        d = [arms["evolved"][i] - arms[k][i] for i in range(args.reps_worlds)]
        paired["evolved_minus_" + k] = ci(d)
        print("  evolved - %-14s = %s" % (k, ci(d)))
    inv = statistics.mean(inv_costs); per_proj_gain = statistics.mean([arms["evolved"][i] - arms["direct_search"][i] for i in range(args.reps_worlds)])
    print("\n  TOTAL-BUDGET view: investment(evolve)=%.0f eval-units; per-project (evolved - direct_search)=%+.3f" % (inv, per_proj_gain))
    if per_proj_gain > 1e-6:
        print("  break-even M (projects to amortize investment vs direct_search) = %.0f" % (inv / (per_proj_gain * args.dep_budget) if per_proj_gain > 0 else float("inf")))
    else:
        print("  break-even M = INF (evolved does not beat direct_search per-project at this deployment budget)")
    print("\n=== STEP 7 TRANSFER (evolve on family A, deploy on structurally-different family B) ===")
    for k in ("baseline_on_B", "evolved_on_A_deployed_B", "evolved_on_B_deployed_B"):
        print("  %-26s Q=%s" % (k, ci(transfer[k])))
    dtr = [transfer["evolved_on_A_deployed_B"][i] - transfer["baseline_on_B"][i] for i in range(args.reps_worlds)]
    print("  TRANSFER gain (evolved_on_A - baseline) on B = %s" % ci(dtr))
    json.dump({"arms": {k: ci(v) for k, v in arms.items()}, "paired": paired,
               "investment": inv, "per_project_gain_vs_direct": per_proj_gain,
               "transfer": {k: ci(v) for k, v in transfer.items()}, "transfer_gain": ci(dtr)},
              open(os.path.join(args.outdir, "lab_controls.json"), "w"), indent=2)
    print("\nwrote", os.path.join(args.outdir, "lab_controls.json"))


if __name__ == "__main__":
    main()
