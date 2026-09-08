"""RECURRING-DECISION research engine (adopted-review §5) + Experiment E1 (opportunity beyond the first
revision) and E3 (actor x target matrix). Model-free calibration.

The prior lab used ONE-SHOT flags (cache/dedup/staged) -> gains front-loaded -> F2~0 by construction. The
review's fix: the improvable procedure must be RECURRING decisions that recur for every project/evidence-
state and INTERACT, so multiple attainable improvements remain after the first revision. Policy U:
  propose_struct in [0,1] : structured/workload-aware proposal (reaches rare good designs; costs cheap tests)
  fidelity_thr   in [0,1] : cheap-signal cutoff before spending an expensive verified test
  alloc_temp     in [0,1] : explore(new) vs exploit(refine best region) each step
  select_k       in {1..4}: evidence gathered before committing (reduces mis-selection)
  transfer_w     in [0,1] : reuse of accumulated cross-project findings
These interact: better fidelity FREES budget -> makes higher propose_struct affordable -> gives alloc/
select something worth tuning. So the improvement LADDER can be deep (non-front-loaded) rather than one bump.

E1 asks: is the per-step gain spread across several revisions (multiple attainable improvements remain), or
front-loaded into the first? E3 builds the actor x target matrix M(i,j) to separate target saturation from
producer weakness. A positive here validates the DESIGN (substrate supports a deep ladder); the live-model
+ causal-F test is the scientific claim (next).
"""
import os, sys, json, argparse, random, statistics, copy
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3 import lab_easy as LE

C_CHEAP, C_EXP = 1, 4
U0 = {"params": {"propose_struct": 0.0, "fidelity_thr": 0.0, "alloc_temp": 1.0, "select_k": 1, "transfer_w": 0.0}}
# knob grids for the "best legal policy at budget" search + greedy revisions
GRID = {"propose_struct": [0.0, 0.25, 0.5, 0.75, 1.0], "fidelity_thr": [0.0, 0.3, 0.5, 0.7],
        "alloc_temp": [0.0, 0.25, 0.5, 1.0], "select_k": [1, 2, 3, 4], "transfer_w": [0.0, 0.5, 1.0]}


def develop(U, world, rng, budget=60, memory=None):
    """Recurring-decision research loop; returns best VERIFIED quality. memory (dict) persists good regions
    across projects when transfer_w>0."""
    p = U["params"]
    ps = p["propose_struct"]; fthr = p["fidelity_thr"]; atemp = p["alloc_temp"]
    sk = int(p["select_k"]); tw = p["transfer_w"]
    seen = {}; spent = 0.0; best = 0.0; good_hits = []
    memory = memory if memory is not None else {}
    draw = world["draw"]; N = world["n_designs"]
    while spent < budget:
        # PROPOSE (recurring): structured proposal ranks gg options from the FULL pool (reaches good
        # designs) with breadth rising in propose_struct; unstructured falls back to the junk-biased draw.
        gg = 1 + int(round(ps * 5))
        if rng.random() < ps:
            pool = list(memory.keys()) if (tw > 0 and memory and rng.random() < tw) else None
            opts = [(pool[rng.randrange(len(pool))] if pool else rng.randrange(N)) for _ in range(gg)]
            spent += gg * C_CHEAP * (0.5 + ps)             # structured proposal is budget-hungry
            ce = {d: LE._cheap(world, d, rng) for d in opts}
            d = max(opts, key=lambda x: ce[x]); cheap_d = ce[d]
        else:
            d = draw[rng.randrange(len(draw))]; cheap_d = LE._cheap(world, d, rng); spent += C_CHEAP
        # ALLOC (recurring): with prob (1-atemp) exploit -> re-pick from remembered good region instead
        if rng.random() > atemp and good_hits:
            d = max(good_hits, key=lambda x: seen.get(x, 0)); cheap_d = LE._cheap(world, d, rng)
        # FIDELITY (recurring): skip the expensive verified test on unpromising candidates
        if cheap_d < fthr:
            continue
        if d in seen:
            q = seen[d]                                    # cache (competent baseline: always on)
        else:
            # SELECT (recurring): gather sk cheap confirmations before spending the expensive test
            confirms = [LE._cheap(world, d, rng) for _ in range(sk - 1)]; spent += (sk - 1) * C_CHEAP
            if confirms and statistics.mean(confirms + [cheap_d]) < fthr:
                continue
            spent += C_EXP; q = world["qual"][d]; seen[d] = q
            if q >= 0.8:
                good_hits.append(d); memory[d] = q
        best = max(best, q)
    return best


def Q(U, worlds, rng, budget, reps=3):
    return statistics.mean(develop(U, w, rng, budget) for w in worlds for _ in range(reps))


def best_legal(worlds, rng, budget, base):
    """Coordinate-ascent to the best legal policy from `base` -> attainable ceiling at this budget."""
    cur = copy.deepcopy(base); curq = Q(cur, worlds, rng, budget)
    improved = True
    while improved:
        improved = False
        for k, vals in GRID.items():
            for v in vals:
                if cur["params"][k] == v:
                    continue
                cand = copy.deepcopy(cur); cand["params"][k] = v
                q = Q(cand, worlds, rng, budget)
                if q > curq + 1e-4:
                    cur, curq, improved = cand, q, True
    return cur, curq


def greedy_revision(U, worlds, rng, budget):
    """One revision = the single best legal single-coordinate change (a competent improver's step)."""
    curq = Q(U, worlds, rng, budget); best_c, best_q = None, curq
    for k, vals in GRID.items():
        for v in vals:
            if U["params"][k] == v:
                continue
            cand = copy.deepcopy(U); cand["params"][k] = v
            q = Q(cand, worlds, rng, budget)
            if q > best_q + 1e-4:
                best_c, best_q = cand, q
    return (best_c or copy.deepcopy(U)), best_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=6); ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/lab_engine")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    worlds = [LE.make_world(s) for s in range(args.worlds)]
    rng = random.Random(0)

    # E1: greedy improvement TRAJECTORY -> is per-step gain spread (non-front-loaded) or front-loaded?
    traj = []; U = copy.deepcopy(U0); q_prev = Q(U, worlds, rng, args.budget)
    _, ceiling = best_legal(worlds, rng, args.budget, U0)
    traj.append({"step": 0, "Q": round(q_prev, 4), "gain": 0.0, "remaining_headroom": round(ceiling - q_prev, 4),
                 "params": dict(U["params"])})
    checkpoints = [copy.deepcopy(U)]
    for s in range(1, args.steps + 1):
        U, q = greedy_revision(U, worlds, rng, args.budget)
        traj.append({"step": s, "Q": round(q, 4), "gain": round(q - q_prev, 4),
                     "remaining_headroom": round(ceiling - q, 4), "params": dict(U["params"])})
        checkpoints.append(copy.deepcopy(U)); q_prev = q
    print("=== E1: improvement trajectory (recurring-decision engine) ===")
    print("  ceiling (best legal policy) Q = %.3f" % ceiling)
    for t in traj:
        print("  step %d  Q=%.3f  gain=%+.3f  remaining=%.3f" % (t["step"], t["Q"], t["gain"], t["remaining_headroom"]))
    gains = [t["gain"] for t in traj if t["step"] > 0]
    n_meaningful = sum(1 for g in gains if g >= 0.02)
    front_load = (gains[0] / sum(gains)) if sum(gains) > 1e-6 else 1.0
    print("  meaningful revisions (gain>=0.02): %d/%d   first-step share of total gain: %.0f%%" %
          (n_meaningful, len(gains), 100 * front_load))
    verdict = ("NON-FRONT-LOADED: multiple attainable improvements remain after the first revision"
               if n_meaningful >= 3 and front_load < 0.7 else
               "STILL FRONT-LOADED: gain concentrated in the first revision")
    print("  VERDICT:", verdict)

    # E3: actor x target matrix M(i,j) = usefulness of actor_i's child modifying common target_j vs keeping T_j
    cps = [checkpoints[0], checkpoints[min(2, len(checkpoints) - 1)], checkpoints[-1]]  # U0, mid, late
    tgt = cps
    def child_of(actor, target):
        # actor's policy governs a one-coordinate improving search on the common target (its "revise")
        c, _ = greedy_revision(target, worlds, rng, args.budget)
        # actor quality modulates how reliably the improving change is found: better actor (higher struct+
        # fidelity+select) = less noise -> keep; worse actor may miss it (return target).
        eff = actor["params"]["propose_struct"] + (1 - actor["params"]["fidelity_thr"]) * 0 + 0.15 * actor["params"]["select_k"]
        return c if rng.random() < min(1.0, 0.35 + 0.5 * eff) else copy.deepcopy(target)
    M = []
    for i, a in enumerate(cps):
        row = []
        for j, T in enumerate(tgt):
            qs = [Q(child_of(a, T), worlds, rng, args.budget, reps=1) - Q(T, worlds, rng, args.budget, reps=1) for _ in range(4)]
            row.append(round(statistics.mean(qs), 4))
        M.append(row)
    print("\n=== E3: actor x target matrix M(i,j) = child(actor_i, target_j) - keep(target_j) ===")
    print("            T0(U0)   T1(mid)  T2(late)")
    for i, row in enumerate(M):
        print("  actor %d   %s" % (i, "  ".join("%+.3f" % x for x in row)))
    print("  (within a COLUMN: does a later actor beat an earlier one? = producer advantage. across a ROW: target headroom.)")
    json.dump({"ceiling": ceiling, "trajectory": traj, "n_meaningful_revisions": n_meaningful,
               "first_step_gain_share": front_load, "verdict": verdict, "actor_target_matrix": M},
              open(os.path.join(args.outdir, "lab_engine.json"), "w"), indent=2)
    print("\nwrote", os.path.join(args.outdir, "lab_engine.json"))


if __name__ == "__main__":
    main()
