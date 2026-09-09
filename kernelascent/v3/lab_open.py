"""OPEN-ENDED research engine (adopted-review §4/§5; the fix for structural front-loading).

Closed knob-spaces read F₂≈0 by construction: greedy improvement converges to a ceiling, first step
largest. The fix is OPEN-ENDEDNESS — the procedure is a GROWING ARCHIVE of composable skills/abstractions,
where unlocking skill k CREATES the opportunity to unlock k+1 (which requires k). Improvement literally
creates new improvement opportunities, and a richer archive makes the improver BETTER at discovering the
next one (library-learning: more building blocks → easier to build the next). This is the DGM/AlphaEvolve
archive-of-growing-capabilities shape.

State U = {"archive": set(skill ids)}. develop(U, task): Q rewards how much of the (valued) task
distribution the archive can solve. revise(actor, target): the actor uses its OWN archive richness (search
competence) to unlock ONE new skill whose prerequisites are already in the TARGET archive, adding it to the
child. A better actor unlocks a more valuable/deeper skill more reliably → F>0, and because the ladder is
deep it REPEATS → F₂ can be >0. Plugs into core.run_lineage (same estimators). This is a CALIBRATION that
an open-ended substrate CAN exhibit compounding (vs closed lab_engine); the live-model discovery is next.
"""
import os, sys, json, argparse, random, statistics, copy
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci

K = 14                                   # skills 0..K-1; deeper = more valuable
PREREQ = {k: ({k - 1} if k > 0 else set()) | ({k - 2} if k >= 4 and k % 3 == 0 else set()) for k in range(K)}
VALUE = {k: (k + 1) / K for k in range(K)}
BASE = {0, 1}                            # U0 starts with base skills


def make_tasks(seed, n=56):
    g = random.Random(seed)
    # tasks require a skill; mild depth weighting so unlocking mid/deep skills gives visible, spread gains
    return [g.choices(range(K), weights=[1.0 / (1 + 0.15 * k) for k in range(K)])[0] for _ in range(n)]


def develop(U, task, rng):
    # run_lineage passes ONE task (a skill id) at a time; solved iff that skill is in the archive
    return VALUE[task] if task in U["params"]["archive"] else 0.0


def make_revise(p_success=0.9):
    def revise(actor, target, rng):
        child = copy.deepcopy(target); arch = child["params"]["archive"]
        actor_arch = actor["params"]["archive"]
        # OPEN-ENDED CORE: the ACTOR can only BUILD a skill whose prerequisites are in the ACTOR's OWN
        # archive (you must know the components to compose the next abstraction); the result is installed on
        # the target. A richer actor can build DEEPER skills a poorer actor cannot -> newer producer builds
        # a better child on the SAME target -> F>0; the child is richer -> next actor builds deeper -> F2>0.
        buildable = [k for k in range(K) if k not in arch and PREREQ[k] <= actor_arch]
        if not buildable:
            return child
        k = max(buildable, key=lambda x: VALUE[x])
        if rng.random() < p_success:
            arch.add(k)
        return child
    return revise


def U0():
    return {"params": {"archive": set(BASE)}}


# ---------------------------------------------------------------- experiments
def run(args):
    tasks = make_tasks(0)
    rng = random.Random(0)
    revise = make_revise()
    Qmean = lambda U: statistics.mean(develop(U, t, rng) for t in tasks)

    # E1: improvement TRAJECTORY under a single actor that re-uses its growing archive to unlock more
    print("=== E1: open-ended improvement trajectory ===")
    U = U0(); q_prev = Qmean(U); traj = [("0", q_prev, 0.0, sorted(U["params"]["archive"]))]
    for s in range(1, args.steps + 1):
        U = revise(U, U, rng); q = Qmean(U)
        traj.append((str(s), q, q - q_prev, sorted(U["params"]["archive"]))); q_prev = q
    for st, q, g, arch in traj:
        print("  step %s  Q=%.3f  gain=%+.3f  archive=%s" % (st, q, g, arch))
    gains = [g for _, _, g, _ in traj[1:]]
    n_meaningful = sum(1 for g in gains if g >= 0.02)
    share = (gains[0] / sum(gains)) if sum(gains) > 1e-9 else 1.0
    print("  meaningful revisions (gain>=0.02): %d/%d   first-step share: %.0f%%  -> %s"
          % (n_meaningful, len(gains), 100 * share,
             "NON-FRONT-LOADED" if n_meaningful >= 3 and share < 0.6 else "front-loaded"))

    # F1/F2 via run_lineage (many lineages for CIs)
    results = [run_lineage(U0(), develop, revise, tasks, random.Random(s), reps=args.reps)
               for s in range(args.lineages)]
    agg = aggregate_lineages(results)
    print("\n=== causal decomposition (open-ended, n=%d lineages) ===" % args.lineages)
    for k in ("q1_minus_q0", "N1", "F1", "N2", "F2"):
        print("  %-12s %s" % (k, agg[k]))

    # CONTROL: frozen-archive + growing "memory" (a no-op knowledge blob) -> should NOT compound
    def revise_frozen(actor, target, rng):
        child = copy.deepcopy(target); child["params"].setdefault("mem", 0)
        child["params"]["mem"] += 1                       # memory grows, ARCHIVE (capability) does not
        return child
    res_fz = [run_lineage(U0(), develop, revise_frozen, tasks, random.Random(s), reps=args.reps)
              for s in range(args.lineages)]
    agg_fz = aggregate_lineages(res_fz)
    print("\n=== control: frozen archive + growing memory (should be ~0) ===")
    for k in ("q1_minus_q0", "F1", "F2"):
        print("  %-12s %s" % (k, agg_fz[k]))

    out = {"trajectory": [(s, round(q, 4), round(g, 4), a) for s, q, g, a in traj],
           "n_meaningful": n_meaningful, "first_step_share": round(share, 3),
           "open_ended": {k: agg[k] for k in ("q1_minus_q0", "N1", "F1", "N2", "F2", "rescue_minus_revert")},
           "control_frozen_archive": {k: agg_fz[k] for k in ("q1_minus_q0", "F1", "F2")}}
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(args.outdir, "lab_open.json"), "w"), indent=2, default=list)
    print("\nwrote", os.path.join(args.outdir, "lab_open.json"))
    fr = agg["F1"]["ci95"]; f2 = agg["F2"]["ci95"]
    print("\nHEADLINE: F1 %s  F2 %s -> %s" % (fr, f2,
          "OPEN-ENDED SUBSTRATE COMPOUNDS (F1 & F2 CIs > 0)"
          if fr and f2 and fr[0] > 0 and f2[0] > 0 else "no resolved compounding"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineages", type=int, default=40); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/lab_open")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
