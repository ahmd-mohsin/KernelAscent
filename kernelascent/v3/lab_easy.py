"""EASY LABORATORY (adopted plan step 3): improve the EFFICIENCY OF EXPERIMENTATION.

The research task is to find the best operator variant in a pool under a FIXED experimentation BUDGET.
The improvable research procedure U has SEPARATE, coadapting components (Experiment 2: the compounding
channel is the proposal x evaluation INTERACTION, not either alone):
  PROPOSAL  : how candidates are chosen -- 'random' vs 'guided' (use cheap signal to prefer promising ones)
  EVALUATION: cache (skip re-testing a seen design), dedup (don't re-propose a seen design),
              staged (cheap-test first; expensive-test only if promising), + a stage threshold.
An inefficient initial procedure wastes budget (re-compiles duplicates, expensive-tests everything) and
so verifies fewer good designs -> lower research productivity Q. Efficiency FREES budget; guided proposal
SPENDS it well -- neither helps much alone (Exp 2), together they compound.

RECURSION (the F channel): revise(actor, target) SEARCHES the procedure space for a better child, and
EVALUATES each candidate child by running it -- using the ACTOR's OWN efficiency, so a more efficient actor
affords more/less-noisy child evaluations and picks a genuinely better child. A better child is itself more
efficient -> next revise is better -> F1,F2 can be > 0. The world is scripted, so a POSITIVE here is
LAB CALIBRATION (the substrate CAN exhibit coadaptive compounding, which the config loop could not); the
scientific test is a LIVE MODEL discovering these changes (Gate 4, model-backed, next).
"""
import os, sys, json, argparse, random, hashlib, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
from kernelascent.v3.core import run_lineage, aggregate_lineages, _mean_ci

C_CHEAP, C_EXP = 1, 4                 # cost units: cheap noisy test vs expensive exact (verified) test
SIGMA_CHEAP = 0.15                    # cheap-test noise


GUIDE_G = 4                            # guided proposal cheap-tests G options to rank -> costs G*C_CHEAP


def make_world(seed, n_designs=80, dup_rate=0.6, good_frac=0.04):
    """A research project: a pool of operator designs. GOOD designs are RARE (top good_frac) and the
    proposal draw is dominated by LOW-quality designs, so random search rarely finds a good one -> guided
    search (cheap-rank several) is needed to locate them, but guided is budget-hungry -> only affordable
    when evaluation is efficient. This makes proposal x evaluation COMPLEMENTARY (as Experiment 2 found)."""
    g = random.Random(seed)
    n_good = max(2, int(n_designs * good_frac))
    qual = [0.2 + 0.35 * g.random() for _ in range(n_designs - n_good)] + [0.85 + 0.15 * g.random() for _ in range(n_good)]
    g.shuffle(qual)
    good = {i for i in range(n_designs) if qual[i] >= 0.8}
    draw_len = int(n_designs / (1 - dup_rate))
    # RANDOM proposal draws only from NON-good designs -> random search (any efficiency) cannot REACH the
    # good ones. Only GUIDED proposal (samples the full pool + cheap-ranks) can access them. This makes
    # proposal (access) and evaluation (affording repeated guided access) attack DIFFERENT bottlenecks.
    non_good = [i for i in range(n_designs) if i not in good]
    draw = g.choices(non_good, k=draw_len)
    return {"qual": qual, "draw": draw, "n_designs": n_designs}


def _cheap(world, d, rng):
    return max(0.0, min(1.0, world["qual"][d] + rng.gauss(0, SIGMA_CHEAP)))


def develop(U, world, rng, budget=60):
    """Run U's research loop under a fixed budget; return best VERIFIED (expensive-tested) quality.
    Guided proposal is EXPENSIVE (ranks GUIDE_G options) but finds rare good designs; efficiency
    (cache/dedup/staged) frees the budget guided needs -> the two compound."""
    p = U["params"]
    guided = p.get("propose") == "guided"; cache = p.get("cache", 0); dedup = p.get("dedup", 0)
    staged = p.get("staged", 0); thr = p.get("stage_thr", 0.5)
    seen = {}; spent = 0.0; best = 0.0
    draw = world["draw"]
    while spent < budget:
        # PROPOSAL
        if guided:
            gg = int(p.get("guide_g", 3))          # guided BREADTH: rank gg options (more=better design, costs more)
            opts = [rng.randrange(world["n_designs"]) for _ in range(gg)]
            if dedup:
                opts = [d for d in opts if d not in seen] or opts
            ce = {d: _cheap(world, d, rng) for d in opts}
            spent += gg * C_CHEAP                  # guided ranking is budget-hungry (scales with breadth)
            d = max(opts, key=lambda x: ce[x]); cheap_d = ce[d]
        else:
            d = draw[rng.randrange(len(draw))]; cheap_d = None
        if dedup and d in seen:
            continue
        # EVALUATION
        if staged:
            if cheap_d is None:
                spent += C_CHEAP; cheap_d = _cheap(world, d, rng)
            if cheap_d < thr:                      # cheap-filter: skip expensive test on unpromising
                continue
        if cache and d in seen:
            q = seen[d]                            # free: reuse verified result
        else:
            spent += C_EXP
            q = world["qual"][d]; seen[d] = q      # expensive VERIFIED test
        best = max(best, q)
        if spent >= budget:
            break
    return best


# ------------------------------------------------------------------ procedure-space search (the improver)
_FLAGS = ["cache", "dedup", "staged"]


def _procedure_neighbors(params, rng):
    """Candidate CHANGES to a research procedure: toggle an efficiency flag, switch proposal, tweak thr."""
    outs = []
    for f in _FLAGS:
        c = dict(params); c[f] = 1 - c.get(f, 0); outs.append(c)
    c = dict(params); c["propose"] = "guided" if c.get("propose") != "guided" else "random"; outs.append(c)
    for dt in (-0.15, +0.15):
        c = dict(params); c["stage_thr"] = round(min(0.9, max(0.1, c.get("stage_thr", 0.5) + dt)), 2); outs.append(c)
    for dg in (-1, +1):                             # tune guided breadth (deeper rung: only pays once efficient+guided)
        c = dict(params); c["guide_g"] = int(min(8, max(2, c.get("guide_g", 3) + dg))); outs.append(c)
    rng.shuffle(outs); return outs


def _efficiency(params):
    """How many child-evaluation trials the actor affords per revision budget: efficient tools -> more
    trials -> less-noisy child-quality estimates -> better child selection. (cache/dedup/staged each help;
    guided proposal makes each trial more informative.)"""
    m = 1.0 + 0.6 * params.get("cache", 0) + 0.4 * params.get("dedup", 0) + 0.5 * params.get("staged", 0)
    if params.get("propose") == "guided":
        m += 0.4
    return m


def make_revise(worlds, rev_reps_base=2, eval_budget=80):
    """revise(actor, target): actor searches target's procedure neighbors and picks the best child, where
    the actor's EFFICIENCY sets how many reps it can afford to estimate each child's Q (less noise = better
    pick). Coadaptation: child Q depends on BOTH proposal and evaluation, so a better actor climbs both."""
    def revise(actor, target, rng):
        import copy
        cands = _procedure_neighbors(target["params"], rng)
        reps = max(1, int(round(rev_reps_base * _efficiency(actor["params"]))))
        n_cands = max(2, int(round(2 * _efficiency(actor["params"]))))     # efficient actor also screens more neighbors
        cands = cands[:n_cands]
        best_child, best_est = None, -1.0
        for c in cands:
            child = {"params": c}
            est = statistics.mean(develop(child, worlds[rng.randrange(len(worlds))], rng, eval_budget) for _ in range(reps))
            if est > best_est:
                best_est, best_child = est, child
        return copy.deepcopy(best_child if best_child else {"params": dict(target["params"])})
    return revise


INEFFICIENT_U0 = {"params": {"propose": "random", "cache": 0, "dedup": 0, "staged": 0, "stage_thr": 0.5, "guide_g": 3}}
EFFICIENT_REF = {"params": {"propose": "guided", "cache": 1, "dedup": 1, "staged": 1, "stage_thr": 0.5, "guide_g": 7}}


# ------------------------------------------------------------------ Gate 4: LIVE MODEL improves the procedure
LAB_TMPL = (
    "You tune a research procedure that searches a pool of operator variants under a fixed budget to find "
    "the fastest correct one. Higher score = better. Maximize the score. Knobs:\n"
    "  propose: 'random' (cheap, only reaches common variants) or 'guided' (costs budget, can reach rare fast ones)\n"
    "  cache 0/1 (reuse a tested result), dedup 0/1 (skip re-testing), staged 0/1 (cheap-filter before the costly test)\n"
    "  stage_thr 0.1-0.9 (cheap-filter cutoff), guide_g 2-8 (how many options guided ranks; higher finds better but costs more)\n"
    "Procedure to edit: {cur}\nMeasured score: {q:.3f}.{hint}\nReturn ONLY JSON with all seven keys.")


def make_model_revise(gen_fn, worlds, eval_budget, rng_seed=0):
    import re, copy
    def revise(actor, target, rng):
        qcur = statistics.mean(develop(target, worlds[rng.randrange(len(worlds))], rng, eval_budget) for _ in range(2))
        # inheritance channel: the actor passes only the settings that WORKED in its prior research
        # (guided + enabled efficiency flags); U0 has none -> no misleading anchor. Better actor -> better prior.
        good = [k for k in ("cache", "dedup", "staged") if actor["params"].get(k)]
        if actor["params"].get("propose") == "guided":
            good = ["guided"] + good
        hint = (" From your prior research these settings worked well: %s." % ", ".join(good)) if good else ""
        raw = gen_fn(LAB_TMPL.format(cur=json.dumps(target["params"]), q=qcur, hint=hint)) or ""
        m = re.search(r"\{.*\}", raw, re.S)
        child = copy.deepcopy(target["params"])
        if m:
            try:
                d = json.loads(m.group(0))
                if d.get("propose") in ("random", "guided"): child["propose"] = d["propose"]
                for k in ("cache", "dedup", "staged"):
                    if k in d: child[k] = 1 if d[k] in (1, True, "1", "true") else 0
                if isinstance(d.get("stage_thr"), (int, float)): child["stage_thr"] = min(0.9, max(0.1, float(d["stage_thr"])))
                if isinstance(d.get("guide_g"), (int, float)): child["guide_g"] = int(min(8, max(2, d["guide_g"])))
            except Exception:
                pass
        return {"params": child}
    return revise


def run_model(args):
    import curate_bedrock as CB
    cur = CB.Curator(args.api_model, args.region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    cur.resolve(); cur.resolve_reasoning(); gen_fn = lambda p: cur.generate(p); who = "api:" + args.api_model
    anchors = [make_world(s) for s in range(args.anchors)]
    worlds = [make_world(1000 + s) for s in range(args.anchors)]
    revise = make_model_revise(gen_fn, worlds, args.budget)
    print("LAB-EASY Gate4 %s anchors=%d lineages=%d" % (who, args.anchors, args.lineages), flush=True)
    results = []
    for s in range(args.lineages):
        r = run_lineage(INEFFICIENT_U0, develop, revise, anchors, random.Random(s), reps=args.reps)
        results.append(r)
        agg = aggregate_lineages(results)
        print("lin%d Q0=%.3f q1-q0=%+.3f F1=%+.3f N1=%+.3f F2=%+.3f" %
              (s, r.Q["U0"], r.q1_minus_q0, r.F1, r.N1, r.F2), flush=True)
        q0 = _mean_ci([x.Q["U0"] for x in results])
        json.dump({"who": who, "lineages": s + 1, "Q0": q0, "agg": agg},
                  open(os.path.join(args.outdir, "lab_easy_%s.json" % who.replace(":", "_").replace("/", "_").replace(".", "_")), "w"), indent=2)
    print("\n=== LAB-EASY Gate4 %s ===" % who)
    print("  Q0:", _mean_ci([x.Q["U0"] for x in results]))
    for k in ("q1_minus_q0", "F1", "N1", "F2", "N2"):
        print("  %-12s %s" % (k, aggregate_lineages(results)[k]))


# ------------------------------------------------------------------ tests / calibration
def calib(args):
    anchors = [make_world(s) for s in range(args.anchors)]
    rng = random.Random(0)
    def Q(U):
        return statistics.mean(develop(U, w, rng, args.budget) for w in anchors for _ in range(3))
    q_bad = Q(INEFFICIENT_U0); q_good = Q(EFFICIENT_REF)
    print("G1 opportunity: inefficient Q=%.3f -> efficient-reference Q=%.3f  dQ=%+.3f" % (q_bad, q_good, q_good - q_bad))
    # per-component (Experiment-2-style coadaptation check on the LAB)
    onlyprop = {"params": {**INEFFICIENT_U0["params"], "propose": "guided"}}
    onlyeval = {"params": {**INEFFICIENT_U0["params"], "cache": 1, "dedup": 1, "staged": 1}}
    qp, qe = Q(onlyprop), Q(onlyeval)
    inter = (q_good - q_bad) - (qp - q_bad) - (qe - q_bad)
    print("   coadaptation: d_prop=%+.3f d_eval=%+.3f d_joint=%+.3f INTERACTION=%+.3f"
          % (qp - q_bad, qe - q_bad, q_good - q_bad, inter))
    # causal recursion: does a better improver build a better next improver? (lineage F1/F2)
    worlds = [make_world(1000 + s) for s in range(args.anchors)]
    revise = make_revise(worlds, rev_reps_base=args.rev_reps, eval_budget=args.budget)
    results = [run_lineage(INEFFICIENT_U0, develop, revise, anchors, random.Random(s), reps=args.reps)
               for s in range(args.lineages)]
    agg = aggregate_lineages(results)
    q0 = _mean_ci([r.Q["U0"] for r in results])
    print("\nLINEAGE (n=%d): Q0=%.3f q1-q0=%s" % (args.lineages, q0["mean"], agg["q1_minus_q0"]))
    for k in ("F1", "N1", "F2", "N2"):
        print("  %-4s %s" % (k, agg[k]))
    out = {"g1_dQ": round(q_good - q_bad, 4), "coadapt_interaction": round(inter, 4),
           "Q0": q0, "agg": agg}
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(args.outdir, "lab_easy_calib.json"), "w"), indent=2)
    print("\nwrote", os.path.join(args.outdir, "lab_easy_calib.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--api-model", default=""); ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--anchors", type=int, default=6); ap.add_argument("--lineages", type=int, default=30)
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--rev-reps", type=int, default=2)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--outdir", default="/tmp/instance_storage/ka_data/lab_easy")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.calib:
        calib(args); return
    if args.api_model:
        run_model(args); return
    print("use --calib (model-free) or --api-model <id> (Gate 4: live model improves the procedure)")


if __name__ == "__main__":
    main()
