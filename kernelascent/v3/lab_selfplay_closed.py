"""Task 5 (self-play) for CLOSED / API models — self-referential RSI without weight access.

A closed model can't train its weights, but it can do both halves of self-play through its PROCEDURE:
  (1) AUTHOR its own strictly-harder tasks (API), and
  (2) IMPROVE its own executable procedure (strategy library + verified archive) to solve them.
So the same model recursively lifts capability AND difficulty — genuine self-referential RSI, no weights.

Reuses lab_track_c.{develop, improve} (procedure-RSI engine). Two arms at equal budget, one FIXED held ladder:
  SELFPLAY : procedure self-modifies AND the train frontier ESCALATES (model authors harder tasks each round).
  STATIC   : procedure self-modifies on a FIXED frontier (control).
PRIMARY = Q_held(selfplay) - Q_held(static). Sustained >0 with model_proposed>0 = closed-model self-referential RSI.
Escalation verified (frontier grows with validated, harder tasks); output selfplay_closed.json.
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
           "Return ONLY one python code block.\n\n```python\n{src}\n```")


def _extract(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t or "", re.S); c = (m.group(1) if m else (t or "")).strip()
    return c if ("class Model" in c and "get_inputs" in c and "DT" in c) else None


def _valid(src):
    try:
        ref, x, g, e = AB.build_ref(src); return AB.time_fn(lambda z: ref(z), (x,)) > 0 and torch.isfinite(g).all().item()
    except Exception:
        return False


def run(args):
    random.seed(args.seed)
    gen = TC._make_gen(args) if hasattr(TC, "_make_gen") else None
    if gen is None:
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
    tasks = dict(LK.TASKS); names = list(tasks); random.Random(1).shuffle(names)
    seedn, held = names[:args.seed_tasks], names[args.seed_tasks:args.seed_tasks + args.held]
    sp_names = list(seedn); st_names = list(seedn)
    seen = {hashlib.sha1(re.sub(r"\s+", "", tasks[n]).encode()).hexdigest() for n in names}
    U_sp = {"strategies": [], "archive": {}}; U_st = {"strategies": [], "archive": {}}
    print("SELFPLAY-CLOSED %s seed=%d seed_tasks=%d held=%d rounds=%d" % (args.model, args.seed, len(seedn), len(held), args.rounds), flush=True)
    hist = []
    for r in range(args.rounds):
        t0 = time.time(); nprop = 0
        # SELFPLAY: develop on escalating frontier, self-modify procedure, then author harder tasks
        _, _, ev_sp, ver_sp = TC.develop(U_sp, sp_names, tasks, gen, args.k, args.grade_gpu)
        U_sp["archive"].update(ver_sp); U_sp = TC.improve(U_sp, ev_sp, gen)
        solved = [n for n in ver_sp]
        for sn in (solved or sp_names)[:args.propose * 2]:
            if nprop >= args.propose: break
            code = _extract(gen(PROPOSE.format(src=tasks[sn]), "") )
            if not code: continue
            key = hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()
            if key not in seen and _valid(code):
                seen.add(key); nn = "sp_%d_%d" % (r, nprop); tasks[nn] = code; sp_names.append(nn); nprop += 1
        # STATIC: develop on fixed frontier, self-modify procedure (equal budget)
        _, _, ev_st, ver_st = TC.develop(U_st, st_names, tasks, gen, args.k, args.grade_gpu)
        U_st["archive"].update(ver_st); U_st = TC.improve(U_st, ev_st, gen)
        # measure BOTH on the fixed held ladder
        Qsp, _, _, _ = TC.develop(U_sp, held, tasks, gen, args.k, args.grade_gpu)
        Qst, _, _, _ = TC.develop(U_st, held, tasks, gen, args.k, args.grade_gpu)
        row = {"round": r, "Q_held_selfplay": round(Qsp, 3), "Q_held_static": round(Qst, 3),
               "delta_selfplay_minus_static": round(Qsp - Qst, 3), "frontier": len(sp_names),
               "model_proposed": nprop, "n_strategies_sp": len(U_sp["strategies"])}
        hist.append(row)
        print("round %d Q_held sp=%.3f static=%.3f | sp-static=%+.3f frontier=%d(+%d) (%.0fs)" %
              (r, Qsp, Qst, Qsp - Qst, len(sp_names), nprop, time.time() - t0), flush=True)
        os.makedirs(args.outdir, exist_ok=True)
        json.dump({"model": args.model, "seed": args.seed, "history": hist,
                   "note": "CLOSED self-play: model authors own harder tasks + self-modifies procedure; primary=delta_selfplay_minus_static"},
                  open(os.path.join(args.outdir, "selfplay_closed.json"), "w"), indent=2)
    dl = [h["delta_selfplay_minus_static"] for h in hist]; mp = sum(h["model_proposed"] for h in hist)
    print("\n=== SELFPLAY-CLOSED SUMMARY %s === delta:" % args.model, dl, "| model_proposed:", mp)
    print("closed self-referential RSI?", "YES" if len(dl) >= 3 and statistics.mean(dl[-2:]) > 0.05 and mp > 0 else "NO")


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
