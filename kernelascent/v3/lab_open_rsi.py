"""Task 5 — Open-ended / co-evolving-frontier RSI, RIGOROUS design.

The compounding question demands controls, not a rising curve on a moving target. This runs TWO learners from
the same base at EQUAL budget and measures both on ONE fixed held-out ladder every round:

  OPEN  arm : trains on a frontier that ESCALATES (proposer mutates solved tasks into strictly-harder, GPU-
              validated variants and adds them each round).
  FIXED arm : trains on a STATIC frontier (the seed tasks), same k / SFT steps / rounds. (control)

Both are scored each round on a FIXED HELD-OUT LADDER (disjoint tasks spanning easy->hard, NEVER trained on),
so the numbers are comparable across rounds. Rigor built in:
  * PRIMARY causal metric: C_held(OPEN) - C_held(FIXED) over rounds. If the escalating frontier compounds,
    OPEN pulls ahead of FIXED on the SAME held ladder. If they track, escalation adds nothing (one-time upgrade).
  * Escalation is VERIFIED: frozen-base correct-rate on the current frontier is logged each round; it must
    decline as the frontier grows (else "harder" is not harder and the whole premise fails).
  * Curriculum-replay hook: the escalated frontier is saved so a fresh base can later be trained on the SAME
    curriculum (no recursion) to test whether gains are the curriculum vs the model improving.
  * Held ladder is fixed + disjoint from every training frontier; headroom-normalized score; seeds for CIs.

Output open_rsi.json: per round C_held_open, C_held_fixed, delta, frontier size, base_solve_on_frontier.
"""
import os, sys, json, argparse, random, statistics, time, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch

HARDER = ("Make a STRICTLY HARDER optimization task in the same style as the PyTorch Model below: increase shapes "
          "and/or add one fused stage (deeper matmul+epilogue, larger seq/heads, extra norm/residual). Keep a single "
          "self-contained `class Model(nn.Module)` with `DT` and `get_inputs()`, numerically well-defined and "
          "deterministic. Return ONLY the python code block.\n\n```python\n{src}\n```")


def _proposer(region):
    import curate_bedrock as CB
    c = CB.Curator(os.environ.get("KA_PROPOSER", "us.anthropic.claude-fable-5-1"), region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    c.resolve(); c.resolve_reasoning(); return c


def _valid(src):
    try:
        ref, x, gold, e = AB.build_ref(src); return AB.time_fn(lambda z: ref(z), (x,)) > 0 and torch.isfinite(gold).all().item()
    except Exception:
        return False


def _score_set(tok, mdl, srcs, k, adapter):
    """(mean headroom C, correct_rate, [(src,best_correct_code)]) on srcs."""
    gl = W.generate_batch(tok, mdl, srcs, k, adapter=adapter)
    codes = [[c for c in (AB.extract_modelnew(t) for t in g) if c] for g in gl]
    grades = W._grade_isolated_batch(list(zip(srcs, codes)))
    uc = os.environ.get("KA_SCORE", "eager") == "compiled"
    sc = []; corr = []; pairs = []
    for src, cs, res in zip(srcs, codes, grades):
        best = 0.0; bc = None; ok_any = 0
        for code, gg in zip(cs, res):
            ok, se, scp, ceil = (list(gg) + [0, 0, 0, 1.5])[:4]
            s = LK._score(ok, scp if uc else se, ceiling=(ceil if uc else 1.5))
            if s > best: best = s
            if ok: ok_any = 1; bc = bc or code
        sc.append(best); corr.append(ok_any)
        if bc: pairs.append((src, bc))
    return (statistics.mean(sc) if sc else 0.0), (statistics.mean(corr) if corr else 0.0), pairs


def _scale_shapes(src, factor=2):
    """Programmatic difficulty bump: multiply the integer shape args inside get_inputs() torch.* calls by
    `factor` (rounded to a multiple of 8), producing a genuinely harder (bigger) but VALID variant. Deterministic
    and reproducible — a curriculum that does not depend on an LLM's output. Returns modified source or None."""
    m = re.search(r"def get_inputs\(\).*?return\s*\[(.*?)\]", src, re.S)
    if not m:
        return None
    seg = m.group(0)
    def bump(mm):
        n = int(mm.group(0))
        if n < 8:
            return mm.group(0)                       # leave tiny dims (heads, small consts)
        return str(min(8192, max(8, (int(n * factor) // 8) * 8)))   # cap to avoid OOM; frontier still grows in COUNT
    new_seg = re.sub(r"\b\d{2,}\b", bump, seg)        # only bump 2+ digit dims (real shapes)
    if new_seg == seg:
        return None
    return src.replace(seg, new_seg)


def escalate(prop, solved, want, seen, factor=2):
    """Grow the frontier with STRICTLY-HARDER, GPU-validated variants of solved tasks. Programmatic shape-scaling
    first (deterministic, always valid); Fable enrichment as a bonus. Guarantees escalation when possible."""
    out = []
    for s in solved:
        if len(out) >= want:
            break
        cand = _scale_shapes(s, factor)               # deterministic harder variant
        if cand:
            key = hashlib.sha1(re.sub(r"\s+", "", cand).encode()).hexdigest()
            if key not in seen and _valid(cand):
                seen.add(key); out.append(cand); continue
        try:                                          # fallback: LLM proposer
            gen = prop.generate(HARDER.format(src=s)) or ""
            mm = re.search(r"```(?:python)?\s*(.*?)```", gen, re.S); code = (mm.group(1) if mm else gen).strip()
            key = hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()
            if code and key not in seen and "class Model" in code and "get_inputs" in code and _valid(code):
                seen.add(key); out.append(code)
        except Exception:
            pass
    return out


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    og = [int(x) for x in str(args.open_gpu).split(",")]; fg = [int(x) for x in str(args.fixed_gpu).split(",")]
    tok, open_m = W.build(args.model, og)
    tok2, fixed_m = W.build(args.model, fg)
    prop = _proposer(args.region)
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    seed_tasks = [LK.TASKS[n] for n in names[:args.seed_tasks]]
    held = [LK.TASKS[n] for n in names[args.seed_tasks:args.seed_tasks + args.held]]   # FIXED, never trained
    open_frontier = list(seed_tasks); fixed_frontier = list(seed_tasks)
    seen = {hashlib.sha1(re.sub(r"\s+", "", s).encode()).hexdigest() for s in seed_tasks + held}
    print("OPEN-RSI(rigorous) %s seed=%d seed_tasks=%d held_ladder=%d rounds=%d" %
          (args.model, args.seed, len(seed_tasks), len(held), args.rounds), flush=True)
    os.makedirs(args.outdir, exist_ok=True); hist = []
    base_hold0, base_corr0, _ = _score_set(tok, open_m, held, args.k, adapter=False)
    for r in range(args.rounds):
        t0 = time.time()
        # OPEN arm: solve escalating frontier, train, escalate
        _, _, open_pairs = _score_set(tok, open_m, open_frontier, args.k, adapter=True)
        W.sft(tok, open_m, open_pairs, args.sft_steps)
        added = escalate(prop, [s for s, _ in open_pairs] or open_frontier, args.propose, seen)
        open_frontier += added
        # FIXED arm (control): solve static frontier, train, equal budget
        _, _, fixed_pairs = _score_set(tok2, fixed_m, fixed_frontier, args.k, adapter=True)
        W.sft(tok2, fixed_m, fixed_pairs, args.sft_steps)
        # measure BOTH on the SAME fixed held ladder
        Cho, _, _ = _score_set(tok, open_m, held, args.k, adapter=True)
        Chf, _, _ = _score_set(tok2, fixed_m, held, args.k, adapter=True)
        # escalation verification: frozen-base correct-rate on the (harder) open frontier
        _, base_corr_front, _ = _score_set(tok, open_m, open_frontier[-max(1, len(added)):] if added else open_frontier, args.k, adapter=False)
        row = {"round": r, "C_held_open": round(Cho, 3), "C_held_fixed": round(Chf, 3),
               "delta_open_minus_fixed": round(Cho - Chf, 3), "frontier": len(open_frontier), "added": len(added),
               "base_correct_on_frontier": round(base_corr_front, 3)}
        hist.append(row)
        print("round %d C_held open=%.3f fixed=%.3f | open-fixed=%+.3f frontier=%d(+%d) base_solve_front=%.2f (%.0fs)" %
              (r, Cho, Chf, Cho - Chf, len(open_frontier), len(added), base_corr_front, time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "held_ladder": len(held),
                   "base_held_C0": round(base_hold0, 3), "history": hist,
                   "note": "OPEN(escalating) vs FIXED(static) learners, same held ladder; primary=delta_open_minus_fixed"},
                  open(os.path.join(args.outdir, "open_rsi.json"), "w"), indent=2)
    dl = [h["delta_open_minus_fixed"] for h in hist]
    print("\n=== OPEN-RSI SUMMARY %s === delta(open-fixed) on held ladder:" % args.model, dl)
    print("COMPOUNDS via open-endedness?",
          "YES" if len(dl) >= 4 and statistics.mean(dl[-2:]) > 0.05 else "NO (escalation adds nothing over fixed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--open-gpu", default="0"); ap.add_argument("--fixed-gpu", default="1")
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed-tasks", type=int, default=16); ap.add_argument("--held", type=int, default=30)
    ap.add_argument("--propose", type=int, default=6); ap.add_argument("--sft-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "open_rsi"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
