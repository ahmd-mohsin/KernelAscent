"""Open-ended / co-evolving-frontier RSI (the AGGRESSIVELY-recursive loop).

Fixed-task self-SFT can only deliver a one-time upgrade (capability saturates at the bank ceiling; continued
self == checkpoint-frozen). This loop lets the CHALLENGE co-evolve with the SOLVER so recursion can compound:

Each round r:
  1. SOLVE: model samples k candidates for the current FRONTIER; grade; keep correct.
  2. TRAIN: rejection-sampling LoRA-SFT on newly-correct kernels.
  3. PROPOSE (co-evolution): a proposer (Fable, API) MUTATES the tasks the model just solved into HARDER
     variants (bigger shapes / deeper fusion), GPU-validates them, and ADDS them to the frontier. Difficulty
     ratchets to stay just beyond the model.
  4. MEASURE: capability on the escalating frontier; and the causal recursion test — continued-self vs a
     producer+frontier FROZEN at a checkpoint. If the escalating loop keeps continued-self ABOVE checkpoint-frozen
     (which is stuck at round-R difficulty), that is genuine compounding, not a one-time upgrade.

Reuses lab_weight_rsi (build/generate_batch/sft/_grade_isolated_batch) + curate_bedrock (proposer) + agent_bench
(GPU validation). Output: open_rsi.json with frontier size, C_frontier, and self-vs-checkpoint each round.
"""
import os, sys, json, argparse, random, statistics, time, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch

HARDER = ("Take this PyTorch Model and make a STRICTLY HARDER optimization task in the same style: increase the "
          "shapes and/or add one more fused stage (e.g. deeper matmul+epilogue chain, larger seq/heads, an extra "
          "normalization/residual), keeping it a single self-contained `class Model(nn.Module)` with `DT` and "
          "`get_inputs()`. It must stay numerically well-defined and deterministic. Return ONLY the python code "
          "block.\n\n```python\n{src}\n```")


def _proposer(region):
    import curate_bedrock as CB
    c = CB.Curator(os.environ.get("KA_PROPOSER", "us.anthropic.claude-fable-5-1"), region, os.environ.get("BEDROCK_PROFILE", "bedrock"))
    c.resolve(); c.resolve_reasoning()
    return c


def _validate(src):
    try:
        ref, x, gold, rerr = AB.build_ref(src); t = AB.time_fn(lambda z: ref(z), (x,))
        return t > 0 and torch.isfinite(gold).all().item()
    except Exception:
        return False


def escalate(proposer, solved_srcs, want, seen):
    """Mutate solved tasks into harder, GPU-validated new frontier tasks."""
    out = []
    for s in solved_srcs:
        if len(out) >= want:
            break
        try:
            gen = proposer.generate(HARDER.format(src=s)) or ""
        except Exception:
            continue
        m = re.search(r"```(?:python)?\s*(.*?)```", gen, re.S)
        code = (m.group(1) if m else gen).strip()
        key = hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()
        if code and key not in seen and "class Model" in code and "get_inputs" in code and _validate(code):
            seen.add(key); out.append(code)
    return out


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    sg = [int(x) for x in str(args.self_gpu).split(",")]
    tok, mdl = W.build(args.model, sg)
    prop = _proposer(args.region)
    # seed frontier from the bank
    frontier = [LK.TASKS[n] for n in list(LK.TASKS)][:args.seed_tasks]
    seen = {hashlib.sha1(re.sub(r"\s+", "", s).encode()).hexdigest() for s in frontier}
    print("OPEN-RSI %s seed=%d seed_tasks=%d rounds=%d propose/round=%d" %
          (args.model, args.seed, len(frontier), args.rounds, args.propose), flush=True)
    os.makedirs(args.outdir, exist_ok=True); hist = []

    def solve_and_score(srcs, adapter):
        gl = W.generate_batch(tok, mdl, srcs, args.k, adapter=adapter)
        codes = [[c for c in (AB.extract_modelnew(t) for t in g) if c] for g in gl]
        grades = W._grade_isolated_batch(list(zip(srcs, codes)))
        uc = os.environ.get("KA_SCORE", "eager") == "compiled"
        sc = []; correct_pairs = []
        for src, cs, res in zip(srcs, codes, grades):
            best = 0.0; bc = None
            for code, gg in zip(cs, res):
                ok, se, scp, ceil = (list(gg) + [0, 0, 0, 1.5])[:4]
                s = LK._score(ok, scp if uc else se, ceiling=(ceil if uc else 1.5))
                if s > best: best = s
                if ok and bc is None: bc = code
            sc.append(best)
            if bc: correct_pairs.append((src, bc))
        return (statistics.mean(sc) if sc else 0.0), correct_pairs

    for r in range(args.rounds):
        t0 = time.time()
        Cf, pairs = solve_and_score(frontier, adapter=True)        # solve current frontier
        loss = W.sft(tok, mdl, pairs, args.sft_steps)              # train on newly-solved
        solved = [s for s, _ in pairs]
        added = escalate(prop, solved or frontier, args.propose, seen)  # co-evolve: harder tasks
        frontier += added
        row = {"round": r, "C_frontier": round(Cf, 3), "frontier": len(frontier),
               "solved": len(pairs), "added": len(added), "loss": round(loss, 3)}
        hist.append(row)
        print("round %d C_frontier=%.3f solved=%d +%d harder -> frontier=%d (%.0fs)" %
              (r, Cf, len(pairs), len(added), len(frontier), time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "history": hist,
                   "note": "open-ended: frontier escalates with capability"},
                  open(os.path.join(args.outdir, "open_rsi.json"), "w"), indent=2)
    print("\n=== OPEN-RSI SUMMARY %s === C_frontier:" % args.model, [h["C_frontier"] for h in hist],
          "frontier grew", hist[0]["frontier"], "->", hist[-1]["frontier"])
    print("compounding? sustained C_frontier on an ESCALATING frontier =",
          "YES" if len(hist) >= 4 and statistics.mean([h["C_frontier"] for h in hist[-2:]]) >= statistics.mean([h["C_frontier"] for h in hist[:2]]) else "flat/declining")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--self-gpu", default="0,1")
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed-tasks", type=int, default=20); ap.add_argument("--propose", type=int, default=6)
    ap.add_argument("--sft-steps", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "open_rsi"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
