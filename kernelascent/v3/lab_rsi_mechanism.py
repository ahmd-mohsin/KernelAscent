"""RSI MECHANISM probe — WHY a model fails to compound, or HOW it succeeds.

Compounding (Task 1-5) tells you IF the curve rises. This tells you WHY. It runs one weight-RSI learner and,
every round, records mechanistic signals whose trajectories diagnose the outcome:

  1. gen_diversity   distinct-2 + pairwise-dissimilarity of the k self-generations (mode collapse from self-SFT
                     shows up as this -> 0: the model stops exploring, so nothing new can be learned).
  2. gen_entropy     mean per-token predictive entropy on a fixed probe prompt (over-confident narrowing).
  3. drift_total     L2 norm of the LoRA-adapter change since last round (drift -> 0 = weights frozen = ceiling
                     reached; drift high but flat score = churning without gain).
  4. drift_by_depth  same L2 split into early / mid / late transformer blocks (WHERE learning localizes).
  5. retention       accuracy on a FROZEN probe of already-solved tasks (catastrophic forgetting: solving new
                     tasks by overwriting old skill, so the net capability never accumulates).
  6. transfer_gap    C_train - C_held on a disjoint held ladder (memorization vs genuine generalization).
  7. sft_loss        final SFT loss each round (learning-signal saturation).

At the end it emits a VERDICT that attributes the outcome to a mechanism:
  FAIL: diversity_collapse | forgetting | drift_saturation | no_transfer | no_headroom
  PASS: sustained_drift + retained + positive_transfer  (and which depth carries it).
Model-agnostic; run across families/sizes to compare failure modes. Output rsi_mechanism.json.
"""
import os, sys, json, argparse, random, statistics, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent import agent_bench as AB
import torch


def _distinct2(texts):
    grams = set(); tot = 0
    for t in texts:
        toks = (t or "").split()
        for i in range(len(toks) - 1):
            grams.add((toks[i], toks[i + 1])); tot += 1
    return len(grams) / tot if tot else 0.0


def _dissim(texts):
    """1 - mean pairwise token-set Jaccard over the generations (0 = identical, 1 = disjoint)."""
    sets = [set((t or "").split()) for t in texts if t]
    if len(sets) < 2: return 0.0
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            sims.append(len(sets[i] & sets[j]) / len(u) if u else 0.0)
    return 1.0 - (statistics.mean(sims) if sims else 0.0)


def _entropy(tok, mdl, prompt, dev):
    try:
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            lg = mdl(**ids).logits[0, -1].float()
        p = torch.softmax(lg, -1); p = p[p > 0]
        return float(-(p * p.log()).sum().item())
    except Exception:
        return float("nan")


def _adapter_vec(mdl):
    """Concatenated LoRA params grouped by (bucket) using the layer index in the param name."""
    early, mid, late, tot = [], [], [], []
    ln = [int(s) for n, _ in mdl.named_parameters() if "lora" in n.lower() for s in __import__("re").findall(r"\.(\d+)\.", n)]
    maxl = max(ln) if ln else 1
    for n, p in mdl.named_parameters():
        if "lora" not in n.lower() or not p.requires_grad: continue
        v = p.detach().float().flatten().cpu(); tot.append(v)
        idx = __import__("re").findall(r"\.(\d+)\.", n)
        li = int(idx[0]) if idx else 0
        (early if li < maxl / 3 else mid if li < 2 * maxl / 3 else late).append(v)
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(1)
    return cat(tot), cat(early), cat(mid), cat(late)


def _norm_delta(a, b):
    n = min(a.numel(), b.numel())
    return float((a[:n] - b[:n]).norm().item()) if n else 0.0


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    gpus = [int(x) for x in str(args.gpu).split(",")]
    tok, mdl = W.build(args.model, gpus)
    dev = next(mdl.parameters()).device
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    train = [LK.TASKS[n] for n in names[:args.train]]
    probe = [LK.TASKS[n] for n in names[:max(4, args.train // 2)]]          # frozen retention probe (early tasks)
    held = [LK.TASKS[n] for n in names[args.train:args.train + args.held]]  # disjoint transfer ladder
    probe_prompt = W._chat(tok, W._prompt(train[0]))                       # same prompt template the loop uses
    print("RSI-MECH %s seed=%d train=%d held=%d rounds=%d" % (args.model, args.seed, len(train), len(held), args.rounds), flush=True)
    os.makedirs(args.outdir, exist_ok=True); hist = []; prev = None

    def score(srcs):
        gl = W.generate_batch(tok, mdl, srcs, args.k, adapter=True)
        codes = [[c for c in (AB.extract_modelnew(t) for t in g) if c] for g in gl]
        grades = W._grade_isolated_batch(list(zip(srcs, codes)))
        sc = []; corr = []; pairs = []
        for src, cs, res in zip(srcs, codes, grades):
            best = 0.0; ok_any = 0; bc = None
            for code, gg in zip(cs, res):
                ok, se, scp, ceil = (list(gg) + [0, 0, 0, 1.5])[:4]
                best = max(best, LK._score(ok, se, ceiling=1.5))
                if ok: ok_any = 1; bc = bc or code
            sc.append(best); corr.append(ok_any)
            if bc: pairs.append((src, bc))
        return statistics.mean(sc), statistics.mean(corr), pairs, gl

    for r in range(args.rounds):
        t0 = time.time()
        Ctr, corr_tr, pairs, gens = score(train)
        flat = [t for g in gens for t in g]
        div2 = _distinct2(flat); diss = statistics.mean([_dissim(g) for g in gens]) if gens else 0.0
        ent = _entropy(tok, mdl, probe_prompt, dev)
        loss = W.sft(tok, mdl, pairs, args.sft_steps)
        try: loss = float(loss)
        except Exception: loss = float("nan")
        Ch, _, _, _ = score(held); _, ret, _, _ = score(probe)
        tv, ev, mv, lv = _adapter_vec(mdl)
        if prev is None:
            dtot = dearly = dmid = dlate = float("nan")
        else:
            dtot = _norm_delta(tv, prev[0]); dearly = _norm_delta(ev, prev[1]); dmid = _norm_delta(mv, prev[2]); dlate = _norm_delta(lv, prev[3])
        prev = (tv, ev, mv, lv)
        row = {"round": r, "C_train": round(Ctr, 3), "C_held": round(Ch, 3), "transfer_gap": round(Ctr - Ch, 3),
               "retention": round(ret, 3), "gen_distinct2": round(div2, 3), "gen_dissim": round(diss, 3),
               "gen_entropy": round(ent, 3), "drift_total": round(dtot, 4) if dtot == dtot else None,
               "drift_early": round(dearly, 4) if dearly == dearly else None,
               "drift_mid": round(dmid, 4) if dmid == dmid else None,
               "drift_late": round(dlate, 4) if dlate == dlate else None, "sft_loss": round(loss, 4) if loss == loss else None}
        hist.append(row)
        print("r%d Ctr=%.3f Chd=%.3f gap=%+.3f ret=%.2f div2=%.3f diss=%.2f ent=%.2f drift=%s loss=%s (%.0fs)" %
              (r, Ctr, Ch, Ctr - Ch, ret, div2, diss, ent, ("%.3f" % dtot if dtot == dtot else "-"),
               ("%.3f" % loss if loss == loss else "-"), time.time() - t0), flush=True)
        json.dump({"model": args.model, "seed": args.seed, "history": hist}, open(os.path.join(args.outdir, "rsi_mechanism.json"), "w"), indent=2)

    # --- verdict: attribute the outcome to a mechanism ---
    def series(k): return [h[k] for h in hist if h.get(k) is not None]
    ch = series("C_held"); reasons = []
    rose = len(ch) >= 3 and (ch[-1] - ch[0]) > 0.03
    d2 = series("gen_distinct2"); ds = series("drift_total"); rt = series("retention"); tg = series("transfer_gap")
    if d2 and d2[-1] < 0.15 and (len(d2) < 2 or d2[-1] < d2[0] * 0.6): reasons.append("diversity_collapse")
    if rt and rt[-1] < 0.5 and (len(rt) < 2 or rt[-1] < rt[0] - 0.15): reasons.append("forgetting")
    if ds and len(ds) >= 3 and statistics.mean(ds[-2:]) < 0.15 * (ds[0] or 1): reasons.append("drift_saturation")
    if tg and statistics.mean(tg[-2:]) > 0.25: reasons.append("no_transfer")
    if not rose and not reasons: reasons.append("no_headroom")
    verdict = {"compounded": rose, "mechanisms": reasons,
               "carrying_depth": (lambda e, m, l: ["early", "mid", "late"][max(range(3), key=lambda i: [e, m, l][i])] if (e or m or l) else None)(
                   sum(series("drift_early")), sum(series("drift_mid")), sum(series("drift_late")))}
    json.dump({"model": args.model, "seed": args.seed, "history": hist, "verdict": verdict},
              open(os.path.join(args.outdir, "rsi_mechanism.json"), "w"), indent=2)
    print("\n=== RSI-MECH VERDICT %s ===" % args.model, json.dumps(verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--train", type=int, default=16); ap.add_argument("--held", type=int, default=24)
    ap.add_argument("--sft-steps", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "rsi_mech"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
