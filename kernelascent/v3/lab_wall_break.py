"""Experiment — DOES A CORRECTNESS PROBE BREAK THE CORRECTNESS WALL? (frontier-panel #1/#2, oral-grade)

For a wall-bound (sub-2B) model, correct kernels are rare, so weight-RSI has no SFT signal (empty set -> no drift).
Claim under test: probe-FILTERED sampling harvests more verified-correct kernels at a MATCHED EXECUTION BUDGET
(verification is the expensive, counted op; generation is cheap), turning an empty SFT set into a non-empty one and
enabling one round of drift where the baseline gets none.

Per wall task, execution budget E verifications:
  ARM A (baseline)      : generate E, verify all E.
  ARM B (probe-filter)  : generate N (=oversample*E), score all with the correctness probe (cheap fwd pass, NO
                          execution), verify only the top-E by probe score.
Both spend exactly E verifications/task. Report verified-correct yield A vs B, then SFT each arm's harvested pairs
(matched steps) on separate model copies and measure held-out gain + LoRA drift. B>A with drift_B>0 = wall broken.

Reuses lab_probe_intervene (probe fit, _layer_feats, _gen_grade) + lab_weight_rsi (build/sft/eval_tasks/_grade_isolated).
CLI: python -m kernelascent.v3.lab_wall_break --model <hf> --gpu 0 --fit 12 --wall 16 --held 12 --E 16 --oversample 8 --outdir <d>
"""
import os, sys, json, argparse, statistics, random
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent.v3 import lab_probe_intervene as PI
from kernelascent import agent_bench as AB
import torch


def _fit_probe(tok, mdl, dev, fit_srcs, K, model):
    """best-layer LDA correctness probe on fit tasks; returns (layer, w, auc) or (None,None,None) if wall."""
    tr = PI._gen_grade(tok, mdl, fit_srcs, K, model)
    feats, labels = [], []
    for rows in tr:
        for code, ok, _ in rows:
            try: feats.append(PI._layer_feats(tok, mdl, code, dev)); labels.append(1 if ok else 0)
            except Exception: pass
    nC, nI = sum(labels), len(labels) - sum(labels)
    if nC < 2 or nI < 2: return None, None, None, (nC, nI)
    L = len(feats[0]); best, bestauc, bestw = 0, -1, None
    for l in range(L):
        X = torch.stack([f[l] for f in feats])
        mu1 = X[[i for i, y in enumerate(labels) if y == 1]].mean(0)
        mu0 = X[[i for i, y in enumerate(labels) if y == 0]].mean(0)
        wl = mu1 - mu0; a = PI._auc((X @ wl).tolist(), labels)
        if a is not None and a > bestauc: best, bestauc, bestw = l, a, wl
    return best, bestw, round(bestauc, 3), (nC, nI)


def _harvest(tok, mdl, dev, srcs, E, oversample, layer, w, model):
    """per task: ARM A verify E of N gens; ARM B verify top-E-by-probe of N gens. Return (pairsA, pairsB, yieldA, yieldB)."""
    N = E * oversample
    gl = W.generate_batch(tok, mdl, srcs, N, max_new=900, adapter=False, bs=PI._bs(model))
    pairsA, pairsB, yA, yB = [], [], [], []
    for src, gens in zip(srcs, gl):
        codes = [c for c in (AB.extract_modelnew(t) for t in gens) if c]
        if not codes: yA.append(0.0); yB.append(0.0); continue
        # ARM A: first E (arbitrary order = unfiltered sampling)
        aCodes = codes[:E]
        gA = W._grade_isolated(src, aCodes)
        okA = [(c, bool((list(g) + [0,0,0])[0])) for c, g in zip(aCodes, gA)]
        cA = [c for c, ok in okA if ok]; yA.append(len(cA) / max(len(aCodes), 1))
        pairsA += [(src, c) for c in cA]
        # ARM B: score ALL codes by probe (no execution), verify top-E
        scored = []
        for c in codes:
            try: s = float(PI._layer_feats(tok, mdl, c, dev)[layer] @ w)
            except Exception: s = float("-inf")
            scored.append((s, c))
        scored.sort(key=lambda t: -t[0])
        bCodes = [c for _, c in scored[:E]]
        gB = W._grade_isolated(src, bCodes)
        okB = [(c, bool((list(g) + [0,0,0])[0])) for c, g in zip(bCodes, gB)]
        cB = [c for c, ok in okB if ok]; yB.append(len(cB) / max(len(bCodes), 1))
        pairsB += [(src, c) for c in cB]
    return pairsA, pairsB, yA, yB


def _drift(mdl):
    from peft import get_peft_model_state_dict
    return float(sum((v.detach().float() ** 2).sum() for v in get_peft_model_state_dict(mdl).values()) ** 0.5)


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    tok, mdl = W.build(args.model, gpus=(0,)); mdl.eval(); dev = next(mdl.parameters()).device
    names = list(LK.TASKS); random.Random(1).shuffle(names)
    fit = names[:args.fit]; wall = names[args.fit:args.fit + args.wall]; held = names[args.fit + args.wall:args.fit + args.wall + args.held]
    print("WALL-BREAK %s fit=%d wall=%d held=%d E=%d oversample=%d" % (args.model, len(fit), len(wall), len(held), args.E, args.oversample), flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    layer, w, auc, (nC, nI) = _fit_probe(tok, mdl, dev, [LK.TASKS[n] for n in fit], args.E, args.model)
    print("probe fit: layer=%s auc=%s (train %d correct/%d incorrect)" % (layer, auc, nC, nI), flush=True)
    res = {"model": args.model, "E": args.E, "oversample": args.oversample, "probe_auc": auc, "fit_correct": nC, "fit_incorrect": nI}
    if w is None:
        res["note"] = "probe unfittable on fit set (deep wall) — cannot run intervention"
        json.dump(res, open(os.path.join(args.outdir, "wall_break.json"), "w"), indent=2)
        print("WALL too deep to even fit a probe; abort", flush=True); return
    pairsA, pairsB, yA, yB = _harvest(tok, mdl, dev, [LK.TASKS[n] for n in wall], args.E, args.oversample, layer, w, args.model)
    C0, _, _, _, _ = W.eval_tasks(tok, mdl, held, args.E, adapter=False)   # frozen-base held baseline
    d0 = _drift(mdl)
    # SFT arm A then measure; reset adapter; SFT arm B then measure
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    init = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(mdl).items()}
    def sft_and_eval(pairs):
        set_peft_model_state_dict(mdl, {k: v.to(dev) for k, v in init.items()})
        if pairs:
            try: W.sft(tok, mdl, pairs, args.sft_steps)
            except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache()
        C, _, _, _, _ = W.eval_tasks(tok, mdl, held, args.E, adapter=True)
        return round(C, 3), round(_drift(mdl) - d0, 4)
    cA, drA = sft_and_eval(pairsA)
    cB, drB = sft_and_eval(pairsB)
    m = lambda x: round(statistics.mean(x), 3) if x else 0.0
    res.update(within_verify_yield_baseline=m(yA), within_verify_yield_probe=m(yB),
               yield_gain=round(m(yB) - m(yA), 3), n_pairs_baseline=len(pairsA), n_pairs_probe=len(pairsB),
               C0_held=round(C0, 3), C_held_after_baseline_sft=cA, C_held_after_probe_sft=cB,
               drift_baseline=drA, drift_probe=drB,
               wall_broken=bool(len(pairsB) > len(pairsA) and drB > 0 and cB >= cA))
    json.dump(res, open(os.path.join(args.outdir, "wall_break.json"), "w"), indent=2)
    print("YIELD base=%.3f probe=%.3f (+%.3f) | pairs base=%d probe=%d | C0=%.3f Csft base=%.3f probe=%.3f | drift base=%.4f probe=%.4f | WALL_BROKEN=%s" %
          (m(yA), m(yB), m(yB) - m(yA), len(pairsA), len(pairsB), C0, cA, cB, drA, drB, res["wall_broken"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--fit", type=int, default=12); ap.add_argument("--wall", type=int, default=16); ap.add_argument("--held", type=int, default=12)
    ap.add_argument("--E", type=int, default=16); ap.add_argument("--oversample", type=int, default=8)
    ap.add_argument("--sft-steps", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "wall_break"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
