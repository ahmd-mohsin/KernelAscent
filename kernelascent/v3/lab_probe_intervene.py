"""PROBE-AS-INTERVENTION (self-verification): turn the correlational interp result (kernel correctness is
linearly decodable from hidden states at AUC~0.98) into a CAUSAL one.

We fit the correctness probe on TRAIN tasks (best-layer Fisher/LDA direction, same as lab_interp_probe), then on
held-out TEST tasks generate K candidates each and compare correct-kernel SELECTION at equal generation budget:
  RANDOM     — expected single-draw success (the model's natural generation rate)
  PROBE-TOP1 — pick the candidate with the highest probe score (self-verification)
  ORACLE     — any-correct-in-K (upper bound)
lift = PROBE-TOP1 − RANDOM. If probe selection lifts a SUB-2B model (natural rate ~0) to nonzero correct-kernel
yield at large K, the correctness wall is (partly) a DECODE artifact and self-verification is the missing RSI
primitive. Also emits a K-sweep: probe-top-n vs random-n "any correct" rate.

1 GPU, bf16, no-grad, truncated context, bs auto-cap for >=7B. Loads via W.build (PEFT; disable_adapter -> base).
"""
import argparse, json, os, math, statistics, torch
from kernelascent import agent_bench as AB
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK


def _auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]; neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg: return None
    ranks = sorted(range(len(scores)), key=lambda i: scores[i]); rk = {}
    for r, i in enumerate(ranks): rk[i] = r + 1
    sp = sum(rk[i] for i, y in enumerate(labels) if y == 1)
    auc = (sp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return max(auc, 1 - auc)


def _bs(model):
    return 1 if any(s in model for s in ("7B", "7b", "8B", "8b", "9B", "13b", "14B", "15b", "32B")) else 2


@torch.no_grad()
def _layer_feats(tok, mdl, code, dev, max_len=1024):
    ids = tok(code, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    with mdl.disable_adapter():
        out = mdl(**ids, output_hidden_states=True)
    return [h[0].float().mean(0).cpu() for h in out.hidden_states]   # [H] per layer (incl embeddings)


def _gen_grade(tok, mdl, srcs, K, model):
    """Return per-src list of (code, ok, sp_compiled) for K generations each."""
    gl = W.generate_batch(tok, mdl, srcs, K, max_new=900, adapter=False, bs=_bs(model))
    out = []
    for src, gens in zip(srcs, gl):
        codes = [c for c in (AB.extract_modelnew(t) for t in gens) if c]
        rows = []
        if codes:
            grades = W._grade_isolated(src, codes)
            for code, g in zip(codes, grades):
                g = g if isinstance(g, list) else [0, 0, 0]
                ok = bool((g + [0, 0, 0])[0]); sp = float((g + [0, 0, 0])[2] or 0)
                rows.append((code, ok, sp))
        out.append(rows)
    return out


def _rand_any(nc, n, K):
    """Expected prob >=1 correct when drawing n of K candidates uniformly (nc correct)."""
    if nc <= 0 or K <= 0: return 0.0
    if n >= K: return 1.0 if nc > 0 else 0.0
    if K - nc < n: return 1.0
    return 1.0 - math.comb(K - nc, n) / math.comb(K, n)


def run(args):
    tok, mdl = W.build(args.model, gpus=(0,)); mdl.eval(); dev = next(mdl.parameters()).device
    names = list(LK.TASKS)
    train_names, test_names = names[: args.n_train], names[args.n_train: args.n_train + args.n_test]
    print("PROBE-INTERVENE %s train=%d test=%d K=%d" % (args.model, len(train_names), len(test_names), args.K), flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    outf = os.path.join(args.outdir, "probe_intervene.json")

    # ---- 1) TRAIN: fit best-layer correctness probe (generate K each so weak models still accrue some correct) ----
    tr = _gen_grade(tok, mdl, [LK.TASKS[n] for n in train_names], args.K, args.model)
    feats, labels = [], []
    for rows in tr:
        for code, ok, _ in rows:
            try: feats.append(_layer_feats(tok, mdl, code, dev)); labels.append(1 if ok else 0)
            except Exception: pass
    nC, nI = sum(labels), len(labels) - sum(labels)
    print("train labeled: %d correct / %d incorrect" % (nC, nI), flush=True)
    result = {"model": args.model, "K": args.K, "train_correct": nC, "train_incorrect": nI}
    layer = w = None; auc = None
    if nC >= 2 and nI >= 2:
        L = len(feats[0]); best, bestauc, bestw = 0, -1, None
        for l in range(L):
            X = torch.stack([f[l] for f in feats])
            mu1 = X[[i for i, y in enumerate(labels) if y == 1]].mean(0)
            mu0 = X[[i for i, y in enumerate(labels) if y == 0]].mean(0)
            wl = mu1 - mu0; a = _auc((X @ wl).tolist(), labels)
            if a is not None and a > bestauc: best, bestauc, bestw = l, a, wl
        layer, w, auc = best, bestw, round(bestauc, 3)
        result.update(layer=layer, layer_frac=round(layer / (L - 1), 2), auc=auc)
        print("probe fit @ layer %d/%d AUC=%.3f" % (layer, L - 1, auc), flush=True)
    else:
        result.update(note="insufficient train labels to fit probe (wall); reporting random vs oracle only")
        print("wall: cannot fit probe; random/oracle only", flush=True)

    # ---- 2) TEST: generate K, score with probe, compare selection strategies ----
    te = _gen_grade(tok, mdl, [LK.TASKS[n] for n in test_names], args.K, args.model)
    rand_rate, probe_ok, oracle_ok, probe_sp = [], [], [], []
    within_auc = []            # per-task probe AUC among THAT task's candidates (Astra: separate from pooled AUC)
    Ns = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= args.K]
    sweep = {n: {"probe": [], "rand": []} for n in Ns}
    for rows in te:
        if not rows: continue
        K = len(rows); oks = [r[1] for r in rows]; nc = sum(oks)
        rand_rate.append(nc / K)                                  # single random draw success
        oracle_ok.append(1 if nc > 0 else 0)
        if w is not None:
            scored = []
            for code, ok, sp in rows:
                try: s = float(_layer_feats(tok, mdl, code, dev)[layer] @ w)
                except Exception: s = float("-inf")
                scored.append((s, ok, sp))
            # WITHIN-TASK ranking quality: does the probe rank THIS task's correct candidates above its incorrect ones?
            if 0 < nc < K:
                wa = _auc([s for s, _, _ in scored], [1 if ok else 0 for _, ok, _ in scored])
                if wa is not None: within_auc.append(wa)
            scored.sort(key=lambda t: -t[0])                      # high probe score first
            probe_ok.append(1 if scored[0][1] else 0)
            if scored[0][1]: probe_sp.append(scored[0][2])
            for n in Ns:
                topn = scored[:n]
                sweep[n]["probe"].append(1 if any(o for _, o, _ in topn) else 0)
                sweep[n]["rand"].append(_rand_any(nc, n, K))
    def _m(x): return round(statistics.mean(x), 3) if x else None
    result.update(
        n_test=len(rand_rate),
        random_correct_rate=_m(rand_rate),
        probe_top1_correct_rate=_m(probe_ok),
        oracle_bestofK_rate=_m(oracle_ok),
        lift=(round(_m(probe_ok) - _m(rand_rate), 3) if (probe_ok and rand_rate) else None),
        probe_pick_speedup=_m(probe_sp),
        within_task_auc=_m(within_auc),          # mean per-task ranking AUC (>0.5 = real within-task signal)
        n_within_task=len(within_auc),           # tasks with both correct & incorrect candidates
        k_sweep=[{"n": n, "probe_anycorrect": _m(sweep[n]["probe"]), "rand_anycorrect": _m(sweep[n]["rand"])} for n in Ns],
    )
    if within_auc:
        print("within-task AUC=%.3f over %d tasks (pooled AUC=%s)" % (_m(within_auc), len(within_auc), auc), flush=True)
    json.dump(result, open(outf, "w"), indent=2)
    print("RESULT random=%s probe_top1=%s oracle=%s lift=%s" %
          (result["random_correct_rate"], result["probe_top1_correct_rate"], result["oracle_bestofK_rate"], result["lift"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-train", type=int, default=10); ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "probe_intervene"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
