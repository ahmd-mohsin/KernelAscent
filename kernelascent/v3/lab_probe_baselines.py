"""P6 — PROBE vs CHEAP SELECTORS AT MATCHED BUDGET (finalize the probe as a rigorous secondary negative).

Both panels: "probe-vs-random is not a result; probe-vs-logprob-rerank is." At a fixed candidate budget K, for each
test task we generate K kernels, grade them (ground truth), and compare TOP-1 selection by several cheap selectors
against the correctness probe and the oracle:
  RANDOM            : expected single-draw correct rate (nc/K).
  LOGPROB           : pick the candidate with highest mean completion log-prob (model confidence).
  LENGTH            : pick the shortest parseable candidate (a trivial prior).
  COMPILE-FILTER    : keep only candidates that statically parse+compile, then random among them.
  PROBE (top-1)     : linear correctness-probe best-layer score (self-verification).
  ORACLE            : best-of-K (any correct).
Headline metric = fraction of the random->oracle gap closed by each selector: (sel - random)/(oracle - random).
If PROBE does not beat LOGPROB/COMPILE at matched budget, "decodable correctness info doesn't translate into useful
matched-budget selection" — a clean, general secondary negative. Also reports within-task AUC for probe & logprob.

Reuses lab_probe_intervene (probe fit, _gen_grade, _layer_feats, _auc) + lab_weight_rsi (build, _chat, _prompt).
CLI: python -m kernelascent.v3.lab_probe_baselines --model <hf> --gpu 0 --n-train 12 --n-test 14 --K 24 --outdir <d>
"""
import os, sys, json, argparse, statistics, math, random, py_compile, tempfile
from kernelascent.v3 import lab_weight_rsi as W
from kernelascent.v3 import lab_kernel as LK
from kernelascent.v3 import lab_probe_intervene as PI
from kernelascent import agent_bench as AB
import torch


def _mean_logprob(tok, mdl, src, code, dev):
    """mean per-token log-prob of the completion (code) given the task prompt, under the model (adapter off)."""
    pr = W._chat(tok, W._prompt(src)); comp = "```python\n" + code.strip() + "\n```" + tok.eos_token
    pids = tok(pr, add_special_tokens=False)["input_ids"]; cids = tok(comp, add_special_tokens=False)["input_ids"]
    ids = (pids + cids)[:1536]
    if len(cids) == 0: return -1e9
    x = torch.tensor([ids]).to(dev)
    with torch.no_grad(), mdl.disable_adapter():
        logits = mdl(x).logits[0]                                   # [T, V]
    lp = torch.log_softmax(logits[:-1].float(), -1)
    tgt = x[0, 1:]
    start = max(len(pids) - 1, 0)
    comp_lp = lp[start:, :].gather(-1, tgt[start:].unsqueeze(-1)).squeeze(-1)
    return float(comp_lp.mean()) if comp_lp.numel() else -1e9


def _compiles(code):
    fd, p = tempfile.mkstemp(suffix=".py"); os.close(fd)
    try:
        open(p, "w").write(code)
        py_compile.compile(p, doraise=True); ok = True
    except Exception:
        ok = False
    finally:
        os.remove(p)
    return ok


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    tok, mdl = W.build(args.model, gpus=tuple(int(x) for x in str(args.gpu).split(","))); mdl.eval()
    dev = next(mdl.parameters()).device
    names = list(LK.TASKS); train, test = names[:args.n_train], names[args.n_train:args.n_train + args.n_test]
    print("PROBE-BASELINES %s train=%d test=%d K=%d" % (args.model, len(train), len(test), args.K), flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    # fit best-layer probe on TRAIN (reuse lab_probe_intervene logic)
    tr = PI._gen_grade(tok, mdl, [LK.TASKS[n] for n in train], args.K, args.model)
    feats, labels = [], []
    for rows in tr:
        for code, ok, _ in rows:
            try: feats.append(PI._layer_feats(tok, mdl, code, dev)); labels.append(1 if ok else 0)
            except Exception: pass
    layer = w = auc = None
    if sum(labels) >= 2 and len(labels) - sum(labels) >= 2:
        L = len(feats[0]); best, ba, bw = 0, -1, None
        for l in range(L):
            X = torch.stack([f[l] for f in feats])
            wl = X[[i for i, y in enumerate(labels) if y]].mean(0) - X[[i for i, y in enumerate(labels) if not y]].mean(0)
            a = PI._auc((X @ wl).tolist(), labels)
            if a is not None and a > ba: best, ba, bw = l, a, wl
        layer, w, auc = best, bw, round(ba, 3)
    print("probe fit layer=%s auc=%s" % (layer, auc), flush=True)
    # TEST: score each candidate by every selector
    sel = {k: [] for k in ("random", "logprob", "length", "compile", "probe", "oracle")}
    wauc_probe, wauc_lp = [], []
    te = PI._gen_grade(tok, mdl, [LK.TASKS[n] for n in test], args.K, args.model)
    for rows in te:
        if not rows: continue
        oks = [r[1] for r in rows]; nc = sum(oks); K = len(rows)
        sel["random"].append(nc / K); sel["oracle"].append(1 if nc else 0)
        src = None  # per-task src via matching (rows carry code only); recompute selectors needing src below
        # find this task's src: PI._gen_grade preserves order of test, so map by index
        # (simpler: recompute scores using the code + the task src we iterate in parallel)
        # -- handled below with test_srcs
        codes = [r[0] for r in rows]
        # length: shortest parseable
        lp_scores = None
        sel["length"].append(1 if any(oks) and oks[min(range(K), key=lambda i: len(codes[i]))] else 0)
        # compile-filter then random
        comp_ok = [i for i in range(K) if _compiles(codes[i])]
        sel["compile"].append((sum(oks[i] for i in comp_ok) / len(comp_ok)) if comp_ok else 0.0)
    # second pass needs task srcs for logprob + probe (recompute cleanly)
    test_srcs = [LK.TASKS[n] for n in test]
    for src, rows in zip(test_srcs, te):
        if not rows: continue
        codes = [r[0] for r in rows]; oks = [r[1] for r in rows]; K = len(rows)
        # logprob rerank
        lps = [_mean_logprob(tok, mdl, src, c, dev) for c in codes]
        li = max(range(K), key=lambda i: lps[i]); sel["logprob"].append(1 if oks[li] else 0)
        if 0 < sum(oks) < K:
            a = PI._auc(lps, [1 if o else 0 for o in oks]);  wauc_lp.append(a) if a is not None else None
        # probe rerank
        if w is not None:
            ps = []
            for c in codes:
                try: ps.append(float(PI._layer_feats(tok, mdl, c, dev)[layer] @ w))
                except Exception: ps.append(-1e9)
            pi = max(range(K), key=lambda i: ps[i]); sel["probe"].append(1 if oks[pi] else 0)
            if 0 < sum(oks) < K:
                a = PI._auc(ps, [1 if o else 0 for o in oks]); wauc_probe.append(a) if a is not None else None
    m = lambda x: round(statistics.mean(x), 3) if x else None
    rnd, orc = m(sel["random"]), m(sel["oracle"])
    def gap(s):
        v = m(sel[s]);  return round((v - rnd) / (orc - rnd), 3) if (v is not None and orc and orc > rnd) else None
    res = {"model": args.model, "K": args.K, "n_test": len(sel["random"]), "probe_auc": auc,
           "rates": {k: m(sel[k]) for k in sel},
           "gap_closed": {k: gap(k) for k in ("logprob", "length", "compile", "probe")},
           "within_task_auc_probe": m(wauc_probe), "within_task_auc_logprob": m(wauc_lp)}
    json.dump(res, open(os.path.join(args.outdir, "probe_baselines.json"), "w"), indent=2)
    print("RATES", json.dumps(res["rates"]), flush=True)
    print("GAP_CLOSED", json.dumps(res["gap_closed"]), "| within-task AUC probe=%s logprob=%s" %
          (res["within_task_auc_probe"], res["within_task_auc_logprob"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-train", type=int, default=12); ap.add_argument("--n-test", type=int, default=14)
    ap.add_argument("--K", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("KA_DATA_DIR", "/tmp/instance_storage/ka_data"), "probe_baselines"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
