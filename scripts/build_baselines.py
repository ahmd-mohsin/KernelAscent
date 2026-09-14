#!/usr/bin/env python3
"""Build the 03c recursion-gain board (docs/data/baselines.json models[]) from per-model baseline runs.
recursion_gain = weight-RSI final C  −  max(best_of_k, self_refine, retrieval)  (all frozen, matched budget).
Merges: existing docs/data/baselines.json models (preserve hand-added), + results/raw/baselines_*.json (raw
per-model {model, results:{method:{C}}}), + rsi_C from docs/data/tier_speed_rsi.json (weight-RSI C_self)."""
import json, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw"); D = os.path.join(ROOT, "docs", "data")

def short(n): return n.split("/")[-1].replace("-Instruct", "").replace("-Chat", "")

def build():
    # rsi_C (weight-RSI final held-out C) per model, from the T2 board
    rsi = {}
    try:
        for m in json.load(open(os.path.join(D, "tier_speed_rsi.json"))).get("models", []):
            rsi[short(m["model"])] = m.get("C_self")
    except Exception: pass
    # start from existing board (preserve prior rows), keyed by model
    board = {}
    try:
        for m in json.load(open(os.path.join(D, "baselines.json"))).get("models", []):
            board[short(m["model"])] = m
    except Exception: pass
    # overlay raw per-model baseline runs
    for f in glob.glob(os.path.join(RAW, "baselines_*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        sm = short(d.get("model", os.path.basename(f)[10:-5])); res = d.get("results", {})
        def C(k):
            r = res.get(k); return r.get("C") if isinstance(r, dict) else None
        bk, sr, rt = C("best_of_k"), C("self_refine"), C("retrieval")
        rc = rsi.get(sm)
        cand = [x for x in (bk, sr, rt) if isinstance(x, (int, float))]
        gain = round(rc - max(cand), 3) if (isinstance(rc, (int, float)) and cand) else None
        board[sm] = dict(model=sm, rsi_C=rc, best_of_k=bk, self_refine=sr, retrieval=rt, recursion_gain=gain)
    rows = sorted(board.values(), key=lambda m: (m.get("recursion_gain") is None, -(m.get("recursion_gain") or -9)))
    json.dump(dict(updated=__import__("datetime").date.today().isoformat(),
                   note="Recursion vs sampling: recursion_gain = weight-RSI C minus best non-recursive baseline (best-of-k / self-refine / retrieval) at matched budget. >0 = training on own kernels beats sampling more.",
                   models=rows), open(os.path.join(D, "baselines.json"), "w"), indent=2)
    print("baselines board:", len(rows), "models;", sum(1 for r in rows if r.get("recursion_gain") is not None), "with recursion_gain")

if __name__ == "__main__":
    build()
