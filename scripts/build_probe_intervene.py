#!/usr/bin/env python3
"""Build the probe-as-intervention board (docs/data/probe_intervene.json) from GPU runs.
Each results/raw/probe_intervene_*.json = {model,K,auc,layer_frac,random_correct_rate,probe_top1_correct_rate,
oracle_bestofK_rate,lift,probe_pick_speedup,k_sweep[]}. Shows self-verification (probe-guided selection) lifting
correct-kernel yield above the natural generation rate — the causal counterpart to the interp AUC result."""
import json, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw"); D = os.path.join(ROOT, "docs", "data")
SIZE = {"0.5":0.5,"1.3":1.3,"1.5":1.5,"1.7":1.7,"3b":3,"3B":3,"-3":3,"6.7":6.7,"7B":7,"7b":7,"8B":8,"8b":8,"9B":9,"14B":14,"15b":15,"15B":15,"32B":32}
def size_of(n):
    for k, v in SIZE.items():
        if k in n: return v
    return 2.0
def build():
    rows = []
    for f in glob.glob(os.path.join(RAW, "probe_intervene_*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        m = d.get("model", os.path.basename(f)[16:-5]); short = m.split("/")[-1]
        rows.append(dict(model=short, size_b=size_of(short), K=d.get("K"), auc=d.get("auc"),
            layer_frac=d.get("layer_frac"), random_rate=d.get("random_correct_rate"),
            probe_rate=d.get("probe_top1_correct_rate"), oracle_rate=d.get("oracle_bestofK_rate"),
            lift=d.get("lift"), probe_pick_speedup=d.get("probe_pick_speedup"), k_sweep=d.get("k_sweep")))
    rows.sort(key=lambda r: r["size_b"])
    json.dump(dict(updated=__import__("datetime").date.today().isoformat(),
        note="Probe-as-intervention (self-verification): at equal budget K, RANDOM=natural single-draw success, PROBE=pick highest correctness-probe score, ORACLE=any-correct-in-K. lift=PROBE-RANDOM. Probe lifting yield above random (esp. lifting sub-2B off the correctness wall) = the wall is partly a DECODE artifact and self-verification is a missing RSI primitive (causal counterpart to the interp AUC finding).",
        models=rows), open(os.path.join(D, "probe_intervene.json"), "w"), indent=2)
    print("probe_intervene board:", len(rows), "models")
if __name__ == "__main__":
    build()
