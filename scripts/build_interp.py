#!/usr/bin/env python3
"""Build the interpretability board (docs/data/interp.json) from GPU interp-probe runs.
Each results/raw/interp_*.json = {model, n_correct, n_incorrect, correctness_rate, [per_layer_auc, best_layer,
best_layer_frac, best_auc, mean_auc, n_layers]}. The board answers: WHERE (which transformer depth) and HOW WELL
a model internally encodes GPU-kernel correctness, vs its generation success rate — and how that scales."""
import json, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw"); D = os.path.join(ROOT, "docs", "data")
SIZE = {"0.5":0.5,"1.3":1.3,"1.5":1.5,"1.7":1.7,"3b":3,"3B":3,"-3":3,"6.7":6.7,"7B":7,"7b":7,"8B":8,"8b":8,"9B":9,"14B":14,"15b":15,"15B":15,"32B":32}
def size_of(n):
    for k,v in SIZE.items():
        if k in n: return v
    return 2.0
def build():
    rows=[]
    for f in glob.glob(os.path.join(RAW,"interp_*.json")):
        try: d=json.load(open(f))
        except Exception: continue
        m=d.get("model",os.path.basename(f)[7:-5]); short=m.split("/")[-1]
        rows.append(dict(model=short, size_b=size_of(short), correctness_rate=d.get("correctness_rate"),
            n_labeled=(d.get("n_correct",0)+d.get("n_incorrect",0)),
            best_auc=d.get("best_auc"), best_layer=d.get("best_layer"), best_layer_frac=d.get("best_layer_frac"),
            n_layers=d.get("n_layers"), mean_auc=d.get("mean_auc"),
            probed=bool(d.get("best_auc") is not None),
            note=d.get("note")))
    rows.sort(key=lambda r:r["size_b"])
    json.dump(dict(updated=__import__("datetime").date.today().isoformat(),
        note="Interpretability: WHERE a model internally encodes kernel correctness. best_auc = linear separability of correct-vs-incorrect kernels at the best transformer layer (best_layer_frac = its relative depth); correctness_rate = how often the model actually GENERATES a correct kernel. High best_auc with low correctness_rate => the model represents correctness internally far better than it can decode it (generation-limited, not knowledge-limited). Not probed => correctness wall (no correct kernels to separate).",
        models=rows), open(os.path.join(D,"interp.json"),"w"), indent=2)
    print("interp board:", len(rows), "models;", sum(1 for r in rows if r["probed"]), "probed")
if __name__=="__main__":
    build()
