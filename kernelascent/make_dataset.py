"""Materialize a task split to disk for release (task.py + meta.json + manifest).

Usage:
  python make_dataset.py --split public  --outdir dataset/public
  python make_dataset.py --split heldout --outdir /private/heldout   # maintainers only, never commit
"""
import os, json, argparse
import gen_source_tasks as G
import splits as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(S.SPLITS), required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    cfg = S.SPLITS[args.split]
    tasks = G.generate_systematic(n_fusion=cfg["n_fusion"], seed0=cfg["seed0"])
    os.makedirs(args.outdir, exist_ok=True)
    fam, tier = {}, {}
    for t in tasks:
        d = os.path.join(args.outdir, t["name"])
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "task.py"), "w").write(t["source"])
        json.dump({k: t[k] for k in ("name", "tier", "family", "tags", "meta")},
                  open(os.path.join(d, "meta.json"), "w"), indent=2)
        fam[t["family"]] = fam.get(t["family"], 0) + 1
        tier[t["tier"]] = tier.get(t["tier"], 0) + 1
    json.dump({"split": args.split, "n": len(tasks), "families": fam, "tiers": tier,
               "seed_range": cfg,
               "tasks": [{"name": t["name"], "tier": t["tier"], "family": t["family"]} for t in tasks]},
              open(os.path.join(args.outdir, "manifest.json"), "w"), indent=2)
    print("%s: wrote %d tasks -> %s  families=%s tiers=%s" % (args.split, len(tasks), args.outdir, fam, tier))


if __name__ == "__main__":
    main()
