"""Validate + dedupe Fable-proposed problems on the box (contamination firewall).

Keeps a proposed task only if it (a) builds a valid fp32 gold and runs, (b) is
non-trivial (>=3 forward ops), and (c) is structurally unique (seed-normalized hash),
deduped both within the proposed pool and against the synthetic generator's tasks.
Proposed tasks are labeled 'fable-proposed' and are NEVER placed in the synthetic
held-out split used for the leaderboard.
"""
import os, glob, json, argparse, hashlib, re, torch
import agent_bench as A
import gen_source_tasks as G


def norm_hash(src):
    s = re.sub(r"SEED\s*=\s*\d+", "SEED=0", src)
    s = re.sub(r"\s+", "", s)
    return hashlib.sha1(s.encode()).hexdigest()


def n_forward_ops(src):
    m = re.search(r"def forward\(self.*?\n(.*?)return", src, re.DOTALL)
    body = m.group(1) if m else ""
    return len([l for l in body.splitlines() if "=" in l and "self." not in l.split("=")[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooldir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ops", type=int, default=3)
    args = ap.parse_args()

    # hashes of the synthetic generator's tasks, to reject anything that duplicates them
    seen = set(norm_hash(t["source"]) for t in G.generate_systematic(n_fusion=64))
    kept, dropped = [], []
    for d in sorted(glob.glob(args.pooldir + "/*")):
        tp = os.path.join(d, "task.py")
        if not os.path.exists(tp):
            continue
        name = os.path.basename(d)
        src = open(tp).read()
        h = norm_hash(src)
        if h in seen:
            dropped.append((name, "duplicate")); continue
        if n_forward_ops(src) < args.min_ops:
            dropped.append((name, "trivial")); continue
        try:
            ref, x, gold, ref_err = A.build_ref(src)
            with torch.no_grad():
                out = ref(x)
            if not (torch.isfinite(out).all().item() and torch.isfinite(gold).all().item()):
                dropped.append((name, "nonfinite")); continue
        except Exception as e:
            dropped.append((name, "build-fail:" + repr(e)[:40])); continue
        seen.add(h)
        meta_p = os.path.join(d, "meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {"name": name}
        meta["valid"] = True
        meta["hash"] = h
        meta["shape"] = list(out.shape)
        json.dump(meta, open(meta_p, "w"), indent=2)
        kept.append(name)
    json.dump({"kept": kept, "dropped": dropped}, open(args.out, "w"), indent=2)
    print("proposed pool: kept=%d dropped=%d" % (len(kept), len(dropped)))
    for n, why in dropped[:20]:
        print("  drop %-40s %s" % (n, why))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
