"""Package the Fable-curated, executable-validated task bank into a release dataset.

Reads the raw per-tier curated JSONs (dataset/_raw/*.json -- each either {tier: [tasks]} or a single
tier list), GLOBALLY DE-DUPLICATES (by slug and by whitespace-insensitive reference_code hash),
re-validates each task with the same executable checks the curator used, then splits every tier
DETERMINISTICALLY into:
  - public  (dev)  : committed to git + published to HuggingFace  -> dataset/public/<tier>.jsonl
  - heldout (test) : NEVER published (gitignored)                 -> dataset/heldout/<tier>.jsonl
The split is by sha1(name) so it is stable across re-runs and independent of generation order; a task
never moves between splits when the bank grows. Held-out fraction is fixed per tier.

Task schema (one JSON object per line):
  name, fn, spec, reference_code, buggy_code, sampler_code, edge_code, examples, tier, split, id
"""
import os, sys, json, glob, re, hashlib, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "kernelascent", "v3"))
import curate_tasks as CT   # reuse validate() + dedup key -> the SAME executable checks

HELDOUT_FRAC = 0.35         # ~35% of each tier is the private leaderboard test set
SPLIT_SALT = "kernelascent-v1"
FAMILY_CAP = 3              # max near-duplicate tasks per problem family per tier


def _iter_raw(raw_dir):
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        try:
            obj = json.load(open(path))
        except Exception as e:
            print("skip %s (%r)" % (path, e)); continue
        if isinstance(obj, dict):
            for tier, tasks in obj.items():
                for t in tasks or []:
                    t.setdefault("tier", tier); yield t
        elif isinstance(obj, list):
            for t in obj:
                yield t


def _split_of(name):
    h = int(hashlib.sha1((SPLIT_SALT + "|" + name).encode()).hexdigest(), 16) % 1000
    return "heldout" if h < int(HELDOUT_FRAC * 1000) else "public"


def _task_id(t):
    return hashlib.sha1((t["tier"] + "|" + CT._norm_code(t.get("reference_code", ""))).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "tasks", "_raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "tasks"))
    ap.add_argument("--no-revalidate", action="store_true")
    args = ap.parse_args()

    seen = set(); kept = {}   # split -> tier -> [tasks]
    counts = {"in": 0, "dup": 0, "invalid": 0, "kept": 0}
    for t in _iter_raw(args.raw):
        counts["in"] += 1
        tier = t.get("tier", "?")
        need = ("name", "fn", "spec", "reference_code", "buggy_code", "sampler_code", "edge_code", "examples")
        if not all(k in t for k in need):
            counts["invalid"] += 1; continue
        key = CT._dedup_key(t)
        if key in seen:
            counts["dup"] += 1; continue
        if not args.no_revalidate:
            ok, why = CT.validate(t, tier)
            if not ok:
                counts["invalid"] += 1; continue
        seen.add(key)
        split = _split_of(t["name"])
        rec = {k: t[k] for k in need}
        # examples is a nested, per-task-heterogeneous list ([args, output] with varying types) which
        # breaks HF/Arrow schema inference -> store it as a JSON string so every column is plain text.
        rec["examples"] = json.dumps(t["examples"])
        rec.update(tier=tier, split=split, id=_task_id({**t, "tier": tier}))
        kept.setdefault(split, {}).setdefault(tier, []).append(rec)
        counts["kept"] += 1

    # Per-family cap: exact-code dedup still lets one problem FAMILY dominate a tier (e.g. 6 "semver"
    # variants). Group by family (first name token) within a tier ACROSS splits, keep at most FAMILY_CAP
    # by id order, drop the rest -> a tier can't be swamped by near-duplicate problems.
    def _family(name):
        return re.split(r"[-_ ]", str(name).lower().strip())[0]
    for tier in ("easy", "medium", "hard", "ultra"):
        flat = [r for sp in kept.values() for r in sp.get(tier, [])]
        by_fam = {}
        for r in sorted(flat, key=lambda r: r["id"]):
            by_fam.setdefault(_family(r["name"]), []).append(r)
        keep_ids = set(); dropped = 0
        for fam, rs in by_fam.items():
            keep_ids.update(r["id"] for r in rs[:FAMILY_CAP]); dropped += max(0, len(rs) - FAMILY_CAP)
        counts["family_capped"] = counts.get("family_capped", 0) + dropped
        for sp in kept.values():
            if tier in sp:
                sp[tier] = [r for r in sp[tier] if r["id"] in keep_ids]

    stats = {"counts": counts, "by_split_tier": {}}
    for split in ("public", "heldout"):
        outdir = os.path.join(args.out, split); os.makedirs(outdir, exist_ok=True)
        for tier in ("easy", "medium", "hard", "ultra"):
            tasks = sorted(kept.get(split, {}).get(tier, []), key=lambda r: r["id"])
            if not tasks:
                continue
            with open(os.path.join(outdir, tier + ".jsonl"), "w") as f:
                for r in tasks:
                    f.write(json.dumps(r) + "\n")
            stats["by_split_tier"].setdefault(split, {})[tier] = len(tasks)
    json.dump(stats, open(os.path.join(args.out, "STATS.json"), "w"), indent=2)
    print("BUILD", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
