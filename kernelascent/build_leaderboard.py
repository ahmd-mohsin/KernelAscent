"""Aggregate per-model grade summaries into Capability-leaderboard rows.

Reports BOTH walls separately: correctness_rate (can the model produce a valid/correct
kernel) and speed_rate/fast_1 (is it faster than the min(eager,torch.compile) roofline),
plus a per-tier (Easy/Medium/Hard/Ultra, or L1/L2/..) breakdown of both.
"""
import json, glob, math, os, re, sys

ORG = {"anthropic": "Anthropic", "openai": "OpenAI", "meta": "Meta", "qwen": "Qwen",
       "mistral": "Mistral", "amazon": "Amazon", "deepseek": "DeepSeek", "google": "Google",
       "nvidia": "NVIDIA", "moonshotai": "Moonshot", "minimax": "MiniMax", "writer": "Writer",
       "ai21": "AI21"}


def params(mid):
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", mid.lower())
    return (m.group(1).upper() + "B") if m else "n/a"


def _tier(t):
    return t.get("tier") or "untiered"


def _correct(t):
    return t.get("pass_at_k", 0) > 0


def _fast(t, p=1.0):
    return t.get("best_speedup_roofline", 0) > p


def row_for(mid, tasks):
    """Build one leaderboard row (both walls + per-tier) from a model's graded tasks."""
    n = len(tasks)
    if n == 0:
        return None
    rate = lambda pred: round(sum(1 for t in tasks if pred(t)) / n, 3)
    sps = [t["best_speedup_roofline"] for t in tasks if t.get("best_speedup_roofline", 0) > 0]
    gm = round(math.exp(sum(math.log(s) for s in sps) / len(sps)), 3) if sps else 0.0
    # per-tier breakdown of both walls
    by_tier = {}
    tiers = {}
    for t in tasks:
        tiers.setdefault(_tier(t), []).append(t)
    for tier, ts in tiers.items():
        m = len(ts)
        by_tier[tier] = dict(
            n=m,
            correctness_rate=round(sum(1 for x in ts if _correct(x)) / m, 3),
            fast_1=round(sum(1 for x in ts if _fast(x, 1.0)) / m, 3),
        )
    return dict(
        model=mid, org=ORG.get(mid.split(".")[0], mid.split(".")[0]),
        type="api", params=params(mid), role="test-taker", k=1, tasks=n,
        correctness_rate=rate(_correct),          # WALL 1: valid/correct kernel
        speed_rate=rate(lambda t: _fast(t, 1.0)),  # WALL 2: beats the roofline (== fast_1)
        pass_at_k=rate(_correct),
        fast_1=rate(lambda t: _fast(t, 1.0)),
        fast_1_5=rate(lambda t: _fast(t, 1.5)),
        fast_2=rate(lambda t: _fast(t, 2.0)),
        geomean_pass=gm,
        by_tier=by_tier,
        notes="held-out, single-shot; correctness_rate=valid-kernel wall, speed_rate=beats-roofline wall",
    )


def build(sumdir, models_json=None):
    slug2id = {}
    if models_json:
        for mid in json.load(open(models_json)):
            slug2id[re.sub(r"[^a-z0-9]+", "-", mid.lower()).strip("-")] = mid
    rows = []
    for f in sorted(glob.glob(sumdir + "/*.json")):
        slug = os.path.basename(f)[:-5]
        tasks = json.load(open(f)).get("tasks", [])
        r = row_for(slug2id.get(slug, slug), tasks)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: (r["speed_rate"], r["correctness_rate"]), reverse=True)
    return rows


if __name__ == "__main__":
    sumdir = sys.argv[1]
    models_json = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".json") and "leaderboard" not in sys.argv[2] else None
    rows = build(sumdir, models_json)
    out = {"updated": "2026-09-04",
           "metric_note": "Two walls: correctness_rate = fraction of tasks with a correct kernel (fp32-gold); speed_rate/fast_1 = fraction beating the min(eager,torch.compile) roofline. by_tier gives both per difficulty tier. Higher is better.",
           "models": rows}
    outpath = sys.argv[3] if len(sys.argv) > 3 else "leaderboard_rows.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print("models=%d" % len(rows))
    for r in rows[:8]:
        print("  %-40s correct=%.2f speed=%.2f gm=%.2f n=%d tiers=%s" %
              (r["model"][:40], r["correctness_rate"], r["speed_rate"], r["geomean_pass"], r["tasks"],
               {k: (v["correctness_rate"], v["fast_1"]) for k, v in r["by_tier"].items()}))
