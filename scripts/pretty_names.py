#!/usr/bin/env python3
"""Normalize model IDs -> human display names across docs/data/*.json + dataset (leaderboards).
Turns 'us.openai.gpt-6-astra' -> 'GPT-6 Astra', 'us.anthropic.claude-sonnet-5' -> 'Claude Sonnet 5', etc."""
import json, glob, re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXACT = {
  "us.openai.gpt-6-astra": "GPT-6 Astra",
  "us.openai.gpt-5.6-terra": "GPT-5.6 Terra", "us.openai.gpt-5.6-sol": "GPT-5.6 Sol",
  "us.openai.gpt-5.6": "GPT-5.6", "gpt56terra": "GPT-5.6 Terra", "gpt56sol": "GPT-5.6 Sol",
  "us.anthropic.claude-sonnet-5": "Claude Sonnet 5", "us.anthropic.claude-opus-5": "Claude Opus 5",
  "us.anthropic.claude-fable-5-1": "Claude Fable 5.1", "sonnet5": "Claude Sonnet 5",
  "opus5": "Claude Opus 5", "astra": "GPT-6 Astra", "kimi": "Kimi K2", "dsv32": "DeepSeek V3.2",
  "mistral3": "Mistral Large 3", "us.deepseek.v3-2": "DeepSeek V3.2",
}
def pretty(v):
    if not isinstance(v, str): return v
    key = v.strip()
    if key in EXACT: return EXACT[key]
    low = key.lower()
    if low in EXACT: return EXACT[low]
    if key.startswith("us.") and key.count(".") >= 2:      # generic bedrock id -> title-case tail
        tail = key.split(".", 2)[2]
        prov = key.split(".")[1]
        name = tail.replace("-", " ").replace("_", " ")
        name = re.sub(r"\bgpt\b", "GPT", name, flags=re.I)
        name = re.sub(r"\bclaude\b", "Claude", name, flags=re.I)
        name = " ".join(w.capitalize() if not w.isupper() and not any(c.isdigit() for c in w) else w for w in name.split())
        return name.strip()
    return v
def walk(o):
    if isinstance(o, dict):
        return {k: (pretty(v) if k in ("model","id","name","researcher","trainee","author") else walk(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(x) for x in o]
    return o
n = 0
for f in glob.glob(os.path.join(ROOT, "docs", "data", "*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    nd = walk(d)
    if json.dumps(nd) != json.dumps(d):
        json.dump(nd, open(f, "w"), indent=1); n += 1
print("prettified model names in %d data files" % n)
