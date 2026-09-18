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
  "deepseek.v3.2": "DeepSeek V3.2", "v3.2": "DeepSeek V3.2", "V3.2": "DeepSeek V3.2",
  "moonshotai.kimi-k2.5": "Kimi K2.5", "Kimi k2.5": "Kimi K2.5", "Kimi-K2.5": "Kimi K2.5",
  "mistral.mistral-large-3-675b-instruct": "Mistral Large 3", "Mistral Large 3 675B Instruct": "Mistral Large 3",
  "GPT-5.6 sol": "GPT-5.6 Sol", "GPT-5.6 terra": "GPT-5.6 Terra", "Nova-Pro": "Nova Pro",
}
def pretty(v):
    if not isinstance(v, str): return v
    key = v.strip()
    if key in EXACT: return EXACT[key]
    low = key.lower()
    if low in EXACT: return EXACT[low]
    PROV = ("us.", "openai.", "anthropic.", "mistral.", "deepseek.", "moonshotai.", "meta.", "qwen.", "writer.", "amazon.")
    stripped = key
    for p in ("us.",):
        if stripped.startswith(p): stripped = stripped[len(p):]
    if any(key.startswith(p) or ("."+key).find("."+p) >= 0 for p in PROV) and "." in stripped:
        tail = stripped.split(".", 1)[1] if "." in stripped else stripped
        # collapse family-repeated ids e.g. mistral.mistral-large-3-675b-instruct
        name = tail.replace("-", " ").replace("_", " ").replace(".", ".")
        name = re.sub(r"\b(\d+)b\b", r"\1B", name)              # 675b -> 675B
        FIX = {"gpt":"GPT","llm":"LLM","v3":"V3","k2":"K2","x5":"X5","oss":"OSS","instruct":"Instruct"}
        out = []
        for w in name.split():
            lw = w.lower()
            if lw in FIX: out.append(FIX[lw])
            elif any(c.isdigit() for c in w): out.append(w)      # keep 5.6, 3-675B, k2.5
            elif w.isupper(): out.append(w)
            else: out.append(w.capitalize())
        r = " ".join(out).strip()
        r = r.replace("Deepseek","DeepSeek").replace("Moonshotai","").replace("Mistral Mistral","Mistral").strip()
        return r or v
    # casing fixes for already-prettyish names
    return v.replace(" terra"," Terra").replace(" sol"," Sol")
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
