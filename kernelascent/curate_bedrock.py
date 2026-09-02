"""Decoupled curation driver (runs on the Mac).

Generates the systematic task set, calls a Bedrock curator model (Fable 5 / Opus 4.8)
to produce k candidate ModelNew per task, and writes candidate .py files to an output
directory. Checkpointed: a task with a DONE marker is skipped, so the multi-hour run
resumes cleanly. No torch needed here; grading happens on the GPU box afterward.

Auth: AWS_PROFILE=bedrock (never inline secrets). Region us-east-1.
"""
import os, re, json, time, argparse
from concurrent.futures import ThreadPoolExecutor
import boto3
import gen_source_tasks as G

SYS = ("You are an elite GPU performance engineer. You optimize PyTorch modules to run "
       "faster on an NVIDIA A100, keeping outputs numerically equivalent.")
PROMPT = """Optimize the following PyTorch module for speed on an A100 GPU.

Rules:
- Keep __init__ EXACTLY the same (same parameters, shapes, and initialization order). Only rewrite forward.
- You may use Triton (import triton, triton.language as tl) or fused PyTorch ops. Prefer custom Triton kernels for reductions, attention, and elementwise fusion.
- Produce numerically equivalent outputs.
- Output ONE complete class named ModelNew with the same interface, inside a single ```python code block. No prose.

Reference module:
```python
{src}
```"""


def extract_modelnew(text):
    for b in re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL):
        if "class ModelNew" in b:
            return b
    if "class ModelNew" in text:
        return text[text.index("class ModelNew"):]
    return None


class Curator:
    def __init__(self, model_id, region, profile):
        sess = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.rt = sess.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def one(self, src, temp, max_tokens=32000):
        for attempt in range(7):
            try:
                r = self.rt.converse(
                    modelId=self.model_id, system=[{"text": SYS}],
                    messages=[{"role": "user", "content": [{"text": PROMPT.format(src=src)}]}],
                    inferenceConfig={"maxTokens": max_tokens, "temperature": temp})
                return "".join(p.get("text", "") for p in r["output"]["message"]["content"])
            except Exception as e:
                if "Throttl" in repr(e) and attempt < 6:
                    time.sleep(min(2 ** attempt, 30)); continue
                return "BEDROCK_ERROR: " + repr(e)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=os.environ.get("BEDROCK_PROFILE", "bedrock"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n-fusion", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4, help="concurrent Bedrock calls")
    ap.add_argument("--limit", type=int, default=0, help="cap total tasks (0 = all); for smoke tests")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tasks = G.generate_systematic(n_fusion=args.n_fusion)
    if args.limit:
        tasks = tasks[:args.limit]
    cur = Curator(args.model_id, args.region, args.profile)
    print("curator=%s tasks=%d k=%d -> %s" % (args.model_id, len(tasks), args.k, args.outdir))
    done_ct = 0
    for ti, task in enumerate(tasks):
        tdir = os.path.join(args.outdir, task["name"])
        if os.path.exists(os.path.join(tdir, "DONE")):
            done_ct += 1
            continue
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "task.py"), "w") as f:
            f.write(task["source"])
        with open(os.path.join(tdir, "meta.json"), "w") as f:
            json.dump({k: task[k] for k in ("name", "tier", "family", "tags", "meta")}, f, indent=2)
        with ThreadPoolExecutor(max_workers=min(args.k, args.workers)) as ex:
            texts = list(ex.map(lambda _: cur.one(task["source"], args.temp), range(args.k)))
        n_ok = 0
        for j, t in enumerate(texts):
            code = extract_modelnew(t or "")
            if code:
                with open(os.path.join(tdir, "cand_%d.py" % j), "w") as f:
                    f.write(code)
                n_ok += 1
        open(os.path.join(tdir, "DONE"), "w").write(str(n_ok))
        done_ct += 1
        if ti % 5 == 0 or n_ok == 0:
            print("[%d/%d] %-22s fam=%-10s cands=%d/%d" %
                  (done_ct, len(tasks), task["name"], task["family"], n_ok, args.k), flush=True)
    print("curation complete: %d tasks in %s" % (done_ct, args.outdir))


if __name__ == "__main__":
    main()
