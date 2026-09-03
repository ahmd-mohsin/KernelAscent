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
    LADDER = (64000, 32000, 16000, 8192, 4096)   # descend from high to find each model's real cap

    def __init__(self, model_id, region, profile):
        from botocore.config import Config
        sess = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.rt = sess.client("bedrock-runtime", region_name=region,
                              config=Config(read_timeout=600, connect_timeout=15,
                                            retries={"max_attempts": 2}))
        self.model_id = model_id
        self.resolved = None       # (working_id, max_tokens) discovered per model
        self.reasoning = "__unset__"  # additionalModelRequestFields for reasoning (dict) or None

    def resolve(self):
        """Find the working id form (bare vs us.*) and the model's actual max output tokens."""
        ids = [self.model_id]
        if not self.model_id.startswith(("us.", "global.", "eu.", "apac.")):
            ids.append("us." + self.model_id)
        probe = [{"role": "user", "content": [{"text": "Reply with: ok"}]}]
        for f in ids:
            bad_form = False
            for mt in self.LADDER:
                try:
                    self.rt.converse(modelId=f, messages=probe,
                                     inferenceConfig={"maxTokens": mt, "temperature": 1.0})
                    self.resolved = (f, mt); return self.resolved
                except Exception as e:
                    esl = repr(e).lower()
                    if "throttl" in esl:
                        time.sleep(3)
                        try:
                            self.rt.converse(modelId=f, messages=probe,
                                             inferenceConfig={"maxTokens": mt, "temperature": 1.0})
                            self.resolved = (f, mt); return self.resolved
                        except Exception:
                            pass
                    if ("inference profile" in esl) or ("on-demand" in esl) or ("isn't supported" in esl):
                        bad_form = True; break        # try next id form
                    if ("maximum tokens" in esl) or ("token" in esl and "valid" in esl):
                        continue                      # cap too high -> lower mt
                    break                             # other error -> next form
            if bad_form:
                continue
        self.resolved = (self.model_id, 4096)         # last resort
        return self.resolved

    # candidate reasoning configs across providers (first accepted / reasoning-emitting wins)
    REASON_CONFIGS = [
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},   # Claude 5 era
        {"thinking": {"type": "enabled", "budget_tokens": 8000}},                  # Claude 4 / others
        {"reasoning_effort": "high"},                                              # OpenAI-style
        {"reasoning": {"effort": "high"}},
        {"enable_thinking": True},                                                 # Qwen-style
    ]

    def resolve_reasoning(self):
        """Discover a working reasoning config for this model (or None if unsupported/inherent)."""
        mid, mt = self.resolved
        probe = [{"role": "user", "content": [{"text": "What is 17*23? Think, then answer."}]}]
        best = None
        for cfg in self.REASON_CONFIGS:
            try:
                r = self.rt.converse(modelId=mid, messages=probe,
                                     inferenceConfig={"maxTokens": min(mt, 6000), "temperature": 1.0},
                                     additionalModelRequestFields=cfg)
                if any("reasoningContent" in b for b in r["output"]["message"]["content"]):
                    self.reasoning = cfg; return cfg      # emits explicit reasoning -> best
                if best is None:
                    best = cfg                            # accepted but no explicit block
            except Exception:
                continue
        self.reasoning = best
        return self.reasoning

    @staticmethod
    def _collect(content):
        """Return reasoning (if any) + answer text, so trajectories capture the thinking."""
        reason, ans = [], []
        for p in content:
            if "text" in p:
                ans.append(p["text"])
            rc = p.get("reasoningContent")
            if rc:
                t = rc.get("reasoningText", {}).get("text")
                if t:
                    reason.append(t)
        out = ("<reasoning>\n" + "\n".join(reason) + "\n</reasoning>\n\n") if reason else ""
        return out + "".join(ans)

    def generate(self, user_text, temp=1.0):
        """Generate for an arbitrary prompt (used by scaffold-RSI with a strategy library)."""
        if self.resolved is None:
            self.resolve()
        if self.reasoning == "__unset__":
            self.resolve_reasoning()
        mid, mt = self.resolved
        reason = self.reasoning if self.reasoning not in (None, "__unset__") else None
        for attempt in range(6):
            try:
                cfg = {"maxTokens": mt, "temperature": 1.0 if reason else temp}
                kw = {"additionalModelRequestFields": reason} if reason else {}
                r = self.rt.converse(modelId=mid, system=[{"text": SYS}],
                                     messages=[{"role": "user", "content": [{"text": user_text}]}],
                                     inferenceConfig=cfg, **kw)
                return self._collect(r["output"]["message"]["content"])
            except Exception as e:
                esl = repr(e).lower()
                if "throttl" in esl and attempt < 5:
                    time.sleep(min(2 ** attempt, 30)); continue
                if reason and any(k in esl for k in ("thinking", "reasoning", "budget", "temperature", "effort", "output_config", "enable_thinking")):
                    reason = None; continue
                return "BEDROCK_ERROR: " + repr(e)[:140]

    def one(self, src, temp, max_tokens=None):
        if self.resolved is None:
            self.resolve()
        if self.reasoning == "__unset__":
            self.resolve_reasoning()
        mid, mt = self.resolved
        msgs = [{"role": "user", "content": [{"text": PROMPT.format(src=src)}]}]
        reason = self.reasoning if self.reasoning not in (None, "__unset__") else None
        for attempt in range(6):
            try:
                cfg = {"maxTokens": mt, "temperature": 1.0 if reason else temp}
                kw = {"additionalModelRequestFields": reason} if reason else {}
                r = self.rt.converse(modelId=mid, system=[{"text": SYS}], messages=msgs,
                                     inferenceConfig=cfg, **kw)
                return self._collect(r["output"]["message"]["content"])
            except Exception as e:
                esl = repr(e).lower()
                if "throttl" in esl and attempt < 5:
                    time.sleep(min(2 ** attempt, 30)); continue
                if reason and any(k in esl for k in ("thinking", "reasoning", "budget", "temperature", "effort", "output_config", "enable_thinking")):
                    reason = None; continue      # model rejects this reasoning cfg -> retry plain
                return "BEDROCK_ERROR: " + repr(e)[:140]

    def _dead_old_one(self, src, temp, max_tokens=32000):
        # Try the given id and, if on-demand isn't supported, the us.* inference profile.
        ids = [self.model_id]
        if not self.model_id.startswith(("us.", "global.", "eu.", "apac.")):
            ids.append("us." + self.model_id)
        last = "BEDROCK_ERROR: no_attempt"
        for mid in ids:
            # Adaptive maxTokens: many models cap converse output below 32k.
            for mt in (max_tokens, 16000, 8192, 4096):
                for attempt in range(6):
                    try:
                        r = self.rt.converse(
                            modelId=mid, system=[{"text": SYS}],
                            messages=[{"role": "user", "content": [{"text": PROMPT.format(src=src)}]}],
                            inferenceConfig={"maxTokens": mt, "temperature": temp})
                        return "".join(p.get("text", "") for p in r["output"]["message"]["content"])
                    except Exception as e:
                        es = repr(e); esl = es.lower()
                        last = "BEDROCK_ERROR: " + es[:140]
                        if "throttl" in esl and attempt < 5:
                            time.sleep(min(2 ** attempt, 30)); continue
                        if ("maximum tokens" in esl) or ("validationexception" in esl and "token" in esl):
                            break                       # lower maxTokens
                        if ("inference profile" in esl) or ("on-demand" in esl) or ("isn't supported" in esl):
                            break                       # try next id (us.*)
                        return last                     # other error: give up
                else:
                    continue
                # broke out of attempt loop on a token error -> try next mt
                if ("maximum tokens" in last.lower()) or ("token" in last.lower() and "validation" in last.lower()):
                    continue
                break                                   # profile error -> break mt loop, try next id
        return last


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
    ap.add_argument("--families", default="", help="comma-separated families to include (default: all)")
    ap.add_argument("--no-raw", action="store_true", help="do not store raw reasoning trajectories")
    ap.add_argument("--seed0", type=int, default=0, help="split seed base (0=public, 10000000=held-out)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tasks = G.generate_systematic(n_fusion=args.n_fusion, seed0=args.seed0)
    if args.families:
        fams = set(f.strip() for f in args.families.split(","))
        tasks = [t for t in tasks if t["family"] in fams]
    if args.limit:
        tasks = tasks[:args.limit]
    cur = Curator(args.model_id, args.region, args.profile)
    wid, wmt = cur.resolve()
    rc = cur.resolve_reasoning()
    print("curator=%s -> id=%s maxTokens=%d reasoning=%s tasks=%d k=%d -> %s" %
          (args.model_id, wid, wmt, (list(rc.keys()) if rc else None), len(tasks), args.k, args.outdir))
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
            if t and not args.no_raw:
                # full reasoning trajectory (reasoning + code), for analysis / a richer dataset
                with open(os.path.join(tdir, "raw_%d.txt" % j), "w") as f:
                    f.write(t)
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
