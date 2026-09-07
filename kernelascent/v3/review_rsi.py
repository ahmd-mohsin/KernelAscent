"""Send the full benchmark log to Fable 5.1 at MAX effort and ask how to make the RSI axis truly
measure recursive self-improvement (it currently reads ~0 for every model). Save + print the answer."""
import os, sys, re
sys.path.insert(0, "/tmp/instance_storage/kernelascent")
import curate_bedrock as CB

LOG = open("/tmp/instance_storage/BENCHMARK_LOG.md").read()
Q = """You are a top-tier reviewer of AI benchmarks and a recursive-self-improvement (RSI) measurement expert.
Below is the COMPLETE design + results log of KernelAscent, a benchmark meant to measure whether an agent's
own improvements causally enable further useful improvements (RSI) on GPU-kernel research.

THE PROBLEM I need your help with: the recursive-compounding axis (the causal producer contrasts F1 and F2,
and N) is coming out ~0 for EVERY model, including frontier (Fable, GPT-5.6). The capability axis is fine
(frontier clearly beats weaker models). I am worried the RSI INSTRUMENT itself is too weak/short/atomic to
ever surface recursive self-improvement even if it were present — i.e. that a 0 here is uninformative rather
than a real negative.

Give me a DETAILED, CONCRETE, actionable answer to: how do I make the RSI part TRULY measure recursive
self-improvement? Specifically address, with specifics I can implement:
1. Is F1/F2=0 more likely "instrument too weak" or "RSI genuinely absent at this scale"? How would I
   DISTINGUISH these two rigorously (positive controls, power, what a valid positive would look like)?
2. Episode/task design: is single-file kernel generation too atomic for a PROCESS-level improvement to pay
   off? What research-workflow episode would give a self-improvement room to compound? Be concrete.
3. The loop: how many links, what budget split (develop vs revise), how many lineages, what should the agent
   INHERIT and EXECUTE so a better improver can actually show up? How to make models edit their OWN improver
   (they currently edit only the solver)?
4. The measurement/estimator: is the actor-vs-actor common-target contrast (F) the right causal estimand? Are
   there better estimands for "using an improvement caused a further improvement"? How to handle coarse vs
   continuous scoring, selection, and the unit of analysis.
5. Model behavior: what makes a model actually improve its own improvement procedure, and how do I elicit it
   without hand-holding (which would be cheating)?
6. A prioritized, minimal set of experiments/changes that would either PRODUCE a real positive RSI signal or
   let me publish a rigorous, calibrated bounded-negative that reviewers will trust.

Be specific and technical. Prefer concrete parameters, estimators, and episode designs over generalities.

=== KERNELASCENT MASTER LOG ===
""" + LOG

cur = CB.Curator("us.anthropic.claude-fable-5-1", "us-east-1", "bedrock")
# max-effort blocking converse can exceed the default read timeout -> give it 1 hour.
import boto3
from botocore.config import Config as _Cfg
_sess = boto3.Session(profile_name="bedrock")
cur.rt = _sess.client("bedrock-runtime", region_name="us-east-1",
                      config=_Cfg(read_timeout=3600, connect_timeout=60, retries={"max_attempts": 1}))
cur.resolve(); cur.resolve_reasoning()
EFFORT = os.environ.get("EFFORT", "xhigh")
cur.reasoning = {"thinking": {"type": "adaptive"}, "output_config": {"effort": EFFORT}}
cur.resolved = (cur.resolved[0], int(os.environ.get("MAXTOK", "22000")))   # cap output so it returns promptly
print("resolved", cur.resolved, "effort=" + EFFORT, flush=True)
out = cur.generate(Q) or ""
ans = re.sub(r"<reasoning>.*?</reasoning>\s*", "", out, flags=re.S).strip()   # drop the reasoning wrapper
open("/tmp/instance_storage/ka_data/fable_rsi_review.txt", "w").write(ans)
print("ANSWER_CHARS", len(ans), flush=True)
print("=====FABLE-MAX RSI REVIEW=====", flush=True)
print(ans, flush=True)
