"""S3 progress store + resume for KernelAscent, via boto3/IRSA (the aws CLI fails cross-account on the pods).

Runs ON a node. Progress is keyed by run-tag = the ka_data subdir name (sp_qwen14b_s0, rmech_*, spc_*), which is
stable across nodes, so a run relaunched on a fresh node restores its own dir and the runner resumes from the last
saved round. Uploads round JSONs, *_trace.jsonl, LoRA adapter checkpoints (ckpt/) and resume_state.json; skips the
per-kernel scratch (cand_modules/, compiled_cache/). Durable-write order for checkpoints: payload first, then a
per-tag manifest last, so a half-written checkpoint is never advertised.

  python s3ckpt.py push            # one upload pass of $HOME/ka/ka_data
  python s3ckpt.py restore         # download all tags into $HOME/ka/ka_data before launch
  python s3ckpt.py daemon [secs]   # push every `secs` (default 300)
"""
import os, sys, time, boto3

BUCKET = os.environ.get("KA_S3_BUCKET", "greenland-intern-artifacts-703671891219-us-east-2-an")
REGION = os.environ.get("KA_S3_REGION", "us-east-2")
PREFIX = os.environ.get("KA_S3_PREFIX", "kernelascent")
DATA = os.path.join(os.environ["HOME"], "ka", "ka_data")
SKIP = ("/cand_modules/", "/compiled_cache/")
_s3 = boto3.client("s3", region_name=REGION)


def _keep(p):
    return not any(s in p for s in SKIP)


def push():
    n = 0
    for root, _, files in os.walk(DATA):
        for f in files:
            fp = os.path.join(root, f)
            if not _keep(fp):
                continue
            key = "%s/%s" % (PREFIX, os.path.relpath(fp, DATA))
            try:
                _s3.upload_file(fp, BUCKET, key); n += 1
            except Exception as e:
                print("push fail %s: %s" % (key, e), flush=True)
    print("pushed %d objects -> s3://%s/%s/" % (n, BUCKET, PREFIX), flush=True)


def restore():
    os.makedirs(DATA, exist_ok=True); n = 0
    tok = None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=PREFIX + "/")
        if tok: kw["ContinuationToken"] = tok
        r = _s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            key = o["Key"]; rel = key[len(PREFIX) + 1:]
            if not rel or not _keep("/" + rel):
                continue
            dst = os.path.join(DATA, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                _s3.download_file(BUCKET, key, dst); n += 1
            except Exception as e:
                print("restore fail %s: %s" % (key, e), flush=True)
        if not r.get("IsTruncated"): break
        tok = r.get("NextContinuationToken")
    print("restored %d objects from s3://%s/%s/" % (n, BUCKET, PREFIX), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daemon"
    if cmd == "push": push()
    elif cmd == "restore": restore()
    elif cmd == "daemon":
        secs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        print("s3 daemon every %ds -> s3://%s/%s/" % (secs, BUCKET, PREFIX), flush=True)
        while True:
            try: push()
            except Exception as e: print("daemon push err: %s" % e, flush=True)
            time.sleep(secs)
    else:
        print("usage: s3ckpt.py push|restore|daemon [secs]"); sys.exit(2)
