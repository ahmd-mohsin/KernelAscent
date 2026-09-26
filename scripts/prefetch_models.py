#!/usr/bin/env python3
"""Pre-download model weights onto the shared cache so training jobs do not spend GPU walltime on it.

Exists as a file rather than a `python -c` because the job manifest is line-based and silently
truncated the inline version to a bare `python -c`, which failed in three seconds with
"Argument expected for the -c option". jobman now refuses a multi-line command and says to do
this; this is that.

It also answers a question the failure left open: whether a compute node can reach the hub at
all. If it cannot, every cross-family cell would fail the same way one at a time, each after
waiting for its own allocation.

  python3 scripts/prefetch_models.py MODEL [MODEL ...]
"""
import os, sys, time

# Weights only, plus the tokenizer and config. No .bin when .safetensors exists, no cards.
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]


def main(argv):
    if not argv:
        print("usage: prefetch_models.py MODEL [MODEL ...]", file=sys.stderr)
        return 2
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("FATAL huggingface_hub not importable: %r" % e, file=sys.stderr)
        return 3

    print("HF_HOME=%s" % os.environ.get("HF_HOME", "<unset>"), flush=True)
    bad = 0
    for m in argv:
        t0 = time.time()
        try:
            p = snapshot_download(m, allow_patterns=PATTERNS)
            n = sum(len(fs) for _, _, fs in os.walk(p))
            sz = sum(os.path.getsize(os.path.join(d, f))
                     for d, _, fs in os.walk(p) for f in fs) / 1e9
            print("OK   %-46s %4.1f GB in %d files, %.0fs" % (m, sz, n, time.time() - t0), flush=True)
        except Exception as e:
            bad += 1
            # Print the type as well as the message: a network failure and a gated-repo failure
            # both read as "could not download" and need opposite responses.
            print("FAIL %-46s %s: %s" % (m, type(e).__name__, str(e)[:160]), flush=True)
    print("\n%d of %d fetched" % (len(argv) - bad, len(argv)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
