"""Publish the PUBLIC split to a HuggingFace dataset repo. The held-out (test) split is never uploaded.

Token: read from $HF_TOKEN or /tmp/ka/hf_token (write-scoped). Repo defaults to $HF_REPO.
Uploads dataset/public/*.jsonl + README.md (dataset card). Refuses to run if a heldout/ file would
be included (belt-and-suspenders against leaking the leaderboard test set).
"""
import os, sys, glob, argparse
from huggingface_hub import HfApi, create_repo

HERE = os.path.dirname(os.path.abspath(__file__))


def _token():
    t = os.environ.get("HF_TOKEN")
    if not t and os.path.exists("/tmp/ka/hf_token"):
        t = open("/tmp/ka/hf_token").read().strip()
    if not t:
        sys.exit("no HF token: set $HF_TOKEN or write /tmp/ka/hf_token")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_REPO", ""))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.repo:
        sys.exit("set --repo or $HF_REPO (e.g. ahmd-mohsin/kernelascent-tasks)")
    base = os.path.join(HERE, "tasks")
    pub = os.path.join(base, "public")
    files = sorted(glob.glob(os.path.join(pub, "*.jsonl")))
    assert files, "no public jsonl -- run build_dataset.py first"
    # heldout stays local; we NEVER add it to the upload list (leaderboard test set must not leak)
    upload = [(f, "data/" + os.path.basename(f)) for f in files]
    card = os.path.join(base, "README.md")
    if os.path.exists(card):
        upload.append((card, "README.md"))
    print("REPO", args.repo, "-> upload:")
    for src, dst in upload:
        print("  %-40s -> %s" % (os.path.basename(src), dst))
    if args.dry_run:
        print("DRY RUN -- nothing uploaded"); return
    tok = _token()
    create_repo(args.repo, repo_type="dataset", token=tok, exist_ok=True, private=False)
    api = HfApi(token=tok)
    for src, dst in upload:
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=args.repo, repo_type="dataset")
        print("  uploaded", dst)
    print("DONE https://huggingface.co/datasets/" + args.repo)


if __name__ == "__main__":
    main()
