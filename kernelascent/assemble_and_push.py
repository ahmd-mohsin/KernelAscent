"""Assemble the public 'split + reference solutions' release and push to repo + HF.

Reads graded bundles (task dirs containing results.json / reference_solution.py),
keeps only the public-split task names (from dataset/public/manifest.json), copies the
release files, writes the dataset card, and uploads to a Hugging Face dataset repo.

HF token via env HF_TOKEN (never inlined). Usage:
  HF_TOKEN=$(cat /tmp/ka/hf_token) python assemble_and_push.py \
     --graded /path/to/cur_fable --public dataset/public --out dataset/public \
     --card hf_dataset_card.md --hf-repo muahmed7338/kernelascent [--no-hf]
"""
import os, json, glob, shutil, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", required=True, help="dir of graded task bundles")
    ap.add_argument("--public", required=True, help="public split dir (has manifest.json)")
    ap.add_argument("--out", required=True, help="release dir to (re)build in place")
    ap.add_argument("--card", required=True, help="dataset card markdown -> README.md")
    ap.add_argument("--hf-repo", default="muahmed7338/kernelascent")
    ap.add_argument("--no-hf", action="store_true", help="build only, skip HF upload")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.public, "manifest.json")))
    names = [t["name"] for t in manifest["tasks"]]
    keep = ["task.py", "meta.json", "reference_solution.py", "results.json"]
    n_ref = 0
    for name in names:
        src = os.path.join(args.graded, name)
        dst = os.path.join(args.out, name)
        os.makedirs(dst, exist_ok=True)
        for f in keep:
            p = os.path.join(src, f)
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join(dst, f))
        if os.path.exists(os.path.join(dst, "reference_solution.py")):
            n_ref += 1
    shutil.copyfile(args.card, os.path.join(args.out, "README.md"))
    # refresh manifest with reference-solution coverage
    manifest["with_reference_solution"] = n_ref
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print("release: %d tasks, %d with reference_solution -> %s" % (len(names), n_ref, args.out))

    if args.no_hf:
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=args.out, repo_id=args.hf_repo, repo_type="dataset",
                      commit_message="KernelAscent public split + reference solutions")
    print("pushed to https://huggingface.co/datasets/%s" % args.hf_repo)


if __name__ == "__main__":
    main()
