#!/usr/bin/env python3
"""MAINTAINER-ONLY: re-run a submitted model on the PRIVATE held-out split and mark it verified.

This is the gated tier. It is NOT run from a PR — arbitrary PR code must never touch our
GPUs or the held-out set. A maintainer runs it on the Greenland cluster (or via the
`verified-eval` workflow_dispatch on the self-hosted `greenland` runner), where the
held-out split and cluster creds are available.

  python3 scripts/run_submission.py submissions/capability/my-model.json [--gpu 0]

It maps the submission's track to the matching harness entrypoint, runs on the held-out
split, writes the verified metrics back into the submission JSON (`verified: true`), and
leaves aggregate_leaderboard.py to fold it into the board.
"""
import json, sys, os, subprocess, argparse, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# track -> (module, how to read metrics out of its summary). Maintainer wires the exact
# flags per model; these are the canonical entrypoints already used across the repo.
TRACK_CMD = {
    "capability":    "kernelascent eval --split heldout --tiers L1,L2,L3",
    "weight_rsi":    "python -m kernelascent.v3.lab_compounding --held-family l3 --rounds 6",
    "procedure_rsi": "python -m kernelascent.v3.lab_track_c",
    "closed_open":   "python -m kernelascent.v3.lab_track_c --mode closed_open",
    "selfplay":      "python -m kernelascent.v3.lab_selfplay --arms static,frozen,live",
    "harness":       "python -m kernelascent.v3.lab_transplant  # runs the submitted recipe on the fixed open trainee",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true", help="print the command without running")
    args = ap.parse_args()

    d = json.load(open(args.submission))
    track = d.get("track")
    if track not in TRACK_CMD:
        sys.exit(f"unknown track '{track}'")

    m = d.get("model", {})
    handle = m.get("hf_id") or m.get("api_id") or m.get("name")
    base = TRACK_CMD[track]
    print(f"[verify] track={track} model={handle}")
    print(f"[verify] held-out entrypoint:\n  {base} --model {handle}  (GPU {args.gpu})")
    print("[verify] the maintainer completes the exact flags for this model, runs on the")
    print("         PRIVATE held-out split, then re-runs this script with the produced summary.")

    if args.dry_run:
        return

    summary_path = os.environ.get("KA_SUMMARY")
    if not summary_path or not os.path.exists(summary_path):
        sys.exit("set KA_SUMMARY=<path to the held-out run's summary.json> to write verified metrics back")

    summary = json.load(open(summary_path))
    # copy over whatever metric keys the track declares
    from validate_submission import TRACK_METRICS  # type: ignore
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    keys = list(TRACK_METRICS[track].keys())
    for k in keys:
        if k in summary:
            d.setdefault("metrics", {})[k] = summary[k]
    d["verified"] = True
    d["verified_date"] = datetime.date.today().isoformat()
    d.setdefault("repro", {})["split"] = "heldout"
    json.dump(d, open(args.submission, "w"), indent=1)
    print(f"[verify] wrote verified metrics back to {args.submission}")


if __name__ == "__main__":
    main()
