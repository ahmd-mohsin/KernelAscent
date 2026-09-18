#!/usr/bin/env python3
"""Validate KernelAscent leaderboard submissions (stdlib only, no pip install needed in CI).

Usage:
  python3 scripts/validate_submission.py submissions/capability/my-model.json   # one file
  python3 scripts/validate_submission.py --all                                  # every file under submissions/
  python3 scripts/validate_submission.py --changed a.json b.json                # a set (CI passes changed files)

Exits non-zero and prints a per-file report if anything fails. This runs in a PR
Action, so keep it dependency-free and the messages actionable.
"""
import json, sys, os, glob, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBDIR = os.path.join(ROOT, "submissions")

# per-track required metric keys + their numeric bounds (None = unbounded)
TRACK_METRICS = {
    "capability":    {"correct_rate": (0, 1), "fast_rate": (0, 1), "meanC": (0, 1), "trials": (1, None)},
    "weight_rsi":    {"rounds": (1, None), "C0": (0, 1), "C_final": (0, 1), "lineage_minus_reset": (-1, 1)},
    "procedure_rsi": {"rounds": (1, None), "Q0": (0, 1), "Q_final": (0, 1), "Q_gain_vs_frozen": (-1, 1)},
    "closed_open":   {"rounds": (1, None), "improved_minus_frozen": (-1, 1), "peak": (-1, 1)},
    "selfplay":      {"rounds": (1, None), "L_minus_F": (-1, 1), "F_minus_S": (-1, 1), "authored": (0, None)},
    "harness":       {"base_trainee": None, "improved_minus_frozen": (-1, 1)},  # base_trainee is a string id
}
KINDS = {"api", "open_weight"}
SPLITS = {"public", "heldout"}
# tracks where "gain" can be faked by pouring in search/compute -> require a metered budget
# manifest so reset/search-only controls can be run and gains ranked paired (Astra P0).
BUDGET_TRACKS = {"weight_rsi", "selfplay", "harness"}
BUDGET_FIELDS = ("starting_checkpoint", "total_budget", "candidate_count", "selection_rule")


def _err(errs, msg):
    errs.append(msg)


def validate_obj(d, path):
    errs = []
    # top-level required keys
    for k in ("schema_version", "track", "model", "submitter", "metrics", "repro"):
        if k not in d:
            _err(errs, f"missing top-level key '{k}'")
    if errs:
        return errs  # can't go further without the skeleton

    track = d["track"]
    if track not in TRACK_METRICS:
        _err(errs, f"track '{track}' not one of {sorted(TRACK_METRICS)}")
        return errs

    # model block
    m = d["model"]
    if not isinstance(m, dict):
        _err(errs, "'model' must be an object")
    else:
        if not m.get("name"):
            _err(errs, "model.name is required")
        if m.get("kind") not in KINDS:
            _err(errs, f"model.kind must be one of {sorted(KINDS)}")
        if m.get("kind") == "open_weight" and not m.get("hf_id") and track != "harness":
            _err(errs, "open_weight models must give model.hf_id so a maintainer can re-run on held-out")
        if m.get("kind") == "api" and track in ("weight_rsi",):
            _err(errs, "weight_rsi is open-weight only (an API model cannot expose weights to train)")

    # submitter
    s = d["submitter"]
    if not isinstance(s, dict) or not s.get("name"):
        _err(errs, "submitter.name is required")

    # metrics
    met = d["metrics"]
    spec = TRACK_METRICS[track]
    if not isinstance(met, dict):
        _err(errs, "'metrics' must be an object")
    else:
        for key, bound in spec.items():
            if key not in met:
                _err(errs, f"metrics.{key} is required for track '{track}'")
                continue
            v = met[key]
            if bound is None:
                if not isinstance(v, str) or not v.strip():
                    _err(errs, f"metrics.{key} must be a non-empty string")
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                _err(errs, f"metrics.{key} must be a number")
                continue
            lo, hi = bound
            if lo is not None and v < lo:
                _err(errs, f"metrics.{key}={v} below minimum {lo}")
            if hi is not None and v > hi:
                _err(errs, f"metrics.{key}={v} above maximum {hi}")

    # repro block — the point of the whole exercise: another person can re-run this
    r = d["repro"]
    if not isinstance(r, dict):
        _err(errs, "'repro' must be an object")
    else:
        if not r.get("harness_commit"):
            _err(errs, "repro.harness_commit is required (git sha of the KernelAscent harness you ran)")
        if r.get("split") not in SPLITS:
            _err(errs, f"repro.split must be one of {sorted(SPLITS)}")
        if "seeds" not in r:
            _err(errs, "repro.seeds is required (list or int)")
        if not r.get("command"):
            _err(errs, "repro.command is required (the exact command you ran)")

    # harness track must point at the recipe
    if track == "harness":
        h = d.get("self_improve_harness")
        if not isinstance(h, dict) or not h.get("repo") or not h.get("entrypoint"):
            _err(errs, "harness track requires self_improve_harness.{repo,entrypoint}")

    # anti-gaming budget manifest: a "gain" that just spent more search/compute is not RSI.
    # These fields let the evaluator run matched reset + search-only controls and rank paired gains.
    if track in BUDGET_TRACKS and isinstance(r, dict):
        budget = r.get("budget") if isinstance(r.get("budget"), dict) else r
        for bf in BUDGET_FIELDS:
            if bf not in budget:
                _err(errs, f"repro.budget.{bf} is required for '{track}' (meters training+search so gain can't be faked by compute; see submissions/README.md)")

    return errs


def validate_file(fp):
    try:
        d = json.load(open(fp))
    except Exception as e:
        return [f"not valid JSON: {e}"]
    return validate_obj(d, fp)


def main(argv):
    if "--all" in argv:
        files = sorted(glob.glob(os.path.join(SUBDIR, "*", "*.json")))
    elif "--changed" in argv:
        files = [a for a in argv[argv.index("--changed") + 1:] if a.endswith(".json")]
    else:
        files = [a for a in argv[1:] if a.endswith(".json")]
    files = [f for f in files if os.path.basename(f) != "schema.json"]
    if not files:
        print("no submission files to validate")
        return 0
    bad = 0
    for fp in files:
        rel = os.path.relpath(fp, ROOT)
        errs = validate_file(fp)
        if errs:
            bad += 1
            print(f"FAIL {rel}")
            for e in errs:
                print(f"   - {e}")
        else:
            print(f"PASS {rel}")
    if bad:
        print(f"\n{bad} submission(s) failed validation.")
        return 1
    print(f"\nall {len(files)} submission(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
