#!/usr/bin/env python3
"""Roll up submissions/*/*.json into docs/data/leaderboard_community.json.

Kept separate from the curated docs/data/leaderboard.json so community self-reports
never clobber the maintainer-verified official board. The site renders both, with a
`verified` badge on rows a maintainer has re-run on the held-out split.

Model display names are normalized through pretty_names (GPT-6 Astra, not us.openai...).
Stdlib only (runs in CI). Idempotent.
"""
import json, os, glob, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBDIR = os.path.join(ROOT, "submissions")
OUT = os.path.join(ROOT, "docs", "data", "leaderboard_community.json")

# reuse the canonical display-name map
try:
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from pretty_names import pretty  # type: ignore
except Exception:
    def pretty(v):
        return v

TRACK_ORDER = ["capability", "weight_rsi", "procedure_rsi", "closed_open", "selfplay", "harness"]
# metric used to sort each track (desc), and its label
SORT_KEY = {
    "capability": "meanC", "weight_rsi": "lineage_minus_reset", "procedure_rsi": "Q_gain_vs_frozen",
    "closed_open": "peak", "selfplay": "L_minus_F", "harness": "improved_minus_frozen",
}


def load_all():
    out = {t: [] for t in TRACK_ORDER}
    for fp in sorted(glob.glob(os.path.join(SUBDIR, "*", "*.json"))):
        if os.path.basename(fp) == "schema.json":
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        t = d.get("track")
        if t not in out:
            continue
        m = d.get("model", {})
        # honest three-state pipeline (Astra P1): schema-valid self-report -> maintainer
        # reproduced -> re-run on the private held-out split (verified). "verified" alone overpromises.
        verified = bool(d.get("verified", False))
        state = "verified" if verified else ("reproduced" if d.get("reproduced") else "self-reported")
        row = {
            "model": pretty(m.get("name", "?")),
            "kind": m.get("kind", "?"),
            "org": m.get("org", ""),
            "params_b": m.get("params_b"),
            "verified": verified,
            "state": state,
            "evaluator_version": d.get("evaluator_version", ""),
            "submitter": (d.get("submitter") or {}).get("name", ""),
            "date": d.get("date", ""),
            "source_file": os.path.relpath(fp, ROOT),
        }
        row.update(d.get("metrics", {}))
        if t == "harness":
            h = d.get("self_improve_harness", {})
            row["recipe"] = h.get("repo", "")
            row["entrypoint"] = h.get("entrypoint", "")
        out[t].append(row)
    # sort each track by its key (desc), verified first on ties
    for t, rows in out.items():
        k = SORT_KEY[t]
        rows.sort(key=lambda r: (r.get("verified", False), r.get(k, -9e9) if isinstance(r.get(k), (int, float)) else -9e9), reverse=True)
    return out


def main():
    tracks = load_all()
    n = sum(len(v) for v in tracks.values())
    doc = {
        "updated": datetime.date.today().isoformat(),
        "note": "Community submissions. Rows marked verified were re-run by a maintainer on the private held-out split; the rest are self-reported.",
        "sort_key": SORT_KEY,
        "tracks": tracks,
        "count": n,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1)
    print(f"wrote {os.path.relpath(OUT, ROOT)} — {n} community submission(s) across {len([t for t,v in tracks.items() if v])} track(s)")


if __name__ == "__main__":
    main()
