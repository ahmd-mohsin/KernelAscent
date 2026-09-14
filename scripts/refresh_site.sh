#!/bin/bash
# Runs on the LAPTOP. Pulls every result JSON from S3 (via a node's boto3/IRSA, since the laptop can't read the
# cross-account bucket), flattens into results/raw/, aggregates -> docs/data/, commits + pushes. Idempotent; re-run
# on each progress check so completed runs flow to the website. Node reached via /tmp/ka/box_1051.sh.
set -o pipefail
REPO="/Users/cmohsinm/MATS/KernelAscent"; RAW="$REPO/results/raw"; BOX=/tmp/ka/box_1051.sh
mkdir -p "$RAW"
# 1) node: restore latest S3 -> ka_data, then tar every board-relevant result json as base64
$BOX 'cd $HOME/ka && python3 scripts/s3ckpt.py restore >/dev/null 2>&1; cd $HOME/ka/ka_data && tar cz \
  sp_*/selfplay_rsi.json spc_*/selfplay_closed.json rmech_*/rsi_mechanism.json \
  trackc_*/track_c.json comb_*/combined_rsi.json open_*/open_rsi.json baselines_*/baselines.json 2>/dev/null | base64' 2>/dev/null \
  | grep -vE "Warning|Permanently|Pseudo" > /tmp/ka_site.b64
[ -s /tmp/ka_site.b64 ] || { echo "no data pulled"; exit 1; }
rm -rf /tmp/ka_site && mkdir -p /tmp/ka_site
base64 -d -i /tmp/ka_site.b64 | tar xz -C /tmp/ka_site 2>/dev/null
# 2) flatten <tag>/<file>.json -> results/raw/<tag>.json  (names match aggregate_all.py globs)
for f in /tmp/ka_site/*/*.json; do
  [ -e "$f" ] || continue
  cp "$f" "$RAW/$(basename "$(dirname "$f")").json"
done
# deterministic cleanup: drop empty-history runs and all-zero closed runs (expired-creds/broken), keep working ones
n=$(python3 - "$RAW" <<'PY'
import json, glob, os, sys
raw=sys.argv[1]; kept=0
for f in glob.glob(os.path.join(raw,"*.json")):
    try: d=json.load(open(f))
    except Exception: continue
    h=d.get("history") or []
    base=os.path.basename(f)
    drop = (not h)
    if base.startswith("spc_") and h:      # closed: broken run = every S/F/L score is zero across all rounds
        drop = all(((r.get("Q_held_static") or 0)==0 and (r.get("Q_held_frozen_author") or 0)==0
                    and (r.get("Q_held_live") or 0)==0) for r in h)
    if drop: os.remove(f)
    else: kept+=1
print(kept)
PY
)
echo "results/raw now holds $n publishable runs"
# 3) aggregate + publish
cd "$REPO"
python3 scripts/aggregate_all.py
python3 scripts/mech_analysis.py 2>/dev/null | tail -4   # re-derive scale->mechanism->RSI findings each cycle
python3 scripts/build_baselines.py 2>/dev/null           # rebuild 03c recursion-gain board from baseline runs
git add results/raw docs/data 2>/dev/null
git commit -q -m "site: refresh boards from S3 ($(date -u +%H:%MZ)) — $n runs" 2>/dev/null \
  && git push -q 2>&1 | tail -1 && echo "pushed" || echo "no changes to commit"
