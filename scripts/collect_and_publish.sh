#!/bin/bash
# Collect all rerun_* result JSONs from the 9-node fleet -> results/raw/, aggregate -> docs/data/,
# commit+push git, upload HF. Idempotent; safe to run on a loop. Reads /tmp/ka helpers + node map.
set -o pipefail
REPO="/Users/cmohsinm/MATS/KernelAscent"; RAW="$REPO/results/raw"; KA=/tmp/ka
mkdir -p "$RAW" "$KA/collect_tmp"
SSHOPT='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=15 -p 2222'
# node map: port -> "mainIP w1 w2"
declare -A NODES=( [1051]="10.3.129.241 10.3.253.112 10.3.231.17" [1052]="10.3.171.65 10.3.90.181 10.3.193.178" [1053]="10.3.82.183 10.3.195.37 10.3.72.151" )

pull_node(){ # $1=port $2=ssh-prefix(local main = "" ; worker = ssh cmd)
  local tag="$1"; local runner="$2"
  # tar every result json under ka_data/rerun_*/ and emit base64
  local b64
  b64=$($runner 'cd $HOME/ka/ka_data 2>/dev/null && tar cz rerun_*/weight_rsi.json rerun_*/combined_rsi.json rerun_*/track_c.json 2>/dev/null | base64' 2>/dev/null | grep -v Warning)
  [ -z "$b64" ] && return
  local d="$KA/collect_tmp/$tag"; rm -rf "$d"; mkdir -p "$d"
  echo "$b64" | base64 -d 2>/dev/null | tar xz -C "$d" 2>/dev/null
  # flatten rerun_TAG/<file>.json -> results/raw/<TAG>.json  (weight_rsi->rerun_, combined->keep comb, trackc->keep)
  for wj in "$d"/rerun_*/weight_rsi.json; do [ -e "$wj" ] || continue; local t=$(basename $(dirname "$wj")); cp "$wj" "$RAW/$t.json"; done
  for cj in "$d"/rerun_*/combined_rsi.json; do [ -e "$cj" ] || continue; local t=$(basename $(dirname "$cj")); cp "$cj" "$RAW/$t.json"; done
  for tj in "$d"/rerun_*/track_c.json; do [ -e "$tj" ] || continue; local t=$(basename $(dirname "$tj")); cp "$tj" "$RAW/$t.json"; done
}

for port in 1051 1052 1053; do
  read -r main w1 w2 <<< "${NODES[$port]}"
  pull_node "n${port}m" "$KA/box_${port}.sh"
  for w in "$w1" "$w2"; do
    pull_node "n${port}_${w##*.}" "$KA/box_${port}.sh env SSH_ASKPASS_REQUIRE=force SSH_ASKPASS=/tmp/ap.sh DISPLAY=:0 ssh $SSHOPT greenland-user@$w"
  done
done

cd "$REPO"
python3 scripts/aggregate_all.py
git add results/raw docs/data 2>/dev/null
git commit -q -m "orchestrator: collect + aggregate definitive re-run $(date -u +%H:%MZ)" 2>/dev/null && git push -q 2>&1 | tail -1
if [ -f "$KA/hf_token" ]; then
  HF=$(cat "$KA/hf_token")
  python3 - "$HF" <<'PY' 2>/dev/null | tail -1
import sys; from huggingface_hub import HfApi
api=HfApi(token=sys.argv[1]); repo="muahmed7338/kernelascent-tasks"
for lf,rf in [("docs/data/tier_speed_rsi.json","leaderboards/per_scale_speed.json"),("docs/data/combined_rsi.json","leaderboards/combined_rsi.json"),("docs/data/trackc.json","leaderboards/procedure_rsi.json"),("docs/data/rigor.json","leaderboards/rigor_cross_seed.json")]:
    try:
        import os
        if os.path.exists(lf): api.upload_file(path_or_fileobj=lf,path_in_repo=rf,repo_id=repo,repo_type="dataset")
    except Exception as e: print("hf-err",str(e)[:50])
print("HF synced")
PY
fi
echo "$(date -u +%H:%MZ) collected $(ls $RAW/rerun_*.json 2>/dev/null|wc -l) rerun files"
