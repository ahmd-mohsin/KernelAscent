#!/usr/bin/env bash
# S3 progress store + resume for KernelAscent. Set KA_S3=s3://<bucket>/<prefix> (e.g. s3://my-bucket/kernelascent).
# Runs ON a node. Two modes:
#   restore : download prior ka_data (result JSONs + resume checkpoints) from S3 into $HOME/ka/ka_data BEFORE launch,
#             so the runners' resume-from-round logic picks up where the dead node left off.
#   daemon  : every 300s, upload $HOME/ka/ka_data/*/ (round JSONs, traces, adapter checkpoints, resume_state) to S3.
# Node identity keeps runs from colliding: we key by TAG dir (sp_<model>_s<seed> etc), which is stable across nodes,
# so a run relaunched on a new node restores its own dir and continues.
set -u
: "${KA_S3:?set KA_S3=s3://bucket/prefix}"
DATA="$HOME/ka/ka_data"
PROF="${KA_S3_PROFILE:-}"; P=(); [ -n "$PROF" ] && P=(--profile "$PROF")
REG="${KA_S3_REGION:-us-east-2}"

restore(){ mkdir -p "$DATA"; aws s3 sync "$KA_S3/" "$DATA/" --region "$REG" "${P[@]}" 2>&1 | tail -3; echo "RESTORED from $KA_S3"; }
push(){ aws s3 sync "$DATA/" "$KA_S3/" --region "$REG" "${P[@]}" \
          --exclude '*/cand_modules/*' --exclude '*/compiled_cache/*' 2>&1 | tail -2; }   # skip per-kernel scratch

case "${1:-daemon}" in
  restore) restore ;;
  once)    push; echo "PUSHED once to $KA_S3" ;;
  daemon)  echo "S3 sync daemon -> $KA_S3 every 300s"; while true; do push; sleep 300; done ;;
  *) echo "usage: s3_sync.sh restore|once|daemon" ; exit 2 ;;
esac
