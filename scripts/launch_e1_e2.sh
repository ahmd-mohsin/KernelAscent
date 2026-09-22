#!/usr/bin/env bash
# Queue the Tier-1 experiments on Marlowe via `mrl`. Open-weight models only; no API key.
#
#   E1  COVERAGE-INJECTION POSITIVE CONTROL
#       Does lineage-minus-reset turn positive when the student is handed correct kernels for
#       tasks it cannot solve itself? This is what makes the compounding null falsifiable:
#       right now a true null and a broken harness produce identical output, and this project
#       has already been burned by exactly that (a wrong KA_ROOT graded every kernel False).
#
#   E2  T3 STRATEGY-CAP ABLATION
#       lab_track_c hardcoded a 12-strategy cap in both the improve prompt and the slice shown
#       back, so every healthy run filled all 12 slots in round 0 and stayed pinned. Model
#       plateau and harness ceiling are therefore confounded in the published T3 result. Run
#       the same protocol at caps {12, 48, unbounded} and see whether the plateau survives.
#
# Usage:
#   scripts/launch_e1_e2.sh teacher     # step 0: harvest injection material (run once, ~1-2 GPU-h)
#   scripts/launch_e1_e2.sh e1          # step 1: control vs inject, 3 scales x 3 seeds
#   scripts/launch_e1_e2.sh e2          # step 2: cap ablation
#   scripts/launch_e1_e2.sh --dry e1    # print the submissions without queueing
set -uo pipefail

MRL="${MRL:-mrl}"
command -v "$MRL" >/dev/null || { echo "mrl not on PATH: export PATH=\$HOME/.marlowe/bin:\$PATH"; exit 1; }

SCRATCH="/scratch/m000215-pm06/muahmed"
TEACHER_JSON="$SCRATCH/teacher_kernels_q14.json"
DATA="$SCRATCH/data"

DRY=""; [ "${1:-}" = "--dry" ] && { DRY="--dry-run"; shift; }
STEP="${1:-help}"

# Scales with published multi-seed compounding data, so E1 is a like-for-like comparison.
SMALL=("Qwen/Qwen2.5-Coder-0.5B-Instruct" "Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct")
TAGS=("q05" "q15" "q3")
SEEDS=(1 2 3)

# The teacher must be able to solve what the students cannot -- 14B has 86% coverage on this
# bank against 14-28% for the sub-2B students, which is the whole point of the injection.
TEACHER_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"

sub() { echo "  + $*"; [ -n "$DRY" ] && return 0; $MRL submit $DRY "$@"; }

case "$STEP" in

teacher)
  echo "STEP 0  harvesting verified kernels from $TEACHER_MODEL"
  echo "        one graded pass; every E1 inject arm reuses the result"
  sub -J ka-teacher -N 1 -G 2 -t 04:00:00 -- \
    "python scripts/make_teacher_kernels.py --model $TEACHER_MODEL --gpus 0,1 --k 8 --out $TEACHER_JSON"
  echo
  echo "  when it finishes, check coverage before running e1:"
  echo "    mrl run 'python -c \"import json;d=json.load(open(\\\"$TEACHER_JSON\\\"));print(len(d[\\\"kernels\\\"]),\\\"tasks solved\\\")\"'"
  ;;

e1)
  echo "STEP 1  E1 positive control: control vs inject, ${#SMALL[@]} scales x ${#SEEDS[@]} seeds"
  echo "        both arms train on the same round data; injection only ADDS kernels for tasks"
  echo "        the student failed, so lineage-minus-reset still isolates accumulation."
  [ -n "$DRY" ] || $MRL run "test -s $TEACHER_JSON" >/dev/null 2>&1 || {
    echo "  !! $TEACHER_JSON missing -- run 'scripts/launch_e1_e2.sh teacher' first"; exit 1; }
  for i in "${!SMALL[@]}"; do
    m="${SMALL[$i]}"; tag="${TAGS[$i]}"
    for s in "${SEEDS[@]}"; do
      # CONTROL: exactly the published protocol, re-run so both arms share code + bank
      sub -J "e1c-$tag-s$s" -N 1 -G 2 -t 08:00:00 -- \
        "python -m kernelascent.v3.lab_compounding --model $m --gpus 0,1 --rounds 6 --seed $s \
--outdir $DATA/e1_control_${tag}_s${s}"
      # INJECT: identical, plus teacher kernels on the student's uncovered tasks
      sub -J "e1i-$tag-s$s" -N 1 -G 2 -t 08:00:00 -- \
        "python -m kernelascent.v3.lab_compounding --model $m --gpus 0,1 --rounds 6 --seed $s \
--inject-kernels $TEACHER_JSON --inject-per-task 1 --outdir $DATA/e1_inject_${tag}_s${s}"
    done
  done
  echo
  echo "  read out with:  python3 scripts/equivalence_tost.py --dir <pulled dir>"
  echo "  DECISION RULE (set now, before the data lands):"
  echo "    inject arm lineage-reset > 0   -> harness has dynamic range; the null is a real"
  echo "                                     finding about coverage. Publish it."
  echo "    inject arm also flat          -> the loop cannot register compounding at all."
  echo "                                     Do NOT publish the null; fix the loop first."
  ;;

e2)
  echo "STEP 2  E2 T3 cap ablation on an open-weight model (no API key needed)"
  # 7B is the smallest model that reliably writes usable strategy text; below that the
  # improve step degenerates into unparseable output (cf. the dsv32/kimi parse failures).
  M="Qwen/Qwen2.5-Coder-7B-Instruct"
  for cap in 12 48 0; do
    label=$([ "$cap" = "0" ] && echo "unbounded" || echo "$cap")
    for s in 1 2; do
      sub -J "e2-cap$label-s$s" -N 1 -G 2 -t 10:00:00 -- \
        "python -m kernelascent.v3.lab_track_c --open-model $M --gpus 0,1 --grade-gpu 0 \
--max-strategies $cap --rounds 6 --seed $s --outdir $DATA/e2_cap${label}_s${s}"
    done
  done
  echo
  echo "  DECISION RULE:"
  echo "    plateau persists at every cap -> the one-shot claim is EARNED and strengthened"
  echo "    gain keeps climbing as cap rises -> the published 'one-shot' framing is a"
  echo "                                        measurement artifact and must be retracted"
  ;;

*)
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  ;;
esac
