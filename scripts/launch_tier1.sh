#!/usr/bin/env bash
# Queue the Tier-1 cells through jobman (so timed-out chunks auto-continue).
#
# Replaces the shell one-liners that kept mangling job names: `set -- $spec` inside a loop
# silently lost fields, producing cells called "e1c--s1" and colliding two models onto one
# manifest entry. Arrays keep the model/tag pairing explicit.
#
#   scripts/launch_tier1.sh e1      control vs coverage-injection (needs the teacher JSON)
#   scripts/launch_tier1.sh e2      strategy-cap ablation
#   scripts/launch_tier1.sh --dry e1
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JM="$REPO/scripts/jobman.sh"
# Outputs live in $HOME: a few small JSON per run, and the shared /scratch group quota is at
# 2.3x its INODE limit, where os.makedirs() fails with EDQUOT intermittently. The HF cache
# stays on /scratch (far too big for the 32GB home).
OUT="/users/muahmed/ka_data"
TEACHER="/scratch/m000215-pm06/muahmed/teacher_kernels_q14.json"
WALL="02:00:00"       # short chunks: walltime drives the backfill estimate here, and both
                      # labs checkpoint per round so a timeout resumes rather than restarts

DRY=""; [ "${1:-}" = "--dry" ] && { DRY=1; shift; }
STEP="${1:-help}"

go() {  # go <name> <command...>
  local name="$1"; shift
  if [ -n "$DRY" ]; then printf '  %-22s %s\n' "$name" "$*"; return 0; fi
  "$JM" run "$name" "$WALL" 1 -- "$*" 2>&1 | tail -1
}

case "$STEP" in

e1)
  # Scales with published multi-seed compounding data, so the control arm is like-for-like.
  MODELS=("Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct")
  TAGS=("q15" "q3")
  if [ -z "$DRY" ] && ! mrl run "test -s $TEACHER" >/dev/null 2>&1; then
    echo "!! teacher kernels missing at $TEACHER -- run the harvest first"; exit 1
  fi
  echo "E1  coverage-injection positive control"
  echo "    control = published protocol; inject = same + teacher kernels on tasks the"
  echo "    student FAILED that round. Both arms train on the same round data, so"
  echo "    lineage-minus-reset still isolates accumulation."
  for i in "${!MODELS[@]}"; do
    m="${MODELS[$i]}"; t="${TAGS[$i]}"
    for s in 1 2; do
      go "e1c-$t-s$s" "python -m kernelascent.v3.lab_compounding --model $m --gpus 0 \
--rounds 6 --seed $s --outdir $OUT/e1_control_${t}_s${s}"
      go "e1i-$t-s$s" "python -m kernelascent.v3.lab_compounding --model $m --gpus 0 \
--rounds 6 --seed $s --inject-kernels $TEACHER --inject-per-task 1 \
--outdir $OUT/e1_inject_${t}_s${s}"
    done
  done
  echo
  echo "  DECISION RULE (fixed before the data lands):"
  echo "    inject lineage-reset > 0  -> the harness CAN register compounding, so the"
  echo "                                 unaugmented null is a real finding about coverage"
  echo "    inject also flat          -> the loop cannot register it at all; do not publish"
  ;;

e2)
  M="Qwen/Qwen2.5-Coder-7B-Instruct"   # smallest model that reliably writes parseable strategies
  echo "E2  strategy-cap ablation (caps 12 / 48 / unbounded)"
  for cap in 12 48 0; do
    lbl=$([ "$cap" = "0" ] && echo unbounded || echo "$cap")
    for s in 1 2; do
      go "e2-c$lbl-s$s-p1" "python -m kernelascent.v3.lab_track_c --open-model $M --gpus 0 \
--grade-gpu 0 --max-strategies $cap --rounds 6 --seed $s --outdir $OUT/e2_cap${lbl}_s${s}"
    done
  done
  ;;

*) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
