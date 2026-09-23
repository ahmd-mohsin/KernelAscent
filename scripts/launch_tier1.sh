#!/usr/bin/env bash
# Queue the Tier-1 cells through jobman (so timed-out chunks auto-continue).
#
# Replaces the shell one-liners that kept mangling job names: `set -- $spec` inside a loop
# silently lost fields, producing cells called "e1c--s1" and colliding two models onto one
# manifest entry. Arrays keep the model/tag pairing explicit.
#
#   scripts/launch_tier1.sh e1      control vs coverage-injection (needs the teacher JSON)
#   scripts/launch_tier1.sh e2      strategy-cap ablation
#   scripts/launch_tier1.sh r1      route 1: calibrate the 456-task DSL bank (resumable)
#   scripts/launch_tier1.sh r2      route 2: can a 32B model beat the baseline?
#   scripts/launch_tier1.sh r3      route 3: E1 re-read under KA_SCORE=passrate
#   scripts/launch_tier1.sh r4      route 4: prompt A/B -- is the plateau prompt-induced?
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

go() {  # go <name> <command...>   ; GPUS=n and WALL=hh:mm:ss override per call
  local name="$1"; shift
  if [ -n "$DRY" ]; then printf '  %-22s (%sg %s) %s\n' "$name" "${GPUS:-1}" "$WALL" "$*"; return 0; fi
  "$JM" run "$name" "$WALL" "${GPUS:-1}" -- "$*" 2>&1 | tail -1
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

r1)
  # ROUTE 1 -- a substrate with a genuine middle band.
  # The 29-task bank is bimodal on H100: 23 tasks score ~0.50 (correct, never faster) and the
  # 4 L3 tasks score 0.00 (unreachable). Nothing in between, so no contrast can move. The
  # 456-task DSL bank is the only candidate with enough tasks to contain a middle band at all.
  # difficulty_filter now RESUMES from its report, so this runs as 2h chunks via `jobman continue`
  # until it reaches the end -- the previous single-shot attempt timed out and lost its work.
  echo "R1  calibrate the 456-task DSL bank against a 3B H100 anchor (resumable chunks)"
  go "ka-calib-dsl" "python -m kernelascent.v3.difficulty_filter \
--model Qwen/Qwen2.5-Coder-3B-Instruct --gpus 0 \
--in dataset/kernel_bank/kernel_tasks_dsl_validated.json \
--out $OUT/bank_dsl_h100_filtered.json --report $OUT/bank_dsl_h100_report.json \
--k 4 --keep-hi 0.75 --min-ceiling 1.3"
  echo "    READ-OUT: the report's best_score histogram. A usable substrate needs mass"
  echo "    strictly BETWEEN 0.5 and 1.0. Another spike at 0.50 means the ceiling is the"
  echo "    hardware baseline, not the bank, and no task curation will fix it."
  ;;

r2)
  # ROUTE 2 -- is there ANY model that beats the torch.compile baseline on this hardware?
  # Best speedups across the bank: 1.5B/3B ~1.00x, and the 14B teacher managed 9x1.00x,
  # 2x1.08x, 1x2.03x. If 32B also lands on 1.00x then the speed dimension is dead for the
  # whole open-weight range and headroom-normalised scoring cannot be rescued by scale.
  # 32B bf16 ~64GB: 2 GPUs, so weights plus activations are not fighting for one 80GB card.
  echo "R2  can a >14B model beat the baseline? (32B speedup probe)"
  GPUS=2 WALL="03:00:00" go "r2-probe-32b" "python scripts/make_teacher_kernels.py \
--model Qwen/Qwen2.5-Coder-32B-Instruct --gpus 0,1 --k 6 \
--out $OUT/teacher_kernels_q32.json"
  echo "    READ-OUT: the speedup_eager distribution. >1.1x on a decent fraction of tasks"
  echo "    means scale restores the speed dimension and Tier-2 should move up a size class."
  ;;

r3)
  # ROUTE 3 -- score correctness-acquisition directly (KA_SCORE=passrate), pre-registered as
  # Amendment 1 in docs/PREREGISTRATION.md BEFORE these runs land. Same E1 design, so it is a
  # like-for-like re-read of an experiment already shown to be flat under the saturated metric.
  MODELS=("Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct")
  TAGS=("q15" "q3")
  if [ -z "$DRY" ] && ! mrl run "test -s $TEACHER" >/dev/null 2>&1; then
    echo "!! teacher kernels missing at $TEACHER -- run the harvest first"; exit 1
  fi
  echo "R3  E1 re-read under KA_SCORE=passrate (fraction of k that verify)"
  for i in "${!MODELS[@]}"; do
    m="${MODELS[$i]}"; t="${TAGS[$i]}"
    for s in 1 2; do
      go "r3c-$t-s$s" "KA_SCORE=passrate python -m kernelascent.v3.lab_compounding --model $m \
--gpus 0 --rounds 6 --seed $s --outdir $OUT/r3_control_${t}_s${s}"
      go "r3i-$t-s$s" "KA_SCORE=passrate python -m kernelascent.v3.lab_compounding --model $m \
--gpus 0 --rounds 6 --seed $s --inject-kernels $TEACHER --inject-per-task 1 \
--outdir $OUT/r3_inject_${t}_s${s}"
    done
  done
  echo
  echo "  DECISION RULE (fixed; see PREREGISTRATION.md Amendment 1):"
  echo "    inject lineage-reset > 0, control flat -> the loop registers acquired coverage but"
  echo "                                              does not generate it: a real finding"
  echo "    both flat                              -> instrument still dead; publish no"
  echo "                                              compounding claim from this hardware"
  echo "    NEVER pool these with the A100 headroom boards -- different metric AND hardware."
  ;;

r4)
  # ROUTE 4 -- is the plateau an artifact of the PROMPT?
  # Of 86 verified kernels harvested from the 14B teacher, ZERO contained triton, CUDA or
  # load_inline: 100% were pure-PyTorch rewrites of the reference, median speedup 1.00x over
  # eager. The published prompt contains "a plain-torch kernel that is correct beats a fancy
  # one that errors", which steers exactly that way. A rewrite scores _score(correct, 1.0) =
  # 0.50 by construction -- the value 76% of all H100 scores took.
  # So before concluding anything about models or hardware, A/B the prompt on identical tasks,
  # grader and seed. Same harvest script, same bank, only KA_PROMPT differs.
  echo "R4  prompt A/B: does asking for a real kernel produce one?"
  for v in safe kernel; do
    GPUS=2 WALL="02:00:00" go "r4-$v-14b" "KA_PROMPT=$v python scripts/make_teacher_kernels.py \
--model Qwen/Qwen2.5-Coder-14B-Instruct --gpus 0,1 --k 6 \
--out $OUT/r4_${v}_q14.json"
  done
  echo "    READ-OUT: custom-kernel rate and speedup distribution, safe vs kernel."
  echo "      kernel arm produces custom kernels AND speedups > 1.05x -> the plateau is"
  echo "        substantially prompt-induced and every headroom board needs re-running"
  echo "      kernel arm still 100% plain-torch -> the models genuinely cannot write kernels,"
  echo "        and the published prompt was not the binding constraint"
  echo "      kernel arm writes kernels that FAIL to verify -> the bottleneck is formation,"
  echo "        not intent; report coverage-vs-ambition explicitly"
  ;;

*) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
