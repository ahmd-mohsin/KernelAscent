#!/usr/bin/env bash
# The Tier-2 program, re-run on KERNEL AUTHORING -- the first task in this project the models
# have not saturated (0/16 custom-kernel attempts under the published prompt; 14/16 when asked,
# 1 of which verifies). Every previous saturation -- the 0.50 floor, the 1.000 ceiling, an
# injection control exhausted by round 2 -- came from measuring a task the models had already
# solved. This measures the task the benchmark claims to be about.
#
# Only possible since the grader was fixed: triton could not verify at all before today.
#
#   scripts/launch_kernel_program.sh t1     capability curve across scale
#   scripts/launch_kernel_program.sh t2     compounding (lineage vs reset), control + inject
#   scripts/launch_kernel_program.sh bank   calibrate the 456-task DSL bank under the kernel prompt
#   scripts/launch_kernel_program.sh base   recursion vs sampling at matched budget
#   scripts/launch_kernel_program.sh mech   mechanism probe (diversity / drift / retention)
#   scripts/launch_kernel_program.sh all
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JM="$REPO/scripts/jobman.sh"
OUT="/users/muahmed/ka_data"
# Short walltimes: the labs checkpoint per round, and backfill is governed by the request.
# These are sized from measured runtimes (r4 arms took 15-22 min at k=6 over 29 tasks).
DRY=""; [ "${1:-}" = "--dry" ] && { DRY=1; shift; }
STEP="${1:-help}"

go() {  # go <name> <wall> <gpus> <command...>
  local name="$1" wall="$2" gpus="$3"; shift 3
  if [ -n "$DRY" ]; then printf '  %-26s (%sg %s) %s\n' "$name" "$gpus" "$wall" "$*"; return 0; fi
  "$JM" run "$name" "$wall" "$gpus" -- "$*" 2>&1 | tail -1
}

do_t1() {
  # CAPABILITY CURVE on kernel authoring. k=12 because the success rate is ~7%: at k=6 a task
  # with p=0.07 is missed 65% of the time, and a coverage number built on that is mostly noise.
  echo "T1-kernel  capability across scale, KA_PROMPT=kernel, k=12"
  for spec in "q05 Qwen/Qwen2.5-Coder-0.5B-Instruct 1 00:40:00" \
              "q15 Qwen/Qwen2.5-Coder-1.5B-Instruct 1 00:40:00" \
              "q3  Qwen/Qwen2.5-Coder-3B-Instruct   1 00:45:00" \
              "q7  Qwen/Qwen2.5-Coder-7B-Instruct   1 00:50:00" \
              "q14 Qwen/Qwen2.5-Coder-14B-Instruct  2 00:55:00"; do
    set -- $spec
    go "t1k-$1" "$4" "$3" "env KA_PROMPT=kernel python scripts/make_teacher_kernels.py \
--model $2 --gpus $([ "$3" = 2 ] && echo 0,1 || echo 0) --k 12 --out $OUT/t1k_$1.json"
  done
  echo "    READ-OUT: verified-kernel rate by scale. This is the capability curve on the task"
  echo "    the benchmark is named after, measured for the first time."
}

do_t2() {
  # COMPOUNDING on kernel authoring. The headline contrast, on an unsaturated task.
  # k=10 balances signal against round cost; 6 rounds, chunked at 50 min (checkpoints per round).
  echo "T2-kernel  lineage vs matched reset under KA_PROMPT=kernel"
  for m in "q15 Qwen/Qwen2.5-Coder-1.5B-Instruct" "q3 Qwen/Qwen2.5-Coder-3B-Instruct"; do
    set -- $m
    for s in 1 2; do
      go "t2k-c-$1-s$s" "00:50:00" 1 "env KA_PROMPT=kernel python -m kernelascent.v3.lab_compounding \
--model $2 --gpus 0 --rounds 6 --seed $s --k 10 --outdir $OUT/t2k_control_$1_s$s"
      go "t2k-i-$1-s$s" "00:50:00" 1 "env KA_PROMPT=kernel python -m kernelascent.v3.lab_compounding \
--model $2 --gpus 0 --rounds 6 --seed $s --k 10 \
--inject-kernels $OUT/t1k_q14.json --inject-per-task 1 --outdir $OUT/t2k_inject_$1_s$s"
    done
  done
  echo "    NOTE: the inject arm consumes t1k_q14.json, so it must land after T1's 14B cell."
  echo "    DECISION RULE (fixed now): lineage-reset > 0 with a CI excluding zero, on a task"
  echo "    where neither arm saturates, is the compounding result this benchmark exists to"
  echo "    produce. Flat with a working positive control is a real null. Both flat means the"
  echo "    task is still wrong and nothing is published."
}

do_bank() {
  # REACHABILITY of the DSL bank under the kernel prompt. The 456-task bank was calibrated on
  # rewriting; on authoring its difficulty distribution is a different question entirely.
  echo "BANK  456-task DSL calibration under KA_PROMPT=kernel (resumable chunks)"
  go "bank-kernel-dsl" "00:50:00" 1 "env KA_PROMPT=kernel python -m kernelascent.v3.difficulty_filter \
--model Qwen/Qwen2.5-Coder-3B-Instruct --gpus 0 \
--in dataset/kernel_bank/kernel_tasks_dsl_validated.json \
--out $OUT/bank_dsl_kernel.json --report $OUT/bank_dsl_kernel_report.json \
--k 6 --keep-hi 0.75 --min-ceiling 1.3"
}

do_base() {
  # RECURSION vs SAMPLING on kernel authoring. "Search beats training at matched budget" is a
  # published POSITIVE of this benchmark, measured on rewriting. It has to be re-tested on the
  # task the benchmark is actually about -- a positive that only holds on the saturated task is
  # as much an artifact as a null that only holds there.
  echo "BASE-kernel  best-of-k / self-refine / retrieval at matched budget"
  for m in "q15 Qwen/Qwen2.5-Coder-1.5B-Instruct" "q3 Qwen/Qwen2.5-Coder-3B-Instruct"; do
    set -- $m
    for s in 1 2; do
      go "basek-$1-s$s" "00:45:00" 1 "env KA_PROMPT=kernel python -m kernelascent.v3.lab_baselines \
--model $2 --gpus 0 --rounds 6 --k 10 --seed $s --outdir $OUT/basek_$1_s$s"
    done
  done
  echo "    Pairs with t2k: recursion gain = T2 final C minus the best non-recursive baseline"
  echo "    at the SAME generation budget. Note the pass-rate caveat does not apply here --"
  echo "    these are scored on the default metric, where best-of-k's max-over-draws works."
}

do_mech() {
  # WHY it does or does not compound, on the real task: generation diversity, LoRA drift by
  # depth, retention, train-held transfer gap. The published mechanism story was measured on
  # rewriting and inherits the same scope problem.
  echo "MECH-kernel  mechanism probe (diversity / drift / retention)"
  for m in "q15 Qwen/Qwen2.5-Coder-1.5B-Instruct" "q3 Qwen/Qwen2.5-Coder-3B-Instruct"; do
    set -- $m
    go "mechk-$1" "00:50:00" 1 "env KA_PROMPT=kernel python -m kernelascent.v3.lab_rsi_mechanism \
--model $2 --gpu 0 --rounds 6 --k 10 --seed 1 --outdir $OUT/mechk_$1"
  done
}

case "$STEP" in
  base) do_base ;;
  mech) do_mech ;;
  t1)   do_t1 ;;
  t2)   do_t2 ;;
  bank) do_bank ;;
  all)  do_t1; echo; do_t2; echo; do_bank; echo; do_base; echo; do_mech ;;
  *)    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
