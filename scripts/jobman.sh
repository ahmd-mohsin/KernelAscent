#!/usr/bin/env bash
# Submit jobs with a recorded command, so a timed-out chunk can be CONTINUED automatically.
#
# On this cluster the scheduling estimate is dominated by requested walltime: a 40-minute job
# started in 2 hours while an otherwise identical 6-hour job was estimated 34 hours out. The
# fix is to run long experiments as SHORT RESUMABLE CHUNKS -- lab_track_c and lab_compounding
# both checkpoint per round and resume from resume_state.json, so re-running the same command
# continues rather than restarting.
#
# Guessing the number of chunks up front is wrong (you don't know the per-round cost yet), so
# instead we record each job's command in a manifest and let `continue` re-issue it whenever a
# chunk hits its walltime. That adapts to the real runtime and stops when the lab reports done.
#
#   scripts/jobman.sh run <name> <walltime> <gpus> -- <command>   submit + record
#   scripts/jobman.sh continue                                    resubmit timed-out chunks
#   scripts/jobman.sh status                                      manifest vs live queue
set -uo pipefail

MRL="${MRL:-mrl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAN="$REPO/.jobman.tsv"        # name <TAB> walltime <TAB> gpus <TAB> command
touch "$MAN"

_jid() { sed -n 's/.*job \([0-9][0-9]*\) queued.*/\1/p' <<<"$1" | head -1; }

case "${1:-status}" in

run)
  name="${2:?name}"; wall="${3:?walltime}"; gpus="${4:?gpus}"; shift 4
  [ "${1:-}" = "--" ] && shift
  cmd="$*"
  out="$($MRL submit -J "$name" -N 1 -G "$gpus" -t "$wall" -- "$cmd" 2>&1)"
  jid="$(_jid "$out")"
  if [ -z "$jid" ]; then
    echo "SUBMIT FAILED for $name"; printf '%s\n' "$out"; exit 1
  fi
  # one manifest line per experiment cell; re-running replaces it
  grep -v "^$name	" "$MAN" > "$MAN.tmp" 2>/dev/null || true
  printf '%s\t%s\t%s\t%s\n' "$name" "$wall" "$gpus" "$cmd" >> "$MAN.tmp"
  mv "$MAN.tmp" "$MAN"
  echo "queued $jid  $name"
  ;;

continue)
  # Resubmit any recorded cell whose most recent job hit the walltime and which is not
  # currently queued or running. TIMEOUT is the only state we auto-continue: the lab wrote its
  # checkpoint, so the same command picks up at the next round. A FAILED job is left alone --
  # three failures in this project were code bugs that would have produced a plausible zero.
  live="$($MRL run "module load slurm >/dev/null 2>&1; squeue -u \$USER -h -o %j 2>/dev/null" 2>/dev/null)"
  n=0
  while IFS=$'\t' read -r name wall gpus cmd; do
    [ -n "$name" ] || continue
    if grep -qx "$name" <<<"$live"; then continue; fi          # still queued/running
    st="$($MRL run "module load slurm >/dev/null 2>&1; sacct -u \$USER -n -X -o JobName%40,State -P -S now-3days 2>/dev/null | grep '^$name|' | tail -1" 2>/dev/null)"
    state="${st##*|}"
    case "$state" in
      TIMEOUT*)
        echo "continue  $name  (last chunk hit $wall; resuming from its checkpoint)"
        "$0" run "$name" "$wall" "$gpus" -- "$cmd" ; n=$((n+1)) ;;
      COMPLETED*) : ;;                                          # done, nothing to do
      "")         echo "skip      $name  (no record yet)" ;;
      *)          echo "HOLD      $name  (last state '$state' -- not auto-continued; needs a look)" ;;
    esac
  done < "$MAN"
  echo "continued $n cell(s)"
  ;;

status)
  printf '%-26s %-9s %-4s %s\n' NAME WALL GPUS LAST-STATE
  live="$($MRL run "module load slurm >/dev/null 2>&1; squeue -u \$USER -h -o '%j %T' 2>/dev/null" 2>/dev/null)"
  while IFS=$'\t' read -r name wall gpus cmd; do
    [ -n "$name" ] || continue
    l="$(grep -m1 "^$name " <<<"$live" | awk '{print $2}')"
    if [ -n "$l" ]; then s="$l (live)"
    else
      s="$($MRL run "module load slurm >/dev/null 2>&1; sacct -u \$USER -n -X -o JobName%40,State -P -S now-3days 2>/dev/null | grep '^$name|' | tail -1" 2>/dev/null)"
      s="${s##*|}"; [ -n "$s" ] || s="-"
    fi
    printf '%-26s %-9s %-4s %s\n' "$name" "$wall" "$gpus" "$s"
  done < "$MAN"
  ;;

*) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
