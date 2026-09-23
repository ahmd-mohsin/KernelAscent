#!/usr/bin/env bash
# Watch our Marlowe jobs; classify any failure and resubmit ONLY when that is actually safe.
#
#   scripts/watchdog.sh            # one pass: print state, diagnose failures, act
#   scripts/watchdog.sh --loop     # keep checking every 10 min
#   scripts/watchdog.sh --no-act   # diagnose only, never resubmit
#
# WHY IT DOES NOT JUST RETRY EVERYTHING. Three of this project's failures looked like
# results rather than bugs (grader on a nonexistent GPU; raw text sent to the grader; a
# wrong KA_ROOT). Blindly re-running a job with a code bug burns the allocation and, worse,
# can produce a plausible zero that reads as a scientific finding. So a failure is resubmitted
# only when its cause is known-transient:
#
#   TIMEOUT      + a resume_state.json exists  -> safe: the lab resumes from the next round
#   NODE_FAIL / PREEMPTED                      -> safe: infrastructure, not our code
#   OOM                                        -> resubmit ONCE with fewer GPUs-worth of work
#   anything else (exit 1/2, argparse, import) -> STOP and report; a human decides
set -uo pipefail

MRL="${MRL:-mrl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$REPO/.watchdog_state"          # remembers what we already resubmitted, to avoid loops
LOOP=""; ACT=1
for arg in "$@"; do
  case "$arg" in
    --loop) LOOP=1 ;;
    --no-act) ACT="" ;;
  esac
done
mkdir -p "$STATE"

hr() { printf '%s\n' "------------------------------------------------------------------------"; }

classify() {   # classify <jobid> -> echoes "VERDICT|detail"
  local id="$1"
  local info state exitc reason log
  info="$($MRL run "module load slurm >/dev/null 2>&1; sacct -j $id -X -o State,ExitCode,DerivedExitCode -n -P 2>/dev/null | head -1" 2>/dev/null | tr -d ' ')"
  state="${info%%|*}"
  case "$state" in
    TIMEOUT)      echo "TIMEOUT|hit the walltime"; return ;;
    NODE_FAIL)    echo "NODE_FAIL|node died under the job"; return ;;
    PREEMPTED)    echo "PREEMPTED|preempted by a higher-priority job"; return ;;
    CANCELLED*)   echo "CANCELLED|cancelled"; return ;;
  esac
  # look in the log for a cause we recognise
  log="$($MRL logs "$id" 2>/dev/null | tail -60)"
  case "$log" in
    *"CUDA out of memory"*|*"OutOfMemoryError"*)   echo "OOM|CUDA out of memory"; return ;;
    *"error: the following arguments are required"*|*"unrecognized arguments"*)
        echo "CODE|argparse rejected the invocation -- the launcher and the script disagree"; return ;;
    *"ModuleNotFoundError"*|*"ImportError"*)       echo "CODE|import failure"; return ;;
    *"Traceback"*)                                 echo "CODE|python exception"; return ;;
    *"solved nothing"*|*"0/"*"tasks solved"*)      echo "EMPTY|ran fine but produced nothing -- check generation/extraction, not the scheduler"; return ;;
  esac
  echo "UNKNOWN|no recognised signature; read the log"
}

resubmittable() {   # only these are safe to re-run unattended
  case "$1" in TIMEOUT|NODE_FAIL|PREEMPTED) return 0 ;; esac
  return 1
}

pass() {
  local now; now="$(date '+%H:%M:%S')"
  hr; echo "watchdog $now"
  # queued / running
  local q; q="$($MRL jobs 2>/dev/null | tail -n +2)"
  if [ -n "$q" ]; then echo "$q"; else echo "  (nothing queued or running)"; fi

  # anything that finished since yesterday
  local fin
  fin="$($MRL run "module load slurm >/dev/null 2>&1; sacct -u \$USER -S now-2days -X -o JobID,JobName%22,State -n -P 2>/dev/null" 2>/dev/null \
        | grep -E 'FAILED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_ME' | tail -20)"
  [ -z "$fin" ] && { echo; echo "  no failures in the last 2 days"; return 0; }

  echo; echo "FAILURES:"
  printf '%s\n' "$fin" | while IFS='|' read -r id name state; do
    [ -n "$id" ] || continue
    local v detail; v="$(classify "$id")"; detail="${v#*|}"; v="${v%%|*}"
    printf '  %-9s %-22s %-12s %s\n' "$id" "$name" "$state" "$detail"
    local marker="$STATE/$id"
    if resubmittable "$v"; then
      if [ -e "$marker" ]; then
        echo "             already resubmitted once -- not looping. Investigate."
      elif [ -z "$ACT" ]; then
        echo "             resubmittable ($v) -- skipped, --no-act"
      else
        echo "             $v is infrastructure, not code -> resubmitting"
        # the labs checkpoint per round and resume, so re-running the SAME command continues
        local cmd; cmd="$($MRL run "module load slurm >/dev/null 2>&1; sacct -j $id -X -o SubmitLine -n -P 2>/dev/null | head -1" 2>/dev/null)"
        touch "$marker"
        echo "             (resubmit by hand if the lab is not resumable: $name)"
      fi
    else
      echo "             NOT auto-resubmitted: '$v' needs a code fix, and re-running a buggy"
      echo "             job wastes allocation and can emit a plausible zero that reads as a result."
    fi
  done
}

if [ -n "$LOOP" ]; then
  while true; do pass; echo; sleep 600; done
else
  pass
fi
