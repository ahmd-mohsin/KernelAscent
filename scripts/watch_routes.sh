#!/usr/bin/env bash
# Emit one line per job STATE CHANGE, and auto-continue chunks that hit their walltime.
# Covers every terminal state, not just success: a silent monitor must mean "nothing changed",
# never "it crashed and my filter didn't match".
export PATH="$HOME/.marlowe/bin:$PATH"
REPO=/Users/muahmed/Desktop/Projects/KernelAscent
LOG="${WATCH_LOG:-/tmp/ka_watch.log}"        # every poll recorded, so a miss is diagnosable
prev=""
polls=0
while true; do
  if ! ssh -O check marlowe >/dev/null 2>&1; then
    echo "SSH MASTER DOWN -- cluster polling stopped; needs 'ssh -f -N marlowe' + Duo"
    sleep 180; continue
  fi
  cur="$(mrl run "module load slurm >/dev/null 2>&1
squeue -u \$USER -h -o '%i|%j|%T' 2>/dev/null
sacct -u \$USER -n -X -o JobID,JobName%30,State -P -S now-1days 2>/dev/null |
  awk -F'|' '\$3!~/RUNNING|PENDING/{print \$1\"|\"\$2\"|\"\$3}'" 2>/dev/null |
        grep -E '^[0-9]+\|' | sort -u)"
  # An empty poll used to `continue` silently, so a run of failed queries was
  # indistinguishable from a quiet cluster -- and a monitor whose silence is ambiguous is
  # not a monitor. Warn once after two consecutive empty polls.
  if [ -z "$cur" ]; then
    empties=$((${empties:-0} + 1))
    [ "$empties" = 2 ] && echo "POLL RETURNING EMPTY x2 -- cluster query failing; job states are NOT being watched"
    sleep 180; continue
  fi
  if [ "${empties:-0}" -ge 2 ]; then echo "poll recovered -- watching again"; fi
  empties=0
  polls=$((polls + 1))
  printf '%s poll=%d rows=%d\n' "$(date +%H:%M:%S)" "$polls" "$(wc -l <<<"$cur")" >> "$LOG"
  printf '%s\n' "$cur" > "$LOG.last"
  # HEARTBEAT. Without one, "no events" is ambiguous between a quiet cluster and a dead
  # monitor -- and this monitor has now missed two real transitions while appearing healthy.
  # One line per ~30 min keeps silence meaningful without drowning the real events.
  if [ $((polls % 10)) -eq 1 ]; then
    echo "heartbeat: $(grep -c "RUNNING" <<<"$cur") running, $(grep -c "PENDING" <<<"$cur") pending (poll $polls)"
  fi
  # RECONCILE ON STARTUP. The auto-continue below fires on a state DIFF between polls, and a
  # fresh watcher has no previous poll, so a transition that lands while no watcher is running
  # is seen by nobody. fed-q15-s2 hit its 2-hour wall in the gap between two 30-minute monitor
  # instances and simply stopped: the queue drained to zero with the cell at 3 of 5 rounds and
  # nothing reported it, because "no diff" and "no baseline" are the same thing on poll 1.
  #
  # So the first poll reconciles against the manifest instead of only recording a baseline.
  # jobman continue is idempotent -- it skips anything queued or running, holds a stalled cell,
  # and resubmits only a timed-out one -- so doing this every time the watcher starts is safe
  # and costs one query.
  if [ -z "$prev" ]; then
    # `continue ` with the trailing space, so the summary line "continued 0 cell(s)" does not
    # match: a clean reconcile must be silent, or every watcher restart prints a line saying
    # nothing happened and the operator stops reading the ones that mean something.
    bash "$REPO/scripts/jobman.sh" continue 2>&1 | grep -E '^(continue |queued |HOLD |ABORT)' \
      | sed 's/^/startup reconcile: /' || true
  fi
  if [ -n "$prev" ]; then
    # only lines that are new or changed since last poll
    comm -13 <(echo "$prev") <(echo "$cur") | while IFS='|' read -r id name st; do
      # A CANCELLED job whose NAME now has a higher job id is one I deliberately replaced
      # (relaunched with a corrected command). Alarming on those is alert fatigue, and this
      # monitor is only useful if every line it prints is worth reading.
      # A cancelled cell is benign if it is PARKED for resubmission, as well as if a newer
      # job already carries its name. Bulk re-parking (e.g. repointing 14 inject arms at a
      # better teacher) cancels many at once and the replacements queue gradually, so the
      # newer-id test alone flagged 8 of 9 as failures. Alert noise from my own actions is how
      # a monitor stops being read.
      if [ "${st#CANCELLED}" != "$st" ] && \
         { awk -F'|' -v n="$name" -v i="$id" '$2==n && $1+0>i+0{f=1} END{exit !f}' <<<"$cur" \
           || cut -f1 "$REPO/.jobman.backlog" 2>/dev/null | grep -qx "$name"; }; then
        echo "note    $name ($id) cancelled -- superseded or parked for resubmission"
        continue
      fi
      case "$st" in
        RUNNING)          echo "RUN     $name ($id)" ;;
        COMPLETED)        echo "DONE    $name ($id)" ;;
        TIMEOUT*)         echo "TIMEOUT $name ($id) -- resumable, continuing" ;;
        FAILED*|CANCELLED*|NODE_FAIL*|PREEMPTED*|OUT_OF_ME*|BOOT_FAIL*|DEADLINE*)
                          echo "!! $st $name ($id)" ;;
        PENDING)          : ;;
        *)                echo "?? $st $name ($id)" ;;
      esac
    done
    # RESUME BEFORE STARTING NEW WORK. These two ran the other way round and it cost real
    # compute: topup filled every free slot from the backlog, then continue tried to resubmit
    # three timed-out dose cells, sbatch rejected all three at the submit cap, and the cells
    # simply left the queue. Fresh cells at round 0 had displaced in-flight cells holding a
    # checkpoint and an hour of grading -- and because a rejected submission looks the same to
    # the stall ledger as a cell that made no progress, one of them had its stall counter
    # incremented toward a HOLD it had not earned.
    #
    # A cell mid-trajectory is strictly more valuable than a parked one: it has already spent
    # the compute, and its arms cannot be re-derived from anything else. Resume first, let the
    # backlog have what is left.
    # auto-continue only walltime hits; a code failure re-run wastes allocation (standing rule 5)
    if comm -13 <(echo "$prev") <(echo "$cur") | grep -q 'TIMEOUT'; then
      bash "$REPO/scripts/jobman.sh" continue 2>&1 | grep -E '^(continue|queued|HOLD)' || true
    fi
    # The account caps concurrent submissions, so finished jobs free slots that parked work
    # should claim immediately -- otherwise the queue drains and nothing replaces it overnight.
    if [ -s "$REPO/.jobman.backlog" ]; then
      bash "$REPO/scripts/jobman.sh" topup 2>&1 | grep -E '^  submitted|^  FAILED' || true
    fi
  fi
  prev="$cur"
  sleep 180
done
