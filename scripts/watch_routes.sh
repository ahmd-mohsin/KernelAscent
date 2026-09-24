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
  if [ -n "$prev" ]; then
    # only lines that are new or changed since last poll
    comm -13 <(echo "$prev") <(echo "$cur") | while IFS='|' read -r id name st; do
      # A CANCELLED job whose NAME now has a higher job id is one I deliberately replaced
      # (relaunched with a corrected command). Alarming on those is alert fatigue, and this
      # monitor is only useful if every line it prints is worth reading.
      if [ "${st#CANCELLED}" != "$st" ] && \
         awk -F'|' -v n="$name" -v i="$id" '$2==n && $1+0>i+0{f=1} END{exit !f}' <<<"$cur"; then
        echo "note    $name ($id) cancelled -- superseded by a newer submission"
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
    # The account caps concurrent submissions, so finished jobs free slots that parked work
    # should claim immediately -- otherwise the queue drains and nothing replaces it overnight.
    if [ -s "$REPO/.jobman.backlog" ]; then
      bash "$REPO/scripts/jobman.sh" topup 2>&1 | grep -E '^  submitted|^  FAILED' || true
    fi
    # auto-continue only walltime hits; a code failure re-run wastes allocation (standing rule 5)
    if comm -13 <(echo "$prev") <(echo "$cur") | grep -q 'TIMEOUT'; then
      bash "$REPO/scripts/jobman.sh" continue 2>&1 | grep -E '^(continue|queued|HOLD)' || true
    fi
  fi
  prev="$cur"
  sleep 180
done
