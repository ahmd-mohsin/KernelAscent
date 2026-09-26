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

# watch_routes.sh exports this and jobman did not, so jobman worked when the watcher invoked it
# and failed when a shell invoked it directly -- every $MRL call returning empty, which `continue`
# then reported as "no record yet" for all 21 cells and exited 0. The environment a script needs
# is the script's business, not the caller's.
export PATH="$HOME/.marlowe/bin:$PATH"
MRL="${MRL:-mrl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAN="$REPO/.jobman.tsv"        # name <TAB> walltime <TAB> gpus <TAB> command
touch "$MAN"

# Fail ONCE, clearly, if the ssh master is down. Without this every manifest row runs its own
# remote query and prints the whole "authenticate once" banner, burying the actual status in
# twenty copies of the same message.
#
# Scoped to the verbs that actually talk to the cluster. `park`, `list` and `drop` only write
# local files, and gating them on a live tunnel meant an experiment matrix could not be prepared
# while the tunnel was down -- which is precisely when you would want to prepare one, since the
# work is queued the moment it comes back. Found by trying to park seven cells after the socket
# dropped and having every one of them refused for a reason that did not apply to parking.
case "${1:-}" in
  park|list|drop|help|"") ;;
  *)
    if ! ssh -O check marlowe >/dev/null 2>&1; then
      echo "no live SSH master -- Marlowe needs SUNet password + Duo once:"
      echo "    ssh -f -N marlowe"
      echo "(a laptop sleeping drops the socket even inside ControlPersist)"
      exit 1
    fi ;;
esac

# The manifest and backlog are line-based TSV: one record is name<TAB>wall<TAB>gpus<TAB>command.
# A command containing a newline cannot be represented, and nothing was checking. Parking a
# python -c with an inline script wrote eight records, seven of them fragments of the script,
# and `continue` would have tried to resubmit lines like "    try:" as if they were cells. The
# job itself submitted fine, so the corruption was invisible until the store was read back.
#
# Refuse rather than mangle. A multi-line command belongs in a file the job invokes.
_assert_one_line() {
  # $'\n' rather than "$(printf '\n')": command substitution strips TRAILING newlines, so the
  # latter evaluates to the empty string and the pattern *""* matches every command. The first
  # version of this guard refused everything, including `echo hello`, and its positive test is
  # what caught it -- a guard that rejects all input passes any negative test you write for it.
  case "$1" in
    *$'\n'*)
      echo "refusing: the command contains a newline, and the manifest is line-based." >&2
      echo "          Put the script in a file and invoke that file instead." >&2
      exit 2 ;;
    *$'\t'*)
      echo "refusing: the command contains a tab, which is the field separator." >&2
      exit 2 ;;
  esac
}

_jid() { sed -n 's/.*job \([0-9][0-9]*\) queued.*/\1/p' <<<"$1" | head -1; }

# The account caps concurrent submissions (MaxSubmitJobsPerAccount; observed limit 32). Work
# beyond that cannot be queued, so it is parked in a backlog and submitted as slots free. This
# is the real constraint on throughput here -- not GPU-hours, of which we have thousands.
BACKLOG="$REPO/.jobman.backlog"
touch "$BACKLOG"

# A PROGRESS LEDGER, because "continue" cannot tell a cell that is advancing from one that is
# looping. Both look like TIMEOUT. Two separate incidents so far: 19 cells restarting at round 0
# forever because their lab had no resume, and 4 baselines cells whose smallest resumable unit
# (one method) grew past the walltime when the token budget was corrected -- each chunk burned
# 50 minutes, recorded nothing, and was faithfully resubmitted. The queue looked busy in both.
#
# name <TAB> fingerprint <TAB> consecutive-no-progress-count
PROGRESS="$REPO/.jobman.progress"
touch "$PROGRESS"
TAB=$'\t'      # `grep "\t"` is a basic-regex literal "t", not a tab -- name the character

# Fingerprint every cell in ONE remote call: rounds recorded, methods recorded, or file size.
# Cheap enough to run on every `continue`, which is the only way it gets used.
_fingerprints() {
  $MRL run "python3 - <<'PYEOF'
import json, os, glob
root = '/users/muahmed/ka_data'
for p in sorted(glob.glob(os.path.join(root, '*'))):
    name = os.path.basename(p)
    fp = None
    if os.path.isdir(p):
        for fn in ('weight_rsi.json', 'compounding.json', 'track_c.json', 'selfplay_rsi.json'):
            f = os.path.join(p, fn)
            if os.path.exists(f):
                try: fp = 'rounds=%d' % len((json.load(open(f)).get('history') or []))
                except Exception: fp = 'unreadable'
                break
        if fp is None:
            f = os.path.join(p, 'baselines.json')
            if os.path.exists(f):
                # A baselines cell is done once it carries all three non-recursive arms. Its
                # fingerprint is 'methods=...' and never the literal 'complete', so the
                # done-verdict below could not fire for it and basek-q3-s1 -- which holds all
                # three, written atomically before an OOM killed the process -- would be
                # reported as needing a look on every continue, forever. That is the same
                # alert-fatigue failure t1k-q14 caused, in a second shape.
                # The partial form stays verbatim, because the stall detector compares it
                # across continues to tell a cell that is progressing from one that is repeating.
                try:
                    _m = sorted(json.load(open(f)).get('results') or {})
                    fp = 'complete' if set(_m) >= {'best_of_k', 'retrieval', 'self_refine'} \
                         else 'methods=%s' % ','.join(_m)
                except Exception: fp = 'unreadable'
        if fp is None:
            fp = 'files=%d' % len(os.listdir(p))
    elif p.endswith('.json'):
        # A file-output cell records its own completeness. Without this, a cell that FINISHED
        # (t1k-q14: complete, 29/29 tasks, a genuine zero) is flagged 'needs a look' on every
        # single continue, forever -- which is how a watchdog stops being read.
        try:
            _d = json.load(open(p))
            fp = 'complete' if isinstance(_d, dict) and _d.get('complete') else 'bytes=%d' % os.path.getsize(p)
        except Exception:
            try: fp = 'bytes=%d' % os.path.getsize(p)
            except Exception: fp = 'gone'
    if fp: print('%s\t%s' % (name, fp))
PYEOF" 2>/dev/null
}

# Map a recorded command to the cell name its fingerprint lives under.
_cell_of() {
  local cmd="$1" v
  v="$(sed -n 's/.*--outdir[ =]\([^ ]*\).*/\1/p' <<<"$cmd" | head -1)"
  [ -z "$v" ] && v="$(sed -n 's/.*--out[ =]\([^ ]*\).*/\1/p' <<<"$cmd" | head -1)"
  [ -n "$v" ] && basename "$v"
}
CAP="${KA_SUBMIT_CAP:-32}"

# MaxSubmitJobsPerAccount counts EVERY job in the allocation, not just yours -- another member's
# queue eats your room. Count the account, not the user, or topup keeps optimistically trying and
# reporting a confusing submit failure.
# Returns the queue depth, or EMPTY if the query failed. The distinction matters: an empty
# result used to be coerced to 0 by `${n:-0}`, which reads a transient SSH hiccup as "the
# account has no jobs" and hands topup the full cap. Observed 2026-09-24: topup printed
# `queued=0 cap=32 room=32` while 18 jobs were in fact queued. Nothing bad happened only
# because sbatch enforces the cap server-side and refused -- i.e. the safety came from Slurm,
# not from this script. A count this script cannot verify is not a count.
_queued() {
  local out
  out="$($MRL run "module load slurm >/dev/null 2>&1; squeue -A ${MRL_ACCOUNT:-marlowe-m000215-pm06} -h 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')"
  [ -n "$out" ] && { printf '%s' "$out"; return 0; }
  # fall back to the per-user view before giving up; they agree in practice
  out="$($MRL run "module load slurm >/dev/null 2>&1; squeue -u \$USER -h 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')"
  printf '%s' "$out"
}

case "${1:-status}" in

park)
  # park <name> <walltime> <gpus> -- <command>   record without submitting
  name="${2:?name}"; wall="${3:?walltime}"; gpus="${4:?gpus}"; shift 4
  [ "${1:-}" = "--" ] && shift
  _assert_one_line "$*"
  grep -v "^$name	" "$BACKLOG" > "$BACKLOG.tmp" 2>/dev/null || true
  printf '%s\t%s\t%s\t%s\n' "$name" "$wall" "$gpus" "$*" >> "$BACKLOG.tmp"
  mv "$BACKLOG.tmp" "$BACKLOG"
  # ALSO update the manifest. `continue` resubmits from the manifest, so parking a corrected
  # command while leaving the old one there means auto-continuation keeps reviving the version
  # you just fixed -- which is how four baselines cells kept OOMing after the fix was written.
  grep -v "^$name	" "$MAN" > "$MAN.tmp" 2>/dev/null || true
  printf '%s\t%s\t%s\t%s\n' "$name" "$wall" "$gpus" "$*" >> "$MAN.tmp"
  mv "$MAN.tmp" "$MAN"
  echo "parked $name ($(wc -l < "$BACKLOG" | tr -d ' ') in backlog)"
  ;;

topup)
  # Submit from the backlog while the account has room. Run it after jobs finish.
  n="$(_queued)"
  if [ -z "$n" ]; then
    echo "REFUSING to top up: cannot read the queue depth."
    echo "  An unreadable queue is not an empty queue. Treating it as 0 would hand this script"
    echo "  the whole cap and fire the entire backlog at once. Retry when the cluster answers."
    exit 1
  fi
  room=$(( CAP - n ))
  echo "queued=$n cap=$CAP room=$room backlog=$(wc -l < "$BACKLOG" | tr -d ' ')"
  [ "$room" -gt 0 ] || { echo "no room -- try again when jobs finish"; exit 0; }
  sent=0
  while [ "$sent" -lt "$room" ]; do
    line="$(head -1 "$BACKLOG")"
    [ -n "$line" ] || break
    name="$(cut -f1 <<<"$line")"; wall="$(cut -f2 <<<"$line")"
    gpus="$(cut -f3 <<<"$line")"; cmd="$(cut -f4- <<<"$line")"
    if "$0" run "$name" "$wall" "$gpus" -- "$cmd" >/dev/null 2>&1; then
      tail -n +2 "$BACKLOG" > "$BACKLOG.tmp" && mv "$BACKLOG.tmp" "$BACKLOG"
      echo "  submitted $name"; sent=$((sent+1))
    else
      echo "  at the account submit cap -- $name stays parked, will retry when a slot frees"; break
    fi
  done
  echo "topped up $sent job(s); $(wc -l < "$BACKLOG" | tr -d ' ') still parked"
  ;;

run)
  name="${2:?name}"; wall="${3:?walltime}"; gpus="${4:?gpus}"; shift 4
  [ "${1:-}" = "--" ] && shift
  _assert_one_line "$*"
  cmd="$*"
  # PARTITION. Wired up while trying to escape a 33-hour queue, and MEASURED NOT TO HELP here.
  # Recorded rather than reverted, with the measurement, so the next person does not spend the
  # same hour rediscovering it:
  #
  #   batch   AllowQos=medium   -> only the -pm06 account. Fair-share 0.050 after 6.9M raw
  #                                usage; 32 cells estimated to start in 33 hours.
  #   preempt DenyQos=medium    -> -pm06 is refused outright. Of the two normal-QOS accounts
  #                                this user holds, marlowe-users is blocked by site policy
  #                                ("did you remember to set your account?") and
  #                                marlowe-m000159 is accepted but estimates a start a further
  #                                26 hours out -- LATER than batch, not sooner.
  #
  # The cluster is saturated (79 running, 113 pending), so there is no queue to game. The flags
  # stay because they cost nothing and the calculus changes when the cluster empties; a
  # preemption is a walltime kill by another name, which every lab here already survives via
  # per-round atomic adapter checkpoints.
  #
  # NOTE if these are ever used in anger: the partition is NOT recorded in the manifest, so
  # `continue` would resubmit a preempted cell back onto the default partition.
  # The ACCOUNT must change with the partition. mrl refuses the -pm06 account on `preempt`
  # because Marlowe then charges the allocation AND keeps the job preemptible -- you pay for
  # discarded work. The base account on preempt is uncharged, which is the combination mrl
  # itself suggests.
  _part=""
  if [ -n "${KA_PART:-}" ]; then
    _part="-p ${KA_PART} --requeue"
    [ -n "${KA_ACCT:-}" ] && _part="$_part -A ${KA_ACCT}"
  fi
  out="$($MRL submit -J "$name" -N 1 -G "$gpus" -t "$wall" $_part -- "$cmd" 2>&1)"
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
  # FAIL CLOSED on an unreachable cluster. `live` empty is ambiguous: it means either "no job
  # is queued" or "the query failed", and the two lead to opposite actions -- the first says
  # resubmit everything, the second says touch nothing. A transient made every remote call in
  # one invocation return empty, and `continue` printed "skip (no record yet)" for all 21
  # cells and exited 0, which reads as "nothing to do". `_queued` was hardened against exactly
  # this; `continue` was not.
  #
  # The probe is a sentinel the login node always answers. If it comes back wrong, the cluster
  # is not being read and no decision taken from these queries is worth acting on.
  if [ "$($MRL run "echo JOBMAN_OK" 2>/dev/null | tr -d '[:space:]')" != "JOBMAN_OK" ]; then
    echo "ABORT  cannot reach the cluster -- not resubmitting anything." >&2
    echo "       A failed query looks identical to an empty queue, and acting on it would" >&2
    echo "       either duplicate running work or silently continue nothing." >&2
    exit 2
  fi
  live="$($MRL run "module load slurm >/dev/null 2>&1; squeue -u \$USER -h -o %j 2>/dev/null" 2>/dev/null)"
  FP_NOW="$(_fingerprints)"
  n=0; blank=0; seen=0
  while IFS=$'\t' read -r name wall gpus cmd; do
    [ -n "$name" ] || continue
    seen=$((seen + 1))
    if grep -qx "$name" <<<"$live"; then continue; fi          # still queued/running
    st="$($MRL run "module load slurm >/dev/null 2>&1; sacct -u \$USER -n -X -o JobName%40,State -P -S now-3days 2>/dev/null | grep '^$name|' | tail -1" 2>/dev/null)"
    state="${st##*|}"
    case "$state" in
      TIMEOUT*)
        # Did the last chunk actually record anything? A cell whose fingerprint is unchanged
        # across two consecutive continues is not resuming, it is repeating -- hold it rather
        # than spend another walltime finding out again.
        cell="$(_cell_of "$cmd")"
        now_fp="$(awk -F'\t' -v c="$cell" '$1==c{print $2; exit}' <<<"$FP_NOW")"
        prev="$(awk -F'\t' -v n="$name" '$1==n{print $2"\t"$3; exit}' "$PROGRESS")"
        prev_fp="${prev%%$TAB*}"; stall="${prev#*$TAB}"
        [ "$stall" = "$prev" ] && stall=0
        if [ -n "$now_fp" ] && [ "$now_fp" = "$prev_fp" ]; then
          stall=$((stall + 1))
        else
          stall=0
        fi
        # The ledger is written AFTER the resubmission succeeds, further down. Writing it here
        # counted a submission the cap rejected as a round that produced nothing, and pushed a
        # healthy cell toward a HOLD it had not earned.
        if [ "$stall" -ge 2 ]; then
          echo "HOLD      $name  (2 consecutive chunks recorded nothing new: $now_fp)"
          echo "          its smallest resumable unit does not fit $wall -- raise the walltime"
          echo "          or make the lab checkpoint at a finer granularity. Not resubmitting."
          continue
        fi
        # Say what the NEXT chunk gets, not "the last chunk hit $wall". $wall is read from the
        # manifest and may have been raised since the dead job started, so the old phrasing
        # claimed a 90-minute job had hit a 3-hour wall. Diagnostic output that misstates what
        # happened is worse than none, and this file exists because of exactly that class of bug.
        echo "continue  $name  (timed out; resubmitting with $wall${now_fp:+, at $now_fp})"
        if "$0" run "$name" "$wall" "$gpus" -- "$cmd"; then
          # Only a cell that is actually back in the queue has had its chance to progress, so
          # only then does an unchanged fingerprint mean anything. A rejected submission leaves
          # the ledger untouched and the cell is retried on the next poll.
          grep -v "^$name	" "$PROGRESS" > "$PROGRESS.tmp" 2>/dev/null || true
          printf '%s\t%s\t%s\n' "$name" "${now_fp:-?}" "$stall" >> "$PROGRESS.tmp"
          mv "$PROGRESS.tmp" "$PROGRESS"
          n=$((n+1))
        else
          echo "          NOT resubmitted (submission refused); stall ledger left unchanged" >&2
        fi ;;
      COMPLETED*) : ;;                                          # done, nothing to do
      "")         blank=$((blank + 1)); echo "skip      $name  (no record yet)" ;;
      *)
        # A cell whose artifact says `complete` needs no look, whatever its exit state. t1k-q14
        # exits non-zero because its teacher bank is empty -- which IS the result (14B attempts
        # kernels and almost never verifies), not a failure to re-run.
        cell="$(_cell_of "$cmd")"
        fp="$(awk -F'\t' -v c="$cell" '$1==c{print $2; exit}' <<<"$FP_NOW")"
        if [ "$fp" = "complete" ]; then
          echo "done      $name  (artifact complete; exit state '$state' is not a failure)"
        elif cut -f1 "$BACKLOG" 2>/dev/null | grep -qx "$name"; then
          # Already parked for resubmission, which IS the look the HOLD is asking for. Without
          # this the same cell is reported on every watcher restart, forever, and a line printed
          # every thirty minutes about a decision already taken is how an operator learns to
          # skip the ones that matter. The cancelled-but-parked case is treated the same way in
          # watch_routes.sh for exactly this reason.
          : ;
        else
          echo "HOLD      $name  (last state '$state' -- not auto-continued; needs a look)"
        fi ;;
    esac
  done < "$MAN"
  # Every cell blank, with more than a couple considered, means sacct answered nothing at all
  # rather than that every cell is genuinely unrecorded. Say so instead of reporting a clean run.
  if [ "$seen" -gt 2 ] && [ "$blank" = "$seen" ]; then
    echo "ABORT  sacct returned no state for any of $seen cells -- the query is failing, not" >&2
    echo "       the manifest. Nothing was resubmitted; re-run once the cluster answers." >&2
    exit 2
  fi
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
