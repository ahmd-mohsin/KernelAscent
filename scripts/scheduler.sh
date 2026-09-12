#!/bin/bash
# Auto-launch queued experiments onto FREE GPUs across the fleet. Called each orchestrator cycle.
# queue file /tmp/ka/queue.txt, one job per line, pipe-delimited:
#   weight|MODEL|TAG|BANK|SEED|NGPU        -> krun_rerun.sh (NGPU=1 small, 2 mid, uses first NGPU free indices)
#   comb|TRAINEE|TAG|RESEARCHER|BANK|NGPU  -> krun_comb.sh  (NGPU=2: trainee+ctrl)
#   trackc|MODEL|TAG|MODE|-|1             -> krun_trackc.sh (1 grade gpu)
# A launched job is removed from the queue. No free capacity => no-op (safe when fleet is full).
QUEUE=/tmp/ka/queue.txt; KA=/tmp/ka
[ -s "$QUEUE" ] || exit 0
SSHOPT='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=15 -p 2222'
nodes_for(){ case "$1" in
  1051) echo "10.3.129.241 10.3.253.112 10.3.231.17";;
  1052) echo "10.3.171.65 10.3.90.181 10.3.193.178";;
  1053) echo "10.3.82.183 10.3.195.37 10.3.72.151";; esac; }
runner_for(){ # $1 port $2 ip ; main ip == first -> local box, else worker ssh
  local port=$1 ip=$2; local main=$(nodes_for $port|awk '{print $1}')
  if [ "$ip" = "$main" ]; then echo "$KA/box_${port}.sh"; else echo "$KA/box_${port}.sh env SSH_ASKPASS_REQUIRE=force SSH_ASKPASS=/tmp/ap.sh DISPLAY=:0 ssh $SSHOPT greenland-user@$ip"; fi
}
free_idx(){ $1 "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '\$2<1500{print \$1}'" 2>/dev/null | grep -v Warning | tr '\n' ' '; }

launched=0
for port in 1051 1052 1053; do
  read -r main w1 w2 <<< "$(nodes_for $port)"
  for ip in "$main" "$w1" "$w2"; do
    [ -s "$QUEUE" ] || break
    R=$(runner_for $port "$ip")
    FREE=($(free_idx "$R")); nfree=${#FREE[@]}
    # SAFE: only schedule onto a FULLY IDLE node (all 8 GPUs free). Never pack onto a node with running
    # jobs — free_idx is fooled by models still loading, which over-subscribes and OOMs live work.
    [ "$nfree" -lt 8 ] && continue
    # try to launch as many queue jobs as fit on this node's free GPUs
    while [ -s "$QUEUE" ]; do
      job=$(head -1 "$QUEUE"); IFS='|' read -r typ a b c d ng <<< "$job"
      ng=${ng:-1}; [ "$nfree" -lt "$ng" ] && break
      g0=${FREE[0]}; g1=${FREE[1]:-$g0}
      # weight ng=1 -> pack self+fresh+grade on ONE gpu (bf16 small fits); ng>=2 -> self g0, fresh g1
      wf=$g0; [ "$ng" -ge 2 ] && wf=$g1
      case "$typ" in
        weight) $R "bash \$HOME/ka/krun_rerun.sh '$a' '$b' '$c' $g0 $wf $g0 ${d:-0} 30 16" >/dev/null 2>&1;;
        comb)   $R "bash \$HOME/ka/krun_comb.sh '$a' '$b' $g0 $g1 '$c' '${d:-sota_run_mid.json}'" >/dev/null 2>&1;;
        trackc) $R "bash \$HOME/ka/krun_trackc.sh '$c' '$b' $g0 '$a'" >/dev/null 2>&1;;
      esac
      echo "$(date -u +%H:%MZ) launched $typ $b on $ip gpus $g0,$g1" >> /tmp/ka/scheduler.log
      sed -i '' '1d' "$QUEUE" 2>/dev/null || sed -i '1d' "$QUEUE"
      FREE=("${FREE[@]:$ng}"); nfree=$((nfree-ng)); launched=$((launched+1))
    done
  done
done
echo "$(date -u +%H:%MZ) scheduler launched=$launched queued_left=$(wc -l < $QUEUE 2>/dev/null||echo 0)" >> /tmp/ka/scheduler.log
