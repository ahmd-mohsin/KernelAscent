#!/bin/bash
# Run ON the main node. Distributes node_calib.sh to the two worker nodes, then
# launches the per-node calibration on all three nodes (detached). Clean single
# ssh hop per worker, script files (not inline command strings), so nothing to
# mangle. Poll each node's local NODE_DONE, then scp summaries back to main.
set -u
CODE=/tmp/instance_storage/kernelascent
NC=$CODE/node_calib.sh
DATA=/tmp/instance_storage/ka_data/nodecal
T="Easy,Medium,Hard,Ultra"; N=30; K=3
W1=10.3.145.192; W2=10.3.84.11
mkdir -p "$DATA"

RJ(){ env SSH_ASKPASS_REQUIRE=force SSH_ASKPASS=/tmp/instance_storage/ap.sh DISPLAY=:0 \
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 \
  -o PreferredAuthentications=password -o PubkeyAuthentication=no -p 2222 greenland-user@"$1" "$2" < /dev/null; }

# push node_calib.sh to workers
for w in $W1 $W2; do RJ "$w" "mkdir -p $CODE $DATA"; cat "$NC" | RJ "$w" "cat > $NC"; done
echo "distributed node_calib.sh"

# main: Coder 0.5/1.5/3/7 on gpu 0-3
setsid bash "$NC" "$DATA" "$T" "$N" "$K" \
  Qwen/Qwen2.5-Coder-0.5B-Instruct:0 Qwen/Qwen2.5-Coder-1.5B-Instruct:1 \
  Qwen/Qwen2.5-Coder-3B-Instruct:2 Qwen/Qwen2.5-Coder-7B-Instruct:3 \
  > "$DATA/node_main.log" 2>&1 < /dev/null &
echo "launched main"

# worker1: Coder-14 + Instruct 0.5/1.5/3 on gpu 0-3
RJ "$W1" "cd $CODE && setsid bash $NC $DATA $T $N $K Qwen/Qwen2.5-Coder-14B-Instruct:0 Qwen/Qwen2.5-0.5B-Instruct:1 Qwen/Qwen2.5-1.5B-Instruct:2 Qwen/Qwen2.5-3B-Instruct:3 > $DATA/node_w1.log 2>&1 < /dev/null &"
echo "launched worker1"

# worker2: Instruct 7/14 on gpu 0-1
RJ "$W2" "cd $CODE && setsid bash $NC $DATA $T $N $K Qwen/Qwen2.5-7B-Instruct:0 Qwen/Qwen2.5-14B-Instruct:1 > $DATA/node_w2.log 2>&1 < /dev/null &"
echo "launched worker2"

echo "all launched; poll with: RJ <node> 'ls $DATA/NODE_DONE'"
