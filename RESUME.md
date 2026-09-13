# RESUME — live fleet state (2026-09-13 redeploy #2, TRUE-RSI matrix)

## Fleet (3 SDB jobs, us-east-2 greenland p4d.24xlarge; 9 nodes x 8 A100-40GB = 72 GPUs). Tunnels mi->local 1051/1052/1053.
- job1 mi-0df997cccbd84d5c6 -> 1051  main 10.3.82.107  workers 10.3.224.173 10.3.194.25
- job2 mi-0fd332351fead959f -> 1052  main 10.3.132.252 workers 10.3.97.49   10.3.34.185
- job3 mi-00dba13f33b64d373 -> 1053  main 10.3.128.80  workers 10.3.89.203  10.3.183.179
HF token muahmed7338 -> $HOME/.cache/huggingface/token (unlocks gated: starcoder2-15b etc). 40GB/GPU: 14/15B tight on 1 GPU (self-play large arms hit 37GB — OK but risky); 32B & mech_large use 2 GPU/arm via KA_MAXMEM_GIB=18-20.
WORKER LAUNCH GOTCHA: launch_matrix go()/setsid + double-hop-stdin unreliable on workers. RELIABLE: stage script on main (box 'cat>/tmp/x' < f), then main pushes+runs (box 'cat /tmp/x | ssh worker "cat>/tmp/x && bash /tmp/x"'). See /tmp/ka/mlw.sh pattern.
launch_matrix ROLE for tiny probes is `mech` NOT `rmech` (tags are rmech_*).
Reconnect: nohup bash /tmp/ka/tunnel_keeper.sh (NO setsid on macOS). box_1051.sh/box_1052.sh; workers via /tmp/ka/worker_deploy_and_run.sh <wip> <ROLE> <seed> (run FROM a main node, empty-pw SSH). No `timeout` on macOS.
Deploy: HF_HOME=$HOME/hf (/tmp/instance_storage is root-only!), git clone github.com/ahmd-mohsin/KernelAscent -> $HOME/ka, pip transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 sentencepiece. KA_DATA_DIR=$HOME/ka/ka_data. Jobs die ~24h (started 07:1x Z 2026-09-13 -> ~07Z 2026-09-14).

## LIVE ALLOCATION (redeploy #2) — 6 nodes launched via scripts/launch_matrix.sh <ROLE>:
- job1 main: selfplay_small s0 (deepseek1.3, qwen1.5 — 3-arm)   | job1 w1: selfplay_mid s0 (qwen7b, deepseek6.7b)
- job1 w2: rmech s0 (8 WHY-RSI probes)                          | job2 main: selfplay_large s0 (qwen14b, starcoder15b)
- job2 w1: mech_large s0 (4 bigger WHY-RSI)                     | job2 w2: selfplay_xlarge s0 (qwen32b 3-arm, 6 GPU)
- PENDING: job3 (need mi) -> selfplay_{small,mid} s1 + closed. CLOSED self-play needs $HOME/.aws/credentials [bedrock] (creds write was classifier-blocked; place manually). NOTE self-play held ladder = LK.TASKS (~13 held after 16 seed).

## DEFINITIVE RE-RUN in flight (n_train=16, bf16, headroom score, SOTA run banks ~90/scale)
- SMALL x3 seeds (job1): main s0 / w253.112 s1 / w231.17 s2 — qwen1.5,deepseek1.3,opencoder1.5,starcoder3b,qwen0.5 (1 GPU each)
- MID x3 seeds (job2): main s0 / w90.181 s1 / w193.178 s2 — qwen7b,deepseek6.7b,starcoder7b,qwen3b (2 GPU each)
- LARGE s0,s1 (job3 main): qwen14b,starcoder15b (bf16 single-GPU per arm; P2 fix for device_map SFT crash)
- FRONTIER T4 (job3 w195.37): GPT-5.6/Kimi-k2.5/Opus-5/DeepSeek-v3.2 -> Qwen1.5 (rerun_comb_*)
- T3 procedure-RSI (job3 w72.151): Fable/GPT-5.6/Kimi/Opus-5/DeepSeek self-modify (rerun_trackc_*)
Collect: rerun_*/weight_rsi.json + rerun_comb_*/combined_rsi.json + rerun_trackc_*/track_c.json -> ci_analysis.py.
Launchers on nodes: krun_rerun.sh (weight), krun_comb.sh (frontier T4), krun_trackc.sh (T3).

## Frontier model IDs (Bedrock, us. prefix): us.openai.gpt-5.6-sol, moonshotai.kimi-k2.5, us.anthropic.claude-opus-5, deepseek.v3.2. See memory kernelascent-bedrock-frontier.
## Blockers: ROADMAP.md PAPER BLOCKERS P1-P7. P1 re-run in flight; P5 frontier now running (minus Gemini).

## ONE-COMMAND LAUNCH (new nodes): scripts/launch_matrix.sh <ROLE> [SEED]  — packs one 8xA100 node, detached, writes $HOME/ka/ka_data/<tag>/
Roles: rmech (8 small/mid WHY-RSI probes) | mech_large (4 bigger, 2 GPU ea) | selfplay_small | selfplay_mid | selfplay_large | selfplay_xlarge (32B) | closed (5 API models).
Suggested 9-node fleet: node1 rmech s0, node2 mech_large s0, node3 selfplay_small s0, node4 selfplay_mid s0, node5 selfplay_large s0, node6 selfplay_xlarge s0, node7 closed s0, node8 selfplay_small s1, node9 selfplay_mid s1.

## Task 5a OPEN self-play — 3-ARM (STATIC / FROZEN-AUTHOR / LIVE-AUTHOR). PRIMARY = L-F (author CO-EVOLUTION).
  python -m kernelascent.v3.lab_selfplay_rsi --model <hf> --s-gpu <g> --f-gpu <g2> --l-gpu <g3> --rounds 12 --seed-tasks 16 --held 30 --propose 6
Only the AUTHOR role differs S/F/L; solver learning identical. L-S=total curriculum, F-S=curriculum w/o author update, L-F=self-referential signal. Anti-hack: reject constant/identity/no-op tasks (_meaningful), dedup, attribution logged. bf16, no creds.

## Task 5b CLOSED self-play (API, no weights) — 3-ARM, co-evolution channel = PROCEDURE. lab_selfplay_closed.py
  python -m kernelascent.v3.lab_selfplay_closed --model <api-id> --region us-east-2 --rounds 6 --grade-gpu <g>  (creds needed)
FROZEN-AUTHOR = author proposes with EMPTY procedure; LIVE-AUTHOR = proposes conditioned on co-evolved strategies. PRIMARY=L-F.
Models: us.openai.gpt-5.6-sol, us.anthropic.claude-opus-5, moonshotai.kimi-k2.5, deepseek.v3.2, us.anthropic.claude-fable-5-1.

## WHY-RSI mechanism probe (fails/passes attribution): lab_rsi_mechanism.py
  python -m kernelascent.v3.lab_rsi_mechanism --model <hf> --gpu <g> --rounds 8   (bf16, 1-2 GPU)
Per round: gen_distinct2/gen_dissim (mode collapse), gen_entropy, drift_total + drift_early/mid/late (LoRA change per depth; ->0=ceiling), retention (forgetting), transfer_gap (memorization). VERDICT attributes outcome to {diversity_collapse|forgetting|drift_saturation|no_transfer|no_headroom} or PASS+carrying_depth.
