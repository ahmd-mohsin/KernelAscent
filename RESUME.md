# RESUME — live fleet state (update each session)

## Fleet (3 SDB jobs, us-east-2, greenland profile; 9 nodes x 8 A100-40GB = 72 GPUs)
Reconnect: tunnel keeper `/tmp/ka/tunnel_keeper.sh` (mi- -> local 1051/1052/1053), or manually
`aws ssm start-session --target <mi> --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["2222"],"localPortNumber":["<port>"]}' --profile greenland --region us-east-2`.
- job1 mi-098e92901bc27319c -> 1051  main 10.3.152.163  workers 10.3.240.86, 10.3.154.127
- job2 mi-0bca9675a5c28c496 -> 1052  main 10.3.117.10   workers 10.3.174.241, 10.3.229.177
- job3 mi-00e44a647515c671c -> 1053  main 10.3.84.154   workers 10.3.250.77, 10.3.243.39
Node SSH via box_<port>.sh; workers via main-SSH (askpass /tmp/ap.sh, empty pw, port 2222).
Deploy: $HOME/ka (code+banks), $HOME/ka/ka_data (KA_DATA_DIR). Base image lacks transformers -> pip transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 (+sentencepiece for Yi/OpenCoder). /tmp/instance_storage is root-only.
Bedrock: creds -> $HOME/.aws/credentials [bedrock]; AWS_PROFILE=bedrock region us-east-2; models fable/nova-pro/llama3-3-70b.

## Definitive re-run (P1) — IN PROGRESS
Banks: dataset/kernel_bank/sota_run_{small,mid,large}.json (~90/scale, roofline-gated headroom>=1.3).
Launcher: krun_rerun.sh (bf16, headroom score, single-GPU arms). Multi-seed (0,1,2).
- small tier seed0: node C (154.127) — qwen1.5/deepseek1.3/opencoder1.5/starcoder3b/qwen0.5 (running).
- TODO: small seeds 1,2; mid tier (bf16 1-2 GPU); large tier (bf16 single-GPU per P2 fix); all x3 seeds.
Collect: rerun_<tag>_s<seed>/weight_rsi.json -> aggregate via ci_analysis.py -> docs/data/rigor.json.

## Done (results/raw/ + docs/data/ + HF leaderboards/)
T2 small/mid weight-RSI (old 1.5x score), T4 Fable/Nova/Llama->Qwen1.5 + Fable->7B, T3 Fable/Nova/Llama trackc,
baselines Qwen1.5/DeepSeek1.3, roofline board, 1420 curated SOTA tasks, rigor cross-seed board.

## Blockers: see ROADMAP.md "PAPER BLOCKERS" P1-P7.
