# RESUME — live fleet state (2026-09-12 redeploy)

## Fleet (3 SDB jobs, us-east-2 greenland; 9 nodes x 8 A100 = 72 GPUs). Tunnels mi->local 1051/1052/1053.
- job1 mi-03c40f7e11b1de8ca -> 1051  main 10.3.129.241  workers 10.3.253.112 10.3.231.17
- job2 mi-07c75540d39ab5616 -> 1052  main 10.3.171.65   workers 10.3.90.181  10.3.193.178
- job3 mi-08e22ccf6a0af3bb0 -> 1053  main 10.3.82.183    workers 10.3.195.37  10.3.72.151
Reconnect: /tmp/ka/tunnel_keeper.sh (MI map updated). box_<port>.sh; workers via main-SSH, askpass /tmp/ap.sh.
Deploy $HOME/ka (git pkg) + $HOME/ka/ka_data. pip transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 sentencepiece.
Bedrock creds -> $HOME/.aws/credentials [bedrock]; AWS_PROFILE=bedrock us-east-2. Jobs die at ~24h wall.

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
