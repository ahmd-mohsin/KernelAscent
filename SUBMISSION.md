# Submitting to the KernelAscent leaderboards

The public leaderboard is scored on a **private held-out split** so results can't be
gamed by overfitting the released tasks. There are two ways to submit.

## Option A — you run generation, we grade (recommended)

1. Install and generate on the **public** split (no GPU needed for API models):
   ```bash
   kernelascent gen --model <MODEL> --tiers L1,L2 --split public --out runs/<model>
   ```
2. Open a **pull request** adding `runs/<model>/` (candidate kernels + trajectories),
   **or** open a *Model submission* issue with your model id and setup.
3. Maintainers re-run your model on the **held-out** split and grade on GPU, then add the
   verified row to `docs/data/leaderboard.json`.

## Option B — you run the full eval and self-report

If you have a GPU:
```bash
kernelascent eval --model <MODEL> --tiers L1,L2 --out runs/<model>
```
Submit the printed metrics + `runs/<model>/summary.json` via PR. Maintainers spot-check
on the held-out split before the row is marked verified.

## What we record per model

`model, org, params, pass@k, fast_1, fast_1.5, fast_2, geomean(pass)`, the track
(Capability / Scaffold-RSI / Weight-RSI), and tiers evaluated. Reasoning is enabled for
reasoning-capable models. Rows verified on the held-out split are flagged `verified`.

## Tracks

- **Capability** — any model, API or open. Single-shot / best-of-k.
- **Scaffold-RSI** — the agent improves its own scaffold across rounds (frozen weights).
- **Weight-RSI** — open-weight only; the training loop (compounding coefficient + Δ_k control).

Questions or a model you want added? Open an issue with the `submission` template.
