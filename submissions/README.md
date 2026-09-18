# Submitting to the KernelAscent leaderboards

The board is scored on a **private held-out split**, so results can't be gamed by
overfitting the released tasks. Submitting is automated through GitHub.

## Do I submit a model or something else? (the RSI question)

RSI is **not a weight state you can upload.** You cannot "fine-tune a model for RSI" — RSI
is a *protocol we run on a base model* and measure whether it compounds. So what you submit
depends on the track:

| Track | You submit | What we run |
|-------|-----------|-------------|
| `capability` | a **model** (open-weight HF id or API id) | one-shot held-out kernel eval → pass@k, speed |
| `weight_rsi` | a **base open-weight model** | our multi-round compounding loop → lineage−reset |
| `procedure_rsi` | a **model** (open or API) | it edits its own frozen-weight procedure → Q gain |
| `closed_open` | a **researcher model** | it rewrites an open trainee's training harness → improved−frozen |
| `selfplay` | a **model** | 3-arm self-play → L−F (author co-evolution) |
| `harness` | a **self-improvement recipe** (code, not a model) | we run *your recipe* on a fixed open trainee |

**The `harness` track is the one genuinely open research contribution**: propose a better way
to make a model self-improve (data selection, LoRA schedule, curriculum, search-amplified
expert iteration, diversity-preserving selection…). Our current result is that naive verified
self-training does **not** compound — a recipe that makes it compound is the headline result
this benchmark is built to surface.

## How to submit (automated)

1. Add one JSON file under `submissions/<track>/<your-name>.json`. Copy an example below and
   fill it in. The required fields per track are in [`schema.json`](schema.json).
2. Open a pull request. The **Validate submission** Action checks your JSON and comments back.
3. On merge, the bot rebuilds `docs/data/leaderboard_community.json` and the site shows your
   row, flagged **self-reported**.
4. A maintainer re-runs your model on the private held-out split (`verified-eval` workflow) and
   the row flips to **verified**.

Prefer not to write JSON? Open a [model-submission issue](../../../issues/new?template=model-submission.yml).

## Reproducibility manifest (`repro`)

Every submission must carry a `repro` block so anyone can re-run it:
`harness_commit` (git sha), `seeds`, `split` (`public` while self-reported), and the exact
`command`. Rows without a valid manifest are rejected by the validator.

## Validate locally before you PR

```bash
python3 scripts/validate_submission.py submissions/capability/your-model.json
```
