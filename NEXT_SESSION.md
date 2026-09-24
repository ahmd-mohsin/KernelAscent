# Pick up here — 2026-09-23 end of session

## One job outstanding

`r4-kernel-14b` (jobid 501682), queued, estimated start **2026-09-24 ~10:08**. It should run
overnight without intervention.

```bash
export PATH="$HOME/.marlowe/bin:$PATH"
bash scripts/jobman.sh status          # manifest vs live queue
mrl pull                               # fetch results into data/trajectories/
python3 scripts/route_readout.py r2    # prompt x scale read-out (decision rule is in the code)
```

**What it measures:** the fraction of Triton attempts that verify, at full scale (29 tasks,
k=6), against the small diagnostic's 1-in-14. The decision rule was fixed before the data
existed and lives in `route_readout.py`.

**Why it matters:** if the verified rate sits in a middle band, kernel authoring becomes the
substrate for the Tier-2 re-run — the first task in this project the models have not saturated.
If it is ~0 even with a working grader, the models genuinely cannot author kernels here, and
the benchmark needs an easier authoring target rather than a harder rewriting one.

## State

* Branch `p0-statistical-rigor`, 52 commits ahead of `main`, tree clean, `make audit` green.
* Paper builds at 21 pages (`cd paper && pdflatex kernelascent_full.tex`).
* `make audit` runs three layers: cross-artifact consistency, negative tests proving each gate
  fires, and an EDQUOT contamination sweep.

## The one thing to know before touching results

Triton kernels **could not verify** on this harness until today (host `CC` leaked into the
container; only Triton compiles C at runtime). Every artifact written before the fix is invalid
for any kernel claim. They are renamed `*.pre_ccfix.json` and `route_readout` refuses them by
name. New artifacts carry `_provenance` — use `provenance.is_valid_for_kernels(path)`, which
returns `None` for unstamped legacy files, meaning *suspect*, not valid.

## Decisions left for you

1. **Merge or rename the branch.** It outgrew its name — it now holds the instrument-validity
   work, two retractions and a scope correction, not just statistics.
2. **The paper's framing.** Its methodological contribution is strong; its empirical headline is
   narrower than this morning. Reframing around the instrument is my recommendation, but that is
   a call about what paper you want, not a technical one. See the end of `INTUITIONS.md` §2.6.
3. **Whether to re-run the A100 boards.** They were measured under a harness that silently
   rejected Triton. The site already says they are provisional.
