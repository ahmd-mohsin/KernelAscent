# KernelAscent — curated bundles (in progress)

Curated task bundles produced by the Fable 5 curator. All tasks here are in the
**public seed range** (the private held-out split, seed range 10,000,000+, is never
published). This snapshot is being extended as curation completes.

Each `<task>/` contains:
- `task.py` — the problem (`Model` + `get_inputs`, seeded)
- `meta.json` — tier, family, tags, shape/dtype/chain
- `cand_0.py … cand_k.py` — Fable 5 candidate solutions (optimized `ModelNew`)
- `DONE` — curation marker (candidate count)

Grading artifacts (`reference_solution.py`, `results.json`, difficulty labels) are added
by the GPU grader; see `SCHEMA.md` for the full bundle spec and `kernelascent/` for the
pipeline. Families: matmul, norm-act, attention, rope-attention, quant-gemm, moe.
