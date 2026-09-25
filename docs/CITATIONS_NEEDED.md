# Citations — what the paper needs, and what the repo already has

State as of 2026-09-24. `paper/` contains **zero `\cite` commands**, **no `.bib` file**, and no
`\bibliography` or `\addbibresource`. This is the last hard blocker on submission and the one
item on the must-add list with no data behind it.

**I have not invented any bibliographic entry in this file.** Where the repo holds a real
reference I name it. Where it does not, the row says so and stays empty. A fabricated citation
in a submitted paper is unrecoverable, and it is the one error here that no gate would catch.

## Already in the repo, not wired into the paper

`KernelAscent_RSI_v2_Redesign_Specification.md` carries **16 attributed references (R1–R16)**
with authors, venues and dates. They cover the RSI and benchmark landscape and need converting
to BibTeX and citing. R1 (KernelBench, Ouyang et al., ICML 2025) is the direct comparison the
paper owes a reader, and it is currently cited nowhere in `paper/`.

## Concepts used in the paper with no reference attached

Counted by word-boundary match across `paper/*.tex`, so these are real uses rather than
substring collisions.

| concept | sites | first occurrence | what is needed |
|---|---|---|---|
| self-play | 18 | `figures.tex:34` | the self-play / curriculum line this design descends from |
| roofline | 15 | `instrument_validity.tex:71` | the roofline model, for the per-task ceiling |
| LoRA | 14 | `figures.tex:79` | the LoRA paper, for the adapter method |
| Triton | 13 | `instrument_validity.tex:30` | the Triton language, for the kernel target |
| `torch.compile` | 11 | `instrument_validity.tex:60` | PyTorch 2 / TorchInductor, for the baseline |
| rejection sampling | 7 | `discussion.tex:79` | rejection-sampling fine-tuning, the training method |
| TOST | 7 | `discussion.tex:80` | two one-sided tests, for every equivalence claim |
| STaR | 1 | `discussion.tex:19` | the self-taught-reasoner line, already named in prose |
| Wilson interval | 1 | `04_rungs.tex` (T1 coverage intervals) | Wilson score interval |
| Bayes factor | 1 | `05_synthesis.tex` (the BF01 convention) | the BF₀₁ convention and its interpretation thresholds |

## Why the statistical rows matter more than they look

TOST, Wilson and Bayes factors are not background citations here. They are the machinery the
central claims rest on. The headline null is an equivalence result at δ=0.05 reported with
BF₀₁ = 5.6, and a reader cannot check whether the interpretation thresholds are the conventional
ones without the reference. An uncited BF₀₁ is a number the reader has to take on trust, which
is the opposite of what the rest of this paper argues for.

## Suggested order

1. Convert R1–R16 from the spec into `paper/refs.bib` and cite them. No new research needed.
2. Add the statistical references. Small, and they carry load.
3. Add the method references (LoRA, Triton, `torch.compile`, roofline, rejection sampling).
4. Write related work around the KernelBench comparison, which the paper promises and omits.
