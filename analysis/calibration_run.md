# Tier calibration and failure analysis (open-weight test-takers)

Setup. 13 open-weight models run as test-takers across the tiered task set on a
3 node / 24 GPU fleet (one model per GPU), Easy/Medium/Hard/Ultra, n=30 per tier,
k=3 candidates per task, graded on GPU against the min(eager, torch.compile)
roofline. Models: Qwen2.5-Coder {0.5, 1.5, 3, 7, 14}B, Qwen2.5-Instruct {0.5, 1.5,
3, 7, 14}B, and cross-family DeepSeek-Coder-6.7B, StarCoder2-15B-Instruct,
CodeLlama-13B-Instruct. Plus a Fable 5.1 gold reference on Easy/Medium.

Two harness robustness fixes were required to complete grading. A native MLIR
compiler abort (SIGABRT) on a pathological candidate was killing the whole grade
process; the SIGALRM per-candidate timeout only catches Python-level hangs, not
native aborts. Grading now runs each task in its own short-lived subprocess, so a
crash only loses that one task (13 native crashes were survived on the gold set).
Separately, pkill -f grade_candidates was matching its own shell and silently
killing relaunches; fixed with a bracket pattern.

## Difficulty re-evaluation: the tiers separate correctly

Mean correctness% / beats-roofline%, averaged within model-size bands.

    band            Easy       Medium     Hard       Ultra
    small (<=3B)    46 / 23    38 / 13    36 / 4     31 / 3
    mid (6-8B)      80 / 28    67 / 12    54 / 6     38 / 6
    large (13-15B)  73 / 30    70 / 22    63 / 7     52 / 11

Correctness falls monotonically Easy to Ultra in every band and rises with model
size. The tier ladder is therefore well calibrated on the correctness axis. The
0.5B models sit below the floor everywhere (7 to 20 percent), which is an honest
out-of-scope finding rather than a tier bug. The Easy and Medium floor is real for
1.5B and up. Cross-family models land in the same bands as Qwen, so the
calibration is not Qwen specific.

The speed axis (beats-roofline) is low everywhere and also falls Easy to Ultra
(down to 3 to 11 percent on Hard and Ultra). This is the honest hard bar and it is
the same wall the frontier models hit in the scaffold-RSI run.

## Why the models fail (4,016 graded candidates, all 13 models)

    correct                                29.8%
    compile error (Triton / inductor)      23.2%
    runtime error                          10.4%
    wrong API (hallucinated tl.* etc.)      9.2%
    wrong output                            7.1%
    type / name / syntax errors            10.4%
    other                                   9.9%

About 70 percent of candidates never produce a correct kernel, and the single
biggest cause is compilation failure. The models write Triton that looks plausible
but does not compile: bad tl.* calls, wrong grid/block/stride configuration, type
mismatches, plus about 9 percent that invent APIs outright. This is the
correctness wall and it dominates below roughly 14B.

## Why self-improvement does not compound on this benchmark

1. The bottleneck is generation skill, not knowledge. With 23 percent compile
   failures and 9 percent hallucinated APIs, the limiter is writing compilable
   Triton. Scaffold-RSI gives the model better textual advice, but advice does not
   fix an inability to emit valid code, so the compile rate does not climb across
   rounds. This is why 17 of 26 models never opened the channel in run 1.

2. The improvement signal is extremely sparse. Even correct kernels beat the
   torch.compile roofline only 3 to 30 percent of the time, and just 3 to 11
   percent on Hard and Ultra. For weight-RSI (GRPO) that is a near zero reward
   gradient on the speed objective, so the policy learns to be valid long before
   it learns to be fast, and on the hard tiers there is almost no reward to climb.
   For scaffold-RSI the strategies that would actually win (expert tiling, warp
   specialization, autotuning) are beyond what the model can execute even when told.

3. The two walls squeeze the improvable band to almost nothing. Where correctness
   is achievable (Easy) the speed is nearly unwinnable because torch.compile
   already saturates memory-bound elementwise ops (Fable 5.1 gold on Easy has a
   median 0.93x and only 32 percent beat the roofline). Where speed is winnable
   (Medium matmul, 61 percent for the gold curator) correctness is already hard for
   the models. The measurable-RSI sweet spot is narrow, roughly Medium tier at 7B
   and up, and everywhere else the loop has no room to compound.

## Fable 5.1 gold reference (Easy/Medium, strong-curator view)

    tier    n    correct   beats-roofline   median speedup (passing)
    Easy    60   98%       32%              0.93x
    Medium  61   100%      61%              1.07x

Easy is easy for correctness but hard for speed (memory bound, torch.compile
already near optimal). Medium is more beatable because matmul with a fused epilogue
offers real fusion wins. The two walls are visible in the dataset itself.

Data source: /tmp/instance_storage/ka_data/nodecal, nodecal_xfam, fable_em on the
3 node fleet; per-node analyze_calib.py rollups merged off box.
