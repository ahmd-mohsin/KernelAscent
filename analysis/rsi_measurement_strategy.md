# Measuring true RSI without long weight training

This is a strategy proposal, not yet executed. It answers one question. Our benchmark
is meant to elicit recursive self-improvement, but the runs so far do not show the
mechanism. Why, and what is the cheapest experiment that would actually reveal true
RSI if it exists.

## 1. Why nothing so far measured RSI

Two separate reasons, and only the second is about the benchmark.

The sweep loop (openweight_rsi.py) is not an RSI mechanism. It is frozen weights,
single task, replace the answer. The model never changes and nothing carries from one
task to the next, so capability cannot compound by construction. Its round over round
decline is a prompt dynamics artifact (told to go faster, the model discards a working
kernel and gambles), not evidence about RSI. It was useful as a probe but it is the
wrong instrument for the RSI question.

The reward is a cliff, not a slope. The improvable signal is speed over torch.compile.
For most tasks there is no smooth path from a correct but slow kernel to one that beats
the compiler. The safe correct answer (call torch) already sits near the roofline, and
the only way past the bar is an expert hand written kernel, which is a discontinuous
jump in skill. RSI needs a gradient of small improvable steps. Kernel speed over a
strong compiler is closer to a step function, so a frozen model has nothing to climb.

True RSI therefore needs two things we did not provide together. A persistence channel
so improvement is stored and reused, and a reward with a climbable slope so there is
somewhere to go.

## 2. What true RSI is, operationally

We should only claim RSI when all of the following hold on a held out task set the
agent never optimized.

Persistence. The improvement lives in an artifact that survives across tasks, either
model weights or an external memory, not just the context of a single task.

Transfer. Performance rises on held out tasks (new problems, new seeds), not only on
the tasks the agent practiced on. This rules out memorization and per task search.

Compounding. The transfer gain grows across rounds rather than saturating after round
one, and it grows faster than a matched compute control that spends the same number of
generations without updating memory.

Attribution. The gain must beat three controls. A frozen memory control, a matched
compute control, and a shuffled or adversarial memory control. If a run beats all three
and transfers to held out tasks, it is RSI. Otherwise it is search, extra compute, or
contamination.

## 3. Primary proposal, memory scaffold RSI with transfer

Frozen weights. Works for both API models (via Bedrock, using the new creds) and open
weight models (the Qwen ladder on the fleet). No training, so it is cheap and fast to
verify, which is exactly the property the weight track lacks.

The agent holds an external, editable memory that it grows from graded feedback, then
we test whether that memory raises performance on disjoint held out tasks. Two kinds of
memory, run as separate arms because they test different claims.

Knowledge memory. A library of text strategies the agent writes for itself and reuses.
This is what run 1 used. It got poisoned for weak models and did not help frontier
models because their limiter was execution not knowledge. We keep it as a baseline arm,
now with the sanitizer and API grounding filter already shipped.

Skill and code memory (the strongest lever). The agent accumulates a library of
correct, reusable kernel building blocks, tiled matmul epilogues, a fused softmax, a
flash attention block, a rmsnorm kernel, each one a concrete verified artifact it
earned by solving a practice task. On a new task the agent retrieves and composes these
blocks rather than writing from scratch. This directly attacks the correctness wall,
because reuse of a known good block is far more reliable than regenerating Triton, and
it is a genuine compounding mechanism, since every solved task can add a block that
makes later tasks easier. This is the most promising path to true RSI without training.

Loop per round k. Practice phase, the agent optimizes a set of practice tasks with its
current memory, gets graded feedback, and reflects to add or edit memory entries or
banked code blocks. Transfer phase, the agent optimizes a disjoint held out set using
the frozen current memory with no further edits, and that transfer score is C_k. RSI is
C_k rising across rounds and the gap over the frozen memory control growing.

## 4. Turn the cliff into a slope

Without this the agent has nowhere to climb, so RSI cannot appear even if the mechanism
works.

Baseline ladder with partial credit. Score speed against eager, then torch.compile,
then a best known expert reference. Give graded credit for beating each rung rather than
a single pass or fail at the compiler bar. Now a kernel that goes from 0.8x to 0.95x to
1.1x earns rising reward, which is a slope.

Keep best. The agent keeps its best correct kernel per task and is never forced to
regress. Told to go faster it proposes an alternative that is accepted only if it stays
correct and is faster. This separates search, best of attempts, from learning, transfer
to held out tasks, and it removes the destructive dynamic we observed.

Rich feedback. Return a timing breakdown and the roofline gap, the compile or runtime
error text, and the closest matching past win retrieved from memory, not a one line
message. A model cannot repair or improve from a signal it cannot read.

Difficulty band. Run RSI where the improvable signal is non zero, which the calibration
says is the Medium band and models of 7B and up. Easy is a speed dead end because torch
already saturates it, and Ultra is a correctness dead end below frontier.

## 5. Other ways to get persistence without full GRPO

Answering the question directly, since long weight training is hard to verify.

Memory scaffold RSI, section 3, is the recommended primary. It has a real persistence
channel and needs no training, and a single round is minutes not hours.

LoRA from wins, an optional middle ground. Instead of full GRPO, periodically fine tune
a small LoRA adapter on the agent's own verified winning solutions, supervised, then
measure transfer. This is a real weight channel but is hours cheaper than online RL and
is easy to checkpoint and evaluate. It sits between the memory arm and full GRPO.

Self curriculum, a stretch arm. The agent proposes its own harder tasks plus reference
solutions, and we measure whether training or memory built from self generated problems
transfers to the human held out set. This tests self directed improvement, the strongest
form of the claim, but it is the noisiest to run.

## 6. Metrics and takeoff shape

Report on the held out transfer set per round. correctness_rate and speed_rate as the
two walls, and a laddered speed score that gives partial credit across the baseline
rungs. The RSI headline is the shape of C_k over rounds, the compounding coefficient b
from a fit, and Delta_k which is C_k self minus C_k control, with its trend. Plateau,
compounds, or degrades, judged against the controls in section 2.

## 7. Concrete run plan to propose

Models. Open weight, Qwen2.5-Coder 7B and 14B on the fleet, plus one strong API model
via Bedrock, for example Fable or Opus, as an upper reference for what a capable agent
does with the same memory mechanism.

Tasks. Procedural, Medium band, public seeds for practice and a disjoint private seed
range for transfer, so transfer cannot be memorized. Practice 24 tasks, transfer 24
tasks, held fixed across rounds.

Rounds. 5 to 6. Arms per model, self memory, frozen memory control, matched compute
control, shuffled memory control. Both memory types, knowledge and skill code.

Budget. Each round is generation plus crash isolated grading, minutes per model, so a
full multi arm run for two open weight models plus one API model is a few hours on the
fleet, not the days a weight track needs.

## 8. What would count as success or failure

RSI shown. On the held out transfer set, self memory C_k rises across rounds, b is
positive, and Delta over all three controls grows. The skill code arm is the most likely
to show this, because reuse of verified blocks compounds.

RSI not shown, and still a strong result. If even the memory and slope fixes give flat
or negative Delta, the honest finding is that current models cannot bootstrap kernel
optimization capability from their own outputs, and the benchmark cleanly demonstrates
the ceiling and localizes it to the correctness wall. Either outcome is publishable.

## 9. Open decisions for you

Which memory arm to lead with, knowledge, skill code, or both. Whether to include the
LoRA from wins middle ground or keep strictly to no training for now. Which API model to
use as the upper reference. Practice and transfer set sizes and round count given the
box eviction window. Once you pick, I will implement the loop, the baseline ladder
reward, and the control arms, and run it one model at a time on the fleet.
