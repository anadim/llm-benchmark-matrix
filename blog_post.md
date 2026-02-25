# You Don't Need to Run Every Eval

I used Claude Code to build a benchmark prediction system, Codex to audit it for bugs, and Claude Sonnet to try to beat it for $1. Here's what I found: LLM evals are so low-rank (in fact rank 2) that most evals are approximately... redundant.

Let me give you an example for why one should expect this.

Here's a table. Four models from the DeepSeek-R1-Distill family, two math benchmarks.

|  | GPQA Diamond | AIME 2024 |
|---|---|---|
| R1-Distill-7B | 49 | 56 |
| R1-Distill-14B | **?** | 70 |
| R1-Distill-32B | 62 | 73 |
| DeepSeek-R1 (671B) | 72 | 80 |

What would you guess the missing entry is? Maybe around 60, 61? (It's 59 — off by a touch, but close enough to get suspicious.)

You probably noticed that the ratio of AIME to GPQA is kinda constant across the three rows where you have both numbers. The rows look like scaled versions of each other. You didn't have to compute anything, you kinda just saw it.

This anecdodtal observation isn't limited to one model family. If you look at reasoning models across totally different providers — o3, o4-mini, Gemini 2.5 Pro, DeepSeek-R1-0528 — that same ratio holds between 1.10 and 1.15. This latent structure shows up across models and evals. Very neat!

So in spite of the fact this is only a feeling and it's anecdotal. But every atom in my brain says there is real structure here, and I wanted to check. So let's see.

## Where this came from

When we train models with my group at MSR, we change something about a model/algorithm/data etc then look at evals, and I've noticed that even sparse early testing lets me build a mental model of performance. I don't have to see all of the AIME years to know how the model will do on HMMT. If I see a Terminal-Bench 2.0 score, I probably have a reasonable idea of how the model will perform on SWE-bench.

Your mileage may vary of course as there are cases that break this "implciit low dimensional structure", eg a model can saturate ARG-AGI and have 0 on MATH500, because it's a specialist model.

So, anywas. an old idea crept in: matrix completion (Hi Ben!), which I (and every other EE theory kid) was really into, maybe 15 years ago. The whole point of that theory is: observe a few entries of a matrix, and if you're lucky and the ground truth matrix is approximately low rank, you can recover the rest from surprisingly few measurements. 

There was even a famous $1M Netflix Prize for this. In 2006, Netflix released a matrix of about 100 million (user,ratings) pairs and offered a million dollars (so like 4 weeks of 1024 H100 on lambda)to whoever could predict the missing ratings 10% better than Netflix's own algorithm, or something. It ran for a few years, drew thousands of teams, and even thugh a team did get the 1M, Netflix never actually deployed the winning solution, lol. But, if am not wrong, the competition basically invented modern recommender systems and launched a few thousand ML (practice and theory!) careers.

Here's the idea. Suppose each model is secretly described by a vector of just two entries, and each benchmark is too. The score is their dot product. That's what "rank 2" means.
Say GPT-5 has vector s = (9, 6) and AIME 2025 has vector b = (8, 2). Then GPT-5's score on AIME 2025 is 9×8 + 6×2 = 84. If SWE-bench has vector b = (3, 7), then GPT-5 on SWE-bench is 9×3 + 6×7 = 69. Two vectors per model, two vectors per benchmark, and you can fill in every cell. The game is: observe a few real scores, solve for the hidden vectors, predict the rest.

So I had a feeling the LLM benchmark matrix had a low-rank structure too.[maybe will add a figure too here showing the top 10 singular values of a complete submatrix of the original one] This should be aesy to check ionce we have the matrix Though I was a bit skeptical, because classic matrix completion operates on much larger matrices — the Netflix one was half a million by eighteen thousand. Ours is 83 models by 49 evals. So maybe it won't work! But what's the cost of checking with Claude Code and Codex? Well, around $0.84 as we'll find out.

## The matrix

I asked Claude to search and verify all possible (model, eval) pairs it could find. It found 83 models across 49 benchmarks, every entry cited with a source URL. The matrix is 34% filled, meaning roughly 1,400 out of 4,000 cells have a ground truth number. The rest are missing because I couldn't find a reported score. 

The 83 models span 21 providers (OpenAI, DeepSeek, Anthropic, Google, Alibaba, Meta, xAI, Mistral, etc.), released between January 2025 and February 2026. The 49 benchmarks cover math, coding, agentic tasks, reasoning, knowledge, multimodal, instruction following, and a few others. All the usual suspects.

The question: does the low-rank structure hold at scale, and how many benchmark scores do you need before you can predict the rest for a given model?

Cute questions if you ask me.

## It mostly works, and the reason is the rank

The matrix turns out to be approximately rank 2. Two principal vectors explain 51% of the variance (think of this as: if you approximate the full 83×49 matrix using only two "directions," you already capture half of what's going on, and the other half is mostly noise or very specific quirks). There's a clean gap between the second component (14%) and the third (6%), and adding a third component actually hurts holdout prediction. Typical matrix completion bounds say you need on the order of r · n entries for recovery. For rank 2 and n = 83, that's a few hundred entries. We have 1,383. Even though the matrix is small, the fill rate relative to the rank is more than enough.

To make the accuracy concrete: if you take Gemini 2.5 Pro, which has scores on 35 benchmarks, hide its AIME 2024 score (actual: 92), and predict from the rest, the method predicts 87. Not perfect, but in the ballpark. Across all models and benchmarks, the median error is about **6.7%** of the true score, and 52% of predictions land within 5 points of the truth.

But getting there required one trick I didn't expect.

## The logit trick

Most benchmark scores live on a 0-100 scale, and many cluster near the ceiling. Going from 88% to 92% on MMLU is a bigger capability jump than going from 48% to 52%, but in raw space they look the same. The logit transform, logit(p) = log(p/(1-p)), stretches the extremes and compresses the middle, so regressions in logit space respect the true difficulty scale. This one change improved accuracy by 11%. It also handled bimodal benchmarks (ARC-AGI, IMO, USAMO) where models either score ~0 or ~25%, because in logit space the gap between those two values becomes enormous and the method naturally classifies rather than interpolates.

## The final method

The best method is embarrassingly simple:

- **"The Predictor"**: for each missing cell, find the 5 most correlated benchmarks (computed in logit space), fit a ridge regression, predict, transform back. This exploits local structure: "AIME 2024 predicts AIME 2025."
- **SVD rank-2 in logit space**: run iterative SVD on the logit-transformed matrix, keeping only rank 2. This exploits global structure: "this model's overall profile looks like GPT-4.1."
- **Average the two**: 60% Predictor + 40% SVD, wherever both have an answer. Where only SVD has one (about 20% of cells, where the Predictor can't find enough correlated benchmarks), use SVD alone.

No neural networks, no model metadata, no provider information. An earlier version used KNN instead of SVD as the second ingredient, but KNN and the Predictor are both local methods that make correlated mistakes. Swapping in SVD gave a 14% improvement because the errors became complementary.

## What didn't work

We tested 34 methods across 3 rounds of exploration. A lot didn't help:

- **Model metadata** (provider, parameter count, reasoning mode, open-weight status): neutral to negative. Every metadata-enriched method performed the same or worse.
- **Gradient boosted trees per benchmark**: overfits with fewer than 40 training rows.
- **Explicit bimodal handling**: 74% accuracy vs 90% from the logit transform doing it implicitly.
- **Missingness-pattern features**: treating the NaN pattern as a feature made things worse.
- **Meta-learner stacking**: second-level regressors overfit at this matrix size.
- **Within-family interpolation**: families too small (3-8 models) to be reliable.

The metadata finding is the one I keep coming back to. I genuinely expected parameter count and reasoning mode to help. They didn't. Every bit of information in "14B reasoning model from DeepSeek" is already captured by the model's pattern of benchmark scores.

## How many benchmarks do you need?

Here's the experiment: take a model, hide all its scores, then reveal them one at a time in random order. After each reveal, predict the rest. How quickly does the error come down?

The answer is: fast. With just 1 known score, median error is about 12%. By 5 known scores it drops to 9%. After that, diminishing returns — the precision of individual predictions keeps improving but the median stabilizes because a few hard-to-predict benchmarks (the bimodal ones) stay noisy regardless. The practical takeaway: **5 benchmarks are enough to locate a model in the rank-2 space.** Everything after that is refinement.

If you can only pick 5, greedy forward selection gives you: {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA}. They span four categories and cover both principal components. Under proper holdout they get about 7.8% error, versus 6.7% for the full method. Not magic, but a practical minimum eval set.

## Can Claude predict the scores without the algorithm?

OK, this next part isn't purely science because some of these models were released before Sonnet's training cutoff, so Claude might just "know" certain scores. But I was curious.

For $0.84, we gave Claude Sonnet the partially-filled matrix as a big CSV and asked it to predict the missing entries. Same holdout, same evaluation.

| Method | MedAPE | ±5 pts | Bimodal Acc | Cost |
|---|---:|---:|---:|---:|
| Claude Sonnet (full matrix) | **5.3%** | **62.3%** | 89.5% | $0.84 |
| LogitSVD (the algorithm) | 5.8% | 57.2% | 89.5% | ~$0 |
| Claude Sonnet (row-only) | 6.6% | 54.7% | 78.9% | $0.67 |
| BenchReg (no logit) | 6.5% | 53.3% | 94.4% | ~$0 |

Claude beats the algorithm on median error by half a point. But the interesting result is the **row-only ablation**: when you strip away the matrix and give Claude only the model's own scores plus its name, accuracy drops from 5.3% to 6.6%. That's a 23% degradation. The matrix context is doing real work: Claude is performing something like implicit matrix completion from the CSV, not just recognizing "DeepSeek-R1 is a reasoning model."

But world knowledge alone still gets you to 6.6%, roughly matching supervised regression. An LLM's training data already encodes the rank-2 structure of the benchmark space. The matrix lets it sharpen that prior.

The bimodal split tells the structural story. Claude predicts that GPT-5.1 and Claude Sonnet 4.5 should score high on MathArena Apex when they actually score low. It has an optimistic prior about frontier models on hard benchmarks that the data corrects. The logit transform doesn't have opinions about model names — it just respects the geometry of bounded scores, and that turns out to be exactly the right inductive bias for edge cases.

## The two principal components

OK so what are these two "directions" the matrix decomposes into?

The first component (37% of variance) is just general capability. The benchmarks that load on it most are GPQA Diamond, LiveCodeBench, and MMLU-Pro. If a model is strong here, it tends to be strong everywhere. This is the axis that separates GPT-5 from Phi-4-mini.

The second component (14%) is something like "frontier reasoning vs. established competence." Benchmarks like SimpleQA, ARC-AGI-2, and HLE load positively on it, while MATH-500 and MMLU load negatively. Models that score high on this component are the ones that do well on genuinely novel, hard tasks — not just the ones that ace established benchmarks. The strongest signal in this space is reasoning mode: the gap between reasoning and non-reasoning models is +1.8σ on HMMT and +1.75σ on HLE. GSM8K is the only benchmark where non-reasoning models win, because it's saturated and everyone scores 90%+.

Two vectors per model, two vectors per benchmark, dot product gives the score. That's the structure behind the table at the top.

## Where it breaks

Some benchmarks resist prediction entirely: ARC-AGI-2 and Terminal-Bench are the worst, with per-benchmark median errors above 60%. These are the interesting ones, because they measure something the rest of the matrix doesn't capture. ARC-AGI tests inductive reasoning that doesn't correlate with anything else. Terminal-Bench measures agentic ability, which is its own dimension — not a linear combination of reasoning and coding.

Competition math (IMO, USAMO) is bimodal: a model either solves olympiad problems or it doesn't. You can't interpolate a step function, but the logit trick classifies which side a model falls on 90% of the time.

If your model's predictions are way off on some benchmark, that benchmark is measuring something genuinely new. That's information. And the opposite is also information: if two benchmarks always predict each other perfectly, you probably don't need both.

## When should you use this?

Three options for getting a number on "how does my model do on benchmark X":

**Option A: Run the benchmark.** Cost: $50 to $10,000. You get the true score.

**Option B: Predict from the matrix.** Cost: ~$0. You get a score within about 5 points.

**Option C: Run a cheaper model as proxy.** Cost: less than A. But you get the right answer about the wrong model.

The decision is about error tolerance. If SWE-bench costs $5,000 and two engineer-days, and 6% error is fine for a go/no-go, just predict. If AIME 2026 costs $2, run it.

Option C is the sneaky one, because it feels rigorous — you ran a real eval — but the information about your actual model might be lower than the free prediction. At least the matrix prediction is about your model.

## Am I convinced?

For frontier labs, honestly not really. They'll run every eval regardless. But for smaller groups trying to train a new model, having a quick sanity check could be helpful? I don't know. But it doesn't matter, because this was fun to try.

The structure itself is the finding, and it's kind of hilarious that in pretty much any setting where you have an incomplete matrix to complete, it almost always turns out to be below rank 5. Matrix completion never dies!

Perhaps the deeper meaning is that most of what we call "evaluation" is measuring the same two things over and over, and the benchmarks that escape this pattern are the ones testing genuinely new capabilities. We've just convinced ourselves we need to climb 100 evals for every model.

I was curious about this for a while and now that we have agents, I tried it, because it took me a week. The full matrix, prediction code, and results are at [github.com/anadim/llm-benchmark-matrix](https://github.com/anadim/llm-benchmark-matrix).

[WORKFLOW DIAGRAM HERE]

---

*Joint work with Claude Code, Codex, and Claude Sonnet. The matrix was assembled from official model announcements, technical reports, leaderboards, and evaluation platforms. Every entry has a source URL. The code was written by Claude Code, audited by Codex, improved by both, and I mostly just gave prompts and argued with results. Priorities: accuracy over coverage, honesty over impressiveness.*
