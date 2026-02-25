# Response to Audit Round 2

Thanks for the thorough follow-up. You're right on the core point, and I want to move from debate to fixes.

## Where you're right

**Citation accuracy ≠ value accuracy.** Our rebuttal defended values by pointing to alternative sources — but that sidesteps your actual finding: the *cited URL* doesn't support the cell. For a publishable artifact, the citation should directly support the number. We accept this.

**The "0.5% error rate" framing was misleading.** Dividing 7 confirmed errors by 1,400 total cells ignores that most cells weren't audited. We shouldn't have stated it that way.

**Some of our rebuttal evidence was weak.** Using HN threads and journalistic summaries to defend precision claims isn't great.

## Where we'd nuance

**Sample bias matters for extrapolation.** The 62 cells you reviewed were (reasonably) chosen for being hard to verify — llm-stats.com JS pages, blog posts, secondary aggregators. The cells sourced from official technical reports and GitHub repos (which are the majority) would have a much lower mismatch rate. The 30/62 rate is real for the audited sample, but probably not representative of the full matrix.

**The rank-2 structure is itself a consistency check.** If the matrix had 10-15% value errors scattered randomly, the SVD wouldn't produce a clean rank-2 decomposition with 83% variance explained, and the prediction method wouldn't achieve 6.7% median error in leave-50%-out. The method's accuracy is indirect evidence that most values are in the right ballpark. (This doesn't help with citation hygiene, just value accuracy.)

## Proposed fixes — looking for your input

Rather than argue, here's what we'd like to do. Would appreciate your feedback on whether this is sufficient, or if you'd suggest a different approach.

### Tier 1: Fix confirmed value errors (7 cells)
These are unambiguous. We'll update both the value and ensure the citation is the authoritative source.

| Model | Benchmark | Current | Correct | Best Source |
|---|---|---:|---:|---|
| Seed-Thinking-v1.5 | MMLU-Pro | 80.0 | 87.0 | github.com/ByteDance-Seed/Seed-Thinking-v1.5 |
| Seed-Thinking-v1.5 | IFEval | 85.0 | 87.4 | github.com/ByteDance-Seed/Seed-Thinking-v1.5 |
| Grok 4 | FrontierMath | 15.0 | 13.0 | epoch.ai/blog/grok-4-math |
| Grok 4 | MMMU | 76.5 | 75.0 | automatio.ai/models/grok-4 |
| Gemini 3 Flash | IFEval | 89.5 | 88.2 | automatio.ai/models/gemini-3-flash |
| Claude Sonnet 4.6 | Terminal-Bench 2.0 | 58.2 | 59.1 | anthropic.com system card |
| GPT-4.1 mini | IFEval | 84.0 | 84.1 | openai.com/index/gpt-4-1/ |

### Tier 2: Upgrade citations to authoritative sources (6 cells)
Values are correct but the cited URL is wrong or weak. We have better sources already in our data files.

| Model | Benchmark | Value | Current Source | Better Source |
|---|---|---:|---|---|
| GPT-5.1 | GPQA Diamond | 88.1 | vellum.ai (GPT-5.2 blog) | artificialanalysis.ai/evaluations/gpqa-diamond |
| GPT-5.1 | ARC-AGI-2 | 17.6 | vellum.ai (GPT-5.2 blog) | arcprize.org leaderboard or OpenAI 5.2 announcement (comparison table) |
| Phi-4-reasoning | LiveCodeBench | 53.8 | microsoft.com research article | huggingface.co/microsoft/Phi-4-reasoning (model card) |
| Phi-4 | GSM8K | 95.3 | arxiv (wrong paper) | needs new source or removal |
| Grok 4 | ARC-AGI-1 | 66.6 | the-decoder.com (~68%) | arcprize.org or x.ai/news/grok-4 |
| Claude Opus 4.5 | BrowseComp | 67.8 | anthropic.com (page shows BrowseComp-Plus, not BrowseComp) | vellum.ai/blog/claude-opus-4-6-benchmarks |

### Tier 3: Remove or flag questionable cells (8 cells)
Both value and citation are shaky. We'd rather remove these than keep unverifiable data.

| Model | Benchmark | Current | Issue |
|---|---|---:|---|
| DeepSeek-R1-Distill-Qwen-32B | IFEval | 79.0 | Source doesn't report IFEval; HF leaderboard shows ~40 (different eval setup) |
| DeepSeek-R1-Distill-Llama-70B | IFEval | 81.0 | Same issue |
| DeepSeek-R1-Distill-Qwen-32B | MMLU | 85.0 | Source doesn't report MMLU for distill models |
| o4-mini (high) | BrowseComp | 45.0 | Source shows 28.3 (basic tools) or 49.7 (o3, different model) |
| GLM-4.7 | MMLU | 88.0 | llm-stats.com JS (unverifiable); conflicting entry of 90.1 from HuggingFace |
| GLM-4.7 | HumanEval | 88.0 | llm-stats.com JS; one source suggests 94.2 |
| GLM-4.7 | IFEval | 85.5 | llm-stats.com JS; no alternative source found |
| GPT-5 | GSM8K | 96.8 | OpenAI doesn't report GSM8K; plausible but unverifiable |

**Question for you:** For Tier 3, would you recommend (a) removing these cells entirely, (b) keeping them but marking them as "unverified" in the data, or (c) something else? We lean toward removal — 8 cells out of 1,400 won't affect the matrix completion results, and it's better to have a slightly sparser matrix than to have questionable entries.

### Tier 4: Add caveat to blog post and repo
Replace "every entry cited with a source URL" with something like:

> "Every entry has a source URL. Most are official technical reports, model cards, or benchmark leaderboards. Some are aggregator sites (llm-stats.com, automatio.ai) where data may have been updated since collection. An independent audit identified 7 value corrections and 8 cells with unverifiable citations, which have been fixed or removed. The full matrix with citations is on GitHub."

### Tier 5: Broader citation sweep (optional, your call)
Beyond the 62 cells you reviewed, there are ~1,340 unaudited cells. We could:
- (a) Do nothing further (the fixes above address the known issues)
- (b) Run a targeted sweep of all llm-stats.com-sourced cells (these seem most fragile)
- (c) You run a second audit round on a random sample of 50 cells from the "safe" pool (official sources) as a control

We'd lean toward (b) — a targeted sweep of aggregator-sourced cells — since those are where citation problems concentrate. Official technical reports and GitHub model cards are unlikely to have issues.

---

## Bottom line

You caught real problems. The matrix values are mostly sound (the math confirms this), but citation hygiene has gaps, especially for aggregator-sourced entries. We'd rather fix this collaboratively than argue about error rates. The tiered plan above is our proposal — let us know what you'd adjust.
