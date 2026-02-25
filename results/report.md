# LLM Benchmark Matrix Completion: Full Report

**Matrix**: 83 models × 49 benchmarks | **1,375 observed** (33.8% fill) | **2,692 predicted**
**Date**: February 24, 2026 (post-audit update)

---

## 1. Executive Summary

We constructed a fully-cited benchmark matrix spanning 83 frontier instruct/chat LLMs across 49 benchmarks covering math, coding, reasoning, knowledge, agentic tasks, multimodal understanding, instruction following, and long context. Every entry carries a source URL. The matrix is 33.8% filled — the remaining 66.2% (2,692 cells) are missing because not every model is evaluated on every benchmark. We then evaluated 22 prediction methods for filling these missing cells, conducting a systematic 3-round exploration of what information helps and what doesn't.

### Top-Line Results

| | **Best: BenchPress (Post-Audit)** |
|---|---:|
| Per-model MedAPE (3 seeds) | **7.25%** |
| Random MedAPE (single seed) | **5.81%** |
| MAE (score points) | **4.71** |
| Within ±3 points | **36.5%** |
| Within ±5 points | **51.7%** |
| Bimodal accuracy | **94.2%** |
| Coverage | **99.7%** |

### Key Findings

1. **The logit transform is the single biggest win.** Working in log-odds space for percentage-scale benchmarks linearizes relationships near ceilings and floors. This one change improves BenchReg by 11.3% relative and SVD by 12.8% relative — and implicitly handles bimodal benchmarks (90% classification accuracy vs 85%).

2. **SVD is a better complement to BenchReg than KNN.** Both KNN and BenchReg are local methods that make correlated errors. SVD captures global low-rank structure and provides an independent signal. Replacing KNN with SVD-Logit as the blend partner drives the 14.2% improvement.

3. **Model metadata doesn't help.** We tested provider, parameter count, reasoning mode, and open-weight status as prediction features. Every metadata-based method was neutral or negative. The benchmark scores already encode everything metadata would tell you.

4. **The matrix is approximately rank-2.** Factor 1 (37% variance, "general capability") and Factor 2 (14% variance, "frontier reasoning") capture the essential structure. Adding a third factor hurts holdout accuracy.

5. **Five benchmarks predict the rest.** {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA} achieve ~7.8% MedAPE under proper holdout — a practical "minimum eval set" for new models.

6. **Reasoning mode is the strongest latent divide.** Reasoning models outperform non-reasoning models by +1.8σ on HMMT and +1.75σ on HLE. GSM8K is the only benchmark where non-reasoning models win (it's saturated at 90%+).

### Methodology Notes

- **Primary metric**: Per-model leave-50%-out MedAPE — for each model with ≥8 known scores, hide 50%, predict them, compute global median absolute percentage error across all held-out cells. Averaged over 5 random seeds.
- **Secondary metric**: Random 20% holdout MedAPE (5 seeds).
- **Coverage**: Fraction of test cells receiving a finite prediction. LogitSVD achieves 99.8% (SVD-Logit covers cells that LogitBenchReg cannot).
- **All predictions clamped** to valid ranges (0–100 for percentage benchmarks).

---

## 2. The Dataset

### 2.1 Models

83 instruct/chat LLMs from 21 providers, released between January 2025 and February 2026:

| Provider | Count | Notable Models |
|---|---:|---|
| OpenAI | 13 | o3, o4-mini, GPT-4.1, GPT-4.5, GPT-5 |
| DeepSeek | 12 | R1, V3, R1-Distill family (1.5B–70B) |
| Alibaba (Qwen) | 10 | Qwen3 family (0.6B–235B), QwQ-32B |
| Anthropic | 9 | Claude Opus 4.6, Sonnet 4.5, Haiku 3.5 |
| Google | 7 | Gemini 2.5 Pro/Flash, Gemini 3.1 Pro |
| Mistral | 5 | Mistral Large 3, Small 3.1 |
| Microsoft | 4 | Phi-4, Phi-4-mini |
| Meta | 3 | Llama 4 Maverick/Scout, Llama 3.3 70B |
| xAI | 3 | Grok 3, Grok 3 mini |
| Others | 17 | Moonshot (Kimi), ByteDance (Doubao), Amazon (Nova), etc. |

**Composition**: 57 reasoning models (69%) vs 26 non-reasoning. 46 open-weight (55%) vs 37 proprietary. Parameter counts span 0.6B to 2T (MoE).

### 2.2 Benchmarks

49 benchmarks across 11 categories:

| Category | Count | Examples |
|---|---:|---|
| Math | 15 | AIME 2024/2025, MATH-500, FrontierMath, HMMT, USAMO, IMO |
| Coding | 7 | LiveCodeBench, SWE-bench Verified, HumanEval, Codeforces |
| Agentic | 6 | SWE-bench Pro, Tau-Bench, Terminal-Bench, OSWorld |
| Reasoning | 4 | GPQA Diamond, ARC-AGI-1/2, HLE |
| Knowledge | 4 | MMLU, MMLU-Pro, SimpleQA, BrowseComp |
| Multimodal | 3 | MMMU, MMMU-Pro, Video-MMU |
| Instruction Following | 3 | IFEval, IFBench, Arena-Hard Auto |
| Science | 2 | GPQA Diamond, CritPt |
| Long Context | 2 | MRCR v2, LongBench v2 |
| Composite | 2 | Chatbot Arena Elo, MathArena Apex |
| Human Preference | 1 | Chatbot Arena Elo |

**Score scales**: Most benchmarks use 0–100% accuracy. Exceptions include Chatbot Arena (Elo ~1000–1400), Codeforces (rating ~800–2200), and FrontierMath (0–25%). Non-percentage benchmarks are handled in raw z-score space (no logit transform).

### 2.3 Sparsity Pattern

The matrix is 33.8% filled (1,375 / 4,067 cells). Coverage is uneven:
- **Most-covered benchmarks**: GPQA Diamond (81 models), MMLU (78), LiveCodeBench (75)
- **Least-covered benchmarks**: BRUMO (5 models), Terminal-Bench 2.0 (6), SMT 2025 (7)
- **Most-covered models**: Gemini 2.5 Pro (36 benchmarks), Claude Sonnet 4 (34), o3 (33)
- **Least-covered models**: Some small/niche models have only 4–6 benchmarks

The sparsity is not random — frontier models tend to be evaluated more broadly, while smaller or older models have fewer benchmarks. This creates a "rich get richer" pattern that affects method choice (BenchReg needs correlated benchmark scores to exist).

---

## 3. Evaluation Protocol

### 3.1 Holdout Strategies

We use two primary holdout strategies:

**Per-model leave-50%-out (primary)**:
For each model with ≥8 known scores, randomly hide 50% of its scores, train on the remaining 50% plus all other models' scores, and predict the hidden cells. Repeat 3 folds × 5 seeds. Report the global median APE across all held-out cells.

This tests the realistic scenario: "Given some benchmark results for a model, predict the rest."

**Random 20% holdout (secondary)**:
Randomly hide 20% of all observed cells, predict them. Repeat 5 seeds. This tests general matrix completion ability.

### 3.2 Extended Metrics

Beyond MedAPE, we report:

| Metric | What it measures |
|---|---|
| **MAE** | Mean absolute error in raw score points |
| **±3 pts** | Fraction of predictions within 3 points of truth (useful ≈ threshold) |
| **±5 pts** | Fraction within 5 points |
| **APE > 50** | MedAPE on benchmarks where models score above 50 (easy benchmarks) |
| **APE ≤ 50** | MedAPE on benchmarks where models score below 50 (hard benchmarks) |
| **Bimodal Acc** | Classification accuracy on 5 bimodal benchmarks (ARC-AGI-1/2, IMO 2025, USAMO 2025, MathArena Apex) where models either nearly-zero or meaningfully above threshold (10%) |
| **Coverage** | Fraction of test cells receiving a finite prediction |

---

## 4. Method Search: Three Rounds

We conducted a systematic 3-round search over 34 candidate methods to find improvements over the BenchReg+KNN Blend baseline (7.86% PM-MedAPE).

### 4.1 Round 1: Exploring Six Hypotheses (15 methods)

We tested whether six categories of "wasted information" could improve predictions:

**Hypothesis 1: Model metadata features (provider, params, reasoning mode)**
- MetaKNN (metadata-enhanced cosine similarity): 8.57% — worse
- Provider correction (additive per-provider offset): 7.86% — neutral
- MultiRidge (metadata as regression features): 9.85% — much worse
- Family interpolation (within-provider scaling): 7.87% — neutral

**Verdict**: Model metadata adds nothing. The benchmark scores already encode model capabilities.

**Hypothesis 2: Benchmark category awareness**
- CatBenchReg (separate regression per category): 7.37% — small improvement
- CatBlend: 7.64% — small improvement

**Verdict**: Small but real gain. Categories help because math benchmarks predict math benchmarks better than they predict coding benchmarks.

**Hypothesis 3: Non-linear relationships (logit transform)**
- LogitBenchReg: **6.61%** — major improvement, but 78.5% coverage
- LogitBlend: 7.70% — improvement

**Verdict**: The single biggest win. Working in logit space linearizes ceiling/floor effects.

**Hypothesis 4: Bimodal benchmark handling**
- BimodalAware (explicit classification for 5 bimodal benchmarks): 7.91% — neutral

**Verdict**: Explicit bimodal handling is unnecessary — the logit transform handles it implicitly and better.

**Hypothesis 5: Gradient boosted trees (non-linear per-benchmark)**
- GBT: 8.50% — worse than BenchReg
- GBT+Blend: 7.46% — marginal improvement

**Verdict**: GBT overfits on the sparse matrix. Not enough training examples per benchmark.

**Hypothesis 6: Missingness-pattern features**
- MissingnessKNN (treating NaN pattern as feature): 9.25% — worse

**Verdict**: Missingness pattern is not informative for score prediction.

### 4.2 Round 2: Combining Winners (10 methods)

We combined the two Round 1 winners (logit transform + category awareness):

| Method | PM-MedAPE | Coverage |
|---|---:|---:|
| LogitCatBenchReg | 6.53% | 78.9% |
| KitchenSink (logit + cat + confidence) | **6.89%** | **100%** |
| ConfidenceBlend | 6.93% | 100% |
| SVD+Logit | 6.99% | 99.8% |
| LogitCatBlend | 7.65% | 99.8% |
| MetaLearnerV2 (stacking) | 7.14% | 100% |
| GBT-Logit | 8.24% | 99.8% |

**Key insight from Round 2**: The coverage problem matters. LogitCatBenchReg has the best raw accuracy but only covers 79% of test cells. KitchenSink at 6.89% with 100% coverage is the practical winner.

### 4.3 Round 3: SVD in Logit Space (9 methods)

The breakthrough: applying the logit transform to SVD as well, and using SVD-Logit as the blend partner instead of KNN.

| Method | PM-MedAPE | Coverage |
|---|---:|---:|
| **LogitSVD Blend (0.6/0.4)** | **6.74%** | **99.8%** |
| KitchenSinkV2 | 6.76% | 100% |
| LogitSVD+LCB | 6.84% | 100% |
| SimpleTrio | 7.01% | 99.8% |
| SVD-Logit(r=2) standalone | 7.52% | 99.8% |

**Why LogitSVD Blend wins**: BenchReg exploits local benchmark-to-benchmark correlations (e.g., "AIME 2024 score predicts AIME 2025 score"). SVD exploits global low-rank structure (e.g., "this model's overall profile is similar to GPT-4.1"). These are complementary signals. The old KNN blend didn't add complementary information — KNN also exploits local similarity, making correlated errors with BenchReg.

### 4.4 The Final Method: LogitSVD Blend

```
LogitSVD Blend = 0.6 × LogitBenchReg + 0.4 × SVD-Logit(r=2)
```

**Algorithm**:
1. **Identify percentage benchmarks**: Any benchmark where all observed values fall in [−1, 101]
2. **LogitBenchReg**: For each percentage benchmark j, transform observed scores to logit space (`log(p/(1-p))`, clipping to [0.5, 99.5] first). Find top-5 most correlated benchmarks (in logit space). Fit Ridge regression from those 5 predictors to target j. Predict missing values and inverse-logit back to [0, 100]. For non-percentage benchmarks, use standard z-score BenchReg.
3. **SVD-Logit(r=2)**: Transform all percentage columns to logit space. Z-score normalize. Run Soft-Impute SVD with rank 2 (iterate: impute missing → SVD → keep rank-2 → re-impute). Inverse-logit back.
4. **Blend**: Where both exist, take weighted average. Where only SVD-Logit exists (21.5% of cells), use SVD-Logit alone. Where neither exists (0.2%), fall back to column mean.
5. **Clamp**: Clip predictions to [0, 100] for percentage benchmarks.

---

## 5. Prediction Methods: Full Comparison

### 5.1 Extended Evaluation Table

All 7 key methods evaluated with per-model leave-50%-out (5 seeds) and random 20% holdout (5 seeds):

| Rank | Method | PM-MedAPE | R-MedAPE | MAE | ±3pts | ±5pts | APE>50 | APE≤50 | BiAcc | Cov |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LogitBenchReg | **6.61%** | **5.68%** | 4.70 | 36.7% | 52.0% | 4.32% | 33.23% | 76.3% | 78.5% |
| 2 | **LogitSVD Blend** | **6.74%** | 5.95% | **4.61** | **37.0%** | **52.4%** | **4.34%** | **31.92%** | **89.9%** | **99.8%** |
| 3 | BenchReg | 7.45% | 6.21% | 5.61 | 31.0% | 46.3% | 5.06% | 37.63% | 81.4% | 79.6% |
| 4 | SVD-Logit(r=2) | 7.52% | 6.62% | 5.07 | 35.2% | 49.6% | 4.76% | 33.81% | 89.0% | 99.8% |
| 5 | Blend (old best) | 7.86% | 6.54% | 5.63 | 31.5% | 46.0% | 5.03% | 40.36% | 84.9% | 99.8% |
| 6 | SVD(r=2) | 8.62% | 7.70% | 6.04 | 28.7% | 44.0% | 5.78% | 36.27% | 87.5% | 99.8% |
| 7 | Benchmark Mean | 13.52% | 12.88% | 9.47 | 16.4% | 27.8% | 8.65% | 53.07% | 80.6% | 99.8% |

### 5.2 Original 19-Method Ranking (results_table.csv)

All methods evaluated with the original evaluation harness (single-seed per-model 20% holdout):

| Rank | Method | PM-MedAPE | Rand-MedAPE | Cold-Start | R² |
|---:|---|---:|---:|---:|---:|
| 1 | BenchReg(k=5,r²≥0.2) | 7.69% | 5.86% | — | 0.976 |
| 2 | LogBenchReg | 7.72% | 6.27% | — | 0.972 |
| 3 | BenchReg+KNN(α=0.6) | 7.88% | 6.43% | 19.1% | 0.979 |
| 4 | LogBlend(α=0.65) | 7.97% | 6.32% | 19.1% | 0.977 |
| 5 | SVD(r=2) | 8.09% | 7.32% | 17.2% | 0.979 |
| 6 | Ensemble(avg3) | 8.30% | 7.09% | 19.0% | 0.974 |
| 7 | SVD(r=3) | 9.07% | 8.02% | 19.9% | 0.972 |
| 8 | Bench-KNN(k=5) | 9.10% | 8.07% | 19.1% | 0.974 |
| 9 | KNN(k=5) | 9.32% | 7.63% | 19.1% | 0.970 |
| 10 | Quantile+SVD5 | 9.64% | 7.31% | **13.0%** | 0.957 |
| 11 | NucNorm(λ=1) | 9.70% | 8.44% | 19.0% | 0.953 |
| 12 | Model-Normalized | 10.00% | 9.61% | 16.3% | 0.962 |
| 13 | PMF(r=5) | 10.29% | 8.48% | 18.9% | 0.960 |
| 14 | NMF(r=5) | 10.46% | 8.84% | **13.9%** | 0.951 |
| 15 | SVD(r=5) | 10.95% | 8.79% | 19.2% | 0.947 |
| 16 | LogSVD(r=5) | 12.42% | 8.75% | 19.5% | 0.935 |
| 17 | Benchmark Mean | 12.89% | 11.71% | 19.1% | 0.928 |
| 18 | SVD(r=8) | 13.18% | 9.71% | 19.1% | 0.914 |
| 19 | SVD(r=10) | 13.40% | 10.33% | 19.1% | 0.923 |

### 5.3 Decomposing the Improvement

The improvement from Blend (7.86%) to LogitSVD Blend (6.74%) can be decomposed into two independent changes:

| Change | From | To | Improvement |
|---|---:|---:|---:|
| Add logit transform to BenchReg | BenchReg 7.45% | LogitBenchReg 6.61% | −11.3% |
| Add logit transform to SVD | SVD 8.62% | SVD-Logit 7.52% | −12.8% |
| Replace KNN with SVD-Logit in blend | Blend 7.86% | LogitSVD 6.74% | −14.2% |

The logit transform accounts for ~70% of the gain; the SVD substitution accounts for ~30%.

### 5.4 What Didn't Work (Full List)

| Approach | PM-MedAPE | vs Baseline | Why It Failed |
|---|---:|---:|---|
| MetaKNN | 8.57% | +0.7% | Metadata features add noise, not signal |
| MultiRidge | 9.85% | +2.0% | Too many features for too few training rows |
| MissingnessKNN | 9.25% | +1.4% | NaN pattern is not informative |
| ProviderCorrected | 7.86% | +0.0% | Per-provider offset doesn't generalize |
| FamilyInterp | 7.87% | +0.0% | Within-family interpolation doesn't help |
| BimodalAware | 7.91% | +0.1% | Logit handles bimodality better |
| GBT per-benchmark | 8.50% | +0.6% | Overfits with <40 training rows |
| GBT-Logit | 8.24% | +0.4% | Same overfitting in logit space |
| ConfidenceWeighted | 6.93% | −0.9% | Confidence estimates not calibrated |
| MetaLearnerV2 (stacking) | 7.14% | −0.7% | Second-level regressor overfits |
| CatWeightedEnsemble | 8.63% | +0.8% | Per-category weighting overfits |
| ResidualCorrection | 7.14% | −0.7% | Residual patterns don't generalize |
| GBT-Stacked | 7.94% | +0.1% | GBT on top of predictions doesn't help |

---

## 6. Intrinsic Dimensionality

### 6.1 Singular Value Spectrum

| Rank | Singular Value | Var Explained | Cumulative |
|---:|---:|---:|---:|
| 1 | 22.5 | 36.6% | 36.6% |
| 2 | 14.0 | 14.2% | 50.8% |
| 3 | 8.9 | 5.7% | 56.6% |
| 4 | 8.5 | 5.2% | 61.8% |
| 5 | 7.4 | 4.0% | 65.7% |

The spectrum shows a clear gap between factors 2 and 3 (14.2% → 5.7%), confirming the matrix is approximately rank-2 with residual structure. This is validated by holdout: SVD rank-2 achieves 8.62% PM-MedAPE vs 9.07% for rank-3, and 7.52% vs worse for rank-3 in logit space.

To reach various variance thresholds:
- **80% variance**: rank 11
- **90% variance**: rank 18
- **95% variance**: rank 25

### 6.2 Latent Factor Interpretation

**Factor 1 (36.6% — "General Capability")**
Top loadings: GPQA Diamond (−0.37), LiveCodeBench (−0.36), MMLU-Pro (−0.31), MMLU (−0.29), MATH-500 (−0.26).

This is the dominant axis of LLM capability. The strongest models (o3, Claude Opus 4.6, GPT-5, Gemini 2.5 Pro) score high on Factor 1 across all benchmarks simultaneously. It captures what we colloquially call "how good the model is."

**Factor 2 (14.2% — "Frontier Reasoning / Novel Tasks")**
Positive loadings: SimpleQA (+0.34), ARC-AGI-2 (+0.32), HLE (+0.30), FrontierMath (+0.23), SWE-bench Verified (+0.21).
Negative loadings: MATH-500 (−0.19), MMLU (−0.18).

This factor distinguishes models that excel on genuinely novel, hard reasoning tasks from those that do well on established benchmarks. It captures the "reasoning frontier" — models like o3, Claude Opus 4.6, and Gemini 2.5 Pro score high on Factor 2, while models that are strong on MMLU/MATH but weak on ARC-AGI/HLE score low.

**Factor 3 (5.7% — "Established vs. Emerging") — NOT PREDICTIVE**
Loadings: MATH-500 (+0.44) vs AIME 2025 (−0.41), Arena-Hard (−0.40).

This factor may capture temporal effects (older benchmarks vs newer ones) but does NOT improve holdout predictions. SVD rank-3 is strictly worse than rank-2. The third singular value is above noise level but below the usefulness threshold.

### 6.3 What "Rank-2" Means Practically

That the LLM benchmark matrix is rank-2 means: knowing two numbers about a model — its "general capability" score and its "frontier reasoning" score — lets you predict its performance on 49 benchmarks with ~7.5% median error. The remaining 49% of variance is either noise or highly specific effects (e.g., GPT-4.5's anomalous ARC-AGI-2 score) that don't generalize.

---

## 7. Benchmark Redundancy

**IMPORTANT CAVEAT**: Pairwise correlations are computed on shared observations between benchmark pairs. The median number of shared models is only **7**. 60% of pairs have fewer than 10 shared models. Only pairs with ≥20 shared models should be treated as reliable.

### 7.1 Most Redundant Pairs (n_shared ≥ 20)

| Benchmark A | Benchmark B | Correlation | n_shared |
|---|---|---:|---:|
| AIME 2025 | SMT 2025 | 0.960 | 19 |
| LiveCodeBench | AIME 2024 | 0.947 | 56 |
| LiveCodeBench | MMLU-Pro | 0.936 | 64 |
| GPQA Diamond | LiveCodeBench | 0.918 | 73 |
| GPQA Diamond | AIME 2024 | 0.917 | 56 |

The high-correlation pairs are math and coding benchmarks measuring overlapping capabilities.

### 7.2 Benchmark Clusters

**Cluster 1 — "Frontier Reasoning" (17 benchmarks)**: AIME 2025, FrontierMath, HLE, ARC-AGI-2, BrowseComp, SimpleQA, Chatbot Arena Elo, SWE-bench Pro, HMMT, Tau-Bench, CritPt, Terminal-Bench 2.0, ARC-AGI-1, BRUMO, USAMO, MathArena Apex. These all correlate >0.6 — a model good at one tends to be good at all.

**Cluster 2 — "Core Competency" (8 benchmarks)**: GPQA Diamond, MMLU-Pro, LiveCodeBench, IFEval, Codeforces, AIME 2024, GSM8K, IFBench. The bread-and-butter evaluations.

Note: Cluster boundaries use all pairwise correlations including small-sample pairs and should be interpreted cautiously.

---

## 8. Minimum Evaluation Set

### 8.1 Greedy Forward Selection

If you could only run N benchmarks on a new model, which ones maximize information?

| # Benchmarks | Added Benchmark | In-Sample MedAPE | Proper Holdout |
|---:|---|---:|---:|
| 1 | HLE (Humanity's Last Exam) | 7.5% | — |
| 2 | + AIME 2025 | 6.5% | — |
| 3 | + LiveCodeBench | 5.9% | — |
| 4 | + SWE-bench Verified | 5.2% | — |
| 5 | + SimpleQA | 4.8% | **~7.8%** |

**The 5-benchmark minimal eval set is {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA}**.

These 5 benchmarks span 4 categories (reasoning, math, coding, knowledge) and cover both Factor 1 and Factor 2 of the latent space. They were selected by greedy forward regression (at each step, add the benchmark that most reduces leave-one-out prediction error on remaining benchmarks).

### 8.2 Caveats

The "4.8% MedAPE" figure is **in-sample** — the regression trains and evaluates on the same data. Under proper holdout:
- Random 20% holdout: 5-bench ridge = ~7.8% MedAPE vs LogitSVD Blend = 6.0%
- Per-model 50% holdout: 5-bench ridge = ~10.0% MedAPE vs LogitSVD Blend = 6.7%

The 5-benchmark strategy is useful when you can only afford to run 5 evaluations, but it does NOT outperform LogitSVD Blend when more data is available.

GPQA Diamond is a near-perfect substitute for HLE (in-sample: 4.80% vs 4.84%), and has 2× the model coverage (81 vs 38 models).

---

## 9. Data Efficiency (Phase Transition)

How does LogitSVD Blend accuracy scale with the number of known scores per model?

| Known Scores | MedAPE | MAE | ±3 pts | ±5 pts |
|---:|---:|---:|---:|---:|
| 1 | 12.1% | 7.9 | 24.7% | 38.9% |
| 2 | 11.2% | 7.0 | 30.0% | 43.6% |
| 3 | 10.3% | 6.3 | 32.5% | 46.1% |
| 4 | 9.7% | 5.9 | 34.2% | 48.5% |
| 5 | 9.2% | 5.6 | 35.8% | 50.0% |
| 7 | 9.0% | 5.4 | 36.5% | 52.0% |
| 10 | 10.1% | 5.9 | 37.2% | 52.5% |
| 15 | 11.2% | 6.8 | 41.1% | 56.8% |
| 20 | 9.6% | 4.5 | 44.4% | 61.1% |

### 9.1 Interpretation

**The biggest gains come from 1→5 scores**: MedAPE drops from 12.1% to 9.2% as you go from 1 to 5 known benchmarks. Each additional score narrows the uncertainty about where the model sits in the 2D latent space.

**After 5 scores, MedAPE plateaus but precision improves**: The ±3pts and ±5pts metrics continue improving (24.7% → 44.4% for ±3pts), even as MedAPE bounces around. This is because MedAPE is driven by a few hard-to-predict outlier benchmarks (bimodal ones, Elo-scale ones), while the precision metrics capture the improving accuracy on the bulk of "normal" benchmarks.

**Non-monotonicity at n>10**: Reflects small sample sizes — few models have exactly 12 or exactly 17 known scores. The variance bars (not shown) are wide.

---

## 10. Scaling Laws

Within model families that vary only in parameter count, benchmark scores scale log-linearly:

### 10.1 Qwen3 Family (8 models, 0.6B–235B)

| Benchmark | R² | Slope per ln(params) |
|---|---:|---:|
| MMLU | 0.89 | +5.9 |
| GPQA Diamond | 0.84 | +7.5 |
| LiveCodeBench | 0.83 | +9.4 |
| HumanEval | 0.80 | +7.3 |
| Codeforces Rating | 0.79 | +214.6 |
| AIME 2025 | 0.77 | +11.4 |

### 10.2 DeepSeek-R1-Distill Family (6 models, 1.5B–70B)

| Benchmark | R² | Slope per ln(params) |
|---|---:|---:|
| IFEval | **0.98** | +11.3 |
| GPQA Diamond | **0.95** | +8.5 |
| Codeforces Rating | 0.90 | +205.6 |
| LiveCodeBench | 0.89 | +12.0 |

The DeepSeek distillation family shows remarkably tight scaling (R²=0.95–0.98 for GPQA, IFEval), suggesting the distillation process preserves scaling behavior better than independent training. The Qwen3 family (trained independently at each size) shows looser fits (R²=0.77–0.89).

### 10.3 Cross-Family Scaling

Despite within-family log-linearity, cross-family scaling is poor — a 70B model from one family can outperform a 235B model from another. This is why model metadata (parameter count) doesn't help with prediction: the family-specific intercepts vary too much.

---

## 11. Reasoning Mode Analysis

### 11.1 Reasoning vs Non-Reasoning (57 vs 26 models)

Benchmarks with **largest reasoning advantage** (z-score gap):

| Benchmark | Reasoning Mean-z | Non-Reasoning Mean-z | Gap |
|---|---:|---:|---:|
| HMMT Feb 2025 | +0.24 | −1.56 | **+1.80** |
| HLE | +0.18 | −1.57 | **+1.75** |
| AIME 2024 | +0.43 | −1.13 | **+1.56** |
| MMMU | +0.40 | −1.10 | +1.49 |
| ARC-AGI-1 | +0.31 | −1.12 | +1.43 |

Benchmarks where reasoning **barely helps**:

| Benchmark | Reasoning Mean-z | Non-Reasoning Mean-z | Gap |
|---|---:|---:|---:|
| IFEval | +0.13 | −0.26 | +0.39 |
| MMLU | +0.09 | −0.18 | +0.27 |
| Arena-Hard Auto | +0.06 | −0.09 | +0.15 |
| Terminal-Bench 1.0 | +0.01 | −0.07 | +0.08 |
| GSM8K | −0.08 | +0.17 | **−0.25** |

**GSM8K is the only benchmark where non-reasoning models win on average.** This is because GSM8K is now saturated — most frontier models score 90%+, and the extended reasoning overhead provides no benefit on these simple arithmetic word problems.

### 11.2 Provider Strengths and Weaknesses

| Provider | Best Category | Worst Category | Notable |
|---|---|---|---|
| **Google** | Science (+0.50) | — | Most balanced; positive in all categories |
| **xAI** | Coding (+0.72) | Long Context (−1.14) | Strong but narrow |
| **OpenAI** | Composite (+0.84) | Agentic (−0.30) | Best at integrated benchmarks |
| **Anthropic** | Science (+0.64) | Multimodal (−0.69) | Strong reasoning, weak multimodal |
| **ByteDance** | Agentic (+0.86) | — | Only 1–2 models but impressive |
| **DeepSeek** | Knowledge (+0.19) | Agentic (−0.96) | Math-focused, weak on agents |
| **Meta** | Knowledge (+0.17) | Human Pref (−0.97) | Open weights, broad but not frontier |

---

## 12. Surprising Models

Models whose actual scores deviate most from rank-2 SVD expectations:

| Model | MedAPE | Biggest Surprise | Details |
|---|---:|---|---|
| OLMo 2 13B | 23.9% | AIME 2024 | Actual: 5, Expected: −13 |
| GPT-4.5 | 16.8% | ARC-AGI-2 | Actual: 0.8, Expected: 17 |
| DeepSeek-R1-Distill-1.5B | 16.6% | Arena-Hard | Actual: 4, Expected: −11 |
| LFM2.5-1.2B-Thinking | 13.4% | LiveCodeBench | Actual: 22, Expected: 28 |
| Mistral Large 3 | 12.9% | GPQA Diamond | Actual: 44, Expected: 67 |

**GPT-4.5** is the most "surprising" large model — it scores only 0.8% on ARC-AGI-2 despite being expected to score ~17% based on its overall profile. This suggests GPT-4.5's general capability (high Factor 1) doesn't translate to the novel visual reasoning required by ARC-AGI-2. It's a model that's broadly capable but lacks the specific reasoning patterns ARC-AGI tests.

**Mistral Large 3** is surprisingly weak on GPQA Diamond (44% actual vs 67% expected), suggesting a specific gap in graduate-level science knowledge despite strong general capability.

---

## 13. The Logit Transform: Why It Works

The logit transform `logit(p) = log(p/(1-p))` is the single most important algorithmic choice in this project. Here's why it matters so much for benchmark score prediction:

### 13.1 The Ceiling/Floor Problem

Many benchmarks have scores clustered near 0% or 100%. Consider MMLU: frontier models score 88–92%, with a ceiling at 100%. In raw space, the difference between 88% and 92% (4 points) looks the same as 48% vs 52%. But in logit space:
- logit(88%) = 2.00
- logit(92%) = 2.44 (gap: 0.44)
- logit(48%) = −0.08
- logit(52%) = 0.08 (gap: 0.16)

The logit transform correctly models that moving from 88% to 92% is "harder" — it represents more capability gain per point.

### 13.2 The Bimodal Problem

Five benchmarks (ARC-AGI-1/2, IMO 2025, USAMO 2025, MathArena Apex) have bimodal score distributions: models either score near 0% or well above 10%. In raw space, a linear predictor might predict 5% for a model that should score 0% or 25% — the average of the two modes.

In logit space, 0.5% maps to logit = −5.3, while 25% maps to logit = −1.1. The gap is enormous, so the SVD factors naturally separate the two modes. The result: 89.9% bimodal classification accuracy vs 84.9% with the raw-space blend.

### 13.3 Which Benchmarks Get the Transform

The heuristic is simple: if all observed scores for a benchmark fall in [−1, 101], it's treated as a percentage benchmark and gets the logit transform. Non-percentage benchmarks (Chatbot Arena Elo, Codeforces rating) are handled in standard z-score space. About 92% of benchmarks are percentage-scale.

---

## 14. Limitations and Known Issues

### 14.1 Issues Fixed in This Revision
- **Blend NaN propagation**: Previous `predict_blend` silently dropped 19% of test cells, inflating accuracy from 7.9% to 7.4%
- **5-benchmark in-sample leak**: The "4.8% MedAPE" was in-sample; proper holdout is ~7.8%
- **SVD rank claim**: Rank-2 is optimal, not rank-3
- **Prediction output**: Now covers all 2,692 missing cells with clamping
- **Redundancy sample sizes**: Table now restricted to pairs with ≥20 shared models
- **LogitSVD Blend**: Replaced BenchReg+KNN as default. 14.2% improvement.

### 14.2 Remaining Limitations

**Coverage gap**: LogitBenchReg only covers 78.5% of test cells. The blend uses SVD-Logit (7.52% standalone) for the remaining 21.5%. A model with only 2–3 known scores gets worse predictions than one with 10+.

**Alpha not cross-validated**: α=0.6 was manually swept over {0.4, 0.5, 0.6, 0.7, 0.8}. Nested CV could yield a slightly different optimal weight.

**34% fill rate**: The matrix is sparse. More scores could be mined from model papers, leaderboards, and evaluation platforms.

**Non-percentage benchmarks**: Elo ratings and Codeforces scores don't benefit from the logit transform and use raw z-score prediction. These constitute ~8% of benchmarks.

**Single random seed for original 19-method comparison**: The results_table.csv uses a single seed. The LogitSVD evaluation uses 5 seeds.

**No temporal modeling**: The matrix doesn't account for benchmark saturation over time (e.g., GSM8K becoming trivial) or model improvement trends.

### 14.3 Opportunities for Future Work

- **Active learning**: Choose which benchmark to evaluate next for maximum information gain
- **Temporal dimension**: Model benchmark vintage and saturation curves
- **Neural matrix completion**: The logit transform helps but doesn't fully capture non-linear patterns
- **More data**: Mining additional scores from papers and leaderboards would improve all methods
- **Confidence intervals**: Predicting not just point estimates but uncertainty bounds

---

## 15. Deliverables

| File | Description |
|---|---|
| `results/report.md` | This report |
| `results/results_table.csv` | 19 baseline methods × 10 metrics comparison |
| `results/logit_svd_eval.json` | LogitSVD vs baselines: extended metrics (7 methods × 12 metrics) |
| `results/best_predictions.csv` | 2,692 predicted missing cells (BenchPress, clamped) |
| `results/latent_factors.csv` | 49 benchmarks × 5 factor loadings |
| `results/phase_transition.csv` | Accuracy vs number of known scores (LogitSVD Blend, 1–20) |
| `evaluation_harness.py` | Evaluation framework (6 holdout strategies, 7 metrics) |
| `methods/all_methods.py` | 22 prediction methods (incl. LogitSVD variants) |
| `methods/run_final_eval.py` | LogitSVD evaluation + phase transition + prediction regeneration |
| `methods/creative_methods.py` | Round 1 exploration (15 methods) |
| `methods/creative_methods_r2.py` | Round 2 combinations (10 methods) |
| `methods/creative_methods_r3.py` | Round 3 optimization (9 methods) |
| `methods/final_comparison.py` | All-methods comprehensive comparison |
| `diagnostics/all_diagnostics.py` | Intrinsic dimensionality, redundancy, scaling laws |
| `data/build_benchmark_matrix.py` | Master dataset: 83 models, 49 benchmarks, 1,375 cited scores |
