# LLM Benchmark Matrix Completion: Analysis Report

**Matrix**: 83 models × 49 benchmarks | **1,383 observed** (34.0% fill) | **2,684 predicted**
**Date**: February 23, 2026 (revised)

---

## 1. Executive Summary

We constructed a cited benchmark matrix spanning 83 instruct/chat LLMs across 49 benchmarks (math, coding, reasoning, knowledge, agentic, multimodal, instruction following, long context). Every entry carries a source URL. We then evaluated 19 prediction methods for filling the 66% of missing cells.

**Key findings:**
- **Best method**: BenchReg+KNN blend (α=0.6) achieves **6.4% MedAPE** on random holdout and **7.9% MedAPE** on per-model leave-50%-out (global-cell median, evaluated at full coverage with KNN fallback)
- **The matrix is approximately rank-2**: Factor 1 explains 37% of variance; 2 factors capture 51%. SVD rank-2 (8.1% PM-MedAPE) beats rank-3 (9.1%)
- **5 benchmarks predict the rest**: {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA} achieve ~7.8% MedAPE under proper holdout (in-sample was 4.8%, but this is overfit)
- **Reasoning mode is the strongest latent factor**: z-score gap of +1.8σ on HMMT, +1.75σ on HLE
- **Cold-start remains hard**: BenchReg-based methods produce no predictions; NMF/Quantile achieve ~13%

**Methodology notes:**
- "PM-MedAPE" = global median of absolute percentage errors across all held-out cells in per-model folds (not median of per-model medians, which would give 7.0%)
- Blend/BenchReg coverage: BenchReg produces predictions for ~81% of test cells; the fixed blend uses KNN fallback for the remaining 19%. Previous versions silently dropped unpredicted cells, inflating reported accuracy
- All predictions clamped to valid ranges (0–100 for percentage benchmarks)

---

## 2. Prediction Methods Comparison

### 2.1 Rankings (sorted by Per-Model MedAPE, primary metric)

All methods evaluated at full coverage (NaN fallback applied where needed).

| Rank | Method | PM-MedAPE | Random-MedAPE | Cold-Start | R² | <10% |
|---:|---|---:|---:|---:|---:|---:|
| 1 | BenchReg(k=5,r²≥0.2) | **7.7%** | **5.9%** | — | 0.966 | 58% |
| 2 | LogBenchReg | **7.7%** | 6.3% | — | 0.975 | 56% |
| 3 | **BenchReg+KNN(α=0.6)** | **7.9%** | 6.4% | 19.1% | 0.978 | 56% |
| 4 | LogBlend(α=0.65) | 8.0% | 6.3% | 19.1% | 0.979 | 63% |
| 5 | **SVD(r=2)** | **8.1%** | 7.3% | **17.2%** | 0.984 | 59% |
| 6 | Ensemble(avg3) | 8.3% | 7.1% | 19.0% | 0.984 | 60% |
| 7 | SVD(r=3) | 9.1% | 8.0% | 19.9% | 0.984 | 59% |
| 8 | Bench-KNN(k=5) | 9.1% | 8.1% | 19.1% | 0.976 | 56% |
| 9 | KNN(k=5) | 9.3% | 7.6% | 19.1% | 0.976 | 58% |
| 10 | Quantile+SVD5 | 9.6% | 7.3% | **13.0%** | 0.979 | 58% |
| 11 | NucNorm(λ=1) | 9.7% | 8.4% | 19.0% | 0.969 | 56% |
| 12 | Model-Normalized | 10.0% | 9.6% | 16.3% | 0.968 | 51% |
| 13 | PMF(r=5) | 10.3% | 8.5% | 18.9% | 0.974 | 55% |
| 14 | NMF(r=5) | 10.5% | 8.8% | **13.9%** | 0.960 | 56% |
| 15 | SVD(r=5) | 10.9% | 8.8% | 19.2% | 0.985 | 55% |
| 16 | LogSVD(r=5) | 12.4% | 8.7% | 19.5% | 0.977 | 55% |
| 17 | Benchmark Mean | 12.9% | 11.7% | 19.1% | 0.935 | 45% |
| 18 | SVD(r=8) | 13.2% | 9.7% | 19.1% | 0.956 | 51% |
| 19 | SVD(r=10) | 13.4% | 10.3% | 19.1% | 0.935 | 49% |

### 2.2 Key Takeaways

**BenchReg dominates on random holdout**: The benchmark-regression family (predicting each benchmark from its k=5 most correlated benchmarks via ridge regression) beats all matrix factorization approaches. However, BenchReg and LogBenchReg only produce predictions for ~81% of test cells in per-model holdout (they cannot predict when a model lacks scores on correlated benchmarks). The Blend adds KNN fallback to achieve full coverage.

**SVD rank-2 is optimal**: Rank 2 (8.1% PM-MedAPE) beats rank 3 (9.1%). The third factor's 5.7% variance does not translate to holdout predictive signal — it overfits to noise. This aligns with the PCA gap between factors 2 and 3.

**Cold-start is fundamentally different**: BenchReg methods cannot make cold-start predictions (they need correlated benchmark scores to exist). For cold-start, Quantile+SVD (13.0%) and NMF (13.9%) are best — both leverage the global low-rank structure rather than benchmark-specific correlations. SVD rank-2 also performs well (17.2%).

**Recommendation**: Use BenchReg+KNN(α=0.6) as the primary predictor (with KNN fallback for uncovered cells), falling back to Quantile+SVD for cold-start models (≤3 known scores).

**Note on alpha**: α=0.6 was chosen by manual comparison of 3 values (0.6, 0.65, 0.7). On the primary folds, α=0.7 is marginally better (7.85% vs 7.88%), but the difference is within noise. No nested CV was used.

---

## 3. Intrinsic Dimensionality

### 3.1 Singular Value Spectrum

| Rank | Singular Value | Var Explained | Cumulative |
|---:|---:|---:|---:|
| 1 | 22.5 | 36.6% | 36.6% |
| 2 | 14.0 | 14.2% | 50.8% |
| 3 | 8.9 | 5.7% | 56.6% |
| 4 | 8.5 | 5.2% | 61.8% |
| 5 | 7.4 | 4.0% | 65.7% |

The spectrum shows a clear gap between factors 2 and 3 (14.2% → 5.7%), suggesting the matrix is approximately rank-2 with residual structure. This is confirmed by holdout evaluation: SVD rank-2 achieves 8.1% MedAPE vs 9.1% for rank-3.

To reach various thresholds:
- **80% variance**: rank 11
- **90% variance**: rank 18
- **95% variance**: rank 25

### 3.2 Latent Factor Interpretation

**Factor 1 (36.6% — "General Capability")**: Loads most heavily on GPQA Diamond (-0.37), LiveCodeBench (-0.36), MMLU-Pro (-0.31). This captures overall model quality — the strongest models score high across all these benchmarks simultaneously.

**Factor 2 (14.2% — "Frontier Reasoning / Novel Tasks")**: Loads on SimpleQA (+0.34), ARC-AGI-2 (+0.32), HLE (+0.30), FrontierMath (+0.23). Negative on MATH-500 (-0.19), MMLU (-0.18). This factor distinguishes models that excel on genuinely novel, hard reasoning tasks vs. those that do well on established benchmarks. It captures the "reasoning frontier" — models like o3, Claude Opus 4.6, and GPT-5 score high on Factor 2.

**Factor 3 (5.7% — "Established vs. Emerging")**: Not reliably predictive (SVD rank-3 is worse than rank-2 on holdout). Loads on MATH-500 (+0.44) vs AIME 2025 (-0.41), Arena-Hard (-0.40). This may capture temporal effects (older vs newer benchmarks) but does not improve predictions.

---

## 4. Benchmark Redundancy

**IMPORTANT CAVEAT**: Pairwise correlations in this section are computed on shared observations between benchmark pairs. The median number of shared models across all 1,176 pairs is only **7**. 60% of pairs have fewer than 10 shared models. Correlations based on fewer than 20 observations should be treated as unreliable estimates.

### 4.1 Most Redundant Pairs (n_shared ≥ 20 only)

| Benchmark A | Benchmark B | Correlation | n_shared |
|---|---|---:|---:|
| AIME 2025 | SMT 2025 | 0.960 | 19 |
| LiveCodeBench | AIME 2024 | 0.947 | 56 |
| LiveCodeBench | MMLU-Pro | 0.936 | 64 |
| GPQA Diamond | LiveCodeBench | 0.918 | 73 |
| GPQA Diamond | AIME 2024 | 0.917 | 56 |

When restricted to pairs with ≥20 shared models, the previously reported "most correlated" pairs (e.g., MATH-500 ↔ Video-MMU at r=0.991, n=6) drop out as statistically unreliable. The reliable high-correlation pairs are predominantly math and coding benchmarks that genuinely measure overlapping capabilities.

### 4.2 Benchmark Clusters

**Cluster 1 (17 benchmarks — "Frontier Reasoning")**: AIME 2025, FrontierMath, HLE, ARC-AGI-2, BrowseComp, SimpleQA, Chatbot Arena Elo, SWE-bench Pro, HMMT, Tau-Bench, CritPt, Terminal-Bench 2.0, ARC-AGI-1, BRUMO, USAMO, MathArena Apex. These all correlate >0.6 — a model good at one tends to be good at all.

**Cluster 2 (8 benchmarks — "Core Competency")**: GPQA Diamond, MMLU-Pro, LiveCodeBench, IFEval, Codeforces, AIME 2024, GSM8K, IFBench. These are the "bread and butter" evaluations.

Note: clustering uses all pairwise correlations including small-sample pairs, so cluster boundaries should be interpreted cautiously.

---

## 5. Minimum Evaluation Set

### 5.1 Greedy Forward Selection

If you could only run N benchmarks on a new model, which ones should you pick?

| # Benchmarks | Added Benchmark | In-Sample MedAPE | Proper Holdout |
|---:|---|---:|---:|
| 1 | HLE (Humanity's Last Exam) | 7.5% | — |
| 2 | + AIME 2025 | 6.5% | — |
| 3 | + LiveCodeBench | 5.9% | — |
| 4 | + SWE-bench Verified | 5.2% | — |
| 5 | + SimpleQA | 4.8% | **~7.8%** |

**The 5-benchmark minimal eval set is {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA}**.

**CRITICAL NOTE**: The "4.8% MedAPE" figure in the table above is **in-sample** — the ridge regression trains on all observed entries and evaluates on those same entries. Under proper holdout:
- **Random 20% holdout**: 5-bench ridge = ~7.8% MedAPE vs BenchReg+KNN blend = 6.4%
- **Per-model 50% holdout**: 5-bench ridge = ~10.0% MedAPE vs BenchReg+KNN blend = 7.9%

The 5-benchmark set is still useful as a practical evaluation strategy (run only 5 benchmarks to estimate the rest), but it does NOT outperform BenchReg+KNN blend trained on all available data.

Notably, GPQA Diamond is a near-perfect substitute for HLE: the GPQA-first set achieves 4.80% in-sample (vs HLE-first 4.84%), and both converge to the same remaining 4 benchmarks. GPQA Diamond has 2× the coverage (81 vs 38 models).

---

## 6. Data Efficiency

How does prediction accuracy scale with matrix fill rate?

| Fill Rate | BenchMean | KNN | BenchReg | SVD(r=3) | Blend |
|---:|---:|---:|---:|---:|---:|
| 10% | 14.2% | 11.7% | 8.9% | 18.0% | 7.2% |
| 15% | 12.4% | 10.6% | 8.7% | 8.2% | 7.7% |
| 20% | 14.6% | 9.7% | 6.6% | 8.2% | 6.1% |
| 25% | 11.8% | 9.0% | 7.0% | 9.8% | 6.7% |
| 30% | 10.6% | 7.9% | 6.8% | 7.7% | 5.9% |
| 34% | 15.0% | 8.8% | 7.9% | 8.5% | 7.0% |

**Note**: Non-monotonic values are due to single-seed sampling noise (a single RandomState is mutated across fill rates). 5-seed averages show SVD is the only perfectly monotonic method; all others are monotonic within their standard deviation bands.

**Blend is the most data-efficient method**: Even at 10% fill, it achieves 7.2% MedAPE. SVD collapses at 10% fill (18.0%) because it doesn't have enough data for stable decomposition.

---

## 7. Scaling Laws

Within model families, score scales log-linearly with parameter count:

### Qwen3 Family (8 models, 0.6B–235B)

| Benchmark | R² | Slope per ln(params) |
|---|---:|---:|
| MMLU | 0.89 | +5.9 |
| GPQA Diamond | 0.84 | +7.5 |
| LiveCodeBench | 0.83 | +9.4 |
| HumanEval | 0.80 | +7.3 |
| Codeforces Rating | 0.79 | +214.6 |
| AIME 2025 | 0.77 | +11.4 |

### DeepSeek-R1-Distill Family (6 models, 1.5B–70B)

| Benchmark | R² | Slope per ln(params) |
|---|---:|---:|
| IFEval | **0.98** | +11.3 |
| GPQA Diamond | **0.95** | +8.5 |
| Codeforces Rating | 0.90 | +205.6 |
| LiveCodeBench | 0.89 | +12.0 |

The DeepSeek distillation family shows remarkably tight scaling (R²=0.95–0.98 for GPQA, IFEval), suggesting the distillation process preserves scaling behavior better than independent training.

---

## 8. Reasoning Mode Analysis

### 8.1 Reasoning vs Non-Reasoning (57 vs 26 models)

Benchmarks with **largest reasoning advantage** (z-score gap):

| Benchmark | Reasoning Mean-z | Non-Reasoning Mean-z | Gap |
|---|---:|---:|---:|
| HMMT Feb 2025 | +0.24 | -1.56 | **+1.80** |
| HLE | +0.18 | -1.57 | **+1.75** |
| AIME 2024 | +0.43 | -1.13 | **+1.56** |
| MMMU | +0.40 | -1.10 | +1.49 |
| ARC-AGI-1 | +0.31 | -1.12 | +1.43 |

Benchmarks where reasoning **barely helps**:

| Benchmark | Reasoning Mean-z | Non-Reasoning Mean-z | Gap |
|---|---:|---:|---:|
| IFEval | +0.13 | -0.26 | +0.39 |
| MMLU | +0.09 | -0.18 | +0.27 |
| Arena-Hard Auto | +0.06 | -0.09 | +0.15 |
| Terminal-Bench 1.0 | +0.01 | -0.07 | +0.08 |
| GSM8K | -0.08 | +0.17 | **-0.25** |

GSM8K is the only benchmark where non-reasoning models outperform reasoning models on average. This is because GSM8K is now saturated — most models score 90%+, and reasoning overhead provides no benefit.

### 8.2 Provider Strengths

| Provider | Best Category | Worst Category | Notable |
|---|---|---|---|
| **Google** | Science (+0.50) | — | Most balanced; positive in all categories |
| **xAI** | Coding (+0.72) | Long Context (-1.14) | Strong but narrow |
| **OpenAI** | Composite (+0.84) | Agentic (-0.30) | Best at integrated benchmarks |
| **Anthropic** | Science (+0.64) | Multimodal (-0.69) | Strong reasoning, weak multimodal |
| **ByteDance** | Agentic (+0.86) | — | Only 1-2 models but impressive spread |
| **DeepSeek** | Knowledge (+0.19) | Agentic (-0.96) | Math-focused, weak on agents |
| **Meta** | Knowledge (+0.17) | Human Pref (-0.97) | Open weights, broad but not frontier |

---

## 9. Surprising Models

Models whose scores deviate most from rank-3 SVD expectations:

| Model | MedAPE | Biggest Surprise | Details |
|---|---:|---|---|
| OLMo 2 13B | 23.9% | AIME 2024 | Actual: 5, Expected: -13 |
| GPT-4.5 | 16.8% | ARC-AGI-2 | Actual: 0.8, Expected: 17 |
| DeepSeek-R1-Distill-1.5B | 16.6% | Arena-Hard | Actual: 4, Expected: -11 |
| LFM2.5-1.2B-Thinking | 13.4% | LiveCodeBench | Actual: 22, Expected: 28 |
| Mistral Large 3 | 12.9% | GPQA Diamond | Actual: 44, Expected: 67 |

**GPT-4.5** is the most "surprising" large model — it scores only 0.8% on ARC-AGI-2 despite being expected to score ~17%. This suggests GPT-4.5's general capability (high Factor 1) doesn't translate to the novel visual reasoning required by ARC-AGI-2.

---

## 10. Limitations and Known Issues

### 10.1 Corrected Issues (this revision)
- **Blend NaN propagation (FIXED)**: Previous version of `predict_blend` silently dropped 19% of test cells where BenchReg had no prediction, inflating MedAPE from 7.9% to 7.4%. Now uses KNN fallback.
- **5-benchmark in-sample leak (FLAGGED)**: The "4.8% MedAPE" was in-sample. Proper holdout: ~7.8% random, ~10.0% per-model.
- **SVD rank claim (CORRECTED)**: Rank-2 is optimal (8.1%), not rank-3 (9.1%).
- **Prediction output (FIXED)**: Now outputs all 2,684 missing cells (was 1,712), with clamping to valid ranges.
- **Redundancy sample sizes (FLAGGED)**: 60% of benchmark pairs have <10 shared models. Redundancy table now restricted to n≥20.

### 10.2 Remaining Limitations
- **BenchReg cold-start failure**: The best method cannot predict for models with ≤3 known scores (structural — insufficient shared observations for regression). A production system needs a hybrid.
- **Leave-one-benchmark/provider**: BenchReg produces no predictions in these scenarios.
- **34% fill rate**: The matrix is sparse. Some model-benchmark combinations exist in papers we haven't mined yet.
- **SVD non-convergence**: Soft-Impute with rank-3 does not converge in 100 iterations (final relative diff ~4e-3 vs tol 1e-4). Rank-2 converges faster but the stopping criterion should be tightened.
- **Alpha selection**: α=0.6 was manually swept over 3 values, not nested-CV'd. α=0.7 may be marginally better.
- **Metric naming**: "Per-model MedAPE" is the global median APE across all cells in per-model folds, not the median of per-model medians (which would be ~7.0% for the blend).

### 10.3 Opportunities
- **Temporal dimension**: Adding benchmark vintage (2024 vs 2025 vs 2026) as a feature could improve scaling law predictions
- **Active learning**: The minimum eval set analysis suggests strategic evaluation — choose which benchmark to run next based on maximum expected information gain
- **Ensemble with model metadata**: Model size, training compute, and architecture type as side information for cold-start
- **Non-linear factorization**: The matrix shows non-linear patterns (saturation on easy benchmarks, threshold effects on hard ones) that could be captured by neural matrix completion

---

## Deliverables

| File | Description |
|---|---|
| `results/results_table.csv` | 19 methods × 10 metrics comparison |
| `results/best_predictions.csv` | 2,684 predicted missing cells (full coverage, clamped) |
| `results/latent_factors.csv` | 49 benchmarks × 5 factor loadings |
| `results/report.md` | This report |
| `evaluation_harness.py` | Evaluation framework (6 holdouts, 7 metrics) |
| `methods/all_methods.py` | All 19 prediction methods |
| `diagnostics/all_diagnostics.py` | All diagnostic analyses |
