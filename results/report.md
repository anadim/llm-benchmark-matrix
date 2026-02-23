# LLM Benchmark Matrix Completion: Analysis Report

**Matrix**: 83 models × 49 benchmarks | **1,383 observed** (34.0% fill) | **1,712 predicted**
**Date**: February 23, 2026

---

## 1. Executive Summary

We constructed a cited benchmark matrix spanning 83 instruct/chat LLMs across 49 benchmarks (math, coding, reasoning, knowledge, agentic, multimodal, instruction following, long context). Every entry carries a source URL. We then evaluated 18 prediction methods for filling the 66% of missing cells.

**Key findings:**
- **Best method**: BenchReg+KNN blend (α=0.6) achieves **5.7% MedAPE** on random holdout and **7.4% MedAPE** on per-model leave-50%-out
- **The matrix is surprisingly low-rank**: Factor 1 alone explains 37% of variance; 3 factors capture 57%
- **5 benchmarks predict the rest**: {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA} achieve 4.8% MedAPE, close to the 5.7% baseline using all 49
- **Reasoning mode is the strongest latent factor**: z-score gap of +1.8σ on HMMT, +1.75σ on HLE
- **Cold-start remains hard**: BenchReg-based methods produce no predictions; NMF/Quantile achieve ~13%

---

## 2. Prediction Methods Comparison

### 2.1 Rankings (sorted by Per-Model MedAPE, primary metric)

| Rank | Method | PM-MedAPE | Random-MedAPE | Cold-Start | R² | <10% |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **BenchReg+KNN(α=0.6)** | **7.4%** | **5.7%** | — | 0.979 | 65% |
| 2 | **LogBlend(α=0.65)** | **7.4%** | **5.5%** | — | 0.979 | 65% |
| 3 | BenchReg(k=5,r²≥0.2) | 7.7% | 5.9% | — | 0.966 | 65% |
| 4 | LogBenchReg | 7.7% | 6.3% | — | 0.975 | 63% |
| 5 | Ensemble(avg3) | 8.3% | 7.1% | 19.0% | 0.984 | 60% |
| 6 | SVD(r=3) | 9.1% | 8.0% | 19.9% | 0.984 | 59% |
| 7 | Bench-KNN(k=5) | 9.1% | 8.1% | 19.1% | 0.976 | 56% |
| 8 | KNN(k=5) | 9.3% | 7.6% | 19.1% | 0.976 | 58% |
| 9 | Quantile+SVD5 | 9.6% | 7.3% | **13.0%** | 0.979 | 58% |
| 10 | NucNorm(λ=1) | 9.7% | 8.4% | 19.0% | 0.969 | 56% |
| 11 | Model-Normalized | 10.0% | 9.6% | 16.3% | 0.968 | 51% |
| 12 | PMF(r=5) | 10.3% | 8.5% | 18.9% | 0.974 | 55% |
| 13 | NMF(r=5) | 10.5% | 8.8% | **13.9%** | 0.960 | 56% |
| 14 | SVD(r=5) | 10.9% | 8.8% | 19.2% | 0.985 | 55% |
| 15 | LogSVD(r=5) | 12.4% | 8.7% | 19.5% | 0.977 | 55% |
| 16 | Benchmark Mean | 12.9% | 11.7% | 19.1% | 0.935 | 45% |
| 17 | SVD(r=8) | 13.2% | 9.7% | 19.1% | 0.956 | 51% |
| 18 | SVD(r=10) | 13.4% | 10.3% | 19.1% | 0.935 | 49% |

### 2.2 Key Takeaways

**BenchReg dominates**: The benchmark-regression family (predicting each benchmark from its k=5 most correlated benchmarks via ridge regression) beats all matrix factorization approaches by 1.7–6.0 percentage points. Blending with KNN (α=0.6) or using log-transform provides marginal further improvement.

**SVD rank 3 is optimal**: Higher ranks overfit. Rank 3 beats rank 5/8/10 on per-model holdout, confirming the matrix has low intrinsic dimensionality. This aligns with the PCA analysis (3 factors = 57% variance).

**Cold-start is fundamentally different**: BenchReg methods cannot make cold-start predictions (they need correlated benchmark scores to exist). For cold-start, Quantile+SVD (13.0%) and NMF (13.9%) are best — both leverage the global low-rank structure rather than benchmark-specific correlations.

**Recommendation**: Use BenchReg+KNN(α=0.6) as the primary predictor, falling back to Quantile+SVD for cold-start models (≤3 known scores).

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

The spectrum shows a clear gap between factors 2 and 3 (14.2% → 5.7%), suggesting the matrix is approximately rank-2 with residual structure. To reach various thresholds:

- **80% variance**: rank 11
- **90% variance**: rank 18
- **95% variance**: rank 25

Excluding agentic or bimodal benchmarks barely changes the structure (rank for 80%: 10 vs 11).

### 3.2 Latent Factor Interpretation

**Factor 1 (36.6% — "General Capability")**: Loads most heavily on GPQA Diamond (-0.37), LiveCodeBench (-0.36), MMLU-Pro (-0.31). This captures overall model quality — the strongest models score high across all these benchmarks simultaneously.

**Factor 2 (14.2% — "Frontier Reasoning / Novel Tasks")**: Loads on SimpleQA (+0.34), ARC-AGI-2 (+0.32), HLE (+0.30), FrontierMath (+0.23). Negative on MATH-500 (-0.19), MMLU (-0.18). This factor distinguishes models that excel on genuinely novel, hard reasoning tasks vs. those that do well on established benchmarks. It captures the "reasoning frontier" — models like o3, Claude Opus 4.6, and GPT-5 score high on Factor 2.

**Factor 3 (5.7% — "Established vs. Emerging")**: Loads on MATH-500 (+0.44), SWE-bench (+0.27). Negative on AIME 2025 (-0.41), Arena-Hard (-0.40), Codeforces (-0.31). This separates performance on well-established benchmarks from newer, harder ones.

**Factor 4 (5.2% — "Breadth vs. Depth")**: Loads on MMLU (+0.36), HumanEval (+0.33). Negative on SWE-bench (-0.29), HMMT (-0.27). Models strong on broad knowledge/coding basics vs. deep specialized tasks.

**Factor 5 (4.0% — "Instruction Following")**: Loads heavily on IFEval (+0.49). Negative on MMLU (-0.32), MMLU-Pro (-0.25). An independent axis capturing instruction-following ability, orthogonal to knowledge.

---

## 4. Benchmark Redundancy

### 4.1 Most Redundant Pairs

| Benchmark A | Benchmark B | Correlation |
|---|---|---:|
| MATH-500 | Video-MMU | 0.991 |
| HMMT Feb 2025 | Tau-Bench Retail | 0.986 |
| AA Long Context | SMT 2025 | 0.986 |
| SMT 2025 | LiveBench | 0.973 |
| Tau-Bench Telecom | SMT 2025 | 0.973 |
| Tau-Bench Telecom | BRUMO 2025 | 0.972 |
| ARC-AGI-2 | CritPt | 0.966 |

The MATH-500 ↔ Video-MMU correlation (0.991) is striking — these are nominally different modalities (math vs. multimodal video), yet model rankings are nearly identical. This suggests MATH-500 could serve as a proxy for Video-MMU evaluation.

### 4.2 Anti-Correlated Pairs

| Benchmark A | Benchmark B | Correlation |
|---|---|---:|
| OSWorld | AA Intelligence Index | -0.921 |
| AA Intelligence Index | MathArena Apex | -0.886 |
| ARC-AGI-2 | AA Intelligence Index | -0.784 |

The AA Intelligence Index is strongly anti-correlated with hard benchmarks — models that score well on "intelligence index" style evaluations tend to score *poorly* on cutting-edge math/reasoning. This suggests the index captures different (possibly superficial) capabilities.

### 4.3 Benchmark Clusters

**Cluster 1 (17 benchmarks — "Frontier Reasoning")**: AIME 2025, FrontierMath, HLE, ARC-AGI-2, BrowseComp, SimpleQA, Chatbot Arena Elo, SWE-bench Pro, HMMT, Tau-Bench, CritPt, Terminal-Bench 2.0, ARC-AGI-1, BRUMO, USAMO, MathArena Apex. These all correlate >0.6 — a model good at one tends to be good at all.

**Cluster 2 (8 benchmarks — "Core Competency")**: GPQA Diamond, MMLU-Pro, LiveCodeBench, IFEval, Codeforces, AIME 2024, GSM8K, IFBench. These are the "bread and butter" evaluations.

**Cluster 3 (5 benchmarks — "Coding & Recent Math")**: SWE-bench Verified, AA Long Context, SMT 2025, HMMT Nov 2025, CMIMC 2025.

**Cluster 4 (2 benchmarks — "Agentic Desktop")**: OSWorld, Terminal-Bench 1.0.

---

## 5. Minimum Evaluation Set

### 5.1 Greedy Forward Selection

If you could only run N benchmarks on a new model, which ones should you pick?

| # Benchmarks | Added Benchmark | MedAPE | Improvement |
|---:|---|---:|---:|
| 1 | HLE (Humanity's Last Exam) | 7.5% | — |
| 2 | + AIME 2025 | 6.5% | -1.0 |
| 3 | + LiveCodeBench | 5.9% | -0.6 |
| 4 | + SWE-bench Verified | 5.2% | -0.7 |
| 5 | + SimpleQA | 4.8% | -0.4 |
| 6 | + GPQA Diamond | 4.6% | -0.2 |
| 7 | + Codeforces Rating | 4.4% | -0.2 |
| 8 | + HumanEval | 4.1% | -0.3 |
| 9 | + MMLU-Pro | 3.8% | -0.3 |
| 10 | + AIME 2024 | 3.5% | -0.3 |

**The 5-benchmark minimal eval set is {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA}**. This achieves 4.8% MedAPE — better than the 5.7% random holdout baseline using all 49 benchmarks. This is because the ridge regression with 5 highly informative benchmarks overfits less than BenchReg working from sparse data.

Notably, GPQA Diamond ranks #1 on individual information value (coverage × avg correlation = 49.7) but is not selected first in greedy selection. This is because HLE, despite lower coverage (38 vs 81 models), has higher discriminative power on the frontier — it separates top-tier models more cleanly.

### 5.2 Benchmark Information Value (Top 10)

| Benchmark | Models Covered | Avg |r| | Info Score |
|---|---:|---:|---:|
| GPQA Diamond | 81 | 0.614 | 49.7 |
| LiveCodeBench | 78 | 0.591 | 46.1 |
| AIME 2025 | 61 | 0.641 | 39.1 |
| MMLU-Pro | 69 | 0.556 | 38.3 |
| HumanEval | 74 | 0.495 | 36.6 |
| AIME 2024 | 62 | 0.587 | 36.4 |
| SWE-bench Verified | 58 | 0.597 | 34.6 |
| MMLU | 72 | 0.473 | 34.1 |
| IFEval | 74 | 0.456 | 33.8 |
| MATH-500 | 74 | 0.397 | 29.4 |

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

**Blend is the most data-efficient method**: Even at 10% fill, it achieves 7.2% MedAPE. SVD collapses at 10% fill (18.0%) because it doesn't have enough data for stable rank-3 decomposition.

BenchReg degrades gracefully — its 8.9% at 10% fill is still better than most methods at 34%. This confirms that benchmark-benchmark correlations are robust structural properties of the LLM evaluation landscape, not artifacts of high data density.

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

**Most surprising individual entries** are dominated by ARC-AGI-2 and MathArena Apex — benchmarks where many frontier models score near 0% despite high general capability. These benchmarks are genuinely orthogonal to the main capability axis.

---

## 10. Limitations and Future Work

### 10.1 Known Limitations

- **BenchReg cold-start failure**: The best method cannot predict for models with ≤3 known scores. A production system needs a hybrid: BenchReg when available, Quantile+SVD fallback.
- **Leave-one-benchmark/provider**: BenchReg produces no predictions in these scenarios. This limits its use for transfer learning across providers.
- **34% fill rate**: The matrix is sparse. Some model-benchmark combinations exist in papers we haven't mined yet.
- **Benchmark scale heterogeneity**: Codeforces ratings (0–2800) vs percentages (0–100) require careful normalization. Z-scores help but aren't perfect.

### 10.2 Opportunities

- **Temporal dimension**: Adding benchmark vintage (2024 vs 2025 vs 2026) as a feature could improve scaling law predictions
- **Active learning**: The minimum eval set analysis suggests strategic evaluation — choose which benchmark to run next based on maximum expected information gain
- **Ensemble with model metadata**: Model size, training compute, and architecture type as side information for cold-start
- **Non-linear factorization**: The matrix shows non-linear patterns (saturation on easy benchmarks, threshold effects on hard ones) that could be captured by neural matrix completion

---

## Deliverables

| File | Description |
|---|---|
| `results/results_table.csv` | 18 methods × 10 metrics comparison |
| `results/best_predictions.csv` | 1,712 predicted missing cells from best method |
| `results/latent_factors.csv` | 49 benchmarks × 5 factor loadings |
| `results/report.md` | This report |
| `results/evaluation_harness.py` | Evaluation framework (6 holdouts, 7 metrics) |
| `results/methods/all_methods.py` | All 18 prediction methods |
| `results/diagnostics/all_diagnostics.py` | All diagnostic analyses |
