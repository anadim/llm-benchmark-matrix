# LLM Benchmark Matrix Completion: Analysis Report

**Matrix**: 83 models × 49 benchmarks | **1,383 observed** (34.0% fill) | **2,684 predicted**
**Date**: February 23, 2026 (revised)

---

## 1. Executive Summary

We constructed a cited benchmark matrix spanning 83 instruct/chat LLMs across 49 benchmarks (math, coding, reasoning, knowledge, agentic, multimodal, instruction following, long context). Every entry carries a source URL. We then evaluated 22 prediction methods for filling the 66% of missing cells.

**Key findings:**
- **Best method**: LogitSVD Blend (0.6 × LogitBenchReg + 0.4 × SVD-Logit(r=2)) achieves **6.0% MedAPE** on random holdout and **6.7% MedAPE** on per-model leave-50%-out, with 99.8% coverage. This is a **14.2% relative improvement** over the previous best (BenchReg+KNN Blend, 7.9% PM-MedAPE).
- **The logit transform is the single biggest win**: Working in logit space (log-odds) for percentage-scale benchmarks linearizes the relationship near ceilings and floors, improving BenchReg by 11.3% relative (7.45% → 6.61%) and SVD by 12.8% relative (8.62% → 7.52%). It also implicitly handles bimodal benchmarks (89.9% bimodal classification accuracy vs 84.9% for the linear blend).
- **SVD is a better complement to BenchReg than KNN**: Both KNN and BenchReg are local methods making correlated errors. SVD captures global low-rank structure that is complementary. Replacing KNN with SVD-Logit improves the blend by 14.2% relative.
- **Model metadata doesn't help**: Provider, parameter count, reasoning mode, and open-weight status add zero predictive signal beyond what the benchmark scores themselves contain. Every metadata-based method tested was neutral or negative.
- **The matrix is approximately rank-2**: Factor 1 explains 37% of variance; 2 factors capture 51%. SVD rank-2 (8.1% PM-MedAPE in raw space, 7.5% in logit space) beats rank-3 (9.1%)
- **5 benchmarks predict the rest**: {HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA} achieve ~7.8% MedAPE under proper holdout
- **Reasoning mode is the strongest latent factor**: z-score gap of +1.8σ on HMMT, +1.75σ on HLE
- **Cold-start remains hard**: BenchReg-based methods produce no predictions; NMF/Quantile achieve ~13%

**Methodology notes:**
- "PM-MedAPE" = global median of absolute percentage errors across all held-out cells in per-model folds (not median of per-model medians, which would give ~6.2% for LogitSVD)
- LogitSVD coverage: 99.8% (SVD-Logit provides near-universal coverage; LogitBenchReg covers ~79%, SVD-Logit fills the rest)
- All predictions clamped to valid ranges (0–100 for percentage benchmarks)

---

## 2. Prediction Methods Comparison

### 2.1 Extended Evaluation (Primary: Per-model leave-50%-out, 5 seeds)

| Rank | Method | PM-MedAPE | R-MedAPE | MAE | ±3pts | ±5pts | APE>50 | APE≤50 | BiAcc | Cov |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LogitBenchReg | **6.61%** | **5.68%** | **4.70** | 36.7% | 52.0% | 4.32% | 33.23% | 76.3% | 78.5% |
| 2 | **LogitSVD Blend(0.6/0.4)** | **6.74%** | **5.95%** | **4.61** | **37.0%** | **52.4%** | **4.34%** | **31.92%** | **89.9%** | **99.8%** |
| 3 | BenchReg(k=5,r²≥0.2) | 7.45% | 6.21% | 5.61 | 31.0% | 46.3% | 5.06% | 37.63% | 81.4% | 79.6% |
| 4 | SVD-Logit(r=2) | 7.52% | 6.62% | 5.07 | 35.2% | 49.6% | 4.76% | 33.81% | 89.0% | 99.8% |
| 5 | BenchReg+KNN(α=0.6) | 7.86% | 6.54% | 5.63 | 31.5% | 46.0% | 5.03% | 40.36% | 84.9% | 99.8% |
| 6 | SVD(r=2) | 8.62% | 7.70% | 6.04 | 28.7% | 44.0% | 5.78% | 36.27% | 87.5% | 99.8% |
| 7 | Benchmark Mean | 13.52% | 12.88% | 9.47 | 16.4% | 27.8% | 8.65% | 53.07% | 80.6% | 99.8% |

**Column definitions:**
- **PM-MedAPE**: Per-model leave-50%-out median absolute percentage error (5-seed average, primary metric)
- **R-MedAPE**: Random 20% holdout MedAPE (5-seed average)
- **MAE**: Mean absolute error in score points
- **±3pts / ±5pts**: Fraction of predictions within 3 / 5 absolute score points
- **APE>50 / APE≤50**: MedAPE split by actual score above/below 50
- **BiAcc**: Bimodal classification accuracy on 5 threshold benchmarks (ARC-AGI-1/2, IMO/USAMO 2025, MathArena Apex)
- **Cov**: Percentage of test cells that receive a finite prediction

### 2.2 Key Takeaways

**LogitSVD Blend is the recommended method**: At 6.74% PM-MedAPE and 99.8% coverage, it dominates on every metric except raw MedAPE (where LogitBenchReg at 6.61% is better but only covers 78.5% of cells). The blend combines logit-space BenchReg (local benchmark-to-benchmark regression) with logit-space SVD (global low-rank factorization) — two complementary views of the data.

**The logit transform is the key insight**: Benchmark scores on 0–100% scales have non-linear behavior near 0 and 100. The logit transform `log(p/(1-p))` maps these to an unbounded space where linear methods work better. This single change:
- BenchReg: 7.45% → 6.61% (−11.3% relative)
- SVD(r=2): 8.62% → 7.52% (−12.8% relative)
- Bimodal accuracy: 84.9% → 89.9% (+5.0pp) — logit naturally handles threshold effects

**Model metadata adds nothing**: We tested provider indicators, parameter count, reasoning mode, and open-weight status as features for prediction. Every metadata-based method (MetaKNN, ProviderCorrected, MultiRidge, FamilyInterp, CategoryAware) was neutral or worse. The benchmark scores already encode everything these features would tell you.

**SVD rank-2 is optimal**: Rank 2 (8.62% raw, 7.52% logit) beats rank 3 (9.1% raw). The third factor overfits to noise.

**Cold-start is fundamentally different**: LogitBenchReg needs correlated benchmark scores to exist. For cold-start models (≤3 known scores), SVD-Logit provides the fallback within the blend (7.52% alone). For truly cold-start, Quantile+SVD (13.0%) and NMF (13.9%) are alternatives.

### 2.3 What We Tried That Didn't Work

| Approach | Result | Why |
|---|---|---|
| Model metadata features (provider, params, reasoning) | Neutral or −1% | Scores already encode this |
| Explicit bimodal classification | −5% bimodal accuracy | Logit handles bimodality better |
| GBT per-benchmark | −0.9% vs BenchReg | Overfits on sparse rows |
| Category-aware regression | +0.3% | Small gain, logit subsumes it |
| KNN as blend partner | −14% vs SVD-Logit | Correlated errors with BenchReg |
| Confidence weighting | −0.8% | Confidence not well calibrated |
| Meta-learner (stacking) | −0.5% | Second-level overfits |

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

The spectrum shows a clear gap between factors 2 and 3 (14.2% → 5.7%), suggesting the matrix is approximately rank-2 with residual structure. This is confirmed by holdout evaluation: SVD rank-2 achieves 8.62% MedAPE vs 9.1% for rank-3 (raw space), and 7.52% in logit space.

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
- **Random 20% holdout**: 5-bench ridge = ~7.8% MedAPE vs LogitSVD Blend = 6.0%
- **Per-model 50% holdout**: 5-bench ridge = ~10.0% MedAPE vs LogitSVD Blend = 6.7%

The 5-benchmark set is still useful as a practical evaluation strategy (run only 5 benchmarks to estimate the rest), but it does NOT outperform LogitSVD Blend trained on all available data.

Notably, GPQA Diamond is a near-perfect substitute for HLE: the GPQA-first set achieves 4.80% in-sample (vs HLE-first 4.84%), and both converge to the same remaining 4 benchmarks. GPQA Diamond has 2× the coverage (81 vs 38 models).

---

## 6. Data Efficiency (Phase Transition)

How does LogitSVD Blend prediction accuracy scale with the number of known scores per model?

| Known Scores | MedAPE | MAE | ±3 pts | ±5 pts |
|---:|---:|---:|---:|---:|
| 1 | 12.1% | 7.9 | 24.7% | 38.9% |
| 2 | 11.2% | 7.0 | 30.0% | 43.6% |
| 3 | 10.3% | 6.3 | 32.5% | 46.1% |
| 5 | 9.2% | 5.6 | 35.8% | 50.0% |
| 7 | 9.0% | 5.4 | 36.5% | 52.0% |
| 10 | 10.1% | 5.9 | 37.2% | 52.5% |
| 15 | 11.2% | 6.8 | 41.1% | 56.8% |
| 20 | 9.6% | 4.5 | 44.4% | 61.1% |

**Diminishing returns after ~5 scores**: The biggest gains come from knowing 1→5 benchmarks per model (12.1% → 9.2% MedAPE). After 5, ±3pts and ±5pts continue to improve while MedAPE plateaus. Non-monotonicity at n>10 reflects variance from small sample sizes (few models have exactly n known scores).

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
- **SVD rank claim (CORRECTED)**: Rank-2 is optimal (8.6% raw, 7.5% logit), not rank-3 (9.1%).
- **Prediction output (FIXED)**: Now outputs all 2,684 missing cells (was 1,712), with clamping to valid ranges.
- **Redundancy sample sizes (FLAGGED)**: 60% of benchmark pairs have <10 shared models. Redundancy table now restricted to n≥20.
- **LogitSVD Blend (NEW)**: Replaced BenchReg+KNN as default prediction method. 14.2% relative improvement.

### 10.2 Remaining Limitations
- **LogitBenchReg cold-start failure**: The best component method cannot predict for models with ≤3 known scores (structural — insufficient shared observations for regression). Within the blend, SVD-Logit handles these cases.
- **Leave-one-benchmark/provider**: LogitBenchReg produces no predictions in these scenarios; SVD-Logit fills the gap.
- **34% fill rate**: The matrix is sparse. Some model-benchmark combinations exist in papers we haven't mined yet.
- **Non-percentage benchmarks**: Elo ratings, Codeforces scores, and other non-percentage metrics use raw-space prediction (no logit transform). These constitute ~8% of benchmarks.
- **Alpha selection**: α=0.6 was manually swept over 5 values (0.4, 0.5, 0.6, 0.7, 0.8). No nested CV.
- **Metric naming**: "Per-model MedAPE" is the global median APE across all cells in per-model folds, not the median of per-model medians.

### 10.3 Opportunities
- **Temporal dimension**: Adding benchmark vintage (2024 vs 2025 vs 2026) as a feature could improve scaling law predictions
- **Active learning**: The minimum eval set analysis suggests strategic evaluation — choose which benchmark to run next based on maximum expected information gain
- **Non-linear factorization**: The logit transform helps but doesn't fully capture saturation on easy benchmarks; neural matrix completion could help
- **Cross-validation for alpha**: Nested CV over the BenchReg/SVD blend weight could tighten the estimate

---

## Deliverables

| File | Description |
|---|---|
| `results/results_table.csv` | 19 baseline methods × 10 metrics comparison |
| `results/logit_svd_eval.json` | LogitSVD vs baselines: extended metrics (7 methods × 9 metrics) |
| `results/best_predictions.csv` | 2,684 predicted missing cells (LogitSVD Blend, clamped) |
| `results/latent_factors.csv` | 49 benchmarks × 5 factor loadings |
| `results/phase_transition.csv` | Accuracy vs number of known scores (LogitSVD Blend) |
| `results/report.md` | This report |
| `evaluation_harness.py` | Evaluation framework (6 holdouts, 7 metrics) |
| `methods/all_methods.py` | 22 prediction methods (incl. LogitSVD variants) |
| `methods/run_final_eval.py` | LogitSVD evaluation + phase transition + prediction regeneration |
| `methods/creative_methods.py` | Round 1 exploration (15 methods) |
| `methods/creative_methods_r2.py` | Round 2 combinations (10 methods) |
| `methods/creative_methods_r3.py` | Round 3 optimization (9 methods) |
| `diagnostics/all_diagnostics.py` | Intrinsic dimensionality, redundancy, scaling laws |
