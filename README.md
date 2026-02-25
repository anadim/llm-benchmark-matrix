# LLM Benchmark Matrix Completion

A cited 83-model x 49-benchmark matrix of LLM evaluation scores, with every entry backed by a source URL. Includes a matrix completion framework comparing 22 prediction methods, finding that **BenchPress** — a logit-space blend of benchmark regression and SVD — achieves **7.2% median error** on held-out scores.

## Quick Predict

```bash
pip install numpy scipy scikit-learn

# Predict a single missing cell
python predict.py --model gpt-5.2 --benchmark simplebench
# -> GPT-5.2 on SimpleBench: 66.9 (predicted)

# Predict all missing scores for a model
python predict.py --model gpt-5.2

# Add your own model with a few known scores, predict the rest
python predict.py --add-model my-model --scores "mmlu=85.0,gpqa_diamond=70.0,aime_2025=55.0"

# List all available models and benchmarks
python predict.py --list-models
python predict.py --list-benchmarks

# Output as JSON
python predict.py --model gpt-5.2 --format json
```

## Matrix Statistics

- **83 models** (OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen, Mistral, xAI, and more)
- **49 benchmarks** (GPQA, AIME, MMLU, SWE-bench, LiveCodeBench, HumanEval, Chatbot Arena, etc.)
- **1,375 observed cells** (33.8% fill rate)
- **2,692 predicted cells** (all unobserved entries, clamped to valid ranges)
- **Every entry has a citation URL** to the original source

## Evaluation Protocol

All results use **per-model leave-50%-out** holdout: for each model, hide 50% of its known scores, train on everything else (all other models keep 100% of their data), predict the hidden cells. Repeat across 3 folds (seed=42). Primary metric is **MedAPE** (median absolute percentage error).

**Metric definitions:**
- **MedAPE** = median( |predicted - actual| / |actual| x 100 ) over all held-out cells
- **MedianAE** = median( |predicted - actual| ) in raw score points
- **Within ±3** = % of predictions within 3 points of actual
- **Within ±5** = % of predictions within 5 points of actual
- **Coverage** = % of held-out cells that received a prediction (see note below)

**Why coverage varies across methods:** Not every method can predict every cell. LogitBenchReg predicts a target benchmark by fitting a regression from the k=5 most correlated benchmarks — but this only works if the model has scores for enough of those correlated benchmarks. When a model has very few known scores, there may not be enough overlap to fit the regression, so LogitBenchReg returns no prediction for that cell (~79% coverage at 50% hiding). SVD-based methods, by contrast, work on the full matrix simultaneously and can predict nearly every cell (~99% coverage). The ~1% SVD misses come from models with so few scores that even the iterative imputation fails to converge. BenchPress inherits SVD's high coverage because it falls back to SVD-Logit whenever LogitBenchReg can't produce a prediction.

---

## BenchPress: The Recommended Method

**BenchPress** = 0.6 x LogitBenchReg + 0.4 x SVD-Logit(rank=2)

A blend of two complementary methods:
1. **LogitBenchReg**: For each target benchmark, fit ridge regression from the k=5 most correlated benchmarks, working in logit space for percentage-scale scores. High accuracy but only ~79% coverage (needs correlated benchmarks to exist).
2. **SVD-Logit (rank=2)**: Soft-impute SVD in logit space. Near-universal coverage (99.2%) with slightly lower accuracy.

The blend fills in LogitBenchReg's coverage gaps with SVD predictions, achieving both high accuracy and near-complete coverage.

### Main Results (50% holdout, seed=42)

| Method | MedAPE | MedianAE | Within ±3 | Within ±5 | Coverage |
|--------|-------:|---------:|----------:|----------:|---------:|
| **BenchPress (0.6/0.4)** | **7.25%** | **4.73** | **36.2%** | **51.5%** | **99.2%** |
| LogitBenchReg (alone) | 6.88% | 4.87 | 36.0% | 50.5% | 78.9% |
| SVD-Logit r=2 (alone) | 7.90% | 5.28 | 33.3% | 48.4% | 99.2% |
| Bench-KNN k=5 | 9.00% | 6.55 | 25.8% | 41.1% | 99.2% |
| KNN k=5 | 9.89% | 6.96 | 27.0% | 40.3% | 99.2% |
| Model-Normalized | 10.18% | 7.35 | 22.8% | 36.1% | 99.2% |
| Benchmark Mean | 13.84% | 9.53 | 16.4% | 27.2% | 99.2% |

---

## SVD Rank Sweep

The rank of the SVD controls model complexity. Rank 2 is optimal for both raw and logit-space SVD — higher ranks overfit the sparse matrix.

### Raw SVD (no logit transform)

| Rank | MedAPE | MedianAE | Within ±3 | Within ±5 |
|-----:|-------:|---------:|----------:|----------:|
| 1 | 9.84% | 7.11 | 24.8% | 38.3% |
| **2** | **8.87%** | **6.15** | **28.8%** | **43.1%** |
| 3 | 9.74% | 6.98 | 24.9% | 38.8% |
| 5 | 11.26% | 7.78 | 22.6% | 34.8% |
| 8 | 13.40% | 9.44 | 18.8% | 30.2% |

### SVD-Logit (logit transform applied)

| Rank | MedAPE | MedianAE | Within ±3 | Within ±5 |
|-----:|-------:|---------:|----------:|----------:|
| 1 | 8.30% | 5.92 | 30.9% | 45.5% |
| **2** | **7.90%** | **5.28** | **33.3%** | **48.4%** |
| 3 | 8.51% | 5.79 | 31.4% | 45.3% |
| 5 | 10.47% | 6.92 | 27.9% | 39.6% |
| 8 | 12.59% | 8.36 | 22.5% | 33.9% |

**The logit transform helps at every rank**: SVD-Logit r=2 (7.90%) vs SVD raw r=2 (8.87%) is an 11% relative improvement. Working in log-odds space linearizes the ceiling/floor effects of percentage-bounded scores.

---

## Blend Ratio Sweep

BenchPress blends LogitBenchReg (alpha) with SVD-Logit r=2 (1-alpha). The optimal range is broad — anything from 0.5 to 0.7 works well.

| Alpha | Blend | MedAPE | MedianAE | Within ±5 |
|------:|-------|-------:|---------:|----------:|
| 0.0 | Pure SVD-Logit | 7.90% | 5.28 | 48.4% |
| 0.1 | 10/90 | 7.84% | 5.23 | 49.0% |
| 0.2 | 20/80 | 7.75% | 5.14 | 49.3% |
| 0.3 | 30/70 | 7.41% | 4.95 | 50.2% |
| 0.4 | 40/60 | 7.41% | 4.96 | 50.2% |
| **0.5** | **50/50** | **7.22%** | **4.82** | **50.8%** |
| **0.6** | **60/40 (default)** | **7.25%** | **4.73** | **51.5%** |
| 0.7 | 70/30 | 7.53% | 4.74 | 51.9% |
| 0.8 | 80/20 | 7.51% | 4.75 | 51.4% |
| 0.9 | 90/10 | 7.43% | 4.79 | 51.8% |
| 1.0 | Pure LogitBenchReg* | 7.68% | 5.00 | 50.0% |

*Alpha=1.0 uses LogitBenchReg where available, falls back to SVD for the ~21% of cells without coverage, so it differs from the standalone LogitBenchReg number (6.88%) which only measures the covered cells.

---

## Hiding Fraction Sweep

How does accuracy change as we hide more or less of each model's data? Less hidden = more training data = better predictions, but also fewer test cells.

### BenchPress

| Hidden | MedAPE | MedianAE | Within ±5 | Coverage |
|-------:|-------:|---------:|----------:|---------:|
| 10% | 6.78% | 4.45 | 54.9% | 99.4% |
| 20% | 6.41% | 4.31 | 56.1% | 99.9% |
| 30% | 5.75% | 3.91 | 57.0% | 99.3% |
| 40% | 6.09% | 4.30 | 55.3% | 99.2% |
| **50%** | **7.25%** | **4.73** | **51.5%** | **99.2%** |

### SVD-Logit r=2

| Hidden | MedAPE | MedianAE | Within ±5 | Coverage |
|-------:|-------:|---------:|----------:|---------:|
| 10% | 6.97% | 5.01 | 50.0% | 99.4% |
| 20% | 6.61% | 4.38 | 54.3% | 99.9% |
| 30% | 6.37% | 4.15 | 56.2% | 99.3% |
| 40% | 7.39% | 4.71 | 51.5% | 99.2% |
| 50% | 7.90% | 5.28 | 48.4% | 99.2% |

### LogitBenchReg

| Hidden | MedAPE | MedianAE | Within ±5 | Coverage |
|-------:|-------:|---------:|----------:|---------:|
| 10% | 6.03% | 4.18 | 56.4% | 93.0% |
| 20% | 6.37% | 4.02 | 56.7% | 89.1% |
| 30% | 5.48% | 3.73 | 58.9% | 87.7% |
| 40% | 5.82% | 4.14 | 55.5% | 84.7% |
| 50% | 6.88% | 4.87 | 50.5% | 78.9% |

**Key insight**: LogitBenchReg has the lowest error at every hiding fraction, but its coverage degrades sharply as you hide more data (93% -> 79%). BenchPress maintains ~99% coverage throughout by filling gaps with SVD.

---

## Full 19-Method Comparison

Sorted by per-model MedAPE (primary metric). All methods evaluated on identical holdout folds.

| # | Method | PM-MedAPE | Rand-MedAPE |
|--:|--------|----------:|------------:|
| 1 | BenchReg (k=5) | 7.69% | 5.86% |
| 2 | LogBenchReg | 7.72% | 6.27% |
| 3 | BenchReg+KNN (alpha=0.6) | 7.88% | 6.43% |
| 4 | LogBlend (alpha=0.65) | 7.97% | 6.32% |
| 5 | SVD r=2 | 8.09% | 7.32% |
| 6 | Ensemble (avg3) | 8.30% | 7.09% |
| 7 | SVD r=3 | 9.07% | 8.02% |
| 8 | Bench-KNN k=5 | 9.10% | 8.07% |
| 9 | KNN k=5 | 9.32% | 7.63% |
| 10 | Quantile+SVD5 | 9.64% | 7.31% |
| 11 | Nuclear Norm | 9.70% | 8.44% |
| 12 | Model-Normalized | 10.00% | 9.61% |
| 13 | PMF r=5 | 10.29% | 8.48% |
| 14 | NMF r=5 | 10.46% | 8.84% |
| 15 | SVD r=5 | 10.95% | 8.79% |
| 16 | LogSVD r=5 | 12.42% | 8.75% |
| 17 | Benchmark Mean | 12.89% | 11.71% |
| 18 | SVD r=8 | 13.18% | 9.71% |
| 19 | SVD r=10 | 13.40% | 10.33% |

Note: BenchReg/LogBenchReg have <100% coverage. BenchPress (not shown here — evaluated separately with extended metrics) combines LogitBenchReg + SVD-Logit for full coverage at 7.25% MedAPE.

---

## Claude as Predictor

We tested whether LLMs can predict benchmark scores by giving Claude the full matrix as CSV context and asking it to fill in held-out cells. All Claude runs use 20% holdout (seed=42).

**Important caveat**: Some models in the matrix were released before Claude's training cutoff, so Claude may "know" certain scores from its training data. The pre-audit run (5.33%) was on a matrix that contained some data entry errors — Claude may have memorized the correct values. The post-audit run (6.08%) is on the corrected matrix and is the fairer number.

### Claude Models

| Model | MedAPE | Within ±3 | Within ±5 | Coverage | Cost |
|-------|-------:|----------:|----------:|---------:|-----:|
| Sonnet 4.5 (pre-audit matrix) | 5.33% | 48.9% | 62.3% | 100% | $0.84 |
| Sonnet 4.5 (row only) | 6.58% | 40.9% | 54.7% | 100% | $0.67 |
| **Sonnet 4.5 (post-audit matrix)** | **6.08%** | **41.5%** | **56.4%** | **100%** | **$0.86** |
| Opus 4 (no thinking) | 7.73% | 39.6% | 48.9% | 81.5% | $3.36 |
| Opus 4 (with thinking) | 9.01% | 33.9% | 46.1% | 59.8% | $15.29 |

### Fair Comparison: Claude vs BenchPress (same post-audit holdout)

| Method | MedAPE | Within ±5 | Coverage |
|--------|-------:|----------:|---------:|
| **BenchPress** | **5.81%** | **58.2%** | **100%** |
| Claude Sonnet 4.5 | 6.08% | 56.4% | 100% |
| BenchReg | 6.35% | 51.5% | 84.0% |
| Benchmark Mean | 11.91% | 25.5% | 100% |

On the same post-audit data, **BenchPress edges out Claude** (5.81% vs 6.08%). BenchPress also runs in <1 second and costs nothing, while Claude takes ~6 minutes and ~$0.85 per run.

### Phase Transition: World Knowledge vs Linear Algebra

The more interesting comparison is how they behave as you vary the amount of data. Take a model, hide all its scores, then reveal them one at a time:

| Known scores | BenchPress | Claude | Winner |
|-------------:|-----------:|-------:|--------|
| 0 (name only) | 16.8% | 2.5% | Claude |
| 1 | 12.0% | 2.9% | Claude |
| 2 | 7.2% | 3.2% | Claude |
| 3 | 4.9% | 3.7% | Claude |
| 5 | 4.9% | 4.7% | ~Tie |
| 7 | 3.7% | 4.9% | BenchPress |
| 10 | 2.7% | 4.2% | BenchPress |
| 15 | 2.4% | 4.7% | BenchPress |

*(MedAPE averaged over Gemini 3.1 Pro and Claude Sonnet 4.6, 3 random trials each)*

At k=0 (just the model's name), Claude already predicts within 2.5% — its world knowledge is worth about 5 benchmark scores of information. But by k=5 BenchPress catches up, and beyond that linear algebra wins decisively. Interestingly, Claude's accuracy *plateaus* around 4-5% regardless of how much data it sees, suggesting that providing partial data via CSV creates a conflict between Claude's prior knowledge and the prompt context.

---

## Per-Benchmark Predictability

Some benchmarks are much easier to predict than others. Ranked by typical error (median absolute error in points):

| Benchmark | Typical Error | Predictability |
|-----------|-------------:|----------------|
| MATH-500 | 1.3 pts | Very Easy |
| MMMU | 2.3 pts | Easy |
| MMLU | 2.8 pts | Easy |
| IFEval | 2.9 pts | Easy |
| HumanEval | 2.9 pts | Easy |
| SWE-bench Verified | 4.0 pts | Moderate |
| MMLU-Pro | 4.1 pts | Moderate |
| GPQA Diamond | 4.2 pts | Moderate |
| AIME 2025 | 4.5 pts | Moderate |
| LiveCodeBench | 5.3 pts | Moderate |
| FrontierMath | 6.7 pts | Hard |
| HLE | 7.3 pts | Hard |
| SWE-bench Pro | 8.5 pts | Hard |
| Terminal-Bench 2.0 | 9.3 pts | Very Hard |
| SimpleQA | 10.2 pts | Very Hard |
| ARC-AGI-1 | 10.7 pts | Very Hard |
| OSWorld | 10.8 pts | Very Hard |
| Arena-Hard Auto | 19.3 pts | Unpredictable |

**Easy to predict** (< 3 pts): Saturated or well-correlated benchmarks where knowing a few other scores pins down a model precisely. **Hard to predict** (> 9 pts): Novel agentic tasks and preference-based rankings with less cross-benchmark signal.

---

## Minimum Evaluation Set

5 benchmarks are sufficient to predict the other 44 with ~7.8% MedAPE:

1. **HLE** (Humanity's Last Exam) — frontier reasoning
2. **AIME 2025** — math competition
3. **LiveCodeBench** — coding
4. **SWE-bench Verified** — agentic coding
5. **SimpleQA** — factual accuracy

These span the major capability axes and provide maximal information for matrix completion.

---

## Adding Your Own Data

### Add a model

To predict scores for a model not in the matrix, provide a few known benchmark scores:

```bash
python predict.py --add-model my-model \
  --scores "mmlu=85.0,gpqa_diamond=70.0,aime_2025=55.0,humaneval=90.0,swe_bench_verified=60.0"
```

The tool finds the 5 most similar existing models and uses their prediction patterns to estimate your model's scores on all 49 benchmarks.

### Add data permanently

To add scores to the matrix itself, append entries to `data/build_benchmark_matrix.py`:

```python
# In the DATA list, add:
("my-model-id", "benchmark_id", score, "https://source-url.com"),
```

Then re-run `python predict.py` to get updated predictions using the new data.

### Add a benchmark

To add a new benchmark, add it to the `BENCHMARKS` list in `data/build_benchmark_matrix.py`:

```python
("my_bench_id", "My Benchmark Name", "Category", "accuracy", 100, "https://benchmark-url.com"),
```

Then add scores for existing models in the `DATA` list.

## Repository Structure

```
llm-benchmark-matrix/
├── predict.py                  # CLI: predict scores, add models/benchmarks
├── evaluation_harness.py       # Core: 6 holdout strategies, 7 metrics
├── data/
│   ├── build_benchmark_matrix.py   # Master data: 83 models, 49 benchmarks, 1375 scores
│   ├── llm_benchmark_matrix.xlsx   # Pre-built Excel spreadsheet (5 sheets)
│   ├── llm_benchmark_data.json     # Machine-readable JSON export
│   └── extra_scores_{1-5}.py       # Supplementary score batches with citations
├── methods/
│   ├── all_methods.py              # 22 prediction methods (incl. BenchPress)
│   ├── full_evaluation.py          # BenchPress comprehensive evaluation
│   ├── run_final_eval.py           # Extended metrics + phase transition
│   ├── creative_methods.py         # Round 1 exploration (15 methods)
│   ├── creative_methods_r2.py      # Round 2 combinations (10 methods)
│   └── creative_methods_r3.py      # Round 3 optimization (9 methods)
├── figures/
│   ├── sparsity_heatmap.png        # Matrix fill pattern
│   ├── svd_complete_submatrix.png  # Singular value spectrum
│   ├── phase_transition_per_model.png
│   ├── claude_vs_algorithm.png     # Claude vs BenchPress comparison
│   └── ...                         # 11 publication-ready figures
├── analysis/
│   ├── walkthrough_single_prediction.py  # Step-by-step prediction walkthrough
│   ├── component_analysis.py             # Component-level evaluation
│   └── ...
├── results/
│   ├── full_evaluation.json        # All experiments: ranks, blends, fractions, Claude
│   ├── best_predictions.csv        # 2,692 predicted missing cells
│   ├── results_table.csv           # 19 methods x 10 metrics comparison
│   ├── phase_transition.csv        # Accuracy vs number of known scores
│   ├── latent_factors.csv          # SVD factor loadings
│   └── report.md                   # Full analysis report
├── report/
│   ├── report.tex                  # Academic report (LaTeX)
│   └── report.pdf                  # Academic report (PDF)
└── requirements.txt
```

## Reproducing Results

```bash
pip install -r requirements.txt

# 1. Load data and print matrix stats
python evaluation_harness.py

# 2. Run all 22 methods and produce results_table.csv
python methods/all_methods.py

# 3. Run BenchPress comprehensive evaluation (ranks, blends, fractions, Claude)
python methods/full_evaluation.py

# 4. Run extended metrics + phase transition
python methods/run_final_eval.py

# 5. Run diagnostics (dimensionality, redundancy, minimum eval set)
python diagnostics/all_diagnostics.py

# 6. Generate all figures
python figures/generate_figures.py
```

## Data Format

Each score in `data/build_benchmark_matrix.py` is a 4-tuple:
```python
(model_id, benchmark_id, score, source_url)
```

Example:
```python
("gpt-4.1", "mmlu", 90.4, "https://openai.com/index/gpt-4-1/")
```

## License

MIT
