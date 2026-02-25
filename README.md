# LLM Benchmark Matrix Completion

A cited 83-model × 49-benchmark matrix of LLM evaluation scores, with every entry backed by a source URL. Includes a matrix completion framework comparing 22 prediction methods, finding that a logit-space blend of benchmark regression and SVD achieves **7.2% median error** on held-out scores.

## Quick Predict

```bash
pip install numpy scipy scikit-learn

# Predict a single missing cell
python predict.py --model gpt-5.2 --benchmark simplebench
# → GPT-5.2 on SimpleBench: 66.9 (predicted)

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
- **1,375 observed cells** (34% fill rate)
- **2,692 predicted cells** (all unobserved entries, clamped to valid ranges)
- **Every entry has a citation URL** to the original source

## Key Results

| Method | PM-MedAPE | R-MedAPE | MAE | ±3pts | BiAcc | Coverage |
|--------|----------:|---------:|----:|------:|------:|---------:|
| **LogitSVD Blend (0.6/0.4)** | **7.25%** | **6.12%** | **4.71** | **36.5%** | **94.2%** | **99.7%** |
| LogitBenchReg | 6.90% | 6.01% | 4.88 | 35.9% | 74.8% | 79.1% |
| BenchReg+KNN (α=0.6) | 8.08% | 6.91% | 5.80 | 31.1% | 92.2% | 99.5% |
| SVD-Logit (r=2) | 7.62% | 6.98% | 5.09 | 34.9% | 93.7% | 99.5% |
| SVD (r=2) | 8.53% | 7.97% | 6.03 | 28.4% | 91.2% | 99.5% |
| Benchmark Mean | 13.89% | 13.57% | 9.51 | 16.2% | 89.2% | 99.5% |

**The logit transform is the single biggest win**: Working in log-odds space for percentage benchmarks improves BenchReg by 11% relative and SVD by 13% relative. Replacing KNN with SVD-Logit as the blend partner adds another 14% improvement. Model metadata (provider, params, reasoning mode) adds zero signal.

**Minimum eval set**: 5 benchmarks (HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA) predict the other 44 with ~7.8% MedAPE on proper holdout.

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
│   ├── all_methods.py              # 22 prediction methods (incl. LogitSVD variants)
│   ├── run_final_eval.py           # LogitSVD evaluation + phase transition + predictions
│   ├── creative_methods.py         # Round 1 exploration (15 methods)
│   ├── creative_methods_r2.py      # Round 2 combinations (10 methods)
│   └── creative_methods_r3.py      # Round 3 optimization (9 methods)
├── figures/
│   ├── sparsity_heatmap.png        # Matrix fill pattern
│   ├── svd_complete_submatrix.png  # Singular value spectrum
│   ├── phase_transition_per_model.png
│   ├── claude_vs_algorithm.png     # Claude vs LogitSVD comparison
│   └── ...                         # 11 publication-ready figures
├── analysis/
│   ├── walkthrough_single_prediction.py  # Step-by-step prediction walkthrough
│   └── ...
├── results/
│   ├── best_predictions.csv        # 2,692 predicted missing cells
│   ├── results_table.csv           # 19 methods × 10 metrics comparison
│   ├── phase_transition.csv        # Accuracy vs number of known scores
│   ├── latent_factors.csv          # SVD factor loadings
│   └── report.md                   # Full analysis report
├── report/
│   ├── blog_post.tex               # Blog post (LaTeX source)
│   └── blog_post.pdf               # Blog post (PDF)
└── requirements.txt
```

## Reproducing Results

```bash
pip install -r requirements.txt

# 1. Load data and print matrix stats
python evaluation_harness.py

# 2. Run all 22 methods and produce results_table.csv
python methods/all_methods.py

# 3. Run LogitSVD evaluation with extended metrics + phase transition
python methods/run_final_eval.py

# 4. Run diagnostics (dimensionality, redundancy, minimum eval set)
python diagnostics/all_diagnostics.py

# 5. Generate all figures
python figures/generate_figures.py
```

## Prediction Methods

**Recommended (LogitSVD family):**
- **LogitSVD Blend (0.6/0.4)**: 0.6 × LogitBenchReg + 0.4 × SVD-Logit(r=2) — best full-coverage method
- LogitBenchReg: BenchReg in logit space — lowest raw MedAPE but 79% coverage
- SVD-Logit(r=2): Soft-impute SVD in logit space — best standalone full-coverage

**Baselines:**
- B0: Benchmark mean
- B1: Model-normalized (z-score matching)
- B2: KNN (k=5 cosine similarity)
- B3: Bench-KNN (k=5 benchmark similarity)

**Matrix factorization:**
- SVD (soft-impute, ranks 2/3/5/8/10), NMF, PMF, Nuclear norm

**Regression-based:**
- BenchReg: Per-target ridge from k=5 most correlated benchmarks
- BenchReg+KNN blend (α=0.6) — previous best

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
