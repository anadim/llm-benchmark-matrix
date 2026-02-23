# LLM Benchmark Matrix Completion

A cited 83-model × 49-benchmark matrix of LLM evaluation scores, with every entry backed by a source URL. Includes a matrix completion framework comparing 22 prediction methods, finding that a logit-space blend of benchmark regression and SVD achieves **6.7% median error** on held-out scores.

## Matrix Statistics

- **83 models** (OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen, Mistral, xAI, and more)
- **49 benchmarks** (GPQA, AIME, MMLU, SWE-bench, LiveCodeBench, HumanEval, Chatbot Arena, etc.)
- **1,383 observed cells** (34% fill rate)
- **2,684 predicted cells** (all unobserved entries, clamped to valid ranges)
- **Every entry has a citation URL** to the original source

## Key Results

| Method | PM-MedAPE | R-MedAPE | MAE | ±3pts | BiAcc | Coverage |
|--------|----------:|---------:|----:|------:|------:|---------:|
| **LogitSVD Blend (0.6/0.4)** | **6.74%** | **5.95%** | **4.61** | **37.0%** | **89.9%** | **99.8%** |
| LogitBenchReg | 6.61% | 5.68% | 4.70 | 36.7% | 76.3% | 78.5% |
| BenchReg+KNN (α=0.6) | 7.86% | 6.54% | 5.63 | 31.5% | 84.9% | 99.8% |
| SVD-Logit (r=2) | 7.52% | 6.62% | 5.07 | 35.2% | 89.0% | 99.8% |
| SVD (r=2) | 8.62% | 7.70% | 6.04 | 28.7% | 87.5% | 99.8% |
| Benchmark Mean | 13.52% | 12.88% | 9.47 | 16.4% | 80.6% | 99.8% |

**The logit transform is the single biggest win**: Working in log-odds space for percentage benchmarks improves BenchReg by 11% relative and SVD by 13% relative. Replacing KNN with SVD-Logit as the blend partner adds another 14% improvement. Model metadata (provider, params, reasoning mode) adds zero signal.

**Minimum eval set**: 5 benchmarks (HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA) predict the other 44 with ~7.8% MedAPE on proper holdout.

## Repository Structure

```
llm-benchmark-matrix/
├── evaluation_harness.py        # Core: 6 holdout strategies, 7 metrics
├── matrix_completion_v8.py      # Original prediction engine with confidence intervals
├── data/
│   ├── build_benchmark_matrix.py    # Master data: 83 models, 49 benchmarks, 1383 scores
│   ├── llm_benchmark_matrix.xlsx    # Pre-built Excel spreadsheet (5 sheets)
│   ├── merge_extra_scores.py        # Tool to merge additional scores
│   └── extra_scores_{1-5}.py        # Supplementary score batches with citations
├── methods/
│   ├── all_methods.py               # 22 prediction methods (incl. LogitSVD variants)
│   ├── run_final_eval.py            # LogitSVD evaluation + phase transition + predictions
│   ├── creative_methods.py          # Round 1 exploration (15 methods)
│   ├── creative_methods_r2.py       # Round 2 combinations (10 methods)
│   ├── creative_methods_r3.py       # Round 3 optimization (9 methods)
│   └── final_comparison.py          # All 19 methods comprehensive comparison
├── diagnostics/
│   └── all_diagnostics.py           # Intrinsic dimensionality, redundancy, scaling laws
├── analysis/
│   ├── walkthrough_single_prediction.py  # Step-by-step BenchReg+KNN walkthrough
│   ├── sanity_checks.py                  # SVD convergence, negative predictions, etc.
│   ├── five_bench_spot_check.py          # 5-benchmark minimum eval set predictions
│   ├── investigate_q4_q5.py              # BenchReg vs 5-bench ridge comparison
│   ├── q10_benchreg_coldstart.py         # Cold-start failure investigation
│   └── phase_transition.py               # How accuracy improves with more known scores
└── results/
    ├── results_table.csv            # 19 baseline methods × 10 metrics comparison
    ├── logit_svd_eval.json          # LogitSVD extended evaluation results
    ├── latent_factors.csv           # SVD factor loadings for 49 benchmarks
    ├── best_predictions.csv         # 2,684 predicted missing cells (LogitSVD Blend, clamped)
    ├── five_bench_predictions.csv   # 665 LOO predictions from 5-benchmark set
    ├── phase_transition.csv         # Accuracy vs number of known scores (LogitSVD Blend)
    └── report.md                    # Full analysis report
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the evaluation harness (loads data, prints matrix stats)
python evaluation_harness.py

# Run all 22 methods and produce results_table.csv
python methods/all_methods.py

# Run LogitSVD evaluation with extended metrics + phase transition + predictions
python methods/run_final_eval.py

# Run diagnostics (intrinsic dimensionality, redundancy, minimum eval set, etc.)
python diagnostics/all_diagnostics.py

# Walk through a single prediction step-by-step
python analysis/walkthrough_single_prediction.py
```

## Prediction Methods

**Recommended (LogitSVD family):**
- **LogitSVD Blend (0.6/0.4)**: 0.6 × LogitBenchReg + 0.4 × SVD-Logit(r=2) — best full-coverage method
- LogitBenchReg: BenchReg in logit space — lowest raw MedAPE but 78.5% coverage
- SVD-Logit(r=2): Soft-impute SVD in logit space — best standalone full-coverage

**Baselines:**
- B0: Benchmark mean
- B1: Model-normalized (z-score matching)
- B2: KNN (k=5 cosine similarity)
- B3: Bench-KNN (k=5 benchmark similarity)

**Matrix factorization:**
- SVD (soft-impute, ranks 2/3/5/8/10)
- NMF, PMF, Nuclear norm minimization

**Regression-based:**
- BenchReg: Per-target ridge from k=5 most correlated benchmarks
- BenchReg+KNN blend (α=0.6) — previous best
- Log-space variants

**Ensemble:**
- Average of top-3 methods

## Holdout Strategies

1. **Per-model leave-50%-out** (primary) — 5 seeds, global MedAPE
2. **Random cells** (20%) — 5 seeds, standard matrix completion
3. **Cold-start** (3 known scores per new model) — realistic deployment scenario
4. **Leave-one-benchmark** — can we predict an entire benchmark?
5. **Stratified difficulty** — balanced across easy/hard benchmarks
6. **Leave-one-provider** — generalization across model families

## Data Format

Each score in `build_benchmark_matrix.py` is a 4-tuple:
```python
(model_id, benchmark_id, score, source_url)
```

Example:
```python
("gpt-4.1", "mmlu", 90.4, "https://openai.com/index/gpt-4-1/")
```

## Citation

If you use this dataset or framework, please cite this repository.

## License

MIT
