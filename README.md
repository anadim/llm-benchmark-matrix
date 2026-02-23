# LLM Benchmark Matrix Completion

A cited 83-model x 49-benchmark matrix of LLM evaluation scores, with every entry backed by a source URL. Includes a full matrix completion evaluation framework comparing 19 prediction methods across 6 holdout strategies.

## Matrix Statistics

- **83 models** (OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen, Mistral, xAI, and more)
- **49 benchmarks** (GPQA, AIME, MMLU, SWE-bench, LiveCodeBench, HumanEval, Chatbot Arena, etc.)
- **1,383 observed cells** (34% fill rate)
- **Every entry has a citation URL** to the original source

## Key Results

| Method | Per-Model MedAPE | R² | % Within 10% |
|--------|----------------:|---:|-------------:|
| BenchReg+KNN (alpha=0.6) | 7.9% | 0.979 | 56.0% |
| LogBlend (alpha=0.65) | 8.0% | 0.977 | 55.9% |
| BenchReg (k=5, r²>=0.2) | 7.7% | 0.976 | 57.9% |
| SVD (rank=2) | 8.1% | 0.979 | 54.9% |
| SVD (rank=3) | 9.1% | 0.972 | 52.1% |
| Benchmark Mean (baseline) | 12.9% | 0.928 | 43.4% |

**Minimum eval set**: 5 benchmarks (HLE, AIME 2025, LiveCodeBench, SWE-bench Verified, SimpleQA) predict the other 44 with ~7.8% MedAPE on proper holdout (LOO cross-validation within the 5-benchmark ridge model).

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
│   └── all_methods.py               # 19 prediction methods (BenchReg, SVD, NMF, etc.)
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
    ├── results_table.csv            # 19 methods x 10 metrics comparison
    ├── latent_factors.csv           # SVD factor loadings for 49 benchmarks
    ├── best_predictions.csv         # 2,684 predicted missing cells (all unobserved, clamped)
    ├── five_bench_predictions.csv   # 665 LOO predictions from 5-benchmark set
    ├── phase_transition.csv         # Accuracy vs number of known scores
    └── report.md                    # Full analysis report
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the evaluation harness (loads data, prints matrix stats)
python evaluation_harness.py

# Run all 19 methods and produce results_table.csv
python methods/all_methods.py

# Run diagnostics (intrinsic dimensionality, redundancy, minimum eval set, etc.)
python diagnostics/all_diagnostics.py

# Walk through a single prediction step-by-step
python analysis/walkthrough_single_prediction.py
```

## Prediction Methods

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
- BenchReg+KNN blend (alpha=0.6)
- Log-space variants

**Ensemble:**
- Average of top-3 methods

## Holdout Strategies

1. **Random cells** (20%) - standard matrix completion
2. **Per-model** (20% of each model's scores) - fair per-model evaluation
3. **Cold-start** (3 known scores per new model) - realistic deployment scenario
4. **Leave-one-benchmark** - can we predict an entire benchmark?
5. **Stratified difficulty** - balanced across easy/hard benchmarks
6. **Leave-one-provider** - generalization across model families

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
