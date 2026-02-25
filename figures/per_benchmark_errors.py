#!/usr/bin/env python3
"""Compute per-benchmark prediction error for LogitSVD Blend."""
import numpy as np
import sys, os, warnings

warnings.filterwarnings('ignore')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'methods'))

from evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, holdout_per_model,
)
from all_methods import predict_logit_svd_blend

# Run 3-fold per-model holdout
folds = holdout_per_model(k_frac=0.5, min_scores=8, n_folds=3, seed=42)

per_bench_abs = {}  # benchmark index -> list of absolute errors
per_bench_ape = {}  # benchmark index -> list of APE

for M_train, test_set in folds:
    M_pred = predict_logit_svd_blend(M_train)
    for i, j in test_set:
        actual = M_FULL[i, j]
        pred = M_pred[i, j]
        if np.isfinite(pred) and np.isfinite(actual):
            abs_err = abs(pred - actual)
            per_bench_abs.setdefault(j, []).append(abs_err)
            if abs(actual) > 1e-6:
                ape = abs_err / abs(actual) * 100
                per_bench_ape.setdefault(j, []).append(ape)

# Print sorted by median absolute error
print(f"{'='*80}")
print(f"  PER-BENCHMARK PREDICTION ERROR (LogitSVD Blend, 3-fold per-model holdout)")
print(f"{'='*80}")
print(f"  {'Benchmark':<35s} {'MedAbsErr':>10s} {'MedAPE':>8s} {'N':>5s} {'Mean±Std':>15s}")
print(f"  {'-'*75}")

results = []
for j in range(N_BENCH):
    if j in per_bench_abs and len(per_bench_abs[j]) >= 3:
        errs = per_bench_abs[j]
        apes = per_bench_ape.get(j, [])
        results.append((
            j,
            np.median(errs),
            np.median(apes) if apes else np.nan,
            len(errs),
            np.mean(errs),
            np.std(errs),
        ))

# Sort by median absolute error descending
results.sort(key=lambda x: -x[1])

for j, med_abs, med_ape, n, mean_abs, std_abs in results:
    bname = BENCH_NAMES.get(BENCH_IDS[j], BENCH_IDS[j])
    ape_str = f"{med_ape:.1f}%" if not np.isnan(med_ape) else "N/A"
    print(f"  {bname:<35s} {med_abs:>10.1f} {ape_str:>8s} {n:>5d} {mean_abs:>7.1f}±{std_abs:.1f}")

# Also print the easiest
print(f"\n  EASIEST (lowest median absolute error):")
results.sort(key=lambda x: x[1])
for j, med_abs, med_ape, n, mean_abs, std_abs in results[:10]:
    bname = BENCH_NAMES.get(BENCH_IDS[j], BENCH_IDS[j])
    ape_str = f"{med_ape:.1f}%" if not np.isnan(med_ape) else "N/A"
    print(f"  {bname:<35s} {med_abs:>10.1f} {ape_str:>8s} {n:>5d}")
